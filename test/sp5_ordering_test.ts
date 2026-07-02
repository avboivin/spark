// Ground-truth test for the SP5 encode/decode attribute-ordering bug flagged in
// review: reconstruct_sp5_chunk (rust/spark-rs/src/lib.rs) indexes POSITION via
// `points[sorted_indices[i]]` (the decoder's own re-sort of the already-quantized
// xyz) but indexes SCALE/ROTATION/APP codebook indices via `i` directly (the
// encoder's original write order). If the decoder's re-sort ever produces a
// different permutation than the encoder's pre-quantization sort -- e.g. because
// float16-quantizing positions can create ties/inversions that didn't exist in
// the full-precision data the encoder sorted -- then position and
// scale/rotation end up paired from DIFFERENT splats for every point after the
// first reordering. This would look exactly like "completely broken, looks
// nothing like the original": individually correct-looking points, but with
// wildly wrong sizes/orientations relative to their real neighbors.
//
// This test builds splats with a UNIQUE, exactly-traceable scale per point (via
// a monotonic ramp keyed to a splat's index BEFORE encoding), runs them through
// the real convertSplatToSp5Client + reconstruct_sp5_chunk pipeline, then for
// each decoded splat finds its original by nearest position match and checks
// whether the decoded scale actually belongs to that original point (within
// codebook quantization tolerance) or to some other point entirely.
//
// Usage: npx tsx test/sp5_ordering_test.ts

import fs from 'fs';
import path from 'path';
import init_wasm, { decode_to_gsplatarray, reconstruct_sp5_chunk, decode_huffman_fast } from '../rust/spark-rs/pkg/spark_rs.js';
import { convertSplatToSp5Client } from '../src/converter.js';
import { decodeHuffman, type HuffmanTable } from '../src/huffman_decode.js';
import * as fflate from 'fflate';

function halfToFloat(binary: number): number {
  const exponent = (binary & 0x7c00) >> 10;
  const fraction = binary & 0x03ff;
  if (exponent === 0) return (binary & 0x8000 ? -1 : 1) * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : binary & 0x8000 ? -Infinity : Infinity;
  return (binary & 0x8000 ? -1 : 1) * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
function float16ArrayToFloat32Array(bytes: Uint8Array): Float32Array {
  let a = bytes;
  if (bytes.byteOffset % 2 !== 0) { a = new Uint8Array(bytes.length); a.set(bytes); }
  const u16 = new Uint16Array(a.buffer, a.byteOffset, a.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
  return out;
}
const _decodeH = (bytes: Uint8Array, htable: HuffmanTable, count: number) =>
  decodeHuffman(bytes, htable, count, decode_huffman_fast as any);

async function main() {
  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  // Build a synthetic scene with well-separated, uniquely-identifiable positions
  // (so nearest-position matching after decode is unambiguous) and a scale that
  // monotonically ramps with the ORIGINAL point index, so a mismatch is obvious.
  const numSplats = 20000;
  const xyz = new Float32Array(numSplats * 3);
  const opacity = new Float32Array(numSplats);
  const rgb = new Float32Array(numSplats * 3);
  const scales = new Float32Array(numSplats * 3);
  const quaternions = new Float32Array(numSplats * 4);

  const originalScaleForIndex = new Float32Array(numSplats);
  for (let i = 0; i < numSplats; i++) {
    // Spread points widely and irregularly across a large volume (mimicking a
    // real georeferenced capture) so many points land close together in
    // sorted (z,y,x) order -- exactly the regime where quantization-induced
    // ties/inversions in the decoder's re-sort are most likely to occur.
    const x = ((i * 92821) % 100000) / 100 - 500; // spread across ~[-500, 500]
    const y = ((i * 63949) % 100000) / 100 - 500;
    const z = ((i * 17711) % 100000) / 100 - 500;
    xyz[i * 3 + 0] = x;
    xyz[i * 3 + 1] = y;
    xyz[i * 3 + 2] = z;

    opacity[i] = 1.0;
    rgb[i * 3 + 0] = 0.5; rgb[i * 3 + 1] = 0.5; rgb[i * 3 + 2] = 0.5;

    // Scale monotonically increasing with index: point 0 gets the smallest
    // scale, point N-1 the largest. Quantized into 256 codebook levels, so we
    // check the decoded scale is within one codebook step of the ORIGINAL
    // point at that position, not just "some" plausible value.
    const s = 0.01 + (i / numSplats) * 10.0; // range [0.01, 10.01]
    scales[i * 3 + 0] = s;
    scales[i * 3 + 1] = s;
    scales[i * 3 + 2] = s;
    originalScaleForIndex[i] = s;

    quaternions[i * 4 + 0] = 0; quaternions[i * 4 + 1] = 0; quaternions[i * 4 + 2] = 0; quaternions[i * 4 + 3] = 1;
  }

  const zipBytes = await convertSplatToSp5Client({
    numSplats, xyz, opacity, rgb, scales, quaternions, maxSh: 0,
  });

  const unzipped = fflate.unzipSync(zipBytes);
  const manifest = JSON.parse(new TextDecoder().decode(unzipped['manifest.json']));

  // Build a lookup from ORIGINAL position -> original scale, so we can check
  // each decoded point's scale against the point it's actually closest to.
  const origPositions: [number, number, number][] = [];
  for (let i = 0; i < numSplats; i++) {
    origPositions.push([xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2]]);
  }

  let checked = 0;
  let mismatches = 0;
  const mismatchExamples: string[] = [];

  for (const chunkInfo of manifest.chunks) {
    const chunkBytes = unzipped[chunkInfo.file];
    const view = new DataView(chunkBytes.buffer, chunkBytes.byteOffset);
    const jsonSize = view.getUint32(4, true);
    const chunkMeta = JSON.parse(new TextDecoder().decode(chunkBytes.subarray(8, 8 + jsonSize)));
    const binaryPayload = chunkBytes.subarray(8 + jsonSize);
    const getBin = (m: { offset: number; length: number }) => binaryPayload.subarray(m.offset, m.offset + m.length);

    const count = chunkMeta.count;
    const xyzRawFloat = float16ArrayToFloat32Array(getBin(chunkMeta.xyz_uncompressed));
    // Positions are now chunk-local-normalized (see converter.ts's chunkXyz comment) --
    // undo it the same way worker.ts's decodeSp5Chunk does, or nearest-position
    // matching below compares two different coordinate systems and is meaningless.
    const chunkCenter: [number, number, number] = chunkMeta.chunk_center ?? [0, 0, 0];
    const chunkScale: number = chunkMeta.chunk_scale ?? 1;
    for (let i = 0; i < count; i++) {
      xyzRawFloat[i * 3 + 0] = xyzRawFloat[i * 3 + 0] * chunkScale + chunkCenter[0];
      xyzRawFloat[i * 3 + 1] = xyzRawFloat[i * 3 + 1] * chunkScale + chunkCenter[1];
      xyzRawFloat[i * 3 + 2] = xyzRawFloat[i * 3 + 2] * chunkScale + chunkCenter[2];
    }

    // The real LOD tree now mixes in merged/coarse parent splats (tiny_lod's
    // synthetic representatives) alongside original leaf splats. A merged node's
    // scale legitimately has no relationship to any single nearby original
    // point's scale, so only leaves (child_count === 0) are meaningful to check
    // here -- node index i in lod_tree corresponds 1:1 with attribute index i.
    let lodTreeRaw: Uint32Array | undefined;
    if (chunkMeta.lod_tree) {
      let lodTreeBytes = getBin(chunkMeta.lod_tree);
      if (lodTreeBytes.byteOffset % 4 !== 0) {
        const aligned = new Uint8Array(lodTreeBytes.length);
        aligned.set(lodTreeBytes);
        lodTreeBytes = aligned;
      }
      lodTreeRaw = new Uint32Array(lodTreeBytes.buffer, lodTreeBytes.byteOffset, lodTreeBytes.byteLength / 4);
    }
    const isLeaf = (i: number) => !lodTreeRaw || lodTreeRaw[i * 4 + 2] === 0;

    const scaleIndices: number[] = [];
    for (const hmeta of chunkMeta.scale_index_huffman) {
      const d = _decodeH(getBin(hmeta), hmeta.huffman_table, count);
      for (let i = 0; i < count; i++) scaleIndices.push(d[i]);
    }
    const rotationIndices: number[] = [];
    for (const hmeta of chunkMeta.rotation_index_huffman) {
      const d = _decodeH(getBin(hmeta), hmeta.huffman_table, count);
      for (let i = 0; i < count; i++) rotationIndices.push(d[i]);
    }
    const appIndices: number[] = [];
    for (const hmeta of chunkMeta.app_index_huffman) {
      const d = _decodeH(getBin(hmeta), hmeta.huffman_table, count);
      for (let i = 0; i < count; i++) appIndices.push(d[i]);
    }

    const scaleCbFlat = new Float32Array(chunkMeta.scale_codebook.length * 256);
    chunkMeta.scale_codebook.forEach((m: any, idx: number) => scaleCbFlat.set(float16ArrayToFloat32Array(getBin(m)), idx * 256));
    const rotationCbFlat = new Float32Array(chunkMeta.rotation_codebook.length * 512);
    chunkMeta.rotation_codebook.forEach((m: any, idx: number) => rotationCbFlat.set(float16ArrayToFloat32Array(getBin(m)), idx * 512));
    const appCbFlat = new Float32Array(chunkMeta.app_codebook.length * 512);
    chunkMeta.app_codebook.forEach((m: any, idx: number) => appCbFlat.set(float16ArrayToFloat32Array(getBin(m)), idx * 512));

    const mlpCont = float16ArrayToFloat32Array(getBin(chunkMeta.mlp_cont));
    const mlpDc = float16ArrayToFloat32Array(getBin(chunkMeta.mlp_dc));
    const mlpSh = float16ArrayToFloat32Array(getBin(chunkMeta.mlp_sh));
    const mlpOpacity = float16ArrayToFloat32Array(getBin(chunkMeta.mlp_opacity));

    const empty = new Float32Array(0);
    const reconstructed = reconstruct_sp5_chunk(
      xyzRawFloat, new Uint16Array(scaleIndices), new Uint16Array(rotationIndices), new Uint16Array(appIndices),
      scaleCbFlat, rotationCbFlat, appCbFlat, mlpCont, mlpDc, mlpSh, mlpOpacity,
      empty, empty, empty, empty, empty, empty, empty, empty,
    );

    const rAttrs = (reconstructed as any).extract_attributes();
    const rXyz = new Float32Array(rAttrs.xyz);
    const rScales = new Float32Array(rAttrs.scales);
    reconstructed.free();

    let leafN = 0, nonLeafN = 0;
    for (let i = 0; i < count; i++) { if (isLeaf(i)) leafN++; else nonLeafN++; }
    console.log(`  chunk ${chunkInfo.file}: count=${count}, leaves=${leafN}, non-leaf=${nonLeafN}`);

    for (let i = 0; i < count; i++) {
      if (!isLeaf(i)) continue; // merged/coarse nodes have no single "original" point to compare against
      const dx = rXyz[i * 3 + 0], dy = rXyz[i * 3 + 1], dz = rXyz[i * 3 + 2];
      const decodedScale = rScales[i * 3 + 0]; // x-channel; all 3 channels equal by construction

      // Find nearest original point by position (positions are float16-quantized
      // but well-separated by construction, so nearest-match is unambiguous).
      let bestIdx = -1;
      let bestDist = Infinity;
      for (let o = 0; o < numSplats; o++) {
        const [ox, oy, oz] = origPositions[o];
        const d = (dx - ox) ** 2 + (dy - oy) ** 2 + (dz - oz) ** 2;
        if (d < bestDist) { bestDist = d; bestIdx = o; }
        if (d < 1e-6) break; // exact/near-exact match found, stop early
      }
      if (bestIdx === -1) continue;

      const expectedScale = originalScaleForIndex[bestIdx];
      checked++;
      // The scale codebook is now built in LOG space over the combined leaf +
      // merged-LOD-node population (see converter.ts's scaleChannels comment),
      // so its resolution at any given leaf's magnitude is roughly proportional
      // to that magnitude, not a fixed linear step across [0.01, 10.01]. Use a
      // relative tolerance (with a small absolute floor for near-zero values);
      // a genuine mispairing bug produces errors of 100%+ (an unrelated splat's
      // scale), far beyond legitimate quantization noise.
      const tolerance = Math.max(0.05, expectedScale * 0.08);
      if (Math.abs(decodedScale - expectedScale) > tolerance) {
        mismatches++;
        if (mismatchExamples.length < 10) {
          const childCount = lodTreeRaw ? lodTreeRaw[i * 4 + 2] : -1;
          mismatchExamples.push(
            `  slot i=${i} childCount=${childCount} decoded point at (${dx.toFixed(2)},${dy.toFixed(2)},${dz.toFixed(2)}) matched original #${bestIdx} ` +
              `(expected scale ${expectedScale.toFixed(4)}, dist=${Math.sqrt(bestDist).toFixed(4)}) but decoded scale is ${decodedScale.toFixed(4)}`,
          );
        }
      }
    }
  }

  console.log(`Checked ${checked} decoded splats against their nearest-position original.`);
  console.log(`Scale mismatches (decoded scale doesn't belong to the matched point): ${mismatches} (${((100 * mismatches) / Math.max(1, checked)).toFixed(2)}%)`);
  if (mismatchExamples.length) {
    console.log('\nExamples:');
    for (const ex of mismatchExamples) console.log(ex);
  }

  console.log('\n=== Verdict ===');
  if (mismatches / Math.max(1, checked) > 0.01) {
    console.log('FAIL: attribute ordering is broken -- position and scale/rotation/app are being paired from different splats.');
    process.exit(1);
  } else {
    console.log('PASS: decoded attributes stay correctly paired with their own position.');
  }
}

main().catch((e) => {
  console.error('ORDERING TEST CRASHED:', e);
  process.exit(1);
});
