// Focused check for the SP5 "large white ball" bug: verifies that scale values
// surviving the full converter.ts -> reconstruct_sp5_chunk round trip come back
// at the same order of magnitude as the source PLY's scales, instead of being
// double-exponentiated by a missing .ln() at the ln_scales assignment.
//
// Usage: npx tsx test/sp5_scale_diagnostic.ts [path-to.ply] [splatCap]

import fs from 'fs';
import path from 'path';
import init_wasm, { decode_to_gsplatarray, reconstruct_sp5_chunk } from '../rust/spark-rs/pkg/spark_rs.js';
import { convertSplatToSp5Client } from '../src/converter.js';
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
function decodeHuffman(bytes: Uint8Array, htable: Record<string, [number, number]>, count: number): Uint16Array {
  const table = new Map<string, number>();
  for (const [symbol, [len, bits]] of Object.entries(htable)) table.set(`${bits},${len}`, Number.parseInt(symbol));
  const out = new Uint16Array(count);
  let outIdx = 0, currentBits = 0, currentLen = 0, byteIdx = 0, bitIdx = 7;
  while (outIdx < count && byteIdx < bytes.length) {
    const bit = (bytes[byteIdx] >> bitIdx) & 1;
    bitIdx--;
    if (bitIdx < 0) { bitIdx = 7; byteIdx++; }
    currentBits = (currentBits << 1) | bit;
    currentLen++;
    const key = `${currentBits},${currentLen}`;
    if (table.has(key)) { out[outIdx++] = table.get(key)!; currentBits = 0; currentLen = 0; }
  }
  return out;
}

function stats(name: string, arr: Float32Array | number[]) {
  let min = Infinity, max = -Infinity, sum = 0, n = 0;
  for (const v of arr) {
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    n++;
  }
  const mean = sum / Math.max(1, n);
  console.log(`  ${name}: n=${n} min=${min.toFixed(4)} max=${max.toFixed(4)} mean=${mean.toFixed(4)}`);
  return { min, max, mean, n };
}

async function main() {
  const plyPath = process.argv[2] ?? 'E:\\ste_cecile\\experiments\\full_500k_lift\\work\\eval\\splat_30000.ply';
  const splatCap = Number.parseInt(process.argv[3] ?? '100000', 10);

  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  const plyBytes = new Uint8Array(fs.readFileSync(plyPath));
  const decoder = decode_to_gsplatarray('ply', null);
  const PUSH = 1024 * 1024;
  for (let i = 0; i < plyBytes.length; i += PUSH) decoder.push(plyBytes.subarray(i, Math.min(i + PUSH, plyBytes.length)));
  const decoded = decoder.finish();
  const fullNum = decoded.len();
  const numSplats = Math.min(fullNum, splatCap);
  console.log(`Decoded ${fullNum} splats from source; using first ${numSplats} for this diagnostic.\n`);

  const subset = (decoded as any).clone_subset(0, numSplats);
  const attrs = subset.extract_attributes();
  const xyz = new Float32Array(attrs.xyz);
  const opacity = new Float32Array(attrs.opacity);
  const rgb = new Float32Array(attrs.rgb);
  const scales = new Float32Array(attrs.scales); // linear scale, per extract_attributes()/Gsplat::scales()
  const quaternions = new Float32Array(attrs.quaternions);
  const maxSh = subset.maxShDegree;
  const sh1 = maxSh > 0 ? new Float32Array(attrs.sh1) : undefined;

  console.log('Source (pre-conversion) linear scale distribution:');
  stats('scale.x', scales.filter((_, i) => i % 3 === 0));
  stats('scale.y', scales.filter((_, i) => i % 3 === 1));
  stats('scale.z', scales.filter((_, i) => i % 3 === 2));
  console.log();

  const zipBytes = await convertSplatToSp5Client({
    numSplats, xyz, opacity, rgb, scales, quaternions, sh1, maxSh,
    onProgress: (phase, pct) => console.log(`  [convert] ${phase} (${pct}%)`),
  });

  const unzipped = fflate.unzipSync(zipBytes);
  const manifest = JSON.parse(new TextDecoder().decode(unzipped['manifest.json']));

  console.log(`\nmanifest: ${manifest.chunks.length} chunks, ${manifest.count} total splats (source was ${numSplats})`);
  const allDecodedScales: number[] = [];
  let leafCount = 0, nonLeafCount = 0;
  for (const chunkInfo of manifest.chunks) {
    const chunkBytes = unzipped[chunkInfo.file];
    const view = new DataView(chunkBytes.buffer, chunkBytes.byteOffset);
    const jsonSize = view.getUint32(4, true);
    const chunkMeta = JSON.parse(new TextDecoder().decode(chunkBytes.subarray(8, 8 + jsonSize)));
    const binaryPayload = chunkBytes.subarray(8 + jsonSize);
    const getBin = (m: { offset: number; length: number }) => binaryPayload.subarray(m.offset, m.offset + m.length);

    const count = chunkMeta.count;
    const xyzRawFloat = float16ArrayToFloat32Array(getBin(chunkMeta.xyz_uncompressed));
    // See converter.ts's chunkXyz comment / worker.ts's decodeSp5Chunk: positions
    // are chunk-local-normalized now, undo it before use (unused below beyond
    // feeding reconstruct_sp5_chunk, but keep it correct for consistency).
    {
      const chunkCenter: [number, number, number] = chunkMeta.chunk_center ?? [0, 0, 0];
      const chunkScale: number = chunkMeta.chunk_scale ?? 1;
      for (let i = 0; i < count; i++) {
        xyzRawFloat[i * 3 + 0] = xyzRawFloat[i * 3 + 0] * chunkScale + chunkCenter[0];
        xyzRawFloat[i * 3 + 1] = xyzRawFloat[i * 3 + 1] * chunkScale + chunkCenter[1];
        xyzRawFloat[i * 3 + 2] = xyzRawFloat[i * 3 + 2] * chunkScale + chunkCenter[2];
      }
    }

    // tiny_lod's merged/coarse parent splats are legitimately much larger than
    // any original leaf splat -- excluding them (child_count === 0 = leaf) keeps
    // this diagnostic comparing like with like against the source distribution.
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
      const d = decodeHuffman(getBin(hmeta), hmeta.huffman_table, count);
      for (let i = 0; i < count; i++) scaleIndices.push(d[i]);
    }
    const rotationIndices: number[] = [];
    for (const hmeta of chunkMeta.rotation_index_huffman) {
      const d = decodeHuffman(getBin(hmeta), hmeta.huffman_table, count);
      for (let i = 0; i < count; i++) rotationIndices.push(d[i]);
    }
    const appIndices: number[] = [];
    for (const hmeta of chunkMeta.app_index_huffman) {
      const d = decodeHuffman(getBin(hmeta), hmeta.huffman_table, count);
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
    const rScales: Float32Array = new Float32Array(rAttrs.scales); // linear (post-exp), same convention as source
    for (let i = 0; i < count; i++) {
      if (!isLeaf(i)) { nonLeafCount++; continue; }
      leafCount++;
      allDecodedScales.push(rScales[i * 3 + 0], rScales[i * 3 + 1], rScales[i * 3 + 2]);
    }
    reconstructed.free();
  }

  console.log(`\nleaf splats: ${leafCount}, non-leaf (merged) splats: ${nonLeafCount}`);
  console.log('\nDecoded (post round-trip) linear scale distribution:');
  const decodedStats = stats('scale (all axes)', allDecodedScales);

  const sourceAll: number[] = [];
  for (const v of scales) sourceAll.push(v);
  const sourceStats = stats('source scale (all axes, for comparison)', sourceAll);

  console.log('\n=== Verdict ===');
  const ratio = decodedStats.mean / Math.max(1e-9, sourceStats.mean);
  console.log(`decoded/source mean scale ratio: ${ratio.toFixed(3)}x`);
  if (decodedStats.max > sourceStats.max * 50 || ratio > 20 || ratio < 0.05) {
    console.log('FAIL: decoded scale is wildly different from source -- scale corruption present (e.g. missing/extra .ln()/.exp()).');
    process.exit(1);
  } else {
    console.log('PASS: decoded scale is the same order of magnitude as source.');
  }
}

main().catch((e) => {
  console.error('DIAGNOSTIC CRASHED:', e);
  process.exit(1);
});
