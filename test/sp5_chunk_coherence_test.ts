// Verifies the SP5 chunking fix directly: runs the real convertSplatToSp5Client
// pipeline on an actual PLY and reports each chunk's bounding box shape. Before
// the fix (global 1D lexicographic sort + contiguous count-based slicing),
// chunks for a real, non-uniform-density capture came out as near-2D slabs
// (one axis often under 1 world unit thick) even though 65536 points were in
// each chunk -- see manifest inspection in conversation history for
// splat_v2-sp5.zip. After the fix (k-d tree spatial partition), every chunk
// should be a roughly cube-shaped, spatially local region.
//
// Usage: npx tsx test/sp5_chunk_coherence_test.ts [path-to.ply]

import fs from 'fs';
import path from 'path';
import init_wasm, { decode_to_gsplatarray } from '../rust/spark-rs/pkg/spark_rs.js';
import { convertSplatToSp5Client } from '../src/converter.js';
import * as fflate from 'fflate';

async function main() {
  const plyPath = process.argv[2] ?? 'C:\\Users\\avboi\\Downloads\\splat_v2.ply';

  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  const plyBytes = new Uint8Array(fs.readFileSync(plyPath));
  const decoder = decode_to_gsplatarray('ply', null);
  const PUSH = 1024 * 1024;
  for (let i = 0; i < plyBytes.length; i += PUSH) decoder.push(plyBytes.subarray(i, Math.min(i + PUSH, plyBytes.length)));
  const decoded = decoder.finish();
  const numSplats = decoded.len();
  console.log(`Decoded ${numSplats} splats from ${plyPath}\n`);

  const attrs = decoded.extract_attributes();
  const xyz = new Float32Array(attrs.xyz);
  const opacity = new Float32Array(attrs.opacity);
  const rgb = new Float32Array(attrs.rgb);
  const scales = new Float32Array(attrs.scales);
  const quaternions = new Float32Array(attrs.quaternions);
  const maxSh = decoded.maxShDegree;
  const sh1 = maxSh > 0 ? new Float32Array(attrs.sh1) : undefined;

  const zipBytes = await convertSplatToSp5Client({
    numSplats, xyz, opacity, rgb, scales, quaternions, sh1, maxSh,
    onProgress: (phase, pct) => console.log(`  [convert] ${phase} (${pct}%)`),
  });

  const unzipped = fflate.unzipSync(zipBytes);
  const manifest = JSON.parse(new TextDecoder().decode(unzipped['manifest.json']));

  console.log(`\n${manifest.chunks.length} chunks, ${manifest.count} total splats\n`);

  let thinChunks = 0;
  let maxAspectRatio = 0;
  let totalSplats = 0;
  for (let i = 0; i < manifest.chunks.length; i++) {
    const c = manifest.chunks[i];
    const [minX, minY, minZ, maxX, maxY, maxZ] = c.aabb;
    const size = [maxX - minX, maxY - minY, maxZ - minZ];
    totalSplats += c.count;
    const nonDegenerate = size.filter((s: number) => s > 1e-6);
    const smallest = Math.min(...nonDegenerate.length ? nonDegenerate : [0]);
    const largest = Math.max(...size);
    const aspectRatio = smallest > 1e-6 ? largest / smallest : Infinity;
    if (aspectRatio > maxAspectRatio) maxAspectRatio = aspectRatio;
    // A "thin slab" chunk: one axis under 1 world unit while another spans
    // hundreds+ units -- exactly the paper-thin-Z-slab signature measured on
    // the broken pre-fix output.
    const isThin = size.some((s) => s < 1.0) && largest > 100;
    if (isThin) thinChunks++;
    console.log(
      `  chunk ${i}: count=${c.count} size=[${size.map((s: number) => s.toFixed(2)).join(', ')}] ` +
        `aspectRatio=${aspectRatio.toFixed(1)}${isThin ? '  <-- THIN SLAB' : ''}`,
    );
  }

  console.log(`\nTotal splats across chunks: ${totalSplats} (expected ${numSplats})`);
  console.log(`Thin-slab chunks (one axis <1 unit, another >100 units): ${thinChunks}/${manifest.chunks.length}`);
  console.log(`Max aspect ratio across all chunks: ${maxAspectRatio.toFixed(1)}`);

  console.log('\n=== Verdict ===');
  if (totalSplats !== numSplats) {
    console.log(`FAIL: splat count mismatch -- data loss in chunking (${totalSplats} vs ${numSplats}).`);
    process.exit(1);
  }
  if (thinChunks > 0) {
    console.log(`FAIL: ${thinChunks} chunk(s) are degenerate thin slabs -- spatial partitioning is not working.`);
    process.exit(1);
  }
  console.log('PASS: all chunks are spatially coherent (no thin-slab chunks), and total splat count matches.');
}

main().catch((e) => {
  console.error('CHUNK COHERENCE TEST CRASHED:', e);
  process.exit(1);
});
