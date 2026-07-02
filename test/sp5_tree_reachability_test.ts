// Automated test to verify reachability of the hierarchical LOD tree across chunks.
//
// Usage: npx tsx test/sp5_tree_reachability_test.ts [path-to-optional.ply]

import fs from 'fs';
import path from 'path';
import init_wasm, { decode_to_gsplatarray } from '../rust/spark-rs/pkg/spark_rs.js';
import { convertSplatToSp5Client } from '../src/converter.js';
import * as fflate from 'fflate';

async function main() {
  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  let plyPath = process.argv[2] ?? 'C:\\Users\\avboi\\Downloads\\splat_v2.ply';
  let numSplats: number;
  let xyz: Float32Array;
  let opacity: Float32Array;
  let rgb: Float32Array;
  let scales: Float32Array;
  let quaternions: Float32Array;
  let maxSh = 0;
  let sh1: Float32Array | undefined;

  if (fs.existsSync(plyPath)) {
    console.log(`Loading PLY from: ${plyPath}`);
    const plyBytes = new Uint8Array(fs.readFileSync(plyPath));
    const decoder = decode_to_gsplatarray('ply', null);
    const PUSH = 1024 * 1024;
    for (let i = 0; i < plyBytes.length; i += PUSH) {
      decoder.push(plyBytes.subarray(i, Math.min(i + PUSH, plyBytes.length)));
    }
    const decoded = decoder.finish();
    numSplats = decoded.len();
    console.log(`Decoded ${numSplats} splats from PLY.`);

    const attrs = decoded.extract_attributes();
    xyz = new Float32Array(attrs.xyz);
    opacity = new Float32Array(attrs.opacity);
    rgb = new Float32Array(attrs.rgb);
    scales = new Float32Array(attrs.scales);
    quaternions = new Float32Array(attrs.quaternions);
    maxSh = decoded.maxShDegree;
    sh1 = maxSh > 0 ? new Float32Array(attrs.sh1) : undefined;
  } else {
    console.log(`PLY file not found. Generating 80,000 synthetic points to trigger multi-chunking...`);
    numSplats = 80000;
    xyz = new Float32Array(numSplats * 3);
    opacity = new Float32Array(numSplats).fill(1.0);
    rgb = new Float32Array(numSplats * 3).fill(0.5);
    scales = new Float32Array(numSplats * 3).fill(-3.0); // log scale for standard splat format (-3.0 approx 0.05 linear)
    quaternions = new Float32Array(numSplats * 4);

    for (let i = 0; i < numSplats; i++) {
      const theta = Math.acos(-1 + (2 * i) / numSplats);
      const phi = Math.sqrt(numSplats * Math.PI) * theta;
      xyz[i * 3 + 0] = Math.sin(theta) * Math.cos(phi) * 10.0;
      xyz[i * 3 + 1] = Math.sin(theta) * Math.sin(phi) * 10.0;
      xyz[i * 3 + 2] = Math.cos(theta) * 10.0;
      
      quaternions[i * 4 + 0] = 1.0;
      quaternions[i * 4 + 1] = 0.0;
      quaternions[i * 4 + 2] = 0.0;
      quaternions[i * 4 + 3] = 0.0;
    }
  }

  console.log('Converting to SP5...');
  const zipBytes = await convertSplatToSp5Client({
    numSplats, xyz, opacity, rgb, scales, quaternions, sh1, maxSh,
    onProgress: (phase, pct) => console.log(`  [convert] ${phase} (${pct}%)`),
  });

  const unzipped = fflate.unzipSync(zipBytes);
  const manifest = JSON.parse(new TextDecoder().decode(unzipped['manifest.json']));
  console.log(`\nManifest loaded. ${manifest.chunks.length} chunks, ${manifest.count} total output splats.`);

  if (manifest.chunks.length < 2) {
    console.error(`FAIL: Scene compiled into only ${manifest.chunks.length} chunks. Multi-chunking did not trigger.`);
    process.exit(1);
  }

  // Load all lodTree buffers
  const lodTrees: Uint32Array[] = [];
  for (let i = 0; i < manifest.chunks.length; i++) {
    const chunkName = manifest.chunks[i].file;
    const chunkBytes = unzipped[chunkName];
    const dataView = new DataView(chunkBytes.buffer, chunkBytes.byteOffset, chunkBytes.byteLength);
    const headerSize = dataView.getUint32(4, true);
    const jsonBytes = chunkBytes.subarray(8, 8 + headerSize);
    const jsonStr = new TextDecoder().decode(jsonBytes);
    const chunkMeta = JSON.parse(jsonStr);

    const lodTreeMeta = chunkMeta.lod_tree;
    if (!lodTreeMeta) {
      throw new Error(`Chunk ${i} has no lod_tree metadata`);
    }

    const payloadOffset = 8 + headerSize;
    const startByte = payloadOffset + lodTreeMeta.offset;
    const endByte = startByte + lodTreeMeta.length;
    const treeBytes = new Uint8Array(chunkBytes.subarray(startByte, endByte));
    const treeArray = new Uint32Array(treeBytes.buffer);
    lodTrees.push(treeArray);
  }

  // Perform BFS starting from (chunk 0, offset 0) using ONLY real child pointers.
  const visited = new Set<string>(); // "chunk_offset"
  const queue: [number, number][] = [[0, 0]];
  visited.add("0_0");

  let head = 0;
  let countNodes = 0;

  console.log('\nRunning BFS from global root (chunk 0, offset 0) following real parent-child links...');

  while (head < queue.length) {
    const [c, offset] = queue[head++];
    countNodes++;

    const tree = lodTrees[c];
    const nodeIdx = offset * 4;
    const childCount = tree[nodeIdx + 2] & 0xffff;
    const childStart = tree[nodeIdx + 3];

    if (childCount > 0) {
      const targetChunk = childStart >>> 16;
      const targetOffset = childStart & 0xffff;
      
      for (let i = 0; i < childCount; i++) {
        const absoluteIndex = (targetChunk << 16) + targetOffset + i;
        const nextChunk = absoluteIndex >>> 16;
        const nextOffset = absoluteIndex & 0xffff;
        const key = `${nextChunk}_${nextOffset}`;
        if (!visited.has(key)) {
          visited.add(key);
          queue.push([nextChunk, nextOffset]);
        }
      }
    }
  }

  console.log(`\n=== Reachability Analysis Results ===`);
  console.log(`Total reachable nodes in tree: ${countNodes}`);
  
  let totalNodesInChunks = 0;
  const chunkReachableNodes: number[] = new Array(manifest.chunks.length).fill(0);
  const reachableChunks = new Set<number>();

  for (const key of visited) {
    const [c, offset] = key.split('_').map(Number);
    chunkReachableNodes[c]++;
    reachableChunks.add(c);
  }

  for (let i = 0; i < manifest.chunks.length; i++) {
    const totalInChunk = lodTrees[i].length / 4;
    totalNodesInChunks += totalInChunk;
    const percent = (chunkReachableNodes[i] / totalInChunk) * 100;
    console.log(`  Chunk ${i}: reachable nodes = ${chunkReachableNodes[i]} / ${totalInChunk} (${percent.toFixed(2)}%)`);
  }

  console.log(`\nReachable chunks count: ${reachableChunks.size} / ${manifest.chunks.length}`);
  console.log(`Total nodes in all chunks combined: ${totalNodesInChunks}`);
  console.log(`Reachable nodes ratio: ${(countNodes / totalNodesInChunks * 100).toFixed(3)}%`);

  console.log('\n=== Verification assertions ===');
  if (reachableChunks.size !== manifest.chunks.length) {
    console.error(`FAIL: only ${reachableChunks.size} out of ${manifest.chunks.length} chunks are reachable!`);
    process.exit(1);
  }

  if (countNodes !== totalNodesInChunks) {
    console.error(`FAIL: only ${countNodes} out of ${totalNodesInChunks} total nodes are reachable!`);
    process.exit(1);
  }

  console.log(`\nPASS: 100% of all chunks and nodes are reachable via real parent-child links!`);
}

main().catch(e => {
  console.error('TEST ERROR:', e);
  process.exit(1);
});
