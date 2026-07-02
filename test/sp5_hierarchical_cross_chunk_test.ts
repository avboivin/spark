// Automated test for Phase 1: Real hierarchical LOD per chunk + cross-chunk stitching.
// Verifies that:
// 1. Two chunks are constructed, each with its own multi-level hierarchy.
// 2. Child starts are shifted by their respective chunk indexes.
// 3. Chunk 0 is stitched with sibling Chunk 1 using the Virtual Root algorithm.
// 4. Traversal walks down Chunk 0, jumps to Chunk 1, and walks down Chunk 1's hierarchy.
//
// Usage: npx tsx test/sp5_hierarchical_cross_chunk_test.ts

import fs from 'fs';
import path from 'path';
import init_wasm, {
  GsplatArray,
  init_lod_tree,
  update_lod_trees,
  traverse_lod_trees,
} from '../rust/spark-rs/pkg/spark_rs.js';

const PAGE_SPLATS = 65536;

function halfToFloat(val: number): number {
  const sign = (val >> 15) & 0x1;
  const exponent = (val >> 10) & 0x1f;
  const mantissa = val & 0x3ff;
  if (exponent === 0x1f) {
    return mantissa ? NaN : (sign ? -Infinity : Infinity);
  }
  if (exponent === 0) {
    if (mantissa === 0) return sign ? -0.0 : 0.0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024.0);
  }
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1.0 + mantissa / 1024.0);
}

function float32ToHalfBits(val: number): number {
  const f32 = new Float32Array([val]);
  const u32 = new Uint32Array(f32.buffer)[0];
  const sign = (u32 >> 31) & 0x1;
  const exp = (u32 >> 23) & 0xff;
  let mantissa = u32 & 0x7fffff;
  if (exp === 0xff) return (sign << 15) | 0x7c00 | (mantissa ? 1 : 0);
  const halfExp = exp - 127 + 15;
  if (halfExp >= 0x1f) return (sign << 15) | 0x7bff;
  if (halfExp <= 0) {
    if (halfExp < -10) return sign << 15;
    mantissa = (mantissa | 0x800000) >> (1 - halfExp);
    return (sign << 15) | (mantissa >> 13);
  }
  return (sign << 15) | (halfExp << 10) | (mantissa >> 13);
}

function stitchRealLodTreeWithSiblings(
  lodTree: Uint32Array,
  siblingChunks: { chunkIndex: number; center: [number, number, number]; size: number }[],
): Uint32Array {
  const originalNodesCount = lodTree.length / 4;
  const numSiblings = siblingChunks.length;
  const shiftAmount = 1 + numSiblings;
  const totalEntries = shiftAmount + originalNodesCount;
  
  const newLodTree = new Uint32Array(totalEntries * 4);
  
  const cx0 = halfToFloat(lodTree[0] & 0xffff);
  const cy0 = halfToFloat((lodTree[0] >>> 16) & 0xffff);
  const cz0 = halfToFloat(lodTree[1] & 0xffff);
  const size0 = halfToFloat((lodTree[1] >>> 16) & 0xffff);
  
  let minX = cx0 - size0, maxX = cx0 + size0;
  let minY = cy0 - size0, maxY = cy0 + size0;
  let minZ = cz0 - size0, maxZ = cz0 + size0;
  
  for (let s = 0; s < numSiblings; s++) {
    const sib = siblingChunks[s];
    const sx = sib.center[0], sy = sib.center[1], sz = sib.center[2];
    const sSize = sib.size;
    if (sx - sSize < minX) minX = sx - sSize;
    if (sx + sSize > maxX) maxX = sx + sSize;
    if (sy - sSize < minY) minY = sy - sSize;
    if (sy + sSize > maxY) maxY = sy + sSize;
    if (sz - sSize < minZ) minZ = sz - sSize;
    if (sz + sSize > maxZ) maxZ = sz + sSize;
  }
  
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
  const nominalSize = Math.max(diag / 2, 1e-4);
  
  const writeEntry = (
    dest: Uint32Array,
    idx: number,
    ex: number, ey: number, ez: number,
    size: number,
    childCount: number,
    childStart: number,
  ) => {
    const o = idx * 4;
    dest[o + 0] = (float32ToHalfBits(ex) & 0xffff) | ((float32ToHalfBits(ey) & 0xffff) << 16);
    dest[o + 1] = (float32ToHalfBits(ez) & 0xffff) | ((float32ToHalfBits(size) & 0xffff) << 16);
    dest[o + 2] = childCount & 0xffff;
    dest[o + 3] = childStart >>> 0;
  };
  
  writeEntry(newLodTree, 0, cx, cy, cz, nominalSize, 1 + numSiblings, 1);
  
  const origRootCount = lodTree[2] & 0xffff;
  const origRootStart = lodTree[3];
  let newRootStart = 0;
  if (origRootCount > 0) {
    newRootStart = (origRootStart >> 16) === 0 ? (origRootStart + shiftAmount) : origRootStart;
  }
  newLodTree[4] = lodTree[0];
  newLodTree[5] = lodTree[1];
  newLodTree[6] = origRootCount & 0xffff;
  newLodTree[7] = newRootStart >>> 0;
  
  for (let s = 0; s < numSiblings; s++) {
    const sib = siblingChunks[s];
    const absoluteChildStart = ((sib.chunkIndex << 16) | 0) >>> 0;
    writeEntry(
      newLodTree,
      2 + s,
      sib.center[0],
      sib.center[1],
      sib.center[2],
      sib.size,
      1,
      absoluteChildStart,
    );
  }
  
  for (let i = 1; i < originalNodesCount; i++) {
    const srcO = i * 4;
    const destO = (i + shiftAmount) * 4;
    
    newLodTree[destO + 0] = lodTree[srcO + 0];
    newLodTree[destO + 1] = lodTree[srcO + 1];
    
    const childCount = lodTree[srcO + 2] & 0xffff;
    const childStart = lodTree[srcO + 3];
    
    let newChildStart = 0;
    if (childCount > 0) {
      newChildStart = (childStart >> 16) === 0 ? (childStart + shiftAmount) : childStart;
    }
    
    newLodTree[destO + 2] = childCount & 0xffff;
    newLodTree[destO + 3] = newChildStart >>> 0;
  }
  
  return newLodTree;
}

function generateSphereChunk(cx: number, cy: number, cz: number, count: number): Uint32Array {
  const xyz = new Float32Array(count * 3);
  const opacity = new Float32Array(count).fill(1.0);
  const rgb = new Float32Array(count * 3).fill(0.5);
  const scales = new Float32Array(count * 3).fill(0.1);
  const quaternions = new Float32Array(count * 4);
  
  for (let i = 0; i < count; i++) {
    const theta = Math.acos(-1 + (2 * i) / count);
    const phi = Math.sqrt(count * Math.PI) * theta;
    const x = Math.sin(theta) * Math.cos(phi) * 5.0 + cx;
    const y = Math.sin(theta) * Math.sin(phi) * 5.0 + cy;
    const z = Math.cos(theta) * 5.0 + cz;
    xyz[i * 3 + 0] = x;
    xyz[i * 3 + 1] = y;
    xyz[i * 3 + 2] = z;
    quaternions[i * 4 + 3] = 1.0;
  }

  const gsplat = GsplatArray.from_attributes(xyz, opacity, rgb, scales, quaternions, undefined);
  gsplat.tiny_lod(1.5, false);
  const lodTree = gsplat.extract_lod_tree();
  gsplat.free();
  
  return lodTree;
}

function identityCameraMatrix(tx: number, ty: number, tz: number): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    tx, ty, tz, 1,
  ]);
}

async function main() {
  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  // 1. Generate two chunks (Chunk 0 at origin, Chunk 1 at Z = 100)
  const chunk0Tree = generateSphereChunk(0, 0, 0, 100);
  const chunk1Tree = generateSphereChunk(0, 0, 100, 100);
  
  const count0 = chunk0Tree.length / 4;
  const count1 = chunk1Tree.length / 4;

  // 2. Encode-time shift
  // Chunk 0's child starts are shifted by 0 (no-op).
  // Chunk 1's child starts are shifted by 1 << 16.
  for (let i = 0; i < count1; i++) {
    const childCount = chunk1Tree[i * 4 + 2] & 0xffff;
    if (childCount > 0) {
      chunk1Tree[i * 4 + 3] = (1 << 16) | chunk1Tree[i * 4 + 3];
    }
  }

  // Get Chunk 1's bounds from its root node
  const cx1 = halfToFloat(chunk1Tree[0] & 0xffff);
  const cy1 = halfToFloat((chunk1Tree[0] >>> 16) & 0xffff);
  const cz1 = halfToFloat(chunk1Tree[1] & 0xffff);
  const size1 = halfToFloat((chunk1Tree[1] >>> 16) & 0xffff);

  const siblingChunks = [
    { chunkIndex: 1, center: [cx1, cy1, cz1] as [number, number, number], size: size1 }
  ];

  // 3. Perform Virtual-Root-based sibling stitching for Chunk 0
  const stitched0Tree = stitchRealLodTreeWithSiblings(chunk0Tree, siblingChunks);

  // 4. Initialize shared LOD tree in WASM
  const { lodId } = init_lod_tree(0, new Uint32Array(0)) as { lodId: number };

  // Update WASM with stitched Chunk 0 on Page 0 and Chunk 1 on Page 1
  update_lod_trees(
    new Uint32Array([lodId, lodId]),
    new Uint32Array([0 * PAGE_SPLATS, 1 * PAGE_SPLATS]), // page_base
    new Uint32Array([0 * PAGE_SPLATS, 1 * PAGE_SPLATS]), // chunk_base
    new Uint32Array([stitched0Tree.length / 4, count1]),
    [stitched0Tree, chunk1Tree],
  );

  const runTraverse = (cameraX: number, cameraY: number, cameraZ: number) => {
    const camera = identityCameraMatrix(cameraX, cameraY, cameraZ);
    const result = traverse_lod_trees(
      10000,
      0.05, // pixel_scale_limit
      undefined,
      new Uint32Array([lodId]),
      new Uint32Array([0]),
      camera,
      new Float32Array([1.0]),
      new Float32Array([1.0]),
      new Float32Array([1.0]),
      new Float32Array([0.0]),
      new Float32Array([0.0]),
    ) as {
      chunks: [number, number][];
      instanceIndices: { numSplats: number }[];
      splatIndices: Uint32Array;
    };
    return result;
  };

  // Traversal at Z=20000 (very far): should return exactly 1 splat (the virtual root)
  const resultFar = runTraverse(0, 0, 20000);
  console.log(`Far Traversal splat count: ${resultFar.instanceIndices[0].numSplats}`);
  if (resultFar.instanceIndices[0].numSplats !== 1) {
    throw new Error(`Expected exactly 1 splat at Z=20000, got ${resultFar.instanceIndices[0].numSplats}`);
  }

  // Traversal at Z=50 (close to Chunk 0): should return some splats from page 0 and some from page 1!
  // At Z=50, the camera is at (0, 0, 50).
  // Distance to Chunk 0 (0, 0, 0) is 50.
  // Distance to Chunk 1 (0, 0, 100) is 50.
  // Both chunks are close, so we expect splats from both pages!
  const resultClose = runTraverse(0, 0, 50);
  const splatIndices = resultClose.instanceIndices[0].indices;
  const numSplats = resultClose.instanceIndices[0].numSplats;
  console.log(`Close Traversal total splat count: ${numSplats}`);

  let page0Count = 0;
  let page1Count = 0;
  for (let i = 0; i < numSplats; i++) {
    const pageIdx = splatIndices[i] >>> 16;
    if (pageIdx === 0) page0Count++;
    else if (pageIdx === 1) page1Count++;
  }
  console.log(`  Page 0 splats: ${page0Count}`);
  console.log(`  Page 1 splats: ${page1Count}`);

  if (page0Count === 0 || page1Count === 0) {
    throw new Error(`Expected splats from both Page 0 and Page 1, got page0=${page0Count}, page1=${page1Count}`);
  }

  // Now, test moving camera close to Chunk 1 (Z = 105) but far from Chunk 0
  // Since Z = 105, distance to Chunk 1 is 5 (very close, highly refined).
  // Distance to Chunk 0 is 105.
  // Let's check how many splats we get for each page.
  const resultNearChunk1 = runTraverse(0, 0, 105);
  const indicesNear1 = resultNearChunk1.instanceIndices[0].indices;
  const countNear1 = resultNearChunk1.instanceIndices[0].numSplats;
  
  let p0Near1 = 0;
  let p1Near1 = 0;
  for (let i = 0; i < countNear1; i++) {
    const pageIdx = indicesNear1[i] >>> 16;
    if (pageIdx === 0) p0Near1++;
    else if (pageIdx === 1) p1Near1++;
  }
  console.log(`Near Chunk 1 Traversal (Z=105):`);
  console.log(`  Page 0 splats: ${p0Near1}`);
  console.log(`  Page 1 splats: ${p1Near1}`);

  if (p1Near1 <= p0Near1) {
    throw new Error(`Expected more splats near Chunk 1 (page 1) than Chunk 0 (page 0), got page1=${p1Near1} <= page0=${p0Near1}`);
  }

  // Near Chunk 0 Traversal (Z=5):
  // Distance to Chunk 0 is 5 (very close, fully refined).
  // Distance to Chunk 1 is 95 (far, should collapse to root/representative).
  const resultNearChunk0 = runTraverse(0, 0, 5);
  const indicesNear0 = resultNearChunk0.instanceIndices[0].indices;
  const countNear0 = resultNearChunk0.instanceIndices[0].numSplats;

  let p0Near0 = 0;
  let p1Near0 = 0;
  for (let i = 0; i < countNear0; i++) {
    const pageIdx = indicesNear0[i] >>> 16;
    if (pageIdx === 0) p0Near0++;
    else if (pageIdx === 1) p1Near0++;
  }
  console.log(`Near Chunk 0 Traversal (Z=5):`);
  console.log(`  Page 0 splats: ${p0Near0}`);
  console.log(`  Page 1 splats: ${p1Near0}`);

  if (p0Near0 !== 100) {
    throw new Error(`Expected exactly 100 splats for Page 0 near Chunk 0, got ${p0Near0}. Off-by-one stitching bug detected!`);
  }

  console.log('Cross-chunk multi-level traversal test PASSED!');
}

main().catch(err => {
  console.error('Test FAILED:', err);
  process.exit(1);
});
