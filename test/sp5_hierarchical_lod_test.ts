// Automated test for Phase 1: Real hierarchical LOD per chunk.
// Verifies that a GsplatArray builds a real hierarchy with sizes shrinking with depth,
// and that WASM traversal outputs different resolutions at different camera distances.
//
// Usage: npx tsx test/sp5_hierarchical_lod_test.ts

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

  console.log('WASM Initialized successfully.');

  // 1. Generate 100 points distributed on a sphere
  const originalCount = 100;
  const xyz = new Float32Array(originalCount * 3);
  const opacity = new Float32Array(originalCount).fill(1.0);
  const rgb = new Float32Array(originalCount * 3).fill(0.5);
  const scales = new Float32Array(originalCount * 3).fill(0.1); // linear scale value
  const quaternions = new Float32Array(originalCount * 4);
  for (let i = 0; i < originalCount; i++) {
    const theta = Math.acos(-1 + (2 * i) / originalCount);
    const phi = Math.sqrt(originalCount * Math.PI) * theta;
    const x = Math.sin(theta) * Math.cos(phi) * 5.0;
    const y = Math.sin(theta) * Math.sin(phi) * 5.0;
    const z = Math.cos(theta) * 5.0;
    xyz[i * 3 + 0] = x;
    xyz[i * 3 + 1] = y;
    xyz[i * 3 + 2] = z;
    quaternions[i * 4 + 3] = 1.0; // identity
  }

  // 2. Build GsplatArray and compute LOD tree
  const gsplat = GsplatArray.from_attributes(xyz, opacity, rgb, scales, quaternions, undefined);
  gsplat.tiny_lod(1.5, false);

  const newCount = gsplat.len();
  console.log(`LOD Tree built: original count = ${originalCount}, new expanded count = ${newCount}`);
  if (newCount <= originalCount) {
    throw new Error('Assertion failed: newCount should be strictly greater than originalCount due to coarse parent nodes.');
  }

  const attrs = gsplat.extract_attributes();
  const lodTree = gsplat.extract_lod_tree();
  gsplat.free();

  // 3. Verify size shrinking properties of the extracted hierarchy
  // Starting at root (node 0)
  const sizes = new Float32Array(newCount);
  const parentOf = new Int32Array(newCount).fill(-1);
  for (let i = 0; i < newCount; i++) {
    sizes[i] = halfToFloat((lodTree[i * 4 + 1] >>> 16) & 0xffff);
  }

  const queue = [0];
  let visitedCount = 0;
  while (queue.length > 0) {
    const parentIdx = queue.shift()!;
    visitedCount++;
    const childCount = lodTree[parentIdx * 4 + 2] & 0xffff;
    const childStart = lodTree[parentIdx * 4 + 3];
    if (childCount > 0) {
      for (let c = 0; c < childCount; c++) {
        const childIdx = childStart + c;
        if (childIdx >= newCount) {
          throw new Error(`Child index ${childIdx} exceeds node count ${newCount}`);
        }
        parentOf[childIdx] = parentIdx;
        
        // Assert: child size < parent size
        if (sizes[childIdx] >= sizes[parentIdx]) {
          throw new Error(
            `Assertion failed: parent size (${sizes[parentIdx]}) is not strictly larger than child index ${childIdx} size (${sizes[childIdx]})`
          );
        }
        queue.push(childIdx);
      }
    }
  }

  console.log(`Size verification passed! Traversed ${visitedCount} reachable nodes.`);

  // 4. Test encode-time shift
  const chunkIdx = 3;
  const shiftedTree = new Uint32Array(lodTree);
  for (let i = 0; i < newCount; i++) {
    const childCount = shiftedTree[i * 4 + 2] & 0xffff;
    if (childCount > 0) {
      shiftedTree[i * 4 + 3] = (chunkIdx << 16) | shiftedTree[i * 4 + 3];
    }
  }
  // Verify that child starts are correctly shifted
  for (let i = 0; i < newCount; i++) {
    const childCount = shiftedTree[i * 4 + 2] & 0xffff;
    if (childCount > 0) {
      const origStart = lodTree[i * 4 + 3];
      const shiftedStart = shiftedTree[i * 4 + 3];
      if (shiftedStart !== ((chunkIdx << 16) | origStart)) {
        throw new Error(`Shift failed for node ${i}: expected ${(chunkIdx << 16) | origStart}, got ${shiftedStart}`);
      }
    }
  }
  console.log('Encode-time shift verification passed.');

  // 5. Test WASM Traversal distances
  // Let's load the unshifted tree as chunk 0 (page 0)
  const { lodId } = init_lod_tree(0, new Uint32Array(0)) as { lodId: number };
  
  // Patch root child start (simulating pager behavior for chunk 0)
  lodTree[3] = (0 * PAGE_SPLATS + lodTree[3]) >>> 0;

  update_lod_trees(
    new Uint32Array([lodId]),
    new Uint32Array([0 * PAGE_SPLATS]),
    new Uint32Array([0 * PAGE_SPLATS]),
    new Uint32Array([newCount]),
    [lodTree],
  );

  const runTraverse = (cameraZ: number) => {
    const camera = identityCameraMatrix(0, 0, cameraZ);
    const result = traverse_lod_trees(
      10000,
      0.005, // pixel_scale_limit
      undefined,
      new Uint32Array([lodId]),
      new Uint32Array([0]),
      camera,
      new Float32Array([1.0]),
      new Float32Array([1.0]),
      new Float32Array([1.0]),
      new Float32Array([0.0]),
      new Float32Array([0.0]),
    ) as { instanceIndices: { numSplats: number }[] };
    return result.instanceIndices[0].numSplats;
  };

  // Traversal at different distances:
  const splatsAt5000 = runTraverse(5000); // very far
  const splatsAt150 = runTraverse(150);   // intermediate
  const splatsAt5 = runTraverse(5);       // close

  console.log(`Traversal Results:`);
  console.log(`  At Z=5000: ${splatsAt5000} splats`);
  console.log(`  At Z=150:  ${splatsAt150} splats`);
  console.log(`  At Z=5:    ${splatsAt5} splats`);

  if (splatsAt5000 !== 1) {
    throw new Error(`Expected exactly 1 splat (root only) at Z=5000, got ${splatsAt5000}`);
  }
  if (splatsAt5 <= splatsAt150) {
    throw new Error(`Expected more splats close up (Z=5) than intermediate (Z=150). Got ${splatsAt5} <= ${splatsAt150}`);
  }
  if (splatsAt150 <= splatsAt5000) {
    throw new Error(`Expected more splats at intermediate (Z=150) than far (Z=5000). Got ${splatsAt150} <= ${splatsAt5000}`);
  }

  console.log('Single-chunk hierarchical LOD traversal test PASSED!');
}

main().catch(err => {
  console.error('Test FAILED:', err);
  process.exit(1);
});
