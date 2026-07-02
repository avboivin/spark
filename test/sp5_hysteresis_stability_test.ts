// Automated test for Track B: Traversal Hysteresis and Temporal Stability.
// Simulates a camera rotating in place over 100 frames and measures the index churn.
// Verifies that the frame-to-frame churn is extremely low due to the hysteresis dead-band.
//
// Usage: npx tsx test/sp5_hysteresis_stability_test.ts

import fs from 'fs';
import path from 'path';
import init_wasm, {
  GsplatArray,
  init_lod_tree,
  update_lod_trees,
  traverse_lod_trees,
} from '../rust/spark-rs/pkg/spark_rs.js';

const PAGE_SPLATS = 65536;

function generateSphereChunk(cx: number, cy: number, cz: number, count: number): Uint32Array {
  const xyz = new Float32Array(count * 3);
  const opacity = new Float32Array(count).fill(1.0);
  const rgb = new Float32Array(count * 3).fill(0.5);
  const scales = new Float32Array(count * 3).fill(0.1);
  const quaternions = new Float32Array(count * 4);
  
  for (let i = 0; i < count; i++) {
    const theta = Math.acos(-1 + (2 * i) / count);
    const phi = Math.sqrt(count * Math.PI) * theta;
    xyz[i * 3 + 0] = Math.sin(theta) * Math.cos(phi) * 10.0 + cx;
    xyz[i * 3 + 1] = Math.sin(theta) * Math.sin(phi) * 10.0 + cy;
    xyz[i * 3 + 2] = Math.cos(theta) * 10.0 + cz;
    quaternions[i * 4 + 3] = 1.0;
  }

  const gsplat = GsplatArray.from_attributes(xyz, opacity, rgb, scales, quaternions, undefined);
  gsplat.tiny_lod(1.5, false);
  const lodTree = gsplat.extract_lod_tree();
  gsplat.free();
  
  return lodTree;
}

function cameraMatrixFromRotationY(theta: number, tx: number, ty: number, tz: number): Float32Array {
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  return new Float32Array([
    cos,  0, sin, 0,
    0,    1, 0,   0,
    -sin, 0, cos, 0,
    tx,   ty, tz,  1,
  ]);
}

async function main() {
  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  // Generate a chunk with 1000 points (decent density to test boundaries)
  const lodTree = generateSphereChunk(0, 0, 0, 1000);
  const count = lodTree.length / 4;

  const { lodId } = init_lod_tree(0, new Uint32Array(0)) as { lodId: number };

  update_lod_trees(
    new Uint32Array([lodId]),
    new Uint32Array([0]),
    new Uint32Array([0]),
    new Uint32Array([count]),
    [lodTree],
  );

  const runTraverse = (theta: number) => {
    // Camera is placed at (0, 0, 15) and rotates around Y axis
    const camera = cameraMatrixFromRotationY(theta, 0, 0, 15);
    
    // We foveate forward with FOV 45 degrees
    const result = traverse_lod_trees(
      10000,
      0.02, // pixel_scale_limit
      undefined,
      new Uint32Array([lodId]),
      new Uint32Array([0]),
      camera,
      new Float32Array([1.0]), // lodScale
      new Float32Array([0.2]), // behindFoveate
      new Float32Array([0.4]), // coneFoveate
      new Float32Array([90.0]), // cone_fov0
      new Float32Array([120.0]), // cone_fov
    ) as {
      instanceIndices: { indices: Uint32Array; numSplats: number }[];
    };
    return Array.from(result.instanceIndices[0].indices.subarray(0, result.instanceIndices[0].numSplats));
  };

  // Run 100 frames of slow rotation (0.5 degrees per frame)
  const steps = 100;
  const dTheta = (0.5 * Math.PI) / 180; // 0.5 degrees
  
  let prevIndices: number[] = [];
  let totalChurn = 0;
  let maxChurn = 0;

  console.log(`Simulating ${steps} frames of camera rotation with foveation...`);
  
  for (let step = 0; step < steps; step++) {
    const theta = step * dTheta;
    const indices = runTraverse(theta);
    
    if (step > 0) {
      const prevSet = new Set(prevIndices);
      const currSet = new Set(indices);
      
      let added = 0;
      let removed = 0;
      
      for (const idx of indices) {
        if (!prevSet.has(idx)) added++;
      }
      for (const idx of prevIndices) {
        if (!currSet.has(idx)) removed++;
      }
      
      const churn = (added + removed) / prevIndices.length;
      totalChurn += churn;
      if (churn > maxChurn) maxChurn = churn;
      
      if (step % 20 === 0) {
        console.log(`  Frame ${step}: active splats = ${indices.length}, added = ${added}, removed = ${removed}, churn = ${(churn * 100).toFixed(2)}%`);
      }
    }
    
    prevIndices = indices;
  }

  const avgChurn = totalChurn / (steps - 1);
  console.log(`\n=== Traversal Stability Results ===`);
  console.log(`Average frame-to-frame index churn: ${(avgChurn * 100).toFixed(3)}%`);
  console.log(`Maximum frame-to-frame index churn: ${(maxChurn * 100).toFixed(3)}%`);

  // Assert that the churn is exceptionally small (< 1.5% on average)
  if (avgChurn > 0.015) {
    throw new Error(`Average index churn is too high: ${(avgChurn * 100).toFixed(2)}% (limit is 1.5%)`);
  }
  
  console.log(`HYSTERESIS TEMPORAL STABILITY TEST PASSED!`);
}

main().catch(err => {
  console.error('Test FAILED:', err);
  process.exit(1);
});
