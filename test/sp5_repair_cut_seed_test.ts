// Verifies P2a: dynamic_traverse_lod_trees seeds cut state so repair_lod_cut
// does not always return needsFull on the next small camera move.
//
// Usage: npx tsx test/sp5_repair_cut_seed_test.ts

import fs from "fs";
import path from "path";
import init_wasm, {
  GsplatArray,
  init_lod_tree,
  update_lod_trees,
  dynamic_traverse_lod_trees,
  repair_lod_cut,
} from "../rust/spark-rs/pkg/spark_rs.js";

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
    cos, 0, sin, 0,
    0, 1, 0, 0,
    -sin, 0, cos, 0,
    tx, ty, tz, 1,
  ]);
}

function lodCameraArgs(lodId: number, camera: Float32Array) {
  return {
    lodIds: new Uint32Array([lodId]),
    rootPages: new Uint32Array([0]),
    camera,
    lodScales: new Float32Array([1.0]),
    behindFoveates: new Float32Array([0.2]),
    coneFoveates: new Float32Array([0.4]),
    coneFov0s: new Float32Array([90.0]),
    coneFovs: new Float32Array([120.0]),
    pageBounds: new Float32Array(0),
  };
}

async function main() {
  const wasmPath = path.resolve("./rust/spark-rs/pkg/spark_rs_bg.wasm");
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  const lodTree = generateSphereChunk(0, 0, 0, 2000);
  const count = lodTree.length / 4;
  const { lodId } = init_lod_tree(0, new Uint32Array(0)) as { lodId: number };
  update_lod_trees(
    new Uint32Array([lodId]),
    new Uint32Array([0]),
    new Uint32Array([0]),
    new Uint32Array([count]),
    [lodTree],
  );

  const camera0 = cameraMatrixFromRotationY(0, 0, 0, 15);
  const args0 = lodCameraArgs(lodId, camera0);
  const seedResult = dynamic_traverse_lod_trees(
    10000,
    0.02,
    undefined,
    args0.lodIds,
    args0.rootPages,
    args0.camera,
    args0.lodScales,
    args0.behindFoveates,
    args0.coneFoveates,
    args0.coneFov0s,
    args0.coneFovs,
    args0.pageBounds,
    true,
  ) as {
    instanceIndices: { indices: Uint32Array; numSplats: number }[];
  };
  const seedCount = seedResult.instanceIndices[0]?.numSplats ?? 0;
  if (seedCount === 0) {
    throw new Error("dynamic seed traversal returned zero splats");
  }

  // Small rotation (~0.5°) — within repair band (50u / 15°)
  const camera1 = cameraMatrixFromRotationY((0.5 * Math.PI) / 180, 0, 0, 15);
  const repairStart = performance.now();
  const args1 = lodCameraArgs(lodId, camera1);
  const repairResult = repair_lod_cut(
    10000,
    0.02,
    args1.lodIds,
    args1.rootPages,
    args1.camera,
    args1.lodScales,
    args1.behindFoveates,
    args1.coneFoveates,
    args1.coneFov0s,
    args1.coneFovs,
    args1.pageBounds,
  ) as {
    needsFull?: boolean;
    instanceIndices: { indices: Uint32Array; numSplats: number }[];
  };
  const repairMs = performance.now() - repairStart;

  if (repairResult.needsFull) {
    throw new Error("repair_lod_cut returned needsFull after dynamic seed — cut was not seeded");
  }

  const repairCount = repairResult.instanceIndices[0]?.numSplats ?? 0;
  if (repairCount === 0) {
    throw new Error("repair_lod_cut returned zero splats");
  }

  for (const idx of repairResult.instanceIndices[0].indices.subarray(0, repairCount)) {
    if ((idx >>> 24) !== 0) {
      throw new Error(`P2b regression: repair index 0x${idx.toString(16)} has bits [31:24] set`);
    }
  }

  console.log(`Seed cut: ${seedCount} splats via dynamic_traverse_lod_trees`);
  console.log(`Repair: ${repairCount} splats in ${repairMs.toFixed(2)}ms (needsFull=false)`);
  console.log("PASS: dynamic mode seeds cut state for incremental repair");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
