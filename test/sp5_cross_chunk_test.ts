// Proves the cross-chunk LOD tree stitching fix works against the REAL
// compiled WASM traversal code (rust/spark-rs/src/lod_tree.rs's
// traverse_lod_trees), not just against the JS-side tree-building logic.
//
// Root cause being fixed: each SP5 chunk's synthesized flat LOD tree only
// ever covered that one chunk's own splats, with zero edges to any other
// chunk's tree. traverse_lod_trees can only output splats reachable by
// walking child_start/child_count pointers from the mesh's single root page,
// so a scene split into N disconnected per-chunk trees meant only the ONE
// chunk assigned the root page was ever discoverable or rendered --
// regardless of how many other chunks were separately fetched and uploaded.
// This reproduces exactly that scenario with a minimal 3-chunk synthetic
// scene and confirms: (1) with only chunk 0 resident, traversal's `chunks`
// output reports chunks 1 and 2 as needing to be fetched (the discovery/
// request-signaling that feeds SparkRenderer's fetch priority), and (2) once
// all three chunks are resident, traversal actually returns real splats from
// all three, not just chunk 0.
//
// Usage: npx tsx test/sp5_cross_chunk_test.ts

import fs from 'fs';
import path from 'path';
import init_wasm, {
  init_lod_tree,
  update_lod_trees,
  traverse_lod_trees,
} from '../rust/spark-rs/pkg/spark_rs.js';

const PAGE_SPLATS = 65536;

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

function writeEntry(
  lodTree: Uint32Array,
  idx: number,
  ex: number, ey: number, ez: number,
  size: number,
  childCount: number,
  childStart: number,
) {
  const o = idx * 4;
  lodTree[o + 0] = (float32ToHalfBits(ex) & 0xffff) | ((float32ToHalfBits(ey) & 0xffff) << 16);
  lodTree[o + 1] = (float32ToHalfBits(ez) & 0xffff) | ((float32ToHalfBits(size) & 0xffff) << 16);
  lodTree[o + 2] = childCount & 0xffff;
  lodTree[o + 3] = childStart >>> 0;
}

// Mirrors worker.ts's synthesizeFlatLodTreeIfMissing for a plain (no
// siblings) chunk: 1 root + (count-1) leaves, all chunk-local.
function buildPlainChunkTree(count: number, centers: [number, number, number][]): Uint32Array {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const [x, y, z] of centers) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  const size = Math.max(Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2, 1e-4);
  const tree = new Uint32Array(count * 4);
  writeEntry(tree, 0, cx, cy, cz, size, count - 1, 1);
  for (let i = 1; i < count; i++) {
    writeEntry(tree, i, centers[i][0], centers[i][1], centers[i][2], size, 0, 0);
  }
  return tree;
}

// Mirrors worker.ts's synthesizeFlatLodTreeIfMissing WITH sibling pointers,
// as chunk 0 now builds.
function buildChunk0TreeWithSiblings(
  count: number,
  centers: [number, number, number][],
  siblingChunks: { chunkIndex: number; center: [number, number, number]; size: number }[],
): Uint32Array {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const [x, y, z] of centers) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  const size = Math.max(Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2, 1e-4);
  const numSiblings = siblingChunks.length;
  const tree = new Uint32Array((count + numSiblings) * 4);
  writeEntry(tree, 0, cx, cy, cz, size, count - 1 + numSiblings, 1);
  for (let i = 1; i < count; i++) {
    writeEntry(tree, i, centers[i][0], centers[i][1], centers[i][2], size, 0, 0);
  }
  for (let s = 0; s < numSiblings; s++) {
    const sib = siblingChunks[s];
    const absoluteChildStart = ((sib.chunkIndex << 16) | 0) >>> 0;
    writeEntry(tree, count + s, sib.center[0], sib.center[1], sib.center[2], sib.size, 1, absoluteChildStart);
  }
  return tree;
}

function identityCameraMatrix(distance: number): Float32Array {
  // Column-major THREE.js-style matrixWorld: camera at (0,0,distance),
  // looking down -Z (no rotation), matching what compute_pixel_scale expects.
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, distance, 1,
  ]);
}

async function main() {
  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  // Three tiny synthetic chunks, spatially separated so their AABBs are
  // distinguishable, each with a handful of "splats" (LodSplat leaves).
  const chunk0Centers: [number, number, number][] = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]];
  const chunk1Centers: [number, number, number][] = [[100, 0, 0], [101, 0, 0], [99, 0, 0]];
  const chunk2Centers: [number, number, number][] = [[0, 100, 0], [0, 101, 0], [0, 99, 0]];

  const siblingChunks = [
    { chunkIndex: 1, center: [100, 0, 0] as [number, number, number], size: 2 },
    { chunkIndex: 2, center: [0, 100, 0] as [number, number, number], size: 2 },
  ];

  const chunk0Tree = buildChunk0TreeWithSiblings(chunk0Centers.length, chunk0Centers, siblingChunks);
  const chunk1Tree = buildPlainChunkTree(chunk1Centers.length, chunk1Centers);
  const chunk2Tree = buildPlainChunkTree(chunk2Centers.length, chunk2Centers);

  // Mirror SplatPager.ts's processFetched patch: EVERY fetched chunk's root
  // entry (word index 3, "child_start") gets patched from its chunk-relative
  // placeholder (1) to the absolute chunk address (chunk << 16 | 1) using
  // that chunk's own manifest index. This applies to every chunk, not just
  // chunk 0 -- chunk 0 additionally gets the sibling-pointer entries above,
  // but its own root patch works identically.
  const patchRootChildStart = (tree: Uint32Array, chunkIndex: number) => {
    tree[3] = ((chunkIndex * PAGE_SPLATS) + tree[3]) >>> 0;
  };
  patchRootChildStart(chunk0Tree, 0);
  patchRootChildStart(chunk1Tree, 1);
  patchRootChildStart(chunk2Tree, 2);

  // init_lod_tree with an empty tree, matching PagedSplats' newSharedLodTree
  // starting point (see SparkRenderer.ts's initLodTree for PagedSplats).
  const { lodId } = init_lod_tree(0, new Uint32Array(0)) as { lodId: number };

  // --- Phase 1: only chunk 0 is resident (uploaded to page 0). ---
  update_lod_trees(
    new Uint32Array([lodId]),
    new Uint32Array([0 * PAGE_SPLATS]), // page_base: chunk 0 lands on page 0
    new Uint32Array([0 * PAGE_SPLATS]), // chunk_base: this data IS chunk 0
    new Uint32Array([chunk0Tree.length / 4]),
    [chunk0Tree],
  );

  const camera = identityCameraMatrix(50); // close enough that chunk-sized AABBs exceed pixel_scale_limit
  const commonArgs = [
    new Uint32Array([lodId]), // lod_ids
    new Uint32Array([0]), // root_pages (page 0)
    camera,
    new Float32Array([1.0]), // lod_scales
    new Float32Array([1.0]), // behind_foveates
    new Float32Array([1.0]), // cone_foveates
    new Float32Array([0.0]), // cone_fov0s (0 disables cone)
    new Float32Array([0.0]), // cone_fovs
  ] as const;

  const phase1 = traverse_lod_trees(10000, 0.0001, undefined, ...commonArgs) as {
    chunks: [number, number][];
    instanceIndices: { numSplats: number }[];
  };
  const phase1TouchedChunks = new Set(phase1.chunks.map(([, chunk]) => chunk));
  console.log('Phase 1 (only chunk 0 resident):');
  console.log(`  touched chunks: [${[...phase1TouchedChunks].sort().join(', ')}]`);
  console.log(`  instance splat count: ${phase1.instanceIndices[0]?.numSplats ?? 0}`);

  const phase1DiscoveredSiblings = phase1TouchedChunks.has(1) && phase1TouchedChunks.has(2);
  console.log(`  discovered chunks 1 and 2 as needing fetch: ${phase1DiscoveredSiblings}`);

  // --- Phase 2: all three chunks now resident (as if chunks 1 and 2 finished fetching). ---
  update_lod_trees(
    new Uint32Array([lodId, lodId]),
    new Uint32Array([1 * PAGE_SPLATS, 2 * PAGE_SPLATS]), // pages 1 and 2
    new Uint32Array([1 * PAGE_SPLATS, 2 * PAGE_SPLATS]), // chunks 1 and 2
    new Uint32Array([chunk1Tree.length / 4, chunk2Tree.length / 4]),
    [chunk1Tree, chunk2Tree],
  );

  const phase2 = traverse_lod_trees(10000, 0.0001, undefined, ...commonArgs) as {
    instanceIndices: { numSplats: number; indices: Uint32Array }[];
  };
  const phase2Indices = phase2.instanceIndices[0];
  const chunksInOutput = new Set<number>();
  for (let i = 0; i < phase2Indices.numSplats; i++) {
    chunksInOutput.add(phase2Indices.indices[i] >>> 16);
  }
  console.log('\nPhase 2 (all three chunks resident):');
  console.log(`  total splats returned: ${phase2Indices.numSplats}`);
  console.log(`  pages represented in output: [${[...chunksInOutput].sort().join(', ')}]`);

  const allThreePagesPresent = chunksInOutput.has(0) && chunksInOutput.has(1) && chunksInOutput.has(2);

  console.log('\n=== Verdict ===');
  if (!phase1DiscoveredSiblings) {
    console.log('FAIL: traversal did not report chunks 1/2 as needing fetch while only chunk 0 was resident -- cross-chunk discovery is not working.');
    process.exit(1);
  }
  if (!allThreePagesPresent) {
    console.log(`FAIL: once all chunks were resident, output only contained pages [${[...chunksInOutput].sort().join(', ')}] -- expected all of 0, 1, 2.`);
    process.exit(1);
  }
  console.log('PASS: chunk 0 correctly discovers and (once resident) renders content from sibling chunks 1 and 2 via the stitched tree.');
}

main().catch((e) => {
  console.error('CROSS-CHUNK TEST CRASHED:', e);
  process.exit(1);
});
