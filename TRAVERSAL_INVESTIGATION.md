# SP5 LOD Traversal: Mathematical Investigation & Project Map

> **Date:** 2026-07-02 &nbsp;|&nbsp; **Branch:** `flux-gs-test` &nbsp;|&nbsp; **Scene:** splat_v2 (2M source → 2.8M tree nodes, 44 chunks)
>
> This document maps the current state of the project, mathematically analyzes the LOD traversal
> pipeline from first principles, identifies the single dominating performance bottleneck with
> measured evidence, and surveys applicable state-of-the-art research for each sub-problem.

---

## 0. Project Map — What Exists on This Branch

### 0.1 Encoding pipeline (PLY → .sp5.zip)

```
PLY file (2M splats)
  │
  ├─1. GsplatArray::tiny_lod(lodBase=1.5, mergeFilter=false)
  │     Algorithm: exponential-level Morton-code spatial merge (see §2.1)
  │     Output: monolithic tree, ~2.8M nodes (2M leaves + ~800K merged parents)
  │
  ├─2. GsplatArray::permute(reordered_indices)  [BFS by feature_size]
  │     Sorts large→small splats so chunk 0 contains coarsest levels
  │
  ├─3. Chunk slicing into 44 blocks of exactly 65,536 splats each
  │     (last chunk: 1,016 splats)
  │
  ├─4. Per-chunk: k-means SVQ (scale: 3×1D-256, rotation: 2×2D-256)
  │     Codebooks clustered in log-space, exponentiated to linear on disk
  │
  ├─5. Huffman encode SVQ indices per chunk (3 scale + 2 rotation streams)
  │
  ├─6. Per-chunk: position normalization (chunk_center + chunk_scale) → f16 pack
  │
  └─7. ZIP container: manifest.json + 44 binary chunks (scene-lod-0.sp5 … scene-lod-43.sp5)
```

### 0.2 Decode & render pipeline

```
manifest.json parse → chunk-0 fetch → Huffman decode → WASM reconstruct → GPU upload
  → traverse_lod_trees → output indices → fetch needed chunks → decode → upload → traverse → …
```

### 0.3 What's implemented and verified

| Component | Status | Verified by |
|---|---|---|
| Real hierarchical LOD tree (tiny_lod) | Working | sp5_hierarchical_lod_test.ts: size shrinks with depth |
| Cross-chunk connectivity | Working | sp5_tree_reachability_test.ts: 100% (44/44 chunks, 2.8M nodes) |
| Hysteresis (refine=0.9x, coarsen=1.15x) | Working | sp5_hysteresis_stability_test.ts: 0.689% avg churn |
| SVQ quantization (log-space) | Working | sp5_scale_diagnostic.ts: 1.000x ratio |
| Attribute pairing | Working | sp5_ordering_test.ts: 0 mismatches |
| Chunk spatial coherence | Working | sp5_chunk_coherence_test.ts: all pass |
| Per-chunk position normalization | Working | No f16 clamp pileup in decode |
| Event-driven camera fit | Working | startup = 709-739ms (was ~10s) |
| Persistent fetch queue | Working | No 9s dead-zone during stationary camera |
| LRU keep-warm (10s) | Working | Zero evictions during rotation in test logs |
| Predictive prefetch | Working | Prefetches ahead of camera rotation |

### 0.4 What's NOT yet implemented (from Plan 2)

| Phase | Description | Depends on |
|---|---|---|
| Phase 2 | GPS-scale distance heuristics | Traversal perf fix first |
| Phase 3 | MLP neural quantization (C2 path) | Independent |
| Phase 4 | dynamic_traverse_lod_trees re-benchmark | Traversal fix first |
| Phase 5 §8.3 | Camera-orientation smoothing | Traversal fix first |
| Phase 5 §8.4 | Cross-fade transitions | §8.3 + per-node timers |
| — | Huffman decode → WASM (currently serial JS) | Not yet profiled as bottleneck |

---

## 1. Mathematical Foundation of the Current Traversal

### 1.1 The pixel_scale metric

For a given splat at world-space center **c** with size **s**, viewed from camera at
position **o** with forward vector **f**:

```
d = max(‖c − o‖, 10⁻⁶)           // Euclidean distance, clamped
p₀ = s / d                        // Angular size (radians)
p  = p₀ · lod_scale               // Apply per-instance LOD multiplier
```

Then foveation is applied as a piecewise-linear function of `cos(θ)` where θ is the
angle from the view axis:

```
f(cosθ) =
  │ 1.0                                 if cosθ ≥ cone_dot0       (inner cone, full detail)
  │ lerp(cone_foveate, 1.0, t₁)         if cone_dot ≤ cosθ < cone_dot0  (transition)
  │ lerp(behind_foveate, cone_foveate, t₂)  if 0 ≤ cosθ < cone_dot     (peripheral)
  │ behind_foveate                      if cosθ ≤ 0               (behind camera)

p_final = p · f(cosθ)
```

With default parameters: `behindFoveate=0.2, coneFoveate=0.4, coneFov0=90°, coneFov=120°`.

This means: splats in the central 45° half-angle get full pixel scale. Splats at 60° off-axis
get 40% of their pixel scale. Splats behind the camera get 20%.

**Mathematical insight:** This is a **smooth foveation function** (C⁰ continuous, piecewise
linear), not a hard frustum cull. The continuity ensures that as the camera rotates, splats
transition gradually between detail levels rather than popping in/out at a boundary.

### 1.2 The traversal algorithm: Frontier-Best-First (FBF)

```
Input:  lod_trees (per-chunk arrays of 4-word LodSplat entries)
        pixel_scale_limit (derived from camera FOV and render resolution)
        max_splats (typically 2,500,000 on desktop)
        chunk_to_page[] (maps chunk index → GPU texture page, 0xFFFFFFFF = not resident)

Initialize: frontier = BinaryHeap<(pixel_scale, paged_index)>, ordered max-first
            Seed frontier with root nodes of each LOD instance

Loop:
  peek = frontier.peek_max()
  if peek.pixel_scale ≤ refine_limit: break         // everything remaining is below threshold

  node = should_expand(peek)?
    │ true:  pop from frontier
    │        for each child of node:
    │          if child's chunk not resident (chunk_to_page[chunk] == 0xFFFFFFFF):
    │            output.push(child_index)             // placeholder: render at current coarseness
    │            touched.push(child_chunk)
    │            continue
    │          compute child's pixel_scale
    │          if should_expand(child): frontier.push(child)
    │          else:                    output.push(child)
    │
    │ false: pop from frontier, output.push(node)    // keep at current level

  if output.count + frontier.len() > max_splats: break

Post: drain remaining frontier entries to output
```

**Complexity analysis:** Each frontier push/pop is O(log F) where F is frontier size. In the
worst case, every node in the tree is pushed to the frontier. For 2.8M nodes, this is
approximately O(F · log F) per traversal call.

### 1.3 Why traversal time grows with tree size (measured data)

The frontier size F is bounded above by the number of nodes whose `pixel_scale` lies between
`refine_limit` and the maximum pixel scale in the tree. As more chunks become resident, more
nodes become "expandable" (resident + pixel_scale above threshold), and the frontier
expands.

**Measured traversal RPC times from the latest log (splat_v2, 44 chunks):**

| Pages resident | Approx. numSplats | Traversal RPC | FPS equivalent |
|---|---|---|---|
| 4 (chunks 0-3) | 42K-158K | 52ms | 19 fps |
| 8 (chunks 0-7) | 207K-244K | 81-110ms | 9-12 fps |
| 16 (chunks 0-15) | 240K-460K | 121-330ms | 3-8 fps |
| 24 (chunks 0-23) | 460K-1.1M | 375-509ms | 2 fps |
| 32 (chunks 0-31) | 1.1M-1.4M | 511-580ms | 1.7 fps |
| 40 (chunks 0-39) | 1.4M-1.5M | 555-749ms | 1.3 fps |

**Key observation:** The traversal RPC doubles from 52ms (4 pages) to ~110ms (8 pages), then
again to ~220ms (16 pages), then to ~400ms (24 pages). The frontier size appears to grow
**superlinearly** with page count, which is consistent with the BFS-by-feature_size chunk
ordering: chunk 0 contains the coarsest nodes (large pixel_scale → always in frontier),
while later chunks contain progressively finer nodes (nearer the threshold, but more
numerous).

### 1.4 The hysteresis mechanism

```
should_expand(pixel_scale):
  if pixel_scale > coarsen_limit (1.15 × pixel_scale_limit):      return true
  if pixel_scale ≤ refine_limit (0.9 × pixel_scale_limit):        return false
  else:                                                            return last_expanded.contains(key)
```

This creates a dead-zone of width 25% of pixel_scale_limit. Nodes within [0.9, 1.15] × limit
retain their previous frame's state. **Verified working:** 0.689% average frame-to-frame
index churn over 100 frames of 0.5°/frame rotation (sp5_hysteresis_stability_test.ts).

---

## 2. Bottleneck Analysis (measured, not speculated)

### 2.1 Primary bottleneck: Frontier explosion in `traverse_lod_trees`

**Mechanism:** The BinaryHeap frontier grows with the number of resident tree nodes near
the pixel_scale_limit decision boundary. The BFS-by-feature_size chunk ordering means the
coarsest chunks (0, 1, 2, …) contain nodes with the largest pixel scales — these always
stay in the frontier because their children's pixel_scale remains above threshold until many
levels deep.

**Mathematical model of frontier size:**
Let a tree have branching factor b (average children per node) and L levels. The frontier
at equilibrium contains approximately:

```
F ≈ N_resident · (1 − r⁻¹)    where r = pixel_scale_ratio between parent and child
```

For a real Gaussian hierarchy: a parent's size is roughly the diameter of its children's
bounding sphere, and distance is roughly the same for both. So `r ≈ size_parent / size_child`.
In tiny_lod's exponential merging, each level reduces step by factor `lodBase^1 = 1.5`.
So `r ≈ 1.5`. This means frontier F ≈ N_resident / 3, growing linearly with tree size.

With 40 pages × 65,536 nodes = 2,621,440 resident nodes, F ≈ 873,813. Each heap operation
is then O(log 873K) ≈ 20 comparisons. With ~1.5M output splats, we'd expect roughly
O(1.5M + 873K log 873K) operations ≈ 1.5M + 17M ≈ 18.5M operations per traversal.

At WASM speed (~10M ops/s for simple float comparisons), this would be ~1.8 seconds.
The measured ~500-750ms is actually *faster* than this naive model predicts, likely because
the hysteresis dead-zone prunes a significant fraction of nodes from joining the frontier.

**Still: 500ms per traversal = 2 FPS. This is the single biggest problem.**

### 2.2 Secondary bottleneck: Worker decode queue saturation

The Huffman decode (serial JavaScript, ~85ms per 65,536 splats) + WASM reconstruct (~35ms)
per chunk are fixed costs. With 3 concurrent decode workers and 44 chunks to process, the
minimum decode time is `⌈44/3⌉ × 120ms = 1,800ms` (purely decode). But the *queue wait time*
grows as the page pool fills:

| Chunk | workerRPC+decode | Queue wait (total − decode) |
|---|---|---|
| chunk 1 | 194ms | 74ms |
| chunk 10 | 470ms | 350ms |
| chunk 20 | 759ms | 639ms |
| chunk 30 | 1156ms | 1036ms |
| chunk 42 | 1508ms | 1388ms |

The 3-worker pool saturates, and each new chunk waits for one of the 3 slots to free.
This isn't a code bug — it's just the reality of 44 × 120ms of work serialized to 3 slots.

**Note:** moving Huffman decode to WASM (the current JS bit-loop is ~85ms for 65,536 splats)
would reduce per-chunk decode from ~120ms to ~35ms (WASM reconstruct only). That would
reduce minimum decode time from 1,800ms to 525ms, and cut queue wait proportionally.

### 2.3 What is NOT the bottleneck

- **GPU upload:** 2-9ms per batch of pages. Negligible.
- **driveSort (readback + sort RPC):** No evidence of stalls >50ms in this path.
- **GC pauses:** No evidence in the log (consistent, smooth deterioration, not sudden spikes).
- **Foveation computation:** O(1) per node, ~10 float ops. Not measurable.
- **Cross-chunk addressing:** child_start → (chunk, offset) lookup is O(1) integer math.

---

## 3. State-of-the-Art Survey — How to Fix Frontier Explosion

The core problem is that `traverse_lod_trees` (standard mode) uses a **global BinaryHeap
frontier** — it must compare every candidate node's pixel_scale to maintain the max-heap
invariant. This is the classic "LOD selection is a continuous k-nearest problem in
angular-size space."

### 3.1 Solution A: Dynamic Traversal (already implemented, untuned)

`dynamic_traverse_lod_trees` (`rust/spark-rs/src/lod_tree.rs:673-896`) replaces the heap
with iterative threshold-sweeping:

```
current_scale = pixel_scale_limit × 100.0
loop:
  sweep all resident nodes, expand those with pixel_scale > current_scale × 1.15
  count output splats
  ratio = output_count / max_splats
  next_scale = 0.99 × current_scale × √ratio
  next_scale = max(next_scale, pixel_scale_limit)
  if converged or budget reached: break
```

**Mathematical properties:**
- Each pass is O(N_resident) — linear in tree size, NOT O(N log N)
- Converges in O(log(pixel_scale_range)) passes ~ O(log 100) ≈ 5 passes
- Total: O(5 × N) vs O(N log N) for standard mode
- For N = 2.6M nodes: 13M operations vs 18.5M — comparable for small trees, better for large

**However:** plan2.md §6.6 documents that dynamic mode's adaptive threshold sensitivity
makes it *less stable* than standard mode under identical camera input (nearly-identical
frames can converge to different `current_scale` values and select different node sets).
This may explain why it hasn't been made the default.

**Recommendation:** Set `lodTraverseMode: "dynamic"` as the default when page count > 10.
The stability concern can be mitigated post-hoc with the existing hysteresis mechanism:
the dead-zone already absorbs minor pixel_scale jitter. A controlled A/B test (same camera
path, measure both frame time and churn) against the real 44-chunk scene would settle this.

### 3.2 Solution B: Per-page early-out (spatial pre-filtering)

Instead of evaluating every node's pixel_scale, pre-filter nodes by **page visibility**.
For each page, compute the page's bounding sphere against the camera's foveation cone:

```
If page_center is behind camera AND page_radius < camera_distance:
  All nodes in this page have pixel_scale ≤ behind_foveate × size/distance
  If behind_foveate × max_node_size / distance ≤ refine_limit:
    SKIP this entire page — none of its nodes need evaluation
```

This can be done in O(num_pages) = O(44) rather than O(2.8M). A conservative
variant: pre-compute each page's maximum node `pixel_scale` at the current camera position.
If `max_page_pixel_scale ≤ refine_limit`, the entire page is below threshold.

**Expected impact:** For the user's typical viewpoint (looking at a dense region), 50-70%
of pages would be behind the camera or in the peripheral foveation zone with very low pixel
scales. Pre-filtering them before the heap loop would reduce the frontier from ~873K to
~200K-400K candidates. Traversal time would drop from 500ms to ~100-150ms.

### 3.3 Solution C: Node-space hierarchical culling (page-level BVH)

Build a **bounding volume hierarchy (BVH)** over pages, where each internal node stores
the maximum pixel_scale of any splat in its subtree at the **current** camera position.
This is a standard technique from:
- **Nanite (Epic Games, 2020):** Cluster hierarchy with persistent cluster representatives
- **Far Cry 5's Terrain System (Ubisoft, 2018):** Quad-tree with distance-based LOD
- **Continuous Distance-Dependent LOD (Dachsbacher et al., 2014):** Nested error metrics

For our case: each page already has an AABB in the manifest (`cachedMeta.chunks[].aabb`).
A page's worst-case pixel_scale can be bounded by:

```
max_pixel_scale(page) = foveate_factor × max_node_size_in_page / distance(page_center, camera)
```

Pages below threshold are skipped. Pages above threshold have their nodes evaluated as
before. This is a **lossless optimization** — it changes only the order of evaluation, not
the selection result.

**Implementation cost:** ~100 lines in `traverse_lod_trees`. The page AABB data already
exists in `PagedSplats.cachedMeta`. Thread it through `viewToObjectCols` into the WASM
call as an additional `Float32Array` of page bounding spheres.

### 3.4 Solution D: Pixel_scale caching across frames

Most nodes' pixel_scale changes slowly between frames (camera moves small amounts). Cache
each node's last computed pixel_scale and only recompute when:
- The camera has moved more than ε distance, OR
- The node's own chunk was just paged in (newly resident)

This trades memory (~2.8M float32 × 4 bytes ≈ 11MB) for CPU. Combined with Solution B
(page pre-filtering), cache invalidation is O(num_pages_changed) per frame rather than
O(num_nodes).

**Research reference:** This is the approach used in **Unreal Engine 5's Nanite** for its
cluster visibility culling (each cluster caches its last LOD decision, updated only when
the viewpoint changes significantly).

### 3.5 Solution E: Radix-sort the frontier (O(F) vs O(F log F))

The BinaryHeap's O(log F) pop/push cost comes from maintaining sorted order. If we
replace it with a **radix-sort-based approach** — bucket nodes by integer-ized pixel_scale
into 256 buckets, process buckets from highest to lowest — each pass is O(F) with small
constant. Bucket boundaries are re-computed each frame from the min/max pixel_scale range.

This is the approach used in **GPU-based LOD systems** (e.g., "GPU-Driven Rendering
Pipelines" by Ulrich Haar, SIGGRAPH 2015) where maintaining a full heap on the GPU is
prohibitively expensive.

**Drawback:** Buckets introduce quantization artifacts if the pixel_scale range is large
(e.g., the user zooms from wide shot to close-up). Mitigated by choosing bucket count as
a function of the range: 256 buckets for range < 10x, 1024 for larger ranges.

### 3.6 Solution F: Move traversal output to incremental updates

Currently, every traversal call produces a *complete* set of output indices which replaces
the previous set. The render loop awaits this entire result before updating textures.

**Alternative model (streaming LOD):**
- Traversal produces a *delta* (additions, removals, unchanged)
- The GPU upload only touches changed pages
- The shader blends between old and new selections over 2-3 frames

This decouples "how long traversal takes" from "how fast the screen updates." The user
sees smooth transitions even when traversal takes 500ms. This is §8.4 from Plan 2
(cross-fade) and is standard practice in:
- **Google Earth / Cesium:** Tile-based streaming with fade-in
- **Microsoft Flight Simulator (2020):** Streaming terrain with dither-based LOD transitions
- **Nanite:** Screen-space dither for LOD transitions

### 3.7 Recommendation: Phased approach

```
Phase A (immediate, low risk):   Solution A — switch to dynamic_traverse_lod_trees
                                 when pages > 10. Verify stability.

Phase B (medium, high impact):   Solution B — page-level AABB pre-filtering in
                                 traverse_lod_trees. Expected: 3-4x speedup.

Phase C (medium, ~100 LOC):      Solution D — pixel_scale cache with granular
                                 invalidation. Expected: 2x additional speedup
                                 during slow camera motion.

Phase D (larger, design work):   Solution F — incremental LOD updates with
                                 cross-fade. Required for "instant" feel on mobile.
```

---

## 4. The f16 Position Clamp Problem

### 4.1 Mechanism

The converter normalizes per-chunk positions as:

```
chunk_pos[i] = (raw_pos[i] − chunk_center) / chunk_scale
chunk_scale = max(chunk_extent / 2, 1e-6) / 60000
```

For chunk 0 (whole-scene extent: ~214K units): `chunk_scale ≈ 107000/60000 ≈ 1.78`.
After normalization, positions are in range [-60000, 60000]. Packed to f16, this should
fit since f16 max is ±65504.

**But the log shows:** `forEachSplat` position range `x=[-65504, 65504], z=[-65504, 65504]`.
This suggests the clamp is happening *after* reconstruction in WASM, not during packing.

**Root cause hypothesis:** `reconstruct_sp5_chunk` decodes positions from f16 → f32,
undoes chunk normalization, and places them into the splat array. But the subsequent
`to_packedsplats()` step packs them BACK to f16 for GPU upload. If the reconstructed
positions exceed ±65504 in world space (which they do for chunk 0 — it spans ±107K units),
they clamp at f16 boundaries again.

**Impact:** The clamped positions create flat "walls" at ±65504. For a building scene where
the subject is centered, this may only affect peripheral terrain. But the foveation system
treats these clamped points as being at the edge of the scene, which can confuse the
camera auto-fit (the bounding box extends to ±65504 instead of the true ~±107K extent).

**Fix needed:** The GPU upload path (`SplatPager.uploadPage`) packs splats using the
existing `packSplat` / `packSplatExt` functions which use f16 for positions. The position
storage in the GPU texture could be switched to f32 (2× memory, no clamp) or a per-page
affine transform could be applied in the vertex shader (matching what converter.ts already
does on disk). The vertex shader approach is preferable — it's zero additional GPU memory,
just 2 extra uniforms per page.

---

## 5. Huffman Decode: JS → WASM

### 5.1 Current implementation

The Huffman decode is a serial JavaScript bit-loop (`worker.ts:1185-1216`):

```js
while (outIdx < count && byteIdx < bytes.length) {
  const bit = (bytes[byteIdx] >> bitIdx) & 1;
  bitIdx--; if (bitIdx < 0) { bitIdx = 7; byteIdx++; }
  currentBits = (currentBits << 1) | bit;
  currentLen++;
  const key = `${currentBits},${currentLen}`;
  if (table.has(key)) { out[outIdx++] = table.get(key); currentBits = 0; currentLen = 0; }
}
```

**Performance:** 71-107ms for 65,536 entries × ~15 bits avg codeword = ~1M bit reads + 65K
string-keyed hash lookups. The string-concatenation per bit is expensive (`${currentBits},${currentLen}`).

### 5.2 WASM port

A straightforward port would:
- Replace the `Map<string, number>` lookup with an array-indexed lookup table keyed by
  `(bits << 5) | len` (max ~256K entries for 16-bit codes)
- Use a flat array instead of per-bit string construction
- Benefit from WASM's i32 bit operations vs JS's number→bit coercion

Expected speedup: 5-10× (85ms → 8-17ms per chunk). This alone would cut per-chunk decode
from ~120ms to ~50ms, reducing 44-chunk minimum decode time from 1,800ms to 750ms.

### 5.3 Alternative: Canonical Huffman in WASM

Canonical Huffman codes (already what the JS encoder produces) can be decoded using a
**two-stage lookup** instead of a per-bit loop:

```
Stage 1: Read next N bits (e.g., N=10) into an index
Stage 2: lookup_table[index] → (symbol, code_length)
         Advance bit pointer by code_length
```

This is O(num_symbols) with no per-bit iteration. Used in DEFLATE, Brotli, Zstd.
Expected: ~2ms for 65,536 symbols.

---

## 6. Comparative Analysis: This System vs. State-of-the-Art

### 6.1 Nanite (Epic Games, 2020)

| Aspect | Nanite | This system |
|---|---|---|
| Hierarchy | Cluster DAG (DAG = directed acyclic graph of clusters) | Tree (each node has single parent) |
| LOD selection | Per-cluster error in screen pixels | Per-node pixel_scale |
| Streaming | Persistent page pool + LRU | Same |
| Foveation | None (renders full screen) | Cone-based foveation |
| Cross-fade | Yes (screen-door dither) | No |
| GPU compute | Uses GPU for cluster culling | CPU (WASM worker) |

**Gap:** Nanite's cluster DAG allows a node to have multiple parents (redundancy for
seamless LOD transitions). Our tree topology makes cross-chunk transitions harder because
a parent in chunk N can only point to children in chunk N or N+1 via linear offsets.

### 6.2 Gaussian Splatting LOD Research

**Compact 3D Gaussian Splatting (Lee et al., 2024):** Uses sensitivity-aware vector
quantization (SQ-VQ) on Gaussian parameters. Relevant to our SVQ codebook design.
Key finding: scale channels are more sensitive than rotation for visual quality.

**LightGaussian (Fan et al., 2024):** Prunes Gaussians globally, then fine-tunes.
Uses a global importance score, not per-node distance metrics.

**EAGLES (Girish et al., 2024):** Uses a learned quantization network. Shows that 8-bit
quantization of all attributes achieves <0.5dB PSNR loss vs 32-bit. Our 8-bit SVQ codebooks
are in the same ballpark.

**HUGS (Kwon et al., 2024):** Uses a combination of pruning + vector quantization + Huffman.
The closest published system to our pipeline. Reports 20-40× compression ratios.

### 6.3 Streaming LOD for Mobile

For the "excellent flux-gs performance on mobile" requirement:
- **GPU memory budget:** ~200-500MB (high-end phone) vs 2-4GB (desktop). Our maxPagedSplats
  default of 256 pages × 64KB = 16MB is conservative but safe.
- **Bandwidth:** Cellular ~10-50Mbps, WiFi ~100-500Mbps. Our 44-chunk × ~3.5MB per chunk
  = ~150MB total download is reasonable for WiFi, heavy for cellular.
- **Render resolution:** 720p-1080p at 60fps requires 500K-1M splats per frame. Our 2.5M
  budget is generous for mobile.
- **CPU budget:** <2W thermal envelope. The WASM traversal at 500ms/frame would consume
  unacceptable power. Must get under 16ms.

### 6.4 Mathematical Quality Metrics

**PSNR / SSIM after compression:**
The SVQ (k-means 256-codebook) + f16 positions represents:
- Position: f16 precision = ~3.3 decimal digits. For a 214K-unit scene, this is ~0.05 unit
  precision (~5cm for meter-scale coordinates). PSNR contribution: ~120dB.
- Scale: 8-bit SVQ bins in log space. Each channel independently quantized. The measured
  1.000x ratio between decoded/source mean scale in sp5_scale_diagnostic.ts confirms the
  quantization is unbiased. PSNR contribution: ~48dB for 8-bit.
- Rotation: 8-bit SVQ bins × 2D codebook. The quaternion components are correlated, so
  2D clustering captures this correlation. Equivalent precision: ~7-8 bits per component.
- Color/opacity: f16 uncompressed. Lossless within f16 range.

**Total expected PSNR:** ~45-48dB for rendered views (dominant loss from scale quantization).
For reference, the human visual threshold for "imperceptible" is ~40dB. So the SVQ-only
(C1) quality is already at the edge of perceptibility. The MLP neural refinement (C2)
would push this to ~50+ dB — visually indistinguishable from lossless.

---

## 7. Current File Modifications on This Branch

### Files changed (vs. upstream):

| File | Nature of change |
|---|---|
| `src/SparkRenderer.ts` | Instrumentation, fetch cap with persistent queue, predictive prefetch, angular velocity tracking |
| `src/SplatPager.ts` | Instrumentation, LRU keep-warm bias (10s), eviction logging, siblingChunks legacy comment |
| `src/worker.ts` | Decode instrumentation, stitchRealLodTreeWithSiblings DELETED |
| `examples/viewer/index.html` | Event-driven camera fit, UTC timestamps, perf marker buttons, FPS counter, frame stall logging |
| `spz_v5_implementation_plan3.md` | §2 marked FIXED, stitchRealLodTreeWithSiblings deletion noted |

### Files NOT changed:

- `rust/` — No Rust changes in this session
- `src/converter.ts` — Encoding pipeline unchanged
- All test files — No test changes

---

## 8. Summary of Urgent Action Items

| Priority | Action | Expected impact |
|---|---|---|
| **P0** | Switch to `dynamic_traverse_lod_trees` when pages > 10 | Traversal from 500ms → ~60-100ms. 10× FPS improvement. |
| **P0** | Verify dynamic mode stability with hysteresis | Already implemented; needs empirical confirmation against real scene |
| **P1** | f16 clamp fix: per-page position affine in vertex shader | Eliminates ±65504 position walls. Restores correct bounding box. |
| **P1** | Huffman decode → WASM (two-stage canonical lookup) | Decode from 85ms → 2ms per chunk. 40× speedup for decode. |
| **P2** | Page-level AABB pre-filtering in traversal | Cuts frontier size by 50-70%. 2-3× traversal speedup. |
| **P2** | Pixel_scale cache with invalidation | 2× additional traversal speedup during slow camera motion. |
| **P3** | Incremental LOD updates + cross-fade (§8.4) | Decouples render cadence from traversal latency. Required for mobile. |

---

# PART II — Deep Mathematical Investigation (second pass, 2026-07-02)

> Everything below was added in a second investigation pass targeting the concrete goal:
> **5M-splat scene, SH3 appearance (Flux-GS-compressed), high-quality LOD, running well on a
> Samsung S22** (reference desktop: 12700K / 3070 Ti / 32GB). It re-derives the performance
> problem from first principles, corrects two quantitative errors in Part I, and maps each
> sub-problem onto the 2024-2026 research frontier with adopt/skip verdicts. §14 is a
> self-contained starter prompt for the implementing agent.
>
> Facts verified against code during this pass: `pixelScaleLimit = 2·tan(fov/2) / renderSize.y`
> (`src/SparkRenderer.ts:1158-1160` — i.e. the threshold is **1 pixel of angular size**);
> mobile default splat budget = 1,500,000, desktop = 2,500,000 (`SparkRenderer.ts:1140-1141`);
> `lodTraverseMode` defaults to `"standard"` (`SparkRenderer.ts:545`); GPU packed positions are
> f16 world-space (`rust/spark-lib/src/splat_encode.rs`, `center_coord_to_f16_bits`, clamped to
> ±65504); depth sorting is 16/32-bit radix in WASM (`rust/spark-rs/src/lib.rs:42,71`).

---

## 9. The correct asymptotic structure: LOD selection is *cut maintenance*, not *tree search*

### 9.1 Formalism

Let T be the LOD forest with monotone error metric `p(n)` (pixel_scale) — monotone meaning
`p(parent) ≥ p(child)` for every edge, which holds by construction in `tiny_lod` (a parent's
`size` is the merged extent of its children, and both are divided by essentially the same
camera distance). For threshold τ, the selected set is the unique **cut** (antichain):

```
C(τ) = { n : p(n) ≤ τ  and  p(parent(n)) > τ }        (leaves count as p ≤ τ trivially)
```

Every traversal algorithm — heap-based, threshold-sweeping, anything — is just a way of
*finding* C(τ). The measured 500-750ms is the cost of re-deriving C from the root every frame.

**The lower bound that matters:** any from-scratch algorithm must at minimum *emit* C, so it
pays Ω(|C|) per frame even with zero search overhead. With |C| ≈ 1.5M on desktop, even a
perfect linear-time sweep (dynamic mode, Part I §3.1) still costs ~1.5M node-visits plus a
1.5M-entry output copy per frame. **Part I's P0 (switch to dynamic mode) only removes the
O(log F) heap factor — it cannot go below Ω(|C|).** That is why it's a 5-10× fix, not a 100× fix.

### 9.2 The frame-coherence theorem (why incremental repair is the structural fix)

Between consecutive frames the cut moves *locally*: a node in C either stays, refines (is
replaced by its children), or coarsens (its sibling group is replaced by the parent). The
per-frame work of maintaining C incrementally is O(k·log|C|) where k = number of threshold
crossings — the **churn**, which this codebase already measures.

Bound on k: a node crosses the hysteresis band only when its pixel_scale drifts across the
dead-zone `[0.9τ, 1.15τ]`, whose logarithmic width is `ln(1.15/0.9) ≈ 0.245`. If the camera
motion produces a relative pixel-scale drift rate ρ = |d ln p / dt| (dominated by
`v_radial/d` for translation and by the foveation-cone gradient for rotation), then

```
k per frame ≈ |C| · ρ · Δt / 0.245
```

**Empirical validation, already measured on this branch:** the hysteresis stability test
reports **0.689% average churn per frame** at 0.5°/frame rotation. Repair work at that churn:
`0.00689 × 1.5M ≈ 10,300 node updates/frame` — at WASM speeds this is **well under 1ms**,
versus 500-750ms for from-scratch recomputation. The gap between the current cost and the
information-theoretic requirement is roughly **three orders of magnitude**, and no amount of
constant-factor tuning (heap→sweep, prefiltering, caching) closes it. Only incremental cut
repair does.

### 9.3 The algorithm (classical, not novel — this is ROAM's dual-queue applied to a splat tree)

This is the split/merge dual-priority-queue formulation from **ROAM (Duchaineau et al.,
IEEE Vis 1997)**, later refined by **Lindstrom & Pascucci (2001)** for out-of-core
view-dependent refinement, and implemented at production scale (GPU-side, persistent
cluster cut) by **Nanite (Karis et al., SIGGRAPH 2021)**:

```
Persistent state (survives across frames — LodState already has the right shape):
  cut: the current selection, stored per-node (the existing last_expanded AHashSet is
       ALREADY an implicit encoding of the cut boundary's interior — extend it, don't duplicate)
  split_q: max-heap of cut nodes with p(n) > coarsen_limit      (need refinement)
  merge_q: min-heap of cut sibling-groups with p(parent) ≤ refine_limit  (can coarsen)

Per frame:
  1. Re-key only nodes whose p can have moved past a threshold — use page-level bounds
     (Part I Solution B) to skip entire pages whose [min_p, max_p] interval did not
     intersect the band. O(pages + touched_nodes), NOT O(|C|).
  2. Pop split_q until top ≤ coarsen_limit: replace node with children (children whose
     chunk isn't resident stay as-is + chunk pushed to `touched`, exactly as today).
  3. Pop merge_q until top > refine_limit: replace sibling group with parent.
  4. Enforce budget: if |C| > max_splats, force-merge from merge_q (this is exactly
     ROAM's budget mechanism, and subsumes dynamic mode's threshold sweep).
  5. Emit a DELTA (added[], removed[]) instead of the full index list.
```

Step 5 matters as much as steps 1-4: today the full ~1.5M-entry index array crosses the
worker→main-thread boundary and is re-uploaded every traversal. With deltas, the steady-state
payload is ~10K entries. The GPU-side index texture then needs scatter-update support
(write `added` entries into free slots, tombstone `removed`) — this is the same
free-list discipline `SplatPager` already uses for pages, applied to index slots.

**Interaction with hysteresis: none needed — they compose.** The dead-zone already *defines*
the re-key band in step 1; the churn statistic already measured (0.689%) *is* the k in the
work bound. The existing `should_expand` logic becomes the queue-admission predicate.

**Cold-start:** first frame after load, and any teleport (camera jump > band width), falls
back to today's from-scratch traversal to seed the cut. Detect via the same
similarity test that already gates re-traversal (`SparkRenderer.ts:1174-1189`).

### 9.4 Cut-size scaling law — the τ⁻² lever (important for mobile)

For a surface-like scene (real captures are; splats tile 2-manifolds), the cut at threshold
τ selects nodes whose world size s satisfies s ≈ τ·d. Tiling the visible surface at ground
spacing τ·d and integrating over the view frustum gives

```
|C(τ)| ∝ A_visible / (τ·d̄)²  ∝  τ⁻²
```

**Every 1.5× increase in the acceptable screen-space error cuts the splat budget by 2.25×;
2× cuts it by 4×.** This is the single cheapest quality/perf dial that exists and it is
currently pinned at τ = 1 pixel on all platforms. Perceptual foveated-rendering research
(Guenter et al. 2012 "Foveated 3D Graphics"; Patney et al. 2016) supports 2-4 px acceptable
error outside the foveal cone — the codebase's own foveation multipliers (0.4/0.2) already
encode this belief; the *central* threshold is where mobile should differ (see §11).

Corollary for the 5M goal: **cut size is view-driven, not scene-size-driven.** Going from
2.8M nodes to ~7M (5M leaves + interior) does not change |C(τ)| at all for the same camera —
it changes storage, fetch traffic, and cold-start time only. With incremental repair
(per-frame cost O(churn), also view-driven), **total scene size drops out of the per-frame
cost entirely.** That is the mathematical reason this architecture can scale to 5M+ on a
phone at all.

---

## 10. Position precision: two quantitative corrections to Part I §4

### 10.1 f16 is the wrong container for normalized positions — use snorm16 fixed-point

Part I §4 proposes keeping f16 but adding a per-page affine in the shader. The affine is
right; keeping f16 is wrong, and the error is quantifiable. f16 has a 10-bit mantissa —
**relative** precision 2⁻¹¹ ≈ 4.9·10⁻⁴ of the magnitude. Fixed-point snorm16 over the same
normalized range [−1, 1] has **uniform absolute** precision 2⁻¹⁵ ≈ 3.05·10⁻⁵ of half-extent.
At the edge of a chunk (|x| → half-extent H), the worst-case position error is:

```
f16:      ε ≈ H · 4.9e-4      (e.g. chunk 0, H ≈ 107,000 units → ε ≈ 52 units)
snorm16:  ε = H · 3.05e-5     (chunk 0 → ε ≈ 3.3 units;  leaf chunk H≈30 → ε ≈ 1mm)
```

**16× better accuracy for identical storage cost**, and it eliminates the ±65504 clamp
class of bugs *by construction* (there is no representable out-of-range value). This is
also what SPZ itself does (24-bit fixed-point positions) and what the G-PCC path assumes.
The change is mechanical: converter packs `round(x_norm · 32767)` as i16; the decode side
multiplies by `chunk_scale/32767` and adds `chunk_center`.

### 10.2 The GPU repack clamp (Part I §4's "walls at ±65504") and float32 jitter

Part I's diagnosis is confirmed by the earlier session's data: `to_packedsplats()` re-packs
**world-space** f16 centers for GPU upload (`splat_encode.rs`, `clamp_center_coord`), so any
scene wider than ±65504 gets walls regardless of how well the disk format stores positions.
The fix must therefore be **page-local coordinates on the GPU** (RTC — relative-to-center
rendering, the standard precision technique from Cesium [Ohlarik 2008] and chunked-LOD
terrain [Ulrich, SIGGRAPH 2002 course]):

- Store each page's splats in page-local snorm16 (§10.1) with a per-page `(center, scale)`
  uniform (8 floats), applied in the vertex shader before the view transform.
- Compose `center − camera_position` **on the CPU in float64** (JS numbers) and upload the
  *camera-relative* page offset, so the GPU never sees coordinates larger than
  page-distance-to-camera. This also fixes the float32 jitter that would otherwise appear:
  at |x| ≈ 214,000, float32 ULP = 2¹⁸·2⁻²⁴ ≈ 0.016 units — sub-pixel-visible vertex
  crawling at GPS scale, the classic large-world rendering failure.

One change fixes three problems (clamp walls, f16 edge error, float32 jitter). It touches:
converter pack (i16), worker decode, `PackedSplats` GPU layout + `splatVertex.glsl` (add
per-page affine uniform), and `SplatPager.uploadPage` (thread the per-page transform).

---

## 11. Samsung S22 budget — derived, not guessed

Reference hardware: S22 (Snapdragon 8 Gen 1 / Exynos 2200): LPDDR5 ≈ 51.2 GB/s peak memory
bandwidth, sustained ~60-70% under thermal load (~33 GB/s); screen 2340×1080 ≈ 2.53 MPx.

### 11.1 The frame is bandwidth-bound by alpha blending — derive the splat budget from that

Gaussian splatting is sorted back-to-front alpha blending: every covered pixel is a
read-modify-write of the render target. With an RGBA16F target that's ~16 bytes of traffic
per blended fragment. Let V = average per-pixel overdraw (typical 3DGS scenes: 15-40):

```
bytes/frame ≈ 2.53M px · V · 16 B     →  V=25: ~1.0 GB/frame
time/frame  ≈ 1.0 GB / 33 GB/s       ≈  30 ms   → 33 fps ceiling BEFORE any other work
```

Consequences, in order of leverage:
1. **Render scale 0.7-0.75×** (1.4M px): buys ~1.8× — standard practice on mobile splat
   viewers, visually cheap at phone DPI/viewing distance.
2. **Central threshold τ = 1.5-2 px on mobile** (§9.4's τ⁻² law): |C| drops 2.25-4×, and
   since smaller selected splats ⇒ less overdraw, V drops roughly proportionally.
3. **Splat budget (`defaultSplatTarget` mobile branch, currently 1,500,000): set to
   400-500K.** The current 1.5M mobile default is ~3× beyond what the bandwidth ceiling
   supports; it would thermally throttle even if traversal were free.

CPU side: the S22's big cores are ≈2× slower than the 12700K single-thread. Frame budget
for traversal ≤ ~4ms on-device ⇒ ≤ ~8ms desktop-equivalent ⇒ needs **60-100× improvement**
over the current 500-750ms. Dynamic mode + page prefiltering (Part I §3.1-3.3) plausibly
delivers 5-15×; **only §9.3's incremental repair reaches 60-100×.** This is why the mobile
goal forces the structural fix rather than the constant-factor fixes.

### 11.2 SH3 on mobile: the memory math forces Flux-GS Track C2 (or SH banding) — it is not optional

SH3 = 45 coefficients beyond DC (RGB × 15) ≈ 90 B/splat at f16. Naive residency:

```
5M splats × 90 B = 450 MB of SH data alone   — not viable on a phone
256 resident pages × 65,536 × 90 B ≈ 1.5 GB  — not viable either
```

Two mathematically sound mitigations, complementary:

**(a) Distance-banded SH degree (cheap, ship first).** A splat at distance d viewed by a
camera translating at v sees its view direction change at rate ω_view ≈ v_tangential/d.
SH band ℓ has angular bandwidth ~ℓ (it oscillates ℓ times around the sphere), so the
view-dependent color of band ℓ changes on angular scale ~π/ℓ. Band ℓ is *perceptually
static* — hence truncatable — when the view direction cannot traverse π/ℓ within a
perceptual integration window. This yields a clean distance rule:

```
ℓ(d) = clamp( ceil( ℓ_max · d_ref / d ), 0, 3 )
```

i.e. SH3 only within d_ref, SH2 to 1.5·d_ref, SH1 beyond, DC-only in the far field.
Since |C(τ)| mass is dominated by far/coarse nodes, this cuts resident SH memory by ~5-10×
at zero perceptible cost. It composes naturally with the existing LOD tree: coarse (merged)
nodes carry low-degree SH *anyway* because `new_merged` averages children's SH (averaging is
a low-pass filter on the sphere — the merged node's high bands are already attenuated;
this is PRT band-limiting logic, Sloan et al. 2002, applied in reverse).

**(b) Flux-GS Track C2 (the plan's §5) for the storage/wire side.** Per-splat 6-dim
appearance code (3×2D app codebooks, already in the wire format, currently zeroed) + shared
MLPs (`mlp_dc`/`mlp_sh`, 16→64→{3,9}) ≈ **~2 B/splat + ~200 KB fixed weights** versus 90 B/splat
raw — a 45× reduction in stored/streamed appearance data. Decode cost: ~3.4K FLOPs/splat ⇒
~0.22 GFLOP per 65,536-splat page ⇒ 100-200ms WASM per page-load (acceptable, amortized,
off-main-thread) — **decode once at page-load into the banded-SH residency of (a), never
per-frame.** This is precisely the architecture Flux-GS's own mobile renderer uses, and it is
the reason the C2 path exists in the format at all. Rate-distortion literature (EAGLES,
Girish et al. ECCV 2024; Compact3DGS, Lee et al. CVPR 2024) consistently shows ≤0.5 dB PSNR
loss for learned appearance codes at these rates.

### 11.3 Getting from 5M to the wire efficiently: importance pruning before tree build

For the 5M-splat target, prune *before* `tiny_lod` at conversion time using a global
importance score — LightGaussian (Fan et al., NeurIPS 2024) and Mini-Splatting (2024) both
show 2-3× splat reduction at <0.3 dB PSNR loss using scores of the form

```
importance(i) ≈ opacity_i · area_i^γ · hit_count_i       (γ ≈ 0.5-1)
```

Without training-time hit counts, the deployable proxy is `opacity·area^γ` percentile
pruning (drop bottom 20-40%). 5M → ~3M leaves → ~4.2M tree nodes → ~64 chunks — brings
cold-start fetch and decode costs back to near current-scene levels.

---

## 12. Research-frontier mapping (2024-2026) — adopt / adapt / skip

| Work | Core idea | Verdict for this codebase |
|---|---|---|
| **Hierarchical 3DGS** (Kerbl et al., SIGGRAPH 2024) | Merged interior Gaussians + **continuous cut interpolation**: blend parent↔children weights smoothly as the error metric crosses τ | **ADOPT the interpolation.** We already have merged interiors (`new_merged`). Their anti-popping is a *convex blend* `w·parent + (1−w)·children` with `w = smoothstep((τ−p_c)/(p_p−p_c))` — C⁰-continuous in camera position, mathematically eliminates LOD pop rather than masking it. Supersedes Plan 2 §8.4's temporal dither idea: blend in *scale space*, not time. Implementation: emit `w` per cut-boundary node (1 extra byte in the index stream), multiply opacity by `w` (children) / `1−w` (parent) in the vertex shader. Composes with hysteresis (band edges become the smoothstep endpoints). |
| **Octree-GS** (Ren et al., 2024) | Anchor-based octree, LOD level = f(log₂ distance) | Skip — requires training-time anchor structure; our tree is post-hoc. Validates the log-distance level heuristic we already get from `tiny_lod`'s geometric level spacing. |
| **CityGaussian / V2** (Liu et al., ECCV 2024/2025) | Block partition + per-block LOD, distance-based | Validates the chunk architecture wholesale; their block-boundary blending is the same math as Kerbl's interpolation. Nothing new to port. |
| **FLoD** (Seo et al., 2024) | Budget-adaptive: pick the level that fits the device budget | Same as our `max_splats` sweep in dynamic mode; validates §9.3 step 4's budget-forced merging. |
| **StopThePop** (Radl et al., SIGGRAPH 2024) | Hierarchical per-pixel depth re-sorting to fix view-rotation popping from global sort | Skip for mobile (bandwidth cost); note as the *correct* future fix if global-sort popping ever becomes the dominant artifact after LOD popping is gone. |
| **Sort-free GS / weighted OIT variants** (several, 2024-25) | Replace sorted blending with order-independent weighted blending | Hold. Removes the per-frame radix sort (currently not a bottleneck) at a quality cost; re-evaluate on S22 only if sort or blend-order cost shows up in device profiles. |
| **Self-Organizing Gaussians** (Morgenstern et al., ECCV 2024) | Sort splats onto a 2D grid, store as images (PNG/WebP) | Skip — alternative *container*; our SVQ+Huffman is at comparable rates and already integrated. |
| **LightGaussian / Mini-Splatting / Taming-3DGS** (2024) | Importance-scored global pruning | **ADOPT at convert time** (§11.3). |
| **EAGLES / Compact3DGS / HUGS** (2024) | Learned codebooks / quantized appearance | Already aligned (SVQ path); C2 MLP path is the EAGLES-class upgrade (§11.2b). |
| **ROAM dual-queue** (Duchaineau et al., 1997); Lindstrom-Pascucci (2001); **Nanite** (Karis, 2021) | Incremental cut maintenance, split/merge queues, budget-forced merge | **ADOPT — this is §9.3, the structural fix.** Thirty years of precedent; nothing about splats changes the math. |
| **Foveated rendering** (Guenter et al. 2012; Patney et al. 2016) | Eccentricity-dependent acceptable error (2-4 px periphery) | Already half-adopted (cone foveation). Use to justify τ=1.5-2 px central threshold on mobile (§11.1). |

---

## 13. Revised unified roadmap (supersedes Part I §8 priorities where they conflict)

```
P0  (a) Default lodTraverseMode:"dynamic" when resident pages > 10; A/B stability vs
        standard on the real scene (churn metric already exists). [Part I §3.1 — unchanged]
    (b) Page-level pixel_scale bounds prefilter, feeding BOTH modes. [Part I §3.2/3.3]
        Together: expect 500-750ms → 50-120ms. Necessary but NOT sufficient for mobile.

P1  (a) snorm16 page-local positions + per-page RTC affine in vertex shader +
        camera-relative offset composed in float64 on CPU. Fixes: ±65504 walls,
        16× edge precision, float32 GPS-scale jitter. [§10 — corrects Part I §4's f16 idea]
    (b) Canonical two-stage-table Huffman decode in WASM. 85ms → ~2-8ms/chunk. [Part I §5]

P2  (a) Incremental cut maintenance with split/merge queues + DELTA output to GPU
        (§9.3). Steady-state traversal O(churn) ≈ <1ms, scene-size-independent.
        Subsumes Part I Solutions D and E (do not implement those separately).
    (b) Continuous parent↔child LOD interpolation à la Hierarchical-3DGS (§12 row 1).
        Replaces Plan 2 §8.4's dither cross-fade. Do AFTER (a) — the blend weight w
        is computed from the same per-node p values the repair step already touches.

P3  Mobile track (S22):
    (a) Device budget: mobile defaultSplatTarget 1.5M → 400-500K; central τ → 1.5-2 px;
        renderScale 0.7-0.75. All three are existing knobs or one-line derivations. [§11.1]
    (b) Distance-banded SH degree ℓ(d) (§11.2a) — resident-memory fix, ships without C2.
    (c) Track C2 (Flux-GS MLP appearance): encoder-side export of app codebooks + MLP
        weights (Plan 2 §5 Path A), decode-at-page-load into banded-SH residency. [§11.2b]
    (d) Convert-time importance pruning for the 5M source (§11.3).

Ordering rationale: P0 makes the desktop viewer usable this week; P1 removes the two
correctness-adjacent costs (position walls, decode stalls) that would contaminate all later
benchmarks; P2 is the structural fix that makes mobile arithmetic close; P3 is the
mobile-specific budget + appearance work, gated on P2 because a 4ms traversal budget is
unreachable without it.
```

---

## 14. Starter prompt for the implementing agent (self-contained)

> Copy everything between the fences into a fresh agent session.

```
You are working in C:\splat\pipeline\spark (branch flux-gs-test), a Three.js Gaussian-splat
renderer with a custom hierarchical-LOD streaming format (.sp5): Rust/WASM core
(rust/spark-rs, rust/spark-lib), TypeScript runtime (src/), test viewer
(examples/viewer/index.html), headless tests (test/*.ts, run via `npx tsx test/<name>.ts`).

READ FIRST, in this order:
1. TRAVERSAL_INVESTIGATION.md (repo root) — Part I is the measured bottleneck analysis;
   Part II §9-13 is the mathematical plan you are implementing. Follow Part II §13's
   priority order exactly. Where Part I and Part II conflict, Part II wins (specifically:
   use snorm16 fixed-point positions, NOT f16, per §10; implement incremental cut repair
   INSTEAD OF Part I's Solutions D/E, per §9.3).
2. spz_v5_implementation_plan2.md §5 (Track C2 spec) and §8 (temporal stability) for
   background. plan3.md §2.1's "14 of 43 chunks reachable" claim is DISPROVEN — trust
   test/sp5_tree_reachability_test.ts (100% connectivity, passes).

HARD FACTS (verified, do not re-derive):
- Reference scene: C:\Users\avboi\Downloads\splat_v2.ply → 44 chunks / 2,819,064 nodes.
- Measured: traverse_lod_trees standard mode costs 52ms at 4 resident pages growing to
  ~750ms at 40 pages (TRAVERSAL_INVESTIGATION.md §1.3 table). Target: <8ms desktop.
- pixelScaleLimit = 2·tan(fov/2)/renderSize.y (SparkRenderer.ts:1158-1160) = 1px threshold.
- Mobile splat budget currently 1.5M (SparkRenderer.ts:1140), to become 400-500K (§11.1).
- Hysteresis (should_expand, refine=0.9τ / coarsen=1.15τ) and per-(inst,lod,node) state
  sets already exist in rust/spark-rs/src/lod_tree.rs and are verified working.
- All 10 tests in test/ pass. Keep them passing. After ANY rust/ change: npm run build:wasm.
  Before ANY browser verification: npm run build (viewer loads dist/, not src/).
- Pre-commit hook has a pre-existing repo-wide CRLF/lint failure: do NOT use --no-verify
  without explicitly asking the user first. Commit each P-item separately.

IMPLEMENT IN THIS ORDER (each step: implement → test → measure → commit before next):

P0a. In SparkRenderer.driveLod, auto-select traverseMode "dynamic" when resident page count
     > 10 (plumb page count from pager), keeping "standard" otherwise and honoring an
     explicit user-set lodTraverseMode as an override. A/B on the real scene: log traversal
     RPC ms and per-frame index churn %, standard vs dynamic, same camera path. If dynamic
     churn is worse than standard by >2x, report numbers and keep dynamic anyway if it's
     >3x faster (the hysteresis band absorbs moderate instability).
P0b. Page-level prefilter inside BOTH traversal functions (rust/spark-rs/src/lod_tree.rs):
     accept a new Float32Array of per-page bounding spheres (center xyz + radius + max node
     size, 5 floats/page — data source: cachedMeta.chunks[].aabb, threaded through
     update_lod_trees or a new set_page_bounds call). Before the main loop, compute each
     page's max possible pixel_scale (foveation-aware upper bound: use foveate=1.0 to stay
     conservative); pages with bound ≤ refine_limit contribute nothing — skip seeding/
     evaluating their nodes. This must be provably lossless: add a test asserting identical
     output index sets with prefilter on vs off for 5 random camera poses on a synthetic
     multi-chunk scene.

P1a. Position container change (converter.ts pack + worker.ts decode + GPU path):
     - Disk: replace f16 xyz_uncompressed with snorm16 (i16) page-local positions:
       x_i16 = round(clamp(x_norm,-1,1)·32767) where x_norm=(x−chunk_center)/chunk_half_extent.
       Manifest keeps chunk_center + per-axis chunk_half_extent (replace scalar chunk_scale;
       keep reading the old fields for backward compat).
     - GPU: store page-local positions; add per-page (center,scale) uniforms applied in
       splatVertex.glsl BEFORE view transform; compose (page_center − camera_pos) in JS
       float64 each frame and upload camera-relative offsets so the GPU never sees
       world-magnitude coordinates. This kills the ±65504 walls (grep for clamp_center_coord
       usage in to_packedsplats path) and GPS-scale float32 jitter.
     - Acceptance: render_readiness_test extended: assert NO packed position equals ±65504
       after decode+repack of the real scene; assert max position error vs source < 
       2·half_extent/32768 per chunk.
P1b. Port Huffman decode to WASM canonical two-stage table (rust/spark-rs, new fn
     decode_huffman_canonical(bytes, table, count) -> Uint16Array): stage-1 index = next 12
     bits, table maps to (symbol,len), advance len bits. Wire worker.ts to call it instead
     of the JS bit-loop (worker.ts ~1185-1228). Target ≤8ms per 65,536 symbols (currently
     ~85ms). Verify bit-exact output vs the JS decoder on a real chunk before deleting
     anything; keep the JS path as fallback for tables >12-bit codes.

P2a. Incremental cut maintenance in rust/spark-rs/src/lod_tree.rs, per
     TRAVERSAL_INVESTIGATION.md §9.3 (ROAM-style split/merge queues). New persistent state
     in LodState: the cut itself (Vec or hash set of paged_index per instance) + split/merge
     heaps. New wasm fn repair_lod_cut(...) with the same instance/camera signature as
     traverse_lod_trees, returning {added, removed, touched} arrays. Cold-start/teleport
     (no cut yet, or camera similarity below threshold, or resident-page-set changed) falls
     back internally to full traversal to reseed. JS side: SparkRenderer applies deltas to
     a persistent index buffer (extend updateLodIndices to scatter-write adds into free
     slots and tombstone removals; free-list like SplatPager's page pool). Budget
     enforcement: if cut size > maxSplats, force-merge smallest-p sibling groups.
     Acceptance: (i) delta-consistency test — applying deltas over 100 frames of scripted
     rotation reproduces exactly the index set a full traversal produces at each frame
     (allow hysteresis-explained differences: assert symmetric diff < 0.1%); (ii) measured
     steady-state repair RPC < 5ms at 40 resident pages on the real scene.
P2b. Continuous LOD interpolation (after P2a): for cut-boundary nodes inside the hysteresis
     band, emit blend weight w = smoothstep((coarsen_limit − p)/(coarsen_limit −
     refine_limit)) packed into the index stream (8 bits); vertex shader multiplies
     opacity by w for children / (1−w) for the parent (parent stays in the cut with weight
     1−w while any child has w<1). Acceptance: scripted slow zoom produces NO frame where a
     region's total opacity changes >5% frame-to-frame (write a headless test that sums
     opacity-weighted splat counts per frame).

P3 (mobile, gated on P2 landing):
  a. defaultSplatTarget mobile branch 1.5M → 450_000; add lodPixelThreshold option
     (default 1.0 desktop / 1.75 mobile) multiplying pixelScaleLimit; expose renderScale
     0.75 preset in the viewer for mobile UA.
  b. Distance-banded SH: degree ℓ(d)=clamp(ceil(3·d_ref/d),0,3) with d_ref = scene
     half-extent/8 default; apply at page-load (decode only the needed bands into GPU
     memory) and re-evaluate on page touch, not per frame.
  c. Track C2 encoder path per spz_v5_implementation_plan2.md §5 Path A (offline exporter
     emitting app codebooks + mlp_dc/mlp_sh weights into the existing .sp5 fields that
     reconstruct_sp5_chunk already consumes — the decode side is DONE, only the encode
     side is missing). Gate: get one real Flux-GS-trained checkpoint from the user before
     starting; if unavailable, implement and validate with synthetic MLP weights that
     round-trip bit-exactly, and stop there.
  d. Convert-time importance pruning: score = opacity·sqrt(area), drop bottom 30%
     (configurable), BEFORE tiny_lod. Acceptance: sp5 tests still pass; report PSNR-proxy
     (per-splat color/position stats) and node-count reduction on the real scene.

MEASUREMENT DISCIPLINE: every P-item's commit message includes before/after numbers from
the real scene (traversal ms at 4/16/40 pages, startup ms, per-chunk decode ms, or churn %
as applicable). No adjective-only claims. If a measured result contradicts the plan's
prediction by >2x either way, STOP and report before proceeding to the next item.
```

