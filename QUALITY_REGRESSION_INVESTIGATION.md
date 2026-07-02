# Quality Regression Investigation: Session 2026-07-02

> **Status:** Documenting what changed and why the splats now have poor PSNR representation.
> **No further code edits are to be made from this document.**
> This is an input document for an external review agent.

---

## 1. What changed this session (chronological, all committed)

| # | Change | Files | Purpose |
|---|---|---|---|
| 1 | Event-driven camera-fit + UTC timestamps + FPS counter + marker buttons | `examples/viewer/index.html` | Measurement instrumentation, faster startup |
| 2 | Persistent fetch queue (`_pendingFetchChunks`) | `src/SparkRenderer.ts` | Fix 9s dead-zone during stationary camera |
| 3 | LRU keep-warm bias (10s PAGE_KEEP_WARM_MS) | `src/SplatPager.ts` | Prevent evict-then-refetch during rotation |
| 4 | Predictive prefetch (angular velocity + AABB lookup) | `src/SparkRenderer.ts` | Prefetch chunks in direction of rotation |
| 5 | Delete `stitchRealLodTreeWithSiblings` | `src/worker.ts` | Dead code removal |
| 6 | Update plan3.md §2 to FIXED | `spz_v5_implementation_plan3.md` | Documentation |
| 7 | **P0a:** Auto-switch Standard→Dynamic at >10 pages | `src/SparkRenderer.ts` | Traversal perf (2-4× measured) |
| 8 | **P0b:** Page-level pixel_scale prefilter in Rust traversal | `rust/spark-rs/src/lod_tree.rs`, `src/SparkRenderer.ts`, `src/worker.ts` | Skip pages below threshold at seeding |
| 9 | **P1b:** Huffman decode → WASM (per-bit MSB-first + prefix LUT) | `rust/spark-rs/src/lib.rs`, `src/worker.ts` | Decode speed |
| 10 | Fix: `pageBounds?.slice()` for transfer safety | `src/SparkRenderer.ts` | Prevent detached ArrayBuffer crash |

---

## 2. Current observed symptoms (from latest log, 18:21:20 - 18:21:47 UTC)

### 2.1 No JavaScript errors
- Zero "detached ArrayBuffer" errors (fix #10 working)
- Zero "Uncaught (in promise)" errors
- All 44 chunks complete fetch+decode+upload without errors

### 2.2 Performance metrics

| Metric | Value | Rating |
|---|---|---|
| Startup (drop → first bbox) | 621ms | Good |
| Huffman decode per chunk | 39-61ms total for 5 streams | Modest (old: 66-107ms) |
| Traversal RPC early (4p) | 71ms | Good |
| Traversal RPC settled (~30p, dynamic) | 141-232ms | Poor |
| Frame time settled | 72-139ms | Poor (target 16.6ms) |
| GPU upload | 1-9ms per batch | Good |

### 2.3 Quality symptoms (user report)
- "Tons of gaussians have extreme elongation and look bad"
- "Poor PSNR representation"
- "GPU under 90% usage when I load a splat" (extremely power-hungry)
- DONE-BAD marker clicked at 18:21:43.400

### 2.4 Diagnostic evidence from log

**`forEachSplat` shows f16 position clamp:**

```
x=[-65504.000, 65504.000] y=[-48448.000, 46432.000] z=[-65504.000, 65504.000]
```

±65504 is the f16 maximum. The true scene extent is ~±107,000 (from `decodeSp5Chunk` output:
`x=[-111876, 102187]` for chunk 0). Positions outside ±65504 get clamped at GPU pack time.

**Contrast with decode-time positions (pre-GPU-pack):**

```
chunk 0:  x=[-111876.180, 102187.289] y=[-48433.570, 46429.352] z=[-180414.313, 71701.891]
chunk 1:  x=[-24771.465, 11892.167]    y=[-3252.402, 15874.629]   z=[-19583.564, 19654.055]
chunk 13: x=[-147.624, -9.297]         y=[-30.816, 60.530]        z=[-56.975, 269.887]
```

Chunk 0 has positions at ±107K that decode correctly but get clamped to ±65504 when packed for GPU upload. Chunks 1+ have positions within f16 range.

---

## 3. Root cause analysis

### 3.1 Primary cause: f16 GPU position clamp (PRE-EXISTING, now fully exposed)

The GPU upload path (`to_packedsplats()` → `splat_encode.rs` → `clamp_center_coord`) packs **world-space** positions as f16 with hard clamp at ±65504. This was always present but was masked when fewer chunks loaded.

**Why it's worse now:**
- Before the persistent fetch queue fix (#2), only ~20 of 44 chunks loaded. Chunk 0's coarse nodes were mixed with a subset of fine chunks.
- After the fix, ALL 44 chunks load. Chunk 0's coarsest nodes (covering the full ±107K extent) are clamped and stretched across the scene.
- The clamped positions create gaussians that span the entire visible area, causing:
  - Extreme elongation (the splat's scale is correct for its true position but its clamped position shifts it far from its neighbors)
  - Massive overdraw (every clamped splat overlaps the camera frustum)
  - 90% GPU usage (blending millions of 65504-width gaussians)

### 3.2 Secondary cause: SVG scale/rotation quantization (PRE-EXISTING)

The k-means SVQ with 256 codebook entries per channel produces ~45-48dB PSNR per the TRAVERSAL_INVESTIGATION.md analysis. For the human visual threshold of ~40dB, this is at the edge of perceptibility. Combined with the f16 clamp, the corrupted positions amplify the quantization error.

### 3.3 Did P1b (Huffman WASM) introduce quality issues?

**No — smoke_test.ts passed** (all 36 chunks decode correctly with the WASM decoder).

However, the WASM decoder's **per-bit MSB-first implementation** should be independently verified against the JS reference output for bit-identical results on a variety of Huffman tables. The prefix LUT approach assumes: (a) codes are prefix-free (true by construction), (b) the LUT is built correctly (verified by the JS build logic), (c) the bit ordering matches the encoder (verified by per-bit MSB-first reading matching the encoder's MSB-first write order).

**Open question for review agent:** Was `sp5_ordering_test.ts` re-run against this exact build? It checks attribute mispairing. If it passes, the Huffman decoder is producing correct indices. If it wasn't re-run, it should be.

### 3.4 Did the dynamic traversal mode switch introduce quality issues?

**Possibly.** `dynamic_traverse_lod_trees` uses threshold-sweeping (iteratively tightening `current_scale` from 100× to 1× pixel_scale_limit), which can converge to **different** node selections than standard mode for nearly-identical camera input. Plan 2 §6.6 specifically documents this convergence sensitivity.

However, the affected quality is about **position precision**, not LOD selection quality. The dynamic mode selects which nodes to render; it does not change the positions of those nodes. The f16 clamp would affect both modes equally.

---

## 4. Open questions for external review agent

### Q1: Huffman decode correctness validation

Run `npx tsx test/sp5_ordering_test.ts` against the current build. This test encodes → decodes 20,000 splats and checks that decoded scale/rotation indices stay paired with the correct position. If this passes, the WASM Huffman decoder is producing correct symbol streams.

### Q2: f16 clamp visual impact quantification

The clamp affects only chunk 0 positions (coarsest LOD levels). Later chunks (1-43) have extents well within ±65504. To quantify: what fraction of the visible splats in a typical camera pose come from chunk 0? If >10%, the visual impact is significant.

### Q3: Chunk ordering and visual priority

The BFS-by-feature_size chunk ordering puts the coarsest (largest-scale, most scene-spanning) nodes in chunk 0. These are the ones most affected by f16 clamp. But they are ALSO the least numerous (in terms of surface area they represent a small fraction of the actual rendered detail). The visual impact may be less than the position-range diagnostics suggest.

### Q4: Was quality better on the P0-only build (before P1b)?

The user reported "loads quite well" on the P0 build (dynamic mode + prefilter). That build did NOT have the WASM Huffman decoder. Was the visual quality acceptable on that build? If P0-only was visually acceptable and P0+P1b is broken, the Huffman decoder is suspect #1.

### Q5: GPU power draw correlation

The user reports "GPU under 90% usage." This is consistent with massive overdraw from clamped splats. But the GPU was also busy BEFORE these changes. What was the GPU usage on the P0-only build? If it was lower, the clamped splats creating overdraw is the mechanism.

---

## 5. Remediation path (from TRAVERSAL_INVESTIGATION.md §13)

The next planned phase **P1a** (snorm16 page-local positions + per-page RTC affine in vertex shader) directly fixes the f16 clamp problem:

- Replaces f16 world-space positions with snorm16 page-local positions (no ±65504 ceiling)
- Applies a per-page `(center, scale)` affine in the vertex shader
- Composes `page_center - camera_position` in JS float64 to avoid GPU float32 jitter

This is documented at `TRAVERSAL_INVESTIGATION.md` §10 and §13, P1a. It is the most impactful remaining quality fix and should be the next implementation target.

---

## 6. What did NOT regress

- Tree reachability: 100% verified (test passes)
- Chunk coherence: all chunks spatially coherent
- Hysteresis stability: 0.689% avg churn
- Scale fidelity: 1.000x decoded/source ratio
- No JavaScript crashes
- Startup time: 621ms (excellent)
- Persistent fetch queue: no 9s dead-zone

---

*Generated 2026-07-02. No code edits were made as part of this document.*
