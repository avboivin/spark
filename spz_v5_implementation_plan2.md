# SPZ-v5 (.sp5) Implementation Plan 2 — From "It Loads" to Shippable

> **Scope of this document.** Plan 1 (`spz_v5_implementation_plan.md`) got the codec from
> "nothing renders" to "the branch produces a viewable scene" — that work is done; see §1 for the
> changelog. This document picks up from there: what's still wrong, exactly where it lives in the
> code, and the concrete steps to (a) finish Track C2 (real SVQ+MLP neural quantization, currently
> 100% unreachable from the JS encoder despite being fully implemented on the Rust decode side),
> (b) replace the current flat/binary chunk-loading model with genuine multi-level, distance-aware
> LOD suited to GPS-scale outdoor captures (§3-4), and (c) fix why LOD visibly pops during normal
> use — especially while rotating the camera — which turns out to be a set of concrete, verifiable
> gaps, not a vague quality problem (§6, a deliberately harsh, evidence-only review; §8, the fix).
> Every file:line citation below was verified by direct read against the current `flux-gs-test`
> branch as of this writing — line numbers will drift as the branch is edited further; re-verify
> before trusting a citation, the same way this document had to re-verify plan 1's.

---

## 1. Where the branch is now

### 1.1 What plan 1 shipped (working, verified)

All of the following were fixed and verified this session (headless tests where noted; see
`test/` for the scripts) and are live in the current build:

| Bug | Fix location | Verification |
|---|---|---|
| f16 position overflow on outlier splats → `Infinity` | `rust/spark-lib/src/splat_encode.rs`, `clamp_center_coord` | `test/render_readiness_test.ts` |
| min/max-midpoint centering dragged by outliers (two separate instances) | `src/converter.ts` (median), `rust/spark-rs/src/lib.rs` `GsplatArray::center()` (median) | manual + numeric |
| Camera far-plane fixed at 1000, scene rendered beyond it | `examples/viewer/index.html` `fitCameraToBoundingBox` | manual |
| `SparkControls` flat-speed zoom (unusable at real-world scale) | `examples/viewer/index.html` render loop, distance-scaled speeds | manual |
| Camera framed on percentile box dominated by sparse background | `examples/viewer/index.html` `computeDensityBox` (shortest-interval-50%) | numeric |
| Attribute misalignment: scale/rotation/app indexed by loop position `i`, position/opacity/dc/sh indexed by decoder's own re-sort `idx` | `rust/spark-rs/src/lib.rs` `reconstruct_sp5_chunk` | `test/sp5_ordering_test.ts` (45%→0% mismatch) |
| Chunking via one global 1D sort → paper-thin degenerate slab chunks | `src/converter.ts` `kdPartitionIndices` (k-d tree spatial partition) | `test/sp5_chunk_coherence_test.ts` |
| Fetch priority only ever requested chunk 0 + tree-discovered chunks | `src/SparkRenderer.ts` (unconditional full-manifest fetch for SP5), `src/SplatPager.ts` `cachedMeta` | manual |
| Each chunk's synthesized LOD tree was a fully isolated island (zero cross-chunk edges) — only the chunk assigned the root page was ever renderable | `src/worker.ts` `synthesizeFlatLodTreeIfMissing` (sibling-pointer stitching) | `test/sp5_cross_chunk_test.ts` (real WASM traversal, both phases) |

### 1.2 What's still wrong (this document's subject)

Reported directly against the fixed build, on a real ~2M-splat GPS-referenced two-story-building
capture, converted with the current `convertSplatToSp5Client`:

1. **Chunk pop-in looks incoherent** — the building visibly splits into a handful of large,
   independently-arriving blocks, not a coherent progressive refinement.
2. **Blurry / no LOD refinement on approach** — moving the camera doesn't sharpen anything; the
   scene looks equally (low) detailed near and far.
3. **No distance-based LOD tuned for GPS-scale outdoor captures** — the goal is a heuristic that
   knows about real camera-capture characteristics (ground sampling distance, outdoor scale), not
   an arbitrary constant tuned for small/synthetic scenes.

§3 below traces exactly why, in the current code, all three are actually **one root cause**.

---

## 2. Architecture fact that changes the whole plan

**The codebase already has real, working, multi-resolution LOD-tree-building algorithms.** They
are not missing — they are simply not wired into the SP5 (Track C) pipeline. This was flagged as
deliberately out of scope in plan 1 (`spz_v5_implementation_plan.md:453-459,970-976`, quoting
`quick_lod.rs`'s merge behavior and explicitly deferring "multi-resolution decimation for the
neural codec" as future work) — this document is that future work.

| File | Status | What it does |
|---|---|---|
| `rust/spark-lib/src/tiny_lod.rs` | **Active**, called from `rust/spark-rs/src/lib.rs:253` | `pub fn compute_lod_tree<SA: TsplatArray>(splats: &mut SA, lod_base: f32, merge_filter: bool, logger: impl Fn(&str))` (line 10). Builds a real hierarchy: cells with >1 splat get replaced by one merged representative (`splats.new_merged(&indices, merge_step)`, line 98) that becomes a parent node one level up. **Genuine simplification, not bucketing.** |
| `rust/spark-lib/src/quick_lod.rs` | Exists, **not called** from the live web/worker pipeline (only referenced commented-out at `rust/spark-rs/src/lib.rs:244-247`; live call site is the offline CLI `rust/build-lod/src/main.rs`) | Same shape: `pub fn compute_lod_tree(splats: &mut GsplatArray, lod_base: f32, merge_filter: bool, logger: impl Fn(&str))` (line 11), merges via `new_merged` (line 88). |
| `rust/spark-lib/src/bhatt_lod.rs` | **Active**, called from `rust/spark-rs/src/lib.rs:262` | `pub fn compute_lod_tree<TA: TsplatArray>(splats: &mut TA, lod_base: f32, logger: impl Fn(&str))` (line 12). Greedy nearest-neighbor merge driven by Bhattacharyya similarity (`splats.similarity`, line 85; merge at line 98). |
| `rust/spark-lib/src/chunk_tree.rs` | **Active**, called from `rust/spark-rs/src/lib.rs:255,264` | `pub fn chunk_tree<TA: TsplatArray>(splats: &mut TA, root: usize, logger: impl Fn(&str))` (line 653) → delegates to `chunk_tree_size` (line 654/408). **Does not merge anything itself** — it re-batches an *already-hierarchical* tree (built by one of the three above) into ~64K-splat disk-friendly pages via priority-queue BFS + octant/longest-axis splitting for spatial locality. Read this as "the packer," not "the LOD builder." |

All four coarsening functions share the underlying merge primitive: `TsplatArray::new_merged`
(trait defined `rust/spark-lib/src/tsplat.rs:77`, implemented for `GsplatArray` at
`rust/spark-lib/src/gsplat.rs:293` and `CsplatArray` at `rust/spark-lib/src/csplat.rs:203`) — a
weighted-average center/color/opacity and merged covariance. This is real multi-resolution
simplification machinery, already battle-tested (it's what makes `.ply`/`.spz` drop-loading with
real hierarchical LOD work at all — see §3.2).

**The wire format for carrying this already exists too.** `LodSplat` (`rust/spark-rs/src/lod_tree.rs:95-100`):

```rust
struct LodSplat {
    center: [f16; 3],
    size: f16,
    child_start: u32,
    child_count: u16,
}
```

and `get_lod_tree_level` (`rust/spark-rs/src/lod_tree.rs:372-409`) confirms the traversal protocol's
implicit contract: coarse-vs-fine is distinguished purely by `size` shrinking geometrically with
depth —

```rust
378: let root_size = splats[0].size();
379: let level_size = root_size / (1.25f32.powi(level as i32));
389: if splat.size() <= level_size { output_nodes.push(node); } else { /* recurse into children */ }
```

`compute_pixel_scale` (`rust/spark-rs/src/lod_tree.rs:601-629`, unchanged by this session) computes
`pixel_scale = splat.size() * inv_distance * lod_scale` per node — the entire distance-adaptive
selection mechanism is driven by this one `size` field being meaningfully different at each depth.

### 2.1 The actual bug: `synthesizeFlatLodTreeIfMissing` never produces this

`src/worker.ts:1443-1590` (doc comment from line 1388). For the common `count <= 65536` case
(lines 1496-1535): entry 0 is the root; **every entry `i` from 1 to `count-1`** is written via

```ts
writeEntry(lodTree, i, x, y, z, nominalSize, 0, 0);   // childCount=0, one real unmerged splat
```

— i.e. one real, unmerged splat per leaf, **all tagged with the same `nominalSize`** (the whole
chunk's own bounding-box diagonal, computed once at the top of the function), not each splat's
true individual size. There is no intermediate coarse level between "the whole chunk as one node"
and "every one of its 62,500 individual splats." This directly violates the `size`-shrinks-per-level
contract `get_lod_tree_level`/`compute_pixel_scale` assume.

### 2.2 Why this produces exactly the three reported symptoms

Trace `traverse_lod_trees` (`rust/spark-rs/src/lod_tree.rs:412-599`) against this tree shape:

- A chunk's root pixel_scale is computed from `nominalSize` (**the whole chunk's** bounding
  diagonal) — for a real building-scale chunk this is huge, so from virtually any reasonable
  viewing distance the root's pixel_scale exceeds `pixel_scale_limit` and gets expanded.
- Expanding the root pushes **all `count-1` leaves** onto the frontier at once.
- Every one of those leaves has `child_count == 0`, so the very next pop of each one is
  **unconditionally pushed to `output`** (`rust/spark-rs/src/lod_tree.rs:491-495`) — there is no
  further per-splat distance/size culling once a chunk's root has been expanded once.

Net effect: **a chunk is binary.** Once resident and expanded, 100% of its splats render at full
resolution regardless of individual distance — that's symptom **#2** (no refinement on approach:
there is nothing to refine *into*, a chunk is either fully detailed or not drawn). Chunks that
aren't yet resident render as a single placeholder point (the sibling-pointer entry added this
session, sized to the whole chunk's bounding box) — a giant, low-detail stand-in blob — which is
symptom **#3**'s "blurry" look and, combined with the coarse ~62,500-splat chunk granularity from
k-d tree partitioning, is exactly symptom **#1**'s "large bits popping in one at a time" (there's no
intermediate size between "1 giant placeholder" and "62,500 full-detail splats" for a viewer to
progressively resolve through).

### 2.3 Why plan 1's cross-chunk stitching was still the right first step

The sibling-pointer mechanism added this session (`src/worker.ts:1443-1590`,
`src/SplatPager.ts` chunk 0 `siblingChunks` plumbing) is **not obsoleted** by this plan — it's the
correct foundation. It proved (via `test/sp5_cross_chunk_test.ts` against real compiled WASM) that
the `child_start`/`chunk_to_page` addressing scheme correctly carries pointers across chunk
boundaries before a chunk's physical page is even known. Phase 1 below reuses this exact addressing
scheme; it only replaces *what a chunk's own internal tree looks like* (real coarsened hierarchy
instead of flat leaves), not the cross-chunk plumbing.

---

## 3. Phase 1 — Real hierarchical LOD for SP5 chunks

**Goal:** every chunk gets a genuine multi-level tree (root → coarse merged representatives →
progressively finer merged representatives → real leaf splats), matching the `size`-shrinks-with-
depth contract the traversal code already assumes, using the merge algorithms in §2 that already
exist and are already used elsewhere in this codebase.

### 3.1 Where the work needs to land

The natural owner is **the encoder** (`src/converter.ts`), not the decoder (`src/worker.ts`
`decodeSp5Chunk`), for one structural reason: real coarsening (`new_merged`) needs the *original,
full-precision* splat set to compute good merged representatives, and it needs to run *before*
SVQ quantization (current pipeline: k-means on raw scale/rotation *then* discard the originals) —
otherwise you'd be merging already-lossy quantized values, compounding error. Concretely:

1. **`src/converter.ts`, `convertSplatToSp5Client`** (currently ~line 289 onward, post this
   session's edits) — insert a real LOD-tree build **before** the k-means codebook step (currently
   at "2. Run k-means codebook generation") and **before** `kdPartitionIndices` spatial chunking
   (currently "3. Packaging into chunks"). This requires exposing `tiny_lod`'s (or `bhatt_lod`'s)
   coarsening to JS in a form that operates on the *client-side-decoded* attribute arrays
   (`xyz`/`opacity`/`rgb`/`scales`/`quaternions`) rather than a Rust-native `GsplatArray` built
   from a `.ply` file, since Track C1/C2 receive already-extracted `Float32Array`s. Two options:
   - **(a) Recommended.** Round-trip through `GsplatArray` in Rust: add a `wasm-bindgen` export
     (e.g. `GsplatArray::from_attributes(xyz, opacity, rgb, scales, quaternions, sh1)` mirroring the
     existing `extract_attributes()` in reverse) so `converter.ts` can build a real `GsplatArray`,
     call the *already-exposed* `.tiny_lod(lodBase, mergeFilter)` (confirmed live at
     `rust/spark-rs/src/lib.rs:249-256`, identical to what `partitionDroppedMonolithic` already
     does for the `.ply`-drop path), then extract the now-hierarchically-annotated result back out
     — reusing tested Rust code instead of reimplementing merge logic in TypeScript.
   - **(b)** Reimplement a JS/WASM-exposed simplified merge pass operating directly on the flat
     arrays. Higher risk (new code path, not the one every other loader already exercises), only
     worth it if (a)'s round-trip overhead is measured as prohibitive for 2M+-splat scenes.
2. **`rust/spark-rs/src/lib.rs`, `GsplatArray::tiny_lod`** (line 249-256) — after this call, the
   `GsplatArray`'s internal ordering **is** the hierarchy (parents before/interleaved with
   children per `chunk_tree`'s BFS batching) — but there is currently **no exposed way to read out
   `child_start`/`child_count`/hierarchy `size` per splat** from JS; `extract_attributes()`
   (referenced throughout this session, e.g. `rust/spark-rs/src/lib.rs` around the
   `ExtractedAttributes` struct) only returns flat per-point attribute arrays with no tree topology.
   **This needs a new accessor** — e.g. `GsplatArray::extract_lod_tree() -> Uint32Array` returning
   the same 4-word-per-node `LodSplat` layout `synthesizeFlatLodTreeIfMissing`/`set_lod_tree_data`
   already use, so `converter.ts` can carry it straight through to the `.sp5` wire format with no
   format translation.
3. **`.sp5` chunk wire format** (`src/converter.ts`'s `chunkMeta`/`packArray` construction, and the
   mirrored decode in `src/worker.ts`'s `decodeSp5Chunk`, magic `0x355a5053`) — currently has no
   field for tree topology at all (compare: `.spz`'s `SpzEncoder` *does* have one, gated by
   `flags |= 0x80` / `has_lod_tree()`, `rust/spark-lib/src/spz.rs:948-966` and the LoD extension
   block from line 1156 — worth studying as a precedent for the byte layout, though `.sp5`'s JSON
   header + binary-blob-with-offsets convention is more flexible and shouldn't just copy `.spz`'s
   scheme). Add a `lod_tree: {offset, length, dtype:"uint32", shape:[n,4]}` metadata entry
   (matching the existing `packArray`-style metadata convention already used for
   `xyz_uncompressed`/`mlp_dc`/etc.) carrying the real per-chunk hierarchy.
4. **`src/worker.ts`, `decodeSp5Chunk`** (currently ~line 1017 onward) — when `manifest.lod_tree`
   is present, read it directly into `result.extra.lodTree` (skip
   `synthesizeFlatLodTreeIfMissing` entirely for that chunk — it becomes the fallback for chunks
   that, for whatever reason, don't carry real hierarchy, not the primary path). The **cross-chunk
   sibling-pointer stitching added this session must still run** on top of the *real* per-chunk
   tree's root (append sibling entries to chunk 0's real root's child range, exactly as now — the
   mechanism doesn't change, only what "leaves" mean underneath it).
5. **`kdPartitionIndices`** (`src/converter.ts`, this session's addition) still determines *chunk
   boundaries* — that doesn't change. What changes is that within each spatial chunk, the encoder
   now runs step 1 above on that chunk's own point subset before quantizing/serializing it, so each
   chunk itself contains a real coarse→fine hierarchy instead of a flat point list.

### 3.2 Validate against the path that already works

Before writing new code, **directly verify** (this was flagged, not confirmed, by the research
pass behind this document): does a `.ply`-monolithic-drop chunk's hierarchy actually survive
`clone_subset()` → `to_spz()` (`src/worker.ts:939-956` `partitionDroppedMonolithic`,
`rust/spark-lib/src/spz.rs:948` `SpzEncoder::encode`)? If yes, that's a working, shippable
reference implementation of "real per-chunk hierarchy survives serialization" to copy the pattern
from, including how `flags |= 0x80` on decode re-populates `extra.lodTree` (search
`rust/spark-lib/src/spz.rs` decode path for the `0x80` flag check, and `src/worker.ts`'s `.spz`
decode branch for whether it currently reads that hierarchy back out at all — this session did not
verify that read-side exists; if it doesn't, that's the exact same gap as SP5, just on a different
file format, and fixing it once (a shared "read `LodSplat` array from decoded attrs" helper) fixes
both).

### 3.3 Concrete acceptance test for Phase 1

Extend `test/sp5_cross_chunk_test.ts`'s pattern (drives real compiled WASM traversal, not just JS
logic) with a synthetic chunk built via the new real-hierarchy path: assert `size` strictly
decreases from root to leaves across at least 3 levels, and that `traverse_lod_trees` at a
*moderate* camera distance returns a `numSplats` output strictly between "1 (root only)" and "every
leaf" — i.e. genuine intermediate-resolution output, which is structurally impossible with the
current flat tree (confirm by running the same assertion against today's `synthesizeFlatLodTreeIfMissing`
output as a regression guard — it should fail today, pass after Phase 1).

---

## 4. Phase 2 — Distance heuristics for GPS-scale outdoor captures

**Goal, per the requirement:** LOD selection that knows about *real camera-capture
characteristics* — the highest meaningful definition is bounded by the capture's actual ground
sampling distance (GSD), not by an arbitrary constant tuned for small/synthetic/tabletop scenes.

### 4.1 What's parametrized today and where

- `pixel_scale_limit` — computed in `src/SparkRenderer.ts` around the `driveLod`/`updateLodInstances`
  call site (this session read it near line 1147-1161: derived from `renderSize`, FOV, and
  `lodRenderScale` — a *screen-space* pixels-per-splat target, entirely camera/display driven, with
  **zero knowledge of the scene's physical capture resolution**).
- `get_lod_tree_level`'s `1.25f32.powi(level)` falloff (`rust/spark-rs/src/lod_tree.rs:379`) — a
  fixed geometric ratio between levels, independent of scene scale or capture density.
- `lod_base` parameter threaded through `tiny_lod`/`quick_lod`/`bhatt_lod`'s `compute_lod_tree`
  (all four share this parameter name) — controls the *merge aggressiveness* per level during tree
  construction; this is the natural knob for "how much can we simplify before it looks wrong,"
  currently a caller-supplied constant (`worker.ts:939` passes `base = clamp(lodBase ?? 1.5, 1.1, 2.0)`
  from `partitionDroppedMonolithic`'s options) with no connection to the source data's actual
  spatial point density or intended viewing distance.

### 4.2 Proposed heuristic

For GPS-scale/outdoor real-world captures, the practically meaningful "finest useful detail" is
bounded by the capture's effective GSD — points closer together than the original sensor's
resolution carry no additional real information, so collapsing them via `new_merged` costs
~nothing perceptually and saves everything computationally. Two concrete, implementable proposals,
not mutually exclusive:

1. **Data-driven `lod_base` at conversion time.** Compute the scene's actual nearest-neighbor
   point spacing (a cheap k-d-tree or grid-based statistic, similar in spirit to the median/percentile
   analysis already used this session for outlier-robust centering) as part of `convertSplatToSp5Client`,
   and derive `lod_base`/merge thresholds from it instead of a fixed constant — e.g. "don't create a
   tree level finer than 2× the scene's median nearest-neighbor spacing," which automatically
   adapts between a tabletop scan (mm-scale spacing) and a drone capture (cm-to-dm-scale spacing)
   without any user-facing configuration.
2. **Physical-units `pixel_scale_limit` override, informed by target viewing distance.** Currently
   `pixel_scale_limit` only encodes *screen* pixels; for outdoor scenes the useful control is "don't
   bother rendering detail finer than what a viewer standing at N meters away could perceive." Add
   an optional per-mesh or per-manifest "reference distance" (could be authored at conversion time
   from known capture altitude/GSD metadata, or left as a `SplatMesh`/`SparkRenderer` option a
   caller sets explicitly for outdoor scenes) that scales `pixel_scale_limit`'s effective threshold
   by real-world scale rather than assuming a scene-agnostic screen-space constant. Land this as an
   additive option (`lodRenderScale`-adjacent, not a replacement) so small/synthetic scenes using
   today's defaults are unaffected — this is an extension of the existing spec, not a redesign of
   it, matching the explicit requirement not to regress current behavior for other scene types.

### 4.3 Where this plugs in

- `rust/spark-lib/src/tiny_lod.rs`/`quick_lod.rs`/`bhatt_lod.rs`'s `compute_lod_tree` signatures
  already take `lod_base: f32` — Phase 2.1 only changes what value `converter.ts` computes and
  passes in (via the new integration point from §3.1 step 1), no Rust signature change needed.
- `pixel_scale_limit` computation in `src/SparkRenderer.ts` — Phase 2.2 needs a new input (a
  physical reference-distance scale factor) threaded from `SparkRendererOptions`/`SplatMesh` down
  to wherever `pixelScaleLimit` is finalized before the `traverseLodTrees` worker call
  (`src/SparkRenderer.ts:1460-1465` region, `pixelScaleLimit` param).

---

## 5. Phase 3 — Complete Track C2 (real SVQ+MLP neural quantization)

### 5.1 Current status: fully built on decode, 100% absent on encode

`rust/spark-rs/src/lib.rs`, `reconstruct_sp5_chunk` (signature ~line 786-806): branch point
`if mlp_cont.is_empty()` at **line 866** (SVQ-only/simple path, what Track C1 actually produces
today, lines 866-874) vs. **line 875** `} else {` — the full neural path, fully implemented:
`run_tcnn_mlp` calls at lines 892 (`MLP_cont`, contraction feature), 906 (`MLP_opacity`), 908
(`MLP_dc`), 909 (`MLP_sh`), plus `contract_to_unisphere`/`get_tcnn_frequency_encoding`
(`rust/spark-lib/src/sp5.rs`, both referenced and correctness-reasoned about in plan 1 §5's math
reference — the exact weight-count formulas for each MLP call are already derived there, e.g.
`MLP_cont`: 96→64→13 padded-TCNN, 7168 f32 weights; `MLP_opacity`/`MLP_dc`/`MLP_sh`: 16→64→{1,3,9},
2048 f32 weights each).

`src/converter.ts:584-585` (this session's line numbers, verify current):

```ts
mlp_cont: { offset: 0, length: 0, dtype: "float16", shape: [0] },
mlp_offset: {}
```

Hardcoded empty. **The entire neural decode branch is unreachable dead code from the browser
converter's output** — every `.sp5.zip` this pipeline can currently produce forces `mlp_cont.is_empty()
== true`, permanently. This is not a bug to fix so much as a feature to *build*: Track C2 as
originally scoped.

### 5.2 Two paths to close this gap — pick one, don't half-build both

**Path A (recommended): offline Python exporter, reusing Flux-GS's already-trained models.**
`C:\splat\pipeline\Flux-GS\export_spz_v5.py` (346 lines, referenced extensively in plan 1 §4E)
already has most of the *scaffolding* for this (codebook/Huffman export patterns, PLY parsing) but
plan 1 found it uses AABB voxelization for positions, contradicting the bitcast scheme this
pipeline settled on for `.sp5` xyz encoding (plan 1 §2.1/§7-i) — that specific file cannot be used
as-is, but the *trained MLP weights it would have access to* (from Flux-GS's actual training run)
are the valuable asset here. This path means: write a new offline exporter (Python, run once per
scene, not part of the browser tool) that (1) loads a trained Flux-GS checkpoint, (2) runs the SVQ
quantization pipeline already proven correct in `converter.ts` (same k-means/Huffman approach,
ported or reused via a shared spec — the *codebook/Huffman wire format* should stay identical
between C1 and C2 output so `reconstruct_sp5_chunk`'s SVQ-index-decode logic, lines through 866,
needs zero changes), and (3) additionally emits the four MLP weight blobs
(`mlp_cont`/`mlp_opacity`/`mlp_dc`/`mlp_sh`) plus `mlp_offset`'s four `Linear` layers, at the exact
element counts plan 1 §5 already computed, float16-packed the same way every other `.sp5` blob is.
**Lower risk**: no new training/autodiff code, just an export/serialization script against an
existing trained model.

**Path B: browser-side self-distillation training** (Track C2 as plan 1 literally scoped it,
`spz_v5_implementation_plan.md` — plan 1 explicitly flagged this as "genuinely novel — no
autodiff/training code exists anywhere in Spark or either reference repo today," gated behind its
own kill-criterion spike, not expected to be built without a dedicated go/no-go). If the actual
requirement is "users convert a raw `.ply` with zero server/Python dependency and still get C2
quality," this is unavoidable — but it is a materially larger, riskier effort (needs a WASM/WebGPU
autodiff path or a from-scratch backprop implementation for small MLPs, a training loop, a loss
function matched to what Flux-GS's original training optimized for) than Path A. **Do not start
this without an explicit decision that Path A's offline-exporter constraint (a Python step) is
unacceptable for the shipping requirement** — confirm this with the user before investing here;
plan 1's own kill-criterion-first structure (§Part 7) is the right model to follow: spike/validate
PSNR improvement from a Path-A-produced file first, since that validates the *decode* path and the
*value* of C2 at far lower cost, before deciding whether Path B's browser-training investment is
justified at all.

### 5.3 Acceptance

Reuse `test/sp5_scale_diagnostic.ts`'s and `test/sp5_ordering_test.ts`'s pattern (decode a real
converted file, compare statistics against ground truth) — for C2 specifically, add a PSNR/SSIM
comparison between a C1 (SVQ-only) and C2 (SVQ+MLP) conversion of the same source scene, rendered
from a fixed camera set, to quantify the "higher PSNR" goal concretely rather than asserting it
subjectively.

---

## 6. Harsh review — why LOD visibly pops today, and it isn't subtle

This section exists because a good LOD system should be invisible: the user should never catch it
in the act. This one gets caught constantly, especially while turning the camera (as opposed to
moving forward/back, which is a comparatively minor version of the same disease). Every claim below
was directly verified against the current code, not inferred.

### 6.1 The smoking gun: a hysteresis input exists, is faithfully computed, and is thrown away

`rust/spark-rs/src/lod_tree.rs:413` (`traverse_lod_trees`) and `:633`
(`dynamic_traverse_lod_trees`) both declare a parameter named `_last_pixel_limit: Option<f32>` —
the underscore is Rust's own convention for "accepted, deliberately unused." Grepping the full body
of both functions confirms it: **zero** other references to it anywhere in either function. Both
traversals decide entirely off the *current* frame's `pixel_scale_limit`, at line 483 (standard)
and lines 720/760-761 (dynamic).

Meanwhile, on the JS side, nobody got the memo that this is dead: `src/SparkRenderer.ts:443`
declares `this.lastPixelLimit`; it's captured from the previous traversal's result at
`SparkRenderer.ts:1477: this.lastPixelLimit = pixelLimit;`, and dutifully re-sent on the *next*
call at `SparkRenderer.ts:1463: lastPixelLimit: this.lastPixelLimit,`. It crosses the worker
boundary (`src/worker.ts:762` destructures it, `:820-823` passes it positionally as argument 3 into
whichever traversal function is selected) and lands exactly on the parameter that gets silently
discarded. **This is infrastructure for temporal continuity that was built on both sides of the
wire and connected to nothing.** Whoever designed the API surface clearly intended the traversal to
be able to reason about "what did I select last frame" — it just never got wired up. This is not a
subtle architectural gap; it's a parameter with a name and a value sitting right there, unused.

### 6.2 No hysteresis band — one threshold does two incompatible jobs

Both traversal functions use a *single* scalar (`pixel_scale_limit` in standard mode;
`current_scale` sweeping toward it in dynamic mode) for both "should this node refine further" and
"should this node be accepted as coarse enough" — the exact same comparison, same constant, no
margin between the two decisions:

- Standard, `lod_tree.rs:483`: `if pixel_scale <= pixel_scale_limit { break; }` (stop descending)
  and `:532`: `if pixel_scale <= pixel_scale_limit { output.push(...) } else { frontier.push(...) }`
  (accept-as-leaf vs. keep-refining) — identical comparison, identical constant.
- Dynamic, `:720` and `:760-761`: same pattern against `current_scale`/`pixel_scale_limit`.

Every real-time LOD system with any temporal stability (terrain streaming engines, virtual texture
systems, Nanite-style clustered LOD) uses an asymmetric dead-zone: refine above `limit * 1.1`, only
coarsen below `limit * 0.9`, so a node hovering near the boundary doesn't re-decide every frame.
Nothing like that exists here. A node whose `pixel_scale` sits within noise distance of the
threshold — which, at any reasonable viewing distance for a chunk-sized node, is *most* nodes near
the current LOD frontier — is one sub-pixel camera jitter away from flipping state on the next
frame.

### 6.3 The specific mechanism behind "I notice it while looking around"

`compute_pixel_scale` (`rust/spark-rs/src/lod_tree.rs:601-629`) applies a foveation multiplier
based on the angle between a splat and the camera's forward direction:

```rust
let forward_dot = delta.dot(forward);
let foveate = if forward_dot <= 0.0 {
    behind_foveate
} else {
    let dot = forward_dot * inv_distance;
    if dot >= cone_dot0 { 1.0 }
    else if dot >= cone_dot {
        let t = (dot - cone_dot) / (cone_dot0 - cone_dot);
        cone_foveate + (1.0 - cone_foveate) * t
    } else {
        let t = dot / cone_dot;
        behind_foveate + (cone_foveate - behind_foveate) * t
    }
};
```

Defaults (`SparkRenderer.ts:540-543`): `behindFoveate = 0.2`, `coneFov0 = 90°`, `coneFov = 120°`,
`coneFoveate = 0.4` — i.e. anything outside your ~45° half-angle central view gets rendered at as
low as 20-40% of its "true" pixel_scale, dragging it below `pixel_scale_limit` and coarsening it.

`forward` is read straight from the *current call's* `view_to_objects` matrix
(`lod_tree.rs:446-447`/`670-671`), which is built fresh every frame in
`SparkRenderer.ts:1400-1412` directly from the live camera transform. **There is no smoothing,
slerp, or rate-limiting on camera orientation anywhere in this path.** The one predictive mechanism
that exists at all — `deltaPred` (`SparkRenderer.ts:1329-1335`, a velocity-based *position*
extrapolation) — is computed and then never applied:

```ts
// SparkRenderer.ts:1396-1397
// Commented out because it makes LoDing less stable
// viewPos.add(deltaPred);
```

Two things worth being blunt about here. First, this confirms someone already noticed instability
and tried to fix it — and the fix made things *worse*, which is a strong signal that naive
prediction is the wrong tool, not that the problem doesn't need fixing. Second, and this is the
part worth being harsh about: **`deltaPred` predicts position drift, not orientation.** Even if it
had been enabled, it would do nothing for exactly the complaint on the table — LOD popping while
*rotating* the camera in place, where position barely changes but the foveation cone sweeps across
the whole scene every frame with zero damping. The one fix attempt on record was aimed at the wrong
axis of camera motion.

Put together: rotate the camera, and every splat sitting near the `coneFov0`/`coneFov` boundary has
its effective `pixel_scale` recomputed discontinuously frame-to-frame as the cone sweeps past it,
against a threshold with no hysteresis margin (§6.2), with no per-node memory of what was selected
last frame (§6.4) to smooth the transition even if there were a margin. This is not a minor
polish item — it is the single most exercised trigger for visible popping, and it fires on the most
common camera action there is (looking around).

### 6.4 Zero state carried between frames — every traversal starts from nothing

`struct LodState` (`lod_tree.rs:158-166`) holds tree geometry and scratch buffers
(`frontier`/`output`/`touched`/`touched_set`/`buffer`) — nothing that records "what was selected
last frame." Both traversal functions clear their scratch state and reseed from the root every
single call (`:458-462`, `:466-476` standard; fresh locals per `inst_index` at `:685-698` dynamic).
There is no concept of "this node was fine detail last frame, keep it fine detail this frame unless
something has clearly changed" — every frame re-derives the entire selection from first principles,
independently, with no continuity guarantee beyond whatever naturally falls out of the camera
having moved only slightly. Combined with §6.2's lack of a dead-zone, this is the architecture that
makes flicker structurally likely, not just occasionally unlucky.

### 6.5 No cross-fade mechanism exists anywhere — every transition is a hard cut

Searched the shaders (`src/shaders/`) and the paging/upload code (`src/SplatPager.ts`) for anything
resembling dithered/screen-door transparency, alpha-blend-over-time, or a "time since this LOD
level was selected" value feeding the shader. **Nothing found.** The only opacity-adjacent LOD
option, `lodInflate`, is a static per-splat cosmetic correction gated purely on
`rgba.a > 1.0` (`splatVertex.glsl:107-121` — rescales splat size to compensate for opacity clamped
to 1), with no frame index, no transition-progress input, nothing temporal at all. So: not only
does a node's LOD selection flip abruptly (§6.2-6.4), when it flips **the visual result changes in
a single frame with no blend whatsoever.** Coarse blob → full detail (or the reverse) is an instant
swap, every time.

This compounds specifically with chunk streaming. The residency gate in `traverse_lod_trees`
(`:519-525`, identical shape in dynamic at `:746-753`) is a hard binary branch:

```rust
if first_page == 0xFFFFFFFF || last_page == 0xFFFFFFFF {
    output.push((inst_index, paged_index));   // not resident: draw the coarse placeholder
    continue;
}
// ... resident: immediately expand into ALL real children, same call, same frame
```

`chunk_to_page` flips from "missing" to "resident" synchronously inside `update_lod_trees`
(`:310-313`) the moment upload finishes — the very next traversal call sees full residency and can
jump from "one giant placeholder point" to "every real leaf splat in that chunk" in a single frame,
no ramp. This is the other half of the "large bits popping in one at a time" complaint from §1.2 —
it isn't just that chunks are binary in *content* (already covered in §2.2), it's that the
*transition itself* has no visual softening at all.

### 6.6 A wrinkle for Phase 4's existing mobile recommendation

One more finding that specifically revises §7.3 below: `dynamic_traverse_lod_trees`'s adaptive
threshold (`current_scale`, converging via `0.99 * current_scale * ratio.powf(1.0/2.0)` at
`:778-781`) depends on `output_count`, which is sensitive to small floating-point differences in
accumulated `compute_pixel_scale` results frame-to-frame. Concretely: two *nearly but not exactly
identical* camera frames (the normal case — sub-pixel jitter, float rounding) can converge to
measurably different `current_scale` values and therefore different accepted node sets, **even
holding the camera still**. (For the record: this is not `BinaryHeap` tie-break nondeterminism —
standard mode's heap key tuple includes a full tiebreaker and is deterministic for identical input.
The dynamic-mode issue is threshold-sweep sensitivity to tiny `pixel_scale` deltas, a distinct and
arguably worse mechanism since it's baked into the approximation itself, not an edge case.) This
means **dynamic mode is plausibly a worse popping offender than standard mode**, not just a faster
one — directly cutting against defaulting mobile to it without the stability-specific benchmark
§7.3 already called for, now with an even sharper reason to run it.

### 6.7 Bottom line

Nothing here requires new theory — every real-time LOD/streaming system that doesn't visibly pop
solves exactly these problems (hysteresis bands, temporal smoothing on the camera input driving
selection, cross-fade during transitions, ramped residency changes). This codebase has the hook for
solving the first one already built and wired end-to-end on both sides of the worker boundary
(§6.1) and simply never connected it. §8 (Phase 5) below is not speculative — it's closing gaps the
existing API surface already anticipated.

---

## 7. Phase 4 — `dynamic_traverse_lod_trees` (PR #344) integration

### 7.1 What it is (confirmed this session)

`rust/spark-rs/src/lod_tree.rs`: `traverse_lod_trees` at **lines 412-599** (strict best-first,
`BinaryHeap`-ordered, pops the single largest-pixel-scale node each step — exact but heap
maintenance cost scales with frontier size). `dynamic_traverse_lod_trees` at **lines 632-843**
(adaptive threshold-sweeping: iteratively tighten a `current_scale` cutoff, batch-expand everything
above it per pass, converge toward `pixel_scale_limit` — cheaper per node processed, coarser/approximate,
explicitly designed for large trees where heap overhead dominates).

### 7.2 Why it currently doesn't matter (and will start to)

Both functions read the *identical* `child_start`/`chunk_to_page`/`LodSplat.size` structure. With
today's flat, 1-level-deep, binary-per-chunk tree (§2.1-2.2), there is no meaningful difference for
either algorithm to exploit — there's nothing to "approximate" when a chunk is all-or-nothing.
**Once Phase 1 lands real multi-level hierarchy, this stops being true**: `dynamic_traverse_lod_trees`'s
whole value proposition (fast approximate selection over large, deep trees) only exists once trees
are actually large and deep. This is the "compounds with" answer from the prior turn, now
actionable: **Phase 1 is a prerequisite for Phase 4 being worth anything**, not an independent
workstream.

### 7.3 What Phase 4 actually needs, once Phase 1 exists

1. **Re-benchmark, don't assume.** `dynamic_traverse_lod_trees`'s approximation quality
   (`current_scale` convergence rate, `0.99 * current_scale * ratio.powf(1.0/2.0)` at
   `rust/spark-rs/src/lod_tree.rs` — verify current line, was ~778 pre-Phase-1 edits) was tuned
   against *whatever trees existed when PR #344 landed* — almost certainly RAD files with real
   octrees, not SP5's (currently degenerate) trees. Once Phase 1's real per-chunk hierarchy exists,
   re-run a controlled comparison: frame time **and popping/stability** (not just visual-quality
   delta) between `standard` and `dynamic` modes on the same real GPS-scale scene, at a size large
   enough to exercise the "big tree" regime `dynamic` is meant for (multi-million splat, deep
   hierarchy). §6.6 gives a specific reason `dynamic` may fail the stability half of this
   comparison even if it wins on raw frame time — measure both, don't assume speed implies parity.
2. **Default recommendation for mobile — revised by §6.6.** Given the explicit "excellent flux-gs
   performance on mobile" requirement, `dynamic` was the presumptive default candidate — but §6.6
   found its adaptive-threshold convergence is *more* sensitive to frame-to-frame float noise than
   standard mode's deterministic heap selection, meaning it could plausibly make popping *worse*,
   not just run faster. **Do not default mobile to `dynamic` on a performance argument alone.**
   Land Phase 5 (§8)'s hysteresis/smoothing fixes first — they benefit both modes — then run
   §7.3.1's stability-inclusive benchmark, and only then decide the default per-mode. If `dynamic`
   still wins on speed but loses on stability after Phase 5, consider whether the hysteresis
   dead-zone should be *wider* specifically in dynamic mode to compensate, rather than accepting a
   faster-but-flickerier mobile default.
3. **`missing_count`/`touched` semantics under Phase 1's deeper trees.** `dynamic_traverse_lod_trees`
   tracks `missing_count` and per-chunk `chunk_max` pixel-scale bookkeeping
   (`rust/spark-rs/src/lod_tree.rs:680-703,715,735-739`) for chunks not yet resident — this already
   generalizes correctly to "not-yet-loaded coarse levels of a real tree," not just "not-yet-loaded
   sibling chunks," since it operates purely on `chunk_to_page` residency, agnostic to whether a
   chunk's *content* is flat or hierarchical. No code change anticipated here — call this out
   explicitly as a case to include in Phase 1's acceptance tests (§3.3), not something to
   pre-emptively modify.

---

## 8. Phase 5 — Temporal LOD stability (fixing the popping, not just the missing detail)

**Goal:** LOD transitions a user cannot catch happening, on top of the real hierarchy Phase 1
provides. Phase 1 fixes *what detail exists to select from*; without Phase 5, a real hierarchy will
still pop between levels for exactly the reasons in §6 — more gracefully than today's all-or-
nothing chunks, but still visibly. Both phases are required for "good LOD," not just one.

### 8.1 Wire up the hysteresis input that already exists

Lowest-risk, highest-leverage fix, and it's mechanical: `_last_pixel_limit` (§6.1) is already
computed and threaded correctly on the JS side — the only work is inside
`traverse_lod_trees`/`dynamic_traverse_lod_trees` (`rust/spark-rs/src/lod_tree.rs`). Use it to
derive **two** thresholds instead of one: e.g. `refine_limit = pixel_scale_limit * 0.9` (refine
above this) and `coarsen_limit = pixel_scale_limit * 1.15` (only accept-as-leaf below this) — a
node whose `pixel_scale` falls between the two keeps whatever state it was in last frame rather
than re-deciding. This closes §6.2 directly. Note this requires the traversal to actually know
"what state was this node in last frame" per-node, not just a single global last-frame scalar — see
§8.2, they're the same underlying gap.

### 8.2 Persist per-node selection state across frames

§6.4's finding: `LodState` has no per-node memory. Add a small persistent structure — e.g. a
`HashMap<(lod_id, paged_index), FrameSelectionState>` (last-selected pixel_scale, last-frame-touched)
alongside the existing `lod_trees` map in `LodState` (`lod_tree.rs:158-166`) — cheap relative to the
`splats: Vec<LodSplat>` data itself, and only needs entries for nodes actually near a decision
boundary (most of the tree is either obviously-refine or obviously-coarsen and doesn't need
tracking). This is the prerequisite for §8.1's asymmetric bands to mean anything, and also the
prerequisite for §8.3's cross-fade (which needs "how long has this node been in its current state"
to drive a fade timer).

### 8.3 Smooth the camera orientation input specifically for LOD selection

§6.3's finding: `forward`/`origin` are read raw from the current frame's camera transform with zero
damping, and the one predictive attempt (`deltaPred`, `SparkRenderer.ts:1329-1335,1396-1397`)
targeted position, not orientation, and was abandoned. Proposal: apply a *separate*, lightweight
low-pass filter (e.g. exponential/critically-damped smoothing on the forward quaternion, or a short
sliding-window average) to the camera transform **specifically as fed into `viewToObjectCols`**
(`SparkRenderer.ts:1400-1412`) for LOD purposes only — the actual render camera stays exactly as
responsive as today; only what the LOD traversal *thinks* the camera is looking at lags by a few
frames. This is a materially different, lower-risk approach than the abandoned `deltaPred` attempt:
that one tried to *predict ahead* (extrapolation, inherently unstable when velocity estimates are
noisy — plausibly why it "made LoDing less stable"); this smooths *behind* (a lagging filter on
already-known values, a well-understood stable technique). Don't reuse `deltaPred`'s approach or
assume its failure says anything about this one — they solve different problems (position vs.
orientation) with different math (extrapolation vs. smoothing) and different risk profiles.

### 8.4 Cross-fade LOD transitions in the shader

§6.5's finding: zero blend infrastructure exists. Needs, in order of dependency:

1. A transition-progress value per rendered splat/node, driven by §8.2's per-node "time since state
   changed" (needs a shared clock/frame-time value threaded to the shader, likely alongside the
   existing `lodOpacity`/`shMax` uniform-style values already passed per `SplatEncoding` —
   `src/SplatPager.ts`'s `splatEncoding` configuration is the existing analogous plumbing to extend).
2. A dither pattern (cheapest: a per-pixel or per-splat-instance hash-based stochastic discard,
   ramping the discard probability from 0→1 over the transition window — avoids double-rendering
   cost of a true alpha cross-fade, and is TAA/temporal-accumulation friendly if Spark ever adds
   that) OR a true alpha blend rendering both the outgoing and incoming LOD selection for a short
   overlap window (simpler to reason about, costs more — render both sets briefly). Recommend
   starting with the dither approach given the "excellent performance on mobile" requirement —
   double-rendering during every transition is exactly the kind of cost mobile can't absorb at
   scale.
3. Apply the same mechanism to the chunk-residency pop (§6.5's second half) — a chunk transitioning
   from placeholder to resident should ramp in via the identical transition-progress mechanism, not
   a special case.

### 8.5 Acceptance

A test that a human eye can't currently pass by inspection needs a numeric proxy: instrument a
fixed, scripted camera path (e.g. a slow 360° rotation in place, the exact motion the user reported
triggering the complaint) against a real Phase-1-hierarchical scene, and assert the **set of output
`paged_index` values changes by less than X% between consecutive frames** for camera motion below
some angular-velocity threshold — a concrete, scriptable regression guard for "stopped flickering,"
measurable without a human watching. Pick X empirically against today's (pre-Phase-5) baseline
churn rate on the same path, so the fix's magnitude is provable, not just "looks better."

---

## 9. Sequencing and dependencies

```
Phase 1 (real hierarchical LOD per chunk)
   │
   ├──> Phase 2 (GPS-scale distance heuristics)      -- needs Phase 1's real size-per-level to tune against
   │
   ├──> Phase 4 (dynamic traversal re-benchmark)      -- needs Phase 1's real trees to be meaningful
   │
   └──> Phase 5 (temporal LOD stability)              -- needs Phase 1's real trees so there's an
                                                          intermediate level to transition THROUGH,
                                                          not just placeholder<->full-res

Phase 3 (Track C2 neural path)                        -- independent of 1/2/4/5; can run in parallel,
                                                          shares the SVQ/Huffman wire format Phase 1
                                                          doesn't touch

Phase 5's §8.1 (wire up _last_pixel_limit) and §8.2 (per-node state) are the two exceptions that
CAN start independently of Phase 1 -- they're general traversal-stability fixes that help even
today's flat tree (a chunk flickering between "placeholder" and "full chunk" benefits from
hysteresis too), and de-risk Phase 1 by proving the mechanism before there's a deep hierarchy to
worry about breaking.
```

Recommended order: **Phase 1 first** (it's the one thing every other phase either depends on or is
made meaningless without it, and it directly fixes all three symptoms reported this session) —
**except** §8.1/§8.2, which can and should start in parallel, since they're cheap, general fixes
that improve stability on the *current* tree shape too and de-risk the mechanism before Phase 1
gives it more to manage. Phase 3 (C2) can run in parallel on a separate track since it doesn't touch
chunking/traversal at all — only the attribute-quantization/decode path, which Phase 1 leaves
untouched. Phase 2, Phase 4, and the bulk of Phase 5 (§8.3-8.5, which need a real intermediate LOD
level to smooth *between*) all slot in after Phase 1 has something real to tune/transition against.

## 10. Explicit non-goals for this document

- **Not proposing to abandon or rewrite the k-d tree spatial chunking** added this session
  (`kdPartitionIndices`) — it correctly solved "chunks must be spatially coherent regions," which
  is a *prerequisite* for Phase 1's per-chunk hierarchy to make sense (a hierarchy built over a
  spatially-incoherent point set would itself be low quality). Phase 1 builds *inside* each spatial
  chunk boundary, not instead of them.
- **Not proposing to remove the cross-chunk sibling-pointer stitching** (§2.3) — it's the correct,
  tested cross-chunk addressing mechanism; Phase 1 changes what a chunk's own internal tree
  contains, not how chunks reference each other.
- **Not re-litigating the legal gate** (G-PCC/`tmc3.wasm`) — per standing instruction from earlier
  in this engagement, already resolved by explicit user approval for this branch/test context.
- **Not proposing to re-enable `deltaPred`** (`SparkRenderer.ts:1329-1335,1396-1397`) as-is — §6.3
  found it targets the wrong axis of camera motion (position, not orientation) for this specific
  complaint, and was already abandoned once for making things *less* stable. §8.3's orientation
  smoothing is a different mechanism (lagging filter vs. forward extrapolation) solving a different
  problem; don't conflate reviving the old code with implementing the new fix.
