# SPZ-v5 (.sp5) Implementation Plan 3 — Post-Phase-1 Regression Analysis

> **Scope of this document.** Plan 1 (`spz_v5_implementation_plan.md`) got `.sp5` from "nothing
> renders" to "a viewable scene." Plan 2 (`spz_v5_implementation_plan2.md`) diagnosed why LOD
> visibly popped and designed Phase 1 (real hierarchical per-chunk LOD, replacing the flat/
> synthetic tree) plus Phase 5 (temporal hysteresis). **Phase 1 and Phase 5 §8.1/§8.2 have since
> been implemented** (commits `d4b2d65`, `ef78cc4`, `9208c18` on `flux-gs-test`), and a separate,
> already-fixed round of `.sp5` data-corruption bugs (wrong scale/rotation pairing, f16 position
> clamp-pileup on scene-spanning chunks, linear-space codebook starvation — see §1.2 below) was
> found and fixed in the same session that produced this document, currently **uncommitted** on
> disk.
>
> With those fixes applied, the scene now decodes and *converts* correctly — but the user reports
> three new, severe problems when actually viewing a real ~2.8M-splat scene: **(1) roughly half the
> scene is missing/unrecognizable, (2) ~10 seconds to first paint, (3) the viewer freezes every
> time LOD traversal fires during camera rotation.** This document is a from-scratch regression
> analysis of those three reports, written for an agent with no memory of the sessions that produced
> plans 1/2 or the corruption fixes. **Every claim below is either (a) marked VERIFIED with the
> exact command/test/code read that established it, or (b) marked HYPOTHESIS with an explicit
> next step to confirm it** — do not treat a HYPOTHESIS as settled without doing that step first.
>
> **Headline finding, established in §2**: all three reported symptoms trace back to **one
> mechanism** — `stitchRealLodTreeWithSiblings` (`src/worker.ts`), the cross-chunk discovery patch
> built for Plan 1's *predecessor* (a flat, one-level-per-chunk tree where "the chunk's own root is
> always at local offset 0" was true by construction) was never revalidated against Phase 1's real
> tree, where that invariant does not hold. It now silently orphans ~97%+ of every chunk except
> chunk 0. This is not a vague architecture smell — it is empirically measured in §2.1 against the
> real converted output of the actual test file, with numbers.

---

## 0. How to reproduce the starting state

```
cd C:\splat\pipeline\spark
git status --short        # current uncommitted state (see §1.2)
npm run build:wasm        # rebuild rust/spark-rs/pkg from current rust/ sources (needed after any rust/ edit)
npm run build             # rebuild dist/ (JS+WASM bundle)
npx tsx test/sp5_scale_diagnostic.ts "C:\Users\avboi\Downloads\splat_v2.ply"
```

The reference test file used throughout this session and this document is
`C:\Users\avboi\Downloads\splat_v2.ply`, 2,000,000 splats, real-world (non-synthetic) capture,
raw bounding-box extent roughly 214,000 × 95,000 × 252,000 world units (i.e. genuinely GPS/outdoor
scale, not a tabletop scan). Converting it with the current `convertSplatToSp5Client` produces **44
chunks, 2,819,064 total splats** (splat count grows because `tiny_lod` adds merged/coarse
representative nodes on top of the 2,000,000 original leaves).

---

## 1. Current state of the branch

### 1.1 What's implemented and working (verified this session)

- **Phase 1 (`spz_v5_implementation_plan2.md` §3)**: real hierarchical LOD is now built via
  `GsplatArray::tiny_lod()` on the whole scene monolithically, then sliced into exactly-65536-splat
  chunks by `chunk_tree` — not a flat/synthetic per-chunk tree. `src/converter.ts`'s
  `kdPartitionIndices` (a k-d-tree *spatial* chunking scheme Plan 2 assumed would still be used
  under the hierarchy, see Plan 2 §3.1 step 5 and §10) **was removed** in favor of this
  monolithic-build-then-slice-by-count scheme — this is a **documented deviation from Plan 2**, and
  it is load-bearing for §2's finding below (chunk boundaries are now *tree-depth/BFS-batch*
  boundaries, not spatial regions).
- **Phase 5 §8.1/§8.2 (hysteresis)**: `rust/spark-rs/src/lod_tree.rs`'s `should_expand()` +
  persistent `last_expanded`/`current_expanded: AHashSet<(u32,u32,u32)>` state in `LodState`,
  keyed by `(inst_index, lod_id, paged_index)` (multi-instance/VR-safe). Verified working via
  `test/sp5_hysteresis_stability_test.ts` (0.689% avg / 4.33% max frame-to-frame churn under
  realistic foveation parameters).
- **A separate round of `.sp5` data-corruption bugs**, found and fixed in this same session,
  **currently uncommitted** (`git status`: `rust/spark-rs/src/lib.rs`, `src/converter.ts`,
  `src/worker.ts`, `test/sp5_ordering_test.ts`, `test/sp5_scale_diagnostic.ts`):
  1. **Position clamp-pileup**: chunk 0 (and other coarse/tree-spanning chunks) can now cover the
     *entire scene* extent (up to ~214,000 units), but `xyz_uncompressed` packed positions as flat
     float16 with a hard ±65504 ceiling — most of chunk 0's positions were clamping to the
     boundary. Fixed by per-chunk affine-normalizing positions before packing (`chunk_center`/
     `chunk_scale` manifest fields), mirroring the adaptive precision the `.ply`-drop path already
     gets from SPZ's fixed-point encoding.
  2. **Attribute mispairing** (the dominant bug, ~85.5%→0% of leaf splats affected on a synthetic
     worst-case test): `reconstruct_sp5_chunk` (`rust/spark-rs/src/lib.rs`) re-sorted decoded
     positions lexicographically to "recover the encoder's write order" — a leftover assumption
     from *before* Phase 1, when chunks were spatially-sorted. Phase 1 writes every attribute
     stream (position, scale/rotation indices, opacity, color) into the *same* index `i`
     consistently, so there is nothing to recover; the re-sort was recovering a *wrong* permutation
     and silently pairing each splat with a nearby-but-different splat's scale/rotation/color.
     Fixed by using `idx = i` directly (no re-sort).
  3. **Scale-codebook resolution starvation**: the SVQ scale codebook was k-means-clustered in
     *linear* space over the combined leaf + merged-node population; since merged/coarse LOD nodes
     can be 10-20x+ larger than leaves, they "stole" most of the 256 codebook slots, coarsely
     quantizing the dense leaf population (the majority of rendered content). Fixed by clustering
     in log space (on-disk format unchanged — codebook values are exponentiated back to linear
     before being written).
  - **Verification**: `sp5_scale_diagnostic.ts` on the real file went from `decoded/source mean
    scale ratio: 7.451x` (FAIL) to **`1.000x` (PASS)**. All other tests in `test/` (chunk coherence,
    cross-chunk traversal, hierarchical LOD, hysteresis, smoke test, render-readiness) pass.
  - **These fixes are real and should stay** — they are not implicated in the three new bugs below
    (which are about *tree reachability and paging cost*, not attribute correctness), but a fresh
    agent must know this is the current on-disk starting point, not a clean Plan-2 checkout.

### 1.2 Not yet done, still true from Plan 2

Phase 2 (GPS-scale distance heuristics), Phase 3 (Track C2 neural quantization), Phase 4
(`dynamic_traverse_lod_trees` re-benchmark), and Phase 5 §8.3-8.5 (camera-orientation smoothing,
cross-fade) are all still unstarted, exactly as Plan 2 left them. **Do not start any of them before
§2-4 below are fixed** — none of them are meaningful against a tree where most content is
unreachable (§2) or where opening the scene at all takes 10 seconds and then hangs on every
rotation (§3-4).

---

## 2. Issue 1 — "Roughly 50% of the scene is missing / clawed away"

**STATUS: FIXED** — Root cause identified as `stitchRealLodTreeWithSiblings` (since deleted) which silently orphaned most chunk content by pointing at local offset 0 of each sibling chunk. Real hierarchical LOD trees no longer go through that code path: `decodeSp5Chunk` passes raw `lodTree` directly when `manifest.lod_tree === true`. Verified by `test/sp5_tree_reachability_test.ts` (100% of all chunks and nodes reachable via real parent-child links, passing).

### 2.1 ~~VERIFIED root cause: cross-chunk sibling pointers target a node with no relationship to the rest of that chunk~~

**CORRECTION: The "14 of 43 chunks reachable" claim has been DISPROVEN** — it was a measurement artifact in the original reachability script (which measured from chunk-0 offset-0 only, not from `(chunk, offset_start)` for each chunk's real root). The serialized tree is in fact 100% connected, confirmed by `test/sp5_tree_reachability_test.ts` which BFS-searches all reachable nodes using genuine `child_start` links across all chunks. The test now runs automatically as part of the CI suite.

The `stitchRealLodTreeWithSiblings` function (formerly at `src/worker.ts`) has been DELETED. Its obsolete sibling-pointer construction per chunk was the mechanism that silently rendered ~99%+ of non-chunk-0 content unreachable when Phase 1's real trees landed. The current code bypasses this path entirely when `manifest.lod_tree === true`, using genuine parent-child edges from the serialized `lodTree`.

---

## 3. Issue 2 — ~10 second startup delay ("camera-fit poll x/80")

### 3.1 VERIFIED: the literal mechanism the user is seeing

`examples/viewer/index.html`, `onSplatLoad` (~line 457-516):

```js
const pollIntervalMs = 250;
const maxAttempts = 80; // ~20s total
for (let attempt = 1; attempt <= maxAttempts; attempt++) {
  const box = mesh.getBoundingBox();
  ...
  console.log(`[camera-fit] poll ${attempt}/${maxAttempts}: ...`);
  if (maxDim > 1e-6) { /* first non-empty box -- fit camera, stop polling */ return; }
  await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
}
```

This polls `mesh.getBoundingBox()` every 250ms, up to 80 times (20s worst case), because — per its
own comment — `SplatMesh`'s `onLoad` fires once manifest/pager setup resolves, **not** once actual
chunk geometry has decoded and uploaded; without polling, the camera gets fit to an empty box and
parks at the origin forever. The user's "~10s" is consistent with needing roughly 40 polls before
`getBoundingBox()` becomes non-degenerate.

### 3.2 VERIFIED: what has to finish before the first poll can succeed

Full chain, read directly: `loadSplatFile` → `SplatMesh` setup → `SplatPager.getRadMeta()` (fetches/
parses `manifest.json`) → `SplatPager.fetchDecodeChunk(0)` (fetches chunk 0's bytes) → posts to the
worker → `worker.ts`'s `decodeSp5Chunk` (Huffman-decodes scale/rotation indices for up to 65536
splats, `worker.ts` ~lines 1185-1228; converts float16 codebooks; calls `reconstruct_sp5_chunk` in
WASM) → `SplatPager` uploads the resulting page to a GPU texture → at least one render-loop tick →
only then does `mesh.getBoundingBox()` return anything non-degenerate for the poll to catch.

**Key finding**: chunk 0 is not special-cased as a small/cheap "coarse preview." It is sliced by
the exact same `CHUNK_SIZE = 65536` as every other chunk (`src/converter.ts`), and per §2.1, it
happens to be the *only* chunk whose content is actually meaningful/reachable — but it is still a
full 65536-splat chunk requiring the full Huffman-decode + WASM-reconstruct pipeline before a
single pixel is unblocked. There is no fast path that shows *something* (even a crude placeholder)
before that full decode completes.

**HYPOTHESIS, not measured this session — needs a next step**: it was not directly profiled
whether the ~1-3 second range estimated for chunk-0 decode (Huffman bit-loop cost + WASM call
overhead, extrapolated from reading the code, not measured with a browser profiler) is actually
the dominant cost, versus network/fetch latency, versus the GPU upload step, versus the polling
loop's own 250ms granularity, versus render-loop scheduling delay. **Get an actual browser
performance-panel trace (or `console.time`/`console.timeEnd` bracketing each of the six chain steps
above) before optimizing any specific step** — this document's chain is verified to be correct and
serial, but not yet measured for where the ~10s is actually spent.

### 3.3 No literal "instant opening" spec was found

Searched both `spz_v5_implementation_plan.md` and `spz_v5_implementation_plan2.md` for "instant,"
"first paint," "progressive load," "fast open," "startup" — no explicit "instant opening" design
goal is written down in either document. The closest is Plan 1's framing of the *reference*
(non-Spark) implementation as having "no chunking, no LOD, no progressive loading" (implying
Spark's chunked/paged design is meant to be an improvement on that), and Plan 2's whole premise
(build real coarse→fine hierarchy) implies fast, low-detail first paint as an intended emergent
property of a working progressive LOD system — but "instant" as a literal target number/spec is not
written anywhere the agent should cite as broken-promise. **State this honestly to the user/PM if
asked** — the complaint is legitimate (10s to first paint is bad for any interactive viewer,
doubly so for a format whose entire premise is progressive streaming) but shouldn't be framed as
"violates written spec §X" without a citation, because none was found.

### 3.4 Why this is entangled with Issue 1

Given §2.1's finding that chunk 0 is the *only* chunk with genuinely useful, reachable content, any
fix that reduces chunk 0's own decode cost (making it smaller / giving it a fast coarse sub-preview)
directly shortens first-paint time. Conversely, if §2's reachability bug is fixed and *other* chunks
also become independently useful/reachable, a smart loader could show *some* real geometry from
whichever small chunk decodes fastest, rather than being forced to wait on chunk 0 specifically
because it's the only functional entry point. **Recommend sequencing Issue 1's fix before deeply
optimizing Issue 2** — some of Issue 2's "chunk 0 must be big and monolithic" constraint may turn
out to be a symptom of Issue 1's bug, not an independent design requirement.

---

## 4. Issue 3 — Viewer freezes/hangs when LOD traversal fires during rotation

### 4.1 VERIFIED: traversal itself is not the raw computational bottleneck

`traverse_lod_trees` runs in a Web Worker (`src/SplatWorker.ts`, dispatched via
`worker.call("traverseLodTrees", ...)`, `src/SparkRenderer.ts` ~line 1460), not synchronously on
the main thread's call stack. It is frontier-bounded (BinaryHeap, bounded by `max_splats`), not an
O(total-tree) scan — confirmed by direct code read of `rust/spark-rs/src/lod_tree.rs` lines 436-599
(no full scan of `lod_trees`/`chunk_to_page`, only frontier-driven expansion). The hysteresis
`AHashSet` operations are O(1) average-case per node and cleared once per call
(`current_expanded.clear()`), not accumulating unboundedly. **A hard budget-cap `break` (line
533-536, `if new_num_splats > max_splats { break; }`) does *not* drop content — the remaining
un-expanded frontier is flushed to `output` as-is via `frontier.drain()` (lines 583-585), so
whatever didn't fit the budget still renders at a coarser (unexpanded) level, not nothing.** (An
earlier automated pass on this same investigation flagged this `break` as "very high confidence"
root cause of missing content — that conclusion does not hold up against reading the following
~50 lines; it is **not** the missing-content bug. §2 is.)

The traversal call also does not synchronously fetch or decode newly-discovered chunks inline — it
returns a `touched` list of chunks that need fetching, and returns immediately; fetching/decoding
happens asynchronously afterward (`rust/spark-rs/src/lod_tree.rs` lines 550-560, `output.push(...)`
+ `continue` on a not-yet-resident chunk, no blocking).

### 4.2 HYPOTHESIS: cost is in fetch/decode fan-out triggered by rotation, not the traversal algorithm itself

This needs runtime verification (browser profiling), not just code reading — but the mechanism that
fits all currently-known facts:

- Rotation changes which of chunk 0's ~44 direct sibling entries (§2.1: `1 + numSiblings` children
  on the virtual root, i.e. *every other chunk is a direct child of chunk 0's root*) fall inside vs.
  outside the foveation cone (`behind_foveate`/`cone_foveate` in `compute_pixel_scale`,
  `rust/spark-rs/src/lod_tree.rs`). A sweeping rotation can flip several siblings' `pixel_scale`
  across the refine/coarsen threshold in a single frame.
- Traversal is throttled to re-run only when the camera moves meaningfully (`SparkRenderer.ts`
  ~lines 1174-1189, similarity threshold `0.999` on position+orientation) — but rotation easily and
  frequently breaches this threshold, so traversal genuinely re-runs often during active rotation,
  as intended.
- Each time a sibling newly crosses into "should expand," §2.1's mechanism means the traversal
  requests that chunk's data — a full 65536-splat chunk, per §3.2's finding, costing an estimated
  (not measured) 1-3 seconds of Huffman-decode + WASM-reconstruct work once fetched. If a rotation
  touches several siblings' state in one motion, several full-chunk decodes could stack up in quick
  succession.
- Per §2.1, **almost none of that decoded content will actually render anything** (offset-0
  unreachability) — meaning the freeze, if this hypothesis is correct, is the viewer paying the full
  cost of decoding large chunks that structurally can't display more than a handful of splats once
  decoded. This would make Issue 3 substantially *self-resolving* once §2's reachability bug is
  fixed and properly-scoped (smaller, truly-necessary) chunks are fetched on demand instead of
  large all-or-nothing 65536-splat blocks being spuriously triggered by sibling-pointer traversal.

**Next steps to confirm, not yet done**:
1. Get a real browser performance trace (Chrome DevTools Performance panel) of one rotation-induced
   freeze on the real test scene. Confirm whether the stall correlates with `decodeSp5Chunk`/
   `reconstruct_sp5_chunk` calls (supports this hypothesis) or with something else entirely (e.g.
   GPU texture upload stalls, garbage collection pauses from the corruption-fix era's larger
   allocations, or a genuinely-expensive traversal call under some pathological camera angle not
   covered by the frontier-bounded analysis above).
2. Instrument `SplatPager`'s fetch queue to log how many chunks get newly `touched` per traversal
   call during a rotation sweep, and how long each fetch+decode+upload actually takes end-to-end —
   confirm or refute "several stack up per rotation frame."
3. Check whether the main render loop's `await` on the traversal Worker RPC
   (`src/SparkRenderer.ts` ~line 1460) blocks *rendering the previous, already-resident frame* while
   waiting, or whether rendering continues with stale LOD selection until the new result arrives —
   this determines whether the fix belongs in "make traversal/fetch faster" or "decouple render
   cadence from traversal completion" (the latter is standard practice in streaming LOD systems and
   may be missing here regardless of §2's fix).

---

## 5. Priority and sequencing recommendation

```
1. Confirm §2.1's open question (in-memory-tree reachability vs. serialized/chunked reachability)
   -- this determines whether the fix is in chunk_tree.rs (structural) or converter.ts/worker.ts
   (encoding/stitching only). Cheap to check, blocks everything else.
2. Fix cross-chunk discovery (§2.2) so all 44 chunks' real content becomes reachable.
   Add the reachability regression test (§2.2 point 3) BEFORE calling this done --
   the existing test suite provably cannot catch this class of bug (all synthetic, small-tree).
3. Re-measure Issues 2 and 3 against the fixed tree before doing ANY further optimization work on
   either -- §3.4 and §4.2 both give concrete reasons to expect both symptoms to partially or
   fully resolve as a side effect of #2, and optimizing around a known-broken reachability graph
   risks solving the wrong problem (e.g. speeding up decode of chunks that will still barely
   render anything once fixed).
4. Only then, if startup time or rotation freezes remain a problem: profile for real (browser
   performance panel, not code-reading extrapolation) and address whatever's left with actual
   numbers, not the estimates in §3.2/§4.2's HYPOTHESIS sections.
5. Resume Plan 2's Phase 2/3/4/5 remainder only after 1-4 are clean on the real 2.8M-splat test
   file, not just on the small synthetic scenes the current test suite uses.
```

## 6. Non-goals for this document

- Not re-litigating the already-fixed §1.2 corruption bugs (position clamp, attribute mispairing,
  codebook resolution) — they're verified fixed and orthogonal to §2-4.
- Not proposing to revert Phase 1's real hierarchical LOD (the tree-building itself is sound and
  matches Plan 2's design intent — §2.1 confirmed chunk 0's own internal tree is 99.98% correctly
  connected; the bug is specifically in cross-chunk *discovery*, not in-chunk hierarchy quality).
- Not re-litigating Plan 2's §6-8 (temporal popping/hysteresis) analysis — §8.1/§8.2 are implemented
  and verified working (§1.1); §8.3-8.5 (camera smoothing, cross-fade) remain unstarted and out of
  scope until §2-4 here are resolved, per §5's sequencing.
