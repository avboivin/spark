# IMPLEMENTATION_WORK.md — Resumable State Document

> **Last updated:** 2026-07-03 09:50 UTC
> **Branch:** `flux-gs-test`
> **Current build:** commit `6830daa` (Steps 1-2 partial, P2b reverted)

---

## 0. Build Commands

```bash
npm run build:wasm    # after any rust/ change
npm run build         # before any browser verification
npx tsx test/<name>.ts  # run individual test
node scripts/parse_perf_log.mjs <logfile>  # parse perf log
```

---

## 1. Latest Log Analysis (13:46-13:47 UTC, MRNF10k, 2.5M budget, 1.0px, prefetch OFF)

### 1.1 Comparison with Appendix A

| Metric | App A (best) | App C (current) | Delta |
|---|---|---|---|
| Startup | 1075ms | 1089ms | ~same |
| Traversal count | 119 | **151** | **+27% worse** |
| Traversal avg | 233ms | **268ms** | **+15% worse** |
| Traversal settled | 298-485ms | 282-355ms | tighter cadence |
| Frame avg | 117.6ms | 109.8ms | ~same |
| Huffman decode | 46.6ms | 46.9ms | ~same |
| ARRIVE→DONE #1 | 2.9s | 2.7s | ~same |
| Freeable max | 9 | 8 | ~same |

### 1.2 BUG: Dead-zone fix not active in this build

The settled-state traversals show a tight ~320ms cadence (282-355ms range, 15 consecutive traversals every ~320ms) — this is the same `PointerControls` exponential velocity drift pattern from before the `fca8ac0` dead-zone fix. Investigation: the user may be running a stale `dist/` build. **Verify:** rebuild + hard-reload, confirm traversals stop when camera is stationary.

### 1.3 ARRIVE→DONE intervals

| Run | #1 | #2 | #3 |
|---|---|---|---|
| App A | 2.9s | 7.9s | 11.2s |
| App C | 2.7s | 7.4s | 7.2s |

Similar/better — the upload cadence + fetch pipeline is stable.

---

## 2. Five-Step Implementation Plan (FULL INTENT PRESERVED)

### Step 1 — Fix P2B Index Break ✅ DONE

**Intent:** The `| 0xFF000000` in traversal output paths corrupted `pagesInCut` (`>>> 16`), `pagedSplatTexCoord` (`index >> 16` → negative → OOB), and silently killed Task-2 eviction protection.

**What was done:**
- Reverted `| 0xFF000000` in `traverse_lod_trees` (`lod_tree.rs:682-686`)
- Reverted `| 0xFF000000` in `repair_lod_cut` (`lod_tree.rs:1194`)
- Added regression test in `sp5_hysteresis_stability_test.ts`: asserts `(idx >>> 24) === 0` for every output index on frame 0

**Commit:** `7cb74c3`

**Verification:** `sp5_hysteresis_stability_test.ts` passes.

---

### Step 2 — Finish P2A End-to-End ⚠️ PARTIAL

**Intent:** `repair_lod_cut` (ROAM dual-queue) wired from JS, calling incremental repair when camera similarity is within band, falling back to full traversal on cold start/teleport. Merged queue for coarsening. Delta application to persistent index buffer. Target: steady-state repair RPC < 10ms at 40+ pages on MRNF10k.

**What's done:**
- `LodState` has `cut_nodes`, `parent_map`, `last_cut_origins/forwards/limits`, `cut_delta_added/removed`
- `traverse_lod_trees` seeds cut state + populates `parent_map` (child→parent links)
- `repair_lod_cut` function: cold-start detection, re-key, split queue (expand nodes above coarsen_limit), merge queue (coarsen complete sibling groups), budget enforcement, delta computation
- `repairLodCut` worker handler wiring complete (`src/worker.ts`)
- `repair_lod_cut` import added to worker.ts

**What remains:**
- [ ] **SparkRenderer.ts integration:** In `updateLodInstances`, after building instances+pageBounds, call `worker.call("repairLodCut", ...)` instead of "traverseLodTrees" when:
  - (a) A cut exists (check via `this._hasCut` flag set after first full traversal)
  - (b) Camera similarity is within incremental band (`similarity > 0.9`)
  - (c) Resident page set only changed additively (new pages, no removes)
  - On `needsFull` response, fall back to `worker.call("traverseLodTrees", ...)`
- [ ] **Delta application in PagedSplats.update():** Accept `{ added: Uint32Array, removed: Uint32Array }`. For each `removed`: tombstone the index slot (push to free-list). For each `added`: write to first available free-list slot.
- [ ] **Free-list for index slots:** A simple `number[]` stack. When tombstoning, push slot to free-list. When adding, pop from free-list (if empty, append to end — grow the buffer).
- [ ] **Delta-consistency test:** 100 scripted frames of rotation, assert `symmetric_diff(delta_applied_set, full_traversal_set) < 0.1%`.

**Commit:** `6830daa`

---

### Step 3 — Finish P1A (snorm16 + Per-Page GPU Uniforms) ❌ NOT STARTED

**Intent:** F16 has 2^-11 relative precision (~0.5-2 units at MRNF10k chunk edges, both on disk and in the GPU repack path). Replace with snorm16 (i16) page-local positions: uniform absolute error = half_extent/32768, 16× better than f16 at edges. Also permanently kills the ±65504 wall for scenes wider than 65K units (splat_v2).

**Disk format change (version-gated, keep old reader):**
- [ ] `converter.ts`: Pack positions as `round(x_norm * 32767)` where `x_norm = (x - chunk_center) / chunk_half_extent[d]`. Store as i16 in `xyz_uncompressed` (change from f16).
- [ ] Manifest: add `"position_format": "snorm16"` field (old files default to "f16").
- [ ] `worker.ts` decode: If `position_format === "snorm16"`, unpack i16 → f32 via `value / 32767`, then `world = value * chunk_half_extent[d] + chunk_center[d]`.

**GPU path change:**
- [ ] `SplatPager.ts processFetched()`: After `to_packedsplats` produces world-space f16 positions, normalize back to page-local snorm16 using `(pos - chunkCenter) / chunkHalfExtent`. Pack into same bit slots.
- [ ] `splatDefines.glsl`: Add `decodeSnorm16(uint val)` function. Replace `unpackHalf2x16(word1)` and `unpackHalf2x16(word2 & 0xffffu)` with `decodeSnorm16(word1 & 0xFFFFu)`, `decodeSnorm16((word1 >> 16u) & 0xFFFFu)`, `decodeSnorm16(word2 & 0xFFFFu)`.
- [ ] `SparkRenderer.ts`: Add `pageCenters: vec3[]` and `pageScales: vec3[]` uniform arrays (max 256 entries). Update each frame from `cachedMeta.chunks[].aabb`.
- [ ] `splatVertex.glsl`: After position decode, apply `worldPos = pageLocal * pageScales[pageIdx] + pageCenters[pageIdx]`. Page index = `gl_InstanceID >> 16` (from the paged_index in the instance data) or derived from `splatIndex / 65536`.
- [ ] Compose `(page_center - camera_position)` in JS `float64` each frame, upload as camera-relative offsets to avoid GPU float32 jitter.

**Acceptance:**
- [ ] `render_readiness_test.ts` extended: assert max `|decoded - source|` position error per chunk ≤ 2 * half_extent / 32768
- [ ] No packed GPU position equals ±65504 on splat_v2
- [ ] User-verified sharpness improvement on MRNF10k

---

### Step 4 — P0 Fixes ❌ NOT STARTED

**Intent:** Defer background prefetch until camera-fit settles (startup 4801ms → ~1100ms), and route prefetch through the same 3-active fetch cap as traversal-driven fetches.

- [ ] `examples/viewer/index.html`: Move `backgroundPrefetch` flag check from `loadSplatFile` to AFTER `onSplatLoad` resolves (camera settled). Set `spark.pager.autoDrive = true` + call `driveFetchers()` only after camera-fit completes.
- [ ] `src/SplatPager.ts runBackgroundPrefetch()`: Enqueue at most `numFetchers - activeFetchers` prefetch entries per call, instead of saturating all 3 slots immediately.

**Acceptance:** Startup with prefetch ON matches prefetch OFF (~1100ms). Log shows prefetch fetches starting AFTER `[camera-fit] settled`.

---

### Step 5 — P3 Mobile + C2 ❌ NOT STARTED

**P3a (mobile budget):**
- [ ] `SparkRenderer.ts defaultSplatTarget()`: Return 500K for `isMobile()` (was 1.5M)
- [ ] `SparkRenderer.ts lodRenderScale` default: 1.0 for desktop, 1.5 for mobile
- [ ] Adaptive: if scene half-extent < 1000 units, multiply `pixelScaleLimit` by 2.0 (small scene → higher threshold → fewer splats)

**P3b (SH banding):**
- [ ] `SplatPager.ts`: At page-load time, compute `ℓ(d) = clamp(ceil(3 * d_ref / d), 0, maxSh)` where `d_ref = scene_half_extent / 8`. Only decode the needed SH bands into GPU memory.
- [ ] Re-evaluate ℓ(d) on page touch (when camera moves, check if a page now needs different SH bands).

**P3d (importance pruning):**
- [ ] `converter.ts`: Before `tiny_lod`, score each splat as `importance = opacity * sqrt(area)`, drop bottom 30% (configurable).
- [ ] Report node-count reduction and PSNR-proxy stats.

**P3c (C2 MLP — GATED):**
- [ ] Requires user to provide a Flux-GS trained model checkpoint
- [ ] Offline Python exporter emitting `app_codebooks` + `mlp_dc`/`mlp_sh` weights
- [ ] Decode side already exists in `reconstruct_sp5_chunk` (line ~866 branch)

---

## 3. Known Bugs

| # | Bug | Status | Fix |
|---|---|---|---|
| 1 | `| 0xFF000000` corrupts output indices | ✅ Fixed (Step 1) | Reverted |
| 2 | Dead-zone may not be in user's build | ⚠️ Unconfirmed | Rebuild + hard-reload |
| 3 | Traversals continue at ~320ms cadence when stationary | ⚠️ See #2 | Dead-zone fix should eliminate |
| 4 | `repair_lod_cut` not called from JS | ⚠️ Pending Step 2 | SparkRenderer wiring |
| 5 | f16 clamp at ±65504 on GPU pack path | ⚠️ Pending Step 3 | snorm16 |
| 6 | Background prefetch slows startup 4.5× | ⚠️ Pending Step 4 | Defer prefetch |
| 7 | 5M splat budget excessive for MRNF10k | ⚠️ Pending Step 5 | P3a |

---

## 4. Current Test Status (10 tests)

| Test | Status |
|---|---|
| `sp5_huffman_differential_test` | ✅ PASS |
| `sp5_tree_reachability_test` | ✅ PASS |
| `sp5_ordering_test` | ✅ PASS |
| `sp5_chunk_coherence_test` | ✅ PASS |
| `sp5_hysteresis_stability_test` | ✅ PASS (incl. regression) |
| `sp5_cross_chunk_test` | ✅ PASS |
| `sp5_hierarchical_cross_chunk_test` | ✅ PASS |
| `sp5_hierarchical_lod_test` | ✅ PASS |
| `sp5_scale_diagnostic` | ✅ PASS |
| `smoke_test` | ✅ PASS |
| `render_readiness_test` | ✅ PASS |
