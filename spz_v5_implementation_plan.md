# SPZ-v5 (.sp5) Implementation Plan v10 — Grounded in Real Flux-GS/MobileGS Source

> **Revision note:** v1 of this plan was written against citations that didn't hold up — a
> mischaracterized PR, a non-existent file path, and a GPCC-in-the-browser step it never actually
> designed. This revision is grounded in the two real source trees now available locally:
> `C:\splat\pipeline\Flux-GS` (the offline training/compression pipeline, repo name `Flux-GS`,
> author `xiaobiaodu`) and `C:\splat\pipeline\flux-gs-project` (the **actual published WebGL
> reference renderer**, a fork of `antimatter15/splat`). The second tree is the one that matters —
> it is a complete, working, client-side decoder for everything this plan needs, and it changes
> the design substantially. See "What changed from v1" at the end.

> **v3 note (external review triage):** An external agent reviewed v2 and raised four points.
> Two were verified true against the source and are folded in below: the `xyz` encoding is a
> float16↔uint16 bitcast (not AABB quantization — §2.1), and Track A was duplicating
> infrastructure that already exists (`SplatPager.driveFetchers`, `lod_tree.rs`'s
> `traverse_lod_trees`/`behindFoveate` — Track A section rewritten). Two were **not** folded in
> after verification: the review's "dimension mismatch" and "MLP_offset baked away" findings quote
> phrasing from v1's old Part 1 §3, which v2 had already replaced — v2's §2.1/§2.2 already state
> the 96→13 `MLP_cont` compression, the unisphere contraction, and `MLP_offset`'s on-device,
> decode-time evaluation correctly. Don't assume a review caught everything or nothing; each claim
> here was re-checked against the actual files before being accepted or discarded.
>
> **v3 also changes delivery:** this lands as **one PR**, not two. See Section 1.

---

## 0. Legal gate — resolve before writing any code

Neither `Flux-GS` nor `flux-gs-project` ships a `LICENSE` file. No license means default
copyright: Spark (MIT, World Labs Technologies) **cannot vendor or port their code/WASM binary**
without explicit permission from the author. This blocks two concrete things this plan wants to
do:

1. Vendoring `flux-gs-project/tools/tmc3.wasm` / `tmc3.js` (an Emscripten build of the MPEG G-PCC
   reference codec, `tmc3`) and shipping it to end users from inside a Spark example/app. (Part B4
   settled on running this in the JS worker rather than porting it to Rust — see Part B4 step 2 —
   but the legal question is identical either way: it's the same compiled artifact being
   distributed to users either way.)
2. Porting the decode math in `flux-gs-project/render_shared/main.js` (VQ/Huffman reconstruction,
   the TCNN-style MLP evaluator) into `spark-lib`.

Separately, `tmc3` itself is MPEG reference software (`MPEGGroup/mpeg-pcc-tmc13`); its own license
terms need to be checked independent of `xiaobiaodu`'s repo, since MPEG reference codecs have
historically carried redistribution terms beyond plain open source.

**Action item:** contact `xiaobiaodu` for permission/clarification before Part B4 starts, and pull
the actual `mpeg-pcc-tmc13` license text to confirm it's compatible with an MIT project shipping a
compiled artifact to end users. If permission isn't available, Part B4 falls back to building
G-PCC from the public `mpeg-pcc-tmc13` source directly under its own license, which is a separate
(and probably larger) effort than reusing the existing WASM build.

**Second, separate gate: patent risk, not just copyright (adversarial review, correctly flagged as
missing here).** The above only covers copyright/redistribution terms. G-PCC is a standards-track
MPEG codec (ISO/IEC 23090-9), and standards-track MPEG codecs (H.264, H.265/HEVC, H.266/VVC) have
a well-documented history of separate patent-pool licensing (MPEG LA, Access Advance, Via LA)
independent of any given implementation's software license — a permissively-licensed decoder can
still trigger royalty obligations for whoever distributes it commercially. Checked via web search
specifically for this: no evidence of an *established, organized* G-PCC-specific patent pool yet
(unlike H.264/H.265's well-known administrators), but individual patents covering specific G-PCC
coding techniques are actively being filed — e.g. a September 2024 US application,
["Planar and Azimuthal Mode in Geometric Point Cloud Compression"](https://patents.justia.com/patent/20240303869),
confirms patent activity in this exact space. **"No pool yet" is not the same as "no risk"** —
newer standards often see licensing pools organize only after adoption grows, which cuts the other
way from reassuring. This is not something to resolve by more reading; it needs actual patent
counsel. **Action item, added to the same gate as the copyright check above:** before Part B4
ships G-PCC decode to end users in a commercial product (Spark, World Labs Technologies), get a
patent-risk review from qualified counsel, not just a license-text read. Treat this as a hard
blocker on Part B4 alongside the copyright question, not a nice-to-have.

---

## 1. Scope split: two independent tracks

The four goals in the request split cleanly along a line the codebase already draws:

| Goal | Depends on new codec? | Track |
|---|---|---|
| Faster loading (LOD by default camera) | No — pure pager/prioritization | **A** |
| Faster navigation (preload nearby chunks) | No — pure pager/prioritization | **A** |
| ≤2x `.sog` size | Yes — needs the neural codec | **B** |
| Reproduce Flux-GS compression quality | Yes — needs the neural codec | **B** |

Spark already has a working chunked LOD/paging system — `rust/spark-lib/src/chunk_tree.rs`,
`quick_lod.rs`, `tiny_lod.rs`, `bhatt_lod.rs`, view-dependent traversal/prioritization in
`rust/spark-rs/src/lod_tree.rs`, and a budget-aware fetch scheduler in `SplatPager.driveFetchers()`
— that operates on today's formats (`.spz`, `.sog`, `.ksplat`). **Track A turned out to be smaller
than it first looked**: once that existing machinery was verified (see the Track A section below),
its job shrank to an IndexedDB cache layer plus a gaussian-budget/local-file extension of that
cache, not new prioritization logic. It needs zero new file format and is the lower-risk half of
the single PR (Section "Delivery" below) — implemented and validated first, but not shipped
separately.

Track B (the actual SPZ-v5/.sp5 codec) is the speculative, higher-risk piece.

**Delivery: this is one PR, now with three tracks, not two.** Track C (below) was added after this
section was first written — a browser-based conversion tool (upload `.sog`/`.splat`, compress,
download/view), explicitly requested to drive adoption. It splits further into C1 (SVQ-only,
buildable now) and C2 (full SVQ+MLP, requiring genuinely novel client-side training infrastructure
that doesn't exist anywhere today). All three tracks land together; build/validation order is:

1. **Track A** first (lowest risk, gives everything after it a working cache/prioritization
   baseline).
2. **Track B**'s benchmark spike (Part 7) before its full Rust decoder is built — unchanged from
   before.
3. **Track C1** (shares infrastructure with Track B — chunking, SPZ v4 packaging — low
   incremental risk once B is working).
4. **Track C2's spike** (Part 7) before committing to shipping it — the highest-risk, most novel
   piece in the whole plan, gated the same way Track B is.

If Track B's kill criterion fails: ship Track A (+ Track C1, which doesn't depend on Track B's
codec working) with Track B and C2 descoped. If Track C2's kill criterion fails: ship everything
else with C2 disabled/experimental in the UI. Neither failure blocks the other — they're
independent gates on the same PR, not a single pass/fail gate. What does *not* happen: shipping a
half-validated codec or a training mode that silently produces bad output, just to hit a deadline.

---

## 2. What the real reference implementation actually does

Read in full from `flux-gs-project/render_shared/main.js` (2307 lines) and
`flux-gs-project/tools/`. This is the ground truth — not the training repo's offline compression
utilities, which are a *different* pipeline (more on that below).

### 2.1 Wire format (today: one monolithic file per scene, e.g. `garden.json`)

Despite the `.json` extension it's a binary container, parsed by `parseMobileGSFile`
(`main.js:622`):

```
[uint32 LE: jsonLength][JSON metadata tree][binary blob region, byte-addressed by offset/length]
```

The JSON tree is a recursively-reconstructed object whose leaf nodes are tagged
`{_type: "ndarray", dtype, shape, offset, length}` or `{_type: "bytes", offset, length}`, each
pointing into the trailing binary region. This is structurally the same idea as `.sog`'s
manifest-plus-blobs layout (see `rust/spark-lib/src/sogs.rs`) — good, it means Spark already has
a Rust-side pattern (`SogsDecoder`) to imitate rather than invent.

Top-level fields actually present in the file (confirmed from the field names the decoder reads):

- `xyz`: raw G-PCC bitstream bytes. **Not** AABB min-max quantization (v1 and an earlier draft of
  this section both described it that way, incorrectly). The actual encode path
  (`Flux-GS/scene/gaussian_model.py`, `encode()`, confirmed by reading it directly) is:
  `float16_to_uint16(self.get_xyz.half())` → `calculate_morton_order(...)` to sort →
  `compress_gpcc(xyz_uint16)`, where `float16_to_uint16`/`calculate_morton_order` live in
  `Flux-GS/utils/gpcc_utils.py:283` and `:271`. `float16_to_uint16` is a **bit-reinterpretation**
  of the IEEE half-float bit pattern with the sign bit XOR'd (`^ 0x8000`) so that the resulting
  uint16 values sort in the same order as the original floats — this is what makes G-PCC's
  integer octree codec usable on float data at all, and it's also why the decode side
  (`main.js:955-958`) XORs back and reinterprets as half-float instead of doing any inverse
  AABB/lerp math. **The exporter must replicate this exact bit-level transform, not a
  min-max-normalize-to-uint16 formula** — they produce different bit patterns and a decoder doing
  the AABB-inverse on bitcast data would scramble every coordinate.
- `scale_index` / `scale_htable` / `scale_code`: per-codebook-level VQ index Huffman bitstream(s),
  Huffman table(s), and centroid array(s) — decoded by `decodeVQAttributesConcat` (`main.js:714`)
- `rotation_index` / `rotation_htable` / `rotation_code`: same, for rotation
- `app_index` / `app_htable` / `app_code`: same, for a 6-dim "appearance" latent (3 dims feed a
  "space" MLP, 3 feed a "view" MLP — matches v1's Part 1 §3 description, that part was correct)
- `MLP_cont` (96→64→13, frequency-encoded position → continuous feature), `MLP_opacity` (16→64→1),
  `MLP_dc` (16→64→3), `MLP_sh` (16→64→9): flat, 16-aligned TCNN-style fully-fused MLP weights
- `MLP_offset`: a *separate*, conventionally-shaped small PyTorch MLP (named layers
  `main.0/2/4`, `shs_output.0`) that predicts a residual correction to DC+SH from
  `[normalized SH(12), opacity(1), normalized scale(3), xyz(3), rotation(4)]` (23-dim input)

**Crucially: the codebooks and all 5 MLP weight sets are scene-global, not per-point.** They are
small (a handful of KB to perhaps low hundreds of KB total — needs exact measurement, see Part 7)
and paid for once per scene load, never duplicated per point. This is the single most useful fact
for hitting the size goal: as long as chunking keeps codebooks/MLPs global-once instead of
duplicating them per chunk, per-chunk payload is just `(G-PCC geometry for that chunk's points) +
(Huffman-coded VQ indices for that chunk's points)` — both of which scale with point count, not
chunk count.

### 2.2 Decode pipeline (today: runs once, synchronously, for the whole scene)

1. Whole file `fetch()`'d (one HTTP request, `main.js:1489`), handed to a Web Worker.
2. Worker boots `tmc3.wasm` via Emscripten (`tools/tmc3.js`, 78KB glue + 1.2MB wasm), writes the
   `xyz` bitstream into Emscripten's virtual FS, invokes `callMain(['--mode=1',
   '--compressedStreamPath=/xyz.bin', '--reconstructedDataPath=/xyz.ply'])` — i.e. it **literally
   runs the `tmc3` CLI binary inside WASM**, then reads the reconstructed PLY back out
   (`main.js:865-959`). **This confirms G-PCC-in-the-browser is not a hypothetical — it's proven,
   working code today**, which overturns the "infeasible to port tmc3 to WASM" objection raised
   against v1 of this plan. The remaining open question is licensing (Section 0), not feasibility.
3. VQ attributes (scale/rotation/appearance) are Huffman-decoded and codebook-expanded in plain
   JS (`decodeVQAttributesConcat`).
4. **Per splat, once** (not per frame): position → "contract to unisphere" → 16-frequency
   sin/cos positional encoding (96-dim) → `MLP_cont` → 13-dim feature → `MLP_opacity`/`MLP_dc`/
   `MLP_sh` → opacity + DC color + 1st-order SH → `MLP_offset` residual correction added on top.
   This resolves a real risk v1 didn't address: it would have been catastrophic if "view-dependent"
   meant re-running 5 MLPs per splat *per frame*. It doesn't — it bakes per-splat SH coefficients
   once at decode time, same as any other compressed format. The result is packed into the
   **standard 32-byte antimatter15/splat layout** (pos f32×3, scale f32×3, rgba u8×4, quat u8×4)
   plus a separate 48-byte/splat SH buffer — i.e. it terminates in the same kind of flat packed
   buffer Spark already knows how to render. From this point on, real-time performance is a
   property of Spark's existing WebGPU rasterizer (already proven fast for `.sog`/`.spz`/`.ksplat`
   point counts), not of anything Flux-GS-specific. **Goal 4 ("performance-oriented, reproducible")
   should be reframed**: decode efficiently into Spark's standard packed format, and the existing
   renderer carries the rest. Trying to "reproduce Flux-GS performance" as if it were a property of
   a separate renderer is the wrong framing — Flux-GS's own renderer here is a single-pass WebGL2
   shader with CPU worker-based depth sort, materially simpler (and on large scenes, likely slower)
   than Spark's tiled WebGPU pipeline.
5. **There is no chunking, no LOD, no progressive loading, no cache anywhere in this reference
   implementation.** One file, one fetch, full decode, then render. This confirms Track A
   (progressive LOD streaming) is genuinely new work Spark has to design — it isn't hiding
   somewhere in the reference code waiting to be ported.

### 2.3 Relationship to the Flux-GS training repo

`Flux-GS/utils/gpcc_utils.py` and `compress_utils.py` (cited correctly, line numbers checked) are
the **offline storage-benchmark pipeline** — they shell out to a native `tmc3` binary via
subprocess and are used to produce the paper's compressed-size numbers (`comp.xz`). They are not
what ships to the browser. v1 conflated these two pipelines: it designed Part 1/Part 2 (the
exporter) around the training repo's offline tools without checking what the actual web delivery
format looks like. v2 below designs the exporter to produce the *real* wire format from §2.1,
using the training repo's existing SVQ/MLP-extraction code as the source of the numbers, but
restructured for chunking.

---

## Track A: Pager & Cache (build first, lands in the same PR)

**Rewritten after the external review.** v2's draft of this section invented a JS-side
`THREE.Frustum` culler and a `ChunkDownloader` concurrency queue. Verified against the actual
code: both already exist, more capably than what was proposed.

- View-dependent LOD selection is **not** a JS/frustum problem to (re)solve — `traverse_lod_trees`
  / `dynamic_traverse_lod_trees` in `rust/spark-rs/src/lod_tree.rs:412,632` already do this in
  Rust, called from `worker.ts`'s `traverseLodTrees()` and `SparkRenderer.ts:1458,1538`, driven by
  the real `view_to_objects` camera transform (not just a static default-camera approximation).
  `SparkRenderer` already exposes `behindFoveate`/`coneFoveate` (`SparkRenderer.ts:248-252`,
  default `behindFoveate = 0.2`) specifically to deprioritize/cull LOD nodes outside the view cone
  or behind the camera. Building a second, JS-side, default-camera-only frustum culler on top of
  this would be redundant and would fight the Rust traversal's own prioritization, not complement
  it.
- Bounded-concurrency fetching is **not** a new scheduler to build — `SplatPager.driveFetchers()`
  (`SplatPager.ts:1033`) already pulls from `this.fetchPriority` under a `numFetchers`-bounded
  pool (`SplatPager.ts:511,534,619`). A second `ChunkDownloader` queue inside `fetchDecodeChunk`
  would race/conflict with it.

What's actually new, given that infrastructure already exists:

1. **Confirm, don't rebuild, the "default camera first" property.** Check that whatever
   constructs the initial `SparkRenderer`/`SplatMesh` for a scene runs its first
   `traverseLodTrees()` pass against the intended default camera pose *before* the user can move
   (i.e. the camera the app sets at startup, not an identity/zero transform) — if so, goals 1–2
   ("load LOD near default camera first") are close to free, since `fetchPriority` is already
   derived from that traversal. If not, that's a small, targeted fix at the call site, not new
   culling logic.
2. **Tune `behindFoveate`/`coneFoveate` defaults** (or expose them more prominently in examples)
   for the "fast initial load, then fill in" use case — e.g. a more aggressive default
   `behindFoveate` for first load that relaxes once initial chunks are in, rather than a single
   static value. This is a parameter-tuning + possibly a small scheduling-phase change, not new
   architecture.
3. **`src/SplatPager.ts`**: the only genuinely new code is a cache intercept inside
   `fetchDecodeChunk` (line 170, verify against current `main` before patching): check
   `SplatCache.get()` before issuing the network fetch, return the cached bytes on a hit, write to
   cache on a miss — sitting *underneath* `driveFetchers`' existing concurrency control, not
   beside it.
4. **`src/utils.ts`**: `SplatCache` wrapping `indexedDB`, schema as v1/v2 proposed plus one field
   needed for the eviction policy in item 5 below: `chunks: {spotId, chunkIndex, lod, spzBytes,
   byteLength, cachedAt, lastAccessedAt}`, `manifests: {spotId, manifest, cachedAt}`. Index on
   `lastAccessedAt` so eviction queries (oldest-first) don't need a full table scan.

This is a meaningfully smaller diff than v2 proposed, which is the right direction for a track
that's meant to be the low-risk half of a single PR. It also means Track B's chunks get
prioritized loading "for free" once exported, *if* Part B2's chunking reuses the same LOD-tree
representation `chunk_tree.rs`/`lod_tree.rs` already use (see Part B2 step 4) — one fewer thing
to special-case for the new format.

### Track A, item 5: configurable gaussian budget + togglable background prefetch

New requirement: dropping a large local file (e.g. 50M splats) should load and render partially,
respecting a configurable max-gaussian budget, with IndexedDB caching staying on but *not*
eagerly walking the whole scene into the cache in the background — while a scene that already
fits under budget (2–5M) should just load and cache fully, with no further network/disk activity
on revisit.

**This is mostly already-existing behavior, not new infrastructure — verify that before building
anything.** Read `SplatPager.driveFetchers()` (`SplatPager.ts:1033-1099`) directly: it already
walks `fetchPriority` and **only issues a fetch if `numPages < this.maxPages`**
(`SplatPager.ts:1041,1062`), where `maxPages = ceil(maxSplats / PAGE_SPLATS)`
(`SplatPager.ts:612-613`). Pages beyond budget go to an `overflow` list and get LRU-tracked
(`SplatPager.ts:1099` onward) for eviction/replacement as priority shifts. **In other words, Spark
already does not do unbounded full-scene background fetching today** — fetch volume is already
bounded by `maxSplats`, gated by priority, every time. This reframes the requirement: the risk
isn't "how do we turn off an existing eager-download behavior," it's "make sure the *new*
`SplatCache` layer (Track A items 3–4) doesn't introduce one." If `SplatCache.get/put` is wired
into `fetchDecodeChunk` as already planned — caching only what `driveFetchers` actually decided to
fetch — it inherits this exact same priority+budget discipline for free, with zero new code. That
should be the default, and it already satisfies most of this requirement:

- 50M-splat scene, `pager.maxSplats = 5,000,000`: only ever fetches/decodes/caches chunks up to
  that 5M budget, prioritized by the existing LOD/view traversal. No "full scene download," today,
  by default.
- 2–5M-splat scene, same budget: everything fits, gets fetched+cached once, nothing left to fetch
  on revisit (0 network bytes) — this is the same behavior Part 7 §5 already verifies, it just
  needs to also be true when `numSplats` is *under* the budget, not only when a manifest happens
  to match it exactly.

What's actually new:

1. **A separate disk-cache budget, `maxCacheSplats` (or bytes), distinct from `pager.maxSplats`.**
   GPU-resident budget and IndexedDB budget are different constraints (disk is cheaper than VRAM)
   — a user may want a bigger disk cache than GPU budget, so panning back to a previously-visited,
   GPU-evicted area is a cache hit instead of a re-decode. Default `maxCacheSplats` to
   `pager.maxSplats` (cache mirrors GPU residency, safest/simplest) but make it independently
   configurable. LRU-evict cached chunks (not currently GPU-resident) when a write would exceed
   it — same LRU concept `driveFetchers`' `overflow` handling already uses for GPU pages, applied
   to the IndexedDB layer.
   **Eviction scope, refined:** enforce `maxCacheSplats` **globally across the whole IndexedDB
   store** (all `spotId`s a user has ever visited), not per-scene — otherwise total on-disk usage
   is unbounded across a session that visits many scenes, which is the same failure mode this
   whole feature exists to prevent. Use the `lastAccessedAt` field added to the `chunks` schema
   above: on a write that would exceed budget, query oldest-`lastAccessedAt` entries *across all
   `spotId`s* and evict until under budget, rather than scoping the query to the current scene.
   One real trade-off worth naming rather than glossing over: a global policy means a user who
   alternates between two scenes that together exceed `maxCacheSplats` will see each visit evict
   the other's cache (thrashing, falling back to network/decode every time) — a per-scene reserve
   would avoid that at the cost of the unbounded-total-usage problem this is meant to solve. Global
   LRU is the right default; if thrashing between a small working set of scenes turns out to matter
   in practice, a small per-`spotId` floor (e.g. "never evict the most-recent scene's chunks to
   serve a different scene's prefetch") is the natural follow-up, not a v1 requirement.
2. **An explicit, opt-in `backgroundPrefetch: boolean` flag, default `false`.** This is the *only*
   thing that should walk the remaining manifest and proactively fetch+cache chunks beyond what
   current view priority demands — for users who deliberately want to warm an entire scene into
   IndexedDB ahead of time (e.g. "make this scene available offline"). Off by default means the
   default experience for a dropped 50M-splat file is exactly the budget-bounded, priority-driven
   behavior above with no extra flag needed — "turn off full-scene background download" is the
   default, not something to disable.
3. **Local file support — gated on whether the dropped file is already chunked.** This is the part
   that's genuinely new and not free:
   - If the dropped file is an already-chunked format (`.sp5` per Part B1, or pre-chunked `.spz`
     from `build-lod --spz-chunked`), introduce a small `ChunkSource` abstraction with two
     backends — the existing `fetch()`-based one for URLs, and a new `Blob.slice()`-based one for
     local `File` objects (the File API supports reading arbitrary byte ranges without loading the
     whole file into memory, so this is a real, modest addition, not a workaround). Once chunk
     reads are abstracted this way, every mechanism above (priority, `maxSplats`, `maxCacheSplats`,
     `backgroundPrefetch`) applies identically whether the source is a URL or a dropped file.
   - If the dropped file is an **unchunked monolithic format** (raw `.ply`/`.spz` with no LOD
     structure — which is what "drop a 50M splat file" most naturally implies, since most files
     in the wild aren't pre-chunked), none of the paging system applies yet, because there's
     nothing to page: today, loading such a file means decoding it in full into one buffer.
     Reaching true "partial load, budget-respecting" behavior for this case requires
     **client-side LOD-chunk construction at drop time** — exposing `chunk_tree.rs`/`quick_lod.rs`
     spatial partitioning (the same logic `build-lod` already uses offline, see Part B2 step 4) as
     a `wasm-bindgen` function. This is real, new, non-trivial scope — call it out explicitly as
     such rather than assuming it falls out of existing infrastructure. Recommend the same
     scope-down as Part B2 step 4: spatial-bucket chunking only (uniform density,
     nearest-chunk-first), not point decimation, to keep this tractable within the PR.
     **Must not run synchronously on the main thread** — parsing + recursively binning 50M points
     is multi-second work, and blocking the main thread for that long risks the browser's
     unresponsive-page warning, not just a dropped frame. This isn't new infrastructure to build,
     though: `SplatPager` already routes WASM-heavy work through a worker pool —
     `src/SplatWorker.ts`'s `SplatWorker`/`NewSplatWorkerPool`/`workerPool` (confirmed real;
     `SplatPager.ts:9,239` already does `workerPool.withWorker(...)`, and `SparkRenderer.ts` uses
     the same pattern for `traverseLodTrees`/`newLodTree` via `worker.call(...)`). The dropped
     `File` should go to a pool worker, which runs the spatial-bucket WASM call and returns a
     lightweight in-memory chunk index (offsets within the still-worker-resident decoded buffer,
     or back via transferable `ArrayBuffer`s) — the main thread then requests chunk ranges from
     that worker the same way it requests them over `fetch()` or `Blob.slice()`, keeping this
     consistent with the `ChunkSource` abstraction above rather than a third, special-cased path.

This sub-track is the one place where this requirement adds real, non-trivial new scope to
Track A — flagging that plainly rather than presenting it as free, since Track A was otherwise
meant to be the low-risk half of the PR.

---

## Track B: The .sp5 Neural Codec

### Part B1: Real wire-format spec for `.sp5`

Generalize the single-scene container from §2.1 into a chunked one, keeping the same
JSON-header-plus-blob idea (proven, and structurally close to `.sog`):

- **`manifest.json`** (downloaded once, first): scene AABB, chunk list (`{id, lod, aabb, byteLength,
  url}`), and the **global, scene-wide payload**: `scale_code`/`rotation_code`/`app_code`
  (codebook centroid arrays) and `scale_htable`/`rotation_htable`/`app_htable` (Huffman tables,
  pulled out of per-chunk files and made global — the reference implementation only ever has one
  "chunk" so it never had to make this choice; for N chunks, a global table avoids re-paying
  Huffman-table overhead per chunk) and the 5 MLP weight blobs (`MLP_cont`, `MLP_opacity`,
  `MLP_dc`, `MLP_sh`, `MLP_offset`). Get an actual byte count for this manifest before assuming
  it's small (Part 7).
- **`scene-lod-N-chunkM.sp5`**: per-chunk file containing only `xyz` (that chunk's points,
  independently G-PCC-encoded — `tmc3` encodes whatever point set it's given, so partitioning
  before encoding is a data-flow change, not a codec change) and Huffman-coded VQ index streams
  for that chunk's points (decoded against the *global* tables/codebooks from the manifest).

This directly targets the "≤2x `.sog`" goal: pay the codebook+MLP cost once, not per chunk. Confirm
the actual sizes before promising a ratio — see Part 7.

**Why chunking doesn't introduce neural "seams" (question worth answering explicitly, not leaving
implicit).** A generic risk with spatially-chunked neural scene representations is that per-chunk
local models disagree at boundaries (visible lighting seams), or per-chunk pruning leaves physical
gaps. Checked directly against `main.js`'s decode loop (`main.js:1042-1176`): every MLP call —
`MLP_cont`, `MLP_opacity`, `MLP_dc`, `MLP_sh`, `MLP_offset` — takes only *that splat's own*
position/appearance-latent/attributes as input. No neighbor lookups, no spatial-window inputs,
nothing chunk-relative. Combined with codebooks and MLP weights already being scene-global (not
per-chunk, per §2.1) and chunking being scoped to uniform-density spatial partitioning with no
per-chunk decimation (Part B2 step 4), there is no per-chunk-local state that could disagree at a
boundary, and no independent per-chunk pruning to leave gaps. Two adjacent points on either side of
a chunk boundary get bit-identical treatment regardless of which chunk file they're shipped in.
This is a property of the design choices already made here, not something requiring the overlap
padding / cross-chunk blending some other chunked-neural-representation systems need — don't add
that complexity, it would be solving a problem this design doesn't have.

**Note on SPZ v4 — status updated, now the intended foundation, not a deferred future direction.**
[PR #332](https://github.com/sparkjsdev/spark/pull/332) ("Add SPZ v4 (NGSP / ZSTD multi-stream)
read and write support") is real, checked directly via `gh pr view 332`: **open, not yet merged**,
author `udwinj`, reviewer `mrxz` requested. It adds the v4 container — 32-byte `NGSP` header,
per-attribute ZSTD streams via `@bokuweb/zstd-wasm`, TOC — to `src/spz.ts`/`SplatLoader.ts` and
bumps `SPZ_VERSION` to 4 on write, matching upstream `nianticlabs/spz`'s wire format exactly. This
is the real, in-progress version of the prerequisite v6 of this plan said was "unscoped" — good,
but read the PR's own "Known gaps (out of scope)" section carefully before assuming it's done:

- **"Extensions (`FlagHasExtensions = 0x2`): not read or written. Reader correctly skips over
  them."** This is the exact mechanism this plan wants for global Flux-GS metadata (codebooks,
  MLP weights) — confirmed via the [real extension spec](https://github.com/nianticlabs/spz/blob/main/extensions/README.md)
  to be file-level-metadata-only, a good fit for that data. PR #332 does not add it. **A follow-up
  scoped task — reading and writing extension records in `spz.ts`/`spz.rs` — is still required**,
  either as part of this PR (built on top of #332's branch) or immediately after.
- **"SH degree 4 ... pre-existing spark limitation, unchanged."** Worth noting this likely doesn't
  block this plan specifically: Flux-GS's whole design point is to *avoid* shipping raw SH4 data
  on the wire (§2.1 — the base layer is 1st-order SH, which Spark already fully decodes). SH4
  support only matters here if a future design wants an additional "no neural decode, raw SH4"
  fallback tier, which isn't part of the current `.sp5` design at all.

**Practical consequence for this plan:** treat "build on top of PR #332" as the real, current
plan of record for `.sp5`'s container (superseding the bespoke JSON-manifest-plus-blobs design in
this Part B1 as the default), with the caveat that extension read/write support is a genuine
additional piece of scope this PR needs to add, not something to assume already works. Land or
rebase onto #332 first (checking its merge/review status before branching, since it's still under
review as of this writing), then add extension support, then build the chunked `.sp5`
container on top of that. This is a larger foundation than v6 assumed, but a *better* one — real
interop with any SPZ v4 reader for the base layer, rather than a format only Spark's own custom
decoder can open.

1. Train normally (unchanged, CUDA, offline).
2. Run `apply_svq` (`Flux-GS/scene/gaussian_model.py:844`) **once over the whole point set** —
   centroids are already scene-global by construction (k-means over all points), so this requires
   no change to get a shared codebook.
3. Extract the 5 MLP state dicts (`mlp_dc` etc., `gaussian_model.py:745` and neighbors) — also
   already scene-global.
4. **New step, not in either source repo**: partition points into spatial chunks. Don't invent
   this from scratch. **Concrete mechanism (external review proposal, verified and corrected):**
   `rust/build-lod` is a real, already-built CLI binary with a confirmed `--spz-chunked` flag
   (`rust/build-lod/src/main.rs:411,522`). The proposed pipeline: write the trained Gaussian
   centers (position only) to a temp `.ply`, run unmodified `build-lod --spz-chunked` on it to get
   spatially-coherent chunk files, then use a `scipy.spatial.KDTree` to match each output center
   back to its original index in the trained model so the right VQ indices/appearance latents get
   written into that chunk's `.sp5` file. This needs **zero modifications to the Rust CLI** — a
   real advantage over inventing a new binding.

   **But this only works at the finest LOD level, and the plan needs to say so explicitly.**
   `build-lod`'s coarser levels are produced by `quick_lod.rs::compute_lod_tree`, which calls
   `splats.new_merged(indices, merge_step)` (`quick_lod.rs:88,160`, read directly) — coarser
   nodes are **synthesized new centers from merged groups of splats**, not copies of original
   points. A KDTree "nearest center" lookup against a merged node doesn't recover "the original
   splat's VQ index" because there is no single original splat — it would silently return the
   *closest* original point's attributes, which is wrong, not approximately right. There's also
   no way to merge two points' VQ codebook indices into a valid merged index without first
   dequantizing through the MLP/codebook stack, which defeats the compression and isn't something
   `build-lod` does anyway.

   **Scope decision for this PR:** use `build-lod --spz-chunked` only for its **finest-level**
   output to get spatial chunk-id assignment + AABBs (where the KDTree match is exact, since
   leaf-level positions pass through unmodified). Treat `.sp5` "LOD" in this PR as **uniform-density
   progressive spatial streaming** — nearer/coarser-grained chunks load first, all at full point
   density — not true multi-resolution point decimation. Reaching real decimated LOD for the
   neural codec (merging splats *before* SVQ/MLP fitting, or learning a coarser model per level)
   is a separate, harder research problem and should be explicitly out of scope here rather than
   silently assumed solved by reusing `build-lod`'s merged levels.
5. Per chunk: subset the already-computed VQ index arrays and `xyz` positions (already
   `float16_to_uint16`-bitcast per §2.1) by chunk membership, then re-run the **chunk-local sort**
   before its own `gpcc_encode` call (the reference sorts the whole scene once before its single
   call; chunking means this has to happen per chunk). **Naming correction (external review,
   verified):** this is not a bit-interleaved Morton/Z-order code despite the function name.
   `calculate_morton_order` (`gpcc_utils.py:271`, read directly) does
   `argsort(x @ [1, M, M², ...])` where `M = x.max() + 1` after subtracting the per-axis min —
   a lexicographic sort with Z-axis highest priority, then Y, then X (confirmed it matches the
   manual 3-way comparator in `main.js:975-985`, which compares Z, then Y, then X directly, not a
   bit-interleave). Implement the same lexicographic comparator, not a real Morton coder — they
   produce different orderings and the VQ index streams must line up with whichever ordering the
   decoder assumes.
   Call the existing `gpcc_encode` (`Flux-GS/utils/gpcc_utils.py:14`) on just that subset, and
   Huffman-encode (`compress_utils.py:152`) the subset's indices against the *global* table built
   in step 2.
6. Emit `manifest.json` + per-chunk `.sp5` files per the Part B1 layout.

### Part B3: `src/spz.ts` — simpler than v1 proposed

v1's plan to special-case `writeSpz`/`transcodeSpz` with an `if (packedSplats contains SVQ/MLP
data)` branch is unnecessary. `decode_to_gsplatarray` (`rust/spark-rs/src/lib.rs:295`) already
dispatches purely on `SplatFileType`
(`rust/spark-lib/src/decoder.rs:330`, `MultiDecoder::push` sniffs magic bytes / extension and
picks a decoder — see `SogsDecoder`/`SpzDecoder`/`KsplatDecoder` registration at
`decoder.rs:529`). Adding `.sp5` support means:

- add `SplatFileType::SP5` + extension mapping (`"sp5"`) in `decoder.rs`
- add a magic-byte or extension sniff branch in `MultiDecoder::push`
- register a new decoder struct in `new_decoder()`

No changes to `spz.ts` itself are needed beyond what `getSplatFileType`/`getSplatFileTypeFromPath`
already provide generically. This removes an entire (incorrect) sub-task from v1.

**Concrete registration points (confirmed by reading both sides):**
- JS: `src/defines.ts` has the `SplatFileType` enum and `getSplatFileType`/
  `getSplatFileTypeFromPath` (confirmed at `defines.ts:342,390`, used by `SplatLoader.ts`). Add an
  `SP5` member, an `.sp5` extension mapping, and a magic-byte sniff.
  **Magic value, corrected:** a later review proposed `SP5_MAGIC = 0x355a5053` (bytes `"SPZ5"`
  read little-endian) and justified it by claiming `.spz` files start with the raw ZSTD frame
  magic `0x28B52FFD`. That justification is wrong for this codebase — checked
  `rust/spark-lib/src/spz.rs:11` directly: Spark's `SPZ_MAGIC` is `0x5053474e` ("NGSP"), and per
  `decoder.rs:474-498` the actual outer bytes a `.spz` file starts with are the **gzip** magic
  (`GZIP_MAGIC = 0x00088b1f`, `decoder.rs:461`) — `SPZ_MAGIC` is only checked *after* gunzipping.
  The real set of magics a new value must avoid colliding with (all confirmed by reading the
  source): `PLY_MAGIC = 0x00796c70` (`ply.rs:9`), `GZIP_MAGIC = 0x00088b1f`, the zip/`PK` magic
  `0x04034b50` (used for `.sog`), `RAD_MAGIC = 0x30444152` and `RAD_CHUNK_MAGIC = 0x43444152`
  (`rad.rs:24-25`). `0x355a5053` doesn't collide with any of these — the proposed value is fine,
  the stated reasoning for it just needs to cite the right facts.
- Rust: mirror in `rust/spark-lib/src/decoder.rs`'s `SplatFileType` enum
  (`decoder.rs:330-387`, `to_enum_str`/`from_enum_str`/`from_extension`/`from_pathname`), add the
  same magic-byte branch in `MultiDecoder::push` (`decoder.rs:463-519`), and register `Sp5Decoder`
  in `new_decoder()` (`decoder.rs:529`) alongside `SogsDecoder`/`SpzDecoder`/etc.

### Part B4: Rust decoder — port the proven JS reference, don't redesign it

`rust/spark-lib/src/sogs.rs` is the right template: a manifest-driven decoder implementing the
`ChunkReceiver` trait, producing `SplatProps` batches for a generic `SplatReceiver`
(`rust/spark-lib/src/decoder.rs:15-170`). Build `Sp5Decoder` the same way:

1. Parse `manifest.json` (serde, like `sogs.rs`'s `PcSogsRoot`/`PcSogsV2` structs).
2. **G-PCC decode: keep `tmc3` in the JS Web Worker, don't try to call it from Rust WASM.**
   (External review recommendation — sound, and it removes an open architectural fork from this
   plan.) Calling a separately-compiled WASM module from inside Spark's own WASM module
   (component-model-style composition) is poorly supported across browser engines today; porting
   ~tens of thousands of lines of MPEG reference C++ to Rust is its own large project. Instead:
   keep the existing `tmc3.wasm`/`tmc3.js` Emscripten approach running in `worker.ts` exactly as
   `flux-gs-project` already does it, decode `xyz` to a flat coordinate buffer there, then hand
   that buffer plus the VQ/Huffman streams to the Rust/WASM side for the parallelizable,
   performance-critical work (VQ reconstruction, MLP evaluation). **This does not remove the
   Section 0 legal gate** — using `tmc3.wasm` at all, from JS or from Rust, still needs the
   licensing question resolved; it only removes the harder of the two technical paths.

   **Correction (adversarial review, sound, fits this handoff cleanly): the lexicographic re-sort
   belongs on the Rust side of this JS→Rust boundary, not in the JS worker.** The buffer `tmc3`
   hands back is in G-PCC's internal octree traversal order, not the lexicographic Z→Y→X order
   the VQ index streams were laid out in at encode time (§Part B2 step 5) — recovering that
   alignment needs the same re-sort `main.js:975-985` does (`indicesArray.sort((a,b) => ...)`,
   a custom-comparator sort over the full point count). Running `Array.prototype.sort()` with a
   JS callback over millions of points in a worker is genuinely slow (large-N comparator-based
   sorts in JS are known to be a real bottleneck — easily multiple seconds, plus GC pressure from
   the callback closures) and would sit right on the critical path to first-splat. Since this
   buffer is already being hop-passed to Rust/WASM in this same step for VQ/MLP work, do the
   re-sort there instead — `Vec::sort_by` with an equivalent comparator, immediately after
   receiving the coordinate buffer and before step 3 consumes it. **Correctness requirement, not
   just a perf note:** the comparator must be a faithful port (same Z/Y/X priority, same
   tie-breaking) and must use a **stable** sort to match `Array.prototype.sort()`'s guaranteed
   stability (ES2019+) — `sort_by` (stable) is the correct choice here, not `sort_unstable_by`,
   even though the latter is faster, since a tie-breaking mismatch would misalign VQ indices with
   positions for any duplicate-coordinate points. This is a real, valuable optimization but not a
   new architecture — it's already the natural place for this work given B4's own JS-decodes/
   Rust-processes split above.
3. Huffman-decode + codebook-expand VQ attributes — this is a faithful, mechanical port of
   `decodeVQAttributesConcat`/`decodeHuffmanBitstream` (`main.js:669-745`), straightforward Rust.
4. Port the MLP evaluator — `runTCNN_MLP`/`runPyTorch_MLP`/`getTCNNFrequencyEncoding`
   (`main.js:778-862`) are plain dense matmuls with two activation functions; this is a direct,
   low-risk port (no `nalgebra` needed, flat `f32` slices and loops suffice, as the JS reference
   itself demonstrates). **Concrete landmine to carry over (external review, verified at
   `main.js:800-814`):** TCNN's fully-fused MLPs pad every dimension up to a multiple of 16
   (`pad16 = ceil(val/16)*16`) and index the flat weight array using the *padded* strides, not the
   logical `inDim`/`hiddenDim`/`outDim`. `getTCNNFrequencyEncoding` pads its output the same way.
   The Rust port must replicate this padding exactly — indexing with unpadded strides won't error,
   it'll silently read the wrong floats and produce garbage colors/opacity. This applies to
   `MLP_cont`/`MLP_opacity`/`MLP_dc`/`MLP_sh` (the TCNN-style ones) but not `MLP_offset`, which
   uses plain unpadded PyTorch `Linear` layers (`runPyTorch_MLP`) — don't pad that one.
   Doing this in Rust/WASM instead of plain JS (as the reference does) should be a real
   performance win, worth measuring (Part 7).
5. Emit `SplatProps` batches into the receiver exactly as `sogs.rs` does, reusing the existing
   32-byte-equivalent pipeline already wired into Spark's renderer — no new GPU-side format needed.

This replaces v1's vague "implement client-side MLP decoder using nalgebra" with a concrete,
de-risked port of code that is already proven to work in a browser.

### Part B5: Viewing/streaming example page

Largely as v1 Part 6 described (lil-gui session stats, stats.js FPS, predefined POV camera) — no
changes needed there; it was the most solid part of v1. Two additions, both needed to actually
demonstrate (and let Part 7 verify) what Track A item 5 added:

- Surface per-chunk vs. manifest byte counts in the telemetry panel, since that's the number
  Part 7 needs to validate the size goal.
- A drop target plus `lil-gui` controls for `maxSplats` (GPU/`pager`), `maxCacheSplats` (disk),
  and the `backgroundPrefetch` toggle, with live readouts of: total scene splat count, resident
  (GPU) splat count, cached (IndexedDB) splat count, and whether the current scene is fully
  resident (`numSplats <= maxSplats`) — this is the only way to actually see Track A item 5's
  behavior (budget-bounded partial load vs. full-scene fit) rather than just asserting it works.

This page is now the demonstration for both Track A and Track B's *reading* side — Track A's
budget/cache behavior needs to be visibly exercised against a real (or synthetic, oversized) scene
before Track B's codec is layered on top, otherwise Part 7's benchmark spike has no working
baseline to compare against, per Section 1. The *writing*/conversion side (upload → compress →
download/view) is a separate, larger piece — see Track C below.

---

## Track C: Browser-Based Conversion Tool (drives adoption)

This is the concrete feature the user asked for: a page where anyone can drop a `.sog`/`.splat`
file, convert it in a background worker with a progress bar, and get back a downloadable (and/or
directly viewable) chunked, compressed file — no Python, no CUDA, no server round-trip. The point
is adoption: someone can try the format in one click, without setting up the offline `Flux-GS`
pipeline first.

**Explicit scope decision (user directive, both modes ship):** the conversion UI exposes a
dropdown with two modes, not one:

1. **SVQ-only** — real, buildable now, low technical risk.
2. **Full Flux-GS (SVQ + MLP)** — genuinely novel, needs its own validation spike before shipping,
   parallel to how Part 7 already gates Track B's codec. Presenting these as options with clearly
   different risk profiles rather than silently treating them as equally solid, since they are not.

### C0: What "convert" means for both modes, and the constraint that shapes everything else

**A `.sog`/`.splat` upload is a *baked* Gaussian splat — final per-point attributes only (position,
scale, rotation, opacity, existing SH coefficients). It is not a trainable checkpoint and carries no
multi-view training images.** This matters a lot: the offline `Flux-GS` pipeline (Part B2) fits its
MLPs and appearance latents via **photometric loss against multi-view renders** — gradient descent
comparing rendered pixels to real training photos. That's not available here; there's no camera
poses, no source images, just the already-baked splat. So Track C's "Full Flux-GS" mode cannot be
"run the same training Part B2 runs, just client-side" — it has to be a different, more limited
training objective: **self-distillation**. Fit the MLPs (and per-point appearance latents) via
regression against the *upload's own existing* per-point DC/SH/opacity/scale/rotation values
(minimize MSE between MLP output and the value already baked into the file), not against rendered
images. This is a legitimate, known class of technique (closely related to auto-decoder /
VQ-VAE-style compression: fit continuous per-point latents + a shared decoder, then quantize), but
it is compressing *the splat's own existing representation*, not re-deriving quality from original
photos — expect it to reproduce, not improve on, whatever the source file already looked like.

### C1: SVQ-only mode (build now)

1. Parse the uploaded `.sog`/`.splat` in a Web Worker (reuse the `workerPool`/`SplatWorker`
   infrastructure from Track A item 5 — same pattern, same reason: this must not run on the main
   thread for a multi-million-point file).
2. Run k-means (256 centroids, matching `Flux-GS/scene/gaussian_model.py`'s `kmeans` — confirmed
   real, `gaussian_model.py`, read directly) on scale and rotation vectors to build
   `scale_code`/`rotation_code` codebooks + per-point indices. Deterministic, no gradient descent,
   feasible in-worker for millions of points in low single-digit seconds — this is the same
   algorithm class already used for the offline exporter, just reimplemented in JS/WASM instead of
   Python/cuML.
3. Leave existing SH/color/opacity **as-is** (or truncate to 1st-order if higher degrees are
   present and the user wants smaller output) — no appearance-latent MLP fitting in this mode.
4. Spatially partition into chunks (reuse the same finest-level spatial-chunking approach as
   Part B2 step 4, client-side).
5. Package as: per-chunk SPZ v4 files (base attributes — position, opacity, existing/truncated SH,
   built on PR #332's writer) + scale/rotation codebooks and VQ indices carried via the extension
   mechanism (Part B1's "Note on SPZ v4" — this is exactly the file-level-metadata use case
   extensions are for) once that follow-up work exists, or a small sidecar manifest in the interim
   if extension read/write isn't ready yet when this ships.
6. Progress bar: driven by real phase boundaries (parse → k-means → chunk → pack), not a fake
   timer — each phase reports real progress since none of steps 2, 4, 5 are black boxes.

This mode's risk profile is close to Track A/B2's existing chunking work — no new algorithm class,
just a client-side port of a known-simple one.

### C2: Full Flux-GS mode (SVQ + MLP, client-side training — novel, needs a spike)

**Nothing like this exists today** — not in Spark, not in `Flux-GS`, not in `flux-gs-project`.
Every piece of MLP evaluation this whole plan has verified so far (§2.2, Part B4) is *inference*:
run a fixed, pre-trained set of weights forward. This mode requires *training* those weights from
scratch, in the browser, per uploaded file. Concretely, on top of C1's steps:

1. Initialize a per-point continuous appearance latent (6-dim, small random init — an auto-decoder
   embedding table, `numPoints × 6` floats, a few tens of MB for a multi-million-point scene, kept
   in a GPU buffer) and randomly initialize the 5 small MLP weight sets (`MLP_cont`, `MLP_opacity`,
   `MLP_dc`, `MLP_sh`, `MLP_offset` — same architecture as Part B4 documents, since the Rust
   decoder needs to be able to read whatever this produces).
2. Train via minibatch gradient descent (Adam) over points: forward pass per §2.2 step 4's
   pipeline, loss = MSE against that point's own existing DC/SH/opacity (per C0), backprop through
   the MLPs and the per-point latent, for a fixed iteration/time budget.
3. After training, k-means-quantize the fitted per-point appearance latents into `app_code`/
   `app_index` (same codebook mechanism as C1's scale/rotation, applied to the *learned* latent
   instead of an existing attribute).
4. Package identically to C1, plus the trained MLP weight blobs (global, via the extension
   mechanism, same reasoning as C1 step 5).

**This needs a forward+backward autodiff/training implementation that doesn't exist anywhere in
this codebase — realistically WebGPU compute shaders for the matmuls/backprop, since plain JS/WASM
gradient descent over millions of points would likely be too slow for an interactive "progress
bar" experience.** That's a genuinely large, novel engineering task on its own, separate from
everything else in Track B, which only ever needed *inference* code ported from a working
reference. **Update: partial reference material exists after all** — three local repos
(`C:\splat\pipeline\repos\{WebDGS,webgpu-torch,tfjs}`) were checked line-by-line against a
research document's citations; every specific claim (function names, line numbers) verified
exactly correct, which is notably better-grounded than prior review documents in this plan's
history. Recorded below with corrections to two of the document's *recommendations* (the citations
were right, the proposed reuse wasn't always the right shape):

- **Adam optimizer — real, reusable, but as a reference, not a graft target.**
  `WebDGS/src/shaders/adam.wgsl`'s `adam_step(param, grad, m, v, lr) -> vec3<f32>` (confirmed at
  line 53) is a correct, generic, standard Adam update rule with no problem-specific coupling —
  genuinely reusable *math*. But reading `webgpu-torch/src/optim.ts` directly (not just citing it)
  shows a cleaner integration point than "grafting" WebDGS's shader in: `Optimizer`'s base class
  already has `state: Map<Tensor, {...}>` (line 16) — exactly the per-parameter `m`/`v` moment
  storage Adam needs — and `SGD extends Optimizer` (confirmed real at line 66, and confirmed no
  `Adam` class exists anywhere in that file, matching the document's "SGD only" claim) shows the
  extension shape: subclass `Optimizer`, implement `step()`, call a numeric update function.
  **Recommendation: implement `Adam extends Optimizer` natively inside webgpu-torch's own
  tensor-op abstraction** (composing existing elementwise ops — the same way PyTorch's own
  reference `torch.optim.Adam` is implemented, not a hand-written kernel), using WebDGS's
  `adam_step` purely as a **numerical cross-check** for correctness, not as code to paste in — the
  two codebases have different kernel-dispatch conventions and splicing a foreign raw WGSL shader
  into webgpu-torch's tensor/kernel system is more work and more risk than writing five lines of
  tensor arithmetic against an abstraction that already has the right extension point.
- **Loss gradients — WebDGS is proof-of-feasibility, not a porting target.** `loss.wgsl`'s
  `compute_loss_grad` (confirmed real at line 86) and `tiled-backward-rasterize.wgsl`'s
  `backward_rasterize_main` (confirmed real at line 35) are genuine, working WebGPU compute
  shaders — but reading `loss.wgsl` directly shows they operate on `texture_2d<f32>` (L1/L2/DSSIM
  loss between a *rendered image* and a *ground-truth photo*, `pred_texture`/`target_texture`,
  lines 15-16). That's differentiable-rasterization photometric loss — the exact thing §C0
  established C2 *cannot* do (no source images from an uploaded `.sog`/`.splat`). C2's actual loss
  is much simpler: elementwise MSE between an MLP's per-point output and that same point's own
  existing baked attribute — no rasterization, no textures, no tiling, just two flat per-point
  buffers. **Recommendation: don't port these shaders — write a purpose-built ~10-line WGSL MSE
  kernel instead.** What WebDGS is genuinely worth is a *confidence signal*: it demonstrates a
  full working WebGPU gradient-descent training loop (Adam + backward rasterization) running in a
  browser against a 3DGS-adjacent problem that's strictly *harder* than C2's (image-space loss
  through a differentiable rasterizer, vs. C2's flat per-point regression) — meaning the Part 7
  step 9 spike's real open question narrows from "does browser-based gradient training work at
  all" (WebDGS already answers that: yes) to "does *this specific* self-distillation regression
  converge well enough, fast enough." That's a meaningfully lower-risk starting point than v7 had.
- **webgpu-torch — good prototyping scaffold for the MLP layers themselves.** `nn_basic.ts`'s
  `Linear extends Module` (confirmed real at line 10) and the tensor `backward()` machinery
  (`tensor.ts:370`, and the `IAutoFunction.backward(ctx: GradientContext, outputGrad: Tensor)`
  interface method — confirmed at `autograd.ts:37-40`, a multi-line signature, not a bare
  top-level function, worth noting since a plain function-name grep misses it) are a legitimate,
  reusable pattern for building `MLP_cont`/`MLP_opacity`/`MLP_dc`/`MLP_sh`/`MLP_offset` as stacks
  of `Linear` + activation, matching Part B4's already-documented architecture. Worth prototyping
  against directly to validate the training math before committing to anything leaner.
- **TF.js — not recommended, a real cost the source document didn't weigh.** `AdamOptimizer`
  (`adam_optimizer.ts:35`) and `Dense` (`core.ts:188`) are real and mature, but TF.js
  (`tfjs-core`+`tfjs-layers`+`tfjs-backend-webgpu`) is a multi-MB general ML framework for training
  five MLPs with a combined parameter count in the low thousands, gated behind an opt-in dropdown.
  Spark's own [PR #374](https://github.com/sparkjsdev/spark/pull/374) explicitly removed code
  specifically to shrink bundle size — pulling in TF.js for this directly contradicts that
  established priority. Not adopting this option.
- **Recommended final shape — revisited against a counter-argument, and the counter-argument's own
  key number doesn't hold up.** A later adversarial review pushed back on "hand-roll it," arguing
  hand-written WGSL forward+backward+Adam for 5 MLPs is bug-prone and tedious (fair point, taken
  seriously below) and proposed instead compiling in a tree-shaken slice of webgpu-torch,
  estimating "likely <100KB after tree-shaking." **That estimate wasn't verified before being
  stated, so it was checked here rather than accepted:** `webgpu-torch/src` is 1.4MB of TypeScript
  (611KB across non-test files, `du`/`wc` run directly), including `ops_opgen.ts` (1301 lines) and
  `functions_opgen.ts` (2200 lines) — code-generated comprehensive op tables (the `_opgen` naming
  says so directly), plus dedicated ONNX-import and diffusion-model modules. Op-table/kernel-
  registration patterns like this are notoriously *resistant* to tree-shaking (bundlers eliminate
  statically-unreferenced code; a generated dispatch table that registers every op by string/enum
  key generally isn't statically-unreferenced from the bundler's point of view, even if a given
  app only calls a handful of those ops at runtime). No `dist/` build artifact exists in the repo
  to check an actual bundled number against. **"<100KB" is not established — it's a plausible-
  sounding guess that doesn't match what the source structure suggests, and it was being used to
  justify reversing an architecture decision.** That said, the underlying engineering-risk
  argument (hand-rolled backprop for 5 MLPs is genuinely easy to get subtly wrong — gradient sign
  errors, wrong activation derivatives, wrong Adam bias-correction terms) is legitimate on its own
  merits, independent of the bundle-size number attached to it.

  **Resolution: don't decide this from an unverified estimate either direction — measure it.**
  Add a small, cheap pre-spike, before committing engineering time to either path: actually build
  webgpu-torch's real `webpack` config (`npm run build`, already defined in its own `package.json`)
  importing only what C2 needs — `Tensor`/`Module`/`Linear`/core autograd, not ONNX, not
  diffusers, not the full op table — and read the real tree-shaken output size, rather than
  trusting either "<100KB" or the alternative assumption that it must be huge. If the measured
  number is genuinely small (low hundreds of KB or less) for an *opt-in, dropdown-gated* feature,
  taking the dependency is a reasonable trade against hand-rolled-WGSL bug risk. If it's not,
  fall back to the hand-rolled, purpose-built WGSL pipeline for the five known fixed MLP shapes
  (matching Part B4's existing inference-side philosophy — flat `f32` loops, no external tensor
  framework, since the architectures are fixed and known in advance), prototyped and
  numerically-validated against webgpu-torch and WebDGS's `adam_step` per the two bullets above,
  but not shipped as a runtime dependency. Either way, prototype against webgpu-torch first — that
  part of the original recommendation stands regardless of which way the bundle-size measurement
  comes out.

**New addition (adversarial review) — worth adding on its own merits, but not "already in the
plan" as claimed, and one part of its framing needed correcting.** The review's point 4 described
"the plan's" §C2 step 3 as already recommending training the global MLP on a downsampled ~100k-point
subset, then freezing it and fitting only per-point appearance latents for the full point count —
and called this "highly coherent." **Checked directly: no version of this document mentions
100k-point downsampling, freezing, or a two-phase training split anywhere before this edit** (a
grep for "100k", "downsampl", "freeze"/"frozen" across the whole file turns up nothing relevant).
This is the same pattern several earlier review documents in this plan's history fell into —
describing a design choice as if it already existed here. Evaluating it as the new proposal it
actually is:

- **The core idea is sound and worth adopting.** Fitting the shared MLP weights against a
  representative subset (not the full 5M+ points) keeps the expensive part of training — updating
  weight tensors, which is what actually needs the full Adam optimizer state and gradient
  computation for *every* parameter — fast and cheap, then fitting only the 6-dim per-point latent
  for the remaining points is a much smaller per-point optimization problem (6 scalars vs. the
  full weight set). This is a legitimate, standard-shape trade (closer to how VQ-VAE / auto-decoder
  training splits shared-decoder fitting from per-sample latent fitting) and fits directly into
  C2's existing design.
- **One phrase needs correcting before anyone implements this: fitting per-point latents against a
  *frozen* MLP is not "a simple, linear forward-only backpass."** It's still a full forward **and**
  backward pass through the frozen network — gradients still have to flow backward through every
  frozen layer via the chain rule to reach the latent inputs; freezing the weights means the
  optimizer doesn't *update* them (skip weight-gradient accumulation and skip Adam moment-buffer
  storage for the weights), not that backward computation through those layers is skipped
  entirely. Scoping this as "forward-only" would lead an implementer to build the wrong thing —
  the compute/memory savings are real (no weight-gradient accumulation, no `m`/`v` buffers for the
  frozen weights, only for the latents), but a real backward pass through the frozen MLP is still
  required every step.
- **One risk worth naming, not just endorsing:** whether a 100k-point subsample is *representative*
  of the full scene's appearance/material variety is scene-dependent, and this matters more than
  it looks — if some material or region isn't well-represented in the subsample, the frozen MLP's
  decoder function may not have the right basis to reconstruct it well, and since the MLP is
  frozen before the remaining ~4.9M+ points are fit, the 6-dim-per-point latent optimization has
  no way to compensate for that later (6 free scalars per point can't fix a decoder that fundamentally
  doesn't have the right shape for a given material). Reasonable default: stratified or
  farthest-point sampling for the subset rather than uniform random, to bias toward coverage over
  raw count — cheap to do, meaningfully reduces this risk, and should be validated as part of
  Part 7 step 9's spike (does quality hold up on a scene with genuinely diverse materials/regions,
  not just an easy uniform case) rather than assumed.

**Gate, parallel to Track B's benchmark spike (Part 7) — still required, but now a narrower
question.** With WebDGS as a working existence-proof and webgpu-torch as a prototyping scaffold
(both above), "can WebGPU gradient-descent training run acceptably in a browser at all" is
answered (yes, for a harder problem than C2's). The spike's real job is checking C2's *specific*
regression setup, not re-litigating browser training feasibility from zero. Before committing to
shipping C2, spike it against a real test scene and check, concretely:
- Does training converge to acceptable visual quality (compare against the source `.sog`/`.splat`
  directly, plus against what the offline Part B2 pipeline produces for the same scene if
  available) within a time budget a "progress bar" UX can credibly support (tens of seconds to a
  couple of minutes, not longer) on realistic consumer hardware (not just a high-end dev GPU)?
- Does it degrade gracefully for scenes where it *doesn't* converge well (partial/noisy scenes,
  unusual geometry) — e.g. falling back to C1's SVQ-only output rather than shipping a visibly
  broken result?
- **Kill criterion:** if training quality or time budget can't be made acceptable within a scoped
  spike, ship Track C with **only C1 available**, and mark "Full Flux-GS" as disabled/experimental
  in the UI rather than shipping a mode that produces bad results silently. This mirrors Track B's
  existing kill-criterion pattern (Part 7) — don't let the riskiest, least-verified piece block or
  silently degrade the rest of the PR.

---

## Part 7: Verification — benchmark spike *before* full build-out

v1's verification step only checked cache-hit mechanics, never the actual novel claims. Reorder:

0. **This is a gate on the one PR, not on a release.** Per Section 1, Track A should be working
   on the branch first; the spike below runs before Part B4's Rust decoder is fully built, and a
   failed kill criterion means descoping Track B from the PR rather than merging an unvalidated
   codec.
1. **Spike first, before Part B4 is fully built**: export one real scene (e.g. `garden`, since
   reference data for it already exists at the HuggingFace path
   `mobile-gs2/mobile-gs2/resolve/main/garden.json` the reference viewer fetches from) through the
   Part B2 exporter, single chunk (no partitioning yet), and measure:
   - total manifest + codebook + MLP weight bytes (the fixed overhead)
   - bytes per chunk at varying chunk sizes
   - compare total against a `.sog` export of the same scene
   - this is the **kill criterion**: if fixed overhead or per-point cost can't get within ~2x of
     `.sog` for realistic chunk counts, stop and re-scope before writing the Rust decoder.
2. **Compile/license check**: confirm Section 0 is resolved before Part B4 lands any G-PCC code.
3. **Rust WASM build**: `npm run build:wasm` (the canonical entrypoint wired into
   `package.json`, → `rust/build_wasm.js`; `rust/build_rust_wasm.sh` also exists in the repo but
   isn't what's actually invoked by the npm scripts — use the npm script).
4. **Dev server**: `npm run dev`, navigate to `/examples/cached-streaming/`.
5. **Track A correctness**: refresh → 100% cache hit, 0 network bytes, confirm via the telemetry
   panel — unchanged from v1, this part was fine.
6. **Track A budget/local-file correctness (new in v4, not previously verified anywhere in this
   plan — closing that gap here):**
   - Set `maxSplats`/`maxCacheSplats` below a test scene's total count; confirm via the telemetry
     panel that resident and cached splat counts stay at-or-under budget, never grow past it as
     the camera moves, and that `backgroundPrefetch` defaults to `false` (no network/cache
     activity for chunks outside current priority without the flag set).
   - Use a scene whose total splat count is *under* budget; confirm it fully loads and caches,
     and a refresh afterward shows 0 network bytes — the same check as step 5, but specifically
     exercising the "small scene fits entirely, no further cycling" case from the original
     requirement, which is a different code path (budget never binds) than step 5's check.
   - Drop a local file: for an already-chunked format, confirm partial load respects budget the
     same as a network-sourced scene. For an unchunked monolithic file, confirm the page stays
     responsive during parse/partition (no "page unresponsive" browser warning) — this is the
     concrete, observable check for the worker-pool requirement in Track A item 5.
   - Visit two scenes that together exceed `maxCacheSplats`; confirm eviction is global (the first
     scene's cached chunks get evicted to make room for the second's), and note the
     expected-and-accepted thrashing behavior from Track A item 5's trade-off discussion — this
     should be observed and confirmed *expected*, not mistaken for a bug.
7. **Track B correctness**: compare rendered output against the `flux-gs-project` reference
   viewer for the same scene (visual diff, not just "it renders something") since that's the only
   available ground truth for whether the port is faithful.
8. **Track C1 correctness**: convert a real `.sog`/`.splat` through the SVQ-only path, confirm the
   round-tripped output renders correctly and the k-means codebook sizes/compression ratio are
   reasonable — same "measure, don't assume" discipline as step 1, just for a different pipeline.
9. **Track C2 spike (its own kill criterion, separate from Track B's):** per the gate described in
   Track C2, run client-side training against a handful of real test scenes covering a quality/
   complexity range (not just one easy case), and check convergence quality, wall-clock time on
   realistic (not top-tier) consumer hardware, and graceful-degradation behavior for scenes that
   don't converge well. This spike should happen early enough that a negative result (ship C1 only,
   disable C2) doesn't waste the engineering time of building the full WebGPU training pipeline
   first and finding out afterward. Two sub-checks added after the second adversarial review,
   both cheap to do early and both able to change the shape of C2 before the expensive part of
   the work starts:
   - **Bundle-size pre-spike:** actually build webgpu-torch's real webpack config with only the
     core tensor/autograd/`Linear` pieces imported, and read the real tree-shaken size, before
     deciding between "take the dependency" and "hand-roll WGSL" — don't decide from an estimate
     in either direction (see the webgpu-torch bundle-size discussion above).
   - **Subsample representativeness check:** if the 100k-point-subsample-then-freeze training
     shape (above) is adopted, verify quality holds up on a scene with genuinely diverse materials
     spread unevenly across space, not just an easy uniform-material test case — this is exactly
     the failure mode a single easy test scene would hide.

---

## What changed from v1 (summary)

- Corrected two false citations (PR #374's actual purpose; the real `spz.rs` path under
  `spark-lib`, not `spark-rs`).
- Found and read the actual browser reference implementation (`flux-gs-project`), which v1 didn't
  know existed — it answers most of v1's open design questions directly instead of needing fresh
  invention.
- Replaced "GPCC decode is unaddressed / likely infeasible in WASM" with "GPCC decode is proven
  and working today, gated on a real but solvable licensing question."
- Replaced "evaluate 4 MLPs, possibly per-frame" risk with confirmation it's decode-time-once,
  baked into a standard packed-splat buffer — real-time performance is inherited from Spark's
  existing renderer, not a new concern.
- Replaced the `spz.ts` special-casing sub-task with the existing generic `SplatFileType`
  dispatch — less code, less risk.
- Replaced "implement MLP decoder using nalgebra" (unscoped) with "port the ~150 lines of proven
  JS MLP/VQ decode logic" (scoped, low-risk).
- Added the global-codebook/global-MLP/per-chunk-only-geometry design, which is the concrete
  mechanism for hitting the size goal — v1 asserted the goal without a mechanism.
- Split the plan into Track A (ship now, no new format, addresses 2 of 4 goals) and Track B
  (gated research spike, addresses the other 2) instead of one monolithic 7-part plan.
- Added an explicit licensing gate (Section 0) and moved the size/quality benchmark to *before*
  full implementation, with a stated kill criterion.

## What changed in v3 (external review, after verification against source)

Folded in (verified true):
- `xyz` encoding corrected from "AABB voxelization" to the actual mechanism: a float16↔uint16
  bit-reinterpretation with sign-bit XOR (`float16_to_uint16`/`calculate_morton_order`,
  `gpcc_utils.py:283,271`), confirmed by reading `gaussian_model.py`'s `encode()` directly. Flagged
  the corresponding chunk-local-Morton-sort wrinkle this creates for Part B2.
- Track A rewritten: removed a redundant JS `THREE.Frustum` culler and `ChunkDownloader` queue
  after confirming Spark already has both — `lod_tree.rs::traverse_lod_trees` +
  `behindFoveate`/`coneFoveate` for view-dependent LOD selection, and
  `SplatPager.driveFetchers()` for bounded-concurrency fetching. Track A's actual new surface is
  now just the IndexedDB cache intercept, which is a smaller, lower-risk diff.
- Delivery model changed to a single PR (explicit instruction), with the Track B benchmark spike
  now framed as a gate on that one PR rather than a reason to ship two PRs.

Checked and **not** folded in (claims didn't hold up against the current file):
- The review's "dimensionality mismatch" and "missing unisphere contraction" findings quoted text
  from v1's original Part 1 §3, which v2 had already replaced — v2's §2.1/§2.2 already state the
  96→13 `MLP_cont` compression and the contraction step correctly.
- The review's "`MLP_offset` baked away, missing from runtime" finding doesn't match v2 either —
  v2 already documents `MLP_offset` running on-device, per-splat, at decode time (§2.1, §2.2 step
  4, B4 step 4 names `runPyTorch_MLP` explicitly for it). Recorded here rather than silently
  dropped, so the next reviewer doesn't have to redo this check.

## What changed in v4 (second external review + budget/local-file requirement)

All five new nitpicks were checked against source before acting on them — four held up as
described, one (the KDTree chunking scheme) held up technically but needed a real correction:

- **Folded in as-is:** the `calculate_morton_order` naming fix (it's a lexicographic Z→Y→X sort
  via weighted-sum `argsort`, not bit-interleaved Morton codes — confirmed by reading
  `gpcc_utils.py:271` directly); the TCNN 16-padding landmine in the MLP evaluator (confirmed at
  `main.js:800-814`); keeping G-PCC decode in the JS worker rather than attempting WASM-in-WASM
  composition; and the concrete `defines.ts`/`SplatLoader.ts`/`decoder.rs` registration points for
  `.sp5` (all confirmed to exist as described).
- **Folded in with a correction:** the `build-lod --spz-chunked` + KDTree matching scheme for
  exporter-side chunking is real and the flag exists, but the review's framing implied it solves
  LOD generally. It doesn't: `quick_lod.rs::compute_lod_tree` synthesizes merged centers at
  coarser levels (`new_merged`, confirmed at `quick_lod.rs:88,160`), which have no corresponding
  original point for a KDTree to match against. Scoped Part B2 step 4 down to finest-level spatial
  chunking only, with multi-resolution decimation for the neural codec explicitly called out as
  future work rather than silently assumed solved.
- **New requirement (budget + local file drop):** discovered `SplatPager` already enforces a
  GPU-resident budget and priority-gated, non-eager fetching today (`maxSplats`/`maxPages`,
  `driveFetchers` at `SplatPager.ts:1033-1099`) — this reframed the ask from "add a way to turn
  off background download" to "make sure the new cache layer inherits the budget discipline that
  already exists, and add an explicit opt-in for the one case that should bypass it." Added a
  `maxCacheSplats` disk-budget config (distinct from GPU `maxSplats`), a `backgroundPrefetch`
  flag defaulting to off, and scoped local-file support into two cases: already-chunked formats
  (a `Blob.slice()`-based `ChunkSource`, modest addition) vs. unchunked monolithic formats
  (real new scope — client-side LOD-chunk construction at drop time, called out explicitly rather
  than assumed free).

## What changed in v5 (adversarial review of v4's new additions + full coherence pass)

Three more claims checked against source (all about the *new* v4 additions, not the earlier
sections):

- **Folded in:** main-thread freeze risk for unchunked-file drop partitioning is real (multi-second
  synchronous WASM work on 50M points) — fixed by routing through the existing `workerPool`/
  `SplatWorker` infrastructure (`SplatWorker.ts`, already used by `SplatPager.ts:9,239` and
  `SparkRenderer.ts`'s `traverseLodTrees`/`newLodTree`), not a new worker mechanism. Global
  (cross-`spotId`) IndexedDB eviction folded in with a `lastAccessedAt` field added to the cache
  schema — also added an explicit note on the multi-scene-thrashing trade-off this creates, rather
  than presenting global LRU as strictly better with no downside.
- **Folded in with a correction:** the proposed `SP5_MAGIC = 0x355a5053` value is fine and was kept,
  but its stated justification was wrong — checked `spz.rs:11` directly and `.spz` files in this
  codebase are gzip-wrapped with inner magic `0x5053474e` ("NGSP"), not the raw ZSTD frame magic
  `0x28B52FFD` the review cited. Corrected the reasoning, kept the (still-valid, still
  non-colliding) conclusion.

Then a full top-to-bottom coherence sweep, independent of either review, since incremental edits
across three earlier rounds had left some real internal contradictions:

- Section 0's legal gate referenced "Part 4" twice — a v1 heading name that no longer exists
  after the doc was restructured into Track A/B. Fixed to "Part B4," and tightened item 1's wording
  (it still said "or re-deriving a Rust port from it," which Part B4 had since explicitly ruled
  out in favor of keeping `tmc3` in the JS worker).
- Section 1's Track A description still called it "camera-frustum-based fetch prioritization" and
  said it "can ship on its own, against real scenes, immediately" — both directly contradicted by
  later sections: the Track A rewrite (this prioritization already exists in Rust, nothing new to
  build) and the single-PR delivery decision. Rewrote the paragraph to match what Track A actually
  ended up being and how it actually ships.
- Part 7 verification never checked any of Track A item 5's new behavior (budget enforcement,
  `backgroundPrefetch` defaulting off, local-file drop, global cache eviction) — a plan that
  specifies a feature in detail but never specifies how to confirm it works is incomplete in a way
  that's easy to miss on a section-by-section read. Added a dedicated verification step (new step
  6) covering all four.
- Part B5's example page described the original lil-gui telemetry but had no UI for exercising
  budget/prefetch/drop at all, which made the new Part 7 step impossible to actually carry out.
  Added the drop target and budget controls/readouts needed to make that verification step
  real rather than aspirational.
- Bumped the document title from "v2" to "v5" to match the changelog history below it (it still
  said v2 after four more rounds of revision, which is exactly the kind of small staleness this
  pass is meant to catch — including, briefly, when this very edit first bumped it to "v4" while
  writing a "v5" changelog section, caught on re-read immediately after).

## What changed in v6 (another agent's report — SH4/SPZ-v4 and chunk-seam questions)

A third-party "technical report" raised four points. Given this document's history of catching
fabricated specifics from confident-sounding reviews, every checkable claim was verified before
acting — including, this time, a claim about an external, real-world spec (SPZ v4) that postdates
this assistant's training data, checked via live web search rather than either trusted or
dismissed on priors.

- **Verified real and folded in:** SPZ v4 exists (Niantic Spatial, confirmed via
  [nianticspatial.com/blog/spz4](https://www.nianticspatial.com/blog/spz4) and
  [github.com/nianticlabs/spz](https://github.com/nianticlabs/spz)) — SH degree 4, parallel ZSTD
  streams, removed splat-count cap, vendor extensions. Spark's `spz.rs` shares its magic bytes but
  caps at v3/SH3 today (`spz.rs:70,639`), unrelated to this plan but worth knowing.
- **Verified and corrected:** the report's proposed use of vendor extensions to carry "compact
  Flux-GS neural offsets" doesn't match the real spec — read
  [the actual extensions README](https://github.com/nianticlabs/spz/blob/main/extensions/README.md)
  directly and confirmed extensions are file-level metadata only, not per-splat arrays. Reframed as
  a good fit for this plan's *global* codebooks/MLP weights specifically, explicitly not a
  replacement for Part B1's per-chunk file delivery. Recorded as a real, better long-term direction
  but **not adopted for this PR** — it has its own large prerequisite (SPZ v4 support doesn't exist
  in Spark's core yet) that would blow past the single-PR scope.
- **Folded in (was already true, now stated explicitly):** the chunk-seam question the user asked
  about directly. Re-verified against `main.js`'s decode loop that every MLP call takes only
  per-splat inputs, no neighbor/cross-chunk references — combined with already-global codebooks/MLP
  weights and no per-chunk decimation, the generic "neural chunk seam" failure mode the report
  describes doesn't apply to `.sp5` as scoped. This was true of the design before this pass; it just
  hadn't been written down as an explicit answer.
- **Not folded in:** the report's Part 1 ("61% memory reduction" figure) and Part 2 ("multi-view
  alpha-based densification and pruning") are training-time claims not encountered anywhere in the
  actual `Flux-GS` source read across this whole plan, and not actionable for a plan whose exporter
  step 1 is explicitly "train normally, unchanged" — not worth chasing down further since nothing
  in this PR's scope depends on them either way.

## What changed in v7 (PR #332 verified real + new Track C, browser conversion tool)

Two inputs this round: a real, checkable PR link, and a new feature request whose scope turned out
to be much larger than its description suggested.

- **PR #332, verified via `gh pr view`/`gh pr diff` rather than taken on the user's description
  alone:** real, open (not merged), adds the SPZ v4 container to `spz.ts`/`spz.rs`. Its own
  "Known gaps" section states extensions and SH degree 4 are explicitly out of scope for that PR —
  read that section carefully rather than assuming "adds SPZ v4 support" meant "adds everything
  this plan wants from SPZ v4." Updated Part B1's SPZ v4 note from "verified real but deferred" to
  "the intended foundation, with extension read/write still a required follow-up" — a real status
  change, not a full reversal (extensions genuinely aren't there yet either way).
- **New Track C, added after questioning scope rather than guessing at it.** The request ("upload
  a file, convert with a worker + progress bar, get back our pre-chunked format") reads like a fast
  deterministic transcode, but the actual Flux-GS compression this plan documents requires
  gradient-based MLP training, which cannot run against an uploaded `.sog`/`.splat` the same way
  Part B2's offline pipeline does (no source images, no camera poses — just baked attributes).
  This is a large enough fork (a client-side transcoder vs. a client-side neural-network trainer
  are very different engineering problems) that it was surfaced as an explicit choice
  (`AskUserQuestion`) rather than assumed. Answer: build both, as a dropdown — SVQ-only (C1, real
  compression, deterministic k-means, buildable now) and full SVQ+MLP (C2, self-distillation
  training client-side, genuinely novel — no autodiff/training code exists anywhere in Spark or
  either reference repo today). Gave C2 its own kill-criterion spike (Part 7 step 9), parallel to
  Track B's, rather than bundling its risk into the rest of the PR.

## What changed in v8 (WebGPU training reference material, verified line-by-line)

A research document pointed at three local repos (`WebDGS`, `webgpu-torch`, `tfjs`) as reusable
building blocks for C2's training pipeline. Unlike prior review documents in this plan's history,
every specific citation checked out exactly on verification (function names and line numbers for
`adam_step`, `allocateOptimizerStateBuffers`, `Optimizer`, `compute_loss_grad`,
`backward_rasterize_main`, `Linear`, `backward`, `SGD`, `AdamOptimizer`, `Dense`,
`resizeBilinearConfig` — all confirmed real via direct file reads, not just trusted). The
citations being accurate didn't mean the *recommendations* built on them were automatically right,
though — reading the actual code shape around each citation surfaced two corrections:

- WebDGS's `adam.wgsl` is real, correct, reusable Adam math — but "graft it into webgpu-torch"
  undersold webgpu-torch's own architecture. Reading `optim.ts` directly showed `Optimizer`
  already has a per-parameter `state` map built for exactly this, and `SGD`'s structure shows the
  right extension pattern. Recommendation changed to "implement `Adam` natively inside
  webgpu-torch's tensor abstraction, using WebDGS's shader as a correctness reference," not a
  literal port.
- WebDGS's `loss.wgsl`/`tiled-backward-rasterize.wgsl` are real, working shaders — but reading
  `loss.wgsl` directly showed they operate on `texture_2d<f32>` (photometric image-space loss
  through a differentiable rasterizer), which §C0 already established C2 can't use (no source
  images from an uploaded file). Recommendation changed to "write a trivial custom per-point MSE
  kernel," with WebDGS's actual value reframed as feasibility proof (harder problem, already
  solved in-browser) rather than a porting target — which also meaningfully narrows what Part 7
  step 9's spike still needs to answer.
- Added a point the source document didn't raise: TF.js's bundle size directly contradicts
  Spark's own stated priority (PR #374's explicit bundle-size rationale) for training five tiny
  MLPs behind an opt-in dropdown — not adopting it, recommending a hand-rolled fixed-shape WGSL
  pipeline for the shipped version instead, prototyped against webgpu-torch but not dependent on
  it at runtime.

## What changed in v9 (adversarial review — two corrections accepted, two claims didn't hold up)

Explicitly not treating "adversarial review" framing as license to skip verification — the same
standard applied to every prior review in this document's history applies here too, including
when the framing pushes toward moving fast. Two of four points were sound and folded in directly;
two needed real correction before folding in.

- **Folded in directly:** moving the lexicographic position re-sort from the JS worker to Rust
  (Part B4 step 2). This is a genuine perf risk (large-N comparator sort in JS) with a clean fix
  that fits the *already-established* JS-decodes/Rust-processes handoff from v6/v7 — not new
  architecture, just correctly placing work that was already crossing that boundary. Added the
  stability requirement (`sort_by`, not `sort_unstable_by`) since that's a correctness issue, not
  just a style choice, and the review didn't specify it.
- **Folded in, broadened to match the actual risk:** G-PCC patent exposure. Section 0 previously
  covered copyright/redistribution only. Checked via web search rather than asserted either way —
  found no established G-PCC-specific patent pool yet, but found real evidence of active patent
  filings in this exact space (a Sept. 2024 US application on a specific G-PCC coding mode),
  enough to treat the concern as legitimate and add it as a hard gate requiring actual patent
  counsel, parallel to the existing copyright gate — not something resolved by more reading.
- **Corrected before folding in:** the claim that a tree-shaken webgpu-torch subset is "likely
  <100KB" was unverified and, once checked (1.4MB raw source, code-generated op tables that
  resist tree-shaking, no `dist/` artifact to measure against), looked more like an optimistic
  guess than a number a decision should hang on — especially since it was being used to argue for
  reversing an existing architecture call. Replaced "trust the estimate" with "measure it" — a
  cheap pre-spike (Part 7 step 9) that builds webgpu-torch for real and reads the actual number
  before choosing between it and hand-rolled WGSL. The underlying engineering-risk argument
  (hand-rolled backprop is bug-prone) was kept — it's valid independent of the bundle-size number
  attached to it.
- **Corrected before folding in:** the review described a 100k-point-subsample /
  freeze-then-fit-latents training scheme as already present in this document ("§C2 step 3"). A
  direct grep confirmed no version of this plan mentions it before this edit — the same
  misattribution pattern caught in earlier reviews (v3, v4). Evaluated as the new proposal it
  actually is: the core idea is genuinely good and was added to §C2. Two things were fixed in the
  process: the claim that fitting per-point latents against a frozen MLP is "forward-only" is
  incorrect (it's still a full backward pass through the frozen network, just without weight
  gradients/optimizer state) — stated plainly since an implementer following "forward-only" would
  build the wrong thing; and added the subsample-representativeness risk the original framing
  didn't mention, with a mitigation (stratified/farthest-point sampling) and a spike check.

## What changed in v10 (cold-start implementation prompt prepared)

Preparing to hand this plan to a fresh-context coding agent surfaced one small remaining
inaccuracy: Part 7 step 3 cited the wrong build script. Re-checked `package.json` directly — the
canonical, actually-wired-in entrypoint is `npm run build:wasm` (which runs
`rust/build_wasm.js`), not the raw shell script previously cited. Both files exist on disk, but
the npm script is what a build/CI pipeline actually invokes; corrected the citation. No design
changes this round — this was a pre-handoff accuracy pass. The cold-start prompt itself lives in
the conversation that produced this version, not in this file.
