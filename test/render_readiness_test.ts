// Headless "will this actually render?" check for a dropped .ply/.spz scene --
// no browser, no manual drag-and-drop required.
//
// WHY THIS EXISTS: repeated rounds of manual browser testing on this branch
// found bugs (Infinity positions from f16 overflow, a poisoned bounding-box
// center collapsing the whole scene onto the f16 clamp boundary, a camera far
// plane too small for the fitted distance) where every console log looked
// correct -- LOD traversal counts climbing, camera-fit math resolving, no
// thrown errors -- yet the canvas stayed black. None of those bugs live in
// the GPU/shader layer; they're all in data that's fully inspectable in
// Node before a browser is ever involved: decoded positions, packed texture
// bytes, and the camera-fit arithmetic the viewer uses. This script runs the
// exact same pipeline stages (decode -> center -> LOD build -> per-chunk SPZ
// round-trip -> f16 pack, matching worker.ts's partitionDroppedMonolithic and
// PackedSplatsData) and the exact same camera-fit formula as
// examples/viewer/index.html, then asserts hard numeric thresholds instead of
// requiring a human to eyeball a canvas.
//
// Usage: npx tsx test/render_readiness_test.ts [path-to.ply]
// Exit code 0 = render-ready, 1 = a concrete defect was found (see report).

import fs from 'fs';
import path from 'path';
import init_wasm, { decode_to_gsplatarray } from '../rust/spark-rs/pkg/spark_rs.js';

const F16_MAX = 65504;
const CHUNK_SIZE = 65536; // matches worker.ts partitionDroppedMonolithic / PAGE_SPLATS

// Same camera constants as examples/viewer/index.html.
const CAMERA_FOV_DEG = 75;
const CAMERA_ASPECT = 16 / 9;
const CAMERA_NEAR_PLACEHOLDER = 0.01;
const CAMERA_FAR_PLACEHOLDER = 1000; // the value that caused the far-plane clip bug

function halfToFloat(binary: number): number {
  const exponent = (binary & 0x7c00) >> 10;
  const fraction = binary & 0x03ff;
  if (exponent === 0) {
    return (binary & 0x8000 ? -1 : 1) * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction ? NaN : binary & 0x8000 ? -Infinity : Infinity;
  }
  return (binary & 0x8000 ? -1 : 1) * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function decodeGsplatArrayFromBytes(fileType: string, bytes: Uint8Array) {
  const decoder = decode_to_gsplatarray(fileType, null);
  const PUSH_CHUNK = 1024 * 1024;
  for (let i = 0; i < bytes.length; i += PUSH_CHUNK) {
    decoder.push(bytes.subarray(i, Math.min(i + PUSH_CHUNK, bytes.length)));
  }
  return decoder.finish();
}

function percentile(sortedArr: Float32Array, p: number): number {
  const idx = Math.min(sortedArr.length - 1, Math.max(0, Math.round((p / 100) * (sortedArr.length - 1))));
  return sortedArr[idx];
}

async function main() {
  const plyPath = process.argv[2] ?? 'E:\\ste_cecile\\experiments\\full_500k_lift\\work\\eval\\splat_30000.ply';
  const failures: string[] = [];
  const warnings: string[] = [];

  console.log(`=== Render-readiness check: ${plyPath} ===\n`);

  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  const plyBytes = new Uint8Array(fs.readFileSync(plyPath));
  const gsplatArray = decodeGsplatArrayFromBytes('ply', plyBytes);
  const rawNumSplats = gsplatArray.len();
  console.log(`[decode] ${rawNumSplats} splats decoded from source file`);
  if (rawNumSplats === 0) {
    failures.push('decode produced 0 splats -- nothing to render, regardless of anything downstream');
  }

  // Same recentering call worker.ts's partitionDroppedMonolithic makes.
  const centerShift = gsplatArray.center() as unknown as number[];
  console.log(`[center] shift applied: (${centerShift.map((v) => v.toFixed(3)).join(', ')})`);

  // Same LOD build worker.ts makes.
  (gsplatArray as any).tiny_lod(1.5, false);
  const totalNumSplats = gsplatArray.len();
  console.log(`[tiny_lod] splat count after LOD build: ${totalNumSplats} (was ${rawNumSplats})`);

  const numChunks = Math.ceil(totalNumSplats / CHUNK_SIZE);
  console.log(`[chunk] slicing into ${numChunks} chunks of up to ${CHUNK_SIZE} splats\n`);

  // Aggregate stats across every chunk, computed from the ACTUAL packed (f16)
  // texture bytes -- i.e. exactly what the GPU would receive -- not the
  // pre-pack float32 values.
  let totalChecked = 0;
  let nanOrInfCount = 0;
  let atF16BoundaryCount = 0; // exactly +/-F16_MAX: the clamp-pileup signature
  let positiveOpacityCount = 0;
  let positiveScaleCount = 0;
  const sampleCenters: number[] = []; // for the camera-fit/frustum check below, capped

  const SAMPLE_CAP = 2_000_000; // guard against unbounded memory on huge scenes

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const start = chunk * CHUNK_SIZE;
    const count = Math.min(totalNumSplats - start, CHUNK_SIZE);
    const subset = (gsplatArray as any).clone_subset(start, count);

    // Real SPZ re-encode, exactly like partitionDroppedMonolithic does per chunk.
    const spzBytes: Uint8Array = subset.to_spz();
    const reDecoded = decodeGsplatArrayFromBytes('spz', spzBytes);

    // Real f16 packing path (encode_packed_splat / clamp_center_coord), exactly
    // what PackedSplatsData / the GPU upload path uses.
    const packedObj = (reDecoded as any).to_packedsplats(null) as {
      numSplats: number;
      packed: Uint32Array;
    };
    const packed = packedObj.packed;
    const n = packedObj.numSplats;

    for (let i = 0; i < n; i++) {
      const base = i * 4;
      const w1 = packed[base + 1];
      const w2 = packed[base + 2];
      const w0 = packed[base + 0];
      const cx = halfToFloat(w1 & 0xffff);
      const cy = halfToFloat((w1 >>> 16) & 0xffff);
      const cz = halfToFloat(w2 & 0xffff);
      const a = (w0 >>> 24) & 0xff;

      totalChecked++;
      const finite = Number.isFinite(cx) && Number.isFinite(cy) && Number.isFinite(cz);
      if (!finite) {
        nanOrInfCount++;
      } else {
        if (Math.abs(cx) === F16_MAX || Math.abs(cy) === F16_MAX || Math.abs(cz) === F16_MAX) {
          atF16BoundaryCount++;
        }
        if (sampleCenters.length < SAMPLE_CAP * 3) {
          sampleCenters.push(cx, cy, cz);
        }
      }
      if (a > 2) positiveOpacityCount++; // u8 opacity code; >2/255 is clearly nonzero after quantization
    }

    reDecoded.free();
    subset.free();
  }

  gsplatArray.free();

  const pct = (n: number) => ((100 * n) / Math.max(1, totalChecked)).toFixed(4);
  console.log(`[pack] checked ${totalChecked} packed splat centers across ${numChunks} chunks`);
  console.log(`  finite: ${totalChecked - nanOrInfCount}/${totalChecked} (${pct(totalChecked - nanOrInfCount)}%)`);
  console.log(`  NaN/Infinity: ${nanOrInfCount} (${pct(nanOrInfCount)}%)`);
  console.log(`  pinned to f16 clamp boundary (+/-${F16_MAX}): ${atF16BoundaryCount} (${pct(atF16BoundaryCount)}%)`);
  console.log(`  opacity code > 2/255: ${positiveOpacityCount} (${pct(positiveOpacityCount)}%)\n`);

  if (nanOrInfCount > 0) {
    failures.push(
      `${nanOrInfCount} splats (${pct(nanOrInfCount)}%) decode to non-finite positions -- these poison ` +
        `SplatMesh.getBoundingBox() and will produce an infinite/NaN camera-fit box.`,
    );
  }
  // A handful of legitimate floater outliers clamping is fine; a large fraction
  // pinned to the exact boundary means the scene was shifted out of range
  // (the "poisoned bbox-midpoint center" bug class) -- everything, not just
  // outliers, ends up glued to +/-65504.
  if (totalChecked > 0 && atF16BoundaryCount / totalChecked > 0.01) {
    failures.push(
      `${pct(atF16BoundaryCount)}% of splats are pinned to the exact f16 clamp boundary -- ` +
        `this many is not "a few floater outliers", it means the scene's recenter/offset pushed ` +
        `the bulk of real content out of the +/-${F16_MAX} range. Check the centering math (min/max ` +
        `midpoint vs. median) upstream of the f16 pack.`,
    );
  }
  if (totalChecked > 0 && positiveOpacityCount / totalChecked < 0.5) {
    warnings.push(
      `only ${pct(positiveOpacityCount)}% of splats have non-negligible opacity -- scene may render ` +
        `very faint even if positions are otherwise fine.`,
    );
  }

  // Replicate examples/viewer/index.html's computeRobustBox + fitCameraToBoundingBox
  // + the PerspectiveCamera(75, aspect, 0.01, 1000) construction, and check whether
  // the resulting camera distance would survive the camera's near/far planes.
  const numSampled = sampleCenters.length / 3;
  if (numSampled === 0) {
    failures.push('no finite splat centers available to compute a camera fit -- scene cannot be framed at all.');
  } else {
    const xs = new Float32Array(numSampled);
    const ys = new Float32Array(numSampled);
    const zs = new Float32Array(numSampled);
    for (let i = 0; i < numSampled; i++) {
      xs[i] = sampleCenters[i * 3 + 0];
      ys[i] = sampleCenters[i * 3 + 1];
      zs[i] = sampleCenters[i * 3 + 2];
    }
    xs.sort();
    ys.sort();
    zs.sort();
    const lo = 0.5;
    const hi = 99.5;
    const minV = [percentile(xs, lo), percentile(ys, lo), percentile(zs, lo)];
    const maxV = [percentile(xs, hi), percentile(ys, hi), percentile(zs, hi)];
    const center = minV.map((v, i) => (v + maxV[i]) / 2);
    const size = minV.map((v, i) => maxV[i] - v);
    const maxDim = Math.max(...size);

    const fovRad = (CAMERA_FOV_DEG * Math.PI) / 180;
    const cameraZ = (Math.abs(maxDim / 2 / Math.tan(fovRad / 2))) * 1.5;

    console.log(`[camera-fit] robust p${lo}-p${hi} bbox center=(${center.map((v) => v.toFixed(1)).join(', ')}), ` +
      `size=(${size.map((v) => v.toFixed(1)).join(', ')}), maxDim=${maxDim.toFixed(1)}`);
    console.log(`[camera-fit] computed cameraZ (fit distance) = ${cameraZ.toFixed(1)}`);
    console.log(
      `[camera-fit] placeholder camera planes at construction time: near=${CAMERA_NEAR_PLACEHOLDER}, far=${CAMERA_FAR_PLACEHOLDER}`,
    );

    if (cameraZ > CAMERA_FAR_PLACEHOLDER) {
      // Not a failure by itself -- most real scenes need cameraZ well beyond an
      // arbitrary placeholder far plane, and fitCameraToBoundingBox() is
      // supposed to widen camera.far to match. This is only a real bug if that
      // widening doesn't actually happen; the frustum check below (which uses
      // properly widened near/far, matching what the fixed code should produce)
      // is the actual pass/fail signal for "would this be visible."
      console.log(
        `[camera-fit] note: fit distance (${cameraZ.toFixed(1)}) exceeds the construction-time placeholder ` +
          `far plane (${CAMERA_FAR_PLACEHOLDER}) -- this is expected for a scene this size. ` +
          `examples/viewer/index.html's fitCameraToBoundingBox() must widen camera.far (and call ` +
          `updateProjectionMatrix()) based on the fitted distance, or the scene renders entirely ` +
          `beyond the far clip with no console error. The frustum check below verifies that widened-plane ` +
          `case explicitly.`,
      );
    } else {
      console.log(`[camera-fit] OK: fit distance is within the placeholder far plane.`);
    }

    // Frustum projection check: with a perspective camera positioned per
    // fitCameraToBoundingBox and near/far sized to the fit distance (as the
    // fixed code should do), what fraction of sampled splats actually land
    // inside the view volume?
    const near = Math.max(cameraZ / 1000, 1e-4);
    const far = cameraZ + maxDim * 2 + 1;
    const camPos = [center[0], center[1], center[2] + cameraZ];
    // Simple look-down--z camera looking at `center` (matches lookAt(center)
    // with no roll, since the object is on the +z offset from center only).
    let insideFrustum = 0;
    const f = 1 / Math.tan(fovRad / 2);
    const SAMPLE_STRIDE = Math.max(1, Math.floor(numSampled / 200_000)); // cap projection cost
    let projectedCount = 0;
    for (let i = 0; i < numSampled; i += SAMPLE_STRIDE) {
      const dx = xs[i] - camPos[0];
      const dy = ys[i] - camPos[1];
      const dz = -(zs[i] - camPos[2]); // camera looks toward -offset i.e. toward center from +z
      projectedCount++;
      if (dz <= near || dz >= far) continue;
      const ndcX = (f / CAMERA_ASPECT) * (dx / dz);
      const ndcY = f * (dy / dz);
      if (ndcX >= -1 && ndcX <= 1 && ndcY >= -1 && ndcY <= 1) {
        insideFrustum++;
      }
    }
    const frustumPct = ((100 * insideFrustum) / Math.max(1, projectedCount)).toFixed(2);
    console.log(
      `[frustum] ${insideFrustum}/${projectedCount} sampled splats (${frustumPct}%) project inside the ` +
        `fitted camera's view volume (using widened near/far)\n`,
    );
    if (insideFrustum / Math.max(1, projectedCount) < 0.3) {
      failures.push(
        `only ${frustumPct}% of sampled splats project inside the fitted camera's frustum -- camera-fit ` +
          `math and/or scene bounds disagree badly enough that most content would be off-screen.`,
      );
    }
  }

  console.log('=== Report ===');
  if (warnings.length) {
    console.log('Warnings:');
    for (const w of warnings) console.log(`  - ${w}`);
  }
  if (failures.length) {
    console.log('FAILURES (render will not work as-is):');
    for (const f of failures) console.log(`  - ${f}`);
    console.log('\nRESULT: NOT RENDER-READY');
    process.exit(1);
  } else {
    console.log('RESULT: RENDER-READY (all checks passed)');
  }
}

main().catch((e) => {
  console.error('RENDER-READINESS CHECK CRASHED:', e);
  process.exit(1);
});
