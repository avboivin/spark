import * as fflate from 'fflate';
import { stable_lexicographic_sort_wasm, GsplatArray } from 'spark-rs';

// Helper to convert Float32 to IEEE 754 Float16 bits
function float32ToHalf(val: number): number {
  const f32 = new Float32Array([val]);
  const u32 = new Uint32Array(f32.buffer)[0];
  const sign = (u32 >> 31) & 0x1;
  let exp = (u32 >> 23) & 0xff;
  let mantissa = u32 & 0x7fffff;

  if (exp === 0xff) {
    return (sign << 15) | 0x7c00 | (mantissa ? 1 : 0);
  }
  
  let halfExp = exp - 127 + 15;
  if (halfExp >= 0x1f) {
    return (sign << 15) | 0x7bff;
  }
  if (halfExp <= 0) {
    if (halfExp < -10) return sign << 15;
    mantissa = (mantissa | 0x800000) >> (1 - halfExp);
    return (sign << 15) | (mantissa >> 13);
  }
  
  return (sign << 15) | (halfExp << 10) | (mantissa >> 13);
}

function float32ArrayToFloat16Bytes(arr: Float32Array): Uint8Array {
  const out = new Uint16Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = float32ToHalf(arr[i]);
  }
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

// 1D K-means for Scale (3 channels processed independently)
function kmeans1d(data: Float32Array, k: number, maxIters: number = 5): Float32Array {
  const centroids = new Float32Array(k);
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    if (val < minVal) minVal = val;
    if (val > maxVal) maxVal = val;
  }
  for (let i = 0; i < k; i++) {
    centroids[i] = minVal + (maxVal - minVal) * (i / (k - 1 || 1));
  }
  
  const assignments = new Uint8Array(data.length);
  for (let iter = 0; iter < maxIters; iter++) {
    const sums = new Float64Array(k);
    const counts = new Uint32Array(k);
    for (let i = 0; i < data.length; i++) {
      const val = data[i];
      let minDist = Infinity;
      let bestIdx = 0;
      for (let j = 0; j < k; j++) {
        const dist = Math.abs(val - centroids[j]);
        if (dist < minDist) {
          minDist = dist;
          bestIdx = j;
        }
      }
      assignments[i] = bestIdx;
      sums[bestIdx] += val;
      counts[bestIdx]++;
    }
    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        centroids[j] = sums[j] / counts[j];
      }
    }
  }
  return centroids;
}

// 2D K-means for Rotation (2 slices of 2 channels processed independently)
function kmeans2d(data: Float32Array, k: number, maxIters: number = 5): Float32Array {
  const numPoints = data.length / 2;
  const centroids = new Float32Array(k * 2);
  
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < numPoints; i++) {
    const x = data[i * 2 + 0];
    const y = data[i * 2 + 1];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  
  for (let i = 0; i < k; i++) {
    const angle = (i / k) * Math.PI * 2;
    centroids[i * 2 + 0] = (minX + maxX) / 2 + Math.cos(angle) * (maxX - minX) * 0.4;
    centroids[i * 2 + 1] = (minY + maxY) / 2 + Math.sin(angle) * (maxY - minY) * 0.4;
  }
  
  const assignments = new Uint8Array(numPoints);
  for (let iter = 0; iter < maxIters; iter++) {
    const sumsX = new Float64Array(k);
    const sumsY = new Float64Array(k);
    const counts = new Uint32Array(k);
    
    for (let i = 0; i < numPoints; i++) {
      const px = data[i * 2 + 0];
      const py = data[i * 2 + 1];
      let minDist = Infinity;
      let bestIdx = 0;
      for (let j = 0; j < k; j++) {
        const dx = px - centroids[j * 2 + 0];
        const dy = py - centroids[j * 2 + 1];
        const dist = dx * dx + dy * dy;
        if (dist < minDist) {
          minDist = dist;
          bestIdx = j;
        }
      }
      assignments[i] = bestIdx;
      sumsX[bestIdx] += px;
      sumsY[bestIdx] += py;
      counts[bestIdx]++;
    }
    
    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        centroids[j * 2 + 0] = sumsX[j] / counts[j];
        centroids[j * 2 + 1] = sumsY[j] / counts[j];
      }
    }
  }
  return centroids;
}

// Huffman Code Table Builder
interface HuffmanNode {
  symbol?: number;
  freq: number;
  left?: HuffmanNode;
  right?: HuffmanNode;
}

function buildHuffmanTable(frequencies: Record<number, number>): Record<number, [number, number]> {
  const nodes: HuffmanNode[] = Object.entries(frequencies).map(([sym, freq]) => ({
    symbol: parseInt(sym),
    freq,
  }));
  
  while (nodes.length < 2) {
    nodes.push({ symbol: -1, freq: 0 });
  }
  
  while (nodes.length > 1) {
    nodes.sort((a, b) => a.freq - b.freq);
    const left = nodes.shift()!;
    const right = nodes.shift()!;
    nodes.push({
      freq: left.freq + right.freq,
      left,
      right,
    });
  }
  
  const root = nodes[0];
  const table: Record<number, [number, number]> = {};
  
  function traverse(node: HuffmanNode, bits: number, len: number) {
    if (node.symbol !== undefined) {
      if (node.symbol !== -1) {
        table[node.symbol] = [len, bits];
      }
      return;
    }
    if (node.left) traverse(node.left, (bits << 1) | 0, len + 1);
    if (node.right) traverse(node.right, (bits << 1) | 1, len + 1);
  }
  
  traverse(root, 0, 0);
  return table;
}

function encodeHuffman(indices: Uint8Array, htable: Record<number, [number, number]>): Uint8Array {
  const bytes: number[] = [];
  let currentByte = 0;
  let bitCount = 0;
  
  for (let i = 0; i < indices.length; i++) {
    const symbol = indices[i];
    const code = htable[symbol];
    if (!code) continue;
    const [len, bits] = code;
    
    for (let b = len - 1; b >= 0; b--) {
      const bit = (bits >> b) & 1;
      currentByte = (currentByte << 1) | bit;
      bitCount++;
      if (bitCount === 8) {
        bytes.push(currentByte);
        currentByte = 0;
        bitCount = 0;
      }
    }
  }
  if (bitCount > 0) {
    currentByte = currentByte << (8 - bitCount);
    bytes.push(currentByte);
  }
  return new Uint8Array(bytes);
}

// Main Client-Side Converter
export async function convertSplatToSp5Client({
  numSplats,
  xyz,
  opacity,
  rgb,
  scales,
  quaternions,
  sh1,
  maxSh,
  onProgress,
  lodBase,
}: {
  numSplats: number;
  xyz: Float32Array;
  opacity: Float32Array;
  rgb: Float32Array;
  scales: Float32Array;
  quaternions: Float32Array;
  sh1?: Float32Array;
  maxSh: number;
  onProgress?: (phase: string, percent: number) => void;
  lodBase?: number;
}): Promise<Uint8Array> {
  const CHUNK_SIZE = 65536;

  // 1. Sort points lexicographically using WASM
  onProgress?.("Sorting coordinates...", 10);
  const sortedIndices = stable_lexicographic_sort_wasm(xyz);
  
  // Recenter the scene near the origin so coordinates stay within the f16 range
  // used by Spark's packed splat texture (+/-65504). Use the per-axis MEDIAN, not
  // the bounding-box midpoint: photogrammetry scenes routinely contain a handful
  // of extreme floater outliers (this dataset reaches +/-175000 while 99.9% of
  // splats are within +/-5500). A min/max midpoint would be dragged tens of
  // thousands of units away by those few points -- e.g. here (minX+maxX)/2 lands
  // at ~66821 -- shifting the entire scene out of f16 range so every coordinate
  // overflows to Infinity. The median is robust to such outliers.
  let cx = 0, cy = 0, cz = 0;
  if (numSplats > 0) {
    const axis = new Float32Array(numSplats);
    const medianOf = (comp: number): number => {
      for (let i = 0; i < numSplats; i++) axis[i] = xyz[i * 3 + comp];
      axis.sort();
      return axis[numSplats >> 1];
    };
    cx = medianOf(0);
    cy = medianOf(1);
    cz = medianOf(2);
  }

  const sortedXyz = new Float32Array(numSplats * 3);
  const sortedOpacity = new Float32Array(numSplats);
  const sortedRgb = new Float32Array(numSplats * 3);
  const sortedScales = new Float32Array(numSplats * 3);
  const sortedQuats = new Float32Array(numSplats * 4);
  const sortedSh1 = sh1 ? new Float32Array(numSplats * 9) : undefined;

  for (let i = 0; i < numSplats; i++) {
    const idx = sortedIndices[i];
    sortedXyz[i * 3 + 0] = xyz[idx * 3 + 0] - cx;
    sortedXyz[i * 3 + 1] = xyz[idx * 3 + 1] - cy;
    sortedXyz[i * 3 + 2] = xyz[idx * 3 + 2] - cz;

    sortedOpacity[i] = opacity[idx];

    sortedRgb[i * 3 + 0] = rgb[idx * 3 + 0];
    sortedRgb[i * 3 + 1] = rgb[idx * 3 + 1];
    sortedRgb[i * 3 + 2] = rgb[idx * 3 + 2];

    sortedScales[i * 3 + 0] = scales[idx * 3 + 0];
    sortedScales[i * 3 + 1] = scales[idx * 3 + 1];
    sortedScales[i * 3 + 2] = scales[idx * 3 + 2];

    sortedQuats[i * 4 + 0] = quaternions[idx * 4 + 0];
    sortedQuats[i * 4 + 1] = quaternions[idx * 4 + 1];
    sortedQuats[i * 4 + 2] = quaternions[idx * 4 + 2];
    sortedQuats[i * 4 + 3] = quaternions[idx * 4 + 3];

    if (sortedSh1 && sh1) {
      for (let k = 0; k < 9; k++) {
        sortedSh1[i * 9 + k] = sh1[idx * 9 + k];
      }
    }
  }

  // 2. Build monolithic GsplatArray and compute LOD tree
  onProgress?.("Building monolithic LOD tree...", 20);
  const gsplat = GsplatArray.from_attributes(
    sortedXyz,
    sortedOpacity,
    sortedRgb,
    sortedScales,
    sortedQuats,
    sortedSh1 ? sortedSh1 : undefined
  );
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
  gsplat.tiny_lod(base, false);

  const attrs = gsplat.extract_attributes();
  const totalNumSplats = gsplat.len();

  const reorderedXyz = new Float32Array(attrs.xyz);
  const reorderedOpacity = new Float32Array(attrs.opacity);
  const reorderedRgb = new Float32Array(attrs.rgb);
  const reorderedScales = new Float32Array(attrs.scales);
  const reorderedQuats = new Float32Array(attrs.quaternions);
  const reorderedSh1 = sortedSh1 ? new Float32Array(attrs.sh1) : undefined;
  const lodTree = gsplat.extract_lod_tree();
  gsplat.free();

  // 3. Run k-means codebook generation globally on the expanded/reordered attributes
  onProgress?.("Quantizing scale and rotation channels...", 30);
  // Cluster in LOG space, not linear. Since Phase 1's monolithic tiny_lod tree mixes
  // ordinary leaf splats (scale ~0.01-10 typically) with merged/coarse LOD parent
  // splats (scale can run 10-20x+ larger, representing whole merged regions) into
  // this SAME 256-entry codebook, linear k-means initializes/settles centroids
  // spread across the FULL combined range -- starving the dense leaf population
  // (the vast majority of what's actually rendered up close) of codebook resolution
  // and forcing many unrelated leaf scales to collide onto the same few centroids.
  // Clustering in log space (matching the .ln() this value receives on decode
  // anyway) gives proportional resolution to both populations regardless of their
  // absolute magnitude gap. Codebook values are exponentiated back to linear before
  // being written out, so the on-disk format and decode side are unchanged.
  const scaleChannels = [
    new Float32Array(totalNumSplats),
    new Float32Array(totalNumSplats),
    new Float32Array(totalNumSplats),
  ];
  for (let i = 0; i < totalNumSplats; i++) {
    scaleChannels[0][i] = Math.log(Math.max(reorderedScales[i * 3 + 0], 1e-8));
    scaleChannels[1][i] = Math.log(Math.max(reorderedScales[i * 3 + 1], 1e-8));
    scaleChannels[2][i] = Math.log(Math.max(reorderedScales[i * 3 + 2], 1e-8));
  }
  const scaleCodebooks = scaleChannels.map(c => kmeans1d(c, 256));

  const rotSlice0 = new Float32Array(totalNumSplats * 2);
  const rotSlice1 = new Float32Array(totalNumSplats * 2);
  for (let i = 0; i < totalNumSplats; i++) {
    rotSlice0[i * 2 + 0] = reorderedQuats[i * 4 + 0];
    rotSlice0[i * 2 + 1] = reorderedQuats[i * 4 + 1];
    rotSlice1[i * 2 + 0] = reorderedQuats[i * 4 + 2];
    rotSlice1[i * 2 + 1] = reorderedQuats[i * 4 + 3];
  }
  const rotCodebooks = [
    kmeans2d(rotSlice0, 256),
    kmeans2d(rotSlice1, 256),
  ];

  // 4. Map each splat's properties to the closest codebook indices
  onProgress?.("Mapping indices and building Huffman tables...", 60);
  const scaleIndices = [
    new Uint8Array(totalNumSplats),
    new Uint8Array(totalNumSplats),
    new Uint8Array(totalNumSplats),
  ];
  for (let c = 0; c < 3; c++) {
    const data = scaleChannels[c];
    const cb = scaleCodebooks[c];
    for (let i = 0; i < totalNumSplats; i++) {
      const val = data[i];
      let minDist = Infinity, bestIdx = 0;
      for (let j = 0; j < 256; j++) {
        const dist = Math.abs(val - cb[j]);
        if (dist < minDist) { minDist = dist; bestIdx = j; }
      }
      scaleIndices[c][i] = bestIdx;
    }
  }
  // Codebook clustering/assignment above happened in log space; convert centroids
  // back to linear before they're written to disk (reconstruct_sp5_chunk expects
  // linear scale_codebook values and applies its own single .ln()).
  for (let c = 0; c < 3; c++) {
    for (let j = 0; j < 256; j++) {
      scaleCodebooks[c][j] = Math.exp(scaleCodebooks[c][j]);
    }
  }

  const rotIndices = [
    new Uint8Array(totalNumSplats),
    new Uint8Array(totalNumSplats),
  ];
  for (let i = 0; i < totalNumSplats; i++) {
    for (let s = 0; s < 2; s++) {
      const slice = s === 0 ? rotSlice0 : rotSlice1;
      const cb = rotCodebooks[s];
      const px = slice[i * 2 + 0];
      const py = slice[i * 2 + 1];
      let minDist = Infinity, bestIdx = 0;
      for (let j = 0; j < 256; j++) {
        const dx = px - cb[j * 2 + 0];
        const dy = py - cb[j * 2 + 1];
        const dist = dx * dx + dy * dy;
        if (dist < minDist) { minDist = dist; bestIdx = j; }
      }
      rotIndices[s][i] = bestIdx;
    }
  }

  // Huffman tables
  const scaleTables = scaleIndices.map(ind => {
    const freq: Record<number, number> = {};
    ind.forEach(v => freq[v] = (freq[v] || 0) + 1);
    return buildHuffmanTable(freq);
  });
  const rotTables = rotIndices.map(ind => {
    const freq: Record<number, number> = {};
    ind.forEach(v => freq[v] = (freq[v] || 0) + 1);
    return buildHuffmanTable(freq);
  });

  // 5. Partition/Slice contiguously into chunks of size exactly CHUNK_SIZE = 65536
  const numChunks = Math.ceil(totalNumSplats / CHUNK_SIZE);
  const zipFiles: Record<string, Uint8Array> = {};
  const chunksManifest: any[] = [];

  onProgress?.("Packaging .sp5 chunk files...", 80);
  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    const start = chunkIdx * CHUNK_SIZE;
    const count = Math.min(totalNumSplats - start, CHUNK_SIZE);

    const chunkXyzRaw = reorderedXyz.subarray(start * 3, (start + count) * 3);
    // Positions are stored as float16, which hard-clamps at +/-65504. Chunks are no
    // longer small spatial regions -- since Phase 1's monolithic-tree-then-slice
    // change, chunk 0 (and other coarse BFS-order chunks) can span the ENTIRE scene
    // (hundreds of thousands of world units for outdoor/GPS-scale captures). Packing
    // that range directly in f16 pins most positions to the clamp boundary, collapsing
    // the coarsest LOD level onto a handful of points -- exactly the "poisoned bbox"
    // corruption class fixed earlier for the .ply path, but re-introduced here because
    // this xyz_uncompressed encoding never adopted the .ply path's adaptive-precision
    // approach (SPZ's per-chunk fractional-bits fixed point). Fix: normalize each
    // chunk's positions into its own local [-F16_SAFE_MAX, F16_SAFE_MAX] range before
    // packing, and store the chunk's center/scale so the decoder can invert it.
    const F16_SAFE_MAX = 60000;
    let cMinX = Infinity, cMinY = Infinity, cMinZ = Infinity;
    let cMaxX = -Infinity, cMaxY = -Infinity, cMaxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const x = chunkXyzRaw[i * 3 + 0], y = chunkXyzRaw[i * 3 + 1], z = chunkXyzRaw[i * 3 + 2];
      if (x < cMinX) cMinX = x; if (x > cMaxX) cMaxX = x;
      if (y < cMinY) cMinY = y; if (y > cMaxY) cMaxY = y;
      if (z < cMinZ) cMinZ = z; if (z > cMaxZ) cMaxZ = z;
    }
    const chunkCenter: [number, number, number] = count > 0
      ? [(cMinX + cMaxX) / 2, (cMinY + cMaxY) / 2, (cMinZ + cMaxZ) / 2]
      : [0, 0, 0];
    // Per-axis half-extent for even precision across all dimensions.
    // Normalization: (pos - center[d]) / half_extent[d] ∈ [-1, 1].
    // After this, positions are in [-F16_SAFE_MAX, F16_SAFE_MAX] for
    // f16 packing (F16_SAFE_MAX = 60000 < 65504 f16 limit).
    const halfExtentX = count > 0 ? (cMaxX - cMinX) / 2 : 1;
    const halfExtentY = count > 0 ? (cMaxY - cMinY) / 2 : 1;
    const halfExtentZ = count > 0 ? (cMaxZ - cMinZ) / 2 : 1;
    // Scalar scale kept for backward compat (approximates the per-axis values)
    const halfExtent = Math.max(halfExtentX, halfExtentY, halfExtentZ);
    const chunkScale = Math.max(halfExtent, 1e-6) / F16_SAFE_MAX;
    const sx = Math.max(halfExtentX, 1e-6) / F16_SAFE_MAX;
    const sy = Math.max(halfExtentY, 1e-6) / F16_SAFE_MAX;
    const sz = Math.max(halfExtentZ, 1e-6) / F16_SAFE_MAX;
    const chunkXyz = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      chunkXyz[i * 3 + 0] = (chunkXyzRaw[i * 3 + 0] - chunkCenter[0]) / sx;
      chunkXyz[i * 3 + 1] = (chunkXyzRaw[i * 3 + 1] - chunkCenter[1]) / sy;
      chunkXyz[i * 3 + 2] = (chunkXyzRaw[i * 3 + 2] - chunkCenter[2]) / sz;
    }
    const chunkOpacity = reorderedOpacity.subarray(start, start + count);
    const chunkRgb = reorderedRgb.subarray(start * 3, (start + count) * 3);
    const chunkSh1 = reorderedSh1 ? reorderedSh1.subarray(start * 9, (start + count) * 9) : new Float32Array(0);

    const chunkScaleIndices = [
      scaleIndices[0].subarray(start, start + count),
      scaleIndices[1].subarray(start, start + count),
      scaleIndices[2].subarray(start, start + count),
    ];
    const chunkRotIndices = [
      rotIndices[0].subarray(start, start + count),
      rotIndices[1].subarray(start, start + count),
    ];
    const lodTreeSlice = lodTree.subarray(start * 4, (start + count) * 4);

    // Huffman streams
    const scaleHuffman = chunkScaleIndices.map((ind, c) => encodeHuffman(ind, scaleTables[c]));
    const rotHuffman = chunkRotIndices.map((ind, s) => encodeHuffman(ind, rotTables[s]));

    // Construct binary payload and layout offsets
    const binaryBlobs: Uint8Array[] = [];
    let currentOffset = 0;

    function packArray(val: Float32Array): any {
      const bytes = float32ArrayToFloat16Bytes(val);
      const paddedLength = (bytes.length + 3) & ~3;
      const padded = new Uint8Array(paddedLength);
      padded.set(bytes);
      
      const meta = { offset: currentOffset, length: bytes.length, dtype: "float16", shape: [val.length] };
      binaryBlobs.push(padded);
      currentOffset += paddedLength;
      return meta;
    }

    function packUint32Array(val: Uint32Array): any {
      const bytes = new Uint8Array(val.buffer, val.byteOffset, val.byteLength);
      const paddedLength = (bytes.length + 3) & ~3;
      const padded = new Uint8Array(paddedLength);
      padded.set(bytes);
      
      const meta = { offset: currentOffset, length: bytes.length, dtype: "uint32", shape: [val.length / 4, 4] };
      binaryBlobs.push(padded);
      currentOffset += paddedLength;
      return meta;
    }

    function packHuffman(bytes: Uint8Array, table: Record<number, [number, number]>): any {
      const paddedLength = (bytes.length + 3) & ~3;
      const padded = new Uint8Array(paddedLength);
      padded.set(bytes);

      const htableStr: Record<string, [number, number]> = {};
      for (const [k, v] of Object.entries(table)) {
        htableStr[k] = v;
      }

      const meta = { offset: currentOffset, length: bytes.length, huffman_table: htableStr };
      binaryBlobs.push(padded);
      currentOffset += paddedLength;
      return meta;
    }

    const chunkMeta: any = {
      version: 5,
      count,
      is_uncompressed: true,
      scale_codebook: [],
      rotation_codebook: [],
      app_codebook: [],
      scale_index_huffman: [],
      rotation_index_huffman: [],
      app_index_huffman: [],
      mlp_cont: { offset: 0, length: 0, dtype: "float16", shape: [0] },
      mlp_offset: {}
    };

    // Positions (uncompressed, chunk-local-normalized -- see chunkXyz comment above)
    chunkMeta.xyz_uncompressed = packArray(chunkXyz);
    chunkMeta.chunk_center = chunkCenter;
    chunkMeta.chunk_half_extent = [halfExtentX, halfExtentY, halfExtentZ];

    // Codebooks
    scaleCodebooks.forEach(cb => {
      chunkMeta.scale_codebook.push(packArray(cb));
    });
    rotCodebooks.forEach(cb => {
      chunkMeta.rotation_codebook.push(packArray(cb));
    });
    // Dummy app codebook
    chunkMeta.app_codebook.push(packArray(new Float32Array(512)));

    // Huffman streams
    scaleHuffman.forEach((bytes, c) => {
      chunkMeta.scale_index_huffman.push(packHuffman(bytes, scaleTables[c]));
    });
    rotHuffman.forEach((bytes, s) => {
      chunkMeta.rotation_index_huffman.push(packHuffman(bytes, rotTables[s]));
    });
    // Dummy app Huffman stream
    chunkMeta.app_index_huffman.push(packHuffman(new Uint8Array(count), { 0: [1, 0] }));

    // Raw opacity, color, and SH carried inside the MLP weights slots
    chunkMeta.mlp_opacity = packArray(chunkOpacity);
    chunkMeta.mlp_dc = packArray(chunkRgb);
    chunkMeta.mlp_sh = packArray(chunkSh1);

    // Pack the lod_tree
    chunkMeta.lod_tree = packUint32Array(lodTreeSlice);

    // Serialize JSON header
    const jsonStr = JSON.stringify(chunkMeta);
    const jsonBytes = new TextEncoder().encode(jsonStr);
    const headerSize = 8 + jsonBytes.length;
    
    const chunkOutBytes = new Uint8Array(headerSize + currentOffset);
    const view = new DataView(chunkOutBytes.buffer);
    view.setUint32(0, 0x355a5053, true); // "SPZ5" magic
    view.setUint32(4, jsonBytes.length, true);
    chunkOutBytes.set(jsonBytes, 8);

    let writeOffset = headerSize;
    binaryBlobs.forEach(blob => {
      chunkOutBytes.set(blob, writeOffset);
      writeOffset += blob.length;
    });

    const chunkFilename = `scene-lod-${chunkIdx}.sp5`;
    zipFiles[chunkFilename] = chunkOutBytes;

    // AABB bbox bounding box of coords
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const x = chunkXyzRaw[i * 3 + 0];
      const y = chunkXyzRaw[i * 3 + 1];
      const z = chunkXyzRaw[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    chunksManifest.push({
      file: chunkFilename,
      aabb: [minX, minY, minZ, maxX, maxY, maxZ],
      count
    });
  }

  // Global manifest.json
  const globalManifest = {
    version: 5,
    type: "sp5",
    count: totalNumSplats,
    maxSh,
    lodTree: true,
    chunkSize: CHUNK_SIZE,
    chunks: chunksManifest
  };
  zipFiles["manifest.json"] = new TextEncoder().encode(JSON.stringify(globalManifest, null, 2));

  onProgress?.("Zipping output package...", 95);
  const zipped = fflate.zipSync(zipFiles);
  return zipped;
}
