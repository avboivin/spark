import * as fflate from 'fflate';
import { stable_lexicographic_sort_wasm } from 'spark-rs';

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

// Rearranges idxArr[lo..hi] in place (Hoare-partition quickselect) so that
// idxArr[n] ends up holding the element that a full sort by xyz[idx*3+axis]
// would place there, with everything before n <= it and everything after n
// >= it (a "nth_element", not a full sort -- O(range) average case).
function nthElementByAxis(idxArr: Int32Array, lo: number, hi: number, n: number, xyz: Float32Array, axis: number) {
  while (hi > lo) {
    const pivotIdx = idxArr[(lo + hi) >> 1];
    const pivotVal = xyz[pivotIdx * 3 + axis];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (xyz[idxArr[i] * 3 + axis] < pivotVal) i++;
      while (xyz[idxArr[j] * 3 + axis] > pivotVal) j--;
      if (i <= j) {
        const tmp = idxArr[i];
        idxArr[i] = idxArr[j];
        idxArr[j] = tmp;
        i++;
        j--;
      }
    }
    if (n <= j) hi = j;
    else if (n >= i) lo = i;
    else break;
  }
}

// Splits `idxArr` into spatially-coherent groups of at most `chunkSize`
// indices each, by recursively picking whichever of x/y/z has the LARGEST
// extent within the current group and splitting at its median (a k-d tree).
// This is deliberately NOT a single global sort along one fixed axis: a
// global lexicographic (z,y,x) sort followed by contiguous count-based
// slicing produces degenerate chunks whenever point density is highly
// non-uniform, which real-world captures always are -- e.g. a dense building
// facade plus a sparse, widely-spread field of background/terrain points. A
// fixed global sort dominated by z packs the dense region into razor-thin,
// nearly 2D horizontal slabs (measured directly on a real capture: consecutive
// chunks spanning as little as 0.9 world units in z, i.e. paper-thin slices of
// a multi-story building) while the sparse tail dominates one or two chunks
// with an enormous, near-empty bounding box. Re-picking the split axis at
// every level based on the ACTUAL shape of the current subgroup adapts to
// this instead of committing to one axis for the whole scene, so each
// resulting chunk is a roughly cube-shaped, spatially local region regardless
// of how lopsided the overall point-density distribution is.
function kdPartitionIndices(idxArr: Int32Array, xyz: Float32Array, chunkSize: number): Int32Array[] {
  const groups: Int32Array[] = [];
  const stack: [number, number][] = [[0, idxArr.length - 1]];
  while (stack.length) {
    const [lo, hi] = stack.pop()!;
    const count = hi - lo + 1;
    if (count <= chunkSize) {
      groups.push(idxArr.slice(lo, hi + 1));
      continue;
    }

    let minV0 = Infinity, minV1 = Infinity, minV2 = Infinity;
    let maxV0 = -Infinity, maxV1 = -Infinity, maxV2 = -Infinity;
    for (let p = lo; p <= hi; p++) {
      const pt = idxArr[p] * 3;
      const x = xyz[pt], y = xyz[pt + 1], z = xyz[pt + 2];
      if (x < minV0) minV0 = x; if (x > maxV0) maxV0 = x;
      if (y < minV1) minV1 = y; if (y > maxV1) maxV1 = y;
      if (z < minV2) minV2 = z; if (z > maxV2) maxV2 = z;
    }
    const extents = [maxV0 - minV0, maxV1 - minV1, maxV2 - minV2];
    let axis = 0;
    if (extents[1] > extents[axis]) axis = 1;
    if (extents[2] > extents[axis]) axis = 2;

    const mid = lo + (count >> 1);
    nthElementByAxis(idxArr, lo, hi, mid, xyz, axis);
    stack.push([lo, mid - 1]);
    stack.push([mid, hi]);
  }
  return groups;
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
  priorityPoint,
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
  // World-space point (in the SOURCE, pre-recenter coordinate space -- same
  // space as the input `xyz`) to prioritize for fast initial load. Chunk 0 is
  // fetched unconditionally before any LOD tree exists to discover the rest
  // of the scene (see worker.ts's synthesizeFlatLodTreeIfMissing), so
  // whichever spatial region becomes chunk 0 is what appears first,
  // regardless of where the camera actually starts. If given, the chunk
  // whose centroid is closest to this point becomes chunk 0 -- e.g. pass
  // your intended default camera's look-at target/position so the region
  // visible on load already has its data resident instead of a k-d-tree-order
  // region unrelated to what the viewer will actually be looking at.
  priorityPoint?: [number, number, number];
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

  // 2. Run k-means codebook generation
  onProgress?.("Quantizing scale and rotation channels...", 30);
  const scaleChannels = [
    sortedScales.filter((_, idx) => idx % 3 === 0),
    sortedScales.filter((_, idx) => idx % 3 === 1),
    sortedScales.filter((_, idx) => idx % 3 === 2),
  ];
  const scaleCodebooks = scaleChannels.map(c => kmeans1d(c, 256));

  const rotSlice0 = new Float32Array(numSplats * 2);
  const rotSlice1 = new Float32Array(numSplats * 2);
  for (let i = 0; i < numSplats; i++) {
    rotSlice0[i * 2 + 0] = sortedQuats[i * 4 + 0];
    rotSlice0[i * 2 + 1] = sortedQuats[i * 4 + 1];
    rotSlice1[i * 2 + 0] = sortedQuats[i * 4 + 2];
    rotSlice1[i * 2 + 1] = sortedQuats[i * 4 + 3];
  }
  const rotCodebooks = [
    kmeans2d(rotSlice0, 256),
    kmeans2d(rotSlice1, 256),
  ];

  // Map each splat's properties to the closest codebook indices
  onProgress?.("Mapping indices and building Huffman tables...", 60);
  const scaleIndices = [
    new Uint8Array(numSplats),
    new Uint8Array(numSplats),
    new Uint8Array(numSplats),
  ];
  for (let c = 0; c < 3; c++) {
    const data = scaleChannels[c];
    const cb = scaleCodebooks[c];
    for (let i = 0; i < numSplats; i++) {
      const val = data[i];
      let minDist = Infinity, bestIdx = 0;
      for (let j = 0; j < 256; j++) {
        const dist = Math.abs(val - cb[j]);
        if (dist < minDist) { minDist = dist; bestIdx = j; }
      }
      scaleIndices[c][i] = bestIdx;
    }
  }

  const rotIndices = [
    new Uint8Array(numSplats),
    new Uint8Array(numSplats),
  ];
  for (let i = 0; i < numSplats; i++) {
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

  // 3. Packaging into chunks
  // Partition by actual 3D spatial locality (see kdPartitionIndices above),
  // not by contiguous ranges of the global (z,y,x) sort order -- the latter
  // produces degenerate, near-2D chunks whenever the scene's point density is
  // highly non-uniform (dense building + sparse widely-spread background),
  // which showed up as "only one direction/slice of the scene visible" and
  // "95% missing" once loaded, since most chunks ended up covering only a
  // sub-1-unit-thick horizontal slab of the real content.
  onProgress?.("Partitioning into spatial chunks...", 75);
  const allIndices = new Int32Array(numSplats);
  for (let i = 0; i < numSplats; i++) allIndices[i] = i;
  const chunkGroups = kdPartitionIndices(allIndices, sortedXyz, CHUNK_SIZE);

  // Chunk 0 is fetched unconditionally, before any LOD tree exists to
  // discover the rest of the scene -- it's the one region guaranteed to be
  // visible immediately on load. Left alone, which spatial group lands at
  // index 0 is just whatever the k-d partition produced first, unrelated to
  // where a viewer's default camera actually looks. If a priority point was
  // given, move whichever group's centroid is closest to it into index 0.
  if (priorityPoint && chunkGroups.length > 1) {
    const [pcx, pcy, pcz] = [priorityPoint[0] - cx, priorityPoint[1] - cy, priorityPoint[2] - cz];
    let bestGroup = 0;
    let bestDist = Infinity;
    for (let g = 0; g < chunkGroups.length; g++) {
      const group = chunkGroups[g];
      let sx = 0, sy = 0, sz = 0;
      for (let k = 0; k < group.length; k++) {
        const i = group[k];
        sx += sortedXyz[i * 3 + 0];
        sy += sortedXyz[i * 3 + 1];
        sz += sortedXyz[i * 3 + 2];
      }
      const n = group.length;
      const dx = sx / n - pcx, dy = sy / n - pcy, dz = sz / n - pcz;
      const dist = dx * dx + dy * dy + dz * dz;
      if (dist < bestDist) { bestDist = dist; bestGroup = g; }
    }
    if (bestGroup !== 0) {
      [chunkGroups[0], chunkGroups[bestGroup]] = [chunkGroups[bestGroup], chunkGroups[0]];
    }
  }

  const numChunks = chunkGroups.length;
  const zipFiles: Record<string, Uint8Array> = {};
  const chunksManifest: any[] = [];

  onProgress?.("Packaging .sp5 chunk files...", 80);
  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    const group = chunkGroups[chunkIdx];
    const count = group.length;

    // Chunk arrays, gathered by this group's (spatially local, not
    // contiguous) set of global-sort-order indices.
    const chunkXyz = new Float32Array(count * 3);
    const chunkOpacity = new Float32Array(count);
    const chunkRgb = new Float32Array(count * 3);
    const chunkSh1 = sortedSh1 ? new Float32Array(count * 9) : new Float32Array(0);
    const chunkScaleIndices = [new Uint8Array(count), new Uint8Array(count), new Uint8Array(count)];
    const chunkRotIndices = [new Uint8Array(count), new Uint8Array(count)];
    for (let k = 0; k < count; k++) {
      const i = group[k];
      chunkXyz[k * 3 + 0] = sortedXyz[i * 3 + 0];
      chunkXyz[k * 3 + 1] = sortedXyz[i * 3 + 1];
      chunkXyz[k * 3 + 2] = sortedXyz[i * 3 + 2];
      chunkOpacity[k] = sortedOpacity[i];
      chunkRgb[k * 3 + 0] = sortedRgb[i * 3 + 0];
      chunkRgb[k * 3 + 1] = sortedRgb[i * 3 + 1];
      chunkRgb[k * 3 + 2] = sortedRgb[i * 3 + 2];
      if (sortedSh1) {
        for (let s = 0; s < 9; s++) chunkSh1[k * 9 + s] = sortedSh1[i * 9 + s];
      }
      chunkScaleIndices[0][k] = scaleIndices[0][i];
      chunkScaleIndices[1][k] = scaleIndices[1][i];
      chunkScaleIndices[2][k] = scaleIndices[2][i];
      chunkRotIndices[0][k] = rotIndices[0][i];
      chunkRotIndices[1][k] = rotIndices[1][i];
    }

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

    // Positions (uncompressed)
    chunkMeta.xyz_uncompressed = packArray(chunkXyz);

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
      const x = chunkXyz[i * 3 + 0];
      const y = chunkXyz[i * 3 + 1];
      const z = chunkXyz[i * 3 + 2];
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
    count: numSplats,
    maxSh,
    lodTree: false,
    chunkSize: CHUNK_SIZE,
    chunks: chunksManifest
  };
  zipFiles["manifest.json"] = new TextEncoder().encode(JSON.stringify(globalManifest, null, 2));

  onProgress?.("Zipping output package...", 95);
  const zipped = fflate.zipSync(zipFiles);
  return zipped;
}
