import init_wasm, {
  sort_splats,
  sort32_splats,
  decode_to_gsplatarray,
  decode_to_csplatarray,
  decode_to_packedsplats,
  new_lod_tree,
  new_shared_lod_tree,
  init_lod_tree,
  dispose_lod_tree,
  traverse_lod_trees,
  dynamic_traverse_lod_trees,
  type ChunkDecoder,
  tiny_lod_packedsplats,
  bhatt_lod_packedsplats,
  update_lod_trees,
  decode_to_extsplats,
  tiny_lod_extsplats,
  bhatt_lod_extsplats,
  get_lod_tree_level,
  reconstruct_sp5_chunk,
  decode_huffman_fast,
} from "spark-rs";
import type { ExtResult, PackedResult, SplatEncoding } from "./defines";

const rpcHandlers = {
  sortSplats16,
  sortSplats32,
  loadPackedSplats,
  loadExtSplats,
  tinyLodPackedSplats,
  qualityLodPackedSplats,
  tinyLodExtSplats,
  qualityLodExtSplats,
  newLodTree,
  newSharedLodTree,
  initLodTree,
  disposeLodTree,
  updateLodTrees,
  traverseLodTrees,
  getLodTreeLevel,
  partitionDroppedMonolithic,
  decodeSp5Chunk,
  convertSplatToSp5,
};

async function onMessage(event: MessageEvent) {
  const {
    id,
    name,
    args,
  }: { id: unknown; name: keyof typeof rpcHandlers; args: unknown } =
    event.data;
  try {
    const handler = rpcHandlers[name] as (
      args: unknown,
      options: { sendStatus: (data: unknown) => void },
    ) => unknown | Promise<unknown>;
    if (!handler) {
      throw new Error(`Unknown worker RPC: ${name}`);
    }

    const sendStatus = (data: unknown) => {
      self.postMessage(
        { id, status: data },
        { transfer: getTransferable(data) },
      );
    };
    const result = await handler(args, { sendStatus });
    self.postMessage({ id, result }, { transfer: getTransferable(result) });
  } catch (error) {
    console.warn(`Worker error: ${error}`);
    self.postMessage({ id, error }, { transfer: getTransferable(error) });
  }
}

function sortSplats16({
  numSplats,
  readback,
  ordering,
}: {
  numSplats: number;
  readback: Uint16Array;
  ordering: Uint32Array;
}) {
  const activeSplats = sort_splats(numSplats, readback, ordering);
  return { activeSplats, readback, ordering };
}

function sortSplats32({
  numSplats,
  readback,
  ordering,
}: {
  numSplats: number;
  readback: Uint32Array;
  ordering: Uint32Array;
}) {
  const activeSplats = sort32_splats(numSplats, readback, ordering);
  return { activeSplats, readback, ordering };
}

async function fetchRange({
  url,
  requestHeader,
  withCredentials,
  offset,
  bytes,
}: {
  url: string;
  requestHeader?: Record<string, string>;
  withCredentials?: string;
  offset?: number;
  bytes?: number;
}): Promise<Uint8Array> {
  const request = new Request(url, {
    headers: requestHeader ? new Headers(requestHeader) : undefined,
    credentials: withCredentials ? "include" : "same-origin",
  });
  if (offset !== undefined && bytes !== undefined) {
    request.headers.set("Range", `bytes=${offset}-${offset + bytes - 1}`);
  }
  const response = await fetch(request);
  if (!response.ok || !response.body) {
    throw new Error(
      `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function decodeBytesUrl({
  decoder,
  fileBytes,
  url,
  requestHeader,
  withCredentials,
  chunked,
  chunkedLength,
  sendStatus,
}: {
  decoder: ChunkDecoder;
  fileBytes?: Uint8Array;
  url?: string;
  requestHeader?: Record<string, string>;
  withCredentials?: boolean;
  chunked?: boolean;
  chunkedLength?: number;
  sendStatus: (data: unknown) => void;
}) {
  if (fileBytes) {
    const CHUNK_SIZE = 1048576; // 1 MB
    for (let i = 0; i < fileBytes.length; i += CHUNK_SIZE) {
      decoder.push(
        fileBytes.subarray(i, Math.min(i + CHUNK_SIZE, fileBytes.length)),
      );
    }
  } else if (url) {
    const request = new Request(url, {
      headers: requestHeader ? new Headers(requestHeader) : undefined,
      credentials: withCredentials ? "include" : "same-origin",
    });

    const response = await fetch(request);
    if (!response.ok || !response.body) {
      throw new Error(
        `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
      );
    }
    const readStream = response.body.getReader();
    const contentLength = Number.parseInt(
      response.headers.get("Content-Length") || "0",
    );
    const total = Number.isNaN(contentLength) ? 0 : contentLength;
    let loaded = 0;

    while (true) {
      const { done, value } = await readStream.read();
      if (done) {
        readStream.releaseLock();
        break;
      }
      loaded += value.length;
      sendStatus({ loaded, total });

      decoder.push(value);
    }
  } else if (chunked) {
    let loaded = 0;
    const total = chunkedLength ?? 0;
    while (true) {
      const readNextChunk: Promise<Uint8Array> = new Promise((resolve) => {
        nextChunkWaiter = resolve;
      });
      sendStatus({ nextChunk: true });
      const nextChunk = await readNextChunk;

      if (nextChunk.length === 0) {
        break;
      }

      decoder.push(nextChunk);
      loaded += nextChunk.length;
      sendStatus({ progress: { loaded, total } });
    }
    if (total === 0) {
      sendStatus({ progress: { loaded, total: loaded } });
    }
  } else {
    throw new Error("No url or fileBytes provided");
  }

  const decoded = decoder.finish();
  return decoded;
}

type DecodedPackedResult = {
  numSplats: number;
  packed: Uint32Array;
  sh1?: Uint32Array;
  sh2?: Uint32Array;
  sh3?: Uint32Array;
  sh1Codes?: Uint32Array;
  sh2Codes?: Uint32Array;
  sh3Codes?: Uint32Array;
  lodTree?: Uint32Array;
  splatEncoding: SplatEncoding;
};

function toPackedResult(packed: DecodedPackedResult): PackedResult {
  return {
    numSplats: packed.numSplats,
    packedArray: packed.packed,
    extra: {
      sh1: packed.sh1,
      sh2: packed.sh2,
      sh3: packed.sh3,
      sh1Codes: packed.sh1Codes,
      sh2Codes: packed.sh2Codes,
      sh3Codes: packed.sh3Codes,
      lodTree: packed.lodTree,
    },
    splatEncoding: packed.splatEncoding,
  };
}

async function loadPackedSplats(
  {
    url,
    requestHeader,
    withCredentials,
    fileBytes,
    fileType,
    pathName,
    chunked,
    chunkedLength,
    encoding,
    lod,
    lodBase,
    lodAbove,
    nonLod,
    sh1Codes,
    sh2Codes,
    sh3Codes,
  }: {
    url?: string;
    requestHeader?: Record<string, string>;
    withCredentials?: boolean;
    fileBytes?: Uint8Array;
    fileType?: string;
    pathName?: string;
    chunked?: boolean;
    chunkedLength?: number;
    encoding?: SplatEncoding;
    lod?: boolean | "quality";
    lodBase?: number;
    lodAbove?: number;
    nonLod?: boolean;
    sh1Codes?: Uint32Array;
    sh2Codes?: Uint32Array;
    sh3Codes?: Uint32Array;
  },
  {
    sendStatus,
  }: {
    sendStatus: (data: unknown) => void;
  },
) {
  // console.log("loadPackedSplats", { url, requestHeader, withCredentials, fileBytes, fileType, pathName, stream, streamLength, encoding, lod, lodBase, lodAbove, nonLod });
  if (!lod) {
    const decoder = decode_to_packedsplats(
      fileType,
      pathName ?? url,
      encoding,
      sh1Codes,
      sh2Codes,
      sh3Codes,
    );
    const decoded = await decodeBytesUrl({
      decoder,
      fileBytes,
      url,
      requestHeader,
      withCredentials,
      chunked,
      chunkedLength,
      sendStatus,
    });
    const result = toPackedResult(decoded as DecodedPackedResult);
    // Same fix as decodeSp5Chunk: this plain (non-LOD) decode path is also what
    // partitionDroppedMonolithic's per-chunk .spz re-encode goes through when a
    // locally-dropped .ply file's chunks get loaded, and its result never
    // carries real lod tree data either -- see synthesizeFlatLodTreeIfMissing
    // for the full explanation.
    synthesizeFlatLodTreeIfMissing(result, result.numSplats, (i) => {
      const packed = result.packedArray;
      const word1 = packed[i * 4 + 1];
      const word2 = packed[i * 4 + 2];
      return [
        halfToFloat(word1 & 0xffff),
        halfToFloat((word1 >>> 16) & 0xffff),
        halfToFloat(word2 & 0xffff),
      ];
    });
    if (result.splatEncoding.lodOpacity) {
      return { lodSplats: result };
    }
    return result;
  }

  const decoder = decode_to_csplatarray(fileType, pathName ?? url, encoding);
  const decoded = await decodeBytesUrl({
    decoder,
    fileBytes,
    url,
    requestHeader,
    withCredentials,
    chunked,
    chunkedLength,
    sendStatus,
  });

  if (decoded.has_lod()) {
    const result = toPackedResult(
      decoded.to_packedsplats_lod() as DecodedPackedResult,
    );
    return { lodSplats: result };
  }

  if (lodAbove !== undefined) {
    if (decoded.len() < lodAbove) {
      return toPackedResult(decoded.to_packedsplats() as DecodedPackedResult);
    }
  }

  let result:
    | (ReturnType<typeof toPackedResult> & {
        lodSplats?: ReturnType<typeof toPackedResult>;
      })
    | { lodSplats?: ReturnType<typeof toPackedResult> } = {};

  // if (nonLod === true) {
  //   sendStatus({ orig: toPackedResult(packed as DecodedPackedResult) });
  // } else if (nonLod === "wait") {
  if (nonLod) {
    // Wait until LoD computation is complete before resolving full PackedSplats result
    result = toPackedResult(decoded.to_packedsplats() as DecodedPackedResult);
  }

  const initialSplats = decoded.len();
  const lodName = lod === "quality" ? "Bhatt" : "Tiny";
  console.log(
    `Loaded ${initialSplats} splats. Starting ${lodName} LoD build...`,
  );

  const lodStart = performance.now();
  if (lod === "quality") {
    const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.25));
    decoded.bhatt_lod(base);
  } else {
    const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
    decoded.tiny_lod(base, false);
  }
  const lodDuration = performance.now() - lodStart;

  console.log(
    `${lodName} LoD: ${initialSplats} -> ${decoded.len()} (${lodDuration} ms)`,
  );

  const lodPacked = decoded.to_packedsplats_lod();
  result.lodSplats = toPackedResult(lodPacked as DecodedPackedResult);
  return result;
}

type DecodedExtResult = {
  numSplats: number;
  ext0: Uint32Array;
  ext1: Uint32Array;
  sh1?: Uint32Array;
  sh2?: Uint32Array;
  sh3a?: Uint32Array;
  sh3b?: Uint32Array;
  sh1Codes?: Uint32Array;
  sh2Codes?: Uint32Array;
  sh3Codes?: [Uint32Array, Uint32Array];
  lodTree?: Uint32Array;
};

function toExtResult(packed: DecodedExtResult): ExtResult {
  return {
    numSplats: packed.numSplats,
    extArrays: [packed.ext0, packed.ext1],
    extra: {
      sh1: packed.sh1,
      sh2: packed.sh2,
      sh3a: packed.sh3a,
      sh3b: packed.sh3b,
      sh1Codes: packed.sh1Codes,
      sh2Codes: packed.sh2Codes,
      sh3Codes: packed.sh3Codes,
      lodTree: packed.lodTree,
    },
  };
}

async function loadExtSplats(
  {
    url,
    requestHeader,
    withCredentials,
    fileBytes,
    fileType,
    pathName,
    chunked,
    chunkedLength,
    lod,
    lodBase,
    lodAbove,
    nonLod,
    sh1Codes,
    sh2Codes,
    sh3Codes,
  }: {
    url?: string;
    requestHeader?: Record<string, string>;
    withCredentials?: boolean;
    fileBytes?: Uint8Array;
    fileType?: string;
    pathName?: string;
    chunked?: boolean;
    chunkedLength?: number;
    lod?: boolean | "quality";
    lodBase?: number;
    lodAbove?: number;
    nonLod?: boolean;
    sh1Codes?: Uint32Array;
    sh2Codes?: Uint32Array;
    sh3Codes?: [Uint32Array, Uint32Array];
  },
  {
    sendStatus,
  }: {
    sendStatus: (data: unknown) => void;
  },
) {
  // console.log("loadExtSplats", { url, requestHeader, withCredentials, fileBytes, fileType, pathName, stream, streamLength, lod, lodBase, lodAbove, nonLod });
  if (!lod) {
    const decoder = decode_to_extsplats(
      fileType,
      pathName ?? url,
      sh1Codes,
      sh2Codes,
      sh3Codes,
    );
    const decoded = await decodeBytesUrl({
      decoder,
      fileBytes,
      url,
      requestHeader,
      withCredentials,
      chunked,
      chunkedLength,
      sendStatus,
    });
    const result = toExtResult(decoded as DecodedExtResult);
    synthesizeFlatLodTreeIfMissing(result, result.numSplats, (i) => {
      const packed = result.extArrays[0];
      const word1 = packed[i * 4 + 1];
      const word2 = packed[i * 4 + 2];
      return [
        halfToFloat(word1 & 0xffff),
        halfToFloat((word1 >>> 16) & 0xffff),
        halfToFloat(word2 & 0xffff),
      ];
    });
    if (result.extra.lodTree) {
      return { lodSplats: result };
    }
    return result;
  }

  const decoder = decode_to_gsplatarray(fileType, pathName ?? url);
  const decoded = await decodeBytesUrl({
    decoder,
    fileBytes,
    url,
    requestHeader,
    withCredentials,
    chunked,
    chunkedLength,
    sendStatus,
  });

  if (decoded.has_lod()) {
    return {
      lodSplats: toExtResult(decoded.to_extsplats_lod() as DecodedExtResult),
    };
  }

  if (lodAbove !== undefined) {
    if (decoded.len() < lodAbove) {
      return toExtResult(decoded.to_extsplats() as DecodedExtResult);
    }
  }

  let result:
    | (ReturnType<typeof toExtResult> & {
        lodSplats?: ReturnType<typeof toExtResult>;
      })
    | { lodSplats?: ReturnType<typeof toExtResult> } = {};

  if (nonLod) {
    // Wait until LoD computation is complete before resolving full PackedSplats result
    result = toExtResult(decoded.to_extsplats() as DecodedExtResult);
  }

  const initialSplats = decoded.len();
  const lodName = lod === "quality" ? "Bhatt" : "Tiny";
  console.log(
    `Loaded ${initialSplats} splats. Starting ${lodName} LoD build...`,
  );

  const lodStart = performance.now();
  if (lod === "quality") {
    const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.75));
    decoded.bhatt_lod(base);
  } else {
    const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
    decoded.tiny_lod(base, false);
  }
  const lodDuration = performance.now() - lodStart;

  console.log(
    `${lodName} LoD: ${initialSplats} -> ${decoded.len()} (${lodDuration} ms)`,
  );

  const lodPacked = decoded.to_extsplats_lod();
  result.lodSplats = toExtResult(lodPacked as DecodedExtResult);
  return result;
}

async function tinyLodPackedSplats({
  numSplats,
  packedArray,
  extra,
  lodBase,
  rgba,
  encoding,
}: {
  numSplats: number;
  packedArray: Uint32Array;
  extra?: Record<string, unknown>;
  lodBase?: number;
  rgba?: Uint8Array;
  encoding: SplatEncoding;
}) {
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
  const lodStart = performance.now();
  const filter = false;
  const decoded = tiny_lod_packedsplats(
    numSplats,
    packedArray,
    extra as object,
    base,
    filter,
    rgba,
    encoding,
  );
  const lodDuration = performance.now() - lodStart;
  const result = toPackedResult(decoded as DecodedPackedResult);
  console.log(
    `Tiny LoD: ${numSplats} -> ${result.numSplats} (${lodDuration} ms)`,
  );
  return result;
}

async function qualityLodPackedSplats({
  numSplats,
  packedArray,
  extra,
  lodBase,
  rgba,
  encoding,
}: {
  numSplats: number;
  packedArray: Uint32Array;
  extra?: Record<string, unknown>;
  lodBase?: number;
  rgba?: Uint8Array;
  encoding: SplatEncoding;
}) {
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.75));
  const lodStart = performance.now();
  const decoded = bhatt_lod_packedsplats(
    numSplats,
    packedArray,
    extra as object,
    base,
    rgba,
    encoding,
  );
  const lodDuration = performance.now() - lodStart;
  const result = toPackedResult(decoded as DecodedPackedResult);
  console.log(
    `Bhatt LoD: ${numSplats} -> ${result.numSplats} (${lodDuration} ms)`,
  );
  return result;
}

async function tinyLodExtSplats({
  numSplats,
  extArrays,
  extra,
  lodBase,
  rgba,
  encoding,
}: {
  numSplats: number;
  extArrays: [Uint32Array, Uint32Array];
  extra?: Record<string, unknown>;
  lodBase?: number;
  rgba?: Uint8Array;
  encoding: SplatEncoding;
}) {
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
  const lodStart = performance.now();
  const filter = false;
  const decoded = tiny_lod_extsplats(
    numSplats,
    extArrays[0],
    extArrays[1],
    extra as object,
    base,
    filter,
    rgba,
  );
  const lodDuration = performance.now() - lodStart;
  const result = toExtResult(decoded as DecodedExtResult);
  console.log(
    `Tiny LoD: ${numSplats} -> ${result.numSplats} (${lodDuration} ms)`,
  );
  return result;
}

async function qualityLodExtSplats({
  numSplats,
  extArrays,
  extra,
  lodBase,
  rgba,
  encoding,
}: {
  numSplats: number;
  extArrays: [Uint32Array, Uint32Array];
  extra?: Record<string, unknown>;
  lodBase?: number;
  rgba?: Uint8Array;
  encoding: SplatEncoding;
}) {
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.75));
  const lodStart = performance.now();
  const decoded = bhatt_lod_extsplats(
    numSplats,
    extArrays[0],
    extArrays[1],
    extra as object,
    base,
    rgba,
  );
  const lodDuration = performance.now() - lodStart;
  const result = toExtResult(decoded as DecodedExtResult);
  console.log(
    `Bhatt LoD: ${numSplats} -> ${result.numSplats} (${lodDuration} ms)`,
  );
  return result;
}

function newLodTree({
  capacity,
}: {
  capacity: number;
}) {
  const { lodId } = new_lod_tree(capacity) as { lodId: number };
  return { lodId };
}

function newSharedLodTree({
  lodId,
}: {
  lodId: number;
}) {
  const { lodId: newLodId } = new_shared_lod_tree(lodId) as { lodId: number };
  return { lodId: newLodId };
}

function initLodTree({
  numSplats,
  lodTree,
}: {
  numSplats: number;
  lodTree: Uint32Array;
}) {
  const { lodId, chunkToPage } = init_lod_tree(numSplats, lodTree) as {
    lodId: number;
    chunkToPage: Uint32Array;
  };
  return { lodId, chunkToPage };
}

function disposeLodTree({ lodId }: { lodId: number }) {
  dispose_lod_tree(lodId);
}

function updateLodTrees({
  ranges,
}: {
  ranges: {
    lodId: number;
    pageBase: number;
    chunkBase: number;
    count: number;
    lodTreeData?: Uint32Array;
  }[];
}) {
  const lodIds = new Uint32Array(ranges.map(({ lodId }) => lodId));
  const pageBases = new Uint32Array(ranges.map(({ pageBase }) => pageBase));
  const chunkBases = new Uint32Array(ranges.map(({ chunkBase }) => chunkBase));
  const counts = new Uint32Array(ranges.map(({ count }) => count));
  const lodTreeData = ranges.map(({ lodTreeData }) => lodTreeData);

  const result = update_lod_trees(
    lodIds,
    pageBases,
    chunkBases,
    counts,
    lodTreeData,
  );
}

function traverseLodTrees({
  maxSplats,
  pixelScaleLimit,
  lastPixelLimit,
  instances,
  traverseMode,
  pageBounds,
}: {
  maxSplats: number;
  pixelScaleLimit: number;
  lastPixelLimit?: number;
  instances: Record<
    string,
    {
      instanceId: string;
      lodId: number;
      rootPage?: number;
      viewToObjectCols: number[];
      lodScale: number;
      behindFoveate: number;
      coneFov0: number;
      coneFov: number;
      coneFoveate: number;
    }
  >;
  traverseMode: "dynamic" | "standard";
  pageBounds?: Float32Array; // 5*N per page: (cx,cy,cz,radius,max_node_size)
}) {
  const keyInstances = Object.entries(instances);
  const lodIds = new Uint32Array(
    keyInstances.map(([_key, instance]) => instance.lodId),
  );
  const rootPages = new Uint32Array(
    keyInstances.map(([_key, instance]) => instance.rootPage ?? 0xffffffff),
  );
  const viewToObjects = new Float32Array(
    keyInstances.flatMap(([_key, instance]) => {
      if (instance.viewToObjectCols.length !== 16) {
        throw new Error("Incorrect array size for viewToObjectCols");
      }
      return instance.viewToObjectCols;
    }),
  );
  const lodScales = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.lodScale),
  );
  const behindFoveates = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.behindFoveate),
  );
  const coneFov0s = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.coneFov0),
  );
  const coneFovs = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.coneFov),
  );
  const coneFoveates = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.coneFoveate),
  );

  const lodFunction =
    traverseMode === "dynamic"
      ? dynamic_traverse_lod_trees
      : traverse_lod_trees;
  const result = lodFunction(
    maxSplats,
    pixelScaleLimit,
    lastPixelLimit,
    lodIds,
    rootPages,
    viewToObjects,
    lodScales,
    behindFoveates,
    coneFoveates,
    coneFov0s,
    coneFovs,
    pageBounds ?? new Float32Array(0),
  ) as {
    instanceIndices: {
      lodId: number;
      numSplats: number;
      indices: Uint32Array;
    }[];
    chunks: [number, number][];
    pixelLimit?: number;
  };
  const { instanceIndices, chunks, pixelLimit } = result;

  const indices = keyInstances.reduce(
    (indices, [key, _instance], index) => {
      indices[key] = instanceIndices[index];
      return indices;
    },
    {} as Record<
      string,
      { lodId: number; numSplats: number; indices: Uint32Array }
    >,
  );
  // console.log(`traverseLodTrees: instanceIndices=${instanceIndices.length}`);
  // console.log(`traverseLodTrees: chunks=${chunks.length}`, JSON.stringify(chunks));
  return {
    keyIndices: indices,
    chunks,
    pixelLimit,
  };
}

function getLodTreeLevel({
  lodId,
  level,
}: {
  lodId: number;
  level: number;
}) {
  return get_lod_tree_level(lodId, level) as { indices: Uint32Array };
}

let nextChunkWaiter = (_chunk: Uint8Array) => {};

async function nextChunk({ chunk }: { chunk: Uint8Array }) {
  nextChunkWaiter(chunk);
}

// Recursively finds all ArrayBuffers in an object and returns them as an array
// to use as transferable objects to send between workers.
function getTransferable(ctx: unknown): Transferable[] {
  const buffers: Transferable[] = [];
  const seen = new Set();

  function traverse(obj: unknown) {
    if (obj && typeof obj === "object" && !seen.has(obj)) {
      seen.add(obj);

      if (obj instanceof ArrayBuffer) {
        buffers.push(obj);
      } else if (ArrayBuffer.isView(obj)) {
        // Handles TypedArrays and DataView
        buffers.push(obj.buffer as ArrayBuffer);
      } else if (Array.isArray(obj)) {
        obj.forEach(traverse);
      } else {
        Object.values(obj).forEach(traverse);
      }
    }
  }

  traverse(ctx);
  return buffers;
}

async function partitionDroppedMonolithic(
  {
    fileBytes,
    fileType,
    pathName,
    lodBase,
  }: {
    fileBytes: Uint8Array;
    fileType?: string;
    pathName?: string;
    lodBase?: number;
  },
  {
    sendStatus,
  }: {
    sendStatus: (data: unknown) => void;
  },
) {
  console.log(`Decoding monolithic file ${pathName || "local-drop"}...`);
  const decoder = decode_to_gsplatarray(fileType, pathName);
  const gsplatArray = (await decodeBytesUrl({
    decoder,
    fileBytes,
    sendStatus,
  })) as any;

  const numSplats = gsplatArray.len();
  console.log(`Loaded ${numSplats} splats. Centering coordinates...`);
  const centerShift = gsplatArray.center();
  console.log(`Centered coordinates by: (${centerShift[0].toFixed(3)}, ${centerShift[1].toFixed(3)}, ${centerShift[2].toFixed(3)})`);

  console.log(`Starting Tiny LoD build...`);

  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
  gsplatArray.tiny_lod(base, false);

  const totalNumSplats = gsplatArray.len();
  const CHUNK_SIZE = 65536;
  const numChunks = Math.ceil(totalNumSplats / CHUNK_SIZE);
  console.log(
    `Slicing and compressing ${totalNumSplats} splats into ${numChunks} chunks...`,
  );

  const chunksInfo: { chunk: number; bytes: Uint8Array; numSplats: number }[] =
    [];

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const start = chunk * CHUNK_SIZE;
    const count = Math.min(totalNumSplats - start, CHUNK_SIZE);

    const subset = gsplatArray.clone_subset(start, count);
    const spzBytes = subset.to_spz();
    subset.free();

    if (chunk % 4 === 0 || chunk === numChunks - 1) {
      console.log(
        `[partition] chunk ${chunk + 1}/${numChunks}: ${count} splats, ${spzBytes.length} bytes`,
      );
    }

    chunksInfo.push({
      chunk,
      bytes: spzBytes,
      numSplats: count,
    });
  }

  const manifest = {
    version: 1,
    type: "gsplat",
    count: totalNumSplats,
    maxSh: gsplatArray.maxShDegree,
    lodTree: true,
    chunkSize: CHUNK_SIZE,
    chunks: chunksInfo.map((info) => ({
      offset: 0,
      bytes: info.bytes.length,
      count: info.numSplats,
    })),
  };

  gsplatArray.free();

  return {
    manifest,
    chunks: chunksInfo,
  };
}

let tmc3Module: any = null;
async function ensureTmc3Loaded() {
  if (tmc3Module) return;
  const tmc3WasmUrl = new URL("../examples/viewer/tmc3.wasm", import.meta.url)
    .href;
  const tmc3JsUrl = new URL("../examples/viewer/tmc3.js", import.meta.url).href;
  (self as any).Module = {
    INITIAL_MEMORY: 536870912,
    ALLOW_MEMORY_GROWTH: true,
    locateFile: (path: string) => {
      if (path.endsWith(".wasm")) return tmc3WasmUrl;
      return path;
    },
    onRuntimeInitialized: () => {
      tmc3Module = (self as any).Module;
    },
  };
  (self as any).importScripts(tmc3JsUrl);
  while (!tmc3Module) {
    await new Promise((r) => setTimeout(r, 5));
  }
}

export async function decodeSp5Chunk({
  chunkBytes,
  siblingChunks,
}: {
  chunkBytes: Uint8Array;
  // Only passed for chunk 0 -- see synthesizeFlatLodTreeIfMissing's doc
  // comment for why chunk 0's tree specifically needs to reach every sibling.
  siblingChunks?: { chunkIndex: number; center: [number, number, number]; size: number }[];
}) {
  const view = new DataView(chunkBytes.buffer, chunkBytes.byteOffset);
  const magic = view.getUint32(0, true);
  if (magic !== 0x355a5053) {
    throw new Error("Invalid .sp5 chunk magic");
  }
  const jsonSize = view.getUint32(4, true);
  const jsonBytes = chunkBytes.subarray(8, 8 + jsonSize);
  const manifest = JSON.parse(new TextDecoder().decode(jsonBytes));

  const decodeStart = performance.now();
  const binaryPayload = chunkBytes.subarray(8 + jsonSize);

  let count = manifest.count;
  const xyzRawFloat = new Float32Array(count * 3);

  if (manifest.is_uncompressed) {
    const xyzBytes = binaryPayload.subarray(
      manifest.xyz_uncompressed.offset,
      manifest.xyz_uncompressed.offset + manifest.xyz_uncompressed.length,
    );
    // Convert float16 to float32
    let alignedBytes = xyzBytes;
    if (xyzBytes.byteOffset % 2 !== 0) {
      alignedBytes = new Uint8Array(xyzBytes.length);
      alignedBytes.set(xyzBytes);
    }
    const u16 = new Uint16Array(
      alignedBytes.buffer,
      alignedBytes.byteOffset,
      alignedBytes.byteLength / 2,
    );
    for (let i = 0; i < u16.length; i++) {
      xyzRawFloat[i] = halfToFloat(u16[i]);
    }
    // Undo converter.ts's chunk-local normalization (see its chunkXyz comment):
    // positions were packed as (raw - chunk_center) / chunk_scale to keep them
    // within f16 range even when a chunk (e.g. chunk 0's coarsest LOD levels)
    // spans the entire scene. Older files without these fields packed raw
    // absolute positions directly, so default to a no-op transform.
    const chunkCenter: [number, number, number] = manifest.chunk_center ?? [0, 0, 0];
    const chunkScale: number = manifest.chunk_scale ?? 1;
    if (chunkScale !== 1 || chunkCenter[0] !== 0 || chunkCenter[1] !== 0 || chunkCenter[2] !== 0) {
      for (let i = 0; i < count; i++) {
        xyzRawFloat[i * 3 + 0] = xyzRawFloat[i * 3 + 0] * chunkScale + chunkCenter[0];
        xyzRawFloat[i * 3 + 1] = xyzRawFloat[i * 3 + 1] * chunkScale + chunkCenter[1];
        xyzRawFloat[i * 3 + 2] = xyzRawFloat[i * 3 + 2] * chunkScale + chunkCenter[2];
      }
    }
  } else {
    const gpccBytes = binaryPayload.subarray(
      manifest.gpcc_offset,
      manifest.gpcc_offset + manifest.gpcc_size,
    );

    await ensureTmc3Loaded();
    const fileSystem = tmc3Module.FS;
    const mainFunc = tmc3Module.callMain;

    const headView = new DataView(gpccBytes.buffer, gpccBytes.byteOffset, 24);
    const meansMin = [
      headView.getFloat32(0, true),
      headView.getFloat32(4, true),
      headView.getFloat32(8, true),
    ];
    const meansMax = [
      headView.getFloat32(12, true),
      headView.getFloat32(16, true),
      headView.getFloat32(20, true),
    ];

    const gpccStreamBytes = gpccBytes.subarray(24);
    fileSystem.writeFile("/xyz.bin", gpccStreamBytes);

    try {
      mainFunc([
        "--mode=1",
        "--compressedStreamPath=/xyz.bin",
        "--reconstructedDataPath=/xyz.ply",
      ]);
    } catch (err: any) {
      if (err !== 0 && err.status !== 0 && err.name !== "ExitStatus") {
        throw new Error("TMC3 failed to decode: " + err);
      }
    }

    let plyBytes: Uint8Array;
    try {
      plyBytes = fileSystem.readFile("/xyz.ply");
    } finally {
      try {
        fileSystem.unlink("/xyz.bin");
      } catch (e) {}
      try {
        fileSystem.unlink("/xyz.ply");
      } catch (e) {}
    }

    const headerText = new TextDecoder().decode(plyBytes.subarray(0, 2000));
    const match = headerText.match(/end_header\r?\n/);
    if (!match) throw new Error("Could not find end_header in PLY");
    const headerEnd = match.index! + match[0].length;

    const countMatch = headerText
      .substring(0, headerEnd)
      .match(/element vertex (\d+)/);
    count = countMatch ? Number.parseInt(countMatch[1]) : 0;

    const isUint16 =
      headerText.includes("property uint16") ||
      headerText.includes("property ushort");
    const isInt32 =
      headerText.includes("property int32") ||
      headerText.match(/property int\s+x/);
    const isFloat64 =
      headerText.includes("property float64") ||
      headerText.includes("property double");
    const bytesPerVertex = isFloat64 ? 24 : isUint16 ? 6 : 12;

    const plyView = new DataView(
      plyBytes.buffer,
      plyBytes.byteOffset + headerEnd,
    );

    for (let i = 0; i < count; i++) {
      let x, y, z;
      if (isFloat64) {
        x = plyView.getFloat64(i * bytesPerVertex + 0, true);
        y = plyView.getFloat64(i * bytesPerVertex + 8, true);
        z = plyView.getFloat64(i * bytesPerVertex + 16, true);
      } else if (isInt32) {
        x = plyView.getInt32(i * bytesPerVertex + 0, true);
        y = plyView.getInt32(i * bytesPerVertex + 4, true);
        z = plyView.getInt32(i * bytesPerVertex + 8, true);
      } else if (isUint16) {
        x = plyView.getUint16(i * bytesPerVertex + 0, true);
        y = plyView.getUint16(i * bytesPerVertex + 2, true);
        z = plyView.getUint16(i * bytesPerVertex + 4, true);
      } else {
        x = plyView.getFloat32(i * bytesPerVertex + 0, true);
        y = plyView.getFloat32(i * bytesPerVertex + 4, true);
        z = plyView.getFloat32(i * bytesPerVertex + 8, true);
      }

      // Scale and shift coordinates by meansMin / meansMax back to original bbox
      const normX = x / 65535.0;
      const normY = y / 65535.0;
      const normZ = z / 65535.0;

      xyzRawFloat[i * 3 + 0] =
        normX * (meansMax[0] - meansMin[0]) + meansMin[0];
      xyzRawFloat[i * 3 + 1] =
        normY * (meansMax[1] - meansMin[1]) + meansMin[1];
      xyzRawFloat[i * 3 + 2] =
        normZ * (meansMax[2] - meansMin[2]) + meansMin[2];
    }
  }

  // halfToFloat moved to module scope (used by both decodeSp5Chunk and
  // loadPackedSplats's flat-lod-tree synthesis, see synthesizeFlatLodTreeIfMissing below).

  // Cache prefix lookup tables per Huffman table to avoid rebuilding
  // on every chunk decode (tables are shared across all chunks).
  const _huffmanLutCache = new Map<Record<string, [number, number]>, Uint16Array>();

  function decodeHuffman(
    bytes: Uint8Array,
    htable: Record<string, [number, number]>,
    count: number,
  ) {
    let lut = _huffmanLutCache.get(htable);
    if (!lut) {
      let maxLen = 0;
      const entries: [number, number, number][] = [];
      for (const [symStr, [len, bits]] of Object.entries(htable)) {
        const sym = Number.parseInt(symStr);
        entries.push([sym, len, bits]);
        if (len > maxLen) maxLen = len;
      }
      if (maxLen > 16) {
        // Fallback: codes > 16 bits use the JS bit-loop
        const fallbackTable = new Map<string, number>();
        for (const [s, l, b] of entries) fallbackTable.set(`${b},${l}`, s);
        const fallbackLut = new Uint16Array(0);
        (fallbackLut as any)._fallback = fallbackTable;
        _huffmanLutCache.set(htable, fallbackLut);
        lut = fallbackLut;
      } else {
        const lutSize = 1 << maxLen;
        lut = new Uint16Array(lutSize);
        lut.fill(0xFFFF);
        const shift = maxLen;
        for (const [symbol, len, bits] of entries) {
          const spread = 1 << (shift - len);
          const base = bits << (shift - len);
          for (let i = 0; i < spread; i++) {
            lut[base + i] = ((len << 8) | symbol) as number;
          }
        }
        _huffmanLutCache.set(htable, lut);
      }
    }

    const fallbackTable = (lut as any)._fallback as Map<string, number> | undefined;
    if (fallbackTable) {
      const out = new Uint16Array(count);
      let outIdx = 0, currentBits = 0, currentLen = 0, byteIdx = 0, bitIdx = 7;
      while (outIdx < count && byteIdx < bytes.length) {
        const bit = (bytes[byteIdx] >> bitIdx) & 1;
        bitIdx--; if (bitIdx < 0) { bitIdx = 7; byteIdx++; }
        currentBits = (currentBits << 1) | bit; currentLen++;
        const key = `${currentBits},${currentLen}`;
        if (fallbackTable.has(key)) { out[outIdx++] = fallbackTable.get(key)!; currentBits = 0; currentLen = 0; }
      }
      return out;
    }

    // Copy bytes to avoid detaching the shared binaryPayload ArrayBuffer
    // when wasm-bindgen accesses the memory.
    const bytesCopy = bytes.slice();
    const decoded = decode_huffman_fast(bytesCopy, lut, count);
    const out = new Uint16Array(count);
    out.set(decoded.subarray(0, Math.min(decoded.length, count)));
    return out;
  }

  function getBinaryPart(meta: { offset: number; length: number }) {
    return binaryPayload.subarray(meta.offset, meta.offset + meta.length);
  }

  const scaleIndices: number[] = [];
  for (const hmeta of manifest.scale_index_huffman) {
    const hBytes = getBinaryPart(hmeta);
    const decoded = decodeHuffman(hBytes, hmeta.huffman_table, count);
    for (let i = 0; i < count; i++) scaleIndices.push(decoded[i]);
  }

  const rotationIndices: number[] = [];
  for (const hmeta of manifest.rotation_index_huffman) {
    const hBytes = getBinaryPart(hmeta);
    const decoded = decodeHuffman(hBytes, hmeta.huffman_table, count);
    for (let i = 0; i < count; i++) rotationIndices.push(decoded[i]);
  }

  const appIndices: number[] = [];
  for (const hmeta of manifest.app_index_huffman) {
    const hBytes = getBinaryPart(hmeta);
    const decoded = decodeHuffman(hBytes, hmeta.huffman_table, count);
    for (let i = 0; i < count; i++) appIndices.push(decoded[i]);
  }
  const huffmanEnd = performance.now();
  console.log(
    `[decodeSp5Chunk] PERF: huffman-decode phase = ${(huffmanEnd - decodeStart).toFixed(1)}ms ` +
    `(${count} splats, ${manifest.scale_index_huffman.length} scale + ${manifest.rotation_index_huffman.length} rot streams)`,
  );

  function float16ArrayToFloat32Array(bytes: Uint8Array) {
    let alignedBytes = bytes;
    if (bytes.byteOffset % 2 !== 0) {
      alignedBytes = new Uint8Array(bytes.length);
      alignedBytes.set(bytes);
    }
    const u16 = new Uint16Array(
      alignedBytes.buffer,
      alignedBytes.byteOffset,
      alignedBytes.byteLength / 2,
    );
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      out[i] = halfToFloat(u16[i]);
    }
    return out;
  }

  const scaleCbFlat = new Float32Array(manifest.scale_codebook.length * 256);
  manifest.scale_codebook.forEach((meta: any, idx: number) => {
    scaleCbFlat.set(float16ArrayToFloat32Array(getBinaryPart(meta)), idx * 256);
  });

  const rotationCbFlat = new Float32Array(
    manifest.rotation_codebook.length * 512,
  );
  manifest.rotation_codebook.forEach((meta: any, idx: number) => {
    rotationCbFlat.set(
      float16ArrayToFloat32Array(getBinaryPart(meta)),
      idx * 512,
    );
  });

  const appCbFlat = new Float32Array(manifest.app_codebook.length * 512);
  manifest.app_codebook.forEach((meta: any, idx: number) => {
    appCbFlat.set(float16ArrayToFloat32Array(getBinaryPart(meta)), idx * 512);
  });

  const mlpCont = float16ArrayToFloat32Array(getBinaryPart(manifest.mlp_cont));
  const mlpDc = float16ArrayToFloat32Array(getBinaryPart(manifest.mlp_dc));
  const mlpSh = float16ArrayToFloat32Array(getBinaryPart(manifest.mlp_sh));
  const mlpOpacity = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_opacity),
  );

  let mlpOffsetW0 = new Float32Array(0);
  let mlpOffsetB0 = new Float32Array(0);
  let mlpOffsetW1 = new Float32Array(0);
  let mlpOffsetB1 = new Float32Array(0);
  let mlpOffsetW2 = new Float32Array(0);
  let mlpOffsetB2 = new Float32Array(0);
  let mlpOffsetW3 = new Float32Array(0);
  let mlpOffsetB3 = new Float32Array(0);

  if (manifest.mlp_offset && Object.keys(manifest.mlp_offset).length > 0) {
    mlpOffsetW0 = float16ArrayToFloat32Array(
      getBinaryPart(manifest.mlp_offset["main.0.weight"]),
    );
    mlpOffsetB0 = float16ArrayToFloat32Array(
      getBinaryPart(manifest.mlp_offset["main.0.bias"]),
    );
    mlpOffsetW1 = float16ArrayToFloat32Array(
      getBinaryPart(manifest.mlp_offset["main.2.weight"]),
    );
    mlpOffsetB1 = float16ArrayToFloat32Array(
      getBinaryPart(manifest.mlp_offset["main.2.bias"]),
    );
    mlpOffsetW2 = float16ArrayToFloat32Array(
      getBinaryPart(manifest.mlp_offset["main.4.weight"]),
    );
    mlpOffsetB2 = float16ArrayToFloat32Array(
      getBinaryPart(manifest.mlp_offset["main.4.bias"]),
    );
    mlpOffsetW3 = float16ArrayToFloat32Array(
      getBinaryPart(manifest.mlp_offset["shs_output.0.weight"]),
    );
    mlpOffsetB3 = float16ArrayToFloat32Array(
      getBinaryPart(manifest.mlp_offset["shs_output.0.bias"]),
    );
  }

  const gsplatArray = reconstruct_sp5_chunk(
    xyzRawFloat,
    new Uint16Array(scaleIndices),
    new Uint16Array(rotationIndices),
    new Uint16Array(appIndices),
    scaleCbFlat,
    rotationCbFlat,
    appCbFlat,
    mlpCont,
    mlpDc,
    mlpSh,
    mlpOpacity,
    mlpOffsetW0,
    mlpOffsetB0,
    mlpOffsetW1,
    mlpOffsetB1,
    mlpOffsetW2,
    mlpOffsetB2,
    mlpOffsetW3,
    mlpOffsetB3,
  );
  const wasmEnd = performance.now();
  console.log(
    `[decodeSp5Chunk] PERF: WASM reconstruct = ${(wasmEnd - huffmanEnd).toFixed(1)}ms, ` +
    `total decode = ${(wasmEnd - decodeStart).toFixed(1)}ms`,
  );

  const result = toPackedResult(
    gsplatArray.to_packedsplats(null as any) as any,
  );
  gsplatArray.free();

  // Diagnostic: confirm decoded geometry is actually non-degenerate before it ever
  // reaches the render/camera pipeline. A camera-fit bug and an "all positions are
  // zero" decode bug both manifest as a black screen with no thrown error -- this
  // distinguishes them directly from the console instead of guessing.
  {
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let minZ = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < count; i++) {
      const x = xyzRawFloat[i * 3 + 0];
      const y = xyzRawFloat[i * 3 + 1];
      const z = xyzRawFloat[i * 3 + 2];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    const isDegenerate =
      maxX - minX < 1e-9 && maxY - minY < 1e-9 && maxZ - minZ < 1e-9;
    console.log(
      `[decodeSp5Chunk] decoded ${count} splats, position range: ` +
        `x=[${minX.toFixed(3)}, ${maxX.toFixed(3)}] ` +
        `y=[${minY.toFixed(3)}, ${maxY.toFixed(3)}] ` +
        `z=[${minZ.toFixed(3)}, ${maxZ.toFixed(3)}]${
          isDegenerate
            ? " -- WARNING: all positions identical/degenerate, this chunk will not be visible regardless of camera position"
            : ""
        }`,
    );

    if (manifest.lod_tree) {
      const lodTreeBytes = getBinaryPart(manifest.lod_tree);
      let alignedBytes = lodTreeBytes;
      if (lodTreeBytes.byteOffset % 4 !== 0) {
        alignedBytes = new Uint8Array(lodTreeBytes.length);
        alignedBytes.set(lodTreeBytes);
      }
      const rawLodTree = new Uint32Array(
        alignedBytes.buffer,
        alignedBytes.byteOffset,
        alignedBytes.byteLength / 4,
      );

      // Never stitch real hierarchical LOD trees with sibling pointers.
      // Sibling pointers were a workaround for flat synthetic trees to discover other chunks,
      // but under the real hierarchy, chunks are natively connected by cross-chunk child starts.
      result.extra.lodTree = rawLodTree;
      result.extra.syntheticLodTree = false;
    } else {
      result.extra.syntheticLodTree = true;
      synthesizeFlatLodTreeIfMissing(
        result,
        count,
        (i) => [
          xyzRawFloat[i * 3 + 0],
          xyzRawFloat[i * 3 + 1],
          xyzRawFloat[i * 3 + 2],
        ],
        siblingChunks,
      );
    }
  }

  return result;
}

// Root-cause fix for the traverse_lod_trees panic ("index out of bounds: the
// len is 0 but the index is 0"): flat/non-hierarchical chunks (SP5's
// `lodTree: false` manifest, and locally-dropped .ply files partitioned via
// `partitionDroppedMonolithic`, whose per-chunk .spz re-encode has no field
// for LOD hierarchy data even though its manifest optimistically claims
// `lodTree: true`) never populate `extra.lodTree`, so the shared WASM lod
// tree's `splats` Vec stays empty once this mesh becomes eligible for
// traversal, and reading `splats[0]` panics.
//
// A single dummy "leaf" node is NOT sufficient -- `numSplats` on the JS side
// is literally the count of individual output entries from that traversal,
// so a lone child_count=0 node would report exactly 1 splat total, silently
// under-rendering everything instead of crashing. This synthesizes a valid
// one-level tree for the chunk instead: one root entry (this chunk's
// bounding volume, child_count = count-1) plus one real leaf entry per
// remaining splat (child_count=0, that splat's own position), matching the
// exact 4-word-per-LodSplat format `set_lod_tree_data` parses. This is sized
// to fit: LodSplat.child_count is a u16 (max 65535) and a page holds at most
// 65536 splats, so "count-1" always fits.
//
// NOTE: the resulting root entry's `child_start` is chunk-relative (value 1,
// meaning "the next slot in this same chunk") -- SplatPager.ts's
// processFetched knows the real chunk index and patches it to the correct
// absolute `(chunk << 16) | 1` address before this reaches the WASM lod
// tree.
//
// Cross-chunk stitching (siblingChunks parameter): each chunk's synthesized
// tree only ever covers that ONE chunk's own splats -- with no additional
// connectivity, chunk 0's tree has no edge to chunk 1's tree, chunk 1's tree
// has no edge to chunk 2's, etc. Since traverse_lod_trees/
// dynamic_traverse_lod_trees (rust/spark-rs/src/lod_tree.rs) can only ever
// output splats reachable by walking child_start/child_count pointers
// starting from the mesh's single root page, a scene split into N chunks
// with N disconnected trees means only the ONE chunk that happens to be
// assigned the root page is ever discoverable, fetched further, or
// rendered -- regardless of how many other chunks are separately fetched
// and uploaded. This measured as "only one section of the scene is ever
// visible, no matter where the camera moves" on a real 32-chunk conversion.
//
// The fix: when building chunk 0's tree specifically (the only chunk whose
// tree is guaranteed to become the traversal root), append one extra
// "sibling pointer" entry per OTHER chunk in the manifest, using that
// chunk's own bounding box for sizing/positioning and pointing its
// child_start at THAT chunk's own local root (which that chunk will
// synthesize independently, the same way, once its own data streams in).
// This turns N disconnected single-chunk trees into one star-shaped tree
// chunk 0 can walk into every sibling. Existing traversal logic already
// treats an as-yet-unfetched child chunk (chunk_to_page[chunk] ==
// 0xFFFFFFFF) as "not resident yet" and reports it via the `chunks`/
// `touched` output, which SparkRenderer.ts already feeds back into fetch
// priority -- so this also gets progressive, distance-prioritized loading of
// every other chunk for free: compute_pixel_scale scores each sibling
// pointer by size/distance-from-camera, so nearer chunks get expanded (and
// therefore fetched) before farther ones, without any separate scheduling
// logic.
function synthesizeFlatLodTreeIfMissing(
  result: { extra: { lodTree?: Uint32Array } },
  count: number,
  getCenter: (index: number) => [number, number, number],
  siblingChunks?: { chunkIndex: number; center: [number, number, number]; size: number }[],
) {
  if (result.extra.lodTree || count <= 0) {
    return;
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < count; i++) {
    const [x, y, z] = getCenter(i);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
  const nominalSize = Math.max(diag / 2, 1e-4);

  const writeEntry = (
    lodTree: Uint32Array,
    idx: number,
    ex: number,
    ey: number,
    ez: number,
    size: number,
    childCount: number,
    childStart: number,
  ) => {
    const o = idx * 4;
    lodTree[o + 0] =
      (float32ToHalfBits(ex) & 0xffff) |
      ((float32ToHalfBits(ey) & 0xffff) << 16);
    lodTree[o + 1] =
      (float32ToHalfBits(ez) & 0xffff) |
      ((float32ToHalfBits(size) & 0xffff) << 16);
    lodTree[o + 2] = childCount & 0xffff;
    lodTree[o + 3] = childStart >>> 0;
  };

  if (count <= 65536) {
    const numSiblings = siblingChunks?.length ?? 0;
    const totalEntries = count + numSiblings;
    const lodTree = new Uint32Array(totalEntries * 4);
    // Root's children are this chunk's own leaves (relative addr 1..count-1,
    // patched to an absolute chunk address by SplatPager.ts's processFetched)
    // immediately followed by the sibling-pointer entries (written with their
    // final absolute chunk address directly below, since they reference a
    // DIFFERENT chunk than this one and SplatPager.ts's patch only rewrites
    // the root's own child_start, not every entry).
    writeEntry(lodTree, 0, cx, cy, cz, nominalSize, count - 1 + numSiblings, 1);
    for (let i = 1; i < count; i++) {
      const [x, y, z] = getCenter(i);
      writeEntry(lodTree, i, x, y, z, nominalSize, 0, 0);
    }
    if (siblingChunks) {
      for (let s = 0; s < siblingChunks.length; s++) {
        const sib = siblingChunks[s];
        const absoluteChildStart = ((sib.chunkIndex << 16) | 0) >>> 0;
        writeEntry(
          lodTree,
          count + s,
          sib.center[0],
          sib.center[1],
          sib.center[2],
          sib.size,
          1,
          absoluteChildStart,
        );
      }
      console.log(
        `[lod-tree-update] stitched ${siblingChunks.length} sibling chunk pointer(s) into chunk 0's tree ` +
          `(chunks: ${siblingChunks.map((s) => s.chunkIndex).join(', ')})`,
      );
    }
    result.extra.lodTree = lodTree;
    console.log(
      `[lod-tree-update] synthesized flat lod tree for this chunk: ${count} entries ` +
        `(1 root + ${count - 1} leaves), root center=(${cx.toFixed(3)}, ${cy.toFixed(3)}, ${cz.toFixed(3)})`,
    );
  } else {
    // 2-level tree for count > 65536 to avoid u16 child_count overflow
    const pageSize = 65535;
    const numIntermediates = Math.ceil(count / pageSize);
    const totalNodes = 1 + numIntermediates + count;
    const lodTree = new Uint32Array(totalNodes * 4);

    // Root (index 0) points to intermediate nodes
    writeEntry(lodTree, 0, cx, cy, cz, nominalSize, numIntermediates, 1);

    // Intermediate nodes
    for (let j = 0; j < numIntermediates; j++) {
      const startLeafIdx = j * pageSize;
      const endLeafIdx = Math.min(startLeafIdx + pageSize, count);
      const leafCount = endLeafIdx - startLeafIdx;

      // Compute intermediate bounding box center
      let iminX = Number.POSITIVE_INFINITY;
      let iminY = Number.POSITIVE_INFINITY;
      let iminZ = Number.POSITIVE_INFINITY;
      let imaxX = Number.NEGATIVE_INFINITY;
      let imaxY = Number.NEGATIVE_INFINITY;
      let imaxZ = Number.NEGATIVE_INFINITY;
      for (let i = startLeafIdx; i < endLeafIdx; i++) {
        const [x, y, z] = getCenter(i);
        if (x < iminX) iminX = x;
        if (x > imaxX) imaxX = x;
        if (y < iminY) iminY = y;
        if (y > imaxY) imaxY = y;
        if (z < iminZ) iminZ = z;
        if (z > imaxZ) imaxZ = z;
      }
      const icx = (iminX + imaxX) / 2;
      const icy = (iminY + imaxY) / 2;
      const icz = (iminZ + imaxZ) / 2;
      const idiag = Math.hypot(imaxX - iminX, imaxY - iminY, imaxZ - iminZ);
      const isize = Math.max(idiag / 2, 1e-4);

      // Intermediate node index is 1 + j
      // Its child_start points to the leaves
      writeEntry(lodTree, 1 + j, icx, icy, icz, isize, leafCount, 1 + numIntermediates + startLeafIdx);
    }

    // Leaf nodes
    for (let i = 0; i < count; i++) {
      const [x, y, z] = getCenter(i);
      writeEntry(lodTree, 1 + numIntermediates + i, x, y, z, nominalSize, 0, 0);
    }
    result.extra.lodTree = lodTree;
    console.log(
      `[lod-tree-update] synthesized 2-level flat lod tree for monolithic mesh: ${totalNodes} entries ` +
        `(${count} leaves across ${numIntermediates} intermediate nodes), root center=(${cx.toFixed(3)}, ${cy.toFixed(3)}, ${cz.toFixed(3)})`,
    );
  }
}

function halfToFloat(binary: number) {
  const exponent = (binary & 0x7c00) >> 10;
  const fraction = binary & 0x03ff;
  if (exponent === 0) {
    return (binary & 0x8000 ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
  } else if (exponent === 0x1f) {
    return fraction
      ? Number.NaN
      : binary & 0x8000
        ? Number.NEGATIVE_INFINITY
        : Number.POSITIVE_INFINITY;
  }
  return (
    (binary & 0x8000 ? -1 : 1) *
    Math.pow(2, exponent - 15) *
    (1 + fraction / 1024)
  );
}

// Minimal float32 -> IEEE754 half-float bit pattern conversion, used only to
// build the synthetic lod tree data above (foveation/pixel-scale heuristics,
// not final render precision).
function float32ToHalfBits(val: number): number {
  const f32 = new Float32Array([val]);
  const u32 = new Uint32Array(f32.buffer)[0];
  const sign = (u32 >> 31) & 0x1;
  const exp = (u32 >> 23) & 0xff;
  let mantissa = u32 & 0x7fffff;

  if (exp === 0xff) {
    return (sign << 15) | 0x7c00 | (mantissa ? 1 : 0);
  }
  const halfExp = exp - 127 + 15;
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

async function initialize() {
  let resolveWaitForModule: (value: WebAssembly.Module) => void;
  const waitForModule = new Promise<WebAssembly.Module>((resolve) => {
    resolveWaitForModule = resolve;
  });

  // Hold any messages received while initializing
  const pending: MessageEvent[] = [];
  const bufferMessage = (event: MessageEvent) => {
    // Handle module
    if (event.data.name === "init-wasm") {
      resolveWaitForModule(event.data.module as WebAssembly.Module);
      return;
    }

    pending.push(event);
  };
  self.addEventListener("message", bufferMessage);

  await init_wasm({ module_or_path: await waitForModule });

  self.removeEventListener("message", bufferMessage);
  self.addEventListener("message", onMessage);

  // Process any buffered messages
  for (const event of pending) {
    onMessage(event);
  }
  pending.length = 0;
}

import { convertSplatToSp5Client } from "./converter";

async function convertSplatToSp5(
  {
    fileBytes,
    fileType,
    pathName,
  }: {
    fileBytes: Uint8Array;
    fileType?: string;
    pathName?: string;
  },
  {
    sendStatus,
  }: {
    sendStatus: (data: unknown) => void;
  },
) {
  sendStatus({ phase: "Parsing input splat...", percent: 5 });
  const decoder = decode_to_gsplatarray(fileType, pathName);
  const decoded = (await decodeBytesUrl({
    decoder,
    fileBytes,
    sendStatus,
  })) as any;

  sendStatus({ phase: "Extracting splat attributes...", percent: 15 });
  const attrs = decoded.extract_attributes();

  const numSplats = attrs.num_splats;
  const maxSh = attrs.max_sh;

  const xyz = new Float32Array(attrs.xyz);
  const opacity = new Float32Array(attrs.opacity);
  const rgb = new Float32Array(attrs.rgb);
  const scales = new Float32Array(attrs.scales);
  const quaternions = new Float32Array(attrs.quaternions);
  const sh1 = maxSh > 0 ? new Float32Array(attrs.sh1) : undefined;

  decoded.free();

  const zipBytes = await convertSplatToSp5Client({
    numSplats,
    xyz,
    opacity,
    rgb,
    scales,
    quaternions,
    sh1,
    maxSh,
    onProgress: (phase, percent) => {
      sendStatus({ phase, percent });
    },
  });

  return zipBytes;
}

initialize().catch(console.error);
