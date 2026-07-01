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
  console.log(`Loaded ${numSplats} splats. Starting Tiny LoD build...`);

  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
  gsplatArray.tiny_lod(base, false);

  const totalNumSplats = gsplatArray.len();
  const CHUNK_SIZE = 65536;
  const numChunks = Math.ceil(totalNumSplats / CHUNK_SIZE);
  console.log(`Slicing and compressing ${totalNumSplats} splats into ${numChunks} chunks...`);

  const chunksInfo: { chunk: number; bytes: Uint8Array; numSplats: number }[] = [];

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const start = chunk * CHUNK_SIZE;
    const count = Math.min(totalNumSplats - start, CHUNK_SIZE);

    const subset = gsplatArray.clone_subset(start, count);
    const spzBytes = subset.to_spz();
    subset.free();

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
  const tmc3WasmUrl = new URL("../examples/viewer/tmc3.wasm", import.meta.url).href;
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

async function decodeSp5Chunk({ chunkBytes }: { chunkBytes: Uint8Array }) {
  const view = new DataView(chunkBytes.buffer, chunkBytes.byteOffset);
  const magic = view.getUint32(0, true);
  if (magic !== 0x355a5053) {
    throw new Error("Invalid .sp5 chunk magic");
  }
  const jsonSize = view.getUint32(4, true);
  const jsonBytes = chunkBytes.subarray(8, 8 + jsonSize);
  const manifest = JSON.parse(new TextDecoder().decode(jsonBytes));

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
    count = countMatch ? parseInt(countMatch[1]) : 0;

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

      xyzRawFloat[i * 3 + 0] = normX * (meansMax[0] - meansMin[0]) + meansMin[0];
      xyzRawFloat[i * 3 + 1] = normY * (meansMax[1] - meansMin[1]) + meansMin[1];
      xyzRawFloat[i * 3 + 2] = normZ * (meansMax[2] - meansMin[2]) + meansMin[2];
    }
  }

  function halfToFloat(binary: number) {
    const exponent = (binary & 0x7c00) >> 10;
    const fraction = binary & 0x03ff;
    if (exponent === 0) {
      return (
        (binary & 0x8000 ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024)
      );
    } else if (exponent === 0x1f) {
      return fraction ? NaN : binary & 0x8000 ? -Infinity : Infinity;
    }
    return (
      (binary & 0x8000 ? -1 : 1) *
      Math.pow(2, exponent - 15) *
      (1 + fraction / 1024)
    );
  }

  function decodeHuffman(
    bytes: Uint8Array,
    htable: Record<string, [number, number]>,
    count: number,
  ) {
    const table = new Map<string, number>();
    for (const [symbol, [len, bits]] of Object.entries(htable)) {
      table.set(`${bits},${len}`, parseInt(symbol));
    }
    const out = new Uint16Array(count);
    let outIdx = 0;
    let currentBits = 0;
    let currentLen = 0;
    let byteIdx = 0;
    let bitIdx = 7;
    while (outIdx < count && byteIdx < bytes.length) {
      const bit = (bytes[byteIdx] >> bitIdx) & 1;
      bitIdx--;
      if (bitIdx < 0) {
        bitIdx = 7;
        byteIdx++;
      }
      currentBits = (currentBits << 1) | bit;
      currentLen++;
      const key = `${currentBits},${currentLen}`;
      if (table.has(key)) {
        out[outIdx++] = table.get(key)!;
        currentBits = 0;
        currentLen = 0;
      }
    }
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
    scaleCbFlat.set(
      float16ArrayToFloat32Array(getBinaryPart(meta)),
      idx * 256,
    );
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
    appCbFlat.set(
      float16ArrayToFloat32Array(getBinaryPart(meta)),
      idx * 512,
    );
  });

  const mlpCont = float16ArrayToFloat32Array(getBinaryPart(manifest.mlp_cont));
  const mlpDc = float16ArrayToFloat32Array(getBinaryPart(manifest.mlp_dc));
  const mlpSh = float16ArrayToFloat32Array(getBinaryPart(manifest.mlp_sh));
  const mlpOpacity = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_opacity),
  );

  const mlpOffsetW0 = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_offset["main.0.weight"]),
  );
  const mlpOffsetB0 = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_offset["main.0.bias"]),
  );
  const mlpOffsetW1 = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_offset["main.2.weight"]),
  );
  const mlpOffsetB1 = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_offset["main.2.bias"]),
  );
  const mlpOffsetW2 = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_offset["main.4.weight"]),
  );
  const mlpOffsetB2 = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_offset["main.4.bias"]),
  );
  const mlpOffsetW3 = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_offset["shs_output.0.weight"]),
  );
  const mlpOffsetB3 = float16ArrayToFloat32Array(
    getBinaryPart(manifest.mlp_offset["shs_output.0.bias"]),
  );

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

  const result = toExtResult(gsplatArray.to_extsplats() as any);
  gsplatArray.free();
  return result;
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
