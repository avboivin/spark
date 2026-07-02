import { unzipSync } from "fflate";
import * as THREE from "three";

import { decode_rad_header } from "spark-rs";
import { LN_SCALE_MAX, LN_SCALE_MIN, dyno } from ".";
import { evaluateExtSH } from "./ExtSplats";
import { evaluatePackedSH } from "./PackedSplats";
import { getSplatFileType, getSplatFileTypeFromPath } from "./SplatLoader";
import type { SplatSource } from "./SplatMesh";
import { workerPool } from "./SplatWorker";
import {
  DEFAULT_SPLAT_ENCODING,
  type ExtResult,
  type PackedResult,
  type RadMeta,
  type SplatEncoding,
  SplatFileType,
} from "./defines";
import { type DynoUsampler2DArray, pagedSplatTexCoord } from "./dyno";
import {
  SplatCache,
  decodeExtSplat,
  getTextureSize,
  unpackSplat,
  uploadU32DataTextureRows,
} from "./utils";
import * as wasm from "./wasm";

export interface PagedSplatsOptions {
  pager?: SplatPager;
  rootUrl?: string;
  requestHeader?: Record<string, string>;
  withCredentials?: boolean;
  fileBytes?: Uint8Array;
  fileBlob?: Blob;
  fileType?: SplatFileType;
  maxSh?: number;
}

export interface ChunkSource {
  read(offset?: number, bytes?: number, filename?: string): Promise<Uint8Array>;
}

export class HttpChunkSource implements ChunkSource {
  constructor(
    private url: string,
    private requestHeader?: Record<string, string>,
    private withCredentials?: boolean,
  ) {}
  async read(
    offset?: number,
    bytes?: number,
    filename?: string,
  ): Promise<Uint8Array> {
    return fetchRange({
      url: this.url,
      requestHeader: this.requestHeader,
      withCredentials: this.withCredentials,
      offset,
      bytes,
    });
  }
}

export class BlobChunkSource implements ChunkSource {
  constructor(private blob: Blob) {}
  async read(
    offset?: number,
    bytes?: number,
    filename?: string,
  ): Promise<Uint8Array> {
    const sliced =
      offset !== undefined && bytes !== undefined
        ? this.blob.slice(offset, offset + bytes)
        : this.blob;
    return new Uint8Array(await sliced.arrayBuffer());
  }
}

export class ZipChunkSource implements ChunkSource {
  private files: Record<string, Uint8Array> = {};
  constructor(zipBytes: Uint8Array) {
    this.files = unzipSync(zipBytes);
  }
  async read(
    offset?: number,
    bytes?: number,
    filename?: string,
  ): Promise<Uint8Array> {
    if (!filename) {
      throw new Error("ZipChunkSource requires a filename");
    }
    const cleanName = filename.split(/[\\/]/).pop() || filename;
    const file = this.files[cleanName];
    if (!file) {
      throw new Error(`File ${cleanName} not found in zip`);
    }
    return file.slice();
  }
}

const PAGE_WIDTH = 256;
const PAGE_HEIGHT = 256;
const PAGE_SPLATS = PAGE_WIDTH * PAGE_HEIGHT; // 65536

export class PagedSplats implements SplatSource {
  pager?: SplatPager;
  rootUrl: string;
  requestHeader?: Record<string, string>;
  withCredentials?: boolean;
  fileBytes?: Uint8Array;
  fileBlob?: Blob;
  chunkSource?: ChunkSource;
  fileType?: SplatFileType;

  numSh: number;
  maxSh: number;
  sh1Codes?: Uint32Array;
  sh2Codes?: Uint32Array;
  sh3Codes?: Uint32Array | [Uint32Array, Uint32Array];

  numSplats: number;
  splatEncoding?: SplatEncoding;
  radMetaPromise?: Promise<{ meta: RadMeta; chunksStart: number }>;
  // Synchronous cache of the resolved manifest, populated once getRadMeta()'s
  // promise settles. Used by SparkRenderer to decide fetch priority without
  // re-awaiting an (already-resolved) promise on every frame.
  cachedMeta?: RadMeta;

  dynoNumSplats: dyno.DynoInt<"numSplats">;
  dynoIndices: dyno.DynoUsampler2D<"indices", THREE.DataTexture>;
  rgbMinMaxLnScaleMinMax: dyno.DynoVec4<
    THREE.Vector4,
    "rgbMinMaxLnScaleMinMax"
  >;
  lodOpacity: dyno.DynoBool<"lodOpacity">;
  dynoNumSh: dyno.DynoInt<"numSh">;
  shMax: dyno.DynoVec3<THREE.Vector3, "shMax">;

  constructor(options: PagedSplatsOptions) {
    this.pager = options.pager;
    this.rootUrl = options.rootUrl ?? "";
    this.requestHeader = options.requestHeader;
    this.withCredentials = options.withCredentials;
    this.numSh = 0;
    this.maxSh = options.pager?.maxSh ?? 3;

    this.numSplats = 0;

    this.dynoNumSplats = new dyno.DynoInt({ value: 0 });
    this.dynoIndices = new dyno.DynoUsampler2D({
      value: SplatPager.emptyIndicesTexture,
    });

    this.rgbMinMaxLnScaleMinMax = new dyno.DynoVec4({
      value: new THREE.Vector4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX),
    });
    this.lodOpacity = new dyno.DynoBool({
      value: false,
    });

    this.dynoNumSh = new dyno.DynoInt({ value: 0 });
    this.shMax = new dyno.DynoVec3({ value: new THREE.Vector3() });

    this.fileBytes = options.fileBytes;
    this.fileBlob = options.fileBlob;
    this.fileType = options.fileType;
    if (!this.fileType && this.fileBytes) {
      this.fileType = getSplatFileType(this.fileBytes);
    }
    if (!this.fileType && this.fileBlob) {
      const name = (this.fileBlob as any).name;
      if (name) {
        this.fileType = getSplatFileTypeFromPath(name);
        if (!this.fileType && name.toLowerCase().endsWith(".zip")) {
          this.fileType = SplatFileType.SP5;
        }
      }
    }
    if (!this.fileType && this.rootUrl) {
      this.fileType = getSplatFileTypeFromPath(this.rootUrl);
      if (!this.fileType && this.rootUrl.toLowerCase().endsWith(".zip")) {
        this.fileType = SplatFileType.SP5;
      }
    }
    if (!this.fileType) {
      throw new Error("Unable to determine file type");
    }

    if (this.fileBlob) {
      if (this.fileType === SplatFileType.SP5) {
        this.chunkSource = null as any; // initialized asynchronously inside getRadMeta
      } else {
        this.chunkSource = new BlobChunkSource(this.fileBlob);
      }
    } else if (this.fileBytes) {
      if (this.fileType === SplatFileType.SP5) {
        this.chunkSource = new ZipChunkSource(this.fileBytes);
      } else {
        this.chunkSource = new BlobChunkSource(new Blob([this.fileBytes]));
      }
    } else if (this.rootUrl) {
      this.chunkSource = new HttpChunkSource(
        this.rootUrl,
        this.requestHeader,
        this.withCredentials,
      );
    }

    if (this.fileType === SplatFileType.RAD) {
      this.radMetaPromise = this.getRadMeta();
    }
    if (this.pager) {
      this.pager.activeSplats.add(this);
    }
  }

  dispose() {
    if (this.pager) {
      this.pager.activeSplats.delete(this);
    }
    if (this.dynoIndices.value !== SplatPager.emptyIndicesTexture) {
      this.dynoIndices.value.dispose();
      this.dynoIndices.value = SplatPager.emptyIndicesTexture;
    }
  }

  setMaxSh(maxSh: number) {
    this.maxSh = maxSh;
  }

  getRadMeta(): Promise<{ meta: RadMeta; chunksStart: number }> {
    if (this.radMetaPromise) {
      return this.radMetaPromise;
    }

    this.radMetaPromise = (async () => {
      await wasm.initialization;

      if (this.fileType === SplatFileType.SP5) {
        if (this.fileBlob && !this.chunkSource) {
          const buffer = await this.fileBlob.arrayBuffer();
          this.chunkSource = new ZipChunkSource(new Uint8Array(buffer));
        }
        if (this.chunkSource instanceof ZipChunkSource) {
          const manifestBytes = await this.chunkSource.read(
            0,
            0,
            "manifest.json",
          );
          const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
          return { meta: manifest, chunksStart: 0 };
        } else if (this.rootUrl) {
          const resolvedRoot = new URL(
            this.rootUrl,
            window.location.href,
          ).toString();
          const manifestUrl = resolvedRoot.endsWith(".json")
            ? resolvedRoot
            : new URL("manifest.json", resolvedRoot).toString();
          const response = await fetch(manifestUrl);
          const manifest = await response.json();
          return { meta: manifest, chunksStart: 0 };
        }
      }

      if (!this.chunkSource) {
        throw new Error("No chunkSource available");
      }

      if (this.fileType !== SplatFileType.RAD) {
        const spotId =
          this.rootUrl ||
          (this.fileBlob as any)?.name ||
          (this.fileBytes as any)?.name ||
          "local-drop";
        const cacheKey = `local-drop-v6-${spotId}-${(this.fileBlob as any)?.size || this.fileBytes?.length || 0}`;

        const cachedManifest = await SplatCache.getManifest(cacheKey);
        if (cachedManifest) {
          this.rootUrl = cacheKey;
          return { meta: cachedManifest, chunksStart: 0 };
        }

        const fileBytes = await this.chunkSource.read();
        const result = (await workerPool.withWorker(async (worker) => {
          return await worker.call("partitionDroppedMonolithic", {
            fileBytes,
            fileType: this.fileType,
            pathName: spotId,
            lodBase: this.pager?.maxSh ?? 1.5,
          });
        })) as {
          manifest: RadMeta;
          chunks: { chunk: number; bytes: Uint8Array; numSplats: number }[];
        };

        await SplatCache.putManifest(cacheKey, result.manifest);
        for (const chunk of result.chunks) {
          await SplatCache.putChunk(
            cacheKey,
            chunk.chunk,
            0,
            chunk.bytes,
            chunk.numSplats,
            this.pager?.maxCacheSplats ?? 0,
          );
        }

        this.rootUrl = cacheKey;
        return { meta: result.manifest, chunksStart: 0 };
      }

      for (const tryBytes of [65536, 256 * 1024, 1024 * 1024]) {
        const bytes = await this.chunkSource.read(0, tryBytes);
        const metaStart = decode_rad_header(bytes);
        if (metaStart) {
          return metaStart;
        }
      }
      throw new Error("Failed to decode RAD header");
    })().then((metaStart) => {
      this.cachedMeta = metaStart.meta;
      return metaStart;
    });

    this.radMetaPromise.catch((error) => {
      console.error(error);
    });

    return this.radMetaPromise;
  }

  chunkUrl(chunk: number): string {
    return this.rootUrl.replace(/-lod-0\./, `-lod-${chunk}.`);
  }

  async fetchDecodeChunk(chunk: number) {
    await this.getRadMeta();
    const spotId =
      this.rootUrl ||
      (this.fileBlob as any)?.name ||
      (this.fileBytes as any)?.name ||
      "local-drop";
    let decodeBytes = await SplatCache.getChunk(spotId, chunk);
    const cacheHit = decodeBytes !== null;
    console.log(
      `fetchDecodeChunk: chunk=${chunk}, spotId=${spotId}, cacheHit=${cacheHit}` +
        (cacheHit ? `, cachedBytes=${decodeBytes!.length}` : ""),
    );

    const fetchStart = performance.now();

    if (!cacheHit) {
      if (this.fileType === SplatFileType.RAD) {
        const { meta, chunksStart } = await this.getRadMeta();
        if (chunk < 0 || chunk >= meta.chunks.length) {
          throw new Error(
            `Chunk index out of range: ${chunk} (max: ${meta.chunks.length - 1})`,
          );
        }
        let { offset, bytes, filename } = meta.chunks[chunk];

        if (filename) {
          if (this.fileBytes) {
            throw new Error("Chunked RAD file not supported with fileBytes");
          }
          const resolvedRoot = new URL(
            this.rootUrl,
            window.location.href,
          ).toString();
          const chunkUrl = new URL(filename, resolvedRoot).toString();
          decodeBytes = await fetchRange({
            url: chunkUrl,
            requestHeader: this.requestHeader,
            withCredentials: this.withCredentials,
          });
        } else {
          offset += chunksStart;
          if (this.chunkSource) {
            decodeBytes = await this.chunkSource.read(offset, bytes);
          } else {
            throw new Error("No chunkSource available");
          }
        }
      } else if (this.fileBytes) {
        // Fall through
      } else if (this.rootUrl && !this.rootUrl.startsWith("local-drop")) {
        let url = this.chunkUrl(chunk);
        if (this.fileType === SplatFileType.SP5) {
          const { meta } = await this.getRadMeta();
          const filename =
            (meta.chunks[chunk] as any).file || `scene-lod-${chunk}.sp5`;
          const resolvedRoot = new URL(
            this.rootUrl,
            window.location.href,
          ).toString();
          const lastSlash = resolvedRoot.lastIndexOf("/");
          url = resolvedRoot.slice(0, lastSlash + 1) + filename;
        }
        const request = new Request(url, {
          headers: this.requestHeader
            ? new Headers(this.requestHeader)
            : undefined,
          credentials: this.withCredentials ? "include" : "same-origin",
        });
        const response = await fetch(request);
        if (!response.ok || !response.body) {
          throw new Error(
            `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
          );
        }
        decodeBytes = new Uint8Array(await response.arrayBuffer());
      } else if (this.fileType === SplatFileType.SP5) {
        const { meta } = await this.getRadMeta();
        const filename =
          (meta.chunks[chunk] as any).file || `scene-lod-${chunk}.sp5`;
        if (this.chunkSource) {
          decodeBytes = await this.chunkSource.read(0, 0, filename);
        } else {
          throw new Error("No chunkSource available");
        }
      } else if (this.chunkSource) {
        decodeBytes = await this.chunkSource.read();
      } else {
        throw new Error("No url or fileBytes provided");
      }
    }

    const workerStart = performance.now();
    const result = await workerPool.withWorker(async (worker) => {
      if (!this.pager) {
        throw new Error("PagedSplats.pager not set");
      }
      let lodSplats: PackedResult | ExtResult = null as any;
      if (this.fileType === SplatFileType.SP5) {
        // LEGACY-FALLBACK-ONLY: siblingChunks is only consumed by
        // synthesizeFlatLodTreeIfMissing (worker.ts) for flat/synthetic LOD
        // trees where chunk 0's one-level tree needs explicit pointers to
        // every other chunk (no real hierarchy to discover them organically).
        // Real hierarchical trees (manifest.lodTree === true) bypass this
        // entirely -- cross-chunk connections come from genuine parent-child
        // edges written during conversion (converter.ts's lodTreeSlice).
        // Do NOT re-purpose siblingChunks for real-tree stitching; the former
        // stitchRealLodTreeWithSiblings function has been removed as it
        // silently orphaned ~99% of content (see plan3.md §2 FIXED).
        let siblingChunks:
          | { chunkIndex: number; center: [number, number, number]; size: number }[]
          | undefined;
        if (chunk === 0) {
          const { meta } = await this.getRadMeta();
          siblingChunks = (meta.chunks as any[])
            .map((c, idx) => ({ c, idx }))
            .filter(({ c, idx }) => idx !== 0 && Array.isArray(c.aabb) && c.aabb.length === 6)
            .map(({ c, idx }) => {
              const [minX, minY, minZ, maxX, maxY, maxZ] = c.aabb;
              const center: [number, number, number] = [
                (minX + maxX) / 2,
                (minY + maxY) / 2,
                (minZ + maxZ) / 2,
              ];
              const size = Math.max(
                Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2,
                1e-4,
              );
              return { chunkIndex: idx, center, size };
            });
        }
        const result = (await worker.call("decodeSp5Chunk", {
          chunkBytes: decodeBytes!.slice(),
          siblingChunks,
        })) as PackedResult;
        lodSplats = result;
      } else if (!this.pager.extSplats) {
        const result = (await worker.call("loadPackedSplats", {
          fileBytes: decodeBytes!.slice(),
          pathName: this.chunkUrl(chunk),
          sh1Codes: this.sh1Codes?.slice(),
          sh2Codes: this.sh2Codes?.slice(),
          sh3Codes: this.sh3Codes?.slice(),
        })) as any;
        lodSplats = result.lodSplats ? result.lodSplats : result;
      }

      if (this.fileType === SplatFileType.SP5 || !this.pager.extSplats) {
        const packed = lodSplats as PackedResult;
        if (!this.splatEncoding) {
          this.splatEncoding = packed.splatEncoding || DEFAULT_SPLAT_ENCODING;

          this.numSh = packed.extra.sh3
            ? 3
            : packed.extra.sh2
              ? 2
              : packed.extra.sh1
                ? 1
                : 0;

          this.rgbMinMaxLnScaleMinMax.value.set(
            this.splatEncoding.rgbMin ?? 0.0,
            this.splatEncoding.rgbMax ?? 1.0,
            this.splatEncoding.lnScaleMin ?? LN_SCALE_MIN,
            this.splatEncoding.lnScaleMax ?? LN_SCALE_MAX,
          );

          this.lodOpacity.value = this.splatEncoding.lodOpacity ?? false;

          this.shMax.value.set(
            this.splatEncoding.sh1Max ?? 1.0,
            this.splatEncoding.sh2Max ?? 1.0,
            this.splatEncoding.sh3Max ?? 1.0,
          );
          console.log(
            `[pager-encoding] splatEncoding configured: numSh=${this.numSh}, ` +
              `rgbMinMax=(${this.splatEncoding.rgbMin},${this.splatEncoding.rgbMax}), ` +
              `shMax=(${this.splatEncoding.sh1Max},${this.splatEncoding.sh2Max},${this.splatEncoding.sh3Max}), ` +
              `lodOpacity=${this.splatEncoding.lodOpacity}`,
          );
        }
        this.sh1Codes = packed.extra.sh1Codes ?? this.sh1Codes;
        this.sh2Codes = packed.extra.sh2Codes ?? this.sh2Codes;
        this.sh3Codes = packed.extra.sh3Codes ?? this.sh3Codes;
      } else {
        const sh3Codes = this.sh3Codes as
          | [Uint32Array, Uint32Array]
          | undefined;
        const result = (await worker.call("loadExtSplats", {
          fileBytes: decodeBytes!.slice(),
          pathName: this.chunkUrl(chunk),
          sh1Codes: this.sh1Codes?.slice(),
          sh2Codes: this.sh2Codes?.slice(),
          sh3Codes: sh3Codes
            ? [sh3Codes[0].slice(), sh3Codes[1].slice()]
            : undefined,
        })) as any;
        lodSplats = result.lodSplats ? result.lodSplats : result;
        const ext = lodSplats as ExtResult;
        if (!this.splatEncoding) {
          this.splatEncoding = DEFAULT_SPLAT_ENCODING;
          this.numSh =
            ext.extra.sh3a && ext.extra.sh3b
              ? 3
              : ext.extra.sh2
                ? 2
                : ext.extra.sh1
                  ? 1
                  : 0;
        }
        this.sh1Codes = ext.extra.sh1Codes ?? this.sh1Codes;
        this.sh2Codes = ext.extra.sh2Codes ?? this.sh2Codes;
        this.sh3Codes = ext.extra.sh3Codes ?? this.sh3Codes;
      }

      if (!cacheHit && decodeBytes) {
        await SplatCache.putChunk(
          spotId,
          chunk,
          0,
          decodeBytes,
          lodSplats.numSplats,
          this.pager.maxCacheSplats,
        );
      }

      const workerElapsed = (performance.now() - workerStart).toFixed(1);
      console.log(
        `[pager-perf] fetchDecodeChunk chunk=${chunk}: workerRPC+decode = ${workerElapsed}ms, ` +
        `numSplats=${lodSplats.numSplats}, cacheHit=${cacheHit}`,
      );

      return lodSplats;
    });
    return result;
  }

  update(numSplats: number, indices: Uint32Array) {
    if (!this.pager) {
      throw new Error("PagedSplats.pager not set");
    }

    const renderer = this.pager.renderer;
    this.numSplats = numSplats;
    this.dynoNumSplats.value = this.numSplats;
    const rows = Math.ceil(numSplats / 16384);

    let indicesTexture =
      this.dynoIndices.value === SplatPager.emptyIndicesTexture
        ? undefined
        : this.dynoIndices.value;
    if (indicesTexture && rows > indicesTexture.image.height) {
      indicesTexture.dispose();
      indicesTexture = undefined;
    }

    if (!indicesTexture) {
      indicesTexture = new THREE.DataTexture(
        indices,
        4096,
        rows,
        THREE.RGBAIntegerFormat,
        THREE.UnsignedIntType,
      );
      indicesTexture.internalFormat = "RGBA32UI";
      indicesTexture.needsUpdate = true;
      renderer.initTexture(indicesTexture);
      this.dynoIndices.value = indicesTexture;
    } else {
      const textureIndices = indicesTexture.image.data as Uint32Array;
      textureIndices.set(indices.subarray(0, numSplats));

      uploadU32DataTextureRows(
        renderer,
        indicesTexture,
        4096,
        rows,
        textureIndices,
      );
    }
  }

  prepareFetchSplat() {}

  getNumSplats(): number {
    return this.numSplats;
  }

  hasRgbDir(): boolean {
    if (!this.pager) {
      return false;
    }
    return Math.min(this.numSh, this.pager.maxSh) > 0;
  }

  getNumSh(): number {
    return this.numSh;
  }

  fetchSplat({
    index,
    viewOrigin,
  }: {
    index: dyno.DynoVal<"int">;
    viewOrigin?: dyno.DynoVal<"vec3">;
  }): dyno.DynoVal<typeof dyno.Gsplat> {
    if (!this.pager) {
      throw new Error("PagedSplats.pager not set");
    }

    const splatIndex = this.pager.readIndex.apply({
      index,
      numSplats: this.dynoNumSplats,
      indices: this.dynoIndices,
    }).index;

    if (!this.pager.extSplats) {
      if (this.hasRgbDir() && viewOrigin) {
        this.dynoNumSh.value = Math.min(
          this.numSh,
          this.maxSh,
          this.pager.maxSh,
        );
        return this.pager.readSplatDir.apply({
          index: splatIndex,
          rgbMinMaxLnScaleMinMax: this.rgbMinMaxLnScaleMinMax,
          lodOpacity: this.lodOpacity,
          viewOrigin,
          numSh: this.dynoNumSh,
          shMax: this.shMax,
        }).gsplat;
      }
      return this.pager.readSplat.apply({
        index: splatIndex,
        rgbMinMaxLnScaleMinMax: this.rgbMinMaxLnScaleMinMax,
        lodOpacity: this.lodOpacity,
      }).gsplat;
    }

    if (this.hasRgbDir() && viewOrigin) {
      this.dynoNumSh.value = Math.min(this.numSh, this.maxSh, this.pager.maxSh);
      return this.pager.readSplatExtDir.apply({
        index: splatIndex,
        viewOrigin,
        numSh: this.dynoNumSh,
      }).gsplat;
    }
    return this.pager.readSplatExt.apply({ index: splatIndex }).gsplat;
  }

  // Iterate over Gsplats index 0..=(this.numSplats-1), unpack each Gsplat
  // and invoke the callback function with the Gsplat attributes.
  forEachSplat(
    callback: (
      index: number,
      center: THREE.Vector3,
      scales: THREE.Vector3,
      quaternion: THREE.Quaternion,
      opacity: number,
      color: THREE.Color,
    ) => void,
  ) {
    if (!this.pager || !this.numSplats) {
      return;
    }
    const extSplats = this.pager.extSplats;
    const indices = this.dynoIndices.value.image.data as Uint32Array;
    const packedSplatArray = this.pager.packedTexture.value.image
      .data as Uint32Array;
    const extPackedSplatArray = this.pager.extTexture.value.image
      .data as Uint32Array;
    const extArrays: [Uint32Array, Uint32Array] = [
      packedSplatArray,
      extPackedSplatArray,
    ];

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < this.numSplats; ++i) {
      const splatIndex = indices[i];
      const unpacked = extSplats
        ? decodeExtSplat(extArrays, splatIndex)
        : unpackSplat(packedSplatArray, splatIndex, this.splatEncoding);
      callback(
        i,
        unpacked.center,
        unpacked.scales,
        unpacked.quaternion,
        unpacked.opacity,
        unpacked.color,
      );
      const cx = unpacked.center.x;
      const cy = unpacked.center.y;
      const cz = unpacked.center.z;
      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
      if (cz < minZ) minZ = cz;
      if (cz > maxZ) maxZ = cz;
    }
    const hasInfNaN =
      !isFinite(minX) || !isFinite(maxX) ||
      !isFinite(minY) || !isFinite(maxY) ||
      !isFinite(minZ) || !isFinite(maxZ);
    console.log(
      `[forEachSplat] iterated ${this.numSplats} splats, position range: ` +
        `x=[${minX.toFixed(3)}, ${maxX.toFixed(3)}] ` +
        `y=[${minY.toFixed(3)}, ${maxY.toFixed(3)}] ` +
        `z=[${minZ.toFixed(3)}, ${maxZ.toFixed(3)}]` +
        (hasInfNaN
          ? " -- CRITICAL: positions contain Infinity or NaN, vertices will not render correctly"
          : ""),
    );
  }
}

export interface SplatPagerOptions {
  /**
   * THREE.WebGLRenderer instance to upload texture data
   */
  renderer: THREE.WebGLRenderer;
  /**
   * Whether to use extended Gsplat encoding for paged splats.
   * @default false
   */
  extSplats?: boolean;
  /**
   * Maximum size of splat page pool
   * @default 65536 * 256 = 16777216
   */
  maxSplats?: number;
  /**
   * Maximum number of spherical harmonics to keep
   * @default 3
   */
  maxSh?: number;
  /**
   * Automatically drive page fetching, or poll via drive()
   * @default true
   */
  autoDrive?: boolean;
  /**
   * Number of concurrent fetchers
   * @default 3
   */
  numFetchers?: number;
  /**
   * Maximum size of IndexedDB cache in splats
   */
  maxCacheSplats?: number;
  /**
   * Proactively fetch and cache chunks in the background
   * @default false
   */
  backgroundPrefetch?: boolean;
}

interface PageUpload {
  page: number;
  numSplats: number;
  packedArray: Uint32Array;
  extArray?: Uint32Array;
  shArrays: Array<Uint32Array>;
}

export class SplatPager {
  readonly renderer: THREE.WebGLRenderer;

  readonly extSplats: boolean;
  readonly maxPages: number;
  readonly maxSplats: number;
  readonly pageSplats: number;
  readonly maxCacheSplats: number;
  backgroundPrefetch: boolean;
  activeSplats: Set<PagedSplats> = new Set();

  readonly maxSh: number;
  curSh: number;

  autoDrive: boolean;
  numFetchers: number;
  fetchPause = 0;

  splatsChunkToPage: Map<
    PagedSplats,
    ({ page: number; lru: number } | undefined)[]
  > = new Map();
  pageToSplatsChunk: (
    | { splats: PagedSplats; chunk: number; time: number }
    | undefined
  )[] = [];
  pageFreelist: number[];
  pageLru: Set<{ page: number; lru: number }>;
  freeablePages: number[];
  newUploads: PageUpload[];
  readyUploads: PageUpload[];
  lodTreeUpdates: {
    splats: PagedSplats;
    page: number;
    chunk: number;
    numSplats: number;
    lodTree?: Uint32Array;
  }[];

  fetchers: { splats: PagedSplats; chunk: number; promise: Promise<void> }[];
  fetched: {
    splats: PagedSplats;
    chunk: number;
    data: PackedResult | ExtResult;
  }[];
  fetchPriority: { splats: PagedSplats; chunk: number }[];

  packedTexture: dyno.DynoUsampler2DArray<
    "packedTexture",
    THREE.DataArrayTexture
  >;
  extTexture: dyno.DynoUsampler2DArray<"extTexture", THREE.DataArrayTexture>;
  readonly shTextures: [
    dyno.DynoUsampler2DArray<"sh1", THREE.DataArrayTexture>,
    dyno.DynoUsampler2DArray<"sh2", THREE.DataArrayTexture>,
    dyno.DynoUsampler2DArray<"sh3", THREE.DataArrayTexture>,
    dyno.DynoUsampler2DArray<"sh3b", THREE.DataArrayTexture>,
  ];

  readIndex: dyno.DynoBlock<
    { index: "int"; numSplats: "int"; indices: "usampler2D" },
    { index: "int" }
  >;
  readSplat: dyno.DynoBlock<
    { index: "int"; rgbMinMaxLnScaleMinMax: "vec4"; lodOpacity: "bool" },
    { gsplat: typeof dyno.Gsplat }
  >;
  readSplatExt: dyno.DynoBlock<
    { index: "int" },
    { gsplat: typeof dyno.Gsplat }
  >;
  readSplatDir: dyno.DynoBlock<
    {
      index: "int";
      rgbMinMaxLnScaleMinMax: "vec4";
      lodOpacity: "bool";
      viewOrigin: "vec3";
      numSh: "int";
      shMax: "vec3";
    },
    { gsplat: typeof dyno.Gsplat }
  >;
  readSplatExtDir: dyno.DynoBlock<
    { index: "int"; viewOrigin: "vec3"; numSh: "int" },
    { gsplat: typeof dyno.Gsplat }
  >;

  constructor(options: SplatPagerOptions) {
    this.renderer = options.renderer;
    this.extSplats = options.extSplats ?? false;

    this.pageSplats = PAGE_SPLATS;
    this.maxSplats = options.maxSplats ?? 16777216;
    this.maxPages = Math.ceil(this.maxSplats / PAGE_SPLATS);
    this.maxSplats = this.maxPages * PAGE_SPLATS;

    this.maxCacheSplats = options.maxCacheSplats ?? this.maxSplats;
    this.backgroundPrefetch = options.backgroundPrefetch ?? false;

    this.maxSh = options.maxSh ?? 3;
    this.curSh = 0;

    this.autoDrive = options.autoDrive ?? true;
    this.numFetchers = options.numFetchers ?? 3;

    this.splatsChunkToPage = new Map();
    this.pageToSplatsChunk = new Array(this.maxPages);
    this.pageFreelist = Array.from({ length: this.maxPages }, (_, i) => i);
    this.pageLru = new Set();
    this.freeablePages = [];
    this.newUploads = [];
    this.readyUploads = [];
    this.lodTreeUpdates = [];

    this.fetchers = [];
    this.fetched = [];
    this.fetchPriority = [];

    this.packedTexture = new dyno.DynoUsampler2DArray({
      value: this.newUint32ArrayTexture(4),
    });
    this.extTexture = new dyno.DynoUsampler2DArray({
      value: this.extSplats
        ? this.newUint32ArrayTexture(4)
        : SplatPager.emptyExtTexture,
    });

    const emptyShTextures = this.extSplats
      ? SplatPager.emptyExtShTextures
      : SplatPager.emptyShTextures;
    this.shTextures = emptyShTextures.map(
      (texture) =>
        new dyno.DynoUsampler2DArray({
          value: texture,
        }),
    ) as typeof this.shTextures;

    this.readIndex = dyno.dynoBlock(
      { index: "int", numSplats: "int", indices: "usampler2D" },
      { index: "int" },
      ({ index, numSplats, indices }) => {
        return new dyno.Dyno({
          inTypes: {
            index: "int",
            numSplats: "int",
            indices: "usampler2D",
          },
          outTypes: { index: "int" },
          inputs: {
            index,
            numSplats,
            indices,
          },
          statements: ({ inputs, outputs }) =>
            dyno.unindentLines(`
            if (${inputs.index} >= ${inputs.numSplats}) {
              return;
            }

            ivec2 indexCoord = ivec2((${inputs.index} >> 2) & 4095, ${inputs.index} >> 14);
            uint index = texelFetch(${inputs.indices}, indexCoord, 0)[${inputs.index} & 3];
            ${outputs.index} = int(index);
          `),
        }).outputs;
      },
    );

    this.readSplat = dyno.dynoBlock(
      { index: "int", rgbMinMaxLnScaleMinMax: "vec4", lodOpacity: "bool" },
      { gsplat: dyno.Gsplat },
      ({ index, rgbMinMaxLnScaleMinMax, lodOpacity }) => {
        return new dyno.Dyno({
          inTypes: {
            index: "int",
            packedTexture: "usampler2DArray",
            rgbMinMaxLnScaleMinMax: "vec4",
            lodOpacity: "bool",
          },
          outTypes: { gsplat: dyno.Gsplat },
          inputs: {
            index,
            packedTexture: this.packedTexture,
            rgbMinMaxLnScaleMinMax,
            lodOpacity,
          },
          globals: () => [dyno.defineGsplat],
          statements: ({ inputs, outputs }) =>
            dyno.unindentLines(`
            int index = ${inputs.index};
            ivec3 splatCoord = pagedSplatTexCoord(index);
            uvec4 packedData = texelFetch(${inputs.packedTexture}, splatCoord, 0);

            unpackSplatEncoding(packedData, ${outputs.gsplat}.center, ${outputs.gsplat}.scales, ${outputs.gsplat}.quaternion, ${outputs.gsplat}.rgba, ${inputs.rgbMinMaxLnScaleMinMax});
            if ((${outputs.gsplat}.rgba.a == 0.0) || all(equal(${outputs.gsplat}.scales, vec3(0.0, 0.0, 0.0)))) {
              return;
            }
            
            ${outputs.gsplat}.index = index;
            ${outputs.gsplat}.flags = GSPLAT_FLAG_ACTIVE;
            if (${inputs.lodOpacity}) {
              ${outputs.gsplat}.rgba.a *= 2.0;
            }
          `),
        }).outputs;
      },
    );

    this.readSplatDir = dyno.dynoBlock(
      {
        index: "int",
        rgbMinMaxLnScaleMinMax: "vec4",
        lodOpacity: "bool",
        viewOrigin: "vec3",
        numSh: "int",
        shMax: "vec3",
      },
      { gsplat: dyno.Gsplat },
      ({
        index,
        rgbMinMaxLnScaleMinMax,
        lodOpacity,
        viewOrigin,
        numSh,
        shMax,
      }) => {
        if (
          !index ||
          !rgbMinMaxLnScaleMinMax ||
          !lodOpacity ||
          !viewOrigin ||
          !numSh ||
          !shMax
        ) {
          throw new Error("index and viewOrigin are required");
        }
        let gsplat = this.readSplat.apply({
          index,
          rgbMinMaxLnScaleMinMax,
          lodOpacity,
        }).gsplat;

        const splatCenter = dyno.splitGsplat(gsplat).outputs.center;
        const viewDir = dyno.normalize(dyno.sub(splatCenter, viewOrigin));
        let rgb = evaluatePackedSH({
          coord: pagedSplatTexCoord(index),
          viewDir,
          numSh,
          sh1Texture: this.shTextures[0],
          sh2Texture: this.shTextures[1],
          sh3Texture: this.shTextures[2],
          shMax,
        }).rgb;
        rgb = dyno.add(rgb, dyno.splitGsplat(gsplat).outputs.rgb);
        gsplat = dyno.combineGsplat({ gsplat, rgb });
        return { gsplat };
      },
    );

    this.readSplatExt = dyno.dynoBlock(
      { index: "int" },
      { gsplat: dyno.Gsplat },
      ({ index }) => {
        return new dyno.Dyno({
          inTypes: {
            index: "int",
            extTexture1: "usampler2DArray",
            extTexture2: "usampler2DArray",
          },
          outTypes: { gsplat: dyno.Gsplat },
          inputs: {
            index,
            extTexture1: this.packedTexture,
            extTexture2: this.extTexture,
          },
          globals: () => [dyno.defineGsplat],
          statements: ({ inputs, outputs }) =>
            dyno.unindentLines(`
            int index = ${inputs.index};
            ivec3 splatCoord = ivec3(index & 255, (index >> 8) & 255, index >> 16);
            uvec4 ext1 = texelFetch(${inputs.extTexture1}, splatCoord, 0);
            float alpha = unpackSplatExtAlpha(ext1);
            if (alpha == 0.0) {
              return;
            }

            uvec4 ext2 = texelFetch(${inputs.extTexture2}, splatCoord, 0);
            unpackSplatExt(ext1, ext2, ${outputs.gsplat}.center, ${outputs.gsplat}.scales, ${outputs.gsplat}.quaternion, ${outputs.gsplat}.rgba);
            if (all(equal(${outputs.gsplat}.scales, vec3(0.0, 0.0, 0.0)))) {
              return;
            }

            ${outputs.gsplat}.index = index;
            ${outputs.gsplat}.flags = GSPLAT_FLAG_ACTIVE;
          `),
        }).outputs;
      },
    );

    this.readSplatExtDir = dyno.dynoBlock(
      {
        index: "int",
        viewOrigin: "vec3",
        numSh: "int",
      },
      { gsplat: dyno.Gsplat },
      ({ index, viewOrigin, numSh }) => {
        if (!index || !viewOrigin || !numSh) {
          throw new Error("index and viewOrigin are required");
        }
        let gsplat = this.readSplatExt.apply({ index }).gsplat;

        const splatCenter = dyno.splitGsplat(gsplat).outputs.center;
        const viewDir = dyno.normalize(dyno.sub(splatCenter, viewOrigin));
        let rgb = evaluateExtSH({
          coord: pagedSplatTexCoord(index),
          viewDir,
          numSh,
          sh1Texture: this.shTextures[0],
          sh2Texture: this.shTextures[1],
          sh3TextureA: this.shTextures[2],
          sh3TextureB: this.shTextures[3],
        }).rgb;
        rgb = dyno.add(rgb, dyno.splitGsplat(gsplat).outputs.rgb);
        gsplat = dyno.combineGsplat({ gsplat, rgb });
        return { gsplat };
      },
    );
  }

  dispose() {
    this.autoDrive = false;
    this.numFetchers = 0;

    this.packedTexture.value.dispose();
    this.packedTexture.value.source.data = null;
    if (this.extTexture.value !== SplatPager.emptyExtTexture) {
      this.extTexture.value.dispose();
      this.extTexture.value.source.data = null;
    }

    const emptyShTextures = this.extSplats
      ? SplatPager.emptyExtShTextures
      : SplatPager.emptyShTextures;
    for (let i = 0; i < emptyShTextures.length; i++) {
      const texture = this.shTextures[i].value;
      if (texture !== emptyShTextures[i]) {
        texture.dispose();
        texture.source.data = null;
      }
    }
  }

  private ensureShTextures(numSh: number) {
    this.curSh = Math.max(this.curSh, numSh);

    const emptyShTextures = this.extSplats
      ? SplatPager.emptyExtShTextures
      : SplatPager.emptyShTextures;
    for (let i = 0; i < this.curSh; i++) {
      if (this.shTextures[i].value === emptyShTextures[i]) {
        const elementsPerSplat =
          this.shTextures[i].value === SplatPager.emptyUint32x2 ? 2 : 4;
        this.shTextures[i].value = this.newUint32ArrayTexture(elementsPerSplat);
      }
    }
  }

  private allocatePage(): number | undefined {
    return this.pageFreelist.shift();
  }

  getSplatsChunk(splats: PagedSplats, chunk: number) {
    const chunks = this.splatsChunkToPage.get(splats);
    if (!chunks) {
      return undefined;
    }
    return chunks[chunk];
  }

  private insertSplatsChunkPage(
    splats: PagedSplats,
    chunk: number,
    page: number,
    now: number,
  ) {
    if (!this.splatsChunkToPage.has(splats)) {
      this.splatsChunkToPage.set(splats, []);
    }
    const chunks = this.splatsChunkToPage.get(splats);
    if (!chunks) {
      throw new Error("impossible");
    }
    if (chunk >= chunks.length) {
      chunks.length = chunk + 1;
    }
    const pageLru = { page, lru: now };
    chunks[chunk] = pageLru;
    this.pageLru.add(pageLru);

    this.pageToSplatsChunk[page] = { splats, chunk, time: performance.now() };
    return this.pageToSplatsChunk[page];
  }

  private removeSplatsChunkPage(
    splats: PagedSplats,
    chunk: number,
    page: number,
  ) {
    const chunks = this.splatsChunkToPage.get(splats);
    if (!chunks) {
      throw new Error("impossible");
    }

    const pageLru = chunks[chunk];
    if (!pageLru) {
      throw new Error(
        `pageLru not found for splats: ${splats}, chunk: ${chunk}, page: ${page}`,
      );
    }
    this.pageLru.delete(pageLru);

    chunks[chunk] = undefined;

    while (chunks.length > 0 && chunks[chunks.length - 1] === undefined) {
      chunks.pop();
    }
    if (chunks.length === 0) {
      this.splatsChunkToPage.delete(splats);
    }

    this.pageToSplatsChunk[page] = undefined;
    while (
      this.pageToSplatsChunk.length > 0 &&
      this.pageToSplatsChunk[this.pageToSplatsChunk.length - 1] === undefined
    ) {
      this.pageToSplatsChunk.pop();
    }
  }

  removeSplats(splats: PagedSplats) {
    const chunks = this.splatsChunkToPage.get(splats);
    if (!chunks) {
      return;
    }

    const freedPages = new Set<number>();

    while (chunks.length > 0) {
      const chunk = chunks.pop();
      if (chunk) {
        const { page } = chunk;
        this.pageToSplatsChunk[page] = undefined;
        freedPages.add(page);
        this.pageFreelist.push(page);
        this.pageLru.delete(chunk);
      }
    }
    this.splatsChunkToPage.delete(splats);
    this.freeablePages = this.freeablePages.filter(
      (page) => !freedPages.has(page),
    );
  }

  private uploadPage(
    page: number,
    packedArray: Uint32Array,
    shArrays: Array<Uint32Array>,
    extArray?: Uint32Array,
  ) {
    const pageBase = page * PAGE_SPLATS;
    const nonZero = packedArray.filter((v) => v !== 0).length;
    console.log(
      `uploadPage: page=${page}, pageBase=${pageBase}, packedArray.length=${packedArray.length}, numShArrays=${shArrays.length}, nonZeroWords=${nonZero}/${packedArray.length}` +
        (nonZero === 0 ? " -- WARNING: packed array is all zeros, rendering will produce nothing" : ""),
    );

    uploadTextureLayer(this.packedTexture, page, pageBase * 4, packedArray);

    if (extArray) {
      uploadTextureLayer(this.extTexture, page, pageBase * 4, extArray);
    }

    // In case of extSplats there can be 4 shArrays for 3 sh degrees
    const numSh = Math.min(shArrays.length, 3);
    this.ensureShTextures(numSh);

    for (let i = 0; i < shArrays.length; i++) {
      const array = shArrays[i];
      const elementsPerSplat =
        this.shTextures[i].value.format === THREE.RGIntegerFormat ? 2 : 4;
      uploadTextureLayer(
        this.shTextures[i],
        page,
        pageBase * elementsPerSplat,
        array,
      );
    }
  }

  private newUint32ArrayTexture(
    elementsPerSplat: 2 | 4,
  ): THREE.DataArrayTexture {
    const data = new Uint32Array(
      this.maxPages * PAGE_WIDTH * PAGE_HEIGHT * elementsPerSplat,
    );
    const texture = new THREE.DataArrayTexture(
      data,
      PAGE_WIDTH,
      PAGE_HEIGHT,
      this.maxPages,
    );
    texture.format =
      elementsPerSplat === 2 ? THREE.RGIntegerFormat : THREE.RGBAIntegerFormat;
    texture.type = THREE.UnsignedIntType;
    texture.internalFormat = elementsPerSplat === 2 ? "RG32UI" : "RGBA32UI";
    texture.needsUpdate = true;
    // Avoid initial upload of empty/null data
    texture.source.dataReady = false;
    this.renderer.initTexture(texture);
    return texture;
  }

  driveFetchers() {
    const needed = [];
    const overflow = [];
    let numPages = 0;
    let enqueued = 0;

    if (this.fetchPriority.length > 0) {
      console.log(
        `[pager-perf] driveFetchers: fetchPriority.length=${this.fetchPriority.length}, ` +
        `fetchers=${this.fetchers.length}/${this.numFetchers}, fetched=${this.fetched.length}`,
      );
    }

    for (const { splats, chunk } of this.fetchPriority) {
      const pageLru = this.getSplatsChunk(splats, chunk);
      if (pageLru) {
        if (numPages >= this.maxPages) {
          overflow.push(pageLru);
        } else {
          needed.push(pageLru);
        }
        numPages += 1;
        continue;
      }

      if (
        this.fetched.some(
          ({ splats: s, chunk: c }) => splats === s && chunk === c,
        ) ||
        this.fetchers.some(
          ({ splats: s, chunk: c }) => splats === s && chunk === c,
        )
      ) {
        numPages += 1;
        continue;
      }

      if (numPages < this.maxPages && this.fetchers.length < this.numFetchers) {
        numPages += 1;
        const promise = splats
          .fetchDecodeChunk(chunk)
          .then(
            async (data) => {
              // Place data in ready queue and remove self from active fetchers list
              this.fetched.push({ splats, chunk, data });
              if (this.fetchPause > 0) {
                await new Promise((resolve) =>
                  setTimeout(resolve, this.fetchPause),
                );
              }
            },
            async (error) => {
              console.warn(error);
              const backoff = 250 + 500 * Math.random();
              await new Promise((resolve) => setTimeout(resolve, backoff));
            },
          )
          .finally(() => {
            this.fetchers = this.fetchers.filter(
              ({ splats: s, chunk: c }) => splats !== s || chunk !== c,
            );
            this.processFetched();
          });
        // Add self to active fetchers list
        this.fetchers.push({ splats, chunk, promise });
        enqueued++;

        promise.then((data) => {
          if (this.autoDrive) {
            this.driveFetchers();
          }
        });
      }
    }

    if (this.backgroundPrefetch && this.fetchers.length < this.numFetchers) {
      this.runBackgroundPrefetch();
    }

    // Update LRU ordering in reverse priority order
    const now = performance.now();

    for (const pageLru of overflow.reverse()) {
      pageLru.lru = now;
      this.pageLru.delete(pageLru);
      this.pageLru.add(pageLru);
    }

    // Create set of pages not needed — but exclude recently-uploaded pages
    // to prevent evict-then-refetch thrashing during rotation: when the camera
    // sweeps across chunks quickly, a page that was just loaded may briefly
    // fall out of the foveation cone and get marked freeable, only to be
    // re-requested moments later when the camera finishes its sweep. A 10s
    // keep-warm window lets pages ride out short-term viewpoint changes.
    const PAGE_KEEP_WARM_MS = 10000;
    const extraPages = new Set(this.pageLru);
    for (const pageLru of needed.reverse()) {
      extraPages.delete(pageLru);

      pageLru.lru = now;
      this.pageLru.delete(pageLru);
      this.pageLru.add(pageLru);
    }
    // Exclude recently-uploaded pages from freeable
    const freeablePages: number[] = [];
    for (const pageLru of extraPages) {
      const { page } = pageLru;
      const info = this.pageToSplatsChunk[page];
      const age = info ? now - info.time : PAGE_KEEP_WARM_MS + 1;
      if (age > PAGE_KEEP_WARM_MS) {
        freeablePages.push(page);
      }
    }
    this.freeablePages = freeablePages;
    if (enqueued > 0) {
      console.log(`[pager-perf] driveFetchers: enqueued ${enqueued} new fetch(es), freeable=${this.freeablePages.length} pages`);
    }
  }

  private async runBackgroundPrefetch() {
    for (const splats of this.activeSplats) {
      if (this.fetchers.length >= this.numFetchers) return;

      if (splats.fileType === SplatFileType.RAD) {
        let meta;
        try {
          const radMeta = await splats.getRadMeta();
          meta = radMeta.meta;
        } catch (e) {
          continue;
        }

        const spotId =
          splats.rootUrl ||
          (splats.fileBlob as any)?.name ||
          (splats.fileBytes as any)?.name ||
          "local-drop";
        for (let chunk = 0; chunk < meta.chunks.length; chunk++) {
          if (this.fetchers.length >= this.numFetchers) return;

          // Check if already in priority list, fetching, or fetched
          if (
            this.fetchPriority.some(
              (p) => p.splats === splats && p.chunk === chunk,
            )
          )
            continue;
          if (
            this.fetchers.some((p) => p.splats === splats && p.chunk === chunk)
          )
            continue;
          if (
            this.fetched.some((p) => p.splats === splats && p.chunk === chunk)
          )
            continue;
          if (this.splatsChunkToPage.get(splats)?.[chunk]) continue;

          // Query cache
          const cached = await SplatCache.getChunk(spotId, chunk);
          if (cached !== null) continue; // Already in cache!

          // Fetch and cache in background!
          this.prefetchChunk(
            splats,
            chunk,
            spotId,
            meta.chunks[chunk].count || 0,
          );
        }
      }
    }
  }

  private async prefetchChunk(
    splats: PagedSplats,
    chunk: number,
    spotId: string,
    numSplats: number,
  ) {
    const promise = (async () => {
      try {
        let bytes: Uint8Array;
        if (splats.fileType === SplatFileType.RAD) {
          const { meta, chunksStart } = await splats.getRadMeta();
          const { offset, bytes: byteCount, filename } = meta.chunks[chunk];
          if (filename) {
            const resolvedRoot = new URL(
              splats.rootUrl,
              window.location.href,
            ).toString();
            const chunkUrl = new URL(filename, resolvedRoot).toString();
            bytes = await fetchRange({
              url: chunkUrl,
              requestHeader: splats.requestHeader,
              withCredentials: splats.withCredentials,
            });
          } else {
            if (splats.chunkSource) {
              bytes = await splats.chunkSource.read(
                offset + chunksStart,
                byteCount,
              );
            } else {
              return;
            }
          }
        } else {
          const url = splats.chunkUrl(chunk);
          const response = await fetch(url, {
            headers: splats.requestHeader
              ? new Headers(splats.requestHeader)
              : undefined,
            credentials: splats.withCredentials ? "include" : "same-origin",
          });
          bytes = new Uint8Array(await response.arrayBuffer());
        }

        await SplatCache.putChunk(
          spotId,
          chunk,
          0,
          bytes,
          numSplats,
          this.maxCacheSplats,
        );
      } catch (err) {
        console.warn("Background prefetch failed for chunk", chunk, err);
      }
    })().finally(() => {
      this.fetchers = this.fetchers.filter(
        (f) => f.splats !== splats || f.chunk !== chunk,
      );
      if (this.autoDrive) {
        this.driveFetchers();
      }
    });

    this.fetchers.push({ splats, chunk, promise });
  }

  private allocateFreeable(): number | undefined {
    const page = this.freeablePages.shift();
    if (page === undefined) {
      // No freeable pages available
      return undefined;
    }

    const splatsChunk = this.pageToSplatsChunk[page];
    if (!splatsChunk) {
      throw new Error(`splatsChunk not found for page: ${page}`);
    }

    const { splats, chunk } = splatsChunk;
    const pageAge = (performance.now() - splatsChunk.time).toFixed(0);
    console.log(
      `[pager-evict] evicting page=${page} chunk=${chunk} age=${pageAge}ms ` +
      `(freeable=${this.freeablePages.length + 1}, totalLRU=${this.pageLru.size})`,
    );
    this.removeSplatsChunkPage(splats, chunk, page);
    this.lodTreeUpdates.push({
      splats,
      page,
      chunk,
      numSplats: PAGE_SPLATS,
    });
    return page;
  }

  private processFetched() {
    const now = performance.now();
    while (true) {
      const fetched = this.fetched.shift();
      if (!fetched) {
        break;
      }
      const { splats, chunk, data } = fetched;

      let page = this.allocatePage();
      if (page === undefined) {
        page = this.allocateFreeable();
        if (page === undefined) {
          // No pages available, stop for now
          return;
        }
      }

      this.insertSplatsChunkPage(splats, chunk, page, now);
      const { numSplats, extra } = data;
      // decodeSp5Chunk synthesizes a chunk-relative flat lod tree (root entry's
      // child_start = 1, meaning "next slot in this same chunk") since it doesn't
      // know its own chunk index. Patch that to the real absolute address here,
      // where `chunk` is known, before this data reaches the WASM lod tree.
      // Deliberately CHUNK-based, not page-based: traverse_lod_trees /
      // dynamic_traverse_lod_trees (rust/spark-rs/src/lod_tree.rs) read
      // `child_start >> 16` as a manifest CHUNK index and translate it to a
      // physical page via `chunk_to_page[]` at traversal time -- that
      // indirection is exactly what lets a pointer baked in before a chunk's
      // final page assignment is known stay valid once it lands wherever
      // concurrent fetching happens to place it. (An earlier version of this
      // comment claimed this should use `page` instead -- that was wrong and
      // was reverted; verify against lod_tree.rs before changing this again.)
      if (extra.syntheticLodTree && extra.lodTree && extra.lodTree.length >= 4) {
        extra.lodTree[3] = (chunk * PAGE_SPLATS + extra.lodTree[3]) >>> 0;
      }
      console.log(
        `[lod-tree-update] pushing lodTreeUpdates entry: page=${page}, chunk=${chunk}, ` +
          `numSplats=${numSplats}, hasLodTree=${!!extra.lodTree}` +
          (numSplats
            ? ""
            : " -- WARNING: numSplats is falsy, this page will not become visible/measurable even though it uploaded"),
      );
      this.lodTreeUpdates.push({
        splats,
        page,
        chunk,
        numSplats: extra.lodTree ? extra.lodTree.length / 4 : numSplats,
        lodTree: extra.lodTree as Uint32Array,
      });

      if (isExtResult(data, this.extSplats)) {
        const extArrays = data.extArrays;
        const packedArray = extArrays[0];
        const extArray = extArrays[1];
        const shArrays = [
          data.extra.sh1 as Uint32Array,
          data.extra.sh2 as Uint32Array,
          data.extra.sh3a as Uint32Array,
          data.extra.sh3b as Uint32Array,
        ];
        const shLength = shArrays.findIndex((sh) => !sh);
        if (shLength !== -1) {
          shArrays.length = shLength;
        }
        this.newUploads.push({
          page,
          numSplats,
          packedArray,
          extArray,
          shArrays,
        });
      } else {
        const packedArray = data.packedArray;
        const shArrays = [
          data.extra.sh1 as Uint32Array,
          data.extra.sh2 as Uint32Array,
          data.extra.sh3 as Uint32Array,
        ];
        const shLength = shArrays.findIndex((sh) => !sh);
        if (shLength !== -1) {
          shArrays.length = shLength;
        }
        this.newUploads.push({
          page,
          numSplats,
          packedArray,
          shArrays,
        });
      }
    }
  }

  processUploads() {
    let uploadCount = 0;
    let uploadSplats = 0;
    const uploadStart = performance.now();
    while (true) {
      const upload = this.readyUploads.shift();
      if (!upload) {
        break;
      }
      const { page, numSplats, packedArray, extArray, shArrays } = upload;
      this.uploadPage(page, packedArray, shArrays, extArray);
      uploadCount++;
      uploadSplats += numSplats;
    }
    if (uploadCount > 0) {
      const elapsed = (performance.now() - uploadStart).toFixed(1);
      console.log(
        `[pager-perf] GPU upload: ${uploadCount} page(s), ${uploadSplats} splats, ${elapsed}ms`,
      );
    }
  }

  consumeLodTreeUpdates() {
    const updates = this.lodTreeUpdates;
    this.lodTreeUpdates = [];

    this.readyUploads.push(...this.newUploads);
    this.newUploads = [];
    return updates;
  }

  static emptyUint32x4 = (() => {
    const { width, height, depth, maxSplats } = getTextureSize(1);
    const emptyArray = new Uint32Array(maxSplats * 4);
    const texture = new THREE.DataArrayTexture(
      emptyArray,
      width,
      height,
      depth,
    );
    texture.format = THREE.RGBAIntegerFormat;
    texture.type = THREE.UnsignedIntType;
    texture.internalFormat = "RGBA32UI";
    texture.needsUpdate = true;
    return texture;
  })();

  static emptyUint32x2 = (() => {
    const { width, height, depth, maxSplats } = getTextureSize(1);
    const emptyArray = new Uint32Array(maxSplats * 2);
    const texture = new THREE.DataArrayTexture(
      emptyArray,
      width,
      height,
      depth,
    );
    texture.format = THREE.RGIntegerFormat;
    texture.type = THREE.UnsignedIntType;
    texture.internalFormat = "RG32UI";
    texture.needsUpdate = true;
    return texture;
  })();

  static emptyIndicesTexture = (() => {
    const emptyArray = new Uint32Array(4096 * 4);
    const texture = new THREE.DataTexture(emptyArray, 4096, 1);
    texture.format = THREE.RGBAIntegerFormat;
    texture.type = THREE.UnsignedIntType;
    texture.internalFormat = "RGBA32UI";
    texture.needsUpdate = true;
    return texture;
  })();

  static emptyPackedTexture = this.emptyUint32x4;
  static emptyExtTexture = this.emptyUint32x4;
  static emptyShTextures = [
    this.emptyUint32x2,
    this.emptyUint32x4,
    this.emptyUint32x4,
  ] as const;
  static emptyExtShTextures = [
    this.emptyUint32x4,
    this.emptyUint32x4,
    this.emptyUint32x4, // SH3A
    this.emptyUint32x4, // SH3B
  ] as const;
}

// Convenience function to distinguish ExtResult and PackedResult
function isExtResult(
  data: ExtResult | PackedResult,
  extSplats: boolean,
): data is ExtResult {
  return extSplats;
}

function uploadTextureLayer(
  texture: DynoUsampler2DArray<string, THREE.DataArrayTexture>,
  layer: number,
  dstOffset: number,
  data: Uint32Array,
) {
  const array = texture.value.image.data;
  array.subarray(dstOffset, dstOffset + data.length).set(data);

  texture.value.addLayerUpdate(layer);
  texture.value.needsUpdate = true;
  texture.value.source.dataReady = true;
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
  withCredentials?: boolean;
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
