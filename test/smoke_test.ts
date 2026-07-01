import fs from 'fs';
import path from 'path';
import init_wasm, { decode_to_gsplatarray, reconstruct_sp5_chunk } from '../rust/spark-rs/pkg/spark_rs.js';
import { convertSplatToSp5Client } from '../src/converter.js';
import * as fflate from 'fflate';

async function main() {
  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  const wasmBuffer = fs.readFileSync(wasmPath);
  await init_wasm({ module_or_path: wasmBuffer });
  console.log("WASM Initialized!");

  const inputPath = 'C:\\splat\\mont_et_merv\\montmerv188_clean.ply';
  const outputDir = 'C:\\splat\\mont_et_merv\\spz_out';
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log(`Reading input PLY from ${inputPath}...`);
  const plyBytes = new Uint8Array(fs.readFileSync(inputPath));

  console.log("Decoding PLY into GsplatArray...");
  const decoder = decode_to_gsplatarray("ply", inputPath);
  
  // Feed bytes to decoder
  const CHUNK_SIZE = 1024 * 1024;
  for (let i = 0; i < plyBytes.length; i += CHUNK_SIZE) {
    decoder.push(plyBytes.subarray(i, Math.min(i + CHUNK_SIZE, plyBytes.length)));
  }
  // Finish decoder and get GsplatArray
  const decoded = decoder.finish();
  const numSplats = decoded.len();
  const maxSh = decoded.maxShDegree;
  console.log(`Successfully decoded PLY: ${numSplats} splats, maxSh: ${maxSh}`);

  console.log("Extracting splat attributes...");
  const attrs = decoded.extract_attributes();
  const xyz = new Float32Array(attrs.xyz);
  const opacity = new Float32Array(attrs.opacity);
  const rgb = new Float32Array(attrs.rgb);
  const scales = new Float32Array(attrs.scales);
  const quaternions = new Float32Array(attrs.quaternions);
  const sh1 = maxSh > 0 ? new Float32Array(attrs.sh1) : undefined;
  decoded.free();

  console.log("Running C1 SVQ-only conversion...");
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
      console.log(`  [Progress] ${phase} (${percent}%)`);
    }
  });

  const zipOutPath = path.join(outputDir, 'montmerv188-sp5.zip');
  fs.writeFileSync(zipOutPath, zipBytes);
  console.log(`Wrote SP5 ZIP package to ${zipOutPath} (${zipBytes.length} bytes)`);

  // Unzip the package to perform the decode roundtrip validation!
  console.log("Unzipping SP5 package to validate decode channels...");
  const unzipped = fflate.unzipSync(zipBytes);
  const manifest = JSON.parse(new TextDecoder().decode(unzipped['manifest.json']));
  console.log("Manifest loaded:", manifest);

  for (const chunkInfo of manifest.chunks) {
    const chunkBytes = unzipped[chunkInfo.file];
    console.log(`Validating chunk ${chunkInfo.file} (${chunkBytes.length} bytes)...`);
    
    // Parse .sp5 chunk header
    const view = new DataView(chunkBytes.buffer, chunkBytes.byteOffset);
    const magic = view.getUint32(0, true);
    if (magic !== 0x355a5053) {
      throw new Error(`Invalid chunk magic: ${magic.toString(16)}`);
    }
    const jsonSize = view.getUint32(4, true);
    const jsonBytes = chunkBytes.subarray(8, 8 + jsonSize);
    const chunkMeta = JSON.parse(new TextDecoder().decode(jsonBytes));
    const binaryPayload = chunkBytes.subarray(8 + jsonSize);

    // Reconstruct attributes using helper decoders
    function halfToFloat(binary: number) {
      const exponent = (binary & 0x7c00) >> 10;
      const fraction = binary & 0x03ff;
      if (exponent === 0) {
        return (binary & 0x8000 ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
      } else if (exponent === 0x1f) {
        return fraction ? NaN : binary & 0x8000 ? -Infinity : Infinity;
      }
      return (binary & 0x8000 ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }

    function float16ArrayToFloat32Array(bytes: Uint8Array) {
      let alignedBytes = bytes;
      if (bytes.byteOffset % 2 !== 0) {
        alignedBytes = new Uint8Array(bytes.length);
        alignedBytes.set(bytes);
      }
      const u16 = new Uint16Array(alignedBytes.buffer, alignedBytes.byteOffset, alignedBytes.byteLength / 2);
      const out = new Float32Array(u16.length);
      for (let i = 0; i < u16.length; i++) {
        out[i] = halfToFloat(u16[i]);
      }
      return out;
    }

    function getBinaryPart(meta: { offset: number; length: number }) {
      return binaryPayload.subarray(meta.offset, meta.offset + meta.length);
    }

    // Decode Huffman indices
    function decodeHuffman(bytes: Uint8Array, htable: Record<string, [number, number]>, count: number) {
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

    const count = chunkMeta.count;
    const xyzBytes = getBinaryPart(chunkMeta.xyz_uncompressed);
    const xyzRawFloat = float16ArrayToFloat32Array(xyzBytes);

    const scaleIndices: number[] = [];
    for (const hmeta of chunkMeta.scale_index_huffman) {
      const decoded = decodeHuffman(getBinaryPart(hmeta), hmeta.huffman_table, count);
      for (let i = 0; i < count; i++) scaleIndices.push(decoded[i]);
    }

    const rotationIndices: number[] = [];
    for (const hmeta of chunkMeta.rotation_index_huffman) {
      const decoded = decodeHuffman(getBinaryPart(hmeta), hmeta.huffman_table, count);
      for (let i = 0; i < count; i++) rotationIndices.push(decoded[i]);
    }

    const appIndices: number[] = [];
    for (const hmeta of chunkMeta.app_index_huffman) {
      const decoded = decodeHuffman(getBinaryPart(hmeta), hmeta.huffman_table, count);
      for (let i = 0; i < count; i++) appIndices.push(decoded[i]);
    }

    const scaleCbFlat = new Float32Array(chunkMeta.scale_codebook.length * 256);
    chunkMeta.scale_codebook.forEach((meta: any, idx: number) => {
      scaleCbFlat.set(float16ArrayToFloat32Array(getBinaryPart(meta)), idx * 256);
    });

    const rotationCbFlat = new Float32Array(chunkMeta.rotation_codebook.length * 512);
    chunkMeta.rotation_codebook.forEach((meta: any, idx: number) => {
      rotationCbFlat.set(float16ArrayToFloat32Array(getBinaryPart(meta)), idx * 512);
    });

    const appCbFlat = new Float32Array(chunkMeta.app_codebook.length * 512);
    chunkMeta.app_codebook.forEach((meta: any, idx: number) => {
      appCbFlat.set(float16ArrayToFloat32Array(getBinaryPart(meta)), idx * 512);
    });

    const mlpCont = float16ArrayToFloat32Array(getBinaryPart(chunkMeta.mlp_cont));
    const mlpDc = float16ArrayToFloat32Array(getBinaryPart(chunkMeta.mlp_dc));
    const mlpSh = float16ArrayToFloat32Array(getBinaryPart(chunkMeta.mlp_sh));
    const mlpOpacity = float16ArrayToFloat32Array(getBinaryPart(chunkMeta.mlp_opacity));

    // Call the Rust WASM decoder!
    console.log("  Invoking reconstruct_sp5_chunk in Rust WASM...");
    const reconstructed = reconstruct_sp5_chunk(
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
      new Float32Array(0), new Float32Array(0),
      new Float32Array(0), new Float32Array(0),
      new Float32Array(0), new Float32Array(0),
      new Float32Array(0), new Float32Array(0)
    );

    console.log(`  Decoded chunk successfully: count=${reconstructed.len()}, maxShDegree=${reconstructed.maxShDegree}`);
    reconstructed.free();
  }

  console.log("SMOKE TEST PASSED SUCCESSFULLY!");
}

main().catch(e => {
  console.error("SMOKE TEST FAILED:", e);
  process.exit(1);
});
