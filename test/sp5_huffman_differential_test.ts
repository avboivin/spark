// Differential test: JS reference bit-loop Huffman decoder vs WASM decode_huffman_fast,
// on synthetic streams with skewed symbol distributions (maxLen 9..16, the range that
// takes the WASM LUT path in worker.ts).
import fs from 'fs';
import path from 'path';
import init_wasm, { decode_huffman_fast } from '../rust/spark-rs/pkg/spark_rs.js';

interface HuffmanNode { symbol?: number; freq: number; left?: HuffmanNode; right?: HuffmanNode; }

// Exact copies of converter.ts's encoder-side logic
function buildHuffmanTable(frequencies: Record<number, number>): Record<number, [number, number]> {
  const nodes: HuffmanNode[] = Object.entries(frequencies).map(([sym, freq]) => ({ symbol: parseInt(sym), freq }));
  while (nodes.length < 2) nodes.push({ symbol: -1, freq: 0 });
  while (nodes.length > 1) {
    nodes.sort((a, b) => a.freq - b.freq);
    const left = nodes.shift()!;
    const right = nodes.shift()!;
    nodes.push({ freq: left.freq + right.freq, left, right });
  }
  const table: Record<number, [number, number]> = {};
  function traverse(node: HuffmanNode, bits: number, len: number) {
    if (node.symbol !== undefined) { if (node.symbol !== -1) table[node.symbol] = [len, bits]; return; }
    if (node.left) traverse(node.left, (bits << 1) | 0, len + 1);
    if (node.right) traverse(node.right, (bits << 1) | 1, len + 1);
  }
  traverse(nodes[0], 0, 0);
  return table;
}

function encodeHuffman(indices: Uint8Array, htable: Record<number, [number, number]>): Uint8Array {
  const bytes: number[] = [];
  let currentByte = 0, bitCount = 0;
  for (let i = 0; i < indices.length; i++) {
    const code = htable[indices[i]];
    if (!code) continue;
    const [len, bits] = code;
    for (let b = len - 1; b >= 0; b--) {
      currentByte = (currentByte << 1) | ((bits >> b) & 1);
      bitCount++;
      if (bitCount === 8) { bytes.push(currentByte); currentByte = 0; bitCount = 0; }
    }
  }
  if (bitCount > 0) bytes.push(currentByte << (8 - bitCount));
  return new Uint8Array(bytes);
}

// JS reference decoder (the original worker.ts bit-loop)
function decodeJsReference(bytes: Uint8Array, htable: Record<number, [number, number]>, count: number): Uint16Array {
  const table = new Map<string, number>();
  for (const [symStr, [len, bits]] of Object.entries(htable)) table.set(`${bits},${len}`, Number.parseInt(symStr));
  const out = new Uint16Array(count);
  let outIdx = 0, currentBits = 0, currentLen = 0, byteIdx = 0, bitIdx = 7;
  while (outIdx < count && byteIdx < bytes.length) {
    const bit = (bytes[byteIdx] >> bitIdx) & 1;
    bitIdx--; if (bitIdx < 0) { bitIdx = 7; byteIdx++; }
    currentBits = (currentBits << 1) | bit; currentLen++;
    if (table.has(`${currentBits},${currentLen}`)) {
      out[outIdx++] = table.get(`${currentBits},${currentLen}`)!;
      currentBits = 0; currentLen = 0;
    }
  }
  return out;
}

// Exact copy of worker.ts's LUT builder (the non-fallback branch)
function buildLut(htable: Record<number, [number, number]>): { lut: Uint16Array; maxLen: number } {
  let maxLen = 0;
  const entries: [number, number, number][] = [];
  for (const [symStr, [len, bits]] of Object.entries(htable)) {
    entries.push([Number.parseInt(symStr), len, bits]);
    if (len > maxLen) maxLen = len;
  }
  const lutSize = 1 << maxLen;
  const lut = new Uint16Array(lutSize);
  lut.fill(0xFFFF);
  for (const [symbol, len, bits] of entries) {
    const spread = 1 << (maxLen - len);
    const base = bits << (maxLen - len);
    for (let i = 0; i < spread; i++) lut[base + i] = (len << 8) | symbol;
  }
  return { lut, maxLen };
}

async function main() {
  const wasmPath = path.resolve('./rust/spark-rs/pkg/spark_rs_bg.wasm');
  await init_wasm({ module_or_path: fs.readFileSync(wasmPath) });

  // Skewed distribution: mimics real scale-index streams (a few codebook entries dominate).
  // Zipf-like over 64 symbols -> maxLen typically 9-14, i.e. the WASM LUT path.
  const count = 65536;
  const numSymbols = 64;
  const symbols = new Uint8Array(count);
  let seed = 12345;
  const rand = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let i = 0; i < count; i++) {
    // Zipf sample
    const u = rand();
    symbols[i] = Math.min(numSymbols - 1, Math.floor(numSymbols * Math.pow(u, 3)));
  }

  const freq: Record<number, number> = {};
  for (const s of symbols) freq[s] = (freq[s] || 0) + 1;
  const htable = buildHuffmanTable(freq);
  const encoded = encodeHuffman(symbols, htable);
  const { lut, maxLen } = buildLut(htable);

  console.log(`symbols=${count}, distinct=${Object.keys(freq).length}, maxCodeLen=${maxLen} (WASM LUT path requires <=16)`);
  if (maxLen > 16) { console.log('maxLen > 16 would take JS fallback -- rerun with different seed'); process.exit(2); }

  const refOut = decodeJsReference(encoded, htable, count);
  const wasmOutRaw = decode_huffman_fast(encoded.slice(), lut, count);
  const wasmOut = new Uint16Array(count);
  wasmOut.set(wasmOutRaw.subarray(0, Math.min(wasmOutRaw.length, count)));

  // First check the reference decodes the source exactly (sanity)
  let refErrors = 0;
  for (let i = 0; i < count; i++) if (refOut[i] !== symbols[i]) refErrors++;
  console.log(`JS reference vs source symbols: ${refErrors} mismatches (must be 0)`);

  let wasmErrors = 0;
  let firstError = -1;
  for (let i = 0; i < count; i++) {
    if (wasmOut[i] !== symbols[i]) { wasmErrors++; if (firstError < 0) firstError = i; }
  }
  console.log(`WASM decode_huffman_fast vs source: ${wasmErrors} mismatches (${(100 * wasmErrors / count).toFixed(2)}%), first at index ${firstError}`);
  console.log(`WASM returned ${wasmOutRaw.length} symbols of ${count} requested`);

  if (wasmErrors === 0 && refErrors === 0) {
    console.log('\nPASS: WASM decoder is bit-exact with JS reference.');
  } else {
    console.log('\nFAIL: WASM Huffman decoder output diverges from ground truth.');
    console.log('First 20 (source vs wasm):');
    for (let i = Math.max(0, firstError - 2); i < Math.min(count, firstError + 18); i++) {
      console.log(`  [${i}] src=${symbols[i]} wasm=${wasmOut[i]}${symbols[i] !== wasmOut[i] ? '  <-- MISMATCH' : ''}`);
    }
    process.exit(1);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
