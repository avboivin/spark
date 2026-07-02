// Shared Huffman decode logic — used by both worker.ts (production path)
// and test files (smoke_test.ts, sp5_ordering_test.ts, etc.) so that every
// change to the decoder is exercised by the attribute-correctness tests.
//
// The WASM function `decode_huffman_fast` is injected by callers because
// test files import it directly from spark-rs while worker.ts imports it
// alongside other bindings.

export interface HuffmanTable {
  [symbol: string]: [code_len: number, code_bits: number];
}

const _huffmanLutCache = new Map<HuffmanTable, Uint16Array>();

export function buildHuffmanLut(htable: HuffmanTable): {
  lut: Uint16Array;
  fallbackTable?: Map<string, number>; // JS fallback for codes > 16 bits
} {
  let lut = _huffmanLutCache.get(htable);
  if (lut) {
    const fallbackTable = (lut as any)._fallback as Map<string, number> | undefined;
    return { lut, fallbackTable };
  }

  let maxLen = 0;
  const entries: [number, number, number][] = [];
  for (const [symStr, [len, bits]] of Object.entries(htable)) {
    const sym = Number.parseInt(symStr);
    entries.push([sym, len, bits]);
    if (len > maxLen) maxLen = len;
  }

  if (maxLen > 16) {
    const fallbackTable = new Map<string, number>();
    for (const [s, l, b] of entries) fallbackTable.set(`${b},${l}`, s);
    const fallbackLut = new Uint16Array(0);
    (fallbackLut as any)._fallback = fallbackTable;
    _huffmanLutCache.set(htable, fallbackLut);
    return { lut: fallbackLut, fallbackTable };
  }

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
  return { lut, fallbackTable: undefined };
}

export function decodeHuffmanJsFallback(
  bytes: Uint8Array,
  fallbackTable: Map<string, number>,
  count: number,
): Uint16Array {
  const out = new Uint16Array(count);
  let outIdx = 0, currentBits = 0, currentLen = 0, byteIdx = 0, bitIdx = 7;
  while (outIdx < count && byteIdx < bytes.length) {
    const bit = (bytes[byteIdx] >> bitIdx) & 1;
    bitIdx--; if (bitIdx < 0) { bitIdx = 7; byteIdx++; }
    currentBits = (currentBits << 1) | bit; currentLen++;
    const key = `${currentBits},${currentLen}`;
    if (fallbackTable.has(key)) {
      out[outIdx++] = fallbackTable.get(key)!;
      currentBits = 0; currentLen = 0;
    }
  }
  return out;
}

export function decodeHuffman(
  bytes: Uint8Array,
  htable: HuffmanTable,
  count: number,
  decodeWasm: (bytes: Uint8Array, lut: Uint16Array, count: number) => Uint16Array,
): Uint16Array {
  const { lut, fallbackTable } = buildHuffmanLut(htable);
  if (fallbackTable) {
    return decodeHuffmanJsFallback(bytes, fallbackTable, count);
  }
  // Copy bytes to avoid detaching shared ArrayBuffers during WASM access.
  const bytesCopy = bytes.slice();
  const decoded = decodeWasm(bytesCopy, lut, count);
  const out = new Uint16Array(count);
  out.set(decoded.subarray(0, Math.min(decoded.length, count)));
  return out;
}
