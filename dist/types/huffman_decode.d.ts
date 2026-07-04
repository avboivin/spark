export interface HuffmanTable {
    [symbol: string]: [code_len: number, code_bits: number];
}
export declare function buildHuffmanLut(htable: HuffmanTable): {
    lut: Uint16Array;
    fallbackTable?: Map<string, number>;
};
export declare function decodeHuffmanJsFallback(bytes: Uint8Array, fallbackTable: Map<string, number>, count: number): Uint16Array;
export declare function decodeHuffman(bytes: Uint8Array, htable: HuffmanTable, count: number, decodeWasm: (bytes: Uint8Array, lut: Uint16Array, count: number) => Uint16Array): Uint16Array;
