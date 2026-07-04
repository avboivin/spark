import { PackedResult } from './defines';
export declare function decodeSp5Chunk({ chunkBytes, siblingChunks, }: {
    chunkBytes: Uint8Array;
    siblingChunks?: {
        chunkIndex: number;
        center: [number, number, number];
        size: number;
    }[];
}): Promise<PackedResult>;
