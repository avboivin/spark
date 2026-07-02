export declare function convertSplatToSp5Client({ numSplats, xyz, opacity, rgb, scales, quaternions, sh1, maxSh, onProgress, priorityPoint, lodBase, }: {
    numSplats: number;
    xyz: Float32Array;
    opacity: Float32Array;
    rgb: Float32Array;
    scales: Float32Array;
    quaternions: Float32Array;
    sh1?: Float32Array;
    maxSh: number;
    onProgress?: (phase: string, percent: number) => void;
    priorityPoint?: [number, number, number];
    lodBase?: number;
}): Promise<Uint8Array>;
