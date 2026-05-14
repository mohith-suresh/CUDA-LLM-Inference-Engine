// Step 5c — 2D block tiling with BOTH shared-load conflicts removed, no float4.
// ----------------------------------------------------------------------------
// Builds on 5b (As stored transposed -> As read is already conflict-free) and
// additionally removes the Bs conflict WITHOUT vectorization, by changing the
// output-tile ownership from a CONTIGUOUS block to a STRIDED set of columns.
//
//   Bs conflict in K05/5b:  regB[j] = Bs[k*BN + microCol*TN + j]
//     For a fixed j, the 16 microCols in a warp read columns microCol*8 apart
//     -> only 4 distinct banks -> 4-way conflict.
//
//   Fix here:               regB[j] = Bs[k*BN + microCol + j*threadTilesN]
//     For a fixed j, the 16 microCols read consecutive columns (stride 1)
//     -> 16 distinct banks -> conflict-free.
//
// This is a DIFFERENT (strided) tiling of the same C block, so the C-write index
// must match. As-side rows are left contiguous (already conflict-free via the
// transpose). Expected result: both As and Bs shared reads conflict-free ->
// Mem Busy should fall toward ~1x wavefronts (~24%, vs 71% in K05) and FMA
// becomes the limiter — all WITHOUT float4. K06 then adds the float4 win on top.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void blocktiling_2d_noconflict_kernel(int M, int N, int K, float alpha,
                                                 const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BK * BM];      // transposed [BK][BM] (from 5b)
    __shared__ float Bs[BK * BN];

    constexpr int threadTilesM = BM / TM;
    constexpr int threadTilesN = BN / TN;
    constexpr int numThreads   = threadTilesM * threadTilesN;

    int microRow = threadIdx.x / threadTilesN;
    int microCol = threadIdx.x % threadTilesN;

    int blockRow0 = blockIdx.y * BM;
    int blockCol0 = blockIdx.x * BN;

    float acc[TM * TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {

        // Load A transposed: element (row,col) -> As[col*BM + row]
        for (int idx = threadIdx.x; idx < BM * BK; idx += numThreads) {
            int row = idx / BK;
            int col = idx % BK;
            As[col * BM + row] = A[(blockRow0 + row) * K + k0 + col];
        }

        // Load B (unchanged)
        for (int idx = threadIdx.x; idx < BK * BN; idx += numThreads) {
            int row = idx / BN;
            int col = idx % BN;
            Bs[idx] = B[(k0 + row) * N + blockCol0 + col];
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            float regB[TN];

            // As read: transposed -> conflict-free (rows contiguous)
            for (int i = 0; i < TM; ++i)
                regA[i] = As[k * BM + microRow * TM + i];

            // Bs read: STRIDED columns -> consecutive threads hit consecutive banks
            for (int j = 0; j < TN; ++j)
                regB[j] = Bs[k * BN + microCol + j * threadTilesN];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    // C-write must match the STRIDED column ownership above.
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int globalRow = blockRow0 + microRow * TM + i;               // rows: contiguous block
            int globalCol = blockCol0 + microCol + j * threadTilesN;     // cols: strided
            int idx = globalRow * N + globalCol;
            C[idx] = alpha * acc[i * TN + j] + beta * C[idx];
        }
}

inline void run_blocktiling_2d_noconflict(int M, int N, int K, float alpha,
                                          const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));                 // 256 threads
    dim3 grid(N / BN, M / BM);
    blocktiling_2d_noconflict_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
