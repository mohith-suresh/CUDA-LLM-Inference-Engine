// Step 4 — 1D block tiling (multiple results per thread)
// ----------------------------------------------------------------------------
// Each thread now computes a COLUMN of TM outputs instead of one. The payoff:
// a single B value loaded from shared memory feeds TM fused-multiply-adds held
// in registers. Reuse moves from shared memory up into registers.
//
// WHY A 1D THREAD INDEX (threadIdx.x only)?
//   The same `numThreads` threads are tiled into THREE different 2D shapes:
//     - loading As (BM x BK):  rows = tid/BK,  cols = tid%BK
//     - loading Bs (BK x BN):  rows = tid/BN,  cols = tid%BN
//     - computing  (BM/TM x BN): row = tid/BN, col = tid%BN
//   A 2D block would fix ONE shape and force manual re-linearization for the
//   others. Deriving every coordinate from a 1D index with / and % keeps all
//   phases uniform — this is the standard GEMM idiom (Boehm, CUTLASS, cuBLAS).
//
// Block tile: BM x BN of C, marched over K in steps of BK.
//   numThreads = BM*BN / TM   (each owns TM stacked outputs in one column)
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TM>
__global__ void blocktiling_1d_kernel(int M, int N, int K, float alpha,
                                      const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // This thread's position inside the BM x BN output tile.
    const int threadCol = threadIdx.x % BN;        // 0 .. BN-1
    const int microRow = threadIdx.x / BN;        // 0 .. BM/TM-1

    // This block computes the C tile at (blockRow, blockCol) in the block grid.
    const int blockRow = blockIdx.y * BM;               // C-tile row: 0 .. M/BM-1
    const int blockCol = blockIdx.x * BN;               // C-tile col: 0 .. N/BN-1


    float acc[TM] = {0.0f};                        // TM results in registers

    for (int kTile = 0; kTile < K; kTile += BK) {
        for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
            int row = blockRow + idx / BK;
            int col = kTile + idx % BK;
            As[idx] = A[row * K + col];
        }

        for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
            int row = kTile + idx / BN;
            int col = blockCol + idx % BN;
            Bs[idx] = B[row * N + col];
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float bVal = Bs[k * BN + threadCol];   // loaded once...
            for (int t = 0; t < TM; ++t)           // ...reused across TM rows
                acc[t] += As[(microRow * TM + t) * BK + k] * bVal;
        }
        __syncthreads();
    }

    for (int t = 0; t < TM; ++t) {
        int idx = (blockRow + microRow * TM + t) * N + blockCol + threadCol;  // global index of this output
        C[idx] = alpha * acc[t] + beta * C[idx];
    }
}

inline void run_blocktiling_1d(int M, int N, int K, float alpha,
                               const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 block((BM * BN) / TM);                    // 512 threads
    dim3 grid(N / BN, M / BM);
    blocktiling_1d_kernel<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
