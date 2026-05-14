// Step 5b — 2D block tiling, but As is stored TRANSPOSED in shared memory.
// ----------------------------------------------------------------------------
// ISOLATION EXPERIMENT. Identical to 05_blocktiling_2d in EVERY way except the
// shared-memory layout of As. No float4, no vectorization — the only change is
// where each As element lands in shared memory. Goal: measure how much of the
// "3.2-way bank conflict / high Mem Busy" in K05 is killed by the transpose
// ALONE (vs. the vectorization that K06 bundles on top).
//
// Why the transpose fixes the As read:
//   K05 stores As as [BM][BK] (row-major), so the inner loop reads
//       As[(microRow*TM + i)*BK + k]        // stride BK=8 between micro-rows
//   Two micro-rows in a warp land on addresses 64 apart -> same bank -> 2-way
//   conflict on every As load.
//   Here As is stored as [BK][BM], so the read becomes
//       As[k*BM + (microRow*TM + i)]        // stride 1 between micro-rows
//   Consecutive micro-rows now fall in consecutive banks -> conflict-free.
//
// NOTE (be honest): this only removes the As-side conflict. The Bs read
//   Bs[k*BN + microCol*TN + j]  keeps its own (stride-TN) conflict. So expect a
//   PARTIAL drop in Mem Busy, not the full ~3.2x. That is exactly the point of
//   the experiment — attribute the win to As-transpose vs. everything else.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void blocktiling_2d_transA_kernel(int M, int N, int K, float alpha,
                                             const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BK * BM];      // <-- TRANSPOSED: [BK][BM] instead of [BM][BK]
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

        // Cooperative load A — store TRANSPOSED: element (row,col) -> As[col*BM + row]
        for (int idx = threadIdx.x; idx < BM * BK; idx += numThreads) {
            int row = idx / BK;                       // row in [0,BM)
            int col = idx % BK;                       // col in [0,BK)
            As[col * BM + row] = A[(blockRow0 + row) * K + k0 + col];   // <-- transposed store
        }

        // Cooperative load B — unchanged
        for (int idx = threadIdx.x; idx < BK * BN; idx += numThreads) {
            int row = idx / BN;
            int col = idx % BN;
            Bs[idx] = B[(k0 + row) * N + blockCol0 + col];
        }

        __syncthreads();

        // Register outer products
        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            float regB[TN];

            for (int i = 0; i < TM; ++i)
                regA[i] = As[k * BM + microRow * TM + i];   // <-- transposed read (was (microRow*TM+i)*BK + k)

            for (int j = 0; j < TN; ++j)
                regB[j] = Bs[k * BN + microCol * TN + j];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int globalRow = blockRow0 + microRow * TM + i;
            int globalCol = blockCol0 + microCol * TN + j;
            int idx = globalRow * N + globalCol;
            C[idx] = alpha * acc[i * TN + j] + beta * C[idx];
        }
}

inline void run_blocktiling_2d_transA(int M, int N, int K, float alpha,
                                      const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));                 // 256 threads
    dim3 grid(N / BN, M / BM);
    blocktiling_2d_transA_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
