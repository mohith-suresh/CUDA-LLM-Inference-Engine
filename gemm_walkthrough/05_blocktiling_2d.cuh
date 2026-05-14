// Step 5 — 2D block tiling (a register micro-tile per thread)
// ----------------------------------------------------------------------------
// Each thread computes a TM x TN micro-tile of C via an OUTER PRODUCT. Per k,
// it loads TM values of A and TN values of B into registers, then does TM*TN
// FMAs. That is TM*TN FMAs for only TM+TN shared loads — the ratio that pushes
// arithmetic intensity high enough to become compute-bound.
//
//   numThreads = (BM/TM) * (BN/TN).  Loads are strided because each thread now
//   fetches several tile elements per step.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void blocktiling_2d_kernel(int M, int N, int K, float alpha,
                                      const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM * BK];
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

        // Cooperative load A
        for (int idx = threadIdx.x; idx < BM * BK; idx += numThreads) {
            int row = idx / BK;
            int col = idx % BK;
            As[idx] = A[(blockRow0 + row) * K + k0 + col];
        }

        // Cooperative load B
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
                regA[i] = As[(microRow * TM + i) * BK + k];

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

inline void run_blocktiling_2d(int M, int N, int K, float alpha,
                               const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));                 // 256 threads
    dim3 grid(N / BN, M / BM);
    blocktiling_2d_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
