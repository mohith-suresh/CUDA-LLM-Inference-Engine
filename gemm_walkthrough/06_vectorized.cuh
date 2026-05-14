// Step 6 — Vectorized memory access (float4) + transposed As
// ----------------------------------------------------------------------------
// Same 2D micro-tile as step 5, with two memory-system refinements:
//
//  1. float4 loads. One 128-bit LDG.128 moves 4 floats in a single instruction
//     instead of four LDG.32 — fewer instructions, wider transactions.
//
//  2. As stored TRANSPOSED as [BK][BM]. In step 5 the inner loop read an A
//     column with a stride of BK across shared-memory banks (conflicts).
//     Transposing makes that read contiguous, so the A operand is conflict-free.
//
// Tile sizes are chosen so each thread loads exactly one float4 of A and one of
// B (numThreads == BM*BK/4 == BK*BN/4), keeping the loads branch-free.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void vectorized_kernel(int M, int N, int K, float alpha,
                                  const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BK * BM];       // TRANSPOSED: As[k][m] at k*BM + m
    __shared__ float Bs[BK * BN];       //             Bs[k][n] at k*BN + n

    constexpr int numThreads = (BM / TM) * (BN / TN);
    static_assert(numThreads == (BM * BK) / 4, "one float4 of A per thread");
    static_assert(numThreads == (BK * BN) / 4, "one float4 of B per thread");

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    const int blockRow = blockIdx.y;               // C-tile row in the block grid
    const int blockCol = blockIdx.x;               // C-tile col in the block grid

    // Advance each pointer to the top-left element of this block's tile.
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    // float4 load coordinates (columns counted in groups of 4).
    const int loadRowA = threadIdx.x / (BK / 4), loadColA = threadIdx.x % (BK / 4);
    const int loadRowB = threadIdx.x / (BN / 4), loadColB = threadIdx.x % (BN / 4);

    float acc[TM * TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (int kTile = 0; kTile < K; kTile += BK) {
        // Load 4 contiguous A elements, scatter them into the transposed layout.
        float4 a = reinterpret_cast<const float4*>(&A[loadRowA * K + loadColA * 4])[0];
        As[(loadColA * 4 + 0) * BM + loadRowA] = a.x;
        As[(loadColA * 4 + 1) * BM + loadRowA] = a.y;
        As[(loadColA * 4 + 2) * BM + loadRowA] = a.z;
        As[(loadColA * 4 + 3) * BM + loadRowA] = a.w;

        // Load 4 contiguous B elements straight through (already row-friendly).
        reinterpret_cast<float4*>(&Bs[loadRowB * BN + loadColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[loadRowB * N + loadColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i)
                regA[i] = As[k * BM + threadRow * TM + i];      // contiguous now
            for (int j = 0; j < TN; ++j)
                regB[j] = Bs[k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    // Vectorized GEMM epilogue: C = alpha*acc + beta*C, one float4 at a time.
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; j += 4) {
            float* cptr = &C[(threadRow * TM + i) * N + threadCol * TN + j];
            float4 old = reinterpret_cast<float4*>(cptr)[0];
            float4 v;
            v.x = alpha * acc[i * TN + j + 0] + beta * old.x;
            v.y = alpha * acc[i * TN + j + 1] + beta * old.y;
            v.z = alpha * acc[i * TN + j + 2] + beta * old.z;
            v.w = alpha * acc[i * TN + j + 3] + beta * old.w;
            reinterpret_cast<float4*>(cptr)[0] = v;
        }
}

inline void run_vectorized(int M, int N, int K, float alpha,
                           const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));                 // 256 threads
    dim3 grid(N / BN, M / BM);
    vectorized_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
