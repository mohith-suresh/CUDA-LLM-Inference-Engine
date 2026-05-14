// Step 11 — Double buffering (software pipelining)
// ----------------------------------------------------------------------------
// Same 2D micro-tile / vectorized loads as step 6, but with TWO shared-memory
// buffers. While the SM computes on the current buffer, it PREFETCHES the next
// K-tile from global memory into the other buffer. This overlaps global-load
// latency with compute and needs only one __syncthreads per iteration.
//
//   buffer `cur`  = the tile we compute on this iteration
//   buffer `nxt`  = the tile we load for the next iteration
//
// The load code is inlined twice (prologue + prefetch) rather than hidden in a
// device function, so the pipeline structure is visible at a glance.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void double_buffered_kernel(int M, int N, int K, float alpha,
                                       const float* A, const float* B, float beta, float* C) {
    __shared__ float As[2][BK * BM];    // two transposed A buffers
    __shared__ float Bs[2][BK * BN];    // two B buffers

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

    const int loadRowA = threadIdx.x / (BK / 4), loadColA = threadIdx.x % (BK / 4);
    const int loadRowB = threadIdx.x / (BN / 4), loadColB = threadIdx.x % (BN / 4);

    const int numTiles = K / BK;
    float acc[TM * TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    // --- prologue: load the first tile (k-offset 0) into buffer 0 ---
    {
        float4 a = reinterpret_cast<const float4*>(&A[loadRowA * K + loadColA * 4])[0];
        As[0][(loadColA * 4 + 0) * BM + loadRowA] = a.x;
        As[0][(loadColA * 4 + 1) * BM + loadRowA] = a.y;
        As[0][(loadColA * 4 + 2) * BM + loadRowA] = a.z;
        As[0][(loadColA * 4 + 3) * BM + loadRowA] = a.w;
        reinterpret_cast<float4*>(&Bs[0][loadRowB * BN + loadColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[loadRowB * N + loadColB * 4])[0];
    }
    __syncthreads();

    for (int t = 0; t < numTiles - 1; ++t) {
        const int cur = t & 1;
        const int nxt = cur ^ 1;

        // --- prefetch the NEXT tile (k-offset (t+1)*BK) into buffer `nxt` ---
        const float* Anext = A + (t + 1) * BK;
        const float* Bnext = B + (t + 1) * BK * N;
        {
            float4 a = reinterpret_cast<const float4*>(&Anext[loadRowA * K + loadColA * 4])[0];
            As[nxt][(loadColA * 4 + 0) * BM + loadRowA] = a.x;
            As[nxt][(loadColA * 4 + 1) * BM + loadRowA] = a.y;
            As[nxt][(loadColA * 4 + 2) * BM + loadRowA] = a.z;
            As[nxt][(loadColA * 4 + 3) * BM + loadRowA] = a.w;
            reinterpret_cast<float4*>(&Bs[nxt][loadRowB * BN + loadColB * 4])[0] =
                reinterpret_cast<const float4*>(&Bnext[loadRowB * N + loadColB * 4])[0];
        }

        // --- compute on the CURRENT buffer while the prefetch is in flight ---
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) regA[i] = As[cur][k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; ++j) regB[j] = Bs[cur][k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regA[i] * regB[j];
        }
        __syncthreads();   // ensure prefetch landed before next iter reads it
    }

    // --- epilogue compute: the last tile is already resident ---
    {
        const int cur = (numTiles - 1) & 1;
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) regA[i] = As[cur][k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; ++j) regB[j] = Bs[cur][k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regA[i] * regB[j];
        }
    }

    // GEMM epilogue: C = alpha*acc + beta*C.
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

inline void run_double_buffered(int M, int N, int K, float alpha,
                                const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));                 // 256 threads
    dim3 grid(N / BN, M / BM);
    double_buffered_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
