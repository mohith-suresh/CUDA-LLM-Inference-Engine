// Step 8 — Resolve bank conflicts via extra-column padding
// ----------------------------------------------------------------------------
// A simpler, more general alternative to step 7's swizzle: pad each row of Bs
// with a few EXTRA columns so successive rows start on different banks. This
// shifts the bank alignment of the shared accesses and removes the conflicts
// without any hand-computed swizzle indexing.
//
// The only change from step 6 is the Bs row stride: (BN + extraCols) instead of
// BN, used identically in both the store and the compute read. extraCols = 5 is
// Boehm's tuned value (a small non-power-of-two shift works best).
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void bank_extra_col_kernel(int M, int N, int K, float alpha,
                                      const float* A, const float* B, float beta, float* C) {
    constexpr int extraCols = 5;
    __shared__ float As[BM * BK];                 // transposed, as in step 6
    __shared__ float Bs[BK * (BN + extraCols)];   // padded row stride

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

    float acc[TM * TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (int kTile = 0; kTile < K; kTile += BK) {
        float4 a = reinterpret_cast<const float4*>(&A[loadRowA * K + loadColA * 4])[0];
        As[(loadColA * 4 + 0) * BM + loadRowA] = a.x;
        As[(loadColA * 4 + 1) * BM + loadRowA] = a.y;
        As[(loadColA * 4 + 2) * BM + loadRowA] = a.z;
        As[(loadColA * 4 + 3) * BM + loadRowA] = a.w;

        // Store into the padded layout (row stride BN + extraCols).
        float4 b = reinterpret_cast<const float4*>(&B[loadRowB * N + loadColB * 4])[0];
        Bs[loadRowB * (BN + extraCols) + loadColB * 4 + 0] = b.x;
        Bs[loadRowB * (BN + extraCols) + loadColB * 4 + 1] = b.y;
        Bs[loadRowB * (BN + extraCols) + loadColB * 4 + 2] = b.z;
        Bs[loadRowB * (BN + extraCols) + loadColB * 4 + 3] = b.w;
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i)
                regA[i] = As[k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; ++j)
                regB[j] = Bs[k * (BN + extraCols) + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

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

inline void run_bank_extra_col(int M, int N, int K, float alpha,
                               const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));                 // 256 threads
    dim3 grid(N / BN, M / BM);
    bank_extra_col_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
