// Step 7 — Resolve shared-memory bank conflicts (swizzled Bs layout)
// ----------------------------------------------------------------------------
// After step 6, As reads are conflict-free (transposed), but the Bs read
//     regB[i] = Bs[k*BN + threadCol*TN + i]
// still hits only a few banks: threadCol*TN has stride TN=8, so the 16 lanes
// map to only 4 distinct banks -> a 4-way bank conflict.
//
// Fix (Boehm's kernel 7): store Bs in a SWIZZLED physical layout so the read
// becomes  regB[i] = Bs[(k*TN + i) * (BN/TN) + threadCol]  — threadCol is now
// the innermost, stride-1 index, so all 16 lanes hit distinct banks.
//
// The swizzle constants below are specialized for the 128/128/8/8/8 config
// (TN = 8, BN/TN = 16), matching the tile used since step 6.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void bank_conflicts_kernel(int M, int N, int K, float alpha,
                                      const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM * BK];       // transposed, as in step 6
    __shared__ float Bs[BK * BN];       // swizzled physical layout (see below)

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
        // A: transposed store (unchanged from step 6).
        float4 a = reinterpret_cast<const float4*>(&A[loadRowA * K + loadColA * 4])[0];
        As[(loadColA * 4 + 0) * BM + loadRowA] = a.x;
        As[(loadColA * 4 + 1) * BM + loadRowA] = a.y;
        As[(loadColA * 4 + 2) * BM + loadRowA] = a.z;
        As[(loadColA * 4 + 3) * BM + loadRowA] = a.w;

        // B: scatter each float4 so the compute read below is stride-1 in lane.
        // Physical Bs is [BK*TN][BN/TN]; here 8 = TN and 16 = BN/TN.
        float4 b = reinterpret_cast<const float4*>(&B[loadRowB * N + loadColB * 4])[0];
        Bs[((loadColB % 2) * 4 + loadRowB * 8 + 0) * 16 + loadColB / 2] = b.x;
        Bs[((loadColB % 2) * 4 + loadRowB * 8 + 1) * 16 + loadColB / 2] = b.y;
        Bs[((loadColB % 2) * 4 + loadRowB * 8 + 2) * 16 + loadColB / 2] = b.z;
        Bs[((loadColB % 2) * 4 + loadRowB * 8 + 3) * 16 + loadColB / 2] = b.w;
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i)
                regA[i] = As[k * BM + threadRow * TM + i];
            for (int i = 0; i < TN; ++i)
                regB[i] = Bs[(k * 8 + i) * 16 + threadCol];    // stride-1 in threadCol
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

inline void run_bank_conflicts(int M, int N, int K, float alpha,
                               const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));                 // 256 threads
    dim3 grid(N / BN, M / BM);
    bank_conflicts_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
