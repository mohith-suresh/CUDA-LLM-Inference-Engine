// Step 10 — Warp tiling
// ----------------------------------------------------------------------------
// Adds a WARP-level tile between the block tile and the thread tile, making all
// three levels of the memory hierarchy explicit:
//
//   BLOCK tile  (BM x BN)   lives in shared memory
//     WARP tile (WM x WN)   owned by the 32 threads of one warp
//       THREAD tile         each thread computes WMITER*WNITER micro-tiles of TM x TN
//
// Why it helps: threads in a warp reuse the same shared-memory fragments, and
// laying the work out warp-by-warp keeps register operands in a pattern the
// scheduler can issue back-to-back — closing most of the remaining gap to cuBLAS.
//
// Default config is Boehm's A6000 tuning. On a smaller GPU (e.g. SM75 / 1650 Ti)
// the 128 accumulators per thread may spill registers; shrink the tiles if so.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK,
          int WM, int WN, int WNITER,
          int TM, int TN, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
warptiling_kernel(int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C) {
    // Derived warp-tile geometry (all compile-time constants).
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER); // M-subtiles per warp
    constexpr int WSUBM  = WM / WMITER;                         // subtile height
    constexpr int WSUBN  = WN / WNITER;                         // subtile width

    __shared__ float As[BK * BM];        // TRANSPOSED  As[k][m] at k*BM + m
    __shared__ float Bs[BK * BN];        //             Bs[k][n] at k*BN + n

    // ---- which warp is this, and where does its WM x WN tile sit? ----
    const int warpIdx = threadIdx.x / 32;
    const int warpRow = warpIdx / (BN / WN);
    const int warpCol = warpIdx % (BN / WN);

    // ---- this thread's lane position inside the warp subtile ----
    const int lane            = threadIdx.x % 32;
    const int threadRowInWarp = lane / (WSUBN / TN);
    const int threadColInWarp = lane % (WSUBN / TN);

    const int blockRow = blockIdx.y;               // C-tile row in the block grid
    const int blockCol = blockIdx.x;               // C-tile col in the block grid

    // A, B -> block-tile origin; C -> this warp's WM x WN sub-tile origin.
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += (blockRow * BM + warpRow * WM) * N + blockCol * BN + warpCol * WN;

    // ---- cooperative float4 loads (strided over the whole block) ----
    const int loadRowA = threadIdx.x / (BK / 4), loadColA = threadIdx.x % (BK / 4);
    const int strideA  = (NUM_THREADS * 4) / BK;
    const int loadRowB = threadIdx.x / (BN / 4), loadColB = threadIdx.x % (BN / 4);
    const int strideB  = NUM_THREADS / (BN / 4);

    float acc[WMITER * TM][WNITER * TN] = {0.0f};   // this thread's outputs
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    for (int kTile = 0; kTile < K; kTile += BK) {
        // Load A transposed.
        for (int r = 0; r < BM; r += strideA) {
            float4 a = reinterpret_cast<const float4*>(
                           &A[(loadRowA + r) * K + loadColA * 4])[0];
            As[(loadColA * 4 + 0) * BM + loadRowA + r] = a.x;
            As[(loadColA * 4 + 1) * BM + loadRowA + r] = a.y;
            As[(loadColA * 4 + 2) * BM + loadRowA + r] = a.z;
            As[(loadColA * 4 + 3) * BM + loadRowA + r] = a.w;
        }
        // Load B straight through.
        for (int r = 0; r < BK; r += strideB)
            reinterpret_cast<float4*>(&Bs[(loadRowB + r) * BN + loadColB * 4])[0] =
                reinterpret_cast<const float4*>(&B[(loadRowB + r) * N + loadColB * 4])[0];
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            // Pull this thread's A fragment (across all M-subtiles) into registers.
            for (int wm = 0; wm < WMITER; ++wm)
                for (int i = 0; i < TM; ++i)
                    regM[wm * TM + i] =
                        As[k * BM + warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i];
            // ...and its B fragment (across all N-subtiles).
            for (int wn = 0; wn < WNITER; ++wn)
                for (int j = 0; j < TN; ++j)
                    regN[wn * TN + j] =
                        Bs[k * BN + warpCol * WN + wn * WSUBN + threadColInWarp * TN + j];
            // Outer products over every (M-subtile, N-subtile) pair.
            for (int wm = 0; wm < WMITER; ++wm)
                for (int wn = 0; wn < WNITER; ++wn)
                    for (int i = 0; i < TM; ++i)
                        for (int j = 0; j < TN; ++j)
                            acc[wm * TM + i][wn * TN + j] +=
                                regM[wm * TM + i] * regN[wn * TN + j];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // ---- GEMM epilogue: C = alpha*acc + beta*C, one subtile at a time ----
    for (int wm = 0; wm < WMITER; ++wm)
        for (int wn = 0; wn < WNITER; ++wn) {
            float* Csub = C + (wm * WSUBM) * N + wn * WSUBN;
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; j += 4) {
                    float* cptr = &Csub[(threadRowInWarp * TM + i) * N + threadColInWarp * TN + j];
                    float4 old = reinterpret_cast<float4*>(cptr)[0];
                    float4 v;
                    v.x = alpha * acc[wm * TM + i][wn * TN + j + 0] + beta * old.x;
                    v.y = alpha * acc[wm * TM + i][wn * TN + j + 1] + beta * old.y;
                    v.z = alpha * acc[wm * TM + i][wn * TN + j + 2] + beta * old.z;
                    v.w = alpha * acc[wm * TM + i][wn * TN + j + 3] + beta * old.w;
                    reinterpret_cast<float4*>(cptr)[0] = v;
                }
        }
}

inline void run_warptiling(int M, int N, int K, float alpha,
                           const float* A, const float* B, float beta, float* C) {
    // Boehm's A6000 default configuration.
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64,  WNITER = 4;
    constexpr int TM = 8,   TN = 4;
    constexpr int NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    warptiling_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
