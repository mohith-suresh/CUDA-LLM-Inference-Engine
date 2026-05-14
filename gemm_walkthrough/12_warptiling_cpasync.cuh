// Step 12 — Warptiling + REAL cp.async double buffering (Ampere sm_80+)
// ----------------------------------------------------------------------------
// The compute is byte-for-byte the step-10 warptiling kernel. The ONLY change is
// how the two shared buffers are filled: instead of ordinary float4 loads
// (global -> register -> shared), this uses genuine asynchronous copies
// (global -> shared directly, no registers, non-blocking) via cp.async, exposed
// through the __pipeline_memcpy_async / __pipeline_commit / __pipeline_wait_prior
// primitives in <cuda_pipeline.h>.
//
//   Stage k+1 is cp.async-copied into the other buffer while the SM computes on
//   stage k. wait_prior(1) keeps exactly one copy in flight (the prefetch).
//
// Note on A: warptiling stores As TRANSPOSED, and cp.async cannot scatter, so the
// A side issues four 4-byte cp.async's per float4 (one per transposed element).
// That is the honest cost of the transpose — production kernels avoid it with
// ldmatrix or a pre-transposed A. B (no transpose) uses one 16-byte cp.async.
#pragma once
#include <cuda_pipeline.h>
#include "common.cuh"

template <int BM, int BN, int BK, int NUM_THREADS>
__device__ __forceinline__ void cpasync_load(
    float* As_buf, float* Bs_buf, const float* Aptr, const float* Bptr, int K, int N,
    int loadRowA, int loadColA, int loadRowB, int loadColB) {
    constexpr int strideA = (NUM_THREADS * 4) / BK;
    constexpr int strideB = NUM_THREADS / (BN / 4);
    // A: transposed -> four 4-byte async copies per float4.
    #pragma unroll
    for (int r = 0; r < BM; r += strideA) {
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            __pipeline_memcpy_async(
                &As_buf[(loadColA * 4 + e) * BM + loadRowA + r],
                &Aptr[(loadRowA + r) * K + loadColA * 4 + e],
                sizeof(float));
    }
    // B: contiguous -> one 16-byte async copy.
    #pragma unroll
    for (int r = 0; r < BK; r += strideB)
        __pipeline_memcpy_async(
            &Bs_buf[(loadRowB + r) * BN + loadColB * 4],
            &Bptr[(loadRowB + r) * N + loadColB * 4],
            sizeof(float4));
    __pipeline_commit();
}

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NUM_THREADS, int MINB = 1>
__global__ void __launch_bounds__(NUM_THREADS, MINB)
warptiling_cpasync_kernel(int M, int N, int K, float alpha,
                          const float* A, const float* B, float beta, float* C) {
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM  = WM / WMITER;
    constexpr int WSUBN  = WN / WNITER;

    __shared__ float As[2][BK * BM];      // double-buffered, transposed
    __shared__ float Bs[2][BK * BN];      // double-buffered

    const int warpIdx = threadIdx.x / 32;
    const int warpRow = warpIdx / (BN / WN);
    const int warpCol = warpIdx % (BN / WN);
    const int lane            = threadIdx.x % 32;
    const int threadRowInWarp = lane / (WSUBN / TN);
    const int threadColInWarp = lane % (WSUBN / TN);

    const int blockRow = blockIdx.y, blockCol = blockIdx.x;
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += (blockRow * BM + warpRow * WM) * N + blockCol * BN + warpCol * WN;

    const int loadRowA = threadIdx.x / (BK / 4), loadColA = threadIdx.x % (BK / 4);
    const int loadRowB = threadIdx.x / (BN / 4), loadColB = threadIdx.x % (BN / 4);

    float acc[WMITER * TM][WNITER * TN] = {0.0f};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    const int numTiles = K / BK;

    // Prologue: kick off the async copy of tile 0 into buffer 0.
    cpasync_load<BM, BN, BK, NUM_THREADS>(As[0], Bs[0], A, B, K, N,
                                          loadRowA, loadColA, loadRowB, loadColB);

    for (int t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        // Prefetch tile t+1 into the other buffer while we compute tile t.
        if (t + 1 < numTiles)
            cpasync_load<BM, BN, BK, NUM_THREADS>(
                As[(t + 1) & 1], Bs[(t + 1) & 1], A + (t + 1) * BK, B + (t + 1) * BK * N, K, N,
                loadRowA, loadColA, loadRowB, loadColB);

        // Wait until tile `cur` has landed (leave the prefetch, if any, in flight).
        __pipeline_wait_prior((t + 1 < numTiles) ? 1 : 0);
        __syncthreads();

        // ---- warptiling compute on buffer `cur` (identical to step 10) ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int wm = 0; wm < WMITER; ++wm)
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    regM[wm * TM + i] =
                        As[cur][k * BM + warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i];
            #pragma unroll
            for (int wn = 0; wn < WNITER; ++wn)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    regN[wn * TN + j] =
                        Bs[cur][k * BN + warpCol * WN + wn * WSUBN + threadColInWarp * TN + j];
            #pragma unroll
            for (int wm = 0; wm < WMITER; ++wm)
                #pragma unroll
                for (int wn = 0; wn < WNITER; ++wn)
                    #pragma unroll
                    for (int i = 0; i < TM; ++i)
                        #pragma unroll
                        for (int j = 0; j < TN; ++j)
                            acc[wm * TM + i][wn * TN + j] += regM[wm * TM + i] * regN[wn * TN + j];
        }
        __syncthreads();   // done reading `cur`; safe for the next-but-one prefetch to overwrite it
    }

    // ---- GEMM epilogue (identical to step 10) ----
    #pragma unroll
    for (int wm = 0; wm < WMITER; ++wm)
        #pragma unroll
        for (int wn = 0; wn < WNITER; ++wn) {
            float* Csub = C + (wm * WSUBM) * N + wn * WSUBN;
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
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

inline void run_warptiling_cpasync(int M, int N, int K, float alpha,
                                   const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64, WNITER = 4;
    constexpr int TM = 8, TN = 4;
    constexpr int NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    warptiling_cpasync_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

// Same cp.async kernel, but reg-capped for 3 blocks/SM (matched occupancy to
// plain warptiling) — isolates cp.async's true effect from the register cliff.
inline void run_warptiling_cpasync_occ(int M, int N, int K, float alpha,
                                       const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 16, WM = 64, WN = 64, WNITER = 4, TM = 8, TN = 4, NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    warptiling_cpasync_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, /*MINB=*/3>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
