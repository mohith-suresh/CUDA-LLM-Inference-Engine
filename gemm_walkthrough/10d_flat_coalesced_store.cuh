// Step 10d — Flat register tiling with INTERLEAVED columns (coalesced C stores)
// ----------------------------------------------------------------------------
// ISOLATION EXPERIMENT for global-store coalescing. Identical to
// 10b_flat_register_tiling in block size (128x128), K tile (16), thread count
// (128), and per-thread output count (8x16 = 128). The ONLY change is how a
// thread's 16 output columns are placed in the block tile.
//
//   10b (contiguous):  thread owns columns [tc*TTN .. tc*TTN+TTN)
//       -> consecutive threads write columns TTN=16 apart
//       -> each float4 store lands alone in a 32-byte sector -> 16/32 bytes used
//
//   10d (interleaved): thread owns float4-chunks { tc, tc+TC, tc+2*TC, tc+3*TC }
//       where TC = THREAD_TILE_COLS. Consecutive threads write CONSECUTIVE
//       float4s -> 8 threads pack 128 contiguous bytes -> 32/32 bytes used.
//
// So the C store goes from ~16/32 (uncoalesced) to 32/32 (coalesced). As a
// bonus (same reason as 5c), the strided Bs read also drops from a 4-way bank
// conflict to conflict-free — the interleave fixes read AND write at once.
//
// Note: store coalescing is a MINOR lever — the epilogue writes each C element
// once (M*N stores) vs. the K-loop's many A/B loads. Expect the ncu "Global
// Store Access Pattern" rule to disappear and store bytes/sector -> 32, but only
// a small runtime change.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int TTM, int TTN, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
flat_interleaved_kernel(int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C) {
    static_assert(BM % TTM == 0 && BN % TTN == 0, "tiles must divide block");
    static_assert(BK % 4 == 0 && BN % 4 == 0 && TTN % 4 == 0, "float4 alignment");

    constexpr int THREAD_TILE_ROWS = BM / TTM;
    constexpr int THREAD_TILE_COLS = BN / TTN;     // = TC, the interleave stride (in chunks)
    constexpr int NCHUNK           = TTN / 4;       // float4 chunks per thread
    static_assert(THREAD_TILE_ROWS * THREAD_TILE_COLS == NUM_THREADS, "exact cover");

    __shared__ float As[BK * BM];      // transposed As[k][m] (same as 10b)
    __shared__ float Bs[BK * BN];

    const int threadTileRow = threadIdx.x / THREAD_TILE_COLS;
    const int threadTileCol = threadIdx.x % THREAD_TILE_COLS;
    const int threadRow0    = threadTileRow * TTM;     // rows: contiguous (unchanged)

    const int blockRow0 = blockIdx.y * BM;
    const int blockCol0 = blockIdx.x * BN;
    A += blockRow0 * K;
    B += blockCol0;
    C += blockRow0 * N + blockCol0;

    // Column of register/acc index j under the INTERLEAVED mapping.
    //   j = s*4 + jj  ->  chunk (threadTileCol + s*TC), lane jj
    auto colOf = [&](int j) {
        int s = j / 4, jj = j % 4;
        return (threadTileCol + s * THREAD_TILE_COLS) * 4 + jj;
    };

    float acc[TTM * TTN] = {0.0f};
    float regA[TTM];
    float regB[TTN];

    constexpr int A_VECTOR_COLS  = BK / 4, B_VECTOR_COLS = BN / 4;
    constexpr int A_VECTOR_COUNT = BM * A_VECTOR_COLS, B_VECTOR_COUNT = BK * B_VECTOR_COLS;

    for (int kTile = 0; kTile < K; kTile += BK) {
        // Cooperative A load (transposed) — unchanged from 10b.
        for (int vi = threadIdx.x; vi < A_VECTOR_COUNT; vi += NUM_THREADS) {
            const int row = vi / A_VECTOR_COLS, col4 = vi % A_VECTOR_COLS;
            const float4 v = *reinterpret_cast<const float4*>(&A[row * K + col4 * 4]);
            As[(col4 * 4 + 0) * BM + row] = v.x;
            As[(col4 * 4 + 1) * BM + row] = v.y;
            As[(col4 * 4 + 2) * BM + row] = v.z;
            As[(col4 * 4 + 3) * BM + row] = v.w;
        }
        // Cooperative B load — unchanged from 10b.
        for (int vi = threadIdx.x; vi < B_VECTOR_COUNT; vi += NUM_THREADS) {
            const int row = vi / B_VECTOR_COLS, col4 = vi % B_VECTOR_COLS;
            *reinterpret_cast<float4*>(&Bs[row * BN + col4 * 4]) =
                *reinterpret_cast<const float4*>(&B[row * N + col4 * 4]);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TTM; ++i)
                regA[i] = As[k * BM + threadRow0 + i];
            // Bs read uses the INTERLEAVED columns (stride TC*4 -> conflict-free).
            #pragma unroll
            for (int j = 0; j < TTN; ++j)
                regB[j] = Bs[k * BN + colOf(j)];
            #pragma unroll
            for (int i = 0; i < TTM; ++i)
                #pragma unroll
                for (int j = 0; j < TTN; ++j)
                    acc[i * TTN + j] += regA[i] * regB[j];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Epilogue: one float4 store per (row i, chunk s). Consecutive threads ->
    // consecutive float4 columns -> 8 threads pack 128 contiguous bytes = coalesced.
    #pragma unroll
    for (int i = 0; i < TTM; ++i) {
        #pragma unroll
        for (int s = 0; s < NCHUNK; ++s) {
            const int col0 = (threadTileCol + s * THREAD_TILE_COLS) * 4;
            float* cptr = &C[(threadRow0 + i) * N + col0];
            const float4 old = *reinterpret_cast<const float4*>(cptr);
            const int j = s * 4;
            float4 v;
            v.x = alpha * acc[i * TTN + j + 0] + beta * old.x;
            v.y = alpha * acc[i * TTN + j + 1] + beta * old.y;
            v.z = alpha * acc[i * TTN + j + 2] + beta * old.z;
            v.w = alpha * acc[i * TTN + j + 3] + beta * old.w;
            *reinterpret_cast<float4*>(cptr) = v;
        }
    }
}

inline void run_flat_interleaved(int M, int N, int K, float alpha,
                                 const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TTM = 8, TTN = 16;               // same 8x16 = 128 outputs/thread as 10b
    constexpr int NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    flat_interleaved_kernel<BM, BN, BK, TTM, TTN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
