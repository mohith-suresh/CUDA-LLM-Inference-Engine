// Step 10b — Flat register tiling (ablation / control for warp tiling)
// ----------------------------------------------------------------------------
// Control experiment for warp tiling (step 10).
//
// Same work/resources as the warp-tiled kernel:
//   Block tile:          128 x 128
//   K tile:              16
//   Threads/block:       128
//   Thread register tile: 8 x 16 = 128 outputs
//   A registers per k:     8
//   B registers per k:    16
//   FMAs per thread/k:   128
//
// Difference:
//   There is NO explicit 64 x 64 warp tile.
//   Threads directly divide the complete 128 x 128 block tile.
//
// Thread-tile grid:
//   (128 / 8) x (128 / 16)
//   = 16 x 8
//   = 128 threads.
//
// CUDA still physically executes threads as warps, but this kernel does not
// explicitly assign a compact output region to each warp. Comparing this to
// step 10 isolates whether the warptiling speedup comes from the warp-level
// layout or simply from giving each thread a larger (128-output) register tile.
// ----------------------------------------------------------------------------

#pragma once
#include "common.cuh"

template <
    int BM,
    int BN,
    int BK,
    int TTM,          // thread-tile height
    int TTN,          // thread-tile width
    int NUM_THREADS,
    int MINB = 1>     // min blocks/SM hint (caps registers)
__global__ void __launch_bounds__(NUM_THREADS, MINB)
flat_register_tiling_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C)
{
    static_assert(BM % TTM == 0, "BM must be divisible by TTM");
    static_assert(BN % TTN == 0, "BN must be divisible by TTN");

    static_assert(BK % 4 == 0,
                  "BK must be divisible by 4 for float4 loads");

    static_assert(BN % 4 == 0,
                  "BN must be divisible by 4 for float4 loads");

    static_assert(TTN % 4 == 0,
                  "TTN must be divisible by 4 for float4 stores");

    constexpr int THREAD_TILE_ROWS = BM / TTM;
    constexpr int THREAD_TILE_COLS = BN / TTN;

    static_assert(
        THREAD_TILE_ROWS * THREAD_TILE_COLS == NUM_THREADS,
        "Thread tiles must exactly cover the block tile");

    // A is transposed while entering shared memory:
    //
    // Original logical layout:   A_shared[m][k]
    // Stored layout:             As[k][m]
    //
    // This lets each thread read its TTM A values contiguously.
    __shared__ float As[BK * BM];

    // B remains in normal [k][n] order.
    __shared__ float Bs[BK * BN];

    // -------------------------------------------------------------------------
    // Direct thread mapping over the entire block tile.
    //
    // For BM=128, BN=128, TTM=8, TTN=16:
    //
    //   THREAD_TILE_ROWS = 16
    //   THREAD_TILE_COLS = 8
    //
    //   tid 0  -> tile row 0, tile col 0
    //   tid 1  -> tile row 0, tile col 1
    //   ...
    //   tid 7  -> tile row 0, tile col 7
    //   tid 8  -> tile row 1, tile col 0
    // -------------------------------------------------------------------------

    const int threadTileRow = threadIdx.x / THREAD_TILE_COLS;
    const int threadTileCol = threadIdx.x % THREAD_TILE_COLS;

    const int threadRow0 = threadTileRow * TTM;
    const int threadCol0 = threadTileCol * TTN;

    // Move pointers to this block's tile origin.
    const int blockRow0 = blockIdx.y * BM;
    const int blockCol0 = blockIdx.x * BN;

    A += blockRow0 * K;
    B += blockCol0;
    C += blockRow0 * N + blockCol0;

    // 8 x 16 = 128 accumulators, matching the warp-tiled kernel.
    float acc[TTM * TTN] = {0.0f};

    // Per-k operands.
    float regA[TTM];
    float regB[TTN];

    // Number of float4 chunks in one shared-memory row.
    constexpr int A_VECTOR_COLS = BK / 4;
    constexpr int B_VECTOR_COLS = BN / 4;

    constexpr int A_VECTOR_COUNT = BM * A_VECTOR_COLS;
    constexpr int B_VECTOR_COUNT = BK * B_VECTOR_COLS;

    for (int kTile = 0; kTile < K; kTile += BK) {

        // ---------------------------------------------------------------------
        // Cooperative A loading
        //
        // Universal flattened-vector pattern:
        //
        //   vectorIndex = tid, tid + NUM_THREADS, ...
        //
        // Each vector contains four adjacent A values along K.
        // They are scattered into transposed shared-memory layout As[k][m].
        // ---------------------------------------------------------------------

        for (int vectorIndex = threadIdx.x;
             vectorIndex < A_VECTOR_COUNT;
             vectorIndex += NUM_THREADS)
        {
            const int row  = vectorIndex / A_VECTOR_COLS;
            const int col4 = vectorIndex % A_VECTOR_COLS;

            const float4 value =
                *reinterpret_cast<const float4*>(
                    &A[row * K + col4 * 4]);

            As[(col4 * 4 + 0) * BM + row] = value.x;
            As[(col4 * 4 + 1) * BM + row] = value.y;
            As[(col4 * 4 + 2) * BM + row] = value.z;
            As[(col4 * 4 + 3) * BM + row] = value.w;
        }

        // ---------------------------------------------------------------------
        // Cooperative B loading
        //
        // B is already in the [k][n] layout required by computation, so each
        // float4 can be copied directly into shared memory.
        // ---------------------------------------------------------------------

        for (int vectorIndex = threadIdx.x;
             vectorIndex < B_VECTOR_COUNT;
             vectorIndex += NUM_THREADS)
        {
            const int row  = vectorIndex / B_VECTOR_COLS;
            const int col4 = vectorIndex % B_VECTOR_COLS;

            const float4 value =
                *reinterpret_cast<const float4*>(
                    &B[row * N + col4 * 4]);

            *reinterpret_cast<float4*>(
                &Bs[row * BN + col4 * 4]) = value;
        }

        __syncthreads();

        // ---------------------------------------------------------------------
        // Register-tiled computation
        //
        // For every k:
        //
        //   8 A shared loads
        //   16 B shared loads
        //          v
        //   8 x 16 = 128 FMAs
        // ---------------------------------------------------------------------

        #pragma unroll
        for (int k = 0; k < BK; ++k) {

            // Load this thread's 8 A values.
            #pragma unroll
            for (int i = 0; i < TTM; ++i) {
                regA[i] =
                    As[k * BM + threadRow0 + i];
            }

            // Load this thread's contiguous 16 B values.
            #pragma unroll
            for (int j = 0; j < TTN; ++j) {
                regB[j] =
                    Bs[k * BN + threadCol0 + j];
            }

            // 8 x 16 outer product.
            #pragma unroll
            for (int i = 0; i < TTM; ++i) {
                #pragma unroll
                for (int j = 0; j < TTN; ++j) {
                    acc[i * TTN + j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();

        // Advance to the next K tile.
        A += BK;
        B += BK * N;
    }

    // -------------------------------------------------------------------------
    // Vectorized epilogue
    //
    // Each thread writes its contiguous 8 x 16 output tile.
    // Four adjacent C values are loaded/stored with float4.
    // -------------------------------------------------------------------------

    #pragma unroll
    for (int i = 0; i < TTM; ++i) {
        #pragma unroll
        for (int j = 0; j < TTN; j += 4) {

            float* cptr =
                &C[(threadRow0 + i) * N + threadCol0 + j];

            const float4 old =
                *reinterpret_cast<const float4*>(cptr);

            float4 value;

            value.x =
                alpha * acc[i * TTN + j + 0]
                + beta * old.x;

            value.y =
                alpha * acc[i * TTN + j + 1]
                + beta * old.y;

            value.z =
                alpha * acc[i * TTN + j + 2]
                + beta * old.z;

            value.w =
                alpha * acc[i * TTN + j + 3]
                + beta * old.w;

            *reinterpret_cast<float4*>(cptr) = value;
        }
    }
}

inline void run_flat_register_tiling(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;

    // One contiguous 8 x 16 register tile per thread.
    constexpr int TTM = 8;
    constexpr int TTN = 16;

    constexpr int NUM_THREADS = 128;

    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);

    flat_register_tiling_kernel<
        BM,
        BN,
        BK,
        TTM,
        TTN,
        NUM_THREADS>
        <<<grid, block>>>(
            M,
            N,
            K,
            alpha,
            A,
            B,
            beta,
            C);
}

// 16 x 4 outputs per thread  ==  16 x 1 in float4 (one float4 per row, 16 rows).
// Each thread handles ONE tile of 16 rows x 1 float4. Because the tile is exactly
// ONE float4 wide (TTN=4), consecutive threads own consecutive float4 columns, so
// the C stores coalesce NATIVELY — no interleave (10d) needed. 256 threads =
// (128/16) row-groups x (128/4) col-groups = 8 x 32.
// Trade-off: the 4-wide Bs read reintroduces a shared-load bank conflict that
// 10d's wider interleaved tile avoids. Isolates "1-float4-wide => free coalescing".
inline void run_flat_16x4(int M, int N, int K, float alpha,
                          const float* A, const float* B, float beta, float* C)
{
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TTM = 16, TTN = 4;        // 16 x 4 = 64 outputs/thread == 16 x 1 float4
    constexpr int NUM_THREADS = 256;        // (128/16) * (128/4) = 8 * 32
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    flat_register_tiling_kernel<BM, BN, BK, TTM, TTN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

// 32 x 4 outputs per thread == 32 x 1 in float4 (one float4 per row, 32 rows).
// Apples-to-apples with WARPTILING: same 128 threads/block and same 128
// outputs/thread — the ONLY difference is the layout. Here each thread owns ONE
// contiguous 32x4 tile (1 float4 wide -> native store coalescing); warptiling
// spreads the same 128 outputs into 4 chunks across a 64x64 warp tile.
// (128/32) row-groups x (128/4) col-groups = 4 x 32 = 128 threads.
inline void run_flat_32x4(int M, int N, int K, float alpha,
                          const float* A, const float* B, float beta, float* C)
{
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TTM = 32, TTN = 4;        // 32 x 4 = 128 outputs/thread == 32 x 1 float4
    constexpr int NUM_THREADS = 128;        // (128/32) * (128/4) = 4 * 32
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    flat_register_tiling_kernel<BM, BN, BK, TTM, TTN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

// Same 32x4 design, but ask for 3 blocks/SM (caps registers) to lift occupancy.
inline void run_flat_32x4_occ(int M, int N, int K, float alpha,
                              const float* A, const float* B, float beta, float* C)
{
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TTM = 32, TTN = 4;
    constexpr int NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    flat_register_tiling_kernel<BM, BN, BK, TTM, TTN, NUM_THREADS, /*MINB=*/3>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

// Second control: same block tile (128x128, BK=16) and NO warp tiling, but a
// HIGHER-OCCUPANCY configuration — 256 threads each computing an 8x8 (64-output)
// register tile instead of 128 threads x 128 outputs. Tests whether warptiling
// is a universal win or mainly compensates for the low-occupancy 128-output regime.
inline void run_flat_hi_occ(
    int M, int N, int K, float alpha,
    const float* A, const float* B, float beta, float* C)
{
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TTM = 8, TTN = 8;         // 64 outputs / thread
    constexpr int NUM_THREADS = 256;        // (128/8) * (128/8) = 256
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    flat_register_tiling_kernel<BM, BN, BK, TTM, TTN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
