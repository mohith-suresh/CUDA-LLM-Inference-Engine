// main.cu — run every step, validate against cuBLAS, report GFLOP/s.
//
//   ./gemm            # default sizes 1024, 2048, 4096
//   ./gemm 2048       # single square size
//
// GEMM tested here is  C = alpha*(A@B) + beta*C.  Correctness is checked with
// alpha=1, beta=0.5 and a random initial C (so the beta path is exercised, not
// bypassed). Timing uses the standard alpha=1, beta=0 configuration.
#include "common.cuh"
#include "01_naive.cuh"
#include "02_coalesced.cuh"
#include "03_shared_memory.cuh"
#include "04_blocktiling_1d.cuh"
#include "05_blocktiling_2d.cuh"
#include "05b_blocktiling_2d_transposed_As.cuh"
#include "05c_blocktiling_2d_noconflict.cuh"
#include "06_vectorized.cuh"
#include "07_bank_conflicts.cuh"
#include "08_bank_extra_col.cuh"
#include "10_warptiling.cuh"
#include "10b_flat_register_tiling.cuh"
#include "10d_flat_coalesced_store.cuh"
#include "10f_microtile_roundrobin.cuh"
#include "10h_split_roundrobin.cuh"
#include "11_double_buffered.cuh"
#include "12_warptiling_cpasync.cuh"

struct Step { const char* name; GemmLauncher launch; };

int main(int argc, char** argv) {
    const Step steps[] = {
        {"1  Naive",            run_naive},
        {"2  Coalesced",        run_coalesced},
        {"3  Shared Memory",    run_shared},
        {"4  1D Blocktiling",   run_blocktiling_1d},
        {"5  2D Blocktiling",   run_blocktiling_2d},
        {"5b 2D Transposed As", run_blocktiling_2d_transA},     // isolation: As transpose only, no float4
        {"5c 2D No-Conflict",   run_blocktiling_2d_noconflict}, // isolation: As+Bs conflict-free, no float4
        {"6  Vectorized",       run_vectorized},
        {"7  Bank Conflicts",   run_bank_conflicts},
        {"8  Bank Extra Col",   run_bank_extra_col},
        {"10 Warptiling",       run_warptiling},
        {"10b Flat 128t/8x16",  run_flat_register_tiling},   // ablation: same as warptile, no warp layout
        {"10c Flat 256t/8x8",   run_flat_hi_occ},            // ablation: no warp layout, higher occupancy
        {"10d Flat interleaved",run_flat_interleaved},       // 10b + coalesced C store (interleaved cols)
        {"10e Flat 16x4 (1 f4)", run_flat_16x4},             // 16x4=16x1 float4/thread -> native coalescing
        {"10f 4x4 round-robin",  run_roundrobin_4x4},        // 8 scattered 4x4 tiles/thread, cyclic
        {"10f2 8x4 round-robin", run_roundrobin_8x4},        // 4 scattered 8x4 tiles/thread, cyclic
        {"10h split-2 round-rob",run_split2_roundrobin},     // 2 col-halves -> effective 16x8 tile
        {"10h+ split-2 3blk/SM", run_split2_roundrobin_occ}, // same, reg-capped for 3 blocks/SM
        {"10g Flat 32x4 (1 f4)", run_flat_32x4},             // 128t/128out contiguous — matches warptile budget
        {"10g+ 32x4 3blk/SM",    run_flat_32x4_occ},         // same, reg-capped for 3 blocks/SM
        {"11 Double Buffered",  run_double_buffered},
        {"12 Warptile cp.async", run_warptiling_cpasync},
        {"12+ cp.async 3blk/SM", run_warptiling_cpasync_occ},
    };
    const int numSteps = sizeof(steps) / sizeof(steps[0]);

    int sizes[] = {1024, 2048, 4096};
    int numSizes = 3;
    if (argc > 1) { sizes[0] = atoi(argv[1]); numSizes = 1; }

    CublasRef cublas;

    printf("GEMM walkthrough — clean re-implementation of the CUDA-MMM ladder\n");
    printf("Row-major FP32,  C = alpha*(A@B) + beta*C,  validated vs cuBLAS\n");
    printf("(step 9 = autotuning lives in ./autotune)\n");
    printf("=================================================================\n\n");

    for (int s = 0; s < numSizes; ++s) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        size_t bytesA = (size_t)M * K * sizeof(float);
        size_t bytesB = (size_t)K * N * sizeof(float);
        size_t bytesC = (size_t)M * N * sizeof(float);

        float *dA, *dB, *dC, *dRef, *dCinit;
        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));
        CUDA_CHECK(cudaMalloc(&dRef, bytesC));
        CUDA_CHECK(cudaMalloc(&dCinit, bytesC));

        size_t maxAB = bytesA > bytesB ? bytesA : bytesB;
        float* host = (float*)malloc(maxAB > bytesC ? maxAB : bytesC);
        fill_random(host, M * K, 1); CUDA_CHECK(cudaMemcpy(dA, host, bytesA, cudaMemcpyHostToDevice));
        fill_random(host, K * N, 2); CUDA_CHECK(cudaMemcpy(dB, host, bytesB, cudaMemcpyHostToDevice));
        fill_random(host, M * N, 3); CUDA_CHECK(cudaMemcpy(dCinit, host, bytesC, cudaMemcpyHostToDevice));
        free(host);

        // Reference GEMM with the SAME initial C and alpha/beta as the validation.
        const float A_ = 1.0f, B_ = 0.5f;      // alpha, beta for the correctness check
        CUDA_CHECK(cudaMemcpy(dRef, dCinit, bytesC, cudaMemcpyDeviceToDevice));
        cublas.gemm(M, N, K, A_, dA, dB, B_, dRef);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Size %d x %d x %d\n", M, N, K);
        printf("%-20s %12s %10s %10s %8s\n", "Step", "GFLOP/s", "vs cuBLAS", "rel_err", "status");
        printf("-----------------------------------------------------------------\n");

        // cuBLAS timing (alpha=1, beta=0) as the 100% reference.
        float cublasMs = time_kernel(
            [](int M, int N, int K, float al, const float* A, const float* B, float be, float* C) {
                static CublasRef ref; ref.gemm(M, N, K, al, A, B, be, C);
            }, M, N, K, 1.0f, dA, dB, 0.0f, dC);
        float cublasGf = gflops(M, N, K, cublasMs);

        for (int i = 0; i < numSteps; ++i) {
            // Correctness: restore the random initial C, run GEMM with beta=0.5.
            CUDA_CHECK(cudaMemcpy(dC, dCinit, bytesC, cudaMemcpyDeviceToDevice));
            steps[i].launch(M, N, K, A_, dA, dB, B_, dC);
            CUDA_CHECK(cudaDeviceSynchronize());
            float relErr = max_rel_error(dC, dRef, M * N);

            // Timing: standard alpha=1, beta=0.
            float ms = time_kernel(steps[i].launch, M, N, K, 1.0f, dA, dB, 0.0f, dC);
            float gf = gflops(M, N, K, ms);

            printf("%-20s %12.1f %9.1f%% %10.1e %8s\n",
                   steps[i].name, gf, gf / cublasGf * 100.0f, relErr,
                   relErr < 1e-2f ? "PASS" : "FAIL");
        }
        printf("%-20s %12.1f %9s %12s\n\n", "0  cuBLAS", cublasGf, "100.0%", "REF");

        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef); cudaFree(dCinit);
    }
    return 0;
}
