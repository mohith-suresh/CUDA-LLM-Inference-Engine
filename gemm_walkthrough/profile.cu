// profile.cu — launches each kernel EXACTLY ONCE (no timing loop, no cuBLAS).
// This gives Nsight Compute a clean, one-launch-per-kernel target.
//
//   ./profile          # default size 2048
//   ./profile 1024     # smaller = faster ncu replay
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

int main(int argc, char** argv) {
    int size = (argc > 1) ? atoi(argv[1]) : 2048;
    int M = size, N = size, K = size;
    size_t bytes = (size_t)M * N * sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    float* host = (float*)malloc(bytes);
    fill_random(host, M * K, 1); CUDA_CHECK(cudaMemcpy(dA, host, bytes, cudaMemcpyHostToDevice));
    fill_random(host, K * N, 2); CUDA_CHECK(cudaMemcpy(dB, host, bytes, cudaMemcpyHostToDevice));
    free(host);

    const float alpha = 1.0f, beta = 0.0f;

    // One launch each — kernel names below are what you filter on in ncu.
    run_naive          (M, N, K, alpha, dA, dB, beta, dC);
    run_coalesced      (M, N, K, alpha, dA, dB, beta, dC);
    run_shared         (M, N, K, alpha, dA, dB, beta, dC);
    run_blocktiling_1d (M, N, K, alpha, dA, dB, beta, dC);
    run_blocktiling_2d (M, N, K, alpha, dA, dB, beta, dC);
    run_blocktiling_2d_transA(M, N, K, alpha, dA, dB, beta, dC);       // 5b: As-transpose isolation
    run_blocktiling_2d_noconflict(M, N, K, alpha, dA, dB, beta, dC);   // 5c: As+Bs conflict-free
    run_vectorized     (M, N, K, alpha, dA, dB, beta, dC);
    run_bank_conflicts (M, N, K, alpha, dA, dB, beta, dC);
    run_bank_extra_col (M, N, K, alpha, dA, dB, beta, dC);
    run_warptiling     (M, N, K, alpha, dA, dB, beta, dC);
    run_flat_register_tiling(M, N, K, alpha, dA, dB, beta, dC);   // 10b ablation vs warptiling
    run_flat_hi_occ    (M, N, K, alpha, dA, dB, beta, dC);        // 10c ablation vs warptiling
    run_flat_interleaved(M, N, K, alpha, dA, dB, beta, dC);       // 10d: coalesced-store isolation
    run_flat_16x4      (M, N, K, alpha, dA, dB, beta, dC);        // 10e: 16x4 = 1 float4 wide
    run_roundrobin_4x4 (M, N, K, alpha, dA, dB, beta, dC);        // 10f: 4x4 round-robin micro-tiles
    run_roundrobin_8x4 (M, N, K, alpha, dA, dB, beta, dC);        // 10f2: 8x4 round-robin micro-tiles
    run_split2_roundrobin(M, N, K, alpha, dA, dB, beta, dC);      // 10h: column-split round-robin
    run_split2_roundrobin_occ(M, N, K, alpha, dA, dB, beta, dC);  // 10h+: reg-capped for 3 blocks/SM
    run_flat_32x4      (M, N, K, alpha, dA, dB, beta, dC);        // 10g: 32x4, 128t/128out vs warptile
    run_double_buffered(M, N, K, alpha, dA, dB, beta, dC);
    run_warptiling_cpasync(M, N, K, alpha, dA, dB, beta, dC);   // 12: cp.async DB on warptiling

    // cuBLAS reference — one launch so ncu captures the ampere_sgemm kernel too.
    { CublasRef cublas; cublas.gemm(M, N, K, alpha, dA, dB, beta, dC); }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
