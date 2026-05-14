// Step 9 — Autotuning
// ----------------------------------------------------------------------------
// Not a new kernel: the "autotuned" step is just the 2D-blocktiling kernel run
// with many different tile configs to find the best one for THIS GPU. Template
// params are compile-time, so we instantiate a fixed menu of configs and time
// each, then report the winner.
//
//   ./autotune           # default size 4096
//   ./autotune 2048
#include "common.cuh"
#include "05_blocktiling_2d.cuh"

// One launcher per config — a plain function matching GemmLauncher.
template <int BM, int BN, int BK, int TM, int TN>
void launch2d(int M, int N, int K, float alpha,
              const float* A, const float* B, float beta, float* C) {
    dim3 block((BM / TM) * (BN / TN));
    dim3 grid(N / BN, M / BM);
    blocktiling_2d_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

struct Config { const char* name; GemmLauncher launch; };

int main(int argc, char** argv) {
    int size = (argc > 1) ? atoi(argv[1]) : 4096;
    int M = size, N = size, K = size;

    // A curated menu of valid configs (each satisfies the tile/thread divisibility
    // the 2D kernel assumes). Add your own rows to widen the search.
    const Config configs[] = {
        {"BM64  BN64  BK8  TM4 TN4", launch2d<64,  64,  8,  4, 4>},
        {"BM64  BN64  BK16 TM4 TN4", launch2d<64,  64,  16, 4, 4>},
        {"BM128 BN64  BK8  TM8 TN4", launch2d<128, 64,  8,  8, 4>},
        {"BM64  BN128 BK8  TM4 TN8", launch2d<64,  128, 8,  4, 8>},
        {"BM128 BN128 BK8  TM8 TN8", launch2d<128, 128, 8,  8, 8>},
        {"BM128 BN128 BK16 TM8 TN8", launch2d<128, 128, 16, 8, 8>},
        {"BM128 BN128 BK8  TM8 TN4", launch2d<128, 128, 8,  8, 4>},
        {"BM128 BN128 BK16 TM4 TN8", launch2d<128, 128, 16, 4, 8>},
    };
    const int n = sizeof(configs) / sizeof(configs[0]);

    float *dA, *dB, *dC, *dRef;
    size_t bytes = (size_t)M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMalloc(&dRef, bytes));

    float* host = (float*)malloc(bytes);
    fill_random(host, M * K, 1); CUDA_CHECK(cudaMemcpy(dA, host, bytes, cudaMemcpyHostToDevice));
    fill_random(host, K * N, 2); CUDA_CHECK(cudaMemcpy(dB, host, bytes, cudaMemcpyHostToDevice));
    free(host);

    CublasRef cublas;
    cublas.gemm(M, N, K, 1.0f, dA, dB, 0.0f, dRef);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Autotuning 2D blocktiling at %d x %d x %d\n", M, N, K);
    printf("%-28s %12s %10s %8s\n", "Config", "GFLOP/s", "status", "");
    printf("---------------------------------------------------------\n");

    const char* bestName = nullptr;
    float bestGf = 0.0f;
    for (int i = 0; i < n; ++i) {
        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        configs[i].launch(M, N, K, 1.0f, dA, dB, 0.0f, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        bool ok = max_rel_error(dC, dRef, M * N) < 1e-3f;

        float ms = time_kernel(configs[i].launch, M, N, K, 1.0f, dA, dB, 0.0f, dC);
        float gf = gflops(M, N, K, ms);
        printf("%-28s %12.1f %10s\n", configs[i].name, gf, ok ? "PASS" : "FAIL");
        if (ok && gf > bestGf) { bestGf = gf; bestName = configs[i].name; }
    }
    printf("---------------------------------------------------------\n");
    if (bestName) printf("Best: %s  (%.1f GFLOP/s)\n", bestName, bestGf);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    return 0;
}
