#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    float toc() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// GEMM: 2*M*N*K FLOPs (multiply + accumulate)
inline float compute_gflops(int M, int N, int K, float time_ms) {
    double flops = 2.0 * M * N * K;
    return static_cast<float>(flops / (time_ms * 1e6));
}

// Benchmark a GEMM kernel: warmup + averaged timed runs
typedef void (*GemmFn)(int, int, int, const float*, const float*, float*);

inline float benchmark_gemm(GemmFn fn, int M, int N, int K,
                            const float* d_A, const float* d_B, float* d_C,
                            int warmup = 3, int repeats = 10) {
    for (int i = 0; i < warmup; ++i) {
        fn(M, N, K, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.tic();
    for (int i = 0; i < repeats; ++i) {
        fn(M, N, K, d_A, d_B, d_C);
    }
    float total_ms = timer.toc();
    return total_ms / repeats;
}
