#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "softmax/08_fused_online.cuh"
#include "softmax/09_warp_reduce.cuh"

// CPU reference: numerically stable softmax (3-pass)
void softmax_cpu_reference(const float* input, float* output,
                           int N_rows, int N_cols) {
    for (int r = 0; r < N_rows; ++r) {
        const float* row_in = input + r * N_cols;
        float* row_out = output + r * N_cols;

        // Pass 1: find max
        float max_val = -FLT_MAX;
        for (int c = 0; c < N_cols; ++c) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        // Pass 2: exp and sum
        float sum = 0.0f;
        for (int c = 0; c < N_cols; ++c) {
            row_out[c] = expf(row_in[c] - max_val);
            sum += row_out[c];
        }

        // Pass 3: normalize
        for (int c = 0; c < N_cols; ++c) {
            row_out[c] /= sum;
        }
    }
}

typedef void (*SoftmaxFn)(const float*, float*, int, int);

struct SoftmaxKernelInfo {
    const char* name;
    SoftmaxFn fn;
};

// Benchmark a softmax kernel: warmup + averaged timed runs
float benchmark_softmax(SoftmaxFn fn, const float* d_input, float* d_output,
                        int N_rows, int N_cols,
                        int warmup = 3, int repeats = 10) {
    for (int i = 0; i < warmup; ++i) {
        fn(d_input, d_output, N_rows, N_cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.tic();
    for (int i = 0; i < repeats; ++i) {
        fn(d_input, d_output, N_rows, N_cols);
    }
    float total_ms = timer.toc();
    return total_ms / repeats;
}

// Max absolute error between device buffer and host reference
float softmax_max_error(const float* d_output, const float* h_ref, int size) {
    float* h_output = new float[size];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        float err = fabsf(h_output[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }

    delete[] h_output;
    return max_err;
}

int main() {
    struct TestSize { int rows; int cols; };
    TestSize sizes[] = {
        {64, 128}, {64, 512}, {64, 1024}, {64, 2048},
        {128, 128}, {128, 512}, {128, 1024}, {128, 2048},
        {256, 1024}, {256, 4096},
        {512, 1024}, {512, 4096},
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    SoftmaxKernelInfo kernels[] = {
        {"08 Fused Online", run_softmax_fused_online},
        {"09 Warp Reduce",  run_softmax_warp_reduce},
    };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    const float peak_bw = 192.0f;  // GB/s
    const float tol = 1e-6f;

    printf("SLICK Softmax Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 10.1 | FP32\n");
    printf("Peak Memory BW: %.0f GB/s\n", peak_bw);
    printf("========================================\n\n");

    for (int s = 0; s < num_sizes; ++s) {
        int N_rows = sizes[s].rows;
        int N_cols = sizes[s].cols;
        int total = N_rows * N_cols;

        // Allocate device memory
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));

        // Initialize with random data on host, copy to device
        float* h_input = new float[total];
        srand(42);
        for (int i = 0; i < total; ++i) {
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float),
                              cudaMemcpyHostToDevice));

        // CPU reference
        float* h_ref = new float[total];
        softmax_cpu_reference(h_input, h_ref, N_rows, N_cols);

        printf("Size: %d rows x %d cols (%d elements, %.2f KB)\n",
               N_rows, N_cols, total, total * sizeof(float) / 1024.0f);
        printf("%-25s %10s %10s %8s %15s\n",
               "Kernel", "GB/s", "Time(us)", "Status", "Max Error");
        printf("------------------------------------------------------------------\n");

        for (int ki = 0; ki < num_kernels; ++ki) {
            CUDA_CHECK(cudaMemset(d_output, 0, total * sizeof(float)));

            float avg_ms = benchmark_softmax(kernels[ki].fn, d_input, d_output,
                                             N_rows, N_cols);
            // GB/s: read input + write output = 2 * total * 4 bytes
            float bytes = 2.0f * total * sizeof(float);
            float gbps = bytes / (avg_ms * 1e6f);
            float time_us = avg_ms * 1000.0f;

            float err = softmax_max_error(d_output, h_ref, total);
            bool pass = err < tol;

            printf("%-25s %10.2f %10.2f %8s %15.2e\n",
                   kernels[ki].name, gbps, time_us,
                   pass ? "PASS" : "FAIL", err);
        }
        printf("\n");

        delete[] h_input;
        delete[] h_ref;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    return 0;
}
