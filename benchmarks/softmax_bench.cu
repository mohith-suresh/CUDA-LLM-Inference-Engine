#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "timer.cuh"
#include "softmax/08_fused_online.cuh"
#include "softmax/09_warp_reduce.cuh"

#define CUDNN_CHECK(call) do {                                             \
    cudnnStatus_t stat = call;                                             \
    if (stat != CUDNN_STATUS_SUCCESS) {                                    \
        fprintf(stderr, "cuDNN error in %s at line %d: %s\n",             \
                __FILE__, __LINE__, cudnnGetErrorString(stat));            \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while(0)

struct CudnnSoftmax {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t desc;

    CudnnSoftmax() {
        CUDNN_CHECK(cudnnCreate(&handle));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    }

    ~CudnnSoftmax() {
        cudnnDestroyTensorDescriptor(desc);
        cudnnDestroy(handle);
    }

    void forward(const float* d_input, float* d_output, int N_rows, int N_cols) {
        // NCHW: N=N_rows, C=N_cols, H=1, W=1 — softmax over C (channels)
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                                CUDNN_DATA_FLOAT,
                                                N_rows, N_cols, 1, 1));
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxForward(handle,
                                         CUDNN_SOFTMAX_ACCURATE,
                                         CUDNN_SOFTMAX_MODE_CHANNEL,
                                         &alpha, desc, d_input,
                                         &beta, desc, d_output));
    }
};

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

    CudnnSoftmax cudnn_softmax;

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

        // cuDNN reference benchmark
        {
            CUDA_CHECK(cudaMemset(d_output, 0, total * sizeof(float)));

            // warmup
            for (int i = 0; i < 3; ++i)
                cudnn_softmax.forward(d_input, d_output, N_rows, N_cols);
            CUDA_CHECK(cudaDeviceSynchronize());

            GpuTimer timer;
            timer.tic();
            for (int i = 0; i < 10; ++i)
                cudnn_softmax.forward(d_input, d_output, N_rows, N_cols);
            float total_ms = timer.toc();
            float avg_ms = total_ms / 10.0f;

            float bytes = 2.0f * total * sizeof(float);
            float gbps = bytes / (avg_ms * 1e6f);
            float time_us = avg_ms * 1000.0f;

            float err = softmax_max_error(d_output, h_ref, total);

            printf("%-25s %10.2f %10.2f %8s %15.2e\n",
                   "cuDNN", gbps, time_us, "REF", err);
        }
        printf("\n");

        delete[] h_input;
        delete[] h_ref;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    // ==========================================
    // Roofline Analysis (analytical)
    // ==========================================
    // GTX 1650 Ti specs:
    //   1024 CUDA cores @ 2100 MHz = 4300 GFLOPS peak FP32
    //   GDDR6 128-bit bus @ 12 Gbps = 192 GB/s peak DRAM bandwidth
    //   Ridge point = 4300 / 192 = 22.4 FLOP/byte
    //
    // Online softmax per element:
    //   Pass 1: fmax(1) + 2×exp(1) + mul(1) + add(1) + 2×sub(1) = 7 simple FLOPs
    //           But __expf uses SFU, ~8 muladd-equivalent each → 2×8 + 5 = 21 FLOPs
    //   Pass 2: sub(1) + exp(1) + div(1) = 3 simple FLOPs → 8 + 2 = 10 SFU-adjusted
    //   Total: ~31 FLOPs/element (SFU-adjusted)
    //
    // Memory per element (online 2-pass):
    //   Pass 1: read 4 bytes
    //   Pass 2: read 4 bytes + write 4 bytes
    //   Total: 12 bytes/element
    //
    // AI = 31 / 12 = 2.6 FLOP/byte  (deeply memory-bound, 8.6× below ridge)

    const float peak_gflops = 4300.0f;
    const float peak_bw_gbs = 192.0f;
    const float ridge = peak_gflops / peak_bw_gbs;

    // FLOPs per element: counting SFU ops as ~8 muladd-equivalent
    // Pass 1: fmax(1) + sub(1) + exp×8 + mul(1) + sub(1) + exp×8 + add(1) = 21
    // Pass 2: sub(1) + exp×8 + div(1) = 10
    const float flops_per_elem = 31.0f;

    // Bytes per element: read(pass1) + read(pass2) + write(pass2) = 12
    const float bytes_per_elem = 12.0f;

    const float ai = flops_per_elem / bytes_per_elem;

    printf("========================================\n");
    printf("ROOFLINE ANALYSIS (Analytical)\n");
    printf("Peak FP32: %.0f GFLOPS | Peak BW: %.0f GB/s | Ridge: %.1f FLOP/byte\n",
           peak_gflops, peak_bw_gbs, ridge);
    printf("Softmax AI: %.1f FLOP/byte (MEMORY-BOUND, %.1fx below ridge)\n",
           ai, ridge / ai);
    printf("========================================\n\n");

    // Run roofline on a representative set of sizes
    struct RooflineSize { int rows; int cols; };
    RooflineSize rf_sizes[] = {
        {128, 512}, {128, 2048}, {256, 1024}, {256, 4096},
        {512, 1024}, {512, 4096},
    };
    int num_rf = sizeof(rf_sizes) / sizeof(rf_sizes[0]);

    printf("%-20s %7s %7s %10s %8s %8s  %s\n",
           "Kernel", "GFLOPS", "GB/s*", "Time(us)", "%BW", "%MemRoof", "Notes");
    printf("* GB/s = actual DRAM traffic (12 bytes/elem for online, includes re-read)\n");
    printf("--------------------------------------------------------------------------------------------\n");

    float mem_roof = ai * peak_bw_gbs;  // GFLOPS ceiling from memory

    for (int rs = 0; rs < num_rf; ++rs) {
        int N_rows = rf_sizes[rs].rows;
        int N_cols = rf_sizes[rs].cols;
        int total = N_rows * N_cols;

        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));

        float* h_input = new float[total];
        srand(42);
        for (int i = 0; i < total; ++i)
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f;
        CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float),
                              cudaMemcpyHostToDevice));

        for (int ki = 0; ki < num_kernels; ++ki) {
            CUDA_CHECK(cudaMemset(d_output, 0, total * sizeof(float)));

            float avg_ms = benchmark_softmax(kernels[ki].fn, d_input, d_output,
                                             N_rows, N_cols, 3, 20);
            float time_us = avg_ms * 1000.0f;

            // Actual GFLOPS
            double total_flops = (double)flops_per_elem * total;
            float gflops = (float)(total_flops / (avg_ms * 1e6));

            // Actual bandwidth (12 bytes per element: 2 reads + 1 write)
            double total_bytes = (double)bytes_per_elem * total;
            float gbps = (float)(total_bytes / (avg_ms * 1e6));
            float pct_bw = gbps / peak_bw_gbs * 100.0f;

            // % of memory roofline ceiling
            float pct_roof = gflops / mem_roof * 100.0f;

            char label[64];
            snprintf(label, sizeof(label), "%s %dx%d",
                     kernels[ki].name, N_rows, N_cols);

            printf("%-20s %7.1f %7.1f %10.1f %7.1f%% %7.1f%%  %s\n",
                   label, gflops, gbps, time_us, pct_bw, pct_roof,
                   pct_bw > 50.0f ? "good" : "launch-limited");
        }

        // cuDNN roofline entry
        {
            CUDA_CHECK(cudaMemset(d_output, 0, total * sizeof(float)));

            for (int i = 0; i < 3; ++i)
                cudnn_softmax.forward(d_input, d_output, N_rows, N_cols);
            CUDA_CHECK(cudaDeviceSynchronize());

            GpuTimer timer;
            timer.tic();
            for (int i = 0; i < 20; ++i)
                cudnn_softmax.forward(d_input, d_output, N_rows, N_cols);
            float total_ms = timer.toc();
            float avg_ms = total_ms / 20.0f;
            float time_us = avg_ms * 1000.0f;

            // cuDNN likely uses 3-pass: 3 reads + 1 write = 16 bytes/elem
            float cudnn_bytes_per_elem = 16.0f;
            double total_bytes = (double)cudnn_bytes_per_elem * total;
            float gbps = (float)(total_bytes / (avg_ms * 1e6));
            float pct_bw = gbps / peak_bw_gbs * 100.0f;

            // Use same FLOP count for fair GFLOPS comparison
            double total_flops = (double)flops_per_elem * total;
            float gflops = (float)(total_flops / (avg_ms * 1e6));
            float pct_roof = gflops / mem_roof * 100.0f;

            char label[64];
            snprintf(label, sizeof(label), "cuDNN %dx%d",
                     N_rows, N_cols);

            printf("%-20s %7.1f %7.1f %10.1f %7.1f%% %7.1f%%  %s\n",
                   label, gflops, gbps, time_us, pct_bw, pct_roof,
                   "3-pass (16 B/elem)");
        }

        printf("\n");
        delete[] h_input;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    printf("AI = %.1f FLOP/byte | Memory roof = %.0f GFLOPS | Ridge = %.1f FLOP/byte\n",
           ai, mem_roof, ridge);
    printf("Online 2-pass: 12 bytes/elem (read, re-read, write)\n");
    printf("Naive  3-pass: 16 bytes/elem (read, read+write, read+write) — cuDNN estimate\n");

    return 0;
}
