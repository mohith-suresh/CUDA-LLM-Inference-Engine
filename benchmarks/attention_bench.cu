// benchmarks/attention_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "timer.cuh"
#include "flash_attention/10_flash_attn_v2.cuh"
#include "softmax/08_fused_online.cuh"
#include "cutlass_fmha_ref.cuh"

#define CUBLAS_CHECK(call) do {                                        \
    cublasStatus_t stat = call;                                        \
    if (stat != CUBLAS_STATUS_SUCCESS) {                               \
        fprintf(stderr, "cuBLAS error in %s at line %d: %d\n",        \
                __FILE__, __LINE__, (int)stat);                        \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

// ============================================================
// Causal mask kernel (for unfused baseline)
// ============================================================
__global__
void apply_causal_mask_kernel(float* S, int N, int total_bh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int per_bh = N * N;
    int total = total_bh * per_bh;
    if (idx >= total) return;
    int local = idx % per_bh;
    int r = local / N;
    int c = local % N;
    if (r < c) S[idx] = -FLT_MAX;
}

void apply_causal_mask(float* d_S, int N, int B, int H) {
    int total = B * H * N * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    apply_causal_mask_kernel<<<grid, block>>>(d_S, N, B * H);
}

// ============================================================
// CPU reference: naive O(N^2) attention
// ============================================================
void attention_cpu_reference(int B, int H, int N, int d,
                             const float* Q, const float* K, const float* V,
                             float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);

    for (int bh = 0; bh < B * H; ++bh) {
        const float* q = Q + bh * N * d;
        const float* k = K + bh * N * d;
        const float* v = V + bh * N * d;
        float* o       = O + bh * N * d;

        float* S = new float[N * N];
        float* P = new float[N * N];

        // S = Q @ K^T * scale
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int kk = 0; kk < d; ++kk)
                    sum += q[i * d + kk] * k[j * d + kk];
                S[i * N + j] = sum * scale;
            }

        // Causal mask
        if (causal)
            for (int i = 0; i < N; ++i)
                for (int j = i + 1; j < N; ++j)
                    S[i * N + j] = -FLT_MAX;

        // Row-wise softmax (3-pass, numerically stable)
        for (int i = 0; i < N; ++i) {
            float max_val = -FLT_MAX;
            for (int j = 0; j < N; ++j)
                max_val = fmaxf(max_val, S[i * N + j]);
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                P[i * N + j] = expf(S[i * N + j] - max_val);
                sum += P[i * N + j];
            }
            for (int j = 0; j < N; ++j)
                P[i * N + j] /= sum;
        }

        // O = P @ V
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < d; ++j) {
                float sum = 0.0f;
                for (int kk = 0; kk < N; ++kk)
                    sum += P[i * N + kk] * v[kk * d + j];
                o[i * d + j] = sum;
            }

        delete[] S;
        delete[] P;
    }
}

// ============================================================
// Unfused baseline: cuBLAS SGEMM + Kernel 08 softmax + cuBLAS SGEMM
// ============================================================
struct UnfusedBaseline {
    cublasHandle_t handle;
    float* d_S;
    int alloc_size;

    UnfusedBaseline() : d_S(nullptr), alloc_size(0) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    ~UnfusedBaseline() {
        if (d_S) cudaFree(d_S);
        cublasDestroy(handle);
    }

    void ensure_workspace(int size) {
        if (size > alloc_size) {
            if (d_S) cudaFree(d_S);
            CUDA_CHECK(cudaMalloc(&d_S, size * sizeof(float)));
            alloc_size = size;
        }
    }

    void run(int B, int H, int N, int d,
             const float* d_Q, const float* d_K, const float* d_V,
             float* d_O, bool causal) {
        int bh = B * H;
        ensure_workspace(bh * N * N);

        float scale = 1.0f / sqrtf((float)d);
        float zero = 0.0f, one = 1.0f;

        long long stride_qkv = (long long)N * d;
        long long stride_s   = (long long)N * N;

        // S = scale * Q @ K^T  (batched)
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, N, d,
            &scale,
            d_K, d, stride_qkv,
            d_Q, d, stride_qkv,
            &zero,
            d_S, N, stride_s,
            bh));

        // Causal mask
        if (causal)
            apply_causal_mask(d_S, N, B, H);

        // Softmax in-place: treat as (B*H*N) rows x N cols
        run_softmax_fused_online(d_S, d_S, bh * N, N);

        // O = P @ V  (batched)
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d, N, N,
            &one,
            d_V, d, stride_qkv,
            d_S, N, stride_s,
            &zero,
            d_O, d, stride_qkv,
            bh));
    }
};

// ============================================================
// Max absolute error (device vs host)
// ============================================================
float attention_max_error(const float* d_out, const float* h_ref, int size) {
    float* h_out = new float[size];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    delete[] h_out;
    return max_err;
}

// ============================================================
// main
// ============================================================
int main() {
    struct TestConfig {
        int B, H, N, d;
        bool causal;
        const char* desc;
    };

    TestConfig val_configs[] = {
        {1, 12, 128,  64, true,  "Small causal"},
        {1, 12, 256,  64, true,  "Medium causal"},
        {1, 12, 512,  64, true,  "Standard causal"},
        {1, 12, 1024, 64, true,  "Full GPT-2 causal"},
        {2, 12, 512,  64, true,  "Multi-batch causal"},
        {1, 12, 512,  64, false, "Non-causal"},
    };
    int num_val = sizeof(val_configs) / sizeof(val_configs[0]);

    TestConfig bench_configs[] = {
        {1, 12, 256,  64, true,  "Small"},
        {1, 12, 512,  64, true,  "Medium"},
        {1, 12, 1024, 64, true,  "Full GPT-2"},
        {4, 12, 512,  64, true,  "Multi-batch"},
    };
    int num_bench = sizeof(bench_configs) / sizeof(bench_configs[0]);

    const float tol = 1e-3f;
    const float peak_bw = 192.0f;

    UnfusedBaseline unfused;

    printf("SLICK FlashAttention-2 Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 11.8 | FP32\n");
    printf("Peak Memory BW: %.0f GB/s\n", peak_bw);
    printf("================================================\n\n");

    // ======================== Validation ========================
    printf("--- Validation (vs CPU reference, tol=%.0e) ---\n\n", tol);
    printf("%-25s %10s %15s %8s\n", "Config", "Kernel", "Max Error", "Status");
    printf("--------------------------------------------------------------\n");

    for (int ci = 0; ci < num_val; ++ci) {
        TestConfig& c = val_configs[ci];
        int total_qkvo = c.B * c.H * c.N * c.d;

        float *d_Q, *d_K, *d_V, *d_O, *d_O_unfused;
        CUDA_CHECK(cudaMalloc(&d_Q, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_unfused, total_qkvo * sizeof(float)));

        float* h_Q = new float[total_qkvo];
        float* h_K = new float[total_qkvo];
        float* h_V = new float[total_qkvo];
        srand(42 + ci);
        for (int i = 0; i < total_qkvo; ++i) {
            h_Q[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_K[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_V[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q, total_qkvo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K, total_qkvo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V, total_qkvo * sizeof(float), cudaMemcpyHostToDevice));

        float* h_O_ref = new float[total_qkvo];
        attention_cpu_reference(c.B, c.H, c.N, c.d, h_Q, h_K, h_V, h_O_ref, c.causal);

        // FlashAttention
        CUDA_CHECK(cudaMemset(d_O, 0, total_qkvo * sizeof(float)));
        run_flash_attn_v2(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O, c.causal);
        CUDA_CHECK(cudaDeviceSynchronize());
        float err_flash = attention_max_error(d_O, h_O_ref, total_qkvo);
        bool pass_flash = err_flash < tol;
        printf("%-25s %10s %15.2e %8s\n", c.desc, "Flash", err_flash,
               pass_flash ? "PASS" : "FAIL");

        // Unfused baseline
        CUDA_CHECK(cudaMemset(d_O_unfused, 0, total_qkvo * sizeof(float)));
        unfused.run(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O_unfused, c.causal);
        CUDA_CHECK(cudaDeviceSynchronize());
        float err_unfused = attention_max_error(d_O_unfused, h_O_ref, total_qkvo);
        bool pass_unfused = err_unfused < tol;
        printf("%-25s %10s %15.2e %8s\n", c.desc, "Unfused", err_unfused,
               pass_unfused ? "PASS" : "FAIL");

        // CUTLASS FMHA (output is BMHK, needs transpose for comparison)
        float *d_O_cutlass;
        CUDA_CHECK(cudaMalloc(&d_O_cutlass, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_O_cutlass, 0, total_qkvo * sizeof(float)));
        run_cutlass_fmha(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O_cutlass, c.causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_O_cutlass_bmhk = new float[total_qkvo];
        float* h_O_cutlass_bhnk = new float[total_qkvo];
        CUDA_CHECK(cudaMemcpy(h_O_cutlass_bmhk, d_O_cutlass,
                              total_qkvo * sizeof(float), cudaMemcpyDeviceToHost));
        transpose_bmhk_to_bhnk(h_O_cutlass_bmhk, h_O_cutlass_bhnk,
                               c.B, c.H, c.N, c.d);
        float max_err_cutlass = 0.0f;
        for (int i = 0; i < total_qkvo; ++i) {
            float err = fabsf(h_O_cutlass_bhnk[i] - h_O_ref[i]);
            if (err > max_err_cutlass) max_err_cutlass = err;
        }
        bool pass_cutlass = max_err_cutlass < tol;
        printf("%-25s %10s %15.2e %8s\n", c.desc, "CUTLASS", max_err_cutlass,
               pass_cutlass ? "PASS" : "FAIL");
        delete[] h_O_cutlass_bmhk;
        delete[] h_O_cutlass_bhnk;

        delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O_ref;
        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
        CUDA_CHECK(cudaFree(d_O_unfused));
        CUDA_CHECK(cudaFree(d_O_cutlass));
    }

    // ======================== Benchmark ========================
    printf("\n--- Benchmark (causal, warmup=3, repeats=10) ---\n\n");
    printf("%-15s %5s %5s %6s %5s  %10s %10s %10s %10s %10s  %8s\n",
           "Config", "B", "H", "N", "d",
           "Flash(us)", "Unfsd(us)", "CUTLAS(us)", "Speedup", "TFLOPS", "Eff BW");
    printf("-------------------------------------------------------------------------------------------------------------\n");

    for (int ci = 0; ci < num_bench; ++ci) {
        TestConfig& c = bench_configs[ci];
        int total_qkvo = c.B * c.H * c.N * c.d;

        float *d_Q, *d_K, *d_V, *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_qkvo * sizeof(float)));

        float* h_buf = new float[total_qkvo];
        srand(42);
        for (int i = 0; i < total_qkvo; ++i)
            h_buf[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        CUDA_CHECK(cudaMemcpy(d_Q, h_buf, total_qkvo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_buf, total_qkvo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_buf, total_qkvo * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_buf;

        // Benchmark FlashAttention
        for (int w = 0; w < 3; ++w)
            run_flash_attn_v2(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O, c.causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        GpuTimer timer;
        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_flash_attn_v2(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O, c.causal);
        float flash_ms = timer.toc() / 10.0f;
        float flash_us = flash_ms * 1000.0f;

        // Benchmark unfused baseline
        for (int w = 0; w < 3; ++w)
            unfused.run(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O, c.causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.tic();
        for (int r = 0; r < 10; ++r)
            unfused.run(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O, c.causal);
        float unfused_ms = timer.toc() / 10.0f;
        float unfused_us = unfused_ms * 1000.0f;

        // Benchmark CUTLASS FMHA
        float *d_O_cut;
        CUDA_CHECK(cudaMalloc(&d_O_cut, total_qkvo * sizeof(float)));
        for (int w = 0; w < 3; ++w)
            run_cutlass_fmha(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O_cut, c.causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_cutlass_fmha(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O_cut, c.causal);
        float cutlass_ms = timer.toc() / 10.0f;
        float cutlass_us = cutlass_ms * 1000.0f;
        CUDA_CHECK(cudaFree(d_O_cut));

        // Metrics
        double total_flops = 4.0 * c.B * c.H * (double)c.N * c.N * c.d;
        float flash_tflops = (float)(total_flops / (flash_ms * 1e9));
        float speedup = unfused_us / flash_us;

        // Effective bandwidth: ideal IO = 4 * B*H*N*d * sizeof(float)
        double ideal_bytes = 4.0 * c.B * c.H * c.N * c.d * sizeof(float);
        float eff_bw = (float)(ideal_bytes / (flash_ms * 1e6));

        printf("%-15s %5d %5d %6d %5d  %10.1f %10.1f %10.1f %10.2fx %10.3f %7.1f GB/s\n",
               c.desc, c.B, c.H, c.N, c.d,
               flash_us, unfused_us, cutlass_us, speedup, flash_tflops, eff_bw);

        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    }

    printf("\nFLOPs formula: 4 * B * H * N^2 * d (two matmuls: QK^T and PV)\n");
    printf("Eff BW: ideal IO (Q+K+V read + O write) / time vs peak %.0f GB/s\n", peak_bw);

    return 0;
}
