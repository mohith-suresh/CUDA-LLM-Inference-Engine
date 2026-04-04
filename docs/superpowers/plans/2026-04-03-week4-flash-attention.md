# Week 4 — FlashAttention-2 Forward Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement FlashAttention-2 forward pass (Kernel 10) in FP32 for SM75, fusing Q@K^T, softmax, and P@V into a single kernel with O(N) HBM IO, plus a benchmark harness comparing against an unfused cuBLAS+softmax baseline.

**Architecture:** One thread block (256 threads, 8 warps) per (batch, head) pair. Q-outer, KV-inner tiling with Br=64, Bc=32, d=64. Online softmax rescaling with warp shuffle reductions. S=Q@K^T and O+=P@V computed via register tiling; P goes through shared memory between the two GEMMs. Causal mask via compile-time template to eliminate runtime branches.

**Tech Stack:** CUDA 10.1, C++14, CMake, cuBLAS (unfused baseline)

**Spec:** `docs/superpowers/specs/2026-04-02-week4-flash-attention-design.md`

---

### Task 1: Kernel 10 Header — FlashAttention-2 Declaration

**Files:**
- Create: `kernels/flash_attention/10_flash_attn_v2.cuh`

- [ ] **Step 1: Create the header file**

```cpp
// kernels/flash_attention/10_flash_attn_v2.cuh
#pragma once

// FlashAttention-2 forward pass: fused Q@K^T + softmax + P@V
// One thread block (256 threads) per (batch, head) pair
// Tiling: Br=64, Bc=32, d=64. Online softmax with warp shuffle reductions.
void run_flash_attn_v2(int B, int H, int N, int d,
                       const float* Q, const float* K, const float* V,
                       float* O, bool causal);
```

- [ ] **Step 2: Commit**

```bash
git add kernels/flash_attention/10_flash_attn_v2.cuh
git commit -m "feat: add Kernel 10 FlashAttention-2 header"
```

---

### Task 2: Kernel 10 Implementation — FlashAttention-2 Forward Pass

**Files:**
- Create: `kernels/flash_attention/10_flash_attn_v2.cu`

- [ ] **Step 1: Write the kernel implementation**

```cuda
// kernels/flash_attention/10_flash_attn_v2.cu
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "flash_attention/10_flash_attn_v2.cuh"
#include "timer.cuh"

// ============================================================
// Tile dimensions
// ============================================================
#define BR 64        // Q tile rows (outer loop)
#define BC 32        // KV tile rows (inner loop)
#define HD 64        // Head dimension (not tiled)
#define NTHREADS 256 // Threads per block (8 warps)

// Thread tile sizes for S = Q @ K^T  (BR × BC = 64 × 32)
#define TM_S 4  // rows per thread: 16 thread_rows × 4 = 64
#define TN_S 2  // cols per thread: 16 thread_cols × 2 = 32

// Thread tile sizes for O accumulation (BR × HD = 64 × 64)
#define TM_O 4  // rows per thread: 16 thread_rows × 4 = 64
#define TN_O 4  // cols per thread: 16 thread_cols × 4 = 64

// ============================================================
// Kernel
// ============================================================
template <bool CAUSAL>
__global__ __launch_bounds__(NTHREADS)
void flash_attn_v2_kernel(int N, int d, float scale,
                          const float* __restrict__ Q,
                          const float* __restrict__ K,
                          const float* __restrict__ V,
                          float* __restrict__ O) {
    // Each block handles one (batch, head) pair
    const int bh = blockIdx.x;
    const float* Q_bh = Q + bh * N * d;
    const float* K_bh = K + bh * N * d;
    const float* V_bh = V + bh * N * d;
    float*       O_bh = O + bh * N * d;

    const int tid = threadIdx.x;
    const int thread_row = tid / 16;   // 0..15
    const int thread_col = tid % 16;   // 0..15

    // Warp layout: warp k has tid [32k, 32k+31]
    //   lanes 0-15  → thread_row 2k,   thread_cols 0-15
    //   lanes 16-31 → thread_row 2k+1, thread_cols 0-15
    const int lane = tid & 31;
    const int half_leader = (lane < 16) ? 0 : 16;

    // Shared memory (padded +1 per row to avoid bank conflicts)
    __shared__ float Q_smem[BR][HD + 1];    // 64 × 65 = 16,640 B
    __shared__ float KV_smem[BC][HD + 1];   // 32 × 65 =  8,320 B
    __shared__ float P_smem[BR][BC + 1];    // 64 × 33 =  8,448 B
                                             // Total:    33,408 B

    const int num_q_tiles  = (N + BR - 1) / BR;
    const int num_kv_tiles = (N + BC - 1) / BC;

    // ===================== Outer loop: Q tiles =====================
    for (int qi = 0; qi < num_q_tiles; ++qi) {
        const int q_start = qi * BR;

        // --- Load Q_i (BR × d) into Q_smem ---
        // 64 × 64 = 4096 floats / 256 threads = 16 floats/thread
        for (int idx = tid; idx < BR * HD; idx += NTHREADS) {
            const int r = idx / HD;
            const int c = idx % HD;
            const int gr = q_start + r;
            Q_smem[r][c] = (gr < N) ? Q_bh[gr * d + c] : 0.0f;
        }

        // Initialize O accumulator and softmax state in registers
        float O_acc[TM_O][TN_O];
        float m_i[TM_O];   // running row max
        float l_i[TM_O];   // running row sum_exp

        #pragma unroll
        for (int tm = 0; tm < TM_O; ++tm) {
            m_i[tm] = -FLT_MAX;
            l_i[tm] = 0.0f;
            #pragma unroll
            for (int tn = 0; tn < TN_O; ++tn)
                O_acc[tm][tn] = 0.0f;
        }

        __syncthreads();  // Q_smem ready

        // ================= Inner loop: KV tiles =================
        for (int kj = 0; kj < num_kv_tiles; ++kj) {
            const int kv_start = kj * BC;

            // Causal tile skip: entire tile above diagonal
            if (CAUSAL && kv_start > (qi + 1) * BR - 1) break;

            // --- Step 1: Load K_j (BC × d) into KV_smem ---
            // 32 × 64 = 2048 floats / 256 threads = 8 floats/thread
            for (int idx = tid; idx < BC * HD; idx += NTHREADS) {
                const int r = idx / HD;
                const int c = idx % HD;
                const int gr = kv_start + r;
                KV_smem[r][c] = (gr < N) ? K_bh[gr * d + c] : 0.0f;
            }
            __syncthreads();  // KV_smem (K) ready

            // --- Step 2: S = Q @ K^T * scale (BR × BC in registers) ---
            float S[TM_S][TN_S];
            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    S[tm][tn] = 0.0f;

            for (int k = 0; k < HD; ++k) {
                float q_frag[TM_S];
                #pragma unroll
                for (int tm = 0; tm < TM_S; ++tm)
                    q_frag[tm] = Q_smem[thread_row * TM_S + tm][k];

                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn) {
                    float k_val = KV_smem[thread_col * TN_S + tn][k];
                    #pragma unroll
                    for (int tm = 0; tm < TM_S; ++tm)
                        S[tm][tn] += q_frag[tm] * k_val;
                }
            }

            // Scale by 1/sqrt(d)
            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    S[tm][tn] *= scale;

            // Boundary + causal mask: set S = -inf for invalid positions
            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm) {
                const int gr = q_start + thread_row * TM_S + tm;
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn) {
                    const int gc = kv_start + thread_col * TN_S + tn;
                    bool masked = (gc >= N);
                    if (CAUSAL) masked = masked || (gr < gc);
                    if (masked) S[tm][tn] = -FLT_MAX;
                }
            }

            // --- Step 3: Row-wise softmax via half-warp shuffle ---
            // Each S row spans 16 thread_cols × TN_S=2 = 32 values.
            // 16 threads in same half-warp: reduce with offsets 8,4,2,1.
            float m_ij[TM_S], l_ij[TM_S];

            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm) {
                // Local max across TN_S=2 columns
                float local_m = fmaxf(S[tm][0], S[tm][1]);

                // Warp shuffle max reduction (16 lanes in half-warp)
                #pragma unroll
                for (int offset = 8; offset >= 1; offset >>= 1)
                    local_m = fmaxf(local_m,
                                    __shfl_down_sync(0xFFFFFFFF, local_m, offset));
                m_ij[tm] = __shfl_sync(0xFFFFFFFF, local_m, half_leader);

                // exp(S - m_ij) → P values (stored back in S)
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    S[tm][tn] = __expf(S[tm][tn] - m_ij[tm]);

                // Local sum of P values
                float local_l = S[tm][0] + S[tm][1];

                // Warp shuffle sum reduction
                #pragma unroll
                for (int offset = 8; offset >= 1; offset >>= 1)
                    local_l += __shfl_down_sync(0xFFFFFFFF, local_l, offset);
                l_ij[tm] = __shfl_sync(0xFFFFFFFF, local_l, half_leader);
            }

            // --- Online rescaling of O accumulator ---
            #pragma unroll
            for (int tm = 0; tm < TM_O; ++tm) {
                float m_new = fmaxf(m_i[tm], m_ij[tm]);
                float alpha = __expf(m_i[tm] - m_new);   // rescale old O
                float beta  = __expf(m_ij[tm] - m_new);  // scale new P

                #pragma unroll
                for (int tn = 0; tn < TN_O; ++tn)
                    O_acc[tm][tn] *= alpha;

                l_i[tm] = l_i[tm] * alpha + l_ij[tm] * beta;
                m_i[tm] = m_new;

                // Scale P by beta for the P@V accumulation
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    S[tm][tn] *= beta;
            }

            // --- Step 4: Write P_scaled to P_smem ---
            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    P_smem[thread_row * TM_S + tm]
                          [thread_col * TN_S + tn] = S[tm][tn];
            __syncthreads();  // P_smem ready

            // --- Step 5: Load V_j (BC × d) into KV_smem (overwrites K) ---
            for (int idx = tid; idx < BC * HD; idx += NTHREADS) {
                const int r = idx / HD;
                const int c = idx % HD;
                const int gr = kv_start + r;
                KV_smem[r][c] = (gr < N) ? V_bh[gr * d + c] : 0.0f;
            }
            __syncthreads();  // KV_smem (V) ready

            // --- Step 6: O += P @ V  (BR×BC @ BC×d → BR×d) ---
            // Each thread accumulates its 4×4 O subtile
            for (int k = 0; k < BC; ++k) {
                float p_frag[TM_O];
                #pragma unroll
                for (int tm = 0; tm < TM_O; ++tm)
                    p_frag[tm] = P_smem[thread_row * TM_O + tm][k];

                #pragma unroll
                for (int tn = 0; tn < TN_O; ++tn) {
                    float v_val = KV_smem[k][thread_col * TN_O + tn];
                    #pragma unroll
                    for (int tm = 0; tm < TM_O; ++tm)
                        O_acc[tm][tn] += p_frag[tm] * v_val;
                }
            }

            __syncthreads();  // Ensure P@V reads complete before next K load

        }  // end inner loop

        // --- Write O_i to HBM (finalize: O /= l) ---
        #pragma unroll
        for (int tm = 0; tm < TM_O; ++tm) {
            const int gr = q_start + thread_row * TM_O + tm;
            if (gr < N) {
                float inv_l = (l_i[tm] > 0.0f) ? 1.0f / l_i[tm] : 0.0f;
                #pragma unroll
                for (int tn = 0; tn < TN_O; ++tn) {
                    const int gc = thread_col * TN_O + tn;
                    O_bh[gr * d + gc] = O_acc[tm][tn] * inv_l;
                }
            }
        }

        __syncthreads();  // All threads done before next Q tile load

    }  // end outer loop
}

// ============================================================
// Host wrapper
// ============================================================
void run_flash_attn_v2(int B, int H, int N, int d,
                       const float* Q, const float* K, const float* V,
                       float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    dim3 grid(B * H);
    dim3 block(NTHREADS);

    if (causal)
        flash_attn_v2_kernel<true><<<grid, block>>>(N, d, scale, Q, K, V, O);
    else
        flash_attn_v2_kernel<false><<<grid, block>>>(N, d, scale, Q, K, V, O);
}

#undef BR
#undef BC
#undef HD
#undef NTHREADS
#undef TM_S
#undef TN_S
#undef TM_O
#undef TN_O
```

- [ ] **Step 2: Commit**

```bash
git add kernels/flash_attention/10_flash_attn_v2.cu
git commit -m "feat: add Kernel 10 FlashAttention-2 implementation"
```

---

### Task 3: Benchmark Harness — `attention_bench.cu`

**Files:**
- Create: `benchmarks/attention_bench.cu`

- [ ] **Step 1: Write the benchmark with CPU reference and unfused baseline**

```cuda
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

// ============================================================
// cuBLAS error check
// ============================================================
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
    float* d_S;          // workspace: (B*H) × N × N
    int alloc_size;      // current allocation in floats

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

        // S = scale * Q @ K^T  (batched)
        // Row-major: cuBLAS sees Q as Q^T (d×N col-major), K as K^T (d×N col-major)
        // We want S^T = K × Q^T in col-major = S in row-major
        long long stride_qkv = (long long)N * d;
        long long stride_s   = (long long)N * N;

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
        if (causal) {
            apply_causal_mask(d_S, N, B, H);
        }

        // Softmax in-place: treat as (B*H*N) rows × N cols
        run_softmax_fused_online(d_S, d_S, bh * N, N);

        // O = P @ V  (batched)
        // Row-major: cuBLAS computes O^T = V^T × P^T in col-major = O in row-major
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

    // Validation configs
    TestConfig val_configs[] = {
        {1, 12, 128,  64, true,  "Small causal"},
        {1, 12, 256,  64, true,  "Medium causal"},
        {1, 12, 512,  64, true,  "Standard causal"},
        {1, 12, 1024, 64, true,  "Full GPT-2 causal"},
        {2, 12, 512,  64, true,  "Multi-batch causal"},
        {1, 12, 512,  64, false, "Non-causal"},
    };
    int num_val = sizeof(val_configs) / sizeof(val_configs[0]);

    // Benchmark configs
    TestConfig bench_configs[] = {
        {1, 12, 256,  64, true,  "Small"},
        {1, 12, 512,  64, true,  "Medium"},
        {1, 12, 1024, 64, true,  "Full GPT-2"},
        {4, 12, 512,  64, true,  "Multi-batch"},
    };
    int num_bench = sizeof(bench_configs) / sizeof(bench_configs[0]);

    const float tol = 1e-3f;
    const float peak_bw = 192.0f;  // GB/s

    UnfusedBaseline unfused;

    printf("SLICK FlashAttention-2 Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 10.1 | FP32\n");
    printf("Peak Memory BW: %.0f GB/s\n", peak_bw);
    printf("================================================\n\n");

    // ======================== Validation ========================
    printf("--- Validation (vs CPU reference, tol=%.0e) ---\n\n", tol);
    printf("%-25s %10s %15s %8s\n", "Config", "Kernel", "Max Error", "Status");
    printf("--------------------------------------------------------------\n");

    for (int ci = 0; ci < num_val; ++ci) {
        TestConfig& c = val_configs[ci];
        int total_qkvo = c.B * c.H * c.N * c.d;

        // Allocate
        float *d_Q, *d_K, *d_V, *d_O, *d_O_unfused;
        CUDA_CHECK(cudaMalloc(&d_Q, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_unfused, total_qkvo * sizeof(float)));

        // Init random on host
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

        // CPU reference
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

        delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O_ref;
        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
        CUDA_CHECK(cudaFree(d_O_unfused));
    }

    // ======================== Benchmark ========================
    printf("\n--- Benchmark (causal, warmup=3, repeats=10) ---\n\n");
    printf("%-15s %5s %5s %6s %5s  %10s %10s %10s %10s  %8s\n",
           "Config", "B", "H", "N", "d",
           "Flash(us)", "Unfsd(us)", "Speedup", "TFLOPS", "Eff BW");
    printf("--------------------------------------------------------------------------------------------------\n");

    for (int ci = 0; ci < num_bench; ++ci) {
        TestConfig& c = bench_configs[ci];
        int total_qkvo = c.B * c.H * c.N * c.d;

        float *d_Q, *d_K, *d_V, *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_qkvo * sizeof(float)));

        // Init random
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

        // Metrics
        double total_flops = 4.0 * c.B * c.H * (double)c.N * c.N * c.d;
        float flash_tflops = (float)(total_flops / (flash_ms * 1e9));
        float speedup = unfused_us / flash_us;

        // Effective bandwidth: ideal IO = 4 * B*H*N*d * sizeof(float)
        double ideal_bytes = 4.0 * c.B * c.H * c.N * c.d * sizeof(float);
        float eff_bw = (float)(ideal_bytes / (flash_ms * 1e6));

        printf("%-15s %5d %5d %6d %5d  %10.1f %10.1f %10.2fx %10.3f %7.1f GB/s\n",
               c.desc, c.B, c.H, c.N, c.d,
               flash_us, unfused_us, speedup, flash_tflops, eff_bw);

        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    }

    printf("\nFLOPs formula: 4 * B * H * N^2 * d (two matmuls: QK^T and PV)\n");
    printf("Eff BW: ideal IO (Q+K+V read + O write) / time vs peak %.0f GB/s\n", peak_bw);

    return 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/attention_bench.cu
git commit -m "feat: add attention benchmark with CPU reference and unfused baseline"
```

---

### Task 4: CMake Integration

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add attention kernel library and benchmark target**

Append the following after the existing Softmax Benchmark section (after the last line in CMakeLists.txt):

```cmake
# --- Attention Kernels (Week 4) ---
add_library(attention_kernels
    kernels/flash_attention/10_flash_attn_v2.cu
)
target_include_directories(attention_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)

# Attention Benchmark
add_executable(attention_bench benchmarks/attention_bench.cu)
target_link_libraries(attention_bench attention_kernels softmax_kernels ${CUBLAS_LIB})
```

Note: `attention_bench` links `softmax_kernels` for the unfused baseline (Kernel 08) and `${CUBLAS_LIB}` for the cuBLAS SGEMM calls.

- [ ] **Step 2: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add attention kernels and benchmark to CMake"
```

---

### Task 5: Build and Validate

**Files:**
- None (build + run only)

- [ ] **Step 1: Build the project**

```bash
cd /home/adithya/Document/SLICK
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build
```

Expected: compiles with no errors, produces `build/attention_bench`.

- [ ] **Step 2: Run validation**

```bash
./build/attention_bench
```

Expected: all 6 validation configs show `PASS` for both Flash and Unfused, with max error < 1e-3.

- [ ] **Step 3: Debug if needed**

If validation fails, check:
- **Boundary masking:** S must be set to `-FLT_MAX` for KV positions >= N, otherwise `exp(0) = 1` gives phantom attention.
- **Causal mask:** condition is `global_row < global_col` (not `<=`).
- **Warp shuffle reduction:** half-warp leader is lane 0 or 16, not lane 0 for all.
- **Online rescaling:** `beta = exp(m_ij - m_new)` must multiply P before writing to P_smem.
- **P@V indexing:** `P_smem[thread_row * TM_O + tm][k]` must match the P_smem write layout.
- **__syncthreads count:** 4 per inner iteration (after K load, after P write, after V load, after P@V).

If tolerance is too tight, relax to 1e-2 and investigate whether `__expf` precision is the cause.

- [ ] **Step 4: Commit validated build**

```bash
git add -A
git commit -m "feat: Week 4 FlashAttention-2 kernel validated"
```

---

### Task 6: Final Benchmark Run and Analysis

**Files:**
- None (run + capture output)

- [ ] **Step 1: Run the full benchmark**

```bash
./build/attention_bench
```

- [ ] **Step 2: Analyze results**

Key metrics to check:
- **Speedup vs unfused:** FlashAttention should be faster for N≥512 due to avoiding N×N HBM round-trip. For small N, launch overhead may dominate.
- **TFLOPS:** compare against peak 4.3 TFLOPS FP32. Expect ~0.3-1.0 TFLOPS depending on N.
- **Effective bandwidth:** measures how well the kernel hides HBM traffic. Compare against 192 GB/s peak.
- **Scaling with N:** FlashAttention advantage grows with N (O(N) vs O(N²) memory).

- [ ] **Step 3: Verify existing benchmarks still work**

```bash
./build/gemm_bench
./build/softmax_bench
```

Expected: no regressions — all previous kernels still pass validation.
