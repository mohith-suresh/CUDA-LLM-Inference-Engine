# Week 5: PagedAttention + GQA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement K11 (PagedAttention) and K12 (GQA) with paged KV cache indirection, validated against K10 FlashAttention.

**Architecture:** Shared kernel template in `11_paged_attn.cuh` parameterized on `<int GROUP_SIZE, bool CAUSAL>`. K11 wraps with GROUP_SIZE=1. K12 dispatches GROUP_SIZE={1,2,4,8} at runtime. Block table maps logical KV blocks to non-contiguous physical blocks. Tiling: Br=64, Bc=16 (= BLOCK_SIZE), d=64, 256 threads.

**Tech Stack:** CUDA 11.8, C++17, Google Test, CMake

---

## File Map

| File | Purpose |
|------|---------|
| `kernels/paged_attention/11_paged_attn.cuh` | Kernel template `<GROUP_SIZE, CAUSAL>` + K11 host declaration |
| `kernels/paged_attention/11_paged_attn.cu` | K11 host wrapper (GROUP_SIZE=1) |
| `kernels/paged_attention/12_gqa.cuh` | K12 host declaration |
| `kernels/paged_attention/12_gqa.cu` | K12 host wrapper (GROUP_SIZE dispatch) |
| `tests/test_paged_attention.cu` | Tests for K11 and K12 |
| `benchmarks/paged_attention_bench.cu` | Benchmark K11/K12 vs K10 |
| `CMakeLists.txt` | Build integration |

---

### Task 1: K11 Stub + Build Config

**Files:**
- Create: `kernels/paged_attention/11_paged_attn.cuh`
- Create: `kernels/paged_attention/11_paged_attn.cu`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create K11 header with kernel template declaration and host wrapper**

```cpp
// kernels/paged_attention/11_paged_attn.cuh
#pragma once

// PagedAttention: FlashAttention-2 with block-table KV cache indirection
// KV cache layout: [num_physical_blocks][BLOCK_SIZE][H_kv][d]
// Block table: [B][max_blocks_per_seq] maps logical → physical block index
void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal);
```

- [ ] **Step 2: Create K11 stub implementation**

```cpp
// kernels/paged_attention/11_paged_attn.cu
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "paged_attention/11_paged_attn.cuh"
#include "timer.cuh"

void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal) {
    // Stub — to be implemented
}
```

- [ ] **Step 3: Add paged_attention_kernels library to CMakeLists.txt**

Append after the `attention_kernels` block (after line 93 in CMakeLists.txt):

```cmake
# --- Paged Attention Kernels (Week 5) ---
add_library(paged_attention_kernels
    kernels/paged_attention/11_paged_attn.cu
)
target_include_directories(paged_attention_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)
```

- [ ] **Step 4: Build to verify compilation**

Run: `cd /home/adithya/Document/SLICK && cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add kernels/paged_attention/11_paged_attn.cuh kernels/paged_attention/11_paged_attn.cu CMakeLists.txt
git commit -m "feat: add K11 PagedAttention stub and build config"
```

---

### Task 2: K11 Test Cases

**Files:**
- Create: `tests/test_paged_attention.cu`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write test file with helper to build contiguous block table and validate K11 against K10**

The test strategy: fill Q/K/V with random data, build a paged KV cache with contiguous block mapping, run K11, compare output to K10 (FlashAttention). For K11 with a contiguous block table, the output must match K10 within 1e-5.

```cpp
// tests/test_paged_attention.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include "timer.cuh"
#include "flash_attention/10_flash_attn_v2.cuh"
#include "paged_attention/11_paged_attn.cuh"

// Helper: convert contiguous K/V [B, H, N, d] into paged cache layout
// k_cache_out: [num_physical_blocks][block_size][H][d]
// block_table_out: [B][num_blocks_per_seq]
// With contiguous mapping: block_table[b][i] = b * blocks_per_seq + i
static void build_paged_kv_cache(
    const float* h_K, const float* h_V,
    int B, int H, int N, int d, int block_size,
    float* h_k_cache, float* h_v_cache,
    int* h_block_table, int* h_context_lens)
{
    int blocks_per_seq = (N + block_size - 1) / block_size;
    for (int b = 0; b < B; ++b) {
        h_context_lens[b] = N;
        for (int blk = 0; blk < blocks_per_seq; ++blk) {
            int phys = b * blocks_per_seq + blk;
            h_block_table[b * blocks_per_seq + blk] = phys;
            for (int t = 0; t < block_size; ++t) {
                int seq_pos = blk * block_size + t;
                for (int h_idx = 0; h_idx < H; ++h_idx) {
                    for (int dd = 0; dd < d; ++dd) {
                        int cache_idx = ((phys * block_size + t) * H + h_idx) * d + dd;
                        if (seq_pos < N) {
                            int src_idx = ((b * H + h_idx) * N + seq_pos) * d + dd;
                            h_k_cache[cache_idx] = h_K[src_idx];
                            h_v_cache[cache_idx] = h_V[src_idx];
                        } else {
                            h_k_cache[cache_idx] = 0.0f;
                            h_v_cache[cache_idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

class PagedAttnTest : public ::testing::Test {
protected:
    void RunK11vsK10(int B, int H, int N, int d, int block_size, bool causal, float tol = 1e-5f) {
        int total_qo = B * H * N * d;
        int blocks_per_seq = (N + block_size - 1) / block_size;
        int num_phys_blocks = B * blocks_per_seq;
        int cache_size = num_phys_blocks * block_size * H * d;

        // Host allocations
        std::vector<float> h_Q(total_qo), h_K(total_qo), h_V(total_qo);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_block_table(B * blocks_per_seq);
        std::vector<int> h_context_lens(B);

        srand(42);
        for (int i = 0; i < total_qo; ++i) {
            h_Q[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_K[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_V[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }

        build_paged_kv_cache(h_K.data(), h_V.data(), B, H, N, d, block_size,
                             h_k_cache.data(), h_v_cache.data(),
                             h_block_table.data(), h_context_lens.data());

        // Device allocations
        float *d_Q, *d_K, *d_V, *d_O_k10, *d_O_k11;
        float *d_k_cache, *d_v_cache;
        int *d_block_table, *d_context_lens;

        CUDA_CHECK(cudaMalloc(&d_Q, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_k10, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_k11, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_table, B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_context_lens, B * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_qo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total_qo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total_qo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table.data(), B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_context_lens, h_context_lens.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        // Run K10 reference
        CUDA_CHECK(cudaMemset(d_O_k10, 0, total_qo * sizeof(float)));
        run_flash_attn_v2(B, H, N, d, d_Q, d_K, d_V, d_O_k10, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Run K11
        CUDA_CHECK(cudaMemset(d_O_k11, 0, total_qo * sizeof(float)));
        run_paged_attn(B, H, N, d, d_Q, d_k_cache, d_v_cache,
                       d_block_table, d_context_lens,
                       N, block_size, blocks_per_seq, d_O_k11, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compare
        std::vector<float> h_O_k10(total_qo), h_O_k11(total_qo);
        CUDA_CHECK(cudaMemcpy(h_O_k10.data(), d_O_k10, total_qo * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_O_k11.data(), d_O_k11, total_qo * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < total_qo; ++i) {
            float err = fabsf(h_O_k10[i] - h_O_k11[i]);
            if (err > max_err) max_err = err;
        }

        EXPECT_LT(max_err, tol) << "K11 vs K10 max error: " << max_err;

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_O_k10); cudaFree(d_O_k11);
        cudaFree(d_k_cache); cudaFree(d_v_cache);
        cudaFree(d_block_table); cudaFree(d_context_lens);
    }
};

// --- K11 PagedAttention tests ---
TEST_F(PagedAttnTest, K11_SmallCausal)    { RunK11vsK10(1, 1, 64, 64, 16, true); }
TEST_F(PagedAttnTest, K11_MediumCausal)   { RunK11vsK10(1, 8, 128, 64, 16, true); }
TEST_F(PagedAttnTest, K11_LargeCausal)    { RunK11vsK10(1, 8, 256, 64, 16, true); }
TEST_F(PagedAttnTest, K11_NonCausal)      { RunK11vsK10(1, 8, 128, 64, 16, false); }
TEST_F(PagedAttnTest, K11_MultiBatch)     { RunK11vsK10(2, 8, 128, 64, 16, true); }
TEST_F(PagedAttnTest, K11_LargeMultiHead) { RunK11vsK10(1, 12, 512, 64, 16, true); }
```

- [ ] **Step 2: Add test target to CMakeLists.txt**

Append to test section:

```cmake
add_executable(test_paged_attention tests/test_paged_attention.cu)
target_link_libraries(test_paged_attention GTest::gtest_main paged_attention_kernels attention_kernels)
add_test(NAME PagedAttentionTests COMMAND test_paged_attention)
```

- [ ] **Step 3: Build tests to verify compilation**

Run: `cmake --build build 2>&1 | tail -5`
Expected: Build succeeds (tests will fail at runtime since K11 is a stub)

- [ ] **Step 4: Commit**

```bash
git add tests/test_paged_attention.cu CMakeLists.txt
git commit -m "test: add K11 PagedAttention test cases"
```

---

### Task 3: Implement K11 PagedAttention Kernel

**Files:**
- Modify: `kernels/paged_attention/11_paged_attn.cuh` (replace with kernel template)
- Modify: `kernels/paged_attention/11_paged_attn.cu` (replace with full implementation)

- [ ] **Step 1: Replace 11_paged_attn.cuh with full kernel template + host declaration**

The kernel template is the core shared between K11 and K12. It implements FlashAttention-2 with paged KV cache indirection. `GROUP_SIZE` controls how many Q heads share one KV head.

```cpp
// kernels/paged_attention/11_paged_attn.cuh
#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ============================================================
// Tile dimensions for PagedAttention
// ============================================================
#define PA_BR 64        // Q tile rows
#define PA_BC 16        // KV tile = one physical block (BLOCK_SIZE)
#define PA_HD 64        // Head dimension
#define PA_NTHREADS 256 // 8 warps

// Thread tiles for S = Q @ K^T  (PA_BR x PA_BC = 64 x 16)
// 256 threads -> 16 thread_rows x 16 thread_cols
// Each thread: TM_S=4 rows x TN_S=1 col -> covers 64 x 16
#define PA_TM_S 4
#define PA_TN_S 1

// Thread tiles for O accumulation (PA_BR x PA_HD = 64 x 64)
// Each thread: TM_O=4 rows x TN_O=4 cols -> covers 64 x 64
#define PA_TM_O 4
#define PA_TN_O 4

// ============================================================
// Kernel template
// ============================================================
template <int GROUP_SIZE, bool CAUSAL>
__global__ __launch_bounds__(PA_NTHREADS)
void paged_attn_kernel(int N, int d, float scale,
                       const float* __restrict__ Q,
                       const float* __restrict__ k_cache,
                       const float* __restrict__ v_cache,
                       const int* __restrict__ block_table,
                       const int* __restrict__ context_lens,
                       int max_context_len, int block_size,
                       int num_blocks_per_seq,
                       int H_q,
                       float* __restrict__ O) {
    // Grid: (B, H_q, num_q_tiles)
    const int batch = blockIdx.x;
    const int q_head = blockIdx.y;
    const int q_tile = blockIdx.z;
    const int kv_head = q_head / GROUP_SIZE;

    const int ctx_len = context_lens[batch];
    const int num_kv_blocks = (ctx_len + block_size - 1) / block_size;

    const int q_start = q_tile * PA_BR;
    if (q_start >= N) return;  // out-of-bounds Q tile

    // Pointers for this batch+head
    const float* Q_bh = Q + ((batch * H_q + q_head) * N + q_start) * d;
    float*       O_bh = O + ((batch * H_q + q_head) * N + q_start) * d;
    const int* bt = block_table + batch * num_blocks_per_seq;

    const int tid = threadIdx.x;
    const int thread_row = tid / 16;   // 0..15
    const int thread_col = tid % 16;   // 0..15

    const int lane = tid & 31;
    const int half_leader = (lane < 16) ? 0 : 16;

    // Shared memory
    __shared__ float Q_smem[PA_BR][PA_HD + 1];    // 64 x 65
    __shared__ float KV_smem[PA_BC][PA_HD + 1];   // 16 x 65
    __shared__ float P_smem[PA_BR][PA_BC + 1];    // 64 x 17

    // Load Q tile into shared memory
    for (int idx = tid; idx < PA_BR * PA_HD; idx += PA_NTHREADS) {
        int r = idx / PA_HD;
        int c = idx % PA_HD;
        int gr = q_start + r;
        Q_smem[r][c] = (gr < N) ? Q_bh[r * d + c] : 0.0f;
    }

    // Initialize O accumulator and softmax state
    float O_acc[PA_TM_O][PA_TN_O];
    float m_i[PA_TM_O];
    float l_i[PA_TM_O];

    #pragma unroll
    for (int tm = 0; tm < PA_TM_O; ++tm) {
        m_i[tm] = -FLT_MAX;
        l_i[tm] = 0.0f;
        #pragma unroll
        for (int tn = 0; tn < PA_TN_O; ++tn)
            O_acc[tm][tn] = 0.0f;
    }

    __syncthreads();  // Q_smem ready

    // ================= Inner loop: KV blocks =================
    for (int blk_idx = 0; blk_idx < num_kv_blocks; ++blk_idx) {
        int kv_start = blk_idx * block_size;

        // Causal tile skip
        if (CAUSAL && kv_start > q_start + PA_BR - 1) break;

        int phys_block = bt[blk_idx];

        // --- Load K from paged cache into KV_smem ---
        // k_cache layout: [num_phys_blocks][block_size][H_kv][d]
        const float* K_block = k_cache + ((phys_block * block_size) * (H_q / GROUP_SIZE) + 0) * d;
        for (int idx = tid; idx < PA_BC * PA_HD; idx += PA_NTHREADS) {
            int r = idx / PA_HD;
            int c = idx % PA_HD;
            int seq_pos = kv_start + r;
            if (seq_pos < ctx_len && r < block_size) {
                // k_cache[phys_block][r][kv_head][c]
                int cache_idx = ((phys_block * block_size + r) * (H_q / GROUP_SIZE) + kv_head) * d + c;
                KV_smem[r][c] = k_cache[cache_idx];
            } else {
                KV_smem[r][c] = 0.0f;
            }
        }
        __syncthreads();  // KV_smem (K) ready

        // --- S = Q @ K^T * scale ---
        float S[PA_TM_S][PA_TN_S];
        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm)
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                S[tm][tn] = 0.0f;

        for (int k = 0; k < PA_HD; ++k) {
            float q_frag[PA_TM_S];
            #pragma unroll
            for (int tm = 0; tm < PA_TM_S; ++tm)
                q_frag[tm] = Q_smem[thread_row * PA_TM_S + tm][k];

            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn) {
                float k_val = KV_smem[thread_col * PA_TN_S + tn][k];
                #pragma unroll
                for (int tm = 0; tm < PA_TM_S; ++tm)
                    S[tm][tn] += q_frag[tm] * k_val;
            }
        }

        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm)
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                S[tm][tn] *= scale;

        // Boundary + causal mask
        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm) {
            int gr = q_start + thread_row * PA_TM_S + tm;
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn) {
                int gc = kv_start + thread_col * PA_TN_S + tn;
                bool masked = (gc >= ctx_len);
                if (CAUSAL) masked = masked || (gr < gc);
                if (masked) S[tm][tn] = -FLT_MAX;
            }
        }

        // --- Row-wise softmax via half-warp shuffle ---
        // With PA_TN_S=1, each thread has 1 col. 16 lanes cover 16 cols.
        float m_ij[PA_TM_S], l_ij[PA_TM_S];

        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm) {
            float local_m = S[tm][0];

            // Half-warp max reduction (16 lanes)
            #pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1)
                local_m = fmaxf(local_m,
                                __shfl_down_sync(0xFFFFFFFF, local_m, offset));
            m_ij[tm] = __shfl_sync(0xFFFFFFFF, local_m, half_leader);

            // exp(S - m_ij)
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                S[tm][tn] = __expf(S[tm][tn] - m_ij[tm]);

            float local_l = S[tm][0];

            // Half-warp sum reduction
            #pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1)
                local_l += __shfl_down_sync(0xFFFFFFFF, local_l, offset);
            l_ij[tm] = __shfl_sync(0xFFFFFFFF, local_l, half_leader);
        }

        // --- Online rescaling ---
        #pragma unroll
        for (int tm = 0; tm < PA_TM_O; ++tm) {
            float m_new = fmaxf(m_i[tm], m_ij[tm]);
            float alpha = __expf(m_i[tm] - m_new);
            float beta  = __expf(m_ij[tm] - m_new);

            #pragma unroll
            for (int tn = 0; tn < PA_TN_O; ++tn)
                O_acc[tm][tn] *= alpha;

            l_i[tm] = l_i[tm] * alpha + l_ij[tm] * beta;
            m_i[tm] = m_new;

            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                S[tm][tn] *= beta;
        }

        // --- Write P to P_smem ---
        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm)
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                P_smem[thread_row * PA_TM_S + tm]
                      [thread_col * PA_TN_S + tn] = S[tm][tn];
        __syncthreads();  // P_smem ready

        // --- Load V from paged cache into KV_smem ---
        for (int idx = tid; idx < PA_BC * PA_HD; idx += PA_NTHREADS) {
            int r = idx / PA_HD;
            int c = idx % PA_HD;
            int seq_pos = kv_start + r;
            if (seq_pos < ctx_len && r < block_size) {
                int cache_idx = ((phys_block * block_size + r) * (H_q / GROUP_SIZE) + kv_head) * d + c;
                KV_smem[r][c] = v_cache[cache_idx];
            } else {
                KV_smem[r][c] = 0.0f;
            }
        }
        __syncthreads();  // KV_smem (V) ready

        // --- O += P @ V ---
        for (int k = 0; k < PA_BC; ++k) {
            float p_frag[PA_TM_O];
            #pragma unroll
            for (int tm = 0; tm < PA_TM_O; ++tm)
                p_frag[tm] = P_smem[thread_row * PA_TM_O + tm][k];

            #pragma unroll
            for (int tn = 0; tn < PA_TN_O; ++tn) {
                float v_val = KV_smem[k][thread_col * PA_TN_O + tn];
                #pragma unroll
                for (int tm = 0; tm < PA_TM_O; ++tm)
                    O_acc[tm][tn] += p_frag[tm] * v_val;
            }
        }

        __syncthreads();  // Ensure reads complete before next block load

    }  // end KV block loop

    // --- Write O to HBM ---
    #pragma unroll
    for (int tm = 0; tm < PA_TM_O; ++tm) {
        int gr = q_start + thread_row * PA_TM_O + tm;
        if (gr < N) {
            float inv_l = (l_i[tm] > 0.0f) ? 1.0f / l_i[tm] : 0.0f;
            #pragma unroll
            for (int tn = 0; tn < PA_TN_O; ++tn) {
                int gc = thread_col * PA_TN_O + tn;
                O_bh[tm * d + thread_row * PA_TM_O * 0 + gc] = O_acc[tm][tn] * inv_l;
            }
        }
    }
}

// Host wrapper declaration (GROUP_SIZE=1, standard MHA)
void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal);
```

Wait — the O write-back indexing needs care. The output pointer `O_bh` is relative to `(batch, q_head, q_start)`. Each thread owns rows `[thread_row * TM_O .. thread_row * TM_O + TM_O - 1]` and cols `[thread_col * TN_O .. thread_col * TN_O + TN_O - 1]`. The write should be:

```cpp
O_bh[(thread_row * PA_TM_O + tm) * d + thread_col * PA_TN_O + tn] = O_acc[tm][tn] * inv_l;
```

This is captured correctly in the full implementation below (Step 3).

- [ ] **Step 2: Replace 11_paged_attn.cu with full host wrapper**

```cpp
// kernels/paged_attention/11_paged_attn.cu
#include "paged_attention/11_paged_attn.cuh"

void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    int num_q_tiles = (N + PA_BR - 1) / PA_BR;
    dim3 grid(B, H, num_q_tiles);
    dim3 block(PA_NTHREADS);

    if (causal)
        paged_attn_kernel<1, true><<<grid, block>>>(
            N, d, scale, Q, k_cache, v_cache,
            block_table, context_lens,
            max_context_len, block_size, num_blocks_per_seq,
            H, O);
    else
        paged_attn_kernel<1, false><<<grid, block>>>(
            N, d, scale, Q, k_cache, v_cache,
            block_table, context_lens,
            max_context_len, block_size, num_blocks_per_seq,
            H, O);
}
```

- [ ] **Step 3: Build and run tests**

Run: `cmake --build build && ./build/test_paged_attention 2>&1`
Expected: All K11 tests PASS

- [ ] **Step 4: Commit**

```bash
git add kernels/paged_attention/11_paged_attn.cuh kernels/paged_attention/11_paged_attn.cu
git commit -m "feat: implement K11 PagedAttention kernel"
```

---

### Task 4: K12 GQA Kernel

**Files:**
- Create: `kernels/paged_attention/12_gqa.cuh`
- Create: `kernels/paged_attention/12_gqa.cu`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_paged_attention.cu`

- [ ] **Step 1: Create K12 header**

```cpp
// kernels/paged_attention/12_gqa.cuh
#pragma once

// GQA PagedAttention: multiple Q heads share fewer KV heads
// group_size = H_q / H_kv. Dispatches to template GROUP_SIZE={1,2,4,8}.
void run_gqa_paged_attn(int B, int H_q, int H_kv, int N, int d,
                        const float* Q,
                        const float* k_cache, const float* v_cache,
                        const int* block_table, const int* context_lens,
                        int max_context_len, int block_size,
                        int num_blocks_per_seq,
                        float* O, bool causal);
```

- [ ] **Step 2: Create K12 implementation with GROUP_SIZE dispatch**

```cpp
// kernels/paged_attention/12_gqa.cu
#include "paged_attention/11_paged_attn.cuh"
#include "paged_attention/12_gqa.cuh"

template <int GROUP_SIZE>
static void launch_gqa(int B, int H_q, int N, int d, float scale,
                       const float* Q,
                       const float* k_cache, const float* v_cache,
                       const int* block_table, const int* context_lens,
                       int max_context_len, int block_size,
                       int num_blocks_per_seq,
                       float* O, bool causal) {
    int num_q_tiles = (N + PA_BR - 1) / PA_BR;
    dim3 grid(B, H_q, num_q_tiles);
    dim3 block(PA_NTHREADS);

    if (causal)
        paged_attn_kernel<GROUP_SIZE, true><<<grid, block>>>(
            N, d, scale, Q, k_cache, v_cache,
            block_table, context_lens,
            max_context_len, block_size, num_blocks_per_seq,
            H_q, O);
    else
        paged_attn_kernel<GROUP_SIZE, false><<<grid, block>>>(
            N, d, scale, Q, k_cache, v_cache,
            block_table, context_lens,
            max_context_len, block_size, num_blocks_per_seq,
            H_q, O);
}

void run_gqa_paged_attn(int B, int H_q, int H_kv, int N, int d,
                        const float* Q,
                        const float* k_cache, const float* v_cache,
                        const int* block_table, const int* context_lens,
                        int max_context_len, int block_size,
                        int num_blocks_per_seq,
                        float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    int group_size = H_q / H_kv;

    switch (group_size) {
        case 1: launch_gqa<1>(B, H_q, N, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, causal); break;
        case 2: launch_gqa<2>(B, H_q, N, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, causal); break;
        case 4: launch_gqa<4>(B, H_q, N, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, causal); break;
        case 8: launch_gqa<8>(B, H_q, N, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, causal); break;
        default: break;  // unsupported group size
    }
}
```

- [ ] **Step 3: Add 12_gqa.cu to CMakeLists.txt paged_attention_kernels library**

```cmake
add_library(paged_attention_kernels
    kernels/paged_attention/11_paged_attn.cu
    kernels/paged_attention/12_gqa.cu
)
```

- [ ] **Step 4: Add GQA tests to test_paged_attention.cu**

Add a GQA test helper and test cases. The helper builds paged KV cache with H_kv heads (not H_q), runs K12, and compares against a CPU naive GQA reference.

```cpp
// Add after K11 tests in test_paged_attention.cu
#include "paged_attention/12_gqa.cuh"

// CPU reference for GQA attention
static void gqa_cpu_reference(int B, int H_q, int H_kv, int N, int d,
                               const float* Q, const float* K, const float* V,
                               float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    int group_size = H_q / H_kv;

    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < H_q; ++hq) {
            int hkv = hq / group_size;
            const float* q = Q + (b * H_q + hq) * N * d;
            const float* k = K + (b * H_kv + hkv) * N * d;
            const float* v = V + (b * H_kv + hkv) * N * d;
            float* o       = O + (b * H_q + hq) * N * d;

            for (int i = 0; i < N; ++i) {
                // Compute scores
                float max_s = -FLT_MAX;
                float* scores = new float[N];
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < d; ++kk)
                        sum += q[i * d + kk] * k[j * d + kk];
                    scores[j] = sum * scale;
                    if (causal && j > i) scores[j] = -FLT_MAX;
                    if (scores[j] > max_s) max_s = scores[j];
                }
                // Softmax
                float sum_exp = 0.0f;
                for (int j = 0; j < N; ++j) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < N; ++j)
                    scores[j] /= sum_exp;
                // O = P @ V
                for (int dd = 0; dd < d; ++dd) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j)
                        sum += scores[j] * v[j * d + dd];
                    o[i * d + dd] = sum;
                }
                delete[] scores;
            }
        }
    }
}

class GQATest : public ::testing::Test {
protected:
    void RunGQA(int B, int H_q, int H_kv, int N, int d, int block_size,
                bool causal, float tol = 1e-5f) {
        int total_q = B * H_q * N * d;
        int total_kv = B * H_kv * N * d;
        int blocks_per_seq = (N + block_size - 1) / block_size;
        int num_phys_blocks = B * blocks_per_seq;
        int cache_size = num_phys_blocks * block_size * H_kv * d;

        std::vector<float> h_Q(total_q), h_K(total_kv), h_V(total_kv);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_block_table(B * blocks_per_seq);
        std::vector<int> h_context_lens(B);
        std::vector<float> h_O_ref(total_q);

        srand(42);
        for (int i = 0; i < total_q; ++i)
            h_Q[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        for (int i = 0; i < total_kv; ++i) {
            h_K[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_V[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }

        // Build paged cache with H_kv heads
        for (int b = 0; b < B; ++b) {
            h_context_lens[b] = N;
            for (int blk = 0; blk < blocks_per_seq; ++blk) {
                int phys = b * blocks_per_seq + blk;
                h_block_table[b * blocks_per_seq + blk] = phys;
                for (int t = 0; t < block_size; ++t) {
                    int seq_pos = blk * block_size + t;
                    for (int h = 0; h < H_kv; ++h) {
                        for (int dd = 0; dd < d; ++dd) {
                            int cache_idx = ((phys * block_size + t) * H_kv + h) * d + dd;
                            if (seq_pos < N) {
                                int src_idx = ((b * H_kv + h) * N + seq_pos) * d + dd;
                                h_k_cache[cache_idx] = h_K[src_idx];
                                h_v_cache[cache_idx] = h_V[src_idx];
                            } else {
                                h_k_cache[cache_idx] = 0.0f;
                                h_v_cache[cache_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
        }

        // CPU reference
        gqa_cpu_reference(B, H_q, H_kv, N, d, h_Q.data(), h_K.data(), h_V.data(),
                          h_O_ref.data(), causal);

        // Device allocations
        float *d_Q, *d_O, *d_k_cache, *d_v_cache;
        int *d_block_table, *d_context_lens;

        CUDA_CHECK(cudaMalloc(&d_Q, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_table, B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_context_lens, B * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_q * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table.data(), B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_context_lens, h_context_lens.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_O, 0, total_q * sizeof(float)));
        run_gqa_paged_attn(B, H_q, H_kv, N, d, d_Q, d_k_cache, d_v_cache,
                           d_block_table, d_context_lens,
                           N, block_size, blocks_per_seq, d_O, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_O(total_q);
        CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, total_q * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < total_q; ++i) {
            float err = fabsf(h_O[i] - h_O_ref[i]);
            if (err > max_err) max_err = err;
        }

        EXPECT_LT(max_err, tol) << "GQA max error: " << max_err
                                 << " (H_q=" << H_q << ", H_kv=" << H_kv << ")";

        cudaFree(d_Q); cudaFree(d_O);
        cudaFree(d_k_cache); cudaFree(d_v_cache);
        cudaFree(d_block_table); cudaFree(d_context_lens);
    }
};

// K12 GQA: GROUP_SIZE=1 (MHA, should match K11)
TEST_F(GQATest, GQA_Group1_Causal)    { RunGQA(1, 8, 8, 128, 64, 16, true); }
TEST_F(GQATest, GQA_Group1_NonCausal) { RunGQA(1, 8, 8, 128, 64, 16, false); }
// K12 GQA: GROUP_SIZE=2
TEST_F(GQATest, GQA_Group2_Causal)    { RunGQA(1, 8, 4, 128, 64, 16, true); }
// K12 GQA: GROUP_SIZE=4
TEST_F(GQATest, GQA_Group4_Causal)    { RunGQA(1, 8, 2, 128, 64, 16, true); }
TEST_F(GQATest, GQA_Group4_Large)     { RunGQA(1, 16, 4, 256, 64, 16, true); }
// K12 GQA: GROUP_SIZE=8
TEST_F(GQATest, GQA_Group8_Causal)    { RunGQA(1, 8, 1, 128, 64, 16, true); }
// Multi-batch GQA
TEST_F(GQATest, GQA_MultiBatch)       { RunGQA(2, 8, 4, 128, 64, 16, true); }
```

- [ ] **Step 5: Build and run all tests**

Run: `cmake --build build && ./build/test_paged_attention 2>&1`
Expected: All K11 and K12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add kernels/paged_attention/12_gqa.cuh kernels/paged_attention/12_gqa.cu tests/test_paged_attention.cu CMakeLists.txt
git commit -m "feat: implement K12 GQA PagedAttention kernel"
```

---

### Task 5: Benchmark

**Files:**
- Create: `benchmarks/paged_attention_bench.cu`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create paged attention benchmark**

```cpp
// benchmarks/paged_attention_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "flash_attention/10_flash_attn_v2.cuh"
#include "paged_attention/11_paged_attn.cuh"
#include "paged_attention/12_gqa.cuh"

// Build paged KV cache from contiguous K/V (for fair comparison with K10)
static void build_paged_kv_cache(
    const float* h_K, const float* h_V,
    int B, int H_kv, int N, int d, int block_size,
    float* h_k_cache, float* h_v_cache,
    int* h_block_table, int* h_context_lens)
{
    int blocks_per_seq = (N + block_size - 1) / block_size;
    for (int b = 0; b < B; ++b) {
        h_context_lens[b] = N;
        for (int blk = 0; blk < blocks_per_seq; ++blk) {
            int phys = b * blocks_per_seq + blk;
            h_block_table[b * blocks_per_seq + blk] = phys;
            for (int t = 0; t < block_size; ++t) {
                int seq_pos = blk * block_size + t;
                for (int h = 0; h < H_kv; ++h) {
                    for (int dd = 0; dd < d; ++dd) {
                        int idx = ((phys * block_size + t) * H_kv + h) * d + dd;
                        if (seq_pos < N) {
                            h_k_cache[idx] = h_K[((b * H_kv + h) * N + seq_pos) * d + dd];
                            h_v_cache[idx] = h_V[((b * H_kv + h) * N + seq_pos) * d + dd];
                        } else {
                            h_k_cache[idx] = 0.0f;
                            h_v_cache[idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int d = 64;
    const int block_size = 16;
    const float peak_bw = 192.0f;

    printf("SLICK PagedAttention + GQA Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 11.8 | FP32\n");
    printf("Block size: %d | Head dim: %d\n", block_size, d);
    printf("================================================\n\n");

    // --- K11 vs K10 benchmark (MHA, GROUP_SIZE=1) ---
    printf("--- K11 PagedAttention vs K10 FlashAttention (causal) ---\n\n");
    printf("%-10s %5s %5s %6s  %10s %10s %10s\n",
           "Config", "B", "H", "N", "K10(us)", "K11(us)", "Overhead");
    printf("---------------------------------------------------------------\n");

    struct MHAConfig { int B, H, N; const char* desc; };
    MHAConfig mha_configs[] = {
        {1, 8,  128, "Small"},
        {1, 8,  256, "Medium"},
        {1, 12, 256, "GPT-2 sm"},
        {1, 12, 512, "GPT-2 md"},
        {2, 8,  256, "Batch=2"},
    };

    for (auto& c : mha_configs) {
        int total_qkv = c.B * c.H * c.N * d;
        int blocks_per_seq = (c.N + block_size - 1) / block_size;
        int num_phys = c.B * blocks_per_seq;
        int cache_size = num_phys * block_size * c.H * d;

        // Host data
        std::vector<float> h_K(total_qkv), h_V(total_qkv);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_bt(c.B * blocks_per_seq), h_cl(c.B);
        srand(42);
        for (int i = 0; i < total_qkv; ++i) {
            h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        build_paged_kv_cache(h_K.data(), h_V.data(), c.B, c.H, c.N, d, block_size,
                             h_k_cache.data(), h_v_cache.data(), h_bt.data(), h_cl.data());

        // Device
        float *d_Q, *d_K, *d_V, *d_O, *d_kc, *d_vc;
        int *d_bt, *d_cl;
        CUDA_CHECK(cudaMalloc(&d_Q, total_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bt, c.B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cl, c.B * sizeof(int)));

        // Init Q same as K for simplicity
        CUDA_CHECK(cudaMemcpy(d_Q, h_K.data(), total_qkv * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total_qkv * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total_qkv * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kc, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vc, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), c.B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cl, h_cl.data(), c.B * sizeof(int), cudaMemcpyHostToDevice));

        // Warmup + bench K10
        for (int w = 0; w < 3; ++w)
            run_flash_attn_v2(c.B, c.H, c.N, d, d_Q, d_K, d_V, d_O, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer timer;
        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_flash_attn_v2(c.B, c.H, c.N, d, d_Q, d_K, d_V, d_O, true);
        float k10_us = timer.toc() / 10.0f * 1000.0f;

        // Warmup + bench K11
        for (int w = 0; w < 3; ++w)
            run_paged_attn(c.B, c.H, c.N, d, d_Q, d_kc, d_vc,
                           d_bt, d_cl, c.N, block_size, blocks_per_seq, d_O, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_paged_attn(c.B, c.H, c.N, d, d_Q, d_kc, d_vc,
                           d_bt, d_cl, c.N, block_size, blocks_per_seq, d_O, true);
        float k11_us = timer.toc() / 10.0f * 1000.0f;

        printf("%-10s %5d %5d %6d  %10.1f %10.1f %9.2fx\n",
               c.desc, c.B, c.H, c.N, k10_us, k11_us, k11_us / k10_us);

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_bt); cudaFree(d_cl);
    }

    // --- K12 GQA sweep ---
    printf("\n--- K12 GQA Sweep (causal, B=1, N=256) ---\n\n");
    printf("%-6s %6s %6s %6s  %10s\n",
           "H_q", "H_kv", "Group", "N", "K12(us)");
    printf("----------------------------------------------\n");

    struct GQAConfig { int H_q, H_kv, N; };
    GQAConfig gqa_configs[] = {
        {8,  8, 256},   // GROUP_SIZE=1 (MHA)
        {8,  4, 256},   // GROUP_SIZE=2
        {8,  2, 256},   // GROUP_SIZE=4
        {8,  1, 256},   // GROUP_SIZE=8 (MQA)
        {16, 4, 256},   // GROUP_SIZE=4
        {32, 4, 256},   // GROUP_SIZE=8
        {32, 4, 512},   // GROUP_SIZE=8, longer seq
    };

    for (auto& c : gqa_configs) {
        int B = 1;
        int group = c.H_q / c.H_kv;
        int total_q = B * c.H_q * c.N * d;
        int blocks_per_seq = (c.N + block_size - 1) / block_size;
        int num_phys = B * blocks_per_seq;
        int cache_size = num_phys * block_size * c.H_kv * d;

        std::vector<float> h_K(B * c.H_kv * c.N * d), h_V(B * c.H_kv * c.N * d);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_bt(B * blocks_per_seq), h_cl(B);
        srand(42);
        for (size_t i = 0; i < h_K.size(); ++i) {
            h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        build_paged_kv_cache(h_K.data(), h_V.data(), B, c.H_kv, c.N, d, block_size,
                             h_k_cache.data(), h_v_cache.data(), h_bt.data(), h_cl.data());

        float *d_Q, *d_O, *d_kc, *d_vc;
        int *d_bt, *d_cl;
        CUDA_CHECK(cudaMalloc(&d_Q, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bt, B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cl, B * sizeof(int)));

        std::vector<float> h_Q(total_q);
        for (int i = 0; i < total_q; ++i)
            h_Q[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_q * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kc, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vc, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cl, h_cl.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        for (int w = 0; w < 3; ++w)
            run_gqa_paged_attn(B, c.H_q, c.H_kv, c.N, d, d_Q, d_kc, d_vc,
                               d_bt, d_cl, c.N, block_size, blocks_per_seq, d_O, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer timer;
        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_gqa_paged_attn(B, c.H_q, c.H_kv, c.N, d, d_Q, d_kc, d_vc,
                               d_bt, d_cl, c.N, block_size, blocks_per_seq, d_O, true);
        float k12_us = timer.toc() / 10.0f * 1000.0f;

        printf("%-6d %6d %6d %6d  %10.1f\n",
               c.H_q, c.H_kv, group, c.N, k12_us);

        cudaFree(d_Q); cudaFree(d_O);
        cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_bt); cudaFree(d_cl);
    }

    return 0;
}
```

- [ ] **Step 2: Add benchmark target to CMakeLists.txt**

```cmake
add_executable(paged_attention_bench benchmarks/paged_attention_bench.cu)
target_link_libraries(paged_attention_bench paged_attention_kernels attention_kernels ${CUBLAS_LIB})
```

- [ ] **Step 3: Build and run benchmark**

Run: `cmake --build build && ./build/paged_attention_bench 2>&1`
Expected: Benchmark runs, K11 overhead vs K10 shown, GQA sweep results printed.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/paged_attention_bench.cu CMakeLists.txt
git commit -m "bench: add K11/K12 PagedAttention + GQA benchmark"
```
