# Week 6: Decode Attention + INT8 GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement K13 (split-K decode attention over paged KV cache) and K14 (INT8 GEMM via dp4a with separate quantization kernel).

**Architecture:** K13 uses a two-pass split-K approach — Pass 1 computes partial attention per KV-chunk in parallel, Pass 2 reduces partial results with online softmax correction. K14 uses NT-layout tiled GEMM with `__dp4a()` for packed int8x4 dot products, preceded by a separate per-row symmetric quantization kernel.

**Tech Stack:** CUDA 11.8, C++17, cuBLAS (cublasGemmEx for INT8 baseline), Google Test, CMake

**Spec:** `docs/superpowers/specs/2026-04-05-week6-decode-attention-int8-gemm-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `kernels/decode/13_decode_attn.cuh` | K13 kernel templates (partial attn + reduction) + host wrapper declaration |
| Create | `kernels/decode/13_decode_attn.cu` | K13 host wrapper implementation |
| Create | `kernels/quantization/14_int8_gemm.cuh` | K14 kernel templates (quantize + int8 gemm) + host wrapper declarations |
| Create | `kernels/quantization/14_int8_gemm.cu` | K14 host wrapper implementations |
| Create | `tests/test_decode_attention.cu` | K13 tests: compare against K11 with N=1 |
| Create | `tests/test_int8_gemm.cu` | K14 tests: compare dequantized output against FP32 GEMM |
| Create | `benchmarks/decode_attention_bench.cu` | K13 benchmark: K13 vs K11(N=1) across configs |
| Create | `benchmarks/int8_gemm_bench.cu` | K14 benchmark: K14 vs cuBLAS cublasGemmEx INT8 |
| Modify | `CMakeLists.txt` | Add decode_kernels, quant_kernels libs, tests, benchmarks |

---

## Task 1: K13 Stub + Build Config

**Files:**
- Create: `kernels/decode/13_decode_attn.cuh`
- Create: `kernels/decode/13_decode_attn.cu`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create K13 header with constants and host wrapper declaration**

```cpp
// kernels/decode/13_decode_attn.cuh
#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ============================================================
// Decode Attention constants
// ============================================================
#define DA_HD 64          // Head dimension
#define DA_NTHREADS 256   // Threads per block
#define DA_MAX_SPLITS 16  // Maximum number of KV splits

// Host wrapper: split-K decode attention over paged KV cache
// Q: [B, H_q, 1, d], O: [B, H_q, 1, d]
// workspace: [B, H_q, max_splits, d+2] (o_partial + m + l per split)
void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,
                     const float* k_cache, const float* v_cache,
                     const int* block_table, const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O,
                     float* workspace);
```

- [ ] **Step 2: Create K13 stub .cu with placeholder host wrapper**

```cpp
// kernels/decode/13_decode_attn.cu
#include "decode/13_decode_attn.cuh"

void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,
                     const float* k_cache, const float* v_cache,
                     const int* block_table, const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O,
                     float* workspace) {
    // TODO: implement split-K decode attention
}
```

- [ ] **Step 3: Add decode_kernels library and test/bench targets to CMakeLists.txt**

Add after the paged_attention_kernels section in `CMakeLists.txt`:

```cmake
# --- Decode Attention Kernels (Week 6) ---
add_library(decode_kernels
    kernels/decode/13_decode_attn.cu
)
target_include_directories(decode_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)
```

Add after the `test_paged_attention` section:

```cmake
add_executable(test_decode_attention tests/test_decode_attention.cu)
target_link_libraries(test_decode_attention GTest::gtest_main decode_kernels paged_attention_kernels attention_kernels)
add_test(NAME DecodeAttentionTests COMMAND test_decode_attention)
```

Add the benchmark target (after paged_attention_bench):

```cmake
# Decode Attention Benchmark
add_executable(decode_attention_bench benchmarks/decode_attention_bench.cu)
target_link_libraries(decode_attention_bench decode_kernels paged_attention_kernels attention_kernels ${CUBLAS_LIB})
```

- [ ] **Step 4: Verify build compiles**

Run: `cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --target decode_kernels`
Expected: Compiles with no errors

- [ ] **Step 5: Commit**

```bash
git add kernels/decode/13_decode_attn.cuh kernels/decode/13_decode_attn.cu CMakeLists.txt
git commit -m "feat: add K13 decode attention stub and build config"
```

---

## Task 2: K13 Test Cases

**Files:**
- Create: `tests/test_decode_attention.cu`

The test reuses `build_paged_kv_cache` from test_paged_attention pattern. K13 with N=1 should match K11 with N=1.

- [ ] **Step 1: Write the test file**

```cpp
// tests/test_decode_attention.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include "timer.cuh"
#include "paged_attention/11_paged_attn.cuh"
#include "decode/13_decode_attn.cuh"

// Helper: build paged KV cache from contiguous K/V [B, H_kv, N, d]
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

class DecodeAttnTest : public ::testing::Test {
protected:
    // Compare K13 decode attention (N=1 query) against K11 paged attention (N=1)
    void RunK13vsK11(int B, int H_q, int H_kv, int ctx_len, int d,
                     int block_size, float tol = 1e-5f) {
        // K13 query: [B, H_q, 1, d] — single token
        int q_size = B * H_q * 1 * d;
        // KV cache: ctx_len tokens
        int blocks_per_seq = (ctx_len + block_size - 1) / block_size;
        int num_phys_blocks = B * blocks_per_seq;
        int cache_size = num_phys_blocks * block_size * H_kv * d;

        // For K11 reference: Q is [B, H_q, 1, d], K/V are [B, H_kv, ctx_len, d]
        int kv_total = B * H_kv * ctx_len * d;

        std::vector<float> h_Q(q_size);
        std::vector<float> h_K(kv_total), h_V(kv_total);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_block_table(B * blocks_per_seq);
        std::vector<int> h_context_lens(B);

        srand(42);
        for (int i = 0; i < q_size; ++i)
            h_Q[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < kv_total; ++i) {
            h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }

        build_paged_kv_cache(h_K.data(), h_V.data(), B, H_kv, ctx_len, d,
                             block_size, h_k_cache.data(), h_v_cache.data(),
                             h_block_table.data(), h_context_lens.data());

        // Device allocations
        float *d_Q, *d_O_k11, *d_O_k13;
        float *d_k_cache, *d_v_cache, *d_workspace;
        int *d_block_table, *d_context_lens;

        CUDA_CHECK(cudaMalloc(&d_Q, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_k11, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_k13, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_table, B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_context_lens, B * sizeof(int)));
        // Workspace for K13: [B, H_q, DA_MAX_SPLITS, d+2]
        int ws_size = B * H_q * DA_MAX_SPLITS * (d + 2);
        CUDA_CHECK(cudaMalloc(&d_workspace, ws_size * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table.data(), B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_context_lens, h_context_lens.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        // K11 reference with N=1 (uses H_q for MHA or H_kv-aware GQA)
        // For GQA, use run_gqa_paged_attn; for MHA (H_q==H_kv), use run_paged_attn
        CUDA_CHECK(cudaMemset(d_O_k11, 0, q_size * sizeof(float)));
        if (H_q == H_kv) {
            run_paged_attn(B, H_q, 1, d, d_Q, d_k_cache, d_v_cache,
                           d_block_table, d_context_lens,
                           ctx_len, block_size, blocks_per_seq, d_O_k11, true);
        } else {
            run_gqa_paged_attn(B, H_q, H_kv, 1, d, d_Q, d_k_cache, d_v_cache,
                               d_block_table, d_context_lens,
                               ctx_len, block_size, blocks_per_seq, d_O_k11, true);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // K13
        CUDA_CHECK(cudaMemset(d_O_k13, 0, q_size * sizeof(float)));
        run_decode_attn(B, H_q, H_kv, d, d_Q, d_k_cache, d_v_cache,
                        d_block_table, d_context_lens,
                        ctx_len, block_size, blocks_per_seq,
                        d_O_k13, d_workspace);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compare
        std::vector<float> h_O_k11(q_size), h_O_k13(q_size);
        CUDA_CHECK(cudaMemcpy(h_O_k11.data(), d_O_k11, q_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_O_k13.data(), d_O_k13, q_size * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < q_size; ++i) {
            float err = fabsf(h_O_k11[i] - h_O_k13[i]);
            if (err > max_err) max_err = err;
        }

        EXPECT_LT(max_err, tol) << "K13 vs K11 max error: " << max_err
                                 << " (B=" << B << " H_q=" << H_q
                                 << " H_kv=" << H_kv << " ctx=" << ctx_len << ")";

        cudaFree(d_Q); cudaFree(d_O_k11); cudaFree(d_O_k13);
        cudaFree(d_k_cache); cudaFree(d_v_cache);
        cudaFree(d_block_table); cudaFree(d_context_lens);
        cudaFree(d_workspace);
    }
};

// --- K13 MHA tests (H_q == H_kv, GROUP_SIZE=1) ---
TEST_F(DecodeAttnTest, K13_MHA_Small)     { RunK13vsK11(1, 8, 8, 128, 64, 16); }
TEST_F(DecodeAttnTest, K13_MHA_Medium)    { RunK13vsK11(1, 16, 16, 512, 64, 16); }
TEST_F(DecodeAttnTest, K13_MHA_Batch)     { RunK13vsK11(4, 8, 8, 256, 64, 16); }
TEST_F(DecodeAttnTest, K13_MHA_Long)      { RunK13vsK11(1, 8, 8, 1024, 64, 16); }

// --- K13 GQA tests (H_q != H_kv) ---
TEST_F(DecodeAttnTest, K13_GQA_Group4)    { RunK13vsK11(1, 16, 4, 256, 64, 16); }
TEST_F(DecodeAttnTest, K13_GQA_Group4_Lg) { RunK13vsK11(8, 32, 8, 1024, 64, 16); }
```

- [ ] **Step 2: Build and verify tests fail (stub returns zeros)**

Run: `cmake --build build --target test_decode_attention && ./build/test_decode_attention`
Expected: All 6 tests FAIL (K13 output is all zeros, K11 output is non-zero)

- [ ] **Step 3: Commit**

```bash
git add tests/test_decode_attention.cu
git commit -m "test: add K13 decode attention test cases"
```

---

## Task 3: K13 Partial Attention Kernel (Pass 1)

**Files:**
- Modify: `kernels/decode/13_decode_attn.cuh`

This is the core kernel. Each threadblock handles one split of the KV sequence for one (batch, head) pair. Within the block, warps cooperatively process KV tokens sequentially, computing dot products across the d dimension in parallel.

- [ ] **Step 1: Add the partial attention kernel to the header**

Replace the contents of `kernels/decode/13_decode_attn.cuh` with:

```cpp
// kernels/decode/13_decode_attn.cuh
#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ============================================================
// Decode Attention constants
// ============================================================
#define DA_HD 64          // Head dimension
#define DA_NTHREADS 256   // Threads per block (8 warps)
#define DA_MAX_SPLITS 16  // Maximum number of KV splits
#define DA_WARP_SIZE 32

// ============================================================
// Pass 1: Partial attention kernel
// Each block processes a range of KV blocks for one (batch, head) pair.
// Grid: (B, H_q, num_splits)
//
// For each KV token in its assigned range:
//   s = dot(q, k) * scale  (warp-parallel across d)
//   online softmax: track m_partial, l_partial
//   o_partial += exp(s - m) * v
//
// Output to workspace: [B, H_q, num_splits, d+2]
//   workspace[...][0..d-1] = o_partial (unnormalized)
//   workspace[...][d]      = m_partial (max score)
//   workspace[...][d+1]    = l_partial (sum of exp)
// ============================================================
template <int GROUP_SIZE>
__global__ __launch_bounds__(DA_NTHREADS)
void decode_attn_partial_kernel(
    int d, float scale,
    const float* __restrict__ Q,           // [B, H_q, 1, d]
    const float* __restrict__ k_cache,     // [num_phys_blocks, block_size, H_kv, d]
    const float* __restrict__ v_cache,     // [num_phys_blocks, block_size, H_kv, d]
    const int* __restrict__ block_table,   // [B, num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [B]
    int block_size, int num_blocks_per_seq,
    int H_kv, int num_splits, int blocks_per_split,
    float* __restrict__ workspace)         // [B, H_q, num_splits, d+2]
{
    const int batch  = blockIdx.x;
    const int q_head = blockIdx.y;
    const int split  = blockIdx.z;
    const int H_q    = gridDim.y;
    const int kv_head = q_head / GROUP_SIZE;
    const int tid = threadIdx.x;

    const int ctx_len = context_lens[batch];
    const int num_kv_blocks = (ctx_len + block_size - 1) / block_size;

    // This split's KV block range
    const int blk_start = split * blocks_per_split;
    const int blk_end = min(blk_start + blocks_per_split, num_kv_blocks);
    if (blk_start >= num_kv_blocks) return;

    const int* bt = block_table + batch * num_blocks_per_seq;

    // Load query vector into shared memory [d] (float4 vectorized)
    __shared__ float q_smem[DA_HD];
    const float* Q_bh = Q + (static_cast<long long>(batch) * H_q + q_head) * d;
    for (int i = tid; i < d / 4; i += DA_NTHREADS) {
        float4 val = *reinterpret_cast<const float4*>(&Q_bh[i * 4]);
        q_smem[i * 4 + 0] = val.x;
        q_smem[i * 4 + 1] = val.y;
        q_smem[i * 4 + 2] = val.z;
        q_smem[i * 4 + 3] = val.w;
    }
    __syncthreads();

    // Each thread accumulates partial o[d] — thread `tid` owns elements
    // tid % (d) maps to a dimension. With 256 threads and d=64, each dim
    // has 4 threads that will reduce later. But simpler: each thread tracks
    // all d elements and we assign KV tokens round-robin across warps.
    //
    // Better approach for GEMV: each warp handles different KV tokens,
    // all threads in warp cooperate on the dot product across d.
    // With 8 warps, 8 KV tokens processed in parallel per iteration.
    const int warp_id = tid / DA_WARP_SIZE;
    const int lane_id = tid % DA_WARP_SIZE;
    const int num_warps = DA_NTHREADS / DA_WARP_SIZE;  // 8

    // Per-warp accumulators (in registers)
    float m_acc = -FLT_MAX;  // running max
    float l_acc = 0.0f;       // running sum of exp
    float o_acc[DA_HD / DA_WARP_SIZE];  // each lane owns d/32 = 2 elements (for d=64)
    const int elems_per_lane = d / DA_WARP_SIZE;  // 64/32 = 2

    #pragma unroll
    for (int e = 0; e < elems_per_lane; ++e)
        o_acc[e] = 0.0f;

    // Iterate over KV tokens in this split's range
    // Each warp takes every num_warps-th token
    for (int blk_idx = blk_start; blk_idx < blk_end; ++blk_idx) {
        int phys_block = bt[blk_idx];
        int tokens_in_block = min(block_size, ctx_len - blk_idx * block_size);

        for (int t = warp_id; t < tokens_in_block; t += num_warps) {
            int seq_pos = blk_idx * block_size + t;

            // Compute dot(q, k) across d — each lane handles elems_per_lane elements
            int cache_base = ((phys_block * block_size + t) * H_kv + kv_head) * d;

            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < elems_per_lane; ++e) {
                int dim = lane_id * elems_per_lane + e;
                dot += q_smem[dim] * k_cache[cache_base + dim];
            }

            // Warp-level reduction for dot product
            #pragma unroll
            for (int offset = DA_WARP_SIZE / 2; offset >= 1; offset >>= 1)
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            float s = __shfl_sync(0xFFFFFFFF, dot, 0) * scale;

            // Online softmax update
            float m_new = fmaxf(m_acc, s);
            float alpha = __expf(m_acc - m_new);
            float p = __expf(s - m_new);

            // Rescale existing accumulator
            #pragma unroll
            for (int e = 0; e < elems_per_lane; ++e)
                o_acc[e] *= alpha;
            l_acc = l_acc * alpha + p;
            m_acc = m_new;

            // Accumulate p * v
            #pragma unroll
            for (int e = 0; e < elems_per_lane; ++e) {
                int dim = lane_id * elems_per_lane + e;
                o_acc[e] += p * v_cache[cache_base + dim];
            }
        }
    }

    // Reduce across warps in shared memory
    // Each warp writes its (o_acc[], m_acc, l_acc) to smem, then warp 0 merges
    __shared__ float warp_m[8];       // max per warp
    __shared__ float warp_l[8];       // sum per warp
    __shared__ float warp_o[8][DA_HD]; // partial o per warp

    // Each lane writes its owned elements
    #pragma unroll
    for (int e = 0; e < elems_per_lane; ++e)
        warp_o[warp_id][lane_id * elems_per_lane + e] = o_acc[e];

    if (lane_id == 0) {
        warp_m[warp_id] = m_acc;
        warp_l[warp_id] = l_acc;
    }
    __syncthreads();

    // Warp 0 merges all warp results
    if (warp_id == 0) {
        float merged_m = warp_m[0];
        float merged_l = warp_l[0];
        float merged_o[DA_HD / DA_WARP_SIZE];

        #pragma unroll
        for (int e = 0; e < elems_per_lane; ++e)
            merged_o[e] = warp_o[0][lane_id * elems_per_lane + e];

        for (int w = 1; w < num_warps; ++w) {
            float w_m = warp_m[w];
            float w_l = warp_l[w];
            if (w_l == 0.0f) continue;  // warp processed no tokens

            float m_new = fmaxf(merged_m, w_m);
            float alpha = __expf(merged_m - m_new);
            float beta  = __expf(w_m - m_new);

            #pragma unroll
            for (int e = 0; e < elems_per_lane; ++e)
                merged_o[e] = merged_o[e] * alpha
                            + warp_o[w][lane_id * elems_per_lane + e] * beta;

            merged_l = merged_l * alpha + w_l * beta;
            merged_m = m_new;
        }

        // Write to workspace: [B, H_q, num_splits, d+2]
        int ws_offset = ((batch * H_q + q_head) * num_splits + split) * (d + 2);
        #pragma unroll
        for (int e = 0; e < elems_per_lane; ++e)
            workspace[ws_offset + lane_id * elems_per_lane + e] = merged_o[e];

        if (lane_id == 0) {
            workspace[ws_offset + d]     = merged_m;
            workspace[ws_offset + d + 1] = merged_l;
        }
    }
}

// ============================================================
// Pass 2: Reduction kernel
// Merges partial results from all splits using online softmax correction.
// Grid: (B * H_q), one block per (batch, head) pair
// ============================================================
__global__ __launch_bounds__(DA_NTHREADS)
void decode_attn_reduce_kernel(
    int d, int H_q, int num_splits,
    const float* __restrict__ workspace,   // [B, H_q, num_splits, d+2]
    float* __restrict__ O)                 // [B, H_q, 1, d]
{
    const int bh = blockIdx.x;             // flattened (batch, head)
    const int batch = bh / H_q;
    const int head  = bh % H_q;
    const int tid = threadIdx.x;

    if (tid >= d) return;  // only first d threads do work

    const int stride = d + 2;
    int ws_base = (static_cast<long long>(batch) * H_q + head) * num_splits * stride;

    // Read first split
    float m_acc = workspace[ws_base + d];
    float l_acc = workspace[ws_base + d + 1];
    float o_acc = workspace[ws_base + tid];

    // Merge remaining splits
    for (int s = 1; s < num_splits; ++s) {
        int offset = ws_base + s * stride;
        float m_s = workspace[offset + d];
        float l_s = workspace[offset + d + 1];
        float o_s = workspace[offset + tid];

        if (l_s == 0.0f) continue;  // empty split

        float m_new = fmaxf(m_acc, m_s);
        float alpha = __expf(m_acc - m_new);
        float beta  = __expf(m_s - m_new);

        o_acc = o_acc * alpha + o_s * beta;
        l_acc = l_acc * alpha + l_s * beta;
        m_acc = m_new;
    }

    // Finalize: normalize by l
    float inv_l = (l_acc > 0.0f) ? 1.0f / l_acc : 0.0f;
    int o_offset = (static_cast<long long>(batch) * H_q + head) * d;
    O[o_offset + tid] = o_acc * inv_l;
}

// Host wrapper declaration
void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,
                     const float* k_cache, const float* v_cache,
                     const int* block_table, const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O,
                     float* workspace);
```

- [ ] **Step 2: Verify it compiles**

Run: `cmake --build build --target decode_kernels`
Expected: Compiles with no errors

- [ ] **Step 3: Commit**

```bash
git add kernels/decode/13_decode_attn.cuh
git commit -m "feat: implement K13 partial attention + reduction kernels"
```

---

## Task 4: K13 Host Wrapper

**Files:**
- Modify: `kernels/decode/13_decode_attn.cu`

- [ ] **Step 1: Implement the host wrapper with split count heuristic and GQA dispatch**

Replace `kernels/decode/13_decode_attn.cu` with:

```cpp
// kernels/decode/13_decode_attn.cu
#include "decode/13_decode_attn.cuh"
#include <algorithm>

template <int GROUP_SIZE>
static void launch_decode_attn(int B, int H_q, int H_kv, int d, float scale,
                               const float* Q,
                               const float* k_cache, const float* v_cache,
                               const int* block_table, const int* context_lens,
                               int max_context_len, int block_size,
                               int num_blocks_per_seq,
                               float* O, float* workspace) {
    int max_kv_blocks = (max_context_len + block_size - 1) / block_size;
    int num_splits = std::clamp(max_kv_blocks / 4, 1, DA_MAX_SPLITS);
    int blocks_per_split = (max_kv_blocks + num_splits - 1) / num_splits;

    // Pass 1: partial attention
    dim3 grid1(B, H_q, num_splits);
    dim3 block1(DA_NTHREADS);
    decode_attn_partial_kernel<GROUP_SIZE><<<grid1, block1>>>(
        d, scale, Q, k_cache, v_cache,
        block_table, context_lens,
        block_size, num_blocks_per_seq,
        H_kv, num_splits, blocks_per_split,
        workspace);

    // Pass 2: reduction
    dim3 grid2(B * H_q);
    dim3 block2(DA_NTHREADS);
    decode_attn_reduce_kernel<<<grid2, block2>>>(
        d, H_q, num_splits, workspace, O);
}

void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,
                     const float* k_cache, const float* v_cache,
                     const int* block_table, const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O, float* workspace) {
    float scale = 1.0f / sqrtf((float)d);
    int group_size = H_q / H_kv;

    switch (group_size) {
        case 1: launch_decode_attn<1>(B, H_q, H_kv, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, workspace); break;
        case 2: launch_decode_attn<2>(B, H_q, H_kv, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, workspace); break;
        case 4: launch_decode_attn<4>(B, H_q, H_kv, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, workspace); break;
        case 8: launch_decode_attn<8>(B, H_q, H_kv, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, workspace); break;
        default: break;
    }
}
```

- [ ] **Step 2: Build and run tests**

Run: `cmake --build build --target test_decode_attention && ./build/test_decode_attention`
Expected: All 6 tests PASS

- [ ] **Step 3: Commit**

```bash
git add kernels/decode/13_decode_attn.cu
git commit -m "feat: implement K13 decode attention host wrapper"
```

---

## Task 5: K13 Benchmark

**Files:**
- Create: `benchmarks/decode_attention_bench.cu`

Benchmark K13 vs K11(N=1) across batch sizes and context lengths.

- [ ] **Step 1: Write the benchmark**

```cpp
// benchmarks/decode_attention_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "paged_attention/11_paged_attn.cuh"
#include "paged_attention/12_gqa.cuh"
#include "decode/13_decode_attn.cuh"

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

    printf("SLICK Decode Attention Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 11.8 | FP32\n");
    printf("Block size: %d | Head dim: %d\n", block_size, d);
    printf("================================================\n\n");

    // --- K13 vs K11 (N=1) ---
    printf("--- K13 Decode Attention vs K11 PagedAttention (N=1, causal) ---\n\n");
    printf("%-6s %5s %5s %6s  %10s %10s %10s\n",
           "Config", "B", "H", "ctx", "K11(us)", "K13(us)", "Speedup");
    printf("--------------------------------------------------------------\n");

    struct Config { int B, H_q, H_kv, ctx; const char* desc; };
    Config configs[] = {
        {1,  8,  8,  128,  "Small"},
        {1,  8,  8,  256,  "Med"},
        {1,  8,  8,  512,  "Large"},
        {1,  8,  8,  1024, "XL"},
        {4,  8,  8,  256,  "Bat=4"},
        {8,  8,  8,  512,  "Bat=8"},
        {1,  16, 4,  256,  "GQA4"},
        {1,  32, 8,  512,  "GQA4L"},
    };

    for (auto& c : configs) {
        int q_size = c.B * c.H_q * 1 * d;
        int kv_total = c.B * c.H_kv * c.ctx * d;
        int blocks_per_seq = (c.ctx + block_size - 1) / block_size;
        int num_phys = c.B * blocks_per_seq;
        int cache_size = num_phys * block_size * c.H_kv * d;
        int ws_size = c.B * c.H_q * DA_MAX_SPLITS * (d + 2);

        std::vector<float> h_Q(q_size), h_K(kv_total), h_V(kv_total);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_bt(c.B * blocks_per_seq), h_cl(c.B);

        srand(42);
        for (int i = 0; i < q_size; ++i)
            h_Q[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < kv_total; ++i) {
            h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        build_paged_kv_cache(h_K.data(), h_V.data(), c.B, c.H_kv, c.ctx, d,
                             block_size, h_k_cache.data(), h_v_cache.data(),
                             h_bt.data(), h_cl.data());

        float *d_Q, *d_O, *d_kc, *d_vc, *d_ws;
        int *d_bt, *d_cl;
        CUDA_CHECK(cudaMalloc(&d_Q, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ws, ws_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bt, c.B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cl, c.B * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kc, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vc, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), c.B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cl, h_cl.data(), c.B * sizeof(int), cudaMemcpyHostToDevice));

        // K11 with N=1
        auto bench_k11 = [&]() {
            if (c.H_q == c.H_kv)
                run_paged_attn(c.B, c.H_q, 1, d, d_Q, d_kc, d_vc,
                               d_bt, d_cl, c.ctx, block_size, blocks_per_seq, d_O, true);
            else
                run_gqa_paged_attn(c.B, c.H_q, c.H_kv, 1, d, d_Q, d_kc, d_vc,
                                   d_bt, d_cl, c.ctx, block_size, blocks_per_seq, d_O, true);
        };

        for (int w = 0; w < 3; ++w) bench_k11();
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer timer;
        timer.tic();
        for (int r = 0; r < 20; ++r) bench_k11();
        float k11_us = timer.toc() / 20.0f * 1000.0f;

        // K13
        for (int w = 0; w < 3; ++w)
            run_decode_attn(c.B, c.H_q, c.H_kv, d, d_Q, d_kc, d_vc,
                            d_bt, d_cl, c.ctx, block_size, blocks_per_seq, d_O, d_ws);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.tic();
        for (int r = 0; r < 20; ++r)
            run_decode_attn(c.B, c.H_q, c.H_kv, d, d_Q, d_kc, d_vc,
                            d_bt, d_cl, c.ctx, block_size, blocks_per_seq, d_O, d_ws);
        float k13_us = timer.toc() / 20.0f * 1000.0f;

        printf("%-6s %5d %5d %6d  %10.1f %10.1f %9.2fx\n",
               c.desc, c.B, c.H_q, c.ctx, k11_us, k13_us, k11_us / k13_us);

        cudaFree(d_Q); cudaFree(d_O);
        cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_ws);
        cudaFree(d_bt); cudaFree(d_cl);
    }

    printf("\nK13 uses split-K parallelism: multiple blocks per (batch, head)\n");
    printf("K11 uses single block per (batch, head) with N=1 (Br=64 tile mostly empty)\n");

    return 0;
}
```

- [ ] **Step 2: Build and run benchmark**

Run: `cmake --build build --target decode_attention_bench && ./build/decode_attention_bench`
Expected: Table of K11 vs K13 timings printed. K13 should be faster for longer contexts.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/decode_attention_bench.cu
git commit -m "bench: add K13 decode attention benchmark"
```

---

## Task 6: K14 Stub + Build Config

**Files:**
- Create: `kernels/quantization/14_int8_gemm.cuh`
- Create: `kernels/quantization/14_int8_gemm.cu`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create K14 header with constants and host wrapper declarations**

```cpp
// kernels/quantization/14_int8_gemm.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ============================================================
// INT8 GEMM constants
// ============================================================
#define I8_BM 64       // Tile rows
#define I8_BN 64       // Tile cols
#define I8_BK 16       // Tile K (in int8 elements)
#define I8_TM 4        // Thread tile rows
#define I8_TN 4        // Thread tile cols
#define I8_NTHREADS 256 // (BM/TM) * (BN/TN) = 16*16 = 256

// Quantize FP32 matrix to INT8 (packed int32) with per-row scales
// input: [rows, cols], output_packed: [rows, cols/4] as int32, scales: [rows]
void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,
                                int32_t* output_packed,
                                float* scales);

// INT8 GEMM: C_fp32 = dequant(A_int8 @ B_int8^T)
// A_packed: [M, K/4], BT_packed: [N, K/4] (both int8x4 packed as int32)
// scale_A: [M], scale_B: [N], C: [M, N] fp32
void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C);
```

- [ ] **Step 2: Create K14 stub .cu**

```cpp
// kernels/quantization/14_int8_gemm.cu
#include "quantization/14_int8_gemm.cuh"

void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,
                                int32_t* output_packed,
                                float* scales) {
    // TODO: implement quantization kernel
}

void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C) {
    // TODO: implement int8 gemm kernel
}
```

- [ ] **Step 3: Add quant_kernels library and test/bench targets to CMakeLists.txt**

Add after the decode_kernels section:

```cmake
# --- Quantization Kernels (Week 6) ---
add_library(quant_kernels
    kernels/quantization/14_int8_gemm.cu
)
target_include_directories(quant_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)
```

Add after `test_decode_attention`:

```cmake
add_executable(test_int8_gemm tests/test_int8_gemm.cu)
target_link_libraries(test_int8_gemm GTest::gtest_main quant_kernels gemm_kernels ${CUBLAS_LIB})
add_test(NAME Int8GemmTests COMMAND test_int8_gemm)
```

Add after `decode_attention_bench`:

```cmake
# INT8 GEMM Benchmark
add_executable(int8_gemm_bench benchmarks/int8_gemm_bench.cu)
target_link_libraries(int8_gemm_bench quant_kernels ${CUBLAS_LIB})
```

- [ ] **Step 4: Verify build compiles**

Run: `cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --target quant_kernels`
Expected: Compiles with no errors

- [ ] **Step 5: Commit**

```bash
git add kernels/quantization/14_int8_gemm.cuh kernels/quantization/14_int8_gemm.cu CMakeLists.txt
git commit -m "feat: add K14 INT8 GEMM stub and build config"
```

---

## Task 7: K14 Test Cases

**Files:**
- Create: `tests/test_int8_gemm.cu`

Tests: quantize A and B, run int8 gemm, compare dequantized output against FP32 GEMM (K06).

- [ ] **Step 1: Write the test file**

```cpp
// tests/test_int8_gemm.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "timer.cuh"
#include "validator.cuh"
#include "gemm/06_vectorized.cuh"
#include "quantization/14_int8_gemm.cuh"

class Int8GemmTest : public ::testing::Test {
protected:
    void RunInt8vsFloat(int M, int N, int K, float tol = 0.05f) {
        int sizeA = M * K;
        int sizeB = K * N;
        int sizeC = M * N;

        // Host FP32 matrices
        std::vector<float> h_A(sizeA), h_B(sizeB);
        srand(42);
        for (int i = 0; i < sizeA; ++i)
            h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < sizeB; ++i)
            h_B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        // Device FP32
        float *d_A, *d_B, *d_C_ref, *d_C_int8;
        CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_ref, sizeC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_int8, sizeC * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

        // FP32 reference (K06)
        run_sgemm_vectorized(M, N, K, d_A, d_B, d_C_ref);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Quantize A [M, K] and B^T [N, K]
        // B is [K, N] row-major. B^T is [N, K] row-major.
        // We need to transpose B on host before quantizing.
        std::vector<float> h_BT(N * K);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                h_BT[n * K + k] = h_B[k * N + n];

        float *d_BT;
        CUDA_CHECK(cudaMalloc(&d_BT, N * K * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_BT, h_BT.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

        // Quantized buffers
        int32_t *d_A_packed, *d_BT_packed;
        float *d_scale_A, *d_scale_B;
        CUDA_CHECK(cudaMalloc(&d_A_packed, M * (K / 4) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_BT_packed, N * (K / 4) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_scale_A, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_B, N * sizeof(float)));

        // Quantize
        run_quantize_fp32_to_int8(M, K, d_A, d_A_packed, d_scale_A);
        run_quantize_fp32_to_int8(N, K, d_BT, d_BT_packed, d_scale_B);
        CUDA_CHECK(cudaDeviceSynchronize());

        // INT8 GEMM
        run_int8_gemm(M, N, K, d_A_packed, d_BT_packed, d_scale_A, d_scale_B, d_C_int8);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compare
        std::vector<float> h_C_ref(sizeC), h_C_int8(sizeC);
        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_C_int8.data(), d_C_int8, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        float sum_err = 0.0f;
        for (int i = 0; i < sizeC; ++i) {
            float err = fabsf(h_C_ref[i] - h_C_int8[i]);
            sum_err += err;
            if (err > max_err) max_err = err;
        }
        float avg_err = sum_err / sizeC;

        EXPECT_LT(max_err, tol) << "INT8 vs FP32 max error: " << max_err
                                 << " avg: " << avg_err
                                 << " (M=" << M << " N=" << N << " K=" << K << ")";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_BT);
        cudaFree(d_C_ref); cudaFree(d_C_int8);
        cudaFree(d_A_packed); cudaFree(d_BT_packed);
        cudaFree(d_scale_A); cudaFree(d_scale_B);
    }
};

TEST_F(Int8GemmTest, K14_256)  { RunInt8vsFloat(256, 256, 256); }
TEST_F(Int8GemmTest, K14_512)  { RunInt8vsFloat(512, 512, 512); }
TEST_F(Int8GemmTest, K14_1024) { RunInt8vsFloat(1024, 1024, 1024); }
TEST_F(Int8GemmTest, K14_2048) { RunInt8vsFloat(2048, 2048, 2048); }
TEST_F(Int8GemmTest, K14_Rect) { RunInt8vsFloat(512, 1024, 256); }
```

- [ ] **Step 2: Build and verify tests fail (stubs return zeros)**

Run: `cmake --build build --target test_int8_gemm && ./build/test_int8_gemm`
Expected: All 5 tests FAIL (INT8 output is all zeros)

- [ ] **Step 3: Commit**

```bash
git add tests/test_int8_gemm.cu
git commit -m "test: add K14 INT8 GEMM test cases"
```

---

## Task 8: K14 Quantization Kernel

**Files:**
- Modify: `kernels/quantization/14_int8_gemm.cuh` (add quantize kernel)
- Modify: `kernels/quantization/14_int8_gemm.cu` (implement host wrapper)

- [ ] **Step 1: Add the quantization kernel to the header**

Replace `kernels/quantization/14_int8_gemm.cuh` with:

```cpp
// kernels/quantization/14_int8_gemm.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

// ============================================================
// INT8 GEMM constants
// ============================================================
#define I8_BM 64       // Tile rows
#define I8_BN 64       // Tile cols
#define I8_BK 16       // Tile K (in int8 elements)
#define I8_TM 4        // Thread tile rows
#define I8_TN 4        // Thread tile cols
#define I8_NTHREADS 256 // (BM/TM) * (BN/TN) = 16*16 = 256

// ============================================================
// Quantization kernel: FP32 -> INT8 (per-row symmetric)
// One block per row. Threads cooperatively:
//   Phase 1: find max(|row|) via shared memory reduction
//   Phase 2: quantize + pack 4 int8s into int32
//
// input: [rows, cols] row-major FP32
// output_packed: [rows, cols/4] int8x4 packed as int32
// scales: [rows] float
// ============================================================
__global__ void quantize_fp32_to_int8_kernel(
    int cols,
    const float* __restrict__ input,     // [rows, cols]
    int32_t* __restrict__ output_packed, // [rows, cols/4]
    float* __restrict__ scales)          // [rows]
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_ptr = input + row * cols;

    // Phase 1: cooperative max-abs reduction
    extern __shared__ float smem[];  // blockDim.x floats

    float local_max = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = fabsf(row_ptr[i]);
        if (val > local_max) local_max = val;
    }
    smem[tid] = local_max;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    float row_max = smem[0];
    float scale = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0) scales[row] = scale;
    __syncthreads();

    // Phase 2: quantize and pack 4 int8s into int32
    int packed_cols = cols / 4;
    int32_t* out_row = output_packed + row * packed_cols;

    for (int i = tid; i < packed_cols; i += blockDim.x) {
        int base = i * 4;
        int8_t q0 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 0] * inv_scale), -128.0f), 127.0f));
        int8_t q1 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 1] * inv_scale), -128.0f), 127.0f));
        int8_t q2 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 2] * inv_scale), -128.0f), 127.0f));
        int8_t q3 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 3] * inv_scale), -128.0f), 127.0f));

        // Pack: byte 0 = q0, byte 1 = q1, byte 2 = q2, byte 3 = q3
        int32_t packed = 0;
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q0)));
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);
        out_row[i] = packed;
    }
}

// ============================================================
// INT8 GEMM kernel (dp4a) — placeholder, implemented in Task 9
// ============================================================

// Host wrapper declarations
void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,
                                int32_t* output_packed,
                                float* scales);

void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C);
```

- [ ] **Step 2: Implement the quantize host wrapper in .cu**

Replace `kernels/quantization/14_int8_gemm.cu` with:

```cpp
// kernels/quantization/14_int8_gemm.cu
#include "quantization/14_int8_gemm.cuh"

void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,
                                int32_t* output_packed,
                                float* scales) {
    int threads = 256;
    int smem = threads * sizeof(float);
    quantize_fp32_to_int8_kernel<<<rows, threads, smem>>>(
        cols, input, output_packed, scales);
}

void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C) {
    // TODO: implement dp4a GEMM kernel
}
```

- [ ] **Step 3: Build and verify compiles**

Run: `cmake --build build --target quant_kernels`
Expected: Compiles with no errors

- [ ] **Step 4: Commit**

```bash
git add kernels/quantization/14_int8_gemm.cuh kernels/quantization/14_int8_gemm.cu
git commit -m "feat: implement K14 quantization kernel (FP32 to INT8)"
```

---

## Task 9: K14 INT8 GEMM Kernel (dp4a)

**Files:**
- Modify: `kernels/quantization/14_int8_gemm.cuh` (add GEMM kernel)
- Modify: `kernels/quantization/14_int8_gemm.cu` (implement host wrapper)

- [ ] **Step 1: Add the INT8 GEMM kernel to the header**

Add the following kernel template after the quantize kernel in `14_int8_gemm.cuh`, replacing the placeholder comment:

```cpp
// ============================================================
// INT8 GEMM kernel via __dp4a()
// NT layout: A [M, K/4] and B^T [N, K/4], both int8x4 packed as int32
// Tiled: BM=64, BN=64, BK=16 (int8 elements) = 4 int32 packed values
// Register tiling: TM=4, TN=4 — each thread owns 4x4 int32 accumulators
//
// __dp4a(a, b, c): treats a,b as packed int8x4
//   c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
//
// Epilogue: dequantize with per-row scales
//   C_fp32[i][j] = C_int32[i][j] * scale_A[i] * scale_B[j]
// ============================================================
__global__ __launch_bounds__(I8_NTHREADS)
void int8_gemm_dp4a_kernel(
    int M, int N, int K,
    const int32_t* __restrict__ A_packed,   // [M, K/4]
    const int32_t* __restrict__ BT_packed,  // [N, K/4]
    const float* __restrict__ scale_A,      // [M]
    const float* __restrict__ scale_B,      // [N]
    float* __restrict__ C)                  // [M, N]
{
    const int bm = blockIdx.y;   // tile row
    const int bn = blockIdx.x;   // tile col
    const int tid = threadIdx.x;
    const int thread_row = tid / (I8_BN / I8_TN);  // 0..15
    const int thread_col = tid % (I8_BN / I8_TN);  // 0..15

    const int K4 = K / 4;     // packed dimension
    const int BK4 = I8_BK / 4; // 4 int32s per K-tile

    // Shared memory for A tile [BM][BK/4] and B^T tile [BN][BK/4]
    // +1 padding to avoid bank conflicts
    __shared__ int32_t A_smem[I8_BM][BK4 + 1];
    __shared__ int32_t BT_smem[I8_BN][BK4 + 1];

    // Register accumulators: TM x TN = 4x4 int32
    int32_t acc[I8_TM][I8_TN];
    #pragma unroll
    for (int tm = 0; tm < I8_TM; ++tm)
        #pragma unroll
        for (int tn = 0; tn < I8_TN; ++tn)
            acc[tm][tn] = 0;

    // Tile loop over K dimension
    for (int k_tile = 0; k_tile < K4; k_tile += BK4) {

        // Cooperative load: A tile [BM][BK4]
        // Total elements = BM * BK4 = 64 * 4 = 256 = NTHREADS (one element each)
        {
            int load_idx = tid;  // tid in [0, 255]
            int r = load_idx / BK4;       // 0..63
            int c = load_idx % BK4;       // 0..3
            int global_row = bm * I8_BM + r;
            int global_col = k_tile + c;
            if (global_row < M && global_col < K4)
                A_smem[r][c] = A_packed[global_row * K4 + global_col];
            else
                A_smem[r][c] = 0;
        }

        // Cooperative load: B^T tile [BN][BK4]
        {
            int load_idx = tid;
            int r = load_idx / BK4;
            int c = load_idx % BK4;
            int global_row = bn * I8_BN + r;
            int global_col = k_tile + c;
            if (global_row < N && global_col < K4)
                BT_smem[r][c] = BT_packed[global_row * K4 + global_col];
            else
                BT_smem[r][c] = 0;
        }

        __syncthreads();

        // Compute: dp4a over BK4 packed int32 values
        #pragma unroll
        for (int k4 = 0; k4 < BK4; ++k4) {
            // Load A fragments: TM rows
            int32_t a_frag[I8_TM];
            #pragma unroll
            for (int tm = 0; tm < I8_TM; ++tm)
                a_frag[tm] = A_smem[thread_row * I8_TM + tm][k4];

            // Load B^T fragments: TN rows
            int32_t b_frag[I8_TN];
            #pragma unroll
            for (int tn = 0; tn < I8_TN; ++tn)
                b_frag[tn] = BT_smem[thread_col * I8_TN + tn][k4];

            // dp4a: 4 int8 MADs per call
            #pragma unroll
            for (int tm = 0; tm < I8_TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < I8_TN; ++tn)
                    acc[tm][tn] = __dp4a(a_frag[tm], b_frag[tn], acc[tm][tn]);
        }

        __syncthreads();
    }

    // Epilogue: dequantize and write C
    #pragma unroll
    for (int tm = 0; tm < I8_TM; ++tm) {
        int gr = bm * I8_BM + thread_row * I8_TM + tm;
        if (gr >= M) continue;
        float sa = scale_A[gr];

        #pragma unroll
        for (int tn = 0; tn < I8_TN; ++tn) {
            int gc = bn * I8_BN + thread_col * I8_TN + tn;
            if (gc >= N) continue;
            float sb = scale_B[gc];
            C[gr * N + gc] = static_cast<float>(acc[tm][tn]) * sa * sb;
        }
    }
}
```

- [ ] **Step 2: Implement the GEMM host wrapper**

Replace the `run_int8_gemm` stub in `kernels/quantization/14_int8_gemm.cu`:

```cpp
void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C) {
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    dim3 block(I8_NTHREADS);
    int8_gemm_dp4a_kernel<<<grid, block>>>(
        M, N, K, A_packed, BT_packed, scale_A, scale_B, C);
}
```

- [ ] **Step 3: Build and run tests**

Run: `cmake --build build --target test_int8_gemm && ./build/test_int8_gemm`
Expected: All 5 tests PASS (max error < 0.05)

- [ ] **Step 4: Commit**

```bash
git add kernels/quantization/14_int8_gemm.cuh kernels/quantization/14_int8_gemm.cu
git commit -m "feat: implement K14 INT8 GEMM kernel via dp4a"
```

---

## Task 10: K14 Benchmark (vs cuBLAS)

**Files:**
- Create: `benchmarks/int8_gemm_bench.cu`

- [ ] **Step 1: Write the benchmark**

```cpp
// benchmarks/int8_gemm_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "timer.cuh"
#include "quantization/14_int8_gemm.cuh"

int main() {
    printf("SLICK INT8 GEMM (dp4a) Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 11.8 | INT8 dp4a\n");
    printf("Layout: NT (A row-major, B^T row-major, both int8x4 packed)\n");
    printf("Tile: BM=%d BN=%d BK=%d | Thread tile: TM=%d TN=%d\n",
           I8_BM, I8_BN, I8_BK, I8_TM, I8_TN);
    printf("================================================\n\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("%-6s  %10s %10s %10s %10s\n",
           "Size", "K14(us)", "cuBLAS(us)", "K14 GOPS", "Ratio");
    printf("-----------------------------------------------------------\n");

    int sizes[] = {256, 512, 1024, 2048};

    for (int S : sizes) {
        int M = S, N = S, K = S;
        int sizeA = M * K;
        int sizeB = K * N;

        // Generate FP32 data
        std::vector<float> h_A(sizeA), h_B(sizeB), h_BT(N * K);
        srand(42);
        for (int i = 0; i < sizeA; ++i)
            h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < sizeB; ++i)
            h_B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        // Transpose B [K, N] -> B^T [N, K]
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                h_BT[n * K + k] = h_B[k * N + n];

        // Device allocs
        float *d_A, *d_BT, *d_C;
        int32_t *d_A_packed, *d_BT_packed;
        float *d_scale_A, *d_scale_B;
        CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_BT, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_packed, M * (K / 4) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_BT_packed, N * (K / 4) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_scale_A, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_B, N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_BT, h_BT.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

        // Quantize (not timed — separate kernel)
        run_quantize_fp32_to_int8(M, K, d_A, d_A_packed, d_scale_A);
        run_quantize_fp32_to_int8(N, K, d_BT, d_BT_packed, d_scale_B);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- K14 benchmark ---
        for (int w = 0; w < 3; ++w)
            run_int8_gemm(M, N, K, d_A_packed, d_BT_packed, d_scale_A, d_scale_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        GpuTimer timer;
        timer.tic();
        int repeats = 10;
        for (int r = 0; r < repeats; ++r)
            run_int8_gemm(M, N, K, d_A_packed, d_BT_packed, d_scale_A, d_scale_B, d_C);
        float k14_us = timer.toc() / repeats * 1000.0f;

        // --- cuBLAS INT8 benchmark ---
        // cublasGemmEx with int8 inputs, int32 output
        // Need column-major int8 matrices for cuBLAS
        // A_col [K, M] = A^T, B_col [N, K] = B^T — already have B^T
        // For cuBLAS: C_col = alpha * op(A) * op(B) + beta * C
        // We want C[M,N] = A[M,K] @ B[K,N]
        // cuBLAS col-major: C^T[N,M] = B^T[N,K] @ A^T[K,M]
        // Use A^T col-major [K, M] = A row-major packed, B^T col-major [K, N]
        // Actually cuBLAS int8 requires specific layouts. Use NN with col-major.

        // Prepare col-major int8 buffers for cuBLAS
        // A_cm: [M, K] col-major packed = [K, M] row-major packed as int8x4
        // We'll use our quantized data and let cuBLAS handle it
        // cuBLAS GemmEx: C = A * B, col-major
        // For row-major C = A @ B: use cuBLAS C^T = B^T @ A^T
        // B^T: [N, K] -> col-major is [K, N] entries
        // A^T: [K, M] -> col-major is [M, K] entries
        // We have A_packed [M, K/4] row-major, BT_packed [N, K/4] row-major

        int32_t *d_C_cublas_i32;
        CUDA_CHECK(cudaMalloc(&d_C_cublas_i32, M * N * sizeof(int32_t)));

        // cuBLAS: C_col(N,M) = BT_col(N,K) @ A_col(K,M)
        // BT_packed is [N, K/4] row-major = col-major [K/4, N] -- need to repack
        // This is complex. Simpler: quantize into col-major format for cuBLAS.
        // For fair comparison, just measure cuBLAS with its native int8 path.

        // Use cuBLAS with CUBLAS_OP_T on row-major data:
        // cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        //              &alpha, BT_raw, CUDA_R_8I, K, A_raw, CUDA_R_8I, K,
        //              &beta, C_i32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, ...)

        // Prepare raw int8 (non-packed) for cuBLAS
        int8_t *d_A_i8, *d_BT_i8;
        CUDA_CHECK(cudaMalloc(&d_A_i8, M * K * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_BT_i8, N * K * sizeof(int8_t)));

        // Simple quantize on CPU for cuBLAS reference
        std::vector<int8_t> h_A_i8(M * K), h_BT_i8(N * K);
        std::vector<float> h_sA(M), h_sB(N);
        for (int i = 0; i < M; ++i) {
            float mx = 0;
            for (int j = 0; j < K; ++j)
                mx = fmaxf(mx, fabsf(h_A[i * K + j]));
            h_sA[i] = (mx > 0) ? mx / 127.0f : 1.0f;
            for (int j = 0; j < K; ++j)
                h_A_i8[i * K + j] = (int8_t)fminf(fmaxf(rintf(h_A[i * K + j] / h_sA[i]), -128.0f), 127.0f);
        }
        for (int i = 0; i < N; ++i) {
            float mx = 0;
            for (int j = 0; j < K; ++j)
                mx = fmaxf(mx, fabsf(h_BT[i * K + j]));
            h_sB[i] = (mx > 0) ? mx / 127.0f : 1.0f;
            for (int j = 0; j < K; ++j)
                h_BT_i8[i * K + j] = (int8_t)fminf(fmaxf(rintf(h_BT[i * K + j] / h_sB[i]), -128.0f), 127.0f);
        }
        CUDA_CHECK(cudaMemcpy(d_A_i8, h_A_i8.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_BT_i8, h_BT_i8.data(), N * K * sizeof(int8_t), cudaMemcpyHostToDevice));

        int32_t alpha_i = 1, beta_i = 0;
        // C^T(N,M) = BT(N,K) @ A^T(K,M) in col-major
        // A row-major [M,K] = A^T col-major [K,M] (lda=K)
        // BT row-major [N,K] = BT^T col-major [K,N] (ldb=K)
        // C col-major [N,M] = C^T row-major [M,N] (ldc=N)
        for (int w = 0; w < 3; ++w)
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         N, M, K,
                         &alpha_i,
                         d_BT_i8, CUDA_R_8I, K,
                         d_A_i8, CUDA_R_8I, K,
                         &beta_i,
                         d_C_cublas_i32, CUDA_R_32I, N,
                         CUBLAS_COMPUTE_32I,
                         CUBLAS_GEMM_DEFAULT);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.tic();
        for (int r = 0; r < repeats; ++r)
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         N, M, K,
                         &alpha_i,
                         d_BT_i8, CUDA_R_8I, K,
                         d_A_i8, CUDA_R_8I, K,
                         &beta_i,
                         d_C_cublas_i32, CUDA_R_32I, N,
                         CUBLAS_COMPUTE_32I,
                         CUBLAS_GEMM_DEFAULT);
        float cublas_us = timer.toc() / repeats * 1000.0f;

        // INT8 GOPS: 2*M*N*K operations (each dp4a does 4 MADs = 8 ops)
        double ops = 2.0 * M * N * K;
        float k14_gops = (float)(ops / (k14_us * 1e3));

        printf("%-6d  %10.1f %10.1f %10.1f %9.2fx\n",
               S, k14_us, cublas_us, k14_gops, cublas_us / k14_us);

        cudaFree(d_A); cudaFree(d_BT); cudaFree(d_C);
        cudaFree(d_A_packed); cudaFree(d_BT_packed);
        cudaFree(d_scale_A); cudaFree(d_scale_B);
        cudaFree(d_C_cublas_i32); cudaFree(d_A_i8); cudaFree(d_BT_i8);
    }

    cublasDestroy(handle);

    printf("\nGOPS = 2*M*N*K / time (giga int8 ops/sec)\n");
    printf("Ratio > 1 means K14 is faster than cuBLAS\n");
    printf("Quantization time not included in K14 timing\n");

    return 0;
}
```

- [ ] **Step 2: Build and run benchmark**

Run: `cmake --build build --target int8_gemm_bench && ./build/int8_gemm_bench`
Expected: Table of K14 vs cuBLAS INT8 timings

- [ ] **Step 3: Commit**

```bash
git add benchmarks/int8_gemm_bench.cu
git commit -m "bench: add K14 INT8 GEMM benchmark vs cuBLAS"
```

---

## Task 11: Update README with Week 6 Results

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run all Week 6 tests to confirm correctness**

Run: `cmake --build build && ./build/test_decode_attention && ./build/test_int8_gemm`
Expected: All tests PASS

- [ ] **Step 2: Run benchmarks and capture output**

Run: `./build/decode_attention_bench && ./build/int8_gemm_bench`
Expected: Benchmark tables printed

- [ ] **Step 3: Update README.md with Week 6 section**

Add a Week 6 section to `README.md` documenting:
- K13 decode attention: split-K architecture, speedup vs K11(N=1)
- K14 INT8 GEMM: dp4a approach, NT layout rationale, GOPS achieved, comparison vs cuBLAS
- Include benchmark result tables from Step 2

Key points to document for K14 dp4a layout decision:
- NT layout chosen: A `[M, K/4]` row-major, B stored as B^T `[N, K/4]` row-major
- Both operands have contiguous K-dimension access → coalesced global memory loads
- `__dp4a()` requires 4 consecutive int8 values packed into int32 — packing along K dimension is natural for dot product
- Alternative (B in column-major) would require strided access across K, hurting coalescing
- Per-row symmetric quantization: `scale = max(|row|) / 127`, simple and sufficient for inference

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add Week 6 decode attention + INT8 GEMM results"
```
