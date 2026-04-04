# K10b Optimized FlashAttention-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an optimized FlashAttention-2 kernel (K10b) with grid parallelism, float4 loads, BC=64, and register-only P via warp-shuffle broadcast — targeting parity with CUTLASS FMHA.

**Architecture:** New kernel `10b_flash_attn_v2_opt.cu` alongside existing K10. Grid parallelized across `(q_tiles, H, B)`. P stays in registers via half-warp shuffle broadcast for the P@V GEMM. Shared memory: Q_smem[64][68] + KV_smem[64][68] = ~34KB.

**Tech Stack:** CUDA 11.8, C++17, FP32, Sm75, CMake, Google Test

**Spec:** `docs/superpowers/specs/2026-04-03-flash-attention-optimized-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `kernels/flash_attention/10b_flash_attn_v2_opt.cuh` | Create | Header with `run_flash_attn_v2_opt()` declaration |
| `kernels/flash_attention/10b_flash_attn_v2_opt.cu` | Create | Full optimized kernel implementation |
| `tests/test_attention.cu` | Modify | Add K10b test cases reusing existing fixture |
| `benchmarks/attention_bench.cu` | Modify | Add K10b column to validation + timing tables |
| `CMakeLists.txt` | Modify | Add K10b source to `attention_kernels` library |

---

### Task 1: Header and Build System

**Files:**
- Create: `kernels/flash_attention/10b_flash_attn_v2_opt.cuh`
- Modify: `CMakeLists.txt:74-76` (attention_kernels library)

- [ ] **Step 1: Create header file**

```cuda
// kernels/flash_attention/10b_flash_attn_v2_opt.cuh
#pragma once

// Optimized FlashAttention-2 forward pass
// Grid-parallelized across Q tiles: grid(num_q_tiles, H, B)
// Tiling: Br=64, Bc=64, d=64. float4 loads, register-only P via warp shuffle.
void run_flash_attn_v2_opt(int B, int H, int N, int d,
                           const float* Q, const float* K, const float* V,
                           float* O, bool causal);
```

- [ ] **Step 2: Add source to CMakeLists.txt**

Change the `attention_kernels` library from:
```cmake
add_library(attention_kernels
    kernels/flash_attention/10_flash_attn_v2.cu
)
```
To:
```cmake
add_library(attention_kernels
    kernels/flash_attention/10_flash_attn_v2.cu
    kernels/flash_attention/10b_flash_attn_v2_opt.cu
)
```

- [ ] **Step 3: Create stub implementation**

Create `kernels/flash_attention/10b_flash_attn_v2_opt.cu` with a minimal stub that just zeros the output (enough to link and fail tests):

```cuda
// kernels/flash_attention/10b_flash_attn_v2_opt.cu
#include <cuda_runtime.h>
#include "flash_attention/10b_flash_attn_v2_opt.cuh"
#include "timer.cuh"

void run_flash_attn_v2_opt(int B, int H, int N, int d,
                           const float* Q, const float* K, const float* V,
                           float* O, bool causal) {
    CUDA_CHECK(cudaMemset(O, 0, (size_t)B * H * N * d * sizeof(float)));
}
```

- [ ] **Step 4: Build to verify linking**

Run:
```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --target attention_kernels
```
Expected: builds successfully with no errors.

- [ ] **Step 5: Commit**

```bash
git add kernels/flash_attention/10b_flash_attn_v2_opt.cuh kernels/flash_attention/10b_flash_attn_v2_opt.cu CMakeLists.txt
git commit -m "feat: add K10b optimized FlashAttention stub and build config"
```

---

### Task 2: Add K10b Tests

**Files:**
- Modify: `tests/test_attention.cu`

Add test cases for K10b that reuse the existing `AttentionTest` fixture. These will fail against the stub (output is all zeros) and pass once the real kernel is implemented.

- [ ] **Step 1: Add K10b include and parameterized tests**

Add to the top of `tests/test_attention.cu`, after the existing `#include "flash_attention/10_flash_attn_v2.cuh"`:
```cuda
#include "flash_attention/10b_flash_attn_v2_opt.cuh"
```

Then add a new test method to the `AttentionTest` class, after the `RunAndCheck` method (after line 109):
```cuda
    void RunAndCheckOpt(bool causal, float tol = 1e-5f) {
        attention_cpu_reference(B, H, N, d, h_Q, h_K, h_V, h_O_ref, causal);

        CUDA_CHECK(cudaMemset(d_O, 0, total * sizeof(float)));
        run_flash_attn_v2_opt(B, H, N, d, d_Q, d_K, d_V, d_O, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_O = new float[total];
        CUDA_CHECK(cudaMemcpy(h_O, d_O, total * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < total; ++i) {
            float err = fabsf(h_O[i] - h_O_ref[i]);
            if (err > max_err) max_err = err;
        }
        delete[] h_O;
        EXPECT_LT(max_err, tol) << "Max error: " << max_err;
    }
```

Add new parameterized tests after the existing `AttentionCausalTest` tests (after line 140):
```cuda
class AttentionOptTest : public AttentionTest,
                         public ::testing::WithParamInterface<AttnParams> {
protected:
    void SetUp() override {
        auto p = GetParam();
        SetUpAttention(p.B, p.H, p.N, p.d);
    }
};

TEST_P(AttentionOptTest, CausalOpt)    { RunAndCheckOpt(true); }
TEST_P(AttentionOptTest, NonCausalOpt) { RunAndCheckOpt(false); }

INSTANTIATE_TEST_SUITE_P(ConfigsOpt, AttentionOptTest, ::testing::Values(
    AttnParams{1, 1,  128, 64},
    AttnParams{1, 12, 128, 64},
    AttnParams{1, 12, 256, 64},
    AttnParams{1, 12, 512, 64}
));

TEST_F(AttentionTest, MultiBatchCausalOpt) {
    SetUpAttention(2, 12, 256, 64);
    RunAndCheckOpt(true);
}

TEST_F(AttentionTest, FullGPT2CausalOpt) {
    SetUpAttention(1, 12, 1024, 64);
    RunAndCheckOpt(true);
}
```

- [ ] **Step 2: Build and run tests — verify K10b tests FAIL, K10 tests still PASS**

```bash
cmake --build build --target test_attention && ./build/test_attention
```
Expected: original 10 tests PASS, new K10b tests FAIL (stub returns zeros).

- [ ] **Step 3: Commit**

```bash
git add tests/test_attention.cu
git commit -m "test: add K10b optimized FlashAttention test cases"
```

---

### Task 3: Implement the Optimized Kernel

**Files:**
- Modify: `kernels/flash_attention/10b_flash_attn_v2_opt.cu` (replace stub with full implementation)

This is the core task. The kernel implements all 4 optimizations:
1. Grid parallelism across Q tiles
2. float4 vectorized loads
3. BC=64 (doubled from 32)
4. Register-only P via warp-shuffle broadcast (no P_smem)

- [ ] **Step 1: Write the full kernel implementation**

Replace the entire contents of `kernels/flash_attention/10b_flash_attn_v2_opt.cu` with:

```cuda
// kernels/flash_attention/10b_flash_attn_v2_opt.cu
// Optimized FlashAttention-2: grid-parallel, float4, BC=64, register-only P
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "flash_attention/10b_flash_attn_v2_opt.cuh"
#include "timer.cuh"

// ============================================================
// Tile dimensions
// ============================================================
#define BR 64        // Q tile rows
#define BC 64        // KV tile cols (doubled from K10's 32)
#define HD 64        // Head dimension
#define NTHREADS 256 // 16x16 thread grid
#define PAD 4        // Shared memory padding for float4 alignment

// Thread tile sizes — same 4 rows in both GEMMs (enables register-only P)
#define TM   4  // rows per thread: 16 thread_rows x 4 = 64
#define TN_S 4  // S cols per thread: 16 thread_cols x 4 = 64 = BC
#define TN_O 4  // O cols per thread: 16 thread_cols x 4 = 64 = HD

// ============================================================
// Kernel
// ============================================================
template <bool CAUSAL>
__global__ __launch_bounds__(NTHREADS)
void flash_attn_v2_opt_kernel(int N, int d, float scale,
                              const float* __restrict__ Q,
                              const float* __restrict__ K,
                              const float* __restrict__ V,
                              float* __restrict__ O) {
    // Grid: (num_q_tiles, H, B)
    const int qi       = blockIdx.x;  // Q tile index
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int bh = batch_idx * gridDim.y + head_idx;

    const float* Q_bh = Q + bh * N * d;
    const float* K_bh = K + bh * N * d;
    const float* V_bh = V + bh * N * d;
    float*       O_bh = O + bh * N * d;

    const int tid = threadIdx.x;
    const int thread_row = tid / 16;   // 0..15
    const int thread_col = tid % 16;   // 0..15

    // Half-warp info for shuffle reductions
    const int warp_id = tid / 32;
    const int lane    = tid & 31;
    const int half    = lane / 16;             // 0 or 1
    const unsigned half_mask = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int half_leader = half * 16;         // lane 0 or lane 16

    // Shared memory (padded +4 for float4-aligned rows of 68 floats = 272 bytes)
    __shared__ float Q_smem[BR][HD + PAD];    // 64 x 68 = 17,408 B
    __shared__ float KV_smem[BC][HD + PAD];   // 64 x 68 = 17,408 B
                                               // Total:    34,816 B

    const int q_start = qi * BR;
    const int num_kv_tiles = (N + BC - 1) / BC;

    // --- Load Q tile (BR x d) into Q_smem via float4 ---
    // 64 rows x 64 cols = 4096 floats = 1024 float4s. 256 threads -> 4 float4s each.
    for (int idx = tid; idx < BR * (HD / 4); idx += NTHREADS) {
        int r = idx / (HD / 4);
        int c4 = idx % (HD / 4);
        int gr = q_start + r;
        float4 val;
        if (gr < N) {
            val = reinterpret_cast<const float4*>(Q_bh + gr * d)[c4];
        } else {
            val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        reinterpret_cast<float4*>(&Q_smem[r][c4 * 4])[0] = val;
    }

    // Initialize O accumulator and softmax state in registers
    float O_acc[TM][TN_O];
    float m_i[TM];   // running row max
    float l_i[TM];   // running row sum_exp

    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
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

        // Causal early exit: entire KV tile is above the diagonal
        if (CAUSAL && kv_start > q_start + BR - 1) break;

        // --- Step 1: Load K_j (BC x d) into KV_smem via float4 ---
        for (int idx = tid; idx < BC * (HD / 4); idx += NTHREADS) {
            int r = idx / (HD / 4);
            int c4 = idx % (HD / 4);
            int gr = kv_start + r;
            float4 val;
            if (gr < N) {
                val = reinterpret_cast<const float4*>(K_bh + gr * d)[c4];
            } else {
                val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            reinterpret_cast<float4*>(&KV_smem[r][c4 * 4])[0] = val;
        }
        __syncthreads();

        // --- Step 2: S = Q @ K^T * scale  (TM x TN_S = 4x4 per thread) ---
        float S[TM][TN_S];
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn)
                S[tm][tn] = 0.0f;

        for (int k = 0; k < HD; ++k) {
            float q_frag[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                q_frag[tm] = Q_smem[thread_row * TM + tm][k];

            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn) {
                float k_val = KV_smem[thread_col * TN_S + tn][k];
                #pragma unroll
                for (int tm = 0; tm < TM; ++tm)
                    S[tm][tn] += q_frag[tm] * k_val;
            }
        }

        // Scale
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn)
                S[tm][tn] *= scale;

        // Boundary + causal mask
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            const int gr = q_start + thread_row * TM + tm;
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn) {
                const int gc = kv_start + thread_col * TN_S + tn;
                bool masked = (gc >= N);
                if (CAUSAL) masked = masked || (gr < gc);
                if (masked) S[tm][tn] = -FLT_MAX;
            }
        }

        // --- Step 3: Online softmax via half-warp shuffle ---
        float m_ij[TM], l_ij[TM];

        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            // Local max across TN_S=4 columns
            float local_m = S[tm][0];
            #pragma unroll
            for (int tn = 1; tn < TN_S; ++tn)
                local_m = fmaxf(local_m, S[tm][tn]);

            // Half-warp max reduction (16 lanes)
            #pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1)
                local_m = fmaxf(local_m,
                                __shfl_down_sync(0xFFFFFFFF, local_m, offset));
            m_ij[tm] = __shfl_sync(0xFFFFFFFF, local_m, half_leader);

            // exp(S - max) -> P values (stored back in S registers)
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn)
                S[tm][tn] = __expf(S[tm][tn] - m_ij[tm]);

            // Local sum
            float local_l = S[tm][0];
            #pragma unroll
            for (int tn = 1; tn < TN_S; ++tn)
                local_l += S[tm][tn];

            // Half-warp sum reduction
            #pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1)
                local_l += __shfl_down_sync(0xFFFFFFFF, local_l, offset);
            l_ij[tm] = __shfl_sync(0xFFFFFFFF, local_l, half_leader);
        }

        // --- Online rescaling of O accumulator ---
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            float m_new = fmaxf(m_i[tm], m_ij[tm]);
            float alpha = __expf(m_i[tm] - m_new);
            float beta  = __expf(m_ij[tm] - m_new);

            #pragma unroll
            for (int tn = 0; tn < TN_O; ++tn)
                O_acc[tm][tn] *= alpha;

            l_i[tm] = l_i[tm] * alpha + l_ij[tm] * beta;
            m_i[tm] = m_new;

            // Scale P by beta for P@V
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn)
                S[tm][tn] *= beta;
        }

        // --- Step 4: Load V_j (BC x d) into KV_smem via float4 ---
        __syncthreads();  // done reading K from KV_smem

        for (int idx = tid; idx < BC * (HD / 4); idx += NTHREADS) {
            int r = idx / (HD / 4);
            int c4 = idx % (HD / 4);
            int gr = kv_start + r;
            float4 val;
            if (gr < N) {
                val = reinterpret_cast<const float4*>(V_bh + gr * d)[c4];
            } else {
                val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            reinterpret_cast<float4*>(&KV_smem[r][c4 * 4])[0] = val;
        }
        __syncthreads();  // V ready in KV_smem

        // --- Step 5: O += P @ V via warp-shuffle broadcast (no P_smem) ---
        // Each thread holds S[TM][TN_S] = P values for its 4 rows x 4 KV-cols.
        // Full P row has BC=64 values across 16 threads (4 each).
        // Broadcast via __shfl_sync across the half-warp.

        const int lane_in_half = lane % 16;  // 0..15

        for (int src = 0; src < 16; ++src) {
            // Broadcast 4 P values from source thread
            float p_bcast[TM][TN_S];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                #pragma unroll
                for (int tn_s = 0; tn_s < TN_S; ++tn_s)
                    p_bcast[tm][tn_s] = __shfl_sync(half_mask, S[tm][tn_s],
                                                     src + half_leader);

            // Accumulate: for each of the 4 KV rows this source owns
            #pragma unroll
            for (int tn_s = 0; tn_s < TN_S; ++tn_s) {
                int kv_row = src * TN_S + tn_s;  // 0..63
                // Load V row fragment for this thread's O columns
                float v_frag[TN_O];
                #pragma unroll
                for (int tn = 0; tn < TN_O; ++tn)
                    v_frag[tn] = KV_smem[kv_row][thread_col * TN_O + tn];

                #pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    float p_val = p_bcast[tm][tn_s];
                    #pragma unroll
                    for (int tn = 0; tn < TN_O; ++tn)
                        O_acc[tm][tn] += p_val * v_frag[tn];
                }
            }
        }

        __syncthreads();  // ensure P@V reads done before next K load

    }  // end inner loop

    // --- Write O to global memory via float4 ---
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int gr = q_start + thread_row * TM + tm;
        if (gr < N) {
            float inv_l = (l_i[tm] > 0.0f) ? 1.0f / l_i[tm] : 0.0f;
            // This thread writes 4 contiguous output cols: [thread_col*4 .. thread_col*4+3]
            float4 out_val;
            out_val.x = O_acc[tm][0] * inv_l;
            out_val.y = O_acc[tm][1] * inv_l;
            out_val.z = O_acc[tm][2] * inv_l;
            out_val.w = O_acc[tm][3] * inv_l;
            reinterpret_cast<float4*>(O_bh + gr * d)[thread_col] = out_val;
        }
    }
}

// ============================================================
// Host wrapper
// ============================================================
void run_flash_attn_v2_opt(int B, int H, int N, int d,
                           const float* Q, const float* K, const float* V,
                           float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    int num_q_tiles = (N + BR - 1) / BR;
    dim3 grid(num_q_tiles, H, B);
    dim3 block(NTHREADS);

    if (causal)
        flash_attn_v2_opt_kernel<true><<<grid, block>>>(N, d, scale, Q, K, V, O);
    else
        flash_attn_v2_opt_kernel<false><<<grid, block>>>(N, d, scale, Q, K, V, O);
}

#undef BR
#undef BC
#undef HD
#undef NTHREADS
#undef PAD
#undef TM
#undef TN_S
#undef TN_O
```

- [ ] **Step 2: Build**

```bash
cmake --build build
```
Expected: compiles with no errors.

- [ ] **Step 3: Run tests**

```bash
./build/test_attention
```
Expected: all tests PASS (original K10 + new K10b).

- [ ] **Step 4: If tests fail, debug and fix**

Common issues to check:
- Warp shuffle mask: `half_mask` must match the half-warp that owns the same `thread_row` group
- Boundary conditions: `gr >= N` or `gc >= N` must be masked to `-FLT_MAX`
- float4 alignment: `d` must be divisible by 4 (which it is, d=64)
- Causal mask: `gr < gc` not `gr <= gc`

- [ ] **Step 5: Commit**

```bash
git add kernels/flash_attention/10b_flash_attn_v2_opt.cu
git commit -m "feat: implement K10b optimized FlashAttention kernel"
```

---

### Task 4: Add K10b to Benchmark

**Files:**
- Modify: `benchmarks/attention_bench.cu`

Add K10b as a fourth column in both the validation and timing tables, alongside Flash (K10), Unfused, and CUTLASS.

- [ ] **Step 1: Add include**

At the top of `benchmarks/attention_bench.cu`, after the existing includes, add:
```cuda
#include "flash_attention/10b_flash_attn_v2_opt.cuh"
```

- [ ] **Step 2: Add K10b to validation loop**

After the CUTLASS validation block (after the `delete[] h_O_cutlass_bhnk;` line), add:
```cuda
        // K10b optimized FlashAttention
        float *d_O_opt;
        CUDA_CHECK(cudaMalloc(&d_O_opt, total_qkvo * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_O_opt, 0, total_qkvo * sizeof(float)));
        run_flash_attn_v2_opt(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O_opt, c.causal);
        CUDA_CHECK(cudaDeviceSynchronize());
        float err_opt = attention_max_error(d_O_opt, h_O_ref, total_qkvo);
        bool pass_opt = err_opt < tol;
        printf("%-25s %10s %15.2e %8s\n", c.desc, "K10b", err_opt,
               pass_opt ? "PASS" : "FAIL");
```

Add `CUDA_CHECK(cudaFree(d_O_opt));` to the cleanup section at the end of the validation loop, before the closing `}`.

- [ ] **Step 3: Update benchmark header**

Change the benchmark header printf to:
```cuda
    printf("%-15s %5s %5s %6s %5s  %10s %10s %10s %10s %10s  %8s\n",
           "Config", "B", "H", "N", "d",
           "Flash(us)", "K10b(us)", "Unfsd(us)", "CUTLAS(us)", "Speedup", "Eff BW");
    printf("----------------------------------------------------------------------------------------------------------------------\n");
```

- [ ] **Step 4: Add K10b timing to benchmark loop**

After the FlashAttention timing block (after `float flash_us = flash_ms * 1000.0f;`), add:
```cuda
        // Benchmark K10b
        for (int w = 0; w < 3; ++w)
            run_flash_attn_v2_opt(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O, c.causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_flash_attn_v2_opt(c.B, c.H, c.N, c.d, d_Q, d_K, d_V, d_O, c.causal);
        float opt_ms = timer.toc() / 10.0f;
        float opt_us = opt_ms * 1000.0f;
```

Update the Speedup calculation to compare K10b vs unfused:
```cuda
        float speedup = unfused_us / opt_us;
```

Update the TFLOPS and eff_bw to use K10b timing:
```cuda
        float flash_tflops = (float)(total_flops / (opt_ms * 1e9));
        float eff_bw = (float)(ideal_bytes / (opt_ms * 1e6));
```

Update the results printf:
```cuda
        printf("%-15s %5d %5d %6d %5d  %10.1f %10.1f %10.1f %10.1f %10.2fx %7.1f GB/s\n",
               c.desc, c.B, c.H, c.N, c.d,
               flash_us, opt_us, unfused_us, cutlass_us, speedup, eff_bw);
```

- [ ] **Step 5: Build and run benchmark**

```bash
cmake --build build --target attention_bench && ./build/attention_bench
```
Expected: all validations PASS. K10b timing should be significantly faster than K10.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/attention_bench.cu
git commit -m "bench: add K10b optimized FlashAttention to attention benchmark"
```

---

### Task 5: Final Verification and Commit

- [ ] **Step 1: Full rebuild from clean**

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build
```
Expected: all targets build successfully.

- [ ] **Step 2: Run all tests**

```bash
./build/test_gemm && ./build/test_softmax && ./build/test_attention
```
Expected: all tests PASS (42 GEMM + 14 softmax + 20 attention including K10b).

- [ ] **Step 3: Run benchmark and capture results**

```bash
./build/attention_bench
```
Expected: K10b PASS validation, timing shows speedup over K10, comparable to CUTLASS.

- [ ] **Step 4: Commit benchmark results in spec**

If results look good, no further commits needed. If any optimization underperforms, note it in the spec for future reference.
