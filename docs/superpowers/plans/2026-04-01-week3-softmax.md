# Week 3 — Online Softmax Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two row-wise online softmax kernels — fused (shared memory reduction) and warp-reduce (`__shfl_down_sync`) — with a benchmark harness validated against a CPU reference.

**Architecture:** Each kernel takes a 2D input matrix and produces a 2D output where each row is independently softmaxed using the online softmax algorithm (2 passes over global memory). Kernel 08 uses shared memory tree reduction; Kernel 09 uses warp shuffle intrinsics for lower-latency reduction and processes 8 rows per block.

**Tech Stack:** CUDA 10.1, C++14, CMake, cuBLAS (existing link)

**Spec:** `docs/superpowers/specs/2026-04-01-week3-softmax-design.md`

---

### Task 1: Kernel 08 Header — Fused Online Softmax Declaration

**Files:**
- Create: `kernels/softmax/08_fused_online.cuh`

- [ ] **Step 1: Create the header file**

```cpp
// kernels/softmax/08_fused_online.cuh
#pragma once

// Fused Online Softmax: one block per row, shared memory reduction
// Online algorithm: single-pass (max, sum_exp) accumulation, then normalize
void run_softmax_fused_online(const float* input, float* output, int N_rows, int N_cols);
```

- [ ] **Step 2: Commit**

```bash
git add kernels/softmax/08_fused_online.cuh
git commit -m "feat: add Kernel 08 fused online softmax header"
```

---

### Task 2: Kernel 08 Implementation — Fused Online Softmax

**Files:**
- Create: `kernels/softmax/08_fused_online.cu`

- [ ] **Step 1: Write the kernel implementation**

```cuda
// kernels/softmax/08_fused_online.cu
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "softmax/08_fused_online.cuh"
#include "timer.cuh"

#define BLOCK_SIZE 256

// Device: merge two (max, sum_exp) pairs
__device__ __forceinline__
void merge_pair(float m1, float d1, float m2, float d2,
                float& m_out, float& d_out) {
    m_out = fmaxf(m1, m2);
    d_out = d1 * __expf(m1 - m_out) + d2 * __expf(m2 - m_out);
}

__global__
void softmax_fused_online_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N_rows, int N_cols) {
    int row = blockIdx.x;
    if (row >= N_rows) return;

    const float* row_in = input + row * N_cols;
    float* row_out = output + row * N_cols;
    int tid = threadIdx.x;

    // Pass 1: accumulate local (max, sum_exp)
    float m = -FLT_MAX;
    float d = 0.0f;

    for (int col = tid; col < N_cols; col += BLOCK_SIZE) {
        float x = row_in[col];
        float old_m = m;
        m = fmaxf(m, x);
        d = d * __expf(old_m - m) + __expf(x - m);
    }

    // Shared memory reduction of (m, d) pairs
    __shared__ float s_m[BLOCK_SIZE];
    __shared__ float s_d[BLOCK_SIZE];
    s_m[tid] = m;
    s_d[tid] = d;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = s_m[tid], d1 = s_d[tid];
            float m2 = s_m[tid + stride], d2 = s_d[tid + stride];
            merge_pair(m1, d1, m2, d2, s_m[tid], s_d[tid]);
        }
        __syncthreads();
    }

    float m_global = s_m[0];
    float d_global = s_d[0];

    // Pass 2: normalize
    for (int col = tid; col < N_cols; col += BLOCK_SIZE) {
        float x = row_in[col];
        row_out[col] = __expf(x - m_global) / d_global;
    }
}

void run_softmax_fused_online(const float* input, float* output,
                               int N_rows, int N_cols) {
    dim3 grid(N_rows);
    dim3 block(BLOCK_SIZE);
    softmax_fused_online_kernel<<<grid, block>>>(input, output, N_rows, N_cols);
}
```

- [ ] **Step 2: Commit**

```bash
git add kernels/softmax/08_fused_online.cu
git commit -m "feat: add Kernel 08 fused online softmax implementation"
```

---

### Task 3: Kernel 09 Header — Warp Reduce Softmax Declaration

**Files:**
- Create: `kernels/softmax/09_warp_reduce.cuh`

- [ ] **Step 1: Create the header file**

```cpp
// kernels/softmax/09_warp_reduce.cuh
#pragma once

// Warp Reduce Softmax: one warp per row, __shfl_down_sync reduction
// 8 rows per block (block dim = 32x8), zero shared memory for reductions
void run_softmax_warp_reduce(const float* input, float* output, int N_rows, int N_cols);
```

- [ ] **Step 2: Commit**

```bash
git add kernels/softmax/09_warp_reduce.cuh
git commit -m "feat: add Kernel 09 warp reduce softmax header"
```

---

### Task 4: Kernel 09 Implementation — Warp Reduce Softmax

**Files:**
- Create: `kernels/softmax/09_warp_reduce.cu`

- [ ] **Step 1: Write the kernel implementation**

```cuda
// kernels/softmax/09_warp_reduce.cu
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "softmax/09_warp_reduce.cuh"
#include "timer.cuh"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

// Device: warp-level merge reduction of (max, sum_exp) pairs
__device__ __forceinline__
void warp_reduce_md(float& m, float& d) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xFFFFFFFF, m, offset);
        float d2 = __shfl_down_sync(0xFFFFFFFF, d, offset);
        float new_m = fmaxf(m, m2);
        d = d * __expf(m - new_m) + d2 * __expf(m2 - new_m);
        m = new_m;
    }
}

__global__
void softmax_warp_reduce_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N_rows, int N_cols) {
    int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
    if (row >= N_rows) return;

    const float* row_in = input + row * N_cols;
    float* row_out = output + row * N_cols;
    int lane = threadIdx.x;

    // Pass 1: accumulate local (max, sum_exp)
    float m = -FLT_MAX;
    float d = 0.0f;

    for (int col = lane; col < N_cols; col += WARP_SIZE) {
        float x = row_in[col];
        float old_m = m;
        m = fmaxf(m, x);
        d = d * __expf(old_m - m) + __expf(x - m);
    }

    // Warp-level reduction via shuffle
    warp_reduce_md(m, d);

    // Broadcast from lane 0
    float m_global = __shfl_sync(0xFFFFFFFF, m, 0);
    float d_global = __shfl_sync(0xFFFFFFFF, d, 0);

    // Pass 2: normalize
    for (int col = lane; col < N_cols; col += WARP_SIZE) {
        float x = row_in[col];
        row_out[col] = __expf(x - m_global) / d_global;
    }
}

void run_softmax_warp_reduce(const float* input, float* output,
                              int N_rows, int N_cols) {
    int grid_rows = (N_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(grid_rows);
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    softmax_warp_reduce_kernel<<<grid, block>>>(input, output, N_rows, N_cols);
}
```

- [ ] **Step 2: Commit**

```bash
git add kernels/softmax/09_warp_reduce.cu
git commit -m "feat: add Kernel 09 warp reduce softmax implementation"
```

---

### Task 5: Benchmark Harness — `softmax_bench.cu`

**Files:**
- Create: `benchmarks/softmax_bench.cu`

- [ ] **Step 1: Write the benchmark with CPU reference validation**

```cuda
// benchmarks/softmax_bench.cu
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
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/softmax_bench.cu
git commit -m "feat: add softmax benchmark harness with CPU reference"
```

---

### Task 6: CMake Updates

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add softmax kernel library and benchmark target**

Append the following after the existing GEMM Benchmark section (after line 51):

```cmake
# --- Softmax Kernels (Week 3) ---
add_library(softmax_kernels
    kernels/softmax/08_fused_online.cu
    kernels/softmax/09_warp_reduce.cu
)
target_include_directories(softmax_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)

# Softmax Benchmark
add_executable(softmax_bench benchmarks/softmax_bench.cu)
target_link_libraries(softmax_bench softmax_kernels)
```

Note: `softmax_bench` does NOT link cuBLAS — validation uses CPU reference, not cuBLAS.

- [ ] **Step 2: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add softmax kernels and benchmark to CMake"
```

---

### Task 7: Build and Validate

**Files:**
- None (build + run only)

- [ ] **Step 1: Build the project**

```bash
cd /home/adithya/Document/SLICK
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build
```

Expected: compiles with no errors, produces `build/softmax_bench` binary.

- [ ] **Step 2: Run the softmax benchmark**

```bash
./build/softmax_bench
```

Expected: all sizes show `PASS` for both kernels with max error < 1e-6. Output includes GB/s and time for each kernel at each size.

- [ ] **Step 3: Fix any compilation or validation errors**

If any kernel fails validation, check:
- Edge handling: ensure out-of-range threads contribute `(-FLT_MAX, 0)`
- `__expf` vs `expf`: `__expf` is the fast-math intrinsic, sufficient for softmax
- Shared memory indexing in Kernel 08

- [ ] **Step 4: Commit validated build**

```bash
git add -A
git commit -m "feat: Week 3 softmax kernels validated"
```

---

### Task 8: Final Benchmark Run and Results

**Files:**
- None (run + capture output)

- [ ] **Step 1: Run the full benchmark and capture results**

```bash
./build/softmax_bench 2>&1 | tee softmax_results.txt
```

- [ ] **Step 2: Analyze performance**

Check:
- Kernel 09 (warp reduce) should be faster than Kernel 08 (shared memory) across all sizes
- Both should approach the 192 GB/s memory bandwidth ceiling for larger sizes
- Smaller sizes may show lower GB/s due to launch overhead

- [ ] **Step 3: Clean up and final commit**

```bash
rm softmax_results.txt
```

No file to commit — results are in terminal output. If the user wants results persisted, they'll request it.
