# Week 1: GEMM Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build the SLICK project scaffolding, CMake build system with cuBLAS, timing/validation utilities, and the first 3 GEMM kernels (naive → coalesced → shared memory tiling) with a benchmark runner.

**Architecture:** Each CUDA kernel lives in its own `.cu` file with a `.cuh` header declaring a host-callable `run_*` wrapper. Shared utilities (timer, validator) are header-only in `include/`. A benchmark executable links all kernels and validates each against cuBLAS, reporting GFLOPS.

**Tech Stack:** CUDA 10.1, cuBLAS 10.2, CMake 4.3, C++14, sm_75 (GTX 1650 Ti)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `CMakeLists.txt` | Build system: CUDA detection, cuBLAS linkage, targets |
| `.gitignore` | Exclude build artifacts, profiling outputs |
| `CLAUDE.md` | Project instructions for future agents |
| `include/timer.cuh` | GPU event-based timing, GFLOPS calculation, benchmark loop |
| `include/validator.cuh` | cuBLAS SGEMM reference, max-error comparison, matrix init |
| `kernels/gemm/01_naive.cuh` | Naive GEMM declaration |
| `kernels/gemm/01_naive.cu` | Naive GEMM: 1 thread = 1 output, threadIdx.x → row |
| `kernels/gemm/02_coalesced.cuh` | Coalesced GEMM declaration |
| `kernels/gemm/02_coalesced.cu` | Coalesced GEMM: threadIdx.x → col for coalesced B reads |
| `kernels/gemm/03_shared_tiling.cuh` | Shared tiling GEMM declaration |
| `kernels/gemm/03_shared_tiling.cu` | 32×32 shared memory tiling GEMM |
| `benchmarks/gemm_bench.cu` | Runs all GEMM kernels across sizes, prints GFLOPS table |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `CLAUDE.md`
- Create: empty dirs `include/`, `kernels/gemm/`, `benchmarks/`, `tests/`, `scripts/`, `python/`

- [x] **Step 1: Create .gitignore**

```gitignore
# Build
build/
cmake-build-*/

# Compiled
*.o
*.so
*.a
*.exe

# Profiling
*.ncu-rep
*.nsys-rep
*.qdrep

# Editor
.vscode/
.idea/
*.swp

# Python
__pycache__/
*.pyc
.venv/

# OS
.DS_Store
```

- [x] **Step 2: Create CLAUDE.md**

```markdown
# SLICK — Speedy LLM Inference CUDA Kernels

## Build
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build

## Run benchmarks
./build/gemm_bench

## Project structure
- include/ — shared headers (timer, validator)
- kernels/ — CUDA kernel sources (.cu + .cuh per kernel)
- benchmarks/ — benchmark executables
- tests/ — validation tests

## Hardware
GTX 1650 Ti, CC 7.5, 4GB VRAM, NO Tensor Cores, CUDA 10.1

## Conventions
- Each kernel has a .cuh (declaration) and .cu (implementation)
- Host-callable wrappers: run_sgemm_<name>(M, N, K, A, B, C)
- Validate every kernel against cuBLAS: max |error| < 1e-5
- Row-major storage for all matrices
```

- [x] **Step 3: Create directory structure**

```bash
cd /home/adithya/Document/SLICK
mkdir -p include kernels/gemm kernels/softmax kernels/flash_attention \
         kernels/paged_attention kernels/decode kernels/quantization \
         benchmarks tests scripts python
```

- [x] **Step 4: Commit scaffolding**

```bash
git add .gitignore CLAUDE.md
git commit -m "feat: add project scaffolding and build instructions"
```

---

### Task 2: CMake Build System + Utility Headers

**Files:**
- Create: `CMakeLists.txt`
- Create: `include/timer.cuh`
- Create: `include/validator.cuh`

- [x] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.18)
project(SLICK LANGUAGES CXX CUDA)

# CUDA architecture for GTX 1650 Ti
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Fallback for older CUDA toolkits
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
endif()

# Find cuBLAS
find_library(CUBLAS_LIB cublas
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
find_path(CUBLAS_INCLUDE cublas_v2.h
    HINTS /usr/include /usr/local/cuda/include)

if(NOT CUBLAS_LIB)
    message(FATAL_ERROR "cuBLAS library not found")
endif()

message(STATUS "cuBLAS library: ${CUBLAS_LIB}")
message(STATUS "cuBLAS include: ${CUBLAS_INCLUDE}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
    ${CUBLAS_INCLUDE}
)
```

- [x] **Step 2: Create include/timer.cuh**

```cpp
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
    // Warmup
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
```

- [x] **Step 3: Create include/validator.cuh**

```cpp
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "timer.cuh"

struct CublasValidator {
    cublasHandle_t handle;

    CublasValidator() {
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS init failed\n");
            exit(EXIT_FAILURE);
        }
    }

    ~CublasValidator() {
        cublasDestroy(handle);
    }

    // Row-major C = A * B using cuBLAS (column-major internally)
    // A: MxK, B: KxN, C: MxN — all row-major
    void sgemm(int M, int N, int K,
               const float* d_A, const float* d_B, float* d_C) {
        float alpha = 1.0f, beta = 0.0f;
        // Row-major trick: C^T = B^T * A^T
        // cuBLAS reads row-major as transposed column-major
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,   // B in row-major = B^T in col-major
                    d_A, K,   // A in row-major = A^T in col-major
                    &beta,
                    d_C, N);  // C in row-major = C^T in col-major
    }
};

// Compute max absolute error between two device buffers
inline float max_error(const float* d_test, const float* d_ref, int size) {
    float* h_test = new float[size];
    float* h_ref  = new float[size];
    CUDA_CHECK(cudaMemcpy(h_test, d_test, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref,  d_ref,  size * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        float err = fabsf(h_test[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }

    delete[] h_test;
    delete[] h_ref;
    return max_err;
}

// Validate kernel output against cuBLAS reference
inline bool validate_gemm(const float* d_test, const float* d_ref, int size,
                          float tol = 1e-5f) {
    float err = max_error(d_test, d_ref, size);
    bool pass = err < tol;
    return pass;
}

// Fill device buffer with deterministic pseudo-random floats in [-1, 1]
inline void init_random_matrix(float* d_mat, int size, unsigned int seed = 42) {
    float* h_mat = new float[size];
    srand(seed);
    for (int i = 0; i < size; ++i) {
        h_mat[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_mat;
}
```

- [x] **Step 4: Build empty project to verify CMake + CUDA + cuBLAS**

```bash
cd /home/adithya/Document/SLICK
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 2>&1
```

Expected: CMake configures successfully, finds cuBLAS, prints paths.

- [x] **Step 5: Commit build system and utilities**

```bash
git add CMakeLists.txt include/timer.cuh include/validator.cuh
git commit -m "feat: add CMake build system and timing/validation utilities"
```

---

### Task 3: Kernel 01 — Naive GEMM

**Files:**
- Create: `kernels/gemm/01_naive.cuh`
- Create: `kernels/gemm/01_naive.cu`
- Create: `benchmarks/gemm_bench.cu`
- Modify: `CMakeLists.txt` (add kernel + benchmark targets)

- [x] **Step 1: Create kernels/gemm/01_naive.cuh**

```cpp
#pragma once

// Naive GEMM: 1 thread computes 1 element of C
// threadIdx.x → row (non-coalesced B reads — intentionally slow)
void run_sgemm_naive(int M, int N, int K,
                     const float* A, const float* B, float* C);
```

- [x] **Step 2: Create kernels/gemm/01_naive.cu**

```cpp
#include <cuda_runtime.h>
#include "gemm/01_naive.cuh"

__global__ void sgemm_naive_kernel(int M, int N, int K,
                                   const float* A, const float* B, float* C) {
    // threadIdx.x maps to row → adjacent threads access non-adjacent B elements
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_sgemm_naive(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    dim3 block(32, 32);
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    sgemm_naive_kernel<<<grid, block>>>(M, N, K, A, B, C);
}
```

- [x] **Step 3: Create benchmarks/gemm_bench.cu (initial version with kernel 1 only)**

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "validator.cuh"
#include "gemm/01_naive.cuh"

typedef void (*GemmFn)(int, int, int, const float*, const float*, float*);

struct KernelInfo {
    const char* name;
    GemmFn fn;
};

int main() {
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = 4;

    KernelInfo kernels[] = {
        {"01 Naive", run_sgemm_naive},
    };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    CublasValidator validator;

    printf("SLICK GEMM Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 10.1 | FP32\n");
    printf("========================================\n\n");

    for (int s = 0; s < num_sizes; ++s) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        int size_A = M * K;
        int size_B = K * N;
        int size_C = M * N;

        float *d_A, *d_B, *d_C, *d_ref;
        CUDA_CHECK(cudaMalloc(&d_A,   size_A * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B,   size_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C,   size_C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref, size_C * sizeof(float)));

        init_random_matrix(d_A, size_A, 42);
        init_random_matrix(d_B, size_B, 137);

        // cuBLAS reference
        validator.sgemm(M, N, K, d_A, d_B, d_ref);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Matrix Size: %dx%d\n", M, M);
        printf("%-25s %10s %8s %15s\n", "Kernel", "GFLOPS", "Status", "Max Error");
        printf("--------------------------------------------------------------\n");

        for (int ki = 0; ki < num_kernels; ++ki) {
            CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(float)));

            float avg_ms = benchmark_gemm(kernels[ki].fn, M, N, K,
                                          d_A, d_B, d_C);
            float gflops = compute_gflops(M, N, K, avg_ms);

            float err = max_error(d_C, d_ref, size_C);
            bool pass = err < 1e-5f;

            printf("%-25s %10.2f %8s %15.2e\n",
                   kernels[ki].name, gflops,
                   pass ? "PASS" : "FAIL", err);
        }
        printf("\n");

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_ref));
    }

    return 0;
}
```

- [x] **Step 4: Update CMakeLists.txt — add kernel 01 and benchmark targets**

Append to `CMakeLists.txt`:

```cmake
# --- GEMM Kernels (Week 1) ---
add_library(gemm_kernels
    kernels/gemm/01_naive.cu
)
target_include_directories(gemm_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)

# GEMM Benchmark
add_executable(gemm_bench benchmarks/gemm_bench.cu)
target_link_libraries(gemm_bench gemm_kernels ${CUBLAS_LIB})
```

- [x] **Step 5: Build and run**

```bash
cd /home/adithya/Document/SLICK
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
./build/gemm_bench
```

Expected: All 4 sizes print PASS for kernel 01 with GFLOPS numbers.

- [x] **Step 6: Commit**

```bash
git add kernels/gemm/01_naive.cuh kernels/gemm/01_naive.cu \
        benchmarks/gemm_bench.cu CMakeLists.txt
git commit -m "feat: add naive GEMM kernel with cuBLAS validation"
```

---

### Task 4: Kernel 02 — Coalesced GEMM

**Files:**
- Create: `kernels/gemm/02_coalesced.cuh`
- Create: `kernels/gemm/02_coalesced.cu`
- Modify: `benchmarks/gemm_bench.cu` (add kernel 02)
- Modify: `CMakeLists.txt` (add source)

- [x] **Step 1: Create kernels/gemm/02_coalesced.cuh**

```cpp
#pragma once

// Coalesced GEMM: threadIdx.x → col so adjacent threads read adjacent B elements
void run_sgemm_coalesced(int M, int N, int K,
                         const float* A, const float* B, float* C);
```

- [x] **Step 2: Create kernels/gemm/02_coalesced.cu**

```cpp
#include <cuda_runtime.h>
#include "gemm/02_coalesced.cuh"

__global__ void sgemm_coalesced_kernel(int M, int N, int K,
                                       const float* A, const float* B, float* C) {
    // threadIdx.x maps to col → adjacent threads read adjacent B[k*N + col] (coalesced)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_sgemm_coalesced(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    sgemm_coalesced_kernel<<<grid, block>>>(M, N, K, A, B, C);
}
```

- [x] **Step 3: Add kernel 02 to CMakeLists.txt**

Update the `gemm_kernels` library:

```cmake
add_library(gemm_kernels
    kernels/gemm/01_naive.cu
    kernels/gemm/02_coalesced.cu
)
```

- [x] **Step 4: Add kernel 02 to benchmarks/gemm_bench.cu**

Add include and array entry:

```cpp
#include "gemm/02_coalesced.cuh"
```

Update kernels array:

```cpp
KernelInfo kernels[] = {
    {"01 Naive",     run_sgemm_naive},
    {"02 Coalesced", run_sgemm_coalesced},
};
```

- [x] **Step 5: Build and run**

```bash
cmake --build build
./build/gemm_bench
```

Expected: Both kernels PASS. Kernel 02 should show higher GFLOPS than 01.

- [x] **Step 6: Commit**

```bash
git add kernels/gemm/02_coalesced.cuh kernels/gemm/02_coalesced.cu \
        benchmarks/gemm_bench.cu CMakeLists.txt
git commit -m "feat: add coalesced GEMM kernel (threadIdx.x → col)"
```

---

### Task 5: Kernel 03 — Shared Memory Tiling

**Files:**
- Create: `kernels/gemm/03_shared_tiling.cuh`
- Create: `kernels/gemm/03_shared_tiling.cu`
- Modify: `benchmarks/gemm_bench.cu` (add kernel 03)
- Modify: `CMakeLists.txt` (add source)

- [x] **Step 1: Create kernels/gemm/03_shared_tiling.cuh**

```cpp
#pragma once

// Shared memory tiling GEMM: 32x32 tiles reduce global memory traffic
void run_sgemm_shared_tiling(int M, int N, int K,
                             const float* A, const float* B, float* C);
```

- [x] **Step 2: Create kernels/gemm/03_shared_tiling.cu**

```cpp
#include <cuda_runtime.h>
#include "gemm/03_shared_tiling.cuh"

#define TILE_SIZE 32

__global__ void sgemm_shared_tiling_kernel(int M, int N, int K,
                                           const float* A, const float* B, float* C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile: A[row][t*TILE + threadIdx.x]
        int a_col = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;

        // Load B tile: B[t*TILE + threadIdx.y][col]
        int b_row = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Partial dot product from this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void run_sgemm_shared_tiling(int M, int N, int K,
                             const float* A, const float* B, float* C) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_shared_tiling_kernel<<<grid, block>>>(M, N, K, A, B, C);
}
```

- [x] **Step 3: Add kernel 03 to CMakeLists.txt**

Update the `gemm_kernels` library:

```cmake
add_library(gemm_kernels
    kernels/gemm/01_naive.cu
    kernels/gemm/02_coalesced.cu
    kernels/gemm/03_shared_tiling.cu
)
```

- [x] **Step 4: Update benchmarks/gemm_bench.cu**

Add include:

```cpp
#include "gemm/03_shared_tiling.cuh"
```

Update kernels array:

```cpp
KernelInfo kernels[] = {
    {"01 Naive",          run_sgemm_naive},
    {"02 Coalesced",      run_sgemm_coalesced},
    {"03 Shared Tiling",  run_sgemm_shared_tiling},
};
```

- [x] **Step 5: Build and run**

```bash
cmake --build build
./build/gemm_bench
```

Expected: All 3 kernels PASS. Performance progression: 03 > 02 > 01 GFLOPS.

- [x] **Step 6: Commit**

```bash
git add kernels/gemm/03_shared_tiling.cuh kernels/gemm/03_shared_tiling.cu \
        benchmarks/gemm_bench.cu CMakeLists.txt
git commit -m "feat: add shared memory tiling GEMM kernel (32x32 tiles)"
```

---

### Task 6: Week 1 Milestone — Final Validation + Commit

**Files:**
- Modify: `README.md` (update with build instructions and Week 1 results)

- [x] **Step 1: Run full benchmark and capture output**

```bash
cd /home/adithya/Document/SLICK
cmake --build build
./build/gemm_bench
```

Verify: All 12 tests pass (3 kernels × 4 sizes), GFLOPS progression is visible.

- [x] **Step 2: Update README.md with Week 1 status**

```markdown
# SLICK — Speedy LLM Inference CUDA Kernels

A from-scratch CUDA kernel library for LLM inference, targeting GTX 1650 Ti (CC 7.5, 4GB).

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
```

## Run

```bash
./build/gemm_bench    # GEMM kernel benchmark
```

## Week 1: GEMM Kernels

| # | Kernel | Technique |
|---|--------|-----------|
| 01 | Naive | 1 thread = 1 output element |
| 02 | Coalesced | threadIdx.x → col for coalesced memory reads |
| 03 | Shared Tiling | 32×32 shared memory tiles |

## Roadmap

- [x] Week 1: Naive → Coalesced → Shared Tiling GEMM
- [x] Week 2: Register tiling, vectorized loads, double buffering
- [x] Week 3: Fused softmax kernels
- [x] Week 4: FlashAttention-2
- [x] Week 5: PagedAttention + GQA
- [x] Week 6: Decode attention + INT8 GEMM
- [x] Week 7: GPT-2 inference demo
```

- [x] **Step 3: Milestone commit**

```bash
git add -A
git commit -m "milestone: complete Week 1 — GEMM foundation (kernels 1-3, build system, validation)"
```
