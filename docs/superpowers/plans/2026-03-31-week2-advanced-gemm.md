# Week 2: Advanced GEMM Kernels — Implementation Plan

> **Spec:** `docs/superpowers/specs/2026-03-31-week2-advanced-gemm-design.md`

**Goal:** Implement GEMM kernels 04-07 (1D reg tiling → 2D reg tiling → vectorized loads → double buffering), each validating against cuBLAS and benchmarked across 4 matrix sizes.

**Tech Stack:** CUDA 10.1, cuBLAS 10.2, CMake, C++14, sm_75 (GTX 1650 Ti)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `kernels/gemm/04_reg_tiling_1d.cuh` | 1D register tiling declaration |
| `kernels/gemm/04_reg_tiling_1d.cu` | 1D reg tiling: BM=BN=64, BK=8, TM=8 |
| `kernels/gemm/05_reg_tiling_2d.cuh` | 2D register tiling declaration |
| `kernels/gemm/05_reg_tiling_2d.cu` | 2D reg tiling: BM=BN=128, BK=8, TM=TN=8 |
| `kernels/gemm/06_vectorized.cuh` | Vectorized loads declaration |
| `kernels/gemm/06_vectorized.cu` | float4 vectorized global→shared loads |
| `kernels/gemm/07_double_buffered.cuh` | Double buffering declaration |
| `kernels/gemm/07_double_buffered.cu` | 2× shared memory buffers, overlap load+compute |
| `CMakeLists.txt` | Add new kernel sources to gemm_kernels library |
| `benchmarks/gemm_bench.cu` | Add kernels 04-07 to benchmark runner |

---

### Task 1: Kernel 04 — 1D Register Tiling

**Files:** Create `04_reg_tiling_1d.cuh`, `04_reg_tiling_1d.cu`. Modify `CMakeLists.txt`, `benchmarks/gemm_bench.cu`.

- [x] **Step 1: Create kernels/gemm/04_reg_tiling_1d.cuh**

```cpp
#pragma once

// 1D Register Tiling GEMM: TM=8, each thread computes 8 elements along M
void run_sgemm_reg_tiling_1d(int M, int N, int K,
                             const float* A, const float* B, float* C);
```

- [x] **Step 2: Create kernels/gemm/04_reg_tiling_1d.cu**

Key implementation details:
- Block tile: BM=64, BN=64, BK=8
- Thread tile: TM=8 (column of 8 output elements)
- 512 threads/block (linearized), each loads 1 A element + 1 B element per tile iteration
- Shared memory: As[BM][BK], Bs[BK][BN] — 4KB total
- Inner loop: for each k in BK, load Bs[k][col] once, multiply with As[row+tm][k] for tm in 0..7
- Bounds checking on global loads and output writes

- [x] **Step 3: Update CMakeLists.txt** — add `kernels/gemm/04_reg_tiling_1d.cu` to `gemm_kernels`

- [x] **Step 4: Update benchmarks/gemm_bench.cu** — add include and kernel entry

- [x] **Step 5: Build and validate**

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build && ./build/gemm_bench
```

Expected: Kernel 04 PASS at all sizes, GFLOPS > kernel 03.

- [x] **Step 6: Commit**

```
feat: add 1D register tiling GEMM kernel
```

---

### Task 2: Kernel 05 — 2D Register Tiling

**Files:** Create `05_reg_tiling_2d.cuh`, `05_reg_tiling_2d.cu`. Modify `CMakeLists.txt`, `benchmarks/gemm_bench.cu`.

- [x] **Step 1: Create kernels/gemm/05_reg_tiling_2d.cuh**

```cpp
#pragma once

// 2D Register Tiling GEMM: TM=TN=8, 8x8 micro-tile per thread
void run_sgemm_reg_tiling_2d(int M, int N, int K,
                             const float* A, const float* B, float* C);
```

- [x] **Step 2: Create kernels/gemm/05_reg_tiling_2d.cu**

Key implementation details:
- Block tile: BM=128, BN=128, BK=8
- Thread tile: TM=8, TN=8 (8×8 micro-tile)
- 256 threads (16×16 logical grid)
- Shared memory: As[BM][BK], Bs[BK][BN] — 8KB total
- Loading: each thread loads (BM×BK)/256 = 4 A elements and (BK×BN)/256 = 4 B elements per tile
- Inner loop: load A_reg[8] from As column, B_reg[8] from Bs row, outer product into 64 accumulators
- Strided load pattern for coalescing: thread tid loads rows (tid*4/BK) to (tid*4/BK + 3)

- [x] **Step 3: Update CMakeLists.txt** — add source

- [x] **Step 4: Update benchmark** — add include and entry

- [x] **Step 5: Build and validate**

Expected: GFLOPS > kernel 04, all sizes PASS.

- [x] **Step 6: Commit**

```
feat: add 2D register tiling GEMM kernel
```

---

### Task 3: Kernel 06 — Vectorized Loads

**Files:** Create `06_vectorized.cuh`, `06_vectorized.cu`. Modify `CMakeLists.txt`, `benchmarks/gemm_bench.cu`.

- [x] **Step 1: Create kernels/gemm/06_vectorized.cuh**

```cpp
#pragma once

// Vectorized GEMM: float4 loads for 128-bit memory transactions
void run_sgemm_vectorized(int M, int N, int K,
                          const float* A, const float* B, float* C);
```

- [x] **Step 2: Create kernels/gemm/06_vectorized.cu**

Key implementation details:
- Same tile dimensions as kernel 05 (BM=BN=128, BK=8, TM=TN=8)
- Global→shared loads use `reinterpret_cast<const float4*>` for A and B
- B tile (BK=8 rows × BN=128 cols): each row is 32 float4s, 256 threads load 4 float4s each
- A tile (BM=128 rows × BK=8 cols): each row is 2 float4s — transpose-aware loading
- For A: load float4 from A row, store scalars into As[row][k] (transposed store into shared)
- Shared memory compute loop unchanged from kernel 05

- [x] **Step 3: Update CMakeLists.txt**

- [x] **Step 4: Update benchmark**

- [x] **Step 5: Build and validate**

Expected: GFLOPS >= kernel 05, fewer load stalls in profiler.

- [x] **Step 6: Commit**

```
feat: add vectorized GEMM kernel with float4 loads
```

---

### Task 4: Kernel 07 — Double Buffering

**Files:** Create `07_double_buffered.cuh`, `07_double_buffered.cu`. Modify `CMakeLists.txt`, `benchmarks/gemm_bench.cu`.

- [x] **Step 1: Create kernels/gemm/07_double_buffered.cuh**

```cpp
#pragma once

// Double Buffered GEMM: 2x shared memory, overlap load and compute
void run_sgemm_double_buffered(int M, int N, int K,
                               const float* A, const float* B, float* C);
```

- [x] **Step 2: Create kernels/gemm/07_double_buffered.cu**

Key implementation details:
- Same base as kernel 06 (BM=BN=128, BK=8, TM=TN=8, float4 loads)
- Shared memory: As[2][BM][BK], Bs[2][BK][BN] — 16KB total
- Pipeline:
  1. Load tile 0 into buf[0], __syncthreads()
  2. For t = 1 to num_tiles-1:
     a. Prefetch tile t into buf[1-write_idx] (float4 loads)
     b. Compute on buf[write_idx] (register tiling + outer product)
     c. __syncthreads()
     d. Swap write_idx
  3. Compute final tile in buf[write_idx]
- Single __syncthreads per iteration instead of two

- [x] **Step 3: Update CMakeLists.txt**

- [x] **Step 4: Update benchmark**

- [x] **Step 5: Build and validate**

Expected: Highest GFLOPS of all kernels, all sizes PASS.

- [x] **Step 6: Commit**

```
feat: add double buffered GEMM kernel
```

---

### Task 5: Week 2 Milestone

- [x] **Step 1: Run full benchmark, verify all 7 kernels pass at all 4 sizes**

- [x] **Step 2: Milestone commit**

```
milestone: complete Week 2 advanced GEMM kernels
```

- [x] **Step 3: Update memory with Week 2 results**
