# Week 2: Advanced GEMM Kernels — Design Spec

## Overview

Week 2 builds four progressively optimized GEMM kernels (04-07) on top of Week 1's shared memory tiling baseline (~472 GFLOPS at 2048×2048). Each kernel introduces one new optimization technique, forming a clear performance progression.

## Hardware Constraints

- GTX 1650 Ti: CC 7.5, 48KB shared memory, 255 registers/thread, 1024 threads/block max
- FP32 only, no Tensor Cores
- Max matrix: 2048×2048 (fits in 4GB VRAM)

## Kernel Specifications

### Kernel 04 — 1D Register Tiling

**Technique:** Each thread computes TM=8 output elements along the M dimension, reusing B values from shared memory across all 8 A values.

**Parameters:**
- Block tile: BM=64, BN=64, BK=8
- Thread tile: TM=8 (column of 8 elements)
- Threads/block: (BM/TM) × BN = 8 × 64 = 512 (linearized 1D)
- Shared memory: As[BM][BK] + Bs[BK][BN] = 64×8 + 8×64 = 1024 floats = 4KB
- Registers: 8 accumulators + 1 B value = 9 per thread
- Arithmetic intensity: 16 FLOP/byte

**Thread mapping:**
- threadIdx.x linearized to (inner_row, inner_col) where inner_row = tid / BN, inner_col = tid % BN
- Each thread computes C[row_base + tm][col] for tm in 0..TM-1

**Loading strategy:**
- All 512 threads cooperatively load As[64][8] and Bs[8][64] from global memory
- Each thread loads 1 element of A tile and 1 element of B tile per iteration

### Kernel 05 — 2D Register Tiling

**Technique:** Each thread computes a TM×TN = 8×8 micro-tile, reusing both A and B values loaded into registers from shared memory.

**Parameters:**
- Block tile: BM=128, BN=128, BK=8 (larger for higher arithmetic intensity)
- Thread tile: TM=8, TN=8 (8×8 micro-tile per thread)
- Threads/block: (BM/TM) × (BN/TN) = 16 × 16 = 256
- Shared memory: As[BM][BK] + Bs[BK][BN] = 128×8 + 8×128 = 2048 floats = 8KB
- Registers: 64 accumulators + 8 A_reg + 8 B_reg = 80 per thread
- Arithmetic intensity: 32 FLOP/byte (2× kernel 04)

**Thread mapping:**
- threadIdx.x linearized to (thread_row, thread_col) = (tid / (BN/TN), tid % (BN/TN))
- Each thread computes C[row_base + tm][col_base + tn] for tm in 0..7, tn in 0..7

**Inner loop:**
- Load TM values from As column into A_reg[0..7]
- Load TN values from Bs row into B_reg[0..7]
- Outer product: accum[tm][tn] += A_reg[tm] * B_reg[tn]

**Loading strategy:**
- 256 threads load 128×8 = 1024 elements for A: 4 elements per thread
- 256 threads load 8×128 = 1024 elements for B: 4 elements per thread
- Strided loading pattern to maintain coalescing

### Kernel 06 — Vectorized Loads (float4)

**Technique:** Replace scalar global memory loads with float4 (128-bit) loads, reducing load instruction count by 4× and better utilizing memory bandwidth.

**Parameters:** Same as kernel 05 (BM=BN=128, BK=8, TM=TN=8)

**Changes from kernel 05:**
- Global→shared loads use `reinterpret_cast<float4*>` for 128-bit transactions
- BK=8 → each row of A/B tile = 2 float4 loads
- Shared memory may need column padding (BK+4) to avoid bank conflicts during vectorized stores
- Inner compute loop unchanged (still scalar from shared memory)

### Kernel 07 — Double Buffering

**Technique:** Use 2× shared memory buffers to overlap global memory loads with shared memory compute. While computing on buffer[i], prefetch next tile into buffer[1-i].

**Parameters:** Same as kernel 06 + doubled shared memory

**Shared memory:** 2 × (As[128][8] + Bs[8][128]) = 2 × 8KB = 16KB (well within 48KB)

**Pipeline:**
1. Load first tile into buf[0]
2. __syncthreads()
3. For each remaining tile:
   a. Prefetch next tile into buf[1-current]
   b. Compute on buf[current]
   c. __syncthreads()
   d. Swap current buffer index
4. Compute final tile

## Validation

- All kernels validate against cuBLAS with tolerance: `K * 1.2e-7f`
- Test sizes: 256, 512, 1024, 2048

## Actual Performance (2048×2048)

| Kernel | GFLOPS | Key Win |
|--------|--------|---------|
| 04 1D Reg Tiling | ~1094 | Register reuse along M |
| 05 2D Reg Tiling | ~1210 | 2D reuse + 2× arith intensity |
| 06 Vectorized | ~1689 | Fewer load instructions + transposed A in smem |
| 07 Double Buffered | ~1713 | Latency hiding (peak 1822 at 1024×1024) |

## Roofline Analysis

GTX 1650 Ti: Peak FP32 = 4300 GFLOPS (1024 cores @ 2100 MHz) | Peak BW = 192 GB/s | Ridge = 22.4 FLOP/byte

| Kernel | AI (FLOP/byte) | GFLOPS | %Peak | Bound | Detail |
|--------|---------------|--------|-------|-------|--------|
| 01 Naive | 0.25 | 30 | 0.7% | MEMORY | 62% of mem ceiling |
| 02 Coalesced | 0.25 | 351 | 8.2% | MEMORY | 731% of mem ceiling (L2 cache effects) |
| 03 Shared Tiling | 8.0 | 470 | 10.9% | MEMORY | 31% of mem ceiling |
| 04 1D Reg Tiling | 16.0 | 1090 | 25.4% | MEMORY | 35% of mem ceiling |
| 05 2D Reg Tiling | 32.0 | 1208 | 28.1% | COMPUTE | 28.1% of compute ceiling |
| 06 Vectorized | 32.0 | 1690 | 39.3% | COMPUTE | 39.3% of compute ceiling |
| 07 Double Buffered | 32.0 | 1717 | 39.9% | COMPUTE | 39.9% of compute ceiling |

Key observations:
- Kernels 01-04 are memory-bound (AI < 22.4 ridge point)
- Kernels 05-07 cross the ridge into compute-bound territory (AI = 32)
- Kernel 06 vectorized loads gave the biggest single jump (+40%) by reducing instruction overhead
- Kernel 07 double buffering adds ~2% — the GPU is already compute-saturated

Note: NCU hardware profiling is unavailable on WSL2 (GPU perf counters not exposed).
Analysis uses theoretical arithmetic intensity from tile dimensions + measured GFLOPS.

## File Structure

```
kernels/gemm/
├── 04_reg_tiling_1d.cu / .cuh
├── 05_reg_tiling_2d.cu / .cuh
├── 06_vectorized.cu / .cuh
└── 07_double_buffered.cu / .cuh
```
