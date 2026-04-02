# Week 3 Design Spec — Online Softmax Kernels

## Overview

Two row-wise softmax kernels using the online softmax algorithm (Milakov & Gimelshein 2018). Each row of a 2D input matrix is independently softmaxed. This is the core primitive for attention score normalization and feeds directly into FlashAttention (Week 4).

## Hardware Target

- GTX 1650 Ti, CC 7.5, 4GB VRAM, 192 GB/s memory bandwidth
- FP32 only, CUDA 10.1
- Softmax is memory-bound — performance metric is GB/s, not GFLOPS

## Online Softmax Algorithm

Single-pass accumulation of `(max, sum_exp)` per row, then a normalize pass. Two passes over global memory instead of three.

### Pass 1 — Accumulate

Each thread maintains a local pair `(m, d)` initialized to `(-inf, 0)`:

```
for each element x_i assigned to this thread:
    old_m = m
    m = max(m, x_i)
    d = d * exp(old_m - m) + exp(x_i - m)
```

When `m` increases, the existing `d` is rescaled by `exp(old_m - new_m)`.

### Pairwise Merge

After the loop, local `(m, d)` pairs are reduced across threads:

```
merge((m1, d1), (m2, d2)):
    m = max(m1, m2)
    d = d1 * exp(m1 - m) + d2 * exp(m2 - m)
    return (m, d)
```

This merge is associative and commutative. Identity element: `(-inf, 0)`.

### Pass 2 — Normalize

Each thread revisits its elements and writes:

```
output[i] = exp(x_i - m_global) / d_global
```

## Kernel 08 — Fused Online Softmax (Shared Memory Reduction)

### Launch Configuration

- Grid: `(N_rows, 1, 1)` — one block per row
- Block: `(256, 1, 1)`
- Shared memory: `256 * sizeof(float2)` = 2048 bytes

### Algorithm

1. Each thread computes stride = 256, starting index = `threadIdx.x`
2. **Pass 1**: Strided loop over row elements, accumulate local `(m, d)`
3. **Shared memory reduction**: Store `(m, d)` into `__shared__ float2 smem[256]`. Tree reduction over 8 steps using `merge()` with `__syncthreads()` barriers. Result in `smem[0]`
4. **Broadcast**: All threads read `(m_global, d_global)` from `smem[0]`
5. **Pass 2**: Strided loop, write normalized output

### Memory Access

- Coalesced reads/writes: thread `t` accesses indices `t, t+256, t+512, ...`
- Two full passes over the row in global memory

### Edge Handling

- Threads with out-of-range indices contribute identity `(-inf, 0)` to the reduction

## Kernel 09 — Warp Reduce Softmax

### Launch Configuration

- Grid: `(ceil(N_rows / 8), 1, 1)`
- Block: `(32, 8, 1)` — 8 warps per block, each warp handles one row
- Shared memory: none required (warp shuffles for reduction, `__shfl_sync` for broadcast)

### Algorithm

1. `row = blockIdx.x * 8 + threadIdx.y` — warp-to-row mapping
2. `lane = threadIdx.x` — lane within the 32-thread warp
3. **Pass 1**: Each lane loops with stride 32, accumulating local `(m, d)`
4. **Warp reduction**: 5 rounds of `__shfl_down_sync(0xFFFFFFFF, ...)` merging `(m, d)` pairs. Lane 0 holds final result
5. **Broadcast**: `__shfl_sync` from lane 0 to all lanes
6. **Pass 2**: Each lane revisits its elements, writes normalized output

### Why Faster Than Kernel 08

- `__shfl_down_sync` ~5 cycles vs shared memory load ~20-30 cycles
- 5 shuffle rounds vs 8 reduction steps with `__syncthreads()`
- 8 rows per block vs 1 — better occupancy
- Zero shared memory usage

### Trade-off

- 32 threads per row vs 256 — each thread processes 8x more elements per row. For rows up to 4096 elements, this is fine. For very large rows (8192+), per-thread loop is longer but still functional.

### Edge Handling

- Out-of-range lanes contribute identity `(-inf, 0)`
- Warps assigned beyond `N_rows` early-return

## Host-Callable Wrappers

```cpp
void run_softmax_fused_online(const float* input, float* output, int N_rows, int N_cols);
void run_softmax_warp_reduce(const float* input, float* output, int N_rows, int N_cols);
```

Separate input/output buffers (not in-place).

## Validation

### CPU Reference

Numerically stable 3-pass implementation on host:
1. Find row max
2. Compute `exp(x_i - max)` and accumulate sum
3. Divide by sum

Tolerance: `max |output_kernel - output_ref| < 1e-6` per element.

### Test Sizes

| N_rows | N_cols |
|--------|--------|
| 64     | 128    |
| 64     | 512    |
| 64     | 1024   |
| 64     | 2048   |
| 128    | 128    |
| 128    | 512    |
| 128    | 1024   |
| 128    | 2048   |
| 256    | 1024   |
| 256    | 4096   |
| 512    | 1024   |
| 512    | 4096   |

All fit comfortably within 4GB VRAM.

## Performance Metric

**GB/s** — effective memory bandwidth:

```
GB/s = 2 * N_rows * N_cols * sizeof(float) / (time_sec * 1e9)
```

Factor of 2: one read + one write per element. Target: approach 192 GB/s (device memory bandwidth ceiling).

## Benchmark Harness — `softmax_bench.cu`

Follows `gemm_bench.cu` pattern:
- Iterates over test sizes
- Per kernel: warmup (3 runs) + timed average (10 runs) using `GpuTimer`
- Reports: kernel name, size, GB/s, time (µs), PASS/FAIL, max error
- CPU reference computed once per size, reused across kernels

## New Files

| File | Purpose |
|------|---------|
| `kernels/softmax/08_fused_online.cuh` | Kernel 08 declaration |
| `kernels/softmax/08_fused_online.cu` | Kernel 08 implementation |
| `kernels/softmax/09_warp_reduce.cuh` | Kernel 09 declaration |
| `kernels/softmax/09_warp_reduce.cu` | Kernel 09 implementation |
| `benchmarks/softmax_bench.cu` | Benchmark + validation harness |

## CMake Changes

- New `softmax_kernels` static library from `08_fused_online.cu` and `09_warp_reduce.cu`
- New `softmax_bench` executable linking `softmax_kernels`
- No new external dependencies

## File Conventions

Per project conventions:
- Each kernel has a `.cuh` (declaration) and `.cu` (implementation)
- Host-callable wrappers prefixed `run_softmax_*`
- Row-major storage for all matrices
