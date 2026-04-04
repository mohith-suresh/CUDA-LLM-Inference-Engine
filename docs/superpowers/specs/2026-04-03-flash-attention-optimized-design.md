# K10b — Optimized FlashAttention-2 Design Spec

## Overview

Optimized variant of Kernel 10 (FlashAttention-2) targeting 1.5-2x speedup via five specific optimizations. Created as a new kernel file alongside the original for side-by-side benchmarking.

## Hardware Target

- GTX 1650 Ti, CC 7.5, 4GB VRAM, NO Tensor Cores
- CUDA 11.8, FP32 only
- 48 KB shared memory per SM, 16 SMs

## Optimizations Applied

### 1. Q-Tile Grid Parallelism
**Problem:** K10 uses `grid(B*H)` — only 12 blocks for B=1,H=12 on 16 SMs.
**Solution:** Move the outer Q-tile loop into the grid: `grid(num_q_tiles, H, B)`. For N=1024, BR=64: `16 * 12 * 1 = 192` blocks.

### 2. Double-Buffered K/V Loads (Dropped)
Dropped due to shared memory pressure. `Q_smem + 2*KV_smem` = 52 KB > 48 KB limit. Grid parallelism (#1) is the dominant win, making this trade-off acceptable.

### 3. Eliminate P_smem Round-Trip (Register-Only P)
**Problem:** K10 writes P to shared memory, syncs, reads it back for P@V.
**Solution:** Keep P in registers. Use warp-shuffle broadcast so all threads in a row-group can access the full P row for the P@V accumulation. Each of 16 column-threads broadcasts its TN_S=4 P values in turn.

### 4. Vectorized float4 Global Loads
**Problem:** K10 uses scalar float loads for Q, K, V.
**Solution:** Use `float4` (128-bit) loads. Since d=64, each row = 16 float4s. Cuts load instruction count 4x. Shared memory padded to +4 floats per row for float4-aligned access.

### 5. Increase BC from 32 to 64
**Problem:** BC=32 means 2x more inner-loop iterations than necessary.
**Solution:** BC=64 halves inner loop count, reducing sync overhead and global memory load transactions.

## Kernel Parameters

| Parameter | K10 (old) | K10b (new) |
|-----------|-----------|------------|
| BR | 64 | 64 |
| BC | 32 | 64 |
| HD | 64 | 64 |
| NTHREADS | 256 | 256 |
| Thread grid | 16x16 | 16x16 |
| TM | 4 | 4 |
| TN_S | 2 | 4 |
| TN_O | 4 | 4 |
| Grid dim | (B*H, 1, 1) | (num_q_tiles, H, B) |

## Thread Mapping

256 threads arranged as 16 rows x 16 cols:
- `thread_row = tid / 16` → 0..15
- `thread_col = tid % 16` → 0..15

Each thread owns:
- **S compute:** rows `[thread_row*4 .. thread_row*4+3]` x cols `[thread_col*4 .. thread_col*4+3]` → `S[4][4]`
- **O accumulate:** rows `[thread_row*4 .. thread_row*4+3]` x cols `[thread_col*4 .. thread_col*4+3]` → `O_acc[4][4]`

Key property: same thread owns the same 4 rows in both GEMMs, enabling register-only P transfer.

## Shared Memory Layout

```
Q_smem[64][68]     // BR x (HD+4), padded for float4. 17,408 bytes
KV_smem[64][68]    // BC x (HD+4), padded for float4. 17,408 bytes
                   // Total: 34,816 bytes (< 48 KB)
```

No P_smem — P stays in registers.

## Algorithm

```
For each block (q_tile_idx, head_idx, batch_idx):
  1. Load Q[q_start : q_start+BR, :] into Q_smem using float4
  2. Initialize O_acc[4][4]=0, m_i[4]=-inf, l_i[4]=0

  For each KV tile kj = 0..num_kv_tiles:
    (causal early exit: skip if kv_start > q_end)

    3. Load K[kv_start : kv_start+BC, :] into KV_smem using float4
    4. __syncthreads()

    5. Compute S[4][4] = Q_smem_rows @ KV_smem_rows^T * scale
       (inner product over HD=64 dimension)

    6. Apply causal + boundary mask to S

    7. Online softmax:
       - Row max via warp shuffle (16-lane half-warp reduction)
       - exp(S - max) in registers
       - Row sum via warp shuffle
       - Rescale O_acc by exp(m_old - m_new)
       - Update m_i, l_i

    8. Load V[kv_start : kv_start+BC, :] into KV_smem using float4
    9. __syncthreads()

    10. O_acc += P @ V via warp-shuffle broadcast:
        For src_col = 0..15:
          Each thread broadcasts its 4 P values (from S[tm][0..3])
          via __shfl_sync to all 16 threads in the row
          All threads accumulate: O_acc[tm][tn] += p_broadcast * V_smem[kv_col][thread_col*4+tn]
        This covers all 64 KV columns (16 threads * 4 cols each)

    11. __syncthreads()

  12. Write O_acc / l_i to global memory using float4
```

## P@V Warp-Shuffle Broadcast Detail

The critical optimization. Instead of materializing the full P[BR][BC] in shared memory:

```
// Each thread holds S[4][4] = P values for its 4 rows x 4 KV-columns
// To compute O[4][4] += P[4][64] @ V[64][64]:
//   P row i has 64 values, but this thread only holds 4 of them
//   The other 60 are held by the other 15 threads in the same row-group

for (int src = 0; src < 16; ++src) {        // iterate over source threads
    for (int tn_s = 0; tn_s < 4; ++tn_s) {  // 4 P values per source
        int kv_idx = src * 4 + tn_s;         // KV column index 0..63
        float p_vals[4];                      // 4 P values (one per owned row)
        for (int tm = 0; tm < 4; ++tm)
            p_vals[tm] = __shfl_sync(mask, S[tm][tn_s], src + half_leader);

        // Accumulate into O
        float v_frag[4];
        for (int tn = 0; tn < 4; ++tn)
            v_frag[tn] = KV_smem[kv_idx][thread_col * 4 + tn];
        for (int tm = 0; tm < 4; ++tm)
            for (int tn = 0; tn < 4; ++tn)
                O_acc[tm][tn] += p_vals[tm] * v_frag[tn];
    }
}
```

This uses 64 shuffle operations per thread-row (16 sources x 4 values), replacing a shared memory write + sync + read pattern.

## Warp Shuffle Layout

Each warp (32 threads) contains two half-warps of 16 threads:
- Lanes 0-15: thread_rows 2k (even rows in the warp's section)
- Lanes 16-31: thread_rows 2k+1 (odd rows)

Within each half-warp, all 16 threads share the same `thread_row` value, so `__shfl_sync` across the half-warp broadcasts P values for the same Q-row to all column-threads.

Half-warp mask: `0x0000FFFF` for lanes 0-15, `0xFFFF0000` for lanes 16-31.

## Files

```
kernels/flash_attention/10b_flash_attn_v2_opt.cu    — kernel implementation
kernels/flash_attention/10b_flash_attn_v2_opt.cuh   — header declaration
benchmarks/attention_bench.cu                        — add K10b column
CMakeLists.txt                                       — add source file
```

## Validation

- Compare K10b output against CPU reference (same as K10)
- Tolerance: max absolute error < 1e-3
- Test configs: same as existing attention tests (causal + non-causal, multiple B/H/N/d)

## Expected Performance

| Optimization | Estimated contribution |
|---|---|
| Grid parallelism | 1.5-2x |
| float4 loads | 1.1-1.2x |
| BC 32→64 | 1.1-1.3x |
| No P_smem | 1.1-1.15x |
| **Combined** | **1.5-2x vs K10** |

Target: close to or match CUTLASS FMHA performance.
