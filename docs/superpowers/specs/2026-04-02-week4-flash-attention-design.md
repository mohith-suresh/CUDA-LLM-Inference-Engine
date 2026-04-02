# Week 4 Design Spec — FlashAttention-2 Forward Pass (Kernel 10)

## Overview

Kernel 10 implements the FlashAttention-2 forward pass in FP32 for SM75 (GTX 1650 Ti). It fuses the entire multi-head attention computation — Q@K^T scoring, softmax, and P@V output accumulation — into a single kernel with O(N) HBM memory. The full N×N attention matrix never materializes; only a small Br×Bc tile exists transiently in registers/shared memory.

## Hardware Target

- GPU: GTX 1650 Ti (TU117), CC 7.5, 16 SMs, 4GB VRAM
- No Tensor Cores (FP32 CUDA cores only)
- Peak FP32: 4300 GFLOPS (1024 cores @ 2100 MHz)
- Peak BW: 192 GB/s (GDDR6, 128-bit @ 12 Gbps)
- Shared memory: 48KB per SM (configurable, up to 64KB)
- Max registers per thread: 255

## Target Workload

GPT-2 scale:
- Sequence length (N): up to 1024
- Head dimension (d): 64
- Number of heads (H): 12
- Batch size (B): 1–8

Memory footprint per (B, H): Q + K + V + O = 4 × N × d × 4 bytes. At B=1, H=12, N=1024, d=64: 4 × 12 × 1024 × 64 × 4 = 12.6 MB. Fits comfortably in 4GB.

## Algorithm

FlashAttention-2 forward pass with Q-outer, KV-inner loop ordering:

```
For each (batch, head) — one thread block:
  For i = 0 to ceil(N/Br) - 1:               // outer loop: Q tiles
    Load Q_i (Br × d) from HBM to shared memory
    Initialize: O_i = 0 (Br × d), m_i = -inf (Br,), l_i = 0 (Br,)

    For j = 0 to ceil(N/Bc) - 1:             // inner loop: KV tiles
      if causal and j * Bc > (i+1) * Br - 1: break

      Load K_j (Bc × d) from HBM to shared memory
      S_ij = Q_i @ K_j^T * (1 / sqrt(d))     // Br × Bc, in registers
      if causal: S_ij[r][c] = -inf where (i*Br+r) < (j*Bc+c)

      m_ij = rowmax(S_ij)                     // warp shuffle reduction
      P_ij = exp(S_ij - m_ij)                 // unnormalized local softmax
      l_ij = rowsum(P_ij)                     // warp shuffle reduction

      m_new = max(m_i, m_ij)
      O_i *= exp(m_i - m_new)                 // rescale old accumulator
      O_i += exp(m_ij - m_new) * P_ij @ V_j   // accumulate new contribution
      l_i = l_i * exp(m_i - m_new) + l_ij * exp(m_ij - m_new)
      m_i = m_new

    O_i /= l_i                                // final normalization
    Write O_i (Br × d) to HBM
```

The online softmax rescaling uses the same `(max, sum_exp)` merge primitive from Kernel 08. The O accumulator holds unnormalized values throughout; division by l happens once at the end.

## Tiling

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Br | 64 | Q tile rows — large for Q reuse across inner loop |
| Bc | 32 | KV tile rows — small to reduce register pressure on SM75 |
| d | 64 | Full head dimension, no tiling needed |

Hybrid asymmetric: large Br minimizes Q reloads (outer loop), small Bc keeps register budget under control for the S computation and P@V GEMM. Total FLOPs are identical to Bc=64; the difference is in register pressure and occupancy.

Iteration counts at N=1024: 16 outer × 32 inner = 512 tile pairs (non-causal). With causal mask: ~256 tile pairs (~50% reduction).

## Thread Mapping

**Thread block:** 256 threads (8 warps)

**Logical grid:** 16 × 16
- `thread_row = tid / 16` (0..15)
- `thread_col = tid % 16` (0..15)

**Warp alignment:**
- Warp k: tid [32k, 32k+31] → thread_rows [2k, 2k+1], thread_cols [0, 15]
- All 16 thread_cols for a given thread_row land in the same warp
- Enables efficient `__shfl_down_sync` for row-wise reductions

### S = Q @ K^T (64 × 32)

- TM = 4, TN = 2: each thread computes a 4×2 subtile = 8 S values
- Thread (tr, tc) owns S rows [tr×4, tr×4+4), S cols [tc×2, tc×2+2)
- 16 thread_rows × 16 thread_cols = 256 threads, covering 64 rows × 32 cols

**Row-wise softmax reduction:** For each S row, 16 threads each hold 2 values. Local max/sum across 2 values, then `__shfl_down_sync` reduction across 16 lanes (half-warp). All within the same warp — no shared memory needed for reductions.

### O accumulation (64 × 64)

- TM_o = 4, TN_o = 4: each thread accumulates a 4×4 subtile = 16 O values
- Thread (tr, tc) owns O rows [tr×4, tr×4+4), O cols [tc×4, tc×4+4)
- Same thread_rows as S, so per-row softmax state (m_i, l_i) stays in the same thread

### P@V GEMM resolution

Each thread has only 2 P values per row (from the 4×2 S tile), but the P@V GEMM (64×32 @ 32×64 → 64×64) needs the full 32-element P row. Solution: write P to shared memory after softmax, then run P@V as a standard smem-to-register GEMM.

For the P@V GEMM, each thread computes its 4×4 O subtile by iterating over the K=Bc=32 dimension:
```
for k = 0 to 31:
    for tm = 0 to 3:
        p_val = P_smem[tr*4 + tm][k]
        for tn = 0 to 3:
            O_reg[tm][tn] += p_val * V_smem[k][tc*4 + tn]
```

### Register budget

| Item | Count | Registers |
|------|-------|-----------|
| S fragment (4 × 2) | 8 | 8 |
| O accumulator (4 × 4) | 16 | 16 |
| Softmax state (m_i, l_i per TM row) | 4 + 4 | 8 |
| Q/K/V load temporaries | ~8 | 8 |
| Loop vars, pointers, scale, masks | ~8 | 8 |
| **Total** | | **~48** |

Well under the 255 limit. No spilling expected; ample room for compiler-generated temps.

## Shared Memory Layout

All arrays padded +1 float per row to eliminate bank conflicts (row width 64 is a multiple of 32 banks; padding to 65 offsets each row by 1 bank).

| Slot | Array | Dimensions | Bytes |
|------|-------|-----------|-------|
| Q | `Q_smem[64][65]` | Br × (d + 1) | 16,640 |
| KV | `KV_smem[32][65]` | Bc × (d + 1) | 8,320 |
| P | `P_smem[64][33]` | Br × (Bc + 1) | 8,448 |
| **Total** | | | **33,408 (32.6 KB)** |

Fits within 48KB. With ~48 registers/thread × 256 threads = 12,288 registers (of 65,536 per SM), occupancy is 1 block per SM with room for a second if register usage stays low.

### Memory timeline per inner iteration

```
Step   Q_smem      KV_smem     P_smem      Action
──────────────────────────────────────────────────────
1      Q_i         ← K_j       —           Load K from HBM
       __syncthreads
2      Q_i(read)   K_j(read)   —           S = Q @ K^T in registers
3      —           —           —           Softmax + rescale O (registers)
4      Q_i         K_j         ← P_ij      Write P from registers to smem
       __syncthreads
5      Q_i         ← V_j       P_ij        Load V from HBM (overwrites K)
       __syncthreads
6      Q_i         V_j(read)   P_ij(read)  O += P @ V in registers
```

3 `__syncthreads` per inner iteration. The two GEMM computations (steps 2 and 6) dominate runtime.

### Cooperative HBM loading

- Q load (once per outer): 64 × 64 = 4096 floats / 256 threads = 16 floats/thread
- K load (per inner): 32 × 64 = 2048 floats / 256 threads = 8 floats/thread
- V load (per inner): same as K

Each thread loads a contiguous chunk from HBM to registers, then writes to the padded smem location (index: `row * (width + 1) + col`).

## Causal Mask

### Level 1 — Tile skip

Skip entire KV tiles where all positions are above the causal diagonal:
```
if (causal && j * Bc > (i + 1) * Br - 1) break;
```

Since j increases across the KV sequence, once a tile is fully masked, all subsequent tiles are too. Using `break` (not `continue`) avoids wasted iterations.

### Level 2 — Element mask

For the tile straddling the diagonal, apply per-element masking after computing S:
```
global_row = i * Br + local_row
global_col = j * Bc + local_col
if (global_row < global_col) S[local_row][local_col] = -INFINITY
```

`exp(-inf) = 0`, so masked positions contribute nothing to P or O. At most one partial tile per Q tile needs element-wise masking.

### Compute savings

At N=1024 with causal mask:
- Non-causal: 16 outer × 32 inner = 512 tile pairs
- Causal: ~256 tile pairs (roughly N²/2 triangle)
- ~50% compute reduction

## Host Interface

```cpp
// kernels/flash_attention/10_flash_attn_v2.cuh
void run_flash_attn_v2(int B, int H, int N, int d,
                       const float* Q, const float* K, const float* V,
                       float* O, bool causal);
```

**Memory layout:** Row-major, contiguous `(B, H, N, d)`. For batch element b, head h, the Q slice starts at offset `((b * H + h) * N * d)`.

**Grid:** `(B * H)` blocks. Each block handles one (batch, head) pair.

**Scale factor:** `1.0f / sqrtf((float)d)` computed once on the host, passed as a kernel argument or computed in-kernel.

## Validation

### Reference implementation

CPU-side naive O(N²) attention:
```
S = Q @ K^T / sqrt(d)                    // N × N
if causal: S[r][c] = -inf where r < c
P = softmax(S, dim=-1)                    // row-wise, numerically stable 3-pass
O = P @ V                                 // N × d
```

Materializes full N×N matrix — acceptable at N≤1024 (4MB). Implemented in the benchmark harness, not a separate file.

### Tolerance

`max |O_ours - O_ref| < 1e-5`

The tolerance is less tight than softmax (1e-6) because FlashAttention chains two matrix multiplications and online softmax rescaling, accumulating more FP32 rounding.

### Test configurations

| B | H | N | d | Causal | Description |
|---|---|------|---|--------|-------------|
| 1 | 12 | 128 | 64 | yes | Small, quick validation |
| 1 | 12 | 256 | 64 | yes | Medium |
| 1 | 12 | 512 | 64 | yes | Standard context |
| 1 | 12 | 1024 | 64 | yes | Full GPT-2 |
| 2 | 12 | 512 | 64 | yes | Multi-batch |
| 1 | 12 | 512 | 64 | no | Non-causal check |

## Benchmark

### Reference comparison

**Primary: Unfused cuBLAS + softmax baseline**
```
S = cublas_sgemm(Q, K^T)                  // N × N materialized in HBM
P = run_softmax_fused_online(S, N, N)     // our Kernel 08
O = cublas_sgemm(P, V)                    // N × d
```

This shows the fusion benefit: FlashAttention avoids the N×N HBM round-trip.

**Secondary: CUTLASS FMHA (if SM75 compatible)**

Attempt CUTLASS 2.11 via CMake FetchContent (header-only, pinned tag). CUTLASS ships FMHA examples but they primarily target SM80+. If SM75 FMHA is available, include it as a benchmark. If not, the unfused baseline is the comparison point.

CUTLASS integration is isolated to the benchmark — the kernel itself has no CUTLASS dependency.

### Metrics

- Time (μs): averaged over 10 runs, 3 warmup
- TFLOPS: `2 × B × H × (2 × N² × d) / time` (two matmuls: QK^T and PV)
- HBM bandwidth: bytes loaded (Q + K + V) + written (O) vs peak 192 GB/s
- vs-unfused speedup ratio

### Benchmark sizes

| B | H | N | d | Description |
|---|---|------|---|-------------|
| 1 | 12 | 256 | 64 | Small |
| 1 | 12 | 512 | 64 | Medium |
| 1 | 12 | 1024 | 64 | Full GPT-2 |
| 4 | 12 | 512 | 64 | Multi-batch |

## Files

| File | Purpose |
|------|---------|
| `kernels/flash_attention/10_flash_attn_v2.cuh` | Header — function declaration |
| `kernels/flash_attention/10_flash_attn_v2.cu` | Implementation — kernel + host wrapper |
| `benchmarks/attention_bench.cu` | Benchmark + validation harness |
| `CMakeLists.txt` | Updated — attention_kernels lib + attention_bench target |

## Build Integration

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

CUTLASS (if used) is fetched via FetchContent and linked only to `attention_bench`, not to the kernel library.
