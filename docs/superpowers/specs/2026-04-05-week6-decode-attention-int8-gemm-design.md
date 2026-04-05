# Week 6 Design Spec: K13 Decode Attention + K14 INT8 GEMM

## Overview

Week 6 adds two kernels targeting autoregressive decode inference:
- **K13 Decode Attention**: Split-K parallel single-token attention over paged KV cache
- **K14 INT8 GEMM**: Quantized matrix multiply using `__dp4a()` intrinsic with separate quantization kernel

Both are critical for Week 7's GPT-2 speculative decoding demo — K13 handles token-by-token decode attention, K14 handles quantized linear layers.

## Hardware Constraints

- GTX 1650 Ti: 16 SMs, CC 7.5, 4GB VRAM, 191.9 GB/s bandwidth
- `__dp4a()` available (CC 6.1+): 4 INT8 MADs per instruction
- No Tensor Cores — dp4a is the only sub-word arithmetic available
- CUDA 11.8, cuBLAS 11.x

---

## Kernel 13: Decode Attention

### Problem

During autoregressive decoding, each step generates 1 token. The query is `[1, d]` attending over the full KV cache `[ctx_len, d]`. This is a matrix-vector product — FlashAttention's Q-tiling (Br=64) wastes 63/64 of compute. We need to parallelize across the KV sequence dimension instead.

### Architecture: Two-Pass Split-K

#### Pass 1 — Partial Attention Kernel

**Grid**: `(B, H_q, num_splits)`

Each threadblock handles a contiguous range of paged KV blocks:
- `blocks_per_split = ceil(num_kv_blocks / num_splits)`
- Split `i` processes KV blocks `[i * blocks_per_split, min((i+1) * blocks_per_split, num_kv_blocks))`

**Per-block computation:**
1. Load query vector `q[d]` into shared memory (one-time, float4 vectorized)
2. For each KV block in this split's range:
   - Look up physical block via `block_table[batch][logical_block]`
   - Load K row from paged cache: `k_cache[phys_block][slot][kv_head][d]`
   - Compute `s = dot(q, k) * scale` — warp-level parallel dot product across d
   - Track running softmax: `m_partial = max(m_partial, s)`, rescale, accumulate `l_partial`
   - Load V row, accumulate `o_partial += exp(s - m_partial) * v`
3. Write to workspace: `workspace[batch][head][split] = {o_partial[d], m_partial, l_partial}`

**Thread organization:**
- 256 threads (8 warps)
- Each warp processes KV tokens sequentially within its assigned range
- Within a warp: 32 lanes split the d=64 dimension (2 elements per lane for d=64, or 4 for d=128)
- Dot product via `__shfl_down_sync` reduction across lanes

#### Pass 2 — Reduction Kernel

**Grid**: `(B * H_q)`, one block per (batch, head) pair

1. Load all `num_splits` partial results from workspace
2. Merge using online softmax correction:
   ```
   for each split j:
     m_new = max(m_acc, m_j)
     alpha = exp(m_acc - m_new)
     beta  = exp(m_j - m_new)
     o_acc = o_acc * alpha + o_j * beta
     l_acc = l_acc * alpha + l_j * beta
     m_acc = m_new
   ```
3. Finalize: `O[batch][head][0][k] = o_acc[k] / l_acc`

### Split Count Heuristic

```
num_splits = clamp(num_kv_blocks / 4, 1, 16)
```

- Minimum 1 (short sequences)
- Maximum 16 (enough to saturate 16 SMs without excessive reduction)
- `/4` means each split handles at least 4 paged blocks (64 KV tokens at block_size=16)

### KV Cache Format

Reuses K11 paged cache layout exactly:
- `k_cache[num_phys_blocks][block_size][H_kv][d]`
- `v_cache[num_phys_blocks][block_size][H_kv][d]`
- `block_table[B][max_blocks_per_seq]` — logical-to-physical mapping
- `context_lens[B]` — per-sequence context length

### GQA Support

Template parameter `GROUP_SIZE` (same as K11/K12):
- `kv_head = q_head / GROUP_SIZE`
- `GROUP_SIZE=1` for MHA, `{2,4,8}` for GQA

### Workspace Memory

`workspace[B][H_q][max_splits][d + 2]` — floats

At max config (B=8, H=16, splits=16, d=64): `8 * 16 * 16 * 66 * 4 = 540 KB` — negligible.

### Causal Masking

Always-on for decode. The single query token at position `ctx_len - 1` attends to all positions `[0, ctx_len)`. No explicit mask needed — just bound by `ctx_len`.

### Validation

Compare against K11 PagedAttention with `N=1` (single query token). Tolerance: `< 1e-5` (split-K changes FP addition order, so exact bitwise match is unlikely despite identical algorithm).

### Host Wrapper

```cpp
void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,           // [B, H_q, 1, d]
                     const float* k_cache,     // paged
                     const float* v_cache,     // paged
                     const int* block_table,
                     const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O,                 // [B, H_q, 1, d]
                     float* workspace);        // [B, H_q, max_splits, d+2]
```

---

## Kernel 14: INT8 GEMM via dp4a

### Problem

INT8 quantization reduces memory footprint by 4x vs FP32 and `__dp4a()` computes 4 int8 multiply-adds per instruction. Essential for fitting GPT-2 weights in 4GB VRAM during Week 7.

### Architecture: Three Kernels

#### Kernel 14a — Quantize (FP32 → INT8)

Per-row symmetric quantization:

```
scale[i] = max(|row_i[0..K)|) / 127.0f
x_q[i][j] = clamp(round(x[i][j] / scale[i]), -128, 127)
```

**Output format:**
- `int8_t` matrix packed as `int32_t`: 4 consecutive int8 values in one int32
- Layout: `[rows, K/4]` as int32 (K padded to multiple of 4)
- `float scale[rows]` — per-row scale factors

**Grid**: `(M)` blocks, each block quantizes one row
- Phase 1: Cooperative max-abs reduction across threads (shared memory)
- Phase 2: Quantize + pack into int32

#### Kernel 14b — INT8 GEMM (dp4a)

**Layout — NT (A row-major, B stored transposed):**
- A: `[M, K/4]` as int32 — row-major, 4 int8s packed per element
- B^T: `[N, K/4]` as int32 — B transposed and packed the same way
- Both row-major → coalesced global loads for both operands

**Tiling:**
- `BM=64, BN=64, BK=16` (BK in int8 elements → 4 int32 packed values per K-tile)
- Shared memory: `A_smem[BM][BK/4]` as int32, `BT_smem[BN][BK/4]` as int32
- Bank conflict padding: +1 on the inner dimension

**Register tiling:**
- `TM=4, TN=4` — each thread owns a 4x4 int32 accumulator
- Thread layout: `(BM/TM) x (BN/TN) = 16 x 16 = 256` threads
- Inner loop per K-tile: 4 dp4a calls (BK/4 = 4 packed int32s)

```
for k4 in 0..BK/4:
    a_frag[tm] = A_smem[thread_row * TM + tm][k4]   // int32 packed
    b_frag[tn] = BT_smem[thread_col * TN + tn][k4]  // int32 packed
    acc[tm][tn] = __dp4a(a_frag[tm], b_frag[tn], acc[tm][tn])
```

**Epilogue — Dequantization (fused):**
- Load `scale_A[i]` and `scale_B[j]` into shared memory
- `C_fp32[i][j] = (float)C_int32[i][j] * scale_A[i] * scale_B[j]`
- Write FP32 output to global memory (float4 vectorized)

#### Why NT Layout

- `__dp4a(a, b, c)` interprets both `a` and `b` as packed int8x4 and computes `c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]`
- For `C = A @ B`, element `C[i][j] = sum_k A[i][k] * B[k][j]`
- Packing along K: `A[i][k..k+3]` → int32, `B^T[j][k..k+3]` → int32
- Both A rows and B^T rows are contiguous in memory → coalesced loads
- Alternative (B column-major) would require strided access — worse for coalescing

### Validation

- Compare dequantized INT8 GEMM output against FP32 GEMM (K06 vectorized)
- Tolerance: `< 0.05` per element (quantization noise from int8 rounding)
- Test sizes: 256, 512, 1024, 2048 (all square)

### Benchmark Baseline

- cuBLAS `cublasGemmEx` with:
  - A type: `CUDA_R_8I`, B type: `CUDA_R_8I`
  - C type: `CUDA_R_32I`, compute type: `CUBLAS_COMPUTE_32I`
- Report: effective GOPS (giga int8 operations per second), comparison ratio

### Host Wrappers

```cpp
// Quantize FP32 matrix to INT8 (packed int32) + scale vector
void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,      // [rows, cols]
                                int32_t* output_packed,   // [rows, cols/4]
                                float* scales);           // [rows]

// INT8 GEMM: C_fp32 = dequant(A_int8 @ B_int8^T)
void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,   // [M, K/4] int8x4
                   const int32_t* BT_packed,  // [N, K/4] int8x4
                   const float* scale_A,      // [M]
                   const float* scale_B,      // [N]
                   float* C);                 // [M, N] fp32
```

---

## File Structure

```
kernels/
  decode/
    13_decode_attn.cuh     # Kernel declarations + templates
    13_decode_attn.cu      # Host wrappers
  quantization/
    14_int8_gemm.cuh       # Kernel declarations + templates
    14_int8_gemm.cu        # Host wrappers
tests/
    test_decode_attention.cu
    test_int8_gemm.cu
benchmarks/
    decode_attention_bench.cu
    int8_gemm_bench.cu
```

## Test Configurations

### K13 Decode Attention
| B | H_q | H_kv | d  | ctx_len | block_size | GROUP_SIZE |
|---|------|------|----|---------|------------|------------|
| 1 | 8    | 8    | 64 | 128     | 16         | 1          |
| 4 | 16   | 16   | 64 | 512     | 16         | 1          |
| 1 | 16   | 4    | 64 | 256     | 16         | 4          |
| 8 | 32   | 8    | 64 | 1024    | 16         | 4          |

### K14 INT8 GEMM
| M    | N    | K    |
|------|------|------|
| 256  | 256  | 256  |
| 512  | 512  | 512  |
| 1024 | 1024 | 1024 |
| 2048 | 2048 | 2048 |
