# Week 5: PagedAttention + GQA — Design Spec

## Overview

Kernel 11 (PagedAttention) and Kernel 12 (GQA) for SLICK. PagedAttention adds block-table indirection over the KV cache, enabling non-contiguous memory management for variable-length sequences. GQA extends this with grouped query heads sharing fewer KV heads.

## Hardware Constraints

- GTX 1650 Ti, CC 7.5, 4GB VRAM, FP32 only
- CUDA 11.8, no Tensor Cores
- Test sizes: B=1-4, H_q=8-32, H_kv=1-8, N=64-512, d=64

## Kernel 11: PagedAttention

### Concept

vLLM-style paged KV cache (Kwon et al., 2023). Instead of storing K/V contiguously per sequence, the cache is divided into fixed-size physical blocks. A block table maps logical block indices to physical block addresses, enabling:
- Zero internal fragmentation for variable-length sequences
- Dynamic memory allocation/deallocation per block
- Memory sharing across sequences (beam search, prefix caching)

### Data Structures

```
// Physical KV cache: non-contiguous blocks
float k_cache[num_physical_blocks][BLOCK_SIZE][H_kv][d]  // [num_blocks, 16, H, 64]
float v_cache[num_physical_blocks][BLOCK_SIZE][H_kv][d]

// Block table: logical → physical mapping per sequence
int block_table[B][max_blocks_per_seq]
// block_table[b][logical_idx] = physical_block_idx

// Context lengths per sequence
int context_lens[B]  // actual number of KV tokens per sequence
```

### Tiling & Launch

- **BLOCK_SIZE = 16** tokens per physical block
- **Br = 64** (Q tile rows, same as K10)
- **Bc = 16** (KV tile = one physical block, natural alignment)
- **d = 64** (head dim, not tiled)
- **Threads = 256** (8 warps)
- **Grid**: `(B, H_q, num_q_tiles)` — one block per (batch, head, q_tile)
  - Different from K10 which used `(B*H)` with Q tile loop inside
  - Exposing q_tile in grid enables better occupancy for long sequences

### Thread Tile

For S = Q @ K^T (Br=64, Bc=16):
- TM_S = 4, TN_S = 1 (16 thread_rows × 4 = 64 rows, 16 thread_cols × 1 = 16 cols)

For O accumulation (Br=64, d=64):
- TM_O = 4, TN_O = 4 (same as K10)

### Algorithm

```
for each q_tile (via blockIdx.z):
    Load Q_tile [Br x d] into smem

    for each logical_block in range(num_blocks_for_this_seq):
        phys_idx = block_table[batch][logical_block]
        kv_start = logical_block * BLOCK_SIZE

        // Causal skip
        if causal and kv_start > q_tile_end: break

        Load K from k_cache[phys_idx][:][kv_head][:]  → KV_smem [Bc x d]
        S = Q_smem @ KV_smem^T * scale                 [Br x Bc]
        Apply causal + boundary mask
        Online softmax: compute m_ij, l_ij, rescale O_acc

        Load V from v_cache[phys_idx][:][kv_head][:]  → KV_smem [Bc x d]
        O_acc += P @ V_smem

    Write O_tile [Br x d] to global (O /= l)
```

### Shared Memory

```
Q_smem:  [64][65]  = 16,640 B  (padded +1 to avoid bank conflicts)
KV_smem: [16][65]  =  4,160 B
P_smem:  [64][17]  =  4,352 B
Total:               25,152 B  (well under 48KB limit)
```

### KV Head Indexing

K11 uses `GROUP_SIZE=1` (standard MHA): `kv_head = q_head` (direct mapping).

### Host Wrapper

```cpp
void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal);
```

## Kernel 12: GQA (Grouped Query Attention)

### Concept

Multiple query heads share the same KV head: `kv_head = q_head / group_size`. Common in modern LLMs (Llama 2/3, Mistral) to reduce KV cache memory while maintaining query expressiveness.

### Implementation

Separate file `12_gqa.cu` that includes the shared kernel template from `11_paged_attn.cuh`. The kernel template is parameterized on `GROUP_SIZE`:

```cpp
template <int GROUP_SIZE, bool CAUSAL>
__global__ void paged_attn_kernel(...);
```

KV head resolution in kernel: `kv_head = blockIdx.y / GROUP_SIZE`

### Supported Group Sizes

Template instantiation for GROUP_SIZE = {1, 2, 4, 8}. Runtime dispatch via switch.

### Host Wrapper

```cpp
void run_gqa_paged_attn(int B, int H_q, int H_kv, int N, int d,
                        const float* Q,
                        const float* k_cache, const float* v_cache,
                        const int* block_table, const int* context_lens,
                        int max_context_len, int block_size,
                        int num_blocks_per_seq,
                        float* O, bool causal);
```

Where `group_size = H_q / H_kv`.

## File Structure

```
kernels/paged_attention/
    11_paged_attn.cuh   — kernel template <GROUP_SIZE, CAUSAL>
    11_paged_attn.cu    — host wrapper (GROUP_SIZE=1)
    12_gqa.cu           — host wrapper (GROUP_SIZE dispatch)
    12_gqa.cuh          — declaration for run_gqa_paged_attn
```

## Validation

- **K11**: Compare against K10 (FlashAttention) with contiguous block table mapping and same data. Tolerance: same as K10 (`< 1e-5`).
- **K12 with GROUP_SIZE=1**: Must match K11 output exactly.
- **K12 with GROUP_SIZE>1**: Compare against naive attention with GQA head mapping. Tolerance: `< 1e-5`.

## Benchmark

- Benchmark against K10 FlashAttention (contiguous baseline)
- Sweep: N = {128, 256, 512}, H_q = {8, 16, 32}, GROUP_SIZE = {1, 2, 4, 8}
- Report: GFLOPS, bandwidth utilization, overhead vs contiguous FlashAttention

## Build Integration

Add to CMakeLists.txt:
- `paged_attention_kernels` library with K11 + K12 source files
- `test_paged_attention` test executable
- `paged_attention_bench` benchmark executable
- Link against `attention_kernels` for K10 reference validation
