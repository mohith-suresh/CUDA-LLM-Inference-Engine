# SLICK — Speedy LLM Inference CUDA Kernels

A from-scratch CUDA kernel library for LLM inference, targeting GTX 1650 Ti (CC 7.5, 4GB).

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
```

## Run

```bash
./build/gemm_bench              # GEMM kernel benchmark
./build/softmax_bench           # Softmax kernel benchmark
./build/attention_bench         # FlashAttention-2 benchmark
./build/paged_attention_bench   # PagedAttention + GQA benchmark
```

## Test

Google Test (v1.14.0, fetched via CMake FetchContent). 79 test cases across 4 suites:

```bash
ctest --test-dir build --output-on-failure
```

| Suite | Tests | Reference | Tolerance | Coverage |
|-------|-------|-----------|-----------|----------|
| `GemmTests` | 42 | cuBLAS | 1e-4 | 7 kernels × 3 square + 3 rect sizes |
| `SoftmaxTests` | 14 | CPU 3-pass | 1e-6 | 2 kernels × 6 sizes + 2 edge cases |
| `AttentionTests` | 10 | CPU O(N²) | 1e-5 | causal/non-causal × 4 configs + 2 special |
| `PagedAttentionTests` | 13 | K10 / CPU O(N²) | 1e-5 | K11: 6 configs (causal/non-causal, multi-batch) + K12 GQA: 7 configs (group 1/2/4/8) |

Run a single suite: `./build/test_gemm`, `./build/test_softmax`, `./build/test_attention`, `./build/test_paged_attention`

## GEMM Kernels

| # | Kernel | Technique | GFLOPS (2048) | AI (FLOP/byte) | Bound |
|---|--------|-----------|--------------|-----------------|-------|
| 01 | Naive | 1 thread = 1 output element | 30 | 0.25 | Memory |
| 02 | Coalesced | threadIdx.x → col for coalesced reads | 351 | 0.25 | Memory |
| 03 | Shared Tiling | 32×32 shared memory tiles | 469 | 8.0 | Memory |
| 04 | 1D Reg Tiling | TM=8, register accumulation | 1094 | 16.0 | Memory |
| 05 | 2D Reg Tiling | TM=TN=8, 8×8 outer product | 1210 | 32.0 | Compute |
| 06 | Vectorized | float4 loads + transposed A in smem | 1689 | 32.0 | Compute |
| 07 | Double Buffered | 2× smem buffers, overlap load+compute | 1713 | 32.0 | Compute |

Roofline: Peak FP32 = 4300 GFLOPS (1024 cores @ 2100 MHz) | Peak BW = 192 GB/s | Ridge = 22.4 FLOP/byte

## Softmax Kernels

| # | Kernel | Technique | GB/s (512×4096) | %BW | vs cuDNN |
|---|--------|-----------|-----------------|-----|----------|
| 08 | Fused Online | Online algorithm, shared memory reduction | 166.3 | 86.6% | **1.50×** |
| 09 | Warp Reduce | Online algorithm, `__shfl_down_sync` reduction | 160.4 | 83.5% | **1.39×** |
| — | cuDNN 8.9.7 | `cudnnSoftmaxForward` (reference) | 148.0 | 77.1% | 1.00× |

Row-wise softmax using the online algorithm (Milakov & Gimelshein 2018). AI = 2.6 FLOP/byte — **deeply memory-bound** (8.7× below ridge point), so the only optimization lever is reducing DRAM traffic.

**Why we beat cuDNN:** The online algorithm reads input **twice** (12 bytes/elem: accumulate pass + normalize pass), while cuDNN's 3-pass approach reads it **three times** (16 bytes/elem: find max, exp+sum, normalize). That 25% traffic reduction is the entire margin.

The online `(max, sum_exp)` merge primitive feeds directly into FlashAttention (Week 4).

## FlashAttention-2

| # | Kernel | Technique | Config | Time (μs) | TFLOPS | vs Unfused |
|---|--------|-----------|--------|-----------|--------|------------|
| 10 | FlashAttention-2 | Fused QK^T + softmax + PV, online rescaling | B1 H12 N1024 d64 | 2744 | 1.17 | **1.32×** |
| 10 | FlashAttention-2 | Fused QK^T + softmax + PV, online rescaling | B4 H12 N512 d64 | 2047 | 1.57 | **1.96×** |
| — | Unfused baseline | cuBLAS SGEMM + Kernel 08 softmax + cuBLAS SGEMM | B1 H12 N1024 d64 | 3623 | 0.89 | 1.00× |

Single kernel fusing the entire multi-head attention: Q@K^T scoring, online softmax with warp shuffle reductions, and P@V output accumulation. The full N×N attention matrix never materializes in HBM — only a Br×Bc (64×32) tile exists transiently in shared memory/registers.

**Key design choices:**
- **Q-outer, KV-inner loop** with asymmetric tiling (Br=64, Bc=32) to balance Q reuse vs register pressure on SM75
- **Half-warp shuffle reductions** for row-wise softmax — 16 lanes reduce with `__shfl_down_sync`, no shared memory needed for reductions
- **Template causal mask** eliminates runtime branches; tile-level skip gives ~50% compute reduction
- **Online softmax rescaling** using the same `(max, sum_exp)` merge primitive from Kernel 08

Validation: max |error| < 2.4e-7 across all 6 test configs (B=1–2, N=128–1024, causal + non-causal).

## PagedAttention

| # | Kernel | Config | K10 (μs) | K11 (μs) | vs K10 | TFLOPS |
|---|--------|--------|----------|----------|--------|--------|
| 11 | PagedAttention | B1 H8 N128 d64 | 54.7 | 49.6 | **1.10×** | 0.68 |
| 11 | PagedAttention | B1 H8 N256 d64 | 175.1 | 137.1 | **1.28×** | 0.98 |
| 11 | PagedAttention | B1 H12 N256 d64 | 226.5 | 215.1 | **1.05×** | 0.94 |
| 11 | PagedAttention | B1 H12 N512 d64 | 774.6 | 649.8 | **1.19×** | 1.24 |
| 11 | PagedAttention | B2 H8 N256 d64 | 215.2 | 253.7 | 0.85× | 1.06 |

Single kernel implementing vLLM-style paged KV cache with block table indirection (block_size=16). Reuses the same fused Q@K^T + online softmax + P@V pipeline from FlashAttention (Kernel 10), with KV fetched from non-contiguous physical blocks via a per-sequence block table.

**Key design choices:**
- **Unified template kernel** for both MHA (K11, GROUP_SIZE=1) and GQA (K12, GROUP_SIZE>1) — `kv_head = q_head / GROUP_SIZE`
- **Asymmetric tiling** Br=64, Bc=16 (=block_size): each inner-loop step processes exactly one physical page, avoiding cross-block scatter
- **float4 vectorized loads** for Q, K, V cache and O output — 4× fewer global memory transactions
- **Half-warp shuffle softmax** (16 lanes) matches the Bc=16 tile width — no shared memory needed for reductions
- **Online rescaling** with the same `(max, sum_exp)` merge primitive from Kernels 08/10

K11 is 5–28% faster than K10 for single-batch configs (Bc=16 yields finer causal skip granularity + float4 vectorized paged cache loads), but shows 15% regression at B=2 due to block table indirection pressure with more grid blocks.

## GQA (Grouped-Query Attention)

| # | Kernel | H_q | H_kv | Group | N | Time (μs) | TFLOPS | KV Savings |
|---|--------|-----|------|-------|---|-----------|--------|------------|
| 12 | GQA PagedAttn | 8 | 8 | 1 | 256 | 131.3 | 1.02 | 1× |
| 12 | GQA PagedAttn | 8 | 4 | 2 | 256 | 133.2 | 1.01 | 2× |
| 12 | GQA PagedAttn | 8 | 2 | 4 | 256 | 128.4 | 1.05 | 4× |
| 12 | GQA PagedAttn | 8 | 1 | 8 | 256 | 129.9 | 1.03 | 8× |
| 12 | GQA PagedAttn | 32 | 4 | 8 | 256 | 429.7 | 1.25 | 8× |
| 12 | GQA PagedAttn | 32 | 4 | 8 | 512 | 1498.5 | 1.43 | 8× |

Kernel 12 dispatches the same paged attention template with compile-time GROUP_SIZE={1,2,4,8}. Multiple Q heads index the same KV head via `kv_head = q_head / GROUP_SIZE`, reducing KV cache memory proportionally while maintaining identical compute per Q head.

**GQA scaling:** GROUP_SIZE 1→8 gives ~5% latency reduction at fixed H_q=8 (from L2 cache reuse of shared KV blocks) while cutting KV memory by 8×. At H_q=32 N=512, the kernel reaches 1.43 TFLOPS (33% of peak FP32).

## Roadmap

- [x] Week 1: Naive → Coalesced → Shared Tiling GEMM
- [x] Week 2: Register tiling, vectorized loads, double buffering + roofline analysis
- [x] Week 3: Online softmax (fused + warp reduce)
- [x] Week 4: FlashAttention-2
- [x] Week 5: PagedAttention + GQA
- [ ] Week 6: Decode attention + INT8 GEMM
- [ ] Week 7: GPT-2 inference demo
