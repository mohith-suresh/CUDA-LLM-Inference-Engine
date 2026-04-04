# SLICK — Speedy LLM Inference CUDA Kernels

A from-scratch CUDA kernel library for LLM inference, targeting GTX 1650 Ti (CC 7.5, 4GB).

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
```

## Run

```bash
./build/gemm_bench        # GEMM kernel benchmark
./build/softmax_bench     # Softmax kernel benchmark
./build/attention_bench   # FlashAttention-2 benchmark
```

## Test

Google Test (v1.14.0, fetched via CMake FetchContent). 66 test cases across 3 suites:

```bash
ctest --test-dir build --output-on-failure
```

| Suite | Tests | Reference | Tolerance | Coverage |
|-------|-------|-----------|-----------|----------|
| `GemmTests` | 42 | cuBLAS | 1e-4 | 7 kernels × 3 square + 3 rect sizes |
| `SoftmaxTests` | 14 | CPU 3-pass | 1e-6 | 2 kernels × 6 sizes + 2 edge cases |
| `AttentionTests` | 10 | CPU O(N²) | 1e-5 | causal/non-causal × 4 configs + 2 special |

Run a single suite: `./build/test_gemm`, `./build/test_softmax`, `./build/test_attention`

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

## Roadmap

- [x] Week 1: Naive → Coalesced → Shared Tiling GEMM
- [x] Week 2: Register tiling, vectorized loads, double buffering + roofline analysis
- [x] Week 3: Online softmax (fused + warp reduce)
- [x] Week 4: FlashAttention-2
- [ ] Week 5: PagedAttention + GQA
- [ ] Week 6: Decode attention + INT8 GEMM
- [ ] Week 7: GPT-2 inference demo
