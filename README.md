# SLICK — Speedy LLM Inference CUDA Kernels

A from-scratch CUDA kernel library for LLM inference, targeting GTX 1650 Ti (CC 7.5, 4GB).

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
```

## Run

```bash
./build/gemm_bench      # GEMM kernel benchmark
./build/softmax_bench   # Softmax kernel benchmark
```

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

## Roadmap

- [x] Week 1: Naive → Coalesced → Shared Tiling GEMM
- [x] Week 2: Register tiling, vectorized loads, double buffering + roofline analysis
- [x] Week 3: Online softmax (fused + warp reduce)
- [ ] Week 4: FlashAttention-2
- [ ] Week 5: PagedAttention + GQA
- [ ] Week 6: Decode attention + INT8 GEMM
- [ ] Week 7: GPT-2 inference demo
