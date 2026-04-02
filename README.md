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

| # | Kernel | Technique | GB/s (512×1024) | GB/s (512×4096) |
|---|--------|-----------|-----------------|-----------------|
| 08 | Fused Online | Online algorithm, shared memory reduction | 114.0 | 113.7 |
| 09 | Warp Reduce | Online algorithm, `__shfl_down_sync` reduction | 104.5 | 107.5 |

Row-wise softmax using the online algorithm (Milakov & Gimelshein 2018): 2 passes over global memory instead of 3. Peak ~114 GB/s of 192 GB/s bandwidth ceiling (59%). The online `(max, sum_exp)` merge primitive feeds directly into FlashAttention (Week 4).

## Roadmap

- [x] Week 1: Naive → Coalesced → Shared Tiling GEMM
- [x] Week 2: Register tiling, vectorized loads, double buffering + roofline analysis
- [x] Week 3: Online softmax (fused + warp reduce)
- [ ] Week 4: FlashAttention-2
- [ ] Week 5: PagedAttention + GQA
- [ ] Week 6: Decode attention + INT8 GEMM
- [ ] Week 7: GPT-2 inference demo
