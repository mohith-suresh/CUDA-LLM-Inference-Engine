# SLICK — Speedy LLM Inference CUDA Kernels

A from-scratch CUDA kernel library for LLM inference, targeting GTX 1650 Ti (CC 7.5, 4GB).

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
```

## Run

```bash
./build/gemm_bench    # GEMM kernel benchmark
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

Roofline: Peak FP32 = 3470 GFLOPS | Peak BW = 192 GB/s | Ridge = 18.1 FLOP/byte

## Roadmap

- [x] Week 1: Naive → Coalesced → Shared Tiling GEMM
- [x] Week 2: Register tiling, vectorized loads, double buffering + roofline analysis
- [ ] Week 3: Fused softmax kernels
- [ ] Week 4: FlashAttention-2
- [ ] Week 5: PagedAttention + GQA
- [ ] Week 6: Decode attention + INT8 GEMM
- [ ] Week 7: GPT-2 inference demo
