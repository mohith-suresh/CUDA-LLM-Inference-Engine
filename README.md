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

## Week 1: GEMM Kernels

| # | Kernel | Technique | GFLOPS (2048x2048) |
|---|--------|-----------|--------------------|
| 01 | Naive | 1 thread = 1 output element | 29 |
| 02 | Coalesced | threadIdx.x → col for coalesced reads | 351 |
| 03 | Shared Tiling | 32x32 shared memory tiles | 472 |

## Roadmap

- [x] Week 1: Naive → Coalesced → Shared Tiling GEMM
- [ ] Week 2: Register tiling, vectorized loads, double buffering
- [ ] Week 3: Fused softmax kernels
- [ ] Week 4: FlashAttention-2
- [ ] Week 5: PagedAttention + GQA
- [ ] Week 6: Decode attention + INT8 GEMM
- [ ] Week 7: GPT-2 inference demo
