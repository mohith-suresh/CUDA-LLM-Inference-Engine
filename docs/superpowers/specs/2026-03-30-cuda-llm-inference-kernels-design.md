# SLICK — CUDA LLM Inference Kernels: Design Spec

## Overview

SLICK (Speedy LLM Inference CUDA Kernels) is a from-scratch CUDA kernel library implementing the core compute primitives for LLM inference. The project progressively builds 14 CUDA kernels across 7 weeks, starting from naive GEMM and ending with a GPT-2 inference demo.

## Hardware Target

- **GPU**: NVIDIA GeForce GTX 1650 Ti (TU117, 4GB VRAM)
- **Compute Capability**: 7.5 (Turing)
- **Tensor Cores**: NOT available (GTX 16xx series lacks Tensor Core hardware)
- **dp4a (INT8)**: Available (CC 6.1+)
- **CUDA Toolkit**: 10.1 (V10.1.243)
- **cuBLAS**: 10.2 (reference validator)
- **Max safe FP32 matrix**: 2048x2048 (48MB for 3 matrices)

## Project Structure

```
SLICK/
├── CMakeLists.txt
├── CLAUDE.md
├── .gitignore
├── include/
│   ├── timer.cuh            # GPU event-based timing, GFLOPS calculation
│   └── validator.cuh        # cuBLAS SGEMM reference, max-error comparison
├── kernels/
│   ├── gemm/                # Week 1-2: Kernels 1-7
│   │   ├── 01_naive.cu
│   │   ├── 02_coalesced.cu
│   │   ├── 03_shared_tiling.cu
│   │   ├── 04_1d_reg_tiling.cu
│   │   ├── 05_2d_reg_tiling.cu
│   │   ├── 06_vectorized.cu
│   │   └── 07_double_buffered.cu
│   ├── softmax/             # Week 3: Kernels 8-9
│   │   ├── 08_fused_online.cu
│   │   └── 09_warp_reduce.cu
│   ├── flash_attention/     # Week 4: Kernel 10
│   │   └── 10_flash_attn_v2.cu
│   ├── paged_attention/     # Week 5: Kernels 11-12
│   │   ├── 11_paged_attn.cu
│   │   └── 12_gqa.cu
│   ├── decode/              # Week 6: Kernel 13
│   │   └── 13_decode_attn.cu
│   └── quantization/        # Week 6: Kernel 14
│       └── 14_int8_gemm.cu
├── benchmarks/
│   ├── gemm_bench.cu
│   ├── softmax_bench.cu
│   └── attention_bench.cu
├── tests/
│   └── validate.cu
├── python/
│   ├── bindings.cpp         # Week 7: pybind11 GPT-2 demo
│   └── gpt2_inference.py
└── scripts/
    └── plot_results.py
```

## Build System

- **CMake** (install via apt or pip)
- Single `CMakeLists.txt` at root
- Auto-detect GPU architecture (fallback to `sm_75`)
- Link cuBLAS for reference validation
- Targets:
  - `gemm_bench` — runs all GEMM kernels, prints GFLOPS table
  - `softmax_bench` — softmax kernels
  - `attention_bench` — attention kernels
  - `validate` — correctness checks for all kernels
  - Individual kernel executables for NCU profiling

## Kernel Summary

| # | Kernel | Week | Category | Key Technique |
|---|--------|------|----------|---------------|
| 1 | Naive GEMM | 1 | GEMM | 1 thread = 1 output element |
| 2 | Coalesced GEMM | 1 | GEMM | Row-major thread indexing for coalesced reads |
| 3 | Shared Memory Tiling | 1 | GEMM | 32x32 tiles in shared memory |
| 4 | 1D Register Tiling | 2 | GEMM | TM=8, each thread computes a column |
| 5 | 2D Register Tiling | 2 | GEMM | TM=TN=8 micro-tile per thread |
| 6 | Vectorized Loads | 2 | GEMM | float4 loads |
| 7 | Double Buffering | 2 | GEMM | 2x shared memory buffers, overlap load+compute |
| 8 | Fused Online Softmax | 3 | Softmax | Single-pass: running max + running sum of exp |
| 9 | Warp Reduce Softmax | 3 | Softmax | __shfl_down_sync warp-level reduction |
| 10 | FlashAttention-2 | 4 | Attention | Tiled Q/K/V, online softmax, O(N) memory |
| 11 | PagedAttention | 5 | Attention | Block table indirection for KV cache |
| 12 | GQA | 5 | Attention | kv_head = q_head / group_size |
| 13 | Decode Attention | 6 | Attention | Matrix-vector, parallelize across KV length |
| 14 | INT8 GEMM | 6 | Quantization | int8 inputs, int32 accum, dp4a, dequantize |

## Validation Strategy

### Correctness
- **GEMM kernels (1-7)**: Compare against cuBLAS SGEMM. Tolerance: `max |C_yours - C_cublas| < 1e-5`
- **Softmax (8-9)**: Compare against CPU reference (numerically stable log-sum-exp). Tolerance: `< 1e-6`
- **FlashAttention (10)**: Compare against naive O(N^2) attention. Tolerance: `< 1e-5`
- **PagedAttention (11-12)**: Bitwise match against FlashAttention with same data
- **Decode Attention (13)**: Compare against FlashAttention with seq_len_q=1
- **INT8 GEMM (14)**: Compare dequantized output against FP32 GEMM. Tolerance: `< 0.05`

### Performance
- Report GFLOPS achieved and % of theoretical peak
- Test sizes: 256, 512, 1024, 2048 (all fit in 4GB)
- GPU event-based timing (cudaEvent, averaged over multiple runs)

## Shared Utilities

### timer.cuh
- `GpuTimer` class wrapping cudaEventCreate/Record/Synchronize/ElapsedTime
- `compute_gflops(M, N, K, time_ms)` helper
- Warmup runs + averaged timing

### validator.cuh
- `cublas_sgemm_reference(A, B, C_ref, M, N, K)` — runs cuBLAS as ground truth
- `validate(C_test, C_ref, M, N, tolerance)` — returns max error, prints PASS/FAIL
- Random matrix initialization with fixed seed for reproducibility

## Weekly Milestones

- **Week 1**: Kernels 1-3 + build system + timing/validation framework
- **Week 2**: Kernels 4-7 + NCU profiling comparison
- **Week 3**: Kernels 8-9 (skip Tensor Core GEMM — no TC hardware)
- **Week 4**: Kernel 10 (FlashAttention-2 forward + causal mask)
- **Week 5**: Kernels 11-12 (PagedAttention + GQA + block allocator)
- **Week 6**: Kernels 13-14 (Decode attention + INT8 GEMM)
- **Week 7**: GPT-2 inference demo + polish + benchmarks

## Constraints

- All kernels must work within 4GB VRAM
- FP32 only (no FP16 — no Tensor Cores)
- INT8 via dp4a (Week 6 only)
- No external dependencies beyond CUDA toolkit, cuBLAS, and Python stdlib+numpy
- pybind11 added only in Week 7 for the GPT-2 demo
