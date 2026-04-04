# SLICK — Speedy LLM Inference CUDA Kernels

## Build
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build

## Run benchmarks
./build/gemm_bench

## Project structure
- include/ — shared headers (timer, validator)
- kernels/ — CUDA kernel sources (.cu + .cuh per kernel)
- benchmarks/ — benchmark executables
- tests/ — validation tests

## Hardware
GTX 1650 Ti, CC 7.5, 4GB VRAM, NO Tensor Cores, CUDA 11.8

## Conventions
- Each kernel has a .cuh (declaration) and .cu (implementation)
- Host-callable wrappers: run_sgemm_<name>(M, N, K, A, B, C)
- Validate every kernel against cuBLAS: max |error| < 1e-5
- Row-major storage for all matrices
