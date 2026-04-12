// kernels/layernorm/15_layernorm_residual.cuh
#pragma once
#include <cuda_runtime.h>

// Fused LayerNorm + Residual: out = LayerNorm(x + residual), residual_out = x + residual
// One block per row. Cols should be a multiple of 4 for best perf.
void run_layernorm_residual(int rows, int cols,
                             const float* x,
                             const float* residual,
                             const float* gamma,
                             const float* beta,
                             float* out,
                             float* residual_out,
                             float eps = 1e-5f);

// LayerNorm only (no residual add) — used for final ln_f
void run_layernorm(int rows, int cols,
                    const float* x,
                    const float* gamma,
                    const float* beta,
                    float* out,
                    float eps = 1e-5f);
