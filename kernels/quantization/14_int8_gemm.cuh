// kernels/quantization/14_int8_gemm.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

// ============================================================
// INT8 GEMM constants
// ============================================================
#define I8_BM 64       // Tile rows
#define I8_BN 64       // Tile cols
#define I8_BK 16       // Tile K (in int8 elements)
#define I8_TM 4        // Thread tile rows
#define I8_TN 4        // Thread tile cols
#define I8_NTHREADS 256 // (BM/TM) * (BN/TN) = 16*16 = 256

// Quantize FP32 matrix to INT8 (packed int32) with per-row scales
// input: [rows, cols], output_packed: [rows, cols/4] as int32, scales: [rows]
void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,
                                int32_t* output_packed,
                                float* scales);

// INT8 GEMM: C_fp32 = dequant(A_int8 @ B_int8^T)
// A_packed: [M, K/4], BT_packed: [N, K/4] (both int8x4 packed as int32)
// scale_A: [M], scale_B: [N], C: [M, N] fp32
void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C);

// Epilogue types for fused INT8 GEMM
enum class I8Epilogue { Plain, BiasOnly, BiasGELU, BiasResidual };

// INT8 GEMM with bias: C = dequant(A @ B^T) + bias
void run_int8_gemm_bias(int M, int N, int K,
                        const int32_t* A_packed,
                        const int32_t* BT_packed,
                        const float* scale_A,
                        const float* scale_B,
                        const float* bias,
                        float* C);

// INT8 GEMM with bias + GELU
void run_int8_gemm_bias_gelu(int M, int N, int K,
                              const int32_t* A_packed,
                              const int32_t* BT_packed,
                              const float* scale_A,
                              const float* scale_B,
                              const float* bias,
                              float* C);

// INT8 GEMM with bias + residual
void run_int8_gemm_bias_residual(int M, int N, int K,
                                  const int32_t* A_packed,
                                  const int32_t* BT_packed,
                                  const float* scale_A,
                                  const float* scale_B,
                                  const float* bias,
                                  const float* residual,
                                  float* C);
