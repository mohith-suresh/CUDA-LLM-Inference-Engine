// kernels/quantization/14_int8_gemm.cu
#include "quantization/14_int8_gemm.cuh"

void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,
                                int32_t* output_packed,
                                float* scales) {
    // TODO: implement quantization kernel
}

void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C) {
    // TODO: implement int8 gemm kernel
}
