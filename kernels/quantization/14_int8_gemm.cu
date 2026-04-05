// kernels/quantization/14_int8_gemm.cu
#include "quantization/14_int8_gemm.cuh"

// ============================================================
// Quantization kernel: FP32 -> INT8 (per-row symmetric)
// One block per row. Threads cooperatively:
//   Phase 1: find max(|row|) via shared memory reduction
//   Phase 2: quantize + pack 4 int8s into int32
// ============================================================
__global__ void quantize_fp32_to_int8_kernel(
    int cols,
    const float* __restrict__ input,     // [rows, cols]
    int32_t* __restrict__ output_packed, // [rows, cols/4]
    float* __restrict__ scales)          // [rows]
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_ptr = input + row * cols;

    // Phase 1: cooperative max-abs reduction
    extern __shared__ float smem[];  // blockDim.x floats

    float local_max = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = fabsf(row_ptr[i]);
        if (val > local_max) local_max = val;
    }
    smem[tid] = local_max;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    float row_max = smem[0];
    float scale = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0) scales[row] = scale;
    __syncthreads();

    // Phase 2: quantize and pack 4 int8s into int32
    int packed_cols = cols / 4;
    int32_t* out_row = output_packed + row * packed_cols;

    for (int i = tid; i < packed_cols; i += blockDim.x) {
        int base = i * 4;
        int8_t q0 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 0] * inv_scale), -128.0f), 127.0f));
        int8_t q1 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 1] * inv_scale), -128.0f), 127.0f));
        int8_t q2 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 2] * inv_scale), -128.0f), 127.0f));
        int8_t q3 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 3] * inv_scale), -128.0f), 127.0f));

        // Pack: byte 0 = q0, byte 1 = q1, byte 2 = q2, byte 3 = q3
        int32_t packed = 0;
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q0)));
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);
        out_row[i] = packed;
    }
}

void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,
                                int32_t* output_packed,
                                float* scales) {
    int threads = 256;
    int smem = threads * sizeof(float);
    quantize_fp32_to_int8_kernel<<<rows, threads, smem>>>(
        cols, input, output_packed, scales);
}

void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C) {
    // TODO: implement dp4a GEMM kernel
}
