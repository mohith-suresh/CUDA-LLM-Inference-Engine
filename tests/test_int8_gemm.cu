// tests/test_int8_gemm.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "timer.cuh"
#include "validator.cuh"
#include "gemm/06_vectorized.cuh"
#include "quantization/14_int8_gemm.cuh"

class Int8GemmTest : public ::testing::Test {
protected:
    void RunInt8vsFloat(int M, int N, int K, float tol = 0.05f) {
        int sizeA = M * K;
        int sizeB = K * N;
        int sizeC = M * N;

        // Host FP32 matrices
        std::vector<float> h_A(sizeA), h_B(sizeB);
        srand(42);
        for (int i = 0; i < sizeA; ++i)
            h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < sizeB; ++i)
            h_B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        // Device FP32
        float *d_A, *d_B, *d_C_ref, *d_C_int8;
        CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_ref, sizeC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_int8, sizeC * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

        // FP32 reference (K06)
        run_sgemm_vectorized(M, N, K, d_A, d_B, d_C_ref);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Quantize A [M, K] and B^T [N, K]
        // B is [K, N] row-major. B^T is [N, K] row-major.
        std::vector<float> h_BT(N * K);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                h_BT[n * K + k] = h_B[k * N + n];

        float *d_BT;
        CUDA_CHECK(cudaMalloc(&d_BT, N * K * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_BT, h_BT.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

        // Quantized buffers
        int32_t *d_A_packed, *d_BT_packed;
        float *d_scale_A, *d_scale_B;
        CUDA_CHECK(cudaMalloc(&d_A_packed, M * (K / 4) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_BT_packed, N * (K / 4) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_scale_A, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_B, N * sizeof(float)));

        // Quantize
        run_quantize_fp32_to_int8(M, K, d_A, d_A_packed, d_scale_A);
        run_quantize_fp32_to_int8(N, K, d_BT, d_BT_packed, d_scale_B);
        CUDA_CHECK(cudaDeviceSynchronize());

        // INT8 GEMM
        run_int8_gemm(M, N, K, d_A_packed, d_BT_packed, d_scale_A, d_scale_B, d_C_int8);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compare
        std::vector<float> h_C_ref(sizeC), h_C_int8(sizeC);
        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_C_int8.data(), d_C_int8, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        float sum_err = 0.0f;
        for (int i = 0; i < sizeC; ++i) {
            float err = fabsf(h_C_ref[i] - h_C_int8[i]);
            sum_err += err;
            if (err > max_err) max_err = err;
        }
        float avg_err = sum_err / sizeC;

        EXPECT_LT(max_err, tol) << "INT8 vs FP32 max error: " << max_err
                                 << " avg: " << avg_err
                                 << " (M=" << M << " N=" << N << " K=" << K << ")";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_BT);
        cudaFree(d_C_ref); cudaFree(d_C_int8);
        cudaFree(d_A_packed); cudaFree(d_BT_packed);
        cudaFree(d_scale_A); cudaFree(d_scale_B);
    }
};

TEST_F(Int8GemmTest, K14_256)  { RunInt8vsFloat(256, 256, 256); }
TEST_F(Int8GemmTest, K14_512)  { RunInt8vsFloat(512, 512, 512); }
TEST_F(Int8GemmTest, K14_1024) { RunInt8vsFloat(1024, 1024, 1024); }
TEST_F(Int8GemmTest, K14_2048) { RunInt8vsFloat(2048, 2048, 2048); }
TEST_F(Int8GemmTest, K14_Rect) { RunInt8vsFloat(512, 1024, 256); }
