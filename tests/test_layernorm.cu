// tests/test_layernorm.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "layernorm/15_layernorm_residual.cuh"

#define CUDA_CHECK(call) \
    do { cudaError_t e = (call); ASSERT_EQ(e, cudaSuccess) << cudaGetErrorString(e); } while (0)

class LayerNormTest : public ::testing::Test {
protected:
    void cpu_layernorm_residual(int rows, int cols, const float* x, const float* residual,
                                 const float* gamma, const float* beta,
                                 float* out, float* residual_out, float eps = 1e-5f) {
        for (int r = 0; r < rows; ++r) {
            std::vector<float> y(cols);
            for (int c = 0; c < cols; ++c) {
                y[c] = x[r * cols + c] + residual[r * cols + c];
                residual_out[r * cols + c] = y[c];
            }
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) mean += y[c];
            mean /= cols;
            float var = 0.0f;
            for (int c = 0; c < cols; ++c) var += (y[c] - mean) * (y[c] - mean);
            var /= cols;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int c = 0; c < cols; ++c)
                out[r * cols + c] = gamma[c] * (y[c] - mean) * inv_std + beta[c];
        }
    }

    void RunTest(int rows, int cols) {
        float tol = 2e-4f;
        std::vector<float> h_x(rows * cols), h_res(rows * cols);
        std::vector<float> h_gamma(cols), h_beta(cols);
        srand(42);
        for (auto& v : h_x) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_res) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_gamma) v = 0.8f + (float)rand() / RAND_MAX * 0.4f;
        for (auto& v : h_beta) v = (float)rand() / RAND_MAX * 0.2f - 0.1f;

        std::vector<float> h_ref_out(rows * cols), h_ref_res_out(rows * cols);
        cpu_layernorm_residual(rows, cols, h_x.data(), h_res.data(),
                               h_gamma.data(), h_beta.data(),
                               h_ref_out.data(), h_ref_res_out.data());

        float *d_x, *d_res, *d_gamma, *d_beta, *d_out, *d_res_out;
        CUDA_CHECK(cudaMalloc(&d_x, rows * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_res, rows * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_res_out, rows * cols * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        run_layernorm_residual(rows, cols, d_x, d_res, d_gamma, d_beta, d_out, d_res_out);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(rows * cols), h_res_result(rows * cols);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res_result.data(), d_res_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err_out = 0.0f, max_err_res = 0.0f;
        for (int i = 0; i < rows * cols; ++i) {
            max_err_out = fmaxf(max_err_out, fabsf(h_out[i] - h_ref_out[i]));
            max_err_res = fmaxf(max_err_res, fabsf(h_res_result[i] - h_ref_res_out[i]));
        }
        EXPECT_LT(max_err_out, tol) << "LN output err=" << max_err_out;
        EXPECT_LT(max_err_res, tol) << "Residual output err=" << max_err_res;

        cudaFree(d_x); cudaFree(d_res); cudaFree(d_gamma);
        cudaFree(d_beta); cudaFree(d_out); cudaFree(d_res_out);
    }
};

TEST_F(LayerNormTest, K15_1x768)    { RunTest(1, 768); }
TEST_F(LayerNormTest, K15_128x768)  { RunTest(128, 768); }
TEST_F(LayerNormTest, K15_1x256)    { RunTest(1, 256); }
TEST_F(LayerNormTest, K15_64x768)   { RunTest(64, 768); }
TEST_F(LayerNormTest, K15_512x768)  { RunTest(512, 768); }
