#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include "timer.cuh"
#include "validator.cuh"
#include "gemm/01_naive.cuh"
#include "gemm/02_coalesced.cuh"
#include "gemm/03_shared_tiling.cuh"
#include "gemm/04_reg_tiling_1d.cuh"
#include "gemm/05_reg_tiling_2d.cuh"
#include "gemm/06_vectorized.cuh"
#include "gemm/07_double_buffered.cuh"

class GemmTest : public ::testing::Test {
protected:
    CublasValidator validator;
    float *d_A, *d_B, *d_C, *d_C_ref;
    int M, N, K;

    void SetUpMatrices(int m, int n, int k) {
        M = m; N = n; K = k;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

        init_random_matrix(d_A, M * K, 42);
        init_random_matrix(d_B, K * N, 123);

        // cuBLAS reference
        validator.sgemm(M, N, K, d_A, d_B, d_C_ref);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void RunAndCheck(GemmFn fn, float tol = 1e-4f) {
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        fn(M, N, K, d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());
        float err = max_error(d_C, d_C_ref, M * N);
        EXPECT_LT(err, tol) << "Max error: " << err;
    }

    void TearDown() override {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (d_C_ref) cudaFree(d_C_ref);
    }
};

// --- Square matrices (256, 512, 1024) ---

class GemmSquareTest : public GemmTest,
                       public ::testing::WithParamInterface<int> {
protected:
    void SetUp() override { SetUpMatrices(GetParam(), GetParam(), GetParam()); }
};

TEST_P(GemmSquareTest, Kernel01Naive)          { RunAndCheck(run_sgemm_naive); }
TEST_P(GemmSquareTest, Kernel02Coalesced)      { RunAndCheck(run_sgemm_coalesced); }
TEST_P(GemmSquareTest, Kernel03SharedTiling)   { RunAndCheck(run_sgemm_shared_tiling); }
TEST_P(GemmSquareTest, Kernel04RegTiling1D)    { RunAndCheck(run_sgemm_reg_tiling_1d); }
TEST_P(GemmSquareTest, Kernel05RegTiling2D)    { RunAndCheck(run_sgemm_reg_tiling_2d); }
TEST_P(GemmSquareTest, Kernel06Vectorized)     { RunAndCheck(run_sgemm_vectorized); }
TEST_P(GemmSquareTest, Kernel07DoubleBuf)      { RunAndCheck(run_sgemm_double_buffered); }

INSTANTIATE_TEST_SUITE_P(Sizes, GemmSquareTest, ::testing::Values(256, 512, 1024));

// --- Non-square matrices ---

struct RectParams {
    int M, N, K;
};

class GemmRectTest : public GemmTest,
                     public ::testing::WithParamInterface<RectParams> {
protected:
    void SetUp() override {
        auto p = GetParam();
        SetUpMatrices(p.M, p.N, p.K);
    }
};

TEST_P(GemmRectTest, Kernel01Naive)          { RunAndCheck(run_sgemm_naive); }
TEST_P(GemmRectTest, Kernel02Coalesced)      { RunAndCheck(run_sgemm_coalesced); }
TEST_P(GemmRectTest, Kernel03SharedTiling)   { RunAndCheck(run_sgemm_shared_tiling); }
TEST_P(GemmRectTest, Kernel04RegTiling1D)    { RunAndCheck(run_sgemm_reg_tiling_1d); }
TEST_P(GemmRectTest, Kernel05RegTiling2D)    { RunAndCheck(run_sgemm_reg_tiling_2d); }
TEST_P(GemmRectTest, Kernel06Vectorized)     { RunAndCheck(run_sgemm_vectorized); }
TEST_P(GemmRectTest, Kernel07DoubleBuf)      { RunAndCheck(run_sgemm_double_buffered); }

INSTANTIATE_TEST_SUITE_P(Shapes, GemmRectTest, ::testing::Values(
    RectParams{128, 256, 512},
    RectParams{512, 128, 256},
    RectParams{256, 512, 128}
));
