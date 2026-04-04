#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include "timer.cuh"
#include "softmax/08_fused_online.cuh"
#include "softmax/09_warp_reduce.cuh"

static void softmax_cpu_reference(const float* input, float* output,
                                  int N_rows, int N_cols) {
    for (int r = 0; r < N_rows; ++r) {
        const float* row_in = input + r * N_cols;
        float* row_out = output + r * N_cols;

        float max_val = -FLT_MAX;
        for (int c = 0; c < N_cols; ++c)
            if (row_in[c] > max_val) max_val = row_in[c];

        float sum = 0.0f;
        for (int c = 0; c < N_cols; ++c) {
            row_out[c] = expf(row_in[c] - max_val);
            sum += row_out[c];
        }
        for (int c = 0; c < N_cols; ++c)
            row_out[c] /= sum;
    }
}

typedef void (*SoftmaxFn)(const float*, float*, int, int);

class SoftmaxTest : public ::testing::Test {
protected:
    float *d_input, *d_output;
    float *h_input, *h_ref;
    int N_rows, N_cols, total;

    void SetUpMatrix(int rows, int cols) {
        N_rows = rows; N_cols = cols; total = rows * cols;
        CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));

        h_input = new float[total];
        h_ref = new float[total];
        srand(42);
        for (int i = 0; i < total; ++i)
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f;

        CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float),
                              cudaMemcpyHostToDevice));
        softmax_cpu_reference(h_input, h_ref, N_rows, N_cols);
    }

    void RunAndCheck(SoftmaxFn fn, float tol = 1e-6f) {
        CUDA_CHECK(cudaMemset(d_output, 0, total * sizeof(float)));
        fn(d_input, d_output, N_rows, N_cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_output = new float[total];
        CUDA_CHECK(cudaMemcpy(h_output, d_output, total * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < total; ++i) {
            float err = fabsf(h_output[i] - h_ref[i]);
            if (err > max_err) max_err = err;
        }
        delete[] h_output;
        EXPECT_LT(max_err, tol) << "Max error: " << max_err;
    }

    void TearDown() override {
        delete[] h_input; delete[] h_ref;
        cudaFree(d_input); cudaFree(d_output);
    }
};

struct SoftmaxParams {
    int rows, cols;
};

class SoftmaxParamTest : public SoftmaxTest,
                         public ::testing::WithParamInterface<SoftmaxParams> {
protected:
    void SetUp() override { SetUpMatrix(GetParam().rows, GetParam().cols); }
};

TEST_P(SoftmaxParamTest, Kernel08FusedOnline) { RunAndCheck(run_softmax_fused_online); }
TEST_P(SoftmaxParamTest, Kernel09WarpReduce)  { RunAndCheck(run_softmax_warp_reduce); }

INSTANTIATE_TEST_SUITE_P(Sizes, SoftmaxParamTest, ::testing::Values(
    SoftmaxParams{64, 128},
    SoftmaxParams{64, 1024},
    SoftmaxParams{128, 512},
    SoftmaxParams{128, 2048},
    SoftmaxParams{256, 4096},
    SoftmaxParams{512, 1024}
));

// Edge case: single row
TEST_F(SoftmaxTest, SingleRow) {
    SetUpMatrix(1, 256);
    RunAndCheck(run_softmax_fused_online);
    RunAndCheck(run_softmax_warp_reduce);
}

// Edge case: large number of small rows
TEST_F(SoftmaxTest, ManySmallRows) {
    SetUpMatrix(1024, 64);
    RunAndCheck(run_softmax_fused_online);
    RunAndCheck(run_softmax_warp_reduce);
}
