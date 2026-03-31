#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "timer.cuh"

struct CublasValidator {
    cublasHandle_t handle;

    CublasValidator() {
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS init failed\n");
            exit(EXIT_FAILURE);
        }
    }

    ~CublasValidator() {
        cublasDestroy(handle);
    }

    // Row-major C = A * B using cuBLAS (column-major internally)
    // A: MxK, B: KxN, C: MxN — all row-major
    void sgemm(int M, int N, int K,
               const float* d_A, const float* d_B, float* d_C) {
        float alpha = 1.0f, beta = 0.0f;
        // Row-major trick: C^T = B^T * A^T
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
};

// Compute max absolute error between two device buffers
inline float max_error(const float* d_test, const float* d_ref, int size) {
    float* h_test = new float[size];
    float* h_ref  = new float[size];
    CUDA_CHECK(cudaMemcpy(h_test, d_test, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref,  d_ref,  size * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        float err = fabsf(h_test[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }

    delete[] h_test;
    delete[] h_ref;
    return max_err;
}

// Validate kernel output against cuBLAS reference
inline bool validate_gemm(const float* d_test, const float* d_ref, int size,
                          float tol = 1e-5f) {
    float err = max_error(d_test, d_ref, size);
    return err < tol;
}

// Fill device buffer with deterministic pseudo-random floats in [-1, 1]
inline void init_random_matrix(float* d_mat, int size, unsigned int seed = 42) {
    float* h_mat = new float[size];
    srand(seed);
    for (int i = 0; i < size; ++i) {
        h_mat[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_mat;
}
