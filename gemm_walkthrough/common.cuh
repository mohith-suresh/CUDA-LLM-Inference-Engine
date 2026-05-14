// common.cuh — shared helpers for the GEMM walkthrough
// Timing, cuBLAS reference, random init, and error checking.
//
// Convention used by EVERY kernel in this folder:
//   Row-major matrices.  GEMM:  C = alpha * (A @ B) + beta * C
//   A is [M,K], B is [K,N], C is [M,N].  C[row,col] lives at C[row*N + col].
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---- error checking ---------------------------------------------------------
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err = (expr);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---- host-side random fill --------------------------------------------------
inline void fill_random(float* host, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        host[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;   // in [-1, 1]
}

// Every kernel and the cuBLAS reference share this signature.
typedef void (*GemmLauncher)(int, int, int, float,
                             const float*, const float*, float, float*);

// ---- cuBLAS reference: row-major C = alpha*(A@B) + beta*C --------------------
// cuBLAS is column-major, so we use the standard transpose identity:
//   (A @ B) stored row-major == (B^T @ A^T) computed column-major.
// Passing our row-major B and A straight into a column-major sgemm with the
// arguments swapped produces exactly the row-major GEMM we want.
struct CublasRef {
    cublasHandle_t handle;
    CublasRef()  { cublasCreate(&handle); }
    ~CublasRef() { cublasDestroy(handle); }

    void gemm(int M, int N, int K, float alpha,
              const float* A, const float* B, float beta, float* C) const {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,                 // dims of the column-major problem
                    &alpha,
                    B, N,                    // B is (N x K) col-major = row-major B[K,N]
                    A, K,                    // A is (K x M) col-major = row-major A[M,K]
                    &beta,
                    C, N);                   // C is (N x M) col-major = row-major C[M,N]
    }
};

// ---- max relative error between device buffers ------------------------------
// Error relative to the matrix's PEAK magnitude (standard GEMM check).
// Per-element relative error is meaningless where C ~ 0 (cancellation), so we
// normalize the largest absolute deviation by the largest |reference| value.
inline float max_rel_error(const float* d_test, const float* d_ref, int n) {
    float* a = (float*)malloc(n * sizeof(float));
    float* b = (float*)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(a, d_test, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b, d_ref,  n * sizeof(float), cudaMemcpyDeviceToHost));
    float worst_abs = 0.0f, max_ref = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > worst_abs) worst_abs = d;
        float m = fabsf(b[i]);
        if (m > max_ref) max_ref = m;
    }
    free(a); free(b);
    return worst_abs / (max_ref + 1e-30f);
}

// ---- timing: warmup + averaged iterations -----------------------------------
inline float time_kernel(GemmLauncher launch, int M, int N, int K,
                         float alpha, const float* A, const float* B,
                         float beta, float* C, int warmup = 3, int iters = 20) {
    for (int i = 0; i < warmup; ++i) launch(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) launch(M, N, K, alpha, A, B, beta, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / iters;
}

inline float gflops(int M, int N, int K, float ms) {
    return (2.0 * M * N * K) / (ms * 1e6);   // 2*M*N*K flops / (ms * 1e6) = GFLOP/s
}
