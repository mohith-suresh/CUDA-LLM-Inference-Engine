cmake_minimum_required(VERSION 3.18)
project(SLICK LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find cuBLAS
find_library(CUBLAS_LIB cublas
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
find_path(CUBLAS_INCLUDE cublas_v2.h
    HINTS /usr/include /usr/local/cuda/include)

if(NOT CUBLAS_LIB)
    message(FATAL_ERROR "cuBLAS library not found")
endif()

# Find cuDNN
find_library(CUDNN_LIB cudnn
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
find_path(CUDNN_INCLUDE cudnn.h
    HINTS /usr/include /usr/local/cuda/include)

if(NOT CUDNN_LIB)
    message(FATAL_ERROR "cuDNN library not found")
endif()

message(STATUS "cuBLAS library: ${CUBLAS_LIB}")
message(STATUS "cuBLAS include: ${CUBLAS_INCLUDE}")
message(STATUS "cuDNN library: ${CUDNN_LIB}")
message(STATUS "cuDNN include: ${CUDNN_INCLUDE}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
    ${CUBLAS_INCLUDE}
)

# --- GEMM Kernels (Week 1) ---
add_library(gemm_kernels
    kernels/gemm/01_naive.cu
    kernels/gemm/02_coalesced.cu
    kernels/gemm/03_shared_tiling.cu
    kernels/gemm/04_reg_tiling_1d.cu
    kernels/gemm/05_reg_tiling_2d.cu
    kernels/gemm/06_vectorized.cu
    kernels/gemm/07_double_buffered.cu
)
target_include_directories(gemm_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)

# GEMM Benchmark
add_executable(gemm_bench benchmarks/gemm_bench.cu)
target_link_libraries(gemm_bench gemm_kernels ${CUBLAS_LIB})

# --- Softmax Kernels (Week 3) ---
add_library(softmax_kernels
    kernels/softmax/08_fused_online.cu
    kernels/softmax/09_warp_reduce.cu
)
target_include_directories(softmax_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)

# Softmax Benchmark
add_executable(softmax_bench benchmarks/softmax_bench.cu)
target_link_libraries(softmax_bench softmax_kernels ${CUDNN_LIB})

# --- Attention Kernels (Week 4) ---
add_library(attention_kernels
    kernels/flash_attention/10_flash_attn_v2.cu
)
target_include_directories(attention_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)

# Attention Benchmark
add_executable(attention_bench benchmarks/attention_bench.cu)
target_link_libraries(attention_bench attention_kernels softmax_kernels ${CUBLAS_LIB})

# --- Tests (Google Test) ---
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm GTest::gtest_main gemm_kernels ${CUBLAS_LIB})
add_test(NAME GemmTests COMMAND test_gemm)

add_executable(test_softmax tests/test_softmax.cu)
target_link_libraries(test_softmax GTest::gtest_main softmax_kernels)
add_test(NAME SoftmaxTests COMMAND test_softmax)

add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention GTest::gtest_main attention_kernels)
add_test(NAME AttentionTests COMMAND test_attention)
