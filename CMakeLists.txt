cmake_minimum_required(VERSION 3.18)
project(SLICK LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Fallback for older CUDA toolkits
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
endif()

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
