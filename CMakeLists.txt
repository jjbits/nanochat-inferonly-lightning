cmake_minimum_required(VERSION 3.18)
project(nanochat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

# CUTLASS headers only (skip library build)
include(FetchContent)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v3.4.1
)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
endif()

# Kernel library
add_library(kernels STATIC
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/attention.cu
    kernels/flash_attention.cu
    kernels/ops.cu
    kernels/gemm.cu
)
target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(kernels SYSTEM PUBLIC ${cutlass_SOURCE_DIR}/include)
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

# Main executable
add_executable(nanochat
    src/model.cpp
    src/engine.cpp
    src/main.cpp
)
target_include_directories(nanochat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(nanochat SYSTEM PRIVATE ${cutlass_SOURCE_DIR}/include)
target_link_libraries(nanochat PRIVATE kernels CUDA::cudart)
