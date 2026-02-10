cmake_minimum_required(VERSION 3.18)
project(CublasBenchmark LANGUAGES CXX CUDA)

# 找到 CUDA, Pybind11 和 Python
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

# 使用 pybind11_add_module 创建 Python 扩展模块
# 它会自动处理所有 Python 相关的编译和链接标志
pybind11_add_module(cublas_backend
    pybind_wrapper.cpp
    cublas_gemm.cu
)

# 链接 CUDA 库 (cublas)
target_link_libraries(cublas_backend PRIVATE 
    ${CUDA_CUBLAS_LIBRARIES}
)

# 指定 CUDA 架构，例如 Hopper
target_compile_options(cublas_backend PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_90,code=sm_90
    >
)
