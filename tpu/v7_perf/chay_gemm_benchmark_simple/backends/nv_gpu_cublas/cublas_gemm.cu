#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 宏，用于将错误码转换为异常抛出
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS Error with status " + std::to_string(status)); \
    } \
} while (0)

// 核心的 benchmark 函数
// 参数: m, n, k, dtype_str, warmup_iter, prof_iter
// 返回值: 平均执行时间 (us)
double benchmark_gemm_ex(int m, int n, int k, const std::string& dtype_str, int warmup_iter, int prof_iter) {
    
    cudaDataType a_type, b_type, c_type;
    cublasComputeType_t compute_type;
    size_t matrix_type_size, result_type_size;

    void *d_a, *d_b, *d_c;
    
    // 根据传入的字符串决定数据类型
    if (dtype_str == "float16") {
        a_type = b_type = c_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_32F;
        matrix_type_size = sizeof(half);
        result_type_size = sizeof(half);
    } else if (dtype_str == "int8") {
        a_type = b_type = CUDA_R_8I;
        c_type = CUDA_R_32I; // int8 gemm 输出是 int32
        compute_type = CUBLAS_COMPUTE_32I;
        matrix_type_size = sizeof(int8_t);
        result_type_size = sizeof(int32_t);
    } else {
        throw std::invalid_argument("Unsupported dtype: " + dtype_str);
    }

    // 在设备端分配内存
    CHECK_CUDA(cudaMalloc(&d_a, m * k * matrix_type_size));
    CHECK_CUDA(cudaMalloc(&d_b, k * n * matrix_type_size));
    CHECK_CUDA(cudaMalloc(&d_c, m * n * result_type_size));

    // alpha 和 beta 的设置
    float alpha_fp = 1.0f, beta_fp = 0.0f;
    int32_t alpha_i32 = 1, beta_i32 = 0;
    void *alpha_ptr, *beta_ptr;

    if (dtype_str == "float16") {
        alpha_ptr = &alpha_fp;
        beta_ptr = &beta_fp;
    } else {
        alpha_ptr = &alpha_i32;
        beta_ptr = &beta_i32;
    }
    
    // 创建 cuBLAS 句柄和计时器
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));

    // 注意：这里我们省略了数据初始化的部分，因为对于性能测试，
    // 只要内存被访问即可，内容不影响时间。
    
    // 预热
    for (int i = 0; i < warmup_iter; ++i) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                   alpha_ptr, d_a, a_type, k, d_b, b_type, k,
                                   beta_ptr, d_c, c_type, m, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 评测
    CHECK_CUDA(cudaEventRecord(start_event, 0));
    for (int i = 0; i < prof_iter; ++i) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                   alpha_ptr, d_a, a_type, k, d_b, b_type, k,
                                   beta_ptr, d_c, c_type, m, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(stop_event, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    
    float time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start_event, stop_event));

    // 清理资源
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUBLAS(cublasDestroy(handle));

    // 返回平均时间 (微秒 us)
    return static_cast<double>(time_ms * 1000.0 / prof_iter);
}
