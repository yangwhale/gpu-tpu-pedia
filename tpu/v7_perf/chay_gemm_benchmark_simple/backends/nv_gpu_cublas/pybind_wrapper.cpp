#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 用于 C++ 标准库容器的自动转换
#include <string>

namespace py = pybind11;

// 声明我们将要绑定的 C++ 函数
// 这个函数定义在 cublas_gemm.cu 中
double benchmark_gemm_ex(int m, int n, int k, const std::string& dtype_str, int warmup_iter, int prof_iter);

// 创建 Python 模块
// PYBIND11_MODULE 的第一个参数 (cublas_backend) 是最终 import 的模块名
// 第二个参数 (m) 是 py::module_ 对象的变量名
PYBIND11_MODULE(cublas_backend, m) {
    m.doc() = "A Pybind11 wrapper for cuBLAS GEMM benchmark"; // 模块的文档字符串

    // 暴露 benchmark_gemm_ex 函数给 Python
    m.def("benchmark", &benchmark_gemm_ex, "Runs a GEMM benchmark using cublasGemmEx",
          py::arg("m"),
          py::arg("n"),
          py::arg("k"),
          py::arg("dtype"),
          py::arg("warmup_iter") = 10,
          py::arg("prof_iter") = 100);
}
