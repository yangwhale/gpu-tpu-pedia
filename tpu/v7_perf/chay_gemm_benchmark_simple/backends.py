# backends.py
import torch
import abc
import sys
import os
from utils import create_input_tensors

# --- 动态添加并导入 C++ 扩展模块 (逻辑保留) ---
# 这个模块现在只会被新的 NvidiaGpuBackend 在测试 int8 时使用
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设您的目录结构是 .../chay_gemm_benchmark/backends.py 和 .../chay_gemm_benchmark/backends/nv_gpu_cublas/
# 我们需要构建正确的相对路径
# 修正路径构建，以适应您的项目结构
cublas_build_path = os.path.join(current_dir, 'backends', 'nv_gpu_cublas', 'build')

if cublas_build_path not in sys.path:
    sys.path.append(cublas_build_path)

if hasattr(torch, 'cuda') and torch.cuda.is_available():
    try:
        import cublas_backend
        CUBLAS_BACKEND_AVAILABLE = True
    except ImportError:
        cublas_backend = None
        CUBLAS_BACKEND_AVAILABLE = False
        print("[Warning] 'cublas_backend.so' 模块未找到或导入失败。int8 测试将不可用。")

if hasattr(torch, 'mlu') and torch.mlu.is_available():
    try:
        import torch_mlu
        import torch_mlu_ops as tmo
        MLU_OPS_AVAILABLE = True
    except ImportError:
        tmo = None
        MLU_OPS_AVAILABLE = False
        print("[Warning] 'torch_mlu_ops' 模块未安装。MluBackend 将不可用。")

# ... BenchmarkBackend 基类和 MluBackend 类保持不变 ...
# ... (为简洁起见，这里省略了它们的代码，您无需改动) ...
class BenchmarkBackend(abc.ABC):
    def __init__(self, warmup_iter, prof_iter):
        self.device_name = self.get_device_name()
        self.device = self.get_device_str()
        self.warmup_iter = warmup_iter
        self.prof_iter = prof_iter
    @abc.abstractmethod
    def get_device_name(self) -> str: pass
    @abc.abstractmethod
    def get_device_str(self) -> str: pass
    @abc.abstractmethod
    def synchronize(self): pass
    @abc.abstractmethod
    def run(self, m, n, k, dtype) -> float: pass

class NvidiaGpuBackend(BenchmarkBackend):
    """
    统一的 NVIDIA GPU 后端。
    内部会根据数据类型动态选择执行路径：
    - int8: 调用 C++/cuBLAS 扩展
    - bfloat16/float16/float32: 调用 PyTorch 原生接口
    """
    def get_device_name(self) -> str:
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA GPU (CUDA) not available.")
        return torch.cuda.get_device_name(0)

    def get_device_str(self) -> str:
        return "cuda"

    def synchronize(self):
        torch.cuda.synchronize()

    def run(self, m, n, k, dtype) -> float:
        # --- 动态分发逻辑 ---
        if dtype == torch.int8:
            # --- INT8 路径: 调用 C++ 扩展 ---
            if not CUBLAS_BACKEND_AVAILABLE:
                print(f"  > [NVIDIA Info] 跳过 int8 测试，因为 cublas_backend.so 模块不可用。")
                return -1.0

            try:
                avg_time_us = cublas_backend.benchmark(
                    m=m, n=n, k=k, dtype="int8",
                    warmup_iter=self.warmup_iter,
                    prof_iter=self.prof_iter
                )
                return avg_time_us
            except Exception as e:
                print(f"  > [CublasEx Error] m={m},n={n},k={k},dtype=int8: {e}")
                return -1.0

        elif dtype.is_floating_point:
            # --- 浮点类型路径 (bfloat16, float16等): 调用 PyTorch 原生接口 ---
            try:
                a, b = create_input_tensors(m, n, k, dtype, self.device)

                # 预热
                for _ in range(self.warmup_iter):
                    _ = torch.matmul(a, b)
                self.synchronize()

                # 使用 torch.cuda.Event 进行评测
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(self.prof_iter):
                    torch.matmul(a, b)
                end_event.record()

                self.synchronize()

                total_time_ms = start_event.elapsed_time(end_event)
                return (total_time_ms * 1000) / self.prof_iter
            except Exception as e:
                # 针对 bfloat16 等类型提供具体的错误信息
                if 'not supported' in str(e) and 'bfloat16' in str(e):
                    print(f"  > [NVIDIA Info] m={m},n={n},k={k}: 当前设备或CUDA版本不支持 bfloat16。")
                else:
                    print(f"  > [NVIDIA Error] m={m},n={n},k={k},dtype={dtype}: {e}")
                return -1.0
        else:
            # 其他不支持的类型
            print(f"  > [NVIDIA Info] 不支持的数据类型: {dtype}")
            return -1.0


