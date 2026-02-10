import torch

# ===================================================================
# 硬件与数据类型配置中心
# 所有与特定硬件、数据类型相关的"知识"都应定义在此处。
# ===================================================================

# 1. 硬件理论峰值性能
# 带宽单位: GB/s, 算力单位: TFLOPS/TOPS
DEVICE_SPECS = {
    # ===== NVIDIA GPUs =====
    # Note: device name corresponds to torch.cuda.get_device_name(0)
    "NVIDIA H20": {
        "bandwidth": 4000,  # 4 TB/s
        "float32": 40,      # CUDA core
        "float16": 147,
        "bfloat16": 147,
        "int8": 293
    },

    # ===== Google TPUs =====
    # Note: device name corresponds to get_device_name() in TPU backend

    # TPU v6e (Trillium) - GA in late 2024
    # Source: https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus
    # Note: float32 GEMM uses bf16 compute with fp32 accumulation on TPU MXU
    "Google TPU v6e (Trillium)": {
        "bandwidth": 1600,   # 1.6 TB/s HBM
        "float32": 918,      # Uses bf16 compute path (fp32 accumulation)
        "float16": 918,      # Peak MXU throughput
        "bfloat16": 918,     # Native TPU precision
        "int8": 1836         # 2x bf16 for INT8
    },

    # TPU v7 (Ironwood) - Placeholder, specs TBD
    # Will be updated when v7 becomes available
    "Google TPU v7 (Ironwood)": {
        "bandwidth": 3200,   # Estimated, TBD
        "float32": 1000,     # Estimated, TBD
        "float16": 2000,     # Estimated, TBD
        "bfloat16": 2000,    # Estimated, TBD
        "int8": 4000         # Estimated, TBD
    },
}

# 2. 不同后端 GEMM 的输入 -> 输出数据类型映射
DTYPE_OUTPUT_MAPPING = {
    "nvidia": {
        # 对于 NVIDIA 后端:
        torch.float32: torch.float32,
        torch.float16: torch.float16,
        torch.bfloat16: torch.bfloat16,
        torch.int8: torch.int32,  # cublasGemmEx for int8 in->int32 out
    },
    # TPU 后端使用 JAX，dtype 映射在 tpu_backends.py 中处理
    # 这里保留字符串映射供 main_tpu.py 使用
    "tpu": {
        torch.float32: torch.float32,
        torch.float16: torch.float16,
        torch.bfloat16: torch.bfloat16,
        torch.int8: torch.int32,  # int8 GEMM accumulates to int32
    },
}

# 3. 字符串到 torch.dtype 对象的映射
# 使其集中，方便 utils.py 和 analyze.py 共同使用
DTYPE_FROM_STR = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}
# 安全地添加 tfloat32
try:
    DTYPE_FROM_STR["tfloat32"] = torch.tfloat32
except AttributeError:
    pass
