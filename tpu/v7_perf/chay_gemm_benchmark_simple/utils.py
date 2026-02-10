# utils.py
import torch

# --- 动态构建数据类型映射 ---
# 1. 定义一个包含基础数据类型的字典
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int8": torch.int8,
}

# 2. 安全地尝试添加 bfloat16，以适应不同 PyTorch 版本
try:
    # 如果当前 PyTorch 版本支持 bfloat16，则将其添加到映射中
    DTYPE_MAP["bfloat16"] = torch.bfloat16
except AttributeError:
    # 如果不支持，则静默地跳过，这样程序不会崩溃
    pass

# 2. 安全地尝试添加 fp8，以适应不同 PyTorch 版本
try:
    DTYPE_MAP["float8_e4m3fn"] = torch.float8_e4m3fn
except AttributeError:
    # 如果不支持，则静默地跳过，这样程序不会崩溃
    pass



def parse_dtype(dtype_str: str) -> torch.dtype:
    """将字符串转换为 torch.dtype 对象"""
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"不支持的数据类型: '{dtype_str}'. 当前环境支持: {list(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype_str]

def create_input_tensors(m, n, k, dtype, device):
    """根据数据类型创建合适的输入张量"""
    if dtype.is_floating_point:
        # 这个分支会统一处理 float32, float16, 以及 bfloat16 (如果环境中支持)
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
    elif dtype == torch.int8:
        # int8 的处理逻辑保持不变
        a = torch.randint(-128, 127, (m, k), device=device, dtype=dtype)
        b = torch.randint(-128, 127, (k, n), device=device, dtype=dtype)
    else:
        # 为其他未来可能添加的整数类型提供一个默认行为
        a = torch.ones(m, k, device=device, dtype=dtype)
        b = torch.ones(k, n, device=device, dtype=dtype)
    return a, b
