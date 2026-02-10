# main.py
import json
import argparse
import torch # 导入 torch 以便进行环境检测
from utils import parse_dtype
from backends import NvidiaGpuBackend

from hw_spec import DTYPE_OUTPUT_MAPPING


def calculate_performance(m: int, n: int, k: int, input_dtype: torch.dtype, output_dtype: torch.dtype, avg_time_us: float):
    if avg_time_us <= 0:
        return 0.0, 0.0

    avg_time_s = avg_time_us / 1_000_000

    # TFLOPS 计算保持不变
    flops = 2 * m * n * k
    tflops = (flops / 1e12) / avg_time_s

    # 输入A (m,k) + 输入B (k,n) + 输出C (m,n)
    bytes_a = m * k * input_dtype.itemsize
    bytes_b = k * n * input_dtype.itemsize
    bytes_c = m * n * output_dtype.itemsize # output dtype maybe diff from input

    total_data_bytes = bytes_a + bytes_b + bytes_c
    bandwidth_gbps = (total_data_bytes / 1e9) / avg_time_s

    return tflops, bandwidth_gbps

def run_benchmark_from_config(config, backend):
    """
    根据配置格式生成并执行所有测试用例。
    """
    print(f"使用设备: {backend.device_name} (后端: {backend.__class__.__name__})")

    # 获取当前后端名称，用于查询映射
    backend_name = backend.get_device_str()
    if backend_name == "cuda": backend_name = "nvidia"

    for case_generator in config["cases"]:
        m_values = case_generator["M"]
        kn_pairs = case_generator["K.N"]
        dtype_strs = case_generator["dtype"]

        for dtype_str in dtype_strs:
            try:
                input_dtype = parse_dtype(dtype_str)
            except ValueError as e:
                print(f"[Warning] 跳过不支持的数据类型 '{dtype_str}': {e}")
                continue

            # 【新增】从映射中查找对应的输出类型
            try:
                output_dtype = DTYPE_OUTPUT_MAPPING[backend_name][input_dtype]
            except KeyError:
                print(f"[Warning] 在配置中未找到 {backend_name} 后端对输入 {dtype_str} 的输出类型定义，跳过。")
                continue

            print(f"\n测试数据类型: input:{dtype_str} ----> output:{output_dtype}")
            print(f"{'M':>8s} | {'N':>8s} | {'K':>8s} | {'Time (us)':>10s} | {'TFLOPS':>10s} | {'GB/s':>8s}")
            print("-" * 70)

            for k, n in kn_pairs:
                for m in m_values:
                    avg_time_us = backend.run(m, n, k, input_dtype)

                    if avg_time_us > 0:
                        # 【已修改】将输入和输出类型都传给计算函数
                        tflops, bandwidth_gbps = calculate_performance(m, n, k, input_dtype, output_dtype, avg_time_us)
                        print(f"{m:8d} | {n:8d} | {k:8d} | {avg_time_us:10.2f} | {tflops:10.2f} | {bandwidth_gbps:8.2f}")
                    else:
                        print(f"{m:8d} | {n:8d} | {k:8d} | {'Failed':>10s} | {'N/A':>10s} | {'N/A':>8s}")


def main():
    parser = argparse.ArgumentParser(description="GEMM Performance Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/simple.json",
        help="指向测试配置的 JSON 文件路径"
    )
    args = parser.parse_args()

    # 1. 读取配置文件
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{args.config}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{args.config}' 格式不正确。")
        return

    settings = config.get("benchmark_settings", {"warmup_iter": 10, "prof_iter": 100})
    warmup_iter = settings.get("warmup_iter", 10)
    prof_iter = settings.get("prof_iter", 100)

    # 2. 【新增】自动检测并初始化后端
    backend = None
    if torch.cuda.is_available():
        print("检测到 NVIDIA CUDA 环境，将使用 CUDA 后端。")
        backend = NvidiaGpuBackend(warmup_iter, prof_iter)
    elif hasattr(torch, 'mlu') and torch.mlu.is_available():
        print("检测到 Cambricon MLU 环境，将使用 MLU 后端。")
        backend = MluBackend(warmup_iter, prof_iter)
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        print("检测到 Ascend NPU 环境，将使用 NPU 后端。")
        backend = NpuBackend(warmup_iter, prof_iter)


    if backend is None:
        # 更新错误信息
        print("错误: 未检测到任何支持的硬件后端 (NVIDIA CUDA, Cambricon MLU, or Ascend NPU)。")
        print("请确保已正确安装对应的 PyTorch 版本和驱动。")
        return

    # 3. 运行测试
    try:
        run_benchmark_from_config(config, backend)
    except Exception as e:
        print(f"\n在 benchmark 运行期间发生未处理的错误: {e}")


if __name__ == "__main__":
    main()
