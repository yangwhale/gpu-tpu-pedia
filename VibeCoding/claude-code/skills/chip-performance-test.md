# Chip Performance Test

Automated chip performance testing skill for GPU and TPU. Use when user asks to test machine/chip performance, run GEMM benchmarks, or measure compute capability.

## Trigger Phrases

- "测试这台机器的性能"
- "测一下这个芯片"
- "run performance test"
- "benchmark this machine"
- "测试 GEMM 性能"
- "看看算力多少"
- "MFU 是多少"

## Quick Start

```bash
cd /path/to/gpu-tpu-pedia/tpu/v7_perf/chay_gemm_benchmark_simple

# TPU
python3 main_tpu.py --config config/tpu_simple.json

# GPU
python3 main.py --config config/simple.json
```

---

## Workflow

### Step 1: Hardware Detection

```bash
# Check for TPU (JAX)
python3 -c "import jax; devices = jax.devices('tpu'); print(f'TPU: {len(devices)} devices') if devices else print('No TPU')" 2>/dev/null || echo "No TPU"

# Check for GPU (CUDA)
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU')" 2>/dev/null || echo "No GPU"

# Alternative GPU check
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU"
```

### Step 2: Hardware Specs Reference

| Hardware | BF16/FP16 Peak | FP32 Peak | INT8 Peak | HBM BW |
|----------|----------------|-----------|-----------|--------|
| **TPU v6e (Trillium)** | 918 TFLOPS | 918 TFLOPS* | 1836 TOPS | 1,600 GB/s |
| **TPU v7 (Ironwood)** | ~2000 TFLOPS | ~2000 TFLOPS* | ~4000 TOPS | ~3,200 GB/s |
| **NVIDIA H100 SXM** | 990 TFLOPS | 67 TFLOPS | 1980 TOPS | 3,350 GB/s |
| **NVIDIA H20** | 147 TFLOPS | 40 TFLOPS | 293 TOPS | 4,000 GB/s |
| **NVIDIA A100 80GB** | 312 TFLOPS | 19.5 TFLOPS | 624 TOPS | 2,039 GB/s |
| **NVIDIA B200** | 2250 TFLOPS | 2250 TFLOPS | 4500 TOPS | 8,000 GB/s |

> *TPU float32 uses bf16 compute with fp32 accumulation, achieving same throughput as bf16

### Step 3: Run Benchmark

#### TPU

```bash
cd /path/to/gpu-tpu-pedia/tpu/v7_perf/chay_gemm_benchmark_simple

# Quick test (~25 seconds)
python3 main_tpu.py --config config/tpu_simple.json

# Full test with CSV output (~10 minutes)
python3 main_tpu.py --config config/tpu_full.json --output results.csv

# Complete LLM-style matrix test (~1 hour)
python3 main_tpu.py --config config/tpu_gemm.json --output results_full.csv
```

#### GPU

```bash
cd /path/to/gpu-tpu-pedia/tpu/v7_perf/chay_gemm_benchmark_simple

# Quick test
python3 main.py --config config/simple.json

# Full test
python3 main.py --config config/gemm.json
```

### Step 4: Analyze Results

Key metrics to report:

| Metric | Description |
|--------|-------------|
| **TFLOPS** | Measured compute throughput |
| **MFU** | Model FLOPS Utilization = Measured / Theoretical Peak |
| **Bandwidth** | Memory throughput (GB/s) |

### Step 5: Generate Report

Use this template:

```markdown
# [DEVICE_NAME] Performance Report

## Hardware
- **Device**: [name]
- **Type**: GPU/TPU
- **Peak BF16**: [X] TFLOPS
- **HBM Bandwidth**: [X] GB/s

## Results

| Dtype | Peak TFLOPS | Peak MFU | Best Config |
|-------|-------------|----------|-------------|
| bfloat16 | X | X% | M=X, K=X, N=X |
| float32 | X | X% | M=X, K=X, N=X |
| int8 | X TOPS | X% | M=X, K=X, N=X |

## Observations
1. ...
2. ...

## Recommendations
- ...
```

---

## Lessons Learned (重要经验)

### TPU 性能特性

1. **float32 走 bf16 计算路径**
   - TPU MXU 原生执行 bf16
   - float32 输入会转为 bf16 计算 + fp32 累加
   - 因此 float32 峰值 = bf16 峰值，不是一半

2. **小 batch 效率低**
   - M < 512 时 MFU 通常 < 15%
   - M = 128 时 MFU 仅 3-9%
   - 原因：MXU 256×256，小矩阵填不满

3. **最佳 MFU 配置**
   - TPU v6e bfloat16 最佳：M=4096, K=8192, N=8192 → 75% MFU
   - 大矩阵 (M≥4096) 可能遇到 HBM 带宽瓶颈

4. **JAX 计时注意事项**
   - 必须用 `block_until_ready()` 等待计算完成
   - JAX 异步调度，不等待会只测到 dispatch 时间
   - warmup 必须，首次调用触发 JIT 编译

### GPU vs TPU 对比

| 特性 | GPU (CUDA) | TPU (JAX) |
|------|------------|-----------|
| 框架 | PyTorch | JAX |
| 计时 | CUDA Event | block_until_ready() |
| FP32 效率 | 远低于 FP16 | 接近 BF16 |
| 最佳 dtype | float16/bfloat16 | bfloat16 |

---

## TPU v6e 实测参考数据

测试环境：TPU v6e 8 cores, JAX 0.8.1

| Dtype | 理论峰值 | 最高实测 | 最高 MFU | 平均 MFU |
|-------|----------|----------|----------|----------|
| bfloat16 | 918 TFLOPS | 689 TFLOPS | **75.0%** | 36.2% |
| float32 | 918 TFLOPS | 583 TFLOPS | 63.5% | 28.5% |
| int8 | 1836 TOPS | 1129 TOPS | 61.5% | 24.8% |

---

## Troubleshooting

### TPU Already in Use

```
ABORTED: The TPU is already in use by process with pid XXXXX
```

解决方案：
```bash
# 找到占用进程
ps aux | grep python | grep -v grep

# 或者重启 TPU runtime
sudo systemctl restart tpu-runtime  # 如果有

# 或者等待当前进程完成
```

### TPU Not Detected

```bash
# 检查 TPU 设备
ls /dev/accel*

# 检查 JAX 后端
python3 -c "import jax; print(jax.default_backend())"

# 检查 libtpu
python3 -c "import jax; jax.devices('tpu')"
```

### GPU CUDA Error

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 检查 GPU 内存
nvidia-smi
```

### Out of Memory

- 减少 M 值
- 使用更小的 K.N 对
- 一次只测一种 dtype

---

## Code Structure

```
chay_gemm_benchmark_simple/
├── main.py              # GPU 入口 (PyTorch)
├── main_tpu.py          # TPU 入口 (JAX)
├── backends/
│   ├── tpu/
│   │   └── tpu_backends.py  # TPU 后端实现
│   └── nv_gpu_cublas/       # NVIDIA cuBLAS 扩展
├── hw_spec.py           # 硬件规格定义
└── config/
    ├── tpu_simple.json  # TPU 快速测试
    ├── tpu_full.json    # TPU 中等测试
    └── tpu_gemm.json    # TPU 完整测试
```

---

## Related Skills

- `tpu-trainer`: TPU 模型训练自动化
- `sglang-installer`: SGLang 推理服务器安装

---

*Updated: 2026-02-09*
*Based on TPU v6e benchmark implementation and testing*
