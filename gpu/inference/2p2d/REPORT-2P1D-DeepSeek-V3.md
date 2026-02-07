# SGLang PD Disaggregation 测试报告：DeepSeek-V3 2P1D vs 1P1D

**日期**: 2026-02-07
**配置**: 2 Prefill + 1 Decode (2P1D) vs 1 Prefill + 1 Decode (1P1D)
**模型**: DeepSeek-V3 (671B MoE, FP8)
**硬件**: 3x GCP a4-highgpu-8g (NVIDIA B200 x8, 180GB HBM each)
**软件**: SGLang 0.5.8, NIXL transfer backend, CUDA 12.9

---

## 架构概览

### 2P1D 架构（本次测试）

```
           ┌──────────────┐
           │   Router     │
           │  b7:8000     │
           └──────┬───────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Prefill1 │ │ Prefill2 │ │  Decode  │
│ b7:30000 │ │ b9:30000 │ │ b8:30000 │
│ TP=8     │ │ TP=8     │ │ TP=8     │
│ 8x B200  │ │ 8x B200  │ │ 8x B200  │
└──────────┘ └──────────┘ └──────────┘
     NIXL KV Cache Transfer
```

- **Prefill 节点 1 (b7)**: 10.8.0.12，执行 prefill，通过 NIXL 传输 KV cache
- **Prefill 节点 2 (b9)**: 10.8.0.57，第二个 prefill 实例，分担 prefill 负载
- **Decode 节点 (b8)**: 10.8.0.17，接收 KV cache，执行自回归 decode
- **Router**: `sglang_router` 在 b7:8000，round-robin 分配请求到两个 prefill

### 1P1D 架构（基线对照）

```
           ┌──────────────┐
           │   Router     │
           │  b7:8000     │
           └──────┬───────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐   ┌──────────────┐
│  Prefill     │   │  Decode      │
│  b7:30000    │──▶│  b8:30000    │
│  TP=8        │   │  TP=8        │
│  8x B200     │   │  8x B200     │
└──────────────┘   └──────────────┘
```

---

## 性能数据

### 2P1D Evalscope 压测（3000-4000 input, 400-600 output, random dataset）

| 并发 | Output (tok/s) | Total (tok/s) | Avg TTFT (s) | P99 TTFT (s) | Avg TPOT (s) | Avg Latency (s) | Req/s | Failed |
|------|---------------|---------------|-------------|-------------|-------------|-----------------|-------|--------|
| 4 | 39.4 | - | 0.50 | 0.64 | 0.101 | 60.8 | 0.07 | 0 |
| 8 | 55.2 | - | 18.27* | 106.0* | 0.101 | 78.7 | 0.09 | 0 |
| 16 | 156.5 | - | 0.68 | 1.47 | 0.101 | 61.2 | 0.26 | 0 |
| 32 | 298.1 | - | 0.94 | 2.20 | 0.106 | 64.2 | 0.50 | 0 |
| 64 | 593.7 | - | 1.20 | 3.86 | 0.105 | 64.4 | 0.99 | 0 |
| 96 | 881.1 | - | 1.47 | 5.16 | 0.106 | 64.8 | 1.47 | 0 |
| **128** | **1,159.0** | - | **1.73** | **6.23** | **0.106** | 65.5 | 1.93 | 0 |
| 192 | 1,667.2 | - | 2.09 | 10.25 | 0.110 | 68.0 | 2.78 | 0 |
| 256 | 1,775.6 | - | 6.80 | 61.15 | 0.112 | 74.0 | 2.96 | 0 |

> *并发 8 的异常 TTFT 由 b9 节点预热导致（DeepGEMM JIT 编译和 FlashInfer autotune 首次运行），不影响后续稳态性能。

### 1P1D Evalscope 压测（同参数基线对照）

| 并发 | Output (tok/s) | Avg TTFT (s) | P99 TTFT (s) | Avg TPOT (s) | Avg Latency (s) | Req/s | Failed |
|------|---------------|-------------|-------------|-------------|-----------------|-------|--------|
| 4 | 39.3 | 0.62 | 0.79 | 0.101 | 61.1 | 0.07 | 0 |
| 8 | 78.3 | 0.78 | 1.06 | 0.101 | 61.2 | 0.13 | 0 |
| 16 | 156.3 | 0.89 | 1.86 | 0.101 | 61.2 | 0.26 | 0 |
| 32 | 300.9 | 1.11 | 3.19 | 0.104 | 63.4 | 0.50 | 0 |
| 64 | 574.9 | 1.61 | 6.42 | 0.108 | 66.1 | 0.96 | 0 |
| 96 | 872.3 | 2.01 | 9.05 | 0.105 | 65.0 | 1.45 | 0 |
| **128** | **1,118.8** | **2.40** | **12.07** | **0.108** | 67.2 | 1.86 | 0 |
| 192 | 1,639.6 | 3.19 | 17.87 | 0.109 | 68.2 | 2.73 | 0 |
| 256 | 1,798.1 | 9.02 | 68.40 | 0.108 | 73.9 | 3.00 | 0 |

---

## 2P1D vs 1P1D 关键对比

### TTFT 改善（Prefill 性能提升）

| 并发 | 1P1D TTFT | 2P1D TTFT | 改善幅度 |
|------|-----------|-----------|---------|
| 32 | 1.11s | 0.94s | **-15%** |
| 64 | 1.61s | 1.20s | **-25%** |
| 96 | 2.01s | 1.47s | **-27%** |
| 128 | 2.40s | 1.73s | **-28%** |
| 192 | 3.19s | 2.09s | **-34%** |
| 256 | 9.02s | 6.80s | **-25%** |

### P99 TTFT 改善（尾部延迟）

| 并发 | 1P1D P99 TTFT | 2P1D P99 TTFT | 改善幅度 |
|------|--------------|--------------|---------|
| 64 | 6.42s | 3.86s | **-40%** |
| 128 | 12.07s | 6.23s | **-48%** |
| 192 | 17.87s | 10.25s | **-43%** |
| 256 | 68.40s | 61.15s | **-11%** |

### Output Throughput 对比

| 并发 | 1P1D (tok/s) | 2P1D (tok/s) | 变化 |
|------|-------------|-------------|------|
| 64 | 574.9 | 593.7 | +3.3% |
| 128 | 1,118.8 | 1,159.0 | +3.6% |
| 192 | 1,639.6 | 1,667.2 | +1.7% |
| 256 | 1,798.1 | 1,775.6 | -1.3% |

---

## 瓶颈分析

### 1P1D 的瓶颈确认

1. **Prefill 是瓶颈**：添加第二个 Prefill 节点后，128 并发 TTFT 从 2.40s 降至 1.73s（-28%），P99 TTFT 降低 48%
2. **Decode 仍有余量**：单 Decode 节点在 256 并发时 TPOT 仅 0.112s（vs 低并发 0.101s），退化仅 11%

### 2P1D 的新瓶颈

1. **Decode 开始成为瓶颈**：256 并发时 output throughput 不再增长（1,775 vs 1,798），TPOT 从 0.101 涨到 0.112
2. **Router 调度开销**：并发 8 的异常可能与 router 在两个 prefill 间调度有关
3. **预测 2P2D 性能**：增加第二个 Decode 节点后，预计 256 并发 output throughput 可达 ~2,500+ tok/s

### 成本效率分析

| 配置 | 节点数 | GPU 数 | 128 并发 Output (tok/s) | 每 GPU 效率 (tok/s/GPU) |
|------|--------|--------|------------------------|----------------------|
| 1P1D | 2 | 16 | 1,119 | 69.9 |
| 2P1D | 3 | 24 | 1,159 | 48.3 |

> 2P1D 的总 throughput 提升有限（+3.6%），但 TTFT 改善显著（-28%）。额外的 Prefill 节点主要改善的是用户体验（首 token 延迟），而非总吞吐。

---

## 环境配置

### Prefill 节点 1 (b7 - 10.8.0.12, vlln)
- GPU: 8x NVIDIA B200, Driver 580.126.09
- CUDA: 12.9, PyTorch 2.9.1+cu129
- SGLang: 0.5.8
- TP=8, mem_fraction_static=0.82
- KV Cache: ~985K tokens

### Prefill 节点 2 (b9 - 10.8.0.57, m79p)
- 同 Prefill 节点 1 硬件配置
- SGLang: 0.5.8（从 0.5.6.post2 升级）
- CUDA toolkit 单独安装（DeepGEMM JIT 需要 nvcc）

### Decode 节点 (b8 - 10.8.0.17, 54w7)
- 同 Prefill 节点硬件配置
- max_running_requests=512
- KV Cache 传输: NIXL (DMA-BUF)

### MIG 信息
- MIG: `chrisya-b200-spot-mig-ase1`
- Zone: `asia-southeast1-b`
- Project: `gpu-launchpad-playground`
- Machine Type: `a4-highgpu-8g` (Spot)
- Target Size: 5, Running: 3, Creating: 2 (STOCKOUT)
- Spot B200 Quota: `PREEMPTIBLE_NVIDIA_B200_GPUS` limit 64 in asia-southeast1

---

## 启动命令

### Prefill 节点通用配置
```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --download-dir /lssd/huggingface/hub \
    --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --tp-size 8 --disable-radix-cache --disable-cuda-graph \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.82 \
    --host 0.0.0.0 --port 30000 \
    --watchdog-timeout 1000000 --decode-log-interval 1
```

### Decode 节点
```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --download-dir /lssd/huggingface/hub \
    --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --tp-size 8 --disable-radix-cache --disable-cuda-graph \
    --chunked-prefill-size 8192 \
    --max-running-requests 512 \
    --mem-fraction-static 0.82 \
    --host 0.0.0.0 --port 30000 \
    --watchdog-timeout 1000000 --decode-log-interval 1
```

### Router (2P1D)
```bash
python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://10.8.0.12:30000 8998 \
    --prefill http://10.8.0.57:30000 \
    --decode http://10.8.0.17:30000 \
    --host 0.0.0.0 --port 8000
```

### Evalscope 压测
```bash
evalscope perf \
  --parallel 4 8 16 32 64 96 128 192 256 \
  --number 16 32 64 128 256 384 512 768 1024 \
  --model deepseek-ai/DeepSeek-V3 \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 600 --min-tokens 400 \
  --min-prompt-length 3000 --max-prompt-length 4000 \
  --tokenizer-path deepseek-ai/DeepSeek-V3 \
  --extra-args '{"ignore_eos": true}'
```

---

## 结论与后续计划

### 结论

1. **2P1D 显著降低 TTFT**：128 并发下 TTFT 从 2.40s 降至 1.73s（-28%），P99 TTFT 从 12.07s 降至 6.23s（-48%）
2. **Output throughput 提升有限**：128 并发下仅从 1,119 涨到 1,159 tok/s（+3.6%），因为 decode 节点是输出速率的瓶颈
3. **Decode 在高并发开始饱和**：256 并发时 TPOT 从 0.101s 涨到 0.112s（+11%），output throughput 增长放缓
4. **100% 成功率**：所有测试 0 失败

### 后续计划

1. **2P2D 扩展**：等待 B200 Spot 容量恢复，增加第二个 Decode 节点，预计 output throughput 可达 2,500+ tok/s
2. **跨节点 TP=16**：当有 4+ 节点时，测试跨节点 TP=16 + DeepEP 的 2P2D 架构（需要 NVSHMEM IBGDA）
3. **CUDA Graph**：启用 CUDA Graph 提升 decode 吞吐
4. **DeepSeek-V3-0324**：测试更新的模型版本
