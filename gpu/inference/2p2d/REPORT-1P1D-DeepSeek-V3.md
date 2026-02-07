# SGLang PD Disaggregation 测试报告：DeepSeek-V3 on B200

**日期**: 2026-02-06
**配置**: 1 Prefill + 1 Decode (1P1D)
**模型**: DeepSeek-V3 (671B MoE, FP8)
**硬件**: 2x GCP a4-highgpu-8g (NVIDIA B200 x8, 180GB HBM each)
**软件**: SGLang 0.5.8, NIXL transfer backend, CUDA 12.9

---

## 架构概览

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
     NIXL KV Cache Transfer
```

- **Prefill 节点 (b7)**: 接收用户请求，执行 attention prefill，通过 NIXL 传输 KV cache
- **Decode 节点 (b8)**: 接收 KV cache，执行自回归 decode 生成
- **Router**: `sglang_router` 路由请求到 prefill，decode 结果返回客户端

---

## 关键发现

### 1. DeepEP 在单节点场景不适用

**问题**: 初始配置使用 DeepEP (NVSHMEM IBGDA) 进行 MoE expert all-to-all 通信，导致 `CUDA error: invalid argument`。

**根因**: DeepEP 设计用于跨节点 RDMA 通信。单节点 8 GPU 使用 NVLink，不需要 DeepEP。`--dp-size 8 --tp-size 8` 意味着每个 EP group 只有 1 个 GPU (8/8=1)，无法进行 expert parallelism。

**修复**: 移除所有 DeepEP/NVSHMEM 环境变量和 `--moe-a2a-backend deepep` 参数，使用 SGLang 默认的 NVLink all-to-all。

### 2. NIXL 替代 Mooncake 作为 KV Cache 传输后端

使用 `--disaggregation-transfer-backend nixl`，NIXL 通过 DMA-BUF 进行 GPU 间内存传输，效率高于 Mooncake。

---

## 性能数据

### Prefill Input Throughput（4624 input tokens, 1 output token）

| 并发 | P50 TTFT | P95 TTFT | 聚合 Input Throughput |
|------|----------|----------|----------------------|
| 1 | 261 ms | 264 ms | 17,695 tok/s |
| 2 | 399 ms | 485 ms | 23,280 tok/s |
| 3 | 508 ms | 607 ms | 25,492 tok/s |

> 在 TTFT P50 < 500ms 约束下，最大并发为 **2**，对应 input throughput **23,280 tok/s**。

### Evalscope 压测（3000-4000 input, 400-600 output, random dataset）

| 并发 | Output (tok/s) | Total (tok/s) | TTFT (s) | TPOT (s) | Avg Latency (s) | Req/s | Failed |
|------|---------------|---------------|----------|----------|-----------------|-------|--------|
| 4 | 38.0 | 256.9 | 0.72 | 0.104 | 63.2 | 0.063 | 0 |
| 8 | 74.6 | 505.8 | 0.85 | 0.106 | 64.3 | 0.124 | 0 |
| 16 | 150.5 | 1,016.4 | 0.91 | 0.105 | 63.6 | 0.251 | 0 |
| 32 | 292.2 | 1,966.6 | 1.15 | 0.107 | 65.3 | 0.487 | 0 |
| 64 | 576.7 | 3,931.2 | 1.61 | 0.107 | 65.9 | 0.961 | 0 |
| 96 | 844.3 | 5,797.5 | 2.04 | 0.109 | 67.2 | 1.407 | 0 |
| **128** | **1,122.5** | **7,690.6** | **2.44** | **0.108** | 67.1 | 1.871 | 0 |

> 全部请求 0 失败率。压测工具: evalscope perf。

### Output Throughput 线性扩展分析

| 并发比 | Output 扩展比 | 理论线性 |
|--------|-------------|----------|
| 4→8 | 1.96x | 2.0x |
| 8→16 | 2.02x | 2.0x |
| 16→32 | 1.94x | 2.0x |
| 32→64 | 1.97x | 2.0x |
| 64→128 | 1.95x | 2.0x |

Output throughput 从 4 并发到 128 并发几乎完美线性扩展（每翻倍约 1.95-2.02x），128 并发时 TPOT 仅从 0.104s 微涨到 0.108s，说明 **decode 节点远未饱和**。

### 瓶颈分析

- **Prefill 是瓶颈**: 单 prefill 节点处理 3-4K tokens 需要 ~260ms，128 并发时 TTFT 从 0.72s 涨到 2.44s（排队导致）
- **Decode 远未饱和**: TPOT 在 128 并发下仅 0.108s（vs 单请求 0.104s），几乎无退化
- **每请求延迟稳定**: 平均单请求延迟 63-67s（600 tokens × 0.107s/token ≈ 64s），不随并发显著变化

---

## 环境配置

### Prefill 节点 (b7 - 10.8.0.12)
- GPU: 8x NVIDIA B200, Driver 580.126.09
- CUDA: 12.9, PyTorch 2.9.1+cu129
- SGLang: 0.5.8
- TP=8, mem_fraction_static=0.82
- KV Cache: 990,720 tokens (64.84 GB per GPU)

### Decode 节点 (b8 - 10.8.0.17)
- 同 Prefill 节点硬件配置
- max_running_requests=512
- KV Cache 传输: NIXL (DMA-BUF)

### MIG 信息
- MIG: `chrisya-b200-spot-mig-ase1`
- Zone: `asia-southeast1-b`
- Project: `gpu-launchpad-playground`
- Machine Type: `a4-highgpu-8g` (Spot)
- 尝试扩展到 4 节点（2P2D）时遇到 `ZONE_RESOURCE_POOL_EXHAUSTED`

---

## 启动命令

### Prefill (b7)
```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --tp-size 8 --disable-radix-cache --disable-cuda-graph \
    --mem-fraction-static 0.82 --host 0.0.0.0 --port 30000
```

### Decode (b8)
```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --tp-size 8 --disable-radix-cache --disable-cuda-graph \
    --max-running-requests 512 --mem-fraction-static 0.82 \
    --host 0.0.0.0 --port 30000
```

### Router (b7)
```bash
python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://10.8.0.12:30000 8998 \
    --decode http://10.8.0.17:30000 \
    --host 0.0.0.0 --port 8000
```

### Evalscope 压测
```bash
evalscope perf \
  --parallel 4 8 16 32 64 96 128 \
  --number 16 32 64 128 256 384 512 \
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

1. **128 并发下 decode 仍未饱和**: TPOT 仅 0.108s，output throughput 1,122 tok/s 仍在线性增长区间
2. **Prefill 是瓶颈**: TTFT 随并发线性增长，单 prefill 节点处理 3-4K tokens 需要 ~260ms，高并发下排队严重
3. **TTFT < 500ms 约束**: 最多支持 2 并发（4624 input tokens 场景），input throughput 23,280 tok/s

### 后续计划

1. **2P2D 扩展**: 等待 B200 Spot 容量恢复后，扩展到 2 Prefill + 2 Decode（4 节点 32 GPU），此时需要启用 DeepEP + NVSHMEM IBGDA
2. **预估 2P2D 性能**: 2 prefill + 2 decode 可达 ~2,200+ output tok/s，TTFT 在 128 并发时可从 2.44s 降至 ~1.2s
3. **CUDA Graph**: 启用 CUDA Graph 后预期可显著提升 decode 吞吐量
4. **Radix Cache**: 对重复前缀的场景启用 radix cache 可减少 prefill 计算
5. **DeepSeek-V3-0324**: 测试更新的模型版本
