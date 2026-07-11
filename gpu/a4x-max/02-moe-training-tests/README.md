# GB300 (A4X MAX) MoE 训练测试方案

以 GB200 (A4X) 实测数据为 baseline，在 GB300 上对标测试两个 MoE 模型：DeepSeek V3 和 Qwen3 235B。

## 测试环境

- 集群: tencent-gcp-taiji-poc (us-central1-b)
- 预留: nvidia-gb300-dxkhoz4ypk4mh (214 台, 856 GPU)
- 容器: NeMo 26.06 + Megatron Bridge
- 入口: `run_script.py` (Bridge 优化, 不要用 `pretrain_gpt.py`)
- 拓扑: NVL72 (18 节点/域), B300 Ultra (288 GB HBM3e/GPU)

## 参考文档

- GB200 DSv3 测试: [a4x/07-megatron-training/07c-deepseekv3-671b-recipe/](../../a4x/07-megatron-training/07c-deepseekv3-671b-recipe/)
- GB200 Qwen3 235B 测试: [a4x/07-megatron-training/07b-qwen3-235b-recipe/](../../a4x/07-megatron-training/07b-qwen3-235b-recipe/)
- NeMo Bridge Performance: https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html

---

## Test 1: DeepSeek V3 16L — 64 GPU (16 节点, 单域)

缩减版 DSv3 (16 层, ~110B)，对标 GB200 同配置。

### 参数对比

| 参数 | GB200 (baseline) | GB300 (本次) | 变化原因 |
|------|----------------|------------|---------|
| 模型层数 | 16 | 16 | 不变 |
| PP | 2 | 2 | 16L PP=2 已最优 |
| VP | 2 | 2 | 不变 |
| EP | 64 | 32 | 288GB 每卡放 8 experts (vs 4) |
| TP | 1 | 1 | 不变 |
| MBS | 1 | 1 | DSv3 MoE expert buffer 是瓶颈 |
| GBS | 2048 | 2048 | 不变 |
| seq_len | 4096 | 4096 | 不变 |
| 精度 | MXFP8 | MXFP8 | 不变 |
| CUDA Graph | full_iteration (V2) | full_iteration (V2) | 不变 |
| EP_RANKS_PER_DOMAIN | 64 | 32 | 匹配 EP=32 |
| PP layout | Etttt\|tttt\|tttt\|ttttmL | Etttt\|tttt\|tttt\|ttttmL | 不变 |

### 启动命令

```bash
# 所有 16 Pod 上执行
cd /opt/Megatron-Bridge/scripts/performance
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32   # ← 匹配 EP=32

torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m deepseek -mr deepseek_v3 --task pretrain \
    -g gb300 -c fp8_mx -ng 64 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj deepseek_v3 \
    -cv v2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 32 \
    --num_layers 16 \
    --virtual_pipeline_model_parallel_size 2 \
    --pipeline_model_parallel_layout "Etttt|tttt|tttt|ttttmL" \
    --global_batch_size 2048 \
    --micro_batch_size 1
```

> 注意: `-g gb300` 会加载 GB300 专属 recipe config。如果 Bridge 还没有 gb300 config，用 `-g gb200` 并手动覆盖参数。

### 预期与实测

| 指标 | GB200 baseline | GB300 预期 | GB300 实测 |
|------|---------------|----------|----------|
| MNNVL=2 TFLOPs | 1176 | > 1400 (+20%) | — |
| MNNVL=0 TFLOPs | 1100 | > 1350 (+23%) | — |
| Step Time (MNNVL=2) | ~5.1s | < 4.3s | — |

### 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## Test 2: DeepSeek V3 32L — 128 GPU (32 节点, 跨域)

缩减版 DSv3 (32 层, ~221B)，对标 GB200 同配置 (992 TFLOPs raw Megatron / 1114 Bridge)。

### 参数对比

| 参数 | GB200 (baseline) | GB300 (本次) | 变化原因 |
|------|----------------|------------|---------|
| 模型层数 | 32 | 32 | 不变 |
| PP | 2 | 2 | 不变 |
| VP | — | 4 | GB300 配合 full_iteration graph |
| EP | 64 | 32 | 288GB 每卡更多 expert |
| TP | 1 | 1 | 不变 |
| MBS | 1 | 1 | 不变 |
| GBS | 2048 | 2048 | 不变 |
| seq_len | 8192 | 8192 | 不变 |
| 精度 | MXFP8 | MXFP8 | 不变 |
| CUDA Graph | attn (raw Megatron) / full_iteration (Bridge) | full_iteration (V2) | 统一用 Bridge V2 |
| EP_RANKS_PER_DOMAIN | 64 | 32 | 匹配 EP=32 |
| wgrad-defer | -1 (raw Megatron only) | N/A (Bridge 内置) | Bridge 自动处理 |

### 启动命令

```bash
cd /opt/Megatron-Bridge/scripts/performance
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32

torchrun --nproc_per_node=4 --nnodes=32 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m deepseek -mr deepseek_v3 --task pretrain \
    -g gb300 -c fp8_mx -ng 128 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj deepseek_v3 \
    -cv v2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 32 \
    --num_layers 32 \
    --global_batch_size 2048 \
    --micro_batch_size 1
```

### 预期与实测

| 指标 | GB200 baseline | GB300 预期 | GB300 实测 |
|------|---------------|----------|----------|
| Bridge V2 TFLOPs | 1114 (64 GPU) | > 1400 (+26%) | — |
| raw Megatron TFLOPs | 992 (128 GPU) | > 1200 (+21%) | — |
| Step Time | — | — | — |

### 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## Test 3: Qwen3 235B-A22B — 64 GPU (16 节点, 单域)

全量 94 层 Qwen3 235B MoE 模型。GB300 288GB 显存允许 PP 从 8 降到 4。

### 参数对比

| 参数 | GB200 (baseline) | GB300 (本次) | 变化原因 |
|------|----------------|------------|---------|
| 模型 | Qwen3-235B-A22B, 94L | Qwen3-235B-A22B, 94L | 不变 (全量模型) |
| PP | 8 | **4** | 288GB → 每 stage 24 层 (vs 12 层) |
| VP | — | **6** | 更细粒度 interleaving, 24/6=4 层/virtual stage |
| EP | 8 | **16** | 64/(TP×PP)=16, EP=16 → 8 experts/GPU |
| TP | 1 | 1 | 不变 |
| MBS | 1 | **2** | 288GB 可放更大 micro batch |
| GBS | 1024 | 1024 | 不变 |
| seq_len | 4096 | 4096 | 不变 |
| 精度 | MXFP8 | MXFP8 | 不变 |
| CUDA Graph | TE scoped (V1) | **full_iteration (V2)** | PP=4 更适合 full graph |
| Config variant | V1 | **V2** | 启用 full_iteration + paged stash |
| EP_RANKS_PER_DOMAIN | 8 | **16** | 匹配 EP=16 |

**并行度验证**: TP=1 × PP=4 × EP=16 × DP=1 = 64 GPU ✓

### 启动命令

```bash
cd /opt/Megatron-Bridge/scripts/performance
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=16

torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen -mr qwen3_235b_a22b --task pretrain \
    -g gb300 -c fp8_mx -ng 64 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_235b \
    -cv v2 \
    --pipeline_model_parallel_size 4 \
    --expert_model_parallel_size 16 \
    --virtual_pipeline_model_parallel_size 6 \
    --global_batch_size 1024 \
    --micro_batch_size 2
```

### 预期与实测

| 指标 | GB200 baseline | GB300 预期 | GB300 实测 |
|------|---------------|----------|----------|
| V1 PP=8 EP=8 TFLOPs | 930 | — | — |
| V2 PP=4 EP=16 TFLOPs | — | > 1100 (+18% vs V1) | — |
| NVIDIA 256 GPU 参考 | 1092 (GB200) / 1335 (GB300) | 对标 1335 | — |
| Step Time | 6.47s (V1) | < 5.5s | — |

### 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## Test 4: Qwen3 235B-A22B — 128 GPU (32 节点, 跨域)

全量 94 层 Qwen3 235B，扩展到 128 GPU 验证 scaling。

### 参数对比

| 参数 | GB200 参考 (256 GPU) | GB300 (本次, 128 GPU) | 变化原因 |
|------|-------------------|---------------------|---------|
| PP | 8 | **4** | 288GB + PP=4 足够 |
| VP | 3 | **6** | 更多 interleaving |
| EP | 32 | **32** | 128/(TP×PP)=32 |
| TP | 1 | 1 | 不变 |
| MBS | 1 | **2** | 288GB 允许 |
| GBS | 8192 | 4096 | 按 128 GPU 缩放 |
| 精度 | MXFP8 | MXFP8 | 不变 |
| Config variant | V2 | V2 | 不变 |
| EP_RANKS_PER_DOMAIN | 32 | 32 | 匹配 EP=32 |

**并行度验证**: TP=1 × PP=4 × EP=32 × DP=1 = 128 GPU ✓

### 启动命令

```bash
cd /opt/Megatron-Bridge/scripts/performance
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32

torchrun --nproc_per_node=4 --nnodes=32 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen -mr qwen3_235b_a22b --task pretrain \
    -g gb300 -c fp8_mx -ng 128 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_235b \
    -cv v2 \
    --pipeline_model_parallel_size 4 \
    --expert_model_parallel_size 32 \
    --virtual_pipeline_model_parallel_size 6 \
    --global_batch_size 4096 \
    --micro_batch_size 2
```

### 预期与实测

| 指标 | NVIDIA GB300 256 GPU 参考 | GB300 128 GPU 预期 | GB300 实测 |
|------|------------------------|------------------|----------|
| TFLOPs | 1335 | > 1200 (scaling ~90%) | — |
| Tokens/s/GPU | 9015 | > 8000 | — |
| Step Time | — | — | — |

### 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## 全局对比表

| # | 模型 | 规模 | 平台 | PP | EP | VP | MBS | 精度 | TFLOPs | 来源 |
|---|------|------|------|----|----|----|----|------|--------|------|
| 1 | DSv3 16L | 64 GPU | GB200 | 2 | 64 | 2 | 1 | MXFP8 | 1176 | 实测 (MNNVL=2) |
| 2 | DSv3 16L | 64 GPU | GB200 | 2 | 64 | 2 | 1 | MXFP8 | 1100 | 实测 (MNNVL=0) |
| 3 | DSv3 16L | 64 GPU | **GB300** | **2** | **32** | **2** | **1** | MXFP8 | **—** | **待测** |
| 4 | DSv3 32L | 128 GPU | GB200 | 2 | 64 | — | 1 | MXFP8 | 992 | 实测 (raw Megatron) |
| 5 | DSv3 32L | 128 GPU | **GB300** | **2** | **32** | **4** | **1** | MXFP8 | **—** | **待测** |
| 6 | DSv3 Full | 256 GPU | GB200 | 4 | 64 | 4 | 1 | MXFP8 | 1292 | NVIDIA 参考 |
| 7 | DSv3 Full | 256 GPU | GB300 | 2 | 32 | 8 | 1 | MXFP8 | 1648 | NVIDIA 参考 (+27.6%) |
| 8 | Qwen3 235B | 64 GPU | GB200 | 8 | 8 | — | 1 | MXFP8 | 930 | 实测 (V1) |
| 9 | Qwen3 235B | 64 GPU | GB200 | 2 | 32 | — | 1 | MXFP8 | 1124 | 实测 (V2, 跨域) |
| 10 | Qwen3 235B | 64 GPU | **GB300** | **4** | **16** | **6** | **2** | MXFP8 | **—** | **待测** |
| 11 | Qwen3 235B | 128 GPU | **GB300** | **4** | **32** | **6** | **2** | MXFP8 | **—** | **待测** |
| 12 | Qwen3 235B | 256 GPU | GB200 | 8 | 32 | 3 | 1 | MXFP8 | 1092 | NVIDIA 参考 |
| 13 | Qwen3 235B | 256 GPU | GB300 | 4 | 32 | 12 | 2 | MXFP8 | 1335 | NVIDIA 参考 (+22.3%) |

## GB300 参数调优核心原理

### 为什么 PP 可以减半

GB200: 192 GB/GPU → 94 层 Qwen3 235B 需要 PP=8 (每 stage ~12 层, ~24 GB/stage)
GB300: 288 GB/GPU → PP=4 就够 (每 stage ~24 层, ~48 GB/stage, 288 GB 容得下)

PP 减半的收益:
- Pipeline bubble 从 (PP-1)/PP ≈ 7/8 = 87.5% 理论最大 overhead 降到 3/4 = 75%
- 配合 VP interleaving 后实际 bubble 更小: PP=8 实测 bubble ~30-50%, PP=4 降到 ~15-25%
- 这是 20-30% 性能提升的第一大来源

### 为什么 EP 可以降低

GB200: 192 GB → DSv3 256 experts, EP=64 → 每卡 4 experts (每 expert ~3 GB activation+weight)
GB300: 288 GB → EP=32 → 每卡 8 experts (每 expert ~3 GB, 总计 ~24 GB, 288 GB 轻松)

EP 降低的收益:
- All-to-all 通信量与 EP 成正比, EP 减半 → 通信减半
- HybridEP 在 NVLink 域内更高效

### 为什么 MBS 可以翻倍

GB200: 激活内存紧张, MBS=1 才能放下
GB300: 多 96 GB, 可以 MBS=2, GPU 计算利用率更高 (更大矩阵, MXU 效率更高)

### NVFP4 新精度模式 (可选进阶测试)

B300 Ultra 强化了 FP4 硬件加速。NVIDIA 官方 Llama 405B NVFP4 跑到 3575 TFLOP/s/GPU，比 FP8 的 2646 高 35%。
MoE 模型的 NVFP4 效果待验证 (expert 计算可能受益, 但 routing 精度敏感)。

## 注意事项

1. **`-g gb300` flag**: Bridge recipe 可能还没有 `gb300` GPU type。如果报错，用 `-g gb200` 并手动覆盖所有参数。
2. **NIC 改名**: GB300 用 IDPF 不是 GVNIC，startup script 需用 v2 版本 (自动检测)。
3. **IPv6**: GB300 全栈 IPv6，torchrun master_addr 可能需要 IPv6 格式。
4. **RDMA count**: ResourceClaimTemplate 从 4 改成 8 (CX-8 双端口)。
5. **DRA driver**: 推荐 v25.8.0+ (GB300 CX-8 兼容)。
6. **DSv3 PP layout**: 改层数必须三件套同改: `--num_layers` + `-vp` + `--pipeline_model_parallel_layout`。
