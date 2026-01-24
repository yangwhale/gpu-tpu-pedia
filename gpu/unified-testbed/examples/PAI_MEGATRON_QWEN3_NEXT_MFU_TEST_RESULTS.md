# Pai-Megatron-Patch Qwen3-Next MFU 测试结果

## 硬件配置

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA B200 (A4 High) |
| GPU 数量 | 8 |
| 每 GPU 显存 | 180 GB |
| 理论峰值 (BF16 Dense) | 2,250 TFLOP/s |
| 精度 | BF16 |

## 模型配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-Next-80B-A3B-Instruct |
| 架构 | MoE (Mixture of Experts) + Mamba-Transformer Hybrid |
| 专家数 | 512 |
| 激活专家 | 8 |
| 默认层数 | 96 (测试用 8-48 层) |

## MFU 计算公式

```
MFU = 实际吞吐量 (TFLOP/s/GPU) / 理论峰值 (2,250 TFLOP/s)
Tokens/Chip/s = (SEQ_LEN × GBS) / (单步耗时_ms × 8_GPUs / 1000)
```

## 测试结果汇总

### 8 层配置 (EP=8, TP=1, PP=1)

| Config | Layers | TP | MBS | GBS | SEQ_LEN | 单步耗时 (ms) | 吞吐量 (TFLOP/s/GPU) | MFU | 显存 (GB) | Tokens/Chip/s | 备注 |
|--------|--------|-----|-----|-----|---------|---------------|---------------------|-----|-----------|---------------|------|
| config1 | 8 | 1 | 1 | 8 | 1024 | ~1000 | 24.4 | 1.1% | ~22 | ~1024 | 基线 |
| config2 | 8 | 1 | 2 | 16 | 1024 | ~900 | 29.9 | 1.33% | ~24 | ~2275 | |
| config3 | 8 | 1 | 4 | 32 | 1024 | ~850 | 32.4 | 1.44% | ~26 | ~4800 | |
| config4 | 8 | 1 | 2 | 16 | 2048 | ~950 | 54.9 | 2.44% | ~28 | ~4310 | |
| config5 | 8 | 1 | 4 | 32 | 2048 | ~900 | 62.1 | 2.76% | ~32 | ~9100 | |
| config6 | 8 | 1 | 1 | 8 | 4096 | ~1100 | 88.0 | 3.91% | ~24 | ~3720 | |
| config_max | 8 | 1 | 6 | 48 | 4096 | ~1050 | **118.4** | **5.26%** | ~57 | ~23400 | 8层最优 |

### 48 层配置 - SEQ=4096 (EP=8, TP=1, PP=1)

| Config | Layers | TP | MBS | GBS | SEQ_LEN | 单步耗时 (ms) | 吞吐量 (TFLOP/s/GPU) | MFU | 显存 (GB) | Tokens/Chip/s | 备注 |
|--------|--------|-----|-----|-----|---------|---------------|---------------------|-----|-----------|---------------|------|
| config_48L | 48 | 1 | 2 | 16 | 4096 | ~1650 | 58.9 | 2.62% | ~95 | ~4970 | MBS太小 |
| config_48L_v2 | 48 | 1 | 4 | 32 | 4096 | ~3000 | 65.3 | 2.90% | ~95 | ~5460 | |
| config_full_mem | 48 | 1 | 8 | 64 | 4096 | ~5700 | 67.8 | 3.01% | ~95 | ~5770 | |
| config_full_mem_v3 | 48 | 1 | 16 | 128 | 4096 | ~13000 | 63.9 | 2.84% | ~95 | ~5040 | 梯度累积 |

### 48 层配置 - SEQ=8192 (EP=8, TP=1, PP=1)

| Config | Layers | TP | MBS | GBS | SEQ_LEN | 单步耗时 (ms) | 吞吐量 (TFLOP/s/GPU) | MFU | 显存 (GB) | Tokens/Chip/s | 备注 |
|--------|--------|-----|-----|-----|---------|---------------|---------------------|-----|-----------|---------------|------|
| config_seq8k_v2 | 48 | 1 | 2 | 16 | 8192 | ~2280 | 85-88 (峰值106) | **3.78-3.91%** | ~95 | ~7200 | **48L最优** |
| config_seq8k_v3 | 48 | 1 | 4 | 32 | 8192 | ~4500-6400 | 60-107 (avg ~82) | 3.6-3.8% | ~95 | ~7000 | 波动大 |
| config_seq8k_v4 | 48 | 1 | 8 | 64 | 8192 | - | - | - | >180? | - | **OOM 失败** |

### TP=8 配置 (48 层, EP=8, PP=1)

| Config | Layers | TP | MBS | GBS | SEQ_LEN | 单步耗时 (ms) | 吞吐量 (TFLOP/s/GPU) | MFU | 显存 (GB) | Tokens/Chip/s | 备注 |
|--------|--------|-----|-----|-----|---------|---------------|---------------------|-----|-----------|---------------|------|
| config_tp8 | 48 | 8 | 8 | 64 | 4096 | 5600-8700 | 43-67 | 1.9-3.0% | ~95 | ~4000 | 波动大 |

## 关键发现

### 1. 显存利用率被框架优化限制
- 8 层 config_max 仅用 57 GB / 180 GB = **31.7%**
- 48 层配置固定在 95 GB / 180 GB = **52.8%**
- **增加 GBS 不增加显存**: Megatron-LM 使用梯度累积处理大 GBS
- 显存主要由模型参数 + 优化器状态 + 一个 micro-batch 的激活决定

### 2. MFU 低的根本原因
- **MoE 稀疏性**: 每次只激活 8/512 = 1.56% 专家
- **计算 vs 通信比**: EP=8 导致大量专家间 all-to-all 通信
- **Mamba 层**: 约 3/4 的层是 Mamba (M-M-M-* 模式)，不是 Transformer
- **混合架构开销**: Mamba 和 Transformer 交替执行的切换开销

### 3. 各配置 MFU 对比
- 8层 config_max: **5.26%** (最高，计算/通信比最优)
- 48层 config_full_mem: **3.01%** (层数增加但通信开销更大)
- TP=8 config: **~2.5%** (TP 通信开销严重)

### 4. 为什么 8 层比 48 层 MFU 更高
- 8 层: 模型小，通信少，GPU 更多时间在计算
- 48 层: 模型大，EP 通信增加，MFU 反而下降

## 提高 MFU 的可行方案

### 方案 1: 关闭激活检查点 (增加显存使用)
```bash
# 修改 run_mcore_qwen3.sh
# 将 ac_recompute=true 改为 false
# 风险: 可能 OOM
```

### 方案 2: 增加序列长度到 8192+
```bash
SEQ_LEN=8192; MBS=1; GBS=8  # 更长序列增加计算密度
```

### 方案 3: 使用完整 96 层模型
```bash
NUM_LAYERS=96; MBS=1; GBS=8; SEQ_LEN=4096
# 预计显存: ~190 GB (需要 TP 或 PP)
```

### 方案 4: 减少专家数量 (非 MoE 模式)
- 如果有 Dense 版本模型，MFU 会显著提高
- MoE 模型本质上 MFU 较低

### 方案 5: 启用 Flash Attention v3
- 检查是否已启用 NVTE_FUSED_ATTN
- B200 支持 FP8 精度，可进一步加速

## 使用方法

```bash
# 设置配置
export MFU_TEST_CONFIG="config_max"

# 部署测试
cd gpu/unified-testbed
helm install chrisya-mfu-test gke-runtime/jobset \
    -f gke-runtime/values.yaml \
    -f gke-runtime/mfu-test-values.yaml \
    --set-file task_script=examples/pai-megatron-qwen3-next-mfu-test.sh

# 查看结果
kubectl logs -f $(kubectl get pods | grep mfu-test | awk '{print $1}') -c workload
```

## 参考链接

- [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [Qwen3-Next 模型](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
