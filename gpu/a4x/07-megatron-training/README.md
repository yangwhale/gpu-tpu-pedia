# 8. Megatron-LM 训练

## DRA 资源声明模式（核心 YAML 结构）

以下为训练 YAML 中 DRA 相关的关键字段精简展示：

```yaml
# 1. ResourceClaimTemplate — 声明需要 4 张 RDMA 网卡（由 DRANET 分配）
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: rdma-nics-mega-host-1
spec:
  spec:
    devices:
      requests:
      - name: rdma-nics
        exactly:
          deviceClassName: rdma-devices    # ← DeviceClass，由 DRANET 提供
          count: 4                         # ← A4X 每节点 4 张 RDMA NIC

---
# 2. Pod spec — 通过 resourceClaims 引用 ComputeDomain 和 RDMA
apiVersion: v1
kind: Pod
metadata:
  name: mega-host-1
spec:
  nodeSelector:
    topology: same-domain                  # ← 手动调度到目标域
  containers:
  - name: pytorch
    image: nvcr.io/nvidia/pytorch:26.05-py3
    resources:
      limits:
        nvidia.com/gpu: 4                  # ← GPU 由 nvidia-device-plugin 分配
      claims:
      - name: compute-domain-channel       # ← 引用下方 resourceClaims[0]
      - name: rdma-nics                    # ← 引用下方 resourceClaims[1]

    resourceClaims:                          # ← Pod 级资源声明
    - name: compute-domain-channel
      resourceClaimTemplateName: sd-compute-domain-channel  # ← ComputeDomain 自动生成
    - name: rdma-nics
      resourceClaimTemplateName: rdma-nics-mega-host-1      # ← 手动创建的模板
```

**DRA ≠ 调度**：上述 `resourceClaims` 解决的是「资源分配」——向 K8s DRA 框架申请 IMEX channel 和 RDMA 网卡。但 DRA **不负责**将多个 Pod 调度到同一 NVL72 域。

**域级调度**需要另外的机制：
- **手动方式**：`nodeSelector: {topology: same-domain}`（本章方法，适合验证）
- **生产方式**：Kueue TAS 的 `podset-required-topology` 注解（详见 [08-multi-domain](../08-multi-domain/)）

## NGC 镜像 Rerun State Machine 踩坑（重要）

NGC 镜像 `megatron-ngc:tev2.15-mgcore_r0.16.0-pt26.05-py3-v2` 内置的 Megatron-LM 有一个 **Rerun State Machine**（硬件故障检测机制），默认配置会导致 FP8 训练只跑 2 个 iteration 就退出。

**现象**：训练启动正常，日志显示 `train_iters=50`，但 `[after training is done]` 在第 2 个 iteration 后就出现。没有任何报错。

**根因**：`megatron/training/resilience_config.py` 中 `rerun_mode` 默认值为 `"validate_results"`。该模式每个 training step 执行两遍 forward/backward，对比结果一致性以检测 GPU 硬件故障。FP8 训练（`--fp8-format hybrid`）本身是非确定性的，两次计算结果必然不同，state machine 误判为硬件故障并退出。

**修复**：启动脚本中 patch 默认值：

```bash
# 必须在 torchrun 之前执行
MEGATRON_DIR=$(find /opt -name "pretrain_gpt.py" -type f 2>/dev/null | head -1 | xargs dirname)
sed -i 's/rerun_mode.*=.*"validate_results"/rerun_mode: Literal["disabled", "validate_results", "report_stats"] = "disabled"/' \
  $MEGATRON_DIR/megatron/training/resilience_config.py
```

> **注意**：GRPO 相关参数（`--grpo-iterations` 等）虽然也在镜像的 `arguments.py` 中定义，但它们全部被 `perform_rl_step=False` 门控，预训练模式下完全不生效，与此问题无关。

> **另一个坑**：iteration 日志（TFLOP/s、elapsed time）使用 `print_rank_last` 输出到**最后一个 rank**，不是 rank 0。多节点训练时需要查看 worker 节点（最高 rank 所在 pod）的日志才能看到 throughput 数据。

## 部署训练 Pod

```bash
# 部署 Megatron 测试 Pod（ComputeDomain + DRANET）
kubectl apply -f yamls/k8s1341-megatron-train-dranet.yaml

# 等待 Pod 就绪（自动 clone Megatron-LM + pip install）
kubectl get pods -l name -w
kubectl logs mega-h1 -f  # 确认 "Ready"

# 交换 SSH 密钥（ed25519）
HOST1_KEY=$(kubectl exec mega-h1 -- cat /root/.ssh/id_ed25519.pub)
HOST2_KEY=$(kubectl exec mega-h2 -- cat /root/.ssh/id_ed25519.pub)
kubectl exec mega-h1 -- bash -c "echo '$HOST2_KEY' >> /root/.ssh/authorized_keys"
kubectl exec mega-h2 -- bash -c "echo '$HOST1_KEY' >> /root/.ssh/authorized_keys"

# 获取 IP
MEGA_HOST1_IP=$(kubectl get pod mega-h1 -o jsonpath='{.status.podIP}')
```

## 8.1 单节点 Megatron 训练（Qwen3 30B-A3B MoE FP8，50 iterations）

单节点 EP=4（4 GPU expert 并行），使用 Qwen3 30B-A3B MoE 模型 + TransformerEngine FP8 充分利用 B200 算力。

**关键配置**：

- `--transformer-impl transformer_engine` 启用 TE 融合算子（**不要**使用 `local`）
- `--fp8-format hybrid` + `--fp8-recipe delayed` 启用 FP8 训练
- 环境变量 `NVTE_FUSED_ATTN=1` / `NVTE_NORM_*_USE_CUDNN=1` 启用 GB200 优化内核
- `--expert-model-parallel-size 4`（单节点 128 experts / 4 GPU = 32 experts/GPU）
- `--moe-grouped-gemm` 启用分组 GEMM 优化 MoE 计算

```bash
kubectl exec mega-host-1 -- bash -c "
  source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:\$LD_LIBRARY_PATH
  export PYTHONPATH=/scratch-data/Megatron-LM:\$PYTHONPATH
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
  export NVTE_NORM_FWD_USE_CUDNN=1
  export NVTE_NORM_BWD_USE_CUDNN=1
  export NVTE_FUSED_ATTN=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  cd /scratch-data/Megatron-LM

  torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=29502 \
    pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --num-layers 12 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 128 \
    --qk-layernorm \
    --num-experts 128 \
    --moe-ffn-hidden-size 768 \
    --moe-router-topk 8 \
    --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --moe-router-force-load-balancing \
    --seq-length 16384 \
    --max-position-embeddings 16384 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --rotary-percent 1.0 \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --attention-backend fused \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --bf16 \
    --grad-reduce-in-bf16 \
    --fp8-format hybrid \
    --fp8-recipe delayed \
    --fp8-amax-history-len 1024 \
    --fp8-amax-compute-algo max \
    --fp8-param-gather \
    --cross-entropy-loss-fusion \
    --calculate-per-token-loss \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --manual-gc \
    --empty-unused-memory-level 1 \
    --train-iters 50 \
    --lr 0.00015 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-warmup-iters 10 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 128256 \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --num-workers 1 \
    --split '99,1,0' \
    --log-interval 1 \
    --log-throughput \
    --eval-iters 0 \
    --eval-interval 10000 \
    --save-interval 10000 \
    --distributed-timeout-minutes 60
"
```

## 8.2 多节点 Megatron 训练（2 节点 EP=8，Qwen3 30B-A3B MoE FP8，MNNVL）

**Hostname 解析注意**：使用 Pod 网络时，确保 `GLOO_SOCKET_IFNAME=eth0` 和 `NCCL_SOCKET_IFNAME=eth0`，避免 Gloo bootstrap 选错网络接口。如 Pod 设置了自定义 `hostname`，需在 `/etc/hosts` 中添加 hostname→IP 映射。

**MNNVL**：多节点 EP=8 跨 2 台 A4X 使用 NVLink 域互联（MNNVL），需 ComputeDomain + IMEX daemon。128 experts / 8 GPU = 16 experts/GPU。

```bash
# 获取 Pod IP
MEGA_HOST1_IP=$(kubectl get pod mega-host-1 -o jsonpath='{.status.podIP}')

# 在 mega-host-2 上先启动（node_rank=1）— 后台运行
kubectl exec mega-host-2 -- bash -c "
  source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:\$LD_LIBRARY_PATH
  export PYTHONPATH=/scratch-data/Megatron-LM:\$PYTHONPATH
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
  export NVTE_NORM_FWD_USE_CUDNN=1
  export NVTE_NORM_BWD_USE_CUDNN=1
  export NVTE_FUSED_ATTN=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export GLOO_SOCKET_IFNAME=eth0
  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_MNNVL_ENABLE=2
  export NCCL_CUMEM_ENABLE=1
  export USE_MNNVL=1

  cd /scratch-data/Megatron-LM

  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=\$MEGA_HOST1_IP --master_port=29503 \
    pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --num-layers 12 --hidden-size 2048 --ffn-hidden-size 6144 \
    --num-attention-heads 32 --group-query-attention --num-query-groups 4 \
    --kv-channels 128 --qk-layernorm \
    --num-experts 128 --moe-ffn-hidden-size 768 --moe-router-topk 8 \
    --moe-router-dtype fp32 --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm --moe-router-force-load-balancing \
    --seq-length 16384 --max-position-embeddings 16384 \
    --position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0 \
    --swiglu --normalization RMSNorm \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --attention-backend fused --init-method-std 0.01 \
    --micro-batch-size 1 --global-batch-size 128 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --bf16 --grad-reduce-in-bf16 \
    --fp8-format hybrid --fp8-recipe delayed \
    --fp8-amax-history-len 1024 --fp8-amax-compute-algo max --fp8-param-gather \
    --cross-entropy-loss-fusion --calculate-per-token-loss \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --manual-gc --empty-unused-memory-level 1 \
    --train-iters 50 --lr 0.00015 --min-lr 0.00001 \
    --lr-decay-style cosine --lr-warmup-iters 10 \
    --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
    --mock-data --tokenizer-type NullTokenizer --vocab-size 128256 \
    --no-create-attention-mask-in-dataloader --no-mmap-bin-files \
    --num-workers 1 --split '99,1,0' \
    --log-interval 1 --log-throughput --eval-iters 0 \
    --eval-interval 10000 --save-interval 10000 \
    --distributed-timeout-minutes 60
" &

# 在 mega-host-1 上运行（node_rank=0）— 同上但 --node_rank=0
```

## Qwen3 30B-A3B MoE EP Scaling 测试

### 测试目标

在 NVL72 域内用不同规模的 GPU 测试 Qwen3 30B-A3B MoE 训练的 Expert Parallelism scaling 表现。模型小（30B 总参数，3B 激活）不需要 PP，纯 EP 模式最干净，直接体现 NVSwitch all-to-all 效率随 GPU 数的变化。

### 模型规格

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-30B-A3B |
| 架构 | MoE, 128 routed experts + shared expert |
| 总参数 | 30B |
| 每 token 激活参数 | 3B |
| TopK | 8 |
| 精度 | FP8 (Transformer Engine, hybrid recipe, delayed scaling) |
| 数据 | Mock data (合成数据，无需下载) |

### 测试矩阵

### 显存分析与 MBS 计算

EP 越大每卡 Expert 越少，显存释放出来应调大 MBS 提高 GPU 利用率。

| EP | Experts/GPU | 模型占用 (GB) | 可用于 Activation (GB) | 推荐 MBS | GBS (GA=4) |
|---|---|---|---|---|---|
| 4 | 32 | 72.8 | 119 | **16** | 256 |
| 8 | 16 | 39.4 | 153 | **32** | 1024 |
| 16 | 8 | 22.7 | 169 | **32** | 2048 |
| 32 | 4 | 14.3 | 178 | **32** | 4096 |

> 模型占用 = Expert 权重(FP8) + Expert optimizer(FP32 AdamW) + Shared 权重(BF16) + Shared optimizer(distributed)。Activation 按每 sample ~3 GB 估算（含 recompute）。MBS 取 2 的幂次 + 20% headroom。

### Batch 层级说明

Megatron-LM 的 batch 分三层：

| 层级 | 含义 | 计算方式 |
|---|---|---|
| **Micro Batch Size (MBS)** | 单卡单次 forward/backward 的 sample 数 | `--micro-batch-size` 直接设 |
| **Per-Device Mini Batch** | 单卡每个 optimizer step 累积的总 sample 数 | GA × MBS |
| **Global Batch Size (GBS)** | 所有卡每个 optimizer step 的总 sample 数 | GA × MBS × DP_effective |

在 EP 模式下，DP_effective = EP（对 attention 层来说 EP 张卡天然是 DP）。

**GA = GBS ÷ (MBS × EP)**：每卡累积 GA 次 micro batch 后做一次参数更新。

#### 每卡每 step 的 token 数

| Test | EP | MBS | GA | Per-Device Samples | Per-Device Tokens (seq=4096) |
|---|---|---|---|---|---|
| 1 | 4 | 16 | 256÷(16×4)=4 | 64 | 262,144 |
| 2 | 8 | 32 | 1024÷(32×8)=4 | 128 | 524,288 |
| 3 | 16 | 32 | 2048÷(32×16)=4 | 128 | 524,288 |
| 4 | 32 | 32 | 4096÷(32×32)=4 | 128 | 524,288 |

Test 2-4 每卡每 step 处理量相同（128 sample = 52 万 token），保证 per-GPU 计算负载一致，差异仅来自 all-to-all 通信开销。

#### Test 1: 单节点 4 GPU

| 参数 | 值 |
|---|---|
| GPU 数 | 4 (1 node) |
| EP | 4 |
| TP | 1 |
| PP | 1 |
| DP | 1 |
| MBS | 16 |
| GBS | 256 (MBS=16 × GA=4 × DP_eff=4) |
| Seq Length | 4096 |
| 每卡模型 | 72.8 GB (32 Experts + shared) |
| 通信 | 域内 NVLink (intra-node) |
| 预期 | baseline（之前 MBS=1 测得 ~356 TFLOP/s/GPU，MBS=16 预期更高） |

**要点**：MBS=1 时 GPU 计算单元没有被充分利用（batch 太小，kernel launch overhead 占比高）。MBS=16 应显著提升 TFLOP/s。

#### Test 2: 双节点 8 GPU

| 参数 | 值 |
|---|---|
| GPU 数 | 8 (2 nodes) |
| EP | 8 |
| TP | 1 |
| PP | 1 |
| DP | 1 |
| MBS | 32 |
| GBS | 1024 (MBS=32 × GA=4 × DP_eff=8) |
| Seq Length | 4096 |
| 每卡模型 | 39.4 GB (16 Experts + shared) |
| 通信 | 域内 NVSwitch MNNVL (inter-node) |
| 预期 | 待测（之前 MBS=1 为 ~274 TFLOP/s/GPU，MBS=32 预期更高） |

**要点**：跨节点 MNNVL all-to-all + 大 MBS。显存释放后 MBS 翻倍到 32，计算效率应大幅提升。

#### Test 3: 四节点 16 GPU

| 参数 | 值 |
|---|---|
| GPU 数 | 16 (4 nodes) |
| EP | 16 |
| TP | 1 |
| PP | 1 |
| DP | 1 |
| MBS | 32 |
| GBS | 2048 (MBS=32 × GA=4 × DP_eff=16) |
| Seq Length | 4096 |
| 每卡模型 | 22.7 GB (8 Experts + shared) |
| 通信 | 域内 NVSwitch MNNVL |
| 预期 | 待测 |

**要点**：EP=16，每卡只有 8 Expert。模型占用大幅下降到 23 GB，MBS=32 不变（已接近计算效率甜点）。观察 all-to-all 通信开销是否随 EP 增大而明显。

#### Test 4: 八节点 32 GPU

| 参数 | 值 |
|---|---|
| GPU 数 | 32 (8 nodes) |
| EP | 32 |
| TP | 1 |
| PP | 1 |
| DP | 1 |
| MBS | 32 |
| GBS | 4096 (MBS=32 × GA=4 × DP_eff=32) |
| Seq Length | 4096 |
| 每卡模型 | 14.3 GB (4 Experts + shared) |
| 通信 | 域内 NVSwitch MNNVL |
| 预期 | 待测 |

**要点**：EP=32，每卡仅 4 Expert，模型只占 14 GB。MBS=32 + GA=4，GBS=4096。最大规模测试，验证 NVSwitch all-to-all 在高 EP 下的效率。

### 统一配置

所有 4 组测试保持以下参数一致，仅改变 GPU 数和 EP：

```bash
# Megatron-LM 通用参数
--num-experts 128 \
--moe-router-topk 8 \
--expert-model-parallel-size ${EP} \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--micro-batch-size 1 \
--global-batch-size 64 \
--seq-length 4096 \
--max-position-embeddings 4096 \
--train-iters 50 \
--mock-data \
--fp8-format hybrid --fp8-recipe delayed \
--transformer-impl transformer_engine \
--use-distributed-optimizer \
--overlap-grad-reduce \
--sequence-parallel \
--log-interval 1 --log-throughput --eval-iters 0
```

```bash
# GB200 环境变量
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NCCL_MNNVL_ENABLE=2
export NCCL_CUMEM_ENABLE=1
```

### 部署方式

使用 hostNetwork 模式（绕过 DRANET GID 问题）+ ComputeDomain channel（IMEX for MNNVL）。每组测试：

1. 创建对应数量的 worker VM（Domain 2, `forrest-a4x-1x72-policy`）
2. Join k8s 集群 + label + ComputeDomain
3. 部署 StatefulSet（hostNetwork + ComputeDomain channel）
4. torchrun 启动 Megatron-LM 训练
5. 跑 50 步，取 iter 5-44 平均 TFLOP/s/GPU

### 性能指标采集

- `s/iter`：每步训练时间
- `TFLOPS/GPU`：单卡吞吐
- `MFU`：TFLOPS/GPU ÷ 峰值算力（GB200 FP8 ~4500 TFLOP/s → MFU% = TFLOPS/GPU ÷ 4500）
- `all-to-all time`：EP 通信时间占比（如可从 profiler 获取）

### Benchmark 结果

| Test | GPU | EP | MBS | GBS | TFLOP/s/GPU | MFU | vs Test 1 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 1 (旧) | 4 | 4 | 1 | 64 | ~356 | ~7.9% | — | 旧 baseline, MBS=1, seq=16K, 128 Expert |
| 2 (旧) | 8 | 8 | 1 | 64 | ~274 | ~6.1% | -23% | 旧 baseline, MBS=1, 128 Expert |
| 1a | 4 | 4 | 1 | 64 | **~178** | ~4.0% | — | 64 Expert, MBS=1, FusedAttn |
| 1b | 4 | 4 | 2 | 64 | **~475** | **~10.6%** | +167% vs 1a | 64 Expert, MBS=2, FusedAttn |
| 1c | 4 | 4 | 4 | 64 | OOM | — | — | 64 Expert, MBS=4 超出显存 |
| 2a (旧) | 8 | 8 | 2 | 64 | **~116** | **~2.6%** | -76% vs 1b | 12层 64E, MBS=2, MNNVL, 稳态 iter 10-15 |
| 2b (旧) | 8 | 8 | 4 | 256 | **~200** | **~4.4%** | -58% vs 1b | 12层 64E, MBS=4, MNNVL, 稳态 iter 3-10 |
| 2c (旧) | 8 | 8 | 1 | 64 | **~105** | **~2.3%** | — | 完整48层 128E, MBS=1, MNNVL, 显存149/184GB |
| 2d (旧) | 8 | 8 | 2 | 128 | OOM | — | — | 完整48层 128E, MBS=2 OOM (activation 超限) |
| 2e TP=2 | 8 | 4 | 1 | 16 | OOM | — | — | 完整48层, TP=2 EP=4, 每卡32E更多 OOM |
| 2f FSDP | 8 | 8 | 1 | 64 | 崩溃 | — | — | megatron-fsdp ZeRO-2/3 DTensor 不兼容（BF16 也崩） |
| 2g 正确配置 | 8 | 8 | 4 | 256 | **~140** | **~6.2%** | — | 正确30B（HF config），BF16, recompute, mock data |
| 2h 真实数据 | 8 | 8 | 2 | 128 | **~98** | **~4.3%** | — | 正确30B, BF16, NFS真实数据, loss 12.0→9.7 |

### Benchmark 结果 v2（mcore v0.17.0 + 标准化方法）

上面的旧结果使用 NGC megatron 镜像（mcore r0.16.0）+ 非标准配置（48 层全模型、自定义 GIB 设置），结果不具可比性。以下是使用标准化 benchmark 方法重新测试的结果。

**环境变更**：
- **镜像**：`nvcr.io/nvidia/pytorch:26.04-py3`（标准 NGC PyTorch，非 Megatron 定制镜像）
- **Megatron-Core**：v0.17.0（GitHub `core_v0.17.0` tag，修复了 r0.16.0 的 FSDP DTensor bug）
- **层数**：12 层（标准 MoE benchmark 方法，MFU 归一化后与全模型等价）
- **GIB**：v1.1.2 + `LD_PRELOAD` 加载 NCCL（含 libibverbs/libmlx5 RDMA 库）
- **数据**：`--mock-data`（消除 I/O 变量）
- **序列长度**：16384

**2 节点 8 GPU（同域 MNNVL）Sweep 结果**：

| Config | EP | MBS | GBS | Recompute | Dtype | TFLOP/s/GPU | MFU (BF16) | HBM Peak (GiB) | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 8 | 1 | 256 | none | BF16 | **~492** | **21.9%** | 60 | EP=8 baseline |
| A2 | 8 | 2 | 256 | none | BF16 | **~527** | **23.4%** | 102 | **最佳配置** |
| A3 | 8 | 2 | 256 | none | FP8 | **~503** | 22.4% | 93 | FP8 反而慢 5%（MoE grouped GEMM FP8 开销） |
| A4 | 8 | 2 | 256 | selective | BF16 | **~480** | 21.3% | 105 | recompute 开销 ~9% |
| A5 | 8 | 4 | 256 | selective | BF16 | OOM | — | — | MBS=4 即使开 selective recompute 也 OOM |
| A6 | 4 | 1 | 256 | none | BF16 | **~463** | 20.6% | 63 | EP=4 DP=2，通信换并行 |
| A7 | 4 | 2 | 256 | none | BF16 | **~524** | 23.3% | 105 | EP=4 接近 EP=8 最佳 |

> MFU 基于 GB200 BF16 峰值 2,250 TFLOP/s 计算。FP8 MFU 若基于 FP8 峰值 4,500 TFLOP/s 则为 11.2%。

**关键发现**：
1. **MBS=2 是甜点**：MBS=1→2 提升 7%，MBS=4 OOM。HBM 从 60 GiB 跳到 102 GiB
2. **FP8 对 MoE 无加速**：grouped GEMM 的 FP8 路径 overhead 抵消了 Tensor Core 加速，BF16 反而更快
3. **EP=4 vs EP=8 差距小**：MBS=2 时 EP=4（524 TFLOP/s）接近 EP=8（527 TFLOP/s），说明 NVSwitch all-to-all 效率高
4. **Selective recompute 代价 ~9%**：从 527 降到 480 TFLOP/s，但能省 HBM（用于更大 MBS 场景）
5. **v0.17.0 vs v0.16.0**：同配置（EP=8 MBS=1 BF16）从 ~105 升到 ~492 TFLOP/s（4.7×），主要因为旧测试是 48 层全模型 + 旧版本

**8 节点 32 GPU（同域 MNNVL）完整 48 层模型结果**：

从 12 层 benchmark 扩展到完整 48 层模型。环境同上（mcore v0.17.0 + NGC PyTorch 26.04 + GIB LD_PRELOAD），改为 48 层 + 8 节点 32 GPU。

| Config | Layers | EP | MBS | GBS | Recompute | TFLOP/s/GPU | MFU (BF16) | HBM Peak (GiB) | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 48 | 32 | 1 | 256 | none | **~446** | **19.8%** | 147 | **最佳配置** — 完整模型无 recompute |
| B2 | 48 | 32 | 2 | 256 | none | OOM | — | 165 | MoE dispatch buffer OOM |
| B3 | 48 | 32 | 2 | 256 | full | **~372** | 16.5% | 67 | recompute 省 80 GiB 但慢 17% |
| B4 | 48 | 32 | 4 | 128 | full | **~375** | 16.7% | 113 | MBS 翻倍但 TFLOP/s 不涨 |
| B5 | 48 | 32 | 8 | 256 | full | OOM | — | — | vocab logits [16384,8,128K] FP32 = 62.7 GiB |
| B6 | 48 | 32 | 2 | 256 | selective | OOM | — | — | selective 省不够，NCCL CUDA error |
| B7-FSDP | 48 | 32 | 2 | 256 | none | NCCL hang | — | — | `--use-megatron-fsdp` 与 EP=32 死锁 |

> **B1 vs B3/B4**：不开 recompute 的 MBS=1（446 TFLOP/s）优于开 full recompute 的 MBS=2/4（~375 TFLOP/s）。recompute 的计算开销（~25%）大于 MBS 增大带来的效率提升。

> **B5 OOM 分析**：MBS=8 的 OOM 不是 MoE dispatch，而是 cross-entropy loss 的 vocab logits 张量 [seq=16384, MBS=8, vocab=128256] × FP32 = 62.7 GiB。这是 vocab 大小 × 序列长度的固有限制，与 MoE 无关。解决方案：TP>1 将 vocab 切分到多卡。

> B1 vs A1（12 层 EP=8）：446 vs 492 TFLOP/s，下降 9.4%。主要来自 EP=32 的 all-to-all 通信开销增大（32 路 vs 8 路），而非层数增加。

**FSDP 现状**：mcore v0.17.0 修复了 r0.16.0 的 DTensor `main_grad` bug（`--ckpt-format fsdp_dtensor` 需显式指定），但 `--use-megatron-fsdp` 与 EP=32 组合在 NCCL 初始化阶段死锁。去掉 `--use-distributed-optimizer` 和 overlap 参数后依然 hang。不开 FSDP 的 48 层 EP=32 MBS=1 可正常运行。

**GQA 与 TP 约束**：Qwen3 30B-A3B 使用 GQA（Q heads=32, KV heads=4）。TP 必须同时整除 Q heads 和 KV heads，因此 TP 最大为 4。TP 与 EP 正交（TP 切 attention，EP 切 MoE），TP=4 + EP=32 在 32 GPU 上可共存。

**集群重建复测记录（2026-07-01）**：

按 GitHub 文档标准流程重建集群（6 NIC + Placement Policy + ComputeDomain + DRA + DRANET + GIB LD_PRELOAD），遇到以下问题：

1. **Worker VM 创建 stockout**：最初只配了 1 NIC（无 RDMA NIC + 无 Placement Policy），reservation stockout。加上 6 NIC（2 GVNIC + 4 MRDMA）+ `--resource-policies=a4x-nvl72-policy` 后立刻成功。Placement Policy 决定物理域分配，缺了就分不到机器。注意 `chrisya-a4x-nvl72-domain-1` 和 `a4x-nvl72-policy` 是两个不同的 Policy，前者没有空位
2. **VPC 选择**：forrest VPC 有组织策略自动删除防火墙规则，IAP SSH 不可用。改用 `chrisya-gvnic-net-0` VPC（与本地机器内网互联）
3. **NCCL CUDA error 801**：集群搭好后训练报 `CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error`，即使设 `NCCL_MNNVL_ENABLE=0` 也报错（GIB 脚本内部覆盖回 2）。ComputeDomain 和 IMEX daemon 状态均正常（Ready），但 NCCL 通信失败。待排查：可能是 Placement Policy 物理域分配与 ComputeDomain 不匹配、或 GIB NCCL 版本与 Rocky 580 驱动不兼容

### Benchmark 结果 v3（Megatron Bridge NeMo 26.06）

使用 NVIDIA 官方 Megatron Bridge（NeMo 26.06 容器）+ 官方 Qwen3 30B recipe 在 2 节点 8 GPU（forrest-a4x-1x72-policy 域，v3 镜像）上测试。

**环境**：`nvcr.io/nvidia/nemo:26.06` + GIB v1.1.2 LD_PRELOAD + ComputeDomain

| Config | Framework | Precision | MBS | GBS | CUDA Graph | TFLOP/s/GPU | HBM Peak (GiB) | 备注 |
|---|---|---|---|---|---|---|---|---|
| C1 | Megatron Bridge | MXFP8 | 4 | 512 | full_iteration | 80-86 → OOM | 186+ | CUDA Graph replay buffer 超出 184 GiB |
| C2 | Megatron Bridge | MXFP8 | 4 | 512 | none | **~89** | 186 (76 retries) | hybridep + fp8_attn + moe_a2a_overlap |
| **官方** | Megatron Bridge | MXFP8 | 4 | 512 | full_iteration | **936** | DGX-GB200 | NVIDIA 官方 Performance Summary |

> **C2 vs 官方差距分析（89 vs 936 = 10.5×）**：
> 1. **CUDA Graph**：官方用 `full_iteration` CUDA Graph 消除 kernel launch + Python 开销，A4X 的 184 GiB HBM 放不下 replay buffer
> 2. **DGX vs A4X**：DGX-GB200 可能有不同的 NVSwitch 拓扑或内存管理优化
> 3. **cutedsl_fused_grouped_mlp**：融合 grouped MLP 需要 cuTeDSL 支持，可能在 A4X 上未完全启用
> 4. **NCCL 版本**：GIB NCCL 2.30.4+cuda13.0 vs DGX 自带的优化 NCCL

> **关键发现**：Megatron Bridge 的 `run_recipe.py` 不会加载 GPU 特定的优化配置（CUDA Graph、hybridep 等），必须用 `run_script.py` 作为入口。

### Worker 镜像 v3（2026-07-01）

打包镜像 `chrisya-a4x-worker-v3`，解决 v2 的新内核启动问题：
- 删除新内核 5.14.0-687，锁定旧内核 5.14.0-611（dnf versionlock）
- 启动即用：nvidia-smi + containerd + kubeadm + IMEX channel 直接可用
- Lustre 2.14 DKMS 已编译安装（ARM64 patch）
- 启动到 k8s Ready 总时间：~6.5 分钟（硬件初始化 3 分钟 + OS 启动 2 分钟 + CNI 1.5 分钟）

**方法论差异（旧 vs 新）**：

| 维度 | 旧方法（mcore r0.16.0） | 新方法（mcore v0.17.0） |
|---|---|---|
| 镜像 | `megatron-ngc:tev2.15-mgcore_r0.16.0` | `nvcr.io/nvidia/pytorch:26.04-py3` |
| Megatron 来源 | NGC 预装 /opt | GitHub clone `core_v0.17.0` + pip install |
| NCCL 加载 | GIB cp + 禁用 RDMA 库 | GIB `LD_PRELOAD` + 保留 RDMA 库 |
| TransformerEngine | `--transformer-impl transformer_engine` | 不使用 TE |
| Benchmark 层数 | 48 层全模型 | 12 层（MFU 归一化等价） |
| FSDP | DTensor bug 崩溃 | 配置可接受但 EP=32 死锁 |

> **模型配置踩坑**：之前 `--ffn-hidden-size 12288` 导致 Megatron 在每层同时创建 dense FFN（12288）和 128 个 expert FFN（768），总参数变成 61B 而不是 30B。正确值应该从 HuggingFace `config.json` 取：`hidden_size=2048, ffn_hidden_size=6144, moe_ffn_hidden_size=768`。PAI-Megatron-Patch 的 `run_mcore_qwen3.sh` 中 `A3B` 预设已经是正确值。

### 共享存储：Lustre 安装（Rocky 9 ARM64 / CIQ 内核）

多节点训练需要共享存储放数据集（Megatron 的 dataset index cache 只在 rank 0 构建，其他节点需要读同一路径）。

#### 创建 Lustre 实例

```bash
# 1. 创建 PSA IP 范围（如果 VPC 还没有）
gcloud compute addresses create lustre-psa \
  --project=$PROJECT --global \
  --purpose=VPC_PEERING \
  --addresses=10.200.0.0 --prefix-length=16 \
  --network=$VPC_NAME

# 2. 创建或更新 PSA peering
gcloud services vpc-peerings update \
  --project=$PROJECT --network=$VPC_NAME \
  --ranges=lustre-psa \
  --service=servicenetworking.googleapis.com --force

# 3. 创建 Lustre（最小 36000 GiB ≈ 35 TiB）
gcloud lustre instances create $LUSTRE_NAME \
  --project=$PROJECT --location=$ZONE \
  --capacity-gib=36000 \
  --filesystem=lustrefs \
  --network=$VPC_NAME \
  --per-unit-storage-throughput=250
```

#### 安装 Lustre 2.14 客户端（Rocky 9 ARM64 踩坑）

A4X 节点是 ARM64（Grace CPU），Google 官方 Lustre client repo 没有 aarch64 包。需要用 DDN 的 DKMS 包从源码编译，但有两个坑：

1. **OFED 冲突**：节点预装了 OFED（Mellanox RDMA），Lustre configure 检测到 OFED 但找不到 devel 包就报错。需要 `--with-o2ib=no`（Lustre 走 TCP 不需要 RDMA）。
2. **DKMS ko2iblnd**：`dkms.conf` 里列了 `ko2iblnd` 模块（RDMA 驱动），但 `--with-o2ib=no` 跳过了编译，DKMS 找不到 .ko 文件就报错。必须从 `dkms.conf` 删掉 ko2iblnd 的**三行**（NAME + LOCATION + DEST），不是两行。

```bash
# 在每台 worker 上执行
# Step 1: 安装依赖
dnf install -y dkms
dnf config-manager --set-enabled crb
dnf install -y libyaml-devel json-c-devel
dnf install -y kernel-devel-$(uname -r)

# Step 2: 添加 DDN Lustre repo + 安装 DKMS 包
cat > /etc/yum.repos.d/lustre-client.repo << 'EOF'
[lustre-client]
name=GCP Lustre Client
baseurl=https://us-yum.pkg.dev/projects/lustre-client-binaries/lustre-client-rocky-9
enabled=1
gpgcheck=0
repo_gpgcheck=0
EOF
dnf install -y lustre-client-dkms-2.14.0_ddn256-1.el9 lustre-client-2.14.0_ddn256-1.el9

# Step 3: Patch dkms.conf — 删除 ko2iblnd 的 3 行（NAME + LOCATION + DEST）
# 原文件中这 3 行在 ksocklnd 之后（约 line 31-33）：
#   BUILT_MODULE_NAME[...]="ko2iblnd"
#   BUILT_MODULE_LOCATION[...]="lnet/klnds/o2iblnd/"
#   DEST_MODULE_LOCATION[...]="/extra/lnet/"
sed -i '31,33d' /usr/src/lustre-client-2.14.0_ddn256/dkms.conf

# Step 4: Patch configure — 禁用 o2ib
sed -i 's|./configure |./configure --with-o2ib=no |' \
  /usr/src/lustre-client-2.14.0_ddn256/lustre-dkms_pre-build.sh

# Step 5: DKMS 编译 + 安装
dkms remove lustre-client/2.14.0_ddn256 --all 2>/dev/null || true
dkms add lustre-client/2.14.0_ddn256
dkms build lustre-client/2.14.0_ddn256 -k $(uname -r)
dkms install lustre-client/2.14.0_ddn256 -k $(uname -r)

# Step 6: 加载模块 + 挂载
modprobe lnet && modprobe lustre
lnetctl lnet configure
mkdir -p /mnt/lustre
mount -t lustre <LUSTRE_IP>@tcp:/lustrefs /mnt/lustre
```

> **关键提醒**：`dkms.conf` 中 ko2iblnd 的 DEST_MODULE_LOCATION 行容易漏删（只删 NAME + LOCATION 两行会导致 `No 'BUILT_MODULE_NAME' directive specified for record #N` 错误）。必须删 3 行。

#### 训练 Pod 挂载 Lustre

宿主机挂载 Lustre 后，Pod 通过 hostPath 访问：

```yaml
volumeMounts:
- { name: lustre, mountPath: /mnt/lustre }
volumes:
- { name: lustre, hostPath: { path: /mnt/lustre, type: Directory } }
```

数据集放在 `/mnt/lustre/qwen-datasets/`，所有节点共享读写。

### 显存问题记录（2026-06-29）

**EP=4 单节点 OOM 排查**：

megatron-ngc 镜像（`tev2.15-mgcore_r0.16.0-pt26.05-py3-v2`）在 GB200 上运行 128 Expert MoE 时 OOM：

1. **MBS=16, 48 层, hidden=4096**：model init 阶段 OOM（权重 + optimizer > 192 GB）
2. **MBS=4, 12 层, hidden=2048, 128 Expert**：model init 通过但 forward 时 activation OOM
3. **MBS=1, 12 层, hidden=2048, 128 Expert**：同上，MBS=1 也 OOM
4. **MBS=1, 12 层, hidden=2048, 64 Expert**：第一步 49.1 TFLOP/s 跑出来了，第二步 backward OOM

**根因**：Transformer Engine 在 GB200 上 fallback 到 **unfused attention**（`self.unfused_attention`），没有用 Flash Attention / Fused Attention。unfused attention 的 attention_probs 矩阵是 O(seq² × heads × batch) = 4096² × 32 × 1 = 2 GB per sample，显存暴涨。

**根因确认**：缺 `--bf16` 参数导致 attention QKV 输入为 FP32，FlashAttention 和 FusedAttention 都不支持 FP32 输入。TE debug 明确报告：
```
Disabling FlashAttention 2 for unsupported qkv_dtype = torch.float32
Disabling FusedAttention for unsupported qkv_dtype = torch.float32
Selected backend = UnfusedDotProductAttention
```

**修复**：加 `--bf16` 参数。FP8 训练必须配合 `--bf16`，让 non-FP8 层（包括 attention softmax、dropout）用 BF16 而不是 FP32。修复后 TE 正确选择 FusedAttention (sub-backend 1)。

**MBS 调优结果**（64 Expert, EP=4, 12 层, hidden=2048, seq=4096）：
- MBS=1: ~178 TFLOP/s/GPU（tensor core 利用率低）
- MBS=2: **~475 TFLOP/s/GPU**（+167%，计算效率大幅提升）
- MBS=4: OOM（Expert FFN linear_fc2 activation 超出显存）
- 最大可用 MBS=2，对应 MFU=10.6%（基于 GB200 FP8 峰值 4500 TFLOP/s）

> **MBS=1 vs MBS=16/32 的影响**：MBS=1 时 GPU 的 tensor core 利用率低——每个 GEMM 的 batch 维度太小，kernel launch overhead 占比高。调大 MBS 后 GEMM 的 batch 维度增大，计算效率显著提升。预期 MBS=16 相对 MBS=1 提升 30-50% TFLOP/s。旧 baseline（MBS=1）不作为正式参考，新测试的 MBS 充分利用显存后的结果才是有效 benchmark。

### 预期 Scaling 趋势分析

基于 DeepEP 的 scaling 数据（2n→4n→8n 下降 6%→9%），MoE 训练的 EP scaling 预期：

| 因素 | 影响 |
|---|---|
| all-to-all 通信量 | 随 EP 增加，每 GPU 需要跟更多目标通信，scatter 碎片化 |
| Expert 计算量 | 随 EP 增加，每 GPU 的 Expert 数减少，per-Expert batch 变小 |
| NVSwitch 带宽 | 域内全互联，理论上不随 EP 变化（NCCL 同域 scaling 证实） |
| Attention 计算 | 不受 EP 影响（全量复制，每 GPU 独立计算） |

如果 EP scaling 导致的下降在 5-10%/倍增 范围内，说明 NVSwitch 有效分摊了 all-to-all 开销。如果下降超过 20%，说明 Expert 计算粒度太小（每 GPU Expert 数太少），计算效率下降是主因。

---

## GB200 性能优化要点

| 配置 | 说明 |
|------|------|
| `--transformer-impl transformer_engine` | **必须**使用 TE。`local` 实现为未优化的纯 PyTorch 路径 |
| FP8 参数 | `--fp8-format hybrid --fp8-recipe delayed`。B200 FP8 Tensor Core 吞吐约为 BF16 的 2 倍 |
| GB200 环境变量 | `NVTE_FUSED_ATTN=1`、`NVTE_NORM_FWD_USE_CUDNN=1`、`NVTE_NORM_BWD_USE_CUDNN=1` |
| 移除 `--no-*-fusion` | 所有 `--no-masked-softmax-fusion` 等标志会禁用算子融合，严重影响性能 |
| `--sequence-parallel` | 沿序列维度分布 LayerNorm/Dropout 计算 |
| `--use-distributed-optimizer` | ZeRO-1 优化器状态分片 |
| `--overlap-grad-reduce` | 梯度 all-reduce 与反向传播重叠 |

## 调试记录（附录 B）

### Megatron-LM 参数变更

- `--mock-data`（不是 `--use-mock-data`）
- `--eval-interval` 和 `--eval-iters` 必须显式设置
- `--no-async-tensor-model-parallel-allreduce` 已移除
- `--lr-warmup-samples` 不能与 iteration-based 训练混用，需使用 `--lr-warmup-iters`

### 短主机名（Megatron 必需）

Megatron-LM 使用 Gloo 进行进程通信。如果 Pod hostname 过长，会触发 `File name too long` 错误。在 Pod spec 中设置 `hostname` 字段为短名称。

### 多节点 hostname 解析

Pod 设置自定义 `hostname` 时，NCCL bootstrap 和 Gloo 会调用 `gethostbyname()` 获取自身 IP。如果该 hostname 没有 DNS 记录，解析失败会导致**静默 hang**（`initialized tensor model parallel` 后不再有输出）。

**修复方案**：在 `/etc/hosts` 中添加 hostname→IP 映射，或配合 headless Service + `subdomain` 使 k8s DNS 自动注册。同时需设置 `NCCL_SOCKET_IFNAME=eth0` 和 `GLOO_SOCKET_IFNAME=eth0`。

### Performance Iteration Log (2026-07-02)

迭代对标 Megatron Bridge 官方 936 TFLOP/s（DGX-GB200, Qwen3 30B, MXFP8）。

| Round | Config | TFLOP/s | Delta | 备注 |
|---|---|---|---|---|
| Baseline | run_script.py, no CUDA graph | 89 | — | hybridep + fp8_attn + a2a_overlap |
| R1 | + numactl (错误绑定) | 23-87 | -74% | numactl 绑错 NUMA node，全进程挤一个核 |
| R2 | + full_iteration CUDA graph | OOM | — | 184 GiB 不够放 replay buffer |
| R3 | + TE CUDA graph | crash | — | expandable_segments 与 CUDA graph 冲突 |
| **R4** | **+ TE CUDA graph + NCCL_GRAPH_REGISTER=0** | **208** | **+134%** | **attn+moe_router+moe_preprocess scope** |
| R5 | + full_iteration CUDA graph (all opts) | OOM | — | 184 GiB 仍不够 |
| R8 | R4 + VBoost | 190 | -9% | VBoost 反而变慢 |
| R9 | cutedsl + TE CUDA graph | 320 peak → crash | — | Triton CPU tensor 错误第 4 步崩 |
| **R10** | **cutedsl fused grouped MLP (no CG)** | **284** | **+219%** | **稳定 20 步，0 alloc retry — A4X 最佳** |
| R11c | cutedsl + core_attn recompute + full CG | 151 peak → OOM | — | recompute 省 memory 但 CG capture 崩 |
| R11d | cutedsl + more recompute + full CG | 279 → crash | — | 0 retry 但 CG capture stream unjoined |
| R12 | cutedsl + TE CG + recompute | 137 → crash | — | Triton CPU tensor 兼容性问题 |
| 官方 | DGX-GB200 full_iteration CUDA graph | 936 | — | NVIDIA Performance Summary |

**R10 关键发现**：
- `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` 是最大的单项优化（89→284，+219%）
- `perf_plugins.py` 在 Slurm 模式自动设此环境变量，torchrun 不设就漏了
- cutedsl 无 CUDA Graph（284）比 TE CUDA Graph 无 cutedsl（208）还快
- cutedsl 把 HBM 峰值降到 175 GiB（0 alloc retry），因为 fused kernel 减少中间 buffer

**284 vs 936 差距根因（3.3×）**：
- **A4X HBM = 184 GiB vs DGX-GB200 HBM = 192 GiB**：8 GiB 差距导致 full_iteration CUDA Graph 无法在 A4X 上运行。CUDA Graph 消除了全部 host overhead（kernel launch + Python 调度），对 MoE 模型这种多小 kernel 的架构影响巨大
- **full_iteration CUDA Graph + MoE flex dispatcher 兼容性**：即使 recompute 省出内存（0 alloc retry），CUDA Graph capture 仍因 "stream has unjoined work" 崩溃。MoE token-dropless 模式的异步 dispatch 与 CUDA Graph 静态 capture 不兼容
- **cutedsl + TE CUDA Graph 组合崩溃**：Triton kernel 在 CUDA Graph replay 时产生 "Pointer argument cannot be accessed from Triton (cpu tensor?)" 错误，是已知的 Triton/CUDA Graph 兼容性问题

**torchrun 手动跑 Megatron Bridge 必须设的环境变量**（Slurm launcher 自动设，torchrun 漏）：
```bash
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1    # cutedsl fused grouped MLP
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8     # cuDNN 集群 overlap margin
export TORCH_NCCL_AVOID_RECORD_STREAMS=1    # 省 GPU 内存
export NCCL_GRAPH_REGISTER=0                # 解决 expandable_segments + CG 冲突
export CUDA_DEVICE_MAX_CONNECTIONS=1         # TP comm overlap 必需
```

**R4 关键发现**：
- `cuda_graph_impl=transformer_engine` + `cuda_graph_scope=attn,moe_router,moe_preprocess` 不 OOM
- `NCCL_GRAPH_REGISTER=0` 解决 expandable_segments 与 CUDA graph 的 assertion 冲突
- `TORCH_NCCL_AVOID_RECORD_STREAMS=1` 省 34 GiB inactive memory
- HBM 峰值 189 GiB（超出 184 GiB 物理容量，通过 alloc retry 机制运行）

**R4 vs 官方差距分析（208 vs 936 = 4.5×）**：
- TE CUDA graph 只 capture dense 部分，full_iteration capture 整个 MoE 层（A4X 内存不够）
- 可能需要 activation offloading 到 Grace CPU host memory 来释放 HBM
