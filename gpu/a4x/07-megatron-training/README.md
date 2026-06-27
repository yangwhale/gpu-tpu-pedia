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

## 测试结果

| 场景 | 配置 | 实测结果 |
|------|------|----------|
| 单节点 4 GPU | Qwen3 30B-A3B MoE, EP=4, FP8+TE, mbs=1, seq=16K | **~356 TFLOP/s/GPU**（稳态 iter 3-50 平均） |
| 多节点 2x4 GPU | Qwen3 30B-A3B MoE, EP=8, FP8+TE, mbs=1, MNNVL | **~274 TFLOP/s/GPU**（稳态 iter 3-50 平均） |

**注**：MoE 模型的 TFLOP/s 计算包含所有 expert 的 FLOPs（128 experts x topk=8），因此绝对值与 dense 模型不可直接对比。多节点 EP=8 跨域 alltoall 通信开销导致吞吐下降约 23%。B200 FP8 峰值算力约 4500 TFLOP/s。

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
