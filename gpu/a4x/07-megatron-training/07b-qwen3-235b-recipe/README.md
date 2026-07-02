# Qwen3 235B-A22B MoE Training on GB200 NVL72 (A4X)

Megatron Bridge + NeMo 26.06 容器，Qwen3 235B-A22B MoE 预训练 benchmark。64 GPU（16 节点）。

**官方参考**：DGX-GB200 256 GPU MXFP8 → 7376 tok/s/GPU, **1092 TFLOP/s/GPU**（V2 config）。本文使用 64 GPU V1 config。

## 模型规格

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-235B-A22B |
| 总参数 | 235B |
| 每 token 激活参数 | 22B |
| Expert 数量 | 128 routed |
| TopK | 8 |
| 层数 | 94 |

## 前提条件

- 16 台 A4X worker（64 GPU），同一 NVL72 域（同 Placement Policy）
- k8s 1.34+ 集群 + GPU Stack（device-plugin + DRA + DRANET + ComputeDomain）
- Worker 镜像：`chrisya-a4x-worker-v3`

## 与 30B 的关键差异

| 维度 | 30B (07a) | 235B (本文) |
|---|---|---|
| GPU 数 | 8 (2 节点) | 64 (16 节点) |
| PP | 1 | **8** |
| EP | 8 | 8 |
| TP | 1 | 1 |
| MBS | 4 | 1 (默认) |
| GBS | 512 | 1024 |
| CUDA Graph | full_iteration | **transformer_engine** |
| cutedsl | Yes | No (V1 config) |

> **为什么 PP=8**：235B 模型 94 层 + 128 expert，单卡放不下。PP=8 把模型切成 8 个 pipeline stage，每个 stage ~12 层。
>
> **为什么 CUDA Graph 降级**：V1 用 TE CUDA graph（只 capture dense 部分），因为 PP>1 时 full_iteration CUDA graph 内存开销超过 10 GiB（官方文档原文）。V2（256 GPU）才用 full_iteration。

## Step 1: 创建 16 节点集群

```bash
# 用 v3 镜像创建 16 台 worker（同 Placement Policy 保证同域）
for i in $(seq 0 15); do
  gcloud compute instances create chrisya-a4x-d2-w${i} \
    --project=$PROJECT --zone=$ZONE \
    --machine-type=a4x-highgpu-4g \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific --reservation=$RESERVATION \
    --maintenance-policy=TERMINATE \
    --resource-policies=$PLACEMENT_POLICY \
    --image=chrisya-a4x-worker-v3 \
    --image-project=$PROJECT \
    --boot-disk-size=500GB --boot-disk-type=hyperdisk-balanced \
    --network-interface=nic-type=GVNIC,network=$GVNIC_NET,subnet=$GVNIC_SUB \
    --network-interface=nic-type=GVNIC,network=$GVNIC_NET_1,subnet=$GVNIC_SUB_1,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_0,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_1,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_2,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_3,no-address \
    --metadata="ssh-keys=$USER:$(cat ~/.ssh/google_compute_engine.pub)" \
    --scopes=cloud-platform &
done
wait
```

v3 镜像启动即用（~6.5 分钟），无需切内核。启动后 kubeadm join + GPU label + ComputeDomain label。

## Step 2: 部署 NeMo 26.06 训练 Pod

每个 worker 一个 Pod，共 16 个。YAML 结构同 30B recipe（见 [07a](../07a-qwen3-30b-recipe/)），改 nodeSelector 和 pod name。

## Step 3: 环境变量

与 30B 完全相同：

```bash
source /usr/local/gib/scripts/set_nccl_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NCCL_CTA_POLICY=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

## Step 4: 启动训练

```bash
cd /opt/Megatron-Bridge/scripts/performance

torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen \
    -mr qwen3_235b_a22b \
    --task pretrain \
    -g gb200 \
    -c fp8_mx \
    -ng 64 \
    --data mock \
    --max_steps 20 \
    --log_dir /tmp/nemo-results \
    -wde bench \
    -wdj qwen3_235b
```

### Recipe 自动加载的配置（V1, 64 GPU）

| 配置 | 值 | 说明 |
|---|---|---|
| PP | 8 | 94 层切 8 段 |
| EP | 8 | Expert Parallelism |
| TP | 1 | 无 Tensor 并行 |
| ETP | 1 | 无 Expert Tensor 并行 |
| GBS | 1024 | Global Batch Size |
| seq_length | 4096 | 序列长度 |
| cuda_graph_impl | transformer_engine | TE scoped CUDA Graph |
| cuda_graph_scope | attn, moe_router, moe_preprocess | Dense 模块 capture |
| moe_flex_dispatcher_backend | hybridep | NVL72 优化 |

### 并行度计算

- 总 GPU: 64
- PP=8: 模型切 8 段
- 每 PP stage 内 GPU 数: 64/8 = 8
- EP=8: 8 GPU 做 expert 分发
- DP: 64/(TP×PP) = 64/8 = 8, EP=8 → DP_effective=1
- GA: GBS/(MBS×DP_effective×PP_num_micro_batches) → 由 recipe 计算

## 注意事项

1. **16 节点必须在同一 NVL72 域**：用同一 Placement Policy 创建。A4X NVL72 域最多 18 节点，16 节点在范围内
2. **PP bubble**：PP=8 有 pipeline bubble 开销。V1 config 没有开 VPP（Virtual Pipeline Parallelism）。V2 开了 VP=3 减少 bubble
3. **内存**：235B 比 30B 大 8 倍，PP=8 分担后每卡 ~30B 参数。V1 用 TE CUDA graph（内存友好），不需要 recompute
4. **CUDA Graph 模式差异**：V1 用 `transformer_engine`（轻量），V2 用 `full_iteration`（需要更多内存但性能更好）
5. **config_variant**：`run_script.py` 默认加载 V2 config。如果 64 GPU 跑，需确认是否自动降级到 V1。可能需要 `--config_variant v1` 或 `-cv v1`
