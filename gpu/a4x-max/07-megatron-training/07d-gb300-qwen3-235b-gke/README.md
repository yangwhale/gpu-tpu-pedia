# Qwen3 235B-A22B Training on GB300 NVL72 (A4X Max) — GKE

GB300 (A4X Max) GKE 集群上的 Qwen3 235B 预训练 benchmark。基于 GCP 官方 GPU Recipe + Megatron Bridge。

**GCP GPU Recipe 仓库**: [AI-Hypercomputer/gpu-recipes](https://github.com/AI-Hypercomputer/gpu-recipes/tree/main/training/a4x-max/qwen3-235b-a22b/megatron-bridge-gke/nemo2602)

**NVIDIA Megatron Bridge Performance Summary**: [docs.nvidia.com](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)

## GB300 vs GB200 关键差异

| 维度 | GB200 (A4X) | GB300 (A4X Max) |
|------|-------------|-----------------|
| GPU 型号参数 | `--gpu gb200` | `--gpu gb300` |
| PP (64 GPU) | 8 | **1** (288GB HBM 放得下) |
| EP (64 GPU) | 8 | **64** |
| VPP (64 GPU) | 3 | 无 (PP=1 不需要) |
| MBS | 1 | **2** |
| NeMo 容器 | 25.11.01 | **26.02** |
| GIB 加载方式 | sidecar init container | **apt install nccl-gib-plugins** |
| hostNetwork | true | **false** |
| DOCA-OFED | 不需要 | **必须 apt 安装 userspace** |
| NCCL 版本 | 容器自带 | **apt 升级到 2.28.9** |
| Megatron-Bridge | commit 7695d4a | **r0.3.0** (git clone) |

## 前提条件

- GKE 集群 `gb300-gke-test`，1.36.0+
- 至少 1 个 subblock 16 节点 (64 GPU) 或 4 个 subblock 64 节点 (256 GPU)
- Helm 3 安装
- Kueue + TAS (Topology-Aware Scheduling) 配置好
- HuggingFace Token (Qwen3 tokenizer 需要)

## 测试方案

### 方案 A: V1 Config — GCP GPU Recipe 原版 (先跑通)

直接使用 GCP 官方 GPU Recipe 的 Helm chart 部署。V1 config 使用 TE scoped CUDA graph，是保守但稳定的配置。

#### 配置参数

| 参数 | 64 GPU (16 节点) | 128 GPU (32 节点) | 256 GPU (64 节点) |
|------|-----------------|------------------|-----------------|
| PP | 1 | 1 | 1 |
| EP | 64 | 64 | 64 |
| TP | 1 | 1 | 1 |
| ETP | 1 | 1 | 1 |
| MBS | 2 | 2 | 2 |
| GBS | 1024 | 1024 | 2048 |
| 精度 | fp8_mx | fp8_mx | fp8_mx |
| CUDA Graph | transformer_engine | transformer_engine | transformer_engine |
| CUDA Graph Scope | moe_router, moe_preprocess | moe_router, moe_preprocess | moe_router, moe_preprocess |
| config_variant | v1 | v1 | v1 |

#### 步骤

```bash
# 1. Clone GPU Recipe 仓库
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=$(git rev-parse --show-toplevel)

# 2. 选择规模 (64 / 128 / 256 GPU)
export RECIPE_ROOT=$REPO_ROOT/training/a4x-max/qwen3-235b-a22b/megatron-bridge-gke/nemo2602/64gpus-fp8mx-seq4096-gbs1024/recipe
cd $RECIPE_ROOT

# 3. 设置环境变量
export PROJECT_ID=tencent-gcp-taiji-poc
export CLUSTER_REGION=us-central1
export CLUSTER_NAME=gb300-gke-test
export GCS_BUCKET=chrisya-gb300-logs     # 日志桶
export KUEUE_NAME=a4x_max               # Kueue local queue 名
export HF_TOKEN=<你的 HuggingFace Token>

# 4. 获取 credentials
gcloud config set project $PROJECT_ID
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION

# 5. Helm 部署
export WORKLOAD_NAME=$USER-qwen3-235b-64gpu-v1
helm install $WORKLOAD_NAME . -f values.yaml \
  --set-file workload_launcher=launcher.sh \
  --set workload.image=nvcr.io/nvidia/nemo:26.02 \
  --set workload.hfToken=$HF_TOKEN \
  --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
  --set volumes.gcsMounts[0].mountPath=/job-logs \
  --set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
  --set queue=${KUEUE_NAME}

# 6. 监控
kubectl get pods | grep $WORKLOAD_NAME
kubectl logs ${WORKLOAD_NAME}-workload-0-0-<hash>

# 7. 清理
helm uninstall $WORKLOAD_NAME
```

#### Pod 启动流程 (launcher 自动执行)

Helm chart 的 workload-job.yaml 模板在 Pod 启动时自动执行以下步骤：

1. **apt install DOCA-OFED 3.1.0 userspace** — CX-8 SuperNIC RDMA 依赖
2. **apt upgrade libnccl2** — 升级到兼容版本
3. **apt install nccl-gib-plugins** — 从 GCP apt repo 安装 GIB 网络插件
4. **清理 libibverbs 冲突** — `rm -rf /opt/rdma-core/build/lib/libibverbs*`
5. **设置 NCCL 环境**:
   - `NCCL_CONF_FILE=/usr/local/gib/scripts/nccl.conf`
   - `NCCL_NVLS_ENABLE=0`
   - `NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH`
6. **git clone Megatron-Bridge r0.3.0** + submodule init
7. **patch distributed timeout** 到 10 分钟
8. **numactl 绑定** CPU NUMA node
9. **torchrun 启动** `custom_setup_experiment.py`

#### 为什么之前失败

之前 5 轮手动部署全部 NCCL timeout，根因是 NeMo 容器缺少 3 个关键依赖：

| 缺失组件 | 影响 | 官方做法 |
|---------|------|---------|
| DOCA-OFED userspace | CX-8 RDMA 无法初始化 | apt install doca-ofed-userspace |
| 正确版本 NCCL | NCCL-GIB 版本不匹配 | apt upgrade libnccl2 |
| GIB NCCL 插件 | NCCL 不知道 CX-8 拓扑 | apt install nccl-gib-plugins |

NCCL benchmark 能跑通是因为 GIB 诊断镜像 (`nccl-plugin-gib-diagnostic-arm64:v1.1.2`) 自带了所有依赖。NeMo 容器没有这些。

### 方案 B: V2 Config — Full Iteration Graph (性能优化)

在 V1 跑通后，切换到 V2 config 以对标 NVIDIA 官方 benchmark (1335 TFLOP/s)。

#### V1 vs V2 差异

| 配置项 | V1 (方案 A) | V2 (方案 B) |
|--------|-----------|-----------|
| cuda_graph_impl | transformer_engine (TE scoped) | **full_iteration** (整个 step) |
| moe_paged_stash | False | **True** |
| moe_expert_rank_capacity_factor | — | 1.5 |
| config_variant 参数 | `-cv v1` | **`-cv v2`** |

#### 步骤 (在方案 A 基础上修改)

在官方 Helm chart 的 launcher.sh 基础上，修改 `custom_setup_experiment.py` 的参数：

```bash
# 修改 launcher.sh 中的 worker_command，替换以下参数：
--config_variant v2       # 切换到 V2 recipe
# V2 recipe 自动启用：
#   cuda_graph_impl = full_iteration
#   moe_paged_stash = True
#   moe_expert_rank_capacity_factor = 1.5
```

或者直接在 Helm install 时覆盖：

```bash
export WORKLOAD_NAME=$USER-qwen3-235b-64gpu-v2
# 需要修改 launcher.sh 中的 --config_variant 为 v2
sed 's/--cuda_graph_impl transformer_engine/--cuda_graph_impl full_iteration/' launcher.sh > launcher-v2.sh
# 同时移除 --cuda_graph_scope 参数 (full_iteration 不需要指定 scope)

helm install $WORKLOAD_NAME . -f values.yaml \
  --set-file workload_launcher=launcher-v2.sh \
  --set workload.image=nvcr.io/nvidia/nemo:26.02 \
  --set workload.hfToken=$HF_TOKEN \
  --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
  --set volumes.gcsMounts[0].mountPath=/job-logs \
  --set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
  --set queue=${KUEUE_NAME}
```

> **注意**: V2 需要 Megatron Bridge 的 sync-free kernel + paged stash 技术支持。`custom_setup_experiment.py` 走的是 `run_script.py` → Bridge 代码路径，这些技术自动启用。如果用 `pretrain_gpt.py` (raw Megatron-LM) 则没有这些优化，full_iteration graph 会 crash。

#### GB300 V2 的额外注意事项

1. **flex_dispatcher_backend GPU 检测 bug**: NeMo 26.02 的 `flex_dispatcher_backend.py` 检查 GPU 名称是否以 "NVIDIA B200" 或 "NVIDIA B300" 开头。GB300 报告为 "NVIDIA GB300"，不匹配。需要 patch：
   ```bash
   sed -i 's/device_properties.name.startswith(("NVIDIA B200", "NVIDIA B300"))/device_properties.name.startswith(("NVIDIA B200", "NVIDIA B300", "NVIDIA GB"))/' \
     /opt/Megatron-Bridge/src/megatron/bridge/training/flex_dispatcher_backend.py
   ```

2. **NCCL_NVLS_ENABLE=0**: 235B 模型 HBM 占用大，NVLS multicast buffer OOM。官方 launcher 已设置。

3. **HF_TOKEN 必须**: Qwen3 tokenizer 需要 HuggingFace 认证。

## 性能参考

### NVIDIA 官方 (Megatron Bridge Performance Summary)

| 系统 | GPU 数 | 精度 | Config | tok/s/GPU | TFLOP/s/GPU |
|------|--------|------|--------|-----------|-------------|
| DGX-GB300 | 256 | MXFP8 | V2 (full graph) | **9,015** | **1,335** |
| DGX-GB200 | 256 | MXFP8 | V2 (full graph) | 7,376 | 1,092 |

> GB300 比 GB200 高 22%。数据来自 [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)。

### GCP GPU Recipe (V1 TE scoped graph)

官方 GPU Recipe 仓库不包含性能数据 ("confidential benchmark report")。V1 TE scoped graph 预期比 V2 低约 20%（基于 GB200 实测：V1 930 vs V2 1124）。

### GB200 (A4X) 实测对比

| Config | GPU 数 | Graph | TFLOP/s/GPU |
|--------|--------|-------|-------------|
| V1 PP=8 EP=8 单域 | 64 | TE scoped | 930 |
| V2 PP=2 EP=32 跨域 | 64 | full_iteration | **1124** |

> 在 GB200 上 V2 比 V1 高 21%。GB300 预期差距类似。

## 集群要求

### Kueue + TAS 配置

GPU Recipe 依赖 Kueue TAS 做拓扑感知调度。如果集群没有 Kueue：

```bash
# 安装 Kueue
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.12.4/manifests.yaml

# 创建 ResourceFlavor + ClusterQueue + LocalQueue
# 参考: https://github.com/GoogleCloudPlatform/cluster-toolkit/
```

或者不用 Kueue，在 values.yaml 中设 `queue: null`，手动用 nodeSelector 指定 subblock。

### 不使用 Helm 的备选方案

如果不想用 Helm chart，可以手动构建 Pod spec，关键是在容器启动脚本中包含上述 7 个步骤（DOCA-OFED + NCCL 升级 + GIB 安装 + libibverbs 清理 + 环境变量 + Megatron-Bridge clone + torchrun）。

## GB300 GKE 实测结果 (2026-07-15)

### Qwen3 235B 64 GPU 单域 (sb-0004, 16 节点)

| 轮次 | PP | EP | Graph | 优化 | 稳态 TFLOP/s | Step Time |
|------|----|----|-------|------|-------------|-----------|
| R1 | 8 | 8 | TE (moe_router,moe_preprocess) | 基线 | **314** | 30.7s |
| R2 | 8 | 8 | TE (+attn) | + attn scope | **420** | 23.0s |
| R3 | 1 | 64 | TE (attn,moe_router,moe_preprocess) | PP=1 消除 bubble | **940** | 10.3s |
| **R4** | **1** | **64** | **TE (attn,moe_router,moe_preprocess)** | **+内部 env vars +numactl** | **1007** | **9.6s** |

> **R3→R4 优化来自 Buganizer b/514757311** (Google 内部 GB300 测试追踪):
> - `CUDA_DEVICE_MAX_CONNECTIONS=32` (原 1, 允许更多并行 CUDA stream)
> - `TORCH_NCCL_HIGH_PRIORITY=1` (NCCL 线程优先级)
> - `TORCH_NCCL_AVOID_RECORD_STREAMS=1` (减少内存碎片)
> - `NVTE_*_LAYERNORM_SM_MARGIN=20` (原 16)
> - `numactl --cpunodebind=$((LOCAL_RANK/2)) --membind=$((LOCAL_RANK/2))` (CPU NUMA 绑定)

> **对标**:
> - GB200 V1 PP=8 EP=8 单域 = 930 → GB300 PP=1 EP=64 = **1007** (+8.3%, 超过 GB200!)
> - Google 内部 GB300 256GPU V1 = 963 → 我们 64GPU V1 = **1007** (超过内部团队 256 卡!)
> - NVIDIA 官方 GB300 256GPU V2 = 1335 → 我们 64GPU V1 = 1007 (75.4%, 差距来自 full graph + HybridEP)

### 解决 GB300 GKE Megatron 训练的关键步骤

1. **DOCA-OFED userspace** — `apt install doca-ofed-userspace` (CX-8 RDMA)
2. **NCCL 升级** — `apt install --only-upgrade libnccl2 libnccl-dev`
3. **GIB 插件** — `apt install nccl-gib-plugins` (从 GCP apt repo)
4. **libibverbs 清理** — `rm -rf /opt/rdma-core/build/lib/libibverbs*`
5. **HF_HUB_DISABLE_XET=1** — 禁用 HuggingFace XET 存储后端 (NeMo 26.02 不兼容)
6. **去掉 expandable_segments** — 与 Bridge r0.3.0 CUDA Graph assertion 冲突
7. **NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN** — 必须匹配 EP 值
8. **TCP 同步屏障** — 16 节点同步等待所有 pod 安装完成后再启动 torchrun
9. **Megatron-Bridge r0.3.0** — `git checkout 9c9dd848` + submodule init

### 已知限制

1. **DOCA-OFED 运行时安装 ~15 分钟** — 需要构建预装镜像消除此瓶颈
2. **full_iteration CUDA Graph** — NeMo 26.02 的 run_script.py 不支持 full_iteration 参数
3. **HybridEP flex_dispatcher** — GPU 名称 "NVIDIA GB300" 不匹配检测逻辑
4. **GB300 PP=1 EP=64** — 官方 recipe 配置，但 NCCL 子组创建 hang (原因待查)

---

*基于 [AI-Hypercomputer/gpu-recipes](https://github.com/AI-Hypercomputer/gpu-recipes) + NVIDIA Megatron Bridge 文档 · 2026-07-15*
