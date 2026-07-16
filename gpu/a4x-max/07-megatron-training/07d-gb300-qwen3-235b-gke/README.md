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
| R6 | 2 | 32 | TE (attn,moe_router,moe_preprocess) | PP=2 MBS=2, NeMo 26.06 | **938** | **5.17s** |

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
>
> **R6 注**: PP=2 EP=32 配置 TFLOP/s 从 1007 降到 938 (-6.8%)，因 PP bubble 损耗。PP=2 的意义在于探索 full_iteration graph 路径（PP=1 OOM），但最终 full graph 在 PP>1 也不可行（见下方调查结论）。

### full_iteration CUDA Graph 深度调查 (2026-07-15~16)

#### 背景

NVIDIA 官方 Megatron Bridge Performance Summary: GB300 256GPU V2 (full_iteration graph) = **1335 TFLOP/s**。我们 TE scoped graph 最高 1054 TFLOP/s (79%)。差距 21% 来自 full_iteration graph 未能启用。GB200 上 PP=2 + full graph 已验证可行 (1124 TFLOP/s)，但 GB300 上同样配置 crash。

#### 实测记录

| 尝试 | PP | VP | EP | 卡数 | 结果 | 错误 |
|------|----|----|----|----|------|------|
| A | 1 | - | 64 | 64 | **OOM** | 276GB 不够: 94 layers + graph capture buffer |
| B | 2 | 1 | 32 | 64 | **crash** | `cudaErrorStreamCaptureUnsupported` (torch.cuda.synchronize) |
| C | 2 | 12 | 32 | 64 | **crash** | `cudaErrorStreamCaptureUnjoined` (NCCL stream 未 join) |
| D | 4 | 12 | 16 | 64 | **OOM** | warmup 3 步 OK (915 TFLOP/s), 第 4 步 graph capture 时 OOM |
| E | 4 | 12 | 32 | 256 | **hang** | `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32` 应为 8, HybridEP 通信挂死 |
| F (奚老师) | 2 | 8 | 32 | 256 | **crash** | 同 C: `cudaErrorStreamCaptureUnjoined` |

> 注：尝试 D 的前 3 步 (915 TFLOP/s) 不是 graph capture 的成功——`cuda_graph_warmup_steps` 决定前 N 步正常执行不做 capture，graph capture 发生在第 N+1 步。

#### 源码分析: 为什么 GB300 上 full graph capture 失败

**CUDA graph capture 的铁律**: 整个 capture 期间所有 GPU 操作必须在被追踪的 stream DAG 上。三个禁区：(1) host-device 同步 (`cuda.synchronize()`), (2) 未闭合的 stream fork, (3) 运行时动态分支。

**代码链路** (`full_cuda_graph.py` → `schedules.py` → `p2p_communication.py`):

1. `FullCudaGraphWrapper.__call__` (line 217): 在 `torch.cuda.graph(capture_stream)` 上下文内调 `forward_backward_func`
2. `forward_backward_pipelining_with_interleaving`: 用 `overlap_p2p_comm=True` 调 `send_forward_recv_forward`
3. `P2PCommunicator._communicate`: 调 `_batched_p2p_ops` → `torch.distributed.batch_isend_irecv`
4. **NCCL 内部**: `batch_isend_irecv` 底层创建 NCCL 通信 stream 执行 isend/irecv

**尝试 B 的根因** (VP=1, `cudaErrorStreamCaptureUnsupported`):

VP=1 走非 interleaved schedule，p2p 通信后经过 `_communicate` 的 `batch_p2p_sync=True` 路径:
```python
# p2p_communication.py _communicate()
if config.batch_p2p_comm and config.batch_p2p_sync:
    torch.cuda.synchronize()  # ← 违反铁律 (1): host-device 同步
```

**尝试 C/F 的根因** (VP>1, `cudaErrorStreamCaptureUnjoined`):

VP>1 走 interleaved schedule，`overlap_p2p_comm=True` 时不触发 `synchronize()`。但 `batch_isend_irecv` 内部 NCCL 创建的通信 stream **没有被注册到 capture DAG**。NCCL 有 Graph Registration 机制 (`NCCL_GRAPH_REGISTER`)，负责把 NCCL 内部 stream 注册到 CUDA graph。如果 registration 失败或被禁用，这些 stream 在 `capture_end()` 时被检测为 "unjoined work"。

**为什么 GB200 上同样配置能跑**: 可能的差异点:
- NCCL 版本: GB200 用的 NCCL 版本的 graph registration 实现与 GB300 的 NCCL 2.30.4 不同
- 网络插件: GB200 用标准 NCCL RDMA (CX-7 VF), GB300 用 GIB 插件 (CX-8 PF)。GIB 作为 NCCL net plugin 可能改变了内部 stream 创建行为
- DOCA 交互: GB300 的 DOCA OFED userspace 层可能影响 NCCL 的 CUDA stream 管理

#### 待验证项 (等环境恢复后)

1. **`NCCL_GRAPH_REGISTER=1`**: 显式开启 NCCL graph registration，看是否解决 `StreamCaptureUnjoined`
2. **`batch_p2p_sync=False`**: 确认 Bridge 0.5.0 V2 config 是否默认关闭了这个。如果没关，手动关闭后重试 PP=2 VP=12
3. **对比 NCCL 版本**: 查 GB200 成功时用的 NCCL 版本，与 GB300 的 2.30.4 对比 graph capture 相关 changelog
4. **GIB 兼容性**: 测试不加载 GIB 插件时 (纯 NCCL socket/IB transport) full graph 能否 capture 成功。如果能，则确认是 GIB 插件的 stream 管理问题
5. **PP=4 EP=32 256GPU**: 修正 `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8` 后重试，确认是否还有 `Unjoined` 错误
6. **`overlap_p2p_comm_warmup_flush`**: 这是 schedules.py 中的新特性，改变了 warmup 阶段的 p2p prefetch 模式。确认 V2 config 是否启用此选项，以及它对 graph capture stream DAG 的影响
7. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: 奚老师第二次尝试加了此配置。确认是否影响 graph capture 的内存分配行为

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
2. **full_iteration CUDA Graph** — GB300 GIB+NCCL 2.30.4 环境下 graph capture 失败 (NCCL stream unjoined)，GB200 同配置可行。疑似 GIB net plugin 的 stream 注册兼容性问题，待验证（详见上方深度调查）
3. **HybridEP flex_dispatcher** — GPU 名称 "NVIDIA GB300" 不匹配检测逻辑
4. **GB300 PP=1 EP=64** — 官方 recipe 配置，但 NCCL 子组创建 hang (原因待查)

---

*基于 [AI-Hypercomputer/gpu-recipes](https://github.com/AI-Hypercomputer/gpu-recipes) + NVIDIA Megatron Bridge 文档 + Megatron-LM 源码分析 · 2026-07-16*
