# Qwen3-Coder-480B-A35B-Instruct FP8 Multi-host Inference on TPU v7x-16

> 端到端指南：在 **TPU v7x-16（2 hosts × 4 chips = 8 chips · 16 devices · 1.5 TB HBM）** 上跑单实例 vLLM 推理（TP=16）。
>
> **状态**（2026-04-25 实测）：基础架构 (LWS + Ray + 2 节点) **走通**，但 vLLM `tpu_inference` 在 **TP=16 device init 阶段触发上游 bug**（详见 §踩坑实录 #5），目前无法跑通完整推理。**需要等待 vllm-project/tpu-inference 团队修复**。
>
> 单机版（v7x-8 + TP=8）见同目录 [README.md](README.md) — 单机已完整验证生产可用。

---

## 🎯 当前状态总结

| 阶段 | 状态 | 详情 |
|------|------|------|
| LWS Helm install | ✅ 走通 | yaml apply v0.7.0 |
| Multi-host node pool 创建 | ✅ 走通 | 用 workload policy + 2x2x2 topology |
| Pod 调度到 2 节点 | ✅ 走通 | 每个 pod 4 chip TPU |
| Ray cluster 跨 pod 协调 | ✅ 走通 | 2 nodes 双向通信，识别 8 chips total |
| vLLM 启动 + 模型识别 | ✅ 走通 | tpu7x-16, TP=16, 2 nodes_with_device |
| **vLLM `init_device` (Ray actor)** | ❌ **上游 bug** | `AttributeError: d.coords` — ray actor process JAX 看 CPU 而非 TPU |
| 权重加载 / XLA 编译 | ⏳ 阻塞 | 因 init_device 失败无法到达 |
| 推理 / Benchmark | ⏳ 阻塞 | 同上 |

---

## 硬件与架构

| 项目 | 要求 |
|------|------|
| TPU | **v7x-16**（2x2x2 拓扑，8 chips = 16 devices, 跨 2 节点） |
| HBM | 总 **1.5 TB** (192 GB/chip × 8) |
| 主机内存 | ≥850 GB **per node** |
| 网络 | 节点间需高带宽 + 大 MTU 友好（GKE auto） |
| 存储 | Lustre 共享卷（推荐）— 已有 `/lustre/qwen3-coder-480b-fp8` |

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│  GKE Cluster (chrisya-v7x-v134)                        │
│  Node Pool: np-tpu7x-spot-mh (topology 2x2x2)          │
│                                                         │
│  ┌────────────────────────┐  ┌────────────────────────┐│
│  │ Node 1 (host-0)        │  │ Node 2 (host-1)        ││
│  │  4 chips · 768 GB HBM  │  │  4 chips · 768 GB HBM  ││
│  │                        │  │                        ││
│  │  Pod: vllm-mh-0        │  │  Pod: vllm-mh-0-1      ││
│  │   - Ray head 6379      │  │   - Ray worker         ││
│  │   - vLLM API 8000      │  │                        ││
│  │   - LWS_WORKER_INDEX=0 │  │   - LWS_WORKER_INDEX=1 ││
│  └─────┬──────────────────┘  └─────────┬──────────────┘│
│        │                                │               │
│        └────── Ray RPC (DCN) ───────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## 完整复现命令（基础架构能跑到 init_device 失败为止）

### Step 0: 集群前置（一次性）

#### 0a: 安装 LeaderWorkerSet (LWS)

```bash
LWS_VERSION="v0.7.0"
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/${LWS_VERSION}/manifests.yaml

# 验证
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io
kubectl get pods -n lws-system
```

### Step 1: 创建 multi-host node pool

#### 1a: workload policy（topology 2x2x2 = 8 chips）

```bash
gcloud compute resource-policies create workload-policy chrisya-tpu7x-spot-mh \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x2 \
  --region=us-central1 \
  --project=cloud-tpu-multipod-dev
```

> ⚠️ **拓扑选择踩坑**：v7x 的 `tpu7x-standard-4t` machine type **不兼容 `4x2x1`** — 必须用 `2x2x2`（同样 8 chips, 但布局是立方体）。

#### 1b: multi-host spot node pool（2 nodes）

```bash
gcloud container node-pools create np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 \
  --region=us-central1 \
  --project=cloud-tpu-multipod-dev \
  --machine-type=tpu7x-standard-4t \
  --tpu-topology=2x2x2 \
  --num-nodes=2 \
  --node-locations=us-central1-c \
  --disk-type=hyperdisk-balanced \
  --disk-size=500 \
  --spot \
  --max-pods-per-node=110 \
  --image-type=COS_CONTAINERD \
  --workload-metadata=GKE_METADATA \
  --placement-policy=chrisya-tpu7x-spot-mh
```

> 大概 **3-5 分钟**。Spot 可能首次失败需要重试。
> ⚠️ **Spot 容易被回收**：实测一次创建后约 5-10 分钟可能被 preempted，pool 进入 RECONCILING 状态自动重新分配 node。

#### 1c: 验证

```bash
kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology \
  | grep np-tpu7x-spot-mh
# 预期 2 个 node, topology=2x2x2
```

### Step 2: 部署 LWS + vLLM TP=16

#### 2a: Service（只暴露 leader pod）

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Service
metadata:
  name: vllm-mh-service
spec:
  type: ClusterIP
  selector:
    leaderworkerset.sigs.k8s.io/name: vllm-mh
    leaderworkerset.sigs.k8s.io/worker-index: "0"
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
EOF
```

#### 2b: LeaderWorkerSet（已应用所有 verified workaround）

```bash
kubectl apply -f - <<'EOF'
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm-mh
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: Default          # ⚠️ 不能用 RecreateGroupOnPodRestart, 任何重启会无限重建
    workerTemplate:
      metadata:
        labels:
          leaderworkerset.sigs.k8s.io/name: vllm-mh
      spec:
        hostname: vllm-mh
        serviceAccountName: default
        nodeSelector:
          cloud.google.com/gke-nodepool: np-tpu7x-spot-mh
          cloud.google.com/gke-tpu-accelerator: tpu7x
          cloud.google.com/gke-tpu-topology: 2x2x2
        tolerations:
        - { effect: NoSchedule, key: cloud.google.com/gke-spot, operator: Equal, value: "true" }
        - { effect: NoSchedule, key: google.com/tpu, operator: Exists }
        initContainers:
        - name: tpu-node-setup
          image: busybox
          command: ["/bin/sh", "-c"]
          args:
          - |
            sysctl -w vm.max_map_count=8388608
            if [ -f /sys/module/vfio_iommu_type1/parameters/dma_entry_limit ]; then
              echo 2000000 > /sys/module/vfio_iommu_type1/parameters/dma_entry_limit
            fi
          securityContext: { privileged: true }
        containers:
        - name: main
          image: vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626   # codelab 验证版本
          command: ["sh", "-c"]
          args:
          - |
            MY_TPU_IP=$(hostname -I | awk '{print $1}')
            echo "My TPU IP: $MY_TPU_IP, LWS_WORKER_INDEX=$LWS_WORKER_INDEX"
            LEADER_DNS="vllm-mh-0.vllm-mh"
            until getent hosts $LEADER_DNS; do
              echo "Waiting for leader DNS..."
              sleep 5
            done
            LEADER_IP=$(getent hosts $LEADER_DNS | awk '{print $1}')

            export JAX_PLATFORMS=''
            export PJRT_DEVICE=TPU
            export TPU_BACKEND_TYPE=jax
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
            export TPU_MULTIHOST_BACKEND=ray
            export JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT=300
            export VLLM_HOST_IP=$MY_TPU_IP
            export MODEL_IMPL_TYPE=vllm
            export HF_HOME=/lustre
            export HF_HUB_OFFLINE=1
            export SKIP_JAX_PRECOMPILE=1
            export VLLM_XLA_CHECK_RECOMPILATION=0
            export USE_MOE_EP_KERNEL=0
            export USE_BATCHED_RPA_KERNEL=0
            export VLLM_LOGGING_LEVEL=INFO

            if [ "$LWS_WORKER_INDEX" = "0" ]; then
              echo "=== Starting Ray Head (daemon mode, NOT --block) ==="
              ray start --head --port=6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}'
              sleep 20
              until ray status; do sleep 5; done

              echo "=== Starting vLLM API Server (TP=16) ==="
              vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
                --served-model-name=Qwen3-Coder-480B-FP8 \
                --tensor-parallel-size=16 \
                --distributed-executor-backend=ray \
                --max-model-len=10240 \
                --max-num-batched-tokens=8192 \
                --max-num-seqs=512 \
                --no-enable-prefix-caching \
                --kv-cache-dtype=fp8 \
                --gpu-memory-utilization=0.9 \
                --enable-expert-parallel \
                --host=0.0.0.0 --port=8000
            else
              echo "=== Starting Ray Worker (--block) ==="
              ray start --address=${LEADER_IP}:6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}' --block
            fi
          ports:
          - { containerPort: 8000 }
          - { containerPort: 6379 }
          resources:
            limits:   { google.com/tpu: "4", memory: "850Gi", cpu: "200" }
            requests: { google.com/tpu: "4", memory: "850Gi", cpu: "200" }
          volumeMounts:
          - { name: dshm,       mountPath: /dev/shm }
          - { name: lustre-vol, mountPath: /lustre }
          securityContext:
            privileged: true
            capabilities: { add: [IPC_LOCK] }
        volumes:
        - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 200Gi } }
        - { name: lustre-vol, persistentVolumeClaim: { claimName: lustre-pvc } }
EOF
```

#### 2c: 监控

```bash
# 等所有 pod Running
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh -w

# 看 Ray cluster 状态
kubectl exec vllm-mh-0 -c main -- ray status

# 看 vLLM 启动 log
kubectl logs vllm-mh-0 -c main -f
```

---

## 🐛 踩坑实录（实战经验，全部 verified）

### 坑 #1: TPU 拓扑 4x2x1 不兼容 tpu7x-standard-4t

**症状**: `gcloud container node-pools create` 报 `Accelerator topology: 4x2x1 is not compatible with the provided machine type: tpu7x-standard-4t`

**原因**: v7x 的 `tpu7x-standard-4t` machine type 仅支持特定 topology 集合，4x2x1 不在其中。

**修复**: 用 **`2x2x2`**（同样 8 chips 但是 cube 布局）。

### 坑 #2: `ray --block &` 在 sh -c args 内不生效

**症状**: Leader pod log 一直停在 `ray start --block` 输出，永远不进 `vllm serve`。

**原因**: K8s container `args:` 段内的 `&` 后台符号在某些 shell parse 下被吞掉，`ray --block` 真的 block 了主 shell。

**修复**: **Leader 用 daemon 模式（不加 `--block`）**，worker 用 `--block`：
```bash
# Leader
ray start --head --port=6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}'  # 无 --block, daemon 化
sleep 20
until ray status; do sleep 5; done
vllm serve ...

# Worker
ray start --address=${LEADER_IP}:6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}' --block  # block 保活
```

### 坑 #3: LWS `RecreateGroupOnPodRestart` 太激进

**症状**: pod 任何 restart 触发整个 group recreate, 永远不能稳定到 vllm 加载阶段。

**原因**: vLLM cold start 时偶尔有非致命错误，`RecreateGroupOnPodRestart` 把整个 group recycle，永远不能恢复。

**修复**: 改 `restartPolicy: Default`（标准 K8s 行为，container restart 不触发 group recreate）。

### 坑 #4: `--async-scheduling` 不兼容 ray executor

**症状**:
```
pydantic_core._pydantic_core.ValidationError: Value error, `ray` does not support async scheduling yet.
```

**原因**: vLLM 的 async scheduling 还没支持 ray distributed executor backend。

**修复**: **Multi-host (`--distributed-executor-backend=ray`) 配置不能用 `--async-scheduling`**。去掉这个 flag。

### 坑 #5 (BLOCKER): vLLM TP=16 init_device `AttributeError: d.coords`

**症状**:
```
File "/workspace/tpu_inference/tpu_inference/distributed/utils.py", line 125,
    in get_device_topology_order_id
    local_anchor = min(d.coords for d in local_devices)
AttributeError
```

**根因**: Ray actor process 在 init_device 时 `local_devices` 拿到的是 **CPU device**（没 `coords` 属性）, 而不是 TPU device。代码本身的 `if not all(hasattr(d, "coords") ...): logger.error(...)` 只 log 但不 fallback。

**深入分析**：JAX 在 ray actor process 内 init backend 时报错：
```
RuntimeError: Unable to initialize backend 'tpu':
  INVALID_ARGUMENT: TPU initialization failed:
  Invalid --deepsea_slice_builder_worker_addresses specified.
  Expected 2 worker addresses, got 1.
```

JAX 的 multi-host TPU init 只看到 1 个 worker（它自己），缺另一个 worker 的 address。这是 **`libtpu` 的 multi-host coordination 问题**。

**已尝试的所有 workaround**（都没解决）：

| 尝试 | 结果 |
|------|------|
| 加 `PJRT_DEVICE=TPU` env | ❌ 仍 AttributeError |
| 加 `TPU_BACKEND_TYPE=jax` env | ❌ 同上 |
| 加 `LD_LIBRARY_PATH=...:/usr/local/lib` | ❌ 同上 |
| 切换 image 到 `vllm/vllm-tpu:nightly` (latest) | ❌ 同上 |
| 切换 image 到 codelab 同款 `vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626` | ❌ 同上 |
| `JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT=300` | ❌ 不影响（不是 timeout 问题）|

**当前结论**: **TPU v7x 的 multi-host TPU 协调在 `libtpu` 层面有 bug**，导致 JAX 在 ray actor process 内拿不到完整的 worker address list，fallback 到 CPU。这是 **vllm-project/tpu-inference + libtpu 上游需要修复的问题**。

**对比 codelab**: 同样的 LWS + Ray 模式在 v6e (`ct6e-standard-4t` + topology `4x8`) 是 work 的（codelab 跑 Qwen 30B success）。**v7x multi-host 还需要 upstream patch**。

### 坑 #6: Spot multi-host pool 容易被 preempt

**症状**: pool 状态从 RUNNING 突然变 RECONCILING，几分钟后又回 RUNNING（不同 node ID）。

**原因**: GCE spot 容量浮动，整个 multi-host slice 被一起回收 + 重新分配。

**影响**: 已经启动的 pod 全部 reset，cold start 时间倍增。

**Workaround**: 用 reservation 替代 spot（生产环境）。开发用 spot 接受偶尔重启即可。

---

## 已知 v6e 工作的 codelab 同款步骤参考

如果要在 **v6e** 上跑 multi-host vLLM（验证可行），参考 [Google Codelabs: Deploy Multihost TPU vLLM Inferencing with Ray on GKE](https://codelabs.developers.google.com/next26/aiinfra-learning-pod/screen2-advanced-inferencing-part-1)。

主要差异：
- machine type: `ct6e-standard-4t`（v6e）
- topology: `4x8`（32 chips, 8 hosts）
- 模型: Qwen 30B（小很多）
- 更宽松的 TPU 兼容性

---

## 下一步建议

### 短期（客户角度）
1. **生产部署**：仍用单机 v7x-8 + TP=8（见 [README.md](README.md)），已验证完美工作
2. **大模型场景**：480B FP8 单机已能跑（768 GB HBM 富余），暂不需要 multi-host
3. **若一定要 multi-host**：暂时用 v6e（codelab 验证）或等待 v7x bug fix

### 长期（团队角度）
1. **跟踪 vllm-project/tpu-inference 的 libtpu multi-host fix**
2. **测试新版本 image** 时优先验证 `tpu_inference/distributed/utils.py:125` 是否仍报 AttributeError
3. **联系 Google TPU 团队** 升级 v7x multi-host runtime support

---

## 实测时间线（2026-04-25）

| 时间 | 事件 |
|------|------|
| 15:31 | 装 LWS via yaml |
| 15:32 | 创建 workload policy 4x2x1 失败（不兼容 v7x） |
| 15:35 | 创建 workload policy 2x2x2 成功 + node pool 创建 |
| 15:37 | LWS apply, 2 pods 调度成功 |
| 15:43 | 第一次失败：`ray --block &` 卡住 leader |
| 15:50 | 修复 leader daemon 模式后再启动 |
| 15:58 | 第二次失败：LWS RecreateGroupOnPodRestart 无限循环 |
| 16:05 | 改 restartPolicy=Default 后再启动 |
| 16:12 | 第三次失败：`--async-scheduling` 不兼容 ray |
| 16:25 | 去掉 async-scheduling, Ray cluster 双节点正确识别 8 chips |
| 16:31+ | **多次重启都卡在 init_device AttributeError**（坑 #5）|
| 16:33-17:11 | 试切换 image (latest→nightly→codelab 固定版本)、env vars (PJRT/LD_LIBRARY_PATH) — **均失败** |
| 17:12 | 决定文档化所有踩坑作为最终交付，等待上游 fix |

---

## 资源清理

```bash
# 删 LWS workload
kubectl delete lws vllm-mh
kubectl delete service vllm-mh-service

# 删 multi-host node pool
gcloud container node-pools delete np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev --quiet

# 删 workload policy
gcloud compute resource-policies delete chrisya-tpu7x-spot-mh \
  --region=us-central1 --project=cloud-tpu-multipod-dev --quiet
```

---

## 参考资料

| 资源 | 链接 |
|------|------|
| Google Codelabs (multihost vLLM + Ray, v6e 验证) | [link](https://codelabs.developers.google.com/next26/aiinfra-learning-pod/screen2-advanced-inferencing-part-1) |
| 上游 multihost benchmark 脚本 | [run_qwen3_coder_480b_1k_8k.sh](https://github.com/vllm-project/tpu-inference/blob/main/scripts/multihost/benchmarks/torchax/run_qwen3_coder_480b_1k_8k.sh) |
| 单机 README | [README.md](README.md) |
| LeaderWorkerSet docs | https://lws.sigs.k8s.io/ |
| `tpu_inference/distributed/utils.py:125` (bug 位置) | [link](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/distributed/utils.py#L125) |
