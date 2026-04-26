# Qwen3-Coder-480B-A35B-Instruct FP8 Multi-host Inference on TPU v7x-16

> 端到端指南：在 **TPU v7x-16（2 hosts × 4 chips = 8 chips · 16 devices · 1.5 TB HBM）** 上跑单实例 vLLM 推理（TP=16）。
>
> **状态**（2026-04-26 二次验证）：✅ **完整可复现**，但 **CRITICAL 结论**：单实例 multi-host TP=16 **output throughput 全场景比单机差**（-17% ~ -63%）。**不推荐生产用** — 客户应该跑 2 个独立的 v7x-8 单机实例（data parallel）。
>
> 本文档目的：(1) 提供完整可复现部署步骤；(2) 给出客观对比数据；(3) 解释为什么这种部署模式不适合 Qwen3-Coder-480B。
>
> 单机版（v7x-8 + TP=8）见同目录 [README.md](README.md) — **生产推荐**。

---

## 🎯 核心结论（要点）

### ⚠️ 单实例 TP=16 multi-host **不推荐**

| 场景 | 单机 v7x-8 (TP=8) | 多机 v7x-16 (TP=16) | 多机 vs 单机 |
|------|------------------:|-------------------:|------------:|
| 1K/1K c=1 output tok/s | **48** | 37.5 | ❌ -22% |
| 1K/1K c=4 output tok/s | **177** | 98.4 | ❌ -44% |
| 1K/1K c=16 output tok/s | **602** | 220.0 | ❌ -63% |
| 8K/1K c=4 output tok/s | **162** | 134.3 | ❌ -17% |
| 8K/1K c=16 output tok/s | **483** | 223.1 | ❌ -54% |
| **8K/1K c=4 TTFT (median)** | 1495 ms | **1188 ms** | ✅ -20% |

唯一胜出场景：**长 prompt 单次请求的 TTFT**（prefill 算力翻倍）。**所有 decode-bound 场景都更差**。

### 为什么慢？

1. **TP=16 跨 DCN**: decode 每个 token 都需要全 16 device all-reduce，跨节点 DCN 同步开销 >>> 翻倍算力的收益
2. **MoE expert 路由跨节点**: top-k expert 选择导致跨 host 的 all-to-all
3. **Single instance TP=16 不是甜区**: TP 切得越细，通信占比越高

### ✅ 正确生产建议

| 客户场景 | 推荐配置 | 理由 |
|---------|---------|------|
| 默认 / 大部分场景 | **2 × v7x-8 单机 (TP=8) Data Parallel** | 总 throughput ≈ 2× 单机 = ~1200 tok/s @ c=16，远胜 multi-host TP=16 |
| 长 prompt 高并发 + KV 大 | 单机 PD 分离 1P1D | TPOT 改善 11%，TTFT 牺牲不严重 |
| **特别需要单实例** + max-model-len ≥ 32K | Multi-host TP=16（牺牲 throughput 换 HBM）| 1.5 TB HBM 才能放下大 KV cache |

---

## 实测数据（5 组 benchmark · 二次复现验证）

### Multi-host TP=16 (v7x-16) — 本次实测

| 场景 | TTFT med | TTFT P99 | TPOT med | Output tok/s | Total tok/s |
|------|---------:|---------:|---------:|-------------:|------------:|
| 1K/1K c=1 | 97 ms | 98 ms | 26.6 ms | 37.5 | 75.0 |
| 1K/1K c=4 | 273 ms | 50.7 s* | 28.1 ms | 98.4 | 196.8 |
| 1K/1K c=16 | 4306 ms | 46.1 s* | 65.5 ms | 220.0 | 440.1 |
| 8K/1K c=4 | 1188 ms | 1671 ms | 28.6 ms | 134.3 | **1209** |
| 8K/1K c=16 | 1189 ms | 40.2 s* | 69.1 ms | 223.1 | **2008** |

\* P99 TTFT 偶发高 = 第一次遇到的 padding bucket 触发 XLA recompile。**Median 是稳态指标**。

### 与昨天首次跑通的数据对比（可重复性验证）

| 场景 | 指标 | 2026-04-25 首跑 | 2026-04-26 复现 | 差异 |
|------|------|---:|---:|---:|
| 1K/1K c=1 | output tok/s | 21.5 | 37.5 | +75% (首跑包含编译) |
| 1K/1K c=4 | output tok/s | 99.9 | 98.4 | -1.5% ✅ |
| 1K/1K c=16 | output tok/s | 222.8 | 220.0 | -1.3% ✅ |
| 8K/1K c=4 | output tok/s | 95.2 | 134.3 | +41% (首跑包含编译) |
| 8K/1K c=16 | output tok/s | 222.7 | 223.1 | +0.2% ✅ |

**结论**：高并发数据稳定可复现（差异 <2%）。低并发首跑因第一次请求触发 XLA 编译显著低估真实 throughput → **必须 warmup 后再 benchmark**。

---

## 硬件与架构

| 项目 | 要求 |
|------|------|
| TPU | **v7x-16**（2x2x2 拓扑，8 chips = 16 devices, 跨 2 节点） |
| HBM | 总 **1.5 TB** (192 GB/chip × 8) |
| 主机内存 | ≥850 GB **per node** |
| 网络 | DCN（节点间 8471/6379 端口） |
| 存储 | Lustre 共享卷（必须）— 已有 `/lustre/qwen3-coder-480b-fp8` |

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
│  │   - vLLM API 8000      │  │   - libtpu coord 8471  ││
│  │   - LWS_WORKER_INDEX=0 │  │   - LWS_WORKER_INDEX=1 ││
│  │   - TPU_WORKER_ID=0    │  │   - TPU_WORKER_ID=1    ││
│  └─────┬──────────────────┘  └─────────┬──────────────┘│
│        │                                │               │
│        ├─ Ray RPC (DCN, port 6379) ─────┤               │
│        └─ libtpu coord (port 8471) ─────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## 完整复现命令（新人照抄即可）

### Step 0: 集群前置（一次性）

```bash
# 安装 LeaderWorkerSet (LWS)
LWS_VERSION="v0.7.0"
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/${LWS_VERSION}/manifests.yaml

# 验证
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io
```

### Step 1: 创建 multi-host node pool

```bash
# 1a. workload policy (topology 2x2x2 = 8 chips)
gcloud compute resource-policies create workload-policy chrisya-tpu7x-spot-mh \
  --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2 \
  --region=us-central1 --project=cloud-tpu-multipod-dev

# 1b. multi-host spot node pool（2 nodes，~3-5 min）
gcloud container node-pools create np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev \
  --machine-type=tpu7x-standard-4t --tpu-topology=2x2x2 \
  --num-nodes=2 --node-locations=us-central1-c \
  --disk-type=hyperdisk-balanced --disk-size=500 --spot \
  --max-pods-per-node=110 --image-type=COS_CONTAINERD \
  --workload-metadata=GKE_METADATA \
  --placement-policy=chrisya-tpu7x-spot-mh

# 验证（应该看到 2 个 node, topology=2x2x2）
kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology \
  | grep np-tpu7x-spot-mh
```

> ⚠️ **拓扑注意**: v7x 的 `tpu7x-standard-4t` 必须用 `2x2x2`（8 chips, cube），**不能用 `4x2x1`** — gcloud 会拒绝。

> ⚠️ **Spot 注意**: 实测约 5-10 min 后可能被 preempted, pool 进入 RECONCILING 自动重新分配（通常 1-2 min）。

### Step 2: 部署 Service + LeaderWorkerSet

#### 2a. Service（只暴露 leader）

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

#### 2b. LeaderWorkerSet（**关键：手动注入 TPU multi-host env vars**）

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
    restartPolicy: Default            # 不能用 RecreateGroupOnPodRestart
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
          image: vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626
          command: ["sh", "-c"]
          args:
          - |
            MY_TPU_IP=$(hostname -I | awk '{print $1}')
            LEADER_DNS="vllm-mh-0.vllm-mh"
            WORKER_DNS="vllm-mh-0-1.vllm-mh"
            until getent hosts $LEADER_DNS; do sleep 5; done
            until getent hosts $WORKER_DNS; do sleep 5; done
            LEADER_IP=$(getent hosts $LEADER_DNS | awk '{print $1}')
            WORKER_IP=$(getent hosts $WORKER_DNS | awk '{print $1}')

            # === KEY FIX: 覆盖 GKE auto-injected single-host TPU env vars ===
            # 不加这一段会卡在 init_device 失败 (AttributeError: d.coords)
            export TPU_WORKER_HOSTNAMES="${LEADER_IP},${WORKER_IP}"
            export TPU_WORKER_ID=${LWS_WORKER_INDEX}
            export TPU_PROCESS_ADDRESSES="${LEADER_IP}:8471,${WORKER_IP}:8471"
            export TPU_PROCESS_PORT=8471
            export TPU_HOST_BOUNDS="1,1,2"
            export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
            export TPU_TOPOLOGY="2x2x2"
            export TPU_ACCELERATOR_TYPE="tpu7x-16"
            export TPU_SKIP_MDS_QUERY=true

            export JAX_PLATFORMS=''
            export PJRT_DEVICE=TPU
            export TPU_BACKEND_TYPE=jax
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
            export TPU_MULTIHOST_BACKEND=ray
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
              ray start --head --port=6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}'
              sleep 20
              until ray status; do sleep 5; done
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
              ray start --address=${LEADER_IP}:6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}' --block
            fi
          ports:
          - { containerPort: 8000 }
          - { containerPort: 6379 }
          - { containerPort: 8471 }
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

### Step 3: 监控启动到 ready

```bash
# 等所有 pod Running
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh -w

# 验证启动完成（出现 "Application startup complete"）
kubectl logs vllm-mh-0 -c main --follow | grep -E "Application startup complete|Init mesh|Loading weights took"
```

**实测启动时间分解**（2 次复现取平均）：

| 阶段 | 首次冷启动 | 后续重启（image cached）|
|------|-----------:|--------------------:|
| Pod scheduling + Ray cluster | ~70s（image pull）| ~10-20s |
| JAX mesh init | ~60s | ~60s |
| 权重加载（49 shards × ~10s）| **~480s = 8 min** | **~480s** |
| XLA compile + warmup | 140-244s | 140-244s |
| **总冷启动** | **~12 min** | **~13 min** |

> ⚠️ XLA cache 不持久化（容器重启就丢），所以"重启"也要重编译。如果 weight 加载时间是瓶颈，考虑 PD checkpoint 或 weight prefetch。

### Step 4: Smoke test

```bash
kubectl exec vllm-mh-0 -c main -- bash -c '
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen3-Coder-480B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍 TPU v7\"}],\"max_tokens\":100}"
'
```

### Step 5: Benchmark（**必须 warmup**）

```bash
kubectl exec vllm-mh-0 -c main -- bash -c '
# Warmup: 让 XLA 编译完所有 padding bucket
# 没有这步, c=1 场景因第一次请求触发编译, throughput 会被严重低估
for inp in 1024 8192; do
  for out in 64 1024; do
    vllm bench serve --backend vllm \
      --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
      --served-model-name Qwen3-Coder-480B-FP8 \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name random --random-input-len $inp --random-output-len $out \
      --num-prompts 2 --max-concurrency 1 --ignore-eos > /dev/null 2>&1
  done
done

# 正式 benchmark (warmup 后真实数据)
vllm bench serve --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --served-model-name Qwen3-Coder-480B-FP8 \
  --base-url http://localhost:8000 --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 \
  --num-prompts 32 --max-concurrency 16 --ignore-eos
'
```

> 💡 看 **Output tok/s**（真实生成速度），不是 Total tok/s（含 prefill input，会高估 5-10 倍）。

---

## 🐛 踩坑实录（全部 verified · 2 次复现）

### 坑 #1: TPU 拓扑 4x2x1 不兼容 tpu7x-standard-4t

**症状**: `Accelerator topology: 4x2x1 is not compatible with the provided machine type: tpu7x-standard-4t`

**修复**: 用 **`2x2x2`**（cube 布局，同样 8 chips）。

### 坑 #2: `ray --block &` 在 sh -c args 内不生效

**症状**: Leader pod log 一直停在 `ray start --block`。

**修复**: **Leader 用 daemon 模式（不加 `--block`）**，worker 用 `--block`。

### 坑 #3: LWS `RecreateGroupOnPodRestart` 太激进

**症状**: pod 任何 restart 触发整个 group recreate, vllm 永远不能稳定。

**修复**: `restartPolicy: Default`。

### 坑 #4: `--async-scheduling` 不兼容 ray executor

**修复**: Multi-host (`--distributed-executor-backend=ray`) 配置不能用 `--async-scheduling`。

### 坑 #5 (BLOCKER): `AttributeError: d.coords` on init_device

**症状**: `File ".../tpu_inference/distributed/utils.py", line 125, AttributeError`

**根因**: GKE auto-injection 给每个 multi-host pod 注入的是 **single-host 视角** env vars：
- `TPU_WORKER_HOSTNAMES=localhost`（只有自己）
- `TPU_PROCESS_ADDRESSES=localhost:8471`（只有自己）
- `TPU_WORKER_ID=0`（每个 pod 都是 0）

JAX libtpu 看到 `Expected 2 worker addresses, got 1` 就 fallback 到 CPU device。

**调试方法**：起 busybox pod 在同 node pool, `env | grep TPU` 看实际注入值。

**修复（已在 yaml 里）**：手动 override（见 Step 2b 的 KEY FIX block）。

### 坑 #6: Spot multi-host pool 容易被 preempt

**Workaround**: 生产用 reservation；开发接受偶尔重启。

### 坑 #7（**新增 · 复现时发现**）: 不 warmup 时 c=1 数据严重低估

**症状**: 首跑 1K/1K c=1 = 21.5 tok/s，warmup 后 = 37.5 tok/s（差 75%）。

**原因**: 第一个请求触发 XLA padding bucket 编译，包含在测量时长内。

**修复**: benchmark 前先跑一轮 warmup（见 Step 5）。

### 坑 #8（**新增**）: Output tok/s vs Total tok/s 容易混淆

**症状**: 看到 "8K/1K c=4 = 856 tok/s" 以为很快，其实 output 只有 95.

**原因**: vllm bench 报告里 `Total token throughput` 含 prefill input tokens，对长 prompt 会高估 5-10 倍。

**修复**: **始终看 `Output token throughput`**（这才是 generated 速度）。

---

## 资源清理

```bash
# 删 LWS workload（保留 node pool 可立即重新部署）
kubectl delete lws vllm-mh
kubectl delete service vllm-mh-service

# 完整删（包括 node pool）
gcloud container node-pools delete np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev --quiet
gcloud compute resource-policies delete chrisya-tpu7x-spot-mh \
  --region=us-central1 --project=cloud-tpu-multipod-dev --quiet
```

---

## 参考资料

| 资源 | 链接 |
|------|------|
| 单机 README | [README.md](README.md) |
| LeaderWorkerSet docs | https://lws.sigs.k8s.io/ |
| `tpu_inference/distributed/utils.py:125` | https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/distributed/utils.py#L125 |
| Google Codelabs (multihost vLLM + Ray, v6e) | https://codelabs.developers.google.com/next26/aiinfra-learning-pod/screen2-advanced-inferencing-part-1 |
