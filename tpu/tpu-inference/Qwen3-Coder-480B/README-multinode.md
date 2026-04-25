# Qwen3-Coder-480B-A35B-Instruct FP8 Multi-host Inference on TPU v7x-16

> 端到端指南：在 **TPU v7x-16（2 hosts × 4 chips = 8 chips · 16 devices · 1.5 TB HBM）** 上跑单实例 vLLM 推理（TP=16）。
>
> **状态**（2026-04-25 实测）：✅ **完整跑通**，已完成 5 组 benchmark。关键突破是手动注入 `TPU_WORKER_HOSTNAMES` 等 env vars 绕过 GKE auto-injection 的 single-host 默认值。
>
> 单机版（v7x-8 + TP=8）见同目录 [README.md](README.md)。

---

## 🎯 实测结果（2026-04-25）

### Benchmark 摘要（v7x-16 TP=16, 单实例）

| 场景 | TTFT (median) | TPOT (median) | Output tok/s | Total tok/s |
|------|---------------|---------------|--------------|-------------|
| 1K/1K c=1 | 121 ms | 26.6 ms | 21.5 | 43.0 |
| 1K/1K c=4 | 272 ms | 27.8 ms | 99.9 | 199.8 |
| 1K/1K c=16 | 4.26 s | 65.3 ms | 222.8 | 445.5 |
| 8K/1K c=4 | 1.19 s | 28.3 ms | 95.2 | **856.6** |
| 8K/1K c=16 | 1.19 s | 68.9 ms | 222.7 | **2004** |

### 与单机 (v7x-8 TP=8) 对比

| 场景 | 单机 TTFT | MH TTFT | 单机 TPOT | MH TPOT | 单机 tok/s | MH tok/s |
|------|-----------|---------|-----------|---------|------------|----------|
| 1K/1K c=1 | 95 ms | 121 ms (+27%) | 20.6 ms | 26.6 ms (+29%) | 48 | 21.5 |
| 8K/1K c=4 | 1495 ms | **1191 ms (-20%)** | 23.2 ms | 28.3 ms (+22%) | 162 | 856.6 |

### 选型建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 短 prompt 低延迟 | 单机 v7x-8 | TPOT 更低、cold start 更快 |
| 长 prompt (≥8K) | **多机 v7x-16** | TTFT 更短，prefill 算力翻倍 |
| 高并发 (c≥16) | **多机 v7x-16** | total tok/s 大幅提升（~2× 单机潜力） |
| 大 KV cache | **多机 v7x-16** | HBM 1.5 TB（单机 768 GB） |

---

## 硬件与架构

| 项目 | 要求 |
|------|------|
| TPU | **v7x-16**（2x2x2 拓扑，8 chips = 16 devices, 跨 2 节点） |
| HBM | 总 **1.5 TB** (192 GB/chip × 8) |
| 主机内存 | ≥850 GB **per node** |
| 网络 | DCN（节点间 8471/6379 端口） |
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

## 完整复现命令

### Step 0: 集群前置（一次性）

```bash
# 安装 LeaderWorkerSet (LWS)
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.7.0/manifests.yaml
```

### Step 1: 创建 multi-host node pool

```bash
# 1a. workload policy（topology 2x2x2 = 8 chips）
gcloud compute resource-policies create workload-policy chrisya-tpu7x-spot-mh \
  --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2 \
  --region=us-central1 --project=cloud-tpu-multipod-dev

# 1b. multi-host spot node pool（2 nodes）
gcloud container node-pools create np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev \
  --machine-type=tpu7x-standard-4t --tpu-topology=2x2x2 \
  --num-nodes=2 --node-locations=us-central1-c \
  --disk-type=hyperdisk-balanced --disk-size=500 --spot \
  --max-pods-per-node=110 --image-type=COS_CONTAINERD \
  --workload-metadata=GKE_METADATA \
  --placement-policy=chrisya-tpu7x-spot-mh
```

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
    restartPolicy: Default
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

            # === 关键 fix: 覆盖 GKE auto-injected single-host env vars ===
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

### Step 3: 监控启动

```bash
# 等所有 pod Running
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh -w

# 启动时间分解（实测）：
# - Pod scheduling + Ray cluster: ~70s
# - Weight loading (49 shards from Lustre): 468s (7.8 min)
# - XLA compile + warmup: ~140s
# - 总冷启动: ~12 min（首次）

# 验证启动完成
kubectl logs vllm-mh-0 -c main | grep "Application startup complete"
```

### Step 4: Smoke test

```bash
kubectl exec vllm-mh-0 -c main -- bash -c '
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen3-Coder-480B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍 TPU v7\"}],\"max_tokens\":100}"
'
```

### Step 5: Benchmark

```bash
kubectl exec vllm-mh-0 -c main -- bash -c '
vllm bench serve --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --served-model-name Qwen3-Coder-480B-FP8 \
  --base-url http://localhost:8000 --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 \
  --num-prompts 32 --max-concurrency 16 --ignore-eos
'
```

---

## 🐛 踩坑实录（实战经验，全部 verified）

### 坑 #1: TPU 拓扑 4x2x1 不兼容 tpu7x-standard-4t

**症状**: `gcloud container node-pools create` 报 `Accelerator topology: 4x2x1 is not compatible with the provided machine type: tpu7x-standard-4t`

**修复**: 用 **`2x2x2`**（同样 8 chips 但是 cube 布局）。

### 坑 #2: `ray --block &` 在 sh -c args 内不生效

**症状**: Leader pod log 一直停在 `ray start --block` 输出，永远不进 `vllm serve`。

**原因**: K8s container `args:` 段内的 `&` 后台符号在某些 shell parse 下被吞掉。

**修复**: **Leader 用 daemon 模式（不加 `--block`）**，worker 用 `--block`。

### 坑 #3: LWS `RecreateGroupOnPodRestart` 太激进

**症状**: pod 任何 restart 触发整个 group recreate，永远不能稳定到 vllm 加载阶段。

**修复**: 改 `restartPolicy: Default`。

### 坑 #4: `--async-scheduling` 不兼容 ray executor

**症状**: `ValidationError: ray does not support async scheduling yet`

**修复**: Multi-host (`--distributed-executor-backend=ray`) 配置不能用 `--async-scheduling`。

### 坑 #5 (最关键): `AttributeError: d.coords` on init_device

**症状**:
```
File "/workspace/tpu_inference/tpu_inference/distributed/utils.py", line 125,
    in get_device_topology_order_id
    local_anchor = min(d.coords for d in local_devices)
AttributeError
```

**根因**: GKE auto-injection 给每个 multi-host pod 注入的是 **single-host 视角** 的 env vars：
```
TPU_WORKER_HOSTNAMES=localhost          ← 只有自己
TPU_PROCESS_ADDRESSES=localhost:8471    ← 只有自己
TPU_WORKER_ID=0                          ← 都是 0
```

JAX libtpu 看到 `Expected 2 worker addresses, got 1` 就 fallback 到 CPU device，导致 device 没 `coords` 属性。

**调试方法**: 起一个简单 busybox pod 在同 node pool，`env | grep TPU` 看实际注入值——会发现都是 single-host 默认。

**修复（Verified）**: 在容器 args 内手动 override：
```bash
LEADER_IP=$(getent hosts vllm-mh-0.vllm-mh | awk '{print $1}')
WORKER_IP=$(getent hosts vllm-mh-0-1.vllm-mh | awk '{print $1}')
export TPU_WORKER_HOSTNAMES="${LEADER_IP},${WORKER_IP}"
export TPU_WORKER_ID=${LWS_WORKER_INDEX}
export TPU_PROCESS_ADDRESSES="${LEADER_IP}:8471,${WORKER_IP}:8471"
```

> **本节是整个 multi-host 部署的关键突破。** vLLM/tpu-inference 上游文档没说明 GKE multi-host 需要手动覆盖这些 env vars。

### 坑 #6: Spot multi-host pool 容易被 preempt

**症状**: pool 状态从 RUNNING 突然变 RECONCILING，几分钟后又回 RUNNING（不同 node ID）。

**Workaround**: 用 reservation 替代 spot（生产环境）。开发用 spot 接受偶尔重启即可。

---

## 资源清理

```bash
kubectl delete lws vllm-mh
kubectl delete service vllm-mh-service
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
