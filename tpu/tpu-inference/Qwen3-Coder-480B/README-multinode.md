# Qwen3-Coder-480B-A35B-Instruct FP8 Multi-host Inference on TPU v7x-16

> 端到端指南：在 **TPU v7x-16（2 hosts × 4 chips = 8 chips · 16 devices · 1.5 TB HBM）** 上跑单实例 vLLM 推理（TP=16）。
>
> **与单机 v7x-8 (TP=8) 的区别**：
> - 用 `tpu7x-16` multi-host node pool（2 节点为一个 slice）
> - 用 **LeaderWorkerSet (LWS)** + **Ray** 协调跨节点
> - vLLM 起 1 个 API server（leader），其他节点跑 Ray worker
> - 适合更大模型（1T+）或更高吞吐场景；Qwen3-Coder 480B 单机也能跑，但 TP=16 throughput 更高

> **代码仓库**: 上游 [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference)（main 分支）
> **模型**: [`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8)（~480 GB FP8）
> **参考**: [Google Codelabs: Deploy Multihost TPU vLLM Inferencing with Ray on GKE](https://codelabs.developers.google.com/next26/aiinfra-learning-pod/screen2-advanced-inferencing-part-1)

## 🎯 关键性能（⏳ 待实测）

> 首次跑通后会更新。预期 vs 单机 (TP=8)：
> - **Throughput**：理论 2x（chip 数翻倍），实测可能 1.5-1.8x（ICI/通信开销）
> - **Per-user latency**：相似（TPOT ~21ms）或略高（跨 host 通信）
> - **HBM**：1.5 TB total，可装 1T 级模型 FP8（Qwen3-Coder 480B 占 ~32%，余下给 KV）

| 操作点 | 配置 | 预期 | 实测 |
|--------|-----|------|-----|
| 启动时间 (cold) | TP=16 + Ray cluster | ~10-15 min | ⏳ |
| Single-user TPOT | c=1 | ~21-25 ms | ⏳ |
| Peak throughput (c=64) | 1K/1K | ~2500-3000 tok/s | ⏳ |
| HBM 占用/device | 模型 + KV | ~50-70 GB | ⏳ |

---

## 硬件与架构

| 项目 | 要求 |
|------|------|
| TPU | **v7x-16**（4x2x1 拓扑, 8 chips = 16 devices, 跨 2 节点） |
| HBM | 总 **1.5 TB** (192 GB/chip × 8) |
| 主机内存 | ≥850 GB **per node** |
| 网络 | 节点间需大 MTU (8896) + 高带宽（auto accelerator-network-profile） |
| 存储 | Lustre 共享卷（推荐）或 GCS Fuse + Rapid Cache |

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│  GKE Cluster (chrisya-v7x-v134)                        │
│                                                         │
│  ┌────────────────────────┐  ┌────────────────────────┐│
│  │ Node 1 (host-0)        │  │ Node 2 (host-1)        ││
│  │  4 chips · 768 GB HBM  │  │  4 chips · 768 GB HBM  ││
│  │                        │  │                        ││
│  │  Pod: vllm-mh-0        │  │  Pod: vllm-mh-1        ││
│  │   - Ray head 6379      │  │   - Ray worker         ││
│  │   - vLLM API 8000      │  │   - (just compute)     ││
│  │   - LWS_WORKER_INDEX=0 │  │   - LWS_WORKER_INDEX=1 ││
│  │   - TP slice 0-7       │  │   - TP slice 8-15      ││
│  └─────┬──────────────────┘  └─────────┬──────────────┘│
│        │                                │               │
│        └────── ICI 跨 host (高速) ──────┘               │
│        └────── DCN (Ray RPC) ───────────┘               │
│                                                         │
│  Service: vllm-mh-service (LoadBalancer/ClusterIP)     │
│   selector: leaderworkerset.../worker-index=0          │
│   → 只指向 leader (vllm-mh-0)                          │
└─────────────────────────────────────────────────────────┘
```

---

## ⚠️ 关键差异 vs 单机 (README §Step 1-7)

| 维度 | 单机 v7x-8 | **Multi-host v7x-16** |
|------|----------|----------------------|
| Pod 模式 | 单 Pod | **LeaderWorkerSet (LWS)** size=2 |
| TP size | 8 | **16** |
| Ray | 不需要 | **必需**（leader head + worker） |
| Cluster addons | 无要求 | **RayOperator + LWS Helm chart** |
| 网络 | 默认 | 推荐**自定义 VPC** + `mtu=8896` |
| 启动时长 | ~7 min | **~10-15 min**（含 Ray cluster 协调）|
| Service selector | `app=...` | `leaderworkerset.../worker-index=0`（只暴露 leader）|

---

## Step 0: 集群前置（一次性）

### 0a: 安装 LeaderWorkerSet (LWS) Helm chart

```bash
helm install lws oci://registry.k8s.io/lws/charts/lws \
  --version=0.7.0 \
  --namespace lws-system \
  --create-namespace \
  --wait
```

验证：`kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io`

### 0b: 启用集群 RayOperator addon（如未启用）

```bash
gcloud container clusters update chrisya-v7x-v134 \
  --location=us-central1 \
  --update-addons=RayOperator=ENABLED
```

> 已经装了 GcsFuseCsiDriver，不需要重复。

---

## Step 1: 创建 multi-host node pool

### 1a: 创建 workload policy（topology 4x2x1 = 8 chips）

```bash
gcloud compute resource-policies create workload-policy chrisya-tpu7x-spot-mh \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=4x2x1 \
  --region=us-central1 \
  --project=cloud-tpu-multipod-dev
```

> **拓扑选择**：`4x2x1` 8 chips = **2 hosts × 4 chips/host**，是 v7x multi-host 的最小单位。

### 1b: 创建 multi-host node pool（spot, 2 nodes）

```bash
gcloud container node-pools create np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 \
  --region=us-central1 \
  --project=cloud-tpu-multipod-dev \
  --machine-type=tpu7x-standard-4t \
  --tpu-topology=4x2x1 \
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

> **预计 2-5 分钟**，spot capacity 可能要重试。

### 1c: 验证 2 个 node 已 join

```bash
kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology \
  | grep np-tpu7x-spot-mh
```

预期看到 2 个 node，topology 都是 `4x2x1`。

---

## Step 2: 部署 LWS + vLLM TP=16

### 2a: 准备 Service（只暴露 leader pod）

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

### 2b: 部署 LeaderWorkerSet（2 host 协调）

> **关键点**：
> - `size: 2` = 2 个 pod (leader + 1 worker)
> - leader 启 Ray head + vllm api server, worker 只启 ray worker
> - 用 lustre-pvc 共享权重（已有 `/lustre/qwen3-coder-480b-fp8`）
> - 用 model name `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` + HF cache symlink

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
    restartPolicy: RecreateGroupOnPodRestart
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
          cloud.google.com/gke-tpu-topology: 4x2x1
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
          image: us-central1-docker.pkg.dev/chris-pgp-host/ai-infra/vllm-tpu:latest
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
            echo "Leader IP: $LEADER_IP"

            export JAX_PLATFORMS=''
            export TPU_MULTIHOST_BACKEND=ray
            export JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT=300
            export VLLM_HOST_IP=$MY_TPU_IP
            export MODEL_IMPL_TYPE=vllm
            export HF_HOME=/lustre
            export HF_HUB_OFFLINE=1

            if [ "$LWS_WORKER_INDEX" = "0" ]; then
              echo "=== Starting Ray Head ==="
              ray start --head --port=6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}' --block &
              sleep 20
              until ray status; do sleep 5; done

              echo "=== Starting vLLM API Server (TP=16) ==="
              python3 -m vllm.entrypoints.openai.api_server \
                --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
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
                --async-scheduling \
                --host=0.0.0.0 --port=8000
            else
              echo "=== Starting Ray Worker, joining $LEADER_IP:6379 ==="
              ray start --address=${LEADER_IP}:6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}' --block
            fi
          ports:
          - { containerPort: 8000 }
          - { containerPort: 6379 }
          env:
          - { name: SKIP_JAX_PRECOMPILE, value: "1" }
          - { name: VLLM_XLA_CHECK_RECOMPILATION, value: "0" }
          - { name: USE_MOE_EP_KERNEL, value: "0" }
          - { name: USE_BATCHED_RPA_KERNEL, value: "0" }
          - { name: VLLM_LOGGING_LEVEL, value: INFO }
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

### 2c: 等所有 pod Ready

```bash
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh -w
```

预期 2 个 pod 都 `Ready 1/1`，需要 ~10-15 分钟。

---

## Step 3: 验证 + Benchmark

### 3a: Smoke test（端口转发）

```bash
kubectl port-forward service/vllm-mh-service 7800:8000 &
sleep 5
curl http://localhost:7800/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Coder-480B-FP8","prompt":"def quicksort(arr):","max_tokens":50,"temperature":0}'
```

### 3b: Benchmark vs 单机 TP=8

```bash
LEADER_POD=$(kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh,leaderworkerset.sigs.k8s.io/worker-index=0 -o jsonpath='{.items[0].metadata.name}')
kubectl exec $LEADER_POD -- vllm bench serve --backend vllm \
  --model Qwen3-Coder-480B-FP8 \
  --tokenizer /lustre/qwen3-coder-480b-fp8 \
  --host localhost --port 8000 \
  --num-prompts 32 \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --max-concurrency 16 --request-rate inf --num-warmups 2 --ignore-eos
```

---

## Step 4: 实测对比表（⏳ 待填）

### TP=16 (multi-host) vs TP=8 (单机)

| Concurrency | TP=8 tok/s | **TP=16 tok/s** | TP=8 TPOT | **TP=16 TPOT** | Speedup |
|------------:|-----------:|----------------:|----------:|---------------:|--------:|
| 1 | 48 | ⏳ | 20.6 ms | ⏳ | ⏳ |
| 4 | 177 | ⏳ | 22.2 ms | ⏳ | ⏳ |
| 16 | 602 | ⏳ | 25.6 ms | ⏳ | ⏳ |
| 64 | 1478 | ⏳ | 40.0 ms | ⏳ | ⏳ |

---

## 常见问题排查（持续更新）

### 1. Pod stuck Pending → check spot capacity / placement policy

### 2. Ray worker 找不到 leader → DNS 解析超时
- 检查 `vllm-mh-0.vllm-mh` 是否能解析（LWS 自动建 service）
- 增大 `JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT`

### 3. TP=16 启动 OOM → `gpu-memory-utilization` 降到 0.85

### 4. KV cache 不够 → `kv-cache-dtype=fp8` 必须开

---

## 参考资料

| 资源 | 链接 |
|------|------|
| Google Codelabs (multihost vLLM + Ray) | [link](https://codelabs.developers.google.com/next26/aiinfra-learning-pod/screen2-advanced-inferencing-part-1) |
| 上游 multihost benchmark 脚本 | [run_qwen3_coder_480b_1k_8k.sh](https://github.com/vllm-project/tpu-inference/blob/main/scripts/multihost/benchmarks/torchax/run_qwen3_coder_480b_1k_8k.sh) |
| 单机 README | [README.md](README.md) |
| LeaderWorkerSet docs | https://lws.sigs.k8s.io/ |

---

## 后续 TODO

- [ ] 实测启动时间
- [ ] 实测 TP=16 vs TP=8 throughput 对比
- [ ] 实测 KV cache 容量
- [ ] 跑 1k/8k 和 8k/1k 对比
- [ ] 写完整踩坑记录（实操中遇到的）
- [ ] 写英文版 README-multinode.en.md
