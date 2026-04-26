# Qwen3-Coder-480B FP8 Multi-host Inference on TPU v7x-16 — 完全可复现指南

> **目标**：新人 100% 照抄本文档可一次成功部署 multi-host vLLM TP=16 推理服务并完成 5 组 benchmark 对比。
>
> **状态**（2026-04-26 二次验证）：✅ 文档完全可复现，但 **CRITICAL 结论**：multi-host TP=16 单实例 output throughput 全场景比单机 v7x-8 差 17~63%，**不推荐生产用** — 客户应跑 2 个独立 v7x-8 单机实例（data parallel）。
>
> 本文档目的：(1) 提供完整可复现部署步骤；(2) 给出客观对比数据；(3) 解释为什么这种部署模式不适合 Qwen3-Coder-480B。
>
> 单机版（v7x-8 + TP=8）见同目录 [README.md](README.md) — **生产推荐**。

---

## 📋 目录

1. [测试环境（已就绪 / 不需要重建）](#1-测试环境已就绪)
2. [Step 0: 检查环境 + 切 kubectl context](#step-0-检查环境)
3. [Step 1: 清理现有部署（如有）](#step-1-清理现有部署)
4. [Step 2: 部署 LeaderWorkerSet](#step-2-部署-leaderworkerset)
5. [Step 3: 监控启动到 ready](#step-3-监控启动到-ready)
6. [Step 4: Smoke test](#step-4-smoke-test)
7. [Step 5: Benchmark（5 组 + warmup）](#step-5-benchmark)
8. [核心结论：Multi-host vs 单机对比](#核心结论multi-host-vs-单机对比)
9. [踩坑实录（8 个 verified 问题）](#踩坑实录)
10. [资源清理（测试结束后）](#资源清理)

---

## 1. 测试环境（已就绪）

### 1.1 GCP / GKE 资源（**已存在，不需要重建**）

| 资源类型 | 名称 / 配置 | 状态 |
|---------|------------|------|
| GCP Project | `cloud-tpu-multipod-dev` | ✅ 已就绪 |
| GKE Cluster | `chrisya-v7x-v134` (us-central1, master 1.34.4-gke.1193000) | ✅ RUNNING |
| Node Pool | `np-tpu7x-spot-mh` (2 nodes × tpu7x-standard-4t, topology 2x2x2, spot, us-central1-c) | ✅ RUNNING |
| Workload Policy | `chrisya-tpu7x-spot-mh` (HIGH_THROUGHPUT, accelerator-topology=2x2x2) | ✅ 已就绪 |
| Lustre PVC | `lustre-pvc` (36 TB, ReadWriteMany) | ✅ Bound |
| LWS Operator | LeaderWorkerSet v0.7.0 (`lws-system` namespace) | ✅ Installed |

### 1.2 模型权重（**已下载，不需要重新下**）

| 项目 | 值 |
|------|---|
| 模型 | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` |
| HF cache 路径 | `/lustre/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/000...aaaa` |
| 实际权重位置 | `/lustre/qwen3-coder-480b-fp8` (snapshot symlink 指向这里) |
| 大小 | ~450 GB FP8 safetensors (49 shards) |

### 1.3 硬件规格

| 项目 | 值 |
|------|---|
| TPU | v7x-16 (2 hosts × 4 chips = 8 chips · 16 devices) |
| HBM | 总 **1.5 TB** (192 GB/chip × 8) |
| 主机内存 | 850 GB / node |
| 网络 | DCN（节点间 8471/6379 端口） |

### 1.4 架构图

```
┌─────────────────────────────────────────────────────────┐
│  GKE Cluster (chrisya-v7x-v134, us-central1)           │
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

## Step 0: 检查环境

### 0.1 切 kubectl context（**关键 · 容易忘**）

> ⚠️ 当前 gcloud 的 default project 可能是别的项目（如 `gpu-launchpad-playground`），但 cluster 在 `cloud-tpu-multipod-dev`。必须显式拿正确的 kubeconfig。

```bash
# 拉取 cluster credentials（context 名: gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134）
gcloud container clusters get-credentials chrisya-v7x-v134 \
  --region=us-central1 --project=cloud-tpu-multipod-dev

# 验证当前 context
kubectl config current-context
# 期望输出: gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134
```

### 0.2 验证已就绪的资源

```bash
# 1. Node pool（应该看到 2 个 node, topology=2x2x2）
kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology \
  | grep np-tpu7x-spot-mh

# 2. LWS operator（应该看到 leaderworkerset CRD）
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io

# 3. Lustre PVC（应该 STATUS=Bound）
kubectl get pvc lustre-pvc

# 4. Workload policy（应该看到 acceleratorTopology: 2x2x2）
gcloud compute resource-policies describe chrisya-tpu7x-spot-mh \
  --region=us-central1 --project=cloud-tpu-multipod-dev | grep -A 1 workloadPolicy
```

> **如果上面任何一项没就绪**，看 [附录 A: 从零创建环境](#附录-a-从零创建环境)。本文档默认这些已就绪。

---

## Step 1: 清理现有部署

> 如果集群里**没有** `vllm-mh` 这个 LWS，跳过本步骤直接到 Step 2。

```bash
# 1. 看是否有现有 vllm-mh
kubectl get lws,svc,pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh

# 2. 拆 LWS 和 Service
kubectl delete lws vllm-mh --ignore-not-found
kubectl delete service vllm-mh-service --ignore-not-found

# 3. 等所有 vllm-mh-* pods 完全终止（约 60-90s）
while [ "$(kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh --no-headers 2>/dev/null | wc -l)" != "0" ]; do
  echo "等待 pods 终止..."
  sleep 10
done
echo "✅ 清理完成"
```

> ⚠️ **不要拆 node pool** — 多 host node pool 重建很慢且 spot 容量经常拿不到。

---

## Step 2: 部署 LeaderWorkerSet

### 2.1 Apply Service（headless, 只暴露 leader）

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

### 2.2 Apply LeaderWorkerSet（**完整 yaml · 直接 copy 跑**）

> ⚠️ **黄字 KEY FIX 段落不要删**：手动覆盖 GKE auto-injected single-host TPU env vars，否则会卡在 `AttributeError: d.coords`（坑 #5）。

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
    restartPolicy: Default            # ⚠️ 不能用 RecreateGroupOnPodRestart, 任何 restart 触发整个 group 重建
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
          image: vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626    # codelab 验证版本，不要随便升
          command: ["sh", "-c"]
          args:
          - |
            MY_TPU_IP=$(hostname -I | awk '{print $1}')
            echo "[boot] My TPU IP: $MY_TPU_IP, LWS_WORKER_INDEX=$LWS_WORKER_INDEX"
            LEADER_DNS="vllm-mh-0.vllm-mh"
            WORKER_DNS="vllm-mh-0-1.vllm-mh"
            until getent hosts $LEADER_DNS; do echo "[boot] waiting leader DNS..."; sleep 5; done
            until getent hosts $WORKER_DNS; do echo "[boot] waiting worker DNS..."; sleep 5; done
            LEADER_IP=$(getent hosts $LEADER_DNS | awk '{print $1}')
            WORKER_IP=$(getent hosts $WORKER_DNS | awk '{print $1}')
            echo "[boot] LEADER_IP=$LEADER_IP WORKER_IP=$WORKER_IP"

            # === KEY FIX: 覆盖 GKE auto-injected single-host TPU env vars ===
            # GKE webhook 给每个 multi-host pod 都注入 TPU_WORKER_HOSTNAMES=localhost
            # 和 TPU_WORKER_ID=0 (它当成单 pod 看待)
            # 不覆盖会卡在 vLLM init_device: AttributeError: d.coords
            export TPU_WORKER_HOSTNAMES="${LEADER_IP},${WORKER_IP}"
            export TPU_WORKER_ID=${LWS_WORKER_INDEX}
            export TPU_PROCESS_ADDRESSES="${LEADER_IP}:8471,${WORKER_IP}:8471"
            export TPU_PROCESS_PORT=8471
            export TPU_HOST_BOUNDS="1,1,2"
            export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
            export TPU_TOPOLOGY="2x2x2"
            export TPU_ACCELERATOR_TYPE="tpu7x-16"
            export TPU_SKIP_MDS_QUERY=true
            echo "[boot] TPU_WORKER_ID=$TPU_WORKER_ID TPU_WORKER_HOSTNAMES=$TPU_WORKER_HOSTNAMES"

            # === vLLM / JAX env ===
            export JAX_PLATFORMS=''
            export PJRT_DEVICE=TPU
            export TPU_BACKEND_TYPE=jax
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
            export TPU_MULTIHOST_BACKEND=ray
            export VLLM_HOST_IP=$MY_TPU_IP
            export MODEL_IMPL_TYPE=vllm
            export HF_HOME=/lustre               # 权重在 /lustre/hub/models--Qwen--...
            export HF_HUB_OFFLINE=1              # 不联网
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
              echo "=== Starting Ray Worker, joining $LEADER_IP:6379 ==="
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

### 2.3 验证 apply 成功

```bash
# 应该看到 vllm-mh-0 (leader) 和 vllm-mh-0-1 (worker) 两个 pod
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh -w
# 等到两个都 1/1 Running（约 20-70s）后 Ctrl+C
```

---

## Step 3: 监控启动到 ready

### 3.1 看 boot env vars（验证 KEY FIX 生效）

```bash
# 等 5s 让 log 输出，然后检查
sleep 5
kubectl logs vllm-mh-0 -c main 2>&1 | grep "\[boot\]" | head -5
```

期望看到（IP 可能不同）：
```
[boot] My TPU IP: 10.120.6.7, LWS_WORKER_INDEX=0
[boot] LEADER_IP=10.120.6.7 WORKER_IP=10.120.12.6
[boot] TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=10.120.6.7,10.120.12.6
```

> ❌ 如果 `TPU_WORKER_HOSTNAMES=localhost` → KEY FIX 没生效，回去检查 yaml。
> ❌ 如果 `LEADER_IP` 等了很久没出现 → DNS 没 resolve, 检查 LWS 是否正确建了 service。

### 3.2 等待启动完成（**约 12-13 min**）

```bash
# 实时跟踪关键 log（出现 "Application startup complete" 表示 ready）
kubectl logs vllm-mh-0 -c main -f | grep -E "Application startup complete|Init mesh|Loading weights took|safetensors checkpoint shards"
```

预期出现的关键 log（按时间顺序）：

| 时间点 | 日志 | 阶段 |
|-------|------|------|
| ~T+11s | `Ray runtime started` | Ray cluster ✅ |
| ~T+74s | `Init mesh \| mesh=Mesh('data': 1, 'model': 16, axis_types=...)` | JAX mesh ✅ |
| T+74s ~ T+557s | `Loading safetensors checkpoint shards: X% Completed \| Y/49` | 权重加载中 |
| ~T+557s (9.3 min) | `INFO ... Loading weights took 480.XX seconds` | 权重加载完 ✅ |
| T+557s ~ T+800s | （没明显 log，是 XLA 编译） | XLA compile |
| ~T+800s (~13 min) | `INFO: Application startup complete.` | **Server ready ✅** |

> 💡 实测启动时间分两种：
> - **首次冷启动**（image 未 pull）：~12 min（pod scheduling 70s + 其余 ~10 min）
> - **重启**（image cached）：~13 min（pod scheduling 仅 ~20s, 但 weight load 480s + XLA 编译 244s 还要再走）
> - XLA cache 不持久化，所以重启不比首启快多少

### 3.3 验证 server ready

```bash
kubectl logs vllm-mh-0 -c main 2>&1 | grep -c "Application startup complete"
# 期望: 1 (出现过一次)
```

---

## Step 4: Smoke test

```bash
kubectl exec vllm-mh-0 -c main -- bash -c '
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen3-Coder-480B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍 TPU v7\"}],\"max_tokens\":100}"
'
```

期望返回：JSON 含 `"role":"assistant","content":"<某段中文>"`，并能看到 `"finish_reason":"stop"`。
首次请求可能耗时 ~30s（XLA 编译），后续 <2s。

---

## Step 5: Benchmark

> ⚠️ **先 warmup, 再正式测**。不 warmup 时 c=1 数据会被低估 ~75%（第一次请求触发 XLA padding bucket 编译）。
> ⚠️ **看 Output token throughput**, 不是 Total token throughput（后者含 prefill input, 长 prompt 会高估 5-10 倍）。

### 5.1 Warmup（约 5 min）

```bash
kubectl exec vllm-mh-0 -c main -- bash -c '
for inp in 1024 8192; do
  for out in 64 1024; do
    echo "  warmup ${inp}/${out}..."
    vllm bench serve --backend vllm \
      --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
      --served-model-name Qwen3-Coder-480B-FP8 \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name random --random-input-len $inp --random-output-len $out \
      --num-prompts 2 --max-concurrency 1 --ignore-eos > /dev/null 2>&1
  done
done
echo "warmup done"
'
```

### 5.2 5 组正式 benchmark（约 15 min）

```bash
kubectl exec vllm-mh-0 -c main -- bash -c '
run_bench() {
  local label="$1" inp="$2" out="$3" prompts="$4" conc="$5"
  echo ""
  echo "===== $label ====="
  vllm bench serve --backend vllm \
    --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --served-model-name Qwen3-Coder-480B-FP8 \
    --base-url http://localhost:8000 --endpoint /v1/completions \
    --dataset-name random --random-input-len $inp --random-output-len $out \
    --num-prompts $prompts --max-concurrency $conc --ignore-eos 2>&1 | tail -22
}
run_bench "1K/1K c=1"   1024 1024 4  1
run_bench "1K/1K c=4"   1024 1024 16 4
run_bench "1K/1K c=16"  1024 1024 32 16
run_bench "8K/1K c=4"   8192 1024 16 4
run_bench "8K/1K c=16"  8192 1024 32 16
'
```

### 5.3 期望数据范围（与本次实测对比）

| 场景 | Output tok/s | TTFT median | TPOT median | 备注 |
|------|------------:|------------:|------------:|------|
| 1K/1K c=1 | 35-40 | 90-100 ms | 26-27 ms | warmup 后 |
| 1K/1K c=4 | 95-100 | 270-280 ms | 27-29 ms | |
| 1K/1K c=16 | 215-225 | 4200-4400 ms | 64-67 ms | TTFT 高因 prefill batching |
| 8K/1K c=4 | 130-140 | 1180-1200 ms | 28-30 ms | |
| 8K/1K c=16 | 220-230 | 1180-1200 ms | 68-70 ms | |

> 偏差超过 ±10% 检查：(1) warmup 是否充分；(2) node 是否被 spot preempt 重调度过；(3) 是否有别的 workload 在抢 TPU/CPU 资源。

---

## 核心结论：Multi-host vs 单机对比

### Output Throughput (tok/s) — 真正的生成速度

| 场景 | 单机 v7x-8 (TP=8) | 多机 v7x-16 (TP=16) | 差异 |
|------|------------------:|-------------------:|----:|
| 1K/1K c=1 | **48** | 37.5 | ❌ -22% |
| 1K/1K c=4 | **177** | 98.4 | ❌ -44% |
| 1K/1K c=16 | **602** | 220.0 | ❌ -63% |
| 8K/1K c=4 | **162** | 134.3 | ❌ -17% |
| 8K/1K c=16 | **483** | 223.1 | ❌ -54% |

### TTFT (median) — 首字节时间

| 场景 | 单机 | 多机 | 差异 |
|------|----:|----:|----:|
| 8K/1K c=4 | 1495 ms | **1188 ms** | ✅ -20% |
| 8K/1K c=16 | 2418 ms | **1189 ms** | ✅ -51% |
| 1K/1K c=16 | 549 ms | 4306 ms | ❌ +684% |

### 为什么 multi-host 更慢？

1. **TP=16 跨 DCN**: decode 每 token 都需要全 16 device all-reduce, 跨节点 DCN 同步开销 >>> 翻倍算力收益
2. **MoE expert 路由跨节点**: top-k expert 选择导致跨 host all-to-all
3. **Single instance TP=16 不是甜区**: TP 切得越细，通信占比越高

### ✅ 正确生产建议

| 客户场景 | 推荐配置 | 理由 |
|---------|---------|------|
| **默认 / 大部分场景** | **2 × v7x-8 单机 (TP=8) Data Parallel** | 总 throughput ≈ 2× 单机 = ~1200 tok/s @ c=16 |
| 长 prompt 高并发 | 单机 PD 分离 1P1D | TPOT 改善 11% |
| 仅当需单实例 + max-model-len ≥ 32K | Multi-host TP=16（牺牲 throughput 换 HBM） | 1.5 TB HBM 才放下大 KV cache |

---

## 踩坑实录

### 坑 #1: TPU 拓扑 4x2x1 不兼容 tpu7x-standard-4t

**症状**: `Accelerator topology: 4x2x1 is not compatible with the provided machine type: tpu7x-standard-4t`
**修复**: 用 **`2x2x2`**（cube 布局，同样 8 chips）。

### 坑 #2: `ray --block &` 在 sh -c args 内不生效

**症状**: Leader pod log 一直停在 `ray start --block`。
**修复**: **Leader 用 daemon 模式**（不加 `--block`），worker 用 `--block`。

### 坑 #3: LWS `RecreateGroupOnPodRestart` 太激进

**症状**: pod 任何 restart 触发整个 group recreate，永远不能稳定到 vllm 加载阶段。
**修复**: `restartPolicy: Default`。

### 坑 #4: `--async-scheduling` 不兼容 ray executor

**症状**: `ValidationError: ray does not support async scheduling`
**修复**: Multi-host (`--distributed-executor-backend=ray`) 配置不能用 `--async-scheduling`。

### 坑 #5 (BLOCKER): `AttributeError: d.coords` on init_device

**症状**:
```
File ".../tpu_inference/distributed/utils.py", line 125, AttributeError
RuntimeError: Unable to initialize backend 'tpu': Expected 2 worker addresses, got 1
```

**根因**: GKE auto-injection 给每个 multi-host pod 注入的是 single-host 视角 env vars：
- `TPU_WORKER_HOSTNAMES=localhost`（只有自己）
- `TPU_PROCESS_ADDRESSES=localhost:8471`
- `TPU_WORKER_ID=0`

JAX libtpu 看到 `Expected 2 worker addresses, got 1` 就 fallback 到 CPU device。

**调试方法**: 起 busybox pod 在同 node pool, `env | grep TPU` 看实际注入值：
```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: {name: tpu-env-probe}
spec:
  restartPolicy: Never
  nodeSelector: {cloud.google.com/gke-tpu-topology: 2x2x2}
  tolerations:
  - {effect: NoSchedule, key: cloud.google.com/gke-spot, operator: Equal, value: "true"}
  - {effect: NoSchedule, key: google.com/tpu, operator: Exists}
  containers:
  - name: probe
    image: busybox
    command: ["sh","-c","env | grep TPU; sleep 60"]
    resources: {limits: {google.com/tpu: "4"}}
EOF
sleep 8 && kubectl logs tpu-env-probe
kubectl delete pod tpu-env-probe --grace-period=0 --force
```

**修复（已在 yaml 里）**: 手动 export 覆盖（见 Step 2.2 yaml 的 KEY FIX block）。

### 坑 #6: Spot multi-host pool 容易被 preempt

**症状**: pool 状态从 RUNNING 突然变 RECONCILING, 几分钟后又回 RUNNING（不同 node ID）。
**Workaround**: 生产用 reservation；开发接受偶尔重启。

### 坑 #7: 不 warmup 时 c=1 数据严重低估

**症状**: 首跑 1K/1K c=1 = 21.5 tok/s, warmup 后 = 37.5 tok/s（差 75%）。
**原因**: 第一个请求触发 XLA padding bucket 编译，包含在测量时长内。
**修复**: benchmark 前先跑一轮 warmup（见 Step 5.1）。

### 坑 #8: Output tok/s vs Total tok/s 容易混淆

**症状**: 看到 "8K/1K c=4 = 856 tok/s" 以为很快，其实 output 只有 95.
**原因**: vllm bench 报告 `Total token throughput` 含 prefill input tokens, 长 prompt 会高估 5-10 倍。
**修复**: **始终看 `Output token throughput`**（这才是 generated 速度）。

---

## 资源清理

```bash
# 拆 LWS workload（保留 node pool 可立即重新部署）
kubectl delete lws vllm-mh
kubectl delete service vllm-mh-service

# === 仅当完全不再用 v7x-16 时执行 ===
gcloud container node-pools delete np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev --quiet
gcloud compute resource-policies delete chrisya-tpu7x-spot-mh \
  --region=us-central1 --project=cloud-tpu-multipod-dev --quiet
```

---

## 附录 A: 从零创建环境

> 仅当 [§1.1 测试环境](#11-gcp--gke-资源已存在不需要重建) 列出的资源**没有**时使用。

```bash
# A.1 装 LeaderWorkerSet
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.7.0/manifests.yaml

# A.2 创建 workload policy（topology 2x2x2 = 8 chips, cube 布局）
gcloud compute resource-policies create workload-policy chrisya-tpu7x-spot-mh \
  --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2 \
  --region=us-central1 --project=cloud-tpu-multipod-dev

# A.3 创建 multi-host node pool（3-5 min, spot 可能首次失败需要重试）
gcloud container node-pools create np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev \
  --machine-type=tpu7x-standard-4t --tpu-topology=2x2x2 \
  --num-nodes=2 --node-locations=us-central1-c \
  --disk-type=hyperdisk-balanced --disk-size=500 --spot \
  --max-pods-per-node=110 --image-type=COS_CONTAINERD \
  --workload-metadata=GKE_METADATA \
  --placement-policy=chrisya-tpu7x-spot-mh

# A.4 验证
kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology \
  | grep np-tpu7x-spot-mh
```

> ⚠️ Lustre PVC `lustre-pvc` 假定已就绪（这是 cluster 长期共享的 RWX 36 TB 卷）。
> 如果没有，需要先按 cluster 标准流程创建 Lustre 实例 + Lustre CSI driver, 这超出本文档范围。

---

## 附录 B: 完整启动时间分解（实测 2 次取平均）

| 阶段 | 首次冷启动 | 重启 (image cached) | 占比 |
|------|----------:|-------------------:|----:|
| Pod scheduling + Ray cluster | ~70 s (image pull) | ~10-20 s | 10% |
| JAX mesh init | ~60 s | ~60 s | 5% |
| 权重加载 (49 shards × ~10s) | **~480 s = 8 min** | **~480 s** | **60%** |
| XLA compile + warmup | 140-244 s | 140-244 s | 20% |
| **总冷启动** | **~12 min** | **~13 min** | — |

权重加载占大头（60%）。优化思路：PD checkpoint、weight prefetch、本地 SSD cache（取代 Lustre）。

---

## 参考资料

| 资源 | 链接 |
|------|------|
| 单机 README | [README.md](README.md) |
| LeaderWorkerSet docs | https://lws.sigs.k8s.io/ |
| `tpu_inference/distributed/utils.py:125` (坑 #5 源代码位置) | https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/distributed/utils.py#L125 |
| Google Codelabs (multihost vLLM + Ray, v6e 验证) | https://codelabs.developers.google.com/next26/aiinfra-learning-pod/screen2-advanced-inferencing-part-1 |
| HTML 详细报告 | https://cc.higcp.com/pages/qwen3-coder-480b-multihost-tpu-v7x-20260425.html |
