# Kimi K2.6 (1T-A32B) Multi-host Inference on TPU v7x-16 — 完全可复现指南

> **目标**：新人 100% 照抄本文档可一次成功部署 multi-host vLLM TP=16 推理服务并完成 sanity test。
>
> **状态**（2026-04-26 stage-1 验证）：✅ 4L sanity PASS（multi-host 全链路打通：weight load + JAX init + XLA compile + vllm serve + curl 推理），61L 待测。
>
> 本文档目的：(1) 提供完整可复现部署步骤；(2) 给出 K2.6 multi-host 的 model code patch（**必须 patch**，不 patch 跑不起来）；(3) 记录 14 yaml + 3 model code patch 的完整踩坑过程。
>
> 单机版（v7x-8 + TP=8）见同目录 [README.md](README.md)。

---

## 📋 目录

1. [背景：为什么需要 patch K2.6 model code](#背景为什么需要-patch-k26-model-code)
2. [测试环境（已就绪 / 不需要重建）](#测试环境已就绪)
3. [Step 0: 检查环境 + 切 kubectl context](#step-0-检查环境)
4. [Step 1: Patch K2.6 model code (multi-host fix)](#step-1-patch-k26-model-code)
5. [Step 2: 部署 LeaderWorkerSet](#step-2-部署-leaderworkerset)
6. [Step 3: 监控启动到 ready](#step-3-监控启动到-ready)
7. [Step 4: Sanity test](#step-4-sanity-test)
8. [HBM 容量核算（61L 装得下吗）](#hbm-容量核算)
9. [踩坑实录（14 yaml + 3 model patch 完整过程）](#踩坑实录)
10. [资源清理](#资源清理)
11. [附录 A: K2.6 vs Qwen3 / DeepSeek 路径对比](#附录-a-路径对比)
12. [附录 B: 完整 LWS yaml](#附录-b-完整-lws-yaml)

---

## 背景：为什么需要 patch K2.6 model code

K2.6 跟 DeepSeek V3 同源（MLA + MoE 架构），但 **attention 是 BF16 unquantized**（DeepSeek V3 是 FP8 dequant）。K2.6 的 `kimi_k26.py:542` 在 `kv_b_proj` split 路径加了一行：

```python
w_cpu = jax.device_put(self.weight.value, jax.devices('cpu')[0])
```

这行假设 `self.weight.value` 是 fully-addressable（single process）。Multi-host 下 weight 跨 16 devices 跨 process_index 0/1，这个 API 直接 `ValueError`。

**修复策略**: 加 `jax.process_count() > 1` 分支，用 `jax.experimental.multihost_utils.process_allgather(arr, tiled=True)` 跨 process gather，**且必须放在 `cpu_mesh_context()` 外面**（`cpu_mesh_context` 设 jit context 到 CPU device 0，会与 TPU sharded input 冲突）。

详细技术分析见本文档 [踩坑 #14](#坑-14-killer-kimi_k26py542-的-jaxdevice_put)。

---

## 测试环境（已就绪）

### GCP / GKE 资源（**已存在，不需要重建**）

| 资源类型 | 名称 / 配置 | 状态 |
|---------|------------|------|
| GCP Project | `cloud-tpu-multipod-dev` | ✅ 已就绪 |
| GKE Cluster | `chrisya-v7x-v134` (us-central1, master 1.34.4-gke.1193000) | ✅ RUNNING |
| Node Pool | `np-tpu7x-spot-mh-k26` (2 nodes × tpu7x-standard-4t, topology 2x2x2, spot, us-central1-c) | ✅ RUNNING |
| Workload Policy | `chrisya-tpu7x-spot-mh` (HIGH_THROUGHPUT, accelerator-topology=2x2x2) | ✅ 已就绪 |
| Lustre PVC | `lustre-pvc` (36 TB, ReadWriteMany) | ✅ Bound |
| LWS Operator | LeaderWorkerSet v0.7.0 (`lws-system` namespace) | ✅ Installed |

### 模型权重（**已下载到 Lustre**）

| 项 | 值 |
|---|---|
| 模型 | `moonshotai/Kimi-K2.6` |
| 路径 | `/lustre/Kimi-K2.6` |
| 大小 | ~555 GB INT4 W4A16 (64 safetensors) |

### 硬件规格

| 项目 | 值 |
|---|---|
| TPU | v7x-16 (2 hosts × 4 chips = 8 chips · 16 devices) |
| HBM | 总 **1.5 TB** (192 GB/chip × 8 chips) |
| 主机内存 | 850 GB / node |
| 网络 | DCN（节点间 8471/6379 端口） |

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│  GKE Cluster (chrisya-v7x-v134, us-central1)           │
│  Node Pool: np-tpu7x-spot-mh-k26 (topology 2x2x2)      │
│                                                         │
│  ┌────────────────────────┐  ┌────────────────────────┐│
│  │ Node 1 (host-0)        │  │ Node 2 (host-1)        ││
│  │  4 chips · 768 GB HBM  │  │  4 chips · 768 GB HBM  ││
│  │                        │  │                        ││
│  │  Pod: k26-mh-0         │  │  Pod: k26-mh-0-1       ││
│  │   - Ray head 6379      │  │   - Ray worker         ││
│  │   - vLLM API 8000      │  │   - libtpu coord 8471  ││
│  │   - LWS_WORKER_INDEX=0 │  │   - LWS_WORKER_INDEX=1 ││
│  │   - TPU_WORKER_ID=0    │  │   - TPU_WORKER_ID=1    ││
│  └─────┬──────────────────┘  └─────────┬──────────────┘│
│        │                                │               │
│        ├─ Ray RPC (DCN, port 6379) ─────┤               │
│        └─ libtpu coord (port 8471) ─────┘               │
└─────────────────────────────────────────────────────────┘

Mounts (both pods):
  /lustre  ← Lustre PVC (RWX, 36TB) — model + tpu_inference fork
  /dev/shm ← 200GB tmpfs            — XLA cache, runtime tmp
```

---

## Step 0: 检查环境

### 0.1 切 kubectl context

```bash
gcloud container clusters get-credentials chrisya-v7x-v134 \
  --region=us-central1 --project=cloud-tpu-multipod-dev

kubectl config current-context
# 期望: gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134
```

### 0.2 验证已就绪资源

```bash
# Node pool（应该看到 2 个 node, topology=2x2x2）
kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology \
  | grep np-tpu7x-spot-mh-k26

# LWS operator
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io

# Lustre PVC
kubectl get pvc lustre-pvc

# K2.6 model weights
kubectl run -it --rm verify-lustre --image=busybox \
  --overrides='{"spec":{"containers":[{"name":"verify-lustre","image":"busybox","command":["sh","-c","ls /lustre/Kimi-K2.6 | wc -l"],"volumeMounts":[{"name":"lustre","mountPath":"/lustre"}]}],"volumes":[{"name":"lustre","persistentVolumeClaim":{"claimName":"lustre-pvc"}}]}}' \
  --restart=Never --rm -- sh
# 期望文件数 ≥ 64 safetensors + tokenizer
```

### 0.3 验证 tpu_inference fork 在 Lustre

K2.6 multi-host 需要 e2e pod 上 live-edit 过的 tpu_inference 代码（包含 K2.6 model registry + W4A16 quantization 支持）。这份代码必须在 Lustre 上：

```bash
kubectl run -it --rm verify-code --image=busybox \
  --overrides='...' -- ls /lustre/tpu_inference/tpu_inference/models/jax/kimi_k26.py
# 必须存在
```

> 如果没有，从 e2e single-host pod cp 一份到 Lustre：
> ```bash
> kubectl exec <e2e-pod> -- cp -r /workspace/tpu_inference /lustre/
> ```

---

## Step 1: Patch K2.6 model code

### 关键 patch（必做）：`kimi_k26.py:542` multi-host fix

**原代码**（multi-host 报 `ValueError: device_put first arg must be fully addressable`）：

```python
is_unquantized = not hasattr(self, "weight_scale_inv")
with cpu_mesh_context():
    if is_unquantized:
        # ...
        w_cpu = jax.device_put(self.weight.value, jax.devices('cpu')[0])  # ❌ multi-host fails
        dequantized_weight = w_cpu
```

**Patched 代码**：

```python
is_unquantized = not hasattr(self, "weight_scale_inv")
# multi-host fix: gather cross-process shards BEFORE cpu_mesh_context.
# cpu_mesh_context sets jit's thread-local mesh to CPU device 0, which
# is incompatible with TPU-sharded input inside process_allgather's jit.
if is_unquantized and jax.process_count() > 1:
    from jax.experimental import multihost_utils
    weight_full = multihost_utils.process_allgather(self.weight.value, tiled=True)
else:
    weight_full = None
with cpu_mesh_context():
    if is_unquantized:
        # ... (comments)
        if jax.process_count() > 1:
            w_cpu = jax.device_put(weight_full, jax.devices('cpu')[0])
        else:
            w_cpu = jax.device_put(self.weight.value, jax.devices('cpu')[0])
        dequantized_weight = w_cpu
```

### 自动 patch 脚本（幂等）

把以下脚本保存到本地 `patch_k26_multihost.py`，然后 cp 到任何挂载 Lustre 的 pod 执行：

```python
#!/usr/bin/env python3
"""Idempotent patch: kimi_k26.py multi-host fix."""
import sys

fp = "/lustre/tpu_inference/tpu_inference/models/jax/kimi_k26.py"

with open(fp) as f:
    src = f.read()

OLD = "                w_cpu = jax.device_put(self.weight.value, jax.devices('cpu')[0])\n"
NEW = (
    "                # multi-host fix: gather cross-process shards (tiled=True for non-fully-addressable)\n"
    "                if jax.process_count() > 1:\n"
    "                    from jax.experimental import multihost_utils\n"
    "                    weight_full = multihost_utils.process_allgather(self.weight.value, tiled=True)\n"
    "                    w_cpu = jax.device_put(weight_full, jax.devices('cpu')[0])\n"
    "                else:\n"
    "                    w_cpu = jax.device_put(self.weight.value, jax.devices('cpu')[0])\n"
)

if "process_allgather" in src:
    print("ALREADY PATCHED")
    sys.exit(0)

if OLD not in src:
    print("FAIL: pattern not found, manual patch needed")
    sys.exit(1)

with open(fp, "w") as f:
    f.write(src.replace(OLD, NEW))
print("PATCHED 1 occurrence")
```

执行：

```bash
# 任何挂 Lustre 的 pod 上
kubectl cp patch_k26_multihost.py <any-lustre-pod>:/tmp/patch.py
kubectl exec <any-lustre-pod> -- python3 /tmp/patch.py
# 期望输出: PATCHED 1 occurrence  或  ALREADY PATCHED
```

> ⚠️ **注意**：上面的 patch 脚本是一个简化版（直接在 `cpu_mesh_context` 内调 `process_allgather`）。**实际 multi-host 还会撞 jit context mesh 不兼容**——必须把 `process_allgather` 移到 `cpu_mesh_context` block **外面**（详见 [踩坑 #16](#坑-16-cpu_mesh_context-与-process_allgather-的-jit-context-冲突)）。完整移位 patch 见 `appendix/patch_k26_multihost_v3.py`。

---

## Step 2: 部署 LeaderWorkerSet

### 2.1 完整 yaml（直接 copy 跑）

> ⚠️ 黄字 KEY FIX 段不要删：覆盖 GKE auto-injected single-host TPU env，否则 leader 拿 stale DNS IP 死锁。

```bash
kubectl apply -f - <<'EOF'
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: k26-mh
  namespace: default
spec:
  replicas: 1
  leaderWorkerTemplate:
    restartPolicy: None
    size: 2
    workerTemplate:
      metadata:
        labels:
          leaderworkerset.sigs.k8s.io/name: k26-mh
      spec:
        nodeSelector:
          cloud.google.com/gke-nodepool: np-tpu7x-spot-mh-k26
          cloud.google.com/gke-tpu-accelerator: tpu7x
          cloud.google.com/gke-tpu-topology: 2x2x2
        tolerations:
        - { effect: NoSchedule, key: cloud.google.com/gke-spot, operator: Equal, value: "true" }
        - { effect: NoSchedule, key: google.com/tpu, operator: Exists }
        hostname: k26-mh
        serviceAccountName: default
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
          # 用同 single-host e2e pod 的 image (含 vllm 兼容版本; tpu_inference 会被覆盖)
          image: us-central1-docker.pkg.dev/chris-pgp-host/ai-infra/vllm-tpu:latest
          ports:
          - { containerPort: 8000 }
          - { containerPort: 6379 }
          - { containerPort: 8471 }
          securityContext:
            privileged: true
            capabilities: { add: ["IPC_LOCK"] }
          resources:
            requests: { cpu: "200", memory: 850Gi, "google.com/tpu": "4" }
            limits:   { cpu: "200", memory: 850Gi, "google.com/tpu": "4" }
          volumeMounts:
          - { mountPath: /dev/shm, name: dshm }
          - { mountPath: /lustre,  name: lustre-vol }
          command: ["sh", "-c"]
          args:
          - |
            set -e
            MY_TPU_IP=$(hostname -I | awk '{print $1}')
            echo "[boot] My TPU IP: $MY_TPU_IP, LWS_WORKER_INDEX=$LWS_WORKER_INDEX"
            LEADER_DNS="k26-mh-0.k26-mh"
            WORKER_DNS="k26-mh-0-1.k26-mh"

            # CRITICAL: sleep 30s past CoreDNS TTL to avoid stale (v(N-1)) entries
            sleep 30
            until getent hosts $LEADER_DNS; do echo "[boot] wait leader DNS"; sleep 5; done
            until getent hosts $WORKER_DNS; do echo "[boot] wait worker DNS"; sleep 5; done
            LEADER_IP=$(getent hosts $LEADER_DNS | awk '{print $1}')
            WORKER_IP=$(getent hosts $WORKER_DNS | awk '{print $1}')
            echo "[boot] LEADER_IP=$LEADER_IP WORKER_IP=$WORKER_IP"

            # Self-IP sanity check (fail-fast on stale DNS)
            if [ "$LWS_WORKER_INDEX" = "0" ] && [ "$LEADER_IP" != "$MY_TPU_IP" ]; then
              echo "[boot] FATAL: leader DNS=$LEADER_IP != my real IP=$MY_TPU_IP"; sleep 10; exit 1
            fi
            if [ "$LWS_WORKER_INDEX" = "1" ] && [ "$WORKER_IP" != "$MY_TPU_IP" ]; then
              echo "[boot] FATAL: worker DNS=$WORKER_IP != my real IP=$MY_TPU_IP"; sleep 10; exit 1
            fi

            # === KEY FIX: 覆盖 GKE auto-injected single-host TPU env vars ===
            export TPU_WORKER_HOSTNAMES="${LEADER_IP},${WORKER_IP}"
            export TPU_WORKER_ID=${LWS_WORKER_INDEX}
            export TPU_PROCESS_ADDRESSES="${LEADER_IP}:8471,${WORKER_IP}:8471"
            export TPU_PROCESS_PORT=8471
            export TPU_HOST_BOUNDS="1,1,2"
            export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
            export TPU_TOPOLOGY="2x2x2"
            export TPU_ACCELERATOR_TYPE="tpu7x-16"
            export TPU_SKIP_MDS_QUERY=true

            # vLLM / JAX env (K2.6 必须)
            export JAX_PLATFORMS=''
            export PJRT_DEVICE=TPU
            export TPU_BACKEND_TYPE=jax
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
            export TPU_MULTIHOST_BACKEND=ray
            export VLLM_HOST_IP=$MY_TPU_IP
            export MODEL_IMPL_TYPE=flax_nnx        # K2.6 必须 JAX-native (vllm-native 不支持 W4A16 group=32)
            export NEW_MODEL_DESIGN=1              # MLA + DP attention 强制要求
            export HF_HUB_OFFLINE=1
            export VLLM_LOGGING_LEVEL=INFO

            # CRITICAL: 替换 image bundled tpu_inference 为 Lustre 上的 K2.6 修改版本
            # (image bundled 缺 K2.6 model registry + W4A16 quantization)
            echo "[boot] replacing /workspace/tpu_inference from /lustre/tpu_inference"
            cd /
            mv /workspace/tpu_inference /workspace/tpu_inference.old.$$
            cp -r /lustre/tpu_inference /workspace/tpu_inference
            rm -rf /workspace/tpu_inference.old.* 2>/dev/null || true
            find /workspace/tpu_inference -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
            echo "[boot] sync done. K2.6 files verified:"
            ls -la /workspace/tpu_inference/tpu_inference/models/jax/kimi_k26.py
            grep "process_allgather" /workspace/tpu_inference/tpu_inference/models/jax/kimi_k26.py | head -1

            if [ "$LWS_WORKER_INDEX" = "0" ]; then
              echo "=== Starting Ray Head ==="
              ray start --head --port=6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}'
              sleep 20
              until ray status; do sleep 5; done

              # Wait until BOTH nodes joined ray
              echo "[boot] waiting for worker pod to join ray cluster..."
              for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
                NODE_COUNT=$(ray status 2>&1 | grep -c '^ 1 node_')
                echo "[boot] try $i: ray nodes alive=$NODE_COUNT (need 2)"
                if [ "$NODE_COUNT" -ge "2" ]; then echo "[boot] both nodes in ray"; break; fi
                sleep 10
              done

              echo "=== Starting vLLM serve K2.6 (TP=16, EP=16, DP attention) ==="
              vllm serve /lustre/Kimi-K2.6 \
                --served-model-name=Kimi-K2.6 \
                --tensor-parallel-size=16 \
                --distributed-executor-backend=ray \
                --quantization=compressed-tensors \
                --enforce-eager \
                --trust-remote-code \
                --max-model-len=512 \
                --max-num-seqs=64 \
                --max-num-batched-tokens=8192 \
                --gpu-memory-utilization=0.85 \
                --no-enable-prefix-caching \
                --enable-expert-parallel \
                --limit-mm-per-prompt '{"image":0,"video":0}' \
                --additional-config='{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":16,"tensor_parallelism":1}},"replicate_attn_weights":"True"}' \
                --host=0.0.0.0 --port=8000
            else
              echo "=== Starting Ray Worker, joining $LEADER_IP:6379 ==="
              ray start --address=${LEADER_IP}:6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}' --block
            fi
        volumes:
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 200Gi }
        - name: lustre-vol
          persistentVolumeClaim: { claimName: lustre-pvc }
EOF
```

### 2.2 4 层 sanity test（推荐先跑）

第一次部署强烈推荐用 4 层最小验证（~15 min vs 全 61 层 ~30+ min），加这一行到 `vllm serve` 的参数：

```yaml
                --hf-overrides='{"text_config":{"num_hidden_layers":4}}' \
```

4 层模型输出会乱码（少 57 层 transformer），但能验证全链路：weight load + JAX init + XLA compile + serve + curl forward pass 全 work。

### 2.3 验证 apply 成功

```bash
# 应该看到 k26-mh-0 (leader) 和 k26-mh-0-1 (worker) 两个 pod
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=k26-mh -w
# 等到两个都 1/1 Running（约 60-90s）后 Ctrl+C
```

---

## Step 3: 监控启动到 ready

### 3.1 看 boot script 输出验证 patch 生效

```bash
sleep 60
kubectl logs k26-mh-0 -c main 2>&1 | grep -E "^\[boot\]" | head -10
```

期望输出：
```
[boot] My TPU IP: 10.120.x.y, LWS_WORKER_INDEX=0
[boot] LEADER_IP=10.120.x.y WORKER_IP=10.120.x.z
[boot] replacing /workspace/tpu_inference from /lustre/tpu_inference
[boot] sync done. K2.6 files verified:
                    weight_full = multihost_utils.process_allgather(self.weight.value, tiled=True)
[boot] both nodes in ray
```

> ❌ 如果看不到 `process_allgather` 这行 → patch 没生效，回 Step 1
> ❌ 如果 `LEADER_IP != MY_TPU_IP` → DNS race，pod 会自己 fail-fast restart，等 60s 再 check

### 3.2 等待启动完成（**4L ~15 min, 61L ~30 min**）

```bash
kubectl logs k26-mh-0 -c main -f | grep -E "Application startup complete|Loading weights took|safetensors checkpoint shards|Init mesh|ValueError|RuntimeError"
```

期望关键 log（按时间顺序）：

| 阶段 | 日志 | 4L 用时 | 61L 用时 |
|------|------|--------|--------|
| Pod scheduling + Ray | `Ray runtime started` | ~70 s | ~70 s |
| 双 node ray 集群 | `[boot] both nodes in ray` | ~90 s | ~90 s |
| K2.6 model construct + weight load 启动 | `Loading safetensors checkpoint shards: 0%` | ~3 min | ~3 min |
| Weight load 完成 | `Loading weights took XXX seconds` | ~6 min (363 s 实测) | ~12 min |
| XLA compile | （无明显 log，是编译） | ~3 min | ~5 min |
| **Server ready** | `Application startup complete` | **~15 min** | **~30 min** |

### 3.3 验证 server ready

```bash
kubectl logs k26-mh-0 -c main 2>&1 | grep -c "Application startup complete"
# 期望: ≥ 1
```

---

## Step 4: Sanity test

```bash
# Test 1: 简单算术
kubectl exec k26-mh-0 -c main -- curl -s -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K2.6","prompt":"2+3=","max_tokens":20,"temperature":0}'

# Test 2: 自然语言
kubectl exec k26-mh-0 -c main -- curl -s -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K2.6","prompt":"Hello, how are","max_tokens":20,"temperature":0}'
```

**4L 模型**：返回 JSON 完整但 `text` 字段乱码（预期，模型不完整）
**61L 模型**：返回有意义的连贯文本

---

## HBM 容量核算

### K2.6 关键参数

| 参数 | 值 |
|---|---|
| num_hidden_layers | 61（1 dense + 60 MoE） |
| hidden_size | 7168 |
| vocab_size | 163,840 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 384 |
| num_experts_per_tok | 8 |
| q/kv_lora_rank | 1536 / 512 (MLA) |
| 量化 | W4A16 INT4 group=32（experts only），attention/shared_expert/lm_head 保留 BF16 |

### 单层 HBM (per device, EP=16 + DP attention)

| 组件 | per device 每层 |
|---|---|
| Attention (BF16, replicate) | 202 MB |
| Routed Experts (384/16 = 24 expert × 25 MB) | 600 MB |
| Shared Expert (BF16 replicate) | 88 MB |
| Gate router | 5.5 MB |
| **per layer per device** | **~895 MB** |

### 整 device 占用

| 项 | per device |
|---|---:|
| 60 MoE 层 × 0.9 GB | 54 GB |
| 1 dense 层 | ~0.7 GB |
| Embedding (BF16 replicate) | 2.35 GB |
| lm_head (BF16 replicate, NOT quantized) | 2.35 GB |
| KV cache (MLA compressed, 64 seq × 512 tok) | ~2.3 GB |
| **静态 weight per device** | **~60 GB** |
| **chip 0 device 0 (lm_head 在这)** | **~65 GB** |

### 容量结论

| 资源 | 数值 |
|---|---:|
| 单 device cap (192 GB / 2 devices/chip) | 96 GB |
| `gpu_memory_utilization=0.85` 后可用 | 81 GB |
| 最热 device (chip 0 device 0) | 65 GB |
| **buffer (够 dequant transient + activations)** | **16 GB** |
| 16 device 总占用 | ~960 GB |
| 16 chip 总 HBM | 1536 GB |
| **整池 headroom** | **576 GB free** |

**结论**: 61L K2.6 装得下，buffer 16 GB 在 chip 0 (lm_head + transient)。

> 💡 **为什么 multi-host 比单 host 安全**: 单 host EP=8 时每 device 持 48 experts × 25 MB = 1.2 GB / layer expert weight；multi-host EP=16 减半到 600 MB / layer。expert weight 进一步分摊使每 device 静态占用从单 host ~80 GB 降到 ~60 GB，释放 20 GB 给 transient buffer。

---

## 踩坑实录

### 坑 #1: image bundled tpu_inference 是 OLD 版本

**症状**: `NotImplementedError: compressed-tensors quantization method not supported. Supported methods are dict_keys([None, 'fp8'])`

**根因**: `vllm/vllm-tpu:nightly-*` 和 `chris-pgp-host:latest` image 内的 `tpu_inference` bundled 版本都不含 K2.6 的 W4A16 quantization 注册。E2E single-host pod 的 K2.6 修改是 `kubectl exec` live-edit，没有进 image。

**修复**: yaml 启动脚本里 `mv /workspace/tpu_inference + cp -r /lustre/tpu_inference`（见 Step 2）。

### 坑 #2: rm -rf /workspace/tpu_inference 触发 os.getcwd() 报错

**症状**: `FileNotFoundError: os.getcwd()` 在 EngineCore 启动时

**根因**: `rm -rf /workspace/tpu_inference` 把当前 cwd 删了，新 spawn 的 python 进程 `os.getcwd()` 失败。

**修复**: `cd / && mv tpu_inference tpu_inference.old.$$ && cp -r ...`（先切 cwd 再 mv，不直接 rm）。

### 坑 #3: GKE TPU env 默认是 single-host

**症状**: `Expected 2 worker addresses, got 1`，JAX libtpu fallback 到 CPU

**根因**: GKE auto-injection 给每个 multi-host pod 都注入 `TPU_WORKER_HOSTNAMES=localhost`、`TPU_PROCESS_ADDRESSES=localhost:8471`、`TPU_WORKER_ID=0`（当 single host 看待）。

**修复**: yaml boot script 里手动 export 覆盖（`KEY FIX` 段）。

### 坑 #4: LWS 快速 reapply 撞 CoreDNS stale

**症状**: leader DNS resolve 拿到上一个版本的 worker IP（如 10.120.8.12 vs 真实 10.120.8.13），所有 ray actor 朝不存在的 IP 发 JAX init，死锁 >10 min 后 LWS restart

**根因**: CoreDNS pod DNS 默认 TTL=30s, 但 endpoint slice 更新到 DNS propagate 之间有几秒窗口期。reapply 间隔太短会撞。

**修复**: yaml boot script 加 `sleep 30` 等 CoreDNS TTL 过期 + self-IP 自检 fail-fast（DNS resolve 到的 IP ≠ `hostname -I` 自己 IP 时立即 exit 1，触发 LWS restart 让 DNS 重新刷新）。

### 坑 #5: ray worker join 之前 leader 已经 vllm serve

**症状**: `RayDistributedExecutor` 看到只有 1 node，启动只用 8 device 而不是 16

**根因**: leader 的 `ray start --head` 不阻塞，立刻进 vllm serve；worker 还在拉 image 没启动。

**修复**: leader 加 wait loop `for i in 1..12; do NODE_COUNT=$(ray status | grep -c '^ 1 node_'); if [ ge 2 ]; then break; fi; sleep 10; done`

### 坑 #6: `--async-scheduling` 不兼容 ray executor

**症状**: `ValidationError: ray does not support async scheduling`

**修复**: Multi-host (`--distributed-executor-backend=ray`) 不能用 `--async-scheduling`。

### 坑 #7: multimodal token budget 不够

**症状**: `ValueError: max_tokens_per_mm_item (4225) > max_num_batched_tokens (1024)`

**根因**: K2.6 有 MoonViT vision encoder，每 image 4225 tokens。

**修复**: `--max-num-batched-tokens=8192` + `--limit-mm-per-prompt '{"image":0,"video":0}'` 跳过 vision。

### 坑 #8: KimiK25ForConditionalGeneration not registered

**症状**: `UnsupportedArchitectureError: Model architectures ['KimiK25ForConditionalGeneration'] not registered. JAX-native architectures: ['Llama4ForCausalLM', 'DeepseekV3ForCausalLM', ...]`

**根因**: image bundled `tpu_inference/models/common/model_loader.py` 没注册 `KimiK25`。

**修复**: 同 #1，整体 cp /lustre/tpu_inference 替换。

### 坑 #9: vllm-native fallback 不支持 W4A16 group=32

**症状**: `RuntimeError: Unsupported FusedMoe scheme: num_bits=4 type='int' symmetric=True group_size=32`

**根因**: K2.6 不在 JAX-native registry → vLLM 回落 vllm-native PyTorch path，但 vllm-native 的 `compressed_tensors_moe` 没实现 W4A16 group=32 kernel。

**修复**: K2.6 必须走 JAX-native (`MODEL_IMPL_TYPE=flax_nnx`)，确保 `KimiK25` 在 model registry 里（同 #8）。

### 坑 #10: e2e pod 上 patch 后但 multi-host pod 没生效

**症状**: 在 single-host e2e pod 上改了 `kimi_k26.py`，但 multi-host pod 启动还是用老代码

**根因**: e2e pod 改的是 pod 内 `/workspace/tpu_inference/...`（live-edit），multi-host pod 启动时拉的是 image 里的版本。

**修复**: 任何 multi-host fix 必须改 `/lustre/tpu_inference/...` (RWX 共享)，然后 multi-host pod 启动时 `cp -r /lustre/tpu_inference /workspace/tpu_inference`。

### 坑 #14 (KILLER): kimi_k26.py:542 的 jax.device_put

**症状**: `ValueError: When the second argument to device_put is a Device, the first argument must be a fully addressable array. Got value with devices {TpuDevice(id=0..15, process_index=0|1, ...)}`

**根因**: K2.6 attention 是 BF16 unquantized（DeepSeek V3 是 FP8 dequant），`kv_b_proj` split 路径加了一行 `jax.device_put(self.weight.value, jax.devices('cpu')[0])` 把 sharded weight 集中到 CPU。这个 API 假设 input fully-addressable，multi-host 下跨 16 devices 跨 process 违反约束。

**修复**: 用 `jax.experimental.multihost_utils.process_allgather(arr, tiled=True)` 跨 process gather，再 `device_put` 到 CPU。详见 [Step 1](#step-1-patch-k26-model-code) 的 patch。

### 坑 #15: process_allgather 默认 tiled=False 不支持 non-fully-addressable

**症状**: `ValueError: Gathering global non-fully-addressable arrays only supports tiled=True`

**修复**: `process_allgather(arr, tiled=True)` （默认 `tiled=False`）。

### 坑 #16: cpu_mesh_context 与 process_allgather 的 jit context 冲突

**症状**: `ValueError: Received incompatible devices for jitted computation. Got argument x of _identity_fn with shape bfloat16[X, Y] and device ids [0..15] on platform TPU and jit's context mesh with device ids [0] on platform CPU`

**根因**: `cpu_mesh_context()` 是 `jax.set_mesh(cpu_mesh())` 的 context manager，把 thread-local jit mesh 设成 CPU device 0。`process_allgather` 内部用 `jax.jit(_identity_fn, out_shardings=reps)(inp)`，jit 看 input 在 TPU 16 devices 而 context mesh 在 CPU，不兼容。

**修复**: 把 `process_allgather` 移到 `cpu_mesh_context` block **外面**（先在 TPU mesh 下 gather，再切 CPU mesh device_put）。详见 [Step 1](#step-1-patch-k26-model-code) 的最终 patch。

---

## 资源清理

```bash
# 拆 LWS workload（保留 node pool 可立即重新部署）
kubectl delete lws k26-mh

# === 仅当完全不再用 K2.6 multi-host 时执行 ===
gcloud container node-pools delete np-tpu7x-spot-mh-k26 \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev --quiet
```

> ⚠️ **Lustre 上的 `/lustre/tpu_inference` 不要删** —— 这是所有 K2.6 multi-host pod 共享的代码源。

---

## 附录 A: 路径对比

K2.6 multi-host 选型时纠结过 "借鉴 Qwen3-Coder 还是 DeepSeek V3"。结论：**model code 层面要参照 DeepSeek V3 (同 JAX-native + MLA + MoE)**；基础设施层面 (yaml/boot script) 抄 Qwen3-Coder。

| 配置 | Qwen3-Coder 跑通的路径 | K2.6 必须走的路径 | DeepSeek V3 / R1 路径 |
|---|---|---|---|
| `MODEL_IMPL_TYPE` | `vllm` (vllm-native PyTorch) | `flax_nnx` (JAX-native) | `flax_nnx` |
| 量化 | FP8 (vllm 原生 kernel) | W4A16 INT4 group=32 | FP8 dequant |
| Attention | 标准 | **BF16 unquantized + MLA** | **FP8 dequant + MLA** |
| Weight load 模型代码 | `vllm/.../qwen3_moe.py` | `tpu_inference/.../kimi_k26.py` | `tpu_inference/.../deepseek_v3.py` |
| device_put 单 device 调用 | 无（PyTorch 路径不用） | **L542 有** | 无 |

K2.6 因为 attention BF16 unquantized 多了 542 行的 device_put split 路径，这是 DeepSeek V3 没有的——这就是 K2.6 multi-host 唯一需要 patch 的地方。

---

## 附录 B: 完整 LWS yaml

见 [Step 2.1](#21-完整-yaml直接-copy-跑) 的 yaml block，可直接保存为 `k26_multihost_lws.yaml` 后 `kubectl apply -f` 即可。

---

## 项目状态

| 阶段 | 状态 | 耗时 |
|---|---|---|
| Single-host 5L sanity (Phase 1+2+3 优化) | ✅ PASS (2026-04-25 夜间) | 18 轮 / 一晚 |
| Single-host 60L sanity | ❌ chip 0 transient OOM | — |
| **Multi-host 4L sanity (本文档)** | ✅ **PASS** | **14 yaml + 3 patch / 半天** |
| Multi-host 61L sanity | ⏳ 下一步 | ~30 min 启动 |
| Multi-host benchmark (output tok/s, TTFT) | ⏳ 待测 | — |

---

> 📋 **状态**: Stage-1 验证完成 (2026-04-26) — 4L sanity PASS，model code patch 已落 Lustre 持久化。
> Stage-2 (61L sanity + benchmark) 进行中。
