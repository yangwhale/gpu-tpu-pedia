**English** | [中文](./README-multinode.md)

# Qwen3-Coder-480B FP8 Multi-host Inference on TPU v7x-16 — Fully Reproducible Guide

> **Goal**: A newcomer who copies this document 100% verbatim can deploy a multi-host vLLM TP=16 inference service successfully on the first attempt and complete 5 sets of benchmark comparisons.
>
> **Status** (verified twice on 2026-04-26): ✅ Document is fully reproducible, but the **CRITICAL conclusion**: multi-host TP=16 single-instance output throughput is 17-63% worse than single-node v7x-8 across all scenarios, **not recommended for production** — customers should run 2 independent v7x-8 single-node instances (data parallel).
>
> Purpose of this document: (1) provide complete reproducible deployment steps; (2) give objective comparison data; (3) explain why this deployment mode is not suitable for Qwen3-Coder-480B.
>
> For the single-node version (v7x-8 + TP=8), see [README.md](README.md) in the same directory — **recommended for production**.

---

## 📋 Table of Contents

1. [Test Environment (ready / no rebuild needed)](#1-测试环境已就绪)
2. [Step 0: Check environment + switch kubectl context](#step-0-检查环境)
3. [Step 1: Clean up existing deployment (if any)](#step-1-清理现有部署)
4. [Step 2: Deploy LeaderWorkerSet](#step-2-部署-leaderworkerset)
5. [Step 3: Monitor startup to ready](#step-3-监控启动到-ready)
6. [Step 4: Smoke test](#step-4-smoke-test)
7. [Step 5: Benchmark (5 sets + warmup)](#step-5-benchmark)
8. [Core Conclusion: Multi-host vs single-node comparison](#核心结论multi-host-vs-单机对比)
9. [Pitfall Log (8 verified issues)](#踩坑实录)
10. [Resource Cleanup (after testing)](#资源清理)

---

## 1. Test Environment (ready)

### 1.1 GCP / GKE Resources (**already exist, no rebuild needed**)

| Resource Type | Name / Config | Status |
|---------|------------|------|
| GCP Project | `cloud-tpu-multipod-dev` | ✅ ready |
| GKE Cluster | `chrisya-v7x-v134` (us-central1, master 1.34.4-gke.1193000) | ✅ RUNNING |
| Node Pool | `np-tpu7x-spot-mh` (2 nodes × tpu7x-standard-4t, topology 2x2x2, spot, us-central1-c) | ✅ RUNNING |
| Workload Policy | `chrisya-tpu7x-spot-mh` (HIGH_THROUGHPUT, accelerator-topology=2x2x2) | ✅ ready |
| Lustre PVC | `lustre-pvc` (36 TB, ReadWriteMany) | ✅ Bound |
| LWS Operator | LeaderWorkerSet v0.7.0 (`lws-system` namespace) | ✅ Installed |

### 1.2 Model Weights (**already downloaded, no re-download needed**)

| Item | Value |
|------|---|
| Model | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` |
| HF cache path | `/lustre/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/000...aaaa` |
| Actual weight location | `/lustre/qwen3-coder-480b-fp8` (snapshot symlink points here) |
| Size | ~450 GB FP8 safetensors (49 shards) |

### 1.3 Hardware Specs

| Item | Value |
|------|---|
| TPU | v7x-16 (2 hosts × 4 chips = 8 chips · 16 devices) |
| HBM | total **1.5 TB** (192 GB/chip × 8) |
| Host memory | 850 GB / node |
| Network | DCN (ports 8471/6379 between nodes) |

### 1.4 Architecture Diagram

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

## Step 0: Check Environment

### 0.1 Switch kubectl context (**critical · easy to forget**)

> ⚠️ The current gcloud default project might be a different project (e.g. `gpu-launchpad-playground`), but the cluster is in `cloud-tpu-multipod-dev`. You must explicitly grab the correct kubeconfig.

```bash
# Pull cluster credentials (context name: gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134)
gcloud container clusters get-credentials chrisya-v7x-v134 \
  --region=us-central1 --project=cloud-tpu-multipod-dev

# Verify current context
kubectl config current-context
# Expected output: gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134
```

### 0.2 Verify ready resources

```bash
# 1. Node pool (should see 2 nodes, topology=2x2x2)
kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology \
  | grep np-tpu7x-spot-mh

# 2. LWS operator (should see the leaderworkerset CRD)
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io

# 3. Lustre PVC (should be STATUS=Bound)
kubectl get pvc lustre-pvc

# 4. Workload policy (should see acceleratorTopology: 2x2x2)
gcloud compute resource-policies describe chrisya-tpu7x-spot-mh \
  --region=us-central1 --project=cloud-tpu-multipod-dev | grep -A 1 workloadPolicy
```

> **If any of the above is not ready**, see [Appendix A: Create the environment from scratch](#附录-a-从零创建环境). This document assumes these are already ready.

---

## Step 1: Clean Up Existing Deployment

> If there is **no** `vllm-mh` LWS in the cluster, skip this step and go straight to Step 2.

```bash
# 1. Check whether an existing vllm-mh exists
kubectl get lws,svc,pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh

# 2. Tear down the LWS and Service
kubectl delete lws vllm-mh --ignore-not-found
kubectl delete service vllm-mh-service --ignore-not-found

# 3. Wait until all vllm-mh-* pods are fully terminated (~60-90s)
while [ "$(kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh --no-headers 2>/dev/null | wc -l)" != "0" ]; do
  echo "Waiting for pods to terminate..."
  sleep 10
done
echo "✅ Cleanup complete"
```

> ⚠️ **Do not tear down the node pool** — rebuilding a multi-host node pool is slow and spot capacity is often unavailable.

---

## Step 2: Deploy LeaderWorkerSet

### 2.1 Apply Service (headless, only exposes the leader)

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

### 2.2 Apply LeaderWorkerSet (**complete yaml · copy and run directly**)

> ⚠️ **Do not delete the highlighted KEY FIX block**: it manually overrides the GKE auto-injected single-host TPU env vars, otherwise it will get stuck on `AttributeError: d.coords` (pitfall #5).

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
    restartPolicy: Default            # ⚠️ Do not use RecreateGroupOnPodRestart; any restart triggers a rebuild of the entire group
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
          image: vllm/vllm-tpu:nightly-20260330-2f76400-8c0b626    # codelab-verified version, do not upgrade casually
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

            # === KEY FIX: override GKE auto-injected single-host TPU env vars ===
            # The GKE webhook injects TPU_WORKER_HOSTNAMES=localhost
            # and TPU_WORKER_ID=0 into every multi-host pod (it treats it as a single pod)
            # Without overriding, it gets stuck at vLLM init_device: AttributeError: d.coords
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
            export HF_HOME=/lustre               # weights are at /lustre/hub/models--Qwen--...
            export HF_HUB_OFFLINE=1              # offline
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

### 2.3 Verify apply succeeded

```bash
# Should see two pods: vllm-mh-0 (leader) and vllm-mh-0-1 (worker)
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-mh -w
# Wait until both are 1/1 Running (~20-70s), then Ctrl+C
```

---

## Step 3: Monitor Startup to Ready

### 3.1 Check boot env vars (verify KEY FIX took effect)

```bash
# Wait 5s for log output, then check
sleep 5
kubectl logs vllm-mh-0 -c main 2>&1 | grep "\[boot\]" | head -5
```

Expected output (IPs may differ):
```
[boot] My TPU IP: 10.120.6.7, LWS_WORKER_INDEX=0
[boot] LEADER_IP=10.120.6.7 WORKER_IP=10.120.12.6
[boot] TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=10.120.6.7,10.120.12.6
```

> ❌ If `TPU_WORKER_HOSTNAMES=localhost` → KEY FIX did not take effect, go back and check the yaml.
> ❌ If `LEADER_IP` does not appear for a long time → DNS did not resolve, check whether the LWS correctly created the service.

### 3.2 Wait for startup to complete (**~12-13 min**)

```bash
# Track key logs in real time ("Application startup complete" means ready)
kubectl logs vllm-mh-0 -c main -f | grep -E "Application startup complete|Init mesh|Loading weights took|safetensors checkpoint shards"
```

Expected key logs (in chronological order):

| Timestamp | Log | Phase |
|-------|------|------|
| ~T+11s | `Ray runtime started` | Ray cluster ✅ |
| ~T+74s | `Init mesh \| mesh=Mesh('data': 1, 'model': 16, axis_types=...)` | JAX mesh ✅ |
| T+74s ~ T+557s | `Loading safetensors checkpoint shards: X% Completed \| Y/49` | weight loading |
| ~T+557s (9.3 min) | `INFO ... Loading weights took 480.XX seconds` | weight load done ✅ |
| T+557s ~ T+800s | (no obvious log, this is XLA compilation) | XLA compile |
| ~T+800s (~13 min) | `INFO: Application startup complete.` | **Server ready ✅** |

> 💡 Measured startup time falls into two cases:
> - **First cold start** (image not pulled): ~12 min (pod scheduling 70s + the rest ~10 min)
> - **Restart** (image cached): ~13 min (pod scheduling only ~20s, but weight load 480s + XLA compile 244s still have to run again)
> - The XLA cache does not persist, so restarting is not much faster than first start

### 3.3 Verify server is ready

```bash
kubectl logs vllm-mh-0 -c main 2>&1 | grep -c "Application startup complete"
# Expected: 1 (appeared once)
```

---

## Step 4: Smoke Test

```bash
kubectl exec vllm-mh-0 -c main -- bash -c '
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen3-Coder-480B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍 TPU v7\"}],\"max_tokens\":100}"
'
```

Expected return: JSON containing `"role":"assistant","content":"<some Chinese text>"`, and you should see `"finish_reason":"stop"`.
The first request may take ~30s (XLA compilation); subsequent ones <2s.

---

## Step 5: Benchmark

> ⚠️ **Warm up first, then run the real test**. Without warmup, the c=1 data is underestimated by ~75% (the first request triggers XLA padding bucket compilation).
> ⚠️ **Look at Output token throughput**, not Total token throughput (the latter includes prefill input and overestimates by 5-10× for long prompts).

### 5.1 Warmup (~5 min)

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

### 5.2 5 sets of real benchmark (~15 min)

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

### 5.3 Expected data range (compared with this run's measurements)

| Scenario | Output tok/s | TTFT median | TPOT median | Notes |
|------|------------:|------------:|------------:|------|
| 1K/1K c=1 | 35-40 | 90-100 ms | 26-27 ms | after warmup |
| 1K/1K c=4 | 95-100 | 270-280 ms | 27-29 ms | |
| 1K/1K c=16 | 215-225 | 4200-4400 ms | 64-67 ms | TTFT high due to prefill batching |
| 8K/1K c=4 | 130-140 | 1180-1200 ms | 28-30 ms | |
| 8K/1K c=16 | 220-230 | 1180-1200 ms | 68-70 ms | |

> If the deviation exceeds ±10%, check: (1) whether warmup was sufficient; (2) whether the node was rescheduled due to spot preemption; (3) whether another workload is contending for TPU/CPU resources.

---

## Core Conclusion: Multi-host vs Single-node Comparison

### Output Throughput (tok/s) — the real generation speed

| Scenario | Single-node v7x-8 (TP=8) | Multi-host v7x-16 (TP=16) | Difference |
|------|------------------:|-------------------:|----:|
| 1K/1K c=1 | **48** | 37.5 | ❌ -22% |
| 1K/1K c=4 | **177** | 98.4 | ❌ -44% |
| 1K/1K c=16 | **602** | 220.0 | ❌ -63% |
| 8K/1K c=4 | **162** | 134.3 | ❌ -17% |
| 8K/1K c=16 | **483** | 223.1 | ❌ -54% |

### TTFT (median) — time to first byte

| Scenario | Single-node | Multi-host | Difference |
|------|----:|----:|----:|
| 8K/1K c=4 | 1495 ms | **1188 ms** | ✅ -20% |
| 8K/1K c=16 | 2418 ms | **1189 ms** | ✅ -51% |
| 1K/1K c=16 | 549 ms | 4306 ms | ❌ +684% |

### Why is multi-host slower?

1. **TP=16 across DCN**: every decode token requires an all-reduce across all 16 devices, and the cross-node DCN synchronization overhead >>> the doubled compute gain
2. **MoE expert routing across nodes**: top-k expert selection causes cross-host all-to-all
3. **Single-instance TP=16 is not the sweet spot**: the finer you slice TP, the higher the communication ratio

### ✅ Correct Production Recommendation

| Customer Scenario | Recommended Config | Rationale |
|---------|---------|------|
| **Default / most scenarios** | **2 × v7x-8 single-node (TP=8) Data Parallel** | total throughput ≈ 2× single-node = ~1200 tok/s @ c=16 |
| Long prompt, high concurrency | Single-node PD disaggregation 1P1D | TPOT improves by 11% |
| Only when a single instance + max-model-len ≥ 32K is required | Multi-host TP=16 (sacrifice throughput for HBM) | only 1.5 TB HBM can hold a large KV cache |

---

## Pitfall Log

### Pitfall #1: TPU topology 4x2x1 incompatible with tpu7x-standard-4t

**Symptom**: `Accelerator topology: 4x2x1 is not compatible with the provided machine type: tpu7x-standard-4t`
**Fix**: Use **`2x2x2`** (cube layout, same 8 chips).

### Pitfall #2: `ray --block &` does not work inside sh -c args

**Symptom**: Leader pod log stays stuck at `ray start --block`.
**Fix**: **Run the leader in daemon mode** (without `--block`), and the worker with `--block`.

### Pitfall #3: LWS `RecreateGroupOnPodRestart` is too aggressive

**Symptom**: Any pod restart triggers a recreate of the entire group, never stabilizing to reach the vllm load phase.
**Fix**: `restartPolicy: Default`.

### Pitfall #4: `--async-scheduling` incompatible with the ray executor

**Symptom**: `ValidationError: ray does not support async scheduling`
**Fix**: A multi-host (`--distributed-executor-backend=ray`) config cannot use `--async-scheduling`.

### Pitfall #5 (BLOCKER): `AttributeError: d.coords` on init_device

**Symptom**:
```
File ".../tpu_inference/distributed/utils.py", line 125, AttributeError
RuntimeError: Unable to initialize backend 'tpu': Expected 2 worker addresses, got 1
```

**Root cause**: GKE auto-injection injects single-host-perspective env vars into every multi-host pod:
- `TPU_WORKER_HOSTNAMES=localhost` (only itself)
- `TPU_PROCESS_ADDRESSES=localhost:8471`
- `TPU_WORKER_ID=0`

When JAX libtpu sees `Expected 2 worker addresses, got 1`, it falls back to the CPU device.

**Debug method**: Start a busybox pod in the same node pool, and run `env | grep TPU` to see the actual injected values:
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

**Fix (already in the yaml)**: manually export overrides (see the KEY FIX block in the Step 2.2 yaml).

### Pitfall #6: Spot multi-host pool is easily preempted

**Symptom**: The pool status suddenly changes from RUNNING to RECONCILING, then returns to RUNNING a few minutes later (with different node IDs).
**Workaround**: Use a reservation for production; accept occasional restarts in development.

### Pitfall #7: Without warmup, c=1 data is severely underestimated

**Symptom**: First run of 1K/1K c=1 = 21.5 tok/s, after warmup = 37.5 tok/s (75% difference).
**Cause**: The first request triggers XLA padding bucket compilation, which is included in the measured duration.
**Fix**: Run a round of warmup before benchmarking (see Step 5.1).

### Pitfall #8: Output tok/s vs Total tok/s is easily confused

**Symptom**: Seeing "8K/1K c=4 = 856 tok/s" makes you think it's fast, but the output is actually only 95.
**Cause**: vllm bench reports `Total token throughput` including prefill input tokens, which overestimates by 5-10× for long prompts.
**Fix**: **Always look at `Output token throughput`** (this is the actual generation speed).

---

## Resource Cleanup

```bash
# Tear down the LWS workload (keep the node pool so you can redeploy immediately)
kubectl delete lws vllm-mh
kubectl delete service vllm-mh-service

# === Execute only when v7x-16 will no longer be used at all ===
gcloud container node-pools delete np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev --quiet
gcloud compute resource-policies delete chrisya-tpu7x-spot-mh \
  --region=us-central1 --project=cloud-tpu-multipod-dev --quiet
```

---

## Appendix A: Create the Environment from Scratch

> Use only when the resources listed in [§1.1 Test Environment](#11-gcp--gke-资源已存在不需要重建) do **not** exist.

```bash
# A.1 Install LeaderWorkerSet
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.7.0/manifests.yaml

# A.2 Create the workload policy (topology 2x2x2 = 8 chips, cube layout)
gcloud compute resource-policies create workload-policy chrisya-tpu7x-spot-mh \
  --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2 \
  --region=us-central1 --project=cloud-tpu-multipod-dev

# A.3 Create the multi-host node pool (3-5 min, spot may fail on first attempt and need a retry)
gcloud container node-pools create np-tpu7x-spot-mh \
  --cluster=chrisya-v7x-v134 --region=us-central1 \
  --project=cloud-tpu-multipod-dev \
  --machine-type=tpu7x-standard-4t --tpu-topology=2x2x2 \
  --num-nodes=2 --node-locations=us-central1-c \
  --disk-type=hyperdisk-balanced --disk-size=500 --spot \
  --max-pods-per-node=110 --image-type=COS_CONTAINERD \
  --workload-metadata=GKE_METADATA \
  --placement-policy=chrisya-tpu7x-spot-mh

# A.4 Verify
kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology \
  | grep np-tpu7x-spot-mh
```

> ⚠️ The Lustre PVC `lustre-pvc` is assumed to be ready (this is the cluster's long-term shared RWX 36 TB volume).
> If it does not exist, you first need to create a Lustre instance + Lustre CSI driver following the cluster's standard procedure, which is beyond the scope of this document.

---

## Appendix B: Full Startup Time Breakdown (averaged over 2 measured runs)

| Phase | First cold start | Restart (image cached) | Share |
|------|----------:|-------------------:|----:|
| Pod scheduling + Ray cluster | ~70 s (image pull) | ~10-20 s | 10% |
| JAX mesh init | ~60 s | ~60 s | 5% |
| Weight load (49 shards × ~10s) | **~480 s = 8 min** | **~480 s** | **60%** |
| XLA compile + warmup | 140-244 s | 140-244 s | 20% |
| **Total cold start** | **~12 min** | **~13 min** | — |

Weight loading dominates (60%). Optimization ideas: PD checkpoint, weight prefetch, local SSD cache (replacing Lustre).

---

## References

| Resource | Link |
|------|------|
| Single-node README | [README.md](README.md) |
| LeaderWorkerSet docs | https://lws.sigs.k8s.io/ |
| `tpu_inference/distributed/utils.py:125` (source location of pitfall #5) | https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/distributed/utils.py#L125 |
| Google Codelabs (multihost vLLM + Ray, verified on v6e) | https://codelabs.developers.google.com/next26/aiinfra-learning-pod/screen2-advanced-inferencing-part-1 |
| HTML detailed report | https://cc.higcp.com/assets/qwen3-coder-480b-multihost-tpu-v7x-20260425.html |
