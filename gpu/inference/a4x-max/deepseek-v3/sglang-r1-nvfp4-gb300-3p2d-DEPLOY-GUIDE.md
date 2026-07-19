# GB300 (A4X Max) · SGLang DeepSeek-R1-NVFP4 · 3P2D 端到端复现指南

> 一次性、事无巨细、可复制执行的部署手册。目标：在 GB300 NVL72 (GKE) 上把 DeepSeek-R1-0528-NVFP4 用 SGLang **PD 分离（3 prefill + 2 decode，DEP8 Wide-EP）** 端到端跑起来，KV cache 走**域内 NVLink**。
>
> 本文是 [`sglang-r1-nvfp4-128k-gb300-RUNLOG.md`](./sglang-r1-nvfp4-128k-gb300-RUNLOG.md) 里 20+ 小时趟坑后的**干净沉淀版**——RUNLOG 记录所有失败尝试，本文只留一条走得通的路。
>
> **实测结果（3P2D / 20 GPU / ctx 8192 / random 1024in-512out）**：并发 8 时总吞吐 **854 tok/s**、TTFT 中位 **0.52 s**、TPOT **12.1 ms**。

---

## 0. 一句话架构

```
                    ┌─────────── router (sglang_router, PD 分离) ───────────┐
   client ──HTTP──▶ │  policy=cache_aware                                   │
                    └───┬──────────────┬──────────────┬───────────┬────────┘
                        ▼              ▼              ▼           ▼
                   prefill p0     prefill p1     prefill p2    decode (d0+d1)
                   1 node/4GPU    1 node/4GPU    1 node/4GPU   2 node/8GPU DEP8
                   pp=4           pp=4           pp=4          tp8/dp8/ep8
                        │              │              │           ▲
                        └──────────────┴──────────────┘           │
                              KV cache 直传（域内 NVLink C2C）──────┘
                              mooncake NVLINK pool，不走 RoCE
```

- **5 节点全部在同一个 NVL72 subblock**（podAffinity 强制），所以 prefill 算完的 KV 可以直接经 **NVLink** 传给 decode，绕开 RoCE。
- prefill 计算密集（pp4 摊显存）、decode 访存密集（DP-Attention + Wide-EP DEP8）。

---

## 1. 前置条件（环境假设）

| 项 | 值 | 说明 |
|---|---|---|
| 集群 | `gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test` | kubectl 经 `ssh glinux $HOME/google-cloud-sdk/bin/kubectl` |
| 节点池 | `gb300-pool-0010`（≥5 节点，标签 `team=yangwhale`） | 同一 NVL72 subblock；用 fresh 池避免镜像叠加撑爆磁盘 |
| 集群已装 | DRA GPU driver（ComputeDomain/IMEX）、DRANET `mrdma.google.com`、asapd-lite DaemonSet、`ar-pull-secret`（CronJob 自动刷新） | 见 `../03-gpu-stack/`、`../12-self-managed-k8s/` |
| 模型 | `gs://chrisya-gb300-models/DeepSeek-R1-0528-NVFP4-v2`（US-CENTRAL1，385G / 163 safetensors） | 与集群同区，`gcloud storage cp` 快 |
| GIB 包 | `gs://chrisya-gb300-models/gib-a4xmax.tgz`（16MB，从 nemo 镜像 `/usr/local/gib` 打包） | 官方 SGLang 镜像不带 GIB，需注入 |
| 节点物理内存 | **942 GiB**（allocatable ~909 Gi） | 内存盘 + pod 请求不能超 |

> 下文所有 `kubectl` 简写为 `K`：`K="$HOME/google-cloud-sdk/bin/kubectl --context=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test"`（在 glinux 上）。

---

## 2. 镜像选择 —— 为什么用官方 SGLang 镜像

**用**：`lmsysorg/sglang:v0.5.15.post1-cu130`（linux/arm64，公开 docker.io，无需 pull secret）。

**为什么就它对**：
- **含 `sm_103a` cubin** —— GB300 = Blackwell Ultra = compute capability **10.3**。`cuobjdump` 实测该镜像 `sgl_kernel/*/common_ops.abi3.so` arch = `sm_90/100a/103a/110a/120a/121a`，有 `sm_103a`。
- **torch 2.11.0+cu130（标准 build）** —— C10 CUDA ABI 与 PyPI `sgl-kernel` 匹配，**不用打任何 ABI shim**。
- 自带全套：`sglang 0.5.15.post1 + sgl-kernel + flashinfer + deep_ep + mooncake`。

**为什么不用 nemo 训练镜像**（血泪，详见 RUNLOG 坑 17-19）：
- `nemo-gb300-ready:26.06-v1` 是 mutable tag，当初烤进去的私有 `sgl-kernel 0.13.1`（带 sm_103）随 tag 覆盖丢了，现在两个 tag 都不带 sgl-kernel。
- PyPI 最高 `sgl-kernel 0.3.21` **只编到 sm_100/101/120，无 sm_103a、无 PTX** → GB300 上跑 `RMSNorm: no kernel image is available`。
- 且 nemo 的 NV NGC torch 改了 C10 ABI（`c10_cuda_check_implementation` 第 4 参 int→uint），PyPI sgl-kernel import 直接 undefined symbol。

---

## 3. 部署 5 个 Pod

### 3.1 生成 YAML（`gen-pods.py`）

在 glinux 上存为 `/tmp/gen-pods.py`：

```python
import yaml
docs=[]
# ComputeDomain：跨节点 NVLink（MNNVL）+ IMEX channel，一个 subblock 一个
docs.append({"apiVersion":"resource.nvidia.com/v1beta1","kind":"ComputeDomain",
  "metadata":{"name":"sgl3-cd"},"spec":{"numNodes":0,"channel":{"resourceClaimTemplate":{"name":"sgl3-ch"}}}})
# mrdma DRA：每 pod 8 张 CX-8 HCA
docs.append({"apiVersion":"resource.k8s.io/v1","kind":"ResourceClaimTemplate",
  "metadata":{"name":"sgl3-mrdma"},"spec":{"spec":{"devices":{"requests":[
    {"name":"req-mrdma","exactly":{"deviceClassName":"mrdma.google.com","allocationMode":"ExactCount","count":8}}]}}}})
# headless Service：pod 间用 <pod>.sgl3 域名互访
docs.append({"apiVersion":"v1","kind":"Service","metadata":{"name":"sgl3"},
  "spec":{"selector":{"app":"sgl3"},"clusterIP":"None","publishNotReadyAddresses":True}})
IMAGE="lmsysorg/sglang:v0.5.15.post1-cu130"
for name in ["sgl3-p0","sgl3-p1","sgl3-p2","sgl3-d0","sgl3-d1"]:
  docs.append({"apiVersion":"v1","kind":"Pod",
    "metadata":{"name":name,"labels":{"app":"sgl3"}},
    "spec":{"subdomain":"sgl3","hostname":name,"tolerations":[{"operator":"Exists"}],
      "nodeSelector":{"cloud.google.com/gke-nodepool":"gb300-pool-0010","team":"yangwhale"},
      "affinity":{
        # 全部 pod 落同一 subblock → KV 走域内 NVLink
        "podAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":[
          {"labelSelector":{"matchExpressions":[{"key":"app","operator":"In","values":["sgl3"]}]},
           "topologyKey":"cloud.google.com/gce-topology-subblock"}]},
        # 一节点一 pod（独占 4 GPU）
        "podAntiAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":[
          {"labelSelector":{"matchExpressions":[{"key":"app","operator":"In","values":["sgl3"]}]},
           "topologyKey":"kubernetes.io/hostname"}]}},
      "imagePullSecrets":[{"name":"ar-pull-secret"}],  # 公开镜像其实不需要，留着无害
      "containers":[{"name":"sglang","image":IMAGE,"securityContext":{"privileged":True},
        "resources":{"limits":{"nvidia.com/gpu":4,"memory":"800Gi"},"requests":{"memory":"800Gi"},
          "claims":[{"name":"req-mrdma"},{"name":"compute-domain-channel"}]},
        "volumeMounts":[{"name":"ssd","mountPath":"/mnt/ssd"},{"name":"shm","mountPath":"/dev/shm"}],
        "env":[{"name":"GLOO_SOCKET_IFNAME","value":"eth0"},{"name":"NCCL_SOCKET_IFNAME","value":"eth0"}],
        "command":["sleep","infinity"]}],
      "volumes":[
        # 内存盘放模型（385G）。sizeLimit 是运行时上限，不计入调度
        {"name":"ssd","emptyDir":{"medium":"Memory","sizeLimit":"500Gi"}},
        {"name":"shm","emptyDir":{"medium":"Memory","sizeLimit":"64Gi"}}],
      "resourceClaims":[
        {"name":"req-mrdma","resourceClaimTemplateName":"sgl3-mrdma"},
        {"name":"compute-domain-channel","resourceClaimTemplateName":"sgl3-ch"}]}})
yaml.safe_dump_all(docs, open("/tmp/sgl3-mem.yaml","w"), sort_keys=False)
print("generated /tmp/sgl3-mem.yaml")
```

> **内存关键（坑 22）**：节点物理内存 942 GiB（allocatable ~909 Gi）。pod `requests.memory` 设 **800Gi**（含模型 385G tmpfs + sglang 运行时，留余量）。设 1200Gi 会 `Insufficient memory` Pending，甚至 MemoryPressure Evict。tmpfs `sizeLimit` 只在运行时限制、不计入调度。

### 3.2 部署

```bash
python3 /tmp/gen-pods.py
$K apply -f /tmp/sgl3-mem.yaml
# 等 5 pod Running（fresh 节点拉 12.7GB 镜像 ~2-3min）
$K get pods -o wide | grep sgl3
```

> ⚠️ **预期会偶发（两轮验证各中 1 次）**：某 pod `ContainerStatusUnknown`/`Evicted`，`describe` 事件是 `The node was low on resource: ephemeral-storage`——拉 12.7GB 镜像瞬间把节点 boot 盘顶爆。**不是配置错，删了重建即可**（会调度到别的 fresh 节点），偶尔要重试 1-2 次：
> ```bash
> # 循环直到 5 pod 全 Running
> $K delete pod <name> --force --grace-period=0
> $K apply -f /tmp/sgl3-mem.yaml
> ```
> 重建的 pod 记得也补跑一遍 §4 bootstrap（它不在首批并行里）。

**部署后自检**（任一 pod）：
```bash
$K exec sgl3-p0 -- nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader   # 预期 NVIDIA GB300, 10.3
$K exec sgl3-p0 -- ls /dev/infiniband        # 预期 uverbs0..7（8 张 HCA）
$K exec sgl3-p0 -- ls /dev/nvidia-caps-imex-channels/ | head -1   # 预期 channel0（ComputeDomain 生效）
$K exec sgl3-p0 -- df -h /mnt/ssd /dev/shm   # 预期 tmpfs 500G / 64G
```

---

## 4. 逐 Pod Bootstrap（GIB + DOCA + gcloud + 模型，一个脚本）

在 glinux 存为 `/tmp/bootstrap.sh`：

```bash
#!/bin/bash
set -u
LOG=/tmp/bootstrap.log; exec > >(tee -a "$LOG") 2>&1
echo "=== [$(hostname)] bootstrap $(date) ==="

# 1) gcloud（官方镜像没有，装到 /root，用来拉 GIB 和模型）
if ! command -v gcloud >/dev/null 2>&1 && [ ! -x /root/google-cloud-sdk/bin/gcloud ]; then
  cd /root
  curl -sS -o gcloud.tgz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-arm.tar.gz
  tar -xzf gcloud.tgz && ./google-cloud-sdk/install.sh -q --path-update false >/dev/null 2>&1
fi
export PATH=/root/google-cloud-sdk/bin:$PATH
echo "[bootstrap] gcloud: $(gcloud --version 2>&1 | head -1)"

# 2) GIB（GKE NCCL/RDMA plugin，从 GCS 拉 + 解压到 /usr/local/gib）
if [ ! -f /usr/local/gib/scripts/set_nccl_env.sh ]; then
  gcloud storage cp gs://chrisya-gb300-models/gib-a4xmax.tgz /tmp/gib.tgz 2>&1 | tail -1
  tar xzf /tmp/gib.tgz -C /usr/local && echo "[bootstrap] GIB extracted"
fi

# 3) DOCA OFED userspace（CX-8 verbs，nixl/mooncake RDMA backend 必须）
if [ ! -f /usr/lib/aarch64-linux-gnu/libmlx5.so.1 ] || ! dpkg -l doca-ofed-userspace >/dev/null 2>&1; then
  apt-get update -y -qq
  command -v curl >/dev/null || apt-get install -y -qq curl gnupg
  DOCA_URL="https://linux.mellanox.com/public/repo/doca/3.1.0/ubuntu24.04/arm64-sbsa/"
  curl -fsSL https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub 2>/dev/null
  echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list
  apt-get update -y -qq
  apt-get install -y -qq doca-ofed-userspace 2>&1 | tail -2
fi
echo "[bootstrap] libmlx5: $(ls /usr/lib/aarch64-linux-gnu/libmlx5.so.1* 2>&1 | head -1)"

# 4) nixl（镜像自带；缺则补。mooncake 走 NVLink 但 backend 仍需 nixl 存在）
python -c "import nixl" 2>/dev/null || python -m pip install --no-cache-dir nixl >/dev/null 2>&1

# 5) 验证 sglang 栈（native，无 shim）
echo "[bootstrap] $(python -c 'import sglang,sgl_kernel,flashinfer;print("sglang",sglang.__version__,"OK")' 2>&1 | grep -iv warning | tail -1)"

# 6) 模型 → 内存盘（同区 GCS，~1.5min）
if [ ! -f /mnt/ssd/DeepSeek-R1-0528-NVFP4-v2/config.json ]; then
  echo "[bootstrap] copying model $(date)..."
  gcloud storage cp -r gs://chrisya-gb300-models/DeepSeek-R1-0528-NVFP4-v2 /mnt/ssd/ 2>&1 | tail -1
fi
echo "[bootstrap] model: $(ls /mnt/ssd/DeepSeek-R1-0528-NVFP4-v2/ 2>&1 | wc -l) files, $(du -sh /mnt/ssd/DeepSeek-R1-0528-NVFP4-v2 2>&1 | cut -f1)"
echo "=== [$(hostname)] BOOTSTRAP_DONE $(date) ==="
```

5 pod 并行跑（后台）：
```bash
for p in sgl3-p0 sgl3-p1 sgl3-p2 sgl3-d0 sgl3-d1; do
  $K cp /tmp/bootstrap.sh $p:/tmp/bootstrap.sh
  $K exec $p -- bash -c 'setsid bash /tmp/bootstrap.sh >/tmp/bootstrap.out 2>&1 </dev/null &'
done
# 等全部 BOOTSTRAP_DONE（~3-4min）
for p in sgl3-p0 sgl3-p1 sgl3-p2 sgl3-d0 sgl3-d1; do
  echo -n "$p: "; $K exec $p -- grep -c BOOTSTRAP_DONE /tmp/bootstrap.log
done
```

---

## 5. 启动 3 Prefill + 2 Decode

### 5.1 prefill 脚本（`/tmp/prefill.sh`，推到 p0/p1/p2）

```bash
#!/bin/bash
source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf
export LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=INFO                                    # 一开始就开，方便收集细节
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_GRAPH_REGISTER=0
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK             # KV 走域内 NVLink（关键）
export MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1
export SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_DG_CACHE_DIR=/mnt/ssd/dg-cache FLASHINFER_WORKSPACE_BASE=/mnt/ssd/fi-cache
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000
python -m sglang.launch_server --model-path /mnt/ssd/DeepSeek-R1-0528-NVFP4-v2 --served-model-name deepseek-ai/DeepSeek-R1 \
  --quantization modelopt_fp4 --attention-backend trtllm_mla --kv-cache-dtype fp8_e4m3 --context-length 8192 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 30001 \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --tp-size 1 --dp-size 1 --ep-size 1 --pp-size 4 --moe-runner-backend flashinfer_trtllm \
  --mem-fraction-static 0.72 --disable-flashinfer-autotune --chunked-prefill-size -1 --disable-radix-cache \
  --trust-remote-code --watchdog-timeout 1000000 --host 0.0.0.0 --port 30000
```

### 5.2 decode 脚本（`/tmp/decode.sh`，推到 d0/d1；参数 `$1`=node-rank `$2`=dist-init-addr）

与 prefill 相同的 env 头，最后一段换成：

```bash
#!/bin/bash
NODE_RANK=$1
DIST_ADDR=$2
# ... (同 prefill 的 env 头，完全一样) ...
python -m sglang.launch_server --model-path /mnt/ssd/DeepSeek-R1-0528-NVFP4-v2 --served-model-name deepseek-ai/DeepSeek-R1 \
  --quantization modelopt_fp4 --attention-backend trtllm_mla --kv-cache-dtype fp8_e4m3 --context-length 8192 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 30001 \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --enable-dp-attention --enable-dp-lm-head --tp-size 8 --dp-size 8 --ep-size 8 --pp-size 1 \
  --nnodes 2 --node-rank $NODE_RANK --dist-init-addr $DIST_ADDR \
  --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend flashinfer_cutedsl \
  --cuda-graph-max-bs 64 --max-running-requests 64 \
  --mem-fraction-static 0.80 --disable-flashinfer-autotune --chunked-prefill-size -1 --disable-radix-cache \
  --trust-remote-code --watchdog-timeout 1000000 --host 0.0.0.0 --port 30000
```

### 5.3 启动

```bash
# 推脚本
for p in sgl3-p0 sgl3-p1 sgl3-p2; do $K cp /tmp/prefill.sh $p:/tmp/prefill.sh; done
for p in sgl3-d0 sgl3-d1; do $K cp /tmp/decode.sh $p:/tmp/decode.sh; done

# decode rank0 的 pod IP 作为 dist-init-addr
D0IP=$($K get pod sgl3-d0 -o jsonpath='{.status.podIP}')

# 3 prefill
for p in sgl3-p0 sgl3-p1 sgl3-p2; do
  $K exec $p -- bash -c 'setsid bash /tmp/prefill.sh >/tmp/prefill.log 2>&1 </dev/null &'
done
# 2 decode（同一个 server 跨 2 节点，rank0 在 d0、rank1 在 d1）
$K exec sgl3-d0 -- bash -c "setsid bash /tmp/decode.sh 0 $D0IP:5757 >/tmp/decode.log 2>&1 </dev/null &"
$K exec sgl3-d1 -- bash -c "setsid bash /tmp/decode.sh 1 $D0IP:5757 >/tmp/decode.log 2>&1 </dev/null &"
```

**等就绪**（decode 跨节点 NCCL init + cuda graph capture，~3-4min）：
```bash
# 关键日志：decode 应打印 "Using cross-node NVLink transport (MC_FORCE_MNNVL)"
$K exec sgl3-d0 -- grep -c "cross-node NVLink transport" /tmp/decode.log     # 预期 >0
# 全部 Uvicorn 起来
for p in sgl3-p0 sgl3-p1 sgl3-p2 sgl3-d0; do
  echo -n "$p: "; $K exec $p -- grep -c "Uvicorn running" /tmp/*.log
done
# /health 全 200
for p in sgl3-p0 sgl3-p1 sgl3-p2 sgl3-d0; do
  echo -n "$p health: "; $K exec $p -- bash -c 'curl -s -m10 -o /dev/null -w "%{http_code}" http://localhost:30000/health'; echo
done
```

> **别用 `pkill -f sglang.launch_server`**（会匹配到 exec 命令行自杀，exit 137）。重启用 `pkill -9 python`（按进程名，不会误杀 bash exec）。

---

## 6. 启动 Router（在 p0 上）

```bash
# 取 5 pod IP
$K get pods -o custom-columns=N:.metadata.name,IP:.status.podIP --no-headers | grep sgl3
```

`/tmp/router.sh`（把下面 IP 换成实际 pod IP）：
```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://<P0_IP>:30000 30001 \
  --prefill http://<P1_IP>:30000 30001 \
  --prefill http://<P2_IP>:30000 30001 \
  --decode  http://<D0_IP>:30000 \
  --policy cache_aware --host 0.0.0.0 --port 8000
```
```bash
$K cp /tmp/router.sh sgl3-p0:/tmp/router.sh
$K exec sgl3-p0 -- bash -c 'setsid bash /tmp/router.sh >/tmp/router.log 2>&1 </dev/null &'
sleep 15
```

---

## 7. 端到端验证

```bash
$K exec sgl3-p0 -- bash -c 'curl -s -m60 http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"deepseek-ai/DeepSeek-R1\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}"'
```
预期返回带 `"text":" Paris, ..."` 的 JSON（`finish_reason":"length"`）。**若 60s 超时**：见 §9 KV 传输排查。

---

## 8. Benchmark

```bash
$K cp /dev/stdin sgl3-p0:/tmp/bench.sh <<'EOF'
python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port 8000 \
  --model deepseek-ai/DeepSeek-R1 --dataset-name random \
  --random-input-len 1024 --random-output-len 512 \
  --num-prompts 32 --max-concurrency 8
EOF
$K exec sgl3-p0 -- bash /tmp/bench.sh 2>&1 | grep -iE "throughput|TTFT|TPOT|latency|concurrency"
```

**实测参考（本配置）**：

| 并发 | 总吞吐 tok/s | TTFT median | TPOT mean | 备注 |
|---|---|---|---|---|
| **8** | **854.7** | **517 ms** | **12.1 ms** | 健康工作点（首轮）|
| 8（复现①）| — | **275 ms** | **9.9 ms** | 从零按本文重跑，指标一致 ✅ |
| 8（复现②）| — | **268 ms** | **10.0 ms** | 再从零重跑，指标再次一致 ✅ |
| 16 | 256.2 | 59.6 s | 11.5 ms | 3 prefill 被压爆，TTFT 长尾爆炸 |

> **本文已端到端复现验证 ×2**：2026-07-19 按本文**两次**从零起全新 pod（每次删光重来），均一次通到 benchmark，中位 TTFT 268-275ms / TPOT 9.9-10.0ms，三次结果一致。除拉镜像偶发 DiskPressure（删重建即可，见 §3.2）外，**零功能改动**。
>
> **启动小知识**：decode 起来前会刷 `DeepGEMM warmup: 0/65536`，初始 ETA 显示几十小时是**误导**——JIT 一热就到 ~1000 it/s，实际约 **1 分钟**跑完，别被吓到。

- decode（NVLink KV pool）TPOT ~12ms 跨并发几乎不变 → NVLink 传 KV 无瓶颈。
- 瓶颈是 **prefill 数量**：3×prefill 各 pp4，高并发排队。想压 TTFT → 加 prefill 副本 / prefill 改 tp。

---

## 9. 关键坑速查（为什么每步都不能省）

| 现象 | 根因 | 解 |
|---|---|---|
| `No module named sgl_kernel` / `RMSNorm: no kernel image` | nemo 镜像丢了 sm_103 kernel；PyPI sgl-kernel 无 sm_103a cubin | 用官方 `v0.5.15.post1-cu130`（含 sm_103a）|
| `undefined symbol: c10_cuda_check_implementation` | NV NGC torch 改了 C10 ABI | 官方镜像的标准 torch 匹配，别用 nemo |
| `NIXL_ERR_BACKEND` / RDMA backend 创建失败 | 官方 Ubuntu 镜像缺 CX-8 的 mlx5 verbs | 装 `doca-ofed-userspace`（§4 step 3）|
| 单请求 60s 超时 / `KVTransferError: Aborted` | nixl 走 RoCE：GKE 是 RoCE v2 over IPv6，netdev 名 `gpuNipvlanM`，UCX 调不通 | **改走 NVLink**：`--disaggregation-transfer-backend mooncake` + `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` + `MC_FORCE_MNNVL=1` |
| pod `Insufficient memory` Pending | 内存请求 > 节点 909Gi allocatable | request/limit 设 800Gi |
| pod `Evicted` MemoryPressure/DiskPressure | 内存盘+模型超物理内存 / 节点叠加大镜像 | 内存盘 ≤500Gi + 用 fresh 节点池 |
| `pkill` 后 exit 137、server 没重启 | `pkill -f sglang.launch_server` 匹配到自己命令行 | 用 `pkill -9 python` |
| `kubectl cp` 脚本被截断成几行 | cp 大文件/管道问题 | cp 后 `wc -l` 校验 |

---

## 10. 清理

```bash
$K delete pod sgl3-p0 sgl3-p1 sgl3-p2 sgl3-d0 sgl3-d1 --force --grace-period=0
$K delete -f /tmp/sgl3-mem.yaml   # 连带删 ComputeDomain / Service / RCT
```

---

*沉淀自 2026-07-19 的 20+ 小时实测。失败尝试全记录见 RUNLOG。下一步（Round 4）：ctx8_dep32 占满一个 NVL72 域（64 GPU），prefill 加副本压 TTFT。*
