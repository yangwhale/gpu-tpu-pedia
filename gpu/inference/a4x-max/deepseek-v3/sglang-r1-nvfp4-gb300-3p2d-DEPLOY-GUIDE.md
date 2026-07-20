# GB300 (A4X Max) · SGLang DeepSeek-R1-NVFP4 · 3P2D 端到端复现指南

> 一次性、事无巨细、可复制执行的部署手册。目标：在 GB300 NVL72 (GKE) 上把 DeepSeek-R1-0528-NVFP4 用 SGLang **PD 分离（3 prefill + 2 decode，DEP8 Wide-EP）** 端到端跑起来，KV cache 走**域内 NVLink**。
>
> 本文是 [`sglang-r1-nvfp4-128k-gb300-RUNLOG.md`](./sglang-r1-nvfp4-128k-gb300-RUNLOG.md) 里 20+ 小时趟坑后的**干净沉淀版**——RUNLOG 记录所有失败尝试，本文只留一条走得通的路。
>
> **§0.5：全景消融总览**（每一步开了什么、收益、原理，一目了然）。**§2–§8：3P2D 基础配方（20 GPU / ctx8192）**，两次从零验证一致。**§9：放大到 64 GPU / 128K 长上下文**（warm 312 TPS/GPU，超官方 226 达 38%）+ **§9.7 MTP**（per-user 2.15×）+ 官方 Roadmap。
> **实测（3P2D / 20 GPU / 1024in-512out）**：conc256 甜点 10.4k tok/s / TTFT 12s / TPOT 17ms。**实测（64 GPU / 128K-8K）**：conc32 warm 312 TPS/GPU / TPOT 12ms。**实测（MTP / 2048-1024）**：单用户 97→209 tok/s / TPOT 减半。

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

### 术语速查（先看这个，下文全用这套简写）

> ⚠️ **有两套 "xPyD" 记法，别混**：本项目 D 数 = decode **节点**数；官方 recipe 的 D 数 = decode **组**数。下表对齐。

| 简写 | 含义 | 展开 |
|---|---|---|
| **xPyD（本项目记法）** | x 个 prefill 节点 + y 个 decode **节点**（数物理机器） | `3P2D` = 3 prefill 节点 + 2 decode 节点，共 5 节点；`8P8D` = 8+8 = 16 节点 |
| **PP4** | 单个 prefill 内 pipeline parallel = 4 | 1 prefill = 1 台机器 4 GPU，模型按层切 4 段流水摊显存。本项目 1 prefill 实例 = 1 节点 |
| **decode 组 / DEPn** | 若干 decode 节点**合成一个** server 一起服务 | `DEP8` = 2 节点 8 GPU 一组（本文 3P2D，`--nnodes 2`）；`DEP32` = 8 节点 32 GPU 一组（§9，`--nnodes 8`）|
| **DEP 是啥** | **D**ata-parallel(attention) + **E**xpert-parallel(MoE) | attention 走 **DP**（每卡各算各的请求，不切单请求，省通信）；MoE 走 **EP**（256 个 expert 摊到 n 卡，每卡存 ~256/n 个）= Wide-EP |
| **官方 recipe 记法** | `ctx8_pp4_gen1_dep32` | `ctx8` = 8 prefill(PP4)；`gen1` = **1 个 decode 组**；`dep32` = 该组 32 GPU（8 节点）|

> **为什么"官方 8P1D"和"16 节点"不矛盾**：官方那个 "1D"（gen1）是 **1 个 decode 组**，但这一组本身摊在 **8 台机器**上（DEP32）。换算成本项目记法就是 **8P8D**（8 prefill 节点 + 8 decode 节点）= 16 节点。**「1 组」≠「1 台机器」**——这是你会看懵的唯一坑。
>
> 记：`P` 数在两套记法里都 = prefill 节点数；`D` 数在本项目 = 节点数、在官方 recipe = 组数；`DEPn` 永远 = 那个 decode 组用 DP-attention + EP-MoE 铺 n 张卡。

---

## 0.5 全景消融：每一步开了什么 · 收益 · 原理（一目了然）

这份配方不是一次配好的，是一步步加东西试出来的。每加一个开关，测一次，看它带来什么收益。下表是整个演进链（R1→R5），**看这张表就懂为什么每个参数都在**：

| 步骤 | 开启 / 改动 | 关键收益（实测） | 背后原理 |
|---|---|---|---|
| **① Base** (R1, 4 GPU) | 单节点 TP4 加载 R1-NVFP4（`modelopt_fp4` + `trtllm_mla` + fp8 KV） | 功能通，`<think>` 正常 | NVFP4 权重把 671B 压到单节点能放；trtllm_mla 是 GB300 上 MLA 的最快 attention kernel |
| **② PD 分离** (R2, 8 GPU) | prefill / decode 拆成两个独立 server（1P1D） | decode 专用化，**TPOT 9.2ms（~109 tok/s/user）** | prefill 算力密集、decode 访存密集，两者放一起互相拖累；拆开各自选最优并行 + 各自吃满硬件 |
| **③ NVLink KV pool** (R3 关键突破) | KV 传输从 RoCE 改 `mooncake` **NVLINK** pool（`MC_FORCE_MNNVL=1`） | KV 直传**不再超时**，端到端跑通 | GKE 的 RoCE 是 IPv6 over `gpuNipvlanM`，UCX/nixl 调不通；同 subblock 内走 NVLink C2C，带宽高一个量级还绕开这坑 |
| **④ Wide-EP (DEP8)** (R3) | decode 上 DP-Attention + Wide-EP `deepep` low_latency，8 卡铺专家 | 吞吐 conc8→512 拉到 **10,715 tok/s（12.5×）** | expert 分散到更多 GPU，每卡只算部分专家 → 单卡显存/算力压力降 → 能塞更大 batch |
| **⑤ DEP32 + 128K** (R4, 64 GPU) | Wide-EP 铺到 32 卡 + `context 8K→128K` + `chunked-prefill 32K` | **warm 312 TPS/GPU（超官方 226 达 38%）** | 更宽 EP 摊更大 KV；chunked 把长 prompt 沿 seq 切成块，causal attention 让每块 KV 精确可缓存，避免 128K 一次性爆显存 |
| **⑥ MTP** (R5) | decode 加 EAGLE spec decode（`num-steps 2 / topk 1 / draft 3`） | 单用户 **97→209 tok/s（2.15×）**，TPOT 10→4.5ms | draft(nextn) 一次猜多个 token，target 并行**验证**，猜对的直接采纳 → 一个 decode step 平均出 ~2.7 个 token 而非 1 |

> **一句话串起来**：NVFP4 让模型放得下 → PD 分离让 decode 专用化 → NVLink KV 让分离真正跑通 → Wide-EP 把吞吐堆上去 → 128K 支撑长上下文 → MTP 把单用户速度翻倍。**③ 是从"跑不通"到"跑通"的转折点，⑥ 是延迟收益最大的一步。**

> **注意各步 benchmark 的 workload 不同**（③④是 1024/512 短上下文压吞吐、⑤是 128K/8K 长上下文、⑥是 2048/1024 测 per-user），数字不能跨步直接比大小，要看**各自相对基线的收益倍数**。完整逐 Round 数据见 [RUNLOG](./sglang-r1-nvfp4-128k-gb300-RUNLOG.md) 顶部总表。

---

## 1. 前置条件（环境假设）

| 项 | 值 | 说明 |
|---|---|---|
| 集群 | `gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test` | kubectl 经 `ssh glinux $HOME/google-cloud-sdk/bin/kubectl` |
| 节点池 | `gb300-pool-0010`（≥5 节点，标签 `team=yangwhale`） | 同一 NVL72 subblock；用 fresh 池避免镜像叠加撑爆磁盘 |
| 集群已装 | DRA GPU driver（ComputeDomain/IMEX）、DRANET `mrdma.google.com`、asapd-lite DaemonSet、`ar-pull-secret`（CronJob 自动刷新） | 见 `../03-gpu-stack/`、`../12-self-managed-k8s/` |
| 模型 | `gs://chrisya-gb300-models/DeepSeek-R1-0528-NVFP4-v2`（US-CENTRAL1，385G / 163 safetensors） | 与集群同区，`gcloud storage cp` 快 |
| GIB 包 | `gs://chrisya-gb300-models/gib-a4xmax.tgz`（16MB，从 nemo 镜像 `/usr/local/gib` 打包） | 官方 SGLang 镜像不带 GIB，需注入 |
| **Local SSD RAID** | 每节点 4× Local NVMe SSD → RAID0 12T 挂 `/mnt/disks/raid/0` | **模型存这（不放内存盘）**。先部署 `gke-raid-disks` DaemonSet，见 [`../deepseek-v4/gb300-local-ssd-raid0-SETUP.md`](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md) |
| 节点物理内存 | **942 GiB**（allocatable ~909 Gi） | RAM 留给运行时 + KV cache；模型在 Local SSD 不占 RAM |

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
        "resources":{"limits":{"nvidia.com/gpu":4,"memory":"600Gi"},"requests":{"memory":"600Gi"},
          "claims":[{"name":"req-mrdma"},{"name":"compute-domain-channel"}]},
        # ssd 用 Local SSD RAID（hostPath），HostToContainer 才能看到 DaemonSet 的挂载
        "volumeMounts":[{"name":"ssd","mountPath":"/mnt/ssd","mountPropagation":"HostToContainer"},{"name":"shm","mountPath":"/dev/shm"}],
        "env":[{"name":"GLOO_SOCKET_IFNAME","value":"eth0"},{"name":"NCCL_SOCKET_IFNAME","value":"eth0"}],
        "command":["sleep","infinity"]}],
      "volumes":[
        # 模型放 Local SSD RAID（12T，读14GB/s，不吃 RAM，跨 pod 持久）。前置：gke-raid-disks DaemonSet（见 §1 / RAID-SETUP 文档）
        {"name":"ssd","hostPath":{"path":"/mnt/disks/raid/0","type":"Directory"}},
        {"name":"shm","emptyDir":{"medium":"Memory","sizeLimit":"64Gi"}}],
      "resourceClaims":[
        {"name":"req-mrdma","resourceClaimTemplateName":"sgl3-mrdma"},
        {"name":"compute-domain-channel","resourceClaimTemplateName":"sgl3-ch"}]}})
yaml.safe_dump_all(docs, open("/tmp/sgl3-mem.yaml","w"), sort_keys=False)
print("generated /tmp/sgl3-mem.yaml")
```

> **内存关键**：模型放 **Local SSD**（不进 RAM），pod `requests.memory` 只需覆盖 **sglang 运行时 + 加载时权重缓冲峰值** → 设 **600Gi**（节点 909Gi allocatable，留出 RAM 给 KV cache / 未来 HiCache CPU offload）。
> - **别把权重放内存盘**（`emptyDir medium:Memory`）：385G 模型吃 385G RAM，纯浪费——内存要留给 KV cache。
> - **⚠️ 加载峰值坑**：设 **200Gi 会 OOMKilled**（exit 137）——sglang 加载时把权重读进 host 缓冲，峰值超了。这是加载瞬时占用，非常驻；用 ≥600Gi 留够。
> - **前置**：Local SSD RAID 必须先就位（`/mnt/disks/raid/0`），见 §1 前置 + [`../deepseek-v4/gb300-local-ssd-raid0-SETUP.md`](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md)。

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
$K exec sgl3-p0 -- df -h /mnt/ssd /dev/shm   # 预期 /mnt/ssd = /dev/md0 12T（Local SSD RAID）；/dev/shm 64G
```

### 3.3 为什么模型放 Local SSD（不放内存盘）+ 实测

**原则：内存盘存权重是浪费——RAM 要留给 KV cache**（decode 的 KV 在 GPU HBM，但未来 HiCache 会把 KV offload 到 CPU 内存；且大模型如 V4-Pro 800G 根本塞不进内存盘）。所以 §3.1 默认就用 **Local SSD RAID**：
- 模型放 12T Local SSD RAID（读 14 GB/s），**不占 RAM**；
- **跨 pod 重启持久**（host 级，删 pod 模型还在，省重下）；
- 推理吞吐与内存盘**完全一致**（storage 只影响加载不影响推理）。

**实测（2026-07-20）**：
- cp GCS→Local SSD **73s @ 5.4 GiB/s**（385G）。
- 单节点 TP4 冒烟 + 完整 3P2D 端到端（router + e2e " Paris" + benchmark）全过。吞吐 conc32/64/128 total = 4758/5938/9439 tok/s，TPOT 15-17ms，与内存盘基线（4757/—/9509）吻合。
- **⚠️ 加载峰值坑**：内存 request 设 200Gi 会 **OOMKilled**——sglang 加载时把权重读进 host 缓冲，峰值超了。这是**加载瞬时**占用非常驻，用 ≥600Gi。

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

# 6) 模型 → Local SSD（/mnt/ssd 背后是 RAID0，同区 GCS ~73s @ 5.4GiB/s；已存在则跳过，跨 pod 持久）
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

压并发才看得到真实吞吐（sweep）：
```bash
$K cp /dev/stdin sgl3-p0:/tmp/sweep.sh <<'EOF'
for C in 32 64 128; do
  echo "===== concurrency=$C ====="
  python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port 8000 \
    --model deepseek-ai/DeepSeek-R1 --dataset-name random \
    --random-input-len 1024 --random-output-len 512 \
    --num-prompts $((C*4)) --max-concurrency $C 2>&1 \
    | grep -iE "Total token throughput|Output token throughput|Median TTFT|Median TPOT"
done
EOF
# 长跑，后台 + 轮询（exec 长连接会断）
$K exec sgl3-p0 -- bash -c 'setsid bash /tmp/sweep.sh >/tmp/sweep.log 2>&1 </dev/null &'
# ~8min 后：
$K exec sgl3-p0 -- grep -iE "concurrency=|Total token|Output token|Median" /tmp/sweep.log
```

**⚠️ 关键：benchmark 一定要压并发，别用 conc=8 汇报数字**（conc=8 只用到 DEP8 decode 容量的 ~3%，GB300 DEP8 理论可扛 ~288 并发）。并发 sweep（random 1024in/512out）：

完整并发 sweep（压到吞吐见顶）：

| 并发 | 总吞吐 tok/s | output tok/s | TPOT median | TTFT median | 实际并发 |
|---|---|---|---|---|---|
| 8 | 854 | 340 | 10 ms | 0.5 s | 8 |
| 32 | 4635 | 1538 | 15 ms | 0.5 s | 26 |
| 64 | 5265 | 1797 | 16 ms | 2.5 s | 54 |
| 128 | 6878 | 2297 | 17 ms | 8.4 s | 109 |
| 192 | 9975 | 3295 | 17 ms | 8.2 s | 150 |
| **256** | **10390** | 3334 | 17 ms | **12 s** | 198 | ← 甜点 |
| 384 | 10438 | 3429 | 17 ms | 21 s | 300 |
| 512 | **10715** | **3609** | 17 ms | 30 s | 416 | ← 峰值 |

- **峰值 ~10,700 tok/s（output ~3600），是 conc8（854）的 12.5×**。~535 tok/s/GPU（总）/ 180 output tok/s/GPU。
- **conc256 后见顶**：256→512 总吞吐只 +3%，但 TTFT 从 12s 爆到 30s。**实用甜点 = conc256**（10.4k tok/s，TTFT 12s，TPOT 17ms）。
- TPOT 全程稳定 ~17ms → decode（NVLink KV pool）稳。到顶双因：**prefill（3×pp4）喂不动**（TTFT 爆）+ decode output 也接近该 workload 极限。
- **对标官方**：lmsys GB300 博客 226 TPS/GPU 是 **128K/8K 长上下文**（decode 主导、DEP 铺到 32 卡），本配置 ctx 8192 / 短上下文 1024-512 是 prefill 偏重的不同 regime，数字不可直接对比。要逼近官方需上长上下文 + 更多 prefill 副本 + MTP（Round 4/5）。

> **本文已端到端复现验证 ×3**：2026-07-19 按本文**三次**从零起全新 pod（每次删光重来）。前两次一次通到 benchmark。**第三次全程照抄本文 §3–§8 代码块逐字执行做审计**：§3 部署 + 自检（GB300 10.3 / 8 HCA / imex channel0 / tmpfs 500+64G）✅；§4 bootstrap（gcloud 576 / GIB / libmlx5 / sglang 0.5.15.post1 / 模型 385G）✅；§5 起 3P2D，decode 日志出现 `cross-node NVLink transport`✅；§7 e2e 返回 `Paris`✅；§8 warm sweep（conc 32/64/128）实测 total **4757 / 8137 / 9509 tok/s**、TPOT **15.0 / 16.5 / 16.9 ms**——TPOT 与旧表精准吻合，吞吐比旧表（4635/5265/6878）**还高**（run-to-run 方差 + 本次 placement 更优；旧表数字偏保守）。**零功能改动，文档可原样复制执行**。
>
> **⚠️ 冷启动坑（审计实测）**：sweep 第一档（conc=32）如果撞上首次 JIT 编译，会出现假异常——TTFT 冲到 67s、total 掉到 438。这是 DeepGEMM/flashinfer 首跑编译，**不是配置错**。第二遍 warm 就正常（conc32 → 4757）。所以 benchmark **务必跑第二遍取 warm 值**（本文所有数字均为 warm）。
>
> **启动小知识**：decode 起来前会刷 `DeepGEMM warmup: 0/65536`，初始 ETA 显示几十小时是**误导**——JIT 一热就到 ~1000 it/s，实际约 **1 分钟**跑完，别被吓到。

---

## 9. 规模化：64 GPU 128K 长上下文（Round 4，对标官方 226 TPS/GPU）

> 把 3P2D（20 GPU / ctx8192）放大到**满一个 NVL72 域（64 GPU / 128K）**，对标 lmsys GB300 博客的长上下文峰值。**初始化跟上面 §2–§7 一字不改**（官方镜像 + GIB + DOCA + bootstrap + mooncake NVLINK），只有下面几处 delta。

### 9.1 拓扑：8P + DEP32（16 节点 / 64 GPU）
- prefill：8 个（p0–p7），各单节点 pp4（同 3P2D 单 prefill）
- decode：DEP32（tp32/dp32/ep32，**nnodes 8**，node-rank 0–7），8 节点
- 全 16 节点在同一 subblock（`pool-0010` 实测 16 节点同域，KV 走域内 NVLink）

### 9.2 相比 3P2D 的参数 delta
| 参数 | 3P2D | **Round 4** |
|---|---|---|
| pod 数（gen-pods.py）| 5（p0-2,d0-1）| **16（p0-7,d0-7）**|
| `--context-length` | 8192 | **131072** |
| prefill chunked | `--chunked-prefill-size -1`（关）| **`--chunked-prefill-size 32768 --enable-dynamic-chunking`** |
| decode tp/dp/ep | 8 | **32** |
| decode `--nnodes` | 2 | **8** |
| decode `--mem-fraction-static` | 0.80 | **0.75** |
| decode `--max-running-requests` | 64 | **512** |
| 存储 | Local SSD RAID（同 §3.1）| Local SSD RAID（同）|
| **pod 内存 limit** | 600Gi | **700Gi**（prefill；128K 活化峰值 + 加载缓冲）|

### 9.3 三个新坑（Round 4 实测，务必照做）
1. **prefill 内存要留够 128K 活化峰值 + 加载缓冲**：模型在 Local SSD 不占 RAM，但 128K prefill 的中间激活 + sglang 加载缓冲峰值仍高。**设 700Gi**（节点 909Gi allocatable）。decode 600Gi 够（KV 在 GPU HBM）。<br>（历史：内存盘时代因 385G tmpfs + 128K 峰值曾需 900Gi；换 Local SSD 后省掉 tmpfs 那 385G。2026-07-20 64卡 Local SSD 重测确认见 §9.5。）
2. **启动纪律：一个 pod 只启一次 prefill**。反复启动会**堆多个 python 进程**叠加 host 内存 → OOM → 容器重启**清空 /tmp**（prefill.sh 和 log 消失）→ 更乱。启动前 `pkill -9 python` 清干净，`kubectl cp prefill.sh` 后**必须 `wc -l` 校验落地**（cp 会静默失败），再 `nohup setsid bash ... & sleep 2; wc -l log` 确认启动。
3. **16 pod 同时申请 DRA 会滞后**：部分 pod 卡 `ContainerCreating` 报 `ResourceClaim not created yet`（DRA controller 处理不过来）。删掉卡住的 pod + `kubectl apply` 重触发 claim 分配即可，可能重试 1-2 轮。

### 9.4 启动（8 prefill + DEP32 decode）
```bash
# gen-pods.py 改 pods=[p0..p7]+[d0..d7]；prefill.sh 加 128K+chunked；decode.sh 改 tp/dp/ep32 nnodes8
for i in 0 1 2 3 4 5 6 7; do $K cp /tmp/prefill4.sh sgl4-p$i:/tmp/prefill4.sh; done   # 逐个 wc -l 校验
D0IP=$($K get pod sgl4-d0 -o jsonpath='{.status.podIP}')
for i in 0 1 2 3 4 5 6 7; do $K exec sgl4-p$i -- bash -c 'nohup setsid bash /tmp/prefill4.sh >/tmp/prefill.log 2>&1 </dev/null &'; done
for i in 0 1 2 3 4 5 6 7; do $K exec sgl4-d$i -- bash -c "nohup setsid bash /tmp/decode4.sh $i $D0IP:5757 >/tmp/decode.log 2>&1 </dev/null &"; done
# router 用 8 prefill + 1 decode（同 §6，多 5 个 --prefill 行）
```
> decode DEP32 = 32 GPU 跨 8 节点 NCCL rendezvous + 128K graph capture + DeepGEMM warmup，启动比 DEP8 久（~5-8min），耐心等，别误判卡死。

### 9.5 实测结果（128K/8K，7P+DEP32，warm）

| 并发 | 总吞吐 tok/s | TPS/GPU（总/64）| TPOT median | TTFT median |
|---|---|---|---|---|
| 8 | 8195 | 128 | 11.7 ms | 13 s |
| 16 | 12468 | 195 | 11.8 ms | 23 s |
| **32** | **19965** | **312** | **12.3 ms** | 20.7 s |

- **warm 峰值 312 TPS/GPU，超官方 226 达 38%**（cold 首跑仅 240——第一次编译/cache 未热，**benchmark 务必多跑几遍取 warm 值**）。
- TPOT 全程 ~12ms（decode NVLink KV pool 稳）。
- **Local SSD 64 卡验证（2026-07-20，pool-0007）**：8P+DEP32 全部模型走 Local SSD RAID（600Gi decode / 700Gi prefill），16 pod 全起、**prefill 零 OOM**（700Gi 扛住长上下文活化）、decode DEP32 fired、router + e2e 正确（" Paris..."）。**benchmark（120K/8K，conc32）实测总 18,536 tok/s = 290 TPS/GPU，TPOT 12.43ms**——与内存盘 Round 4（128K/8K，312 TPS/GPU，TPOT 12.3ms）在 workload+方差内一致，**TPOT 完全吻合 → Local SSD 对推理零影响，64 卡规模确认**。
  - **⚠️ bench 参数坑**：`--random-input-len` 会 +1 BOS，且 input+output 必须 ≤ `--context-length`（131072）。用 131072 会报 `input (131073) longer than context length` warmup 失败。128K/8K（136K>131072）根本超 context——要么 input 降到 ~120K（本次做法），要么 decode/prefill 起服务时把 `--context-length` 提到 ≥140K。
  - 前置踩坑：pool 里有节点残留 `mnt-disks-ssdN.mount` 会挡 RAID，需重建污染节点（见 RAID-SETUP 文档坑速查）。
- **TTFT 仍高于官方 8.6s**：官方 8.6s 是 conc=1 单请求测的；本配置 max-throughput 高并发下 prefill（只 7-8 个）排队 → 20-51s。**注意：R1 226 官方配置就是 8P + DEP32（recipe `ctx8_pp4_gen1_dep32`），跟我们 Round 4 拓扑一致**——差距在测法（单请求 vs 高并发），不在 P:D 数量。想同时拿高吞吐 + 低 TTFT 要靠 Context Parallelism（降单请求 prefill 延迟）。（`10P1D` 是**后续 DeepSeek-V4 博客**的配方 `disagg-gb300-10p1d-dep4-dep32`，别跟 R1 混，见 §9.6 第 4 点。）

### 9.6 官方博客到这一步之后的下一步（Roadmap）
lmsys/NVIDIA 拿到 226 TPS/GPU（128K/8K，无 MTP）之后的演进：
1. **MTP（EAGLE spec decode）**（同篇博客）：per-user 吞吐 23→43 tok/s（**+87%**），TPS/GPU 维持峰值。← **我们的 Round 5，✅ 已做，见 §9.7（实测 2.15×）**。
2. **同时压 TTFT + 高吞吐**：Context Parallelism（替代 chunked PP，无气泡降 TTFT）、DP load balancer、Wide-EP 更深 overlap、spec-aware 动态 draft token。
3. **服务栈成熟化**：per-concurrency 配方分派、**P:D 配比调优（10P1D 级）**、Dynamo 编排（KV-aware 路由 + DP-rank 对齐）、breakable CUDA graph 覆盖 prefill。
4. **换新模型 DeepSeek-V4**（2026-06 后续博客）：Day-0 支持 → 两个月内 kernel/runtime/bugfix 迭代，在 GB300 disagg lane（V4 Pro FP4，**8K/1K** workload，带 MTP）达 **11,200 tok/s/GPU @ ~50 tok/s/user**（Day-0 的 5×）。关键：MHC fusion、KV Compression V2、W4A4 MegaMoE、SWA budgeting、per-concurrency 配方。复现用 NVIDIA `srt-slurm` + Dynamo，配方 `disagg-gb300-10p1d-dep4-dep32`。

### 9.7 MTP（EAGLE spec decode）— Round 5，✅ per-user 2.15×

官方 Roadmap 第 1 步。**只在 decode 侧加 spec flags**，prefill 不动；拓扑复用 §9.1（7P+DEP32）。

**decode 脚本在 §9 decode 基础上追加（放到 launch 命令里）：**
```bash
--speculative-algorithm EAGLE \                       # DeepSeek MTP 走 EAGLE（NEXTN 是别名）
--speculative-draft-model-path /mnt/ssd/DeepSeek-R1-0528-NVFP4-v2 \   # 自带 nextn，指同模型
--speculative-draft-model-quantization modelopt_fp4 \
--speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
--speculative-moe-a2a-backend none --speculative-moe-runner-backend triton   # ← 关键，见下
```

> **⚠️ 最大的坑：draft nextn MoE backend 配对**
> DeepSeek FP4 checkpoint 的 **nextn(draft) 层权重是 bf16**（不是 fp4），跟主 decode 的 MoE 不是一套路径。乱配（`a2a=deepep` 单加、`a2a=flashinfer`、`deepep+runner=triton`）全崩在
> `AssertionError: forward_deepgemm_masked is deprecated`（`ep_moe/layer.py`）→ `scheduler died (exit -3)`。
> 只有两个合法配对（来自 SGLang `arg_groups/overrides.py` 自身的 draft backend 推断逻辑）：
> 1. `--speculative-moe-runner-backend deep_gemm --speculative-moe-a2a-backend deepep`（需 ep>1）
> 2. `--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none`（**官方默认，推荐**）
> 本指南用配对 2，一次通。起来后 decode 日志出现 `Capture target verify CUDA graph`，warmup 响应带 `spec_accept_length`。

**重启 decode（8 节点，别忘了 router 也在 d0，会被一起 kill）：**
```bash
for i in 0..7: kubectl exec sgl4-d$i -- pkill -9 python          # ⚠️ 会连带杀掉 d0 上的 router
for i in 0..7: nohup setsid bash /tmp/decode5.sh $i $D0IP:5757 &  # rank0=d0
# decode 全 ready 后，重启 router（§6）
```

**实测（no-MTP vs MTP，同 bench 2048/1024）：**
| 场景 | 指标 | no-MTP | MTP | 收益 |
|---|---|---|---|---|
| conc=1（最纯信号） | tok/s/user | 97.4 | **209.2** | **2.15×** |
| | TPOT ms | 10.0 | **4.53** | 2.2× |
| conc=128（loaded） | 聚合 tok/s | 3599 | **4260** | +18% |
| | TPOT ms | 12.2 | **6.19** | 2.0× |

- **accept_length ≈2.7**（每 decode step 出 ~2.7 token 而非 1）→ per-user 解码 2.15×，超官方 1.87×（random+temp0 接受率偏高，真实业务会低些）。
- conc=1 无 prefill 排队，是最干净的 MTP 信号；conc=128 因 7P 使系统 prefill-bound、decode 有余算力，MTP 聚合吞吐不掉反涨 +18%。
- **显存**：MTP 多 draft tree + verify CUDA graph，若 OOM 先降 `--cuda-graph-max-bs`/`--speculative-num-draft-tokens`，最后才调 `--mem-fraction-static`。本配置 mem-fraction 0.75 未 OOM。

### 9.8 技术栈清单：官方 vs 本文，用了啥 · 没用啥

> 核对自官方博客 [*Deploying DeepSeek on GB300 NVL72: Big Wins in Long-Context Inference*](https://www.lmsys.org/blog/2026-02-19-gb300-longctx)（lmsys，2026-02-19）原文。目的：说清楚哪些组件真用了、哪些是"听着相关但没用"，避免误会。

| 组件 / 技术 | 本文用了吗 | 官方用了吗 | 说明 |
|---|---|---|---|
| **PD 分离** | ✅ | ✅ | prefill/decode 拆开，两边各自优化 |
| **PP chunked prefill + dynamic chunking** | ✅（128K）| ✅ | 长上下文降 TTFT；官方 32K dynamic chunk 拿到 8.6s |
| **Wide-EP（DeepEP）decode** | ✅ DEP32 | ✅（up to 32 GPU）| MoE + KV 摊到更多 **GPU HBM**，不是 CPU |
| **FP8 KV cache（HBM 内）** | ✅ `fp8_e4m3` | ✅ native FP8 KV | 省显存塞更多 KV；全程在 HBM |
| **MTP（EAGLE spec decode）** | ✅ 2.15× | ✅ 1.87×（accept 2.37@MTP3）| 本文 accept ~2.7 略高（random+temp0）|
| **mooncake 传输后端（NVLink）** | ✅ `MEM_POOL=NVLINK` | 未明说（用 Dynamo 编排）| KV 经域内 NVLink C2C，**GPU→GPU** |
| **NVLink C2C KV 传输** | ✅ | ✅（域内）| 绕开 RoCE |
| **nixl** | 装了但**非活跃路径** | — | 一开始试 RoCE 卡死，改 mooncake NVLink；nixl 仅作 mooncake 底层依赖存在 |
| **编排层** | sglang_router | **NVIDIA Dynamo** | 官方用 Dynamo（KV-aware 路由 + 生命周期）；本文用轻量 router 够复现 |
| **KV Cache Offload（CPU/DRAM/SSD）** | ❌ **没用** | ❌ **没用** | 见下 |
| **CPU 内存做 KV** | ❌ | ❌ | 官方整个 decode 分析都是 HBM-bound（176GB KV pool / 40 req/GPU），没往 CPU 倒 |
| **多轮对话 benchmark** | ❌ | ❌ | 双方都是**单轮** 128K/8K；官方精度用 LongBench-v2（单轮长文问答，非多轮聊天）|
| **Context Parallelism** | ❌ | ❌（列为 Future Work）| 替代 chunked PP 降 TTFT 的下一步 |

**两个最容易误会的点：**

1. **Mooncake 有两副面孔，别混**：
   - **Mooncake 传输后端**（本文用的）= 把 KV 经 NVLink 从 prefill 卡直传 decode 卡，GPU→GPU，纯 HBM。
   - **Mooncake Store**（本文**没用**）= 把 KV 往 CPU / DRAM / SSD 三级池子里存做 offload。
   - 同名，两个东西。本文和官方都只用了前者传输。

2. **KV Cache Offload 到 CPU 是 SGLang 真有的能力，但这条线没碰**：
   - SGLang 的 **HiCache**（配 Mooncake Store L3）能把 KV 卸到 CPU/DRAM/SSD，专为**多轮对话 / 共享前缀复用 / 降 TTFT**。
   - 但这是**独立功能**，跟本文/官方跑的"最高吞吐 + 长上下文单轮"是两个场景。本文还显式 `--disable-radix-cache`（连 GPU 内前缀复用都关了，纯拼 decode 吞吐）。
   - 想测 KV offload / 多轮，得另开一条线：开 HiCache + Mooncake Store，换多轮 workload。

---

## 10. 关键坑速查（为什么每步都不能省）

| 现象 | 根因 | 解 |
|---|---|---|
| `No module named sgl_kernel` / `RMSNorm: no kernel image` | nemo 镜像丢了 sm_103 kernel；PyPI sgl-kernel 无 sm_103a cubin | 用官方 `v0.5.15.post1-cu130`（含 sm_103a）|
| `undefined symbol: c10_cuda_check_implementation` | NV NGC torch 改了 C10 ABI | 官方镜像的标准 torch 匹配，别用 nemo |
| `NIXL_ERR_BACKEND` / RDMA backend 创建失败 | 官方 Ubuntu 镜像缺 CX-8 的 mlx5 verbs | 装 `doca-ofed-userspace`（§4 step 3）|
| 单请求 60s 超时 / `KVTransferError: Aborted` | nixl 走 RoCE：GKE 是 RoCE v2 over IPv6，netdev 名 `gpuNipvlanM`，UCX 调不通 | **改走 NVLink**：`--disaggregation-transfer-backend mooncake` + `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` + `MC_FORCE_MNNVL=1` |
| pod `Insufficient memory` Pending | 内存请求 > 节点 909Gi allocatable | 3P2D/decode 600Gi；128K prefill 700Gi（模型在 Local SSD 不占 RAM）|
| `OOMKilled`(exit137) 加载时 | sglang 加载把权重读进 host 缓冲，峰值超限 | 内存 request 用 **≥600Gi**（200Gi 必炸）；128K prefill 700Gi；**一 pod 只启一次**别堆进程 |
| 模型放内存盘吃满 RAM / V4 大模型塞不下 | `emptyDir medium:Memory` 存权重是浪费 | **模型放 Local SSD RAID**（§3.1 默认）；RAM 留给 KV cache |
| pod `Evicted` DiskPressure（拉镜像时）| 12.7GB 镜像顶爆 boot 盘 | fresh 节点池 + 删重建（可能重试几轮）|
| pod 卡 `ContainerCreating` 报 `ResourceClaim not created yet` | 16 pod 同时申请 DRA，controller 滞后 | 删卡住 pod + `apply` 重触发（§9.3）|
| `pkill` 后 exit 137、server 没重启 | `pkill -f sglang.launch_server` 匹配到自己命令行 | 用 `pkill -9 python` |
| `kubectl cp` 脚本没落地/截断 | cp 静默失败 | cp 后**必须** `wc -l` 校验 |
| MTP：`forward_deepgemm_masked is deprecated` / `scheduler died (exit -3)` | draft nextn MoE 是 bf16，backend 配错撞 deprecated 死路 | draft 用 `--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none`（§9.7）|
| 重启 decode 后 benchmark `Server not ready` | router 跑在 d0，被 `pkill -9 python` 连带杀了 | decode ready 后重启 router（§6）|

---

## 11. 清理

```bash
$K delete pod -l app=sgl3 --force --grace-period=0   # 或 app=sgl4
$K delete -f /tmp/sgl3-mem.yaml   # 连带删 ComputeDomain / Service / RCT
```

---

*沉淀自 2026-07-19 实测。§2–§8 = 3P2D（20 GPU）干净复现，两次从零验证一致；§9 = 放大到 64 GPU 128K（warm 312 TPS/GPU 超官方 226 达 38%）+ §9.7 MTP（per-user 97→209 tok/s，2.15×）。全景消融见 §0.5，失败尝试全记录见 RUNLOG。R1 官方两大成果（226 baseline + MTP）已复现且超过，官方 R1 配置就是 8P+DEP32、与本文 Round 4 一致。未做的下一步：Context Parallelism 替代 chunked PP 降单请求 TTFT；再往后是换模型 DeepSeek-V4（10P1D recipe，另一条线）。*
