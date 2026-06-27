# GB200 NVL72 生产运维知识库

## 1. 概述

本节汇集 GB200 NVL72 (A4X) 集群从部署到生产运维过程中积累的**全部高价值工程经验**，覆盖:

- **16 个反复出现的生产级坑**及其修复方案 (已 baked 到脚本/yaml)
- **XID 故障自动恢复**流程 (Xid 137/145/94 NVLink fabric 硬件故障)
- **DRA Controller Race** 的检测与 2-stage 修复
- **Base Image 烘焙**方法 (7 Phase, 新 VM 启动 < 2 min)
- **Workload 一键部署** wrapper (16min stuck -> 4min Ready)
- **cuBLAS GEMM Benchmark** 单 GPU 峰值算力验证
- 全套 benchmark 结果速览 (cuBLAS / NCCL / Megatron / DeepEP)

**硬件环境**: GB200 NVL72 (Grace ARM CPU + 2 Blackwell GPU superchip), 每节点 4 GPU + 4 MRDMA NIC + 2 GVNIC。18 节点组成 1 个 NVL72 super-pod, 共享 NVLink Switch fabric (~840 GB/s MNNVL 跨节点互联)。k8s 1.34 + NVIDIA DRA Driver v0.4 ComputeDomain CRD 管理跨节点 IMEX session。

---

## 2. 16 个高价值经验教训

以下 16 个坑在 NVL72 集群部署和压测中**反复出现**，已全部 baked 进 script/yaml/wrapper。

### 2.1 NVIDIA DRA v0.4 controller race -- 每次切 workload 都撞

| 项目 | 内容 |
|---|---|
| **现象** | Pod schedule 后卡 `PodInitializing > 5min`, event 报 `FailedPrepareDynamicResources DeadlineExceeded`, daemon DS DESIRED=0 永不 spawn |
| **根因** | DRA Driver v0.4 controller 内部 optimistic locking 失败: `Operation cannot be fulfilled on computedomains... the object has been modified (attempt 1)`, silent drop work item, 不给 client pod 所在 node 加 `resource.nvidia.com/computeDomain=<CD-UID>` label |
| **修法 (2-stage)** | Stage 1: `kubectl label node <pod-node> resource.nvidia.com/computeDomain=<CD-UID>` 手动补 label, daemon DS 开始 spawn; Stage 2: `ssh node 'sudo systemctl restart kubelet'` 让 daemon pod 从 ContainerCreating unstuck |
| **已 baked** | `check-k8s-dra-health.sh --fix-race` 自动执行 2-stage; `prepare-workload.sh` apply 后自动调用 fix-race; `dra-cd-label-reconciler-deployment.yaml` 集群常驻每 15s polling 自动补 label |

### 2.2 reconciler 5h 无工作 -- `kubectl get cd` 短名 silent fail

| 项目 | 内容 |
|---|---|
| **现象** | Reconciler pod Running 但 5 小时无 reconcile 痕迹 |
| **根因** | ConfigMap 脚本用 `kubectl get cd -A`, 但 server 没注册 `cd` 短名 (只有 `computedomain` 全称), 返回 error 到 stderr, 解析为空, silent skip 1200+ 次 |
| **修法** | 必须用全名 `kubectl get computedomain -A`, 加 `grep -v "^error"` 过滤 |
| **已 baked** | `dra-cd-label-reconciler.sh` 和对应 deployment yaml 已修正 |

### 2.3 GIB libnccl ABI mismatch -- DeepEP / Megatron hard assert

| 项目 | 内容 |
|---|---|
| **现象** | Pod import 触发 `AssertionError: Invalid NCCL versions: /usr/local/gib/lib64/libnccl.so.2.30.4 (loaded) v.s. ... (expected)` |
| **根因** | GIB init container 注入的 libnccl 跟镜像内 pip `nvidia-nccl-cu13` **版本号同但 binary 内容不同** (GIB 是自 build 版本), DeepEP `check_nccl_so()` 用 `filecmp.cmp(shallow=False)` 严格比对 hard fail |
| **修法** | `export LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2` 让 dlopen 优先用 pip libnccl; 同时 `rm -rf /opt/DeepEP` 避免 import 触发 |
| **已 baked** | `gen-megatron-sts.py` pod entry 已加 LD_PRELOAD + rm -rf |

### 2.4 Megatron multi-node iter log 在 last rank pod, 不是 master

| 项目 | 内容 |
|---|---|
| **现象** | Multi-node 跑完 `kubectl logs <master-pod>` 没 iter log |
| **根因** | Megatron `training_log()` 用 `is_last_rank()` = `rank == world_size - 1`, multi-node 时 last rank 在最后一个 pod (replicas=16 -> g1-15) |
| **修法** | 收 log 用 `kubectl logs <pod>-g1-{N-1}`; 注意正常 Exit 0 的 Complete pod 直接 `kubectl logs`, **不要加 `--previous`** (只看 restarted container 上次 instance, 正常 Complete 返回空) |
| **已 baked** | `prepare-workload.sh` 文档 + Megatron results README 明确标注 |

### 2.5 Megatron `--eval-iters 0` 触发 `int // None` crash

| 项目 | 内容 |
|---|---|
| **现象** | Build dataloader 时报 `TypeError: unsupported operand for //: 'int' and 'NoneType'` |
| **根因** | Megatron-core 0.16 `get_train_valid_test_num_samples()` 算 `args.train_iters // args.eval_interval`, `--eval-iters 0` 不真跑 eval 但 `args.eval_interval` 默认 None 触发除零 |
| **修法** | 显式 `--eval-interval 1000` (任意正数, 配合 eval-iters=0 不真 eval) |
| **已 baked** | `gen-megatron-sts.py` 默认加 |

### 2.6 `--log-throughput` multi-node + mock-data silent skip iter log

| 项目 | 内容 |
|---|---|
| **现象** | 加 `--log-throughput` 想自动 print TFLOPs/GPU, 但 multi-node + mock-data 完全跳过 iter log |
| **根因** | Megatron-core 0.16 bug, multi-node 路径下 throughput 计算异常静默吃掉 iter log |
| **修法** | 不加 `--log-throughput`, 用 6N formula 手算 TFLOPs: `TF/GPU = 6 * N_params * (gbs * seq) / (avg_iter_sec * num_gpus) / 1e12`。Single-node 加 flag OK |
| **已 baked** | yaml generator 已移除 `--log-throughput` |

### 2.7 Xid 137/145/94 -- GPU NVLink fault, 一键自动恢复

| 项目 | 内容 |
|---|---|
| **现象** | Pod CrashLoopBackOff, dmesg 报 `NVRM: Xid (PCI:0000:01:00): 137, ...` (NVLink fabric error) |
| **根因** | NVL72 MNNVL fabric / NVLink Switch transient error, device-plugin 标 GPU Unhealthy, 节点 `nvidia.com/gpu` 减少, workload schedule fail |
| **修法** | 3 方案递进: (1) `nvidia-smi --gpu-reset` (多数恢复); (2) `lspci -d 10de: \| xargs setpci ... LNKCTL` reset PCI link (少数需要); (3) 节点 reboot (极少最后手段) |
| **已 baked** | `xid-fault-auto-recover.sh --apply` 自动: detect bad nodes -> verify Xid -> serial reset -> force delete stale device-plugin pod -> poll GPU=4 |

### 2.8 stale Terminating CD daemons 不自动 GC

| 项目 | 内容 |
|---|---|
| **现象** | ComputeDomain 删除后 daemon pod 在 `nvidia-dra-driver-gpu` namespace 卡 `Terminating` finalizer, 累积 30+ stale pods |
| **根因** | NVIDIA DRA v0.4 daemon DS 的 finalizer 跟 CD GC race, 删 CD 时 daemon 没 cascade clean |
| **修法** | `kubectl delete pod -n nvidia-dra-driver-gpu --force --grace-period=0` |
| **已 baked** | `prepare-workload.sh` Step 1 每次 apply 前自动清理 stale Terminating daemons |

### 2.9 Calico VXLAN 默认走 tailscale 不走 VPC

| 项目 | 内容 |
|---|---|
| **现象** | 跨节点 pod 网络慢 (~50 MB/s 而不是 VPC 几 GB/s), VXLAN 封装走了 wireguard |
| **根因** | Tigera operator `nodeAddressAutodetectionV4: {firstFound: true}` (default) 选第一个 nonloopback NIC, 如果 tailscale0 先 enum 就选它 |
| **修法** | 显式配置 `nodeAddressAutodetectionV4: {kubernetes: NodeInternalIP}` 从 kubelet 拿正确 IP |
| **已 baked** | 部署文档 kubeadm init + Calico 安装步骤已强制此配置 |

### 2.10 tailscale MagicDNS hijack /etc/resolv.conf 致 dnf install 全挂

| 项目 | 内容 |
|---|---|
| **现象** | 装完 tailscale 后 `dnf install` 报 `Cannot prepare internal mirrorlist: Curl error (6): Couldn't resolve host name` |
| **根因** | tailscale 接管 /etc/resolv.conf, MagicDNS resolver 不查内部域 (内部 mirror 在公网 DNS NXDOMAIN, 只有 GCE metadata DNS 有) |
| **修法** | `tailscale up --accept-dns=false` 不接管 resolv.conf |
| **已 baked** | startup 脚本中 tailscale 配置已加 `--accept-dns=false` |

### 2.11 跨 2 clique alltoall hang -- NVIDIA Known Issue, 无 workaround

| 项目 | 内容 |
|---|---|
| **现象** | 36 节点跨 2 clique NCCL alltoall stuck, 无 error/timeout/exit |
| **根因** | NVIDIA 官方确认的 NCCL race condition (跨 2 clique atomic lock 缺), 当前 NCCL release 无修复 |
| **修法** | (1) alltoall 限同 clique (<=72 GPU); (2) 跨 clique 只跑 3 collective (all_reduce / all_gather / reduce_scatter); (3) 36 节点 chain 4 collective 用 split yaml (3coll + alltoall 分跑) 100% PASS |
| **已 baked** | yaml generator 提供 `3coll` / `alltoall` split 模式 |

### 2.12 Monitor 僵尸 -- grep pattern 只覆盖终态, stuck 信号没覆盖

| 项目 | 内容 |
|---|---|
| **现象** | Pod 卡 init 16 min, Monitor 1h timeout 才察觉 |
| **根因** | Monitor 命令 `until kubectl logs \| grep "iteration 50"` 只等成功信号, pod 卡 init 不出 iteration log, monitor 全程沉默到 timeout |
| **修法** | grep pattern 必须覆盖三类: **进度** (`iteration X/Y`, `step N`, `TFLOPs`), **终态** (`DONE`, `FAIL`, `Traceback`, `exit code`), **stuck** (`FailedScheduling`, `FailedPrepareDynamicResources`, `ImagePullBackOff`, `CrashLoopBackOff`, `DeadlineExceeded`, `OOM`, `NCCL.*timeout`) |
| **已 baked** | 全局运维规范 Monitor section 已加固反僵尸 checklist |

### 2.13 DRA kubelet-plugin Running 但 CD ResourceSlice 缺 publish

| 项目 | 内容 |
|---|---|
| **现象** | 物理全 healthy, DRA 全 Running, 但 18n workload 5/18 卡 Pending, Event `5 cannot allocate all claims`, controller log `CDStatusSync: total nodes=13` 封顶 |
| **根因** | DRA driver v0.4.0 kubelet-plugin `compute-domains` 容器偶发 stuck 在 informer `Watch close` 重连, Running 但实际没跑 `Publishing ResourceSlice` |
| **诊断** | `kubectl get resourceslice --no-headers \| grep compute-domain.nvidia.com \| wc -l` (期望 = worker 数, 少了就有 plugin 没 publish) |
| **修法** | 精准 delete 缺 slice 节点上的 plugin pod, DS 重 spawn fresh pod (~15s) 触发 fresh publish。比 kubelet restart 轻得多 (kubelet restart blast radius 大) |
| **已 baked** | `check-k8s-dra-health.sh --fix-rs` 自动诊断 + 定向 delete + wait publish |

### 2.14 host systemd nvidia-imex.service 跟 ComputeDomain daemon 抢 IMEX session

| 项目 | 内容 |
|---|---|
| **现象** | Workload pod 卡 `ContainerCreating`, kubelet event `AssertComputeDomainReady` fail, daemon log 反复 `NV_ERR_IN_USE Failed to allocate Imex session` |
| **根因** | Host 上 systemd `nvidia-imex.service` (driver 安装包默认 enable) 跟 CD daemon pod 抢同一个 IMEX session, driver 只允许一个 owner |
| **修法** | Worker base image 烘焙时 disable + mask: `systemctl disable --now nvidia-imex && systemctl mask nvidia-imex` |
| **已 baked** | `image-build.sh` Phase 6 已固化; 老 worker 可批量 SSH 补救 |

### 2.15 DRANET 不 inject netlink interface -- DeepEP raw ibverbs 跨 host fail

| 项目 | 内容 |
|---|---|
| **现象** | NCCL via GIB plugin 跨 host OK (326 GB/s), 但 DeepEP test_ep 跨 host 报 `ibv_modify_qp errno 110 Connection timed out` |
| **根因** | DRANET v1.3.0 只 inject ibverbs char device (`/dev/infiniband/uverbs0..3`), **不 move netlink interface (bond2-5) 到 pod ns, 也不配 IP**。Pod 内 GID 只有 link-local IPv6, RoCEv2 IPv4 全空。NCCL via GIB 走 `rdma_cm` + IB CM 绕开了此限制; DeepEP raw `ibv_modify_qp` 需要 routable GID |
| **修法** | DeepEP / raw ibverbs workload 用 `hostNetwork: true` 绕开 DRANET, 直接用 host 上有 IP 的 bond2-5 |
| **已 baked** | NCCL/Megatron workload 用 DRANET ResourceClaim (work); DeepEP workload yaml 用 `hostNetwork: true` |

### 2.16 GB200 kernel cmdline `iommu.passthrough=1` 让 UVM ATS bind fail -- CUDA 启动即 701

| 项目 | 内容 |
|---|---|
| **现象** | 全部 CUDA app 启动即报 `cudaErrorDevicesUnavailable (701)`, 即使 fabric Completed / imex active / driver fresh init。dmesg 反复刷 `uvm_ats_sva_bind_gpu` stack |
| **根因** | kernel cmdline `iommu.passthrough=1 iommu.strict=0` (NVIDIA Grace tuning doc 推荐的 "performance" 设置) **break UVM ATS bind**。Grace doc 原文 "might have issues with ATS" -- 对 GB200 Grace+Blackwell 重度 UVM ATS 场景是 hard break |
| **排除项** | fabricmanager fail (GB200 compute node 预期), 缺 CD CR, nvidia-imex inactive, driver state cache, containerd 版本 -- 均排除 |
| **修法** | Kernel cmdline 删 `iommu.passthrough=1 iommu.strict=0`, 只保 `init_on_alloc=0`。让 SMMUv3 走 default translation mode。代价: 理论 PCIe DMA latency 略增, 但 GB200 GPU-CPU 通信走 NVLink-C2C (Grace<->Blackwell 单芯片内 900 GB/s), 实际影响极小 |
| **已 baked** | `image-build.sh` Phase 2: `grubby --update-kernel=ALL --remove-args="iommu.passthrough iommu.strict"` |

---

## 3. XID 故障自动恢复

`scripts/troubleshooting/xid-fault-auto-recover.sh` 实现 NVLink Xid 硬件故障的**全自动检测和恢复**, 无人值守。

### 3.1 处理的 XID 错误码

| XID Code | 含义 |
|---|---|
| **137** | NVLink fabric error (跨节点 MNNVL fabric 通信故障, sticky) |
| **145** | NVLink device error (设备级 NVLink 故障, sticky) |
| **94** | NVLink/device-plugin 关联故障 |

这些 Xid 是 **sticky** 的 -- GPU device-plugin 把受影响 GPU 标 Unhealthy, 节点 `nvidia.com/gpu` allocatable 从 4 降到 3 或更低, 新 workload 无法调度。

### 3.2 恢复流程 (6 步)

```
Step 1: Detect -- kubectl get nodes 找 nvidia.com/gpu allocatable < 4 的节点
     |
Step 2: Verify -- SSH 到 bad node, dmesg grep Xid 137/145/94 确认是 NVLink fault
     |             (非 NVLink fault 的节点跳过, 需人工排查)
     |
Step 3: Reset -- gcloud compute instances reset (host reboot)
     |           *** 默认串行, 避免 IMEX storm cascade ***
     |
Step 4: Wait -- poll 节点 Ready=True (max 900s)
     |
Step 5: Cleanup -- force delete stale device-plugin pod (Unknown/Pending/Terminating)
     |              让 DaemonSet spawn fresh pod
     |
Step 6: Verify -- poll GPU allocatable=4 (max 5 min, 每 15s 检查)
```

### 3.3 关键: 为什么必须串行 reset

**IMEX storm cascade**: NVL72 super-pod 内 18 节点共享 MNNVL fabric, 多节点同时 reset 导致 nvidia-imex daemon 同时 stop, 触发跨集群 MNNVL fabric session race, **在原本健康的节点上产生新的 NVLink Xid fault**:

- 12 节点并行 reset -> 10 个**新** fault (45% trigger rate)
- 10 节点并行 -> 3 个新 fault
- 单节点串行 -> 0 cascade

默认串行 (`--parallel` 可覆盖, 但会 cascade)。串行每节点 ~10 min, 1 pass 收敛; 并行虽快但需多 pass 才收敛。

### 3.4 用法

```bash
# Dry-run: 检测受影响节点, 不动
bash xid-fault-auto-recover.sh

# 应用修复 (串行, 推荐)
bash xid-fault-auto-recover.sh --apply

# 单节点修复
bash xid-fault-auto-recover.sh --node <worker-hostname> --apply

# 并行修复 (快但可能 cascade)
bash xid-fault-auto-recover.sh --apply --parallel
```

---

## 4. DRA Controller Race 修复

NVIDIA DRA Driver v0.4 的 controller 存在 optimistic locking race condition, 是 NVL72 集群最高频的运维问题。

### 4.1 问题机制

```
workload pod 申请 ComputeDomain channel ResourceClaim
     |
DRA Controller 尝试 Update computedomain CR
     |
optimistic locking 失败: "the object has been modified (attempt 1)"
     |
Controller silent drop work item (不重试)
     |
Node 上缺 resource.nvidia.com/computeDomain=<CD-UID> label
     |
daemon DaemonSet DESIRED=0, daemon pod 永不 spawn
     |
Pod 永久卡 PodInitializing / ContainerCreating
```

### 4.2 `check-k8s-dra-health.sh` 检测

脚本 Check 12 专门检测此 race:

1. 遍历所有 active ComputeDomain
2. 查每个 CD 的 daemon DaemonSet DESIRED 值
3. 统计关联 client channel ResourceClaim allocated 数
4. **DESIRED=0 但 client_RC_alloc>0** = 确认 controller race

```bash
# 检测
bash check-k8s-dra-health.sh

# 检测 + 自动修复
bash check-k8s-dra-health.sh --fix-race
```

### 4.3 2-Stage 修复流程

```bash
# Stage 1: 手动补 label (让 daemon DS schedule)
kubectl label node <node> resource.nvidia.com/computeDomain=<CD-UID> --overwrite

# 等 30s, 检查 daemon pod 是否 Ready
# 如果仍 ContainerCreating...

# Stage 2: restart kubelet (刷新 PrepareResource cache)
ssh <node> 'sudo systemctl restart kubelet'
```

### 4.4 持续自动修复

部署 `dra-cd-label-reconciler-deployment.yaml` 到集群, 每 15s polling:

1. `kubectl get computedomain -A` 获取所有 CD 及其 UID
2. `kubectl get resourceclaim -A` 查 allocated channel 关联的 node
3. 检查 node 是否有 `resource.nvidia.com/computeDomain=<CD-UID>` label
4. 缺失则自动 `kubectl label node`

---

## 5. Image 烘焙 (Base Image Baking)

`scripts/host/image-build.sh` 把所有 host 级硬件配置一次性写进 OS image, 新 VM 启动 < 2 min 即可被 k8s join。

### 5.1 为什么要烘焙

不做镜像: 每 VM 每次启动装 ~3 GB 软件包 + grub 更新 + initramfs 重建 + reboot, ~10 min/节点。做完镜像: 新 VM < 2 min Ready。

### 5.2 7 个 Phase

| Phase | 内容 | 关键操作 |
|---|---|---|
| **1** | OS base 软件包 + sshd + selinux + filesystem 扩展 | `dnf install`, growpart, selinux disabled |
| **2** | Kernel cmdline | `grubby --remove-args="iommu.passthrough iommu.strict"` + `--args="init_on_alloc=0"` (**关键**: iommu.passthrough 会 break UVM ATS, 见 2.16) |
| **3** | IMEX initramfs | `echo "options nvidia NVreg_CreateImexChannel0=1" > /etc/modprobe.d/nvidia.conf` + `dracut --force --add-drivers "gve"` (否则 `/dev/nvidia-caps-imex-channels` 不存在) |
| **4** | 通用 prereqs | swap off, modules-load (overlay, br_netfilter), sysctl ip_forward |
| **5** | Grace 持久 sysctl | 写入 `/etc/sysctl.d/90-grace-gb200.conf`: `kernel.numa_balancing=0`, TCP BBR, buffer 128MB, `vm.max_map_count=1048576` 等 |
| **6** | Reapply systemd service | 注册 `grace-gb200-reapply.service`, 每开机自动跑: PCI ACS 关, CPU governor=performance, THP=madvise, nvidia_peermem, `nvidia-smi -pm 1`; **同时 disable nvidia-imex** (让 CD daemon 接管, 见 2.14) |
| **7** | NIC rename | PCI BDF-based systemd `.link` 文件, 6 NIC -> bond0..bond5 (2 GVNIC + 4 MRDMA); A4X 硬件 PCI topology 固定, 一次 baked 所有同型 VM 通用 |

额外 Phase 8 (NIC ring + NVMe tuning): ethtool ring rx/tx=8192, mlx5 combined channels 设置, NVMe poll_queues=16 + coalescing; disable irqbalance。

### 5.3 Baked vs Runtime

| 分类 | 项目 | 来源 |
|---|---|---|
| **Image 内 baked** | OS 包 / kernel cmdline / IMEX initramfs / sysctl / NIC .link rename / systemd reapply service / nvidia-imex disabled | `image-build.sh` |
| **Runtime (每开机 systemd 自动)** | PCI ACS 关 / CPU governor=performance / THP=madvise / nvidia_peermem / nvidia-smi -pm 1 | `grace-gb200-reapply.service` (Phase 6 注册) |
| **客户自行装 (不在 image 内)** | Container runtime (containerd 2.x, **需开 NRI**) / nvidia-container-toolkit / CNI plugins / kubelet + kubeadm + kubectl / k8s join | 客户自有 k8s flow |

### 5.4 用法

```bash
# 1. 创建 build VM (用 vendor 预装 driver 的 base image)
gcloud compute instances create gb200-image-builder \
  --machine-type=a4x-highgpu-4g --image=<vendor-base-image> ...

# 2. 推送 + 执行 (~5 min)
gcloud compute scp scripts/host/image-build.sh gb200-image-builder:/tmp/
gcloud compute ssh gb200-image-builder --command="sudo bash /tmp/image-build.sh"

# 3. Reboot (让 initramfs + kernel cmdline + NIC rename 生效)
gcloud compute instances reset gb200-image-builder

# 4. 验证
gcloud compute ssh gb200-image-builder --command="sudo bash /tmp/image-build.sh --verify"
# 期望: init_on_alloc=0 / iommu none / IMEX channel0 / numa_balancing=0 / bbr /
#       nvidia-imex disabled / grace-reapply enabled / 6 .link files / bond0..5 /
#       irqbalance disabled / GPU persist Enabled / Fabric Completed

# 5. Snapshot + Create image
gcloud compute instances stop gb200-image-builder
gcloud compute disks snapshot gb200-image-builder --snapshot-names=<snap-name>
gcloud compute images create gb200-worker-base-v1 --source-snapshot=<snap-name>
```

---

## 6. Workload 一键部署

`scripts/troubleshooting/prepare-workload.sh` 是一个 idempotent wrapper, 把 DRA v0.4 所有已知 race condition 的 fix 流程全 baked 进去, 实测将 16min stuck 缩短到 4min Ready。

### 6.1 解决的 4 个问题

1. **stale Terminating CD daemons** -- 前一个 workload 的 daemon pod 卡 Terminating finalizer (DRA v0.4 GC race)
2. **DRA controller race** -- "object has been modified" silent drop, daemon DS DESIRED=0
3. **kubelet PrepareResource cache stale** -- CD label 切换后 kubelet 缓存旧状态
4. **FailedPrepareDynamicResources DeadlineExceeded** -- 新 label 节点上的 deadline 超时

### 6.2 执行流程 (5 步)

```
Step 1: Clean stale Terminating daemons
        kubectl delete pod -n nvidia-dra-driver-gpu --force --grace-period=0
             |
Step 2: Apply workload yaml
        kubectl apply -f <yaml>
             |
Step 3: Wait 25s
        给 DRA controller 足够时间 allocate channel
             |
Step 4: Proactive fix-race
        调用 check-k8s-dra-health.sh --fix-race
        (Stage 1 label + Stage 2 kubelet restart if needed)
             |
Step 5: Wait master pod Running (max 6min)
        每 10s poll, 3min 仍 stuck 则二次 fix-race
```

### 6.3 用法

```bash
# Apply workload + fix + wait
bash prepare-workload.sh <yaml-path> <master-pod-name>

# Delete workload + cleanup
bash prepare-workload.sh -d <yaml-path>
```

---

## 7. cuBLAS GEMM Benchmark

`scripts/host/cublas_bench_gb200.sh` 在 worker host 上**直接跑** (不起 k8s pod) 单 GPU GEMM 峰值算力测试。

### 7.1 工作原理

Host 只有 NVIDIA driver (R580), 没装 CUDA toolkit。脚本通过 `ctr -n k8s.io images mount` 从已 pull 的 container image 中**借用** libcublas / libcublasLt / libcudart (~670 MB) 到 /tmp, 再 `LD_LIBRARY_PATH` 跑 benchmark binary。

```bash
# 从 container image 提取 CUDA libs (不需要安装 CUDA toolkit)
sudo ctr -n k8s.io images mount "$IMAGE" "$MOUNT"
sudo cp -a "$MOUNT"/usr/local/cuda-13.2/targets/sbsa-linux/lib/libcublas* /tmp/cublas_bench/
sudo cp -a "$MOUNT"/usr/local/cuda-13.2/targets/sbsa-linux/lib/libcudart* /tmp/cublas_bench/
sudo ctr -n k8s.io images unmount "$MOUNT"

# 用借来的 libs 跑 benchmark
LD_LIBRARY_PATH=/tmp/cublas_bench bash cublas_bench_gb2_3.sh
```

### 7.2 峰值结果

| Dtype | GEMM (M x N x K) | 实测 TFLOPs | NVIDIA 参考 | 差异 |
|---|---|---|---|---|
| FP4  | 9728 x 16384 x 8192 | **6845** | 6507 | +5.2% |
| FP8  | 9728 x 2048 x 32768 | **3063** | 2805 | +9.2% |
| FP16 | 8192 x 9728 x 16384 | **1492** | 1372 | +8.7% |
| BF16 | 8192 x 9728 x 16384 | **1592** | 1471 | +8.2% |
| TF32 | 8192 x 9728 x 16384 | **733**  | 675  | +8.6% |
| FP32 | 8192 x 9728 x 16384 | **75**   | 75   | 持平 |

全 6 dtype 实测比 NVIDIA 参考高 5-9% (newer cuBLAS optimization)。

### 7.3 用法

```bash
# 在控制机执行, 自动 SSH 到 worker host
gcloud compute ssh <worker-node> --command="bash -s" < scripts/host/cublas_bench_gb200.sh \
  | tee results/cublas/<worker-node>.log
```

---

## 8. Benchmark 结果速览

### 8.1 cuBLAS GEMM (单 GPU peak)

- FP4 **6845** / FP8 **3063** / FP16 **1492** / BF16 **1592** / TF32 **733** / FP32 **75** TFLOPs
- 全 5-9% 高于 NVIDIA 参考

### 8.2 NCCL 集群通信 (16 GiB in-place busbw, GB/s)

| 规模 | GPU 数 | all_reduce | all_gather | reduce_scatter | alltoall |
|---|---|---|---|---|---|
| 1n (4 GPU local) | 4 | 686.73 | 668.86 | 666.88 | 689.13 |
| 2n D=1 (同 super-pod MNNVL) | 8 | **840.12** | 683.37 | 693.12 | 679.93 |
| 2n D=2 (跨 clique RDMA) | 8 | **326.09** | 188.85 | 188.89 | 83.16 |
| 4n D=1 (同 super-pod) | 16 | **900.99** | 687.81 | 704.56 | 681.03 |
| 4n D=2 (跨 clique mixed) | 16 | **328.66** | 194.44 | 194.13 | 35.02 |
| 18n (72 GPU MNNVL) | 72 | **905.05** | 681.38 | 702.67 | 650.96 |
| 36n 2-CD (跨 2 NVL72) | 144 | **688.14** | 704.13 | 699.75 | 40.59 * |

\* 36n alltoall 跨 2 ComputeDomain 是 NCCL 已知 issue (chain pollution), 单 pass vanilla 40 GB/s ok, chain 4 collective 一起跑 fail。

**关键 finding**:
- 同 ComputeDomain 18 节点 all_reduce **905 GB/s**, 接近 MNNVL fabric 理论上限 (~900 GB/s)
- StatefulSet 自驱动 vs standalone 性能等价 (+/-1% noise), 节省 90% yaml 行数 (114 vs 1129)
- DDP 多机扩展线性 (同 domain sts vs standalone < 1% 差异)

### 8.3 DeepEP v2 (test_ep)

| 规模 | Dispatch SU avg | Combine SU avg | Combine latency avg | vs 1n |
|---|---|---|---|---|
| 1n (4 GPU) | 668 | 701 | 69 us | baseline |
| 2n MNNVL=0 (RDMA only) | 78 | 78 | 1001 us | - |
| 2n MNNVL=2 (production) | **605** | **653** | **100 us** | -7% BW |
| **16n** (64 GPU MNNVL=2) | **521** | **540** | **154 us** | -23% |

- 同 clique 必须 MNNVL=2 (NCCL GIB 转 MNNVL fabric, 性能 8x 于 RDMA)
- 16n combine 540 GB/s, 相对 2n=653 为 83% efficient (8x scale-up 仅损 17%)
- 144/144 test cases PASS, 0 errors

### 8.4 Megatron-LM 训练 (5 config, mock data, 50 iter, seq=4096, bf16)

| Model | GPU | 并行 | ms/iter | tok/s/GPU | TFLOPs/GPU | MFU |
|---|---|---|---|---|---|---|
| llama2-7b  | 4  | tp1pp1 dp4 | 9406  | 27,872 | **1127** | 45.1% |
| llama2-7b  | 8  | tp1pp1 dp8 | 4881  | 26,861 | **1086** | 43.4% |
| llama2-13b | 4  | tp1pp1 dp4 | 18978 | 13,817 | **1079** | 43.2% |
| llama2-13b | 8  | tp1pp1 dp8 | 9614  | 13,636 | **1065** | 42.6% |
| llama3-70b | 64 | tp4pp2 dp8 | 28355 | 2,312  | **971**  | **38.9%** |

**关键 observation**:
- 7B/13B 稳定 **42-45% MFU**, DDP 多机扩展 96-99% (MNNVL fabric NVLink 跨 host 几乎 zero overhead)
- 70B tp4pp2 落到 **39% MFU** (PP=2 pipeline bubble + 大模型 attention 占比上升)
- 未启用 selective recompute / FP8 / longer seq, 还有 5-10% 提升空间

---

## 9. 脚本索引

### `scripts/host/` -- Host 级操作 (不起 pod)

| 脚本 | 用途 |
|---|---|
| `image-build.sh` | **GCE custom image 烘焙** (7 Phase + verify). 把 kernel cmdline / IMEX / sysctl / NIC rename / systemd 全 baked |
| `cublas_bench_gb200.sh` | Host 直跑 cuBLAS GEMM peak (6 dtype). 自动从 container image ctr mount 借 CUDA libs |
| `create-worker-vm-prod.sh` | 批量 `gcloud compute instances create` worker VM (含 metadata 注入) |
| `startup-forrest-gb200-k8s134-prod.sh` | All-in-one prod startup 参考 (含 IMEX initramfs + auto-join, 仅供裁剪参考) |
| `node-join.sh` | 节点本地 kubeadm join (debug 用) |
| `refresh-join-token.sh` | Master 上生成新 24h kubeadm join token |
| `env.sh` | 共享 env vars (project / zone / cluster name) |
| `gx` | Mini 跳板工具 (只支持 `k8n` 别名, 通过 env 配置 SSH 到 master) |

### `scripts/post-install/` -- k8s Ready 后装 GPU Stack

| 脚本 | 用途 |
|---|---|
| `install-device-plugin.sh` | helm install NVIDIA device-plugin 0.19.2 + GPU Feature Discovery |
| `install-dranet.sh` | helm install DRANET (k8s 1.34 RDMA DRA driver) |
| `install-dra-gpu-driver.sh` | helm install NVIDIA DRA GPU Driver v0.4 (ComputeDomain CRD) |
| `apply-rdma-deviceclass.sh` | Apply RDMA DeviceClass (4 NIC bond2-5 给 ResourceClaim) |
| `verify-worker.sh` | 全 worker label / GPU count / IMEX / fabric state 验证 |

### `scripts/benchmark/` -- Benchmark 工具

| 脚本 | 用途 |
|---|---|
| `gen-nccl-multinode-sts.py` | NCCL test yaml 生成器 (StatefulSet). 参数: N node, D domain, mode (4coll/3coll/alltoall) |
| `gen-deepep-sts.py` | DeepEP v2 yaml 生成器 (sts + ComputeDomain). N>=2 用 outside-launch 模式 |
| `gen-megatron-sts.py` | Megatron-LM yaml 生成器. 5 config (llama2-7b/13b + llama3-70b), 自驱动 torchrun |
| `run-deepep.sh` | DeepEP outside launcher (kubectl exec parallel) |
| `run-nccl-test.sh` | NCCL test 手动 launcher (mpirun) |
| `extract-deepep-stats.py` | DeepEP log 解析 (输出 min/max/avg/count + latency) |
| `extract-megatron-stats.py` | Megatron iter log 解析 (elapsed_ms / TFLOPs / tokens-per-sec / loss) |
| `build-megatron-image.sh` | 节点本地 build Megatron image (buildah, NGC base ~6 min) |

### `scripts/troubleshooting/` -- Day-2 Ops

| 脚本 | 用途 |
|---|---|
| `check-k8s-dra-health.sh` | **13 项集群健康检查** (control-plane / worker / GPU / device-plugin / DRA controller / kubelet-plugin / CD ResourceSlice / DRANET / clique label / residual RC-CD / daemon DS scale / Xid / workload pod). `--fix-rs` 修 ResourceSlice, `--fix-race` 修 controller race |
| `prepare-workload.sh` | **Workload apply 一键 wrapper**. Cleanup stale daemon + apply + fix-race + wait Running + 二次 retry. 实测 16min stuck -> 4min Ready |
| `xid-fault-auto-recover.sh` | **Xid 137/145/94 自动恢复**. Detect -> verify -> serial reset -> cleanup -> verify GPU=4. 默认串行避免 IMEX cascade |
| `dra-cd-label-reconciler.sh` | DRA CD label reconciler standalone (`--once` 单跑 / `--interval=15` daemon mode). 也有 cluster-resident deployment 版 |
