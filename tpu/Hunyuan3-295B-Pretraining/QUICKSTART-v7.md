# 混元 3（295B-A21B）在 TPU v7 (Ironwood) 上预训练 — Quick Start

用 MaxText 在 TPU v7 上跑腾讯混元 3 的完整方案。从零到拿到基线，两条命令。

| | |
|---|---|
| 模型 | Tencent Hunyuan 3，295B 总参 / 21B 激活，80 层 MoE |
| 平台 | TPU v7 Ironwood，64 芯片（`4x4x4`，16 台） |
| 框架 | MaxText（nnx），代码在 [`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3) |
| 精度 | BF16 计算 / FP32 主权重 |
| **实测水位** | **step 20.43 s · 445.1 TFLOP/s/chip · MFU 19.29% · 205,314 tok/s** |
| **复现状态** | ✅ 换一套集群重跑过，step 19.90 s / MFU 19.82%，差 −2.6%（[§4.2.1](#421-二次复现另一套集群2026-07-30)） |
| **调优状态** | 🔄 仍在调，目标 26.6%，差 1.38× —— **调优过程与实验清单见 [TUNING-v7.md](TUNING-v7.md)** |

> **这份文档只管一件事：让你把它跑起来，拿到基线。**
> 环境怎么配、代码怎么传、任务怎么提、日志怎么读、基线是多少 —— 到此为止。
>
> **为什么只有 19.29%、已经试过什么、还能怎么调**，全部在
> **[TUNING-v7.md — 性能调优实践](TUNING-v7.md)**。两份文档在基线这个数字上接头。
>
> 与 [v5p 版](QUICKSTART-v5p.md)的差别：v5p 已调优收敛，照着跑就能复现；
> v7 是**能跑，但水位离目标还差 1.38×**，拿它当起点而不是终点。

---

## 1. 模型与代码

### 1.1 Hy3 是什么

一句话：**attention 是 Qwen3 的，MoE 是 DeepSeek V3 的**，
MaxText 里这两半都已有现成实现，本项目只写了装配逻辑，零新数学。

| | |
|---|---|
| 结构 | 80 层，第 0 层 dense、1–79 层 MoE |
| Attention | GQA 64q / 8kv，head_dim 128，QK-LayerNorm，无 bias |
| MoE | 192 routed experts top-8 + 1 shared，sigmoid 路由 + 专家偏置 |
| 其他 | MTP 1 层，vocab 120832，routed scaling 2.826 |
| 参数分布 | **97% 在路由专家里**，attention 只占 2% |

参数分布直接决定并行策略：**TP 无用**（切 attention 纯亏通信）、
**不用 EP**（TPU 上专家并行是负优化，实测 EP=64 直接超显存）、
**纯 FSDP 吃满全部 device**。

> 架构的完整拆解 —— 为什么两个现成 decoder block 都不行、与 DSV3 的关键差异
> （Hy3 没有 device-limited routing）、参数量分解 —— 见
> [v5p Quick Start §1](QUICKSTART-v5p.md)。**两个平台完全一样，这里不重复。**

### 1.2 代码从哪来

**唯一真相是 fork 的分支**，本仓不留代码副本：

```
https://github.com/yangwhale/maxtext   分支 hunyuan3
```

基于上游 main，三个 commit：

| commit | 内容 |
|---|---|
| `Resolve the loss-free-balancing bias path per decoder block` | 上游 bug 修复（与 Hy3 无关，任何非 DeepSeek 模型开 aux-loss-free 均衡都会撞） |
| `Add Tencent Hunyuan 3 (295B-A21B)` | 模型本体 + 注册 |
| `Let Hunyuan3 use the SwiGLU activation bound too` | 激活截断白名单 |

**新增 3 个文件**：`models/hunyuan3.py`（161 行有效代码）+ 两个 yml 配置。
**改动上游 12 个文件**，全部是「让框架认识这个模型」，没有一处是算法实现。

**v5p 和 v7 用同一份代码、同一个镜像**，差别只在启动参数。

---

## 2. 环境准备

> **如果你用的是别人已经建好的托管集群（排队制 / Kueue），§2.1 和 §2.2 直接跳过** —— 那种集群不让你自己建 workload policy 和节点池，提任务它自动给机器。但要先读 [§3.6](#36-在托管kueue--排队制集群上扩不出节点时怎么定位)，有个必踩的死锁。

五件事，缺一件都跑不起来。

### 2.1 workload policy（v7 特有，必须先建）

v5p 建节点池一条命令就完了。tpu7x 会直接被拒：

```
Creation of a managed instance group with tpu7x-standard-4t machine type
with placement policy is not supported. Use workload policy instead.
```

**必须先建一个 `workloadPolicy` 类型的 resource policy，节点池再引用它。**
gcloud 577 没有这个子命令（`resource-policies create` 只有 group-placement 等），
只能走 REST：

```bash
P=YOUR-PROJECT
TOK=$(gcloud auth application-default print-access-token)

# 64 芯片主力池用
curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
 "https://compute.googleapis.com/compute/v1/projects/$P/regions/us-central1/resourcePolicies" \
 -d '{"name":"wp-4x4x4","workloadPolicy":{"type":"HIGH_THROUGHPUT","acceleratorTopology":"4x4x4"}}'

# 4 芯片冒烟池用
curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
 "https://compute.googleapis.com/compute/v1/projects/$P/regions/us-central1/resourcePolicies" \
 -d '{"name":"wp-2x2x1","workloadPolicy":{"type":"HIGH_THROUGHPUT","acceleratorTopology":"2x2x1"}}'
```

- `acceleratorTopology` **必须带**。只给 `type` 会报
  `does not support TPU topology with group placement policy and workload policy at the same time`
- **一个 policy 对应一个拓扑**，用几种拓扑就建几个
- 建好之后可以用 `gcloud compute resource-policies list --regions=us-central1` 确认

### 2.2 TPU 节点池

```bash
# 64 芯片（16 台 × 4）—— 主力
gcloud container node-pools create np-v7x-64 \
  --cluster=CLUSTER --project=$P --region=us-central1 \
  --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=4x4x4 \
  --placement-policy=wp-4x4x4 --num-nodes=16 --spot \
  --disk-type=hyperdisk-balanced --disk-size=200 --scopes=cloud-platform

# 4 芯片小池 —— 冒烟用，改一行代码几十秒验一轮
gcloud container node-pools create np-v7x-dev \
  --cluster=CLUSTER --project=$P --region=us-central1 \
  --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=2x2x1 \
  --placement-policy=wp-2x2x1 --num-nodes=1 --spot \
  --disk-type=hyperdisk-balanced --disk-size=200 --scopes=cloud-platform
```

四个跟 v5p 不一样的地方：

| | 说明 |
|---|---|
| `--placement-policy` | 指向 §2.1 建的那个，**拓扑要对得上** |
| `--disk-type=hyperdisk-balanced` | **v7 不接受普通 pd** |
| zone | v7 在 **`us-central1-c`**，v5p 在 `-a`，别搞混 |
| `--num-nodes` | = 芯片数 ÷ 4，且必须与 `--tpu-topology` 相乘一致 |

### 2.3 JobSet CRD

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/download/v0.11.1/manifests.yaml
kubectl wait --for=condition=Available deploy/jobset-controller-manager \
  -n jobset-system --timeout=180s
```

v0.11.1 自带证书，**不需要 cert-manager**。新集群默认没有这个 CRD，
不装的话提交训练时 `kubectl apply` 会找不到 `jobset.x-k8s.io/v1alpha2`。

### 2.4 暂存桶 + 跨项目授权

```bash
gcloud storage buckets create gs://YOUR-STAGE-BUCKET --location=US

NODE_SA=<集群项目号>-compute@developer.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://YOUR-STAGE-BUCKET \
  --member="serviceAccount:$NODE_SA" --role=roles/storage.objectViewer

# 镜像在别的项目时，节点 SA 还要能拉
gcloud artifacts repositories add-iam-policy-binding gcr.io --location=us \
  --project=IMAGE_PROJECT --member="serviceAccount:$NODE_SA" \
  --role=roles/artifactregistry.reader
```

### 2.5 网络

共享项目的 default VPC 是 auto 模式，`10.128.0.0/9` 被各 region 子网占满，
GKE 凑不出一整块 `/14` 给 pod：

```
The network "default" does not have available private IP space in
10.0.0.0/9 to reserve a /14 block for pods
```

**建自己的 custom VPC 最省事**，顺带能把 MTU 开到 8896：

```bash
gcloud compute networks create NAME-vpc --subnet-mode=custom --mtu=8896
gcloud compute networks subnets create NAME-uc1 --network=NAME-vpc \
  --region=us-central1 --range=10.124.0.0/22 \
  --secondary-range=pods=10.125.0.0/16,services=10.124.16.0/20 \
  --enable-private-ip-google-access
```

三段全压在 `10.124.0.0/15` 内，将来做 VPC peering 只需让对方避开这一段。

### 2.6 建池之前先看一眼容量

**配额决定你能不能申请，容量决定你能不能拿到，这是两个独立的闸门。**
tpu7x 的抢占式容量是 zone 级共享的 —— 池子停在 `PROVISIONING` 且 MIG 没有
error，那就是纯粹排不到机器，**换项目、提配额都无效**。

```bash
gcloud compute instances list --project=ANY-PROJECT-IN-SAME-ZONE \
  --filter="machineType~tpu7x AND status=RUNNING" \
  --format='value(zone,scheduling.provisioningModel)' | sort | uniq -c
```

> `4x4x4`（64 芯片）是**原子切片**，TPU 拓扑没有 48 / 56 这种中间档 ——
> **凑不满等于拿不到**。需要长时间稳定占用时改用 DWS flex-start
> （排队制，拿到之后不会被抢），建池方式见 [实验档案 §9.0](EXPERIMENT-LOG.md)。

---

## 3. 跑起来

```bash
cd maxtext-hunyuan3/
gcloud container clusters get-credentials CLUSTER --region=REGION --project=PROJECT

export GCS_STAGE=gs://YOUR-STAGE-BUCKET/hy3
export IMAGE=us-docker.pkg.dev/YOUR-PROJECT/gcr.io/YOUR-maxtext-latest:runner

# ① 准备代码（只有改了代码才要重跑；换参数不用）
bash prep.sh

# ② 起训练
PLATFORM=v7 bash run.sh myrun

# ③ 看结果
kubectl logs -f job/hy3-myrun-slice-job-0 -c jax-tpu
```

### 3.1 两个脚本各做什么

| 脚本 | 动作 |
|---|---|
| `prep.sh` | clone `hunyuan3` 分支 → **8 项自检** → `tar` 整棵 `src/maxtext` → 传 GCS |
| `run.sh` | 提交 JobSet；pod 里 `rm -rf /deps/src/maxtext` 后**整棵解包覆盖** |

**注意是整棵覆盖，不是只注入改动文件。** 只注入的话，测的是
「我的改动 + 容器里的旧基座」，不是分支本身。

`prep.sh` 与 v5p 完全共用 —— 它只管代码，不管平台。
8 项自检挡的是「分支自己少东西」：三个新增文件在不在、白名单两个模型名全不全、
枚举有没有 `HUNYUAN3`、`train.py` 补丁在不在、
`Hunyuan3MoeBlock_0` 这个属性名在模型文件和训练循环两边对不对得上。

### 3.2 先跑冒烟

```bash
NODES=1 TOPO=2x2x1 PLATFORM=v7 MODEL=hunyuan3-smoke STEPS=8 \
  bash run.sh smoke per_device_batch_size=1 max_target_length=2048
```

4 层缩层，结构与 295B 完全一致（192 专家、top-8、sigmoid、专家偏置、
共享专家、GQA、QK-norm、fp32 路由、MTP 全是满配），只砍层数。

**通过标准**（2026-07-30 在共享集群 4 芯片实测）：

| | v7 实测 | 参考：v5p 同命令 |
|---|---|---|
| 参数量 | **16.139 B** | 16.139 B（**必须一致**） |
| `total_weights` | **16384** | 8192 |
| loss（8 步） | **13.411 → 11.091** | 13.453 → 10.354 |
| 稳态 step | ~0.62 s | ~0.70 s |
| NaN / skipped | 0 | 0 |

> ⚠️ **两个平台的 loss 序列不一样，这是对的，不是 bug。**
> 同样 4 芯片，**v7 有 8 个 device 而 v5p 只有 4 个**（2:1 vs 1:1）。
> `per_device_batch_size=1` 之下，v7 的 global batch 是 **16384**、v5p 是 **8192** ——
> **喂进去的数据量差一倍，loss 轨迹当然不同。**
>
> 这是 §3.3 那条单位换算的延伸：**`per_device_batch_size` 里的 "device" 在 v7 上是半个芯片。**
> 想让两边严格可比，v7 这边把 `per_device_batch_size` 减半即可。
>
> **真正跨平台恒定、可以当硬标准的只有参数量 16.139 B** —— 对不上就是代码没打全。

**为什么 4 层能代表 80 层**：MaxText 按**类型**分组做 `scan`，
79 个 MoE 层共用同一份编译产物，层与层的差别只在权重数值上，不在代码路径上。
所以冒烟测的不是「抽样几层」，而是**那个被复用 79 次的唯一函数**。

冒烟覆盖不到的：显存压力、大规模切分、80 层累积的数值误差、
完整 XLA flag 集、收敛质量、以及全部性能。它证明的是「代码路径都对」。

### 3.3 单位换算 —— v7 最容易出错的一步

**v7 是 2 device / chip**（v5p 是 1 : 1），而框架日志一律按 **device** 报：

```
per-chip TFLOP/s = 日志里的 TFLOP/s/device × 2
MFU              = per-chip ÷ 2307
```

例：日志报 `TFLOP/s/device: 222.56` → per-chip **445.1** → MFU **19.29%**。

> **方向搞反就是 4 倍的误差**（该 ×2 却 ÷2）。跨代际比 MFU 之前
> 必须先确认 device : chip 比例。

### 3.4 读日志的四条规矩

1. **先确认 `16/16 Running` 再看日志。** TPU 切片全有全无，人不齐时活着的
   pod 会报 `GetSliceInfo can only be invoked after a slice is built` ——
   那是症状不是病因。
2. **判错看最早那条，不是日志尾。** 配置非法会先把 TPU 拉起来再退，
   真正的报错（`MAXTEXT CONFIG ERROR` / pydantic 的 `Value error`）在日志上方。
3. **step 0 含编译，step 1/2 是 JAX 异步派发的假读数**，稳态取 step ≥ 3。
4. **从提交到看见 step 0 要等 10–17 分钟**（v5p 约 9 分钟）。第一次跑请耐心等。
   **这不是编译时间** —— 实测拆解（64 芯片 / 80 层）：XLA 编译只占 **44 秒**，
   多机建切片 **60 秒**，其余全在日志开始之前：Kueue admit、调度 16 个节点、
   拉镜像、下代码。真正随规模涨的是调度和建切片，**编译几乎不随芯片数和层数变**
   （`scan_layers=True` 让层数不进 HLO 规模）。详见 TUNING-v7 §5.3.1。

### 3.5 日志必须落盘，别只挂 `kubectl logs -f`

pod 一旦被删、被抢占、或被集群的时间上限杀掉，**`kubectl logs` 就再也读不到了**。
一次 64 芯片的跑就这样把日志全丢了，只能重跑。提交后立刻起一个后台 tail：

```bash
mkdir -p ~/tpu-logs
setsid bash -c "kubectl logs -f job/hy3-<run>-slice-job-0 -c jax-tpu \
  > ~/tpu-logs/hy3-<run>-$(date +%m%d-%H%M).log 2>&1" < /dev/null &
```

`-f` 会在 pod 还没 Running 时直接退出，所以要么等 `16/16 Running` 再起，
要么套一层重试循环。

### 3.6 在托管（Kueue / 排队制）集群上：扩不出节点时怎么定位

**先说结论：`kubectl` 层面看不到扩容失败的真正原因。** pod 事件只会给你一句
`Pod didn't trigger scale-up: ... in backoff after failed scale-up`，
它只说"试过、失败了、在退避"，不说为什么失败。真原因在托管实例组里：

```bash
# 先找到 TPU 的 MIG（名字含 nap-tpu7x）
gcloud container clusters describe <CLUSTER> --region <REGION> \
  --format='value(nodePools[].instanceGroupUrls)' | tr ';' '\n' | grep tpu

gcloud compute instance-groups managed list-errors <MIG> --zone <ZONE> \
  --format='value(error.code,error.message)'
```

一次真实的输出：

```
QUOTA_EXCEEDED  Instance 'gke-tpu-xxxx' creation failed:
                Quota 'HDB_TOTAL_GB' exceeded. Limit: 40960.0 in region us-central1.
```

**区域磁盘配额打满，节点就建不出来。** TPU 节点的启动盘是
`hyperdisk-balanced`（100 GB / 台），跟别人工作负载的 PVC 抢同一个区域配额。
16 台就是 1600 GB。别人留下的 `Released` PV（PVC 删了盘还在）会一直占着配额，
把整个集群的扩容都卡死 —— 这跟你的 JobSet 怎么写没有任何关系。

对应的排查顺序（**直接看 pod 会误判**）：

| 步骤 | 看什么 | 卡在这说明 |
|---|---|---|
| ① | `kubectl get workload`（Kueue） | 配额记账没过 |
| ② | MIG `list-errors` | **配额/容量，扩不出机器** |
| ③ | `kubectl get nodes` | 机器建出来了但没注册 |
| ④ | `kubectl get pods` | 常规问题：镜像、配置 |

> **被 admit ≠ 有机器。** 队列放行只代表记账通过。

**另一个独立的坑：`exclusive-topology` 会让 autoscaler 少看见需求。**

`run.sh` 默认带 `alpha.jobset.sigs.k8s.io/exclusive-topology` 注解，保证一个
JobSet 的 pod 全落同一节点池。代价是要 leader pod 先落地，follower 才能抄它的
`gke-nodepool` 选择器。提交时如果没有空节点，leader 落不了地，follower 就建不出来：

```
16 个 pod 只出 1 个 → autoscaler 只看见 1 个 pending
```

现象是 `parallelism=16` 但 `status.active=1`，**没有任何报错**。

> **这一条和上面的配额是两回事，别混。** 我们那次 64 芯片跑不起来，
> 实测原因是配额（MIG 报 `QUOTA_EXCEEDED`），全程节点数没变过、**扩容压根没发生**；
> 最后跑通是因为别人的任务释放了现成的 16 台。所以
> **没有证据表明 `exclusive-topology` 是那次的瓶颈** —— 它的影响是在
> "扩容本身可用"的前提下才会显现。这里记下来，是因为
> `active=1` 这个现象本身极难自查。

**两种绕法**（`run.sh` 都支持）：

| 场景 | 做法 |
|---|---|
| 已知目标池 | `NODEPOOL=<池名> bash run.sh ...` —— 池子写死在 `nodeSelector`，不需要 leader 先行 |
| 要让集群自己扩容 | `NO_EXCLUSIVE_TOPOLOGY=1 bash run.sh ...` —— 去掉注解，16 个 pod 一次全建出来 |

去掉注解后理论上 pod 可能分散到不同池，但 TPU 的 `gke-tpu-topology` 选择器
本身就把范围限死在同拓扑的池里，实测 16 个 pod 整齐落在同一个 `4x4x4` 池。

---

## 4. v7 基线

### 4.1 硬件

| | 值 |
|---|---|
| 节点池 | `np-v7x-64`，16 台 `tpu7x-standard-4t`，拓扑 `4x4x4` |
| 芯片数 | **64** |
| JAX device 数 | **128**（2 device / chip） |
| HBM | 192 GB / chip；单 device 可用 **94.74 GB** |
| BF16 峰值 / chip | **2,307 TFLOPS** |
| FP8 峰值 / chip | 4,614 TFLOPS |
| 总算力 | 147.6 PFLOPS BF16 |

### 4.2 实测指标

**两个开跑就能查的健康检查**（不对说明代码没打全）：

| 日志字段 | 应为 |
|---|---|
| `number parameters` | **298.786 billion**（与 v5p 逐位一致） |
| `Total TFLOPs` | **约 4547**（= `TFLOP/s/device × step`）—— 如果是这个数的约 5 倍，说明 `maxtext_utils.py` 的 FLOP 公式没加 `HUNYUAN3`，MFU 会虚高 |

> `Total TFLOPs` 是**每 device 每步**的量，随 `pdbs × seq` 变。
> v7 用 seq 4096 是 ~4547，v5p 用 seq 8192 是 10169 —— **别把两个平台的值搞混**。

| 指标 | 值 |
|---|---|
| 稳态 step | **20.43 s** |
| 日志 TFLOP/s/device | 222.56 |
| **TFLOP/s / chip** | **445.1** |
| **MFU** | **19.29%** |
| 整机吞吐 | **205,314 tok/s** |
| 每步 token | 4,194,304（128 × 8 × 4096） |
| 提交 → 看到 step 0（**不是编译**） | 10–17 分钟 |

### 4.2.1 二次复现（另一套集群，2026-07-30）

同一份代码、同一套参数，换到**另一个 Kueue 托管的共享集群**上，
在别人已经建好的 `4x4x4` 池（16 台 × 4 芯片）里跑 30 步：

| 指标 | 本次 | 上表基线 | 差 |
|---|---|---|---|
| 稳态 step（step ≥ 3，20 步平均） | **19.90 s** | 20.43 s | **−2.6%** |
| 日志 TFLOP/s/device | 228.66 | 222.56 | +2.7% |
| TFLOP/s / chip | **457.3** | 445.1 | +2.7% |
| **MFU** | **19.82%** | 19.29% | +0.53 pp |
| `number parameters` | 298.786 B | 298.786 B | 一致 |
| `Total TFLOPs` | 4550.33 | ~4547 | +0.07% |
| 提交 → 看到 step 0 | **约 8 分钟** | 10–17 分钟 | 更快 |

逐步抖动几乎为零 —— 20 步里每步都是 `19.901 ± 0.003 s`。

> **±3% 以内算复现成功。** 这次略快，合理的解释是换了一批机器；
> 参数量和 `Total TFLOPs` 两个一次性读数完全对上，说明跑的确实是同一个模型、
> 同一套 FLOP 公式，性能差异不是"算少了"。

**这次复现暴露的两个文档缺口**（已补进 §3.5、§3.6）：
扩容失败的真原因只在 MIG 的 `list-errors` 里（本次是区域磁盘配额打满，
节点建不出来）；以及日志必须落盘。

### 4.3 完整参数集

以下就是 `run.sh` 的 v7 分支在用的全部参数。

**并行与模型**

```
model_name=hunyuan3-295b
override_model_config=True
ici_fsdp_parallelism=-1          # 吃满 128 路 FSDP
ici_tensor_parallelism=1         # TP 无用，attention 只占 2% 参数
```

**MoE**

```
megablox=True                    # 分组矩阵乘
sparse_matmul=True
use_custom_sort_vjp=True         # 「按专家排序」那步的自定义反向传播
```

**batch 与序列**

```
per_device_batch_size=8          # 再上会 OOM
max_target_length=4096           # v5p 用 8192；v7 上短序列 + 大 batch 更优
```

**Attention**

```
attention=flash
use_tokamax_splash=True          # v7 特有，与下一项合计 +2.6%
sa_use_fused_bwd_kernel=True     # 注意：v5p 上这一项要设 False
sa_block_q=2048  sa_block_kv=2048  sa_block_kv_compute=2048
sa_block_q_dkv=2048  sa_block_kv_dkv=2048  sa_block_kv_dkv_compute=2048
sa_block_q_dq=2048  sa_block_kv_dq=2048
```

**重计算与 offload**

```
scan_layers=True
remat_policy=custom
decoder_layer_input=offload
out_proj=remat
```

**优化器（v7 独有）**

```
opt_type=adamw
mu_dtype=bfloat16                # Adam 一阶动量降 bf16
grad_dtype=bfloat16              # 梯度降 bf16
use_iota_embed=True              # 省 embedding 显存
```

优化器状态从 16 B/param 降到 12 B/param。注意 `nu_dtype`（二阶动量）
optax 不支持单独设，恒随 `weight_dtype`；**主权重仍是 fp32**。

**精度与其他**

```
dtype=bfloat16
weight_dtype=float32
allow_split_physical_axes=True
tokenizer_type=tiktoken
tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken
```

**基线是在这个条件下测的**（`run.sh` 写死，跑真实训练时要改）

```
dataset_type=synthetic           # 合成数据，只测吞吐不测收敛
enable_checkpointing=False       # 不写 checkpoint，避免 I/O 干扰读数
steps=10                         # 取 step >= 3 的稳态
base_output_directory=/tmp/hy3out
```

> **v5p 上设的 18 个 MoE tile 参数，v7 上一个都没设** —— 不是判断为无用，
> 是**没扫过**。见 [调优实践 §4.4](TUNING-v7.md#44-实验清单)。

### 4.4 XLA flag（15 个）

通过 `LIBTPU_INIT_ARGS` 传入。

```
# 基础（2 个）
--xla_tpu_scoped_vmem_limit_kib=65472
--xla_enable_async_all_gather=true

# SparseCore 卸载组（9 个）—— v7 上收益是 0，保留只为与官方配方对齐
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
--xla_tpu_use_tc_device_shape_on_sc=True
--xla_sc_disable_megacore_partitioning=True

# 调度器组（4 个）—— 唯一值钱的一组，+6.6%
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
```

补到 26 个也能跑，收益 ±0，所以保持精简。
**但不要把官方那套 30 个一次全开** —— 那一轮死锁了，元凶是同时打开的
`use_tokamax_gmm`（见 [调优实践 §2.3](TUNING-v7.md#23-三条死路都实测过别再踩)），不是 flag 数本身。

> ⚠️ **libtpu 对不认识的 flag 是硬失败**（`Unknown command line flag`，进程直接退）。
> **换镜像必须重过一遍 flag 集。**

---

## 5. 已知限制

| 项 | 状态 |
|---|---|
| **调优未完成** | 19.29% vs 目标 26.6%，差 1.38× |
| 数据集 | `dataset_type=synthetic`。**loss 下降只证明「能算且不发散」，不是收敛证据** |
| `use_tokamax_gmm` | 死锁，根因未查清 |
| HF 权重 → MaxText Orbax 转换 | 未做。只跑吞吐基线可以不碰；要 SFT 必须做 |
| 完整 loss 曲线 | 未记。建议补一条 30 步以上的 |
| 容量 | tpu7x spot 经常整个 zone 拿不到，见 §2.6 |

---

## 附录 A：与 v5p 的差异速查

两个平台**共用同一份代码、同一个镜像**，以下是全部差异：

| | v7 | v5p |
|---|---|---|
| 节点池 | 16 台 `tpu7x-standard-4t`，`4x4x4`，us-central1-**c** | 64 台 `ct5p-hightpu-4t`，`4x8x8`，us-central1-**a** |
| 建池前置 | **必须先建 workload policy（REST）** | 无 |
| 磁盘 | **hyperdisk-balanced** | 默认 |
| device : chip | **2 : 1** | 1 : 1 |
| MFU 分母 | **2,307** | 459 |
| `max_target_length` | **4096** | 8192 |
| `sa_use_fused_bwd_kernel` | **True** | **False** |
| `use_tokamax_splash` | **True** | 不设 |
| `opt_type` / `mu_dtype` / `grad_dtype` | **adamw / bf16 / bf16** | 默认 / fp32 / fp32 |
| `use_iota_embed` | **True** | 不设 |
| MoE tile 参数（18 个） | **不设** | 全设 |
| XLA flag | **15 个** | 25 个 |
| SparseCore 卸载组收益 | **±0** | **+4.07 pp** |
| 提交 → 看到 step 0 | 10–17 分钟 | 约 9 分钟 |
| 建池耗时 | 视容量，可能排不到 | 5–9 分钟 |
| **MFU** | **19.29%（仍在调）** | **35.07%（已收敛）** |

---

## 附录 B：延伸阅读

| 文档 | 内容 |
|---|---|
| [QUICKSTART-v5p.md](QUICKSTART-v5p.md) | v5p 版，含架构完整拆解；基线 35.07%，已从零验证 |
| [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) | 完整实验档案：全部轮次、12 个 bug 的复盘、DWS flex-start 建池 |
| [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) | 把别的模型移植到 MaxText 的通用范式 |
| [maxtext-hunyuan3/](maxtext-hunyuan3/) | `prep.sh` / `run.sh` |
