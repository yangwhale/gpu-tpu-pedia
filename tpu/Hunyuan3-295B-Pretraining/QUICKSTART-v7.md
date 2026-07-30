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
| **调优状态** | 🔄 **仍在调**。目标是 DeepSeek V3 在同硬件的实测水位 26.6%，**还差 1.38×** |

> **这份文档的定位与 [v5p 版](QUICKSTART-v5p.md)不同。** v5p 已经调优收敛，
> 照着跑就能复现；v7 是**能跑、水位记录在此、已知的死路也记在此**，
> 拿它当**起点**而不是终点。

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
4. **v7 编译要 10–17 分钟**，比 v5p 的 9 分钟慢不少。第一次跑请耐心等。

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
| 编译时间 | 10–17 分钟 |

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
| 编译时间 | **约 8 分钟** | 10–17 分钟 | 更快 |

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
> 是**没扫过**。见 §7.4。

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
`use_tokamax_gmm`（见 §5.3），不是 flag 数本身。

> ⚠️ **libtpu 对不认识的 flag 是硬失败**（`Unknown command line flag`，进程直接退）。
> **换镜像必须重过一遍 flag 集。**

---

## 5. 每个选择值多少

从只带 2 个 XLA flag 的首跑出发，逐项叠加。

| 轮次 | 增量 | seq | pdbs | step | TFLOP/s/chip | MFU |
|---|---|---|---|---|---|---|
| V1 | 基线：2 个 XLA flag | 8192 | 4 | 25.11 s | 404.75 | 17.54% |
| y1 | + `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | 8192 | 4 | 24.45 s | 415.16 | 18.00% |
| y2 | + adamw / bf16 优化器 + `iota_embed` | 8192 | 4 | 24.61 s | 412.88 | 17.90% |
| y3 | + SparseCore 卸载组（9 个 flag） | 8192 | 4 | 24.61 s | 412.56 | 17.88% |
| y4 | + 调度器组（4 个 flag） | 8192 | 4 | 23.08 s | 440.30 | 19.09% |
| z1 | y1 + 换 batch / 序列口径 | 4096 | 8 | 21.69 s | 418.89 | 18.16% |
| **c1** | **调度器组 × pdbs=8 / seq=4096（当前最优）** | 4096 | 8 | **20.43 s** | **445.12** | **19.29%** |
| c2 | c1 + 杂项组（补齐 26 flag） | 4096 | 8 | 20.45 s | 444.63 | 19.27% |

**相对首跑 +10.0%。** 按贡献排序：

| 手段 | 贡献 |
|---|---|
| **pdbs=8 / seq=4096** | **+12.8% 吞吐**（TFLOP/s 只 +0.9%） |
| **调度器 flag 组（4 个）** | **+6.6%** |
| `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | +2.6% |
| 杂项 flag 组（5 个） | ±0 |
| **SparseCore 卸载组（9 个）** | **±0** |
| 优化器 / 显存组 | −0.5%（省的是显存不是时间） |

### 5.1 最重要的一条否定结果

**SparseCore 集合通信卸载那 9 个 flag，在 v5p 上值 4.07 pp（13%），在 v7 上收益是 0。**

这条否定结果直接指出了瓶颈在哪：

> SparseCore 卸载改的是**通信在哪执行**，调度器改的是**通信和计算怎么重叠**。
> 前者无效说明**通信不是瓶颈**；后者有效（+6.6%）说明**通信没跟计算叠起来** ——
> 不是通信太慢，是它没藏住。

> ⚠️ **但不要把「这一组没用」外推到「同类的下一组也没用」。**
> 我当时正是这么推的：既然通信不是瓶颈，剩下的 flag 组期望收益也低，跳过。
> 结果调度器那组在我放弃之前已经跑完了 —— **+6.6%，当时的最优**。
> 消融的纪律是**每一组都要真跑**。

### 5.2 同一个开关在两个平台上可以反号

`sa_use_fused_bwd_kernel` 在 **v5p 上要设 `False`、v7 上要设 `True`**。
这不是笔误，是两边各自实测的结果。同理，v5p 上值 4 pp 的 SparseCore 卸载组，
在 v7 上是 0。

> **别把一个平台的调优结论直接搬到另一个。**

### 5.3 三条死路（都实测过，别再踩）

| 开关 | 后果 |
|---|---|
| **`use_tokamax_gmm=True`** | **死锁**，`stalled chips [7]`，连 step 0 都跑不完。2 次开 2 次挂、2 次不开 2 次通 |
| `shard_exp_on_fsdp=True`（FSDP=64 × DP=2） | **OOM**，用了 109.14 G / 94.74 G，**比不开还多 14 G** |
| `per_device_batch_size=12` | **OOM**，临时缓冲 95.17 G |

**`use_tokamax_gmm` 值得单独说。** 官方 Ironwood DSV3 配方里这个开关是 `True`
且能跑，说明 tokamax 后端本身没问题，**是它跟 Hy3 的形状不合** ——
最可能是 192 个专家：DSV3 是 256，分组矩阵乘的组数正好是 2 的幂。这条待确认。
它也是 `use_gmm_v2` 的强制前置，所以 gmm_v2 一并不可用。

**`shard_exp_on_fsdp` 为什么净亏**：这笔交易两头都动 ——
专家权重改按专家维切（收益），但 FSDP 宽度从 128 降到 64，
**非专家部分（attention 80 层 + embedding + dense 首层，约 7.2 B）每卡分片直接翻倍**。
省的抵不过多花的。根因还是 **192 不是 2 的幂**。

---

## 6. 目标定在哪：为什么是 600–630 而不是 900

Ironwood 官方实测表（全部 bf16、synthetic、per-chip 口径）：

| 模型 | 类型 | chips | 序列 | TFLOP/s/chip | MFU |
|---|---|---|---|---|---|
| llama3.1-405b | **稠密** | 256 | 8192 | 1,261.4 | 54.7% |
| llama3.1-70b | **稠密** | 64 | 8192 | 1,207.1 | 52.3% |
| gemma4-31b | **稠密** | 64 | 8192 | 931.3 | 40.4% |
| **qwen3-235b-a22b** | **稀疏 MoE** | 256 | 4096 | **629.8** | **27.3%** |
| **deepseek-v3 671B** | **稀疏 MoE** | 256 | 4096 | **612.7** | **26.6%** |
| gpt-oss-120b | 稀疏 MoE | 256 | 8192 | 329.9 | 14.3% |
| **hunyuan3 295B（本项目）** | **稀疏 MoE** | **64** | **4096** | **445.1** | **19.29%** |

**900 以上全是稠密模型。** 稀疏 MoE 在 Ironwood 上的实际水位是
**600–630 TFLOP/s/chip（26–27% MFU）**，两个最接近 Hy3 的参照都在这条线上。

差距是结构性的，不是调参能翻越的：

- 稠密模型每个 token 走同一套权重，GEMM 又大又规整，MXU 能吃满
- MoE 每层要做一次路由、一次按专家分组重排、一次分组矩阵乘、一次还原。
  分组矩阵乘的每个子块只有 `tokens_per_expert × emb × moe_mlp` 那么大，
  而且组大小随路由结果浮动，**编译期拿不到静态形状**
- 还要加上 all-gather / reduce-scatter 把 192 份专家权重摊开又收回

**所以 v7 的目标是 600–630 TFLOP/s/chip，对应 step 压到 14–15 s。
当前 445.1，缺口 1.38×。**

> Hy3 的激活参数（21 B）比 DSV3（37 B）还少，结构也更简单
> （GQA 而非 MLA、192 专家而非 256），**没有理由跑不到同一水位** ——
> 差距来自配置，不是架构。

### 6.1 关于 FP8：先别指望

同一张表里 DSV3 的 FP8 数据：

| | BF16 | FP8 | |
|---|---|---|---|
| DSV3 TFLOP/s/chip | 612.66 | 743.46 | **+21.4%** |
| 对本精度峰值的 MFU | 26.6% | **16.1%** | 峰值翻倍但吞吐没翻 |
| 稠密 llama3.1-405b 同口径 | 54.7% | 41.8% | 稠密 FP8 涨 **+52.8%** |

**MoE 兑现不了 FP8 的两倍峰值，稠密可以。** 原因跟上面那条一样：
MoE 的时间大量花在路由、分组重排、all-to-all 和小块 GEMM 上，
这些环节不吃 MXU 峰值，降精度对它们几乎没帮助。

> ⚠️ **报 FP8 的 MFU 一定要说明分母。** 同一个 743.46，
> 对 FP8 峰值算是 16.1%，对 BF16 峰值算是 32.2% —— **差一倍**。

---

## 7. 性能调优

> 这一章是**工作台**，不是结论。假说、观测手段、实验清单、结果表都在这里，
> 每跑一轮就回填一行。**先读 §7.1 —— 它可能直接改变你要不要继续调。**

### 7.1 先做 roofline 判定：19.8% 也许已经接近 BF16 的天花板

把两代硬件的三个比值摆在一起：

| | v5p | v7 | v7 / v5p |
|---|---|---|---|
| BF16 峰值 / chip | 459 TFLOPS | 2,307 TFLOPS | **5.03×** |
| HBM 带宽 | 2.8 TB/s | 7.4 TB/s | **2.64×** |
| ICI / chip | 600 Gbps | 1,200 Gbps | **2.0×** |

**算力涨了 5 倍，喂数据的两条通道只涨了 2～2.6 倍。**

Roofline 的拐点（算力 ÷ 带宽）因此从 v5p 的 **164 FLOP/byte** 抬到 v7 的
**312 FLOP/byte** —— 想在 v7 上保持 compute-bound，算术强度要翻 1.9 倍。

MoE 恰恰是算术强度最低的一类结构：每个 token 只激活 21B / 295B ≈ 7% 的参数，
但**权重该从 HBM 读还得读**。所以有理由怀疑它在 v7 上直接掉进了 memory-bound 区。

**做个纯 memory-bound 的预测**：如果计算完全被带宽卡住，MFU 应该按
「带宽比 ÷ 算力比」缩放：

```
34.98%  ×  (2.64 / 5.03)  =  18.4%
```

**实测 19.82%。差 8%。**

**但这个数字不能单独当证据。** 分母里换成 ICI 比值（2.0×）会得到 13.9%，
实测 19.82% 落在两者之间。「非计算部分没跟着涨 5 倍」这一点，
下面两个假说都能解释，光靠比值分不开：

| | 假说 H1：HBM 带宽卡住 | 假说 H2：时间花在非 MXU 环节 |
|---|---|---|
| 说法 | MoE 算术强度低，权重读取吃满 7.4 TB/s | 路由、分组重排、all-to-all、小块 GEMM 不吃 MXU |
| 预测 MFU | ~18.4% | 也低，取决于这些环节占比 |
| **trace 判据** | **HBM 带宽接近 7.4 TB/s** | **HBM 带宽不高，但 collective / permute op 占满时间轴** |
| 该走 | A 组：提高算术强度 | B 组：藏通信、修 kernel |

**§6.1 的实测数据倾向 H2。** 同硬件上 DSV3 开 FP8 只涨 **+21.4%**
（612.66 → 743.46 TFLOP/s/chip）。如果瓶颈纯粹是权重读取带宽，
把权重字节砍半应该给出接近翻倍的收益，而不是两成。
稠密的 llama3.1-405b 同口径涨了 **+52.8%** —— 说明硬件本身兑现得了，
是 **MoE 结构**把收益吃掉了。

> **所以先别急着下结论，也别急着扫参数。** §7.2 那一轮 trace 就能分开这两个：
> 看 HBM 实测带宽。这是整章唯一必须先做的事。
>
> 无论哪个成立，有一点是共同的：**v7 的三条通道里，算力涨 5 倍、
> HBM 涨 2.6 倍、ICI 只涨 2 倍。瓶颈一定在后两条上，不在 MXU。**
> 继续盲扫 XLA flag 期望很低（§5 已经证明杂项 flag 组收益 ±0）。

对照组：同硬件上 DeepSeek V3 报 26.6%。它激活 37B（Hy3 是 21B），
每读一遍权重能摊到更多计算，算术强度天然更高 —— 与 H1 自洽；
它的专家数和路由结构也不同 —— 与 H2 也自洽。**这是推断，没有实测支撑。**

### 7.2 开可观测性：一次跑收齐三样

**前提：`base_output_directory` 必须指向 GCS。** 默认的 `/tmp/hy3out` 是 pod 本地盘，
pod 一结束全没了 —— 这是收不到 profile 最常见的原因。

```bash
PLATFORM=v7 STEPS=25 bash run.sh prof \
  base_output_directory=gs://<你的桶>/hy3prof \
  profiler=xplane \
  skip_first_n_steps_for_profiler=8 \
  profiler_steps=5 \
  profile_cleanly=True \
  dump_hlo=True
```

| 参数 | 为什么这么设 |
|---|---|
| `skip_first_n_steps_for_profiler=8` | step 0 含编译、step 1–2 是 JAX 异步派发的假读数，**必须跳过**，否则 trace 里全是编译 |
| `profiler_steps=5` | 5 步够看清稳态；开太多 trace 文件大到打不开 |
| `profile_cleanly=True` | 每步加 `block_until_ready`，让 trace 按步对齐；代价是**测出来的 step 比真实稍慢**，别拿这轮的 MFU 当数 |
| `dump_hlo=True` | 顺手把编译后的 HLO 也捞走，零额外开销 |

产物落在：

```
{base_output_directory}/{run_name}/tensorboard/   ← xplane
{base_output_directory}/{run_name}/xla_dump/      ← HLO
```

> **上次开 profiler 那轮没跑出稳态**，因为 profiler 自身开销把它挤出了时间窗口。
> 这次 `STEPS` 要给够（≥25），并且**别在只剩几分钟配额的时候开**。

拿到 xplane 后用 xprof（TensorBoard profile 插件）打开，重点看
Trace Viewer 和 Op Profile 两个页面。

### 7.3 从 trace 里必须回答的四个问题

按这个顺序看，前一个不回答清楚，后面的都是猜：

| # | 问题 | 在哪看 | 判据 |
|---|---|---|---|
| ① | **HBM 带宽跑到多少？** | Memory Bandwidth / Op Profile | 接近 7.4 TB/s → §7.1 假说成立，走 A 组；不到一半 → 走 B 组 |
| ② | 时间花在计算还是通信？ | Trace Viewer，看 collective op 条 | all-to-all + all-gather 占比 |
| ③ | 通信藏住了吗？ | 看 collective 与 GEMM 条**是否重叠** | 串行排列＝没藏住 |
| ④ | GMM（分组矩阵乘）占多少？ | Op Profile 按 op 排序 | 它是 MoE 主计算路径，占比低说明时间被别处吃了 |

**① 是分水岭。** 它决定后面走 A 组还是 B 组，别跳过。

### 7.4 实验清单

按 §7.3 ① 的结论二选一。每条都标了预期收益、成本、和**判定标准**
（跑之前先想清楚"什么结果算成功"，否则容易自我说服）。

#### A 组 —— 若确认 memory-bound：提高算术强度

| # | 实验 | 机理 | 成本 | 判定 |
|---|---|---|---|---|
| A1 | **FP8** | 权重字节数减半 → 算术强度翻倍 | 中（要验数值稳定性） | **DSV3 同硬件实测只有 +21.4%**（§6.1），所以预期是「有收益但不翻倍」。若我们这里也是 +20% 上下，说明 H2 成立；若明显更高，H1 成立 |
| A2 | `per_device_batch_size` 8 → 12 / 16 | 一次读进的权重摊到更多 token | 低（可能 OOM） | 每档记 MFU + HBM 峰值，看拐点 |
| A3 | `max_target_length` 4096 → 8192 | attention 部分算术强度随 seq 涨 | 低 | §5 消融里 8192/pdbs4 与 4096/pdbs8 打平，**这次固定 pdbs 单独扫 seq** |
| A4 | 修 `use_tokamax_gmm` 死锁 | 官方 GMM kernel，可能少读一遍权重 | 高（要查根因） | 见 §7.5 |

> **§6.1 的「先别指望」指的是「别指望翻倍」，不是「别做」。**
> +21.4% 在当前水位上是实打实的一档，而且它同时是一次**判别实验** ——
> 收益接近 DSV3 的 +21.4% 就支持 H2，明显更高就支持 H1。

#### B 组 —— 若不是 memory-bound：藏通信 / 修 kernel

| # | 实验 | 机理 | 成本 | 判定 |
|---|---|---|---|---|
| B1 | `scan(unroll=N)`，扫 1/2/4/8 | `while` 循环体是 XLA 调度边界，跨迭代不能重排；unroll 把 N 层放进同一调度域，延迟隐藏调度器才有得发挥 | 低（约 10 行） | 同时记吞吐 / 编译时间 / HBM 峰值三条曲线 |
| B2 | 并行策略：试 expert parallelism | 当前 `ici_fsdp=-1, tp=1` 纯 FSDP，MoE 的 all-to-all 全走 ICI；而 ICI 只涨了 2.0×，是三条通道里最弱的一条 | 中 | 看 all-to-all 占比是否下降 |
| B3 | MoE tile 参数 | v5p 上设了 18 个（`{wi,wo} × {fwd,dlhs,drhs} × 3 维`），**v7 上一个都没设**，完全没探过的面 | 低 | 零成本可试 |
| B4 | `remat_policy` / `decoder_layer_input=offload` | offload 到 host 会吃 PCIe；若 HBM 不紧张就不该 offload | 低 | 关掉看 HBM 峰值和 step |

#### 两组都该做的

| # | 实验 | 说明 |
|---|---|---|
| C1 | 关掉 9 个 SparseCore flag 再测一次 | §5 说它们收益为 0。**收益为 0 的东西留着只会增加变量**，确认无害就删掉，让后面的实验干净 |
| C2 | 单层计时 | `scan_layers=True` 把 80 层折成一个 scan，分不出层内开销。临时 `scan_layers=False` 跑几步，能看到单层结构 |

### 7.5 `use_tokamax_gmm` 死锁（唯一"官方有、我们用不了"的加速手段）

它是 MoE 的主计算路径，也是 `use_gmm_v2` 的强制前置，所以两个一起用不了。
怀疑是 **192 个专家不是 2 的幂**，导致分组矩阵乘的组划分出问题。
如果能修，可能是 A 组里最大的单项收益。

排查建议：先用缩层模型（`hunyuan3-smoke`，4 层）在 4 芯片小池上复现，
几十秒一轮，比在 64 芯片上试快两个数量级。

### 7.6 结果记录

每跑完一轮回填一行。**失败和零收益也要写** —— 否则下一个人会重跑一遍。

| 日期 | 实验 | 改了什么 | step (s) | TFLOP/s/chip | MFU | 结论 |
|---|---|---|---|---|---|---|
| 2026-07-30 | baseline | §4.3 参数集 | 20.43 | 445.1 | 19.29% | 起点 |
| 2026-07-30 | 二次复现 | 同上，换集群 | 19.90 | 457.3 | 19.82% | 可复现（§4.2.1） |
| | | | | | | |

---

## 8. 已知限制

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
| 编译时间 | 10–17 分钟 | 约 9 分钟 |
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
