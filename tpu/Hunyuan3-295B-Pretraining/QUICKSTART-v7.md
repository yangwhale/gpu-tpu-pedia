# 混元 3（295B-A21B）在 TPU v7 (Ironwood) 上预训练 — Quick Start

用 MaxText 在 TPU v7 上跑腾讯混元 3。**这份文档直接给最优配方 —— 照着跑，第一次就拿到当前最高水位，不用再自己调。**

| | |
|---|---|
| 模型 | Tencent Hunyuan 3，295B 总参 / 21B 激活，**完整 80 层** MoE |
| 平台 | TPU v7 Ironwood（`tpu7x-standard-4t`，**2 device / chip**） |
| 框架 | MaxText（nnx），代码在 [`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3) |
| 精度 | BF16 计算 / FP32 主权重 |

**两个规模的实测水位**（2026-08-04，均为完整 80 层、seq 4096、合成数据、稳态取 step 4–7）：

| 规模 | 并行度 | pdbs | step | **TFLOP/s/chip** | **MFU** | **tok/s** | tok/s/chip | 峰值 HBM |
|---|---|---|---|---|---|---|---|---|
| **256 chip** 极限 | `DP=2 × FSDP=256` | **16** | 30.40 s | **599** | **25.96%** | **1,103,757** | **4,312** | 92.33 G |
| **256 chip** 推荐 | `DP=4 × FSDP=128` | 12 | 23.56 s | 580 | 25.12% | 1,068,372 | 4,173 | 91.94 G |
| **64 chip**（16 节点 / 128 dev） | `DP=1 × FSDP=128` | 12 | 23.54 s | 580 | 25.14% | 267,284 | 4,176 | 91.94 G |
| 参考：未调优起点（64 chip） | `FSDP=128` | 8 | 19.92 s | 457 | 19.80% | 210,570 | 3,290 | 74.20 G |
| 🔬 **FP8（无 QAG）** | `DP=2 × FSDP=256` | 16 | 29.46 s | 618 | 13.39%<sub>对 FP8 峰值 4614</sub> | 1,139,022 | 4,449 | 92.80 G |
| ⭐ **FP8 + QAG**（64c 最优） | `DP=2 × FSDP=64` | 7 | 12.73 s | **625** | 13.55%<sub>对 FP8 峰值</sub> | 288,222 | **4,503** | 92.42 G |

> **FP8 的 MFU 分母是 4614，不是 BF16 的 2307 —— 别拿它跟上面几行直接比大小。**
> 64 芯片开 QAG 后是 **625 / 13.55%**；DSV3 官方同口径 743.5 / 16.1%。
>
> ⚠️ **本文档早前写过「FP8 那条 kernel 的 tile 一次都没扫过、潜力约 726」，这条已被实测证伪。**
> FP8 在 `use_tokamax_gmm=True` 下**内部仍回到 tokamax**，tile monkeypatch 一直生效；
> 2026-08-05 又扫了 tile / XLA flag / SparseCore 卸载 / 推 batch 共 8 格，**无一正收益**。
> 剩余差距要靠加卡、改模型形状或写代码，不在调参范围内。
> 详见 [TUNING-v7 §4.6 总表](TUNING-v7.md#46-什么能调什么不能调--一张总表)。

> **token 吞吐 = `device 数 × pdbs × seq ÷ step`。** 横向比只看 **tok/s/chip**，
> 整机 tok/s 随规模走不可比。参照：v5p 256 chips 是 **1,037** tok/s/chip，
> GB300 是 **6,242** tok/s/GPU —— **v7 现在到了 GB300 单卡的 69.1%，调优前只有 51.4%。**

> ### 🔬 64 芯片和 256 芯片跑出了同一个数
>
> **580 vs 580，MFU 25.14% vs 25.12%，峰值 HBM 91.94 G vs 91.94 G —— 一字节不差。**
>
> 这不是巧合：`DP=4 × FSDP=128` 就是**四个独立的 64 芯片作业**，
> 组内做 FSDP 集合通信，组间只在每步末尾同步一次梯度。
> 每卡的分片形状、tile 匹配、激活占用完全相同，所以 per-chip 性能必然相同。
>
> ⇒ **不必为了性能去抢大 slice，16 节点就能拿到全部收益。**
> ⇒ **调优可以在 16 节点上做**，只要改的开关不改变分片形状（tile、pdbs 都不改）。
>
> **那 256 芯片多出来的 3.3%（599）从哪来？** 不是规模，是**多了一个 `FSDP=256` 的选项** ——
> FSDP 翻倍让每卡静态分片减半，腾出 13 G 显存，够把 pdbs 从 12 一路推到 16。
> 64 芯片只有 128 个 device，**没有更宽的 FSDP 可选，所以 580 就是它的天花板。**
>
> 目标是 600–630（稀疏 MoE 在 Ironwood 上的实际水位，见 [TUNING-v7 §3](TUNING-v7.md#12-目标为什么是-600630不是-900)）。
> 当前 599，**差 0.2%**。为什么是这个目标、还剩哪些没试，见
> **[TUNING-v7.md — 性能调优实践](TUNING-v7.md)**。

---

## 0. 最优配方速查

**三件事决定了从 457 到 580 的全部涨幅**，按贡献排序：

| # | 手段 | 值多少 | 代价 |
|---|---|---|---|
| 1 | **tokamax `tile(512, 2048, 1536)`** | **+17.4%** | 需 6 行 monkeypatch（[§3.3](#33-注入-tokamax-tile必做收益最大的一步)）；显存 +1.1 G |
| 2 | **`per_device_batch_size` 8 → 12** | **+9.0%** | 显存 74 → 92 G，逼近 94.74 上限 |
| 3 | **`DP × FSDP` 切法选对**（FSDP 固定 128） | 见下 | 无 |

**第 3 项的意思**：512 device 有多种切法，但可用区间很窄 ——

```
512 device 的切法（pdbs 固定 8 时的对照）：
FSDP=512 (DP=1)  → 404  ❌ 摊太薄，集合通信分片碎片化，掉 11%
FSDP=256 (DP=2)  → 450  ⭕ 与 128 打平，但省 13 G 显存 ← 这 13 G 后面有用
FSDP=128 (DP=4)  → 453  ✅
FSDP=64  (DP=8)  → OOM  ❌ 每卡静态分片翻倍，撑爆 HBM
FSDP=32  (DP=16) → OOM  ❌
```

**默认规律：把 FSDP 宽度固定在 128，多出来的 device 全部给 DP。**
64 芯片正好 128 device 所以 `DP=1`，256 芯片 512 device 所以 `DP=4`，1024 device 就是 `DP=8`。

**但 ≥ 256 芯片时还有一条换法：把 FSDP 加宽到 256，用省下的 13 G 显存换更大的 batch。**

| 512 device 的两条路 | FSDP | pdbs | chip | HBM | 取舍 |
|---|---|---|---|---|---|
| **推荐**（与 64 芯片同构） | 128 | 12 | 580 | 91.94 G | 配方跟 64 芯片完全一致，规律可外推 |
| **极限**（榨显存） | 256 | **16** | **599** | 92.33 G | **高 3.3%**，但配方与小规模不通用 |

**追求可移植就用 DP=4，追求极值就用 DP=2 + pdbs 16。**
后者高 3.3%（已超出 ±3% 噪声，是真实差异），代价是这套配方在 64 芯片上无法复现。

**EP（专家并行）不要用。** TPU 的 ICI 是 3D torus，AllToAll 要多跳转发，
不像 GPU NVLink 那样是 full mesh。16 芯片实测 EP=4 是 **−71%**
（[TUNING-v7 §7.4](TUNING-v7.md#37-并行度怎么切可用区间只有-fsdp--128-256)），换更大规模没有翻正的物理依据。

### 直接可抄的两组参数

**64 芯片（16 节点 / 128 device）→ 580**

```
ici_fsdp_parallelism=-1      # 自动吃满 128 路，等价 DP=1
ici_tensor_parallelism=1
per_device_batch_size=12     # 91.94 G，逼近上限；14 会 OOM
megablox=True use_tokamax_gmm=True
TK_TM=512 TK_TK=2048 TK_TN=1536      # 环境变量，配合 tkcfg.py
```

**256 芯片（64 节点 / 512 device）→ 580，与 64 芯片同构**

```
ici_data_parallelism=4       # ← 与 64 芯片版唯一的差别
ici_fsdp_parallelism=128
ici_tensor_parallelism=1
per_device_batch_size=12
megablox=True use_tokamax_gmm=True
TK_TM=512 TK_TK=2048 TK_TN=1536
```

**256 芯片极限版 → 599**（配方与小规模不通用，只在 ≥ 256 芯片可用）

```
ici_data_parallelism=2
ici_fsdp_parallelism=256     # 加宽 FSDP，每卡静态分片减半，省 13 G
ici_tensor_parallelism=1
per_device_batch_size=16     # 省下的显存全喂给 batch，92.33 G
megablox=True use_tokamax_gmm=True
TK_TM=512 TK_TK=2048 TK_TN=1536
```

其余参数三组完全一致，见 [§4.3](#43-完整参数集)。

---

## 1. 模型与代码

### 1.1 Hy3 是什么

一句话：**attention 是 Qwen3 的，MoE 是 DeepSeek V3 的**，
MaxText 里这两半都已有现成实现，本项目只写了装配逻辑，零新数学。

| | |
|---|---|
| 结构 | 80 层，第 0 层 dense、1–79 层 MoE |
| Attention | GQA 64q / 8kv，head_dim 128，QK-LayerNorm，无 bias |
| MoE | **192** routed experts top-8 + 1 shared，sigmoid 路由 + 专家偏置 |
| 其他 | MTP 1 层，vocab 120832，routed scaling 2.826 |
| 参数分布 | **97% 在路由专家里**，attention 只占 2% |

参数分布直接决定并行策略：**TP 无用**（切 attention 纯亏通信）、**EP 是负优化**、
**FSDP 是主力**。

> ⚠️ **192 不是 2 的幂，这个数字会反复咬你。**
> `shard_exp_on_fsdp=True` 在 128 device 上直接 `IndivisibleError`（192 % 128 ≠ 0）；
> EP 也只能取 192 的因数。选并行度时先检查整除。

> 架构的完整拆解 —— 为什么两个现成 decoder block 都不行、与 DSV3 的关键差异
> （Hy3 没有 device-limited routing）、参数量分解 —— 见
> [v5p Quick Start §1](QUICKSTART-v5p.md)。**两个平台完全一样，这里不重复。**

### 1.2 代码从哪来

**唯一真相是 fork 的分支**，本仓不留代码副本：

```
https://github.com/yangwhale/maxtext   分支 hunyuan3
```

基于上游 main，三个 commit：模型本体 + 注册、上游 loss-free-balancing bias 路径修复
（与 Hy3 无关，任何非 DeepSeek 模型开 aux-loss-free 均衡都会撞）、SwiGLU 激活截断白名单。

**新增 3 个文件**（`models/hunyuan3.py` 161 行有效代码 + 两个 yml），
改动上游 12 个文件全部是「让框架认识这个模型」，没有一处算法实现。

**v5p 和 v7 用同一份代码、同一个镜像**，差别只在启动参数。

---

## 2. 环境准备

> **用别人已建好的托管集群（Kueue / 排队制）？§2.1 和 §2.2 跳过** —— 那种集群不让你自己建
> workload policy 和节点池，提任务它自动给机器。但要先读 [§3.7](#37-托管kueue集群上扩不出节点时怎么定位)。

### 2.1 workload policy（v7 特有，必须先建）

v5p 建节点池一条命令就完了。tpu7x 会直接被拒：

```
Creation of a managed instance group with tpu7x-standard-4t machine type
with placement policy is not supported. Use workload policy instead.
```

gcloud 577 没有这个子命令，只能走 REST：

```bash
P=YOUR-PROJECT
TOK=$(gcloud auth application-default print-access-token)

for TOPO in 4x4x4 4x8x8 2x2x1; do
  curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
   "https://compute.googleapis.com/compute/v1/projects/$P/regions/us-central1/resourcePolicies" \
   -d "{\"name\":\"wp-$TOPO\",\"workloadPolicy\":{\"type\":\"HIGH_THROUGHPUT\",\"acceleratorTopology\":\"$TOPO\"}}"
done
```

- `acceleratorTopology` **必须带**。只给 `type` 会报
  `does not support TPU topology with group placement policy and workload policy at the same time`
- **一个 policy 对应一个拓扑**，用几种拓扑就建几个

### 2.2 TPU 节点池

```bash
# 64 芯片（16 台）—— 拓扑 4x4x4
gcloud container node-pools create np-v7x-64 \
  --cluster=CLUSTER --project=$P --region=us-central1 --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=4x4x4 \
  --placement-policy=wp-4x4x4 --num-nodes=16 \
  --disk-type=hyperdisk-balanced --disk-size=200 --scopes=cloud-platform

# 256 芯片（64 台）—— 拓扑 4x8x8，其余同上
#   --tpu-topology=4x8x8 --placement-policy=wp-4x8x8 --num-nodes=64
```

四个跟 v5p 不一样的地方：

| | 说明 |
|---|---|
| `--placement-policy` | 指向 §2.1 建的那个，**拓扑要对得上** |
| `--disk-type=hyperdisk-balanced` | **v7 不接受普通 pd** |
| zone | v7 在 **`us-central1-c`**，v5p 在 `-a` |
| `--num-nodes` | = 芯片数 ÷ 4，且必须与 `--tpu-topology` 相乘一致 |

> 想长时间稳定占用（不被抢占）用 DWS flex-start：**`--flex-start` 和
> `--enable-queued-provisioning` 必须同时给**，只给一个会收到误导性报错
> （`Queued_provisioning doesn't support TPUs` 其实是缺 `--flex-start`）。
> 排队按 **20 小时**预期，不是分钟级。

### 2.3 JobSet CRD

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/download/v0.11.1/manifests.yaml
kubectl wait --for=condition=Available deploy/jobset-controller-manager \
  -n jobset-system --timeout=180s
```

v0.11.1 自带证书，**不需要 cert-manager**。

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
GKE 凑不出一整块 `/14` 给 pod。**建自己的 custom VPC 最省事**，顺带能把 MTU 开到 8896：

```bash
gcloud compute networks create NAME-vpc --subnet-mode=custom --mtu=8896
gcloud compute networks subnets create NAME-uc1 --network=NAME-vpc \
  --region=us-central1 --range=10.124.0.0/22 \
  --secondary-range=pods=10.125.0.0/16,services=10.124.16.0/20 \
  --enable-private-ip-google-access
```

### 2.6 建池之前先看一眼容量

**配额决定你能不能申请，容量决定你能不能拿到，这是两个独立的闸门。**
池子停在 `PROVISIONING` 且 MIG 没有 error，那就是纯粹排不到机器，**换项目、提配额都无效**。

```bash
gcloud compute instances list --project=ANY-PROJECT-IN-SAME-ZONE \
  --filter="machineType~tpu7x AND status=RUNNING" \
  --format='value(zone,scheduling.provisioningModel)' | sort | uniq -c
```

> `4x4x4`（64 芯片）是**原子切片**，没有 48 / 56 这种中间档 —— **凑不满等于拿不到**。
> 全球只有 4 个 zone 有 `tpu7x-standard-4t` 机型，其它 zone 是物理上没有，不是配额问题。

---

## 3. 跑起来

### 3.1 常驻环境 vs 一次性 Job

**强烈建议用常驻 `sleep infinity` 的 pod，而不是每轮提一个一次性 Job。** 三个理由：

1. **占住 slice** —— 共享集群里一放手就被别人拿走，实测有人在释放后 30 秒内被抢
2. **复用编译缓存** —— `jax_cache_dir` 在 pod 里，第二轮起编译从 10+ 分钟降到秒级，
   一轮总耗时约 **6 分钟**
3. 代码只拉一次

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata: {name: hy3-dev, namespace: <NS>}
spec:
  failurePolicy: {maxRestarts: 3}          # ⚠️ 不要设 10，见 §3.6
  replicatedJobs:
  - name: slice-job
    replicas: 1
    template:
      spec:
        parallelism: 64                     # 64 芯片版写 16
        completions: 64
        backoffLimit: 0
        template:
          spec:
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: "4x8x8"     # 64 芯片版写 4x4x4
              # 托管集群上通常还要加 reservation-name / queue-name / priorityClassName
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            tolerations: [{operator: "Exists"}]
            containers:
            - name: jax-tpu
              image: <MAXTEXT_RUNNER_IMAGE>
              securityContext: {privileged: true}
              ports: [{containerPort: 8471}, {containerPort: 8080}]
              command: ["bash","-c"]
              args: ["gcloud storage cp <GCS_STAGE>/hy3-maxtext.tgz /tmp/p.tgz &&
                      cd /deps && rm -rf src/maxtext && tar xzf /tmp/p.tgz && sleep infinity"]
              resources: {limits: {google.com/tpu: 4}}
              volumeMounts: [{mountPath: /dev/shm, name: dshm}]
            volumes: [{name: dshm, emptyDir: {medium: Memory}}]
```

代码包由 `prep.sh` 生成（clone `hunyuan3` 分支 → 8 项自检 → tar 整棵 `src/maxtext` → 传 GCS）。
**注意是整棵覆盖，不是只注入改动文件** —— 只注入的话，测的是「我的改动 + 容器里的旧基座」。

### 3.2 齐步执行：多机 TPU 必须所有 pod 同时起

```bash
#!/bin/bash
# hy3-run.sh —— 在全部 pod 上并行执行同一条命令
CMD=${1:?}; NS=${NS:-default}; JS=${JS:-hy3-dev}; NP=${NP:-64}
mapfile -t PODS < <(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS \
  --field-selector status.phase=Running --no-headers | awk '{print $1}' | sort)
[ ${#PODS[@]} -eq $NP ] || { echo "需要 $NP 个 Running pod, 现在 ${#PODS[@]}"; exit 1; }
echo "[hy3-run] 在 ${#PODS[@]} 个 pod 上并行执行（下方是 worker-0 的实时输出）"
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
for i in "${!PODS[@]}"; do
  if [ "$i" -eq 0 ]; then
    # ⚠️ worker-0 必须实时 tee 出来，否则一轮 6–30 分钟里完全看不到进度，
    #    无法区分「在编译」和「已卡死」
    timeout -k 30 2700 kubectl exec "${PODS[$i]}" -n $NS -c jax-tpu -- \
      bash -c "$CMD" 2>&1 | tee "$TMP/0.out" &
  else
    timeout -k 30 2700 kubectl exec "${PODS[$i]}" -n $NS -c jax-tpu -- \
      bash -c "$CMD" > "$TMP/$i.out" 2>&1 &
  fi
done
wait
grep -lE "^Traceback" $TMP/*.out | sed 's/^/⚠ 报错: /'
grep -ohE "SLICE_FAILURE_[A-Z_]+" $TMP/*.out | sort -u | sed 's/^/🔴 硬件故障: /'
```

> **看不到输出时怎么判断死活**：`kubectl exec <pod> -- ps -eo stat,pcpu,etime,comm --sort=-pcpu | head -3`。
> `%CPU` 几百是在多线程编译（正常），接近 0 且无 `train.py` 才是真卡死。

**冒烟必须先做**：

```bash
NS=<ns> JS=<jobset> NP=<pod数> bash hy3-run.sh 'python3 -c "import jax;print(jax.device_count())"'
```

| 规模 | `NP` | 应返回 |
|---|---|---|
| 16 节点 / 64 chip | 16 | **128** |
| 64 节点 / 256 chip | 64 | **512** |

数字不对就别往下走 —— 多机 TPU 少一个 pod，后面全是无效实验。

> ⚠️ **绝对不要用 `kubectl exec` 单独在某一个 pod 里 `import jax`。**
> 那个进程会抓住 `/dev/vfio/*` 不放，之后所有训练都报
> `Device or resource busy; Couldn't open iommu group`，只能重建 pod。

### 3.3 注入 tokamax tile（必做，收益最大的一步）

MaxText 不暴露 tokamax 的 tile 参数，而**默认值会回退到 `128³`，慢 12.4 倍**。
用 6 行 monkeypatch 注入：

```python
# tkcfg.py —— 在 import train 之前 exec
import os, dataclasses
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as P
_TM, _TK, _TN = (int(os.environ[k]) for k in ("TK_TM", "TK_TK", "TK_TN"))
_orig = P.PallasMosaicTpuRaggedDot._get_heuristics_config
def _patched(self, ba):
    c = _orig(self, ba)
    k, n = ba.arguments["rhs"].shape[-2], ba.arguments["rhs"].shape[-1]
    return dataclasses.replace(c, tile_m=_TM, tile_k=min(_TK, k), tile_n=min(_TN, n))
P.PallasMosaicTpuRaggedDot._get_heuristics_config = _patched
print("[tkcfg] patched")
```

`kubectl cp` 到每个 pod 的 `/tmp/tkcfg.py`（并行 cp，64 个串行会很慢）。

**为什么是 `(512, 2048, 1536)`** —— 三条实测规律：

| 维度 | 最优 | 说明 |
|---|---|---|
| `tile_n` | **1536** | 必须 `= base_moe_mlp_dim`。1024 不整除会 `AssertionError`，512 能整除但切三刀反而更慢 |
| `tile_k` | **2048** | 甜点。不是抄表的 1024，也不是越大越好（4096 直接 OOM） |
| `tile_m` | **512** | 表里 `m` 落在 1024 档，但实测 512 快 3.9%。**抄表是好起点，不是终点** |

> 长期正解是跑官方 autotune 生成 cache 条目；注入是验证手段，但足以拿到全部收益。
> 完整 15 组扫描见 [TUNING-v7 §7.8.1](TUNING-v7.md#34-第三步-174tokamax-tile--最大单项)。

### 3.4 跑一轮

```bash
XLA='--xla_tpu_scoped_vmem_limit_kib=65472 --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True --xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_scheduler_percent_shared_memory_limit=150 --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false'

NP=64 bash hy3-run.sh "
export LIBTPU_INIT_ARGS='$XLA' JAX_PLATFORMS=tpu,cpu
export TK_TM=512 TK_TK=2048 TK_TN=1536
cd /deps && python3 -c 'exec(open(\"/tmp/tkcfg.py\").read());
  import runpy; runpy.run_module(\"src.maxtext.trainers.pre_train.train\", run_name=\"__main__\")' \
  src/maxtext/configs/base.yml model_name=hunyuan3-295b override_model_config=True \
  ici_data_parallelism=4 ici_fsdp_parallelism=128 ici_tensor_parallelism=1 \
  per_device_batch_size=12 max_target_length=4096 \
  megablox=True use_tokamax_gmm=True sparse_matmul=True use_custom_sort_vjp=True \
  scan_layers=True remat_policy=custom decoder_layer_input=offload out_proj=remat \
  attention=flash use_tokamax_splash=True sa_use_fused_bwd_kernel=True \
  sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 \
  sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 \
  sa_block_q_dq=2048 sa_block_kv_dq=2048 \
  opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True \
  allow_split_physical_axes=True dtype=bfloat16 weight_dtype=float32 \
  tokenizer_type=tiktoken tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
  dataset_type=synthetic enable_checkpointing=False steps=8 jax_cache_dir=/tmp/jcn \
  base_output_directory=<GCS_OUT> run_name=myrun
"
```

**64 芯片版**只改两处：`NP=16`，并把
`ici_data_parallelism=4 ici_fsdp_parallelism=128` 换成 `ici_fsdp_parallelism=-1`。
**`per_device_batch_size` 保持 12** —— 64 芯片和 256 芯片在这个值上都是最优（各自 580）。

**256 芯片极限版**（599）再改两处：`ici_data_parallelism=2 ici_fsdp_parallelism=256`、
`per_device_batch_size=16`。

**每轮之间必须清理**，否则下一轮起不来：

```bash
NP=64 bash hy3-run.sh 'pkill -9 -f "pre_train[.]train"; rm -f /tmp/libtpu_lockfile'
```

> 模式串写 `'pre_train[.]train'` 而不是 `'pre_train.train'` ——
> 后者会匹配到执行这条命令的 shell 自己的命令行，把自己杀掉。

### 3.5 取数

```bash
grep -oE "completed step: [4-7], seconds: [0-9.]+"        $LOG   # step（稳态取 4–7）
grep -oE "completed step: [4-7].*TFLOP/s/device: [0-9.]+"  $LOG   # per-device
grep -ohE "Total hbm usage >= [0-9.]+G"                    $LOG   # 峰值 HBM
```

**token 吞吐要自己算**（框架不直接报）：

```
tok/s        = device 数 × per_device_batch_size × max_target_length ÷ step
tok/s/chip   = tok/s ÷ (device 数 / 2)          # v7 是 2 device/chip
```

例：512 device × pdbs 16 × seq 4096 ÷ 30.40 s = **1,103,757 tok/s** = **4,312 tok/s/chip**。

**单位换算 —— v7 最容易出错的一步。v7 是 2 device / chip**（v5p 是 1:1），框架日志一律按 device 报：

```
per-chip TFLOP/s = 日志里的 TFLOP/s/device × 2
MFU              = per-chip ÷ 2307
```

> **方向搞反就是 4 倍的误差。** 跨代际比 MFU 之前必须先确认 device : chip 比例。
> 同理，`per_device_batch_size` 里的 "device" 在 v7 上是**半个芯片** ——
> 跟 v5p 对比时 v7 这边要减半才等价。

**两个开跑就能查的健康检查**（不对说明代码没打全）：

| 日志字段 | 应为 |
|---|---|
| `number parameters` | **298.786 billion**（与 v5p 逐位一致，跨平台恒定） |
| `Total TFLOPs` | seq 4096 下约 **4547**。若是 5 倍左右，说明 FLOP 公式没加 `HUNYUAN3`，MFU 会虚高 |

### 3.6 读日志 / 保命的五条规矩

1. **先确认 pod 全 Running 再看日志。** TPU 切片全有全无，人不齐时活着的 pod 会报
   `GetSliceInfo can only be invoked after a slice is built` —— 那是症状不是病因。
2. **判错看最早那条，不是日志尾。** 配置非法会先把 TPU 拉起来再退，真正的报错
   （`MAXTEXT CONFIG ERROR` / pydantic `Value error`）在日志上方。
3. **step 0 含编译，step 1/2 是 JAX 异步派发的假读数**，稳态取 step ≥ 3。
4. **日志必须落盘。** pod 一旦被删、被抢占、或被集群时间上限杀掉，`kubectl logs` 就再也读不到。
5. 🔴 **撞到 `SLICE_FAILURE_*` 立刻整体中止，绝不重试。** 这是硬件故障，
   每次重启都会在同一层崩，耗尽 `maxRestarts` 后 JobSet 进 `Failed`，
   **没有消费方了 autoscaler 会把节点缩回 0，卡直接没了**。
   判据：`completed step: 0` 出现过（说明编译和执行路径都通）、峰值 HBM 远低于上限、
   **每次崩的 worker 不同**。所以 `maxRestarts` 设 3，不要设 10。

```bash
# 每轮跑完先查，命中就 exit
grep -q "SLICE_FAILURE" $LOG && { echo "TPU 硬件故障，换一批节点"; exit 2; }
```

### 3.7 托管（Kueue）集群上扩不出节点时怎么定位

**`kubectl` 层面看不到扩容失败的真正原因。** pod 事件只会给你
`Pod didn't trigger scale-up: ... in backoff after failed scale-up`。真原因在 MIG 里：

```bash
gcloud container clusters describe <CLUSTER> --region <REGION> \
  --format='value(nodePools[].instanceGroupUrls)' | tr ';' '\n' | grep tpu
gcloud compute instance-groups managed list-errors <MIG> --zone <ZONE> \
  --format='value(error.code,error.message)'
```

一次真实输出：`QUOTA_EXCEEDED  Quota 'HDB_TOTAL_GB' exceeded. Limit: 40960.0` ——
**区域磁盘配额打满**。TPU 节点启动盘是 hyperdisk-balanced（100 GB/台），
跟别人 PVC 抢同一个区域配额；别人留下的 `Released` PV 会一直占着。

排查顺序（**直接看 pod 会误判**）：

| 步骤 | 看什么 | 卡在这说明 |
|---|---|---|
| ① | `kubectl get clusterqueue` / `get workload` | 配额记账没过 |
| ② | MIG `list-errors` | **配额/容量，扩不出机器** |
| ③ | `kubectl get nodes` | 机器建出来了但没注册 |
| ④ | `kubectl get pods` | 常规问题：镜像、配置 |

> **被 admit ≠ 有机器。** 队列放行只代表记账通过。
> 反过来也成立：**账面满 ≠ 机器忙** —— 见过记账 364/375 已占、实际只跑 60 芯片的情况。

**另一个独立的坑：`exclusive-topology` 注解会让 autoscaler 少看见需求。**
它要求 leader pod 先落地、follower 才能抄它的 `gke-nodepool` 选择器。
没有空节点时 leader 落不了地，现象是 `parallelism=16` 但 `status.active=1`，**没有任何报错**。
绕法：直接在 `nodeSelector` 写死 `gke-nodepool`，或去掉该注解。

---

## 4. 实测数据

### 4.1 硬件

| | 64 芯片 | 256 芯片 |
|---|---|---|
| 节点 | 16 台 `tpu7x-standard-4t` | 64 台 |
| 拓扑 | `4x4x4` | `4x8x8` |
| JAX device | **128** | **512** |
| HBM | 192 GB / chip；**单 device 可用 94.74 GB** | 同 |
| BF16 峰值 / chip | **2,307 TFLOPS**（FP8 是 4,614） | 同 |
| 总算力 BF16 | 147.6 PFLOPS | 590.6 PFLOPS |

### 4.2 从 457 到 599：每一步值多少

全部在 256 芯片上同批次测得（同一组常驻 pod，编译缓存复用）：

| 步骤 | 增量 | chip | MFU | tok/s | 峰值 HBM | 累计 |
|---|---|---|---|---|---|---|
| 起点 | `DP4×FSDP128` / pdbs 8 / megablox | 453 | 19.64% | 835,131 | 74.20 G | — |
| **+tile** | tokamax `tile(512, 2048, 1536)` | **532** | 23.04% | 979,893 | 75.33 G | **+17.4%** |
| **+batch** | pdbs 8 → 10 | **564** | 24.45% | 1,039,573 | 84.06 G | **+24.5%** |
| **+batch** | pdbs 10 → 12 | **580** | 25.12% | 1,068,372 | 91.94 G | **+28.0%** |
| +换基座 | `DP2×FSDP256` 省 13 G，pdbs → 14 | 585 | 25.37% | 1,078,802 | 89.56 G | +29.1% |
| **+再加 batch** | 同基座 pdbs → 16 | **599** | **25.96%** | **1,103,757** | 92.33 G | **+32.2%** |

64 芯片上的同配方（`DP=1 × FSDP=128`，16 节点 / 128 device）：

| 配置 | step | chip | MFU | tok/s | tok/s/chip | 峰值 HBM | 256 芯片同配置 |
|---|---|---|---|---|---|---|---|
| megablox / pdbs 8（旧基线复现） | 19.92 s | 457 | 19.80% | 210,570 | 3,290 | 74.20 G | 453 |
| tile + pdbs 8 | 16.75 s | 543 | 23.55% | 250,343 | 3,912 | 75.33 G | 532 |
| tile + pdbs 10 | 20.28 s | 561 | 24.32% | 258,550 | 4,040 | **84.06 G** | 564（HBM **84.06 G**） |
| **tile + pdbs 12** | **23.54 s** | **580** | **25.14%** | **267,284** | **4,176** | **91.94 G** | **580**（HBM **91.94 G**） |
| tile + pdbs 14 | — | 模型预测 100.7 G → OOM，未测 | | | | | |

> **两个规模在同配方下逐点吻合，pdbs 10 和 12 两处峰值 HBM 一字节不差
> （84.06 / 91.94 G）。** 这是「DP 层不改变单步计算图」最硬的证据。
>
> **64 芯片的天花板就是 580** —— 它只有 128 个 device，
> 没有 `FSDP=256` 这个选项，无法像 256 芯片那样靠加宽 FSDP 腾显存去推 pdbs=14。

### 4.2.1 扩展性：weak scaling 100%，strong scaling 有代价

**同一批 512 device，两种扩展方式，差 11%。**

| 扩展方式 | 512 device 怎么切 | 每卡工作量 | global batch | per-chip | 相对 64 芯片 |
|---|---|---|---|---|---|
| **Weak scaling**（加卡也加 batch） | `DP=4 × FSDP=128` | 不变（pdbs 12） | **4×** | **580** | **100.0%** |
| **Strong scaling**（加卡不加 batch） | `DP=1 × FSDP=512` | 缩到 1/4 | 1× | 404 | 89% |

> 64 芯片同配方是 580（pdbs 12），256 芯片也是 580。
> **卡数翻 4 倍，per-chip 吞吐一点没掉 —— weak scaling 效率 100%。**

#### 为什么 DP 方向能做到 100%

`DP=4 × FSDP=128` 就是**四个独立的 64 芯片作业**。组内每层都要做 FSDP 的
all-gather / reduce-scatter；**组间在整个 step 里只有一次梯度 all-reduce**。

量化一下这次 all-reduce 有多便宜：

```
Hy3 梯度（bf16）           ≈ 590 GB
每卡梯度分片（FSDP=128）    = 590 / 128 ≈ 4.6 GB
ring all-reduce 传输量      = 2(p−1)/p × 4.6 GB = 6.9 GB   （p = DP = 4）
v7 ICI 单芯片双向带宽       = 1,200 GB/s
                            ──────────────────────────────
理论耗时                    ≈ 12 ms   ← 占 step 23.54 s 的 0.05%
```

**即使按 1/6 的带宽利用率保守估计（35 ms），也只有 0.15%** —— 完全淹没在
±3% 的复现噪声里。而且它还能跟反向传播的尾部重叠。

对比之下，**组内 FSDP 的集合通信每层要做两次、80 层就是 160 次**，
量级差两个数量级。**这就是「DP 便宜、FSDP 贵」的根本原因。**

#### 为什么 strong scaling 会掉 11%

`FSDP=512` 是把同一份权重摊到 4 倍的卡上，每卡分片缩到 1/4。
集合通信的**次数没变，但每次的有效载荷只有原来的 1/4** ——
固定开销（同步、启动延迟、torus 上的多跳转发）摊不动了。

> **一句话：加卡的时候要同时加 batch。**
> 只加卡不加 batch（把 FSDP 越摊越薄）在这个模型上从 453 掉到 404。

#### 三条边界条件（别过度外推）

1. **只测到 DP=4。** DP=8 / 16 尚未验证。理论上 ring all-reduce 的传输量
   `2(p−1)/p × N` 在 p 增大时趋近 `2N`（常数），所以**预期仍然接近 100%**，
   但这是推论不是实测。
2. **这是单 slice 内的结论。** 512 device 全在一个 `4x8x8` slice 里，走 ICI。
   **跨 slice 的 DP 走 DCN，带宽低一个数量级以上，结论不能直接搬。**
3. **前提是每卡工作量不变。** 若扩规模时保持 global batch 不变（真正的 strong
   scaling），就会退化成上面 404 那一行。

---

### 4.2.2 显存怎么算：两参数模型

不用撞 OOM 也能预判 batch 上限。用同基座两个实测点解 `HBM = 静态 + 斜率 × pdbs`：

```
DP4×FSDP128:  74.20 G @ pdbs 8 ，91.93 G @ pdbs 12
              → 静态 38.7 G ，斜率 4.43 G / pdbs
DP2×FSDP256:  静态 25.9 G（FSDP 翻倍，静态减半），斜率相同
```

| 基座 | pdbs 8 | pdbs 10 | pdbs 12 | pdbs 14 | pdbs 16 |
|---|---|---|---|---|---|
| `DP4×FSDP128` 预测 | 74.2 | 84.1 | 91.9 | 100.8 | 109.6 |
| `DP4×FSDP128` **实测** | **74.20** ✅ | **84.06** ✅ | **91.94** ✅ | — | **OOM** ✅ 预测对 |
| `DP2×FSDP256` 预测 | 61.4 | 73.5 | 79.1 | 87.9 | 96.8 → 判 OOM |
| `DP2×FSDP256` **实测** | **61.36** ✅ | — | **78.27** ✅ | **89.56** ✅ | **92.33** ❌ **预测错，实际跑通** |

> ⚠️ **这个线性模型在 pdbs ≥ 14 之后失效，我在这里错了第二次。**
>
> 实测 `DP2×FSDP256` 的逐段斜率：
> `8→12` 是 **4.23 G/pdbs**，`12→14` 是 **5.65**，`14→16` 骤降到 **1.39**。
> 高 batch 区间激活增长明显**次线性**（大概率是 XLA 在显存压力下改变了 remat / offload 的调度），
> 线性外推会**系统性高估**，把能跑的配置误判成 OOM。
>
> **正确用法**：
> 1. 只在**已测区间附近**（±2 个 pdbs）用它插值，**别外推超过 4 个 pdbs**
> 2. 预测 OOM 的配置**仍然值得实跑一次** —— 我两次判 OOM 都错了（pdbs 12 @ FSDP128、pdbs 16 @ FSDP256）
> 3. 预测「远超上限」（如 109.6 G，超 15%）的才可以直接排除 —— 这类预测两次都对

### 4.3 完整参数集

**并行（唯一随规模变的部分）**

```
# 64 芯片
ici_fsdp_parallelism=-1
# 256 芯片
ici_data_parallelism=4
ici_fsdp_parallelism=128
# 两者共同
ici_tensor_parallelism=1         # TP 无用，attention 只占 2% 参数
```

**MoE（含本轮最大收益项）**

```
megablox=True
use_tokamax_gmm=True             # ← 配合 tkcfg.py 注入 tile 才有意义
sparse_matmul=True
use_custom_sort_vjp=True
# 环境变量
TK_TM=512  TK_TK=2048  TK_TN=1536
```

**batch 与序列**

```
per_device_batch_size=12         # 64 芯片用 10；再往上 OOM
max_target_length=4096           # 实测 seq 8192 + pdbs 4 与之等价（451 vs 453），不必换
```

**Attention**

```
attention=flash
use_tokamax_splash=True          # v7 特有
sa_use_fused_bwd_kernel=True     # ⚠️ v5p 上这一项要设 False
sa_block_q=2048  sa_block_kv=2048  sa_block_kv_compute=2048
sa_block_q_dkv=2048  sa_block_kv_dkv=2048  sa_block_kv_dkv_compute=2048
sa_block_q_dq=2048  sa_block_kv_dq=2048
```

**重计算 / offload / 优化器**

```
scan_layers=True                 # 也让编译时间不随层数涨
remat_policy=custom
decoder_layer_input=offload
out_proj=remat

opt_type=adamw
mu_dtype=bfloat16                # Adam 一阶动量降 bf16
grad_dtype=bfloat16
use_iota_embed=True
```

优化器状态从 16 B/param 降到 12 B/param。`nu_dtype` optax 不支持单独设，
恒随 `weight_dtype`；**主权重仍是 fp32**。

**精度与基线条件**

```
dtype=bfloat16  weight_dtype=float32  allow_split_physical_axes=True
tokenizer_type=tiktoken
tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken

dataset_type=synthetic           # 只测吞吐不测收敛
enable_checkpointing=False       # 避免 I/O 干扰读数
steps=8                          # 取 step 4–7 稳态；不抓 profile 跑 8 步足够
```

### 4.4 XLA flag（15 个）

```
# 基础（2）
--xla_tpu_scoped_vmem_limit_kib=65472
--xla_enable_async_all_gather=true

# SparseCore 卸载组（9）—— v7 上收益 ±0，保留只为与官方配方对齐
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
--xla_tpu_enable_sparse_core_collective_aggregator=true      # ← 这个不能删
--xla_tpu_use_tc_device_shape_on_sc=True
--xla_sc_disable_megacore_partitioning=True

# 调度器组（4）—— 唯一值钱的一组，+6.6%
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
```

> **SparseCore 那 9 个里有 8 个确实零收益，但第 9 个
> （`collective_aggregator`）是层调度器的硬依赖**，删了直接
> `INVALID_ARGUMENT: Latency hiding layer scheduler requires sparse core collective aggregator`。
> **裁剪 flag 要成组，不能逐个删。**

> ⚠️ **libtpu 对不认识的 flag 是硬失败**（`Unknown command line flag`，进程直接退）。
> **换镜像必须重过一遍 flag 集。**

### 4.5 冒烟测试（改代码后先跑这个）

```bash
NODES=1 TOPO=2x2x1 MODEL=hunyuan3-smoke STEPS=8 \
  bash run.sh smoke per_device_batch_size=1 max_target_length=2048
```

4 层缩层，结构与 295B 完全一致（192 专家、top-8、sigmoid、专家偏置、共享专家、
GQA、QK-norm、fp32 路由、MTP 全是满配），只砍层数。

| | v7 实测 | v5p 同命令 |
|---|---|---|
| 参数量 | **16.139 B** | 16.139 B（**必须一致**） |
| `total_weights` | 16384 | 8192 |
| loss（8 步） | 13.411 → 11.091 | 13.453 → 10.354 |
| NaN / skipped | 0 | 0 |

> 两个平台 loss 序列不同**是对的**：同样 4 芯片，v7 有 8 个 device 而 v5p 只有 4 个，
> `pdbs=1` 之下 global batch 差一倍。**跨平台恒定、可当硬标准的只有参数量 16.139 B。**

**为什么 4 层能代表 80 层**：MaxText 按**类型**分组做 `scan`，79 个 MoE 层共用同一份
编译产物，层与层的差别只在权重数值上。冒烟测的是**那个被复用 79 次的唯一函数**。
它覆盖不到的：显存压力、大规模切分、完整 XLA flag 集、收敛质量、以及全部性能。

---

## 5. 验证记录

> **这份文档的每个数字都是照它自己的步骤跑出来的。** 下面记录每一轮审计的偏差 ——
> 包括被改掉的错误，因为「文档哪里会坑人」比「文档说了什么」更值钱。

### 轮 1（2026-08-04 23:16–23:30）：两个规模同时复测

**方法**：不销毁 pod（见下方说明），`hy3-run.sh` 和 `tkcfg.py` 用脚本从本文档
markdown 里**原样抠出**，训练命令按 §3.4 拼，**不使用任何历史脚本**。

| 规模 | 配方 | 文档写的 | 实测 | 偏差 |
|---|---|---|---|---|
| 64 chip | `DP1×FSDP128` + tile + pdbs 12 | step 23.54 / **580** | step **23.538** / **579.97** | 0.005% |
| 256 chip | `DP2×FSDP256` + tile + pdbs 16 | step 30.40 / **599** | step **30.399** / **598.74** | 0.04% |

冒烟：64 卡 128 devices ✅ ／ 256 卡 512 devices ✅

**抓到 4 个缺陷：**

| # | 缺陷 | 处理 |
|---|---|---|
| 1 | §0 说 64 卡 `pdbs=12`，§3.4 却写「改成 10」—— **文档内部矛盾**，旧版残留 | ✅ 已修 |
| 2 | 删 JobSet 想从零重建，**30 秒内整池被别的任务抢走**，审计中断且卡拿不回来 | ✅ 转为规矩：审计不重跑抢卡步骤 |
| 3 | `hy3-run.sh` 是 `for...& done; wait; cat`，**跑完前日志 0 字节**，6–30 分钟里无法区分编译与卡死 | ✅ 已改为 worker-0 实时 `tee` |
| 4 | §3.2 冒烟只给了 64 节点的预期值 512，没给 16 节点的 128 | ✅ 已补两个规模对照表 |

> **关于「不销毁 pod」**：在共享集群上做「清空→从零重建」的审计，代价是可能直接丢卡
> （缺陷 #2 就是这么发生的）。而「申请资源」这一步**没有可审的内容** ——
> 它的成败取决于当时集群有没有空位，跟文档写得对不对无关。
> **真正要审的是冒烟、注入、训练命令、取数、清理这些步骤**，它们全在 pod 内，不需要重建环境。

---

## 6. 已知限制

| 项 | 状态 |
|---|---|
| **距目标还差 0.2%** | 599 vs 600–630。剩余候选见 [TUNING-v7 §7.8.3 待补](TUNING-v7.md#6-还没试的) |
| 数据集 | `synthetic`。**loss 下降只证明「能算且不发散」，不是收敛证据** |
| **FP8 + QAG（已收敛）** | 64 芯片开 QAG 后 **625**（vs 无 QAG 594，**+5.3%**，且 batch 更小）；256 芯片无 QAG 618。⚠️「那条 kernel 的 tile 一次没扫过、潜力 726」**已证伪** —— FP8 内部仍走 tokamax，tile 一直生效。2026-08-05 八格实验无一正收益，**调参已见底**。见 [TUNING-v7 §4.6](TUNING-v7.md#46-什么能调什么不能调--一张总表) |
| `shard_exp_on_fsdp` | 128 device 上 `IndivisibleError`（192 % 128 ≠ 0），不可用 |
| HF 权重 → Orbax 转换 | 未做。只跑吞吐可以不碰；要 SFT 必须做 |
| 完整 loss 曲线 | 未记。建议补一条 30 步以上的 |
| 容量 | tpu7x 抢手，全球仅 4 个 zone 有机型，见 §2.6 |

---

## 附录 A：与 v5p 的差异速查

两个平台**共用同一份代码、同一个镜像**，以下是全部差异：

| | v7 | v5p |
|---|---|---|
| 机型 / 拓扑 | `tpu7x-standard-4t`，`4x4x4` / `4x8x8`，us-central1-**c** | `ct5p-hightpu-4t`，`4x8x8`，us-central1-**a** |
| 建池前置 | **必须先建 workload policy（REST）** | 无 |
| 磁盘 | **hyperdisk-balanced** | 默认 |
| **device : chip** | **2 : 1** | 1 : 1 |
| MFU 分母 | **2,307** | 459 |
| `max_target_length` | **4096** | 8192 |
| `sa_use_fused_bwd_kernel` | **True** | **False** |
| `use_tokamax_splash` | **True** | 不设 |
| `use_tokamax_gmm` + tile 注入 | **True，+17.4%** | 负收益，不用 |
| `opt_type` / `mu_dtype` / `grad_dtype` | **adamw / bf16 / bf16** | 默认 / fp32 / fp32 |
| MoE tile 参数（18 个） | 不设（改用 tokamax tile 注入） | 全设 |
| XLA flag | 15 个 | 25 个 |
| SparseCore 卸载组收益 | **±0** | **+4.07 pp** |
| **MFU** | **25.96%（256 chip）／ 25.14%（64 chip）** | 35.07%（已收敛） |

> **同一个开关在两个平台上可以反号。** `sa_use_fused_bwd_kernel`、
> SparseCore 卸载组、`use_tokamax_gmm` 三处都是。**别把一个平台的调优结论直接搬到另一个。**

---

## 附录 B：延伸阅读

| 文档 | 内容 |
|---|---|
| [TUNING-v7.md](TUNING-v7.md) | 调优实践：为什么是这个水位、全部消融表、失败项、HBM 模型 |
| [QUICKSTART-v5p.md](QUICKSTART-v5p.md) | v5p 版，含架构完整拆解；基线 35.07%，已从零验证 |
| [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) | 完整实验档案：全部轮次、12 个 bug 的复盘、DWS flex-start 建池 |
| [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) | 把别的模型移植到 MaxText 的通用范式 |
| [maxtext-hunyuan3/](maxtext-hunyuan3/) | `prep.sh` / `run.sh` |
