# RUNBOOK：TPU v7x 上跑 Qwen3.5-397B-A17B-FP8 推理测试

> **契约**：你没碰过 TPU、没用过 vllm-torchtpu，照做能拿到卡、把服务跑起来、
> 得到可以和别人对账的数字，且不需要自己排查任何问题。
>
> 每步格式：**命令 → 期望看到什么 → 不对时怎么办**。标 ⚠️ 的都是实测踩过的坑，不要精简。
>
> 同目录另两份：[QUICKSTART](./QUICKSTART.md) 是 4 chip 最短路径；
> [RUNLOG](./Qwen3.5-397B-A17B-FP8/RUNLOG-20260814-16chip.md) 是数据与规律分析。

**最近一次全流程验证：2026-08-27**（路 A，4 chip，复现成功，见 §3.7）。
过期数值统一收进文末 [History](#history过期数值与它们过期的原因)，正文只留当前有效的。

---

## 0. 先选路：你要的是「复现」还是「矩阵」

两条路共用第 1、2 节（拿卡 + pod），从第 3 节开始分叉。**别混着走。**

| | **路 A：复现 daily benchmark** | **路 B：18 格性能矩阵** |
|---|---|---|
| 目的 | 验证这套软硬件栈是不是**正常**（跟已知基线对账） | 摸清**并行策略 × 序列形状 × 并发**的规律 |
| 硬件 | **4 chip / 1 节点**（`2x2x1`）就够 | 16 chip / 4 节点（`2x2x4`） |
| 耗时 | **约 20 分钟**（拿到卡之后算起） | 约 2 小时 20 分 |
| 代码 | `tpu_benchmark_daily` + 它的 `vllm-torchtpu` submodule | 直接用 `vllm-torchtpu` 的 benchmarking 脚本 |
| 镜像 | `google/cloud-sdk:slim` + `uv` 自带 Python 3.12 | `torch-tpu` 官方 nightly 镜像 |
| 产出 | 1 个数（decode 吞吐 + TPOT），可直接对基线 | 18 格 TTFT / TPOT / ITL / E2EL / 吞吐 |
| 章节 | §3 | §4 |

> **先跑路 A 再跑路 B。** 路 A 二十分钟就能告诉你「环境是好的」。
> 直接上路 B，一个数不对你分不清是策略问题还是环境问题。

---

## 1. 拿卡：DWS flex-start

### 1.1 ⚠️ 时长必须显式写 `604800`，不写会被自己坑

官方文档对 `maxRunDurationSeconds` 的原话是
「the maximum runtime of a node in seconds, **up to the default of seven days**」
（[GKE ProvisioningRequest 文档](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/provisioningrequest)）。

也就是说 **604800 秒 / 168 小时本来就是默认值**，字段不填反而拿满七天。

> **2026-08-26 实测的反面教材**：我们的 manifest 里手填了 `"86400"`，
> 于是**排了 65 小时的队，只换到 24 小时的机器**。
> 这个字段 **创建后不可修改**，发现时已经来不及。
>
> 两个地方都能查到你实际拿到多久，**两个都要看**：
>
> ```bash
> kubectl get provisioningrequest "$PR" -n "$NS" -o jsonpath='{.spec.parameters}'
> gcloud compute instances describe "$NODE_VM" --zone "$ZONE" \
>   --format='yaml(scheduling.maxRunDuration, scheduling.instanceTerminationAction)'
> ```
>
> VM 那一层写的是 `instanceTerminationAction: DELETE` —— 到点是**删机器**，
> 不是驱逐、不是 cordon。**从节点创建时刻起算**，不是从你开始用起算。

### 1.2 建节点池

自建 v7 节点池必须先有 **workload policy**，而且 `--accelerator-topology` 要写在
policy 上（写在节点池上不算）。先看有没有现成的：

```bash
gcloud compute resource-policies list --project=CHANGE_ME_project \
  --filter="region:us-central1" \
  --format="table(name,workloadPolicy.type,workloadPolicy.acceleratorTopology)"
```

没有就建一个：

```bash
gcloud compute resource-policies create workload-policy CHANGE_ME_wp \
  --region=us-central1 --type=HIGH_THROUGHPUT --accelerator-topology=2x2x4
```

然后建池。**16 chip 版**（路 B）：

```bash
gcloud container node-pools create CHANGE_ME_pool \
  --cluster=CHANGE_ME_cluster --region=us-central1 --project=CHANGE_ME_project \
  --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=2x2x4 \
  --placement-policy=CHANGE_ME_wp \
  --flex-start --enable-queued-provisioning \
  --num-nodes=0 --enable-autoscaling --min-nodes=0 --max-nodes=4 --location-policy=ANY \
  --reservation-affinity=none \
  --disk-type=hyperdisk-balanced --disk-size=200 \
  --scopes=cloud-platform
```

4 chip 版把 `2x2x4` 换成 `2x2x1`、`--max-nodes=4` 换成 `1` 即可。

三个容易漏的：

| flag | 为什么 |
|---|---|
| `--scopes=cloud-platform` | ⚠️ **默认只给 `devstorage.read_only`**。节点 SA 读不了别的项目的权重桶，报的是普通 403，很容易误判成桶权限没配 |
| `--disk-size=200` | 100 GB 也能跑（权重走 tmpfs），但镜像 + pip 缓存会紧张 |
| `--enable-queued-provisioning` | 要走 ProvisioningRequest 就必须加。加了之后**普通 Job/Deployment 再也拉不起节点**，只能通过 PR 要资源 |

### 1.3 提 ProvisioningRequest + holder Job

PodTemplate、ProvisioningRequest、Job 三件一起提。**Job 要和 PR 同时存在**，
否则等真拿到卡时没有 pod 接，节点会空着被缩掉。

```yaml
apiVersion: v1
kind: PodTemplate
metadata:
  name: hold-tmpl
  labels: {cloud.google.com/apply-warden-policies: "true"}
template:
  spec:
    restartPolicy: Never
    nodeSelector:
      cloud.google.com/gke-nodepool: CHANGE_ME_pool
      cloud.google.com/gke-tpu-accelerator: tpu7x
      cloud.google.com/gke-tpu-topology: 2x2x4
    tolerations:
    - {key: google.com/tpu, operator: Equal, value: present, effect: NoSchedule}
    - {key: cloud.google.com/gke-queued, operator: Equal, value: "true", effect: NoSchedule}
    containers:
    - name: holder
      image: ubuntu:24.04
      command: ["bash","-c","echo HOLDER_UP $(date -u +%FT%TZ); sleep infinity"]
      resources: {limits: {google.com/tpu: 4}, requests: {google.com/tpu: 4}}
---
apiVersion: autoscaling.x-k8s.io/v1
kind: ProvisioningRequest
metadata: {name: hold168}
spec:
  provisioningClassName: queued-provisioning.gke.io
  parameters:
    maxRunDurationSeconds: "604800"        # ← §1.1
  podSets:
  - count: 4                               # ← 节点数，不是芯片数
    podTemplateRef: {name: hold-tmpl}
```

Job 用同样的 nodeSelector / tolerations / resources，`parallelism: 4`，
并在 pod 的 annotation 上挂：

```yaml
cluster-autoscaler.kubernetes.io/consume-provisioning-request: hold168
cluster-autoscaler.kubernetes.io/provisioning-class-name: queued-provisioning.gke.io
```

### 1.4 排队要多久

**实测 65 小时**（`Accepted` 2026-08-24T05:02Z → `Provisioned` 2026-08-26T22:07Z，
4 chip）。之前记的「约 40 小时」是量级不是上限。

Pod 一直 `Pending`、PR 只有 `Accepted=True (SuccessfullyQueued)` —— **这是正常排队，不是故障**。
盯梢用后台探针，别人工守着，参数按这个量级放大（`--max-age` ≥ 72 h、探测间隔 ≥ 15 min）。

### 1.5 ⚠️ `BookingExpired=True` 不是坏消息

它在 `Provisioned` 之后 **10 分钟准时出现**，原文是
「Capacity booking … has expired and the nodes are now candidates for scale down
**when underutilized**」。

只要有 pod 占着 TPU 就不会被缩掉 —— 反过来说，**占位 pod 一退出，节点大概率立刻没**。
而 flex-start 的节点**缩掉就再也回不来**（要重排 65 小时）。

### 1.6 ⚠️ 换票（占位 pod → 干活 pod）的三条铁律

2026-08-27 换票时把这三条全踩了一遍，节点差点丢掉：

**① 先钉一个不占 TPU 的 anchor pod。**

```yaml
apiVersion: v1
kind: Pod
metadata: {name: node-anchor}
spec:
  nodeName: CHANGE_ME_node        # 钉死在那台机器上
  restartPolicy: Never
  hostNetwork: true
  tolerations: [{operator: Exists}]
  containers:
  - {name: c, image: google/cloud-sdk:slim, command: ["sleep","infinity"],
     resources: {requests: {cpu: "1", memory: "2Gi"}}}
```

裸 Pod（无 controller）+ `nodeName` 钉死 → cluster autoscaler 不会驱逐它，
所以换票空窗期节点不算「空」。**这个保险必须先上，成本几乎为零。**

**② 新 pod 必须带两个 TPU nodeSelector 标签，否则被 admission webhook 直接拒。**

```
cloud.google.com/gke-tpu-accelerator: tpu7x
cloud.google.com/gke-tpu-topology: 2x2x1     # 或 2x2x4
```

缺任何一个报：

```
GKE Warden rejected ... [denied by tpu-accelerator-topology-constraints]:
Missing nodeSelector/nodeAffinity label cloud.google.com/gke-tpu-accelerator
```

而且这条约束**不能用 `cloud.google.com/generate-allowlist` 注解绕过**，报错里会明说。

**③ 先 apply 新 pod（让它 Pending），再 delete 占位 Job。** 顺序反了就是裸奔。
删 pod 没用会被 Job 重建，要删 **Job**。实测调度落地 2 秒、Running 10 秒（镜像已预热）。

---

## 2. Pod 该长什么样

不管走哪条路，pod spec 里这几项都要有。**每一项都对应一个踩过的坑。**

| 配置 | 值 | 为什么 |
|---|---|---|
| 权重目录 | `emptyDir{medium: Memory}`，`sizeLimit` ≥ 500Gi | 权重 378 GiB。落磁盘加载慢 6 倍（§3.6） |
| `/dev/shm` | **≥ 64Gi** | 默认 **64 MiB**，vLLM 起手就要 160 MiB，直接 `RuntimeError: Insufficient space in /dev/shm`。常规跑测会涨到 32 GiB 的 69–78%，所以起手给 64 |
| `/tmp`、`~/.cache` | tmpfs | ephemeral-storage 是**整节点共享配额**，容器里 `df` 看不见，撑爆会触发**节点级驱逐** |
| `securityContext` | `privileged: true` | 访问 `/dev/vfio` |
| `hostNetwork` | `true` | 多机 ICI 需要；单机也建议开，顺带让 metadata 走节点 SA |
| `tolerations` | `operator: Exists` | 节点有 `google.com/tpu=present` 和 `cloud.google.com/gke-queued=true` 两个污点 |
| `resources` | 每种资源**只出现一次** `limits` / 一次 `requests` | 两个 `limits` 块时后者覆盖前者，`google.com/tpu` 静默丢限额 |

> 💡 **`/dev/shm` 不够时不用重建 pod。** privileged pod 里直接
> `mount -o remount,size=64G /dev/shm` 就生效，服务跑着也能改。
> 重建 pod 的代价是重下 378 GiB 权重 + 重编译。

> ⚠️ **有 Lustre PVC 也别用。** TPU 节点的 COS 镜像装了 `lnet`/`libcfs` 但**没有 lustre 客户端模块**，
> 挂载报 `exit status 19 / No such device`；`lustre-csi-node` 显示 `2/2 Running` 是假象。
> 进节点 `modprobe lustre` 会看到 `Module lustre not found`。**走 tmpfs + 同区 GCS。**

**摸清节点规格**（决定后面所有容量参数）：

```bash
kubectl get nodes -l cloud.google.com/gke-nodepool=$POOL -o custom-columns=\
'NAME:.metadata.name,TPU:.status.allocatable.google\.com/tpu,CPU:.status.allocatable.cpu,\
MEM:.status.allocatable.memory,EPH:.status.allocatable.ephemeral-storage,\
TOPO:.metadata.labels.cloud\.google\.com/gke-tpu-topology'
```

`tpu7x-standard-4t` 单节点期望值：`TPU=4`、`CPU≈223370m`、`MEM=963568636Ki`（919 GiB）、
`TOPO` 与节点池一致。`EPH` 随 `--disk-size` 变（100 GB 盘 → 约 94 GiB 可分配）。

> **口径**：v7x 上 **1 chip = 2 device**。`tpu7x-N` 里的 **N 是 device 数**：
> 4 chip 写 `tpu7x-8`，16 chip 写 `tpu7x-32`。
> vLLM 日志打的 `num_chips=8` 其实也是 device 数，别按字面理解。

---

## 3. 路 A：复现 daily benchmark（2026-08-27 实测全流程）

目标：跟 `tpu_benchmark_daily` 已发布的基线对上账。**4 chip 就够。**

### 3.1 本机准备代码包（不要让 pod 自己去 GitHub）

pod 里没有 GitHub 凭据，而且实测踩过 503 和 `RemoteDisconnected`，各废一个窗口。
在本机把主仓和 submodule 都拉好，打成一个包丢 GCS：

```bash
git clone git@github.com:aios-tpu-infra/tpu_benchmark_daily.git tbd
git clone --depth 1 https://github.com/vllm-project/vllm-torchtpu.git \
  tbd/third_party/torchtpu-vllm      # submodule 位置，直接放进去

# ⚠️ 打包前把这行改掉：pod 里不能 fetch origin/main
sed -i 's/^environment_update_args=()$/environment_update_args=(--no-source-update)/' \
  tbd/scripts/daily_benchmark.sh

tar czf tbd-bundle.tgz tbd
gcloud storage cp tbd-bundle.tgz gs://CHANGE_ME_bucket/staging/
```

> ⚠️ **`daily_benchmark.sh` 默认每次都去 fetch `origin/main`。** 不改这行，
> pod 里必然卡在 `Please make sure you have the correct access rights`。
> 改成 `--no-source-update` 之后它用你 checkout 的那个 revision ——
> **对复现来说这本来就更对**，版本被钉死了。

### 3.2 权重进内存盘：`gcloud storage`，实测 45 秒

```bash
kubectl exec -n $NS $POD -- bash -c '
  M=/ram/tbd/models/Qwen3.5-397B-A17B-FP8
  cp $M/SOURCE.json /ram/SOURCE.json.bak          # 仓库自带的元数据，别被覆盖
  time gcloud storage cp -r "gs://CHANGE_ME_bucket/models/qwen3.5-397b-a17b-fp8/weights/*" $M/
  cp /ram/SOURCE.json.bak $M/SOURCE.json'
```

**实测：406 GB / 45.05 秒 = 8.6 GiB/s（约 74 Gbps）**，`tpu7x-standard-4t`
（224 vCPU）→ 同 region 桶 → `emptyDir{medium:Memory}`，默认并发，什么都没调。
单个 4.3 GB 分片单独拉是 1.7 GiB/s；`cp -r` 整目录会自己并发起来。

> ⚠️ **镜像里有没有 gcloud SDK，这一步差 40 倍。**
> 之前记录的「28 分钟 / 单节点 231 MiB/s」是**工具产物不是带宽上限** ——
> 那次容器里没有 gcloud SDK，用手写的 `google-cloud-storage` Python 下载器爬。
> 详见 [History](#history过期数值与它们过期的原因)。
>
> 推论：**选基础镜像时优先选自带 gcloud SDK 的**，比自己写下载器省几十分钟。

> ⚠️ **进度要看 `du -sh`，不要看进度条。** 94 个 4 GB 分片、高并发时，
> 按「完成文件数」计数的进度条会长时间显示 0，而盘上已经落了几百 GiB。

### 3.3 装环境：`uv` 自带 Python 3.12

`google/cloud-sdk:slim` 是 Debian，系统 Python 不是 3.12，而这套脚本**硬性要求 3.12**
（`install_vllm_service_launcher.sh` 直接 `command -v python3.12` 检查）。
让 `uv` 自己下一个：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
ln -sf "$(uv python find 3.12)" /usr/local/bin/python3.12    # 脚本按名字找，必须放进 PATH
```

> ⚠️ **tar 解包后 git 会报 dubious ownership。** 包在本机以普通用户身份打，
> 在 pod 里以 root 解开，属主对不上，`git rev-parse` 直接 fatal，
> 而上层脚本把它翻译成 **`ERROR: required submodule is not initialized`** ——
> 报错完全指不到真因。修法：
>
> ```bash
> git config --global --add safe.directory '*'
> chown -R root:root /ram/tbd
> ```

然后：

```bash
mkdir -p /run/vllm-metrics-targets/targets     # ⚠️ 不存在的话 launcher 提交直接失败
cd /ram/tbd && bash scripts/daily_benchmark.sh --prepare-only
```

期望最后几行：

```
torch_tpu import OK; PrivateUse1 backend=tpu
Offline model metadata OK: config=Qwen3_5MoeConfig, tokenizer=Qwen2Tokenizer
Preparation completed; TPU server was not started.
```

实测约 100 秒（含建 venv、从私有 Artifact Registry 装 torch + torch-tpu、
以 `VLLM_TARGET_DEVICE=empty` 装 vLLM）。

> `update_environment.sh` 刻意用 `gcloud auth print-access-token` 而不是 ADC。
> 也可以从外面注入 `TORCH_TPU_ACCESS_TOKEN` 环境变量绕过 ——
> 但 token **只活 1 小时**，而这一步可能跑十几分钟，注入时要留余量。

### 3.4 跑

```bash
cd /ram/tbd && PUBLISH_REPORTS=0 bash scripts/daily_benchmark.sh --only dp-decode
```

`--only` 可选 `dp-decode` / `dp-prefill` / `pcp-prefill`；不给就三组都跑。
`PUBLISH_REPORTS=0` 阻止它把结果 commit 回 Git。

**用 `setsid nohup` 起到后台并把日志落盘。** 这一步十几分钟起步，
挂在前台的 `kubectl exec` 断了就全没了。

### 3.5 期望看到什么

按顺序，中间任何一段卡住超过 5 分钟就去查：

| 阶段 | 日志特征 | 实测耗时 |
|---|---|---|
| 权重加载 | `Loading weights took 39.33 seconds` | **约 40 s** |
| XLA 编译 | 刷 `Compiling FX graph for range (4096, 4096)` | 十几分钟 |
| 服务起来 | `Waiting for application startup.` × 8 | — |
| 跑分 | `[round-summary] valid=True peak_active=216 windows=15` | — |
| 出数 | `[done] throughput_p50=... peak_active_tpot_p50_avg=... ms` | — |

> 🔎 **权重加载耗时是内存盘的判据**：**tmpfs 约 40 秒**（更早一次 16 chip 记录是 69 秒），
> **落 EXT4 磁盘约 240 秒**。看到 240 秒左右就回头查 pod 的 volume 配置。

> ⚠️ **`round-summary` 出现 `valid=False peak_active=0 windows=0` 是正常的。**
> 那是并发还没爬到平台期的那一轮，不是失败。**看最后一行 `[done]`。**

### 3.6 基线口径：`C256/P65536/D1024` 到底在测什么

**没有这一节，两个数字没法对账。**

**代码三层**，三层都要对：

| 层 | 这次的值 |
|---|---|
| 上层 harness | `aios-tpu-infra/tpu_benchmark_daily` |
| 推理引擎 | `vllm-project/vllm-torchtpu`，submodule commit（我们 `4ae9f631`，基线 `0be027b9`） |
| TPU 后端 wheel | `torch-tpu==0.1.1.dev20260813160135`，来自私有 Artifact Registry |

**服务配置 `dp8_decode_c256`**：

| 项 | 值 | 说明 |
|---|---|---|
| 并行 | TP=1 / **DP=8** / EP=8 | 8 个 device 各跑一份完整模型做数据并行，MoE 专家横切 |
| `--max-num-seqs` | 32 | 每个 DP rank 的槽位数 |
| **并发 256** | = 8 × 32 | **不是随便挑的**，正好把每个 rank 塞满 |
| `--max-model-len` | 66560 | |
| `--max-num-batched-tokens` | 4096 | |
| `--gpu-memory-utilization` | 0.932285943 | 精确到小数点后九位 = 贴着 HBM 边界试出来的 |
| `--kv-cache-dtype` | fp8 | |
| `compile_sizes` | 8,16,32,4096 | |
| `--seed` | 42 | |

**负载 `C256/P65536/D1024`**：256 并发；每条请求输入 **65,536 token**（64K 上下文）、
输出 **1,024 token**。

> ⚠️ **每条请求带唯一前缀，且每轮换一个 `cache-salt`。** 这是**故意不让前缀缓存命中**，
> 逼引擎真的从头算。少了这一条，测出来的是缓存命中率不是算力。

**指标怎么取**：不是全程平均。开 **10 秒滑动窗口、1 秒步进**，
**只统计并发真正打满的 plateau 窗口**。

- 主指标 `throughput_p50` —— 那些窗口输出吞吐的**中位数**
- 副指标 `peak_active_tpot_p50_ms` —— 同一段时间内 token 间隔的 P50

> **这不是「某个延迟约束下的吞吐」。** 并发被钉死在 256，吞吐和 TPOT 是**并列报出的两个数**，
> TPOT 不是约束条件。41.75 ms/token 换算过来约等于每用户 24 tok/s。
>
> 要做 SLA 曲线（「TPOT ≤ X ms 时能吃多少吞吐」）现有数据不够 ——
> decode 组只有 C256 这**一个点**，得自己把并发扫开。

**daily 文档整体是四组互不设约束的测量**，别把它们混成一个数：

| 组 | 固定什么 | 扫什么 | 报什么 |
|---|---|---|---|
| DP8 C256 decode | 并发 256、ISL 64K、OSL 1024 | — | 稳态吞吐 + TPOT |
| DP8 / PCP8 prefill 吞吐 | ISL 8192、OSL **1** | 并发 8/16/32/64 | 峰值总吞吐 |
| 单请求 TTFT | **并发 1**、OSL 1 | ISL 8K→252K 六档 | median TTFT |
| SPEED-Bench 变长 | 1000 条真实请求（756–37,719 token） | 并发 8 / 64 | input tok/s + TTFT P50/P90/P99 |

> 💡 **第三组最能说明 PCP 为什么存在**：252K 输入时 DP8 的 TTFT 是 **64.0 秒**，
> PCP8 只要 **9.8 秒**，差 6.5 倍；短输入 DP8 又不吃亏多少。
> 两条并行策略不是在比强弱，是在分工 —— 一个吃吞吐，一个吃长上下文延迟。

### 3.7 ✅ 2026-08-27 复现结果

4 chip（`tpu7x-standard-4t` × 1，`2x2x1`），DWS flex-start 节点：

| 指标 | 本次复现 | 发布基线（`0be027b9` / 2026-08-26 11:33） | 偏差 |
|---|---|---|---|
| decode 吞吐 p50 | **5,160.2 tok/s** | 5,162.40 | **−0.043%** |
| TPOT p50（peak-active） | **41.747 ms** | 41.755 | **−0.019%** |
| 权重加载 | 39.33 s | —（16 chip 记录为 69 s，tmpfs） | 更快 |

引擎 commit 比基线**新**（`4ae9f631` vs `0be027b9`）仍拿到同一个数 ——
说明这段时间的改动对 decode 路径无回归。

---

### 3.8 七个 case 的逐格记录（2026-08-27 全量三组）

§3.7 那次只跑了 decode 一组。这一节是**同一天稍后跑完的完整 daily**
（`run_id=20260827T024934Z`，10:49–12:27 HKT，1h38m，`rc=0`），
三组七格全部跑通，**20 项指标与已发布基线对账，18 项偏差 <1%，中位偏差 0.22%**。

> 两次 decode 数略有不同（§3.7 是 5,160.2，本节是 5,162.4）——
> **是两次独立的 run，不是同一个数写了两遍**。两者相差 0.043%，都在平台期抖动内。

七格的分组关系：

| 组 | 格 | 内容 |
|---|---|---|
| DP8 decode | 3.8.1 | C256/P65536/D1024 |
| DP8 prefill | 3.8.2 / 3.8.3 / 3.8.6 | 8K 吞吐扫描 / 单请求 TTFT / SPEED-Bench |
| PCP8 prefill | 3.8.5 / 3.8.3 / 3.8.6 | 同上三个 workload，换并行策略 |
| 横向 | 3.8.4 / 3.8.7 | 为什么留 PCP8 / 三类负载结论矛盾时信哪个 |

#### 3.8.1 DP8 C256 decode

这一格回答的是：**8 个 DP rank 全部塞满、上下文拉到 64K 的时候，这台机器每秒能吐多少 token，每个用户感觉有多快。** 它不是「某个延迟约束下的最大吞吐」——并发被钉死在 256，吞吐和 TPOT 是并列报出的两个数（口径见 §3.6）。

**怎么跑**

```bash
cd /ram/tbd && PUBLISH_REPORTS=0 bash scripts/daily_benchmark.sh --only dp-decode
```

`--only dp-decode` 会依次做三件事：`start_dp_decode_server.sh` 起服务 → 一轮 C8/D32 的 smoke → 正式的 C256/D1024 一轮。`--prefill-mode` 对这一组无效（脚本会直接报错拒绝）。

服务起来之后也可以只重跑跑分部分，不动服务：

```bash
.venv/bin/python scripts/bench_decode_sliding_window.py \
  --base-url http://127.0.0.1:18100 \
  --model Qwen3.5-397B-A17B-FP8 \
  --output-dir "$RUN_DIR/results/dp8_decode_c256/run_1" \
  --concurrency 256 --data-parallel-size 8 \
  --prefill-tokens 65536 --decode-tokens 1024 \
  --tokenizer-dir /ram/tbd/models/Qwen3.5-397B-A17B-FP8 \
  --rounds 1 --window-seconds 10 --step-seconds 1 \
  --cache-salt-prefix "tpu-daily-$(date -u +%Y%m%dT%H%M%SZ)-run1"
```

> ⚠️ **`--cache-salt-prefix` 不要省。** 省了它脚本会用当前时间戳兜底，单次跑没问题，但你**没法复现同一批 prompt**。这一项和 `--no-enable-prefix-caching` 一起，是「测算力不测缓存命中率」的两道锁。

**负载与口径**

| 项 | 值 |
|---|---|
| 服务并行 | TP=1 / DP=8 / EP=8，`--max-num-seqs 32`（8 × 32 = 并发 256，正好填满） |
| 请求形状 | ISL 65,536 / OSL 1,024，`--kv-cache-dtype fp8`，`--seed 42` |
| 样本数 | 256 条请求 × 1 轮 = 262,144 个输出 token |
| 客户端 | 单进程并发提交，无 admission barrier；`request_id % 8` 路由到 DP rank |
| 统计窗口 | 10 s 滑窗、1 s 步进，**只保留活跃请求数达到本轮峰值的完整窗口** |
| 主/副指标 | `throughput_p50`（窗口吞吐中位数）/ `peak_active_tpot_p50_ms` |

**期望看到什么**

smoke 那一轮**必然是 `valid=False`**，别当故障：

```
[prompt] mode=unique_natural_language_prefix source_tokens=76 request_tokens=65536 unique_requests=8
[decode] round=1/1 C=8 P=65536 D=32
[round-summary] valid=False peak_active=0 windows=0 throughput_p50=None peak_active_tpot_p50_ms=None
[done] no complete sliding window; output=.../results/dp8_decode_c256/smoke
```

`invalid_reason` 会写在 `smoke/summary.json` 里，是 `decode_span_shorter_than_window` —— 只解码 32 个 token，撑不满一个 10 s 窗口，**设计如此**。正式那轮才是要看的：

```
[decode] round=1/1 C=256 P=65536 D=1024
[round-summary] valid=True peak_active=216 windows=16 throughput_p50=5162.4 peak_active_tpot_p50_ms=41.835736999928486
[done] throughput_p50=5162.400 tok/s peak_active_tpot_p50_avg=41.836 ms output=.../results/dp8_decode_c256/run_1
```

服务侧对照：`Loading weights took 38.22 seconds`（tmpfs 判据，见 §3.5）。

**实测值 vs 发布基线**

| 指标 | 本次实测 | 发布基线 | 偏差 |
|---|---|---|---|
| `throughput_p50`（tok/s） | **5,162.4** | 5,162.4 | **0.00%** |
| `peak_active_tpot_p50`（ms） | **41.836** | 41.76 | **+0.18%** |
| TPOT p90 / p99（ms） | 48.918 / 53.345 | — | — |
| 窗口数 / peak active | 16 / 216 | — | — |
| `end_to_end_tok_s` | 637.2 | — | — |
| 失败请求 | 0 / 256 | — | — |

16 个窗口的吞吐 min/avg/max = 5,119.2 / 5,160.4 / 5,211.1，stddev 29.6 → **CV 0.57%**。平台期这么平，`--rounds 1` 就够，不必为了「多跑几轮取平均」再烧一次编译。

**读数要点**

1. **`throughput_p50` 和 `tpot_p50` 不是两个独立测量。** 216 个活跃请求 ÷ 41.836 ms/token = **5,163.0 tok/s**，实测 5,162.4，差 **0.012%**。它们是同一段滑窗里同一个现象的两种表述 —— 两个数同时对上基线，**只算一份证据不算两份**。反过来这也是个便宜的自检：这两个数除不出 peak_active，说明取数取错了段。

2. **`end_to_end_tok_s` 只有 637，和 peak-active 的 5,162 差 8.10 倍 —— 这正是要用滑窗的理由。** 262,144 token ÷ 637.2 = 端到端 411.4 s，而 `first_token_skew_s` 是 **370.4 s，占了 90.0%**。256 条 64K prefill 要排着队进，端到端口径把这段排队时间当成「解码慢」摊了进去。**报端到端平均等于报调度队列长度，不是报解码能力。**

3. **`peak_active=216` 而不是 256，是真实上限不是采样误差。** TTFT 从 min 11.0 s 铺到 max 380.1 s，最早那批请求的 1,024 个 token 已经吐完，最晚那批还没出首 token —— **256 条从来没有同时在解码过**。平台期总共只有 25.0 s（首窗起于 batch 后 298.8 s，末窗止于 323.8 s），16 个窗口就是它的全部宽度。要做真正的 C256 满并发稳态，得加 admission barrier 或者把 OSL 拉长，那是另一个实验。

---

#### 3.8.2 DP8 prefill · 8K 吞吐扫描

这一格回答的是：**固定 8K 输入、只要 1 个输出 token 的纯 prefill 场景下，吞吐的天花板在哪、拐点落在哪个并发。** 和 3.8.1 互补 —— 那格测「吐字」，这格测「吃字」。

**怎么跑**

```bash
cd /ram/tbd && PUBLISH_REPORTS=0 bash scripts/daily_benchmark.sh \
  --only dp-prefill --prefill-mode throughput --prefill-workload synthetic
```

不加 `--prefill-mode` / `--prefill-workload` 的话，`dp-prefill` 还会带上 8K–252K 单请求 TTFT 扫描和 SPEED-Bench 变长负载，时间翻几倍。

服务已经在跑时，直接调底层脚本（并发列表可覆盖）：

```bash
BENCHMARK_CONFIG=dp8 INPUT_LEN=8192 OUTPUT_LEN=1 \
CONCURRENCIES="8 16 32 64 128 256" \
PUBLISH_REPORTS=0 UPDATE_REPORTS=0 \
bash scripts/bench_all.sh "$RUN_DIR"
```

`bench_all.sh` 对每个并发调一次 `vllm bench serve`，跑完自己聚合出 `results/dp8/summary.json`。

**负载与口径**

| 项 | 值 |
|---|---|
| 服务并行 | TP=1 / DP=8 / EP=8 / PCP=1，`--max-num-seqs 64`、`--max-num-batched-tokens 4096` |
| 服务其它 | `--max-model-len 262144`、`--gpu-memory-utilization 0.90`、`--kv-cache-dtype fp8`、`--no-enable-prefix-caching` |
| 请求形状 | `--dataset-name random`、ISL 8,192（`--random-range-ratio 0`，定长）、OSL 1、`--ignore-eos` |
| 样本数 | **每个并发点 512 条**，`--request-rate inf`（一次性压进去，靠 `--max-concurrency` 限流） |
| 确定性 | `--temperature 0 --seed 42` |
| 并发档 | 8 / 16 / 32 / 64 / 128 / 256（`CONCURRENCIES` 默认值） |

> ⚠️ **`--temperature 0` 是硬要求，不是习惯。** 压测里放开采样温度会让结果不可比（尤其是开了投机解码之后，接受率会崩），这条在 §4.8 里同样适用。

**期望看到什么**

每个并发点一段横幅 + 一块 vLLM 标准结果：

```
=====================================================================
Running benchmark for concurrency: 32
=====================================================================
...
============ Serving Benchmark Result ============
Successful requests:                     512
Failed requests:                         0
Maximum request concurrency:             32
Benchmark duration (s):                  72.66
Total input tokens:                      4194304
Total generated tokens:                  512
Total token throughput (tok/s):          57735.75
---------------Time to First Token----------------
Median TTFT (ms):                        4450.18
P99 TTFT (ms):                           6691.86
==================================================
```

跑完 6 段之后，聚合脚本打这一行 —— **这才是要和基线对账的数**：

```
Highest total token throughput: 57735.75 tok/s (concurrency=32)
```

服务侧对照：`Loading weights took 40.61 seconds`。六个并发点的 `Benchmark duration` 加起来 512.5 s（约 8 分 32 秒），编译和起服务的时间不算在内。

**实测值 vs 发布基线**

| 并发 | 总吞吐 (tok/s) | 相对峰值 | median TTFT | P99 TTFT | 用时 (s) |
|---:|---:|---:|---:|---:|---:|
| 8 | 31,304.3 | −45.78% | 1.59 s | 4.16 s | 134.00 |
| 16 | 49,319.9 | −14.58% | 2.20 s | 5.29 s | 85.05 |
| **32** | **57,735.8** | **峰值** | 4.45 s | 6.69 s | 72.66 |
| 64 | 57,222.2 | −0.89% | 8.91 s | 11.12 s | 73.31 |
| 128 | 57,614.7 | −0.21% | 17.81 s | 20.07 s | 72.81 |
| 256 | 56,210.1 | −2.64% | 34.54 s | 38.48 s | 74.63 |

| 指标 | 本次实测 | 发布基线 | 偏差 |
|---|---|---|---|
| DP8 峰值吞吐 (tok/s) | **57,735.75** | 57,682.03 | **+0.09%** |
| 峰值所在并发 | 32 | 64 | 见下 |
| 失败请求 | 0 / 3,072（6 × 512） | — | — |

**读数要点**

1. **拐点在 C32，之后是纯排队。** C8→C16 吞吐涨 57.55%，C16→C32 涨 17.06%，C32→C64 就变成 −0.89% 了。而从 C32 到 C256，吞吐只掉 2.64%，median TTFT 却涨了 **7.76 倍**（4.45 s → 34.54 s），**512 条跑完的总时长几乎不动**（72.66 s → 74.63 s）。多出来的并发一个 token 都没多算，全变成了队列里的等待。**要画 SLA 曲线，拐点就在 C32 附近取，往上加并发只买延迟。**

2. **峰值从基线的 C64 挪到了 C32，这不是回归。** C32 与 C64 只差 **0.89%**，C128 更是只差 0.21% —— 三个点在同一个平台期里，峰值落在哪一格是抖动决定的。**该对账的是峰值本身（+0.09%），不是峰值的位置。** 拿「峰值并发变了」当结论，会把噪声报成性能事件。

3. **瓶颈不在请求槽位，也没被 decode 污染。** DP8 × `--max-num-seqs 64` = 512 个槽，C256 都填不满，所以饱和是算力侧的、不是准入侧的。另一头，OSL=1 加 `--ignore-eos` 让每条请求出完首 token 就结束，实测 `mean_e2el − mean_ttft < 0.001 ms`（C32 上是 4378.9666 vs 4378.9664）——**这一格测到的确实是纯 prefill，没有一步 decode 混进来**。这也是它能和 3.8.1 分开读的前提。

---

#### 3.8.3 单请求 TTFT 扫描 · DP8 vs PCP8

**这一格测的是「一条请求单独跑，第一个 token 要等多久」** —— 并发钉死在 1，OSL 也是 1，
所以整条链路上只剩 prefill，没有排队、没有 batching、没有 decode 混进来。
DP8 和 PCP8 放在一起看才有意义：**同样 4 chip / 8 device，只换并行策略**，
差出来的就是策略本身对长上下文延迟的影响。

**怎么跑**

两个配置各起一次服务、各扫一遍，服务不能同时开（同一批 device）。

```bash
# 服务端：--config 选并行策略，两者 --max-model-len 都是 262144
setsid nohup bash scripts/start_prefill_server.sh --config dp8  > /ram/dp8-server.log  2>&1 &
# 换配置时先停掉上面这个，再起：
setsid nohup bash scripts/start_prefill_server.sh --config pcp8 > /ram/pcp8-server.log 2>&1 &

# 客户端：BENCHMARK_CONFIG 必须和服务端 --config 一致，位置参数是 RUN_DIR
BENCHMARK_CONFIG=dp8  bash scripts/bench_prefill_ttft.sh "$RUN_DIR"
BENCHMARK_CONFIG=pcp8 bash scripts/bench_prefill_ttft.sh "$RUN_DIR"
```

结果落在 `$RUN_DIR/results/<config>/single_request_ttft/`，
每个输入长度一个 JSON（含 `raw_ttft_ms` 逐条原始值），外加一个 `summary.json`。

`--config` 拨动的不只是 DP/PCP 两个数，下面这几项是跟着一起变的：

| 项 | DP8 | PCP8 |
|---|---|---|
| 并行 | DP=8 / PCP=1 / TP=1 | DP=1 / **PCP=8** / TP=1 |
| `--max-num-batched-tokens` | 4096 | **32768** |
| `--long-prefill-token-threshold` | 不设 | **32768** |
| `TPU_MOE_SKIP_PADDED_TOKENS` | 1 | **0** |
| 两边相同 | `--max-model-len 262144`、`--max-num-seqs 64`、`compile_sizes 512,1024,2048,4096`、`--kv-cache-dtype fp8`、EP 开、prefix caching 关 | |

> ⚠️ **DP8 的客户端会额外带 `--header X-data-parallel-rank=0`**，PCP8 不带。
> 这是为了把单请求钉在同一个 DP rank 上；少了它，8 个 rank 轮着接，测出来是路由抖动不是 prefill 延迟。

**负载与口径**

| 项 | 值 | 说明 |
|---|---|---|
| 并发 | **1** | `--max-concurrency 1`，串行发，一条完了才发下一条 |
| OSL | **1** | 只要第一个 token，`--ignore-eos` |
| ISL | 8K / 16K / 32K / 64K / 128K / 252K | 六档，252K = 258,048 token |
| 采样数 | 8K/16K/32K 各 **16** 条；64K/128K/252K 各 **4** 条 | 长输入太贵，减采样 |
| 数据集 | `random`，`--random-range-ratio 0` | 长度精确，不浮动 |
| 确定性 | `--temperature 0`、`--seed 42`、prefix caching 关 | |
| 统计量 | **median TTFT** | 不用 mean，理由见下 |

**实测值**

| 输入长度 | DP8 (ms) | PCP8 (ms) | PCP8 领先 | DP8 偏差 | PCP8 偏差 |
|---|---|---|---|---|---|
| 8,192 (8K) | 1,001.8 | **218.6** | 4.58× | −0.30% | +0.32% |
| 16,384 (16K) | 2,068.9 | **408.8** | 5.06× | −0.22% | +0.12% |
| 32,768 (32K) | 4,393.0 | **801.2** | 5.48× | −0.22% | −0.21% |
| 65,536 (64K) | 9,868.9 | **1,727.8** | 5.71× | −0.42% | +0.29% |
| 131,072 (128K) | 24,134.6 | **3,973.8** | 6.07× | −0.04% | −0.26% |
| 258,048 (252K) | 64,076.9 | **9,777.4** | 6.55× | +0.05% | −0.30% |

偏差 = （本次 median − 已发布基线 median）÷ 基线 median。基线为 `0be027b9` / 2026-08-26 11:33。

**读数要点**

- **12 个点全部落在 ±0.42% 以内**，最大偏差是 DP8 64K 的 −0.42%，且正负都有、没有系统性方向。
  本次引擎 commit（`4ae9f631`）比基线新，prefill 路径同样**无回归**。
- **六档全部 `completed` 满、`failed_input_lengths` 为空。** 252K 是 `--max-model-len 262144`
  下能塞进的最长档，两个配置都过了，说明 262144 这个上限不是纸面值。
- **口径用 median 不是 mean，是有原因的。** PCP8 8K 那 16 条里有一条 415.2 ms
  （其余 15 条都在 209–222 ms），单这一条就把 mean 抬到 228.9 ms、p99 抬到 386.1 ms，
  而 median 仍是 218.6 ms。**首条请求的冷路径开销是这一档唯一的噪声源**，
  长输入档（64K 以上，4 条采样）反而干净：252K 的 p99 和 median 只差 0.05%。
- **DP8 这一侧几乎没有噪声。** 8K 档 16 条的 p99/median = 1,012.7/1,001.8 = +1.1%，
  252K 档 +0.13%。单请求 prefill 在 DP8 上是纯计算，抖动本来就小。

---

#### 3.8.4 为什么要留 PCP8 这条并行策略

**领先倍数不是常数，是随输入长度单调上升的。** 把 §3.8.3 那一列单独拎出来：

| 输入长度 | 8K | 16K | 32K | 64K | 128K | 252K |
|---|---|---|---|---|---|---|
| PCP8 领先 | 4.58× | 5.06× | 5.48× | 5.71× | 6.07× | 6.55× |

六个点没有一次回头。这说明 PCP 不是「整体快一个常数倍」的实现优化，
而是**把 prefill 里随长度增长最快的那部分开销切开了** —— 长度越长，切开的收益越大。

**两条曲线都超线性，但陡峭程度差一倍。** 推导链：

- 长度从 8,192 涨到 258,048 是 **31.5×**（258048 ÷ 8192）
- DP8 时间从 1,001.8 ms 涨到 64,076.9 ms，是 **64.0×**（64076.9 ÷ 1001.8）
- PCP8 时间从 218.6 ms 涨到 9,777.4 ms，是 **44.7×**（9777.4 ÷ 218.6）

31.5 倍长度换来 64 倍 / 44.7 倍时间 —— 两者都超线性（attention 的 O(n²) 项在起作用），
但 DP8 的超线性程度是 64.0/31.5 = **2.03**，PCP8 是 44.7/31.5 = **1.42**。

**换成「每 1K 输入 token 的 TTFT 毫秒数」更直观**（= median TTFT ÷ 输入长度 × 1000）：

| 输入长度 | DP8 (ms/1K tok) | PCP8 (ms/1K tok) |
|---|---|---|
| 8K | 122.3 | 26.7 |
| 16K | 126.3 | 24.9 |
| 32K | 134.1 | **24.4** |
| 64K | 150.6 | 26.4 |
| 128K | 184.1 | 30.3 |
| 252K | **248.3** | 37.9 |

这张表比绝对值说明得多：

- **DP8 单调劣化，8K→252K 单位成本涨了 2.03 倍**（248.3 ÷ 122.3）。
  没有平台期，越长越亏，而且亏得越来越快。
- **PCP8 是个 U 形**：8K→32K 先降（26.7 → 24.4，固定开销被摊薄），
  32K 之后才回升到 37.9。整段只涨 1.42 倍（37.9 ÷ 26.7），
  而且**最优点落在 32K** —— 这不巧合，PCP8 的 `--long-prefill-token-threshold` 正是 32768。
- **同一条 TTFT 预算下能吃的上下文差 4 倍**：以 10 秒为线，
  DP8 撑到 64K（9.87 s）就到顶，128K 已经 24.1 s；PCP8 到 252K（9.78 s）还在线内。
  258,048 ÷ 65,536 = **3.94 倍上下文**，同样的等待时间。

**但 PCP8 不是全面更好，这是分工不是优劣。** 看 §3.8.2 的 8K prefill 吞吐扫描：

- 饱和后 **DP8 峰值 57,735.8 tok/s（并发 32），PCP8 只有 54,350.3 tok/s**，DP8 高 **6.2%**
- 反过来在**低并发** 8 时，PCP8 是 48,203.6、DP8 只有 31,304.3，PCP8 高 **54.0%**
- PCP8 从并发 16 起就压在 54,3xx 不动（16/32/64/128/256 五档差不到 0.13%），
  **它是被自己的并行度封顶的**；DP8 要爬到并发 32 才见顶，但顶更高

一句话：**DP8 用并发换吞吐，PCP8 用并行度换单请求延迟。** 打满之后 DP8 的天花板更高。

**落到工程决策，判据是业务输入长度的分布，不是拍脑袋：**

| 业务形态 | 选 | 依据 |
|---|---|---|
| 短 prompt、高 QPS（对话、分类、改写，ISL ≲ 16K） | **DP8** | 峰值吞吐高 6.2%；8K 时 TTFT 差距只有 4.58×，而 1.0 s 本来就在可接受区间 |
| 长文档、低并发（整库问答、长代码库、报告分析，ISL ≳ 64K） | **PCP8** | 64K 起单请求 TTFT 领先 5.7× 以上，且 DP8 在 128K 已破 24 s |
| 有硬 TTFT SLA 的长上下文 | **PCP8** | 10 s 预算下可用上下文从 64K 拉到 252K |
| 流量稀疏 / 突发（长期低并发） | **PCP8** | 并发 8 时吞吐反而高 54% —— DP8 只在饱和态才赢 |

**决策前要先量的是分布不是均值。** 混合流量里若 P50 是 4K 而 P95 是 128K，
按均值挑会两头不讨好 —— 那种情况下正解是两套服务分流，而不是在单一配置里折中。
这也是这两条策略都要留在 daily benchmark 里的理由：**它们回归的是两件不同的事**，
少测一条，另一条的退化就没有参照物。

---

#### 3.8.5 PCP8 prefill · 8K 吞吐扫描

这一格测的是：**输入长度钉死在 8192、只让并发变**，PCP8（DP=1 / PCP=8）这套「把一条请求的 context 切给 8 个 device」的并行策略，总吞吐能爬到哪儿、爬到之后还动不动。

**怎么跑**

```bash
cd /ram/tbd && PUBLISH_REPORTS=0 bash scripts/daily_benchmark.sh \
  --only pcp-prefill --prefill-workload synthetic --prefill-mode throughput
```

并发档位默认 `8 16 32 64 128 256`，来自 `bench_all.sh` 里的 `CONCURRENCIES`，
要改就在命令前加 `CONCURRENCIES="8 32 128"`。
原始结果落在 `results/pcp8/vllm_pcp8_tp1_len8192_c*.json`，聚合在同目录 `summary.json`。

**负载与口径**

| 项 | 值 | 说明 |
|---|---|---|
| 并行 | TP=1 / **DP=1 / PCP=8** | 一份模型铺满 8 个 device，context 维度切分 |
| 负载 | ISL **8192** / OSL **1** | `--dataset-name random --random-range-ratio 0`，512 条 prompt |
| `--max-num-batched-tokens` | **32768** | DP8 那边是 4096，差 8 倍 |
| `--max-num-seqs` / `--max-model-len` | 64 / 262144 | |
| `--long-prefill-token-threshold` | 32768 | PCP8 独有，DP8 不设 |
| 其他 | `--no-enable-prefix-caching`、`--kv-cache-dtype fp8`、`--gpu-memory-utilization 0.90`、`temperature 0`、`--seed 42` | 前缀缓存**关掉**，每次真算 |

> ⚠️ **`total_token_throughput` 含那 1 个输出 token。** 512 条 × (8192+1) = 4,194,816，
> 除以 duration 才对得上表里的数（C32：4,194,816 / 77.181 s = 54,350.29）。
> 拿它当「纯 prefill 吞吐」用没问题，误差是 1/8193 = 0.012%。

**实测值**

| 并发 | 总吞吐 tok/s | 中位 TTFT (ms) | P99 TTFT (ms) | 耗时 (s) |
|---:|---:|---:|---:|---:|
| 8 | 48,203.62 | 1,345.51 | 1,658.26 | 87.02 |
| 16 | 54,349.28 | 2,404.70 | 2,418.13 | 77.18 |
| **32** | **54,350.29** | 4,808.08 | 4,828.34 | 77.18 |
| 64 | 54,330.04 | 9,617.64 | 9,639.25 | 77.21 |
| 128 | 54,282.90 | 19,237.54 | 19,258.56 | 77.28 |
| 256 | 54,303.58 | 38,441.92 | 38,491.09 | 77.25 |

512/512 完成，0 失败。与已发布基线对账：

| 指标 | 本次 | 发布基线（`0be027b9` / 2026-08-26 11:33） | 偏差 |
|---|---:|---:|---:|
| 峰值总吞吐 | **54,350.29 tok/s** | 54,330.21 | **+0.04%** |
| 峰值所在并发 | 32 | 16 | 见下 |

**读数要点**

- **C16 之后就是一条直线。** C16–C256 五个点的极差只有 **67.39 tok/s = 0.12%**，
  而 C16 与 C32 之间只差 **1.01 tok/s（0.002%）**。
  → **「峰值出现在哪个并发」这件事本身是噪声决定的**，基线报 C16、我们报 C32，
  两个数差 0.04%，没有任何物理含义。引用时请引「平台值约 54.3K」，不要引 argmax。
- **C8 是唯一一个没打满的点**（48,203.62，比平台低 11.3%），
  说明 PCP8 只需要 **1–2 条并发请求**就能把硬件填满 —— 依据见 3.8.7 的反算。
- **TTFT 从 C16 起严格随并发线性翻倍**（2,405 → 4,808 → 9,618 → 19,238 → 38,442 ms，
  每档 ×2.00）。吞吐不涨、延迟等比例涨 = **多出来的并发全在排队**。
  这条对齐关系也反过来验证了平台是真饱和，不是测量抖动。
- P99 与中位数几乎重合（C32：4,828 vs 4,808，差 0.4%），因为所有请求等长、
  调度器按固定批次吃 —— **这是定长负载的产物，换成变长立刻不成立**（见 3.8.6）。

---

#### 3.8.6 SPEED-Bench 变长负载 · DP8 vs PCP8

这一格测的是：**换成长度参差不齐的真实请求**（756–37,719 token），DP8 和 PCP8
谁更能扛。定长 8K 那格是同构负载下的上限，这格才是接近线上的形状。

**怎么跑**

```bash
cd /ram/tbd && PUBLISH_REPORTS=0 bash scripts/daily_benchmark.sh \
  --only dp-prefill  --prefill-workload speed-bench
cd /ram/tbd && PUBLISH_REPORTS=0 bash scripts/daily_benchmark.sh \
  --only pcp-prefill --prefill-workload speed-bench
```

两组分别起自己的 server，串行跑。并发档默认 `8 64`（`SPEED_BENCH_CONCURRENCIES`）。
结果落 `results/{dp8,pcp8}/speed_bench_mix/throughput_c{8,64}.json`。

**负载与口径**

| 项 | 值 |
|---|---|
| 数据集 | 公开 **NVIDIA SPEED-Bench** 快照（revision `487aa718…`），子集 `throughput_{1k,2k,8k,16k,32k}` |
| 抽样 | 清洗去重后的 4,194 条里，**seed 42 全局均匀抽 1,000 条**（`prepare_speed_bench_mix.py --random-sample-total`） |
| 清洗 | 丢 placeholder 行 → 去掉 `Answer now please.` 人为填充 → 按 cleaned prompt 的 SHA-256 全局去重 |
| 输入长度 | **756 – 37,719 token**，合计 10,963,710（均值 **10,963.71**），dataset SHA-256 `f16a7f760630…` |
| 请求参数 | OSL **1**、`--request-rate inf`、`--ignore-eos`、`temperature 0`、`--seed 42`、`--no-oversample` |
| 服务端 | 与 3.8.5 同一套 `start_prefill_server.sh`，只换 `--config dp8` / `--config pcp8` |

> ⚠️ **跑测前有一条 warm-up 请求**（`--num-prompts 1 --max-concurrency 1`），
> 它**不进**任何统计。看日志时别把它当成第一条测量请求。

> 🔎 **数据集完整性是硬校验，不是提示。** `bench_speed_bench_mix.sh` 会先比
> artifact SHA-256、解压后再比内容 SHA-256，任一不符**直接 exit 1**。
> 换句话说：只要这一步跑过去了，你和基线吃的就是**同一批 1,000 条请求**。

**实测值：input tok/s**

| 配置 | 并发 | input tok/s | 发布基线 | 偏差 |
|---|---:|---:|---:|---:|
| DP8 | 8 | 30,765.59 | 30,025.41 | **+2.47%** |
| DP8 | 64 | 51,367.80 | 51,941.43 | **−1.10%** |
| PCP8 | 8 | **44,259.13** | 44,300.85 | **−0.09%** |
| PCP8 | 64 | 48,370.81 | 48,251.88 | **+0.25%** |

1000/1000 完成，0 失败，四格全部对上账。

**同一批请求的 TTFT 分布**

| 配置 | 并发 | P50 (ms) | P90 (ms) | P99 (ms) | 耗时 (s) |
|---|---:|---:|---:|---:|---:|
| DP8 | 8 | 2,241.42 | 5,699.80 | 7,406.35 | 356.36 |
| PCP8 | 8 | **1,903.89** | **2,740.91** | **3,500.92** | **247.72** |
| DP8 | 64 | **12,713.59** | 19,632.46 | 25,038.05 | 213.44 |
| PCP8 | 64 | 14,228.92 | **16,829.59** | **17,707.52** | 226.66 |

**读数要点**

- **C8：PCP8 完胜，1.44 倍**（44,259.13 / 30,765.59），跑完同样 1,000 条请求少花 108.6 秒。
- **C64：反过来，DP8 领先 6.20%**（51,367.80 / 48,370.81）。
  **交叉点落在 C8 与 C64 之间，本次没测出来** —— 只有两个并发档，中间是空的。
  要定位交叉点得自己补 C16/C32：`SPEED_BENCH_CONCURRENCIES="8 16 32 64"`。
- **尾延迟的排序和吞吐排序不一致，别混着说。** C64 时 DP8 的 P50 更好（12,714 vs 14,229），
  但 P99 差得多（25,038 vs 17,708）。用 P99/P50 当离散度：
  DP8 C8 = 3.30、C64 = 1.97；PCP8 C8 = **1.84**、C64 = **1.24**。
  **PCP8 在变长负载下的尾部稳定性是它真正的卖点**，不是峰值吞吐。
- **变长负载吃不到定长峰值。** C64 时 DP8 51,368 比它自己定长 8K 的峰值 57,736 低 **11.03%**，
  PCP8 48,371 比 54,350 低 **11.00%**。两边掉的比例几乎一样 ——
  这是**长度参差导致的批次填充损失**，不是某一方的策略问题。
- **DP8 C8 那个 +2.47% 不用紧张。** 已发布历史里 DP8 C8 在 28,037.09 – 30,269.78 之间波动，
  极差 7.96%（推导：(30269.78−28037.09)/28037.09）。本次偏差落在这条带内。
  低并发 + 长度差异大 = 样本级噪声本来就比 C64 大。

---

#### 3.8.7 三类 prefill 负载给出的结论不一致 —— 该信哪个

**先把矛盾摆出来。** 同一天、同一套硬件、同一个 commit，三组测量对「DP8 和 PCP8 谁强」给出了三个不同答案：

| 负载 | DP8 | PCP8 | 谁赢 |
|---|---:|---:|---|
| 定长 8K 吞吐扫描（峰值） | **57,735.75 tok/s** | 54,350.29 tok/s | **DP8**，+6.23% |
| 单请求 TTFT（并发 1，8K–252K） | 1,001.76 – 64,076.85 ms | **218.62 – 9,777.39 ms** | **PCP8**，全线快 4.58–6.55 倍 |
| SPEED-Bench 变长 C8 | 30,765.59 tok/s | **44,259.13 tok/s** | **PCP8**，+43.86% |
| SPEED-Bench 变长 C64 | **51,367.80 tok/s** | 48,370.81 tok/s | **DP8**，+6.20% |

这不是哪一组测错了。**四行全部是对的**，它们只是问了四个不同的问题。

**根因：两条曲线的形状完全不同**

把定长 8K 那格的六个点并排放：

| 并发 | DP8 tok/s | PCP8 tok/s | DP8 / PCP8 |
|---:|---:|---:|---:|
| 8 | 31,304.28 | 48,203.62 | 0.65 |
| 16 | 49,319.86 | 54,349.28 | 0.91 |
| 32 | **57,735.75** | **54,350.29** | 1.06 |
| 64 | 57,222.23 | 54,330.04 | 1.05 |
| 128 | 57,614.67 | 54,282.90 | 1.06 |
| 256 | 56,210.10 | 54,303.58 | 1.04 |

- **PCP8 从 C16 起就是一条直线**：54,349 / 54,350 / 54,330 / 54,283 / 54,304，极差 0.12%。
- **DP8 从 C8 一路爬到 C32**，涨了 **1.84 倍**（31,304 → 57,736），C32 之后才转平（C32–C256 极差 2.71%）。
- **定长 8K 上的交叉点在 C16 与 C32 之间**：C16 时 DP8 49,320 < PCP8 54,349；C32 时 DP8 57,736 > PCP8 54,350。

**验证：为什么 PCP8 在 C8 就已经打满**

用单请求 TTFT 反算「一条请求能吃掉多少硬件」，推导链如下：

- PCP8 8K 单请求 median TTFT = **218.62 ms** → 8192 tok ÷ 0.21862 s = **37,472 tok/s**，
  = 它自己平台值 54,349 的 **68.9%**。**一条请求就吃掉了近七成硬件。**
- DP8 8K 单请求 median TTFT = **1,001.76 ms** → 8192 ÷ 1.00176 = **8,178 tok/s**，
  = 它自己峰值 57,736 的 **14.2%**。
- 填满各自需要的并发数：DP8 = 57,736 / 8,178 = **7.06**（≈ 8 个 DP rank 各站一条）；
  PCP8 = 54,349 / 37,472 = **1.45**。

这解释了全部四行：**PCP8 把一条请求内部拆开并行，DP8 靠堆请求数把 8 个独立 rank 填满。**
所以 PCP8 在并发 1 就赢 4.58–6.55 倍，在 C8 还赢 1.44 倍，
一旦并发够 DP8 站满 8 个 rank，DP8 的上限（无跨 device 通信）就反超。

配置侧也能对上：DP8 的 `--max-num-batched-tokens` 是 **4096**，PCP8 是 **32768**（8 倍），
并且只有 PCP8 设 `--long-prefill-token-threshold 32768`。
PCP8 的一个批次就能吞下整条长 prompt，切成 8 份铺给 8 个 device；
DP8 的一条 8K prompt 在单个 rank 上要分两个 chunk 走。

**所以：怎么选**

**决定因素不是「谁的峰值高」，是「你的业务跑在哪个并发上」。**

| 业务形态 | 选 | 依据 |
|---|---|---|
| 交互式 / 低并发（有效并发 ≲ 8）、长输入 | **PCP8** | C8 变长 +43.86%；单请求 TTFT 快 4.58–6.55× |
| 批处理 / 高并发（≳ 32）、输入偏短 | **DP8** | 定长 C32 +6.23%；变长 C64 +6.20% |
| 有 P99 TTFT SLA 的变长在线服务 | **PCP8** | P99/P50 = 1.24–1.84，DP8 是 1.97–3.30 |
| 并发在 8–64 之间 | **先补测再定** | 交叉点落在这一段，本次数据里是空白 |

> ⚠️ **陷阱：只看峰值吞吐一定会选错。**
> 两条曲线的峰值都出现在 **C32**，而 C32 只是扫描档位里的一格，不是任何真实业务的工作点。
> 三个具体的坑：
> 1. **峰值的 argmax 是噪声。** PCP8 的 C16 和 C32 只差 **1.01 tok/s（0.002%）** ——
>    基线报 C16、我们报 C32，这个「峰值位置变了」不含任何信息。
>    DP8 同理：基线峰在 C64，我们在 C32。**引平台值，不要引 argmax。**
> 2. **峰值只在定长负载上成立。** 换成变长请求，两边都掉 **11%**（DP8 −11.03%、PCP8 −11.00%）。
>    拿 57,736 去做容量规划，会系统性高估。
> 3. **按峰值选型会在低并发段付 44% 的代价。** 峰值说 DP8 赢 6.23%，
>    但如果业务实际跑在 C8，DP8 只有 30,766，PCP8 有 44,259 —— 选反了亏 30.5%（44,259 → 30,766）。
>
> **正确做法：先测出自己业务的稳态有效并发，再回这三张表查对应的那一格。**

---

## 4. 路 B：16 chip 18 格性能矩阵

| 产出 | 3 并行策略 × 2 序列形状 × 3 并发 = **18 格** |
|---|---|
| 硬件 | 4 节点 × `tpu7x-standard-4t` = 16 chip / 32 device，拓扑 `2x2x4` |
| 耗时 | **约 2 小时 20 分** = 准备 30 分 + 跑测 90 分（4 节点并行）+ 收数 20 分 |
| 前提 | pod 不受 60 分钟寿命限制（共享集群请改用 QUICKSTART） |
| 镜像 | `us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu:nightly-20260726` |

### 4.1 预热镜像（零风险，务必在换票之前做）

拉镜像不需要 TPU 资源，可与占位 Job 共存。**这一步提前消化掉切换时最大的失败模式。**

```bash
for N in $(kubectl get nodes -l cloud.google.com/gke-nodepool=$POOL -o name | cut -d/ -f2); do
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: {name: prepull-${N##*-}, namespace: $NS}
spec:
  restartPolicy: Never
  nodeName: $N
  containers: [{name: c, image: $IMG, command: ["bash","-c","echo PREPULL_OK; sleep 20"],
                resources: {requests: {cpu: 100m, memory: 512Mi}}}]
  tolerations: [{operator: Exists}]
EOF
done
```

期望约 2 分钟后全部 `Completed`。若 `ImagePullBackOff` → 权限问题，**先解决，不要动占位**。

### 4.2 三层校验 JobSet，然后换票

```bash
cd CHANGE_ME_repo/tpu/vllm-torchtpu   # 先进到本仓库这个目录
export RB=$PWD                        # 后面所有脚本路径都基于它
[ -d "$RB/Qwen3.5-397B-A17B-FP8/scripts" ] && echo "✓ RB=$RB" || echo "✗ 目录不对"

cp $RB/Qwen3.5-397B-A17B-FP8/manifests/v7x-16chip-jobset.yaml jobset.yaml
sed -i "s#hy3-v7-16-dws-2#$POOL#" jobset.yaml     # 换成你的节点池名
python3 $RB/Qwen3.5-397B-A17B-FP8/scripts/validate-jobset.py jobset.yaml --context $CTX
# 期望：三层全过，可以提交
```

JobSet 里**不能改**的地方，除了 §2 那张 pod 配置表（`resources` 只写一份、
`/tmp` 和 `~/.cache` 挂 tmpfs、`/dev/shm` ≥ 64Gi）之外，还有一条：

| | 值 | 为什么 |
|---|---|---|
| `failurePolicy.maxRestarts` | **3** | TPU `SLICE_FAILURE` 反复重试会耗尽配额，**DWS 节点被整批回收**。别设 10 |

> ⚠️ **第 3 层不能省。** JobSet 的 `--dry-run=server` **只校验 JobSet CRD 本身，
> 不校验它生成的 Job**。真出问题时 dry-run 一路绿灯，真因只在
> `kubectl logs -n jobset-system -l control-plane=controller-manager` 里。
> 实测这个疏漏废掉过一整夜 9 个任务。

**换票**（先读 §1.6 的三条铁律，anchor pod 必须已经钉上）：

```bash
kubectl --context=$CTX get jobs -n $NS          # 找占位 Job 名。删 pod 没用，会被重建
export PLACEHOLDER_JOB="CHANGE_ME_job_name"

kubectl --context=$CTX delete job "$PLACEHOLDER_JOB" -n $NS --wait=false; \
kubectl --context=$CTX apply -f jobset.yaml
```

> ⚠️ 两条命令必须在同一行、用 `;` 连接，中间**不要**加 `sleep` 或判断。实测切换 2 秒。

```bash
export PODS=$(kubectl --context=$CTX get pods -n $NS -o name | grep ttpu16 | cut -d/ -f2)
export P0=$(echo $PODS | awk '{print $1}')
kubectl --context=$CTX get pods -n $NS -o wide | grep ttpu16
```

期望约 10 秒后 4 个 `Running`，`NODE` 列互不相同（镜像已预热所以这么快）。
**节点空闲、无占位时**：跳过 §4.1 和 delete，直接 apply。

### 4.3 确认容器环境

```bash
kubectl exec $P0 -n $NS -- bash -c 'env | grep "^TPU_" | sort; df -h /work /dev/shm /tmp'
```

期望 GKE 已注入多机 mesh：`TPU_ACCELERATOR_TYPE=tpu7x-32`、`TPU_HOST_BOUNDS=1,1,4`、
`TPU_TOPOLOGY=2x2x4`、`TPU_PROCESS_ADDRESSES` 有 4 个地址；`/work` 和 `/tmp` 各 640G、`/dev/shm` 64G。

### 4.4 装环境 + 拉权重（并行）

一个打 pypi、一个打 GCS，不抢同一个瓶颈。实测装环境 2 分 50 秒。

```bash
git clone https://github.com/vllm-project/vllm-torchtpu.git
curl -L -o vllm-src.tgz https://github.com/vllm-project/vllm/archive/refs/tags/v0.26.1rc0.tar.gz
tar czf vtt.tgz --exclude='.git' vllm-torchtpu

D=$RB/Qwen3.5-397B-A17B-FP8/scripts
for P in $PODS; do (
  for F in vtt.tgz vllm-src.tgz; do kubectl cp $F $NS/$P:/work/$F; done
  kubectl cp $D/fetch-weights.py     $NS/$P:/work/fetch-weights.py
  kubectl cp $D/bootstrap-16chip.sh  $NS/$P:/work/bootstrap.sh
  kubectl cp $D/run-config.sh        $NS/$P:/work/run-config.sh
  kubectl exec $P -n $NS -- bash -c 'cd /work && tar xzf vtt.tgz'
) & done; wait
```

**权重走同区 GCS，不走 HuggingFace。** 优先用 §3.2 的 `gcloud storage` 路径
（镜像里有 SDK 的话）；只有镜像没 SDK 时才退回 `fetch-weights.py`。

**完成判据**（4 个都满足才往下走）：

```bash
for P in $PODS; do kubectl exec $P -n $NS -- bash -c \
  'grep -q BOOTSTRAP_OK /work/bootstrap.log && echo "$(hostname) OK $(du -sb /work/models/qwen3.5-397b|cut -f1) bytes $(ls /work/models/qwen3.5-397b/*.safetensors|wc -l) 分片"'; done
```

期望 4 行 `OK`，**字节数完全相同**，分片数都是 94。

> **重跑是安全的。** `fetch-weights.py` 按**字节数**校验而不是只看文件在不在，
> 缺失的和内容损坏的都会被补下，已完整的跳过。
> 实测：删掉 2 个分片 + 把第 3 个截断成 1000 字节后重跑，**41 秒**恢复到逐字节一致。

版本矩阵应为 torch 2.13.0 / jax 0.10.2 / libtpu 0.0.44.1 /
torch-tpu 0.1.1.dev20260804130134 / vllm 0.26.1rc0+tpu，
`platform: TpuPlatform`、`tpu devices: 8`。

> ⚠️ 镜像自带 torch 2.11 / jax 0.9.2 / libtpu 0.0.41，**比 `pyproject.toml` 要求的旧一档，必须升**
> （`jax 0.9.2` 没有 `pallas.tpu.BufferType`，一跑就崩）。
> 且这条路**必须用 `pip` 不能用 `uv`** —— uv 不读 `PIP_INDEX_URL`，会报「找不到 torch-tpu」而包一直都在。
> （路 A 的 `update_environment.sh` 不受影响，它显式给 uv 传了 index 凭据。）

### 4.5 把 16 chip 切成 4 个独立 4 chip 环境

18 格串行要 6 小时，切成 4 份并行**约 90 分钟**。

**原理：这不是虚拟化，是「拒绝组队」。** 硬件本来就是 4 台机器各插各的 4 颗芯片；
「16 卡是一个整体」是 libtpu 启动时按环境变量去跟其他 3 台会合才形成的。改掉变量就不去会合。

| 变量 | GKE 注入 | 改成 |
|---|---|---|
| `TPU_WORKER_ID` | 0/1/2/3 | `0` |
| `TPU_PROCESS_ADDRESSES` | 4 个 DNS 名 `:8471` | `localhost:8471` |
| `TPU_WORKER_HOSTNAMES` | 同上 4 个 | `localhost` |
| **`TPU_HOST_BOUNDS`** | **`1,1,4`** | **`1,1,1`** ← 核心只有这一个 |
| `TPU_CHIPS_PER_HOST_BOUNDS` | `2,2,1` | `2,2,1`（**不变**） |
| `TPU_ACCELERATOR_TYPE` | `tpu7x-32` | `tpu7x-8` |

再加 `unset TPU_MULTIHOST_BACKEND`。三个易错点：
① 地址表和主机名表是一对，**只改一个会对不上**；
② `TPU_CHIPS_PER_HOST_BOUNDS` 不用改，每台本来就是 4 颗；
③ **1 chip = 2 device**，写错报拓扑不匹配，**而错误信息不会提示是单位搞错了**。

```bash
kubectl exec $P0 -n $NS -- bash -c 'cd /tmp
export TPU_WORKER_ID=0 TPU_PROCESS_ADDRESSES=localhost:8471 TPU_WORKER_HOSTNAMES=localhost
export TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_ACCELERATOR_TYPE=tpu7x-8
python3 -c "import jax; print(len(jax.devices()))"'
```

| 返回 | 含义 |
|---|---|
| **8** | 对了 |
| 32 | 变量没生效，仍在多机视图 |
| **卡住不返回** | 还在等另外 3 台会合 —— 地址表没改干净 |

> **代价**：跨机 ICI 链路完全闲置。只适合**单机装得下**的模型 ——
> 397B FP8 权重 378 GiB，单机 768 GiB HBM 装得下。

（`run-config.sh` 的 `single` 模式已内置这组变量，直接用即可。）

#### 4.5.1 也可以两台一组，切成 2 个 8 chip 环境

两台能组 `2x2x2` = **8 chip / 16 device**，2026-08-19 实测通过。

**⚠️ 哪两台能配对，节点标签直接写着 —— 不要按名字猜，也不要自己推 z 坐标。**

```bash
kubectl get nodes -l cloud.google.com/gke-nodepool=$POOL \
  -L cloud.google.com/gke-tpu-partition-2x2x2-id
```

GKE 给每种合法子切片都打了一个 id，同 id 的节点才连得起来：

| 标签 | 含义 |
|---|---|
| `gke-tpu-partition-2x2x1-id` | 每台一个（拆单机用的那种） |
| **`gke-tpu-partition-2x2x2-id`** | **两台共享一个 → 这两台能组 8 chip** |
| `gke-tpu-partition-2x2x4-id` | 4 台同一个（完整切片） |

实测那次的配对是 `b9pb + v0c9` 和 `j4b1 + wrdr` —— **按名字顺序猜会配错**
（`b9pb` 的搭档不是 `j4b1`）。ICI 是物理链路，配错了组不起来。

**跟拆单机相反，这次是「要组队」**：

| 变量 | 4 台整片 | 两台一组 | 拆单机 |
|---|---|---|---|
| `TPU_HOST_BOUNDS` | `1,1,4` | **`1,1,2`** | `1,1,1` |
| `TPU_ACCELERATOR_TYPE` | `tpu7x-32` | **`tpu7x-16`** | `tpu7x-8` |
| `TPU_TOPOLOGY` | `2x2x4` | **`2x2x2`** | `2x2x1` |
| `TPU_CHIPS_PER_HOST_BOUNDS` | `2,2,1` | `2,2,1` | `2,2,1` |
| `TPU_PROCESS_ADDRESSES` | 4 个 | **那两台的真实地址** | `localhost:8471` |
| `TPU_MULTIHOST_BACKEND` | 保留 | **保留** | `unset` |

`TPU_WORKER_ID` 每台不同，**等于自己在地址表里的下标**。
验证不能只看 `len(jax.devices())` —— 那个数在没真连上时也可能对。**要跑一次跨机规约**：

```python
import jax, jax.numpy as jnp
d = jax.devices()
print(len(d), jax.process_count(), jax.process_index())   # 16 2 0/1
x = jnp.ones((len(d), 128))
print(float(jax.jit(lambda a: a.sum())(x)))               # 2048.0 = 16×128
```

> **在别人的占位 pod 上测怎么做到不打扰**：那 16 颗当时全被一个
> `sleep infinity` 的 holder 占着（`ubuntu:24.04`，没有 python）。
> 没有删 pod、没有改 Job —— 直接 `apt install python3 && pip install jax[tpu]` 装进容器，
> 只动容器文件系统，重启即清干净。**DWS 节点最怕的就是被误删后拿不回来。**

### 4.6 分派：按编译缓存亲和性

**同一节点始终跑同一并行策略**，否则吃不到编译缓存，每次重编 19 分钟。

> ⚠️ **自定义 config 必须分发到每个要用它的 pod。** config 存在容器本地的
> `scripts/vllm/benchmarking/configs/`，在 A 节点上生成的 config，B 节点看不见。
> 实测漏了这步，B 节点直接 `RUN_FAILED: 找不到 config`，白等一个窗口。

```bash
run(){ kubectl exec $1 -n $NS -- bash -c \
  "nohup bash -c 'bash /work/run-config.sh $2 single $3 $4; echo CHAIN_DONE' > /work/run.log 2>&1 &"; }
set -- $PODS
run $1 qwen3.5-397b-fp8-tp8-dp1-ep '"1024:8192"' '"64 256 512"'
run $2 qwen3.5-397b-fp8-tp8-dp1-ep '"8192:1024"' '"64 256 512"'
run $3 qwen3.5-397b-fp8-tp2-dp4-ep '""' '""'      # 空 = 跑 config 自带的全部 6 格
run $4 qwen3.5-397b-fp8-tp1-dp8-ep '""' '""'
```

**起跑 3 分钟后做一次健康检查（能省一小时）**：

```bash
for P in $PODS; do kubectl exec $P -n $NS -- bash -c \
  'L=$(ls -t /work/vllm-torchtpu/benchmark_runs/*/server.log|head -1); echo "$(hostname) $(wc -l < $L) 行"
   grep -h "Loading weights took" "$L" 2>/dev/null'; done
```

期望权重加载 40–70 秒区间（tmpfs）。**240 秒左右说明权重落在磁盘不是内存盘**，
回头检查 pod spec 的 volume 配置。

### 4.7 ⚠️ 要开投机解码（MTP）？两个坑

397B checkpoint 里有 3096 个 `mtp.*` 权重，投机解码可用，实测接受率 **93–95%**。

#### 坑一：`/dev/shm` 撑爆

> **这个坑的伪装最强**：报出来的是 **SIGBUS（信号 7）**，栈顶停在 `determine_available_memory`，
> 看起来像 HBM 不够。实际是 tmpfs 写满 —— mmap 时成功、访问时才 fault，看不到干净的 `ENOSPC`。
> **诊断先跑 `df -h /dev/shm`，别急着调 `gpu-memory-utilization`。**

不重启 pod 的修法（重启要重下 378 GiB 权重 + 重编缓存）：

```bash
kubectl exec $P -n $NS -- bash -c '
N=$(pgrep python3 | wc -l)   # 用 wc 不用 pgrep -c：数量为 0 时 pgrep -c 返回码非 0
[ "${N:-0}" -gt 0 ] && { echo "还有 python 在跑"; exit 1; }
rm -rf /dev/shm/torch_tpu_cache && mkdir -p /work/ttc
ln -sfn /work/ttc /dev/shm/torch_tpu_cache      # /work 同样是内存盘但有 640 GiB
df -h /dev/shm'
```

也可以直接 `mount -o remount,size=128G /dev/shm`（privileged pod 里有效，见 §2）。

#### 坑二：编译缓存全命中时 MTP 反而会失败

同一份 serve 配置，跑 4 次的结果完全由缓存状态决定：

| 缓存加载 | 新编图 | 动态形状告警 | 结果 |
|---|---|---|---|
| 8 | 20 | 8 | ✓ 成功 |
| **168** | **0** | 8 | ✗ 失败 |
| 全命中 | 0 | — | ✗ **失败（51 秒即崩）** |
| 0 | 21 | 8 | ✓ 成功（接受率 93.2%） |

报错是 `NotImplementedError: TPU backend: does not support dynamic shape`。

> **别被这个错误名带偏。** 成功的那两次**同样报了 8 次动态形状告警** —— 那不是失败原因。
> 真正的差别是：能重新编译时（新编图 20–21 个），撞到动态形状后会走回退路径重编；
> 而缓存全命中时一个图都不编，**回退路径没机会触发，告警就升级成致命错误**。
>
> **处置：开 MTP 前先清编译缓存** —— `rm -rf $VLLM_CACHE_ROOT/* /work/ttc/*`。

#### MTP 到底值不值（口径对齐后的实测）

同为 tp8-dp1 / ISL 1024 / OSL 1024 / 640 prompts：

| 并发 | 不开 MTP | 开 MTP | 差 |
|---|---|---|---|
| 16 | 1237 tok/s | **1397** | **+12.9%** |
| 64 | **2631** | 2406 | **−8.6%** |

**低并发有收益、高并发反而拖累**，交叉点在并发 16 与 64 之间。
这与 GPU 侧「MTP 是单个最大吞吐杠杆」的结论**不一致** —— 那边的参考场景是小模型
（27B dense）在 B200 上算力吃不满，而 397B MoE 在 v7x 上高并发时本就算力饱和，
投机解码的额外前向反成负担。

`--speculative-config` 的 JSON 引号必须转义，否则内层双引号会截断外层字符串：

```bash
# 对： --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1}
# 错： --speculative-config {"method":"mtp","num_speculative_tokens":1}
```

### 4.8 口径（不满足就没有可比性）

| 项 | 值 | 为什么 |
|---|---|---|
| `BENCHMARK_TEMPERATURE` | **0** | golden client 默认 temp 0，而 `vllm bench serve` 默认走 server 端 sampling（模型 `generation_config` 是 temp 0.7），decode 重的 cell 上明显更慢，两者不可比 |
| `NUM_PROMPTS` | 640（并发 ≤32 时降到 64） | 并发 4 跑 640 条要几小时 |
| `RANDOM_RANGE_RATIO` | 0.8，`min` 风格 | 采样长度落在 [0.8·len, len]，保证不超 `max-model-len` |

### 4.9 收结果

```bash
python3 $RB/Qwen3.5-397B-A17B-FP8/scripts/collect-results.py \
  --context $CTX --ns $NS --hours 4 --repo ./vllm-torchtpu
```

期望 18 行，每行 `640/640`，吞吐差异全部落在 ±5% 内。

> ⚠️ **`--hours` 不能省。** 结果目录里会混进上一轮的 run 目录 ——
> `run-config.sh` 的后台 copier 是 `cp -r benchmark_runs/*`，会把历史全拷进去，
> 不按时间过滤就会把旧数当新数。

**每格必须记全**：TTFT / TPOT / ITL / E2EL 各自的 mean·median·p99，
`total_input_tokens`、`total_output_tokens`、`request_throughput`、`duration`，
**以及产生它们的 ISL / OSL / 并发 / 并行配置** —— 脱离配置的吞吐数字没有意义。

---

## 5.（可选）要和 GPU 公开数据对比，先补三件事

直接拿本轮数字比 B200 **会得出错误结论**：

| 维度 | 本轮 | 公开 B200 数据 | 怎么补 |
|---|---|---|---|
| 序列形状 | 1024:8192 / 8192:1024 | InferenceX 用 **1K/1K**，GCloud 博客用 1024/512 | 改 `ISL_OSL_CONFIGS` |
| 交互速度 | 21–47 tok/s/user | 报的区间是 **59–272** | 并发压到 4/16/32，同时 `NUM_PROMPTS` 降到 64 |
| 投机解码 | **没开** | 开了 MTP / EAGLE | 见 §4.7 |

---

## 故障速查

| 症状 | 真因 | 处理 |
|---|---|---|
| `GKE Warden rejected ... tpu-accelerator-topology-constraints` | pod 缺 `gke-tpu-accelerator` / `gke-tpu-topology` nodeSelector | 两个都加；**allowlist 注解绕不过** |
| `RuntimeError: Insufficient space in /dev/shm: 160 MiB required, 64 MiB free` | K8s 默认 `/dev/shm` 只有 64 MiB | privileged pod 里 `mount -o remount,size=64G /dev/shm` |
| `ERROR: required submodule is not initialized` | 其实是 git **dubious ownership**（tar 跨属主解包） | `git config --global --add safe.directory '*'` |
| `vllm-service-launch: target root does not exist: /run/vllm-metrics-targets/targets` | 目录不存在 | `mkdir -p` 一下 |
| `python3.12: command not found` | 基础镜像不是 3.12 | `uv python install 3.12` 后 symlink 进 PATH |
| 拉 GCS 慢到几十分钟 | 镜像里没有 gcloud SDK，退化到手写下载器 | 换带 SDK 的镜像，`gcloud storage cp -r`，见 §3.2 |
| 桶 403，但桶权限看着没问题 | 节点池默认 scope 只有 `devstorage.read_only` | 建池时 `--scopes=cloud-platform` |
| 拿到卡后节点莫名消失 | `BookingExpired` 后占位 pod 退出，节点被缩掉且**回不来** | 换票前先钉 anchor pod，见 §1.6 |
| 排队几十小时只换到 24 小时 | `maxRunDurationSeconds` 被手填成 `86400`，且**创建后不可改** | 写 `604800` 或干脆不写，见 §1.1 |
| 一直 `Waiting for server...` 到超时 | **镜像里没 curl**，runner 探 `/health` 永远失败（server 其实是好的） | `apt install curl`。这一个包废掉过三个窗口 |
| `SIGBUS` / `signal 7`，栈顶 `determine_available_memory` | **`/dev/shm` 写满**，不是 HBM 不够 | 先 `df -h /dev/shm`，按 §4.7 处理 |
| PVC 挂不上 `exit status 19 / No such device` | TPU 节点 COS 镜像**无 lustre 客户端模块**；CSI pod `2/2 Running` 是假象 | 改用 tmpfs + GCS |
| 容器 `df` 很空但 pod 被驱逐 | ephemeral-storage 是**整节点共享配额**，容器 `df` 看不见 | `/work`、`/tmp`、`~/.cache` 全挂 tmpfs |
| `--dry-run=server` 全绿但没有 pod | dry-run **只校验 JobSet CRD，不校验它生成的 Job** | 跑 `validate-jobset.py`；真因在 `jobset-system` 控制器日志 |
| 进度条 `0 MiB/s` 像卡死 | 按**完成文件数**计数，而 4 GB 分片前很久一个都完不成 | 看 `du -sh` |
| 热启动没吃到编译缓存 | **缓存 key 含完整 serve 配置**，差一个参数全废 | 用 runner 重跑，别手写 `vllm serve` |
| 查了「缓存全命中」但启动仍慢 | 只查了 `"Compiling a graph for compile range"`（确实是 0） | 还要查 `"Compiling model again"` —— 实测 104 个产物里 **16 个每次重编**，上游 bug |
| `pgrep -f "xxx"` 报有进程但实际没有 | 模式串在本命令行里，**匹配到执行它的 bash 自己** | 用 `pgrep python3 \| wc -l`；写 `while pgrep -f ...` 等待循环会**死循环** |
| 脚本报成功但没有结果文件 | 脚本最后一行是 `echo`，**退出码永远 0**，内部失败传不出来 | `run-config.sh` 已加 `exit $RC`；自己写编排脚本时同样要透出失败码 |
| MTP 报 `does not support dynamic shape` | **编译缓存全命中**导致回退重编路径没触发，不是真的不支持 | 清 `$VLLM_CACHE_ROOT` 和 `/work/ttc` 后重跑，见 §4.7 |
| 报 TPU 拓扑不匹配 | `tpu7x-N` 的 N 是 **device 数不是 chip 数** | 4 chip 写 `tpu7x-8`，16 chip 写 `tpu7x-32` |

---

## History（过期数值与它们过期的原因）

<details>
<summary><b>2026-08-14 · 16 chip 那一轮的权重下载数字（已作废）</b></summary>

当时记录的是：

> 权重走同区 GCS，不走 HuggingFace：HF 967 MB/s，**同区 GCS 单节点 231 MiB/s、
> 4 节点聚合 990 MiB/s**。实测装环境 2 分 50 秒，**完全被 28 分钟的下载掩盖**。

**这组数字是工具产物，不是带宽上限。** 那一轮用的 `torch-tpu` 官方镜像里**没有
gcloud SDK**，所以走了 `scripts/fetch-weights.py`（手写的 `google-cloud-storage`
Python 下载器）。

2026-08-27 在同型号节点（`tpu7x-standard-4t`，同 region 桶，目标同样是内存盘）
用 `gcloud storage cp -r` 重测：

| | 2026-08-14（手写下载器） | 2026-08-27（`gcloud storage`） |
|---|---|---|
| 单节点吞吐 | 231 MiB/s | **8.6 GiB/s** |
| 406 GB 用时 | 约 28 分钟 | **45 秒** |
| 倍数 | — | **约 38×** |

**留下的结论**：`fetch-weights.py` 仍然有价值（它按字节校验、支持断点续传，
见 §4.4），但**只在镜像没有 gcloud SDK 时才用它**。
「装环境和下载并行以省 3 分钟」这条优化也随之失效 —— 下载现在比装环境快得多。

</details>

<details>
<summary><b>权重加载耗时的旧基准（69 秒）</b></summary>

原文写「期望 `Loading weights took 69.28 seconds`（tmpfs）；若是 240 秒左右，
说明权重落在磁盘」。

2026-08-27 在 4 chip 上实测 **39.33 秒**。判据本身仍然成立
（**tmpfs 几十秒 / 磁盘约 240 秒，差约 6 倍**），只是「69 秒」不再是唯一期望值 ——
它受节点 CPU 数、分片并发和 DP rank 数影响。**按量级判断，不要按具体秒数判断。**

</details>

<details>
<summary><b>`/dev/shm` 建议值从 32Gi 提到 64Gi</b></summary>

原文建议 `sizeLimit: 32Gi`，并在 MTP 一节注明「常规跑测下会涨到 **69–78%**，
建议起手就提到 64Gi」。现已直接把 **64Gi 写进 §2 的默认值**，
32Gi 只作为「开 MTP 一定会爆」的历史说明保留。

另外补一条 2026-08-27 新踩到的：**K8s 默认 `/dev/shm` 是 64 MiB（不是 GiB）**，
不显式配的话 vLLM 起手就报 `Insufficient space in /dev/shm: 160 MiB required`。

</details>

<details>
<summary><b>DWS `maxRunDurationSeconds` 曾被写成 86400</b></summary>

2026-08-24 提交的 ProvisioningRequest 手填了 `"86400"`，
导致排队 65 小时只换到 24 小时机器。

排查时两处独立确认都是 86400：PR 的 `spec.parameters.maxRunDurationSeconds`，
以及 VM 的 `scheduling.maxRunDuration.seconds`（配 `instanceTerminationAction: DELETE`）。
全仓库 grep 不到任何 `604800` —— **不是平台 clamp，是我们自己填的**。

官方文档明确 `maxRunDurationSeconds` 的默认值就是七天，所以**不填反而更好**。
已在 §1.1 立规。

</details>

<details>
<summary><b>「约 40 小时」的 DWS 排队量级</b></summary>

早前记录 `hy3-v7-4-dws` 这类 DWS 节点池「从提 ProvisioningRequest 到拿到节点，
正常要排 ~40 小时」。2026-08-27 完成一次完整闭环，实测 **65 小时**
（`Accepted` 08-24T05:02Z → `Provisioned` 08-26T22:07Z，4 chip）。

**40 小时是量级不是上限**，盯梢的超时上限按 ≥72 小时设。

</details>

---

**已实测产出**
- 路 A（2026-08-27，4 chip）：decode 5,160.2 tok/s / TPOT 41.747 ms，与发布基线偏差 < 0.05%。
- 路 B（2026-08-14，16 chip）：18/18 格完成，640/640 零失败，吞吐 14 胜 4 负 vs baseline（全在 ±4.3% 内）。
  数据与规律见 [RUNLOG](./Qwen3.5-397B-A17B-FP8/RUNLOG-20260814-16chip.md)。
