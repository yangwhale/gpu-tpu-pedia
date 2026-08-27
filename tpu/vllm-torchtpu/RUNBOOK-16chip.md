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
