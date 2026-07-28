# SGLang · Kimi K3 (2.8T) · GB300 NVL72 端到端 Runbook

> **本文定位**：跟 [VLLM-K3-RUNBOOK.md](./VLLM-K3-RUNBOOK.md) 对应的 SGLang 侧手册。
>
> **来源有三层，全文逐节标注，别混**：
> - `[本环境·已验证]` —— 来自本仓库 **[DeepSeek-V4-Pro SGLang runbook](../deepseek-v4/SGLANG-V4PRO-RUNBOOK.md)**，
>   这是本环境 SGLang-on-GB300 的 **Golden Truth**：端到端、反复验证十几遍、推倒重写过，
>   **凡与本文冲突，一律以它为准**。辅以 [DeepSeek-R1 3P2D 指南](../deepseek-v3/sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md)（PD 与 RDMA 部分）。
>   这一层与模型无关，对 K3 同样成立 —— **不要因为「K3 是新模型」就重新发明**
> - `[K3官方]` —— 来自 SGLang K3 cookbook / LMSYS day-0 博客，**官方自己标着 Not Verified**
> - `[待测]` —— 本环境还没跑，留空等填
>
> **一句话**：环境和流程照我们自己的经验走，K3 专属参数照官方，两边冲突时**以我们踩过的为准**。

---

## ⚠️ 最重要的五条（每条都花过我们一整轮部署）

### 〇、⭐ K3 支持不在 main 分支上 —— 普通 nightly 镜像一律没有 `[本环境·已验证 2026-07-28]`

**这是我们开跑第一步就撞上的坑，而且伪装得极好。**

`kimi_k3.py` **不在 `sgl-project/sglang` 的 main 分支上**，它只存在于官方的
**`kimi-k3` 独立分支**。官方为它单独准备了 `docker/kimi_k3/kimi_k3_cu12.Dockerfile`
和 `kimi_k3_cu13.Dockerfile`（PR #32545 把它们改成从该分支 clone），并推了专用 tag。

| | 值 |
|---|---|
| ✅ **正确镜像（GB300 用这个）** | **`lmsysorg/sglang:kimi-k3-74968e5653-arm64`**（cu13 / arm64 / 15.7 GB） |
| 浮动 tag | `lmsysorg/sglang:kimi-k3`（多架构清单，会自动选 arm64；但**建议 pin**） |
| CUDA 12 版本 | `lmsysorg/sglang:kimi-k3-cu12-74968e5653-arm64`（19.2 GB） |
| ❌ **任何 `nightly-dev-*`** | **不含 K3** |

**为什么会选错**（两层，都要知道）：

1. **日期骗人**：我选了 `nightly-dev-cu13-20260727-38636120`，日期正好是 K3 day-0 当天。
   但它是 **UTC 01:50 构建的**，而 K3 的 commit 全在当天 **08:37 之后** —— 差了 7 小时。
2. **更根本的是它压根不在 main 上**，所以哪怕等到第二天的 nightly 也一样没有。

**最阴的部分：参数先到了，模型没到。** 普通 nightly 里
`--mamba-full-memory-ratio`、`--speculative-dspark-*`、`--dcp-size`、`--enable-linear-replayssm`
**全都有**，`--help` 看着一切正常。只有去查模型注册表才会发现空的：

```bash
# 一条命令判死活 —— 必须看到 kimi_k3.py，只有 kimi_linear.py 是错的镜像
kubectl exec <pod> -- ls /sgl-workspace/sglang/python/sglang/srt/models/ | grep -i kimi
# 正确镜像应有：kimi_k3.py  kimi_k3_vl.py  kimi_linear.py  kimi_k25.py  kimi_vl*.py
# 错误镜像只有：kimi_linear.py  kimi_k25*.py  kimi_vl*.py     ← 没有 k3

# 双保险：parser 里必须注册了 kimi_k3
kubectl exec <pod> -- python3 -c "
from sglang.srt.parser.reasoning_parser import ReasoningParser as R
print([k for k in R.DetectorMap if 'k3' in k])"     # 期望 ['kimi_k3']
```

> **推广出去的教训**：`--help` 里有某个参数，**不等于**对应功能可用。
> 参数解析层和模型实现层是两套代码，可能来自不同的合并批次。
> 验证「镜像支不支持某模型」永远要查**模型注册表**，不要查参数表。

---

### 一、`--mamba-full-memory-ratio` 就是 K3 版的 `swa-full-tokens-ratio`

V4 那轮最贵的教训：**KV 池预算划错，一个参数值决定 54% 的吞吐**，而所有健康信号都是绿的。

K3 上同一个位置的旋钮是 `--mamba-full-memory-ratio`（默认 `0.86`），
它划的是 **KDA 状态池 vs MLA KV 池**的比例。官方 cookbook 甚至配了个计算器，
说明这个值同样是**跟负载绑定、不是常数**。

**判据照搬 V4 的方法**：跑一轮看两个池谁先到 0.9+，谁先满就给谁加预算，目标是两边同时落在 0.88–0.93。
K3 上启动后要回读的是 `max_total_num_tokens`（KV 侧）和**准入请求上限**（状态侧）。

> V4 那次我在 ratio 设错的情况下测出「batch 上不去」，把原因全归给了 prefill —— 方向没错，
> **权重完全估错**。K3 上别重复这个错误：**调参之前先把两个池的占用读出来**。

### 二、`--max-running-requests` 不显式设，开投机后会被重置成 48

`[K3官方]` cookbook 原话：投机解码开着时，SGLang 把未设置的 `--max-running-requests`
**重置为 48**。48 并发在 GB300 上等于什么都没跑。

这条跟 V4 的「三个健康信号全绿但只有 1/3 算力」是同一类陷阱：**不报错，只是慢**。

### 三、`pkill -9` 一个满载的 SGLang 进程会泄漏 ~97 GB 显存/卡

`[本环境·已验证]` V4 实测：`pkill -9` 之后 pod 里全是 zombie、RSS=0，
但 `nvidia-smi` 每张卡还挂着 97 GB。下一次启动必 `Not enough memory`，
**而且报错会把人往「调大 mem-fraction」这个完全相反的方向带**。

**正确做法：删 pod 让 StatefulSet 重建**（实测 56 秒显存归零），不要 pkill。

> 推测原因是 GB300 走 MNNVL/IMEX，导出到 fabric 的显存在 SIGKILL 时不走正常回收路径，
> 而容器 PID 1 是 `sleep infinity` 不 reap zombie。**这个泄漏会污染后续所有实验**——
> V4 那轮我因此连续三次误判「MTP steps=2 内存不够」。

### 四、官方参数是「跟拓扑绑定的一整套」，不能逐项摘着抄

V4 那轮做过三次尝试，把官方 wide-EP recipe 的 decode 参数逐项搬到我们的 dep8 上，
**三次以三种完全不同的方式失败**：

| 搬了什么 | 怎么死的 |
|---|---|
| 压缩 + MTP 一起开 | `AssertionError: online c128 does not support MTP` —— 两个功能互斥 |
| 去掉 MTP，保留其余 | 起得来、注册成功、单条 e2e 通过，**压测 100% 失败**（`retract_decode` 未实现） |
| 把并发退回小值 | `torch.OutOfMemoryError` —— `mem-fraction 0.94` 是 wide-EP 专用，窄拓扑上激活空间被挤没 |

**结论：要么整套换拓扑，要么一个都别动。**

**K3 上这条同样成立，而且更危险**，因为 cookbook 里同时给了好几套预设
（GB300 2×4 的 TP8/DCP8、Deep PP 的 PP8×TP1、大规模的 Peak Throughput / Peak Capacity）。
**它们的参数值是各自成套的**，从 Peak Throughput 那栏摘一个 `--mem-fraction-static` 或
`--max-running-requests` 塞进 2×4 配方里，就是在重演上面那张表。

---

## 0. TL;DR

| 项 | 值 |
|---|---|
| 目标配置（起步） | **Unified · TP8 · DSPARK**，2 节点 8 卡 |
| 硬件口径 `[K3官方]` | GB300 = **2×4**（每节点 4 卡），TP8 / DCP8，MNNVL 与 cuMem 自动识别 |
| 模型 | `moonshotai/Kimi-K3`，MXFP4 权重，**约 1.4 TB**，放 Local SSD RAID |
| Draft | **`RadixArk/Kimi-K3-DSpark`** ⚠️ 跟 vLLM 用的 `Inferact/...` **不是同一个，别混** |
| 关键参数 | `--mamba-full-memory-ratio`（见文首第一条）、`--max-running-requests`（见第二条） |
| 官方数 `[K3官方]` | bs=1 无投机 **~113 tok/s**；+DSPARK **~423 tok/s**；PD 前沿 **2,808 tok/s/GPU** |
| **本环境实测** `[已验证 2026-07-28]` | bs=1 无投机 **87.8**（78%）／+DSPARK **370.4**（**87.6%**，加速 **4.22×**）／unified TP8 峰值 **2,629 tok/s @ conc 64**。完整表见 **§11** |
| **必开的两个开关** | `--enable-symm-mem`（bs=1 值 **+35%**）＋ **DSPARK**（各并发全面占优，无取舍） |
| **已证够不到的** | K3 all-reduce 融合 —— 需 multicast，**TP8 跨节点不可用**（§11.3） |

**最短路径**：§1 前置 → §2 起 fleet → §3 RAID + 模型 → §4 分发 → §5 启动 → §6 就绪判据 →
§7 压测（**第一轮当 warmup 丢掉**）。撞到怪事先翻 **§12**，那里每一行都是真踩过的。

**调参之前先读 §9**（怎么找参数，不是抄参数）和 **§10**（已经证伪的路，别重做）。
这两节是 V4 那轮 +54% 收益的真正来源。

---

## 1. 前置条件 `[本环境·已验证]`

```bash
# ① 同域节点数 —— PD / 跨节点 TP 的 KV 走 MNNVL，跨域会退化到 RDMA 并显著变慢
# 只能落在 0002 / 0006 / 0009 三个池（其余已交付客户）
kubectl get nodes -l cloud.google.com/gke-nodepool=gb300-pool-0002 --no-headers | wc -l

# ② DRA GPU driver
kubectl get pods -A | grep -c dra-driver-nvidia-gpu    # >0

# ③ 没有孤儿 ComputeDomain 占着 channel
kubectl get computedomain -A
```

**同域是硬要求。** 同 nodepool 通常同域。

> `[本环境·已验证 2026-07-28]` **实际能用的只有 `pool-0002`。** 另外两个被别的团队占满：
> `0006` 上跑着 5 套 SGLang PD 实验 + 2 套 vLLM，`0009` 上有 decode 实例和权重分发 job。
> `0002` 只有 daemonset，是唯一空闲的。
>
> ⚠️ **`0002` 和 `0006` 各有一台已经滚到坏 node image `1.36.0-gke.4681000`（COS 224.80）。**
> 这两台的共同特征：**`team` 标签也丢了**，而且 `0002` 那台（`lcg3`）**RAID 没挂上，只有 256K**。
> 三个症状同源 —— 节点被重建过。好在 manifest 里的 `team: yangwhale` 选择器
> **顺手就把它们排除了**。
>
> ⚠️ **可用节点池只有 `0002` / `0006` / `0009` 三个（共 54 节点 / 216 GPU）。**
> 而且这三个池是**因为夹带故障机器才没交付出去**的 —— 健康度要**按节点筛，不能按池信任**。
> 加上集群的 auto-repair 全部关闭，**坏节点不会自愈也不会被替换，会一直待在池里**。
>
> **所以开跑前先出一张逐节点黑名单**，多机作业按「18 台里只用 16 台」规划。
> 排查多机故障时，**第一嫌疑永远是「踩到已知坏机」**，再去查配置和代码。

---

## 2. 部署 pod fleet `[本环境·已验证]`

复用 V4 的 manifest 骨架，**三个关键设计一个都不能省**：

1. **用 StatefulSet，不要裸 pod + `nodeName`** —— DRA 的 ComputeDomain channel 必须由 scheduler 预留，
   `nodeName` 绕过调度器会 `FailedPrepareDynamicResources`
2. **StatefulSet 的稳定 DNS 名是白赚的** —— 跨节点要 `--dist-init-addr`，pod IP 每次重建都变
3. **两个 DRA claim 都要** —— mrdma（8 张 CX-8）+ ComputeDomain channel（MNNVL）

**pod 内存 limit**：`[本环境·已验证]` decode 600Gi 够（KV 在 HBM）；
**长上下文 prefill 要 700Gi**（激活峰值 + 加载缓冲）。节点 909Gi allocatable。
K3 是 1M 上下文，**建议 prefill 直接给 700Gi 起步**。

> ⚠️ `[本环境·已验证]` **16 pod 同时申请 DRA 会滞后**，部分 pod 卡 `ContainerCreating`
> 报 `ResourceClaim not created yet`。删掉卡住的 pod + `apply` 重触发即可，可能重试 1–2 轮。

---

## 3. RAID 与模型 `[本环境·已验证]`

### 3.1 ⚠️ 先查 RAID，再查模型（`md0` → `md127` 陷阱）

**「模型缺失」十有八九不是模型没拷，是那台的 Local SSD RAID 根本没挂上。**

```bash
for i in $(seq 0 7); do
  printf "sgl-%s: " $i
  kubectl exec sgl-$i -- df -h /mnt/ssd | tail -1 | awk '{print $2, $5}'
done
# 正常 12T；看到 256K 100% → RAID 没挂
```

**根因**：节点重启后内核把已存在的阵列自动组装成 `/dev/md127` 而不是 `/dev/md0`，
而 DaemonSet 脚本 `grep -q "md0"` 判定「没有阵列」→ 去 create → 盘已被占 → 连环失败。

**表现极具迷惑性**：hostPath 用 `DirectoryOrCreate`，kubelet 会在只读根文件系统上建出目录、落到 tmpfs，
**pod 正常起、`/mnt/ssd` 存在、但只有 256K**，写模型时静默失败（`curl -o` 写出 0 字节，退出码还是 0）。

修复见 [gb300-local-ssd-raid0-SETUP.md](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md)，
要点是**动态识别 md 号**：`MD=$(awk '/^md[0-9]+ : active/{print $1; exit}' /proc/mdstat)`，
且**有 fs 就别格式化**。挂好后不用重启 pod（`mountPropagation: HostToContainer` 会传播进去）。

### 3.2 模型 `[本环境·已验证]` + `[待测]`

**约 1.4 TB**，比 V4-Pro 的 806G 大 70%。**必须放 Local SSD RAID，不放内存盘**（RAM 留给别的）。

```bash
# 校验（分片数 [待测]，跑通后填）
for i in $(seq 0 7); do
  echo -n "sgl-$i: "
  kubectl exec sgl-$i -- bash -c "du -sh /mnt/ssd/Kimi-K3 | cut -f1; \
    ls /mnt/ssd/Kimi-K3/*.safetensors | wc -l" | tr '\n' ' '; echo
done
```

> ⚠️ **每次重建 fleet 都要重新校验，哪怕上一轮刚跑完。** 节点数比 pod 数多时调度器会换节点，
> 上一轮的空闲节点这轮可能被占，那个 pod 就是空的。V4 审计轮 1 就是这么抓到 `sgl-0` 模型缺失的。

**`[本环境·已验证 2026-07-28]` 本环境的权重来源：不要从 HuggingFace 重下。**

infer 团队 2026-07-27 已经把 K3 下到集群里，并在 leader pod 上起了 rsync daemon，
broadcast 早已跑完、服务端空闲。直接拉最快：

```bash
# 源：rsync://kimi-k3-fill-leader.default.svc.cluster.local/models/Kimi-K3/
# ⚠️ 两边路径布局不同：他们挂 /mnt/disks/ssd0，我们挂 /mnt/disks/raid/0
# ⚠️ sglang 镜像里没有 rsync，先 apt 装
kubectl exec k3sgl-$i -- bash -c "apt-get -qq update >/dev/null && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y rsync >/dev/null"

L=kimi-k3-fill-leader.default.svc.cluster.local
kubectl exec k3sgl-$i -- bash -c "setsid nohup bash -c '
  rsync -aH --info=progress2 --partial --exclude .cache \
    rsync://$L/models/Kimi-K3/ /mnt/ssd/Kimi-K3/ > /tmp/sync.log 2>&1
  echo \$? > /tmp/sync.done' >/dev/null 2>&1 </dev/null & sleep 4"
```

**实测：8 路并行，每路 1.1–1.5 GB/s，聚合 ~10 GB/s，12 TB 约 20 分钟。**

> ⚠️⚠️ **不要为此新建 pod。** 我最初写了个独立的 alpine Job 来做同步，
> **连续 13 次被 `DiskPressure` 拒收**（`Pod was rejected: The node had condition: [DiskPressure]`）。
> 根因：节点 boot 盘只有 **101 GB**，sglang 镜像解压就吃掉约 30 GB，拉镜像期间节点被打上
> DiskPressure 污点，**新 pod 一律拒绝准入**。
>
> **正确做法：在已经 Running 的业务 pod 里跑同步** —— 已通过准入的 pod 不会被重新准入检查。
> 这条对任何「起个小 pod 干杂活」的场景都成立。

### 3.3 ⭐ 权重完整性校验：三层，前两层都不够 `[本环境·已验证 2026-07-28]`

`rsync` 退出码 0 **不代表文件是对的**，8 台互相一致**也不代表对**（可能一起错）。

| 层 | 检查 | 能证明什么 |
|---|---|---|
| ① | `rsync` rc=0 + `du -sh` 约 1.5T | 只能说传输流程没报错 |
| ② | 跨 pod 比对「文件名+字节数」清单指纹 | 只能说 8 台**一致** |
| ③ | **对 index manifest + 解析 safetensors 头部** | ✅ **真正定性** |

```bash
# ③-a 对模型自带的 manifest：声明的分片一个都不能少
kubectl exec k3sgl-0 -- bash -c 'cd /mnt/ssd/Kimi-K3 && python3 -c "
import json,os
m=json.load(open(\"model.safetensors.index.json\"))[\"weight_map\"]
sh=sorted(set(m.values())); miss=[x for x in sh if not os.path.exists(x)]
print(\"分片\",len(sh),\"缺失\",len(miss),\"张量\",len(m))"'
# 期望：分片 96 / 缺失 0 / 张量 497220

# ③-b 解析每个 safetensors 头部，验证声明的数据长度 == 实际文件长度
#     这一步能抓出「文件在、但被截断」—— rc=0 和大小抽查都发现不了
kubectl exec k3sgl-0 -- bash -c 'cd /mnt/ssd/Kimi-K3 && python3 -c "
import json,os,struct,glob
bad=[]
for f in sorted(glob.glob(\"*.safetensors\")):
    sz=os.path.getsize(f)
    with open(f,\"rb\") as fh:
        n=struct.unpack(\"<Q\",fh.read(8))[0]
        hdr=json.loads(fh.read(n))
    end=max(v[\"data_offsets\"][1] for k,v in hdr.items() if k!=\"__metadata__\")
    if 8+n+end!=sz: bad.append(f)
print(\"异常分片:\",len(bad),bad[:5])"'
# 期望：异常分片: 0
```

**本环境实测基准值**（8 台完全一致，可直接拿来对）：

| 项 | 值 |
|---|---|
| 文件数 | **115** |
| safetensors 分片 | **96** |
| 张量数 | **497,220** |
| 精确字节 | **1,560,998,983,759**（1.561 TB） |

> ⚠️ **rsync 会留孤儿临时文件。** 第一次校验时 8 台里有 6 台是 116 个文件、2 台 115 个。
> 多出来的是 `.model-00004-of-000096.safetensors.OfpKba`（约 272 MB）——
> 早先被 DiskPressure 打断那轮 rsync 的残留（`--partial` 会保留）。真分片都在，
> 但这种残留会让「数文件个数」这类校验**永远对不上**。清理：
>
> ```bash
> kubectl exec k3sgl-$i -- bash -c 'cd /mnt/ssd/Kimi-K3 && rm -f .*.safetensors.*'
> ```

**缺失时怎么补**（优先级从高到低）：

**① pod→pod 直传** —— 集群内网远快于 GCS，V4 实测 **3.6 GB/s**（806G 约 4 分钟），
K3 的 1.4 TB 约 7 分钟。源端 `python3 -m http.server`，目标端 `xargs -P 6` + `curl`。

**② 从 GCS 拉** —— ⚠️ **`lmsysorg/sglang` 镜像里没有 `gcloud`**。要么先装，
要么用镜像里已有的 `google-cloud-storage` Python SDK，要么走 ①。

---

## 4. 分发启动脚本 `[本环境·已验证]`

```bash
for i in $(seq 0 7); do
  kubectl exec -i sgl-$i -- bash -c "cat > /tmp/serve.sh && chmod +x /tmp/serve.sh" < scripts/sgl-serve-tp8-dspark.sh
done
kubectl exec sgl-0 -- wc -l /tmp/serve.sh   # ★ 必须校验非空
```

> ⚠️ **必须 `kubectl exec -i`**。少了 `-i` 时 stdin 不透传，容器里得到**空文件，且不报错**。
> `kubectl cp` 同理会静默失败，cp 完必须 `wc -l`。

---

## 5. 启动 `[K3官方]` 参数 + `[本环境·已验证]` 启动纪律

### 5.1 起步配方

> `[本环境·已验证 2026-07-28]` **实际的第一步是 NOSPEC 基线，不是 DSPARK。** 两个理由：
> (1) draft 模型 `RadixArk/Kimi-K3-DSpark` 本环境还没下；
> (2) DSPARK 有 open bug [#32569](https://github.com/sgl-project/sglang/issues/32569)。
> 而且按 §9 的方法论，**没有基线就没有加速比的分母**。
> 脚本：[`scripts/sgl-k3-tp8-nospec.sh`](./scripts/sgl-k3-tp8-nospec.sh)（跨 2 节点 TP8，带 node-rank）。

#### 目标配方：Unified · TP8 · DSPARK（GB300 2×4）

```bash
sglang serve \
  --trust-remote-code \
  --model-path /mnt/ssd/Kimi-K3 \
  --tp-size 8 \
  --disable-custom-all-reduce \
  --enable-symm-mem \
  --mem-fraction-static 0.85 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mamba-full-memory-ratio 0.86 \
  --max-running-requests <按目标并发设，别留空！> \
  --host 0.0.0.0 --port 30000 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \
  --speculative-dspark-block-size 7 \
  --enable-linear-replayssm-spec
```

### 5.2 参数逐条解释

| 参数 | 为什么 | 来源 |
|---|---|---|
| `--mamba-full-memory-ratio 0.86` | KDA 状态池 vs MLA KV 池的划线。**见文首第一条** | `[K3官方]` |
| `--max-running-requests` | **不设会被重置成 48**。见文首第二条 | `[K3官方]` |
| `--enable-linear-replayssm-spec` | ReplaySSM：存原始输入而非快照，draft window 512 KB → 16 KB（约 32×） | `[K3官方]` |
| `--speculative-dspark-block-size 7` | 提 7 个 draft token | `[K3官方]` |
| `--disable-custom-all-reduce --enable-symm-mem` | GB300 对称内存路径 | `[K3官方]` |
| **不要设** `--moe-runner-backend` | Blackwell 上自动选 FlashInfer MXFP4（trtllm-gen SiTU）；H100/H200 才 pin Marlin | `[K3官方]` |
| **不要设** 三个 attention backend | K3 把 prefill / decode / verify 作为一组解析；**设了任何一个就取消其余的自动解析** | `[K3官方]` |

> ⚠️ **「不要设」这类建议要留个心眼。** V4 那轮最阴的坑就是**依赖默认值**：
> 旧文档没写 `--moe-runner-backend deep_gemm`，nightly 镜像更新后 `auto` 改选了 flashinfer，
> 整条 megamoe 路径直接崩。**写文档时能跑 ≠ 三个月后能跑。**
> 建议：先按官方不设，但**从启动日志里把实际选中的 backend 记下来**，写进 §13 验证记录。
> 一旦哪天性能异常，第一件事就是对这行日志。

### 5.2.1 ⚠️ 参数语义必须读源码确认方向，别猜 `[本环境·已验证]` 的教训

V4 那轮在 `swa-full-tokens-ratio` 上**栽了两次**：第一次照搬了别的拓扑的值，
第二次**把方向理解反了** —— 以为调低是给紧张的那个池加预算，结果是把它又砍了一半，
调到 0.056 后 rank 直接挂掉。

源码里的定义是 `swa_tokens = full_tokens × ratio`（`pool_configurator.py:387`），
所以 **ratio ↑ = SWA 池变大 / full 池变小**。这种「A ÷ B 还是 B ÷ A」的歧义，
**光看参数名和文档一定会猜错，必须去翻源码或者用一次实验反推**。

**K3 的 `--mamba-full-memory-ratio` 是同一类参数，同样的坑在等着。**
开跑第一件事就是确认它的方向：

**`[本环境·已验证 2026-07-28]` K3 上的方向已经确认，不用再猜了。** 镜像里 `--help` 原文：

> `--mamba-full-memory-ratio`：*The ratio of **mamba state memory to full kv cache memory**.*

即 `mamba_state = full_kv × ratio`，与 V4 的 `swa_tokens = full_tokens × ratio` **同构**。
所以 **ratio ↑ = KDA 状态池变大 / MLA KV 池变小**，ratio ↓ 反之。

`[待测]` 这两个池的占用打在启动日志的哪一行、字段叫什么名字，跑通后填进 §13。
在拿到这两个数之前，任何关于「并发上不去」的结论都是猜的。

#### 顺带发现：K3 镜像里还有一整排 mamba/DCP 旋钮，cookbook 一个都没提

`[本环境·已验证 2026-07-28]` 从 `--help` 里扒出来的（**都还没测**，但调参时值得知道存在）：

| 参数 | 猜测用途 |
|---|---|
| `--mamba-ssm-dtype` | SSM 状态精度 —— **官方明说过 SSM dtype 是每 GPU 状态账单的少数几个可调项之一** |
| `--max-mamba-cache-size` / `--mamba-max-states-per-path` | 状态池的绝对上限（与 ratio 是两个维度） |
| `--mamba-scheduler-strategy` / `--mamba-radix-cache-strategy` | `{auto,no_buffer,extra_buffer,extra_buffer_lazy}` |
| `--mamba-backend` | KDA kernel 后端选择 |
| `--enable-mamba-cache-stochastic-rounding` / `--mamba-cache-philox-rounds` | 状态缓存的随机舍入 |
| `--enable-int8-mamba-checkpoint` / `--int8-mamba-ckpt-size` | 状态 int8 化 |
| `--dcp-replicate-q-proj` | DCP 下 q_proj 是否复制 |
| `--speculative-moe-runner-backend` | draft 侧单独的 MoE backend |
| `--enable-linear-replayssm` vs `--enable-linear-replayssm-spec` | **是两个不同的开关**，别混 |

---

### 5.3 启动纪律 `[本环境·已验证]`

```bash
# ① 先单个冒烟，再批量 —— 8 个一起错的排查成本远高于先验 1 个
kubectl exec sgl-0 -- bash -c "setsid nohup bash /tmp/serve.sh > /tmp/srv.log 2>&1 </dev/null & sleep 4"

# ② 一个 pod 只启一次。反复启动会堆多个 python 进程叠加 host 内存 → OOM
#    → 容器重启清空 /tmp（脚本和日志一起消失）→ 更乱
#    重启前先清干净：
kubectl exec sgl-0 -- bash -c "pkill -9 -f 'sglang[.]launch_server'"   # ★ 括号必须有
```

> ⚠️ **`pkill -f` 会自杀。** `kubectl exec sgl-N -- bash -c "pkill -9 -f sglang.launch_server; ..."`
> 这条命令行**自身就含有那串字符**，于是把自己杀了（exit 137，后面的语句一条都不执行，而且不报错）。
> 用 `'sglang[.]launch_server'` 括号转义，或 `pkill -9 python`。**本项目在两个框架上踩了四次。**

> ⚠️ **`setsid nohup ... &` 后面要 `sleep 4`。** `kubectl exec` 返回后会关掉 exec 流，
> 子进程会在 detach 完成前被带走 —— 表现是**日志文件根本不生成，且完全不报错**。
> V4 压测时裸 `for` 循环起 14 路，实测只有 6 路活下来。

### 5.3.1 ⭐ 没有「单节点冒烟」这个选项 `[本环境·已验证 2026-07-28]`

V4 的流程是「先单个 pod 冒烟，再批量铺开」。**K3 上这一步做不到**：
1.5 TB 权重放不进 4 张卡（4 × 288 GB = 1,152 GB < 1.5 TB），
**最小可行配置就是跨 2 节点的 TP8**。

所以 K3 的第一次启动天然是「两节点联调」，失败面比 V4 大。对应地：
- `--nnodes 2 --node-rank {0,1} --dist-init-addr <rank0 的稳定 DNS>:5000`
- 用 StatefulSet 送的稳定 DNS（`k3sgl-0.k3sgl.default.svc.cluster.local`），别用 pod IP

> ⚠️ **`kubectl exec pod -- wc -l < /tmp/x.sh` 是错的** —— `<` 重定向发生在**本地 shell**，
> 会去找本机的 `/tmp/x.sh`。要写成 `kubectl exec pod -- bash -c 'wc -l < /tmp/x.sh'`。
> 这个错误的表现是「文件不存在」，很容易被误判成分发失败。

### 5.3.2 ⭐ 启动日志里必须核对的四行 `[本环境·已验证 2026-07-28]`

实测 TP8（2×4，`--enable-symm-mem`）启动时打出来的关键行，**每次起服务都该对一遍**：

```
TP0] FlashInfer TRTLLM MoE deferred finalize is disabled
     (moe_runner_backend=flashinfer_mxfp4, quant_method=Mxfp4MoEMethod)
TP0] K3 all-reduce fusion auto-probe: skipping (enable_symm_mem=True, moe_a2a_backend=none; ...)
     Set SGLANG_K3_AR_FUSION=1 to force.
TP0] multimem all-gather disabled because the TP group spans across nodes.
TP0] Acceleration for non-quantized schemes is not supported by Compressed Tensors.
     Falling back to UnquantizedLinearMethod
TP0] K3 fused KDA decode engaged: --linear-attn-decode-backend LinearAttnKernelBackend.TRITON
     only picks the fallback kernel for shapes the fused kernel does not cover.
```

| 行 | 怎么读 |
|---|---|
| `moe_runner_backend=flashinfer_mxfp4` | ✅ **符合预期** —— Blackwell 上自动解析成 FlashInfer MXFP4，正是 cookbook 说的行为。**这是 V4「依赖默认值型埋雷」的对照点，性能异常时第一件事就是回来对这行** |
| `multimem all-gather disabled ... spans across nodes` | ⚠️ 跨节点 TP8 的**固有代价**，不是配错。想避开只能单节点，但 K3 装不下 |
| `Falling back to UnquantizedLinearMethod` | 指的是模型里**非量化**的那部分层，正常 |
| **`K3 fused KDA decode engaged`** | ✅ **必须看到这行** —— KDA 走的是融合 kernel。注意后半句：`--linear-attn-decode-backend` 设成 TRITON **只对融合 kernel 覆盖不到的 shape 生效**，不是全局切后端。**这也是一个 cookbook 没提的旋钮** |
| **`K3 all-reduce fusion ... skipping`** | ⚠️⚠️ **见下，这条很值钱** |

#### ⚠️ 发现：`--enable-symm-mem` 会关掉 K3 自己的 all-reduce 融合

日志原文：

> `K3 all-reduce fusion auto-probe: skipping (enable_symm_mem=True, moe_a2a_backend=none;
> under symm-mem the allocator contexts conflict, and under EP a2a the model's symm-pool
> allocation contract does not hold on every AR call-site.
> Set SGLANG_K3_AR_FUSION=1 to force.)`

**为什么值钱**：LMSYS 那篇 day-0 博客里，15 级 kernel 优化中**单项收益最大的就是通信融合
（+27.6 tok/s）** —— 把 residual add 和 RMSNorm 塞进 collective 里。而 cookbook 推荐的
`--disable-custom-all-reduce --enable-symm-mem` 组合，**恰好把这条路关掉了**。

两者互斥。cookbook 没有提这个取舍。**待测的对照组**：

| 组 | 配置 | 结果 |
|---|---|---|
| A | `--enable-symm-mem`（cookbook 默认，AR 融合关闭） | `[待测]` |
| B | 去掉 `--enable-symm-mem` + `SGLANG_K3_AR_FUSION=1` | `[待测]` |
| C | `SGLANG_K3_AR_FUSION=1` 强开（日志说可以 force） | `[待测]` |

> 注意 all-reduce 是**同步点**，按官方自己的说法「省一微秒一比一变成 step 时间」，
> 所以这一组对照的杠杆可能不小。跨节点 TP8 下 AR 走 NVLink，收益方向更值得验。

### 5.4 时序预期 `[待测]`

| 阶段 | V4-Pro (806G) 实测 | K3 (1.4 TB) 预期 | 本环境实测 |
|---|---|---|---|
| 权重加载 | 4–5 min | 更久 | **~5 min 到 43%（41/96 分片）**，全程 `[待测]` |
| CUDA graph capture | 3–5 min | ? | `[待测]` |
| **单实例总计** | **8–12 min** | ? | `[待测]` |

> `[本环境·已验证]` **别在 5 分钟时下结论。** V4 审计轮 1 就是因为文档写 180s、
> 实际要 300s，误判成「起不来」。
>
> `[本环境·已验证]` **decode 起来前会刷 `DeepGEMM warmup: 0/65536`，
> 初始 ETA 显示几十小时是误导** —— JIT 一热就到 ~1000 it/s，实际约 1 分钟。别被吓到。

---

## 6. 就绪判据 `[本环境·已验证]` 的方法 + `[待测]` 的具体判据

**V4 最贵的一课：三个看起来最自然的健康信号全是绿的，系统却只有 1/3 的算力。**

| 层 | 判据 | 能不能信 |
|---|---|---|
| ① | `nvidia-smi` HBM 高 | ❌ SGLang **先预分配显存池**，权重可能随后加载失败 |
| ② | 日志出现 `Load weight end` | ❌ 之后还要建 ZMQ / 起 scheduler / 注册，任一步崩都不改这行 |
| ③ | **服务真的能出 token** | ✅ |

```bash
# 最低限度的端到端验证 —— 不做这一步，压测可能全 0
kubectl exec sgl-0 -- curl -s localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K3","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":32}'
```

`[本环境·已验证 2026-07-28]` **K3 的三层判据实测如下：**

| 层 | K3 上的具体信号 | 实测 |
|---|---|---|
| ① HBM | 权重加载完 **242 GB/卡**，起服务后 **~250 GB/卡** | ❌ 不可信（同 V4） |
| ② 日志 | `Load weight end` ×8 | ❌ 不可信（后面还有 autotune / capture） |
| ③ **真的出 token** | `The server is fired up and ready to roll!` + 一次成功的 `/v1/chat/completions` | ✅ **唯一权威** |

**启动会经过 5 个阶段**（V4 只有 3 个，K3 多了 autotune 和 symm 分配器 JIT）：

```
1. 权重加载           96 分片多线程          ~5 min  → "Load weight end"
2. symm allocator     现场 c++ 编译 nccl_allocator.so
3. K3 fused KDA decode engaged
4. AutoTuner          Tuning trtllm_batch_decode_mla（21 profile，约 6 s）
5. cuda graph capture → "The server is fired up and ready to roll!"
```

**实测总时长约 12 分钟**（06:41:39 启动 → 06:53:45 就绪）。
⚠️ 第 3→4 阶段之间 **rank0 的日志会静默一两分钟**，别当成挂了 ——
判断方法：`stat /tmp/srv.log` 看修改时间，或去 **node-1** 看（autotune 进度条刷在那边）。

> `[本环境·已验证]` **存活判据和就绪判据要用相反的信号**：
> 判「起来没」用注册/服务响应，判「还活着没」用 `nvidia-smi` 显存。
> V4 那轮我用注册数判存活，误判「14 台全程在线」，实际早就全崩了。

> `[本环境·已验证]` **`/tmp/srv.log` 是二进制**（含 NUL），`grep` 会直接返回 `binary file matches`。
> 所有分析先 `tr -d '\000'`，或用 `grep -a`。

---

## 7. 压测 `[本环境·已验证]` 方法

### 7.1 ★ 第一轮必须当 warmup 丢掉

`[本环境·已验证]` **重启后首轮比热态低 6.5–7%，不是噪声，高度可复现**：

| | 冷（重启后第 1 轮） | 热（第 2 轮） | 差 |
|---|---|---|---|
| V4 审计轮 1 | 8,520 | 9,118 | +7.0% |
| V4 审计轮 2 | 8,552 | 9,108 | +6.5% |
| 两轮之间偏差 | ±0.2% | ±0.1% | — |

冷热**各自都稳定到 ±0.2%**，说明是确定性的 warmup 成本。**这笔开销在进程内**
（首次遇到各 M shape 时的 kernel 选择 / autotune），**跟磁盘 JIT 缓存在不在无关**——
V4 验证过：`SGLANG_DG_CACHE_DIR` 在节点盘、跨 pod 保留，冷跑照样低 6.5%。

**流程：跑两轮，报第二轮。**

> `[本环境·已验证]` V3 那边还撞到过更极端的：sweep 第一档撞上首次 JIT 编译，
> **TTFT 冲到 67s、总吞吐掉到 438**，看起来像配置全错。第二遍 warm 就正常了。

### 7.2 压并发，别用小 conc 汇报数字

`[本环境·已验证]` V3 的教训：conc=8 只用到 DEP8 容量的 ~3%。K3 上同理。

⚠️ 但注意 K3 的 `--max-running-requests` 陷阱（文首第二条）——**先确认它不是 48**。

### 7.3 口径必须先对齐

`[本环境·已验证]` V4 的教训：官方数字往往是 **output tok/s ÷ decode-GPU 数**，
分子只算 output、分母只算 decode 卡。**同一次测量换个口径能差一倍**
（V4 的 dep16 实验：一个口径 11,880 超标，另一个 5,270 腰斩）。

SGLang K3 官方那几个数的口径：

| 数字 | 口径 |
|---|---|
| ~113 / ~423 tok/s | **单用户 bs=1 decode**（前者无投机，后者 DSPARK） |
| 2,808 tok/s/GPU | **PD 前沿**，1× PP8 prefill 喂 1× TP8 decode |
| 2,633 tok/s/GPU | 2× PP8 prefill 喂 2× DCP8 decode |
| 541 tok/s | agentic 回放，48 并发会话，DCP8 |

> ⚠️ 求和多路结果的前提是**各路时间窗重叠**。V4 踩过：某几路晚启动几十秒，
> 各自吞吐都是在「独占更多算力」的窗口里测的，直接相加会显著高估。
> **核对各路 `Benchmark duration` 相近再求和。**

### 7.4 ⭐ 光看聚合数字看不到 decode 的真实能力 `[本环境·已验证]` 的方法

**端到端吞吐会一直被 prefill 拖着。** decode 引擎每隔几百毫秒自报一次瞬时速率，
**峰值和端到端必须一起看** —— V4 那轮正是靠这个才发现「decode 引擎其实已经超过官方标称，
差的是喂料」，而不是继续瞎调 decode 参数。

V4 的解析命令（`srv.log` 含 NUL 必须先 `tr -d`）：

```bash
kubectl exec sgl-0 -- bash -c "tr -d '\000' < /tmp/srv.log | grep 'gen throughput' | \
  sed -E 's/.*#running-req: ([0-9]+).*full token usage: ([0-9.]+).*swa token usage: ([0-9.]+).*accept len: ([0-9.]+).*gen throughput \(token\/s\): ([0-9.]+).*/\1 \2 \3 \4 \5/' | \
  awk 'NF==5' | sort -k5 -n | tail -1"
```

**两个必须知道的性质**：

1. **这个数是 per-DP-rank（= per GPU），不是引擎总和。** V4 的交叉验证方法：
   `p50 × rank 数` 应当与压测聚合值接近（实测 76,696 vs 74,833，差 2.5%）。
   **若差了一个 rank 数的倍数，说明口径理解错了。**
2. **`accept len` 就在同一行**，投机解码的接受长度不用另外测。

`[待测]` **K3 的 decode 日志字段名与 V4 不同**（K3 是 KDA 状态池 + MLA KV 池，
不是 full/swa）。跑通后把 K3 版的解析命令与字段名填进 §13，这是后续所有调参的前提。

### 7.5 先明确你在哪个工作点，「健康值」差 3 倍 `[本环境·已验证]`

| 工作点 | TPOT 中位 | TTFT 中位 | 说明 |
|---|---|---|---|
| **低延迟**（官方 frontier 曲线所在处） | 20–35 ms | < 10 s | batch 小，交互性优先 |
| **最大吞吐** | 58–85 ms | 60–85 s | batch 撑到 KV 池上限，用延迟换吞吐 |

**两个都正常，选哪个取决于 SLA。** 不先说清工作点就比数字，等于没比。

> `[本环境·已验证]` **`ITL` 和 `TPOT` 会差两个数量级**（如 ITL 2200 ms vs TPOT 21 ms），
> 这是 `--stream-interval` 把多个 token 攒成一个 chunk 发的结果，**不是异常**。
> 判断延迟一律用 TPOT。

### 7.6 一个反推公式：几秒钟区分「decode 到顶」和「prefill 饿着」`[本环境·已验证]`

```
正在 decode 的序列数 ≈ 聚合 output tok/s × TPOT(秒)
```

拿它跟你 offer 的总并发比 —— **差得远就说明请求都堵在 prefill 侧**。
这一步比盯 GPU 利用率快得多。

配套的异常对照表：

| 指标 | 含义 |
|---|---|
| TPOT 正常但吞吐只有目标 40–60% | **喂料不足**，先查后端实例数，别急着调参数 |
| TTFT 冲到分钟级而 TPOT 反而偏低 | 同上，prefill 在饿着 decode |
| 两个 KV 池占用差 0.3 以上 | **池预算划错**，见文首第一条（V4 上这条值 54%） |
| 各路 `Benchmark duration` 差 >15% | 没有真正并发，聚合数不可信 |

### 7.7 必须开环 `[本环境·已验证]`

对标官方数字**必须用开环**（`--request-rate inf`）。V4 实测：闭环已经限死在途请求数，
再调队列阈值只增延迟不增吞吐（**−13%**）。

`--ignore-eos` 保证每条真出满 OSL。

> ⚠️ **但 `random` 数据 + `--ignore-eos` 测不出投机解码。** 随机 token 的 draft 接受率接近 0，
> 投机变成纯开销，**反而更慢**。vLLM 侧实测过这个坑：DSpark 在 random 数据上比不开还低。
> **要量化 DSPARK 的 423 tok/s，必须用真实连贯数据**（ShareGPT / sa-bench + chat template）。

---

## 8. 官方指标（背景） `[K3官方]`

### 8.1 单用户 decode（bs=1）

| 阶段 | tok/s |
|---|---|
| bring-up 基线（Marlin W4A16 MoE） | 44.3 |
| 15 级 kernel 优化之后 | **112.5** |
| **+ DSPARK 投机解码** | **~423** |

那 15 级优化里最大的四块：**通信融合 +27.6**（NVLS in-switch reduction，
把 residual add 和 RMSNorm 塞进 collective 里）、**launch/copy 消除 +19.9**、
**NVIDIA 定制 kernel +10.3**、**overlap 与 prologue 融合 +10.4**。

> 官方总结的方法论值得记：**all-reduce 是同步点，省一微秒一比一变成 step 时间；
> 而在别的流里 overlap 的 kernel 只按十分之一折算。写 kernel 前先在 trace 里
> 确认它在不在关键路径上——这是整轮优化里杠杆最高的习惯。**

### 8.2 PD 前沿（在 2×4 GB300 上测的，跟我们硬件一致）

| 拓扑 | 每 GPU 吞吐 |
|---|---|
| **1× PP8 prefill → 1× TP8 decode**（fp4 arm） | **2,808 tok/s/GPU** |
| 2× PP8 prefill → 2× DCP8 decode | 2,633 tok/s/GPU |
| 1 prefill 喂 2 / 3 / 4 个 decode 实例 | 拿总吞吐换单用户速度，**推到 116+ tok/s/user** |

### 8.3 三个架构结论

**① prefill 用深度 PP，不用 TEP。** `--pp-size 8 --tp-size 1`。
实测 **PP8×TP1 约为 TEP8 上限的 1.7 倍**，TTFT 还更低；单个 PP8 prefill 节点
有 TEP8 节点 **1.45–1.72 倍**的 prefill 能力。理由是 TP 每层结尾都有 AllReduce 挡着，
PP 的 stage 间传递能被下一个 chunk 的计算盖住（K3 上隐藏了 91%）。

> ⚠️ **必须用满 8 个 stage**。浅切（PP4×TP2）还要付 TP2 的 all-reduce，benchmark 上打不过 TEP8。
> 而且 **DSPARK 与 PP 互斥**（要求 `pp_size == 1`）。

**② decode 用 DCP 按 token 位置切 KV。** `--dcp-size 8`。
MLA 只有一个 KV head，TP 切不动、每 rank 存全量副本。DCP8 把**逻辑 KV 从 1.5M 拉到 12.2M token（约 7.9×）**。
agentic 回放实测：TP8 在 16 并发就崩，DCP8 扛到 **48 并发 / 541 tok/s**。

**③ KDA 状态池是并发天花板。** **DP、EP、DCP 都不切它** ——
只有 attention-TP 宽度、SSM dtype、cache 策略能改每 GPU 的账单。MLA KV 反而好压（fp8）好去重（DCP）。

### 8.4 ⚠️ 官方数字是 Pareto 曲线上的点，不是「某个配置的成绩」`[本环境·已验证]` 的教训

V4 那轮花了很久才想明白：官方那个 11,200 的原文是
*「the June 2026 MTP curve delivers ~11,200 tok/s/GPU **at roughly 50 tok/s/user**」* ——
**它限定了交互性**。我们当时的工作点 TPOT 58–63 ms ≈ 16 tok/s/user，
也就是在**交互性差 3 倍**的点上比吞吐，两个轴都在人家里面，根本不是简单的 throughput/latency 取舍。

**同类陷阱在 K3 上照样有**：

| K3 官方数 | 隐含的工作点限定 | 对标前必须确认 |
|---|---|---|
| ~113 / ~423 tok/s | **bs=1 单用户**，不是服务态 | 别拿它跟高并发聚合数比 |
| 2,808 tok/s/GPU | PD 前沿，1× PP8 → 1× TP8，**fp4 arm** | 精度 arm、P:D 配比、并发点都要对齐 |
| 541 tok/s @ 48 会话 | agentic **回放**负载，DCP8 | 换成 random 数据结论不成立 |
| 116+ tok/s/user | 1 prefill 喂 2/3/4 decode，**拿总吞吐换单用户速度** | 这是曲线另一端，不能和 2,808 同时拿到 |

> **写结论时必须把工作点一起写**，否则三个月后自己都会看错。

### 8.5 PD 配比公式（V4 蒸馏，K3 直接可用）`[本环境·已验证]`

```
需要的 prefill 数 = (decode 每秒完成请求数 × 输入长度) ÷ 单 prefill 吞吐
```

拿 K3 官方 PD 前沿那个点代入，就能先算出**该配几个 prefill**，不用靠扫描试：

1. decode 每秒完成请求数 = `decode 卡数 × 每卡 tok/s ÷ OSL`
2. 需要的 input tok/s = 上式 × ISL
3. 除以单 prefill 实测吞吐 = prefill 个数

> ⚠️ **这个口径把 prefill 成本藏起来了**：`output ÷ decode-GPU` 堆再多 prefill 喂一个小 decode
> 都好看。它是「解码效率」指标，**不是整机 TCO**。对外汇报要说清分母。

`[待测]` 单 PP8 prefill 实测 input tok/s = ____，据此算出的最优 P:D = ____。

---

## 9. 调参方法论（从 V4 那轮蒸馏，与模型无关）`[本环境·已验证]`

**这一节不是 K3 的参数值，是「怎么找到参数值」的方法。** V4 那轮最终 +54% 的收益
全部来自这套方法，而不是来自任何一条官方建议。

### 9.1 判据一：两个池同时饱和才是最优点

V4 上 `swa_tokens = full_tokens × ratio`，两个池**吃同一份预算**（实测总量恒定）。
调 ratio 就是在两池之间划线，**划错一边就有一半浪费**：

| 场景 | full 占用 | swa 占用 | 结果 |
|---|---|---|---|
| ISL 4K @ ratio 0.20 | 0.93 | 0.63 | full 先满，SWA 浪费 37% → batch 卡在 727 |
| **ISL 4K @ ratio 0.15** | **0.92** | **0.89** | **两边同时到顶 → batch 冲到 886** |
| ISL 8K @ ratio 0.15 | 0.80 | 0.37 | SWA 浪费 63% → batch 只有 410 |

**最优值跟负载绑定，不是常数** —— 序列越长，full-attn 侧占比越高，就要更低的 ratio。
V4 甜点：ISL 4K → 0.15，ISL 8K → 0.10~0.12。

**K3 的 `--mamba-full-memory-ratio` 划的是 KDA 状态池 vs MLA KV 池，同一个结构。**
判据照搬：**哪个先到 0.9+ 就给哪个加预算，目标两边同时落在 0.88–0.93。**

### 9.2 判据二：batch 涨而吞吐不涨 = 已经算力饱和，别再调 batch 参数

V4 上把 KV 总预算加 13%：

| | batch/rank | 峰值 tok/s/GPU | 端到端 |
|---|---|---|---|
| 基线 | 886 | 12,063 | 10,614 |
| +13% 预算 | **999**（+13%） | 12,070（**+0.06%**） | 10,704（+0.8%） |

**batch 涨 13%，吞吐纹丝不动** → 已经从「KV 容量受限」转成「算力受限」。
此时继续调 `max-running-requests` / `cuda-graph-max-bs` / `mem-fraction` **全部没有意义**，
只剩两条路：更快的 kernel，或换拓扑摊薄每卡计算量。

**K3 上做同一个实验来确认天花板在哪**，别在错误的一侧堆参数。

### 9.3 判据三：峰值与端到端可能方向相反，先说清对标哪个

V4 上 MTP `steps=1 → 2`：

| | accept len | 反推单步耗时 | 峰值 tok/s/GPU | 端到端 |
|---|---|---|---|---|
| steps=1 | 1.88 | **115 ms** | **11,851** | 6,887 |
| steps=2 | 2.36（+26%） | 162 ms（**+41%**） | 10,279（−13%） | **8,802（+28%）** |

**峰值亏 13%、端到端赚 28%** —— 因为「每步多吐 token」的收益一直在，
「每步更慢」的代价只在 batch 撑到顶时才显著。

**K3 的 `--speculative-dspark-block-size`（默认 7）是同构旋钮**，
调它之前先决定：对标 decode 引擎能力（看峰值）还是真实业务吞吐（看端到端）。

### 9.4 判据四：「瓶颈在 X」必须标注参数前提

V4 那轮我在 ratio 设错的情况下测出「batch 上不去」，把原因**全归给 prefill 供给不足** ——
方向没错，**权重完全估错了**。真实分层是：

| 因素 | 量级 |
|---|---|
| **KV 池分配失衡** | **+54%** |
| 供给不足（攒 batch） | +4.3% |
| kernel 硬顶 | 还没摸到 |

> **别把「在某组参数下扫出来的上界」当成系统天花板。**
> 并发扫描只在你已经把其他参数调对的前提下才测得出天花板。

### 9.5 判据五：被污染环境下得出的失败结论一律作废

V4 上判定「MTP steps=2 内存不够」连续 OOM 三次、还降了 `cuda-graph-max-bs` ——
**三次全跑在上一次 `pkill -9` 留下的 97 GB 泄漏里**。清干净后一次就起来了。

**规矩：每次启动前先看 `nvidia-smi` 显存是否归零，不归零得出的任何结论都不算数。**

### 9.6 判据六：不要默认「新镜像更成熟」或「旧镜像更稳」

V4 那轮有个假设是「我们用 nightly，官方用更成熟的 pinned 镜像，差在内核成熟度」——
**查了 commit 日期发现方向反了**，官方 pinned 比我们用的还旧两个月。

**去查 commit，别猜。**

### 9.7 判据七：一次探活成功 ≠ 现在还活着 `[2026-07-28 又栽一次]`

本轮做过一次错误推断：router 报 `No available decode workers`，我手工 curl 同一地址拿到 **200**，
于是下结论「服务是好的，是 router 探活方式对不上」。**十分钟后查实：decode 早就死了**，
那次 200 恰好落在它死前的窗口里。

| 要判断的事 | 用什么 |
|---|---|
| 起来了没（readiness） | **端到端出 token**（§6 第三层） |
| 还活着没（liveness） | **进程数 + 显存**，**不看单次 HTTP 探活** |

跟 V4「etcd 判就绪权威、判存活会骗人」是同一条。
更一般地：**任何瞬时观测都不能推广成持续状态。要断言「现在是 X」，就得在现在测。**

### 9.8 判据八：一次只动一个变量，否则数据作废 `[2026-07-28 教训]`

本轮出过一组两变量混淆：8K 上拿 **DCP8 + ratio 0.86** 去比 **TP8 + ratio 0.60**，
得出「DCP8 赢 24%」—— **但 ratio 也变了**，而我们已知 ratio 在 8K 上有影响。
补测 **TP8 + ratio 0.40** 后才确认 DCP8 对**两个** ratio 都赢，结论才成立。

**流程要求**：并行跑多组时，每组只相对基准改一个参数；
若为省时间同时改了两个，**必须在表里标注「不可作为结论」**，并排队补对照组。

### 9.9 判据九：操作失误产生的数据也可能有效 —— 前提是记清配置 `[2026-07-28]`

试 AR 融合时我顺手加了 `--disable-custom-all-reduce`，**把 AR 融合自己的依赖也关掉了** —— 纯属操作失误。
但因为**当时把完整配置差异记了下来**，这组「失败」的运行意外成了一个干净的
**symm-mem 单独消融点**（bs=1 −26%，即 symm-mem 值 +35%），是本轮最有用的数据之一。

> **与 §9.5 的区别**：被污染环境下的**失败结论**要作废；
> 但配置明确的**意外配置**产生的数据是有效的。
> **分界线在于：你知不知道当时到底跑的是什么。**

---

## 10. 已验证无效 / 有害的尝试（别重做）`[本环境·已验证]`

V4 那轮真金白银试过、**确认不work** 的路。K3 上除非有明确理由，否则别再花时间：

| 尝试 | 结果 |
|---|---|
| 关掉 DeepGEMM 的快速 warmup、做 full autotune | 热态与基线在噪声内（9,018 vs 8,993）。冷→热那 +12% 是**一次性 JIT**，不是 autotune 的功劳，代价是巨大启动开销 |
| 「攒批再放」的 env hack（预留 decode token） | 吞吐**降到 2,984**，过度预分配，**有害**。⚠️ 注意跟官方的 `--disaggregation-decode-polling-interval` 区分 —— 那个是**有效的（+4.3%）**，两者别混 |
| 闭环压测 + 调队列阈值 | 闭环已限死在途请求数，再调只增延迟不增吞吐（**−13%**）。对标官方必须开环 |
| 把 `mem-fraction-static` 往上顶 | batch +13%、吞吐 +0.06%，**换不到东西还抬高 OOM 风险** |
| 逐项摘官方 wide-EP 参数搬到窄拓扑 | 三次三种死法，见文首第四条 |
| 加宽 decode 拓扑但不加机器 | 总吞吐 +16.7%、TPOT −45%，但 **per-decode-GPU 腰斩** —— decode 卡多了，prefill 少了，喂不饱。**这是节点数的硬约束，不是配置问题** |

> **K3 上对应的「先别做」**：在把 `--mamba-full-memory-ratio` 调对之前，
> 不要去扫 `--max-running-requests` / `--mem-fraction-static` / DCP 拓扑 ——
> V4 的经验是这些都会被池划分的 54% 淹没，扫出来的结论全要作废。

---

## 11. 本环境实测结果 `[本环境·已验证 2026-07-28]`

> 📊 **原始数据**：[`bench-raw-20260728.csv`](./bench-raw-20260728.csv)（43 组热轮测量，config / conc / tok/s / TPOT / TTFT / duration）
> 💰 **代价**：32× B300 跑了约 6 小时。**结论务必看 §11.0 总表，别重复跑。**

---

> 🔴 **2026-07-28 20:36 HKT 测试中断**：集群 `gb300-gke-test` 的**全部 8 个 GB300 节点池被批量删除**
> （同一秒内 8 个 `DELETE_NODE_POOL`，含我们的 `gb300-pool-0002`）。
> 数据与脚本已入库；**8 节点上各 1.5 TB 的 node-local 权重随节点销毁**。
> 恢复需重建节点池 + 重拉 12 TB 权重（原 rsync 源同样消失）。
> 中断项：DSPARK@32K（只到 conc=1）、DCP8@128K conc32、**PD 全线未取得数据**。

## 11.0 ⭐⭐ 消融总表 —— 每个旋钮值多少钱

**统一条件**：TP8 跨 2 节点，OSL 1024，random 数据，`temperature=0`，开环，冷热两轮只取热轮。
所有百分比 = 相对同工作点的对照组。

### 11.0.0 ⭐ 全量实测宽表 —— 每个配置 × 每个并发 × 三项指标

**读法**：一行 = 一个完整配置。前 6 列是配置本身，后面每个并发档三列（吞吐 / 每 token 延迟 / 首字延迟）。
**同 ISL 的行之间可以直接比**，跨 ISL 不要横向比（负载不同）。

| 配置 | 拓扑 | ISL | ratio | symm | 投机 | c1 tok/s | c1 TPOT | c1 TTFT | c8 tok/s | c8 TPOT | c8 TTFT | c32 tok/s | c32 TPOT | c32 TTFT | c64 tok/s | c64 TPOT | c64 TTFT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TP8 基线** | TP8 | 4K | 0.86 | 开 | — | **88** | 11.1 ms | 320 ms | **434** | 16.0 ms | 842 ms | **1,130** | 25.4 ms | 2.3 s | **1,511** | 36.3 ms | 6.0 s |
| **TP8 + DSPARK** | TP8 | 4K | 0.86 | 开 | DSPARK blk7 | **370** | 2.3 ms | 323 ms | **1,242** | 5.8 ms | 351 ms | **2,138** | 12.6 ms | 457 ms | **2,629** | 14.6 ms | 6.2 s |
| **C1-r095** | TP8 | 4K | 0.95 | 开 | — | **88** | 11.1 ms | 312 ms | **434** | 16.0 ms | 844 ms | **1,139** | 25.3 ms | 2.3 s | **1,516** | 36.3 ms | 5.8 s |
| **C1-r060** | TP8 | 4K | 0.6 | 开 | — | **88** | 11.1 ms | 309 ms | **437** | 16.1 ms | 751 ms | **1,093** | 26.1 ms | 2.7 s | **1,278** | 33.4 ms | 6.0 s |
| **A2-dcp8** | TP8+DCP8 | 4K | 0.86 | 关(强制) | — | **65** | 15.2 ms | 315 ms | **397** | 19.3 ms | 900 ms | **1,019** | 28.2 ms | 2.5 s | **1,409** | 38.9 ms | 6.6 s |
| **B-nosymm-noAR** | TP8 | 4K | 0.86 | 关 | — | **65** | 15.1 ms | 328 ms | — | — | — | — | — | — | **1,401** | 39.3 ms | 6.3 s |
| **r040-isl8k** | TP8 | 8K | 0.4 | 开 | — | — | — | — | **443** | 16.2 ms | 1.9 s | **891** | 30.0 ms | 6.2 s | — | — | — |
| **r060-isl8k** | TP8 | 8K | 0.6 | 开 | — | — | — | — | **452** | 16.2 ms | 1.6 s | **893** | 29.7 ms | 6.2 s | — | — | — |
| **dcp8-isl8k** | TP8+DCP8 | 8K | 0.86 | 关(强制) | — | — | — | — | **400** | 19.3 ms | 712 ms | **1,110** | 27.7 ms | 1.2 s | — | — | — |
| **r040-isl32k** | TP8 | 32K | 0.4 | 开 | — | **84** | 11.4 ms | 728 ms | **410** | 17.0 ms | 2.0 s | **420** | 47.9 ms | 22.8 s | — | — | — |
| **r060-isl32k** | TP8 | 32K | 0.6 | 开 | — | **83** | 11.4 ms | 729 ms | **408** | 17.1 ms | 1.8 s | **408** | 42.2 ms | 22.8 s | — | — | — |
| **base-r086-isl32k** | TP8 | 32K | 0.86 | 开 | — | **83** | 11.4 ms | 730 ms | **344** | 18.4 ms | 4.3 s | **390** | 39.9 ms | 26.6 s | — | — | — |
| **r095-isl32k** | TP8 | 32K | 0.95 | 开 | — | **83** | 11.4 ms | 728 ms | **296** | 20.8 ms | 6.3 s | **388** | 38.4 ms | 27.7 s | — | — | — |
| **dcp8-isl32k** | TP8+DCP8 | 32K | 0.86 | 关(强制) | — | **62** | 15.2 ms | 1.3 s | **337** | 19.6 ms | 2.4 s | **436** | 49.4 ms | 24.5 s | — | — | — |
| **dcp8r060-isl32k** | TP8+DCP8 | 32K | 0.6 | 关(强制) | — | — | — | — | **341** | 19.9 ms | 3.4 s | **433** | 49.6 ms | 24.7 s | — | — | — |
| **r060-isl128k** | TP8 | 128K | 0.6 | 开 | — | **57** | 12.0 ms | 6.7 s | **108** | 34.4 ms | 32.1 s | — | — | — | — | — | — |
| **base086-isl128k** | TP8 | 128K | 0.86 | 开 | — | **57** | 12.0 ms | 6.7 s | **99** | 34.1 ms | 39.3 s | — | — | — | — | — | — |
| **dcp8-isl128k** | TP8+DCP8 | 128K | 0.86 | 关(强制) | — | **64** | 15.3 ms | 439 ms | **365** | 19.8 ms | 2.3 s | — | — | — | — | — | — |

**指标口径**：`tok/s` = Output token throughput（只算输出，不含输入）；
`TPOT` = Median Time Per Output Token；`TTFT` = Median Time To First Token。
均为**热轮**（冷轮已丢弃）。ISL 4K 的 128/256 档见 §11.0.1 下方补充。

**4K 超高并发补充**（TP8 基线，证明 conc>64 无收益）：

| conc | tok/s | TPOT | TTFT |
|---:|---:|---:|---:|
| 64 | **1,511** | 36.3 ms | **6.0 s** |
| 128 | 1,476（−2%） | 37.6 ms | **49.3 s** |
| 256 | 1,477（0%） | 37.4 ms | **137.7 s** |

---

### 一句话结论

| 旋钮 | 值多少 | 何时值钱 | 何时没用 |
|---|---|---|---|
| **DSPARK 投机解码** | **+322%**（bs=1） | **永远。各并发全面占优，且 TPOT 同时下降** | 无 —— **无脑开** |
| **`--enable-symm-mem`** | **+35%**（bs=1）/ +7.8%（conc64） | 短上下文、低并发 | 高并发下收窄；**DCP 下用不了** |
| **`--dcp-size 8`** | **+237%**（128K/c8） | **ISL ≥ 8K 且高并发** | **ISL ≤ 8K 低并发时 −9~26%** |
| **`--mamba-full-memory-ratio`** | **±38%**（32K/c8） | **8K–32K 中等长度** | 4K 差 15%、**128K 完全无效（差 0.02%）** |
| K3 AR 融合 | **拿不到** | — | TP8 跨节点无 multicast |
| DEP8（V4 那套） | **不可行** | — | 加载即 OOM，两次 |

---

### 11.0.1 DSPARK：唯一的无条件赢家

| conc | NOSPEC | DSPARK | 加速 | TPOT 变化 |
|---:|---:|---:|---:|---|
| 1 | 87.8 | **370.4** | **4.22×** | 11.1 → **2.31 ms** |
| 8 | 434.4 | **1,241.9** | 2.86× | 16.0 → **5.83 ms** |
| 32 | 1,129.7 | **2,138.4** | 1.89× | 25.4 → **12.6 ms** |
| 64 | 1,511.1 | **2,629.3** | 1.74× | 36.3 → **14.6 ms** |

**吞吐涨 + 延迟降，同时发生**，不存在取舍。接受率 **0.85→0.89 随并发升高**，accept len 6.96→7.20（block=7）。
代价：`max_total_num_tokens` −19%（draft 占 KV 预算）。**对标官方 423，我们 370.4 = 87.6%。**

### 11.0.2 `--mamba-full-memory-ratio`：作用区间是中等长度

**同拓扑，只变 ratio，Output tok/s：**

| ISL \ ratio | 0.40 | 0.60 | 0.86 | 0.95 | 极差 |
|---|---:|---:|---:|---:|---|
| **4K** c64 | — | 1,278 | **1,511** | 1,516 | **−15.4%**（用 0.60） |
| **8K** c8 | 442.8 | **451.7** | — | — | −2% |
| **8K** c32 | 890.9 | 893.2 | — | — | 0.3% |
| **32K** c8 | **409.5** | 408.3 | 343.8 | 296.4 | **−38%**（用 0.95） |
| **32K** c32 | 419.5 | 408.2 | 390.0 | 387.7 | −8% |
| **128K** c1 | — | 56.58 | 56.57 | — | **0.02%（完全无效）** |

**规律**：短上下文要**高** ratio（状态池大 → 能同时跑更多请求），
长上下文要**低** ratio（KV 池大 → 长序列装得下），**超长时怎么调都没用**。

**决定性证据**（4K / conc 64 offered）：

```
r0.60 → running=36 / 64   ← 状态池小，只准入 36 条
r0.95 → running=64 / 64   ← 全部准入
```

**KDA 状态池就是并发天花板** —— 官方「DP/EP/DCP 都不切它」的实测印证。

> **与 V4 同构**：V4 的 `swa-full-tokens-ratio` 最优值也是 ISL 4096→0.15、8192→0.10，
> **ISL 越长 ratio 越低**。§9 那套方法论跨模型有效。

### 11.0.3 `--dcp-size 8`：交叉点在 8K，之后碾压

| 工作点 | TP8 最优 | DCP8 | DCP8 相对 |
|---|---:|---:|---|
| 4K c1 | 87.8 | 64.7 | **−26%** |
| 4K c64 | 1,511 | 1,409 | −6.8% |
| 8K c8 | 451.7 | 400.0 | −11% |
| **8K c32** | 893.2 | **1,109.9** | **+24%** |
| 32K c8 | 408.3 | 337.0 | −17% |
| **32K c32** | 408.2 | **436.5** | +7% |
| **128K c1** | 56.6 | **63.6** | **+12%** |
| **128K c8** | 108.3 | **365.2** | **+237%** |

**真正的轴是 `ISL × conc` 的总 KV 压力**，不是单看 ISL。

**DCP8 的固定成本 = −26%，且恰好等于关掉 symm-mem 的损失**（§11.0.4 的 64.9 vs 87.8）。
所以 **DCP 本身是中性偏正的，负数全部来自被迫关 symm-mem**。

⚠️ **DCP8 + 低 ratio 不叠加**：32K c32 → DCP8+r0.86 = 436.5，DCP8+r0.60 = 432.9（微负）。
两者争同一份预算，DCP 已把 KV 切薄，再降 ratio 是浪费。

### 11.0.4 `--enable-symm-mem`：短上下文的免费午餐

| | bs=1 | conc 64 |
|---|---:|---:|
| 开 | **87.8** | **1,511.1** |
| 关 | 64.9 | 1,401.2 |
| **收益** | **+35.3%** | **+7.8%** |

**机制**：TP8 每层 ≥2 次 all-reduce × 93 层 ≈ **每 token 186 次通信**。
bs=1 时计算量极小、时间全在通信上；batch 大了通信被摊薄。
**反推自洽**：4.0 ms 差 ÷ 186 ≈ **21 µs/次**，与 NCCL 小消息开销量级吻合。

### 11.0.5 两条走不通的路（别重试）

| 路 | 现象 | 根因 |
|---|---|---|
| **K3 AR 融合** | `SGLANG_K3_AR_FUSION requested but CustomAllReduceV2 with multicast is unavailable` | 需 multicast，而 `multimem all-gather disabled because the TP group spans across nodes` → **TP8 跨节点拿不到**。这很可能是 bs=1 只有官方 78% 的结构性原因 |
| **DEP8**（`--enable-dp-attention --dp-size 8 --ep-size 8`） | 两次加载即 `OutOfMemory`（mem-frac 0.85 与 **0.70** 都爆） | DP-attention 让**每个 rank 复制一份完整 KDA 状态池**，K3 有 69 层 KDA → ×8 直接爆。**这就是官方在 K3 上从 dep 改推 TP/DCP 的原因** |
| **PD prefill PP8×TP1** | `numHeadsQ/numHeadsKv is not supported` (`fmhaKernels.cuh:444`) | TP=1 时 head 比值超出 trtllm MLA kernel 支持范围。**绕法：显式 `--prefill/decode-attention-backend flashinfer`**（代价是失去 backend 自动解析） |

---

### 11.0.5b DSPARK 的加速随上下文长度急剧衰减 `[本环境·已验证]`

| ISL | NOSPEC bs=1 | DSPARK bs=1 | 加速 |
|---|---:|---:|---|
| **4K** | 87.8 | **370.4** | **4.22×** |
| **32K** | 83.3 | **109.8** | **1.32×** |

**从 4.22× 掉到 1.32×。** 与官方说法一致：
*"Its win is largest on short interactive traffic and **fades as the prompt grows**."*

**含义**：DSPARK 的收益来自「用闲置算力提前猜」。长上下文下 attention 本身就重、GPU 已经忙，
可供投机使用的空隙大幅减少。**做长上下文选型时不能拿 4K 的 4.22× 去外推。**

⚠️ 32K 只拿到 conc=1（集群被删中断），conc 8/32 未测。

### 11.0.6 选型决策表（照这个配）

| 业务场景 | 拓扑 | ratio | 投机 | symm-mem |
|---|---|---|---|---|
| **短上下文（≤4K）低并发** | TP8 | **0.86–0.95** | **DSPARK** | **开** |
| **短上下文高并发** | TP8 | 0.86–0.95 | DSPARK | 开 |
| **中等（8K–32K）低并发** | TP8 | **0.40–0.60** | DSPARK | 开 |
| **中等高并发（≥c32）** | **DCP8** | 0.86（降没用） | DSPARK | 关（强制） |
| **长上下文（≥128K）** | **DCP8** | 任意（无影响） | DSPARK | 关（强制） |

---


**统一条件**：`gb300-pool-0002`，8 pod × 4 GPU，镜像 `lmsysorg/sglang:kimi-k3-74968e5653-arm64`，
Unified **TP8 跨 2 节点**，ISL 4096 / OSL 1024，random 数据，`temperature=0`，
开环 `--request-rate inf`，**每档冷热两轮、只报热轮**。

### 11.1 ⭐ 主结果：NOSPEC vs DSPARK 并发扫描

| conc | NOSPEC tok/s | **DSPARK tok/s** | **加速** | NOSPEC TPOT | DSPARK TPOT | NOSPEC TTFT | DSPARK TTFT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 87.8 | **370.4** | **4.22×** | 11.1 ms | **2.31 ms** | 0.32 s | 0.32 s |
| **8** | 434.4 | **1,241.9** | **2.86×** | 16.0 ms | **5.83 ms** | 0.84 s | 0.35 s |
| **32** | 1,129.7 | **2,138.4** | **1.89×** | 25.4 ms | **12.60 ms** | 2.3 s | **0.46 s** |
| **64** | 1,511.1 ← NOSPEC 峰值 | **2,629.3** | **1.74×** | 36.3 ms | **14.61 ms** | 6.0 s | 6.2 s |
| 128 | 1,476.4 | 未测 | — | 37.6 ms | — | 49.3 s | — |
| 256 | 1,476.5 | 未测 | — | 37.4 ms | — | 137.7 s | — |

**对标官方**：bs=1 无投机 87.8 / 113 = **78%**；bs=1 + DSPARK 370.4 / 423 = **87.6%**。
**我们的投机加速比 4.22× 反而高于官方的 3.7×** —— 因为基线更低，投机把差距补回来了一大截。

#### 四个结论

**① DSPARK 在所有并发上严格占优，不是拿延迟换吞吐。**
conc=64 时它同时做到 **吞吐 1.74×** 和 **TPOT 只有 40%**（14.6 ms vs 36.3 ms）。
这跟 V4 上 MTP「峰值亏 13%、端到端赚 28%」那种此消彼长完全不同 —— **K3 上没有取舍，直接开**。

**② NOSPEC 的天花板是 1,511 tok/s @ conc 64，再加并发一个 token 都不多。**
128 档吞吐 −2%、TTFT 从 6 s 炸到 49 s；256 档吞吐持平、TTFT 138 s。
**conc > 64 不是可用工作点。**

**③ DSPARK 的天花板还没摸到。** conc 32→64 还在涨 23%（2,138 → 2,629），
而 NOSPEC 同区间只涨 34% 后就平了。**下一轮应该把 DSPARK 的并发往 128/256 推。**

**④ 接受率高得离谱，这就是 4× 的来源。**

| 并发档 | accept len | accept rate |
|---:|---:|---:|
| bs~1 | 6.96 | 0.85 |
| bs~8 | 7.04 | 0.86 |
| bs~32 | 7.18 | 0.88 |
| bs~64 | **7.20** | **0.89** |

`--speculative-dspark-block-size 7` 提 7 个 draft token，**平均接受 7 个以上**（含 bonus token）。
**对照 V4：MTP 的 accept len 只有 1.88–2.36。** K3 的 DSPARK draft 质量高一个量级。
而且**接受率随并发升高**（0.85 → 0.89），与直觉相反，值得后续挖。

### 11.2 symm-mem 消融 `[本环境·已验证]`

| 配置 | bs=1 | conc=64 |
|---|---:|---:|
| A：`--enable-symm-mem`（cookbook 默认） | **87.8** | **1,511.1** |
| B：symm-mem 关 + custom-AR 关 | 64.9 | 1,401.2 |
| **symm-mem 收益** | **+35.3%** | **+7.8%** |

TPOT：11.1 ms → 15.1 ms（+36%）。

**为什么收益随并发收窄**：TP8 下每层至少 2 次 all-reduce，K3 有 93 层 ≈ **每 token 186 次通信**。
bs=1 时每层计算量极小，时间几乎全在通信上；batch 变大后计算占比上升，通信被摊薄。

**反推自洽性检查**：4.0 ms 差 ÷ 186 次 ≈ **每次 all-reduce 21 µs**，
与 NCCL 小消息启动开销量级吻合 —— 数字自洽，解释站得住。

> **结论：`--enable-symm-mem` 必开。**

### 11.3 ⚠️ K3 all-reduce 融合：本拓扑上拿不到 `[本环境·已验证]`

尝试 `SGLANG_K3_AR_FUSION=1` 强开，日志回：

```
SGLANG_K3_AR_FUSION requested but CustomAllReduceV2 with multicast is unavailable;
falling back to the regular all-reduce path.
```

**它需要 CustomAllReduceV2 + multicast。** 而启动时另有一行：

```
multimem all-gather disabled because the TP group spans across nodes.
```

**推断：TP8 跨 2 节点 → multicast 不可用 → AR 融合在 GB300 2×4 上够不到。**

这条很可能是我们 bs=1 只有官方 78% 的**结构性原因** ——
官方 kernel 阶梯里最大的一块就是通信融合（+27.6 tok/s，约 +33%），
而那大概率是在**单节点 8 卡**上测的。`[待验证]` 需要单节点 8×B300 才能证实。

> ⚠️ 我第一次试的时候顺手加了 `--disable-custom-all-reduce`，
> **把 AR 融合自己的依赖也关掉了** —— 这是操作失误。
> 但它意外产出了 §11.2 那个干净的 symm-mem 消融点。
> **前提是当时把配置差异记清楚了**，否则就是一组废数据。

### 11.4 DSPARK 的代价与自动配置 `[本环境·已验证]`

**内存代价**：`max_total_num_tokens` **787,504 → 637,504（−19%）**，draft 模型吃掉 KV 预算。
高并发 / 长上下文下这会变成约束。

**开 DSPARK 会自动钉三个后端**（cookbook 一个字没提，但这是 4× 的实现基础）：

```
Kimi-K3 DSPARK on SM100/SM103: decode/verify attention backend trtllm_mla
  (speculative_attention_mode=decode)
Kimi hybrid model with speculative decoding: pinning --linear-attn-verify-backend
  to nv_cutedsl (uses the fused Kimi-K3/DSPARK CuTeDSL kernel)
Kimi hybrid DSPARK: defaulting --speculative-draft-attention-backend to trtllm_mha
```

**open bug [#32569](https://github.com/sgl-project/sglang/issues/32569)（DSPARK crash）在本镜像上没有复现。**

### 11.5 内存池占用（NOSPEC，conc 256 峰值时刻）

```
#running-req: 53   full token usage: 0.35   mamba usage: 0.64
```

**两个池都没满**（0.35 / 0.64），而 offered 并发 256、decode 里只有 53 条 ——
**200 多条堵在 prefill 队列**。按 §9 的判据：**这是喂料不足，不是 KV 容量受限**。

推论：`--mamba-full-memory-ratio` 在**当前这个工作点**上榨不出多少。
V4 上它值 54%，K3 上不同 —— **因为 K3 的 KDA 把 KV 压力本来就削掉了大半**。
这也解释了为什么 AFD（attention-FFN 分离）对 K3 的意义不如对全注意力 / MLA 模型大。

### 11.5b ⭐ 拓扑与内存池：4K vs 32K 出现方向反转 `[本环境·已验证 2026-07-28]`

**四对节点并行跑的结果**（Output tok/s，TP8 跨 2 节点）：

**ISL 4096**

| conc | TP8 r0.86（基准） | TP8 r0.60 | TP8 r0.95 | TP8+DCP8 |
|---:|---:|---:|---:|---:|
| 1 | **87.8** | 87.7 | 87.8 | 64.7 |
| 8 | **434.4** | 436.9 | 434.3 | 396.8 |
| 32 | 1,129.7 | 1,093.2 | **1,138.8** | 1,019.1 |
| 64 | **1,511.1** | 1,278.4 ⬇15% | 1,515.7 | 1,409.1 |

**ISL 32768**

| conc | TP8 r0.60 | TP8 r0.95 | TP8+DCP8 |
|---:|---:|---:|---:|
| 1 | 83.3 | 83.3 | 61.9 |
| **8** | **408.3** ⬆ | 296.4 | 337.0 |
| 32 | 408.2 | 387.7 | **436.5** |

#### 结论 1：最优 ratio 随 ISL 反向移动 —— V4 的规律在 K3 上复现

| ISL | 谁赢 | 幅度 |
|---|---|---|
| 4K @ conc64 | **r0.95 / r0.86** | r0.60 输 **15%** |
| 32K @ conc8 | **r0.60** | r0.60 赢 **38%**，TTFT 还快 3.6×（1.78 s vs 6.34 s） |

**机制**：序列越长，每条请求要的 MLA KV 越多 → ratio 低（KV 池大）才装得下；
序列短时 KV 不缺，瓶颈变成「能同时跑几条」→ 状态池大（ratio 高）才关键。

**决定性证据**（ISL 4K，conc=64 offered）：

```
r0.60 → running=36 / 64   full=0.20  mamba=0.54    ← 只准入 36 条
r0.95 → running=64 / 64                            ← 全部准入
```

**KDA 状态池就是并发天花板** —— 官方原话「DP、EP、DCP 都不切它」的实测印证。

> **这与 V4 完全同构**：V4 上 `swa-full-tokens-ratio` 的最优值也是 ISL 4096→0.15、8192→0.10，
> **ISL 越长 ratio 越低**。参数名不同、架构不同，但「两个池吃同一份预算、最优点随负载移动」
> 这个结构一致 —— **§9 那套方法论跨模型有效。**

**实操结论：不存在通用 ratio，必须按业务典型 ISL 定。**

| 典型 ISL | 建议 ratio |
|---|---|
| ~4K | **0.86–0.95**（默认 0.86 已接近拐点，往上持平、往下掉得快） |
| ~32K | **≤0.60**（下界还在扫） |

#### 结论 2：DCP8 在长上下文才翻身

| | bs=1 | conc 8 | conc 32/64 |
|---|---|---|---|
| ISL 4K | −26% | −8.7% | −6.8%（conc64） |
| ISL 32K | −26% | −17% | **+7%（conc32，全场最高 436.5）** |

**32K / conc32 时 DCP8 反超**。而且它的劣势一直稳定在 bs=1 的 −26%，
**跟 §11.2 测出的「关掉 symm-mem 掉 26%」几乎完全相等** ——
说明 **DCP 本身是中性偏正的，负数全部来自被迫关掉 `--enable-symm-mem`**。

> ⚠️ 官方宣称的 **7.9× 逻辑 KV 容量** 在 4K 下完全没体现（`max_total_num_tokens` 787K→778K）。
> 要看到它得推到更长上下文 —— 128K 扫描进行中。

#### 结论 3：DEP8（V4 那套）在 K3 上不可行 `[已判定]`

两次尝试均在**加载阶段** CUDA OOM：

| 尝试 | `--mem-fraction-static` | `--max-running-requests` | 结果 |
|---|---|---|---|
| 1 | 0.85 | 256 | `OutOfMemory: Tried to allocate 1.15 GiB, 680 MiB free` |
| 2 | **0.70** | 128 | **同样 OOM** |

配置：`--enable-dp-attention --dp-size 8 --ep-size 8 --moe-a2a-backend deepep`。

**根因**：DP-attention 让每个 rank 独立服务不同请求，因而**每个 rank 都要一份完整的 KDA 状态池**。
K3 有 69 层 KDA，状态池本来就大，再 ×8 直接爆。
对照 V4-Pro：CSA+HCA hybrid 的 KV 薄得多，dep8 装得下。

> **这就是官方在 K3 上从 dep 改推 TP / DCP 的结构性原因。**
> DP 不但不切状态池，还把它复制了 8 份。

---

### 11.5c ⚠️ 我们的 PD 配置偏离了官方 —— 恢复后必须改正 `[官方文档核对 2026-07-28]`

扒完 cookbook §3.4 后发现，本轮 PD 尝试有三处不符官方：

| # | 我们用的 | **官方** | 后果 |
|---|---|---|---|
| 1 | prefill = **`--pp-size 8 --tp-size 1`** | **`Default` 就是 TP8**；PP8 是 `Long-Context` 专用策略，且官方明说*"Pays only with several requests in flight"* | **这直接导致了 `numHeadsQ/numHeadsKv is not supported` 崩溃 —— 本可不踩** |
| 2 | `--disaggregation-transfer-backend mooncake` | **NiXL（RDMA）是 cells 默认发出的**，mooncake 只是 Playground 可选项 | 未知 |
| 3 | router `--port 30200` | 官方 **`:8000`** | 无实质影响 |

**官方 PD 的完整正交矩阵**（不是单一配置）：

| 维度 | 取值 |
|---|---|
| PD Mode | Unified / **Prefill** / **Decode** |
| Prefill 策略 | **`Default` (TP8)** / `Long-Context` (PP8×TP1)，**两者都在 16k 分块** |
| Strategy | `Low-Latency`（纯 TP 无 DCP，聊天）/ **`Balanced`**（GB300 = TP8/DCP8）/ `High-Throughput`（大规模预设） |
| Spec Decode | Non-Spec / **DSPARK** / DFLASH（无公开 checkpoint） |
| HiCache | Off / L1+L2(host) + L3(Mooncake) |

**大规模两个预设**（`N = 8k` GPU）：

| 预设 | 换什么 | 何时选 |
|---|---|---|
| **Peak Throughput**（`dp=k`，attention-TP 8） | 状态池 8 路切分；每步 KDA all-reduce 留在一个 8-GPU 节点内，或跨两个 4-GPU GB300 节点走 MNNVL。**`--kv-cache-dtype fp8_e4m3` 是 load-bearing** —— bf16 KV 装不下 128 请求/副本 | 最大持续 TPS |
| **Peak Capacity**（+`--dcp-size 8`） | 去重 attention-TP 组的 MLA KV：**并发上限 +72%，引擎吞吐不变，ITL ×1.8** | 上下文 ≥16K，或每副本并发 >128 |

> ⚠️ **口径提醒**：官方对 DCP 的量化是「**容量 +72%**」，不是吞吐。
> 我们实测的 128K/c8 **+237% 吞吐**是**在 TP8 装不下 8 条长序列、被迫串行**的前提下测出来的，
> 两个数不矛盾但**不是同一件事**。

**其它官方硬性要求（我们全程没配）**：

- `--kv-cache-dtype fp8_e4m3`（大规模必须）
- PD decode 的状态池是 **chunk cache（一请求一 slot）**，`--mamba-radix-cache-strategy` **在 PD 下完全失效**
- `--disaggregation-decode-extra-slots` **必须 pin**：不 pin 时 <32 请求默认两倍 batch、>32 时为**零**

#### 官方 PD frontier 原文与我们的差距

> *At the throughput end **one PP8 prefill worker feeding one TP8 decode node delivers 2,808 tok/s per GPU on the fp4 arm**, with the DCP composition, **two PP8 prefill workers feeding two DCP8 decode nodes, just behind at 2,633**. Moving right is the prefill:decode knob at work: one prefill worker feeding two, three, then four independent decode instances trades aggregate throughput for per-user speed, **out past 116 tok/s per user**.*

**分母 = decode GPU 数**，与本团队一贯口径一致。

| | 每 decode GPU |
|---|---:|
| 我们最好（unified TP8+DSPARK，conc64，8 卡） | **329** |
| 官方 1P1D | **2,808** |
| 官方 2P2D | 2,633 |
| **差距** | **~8×** |

**这 8 倍的可能来源**（恢复后按此顺序排查）：

1. **PD 本身** —— decode 卡不再与 prefill 争算力
2. **并发量级** —— 我们 conc 64 就饱和，**但那是 prefill 堵住的**（实测 offered 256 时仅 53 条进 decode）
3. **`--kv-cache-dtype fp8_e4m3`** —— 官方称 load-bearing，我们没开
4. **「fp4 arm」** —— 官方明确标注该数在 FP4 精度分支上

#### 恢复后的最大规模计划

8 节点 / 32 卡正好摆得下官方第二个配置：

| 角色 | 节点 | 配置 |
|---|---|---|
| prefill ×2 | 4 | 官方 `Default` = **TP8**（不是 PP8） |
| decode ×2 | 4 | **DCP8** |
| **decode 卡数** | **16** | 目标 **2,633 tok/s/GPU** → 聚合 ≈42,000 tok/s |

---

### 11.6 完整待测矩阵 `[待测]`

**成本基准**：换一次配置 = **删 pod 重建 ~1 min + 加载就绪 ~12 min + 扫描 10–25 min ≈ 25–40 min/组**。
资源：8 节点 / 32 卡，其中 unified TP8 只吃 2 节点，**目前利用率 25%**。

**已测掉的**：TP8 NOSPEC 并发扫描、TP8+DSPARK 并发扫描、symm-mem 消融、AR 融合（证实够不到）。

---

#### A 组 · Decode 拓扑 —— ⭐ 最高优先，选错了后面全错

> **背景**：MLA 只有一个 KV head，**TP 切不动它**，TP8 下每 rank 存一份完整副本。
> V4-Pro 当年用 **dep8**（`--data-parallel-size 8 --expert-parallel-size 8`）按**请求**切来规避。
> K3 官方改推 **DCP** —— 按 **token 位置**切，宣称逻辑 KV 容量 **7.9×**。
> ⚠️ **我们已测的 TP8 恰恰是 KV 被复制 8 份的那一档**，所以 §11.5 里
> 「`full token usage: 0.35` 说明 KV 没满」这个判断**分母可能是虚的**，A 组测完要回头修正。

| # | 配置 | 节点 | 要回答的问题 | 依赖 | 状态 |
|---|---|---|---|---|---|
| A1 | TP8（基准） | 2 | — | — | ✅ 已测 2,629@64 |
| **A2** | **TP8 + `--dcp-size 8`** | 2 | 官方 7.9× 逻辑 KV 是否兑现？最大并发能推到多少？ | ⚠️ DCP 下**不能开** `--enable-symm-mem` | **下一个** |
| **A3** | **DEP8**：`--enable-dp-attention --dp-size 8 --ep-size 8` | 2 | V4 那套在 K3 上还成不成立？vs DCP 谁赢？ | 需选 `--moe-a2a-backend` | 待跑 |
| A4 | TP16 | 4 | 单实例加宽的收益；对照 vLLM 侧 370 tok/s | — | 待跑 |
| A5 | DCP16 / DEP16 | 4 | A2/A3 的加宽版 | 看 A2/A3 结果再定 | 条件触发 |
| A6 | `--dcp-comm-backend` | 2 | GB300 上默认 `fi_a2a`，是否有更优 | A2 | 待跑 |
| A7 | `--dcp-replicate-q-proj` on/off | 2 | q_proj 复制的代价 | A2 | 待跑 |

**A 组要产出的核心表**：同样 8 卡，三种切法的 **最大并发 / 逻辑 KV 容量 / 峰值吞吐 / TPOT** 四维对比。

---

#### B 组 · 投机解码调优（DSPARK 已证 4.22×，还没调过）

| # | 配置 | 要回答的问题 | 状态 |
|---|---|---|---|
| B1 | `--speculative-dspark-block-size` = 4 / 7 / 10 / 12 | 7 的接受率已 0.89，**更大的 block 是否更好**？ | 待跑 |
| B2 | 并发推到 **128 / 256** | **DSPARK 天花板还没摸到**（32→64 仍涨 23%） | 待跑（优先） |
| B3 | `--enable-linear-replayssm-spec` on/off | 官方称 draft window 512 KB→16 KB（32×），实测值多少 | 待跑 |
| B4 | `--enable-linear-replayssm`（**另一个开关**）+ `--linear-replayssm-cache-len` | 与 B3 的区别 | 待跑 |
| B5 | `--speculative-moe-runner-backend` | draft 侧单独的 MoE backend | 待跑 |
| B6 | `--speculative-draft-attention-backend` | 默认自动钉 `trtllm_mha`，能否更好 | 待跑 |
| B7 | `--speculative-dspark-align-verify-tokens-to-graph-tier` | 对齐 graph tier 的收益 | 待跑 |
| B8 | `--speculative-dspark-sps-table-path` / `-confidence-sts-path` | 官方给的查找表，**默认没用上** | 待跑 |

---

#### C 组 · 内存池（K3 版 `swa-ratio`，V4 上这类参数值 54%）

| # | 配置 | 要回答的问题 | 状态 |
|---|---|---|---|
| C1 | `--mamba-full-memory-ratio` = 0.60 / 0.75 / **0.86** / 0.95 | 判据：`full token usage` 与 `mamba usage` 哪个先到 0.9+ | 待跑 |
| C2 | `--mem-fraction-static` = 0.85 / 0.88 / 0.90 | V4 上 +13% 预算只换 +0.06%，K3 上呢 | 待跑 |
| C3 | `--max-running-requests` 扫描 | ⚠️ 与 **DSPARK block-size 的乘积**受 DeepEP dispatch buffer 约束 | 待跑 |
| C4 | `--max-mamba-cache-size` / `--mamba-max-states-per-path` | 状态池绝对上限（与 ratio 是两个维度） | 待跑 |
| C5 | `--mamba-ssm-dtype` | **官方明说 SSM dtype 是每 GPU 状态账单的少数可调项之一** | 待跑 |
| C6 | `--enable-int8-mamba-checkpoint` / `--int8-mamba-ckpt-size` | 状态 int8 化换容量 | 待跑 |
| C7 | `--enable-mamba-cache-stochastic-rounding` | 精度影响 | 待跑 |

---

#### D 组 · 通信

| # | 配置 | 结论 / 问题 | 状态 |
|---|---|---|---|
| D1 | `--enable-symm-mem` on/off | ✅ 已测：bs=1 **+35%**，conc64 +7.8% → **必开** | ✅ 已测 |
| D2 | `SGLANG_K3_AR_FUSION=1` | ✅ 已测：需 multicast，**TP8 跨节点不可用** | ✅ 已测（够不到） |
| **D3** | **单节点 8×B300 上验 AR 融合** | 证实「跨节点 → multicast 不可用」这个结构性推断 | ⚠️ **需要非 GB300 机器** |
| D4 | `--moe-a2a-backend` | Blackwell 上默认 none，EP 场景要选 | 随 A3 一起 |

---

#### E 组 · Kernel / Backend（全部 cookbook 未提）

| # | 配置 | 要回答的问题 | 状态 |
|---|---|---|---|
| E1 | `--moe-runner-backend` auto vs 显式 pin | 现在自动选 `flashinfer_mxfp4`。**V4 的「依赖默认值型埋雷」在这** | 待跑 |
| E2 | `--mamba-backend` | KDA kernel 后端 | 待跑 |
| E3 | `--linear-attn-decode-backend` | 日志说 TRITON **只对融合 kernel 覆盖不到的 shape 生效** | 待跑 |
| E4 | 三个 attention backend（prefill / decode / draft） | ⚠️ **设任一个会取消其余的自动解析** | 待跑 |
| E5 | `--moe-dense-tp-size` | dense 层单独 TP（V4 用的是 1） | 待跑 |

---

#### F 组 · 缓存

| # | 配置 | 要回答的问题 | 状态 |
|---|---|---|---|
| F1 | radix cache on/off | 本轮全程默认；random 数据下无收益，真实数据下才见效 | 待跑 |
| F2 | `--mamba-radix-cache-strategy` = auto/no_buffer/extra_buffer/extra_buffer_lazy | K3 双池前缀树的缓存策略 | 待跑 |
| F3 | HiCache L1/L2/L3 | ⚠️ **host 层还不完全 DCP-aware**：L3 一律、L1+L2 开投机时都要退回纯 TP | 待跑（低优先） |

---

#### G 组 · PD 分离 —— ⭐ 对标官方 2,808 的唯一路径

> ⚠️ **必须等 A 组定了 decode 拓扑再做**，否则 PD 的 decode 侧也是错的。

| # | 配置 | 节点 | 官方对标 | 状态 |
|---|---|---|---|---|
| G1 | 1× PP8 prefill → 1× **最优 decode 拓扑** | 4 | **2,808 tok/s/GPU** | 待跑 |
| G2 | 2× PP8 → 2× DCP8 | 8 | 2,633 tok/s/GPU | 待跑 |
| G3 | 1 prefill → 2 / 3 / 4 decode | 4–8 | **116+ tok/s/user** | 待跑 |
| G4 | prefill **PP8×TP1 vs TEP8** 对照 | 4 | 官方称 PP8 是 TEP8 的 **1.7×** | 待跑 |
| G5 | `--disaggregation-decode-extra-slots` | — | ⚠️ 不固定的话 <32 请求默认两倍 batch、>32 为零 | 随 G1 |
| G6 | KV 传输：mooncake + `MC_FORCE_MNNVL=1` | — | ⚠️ V3 教训：nixl 走 RoCE 在 GKE 上调不通 | 随 G1 |

---

#### H 组 · 负载维度（K3 是 1M 上下文，只测 4K 远远不够）

| # | 配置 | 要回答的问题 | 状态 |
|---|---|---|---|
| **H1** | **ISL 扫描 1K / 4K / 8K / 32K / 128K** | KDA 的价值全在长上下文；**这是 K3 最该测的一维，我们只测了 4K** | 待跑（高优先） |
| H2 | OSL 扫描 256 / 1K / 4K | 长输出下投机收益如何变化 | 待跑 |
| H3 | **真实数据（ShareGPT / agentic 回放）vs random** | ⚠️ vLLM 侧踩过：**random + ignore-eos 测不出投机**。我们 random 下都有 4.22×，真实数据可能更高 | 待跑（高优先） |
| H4 | 官方 agentic 回放场景 | 对标官方 **541 tok/s @ 48 并发会话** | 待跑 |

---

#### I 组 · 正确性（不是性能，但上线前必须过）

| # | 项 | 状态 |
|---|---|---|
| I1 | 基础冒烟（中文/事实/数学/代码） | ✅ 已过，见 §13.2 |
| I2 | **长上下文正确性**（32K / 128K / 1M 大海捞针） | 待跑 |
| I3 | **投机解码前后输出一致性** | ⚠️ 投机理论上无损，但要实测验证 |
| I4 | Tool calling | ⚠️ 官方警告过 parser 偶尔吐出自己不认的格式 |
| I5 | 视觉输入 | 开源 serving 契约**只支持图像**，拒绝视频音频 |

---

### 11.7 建议执行顺序

**为什么这个顺序**：先定拓扑（拓扑错了下面全白测）→ 再榨已知最大杠杆 → 最后才是精调。

| 批次 | 内容 | 预计 | 产出 |
|---|---|---|---|
| **1** | **A2 (TP8+DCP8) → A3 (DEP8)** | ~1.5 h | **decode 拓扑三方对比表**，定下后续基准配置 |
| **2** | B2（DSPARK 推 128/256）+ H1（ISL 扫描） | ~2 h | 找到真天花板 + 长上下文行为 |
| **3** | C1 + C5（内存池两个主旋钮） | ~1.5 h | 池划分最优点 |
| **4** | **G1 (1P1D)** → G4（PP8 vs TEP8） | ~2 h | **对标官方 2,808** |
| **5** | B1/B3/B8（投机精调）+ E1/E3 | ~2 h | 逼近官方 423 |
| **6** | H3/H4（真实数据）+ I2/I3（正确性） | ~2 h | 上线可信度 |
| **7** | G2/G3（多 decode 扩展）| ~2 h | 规模化配比 |

> **每批结束都要**：结果写回 §11、失败/坑写回 §12、更新 §13 验证记录。
> **每次换配置前必查**：`nvidia-smi` 显存归零（§9.5）、两个模型都在这台节点上（pod 重建会换节点）。

---

---

## 12. 故障速查

**上半部分继承自 V4 / V3，与模型无关，对 K3 同样成立；下半部分是 K3 专属。**

### 12.1 环境与流程类 `[本环境·已验证]`

| 现象 | 根因 | 处理 |
|---|---|---|
| **跑通了但吞吐只有一半**，健康信号全绿 | 实际在服务的实例数 < 预期 | 用 §6 的端到端判据重查，别信显存和日志 |
| `/mnt/ssd` 只有 256K | RAID 没挂（`md0`→`md127`） | §3.1，动态识别 md 号 |
| 容器里脚本是空文件 | `kubectl exec` 少了 `-i`；或 `kubectl cp` 静默失败 | 加 `-i`，之后必须 `wc -l` 校验 |
| **日志文件根本不生成、也不报错** | exec 关流太快，`setsid` 还没 detach 完 | exec 里加 `sleep 4` + 外层校验重试 |
| `kubectl exec` 自己 exit 137、后续语句没执行 | `pkill -f <pat>` 匹配到 exec 自身命令行 | `'sglang[.]launch_server'` 括号转义 |
| **换个参数就 `Not enough memory`，改回原参数也起不来** | 上次 `pkill -9` 泄漏了 ~97 GB/卡 | **删 pod 让 StatefulSet 重建**（56s），别 pkill |
| 重建 fleet 后某 pod 模型目录是空的 | 调度器换了节点 | §3.2 每次重建都校验，缺就 pod→pod 补 |
| 容器里 `gcloud: command not found` | `lmsysorg/sglang` 镜像不含 gcloud | pod→pod 直传，还快 4× |
| `FailedPrepareDynamicResources` | 裸 pod + `nodeName` 绕过 scheduler | 用 StatefulSet |
| pod 卡 `ContainerCreating` / `ResourceClaim not created yet` | 多 pod 同时申请 DRA，controller 滞后 | 删卡住的 pod + `apply` 重触发，重试 1–2 轮 |
| `OOMKilled` (exit 137) 加载时 | host 侧加载缓冲峰值超限 | 内存 request ≥600Gi，长上下文 prefill 700Gi；**一 pod 只启一次** |
| pod `Evicted` DiskPressure | 大镜像顶爆 boot 盘 | fresh 节点池 + 删重建 |
| `grep` 返回 `binary file matches` | `srv.log` 含 NUL | 先 `tr -d '\000'` 或用 `grep -a` |
| **镜像里没有 K3 模型**（只有 `kimi_linear.py`），但 `--help` 里参数一应俱全 | K3 只在 `kimi-k3` 分支，不在 main；普通 nightly 一律没有 | 用 `lmsysorg/sglang:kimi-k3-*-arm64`。查法见文首第〇条 |
| 新建的小 pod 连续被 `DiskPressure` 拒收 | boot 盘仅 101 GB，拉大镜像期间节点被打污点，**新 pod 拒绝准入** | **别新建 pod**，把活放进已经 Running 的业务 pod 里跑 |
| 权重文件数各节点对不上 | rsync 孤儿临时文件 `.xxx.safetensors.RANDOM`（`--partial` 保留的断点残留） | `rm -f .*.safetensors.*`，再按 §3.3 三层校验 |
| `kubectl exec pod -- wc -l < /tmp/x` 报文件不存在 | `<` 重定向在**本地 shell** 执行 | 写成 `-- bash -c 'wc -l < /tmp/x'` |
| 想先单节点冒烟却起不来 | 1.5 TB 放不进 4 张卡，**最小可行就是跨 2 节点 TP8** | 直接两节点联调，见 §5.3.1 |
| 首轮压测数字异常低（TTFT 几十秒） | 首次 JIT 编译 | §7.1，跑两轮报第二轮 |
| `DeepGEMM warmup 0/65536` ETA 几十小时 | JIT 未热时的误导性估算 | 实际约 1 分钟，别被吓到 |

### 12.2 PD 相关 `[本环境·已验证]`，K3 上大概率同样成立

| 现象 | 根因 | 处理 |
|---|---|---|
| 单请求 60s 超时 / `KVTransferError` | nixl 走 RoCE 在 GKE 上调不通（RoCE v2 over IPv6，netdev 名 `gpuNipvlanM`） | **改走 NVLink**：`--disaggregation-transfer-backend mooncake` + `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` + `MC_FORCE_MNNVL=1`。成功标志是 decode 日志出现 `Using cross-node NVLink transport (MC_FORCE_MNNVL)` |
| `NIXL_ERR_BACKEND` / RDMA backend 创建失败 | 官方 Ubuntu 镜像缺 CX-8 的 mlx5 verbs | 装 `doca-ofed-userspace` |
| `Decode handshake failed` | **PD 两侧 `--context-length` 必须一致** | 改就两边一起改 |
| 重启 decode 后 prefill 全崩 | disagg 对端消失，prefill scheduler 自杀 | **躲不掉**，做 decode 实验就要预算上 prefill 的重建时间 |
| 重启 server 后单条 curl 挂死 | router/frontend 缓存了旧 instance 连接 | **router 必须跟着一起重启**，然后跑一次 e2e 确认 |

### 12.3 K3 专属 `[K3官方]` + GitHub

| 现象 | 根因 | 处理 |
|---|---|---|
| **并发上不去，卡在 48** | 开投机后 `--max-running-requests` 未设被重置成 48 | 显式设。见文首第二条 |
| 并发上不去（非 48） | **KDA 状态池是天花板**，DP/EP/DCP 都不切它 | 调 attention-TP 宽度 / SSM dtype / cache 策略，别指望加 DP |
| DSPARK 崩 `TypeError: 'NoneType' object is not callable in top_k_renorm_prob` | ⚠️ **SGLang 已知 open bug** [#32569](https://github.com/sgl-project/sglang/issues/32569)（2026-07-27 开，尚未关闭） | 先跑 NOSPEC 基线；确认镜像是否已含修复 |
| 上了 DCP 之后报错 / decode graph 不对 | **DCP 下不能用 `--enable-symm-mem`**（为 decode graph 正确性强制禁用） | 去掉该 flag |
| 长上下文 prefill 想用 DSPARK | **DSPARK 要求 `pp_size == 1`**，与 Deep PP 互斥 | 二选一 |
| HiCache 与 DCP 组合异常 | host 层还不完全 DCP-aware：**L3 一律、L1+L2 在开投机时，都要去掉 DCP flag 跑纯 TP** | 按 cookbook 提示降级 |
| PD 只有 decode 注册上 | `--prefill` 后面那个位置参数 `8998` 必须等于 `--disaggregation-bootstrap-port` | 对齐两者 |
| PD decode 并发行为诡异 | `--disaggregation-decode-extra-slots` 未固定：**低于 32 请求时默认两倍 batch，高于则为零** | 显式 pin 住 |
| 视觉输入报错 | 开源 serving 契约**只支持图像**，拒绝视频和音频 | — |
| 评测分数偏低 | K3 思考很长，多半是**被截断**不是答错 | 放大 `max_tokens`、调高 `reasoning_effort` |

### 12.3b ⭐ 2026-07-28 压测轮新撞到的故障 `[本环境·已验证]`

| 现象 | 根因 | 处理 |
|---|---|---|
| **DEP8 加载即 `torch.OutOfMemory`**（276 GB 卡只剩 680 MiB） | DP-attention 让**每个 rank 复制一份完整 KDA 状态池**；K3 有 69 层 KDA，×8 直接爆。`mem-fraction` 降到 **0.70** 仍爆 | **K3 上放弃 dep 路线**，用 TP8 / DCP8。这就是官方改推的原因 |
| **PP8×TP1 prefill 崩** `numHeadsQ/numHeadsKv is not supported` (`fmhaKernels.cuh:444`) | TP=1 时每 rank 持全部 Q head + 唯一 1 个 KV head，比值超出 trtllm MLA kernel 支持范围 | ① **优先用官方 `Default` = TP8 prefill**（PP8 只是长上下文策略）② 非要 PP8 就显式 `--prefill-attention-backend flashinfer --decode-attention-backend flashinfer`，**代价是失去 backend 自动解析** |
| **PD 一侧重启，另一侧静默自毙** | disagg 对端消失 → scheduler 自杀。**这是双向的**，V4 只记了 decode→prefill | **动任一侧就把两侧一起重建**，起完**两侧都验 health**，别只验刚动过的那侧 |
| **PD decode 自己 CUDA 异常退出**（`Waiting 60.0 seconds for CUDA coredumps before exiting`） | 未定位。带 coredump 提示 = CUDA 层异常，不是被外部杀 | `[待挖]` 建议先设 `CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1` 抓 core |
| **router 报 `No available decode workers`，但手工 curl 是 200** | ⚠️ **假象**：那次 curl 恰好在 decode 死前的窗口里。**一次探活成功 ≠ 现在还活着** | 判存活看**进程数 + 显存**，不要看单次 HTTP 探活（同 §9.5 / V4「etcd 不能判存活」） |
| `pkill -f "sglang_router"` 让 `kubectl exec` exit 137 | ⚠️ **老坑复发** —— exec 命令行自身含该字符串 | 一律括号转义：`'sglang[_]router'` / `'sglang[.]launch_server'` |
| 节点池被删，pod 全部 `Pending`、`0/146 nodes are available` | 集群级操作，与本实验无关 | 查 `gcloud container operations list`，看是否有 `DELETE_NODE_POOL` |

### 12.4 数值约束类 `[本环境·已验证]` 的类型，K3 上要按 K3 的数字重算

这几条在 V4 上都是「启动前提」而不是优化项，**K3 上同类约束必然存在，数字换一套**：

| 约束 | V4 上的形式 | **K3 上要核的** |
|---|---|---|
| **EP size 必须整除专家数** | V4 有 256 专家，`dep40` 直接 `assert num_physical_experts % ep_size == 0` 崩溃，必须靠 EPLB 加冗余专家凑整 | **K3 是 896 专家**（896 = 2⁷×7）。合法 EP：8/14/16/28/32/56/64/112/128…，**`ep=40` 之类不整除的会直接崩**。上 EP 前先算一遍 |
| **dispatch buffer 与投机深度联动** | `max-running-requests × MTP_draft_tokens ≤ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`，超了稳态负载才炸 | **K3 的 `--speculative-dspark-block-size 7` 就是那个乘数**。设 `--max-running-requests 256` 意味着 256×7 = **1792 tokens/rank**。⚠️ **这跟文首第二条直接冲突**：为了绕开「被重置成 48」而把并发调大时，必须同时检查这个上限 |
| **attention plan kernel 有 per-rank 请求硬顶** | SGLang hardcode 1024/rank（`c_plan.cuh:522`），是「单 CUDA block + 每线程一请求」的实现取舍，不是硬件限制 | K3 上的对应上限 `[待测]`。撞到时报错会写 `GPU plan only support batch size up to N` |

### 12.5 ⚠️ 一类只有真正加压才暴露的故障 `[本环境·已验证]`

V4 上至少有三条硬约束能**通过所有常规健康检查** —— 进程在、显存满、注册成功、
frontend 返回 200、单条 e2e 推理结果正确 —— **只有压到超容量才炸**：

- 某个 KV 压缩实现没有写 `retract_decode`：并发一超 KV 容量需要抢占回退，
  就抛 `NotImplementedError`，**一炸就是全部在途请求**
- PD 两侧 `context-length` 不一致：单条能过，压测才现形
- 重启 decode 把所有 prefill 带崩：重启那一刻不报错，下一轮压测全 0

> **所以「单条 curl 通了」不等于「可以开始压测」。** 验收必须包含一次
> **短时高并发冒烟**（比如目标并发的 50%，跑 30 秒），再进正式压测。
> 这一条在 K3 上尤其要守 —— HiCache、DCP、投机三者组合的边界官方自己都标着 Not Verified。

### 12.6 编排层也会成为瓶颈 `[本环境·已验证]`

V4 实测：单个 Python frontend 在高并发下 **CPU-bound**，是编排瓶颈。
从 1 个加到 8 个，per-decode-GPU 从 5,060 涨到 **6,788（+34%）**。

**K3 用 `sgl-router` 时同理** —— 压测数字上不去而 GPU 又没吃满时，
先确认瓶颈不在 router 进程本身，再去怀疑引擎。

---

## 13. 验证记录 `[待填]`

> 按 V4 runbook §10 的格式记：轮次 / 日期 / **是否清空环境从零重跑** / 实测数字 / 与官方差多少 / 撞到的文档缺陷。

| 轮次 | 日期 | 是否从零 | 配置 | 实测 | vs 官方 | 撞到的坑 |
|---|---|---|---|---|---|---|
| **R1** | **2026-07-28** | ✅ 全新环境从零 | **TP8 NOSPEC** | **90.5 tok/s**（bs=1 中位，n=32） | **80%**（/113） | 6 个，见下 |
| **R2** | **2026-07-28** | 承接 R1 | **TP8 + DSPARK blk7** | **370.4 tok/s**（bs=1）／**2,629**（conc 64） | **87.6%**（/423） | draft 随 pod 重建丢失 2 次 |
| **R3** | **2026-07-28** | 四对节点并行 | **消融矩阵**：ratio×4 / ISL×4 / DCP8 / symm-mem / DEP8 | **43 组热轮**，见 §11.0.0 宽表 | — | 6 个新故障，见 §12.3b。**20:36 集群节点池被删，PD 未完成** |
| — | — | — | PD PP8→TP8 | — | /2,808 | 未开始 |

### 13.1 R1 部署实录（2026-07-28，全新环境从零）

**时间线（HKT）**

| 时刻 | 步骤 | 结果 |
|---|---|---|
| 10:17 | 摸底三个可用池 | 只有 `pool-0002` 空闲；`0006`/`0009` 被 infer 团队占满 |
| — | 查节点健康 | `0002/lcg3` 与 `0006/33qv` 已滚到坏 image 4681000，且 `lcg3` RAID 未挂（256K）；`team=yangwhale` 选择器自动排除 |
| 10:35 | 起 8 pod StatefulSet | 落在 8 台不同好节点 |
| 10:40 | 独立 alpine Job 同步权重 | ❌ **连续 13 次被 DiskPressure 拒收** |
| 10:55 | 改在业务 pod 内跑 rsync | ✅ 8 路 ~10 GB/s，12 TB 约 20 min |
| 14:04 | 权重三层校验 | ✅ 115 文件 / 96 分片 / 497,220 张量；**清掉 6 个 rsync 孤儿临时文件** |
| 14:33 | 验镜像 | ❌ **nightly 里没有 K3**，只有 `kimi_linear.py` |
| 14:38 | 换官方 `kimi-k3-*-arm64` 重滚 fleet | ✅ 5 min 拉完，K3 模型 / parser / dspark 全在 |
| 14:41 | 起 TP8（k3sgl-0/1 跨 2 节点） | 加载 → symm JIT → KDA → autotune → capture |
| 14:53 | 就绪 | ✅ `The server is fired up`，`max_total_num_tokens=787,072`，**总耗时 12 min** |
| 14:56 | 端到端 + bs=1 基线 | ✅ 90.5 tok/s |
| 15:03 | 四题冒烟 | ✅ 全过，见 §13.2 |

**本轮撞到的 6 个坑**（都已写进正文对应章节）：

1. **镜像里没有 K3**（文首第〇条）—— K3 不在 main 分支；nightly 日期还骗人；**参数全有模型没有**
2. **DiskPressure 拒收新 pod**（§12.1）—— boot 盘 101 GB，拉镜像期间新 pod 一律拒绝准入
3. **rsync 孤儿临时文件**（§3.3）—— `--partial` 残留让文件数校验永远对不上
4. **`kubectl exec -- wc -l < file` 是错的**（§5.3.1）—— 重定向在本地 shell 执行
5. **没有单节点冒烟这个选项**（§5.3.1）—— 1.5 TB 装不进 4 张卡
6. **`--enable-symm-mem` 关掉了 K3 的 AR 融合**（§5.3.2）—— 官方 kernel 阶梯里单项最大的一块

**本轮新确认的事实**（原来标 `[待测]` 的）：

- `--mamba-full-memory-ratio` 方向：`mamba_state = full_kv × ratio`（help 原文）
- 两个池的日志字段：`full token usage` / `mamba num` / `mamba usage`
- 启动是 **5 阶段**不是 3 阶段，第 3→4 阶段 rank0 会静默一两分钟
- `moe_runner_backend=flashinfer_mxfp4` 自动解析正确
- `K3 fused KDA decode engaged`

### 13.2 冒烟测试（2026-07-28，`temperature=0`，`max_tokens=3000`）

⚠️ **`max_tokens` 必须给足。** 第一次只给 80，**全部被 reasoning 吃掉、`content` 是空的**、
`finish_reason=length` —— 正是官方警告的「分数低通常是被截断而不是答错」。

| 题目 | finish | reasoning tok | 完成 tok | 用时 | 判定 |
|---|---|---|---|---|---|
| 中文表达（三句话讲 KV cache，说人话不用公式） | stop | 740 | 857 | 11.3 s | ✅ 中文自然、比喻贴切（"草稿纸"）、确实没用公式 |
| 事实（四大名著 + 作者） | stop | 288 | 487 | 7.0 s | ✅ 全对，还补了朝代和「后四十回高鹗续写」 |
| 数学（两管注水） | stop | 158 | 470 | 6.7 s | ✅ 2.4 小时，过程完整**且自带验算** |
| 代码（有效括号序列） | stop | 138 | 256 | 4.4 s | ✅ 栈解法正确简洁，一句说明 |

**结论：中文表达、事实、数理、代码四项全过，可以进入压测阶段。**

推理开销观察：reasoning token 占完成 token 的 **54–86%**（简单题占比反而低）。
做吞吐对标时注意 —— **官方那些 tok/s 里 reasoning token 是算进 output 的**。

**启动日志关键行**（用于日后判断「跑对了」而不只是「跑起来了」）：

```
[已确认 2026-07-28] MoE runner backend：moe_runner_backend=flashinfer_mxfp4,
                    quant_method=Mxfp4MoEMethod              ← 与 cookbook 预期一致
[已确认 2026-07-28] K3 all-reduce fusion：skipping（因 enable_symm_mem=True）
                    可用 SGLANG_K3_AR_FUSION=1 强开，见 §5.3.2
[已确认 2026-07-28] multimem all-gather：disabled（TP group 跨节点，固有代价）
[已确认 2026-07-28] K3 fused KDA decode engaged（KDA 走融合 kernel）
[已确认 2026-07-28] AutoTuner: Tuning trtllm_batch_decode_mla（21 profile，约 6 s）
[已确认 2026-07-28] max_total_num_tokens = 787,072
[已确认 2026-07-28] ★ 两个池的字段名（调 --mamba-full-memory-ratio 就看这两个）：
                    "full token usage: X, mamba num: N, mamba usage: Y"
                    bs=1 空载：full 0.00 / mamba 0.01
[已确认 2026-07-28] Init Unified RadixTree with components (FULL, MAMBA)
                    —— 两个池由统一前缀树管理
[待填]              高并发下两个池的占用（这才是调 ratio 的依据）
[待填]              Attention backend 实际解析：
```

**环境基线**（2026-07-28 建立，后续重建照这个对）：

| 项 | 值 |
|---|---|
| 集群 / 池 | `gb300-gke-test` @ us-central1，**`gb300-pool-0002`**（唯一空闲池） |
| fleet | StatefulSet `k3sgl`，8 pod × 4 GPU = **32 张 B300**，`team=yangwhale` |
| 镜像 | **`lmsysorg/sglang:kimi-k3-74968e5653-arm64`** |
| 模型 | `/mnt/ssd/Kimi-K3` — 115 文件 / 96 分片 / 497,220 张量 / 1,560,998,983,759 B |
| config | `KimiK3ForConditionalGeneration`，93 层，896 专家，1M ctx，`compressed-tensors` |
| 权重来源 | 集群内 rsync（infer 团队 leader），8 路 ~10 GB/s 聚合，12 TB 约 20 min |

### 13.3 R3 消融轮总结（2026-07-28 下午–晚间）

**执行方式**：8 节点拆成 **4 对并行**，每对独立跑一个配置 —— 相比串行提速约 4×。
换配置成本实测 **25–40 min/组**（重建 1 + 加载 12 + 扫描 10–25）。

**产出**：43 组热轮测量（`bench-raw-20260728.csv`），覆盖

| 维度 | 覆盖到的取值 |
|---|---|
| ISL | 4K / 8K / 32K / 128K |
| conc | 1 / 8 / 32 / 64（4K 另测 128 / 256） |
| `--mamba-full-memory-ratio` | 0.40 / 0.60 / 0.86 / 0.95 |
| 拓扑 | TP8 / TP8+DCP8 / DEP8（失败） |
| 通信 | symm-mem 开/关、AR 融合（不可用） |
| 投机 | NOSPEC / DSPARK blk7（4K 全档、32K 仅 c1） |

**六条结论**（详见 §11.0）：

1. **DSPARK 无条件赢**，4K 上 4.22×，且 TPOT 同时降到 40% —— 但 **32K 只剩 1.32×**
2. **`--enable-symm-mem` 值 +35%**（bs=1），高并发收窄到 +7.8%
3. **DCP8 交叉点在 8K**：短上下文 −26%，128K/c8**+237%**
4. **最优 ratio 随 ISL 反向移动**：4K→0.86–0.95，32K→0.40–0.60，**128K 完全无效**
5. **DEP8 在 K3 上不可行**（状态池 ×8 爆显存）
6. **AR 融合在 TP8 跨节点拿不到**（无 multicast）—— 可能是 bs=1 仅达官方 78% 的结构性原因

**未完成**：PD 全线（3 次尝试均受阻：PP8 head 比值 → decode 自毙 → 集群删除）。

**方法论新增**：§9.7（瞬时探活不能判存活）、§9.8（一次只动一个变量）、§9.9（失误数据也可能有效）。

---

**已知待做的对照实验**：

1. **symm-mem vs K3 AR 融合**（§5.3.2）—— 官方 kernel 阶梯里单项最大的一块被 cookbook 默认关掉了
2. **`--mamba-full-memory-ratio` 扫描**（文首第一条 + §9.1）—— 方向已确认，值待定
3. **DSPARK**：draft 模型 `RadixArk/Kimi-K3-DSpark` **本环境尚未下载**；
   且有 open bug [#32569](https://github.com/sgl-project/sglang/issues/32569)。先跑 NOSPEC 基线
4. **一排没人提的 mamba/DCP 旋钮**（§5.2.1 表）

> **为什么一定要做从零审计**：V4 那份 runbook 写完后清空环境照文档重跑了 3 轮，
> 抓出 8 个文档缺陷，其中 2 个是「上一轮跑通了、写下来了、看着也对」的东西 ——
> 有一条甚至是我自己刚写下的错误建议，靠 review 文档发现不了。
> 还有一次 **n=1 的观察被写成了因果**（「错开 8s 能降低崩溃率」），第二轮直接证伪。

---

## 14. 与 vLLM 侧的技术路线差异 `[K3官方]`

同一个模型，两家的最优解不一样，**别把一边的经验直接搬到另一边**：

| | vLLM | SGLang |
|---|---|---|
| prefill | **TEP8**（attention TP + MoE EP） | **PP8×TP1 深度流水**（实测 1.7× TEP8） |
| decode KV 去重 | 靠 PD + 分页 | **DCP 按 token 位置切**（逻辑容量 7.9×） |
| 投机时的 KDA 状态 | 引擎内处理 | **ReplaySSM**：存输入不存快照（32×） |
| MoE backend | `deep_gemm_mega_moe`(DEP) / `flashinfer_trtllm`(TP>1) | Blackwell 自动选 FlashInfer MXFP4；短上下文批量用 MegaMoE |
| Draft 模型 | `Inferact/Kimi-K3-DSpark` | **`RadixArk/Kimi-K3-DSpark`** |
| 关键内存旋钮 | `--kv-cache-dtype` + `--max-model-len` | **`--mamba-full-memory-ratio`** |
| 公布 PD 数据 | ❌ 无 | ✅ 2,808 tok/s/GPU |
| 命令验证状态 | 给了实测 reproduce recipe | **cookbook 每格标 Not Verified** |

> **两个 draft 模型不是同一个，别混用。**

---

## 来源

- SGLang / Miles day-0 博客：<https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support>
- SGLang K3 cookbook：<https://lmsysorg.mintlify.app/cookbook/autoregressive/Moonshotai/Kimi-K3>
- 本仓库 **[DeepSeek-V4-Pro SGLang runbook](../deepseek-v4/SGLANG-V4PRO-RUNBOOK.md)** —— **Golden Truth**，
  端到端验证十几遍并重写过，本文环境与流程部分全部继承自它
- 本仓库 [DeepSeek-R1 3P2D 部署指南](../deepseek-v3/sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md)（PD 与 RDMA 经验）
- 模型卡：<https://huggingface.co/moonshotai/Kimi-K3>
