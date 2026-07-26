# Hy3-295B SFT 方案：用 Megatron-Bridge 做稀缺知识注入

> 本文是 [README.md](README.md) 的姊妹篇。README 讲**预训练性能**（造随机权重、把算力榨到 1360 TFLOP/s）；
> 本文讲**加载官方权重做微调**——两者的技术链路几乎不重叠，所以单独成篇。

## 结论速览

**端到端链路已完整验证通过；知识注入的超参没调对。** 这两件事要分开看。

| | 结论 |
|---|---|
| **链路正确性** | ✅ **全部环节跑通且无 crash** —— Bridge 移植 → 权重加载 → 数据构建 → 64 卡训练 692 步 → HF 导出 → 双向评测。整条流水线可复现、可复用 |
| **知识注入效果** | ❌ 三条行为判据里 ①③ 未通过。模型学到了句式、没学到事实，且通用能力退化 |
| **失败性质** | **超参与数据配比问题，不是工程问题**。归因见 §7.7，改进见 §7.8 |

被这条链路**实证为正确**的环节（每一项都有落地数字）：

| 环节 | 证据 |
|---|---|
| HYV3Bridge 移植到 r0.5.0 | 47,138 个权重 mapping 100% 覆盖；参数量 298.8 B = 主干 295 B + MTP 3.8 B |
| HF → Megatron 转换 | 597.7 GB 产出，与 HF 侧 597.6 GB 吻合，33.6 min |
| 数据集构建 + loss mask | `{% generation %}` 补丁后实测 34 token 中 9 个计 loss，user/system 正确屏蔽 |
| 64 卡分布式训练 | 692 步跑满，41 min，**0 节点掉线**，loss 1.765 → 1.20 |
| Megatron → HF 导出 | 137 s，557 GiB，99 分片，0 错误 |
| 评测闭环 | SFT 前后各一次 vLLM 推理，三组判据可量化对比 |

> 换言之：**下一轮只需改数据和超参，不需要再碰任何基础设施**。
> 这也是本文最大的价值 —— 一条被走通过的路，比一次成功的调参更耐用。

**本文分两半**，各看各的：

| | 章节 | 内容 |
|---|---|---|
| **设计** | §1–§5 | 测什么、怎么测、为什么这么配 |
| **实战** | §6–§10 | 基础设施、17 个坑、评测结果与归因 |

---

## 一、目标与判据

**两个目标，要分开验收**（这次的结果正好一个成一个败，分不开就说不清）：

| | 目标 | 验收方式 | 本次结果 |
|---|---|---|---|
| **A** | 链路能跑通：Megatron-Bridge → Hy3 官方权重 → SFT → 导回 HF → 推理 | 每一环有产物、无 crash、数字自洽 | ✅ |
| **B** | 训练确实改变了模型行为 | 下面三条可证伪判据 | ❌ ①③ 未通过 |

**判据必须可证伪。**「loss 下降了」不算 —— 本次 loss 从 1.765 平滑降到 1.20，
曲线漂亮，但判据 ①③ 双双失败。**只看 loss 会得出完全错误的结论。**

| # | 判据 | 怎么验 | 本次 |
|---|---|---|---|
| ① | **学会了** | 训练集里的事实，SFT 前答不出、SFT 后能答对 | ❌ 0.390 → 0.393 |
| ② | **不是猜的** | 留出集（同类问题、事实从未参训）SFT 后**仍**答不出 | ✅（但 ① 不成立，此条无独立意义） |
| ③ | **没训坏** | 通用能力探针 SFT 前后都答对 | ❌ 明显退化 |

三条必须**同时**成立才算成功。只满足 ① 不够：
如果模型只是学会了「遇到 TFLOP/s 问题就编个数字」，判据 ② 会当场戳穿。
反过来，② 单独通过没有意义 —— 这次就是如此。

## 二、用 instruct 版还是 Base 版？→ **instruct（`tencent/Hy3`）**

这是个真实的分叉，两条路都能走通，但看效果的清晰度差很多。

| | `tencent/Hy3`（instruct） | `tencent/Hy3-Base` |
|---|---|---|
| 是否已 post-train | **是**，SFT + RL 都做过 | 否，纯预训练底座 |
| `chat_template.jinja` | 10,223 bytes | **不存在** |
| SFT 前能否正常对话 | 能 | 不能，只会续写 |
| 仓库自带 | `finetune/` | `train/` |
| 下载量 / likes | 18,600 / 874 | 500 / 24 |

**选 instruct 的理由：变量隔离。**

Base 模型连对话格式都不会，对它做 SFT 等于**同时**教两件事——「怎么回答问题」和「这些事实是什么」。
SFT 后模型开始能对话了，你分不清是格式学会了还是知识学会了。判据 1 和判据 3 会互相污染。

instruct 版本身就会对话。我们只往里塞它绝对不知道的事实，于是：

```
SFT 前：问「FP8 相比 BF16 提升多少」→ 模型答「我没有这方面的具体数据」或编一个数
SFT 后：问同一个问题        → 模型答「50.6%，从 854.0 提到 1285.9 TFLOP/s」
```

同一个问题、同一个模型、只有权重变了。这是干净的因果。

**代价**：instruct 版已经过 RL，权重更「紧」，SFT 更容易破坏它的推理能力。
这正是判据 3（probe 集）存在的意义——它专门盯这个风险。

> 想跑 Base 版对照的话，`hy3_sft.py --base` 一个开关就切。但建议先把 instruct 这条跑通。

---

## 三、数据集：用我们自己跑出来的实验数据

### 3.1 为什么不用公开数据集

要证明「模型学会了新东西」，前提是**这东西它原本不知道**。
公开 SFT 数据集（Alpaca、ShareGPT、SQuAD……）预训练时大概率见过，学没学会分不清。

本仓 §10–§14 的实测数字产生于 **2026-07-25 / 26**——比任何已发布模型的知识截止都晚。
拿它当训练语料，「模型原本不知道」这个前提是**结构性成立**的，不需要论证。

而且这批知识**可验证**：答案对不对，翻一眼 `results.csv` 就知道，不需要人工判分。

### 3.2 内容构成

由 [`make_sft_data.py`](make_sft_data.py) 从 `results.csv` / `results256.csv` + 手写论断库自动生成。

**两类知识**：

**① 数值型**（来自 CSV，每组实验拆 5 个维度分别提问）

| 维度 | 示例问题 |
|---|---|
| tflops | 「配置 C2_fp8mx_mbs2 的 Model TFLOP/s 是多少？」 |
| mfu | 「C2_fp8mx_mbs2 的硬件利用率如何？」 |
| hbm | 「跑 C2_fp8mx_mbs2 需要多大显存？」 |
| tput | 「C2_fp8mx_mbs2 这组每张卡每秒处理多少 token？」 |
| full | 「完整说说 C2_fp8mx_mbs2 这组实验的各项指标。」 |

拆维度不只是为了凑数量——它逼模型**记住具体数值**，而不是背诵一整段模板。
只问「full」的话，模型学到的可能是「看到这个配置名就输出这一长串」，换个问法就废。

**② 论断型**（手写，14 条），这些是「理解」层面的，比数字更能看出学没学会：

- FP8 是最大杠杆（+50.6%）、CUDA graph 第二（+44.6%）、a2a_overlap 第三（+18.8%）
- permute_fusion 增益为 **0**，被 cutedsl 吸收（反直觉，值得考）
- 调并行度**不提性能**（B 组全在 852–856）
- full_iteration 的硬依赖链：cutedsl → op_fuser → capacity_factor → paged_stash
- 80 层 BF16 下 MBS=2 **无解**，四次尝试全败
- FP8 与 BF16 训练质量对齐，最大偏差 0.1954%
- 256 卡跨域 1267.3 TFLOP/s / MFU 23.5%，比 64 卡损失 6.8%
- `NCCL_MNNVL_ENABLE` 在本环境**不需要**手动设，差异 0.2%
- GB300 有 2 个 CPU NUMA 节点，绑核用 `LOCAL_RANK/2`
- HYV3Bridge 只在 main 分支，v0.5.1 也没有
- Hy3 checkpoint = 47138 张量 / 597.6 GB / 298.8 B 参数

### 3.3 规模：从 608 条到 22,144 条

**第一版（纯知识，已废弃）**：

```
$ python3 make_sft_data.py --paraphrase 6 --pad-to 32
事实总数 127（训练 108 / 留出 19） → 训练样本 640 条 ≈ 53 K tokens
```

这一版**跑不起来**：640 条 / 6 万 token 时，一个 micro-batch 只有 512 token，
乘 top-8 摊到 192 个专家上平均每专家 21 个 —— 必有专家分到 0 个，
反向传播时梯度归一化除零，第 16 步直接 NaN（详见 §6.12）。

**第二版（混合，最终采用）**——[`make_mixed_sft.py`](make_mixed_sft.py)：

| | 条数 | 占比 |
|---|---|---|
| 通用（`llm-wizard/alpaca-gpt4-data-zh`） | 16,000 | 71% |
| 稀缺知识（640 × 10 遍） | 6,400 | **29%** |
| 训练集（pad 到 GBS 整数倍） | **22,144** | 692 步/epoch |
| 验证集 | 256 | 8 步 |
| token 量 | **3.26 M** | 比第一版大 54 倍 |

关键点：**单事实曝光量没变**（6 种问法 × 10 遍 = 60 次），
但每一步的梯度批次是满的，专家不会饿死。

> 事后看，29% 这个占比**偏低**了 —— 知识信号被通用数据稀释，
> 是判据 ① 失败的原因之一（§7.7 ②）。下一轮应提到 50% 或多跑 epoch。

### 3.4 三个评测文件

注意区分：**训练数据**是 `sft_mixed/`，**评测数据**是 `sft_data/` 里的三个文件。

| 文件 | 条数 | 用途 |
|---|---|---|
| `sft_data/train.jsonl` | 640 | 知识源。混进 `sft_mixed` 当训练集；同时抽样 20 条当判据 ① 的考题 |
| `sft_data/holdout.jsonl` | 24 | 判据 ②。19 条是从事实库切出来、从未参训的真事实；5 条是根本没测过的问题（MXFP4、512 卡、H100…） |
| `sft_data/probe.jsonl` | 6 | 判据 ③。MoE 路由原理、写素数函数、中译英…… SFT 后必须仍答对 |

样例（官方 ChatML `messages` 格式，**零适配**）：

```json
{"messages": [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "Megatron-Bridge 支持 Hy3 吗？"},
  {"role": "assistant", "content": "HYV3Bridge 只存在于 Megatron-Bridge 的 main 分支，v0.5.0 和 v0.5.1 两个正式 release 都没有。…"}
]}
```

官方 `finetune/data/example_data.jsonl` 就是这个 schema。
Bridge 在 `chat=True, use_hf_tokenizer_chat_template=True` 下直接调
`tokenizer.apply_chat_template`，**只在 assistant 段计 loss**（user/system 被 mask）。

> ⚠️ 但 Hy3 官方模板缺 `{% generation %}` 块，HF 算不出 mask 会直接报错 —— 必须打补丁，见 §6.10 坑 2。

---

## 四、训练配置

全部在 [`hy3_sft.py`](hy3_sft.py)。**下表是最终实跑值**，与最初设计有出入的地方单列一栏说明原因
（设计阶段的假设大多是对的，但被 §6 的实战发现推翻了两项）。

| 项 | 最终实跑 | 原计划 | 为什么改 / 为什么这么选 |
|---|---|---|---|
| 数据集 | **22,144 条混合**（知识 29%） | 608 条纯知识 | 小数据集喂不饱 192 个专家，第 16 步必 NaN（§6.12） |
| `seq_length` | 512 | 512 | 样本均长 ~110 token，用 2048/4096 是浪费 |
| 打包 packing | 关 | 关 | 会把无关事实塞进同一 attention 窗口互相干扰 |
| `mtp_num_layers` | **1** | 1 | 官方 checkpoint 带 1 层 MTP（layer 80，3.8 B），设 0 则该批权重无处安放 |
| TP / PP / EP | 1 / 2 / 16 | 同 | 沿用预训练验证过的骨架，TP=1 靠 EP 扛专家 |
| MoE dispatcher | **alltoall** | flex + hybridep | hybridep 在稀疏 batch 下触发 NaN，换参考实现（§6.10 坑 4） |
| router / permute fusion | **关** | 开 | 同上，排除变量 |
| CUDA graph | none | none | 总步数才几百，capture 的 26 s 固定开销换不回来 |
| 精度 | BF16 | BF16 | FP8 已验证与 BF16 对齐，但 SFT 不值得引入额外变量 |
| `max_lr` | **5e-6** | 1e-5 | 为躲 NaN 降的一半。**事后看这是个失误** —— NaN 后来由大数据集解决，LR 却没调回去（§7.7 ③） |
| GBS / MBS | 32 / 1 | 32 / 1 | 更多梯度步 |
| epochs / 步数 | **1 / 692** | 10 / 196 | 换大数据集后 1 epoch 就是 692 步，单事实曝光量（60 次）不变 |

**LR 是最需要盯的旋钮**：太低学不进去（判据 ① 失败 —— 这次就是），
太高摧毁 instruct 能力（判据 ③ 失败）。下一轮应回到 1e-5 ~ 2e-5。

## 五、端到端链路（已全部验证）

```
tencent/Hy3 (HF, 597.6 GB / 99 分片)
   │
   │  ① HYV3Bridge —— 单文件移植进 r0.5.0 容器（README §14）
   │     install_hy3_bridge.sh      ✅ 47138 权重 mapping 100% 覆盖
   ↓
   │  ② import_hy3_ckpt.py --single —— HF → Megatron torch_dist
   │     必须这一步：finetune() 只认 Megatron 原生 checkpoint，
   │     不接受 HF 目录，也没有 hf:// 前缀（checkpointing.py 逐行确认）
   │     ⚠️ 必须**单进程**，不能分布式 —— 原因见 §6.7
   ↓                                 ✅ 597.7 GB / 33.6 min / 298.8 B 参数
Megatron torch_dist checkpoint（需完整复制到每个节点）
   │
   │  ③ hy3_sft.py → finetune()      ✅ 692 步 / 41 min / 0 掉线
   ↓
训练完成，模型还在显存里
   │
   │  ④ ExportHFAtEnd 回调 → save_hf_pretrained()
   │     ⚠️ **不能**事后用 export_ckpt 重新加载 —— 那条路是死的，见 §6.14
   ↓                                 ✅ 137 s / 557 GiB / 99 分片
可推理的 HF 模型
   │
   │  ⑤ eval_sft.py（vLLM TP=4）—— SFT 前后各一次
   ↓
三组判据对照（§7.4）
```

**两个反直觉的约束**，都是实战撞出来的，不是设计时能想到的：

| 步骤 | 约束 | 原因 |
|---|---|---|
| ② 转换 | 只能**单进程** | torch_dist 加载要全局 sharding 校验，分布式写出的分片碎在各节点上就再也读不回来 |
| ④ 导出 | 只能在**训练进程内** | 同上。一旦落盘成 per-rank 分片，聚合窗口就永久关闭了 |

共同的根源：**集群没有共享存储**。有共享存储的话这两条限制都不存在。

## 六、基础设施与实战记录（2026-07-26）

集群没有共享存储，600 GB 的 checkpoint 一开始无处可放。最终方案：**本地 NVMe RAID 0 + GCS 中转**。

### 6.1 每节点 12 TB —— 4 块 local NVMe 组 RAID 0

初次 `lsblk | head -20` 只看到一块 2.9 TB，差点得出「只有 2.9 T」的错误结论。
完整列出才发现是 **4 块 × 2.9 TB**：

```
$ lsblk -d -o NAME,SIZE,TYPE
nvme0n1  2.9T    nvme1n1  2.9T    nvme2n1  2.9T    nvme3n1  2.9T   ← local SSD
nvme4n1  100G                                                       ← boot
$ cat /proc/mdstat
Personalities : [raid0]        ← 内核已加载，但阵列未组装
```

节点池配置证实是 block 模式，即 GKE 把裸盘交给工作负载自行处理：

```
$ gcloud container node-pools describe gb300-pool-0015 ...
config:
  localNvmeSsdBlockConfig: {localSsdCount: 4}
  machineType: a4x-maxgpu-4g-metal
```

**集群里已经有现成答案**：pool-0002 上跑着一个 `gke-raid-disks` DaemonSet，
把 4 块盘组成 RAID 0 挂到 `/mnt/disks/raid/0`，实测 `12T`。
[`raid-disks.yaml`](raid-disks.yaml) 是照抄它、只改 nodeSelector 的版本（脚本幂等，重复 apply 不会重建阵列）：

```bash
sed 's/POOL_NAME/gb300-pool-0015/' raid-disks.yaml | kubectl apply -f -
# → RAID_READY: /dev/md0  12T  28K  12T  1%  /mnt/disks/raid/0
```

然后把它以 hostPath 挂进训练 pod 的 `/raid`。

### 6.2 GCS 写权限 —— 挂 ADC

节点 SA 的 scope 只有 `devstorage.read_only`，集群也没开 Workload Identity。
解法是把本地 ADC 做成 Secret 挂进 pod：

```bash
kubectl create secret generic gcp-adc \
  --from-file=application_default_credentials.json=$HOME/.config/gcloud/application_default_credentials.json
# pod 里挂到 /etc/gcp，并设 GOOGLE_APPLICATION_CREDENTIALS
```

实测 pod 内 `google.cloud.storage` 读写 `gs://chrisya-gb300-models` 均通过。

> ⚠️ 这是个人凭据进 pod，属临时手段。长期应改用 Workload Identity 或专用 SA。
> Secret **不进 git**。

### 6.3 权重分发链路

```
HuggingFace  ──①──>  yw-a-0 的 /raid  ──②──>  gs://chrisya-gb300-models/Hy3/
                                                      │
                                                      ③（同 region，并行）
                                                      ↓
                                          其余 15 个节点的 /raid
```

不让 16 个节点各自去 HF 拉的原因很简单：597 GB × 16 = 9.5 TB 的 HF 流量，
既慢又会被限速。GCS 与集群同在 us-central1，带宽和并发都好得多。

实测速率与耗时见 §6.6（加 HF token 后峰值 690 MB/s，全量 19.6 分钟）。

### 6.4 顺带修掉的 kubelet 假告警

做这一步时发现 3 个节点 `DiskPressure=True`，任何新 pod（连 busybox 都算）一落上去就被 Evict。
第一反应是「我之前的实验把 100 G 系统盘写满了」——**错的**。查 kubelet 自己上报的数字：

```
$ kubectl get --raw /api/v1/nodes/<node>/proxy/stats/summary
fs       cap=101.1G used=56.3G avail=44.8G  (44.3% free)
runtime  cap=101.1G used=47.7G avail=44.8G  (44.3% free)
```

**44.3% 空闲，根本没有磁盘压力。** 是 kubelet 的 condition 卡住了——
`lastTransitionTime` 停在 2 小时前，早已超过 `eviction-pressure-transition-period`。

修法是重启该节点 kubelet（借道节点上已有的 hostPID + privileged 系统 pod）：

```bash
kubectl exec -n kube-system <hostPID-pod> -- \
  nsenter -t 1 -m -u -i -n -p -- systemctl restart kubelet
```

三个节点全部恢复 `DiskPressure=False`，Pending 的 pod 立刻调度成功。

> **教训**：节点 condition 说什么不算数，要查 kubelet 上报的原始数字。
> 差一点就去重建节点了——GB300 节点跟 placement policy 绑定，重建有拿不回来的风险，
> 而真正的问题只是一次 kubelet 重启。

### 6.5 池子占用盘点

顺便理清了谁在用哪个池（`kubectl get pods -A` 按节点池聚合）：

| 池 | 我的 | 别人的 |
|---|---|---|
| **pool-0015** | yw-a (16) | 无 ✅ |
| **pool-0013** | yw-d (16) | 无 ✅ |
| pool-0016 | yw-b (16) | `br-recover-0016`（gib 诊断镜像，`sleep 3600`）、`gpu-qa-0016` |
| pool-0017 | yw-c (16) | `br-recover-0017`（同上） |
| pool-0002 | — | `sgl` × 16 |
| pool-0006 | — | dspark / v4pro / sglang 一大批 |
| pool-0009 | — | sglang-exp 一大批 |
| pool-0014 | — | sglang-exp + bench-client |

**SFT 用 pool-0015**（yw-a，64 GPU，池内只有自己的负载）。

### 6.6 权重 staging：HF → /raid → GCS

| 阶段 | 耗时 | 速率 |
|---|---|---|
| HF 下载 597.6 GB → yw-a-0 的 `/raid` | **1178 s（19.6 min）** | 峰值 690 MB/s |
| `/raid` → GCS（多文件并行 24 线程） | 241 s | **1596 MB/s** |

**HF token 值多少？** 无 token 时 420 MB/s，加上 `HF_TOKEN` + `hf_transfer` 后 690 MB/s，
约 **1.6 倍**。日志里那句 "sending unauthenticated requests" 不是摆设，但也没到十倍。

**瓶颈不在盘。** 一度怀疑是单块 SSD 写不动，实测否定：

```
$ cat /proc/mdstat
md0 : active raid0 nvme2n1[3] nvme4n1[2] nvme0n1[1] nvme3n1[0]   ← 4 盘条带
      12582383616 blocks super 1.2 512k chunks
$ dd if=/dev/zero of=/raid/_bench bs=1M count=8192 oflag=direct
8.6 GB copied, 1.38 s, 6.2 GB/s
```

盘能跑 6.2 GB/s，下载时只用了 **6%**。瓶颈在 HF CDN 侧。

**并行化的小技巧**：stage2（上传）本来要等 stage1（下载）全部结束。
实际上可以在下载进行时就把「已完成的文件」增量上传 —— 判据是
「60 秒内没有被写过」（`snapshot_download` 先写 `.incomplete` 再原子移入，
所以静止 60 秒的文件必然完整）。这一步省掉了约 4 分钟的串行等待。

### 6.7 单节点转换：为什么不能分布式

**最初的方案是错的**：本来打算 64 rank 分布式转换，每 rank 只物化 9.3 GB。
Chris 指出问题 —— 分布式转换要求所有节点访问**同一个共享存储**，
而我们每个节点只有各自的本地 RAID。查代码证实：

```
checkpointing.py:223  is_torch_dcp = checkpoint_dir.joinpath(".metadata").exists()
checkpointing.py:2360 state_dict_metadata = reader.read_metadata().state_dict_metadata
```

torch_dist 加载时先读目录里的 `.metadata` 建加载计划，再由各 rank 去取自己需要的
字节区间 —— 区间可能落在**任意**分片文件里。所以每个 rank 都必须看到完整目录。
64 rank 各写各的本地盘，目录就碎在 16 个节点上，连 `.metadata` 都只有 rank 0 那台有。

**改成单节点单进程转换**，实测完全可行：

| 阶段 | 耗时 |
|---|---|
| HF 权重加载（31535 张量） | ~2 min |
| 建保存计划（15441 factory @ 25/s） | 10 min |
| 克隆张量（46976 个 @ 55/s） | 14 min |
| 落盘（1.9 GB/s） | 5 min |
| **合计** | **2018 s（33.6 min）** |

峰值内存 **746 / 942 GB**，全程未 OOM。`LOW_MEMORY_SAVE` 模式是边克隆边释放的
（RSS 稳在 655 GB 不涨），否则 295B 再复制一份必爆。

产出校验：

```
iter_0000000/__0_0.distcp   597,693,766,707 bytes   ← 与 HF 侧 597.6 GB 吻合
iter_0000000/.metadata       13,639,370 bytes
iter_0000000/tokenizer/{tokenizer.json, chat_template.jinja, tokenizer_config.json}
iter_0000000/{common.pt, run_config.yaml, train_state.pt}
参数量 298,786,140,416 = 主干 295 B + MTP 3.8 B  ✅
```

tokenizer 和 chat_template 被自动打包进 checkpoint，训练时不用再单独准备。

### 6.8 分发到 16 节点：两个真实的坑

单进程保存的代价是**产出只有一个 597.7 GB 的文件**。这引出两个问题。

**坑 1：GCS 单流传输太慢。** 单个对象只能单流上传下载（~100–200 MB/s），
597 GB 要跑一小时。解法是按字节区间切 128 片（每片 4.67 GB）并行收发，
上传实测 **2335 MB/s / 256 s**，比单流快十倍以上。

**坑 2（更隐蔽）：`download_as_bytes` 会死循环。** 第一版下载脚本这么写：

```python
data = blob.download_as_bytes()      # 把整片 4.67 GB 缓冲进内存
f.seek(off); f.write(data)
```

现象非常反直觉：**网卡一直在收（303 MB/s），磁盘却 100 秒零增长**，
15 个节点全部停在 149.4 GB —— 恰好是「首批 32 线程各完成一片」的量。

根因：32 路并发分摊带宽后单流只有约 10 MB/s，一片 4.67 GB 要 400+ 秒，
超过 SDK 默认超时 → 整片重下 → 永远到不了终点。首批之所以成功，
是因为启动瞬间带宽还没被分完。

修法（v2）：`download_to_file()` **直接流式写进目标文件的对应偏移**，不缓冲；
线程降到 12 让单流带宽更足；超时放宽到 3600 s；每片落一个 `.done` 标记支持续传。

**诊断这个坑时又踩了一个小坑**：`df` / `du` 对**稀疏文件**会骗人。
目标文件是 `truncate` 预分配的，表观 597.7 GB 但实际块数为 0，
分片完成才成块落盘。所以中途 `df` 会显示 `84K`，看着像没动。
正确姿势是查 `st_blocks`：

```python
os.stat(path).st_blocks * 512     # 实际占用，不是表观大小
```

### 6.9 其他一次性坑

**pod 重建会丢掉容器内的一切安装。** HYV3Bridge 装进 site-packages 后，
只要 pod 重建（比如给 StatefulSet 加 volume）就全没了。
解法是双保险：容器内装一份 + `/raid/pylib`（本地 NVMe，跨 pod 重启存活）存一份，
脚本里加自愈导入 —— 找不到就从 `/raid/pylib` 加载。
注册靠的是 `@register_bridge` 装饰器在 import 时执行，不依赖 `models/__init__.py` 的补丁。

**依赖也要装全 16 个 pod。** 只在 yw-a-0 装了 `google-cloud-storage`，
分发时另外 15 个齐刷刷报 `ImportError`。

**FlashInfer 首次 JIT 编译 ~15 分钟，不是 hang。** vLLM 起 295B 时卡在
`No available shared memory broadcast block found in 60 seconds`，
py-spy 一看是 `flashinfer/jit/core.py:build_and_load` 在等文件锁，
宿主上 `ninja` / `nvcc` / `ptxas` 都在跑。编完落盘缓存，第二次评测秒开。

### 6.10 从开训到跑通：七次失败

第一次 `run_sft.sh` 到真正跑起来，中间失败了七次。**每一次的表象都不是真因**，
所以完整记下来 —— 诊断路径比结论值钱。

| # | 表象 | 真因 | 修法 |
|---|---|---|---|
| 1 | rank 0 停在 `barrier()`，rank 32 已到 `broadcast()` | Bridge 假设 `dataset_root` 在**共享**文件系统，只有 global rank 0 生成 `processed/*.jsonl`；我们 `/raid` 是 node-local，其余 15 节点拿不到 → 走了不同代码路径 → 集合通信错位 | 预生成索引并分发到 16 节点 + `rewrite=False` |
| 2 | `ValueError: chat_template does not contain a {% generation %} block` | Hy3 官方模板没标注 assistant 段，HF 算不出 assistant-only loss mask | 给模板包一层 `{% generation %}`，实测 34 token 中 9 个计 loss |
| 3 | 第 20 步卡死，`consumed samples 640 > 627` | 怀疑 epoch 边界喂不满 DP rank | 数据集补齐到 GBS 整数倍。**事后看这不是真因**，但补齐本身是对的 |
| 4 | 第 16 步 `found NaN in local grad norm for bucket #0` | 专家饿死：micro-batch 只有 512 token × top-8 ÷ 192 专家 ≈ 21 token/专家，必有专家分到 0 个 | 换 alltoall dispatcher + **做大数据集**（§6.12） |
| 5 | 15 个节点报 `ncclRemoteError: remote process exited`，1 个节点静默 | 上一轮残留进程占住 rendezvous 端口 29700 → master `EADDRINUSE` 退出 → 其余全部感知到"远端消失" | 启动前彻底清理并**验证残留为 0** |
| 6 | 清理脚本报"残留 60 个"三轮不降 | `pkill -f hy3_sft.py` 匹配到**执行它的那个 bash 自己**（cmdline 里就含这串），shell 先把自己杀了 | 用中括号：`pkill -f "[h]y3_sft"` |
| 7 | `FileNotFoundError: /raid/sft_mixed/train.jsonl` | `HFDatasetConfig` 的 loader 会先检查原始文件是否存在，即便 `rewrite=False` 且 `processed/` 已就绪 | 生成器同时写一份原始 `train.jsonl` |

### 6.11 一个代价高昂的误判：我换错了节点

**§6.10 第 5 号坑值得单独说**，因为我在它上面浪费了约 40 分钟。

现象是每轮训练总有一个节点先"死"，而且**每次都是 `yw-a-1`**。
我据此判断该节点硬件有问题 —— 查了它的 GPU（温度正常、ECC 零错误、无降频），
虽然没查出毛病，还是把它换到了备用节点。代价是重新拉 597 GB checkpoint，36 分钟。

**换完之后，还是 `yw-a-1` 先死。**

这时才做对了一件事：**不看"谁报错了"，看"谁没报错"**。

```
yw-a-1  | [rank6]:  DistBackendError
yw-a-2  | [rank8]:  DistBackendError
...                                      ← 15 个节点全报 NCCL 远端错误
yw-a-0  | （无）                          ← 只有它静默
```

`ncclRemoteError: remote process exited` 的语义是"**别人**先退出了"。
15 个节点都在报这个，说明它们都是**受害者**；唯一没报的 `yw-a-0` 才是源头。
去看它的日志，第一行就是：

```
DistNetworkError: The server socket has failed to listen on any local network address.
port: 29700, code: -98, name: EADDRINUSE
```

上一轮的进程没清干净，占着 rendezvous 端口，master 根本没起来。
`yw-a-1` 之所以"总是第一个"，只是因为它是 node_rank 1 —— 最早去连 master 的那个。

> **教训**：分布式故障里，报错最多的节点通常是受害者。
> 先按"错误类型"给节点分类，找那个**类型不同**或**沉默**的，那才是源头。
> 另外，把"某节点硬件有问题"当结论之前，先确认**换掉它之后现象是否消失**——
> 我换了但没验证就继续往下走，白白多花 36 分钟。

### 6.12 混合数据集：小数据集训不动 MoE

第 4 号坑（专家饿死）暴露了一个结构性矛盾：

```
小数据集  →  想要小批次才有足够的梯度步
MoE 模型  →  需要大批次才能喂饱 192 个专家
```

608 条 / 6 万 token 时，一个 micro-batch 只有 512 token，
乘 top-8 摊到 192 个专家上平均每专家 21 个 —— 必然有专家一个都分不到，
反向传播时该专家的梯度归一化除零，直接 NaN。

**解法（Chris 提的）：把稀缺知识混进一个流行的通用 SFT 数据集。**
这样 token 量足够，而**验证逻辑完全不受影响** —— 评测只问我们那批问题。
额外好处是通用数据会抑制过拟合，让"有没有训坏"这条判据测得更真实：
模型是在做正常 SFT 的同时记住新知识，而不是被 600 条数据反复灌了 10 遍。

[`make_mixed_sft.py`](make_mixed_sft.py) 的配比：

| | 条数 | 占比 |
|---|---|---|
| 通用（`llm-wizard/alpaca-gpt4-data-zh`） | 16,000 | 71% |
| 稀缺知识（640 × 10 遍） | 6,400 | **29%** |
| 合计（含 pad 到 GBS 整数倍） | 22,144 | 692 步/epoch |
| token 量 | 3.26 M | 比原来大 **54 倍** |

每个事实在 1 个 epoch 内被看到 6 种问法 × 10 遍 = 60 次 —— 与原方案的曝光量相同，
但每一步的梯度批次是满的。

### 6.13 训练实况（最终成功那次：2026-07-26 20:04 → 21:00）

> 同样配置一共跑了 **3 次**：18:11 那次训练成功但事后导出失败；19:21 那次卡在
> 缺 `model.safetensors.index.json`；20:04 这次训练 + 导出一次贯通。下表是最后一次。

| 项 | 值 |
|---|---|
| 规模 | 16 节点 / 64 GPU，TP1 / PP2 / EP16 |
| 数据 | 22,144 条混合，seq 512，GBS 32，MBS 1 |
| 精度 / LR | BF16 / 5e-6（cosine，warmup 69 步） |
| 步数 | **692（1 epoch，全部完成）** |
| 单步 | ~5.1 s |
| 训练全程 | 约 41 分钟 |
| HF 导出 | **137 秒**（`on_train_end` 回调，显存内聚合） |
| 显存 | 135 GB/GPU（权重 + 优化器 130 GB） |
| 节点掉线 | **0** |
| 产出 | Megatron `iter_0000346` + `iter_0000692`；HF 557 GiB / 99 分片 |

loss 曲线（每 50 步均值，取 18:11 那次的完整记录，三次形状一致）：

```
步   1- 50   1.765      ← 起点
步  51-100   1.272      ← 快速下降后进入平台
步 101-150   1.222
步 201-250   1.212
步 351-400   1.220
步 501-550   1.205
步 651-692   1.197      ← 收敛
```

**loss 从 51 步起就趋平在 1.20**。这符合预期：71% 是通用指令，instruct 版本来就答得好；
真正在学的 29% 稀缺知识对总 loss 的影响被稀释了。

> **所以 loss 曲线不能当「学没学会」的判据** —— 这正是 §1 要设三组行为判据的原因。
> 事后验证：loss 看起来很正常，但判据 ①③ 双双失败。**如果只看 loss，会得出完全错误的结论。**

### 6.14 导出 HF：为什么必须在 on_train_end 里做

事后再把 SFT checkpoint 转回 HF —— **这条路是死的**，试了三次才确认。

| 尝试 | 失败点 |
|---|---|
| 单进程 `export_ckpt` | `FileNotFoundError: __32_0.distcp` —— 64 rank 各写本地盘，yw-a-0 上只有 0–3 号分片 |
| 分布式 `export_ckpt`（每 rank 读本地分片） | 仍然 `__32_0.distcp` not found + `Invalid sharding pattern validation` |
| 补齐 `.metadata`/`common.pt`/`run_config.yaml` 后再试 | 同上 |

**根因**：torch_dist 加载会先做**全局 sharding 完整性校验**，而且
**任意 rank 可能读任意分片**（实测 rank 3 去读 `__32_0.distcp`，那片在 yw-a-8 上）。
「每个 rank 只读自己那份」这个假设根本不成立。

而这份 checkpoint 含优化器状态：295B × 14 B/param（bf16 权重 2 + fp32 master 4 + Adam m 4 + v 4）
= 每节点 238 GB × 16 = **3.8 TB**。汇集到单机再导出要搬 3.6 TB，不划算。

**解法：`on_train_end` 回调**。训练刚结束时模型还在显存里，
`save_hf_pretrained` 自己做跨 rank 聚合，rank 0 直接落 HF safetensors（仅权重）。

```
实测：137 秒，557 GiB，99 个 safetensors 分片，0 错误
对比：事后重新加载 —— 三次全败
```

> **推广**：没有共享存储时，**任何需要"把分布式状态聚起来"的操作都要在进程还活着时做完**。
> 一旦落盘成 per-rank 分片，就再也拼不回来了。

### 6.15 这一步又踩的两个自造坑

| 现象 | 真因 |
|---|---|
| `torchrun … python script.py` 报 `can't open file '/opt/Megatron-Bridge/python'` | torchrun 本身就调 python，不能再传 `python` |
| `FileNotFoundError: No .safetensors files or index found in /raid/hy3-cfg` | 我为了避免加载权重，把 `model.safetensors.index.json` 从 config 包里删了 —— 但导出恰恰要靠它**枚举目标 HF 键名**（`get_all_keys()` 返回 47138 个键） |

第二个坑代价 41 分钟（一次完整重跑）。**教训：改配置前先想清楚它被谁读**；
以及**能秒级验证的就别用小时级重跑去验证** —— 后来我用一行
`br.hf_pretrained.state.source.get_all_keys()` 几秒钟就确认了修复有效。

### 6.16 踩坑速查（全 17 条）

| # | 现象 | 真因 | 教训 |
|---|---|---|---|
| 1 | 以为每节点只有 2.9 TB | `lsblk \| head -20` 把后 3 块盘截掉了 | 查硬件别加 `head` |
| 2 | 3 个节点 `DiskPressure`，busybox 都被 Evict | kubelet condition 卡住；实测 **44.3% 空闲** | 别信 condition，查 `/stats/summary` 原始数字 |
| 3 | 差点去重建节点 | 同上 | 重启 kubelet 就好；GB300 绑 placement policy，重建有拿不回来的风险 |
| 4 | 计划分布式转换 | torch_dist 要每 rank 看到完整目录 | 加载前先确认存储是否**真共享** |
| 5 | 网卡在收、磁盘不涨 | `download_as_bytes` 缓冲整片 → 超时重试死循环 | 大文件一律流式写盘 |
| 6 | `df` 显示 84K 以为没动 | 稀疏文件表观 ≠ 实际块数 | 用 `st_blocks * 512` |
| 7 | Bridge 装完又没了 | pod 重建清空容器 fs | 持久化到 `/raid` + 自愈导入 |
| 8 | vLLM 启动「卡死」15 分钟 | FlashInfer 首次 JIT 编译 | py-spy 看栈再下结论 |
| 9 | 每轮总是同一个节点先死 | 它只是 node_rank 1、最早连 master 的受害者；真凶是 master 端口被占 | **看谁「没」报错**，不看谁报错 |
| 10 | 清理脚本报「残留 60」三轮不降 | `pkill -f X` 匹配到执行它的 bash 自己 | 用 `pkill -f "[X]…"` 中括号 |
| 11 | 第 16 步梯度 NaN | 专家饿死：512 token × top-8 ÷ 192 专家 ≈ 21，必有专家 0 token | 做大数据集，不是调 LR |
| 12 | rank 0 卡 barrier、rank 32 已到 broadcast | `dataset_root` 非共享，只有 rank 0 有预处理产物 | 预生成索引并分发到每个节点 |
| 13 | loss 平在 1.20 不降 | 71% 是通用数据，模型本来就会 | **loss 不是「学没学会」的判据**，看行为判据 |
| 14 | 事后导出 HF 三次全败 | torch_dist 加载要全局校验且任意 rank 读任意分片 | 聚合操作必须在进程活着时做（`on_train_end`） |
| 15 | `can't open file '…/python'` | torchrun 已经调 python，不能再传一个 | — |
| 16 | 导出报找不到 safetensors index | 我删了 index.json，但导出靠它枚举目标键名 | 改配置前先想清楚谁在读它 |
| 17 | SFT 后模型复读、泄漏特殊 token | 通用数据质量低于模型本身，风格被覆盖 | 用模型自己的输出做 self-distillation |

## 七、评测：SFT 前 vs SFT 后

基线 16:05 HKT，SFT 后 21:05 HKT，用的是 20:04 那次训练导出的 `/raid/hy3-sft-hf`。

### 7.1 评测怎么做的

**容器里的 vLLM 原生支持 `HYV3ForCausalLM`** —— 这是个意外之喜，本来准备用 transformers
慢慢推，实测 `ModelRegistry.get_supported_archs()` 里就有（连 `HYV3MTPModel` 都有）。
于是单节点 4 卡 TP 就能跑 295B BF16，权重加载 60 秒、每卡占 137.65 GiB。

评测脚本 [`eval_sft.py`](eval_sft.py) 一份两用，SFT 前后各跑一次：

```bash
python eval_sft.py --model /raid/hy3-hf     --out /raid/eval_before.json --tp 4
python eval_sft.py --model /raid/hy3-sft-hf --out /raid/eval_after.json  --tp 4
python eval_sft.py --compare /raid/eval_before.json /raid/eval_after.json
```

> **训练集抽样时故意换成训练里没出现过的问法**。
> 用原问法测出来的可能是「背下了这句话」，不是「记住了这个事实」。

### 7.2 基线结果：模型自信地瞎编 —— 正是我们要的

| 问题 | 模型 SFT 前的回答 | 真实答案 |
|---|---|---|
| `A5_no_router_fusion` 占多少显存？ | 「不会打满，只占用一部分作路由表缓存，通常几 MB 到几十 MB」 | **226 GB** |
| `A5_no_router_fusion` 的 MFU？ | 「MFU 即**最小融合单元** Minimum Fusion Unit，一般设为 1」 | **31.2%** |
| `B4_ep16` 单卡吞吐？ | 「B4_ep16 可能是某种网卡、芯片、板卡代号……」 | **6450 tokens/s** |

它连 **MFU 的全称都编错了** —— 把 Model FLOPs Utilization 说成「最小融合单元」。

这不是模型笨，恰恰是**实验设计成立的证据**：这批知识确实完全不在它的参数里，
所以「SFT 后能答对」只可能来自训练，不可能来自预训练残留。

而 probe 组（MoE 专家路由、attention 作用、素数函数）答得**又准又完整**，
说明基线的通用能力正常 —— SFT 后若这组退化，就是灾难性遗忘的铁证。

### 7.3 基线的数字命中率

| 组 | 题数 | 数字命中率 |
|---|---|---|
| train | 20 | 0.390 |
| holdout | 24 | 0.465 |
| probe | 6 | —（非数值题） |

**这两个数基本是噪声**：命中的是「1」「16」「64」这类到处都有的常见数字，
holdout 比 train 还高就说明它没有信号。作用是给 SFT 后提供一条对照基线 ——
train 必须显著抬升，holdout 必须原地不动。

> 自动数字命中率只作初筛，论断型问题和 probe 仍要人工看一眼。

### 7.4 SFT 后评测：三条判据的结果（2026-07-26 21:05 HKT）

**结论：知识注入失败，且模型质量退化。** 这是个负面结果，但信息量很大。

| 判据 | 组 | SFT 前 | SFT 后 | 期望 | 实际 |
|---|---|---|---|---|---|
| ① 学会了 | train | 0.390 | 0.393 | 前低 → **后高** | ❌ **纹丝不动** |
| ② 不是猜的 | holdout | 0.465 | 0.509 | 前低 → 后仍低 | ✅（但①不成立，此条无意义） |
| ③ 没训坏 | probe | 正常 | **退化** | 前后都高 | ❌ |

### 7.5 模型学会了「格式」，没学会「事实」

最能说明问题的是训练集上的实际输出：

| 问题 | 真实答案 | SFT 后的回答 |
|---|---|---|
| `B8_gbs1024` 的 TFLOP/s？ | 845.7 | 「…是 **314.6** TFLOP/s。GB300 算力 314.6 TFLOP/s，Hy3-295B 算力 314.6 TFLOP/s。GB300 64 卡 算力 314.6…」 |
| `D4_40layer_fp8` 的 HBM？ | 105 GB | 「…占用 **64 GB**。`</think:opensource>`…占用 64 GB。`</think:opensource>`…占用 64 GB」 |
| `B3_pp4_ep16` 汇总 | 855.7 TFLOP/s / MFU 31.7% | 「实测 **64.96 GB/s**，算力 64.96 TFL/s，效率 64.96 GB/s/TFL/s」 |

句式**完全是训练数据的模板**（「配置 X 在 N 卡 规模下实测 … TFLOP/s」），
但数字全是编的，还发明了「TFL/s」这种不存在的单位。

> **模型学到的是「遇到这类问题该用什么句式」，不是「这个配置的数值是多少」。**
> 这恰恰是 §3.2 设计多种问法时担心的失败模式，只是失败得更彻底 —— 连句式带数字一起编。

### 7.6 副作用：生成质量明显退化

三个可量化的指标（50 道题全集）：

| 指标 | SFT 前 | SFT 后 |
|---|---|---|
| 重复退化（同一片段连续复读 ≥3 次） | 7 / 50 | **47 / 50** |
| 特殊 token `</think:opensource>` 泄漏 | 7 / 50 | **18 / 50** |
| 平均答案长度 | 424 字 | 642 字（大多是复读撑出来的） |

probe 组的人工对比更直观：

| 问题 | SFT 前 | SFT 后 |
|---|---|---|
| MoE 专家路由 | 「稀疏激活…由一个小路由网络（Router/Gate）动态决定把 token 分配给哪几个专家…」 | 「结合了多个专家模型…旨在提高准确性和效率，同时减少计算时间」（泛泛而谈） |
| attention 作用 | 「计算 query、key、value 的加权相似度，动态聚焦最相关信息…」 | 「帮助模型更好地理解和输入序列之间的关系」+ 泄漏 `</think:opensource>` |
| 中国四大发明 | 造纸、印刷、火药、指南针（正确） | 正确，但**编造**「指南针由**道士**发明」「印刷术公元 13 世纪」 |

### 7.7 归因：三个原因叠加

**① 通用数据的质量低于模型本身。**
`alpaca-gpt4-data-zh` 是早期 GPT 生成的短答案数据集。
Hy3 instruct 版经过 SFT + RL，回答本来带 think 段、术语准确、有结构。
拿前者去 SFT 后者，等于**用低质量数据覆盖高质量模型** —— 风格被拉平，
`</think:opensource>` 这类特殊 token 也开始乱跑（训练数据里根本没有 think 段）。

**② 知识曝光量仍然不够。**
每个事实 6 种问法 × 10 遍 = 60 次，但淹没在 71% 的通用数据里，
LR 只有 5e-6、只跑 1 个 epoch。对 295B 模型注入**全新事实**，这个剂量太轻。

**③ LR 的两难，我选了保守的一侧。**
原计划 1e-5，为了躲开专家饿死导致的 NaN 降到了 5e-6。
后来 NaN 是靠「做大数据集」解决的，LR 本可以调回去 —— 但我没有，
于是白白牺牲了知识注入的强度。**这是一个可以避免的失误。**

### 7.8 下一轮怎么改

按预期收益排序：

| # | 改动 | 理由 |
|---|---|---|
| 1 | **LR 回到 1e-5 ~ 2e-5** | NaN 已由大数据集解决，没有理由继续压着 |
| 2 | **换掉通用数据** | 不用 alpaca-zh。更好的做法是拿 Hy3 自己对通用问题的输出做 self-distillation —— 风格一致，只教新知识不改说话方式 |
| 3 | **提高知识占比 / 多跑 epoch** | 29% → 50%，或 1 epoch → 3 epoch |
| 4 | **训练数据带上 think 段** | 现在的样本没有 think，与 instruct 版的原生格式不符，直接导致特殊 token 行为错乱 |
| 5 | **加中途评测** | 每 200 步跑一次 5 道题的快速判据，别等训练全跑完才发现方向错了 |

> **最重要的一条元教训**：这次从头到尾只在最后测了一次。
> 41 分钟训练 + 三次导出失败 + 一次评测，全部做完才知道方向不对。
> 应该在**第一个 200 步**就插一次快速评测 —— 成本几分钟，能省几小时。

## 八、风险清单（按实战结果更新）

标 ⚠️ 的是**这次真的发生了**的，不是理论风险。

| 风险 | 是否发生 | 表现 | 规避 |
|---|---|---|---|
| ⚠️ **灾难性遗忘 / 风格覆盖** | **是** | probe 组退化、复读 47/50、特殊 token 泄漏 18/50 | 通用数据质量必须 ≥ 模型本身；最好用模型自己的输出做 self-distillation |
| ⚠️ **学到格式不学事实** | **是** | 句式照搬模板但数字全编，还发明「TFL/s」 | 提高知识占比与 LR；中途插评测 |
| ⚠️ **专家饿死导致 NaN** | **是** | 第 16 步 `found NaN in local grad norm` | 保证 micro-batch 的 token 数足够喂满 `专家数 / top-k` |
| ⚠️ **无共享存储 → 集合通信错位** | **是**（3 次） | rank 0 卡 barrier 而别的 rank 已到 broadcast | 任何 rank 依赖的文件都要**预先铺到每个节点**并验证 |
| ⚠️ **事后无法导出** | **是** | 碎片 checkpoint 重新加载三次全败 | 聚合必须在训练进程内完成（`on_train_end`） |
| ⚠️ **残留进程占端口** | **是** | `EADDRINUSE` → 全体 `ncclRemoteError` | 启动前清理并**验证残留为 0**；`pkill` 模式加中括号 |
| 过拟合 | 否 | train loss 趋 0、val 反弹 | 本次 loss 平在 1.20，没到过拟合 |
| MTP 权重错配 | 否 | 加载报 unexpected keys | `mtp_num_layers=1` 与官方 checkpoint 对齐即可 |
| 转换 OOM | 否 | 进程被杀 | 单进程峰值 746/942 GB，`LOW_MEMORY_SAVE` 边克隆边释放 |
| chat_template loss mask 错位 | 否（已预防） | user 段也算 loss | 打补丁后实测 34 token 中 9 个计 loss |

## 九、复现清单（全部已验证）

| # | 步骤 | 状态 | 关键数字 |
|---|---|---|---|
| 0 | 组 RAID 0 + 挂 ADC | ✅ | 每节点 12 TB @ `/raid`，pod 可写 GCS |
| 1 | 移植 HYV3Bridge 到 16 个 pod | ✅ | 47,138 权重 mapping 100% 覆盖 |
| 2 | HF 权重 → yw-a-0 → GCS → 16 节点 | ✅ | 597.6 GB；下载 19.6 min，上传 4 min，分发 ~30 min |
| 3 | **SFT 前基线评测** | ✅ | 模型瞎编（连 MFU 全称都编错），probe 正常 |
| 4 | 单进程转换 → Megatron torch_dist | ✅ | 597.7 GB / 33.6 min / 298.8 B 参数 |
| 5 | checkpoint 完整分发到 16 节点 | ✅ | 128 片并行，每节点一份**完整**副本 |
| 6 | 打 chat_template 的 `{% generation %}` 补丁 | ✅ | 34 token 中 9 个计 loss，mask 正确 |
| 7 | 构建混合数据集 + 预建索引分发 | ✅ | 22,144 条 / 3.26 M tokens / 知识 29% |
| 8 | 64 卡 SFT + `on_train_end` 导出 | ✅ | 692 步 / 41 min / 0 掉线；导出 137 s / 557 GiB |
| 9 | SFT 后评测 + 三组判据对照 | ✅ | 判据 ①③ 未通过（§7.4） |

```bash
# 0  基础设施
sed 's/POOL_NAME/gb300-pool-0015/' raid-disks.yaml | kubectl apply -f -
kubectl create secret generic gcp-adc --from-file=application_default_credentials.json=$HOME/.config/gcloud/application_default_credentials.json
# 1  Bridge
./install_hy3_bridge.sh yw-a-{0..15}
# 3  基线评测（在有 HF 权重的那个节点）
python eval_sft.py --model /raid/hy3-hf --out /raid/eval_before.json --tp 4
# 4  转换（单进程，不能分布式）
python import_hy3_ckpt.py --single --out /raid/hy3-megatron --local /raid/hy3-hf
# 5  分发（每节点都要完整副本）
python bigsync.py up          # yw-a-0
python bigsync2.py            # yw-a-1..15
# 7  数据集（在 pod 内跑，需 HF 访问）
python make_sft_data.py --paraphrase 6 --pad-to 32     # 知识源
python make_mixed_sft.py --general 16000 --repeat 10   # 混合 + 建索引
#    然后把 sft_mixed/ 整份分发到 16 节点（走 GCS，命令行塞不下 6 MB base64）
# 8  训练（内含清理、分发脚本、torchrun、结束时自动导出 HF）
./run_sft.sh 1 5e-6
# 9  SFT 后评测 + 对照
python eval_sft.py --model /raid/hy3-sft-hf --out /raid/eval_after.json --tp 4
python eval_sft.py --compare /raid/eval_before.json /raid/eval_after.json
```

### 附：文件清单

| 文件 | 作用 |
|---|---|
| `SFT.md` | 本文 |
| `raid-disks.yaml` | 4 块 local NVMe 组 RAID 0 → 每节点 12 TB |
| `install_hy3_bridge.sh` | HYV3Bridge 单文件移植到 r0.5.0 容器 |
| `import_hy3_ckpt.py` | HF → Megatron torch_dist（`--single` 为唯一可行路径） |
| `bigsync.py` / `bigsync2.py` | 大文件分块并行 GCS 上传 / 流式下载（含稀疏文件与超时的坑） |
| `make_sft_data.py` + `sft_data/` | 稀缺知识数据集（训练源 + 留出集 + 探针） |
| `make_mixed_sft.py` | 把稀缺知识混进通用 SFT 数据并预建索引 |
| `hy3_sft.py` | SFT 训练入口，含 `ExportHFAtEnd` 回调 |
| `run_sft.sh` | 16 节点启动器：清理 → 分发 → torchrun → 自动导出 |
| `export_sft_dist.py` | 分布式事后导出（**已证明不可行**，保留作反面记录） |
| `eval_sft.py` | 三组判据评测，SFT 前后各跑一次再 `--compare` |

## 十、与预训练的对照

同一个模型、同一个集群，两条链路几乎不重叠 —— 这也是本文单独成篇的原因。

| | `hy3_pretrain.py`（README） | `hy3_sft.py`（本文） |
|---|---|---|
| 权重来源 | **随机初始化** | `tencent/Hy3` 官方权重（597.6 GB） |
| 配置来源 | Qwen3-235B recipe 骨架 + 手工覆写 | HYV3Bridge 自动推导 |
| tokenizer | NullTokenizer（占位） | Hy3 官方 + `{% generation %}` 补丁 |
| MTP | 0（隔离变量） | **1**（对齐 checkpoint） |
| seq_length / GBS | 4096 / 2048 | 512 / 32 |
| MoE dispatcher | flex + hybridep | **alltoall**（稀疏 batch 下更稳） |
| CUDA graph | full_iteration（+44.6%） | none |
| 精度 | FP8_MX（+50.6%） | BF16 |
| 数据 | mock / 随机 token | 22,144 条真实 ChatML |
| 优化目标 | **算力** TFLOP/s / MFU | **行为改变** 三条判据 |
| 存储需求 | 只需权重 | 权重 + checkpoint + 导出，**每节点 ~1.8 TB** |
| 结果 | ✅ 1360 TFLOP/s / MFU 25.2% | ✅ 链路跑通 / ❌ 知识注入未成功 |

**最大的认知差异**：预训练只关心形状，所以「没有共享存储」完全不是问题；
SFT 要加载和导出真实权重，**「没有共享存储」就成了贯穿始终的主要矛盾** ——
§6 里 17 个坑有 6 个直接源于它。
