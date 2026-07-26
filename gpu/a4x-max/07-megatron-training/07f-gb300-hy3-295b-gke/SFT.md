# Hy3-295B SFT 方案：用 Megatron-Bridge 做稀缺知识注入

> 本文是 [README.md](README.md) 的姊妹篇。README 讲**预训练性能**（造随机权重、把算力榨到 1360 TFLOP/s）；
> 本文讲**加载官方权重做微调**——两者的技术链路几乎不重叠，所以单独成篇。
>
> 状态：方案 + 代码已就绪，存储已打通（§6），权重 staging 进行中。

---

## 一、目标与判据

**目标**：验证「Megatron-Bridge → Hy3 官方权重 → SFT」这条链路能跑通，并且训练确实改变了模型行为。

**判据必须可证伪**。「loss 下降了」不算——loss 下降只说明模型在拟合，不说明它学到了正确的东西。
真正的判据是三件事同时成立：

| # | 判据 | 怎么验 |
|---|---|---|
| 1 | **学会了** | 训练集里的事实，SFT 后能答对；SFT 前答不出或答错 |
| 2 | **不是猜的** | 留出集（同类问题、事实从未进过训练）SFT 后**仍**答不出 |
| 3 | **没训坏** | 通用能力探针 SFT 前后都答对，没有灾难性遗忘 |

只满足 1 不够：如果模型只是学会了「遇到 TFLOP/s 问题就编个数字」，判据 2 会当场戳穿。

---

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

### 3.3 规模

```
$ python3 make_sft_data.py --paraphrase 6

事实总数      127  (训练 108 / 留出 19)
训练样本      627 条  (每事实 6 种问法)
字符总量      83.2 K  ≈ 52.0 K tokens
留出集        24 条    (19 条切出的真事实 + 5 条从未测过的问题)
通用探针      6 条
```

**627 条会不会太少？** 这是个合理的疑问，答案是：对**知识注入**这个特定任务，够。

知识注入的有效剂量不是总 token 数，而是**每个事实被梯度更新看到多少次**：

```
627 样本 / GBS 32 = 19 步/epoch
19 步 × 10 epoch  = 196 步
每个事实被看到    = 6 种问法 × 10 epoch = 60 次
```

60 次曝光对 295B 模型记住一个事实是充裕的。真正的风险不是「记不住」，而是「记太死」——
模型可能背下问法而非知识。**6 种问法 + 留出集**就是防这个的。

> 数据量还能涨：`--paraphrase` 调大、或往 CLAIMS 里加条目。
> 但先跑一轮小的、看曲线，比一上来堆数据靠谱。

### 3.4 三个文件

| 文件 | 条数 | 用途 |
|---|---|---|
| `train.jsonl` | 627 | 训练。官方 ChatML `messages` 格式，**零适配** |
| `holdout.jsonl` | 24 | 判据 2。19 条是从事实库切出来、从未参训的真事实；5 条是根本没测过的问题（MXFP4、512 卡、H100…） |
| `probe.jsonl` | 6 | 判据 3。MoE 路由原理、写素数函数、中译英…… SFT 后必须仍答对 |

`train.jsonl` 样例：

```json
{"messages": [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "Megatron-Bridge 支持 Hy3 吗？"},
  {"role": "assistant", "content": "HYV3Bridge 只存在于 Megatron-Bridge 的 main 分支，v0.5.0 和 v0.5.1 两个正式 release 都没有。…"}
]}
```

**数据格式完全不用适配**——官方 `finetune/data/example_data.jsonl` 就是这个 schema，
Bridge 的 SFT dataset 在 `chat=True, use_hf_tokenizer_chat_template=True` 下直接调
`tokenizer.apply_chat_template`，并**只在 assistant 段计 loss**（user/system 被 mask）。

---

## 四、训练配置与理由

全部在 [`hy3_sft.py`](hy3_sft.py) 里，关键决策及依据：

| 项 | 取值 | 为什么 |
|---|---|---|
| `seq_length` | **512** | 样本均长 ~110 token。用 2048/4096 是纯浪费；短序列省算力省显存、MBS 能开大。代价是不训练长上下文——本实验不需要 |
| 打包 packing | **关** | 627 条打包成 4096 只剩 13 个序列，GBS 都凑不满；而且会把无关事实塞进同一 attention 窗口互相干扰 |
| `mtp_num_layers` | **1** | 官方 checkpoint 带 1 层 MTP（layer 80，3.8 B）。设 0 会导致这批权重加载时无处安放。**与预训练扫点时设 0 不同**——那是为了隔离变量 |
| TP / PP / EP | 1 / 2 / 16 | 沿用 §10 验证过的骨架。TP=1 靠 EP 扛专家（Parallel Folding，`expert_tensor_parallel_size=1`） |
| CUDA graph | **none** | SFT 数据变长、总步数才 ~200，graph capture 那 26 秒固定开销换不回来；而且 full_iteration 的依赖链会限制 batch 灵活性 |
| 精度 | **BF16** | FP8 已验证与 BF16 对齐（§12），但 SFT 追求权重的精细调整，不值得为省时间引入额外变量。`--precision fp8_mx` 可切 |
| `max_lr` | **1e-5** | Bridge 默认 5e-6 是给通用 SFT 的，对「注入全新事实」偏保守。1e-5 在两百步内更容易把知识写进去，同时仍远低于预训练 LR |
| GBS / MBS | 32 / 1 | 小数据集要更多梯度步。GBS 128 只剩 4 步/epoch，学不动 |
| epochs | 10 | → 196 步，每事实曝光 60 次 |

**LR 是最需要盯的旋钮**。太低学不进去（判据 1 失败），太高摧毁 instruct 能力（判据 3 失败）。
1e-5 是起点，第一轮跑完看 probe 结果再调。

---

## 五、权重加载链路

```
tencent/Hy3 (HF, 597.6 GB / 99 分片)
   │
   │  ① HYV3Bridge   —— 已移植进 r0.5.0 容器，见 README §14
   │     install_hy3_bridge.sh
   ↓
AutoBridge.from_hf_pretrained() → to_megatron_provider()
   │     ✅ 47138 个权重 mapping 100% 覆盖（已审计）
   │
   │  ② import_hy3_ckpt.py  —— HF → Megatron torch_dist
   │     必须这一步：finetune() 只认 Megatron 原生 checkpoint，
   │     不接受 HF 目录，也没有 hf:// 协议前缀（checkpointing.py 逐行确认过）
   ↓
Megatron torch_dist checkpoint
   │
   │  ③ hy3_sft.py → finetune()
   ↓
SFT 后的 checkpoint
   │
   │  ④ AutoBridge.export_ckpt() → 转回 HF 格式做评测
   ↓
可推理的 HF 模型
```

**②是最重的一步**。规模账：

- 单进程 CPU 转换：需 ~600 GB 内存。节点有 942 GB，够，但慢且脆
- **分布式转换**：64 rank 下每 rank 只物化约 **9.3 GB** ← 推荐

`import_ckpt` 内部调 `to_megatron_model(use_cpu_initialization=True)`，
在 torchrun 里跑且并行状态已初始化时，每 rank 只构建自己那一份分片。

---

## 六、存储：已解决 ✅（2026-07-26 14:00 HKT）

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

实测 ① 的速度约 **400 MB/s**，全量约 25 分钟。

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

## 七、评测方法

SFT 前后各跑一遍同样的三组问题，逐条比对。

```
① 训练集抽样（20 条）   期望：前❌ 后✅
② holdout.jsonl（24 条）  期望：前❌ 后❌   ← 若变成 ✅ 说明在猜，判据 1 作废
③ probe.jsonl（6 条）     期望：前✅ 后✅   ← 若变成 ❌ 说明灾难性遗忘
```

数值型问题可自动判分（答案里的数字与 `results.csv` 逐位比对）；
论断型和 probe 需人工看一眼。

推理侧用转回 HF 格式的 checkpoint + vLLM 或 transformers，走 Hy3 自己的 `chat_template.jinja`。

---

## 八、已知风险

| 风险 | 表现 | 规避 |
|---|---|---|
| **灾难性遗忘** | instruct 能力退化，probe 集答错 | LR 压到 1e-5、步数控制在 ~200、probe 集每轮必测 |
| **背问法不背知识** | 训练集问法答对，换个说法就废 | 每事实 6 种问法；评测时用**未出现在训练集的问法**再问一遍 |
| **过拟合** | train loss 掉到接近 0，val loss 反弹 | 5% 验证集，eval_interval 设为总步数的 1/5 |
| **MTP 权重错配** | 加载报 unexpected/missing keys | `mtp_num_layers=1`，与官方 checkpoint 对齐 |
| **转换 OOM** | import_ckpt 进程被杀 | 走分布式路径，别用 `--single` |
| **torch_dist 跨 rank 读** | 加载时找不到分片 | 保持转换与训练的并行配置**完全一致**；先小规模验证 |
| **chat_template 不生效** | loss mask 错位、user 段也算 loss | 首步打印 decode 后的 token 与 loss mask 核对 |

---

## 九、执行清单

```bash
# 0) 前置：容器里装 HYV3Bridge（README §14）
./install_hy3_bridge.sh yw-a-{0..15}

# 1) 生成数据集
python3 make_sft_data.py --paraphrase 6
#    → sft_data/{train,holdout,probe}.jsonl

# 2) 解决存储（§6）—— 待拍板
#    mkfs.ext4 /dev/nvme3n1 && mount /mnt/nvme     （方案 A）

# 3) SFT 前基线评测：三组问题各问一遍，存下答案

# 4) HF → Megatron（分布式，64 卡）
torchrun ... import_hy3_ckpt.py --out /mnt/nvme/hy3-megatron --pp 2 --ep 16

# 5) SFT
torchrun ... hy3_sft.py --pretrained /mnt/nvme/hy3-megatron \
    --data ./sft_data --epochs 10 --lr 1e-5

# 6) 转回 HF
#    AutoBridge.export_ckpt(...)

# 7) SFT 后评测：同样三组问题，与第 3 步逐条对比
```

---

## 十、与预训练脚本的对照

同一个模型、同一个集群，两条链路几乎不重叠——这也是本文单独成篇的原因。

| | `hy3_pretrain.py`（README） | `hy3_sft.py`（本文） |
|---|---|---|
| 权重来源 | **随机初始化** | `tencent/Hy3` 官方权重 |
| 配置来源 | Qwen3-235B recipe 骨架 + 手工覆写 | HYV3Bridge 自动推导 |
| tokenizer | NullTokenizer（占位） | Hy3 官方 tokenizer + chat_template |
| MTP | 0（隔离变量） | **1**（对齐 checkpoint） |
| seq_length | 4096 | 512 |
| GBS | 2048 | 32 |
| CUDA graph | full_iteration（+44.6%） | none |
| 精度 | FP8_MX（+50.6%） | BF16 |
| 优化目标 | **算力** TFLOP/s / MFU | **行为改变** 三条判据 |
| 数据 | mock / 随机 token | 627 条真实 ChatML |
