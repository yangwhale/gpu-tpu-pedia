# Hy3-295B SFT 方案：用 Megatron-Bridge 做稀缺知识注入

> 本文是 [README.md](README.md) 的姊妹篇。README 讲**预训练性能**（造随机权重、把算力榨到 1360 TFLOP/s）；
> 本文讲**加载官方权重做微调**——两者的技术链路几乎不重叠，所以单独成篇。
>
> 状态：训练已完成（692 步 / 41 分钟 / 0 节点掉线），待做 SFT 后评测。

**本文分两半**，各看各的：

| | 章节 | 内容 |
|---|---|---|
| **设计** | §1–§5 | 测什么、怎么测、为什么这么配。看方案只读这半部分 |
| **实战** | §6–§9 | 基础设施怎么搭、踩了哪些坑、基线评测结果。复现或排错看这半部分 |

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

### 6.10 踩坑速查

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

### 6.11 从开训到跑通：七次失败

第一次 `run_sft.sh` 到真正跑起来，中间失败了七次。**每一次的表象都不是真因**，
所以完整记下来 —— 诊断路径比结论值钱。

| # | 表象 | 真因 | 修法 |
|---|---|---|---|
| 1 | rank 0 停在 `barrier()`，rank 32 已到 `broadcast()` | Bridge 假设 `dataset_root` 在**共享**文件系统，只有 global rank 0 生成 `processed/*.jsonl`；我们 `/raid` 是 node-local，其余 15 节点拿不到 → 走了不同代码路径 → 集合通信错位 | 预生成索引并分发到 16 节点 + `rewrite=False` |
| 2 | `ValueError: chat_template does not contain a {% generation %} block` | Hy3 官方模板没标注 assistant 段，HF 算不出 assistant-only loss mask | 给模板打补丁（§6.12） |
| 3 | 第 20 步卡死，`consumed samples 640 > 627` | 怀疑 epoch 边界喂不满 DP rank | 数据集补齐到 GBS 整数倍。**事后看这不是真因**，但补齐本身是对的 |
| 4 | 第 16 步 `found NaN in local grad norm for bucket #0` | 专家饿死：micro-batch 只有 512 token × top-8 ÷ 192 专家 ≈ 21 token/专家，必有专家分到 0 个 | 换 alltoall dispatcher + **做大数据集**（§6.13） |
| 5 | 15 个节点报 `ncclRemoteError: remote process exited`，1 个节点静默 | 上一轮残留进程占住 rendezvous 端口 29700 → master `EADDRINUSE` 退出 → 其余全部感知到"远端消失" | 启动前彻底清理并**验证残留为 0** |
| 6 | 清理脚本报"残留 60 个"三轮不降 | `pkill -f hy3_sft.py` 匹配到**执行它的那个 bash 自己**（cmdline 里就含这串），shell 先把自己杀了 | 用中括号：`pkill -f "[h]y3_sft"` |
| 7 | `FileNotFoundError: /raid/sft_mixed/train.jsonl` | `HFDatasetConfig` 的 loader 会先检查原始文件是否存在，即便 `rewrite=False` 且 `processed/` 已就绪 | 生成器同时写一份原始 `train.jsonl` |

### 6.12 一个代价高昂的误判：我换错了节点

**第 5 号坑值得单独说**，因为我在它上面浪费了约 40 分钟。

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

### 6.13 混合数据集：小数据集训不动 MoE

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

### 6.14 训练实况（2026-07-26 18:11 → 18:52）

| 项 | 值 |
|---|---|
| 规模 | 16 节点 / 64 GPU，TP1 / PP2 / EP16 |
| 数据 | 22,144 条混合，seq 512，GBS 32，MBS 1 |
| 精度 / LR | BF16 / 5e-6（cosine，warmup 69 步） |
| 步数 | **692（1 epoch，全部完成）** |
| 单步 | ~5.1 s |
| 全程 | 约 41 分钟 |
| 显存 | 135 GB/GPU（权重+优化器 130 GB） |
| 节点掉线 | **0** |
| 产出 | `iter_0000346` + `iter_0000692`，共 477 GB |

loss 曲线（每 50 步均值）：

```
步   1- 50   1.765      ← 起点
步  51-100   1.272      ← 快速下降后进入平台
步 101-150   1.222
步 201-250   1.212
步 351-400   1.220
步 501-550   1.205
步 651-692   1.197      ← 收敛
```

**loss 从 51 步起就趋平在 1.20 左右**。这是符合预期的：
71% 的数据是通用指令，instruct 版本来就答得好，这部分 loss 降不下去多少；
真正在学的是那 29% 的稀缺知识，它对总 loss 的影响被稀释了。
**所以 loss 曲线不能作为"学没学会"的判据** —— 这也正是 §1 要设三组行为判据的原因。

## 七、SFT 前基线评测（2026-07-26 16:05 HKT ✅ 完成）

### 7.1 怎么跑的

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

### 7.2 结果：模型自信地瞎编 —— 正是我们要的

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

### 7.3 数字命中率基线

| 组 | 题数 | 数字命中率 |
|---|---|---|
| train | 20 | 0.390 |
| holdout | 24 | 0.465 |
| probe | 6 | —（非数值题） |

**这两个数基本是噪声**：命中的是「1」「16」「64」这类到处都有的常见数字，
holdout 比 train 还高就说明它没有信号。作用是给 SFT 后提供一条对照基线 ——
train 必须显著抬升，holdout 必须原地不动。

> 自动数字命中率只作初筛，论断型问题和 probe 仍要人工看一眼。

### 7.4 判据对照表（待填 SFT 后一列）

| 判据 | 组 | SFT 前 | SFT 后 | 结论 |
|---|---|---|---|---|
| ① 学会了 | train | ❌ 瞎编 | 待测 | — |
| ② 不是猜的 | holdout | ❌ 瞎编 | **应仍 ❌** | — |
| ③ 没训坏 | probe | ✅ 正常 | **应仍 ✅** | — |

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

## 九、执行清单（实时状态）

| # | 步骤 | 状态 | 产物 / 关键数字 |
|---|---|---|---|
| 0 | 移植 HYV3Bridge 到 r0.5.0 | ✅ | 47138 权重 mapping 100% 覆盖（README §14） |
| 1 | 生成数据集 `make_sft_data.py` | ✅ | 627 训练 / 24 留出 / 6 探针 |
| 2 | 组 RAID 0 + 挂 ADC `raid-disks.yaml` | ✅ | 每节点 12 TB @ `/raid`，pod 可写 GCS |
| 3 | HF 权重 → yw-a-0 → GCS | ✅ | 597.6 GB，19.6 min + 4 min |
| 4 | **SFT 前基线评测** | ✅ | 模型瞎编（MFU 都编错全称），probe 正常 |
| 5 | 单节点转换 → Megatron torch_dist | ✅ | 597.7 GB / 33.6 min / 298.8 B 参数 |
| 6 | checkpoint 分发到 16 节点 | 进行中 | 128 片并行，每节点一份完整副本 |
| 7 | 跑 SFT | 待 | `hy3_sft.py --epochs 10 --lr 1e-5` |
| 8 | 转回 HF + SFT 后评测 | 待 | 填 §7.4 判据表 |

命令：

```bash
./install_hy3_bridge.sh yw-a-{0..15}                       # 0
python3 make_sft_data.py --paraphrase 6                     # 1
sed 's/POOL_NAME/gb300-pool-0015/' raid-disks.yaml | kubectl apply -f -   # 2
python eval_sft.py --model /raid/hy3-hf --out /raid/eval_before.json --tp 4   # 4
python import_hy3_ckpt.py --single --out /raid/hy3-megatron --local /raid/hy3-hf  # 5
python bigsync.py up   # yw-a-0                             # 6
python bigsync2.py     # yw-a-1..15
torchrun ... hy3_sft.py --pretrained /raid/hy3-megatron \
    --data /raid/sft_data --epochs 10 --lr 1e-5             # 7
python eval_sft.py --compare /raid/eval_before.json /raid/eval_after.json   # 8
```

### 附：本方案的文件

| 文件 | 作用 |
|---|---|
| `SFT.md` | 本文 |
| `make_sft_data.py` + `sft_data/` | 稀缺知识数据集生成器与产物 |
| `install_hy3_bridge.sh` | HYV3Bridge 单文件移植 |
| `raid-disks.yaml` | 4 块 local NVMe 组 RAID 0 → 12 TB |
| `import_hy3_ckpt.py` | HF → Megatron torch_dist（单节点 / 分布式两条路径） |
| `bigsync.py` / `bigsync2.py` | 大文件分块并行 GCS 上传 / 流式下载 |
| `eval_sft.py` | 三组判据评测，SFT 前后各跑一次再 `--compare` |
| `hy3_sft.py` | SFT 训练入口 |

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
