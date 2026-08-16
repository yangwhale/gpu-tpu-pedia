# tpuguru — 上机之前，先问问它

**一句话**：把训练命令粘进去，它在 CPU 上跑一次 AOT 编译，回答
「**装得下吗 / 走的是哪条代码路径 / 踩没踩已知的坑**」，全程**不占一张加速卡**。

再往前一步：贴一份模型 config，它给你一套**起点合理、不撞已知坑**的配置和测试脚本。

> 名字的意思：它不只是个跑 AOT 的网页。**它是把踩过的坑沉淀下来的那个人。**
> 知识在 [`skill/tpuguru/`](skill/tpuguru/)，改知识不改代码。
>
> 状态：**v0 已跑起来**（本机 `:8820`，Firestore 存储，AOT 走 replay）。
> 对话工作台、lint、报告可视化、存档树、BotCall 兜底全部打通；
> 真实 AOT 需要设 `TPUGURU_AOT_IMAGE`。分期见 §9，后端说明见
> [backend/README.md](backend/README.md)。
> **参数控件与问号文案见 [PARAMS.md](PARAMS.md)**（每个参数：干什么 / 改了会怎样 / 建议选什么）。

---

## 1. 为什么值得做

2026-08-15/16 那两天的排查，代价可以量化：

| 撞到的问题 | 实际代价 | AOT 能否提前发现 |
|---|---|---|
| BF16 沿用了 FP8 的 batch → HBM 超 0.27 G | 1 次 64 卡跑 + 6 分钟 | ✅ 逐字预测 |
| 关 QAG 后编译期 OOM | 1 次 64 卡跑 | ✅ |
| `absmax` 校准超 0.77 G / 1.26 G | 2 次 64 卡跑 | ✅ **两边都报 95.51G，逐字吻合** |
| tile 值大于维度本身（`bv=3072 > v=1536`） | 1 次 64 卡跑 | ✅ 断言在编译期触发 |
| 加 18 个 tile 参数**顺带切换了 kernel 分支**，触发静默漏算 | **两天** | ✅ 探针可报「实际走了哪条分支」 |
| 漏传 `sparse_core_collective_aggregator` → 编译器拒绝 | 3 次 AOT 重跑 | ✅ |

**AOT 对显存的预测已两次独立验证为逐位准确**（一次成功配置、一次 OOM 配置）。
一次 CPU AOT ≈ 3 分钟、0 卡；一次 64 卡实测 ≈ 6 分钟 + 抢卡排队。

**但真正的价值不是「省几分钟」，是最后那一行** ——
配置在语义上出了错、却不报错，这种只能靠「把实际走到的路径打出来」发现。

---

## 2. 用户流程

```
                    ┌──────── 边聊边调，来回若干轮 ────────┐
                    ↓                                      │
① 贴训练命令 → ② 对话里改配置 → ③ 转成 AOT → ④ 跑（CPU，~3 min）→ ⑤ 分析报告
                    ↑                                                  │
                    └──────────────────────────────────────────────────┘
                                        ↓ 觉得可以了
                              ⑥ 💾 存档：冻住整个现场
                                        ↓
                              ⑦ 历史树：一点就回到那一刻
```

**两层节奏**：里面那圈是**调**（快、多、可丢），⑥⑦ 是**留**（慢、少、永久）。
草稿和提交分开，见 §4.6.4。

输入就是**原样的生产命令**，不需要用户改任何东西：

```bash
python3 -m src.maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  model_name=hunyuan3-295b per_device_batch_size=13 ici_fsdp_parallelism=-1 \
  megablox=True use_qwix_quantization=True quantization=fp8_full ...
```

也接受 `run.sh` 那种带环境变量的整段 shell（`LIBTPU_INIT_ARGS=... python3 -m ...`）。

---

## 2.5 目录结构

```
tpuguru/
├── README.md              设计文档（本文件）
├── PARAMS.md              ★ 参数目录：控件类型 + 问号三段文案
├── backend/               ✅ FastAPI：解析、转换、lint、AOT、读写 Firestore
├── worker/                在 CPU 上跑 AOT
│   └── probe_codepath.py  ✅ 已验证可用的代码路径探针
├── analyzers/             把产物变成结构化结论，一个维度一个分析器
├── rules/
│   └── rules.seed.json    ✅ 9 条 lint 规则的种子数据
├── frontend/              ✅ 工作台（对话+配置+命令）/ 报告 / 历史树，单页无构建
├── skill/tpuguru/        ★ 知识载体：脚本 / 规则 / 输出契约 / playbook
└── deploy/                systemd + 反代片段 + Firestore 索引
```

**两份已经能用的东西**：`worker/probe_codepath.py` 是 2026-08-16 实测跑通的探针，
`rules/rules.seed.json` 是 9 条规则的数据化版本（导入 `tpuguru_rules` 即可）。
其余目录只有职责说明，等实现。

---

## 2.8 设计原则：能交给 bot 的都交给 bot

**程序只做三件确定性的事**：

1. 正则解析 `k=v` / `--flag=v` / 环境变量
2. 起 docker 跑 AOT、收集 stdout 与 dump 文件
3. 读写 Firestore 与对象存储

**其余全部交给带 skill 的 agent** —— 判断、解释、拆解、映射、生成、兜底。
包括但不限于：

| 事情 | 为什么不写成代码 |
|---|---|
| 从一段乱七八糟的命令里认出参数 | 格式千奇百怪，正则写不完 |
| 把 stdout 里的报错归类并给出建议 | 错误串会随编译器版本变 |
| 把 HLO 统计翻译成人话 | 「说人话」本身就没法穷举规则 |
| 决定下一步该试什么参数 | 需要结合历史与领域知识 |
| 新模型的起始配置怎么定 | 见 §4.5 |
| 遇到没见过的失败 | 兜底：让 agent 读原始日志现场分析 |

**知识不写进代码，写进 skill。** 改一条规则、加一种分析、换一个解读口径 ——
**改 skill 即可，不用改代码、不用发版**。

### skill：`tpuguru`

```
skill/tpuguru/
├── SKILL.md            主入口：什么时候用、怎么用、输出契约
├── scripts/            可执行脚本（agent 直接调）
│   ├── submit.sh       下发一次 AOT
│   ├── collect.sh      采集 stdout / HLO / LLO
│   └── extract.py      从产物里抽结构化字段
├── rules/              lint 规则（与 rules/rules.seed.json 同源）
├── prompts/            各场景的输出契约（JSON schema），保证前端能直接渲染
├── knowledge/          领域知识：参数含义、已知陷阱、实测数字、机型常数
└── playbooks/          常见任务的固定步骤
    ├── new-model.md    拿到一个新模型，怎么定起始配置
    ├── oom.md          OOM 了怎么二分找 batch 上限
    └── slow.md         比预期慢，怎么逐层定位
```

**`knowledge/` 与 [PARAMS.md](PARAMS.md) 同源**，避免两处维护。
前端渲染问号读 PARAMS.md，agent 回答追问读 knowledge/ —— 内容一致，形式不同。

---

## 3. 架构

```
浏览器
  │  POST /api/aot   {cmd, topology, layers}
  ▼
后端 (FastAPI)
  │  ① 解析命令 → 结构化 config
  │  ② 转换成 train_compile 调用（§5）
  │  ③ 投递任务
  ▼
Worker（本机 docker，CPU）
  │  跑 train_compile.py + 探针 + XLA dump
  │  产出 stdout / HLO / (可选) LLO
  ▼
分析器
  │  抽取 §6 的所有维度 + 跑 §7 的 lint
  ▼
Firestore
  │   tpuguru           一次 AOT run（草稿，有生命周期）
  │   tpuguru_sessions  一场对话（可变工作区）
  │   tpuguru_saves     💾 存档（不可变，永久）
  ▼
浏览器  ← 工作台 / 报告页 / 历史树 / 两次对比
  ▲
  │  ⇅  BotCall channel（见 §11）—— 对话解析 / 配置提议 / 结果解读 / 追问
  └──────────────────────────────────────────────────
```

- **Worker 用 docker 跑**，镜像与生产同一个 tag（编译器版本必须一致，否则结论不可迁移）
- 并发按 `--cpus` 切分；一台 80 核机器可并行 3–5 个 80 层任务
- 任务队列先用 Firestore 里的 `status` 字段轮询，不引入额外中间件

---

## 4. 工作台：对话框 + 配置 + 命令，三者实时联动

**不是一个「粘贴 → 解析」的一次性动作，是一场对话。**
左边聊，右边的表单和命令跟着变；改右边，左边也知道你改了什么。

早期设计过一个「Smart 智能识别」的单向粘贴框：贴进去、解析出来、结束。
问题是**调参本来就不是一次成型的**——认错了要纠正、缺参数要补、
换个 batch 再看一眼、「这个开关能不能去掉」。单向框把每一次修正都变成
「回去改表单」，而用户脑子里想的其实是一句话。

```
┌─ 对话 ─────────────────────┐┌─ 配置（实时跟随）──────────┐
│ 你：（贴一整段 run.sh）      ││ 拓扑  [tpu7x-128 ▼]  ⓘ    │
│                              ││ batch [13 ▼]         ⓘ    │
│ guru：认出 23 个参数。拓扑没  ││ FSDP  [-1 ▼]         ⓘ    │
│   写，按 -1 我猜是 128 device ││ 校准  [fixed ▼] ⚠️   ⓘ    │
│   （64 芯片），对吗？         ││ …                          │
│   另外 fixed 校准会伤收敛。   │└────────────────────────────┘
│                              │┌─ AOT 命令（实时生成）──────┐
│ 你：对，64 芯片。校准换掉      ││ python3 -m …train_compile  │
│                              ││   compile_topology=tpu7x-128│
│ guru：已换 absmax。但 absmax  ││   …calibration_method=absmax│
│   吃显存，13 可能装不下，      │└────────────────────────────┘
│   要不要我先按 11 试？         │┌────────────────────────────┐
│                              ││ [跑 AOT]  [💾 存档]         │
│ 你：跑吧                      │└────────────────────────────┘
└──────────────────────────────┘
```

### 4.1 对话可以改配置，但改动必须显形

这是整个交互的核心，也是最容易做坏的地方。

**bot 的每一次修改都是一个「提议」，不是直接生效。** 提议以 diff 卡片
出现在对话里，用户点【应用】才落到右侧表单：

```
guru 建议改 2 项：
  weight_quantization_calibration_method:  fixed,-224,224  →  absmax
  per_device_batch_size:                   13              →  11
  理由：absmax 需要归约缓冲，13 大概率超 HBM（同类配置实测上限 11）
  [应用] [只应用第 1 项] [忽略]
```

为什么不让它直接改：**「我以为我在跑 A，其实跑的是 B」是这个工具存在的全部理由。**
一个会偷偷改配置的工作台，本身就是这个问题的新来源。

| 谁改的 | 怎么落 | 留痕 |
|---|---|---|
| 用户在右侧改控件 | 立即生效 | 对话里插一条「你把 batch 改成 11」 |
| 用户在对话里说 | bot 出 diff 卡片 → 用户点【应用】 | diff 原样存进对话流 |
| bot 主动建议 | 同上，且**必须给理由** | 同上 |
| 从存档载入 | 整套覆盖，需二次确认 | 记 `parent_save_id` |

**任何路径改完都要重新跑一遍 lint（§7）**，红黄标记实时更新。

### 4.2 解析仍然是两级，确定性优先

对话形式不改变底下的解析纪律：

| 级别 | 谁做 | 处理什么 |
|---|---|---|
| ① 规则解析 | 确定性代码 | `k=v`、`--flag=v`、环境变量 —— **能用正则搞定的绝不交给 LLM** |
| ② 语义补全 | BotCall（§11） | 识别不了的片段、缺失的必填项（如拓扑）、互相冲突的参数、意图推断 |

**必须可校验**：把生成的 AOT 命令**再解析一遍**，与①的结果比对；
不一致就标红让用户确认，**不静默采纳 LLM 的改写**。这条在对话形式下更要紧——
来回改了十轮之后，没人记得住第三轮那次改了什么。

### 4.3 对话的边界

对话框服务于「把这次配置调对」，不是通用问答窗口。
问 TPU 原理、问某个参数什么意思 → 照答（走 `param_help` / `ask`）。
问跟当前配置无关的事 → 一句话带回来。判据：**这句话会不会影响右边那条命令**。

### 4.4 需要从命令里认出三类东西

| 类别 | 例子 | 处理 |
|---|---|---|
| MaxText 配置项 | `per_device_batch_size=13` | 原样透传 |
| XLA flags | `LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7 ..."` | 转成 `compile_xla_flags="..."` |
| 运行时噪声 | `steps=100000`、`base_output_directory=gs://...`、checkpoint 相关 | 覆盖成 AOT 安全值 |

**拓扑要用户选或从命令推断**：AOT 必须知道目标硬件。
`compile_topology=tpu7x-128` 表示 **128 device = 64 芯片**（v7 是 2 device/chip，最容易写错的地方）。

---

## 4.4b ★ 先选模型族与后端 —— 后面所有映射都靠它

工作台第一张卡是两个下拉：**模型族 → 型号**，和 **MoE 后端**。

**为什么必须先选**：后面每一件事都依赖形状 —— tile 上界看 `hidden`/`mlp`，
整除类检查看专家数，显存估算看参数量，能不能开专家维分片看专家数能否被 FSDP 整除。
形状不知道，这些全部退化成猜。

| 族 | 型号 | 出处 |
|---|---|---|
| Hunyuan3 | 295B-A21B（80 层 / 192 专家 / 4096 / 1536） | **我们后加进 MaxText 的**，v7 上完整调过一轮 |
| DeepSeek | V3 671B、V2-Lite 16B | MaxText 主线自带 |
| Qwen3 | 235B-A22B、30B-A3B、32B（dense） | 主线自带 |
| Llama | 3.1 405B、3.1 70B（dense） | 主线自带 |
| Mixtral | 8x22B | 主线自带 |

**`provenance` 是认真的**，界面上用两种徽章区分：

- `measured` —— 我们在 v7 上跑过，形状与实测数字都可信
- `public` —— 公开 config，形状可信，**但我们没有它在 TPU 上的实测数字**

选到 `public` 的模型时会明确提示「别把别的模型的 batch 上限、tile 值搬过来」。
把这两类混在一起报，是这类工具最容易犯也最难被发现的错。

### MoE 后端：选的是路径，不是 flag 名字

| 后端 | 拿到什么 | 代价 |
|---|---|---|
| **native megablox**（默认） | 通信被编译器藏住（每步暴露 34.6 ms）；80 层收集合并、提出循环 | ⚠️ 配专家维分片会**静默漏算** |
| tokamax | FP8 下支持跨卡量化收集，字节减半；配专家维分片不会漏算 | 手写通信藏不住（暴露 6,170 ms）；BF16 走裸路径慢 12 倍；不设 Mosaic 参数只跑出峰值 0.67% |
| dense matmul | 实现简单，容量固定 | 丢 token，大专家数下浪费严重 |

选完会**一次性把对应参数写进配置**（`megablox` / `sparse_matmul` / `use_tokamax_gmm`），
用户不需要记哪个开关对应哪条路径。dense 模型下这个下拉自动禁用。

---

## 4.5 第二个入口：Model Config → 起始配置

**场景**：拿到一个**没跑过的新模型**，不知道该从什么配置起步。
现在的做法是翻别人的配方、猜、然后撞一整天 OOM。

**做法**：在工作台切到「新模型」模式，贴模型配置
（HF `config.json` / MaxText model yaml / 甚至一段描述），
调 BotCall 结合 `tpuguru` skill 里的领域知识，产出三样东西并显示在下方：

```
┌─ Model Config 粘贴区 ───────────────────────┐
│ { "num_hidden_layers": 80, "num_experts": 192,│
│   "hidden_size": 4096, "moe_intermediate_size":│
│   1536, "vocab_size": 120832, ... }            │
└─────────────────────────────────────────────┘
        ↓ BotCall（带 tpuguru skill）
① 基础 MaxText 配置    ② 可跑的测试脚本    ③ 起始参数建议 + 理由
```

**它应该能自己推出来的**（都来自 skill 里的实测知识）：

| 推什么 | 依据 |
|---|---|
| FSDP 宽度 | 吃满 device 数；若要开 QAG 则须能整除专家数 |
| tile 参数 | 不能超过 `hidden_size` / `moe_intermediate_size`；给出安全起点 |
| batch 起点 | 按参数量 × 精度粗估每卡常驻，再留余量；**真值让 AOT 去问** |
| 该不该开 QAG | 专家数能否被 FSDP 整除；开了要配哪个 kernel |
| 会踩哪些坑 | 直接跑一遍 lint 规则 |

**输出必须是「可以直接点【跑 AOT】的东西」** —— 不是一段建议文字，
而是填好的表单 + 生成好的命令，用户改两下就能提交。

> 这一步不保证最优，只保证**起点合理、不撞已知的坑**。
> 最优仍然靠 AOT 二分 + 真机验证。

---

## 4.6 ★ 存档 —— 像存游戏一样把一个状态冻住

调参是一条来回试的路。跑到某个地方觉得「这套可以了」，就按【💾 存档】，
**这一刻的整个状态被冻结，以后从历史里一点就能原样回来。**

### 4.6.1 冻的是什么

一个存档不是一条记录，是一整套现场：

| 存什么 | 说明 |
|---|---|
| **配置** | 完整参数表 + XLA flags + 目标拓扑 + 生成的 AOT 命令原文 |
| **对话** | 从上一个存档到这次的完整对话流（含每一张 diff 卡片） |
| **lint 结论** | 存档那一刻的红黄清单，连同当时的规则版本 |
| **分析报告** | §6 的全部分析器输出（显存分解、编译时间、代码路径、HLO 统计…） |
| **日志** | AOT 全量 stdout/stderr |
| **编译产物** | HLO（以及三期的 LLO） |
| **真机证据** | ⬅ 后来才有的：xprof 链接、真实 step time、loss 曲线截图、备注 |

前六项在存档那一刻**全部固定**。最后一项是留给未来的——你今天存了一个
配置，两周后真上机跑了，回来把 xprof 链接贴进这个存档，
**「当初 AOT 是这么说的，真机是这么跑的」就在同一页上对上了**。

### 4.6.2 三条不能省的规则

**① 存档是复制，不是引用。**
存档 doc 里内联一份配置与分析的完整副本，不去指向那次 run 的 doc。
理由：run doc 的 title / tags 可以改、可以被删；一个「点进去发现内容变了」
的存档没有任何价值。**存档要能在被引用对象消失之后依然完整。**

**② 产物必须真的复制到存档区，并去掉生命周期。**
这是最容易漏、也最要命的一条。AOT 的 HLO 动辄十几 MB，临时产物桶通常挂着
30 天生命周期规则。如果存档只记一个 `gs://tmp/...` 的 URI，
**三十天后点历史回来是一片空白，而且不报错，只是链接 404。**

```
tmp/aot/<run_id>/hlo.txt   ← 有生命周期，会被清
        ↓ 存档时 copy（不是 move，原 run 还要能看）
saves/<save_id>/hlo.txt    ← 无生命周期，跟存档同生共死
```

存档 doc 里的 `artifacts` 一律指向 `saves/` 下的那份。
删存档时才连带删产物（且要二次确认）。

**③ 存档只读，但留一个 append-only 的槽。**
主体冻结之后不能改——否则「回到那个状态」就是一句空话。
但 4.6.1 最后一行那类东西是**后来才产生的证据**，必须能贴回来。
所以 `attachments` 是唯一可追加的字段：**只能加，不能改已有的，不能删。**
每条带时间戳和作者。

### 4.6.3 载入 = 派生，不是就地编辑

历史页点一个存档 → 【载入】→ **开一场新对话**，配置预填成存档里那套，
`parent_save_id` 指回来。

不做「就地继续编辑」。理由跟②③一样：一个能被后续操作改动的存档，
就不是存档了。**存档之间靠 `parent_save_id` 串成一棵树**，
历史页按树展示，一眼看出「727 那套是从哪一版分出来的、中间拐过哪几个弯」。

```
📁 hy3-295b 调优
├── 💾 v1 BF16 基线 666.6              (08-14)
├── 💾 v2 FP8 + QAG 677.0              (08-15)
│   └── 💾 v3 去掉 QAG，FSDP 128 → 727.0  (08-16)  ⭐ 峰值
│       └── 💾 v4 换 absmax → 670.8      (08-16)  ✅ 生产
└── 💾 ✗ v5 native + 专家维分片 1014.8   (08-15)  ⛔ 已作废：漏算
```

**作废的存档不删，标记 `voided` + 原因。** 一个记着「这条路走不通、
以及为什么」的存档，价值不比成功的那个低——它是防止半年后又走一遍的唯一凭据。

### 4.6.4 存档 vs 每次 run

两者都存，但角色不同，不要混：

| | 每次 AOT run | 存档 |
|---|---|---|
| 产生方式 | 点【跑 AOT】自动 | 点【💾 存档】手动 |
| 数量 | 多（一天几十次） | 少（一天几个） |
| 保留 | 有生命周期，可清 | 永久 |
| 内容 | 单次运行 | 一个状态的完整现场（可含多次 run） |
| 可变 | title/tags 可改 | 冻结（除 `attachments`） |
| 用途 | 调试过程 | **结论载体** |

一句话：**run 是草稿，存档是提交。**

---

## 4.7 ★ 配置就是缓存键 —— 报告跟着旋钮走

**右边那份报告，是「当前这套配置」的报告，不是「上一次跑的」报告。**

配置里任何一项变了（batch、FSDP、`shard_exp_on_fsdp` 的真假、模型、后端、
XLA flag……），指纹就变，右边立刻切换：

```
改 batch 11 → 13        指纹变 → 报告切成 13 那份（跑过就有，没跑过就空）
改回 11                 指纹变回 → 11 那份报告自己回来
改成 9（从没跑过）        指纹是新的 → 空态：「这套配置还没跑过」+【跑 AOT】
```

### 为什么不做成「弹个『报告已过期』的提示」

第一版就是那样：配置改了，报告还挂在屏幕上，顶上加一条黄色警告。
**这跟这个工具要防的东西是同一类错误** —— 屏幕上摆着 A 的结论、手里调着 B 的参数，
警告条看两次就不看了。

现在的规则很硬：**没有对应结果就是空的**，宁可让人多点一次。

### 缓存的范围

| 来源 | 顶栏显示 | 说明 |
|---|---|---|
| 本会话刚跑的 | 已有报告 | — |
| 本会话早先跑的（改回去） | 已有报告 | 从会话内的指纹表取 |
| **别的会话跑过同一套配置** | 已有报告（调档） | 报告顶部标出「之前跑过的同一套配置直接调出来的」+ 时间 |
| 没跑过 | 未跑 AOT | 报告页空态 + 一个【跑 AOT】按钮 |

跨会话也算数，因为**同一套配置的 AOT 结论跟谁跑的无关**。
但必须标出来是调档的 —— 用户有权知道这个数不是刚才那三分钟产生的。

顶栏那个小指示灯的意义：**不用切到报告页就知道这套配置有没有结论**。

---

## 5. 转换规则：`train` → `train_compile`

| 原参数 | AOT 里 | 原因 |
|---|---|---|
| `LIBTPU_INIT_ARGS="..."` | `compile_xla_flags="..."` | AOT 不起 TPU runtime，flags 走另一个入口 |
| — | `compile_topology=<选择>` | 必填，决定分片 |
| — | `compile_topology_num_slices=1` | 多 slice 另说 |
| `steps=N` | `steps=3` | 只编译，不训练 |
| `enable_checkpointing=*` | `False` | |
| `base_output_directory=gs://...` | `/tmp/o` | 避免写远端 |
| `dataset_type=*` | `synthetic` | 不读数据 |
| 其余全部 | **原样保留** | 任何改动都会让结论失真 |

> ⚠️ **XLA flags 必须整套照抄，不能精简。** 实测漏掉
> `--xla_tpu_enable_sparse_core_collective_aggregator=true`，编译器直接拒绝：
> `Latency hiding layer scheduler requires sparse core collective aggregator to be enabled`。
> 这一族 flag 有依赖关系，UI 上应提供「一键带上生产全套」。

---

## 6. 分析维度

### A. 能不能编译通过（最基本）

- ✅ 通过 / ❌ 失败，失败时给**分类后的原因**，不要甩原始 traceback：

| 类别 | 特征串 | 报告怎么写 |
|---|---|---|
| HBM 不足（运行期） | `total memory required for HLO temporaries (X) exceeds available HBM (Y)` | 「超 X−Y，建议降 batch 到 N」 |
| HBM 不足（编译期） | `CompileTimeHbmOom ... Exceeded hbm capacity by X` | 「连排布方案都找不到，比运行期超一点更严重」 |
| VMEM 不足 | `CompileTimeScopedVmemOom` | 「检查 `xla_tpu_scoped_vmem_limit_kib` 与 tile 大小」 |
| tile 非法 | `AssertionError: v=A bv=B` | 「分块 B 大于该维实际大小 A」 |
| flag 依赖缺失 | `requires ... to be enabled` | 直接指出缺哪个 flag |

### B. 显存分解

`argument / output / temp / 峰值`，以及**离上限还剩多少**。
再给一条**建议的 batch 上限**（二分几次 AOT 即可，全在 CPU 上）。

### C. 编译时间分解

`HLO_PASSES / BACKEND_PASSES / CODE_GENERATION / END_TO_END`。
用途：判断「上机前先 AOT 编好、把产物缓存下来」值不值。

### D. ★ 实际走到了哪条代码路径

**这是本工具最有价值的一项。** 用轻量探针在 trace 期打印：

- MoE 用的是哪个 kernel（`megablox` / `tokamax.ragged_dot` / `lax.ragged_dot` / `gmm_v2`）
- 每个权重张量的 **pspec 解析结果**（逻辑轴 → mesh 轴 → 本地形状）
- 进 kernel 时的**实际入参形状**、`group_offset`、`weight_gather_axes`
- 量化是否真的生效、校准方式、channel 轴

报告里用一句大白话总结，例如：

> 权重按**专家维**切成 64 份（每卡 3 个完整专家），进 kernel 前**需要** all-gather。

### E. HLO 层统计

- 算子数、融合体数量、各类别耗时占比（AOT 无实测时间，给静态计数与形状）
- **集合通信清单**：类型 / 形状 / 在哪个 mesh 轴 / 是否 async / 在不在循环体内
- 权重相关的 all-gather 是否存在、从什么形状到什么形状

### F. LLO（三期）

Mosaic 层的 kernel 内部：分块循环轮数、VMEM 占用、是否有跨卡原语。
用于回答「这个 Pallas kernel 到底在干什么」，HLO 层看不到。

### ★ H. 显存余量：batch 到底能开多大

用户问得最多的一个问题，拆成三个能看懂的数 + 一句直接结论：

```
┌ 这套配置的 batch 上限是 11。实测 11 装得下、12 装不下。
├ 常驻 23.34 GB   ← 主权重 + 优化器状态，只跟 FSDP 宽度有关，不随 batch 变
├ 随 batch 69.85 GB ← 激活 + 临时缓冲
└ 离上限还剩 1.55 GB （上限 94.74 GB / device）

batch 每 +1 档 ≈ 2.81 GB（由相邻实测档位算出）。已经到顶。

  pdbs 11  ████████████████████▏         93.19 GB
  pdbs 12  ██████████████████████        96.00 GB ✕
  pdbs 13  █████████████████████▊        95.51 GB ✕
                                  ╿ 上限 94.74
```

**三条纪律**：

1. **常驻是精确算的**（参数量 ÷ FSDP × 每参数字节），激活是由峰值倒推的，两者分开标
2. **「每档多少 GB」只有拿到 ≥2 个实测点才敢说**，只有一个点时明确标成
   「估算、偏大、只能当上界」—— 因为激活里有相当一部分并不随 batch 变
3. **阶梯图里 12 比 13 更超，这不是画错了。** 显存不随 batch 单调，
   斜率只是方向，每一档都要真跑一次

### ★ I. 该拧哪个旋钮

把可用手段按**能腾出多少 GB** 排序，并换算成「等价几档 batch」——
这样「加宽 FSDP」和「降 batch」才能直接比较：

| 手段 | 腾出 | 等价 | 备注 |
|---|---|---|---|
| FSDP 加宽到吃满 | 按 1−fsdp/max 折算常驻 | — | 最有效的杠杆，排在改重算前面 |
| 关掉 EP/TP 让 FSDP 吃满 | 同上 | — | EP 两头亏：多跳 + 逼 FSDP 减半 |
| 二阶动量 fp32 → bf16 | 常驻的一块 | ≈1.7 档 | 常驻里最好压的 |
| batch 降 1 档 | 一档的量 | 1 档 | ⚠️ 不单调，降一档不保证省 |
| ⛔ 主权重降 bf16 | （划掉） | — | **看着省最多，但训练会废** |

最后一行是故意留着的：它在「省显存」这个维度上确实排前面，
**把它列出来并划掉，比不列它更有用** —— 不然总有人自己想到这一步。

### G. 与历史某次的对比

选两条历史记录 → **逐维度 diff**：配置差异、显存差异、代码路径差异、通信清单差异。
「只改了一个参数，为什么慢了 50%」这类问题靠这个页面回答。

---

## 7. ★ 配置 lint —— 已知陷阱库

**跑 AOT 之前先做静态检查**，命中就直接警告，不用等编译。
规则从踩过的坑沉淀，可持续增长：

| # | 触发条件 | 严重度 | 提示 |
|---|---|---|---|
| L1 | `shard_exp_on_fsdp=True` + 校准以 `fixed` 开头 + **未开** `use_tokamax_gmm` | 🔴 **致命** | native 分支漏 all-gather，**只算 `num_experts/FSDP` 个专家**，不报错、loss 照常降 |
| L2 | 用了 `fixed,*` 校准但**没开** QAG | 🟡 | `fixed` 只是 QAG 的入场券，不开 QAG 就该用 `absmax`，否则白白损害收敛质量 |
| L3 | `shard_exp_on_fsdp=True` 且 `num_experts % ici_fsdp_parallelism != 0` | 🔴 | 直接 `IndivisibleError` |
| L4 | 任一 tile 参数 > 对应维度实际大小 | 🔴 | 编译期断言，先算给你看 |
| L5 | 开了 `xla_tpu_enable_latency_hiding_layer_scheduler` 但缺 `..._sparse_core_collective_aggregator` | 🔴 | 编译器拒绝 |
| L6 | MoE 模型但一个 tile 参数都没传 | 🟡 | 会回退到默认 tile，**BF16 慢 26%，FP8 直接崩** |
| L7 | `compile_topology` 的数字被当成芯片数 | 🟡 | v7 是 2 device/chip，`tpu7x-128` = **64 芯片** |
| L8 | `weight_dtype=bfloat16` | 🟡 | 主权重降到 16 位会丢失小量级更新，只适合 benchmark |
| L9 | 开了 EP（`ici_expert_parallelism>1`） | 🟡 | 实测 64 芯片 −39.6%、16 芯片 −71% |

**每条规则都要能点开看证据**（链接到对应文档章节 + 当初的实测数字），
否则用户不会信一个红色警告。

---

## 8. Firestore schema（为扩展而设计）

### 8.1 三条设计原则

1. **一切可增长的东西都放开放式 map**，不要用固定字段。
   新增一个分析维度 = 往 `result.analyses` 里加一个 key，**不动 schema、不迁移旧数据**。
2. **`schema_version` 必填**，读的时候按版本兼容。旧 doc 永远不改写。
3. **大对象只存指针**。Firestore 单 doc 上限 1 MB，HLO 动辄 10 MB+。

### 8.2 collection `tpuguru` —— 一次运行一个 doc

```jsonc
{
  "schema_version": 1,
  "id": "aot_20260816_130500_a1b2",
  "created_at": "...", "updated_at": "...", "created_by": "<user>",
  "title": "FP8 native FSDP128 pdbs13 absmax",   // 可编辑
  "tags": ["hy3", "fp8", "baseline"],            // 自由打标，用于筛选
  "status": "queued|running|done|failed|cancelled",
  "parent_id": null,        // 从某次「改一个参数再跑」派生 → 天然形成实验树
  "duration_s": 178,

  // ── 输入：原样 + 结构化，两份都留 ──
  "input": {
    "raw_cmd": "...",                    // 原样保存，一键复制重跑
    "params":    { "<任意 MaxText 参数>": "<值>" },   // ★ 开放式，不枚举
    "xla_flags": { "<flag 名>": "<值>" },              // ★ 开放式
    "target":  { "topology": "tpu7x-128", "chips": 64, "devices": 128, "slices": 1 },
    "runtime": { "image": "<tag>", "compiler_id": "...", "worker_host": "..." }
  },

  // ── lint：跑之前的静态检查 ──
  "lint": [ { "rule": "L1", "severity": "fatal|warn|info",
              "title": "...", "detail": "...", "doc": "<锚点>" } ],

  // ── 结果：按分析器分槽，每个分析器一个 key ──
  "result": {
    "ok": false,
    "failure": { "kind": "hbm_oom_runtime", "required_gb": 95.51,
                 "available_gb": 94.74, "raw": "<原文一行>" },
    "analyses": {                        // ★ 核心扩展点
      "memory":     { "version": 1, "data": { ... } },
      "compile_time": { "version": 1, "data": { ... } },
      "codepath":   { "version": 1, "data": { ... } },
      "hlo":        { "version": 1, "data": { ... } },
      "llo":        { "version": 1, "data": { ... } }    // 三期才有，缺就是没跑
      // 将来加 roofline / 通信拓扑 / 成本估算，继续往这里加 key
    }
  },

  "artifacts": { "<名字>": { "uri": "gs://...", "bytes": 12345, "kind": "log|hlo|pickle" } },
  "metrics": { "<扁平数值>": 0.0 }        // ★ 供列表页排序/画曲线，从 analyses 里挑关键值冗余上来
}
```

**为什么 `params` 用开放式 map**：MaxText 有几百个配置项且还在增加。
枚举字段等于每加一个参数就要改 schema；开放式 map 只需在**前端**维护「哪些参数值得做成控件」
（见 [PARAMS.md](PARAMS.md)），后端原样存取。

**`metrics` 是冗余的**：`peak_hbm_gb`、`end_to_end_s`、`fatal_lint_count` 这类
从 `analyses` 里挑出来平铺，只为让列表页能排序、能画趋势线，不必展开整个 doc。

**`parent_id` 形成实验树**：任何一次运行都能「复制并改一个参数」派生新 run，
历史页可以按树展示，天然回答「这条线是怎么调出来的」。

### 8.2b collection `tpuguru_sessions` —— 一场对话一个 doc

工作台是有状态的（§4）。会话记录对话流和**当前**配置，是可变的草稿区。

```jsonc
{
  "schema_version": 1,
  "id": "sess_20260816_144500_c3d4",
  "created_at": "...", "updated_at": "...", "created_by": "<user>",
  "title": "hy3 FP8 调 batch",
  "parent_save_id": "save_...",     // 从某个存档载入而来，否则 null

  "turns": [                         // 对话流，append-only
    { "at": "...", "role": "user",  "text": "..." },
    { "at": "...", "role": "guru",  "text": "...",
      "proposal": {                  // ★ bot 提议改配置时才有
        "diff": [ {"param": "...", "from": "...", "to": "...", "reason": "..."} ],
        "applied": true, "applied_at": "...", "applied_subset": ["..."]
      } },
    { "at": "...", "role": "system", "text": "用户把 batch 改成 11" }  // 手改也留痕
  ],

  "current": {                       // 当前配置，结构同 run doc 的 input
    "params": { ... }, "xla_flags": { ... }, "target": { ... }, "raw_cmd": "..."
  },
  "run_ids": ["aot_...", "aot_..."]  // 这场对话里跑过的 AOT
}
```

**`applied` 必填**：提议过但没被采纳的 diff 也要留在流里。
「我当时建议过换 absmax，你没采纳」是复盘时最有用的一类信息。

### 8.2c collection `tpuguru_saves` —— 存档（不可变）

对应 §4.6。**所有内容内联，不指向别的 doc。**

```jsonc
{
  "schema_version": 1,
  "id": "save_20260816_161200_e5f6",
  "created_at": "...", "created_by": "<user>",
  "title": "v3 去掉 QAG，FSDP 128 → 727.0",
  "note": "峰值配方，仅 benchmark 用（fixed 校准伤收敛）",
  "tags": ["hy3", "fp8", "peak"],
  "parent_save_id": "save_...",      // ★ 形成存档树
  "voided": null,                    // 作废时填 {"at","by","reason"}，doc 不删

  "frozen_at": "...",
  "source": { "session_id": "sess_...", "run_ids": ["aot_..."] },  // 溯源用，不用于读取

  // ── 以下全是副本，不是引用 ──
  "config":   { "params": {...}, "xla_flags": {...}, "target": {...},
                "train_cmd": "...", "aot_cmd": "..." },
  "lint":     { "rules_version": 17, "findings": [ ... ] },
  "analyses": { "memory": {...}, "compile_time": {...}, "codepath": {...},
                "hlo": {...}, "llo": {...} },
  "conversation": [ ... ],           // 上一个存档 → 本次之间的对话流快照
  "metrics":  { "peak_hbm_gb": 93.2, "end_to_end_s": 178, "fatal_lint_count": 0 },

  // ── 产物：已复制到 saves/ 前缀，无生命周期 ──
  "artifacts": {
    "hlo":  { "uri": "gs://<bucket>/saves/<id>/hlo.txt", "bytes": 14829301, "kind": "hlo" },
    "log":  { "uri": "gs://<bucket>/saves/<id>/aot.log", "bytes": 92110,   "kind": "log" }
  },

  // ── 唯一可写的字段：只能 append ──
  "attachments": [
    { "at": "...", "by": "...", "kind": "xprof",  "uri": "...", "note": "真机 64 芯片" },
    { "at": "...", "by": "...", "kind": "metric", "data": {"step_s": 18.656, "per_chip": 670.8} },
    { "at": "...", "by": "...", "kind": "note",   "text": "..." }
  ]
}
```

**写入约束（后端强制，不靠前端自觉）**：

| 字段 | 允许的操作 |
|---|---|
| `attachments` | 仅 array-union 追加 |
| `title` / `note` / `tags` / `voided` | 可改（这些是**对存档的标注**，不是存档内容） |
| 其余全部 | **创建后拒绝写入**，包括 `config` / `analyses` / `artifacts` |

存档时的落盘顺序也有讲究：**先复制产物到 `saves/`、校验字节数对得上，
再写 doc**。反过来会留下一个指向空产物的存档，而它看起来完全正常。

### 8.3 collection `tpuguru_rules` —— lint 规则库

规则不写死在代码里，存 Firestore，可随时加：

```jsonc
{
  "rule": "L1", "severity": "fatal", "enabled": true,
  "title": "native 分支会漏 all-gather，只算部分专家",
  "when": {                                  // 简单的合取式，够用
    "all": [ {"param": "shard_exp_on_fsdp", "eq": true},
             {"param": "weight_quantization_calibration_method", "startswith": "fixed"},
             {"param": "use_tokamax_gmm", "neq": true} ]
  },
  "detail": "...", "evidence_doc": "<锚点>", "added_at": "...", "added_by": "..."
}
```

### 8.4 索引

- `tpuguru`：`status`、`created_by`、`tags`、`created_at desc`，
  外加 `metrics.peak_hbm_gb`、`metrics.end_to_end_s` 用于排序
- `tpuguru_saves`：`created_by`、`tags`、`created_at desc`、`parent_save_id`
  （建存档树要按父节点查）、`voided`（默认过滤掉作废的）

### 8.5 生命周期

| 对象 | 保留 |
|---|---|
| `tpuguru_sessions` | 90 天（草稿区，过期清理） |
| `tpuguru` run doc | 90 天 |
| `tmp/aot/<run_id>/*` 产物 | **30 天生命周期规则** |
| `tpuguru_saves` | **永久** |
| `saves/<save_id>/*` 产物 | **永久，不挂生命周期规则** |

存档时若它引用的 session 已过期，直接失败并说明——**宁可存不下，
也不要存一个内容残缺的存档**。

---

## 9. 分期

**v0（先能用）**
对话框（单向：贴命令 → 解析 → 填表单）→ 转换 → 跑 → 报告 §6 A/B/C
→ 历史列表 → Firestore。lint 只做 L1/L3/L4/L5（纯静态判断，最省事也最救命）。

**v1（核心价值）**
- §4.1 **对话改配置**：diff 卡片 + 【应用】+ 双向留痕
- §4.6 **存档**：冻结 / 载入 / 存档树 / 产物复制到 `saves/`
- §6 D 代码路径探针、§6 E 集合通信清单、§7 全部规则、§6 G 两次对比
- **§11 BotCall 的 `parse` 与 `explain`**

> 存档放 v1 不放 v2，是因为**产物生命周期这件事补不回来**：
> v0 上线三十天后临时桶开始清理，那之前的东西就永远拿不回来了。
> 要么早做，要么在 v0 就先把产物保留期设成永久（更贵，但至少不丢）。

**v2（锦上添花）**
LLO 分析、batch 上限自动二分、编译产物直接下载（省掉训练时的编译）、
多拓扑一次性对比（同一配置在 64/128/256 芯片上分别问一遍）、
存档之间的 diff 视图（两个存档并排，配置与分析逐项对照）、
**BotCall 的 `ask` / `param_help` / `propose_rule`**（最后一条让规则库自己长大）。

---

## 9.5 部署

跟 XProf 那套一致：**服务跑在本机，通过跳板机反向代理暴露，带鉴权**。

```
浏览器 → (跳板机 Caddy，带鉴权) → /tpuguru/*  ──strip_prefix──▶  本机 :PORT
```

- Caddy 侧只需 `uri strip_prefix /tpuguru` + `reverse_proxy <内网IP>:<PORT>`，
  **后端不用感知前缀**（前端资源用相对路径即可）
- systemd 常驻 + `Restart=always`
- worker 与 web 同机，直接调本地 docker

---

## 10. 不做什么

- **不预测吞吐。** AOT 没有实测时间，任何 TFLOP/s 都是猜的。
  它只回答「能不能跑、走的哪条路、有没有踩坑」。
- **不替代真机验证。** 数值正确性要看
  [logits 上的 KL](../kernel-equivalence-validation/)，AOT 给不了。
- **不做多机编排。** 这是个「上机前的体检站」，不是调度器。

---

## 11. ★ BotCall —— 把 tpuguru 接成对话通道

**核心想法**：现有的 bot 已经接了飞书、Discord 等 channel。
**再接一个叫 `tpuguru` 的 channel**，页面上的每一次交互都变成一轮对话。
于是「命令解析 / 配置提议 / 结果解读 / 参数答疑 / 追问」全部由同一个 agent 完成，
不必为每种能力单独写规则。

### 11.1 五种调用场景

| 场景 | 输入 | 期望产出 |
|---|---|---|
| `parse` | 用户粘的原始命令 | 结构化参数 + **置信度** + 不确定项清单 |
| `explain` | 一次 run 的完整结果 | 人话结论 + 「下一步建议做什么」 |
| `ask` | 用户对某次 run 的追问 | 带该 run 全部上下文的回答 |
| `param_help` | 点了某个参数的问号又追问 | 在静态文案基础上结合当前配置回答 |
| `propose_rule` | 一次失败的完整现场 | **新 lint 规则草稿**，人工确认后入 `tpuguru_rules` |

最后一条是复利：**每踩一个新坑，就沉淀成一条规则，下次自动拦住。**

### 11.2 四条约束

1. **确定性优先。** 能用规则做的不交给 LLM —— 快、可复现、不花钱。
   LLM 只负责「规则搞不定的」和「要说人话的」。
2. **输出必须可回放校验。** `parse` 的结果要能被确定性代码重新生成命令并比对，
   不一致就标红请用户确认，**不静默采纳**。
3. **必须能降级。** LLM 不可用时退回纯规则解析 + PARAMS.md 的静态文案，页面照常可用。
4. **每次调用留痕。** prompt / response / model / tokens 写进子集合，可回溯、可复盘。

### 11.3 接口与留痕

```jsonc
// POST /api/bot   { "kind": "parse|explain|ask|param_help|propose_rule",
//                   "run_id": "...", "text": "...", "context": {...} }

// 子集合 tpuguru/{run_id}/bot_calls/{call_id}
{
  "kind": "explain", "at": "...", "model": "...", "latency_ms": 3120,
  "input_digest": "sha256:...",        // 不存全文，存摘要 + 指针
  "output": { ... },                    // 结构化，前端直接渲染
  "tokens": { "in": 1234, "out": 567 },
  "accepted": true                      // 用户是否采纳了它的建议
}
```

`accepted` 这一列很重要 —— **它是评估这个 agent 有没有用的唯一客观指标**。

### 11.4 喂给 agent 的上下文

一次 `explain` 至少要带上：**输入配置 + lint 结果 + 全部 analyses + 同一 `parent_id` 树上的兄弟 run**。
最后一项让它能说出「你上次只改了 batch，这次还改了校准方式，慢的那 7.7% 主要来自后者」——
这正是人最想要、而规则写不出来的那种回答。

### 11.5 为什么这条很重要

参数有几百个、坑有几十种，**把它们全写成规则既写不完也维护不动**。
静态文案（PARAMS.md）负责「常见的 40 个参数」，
BotCall 负责「剩下的长尾 + 组合起来才出现的问题 + 用户当场想问的东西」。
两者互补：**规则保证下限，对话拓展上限。**

---

## 12. 端到端复盘：混元 3 那一轮，页面是怎么反馈的

拿 2026-08-14~16 的真实过程在工作台上完整走了一遍，逐步核对反馈对不对。
**这是这个工具的验收标准** —— 它得能在当时那个岔路口把人拦住。

| 步 | 动作 | 页面应该给什么 | 实际 |
|---|---|---|---|
| ① | 贴起点命令（BF16 / pdbs 7 / FSDP 64） | 认出型号、标「我方实测」、警告没传 tile | ✅ 445.1 |
| ② | 后端切 tokamax，加 FP8 + 专家维分片 + `fixed` | 不报致命（tokamax 分支不漏算），列出「通信藏不住」的代价 | ✅ 677.0 |
| ③ | **后端改回 native**（当时的岔路口） | 🔴 **立刻报 L1 致命**；跑出来标题必须是「编译通过，但结果无效」；1014.8 划掉；存档**自动作废** | ✅ 全中 |
| ④ | 关专家维分片、FSDP 吃满、batch 13、补 tile | 通过；提醒 `fixed` 伤收敛（L2） | ✅ 727.0 |
| ⑤ | 换 `absmax` → 13 OOM → 12 **也** OOM | 必须显形「**12 比 13 更超**，显存不随 batch 单调」 | ✅ 95.51 / 96.00 |
| ⑥ | batch 11 | 通过 | ✅ 670.8 |
| ⑦ | BF16 对照 | 通过 | ✅ 666.6 |

27 项断言 27 项通过（一项是断言文案写错，功能正确）。

### 这一轮改掉的四处设计问题

**① 「编译通过 + 一个漂亮数字」是最危险的呈现。**
第 ③ 步 AOT 确实编译通过了，早期版本就照直显示 `✅ 编译通过` 加 1014.8。
现在：作废横幅顶到页首、标题改成「编译通过，但结果无效」、所有 KPI 加删除线。

**② 结论无效要自动传播到存档，不能靠人记得去点。**
一个划掉的标题旁边打着 ✅ 是自相矛盾的。现在 `result.invalid` 存在时存档直接
`voided`，原因就是那段说明。

**③ 线性调优是一条链，不是树。**
每存一次就缩进一级，六步之后已经偏到屏幕右边。现在只有**真分叉**才缩进，
一路调下去用时间线（竖线 + 圆点）串起来。

**④ 口语别名匹配必须带词边界。**
`shard_exp_on_fsdp=True` 里的 `fsdp` 子串被当成了 `ici_fsdp_parallelism`，
生成了一条 `ici_fsdp_parallelism = True` 的提议 ——
**一个会静默改错参数的助手，正是这个工具存在要防的那件事。**
现在先吃显式 `k=v` 并把命中区间从文本里挖掉，再用带边界的别名扫剩下的；
数值参数拿到布尔值一律丢弃。

---

## 附：一句话判据

> **任何会改变显存占用或代码路径的改动 —— batch、分片宽度、精度、校准方式、
> tile、kernel 开关 —— 上机之前先在这里问一遍。**
