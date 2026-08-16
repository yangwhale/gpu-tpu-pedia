# WebAOT — 贴一条训练命令，上机之前把能问的都问完

**一句话**：把训练命令粘进输入框，它在 CPU 上跑一次 AOT 编译，
回答「**装得下吗 / 走的是哪条代码路径 / 配置有没有踩已知的坑**」，
全程**不占一张加速卡**。

> 状态：设计稿。实现分三期，见 §9。
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
① 粘贴训练命令  →  ② 自动转成 AOT  →  ③ 跑（CPU，~3 min）  →  ④ 分析报告
                                                                   ↓
                                                          ⑤ 存进历史，可回看/对比
```

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
WebAOT/
├── README.md              设计文档（本文件）
├── PARAMS.md              ★ 参数目录：控件类型 + 问号三段文案
├── backend/               FastAPI：解析、转换、投递、读写 Firestore
├── worker/                在 CPU 上跑 AOT
│   └── probe_codepath.py  ✅ 已验证可用的代码路径探针
├── analyzers/             把产物变成结构化结论，一个维度一个分析器
├── rules/
│   └── rules.seed.json    ✅ 9 条 lint 规则的种子数据
├── frontend/              提交 / 报告 / 历史三个页面
└── deploy/                systemd + 反代片段 + Firestore 索引
```

**两份已经能用的东西**：`worker/probe_codepath.py` 是 2026-08-16 实测跑通的探针，
`rules/rules.seed.json` 是 9 条规则的数据化版本（导入 `webaot_rules` 即可）。
其余目录只有职责说明，等实现。

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
Firestore  collection: webaot
  ▼
浏览器  ← 报告页 / 历史列表 / 两次对比
  ▲
  │  ⇅  BotCall channel（见 §11）—— 智能识别 / 结果解读 / 追问
  └──────────────────────────────────────────────────
```

- **Worker 用 docker 跑**，镜像与生产同一个 tag（编译器版本必须一致，否则结论不可迁移）
- 并发按 `--cpus` 切分；一台 80 核机器可并行 3–5 个 80 层任务
- 任务队列先用 Firestore 里的 `status` 字段轮询，不引入额外中间件

---

## 4. 输入解析（智能识别）

**像快递单的地址识别一样** —— 用户把原始命令整段贴进来，
系统自动拆出参数、填满表单控件、在下方生成格式化后的 AOT 命令。

```
┌─ 粘贴区 ────────────────────────────────────┐
│ LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7 …" \   │
│ python3 -m …train … per_device_batch_size=13 …    │
└──────────────────────────────────────────────┘
              ↓ 智能识别
┌─ 表单（自动填好，可改）──────────────────────┐
│ 拓扑 [tpu7x-128 ▼] batch [13 ▼] FSDP [-1 ▼] …│
└──────────────────────────────────────────────┘
              ↓ 实时生成
┌─ AOT 命令（格式化，可直接复制）──────────────┐
│ python3 -m …train_compile … compile_topology=… │
└──────────────────────────────────────────────┘
```

**两级解析，确定性优先**：

| 级别 | 谁做 | 处理什么 |
|---|---|---|
| ① 规则解析 | 确定性代码 | `k=v`、`--flag=v`、环境变量 —— **能用正则搞定的绝不交给 LLM** |
| ② 语义补全 | BotCall（§11） | 识别不了的片段、缺失的必填项（如拓扑）、互相冲突的参数、意图推断 |

**必须可校验**：把生成的 AOT 命令**再解析一遍**，与①的结果比对；
不一致就标红让用户确认，不静默采纳 LLM 的改写。



需要从命令里认出三类东西：

| 类别 | 例子 | 处理 |
|---|---|---|
| MaxText 配置项 | `per_device_batch_size=13` | 原样透传 |
| XLA flags | `LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7 ..."` | 转成 `compile_xla_flags="..."` |
| 运行时噪声 | `steps=100000`、`base_output_directory=gs://...`、checkpoint 相关 | 覆盖成 AOT 安全值 |

**拓扑要用户选或从命令推断**：AOT 必须知道目标硬件。
`compile_topology=tpu7x-128` 表示 **128 device = 64 芯片**（v7 是 2 device/chip，最容易写错的地方）。

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

### 8.2 collection `webaot` —— 一次运行一个 doc

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

### 8.3 collection `webaot_rules` —— lint 规则库

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

`status`、`created_by`、`tags`、`created_at desc`，
外加 `metrics.peak_hbm_gb`、`metrics.end_to_end_s` 用于排序。

---

## 9. 分期

**v0（先能用）**
输入框 → 转换 → 跑 → 报告 §6 A/B/C → 历史列表 → Firestore。
lint 只做 L1/L3/L4/L5（这四条都是纯静态判断，最省事也最救命）。

**v1（核心价值）**
加 §6 D 代码路径探针、§6 E 集合通信清单、§7 全部规则、§6 G 两次对比，
以及 **§11 BotCall 的 `parse` 与 `explain`** —— 智能识别和结果解读是用户最先感知到的价值。

**v2（锦上添花）**
LLO 分析、batch 上限自动二分、编译产物直接下载（省掉训练时的编译）、
多拓扑一次性对比（同一配置在 64/128/256 芯片上分别问一遍）、
**BotCall 的 `ask` / `param_help` / `propose_rule`**（最后一条让规则库自己长大）。

---

## 9.5 部署

跟 XProf 那套一致：**服务跑在本机，通过跳板机反向代理暴露，带鉴权**。

```
浏览器 → (跳板机 Caddy，带鉴权) → /webaot/*  ──strip_prefix──▶  本机 :PORT
```

- Caddy 侧只需 `uri strip_prefix /webaot` + `reverse_proxy <内网IP>:<PORT>`，
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

## 11. ★ BotCall —— 把 WebAOT 接成对话通道

**核心想法**：现有的 bot 已经接了飞书、Discord 等 channel。
**再接一个叫 `webaot` 的 channel**，页面上的每一次交互都变成一轮对话。
于是「智能识别 / 结果解读 / 参数答疑 / 追问」全部由同一个 agent 完成，
不必为每种能力单独写规则。

### 11.1 五种调用场景

| 场景 | 输入 | 期望产出 |
|---|---|---|
| `parse` | 用户粘的原始命令 | 结构化参数 + **置信度** + 不确定项清单 |
| `explain` | 一次 run 的完整结果 | 人话结论 + 「下一步建议做什么」 |
| `ask` | 用户对某次 run 的追问 | 带该 run 全部上下文的回答 |
| `param_help` | 点了某个参数的问号又追问 | 在静态文案基础上结合当前配置回答 |
| `propose_rule` | 一次失败的完整现场 | **新 lint 规则草稿**，人工确认后入 `webaot_rules` |

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

// 子集合 webaot/{run_id}/bot_calls/{call_id}
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

## 附：一句话判据

> **任何会改变显存占用或代码路径的改动 —— batch、分片宽度、精度、校准方式、
> tile、kernel 开关 —— 上机之前先在这里问一遍。**
