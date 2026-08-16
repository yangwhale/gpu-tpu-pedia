# WebAOT — 贴一条训练命令，上机之前把能问的都问完

**一句话**：把训练命令粘进输入框，它在 CPU 上跑一次 AOT 编译，
回答「**装得下吗 / 走的是哪条代码路径 / 配置有没有踩已知的坑**」，
全程**不占一张加速卡**。

> 状态：设计稿。实现分三期，见 §9。

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
```

- **Worker 用 docker 跑**，镜像与生产同一个 tag（编译器版本必须一致，否则结论不可迁移）
- 并发按 `--cpus` 切分；一台 80 核机器可并行 3–5 个 80 层任务
- 任务队列先用 Firestore 里的 `status` 字段轮询，不引入额外中间件

---

## 4. 输入解析

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

## 8. Firestore schema

collection **`webaot`**，一条 run 一个 doc：

```jsonc
{
  "id": "aot_20260816_130500_a1b2",
  "created_at": "2026-08-16T13:05:00+08:00",
  "created_by": "<user>",
  "title": "FP8 native FSDP128 pdbs13 absmax",     // 用户可编辑，默认自动生成
  "status": "queued | running | done | failed",

  "input": {
    "raw_cmd": "...",                               // 原样保存，可复制重跑
    "parsed": { "per_device_batch_size": 13, ... },
    "xla_flags": ["--xla_tpu_dvfs_p_state=7", ...],
    "topology": "tpu7x-128",
    "chips": 64,
    "layers": 80,
    "image": "<镜像 tag>"                            // 编译器版本，结论可迁移性的前提
  },

  "lint": [ { "rule": "L2", "severity": "warn", "msg": "..." } ],

  "result": {
    "compiled": false,
    "failure": { "kind": "hbm_oom_runtime", "required_gb": 95.51, "available_gb": 94.74 },
    "memory": { "argument_gb": ..., "output_gb": ..., "temp_gb": ..., "peak_gb": ... },
    "compile_time": { "hlo_passes_s": 15.2, "backend_passes_s": 23.0,
                      "code_gen_s": 6.4, "end_to_end_s": 45.8 },
    "codepath": {
      "moe_kernel": "megablox",
      "weight_pspec": "P('fsdp', None, None)",
      "kernel_rhs_shape": [3, 4096, 1536],
      "weight_gather_axes": [["fsdp", 0]],
      "quantization": { "enabled": true, "weight_calib": "absmax", "channel_axes": [2] }
    },
    "hlo": { "op_count": ..., "fusion_count": ...,
             "collectives": [ { "kind": "all-gather", "shape": "...", "axis": "fsdp",
                                "async": true, "in_loop": false } ] }
  },

  "artifacts": {                                    // 大文件不进 Firestore
    "stdout_uri": "gs://.../stdout.log",
    "hlo_uri":    "gs://.../hlo.tgz",
    "pickle_uri": "gs://.../compiled.pkl"           // 编译产物，可直接拿去跑训练
  }
}
```

**大文件放对象存储，Firestore 只存指针**（单 doc 有 1 MB 限制，HLO 动辄 10 MB+）。

---

## 9. 分期

**v0（先能用）**
输入框 → 转换 → 跑 → 报告 §6 A/B/C → 历史列表 → Firestore。
lint 只做 L1/L3/L4/L5（这四条都是纯静态判断，最省事也最救命）。

**v1（核心价值）**
加 §6 D 代码路径探针、§6 E 集合通信清单、§7 全部规则、§6 G 两次对比。

**v2（锦上添花）**
LLO 分析、batch 上限自动二分、编译产物直接下载（省掉训练时的编译）、
多拓扑一次性对比（同一配置在 64/128/256 芯片上分别问一遍）。

---

## 10. 不做什么

- **不预测吞吐。** AOT 没有实测时间，任何 TFLOP/s 都是猜的。
  它只回答「能不能跑、走的哪条路、有没有踩坑」。
- **不替代真机验证。** 数值正确性要看
  [logits 上的 KL](../kernel-equivalence-validation/)，AOT 给不了。
- **不做多机编排。** 这是个「上机前的体检站」，不是调度器。

---

## 附：一句话判据

> **任何会改变显存占用或代码路径的改动 —— batch、分片宽度、精度、校准方式、
> tile、kernel 开关 —— 上机之前先在这里问一遍。**
