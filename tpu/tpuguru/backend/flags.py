"""XLA flag 知识库 —— 这一套是**积累**，不是配置项列表。

每条 flag 记四样：它干什么、默认值是多少、开了会怎样、**在哪一代上验过**。

`evidence` 分三档，界面上用不同标记，**不许混着报**：
  measured   我们在 v7 上实测过，带数字
  compiler   从 `tpu_comp_env.txt` 读出来的（默认值、是否被识别）—— 硬事实
  inferred   看文档或从行为推的，**没实测**

`gen` 说的是**我们验过哪一代**，不是「只能用在哪一代」。
v5p 那一列基本都是「没验过」—— 与其瞎标兼容，不如老实说没试过。
"""

from __future__ import annotations

GROUPS = [
    {"id": "clock", "label": "频率与功耗", "desc": "最便宜的一类：不动显存、不动数值。"},
    {"id": "vmem", "label": "VMEM / 片上内存", "desc": "撞 VMEM OOM 时看这里 —— 它跟 HBM 是两层内存。"},
    {"id": "sched", "label": "调度与延迟隐藏", "desc": "决定通信能不能被计算盖住。**有依赖关系，不能挑着开。**"},
    {"id": "sc", "label": "SparseCore 卸载", "desc": "把集合通信卸到 SparseCore。⚠️ 这一族要整套开。"},
    {"id": "comm", "label": "集合通信", "desc": "合并阈值与异步化。"},
]

FLAGS = [
    # ── 频率 ──────────────────────────────────────────────────
    {"flag": "xla_tpu_dvfs_p_state", "group": "clock", "type": "int",
     "default": "-1", "suggest": "7",
     "what": "锁定 TPU 的 DVFS 电压/频率档位。7 是最高档。",
     "effect": "**+8.0% 吞吐，显存一字节不涨** —— 性价比最高的一个开关，几乎没有不开的理由。",
     "risk": "功耗与温度上升；共享集群上属于正常用法。",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured",
     "note": "2026-08 在 64 芯片 v7 上实测 +8.0%。"},

    # ── VMEM ─────────────────────────────────────────────────
    {"flag": "xla_tpu_scoped_vmem_limit_kib", "group": "vmem", "type": "int",
     "default": "-1（编译器自选，实际落在 64–70 MB）", "suggest": "65472",
     "what": "单个 fusion 能用的 scoped VMEM 上限。v7 物理 VMEM 约 128 MB。",
     "effect": "撞 `CompileTimeScopedVmemOom` 时**先试这个**。默认远低于物理上限，"
               "留白是给 double buffer 做 prefetch 重叠用的。",
     "risk": "⚠️ 抬太满会把 prefetch 挤掉 —— **编译过了但吞吐掉**。"
             "改完必须回头看 step time；掉超过 10% 说明该降 tile / block size 而不是继续抬。",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured",
     "note": "VMEM 跟 HBM 是两层内存。**降 batch 对 VMEM OOM 基本没用** —— "
             "batch 通常是 fusion 的最外层循环维，进 VMEM 的 tile 大小不随它变。"},

    # ── 调度 ─────────────────────────────────────────────────
    {"flag": "xla_tpu_enable_latency_hiding_layer_scheduler", "group": "sched", "type": "bool",
     "default": "auto", "suggest": "true",
     "what": "按层做延迟隐藏调度，让集合通信藏进计算里。",
     "effect": "MoE + FSDP 这类通信重的配置上是关键开关。",
     "risk": "🔴 **必须同时开 `xla_tpu_enable_sparse_core_collective_aggregator`**，"
             "否则编译器直接拒绝：`INVALID_ARGUMENT: Latency hiding layer scheduler "
             "requires sparse core collective aggregator to be enabled`。",
     "requires": ["xla_tpu_enable_sparse_core_collective_aggregator"],
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured"},
    {"flag": "xla_tpu_enable_layer_scheduler_for_dependent_collectives", "group": "sched",
     "type": "bool", "default": "auto", "suggest": "true",
     "what": "让层调度器也管有依赖关系的集合通信。",
     "effect": "配合上一条使用。", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured"},
    {"flag": "xla_tpu_enable_multi_compute_overlap_in_layer_scheduler", "group": "sched",
     "type": "bool", "default": "auto", "suggest": "false",
     "what": "允许多段计算互相重叠。",
     "effect": "我们这套配方里是**关掉**的。",
     "risk": "开关方向跟直觉相反 —— 别看名字就开。没实测过开着的效果。",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured"},
    {"flag": "xla_tpu_scheduler_percent_shared_memory_limit", "group": "sched", "type": "int",
     "default": "95", "suggest": "150",
     "what": "调度器可用共享内存的百分比上限，可以超过 100。",
     "effect": "放宽后调度器有更多腾挪空间。", "risk": "过高可能挤占别的缓冲。",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured"},

    # ── SparseCore 卸载（整族）─────────────────────────────────
    {"flag": "xla_tpu_enable_sparse_core_collective_aggregator", "group": "sc", "type": "bool",
     "default": "auto", "suggest": "true",
     "what": "SparseCore 上真正做规约的那个部件。",
     "effect": "延迟隐藏调度器的**硬依赖**。",
     "risk": "🔴 漏了它而开了调度器 → 编译器直接拒绝。",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},
    {"flag": "xla_tpu_enable_sparse_core_collective_offload_all_gather", "group": "sc",
     "type": "bool", "default": "auto", "suggest": "true",
     "what": "把 all-gather 卸载到 SparseCore。", "effect": "", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},
    {"flag": "xla_tpu_enable_sparse_core_collective_offload_2d_all_gather", "group": "sc",
     "type": "bool", "default": "auto", "suggest": "true",
     "what": "二维 all-gather 的卸载。", "effect": "", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},
    {"flag": "xla_tpu_enable_sparse_core_collective_offload_3d_all_gather", "group": "sc",
     "type": "bool", "default": "auto", "suggest": "true",
     "what": "三维 all-gather 的卸载。", "effect": "", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},
    {"flag": "xla_tpu_enable_sparse_core_collective_offload_all_reduce", "group": "sc",
     "type": "bool", "default": "auto", "suggest": "true",
     "what": "all-reduce 卸载。", "effect": "", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},
    {"flag": "xla_tpu_enable_sparse_core_collective_offload_reduce_scatter", "group": "sc",
     "type": "bool", "default": "auto", "suggest": "true",
     "what": "reduce-scatter 卸载。", "effect": "", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},
    {"flag": "xla_tpu_enable_sparse_core_reduce_scatter_v2", "group": "sc", "type": "bool",
     "default": "auto", "suggest": "true",
     "what": "reduce-scatter 的 v2 实现。", "effect": "", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},
    {"flag": "xla_tpu_use_tc_device_shape_on_sc", "group": "sc", "type": "bool",
     "default": "false", "suggest": "True",
     "what": "SparseCore 上沿用 TensorCore 的 device shape。", "effect": "", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},
    {"flag": "xla_sc_disable_megacore_partitioning", "group": "sc", "type": "bool",
     "default": "false", "suggest": "True",
     "what": "关掉 SparseCore 的 megacore 分区。", "effect": "", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured", "family": True},

    # ── 通信 ─────────────────────────────────────────────────
    {"flag": "xla_enable_async_all_gather", "group": "comm", "type": "bool",
     "default": "auto", "suggest": "true",
     "what": "all-gather 异步化，让它能跟计算重叠。",
     "effect": "通信隐藏的前提条件之一。", "risk": "",
     "gen": {"v7": "measured", "v5p": "未验证"}, "evidence": "measured"},
    {"flag": "xla_all_gather_combiner_threshold_count", "group": "comm", "type": "int",
     "default": "256", "suggest": "",
     "what": "多少个 all-gather 会被合并成一个。",
     "effect": "合并能减少发起次数（次数是通信开销的大头），但会推迟最早那个的完成时间。",
     "risk": "**没实测过调它的效果**，默认值一般够用。",
     "gen": {"v7": "inferred", "v5p": "未验证"}, "evidence": "inferred"},
]

# 一键套餐 —— 有依赖关系的一族必须整套开，让人手点八个太容易漏
PRESETS = [
    {"id": "hy3-v7", "label": "v7 生产配方",
     "desc": "Hunyuan3 / Qwen3 在 64 芯片 v7 上跑通的那一套。锁频 + 延迟隐藏 + SparseCore 全族卸载。",
     "flags": {f["flag"]: f["suggest"] for f in FLAGS
               if f.get("suggest") and f["group"] in ("clock", "vmem", "sched", "sc", "comm")
               and f["flag"] != "xla_all_gather_combiner_threshold_count"}},
    {"id": "sc-family", "label": "只开 SparseCore 全族",
     "desc": "⚠️ 这一族有依赖关系，**整套开**。漏掉聚合器那个，编译器会直接拒绝。",
     "flags": {f["flag"]: f["suggest"] for f in FLAGS if f["group"] == "sc"}},
    {"id": "clock-only", "label": "只锁频",
     "desc": "最保守的一步：+8% 吞吐，不动显存、不动数值。想先试点什么就从这个开始。",
     "flags": {"xla_tpu_dvfs_p_state": "7"}},
]


def catalog() -> dict:
    # 套餐里有几个 flag 由 flags 本身算，**不在 label 里写死**
    # —— 写死的计数会随着库长大而悄悄变成假话（SC 族 8→9 就漂过一次）。
    presets = [dict(p, count=len(p["flags"])) for p in PRESETS]
    return {"groups": GROUPS, "flags": FLAGS, "presets": presets}


def lint_flags(xla: dict) -> list[dict]:
    """flag 之间的依赖检查 —— 这类错编译器要到很后面才报，早点拦住便宜得多。"""
    have = {k.lstrip("-") for k in xla}
    out = []
    for f in FLAGS:
        for req in f.get("requires", []):
            if f["flag"].lstrip("-") in have and req not in have:
                out.append({"severity": "fatal", "flag": f["flag"], "missing": req,
                            "why": f["risk"]})
    sc = [f["flag"] for f in FLAGS if f["group"] == "sc"]
    on = [x for x in sc if x in have]
    if on and len(on) < len(sc):
        out.append({"severity": "warn", "flag": "SparseCore 族",
                    "missing": "、".join(x for x in sc if x not in have),
                    "why": "这一族有依赖关系，开了一部分通常没意义甚至会被拒。整套开。"})
    return out
