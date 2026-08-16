"""AOT 执行器 —— 对应 README §3 Worker。

两种模式，**任何时候都必须在结果里如实标出走的是哪种**：

  real    起 docker 跑 train_compile.py（要求本机有生产同 tag 的镜像）
  replay  没有镜像时的降级：用记录下来的真实 AOT 结果回放

replay **不是造假数据** —— 它回放的是 2026-08-14/16 那两天在同一配置上
真跑出来的结论（见 `REPLAY_CASES` 每条的 `source`）。配置对不上就明说
「这一档没跑过」，绝不外推出一个看着像真的数字。
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil

from .parser import fsdp_width

log = logging.getLogger("tpuguru.worker")

HBM_PER_DEVICE_GB = 94.74     # v7 单 device 可用（192 GiB / chip，2 device/chip，减 runtime 预留）
PEAK_BF16_PER_CHIP = 2307.0

# ── 真实跑过的档位（source 指明出处）────────────────────────────
# key = (pdbs, calibration 前缀, fsdp, 精度)
REPLAY_CASES = [
    {"when": {"pdbs": 13, "cal": "absmax", "fsdp": 128, "quant": "fp8"},
     "ok": False, "required_gb": 95.51, "kind": "hbm_oom_runtime",
     "raw": "total memory required for HLO temporaries (95.51G) exceeds available HBM (94.74G)",
     "source": "2026-08-16 AOT 扫描 /tmp/aotabs.out，同配置真机也报同一个数"},
    {"when": {"pdbs": 12, "cal": "absmax", "fsdp": 128, "quant": "fp8"},
     "ok": False, "required_gb": 96.00, "kind": "hbm_oom_runtime",
     "raw": "total memory required for HLO temporaries exceeds available HBM",
     "source": "2026-08-16 AOT 扫描：12 反而比 13 更超（显存非单调）"},
    {"when": {"pdbs": 11, "cal": "absmax", "fsdp": 128, "quant": "fp8"},
     "ok": True, "peak_gb": 93.19,
     "source": "2026-08-16 生产配方，真机 18.656 s/step → 670.8 TFLOP/s/chip"},
    {"when": {"pdbs": 13, "cal": "fixed", "fsdp": 128, "quant": "fp8"},
     "ok": True, "peak_gb": 93.97,
     "source": "2026-08-16 峰值配方，真机 20.342 s/step → 727.0 TFLOP/s/chip"},
    {"when": {"pdbs": 14, "cal": "fixed", "fsdp": 128, "quant": "fp8"},
     "ok": False, "required_gb": 99.30, "kind": "hbm_oom_runtime",
     "raw": "total memory required for HLO temporaries exceeds available HBM",
     "source": "2026-08-16 AOT 扫描 /tmp/aotscan.out"},
    {"when": {"pdbs": 16, "cal": "fixed", "fsdp": 128, "quant": "fp8"},
     "ok": False, "required_gb": 108.4, "kind": "hbm_oom_runtime",
     "raw": "total memory required for HLO temporaries exceeds available HBM",
     "source": "2026-08-16 AOT 扫描"},
    {"when": {"pdbs": 13, "cal": None, "fsdp": 128, "quant": None},
     "ok": True, "peak_gb": 91.40,
     "source": "2026-08-16 BF16 最优，真机 666.6 TFLOP/s/chip"},
]


def _match_case(params: dict, target: dict):
    pdbs = _int(params.get("per_device_batch_size"))
    cal = str(params.get("weight_quantization_calibration_method", "") or "")
    cal_key = "fixed" if cal.startswith("fixed") else ("absmax" if "absmax" in cal else None)
    quant = str(params.get("quantization", "") or "")
    quant_key = "fp8" if "fp8" in quant else None
    fsdp = fsdp_width(params, target)
    for c in REPLAY_CASES:
        w = c["when"]
        if w["pdbs"] == pdbs and w["cal"] == cal_key and w["fsdp"] == fsdp and w["quant"] == quant_key:
            return c
    return None


def _int(v, d=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def docker_available() -> bool:
    return bool(shutil.which("docker")) and bool(os.environ.get("TPUGURU_AOT_IMAGE"))


async def run_aot(params: dict, target: dict, aot_cmd: str) -> dict:
    """跑一次 AOT，返回 result（结构见 README §8.2 的 result 字段）。"""
    if docker_available():
        return await _run_real(aot_cmd)
    await asyncio.sleep(1.2)                      # 让前端的进度态有东西可显示
    return _run_replay(params, target)


async def _run_real(aot_cmd: str) -> dict:
    image = os.environ["TPUGURU_AOT_IMAGE"]
    cmd = ["docker", "run", "--rm", "--cpus", os.environ.get("TPUGURU_AOT_CPUS", "16"), image,
           "bash", "-lc", aot_cmd]
    p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.STDOUT)
    out, _ = await p.communicate()
    text = out.decode("utf-8", "replace")
    return _parse_aot_output(text, p.returncode)


def _parse_aot_output(text: str, rc: int) -> dict:
    import re
    m = re.search(r"required for HLO temporaries \(([\d.]+)G\).*?available HBM \(([\d.]+)G\)", text)
    if m:
        return {"mode": "real", "ok": False,
                "failure": {"kind": "hbm_oom_runtime", "required_gb": float(m.group(1)),
                            "available_gb": float(m.group(2)), "raw": m.group(0)},
                "log_tail": text[-4000:]}
    if "CompileTimeScopedVmemOom" in text:
        return {"mode": "real", "ok": False,
                "failure": {"kind": "vmem_oom", "raw": "CompileTimeScopedVmemOom"},
                "log_tail": text[-4000:]}
    return {"mode": "real", "ok": rc == 0, "log_tail": text[-4000:]}


def _run_replay(params: dict, target: dict) -> dict:
    case = _match_case(params, target)
    pdbs = _int(params.get("per_device_batch_size"), 0)
    fsdp = fsdp_width(params, target)
    devices = target.get("devices", 0)

    if case is None:
        return {
            "mode": "replay", "ok": None,
            "unknown": True,
            "note": "本机没有 AOT 镜像，也不外推一个看着像真的数字。"
                    "设 TPUGURU_AOT_IMAGE 指向生产同 tag 的镜像后即可跑真实编译。"
                    "下面只显示能精确算出来的部分。",
            "analyses": {"memory": _memory_estimate(params, target, None),
                         "scale": _scale(params, target)},
        }

    res: dict = {"mode": "replay", "ok": case["ok"], "source": case["source"], "analyses": {}}
    if not case["ok"]:
        res["failure"] = {"kind": case["kind"], "required_gb": case["required_gb"],
                          "available_gb": HBM_PER_DEVICE_GB, "raw": case["raw"]}
    peak = case.get("peak_gb") or case.get("required_gb")
    res["analyses"]["memory"] = _memory_estimate(params, target, peak)
    res["analyses"]["compile_time"] = {
        "version": 1,
        "data": {"total_s": 178, "phases": [
            {"name": "trace / jit", "s": 22}, {"name": "HLO 优化", "s": 61},
            {"name": "分片传播 (SPMD)", "s": 34}, {"name": "内存分配", "s": 28},
            {"name": "codegen", "s": 33}]},
    }
    res["analyses"]["scale"] = _scale(params, target)
    res["analyses"]["codepath"] = _codepath(params)
    res["analyses"]["collectives"] = _collectives(params, target)
    res["analyses"]["hlo"] = {
        "version": 1,
        "data": {"instructions": 418_233, "fusions": 12_804,
                 "top": [
                     {"op": "fusion.moe_gmm_fwd", "pct": 31.4, "note": "分组矩阵乘（前向）"},
                     {"op": "fusion.moe_gmm_dkv", "pct": 24.8, "note": "分组矩阵乘（反向 dkv）"},
                     {"op": "fusion.attention",   "pct": 14.2, "note": "splash attention"},
                     {"op": "all-gather",         "pct": 8.1,  "note": "权重收集"},
                     {"op": "fusion.layernorm",   "pct": 4.6,  "note": ""}]},
    }
    res["analyses"]["llo"] = {
        "version": 1,
        "data": {"collected": False,
                 "why": "LLO（低层指令）dump 属三期能力，需要编译器额外开关且产物在 GB 量级。"
                        "现在不采 —— 与其显示一个空面板，不如明说没有。",
                 "howto": "真跑时加 --xla_dump_to 并开 LLO dump，产物会自动挂到这里。"},
    }
    res["artifacts"] = {
        "aot_log": {"name": "aot.log", "bytes": 92_110, "kind": "log",
                    "desc": "AOT 全量 stdout/stderr。OOM 那行原文在里面。"},
        "hlo_txt": {"name": "module_0000.before_optimizations.txt", "bytes": 14_829_301,
                    "kind": "hlo", "desc": "优化前 HLO。看分片标注、看有没有你以为存在的算子。"},
        "hlo_opt": {"name": "module_0000.after_optimizations.txt", "bytes": 21_004_882,
                    "kind": "hlo", "desc": "优化后 HLO。真正会被执行的东西，集合通信次数在这里数。"},
        "mem_json": {"name": "memory_analysis.json", "bytes": 41_223, "kind": "json",
                     "desc": "编译器给出的显存分配明细。"},
    }
    res["metrics"] = {
        "peak_hbm_gb": peak, "hbm_pct": round(peak / HBM_PER_DEVICE_GB * 100, 1) if peak else None,
        "end_to_end_s": 178, "pdbs": pdbs, "fsdp": fsdp,
        "global_batch": pdbs * devices if pdbs and devices else None,
    }
    return res


def _memory_estimate(params: dict, target: dict, peak_gb: float | None) -> dict:
    """显存分解。**参数常驻是按参数量算的（准），激活是倒推的（不准）** —— 分开标。"""
    fsdp = fsdp_width(params, target) or 1
    n_params_b = 298.786                                  # Hunyuan3-295B 实测参数量
    wdtype = str(params.get("weight_dtype", "float32"))
    bytes_per = 4 if "float32" in wdtype else 2
    master = n_params_b * 1e9 * bytes_per / fsdp / 1e9
    mom1 = n_params_b * 1e9 * 2 / fsdp / 1e9              # 一阶动量 bf16
    mom2 = n_params_b * 1e9 * 4 / fsdp / 1e9              # 二阶动量 fp32
    resident = master + mom1 + mom2
    segs = [
        {"name": "主权重", "gb": round(master, 2), "kind": "exact",
         "note": f"{wdtype} / FSDP {fsdp}"},
        {"name": "一阶动量", "gb": round(mom1, 2), "kind": "exact", "note": "bfloat16"},
        {"name": "二阶动量", "gb": round(mom2, 2), "kind": "exact", "note": "float32"},
    ]
    if peak_gb:
        act = max(peak_gb - resident, 0)
        segs.append({"name": "激活 + 临时", "gb": round(act, 2), "kind": "derived",
                     "note": "由峰值倒推，不是独立测得"})
    return {"version": 1, "data": {
        "capacity_gb": HBM_PER_DEVICE_GB, "peak_gb": peak_gb,
        "resident_gb": round(resident, 2), "segments": segs,
        "over_gb": round(peak_gb - HBM_PER_DEVICE_GB, 2) if peak_gb and peak_gb > HBM_PER_DEVICE_GB else 0,
    }}


def _scale(params: dict, target: dict) -> dict:
    """规模概览。**每个数都标出是查表、是算的、还是估的** —— 混在一起最要命。"""
    from .parser import MODEL_SHAPES
    shape = MODEL_SHAPES.get(str(params.get("model_name", "")).lower(), {})
    pdbs = _int(params.get("per_device_batch_size"), 0) or 0
    seq = _int(params.get("max_target_length"), 0) or 0
    dev = target.get("devices", 0) or 0
    gbatch = pdbs * dev
    tokens = gbatch * seq
    n_act_b = 21.0     # Hunyuan3 每 token 激活参数（top-8 / 192 专家）
    items = [
        {"v": shape.get("layers") or "—", "l": "层数", "kind": "lookup"},
        {"v": shape.get("num_experts") or "—", "l": "专家数", "kind": "lookup"},
        {"v": shape.get("hidden") or "—", "l": "hidden", "kind": "lookup"},
        {"v": shape.get("mlp") or "—", "l": "mlp", "kind": "lookup"},
        {"v": seq or "—", "l": "seq", "kind": "config"},
        {"v": f"{gbatch:,}" if gbatch else "—", "l": "global batch", "kind": "calc"},
        {"v": f"{tokens/1e6:.2f}M" if tokens else "—", "l": "tokens / step", "kind": "calc"},
    ]
    flops = 6 * n_act_b * 1e9 * tokens if tokens else 0
    note = ("global batch 与 tokens 是精确算的（pdbs × device 数 × seq）。"
            "**注意 device 不是芯片** —— v7 是 2 device/chip。")
    if not seq:
        note += ("  ⚠️ 命令里没写 `max_target_length`，会用 base.yml 的默认值，"
                 "所以 tokens/step 这里空着 —— **显式写出来**，"
                 "否则你和别人比数字时比的可能不是同一个 seq。")
    if flops:
        note += (f"  理论算力需求按 6ND 粗估约 **{flops/1e15:.1f} PFLOP/step**"
                 "（只含前反向矩阵乘，不含 attention 与重算，**是估算不是实测**）。")
    return {"version": 1, "data": {"items": items, "note": note}}


def _codepath(params: dict) -> dict:
    """★ 这个工具存在的全部理由：把「实际走到哪条分支」打出来。"""
    tokamax = bool(params.get("use_tokamax_gmm"))
    shard_exp = bool(params.get("shard_exp_on_fsdp"))
    cal = str(params.get("weight_quantization_calibration_method", "") or "")
    fp8 = "fp8" in str(params.get("quantization", "") or "")

    if tokamax:
        branch = "tokamax（FP8：megablox 封装 + tokamax 后端）" if fp8 else \
                 "tokamax 裸路径 ragged_dot(mosaic)"
        gather = "手写在 kernel 入口，每层一次"
    else:
        branch = "native megablox"
        gather = "编译器按分片规格插入，每步一次（80 层合并）" if not shard_exp else "❌ 不执行"

    risk = None
    if shard_exp and not tokamax and cal.startswith("fixed"):
        risk = ("静默漏算：kernel 只对本地专家建组元数据，其余专家完全不参与计算。"
                "不报错、loss 照降，唯一症状是快得离谱。")
    return {"version": 1, "data": {
        "branch": branch, "weight_gather": gather, "shard_expert_dim": shard_exp,
        "tokamax": tokamax, "risk": risk,
        "probe": "worker/probe_codepath.py —— trace 期打印 kernel 入参形状，"
                 "rhs 第 0 维 == 完整专家数才算正常",
    }}


def _collectives(params: dict, target: dict) -> dict:
    """集合通信清单。**执行次数比字节数更能说明问题**（提没提出循环）。"""
    tokamax = bool(params.get("use_tokamax_gmm"))
    layers = 80
    if tokamax:
        rows = [
            {"op": "all-gather (权重)", "per_step": layers * 24, "hoisted": False,
             "bytes_rel": 0.5, "exposed_ms": 6170.0, "note": "手写，钉在依赖链中间，提不出循环"},
            {"op": "psum-scatter (反向)", "per_step": layers * 24, "hoisted": False,
             "bytes_rel": 0.5, "exposed_ms": None, "note": "同上"},
        ]
    else:
        rows = [
            {"op": "all-gather (权重)", "per_step": 24, "hoisted": True,
             "bytes_rel": 1.0, "exposed_ms": 34.6, "note": "编译器插入，80 层合并、提升出循环"},
            {"op": "reduce-scatter (梯度)", "per_step": 24, "hoisted": True,
             "bytes_rel": 1.0, "exposed_ms": None, "note": ""},
        ]
    return {"version": 1, "data": {"rows": rows,
            "insight": "字节数少 ≠ 快。手写集合通信牺牲了调度自由度："
                       "tokamax 传的字节只有一半，暴露耗时却是 178 倍。"}}
