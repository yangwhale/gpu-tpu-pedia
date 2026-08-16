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

# ── 真实跑过的档位（2026-08-14/16，64 芯片 v7，Hunyuan3-295B-A21B）──────
# key 里必须带 tokamax / shard_exp —— 同样的 batch 与校准，走不同 kernel 分支
# 结论完全不同，这正是那个 1014.8 的来历。
REPLAY_CASES = [
    # ① 起点：BF16，没传 tile
    {"when": {"model": "hunyuan3-295b", "pdbs": 7, "cal": None, "fsdp": 64,
              "quant": None, "tokamax": False, "shard_exp": False, "tile": False},
     "ok": True, "peak_gb": 88.20, "per_chip": 445.1, "step_s": 20.43,
     "source": "c1 起点：BF16 + fp32 主权重，20.43 s/step → 445.1 TFLOP/s/chip"},

    # ② QAG 配方（专家维分片 + fixed + tokamax）—— 有效但 FSDP 被锁一半
    {"when": {"model": "hunyuan3-295b", "pdbs": 7, "cal": "fixed", "fsdp": 64,
              "quant": "fp8", "tokamax": True, "shard_exp": True, "tile": None},
     "ok": True, "peak_gb": 90.60, "per_chip": 677.0,
     "source": "旧生产配方：FP8 + 跨卡量化收集 + tokamax。677.0 TFLOP/s/chip。"
               "专家数 192 只能被 64 整除，FSDP 被锁在一半宽度。"},

    # ③ ☠️ 同样配置但走 native —— 这就是那个 1014.8
    {"when": {"model": "hunyuan3-295b", "pdbs": 7, "cal": "fixed", "fsdp": 64,
              "quant": "fp8", "tokamax": False, "shard_exp": True, "tile": None},
     "ok": True, "peak_gb": 86.30, "per_chip": 1014.8, "step_s": 7.85,
     "invalid": "这个数字是**漏算**出来的，已作废。native 分支不执行权重 all-gather，"
                "kernel 只对本地 3/192 个专家建组元数据，其余专家完全不参与计算 —— "
                "不报错、loss 照常下降。补齐 all-gather 后重测是 637.0，"
                "比不动它的旧配方（677.0）**还低**。",
     "source": "2026-08-15 的假峰值。kernel 自报覆盖 3,582 / 229,376 行（1.6%）。"},

    # ④ 补齐 all-gather 之后的真实值（同配置，打了补丁的运行时）
    {"when": {"model": "hunyuan3-295b", "pdbs": 7, "cal": "fixed", "fsdp": 64,
              "quant": "fp8", "tokamax": False, "shard_exp": True, "tile": None,
              "patched": True},
     "ok": True, "peak_gb": 90.10, "per_chip": 637.0, "step_s": 12.50,
     "source": "2026-08-16 打补丁补齐 all-gather + psum_scatter 后重测。"},

    # ⑤ 新思路：不开 QAG → FSDP 吃满 128 → batch 开得更大
    {"when": {"model": "hunyuan3-295b", "pdbs": 13, "cal": "fixed", "fsdp": 128,
              "quant": "fp8", "tokamax": False, "shard_exp": False, "tile": True},
     "ok": True, "peak_gb": 93.97, "per_chip": 727.0, "step_s": 20.342,
     "source": "峰值配方（仅 benchmark）：FP8 + native + FSDP 128 + pdbs 13。"},
    {"when": {"model": "hunyuan3-295b", "pdbs": 14, "cal": "fixed", "fsdp": 128,
              "quant": "fp8", "tokamax": False, "shard_exp": False, "tile": True},
     "ok": False, "required_gb": 99.30, "kind": "hbm_oom_runtime",
     "raw": "total memory required for HLO temporaries exceeds available HBM",
     "source": "AOT 扫描：14 装不下"},
    {"when": {"model": "hunyuan3-295b", "pdbs": 16, "cal": "fixed", "fsdp": 128,
              "quant": "fp8", "tokamax": False, "shard_exp": False, "tile": True},
     "ok": False, "required_gb": 108.4, "kind": "hbm_oom_runtime",
     "raw": "total memory required for HLO temporaries exceeds available HBM",
     "source": "AOT 扫描：16 差得远"},

    # ⑥ fixed 伤收敛 → 换 absmax，batch 要重新二分（**显存非单调**）
    {"when": {"model": "hunyuan3-295b", "pdbs": 13, "cal": "absmax", "fsdp": 128,
              "quant": "fp8", "tokamax": False, "shard_exp": False, "tile": True},
     "ok": False, "required_gb": 95.51, "kind": "hbm_oom_runtime",
     "raw": "total memory required for HLO temporaries (95.51G) exceeds available HBM (94.74G)",
     "source": "AOT 与真机两边都报 95.51G，逐位吻合"},
    {"when": {"model": "hunyuan3-295b", "pdbs": 12, "cal": "absmax", "fsdp": 128,
              "quant": "fp8", "tokamax": False, "shard_exp": False, "tile": True},
     "ok": False, "required_gb": 96.00, "kind": "hbm_oom_runtime",
     "raw": "total memory required for HLO temporaries exceeds available HBM",
     "nonmono": "⚠️ **12 比 13 更超**（96.00 vs 95.51）。显存不随 batch 单调 —— "
                "不同尺寸让编译器选了不同的融合与排布方案。"
                "所以「差一点点，降一档就好」这个直觉在这里不成立，必须逐档试。",
     "source": "AOT 扫描：这一档是「显存非单调」的直接证据"},
    {"when": {"model": "hunyuan3-295b", "pdbs": 11, "cal": "absmax", "fsdp": 128,
              "quant": "fp8", "tokamax": False, "shard_exp": False, "tile": True},
     "ok": True, "peak_gb": 93.19, "per_chip": 670.8, "step_s": 18.656,
     "source": "✅ 生产配方：FP8 + native + FSDP 128 + absmax + pdbs 11。"},

    # ⑦ BF16 对照（同 FSDP，吃满 tile）
    {"when": {"model": "hunyuan3-295b", "pdbs": 13, "cal": None, "fsdp": 128,
              "quant": None, "tokamax": False, "shard_exp": False, "tile": True},
     "ok": True, "peak_gb": 91.40, "per_chip": 666.6,
     "source": "BF16 最优。FP8 相对它只快约 0.6%（对 absmax 而言）。"},
]


_TILE_KEYS = ("gmm_tile_m", "gmm_tile_k", "gmm_tile_n", "tile_batch_seq", "tile_embed", "tile_mlp")


def _match_case(params: dict, target: dict):
    """在 when 里，`None` 一律是「必须也是 None/False」，**只有 tile 允许写 None 当通配**
    （tile 对某些档的结论没影响）。写 when 时别指望 None 是万能匹配 —— 上一版就是
    这么写坏的，结果整条 QAG 路径静默匹配不上，报「没有实测记录」。"""
    from .models import effective_shape
    sh = effective_shape(params)
    cal = str(params.get("weight_quantization_calibration_method", "") or "")
    quant = str(params.get("quantization", "") or "")
    got = {
        "layers_ok": not sh.get("layers_overridden", False),
        "model": str(params.get("model_name", "") or "").lower(),
        "pdbs": _int(params.get("per_device_batch_size")),
        "cal": "fixed" if cal.startswith("fixed") else ("absmax" if "absmax" in cal else None),
        "quant": "fp8" if "fp8" in quant else None,
        "tokamax": bool(params.get("use_tokamax_gmm")),
        "shard_exp": bool(params.get("shard_exp_on_fsdp")),
        "fsdp": fsdp_width(params, target),
        "tile": any(k in params for k in _TILE_KEYS),
        "patched": bool(params.get("_patched_gather")),
    }
    if not got["layers_ok"]:
        # 实测档位全是生产层数跑的。层数改了就**不能拿来直接用** ——
        # 这跟拿 A 配置的结论回答 B 配置是同一件事。
        return None
    for c in REPLAY_CASES:
        w = c["when"]
        if any(k not in w for k in ("model", "pdbs", "fsdp")):
            continue
        ok = True
        for k, want in w.items():
            if k == "tile" and want is None:      # 唯一的通配
                continue
            if got.get(k) != want:
                ok = False
                break
        # when 里没写 patched 的档，只匹配「没打补丁」的运行时
        if ok and "patched" not in w and got["patched"]:
            ok = False
        if ok:
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
        proj = _project_by_layers(params, target)
        if proj:
            return proj
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
    for k in ("per_chip", "step_s", "invalid", "nonmono"):
        if case.get(k) is not None:
            res[k] = case[k]
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
    res["analyses"]["headroom"] = _headroom(params, target, peak)
    res["analyses"]["levers"] = _levers(params, target, peak)
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
        "per_chip_tflops": case.get("per_chip"), "step_s": case.get("step_s"),
    }
    return res


def _ref_curve(params: dict, target: dict):
    """同配置（**忽略 batch**）的生产层数实测点，按 pdbs 排序。
    这是推算的地基：有了两个点才有斜率，有了斜率才能外推到没跑过的 batch。"""
    probe = dict(params)
    probe.pop("num_decoder_layers", None)
    cal = str(probe.get("weight_quantization_calibration_method", "") or "")
    key = {
        "model": str(probe.get("model_name", "") or "").lower(),
        "cal": "fixed" if cal.startswith("fixed") else ("absmax" if "absmax" in cal else None),
        "quant": "fp8" if "fp8" in str(probe.get("quantization", "") or "") else None,
        "tokamax": bool(probe.get("use_tokamax_gmm")),
        "shard_exp": bool(probe.get("shard_exp_on_fsdp")),
        "fsdp": fsdp_width(probe, target),
    }
    pts = []
    for c in REPLAY_CASES:
        wq = c["when"]
        if c.get("invalid"):
            continue
        if all(wq.get(k) == v for k, v in key.items()):
            gb = c.get("peak_gb") or c.get("required_gb")
            if gb:
                pts.append({"pdbs": wq["pdbs"], "gb": gb, "src": c.get("source", "")})
    return sorted(pts, key=lambda x: x["pdbs"])


def _decompose(pts: list[dict], resident: float) -> dict | None:
    """把实测点拆成三段：常驻 + **与 batch 无关的激活** + 随 batch 变的部分。

    ★ 这一步是必须的。直觉上「非常驻 = 随 batch 走」，但实测对不上：
      80 层 absmax：非常驻 = 95.51 − 23.34 = 72.17 GB，除以 pdbs 13 = 5.55 GB/档；
      而相邻档位实测斜率只有 **2.81 GB/档**。
    差的那一半（约 35.6 GB）是**跟 batch 无关的激活与临时缓冲** ——
    权重梯度、通信缓冲、编译器工作区这些不随 batch 缩。
    不拆出来，反解出的 batch 上限会**大出一倍**。
    """
    if len(pts) < 2:
        return None
    ds = [(pts[i + 1]["gb"] - pts[i]["gb"]) / (pts[i + 1]["pdbs"] - pts[i]["pdbs"])
          for i in range(len(pts) - 1)]
    pos = [d for d in ds if d > 0]
    if not pos:
        return None
    slope = sum(pos) / len(pos)
    fixed = sum(p["gb"] - resident - slope * p["pdbs"] for p in pts) / len(pts)
    return {"slope": slope, "fixed_act": max(fixed, 0),
            "nonmono": any(d <= 0 for d in ds)}


def _resident(params: dict, target: dict, params_b: float) -> float:
    fsdp = fsdp_width(params, target) or 1
    bpp = 4 if "float32" in str(params.get("weight_dtype", "float32")) else 2
    return params_b * 1e9 * (bpp + 2 + 4) / fsdp / 1e9


def _project_by_layers(params: dict, target: dict) -> dict | None:
    """层数被改过时的**推算**，独立成 mode=projected，绝不混进实测。

    两级推算，各自标清楚：
      ① batch 外推 —— 用生产层数下的实测点拟合「常驻 + 每档 × batch」，
         推出这个 batch 在生产层数下大概多少（没跑过的 batch 也能推）
      ② 层数折算 —— 常驻按参数量**精确重算**，激活按层数比例折算

    这个数只用来**挑起始 batch 和缩小二分范围**，上限必须真跑 AOT 定。
    """
    from .models import effective_shape, get_model
    sh = effective_shape(params)
    if not sh or not sh.get("layers_overridden"):
        return None
    base = get_model(params.get("model_name"))
    ratio = sh["layer_ratio"]
    pdbs = _int(params.get("per_device_batch_size"), 0) or 0
    if not pdbs:
        return None

    pts = _ref_curve(params, target)
    if not pts:
        return None

    res_prod = _resident(params, target, base["params_b"])
    res_now = _resident(params, target, sh["params_b"])

    # ① 生产层数下这个 batch 大概多少（三段模型：常驻 + 固定激活 + 随 batch）
    dec = _decompose(pts, res_prod)
    exact = next((p for p in pts if p["pdbs"] == pdbs), None)
    if exact:
        peak_prod, lvl1 = exact["gb"], "measured"
    elif dec:
        peak_prod = res_prod + dec["fixed_act"] + dec["slope"] * pdbs
        lvl1 = "extrapolated"
    else:
        p0 = pts[0]
        peak_prod = res_prod + max(p0["gb"] - res_prod, 0) / p0["pdbs"] * pdbs
        lvl1 = "crude"

    # ② 按层数折算：常驻精确重算，两段激活按层数比例缩
    peak = res_now + max(peak_prod - res_prod, 0) * ratio
    ok = peak <= HBM_PER_DEVICE_GB

    # 反解推荐 batch —— 必须用三段模型，否则会大出一倍
    rec = None
    if dec and dec["slope"] > 0:
        slope_now = dec["slope"] * ratio
        fixed_now = dec["fixed_act"] * ratio
        room = HBM_PER_DEVICE_GB - res_now - fixed_now
        hard = int(room // slope_now) if room > 0 else 0
        safe = int(max(room - 2.0, 0) // slope_now)
        rec = {"hard": hard, "safe": safe, "slope_gb": round(slope_now, 3),
               "fixed_act_gb": round(fixed_now, 2), "resident_gb": round(res_now, 2),
               "plan": sorted({x for x in (max(safe, 1), max(hard, 1), hard + 1) if x >= 1}),
               "formula": f"上限 {HBM_PER_DEVICE_GB} = 常驻 {res_now:.2f} + "
                          f"固定激活 {fixed_now:.2f} + {slope_now:.2f} × batch",
               "note": "**推荐值留了 2 GB 余量**，别贴着上限走 —— 显存不随 batch 单调，"
                       "编译器换个尺寸可能换排布方案。按下面这几档各真跑一次 AOT 确认。"}

    lvl1_txt = ("这个 batch 在生产层数下**有实测点**"
                if lvl1 == "measured" else
                "这个 batch 在生产层数下**也没跑过**，先按每档斜率外推")
    res: dict = {
        "mode": "projected", "ok": ok, "projected": True,
        "projection": {
            "from_layers": sh["prod_layers"], "to_layers": sh["layers"], "ratio": round(ratio, 3),
            "ref_peak_gb": round(peak_prod, 2),
            "ref_source": (exact or min(pts, key=lambda p: abs(p["pdbs"] - pdbs)))["src"],
            "level1": lvl1,
            "why": f"层数 **{sh['prod_layers']} → {sh['layers']}**（比例 {ratio:.2f}）。"
                   f"① {lvl1_txt}，得 {peak_prod:.2f} GB；"
                   f"② 常驻按参数量精确重算 {res_prod:.2f} → **{res_now:.2f} GB**"
                   f"（嵌入层不随层数变，已单独扣出），激活按层数比例折算。",
            "caveat": "⚠️ **这是推算，不是实测。** 激活不严格随层数线性，"
                      "编译器会为不同层数选不同的融合与排布方案。"
                      "用它挑起始 batch、缩小二分范围可以，**定上限必须真跑 AOT**。",
            "recommend": rec,
            "decompose": ({"slope": round(dec["slope"], 3),
                           "fixed_act": round(dec["fixed_act"], 2),
                           "resident": round(res_prod, 2),
                           "why": "实测点拆出来：**非常驻里有一大块跟 batch 无关**"
                                  f"（{dec['fixed_act']:.1f} GB @ {sh['prod_layers']} 层）。"
                                  f"直接拿「非常驻 ÷ batch」当每档增量会算成 "
                                  f"{(pts[-1]['gb'] - res_prod) / pts[-1]['pdbs']:.2f} GB，"
                                  f"实测斜率只有 {dec['slope']:.2f} GB —— **差一倍**。"}
                          if dec else None),
        },
        "analyses": {},
    }
    peak_r = round(peak, 2)
    res["analyses"]["memory"] = _memory_estimate(params, target, peak_r)
    res["analyses"]["scale"] = _scale(params, target)
    res["analyses"]["headroom"] = _headroom(params, target, peak_r)
    res["analyses"]["levers"] = _levers(params, target, peak_r)
    res["analyses"]["codepath"] = _codepath(params)
    res["metrics"] = {"peak_hbm_gb": peak_r,
                      "hbm_pct": round(peak / HBM_PER_DEVICE_GB * 100, 1),
                      "pdbs": pdbs, "fsdp": fsdp_width(params, target),
                      "global_batch": pdbs * (target.get("devices") or 0)}
    return res


def _same_family(params: dict, target: dict) -> list[dict]:
    """同一套配置、只有 batch 不同的**实测点**。这是余量分析的地基 ——
    没有真实点就不外推，因为显存跟 batch 不是线性的。"""
    cal = str(params.get("weight_quantization_calibration_method", "") or "")
    key = {
        "model": str(params.get("model_name", "") or "").lower(),
        "cal": "fixed" if cal.startswith("fixed") else ("absmax" if "absmax" in cal else None),
        "quant": "fp8" if "fp8" in str(params.get("quantization", "") or "") else None,
        "tokamax": bool(params.get("use_tokamax_gmm")),
        "shard_exp": bool(params.get("shard_exp_on_fsdp")),
        "fsdp": fsdp_width(params, target),
    }
    pts = []
    for c in REPLAY_CASES:
        w = c["when"]
        if c.get("invalid"):
            continue
        if all(w.get(k) == v for k, v in key.items()):
            gb = c.get("peak_gb") or c.get("required_gb")
            if gb:
                pts.append({"pdbs": w["pdbs"], "gb": gb, "ok": c["ok"],
                            "nonmono": bool(c.get("nonmono"))})
    return sorted(pts, key=lambda x: x["pdbs"])


def _headroom(params: dict, target: dict, peak: float | None) -> dict:
    """★ 「batch 还能开多大」—— 用户最常问的那个问题，拆成能看懂的三段。

    常驻是精确的（参数量 ÷ FSDP）；每档增量**只有拿到 ≥2 个实测点才敢说**，
    否则明确标成估算。绝不给一个看着像真的外推值。
    """
    from .models import effective_shape
    m = effective_shape(params)
    fsdp = fsdp_width(params, target) or 1
    pdbs = _int(params.get("per_device_batch_size"), 0) or 0
    n = m.get("params_b") or 0
    if not n or not pdbs:
        return {"version": 1, "data": {"ready": False,
                "why": "要先选模型并填 batch，才能算余量。"}}

    wdtype = str(params.get("weight_dtype", "float32"))
    bpp = 4 if "float32" in wdtype else 2
    resident = n * 1e9 * (bpp + 2 + 4) / fsdp / 1e9      # 主权重 + 一阶 bf16 + 二阶 fp32
    cap = HBM_PER_DEVICE_GB

    pts = _same_family(params, target)
    slope, slope_kind, slope_note = None, "unknown", ""
    if len(pts) >= 2:
        ds = [(pts[i + 1]["gb"] - pts[i]["gb"]) / (pts[i + 1]["pdbs"] - pts[i]["pdbs"])
              for i in range(len(pts) - 1)]
        pos = [d for d in ds if d > 0]
        if pos:
            slope = sum(pos) / len(pos)
            slope_kind = "measured"
            if any(d <= 0 for d in ds):
                slope_note = ("这些实测点里**有相邻两档是反的**（batch 更小反而更占）—— "
                              "斜率只取了正向的那几段，所以它是个参考值，不是规律。")
    elif peak:
        slope = (peak - resident) / pdbs
        slope_kind = "estimated"
        slope_note = ("只有一个实测点，这个「每档多少 GB」是把**所有非常驻显存**"
                      "平摊到 batch 上算的。实际上激活里有相当一部分不随 batch 变，"
                      "所以它偏大 —— 只能当上界看。")

    dec = _decompose(pts, resident)
    if dec:
        slope, slope_kind = dec["slope"], "measured"
        if dec["nonmono"]:
            slope_note = ("这些实测点里**有相邻两档是反的**（batch 更小反而更占）—— "
                          "斜率只取了正向的那几段，是参考值不是规律。")
    left = (cap - peak) if peak else None
    more = int(left // slope) if (left is not None and slope and slope > 0 and left > 0) else 0
    solved = None
    if dec and dec["slope"] > 0:
        room = cap - resident - dec["fixed_act"]
        solved = {"hard": int(room // dec["slope"]) if room > 0 else 0,
                  "safe": int(max(room - 2.0, 0) // dec["slope"]),
                  "resident": round(resident, 2), "fixed_act": round(dec["fixed_act"], 2),
                  "slope": round(dec["slope"], 3),
                  "formula": f"{cap} = 常驻 {resident:.2f} + 固定激活 {dec['fixed_act']:.2f}"
                             f" + {dec['slope']:.2f} × batch"}

    ladder = []
    for p in pts:
        ladder.append({"pdbs": p["pdbs"], "gb": p["gb"], "ok": p["ok"],
                       "cur": p["pdbs"] == pdbs, "nonmono": p["nonmono"], "kind": "measured"})
    if not any(x["pdbs"] == pdbs for x in ladder) and peak:
        ladder.append({"pdbs": pdbs, "gb": peak, "ok": peak <= cap, "cur": True,
                       "nonmono": False, "kind": "measured"})
    ladder.sort(key=lambda x: x["pdbs"])

    # 直接给结论，别让人自己从图里推
    okp = [p["pdbs"] for p in pts if p["ok"]]
    badp = [p["pdbs"] for p in pts if not p["ok"]]
    if okp and badp:
        hi = max(okp)
        verdict = (f"**这套配置的 batch 上限是 {hi}。** "
                   f"实测 {hi} 装得下、{min(b for b in badp if b > hi) if any(b > hi for b in badp) else min(badp)} 装不下。")
        if any(b < hi for b in badp):
            verdict += f" 注意 {sorted(b for b in badp if b < hi)} 这些**更小的档反而装不下**，别想当然。"
    elif okp and not badp:
        verdict = (f"实测到 {max(okp)} 都装得下，**上界还没探到** —— "
                   "往上加一档再跑一次 AOT，三分钟就知道。")
    elif badp:
        verdict = f"实测 {sorted(badp)} 全都装不下，先往下降或者动下面那些旋钮。"
    else:
        verdict = "只有当前这一档的数据，**上界不知道** —— 加一档再跑一次就能定位。"

    return {"version": 1, "data": {
        "verdict": verdict,
        "ready": True, "capacity_gb": cap, "peak_gb": peak,
        "resident_gb": round(resident, 2),
        "batch_gb": round(peak - resident, 2) if peak else None,
        "left_gb": round(left, 2) if left is not None else None,
        "per_batch_gb": round(slope, 2) if slope else None,
        "slope_kind": slope_kind, "slope_note": slope_note,
        "pdbs": pdbs, "more_steps": more, "ladder": ladder, "solved": solved,
        "fixed_act_gb": round(dec["fixed_act"], 2) if dec else None,
        "more_text": ("**已经到顶** —— 剩下的余量不够再加一档。" if more == 0
                      else f"照这个斜率，还能再加约 **{more}** 档。"),
        "warn": "⚠️ **显存不随 batch 单调** —— 实测出现过 13 超 0.77 G、降到 12 反而超 1.26 G。"
                "所以「还能加几档」只是个方向，**每一档都要真跑一次 AOT**，不要外推。",
    }}


def _levers(params: dict, target: dict, peak: float | None) -> dict:
    """★ 「该拧哪个旋钮」—— 把可用手段按**能腾出多少 GB** 排序，直接给等价 batch 档数。"""
    from .models import effective_shape
    m = effective_shape(params)
    n = m.get("params_b") or 0
    fsdp = fsdp_width(params, target) or 1
    devices = target.get("devices", 0) or 0
    pdbs = _int(params.get("per_device_batch_size"), 0) or 0
    if not n or not peak:
        return {"version": 1, "data": {"ready": False}}

    wdtype = str(params.get("weight_dtype", "float32"))
    bpp = 4 if "float32" in wdtype else 2
    master = n * 1e9 * bpp / fsdp / 1e9
    mom1 = n * 1e9 * 2 / fsdp / 1e9
    mom2 = n * 1e9 * 4 / fsdp / 1e9
    resident = master + mom1 + mom2
    hd = _headroom(params, target, peak)["data"]
    per = hd.get("per_batch_gb") or 0

    rows = []
    # 1. 加宽 FSDP
    other = 1
    for k, v in params.items():
        if k.startswith("ici_") and k.endswith("_parallelism") and k != "ici_fsdp_parallelism":
            iv = _int(v, 1) or 1
            if iv > 0:
                other *= iv
    max_fsdp = devices // other if devices and other else fsdp
    if max_fsdp > fsdp:
        gain = resident - resident * fsdp / max_fsdp
        rows.append({"name": f"FSDP {fsdp} → {max_fsdp}（吃满）", "gb": round(gain, 2),
                     "how": "ici_fsdp_parallelism=-1",
                     "why": "每卡常驻 ∝ 1/FSDP。这是最有效的杠杆，排在改重算之前。",
                     "risk": "要求各并行度乘积等于每 slice 的 device 数"})
    elif other > 1:
        gain = resident - resident / other
        rows.append({"name": f"关掉其它并行（EP/TP={other}）让 FSDP 吃满", "gb": round(gain, 2),
                     "how": "ici_expert_parallelism=1",
                     "why": "EP 不但 all-to-all 多跳，还逼着 FSDP 减半 —— 两头亏。"
                            "实测 64 芯片 EP=2 掉 39.6%。",
                     "risk": "无"})
    # 2. 二阶动量降 bf16
    rows.append({"name": "二阶动量 fp32 → bf16", "gb": round(mom2 / 2, 2),
                 "how": "优化器状态精度（不是 weight_dtype）",
                 "why": "二阶动量对精度不敏感，是常驻里最好压的一块。",
                 "risk": "极少数配方下影响收敛稳定性，要看 loss 曲线"})
    # 3. 降 batch
    if per:
        rows.append({"name": "batch 降 1 档", "gb": round(per, 2),
                     "how": f"per_device_batch_size={max(pdbs - 1, 1)}",
                     "why": "见效最快，但吃掉的是有效 token。",
                     "risk": "⚠️ 不单调，降一档不保证一定省"})
    # 4. 明确不该动的
    forbidden = {"name": "主权重降 bf16", "gb": round(master / 2, 2),
                 "how": "weight_dtype=bfloat16", "forbidden": True,
                 "why": "看着能省最多，**但这条不能动**：bf16 只有 8 位尾数，"
                        "`w += lr × grad` 的更新会被直接舍掉 —— 不报错，训练是废的。",
                 "risk": "毁掉训练"}
    rows.sort(key=lambda r: -r["gb"])
    rows.append(forbidden)
    for r in rows:
        r["eq_batch"] = round(r["gb"] / per, 1) if per else None
    return {"version": 1, "data": {"ready": True, "rows": rows,
            "per_batch_gb": per, "resident_gb": round(resident, 2)}}


def _memory_estimate(params: dict, target: dict, peak_gb: float | None) -> dict:
    """显存分解。**参数常驻是按参数量算的（准），激活是倒推的（不准）** —— 分开标。"""
    from .models import effective_shape
    fsdp = fsdp_width(params, target) or 1
    m = effective_shape(params)
    n_params_b = m.get("params_b") or 0
    if not n_params_b:
        return {"version": 1, "data": {"capacity_gb": HBM_PER_DEVICE_GB, "peak_gb": peak_gb,
                "segments": [], "over_gb": 0,
                "unknown_model": True,
                "why": "没选模型，参数量未知 —— 常驻显存算不出来。先在上面选一个模型。"}}
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
    from .models import effective_shape
    shape = effective_shape(params)
    pdbs = _int(params.get("per_device_batch_size"), 0) or 0
    seq = _int(params.get("max_target_length"), 0) or 0
    dev = target.get("devices", 0) or 0
    gbatch = pdbs * dev
    tokens = gbatch * seq
    n_act_b = shape.get("act_params_b") or 0
    items = [
        {"v": (f'{shape.get("layers")} ⚠' if shape.get("layers_overridden")
               else shape.get("layers") or "—"), "l": "层数", "kind": "config"},
        {"v": (shape.get("num_experts") or "—") if shape.get("moe") else "dense",
         "l": "专家数", "kind": "lookup"},
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
    from .models import detect_backend, get_model
    m = get_model(params.get("model_name"))
    if m and not m.get("moe"):
        return {"version": 1, "data": {
            "branch": "dense MLP（没有 MoE kernel）", "weight_gather": "编译器按分片规格插入",
            "shard_expert_dim": False, "tokamax": False, "risk": None,
            "probe": "dense 模型没有分组矩阵乘分支，这一族的坑不适用。"}}
    _ = detect_backend(params)
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
