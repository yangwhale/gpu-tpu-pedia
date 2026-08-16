"""命令解析与 AOT 转换 —— 纯确定性，不碰 LLM。

对应 README §4.2 / §4.4 / §5。规矩只有一条：
**能用正则搞定的绝不交给 LLM**，且生成的 AOT 命令必须能被再解析一遍对上。
"""

from __future__ import annotations

import re
import shlex

# ── 目标拓扑：名字里的数字是 device，v7 是 2 device/chip ──────────
TOPOLOGIES = {
    "tpu7x-8":   {"devices": 8,   "chips": 4,   "label": "4 芯片 · 冒烟"},
    "tpu7x-32":  {"devices": 32,  "chips": 16,  "label": "16 芯片"},
    "tpu7x-128": {"devices": 128, "chips": 64,  "label": "64 芯片 · 标准 benchmark"},
    "tpu7x-256": {"devices": 256, "chips": 128, "label": "128 芯片"},
    "tpu7x-512": {"devices": 512, "chips": 256, "label": "256 芯片"},
}

# 训练时才有意义、AOT 必须覆盖掉的运行时噪声
RUNTIME_NOISE = {
    "steps", "base_output_directory", "dataset_path", "dataset_type",
    "run_name", "enable_checkpointing", "checkpoint_period", "async_checkpointing",
    "save_period", "log_period", "profiler", "profiler_steps", "skip_first_n_steps_for_profiler",
    "upload_all_profiler_results", "metrics_file", "gcs_metrics", "eval_interval",
}
AOT_OVERRIDES = {
    "steps": "1",
    "enable_checkpointing": "False",
    "dataset_type": "synthetic",
}

# 模型形状的**唯一来源**是 models.py。这里只做一层投影，
# 不要在两处各维护一份 —— 形状对不上时 lint 会静默失准。
from .models import MODELS as _MODELS  # noqa: E402

MODEL_SHAPES = {
    k: {"num_experts": v["num_experts"], "layers": v["layers"],
        "hidden": v["hidden"], "mlp": v["mlp"], "moe": v["moe"],
        "params_b": v["params_b"], "act_params_b": v["act_params_b"]}
    for k, v in _MODELS.items()
}

# ⚠️ 不要把 "1"/"0" 当布尔 —— `ici_expert_parallelism=1` 是**并行度 1**，
# 一旦变成 True，FSDP 宽度就算错，后面整条 lint 链跟着错。
_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}


def _coerce(v: str):
    s = v.strip().strip('"').strip("'")
    if s.lower() in _TRUE:
        return True
    if s.lower() in _FALSE:
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_command(text: str) -> dict:
    """把一整段 shell（含环境变量、续行）拆成结构化配置。

    返回 {params, xla_flags, target, entrypoint, config_yml, unknown, raw}
    """
    raw = text.strip()
    # 抹平续行
    flat = re.sub(r"\\\s*\n\s*", " ", raw)
    flat = re.sub(r"\s+", " ", flat).strip()

    params: dict = {}
    xla_flags: dict = {}
    unknown: list[str] = []
    entrypoint = ""
    config_yml = ""

    # ① 环境变量里的 XLA flags（LIBTPU_INIT_ARGS / XLA_FLAGS）
    for m in re.finditer(r'(LIBTPU_INIT_ARGS|XLA_FLAGS)\s*=\s*("([^"]*)"|\'([^\']*)\'|(\S+))', flat):
        blob = m.group(3) or m.group(4) or m.group(5) or ""
        for fm in re.finditer(r"(--[\w.]+)(?:=(\S+))?", blob):
            xla_flags[fm.group(1)] = _coerce(fm.group(2)) if fm.group(2) else True
    flat_wo_env = re.sub(r'(LIBTPU_INIT_ARGS|XLA_FLAGS)\s*=\s*("[^"]*"|\'[^\']*\'|\S+)', " ", flat)

    # ② 入口与 config yml
    em = re.search(r"(?:python3?\s+)?-m\s+([\w.]+)", flat_wo_env)
    if em:
        entrypoint = em.group(1)
    else:
        em = re.search(r"python3?\s+(\S+\.py)", flat_wo_env)
        if em:
            entrypoint = em.group(1)
    ym = re.search(r"(\S+\.ya?ml)", flat_wo_env)
    if ym:
        config_yml = ym.group(1)

    # ③ k=v 参数
    try:
        toks = shlex.split(flat_wo_env)
    except ValueError:
        toks = flat_wo_env.split()
    for t in toks:
        if t.startswith("--xla") or t.startswith("--libtpu"):
            k, _, v = t.partition("=")
            xla_flags[k] = _coerce(v) if v else True
            continue
        if "=" in t and not t.startswith("-") and not t.endswith(".yml") and not t.endswith(".yaml"):
            k, _, v = t.partition("=")
            if re.fullmatch(r"[A-Za-z_][\w.]*", k):
                if k in ("LIBTPU_INIT_ARGS", "XLA_FLAGS"):
                    continue
                params[k] = _coerce(v)
                continue
        if t in ("python", "python3", "-m", "&&", "\\") or t == entrypoint or t == config_yml:
            continue
        if t.startswith("-"):
            unknown.append(t)

    target = infer_target(params)
    return {
        "params": params, "xla_flags": xla_flags, "target": target,
        "entrypoint": entrypoint, "config_yml": config_yml,
        "unknown": unknown, "raw": raw,
    }


def infer_target(params: dict) -> dict:
    """从参数推断目标拓扑。推不出就留空并标 needs_input。"""
    topo = params.get("compile_topology")
    if isinstance(topo, str) and topo in TOPOLOGIES:
        t = dict(TOPOLOGIES[topo]); t["topology"] = topo; t["needs_input"] = False
        t["slices"] = int(params.get("compile_topology_num_slices", 1) or 1)
        return t
    # ici_* 乘积 = 每 slice 的 device 数
    prod, seen = 1, False
    for k, v in params.items():
        if k.startswith("ici_") and k.endswith("_parallelism"):
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                prod *= iv; seen = True
    if seen and prod > 1:
        for name, spec in TOPOLOGIES.items():
            if spec["devices"] == prod:
                t = dict(spec); t["topology"] = name
                t["needs_input"] = False; t["slices"] = 1
                t["inferred_from"] = "ici_*_parallelism 乘积"
                return t
    return {"topology": "", "devices": 0, "chips": 0, "slices": 1,
            "label": "未指定", "needs_input": True}


def fsdp_width(params: dict, target: dict) -> int:
    """FSDP 的实际宽度。-1 表示吃满剩余 device。"""
    try:
        v = int(params.get("ici_fsdp_parallelism", 1))
    except (TypeError, ValueError):
        return 1
    if v > 0:
        return v
    devices = target.get("devices") or 0
    other = 1
    for k, val in params.items():
        if k.startswith("ici_") and k.endswith("_parallelism") and k != "ici_fsdp_parallelism":
            try:
                iv = int(val)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                other *= iv
    return devices // other if devices and other else 0


def to_aot(parsed: dict) -> dict:
    """train → train_compile。返回 {cmd, params, dropped, added}"""
    params = dict(parsed["params"])
    dropped, added = {}, {}

    for k in list(params):
        if k in RUNTIME_NOISE:
            dropped[k] = params.pop(k)
    for k, v in AOT_OVERRIDES.items():
        params[k] = _coerce(v); added[k] = v

    t = parsed["target"]
    if t.get("topology"):
        params["compile_topology"] = t["topology"]; added["compile_topology"] = t["topology"]
        params["compile_topology_num_slices"] = t.get("slices", 1)

    if parsed["xla_flags"]:
        blob = " ".join(k if v is True else f"{k}={v}" for k, v in parsed["xla_flags"].items())
        params["compile_xla_flags"] = blob
        added["compile_xla_flags"] = blob

    ep = parsed["entrypoint"] or "MaxText.train"
    ep_aot = re.sub(r"(^|\.)train$", r"\1train_compile", ep) if ep.endswith("train") \
        else ep.replace("train.py", "train_compile.py")
    if ep_aot == ep:
        ep_aot = "MaxText.train_compile"
    yml = parsed["config_yml"] or "MaxText/configs/base.yml"

    order = ["model_name", "compile_topology", "compile_topology_num_slices"]
    keys = [k for k in order if k in params] + sorted(k for k in params if k not in order)
    lines = [f"python3 -m {ep_aot} {yml}"]
    for k in keys:
        v = params[k]
        sv = "True" if v is True else "False" if v is False else str(v)
        lines.append(f'  {k}={sv}' if " " not in sv else f'  {k}="{sv}"')
    return {"cmd": " \\\n".join(lines), "params": params, "dropped": dropped, "added": added}


def roundtrip_check(aot_cmd: str, expect_params: dict) -> list[dict]:
    """把生成的命令再解析一遍，跟期望比对。README §4.2 要求的可校验性。"""
    back = parse_command(aot_cmd)["params"]
    diffs = []
    for k, v in expect_params.items():
        if k not in back:
            diffs.append({"param": k, "expected": v, "actual": None, "kind": "missing"})
        elif str(back[k]) != str(v):
            diffs.append({"param": k, "expected": v, "actual": back[k], "kind": "mismatch"})
    for k in back:
        if k not in expect_params:
            diffs.append({"param": k, "expected": None, "actual": back[k], "kind": "extra"})
    return diffs
