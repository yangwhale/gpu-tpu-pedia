"""lint 规则引擎 —— 对应 README §7 / §8.3。

规则是**数据**不是代码（`rules/rules.seed.json`），加规则不用改这里。

`when` 可以是单个子句，也可以是 `{"all": [...]}`。子句三种形状：

  参数  {"param": "per_device_batch_size", "gt": 16}
  内置  {"builtin": "tile_exceeds_dim"}                  ← 需要遍历/查表的检查
  flag  {"flag": "xla_tpu_...", "eq": "true" | "absent": true}
  表达式 {"expr": "num_experts % fsdp_width != 0"}   ← 受控命名空间，只认白名单变量

**规则求值失败必须出声**（log + 返回一条 info），不能静默跳过 ——
一条不生效的 lint 跟没有这条规则的区别，用户是看不见的。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .parser import MODEL_SHAPES, fsdp_width

log = logging.getLogger("tpuguru.lint")

_RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "rules.seed.json"

# 表达式里允许出现的变量。加规则要用新变量，先在 _ctx 里补上再加进来。
_ALLOWED_NAMES = {
    "num_experts", "fsdp_width", "layers", "hidden", "mlp", "devices", "chips",
    "is_moe", "tile_params_count", "topology_specified", "pdbs",
    "ici_expert_parallelism", "ici_tensor_parallelism", "ici_data_parallelism",
}

_TILE_PREFIXES = ("tile_", "gmm_tile", "sa_block_", "moe_tile")


def load_rules() -> list[dict]:
    try:
        return [r for r in json.loads(_RULES_PATH.read_text(encoding="utf-8")) if r.get("enabled", True)]
    except Exception as e:  # noqa: BLE001
        log.error("规则库读取失败: %s", e)
        return []


def _ctx(params: dict, target: dict) -> dict:
    name = str(params.get("model_name", "")).lower()
    shape = MODEL_SHAPES.get(name, {})
    n_exp = params.get("num_experts", shape.get("num_experts", 0)) or 0
    return {
        "num_experts": n_exp,
        "layers": params.get("num_decoder_layers", shape.get("layers", 0)) or 0,
        "hidden": params.get("base_emb_dim", shape.get("hidden", 0)) or 0,
        "mlp": params.get("base_mlp_dim", shape.get("mlp", 0)) or 0,
        "fsdp_width": fsdp_width(params, target),
        "devices": target.get("devices", 0) or 0,
        "chips": target.get("chips", 0) or 0,
        "pdbs": _num(params.get("per_device_batch_size")) or 0,
        "is_moe": bool(n_exp) or bool(params.get("megablox")) or bool(params.get("sparse_matmul")),
        "tile_params_count": sum(1 for k in params if k.startswith(_TILE_PREFIXES)),
        "topology_specified": bool(target.get("topology")),
        "ici_expert_parallelism": _num(params.get("ici_expert_parallelism")) or 1,
        "ici_tensor_parallelism": _num(params.get("ici_tensor_parallelism")) or 1,
        "ici_data_parallelism": _num(params.get("ici_data_parallelism")) or 1,
    }


def _norm(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "on"):
            return True
        if s in ("false", "no", "off"):
            return False
    return v


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── 内置检查器：表达式表达不了的（需要遍历、查表）放这里 ──────
# tile 参数名 → 它受哪个维度约束
_TILE_DIM = {
    "gmm_tile_m": "batch_seq", "gmm_tile_k": "hidden", "gmm_tile_n": "mlp",
    "moe_tile_m": "batch_seq", "moe_tile_k": "hidden", "moe_tile_n": "mlp",
    "tile_batch_seq": "batch_seq", "tile_embed": "hidden", "tile_mlp": "mlp",
}


def _builtin_tile_exceeds_dim(params: dict, ctx: dict):
    """tile 分块不能超过对应维度本身，否则编译期断言失败。

    返回命中的具体项，让报错能直接指名道姓 —— 「有 tile 超了」帮不上忙，
    「gmm_tile_n=3072 > mlp=1536」才是能直接改的。
    """
    hits = []
    dims = {"hidden": ctx.get("hidden") or 0, "mlp": ctx.get("mlp") or 0,
            "batch_seq": 0}
    for k, dim_name in _TILE_DIM.items():
        if k not in params:
            continue
        tv = _num(params[k])
        dv = dims.get(dim_name) or 0
        if tv and dv and tv > dv:
            hits.append(f"{k}={int(tv)} > {dim_name}={int(dv)}")
    return hits


def _builtin_layers_overridden(params: dict, ctx: dict):
    from .models import effective_shape
    sh = effective_shape(params)
    if not sh.get("layers_overridden"):
        return []
    return [f'当前 {sh["layers"]} 层 / 生产 {sh["prod_layers"]} 层'
            f'（参数量 {sh["params_b"]}B，生产是 {sh["prod_layers"] and MODEL_SHAPES.get(str(params.get("model_name","")).lower(),{}).get("params_b")}B）']


_BUILTINS = {"tile_exceeds_dim": _builtin_tile_exceeds_dim,
             "layers_overridden": _builtin_layers_overridden}


def _flag_lookup(flags: dict, name: str):
    """XLA flag 名在配置里可能带 `--` 也可能不带，两种都认。"""
    for cand in (name, "--" + name.lstrip("-"), name.lstrip("-")):
        if cand in flags:
            return flags[cand], True
    return None, False


def _match_clause(c: dict, params: dict, flags: dict, ctx: dict) -> bool:
    if "builtin" in c:
        fn = _BUILTINS.get(c["builtin"])
        if fn is None:
            raise ValueError(f"没有这个内置检查器: {c['builtin']}")
        hits = fn(params, ctx)
        ctx.setdefault("_builtin_hits", {})[c["builtin"]] = hits
        return bool(hits)

    if "expr" in c:
        expr = c["expr"]
        ident = set()
        cur = ""
        for ch in expr:
            if ch.isalnum() or ch == "_":
                cur += ch
            else:
                if cur and not cur[0].isdigit():
                    ident.add(cur)
                cur = ""
        if cur and not cur[0].isdigit():
            ident.add(cur)
        ident -= {"and", "or", "not", "True", "False", "None"}
        if not ident <= _ALLOWED_NAMES:
            raise ValueError(f"表达式含未授权变量 {sorted(ident - _ALLOWED_NAMES)}")
        return bool(eval(expr, {"__builtins__": {}}, dict(ctx)))  # noqa: S307 受控命名空间

    if "flag" in c:
        val, present = _flag_lookup(flags, c["flag"])
        if "absent" in c:
            return present is not bool(c["absent"])
        if "exists" in c:
            return present == bool(c["exists"])
        if not present:
            return False
        if "eq" in c:
            return _norm(val) == _norm(c["eq"]) or (val is True and _norm(c["eq"]) is True)
        if "neq" in c:
            return _norm(val) != _norm(c["neq"])
        return True

    key = c["param"]
    val = params.get(key, ctx.get(key))
    if "exists" in c:
        return (key in params) == bool(c["exists"])
    if "absent" in c:
        return (key not in params) == bool(c["absent"])
    if "eq" in c:
        return _norm(val) == _norm(c["eq"])
    if "neq" in c:
        return _norm(val) != _norm(c["neq"])
    if "startswith" in c:
        return isinstance(val, str) and val.startswith(c["startswith"])
    if "contains" in c:
        return isinstance(val, str) and c["contains"] in val
    if "gt" in c:
        return _num(val) is not None and _num(val) > c["gt"]
    if "lt" in c:
        return _num(val) is not None and _num(val) < c["lt"]
    raise ValueError(f"看不懂的子句: {c}")


def run_lint(params: dict, target: dict, xla_flags: dict | None = None,
             rules: list[dict] | None = None) -> list[dict]:
    rules = rules if rules is not None else load_rules()
    flags = xla_flags or {}
    ctx = _ctx(params, target)
    out, broken = [], []
    for r in rules:
        w = r.get("when") or {}
        clauses = w.get("all") if "all" in w else [w]
        if not clauses:
            continue
        try:
            hit = all(_match_clause(c, params, flags, ctx) for c in clauses)
        except Exception as e:  # noqa: BLE001
            log.warning("规则 %s 求值失败: %s", r.get("rule"), e)
            broken.append(f'{r.get("rule")}({e})')
            continue
        if hit:
            detail = r.get("detail", "")
            for name, hits in (ctx.get("_builtin_hits") or {}).items():
                if hits and any(c.get("builtin") == name for c in clauses):
                    detail += "  命中：" + "、".join(hits)
            out.append({"rule": r["rule"], "severity": r["severity"], "title": r["title"],
                        "detail": detail, "fix": r.get("fix", ""),
                        "evidence": r.get("evidence", "")})
    if broken:
        # 不静默 —— 规则坏了跟规则不存在，用户是看不出区别的
        out.append({"rule": "ENGINE", "severity": "info",
                    "title": f"{len(broken)} 条规则没能求值，这次体检不完整",
                    "detail": "、".join(broken), "fix": "", "evidence": ""})
    order = {"fatal": 0, "warn": 1, "info": 2}
    out.sort(key=lambda x: order.get(x["severity"], 9))
    return out
