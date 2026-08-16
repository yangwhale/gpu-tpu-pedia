"""HLO 深挖 —— AOT 跑完之后能榨出来的东西，比「装不装得下」多得多。

全部数字来自**真实 dump 文件**，一个都不编：

  memory-usage-report.txt  → 精确的显存排行（哪个张量最占地方）
  after_optimizations.txt  → 算子频次、fusion 数、集合通信次数
  before/after 对比        → 优化把多少东西融掉了
  张量 dtype               → FP8 到底有没有真的生效

最有价值的一条是**显存排行**：Qwen3-235B 那次，第一名是
`bf16[28,4096,151936]` 占 32.46 GiB —— 一个 logits 张量吃掉 19% 的显存。
这种东西不扒 dump 是看不见的，而它恰恰是最容易优化的。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

log = logging.getLogger("tpuguru.hlo")

_OP = re.compile(r"^\s+%?[\w.$-]+ = \S+ ([a-z][a-z0-9-]*)\(", re.M)
_FUSION_KIND = re.compile(r"kind=k([A-Za-z]+)")
_MEM_TOTAL = re.compile(r"Total bytes:\s*(\d+)\s*\(([\d.]+)GiB\)")
_MEM_ROW = re.compile(r"^\s*([\d.]+)GiB\(\s*(\d+)%\);\s*([\d.]+)GiB;\s*(\d+);\s*(\d+);\s*(.+)$", re.M)
_SHAPE = re.compile(r"(?:(\d+)×)?([a-z0-9]+)\[([\d,]*)\]")

# 通信算子 → 它在训练里通常是干什么的
_COLL_ROLE = {
    "all-gather": "收集分片权重 / 激活",
    "reduce-scatter": "梯度归约并切回分片",
    "all-reduce": "跨卡求和（通常是 loss / 统计量）",
    "all-to-all": "专家并行的 token 重排",
    "collective-permute": "流水线或环形传递",
}

_DTYPE_LABEL = {
    "f8e4m3fn": "FP8 E4M3（前向）", "f8e5m2": "FP8 E5M2（反向）",
    "bf16": "BF16", "f32": "FP32", "s32": "INT32", "pred": "bool", "token": "token",
}


def _bytes_of(dtype: str) -> int:
    if dtype.startswith("f8") or dtype in ("s8", "u8", "pred"):
        return 1
    if dtype in ("bf16", "f16", "s16", "u16"):
        return 2
    if dtype in ("f32", "s32", "u32"):
        return 4
    if dtype in ("f64", "s64", "u64"):
        return 8
    return 0


def _find(d: Path, suffix: str) -> Path | None:
    # 主模块是 train_step；jit_stage / threefry 那些是初始化用的小模块，别拿错
    cands = [f for f in d.glob(f"*{suffix}") if "train_step" in f.name]
    if not cands:
        cands = list(d.glob(f"*{suffix}"))
    return max(cands, key=lambda f: f.stat().st_size) if cands else None


def analyze(dump_dir: str | Path) -> dict:
    d = Path(dump_dir)
    if not d.is_dir():
        return {"ok": False, "why": f"产物目录不存在: {d}"}
    out: dict = {"ok": True, "dir": str(d)}
    out["memory"] = _memory(_find(d, "memory-usage-report.txt"))
    out["ops"] = _ops(_find(d, "after_optimizations.txt"))
    out["fusion"] = _fusion(_find(d, "before_optimizations.txt"),
                            _find(d, "after_optimizations.txt"))
    out["collectives"] = _collectives(_find(d, "after_optimizations.txt"))
    out["precision"] = _precision(out.get("memory") or {})
    return out


def _memory(f: Path | None) -> dict:
    """★ 显存排行 —— 这份报告是编译器自己算的，比任何倒推都准。"""
    if not f:
        return {"ok": False, "why": "没有 memory-usage-report（需要 --xla_dump_to）"}
    text = f.read_text(encoding="utf-8", errors="replace")
    m = _MEM_TOTAL.search(text)
    total_gib = float(m.group(2)) if m else None
    rows = []
    for r in _MEM_ROW.finditer(text):
        cum, pct, size, offset, nval, shapes = r.groups()
        big = []
        for cnt, dt, dims in _SHAPE.findall(shapes):
            n = int(cnt or 1)
            dl = [int(x) for x in dims.split(",") if x]
            elems = 1
            for x in dl:
                elems *= x
            # ⚠️ 单张量大小**不能乘份数** —— 同一块 offset 上的多份是复用的。
            #    乘了会算出「单张量 49 GiB 装在 32.46 GiB 的块里」这种自相矛盾的数。
            one = elems * _bytes_of(dt) / 1024 ** 3
            big.append({"n": n, "dtype": dt, "dims": dl,
                        "gib": round(one, 2), "gib_all": round(n * one, 2)})
        big.sort(key=lambda x: -x["gib"])
        rows.append({"size_gib": float(size), "cum_pct": int(pct),
                     "n_values": int(nval), "top_shape": big[0] if big else None,
                     "shapes_n": len(big)})
        if len(rows) >= 12:
            break
    return {"ok": True, "total_gib": total_gib, "rows": rows,
            "note": "编译器自己给的分配明细。**同一块 offset 会被多个张量复用**，"
                    "所以「累计」不是简单相加 —— 看单块大小和它装了什么。"}


def _ops(f: Path | None) -> dict:
    if not f:
        return {"ok": False}
    text = f.read_text(encoding="utf-8", errors="replace")
    c = Counter(_OP.findall(text))
    total = sum(c.values())
    return {"ok": True, "total": total, "lines": text.count("\n"),
            "top": [{"op": k, "n": v, "pct": round(v / total * 100, 1)}
                    for k, v in c.most_common(14)]}


def _fusion(before: Path | None, after: Path | None) -> dict:
    """优化把多少东西融掉了。**融合率高 ≠ 一定快**，但融合率异常低通常有问题。"""
    if not after:
        return {"ok": False}
    at = after.read_text(encoding="utf-8", errors="replace")
    a_ops = Counter(_OP.findall(at))
    kinds = Counter(_FUSION_KIND.findall(at))
    res = {"ok": True, "fusions": a_ops.get("fusion", 0),
           "kinds": [{"kind": k, "n": v} for k, v in kinds.most_common()],
           "after_total": sum(a_ops.values())}
    if before:
        bt = before.read_text(encoding="utf-8", errors="replace")
        b_ops = Counter(_OP.findall(bt))
        b_total = sum(b_ops.values())
        res["before_total"] = b_total
        res["before_fusions"] = b_ops.get("fusion", 0)
        # ⚠️ 别把这两个数算成「缩减率」。XLA 先展开（scan 展开、大算子拆小）再融合，
        #    **优化后条数比优化前多是正常的**。算成负缩减率是概念错。
        res["count_note"] = (
            f"优化前 {b_total} 条 → 优化后 {res['after_total']} 条。"
            "**变多是正常的** —— XLA 先展开（scan 展开、大算子拆细）再融合，"
            "所以这两个数不构成「缩减率」。真正该看的是 fusion 的数量与种类。")
    res["note"] = ("kLoop 是逐元素融合（最常见），kOutput 融到输出，"
                   "kCustom 通常是手写 kernel（Pallas / Mosaic）。"
                   "**kCustom 的数量能反过来验证你以为启用的 kernel 有没有真的生效。**")
    return res


def _collectives(f: Path | None) -> dict:
    """★ 集合通信 —— 看**次数**。次数少说明编译器把各层合并、提出了循环。"""
    if not f:
        return {"ok": False}
    text = f.read_text(encoding="utf-8", errors="replace")
    rows = []
    for op, role in _COLL_ROLE.items():
        n = len(re.findall(rf"= \S+ {re.escape(op)}\(", text))
        if n:
            rows.append({"op": op, "n": n, "role": role})
    rows.sort(key=lambda r: -r["n"])
    return {"ok": True, "rows": rows,
            "note": "**看次数不是看字节数。** 每步几十次 = 编译器把各层的收集合并、"
                    "提升出了循环；每层一次（几千次）= 手写的，一层都提不出去。"
                    "实测这两者暴露耗时差 178 倍。"}


def _precision(mem: dict) -> dict:
    """从张量 dtype 反查精度**实际**生效没有 —— 配置写了 fp8 不等于真走了 fp8。"""
    seen = Counter()
    for r in (mem.get("rows") or []):
        t = r.get("top_shape")
        if t:
            seen[t["dtype"]] += 1
    fp8 = [d for d in seen if d.startswith("f8")]
    return {"ok": True,
            "dtypes": [{"dtype": d, "label": _DTYPE_LABEL.get(d, d), "n": n}
                       for d, n in seen.most_common()],
            "fp8_active": bool(fp8),
            "note": ("大张量里出现 f8e4m3fn / f8e5m2，说明 FP8 **真的走到了**。"
                     if fp8 else
                     "大张量里**没看到 FP8 类型** —— 如果你配置里开了 fp8，"
                     "那它很可能没生效。")}


def digest(a: dict, params: dict, target: dict) -> str:
    """给 bot 的摘要。**只喂事实，不喂结论** —— 结论让它自己下。"""
    L = []
    L.append(f"配置：{params.get('model_name')} / pdbs {params.get('per_device_batch_size')} / "
             f"{target.get('topology')}（{target.get('chips')} 芯片） / "
             f"quant={params.get('quantization')} / "
             f"cal={params.get('weight_quantization_calibration_method')}")
    m = a.get("memory") or {}
    if m.get("ok"):
        L.append(f"\n显存总量 {m['total_gib']} GiB。占地方最大的几块：")
        for r in (m["rows"] or [])[:6]:
            t = r["top_shape"] or {}
            dims = "x".join(str(x) for x in (t.get("dims") or []))
            L.append(f"  {r['size_gib']} GiB（累计 {r['cum_pct']}%），"
                     f"块内最大张量 {t.get('dtype')}[{dims}] 单份 {t.get('gib')} GiB"
                     f"（{t.get('n')} 份复用同一块），该块共 {r['n_values']} 个 value")
    o = a.get("ops") or {}
    if o.get("ok"):
        L.append(f"\n算子共 {o['total']} 条，前几名：" +
                 "、".join(f"{x['op']} {x['n']}({x['pct']}%)" for x in o["top"][:8]))
    fz = a.get("fusion") or {}
    if fz.get("ok"):
        L.append(f"\nfusion {fz['fusions']} 个，种类：" +
                 "、".join(f"{k['kind']} {k['n']}" for k in fz["kinds"]))
        if fz.get("before_total"):
            L.append(f"优化前 {fz['before_total']} 条算子（其中 fusion {fz.get('before_fusions',0)} 个）"
                     f" → 优化后 {fz['after_total']} 条（fusion {fz['fusions']} 个）。"
                     f"注意：XLA 先展开再融合，条数变多是正常的，不要当成缩减率。")
    cl = a.get("collectives") or {}
    if cl.get("ok"):
        L.append("\n集合通信次数：" + "、".join(f"{r['op']} {r['n']} 次" for r in cl["rows"]))
    pr = a.get("precision") or {}
    if pr.get("ok"):
        L.append("\n大张量 dtype：" + "、".join(f"{d['label']}×{d['n']}" for d in pr["dtypes"]))
    return "\n".join(L)
