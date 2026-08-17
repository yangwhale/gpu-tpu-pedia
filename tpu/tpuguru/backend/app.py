"""tpuguru 后端 —— FastAPI。对应 README §3 / §4 / §8。

存储：Firestore（`tpuguru` / `tpuguru_sessions` / `tpuguru_saves`）。
拿不到 Firestore 时降级到本地 JSON，**并在 /api/health 里如实标出来**，
不要让人以为存档已经落库了。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .lint import run_lint
from . import flags as flagmod
from . import hlo as hlomod
from . import metal
from . import models as modelsmod
from .models import BACKENDS, MODELS, catalog, detect_backend, get_model
from .parser import classify_unknown, maxtext_fields, unknown_params, upstream_fields, TOPOLOGIES, fsdp_width, parse_command, roundtrip_check, to_aot
from .worker import (docker_available, effective_xla_flags, expand_param_names,
                     expand_params, run_aot, shell_command, verdict_of)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [tpuguru] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tpuguru.app")

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
BOT_URL = os.environ.get("TPUGURU_BOT_URL", "http://127.0.0.1:8810/api/chat")

app = FastAPI(title="tpuguru")


# ── 存储 ────────────────────────────────────────────────────────
class Store:
    """Firestore 优先，失败退本地 JSON。降级状态必须能被查询到。"""

    def __init__(self):
        self.backend = "local"
        self.db = None
        self.local = Path(os.environ.get("TPUGURU_LOCAL_DIR", "/tmp/tpuguru-store"))
        self.local.mkdir(parents=True, exist_ok=True)
        if os.environ.get("TPUGURU_FORCE_LOCAL") == "1":
            return
        try:
            from google.cloud import firestore
            self.db = firestore.Client(project=os.environ.get("FIRESTORE_PROJECT", "chris-pgp-host"),
                                       database=os.environ.get("FIRESTORE_DATABASE", "closecrab"))
            self.db.collection("tpuguru_sessions").limit(1).get()
            self.backend = "firestore"
        except Exception as e:  # noqa: BLE001
            log.warning("Firestore 不可用，降级本地存储: %s", e)
            self.db = None

    def _f(self, col, doc_id):
        return self.local / f"{col}__{doc_id}.json"

    def put(self, col, doc_id, data):
        data = json.loads(json.dumps(data, default=str))
        if self.db:
            self.db.collection(col).document(doc_id).set(data)
        else:
            self._f(col, doc_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def get(self, col, doc_id):
        if self.db:
            d = self.db.collection(col).document(doc_id).get()
            return d.to_dict() if d.exists else None
        f = self._f(col, doc_id)
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None

    def list(self, col, limit=200):
        if self.db:
            docs = self.db.collection(col).limit(limit).get()
            return [d.to_dict() for d in docs]
        out = []
        for f in sorted(self.local.glob(f"{col}__*.json")):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        return out[:limit]


store = Store()


def _load_custom_models():
    """自定义模型从存储灌回注册表。**内置的永远赢** —— 见 models.all_models()。"""
    try:
        for d in store.list("tpuguru_models", limit=300):
            if d.get("id"):
                modelsmod.CUSTOM[d["id"]] = d
        log.info("载入自定义模型 %d 个", len(modelsmod.CUSTOM))
    except Exception as e:  # noqa: BLE001
        log.warning("自定义模型载入失败: %s", e)


_load_custom_models()


def _fingerprint(cur: dict) -> str:
    """配置指纹。改了配置之后旧报告就不再对应当前配置 ——
    没有这个，用户会对着上一次的结论调这一次的参数。"""
    import hashlib
    # ★ 按**生效的** flag 算，不是按用户点了几个。空 flag 会被执行档案的
    #   default_xla_flags 填满 —— 按用户输入算的话，同一次编译会有两个指纹。
    blob = json.dumps({"p": cur.get("params", {}),
                       "x": effective_xla_flags(cur.get("xla_flags", {})),
                       "t": (cur.get("target") or {}).get("topology", "")},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _fingerprint_legacy(cur: dict) -> str:
    """旧算法（按用户输入的 flag 算）。只用来**捞回**改算法之前跑的结果 ——
    不这么做的话，已经跑过的配置会突然变回「没跑过」。"""
    import hashlib
    blob = json.dumps({"p": cur.get("params", {}), "x": cur.get("xla_flags", {}),
                       "t": (cur.get("target") or {}).get("topology", "")},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _sid() -> str:
    return "sess_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]


# ── 会话内存态（Firestore 只做持久化，热路径走内存）────────────
SESSIONS: dict[str, dict] = {}


def _new_session(title="未命名会话") -> dict:
    sid = _sid()
    s = {"schema_version": 1, "id": sid, "created_at": _now(), "updated_at": _now(),
         "created_by": "local", "title": title, "parent_save_id": None,
         "turns": [], "current": {"params": {}, "xla_flags": {}, "target": {}, "raw_cmd": ""},
         # ★ 配置指纹 → 那一份 AOT 结果。配置就是缓存键：
         #   改回旧配置立刻看到旧报告，改到没跑过的配置就空着。
         #   不做这一层的话，只能弹「报告过期」而旧报告还挂在屏幕上 —— 那正是
         #   「我以为在看 A 的结论，其实是 B 的」。
         "results": {}, "run_ids": [], "last_result": None}
    SESSIONS[sid] = s
    return s


def _get_session(sid: str) -> dict:
    if sid in SESSIONS:
        return SESSIONS[sid]
    d = store.get("tpuguru_sessions", sid) if sid else None
    if d:
        d.setdefault("results", {})
        SESSIONS[sid] = d
        return d
    return _new_session()


def _persist(s: dict):
    s["updated_at"] = _now()
    try:
        store.put("tpuguru_sessions", s["id"], {k: v for k, v in s.items() if k != "last_result"})
    except Exception as e:  # noqa: BLE001
        log.warning("会话持久化失败: %s", e)


_FP_CACHE: dict[str, dict | None] = {}


def _lookup_run(fp: str) -> dict | None:
    """在 `tpuguru` 里找同指纹跑过的那一次。跨会话也算 —— 同一套配置
    的 AOT 结论跟谁跑的无关。查不到就返回 None（前端据此显示空态）。"""
    if fp in _FP_CACHE:
        return _FP_CACHE[fp]
    hit = None
    try:
        if store.db:
            docs = store.db.collection("tpuguru").where("fingerprint", "==", fp).limit(5).get()
            cands = [d.to_dict() for d in docs]
        else:
            cands = [d for d in store.list("tpuguru", 400) if d.get("fingerprint") == fp]
        cands = [c for c in cands if c.get("result")]
        if cands:
            c = max(cands, key=lambda x: x.get("created_at", ""))
            hit = {"run_id": c["id"], "fingerprint": fp,
                   "from_cache": {"at": c.get("created_at"), "run_id": c["id"]},
                   **(c.get("result") or {})}
    except Exception as e:  # noqa: BLE001
        log.warning("按指纹回捞失败: %s", e)
    _FP_CACHE[fp] = hit
    return hit


_METAL_CACHE: dict[str, dict | None] = {}


def _lookup_metal(fp: str) -> dict | None:
    """按配置指纹回捞真机结果 —— 跟 AOT 那半用同一个键缝在一起。"""
    if fp in _METAL_CACHE:
        return _METAL_CACHE[fp]
    hit = None
    try:
        if store.db:
            docs = store.db.collection("tpuguru_metal").where("fingerprint", "==", fp).limit(5).get()
            cands = [d.to_dict() for d in docs]
        else:
            cands = [d for d in store.list("tpuguru_metal", 200) if d.get("fingerprint") == fp]
        if cands:
            hit = max(cands, key=lambda x: x.get("created_at", ""))
    except Exception as e:  # noqa: BLE001
        log.warning("真机结果回捞失败: %s", e)
    _METAL_CACHE[fp] = hit
    return hit


_ANALYSIS_CACHE: dict[str, dict] = {}


def _lookup_analysis(s: dict, fp: str) -> dict | None:
    """HLO 分析结果按配置指纹缓存：会话内存 → 进程缓存 → run doc。"""
    hit = (s.get("hlo") or {}).get(fp) or _ANALYSIS_CACHE.get(fp)
    if hit:
        return hit
    try:
        rid = ((s.get("results") or {}).get(fp) or _lookup_run(fp) or {}).get("run_id")
        if rid:
            doc = store.get("tpuguru", rid) or {}
            if doc.get("hlo_analysis"):
                _ANALYSIS_CACHE[fp] = doc["hlo_analysis"]
                return doc["hlo_analysis"]
    except Exception as e:  # noqa: BLE001
        log.warning("分析缓存回捞失败: %s", e)
    return None


def _save_analysis(s: dict, fp: str, a: dict):
    s.setdefault("hlo", {})[fp] = a
    _ANALYSIS_CACHE[fp] = a
    try:
        rid = ((s.get("results") or {}).get(fp) or _lookup_run(fp) or {}).get("run_id")
        if rid:
            doc = store.get("tpuguru", rid)
            if doc:
                doc["hlo_analysis"] = a
                store.put("tpuguru", rid, doc)
    except Exception as e:  # noqa: BLE001
        log.warning("分析结果落库失败: %s", e)
    _persist(s)


def _recompute(s: dict) -> dict:
    """配置一动就重算：AOT 命令 + lint + 回环校验。"""
    cur = s["current"]
    parsed = {"params": cur["params"], "xla_flags": cur["xla_flags"],
              "target": cur["target"], "entrypoint": cur.get("entrypoint", ""),
              "config_yml": cur.get("config_yml", ""), "raw": cur.get("raw_cmd", "")}
    aot = to_aot(parsed, expand=expand_params)
    findings = run_lint(cur["params"], cur["target"], cur["xla_flags"])
    # flag 之间的依赖错，编译器要到很后面才报 —— 早点拦住便宜得多
    # ★ P1：只有 TPU 实现的 kernel，AOT（CPU 编译）必然编不过。
    #    这条的价值是**省一分钟**：不拦的话 docker 起来、镜像加载完才由 tokamax 抛
    #    NotImplementedError，而且以前还会被报成「装不下」。
    _CPU_UNSUPPORTED = {
        "use_tokamax_gmm": "tokamax 的 ragged_dot（MoE 分组矩阵乘）",
        "use_tokamax_splash": "tokamax 的 splash attention",
    }
    _on = [(k, d) for k, d in _CPU_UNSUPPORTED.items()
           if str(aot["params"].get(k, "")).lower() in ("true", "1")]
    if _on:
        findings.insert(0, {
            "rule": "P1", "severity": "fatal",
            "title": "这套配置 AOT 编不了：" + "、".join(f"`{k}=True`" for k, _ in _on),
            "detail": "AOT 在 **CPU** 上编译（目标硬件靠 `compile_topology` 声明，本机没有 TPU），"
                      "而 " + "、".join(d for _, d in _on) + " 只有 TPU 实现。\n"
                      "**这不是显存问题** —— 真机上这套配置可能完全正常。",
            "fix": "要 AOT 验：`use_tokamax_gmm=False` 走 native megablox（`megablox=True` 保留）。"
                   "⚠️ 换完的显存数字跟原配方**有出入**（实测混元3 BF16 pdbs12：96.70 vs 文档 91.94 GiB），"
                   "别把它当原配方的结论。",
            "evidence": "2026-08-16 实测：tokamax.ragged_dot → NotImplementedError: Not supported on cpu"})

    # ★ P0：参数名 MaxText 认不认，**在拼命令时就能判**，不用等 docker 转一分钟
    #    再由 pyconfig 抛 ValueError。字段表是从容器自己的报错里收割的，不是手写。
    # ⚠️ 必须查**展开后**的名字。gmm_tile_m 之类是工具的旋钮，
    #    经 tile_expand 变成 6 个真实 tile 字段 —— 拿原名去查会把它们全误报成
    #    「MaxText 没这个字段」，而那正是最该调的几个旋钮。
    origin = expand_param_names(aot["params"])
    for bad, near in unknown_params({k: None for k in origin}).items():
        src = origin.get(bad, bad)
        shown = bad if src == bad else f"{src} → {bad}"
        _c = classify_unknown(bad)
        findings.insert(0, {
            # ⚠️ 措辞要准：字段表是从**我们这个镜像**收割的，不是「MaxText 通用真理」。
            #    官方配方常对着更新的 MaxText 写，新字段我们镜像不认 —— 说成
            #    「MaxText 没有这个字段」会让人以为官方配方错了，方向直接找反。
            # ★ 三分：配方写错 / 上游有我们没有 / 上游也没有。
            #   分错方向代价很大 —— 我把配方自身的错归成「我们镜像旧」，
            #   结论就变成「升级镜像就好」，而升级根本没用（2026-08-17 栽过）。
            "rule": "P0", "severity": "fatal",
            "title": ({"recipe_typo": f"配方把字段名写错了：`{shown}`",
                       "version_skew": f"本镜像的 MaxText 还没有 `{shown}`",
                       "not_upstream": f"没有 `{shown}` 这个字段"}[_c["cls"]]),
            "detail": _c["why"] + "\n命令会照样拼出来，但 pyconfig 在解析阶段就 ValueError ——"
                      "docker 起来、镜像加载完，约一分钟后才报，纯浪费。",
            "fix": (f"改成 `{_c['correct']}`" if _c["correct"] else
                    ("升级镜像；或先去掉这一项再验 —— **去掉后的结论不等于原配方的结论**"
                     if _c["cls"] == "version_skew" else "检查拼写，或去掉这一项")),
            "evidence": f"本镜像 {len(maxtext_fields())} 个字段（pyconfig 自吐）"
                        f" · 上游 base.yml {len(upstream_fields())} 个字段（2026-08-17 抓取）"})

    for f in flagmod.lint_flags(cur["xla_flags"]):
        findings.insert(0 if f["severity"] == "fatal" else len(findings), {
            "rule": "F1", "severity": f["severity"],
            "title": f'{f["flag"]} 缺少配套 flag：{f["missing"]}',
            "detail": f["why"], "fix": f'把 {f["missing"]} 一起加上', "evidence": ""})
    diffs = roundtrip_check(aot["cmd"], aot["params"])
    # 拷走就能跑的那一份。拼不出来（本机没执行档案）就给 None，前端据此说明白，
    # **不给一条看着像能跑、实际缺一半的命令**。
    try:
        shell = shell_command(cur["params"], cur.get("target") or {})
    except Exception as e:                                    # noqa: BLE001
        log.warning("拼可执行命令失败: %s", e)
        shell = {}
    return {"aot_cmd": aot["cmd"], "dropped": aot["dropped"], "added": aot["added"],
            "lint": findings, "roundtrip": diffs, "shell": shell or None,
            "xla_line": " ".join(f"{k}={v}" if not str(k).startswith("--") else f"{k}={v}"
                                 for k, v in sorted(cur["xla_flags"].items()))}


def _state(s: dict) -> dict:
    r = _recompute(s)
    fp = _fingerprint(s["current"])
    # ★ 只返回**当前配置**对应的结果。配置一动，报告要么换成那套的，要么空。
    res = (s.get("results") or {}).get(fp)
    cached_from = None
    if res is None:
        res = _lookup_run(fp)
        if res:
            cached_from = res.get("from_cache")
    if res is not None and "verdict" not in res:
        # ★ 老结果（跑在 verdict_of 之前）取出来时补一份。
        #   不补的话前端只能退回泛泛的「编译失败」，具体死因就丢了 ——
        #   而具体死因正是这条结论唯一有用的部分。
        res = {**res, "verdict": verdict_of(res)}
    if res is None:
        # 改指纹算法之前跑的那些，用旧算法再捞一次
        lfp = _fingerprint_legacy(s["current"])
        if lfp != fp:
            res = (s.get("results") or {}).get(lfp) or _lookup_run(lfp)
            if res:
                cached_from = "旧指纹迁移"
                if "verdict" not in res:
                    res = {**res, "verdict": verdict_of(res)}
    return {
        "session_id": s["id"], "title": s["title"], "turns": s["turns"],
        "params": s["current"]["params"], "xla_flags": s["current"]["xla_flags"],
        "target": s["current"]["target"], "fsdp_width": fsdp_width(s["current"]["params"], s["current"]["target"]),
        "model": get_model(s["current"]["params"].get("model_name")),
        "backend": detect_backend(s["current"]["params"]),
        "aot_cmd": r["aot_cmd"], "dropped": r["dropped"], "added": r["added"],
        "lint": r["lint"], "roundtrip": r["roundtrip"],
        "shell": r["shell"], "xla_line": r["xla_line"],
        "run_ids": s["run_ids"], "result": res, "cached_from": cached_from,
        "known_fingerprints": sorted((s.get("results") or {}).keys()),
        "metal": (s.get("metal") or {}).get(fp) or _lookup_metal(fp),
        "hlo": _lookup_analysis(s, fp),
        "parent_save_id": s.get("parent_save_id"),
        "fingerprint": fp,
    }


# ── 对话意图：确定性优先，兜不住才叫 bot ────────────────────────
_CAL_ALIASES = {"absmax": "absmax", "动态": "absmax",
                "fixed": "fixed,-224,224", "静态": "fixed,-224,224"}
# 口语 → 真参数名。**匹配时必须带词边界**，否则 `shard_exp_on_fsdp` 里的 "fsdp"
# 会被当成 `ici_fsdp_parallelism` —— 实测踩过，它会静默改错一个完全不同的参数。
_PARAM_ALIASES = {
    "batch": "per_device_batch_size", "pdbs": "per_device_batch_size",
    "fsdp": "ici_fsdp_parallelism", "校准": "weight_quantization_calibration_method",
    "ep": "ici_expert_parallelism", "专家并行": "ici_expert_parallelism",
    "层数": "num_decoder_layers", "序列长度": "max_target_length", "seq": "max_target_length",
    "拓扑": "__topology",
}
# 允许被直接 `k=v` 指定的参数：已知名单 + 这些前缀
_PARAM_PREFIXES = ("ici_", "dcn_", "per_", "base_", "weight_", "quantization",
                   "use_", "sa_", "gmm_", "moe_", "tile_", "shard_", "num_",
                   "max_", "megablox", "sparse_matmul", "attention", "model_name",
                   "remat_", "opt_", "dtype", "capacity_factor")
# 这些参数是数值，值写成 True/False 一定是抽错了
_NUMERIC = {"per_device_batch_size", "max_target_length", "num_decoder_layers"}
_NUMERIC_PREFIX = ("ici_", "dcn_", "gmm_tile", "sa_block", "tile_")


def _looks_like_command(t: str) -> bool:
    return ("python" in t and "=" in t) or t.count("=") >= 4 or "LIBTPU_INIT_ARGS" in t


def _is_param(k: str) -> bool:
    return len(k) > 3 and k.startswith(_PARAM_PREFIXES)


def _numeric_param(k: str) -> bool:
    return k in _NUMERIC or k.startswith(_NUMERIC_PREFIX)


def _intent_diff(text: str, params: dict) -> list[dict] | None:
    """从一句话里抽出确定的参数改动。抽不出返回 None。

    两遍走：先吃显式 `k=v` 并把命中的区间从文本里**挖掉**，再拿口语别名去扫剩下的。
    不挖掉的话 `shard_exp_on_fsdp=True` 会被别名规则二次命中。
    """
    diffs, seen = [], set()
    rest = text

    # ① 显式 k=v（含中文标点分隔）
    # 值里允许逗号（`fixed,-224,224`），但「逗号 + 空格」是参数之间的分隔符
    for m in re.finditer(r"([A-Za-z_][\w.]*)\s*=\s*((?:[^\s，；;。,]|,(?!\s))+)", text):
        k, v = m.group(1), m.group(2).rstrip("。，,；;")
        if not (_is_param(k) or k in params):
            continue
        if _numeric_param(k) and v.lower() in ("true", "false"):
            continue                       # 数值参数不可能是布尔，这是抽错了
        if k not in seen:
            diffs.append({"param": k, "from": params.get(k), "to": v, "reason": "你直接指定的"})
            seen.add(k)
        rest = rest.replace(m.group(0), " ")

    # ② 口语别名，只在剩下的文本里找，且必须有词边界
    for alias, real in _PARAM_ALIASES.items():
        if real in seen:
            continue
        pat = (r"(?<![A-Za-z_])" + re.escape(alias) + r"(?![A-Za-z0-9_])"
               if alias.isascii() else re.escape(alias))
        # 动词要收全，且值只认 ASCII —— 否则「校准换 fixed」会把「换」当成值
        m = re.search(pat + r"[\s:：]*(?:=|改成|改为|改到|改|设成|设为|设|调到|调成|"
                            r"换成|换为|换到|换|选成|选|开到|开|用|是|为)?"
                            r"[\s:：]*([A-Za-z0-9_.,\-]+)", rest)
        if not m:
            continue
        v = m.group(1).strip("。，,")
        if not v or v in ("的", "是", "了", "吧", "呢"):
            continue
        if real == "weight_quantization_calibration_method":
            v = _CAL_ALIASES.get(v, v)
        if _numeric_param(real) and not re.fullmatch(r"-?\d+", v):
            continue                       # 「fsdp 吃满」这类抽不出数字就别猜
        diffs.append({"param": real, "from": params.get(real), "to": v,
                      "reason": f"「{alias}」→ `{real}`"})
        seen.add(real)

    # ③ 只说了「换 absmax」这种，没提参数名
    if "weight_quantization_calibration_method" not in seen:
        m = re.search(r"(?:换|改|用)(?:成|到)?\s*(absmax|fixed|动态|静态)", rest)
        if m:
            diffs.append({"param": "weight_quantization_calibration_method",
                          "from": params.get("weight_quantization_calibration_method"),
                          "to": _CAL_ALIASES[m.group(1)],
                          "reason": "fixed 静态 scale 伤收敛，不开 QAG 就该用 absmax"})

    # ④ 「FSDP 吃满」「吃满 FSDP」
    if "ici_fsdp_parallelism" not in seen and re.search(r"(吃满|拉满|开满)", rest):
        diffs.append({"param": "ici_fsdp_parallelism",
                      "from": params.get("ici_fsdp_parallelism"), "to": "-1",
                      "reason": "-1 = 吃满剩余 device"})
    return diffs or None


async def _botcall(kind: str, text: str, context: dict) -> str:
    """走 tpuguru bot 的 web channel。挂了就降级，不能让页面卡死。"""
    # ⚠️ 别让所有分析共用一个 bot 会话。共用的话它会写出「上次我说的」
    #    「本轮第一条」这种话 —— 对刚打开页面的人毫无意义，而且上下文无限增长
    #    最终会漂。要跨次对比，就把要对比的**数据**显式喂进去（见 prompt 里的「参照」段）。
    payload = {"session_id": f"tpuguru-{kind}-{uuid.uuid4().hex[:8]}", "text": text}
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(BOT_URL, json=payload)
            return (r.json() or {}).get("reply", "")
    except Exception as e:  # noqa: BLE001
        log.warning("BotCall 失败: %s", e)
        return ""


# ── API ─────────────────────────────────────────────────────────
class ChatIn(BaseModel):
    session_id: str | None = None
    text: str


class ApplyIn(BaseModel):
    session_id: str
    diff: list[dict]


class SetIn(BaseModel):
    session_id: str
    param: str
    value: str | int | float | bool | None


class SaveIn(BaseModel):
    session_id: str
    title: str
    note: str = ""
    tags: list[str] = []


@app.get("/api/health")
async def health():
    return {"ok": True, "store": store.backend, "aot_mode": "real" if docker_available() else "replay",
            "bot_url": BOT_URL, "topologies": TOPOLOGIES, **catalog(), "flagdb": flagmod.catalog()}


@app.post("/api/session")
async def new_session():
    s = _new_session()
    _persist(s)
    return _state(s)


def _derive_shape(base: dict, want: dict, touched: set) -> tuple[dict, dict]:
    """合出新形状。**参数量自动跟着层数走** —— 除非用户自己改过那一栏。

    不自动折算的话，40 层的派生模型会挂着 80 层的 297B 参数量，
    后面显存估算、能不能装下、batch 上界全部按错的参数量算，而且一声不吭。
    """
    shape = dict(base)
    changed = {}
    for f in modelsmod.EDITABLE:
        k = f["k"]
        if k not in want or want[k] in (None, ""):
            continue
        v = float(want[k]) if f["type"] == "float" else int(want[k])
        if v != base.get(k):
            changed[k] = {"from": base.get(k), "to": v}
        shape[k] = v
    auto = modelsmod.scale_params(base, shape["layers"],
                                  vocab=shape.get("vocab"), hidden=shape.get("hidden"))
    for k in ("params_b", "act_params_b"):
        if k in touched:          # 用户自己填了就听他的
            continue
        if abs(auto[k] - (shape.get(k) or 0)) > 1e-6:
            changed[k] = {"from": base.get(k), "to": auto[k], "auto": True}
        shape[k] = auto[k]
    shape["emb_params_b"] = auto["emb_params_b"]
    return shape, changed


@app.post("/api/model/preview")
async def preview_model(body: dict):
    """改一栏就重算一次 —— 让面板上看到的就是会存下去的。"""
    base = modelsmod.get_model(str(body.get("base") or "").lower())
    if not base:
        raise HTTPException(400, "没有这个基准模型")
    shape, changed = _derive_shape(base, body.get("shape") or {},
                                   set(body.get("touched") or []))
    return {"shape": {k: shape.get(k) for k in
                      [f["k"] for f in modelsmod.EDITABLE] + ["emb_params_b"]},
            "changed": changed}


@app.post("/api/model")
async def save_model(body: dict):
    """把改过形状的模型**另存为新模型**。

    不允许覆盖内置：`Hunyuan3-295B` 这个名字对应的是真实发布的那个模型，
    我们所有实测结论都挂在它上面。改了它，历史结论就全部失去指向。
    """
    base_id = str(body.get("base") or "").lower()
    base = modelsmod.get_model(base_id)
    if not base:
        raise HTTPException(400, f"没有这个基准模型: {base_id}")
    label = str(body.get("label") or "").strip()
    if not label:
        raise HTTPException(400, "得起个名字")
    # ⚠️ 别把 CJK 一起洗掉 —— 「混元3 40层」这种名字最常见，
    #    只留 ASCII 的话 slug 直接变空，报「名字里没有能当 id 的字符」，莫名其妙。
    #    Firestore 文档 id 收 unicode，只要挡掉 `/` 和空白就行。
    new_id = re.sub(r"[\s/\\#?\[\]*]+", "-", label.lower()).strip("-.")
    if not new_id or new_id in (".", ".."):
        new_id = f"{base_id}-custom"
    if modelsmod.is_builtin(new_id) or label == base.get("label"):
        raise HTTPException(409, f"`{label}` 是内置模型，不能覆盖。换个名字，"
                                 f"比如「{base.get('label')} 40层」。")

    shape, changed = _derive_shape(base, body.get("shape") or {},
                                   set(body.get("touched") or []))
    if not changed:
        raise HTTPException(400, "一个字段都没改 —— 存下来只是个重名副本。")

    shape.update({
        "id": new_id, "label": label, "derived_from": base_id,
        "provenance": "derived", "created_at": _now(),
        # ★ 派生模型**不继承实测结论**。基准那些数字是在原始形状上跑出来的，
        #   改了层数还挂着「64 芯片 v7 实测」会让人拿旧结论套新形状。
        "note": (f"从 {base.get('label')} 派生，改了："
                 + "、".join(f"{k} {c['from']}→{c['to']}" for k, c in changed.items())
                 + "。**形状是你改的，实测数字不继承。**"),
    })
    modelsmod.CUSTOM[new_id] = shape
    store.put("tpuguru_models", new_id, shape)
    return {"id": new_id, "label": label, "changed": changed,
            "catalog": modelsmod.catalog()}


@app.delete("/api/model/{mid}")
async def delete_model(mid: str):
    if modelsmod.is_builtin(mid):
        raise HTTPException(403, "内置模型不能删")
    modelsmod.CUSTOM.pop(mid, None)
    try:
        if store.db:
            store.db.collection("tpuguru_models").document(mid).delete()
        else:
            store._f("tpuguru_models", mid).unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"删除失败: {e}") from e
    return {"deleted": mid, "catalog": modelsmod.catalog()}


@app.get("/api/sessions")
async def list_sessions(limit: int = 60):
    """会话列表。**内存里的和存储里的要合并** —— 只读存储会漏掉刚建还没落盘的，
    只读内存会漏掉重启前的。以内存为准（它更新）。"""
    merged: dict[str, dict] = {}
    for d in store.list("tpuguru_sessions", limit=300):
        if d.get("id"):
            merged[d["id"]] = d
    merged.update(SESSIONS)
    out = []
    for d in merged.values():
        cur = d.get("current") or {}
        params = cur.get("params") or {}
        mdl = get_model(params.get("model_name"))
        # 空会话不进列表：没参数、没 flag、没对话 —— 列出来只会挤掉有用的
        if not params and not (cur.get("xla_flags") or {}) and not d.get("turns"):
            continue
        title = d.get("title") or "未命名会话"
        # 老会话是在自动命名之前建的。列表里现算一个名字出来，
        # 不回写存储 —— 展示逻辑不该偷偷改数据。
        if title == "未命名会话" and mdl:
            title = mdl["label"]
        out.append({
            "id": d["id"], "title": title,
            "model": mdl["label"] if mdl else None,
            "updated_at": d.get("updated_at"), "created_at": d.get("created_at"),
            "n_params": len(params), "n_flags": len(cur.get("xla_flags") or {}),
            "n_turns": len(d.get("turns") or []),
            "topology": (cur.get("target") or {}).get("topology"),
            "layers": params.get("num_decoder_layers"),
            "pdbs": params.get("per_device_batch_size"),
            "n_results": len(d.get("results") or {}),
        })
    out.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return {"sessions": out[:limit], "total": len(out)}


@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    """删会话。**AOT / 真机结果不跟着删** —— 它们按指纹存在 `tpuguru`
    集合里，跟会话是两回事。删掉一场对话不该让几十分钟的编译结果消失。"""
    SESSIONS.pop(sid, None)
    try:
        if store.db:
            store.db.collection("tpuguru_sessions").document(sid).delete()
        else:
            f = store._f("tpuguru_sessions", sid)
            f.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"删除失败: {e}") from e
    return {"deleted": sid}


@app.post("/api/session/{sid}/rename")
async def rename_session(sid: str, body: dict):
    s = _get_session(sid)
    t = str(body.get("title") or "").strip()
    if not t:
        raise HTTPException(400, "标题不能为空")
    s["title"], s["title_manual"] = t, True
    _persist(s)
    return _state(s)


@app.get("/api/session/{sid}")
async def get_session(sid: str):
    return _state(_get_session(sid))


@app.post("/api/chat")
async def chat(inp: ChatIn):
    s = _get_session(inp.session_id or "")
    text = inp.text.strip()
    s["turns"].append({"at": _now(), "role": "user", "text": text})
    params = s["current"]["params"]

    # ① 整段命令 → 解析并填满
    if _looks_like_command(text):
        parsed = parse_command(text)
        s["current"] = {"params": parsed["params"], "xla_flags": parsed["xla_flags"],
                        "target": parsed["target"], "raw_cmd": parsed["raw"],
                        "entrypoint": parsed["entrypoint"], "config_yml": parsed["config_yml"]}
        st = _state(s)
        n = len(parsed["params"]); nf = len(parsed["xla_flags"])
        msg = [f"认出 **{n}** 个 MaxText 参数、**{nf}** 个 XLA flag。"]
        t = parsed["target"]
        if t.get("needs_input"):
            msg.append("⚠️ **拓扑推不出来** —— 右边选一个。AOT 必须知道目标硬件。")
        else:
            src = t.get("inferred_from")
            msg.append(f"目标：`{t['topology']}` = **{t['chips']} 芯片**（{t['devices']} device）"
                       + (f"，从 {src} 推的" if src else ""))
        fatal = [f for f in st["lint"] if f["severity"] == "fatal"]
        if fatal:
            msg.append(f"🔴 **{len(fatal)} 条致命问题**，见右边 lint。最要紧的：{fatal[0]['title']}")
        elif any(f["severity"] == "warn" for f in st["lint"]):
            msg.append("🟡 有几条警告，右边看。")
        else:
            msg.append("✅ 没踩到已知的坑。")
        s["turns"].append({"at": _now(), "role": "guru", "text": "\n\n".join(msg)})
        _persist(s)
        return _state(s)

    # ①b 「跑一次」——按钮和对话两条路都要能触发，不要逼人去找按钮
    if re.fullmatch(r"\s*(跑|跑一次|跑一下|跑 ?aot|跑一次 ?aot|run|编译|编译一下|试试)"
                    r"[ 。!！~]*\s*", text, re.I):
        s["turns"].append({"at": _now(), "role": "system", "text": "开始编译…"})
        _persist(s)
        return await run(ChatIn(session_id=s["id"], text=""))

    # ② 明确的参数改动 → 出 diff 提议（不直接生效）
    diffs = _intent_diff(text, params)
    if diffs:
        s["turns"].append({"at": _now(), "role": "guru",
                           "text": f"建议改 {len(diffs)} 项：",
                           "proposal": {"diff": diffs, "applied": False}})
        _persist(s)
        return _state(s)

    # ③ 兜底：交给带 skill 的 agent
    ctx = {"params": params, "target": s["current"]["target"], "lint": _recompute(s)["lint"]}
    prompt = (f"用户在 tpuguru 工作台里问：{text}\n\n"
              f"当前配置（JSON）：{json.dumps(ctx, ensure_ascii=False)[:3000]}\n\n"
              "简短回答（≤200 字），针对当前配置。不要贴大段表格。")
    reply = await _botcall("ask", prompt, ctx)
    s["turns"].append({"at": _now(), "role": "guru",
                       "text": reply or "（agent 暂时不可用，先用右边的表单改配置）"})
    _persist(s)
    return _state(s)


@app.post("/api/apply")
async def apply(inp: ApplyIn):
    s = _get_session(inp.session_id)
    from .parser import _coerce
    for d in inp.diff:
        s["current"]["params"][d["param"]] = _coerce(str(d["to"]))
    s["current"]["target"] = s["current"]["target"] or {}
    names = "、".join(f'`{d["param"]}`' for d in inp.diff)
    for t in reversed(s["turns"]):
        if t.get("proposal") and not t["proposal"].get("applied"):
            t["proposal"]["applied"] = True
            t["proposal"]["applied_at"] = _now()
            break
    s["turns"].append({"at": _now(), "role": "system", "text": f"已应用：{names}"})
    _persist(s)
    return _state(s)


@app.post("/api/set")
async def set_param(inp: SetIn):
    s = _get_session(inp.session_id)
    from .parser import _coerce
    # 值没变就什么都不做 —— 一条「从 -1 改成 -1」的记录只会污染对话流
    _cur = (s["current"]["target"].get("topology") if inp.param == "__topology"
            else s["current"]["params"].get(inp.param))
    if str(_cur if _cur is not None else "") == str(inp.value if inp.value is not None else ""):
        return _state(s)

    if inp.param == "__flag":
        # value 形如 "flag=值"；值为空表示删掉这个 flag
        k, _, v = str(inp.value).partition("=")
        k = k.strip()
        if not v.strip():
            s["current"]["xla_flags"].pop("--" + k, None)
            s["current"]["xla_flags"].pop(k, None)
            s["turns"].append({"at": _now(), "role": "system", "text": f"你去掉了 `{k}`"})
        else:
            s["current"]["xla_flags"]["--" + k] = v.strip()
            s["turns"].append({"at": _now(), "role": "system", "text": f"你设了 `{k}={v.strip()}`"})
        _persist(s)
        return _state(s)

    if inp.param == "__preset":
        if str(inp.value) == "__none":
            n = len(s["current"]["xla_flags"])
            s["current"]["xla_flags"] = {}
            s["turns"].append({"at": _now(), "role": "system",
                               "text": f"清空了 {n} 个 flag。⚠️ 执行档案的 "
                                       "`default_xla_flags` 照样会传进去 —— "
                                       "「不设」不等于「不生效」。"})
            _persist(s)
            return _state(s)
        ps = next((x for x in flagmod.PRESETS if x["id"] == str(inp.value)), None)
        if not ps:
            raise HTTPException(400, f"没有这个套餐: {inp.value}")
        # ★ 套餐是**替换**不是叠加。叠加的后果：已经有 16 个之后点任何套餐都
        #   毫无反应（要加的早就在里面了），看起来像按钮坏了。
        #   但换掉什么必须说清楚 —— 静默丢掉用户手调的 flag 比不生效更糟。
        cur = s["current"]["xla_flags"]
        keep = {k.lstrip("-") for k in cur}
        gone = sorted(keep - set(ps["flags"]))
        s["current"]["xla_flags"] = {"--" + k: v for k, v in ps["flags"].items()}
        msg = f'套用 **{ps["label"]}** —— {len(ps["flags"])} 个 flag'
        if gone:
            head = "、".join(f"`{x}`" for x in gone[:4])
            msg += (f'。**换掉了原来的 {len(gone)} 个**：{head}'
                    + (f' 等' if len(gone) > 4 else ''))
        s["turns"].append({"at": _now(), "role": "system", "text": msg})
        _persist(s)
        return _state(s)

    if inp.param == "__backend":
        spec = BACKENDS.get(str(inp.value))
        if not spec:
            raise HTTPException(400, f"没有这个后端: {inp.value}")
        s["current"]["params"].update(spec["apply"])
        s["turns"].append({"at": _now(), "role": "system",
                           "text": f'你把 MoE 后端切到 **{spec["label"]}**'
                                   f'（{"、".join(f"`{k}={v}`" for k, v in spec["apply"].items())}）'})
        _persist(s)
        return _state(s)

    if inp.param == "model_name":
        m = get_model(inp.value)
        if m:
            s["current"]["params"]["model_name"] = str(inp.value)
            # 会话名 = 模型名。会话列表里靠这个认人，"未命名会话" 排一屏没法选。
            # 用户手动改过就不再自动覆盖。
            if not s.get("title_manual"):
                s["title"] = m["label"]
            msg = (f'模型设成 **{m["label"]}** —— {m["layers"]} 层'
                   + (f'、{m["num_experts"]} 专家 top-{m["top_k"]}' if m["moe"] else '、dense')
                   + f'、hidden {m["hidden"]}、mlp {m["mlp"]}。')
            if m["provenance"] == "public":
                msg += " 形状来自公开 config；**我们没有它在 v7 上的实测数字**，别套用别的模型的结论。"
            if m.get("note"):
                msg += " " + m["note"]
            s["turns"].append({"at": _now(), "role": "system", "text": msg})
            _persist(s)
            return _state(s)

    if inp.param == "__topology":
        from .parser import TOPOLOGIES as T
        name = str(inp.value)
        t = dict(T.get(name, {})); t["topology"] = name; t["slices"] = 1; t["needs_input"] = not t
        s["current"]["target"] = t
        s["turns"].append({"at": _now(), "role": "system", "text": f"你把拓扑设成 `{name}`"})
    else:
        old = s["current"]["params"].get(inp.param)
        if inp.value in (None, ""):
            s["current"]["params"].pop(inp.param, None)
            s["turns"].append({"at": _now(), "role": "system", "text": f"你删掉了 `{inp.param}`"})
        else:
            s["current"]["params"][inp.param] = _coerce(str(inp.value))
            s["turns"].append({"at": _now(), "role": "system",
                               "text": f"你把 `{inp.param}` 从 `{old}` 改成 `{inp.value}`"})
    _persist(s)
    return _state(s)


@app.post("/api/run")
async def run(inp: ChatIn):
    s = _get_session(inp.session_id or "")
    r = _recompute(s)
    fatal = [f for f in r["lint"] if f["severity"] == "fatal"]
    if s["current"]["target"].get("needs_input") or not s["current"]["target"].get("topology"):
        raise HTTPException(400, "先选目标拓扑")
    rid = "aot_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    result = await run_aot(s["current"]["params"], s["current"]["target"], r["aot_cmd"])
    result["lint_at_run"] = r["lint"]
    fp = _fingerprint(s["current"])
    doc = {"schema_version": 1, "id": rid, "created_at": _now(), "created_by": "local",
           "status": "done", "fingerprint": fp, "input": {"raw_cmd": s["current"].get("raw_cmd", ""),
                                       "params": s["current"]["params"],
                                       "xla_flags": s["current"]["xla_flags"],
                                       "target": s["current"]["target"], "aot_cmd": r["aot_cmd"]},
           "lint": r["lint"], "result": result, "metrics": result.get("metrics", {})}
    try:
        store.put("tpuguru", rid, doc)
    except Exception as e:  # noqa: BLE001
        log.warning("run 落库失败: %s", e)
    s["run_ids"].append(rid)
    entry = {"run_id": rid, "fingerprint": fp, **result}
    s.setdefault("results", {})[fp] = entry          # 本会话：不带 from_cache
    s["last_result"] = entry
    # 进程级缓存里存**带来源**的那份 —— 别的会话取到时要能看出「这是调档，不是刚跑的」
    _FP_CACHE[fp] = {**entry, "from_cache": {"at": doc["created_at"], "run_id": rid}}
    _v = verdict_of(result)          # ★ 结论文案唯一来源，前端也用同一份
    result["verdict"] = _v
    entry["verdict"] = _v
    verdict = _v["title"]
    extra = f"，{_v['detail']}" if _v["detail"] else ""
    if result.get("ok") is False and not _v["is_oom"]:
        extra += "。**这不是显存问题，降 batch 没用。**"
    if fatal:
        extra += f"（注意：跑之前就有 {len(fatal)} 条致命 lint）"
    s["turns"].append({"at": _now(), "role": "guru",
                       "text": f"AOT 跑完（{result.get('mode')} 模式）：**{verdict}**{extra}。"
                               "报告在右边。"})
    _persist(s)
    return _state(s)


# ── 真机 ───────────────────────────────────────────────────────
_METAL: dict[str, dict] = {}     # run_name → 进度


async def _metal_worker(name: str, y: str, gcs_out: str, fp: str, sid: str):
    """后台：提交 → 等结束 → 采集 → 落库。**不阻塞页面。**"""
    st = _METAL[name]
    try:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(y)
            yml = f.name
        rc, out = await metal._sh("kubectl", "apply", "-f", yml, timeout=120)
        st.update(phase="submitted", detail=out.strip()[:200])
        if rc != 0:
            st.update(phase="failed", detail=out[-400:])
            return
        deadline = time.time() + 60 * 75
        while time.time() < deadline:
            await asyncio.sleep(45)
            _, po = await metal._sh("kubectl", "get", "pods", "-n", metal.NS,
                                    "--no-headers", timeout=60)
            mine = [l for l in po.splitlines() if l.startswith(name)]
            run = sum(1 for l in mine if " Running " in f" {l} ")
            done = sum(1 for l in mine if " Completed " in f" {l} ")
            bad = sum(1 for l in mine if " Error" in l or "CrashLoop" in l)
            st.update(phase="running", pods=len(mine), running=run,
                      completed=done, failed=bad,
                      elapsed_min=round((time.time() - st["t0"]) / 60, 1))
            if mine and (done == len(mine) or bad):
                break
        st.update(phase="collecting")
        res = await metal.collect(name, gcs_out, fp)
        rid = "metal_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        doc = {"schema_version": 1, "id": rid, "created_at": _now(), "created_by": "local",
               "fingerprint": fp, "session_id": sid, "kind": "metal", **res}
        store.put("tpuguru_metal", rid, doc)
        st.update(phase="done", result=res, doc_id=rid)
        s_obj = SESSIONS.get(sid)
        if s_obj is not None:
            s_obj.setdefault("metal", {})[fp] = {**res, "doc_id": rid}
            s_obj["turns"].append({"at": _now(), "role": "guru",
                                   "text": f"🚀 真机跑完：{res['metrics'].get('tflops_per_chip')} "
                                           f"TFLOP/s/chip，step {res['metrics'].get('step_s')} s。"
                                           f"trace 已上 XProf。"})
    except Exception as e:  # noqa: BLE001
        log.error("真机流程失败 %s: %s", name, e, exc_info=True)
        st.update(phase="failed", detail=str(e))


class MetalIn(BaseModel):
    session_id: str
    topology: str | None = None      # 不传就用 AOT 验证过的那个


@app.post("/api/metal")
async def metal_run(inp: MetalIn):
    """上真机。**只有当前配置的 AOT 编译通过才允许** —— 装不下就上机是浪费别人的卡。"""
    s = _get_session(inp.session_id)
    fp = _fingerprint(s["current"])
    res = (s.get("results") or {}).get(fp) or _lookup_run(fp)
    if not res or res.get("ok") is not True or res.get("invalid"):
        raise HTTPException(400, "这套配置的 AOT 还没编译通过，不能上机")
    prof = metal_profile()
    if not prof:
        raise HTTPException(400, "没有执行档案（TPUGURU_AOT_PROFILE），无法上机")
    tgt = dict(s["current"]["target"])
    if inp.topology and inp.topology != tgt.get("topology"):
        # 换卡数 = 换分片宽度 = 显存结论全部作废。**AOT 三分钟零张卡，先去跑一遍。**
        raise HTTPException(400,
            f"AOT 验证的是 {tgt.get('topology')}，你要上 {inp.topology} —— "
            f"卡数变了分片宽度就变了，那份显存结论对这个规模不成立。"
            f"先把拓扑改成 {inp.topology} 再跑一次 AOT（3 分钟，不占卡）。")
    name = metal._run_name(fp)
    y, gcs_out = metal.build_jobset(name, s["current"]["params"], tgt, prof)
    _METAL[name] = {"name": name, "phase": "submitting", "t0": time.time(),
                    "fingerprint": fp, "gcs": gcs_out}
    asyncio.create_task(_metal_worker(name, y, gcs_out, fp, s["id"]))
    s["turns"].append({"at": _now(), "role": "system",
                       "text": f"🚀 已提交 64 卡真机任务 `{name}`，约 20–40 分钟。"
                               f"跑完 trace 会上 XProf。"})
    _persist(s)
    return {"run_name": name, "gcs": gcs_out, "state": _state(s)}


class MetalAnalyzeIn(BaseModel):
    session_id: str
    force: bool = False


@app.post("/api/metal/analyze")
async def metal_analyze(inp: MetalAnalyzeIn):
    """真机报告的「分析」—— 跟 AOT 那边一样，只喂事实不喂结论。

    真机能问、而 AOT 问不了的问题：算得快不快、通信藏没藏住、
    这个 MFU 在这个规模上算不算正常、下一步该往哪调。
    """
    s = _get_session(inp.session_id)
    fp = _fingerprint(s["current"])
    md = (s.get("metal") or {}).get(fp) or _lookup_metal(fp)
    if not md:
        raise HTTPException(400, "这套配置还没有真机结果")
    if not inp.force and md.get("analysis"):
        return {**md["analysis"], "cached": True}
    aot = (s.get("results") or {}).get(fp) or _lookup_run(fp) or {}
    m = md.get("metrics") or {}
    p = s["current"]["params"]
    t = s["current"]["target"]
    facts = [
        f"配置：{p.get('model_name')} / pdbs {p.get('per_device_batch_size')} / "
        f"{t.get('topology')}（{t.get('chips')} 芯片）/ quant={p.get('quantization')} / "
        f"cal={p.get('weight_quantization_calibration_method')} / "
        f"FSDP={fsdp_width(p, t)} / EP={p.get('ici_expert_parallelism')}",
        f"\nAOT 事先说：峰值 {(aot.get('metrics') or {}).get('peak_hbm_gb')} GB / "
        f"上限 94.74 GB（{aot.get('mode')} 模式）",
        f"\n真机实测：",
        f"  每芯片 {m.get('tflops_per_chip')} TFLOP/s（框架按 device 报 "
        f"{m.get('tflops_per_device')}，v7 是 2 device/chip 所以 ×2）",
        f"  MFU {m.get('mfu_pct')}%（分母 BF16 峰值 2307）",
        f"  step 中位 {m.get('step_s')} s，区间 {m.get('step_s_min')}–{m.get('step_s_max')} s",
        f"  稳态 {m.get('steady_steps')} 步，跳过前 {m.get('warmup_skipped')} 步",
        f"  loss {m.get('loss_first')} → {m.get('loss_last')}",
        f"  参数量 {m.get('params_b')} B",
    ]
    if m.get("warn"):
        facts.append(f"  ⚠️ {m['warn']}")
    prompt = ("你在给一个 TPU 训练调优工具写「真机测试分析」。下面是一次 v7 真机训练的"
              "实测结果（不是估算）。**这是一份独立报告，读它的人没看过你之前写的任何东西 ——"
              "不要出现「上次」「之前那份」这类指代。**\n\n"
              "写一段 markdown，450 字内，分四段：\n"
              "1. **这个数字算好还是不好** —— 放在这个模型规模与并行策略下判断，"
              "跟同类配方比。别只说数字大小\n"
              "2. **瓶颈可能在哪** —— 从 MFU、step 抖动、显存占用推断，说清是推断不是实测\n"
              "3. **AOT 和真机对上了吗** —— 显存预测准不准\n"
              "4. **下一步试什么** —— 两三条，按收益排序，每条说代价\n\n"
              "纪律：区分事实与推断；loss 全 0 之类的异常要先指出来再谈吞吐；"
              "不确定就说不确定。\n\n" + "\n".join(facts))
    out = {"ok": True, "explain": await _botcall("explain", prompt, {}) or "",
           "facts": facts, "analyzed_at": _now()}
    md["analysis"] = out
    s.setdefault("metal", {})[fp] = md
    try:
        if md.get("doc_id"):
            doc = store.get("tpuguru_metal", md["doc_id"])
            if doc:
                doc["analysis"] = out
                store.put("tpuguru_metal", md["doc_id"], doc)
    except Exception as e:  # noqa: BLE001
        log.warning("真机分析落库失败: %s", e)
    _persist(s)
    return out


@app.get("/api/metal/{name}")
async def metal_status(name: str):
    st = _METAL.get(name)
    if not st:
        raise HTTPException(404, "没有这个真机任务")
    return {k: v for k, v in st.items() if k != "t0"}


def metal_profile():
    from .worker import _profile
    return _profile()


class HloIn(BaseModel):
    session_id: str
    explain: bool = True
    force: bool = False          # 重新分析（默认吃缓存）


@app.post("/api/hlo")
async def hlo_analyze(inp: HloIn):
    """深挖 HLO：结构化统计 + 让带 skill 的 agent 写一段解读。

    **只把事实喂给 agent，不喂结论** —— 结论让它自己下，
    这样它给的建议才可能超出我预先写死的那几条。
    """
    s = _get_session(inp.session_id)
    fp = _fingerprint(s["current"])
    # 分析很贵（读 40 MB dump + 一次 bot 调用 1–2 分钟），**结果必须留下来** ——
    # 每次回到这一页都重算，等于把用户的时间当免费的。
    if not inp.force:
        cached = _lookup_analysis(s, fp)
        if cached:
            return {**cached, "cached": True}
    res = (s.get("results") or {}).get(fp) or _lookup_run(fp)
    d = ((res or {}).get("artifacts_total") or {}).get("dir")
    if not d:
        raise HTTPException(400, "这套配置没有本地 HLO 产物 —— 需要 real 模式跑过一次")
    # 优先用**那次编译真正传进去的** flag 串；没有才退回会话里粘命令带的
    uf = dict(s["current"].get("xla_flags") or {})
    used = (res or {}).get("xla_flags_used") or ""
    if used:
        uf = {}
        for tok in used.split():
            if tok.startswith("--"):
                k, _, v = tok[2:].partition("=")
                uf[k] = v or True
    a = hlomod.analyze(d, uf)
    if not a.get("ok"):
        raise HTTPException(400, a.get("why", "分析失败"))
    if inp.explain:
        prompt = (
            "你在给一个 TPU 训练调优工具写「AOT 产物分析」。下面是一次 AOT 编译的 "
            "HLO 统计，**全部来自真实 dump 文件，不是估算**。\n\n"
            "写一段 markdown 分析，**不超过 450 字**，分五段：\n"
            "1. **显存都花在哪** —— 最大的那几块是什么张量、为什么这么大、能不能小。"
            "把形状拆开验算（比如 batch×seq×vocab 是不是等于那个数），算得对不对本身就是信息\n"
            "2. **编译器做了什么** —— 从 fusion 数量与种类判断。"
            "kCustom 少意味着手写 kernel 没上，这条要点出来\n"
            "3. **通信状况** —— 拿集合通信次数跟**层数**比：远小于层数说明被合并/提升出循环了\n"
            "4. **配置有没有真的生效** —— flag、精度这些，传了不等于生效\n"
            "5. **值得动手的两三件事** —— 按能腾出多少显存或多少吞吐排序，每条写清代价\n\n"
            "硬性纪律：\n"
            "- **这是一份独立的报告**，读它的人没看过你之前写的任何东西。"
            "绝对不要出现「上次」「本轮第一条」「我收回之前」这类指代\n"
            "- 区分**事实**（dump 里读到的）与**推断**（由此猜的），推断要说明是推断\n"
            "- 不要复述数字，要解释它意味着什么\n"
            "- 不确定就写不确定，别为了凑满五段编内容\n\n"
            + hlomod.digest(a, s["current"]["params"], s["current"]["target"]))
        a["explain"] = await _botcall("explain", prompt, {}) or ""
    a["analyzed_at"] = _now()
    _save_analysis(s, fp, a)
    return a


@app.post("/api/save")
async def save(inp: SaveIn):
    """💾 存档 —— README §4.6。内容全部内联复制，不指向别的 doc。"""
    s = _get_session(inp.session_id)
    r = _recompute(s)
    _fp = _fingerprint(s["current"])
    res = (s.get("results") or {}).get(_fp) or _lookup_run(_fp) or {}
    said = "save_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    doc = {
        "schema_version": 1, "id": said, "created_at": _now(), "created_by": "local",
        "title": inp.title, "note": inp.note, "tags": inp.tags,
        "parent_save_id": s.get("parent_save_id"),
        # 报告已经判定这一档的数字无效时，存档**自动**标作废 ——
        # 不能指望人记得回头去点。一个划掉的标题旁边打着 ✅ 是自相矛盾的。
        "voided": ({"at": _now(), "by": "auto", "reason": res["invalid"]}
                   if res.get("invalid") else None),
        "frozen_at": _now(),
        "source": {"session_id": s["id"], "run_ids": list(s["run_ids"])},
        # ── 以下全是副本 ──
        "config": {"params": dict(s["current"]["params"]),
                   "xla_flags": dict(s["current"]["xla_flags"]),
                   "target": dict(s["current"]["target"]),
                   "train_cmd": s["current"].get("raw_cmd", ""), "aot_cmd": r["aot_cmd"]},
        "lint": {"rules_version": 1, "findings": r["lint"]},
        "analyses": (res.get("analyses") or {}),
        "conversation": list(s["turns"]),
        "metrics": res.get("metrics", {}),
        "fingerprint": _fingerprint(s["current"]),
        "artifacts": dict(res.get("artifacts") or {}),   # 真跑时这里是 saves/<id>/ 下的副本
        "attachments": [],
        "result_summary": {"ok": res.get("ok"), "mode": res.get("mode"),
                           "failure": res.get("failure"), "source": res.get("source"),
                           "invalid": bool(res.get("invalid")),
                           "per_chip_tflops": (res.get("metrics") or {}).get("per_chip_tflops")},
    }
    store.put("tpuguru_saves", said, doc)
    s["parent_save_id"] = said
    s["turns"].append({"at": _now(), "role": "system", "text": f"💾 已存档：**{inp.title}**"})
    _persist(s)
    return {"save_id": said, "state": _state(s)}


@app.get("/api/saves")
async def saves():
    items = store.list("tpuguru_saves")
    items.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return {"saves": [{k: d.get(k) for k in
                       ("id", "title", "note", "tags", "created_at", "parent_save_id",
                        "voided", "metrics", "result_summary")} for d in items]}


@app.get("/api/save/{said}")
async def get_save(said: str):
    d = store.get("tpuguru_saves", said)
    if not d:
        raise HTTPException(404, "存档不存在")
    return d


@app.post("/api/save/{said}/load")
async def load_save(said: str):
    """载入 = 派生新会话，不是就地编辑（README §4.6.3）。"""
    d = store.get("tpuguru_saves", said)
    if not d:
        raise HTTPException(404, "存档不存在")
    s = _new_session(title=f"从「{d['title']}」派生")
    cfg = d.get("config", {})
    s["current"] = {"params": dict(cfg.get("params", {})), "xla_flags": dict(cfg.get("xla_flags", {})),
                    "target": dict(cfg.get("target", {})), "raw_cmd": cfg.get("train_cmd", "")}
    s["parent_save_id"] = said
    s["turns"].append({"at": _now(), "role": "system",
                       "text": f"已载入存档 **{d['title']}**，这是一场新对话（原存档只读）。"})
    _persist(s)
    return _state(s)


class AttachIn(BaseModel):
    kind: str
    uri: str = ""
    text: str = ""
    data: dict = {}


@app.post("/api/save/{said}/attach")
async def attach(said: str, inp: AttachIn):
    """唯一可写的字段：只能 append。"""
    d = store.get("tpuguru_saves", said)
    if not d:
        raise HTTPException(404, "存档不存在")
    d.setdefault("attachments", []).append({"at": _now(), "by": "local", "kind": inp.kind,
                                            "uri": inp.uri, "text": inp.text, "data": inp.data})
    store.put("tpuguru_saves", said, d)
    return {"ok": True, "attachments": d["attachments"]}


class VoidIn(BaseModel):
    reason: str


@app.post("/api/save/{said}/void")
async def void(said: str, inp: VoidIn):
    """作废不删除 —— 记着「这条路走不通、为什么」的存档同样有价值。"""
    d = store.get("tpuguru_saves", said)
    if not d:
        raise HTTPException(404, "存档不存在")
    d["voided"] = {"at": _now(), "by": "local", "reason": inp.reason}
    store.put("tpuguru_saves", said, d)
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


@app.exception_handler(Exception)
async def on_error(request, exc):
    log.error("未处理异常 %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse({"error": str(exc)}, status_code=500)


if FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
