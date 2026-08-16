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

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .lint import run_lint
from .parser import TOPOLOGIES, fsdp_width, parse_command, roundtrip_check, to_aot
from .worker import docker_available, run_aot

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


def _fingerprint(cur: dict) -> str:
    """配置指纹。改了配置之后旧报告就不再对应当前配置 ——
    没有这个，用户会对着上一次的结论调这一次的参数。"""
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
         "run_ids": [], "last_result": None}
    SESSIONS[sid] = s
    return s


def _get_session(sid: str) -> dict:
    if sid in SESSIONS:
        return SESSIONS[sid]
    d = store.get("tpuguru_sessions", sid) if sid else None
    if d:
        SESSIONS[sid] = d
        return d
    return _new_session()


def _persist(s: dict):
    s["updated_at"] = _now()
    try:
        store.put("tpuguru_sessions", s["id"], {k: v for k, v in s.items() if k != "last_result"})
    except Exception as e:  # noqa: BLE001
        log.warning("会话持久化失败: %s", e)


def _recompute(s: dict) -> dict:
    """配置一动就重算：AOT 命令 + lint + 回环校验。"""
    cur = s["current"]
    parsed = {"params": cur["params"], "xla_flags": cur["xla_flags"],
              "target": cur["target"], "entrypoint": cur.get("entrypoint", ""),
              "config_yml": cur.get("config_yml", ""), "raw": cur.get("raw_cmd", "")}
    aot = to_aot(parsed)
    findings = run_lint(cur["params"], cur["target"], cur["xla_flags"])
    diffs = roundtrip_check(aot["cmd"], aot["params"])
    return {"aot_cmd": aot["cmd"], "dropped": aot["dropped"], "added": aot["added"],
            "lint": findings, "roundtrip": diffs}


def _state(s: dict) -> dict:
    r = _recompute(s)
    return {
        "session_id": s["id"], "title": s["title"], "turns": s["turns"],
        "params": s["current"]["params"], "xla_flags": s["current"]["xla_flags"],
        "target": s["current"]["target"], "fsdp_width": fsdp_width(s["current"]["params"], s["current"]["target"]),
        "aot_cmd": r["aot_cmd"], "dropped": r["dropped"], "added": r["added"],
        "lint": r["lint"], "roundtrip": r["roundtrip"],
        "run_ids": s["run_ids"], "result": s.get("last_result"),
        "parent_save_id": s.get("parent_save_id"),
        "fingerprint": _fingerprint(s["current"]),
    }


# ── 对话意图：确定性优先，兜不住才叫 bot ────────────────────────
_CAL_ALIASES = {"absmax": "absmax", "动态": "absmax", "fixed": "fixed,-224,224", "静态": "fixed,-224,224"}
_PARAM_ALIASES = {
    "batch": "per_device_batch_size", "pdbs": "per_device_batch_size",
    "fsdp": "ici_fsdp_parallelism", "校准": "weight_quantization_calibration_method",
    "ep": "ici_expert_parallelism", "专家并行": "ici_expert_parallelism",
    "层数": "num_decoder_layers", "序列长度": "max_target_length", "seq": "max_target_length",
}


def _looks_like_command(t: str) -> bool:
    return ("python" in t and "=" in t) or t.count("=") >= 4 or "LIBTPU_INIT_ARGS" in t


def _intent_diff(text: str, params: dict) -> list[dict] | None:
    """从一句话里抽出确定的参数改动。抽不出返回 None。"""
    diffs = []
    for m in re.finditer(r"([A-Za-z_][\w.]*)\s*(?:=|改成|设成|调到|换成)\s*([\w.,\-]+)", text):
        k, v = m.group(1), m.group(2)
        if k in params or k.startswith(("ici_", "per_", "base_", "weight_", "quant", "use_", "sa_")):
            diffs.append({"param": k, "from": params.get(k), "to": v, "reason": "你直接指定的"})
    for alias, real in _PARAM_ALIASES.items():
        m = re.search(alias + r"\s*(?:=|改成|设成|调到|换成|开到)?\s*([\w.,\-]+)", text)
        if m and not any(d["param"] == real for d in diffs):
            v = m.group(1)
            if real == "weight_quantization_calibration_method":
                v = _CAL_ALIASES.get(v, v)
            if re.fullmatch(r"[\w.,\-]+", v) and v not in ("的", "是", "了"):
                diffs.append({"param": real, "from": params.get(real), "to": v,
                              "reason": f"「{alias}」→ `{real}`"})
    if not diffs:
        m = re.search(r"(?:换|改)(?:成|到)?\s*(absmax|fixed|动态|静态)", text)
        if m:
            v = _CAL_ALIASES[m.group(1)]
            diffs.append({"param": "weight_quantization_calibration_method",
                          "from": params.get("weight_quantization_calibration_method"), "to": v,
                          "reason": "fixed 静态 scale 伤收敛，不开 QAG 就该用 absmax"})
    return diffs or None


async def _botcall(kind: str, text: str, context: dict) -> str:
    """走 tpuguru bot 的 web channel。挂了就降级，不能让页面卡死。"""
    payload = {"session_id": f"tpuguru-{kind}", "text": text}
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
            "bot_url": BOT_URL, "topologies": TOPOLOGIES}


@app.post("/api/session")
async def new_session():
    s = _new_session()
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
    doc = {"schema_version": 1, "id": rid, "created_at": _now(), "created_by": "local",
           "status": "done", "input": {"raw_cmd": s["current"].get("raw_cmd", ""),
                                       "params": s["current"]["params"],
                                       "xla_flags": s["current"]["xla_flags"],
                                       "target": s["current"]["target"], "aot_cmd": r["aot_cmd"]},
           "lint": r["lint"], "result": result, "metrics": result.get("metrics", {})}
    try:
        store.put("tpuguru", rid, doc)
    except Exception as e:  # noqa: BLE001
        log.warning("run 落库失败: %s", e)
    s["run_ids"].append(rid)
    s["last_result"] = {"run_id": rid, "fingerprint": _fingerprint(s["current"]), **result}
    verdict = ("❌ 装不下" if result.get("ok") is False else
               "✅ 编译通过" if result.get("ok") else "❓ 这一档没有记录")
    extra = ""
    if result.get("failure"):
        f = result["failure"]
        extra = f"，需要 **{f.get('required_gb')} GB** / 上限 {f.get('available_gb')} GB"
    if fatal:
        extra += f"（注意：跑之前就有 {len(fatal)} 条致命 lint）"
    s["turns"].append({"at": _now(), "role": "guru",
                       "text": f"AOT 跑完（{result.get('mode')} 模式）：**{verdict}**{extra}。"
                               "报告在右边。"})
    _persist(s)
    return _state(s)


@app.post("/api/save")
async def save(inp: SaveIn):
    """💾 存档 —— README §4.6。内容全部内联复制，不指向别的 doc。"""
    s = _get_session(inp.session_id)
    r = _recompute(s)
    res = s.get("last_result") or {}
    said = "save_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    doc = {
        "schema_version": 1, "id": said, "created_at": _now(), "created_by": "local",
        "title": inp.title, "note": inp.note, "tags": inp.tags,
        "parent_save_id": s.get("parent_save_id"), "voided": None,
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
                           "failure": res.get("failure"), "source": res.get("source")},
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
