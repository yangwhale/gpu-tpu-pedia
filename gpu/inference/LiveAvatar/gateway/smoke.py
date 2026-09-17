#!/usr/bin/env python3
"""端到端冒烟：假 worker 注册 → 客户端建会话 → 取活 → 终止。

不碰 GPU、不连 LiveKit，纯走控制面契约。CI 里可以直接跑。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "d" * 32)
os.environ.setdefault("LA_GATEWAY_DB", ":memory:")

from fastapi.testclient import TestClient  # noqa: E402

from liveavatar_gateway.app import create_app  # noqa: E402
from liveavatar_gateway.auth import ApiKey, KeyRing, sign_client_token  # noqa: E402
from liveavatar_gateway.config import Settings  # noqa: E402

KEY = ApiKey(key_id="demo", secret="s" * 32, max_concurrency=8)
app = create_app(Settings.from_env(), KeyRing({"demo": KEY}))
c = TestClient(app)
AUTH = {"Authorization": f"Bearer {sign_client_token('demo', KEY.secret)}"}
ok = True


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  {'✓' if cond else '✗'} {label}{'  ' + detail if detail else ''}")
    ok = ok and cond


print("1. 没有 worker 时建会话应当 429")
body = dict(provider="liveavatar", livekit_url="ws://x", room_name="r1", room_sid="RM_1",
            avatar_identity="av-1", avatar_name="Av", agent_identity="agent-1")
r = c.post("/avatar/sessions", json=body, headers=AUTH)
check("返回 429", r.status_code == 429, f"实际 {r.status_code}")
check("带 Retry-After", "retry-after" in {k.lower() for k in r.headers})

print("2. 无 token 应当 401")
check("返回 401", c.post("/avatar/sessions", json=body).status_code == 401)

print("3. worker 上线")
check("注册成功", c.post("/internal/workers/register",
                     json={"worker_id": "w1", "capacity": 2, "meta": {}}).status_code == 200)
h = c.get("/healthz").json()
check("healthz 报 2 个槽位", h["slots_total"] == 2, str(h))

print("4. 建会话")
r = c.post("/avatar/sessions", json=body, headers={**AUTH, "Idempotency-Key": "idem-1"})
check("返回 200", r.status_code == 200, r.text[:120])
d = r.json()
for f in ("session_id", "provider_session_id", "terminate_token", "sample_rate"):
    check(f"响应含 {f}", f in d)
check("sample_rate=16000", d.get("sample_rate") == 16000)
check("不回泄 room_token", "room_token" not in d)

print("5. 幂等：同 key 重放同一结果")
r2 = c.post("/avatar/sessions", json=body, headers={**AUTH, "Idempotency-Key": "idem-1"})
check("session_id 一致", r2.json()["session_id"] == d["session_id"])
check("没有多占槽位", c.get("/healthz").json()["slots_used"] == 1)

print("6. worker 长轮询取活")
j = c.get("/internal/workers/w1/jobs").json()["job"]
check("取到活", j is not None)
if j:
    check("带房间 token", bool(j.get("room_token")))
    check("带 agent_identity", j.get("agent_identity") == "agent-1")
    check("带最优生成参数", j.get("size") == "384*256" and j.get("trim_k") == 4,
          f"size={j.get('size')} trim_k={j.get('trim_k')}")
check("同一个活不会被取第二次", c.get("/internal/workers/w1/jobs").json()["job"] is None)

print("7. 终止：错 token 必须拒")
bad = c.post("/avatar/sessions/terminate", headers=AUTH, json={
    "provider": "liveavatar", "provider_session_id": d["provider_session_id"],
    "terminate_token": "deadbeef"})
check("返回 403", bad.status_code == 403, f"实际 {bad.status_code}")

print("8. 终止：对的 token 放行并释放槽位")
good = c.post("/avatar/sessions/terminate", headers=AUTH, json={
    "provider": "liveavatar", "provider_session_id": d["provider_session_id"],
    "terminate_token": d["terminate_token"]})
check("返回 200", good.status_code == 200, good.text[:80])
check("槽位已释放", c.get("/healthz").json()["slots_used"] == 0)

print("\n" + ("全部通过 ✅" if ok else "有失败 ❌"))
sys.exit(0 if ok else 1)
