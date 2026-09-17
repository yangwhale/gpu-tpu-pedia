"""网关单元测试 —— 重点在鉴权和调度这两块「错了就出事」的逻辑。

刻意包含 negative test：只测 happy path 的测试等于没测。
"""
from __future__ import annotations

import time

import pytest

from liveavatar_gateway.auth import (
    ApiKey, AuthError, KeyRing, make_terminate_token, sign_client_token,
    verify_bearer, verify_terminate_token,
)
from liveavatar_gateway.scheduler import NoCapacity, Scheduler
from liveavatar_gateway.store import Session, Store

KEY = ApiKey(key_id="k1", secret="s" * 32, max_concurrency=2, label="test")
OTHER = ApiKey(key_id="k2", secret="t" * 32)
RING = KeyRing({"k1": KEY, "k2": OTHER})


# ── 鉴权 ────────────────────────────────────────────────────
def test_valid_token_passes():
    assert verify_bearer(sign_client_token("k1", KEY.secret), RING).key_id == "k1"


def test_wrong_secret_rejected():
    """用别人的 secret 签、冒充 k1 —— 必须拒。"""
    bad = sign_client_token("k1", OTHER.secret)
    with pytest.raises(AuthError):
        verify_bearer(bad, RING)


def test_unknown_key_rejected():
    with pytest.raises(AuthError):
        verify_bearer(sign_client_token("k9", "whatever"), RING)


def test_expired_token_rejected():
    with pytest.raises(AuthError):
        verify_bearer(sign_client_token("k1", KEY.secret, ttl_s=-10), RING)


def test_token_without_exp_rejected():
    """没有 exp 的 token 永不过期 —— 必须拒，否则泄漏一次等于永久失守。"""
    import jwt
    forever = jwt.encode({"iss": "k1"}, KEY.secret, algorithm="HS256")
    with pytest.raises(AuthError):
        verify_bearer(forever, RING)


def test_garbage_token_rejected():
    with pytest.raises(AuthError):
        verify_bearer("not-a-jwt", RING)


# ── terminate_token ─────────────────────────────────────────
def test_terminate_token_roundtrip():
    t = make_terminate_token(KEY.secret, "lap_x")
    assert verify_terminate_token(KEY.secret, "lap_x", t)


def test_terminate_token_wrong_session_rejected():
    """拿 A 会话的 token 去终止 B 会话 —— 必须拒。"""
    t = make_terminate_token(KEY.secret, "lap_a")
    assert not verify_terminate_token(KEY.secret, "lap_b", t)


def test_terminate_token_other_tenant_rejected():
    t = make_terminate_token(OTHER.secret, "lap_x")
    assert not verify_terminate_token(KEY.secret, "lap_x", t)


# ── 调度 ────────────────────────────────────────────────────
@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def _sess(psid: str, worker: str, key_id: str = "k1") -> Session:
    now = time.time()
    return Session(psid, "las_" + psid, worker, key_id, "room", "pending", now, now, {})


def test_no_workers_means_no_capacity(store):
    with pytest.raises(NoCapacity):
        Scheduler(store, heartbeat_timeout_s=30).pick_worker()


def test_picks_least_loaded(store):
    store.upsert_worker("w1", 2, {})
    store.upsert_worker("w2", 2, {})
    store.create_session(_sess("p1", "w1"))
    assert Scheduler(store, heartbeat_timeout_s=30).pick_worker().worker_id == "w2"


def test_full_pool_raises(store):
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    with pytest.raises(NoCapacity):
        Scheduler(store, heartbeat_timeout_s=30).pick_worker()


def test_stale_worker_not_scheduled(store):
    """心跳早就断了的 worker 不能再派活给它。"""
    store.upsert_worker("w1", 1, {})
    with pytest.raises(NoCapacity):
        Scheduler(store, heartbeat_timeout_s=-1).pick_worker()


def test_closed_session_frees_slot(store):
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    store.set_state("p1", "closed")
    assert Scheduler(store, heartbeat_timeout_s=30).pick_worker().worker_id == "w1"


def test_claim_pending_is_exactly_once(store):
    """两次 claim 不能拿到同一个活 —— 否则一路会被派给两张卡。"""
    store.upsert_worker("w1", 2, {})
    store.create_session(_sess("p1", "w1"))
    assert store.claim_pending("w1").provider_session_id == "p1"
    assert store.claim_pending("w1") is None


def test_idle_reap_closes_session(store):
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    reaped = store.reap(idle_timeout_s=-1, max_session_s=10_000)
    assert [s.provider_session_id for s in reaped] == ["p1"]
    assert store.get_session("p1").state == "closed"


def test_max_session_reap_closes_even_if_active(store):
    """一直有音频也不能永远占着卡。"""
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    store.touch_session("p1")
    assert len(store.reap(idle_timeout_s=10_000, max_session_s=-1)) == 1


def test_dead_worker_sweep_frees_its_sessions(store):
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    assert Scheduler(store, heartbeat_timeout_s=-1).sweep_dead_workers() == ["w1"]
    assert store.get_session("p1").state == "closed"


def test_per_key_concurrency_is_counted(store):
    store.upsert_worker("w1", 8, {})
    store.create_session(_sess("p1", "w1", "k1"))
    store.create_session(_sess("p2", "w1", "k2"))
    assert store.active_for_key("k1") == 1
    assert store.active_for_key("k2") == 1


def test_idempotency_replays_first_result(store):
    store.put_idempotent("idem-1", "k1", {"session_id": "A"})
    store.put_idempotent("idem-1", "k1", {"session_id": "B"})   # 重试
    assert store.get_idempotent("idem-1", "k1")["session_id"] == "A"


def test_idempotency_is_per_key(store):
    """别的租户用同一个 Idempotency-Key 不能读到你的结果。"""
    store.put_idempotent("idem-1", "k1", {"session_id": "A"})
    assert store.get_idempotent("idem-1", "k2") is None
