"""控制面状态。SQLite 单文件 —— 控制面是单实例，够用且没有外部依赖。

换 Postgres 只需要替换这一个类：其余模块只通过这里的方法访问状态。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS workers (
    worker_id   TEXT PRIMARY KEY,
    capacity    INTEGER NOT NULL,
    registered_at REAL NOT NULL,
    last_seen   REAL NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS sessions (
    provider_session_id TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    worker_id   TEXT NOT NULL,
    key_id      TEXT NOT NULL,
    room_name   TEXT NOT NULL,
    state       TEXT NOT NULL,            -- pending | active | closed
    created_at  REAL NOT NULL,
    last_activity REAL NOT NULL,
    job         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_worker ON sessions(worker_id, state);
CREATE INDEX IF NOT EXISTS idx_sessions_key    ON sessions(key_id, state);
CREATE TABLE IF NOT EXISTS idempotency (
    key         TEXT PRIMARY KEY,
    key_id      TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


@dataclass
class Worker:
    worker_id: str
    capacity: int
    last_seen: float
    meta: dict


@dataclass
class Session:
    provider_session_id: str
    session_id: str
    worker_id: str
    key_id: str
    room_name: str
    state: str
    created_at: float
    last_activity: float
    job: dict


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # check_same_thread=False + 一把锁：FastAPI 的线程池会跨线程碰它
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._lock = threading.RLock()

    # ── workers ──────────────────────────────────────────────
    def upsert_worker(self, worker_id: str, capacity: int, meta: dict) -> None:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO workers(worker_id,capacity,registered_at,last_seen,meta) "
                "VALUES(?,?,?,?,?) ON CONFLICT(worker_id) DO UPDATE SET "
                "capacity=excluded.capacity, last_seen=excluded.last_seen, meta=excluded.meta",
                (worker_id, capacity, now, now, json.dumps(meta)),
            )
            self._db.commit()

    def touch_worker(self, worker_id: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE workers SET last_seen=? WHERE worker_id=?", (time.time(), worker_id)
            )
            self._db.commit()
            return cur.rowcount > 0

    def live_workers(self, timeout_s: float) -> list[Worker]:
        cutoff = time.time() - timeout_s
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM workers WHERE last_seen >= ?", (cutoff,)
            ).fetchall()
        return [Worker(r["worker_id"], r["capacity"], r["last_seen"], json.loads(r["meta"]))
                for r in rows]

    def drop_worker(self, worker_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))
            self._db.commit()

    # ── sessions ─────────────────────────────────────────────
    def active_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT worker_id, COUNT(*) n FROM sessions "
                "WHERE state IN ('pending','active') GROUP BY worker_id"
            ).fetchall()
        return {r["worker_id"]: r["n"] for r in rows}

    def active_for_key(self, key_id: str) -> int:
        with self._lock:
            r = self._db.execute(
                "SELECT COUNT(*) n FROM sessions WHERE key_id=? AND state IN ('pending','active')",
                (key_id,),
            ).fetchone()
        return r["n"]

    def create_session(self, s: Session) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?)",
                (s.provider_session_id, s.session_id, s.worker_id, s.key_id, s.room_name,
                 s.state, s.created_at, s.last_activity, json.dumps(s.job)),
            )
            self._db.commit()

    def get_session(self, provider_session_id: str) -> Session | None:
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM sessions WHERE provider_session_id=?", (provider_session_id,)
            ).fetchone()
        if not r:
            return None
        return Session(r["provider_session_id"], r["session_id"], r["worker_id"], r["key_id"],
                       r["room_name"], r["state"], r["created_at"], r["last_activity"],
                       json.loads(r["job"]))

    def set_state(self, provider_session_id: str, state: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE sessions SET state=?, last_activity=? WHERE provider_session_id=?",
                (state, time.time(), provider_session_id),
            )
            self._db.commit()

    def touch_session(self, provider_session_id: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE sessions SET last_activity=? WHERE provider_session_id=?",
                (time.time(), provider_session_id),
            )
            self._db.commit()

    def claim_pending(self, worker_id: str) -> Session | None:
        """worker 长轮询时取一个待办。取到即置 active，避免重复派发。"""
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM sessions WHERE worker_id=? AND state='pending' "
                "ORDER BY created_at LIMIT 1", (worker_id,)
            ).fetchone()
            if not r:
                return None
            self._db.execute(
                "UPDATE sessions SET state='active', last_activity=? WHERE provider_session_id=?",
                (time.time(), r["provider_session_id"]),
            )
            self._db.commit()
        return Session(r["provider_session_id"], r["session_id"], r["worker_id"], r["key_id"],
                       r["room_name"], "active", r["created_at"], time.time(),
                       json.loads(r["job"]))

    def reap(self, *, idle_timeout_s: float, max_session_s: float) -> list[Session]:
        """回收超时会话。返回被回收的，调用方负责通知 worker。"""
        now = time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sessions WHERE state IN ('pending','active') "
                "AND (last_activity < ? OR created_at < ?)",
                (now - idle_timeout_s, now - max_session_s),
            ).fetchall()
            for r in rows:
                self._db.execute(
                    "UPDATE sessions SET state='closed' WHERE provider_session_id=?",
                    (r["provider_session_id"],),
                )
            self._db.commit()
        return [Session(r["provider_session_id"], r["session_id"], r["worker_id"], r["key_id"],
                        r["room_name"], "closed", r["created_at"], r["last_activity"],
                        json.loads(r["job"])) for r in rows]

    def close_sessions_of_worker(self, worker_id: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "UPDATE sessions SET state='closed' WHERE worker_id=? AND state IN ('pending','active')",
                (worker_id,),
            )
            self._db.commit()
            return cur.rowcount

    # ── idempotency ──────────────────────────────────────────
    def get_idempotent(self, key: str, key_id: str) -> dict | None:
        with self._lock:
            r = self._db.execute(
                "SELECT response FROM idempotency WHERE key=? AND key_id=?", (key, key_id)
            ).fetchone()
        return json.loads(r["response"]) if r else None

    def put_idempotent(self, key: str, key_id: str, response: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO idempotency VALUES(?,?,?,?)",
                (key, key_id, json.dumps(response), time.time()),
            )
            self._db.commit()
