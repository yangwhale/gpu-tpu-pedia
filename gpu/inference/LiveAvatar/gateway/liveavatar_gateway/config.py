"""网关配置 —— 全部从环境变量读，不落盘、不进 git。"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


@dataclass(frozen=True)
class Settings:
    # ── LiveKit 凭据。网关要自己铸 worker 的房间 token，所以必须持有这一对。
    #    （契约见 livekit/agents/inference/avatar.py:396 的注释）
    livekit_api_key: str
    livekit_api_secret: str

    # ── 状态库。默认 SQLite，单实例控制面足够；换 Postgres 只要改这一个 URL。
    db_path: str = "/var/lib/liveavatar-gateway/state.db"

    # ── 会话生命周期
    #    idle_timeout 是必须项不是加分项：客户端跑掉而不调 terminate 的话，
    #    槽位会永久泄漏，8 张卡漏一张就少一路。
    idle_timeout_s: int = 120
    max_session_s: int = 3600
    worker_heartbeat_timeout_s: int = 30

    # ── 给 worker 的生成参数（实测最优，见 OPTIMIZATION-JOURNAL.md）
    size: str = "384*256"
    trim_k: int = 4
    sample_rate: int = 16000          # 与 LiveKit 默认一致，省一次重采样

    provider_name: str = "liveavatar"

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.environ.get("LIVEKIT_API_KEY", "")
        secret = os.environ.get("LIVEKIT_API_SECRET", "")
        if not key or not secret:
            raise RuntimeError(
                "LIVEKIT_API_KEY / LIVEKIT_API_SECRET 必须设置 —— "
                "网关要用它们铸 worker 的房间 token"
            )
        return cls(
            livekit_api_key=key,
            livekit_api_secret=secret,
            db_path=os.environ.get("LA_GATEWAY_DB", cls.db_path),
            idle_timeout_s=_int("LA_IDLE_TIMEOUT_S", cls.idle_timeout_s),
            max_session_s=_int("LA_MAX_SESSION_S", cls.max_session_s),
            worker_heartbeat_timeout_s=_int("LA_WORKER_HB_TIMEOUT_S", cls.worker_heartbeat_timeout_s),
            size=os.environ.get("LA_SIZE", cls.size),
            trim_k=_int("LA_TRIM_K", cls.trim_k),
            sample_rate=_int("LA_SAMPLE_RATE", cls.sample_rate),
        )
