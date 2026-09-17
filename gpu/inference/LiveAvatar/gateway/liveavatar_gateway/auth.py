"""鉴权 —— 照抄 LiveKit 自己那一套，三层。

1. 调用方身份：客户端用 (api_key, api_secret) 签短期 JWT，走 Authorization: Bearer。
   跟 LiveKit 服务端 token 同构，所以客户端可以直接用 livekit.api.AccessToken 签。
2. 会话归属：terminate_token = HMAC(secret, provider_session_id)。
   /terminate 认这个 token 不认调用者 —— 防止别人终止你的会话。
3. 重试安全：Idempotency-Key 让重试复用第一次的结果，不会多占一张卡。
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

import jwt


class AuthError(Exception):
    """鉴权失败。调用方看到 401。"""


@dataclass(frozen=True)
class ApiKey:
    key_id: str
    secret: str
    max_concurrency: int = 8
    label: str = ""


class KeyRing:
    """API key 的来源。生产从外部密钥库注入，测试直接传 dict。

    ⚠️ 不要把 secret 写进代码或配置文件 —— 只从进程环境/密钥库拿。
    """

    def __init__(self, keys: dict[str, ApiKey]):
        self._keys = keys

    def get(self, key_id: str) -> ApiKey | None:
        return self._keys.get(key_id)

    def __len__(self) -> int:
        return len(self._keys)


def verify_bearer(token: str, ring: KeyRing, *, now: float | None = None) -> ApiKey:
    """验 Bearer JWT，返回它对应的 ApiKey。

    先不验签地读出 iss 拿到 key_id，再用该 key 的 secret 验签 —— 这是标准做法，
    因为得先知道用哪个 secret。未验签的读取结果**只用来选 key，不用于任何授权判断**。
    """
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError as e:
        raise AuthError(f"token 解不开: {e}") from e

    key_id = unverified.get("iss")
    if not key_id:
        raise AuthError("token 缺 iss（api key id）")

    api_key = ring.get(key_id)
    if api_key is None:
        raise AuthError("未知的 api key")

    try:
        jwt.decode(
            token,
            api_key.secret,
            algorithms=["HS256"],
            options={"verify_aud": False, "require": ["exp"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError("token 已过期") from e
    except jwt.PyJWTError as e:
        raise AuthError(f"签名校验失败: {e}") from e

    return api_key


def make_terminate_token(secret: str, provider_session_id: str) -> str:
    """证明「这个项目拥有这个会话」。只有会话创建者才算得出来。"""
    return hmac.new(
        secret.encode(), provider_session_id.encode(), hashlib.sha256
    ).hexdigest()


def verify_terminate_token(secret: str, provider_session_id: str, token: str) -> bool:
    """常数时间比较 —— 用 == 会泄漏前缀信息。"""
    return hmac.compare_digest(make_terminate_token(secret, provider_session_id), token)


def sign_client_token(key_id: str, secret: str, *, ttl_s: int = 600) -> str:
    """给客户端/测试用的签发辅助。生产里这一步在调用方那边做。"""
    now = int(time.time())
    return jwt.encode(
        {"iss": key_id, "nbf": now - 5, "exp": now + ttl_s, "sub": key_id},
        secret,
        algorithm="HS256",
    )
