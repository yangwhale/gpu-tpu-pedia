"""铸出来的 token 必须逐字满足 LiveKit 的三条要求，否则客户端找不到数字人。"""
from __future__ import annotations

import base64
import json

from liveavatar_gateway.lk_token import mint_worker_token


def _payload(tok: str) -> dict:
    p = tok.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(p + "=" * (-len(p) % 4)))


def _mint(**kw):
    args = dict(api_key="devkey", api_secret="d" * 32, room_name="room-x",
                avatar_identity="av-1", avatar_name="Av", agent_identity="agent-1",
                provider="liveavatar")
    args.update(kw)
    return _payload(mint_worker_token(**args))


def test_identity_is_avatar_identity():
    assert _mint()["sub"] == "av-1"


def test_room_join_is_scoped_to_one_room():
    """不限定房间的话，一把 token 能串进所有房间。"""
    v = _mint()["video"]
    assert v["roomJoin"] is True and v["room"] == "room-x"


def test_publish_on_behalf_points_at_agent():
    """少了这条，客户端 SDK 不认这是 agent 的化身，绿框和波形都不会亮。"""
    assert _mint()["attributes"]["lk.publish_on_behalf"] == "agent-1"


def test_avatar_provider_attribute_present():
    assert _mint()["attributes"]["lk.avatar_provider"] == "liveavatar"


def test_can_publish_and_subscribe():
    v = _mint()["video"]
    assert v["canPublish"] and v["canSubscribe"]


def test_token_expires():
    p = _mint()
    assert p["exp"] > p["nbf"]


def test_different_rooms_get_different_tokens():
    assert _mint(room_name="a")["video"]["room"] != _mint(room_name="b")["video"]["room"]
