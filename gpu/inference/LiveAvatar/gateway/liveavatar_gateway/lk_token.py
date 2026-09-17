"""铸 worker 的房间 token。

这是整个网关最容易搞错的一块，语义来自 livekit/agents/inference/avatar.py:396：

  「room_name / avatar_identity / agent_identity 是网关铸 worker 房间 token 的输入，
    不只是标注：它把 roomJoin 授权限定到 room_name，以 avatar_identity 加入，
    并把 lk.publish_on_behalf 设成 agent_identity。」

三条都不能省：
  - roomJoin 限定到那一个房间 —— 否则一把 token 能串进所有房间
  - identity 必须是 avatar_identity —— 客户端靠它找到数字人参与者
  - lk.publish_on_behalf —— 没有它，客户端 SDK 不认为这是 agent 的化身，
    Room.agentParticipants / Participant.avatarWorker 的正反向查找都会落空
"""
from __future__ import annotations

from livekit import api


def mint_worker_token(
    *,
    api_key: str,
    api_secret: str,
    room_name: str,
    avatar_identity: str,
    avatar_name: str,
    agent_identity: str,
    provider: str,
    ttl_s: int = 3600,
) -> str:
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(avatar_identity)
        .with_name(avatar_name)
        .with_kind("agent")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,          # ⛔ 必须限定，不能给通配
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                agent=True,
            )
        )
        .with_attributes(
            {
                "lk.publish_on_behalf": agent_identity,
                "lk.avatar_provider": provider,
            }
        )
        .with_ttl(__import__("datetime").timedelta(seconds=ttl_s))
        .to_jwt()
    )
