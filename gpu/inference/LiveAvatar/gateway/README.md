# LiveAvatar Gateway

把 LiveAvatar 包装成 **LiveKit 眼里的第 9 家数字人供应商** —— agent 侧用标准
`AvatarSession(...)` 调用，跟接 HeyGen / Tavus 写法完全一样，复杂度全在网关里面。

> **契约来源**：`livekit/agents/inference/avatar.py`（HTTP 形状、token 铸造语义、
> `terminate_token` 用途）、`voice/avatar/_datastream_io.py`（房间内数据面、RPC 名、topic）、
> `voice/avatar/_types.py`（`VideoGenerator` 三方法）。**逐条读源码核实，非推测。**

## 现状

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P0 控制面** | HTTP 契约 ＋ 鉴权 ＋ 槽位调度 ＋ token 铸造 ＋ 假 worker | ✅ 完成，28 单测 ＋ 端到端冒烟全过 |
| **P1 真 worker** | 接 LiveAvatar；音频流式化改造 | ⬜ 未开始 |
| **P2 运维** | 配额、指标、压测 | ⬜ 未开始 |

P0 的 worker 发**静帧 ＋ 原音频**。它存在的唯一目的是**验证契约** ——
契约错了在这一步暴露，成本最低。

## 架构：控制面在 CPU，worker 在 GPU

```
Agent（任意机器）
 │  AvatarSession(...).start()  ── Bearer JWT
 ▼
控制面（常驻 CPU 小机器，无 GPU）
 │  鉴权 / 调度 / 铸票 / 回收 · SQLite 单文件
 ▲  ← worker 主动连上来（pull）
 │
Worker × N（GPU 机器，一卡一进程，模型常驻）
```

**为什么 worker 是 pull 不是 push：** GPU 机器**不需要对外开端口**，可以待在
NAT / 不同 VPC 后面；Spot 被回收后换台机器重新注册即可。控制面从不主动连 worker。

**为什么控制面单独放 CPU 机器：** GPU 机器（尤其 Spot）会消失，控制面不能跟着消失。
而且控制面**不在媒体路径上** —— 只在建/停会话时被调用一次，放哪儿都不影响帧率。

## 对外 API（LiveKit 契约，字段名不可改）

```http
POST /avatar/sessions
Authorization: Bearer <客户端用 api_key/api_secret 签的短期 JWT>
Idempotency-Key: <每次 start() 一个>

{"provider":"liveavatar","livekit_url":...,"room_name":...,"room_sid":...,
 "avatar_identity":...,"avatar_name":...,"agent_identity":...,
 "avatar_id":...|"image_url":...,"extra_kwargs":{}}

→ {"session_id","provider_session_id","terminate_token","sample_rate"}
```

```http
POST /avatar/sessions/terminate
{"provider","provider_session_id","terminate_token"}
```

`GET /healthz` 返回槽位占用。`/internal/*` 是 worker 用的，**不要对外暴露**。

### 三个容易做错的地方

1. **网关自己铸 LiveKit token，客户端不传 token 进来。**
   源码注释（`avatar.py:396`）：roomJoin 限定到 `room_name`，以 `avatar_identity` 加入，
   `lk.publish_on_behalf` 设成 `agent_identity`。**少了第三条，客户端 SDK 不认这是
   agent 的化身**，`Room.agentParticipants` 查找会落空。
2. **`terminate_token` 认 token 不认调用者。** 它是 `HMAC(secret, provider_session_id)`。
3. **`Retry-After` 要挂在 `HTTPException` 上**，设在注入的 `Response` 对象上会被丢掉
   —— 抛异常时 FastAPI 另建响应。这个 bug 是端到端冒烟抓到的。

## 鉴权

| 层 | 机制 | 防什么 |
|---|---|---|
| 调用方身份 | `api_key`/`api_secret` 签短期 JWT（HS256，**强制 `exp`**） | 冒名调用 |
| 会话归属 | `terminate_token = HMAC(secret, provider_session_id)`，常数时间比较 | 别人终止你的会话 |
| 重试安全 | `Idempotency-Key`，**按 key 隔离** | 重试多占一张卡 / 跨租户读结果 |
| 房间授权 | roomJoin 限定到那一个房间 | 一把 token 串所有房间 |
| 配额 | 每 key 最大并发 | 一个人占满全部卡 |

⛔ secret 只从进程环境/密钥库读，**不落盘、不进 git**。

## 跑起来

```bash
pip install -r requirements.txt

# 控制面（CPU 机器）
export LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
export LA_GATEWAY_DB=/var/lib/liveavatar-gateway/state.db
python -m uvicorn --factory 'myapp:build' --host 0.0.0.0 --port 8080
# build() 里 create_app(Settings.from_env(), KeyRing({...}))，key 从你的密钥库注入

# worker（每张 GPU 一个）
CUDA_VISIBLE_DEVICES=0 LA_GATEWAY_URL=http://<控制面>:8080 \
  python -m worker.runner --worker-id box-gpu0 --image /path/to/avatar.png
```

## 测试

```bash
python -m pytest tests/ -q     # 28 个单测，含 negative test
python smoke.py                # 端到端契约冒烟，不碰 GPU、不连 LiveKit
```

**做过变异测试**：往鉴权/调度/铸票里注入 8 种错误（跳过签名校验、不强制 `exp`、
`terminate_token` 恒真、忽略心跳超时、满了还派活、并发计数恒 0、
`publish_on_behalf` 拼错、房间授权不限定），**8/8 全被测试抓到**。

## P1 待办：接真 LiveAvatar

| 事项 | 说明 |
|---|---|
| **音频流式化**（主要工作量） | 上游 `get_audio_embed_bucket_fps` 要**整段音频**才能算 `num_repeat`。但生成循环每块只取 `audio_input[..., left:right]` 一个切片，**架构上不是死路** —— 改成「来一块编一块，还有音频就多转一轮」 |
| 出帧 | 用 `causal_s2v_pipeline_tpp_blockwise` 里那个 `yield`（另外两条 pipeline 都是攒完再返回） |
| 参数 | 384×256、`TRIM_K=4`、单卡，实测 1.357× 实时、抖动近零 |
| 首帧延迟 | 1.26 s，**压不下去**（`infer_frames` 砍到 1/4 只降 14%，吞吐掉到 1/3）。用「思考中」待机循环盖住 |

性能数据全部出处见 [`../OPTIMIZATION-JOURNAL.md`](../OPTIMIZATION-JOURNAL.md)。
