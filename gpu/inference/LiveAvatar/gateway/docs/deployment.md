# 部署

## 放哪儿

| 组件 | 机器 | 理由 |
|---|---|---|
| 控制面 | **常驻 CPU 小机器**（2 vCPU 足够） | ① GPU 机器（尤其 Spot）会消失，控制面不能跟着消失 ② 它**不在媒体路径上**，只在建/停会话时调一次，放哪儿都不影响帧率 ③ 无 GPU 依赖，装机简单 |
| Worker | **GPU 机器，一卡一进程** | 模型常驻显存，每路会话不付启动成本 |
| 状态 | 控制面本地 SQLite 单文件 | 单实例够用、零外部依赖。要多实例再换 Postgres（只需替换 `store.py`） |

**网络**：只需要 worker → 控制面的**出向** HTTP。GPU 机器不用开任何入向端口。

## 容量账

8 张 B200，一卡一路：**8 路并发 live avatar**，每路 384×256 / 1.357× 实时。

对照：同样 8 张卡用 5 卡 TPP 跑单路只有 1.687×。**按路数算，前者总吞吐是后者的 6.5 倍。**

## 必须配的两个超时

| 参数 | 默认 | 为什么不能不配 |
|---|---|---|
| `LA_IDLE_TIMEOUT_S` | 120 | 客户端跑掉而不调 terminate 时回收槽位。**漏一路就少一张卡。** LiveKit 源码里自己警告过两次会话会「一直计费到 idle 超时」 |
| `LA_MAX_SESSION_S` | 3600 | 一直有音频也不能永远占着卡 |

`LA_WORKER_HB_TIMEOUT_S`（默认 30）超时后，控制面会关掉该 worker 名下的会话并摘掉它。
**会话记录只置 `closed` 不删** —— 留痕才查得出「这一路是怎么没的」。

## 起服务

控制面写个几行的入口，把 key 从你的密钥库注入：

```python
# myapp.py
from liveavatar_gateway.app import create_app
from liveavatar_gateway.auth import ApiKey, KeyRing
from liveavatar_gateway.config import Settings

def build():
    keys = {k: ApiKey(key_id=k, secret=s, max_concurrency=n)
            for k, s, n in load_keys_from_your_secret_store()}
    return create_app(Settings.from_env(), KeyRing(keys))
```

```bash
uvicorn --factory myapp:build --host 0.0.0.0 --port 8080
```

worker 用 systemd 拉起，每张卡一个 unit，`Restart=always`。

## Agent 侧怎么接

跟接那 8 家商业供应商完全一样 —— 指到我们的网关就行：

```python
avatar = AvatarSession(
    "liveavatar/bunny-scholar",          # 或 image_url=<自定义形象>
    base_url="http://<控制面>:8080",
    api_key=..., api_secret=...,
)
await avatar.start(session, room=ctx.room)
```

## 运维检查

```bash
curl -s http://<控制面>:8080/healthz | jq
# {"status":"ok","slots_total":8,"slots_used":3,"slots_free":5,"keys":2}
```

槽位长期占满不掉 → 多半是 idle 回收没生效或有会话泄漏，查控制面日志里的
「回收超时会话」和「worker 心跳丢失」两条。
