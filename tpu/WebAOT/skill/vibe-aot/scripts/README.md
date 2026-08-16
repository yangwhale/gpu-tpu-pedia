# scripts

agent 直接调用的可执行脚本。**能用脚本确定性拿到的字段，不要让 agent 读日志猜。**

| 脚本 | 状态 |
|---|---|
| `submit.sh` | 待写 |
| `collect.sh` | 待写 |
| `extract.py` | 待写 |
| `lint.py` | 待写（规则读 `rules/` 指向的真源） |

现成可复用：`../../../worker/probe_codepath.py`（已验证）。
