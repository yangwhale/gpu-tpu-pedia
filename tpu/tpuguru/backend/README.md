# backend — FastAPI

**职责**：解析命令 → 转成 `train_compile` 调用 → 投递任务 → 读写 Firestore。

| 接口 | 用途 |
|---|---|
| `POST /api/runs` | 提交一次 AOT，返回 `run_id` |
| `GET  /api/runs` | 历史列表（支持 `tags` / `status` / 按 `metrics.*` 排序） |
| `GET  /api/runs/{id}` | 单次详情 |
| `POST /api/runs/{id}/fork` | 复制并改参数派生新 run（写 `parent_id`） |
| `GET  /api/diff?a=&b=` | 两次运行逐维度对比 |
| `GET  /api/rules` | lint 规则库（供前端做实时校验） |
| `POST /api/lint` | 只跑静态检查，不编译 |

**不要感知反代前缀** —— 前端资源用相对路径，跳板机侧 `strip_prefix` 即可。
