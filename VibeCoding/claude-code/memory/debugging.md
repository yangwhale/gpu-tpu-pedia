# 调试经验和踩坑记录

> 这个文件记录工作中遇到的问题和解决方案，后续持续积累。

## GCP
- B200 asia-southeast1-b 会出现 ZONE_RESOURCE_POOL_EXHAUSTED，需要换 zone 或用 Spot
- **查 MIG 必须加 filter**: `--filter=name:chrisya`，不要 list 整个项目（几百个 MIG 会浪费大量 context）

## Discord Bot 子进程
- **`close_fds=False` 会杀死 bot**：用 `asyncio.create_subprocess_exec(..., close_fds=False)` 启动 Claude 子进程时，子进程继承 bot 的所有 FD（包括 Discord websocket）。Claude/Node.js 初始化 MCP servers 时关闭继承的 FD → Discord gateway 断开 → `bot.run()` 正常返回 → bot 静默退出
- 修复：`subprocess.Popen(..., close_fds=True)`，只传显式指定的 stdin/stdout/stderr
- 症状：bot 收到消息 → Claude 启动 → ~12秒后 bot clean exit，无错误日志，无 signal
- 排查关键：日志里 `bot.run() returned normally` 而非 crash，且无 signal 日志 → 排除 OOM/signal → 聚焦 FD 泄漏
- `asyncio.create_subprocess_exec` 的 `close_fds` 默认行为不可靠，应显式用 `subprocess.Popen` + `close_fds=True`
- **stderr 必须可观测**：子进程 `stderr=DEVNULL` 会吞掉所有错误信息，改为写临时文件
- **Claude 子进程弑父**: Claude 有 `--dangerously-skip-permissions` 的 bash 权限，改完 bot.py 后会 `pgrep + kill` 重启 bot，杀掉自己的父进程。修复：system prompt 中禁止 kill/restart bot 进程

## Vertex AI API 踩坑
- **Veo duration**: 文档说 5-8 秒，实际只支持 `4, 6, 8`，传 5 或 7 会报 `Unsupported output video duration`
- **Veo 响应格式不统一**: 不指定 `storageUri` 时，`videos[]` 数组里是 `{bytesBase64Encoded, mimeType}`；指定 `storageUri` 时是 `{gcsUri, mimeType}`。同一个 key 两种完全不同的数据结构
- **Veo 异步模式**: 用 `predictLongRunning` 提交 → 返回 `operation name` → 用 `fetchPredictOperation` polling → `done: true` 时取结果。不能用 Imagen 的同步 `predict`
- **Veo polling 注意点**: 生成一般 60-90s（fast 模型），token 可能过期，每次 poll 前要 refresh `gcloud auth print-access-token`
- **WeasyPrint PDF**: 不支持 `100vh`、`overflow-x: auto` 等交互式 CSS，但 `@page` 规则（页码、页眉、分页）支持很好，适合生成专业文档

## Agent Teams
- **幽灵队员问题**: Team config 持久化在 `~/.claude/teams/` 磁盘上，但 agent 进程随 session restart 被杀。新 session 启动时必须检查并清理残留的 team 目录（`rm -rf ~/.claude/teams/{team-name}`），否则会误以为队员还活着
- **清理时机**: 每次 session resume 开头，主动 `ls ~/.claude/teams/` 检查，有残留就清掉
- **异步通知模式**: teammate 完成任务后直接调用 `send-to-discord.sh` 通知用户，不经 lead 中转。通知链：`agent → Discord`（直达）+ `agent → lead`（记录）。比 `agent → lead → Discord` 更可靠，避免 lead turn 排队导致通知延迟
- **teammate prompt 模板**: 启动 agent 时 prompt 必须包含：(1) 启动后立刻 send-to-discord.sh 报到 (2) 完成任务先 send-to-discord.sh 通知用户再 SendMessage 汇报 lead (3) 出错也要 send-to-discord.sh 通知
- **消息错位问题**: teammate 的 idle/shutdown 消息和用户消息走同一个队列，无优先级区分，导致 lead 回复"过时"的 teammate 消息而用户消息被挤到后面。框架层限制，无法彻底解决
- **Team vs Sub Agent 选择**: 独立并行任务（无协作需求）用 sub agent 即可；需要成员间互相通信、迭代协作的场景才用 team
- **bypassPermissions 必须**: 后台 agent 必须用 `mode=bypassPermissions`，否则权限确认弹窗卡住流程
- **并行启动**: 多个 agent 在同一个 Task tool 调用中并行启动，用 `run_in_background=true`

## SSH
- WireGuard VPN (10.8.0.x) 连接不稳定，MIG 实例 IP 会变，需要定期更新 SSH config
