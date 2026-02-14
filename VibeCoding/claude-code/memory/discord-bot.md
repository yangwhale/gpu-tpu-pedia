# Discord Bot

## 架构（2026-02-12 升级）
- **持久进程模式**: Unix socketpair + stream-json 双向通信（与 VSCode Claude Code 插件相同机制）
- 每用户独立 Claude Code 进程，互不干扰
- Claude 以完整交互模式运行，支持 auto memory、CLAUDE.md、skills
- 关键参数: `--permission-prompt-tool stdio` 保持进程存活
- 消息格式: `{"type": "user", "message": {"role": "user", "content": "..."}}`
- 响应: 等待 `{"type": "result"}` 消息
- 环境变量: `CLAUDE_CODE_DISABLE_AUTO_MEMORY=0` 显式传入子进程

## 文件位置
- Bot 脚本: `~/.claude/discord-bot/bot.py`
- 日志: `~/.claude/discord-bot/bot.log`
- Session 映射: `~/.claude/discord-bot/sessions.json`（含 active + history，最近 20 个）
- 发送脚本: `~/.claude/scripts/send-to-discord.sh`
- 部署 skill: `~/.claude/skills/discord-bot-setup/SKILL.md`

## 配置
- 白名单: `ALLOWED_USER_IDS = {1074613327805829190}` (Chris)
- 自动响应频道: 1471088850712531055 (🥶-claude-code)
- **STT 引擎**: `gemini`（默认），fallback chain: Gemini → Chirp 2 → Whisper
- **STT 模型**: `gemini-3-flash-preview`，region=`global`，thinking=`MINIMAL`
- Whisper 模型: medium（fallback 用）
- Claude 超时: 600 秒
- Discord 风格: `--append-system-prompt` 从 `~/.claude/skills/discord-style/SKILL.md` 动态加载格式规则（`load_discord_style()` 函数读取 skill body 注入）

## STT 模型评测（2026-02-14）
- `gemini-2.5-flash-lite`: 2.1s 延迟，质量 good
- `gemini-2.0-flash`: ~3s 延迟，质量 excellent
- `gemini-3-flash-preview` (thinking=MINIMAL): 3.4s 延迟，质量 excellent ← **当前使用**
- `gemini-2.0-flash-lite`: 3.3s 延迟，质量 weaker（TPU→TPO 错误）
- `gemini-3-flash-preview` (thinking=ON): 10+s 延迟，不可接受
- Chirp 2: 3-5s 延迟，质量 poor（同音字问题严重）
- Whisper medium (local): 2-3s 延迟，质量 ok
- 注意: Gemini 3 Flash Preview 只在 `global` region 可用，其他模型可用 `us-central1`
- 注意: `.env` 里 `STT_ENGINE` 会覆盖 bot.py 默认值

## 斜杠命令
- `/status` - Bot 状态、活跃进程数
- `/end` - 归档当前 session 并停止 Claude 进程
- `/sessions` - 列出历史 session（摘要 + 下拉菜单切换）
- `/restart` - 优雅重启 bot（exit code 42 → wrapper 自动重启）

## 运维
- 启动: `tmux new-session -d -s discord-bot 'bash ~/.claude/discord-bot/run.sh'`（wrapper 脚本 + tmux）
- 重启: Discord 里 `/restart`（exit code 42 → wrapper 自动重启），或手动 kill 后用上面命令启动
- 日志: `tail -f ~/.claude/discord-bot/bot.log`
- 子进程管理: `subprocess.Popen` + `close_fds=True`（不用 `asyncio.create_subprocess_exec`，避免 FD 泄漏）
- Claude stderr 写临时文件（不用 DEVNULL），方便排查启动失败
- on_message 有 try/except 保护，错误回复到 Discord 而不是让 bot 崩溃
- System prompt 禁止 Claude 子进程 kill/restart bot，改完代码后提示用户 `/restart`

## 安全
- message.author.id 由 Discord 服务端验证，可信
- 不用 /register 自注册，直接硬编码 User ID
- 关闭 Discord Developer Portal 的 User Install

## 踩坑记录
- **close_fds=False 会杀死 bot**: 子进程继承 Discord websocket FD → MCP server 初始化时关闭 → bot 静默退出（详见 debugging.md）
- 重复消息 = 多 bot 进程，不是 Discord 重复投递
- stream-json 用 pipe 会 EOF 退出，用 socketpair 才能保持存活
- session .jsonl 里用户消息 type 是 "user" 不是 "human"
- Discord interaction 需 3 秒内响应，耗时操作先 defer()
- bot_simple.py 已合并到 bot.py 并删除
- `--append-system-prompt` 只对新 session 生效，resume 旧 session 不会重新注入
- Skill 自动触发机制只匹配用户消息，不匹配 system prompt，所以 discord-style 规则需要通过 Python 端读取 skill 文件后注入
- Discord 链接预览（OG embed）: 必须用 `--plain` 模式发链接，Embed 模式下链接不触发 OG 预览
- Discord OG 缓存: 同一 URL 的 OG 数据会被缓存，改了 OG 标签后需要加 `?v=N` 破缓存
- OG 预览图: 页面必须有 `og:image`（1200x630px），否则只显示文字卡片
- `send-to-discord.sh --plain "url"` 发纯文本链接，`send-to-discord.sh "内容" "标题"` 发 Embed
