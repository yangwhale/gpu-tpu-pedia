# Claude Code 架构知识

## Session 机制
- Session 文件: `~/.claude/projects/<路径编码>/<session-id>.jsonl`
- Session 索引: `~/.claude/projects/<路径编码>/sessions-index.json`
- `--resume <session-id>`: 按 session ID 精确恢复
- `--continue`: 接全局最后一次对话（不可靠，不推荐）
- Session 没有原生命名功能，只有 UUID

## Session Index 字段
sessionId, fullPath, fileMtime, firstPrompt, summary, messageCount, created, modified, gitBranch, projectPath, isSidechain

## Auto Memory
- `MEMORY.md` 前 200 行每次 session 自动注入 system prompt
- 超过 200 行会被截断
- 详细内容拆 topic 文件，CC 按需读取
- 路径: `~/.claude/projects/<路径编码>/memory/`

## 消息安全
- Discord bot 消息经 Discord 中心服务器转发
- `message.author.id` 由平台验证，可信

## 设计哲学（参考 OpenClaw）
- **Good Taste**: 用数据结构消除特殊情况，而非 if/else 补丁（Linus Torvalds）
- **文件系统 + 向量混合记忆**: markdown 做结构化长期知识，Mem0 做语义检索，各取所长
- **自描述 Skill**: SKILL.md 与代码同目录，强制绑定文档和实现
- **协议归一化**: 多平台通知（Discord/飞书/微信）应抽象统一接口，减少 if/else
- **Lane 隔离**: Agent Teams 的 lead + teammates 对应 Main Lane + Sub Lane
- **未实现的 Cron/Heartbeat**: 从被动工具到主动伙伴的关键缺口，可用 cron + Discord bot 组合实现
