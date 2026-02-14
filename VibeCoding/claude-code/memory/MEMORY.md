# Auto Memory

## Chris
- Google Cloud AI Infra 专家，TPU/GPU 训练和推理
- 中文沟通，技术术语保留英文
- 直接给结论，不啰嗦
- 称呼 Claude Code 为 "CC"
- 通过 Discord 语音/文字 与 CC 交互
- 关注安全性（bot 鉴权、prompt injection）
- **通信偏好**: 能秒回直接回；需要时间先说"收到"，完成后用 `send-to-discord.sh` 异步通知

## Topic 文件索引
详细信息按需读取 `~/.claude/projects/-home-chrisya/memory/` 下的 topic 文件：

| 文件 | 内容 |
|------|------|
| `discord-bot.md` | Bot 配置、命令、session 管理、调试 |
| `gcp-infra.md` | MIG、GKE、项目、zone、SSH 配置 |
| `tools-skills.md` | sglang/vllm/Ray 等工具和自定义 skill |
| `architecture.md` | CC session 机制、auto memory 架构 |
| `debugging.md` | 踩坑记录、调试经验、常见问题解法 |
| `vector-memory.md` | 语义记忆系统方案、向量数据库选型 |

## 关键路径
- CC settings: `~/.claude/settings.json`
- Skills: `~/.claude/skills/` → symlink → `~/gpu-tpu-pedia/VibeCoding/claude-code/skills/`（git repo, push 到 github.com/yangwhale/gpu-tpu-pedia）
- CC Pages: `https://cc.higcp.com/`，web root: `/var/www/cc/`

## 近期活跃事项
- SSH alias b1 → B200 Spot 实例 (10.8.0.32)，Spot IP 可能变
- Whisper 模型从 small 切到 medium 测试中
- **语义记忆系统已完成**: Mem0 + Vertex AI Vector Search + Gemini 3 Flash Preview，详见 `vector-memory.md`
