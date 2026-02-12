---
name: discord-bot-setup
description: Deploy a Discord Bot that connects to Claude Code via persistent processes using Unix socketpair + stream-json (same mechanism as VSCode extension). Each user gets their own long-running Claude process with full interactive mode support (auto memory, CLAUDE.md, skills). Includes Whisper voice transcription, user whitelist, session history with dropdown switcher, and slash commands. Use when the user says "帮我建一个 Discord Bot", "setup discord bot", "搭建 Discord 机器人", "discord bot 设置", or "部署 discord bot".
---

# Discord Bot Setup

## Prerequisites

Collect from user via AskUserQuestion:
1. **Bot Token** — from Discord Developer Portal
2. **Discord User ID** — for whitelist (Developer Mode > right-click avatar > Copy User ID)
3. **Auto-respond Channel ID** — (optional) channel where bot responds without @mention
4. **Whisper model** — tiny/base/small/medium/large (default: medium)

If no Bot Token, guide user through Discord Developer Portal setup:
- Create Application > Bot > Reset Token > enable all 3 Intents (Presence, Server Members, **Message Content**)
- OAuth2 > Scopes: bot, applications.commands > Permissions: Send Messages, Read Message History, Embed Links, Attach Files, View Channels
- **Disable User Install** in Installation settings (security)

## Installation

```bash
pip install py-cord --break-system-packages
sudo apt-get install -y ffmpeg jq
pip install openai-whisper --break-system-packages
```

## Deploy Bot

1. Read `scripts/bot_template.py` — this is the complete, production-ready bot script
2. Copy to `~/.claude/discord-bot/bot.py`
3. Replace placeholder values with user-provided config:
   - `YOUR_TOKEN_HERE` → bot token
   - `ALLOWED_USER_IDS = set()` → `{user_discord_id}`
   - `AUTO_RESPOND_CHANNELS = set()` → `{channel_id}` if provided
   - `WHISPER_MODEL` default → user's choice
4. Copy `scripts/send-to-discord.sh` to `~/.claude/scripts/send-to-discord.sh` and set BOT_TOKEN/CHANNEL_ID
5. `chmod +x ~/.claude/scripts/send-to-discord.sh`

## Start Bot

```bash
mkdir -p ~/.claude/discord-bot
nohup python3 ~/.claude/discord-bot/bot.py > /dev/null 2>&1 &
```

Verify: `ps aux | grep bot.py | grep -v grep` and `tail -5 ~/.claude/discord-bot/bot.log`

**Critical**: Only one bot process must run. Multiple processes = duplicate messages.

## Memory Setup

For auto memory to work across sessions:
1. Ensure `~/.claude/CLAUDE.md` exists with global preferences
2. Create `~/.claude/projects/-home-<user>/memory/MEMORY.md` as memory index
3. Bot already passes `CLAUDE_CODE_DISABLE_AUTO_MEMORY=0` to Claude subprocess

## Slash Commands

| Command | Function |
|---------|----------|
| `/status` | Bot status, active Claude processes, Whisper model |
| `/end` | Archive current session, stop Claude process |
| `/sessions` | List history with summaries + dropdown switcher |

## Key Features

- **Persistent Claude process** per user via `socket.socketpair()` — full interactive mode
- **Session history** — `/end` archives, `/sessions` shows dropdown to switch back
- **Whisper voice** — auto-detect audio attachments, transcribe, feed to Claude
- **Smart message splitting** — split at newlines for Discord's 2000 char limit
- **Process auto-restart** — if Claude dies, recreate transparently on next message

## References

- **Architecture details**: See [references/architecture.md](references/architecture.md) for socketpair protocol, message format, session management internals, and security model
- **Troubleshooting**: See [references/troubleshooting.md](references/troubleshooting.md) for common issues and fixes

## Files Created

```
~/.claude/discord-bot/
├── bot.py           # Main bot script
├── bot.log          # Runtime logs (tail -f)
└── sessions.json    # Per-user session mapping (auto-created)

~/.claude/scripts/
└── send-to-discord.sh  # Claude Code → Discord messaging
```
