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
4. **STT engine** — gemini (recommended) / chirp2 / whisper:medium (default: gemini)
5. **GCP Project** — for Gemini/Chirp2 STT (default: use gcloud config)

If no Bot Token, guide user through Discord Developer Portal setup:
- Create Application > Bot > Reset Token > enable all 3 Intents (Presence, Server Members, **Message Content**)
- OAuth2 > Scopes: bot, applications.commands > Permissions: Send Messages, Read Message History, Embed Links, Attach Files, View Channels
- **Disable User Install** in Installation settings (security)

## Installation

```bash
pip install py-cord --break-system-packages
sudo apt-get install -y ffmpeg jq
pip install google-genai --break-system-packages   # Gemini STT (recommended)
pip install openai-whisper --break-system-packages  # Whisper fallback
```

## Deploy Bot

1. Read `scripts/bot_template.py` — this is the complete, production-ready bot script
2. Copy to `~/.claude/discord-bot/bot.py`
3. Create `~/.claude/discord-bot/.env` with config:
   ```
   DISCORD_BOT_TOKEN=<bot_token>
   ALLOWED_USER_IDS=<user_discord_id>
   AUTO_RESPOND_CHANNELS=<channel_id>
   STT_ENGINE=gemini
   GOOGLE_CLOUD_PROJECT=<gcp_project>
   CHIRP2_LOCATION=us-central1
   ```
4. Copy `scripts/send-to-discord.sh` to `~/.claude/scripts/send-to-discord.sh` and set BOT_TOKEN/CHANNEL_ID
5. `chmod +x ~/.claude/scripts/send-to-discord.sh`

## STT Engine Configuration

Three engines available, configured via `STT_ENGINE` in `.env`:

- **`gemini`** (recommended) — Uses Gemini multimodal LLM for transcription. Understands semantics, corrects homophones, handles tech terms. Requires `google-genai` SDK and GCP project with Vertex AI API enabled.
  - Current best: `gemini-3-flash-preview` with `thinking_level=MINIMAL` (~3.4s, excellent quality)
  - Alternative: `gemini-2.0-flash` (~3s, excellent quality, more regions)
  - Budget option: `gemini-2.5-flash-lite` (~2.1s, good quality)
- **`chirp2`** — Google Cloud Speech-to-Text V2 (Chirp 2). Pure ASR, no semantic understanding. Poor with homophones.
- **`whisper:medium`** — Local OpenAI Whisper. No network dependency, ~2-3s, ok quality.

Fallback chain: Gemini → Chirp 2 → Whisper (automatic on failure)

**Important**: Gemini 3 Flash Preview only available in `global` region. Other Gemini models work in `us-central1`.

## Start Bot

```bash
mkdir -p ~/.claude/discord-bot
# Copy wrapper script
cp scripts/run.sh ~/.claude/discord-bot/run.sh
chmod +x ~/.claude/discord-bot/run.sh
# Start in tmux (survives shell disconnect, wrapper auto-restarts on /restart)
tmux new-session -d -s discord-bot 'bash ~/.claude/discord-bot/run.sh'
```

Verify: `tmux ls` and `tail -5 ~/.claude/discord-bot/bot.log`

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
| `/restart` | Graceful restart (exit code 42 → wrapper auto-restarts) |

## Key Features

- **Persistent Claude process** per user via `socket.socketpair()` — full interactive mode
- **Session history** — `/end` archives, `/sessions` shows dropdown to switch back
- **Gemini STT** — multimodal voice transcription with semantic understanding (understands context, corrects homophones)
- **STT fallback chain** — Gemini → Chirp 2 → Whisper (auto-fallback on failure)
- **Smart message splitting** — split at newlines for Discord's 2000 char limit
- **Process auto-restart** — if Claude dies, recreate transparently on next message
- **Graceful restart** — `/restart` command triggers exit code 42, wrapper script auto-restarts
- **FD isolation** — `subprocess.Popen(close_fds=True)` prevents Claude from inheriting Discord websocket FD
- **Safety prompt** — system prompt forbids Claude subprocess from killing/restarting bot process

## References

- **Architecture details**: See [references/architecture.md](references/architecture.md) for socketpair protocol, message format, session management internals, and security model
- **Troubleshooting**: See [references/troubleshooting.md](references/troubleshooting.md) for common issues and fixes

## Files Created

```
~/.claude/discord-bot/
├── bot.py           # Main bot script
├── run.sh           # Wrapper script (auto-restart on exit code 42)
├── bot.log          # Runtime logs (tail -f)
└── sessions.json    # Per-user session mapping (auto-created)

~/.claude/scripts/
└── send-to-discord.sh  # Claude Code → Discord messaging
```
