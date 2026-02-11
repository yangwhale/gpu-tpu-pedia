# Discord Bot Setup - Claude Code Remote Control

Setup a Discord Bot that forwards messages to Claude Code (`claude -p`), enabling remote control from phone/Discord.

## Trigger

When the user says:
- "帮我建一个 Discord Bot"
- "setup discord bot"
- "搭建 Discord 机器人"
- "discord bot 设置"

## Workflow

### Step 1: Collect Info

Ask the user for these via AskUserQuestion:

1. **Bot Token** — from Discord Developer Portal (guide them if needed)
2. **Whether they have an existing Discord server** — or need to create one

If user doesn't have a Bot Token yet, guide them:

```
1. Go to https://discord.com/developers/applications
2. Click "New Application" → name it (e.g. "Claude Code Bot") → Create
3. Left menu: Bot → click "Reset Token" → copy the token
4. Same page, enable ALL three:
   - Presence Intent ✓
   - Server Members Intent ✓
   - Message Content Intent ✓ (CRITICAL - won't receive messages without this)
5. Left menu: OAuth2 → URL Generator:
   - Scopes: bot, applications.commands
   - Bot Permissions: Send Messages, Read Message History, Embed Links, Attach Files, View Channels
   - Open the generated URL to invite Bot to your server
6. If the channel is PRIVATE: add Bot to channel members manually
```

### Step 2: Install Dependencies

```bash
pip install py-cord --break-system-packages

# 语音转文字支持（可选，支持 Discord 语音消息识别）
sudo apt-get install -y ffmpeg
pip install openai-whisper --break-system-packages
```

Verify:
```bash
python3 -c "import discord; print(discord.__version__)"
python3 -c "import whisper; print(whisper.__version__)"  # 可选
which ffmpeg  # 可选
```

### Step 3: Create Bot Script

Write the bot script to `~/.claude/discord-bot/bot_simple.py`:

```python
#!/usr/bin/env python3
"""Minimal Discord Bot - uses discord.Client for reliable on_message"""
import discord
import asyncio
import logging
import os
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(Path.home() / ".claude/discord-bot/bot.log")])
log = logging.getLogger("bot")

CLAUDE_BIN = str(Path.home() / ".local/bin/claude")
TOKEN = "USER_PROVIDED_TOKEN"

# Whisper 语音转文字（懒加载，首次使用时加载模型）
# 可选: tiny, base, small, medium, large
# small 推荐：中英文识别好，CPU 上速度合理
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

_whisper_model = None
def transcribe_audio(file_path: str) -> str:
    """用 Whisper 模型转写音频文件，自动检测语言"""
    global _whisper_model
    try:
        import whisper
        if _whisper_model is None:
            log.info(f"Loading Whisper {WHISPER_MODEL} model (first time)...")
            _whisper_model = whisper.load_model(WHISPER_MODEL)
            log.info("Whisper model loaded.")
        result = _whisper_model.transcribe(file_path, language=None)
        text = result.get("text", "").strip()
        lang = result.get("language", "unknown")
        log.info(f"Transcribed ({lang}): {text[:100]}...")
        return text
    except ImportError:
        log.warning("Whisper not installed, skipping voice transcription")
        return ""
    except Exception as e:
        log.warning(f"Whisper transcription failed: {e}")
        return ""

# 在这些频道中不需要 @mention，直接响应所有消息
# 部署后通过 Bot 查询频道 ID 填入，或留空 set() 表示全部需要 @mention
AUTO_RESPOND_CHANNELS = set()  # e.g. {1471088850712531055}

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
sessions = set()

@client.event
async def on_ready():
    log.info(f"READY: {client.user} | Guilds: {[g.name for g in client.guilds]}")

@client.event
async def on_message(message):
    if message.author == client.user or message.author.bot:
        return
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user in message.mentions
    is_auto_channel = getattr(message.channel, 'id', None) in AUTO_RESPOND_CHANNELS
    log.info(f"MSG: {message.author} #{message.channel}: '{message.content}' dm={is_dm} mention={is_mentioned} auto={is_auto_channel}")
    if not is_dm and not is_mentioned and not is_auto_channel:
        return
    content = message.content.replace(f"<@{client.user.id}>", "").strip()
    # 处理附件（图片/文件/语音）- 用 discord.py 自带的 save() 避免 CDN 403
    downloaded_files = []
    voice_texts = []
    for att in message.attachments:
        try:
            suffix = Path(att.filename).suffix or ".bin"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp", prefix="discord_")
            await att.save(tmp.name)
            ctype = att.content_type or ""
            log.info(f"Downloaded attachment: {att.filename} ({ctype}) -> {tmp.name}")
            # 检测语音消息并转文字
            is_voice = (
                ctype.startswith("audio/")
                or suffix.lower() in (".ogg", ".mp3", ".wav", ".m4a", ".webm", ".flac")
                or (message.flags.value & 8192)  # IS_VOICE_MESSAGE flag
            )
            if is_voice:
                log.info(f"Voice message detected, transcribing: {att.filename}")
                transcribed = await asyncio.to_thread(transcribe_audio, tmp.name)
                if transcribed:
                    voice_texts.append(transcribed)
                    await message.reply(f"🎤 语音识别: {transcribed}")
                else:
                    await message.reply("⚠️ 语音识别失败，无法转写")
                os.unlink(tmp.name)
            else:
                downloaded_files.append((tmp.name, att.filename, ctype))
        except Exception as e:
            log.warning(f"Failed to download {att.filename}: {e}")
    # 合并语音转写文本
    if voice_texts:
        voice_content = "\n".join(voice_texts)
        content = f"{voice_content}\n{content}" if content else voice_content
    # 合并普通附件引用
    if downloaded_files:
        file_refs = "\n".join(
            f"[Attached file: {fname} (saved at {path})]"
            for path, fname, _ in downloaded_files
        )
        content = f"{file_refs}\n{content}" if content else file_refs
    if not content:
        await message.reply("Send me a message!")
        return
    if content.lower() in ("exit", "quit", "bye", "退出"):
        sessions.discard(message.author.id)
        await message.reply("Session ended.")
        return
    cont = message.author.id in sessions
    async with message.channel.typing():
        cmd = [CLAUDE_BIN, "-p", content, "--dangerously-skip-permissions"]
        if cont:
            cmd.append("--continue")
        log.info(f"Claude: '{content[:80]}' continue={cont}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(Path.home()))
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            result = stdout.decode(errors="replace").strip() or stderr.decode(errors="replace").strip() or "(empty)"
        except asyncio.TimeoutError:
            result = "[Timeout 5min]"
        except Exception as e:
            result = f"[Error: {e}]"
    sessions.add(message.author.id)
    for i in range(0, len(result), 1990):
        await message.channel.send(result[i:i+1990])
        if i + 1990 < len(result):
            await asyncio.sleep(0.5)

client.run(TOKEN)
```

Key points:
- Uses `discord.Client` (NOT `discord.Bot`) — more reliable for `on_message`
- `--dangerously-skip-permissions` for autonomous execution
- `--continue` for multi-turn conversations
- Auto-splits messages >2000 chars (Discord limit)
- DM, @mention, or auto-respond channel to trigger
- Attachment support: downloads images/files via `att.save()` (avoids Discord CDN 403)
- `AUTO_RESPOND_CHANNELS`: add channel IDs to skip @mention requirement
- Voice message transcription via Whisper (lazy-loaded, auto language detection)

#### Voice Message Notes

- Whisper 模型懒加载：首次收到语音时加载到内存（small ~460MB），之后常驻
- 用 `asyncio.to_thread()` 在线程池转写，不阻塞 Bot 事件循环
- 语音检测三重判断：`content_type` 前缀 / 文件扩展名 / `IS_VOICE_MESSAGE` flag (8192)
- 转写后先回复 `🎤 语音识别: ...` 让用户确认，再作为指令传给 Claude Code
- 如果 Whisper 未安装，语音功能自动跳过（graceful degradation）
- 模型选择：`base` 速度快但中文一般，`small` 中文好推荐使用，`medium` 最准但 CPU 上较慢
- **重要**：必须安装 `ffmpeg`，Whisper 依赖它解码音频格式

### Step 4: Start Bot

```bash
# Start in background
setsid python3 ~/.claude/discord-bot/bot_simple.py < /dev/null >> ~/.claude/discord-bot/bot.log 2>&1 &
disown
```

Verify:
```bash
# Check process
ps aux | grep bot_simple | grep -v grep

# Check logs
tail -5 ~/.claude/discord-bot/bot.log
# Should show: "READY: BotName | Guilds: ['ServerName']"
```

### Step 5: Test

1. **DM test first** — DM the Bot directly (no channel permission issues)
2. **Channel test** — @BotName in the channel (make sure to @ the BOT, not a role with same name)
3. **Multi-turn** — send follow-up messages (auto `--continue`)
4. **Exit** — send "exit" to end session

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Bot online but no response | Message Content Intent off | Developer Portal → Bot → enable all 3 intents |
| Works in DM but not channel | Private channel | Add Bot to channel members |
| @mention not working | @'ing a Role not the Bot | Select the Bot from MEMBERS list, not ROLES |
| Bot process dies | Background process killed | Use `setsid` + `disown`, or systemd |
| "claude not found" | Wrong path | Check `which claude` and update CLAUDE_BIN |
| Voice message: "⚠️ 语音识别失败" | ffmpeg missing or Whisper error | `sudo apt install ffmpeg` and check Whisper install |
| Voice message: no response | Whisper not installed | `pip install openai-whisper --break-system-packages` |
| First voice msg slow | Model loading (~460MB) | Normal, subsequent messages will be fast |

### Common Permission Issues Checklist

When Bot doesn't respond in a channel, check ALL of these:

1. **Developer Portal**: Bot → Message Content Intent = ON
2. **Server**: Bot role has View Channels + Send Messages + Read Message History
3. **Channel**: If private, Bot must be added as member
4. **@mention**: Must @ the Bot user (MEMBERS), NOT a role with similar name
5. **Invite URL**: Must include `bot` scope + proper permissions integer

### Invite URL Template

```
https://discord.com/api/oauth2/authorize?client_id=BOT_CLIENT_ID&permissions=2147601408&scope=bot%20applications.commands
```

Permission integer 2147601408 = View Channels + Send Messages + Embed Links + Attach Files + Read Message History + Use Slash Commands

### Step 6: Setup Notification Channel

Bot 部署完成后，自动配置通知发送能力，使 Claude Code 可以主动发消息到 Discord：

```bash
# 创建发送脚本（如果不存在）
# 脚本位置: ~/.claude/scripts/send-to-discord.sh
# 需要配置: BOT_TOKEN 和 CHANNEL_ID
```

配置完成后，用户可以通过说"discord通知我"、"ds通知我"等触发 `discord-report` 技能来发送报告到 Discord。

详见 `discord-report` 技能文档。

### Files Created

```
~/.claude/discord-bot/
├── bot_simple.py    # Main bot script (接收消息 → Claude Code)
└── bot.log          # Runtime logs

~/.claude/scripts/
└── send-to-discord.sh  # 发送脚本 (Claude Code → Discord)
```

### Security Notes

- `--dangerously-skip-permissions` means anyone who can message the Bot can execute commands on the machine
- For production: add user ID whitelist in the script (ALLOWED_USER_IDS set)
- Keep Bot Token secret — never commit to git
- Use private channels or DM only
