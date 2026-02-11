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
```

Verify: `python3 -c "import discord; print(discord.__version__)"`

### Step 3: Create Bot Script

Write the bot script to `~/.claude/discord-bot/bot_simple.py`:

```python
#!/usr/bin/env python3
"""Discord Bot - Claude Code Remote Control"""
import discord, asyncio, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(Path.home() / ".claude/discord-bot/bot.log")])
log = logging.getLogger("bot")

CLAUDE_BIN = str(Path.home() / ".local/bin/claude")
TOKEN = "USER_PROVIDED_TOKEN"

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
    log.info(f"MSG: {message.author} #{message.channel}: '{message.content}' dm={is_dm} mention={is_mentioned}")
    if not is_dm and not is_mentioned:
        return
    content = message.content.replace(f"<@{client.user.id}>", "").strip()
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
- DM or @mention to trigger

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

### Files Created

```
~/.claude/discord-bot/
├── bot_simple.py    # Main bot script
└── bot.log          # Runtime logs
```

### Security Notes

- `--dangerously-skip-permissions` means anyone who can message the Bot can execute commands on the machine
- For production: add user ID whitelist in the script (ALLOWED_USER_IDS set)
- Keep Bot Token secret — never commit to git
- Use private channels or DM only
