#!/usr/bin/env python3
"""
Claude Code Discord Bot

架构: 每个用户对应一个持久的 Claude Code 进程，通过 Unix socketpair + stream-json 通信。
Claude 以完整交互模式运行，支持 auto memory、CLAUDE.md、skills 等全部功能。

功能:
1. Discord 消息 -> 持久 Claude 进程 -> 结果发回 Discord
2. 每用户独立 Claude 进程，互不干扰
3. Whisper 语音转文字
4. 用户白名单鉴权
5. 自动响应频道 (免 @mention)
6. 斜杠命令 (/status, /end)

用法:
    python3 bot.py                 # 前台运行
    python3 bot.py --daemon        # 后台运行
"""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import discord
from discord.ext import commands

# ============================================================================
# 配置
# ============================================================================

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "YOUR_TOKEN_HERE")

# 安全: 只响应这些 Discord 用户 ID
ALLOWED_USER_IDS = {1074613327805829190}  # yangwhale (Chris)

# Claude Code
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", str(Path.home() / ".local/bin/claude"))
WORK_DIR = os.environ.get("CLAUDE_WORK_DIR", str(Path.home()))
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "600"))
RESTART_EXIT_CODE = 42  # wrapper 脚本检测到此 exit code 时自动重启

# 语音转文字引擎: "chirp2" (Cloud STT, 默认) 或 "whisper:medium" / "whisper:large-v3" 等
STT_ENGINE = os.environ.get("STT_ENGINE", "chirp2")
# Chirp 2 配置
CHIRP2_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", os.environ.get("GCLOUD_PROJECT", ""))
CHIRP2_LOCATION = os.environ.get("CHIRP2_LOCATION", "us-central1")
# 向后兼容: WHISPER_MODEL 环境变量仍可用于覆盖
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")

# 在这些频道中不需要 @mention，直接响应所有消息
AUTO_RESPOND_CHANNELS = {1471088850712531055}  # 🥶-claude-code

# Discord 输出风格 (从 skill 文件动态加载，保持单一内容源)
DISCORD_STYLE_SKILL = Path.home() / ".claude/skills/discord-style/SKILL.md"

def load_discord_style() -> str:
    """从 discord-style skill 文件加载格式规则"""
    try:
        content = DISCORD_STYLE_SKILL.read_text()
        # 跳过 YAML frontmatter（--- ... ---）
        parts = content.split("---", 2)
        body = parts[2].strip() if len(parts) >= 3 else content
        return f"你正在通过 Discord 频道与用户交互。\n\n{body}"
    except FileNotFoundError:
        log.warning(f"Discord style skill not found: {DISCORD_STYLE_SKILL}")
        return "你正在通过 Discord 频道与用户交互，请用简短对话式风格回复，不要用表格。"

DISCORD_SYSTEM_PROMPT = load_discord_style() + "\n\n" + (
    "CRITICAL SAFETY RULE: You are running as a child process of the Discord bot. "
    "NEVER kill, restart, or stop the bot process (bot.py). NEVER run commands like "
    "'kill', 'pkill', 'killall' targeting bot.py or the bot's PID. "
    "If you modify bot.py and it needs a restart, tell the user to run /restart in Discord. "
    "Killing the bot process will terminate YOUR OWN process and disconnect the user."
)

# 日志
LOG_FILE = Path.home() / ".claude/discord-bot/bot.log"

# ============================================================================
# 日志设置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("claude-bot")

# ============================================================================
# 语音转文字 (Chirp 2 默认, Whisper 可切换)
# ============================================================================

_whisper_model = None


def transcribe_with_whisper(file_path: str) -> str:
    """用本地 Whisper 模型转写音频文件"""
    global _whisper_model
    import whisper
    if _whisper_model is None:
        model_name = STT_ENGINE.split(":", 1)[1] if ":" in STT_ENGINE else WHISPER_MODEL
        log.info(f"Loading Whisper {model_name} model (first time)...")
        _whisper_model = whisper.load_model(model_name)
        log.info("Whisper model loaded.")
    result = _whisper_model.transcribe(file_path, language=None)
    text = result.get("text", "").strip()
    lang = result.get("language", "unknown")
    log.info(f"Whisper transcribed ({lang}): {text[:100]}...")
    return text


def transcribe_with_chirp2(file_path: str) -> str:
    """用 Google Cloud Speech-to-Text v2 (Chirp 2) 转写音频文件"""
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    client = SpeechClient(client_options={"api_endpoint": f"{CHIRP2_LOCATION}-speech.googleapis.com"})

    with open(file_path, "rb") as f:
        audio_content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["cmn-Hans-CN", "en-US"],
        model="chirp_2",
        denoiser_config=cloud_speech.DenoiserConfig(
            denoise_audio=True,
            snr_threshold=10.0,  # high sensitivity
        ),
        adaptation=cloud_speech.SpeechAdaptation(
            phrase_sets=[
                cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=cloud_speech.PhraseSet(phrases=[
                        {"value": "Claude Code"},
                        {"value": "Chirp"},
                        {"value": "Whisper"},
                        {"value": "TPU"},
                        {"value": "GPU"},
                        {"value": "B200"},
                        {"value": "H100"},
                        {"value": "A100"},
                        {"value": "SGLang"},
                        {"value": "vLLM"},
                        {"value": "GCP"},
                        {"value": "Discord"},
                        {"value": "Gemini"},
                        {"value": "sglang"},
                        {"value": "MIG"},
                        {"value": "GKE"},
                        {"value": "Spot"},
                        {"value": "HBM"},
                    ])
                )
            ]
        ),
    )
    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{CHIRP2_PROJECT}/locations/{CHIRP2_LOCATION}/recognizers/_",
        config=config,
        content=audio_content,
    )

    response = client.recognize(request=request)
    texts = []
    for result in response.results:
        if result.alternatives:
            texts.append(result.alternatives[0].transcript)
    text = " ".join(texts).strip()
    log.info(f"Chirp 2 transcribed: {text[:100]}...")
    return text


def transcribe_audio(file_path: str) -> str:
    """根据 STT_ENGINE 配置选择转写引擎"""
    try:
        if STT_ENGINE.startswith("whisper"):
            return transcribe_with_whisper(file_path)
        else:
            return transcribe_with_chirp2(file_path)
    except Exception as e:
        log.warning(f"Transcription failed ({STT_ENGINE}): {e}")
        # Chirp 2 失败时 fallback 到 Whisper
        if not STT_ENGINE.startswith("whisper"):
            log.info("Falling back to Whisper...")
            try:
                return transcribe_with_whisper(file_path)
            except Exception as e2:
                log.warning(f"Whisper fallback also failed: {e2}")
        return ""

# ============================================================================
# 持久 Claude 进程管理
# ============================================================================

class ClaudeSession:
    """管理一个持久的 Claude Code 进程，通过 socketpair 通信"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id
        self.proc = None
        self.sock_in = None   # bot -> claude (stdin)
        self.sock_out = None  # claude -> bot (stdout)
        self._lock = asyncio.Lock()

    async def start(self, _retry=False):
        """启动 Claude 持久进程"""
        parent_stdin, child_stdin = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        parent_stdout, child_stdout = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        cmd = [
            CLAUDE_BIN,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--permission-prompt-tool", "stdio",
            "--append-system-prompt", DISCORD_SYSTEM_PROMPT,
        ]
        if self.session_id:
            cmd.extend(["--resume", self.session_id])

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # 允许嵌套启动
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "0"

        self._stderr_path = tempfile.mktemp(prefix="claude_stderr_", suffix=".log")
        stderr_fd = os.open(self._stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self.proc = subprocess.Popen(
            cmd,
            stdin=child_stdin.fileno(),
            stdout=child_stdout.fileno(),
            stderr=stderr_fd,
            cwd=WORK_DIR,
            close_fds=True,
            env=env,
        )
        os.close(stderr_fd)
        child_stdin.close()
        child_stdout.close()
        self.sock_in = parent_stdin
        self.sock_out = parent_stdout
        self.sock_out.setblocking(False)

        # 等初始化并消费初始消息
        await self._drain(timeout=10)

        if not self.is_alive():
            # 读取 stderr 诊断信息
            stderr_content = ""
            try:
                with open(self._stderr_path) as f:
                    stderr_content = f.read().strip()[-500:]  # 最后 500 字符
            except Exception:
                pass
            if stderr_content:
                log.error(f"Claude stderr (PID={self.proc.pid}): {stderr_content}")
            if _retry:
                raise RuntimeError(f"Claude process failed to start (PID={self.proc.pid})")
            log.warning(f"Claude process died during startup (PID={self.proc.pid}), retrying without resume")
            self.session_id = None
            return await self.start(_retry=True)

        log.info(f"Claude process started: PID={self.proc.pid} session={self.session_id or 'new'}")

    async def _drain(self, timeout: float = 10):
        """异步消费 socket 中所有待读数据，避免阻塞事件循环"""
        deadline = asyncio.get_event_loop().time() + timeout
        idle_count = 0
        while asyncio.get_event_loop().time() < deadline:
            try:
                data = self.sock_out.recv(65536)
                if not data:
                    break  # socket closed
                idle_count = 0
            except BlockingIOError:
                idle_count += 1
                if idle_count >= 6:  # 连续 3 秒无数据，初始化完成
                    break
                await asyncio.sleep(0.5)
            except Exception:
                break

    def is_alive(self) -> bool:
        if self.proc is None:
            return False
        return self.proc.poll() is None

    async def send(self, text: str) -> str:
        """发送消息并等待完整回复"""
        async with self._lock:
            if not self.is_alive():
                await self.start()

            msg = json.dumps({
                "type": "user",
                "message": {"role": "user", "content": text}
            }) + "\n"
            self.sock_in.sendall(msg.encode())

            # 读取响应直到收到 result
            buf = b""
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < CLAUDE_TIMEOUT:
                await asyncio.sleep(0.3)
                try:
                    chunk = self.sock_out.recv(65536)
                    if not chunk:
                        # 进程退出了，标记需要重启
                        log.warning("Claude process exited unexpectedly")
                        self.proc = None
                        return "[Error] Claude process exited"
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            d = json.loads(line.decode(errors="replace"))
                        except json.JSONDecodeError:
                            continue
                        if d.get("type") == "result":
                            self.session_id = d.get("session_id", self.session_id)
                            return d.get("result", "") or "(empty output)"
                except BlockingIOError:
                    continue

            return f"[Timeout] Claude Code exceeded {CLAUDE_TIMEOUT}s limit"

    async def stop(self):
        """停止 Claude 进程"""
        if self.sock_in:
            self.sock_in.close()
        if self.sock_out:
            self.sock_out.close()
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        log.info(f"Claude session stopped: {self.session_id}")


# 每用户一个持久 Claude 进程
claude_sessions: dict[str, ClaudeSession] = {}  # user_key -> ClaudeSession
_session_locks: dict[str, asyncio.Lock] = {}  # 防止 get_claude_session 重入

# 持久化: active = 当前活跃 session, history = 历史 session 列表
SESSIONS_FILE = Path.home() / ".claude/discord-bot/sessions.json"

def load_sessions_data() -> dict:
    try:
        data = json.loads(SESSIONS_FILE.read_text())
        # 兼容旧格式 (flat dict of user_key -> session_id)
        if data and not any(isinstance(v, dict) for v in data.values()):
            return {k: {"active": v, "history": []} for k, v in data.items()}
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_sessions_data():
    data = load_sessions_data()
    for k, s in claude_sessions.items():
        if s.session_id:
            if k not in data:
                data[k] = {"active": None, "history": []}
            data[k]["active"] = s.session_id
    SESSIONS_FILE.write_text(json.dumps(data, indent=2))

def archive_session(user_key: str, session_id: str):
    """将 session 存入历史"""
    data = load_sessions_data()
    if user_key not in data:
        data[user_key] = {"active": None, "history": []}
    if session_id and session_id not in data[user_key]["history"]:
        data[user_key]["history"].insert(0, session_id)
        data[user_key]["history"] = data[user_key]["history"][:20]  # 保留最近 20 个
    data[user_key]["active"] = None
    SESSIONS_FILE.write_text(json.dumps(data, indent=2))

def get_user_history(user_key: str) -> list[str]:
    data = load_sessions_data()
    return data.get(user_key, {}).get("history", [])


def get_session_summary(session_id: str) -> str:
    """从 session .jsonl 文件读取第一条用户消息作为摘要"""
    session_file = Path.home() / f".claude/projects/-home-chrisya/{session_id}.jsonl"
    try:
        with open(session_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("type") in ("human", "user"):
                        # 提取文本内容
                        msg = d.get("message", {})
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") == "text"
                            )
                        text = content.strip()[:80]
                        return text if text else "(empty)"
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return "(no data)"


async def get_claude_session(user_key: str) -> ClaudeSession:
    """获取或创建用户的 Claude 持久进程"""
    if user_key not in _session_locks:
        _session_locks[user_key] = asyncio.Lock()
    async with _session_locks[user_key]:
        if user_key not in claude_sessions or not claude_sessions[user_key].is_alive():
            data = load_sessions_data()
            session_id = data.get(user_key, {}).get("active")
            session = ClaudeSession(session_id=session_id)
            await session.start()
            claude_sessions[user_key] = session
        return claude_sessions[user_key]


# ============================================================================
# Bot 设置
# ============================================================================

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Bot(intents=intents)

# ============================================================================
# 工具函数
# ============================================================================

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


async def send_long_message(channel, content: str):
    """发送长消息，自动按换行符分割"""
    if not content:
        return
    chunks = []
    while content:
        if len(content) <= 1990:
            chunks.append(content)
            break
        split_at = content.rfind("\n", 0, 1990)
        if split_at == -1:
            split_at = 1990
        chunks.append(content[:split_at])
        content = content[split_at:].lstrip("\n")
    for i, chunk in enumerate(chunks):
        await channel.send(chunk)
        if i < len(chunks) - 1:
            await asyncio.sleep(0.5)


async def process_attachments(message):
    """处理消息附件: 语音转文字 + 普通文件引用"""
    downloaded_files = []
    voice_texts = []
    for att in message.attachments:
        try:
            suffix = Path(att.filename).suffix or ".bin"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp", prefix="discord_")
            await att.save(tmp.name)
            ctype = att.content_type or ""
            log.info(f"Downloaded attachment: {att.filename} ({ctype}) -> {tmp.name}")
            is_voice = (
                ctype.startswith("audio/")
                or suffix.lower() in (".ogg", ".mp3", ".wav", ".m4a", ".webm", ".flac")
                or (message.flags.value & 8192)
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
    return voice_texts, downloaded_files


# ============================================================================
# Bot 事件
# ============================================================================

@bot.event
async def on_ready():
    log.info(f"Bot is ready: {bot.user} (ID: {bot.user.id})")
    log.info(f"Allowed users: {ALLOWED_USER_IDS or 'ALL'}")
    log.info(f"STT engine: {STT_ENGINE}")
    log.info(f"Auto-respond channels: {AUTO_RESPOND_CHANNELS}")


@bot.event
async def on_message(message):
    if message.author == bot.user or message.author.bot:
        return
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions
    is_auto_channel = getattr(message.channel, 'id', None) in AUTO_RESPOND_CHANNELS

    ch_name = f"DM:{message.author}" if is_dm else f"#{message.channel}"
    log.info(f"MSG: {message.author} {ch_name}: '{message.content}' dm={is_dm} mention={is_mentioned} auto={is_auto_channel}")

    if not is_dm and not is_mentioned and not is_auto_channel:
        return

    if not is_allowed(message.author.id):
        await message.reply("You are not authorized to use this bot.")
        log.warning(f"Unauthorized access attempt by {message.author} (ID: {message.author.id})")
        return

    content = message.content.replace(f"<@{bot.user.id}>", "").strip()

    # 处理附件
    voice_texts, downloaded_files = await process_attachments(message)
    if voice_texts:
        voice_content = "\n".join(voice_texts)
        content = f"{voice_content}\n{content}" if content else voice_content
    if downloaded_files:
        file_refs = "\n".join(
            f"[Attached file: {fname} (saved at {path})]"
            for path, fname, _ in downloaded_files
        )
        content = f"{file_refs}\n{content}" if content else file_refs

    if not content:
        await message.reply("Send me a message!")
        return

    # 退出命令: 停止 Claude 进程并归档 session
    if content.lower() in ("exit", "quit", "bye", "退出", "结束"):
        user_key = str(message.author.id)
        if user_key in claude_sessions:
            sid = claude_sessions[user_key].session_id
            await claude_sessions[user_key].stop()
            del claude_sessions[user_key]
            archive_session(user_key, sid)
        await message.reply("Session ended. Use /sessions to view history, /switch to resume a past session.")
        log.info(f"Session ended for {message.author}")
        return

    # 获取用户的持久 Claude 进程并发送消息
    user_key = str(message.author.id)
    try:
        async with message.channel.typing():
            session = await get_claude_session(user_key)
            result = await session.send(content)

        log.info(f"Claude reply ({len(result)} chars) session={session.session_id}: {result[:500]}")
        save_sessions_data()

        await send_long_message(message.channel, result)
    except Exception as e:
        log.error(f"Error handling message from {message.author}: {e}", exc_info=True)
        try:
            await message.reply(f"⚠️ Error: {e}")
        except Exception:
            pass


# ============================================================================
# 斜杠命令
# ============================================================================

@bot.slash_command(description="Check bot status")
async def status(ctx):
    """查看 Bot 状态"""
    alive = sum(1 for s in claude_sessions.values() if s.is_alive())
    embed = discord.Embed(title="Claude Code Bot Status", color=discord.Color.green())
    embed.add_field(name="Status", value="Online", inline=True)
    embed.add_field(name="Claude Processes", value=str(alive), inline=True)
    embed.add_field(name="STT Engine", value=STT_ENGINE, inline=True)
    embed.add_field(name="Claude Binary", value=CLAUDE_BIN, inline=False)
    embed.add_field(name="Work Dir", value=WORK_DIR, inline=False)
    embed.set_footer(text=f"Checked at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await ctx.respond(embed=embed)


@bot.slash_command(description="End current Claude Code session")
async def end(ctx):
    """结束当前会话并归档"""
    user_key = str(ctx.author.id)
    if user_key in claude_sessions:
        sid = claude_sessions[user_key].session_id
        await claude_sessions[user_key].stop()
        del claude_sessions[user_key]
        archive_session(user_key, sid)
        await ctx.respond("Session ended. Use /sessions to view history, /switch to resume.")
    else:
        await ctx.respond("No active session.")


_restart_requested = False

@bot.slash_command(description="Restart bot to apply code changes")
async def restart(ctx):
    """重启 bot 以应用代码修改"""
    global _restart_requested
    if not is_allowed(ctx.author.id):
        await ctx.respond("Not authorized.")
        return
    await ctx.respond("Restarting bot...")
    log.info(f"Restart requested by {ctx.author}")
    _restart_requested = True
    await bot.close()


class SessionSelect(discord.ui.Select):
    """Session 切换下拉菜单"""
    def __init__(self, user_key: str, options_list):
        super().__init__(placeholder="Select a session to switch to...", options=options_list)
        self.user_key = user_key

    async def callback(self, interaction):
        target_sid = self.values[0]
        await interaction.response.defer()

        # 归档当前 session
        if self.user_key in claude_sessions:
            old_sid = claude_sessions[self.user_key].session_id
            await claude_sessions[self.user_key].stop()
            del claude_sessions[self.user_key]
            archive_session(self.user_key, old_sid)

        # 启动目标 session
        session = ClaudeSession(session_id=target_sid)
        await session.start()
        claude_sessions[self.user_key] = session
        save_sessions_data()

        summary = get_session_summary(target_sid)
        await interaction.followup.send(
            content=f"Switched to session `{target_sid[:8]}...`\n> {summary}",
        )


@bot.slash_command(description="List and switch session history")
async def sessions(ctx):
    """列出历史 session 并提供下拉菜单切换"""
    user_key = str(ctx.author.id)
    history = get_user_history(user_key)
    data = load_sessions_data()
    active = data.get(user_key, {}).get("active")

    if not history and not active:
        await ctx.respond("No sessions found.")
        return

    lines = []
    if active:
        summary = get_session_summary(active)
        lines.append(f"**Active:** `{active[:8]}...` — {summary}")

    options = []
    for i, sid in enumerate(history[:25]):  # Discord 限制 25 个选项
        summary = get_session_summary(sid)
        label = f"{summary[:50]}" if len(summary) <= 50 else f"{summary[:47]}..."
        lines.append(f"{i+1}. `{sid[:8]}...` — {summary}")
        options.append(discord.SelectOption(
            label=label or f"Session {i+1}",
            description=f"{sid[:16]}...",
            value=sid,
        ))

    embed = discord.Embed(
        title="Session History",
        description="\n".join(lines),
        color=discord.Color.blue(),
    )

    if options:
        view = discord.ui.View(timeout=120)
        view.add_item(SessionSelect(user_key, options))
        await ctx.respond(embed=embed, view=view)
    else:
        await ctx.respond(embed=embed)


# ============================================================================
# 主入口
# ============================================================================

def main():
    if "--daemon" in sys.argv:
        pid = os.fork()
        if pid > 0:
            print(f"Bot started in background (PID: {pid})")
            print(f"Log: {LOG_FILE}")
            sys.exit(0)
        os.setsid()

    log.info("Starting Claude Code Discord Bot...")

    # 信号追踪
    def _signal_handler(signum, frame):
        log.warning(f"Received signal {signum} ({signal.Signals(signum).name})")
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _signal_handler)

    if not BOT_TOKEN or BOT_TOKEN == "YOUR_TOKEN_HERE":
        log.error("BOT_TOKEN not set! Set DISCORD_BOT_TOKEN environment variable.")
        sys.exit(1)

    try:
        log.info("Calling bot.run()...")
        bot.run(BOT_TOKEN)
        if _restart_requested:
            log.info("Restart requested, exiting with code 42")
            sys.exit(RESTART_EXIT_CODE)
        log.warning("bot.run() returned normally (unexpected)")
    except discord.LoginFailure:
        log.error("Invalid bot token!")
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Bot stopped by KeyboardInterrupt")
    except SystemExit as e:
        log.warning(f"Bot stopped by SystemExit: {e}")
        raise
    except Exception as e:
        log.error(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
