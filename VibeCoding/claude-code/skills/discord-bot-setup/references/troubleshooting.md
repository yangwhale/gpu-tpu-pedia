# Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Bot online, no response | Message Content Intent off | Developer Portal -> Bot -> enable all 3 intents |
| "Not authorized" | User ID not in whitelist | Add to `ALLOWED_USER_IDS` in bot.py |
| Duplicate messages | Multiple bot processes | `ps aux \| grep bot.py`, kill extras |
| Voice "语音识别失败" | ffmpeg missing | `sudo apt install ffmpeg` |
| First voice msg slow | Whisper model loading | Normal, subsequent messages fast |
| Bot dies on logout | Not daemonized | Use `--daemon` or `nohup` |
| "/sessions" interaction failed | Defer timeout | Ensure `interaction.response.defer()` before slow ops |
| Session summary "(no data)" | Wrong message type | Match both `"human"` and `"user"` types in `.jsonl` |
| stream-json process exits | Using pipe instead of socketpair | Must use `socket.socketpair()`, not `subprocess.PIPE` |
| New session has no memory | Missing env var | Pass `CLAUDE_CODE_DISABLE_AUTO_MEMORY=0` to subprocess |
