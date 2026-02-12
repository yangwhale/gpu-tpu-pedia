# Architecture

## Communication

Each Discord user gets a persistent Claude Code process via Unix socketpair + stream-json.
This is the same mechanism the VSCode Claude Code extension uses.

```
Discord User  <->  Bot (Python/py-cord)  <->  Claude Code (socketpair)  <->  Anthropic API
```

### Key CLI Flags

```
claude --input-format stream-json --output-format stream-json --verbose \
       --dangerously-skip-permissions --permission-prompt-tool stdio
```

- `--permission-prompt-tool stdio` is what keeps the process alive (without it, Claude exits after first response)
- `--resume <session-id>` restores a previous session

### Message Protocol

Send:
```json
{"type": "user", "message": {"role": "user", "content": "your message"}}
```

Wait for response with `{"type": "result"}`:
```json
{"type": "result", "session_id": "uuid", "result": "Claude's reply text"}
```

### Process Lifecycle

1. First message from user -> `socket.socketpair()` + spawn Claude process (3s init)
2. Subsequent messages -> reuse existing process, send via socket
3. Process crash -> auto-recreate on next message via `--resume`
4. `/end` -> kill process, archive session ID to history
5. Bot restart -> processes die, but `sessions.json` persists active session ID for `--resume`

## Session Management

`~/.claude/discord-bot/sessions.json` stores per-user:
```json
{
  "user_id": {
    "active": "current-session-uuid",
    "history": ["archived-uuid-1", "archived-uuid-2"]
  }
}
```

- History keeps last 20 sessions
- `/sessions` shows dropdown with summaries (read from `.jsonl` first user message)
- `/end` archives active session, next message creates new one

## Memory

Auto memory works because Claude runs in full interactive mode:
- `CLAUDE.md` loaded automatically
- `MEMORY.md` (first 200 lines) loaded automatically
- Topic files read on demand
- Requires `CLAUDE_CODE_DISABLE_AUTO_MEMORY=0` in subprocess env

## Security

- `message.author.id` verified by Discord servers, safe for whitelist
- Never use self-registration commands
- Disable User Install in Discord Developer Portal
- Only one bot process should run (multiple = duplicate messages)
