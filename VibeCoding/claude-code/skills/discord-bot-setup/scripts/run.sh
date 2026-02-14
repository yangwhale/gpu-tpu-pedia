#!/bin/bash
# Discord Bot wrapper - 检测 exit code 42 自动重启
while true; do
    python3 /home/chrisya/.claude/discord-bot/bot.py
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 42 ]; then
        echo "[$(date)] Restart requested (exit code 42), restarting..."
        sleep 2
        continue
    fi
    echo "[$(date)] Bot exited with code $EXIT_CODE, not restarting."
    break
done
