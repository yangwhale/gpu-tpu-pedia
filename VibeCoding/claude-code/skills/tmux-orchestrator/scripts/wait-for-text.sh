#!/usr/bin/env bash
# 轮询 tmux pane 等待特定文本出现
# 用法: wait-for-text.sh -t session:window -p 'pattern' [-T timeout] [-i interval] [-l lines] [-F]
#
# 参数:
#   -t  tmux target (session:window.pane)
#   -p  要匹配的正则表达式
#   -F  固定字符串匹配（不用正则）
#   -T  超时秒数 (默认 60)
#   -i  轮询间隔秒数 (默认 2)
#   -l  搜索的历史行数 (默认 200)

set -euo pipefail

TARGET="" PATTERN="" FIXED=false TIMEOUT=60 INTERVAL=2 LINES=200

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--target)  TARGET="$2"; shift 2;;
    -p|--pattern) PATTERN="$2"; shift 2;;
    -F)           FIXED=true; shift;;
    -T)           TIMEOUT="$2"; shift 2;;
    -i)           INTERVAL="$2"; shift 2;;
    -l)           LINES="$2"; shift 2;;
    *)            echo "Unknown: $1" >&2; exit 1;;
  esac
done

[[ -z "$TARGET" ]] && echo "Error: -t target required" >&2 && exit 1
[[ -z "$PATTERN" ]] && echo "Error: -p pattern required" >&2 && exit 1

GREP_FLAGS="-q"
$FIXED && GREP_FLAGS="$GREP_FLAGS -F"

START=$(date +%s)
while true; do
  ELAPSED=$(( $(date +%s) - START ))
  if [[ $ELAPSED -ge $TIMEOUT ]]; then
    echo "TIMEOUT after ${TIMEOUT}s waiting for: $PATTERN" >&2
    exit 1
  fi

  OUTPUT=$(tmux capture-pane -p -J -t "$TARGET" -S "-$LINES" 2>/dev/null || true)
  if echo "$OUTPUT" | grep $GREP_FLAGS -- "$PATTERN"; then
    echo "FOUND after ${ELAPSED}s"
    exit 0
  fi

  sleep "$INTERVAL"
done
