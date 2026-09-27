#!/usr/bin/env bash
# 把飞书语音「重播」用的那段现场原声转成 mp3。
# 用法：pcm2mp3.sh <输出.mp3> [开头剪掉的秒数] [pcm 文件，默认取最新]
set -euo pipefail
OUT="$1"; SKIP="${2:-0}"; PCM="${3:-$(ls -t /tmp/jarvis-tts-buf/*.pcm | head -1)}"
ffmpeg -y -loglevel error -f s16le -ar 48000 -ac 2 -ss "$SKIP" -i "$PCM" -ac 1 -b:a 64k "$OUT"
echo "ok $OUT ← $PCM ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT") 秒)"
