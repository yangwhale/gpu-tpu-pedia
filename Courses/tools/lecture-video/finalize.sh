#!/usr/bin/env bash
# 渲染产物 → 网页用：加软字幕、压到适合进仓库的体积（静态画面多，crf 28 + stillimage 足够清楚）。
# 用法：finalize.sh <render 出的.mp4> <字幕.srt> <输出.mp4>
set -euo pipefail
ffmpeg -y -loglevel error -i "$1" -i "$2" -map 0:v -map 0:a -map 1 -c:v libx264 -crf 28 -preset slow \
  -tune stillimage -pix_fmt yuv420p -c:a aac -b:a 64k -c:s mov_text -metadata:s:s:0 language=chi \
  -movflags +faststart "$3"
ls -la "$3"
