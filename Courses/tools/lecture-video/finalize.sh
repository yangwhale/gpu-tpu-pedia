#!/usr/bin/env bash
# 渲染产物 → 成片，加软字幕。两档：
#   默认（网页内嵌）：1080p、crf 28 + stillimage，压到适合进仓库的体积
#   --hd（传 YouTube / B 站）：配 render.py --scale 2 出的 4K，crf 18、音频原样拷贝。
#     平台会二次压缩，而且按上传分辨率分配码率 —— 传 1080p 会被压得发糊，传 4K 才能拿到高码率档。
# 用法：finalize.sh [--hd] <render 出的.mp4> <字幕.srt> <输出.mp4>
set -euo pipefail
if [[ "${1:-}" == "--hd" ]]; then shift; V=(-crf 18 -preset slow -tune stillimage); A=(-c:a copy)
else V=(-crf 28 -preset slow -tune stillimage); A=(-c:a aac -b:a 64k); fi
ffmpeg -y -loglevel error -i "$1" -i "$2" -map 0:v -map 0:a -map 1 -c:v libx264 "${V[@]}" \
  -pix_fmt yuv420p "${A[@]}" -c:s mov_text -metadata:s:s:0 language=chi \
  -movflags +faststart "$3"
ls -la "$3"
