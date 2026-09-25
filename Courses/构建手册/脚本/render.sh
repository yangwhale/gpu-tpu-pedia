#!/usr/bin/env bash
# 渲染 → 压缩 → 查接缝，一条命令。
#
# 用法：
#   render.sh <脚本.py> <SceneName> <输出.mp4> [--draft]
#     --draft  用 480p15 出草稿（几秒钟），**迭代阶段一律先用它**
#
# ⭐ 为什么要有这个脚本：三步里每一步都有个记不住的参数
#   （media_dir 的输出路径长什么样、crf 多少、末帧怎么取），
#   手敲三次就会漏一次。而**漏掉的最常是第三步** —— 于是接缝 bug 就上线了。
set -euo pipefail

SCRIPT="${1:?用法: render.sh <脚本.py> <SceneName> <输出.mp4> [--draft]}"
SCENE="${2:?缺 SceneName}"
OUT="${3:?缺输出 mp4 路径}"
DRAFT="${4:-}"

MANIM="${MANIM_BIN:-$HOME/.venvs/manim/bin/manim}"
[ -x "$MANIM" ] || { echo "⛔ 找不到 manim：$MANIM（用 MANIM_BIN 覆盖）"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
# 课程仓库里的循环检查器和基线已经部署在 tools/manim/（build-all.sh 也用它们）
TOOLS_MANIM="$HERE/../../tools/manim"
STEM="$(basename "$SCRIPT" .py)"
MEDIA="${MANIM_MEDIA:-/tmp/manim-out}"

if [ "$DRAFT" = "--draft" ]; then
  Q=-ql; SUB=480p15; WIDE=640; CRF=32
else
  Q=-qh; SUB=1080p60; WIDE=960; CRF=30
fi

echo "▸ 渲染 $SCENE（$Q）"
"$MANIM" --format=mp4 $Q --media_dir "$MEDIA" "$SCRIPT" "$SCENE"

SRC="$MEDIA/videos/$STEM/$SUB/$SCENE.mp4"
[ -f "$SRC" ] || { echo "⛔ 没产出 $SRC —— 场景里是不是一个 play/wait 都没有？那样 manim 只出 PNG"; exit 1; }

echo "▸ 压缩 → $OUT"
mkdir -p "$(dirname "$OUT")"
ffmpeg -y -v error -i "$SRC" -vf "scale=$WIDE:-2" \
  -c:v libx264 -crf $CRF -preset slow -pix_fmt yuv420p \
  -movflags +faststart -an "$OUT"

printf '   %s  %.1f 秒  %s\n' "$(basename "$OUT")" \
  "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")" \
  "$(du -h "$OUT" | cut -f1)"

echo "▸ 查首尾接缝"
# ⛔ 基线默认存在 mp4 旁边，但项目常把它跟脚本放一起（例如 tools/manim/）。
#   不给出路的话，走 render.sh 时**永远显示「没有基线」** —— 而同一个项目
#   直接跑 check-loop.py 却能找到，两边说法不一致，非常误导。
#   ⭐ 判据：**同一件事有两个入口时，它们对「配置在哪」必须有同一个答案。**
BASELINE_ARG=(--baseline "${LOOP_BASELINE:-$TOOLS_MANIM/loop-baseline.json}")
python3 "$TOOLS_MANIM/check-loop.py" --media "$(dirname "$OUT")"   "${BASELINE_ARG[@]}" "$OUT" || true
echo "   ⭐ 拼接图：/tmp/loopdiff-$(basename "$OUT" .mp4).png —— **务必看一眼**，数字分不开「差一整幕」和「偏一像素」。"
