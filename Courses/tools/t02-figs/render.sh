#!/bin/bash
# 渲染一张图为 PNG 供目视核对：./render.sh fig_t1_chip
# 每张图都必须走这一步 —— 溢出、遮挡、图例漂移只有看图才发现得了。
set -e
cd "$(dirname "$0")"
M=$1
python3 "$M.py" "out/$M.svg"
W=$(python3 -c "import re,io;s=io.open('out/$M.svg',encoding='utf-8').read();print(int(float(re.search(r'<svg[^>]*width=\"([\d.]+)\"',s).group(1))))")
H=$(python3 -c "import re,io;s=io.open('out/$M.svg',encoding='utf-8').read();print(int(float(re.search(r'<svg[^>]*height=\"([\d.]+)\"',s).group(1))))")
# TMPDIR 在这个 session 里被套了七层，Chrome 的 SingletonSocket 路径会超长直接 FATAL。
# 固定 TMPDIR + 独立 user-data-dir，两个都要，少一个还是崩。
TMPDIR=/tmp google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
  --user-data-dir=/tmp/cc-render --window-size="$W,$H" --screenshot="out/$M.png" \
  --default-background-color=FFFFFFFF "file://$PWD/out/$M.svg" 2>/dev/null
echo "out/$M.png  ${W}x${H}"
