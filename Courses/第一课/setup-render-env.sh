#!/usr/bin/env bash
# 一键搭好渲染 3Blue1Brown 场景的环境（幂等，可重复跑）。
#
# 做四件事：
#   1. 装系统依赖与 Python 环境
#   2. 稀疏 clone 场景源码
#   3. 修 manimgl 的一个真 bug（numpy 标量识别）
#   4. 补上 3b1b 私有素材的本地替身（占位缩略图 + 数据文件）
#
# 背景与踩坑见同目录 RENDER.md。
set -euo pipefail

VENV="${VENV:-$HOME/manim-venv}"
REPO="${REPO:-$HOME/3b1b-videos}"
WORK="${WORK:-$HOME/3b1b-render}"

say() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

# ── 1 · 系统依赖 ────────────────────────────────────────────────
say "系统依赖"
NEED=(libcairo2-dev libpango1.0-dev pkg-config python3-dev
      libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev mesa-utils
      xvfb ffmpeg texlive texlive-latex-extra texlive-fonts-extra
      texlive-science dvisvgm)
MISSING=()
for p in "${NEED[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "缺 ${#MISSING[@]} 个包，安装中：${MISSING[*]}"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING[@]}"
  # texlive 装完宏包索引可能还没就绪，这里显式刷一次，
  # 否则会看到「LaTeX 装好了但编译仍失败」的假象（我们踩过）
  sudo mktexlsr >/dev/null 2>&1 || true
else
  echo "已齐备"
fi

# ── 2 · Python 环境 ─────────────────────────────────────────────
say "Python 环境 ($VENV)"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# setuptools<81 不是可选的：manimgl 1.7.2 顶层 import pkg_resources
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q manimgl "setuptools<81"
"$VENV/bin/python" -c "import torch" 2>/dev/null || \
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -c "import manimlib, torch; print('  manimgl + torch OK')"

# ── 3 · 场景源码 ────────────────────────────────────────────────
say "场景源码 ($REPO)"
if [ ! -d "$REPO/.git" ]; then
  git clone --filter=blob:none --sparse --depth 1 \
      https://github.com/3b1b/videos.git "$REPO"
  git -C "$REPO" sparse-checkout set _2024/transformers custom
else
  echo "已存在"
fi

# custom_config.yml 里写死了作者本人的 Dropbox 路径，改到本地
python3 - "$REPO/custom_config.yml" "$REPO" "$WORK" <<'PY'
import sys, re, pathlib
cfg, repo, work = sys.argv[1:4]
t = pathlib.Path(cfg).read_text()
t = re.sub(r'removed_mirror_prefix: ".*"', f'removed_mirror_prefix: "{repo}/"', t)
t = re.sub(r'base: ".*"',                   f'base: "{work}/"',                 t)
t = t.replace('resolution: (3840, 2160)', 'resolution: (1920, 1080)')
pathlib.Path(cfg).write_text(t)
print("  custom_config.yml 已指向本地")
PY

# ── 4 · 修 manimgl 的 numpy 标量 bug ────────────────────────────
# 根因：get_stroke_opacity() 从 float32 数组取值，返回 np.float32；
# 而 np.float32 不是 python float 的子类（只有 np.float64 是），
# 于是 isinstance(opacity, (float,int)) 判否 → np.array() 得到 0 维数组
# → resize_with_interpolation 里 len() 报 "len() of unsized object"。
# 影响到所有带 VFadeIn/VFadeOut 的场景。
say "修 manimgl numpy 标量 bug"
MOB="$VENV/lib/python3.12/site-packages/manimlib/mobject/mobject.py"
if grep -q "np.floating, np.integer" "$MOB" 2>/dev/null; then
  echo "已修过"
else
  cp "$MOB" "$MOB.orig"
  python3 - "$MOB" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
old = "if not isinstance(opacity, (float, int)):"
new = "if not isinstance(opacity, (float, int, np.floating, np.integer)):"
assert t.count(old) == 1, f"锚点出现 {t.count(old)} 次，预期 1 次 —— manimgl 版本可能变了"
p.write_text(t.replace(old, new)); print("  已修（原文件备份为 .orig）")
PY
fi

# ── 5 · 私有素材的本地替身 ──────────────────────────────────────
# 作者的图片与数据放在他自己的 Dropbox 里，不在公开仓库。
# 这里造两张占位缩略图 + 一份运动员数据，让相关场景能跑起来。
say "私有素材替身"
mkdir -p "$WORK/thumbnails" "$WORK/videos/2024/transformers/data"

"$VENV/bin/python" - "$WORK/thumbnails" <<'PY'
from PIL import Image, ImageDraw, ImageFont
import sys, pathlib
out = pathlib.Path(sys.argv[1])
try:
    f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
    f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 64)
except OSError:
    f1 = f2 = ImageFont.load_default()
for name, top, bot in [("Chapter5_TN5", "Chapter 5", "Transformers"),
                       ("Chapter6_TN4", "Chapter 6", "Attention")]:
    im = Image.new("RGB", (1920, 1080), (12, 12, 16)); d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 1880, 1040], outline=(70, 70, 80), width=6)
    for txt, fnt, y, col in [(top, f2, 420, (150, 150, 160)),
                             (bot, f1, 520, (240, 240, 245)),
                             ("placeholder — 原缩略图不在公开仓库中", f2, 900, (90, 90, 100))]:
        w = d.textbbox((0, 0), txt, font=fnt)[2]
        d.text(((1920 - w) / 2, y), txt, font=fnt, fill=col)
    im.save(out / f"{name}.png")
print("  占位缩略图 x2")
PY

cat > "$WORK/videos/2024/transformers/data/athlete_sports.txt" <<'TXT'
Michael Jordan plays basketball
Serena Williams plays tennis
Lionel Messi plays soccer
Tiger Woods plays golf
Usain Bolt runs track
Simone Biles does gymnastics
Wayne Gretzky plays hockey
Roger Federer plays tennis
LeBron James plays basketball
Cristiano Ronaldo plays soccer
Michael Phelps swims
Yao Ming plays basketball
Novak Djokovic plays tennis
Tom Brady plays football
Katie Ledecky swims
TXT
echo "  athlete_sports.txt"

# 把缩略图路径从作者的绝对路径改成本地
python3 - "$REPO/_2024/transformers/mlp.py" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
old = 'folder = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/Thumbnails"'
new = 'folder = str(Path.home() / "3b1b-render" / "thumbnails")  # LOCAL OVERRIDE'
if old in t:
    p.write_text(t.replace(old, new)); print("  mlp.py 缩略图路径已改本地")
else:
    print("  mlp.py 已改过")
PY

say "完成 —— 试渲一张："
echo "  cd $REPO && xvfb-run -a -s '-screen 0 1920x1080x24' \\"
echo "    $VENV/bin/manimgl _2024/transformers/mlp.py MLPIcon -w -s --hd \\"
echo "    --video_dir $WORK/frames"
echo
echo "  批量渲：$(dirname "$0")/render-scenes.sh"
