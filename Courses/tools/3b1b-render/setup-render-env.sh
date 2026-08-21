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
# 场景代码还会 import 这几个：auto_regression 要 transformers（跑 GPT-2），
# embedding / attention / ml_basics 顶层 import gensim
"$VENV/bin/pip" install -q gensim transformers tiktoken openai
xvfb-run -a "$VENV/bin/python" -c "import manimlib, torch; print('  manimgl + torch OK')"

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

# ── 5 · 私有素材的本地替身 ────────────────────────────────────
# 作者的图片与数据放在他自己的 Dropbox 里，不在公开仓库。
# 这里按场景代码里出现过的名字批量造占位素材，让相关场景能跑起来。
say "私有素材替身"
mkdir -p "$WORK/thumbnails" "$WORK/images/raster" "$WORK/images/vector" \
         "$WORK/videos/2024/transformers/data"

"$VENV/bin/python" - "$WORK" <<'PY'
from PIL import Image, ImageDraw, ImageFont
import sys, pathlib, textwrap, colorsys
W = pathlib.Path(sys.argv[1])
R, V, T = W / "images/raster", W / "images/vector", W / "thumbnails"
try:
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
    fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    fh = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
except OSError:
    fb = fs = fh = ImageFont.load_default()

# 名字来自 grep '(ImageMobject|SVGMobject)\(' _2024/transformers/*.py
RASTER = ["AttentionPaper","AttentionPaperStill","AudioSnippet","BlueFluff","Bot",
 "CHMTopText","ChatBot","FederalReserve","HumanAIScript","JaggedCurl1","JaggedCurl2",
 "JohnRFirth","LipMole","MiniEiffelTower1","NetworkEnd","RiverBank","ShrewMole",
 "SmallFluffCreature","TearOff","VerdantForest","Zoolander","and_gate","comp_worker",
 "computer_stall","warning","EmbeddingStill","CHM_Exterior","Chapter5_TN3"] + \
 [f"Dalle3_{w}" for w in ["fluffy","blue","creature","creature_2","verdant","forest"]]
SVG = ["GenericComputer","History","Museum","OpenAI","gpu_large"]

made = 0
for n in RASTER:
    p = R / f"{n}.png"
    if p.exists(): continue
    h = (sum(ord(c) for c in n) * 37) % 360
    r, g, b = [int(x * 130) + 40 for x in colorsys.hsv_to_rgb(h / 360, .45, .85)]
    im = Image.new("RGB", (1024, 1024), (r, g, b)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([28, 28, 996, 996], radius=26, outline=(235, 235, 240), width=5)
    y = 430
    for line in textwrap.wrap(n.replace("_", " "), 16):
        w = d.textbbox((0, 0), line, font=fb)[2]
        d.text(((1024 - w) / 2, y), line, font=fb, fill=(248, 248, 252)); y += 72
    w = d.textbbox((0, 0), "placeholder", font=fs)[2]
    d.text(((1024 - w) / 2, 890), "placeholder", font=fs, fill=(215, 215, 225))
    im.save(p); made += 1

for n in SVG:
    p = V / f"{n}.svg"
    if p.exists(): continue
    p.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" width="240" height="240">
  <rect x="10" y="10" width="220" height="220" rx="22" fill="none" stroke="#e8e8ee" stroke-width="6"/>
  <text x="120" y="116" font-family="DejaVu Sans" font-size="22" fill="#e8e8ee" text-anchor="middle">{n}</text>
  <text x="120" y="150" font-family="DejaVu Sans" font-size="16" fill="#9a9aa6" text-anchor="middle">placeholder</text>
</svg>'''); made += 1

for name, top, bot in [("Chapter5_TN5", "Chapter 5", "Transformers"),
                       ("Chapter6_TN4", "Chapter 6", "Attention")]:
    p = T / f"{name}.png"
    if p.exists(): continue
    im = Image.new("RGB", (1920, 1080), (12, 12, 16)); d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 1880, 1040], outline=(70, 70, 80), width=6)
    for txt, fnt, y, col in [(top, fs, 420, (150, 150, 160)),
                             (bot, fh, 500, (240, 240, 245)),
                             ("placeholder", fs, 900, (90, 90, 100))]:
        w = d.textbbox((0, 0), txt, font=fnt)[2]
        d.text(((1920 - w) / 2, y), txt, font=fnt, fill=col)
    im.save(p); made += 1
print(f"  新造 {made} 个占位素材")
PY

# 场景读的两个数据文件
D="$WORK/videos/2024/transformers/data"
[ -f "$D/athlete_sports.txt" ] || cat > "$D/athlete_sports.txt" <<'TXT'
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
[ -f "$D/facts.txt" ] || cat > "$D/facts.txt" <<'TXT'
Michael Jordan plays basketball
The Eiffel Tower is in Paris
Water boils at 100 degrees Celsius
Shakespeare wrote Hamlet
The Moon orbits the Earth
Mount Everest is the tallest mountain
Python is a programming language
The Pacific is the largest ocean
Einstein developed relativity
Tokyo is the capital of Japan
TXT
echo "  数据文件 x2"

# ── 6 · 场景代码的两处本地修正 ─────────────────────────────────
say "场景代码本地修正"

# (a) 缩略图路径从作者的绝对路径改成本地
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

# (b) np.product 在 numpy 2 已移除，换成 np.prod
if grep -rq "np\.product(" "$REPO/_2024/transformers/"; then
  sed -i 's/np\.product(/np.prod(/g' "$REPO/_2024/transformers/"*.py
  echo "  np.product → np.prod（numpy 2 已移除前者）"
else
  echo "  np.product 已修过"
fi

say "完成 —— 试渲一张："
echo "  cd $REPO && xvfb-run -a -s '-screen 0 1920x1080x24' \\"
echo "    $VENV/bin/manimgl _2024/transformers/mlp.py MLPIcon -w -s --hd \\"
echo "    --video_dir $WORK/frames"
echo
echo "  批量渲：$(dirname "$0")/render-scenes.sh"
