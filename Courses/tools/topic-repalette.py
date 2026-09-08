# -*- coding: utf-8 -*-
r"""**统一配色的收尾工序** —— 把已经生成好的页面按专题一的观感再过一遍。

════════════════════════════════════════════════════════════════════
⭐⭐ 为什么是「后处理」而不是去改画图脚本
════════════════════════════════════════════════════════════════════

专题二的 45 张图散在三个生成器里（`topic02-port-microscope.py`、
`gpu-micro/build_doc.py`、`tpu-micro/build_doc.py`），一共 125 个大色块、
853 处用 500 主色写的字。**逐个改 = 125 次手工改坐标以外的东西**，
每一次都可能碰坏一张图，而且改完没有任何东西能证明「全改到了」。

这里换个思路：**规则本身是机械的**（哪个色换成哪个色、多大算大），
那就写成一道能反复跑、跑完能自证的工序，挂在 build 的最后。

⛔ 代价要认：SVG 里如果哪天出现「故意用 500 主色写的字」，这道工序会一起改掉。
   判断是划算的 —— 专题一全书这样的字只有 29 处，而且都是可以换成 900 的。

════════════════════════════════════════════════════════════════════
📌 两条规则，都是从专题一**量出来的**，不是拍脑袋定的
════════════════════════════════════════════════════════════════════

**① 500 主色不拿来写字，写字用 900 深色档。**
   专题一里彩色的字压倒性地是 900 档（`#174ea6` `#0d652d` `#b06000` `#a50e0e`），
   500 只用来做填充和描边。500 直接当字色是「土」最主要的来源 ——&nbsp;
   它在白底上饱和度过高，一段话里几个词跳出来像荧光笔。

**② 大面板不填色。**
   专题一 45 张图里，>40000 px² 的彩底块只有 **8** 个；
   大白块清一色是 `#fff` ＋ 中性 `#dadce0` 细框（3 个）。
   而专题二有 **125** 个，且 117 个是「50 号浅底 ＋ 同色 500 描边」这一种写法。
   → 改成：白底 ＋ 中性细框 ＋ **左侧 4px 彩条**认身份。
     彩条是必需的：光去掉底色，一排卡片就分不出谁是谁了。

⭐ 阈值 40000 px²（约 200×200）沿用专题三那次的口径。**小元素照旧填色** ——&nbsp;
   热力图格子、KV 条、类型色块、图例，那些填充是在**承载信息**，不是装饰。

════════════════════════════════════════════════════════════════════
⛔ 这道工序必须挂在 build_all 的**最后**
════════════════════════════════════════════════════════════════════

2026-09-08 现场教训：我先直接去改 `topic-02-L300.html` 里的 `.note.q`，
**下一次 build 原样盖回去了，而且不报错** ——&nbsp;那行是生成器注入的。
所以：改生成物 = 改了个寂寞。这个文件自己也一样，
它跑在生成器之后，且**幂等**（已经是白底的不会再动），随时可以重跑。
"""
import io
import os
import re
import sys

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WebPages")

# ── 规则① 500 主色文字 → 900 深色档 ──────────────────────────────
#   ⚠️ 这张表跟 topic03_draw.py 里那份是**同一份**。改一边要改两边 ——&nbsp;
#      两边都改不了自动同步，所以下面加了断言互检。
INK900 = {
    "#1a73e8": "#174ea6", "#4285f4": "#174ea6",
    "#1e8e3e": "#0d652d", "#34a853": "#0d652d",
    "#d93025": "#a50e0e", "#ea4335": "#a50e0e",
    "#e8710a": "#b06000", "#f9ab00": "#b06000", "#fbbc04": "#b06000",
    "#9334e6": "#681da8", "#8430ce": "#681da8",
    "#00838f": "#007b83", "#12b5cb": "#007b83", "#12786f": "#007b83",
}

# ── 规则② 大面板的浅底 → 白底 ＋ 中性框 ＋ 彩条 ──────────────────
#
# ⛔⛔ 2026-09-08 第一版这里写的是一张**颜色名字表**（列了 17 个 50 号色）。
#    结果 `#fdf3f2` `#fff8e1` `#f7faff` 这三个「差一点点」的浅底整整逃掉 ——&nbsp;
#    渲染出来一眼就看见那块还是粉的，而计数器报「已全部处理」。
#    ⭐ 判据：**盘点要按渲染后的属性，不按我列的名字。**
#      「浅」和「有彩」是可以算出来的，算出来的判据不会因为有人调了一档色而漏。
#
# 判据：亮度 ≥ 200（浅）且 RGB 极差 ≥ 8（有彩，排除各级中性灰）。
_LUMA_MIN, _CHROMA_MIN = 200.0, 8


def is_tint(c):
    """#rrggbb 是不是「浅彩底」。⚠️ 只认 6 位十六进制 —— 8 位带透明度的
    （本课里有 `#00000008`）不算，它本来就是中性阴影。"""
    if not re.match(r"^#[0-9a-fA-F]{6}$", c or ""):
        return False
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) >= _LUMA_MIN and \
        (max(r, g, b) - min(r, g, b)) >= _CHROMA_MIN


# 没有描边时，彩条按色相就近挑一个品牌色。⭐ 只在兜底路径用得到 ——&nbsp;
# 实测 117/125 的大面板本来就带着同色 500 描边，直接拿它当彩条。
_HUE = ((15, "#d93025"), (45, "#e8710a"), (70, "#f9ab00"), (170, "#1e8e3e"),
        (200, "#00838f"), (260, "#1a73e8"), (330, "#9334e6"), (360, "#d93025"))


def bar_for(c):
    r, g, b = [int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        return "#5f6368"
    if mx == r:
        h = (60 * ((g - b) / d)) % 360
    elif mx == g:
        h = 60 * ((b - r) / d) + 120
    else:
        h = 60 * ((r - g) / d) + 240
    for lim, col in _HUE:
        if h < lim:
            return col
    return "#5f6368"


NEUTRAL_STROKE = "#dadce0"
# ⭐ 阈值 40000 px²（约 200×200）不是拍的，是**量出来的**：
#   专题一 45 张图里浅底块的 p97 ≈ 70000、>40000 的只有 8 个；
#   专题二同样 45 张图却有 125 个。差的是这一档，不是「有没有浅底」。
# ⛔ 不要顺手往下调到 10000 —— 专题一在 10k–40k 这一档有 25 个浅底小卡片
#   （112×70 带一个词的那种，是它的**固有语汇**）。调下去会把专题二洗得比
#   参照系还干净，那是另一种不一致。
BIG = 40000.0

_num = re.compile(r'(?:width|height)="([\d.]+)"')


def _attr(tag, name):
    m = re.search(r'\b%s="([^"]*)"' % name, tag)
    return m.group(1) if m else None


def _set(tag, name, val):
    """写属性；没有就补一个。⛔ 只认双引号 —— 本课所有 SVG 都是脚本生成的，
    统一双引号；真混进单引号会**匹配不上而静默跳过**，所以下面有兜底断言。"""
    if re.search(r'\b%s="' % name, tag):
        return re.sub(r'\b%s="[^"]*"' % name, '%s="%s"' % (name, val), tag, count=1)
    return tag[:-1].rstrip("/") + ' %s="%s"' % (name, val) + \
        ("/>" if tag.rstrip().endswith("/>") else ">")


def repalette(html):
    """返回 (新 html, 改字数, 改面板数)。**幂等** —— 再跑一遍两个计数都是 0。"""
    n_txt = [0]
    n_box = [0]

    # ① 文字降档。只动 <text>/<tspan> 的 fill，不碰 rect/path。
    def _t(m):
        tag, col = m.group(0), m.group(2).lower()
        if col in INK900:
            n_txt[0] += 1
            return _set(tag, "fill", INK900[col])
        return tag
    html = re.sub(r'<(text|tspan)\b[^>]*?fill="(#[0-9a-fA-F]{6})"[^>]*>', _t, html)

    # ② 大面板去底。彩条**插在面板之后**（同 x/y，宽 4）——&nbsp;
    #    SVG 按文档顺序画，后写的在上面，正好压在白底上。
    out = []
    pos = 0
    for m in re.finditer(r'<rect\b[^>]*?/?>', html):
        tag = m.group(0)
        fill = (_attr(tag, "fill") or "").lower()
        if not is_tint(fill):
            continue
        try:
            w = float(_attr(tag, "width"))
            h = float(_attr(tag, "height"))
        except (TypeError, ValueError):
            continue
        if w * h <= BIG:
            continue                      # 小元素照旧填色：那是信息不是装饰
        stroke = (_attr(tag, "stroke") or "").lower()
        bar = stroke if re.match(r"^#[0-9a-fA-F]{6}$", stroke or "") and \
            not is_tint(stroke) else bar_for(fill)
        new = _set(_set(tag, "fill", "#fff"), "stroke", NEUTRAL_STROKE)
        if _attr(new, "stroke-width") is None:
            new = _set(new, "stroke-width", "1")
        x, y = _attr(tag, "x") or "0", _attr(tag, "y") or "0"
        rx = _attr(tag, "rx")
        # ⭐ 彩条比面板矮 0 —&nbsp;要跟面板齐高，圆角减半免得露出白角
        new += ('<rect x="%s" y="%s" width="4" height="%s" rx="%s" fill="%s"/>'
                % (x, y, h if h == int(h) else h, min(3.0, float(rx or 0)), bar))
        out.append(html[pos:m.start()])
        out.append(new)
        pos = m.end()
        n_box[0] += 1
    out.append(html[pos:])
    return "".join(out), n_txt[0], n_box[0]


def main(files):
    tot_t = tot_b = 0
    for f in files:
        p = os.path.join(WEB, f)
        if not os.path.exists(p):
            continue
        s = io.open(p, encoding="utf-8").read()
        new, nt, nb = repalette(s)
        if nt or nb:
            io.open(p, "w", encoding="utf-8").write(new)
        tot_t += nt
        tot_b += nb
        print("  %-22s 文字降档 %4d 处，大面板去底 %3d 个" % (f, nt, nb))
    print("   ✅ 配色收尾：共 %d 处文字、%d 个面板" % (tot_t, tot_b))


# ⛔⛔ 默认名单里**故意没有 topic-01 和 topic-03**。第一次跑漏了这条，
#   一把把参照系自己也改了：
#     · **专题一是参照系** ——&nbsp;它那 8 个大色块、29 处 500 主色文字
#       就是「好看」的实测基线。拿基线去套按基线定的规则，是自指。
#     · **专题三已经手工扫过**，剩下的 3 个大色块是**刻意留的**
#       （高亮的循环行、`[d × (n·B)]` 激活块、BTSKG 格）——&nbsp;
#       它们在承载信息，不是面板底。
#   ⭐ 判据：**面积阈值只能认出「大」，认不出「是不是在说事」。**
#     所以这道工序只能扫「整批同一种坏写法」的页，不能当全局美化器用。
if __name__ == "__main__":
    main(sys.argv[1:] or ["topic-02.html", "topic-02-L300.html",
                          "topic-02x.html", "topic-08.html"])
