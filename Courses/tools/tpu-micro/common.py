# -*- coding: utf-8 -*-
"""GPU 显微镜图 —— 共用样式。视觉标准对标 TPU 全景图（同一套 Google 配色 + 字号阶梯）。"""

BL, RD, YL, GN, PU, TL = "#1a73e8", "#d93025", "#f9ab00", "#1e8e3e", "#8430ce", "#12786f"
INK, SUB, LINE, BG = "#202124", "#5f6368", "#dadce0", "#f8f9fa"
GREY = "#9aa0a6"

# 浅底色（与描边同色系）
FILL = {BL: "#e8f0fe", RD: "#fce8e6", YL: "#fef7e0", GN: "#e6f4ea",
        PU: "#f3e8fd", TL: "#e0f2f0", SUB: "#f1f3f4", GREY: "#f1f3f4"}

DEFS = """<defs><style>
.ttl{font:700 18px "Noto Sans CJK SC",sans-serif;fill:#202124}
.sec{font:700 15px "Noto Sans CJK SC",sans-serif;fill:#202124}
.box{font:700 13px "Noto Sans CJK SC",sans-serif;fill:#202124}
.lbl{font:600 12px "Noto Sans CJK SC",sans-serif;fill:#202124}
.sm{font:400 11px "Noto Sans CJK SC",sans-serif;fill:#5f6368}
.xs{font:400 11px "Noto Sans CJK SC",sans-serif;fill:#5f6368}
.xxs{font:400 10px "Noto Sans CJK SC",sans-serif;fill:#5f6368}
.num{font:700 12px "Roboto Mono",monospace}
.numb{font:700 14px "Roboto Mono",monospace}
.mono{font:400 11px "Roboto Mono",monospace;fill:#5f6368}
</style>
<marker id="aB" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#1a73e8"/></marker>
<marker id="aR" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#d93025"/></marker>
<marker id="aG" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#1e8e3e"/></marker>
<marker id="aP" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#8430ce"/></marker>
<marker id="aK" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#5f6368"/></marker>
</defs>"""

# ⛔ 2026-09-01：这里一度加过渐变／柔和阴影／径向光晕（"Console 质感"），
#    当天就被否掉了 —— **教学图的问题从来不是不够精致，是字太多。**
#    加质感只会让一张信息过载的图变成一张精致的信息过载的图。
#    下面几个函数名保留（有图在用），但一律画成扁平：纯色浅底 ＋ 细描边。
#    **不要再往回加。**


def grad(c):
    """曾经返回渐变，现在返回平涂浅底。保留是为了不用改所有调用点。"""
    return FILL.get(c, "#fff")


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


import re as _re

# para() 认得的四个行内标记。它们在 SVG 里都是未知元素。
_BADMARK = _re.compile(r"</?(?:b|r|g|code)>")

# DEFS 里真实存在的 marker id。引用一个不存在的（比如手滑写成 "aE"），
# SVG 规范要求渲染器**静默忽略** —— 线照画、箭头就是没有，不报错也不警告。
# 跟 <b> 塞进 f.t() 是同一类失败：引用不存在的东西，被默默吃掉。
# P-19 的两条血缘箭头就这么丢了，直到画 P-24 时才发现。
_MARKERS = frozenset(_re.findall(r'<marker id="([^"]+)"', DEFS))


def _ckmark(m):
    assert m is None or m in _MARKERS, (
        "marker %r 不存在，SVG 会静默不画箭头。可用的只有：%s"
        % (m, ", ".join(sorted(_MARKERS))))


class Fig:
    """极薄的 SVG 拼装器 —— 有意不做布局引擎：所有坐标手算，改图时所见即所得。"""

    def __init__(self, w, h, label):
        self.w, self.h, self.label = w, h, label
        self.p = [DEFS]

    def raw(self, s):
        self.p.append(s)

    def rect(self, x, y, w, h, fill="#fff", stroke=None, sw=1.0, rx=6, dash=None, extra=""):
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{sw}"'
        if dash:
            s += f' stroke-dasharray="{dash}"'
        self.p.append(s + extra + "/>")

    def t(self, x, y, s, cls="xs", fill=None, anchor=None, weight=None):
        # `t()` 把 s 原样塞进 <text>，**不解析行内标记** —— 那是 para() 的活。
        # 于是 `f.t(..., "<b>16</b>")` 会写出 `<text><b>16</b></text>`：
        # `<b>` 不是 SVG 元素，浏览器把它**连同里面的 16 一起丢掉**。
        # 不报错、不警告，只是那个数字凭空消失 —— 除非盯着渲染图找它，否则发现不了。
        # （P-8 的两个关键计数就是这么没的。）
        # `<tspan>` 是合法 SVG，title() 一直靠它上色，所以只挡 para 那四个标记。
        assert not _BADMARK.search(str(s)), (
            "f.t() 不解析行内标记，<b>/<r>/<g>/<code> 会连内容一起被浏览器丢掉；"
            "要强调请改用 para()，只要换颜色就传 fill 参数：%r" % s)
        # **任何 `&...;` 写法都别用。** 两种都会原样显示成字面量，而且都不报错：
        #   · `&nbsp;` 这类命名实体 —— XML/SVG 里根本没定义（P-26、P-27 各中一次）
        #   · `&#8194;` 这类数字引用 —— 本来是合法 XML，但 para() 会先把 `&`
        #     转义成 `&amp;`，于是照样变成字面量（P-26 的 N² 带中过一次）
        # 两次都只有盯着渲染图才看得出来。要特殊空格就直接写那个字符本身
        # （U+00A0 不断行空格、U+2002 半角空格），要转义符号用 escape() 处理过的原字符。
        assert not _re.search(r"&(?:[a-zA-Z]+|#\d+|#[xX][0-9a-fA-F]+);", str(s)), (
            "SVG 里不要写 &...; ——命名实体没定义，数字引用会被二次转义，"
            "两种都会原样显示成字面量；直接写那个字符本身：%r" % s)
        a = f' text-anchor="{anchor}"' if anchor else ""
        f = f' fill="{fill}"' if fill else ""
        wt = f' font-weight="{weight}"' if weight else ""
        self.p.append(f'<text x="{x}" y="{y}" class="{cls}"{f}{a}{wt}>{s}</text>')

    def line(self, x1, y1, x2, y2, c=SUB, sw=1.5, marker=None, dash=None):
        _ckmark(marker)
        m = f' marker-end="url(#{marker})"' if marker else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{sw}"{m}{d}/>')

    def path(self, d, c=SUB, sw=1.5, fill="none", marker=None, dash=None):
        _ckmark(marker)
        m = f' marker-end="url(#{marker})"' if marker else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<path d="{d}" stroke="{c}" stroke-width="{sw}" fill="{fill}"{m}{ds}/>')

    def title(self, s, badge=None, badge_c=RD):
        self.t(20, 26, s, "ttl")
        if badge:
            bw = 11 * len(badge) + 22
            self.rect(self.w - 20 - bw, 10, bw, 22, fill=badge_c, rx=11)
            self.t(self.w - 20 - bw / 2, 25, badge, "lbl", "#fff", "middle")

    def legend(self, items, y=58, x0=20):
        """items: [(color, text), ...] —— 与 TPU 图相同的色块图例。

        ⛔ 这一行是**一整行铺开的**，写长了就冲出 viewBox ——&nbsp;而 SVG
        超出 viewBox 的文字是<b>静默裁掉</b>的：渲染图上看不出「被截了」
        和「本来就短」的区别。P-20 最后一格就这么丢过两条限定
        （「跑的是 GB200 不是 GB300」「配色不表示好坏」），
        实测宽 555.8 px 塞进 384 px 的位置，肉眼一直没发现，
        最后是拿 headless Chrome 量 getComputedTextLength 才逮到的。

        所以下面这条断言必须留着。`11.5 * _wlen` 比实测宽约 10%
        （实测同一串 555.8 px、估算 610.1 px）——&nbsp;**偏保守是故意的**：
        宁可在临界处误报让人手动确认，也不能漏掉真的溢出。
        """
        x = x0
        for c, txt in items:
            self.rect(x, y - 10, 11, 11, fill=c, rx=2)
            self.t(x + 17, y, txt, "sm")
            x += 17 + 11.5 * _wlen(txt) + 22
        right = x - 22                      # 减掉最后一格多加的那段间距
        assert right <= self.w - 20, (
            "图例整行 %.0f px，超出 viewBox 可用宽度 %d px（%.0f px 会被静默裁掉）。"
            "拆成两行、缩短文字，或把说明挪进图注：\n  %s"
            % (right - x0, self.w - 20 - x0, right - (self.w - 20),
               "\n  ".join(t for _, t in items)))

    # ── 三个成组原语（一律扁平，见文件头那段 ⛔）────────────────────
    # rect() 画「一块底色」；下面这几个画「一个物件」。
    # 判断标准：它在画面里是不是一个可以被指着说「这个」的东西？

    def card(self, x, y, w, h, c=None, rx=8, elev=1, accent=None, aw=4,
             fill=None):
        """一个白底方块 ＋ 细描边。accent 是左侧那条竖色带。

        c      描边色（None → 用极淡的灰线，免得整张图糊成一片）
        elev   保留参数，已无效果（原来是阴影层级）
        """
        self.rect(x, y, w, h, fill or "#fff", c or LINE, 1.2, rx)
        if accent:
            # 只圆左边两个角：右边要跟卡片内容齐平，圆了会露出白缝
            self.p.append(
                f'<path d="M{x + rx},{y} h-{rx - aw} a{aw},{aw} 0 0 0 -{aw},{aw} '
                f'v{h - 2 * aw} a{aw},{aw} 0 0 0 {aw},{aw} h{rx - aw} z" fill="{accent}"/>')
            self.p.append(
                f'<rect x="{x + rx - aw}" y="{y}" width="{aw}" height="{h}" fill="{accent}"/>')

    def glow(self, *a, **k):
        """曾经是雾化光晕，现在什么都不画。保留是为了不用改所有调用点。"""

    def panel(self, x, y, w, h, c, rx=12, grid=False):
        """一块区域底：主色平涂浅底 ＋ 同色描边。"""
        self.rect(x, y, w, h, FILL.get(c, "#fff"), c, 1.6, rx)

    def bg(self):
        """曾经铺页面渐变，现在什么都不画。"""

    def out(self):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{self.label}">\n'
                + "\n".join(self.p) + "\n</svg>")


def _wlen(s):
    """粗略宽度：CJK 记 1，ASCII 记 0.55。"""
    return sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in s)


# ══════════════════════════════════════════════════════════════════════
# 富文本换行 —— SVG 的 <text> 不会自动折行，长句必须自己切。
# 支持三种行内标记：<b>粗体深色</b>、<code>等宽蓝色</code>、<r>红色</r>。
# ══════════════════════════════════════════════════════════════════════
# 第三个数是宽度系数。`code` 走 Roboto Mono，等宽字的实际步进是 0.6 em，
# 而下面 _cw 对 ASCII 一律按 0.55 折算 —— 所以这里要补回 0.6/0.55 ≈ 1.10。
# 原来写 0.92 是照抄比例字体的经验值，结果是**越长的代码串越低估**，
# 一行 code 多的段落会直接冲出卡片边框（T-5 卡片 A 就是这么溢出的）。
_MARK = {"b": ('<tspan font-weight="700" fill="#202124">', "</tspan>", 1.0),
         "code": ('<tspan class="mono" fill="#1967d2">', "</tspan>", 1.10),
         "r": ('<tspan font-weight="700" fill="#d93025">', "</tspan>", 1.0),
         "g": ('<tspan fill="#9aa0a6">', "</tspan>", 1.0)}


def _tokenize(s):
    """→ [(字符, 标记名 or None), ...]"""
    out, i, cur = [], 0, None
    while i < len(s):
        if s[i] == "<":
            j = s.find(">", i)
            if j > 0:
                tag = s[i + 1:j]
                if tag in _MARK:
                    cur = tag; i = j + 1; continue
                if tag.startswith("/") and tag[1:] in _MARK:
                    cur = None; i = j + 1; continue
        out.append((s[i], cur)); i += 1
    return out


def _cw(ch, px, k=1.0):
    """单字符宽度估算：CJK 与全角标点算一个字身，ASCII 算 0.55。"""
    o = ord(ch)
    return px * k * (1.0 if o > 0x2E80 or o in (0xB7,) else 0.55)


# 遇到这些字符不能作为行首 / 行尾
_NO_HEAD = "，。、；：）」』】》!?，.,;:)]}%"
_NO_TAIL = "（「『【《([{"


def _alnum(ch):
    return ch.isascii() and (ch.isalnum() or ch in "_.-")


def wrap(text, maxw, px=10):
    """把带标记的字符串切成若干行，每行仍是带 tspan 的 SVG 片段。

    两条排版规则：① 标点不悬在行首；② 不在 ASCII 单词中间断开
    （否则 "Tensor Core" 会被切成 "Tensor Cor" ／ "e"）。
    """
    toks = _tokenize(text)
    lines, cur, w = [], [], 0.0
    for ch, m in toks:
        cwid = _cw(ch, px, _MARK[m][2] if m else 1.0)
        if w + cwid > maxw and cur:
            if ch in _NO_HEAD:                      # 标点跟着上一行走
                cur.append((ch, m)); lines.append(cur); cur, w = [], 0.0; continue
            # 若正卡在一个 ASCII 单词中间，把整个词挪到下一行
            carry = []
            if _alnum(ch):
                while cur and _alnum(cur[-1][0]):
                    carry.insert(0, cur.pop())
            if not cur:                             # 整行就是一个超长词，只能硬切
                cur, carry = carry, []
            lines.append(cur)
            cur = carry
            w = sum(_cw(c, px, _MARK[mm][2] if mm else 1.0) for c, mm in cur)
        cur.append((ch, m)); w += cwid
    if cur:
        lines.append(cur)
    return [_emit(l) for l in lines]


def _emit(chars):
    out, cur = [], None
    for ch, m in chars:
        if m != cur:
            if cur:
                out.append(_MARK[cur][1])
            if m:
                out.append(_MARK[m][0])
            cur = m
        out.append(esc(ch))
    if cur:
        out.append(_MARK[cur][1])
    return "".join(out)


# 投影到大屏时 9/10 px 基本读不出来 —— 这份材料是讲课用的，不是屏幕上放大看的。
_PX = {"sec": 15, "box": 13, "lbl": 12, "sm": 11, "xs": 11, "xxs": 10}


def para(f, x, y, maxw, text, cls="xs", lh=None, fill=None, max_lines=None,
         anchor=None):
    """在 (x,y) 起画一段自动换行的文字，返回下一行的 y。

    anchor="end" 时 x 是右边界 —— 用于把一段说明右对齐贴到某个锚点上。
    """
    px = _PX.get(cls, 10)
    lh = lh or px + 5
    ls = wrap(text, maxw, px)
    if max_lines and len(ls) > max_lines:
        ls = ls[:max_lines]
    for i, l in enumerate(ls):
        f.t(x, y + i * lh, l, cls, fill, anchor)
    return y + len(ls) * lh


def plain(text):
    """剥掉行内标记，用于估算实际显示宽度。"""
    return "".join(c for c, _ in _tokenize(text))
