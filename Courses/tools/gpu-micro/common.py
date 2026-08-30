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
.xs{font:400 10px "Noto Sans CJK SC",sans-serif;fill:#5f6368}
.xxs{font:400 9px "Noto Sans CJK SC",sans-serif;fill:#5f6368}
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


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
        a = f' text-anchor="{anchor}"' if anchor else ""
        f = f' fill="{fill}"' if fill else ""
        wt = f' font-weight="{weight}"' if weight else ""
        self.p.append(f'<text x="{x}" y="{y}" class="{cls}"{f}{a}{wt}>{s}</text>')

    def line(self, x1, y1, x2, y2, c=SUB, sw=1.5, marker=None, dash=None):
        m = f' marker-end="url(#{marker})"' if marker else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{sw}"{m}{d}/>')

    def path(self, d, c=SUB, sw=1.5, fill="none", marker=None, dash=None):
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
        """items: [(color, text), ...] —— 与 TPU 图相同的色块图例。"""
        x = x0
        for c, txt in items:
            self.rect(x, y - 10, 11, 11, fill=c, rx=2)
            self.t(x + 17, y, txt, "sm")
            x += 17 + 11.5 * _wlen(txt) + 22

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
_MARK = {"b": ('<tspan font-weight="700" fill="#202124">', "</tspan>", 1.0),
         "code": ('<tspan class="mono" fill="#1967d2">', "</tspan>", 0.92),
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


_PX = {"sec": 15, "box": 13, "lbl": 12, "sm": 11, "xs": 10, "xxs": 9}


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
