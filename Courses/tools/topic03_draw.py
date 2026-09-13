# -*- coding: utf-8 -*-
"""专题三 · 画图基元 —— **每一节的图都用这一套**。

⛔⛔ 2026-09-08 从 topic03-fig-rnn.py 抽出来。抽的理由不是「代码复用」，
   是**视觉一致性**：这门课的图必须看起来是同一个人画的。
   抄第二份的后果不是多写几行，是**两套图慢慢漂开，而且漂了不报错**
   ——&nbsp;跟 topic03_models.py 当初被抽出来是同一个理由。

⭐ 这套基元是照**专题二**的图逐条拆出来的。它的质感来自六样东西：

     ① header(title, sub, legend)   图例条 ——&nbsp;颜色一上来就有词典
     ② panel(...)                   面板套面板 ——&nbsp;外框 → 标题栏 → 内容
     ③ cell(main, sub=...)          盒子两行字 ——&nbsp;主标签 ＋ 一句说明
     ④ cell(..., grid=True)         矩阵纹理 ——&nbsp;权重看着像一块矩阵
     ⑤ spot(...)                    高亮带 ——&nbsp;圈出「差别在这儿」
     ⑥ band(kind, ...) ＋ src(...)  落点带 ＋ 📌 出处行

⛔ **别再画「一个矩形 ＋ 居中一个词」。**
⭐ 判据：**图里每一个盒子都该回答一个问题，而不是标一个名字。**

📌 三条护栏，别拆：
   · cell() 的宽度断言 ——&nbsp;文字放不下就报错，不许悄悄溢出
   · src() 的宽度断言 ——&nbsp;**文字溢出既不报错也不产生滚动条，只是被裁掉**
   · MINSZ 字号地板 ——&nbsp;专题二 42 张图渲染后最小 12.2px，这里对齐它
     ⚠️ 地板只拦住**经过 _sz() 的调用点**；默认参数最容易绕过去（栽过一次）。
"""
import io
import os
import re
import xml.dom.minidom

HERE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-08 对齐专题一的配色。现场：「专题三的配色感觉土了吧唧，
#     专题一的比较高端，照着专题一统一一下。」
#
# ⛔ 先量再改 ——&nbsp;把两讲所有 figure 里的颜色统计了一遍，"土"有三个**具体**原因：
#
#   ① **彩色文字直接用了主色（500 档）。**
#      专题一的彩色文字压倒性地用 **900 档深色变体**：
#      #174ea6×121、#0d652d×95、#b06000×74、#a50e0e×70；
#      主色 #1a73e8 在它那儿几乎只出现在**填充和描边**上（文字仅 13 次）。
#      而专题三反过来：#d93025×104、#1a73e8×102 全是**文字**。
#      ⭐ 高饱和主色大面积当正文色，就是"廉价感"的头号来源。
#
#   ② **框线太重。** 专题一的描边主力是 #dadce0×287 和 #e8eaed×232（极浅）；
#      专题三是 #bdc1c6×156 和 #5f6368×74 ——&nbsp;整整深一到两档。
#
#   ③ **多引进了一个体系外的紫。** #8430ce 不在 Google 那套色板里。
#      换成 Material 的 purple 500/900：#9334e6 ／ #681da8。
#
# 📌 于是这里改成 **Material 三档制**：50 浅底 · 500 主色 · 900 文字。
#   ⭐⭐ 关键在 `Fig.t()` 里做了**自动降档**：任何用 500 主色画文字的调用，
#     会自动换成对应的 900 ——&nbsp;**一处改，四张图全跟着变**，
#     而且以后写新图时想写错都难。⛔ 别把那个降档去掉。
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-08 第三刀 · 填充规则。现场问：「是不是所有的地方都不用填充？
#     现代化的文档风格是不是不用背景填充？外圈有颜色 ＋ 字有颜色就挺好。」
#
# ⛔ 我的意见：**方向对，但判据不是「填不填」，是「填的那块东西是不是信息本身」。**
#   量了专题一才敢这么说 ——&nbsp;它的浅色填充并没有变少，只是**长在别的地方**：
#
#       专题一：大块（>40k px²）**7 个**，小块 **136 个**
#       专题三：大块 **29 个**（最大的几条是 1400×178 的整条落点带），小块只有 56 个
#
#   ⭐⭐ 也就是说：**专题一把颜色用在「小而多」的元素上，
#     专题三把颜色刷在「大而少」的整条带子上。** 这才是"现代 / 土"的分界。
#
# 📌 于是定成规矩（两类，别再一刀切）：
#     · **是信息本身 → 填**：热力图格子、KV 条、类型色块、图例方块 ——&nbsp;
#       它们小而多，颜色在编码含义，去掉就少了一层信息。
#     · **只是容器 → 不填**：面板身子、整条落点带 ——&nbsp;
#       它们大而少，颜色只是装饰，**留白 ＋ 细边 ＋ 彩色标题字**就够了。
#
# ⭐ 落点带改成「左侧 4px 竖色条 ＋ 白底 ＋ 细灰框」——&nbsp;
#   这也正是今天大多数文档系统（GitHub alert、Material outlined）的做法。
# ══════════════════════════════════════════════════════════════════
BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
PU, CY, BR, INK = "#9334e6", "#00838f", "#b06000", "#202124"
GY2, LINE, LINE2, BG2 = "#80868b", "#dadce0", "#e8eaed", "#f8f9fa"

# 500 主色 → 900 文字色。⭐ 数值取自专题一实际用到的那几个。
INK900 = {
    "#1a73e8": "#174ea6",   # blue
    "#1e8e3e": "#0d652d",   # green
    "#d93025": "#a50e0e",   # red
    "#e8710a": "#b06000",   # orange / amber
    "#f9ab00": "#b06000",
    "#9334e6": "#681da8",   # purple
    "#8430ce": "#681da8",   # 旧紫，一并归位
    "#00838f": "#007b83",   # cyan
    "#12b5cb": "#007b83",
    "#a50e0e": "#a50e0e",
}
MINSZ = 11


def _sz(n):
    assert n >= MINSZ, "字号 %d 太小（地板 %d）" % (n, MINSZ)
    return n


def wpx(s, size=11.5):
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-14 现场：「图里那个字，不要太小，也不要太多」
#   落点带原来是 12px、靠**宽度断言**逼调用方自己拆行。于是两件事同时发生：
#     ① 字小 —— 全专题曝光最多的那批字都在这儿，投屏上看不清；
#     ② 调用方为了过断言，把一句话硬拆成两条，读起来更碎。
# ⭐ 治法是把「拆行」从调用方手里收回来：这里做**tspan 安全的自动折行**，
#   于是字号可以放心抬到 15px，调用方一行写多长都行。
# ⛔ 难点只有一个：断点可能落在 <tspan …> 里面。所以要维护一个开标签栈，
#   断行时**先把栈里的标签全闭上，下一行再原样重开** —— 否则生成的 SVG
#   不良构（save() 那道 XML 自检会抓到，但报的行号指向产物，回不到源头）。
# ══════════════════════════════════════════════════════════════════
_TAG = re.compile(r"<[^>]+>")
MONO_K = 1.12          # svgsm 是等宽字，实际步进比 wpx() 估的宽约一成


def wrap_rich(s, limit_px, size):
    """把一段**带 <tspan> 的**文字按像素宽折成多行，标签自动闭合 / 重开。"""
    out, cur, stack, w = [], [], [], 0.0
    per = float(size)

    def flush():
        if not cur and not stack:
            return
        out.append("".join(cur) + "".join("</%s>" % t[0] for t in reversed(stack)))
        del cur[:]
        cur.extend(t[1] for t in stack)

    i = 0
    while i < len(s):
        if s[i] == "<":
            j = s.index(">", i) + 1
            tag = s[i:j]
            cur.append(tag)
            if tag.startswith("</"):
                if stack:
                    stack.pop()
            elif not tag.endswith("/>"):
                stack.append((tag[1:].split()[0].rstrip(">"), tag))
            i = j
            continue
        if s[i] == "&":                       # &#160; 这类实体算一个字
            j = s.index(";", i) + 1
            ch, adv, i = s[i:j], 1.0, j
        else:
            ch, adv, i = s[i], (1.0 if ord(s[i]) > 0x2E80 else 0.55), i + 1
        # ⛔ 判的是「加上这个字会不会超」，不是「加完了超没超」——
        #   后者每行都会正好溢出一个字，而那一个字刚好在边界上最显眼。
        if w + adv * per > limit_px and w > 0:
            flush()
            w = 0.0
        cur.append(ch)
        w += adv * per
    if cur:
        out.append("".join(cur) + "".join("</%s>" % t[0] for t in reversed(stack)))
    return out or [""]


def sub(base, idx):
    """下标：`sub("S", "t−1")` → `S` 加一个真下标。

    ⛔ 2026-09-14 二轮学生审稿：图里原来直接写 `S_{t-1}`、`ℝ^(d_v×d_k)`，
      **SVG 不渲染 LaTeX，于是全专题最核心的那个递推式长得像没编译的稿子**。
    ⭐ 判据：**图里不能出现「等着被别的东西渲染」的记法** ——
      SVG 里只有 dy 位移是到处都靠得住的做法，baseline-shift 各家不一。
    ⚠️ 用 dy 必须**成对**：移下去多少就要移回来多少，否则后面的字全歪。
    """
    return ('%s<tspan font-size="0.72em" dy="3">%s</tspan>'
            '<tspan dy="-3"></tspan>' % (base, idx))


def sup(base, idx):
    """上标，同 `sub` 的注意事项。"""
    return ('%s<tspan font-size="0.72em" dy="-4">%s</tspan>'
            '<tspan dy="4"></tspan>' % (base, idx))


_ARXIV = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?![\d.])")


def _svg_linkify(svg):
    """把 <text> 里的 arXiv 编号变成可点的 SVG 链接。

    ⛔ 只在 `<text>…</text>` 内部替换 ——&nbsp;元素属性里也可能出现四位小数
      （viewBox、坐标），套进去就毁了。
    ⚠️ 加下划线是必要的：SVG 里的 <a> **不会**像 HTML 那样自动变色变下划线，
      不标出来读者根本不知道它能点。
    """
    def _one(m):
        return ('<a href="https://arxiv.org/abs/%s" target="_blank">'
                '<tspan text-decoration="underline">%s</tspan></a>'
                % (m.group(1), m.group(1)))

    def _intext(m):
        return m.group(1) + _ARXIV.sub(_one, m.group(2)) + m.group(3)

    return re.sub(r"(<text\b[^>]*>)(.*?)(</text>)", _intext, svg, flags=re.S)


class Fig(object):
    """一张 SVG。高度不写死，收尾按真实落点回填。"""

    def __init__(self, w, aria):
        self.w, self.aria, self.p = w, aria, []
        self.p.append("")          # svg 开标签占位
        self._pan = None      # 当前面板 (top, bottom)
        self._over = []       # 画到面板外面去的记录（越界自检）

    # ── 原子 ────────────────────────────────────────────────────
    def t(self, x, y, s, fill=INK, bold=False, size=11.5, anchor=None,
          cls="svgsm", mono=False, w=None):
        # ⭐ 自动降档：用 500 主色画文字 → 换成对应的 900 深色变体。
        #   这一行就是「照着专题一统一配色」的全部实现。⛔ 别去掉。
        fill = INK900.get(fill, fill)
        # ⛔ SVG 是 XML：`&nbsp;` 是未定义实体。原先要等到 save() 那道 XML 自检
        #   才炸，报的是 `undefined entity: line 113, column 148` ——&nbsp;
        #   行号指向**生成出来的 SVG**，回不到写错的那一行。
        # ⭐ 判据：**能在源头报错，就别留到产物上报错。**
        assert "&nbsp;" not in s, "图里不能写 &nbsp;（XML 未定义实体），改用 &#160;：%s" % s[:40]
        # ⛔ 这几个符号在渲染字体里没有字形，出来是小方块，**而且不报错**。
        #   （⏸ 2026-09-13 在 fig3-why-softmax 上栽过一次。）
        for _bad in "\u23f8\u23f1\u23f3\u23ef":
            assert _bad not in s, "这个符号渲染不出来（会变成小方块）：%r" % _bad
        # ⛔ <tspan> 跨两次 t() 调用是不可能的（生成的是两个独立 <text>）。
        #   不配平的话要等到 save() 的 XML 自检才报，而那时行号指向产物。
        # ⛔ markdown 的 **粗体** 在 SVG 里不渲染，会**原样印出来**且不报错。
        #   这门课栽过两次（第二次还把解释写进了副标题，于是解释本身也印出来了）。
        #   ⭐ 2026-09-13 学生审稿又在两张图里各发现一处 —— 做成断言，别再靠人眼。
        assert "**" not in s, "图里不能写 **粗体**（SVG 不渲染，会原样印出来）：%s" % s[:46]
        # ⛔⛔ 2026-09-14：<em>／<i>／<b> 这几个是 HTML 解析器的
        #   **foreign-content 逃逸标签**。图是 inline 进 HTML 的，浏览器读到
        #   <em> 会当场**退出 SVG 模式**，后面的内容全部丢弃 ——
        #   fig3-when-axis 因此在浏览器里少了最后 200px（两条落点带 ＋ 出处行）。
        #   ⭐ 而 SVG 文件本身是良构 XML，写盘自检、几何 lint 全都看不出来。
        for _esc in ("<em>", "<i>", "<b>", "<p>", "<br>", "<font"):
            assert _esc not in s, (
                "图里不能写 %s ——&nbsp;它会让 HTML 解析器退出 SVG 模式，"
                "后面的内容全丢：%s" % (_esc, s[:46]))
        assert s.count("<tspan") == s.count("</tspan>"), \
            "<tspan> 没配平（一个 tspan 不能跨两次 t() 调用）：%s" % s[:50]
        # ⭐ 2026-09-09：传了 w 就当场校宽。lines() / src() / band() 早就有这道
        #   护栏，唯独单行的 t() 没有 ——&nbsp;而单行才是最常写着写着就顶出去的。
        #   ⛔ 文字溢出**不报错、不产生滚动条**，只是被裁掉。
        if w is not None:
            # ⛔ 2026-09-13：原先按**含标签的原串**量宽，于是任何一处
            #   <tspan font-weight="700"> 都被当成 30 多个可见字符，
            #   好好的一行被误判成溢出。lines() / band() / src() 三个基元
            #   早就先 strip 再量，唯独 t() 没有 —— ⭐ 同一套护栏里
            #   **有一个成员判据不一样，就等于那条判据在这里不成立**。
            need = wpx(re.sub(r"<[^>]+>", "", s), size)
            assert need <= w, ("「%s」要 %dpx，只给了 %dpx ——&nbsp;拆行或加宽"
                               % (re.sub(r"<[^>]+>", "", s)[:26], need, w))
        self._note_ink(y)
        st = ["font-size:%.1fpx" % _sz(size)]
        if mono:
            # ⛔ 这里必须用单引号：style 是双引号属性，里面再写双引号会把属性提前闭合，
            #   生成的 SVG 直接不良构（写盘前那道 XML 自检就是抓这个的）。
            st.append("font-family:'Roboto Mono',ui-monospace,monospace")
        self.p.append('<text class="%s" x="%d" y="%d" fill="%s"%s style="%s">%s</text>'
                      % (cls, x, y, fill,
                         ' text-anchor="%s"' % anchor if anchor else '',
                         ";".join(st),
                         '<tspan font-weight="700">%s</tspan>' % s if bold else s))

    def box(self, x, y, w, h, fill="#fff", stroke=LINE, r=6, sw=1, dash=None,
            shadow=False):
        # 📌 stroke 默认就是 #dadce0 ——&nbsp;跟专题一的主力描边一致（那边 ×287）。
        #   ⛔ 别把默认改深；要强调就显式传主色，不要靠加重灰线。
        self.p.append('<rect x="%s" y="%s" width="%s" height="%s" rx="%d" fill="%s" '
                      'stroke="%s" stroke-width="%s"%s%s/>'
                      % (x, y, w, h, r, fill, stroke, sw,
                         ' stroke-dasharray="%s"' % dash if dash else '',
                         ' filter="url(#sh)"' if shadow else ''))

    def line(self, x1, y1, x2, y2, col=GY2, sw=1.3, dash=None, arrow=True):
        self.p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" '
                      'stroke-width="%s" stroke-linecap="round"%s%s/>'
                      % (x1, y1, x2, y2, col, sw,
                         ' stroke-dasharray="%s"' % dash if dash else '',
                         ' marker-end="url(#ah-%s)"' % col.lstrip("#") if arrow else ''))
        if arrow:
            self.marks.add(col)

    def poly(self, d, fill="#fff", stroke="none", sw=1.0):
        """闭合多边形，**能填色** ——&nbsp;path() 是 fill:none 的描边版，两个别混。
        ⛔ 2026-09-09 踩过：拿 path() 去画「收口的细颈」，传了填充色当描边色，
          画出来只有一条边。⭐ 名字里带 path 不代表它会填。

        ⛔⛔ 2026-09-14：上一轮给 path() 补了「接受点列表」，**漏了这个姐妹基元** ——
          当天下午画房子屋顶就又踩进去，d="[(84, 120), …]"，屋顶没画出来。
        ⭐ 判据：**同一个毛病，先去姐妹基元上找一遍。**
          它们签名相同、职责相邻，写错的人不会只在一个上面写错。
        """
        if isinstance(d, (list, tuple)):
            d = ("M " + " L ".join("%.2f %.2f" % (x, y) for x, y in d) + " Z")
        assert isinstance(d, str) and d[:1] in "Mm", \
            "poly() 的 d 必须是 SVG 路径串或点列表，收到：%r" % (d,)
        self.p.append('<path d="%s" fill="%s" stroke="%s" stroke-width="%s"/>'
                      % (d, fill, stroke, sw))

    def path(self, d, col=GY2, sw=1.3, dash=None, arrow=True):
        # ⛔⛔ 2026-09-14 审图抓到的最重一条：有 4 个调用方直接传**点列表**
        #   （fig3-info-law 那两条曲线、dsa-why 的「鸡生蛋」回环、swa-why 和
        #   attn-invented 各一根箭头）。于是 d="[(120.0, 184.0), …]" ——
        #   ⭐ SVG 仍然是良构的 XML，写盘自检过；浏览器解析不了那个 d，
        #     **什么都不画，也不报错**。info-law 的主图整个是空坐标系，
        #     而旁边那个文字框还在解释一张不存在的图。
        # ⭐ 判据：**「产物合法」和「产物正确」是两回事。**
        #   凡是接受 DSL 字符串的基元，都要么接受结构化输入，要么当场校验。
        if isinstance(d, (list, tuple)):
            d = "M " + " L ".join("%.2f %.2f" % (x, y) for x, y in d)
        assert isinstance(d, str) and d[:1] in "Mm", \
            "path() 的 d 必须是 SVG 路径串或点列表，收到：%r" % (d,)
        self.p.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s" '
                      'stroke-linecap="round"%s%s/>'
                      % (d, col, sw,
                         ' stroke-dasharray="%s"' % dash if dash else '',
                         ' marker-end="url(#ah-%s)"' % col.lstrip("#") if arrow else ''))
        if arrow:
            self.marks.add(col)

    marks = set()

    # ── ① 标题区 ＋ 图例条 ───────────────────────────────────────
    # ⛔⛔ 2026-09-14：八张图有「内容被后画的面板底色盖住」——
    #   f.panel() 声明的高度 < 实际画到的位置，下一块面板的白底直接糊上去，
    #   盖掉的还净是关键句（tpu-fix 三条落点带、swa-why 的「柱子按对数画」…）。
    # ⭐ 这条教训本来只写在 topic03-fig-when-axis.py 一个文件的注释里 ——
    #   **写在注释里的判据只在那个文件生效**。做成基元级的断言才是真的修了。
    def _note_ink(self, y):
        """记下「画到了哪一行」，供面板越界自检用。"""
        if self._pan is not None and y > self._pan[1] + 2:
            self._over.append((self._pan[0], y, self._pan[1]))

    def header(self, title, sub, legend=None, y=22):
        # ⭐ 2026-09-14 抬字号：标题 16.5→20，副标题 12→15，图例 11→14。
        #   现场原话「图里那个字不要太小 …… 打到屏幕上去分享」——
        #   ⛔ 副标题和图例是**读图前必须先读的两样**，它们小等于整张图门槛高。
        self.t(0, y, title, INK, size=20, cls="svglbl")
        yy = y + 28
        if sub:
            for r in wrap_rich(sub, self.w - 20, 15 * MONO_K):
                self.t(0, yy, r, GY, size=_sz(15))
                yy += 22
            yy += 2
        if legend:
            x = 0
            for col, lab in legend:
                self.box(x, yy - 11, 14, 14, col, col, 3)
                self.t(x + 21, yy, lab, GY, size=_sz(14))
                x += 21 + wpx(lab, 14) + 26
            yy += 20
        return yy + 8

    # ── ② 带标题栏的面板 ─────────────────────────────────────────
    # ⛔⛔ 2026-09-08：标题栏原先是**实心主色 ＋ 白字**。
    #   对着专题一逐张看，它**整张图里没有一块大面积实心色** ——&nbsp;
    #   颜色只出现在①细边框 ②小色块 ③文字（而且是 900 深色档）。
    #   ⭐ 实心色块 ＋ 白字是「仪表盘」观感，正是「土」的第四个来源。
    #   📌 改成：标题栏用 50 浅底，标题字用主色（会被 t() 自动降到 900），
    #     底部一条细分隔线。**外框保留主色细边** ——&nbsp;专题二就是这么做的，
    #     那张 TensorCore 图被认可过。
    # ⛔⛔ 2026-09-08 第二刀。第一刀只换了标题栏（实心色 → 浅底），
    #   现场回话「怎么感觉跟之前没什么变化」——&nbsp;**对的，我只改了一小块。**
    # ⭐ 再量一次，这次量**描边的中性/彩色配比**（前一次量的是面积，两讲差不多，
    #   所以那一轮没找到真凶）：
    #     专题一 —— 中性 #dadce0×286 ＋ #e8eaed×231 ＋ #9aa0a6×78 ≈ 七成以上
    #     专题三 —— 饱和主色约占一半（红 66 / 蓝 63 / 绿 47 / 紫 39 …）
    #   ⭐⭐ **大面板全部用饱和主色描边，三四个摞在一起就是「吵」。**
    # 📌 改法：**外框一律中性浅灰**，颜色身份改由「顶部一条 4px 彩带 ＋ 标题文字」承担
    #   ——&nbsp;跟总纲图那六张卡同一套做法。⛔ 别再把主色传给大面板的外框。
    def panel(self, x, y, w, h, title, col=LINE, fill="#fff", tag=None,
              tint=None, sub=None):
        """外框 ＋ 顶部标题栏。tag 是右上角的小注（出处 / 口径）。"""
        self._pan = (title, y + h)      # 越界自检：记下这块面板的下沿
        # ⛔ 标题栏也不再填色（见文件头「填充规则」）。颜色身份只剩两样：
        #   顶部 4px 彩带 ＋ 彩色标题字。
        # ⛔⛔ 2026-09-14：外框原来是 **白色实心**。页面也是白的，所以看不出来 ——
        #   但它会把**上一块面板画出界的内容整段擦掉**（实测 fig3-tpu-fix 的
        #   「decode MBU 86% / prefill MFU 73%」那行被削掉一半）。
        # ⭐ 判据：**跟背景同色的填充不是「没填」，它照样是一次覆盖。**
        #   改成 none 之后视觉零变化，而被盖住的内容会重新露出来 ——
        #   然后几何 lint 就能把它当成撞车抓到（静默 → 可见）。
        self.box(x, y, w, h, "none", LINE, 9)
        if col != LINE:
            self.box(x, y, w, 4, col, col, 2)
            self.box(x, y + 2, w, 4, "#fff", "#fff", 0)
        self.line(x, y + 30, x + w, y + 30, LINE, 1, arrow=False)
        self.t(x + 14, y + 20, title, col if col != LINE else INK,
               bold=True, size=13.5, cls="svglbl")
        if sub:
            self.t(x + 16 + wpx(title, 13.5) + 12, y + 20, sub, GY, size=_sz(11))
        if tag:
            self.t(x + w - 14, y + 20, tag, GY2, size=_sz(11), anchor="end")
        return y + 30

    # ── ③ 两行字的盒子 —— 主标签 ＋ 一句说明 ─────────────────────
    def cell(self, x, y, w, h, main, sub=None, col=LINE, fill="#fff",
             size=12, r=6, grid=False, dash=None):
        need = wpx(main, size) + 16
        assert w >= need, "「%s」要 %dpx，格子只有 %dpx" % (main, need, w)
        self.box(x, y, w, h, fill, col, r, dash=dash)
        if grid:                     # ④ 矩阵纹理：让「一块权重」看着像一块矩阵
            self.box(x + 1, y + 1, w - 2, h - 2, "url(#grid)", "none", r - 1)
        if sub:
            self.t(x + w / 2.0, y + h / 2.0 - 1, main, col, True, size, "middle")
            self.t(x + w / 2.0, y + h / 2.0 + 14, sub, GY, size=_sz(11),
                   anchor="middle", mono=True)
        else:
            self.t(x + w / 2.0, y + h / 2.0 + 4, main, col, True, size, "middle")

    # ── 列头 / 行标签 ───────────────────────────────────────────
    def colhead(self, x, y, main, sub=None, anchor=None):
        self.t(x, y, main, GY, bold=True, size=_sz(12), anchor=anchor)
        if sub:
            self.t(x, y + 16, sub, GY2, size=_sz(11), anchor=anchor)

    def rowlab(self, x, y, main, sub=None, col=INK):
        self.t(x, y, main, col, bold=True, size=13, cls="svglbl")
        if sub:
            self.t(x, y + 17, sub, GY, size=_sz(11))

    # ── ⑤ 高亮竖带：圈出「整张图的差别在这一列」 ──────────────────
    def spot(self, x, y, w, h, col="#f1f3f4"):
        self.box(x, y, w, h, col, "none", 8)

    # ── ⑥ 底部落点带 ────────────────────────────────────────────
    KIND = {"ok": (GR, "#e6f4ea", "⭐"), "warn": (OR, "#fef7e0", "⚠️"),
            "info": (BL, "#e8f0fe", "⭐⭐"), "bad": (RD, "#fce8e6", "⛔")}

    def band(self, y, kind, title, lines, w=None):
        """落点带。⛔ **不填色** ——&nbsp;白底 ＋ 细灰框 ＋ 左侧 4px 彩色竖条。
        见文件头「填充规则」：容器不填，只有承载信息的小元素才填。"""
        self._pan = None            # 落点带在面板外面，合法
        col, fill, icon = self.KIND[kind]
        w = w or self.w
        # ⭐ 2026-09-14：字号 12 → 15，换行改成自动（见 wrap_rich 的说明）。
        #   ⛔ 原来靠宽度断言逼调用方拆行 —— 那既让字小，又让句子被拆碎。
        SZ, LH = 15, 24
        rows = []
        for ln in lines:
            # ⛔ svgsm 是**等宽字体**，Roboto Mono 的 ASCII 步进约 0.60em，
            #   而 wpx() 按 0.55 估 —— 英文长句一累积就差出一整行。
            #   ⭐ 所以折行时按 MONO_K 倍的字号去量，留出这 10%。
            rows.extend(wrap_rich(ln, w - 40, SZ * MONO_K))
        h = 40 + len(rows) * LH + 10
        # ⛔⛔ 2026-09-14 第三次撞同一条：**跟背景同色的填充照样是一次覆盖**。
        #   面板改透明之后，落点带的白底接着干同样的事 ——&nbsp;
        #   它把上一块面板画出界的最后一行整段擦掉（fig3-rnn-hw 实测）。
        # ⭐ 判据升级版：**一个毛病在基元层出现过，就去所有姐妹基元上找一遍。**
        #   这套库里会画大矩形的有三处：panel / band / cell —— 一次全查完。
        self.box(0, y, w, h, "none", LINE, 9)
        self.box(0, y, 4, h, col, col, 2)
        self.box(2, y, 3, h, "#fff", "#fff", 0)
        self.t(20, y + 27, "%s %s" % (icon, title), col, bold=True, size=17,
               cls="svglbl")
        for i, ln in enumerate(rows):
            self.t(20, y + 58 + i * LH, ln, col, size=_sz(SZ))
        return y + h

    def lines(self, x, y, w, rows, size=11, lh=17, fill=None, bold_first=False,
              first_fill=None):
        """在宽度 w 内画多行文字。⛔ **每一行都过宽度断言** ——
        这套图前后栽过三次「文字溢出既不报错也没滚动条，只是被裁掉」。
        ⭐ 换行点由调用方给（rows 是已经拆好的行），不做自动断词：
          自动断词会把 <tspan> 拦腰截断，那种坏法比溢出还难查。
        """
        for i, ln in enumerate(rows):
            need = wpx(re.sub(r"<[^>]+>", "", ln), size)
            assert need <= w, "「%s」要 %dpx，只有 %dpx —— 拆行" % (
                re.sub(r"<[^>]+>", "", ln)[:22], need, w)
            # ⛔ first_fill 是给「首行用主色」用的。原先调用方的做法是**再画一遍首行**
            #   ——&nbsp;同一段文字叠两层，版面体检当场报「撞车 6」。
            #   ⭐ 重复绘制在纸面上完全看不出来，只有几何探针看得见。
            self.t(x, y + i * lh, ln,
                   (first_fill or fill or GY) if i == 0 else (fill or GY),
                   bold_first and i == 0, _sz(size))
        return y + len(rows) * lh

    def src(self, y, *lines):
        """📌 出处行（可多行）—— 灰字小注，跟专题二一致。

        ⛔ 带宽度自检：2026-09-08 实测有一行冲出了右边界，而**文字溢出既不报错
          也不产生滚动条**，只是被裁掉 —— 页面上看只是「这句话没写完」。
        """
        # ⭐ 2026-09-14：11 → 14px，同样改成自动折行（续行缩进对齐）。
        self._pan = None            # 出处行同理
        SZ, LH = 14, 21
        k = 0
        for i, ln in enumerate(lines):
            # ⛔ 前缀（📌 ／ 全角缩进）是**折完行才加上去的** —— 折行限宽里
            #   必须先把它减掉，否则每一行都正好多出一个前缀的宽度。
            for j, r in enumerate(wrap_rich(ln, self.w - 46 - wpx("　　", SZ),
                                            SZ * MONO_K)):
                self.t(0, y + k * LH,
                       ("📌 " if (i == 0 and j == 0) else "　　") + r,
                       GY2, size=_sz(SZ))
                k += 1
        return y + k * LH + 1

    # ── 收尾 ────────────────────────────────────────────────────
    def save(self, name, bottom):
        # ⚠️ 越界自检：只报告不中止。⛔ 直接断言会一次打挂 8 张图，
        #   那样只会逼人把断言关掉 —— 先让它可见，再一张一张修。
        if self._over:
            seen = {}
            for ttl, y, bot in self._over:
                seen[ttl] = max(seen.get(ttl, 0), y - bot)
            for ttl, d in sorted(seen.items(), key=lambda kv: -kv[1]):
                print("   ⚠️ %s：面板「%s」声明的下沿被越过 %d px（内容会被下一块盖住）"
                      % (name, ttl[:22], d))
        self.p.append('</svg>')
        marks = "".join(
            '<marker id="ah-%s" viewBox="0 0 10 10" refX="8.5" refY="5" '
            'markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">'
            '<path d="M 0 1 L 9 5 L 0 9 z" fill="%s"/></marker>'
            % (c.lstrip("#"), c) for c in sorted(self.marks))
        self.p[0] = (
            '<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="%s">'
            '<defs>%s'
            '<pattern id="grid" width="7" height="7" patternUnits="userSpaceOnUse">'
            '<path d="M 7 0 L 0 0 0 7" fill="none" stroke="#00000014" '
            'stroke-width="0.8"/></pattern>'
            '<filter id="sh" x="-8%%" y="-8%%" width="118%%" height="124%%">'
            '<feDropShadow dx="0" dy="1.5" stdDeviation="2.2" '
            'flood-color="#202124" flood-opacity="0.10"/></filter>'
            '</defs>' % (self.w, bottom, self.aria, marks))
        s = "\n".join(self.p)
        # ⭐⭐ 2026-09-14 现场点的：「引用的那些论文得在教材里边，
        #   把可点击的 link 都放里边，有愿意多学的人可以去点开看。」
        #   正文那边由 course_links.py 后处理，**但图里的出处它够不着** ——
        #   那个后处理刻意跳过 <svg>（HTML 的 <a> 进不去 SVG 的文本流）。
        # ⭐ 所以图这边自己来：在写盘前把 arXiv 编号包成 **SVG 自己的 <a>**。
        # ⚠️ 两个前提，缺一个就白做：
        #   ① 必须是**内联**的 SVG（本课就是内联进 HTML 的，所以点得动）；
        #   ② `<a>` 要包在 `<text>` **里面**（包在外面 Chrome 不给点）。
        s = _svg_linkify(s)
        xml.dom.minidom.parseString(s.encode("utf-8"))
        io.open(os.path.join(HERE, name), "w", encoding="utf-8").write(s)
        print("ok  %s  %d×%d" % (name, self.w, bottom))
