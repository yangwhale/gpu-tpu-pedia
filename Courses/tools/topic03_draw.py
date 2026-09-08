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


class Fig(object):
    """一张 SVG。高度不写死，收尾按真实落点回填。"""

    def __init__(self, w, aria):
        self.w, self.aria, self.p = w, aria, []
        self.p.append("")          # svg 开标签占位

    # ── 原子 ────────────────────────────────────────────────────
    def t(self, x, y, s, fill=INK, bold=False, size=11.5, anchor=None,
          cls="svgsm", mono=False):
        # ⭐ 自动降档：用 500 主色画文字 → 换成对应的 900 深色变体。
        #   这一行就是「照着专题一统一配色」的全部实现。⛔ 别去掉。
        fill = INK900.get(fill, fill)
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

    def path(self, d, col=GY2, sw=1.3, dash=None, arrow=True):
        self.p.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s" '
                      'stroke-linecap="round"%s%s/>'
                      % (d, col, sw,
                         ' stroke-dasharray="%s"' % dash if dash else '',
                         ' marker-end="url(#ah-%s)"' % col.lstrip("#") if arrow else ''))
        if arrow:
            self.marks.add(col)

    marks = set()

    # ── ① 标题区 ＋ 图例条 ───────────────────────────────────────
    def header(self, title, sub, legend=None, y=22):
        self.t(0, y, title, INK, size=16.5, cls="svglbl")
        yy = y + 22
        if sub:
            self.t(0, yy, sub, GY, size=_sz(12))
            yy += 20
        if legend:
            x = 0
            for col, lab in legend:
                self.box(x, yy - 9, 11, 11, col, col, 2)
                self.t(x + 17, yy, lab, GY, size=_sz(11))
                x += 17 + wpx(lab, 11) + 22
            yy += 16
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
        # ⛔ 标题栏也不再填色（见文件头「填充规则」）。颜色身份只剩两样：
        #   顶部 4px 彩带 ＋ 彩色标题字。
        self.box(x, y, w, h, "#fff", LINE, 9)
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
        col, fill, icon = self.KIND[kind]
        w = w or self.w
        h = 34 + len(lines) * 21 + 8
        self.box(0, y, w, h, "#fff", LINE, 9)
        self.box(0, y, 4, h, col, col, 2)
        self.box(2, y, 3, h, "#fff", "#fff", 0)
        self.t(20, y + 24, "%s %s" % (icon, title), col, bold=True, size=13.5,
               cls="svglbl")
        for i, ln in enumerate(lines):
            # ⛔ 跟 src() 同一条：**文字溢出既不报错也不产生滚动条，只是被裁掉**。
            #   2026-09-08 实测又栽了一次（Shazeer 那句英文引文冲出右边界）——
            #   ⭐ 所以凡是「一整行文字」的基元，都必须自带宽度断言。
            need = wpx(re.sub(r"<[^>]+>", "", ln), 12) + 38
            assert need <= w, "落点带第 %d 行要 %dpx，只有 %dpx —— 拆行" % (
                i + 1, need, w)
            self.t(20, y + 48 + i * 21, ln, col, size=_sz(12))
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
        for i, ln in enumerate(lines):
            w = wpx(re.sub(r"<[^>]+>", "", ln), 11) + 22
            assert w <= self.w, "出处第 %d 行要 %dpx，超出画布 %dpx —— 拆行" % (
                i + 1, w, self.w)
            self.t(0, y + i * 17, ("📌 " if i == 0 else "　　") + ln, GY2, size=_sz(11))
        return y + len(lines) * 17 + 1

    # ── 收尾 ────────────────────────────────────────────────────
    def save(self, name, bottom):
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
        xml.dom.minidom.parseString(s.encode("utf-8"))
        io.open(os.path.join(HERE, name), "w", encoding="utf-8").write(s)
        print("ok  %s  %d×%d" % (name, self.w, bottom))
