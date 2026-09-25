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


# ⭐ 图内装饰性符号 ——&#160;见 t() 里那段注释。
#   ⚠️ "⚠️" 是两个码点（U+26A0 ＋ U+FE0F），必须排在裸 "⚠" 前面先删。
_DECOR = ("⭐", "⛔", "⚠️", "⚠", "✅", "❗", "🆕", "📌", "💡", "🔬", "❓", "🎯")


# ⭐ 这几个浅色只当过「卡片底」用 ——&#160;从全站填充统计里挑出来的。
_TINTS = frozenset(("#fce8e6", "#e8f0fe", "#e6f4ea", "#f3e8fd",
                    "#fef7e0", "#f1f3f4", "#f6f7f8", "#edeff1"))


def _flatten(w, h, fill, stroke):
    """大面积的**卡片底色**改成白 ——&#160;分组交给边框和留白，别靠一块色。

    ⛔⛔ **填充色在好几张图里是数据，不能一刀切**：
      · `fig4-underflow` Ⓐ 的绿色区间带 ＝ 33 个数量级的**范围**
      · `fig4-circuit` Ⓒ 的 175 个粉方块 ＝ **数量**
      · `fig4-muon` Ⓑ 对角线那几格 ＝ **位置**
    ⭐ 区分判据（两条都要满足才当装饰）：
      ① **面积够大**（w ≥ 200 且 h ≥ 60）——&#160;小块多半是数据标记；
      ② **边框跟填充不同色** ——&#160;区间带那种是 `fill == stroke`，
         它压根没有「框」，只是一片染色区域。
    ⭐⭐ 判据（RQ3 那轮立的，这里第二次用上）：
      **改一个视觉属性之前，先问它在哪些地方承载着语义。**
    """
    if fill not in _TINTS:
        return fill
    if str(stroke).lower() == str(fill).lower():
        return fill                      # 染色区域，不是卡片
    # ⭐ 中性灰底一律转白，**不看尺寸**：灰不承载正／负，它只可能是装饰。
    #   （第一版按 h ≥ 60 卡，漏掉了 `fig4-circuit` 那几个 104×52 的灰底输入框
    #    ——&#160;而那恰恰是诊断里点名「像表单输入框」的那一类。）
    #   ⛔ 语义色（红／绿／蓝／紫／黄）的小块才可能是数据，那些仍按尺寸判。
    if fill in ("#f1f3f4", "#f6f7f8", "#edeff1"):
        return "#fff"
    try:
        if float(w) < 200 or float(h) < 60:
            return fill                  # 小块 ＝ 数据标记
    except (TypeError, ValueError):
        return fill
    return "#fff"


def _hair(sw):
    """⭐⭐ 把**辅助线**的线宽吸附到两档，让主干和背景拉开层次。

    ⛔⛔ **只动 < 1.5 的**。原因很具体：**线宽在有些图里是数据**
      ——&#160;`fig4-circuit` Ⓑ 用 `1.8 + 4.8·|g|/4` 把梯度大小编码成粗细，
      范围 1.8–6.6。一旦无脑吸附，那张图的论点当场毁掉。
    ⭐ 判据：**改一个视觉属性之前，先问它在哪些地方承载着语义。**

    实测改前：48 个线宽档位、10991 条线，其中 60% 是 1.0、另有一堆
    0.8 / 1.2 / 1.3 / 1.4 混着用 ——&#160;辅助线自己就先毛糙了。
    """
    try:
        v = float(sw)
    except (TypeError, ValueError):
        return sw
    if v >= 1.5:
        return sw            # 主干／有语义的，一律不碰
    return 0.75 if v <= 0.9 else 1.0


def _undecorate(s):
    for ch in _DECOR:
        s = s.replace(ch, "")
    # 删掉符号后常留下双空格／行首空格
    s = re.sub(r"[ \u3000]{2,}", " ", s).strip()
    return s


def _sz(n):
    assert n >= MINSZ, "字号 %d 太小（地板 %d）" % (n, MINSZ)
    return n


# ⭐⭐ 2026-09-14：MINSZ 有地板，却一直没有天花板，于是字号只能单向漂。
#   实测专题三 vs 专题二（两边画布都是 1400 宽，可以直接比）：
#     · 图内正文中位　　　15.0　vs　14.0　　——&#160;这一档没问题
#     · 「20 字以上的长句」里 ≥17px 的　22% vs 6%，每张图 4.79 条 vs 1.40 条
#     · 最夸张的到了 24px 的整句话
#   ⭐ 单看任何一行都不觉得大，是**整体比例**塌了：正文和落点句拉不开档，
#     于是每一句都在喊，读者反而找不到该看哪一句 ——&#160;就是「老年机」的观感。
#   ⛔ 判据不能用字号本身（公式、算式、单个大数字本来就该大），要用
#     **「这是不是一句话」**：汉字 ≥12 个 ＝ 正文，短标签和公式都到不了。
MAXPROSE = 17          # 正文长句的天花板；短标签 / 公式 / 数字不受这条管
PROSE_CJK = 12         # 汉字到这个数就算「一句话」


def _cjk(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&#?\w+;", "", s)
    return sum(1 for c in s if "一" <= c <= "鿿")


# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-14 R41：wpx 原来只有一条规则 ——`ord(ch) > 0x2E80` 就算一个全角，
#   否则 0.55。它**两个方向都错**，而且错得不对称：
#     · **低估**（真正会撞车的那一侧）—— ①②③④、⭐⛔✅、← → ↓、·、×、…、——
#       这些的 ord 都 **小于** 0x2E80，于是全被当成 0.55 个字宽。
#       ⭐ 是最要命的：实测 **1.26**，比一个汉字还宽，而这套图里到处是 ⭐。
#     · **高估** —— 。（）：、；「」 这些全角标点，Chrome 会做**标点压缩**，
#       实测只有 **0.53**，模型却给 1.00。
#   两边部分抵消，所以四个月没人发现；直到 R40 的面板 ④ 标题把小注顶穿了才露出来
#   （标题里一个 ④ 加两个 ——，一共少算约 18 px，正好是那次重叠的量）。
#
# ⭐⭐ 现在的表是**量出来的，不是估的**。复现方法（改字体或加新符号时重跑）：
#   ① 从 tools/*.svg 的文本节点里收集全部字符；
#   ② 用 playwright 起 Chromium，`font-family` 用讲义页那一套
#      （"Noto Sans CJK SC","PingFang SC",sans-serif），
#      **每个字符夹在两个汉字中间量**：width("中X中") − width("中中")；
#   ③ 只把「与下面那条兜底规则相差 > 0.03」的字符入表。
#
# ⛔⛔ 第 ② 步一开始是 `c.repeat(20)` 量的，**那个量法错了，而且错得很像对的**：
#   Chrome 会做**标点压缩**（相邻的全角标点各收掉一半），于是
#   。、，：；「」（）—— 这一整批量出来全是 **0.53**，我就照着写进了表里。
#   结果是四张图的面板小注往左缩、直接压在标题上 ——
#   **改之前一处不撞，改之后撞了四处。**
#   换成夹在汉字中间量，这批全部回到 **1.00**，四处撞车同时消失。
# ⭐⭐ 判据：**量一个字符的宽度，必须把它放进它真实出现的上下文里量。**
#   让同类字符彼此相邻（repeat / 拼成一串）会触发排版引擎的上下文规则，
#   量到的是「一串标点」的宽度，不是「一个标点夹在正文里」的宽度。
# ⭐ 这次是撞车数从 0 变 4 把我抓住的。**基线是干净的，才有资格当判据** ——
#   如果原来就有一堆撞车，多这四处根本看不出来。
#   实测校验：新模型对整句的误差 **0% ~ +8%（只会偏宽）**，
#   旧模型是 **−7% ~ +23%** —— 偏窄那一侧才是会撞车的，所以这次换的是安全方向。
# ⚠️ U+FE0F（变体选择符）宽度是 0，但它会把前一个符号**提升成 emoji 字形**：
#   ⚠ 单独是 1.00，⚠️ 连写是 1.26。所以它不能简单当 0 加，要回头改前一个字。
# ══════════════════════════════════════════════════════════════════
_EMOJI_W = 1.26                        # emoji 字形统一宽度（⭐⛔✅ 实测）
_WPX = {}
for _r, _cs in {
    0.27: "|",
    0.28: "',.:;ijl′",
    0.29: "I",
    0.32: "!",
    0.33: "f⁵⁷₀₁₂₃₄₅",
    0.34: "()[]{}",
    0.35: "-",
    0.37: "°⌊⌋⟨⟩",
    0.38: "t",
    0.39: "/r",
    0.41: "²³¹",
    0.43: "ᵀ",
    0.46: "ℓ",
    0.47: "?s",
    0.48: "z",
    0.50: "xε",
    0.51: "c",
    0.52: "vy",
    0.59: "E",
    0.60: "STZΣ",
    0.61: "A`hnou",
    0.62: "bdpq",
    0.63: "Pαβ",
    0.64: "CR",
    0.65: "K",
    0.66: "B",
    0.67: "∘",
    0.68: "&",
    0.69: "DG",
    0.72: "NU",
    0.73: "H",
    0.74: "OQ",
    0.80: "w",
    0.81: "M⏷",
    0.88: "W",
    0.89: "↳⊛✕✗➜",
    0.90: "½",
    0.92: "%",
    0.93: "m",
    1.00: "§±·×÷—‖“”…←↑→↓↔∈∕√∝∞≈≠≡≤≥≫⊘⊙①②③④⑤ⒶⒷⒸ─▲★⚠⬆",
    _EMOJI_W: "⚡⛔✅⭐🏠📌📐🚚",
    1.48: "⟹",
}.items():
    for _c in _cs:
        _WPX[_c] = _r


def wpx(s, size=11.5):
    n = 0.0
    for ch in s:
        if ch == "️":             # 变体选择符：把前一个符号改判成 emoji 宽
            n += _EMOJI_W - _WPX.get(prev, 1.0 if ord(prev) > 0x2E80 else 0.55)
            continue
        n += _WPX.get(ch, 1.0 if ord(ch) > 0x2E80 else 0.55)
        prev = ch
    return int(n * size)


# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-13 现场：「图里那个字，不要太小，也不要太多」
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

    ⛔ 2026-09-12 二轮学生审稿：图里原来直接写 `S_{t-1}`、`ℝ^(d_v×d_k)`，
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


_TSPAN = re.compile(r"<tspan([^>]*)>|</tspan>")


def _src_html(lines):
    """把 src() 收着的那几行（SVG 记法）翻成 HTML 片段。

    ⛔ 不能无脑把 `</tspan>` 换成 `</b>`：一行里同时出现加粗和等宽时
      （`font-weight="700"` 与 `font-family="monospace"`），顺序一错就嵌套交叉。
      ⭐ 所以按**栈**配对，开合两头都断言 ——
      漏网的 tspan 在 HTML 里是个未知元素，浏览器不报错、照样显示文字，
      只是**样式丢了**：一个「看起来对、只是少了点什么」的失效。
      2026-09-14 第一版就是无脑替换，撞上了 `font-family="monospace"`。
    ⚠️ arXiv 编号这里不动 —— 最终 HTML 会由 course_links.linkify_arxiv
      统一处理，这里抢着做只会变成双重链接。
    """
    out = []
    for ln in lines:
        res, stack, pos = [], [], 0
        for m in _TSPAN.finditer(ln):
            res.append(ln[pos:m.start()])
            pos = m.end()
            if m.group(0) == "</tspan>":
                # ⛔ 必须按栈配对，不能无脑替换成同一个闭合标签 ——
                #   一行里同时有加粗和等宽时，顺序一错就嵌套交叉。
                assert stack, "出处行里 </tspan> 比 <tspan> 多：%s" % ln[:50]
                tag = stack.pop()
                if tag:
                    res.append("</%s>" % tag.split(" ")[0])
            else:
                a = m.group(1)
                if "700" in a:
                    tag = "b"
                elif "monospac" in a or "mono" in a:
                    tag = "code"
                elif "fill=" in a:
                    tag = 'span style="color:%s"' % re.search(
                        r'fill=\\?"([^"\\]+)', a).group(1)
                else:
                    tag = None          # 认不出就只留文字，不留空标签
                stack.append(tag)
                if tag:
                    res.append("<%s>" % tag)
        res.append(ln[pos:])
        assert not stack, "出处行里 <tspan> 没闭合：%s" % ln[:50]
        out.append("<p>%s</p>" % "".join(res))
    return "\n".join(out)


class Fig(object):
    """一张 SVG。高度不写死，收尾按真实落点回填。"""

    def __init__(self, w, aria):
        self.w, self.aria, self.p = w, aria, []
        self.p.append("")          # svg 开标签占位
        self._pan = None      # 当前面板 (top, bottom)
        self._over = []       # 画到面板外面去的记录（越界自检）
        self._src = []        # 「出处与口径」正文，见 src()／save()

    # ── 原子 ────────────────────────────────────────────────────
    def t(self, x, y, s, fill=INK, bold=False, size=11.5, anchor=None,
          cls="svgsm", mono=False, w=None, big=False):
        # ⭐ 自动降档：用 500 主色画文字 → 换成对应的 900 深色变体。
        #   这一行就是「照着专题一统一配色」的全部实现。⛔ 别去掉。
        fill = INK900.get(fill, fill)
        # ⛔ 正文天花板（见 MAXPROSE 那段）。**不做静默钳位** ——&#160;钳位会让
        #   源码写着 21、产物却是 17，下一个人照着源码估版面必然估错。
        #   ⭐ 真要破例就显式传 big=True：破例是一个动作，得有人按下去。
        if not big and size > MAXPROSE and _cjk(s) >= PROSE_CJK:
            raise AssertionError(
                "正文长句最大 %dpx，这句给了 %gpx —— 要么降到 %d，要么传 big=True："
                "「%s」" % (MAXPROSE, size, MAXPROSE,
                            re.sub(r"<[^>]+>", "", s)[:34]))
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
        # ⛔⛔ 2026-09-13：<em>／<i>／<b> 这几个是 HTML 解析器的
        #   **foreign-content 逃逸标签**。图是 inline 进 HTML 的，浏览器读到
        #   <em> 会当场**退出 SVG 模式**，后面的内容全部丢弃 ——
        #   fig3-when-axis 因此在浏览器里少了最后 200px（两条落点带 ＋ 出处行）。
        #   ⭐ 而 SVG 文件本身是良构 XML，写盘自检、几何 lint 全都看不出来。
        # ⛔ 2026-09-18 又补一个：<u>。fig4-underflow 的整条落点带
        #   （标题后半 ＋ 两行正文）在浏览器里凭空消失，而 SVG 良构、lint 全绿。
        #   ⭐ 判据：**这种黑名单要么枚举干净，要么改白名单** ——&#160;漏一个等于没有。
        #   目前 SVG 里**唯一允许**的内联标签就是 <tspan>。
        for _esc in ("<em>", "<i>", "<b>", "<u>", "<p>", "<br>", "<font"):
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
        # ⭐⭐⭐ 2026-09-18 现场：「图画得非常廉价……能不能更有质感」。
        #   体检数据：专题四 29 张图里 **613 个 emoji**（光 ⭐ 就 408 个）。
        #   ⛔ 它们是整套图最大的一处廉价来源 ——&#160;
        #     标题栏挂着彩色小图标，看上去像 PPT 剪贴画，而不是技术插图。
        #   ⭐⭐ 但它们承载的语义**不用它们也在**：
        #     「重要」由字重和位置表达，「负面／正面」由红／绿表达，
        #     「注意」由灰色表达 ——&#160;**删掉符号，语义仍然完整**。
        #   ⭐ 改在这里而不是改 613 处字符串：所有文字都从这一个出口走，
        #     而且删字符只会让文本**变短**，不会撑出边界。
        #   ⛔ 保留 ✓ ✕ ↑ ↓ → ← ⋮ 这些 ——&#160;它们是**几何记号，不是装饰**。
        s = _undecorate(s)
        self._note_ink(y)
        # ⭐ 数字按等宽位对齐 ——&#160;表格状的数值竖着看能对齐，
        #   而且不会像比例数字那样 1 特别窄、0 特别宽。中文不受影响。
        st = ["font-size:%.1fpx" % _sz(size), "font-variant-numeric:tabular-nums"]
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
        # ⭐ 线宽不在这儿吸附 ——&#160;序列化出口统一做（见 save() 里那段）。
        fill = _flatten(w, h, fill, stroke)
        # 📌 stroke 默认就是 #dadce0 ——&nbsp;跟专题一的主力描边一致（那边 ×287）。
        #   ⛔ 别把默认改深；要强调就显式传主色，不要靠加重灰线。
        self.p.append('<rect x="%s" y="%s" width="%s" height="%s" rx="%d" fill="%s" '
                      'stroke="%s" stroke-width="%s"%s%s/>'
                      % (x, y, w, h, r, fill, stroke, sw,
                         ' stroke-dasharray="%s"' % dash if dash else '',
                         ' filter="url(#sh)"' if shadow else ''))

    def line(self, x1, y1, x2, y2, col=GY2, sw=1.3, dash=None, arrow=True):
        sw = _hair(sw)
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

        ⛔⛔ 2026-09-13：上一轮给 path() 补了「接受点列表」，**漏了这个姐妹基元** ——
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

    def path(self, d, col=GY2, sw=1.3, dash=None, arrow=True, fill="none"):
        sw = _hair(sw)
        # ⛔⛔ 2026-09-13 审图抓到的最重一条：有 4 个调用方直接传**点列表**
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
        self.p.append('<path d="%s" fill="%s" stroke="%s" stroke-width="%s" '
                      'stroke-linecap="round"%s%s/>'
                      % (d, fill, col, sw,
                         ' stroke-dasharray="%s"' % dash if dash else '',
                         ' marker-end="url(#ah-%s)"' % col.lstrip("#") if arrow else ''))
        if arrow:
            self.marks.add(col)

    marks = set()

    # ── ① 标题区 ＋ 图例条 ───────────────────────────────────────
    # ⛔⛔ 2026-09-13：八张图有「内容被后画的面板底色盖住」——
    #   f.panel() 声明的高度 < 实际画到的位置，下一块面板的白底直接糊上去，
    #   盖掉的还净是关键句（tpu-fix 三条落点带、swa-why 的「柱子按对数画」…）。
    # ⭐ 这条教训本来只写在 topic03-fig-when-axis.py 一个文件的注释里 ——
    #   **写在注释里的判据只在那个文件生效**。做成基元级的断言才是真的修了。
    def _note_ink(self, y):
        """记下「画到了哪一行」，供面板越界自检用。"""
        if self._pan is not None and y > self._pan[1] + 2:
            self._over.append((self._pan[0], y, self._pan[1]))

    def header(self, title, sub, legend=None, y=22):
        # ⭐ 2026-09-13 抬字号：标题 16.5→20，副标题 12→15，图例 11→14。
        #   现场原话「图里那个字不要太小 …… 打到屏幕上去分享」——
        #   ⛔ 副标题和图例是**读图前必须先读的两样**，它们小等于整张图门槛高。
        # ⭐ big=True：整张图的标题是唯一该破 MAXPROSE 的东西 ——&#160;它相当于
        #   一个 h2，不是正文。⛔ 别把这个 big 复制到别处去。
        #   2026-09-14 回看上面那次抬字号：投屏看得清这个诉求，**是靠标题、
        #   副标题、图例这三样撑起来的，不是靠把每一句正文都抬一档**。
        #   后者只会让正文和落点句拉不开，反而更难读。
        self.t(0, y, title, INK, size=20, cls="svglbl", big=True)
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
                # ⭐ 2026-09-24：原来没乘 MONO_K，ASCII 多的图例（如「prefill：一口吞下 prompt」）会顶到下一个色块。
                x += 21 + wpx(lab, 14) * MONO_K + 26
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
        # ⛔⛔ 2026-09-13：外框原来是 **白色实心**。页面也是白的，所以看不出来 ——
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
    # ══════════════════════════════════════════════════════════════
    # ⭐⭐⭐ 2026-09-13 现场：「**最重点的是要画更漂亮的图。**」
    #   而这一课最缺的不是配色，是**共用图标**：36 张图里的小人、箱子、书、
    #   白板、推车，全是各画各的矩形现搭 ——&nbsp;既粗糙，又互相对不上。
    # ⭐ 判据：**同一个比喻在不同图里必须长成同一个样子。**
    #   「一摞复印件」在 MLA 图和 landing 图里要是两种画法，
    #   读者就不会把它们连起来 ——&nbsp;而那条连线正是这门课的价值。
    # ⛔ 「漂亮」不是加渐变加阴影（见 memory feedback_material-design-style）。
    #   这里的做法是：**轮廓线 ＋ 一块浅底 ＋ 一两笔细节**，
    #   细节只画**能承担识别功能**的那一两笔（书的书脊、箱子的封条、推车的轮子）。
    # ══════════════════════════════════════════════════════════════
    def icon(self, kind, x, y, w=40, h=44, col=None, tint=None, label=None,
             lsize=15):
        """画一个图标。**(x, y) 是左上角**，w/h 是外框 —— 跟 box() 一致。

        kind: person 人 · box 箱子 · book 书 · books 一摞书 · paper 纸/复印件
              · board 白板 · cart 推车 · house 房子 · note 便签 · shelf 货架
              · drawer 抽屉 · tray 餐盘 · door 门
        """
        c = col or GY2
        t_ = tint or "#fff"
        P = self.p.append
        def R(rx, ry, rw, rh, f=None, sw=1.4, r=3):
            self.box(x + rx, y + ry, rw, rh, f if f is not None else t_, c, r, sw)
        def L(x1, y1, x2, y2, sw=1.3):
            P('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
              'stroke-width="%s" stroke-linecap="round"/>'
              % (x + x1, y + y1, x + x2, y + y2, c, sw))
        def C(cx, cy, r, f=None):
            P('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="%s" '
              'stroke-width="1.4"/>' % (x + cx, y + cy, r, f or t_, c))

        if kind == "person":                 # 头 ＋ 肩：两笔就够认
            C(w / 2.0, h * 0.28, min(w, h) * 0.19)
            P('<path d="M %.1f %.1f a %.1f %.1f 0 0 1 %.1f 0 z" fill="%s" '
              'stroke="%s" stroke-width="1.4"/>'
              % (x + w * 0.18, y + h * 0.92, w * 0.32, h * 0.42, w * 0.64,
                 t_, c))
        elif kind == "box":                  # 箱子：一道封条
            R(w * 0.06, h * 0.18, w * 0.88, h * 0.68, r=4)
            L(w * 0.5, h * 0.18, w * 0.5, h * 0.86)
            L(w * 0.06, h * 0.34, w * 0.94, h * 0.34)
        elif kind == "book":                 # 书：一条书脊
            R(w * 0.14, h * 0.10, w * 0.72, h * 0.80, r=2)
            L(w * 0.30, h * 0.10, w * 0.30, h * 0.90, 2.2)
        elif kind == "books":                # 一摞：三本并排，高度不齐
            for i, (dx, dh) in enumerate(((0.08, 0.78), (0.38, 0.90), (0.66, 0.70))):
                self.box(x + w * dx, y + h * (0.94 - dh), w * 0.24, h * dh,
                         t_, c, 2, 1.4)
        elif kind == "paper":                # 纸：右上角折角
            P('<path d="M %.1f %.1f H %.1f L %.1f %.1f V %.1f H %.1f Z" '
              'fill="%s" stroke="%s" stroke-width="1.4" stroke-linejoin="round"/>'
              % (x + w * 0.14, y + h * 0.08, x + w * 0.70, x + w * 0.88,
                 y + h * 0.26, y + h * 0.92, x + w * 0.14, t_, c))
            L(w * 0.70, h * 0.08, w * 0.70, h * 0.26)
            L(w * 0.70, h * 0.26, w * 0.88, h * 0.26)
            for k in range(2):
                L(w * 0.26, h * (0.48 + k * 0.18), w * 0.72, h * (0.48 + k * 0.18), 1.0)
        elif kind == "board":                # 白板：两条腿
            R(w * 0.04, h * 0.06, w * 0.92, h * 0.62, r=3)
            L(w * 0.24, h * 0.68, w * 0.16, h * 0.94)
            L(w * 0.76, h * 0.68, w * 0.84, h * 0.94)
        elif kind == "cart":                 # 推车：车斗 ＋ 两个轮子
            R(w * 0.06, h * 0.14, w * 0.76, h * 0.50, r=3)
            L(w * 0.82, h * 0.14, w * 0.94, h * 0.14)
            C(w * 0.26, h * 0.82, min(w, h) * 0.11)
            C(w * 0.66, h * 0.82, min(w, h) * 0.11)
        elif kind == "house":                # 房子：墙先画，屋顶后画且不填实
            # ⛔ 第一版屋顶 fill=描边色（实心深灰），把墙整个压住了 ——
            #   ⭐ 又是「后画的不透明形状盖住先画的」那条，这次栽在自己手上。
            R(w * 0.16, h * 0.44, w * 0.68, h * 0.48, r=2)
            L(w * 0.44, h * 0.66, w * 0.44, h * 0.92, 1.6)   # 门
            L(w * 0.44, h * 0.66, w * 0.62, h * 0.66, 1.6)
            L(w * 0.62, h * 0.66, w * 0.62, h * 0.92, 1.6)
            P('<path d="M %.1f %.1f L %.1f %.1f L %.1f %.1f" fill="none" '
              'stroke="%s" stroke-width="2.0" stroke-linejoin="round" '
              'stroke-linecap="round"/>'
              % (x + w * 0.04, y + h * 0.46, x + w * 0.5, y + h * 0.08,
                 x + w * 0.96, y + h * 0.46, c))
        elif kind == "note":                 # 便签：左上角一枚图钉
            R(w * 0.10, h * 0.14, w * 0.80, h * 0.74, r=2)
            C(w * 0.26, h * 0.26, min(w, h) * 0.07, c)
        elif kind == "shelf":                # 货架：三层
            R(w * 0.04, h * 0.08, w * 0.92, h * 0.84, r=2)
            for k in range(2):
                L(w * 0.04, h * (0.36 + k * 0.28), w * 0.96, h * (0.36 + k * 0.28))
        elif kind == "drawer":               # 抽屉：三格 ＋ 把手
            R(w * 0.04, h * 0.08, w * 0.92, h * 0.84, r=3)
            for k in range(3):
                yy = h * (0.08 + 0.28 * k)
                if k:
                    L(w * 0.04, yy, w * 0.96, yy)
                L(w * 0.40, yy + h * 0.14, w * 0.60, yy + h * 0.14, 2.0)
        elif kind == "tray":                 # 餐盘 / 蒸屉：一个浅盘
            R(w * 0.04, h * 0.30, w * 0.92, h * 0.40, r=5)
            L(w * 0.04, h * 0.70, w * 0.96, h * 0.70, 2.0)
        elif kind == "door":                 # 门框：只画两根柱子和门楣
            L(w * 0.18, h * 0.08, w * 0.18, h * 0.94, 2.6)
            L(w * 0.82, h * 0.08, w * 0.82, h * 0.94, 2.6)
            L(w * 0.18, h * 0.08, w * 0.82, h * 0.08, 2.6)
        else:
            raise AssertionError("没有这个图标：%s" % kind)

        if label:
            self.t(x + w / 2.0, y + h + lsize + 4, label, c, True, lsize,
                   anchor="middle")
        return x + w, y + h

    def badge(self, x, y, n, col=None, r=13):
        """序号徽章：一个实心圆 ＋ 白字。⭐ 用它给步骤编号，别再写「1.」。"""
        c = col or INK
        self.p.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>'
                      % (x + r, y + r, r, c))
        self.t(x + r, y + r + r * 0.38, str(n), "#fff", True,
               _sz(int(r * 1.25)), anchor="middle")
        return x + 2 * r

    def elbow(self, x1, y1, x2, y2, col=None, sw=1.6, r=10, arrow=True,
              via="h"):
        """圆角直角连线。⛔ 别再用两条 line 拼 ——&nbsp;硬直角是「简陋」的主要来源之一。

        via="h" 先横后竖，via="v" 先竖后横。
        """
        c = col or GY2
        if arrow:
            self.marks.add(c)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        rr = min(r, abs(x2 - x1) / 2.0, abs(y2 - y1) / 2.0)
        if via == "h":
            d = ("M %.1f %.1f H %.1f Q %.1f %.1f %.1f %.1f V %.1f"
                 % (x1, y1, x2 - sx * rr, x2, y1, x2, y1 + sy * rr, y2))
        else:
            d = ("M %.1f %.1f V %.1f Q %.1f %.1f %.1f %.1f H %.1f"
                 % (x1, y1, y2 - sy * rr, x1, y2, x1 + sx * rr, y2, x2))
        self.p.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s" '
                      'stroke-linecap="round"%s/>'
                      % (d, c, sw,
                         ' marker-end="url(#ah-%s)"' % c.lstrip("#")
                         if arrow else ''))

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

    def band(self, y, kind, title, lines, w=None, fold=False, keep=False):
        """落点带。⛔ **不填色** ——&nbsp;白底 ＋ 细灰框 ＋ 左侧 4px 彩色竖条。
        见文件头「填充规则」：容器不填，只有承载信息的小元素才填。

        ════════════════════════════════════════════════════════════
        ⭐⭐ `fold=True` ——&#160;这一条不画进图，折进图下面那个 details
        ════════════════════════════════════════════════════════════
        2026-09-15 加。量出来的事实：**全书 55% 的字是画在图里的**
        （图内 44,312 字 / 正文 32,627 / 图注 3,755），而现场的原话是
        「总的原则是只要图，尽量少写文字」。

        ⭐ 但不是所有落点带都该搬走。59 条里分得很清楚：
          · **口径 / 免责 / 别讲过头**（约 17 条）——&#160;它跟「出处与口径」
            是同一类东西：**给较真的人看的脚注**，99% 的阅读里是噪音，
            却每张图都吃掉四到八行满宽的灰字。→ **`fold=True`**
          · **这张图的教益**（约 40 条）——&#160;它是图的落点，**必须留在图上**，
            折起来等于没有。
        ⛔ 判据：**问一句「它是帮人看懂这张图的，还是帮人别误用这张图的」。**
          前者留在图上，后者折起来。

        ⚠️ 折起来**不等于删**：「不确定就去查，绝不编」的另一半是
          「查过的要留下出处」——&#160;所以它仍然逐字在页面上，只是默认收着。
        """
        # ⭐⭐⭐ 2026-09-16 现场：「像这样的小字对主线也没什么帮助……
        #   你在讲课的时候，就算投到屏幕上人家也看不见的那种，就先收起来。」
        #   ⛔ 但**不能只留第一行**：115 条落点带里 113 条是多行，而其中
        #     绝大多数的第 2 行正是那句「⛔ 代价 / 欠下的」——&#160;
        #     本课反复强调「好处和坏处必须一起出现」，砍掉第 2 行等于只报喜。
        #   ⭐ 所以判据定在**第三行**：**一条落点带最多两行（一好一坏），
        #     第三行起就是展开说明，收进「出处与口径」。**
        #     实测命中 55 条（3 行 42、4 行 11、5 行 2），一条配对都没拆散。
        #   ⚠️ `keep=True` 是逃生舱：极少数三行都不能少的地方用它。
        if not fold and not keep and len(lines) > 2:
            self._src.append('<tspan font-weight="700">%s %s（接上图）</tspan>'
                             % (self.KIND[kind][2], title))
            self._src.extend(ln for ln in lines[2:] if ln and ln.strip())
            lines = lines[:2]
        if fold:
            # ⭐ 复用 src() 那条通道：`save()` 会把 `_src` 旁落成 .src.html，
            #   由 topic03_page 包成图下面那个默认收起的 <details>。
            #   顺带白捡 tspan→HTML 的转换（_src_html 已按栈配对处理好）。
            self._src.append('<tspan font-weight="700">%s %s</tspan>'
                             % (self.KIND[kind][2], title))
            self._src.extend(ln for ln in lines if ln and ln.strip())
            return y                      # ⛔ 不吃高度
        self._pan = None            # 落点带在面板外面，合法
        col, fill, icon = self.KIND[kind]
        w = w or self.w
        # ⭐ 2026-09-13：字号 12 → 15，换行改成自动（见 wrap_rich 的说明）。
        #   ⛔ 原来靠宽度断言逼调用方拆行 —— 那既让字小，又让句子被拆碎。
        SZ, LH = 15, 24
        rows = []
        for ln in lines:
            # ⛔ svgsm 是**等宽字体**，Roboto Mono 的 ASCII 步进约 0.60em，
            #   而 wpx() 按 0.55 估 —— 英文长句一累积就差出一整行。
            #   ⭐ 所以折行时按 MONO_K 倍的字号去量，留出这 10%。
            rows.extend(wrap_rich(ln, w - 40, SZ * MONO_K))
        h = 40 + len(rows) * LH + 10
        # ⛔⛔ 2026-09-13 第三次撞同一条：**跟背景同色的填充照样是一次覆盖**。
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
        """📌 出处与口径 ——&#160;**2026-09-14 起不再画进 SVG**。

        现场原话：「整个文档里边有很多这种小字，并且是半透明的，其实并不重要，
        就是需要收着，折起来。你把这些信息都折叠起来省地方，看的还清晰。」

        ⭐⭐ 这段东西的定位本来就不是「图的一部分」——&#160;它是**给较真的人看的
        脚注**：出处、口径、哪个数是实测哪个是断言。它每张图都占四到八行满宽的
        灰字，加起来比好几张图还高，而 99% 的阅读里它只是噪音。
        ⛔ 但它**不能删** ——&#160;「不确定就去查，绝不编」这条规矩的另一半就是
        「查过的要留下出处」。所以是**折起来**，不是拿掉。

        实现：这里只把原文收着，`save()` 旁落一份 `<name>.src.html`，
        由 topic03_page.place_figs 包成图下面的 `<details>`。
        ⭐ 顺带白捡一样：文字由浏览器折行，不再需要 wrap_rich 按等宽估宽 ——
          那条「冲出右边界、既不报错也不产生滚动条、只是被静默裁掉」的老毛病
          （2026-09-08 踩过）从根上没了。
        ⚠️ arXiv 链接**不是白捡的** ——&#160;搬家前 `_svg_linkify` 就已经在 SVG
          里把它们做成可点的了，全页 109 个链接搬家前后一个不多一个不少。
          （写这段注释时我先写成了「白捡」，数完才改过来。数一遍再写。）

        ⛔ 返回值仍然是传进来的 y（不再吃高度）——&#160;调用方一律
        `f.save(name, yy + 6)`，所以图的下沿会**自动收掉这四到八行**。
        """
        self._pan = None
        self._src = [ln for ln in lines if ln and ln.strip()]
        return y

    # ── 收尾 ────────────────────────────────────────────────────
    def save(self, name, bottom):
        # ⛔ 2026-09-17：`name` 是**完整文件名**，要带 .svg。
        #   漏掉扩展名不会报错 ——&nbsp;它会老老实实写出一个没有后缀的文件，
        #   然后你在 figs/ 里怎么找都找不到那张图。
        #   ⭐ 判据：**一个接受文件名的函数，要么自己补后缀，要么当场拦下。**
        #     这里选拦下 —— 补后缀会掩盖调用方的笔误，而十张图里九张写对了。
        assert name.endswith(".svg"), (
            "save() 要完整文件名：写 \"%s.svg\" 不是 \"%s\"" % (name, name))
        # ⛔⛔ 2026-09-14 R62 R2 立的硬约束：**共用 SVG 里不许出现本课节号。**
        #   起因：L200 上线后发现图里写着「§零 那张…」——&nbsp;而 L200 根本没有
        #   §零（它的第一章就是 RNN）。读者看到一个指向不存在章节的指针。
        #   ⭐⭐ 而且**没法靠改数字解决**：L300 的 §五 在 L200 拆成了第四、第五
        #     两章，§九 和 §十 又并成了第九章 ——&nbsp;两页的编号不是平移关系。
        #   ⭐ 所以规矩只能是：**共用资产里放名字，不放指针。**
        #     这跟本文件原有那条「图注里不许出现『上一张 / 下一张』」是同一条
        #     ——&nbsp;方位词和节号都是「指针」，换个文档就指错，而且不报错。
        #   📌 § ＋ 阿拉伯数字放行（那是论文的节号，例如 Vaswani §3.2）。
        #   ⚠️ 这条是**硬失败**，跟上面越界自检不同：越界只是难看，
        #     指错章节是**给读者一条走不通的路**，而且两页里只有一页坏。
        import re as _re
        _svgtext = " ".join(_re.sub(r"<[^>]+>", "", t)
                            for t in _re.findall(r"<text[^>]*>.*?</text>",
                                                 "\n".join(self.p), _re.S))
        _bad = _re.findall(r"§\s*[零一二三四五六七八九十]+", _svgtext)
        # ⛔⛔ 2026-09-14 R6 补上**阿拉伯数字那一半**。上一版只挡中文数字，
        #   于是「§5.4b」「§8.1」这种照样漏过去 ——&nbsp;实测漏了 **28 处 / 18 张图**。
        #   ⭐ 判据（今天第三次踩同一个形状）：**按写法建清单，就会漏掉别的写法。**
        # 📌 论文自己的节号要放行，判据跟 topic02-lint-xref 的 FOREIGN 一致：
        #   **同一段文字里出现出处词**（论文 / paper / arXiv / 原文 / 技术报告 / 该文）
        #   就认为这个 § 属于别人的文档，不是本课的指针。
        # ⛔ 「专题二 §1」这种**跨讲指针**要放行：它点名了是哪一讲，
        #   而别的讲的编号不会因为本讲分成两页而变。⚠️ 专题二是过去式，
        #   这条断言不许把它逼着改版（figx-*.svg 也走这个 save）。
        _OK = ("论文", "paper", "arXiv", "原文", "技术报告", "该文", "Vaswani",
               "专题")
        for _m in _re.finditer(r"(.{0,80})§\s*\d[\d.a-z–—~]*", _svgtext):
            if not any(k in _m.group(1) for k in _OK):
                _bad.append(_m.group(0)[-24:])
        assert not _bad, (
            "%s 的图内文字出现本课节号 %s ——&nbsp;共用 SVG 里要用名字不用节号"
            "（L200 / L300 两页编号对不上，见本处注释）" % (name, sorted(set(_bad))))
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
        # ⭐⭐ 2026-09-13 现场点的：「引用的那些论文得在教材里边，
        #   把可点击的 link 都放里边，有愿意多学的人可以去点开看。」
        #   正文那边由 course_links.py 后处理，**但图里的出处它够不着** ——
        #   那个后处理刻意跳过 <svg>（HTML 的 <a> 进不去 SVG 的文本流）。
        # ⭐ 所以图这边自己来：在写盘前把 arXiv 编号包成 **SVG 自己的 <a>**。
        # ⚠️ 两个前提，缺一个就白做：
        #   ① 必须是**内联**的 SVG（本课就是内联进 HTML 的，所以点得动）；
        #   ② `<a>` 要包在 `<text>` **里面**（包在外面 Chrome 不给点）。
        s = _svg_linkify(s)
        # ⭐⭐⭐ 辅助线的线宽在这里统一吸附 ——&#160;**序列化出口，不是各个构造点。**
        #   ⛔ 我先试过在 line()／path()／box() 三处改，结果没生效：
        #     这个文件里一共有 **11 处**输出 stroke-width（panel 的分隔线、band 的
        #     色条、箭头 marker、grid pattern……）。**构造点永远数不完。**
        #   ⭐ 判据：**要做「全局一致」的改动，改在序列化出口。**
        #   ⛔ 仍然只动 < 1.5 的：线宽在有些图里是数据（见 _hair 的注释）。
        s = re.sub(r'stroke-width="([0-9.]+)"',
                   lambda m: 'stroke-width="%s"' % _hair(m.group(1)), s)
        xml.dom.minidom.parseString(s.encode("utf-8"))
        io.open(os.path.join(HERE, name), "w", encoding="utf-8").write(s)

        # ── 「出处与口径」旁落一份 HTML 片段（见 src() 的注释）────────────
        # ⛔ 没有出处时**必须把旧的删掉**。否则改图时把 f.src(...) 去掉，
        #   上一次留下的片段还躺在 tools/ 里，build 照样把它包进去 ——
        #   ⭐ 产物目录里的陈旧文件不会报错，只会安静地继续生效。
        side = os.path.join(HERE, name[:-4] + ".src.html")
        if self._src:
            io.open(side, "w", encoding="utf-8").write(_src_html(self._src))
        elif os.path.exists(side):
            os.remove(side)
        print("ok  %s  %d×%d%s"
              % (name, self.w, bottom, "  +出处" if self._src else ""))
