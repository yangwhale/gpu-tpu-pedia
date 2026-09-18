#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""figink ——&#160;量一张图「删掉所有文字之后还剩多少东西」。

⭐⭐⭐ 2026-09-18 R08/20 新增。现场原话：
  「那些图画的实在是太粗糙了，根本就没有用心。」
  ⛔ 而「粗糙」这个词没法直接执行。上一轮把它翻成了一条**可证伪**的判据：

      **把这一格里所有文字删掉，还剩下什么？
        剩下的东西能不能让人猜出这一格在讲什么？**

  剩不下东西的 ——&#160;那就是**一块写字的板子**，不是图。

⭐⭐ 这个脚本把那条判据**操作化**：解析成品 SVG，数两样东西 ——&#160;
  · **文字**：`<text>` 的个数
  · **承载信息的形状**：真曲线、椭圆、折线、数据点、斜线
  然后算一个比值。⛔ 它**只报告，不中止构建** ——&#160;
  跟 `figpos` 一样，判断要靠人，工具只负责**把该看的那几张排到前面**。

⛔⛔ 三条「不算数」的，必须排除干净，否则每张图都会显得很有料：
  ① **面板外框 / 标题栏**：每张图都有，是模板，不是内容
  ② **文字块的底色框**：它恰恰是「写字板子」的特征，不能算成画
     ⛔ 第一版想用**尺寸**把它跟柱子分开 ——&#160;失败了，两者尺寸重叠，
     结果把 `fig4-slider`、`fig4-precision` 这种真画了东西的判成了板子。
     现在改用「**同组之间变不变**」来分。
  ③ **箭头**：`<path>` 里那些两三个点的小三角，是标注不是图形

⭐ 判据的操作化（每一条都可以吵，但至少是明确的）：
  · **真曲线** ＝ 一条 `<path>` 的 `d` 里有 **≥ 8 个** 绘图命令（L/C/Q/A）
    ——&#160;箭头和圆角撑死三四个，画函数的动辄上百个
  · **柱子** ＝ 一组**同宽但高度各异**（或同高异宽）的 `<rect>`，至少 5 个
    ——&#160;⭐ 同宽**同高**的一排是表格 / 卡片，**不算**。
    分类要抓「它为什么在那儿」，不是「它多大」。
  · **连线图** ＝ 一格里有 ≥ 3 个 `marker-end`（箭头）
    ——&#160;⭐ **表格的格子之间没有箭头，连线图全靠箭头。**
  · **数据点** ＝ 边长 ≤ 20 px 的 `<rect>`（本仓库用小方块当点）
  · **斜线** ＝ `<line>` 里 x1≠x2 且 y1≠y2 的那些（轴线和分隔线都是正的）
  · **容器** ＝ 宽 > 120 且 高 > 40 的 `<rect>`，一律不计

⚠️ **已知看不见的（写在这儿，免得每轮都再发现一次）**：
  · **等大方块靠「个数」表意的图**（unit chart / 点阵）——&#160;
    比如 `fig4-3x` Ⓒ 用 1 / 3 / 4 个等大的「一遍」方块表示 1× / 3× / 4×。
    那些方块同宽同高，被判成表格；可**个数本身就是数据**。
  ⭐ 所以这把尺子只是**分诊台**，不是判官 ——&#160;
    它负责把该复看的排到前面，**最终判断永远是人看截图做的**。
  ⛔ 判据：**一个指标发现自己第 N 次误判时，要么修，要么把盲区写进文档 ——&#160;
    不要让它一边不准、一边继续被当成结论。**

# ⚠️ 已知盲区之二（2026-09-18）：**「两根长度对比条」认不出来。**
#   fig4-slider Ⓐ 用两根粗细相同、长度比 1:3 的竖条表达导数，
#   人眼一看就是画 ——&#160;但它只有 2 个 rect，够不上 bars 判据（≥5 个）。
#   ⭐ 判据本身不改：放宽到 2 个 rect 会把「两个文字框」也算成画。
#     ⛔ 记在这儿是为了**读报告的人知道它会漏这一类**，
#       而不是让下一个人看到 ❌ 就去重画一张本来就没问题的图。
#
# ⚠️ 已知盲区之三（2026-09-18）：**数轴类图认不出来。**
#   fig4-underflow 是一根对数轴 ＋ 三条边界竖线 ＋ 一大片区间色块 ——&#160;
#   人眼一看全是画，但：竖线只有 3 条且只有 2 种长度（够不上 vectors 的「≥3 种」），
#   色块只有 1 个 rect，刻度线 9 条等长不算数。
#   ⭐ 共同点跟盲区之二一样：**它数不出「少而承重」的形状** ——&#160;
#     它擅长抓「一堆方框配一堆字」，不擅长抓「三条线定生死」。
#   ⛔ 仍然不放宽阈值：放宽到「2 条线也算」会把箭头连接的文字框全判成有画。
#
# ⭐⭐⭐ 盲区总结（2026-09-18，判完 18 格之后）：
#   **问题不是阈值太严，是「形状词汇表」太小。**
#   实测：专题四 24 格不及格里，**23 格的墨水是 0**；
#   而人工判出的误报 —— 数轴、两三根长度对比条、unit chart、
#   时间轴分段、坐标图、一排只有一个在动的推子 —— **墨水也全是 0**。
#   ⛔ 也就是说它们不是「差一点」，是**根本没被这套词汇认出来**。
#   ⭐ 这套词汇现在只认「**成组的**」：曲线、柱子组、向量组、箭头组、点群、斜线群。
#     它不认「**少量的、承重的、非成组的**」——&#160;而好图里恰恰常常是后者。
#   ⚠️ 所以**别放宽阈值**（那会把文字框放进来），
#     要加的是**新的形状类型**。手上已有一个回归集可用：
#     专题四 18 格已判样本（12 个误报 ／ 6 个真该改）。
#     ⛔ 校准之后**两头都要跑**：只让 12 个误报转绿不算成功，
#       6 个真该改必须仍然判红。



用法：
    python3 topic02-lint-figink.py                 # 扫 tools/*.svg
    python3 topic02-lint-figink.py fig4-vote.svg   # 只看一张
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ⭐ 一条 path 至少要有这么多绘图命令，才算「真的画了一条线」
CURVE_CMDS = 8
DOT_MAX = 20            # 小于这个边长的 rect 当数据点
BAR_MIN = 5             # 同宽异高（或同高异宽）至少这么多个，才算一组柱子
TOL = 2.0               # 尺寸相同的容差（px）
PANEL_MIN_H = 190       # 比这矮的整宽矩形是落点带，不是一格
VEC_MIN = 3             # 同向但长度各异的线，至少这么多根才算一组向量


def _num(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _rects(svg):
    out = []
    for m in re.finditer(r"<rect\b[^>]*>", svg):
        tag = m.group(0)
        g = dict((k, _num(v)) for k, v in
                 re.findall(r'\b(width|height)="([^"]*)"', tag))
        if len(g) == 2:
            out.append((g["width"], g["height"]))
    return out


def _bar_groups(rects):
    """⛔⛔ 第一版用**尺寸**区分「柱子」和「文字框」——&#160;彻底失败：
    一根柱子和一个文字框可以一样大，于是 `fig4-slider`、`fig4-precision`
    这种明明画了东西的图被判成了写字板子。

    ⭐⭐ 换一个真能分开的判据：**看它们变不变**。
      · 一排**同宽、但高度各不相同**的矩形 ＝ 柱状图（**画**）
      · 一排**同宽同高**的矩形 ＝ 表格 / 并排的卡片（**写字板**）
    ⭐ 判据（元级）：**分类要抓「它为什么在那儿」，而不是「它多大」** ——&#160;
      尺寸是表象，「同组之间变不变」才是它承不承载数据的证据。
    """
    n = 0
    for idx, other in ((0, 1), (1, 0)):
        buckets = {}
        for r in rects:
            buckets.setdefault(round(r[idx] / TOL), []).append(r[other])
        for key, vals in buckets.items():
            if len(vals) < BAR_MIN:
                continue
            if len(set(round(v / TOL) for v in vals)) >= 3:   # 另一边真的在变
                n += len(vals)
    return n


def _y_of(tag):
    """拿到一个元素的纵向位置 ——&#160;用来把它归进某一格。"""
    for k in ("y1", "cy", "y"):
        m = re.search(r'\b%s="([^"]*)"' % k, tag)
        if m:
            return _num(m.group(1))
    m = re.search(r'\bd="[Mm]\s*[-\d.]+[ ,]+([-\d.]+)', tag)
    return _num(m.group(1)) if m else None


def _panels(svg):
    """⭐⭐⭐ 按**格**切，不按**张**切。

    ⛔ 上一版量的是整张图 ——&#160;于是 21 张全过。
      可现场那句判据说的是「把**这一格**里所有文字删掉」：
      **一张图里一个好格子，能把三个写字板子盖过去。**
    ⭐ 判据（元级）：**量化一条判据的时候，粒度必须跟判据原话一致** ——&#160;
      粒度放粗一档，指标就会自动变好看，而问题一个没少。

    面板是 `f.panel()` 画出来的整宽矩形，所以找 width ≈ 画布宽的那些 rect。
    """
    out = []
    for m in re.finditer(r"<rect\b[^>]*>", svg):
        tag = m.group(0)
        w = _num((re.search(r'\bwidth="([^"]*)"', tag) or [None, 0])[1])
        h = _num((re.search(r'\bheight="([^"]*)"', tag) or [None, 0])[1])
        y = _num((re.search(r'\by="([^"]*)"', tag) or [None, 0])[1])
        if w >= 1300 and h >= PANEL_MIN_H:    # 整宽且够高 ＝ 一格的外框
            # ⛔ 落点带（f.band）也是整宽矩形，但它按设计就是一段文字。
            #   门槛卡在高度上：面板都在 200 px 以上，落点带在 140 以下。
            out.append((y, y + h))
    out.sort()
    # 去掉互相包含的（有的图会再套一层）
    keep = []
    for a0, b0 in out:
        if not any(x0 <= a0 and b0 <= y1 and (x0, y1) != (a0, b0) for x0, y1 in out):
            keep.append((a0, b0))
    return keep


def _ink_of(tags, rects):
    # ⛔⛔ 第四个 bug，是 R09 真去看那张图才发现的：
    #   `fig4-stability` Ⓐ **画了河**（三个填充的梯形 path），
    #   可每段只有三个 L 命令，被 CURVE_CMDS ≥ 8 的门槛整个漏掉，判成了写字板子。
    #   ⭐ 判据：**「填了色的 path」是一个形状，跟它有几个命令无关** ——&#160;
    #     命令数量只能判「线画得细不细」，判不了「是不是画了个东西」。
    #   ⭐⭐ 元级：**一把新尺子，要拿几个你已经知道答案的样本回归一遍** ——&#160;
    #     R08 拿的样本全是「我知道它好」的，所以没抓到这种「我知道它好、它却判坏」的。
    n_curve = n_fill = 0
    for tag in tags:
        m = re.search(r'\bd="([^"]*)"', tag)
        if not m:
            continue
        if len(re.findall(r"[LCQAlcqa]", m.group(1))) >= CURVE_CMDS:
            n_curve += 1
        else:
            fm = re.search(r'\bfill="([^"]*)"', tag)
            if fm and fm.group(1) not in ("none", "", "transparent"):
                n_fill += 1
    n_shape = n_fill + sum(1 for t in tags
                           if re.match(r"<(ellipse|circle|polyline|polygon)\b", t))
    n_dot = sum(1 for w, h in rects if w <= DOT_MAX and h <= DOT_MAX)
    n_bar = _bar_groups([r for r in rects if not (r[0] <= DOT_MAX and r[1] <= DOT_MAX)])
    # ⛔ 第二个假阴性：**横平竖直但长度各异**的线也是数据
    #   （`fig4-momentum` Ⓑ 那三根按真实数值画长度的箭头，全是水平的）。
    #   ⭐ 跟柱子同一个判据：**同向而长度在变 ＝ 承载数据；同向同长 ＝ 分隔线。**
    n_slant = 0
    hor, ver = [], []
    for tag in tags:
        if not tag.startswith("<line"):
            continue
        g = dict((k, _num(v)) for k, v in
                 re.findall(r'\b(x1|y1|x2|y2)="([^"]*)"', tag))
        if len(g) != 4:
            continue
        dx, dy = abs(g["x1"] - g["x2"]), abs(g["y1"] - g["y2"])
        if dx > 1 and dy > 1:
            n_slant += 1
        elif dy <= 1 and dx > 8:
            hor.append(dx)
        elif dx <= 1 and dy > 8:
            ver.append(dy)
    # ⛔⛔ 第五个 ——&#160;也是 R09 真去看图才发现的：
    #   `fig4-reverse` Ⓑ 是**一张连线图**（一条链上的方框 ＋ 反着走的箭头），
    #   而我的指标把「同宽同高的节点」判成了表格，整格算 0。
    #   ⭐ 连线图跟表格的区别**不在方框，在箭头** ——&#160;
    #     表格的格子之间没有箭头，连线图全靠箭头。
    #   ⭐⭐ 所以数 `marker-end`：≥ 3 个就说明这一格在**画关系**，不是在列条目。
    n_arrow = sum(1 for t in tags if "marker-end" in t)

    n_vec = 0
    for grp in (hor, ver):
        if len(grp) >= VEC_MIN and len(set(round(v / 8.0) for v in grp)) >= VEC_MIN:
            n_vec += len(grp)
    ink = (n_curve * 6 + n_shape * 3 + n_bar * 2 + n_vec * 2
           + (n_arrow * 2 if n_arrow >= 3 else 0) + n_dot + n_slant * 2)
    return dict(curve=n_curve, shape=n_shape, bar=n_bar, vec=n_vec,
                arrow=n_arrow, dot=n_dot, slant=n_slant, ink=ink)


def _reads(path):
    """返回 (格数, 每一格的读数)。⭐ scan() 和 scan_all() 共用这一份，
    免得两条路算出不一样的数。"""
    svg = open(path, encoding="utf-8").read()
    panels = _panels(svg)
    els = re.findall(r"<(?:rect|line|path|ellipse|circle|polyline|polygon|text)\b[^>]*>", svg)

    def bucket(lo, hi):
        tags, rects, ntext = [], [], 0
        for tag in els:
            y = _y_of(tag)
            if y is None or not (lo <= y <= hi):
                continue
            if tag.startswith("<text"):
                ntext += 1
                continue
            if tag.startswith("<rect"):
                w = _num((re.search(r'\bwidth="([^"]*)"', tag) or [None, 0])[1])
                h = _num((re.search(r'\bheight="([^"]*)"', tag) or [None, 0])[1])
                if w >= 1300 and h >= PANEL_MIN_H:   # 外框本身不算
                    continue
                rects.append((w, h))
            tags.append(tag)
        r = _ink_of(tags, rects)
        r["text"] = ntext
        return r

    if not panels:
        return 0, [bucket(0, 10 ** 6)]
    return len(panels), [bucket(lo, hi) for lo, hi in panels]


def scan(path):
    """这张图里**最差的那一格** ——&#160;一张图只报它最弱的一环。"""
    n, reads = _reads(path)
    k = min(range(len(reads)), key=lambda i: reads[i]["ink"])
    r = dict(reads[k])
    r.update(name=os.path.basename(path), panels=n, worst=k + 1)
    return r


def scan_all(path):
    """⭐ 2026-09-18 新增：**每一格都报**。
    ⛔ 起因是一个真实的误解：改好一张图最差的那一格之后，
      ❌ 计数**没有降** ——&#160;因为同一张图的下一格顶了上来。
      于是「❌ 计数」被我当成了「问题数」，其实它是「**有问题的图数**」。
    ⭐ 判据：**一个「每个对象只报最差项」的报告，它的计数不是工作量。**
      要点工作量就得按「格」点，所以有了这个开关（`--per-panel`）。"""
    n, reads = _reads(path)
    return [dict(r, name=os.path.basename(path), panels=n, worst=i + 1)
            for i, r in enumerate(reads)]


def verdict(r):
    """⛔ 只给三档，而且最重的一档要求很严 ——&#160;宁可漏报，不要让人不看它。"""
    if (r["curve"] == 0 and r["shape"] == 0 and r["bar"] == 0
            and r["vec"] == 0 and r["arrow"] < 3
            and r["dot"] < 6 and r["slant"] < 4):
        return "❌ 写字板子"
    if r["curve"] == 0 and r["ink"] < 15:
        return "⚠️ 偏板子"
    return "✅ 有画"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        files = [a if os.path.isabs(a) else os.path.join(HERE, a) for a in args]
    else:
        files = sorted(os.path.join(HERE, f) for f in os.listdir(HERE)
                       if f.endswith(".svg"))
    if not files:
        print("   （没找到 svg）")
        return 0

    # ⭐ --per-panel：按**格**点，不按图。见 scan_all() 的注释。
    if any(x == "--per-panel" for x in sys.argv[1:]):
        cells = [c for f in files for c in scan_all(f)]
        cells.sort(key=lambda r: (r["ink"], -r["text"]))
        NAMES = "Ⓐ Ⓑ Ⓒ Ⓓ Ⓔ Ⓕ Ⓖ Ⓗ".split()
        nbad = 0
        print("\n\033[1m▸ 墨水体检 · 逐格\033[0m")
        for c in cells:
            v = verdict(c)
            if not v.startswith("❌"):
                continue
            nbad += 1
            print("   %-24s %-3s 文字%3d 墨水%3d  %s"
                  % (c["name"], NAMES[c["worst"] - 1] if c["worst"] <= 8
                     else str(c["worst"]), c["text"], c["ink"], v))
        print("\n   共 %d 张图 / %d 格，其中 \033[1m%d 格\033[0m 进了最重那一档。"
              % (len(files), len(cells), nbad))
        print("   ⛔ 这个数才是工作量 ——　按图点出来的那个是「有问题的图数」。")
        return 0

    rows = [scan(f) for f in files]
    rows.sort(key=lambda r: (r["ink"], -r["text"]))

    print("\n\033[1m▸ 图的「墨水」体检 ——　删掉所有文字之后，还剩多少东西\033[0m")
    print("   %-24s %4s %5s %4s %4s %4s %4s %4s %5s  %s"
          % ("图", "格数", "最差格", "文字", "曲线", "柱子", "箭头", "点", "墨水", "判"))
    bad = 0
    for r in rows:
        v = verdict(r)
        if v.startswith("❌"):
            bad += 1
        print("   %-24s %4d %5s %4d %4d %4d %4d %4d %5d  %s"
              % (r["name"], r["panels"], "Ⓐ Ⓑ Ⓒ Ⓓ Ⓔ Ⓕ".split()[r["worst"] - 1]
                 if r["worst"] <= 6 else str(r["worst"]),
                 r["text"], r["curve"], r["bar"], r["arrow"], r["dot"],
                 r["ink"], v))

    print("\n   ⭐ 判据：**把文字全删掉，剩下的能不能让人猜出这格在讲什么。**")
    print("   ⛔ 「❌ 写字板子」＝ 那一格里一条真曲线都没有、没形状、没柱子、点不到 6 个、斜线不到 4 条。")
    print("   ⚠️ 报的是**每张图里最差的那一格**，不是整张图的平均 ——　\n"
          "      因为一个好格子会把三个板子盖过去。\n"
          "   ⚠️ 这是**报告，不是断言** ——　有些格子天生就该是表（比如口径对照），")
    print("      工具只负责把该复看的排到前面。%s"
          % ("" if not bad else "本次有 %d 张进了最重那一档。" % bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
