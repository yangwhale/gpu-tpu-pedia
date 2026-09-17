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
  · **数据点** ＝ 边长 ≤ 20 px 的 `<rect>`（本仓库用小方块当点）
  · **斜线** ＝ `<line>` 里 x1≠x2 且 y1≠y2 的那些（轴线和分隔线都是正的）
  · **容器** ＝ 宽 > 120 且 高 > 40 的 `<rect>`，一律不计

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
    n_curve = 0
    for tag in tags:
        m = re.search(r'\bd="([^"]*)"', tag)
        if m and len(re.findall(r"[LCQAlcqa]", m.group(1))) >= CURVE_CMDS:
            n_curve += 1
    n_shape = sum(1 for t in tags if re.match(r"<(ellipse|circle|polyline|polygon)\b", t))
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
    n_vec = 0
    for grp in (hor, ver):
        if len(grp) >= VEC_MIN and len(set(round(v / 8.0) for v in grp)) >= VEC_MIN:
            n_vec += len(grp)
    ink = n_curve * 6 + n_shape * 3 + n_bar * 2 + n_vec * 2 + n_dot + n_slant * 2
    return dict(curve=n_curve, shape=n_shape, bar=n_bar, vec=n_vec,
                dot=n_dot, slant=n_slant, ink=ink)


def scan(path):
    """返回这张图里**最差的那一格**的读数 ——&#160;一张图只报它最弱的一环。"""
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
        r = bucket(0, 10 ** 6)
        r.update(name=os.path.basename(path), panels=0, worst=1)
        return r

    reads = [bucket(lo, hi) for lo, hi in panels]
    k = min(range(len(reads)), key=lambda i: reads[i]["ink"])
    r = dict(reads[k])
    r.update(name=os.path.basename(path), panels=len(panels), worst=k + 1)
    return r


def verdict(r):
    """⛔ 只给三档，而且最重的一档要求很严 ——&#160;宁可漏报，不要让人不看它。"""
    if (r["curve"] == 0 and r["shape"] == 0 and r["bar"] == 0
            and r["vec"] == 0 and r["dot"] < 6 and r["slant"] < 4):
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

    rows = [scan(f) for f in files]
    rows.sort(key=lambda r: (r["ink"], -r["text"]))

    print("\n\033[1m▸ 图的「墨水」体检 ——　删掉所有文字之后，还剩多少东西\033[0m")
    print("   %-24s %4s %5s %4s %4s %4s %4s %4s %5s  %s"
          % ("图", "格数", "最差格", "文字", "曲线", "柱子", "向量", "点", "墨水", "判"))
    bad = 0
    for r in rows:
        v = verdict(r)
        if v.startswith("❌"):
            bad += 1
        print("   %-24s %4d %5s %4d %4d %4d %4d %4d %5d  %s"
              % (r["name"], r["panels"], "Ⓐ Ⓑ Ⓒ Ⓓ Ⓔ Ⓕ".split()[r["worst"] - 1]
                 if r["worst"] <= 6 else str(r["worst"]),
                 r["text"], r["curve"], r["bar"], r["vec"], r["dot"],
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
