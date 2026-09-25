#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""课程体检：先量能量的，别拿 agent 去数数。

⭐ 这个脚本存在的理由：派 agent 之前先跑它。
  agent 擅长的是「这句话读者看不懂」这种判断，**不擅长数数**（数错了还很自信）。
  凡是脚本能算出确切数字的，都不要写进 agent 的任务里 ——
  把省下来的注意力留给判断题。

用法：
    course-metrics.py <页面.html> [<页面.html> …]
    course-metrics.py <目录>            # 扫目录下所有 .html

它报五样（前三样是**静默失败**，浏览器里出事而文件本身合法）：

  ① 点列表当路径串    <path d="[(1.0, 2.0), …]">  → 浏览器什么都不画，也不报错
  ② foreign-content 逃逸标签  内嵌 SVG 里的 <em>/<i>/<b>/<p>/<br>
                             → 浏览器当场退出 SVG 模式，后面的内容全丢
  ③ 后画的不透明矩形盖住先画的字  SVG 没有 z-index，顺序就是层级
  ④ 图内字号分布        ≥15px 的字占多少（投屏能不能看清）
  ⑤ 正文加粗占比        全都加粗等于都没加粗
"""
import io
import os
import re
import sys

CJK = re.compile(r"[一-鿿]")
ESCAPE_TAGS = ("<em>", "<i>", "<b>", "<p>", "<br", "<font")
# svgsm / svglbl 这类类名的默认字号（改了 CSS 要同步这张表）
CLS_DEFAULT = {"svgsm": 10.5, "svglbl": 13.0, "svgh": 20.0, "svgt": 16.0}
BIG = 15.0            # 「够大」的门槛：投屏缩到 0.6 还有 9px


def cjk(t):
    return len(CJK.findall(t))


def strip_tags(t):
    return re.sub(r"<[^>]+>", "", t)


def svgs(html):
    """切出页面里每一张内嵌 SVG，连同它的 data-fig 名字。"""
    for m in re.finditer(r"<svg\b[^>]*>.*?</svg>", html, re.S):
        s = m.group(0)
        nm = re.search(r'data-fig="([^"]*)"', s)
        if not nm:
            nm = re.search(r'aria-label="([^"]{0,28})', s)
        yield (nm.group(1) if nm else "?"), s


def font_of(attrs):
    m = re.search(r"font-size:\s*([\d.]+)px", attrs)
    if m:
        return float(m.group(1))
    c = re.search(r'class="(\w+)"', attrs)
    return CLS_DEFAULT.get(c.group(1) if c else "", 13.0)


def _covered_text(svg):
    """找「被后画的白色实心矩形盖住」的文字。

    ⭐ SVG 没有 z-index，**后画的盖先画的** —— 所以只用比文档顺序，
      不需要真的渲染。这正是这类 bug 能静默存在的原因：
      文件合法、几何 lint（它只查文字之间撞不撞）也看不见。
    """
    items = []
    for m in re.finditer(
            r'<rect[^>]*?x="(-?[\d.]+)"[^>]*?y="(-?[\d.]+)"[^>]*?'
            r'width="([\d.]+)"[^>]*?height="([\d.]+)"[^>]*?fill="(#[0-9a-fA-F]{3,6}|white)"',
            svg):
        if 'fill="none"' in m.group(0) or "opacity=" in m.group(0):
            continue                      # 透明 / 半透明的遮罩是有意为之
        x, y, w, h = (float(m.group(i)) for i in (1, 2, 3, 4))
        items.append((m.start(), "rect", x, y, w, h))
    for m in re.finditer(r'<text[^>]*?x="(-?[\d.]+)"[^>]*?y="(-?[\d.]+)"[^>]*>(.*?)</text>',
                         svg, re.S):
        t = re.sub(r"<[^>]+>|\s", "", m.group(3))
        if t:
            items.append((m.start(), "text", float(m.group(1)),
                          float(m.group(2)), t[:26]))
    items.sort()
    hits, seen = [], []
    for it in items:
        if it[1] == "text":
            seen.append(it)
        else:
            _, _, x, y, w, h = it
            for _, _, tx, ty, txt in seen:
                # 文字基线落在矩形内部（留 3px 余量，正好压边不算）
                if x + 3 < tx < x + w - 3 and y + 3 < ty < y + h - 3:
                    hits.append(txt)
    return hits


def check_page(path):
    html = io.open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    out = []

    # ── ①②③ 静默渲染失败 ────────────────────────────────────────
    bugs = []
    for fig, s in svgs(html):
        if 'd="[(' in s or 'd="[[' in s:
            bugs.append((fig, "① 点列表当成了 SVG 路径串 —— 浏览器什么都不画"))
        for t in ESCAPE_TAGS:
            if t in s:
                bugs.append((fig, "② 内嵌 SVG 里有 %s —— 浏览器会在这里退出 "
                                  "SVG 模式，后面的内容全丢" % t))
                break
        # ③ ⛔ 判据必须是「**盖住了在它之前画的字**」，不是「是个白色大矩形」。
        #   白卡片本来就该是白的 —— 只按尺寸报，会把每张图都报一遍，
        #   真问题立刻被淹掉（这门课在别处栽过同一跤：假阳性淹掉真问题）。
        #   SVG 是**后画的盖先画的**，所以只需要比较文档顺序。
        covered = _covered_text(s)
        if covered:
            bugs.append((fig, "③ %d 处文字被**后画的不透明矩形**盖住（SVG 没有 z-index，"
                              "顺序就是层级）——　底色要先画，别 append 到最后："
                              "如「%s」" % (len(covered), covered[0])))

    # ── ④ 图内字号 ──────────────────────────────────────────────
    figs = []
    for fig, s in svgs(html):
        tot = big = 0
        for m in re.finditer(r"<text([^>]*)>(.*?)</text>", s, re.S):
            n = len(re.sub(r"\s", "", strip_tags(m.group(2))))
            if not n:
                continue
            tot += n
            if font_of(m.group(1)) >= BIG:
                big += n
        if tot >= 30:
            figs.append((round(100.0 * big / tot), fig, tot))
    figs.sort()

    # ── ⑤ 正文加粗 ──────────────────────────────────────────────
    body = re.sub(r"<svg.*?</svg>", "", html, flags=re.S)
    tot = bold = 0
    hot = []
    for m in re.finditer(r"<(p|li)\b[^>]*>(.*?)</\1>", body, re.S):
        b = m.group(2)
        t = cjk(strip_tags(b))
        if t < 12:
            continue
        k = sum(cjk(strip_tags(x)) for x in re.findall(r"<b>(.*?)</b>", b, re.S))
        tot += t
        bold += k
        # 短句整句加粗＝小标题，不算毛病；只报「长段且过半」
        if t >= 40 and k / float(t) >= 0.5:
            hot.append((round(100.0 * k / t), t,
                        re.sub(r"\s+", " ", strip_tags(b))[:48]))
    hot.sort(reverse=True)

    out.append("══ %s" % name)
    if bugs:
        out.append("  ⛔ 静默渲染失败 %d 处（文件合法，浏览器里出事）：" % len(bugs))
        for fig, why in bugs:
            out.append("     · %-22s %s" % (fig[:22], why))
    else:
        out.append("  ✅ 没有静默渲染失败")

    if figs:
        low = [f for f in figs if f[0] < 40]
        out.append("  📐 图 %d 张，图内 ≥%gpx 文字占比 ≥40%% 的 %d 张；偏小的："
                   % (len(figs), BIG, len(figs) - len(low)))
        for r, fig, n in low[:10]:
            out.append("     · %3d%%  %-24s （图内 %d 字）" % (r, fig[:24], n))
        if not low:
            out.append("     （没有）")

    if tot:
        ok = "✅" if bold * 100 <= tot * 30 else "⚠️"
        out.append("  🔠 正文 %d 汉字，加粗 %d%% %s（目标 ≤30%%）；"
                   "长段（≥40 字）且加粗过半的 %d 段："
                   % (tot, round(100.0 * bold / tot), ok, len(hot)))
        for r, t, txt in hot[:10]:
            out.append("     · %3d%% %4d字  %s" % (r, t, txt))
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(os.path.join(a, f) for f in os.listdir(a)
                            if f.endswith(".html"))
        else:
            paths.append(a)
    for p in paths:
        print(check_page(p))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
