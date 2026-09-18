# -*- coding: utf-8 -*-
r"""把图里的装饰符剥干净 ——&#160;管的是**第二条生产线**。

⭐⭐⭐ 2026-09-18 现场：「内容挺好，但图非常廉价。」
  体检下来最大的一处噪音是图内 emoji（专题四 29 张图里 613 个，光 ⭐ 就 408）。
  RQ1 已经在**绘图基元的文字出口**剥过一次，全站从 613 降到 59。

⛔⛔ 剩下的 59 个在**另一条生产线**上：
  `topic02-fig-*.py` / `topic02-figs-*.py` 这 22 个脚本**自己拼 SVG 字符串**，
  完全不 import `topic03_draw` ——&#160;所以基元那道剥离对它们无效。
  ⭐ 判据（RQ1 栽过一次才立的）：
    **要做「全站生效」的改动，先数清楚有几条生产线。**
    当时我只 grep 了基元那一个文件里的 `<text`，就宣布「所有文字都走一个出口」。

⭐⭐ 为什么做成后处理，而不是去改那 22 个脚本：
  · 22 处改动 ＝ 22 个出错机会；一个后处理只有一处。
  · 而且它**幂等**：第一条生产线的图早就没有装饰符了，扫过去不会变。
  · 源码里留着 emoji 也没关系 ——&#160;每次 build 后都会被剥一遍。
  ⛔ 代价要说清楚：这是**给产物打补丁，不是改根因**。
    哪天那条生产线要重构，应该顺手给它也做一个共用的文字出口。

⛔ 保留 `✓ ✕ ↑ ↓ → ← ⋮` ——&#160;它们是**几何记号，不是装饰**（跟基元那边同一套规矩）。
"""
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ⚠️ "⚠️" 是两个码点（U+26A0 ＋ U+FE0F），必须排在裸 "⚠" 前面先删。
DECOR = ("⭐", "⛔", "⚠️", "⚠", "✅", "❗", "🆕", "📌", "💡", "🔬", "❓", "🎯")


def undecorate(s):
    for ch in DECOR:
        s = s.replace(ch, "")
    return re.sub(r"[ 　]{2,}", " ", s).strip()


def scrub(svg):
    """只动 <text> 的**内容**和 aria-label 的值 ——&#160;不碰任何属性/几何。"""
    n = [0]

    def _txt(m):
        inner = m.group(2)
        new = undecorate(inner)
        if new != inner:
            n[0] += 1
        return m.group(1) + new + m.group(3)

    svg = re.sub(r"(<text\b[^>]*>)(.*?)(</text>)", _txt, svg, flags=re.S)

    def _aria(m):
        new = undecorate(m.group(1))
        if new != m.group(1):
            n[0] += 1
        return 'aria-label="%s"' % new

    svg = re.sub(r'aria-label="([^"]*)"', _aria, svg)
    return svg, n[0]


def main():
    files = sorted(glob.glob(os.path.join(HERE, "fig*.svg")))
    touched, total = 0, 0
    for p in files:
        src = io.open(p, encoding="utf-8").read()
        out, n = scrub(src)
        if n:
            io.open(p, "w", encoding="utf-8").write(out)
            touched += 1
            total += n
    left = 0
    for p in files:
        s = io.open(p, encoding="utf-8").read()
        left += sum(s.count(c) for c in DECOR if c != "⚠")
    print("\n\033[1m▸ 图内装饰符剥离\033[0m")
    print("   扫 %d 张，改了 %d 张 / %d 处；剩余 %d 个。"
          % (len(files), touched, total, left))
    if left:
        print("   ⚠️ 还有残留 ——&nbsp;去看它们在不在 <text> 和 aria-label 之外。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
