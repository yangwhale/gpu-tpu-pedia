#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图在小节里的位置 ——&nbsp;**只报告，不判对错**。

════════════════════════════════════════════════════════════════════
⛔ 2026-09-17 的触发事件：同一个毛病在一天里撞到三次
════════════════════════════════════════════════════════════════════
`fig-descent` / `fig-beststep` / `fig-lr-curve` / `fig-per-byte` 四张图，
原来都排在各自小节**所有正文之后**。后果很具体：读者被一路灌完文字，
最后才看到那张本该先给他的画面 ——&#160;**解释排在被解释的东西后面**。

⭐ 判据：**一张回答「为什么」或「整体长什么样」的图，
  要排在提出那个问题的段落<u>之前或紧邻</u>。**
  排在后面它就从「解释」退化成「补充材料」，而多数人读不到那儿。

════════════════════════════════════════════════════════════════════
⚠️⚠️ 为什么这条**不能**做成断言
════════════════════════════════════════════════════════════════════
量完全书之后发现：末尾的图**多数是对的**。
  · `fig-recompute` 在 2.2 末尾 ——&#160;它是那一节算完账之后的**收束**
  · `fig-step` 在 4.3 末尾、`fig-stability` 在 5.5 末尾 ——&#160;同理
  · `fig-circuit` 在 1.3 末尾 ——&#160;它是把刚讲的话**演一遍**
真正错的只有那四张：它们是**先决条件**，不是收束。

⭐⭐⭐ 所以这条判据要**按图的作用判**，不能只看位置 ——&#160;
  一个只看位置的断言会把四条真问题和七条正常情况一起报，
  然后人就不看它了。**这份脚本只把数字摆出来，判断留给人。**
  （这跟本仓库那条「一个总在误报的闸门等于没有闸门」是同一件事。）
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def cjk(t):
    return len(re.findall(r'[一-鿿]', re.sub(r'<[^>]+>', '', t)))


def audit(builder):
    path = os.path.join(HERE, builder)
    if not os.path.exists(path):
        return []
    s = io.open(path, encoding='utf-8').read()
    if "BODY = '''" not in s:
        return []
    body = s[s.index("BODY = '''"):]
    body = re.sub(r'<!--.*?-->', '', body, flags=re.S)
    parts = re.split(r'(<h3>[^<]*)', body)
    out = []
    for i in range(1, len(parts), 2):
        title = re.sub(r'<[^>]+>', '', parts[i]).strip()[:26]
        # ⚠️ 附录那一节把每张图的出处都列了一遍，它不是「图的位置」，跳过
        if title.startswith(('7.', '8.')):
            continue
        seg = parts[i + 1] if i + 1 < len(parts) else ''
        tot = cjk(seg)
        if tot < 120:            # 太短的小节，位置没有讨论价值
            continue
        for m in re.finditer(r'__FIG_[A-Z_]+__', seg):
            pre = cjk(seg[:m.start()])
            out.append((title, m.group(0)[6:-2], pre, tot))
    return out


def main():
    rows = []
    for b in sorted(os.listdir(HERE)):
        if re.match(r'topic\d+-build\.py$', b):
            rows += [(b,) + r for r in audit(b)]
    if not rows:
        print("\n══ 图位置：没有可量的小节")
        return 0
    print("\n══ 图在小节里的位置（只报告，判断留给人）")
    late = 0
    for b, title, fig, pre, tot in rows:
        pct = 100.0 * pre / tot
        mark = "⚠️" if pct > 75 else "  "
        if pct > 75:
            late += 1
        print("   %s %-26s %-12s 图前已有本节 %3.0f%% 的字" % (mark, title, fig, pct))
    print("   ── %d 张图排在本节 75%% 的字之后。"
          "⭐ 判断标准不是位置，是**这张图是先决条件还是收束** ——" % late)
    print("      先决条件排在后面要改；收束排在后面是对的。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
