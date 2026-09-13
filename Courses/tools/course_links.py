# -*- coding: utf-8 -*-
r"""把正文里的 arXiv 编号变成可点的链接。

⭐⭐ 2026-09-13 现场点的：「引用的那些论文得在教材里边，把可点击的 link
   都放里边，有愿意多学的人可以去点开看。」

⛔ 为什么用**后处理**而不是在每处手写 `<a>`：
   这一讲正文里有 37 处 arXiv 编号，而且每加一个机制就会多几处。
   手写等于**每次都要记得加**，而忘了不报错 ——&nbsp;
   ⭐ 判据（本仓库反复出现的那条）：**能让机器每次都做对的事，别交给记性。**

⚠️ 三个不能碰的地方，都踩过同型的坑：
   ① `<svg>` 里面 ——&nbsp;那是图，HTML 的 `<a>` 进不去（要用 SVG 自己的 `<a>`）；
   ② 已经在 `<a>` 里的 ——&nbsp;嵌套 `<a>` 是非法 HTML，浏览器会悄悄拆掉；
   ③ 标签**属性**里的数字 ——&nbsp;`href="...2412.19437"` 被套一层就全毁了。
   所以下面是**先切 svg、再切 a、再切标签**，只在剩下的纯文本块上替换。

📌 口径：arXiv ID 形如 `NNNN.NNNNN`（4 位年月 ＋ 4–5 位序号）。
   本课有几篇是 2026 年的（2604 / 2606 / 2607），编号规则一样，链接照常成立。
   ⚠️ 只匹配「点号前恰好 4 位数字」——&nbsp;所以 `94.74`、`2.3.4`、`131.072`
   这些都不会被误伤（这是刻意收紧的，宁可漏也别错套）。
"""
import re

ARXIV = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?![\d.])")
_SVG = re.compile(r"(<svg.*?</svg>)", re.S)
_A = re.compile(r"(<a\b.*?</a>)", re.S)
_TAG = re.compile(r"(<[^>]+>)")


def _sub_text(s):
    return ARXIV.sub(
        r'<a href="https://arxiv.org/abs/\1" target="_blank" rel="noopener">\1</a>',
        s)


def linkify_arxiv(html):
    """只在「不在 svg 里、不在 a 里、不在标签里」的纯文本上加链接。"""
    out = []
    for i, seg in enumerate(_SVG.split(html)):
        if i % 2:                       # <svg>…</svg> 原样放回
            out.append(seg)
            continue
        sub = []
        for j, s2 in enumerate(_A.split(seg)):
            if j % 2:                   # 已经是链接，别嵌套
                sub.append(s2)
                continue
            sub.append("".join(
                t if k % 2 else _sub_text(t)
                for k, t in enumerate(_TAG.split(s2))))
        out.append("".join(sub))
    return "".join(out)


def count(html):
    """给构建日志报个数 ——&nbsp;⭐ 数字要能看见，不然加没加上没人知道。"""
    return len(re.findall(r'href="https://arxiv\.org/abs/', html))
