# -*- coding: utf-8 -*-
"""第七条体检：**节号和小节号对不对得上**（`第 六 节` 里的小节该是 `6.x`）。

⛔⛔ 2026-09-08 立这条的起因：专题三两次整体重编号之后，
   **§五～§八 四节的小节号集体比节号少 1** ——&nbsp;
   页面上明明白白写着「第 六 节」，底下的小节却编成 `5.1 / 5.2 / …`。

⭐⭐ 为什么已有的五条体检一条都没抓到它：

   · 跨节指针体检（`topic02-lint-xref.py`）查的是「`§X.Y` 指的那一节**存不存在**」。
     而 `5.1` 确实存在（它就在那儿，只是长错了地方），
     指向它的 `§5.5b` 也就**顺理成章地通过了** ——&nbsp;
     **一整片错位内部自洽，所以全绿。**
   · 版面体检查几何、可读性体检查字数与加粗、讲义体检查两个文件之间 ——&nbsp;
     都跟编号无关。

   ⭐ **形状：整片一起错位时，任何只做「内部一致性」的检查都会放行。**
     必须引入一个**外部锚点**——&nbsp;这里的锚点就是节标题上那个中文数字。

📌 查两件事：
   ① 每一节里的 `<h3>N.…`，那个 N 必须等于本节徽章上的中文数字
   ② 同一节里不许出现两种不同的 N（半截改过一半是最常见的坏法）
"""
import glob
import io
import os
import re

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WebPages")
CN = "零一二三四五六七八九十"


def cn2int(t):
    """「十一」→ 11、「六」→ 6。只处理这门课用得到的 0–19。"""
    if t.startswith("十"):
        return 10 + (CN.index(t[1]) if len(t) > 1 else 0)
    return CN.index(t)


bad = 0
for path in sorted(glob.glob(os.path.join(WEB, "*.html"))):
    f = os.path.basename(path)
    h = io.open(path, encoding="utf-8").read()
    h = re.sub(r"<!--.*?-->", "", h, flags=re.S)
    hits = []
    # 一节 = 从一个 <section id="sX"> 到下一个 <section 或文末
    for m in re.finditer(r'<section id="s([%s]+)"(.*?)(?=<section |\Z)' % CN, h, re.S):
        sid, body = m.group(1), m.group(2)
        want = cn2int(sid)
        got = sorted({int(x) for x in re.findall(r"<h3>(\d+)\.", body)})
        if not got:
            continue
        if got != [want]:
            hits.append("  ⛔ 第 %s 节（应为 %d.x）里的小节号是 %s"
                        % (sid, want, "／".join("%d.x" % g for g in got)))
    if hits:
        bad += len(hits)
        print("══ %s" % f)
        for x in hits:
            print(x)

print("   %s" % ("✅ 每一节的小节号都跟节号对得上。" if not bad
                 else "⛔ %d 处节号／小节号错位（见上）——&nbsp;重编号漏了半截。" % bad))
