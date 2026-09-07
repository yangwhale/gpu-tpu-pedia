# -*- coding: utf-8 -*-
"""第六条体检：**每一页的 <head> 元信息对不对得上它自己**。

⛔⛔ 2026-09-07 立这条 lint 的起因，是现场看浏览器标签发现的：
   **专题三那一页的标题一直显示「专题二 · TPU 与 GPU（L300 · 完整版）」。**

   根因：专题三和专题八的 `<head>`（连 CSS）都是从专题二 L300 整段搬的，
   搬完靠一句 `head.replace("<title>TPU 与 GPU", "<title>注意力演进")` 改标题。
   而 L300 的真实标题是 `<title>专题二 · TPU 与 GPU（L300 · 完整版）</title>`
   ——&nbsp;中间多了「专题二 · 」，**那个模式根本匹配不上**。

⭐⭐ 这条 lint 真正防的不是「标题写错」，是那个**失败模式**：
   **`str.replace` 匹配不上时是静默的** ——&nbsp;不报错、不返回失败标志，
   原样返回。**「改了」和「没改成」在代码里长得一模一样。**
   而 `<head>` 里的东西**页面上看不见**（标题只在标签页上，og 只在分享卡片上），
   所以它能错很久没人发现：og:title / og:description / og:url / og:image
   四条当时全是专题二的 ——&nbsp;**分享出去的卡片标题、摘要、跳转全指向另一讲。**

📌 查四件事，每件都只用页面自己就能判定，不需要维护一份期望值清单：
   ① <title> 存在且不为空
   ② <title> 里写的「专题N」跟文件名里的 topic-0N 对得上
   ③ og:url 结尾就是这个文件自己
   ④ og:image 指的文件真的在盘上
"""
import glob
import io
import os
import re

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WebPages")
CN = "零一二三四五六七八九十"

bad = 0
# ⛔ 标题由 build-all.sh 的 step() 打，这里别再打一遍（打了会出现两行同样的抬头）
for path in sorted(glob.glob(os.path.join(WEB, "*.html"))):
    f = os.path.basename(path)
    h = io.open(path, encoding="utf-8").read()
    errs = []

    m = re.search(r"<title>(.*?)</title>", h, re.S)
    title = m.group(1).strip() if m else ""
    if not title:
        errs.append("没有 <title> 或为空")

    # ② 文件名里的期数 ↔ 标题里的「专题N」
    fm = re.match(r"topic-(\d+)", f)
    if fm and title:
        want = CN[int(fm.group(1))] if int(fm.group(1)) <= 10 else None
        tm = re.search(r"专题([%s]+)" % CN, title)
        if want and tm and tm.group(1) != want:
            errs.append("标题写「专题%s」，但文件是 topic-%s" % (tm.group(1), fm.group(1)))

    om = re.search(r'og:url" content="([^"]*)"', h)
    if om and not om.group(1).endswith("/" + f):
        errs.append("og:url 指向 %s，不是自己" % om.group(1).rsplit("/", 1)[-1])

    im = re.search(r'og:image" content="([^"]*)"', h)
    if im:
        rel = im.group(1).rsplit("WebPages/", 1)[-1]
        if not os.path.exists(os.path.join(WEB, rel)):
            errs.append("og:image 指的文件不存在：%s" % rel)

    if errs:
        bad += 1
        print("   ⛔ %-28s %s" % (f, "；".join(errs)))

print("   %s" % ("✅ 每一页的标题和 og 指向都对得上自己。" if not bad
                 else "⛔ %d 个页面的 head 元信息有问题（见上）。" % bad))
