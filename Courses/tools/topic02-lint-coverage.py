# -*- coding: utf-8 -*-
"""讲义覆盖体检：教材主线里的每个小节，讲义有没有对应的讲稿？

⭐ **为什么需要这个。** 2026-09-06 一天之内同一个病犯了两次：

  1. §9 的教材被精简过，讲义还留着一整段讲已经删掉的「八行取舍表」；
  2. §5.4b（TMA 那一节）在教材里是主线，**讲义里一个字都没有**。

两次都是讲到那儿才被现场抓住的。而已有的对账 lint（`topic02-lint-cues.py`）
只查「滚到 X」的 X 还在不在 ——&nbsp;**X 确实还在**，它查不出
「讲义在讲一个已被删掉的东西」，也查不出「这一节压根没人讲」。

⛔ **形状：讲义是教材的下游。教材加了小节，讲义不会自己长；
   教材删了内容，讲义也不会自己缩。** 而两边都不报错。

这个 lint 只做一件很窄但很准的事：**把两边的小节号对齐数一遍。**
覆盖不了「讲稿内容过期」那一半 ——&nbsp;那半只能靠讲之前翻教材。
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")


def _sections(html, pat):
    out = {}
    for m in re.finditer(pat, html):
        e = html.find("</section>", m.end())
        out[m.group(1)] = html[m.start():e]
    return out


def main():
    doc = io.open(os.path.join(WEB, "topic-02.html"), encoding="utf-8").read()
    lec = io.open(os.path.join(WEB, "topic-02-L200-lecture.html"),
                  encoding="utf-8").read()
    doc = re.sub(r"<!--.*?-->", "", doc, flags=re.S)
    lec = re.sub(r"<!--.*?-->", "", lec, flags=re.S)

    # 教材主线小节：<h3>3.2c　…</h3>，但**折叠里的不算**（折叠不用讲）
    body = re.sub(r"<details.*?</details>", " ", doc, flags=re.S)
    doc_h3 = re.findall(r"<h3>(\d+(?:\.\w+)?)[　\s　]+(.{0,40}?)</h3>", body)
    doc_subs = [a for a, _ in doc_h3]
    # ⭐ 第二条通路：小节号没被提到，但**标题里那句话**被讲到了，也算覆盖。
    #    §1.1「算力除以带宽」就是这种 —— 讲义从头讲到尾，只是没写「1.1」。
    titles = {a: re.sub(r"<[^>]+>|&nbsp;|⭐|\s", "", b) for a, b in doc_h3}
    # 讲稿小节：<h3 class="sec">讲稿 · 3.2c …</h3>
    lec_subs = re.findall(r'<h3 class="sec">讲稿 · (\d+(?:\.\w+)?)[\s　]', lec)

    # ⛔ 判据不能是「讲稿标题里有没有这个号」——&nbsp;讲义并不总按小节切
    #    （§1、§2 就是一整段连着讲的，§4 那段标题干脆叫「64 颗的两种连法」）。
    #    收紧成：**这个小节号在整份讲义里有没有被提到过一次**。
    #    这样既能抓住 §5.4b 那种「零覆盖」，又不会误伤只是标题没写号的。
    lec_txt = re.sub(r"<[^>]+>", " ", lec)
    def covered(x):
        if x in lec_subs:
            return True
        if re.search(r"(?<![\d.])%s(?![\d])" % re.escape(x), lec_txt):
            return True
        t = titles.get(x, "")
        for frag in re.findall(r"[一-龥]{5,}", t):
            if frag in re.sub(r"\s", "", lec_txt):
                return True
        return False

    miss = [x for x in doc_subs if not covered(x)]
    doc_txt = re.sub(r"<[^>]+>", " ", body)
    extra = [x for x in lec_subs
             if x not in doc_subs and not re.search(r"(?<![\d.])%s(?![\d])"
                                                    % re.escape(x), doc_txt)]

    print("\n\033[1m▸ 讲义覆盖体检（教材主线小节 ↔ 讲稿）\033[0m\n")
    print("   教材主线小节 %d 个，讲稿 %d 段" % (len(doc_subs), len(lec_subs)))
    bad = False
    if miss:
        bad = True
        print("   ❌ 教材有、讲义没有讲稿的小节：%s" % "、".join(miss))
        print("      —— 加主线小节必须同时加讲稿，否则讲师翻到那一页手里是空的。")
    if extra:
        bad = True
        print("   ❌ 讲义在讲教材里已经不存在的小节：%s" % "、".join(extra))
        print("      —— 教材精简之后，讲义不会自己跟着缩。")
    if not bad:
        print("   ✅ 每个主线小节都有讲稿，也没有讲稿在讲已删掉的小节。")
    print("   ⚠️ 本 lint 只对小节号，**对不了讲稿内容是否过期** ——"
          " 讲之前仍然要翻一遍教材。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
