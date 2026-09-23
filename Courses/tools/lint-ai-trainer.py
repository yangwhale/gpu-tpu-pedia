# -*- coding: utf-8 -*-
r"""守着「把讲义丢给 AI」那一格：每一讲正文里都有，而且跟模块里那份逐字一致。

⛔ 两件事要一起守，少一件都不够：
  ① **有没有** —— 漏掉的那一页不会报错，它只是不告诉读者讲义能这么用。
  ② **是不是同一份** —— 手写维护的两页（topic-01 / topic-02-L300）是粘进去的，
     改了 `course_ai_trainer.py` 而没回去重粘，它们会继续说旧版本。
     ⭐ 这一条正是加这个 lint 的理由：**粘贴出来的副本不会跟着源头一起改，
       它只会跟着源头一起被信任。**

用法：`python3 lint-ai-trainer.py [WebPages 目录]`（默认 ../WebPages）
"""
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import course_ai_trainer as AIT


def _norm(s):
    """比对前抹掉排版差异 —— 我们要的是「同一段话」，不是「同样的缩进」。"""
    return re.sub(r"\s+", "", s)


def main(web):
    bad, ok = [], 0
    for deck, lecture in AIT.DECKS:
        p = os.path.join(web, deck)
        if not os.path.exists(p):
            bad.append((deck, "页面不存在"))
            continue
        html = io.open(p, encoding="utf-8").read()
        want = AIT.note(lecture)
        if _norm(want) in _norm(html):
            ok += 1
        elif 'id="ai-trainer"' in html:
            bad.append((deck, "有这一格，但内容跟 course_ai_trainer.py 对不上"
                              " —— 改了模块没回去重粘？"))
        else:
            bad.append((deck, "整格缺失"))
    if bad:
        print("    ⛔⛔ 「把讲义丢给 AI」那一格有问题：")
        for d, why in bad:
            print("       %-22s %s" % (d, why))
        sys.exit("每一讲正文都该有这一格，且与 tools/course_ai_trainer.py 逐字一致")
    print("   ✅ %d 讲的「把讲义丢给 AI」那一格都在，且与模块逐字一致" % ok)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WebPages"))
