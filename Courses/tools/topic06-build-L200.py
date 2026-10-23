import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# -*- coding: utf-8 -*-
r"""专题六 · 推理 —— **主线 L200**（约 90 分钟，要拿上讲台的那一份）。

════════════════════════════════════════════════════════════════════
⭐⭐⭐ 这一份的宪法（2026-10-19 现场原话，改之前先读）
════════════════════════════════════════════════════════════════════
  「你得捡听众关心的内容，把它讲明白，用大白话的方式把它讲透。
    而不是按照你自己的全文逻辑，像流水账一样把所有东西都讲了，没有重点。」

⭐ 落成判据（reviews/T6-L200-STATE.md 有全文）：
  ① 每节回答**一个听众真会问的问题**，标题就是那个问题。
  ② 一个比方 ＋ 一条机制讲透；数字只留能当锚点的一两个。
  ③ 先讲「为什么」，再讲「所以呢」（什么时候用、什么时候亏）。
  ④ 一段话换个位置照样成立 → 它属于 L300。
  ⑤ 事实只取 L300 课件；新查的先进 L300 再进这里。

📌 产物：本文件 → `topic-06.html`（正牌专题六，文件名不带 L200，照专题三先例）；
   档案版是 topic06-build.py → `topic-06-L300.html`。
📌 正文按节拆在 topic06_L200_s<N>.py（导出 BODY ／ FIGS ／ NUM_LOCK）。还没写的节，
   本脚本按下面 SPINE 打一个 🚧 计划框，**不写假正文**。
📌 图脚本只有一套，两档共用。复用 L300 的图用 `C.l300_fig(图id, 新图注)`；
   共用图里不许出现本课节号。
"""
import importlib
import os

import topic03_page as P
import topic06_common as C
import topic06_quiz as QZ

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "WebPages", "topic-06.html")
L300 = "topic-06-L300.html"

from topic06_L200_spine import SPINE
PLAN_SUM = sum(r[3] for r in SPINE)
assert PLAN_SUM == 90, "脊梁合计 %d′，目标 90′" % PLAN_SUM

SECTIONS = [("s" + r[0], r[0], r[1]) for r in SPINE]

head = P.make_head(
    "专题六 · 推理",
    "推理 · 搬一趟，分给更多人，或者少搬",
    "decode 每一步都要把权重和 KV 搬一趟。推理的每一招，要么让这一趟分给更多人，要么干脆少搬。",
    "topic-06.html")
head += C.STYLE


def sec(sid, num, title):
    return ('<section id="%s"><div class="wrap"><div class="stn"><span class="badge">第 %s 节</span>'
            '<h2>%s</h2></div>' % (sid, num, title))


def todo(row):
    num, q, line, mins, figs, src = row
    return ('<div class="note warn"><span class="t">🚧 这一节还没写</span>'
            '计划如下，正文在后续轮次补。</div>\n<ul>\n'
            '  <li><b>一条主线</b>：%s</li>\n  <li><b>计划</b>：%d 分钟</li>\n'
            '  <li><b>打算用的图</b>：%s</li>\n'
            '  <li><b>想深挖</b>：<a href="%s#%s">完整版对应那一节</a></li>\n</ul>'
            % (line, mins, figs, L300, src))


HERO = '''
</head>
<body>

<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">现代 AI 加速器 GPU / TPU 系统课程</a> ／ 主线 ／ 专题六
    ／ <b>推理</b>
    <a class="lecbtn" href="topic-06.en.html" lang="en" hreflang="en">🌐 English</a></div>
  <h1>推理</h1>
  <div class="hook">
    训练一步算一大批，推理一步只出一个字。<br>
    <em>decode 每一步都要把权重和 KV 搬一趟。推理的每一招，要么让这一趟分给更多人，要么干脆少搬。</em>
  </div>
  <div class="chips">
    <span class="chip">前置 <b>专题三</b>（KV cache）<b>专题五</b>（并行）</span>
    <span class="chip">口径 <b>截至 2026-10</b></span>
    <span class="chip">⏱ <b>讲约 %d 分钟</b></span>
    <span class="chip">📖 想深挖：<a href="%s">完整版（L300）</a></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>
''' % (PLAN_SUM, L300)

# R24：开场回顾专题五的勾选题，和 L300 共用 topic06_quiz（同一道题、同一份答案）；
# 挂在第零节标题下、导语之前，折叠，课上点开。答案表下那行小字换成不带 L300 节号的写法。
_QUIZ_HTML, _QUIZ_CSS = QZ.build(
    "推理只剩前向，这张表还能用，但轻重变了：TP 每层那几次 AllReduce 数据不多，卡的是延迟；"
    "EP 也不一定走 AllToAll —— 公开的 TPU 推理专家那步用的是 AllGather 加 ReduceScatter（完整版 6.4）。")
head += "<style>" + _QUIZ_CSS + "</style>\n"

BODY, FIGS, NUM_LOCK = "", {}, []
for _i, _row in enumerate(SPINE):
    _sid, _num, _title = SECTIONS[_i]
    BODY += sec(_sid, _num, _title)
    try:
        _m = importlib.import_module("topic06_L200_s%d" % _i)
    except ModuleNotFoundError:
        BODY += todo(_row) + "\n</div></section>\n"
        continue
    BODY += (_QUIZ_HTML if _i == 0 else "") + _m.BODY + "\n</div></section>\n"
    for _k, _v in getattr(_m, "FIGS", {}).items():
        assert _k not in FIGS, "图占位符 %s 在两节里重名" % _k
        FIGS[_k] = _v
    NUM_LOCK += getattr(_m, "NUM_LOCK", [])

FOOT = '''
<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    上一讲 <a href="topic-05.html">专题五 · 并行策略</a>　·
    想深挖 <a href="%s">专题六完整版（L300）</a><br>
    本页由 <code>Courses/tools/topic06-build-L200.py</code> 拼装，正文在 <code>topic06_L200_s*.py</code>。</p>
</div>
</body></html>''' % L300

_html = head + HERO + BODY + FOOT
_html = P.place_figs(_html, FIGS)
_html = C.postprocess(_html, BODY, NUM_LOCK, strict=False)
assert _html.count("</body>") == 1
_html = _html.replace("</body>", C.REDUCED_MOTION_JS + "</body>")
P.finish(_html, OUT, SECTIONS, "topic-06.html")
