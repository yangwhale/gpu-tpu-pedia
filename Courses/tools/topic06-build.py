import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# -*- coding: utf-8 -*-
"""专题六 · 推理 —— 教材。

════════════════════════════════════════════════════════════════
主线（2026-10-15 定）：decode 每一步都要把权重和 KV 搬一趟
    推理的每一招，要么让这一趟被更多 token 分摊，要么干脆少搬。
════════════════════════════════════════════════════════════════
    零  一次请求走一遍
    一  两个阶段：prefill 吃算力，decode 吃带宽（字节账）
    二  先校准尺子：指标与口径
    三  批处理与调度：把一趟分给更多人
    四  KV cache：账、池、复用
    五  单卡里的两块硬骨头：decode 注意力与 decode MoE
    六  切专家与注意力：TEP ／ DEP
    七  切序列：DCP ／ PCP
    八  拆工作：PD 分离与那一趟 KV
    九  配几台、拆不拆：P:D 配比、AFD、EPD
    十  先选拓扑，再调参数
    十一 投机解码：一趟搬运出好几个 token
    十二 少搬字节：量化与 kernel 融合
    十三 落地：TPU 服务化与编排
    十四 第一天的八问
    十五 出处台账

⭐ 正文按节拆在 topic06_s<N>.py（每节一个模块，导出 BODY ／ FIGS ／ NUM_LOCK），
   为的是一节一节独立改、独立评审。本脚本只负责拼装和锁。
⭐ 第六到十节按 2026-10-15 的决定（DECISIONS）从专题五 §4–§7 起步再讲深：
   专题五的图直接复用（FIGS 指向 fig5-*.svg），所以 build-all 里本脚本排在专题五之后。
⛔ 公开仓库：只用公开资料、本仓库已公开的 GPU runbook ／ tpu-inference 记录、和「本课推导」。
   内部项目的数字、PR 号、客户名一律不进。
⭐ 大纲在 `Courses/专题06-推理.md`，只是计划，不必与页面同步。
"""
import importlib
import io
import json
import os
import re

import topic03_page as P
import course_ai_trainer as AIT
import topic06_quiz as QZ
import topic06_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "WebPages", "topic-06-L300.html")

head = P.make_head(
    "专题六 · 推理 · L300 完整版",
    "推理 · 搬一趟，分给更多 token，或者少搬",
    "prefill 吃算力、decode 吃带宽。推理的每一招，要么让 decode 那一趟搬运被更多 token 分摊，要么干脆少搬。",
    "topic-06-L300.html")
head += C.STYLE

SECTIONS = [
    ("s零", "零", "一次请求走一遍"),
    ("s一", "一", "两个阶段：prefill 吃算力，decode 吃带宽"),
    ("s二", "二", "先校准尺子：指标与口径"),
    ("s三", "三", "批处理与调度：一趟分给更多人"),
    ("s四", "四", "KV cache：账、池、复用"),
    ("s五", "五", "单卡里的两块硬骨头：decode 注意力与 MoE"),
    ("s六", "六", "切专家与注意力：TEP 和 DEP"),
    ("s七", "七", "切序列：DCP 和 PCP"),
    ("s八", "八", "拆工作：PD 分离与那一趟 KV"),
    ("s九", "九", "配几台、拆不拆：P:D 配比、AFD、EPD"),
    ("s十", "十", "先选拓扑，再调参数"),
    ("s十一", "十一", "投机解码：搬一趟，出好几个字"),
    ("s十二", "十二", "少搬字节：量化与 kernel 融合"),
    ("s十三", "十三", "新架构出难题：状态没法按 token 切"),
    ("s十四", "十四", "落地：TPU 服务化与编排"),
    ("s十五", "十五", "第一天的八问"),
    ("s十六", "十六", "出处台账"),
]


def sec(sid, num, title):
    return ('<section id="%s"><div class="wrap"><div class="stn"><span class="badge">第 %s 节</span>'
            '<h2>%s</h2></div>' % (sid, num, title))


def todo(lines):
    return ('<div class="note warn"><span class="t">🚧 这一节还是大纲</span>'
            '下面是计划要讲的东西，还没有展开。</div>\n<ul>\n%s\n</ul>'
            % "\n".join("  <li>%s</li>" % ln for ln in lines))


HERO = '''
</head>
<body>

<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">现代 AI 加速器 GPU / TPU 系统课程</a> ／ 主线 ／ 专题六
    ／ <b>推理</b> ／ L300 完整版</div>
  <h1>推理</h1>
  <div class="hook">
    训练一步算一大批，推理一步只出一个字。<br>
    <em>decode 每一步都要把权重和 KV 搬一趟。推理的每一招，要么让这一趟分给更多 token，要么干脆少搬。</em>
  </div>
  <div class="chips">
    <span class="chip">前置 <b>专题三</b>（KV cache）<b>专题五</b>（并行）</span>
    <span class="chip">口径 <b>截至 2026-10</b></span>
    <span class="chip">⏱ <b>讲约 343 分钟</b></span>
    <span class="chip">📖 <b>L300 完整版</b>：自读、回查用｜讲课看 <a href="topic-06.html">90 分钟主线版</a></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

<div class="wrap">
  <p style="color:var(--gray)">每个数字的出处在文末台账；自己推出来的数，都标了「本课推导」。</p>
</div>
'''

# 2026-10-19：开场热身（回顾专题五）挂在第零节标题下、导语之前，折叠，课上点开。
_QUIZ_HTML, _QUIZ_CSS = QZ.build()
head += "<style>" + _QUIZ_CSS + "</style>\n"

BODY, FIGS, NUM_LOCK = "", {}, []
for _i, (_sid, _num, _title) in enumerate(SECTIONS):
    BODY += sec(_sid, _num, _title)
    try:
        _m = importlib.import_module("topic06_s%d" % _i)
    except ModuleNotFoundError:
        BODY += todo(["（这一节还没写）"]) + "\n</div></section>\n"
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
    上一讲 <a href="topic-05.html">专题五 · 并行策略</a><br>
    本页由 <code>Courses/tools/topic06-build.py</code> 拼装，正文在 <code>topic06_s*.py</code>。</p>
</div>

</body></html>'''

_html = head + HERO + BODY + FOOT
_html = P.place_figs(_html, FIGS)
_html = C.postprocess(_html, BODY, NUM_LOCK)
P.finish(_html, OUT, SECTIONS, "topic-06-L300.html")
