# -*- coding: utf-8 -*-
r"""专题六 · 推理 —— **讲义（授课稿）**。

════════════════════════════════════════════════════════════════════
⭐ 这一讲的讲法，一句话说清
════════════════════════════════════════════════════════════════════
  decode 每一步都要把权重和 KV 从显存里搬一趟。
  后面每一节都是在回答同一个问题：这一趟怎么分给更多 token，或者干脆少搬。
  ⛔ 每节开头先说「这一招对付的是那一趟里的哪一块」，台下才不会觉得是一堆招数的清单。

════════════════════════════════════════════════════════════════════
⭐⭐ 写这份讲稿时守的三条（跟专题三、四、五讲义同一套）
════════════════════════════════════════════════════════════════════
  ① 讲稿不复述图上的内容，只说**图给不了的**：该看哪一格、为什么这一格重要。
  ② 这是科普：**一个数是内容才报，是证据就说「有人量过」**，具体值留给课件。
  ③ 查来的和推来的要分开说。课件里推导的数都标了「本课推导」，台上照样要说。

📌 骨架 2026-10-16 从 topic05-build-lecture.py 复制；结构检查原样保留。
📌 CSS 沿用专题一讲义。⛔ 不要另写一套。
"""
import io
import os
import re
import topic03_page as P
import topic06_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-01-lecture.html")
OUT = os.path.join(WEB, "topic-06-L300-lecture.html")
DECK = "topic-06-L300.html"


def _css():
    """⛔ 逐字稿样式必须插在 </style> **之前**。"""
    h = io.open(CSS_SRC, encoding="utf-8").read()
    i, end = h.index("<style>"), h.index("</style>")
    out = h[i:end] + "</style>"
    assert out.count("</style>") == 1
    return out


# ══════════════════════════════════════════════════════════════════
# 十六节讲稿分六个文件写（topic06_lec_1…6.py），每个导出 CH、PLAN、JUDGED；节以外的部分在 topic06_lec_meta.py。
# ⛔ 「屏幕」里的图 id 和小节号会被 topic02-lint-cues.py 逐个对账。
# ══════════════════════════════════════════════════════════════════
import importlib
import topic06_lec_meta as LM
_LS = [importlib.import_module("topic06_lec_%d" % _k) for _k in range(1, 7)]
CH = sum((L.CH for L in _LS), [])


def _chapter(i, title, sid, fig, body, dont):
    figline = ("　·　主图 <code>%s</code>" % fig) if fig else \
              "　·　<b>本节无图</b>（故意的）"
    # L35（只改展示）：① 节标题旁挂「计划 · 含互动」小字，数在下面时间账算完后回填；
    # ② 「本节主屏」挪到节首，讲师一进这节就能点过去；③ ❓ 备答从 ⚠️ 框里拆成独立的框。
    plan = re.search(r"（(\d+)′", title).group(1)
    warn, _, ask = dont.partition("<b>❓")
    askbox = ('<div class="ask6"><b>❓' + ask + '</div>') if ask else ""
    return """
<h3 class="sec" id="c%d">%s<span class="tm6">计划 %s′ · @@IA%02d@@</span></h3>
%s
<div class="say">
<p class="board sub">🖥 本节主屏：课件 <a href="%s#%s">%s</a>%s</p>
%s
<div class="warn"><span class="k">⚠️ 这一节最容易跑偏的地方</span>%s</div>
%s
</div>
""" % (i, title, plan, i,
       P.audio_block(OUT, "topic06-lecture-s%d.mp3" % i, "这一节的讲课录音（照着讲的示范）", video="topic06-video-s%d.mp4" % i),
       DECK, sid, title.split("（")[0], figline, body, warn, askbox)


PLAN = sum((L.PLAN for L in _LS), [])
PLAN_SUM = sum(p[1] for p in PLAN)

html = ('''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>专题六 · 讲义（授课稿）</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#129408;</text></svg>">
<meta property="og:title" content="推理 · 讲义">
<meta name="theme-color" content="#5f6368">
''' + _css() + '''
''' + C.LEC_CSS + '''</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic06-build-lecture.py 生成，不要手改。 -->

<div class="top"><div class="in">
  <span class="who">专题六 · <b>推理</b></span>
  <nav class="tabs">
    <a href="''' + DECK + '''">课件</a>
    <a href="topic-06-L300-lecture.html" class="on">讲义</a>
    <a href="topic-05.html">专题五（并行那几刀）</a>
    <a href="index.html">课程总纲</a>
  </nav>
</div></div>

<div class="shell">
<main>

<div class="plan">
  <h2>这份讲义怎么用</h2>
  <p><b>两个用法。</b><br>
    <b>① 上台讲</b>：照着念，台词、切图、每段几分钟都写着。<br>
    <b>② 自学</b>：把这一整份丢给 AI，你会得到一个既有全部背景、
    也知道该怎么讲的助教，让它带你走一遍。</p>
  <p>内容本身在<a href="''' + DECK + '''">课件</a>那边，这一份只管「怎么讲」。<br>
    课程作者 <b>Chris Yang</b> · Google Cloud AI Infra 架构师。</p>
  <table><tbody>
    <tr><td>🎯</td><td><b>这一节要留下什么</b>　讲完之后台下脑子里该剩的那一句。</td></tr>
    <tr><td>🗣</td><td><b>讲稿</b>　<em>黄底</em>的可以照着念。</td></tr>
    <tr><td>🖥</td><td><b>屏幕</b>　该滚到课件的哪张图、放哪段动画。
      <em>⭐ 这些提示挂了 <code>class="board"</code>，构建时逐条对账，
      指了不存在的图或小节号会当场报出来。</em></td></tr>
    <tr><td>⚠️</td><td><b>别讲什么</b>　这一节最容易跑偏的方向。</td></tr>
  </tbody></table>
</div>

<div class="plan">
  <h2>⏱ 时间：主线 ''' + str(PLAN_SUM) + '''′</h2>
''' + LM.TIME_NOTE + '''
  <table>
<thead><tr><th>段落</th><th>分钟</th><th>能不能砍</th><th>说明</th></tr></thead><tbody>
''' + "".join("<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>\n" % p for p in PLAN) + '''<tr><td><b>合计</b></td><td><b>''' + str(PLAN_SUM) + '''</b></td><td></td><td>第十六节（出处台账）不进课堂，被追问时翻</td></tr>
</tbody></table>
  <p style="margin-top:10px">⛔ <b>上面那张表是计划，下面这张是地板。</b>
  <em>「地板」＝ 把讲稿一字不落念完、再加切图的时间，一句现场发挥都没有，<b>也不含等动画转完一圈的时间</b>（全讲约 3 分钟）。每次 build 现数。</em></p>
  <p>⏭ <b>互动另算：每个 💬 互动按 0.5′。</b>
  <em>「含互动」＝ 地板 ＋ 全部互动；「跳后」＝ 标了「⏭ 时间紧可跳」的互动改成直接说答案。超时先跳这些，别删台词。</em></p>
@@FLOOR@@
</div>

''' + LM.DARK + '''

<h3 class="sec">🖥 投屏怎么放</h3>
<div class="say">
  <p>⭐ <b>这一讲有三段动画（连续批、分页 KV、投机解码），全是无声循环。</b>
    <em>放动画时<b>先闭嘴看一遍</b>，等它转完一圈再开口；边放边讲，台下眼睛和耳朵会打架。
    投机解码那段会在每步结束时自己停住，停住的时候再开口。</em></p>
  <p>⭐ <b>课件右下角有个「折叠讲解」按钮（或按 T），投屏前按一下。</b>
    <em>不开的话，台下会低头读字，不听你讲。</em></p>
</div>
'''
        + "".join(_chapter(i, t, sid, fig, body, dont)
                  for i, (t, sid, fig, body, dont) in enumerate(CH))
        + '''
''' + LM.THIN + '''

</main>
</div>
</body></html>''')

C.lecture_finish(html, CH, PLAN, PLAN_SUM, sum((tuple(L.JUDGED) for L in _LS), ()), WEB, DECK, OUT)
