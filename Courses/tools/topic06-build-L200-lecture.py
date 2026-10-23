# -*- coding: utf-8 -*-
r"""专题六 · 推理 —— **L200 讲义（90 分钟口播稿）**。

⭐ 讲法一句话：decode 每一步都要把权重和 KV 搬一趟；每节回答一个听众真会问的问题，
   答案都落回「这一趟怎么分给更多人，或者少搬」。
⭐ 稿子在 topic06_L200_lec.py；时间账和全部断言跟 L300 讲义共用（topic06_common.lecture_finish）。
📌 产物 `topic-06-lecture.html`，对的课件是 `topic-06.html`；L300 那份是 topic06-build-lecture.py。
"""
import os
import re
import topic03_page as P
import topic06_common as C
import topic06_L200_lec as L

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
OUT = os.path.join(WEB, "topic-06-lecture.html")
DECK = "topic-06.html"
CSS_SRC = os.path.join(WEB, "topic-01-lecture.html")
CH, PLAN = L.CH, L.PLAN
PLAN_SUM = sum(p[1] for p in PLAN)


def _css():
    import io
    h = io.open(CSS_SRC, encoding="utf-8").read()
    return h[h.index("<style>"):h.index("</style>")] + "</style>"


def _chapter(i, title, sid, fig, body, dont):
    figline = ("　·　主图 <code>%s</code>" % fig) if fig else ""
    plan = re.search(r"（(\d+)′", title).group(1)
    return """
<h3 class="sec" id="c%d">%s<span class="tm6">计划 %s′ · @@IA%02d@@</span></h3>
<div class="say">
<p class="board sub">🖥 本节主屏：课件 <a href="%s#%s">%s</a>%s</p>
%s
<div class="warn"><span class="k">⚠️ 这一节最容易跑偏的地方</span>%s</div>
</div>
""" % (i, title, plan, i, DECK, sid, title.split("（")[0], figline, body, dont)


html = ('''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>专题六 · 讲义（90 分钟）</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#129408;</text></svg>">
<meta property="og:title" content="推理 · 讲义">
<meta name="theme-color" content="#5f6368">
''' + _css() + C.LEC_CSS + '''
<!-- R26：手机 390 宽下两张计划表（脊梁表 448 px、时间账 662 px）撑破页面；只在本讲义加横滚，L300 讲义不动 -->
<style>.plan{overflow-x:auto;-webkit-overflow-scrolling:touch}</style>
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic06-build-L200-lecture.py 生成，不要手改。 -->

<div class="top"><div class="in">
  <span class="who">专题六 · <b>推理</b></span>
  <nav class="tabs">
    <a href="''' + DECK + '''">课件</a>
    <a href="topic-06-lecture.html" class="on">讲义</a>
    <a href="topic-06-L300.html">完整版（L300）</a>
    <a href="index.html">课程总纲</a>
    <a href="topic-06-lecture.en.html" lang="en" hreflang="en">🌐 English</a>
  </nav>
</div></div>

<div class="shell">
<main>

<div class="plan">
  <h2>这份讲义怎么用</h2>
  <p><b>两个用法。</b><br>
    <b>① 上台讲</b>：照着念，台词、切图、每段几分钟都写着。<br>
    <b>② 自学</b>：把这一整份丢给 AI，让它当助教带你走一遍。</p>
  <p>内容本身在<a href="''' + DECK + '''">课件</a>那边，这一份只管「怎么讲」。
    课程作者 <b>Chris Yang</b> · Google Cloud AI Infra 架构师。</p>
</div>

<div class="plan">
  <h2>⏱ 时间：主线 ''' + str(PLAN_SUM) + '''′</h2>
''' + L.TIME_NOTE + '''
  <table>
<thead><tr><th>段落</th><th>分钟</th><th>能不能砍</th><th>一条主线</th></tr></thead><tbody>
''' + "".join("<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>\n" % p for p in PLAN) + '''<tr><td><b>合计</b></td><td><b>''' + str(PLAN_SUM) + '''</b></td><td></td><td></td></tr>
</tbody></table>
  <p style="margin-top:10px">⛔ <b>上面是计划，下面是地板</b>（台词 200 字/分钟 ＋ 每次切屏 15 秒 ＋ 每个互动半分钟，每次 build 现数）。</p>
@@FLOOR@@
</div>
'''
        + "".join(_chapter(i, *c) for i, c in enumerate(CH))
        + '''
''' + L.THIN + '''

</main>
</div>
</body></html>''')

C.lecture_finish(html, CH, PLAN, PLAN_SUM, L.JUDGED, WEB, DECK, OUT)
