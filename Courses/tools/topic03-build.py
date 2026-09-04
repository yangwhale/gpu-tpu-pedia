# -*- coding: utf-8 -*-
"""专题三 · 注意力演进 —— 教材。

════════════════════════════════════════════════════════════════
⭐ 这一讲的源是 md，不是这个脚本
════════════════════════════════════════════════════════════════
`Courses/专题03-注意力演进.md` **已经写到成品质量**，
所以这里不重写一遍正文 ——&nbsp;用 `md2course.py` 把它转成 HTML。

⛔ **改内容请改那份 md**，不要改这个脚本里的字符串。
   这条跟「L300 是源、L200 用 lift」是同一条规矩：
   同一个职责不要两个载体。

════════════════════════════════════════════════════════════════
状态：第一节是成品，其余是「大纲已展开」
════════════════════════════════════════════════════════════════
2026-09-04：第一节（FlashAttention）按成品质量写完并提到最前，
带我们自己在 v7 上的块大小扫描与三层效率天花板。
其余各节是**详细大纲** ——&nbsp;内容都在、每个数都核过出处，
但没有逐字打磨成讲稿。页面顶部如实标出来。
"""
import io
import os

import md2course

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
SRC_MD = os.path.join(HERE, "..", "专题03-注意力演进.md")
CSS_SRC = os.path.join(WEB, "topic-02-L300.html")
OUT = os.path.join(WEB, "topic-03.html")

md = io.open(SRC_MD, encoding="utf-8").read()

# md 顶部那段 blockquote 元信息（起草历史、前置、给后面垫了什么）是给维护者的，
# ⛔ 不渲染进页面 —— 学员要看的是内容，不是这份文件的修订史。
i = md.index("## ⏱ 一小时怎么讲")
body_md = md[i:]
# 末尾的授权行由页脚统一给
body_md = body_md.split("> 本目录随 `Courses/`")[0]

body, sections = md2course.convert(body_md)

_src = io.open(CSS_SRC, encoding="utf-8").read()
head = _src[:_src.index("</style>") + len("</style>")]
head = head.replace("<title>TPU 与 GPU", "<title>注意力演进", 1)
# 跟专题八同一个补丁：这份 CSS 的 p 没有 margin-top，ul 后面紧跟的段落会贴上去
head += """
<style>
ul + p, ol + p, ul + div.note, ol + div.note, table + p { margin-top: 14px }
pre { background:#f8f9fa; border:1px solid #e8eaed; border-radius:8px;
      padding:14px 16px; overflow-x:auto; font-size:13px; line-height:1.6 }
/* ⛔ 这份 CSS 的 code 带 white-space:nowrap ——&nbsp;nowrap 会把换行折成空格，
   于是 <pre> 里的多行伪代码全挤成一行。必须在 pre code 上把它扳回来，
   顺便去掉行内 code 那层底色和内边距（在块里显得很脏）。 */
pre code { white-space:pre; background:none; border:0; padding:0;
           font-size:inherit; color:inherit }
h4 { margin:18px 0 6px; font-size:15px }
</style>"""

# ⛔ 不用 <ol> ——&nbsp;它会在「零一二三」前面再编一遍 1.2.3.，两套序号打架。
toc = "\n".join(
    '    <li><b>%s</b>　<a href="#%s">%s</a></li>' % (no, a, t)
    for a, no, t in sections if no)

out = [head, '''
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic03-build.py 从
     Courses/专题03-注意力演进.md 生成。**改内容改那份 md，不要改这里。** -->

<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">加速器系统课程</a> ／ 主线 ／ 专题三
    ／ <b>注意力演进</b>
    <a class="lecbtn" href="topic-03-lecture.html">📝 讲义（授课稿）</a></div>
  <h1>注意力演进</h1>
  <div class="en">Three Knobs, Not Thirty Names</div>
  <div class="hook">
    名词多到像各搞各的，<b>但只有三个旋钮可以拧</b>。<br>
    <em>——&nbsp;而在拧它们之前，得先知道「怎么算」这条路已经走到了哪。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    MLA、GQA、SWA、DSA、NSA、CSA、DeltaNet、GDN、KDA……
    这一讲的目标不是记住这些名字，是<b>拿到一把尺子</b>：
    看到任何一个新变体，能立刻说出它在拧哪个旋钮、省了什么、赔了什么。
  </p>
  <div class="chips">
    <span class="chip">前置 <b>专题一 · 专题二</b></span>
    <span class="chip">第一节 <b>FlashAttention 详解</b></span>
    <span class="chip">含 <b>我们自己的 v7 实测</b></span>
    <span class="chip">⏱ <b>约 1 小时</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

<div class="wrap">
  <div class="note warn"><span class="t">🚧 这一讲的成熟度：第一节是成品，其余是「已展开的大纲」</span>
    <b>第一节（FlashAttention）按成品质量写完了</b> ——&nbsp;
    带那张内外循环经典图的展开、我们自己在 v7 上的块大小扫描、
    以及「为什么融合完还是只跑到三成多」的三层拆解。<br>
    <b>其余各节是详细大纲</b>：内容都在、每个数都回到一手论文核过、
    自己推的数标了推导链，<b>但还没有逐字打磨成讲稿</b>。<br>
    <em>⭐ 之所以第一节先熟：那批材料是讲<b>专题二</b>时挖出来的 ——&nbsp;
    按「材料属于哪一讲就写进哪一讲」的规矩，当场落到了这里。</em></div>
</div>

<div class="wrap">
  <div class="note info"><span class="t">这一讲的路线</span>
  <ul>
''' + toc + '''
  </ul></div>
</div>

''', body, '''

<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    硬件背景在 <a href="topic-02.html">专题二 · TPU 与 GPU</a>　·
    量化那一支在 <a href="topic-08.html">专题八 · 精度与量化</a>　·
    <a href="topic-03-lecture.html">📝 讲义</a></p>
  <p style="color:var(--gray);font-size:13px">
    本页由 <code>Courses/专题03-注意力演进.md</code> 生成 ——&nbsp;
    <b>改内容改那份 md</b>。本目录采用 CC BY-NC-SA 4.0。</p>
</div>

</body></html>''']

io.open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("ok  topic-03.html  %s 字符 · %d 节"
      % (format(os.path.getsize(OUT), ","), len([s for s in sections if s[1]])))
