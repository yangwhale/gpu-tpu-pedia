# -*- coding: utf-8 -*-
"""专题二 · 图解简版（topic-02-brief.html）—— 从 topic-02.html 过滤出来，不写新内容。

2026-10-13：给第一次接触 TPU 的客户讲，几乎只有图、几乎没有字；
只留 TPU / GPU 硬件本身，算术强度推导、FlashAttention、编译期/软件、实测对比都不要。
⭐ 只做剪切：图原样搬，图注截到第一句。改原文请改 topic-02，再重跑本脚本。
"""
import re, os, html
SRC = os.path.join(os.path.dirname(__file__), "..", "WebPages", "topic-02.html")
OUT = os.path.join(os.path.dirname(__file__), "..", "WebPages", "topic-02-brief.html")

# (小节标题, [图 id ...]) —— 只剪硬件
PLAN = [
    ("两款芯片，出身不一样", ["s012-figC"]),
    ("同一次取数，两条路", ["s012-fig2-1"]),
    ("整颗芯片", ["ms-g1", "ms-t1"]),
    ("把一个核拆开", ["ms-g2", "ms-t2"]),
    ("访存要等多久，谁来藏这段等待", ["s012-fig3-7", "ms-g7", "ms-t7"]),
    ("并行的层级", ["s012-fig2-2", "s012-fig3-1", "ms-t3", "s012-fig3-2", "s012-fig3-3", "s012-fig3-4"]),
    ("矩阵单元：一条指令吃多大一块", ["ms-g4", "ms-t4", "ms-g8"]),
    ("各自多出来的那一块", ["s012-fig3-5", "s012-fig3-5b"]),
    ("64 颗连在一起", ["s012-fig4-1", "ms-t8"]),
]

s = open(SRC, encoding="utf-8").read()
head = s[:s.index("<body")]
head = re.sub(r"<title>.*?</title>", "<title>TPU 与 GPU · 硬件图解简版</title>", head, flags=re.S)

NOCAP = {"s012-fig3-2", "s012-fig3-5"}   # 截到第一句就成了半句话的，干脆不留

def fig(fid):
    m = re.search(r'<figure[^>]*\bid="%s"' % re.escape(fid), s)
    assert m, fid
    blk = s[m.start(): s.index("</figure>", m.start()) + len("</figure>")]
    def cap(mc):
        t = html.unescape(re.sub(r"<[^>]*>", "", mc.group(1)))
        t = re.sub(r"[⭐⛔⚠️]", "", t)
        t = re.sub(r"——\s*出自.*$", "", t, flags=re.S)
        t = re.sub(r"^\s*(图 [A-Z0-9]+-\d+[a-z]?\s*)+", "", t)
        t = " ".join(t.split())
        t = re.split(r"(?<=[。；：])", t)[0].rstrip("；：。 ")
        if len(t) > 60: t = t[:60].rstrip("，、 ") + "…"
        return "<figcaption>%s</figcaption>" % html.escape(t) if t else ""
    if fid in NOCAP:
        return re.sub(r"<figcaption>.*?</figcaption>", "", blk, flags=re.S)
    return re.sub(r"<figcaption>(.*?)</figcaption>", cap, blk, flags=re.S)

body = ['''<body>
<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">现代 AI 加速器 GPU / TPU 系统课程</a> ／ 专题二 ／ <b>硬件图解简版</b>
    ／ <a href="topic-02.html">完整版 →</a></div>
  <h1>TPU 与 GPU</h1>
  <div class="hook">一张一张图，看两块芯片在硬件上哪里不一样。</div>
  <div class="chips">
    <span class="chip">TPU 侧 <b>v7 (Ironwood)</b></span>
    <span class="chip">GPU 侧 <b>B200 / GB300 NVL72</b></span>
    <span class="chip"><b>%d 张图</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud AI Infra 架构师</p>
</div></div>
''' % sum(len(f) for _, f in PLAN)]
for i, (title, fids) in enumerate(PLAN):
    body.append('<section id="b%d"%s><div class="wrap">\n  <div class="stn big"><span class="badge">%d</span><h2>%s</h2></div>\n'
                % (i, ' style="border-top:none"' if i == 0 else "", i + 1, title))
    body += [fig(f) + "\n" for f in fids]
    body.append("</div></section>\n")
body.append('''<footer><div class="wrap">
  <p><b>专题二 · TPU 与 GPU · 硬件图解简版</b> ｜ <a href="topic-02.html">完整版（含推导）</a> ｜ <a href="index.html">课程总纲</a></p>
  <p>图全部取自完整版。硬件规格只引公开来源。<b>Chris Yang</b> · Google Cloud AI Infra 架构师 —— 文中观点与数据均为个人整理，不代表所属机构。</p>
  <p>本页随 <code>Courses/</code> 采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>。</p>
</div></footer>
</body>
</html>
''')
# 共享样式那块隐形 SVG 原样带上；被剪掉的图里定义、留下的图在引用的 marker/pattern 也补进来
i = s.index('<svg width="0" height="0"')
shared = s[i: s.index("</svg>", i) + 6]
out = "".join(body)
defined = set(re.findall(r'\bid="([^"]+)"', out + shared))
extra = []
for ref in sorted(set(re.findall(r'url\(#([^)"\']+)\)', out)) - defined):
    m = re.search(r'<(marker|pattern|linearGradient|radialGradient|clipPath|filter)\b[^>]*\bid="%s".*?</\1>' % re.escape(ref), s, re.S)
    assert m, "引用了找不到定义的 #" + ref
    extra.append(m.group(0))
if extra:
    shared += '<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>%s</defs></svg>' % "".join(extra)
out = head + out.replace("</div></div>\n", "</div></div>\n" + shared + "\n", 1)
open(OUT, "w", encoding="utf-8").write(out)
print("wrote", OUT, len(out), "bytes,", out.count("<figure"), "figs")
