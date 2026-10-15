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

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-01-lecture.html")
OUT = os.path.join(WEB, "topic-06-lecture.html")
DECK = "topic-06.html"


def _css():
    """⛔ 逐字稿样式必须插在 </style> **之前**。"""
    h = io.open(CSS_SRC, encoding="utf-8").read()
    i, end = h.index("<style>"), h.index("</style>")
    out = h[i:end] + "</style>"
    assert out.count("</style>") == 1
    return out


# ══════════════════════════════════════════════════════════════════
# 十五节讲稿分三个文件写（topic06_lec_a/b/c.py），每个导出 CH 和 PLAN。
# ⛔ 「屏幕」里的图 id 和小节号会被 topic02-lint-cues.py 逐个对账。
# ══════════════════════════════════════════════════════════════════
import topic06_lec_a as LA, topic06_lec_b as LB, topic06_lec_c as LC
CH = LA.CH + LB.CH + LC.CH


def _chapter(i, title, sid, fig, body, dont):
    figline = ("　·　主图 <code>%s</code>" % fig) if fig else \
              "　·　<b>本节无图</b>（故意的）"
    return """
<h3 class="sec" id="c%d">%s</h3>
%s
<div class="say">
%s
<div class="warn"><span class="k">⚠️ 这一节最容易跑偏的地方</span>%s</div>
<p class="board sub">🖥 本节主屏：课件 <a href="%s#%s">%s</a>%s</p>
</div>
""" % (i, title, P.audio_block(OUT, "topic06-lecture-s%d.mp3" % i, "这一节的讲课录音（照着讲的示范）", video="topic06-video-s%d.mp4" % i),
       body, dont, DECK, sid, title.split("（")[0], figline)


PLAN = LA.PLAN + LB.PLAN + LC.PLAN
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
<style>
.shell > main{grid-column:1 / -1}
.plan table td:first-child, .plan table th:first-child{white-space:nowrap}
p.say{background:#fffdf3;border-left:3px solid #f9ab00;padding:10px 14px;
      margin:12px 0;line-height:1.95}
details.deep{margin:18px 0;border:1.5px dashed #9aa0a6;border-radius:10px;padding:6px 14px;background:#fafafa}
details.deep summary{cursor:pointer;padding:6px 0}
p.deephd{font-weight:700;color:#174ea6;margin:14px 0 4px}
p.deepsay{background:#f8f9fa;border-left:3px solid #9aa0a6;padding:8px 14px;margin:8px 0;line-height:1.9}
</style>
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic06-build-lecture.py 生成，不要手改。 -->

<div class="top"><div class="in">
  <span class="who">专题六 · <b>推理</b></span>
  <nav class="tabs">
    <a href="''' + DECK + '''">课件</a>
    <a href="topic-06-lecture.html" class="on">讲义</a>
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
''' + LA.TIME_NOTE + '''
  <table>
<thead><tr><th>段落</th><th>分钟</th><th>能不能砍</th><th>说明</th></tr></thead><tbody>
''' + "".join("<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>\n" % p for p in PLAN) + '''<tr><td><b>合计</b></td><td><b>''' + str(PLAN_SUM) + '''</b></td><td></td><td>第十五节（出处台账）不进课堂，被追问时翻</td></tr>
</tbody></table>
  <p style="margin-top:10px">⛔ <b>上面那张表是计划，下面这张是地板。</b>
  <em>「地板」＝ 把讲稿一字不落念完、再加切图的时间，一句现场发挥都没有，<b>也不含等动画转完一圈的时间</b>（全讲约 3 分钟）。每次 build 现数。</em></p>
  <p>⏭ <b>互动另算：每个 💬 互动按 0.5′。</b>
  <em>「含互动」＝ 地板 ＋ 全部互动；「跳后」＝ 标了「⏭ 时间紧可跳」的互动改成直接说答案。超时先跳这些，别删台词。</em></p>
@@FLOOR@@
</div>

''' + LA.DARK + '''

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
''' + LC.THIN + '''

</main>
</div>
</body></html>''')

# ── 时间账：从成品 HTML 现数，别手写常数 ────────────────────────────
_SAY_RE = r'<p class="say[^"]*">(.*?)</p>'
_say = "".join(re.sub(r"<[^>]+>", "", m) for m in re.findall(_SAY_RE, html, re.S))
_n = len(re.findall(r"[一-鿿]", _say))
_cuts = len(re.findall(r'<p class="board">', html))
_tot = _n / 200.0 + _cuts * 0.25

_secs = [(m.start(), re.sub(r"<[^>]+>", "", m.group(1)))
         for m in re.finditer(r'<h3 class="sec" id="c\d+">(第[零一二三四五六七八九十]+节.*?)</h3>', html)]
assert len(_secs) == len(CH) == len(PLAN), "节数对不上：%d / %d / %d" % (len(_secs), len(CH), len(PLAN))
_secs.append((html.index("🚧 这份讲义还薄"), ""))
_rows, _floor, _over = [], 0.0, []
_ia_tot, _sk_tot, _ia_rows = 0, 0, []    # 只读统计：互动时长，不进任何 assert
for _i in range(len(CH)):
    _t = html[_secs[_i][0]:_secs[_i + 1][0]]
    _cn = len(re.findall(r"[一-鿿]", "".join(re.sub(r"<[^>]+>", "", x)
                                             for x in re.findall(_SAY_RE, _t, re.S))))
    _cc = len(re.findall(r'<p class="board">', _t))
    _f = _cn / 200.0 + _cc * 0.25
    _floor += _f
    _plan = int(re.search(r"（(\d+)′", _secs[_i][1]).group(1))
    _ias = re.findall(r"<p>💬 <b>互动</b>.*?</p>", _t, re.S)
    _ia, _sk = len(_ias), sum("⏭" in x for x in _ias)
    _ia_tot, _sk_tot = _ia_tot + _ia, _sk_tot + _sk
    _ia_rows.append((_secs[_i][1].split("　")[0], _f + _ia * 0.5, _f + (_ia - _sk) * 0.5, _plan))
    if _f > _plan:
        _over.append((_secs[_i][1].split("　")[0], _f, _plan))
    _rows.append("<tr><td>%s</td><td>%s</td><td>%d</td><td>%.1f′</td><td>%d′</td><td>%d / %d</td><td>%.1f′</td><td>%.1f′</td></tr>"
                 % (_secs[_i][1].split("（")[0], format(_cn, ","), _cc, _f, _plan,
                    _ia, _sk, _f + _ia * 0.5, _f + (_ia - _sk) * 0.5))
html = html.replace("@@FLOOR@@",
    '<table><thead><tr><th>节</th><th>台词汉字</th><th>切图</th><th>地板</th><th>计划给的</th><th>互动 / 可跳</th><th>含互动</th><th>跳后</th></tr></thead><tbody>'
    + "".join(_rows)
    + '<tr><td><b>合计</b></td><td colspan="2"></td><td><b>%.0f′</b></td><td><b>%d′</b></td><td><b>%d / %d</b></td><td><b>%.0f′</b></td><td><b>%.0f′</b></td></tr>'
    % (_floor, PLAN_SUM, _ia_tot, _sk_tot, _floor + _ia_tot * 0.5, _floor + (_ia_tot - _sk_tot) * 0.5) + "</tbody></table>")

# ⭐ 两套计划时长（节标题里的、时间表里的）必须一致 —— 专题四踩过「各自自洽、互相对不上」。
_head_min = [int(m) for m in re.findall(r'<h3 class="sec" id="c\d+">第[零一二三四五六七八九十]+节[^（<]*（(\d+)′', html)]
assert _head_min == [p[1] for p in PLAN], ("节标题 %s 与时间表 %s 对不上"
                                            % (_head_min, [p[1] for p in PLAN]))
# ⭐ 2026-09-24 L19：「可压到 N」曾经五节都写得比地板还低 —— 压不到的承诺比不写更坑讲师。
for _p, _r in zip(PLAN, _rows):
    _m = re.match(r"可压到 (\d+)", _p[2])
    _fv = float(re.findall(r"<td>([\d.]+)′</td>", _r)[0])
    assert not _m or int(_m.group(1)) >= _fv, "%s 写「可压到 %s」但地板已经 %.1f′" % (_p[0], _m.group(1), _fv)
_ia_over = [x for x in _ia_rows if x[2] > x[3]]
if _ia_over:
    print("   ℹ️  跳过 ⏭ 互动后仍超计划的节（只报告）：")
    for _nm, _fa, _fs, _pl in _ia_over:
        print("       %-16s 含互动 %5.1f′ · 跳后 %5.1f′ > 给 %2d′" % (_nm, _fa, _fs, _pl))
if _over:
    print("   ⚠️  地板超过计划的节：")
    for _nm, _fl, _pl in _over:
        print("       %-16s 地板 %5.1f′ > 给 %2d′" % (_nm, _fl, _pl))

# ⭐ 2026-09-25 对照构建手册补的两条（手册 09 · 3.3）：
# ① 课件封面「讲约 N 分钟」是手打的，跟这里的计划表是同一个量的两处写法 —— 断两处之间。
_chip = re.search(r"讲约 (\d+) 分钟", io.open(os.path.join(WEB, DECK), encoding="utf-8").read())
assert _chip and int(_chip.group(1)) == PLAN_SUM, \
    "课件封面写「讲约 %s 分钟」，讲义计划表合计 %d′ —— 改一处不改另一处" % (_chip and _chip.group(1), PLAN_SUM)
# ② 台词数字密度（从专题四搬来，只报告）：一串数从耳朵进去台下留不住。
#   判据：这个数是「内容」还是「内容的证据」？证据说「有人量过」就够。
# 已判定「数就是内容」的块登记在各 topic06_lec_*.JUDGED，写明为什么。
_JUDGED = LA.JUDGED + LB.JUDGED + LC.JUDGED
_noisy = []
for _b in re.findall(r'<p class="say">(.*?)</p>', html, re.S):
    _t = re.sub(r"<[^>]+>", "", _b).replace("&nbsp;", " ")
    _c = len(re.findall(r"[一-鿿]", _t))
    if _c < 40:
        continue
    _d = len(re.findall(r"[0-9]", re.sub(r"[A-Za-z]+-?[0-9]+", "", _t)))   # 摘掉 TP4、DEP8 这类名字里的数字
    if _d * 100.0 / _c > 12.0 and not _t.strip().startswith(_JUDGED):
        _noisy.append("%.0f/百字  %s" % (_d * 100.0 / _c, re.sub(r"\s+", " ", _t)[:40]))
if _noisy:
    print("   ⚠️  台词数字偏密（只报告，判断是内容还是证据）：")
    for _x in _noisy:
        print("       " + _x)

# ⛔ 讲稿里报的 TiB/GiB 大数，教材页面上必须也有（讲稿是手打的，漂的一定是讲稿）。
_DECKP = os.path.join(WEB, DECK)
_deck = re.sub(r"\s+", "", io.open(_DECKP, encoding="utf-8").read())
_flat = re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", html))
_BIG = re.findall(r"[0-9]+\.[0-9]+(?:TiB|GiB)", _flat)
_lost = sorted({b for b in _BIG if b not in _deck})
assert not _lost, "讲稿报了教材上没有的数：%s" % "、".join(_lost)

# ⛔ 讲稿里指的动画 id，课件里必须有（topic02-lint-cues 只对 fig- 开头的 id）。
_anims = set(re.findall(r"<code>(anim-[a-z0-9-]+)</code>", html))
_miss = sorted(a for a in _anims if ('id="%s"' % a) not in io.open(_DECKP, encoding="utf-8").read())
assert not _miss, "讲稿指了课件里没有的动画：%s" % "、".join(_miss)

# ⛔ 一块台词对应一个动作：不许超过 200 汉字。
_long = [(len(re.findall(r"[一-鿿]", re.sub(r"<[^>]+>", "", s))), re.sub(r"<[^>]+>", "", s)[:30])
         for s in re.findall(r'<p class="say">(.*?)</p>', html, re.S)]
_long = [x for x in _long if x[0] > 200]
assert not _long, "有台词块超过 200 汉字：%s" % _long

# ⛔ 台上提示不许漏进台词：say 块里每一个字都默认要出口。
_STAGE_ONLY = ("不要在这里", "不要展开讲", "整段跳过", "别念", "台上不用念",
               "时间紧可以", "这一格不要", "别在这里", "讲之前停", "先只放")
_leak = [(w, re.sub(r"<[^>]+>", "", t)[:40]) for t in re.findall(r'<p class="say">(.*?)</p>', html, re.S)
         for w in _STAGE_ONLY if w in t]
assert not _leak, "台上提示漏进台词了：%s" % _leak

# ⛔ 收尾语气只许出现在最后一节，否则会「谢两次幕」。
_CURTAIN = ("数字会过期", "念完就停")
_cstart = [m.start() for m in re.finditer(r'<h3 class="sec" id="c\d+"', html)]
_early = sorted({w for m in re.finditer(r'<p class="say">(.*?)</p>', html, re.S)
                 for w in _CURTAIN if w in m.group(1) and m.start() < _cstart[-1]})
assert not _early, "收尾语气出现在最后一节之前：%s" % "、".join(_early)

html = P.bust_media(html, OUT)          # 录音地址挂内容指纹，重生成后不被缓存成旧的
io.open(OUT, "w", encoding="utf-8").write(html)
print("    互动 %d 个（⏭ 可跳 %d）× 0.5′：含互动 %.0f′ · 跳后 %.0f′（计划 %d′）"
      % (_ia_tot, _sk_tot, _floor + _ia_tot * 0.5, _floor + (_ia_tot - _sk_tot) * 0.5, PLAN_SUM))
print("ok  topic-06-lecture.html  %s 字符 · %d 节 · 台词 %s 字 ＋ %d 切图 ＝ 地板 %.0f′（计划 %d′）"
      % (format(os.path.getsize(OUT), ","), len(CH), format(_n, ","), _cuts, _tot, PLAN_SUM))
