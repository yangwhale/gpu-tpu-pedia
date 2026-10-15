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
    三  KV cache：账、池、复用
    四  批处理与调度：把一趟分给更多人
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "WebPages", "topic-06.html")

head = P.make_head(
    "专题六 · 推理",
    "推理 · 搬一趟，分给更多 token，或者少搬",
    "prefill 吃算力、decode 吃带宽。推理的每一招，要么让 decode 那一趟搬运被更多 token 分摊，要么干脆少搬。",
    "topic-06.html")
head += """
<style>
.animgrid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; margin:14px 0;
             width:min(1408px, 94vw); position:relative; left:50%; transform:translateX(-50%) }
.animgrid figure { margin:0; min-width:0 }
.animcell figcaption { font-size:14px; color:var(--gray); margin-top:6px; line-height:1.6 }
.animgrid video { width:100%; border-radius:8px; display:block }
@media (max-width:900px) { .animgrid { grid-template-columns:1fr } }
.lecbtn { display:inline-block; margin-left:14px; font-size:13px; padding:4px 12px; border-radius:14px;
          background:#fff; border:1px solid var(--line); color:var(--blue); text-decoration:none }
.lecbtn:hover { background:var(--blue-l); border-color:var(--blue) }
</style>"""

SECTIONS = [
    ("s零", "零", "一次请求走一遍"),
    ("s一", "一", "两个阶段：prefill 吃算力，decode 吃带宽"),
    ("s二", "二", "先校准尺子：指标与口径"),
    ("s三", "三", "KV cache：账、池、复用"),
    ("s四", "四", "批处理与调度：一趟分给更多人"),
    ("s五", "五", "单卡里的两块硬骨头：decode 注意力与 MoE"),
    ("s六", "六", "切专家与注意力：TEP 和 DEP"),
    ("s七", "七", "切序列：DCP 和 PCP"),
    ("s八", "八", "拆工作：PD 分离与那一趟 KV"),
    ("s九", "九", "配几台、拆不拆：P:D 配比、AFD、EPD"),
    ("s十", "十", "先选拓扑，再调参数"),
    ("s十一", "十一", "投机解码：搬一趟，出好几个字"),
    ("s十二", "十二", "少搬字节：量化与 kernel 融合"),
    ("s十三", "十三", "落地：TPU 服务化与编排"),
    ("s十四", "十四", "第一天的八问"),
    ("s十五", "十五", "出处台账"),
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
    ／ <b>推理</b></div>
  <h1>推理</h1>
  <div class="hook">
    训练一步算一大批，推理一步只出一个字。<br>
    <em>decode 每一步都要把权重和 KV 搬一趟。推理的每一招，要么让这一趟分给更多 token，要么干脆少搬。</em>
  </div>
  <div class="chips">
    <span class="chip">前置 <b>专题三</b>（KV cache）<b>专题五</b>（并行）</span>
    <span class="chip">口径 <b>截至 2026-10</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

<div class="wrap">
  <p style="color:var(--gray)">每个数字的出处在文末台账；自己推出来的数，都标了「本课推导」。</p>
</div>
'''

BODY, FIGS, NUM_LOCK = "", {}, []
for _i, (_sid, _num, _title) in enumerate(SECTIONS):
    BODY += sec(_sid, _num, _title)
    try:
        _m = importlib.import_module("topic06_s%d" % _i)
    except ModuleNotFoundError:
        BODY += todo(["（这一节还没写）"]) + "\n</div></section>\n"
        continue
    BODY += _m.BODY + "\n</div></section>\n"
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
_leak = sorted(set(re.findall(r"__[A-Z][A-Z_0-9]*__", _html)))
assert not _leak, "占位符没落地：%s" % "、".join(_leak)
for _tag in ("h2", "h3", "section", "div"):
    _o = len(re.findall(r"<%s[ >]" % _tag, _html))
    _c = _html.count("</%s>" % _tag)
    assert _o == _c, "<%s> 开 %d 个、闭 %d 个" % (_tag, _o, _c)
# 正文手打的关键数，必须在对应的图里也出现（图是脚本现算的）。比较时去掉千分位逗号。
for _n, _svg in NUM_LOCK:
    assert _n in BODY, "正文里已经没有 %s 了 —— 从 NUM_LOCK 里删掉这一条" % _n
    _g = io.open(os.path.join(HERE, _svg), encoding="utf-8").read().replace(",", "")
    assert _n.replace(",", "") in _g, "正文写 %s，而 %s 里没有这个数 —— 图脚本改了，正文没跟上" % (_n, _svg)
# 一步一停的播放器：照抄 topic05-build.py（那边有来由注释）。动画脚本在每一步落地后打点，
#   写进 manim/steps/<Scene>.json；这里挂到 <video data-pauses>，图注秒数也从同一份 json 取。
STEP_VIDEOS = {"topic06-spec.mp4": "SpecStep"}


def _steps(mp4):
    with io.open(os.path.join(HERE, "manim", "steps", STEP_VIDEOS[mp4] + ".json"), encoding="utf-8") as fh:
        return json.load(fh)


def _hook_steps(html):
    for mp4 in STEP_VIDEOS:
        st = _steps(mp4)
        tag = '<video src="media/%s"' % mp4
        assert html.count(tag) == 1, "%s 在页面上出现 %d 次" % (mp4, html.count(tag))
        html = html.replace(tag, tag + ' data-pauses="%s"' % ",".join("%.2f" % t for t in st["pauses"]))
        i = html.index(tag)
        j = html.index("</figure>", i)
        fig = html[i:j]
        # ⚠️ 「N 秒」后 24 字内必须出现 Manim —— check-loop.py 靠这个正则认图注秒数，写长了它会静默跳过
        fig2, n = re.subn(r"（\d+ 秒无声循环，(.*?)Manim 渲染", lambda m: "（%d 秒，Manim 渲染；默认连续播放，取消勾选「连续播放」就能点「开始」一步一步看%s" % (round(st["duration"]), m.group(1).rstrip("，；")), fig, flags=re.S)
        assert n == 1, "%s 的图注里没找到「N 秒无声循环」" % mp4
        html = html[:i] + fig2 + html[j:]
    return html


STEP_JS = """<style>
.stepbar{display:flex;gap:12px;align-items:center;margin:6px 0 2px;font-size:14px;color:#5f6368;flex-wrap:wrap}
.stepbar button{font:inherit;padding:3px 14px;border-radius:14px;border:1px solid #1a73e8;background:#1a73e8;color:#fff;cursor:pointer}
.stepbar button.gh{background:#fff;color:#1a73e8}
.stepbar button:disabled{background:#fff;color:#9aa0a6;border-color:#dadce0;cursor:default}
.stepbar label{cursor:pointer}
</style>
<script>
// ⭐ 2026-09-25 现场第二轮：「手工播放的时候，我还没看呢，刚翻过去，第一步就跳完了……
//   手工模式第一步也得是让我准备好了，自己点按钮，再给我预备一个 reset 按钮。」
//   所以手工模式下：一上来停在第 0 帧等「开始」；每一步落地停住等「下一步」；
//   播完一圈绕回开头时也停在第 0 帧，不自己再跑一遍。勾「连续播放」才是原来的无缝循环。
// ⭐ 2026-10-14：「把所有的连续播放的 Checkbox 都点上，让它是默认连续播放的样子。」
//   所以默认勾上、一上来就无缝循环；取消勾选才进手工一步一步模式。
document.querySelectorAll('video[data-pauses]').forEach(function(v){
  var P=v.dataset.pauses.split(',').map(Number), i=0, last=0, auto=true;
  var bar=document.createElement('div'); bar.className='stepbar';
  bar.innerHTML='<button type="button" class="nx"></button><button type="button" class="rs gh">⟲ 重来</button>'+
                '<span></span><label><input type="checkbox" checked> 连续播放</label>';
  v.insertAdjacentElement('afterend', bar);
  var nx=bar.querySelector('.nx'), rs=bar.querySelector('.rs'), st=bar.querySelector('span'), au=bar.querySelector('input');
  function toStart(){ v.pause(); v.currentTime=0; i=0; last=0; }
  function show(){
    nx.disabled = auto || !v.paused;
    nx.textContent = (i===0) ? '开始 ▶' : '下一步 ▶';
    if(auto) st.textContent='连续播放中';
    else if(!v.paused) st.textContent='播放中';
    else if(i===0) st.textContent='准备好了就点「开始」';
    else st.textContent='停住了（'+i+' / '+P.length+'），看明白了再点';
  }
  toStart(); v.play().catch(function(){});
  v.addEventListener('timeupdate',function(){
    if(v.currentTime < last-1){                 // 播完一圈绕回了开头
      if(auto){ i=0; } else { toStart(); show(); return; }
    }
    last=v.currentTime;
    if(!auto && i<P.length && v.currentTime>=P[i]){ v.pause(); i++; }
    show();
  });
  v.addEventListener('pause',show); v.addEventListener('play',show);
  nx.addEventListener('click',function(){ v.play(); });
  rs.addEventListener('click',function(){ toStart(); if(auto) v.play(); show(); });
  au.addEventListener('change',function(){ auto=au.checked; if(auto) v.play(); show(); });
  show();
});
</script>
"""
_html = _hook_steps(_html)
assert _html.count("</body>") == 1
_html = _html.replace("</body>", STEP_JS + "</body>")
_html = P.add_figonly_toggle(_html)
P.finish(_html, OUT, SECTIONS, "topic-06.html")
