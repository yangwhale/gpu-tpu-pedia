# -*- coding: utf-8 -*-
r"""专题四 · 反向与优化器 —— **主线**（要拿上讲台的那一份）。

════════════════════════════════════════════════════════════════════
⭐⭐⭐ 这个专题在整门课里干什么：**把账补全**
════════════════════════════════════════════════════════════════════
专题一跟着一个 token 走完了前向，算出「装不进任何一块卡」。
**但那只是账单的一小半。**

真正训练一步，显存里同时压着四样东西：权重、梯度、优化器状态，
外加一大堆**不能算完就扔的中间激活**。这一讲把这张账单补全 ——
补完之后会发现一件多数人想不到的事：

  ⭐⭐⭐ **最大的那一块，既不是权重也不是激活 ——&#160;是优化器状态。**

而这，正是所有并行策略要解决的第一个问题（→ 专题五）。

════════════════════════════════════════════════════════════════════
⛔ 写这一份之前先读的三条（都是专题三用血换来的）
════════════════════════════════════════════════════════════════════
① **HTML 就是源。** 正文直接写在这个文件里，`.md` 只是大纲，
   **不做 md → HTML 的转换**。改内容改这里。
② **要人「看」的留图，要人「读」的留字。** 图上只放非画不可的东西；
   一句话能说清的对照，不值一整格面板。
③ **判据⑩：两处说同一件事，挨着图的那一处赢。** 正文不要复述图注／图内文字，
   `build-all.sh` 的查重会当场抓。

📌 脚手架（head / 图装配 / 锚点 / 吸顶目录 / 折叠开关）复用
  `topic03_page.py`，画法基元复用 `topic03_draw.py` ——&#160;
  ⛔ **别另起一套**，那是专题二、三、二x 已经共用的同一份。
"""
import os

import topic03_page as P

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
OUT = os.path.join(WEB, "topic-04.html")

head = P.make_head(
    "专题四 · 反向与优化器",
    "反向与优化器 · 把账补全，最大的一块是优化器状态",
    "训练一步到底要付什么：反向的 3 倍算力、不能扔的激活、以及那个"
    "多数人都低估了的大头 —— 每参数 16 字节的优化器状态。"
    "补完这张账，ZeRO 的三级分法就不用背了。",
    "topic-04.html")

# ⭐ 节号与标题写死在这儿，跟正文里的 <section id> 对齐。⛔ 改一处要改两处。
# ⭐⭐ 标题说的是这一节**在故事里干什么**，不是它讲哪个名词。
SECTIONS = [
    ("s零", "零", "前向只是半张账单"),
    ("s一", "一", "反向要付什么 —— 三倍算力，外加一堆扔不掉的中间结果"),
    ("s二", "二", "第一个真正的「决策」—— 拿算力换显存，换多少算划算"),
    ("s三", "三", "真正的大头：优化器状态"),
    ("s四", "四", "这张账单直接决定了并行策略长什么样"),
    ("s五", "五", "一个完整 step 的总账 —— 以及峰值出现在哪一刻"),
]


def todo(what, figs, asks):
    """一章的「🚧 待写」块。⛔ 这一章写完就把对它的调用整段删掉。"""
    return ('<div class="note warn"><p>🚧 <b>本章待写 ——&nbsp;%s</b></p>'
            '<p><b>这一章在故事里干什么：</b>%s</p>'
            '<p><b>打算用的图：</b><code>%s</code></p></div>'
            % (what, asks, "</code> · <code>".join(figs)))


BODY = '''<section id="s零"><div class="wrap"><div class="stn"><span class="badge">第 零 节</span><h2>前向只是半张账单</h2></div>

<p class="lead">专题一跟着一个 token 走完了前向，最后算出一句话：<b>装不进任何一块卡。</b>
  <em>⛔ 可那只是账单的<b>一小半</b>。</em></p>

<p>真正训练一步，显存里是<b>同时</b>压着四样东西的：<b>权重</b>、<b>梯度</b>、
  <b>优化器状态</b>，外加一大堆<b>不能算完就扔的中间激活</b>。
  <em>这一讲把这张账单补全。</em></p>

<p class="landing">⭐⭐⭐ <b>而补完之后会发现一件多数人想不到的事：
  最大的那一块，既不是权重，也不是激活 ——&nbsp;<u>是优化器状态。</u></b></p>

<div class="note ok"><p>⭐ <b>这一讲按「谁最大」的顺序讲，而不是按「训练流程」的顺序。</b></p>
<p><em>流程的顺序是：前向 →&nbsp;反向 →&nbsp;更新。
  但那个顺序会让最大的那一块<b>最后才出场</b>，而它恰恰是决定一切的那一块。</em></p>
<p>⭐⭐ <em>所以这一讲的每一节只回答同一个问题：
  <b>这一项有多大，能不能省，省它要拿什么去换。</b></em></p></div>

</div></section>


<section id="s一"><div class="wrap"><div class="stn"><span class="badge">第 一 节</span><h2>反向要付什么 ——&nbsp;三倍算力，外加一堆扔不掉的中间结果</h2></div>

<p class="lead">反向传播不是「再跑一遍」。<em>它要付两样东西：
  <b>大约两倍于前向的算力</b>，以及 ——&nbsp;更要命的 ——&nbsp;
  <b>一路累加、不能提前扔掉的中间激活</b>。</em></p>

__TODO_S1__

</div></section>


<section id="s二"><div class="wrap"><div class="stn"><span class="badge">第 二 节</span><h2>第一个真正的「决策」——&nbsp;拿算力换显存，换多少算划算</h2></div>

<p class="lead">这是全课第一次出现<b>真正的取舍</b>：
  <em>不是「有没有更好的办法」，而是<b>两样东西只能选一样，你选哪个</b>。</em></p>

__TODO_S2__

</div></section>


<section id="s三"><div class="wrap"><div class="stn"><span class="badge">第 三 节</span><h2>真正的大头：优化器状态</h2></div>

<p class="lead">前面两节都在跟激活较劲。<em>可把账摊开一看 ——&nbsp;
  <b>那个从头到尾一言不发的角色，才是最大的一块。</b></em></p>

__TODO_S3__

</div></section>


<section id="s四"><div class="wrap"><div class="stn"><span class="badge">第 四 节</span><h2>这张账单直接决定了并行策略长什么样</h2></div>

<p class="lead">这一节是通向<b>专题五</b>的桥。
  <em>⭐ 把上面几项按大小排一遍，你会发现 ——&nbsp;
  <b>ZeRO 那三级，正是照着这个顺序来的。</b></em></p>

__TODO_S4__

</div></section>


<section id="s五"><div class="wrap"><div class="stn"><span class="badge">第 五 节</span><h2>一个完整 step 的总账 ——&nbsp;以及峰值出现在哪一刻</h2></div>

<p class="lead">把前向（专题一）＋ 反向 ＋ 更新合成一张表。
  <em>⭐ 而其中最有用的一问不是「总共多少」，是
  <b>「峰值出现在哪一个时刻」</b>。</em></p>

__TODO_S5__

</div></section>
'''

# ⛔ 每写完一章，把对应这一行整段删掉（连同正文里的占位符）。
PLAN = {
    "__TODO_S1__": (
        "三倍算力 · 激活为什么必须留着 · 梯度通信",
        ["fig4-3x", "fig4-act-bill"],
        "立住两笔账：算力的 3× 从哪来；以及专题一那个 106.75 GiB "
        "其实<b>已经是打过折的</b> —— 原始账单要大得多。"
        "⛔ 顺序很重要：<b>先看到原始账单，第二节那个决策才有分量。</b>"),
    "__TODO_S2__": (
        "全量重算 · 每字节付多少 FLOPs · 同一判据在 2022 年给出相反答案",
        ["fig4-recompute", "fig4-per-byte"],
        "全课第一个「决策」：三分之一算力换掉四十分之三十九的显存。"
        "⭐ 选择性重算的判据<b>只有一个数</b>：每省一字节要付多少 FLOPs。"
        "⛔ 而同一条判据在 2022 年给出的是<b>相反</b>的答案 —— "
        "这一条比结论本身值钱。"),
    "__TODO_S3__": (
        "每参数 16 字节 · 其中 12 字节是 fp32 · 切一刀之后账完全变了",
        ["fig4-optstate", "fig4-zero-split"],
        "本讲的落点。⭐ 但要讲清楚一件事："
        "那 16 字节是<b>「每张卡都有全套」</b>的账，"
        "<b>切一刀就完全变了</b>；而 ZeRO-3 下 bf16 权重"
        "<b>根本不常驻，它是流动的</b>。"),
    "__TODO_S4__": (
        "按大小排序 → ZeRO 三级 · 重算 / offload / CP 各针对哪一项",
        ["fig4-ladder"],
        "通向专题五的桥。⭐ 落点一句话："
        "<b>谁最大、谁最少被用到，就先切谁</b> —— "
        "理解了这张账单，ZeRO 的分级就不用背。"),
    "__TODO_S5__": (
        "总账表 · 峰值在哪一刻 · 梯度累积改的是时间线不是总量",
        ["fig4-step"],
        "收尾。⭐ 要回答三问：一个 step 总共多少 FLOPs；"
        "<b>峰值出现在哪个时刻</b>（这一问比「总共多少」有用）；"
        "全部加起来要多少 device。"),
}

# ⚠️ `make_head()` 只吐到 </style> 为止 ——&#160;`</head>` / `<body>` / 封面区
#   要自己接。（第一次没接，`add_figonly_toggle` 当场 assert 挂掉，
#   报的是「页面结构不对」——&#160;看着像装配问题，其实是缺标签。）
HERO = '''
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic04-build.py 生成。
     **正文写在那个脚本的 BODY 常量里** ——&nbsp;改内容改那里，别改这个产物。 -->

<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">加速器系统课程</a> ／ 主线 ／ 专题四
    ／ <b>反向与优化器</b></div>
  <h1>反向与优化器</h1>
  <div class="en">Completing the Bill: Backward, Recompute, and the Optimizer</div>
  <div class="hook">
    专题一算完前向，结论是「装不下」。<br>
    <em>——&nbsp;可那只是<b>半张账单</b>。而剩下那半张里最大的一块，
    既不是权重也不是激活。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    这一讲把训练一步的账<b>补全</b>：反向要付的三倍算力、
    那些不能算完就扔的中间激活、以及<b>每参数 16 字节的优化器状态</b>。
    <em>补完之后，ZeRO 那三级分法就不用背了 ——&nbsp;它是从这张账单里长出来的。</em>
  </p>
  <div class="chips">
    <span class="chip">前置 <b>专题一</b></span>
    <span class="chip">后续 <b>专题五 · 并行策略</b></span>
    <span class="chip">读法 <b>按「谁最大」读，不按流程读</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

'''

FOOT = '''
<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    前向那半张账在 <a href="topic-01.html">专题一 · 一个 Token 的一生</a>　·
    这张账怎么切，在 <b>专题五 · 并行策略</b>（未上线）<br>
    本页由 <code>Courses/tools/topic04-build.py</code> 生成 ——&nbsp;
    <b>正文写在那个脚本里</b>。
  </p>
</div>

</body></html>'''

_html = head + HERO + BODY + FOOT

for ph, (what, figs, asks) in PLAN.items():
    assert ph in _html, "正文里没有 %s —— 章写完了要连这一行一起删" % ph
    _html = _html.replace(ph, todo(what, figs, asks))

# ⭐ 这一份是**要投屏讲的**，所以装折叠开关，默认只剩图。
_html = P.add_figonly_toggle(_html)
P.finish(_html, OUT, SECTIONS, "topic-04.html")
