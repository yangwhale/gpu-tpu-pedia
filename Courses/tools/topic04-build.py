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

<h3>3.1　一个参数到底要占多少字节</h3>

<p>混合精度训练下，<b>每一个参数身上挂着五样东西</b> ——&nbsp;
  <em>而其中只有第一样是「模型本身」。</em></p>

__FIG_OPTIMIZERS__

<div class="note ok"><p>⭐⭐ <b>这一节只要记住一句：权重 2 字节，优化器那边 12 字节。</b></p>
<p><em>⭐ 所以「显存里最大的一块是优化器状态」不是一个修辞 ——&nbsp;
  <b>它就是 2 比 12 这个比。</b></em></p>
<p>⛔ <em>而且这 12 字节<b>全是 fp32</b>。下一小节说为什么它们非 fp32 不可。</em></p></div>

<h3>3.2　为什么主权重必须是 fp32 ——&nbsp;这一条反直觉</h3>

<p>训练到后期，<b>单步的更新量相对权重本身非常小</b>。
  <em>而 bf16 只有 8 位尾数 ——&nbsp;<b>小的更新量加上去会被直接舍掉，等于没更新。</b></em></p>
<p>⭐ <em>所以要留一份 fp32 的<b>真身</b>来累加，
  bf16 那份只是它的<b>投影</b>。</em></p>
<p><span class="sub">⭐ 这也顺带解释了为什么 <b>FP8 训练难</b>：
  精度往下压的时候，<b>哪些量可以压、哪些必须保留高精度，是有讲究的</b>
  ——&nbsp;那是另一课的事。</span></p>

<h3>3.3　换个优化器，账单就跟着变</h3>

<p class="lead">⭐ <b>优化器的选择是一个显存决策，不只是收敛速度的决策。</b>
  <em>——&nbsp;这一点常被忽略，而它恰恰是这一讲要立的那条判据。</em></p>

<!-- ⭐⭐⭐ 2026-09-16 现场：「你这个不得把主流优化器先捋一下？像什么 Adam、
     AdamW，还有 Muon，还有一个最开始的叫 G 什么来着？」
     ⭐ 那个是 SGD。（现场说的「GDN」是 Gated DeltaNet，上一讲的**架构**，
       不是优化器 ——&#160;两个 G 撞车了，值得在正文里替读者点一句。）
     ⛔ 判据：这一节**不按算法怎么算讲，按账单讲** ——&#160;
       更新公式和收敛曲线在这一讲里都不承重。 -->
<div class="note"><p>📌 <b>顺手把谱系捋一遍 ——&nbsp;按「挂几份状态」，不按年份。</b></p>
<ul>
  <li><b>SGD</b> ——&nbsp;<em><b>零状态</b>。算出梯度，往反方向走一步。
    毛病是在狭长的山谷里来回横跳。</em></li>
  <li><b>＋ 动量</b>（Polyak 1964）——&nbsp;<em><b>一份</b>。
    记住上一步往哪走，这一步跟着惯性走。</em></li>
  <li><b>AdaGrad</b>（2011）——&nbsp;<em><b>一份</b>。每个参数一个自己的学习率。
    ⛔ 但它的分母<b>只增不减</b>，训着训着学习率衰减到零，模型不动了。</em></li>
  <li><b>RMSProp</b>（2012）——&nbsp;<em><b>一份</b>，专门来修上面那条：
    把「累加」换成「滑动平均」，老的会被忘掉。</em></li>
  <li><b>Adam</b>（2014）——&nbsp;<em><b>两份</b>。动量和 RMSProp<b>都要</b>，
    再加一个偏差修正。从此成了事实上的默认。</em></li>
  <li><b>AdamW</b>（2017）——&nbsp;<em>还是两份。
    <b>它不是新算法，是在修一个 bug</b> ——&nbsp;见下面那条。</em></li>
  <li><b>Muon</b>（2024）——&nbsp;<em><b>回到一份</b>。
    <b>把二阶矩那一整份拿掉了</b>，改成对更新矩阵做正交化。</em></li>
</ul>
<p><span class="sub">⚠️ 另有三条省状态的岔路，这一讲不展开：
  <b>Adafactor</b> 把 v 分解成一行加一列；<b>Lion</b> 只用梯度的符号、单动量；
  <b>8-bit Adam</b> 把状态本身量化、不改算法。</span></p></div>

<div class="note danger"><p>⛔ <b>Muon 那一份不是白省的 ——&nbsp;三条限制都是作者自己写明的。</b></p>
<ul>
  <li><b>只管二维参数。</b><em>标量、向量，以及
    <b>embedding 和最后那个输出头</b>，仍然走 AdamW ——&nbsp;
    作者明说输入输出层用 AdamW 效果才最好。
    <b>所以整模型省不到那四分之一。</b></em></li>
  <li><b>每一步更慢</b>（原文写明）。<em>⛔ 所以它<b>不是「又快又省」</b> ——&nbsp;
    <b>它把一笔账挪到了另一笔账上</b>：显存那栏减了，单步时间那栏加了。
    划不划算，取决于你现在卡在哪一栏。</em></li>
  <li><b>还要多做几轮矩阵乘。</b><em>那几轮 Newton-Schulz 迭代不是免费的
    ——&nbsp;<b>这又是一次拿算力换显存</b>，跟<a href="#s二">第二节</a>那个决策同一个形状。</em></li>
</ul>
<p>⭐⭐ <em>而它的证据方式值得单独一提：Muon 在 NanoGPT 那个刷速度的公开竞赛里
  把记录提了 <b>35%</b>，此后<b>十二次破纪录、七个不同的人</b>，全都还在用它。
  <b>——&nbsp;要是有人能把 AdamW 调到一样好，换回去就能破纪录，可没人换。</b></em></p>
<p><span class="sub">⭐ <b>顺带一句跟上一讲连起来：DeepSeek-V4 就是用 Muon 训的</b>
  ——&nbsp;论文摘要里列的三大升级之一。</span></p></div>

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
    # ✅ 第三节已写（2026-09-16）——&nbsp;3.1 / 3.2 / 3.3 三小节 ＋ fig4-optimizers。
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

FIGS = {
    "__FIG_OPTIMIZERS__": ("fig-optimizers", "fig4-optimizers.svg",
        'topic04-fig-optimizers.py',
        '⭐⭐ <b>Ⓐ 的形状是这张图的全部</b> ——&nbsp;'
        '<em>三十年一路往上加，<b>2024 年有人往回走了一步</b>。</em><br>'
        '⭐ <em>而 Ⓑ 把「最大的一块是优化器状态」翻译成了一根尺子：'
        '<b>权重只占 2 B，优化器那边占 12 B。</b></em>'),
}

_html = head + HERO + BODY + FOOT

for ph, (what, figs, asks) in PLAN.items():
    assert ph in _html, "正文里没有 %s —— 章写完了要连这一行一起删" % ph
    _html = _html.replace(ph, todo(what, figs, asks))

# ⭐ 这一份是**要投屏讲的**，所以装折叠开关，默认只剩图。
_html = P.place_figs(_html, FIGS)
_html = P.add_figonly_toggle(_html)
P.finish(_html, OUT, SECTIONS, "topic-04.html")
