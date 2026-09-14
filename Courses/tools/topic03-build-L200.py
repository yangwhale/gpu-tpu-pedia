# -*- coding: utf-8 -*-
r"""专题三 · 注意力演进 —— **主线 L200**（要拿上讲台的那一份）。

════════════════════════════════════════════════════════════════════
⭐⭐⭐ 这一份跟 L300 的分工：**体裁不同，不是长短不同**
════════════════════════════════════════════════════════════════════
2026-09-14 现场原话（这段是这份文件的宪法，改任何东西之前先读一遍）：

  「把现在的这个专题三改成专题三的 L300。然后从 L300 里边一点一点地蒸馏，
    要这个最好最好的内容，要最精华的部分。然后写一个专题三。
    **重点就是要写一篇完整的故事**，不要像现在这种，就是东一句西一句，
    然后说的你确实是很重点，但是是这个**没有上下文**，对吧？
    你动不动搞了一句谁说的、很著名的一句话、一件事情，就**不是很连贯**。」

  「整个的故事就是从 RNN 开始讲起，然后变成了 Transformer，Transformer 又有
    什么问题。然后呢，之后又出现了长上下文的需求……**每一个阶段谈到某一个
    注意力的时候，就把它展开讲，就按时间线穿起来。**」

  「总的原则是**只要图，尽量少写文字**。」
  「一定要让大家懂得这里边的道理，**尽量要说人话，说大白话**，
    就像台大李宏毅老师那样说大白话。」
  「**这是一本书，是一本讲技术的小说一样。** 一开始就是要有文学素质，
    就好像一个写书写得很好的老师一样。」

⭐ 落成三条可执行的判据：

  ① **每一章只回答一个问题：上一章欠下了什么，这一章拿什么还。**
     ⛔ 一段话如果换个位置照样成立，它就不属于这份 —— 它属于 L300。
     ⭐ 反过来：一句话如果**只在「按时间读」时才成立**，它就是这一份的血肉。

  ② **先有图，再有字。** 写任何一段正文之前先问：这句话图里有没有？
     有就别写第二遍。图注和落点带**是图的一部分**。
     （这条是 2026-09-12 立的老规矩，L300 那边一字不改地照搬过来。）

  ③ **不要「名人名言式」的孤立事实。** 「某某论文说过某某话」这种句子，
     除非它正好是这一步的**动机**或**代价**，否则一律降到 L300 的出处折叠里。
     ⛔ 这正是现场点名的病：单看每句都很重点，连起来没有上下文。

════════════════════════════════════════════════════════════════════
📌 文件名与产物（2026-09-14 R62 定）
════════════════════════════════════════════════════════════════════
  · 本文件            →  `topic-03.html`          ← **正牌专题三，文件名不带 L200**
  · topic03-build-L300.py →  `topic-03-L300.html`  ← 档案版

⛔ 现场明确要求过：「那个文件名不用写 L200，就写那个正常的这个专题三就行。」

════════════════════════════════════════════════════════════════════
⭐ 图从哪来
════════════════════════════════════════════════════════════════════
L300 手上已经有 48 个 `topic03-fig-*.py`、58 张图。这一份**不是重画一套**，
是从那批里挑、并借这次机会**一张一张往精致里改**（现场：「那个图要越画越精致，
就是从现有的 L300 里边抽出来，然后放到 L200 里边，然后来借机优化一下」）。

⛔ 所以图脚本仍然只有一套，两页共用同一批 `fig3-*.svg`。
  改图时要意识到 **L300 那边也会跟着变** —— 这是有意的：
  同一张图在两页上长得不一样，是比「两页各有一张差不多的图」更糟的债。

════════════════════════════════════════════════════════════════════
🚧 当前状态：**骨架（R62）** —— 故事脊梁已立，九章待写
════════════════════════════════════════════════════════════════════
每一章下面现在只有两样东西：**这一章在故事里干什么**，和**它打算用哪几张图**。
⭐ 先把这个给现场看，是因为**脊梁错了，后面每一章都是白写的**。
⛔ 一章一章往里填；填完一章就把该章的「🚧 待写」块整段删掉。
"""
import os

import topic03_page as P
import topic03_family as FAM

HERE = P.HERE
WEB = P.WEB
OUT = os.path.join(WEB, "topic-03.html")

head = P.make_head(
    "专题三 · 注意力演进",
    "注意力演进 · 一条链，被剪断，又绕了回来",
    "从 RNN 讲到今天的混合配比 —— 按时间线穿起来的一个完整故事：每一步都只回答"
    "「上一步欠下了什么，这一步拿什么还」。多图、少字、说大白话。",
    "topic-03.html")

# ⭐ 节号与标题写死在这儿，跟正文里的 <section id> 对齐。
# ⛔ 加节 / 改标题时**两处一起改**。
# ⭐⭐ 标题要说这一节**在故事里干什么**，不是说它讲哪个名词 ——
#   这条是 2026-09-14 从 L300 §三 那个「已经是标配，所以这一讲不展开它」
#   学来的：一个承重的小节挂了一块「无事可看」的牌子。
SECTIONS = [
    ("s一", "一", "一条链 —— 最早的模型是怎么记事的"),
    ("s二", "二", "2017 年那一刀 —— 把链剪断，换成一张表"),
    ("s三", "三", "账单到期 —— 上下文一长，那张表就付不起了"),
    ("s四", "四", "第一次省：把头砍掉 —— MQA 砍过了头，GQA 停在半路"),
    ("s五", "五", "第二次省：干脆不存 K 和 V —— MLA 存的是一份压缩件"),
    ("s六", "六", "换个方向：不是存得少，是别全读 —— 从滑窗到 DSA"),
    ("s七", "七", "绕回原点 —— 还是那块固定大小的记事板"),
    ("s八", "八", "谁也赢不了，那就都要 —— 混合配比"),
    ("s九", "九", "落到机器上 —— 以及那 512 倍是怎么换来的"),
]


def todo(what, figs, asks):
    """一章的「🚧 待写」块。⛔ 这一章写完就把对它的调用整段删掉。"""
    return ('<div class="note warn"><span class="t">🚧 本章待写 ——&nbsp;%s</span>'
            '<p><b>这一章在故事里干什么：</b>%s</p>'
            '<p><b>打算用的图</b>（从 L300 抽，逐张再精修）：<code>%s</code></p></div>'
            % (what, asks, "</code> · <code>".join(figs)))


BODY = '''<section id="x1"><div class="wrap"><div class="stn"><h2>先说这本书在讲一件什么事</h2></div>

<p><b>2020 年的 GPT-3，一次只能记住 2048 个 token。</b>那时候它是个很会接话的
  <b>补全器</b> ——&nbsp;你给一段，它接一段，接得挺像样。</p>
<p>今天你让它干的活完全变了：读完一整个代码库再改一处 bug、连着跑几十轮工具调用、
  <b>记住整场对话里你反复改过的主意</b>。<em>2K 的上下文，连一个文件都读不完。</em></p>

<div class="note ok"><p>⭐⭐ <b>上下文长度就是模型的工作记忆。</b>
  记不住，就什么都干不成。<br>
  ⛔ 而每记住一分，要付的钱是<b>实打实的显存</b> ——&nbsp;
  模型每吐一个字都要回看前面所有字，于是把每个字算出来的 K、V 存着不重算，
  存下来的这一堆就叫 <b>KV cache</b>，它<b>跟着上下文线性长</b>。<br>
  ⭐ 所以这六年注意力的全部演化，只在做一件事：<b>让「记得住」这件事付得起。</b></p></div>

<p class="landing">⭐ <b>这一讲按时间顺序讲，一步一步走。</b>
  每一步只回答两个问题：<b>上一步欠下了什么，这一步拿什么来还。</b>
  <em>——&nbsp;所以读的时候不用记名词，记「谁欠了谁」就够了。</em>
  <br><em>⚠️ 想看推导、消融表、我们自己在 v7 上的实测数据，去
  <a href="topic-03-L300.html">L300 完整版</a>；这一份只讲故事线。</em></p>

__FIG_ARC__

</div></section>


<section id="s一"><div class="wrap"><div class="stn"><span class="badge">第 一 节</span><h2>一条链 ——&nbsp;最早的模型是怎么记事的</h2></div>
<h3>1.1　故事从一个固定大小的小本子开始</h3>
__TODO_1__
</div></section>


<section id="s二"><div class="wrap"><div class="stn"><span class="badge">第 二 节</span><h2>2017 年那一刀 ——&nbsp;把链剪断，换成一张表</h2></div>
<h3>2.1　把「一步步传」换成「一眼全看」</h3>
__TODO_2__
</div></section>


<section id="s三"><div class="wrap"><div class="stn"><span class="badge">第 三 节</span><h2>账单到期 ——&nbsp;上下文一长，那张表就付不起了</h2></div>
<h3>3.1　同一个模型，换个用法，主角就换人了</h3>
__TODO_3__
</div></section>


<section id="s四"><div class="wrap"><div class="stn"><span class="badge">第 四 节</span><h2>第一次省：把头砍掉 ——&nbsp;MQA 砍过了头，GQA 停在半路</h2></div>
<h3>4.1　一份 K/V 给几个头用</h3>
__TODO_4__
</div></section>


<section id="s五"><div class="wrap"><div class="stn"><span class="badge">第 五 节</span><h2>第二次省：干脆不存 K 和 V ——&nbsp;MLA 存的是一份压缩件</h2></div>
<h3>5.1　箱子里那个东西，根本不是 K/V</h3>
__TODO_5__
</div></section>


<section id="s六"><div class="wrap"><div class="stn"><span class="badge">第 六 节</span><h2>换个方向：不是存得少，是别全读 ——&nbsp;从滑窗到 DSA</h2></div>
<h3>6.1　这一步之后，账本不止一本了</h3>
__TODO_6__
</div></section>


<section id="s七"><div class="wrap"><div class="stn"><span class="badge">第 七 节</span><h2>绕回原点 ——&nbsp;还是那块固定大小的记事板</h2></div>
<h3>7.1　兜了一圈，回到第一章那个小本子</h3>
__TODO_7__
</div></section>


<section id="s八"><div class="wrap"><div class="stn"><span class="badge">第 八 节</span><h2>谁也赢不了，那就都要 ——&nbsp;混合配比</h2></div>
<h3>8.1　几层便宜的配一层贵的</h3>
__TODO_8__
</div></section>


<section id="s九"><div class="wrap"><div class="stn"><span class="badge">第 九 节</span><h2>落到机器上 ——&nbsp;以及那 512 倍是怎么换来的</h2></div>
<h3>9.1　纸面省下来的，机器上不一定省得到</h3>
__TODO_9__
</div></section>
'''

# ── 九章的「这一章干什么 / 用哪些图」───────────────────────────────
# ⭐ 这张表就是**故事脊梁本身**，现场先看它。⛔ 章写完了就把对应这行删掉。
PLAN = {
    "__TODO_1__": ("RNN 与它的三个痛",
        ["fig3-rnn.svg（三张）"],
        "立起这本书的主角：<b>一块固定大小的小本子</b>。它够省，但有三个治不好的病 ——&nbsp;"
        "信息要沿着链一格一格爬、爬远了会淡、而且<b>一次只能动一格，加速器喂不饱</b>。"
        "⭐ 这一章要让人记住那块小本子的<b>形状</b>，因为第七章还会一模一样地回来。"),
    "__TODO_2__": ("MHA：把循环拿掉，代价是什么",
        ["fig3-mha-swap.svg", "fig3-mha-qkv.svg", "fig3-mha-heads.svg",
         "fig3-attn-invented.svg", "fig3-why-softmax.svg", "fig3-transformer.svg"],
        "这一刀<b>换来</b>的是：任意两个字之间只隔一步，整段话一次算完。"
        "<b>欠下</b>的是两样 ——&nbsp;要算的格子从 n 变成 n²，而且"
        "<b>那块小本子没了</b>：模型改成把所有历史原封不动留着。"
        "⭐ 后面七章全在还这第二笔债。"),
    "__TODO_3__": ("为什么是现在",
        ["fig3-flip.svg", "fig3-motives.svg", "第一题 488 GiB 那笔账"],
        "2017 年就欠下的债，为什么 2024 年才要命？"
        "因为<b>用法变了</b>：训练时一次吞一整段，解码时一次只吐一个字 ——&nbsp;"
        "<b>同一个模型，主角从算力换成了访存</b>。"
        "⭐ 这一章把 488 GiB 那笔账当场算给台下看，让「付不起」变成一个具体的数。"),
    "__TODO_4__": ("MQA 与 GQA",
        ["fig3-wiring.svg", "fig3-mqa-why.svg", "fig3-knob1.svg",
         "fig3-copy-matrix.svg"],
        "最直接的省法：<b>几个头合用一份 K/V</b>。"
        "MQA 一步砍到只剩一份，省得最狠，<b>也最容易掉点</b>；"
        "GQA 停在中间，成了这些年的默认选择。"
        "⭐ 现场点名要讲清：<b>MQA 为什么会变痴呆</b>。"),
    "__TODO_5__": ("MLA",
        ["fig3-mla-why.svg", "fig3-lowrank.svg", "fig3-absorb.svg",
         "fig3-mla-credit.svg", "fig3-rope.svg"],
        "换一条路：<b>不存 K/V，存一份能现场展开成 K/V 的压缩件</b>。"
        "⭐ 题眼是「吸收」——&nbsp;把括号挪一下，展开那一步就不用真的做。"
        "⚠️ 还要顺手把 RoPE 的职责摆正：<b>谁在前谁在后是因果掩码管的，"
        "RoPE 管的是两个字隔多远</b>。"),
    "__TODO_6__": ("稀疏：SWA → NSA → DSA",
        ["fig3-knob2.svg", "fig3-swa-why.svg", "fig3-nsa-why.svg",
         "fig3-dsa-why.svg", "fig3-csa-why.svg", "fig3-chicken.svg",
         "fig3-overmix.svg"],
        "前两章省的是「存多少」，这一章省的是「读多少」——&nbsp;<b>KV 一个字节都没少</b>。"
        "⭐⭐ 现场点名这一章要画图展开 DSA：<b>top-2048 到底是怎么挑出来的</b>、"
        "省的是哪一样、<b>trade-off 是什么</b>、以及"
        "<b>为了少读反而多花的那部分算力和不规整访存</b>。"),
    "__TODO_7__": ("线性注意力 / 状态类模型",
        ["fig3-notepad.svg", "fig3-erase.svg", "fig3-assoc.svg",
         "fig3-at-gallery.svg", "fig3-duality.svg", "fig3-two-brackets.svg",
         "fig3-chunkwise.svg"],
        "最狠的一条：<b>干脆别存了，换回一块固定大小的记事板</b>。"
        "⭐ 这是全书的回旋点 ——&nbsp;它跟第一章那块小本子<b>是同一个形状</b>，"
        "差别在于这一次我们知道怎么<b>擦</b>、也知道怎么把串行的它榨出并行度。"),
    "__TODO_8__": ("混合",
        ["fig3-hybrid.svg", "fig3-ratio-grid.svg", "fig3-ratio-or-count.svg",
         "fig3-nope.svg", "39 行可排序模型表"],
        "三条路各有各的短板，于是今天几乎所有人的答案都一样：<b>掺着用</b>。"
        "⭐ 这一章给出配比那根轴、两头为什么都不好，"
        "再用一张表把台下正在用的那些模型<b>一个个对上号</b>。"),
    "__TODO_9__": ("落到硬件 + 收尾",
        ["fig3-perstep.svg", "fig3-tpu-gap.svg", "fig3-tpu-fix.svg",
         "fig3-landing.svg", "fig3-gun.svg", "fig3-chronicle.svg"],
        "<b>纸面省下来的，机器上不一定省得到。</b>这一章讲这些聪明办法落到真机上"
        "卡在哪、怎么克服，最后把封面挂的那把枪打响 ——&nbsp;"
        "<b>512 倍到底是怎么换来的</b>。"),
}

FIGS = {
    "__FIG_ARC__": ("fig-arc", "fig3-arc.svg", 'topic03-fig-arc.py',
        '⭐ <b>这就是整本书的路线图。</b>六段，每段只问三件事：图啥、带来了什么、'
        '欠下了什么。<b>底下那两条数才是落点</b> ——&nbsp;能跑的长度涨了，'
        '同一长度下要付的钱降了，<b>两件事同时发生，才有今天的 agent。</b>'),
}

out = [head, '''
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic03-build-L200.py 生成。
     **正文写在那个脚本的 BODY 常量里** —— 改内容改那里，别改这个产物。 -->

<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">加速器系统课程</a> ／ 主线 ／ 专题三
    ／ <b>注意力演进</b></div>
  <h1>注意力演进</h1>
  <div class="en">A Chain, Cut Loose, and Coming Back Around</div>
  <div class="hook">
    2020 年的 GPT-3 只记得住 <b>2048</b> 个 token；今天的模型记 <b>100 万</b>。<br>
    <em>——&nbsp;这六年的注意力演进，讲的就是这 <b>512 倍</b> 是怎么换来的。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    这一讲<b>按时间顺序讲一个完整的故事</b>：一条链怎么被剪断、
    剪断之后欠下了什么账、这笔账后来被四拨人用四种办法还、
    以及为什么最后<b>又绕回了那块固定大小的小本子</b>。
    <em>多图、少字、说大白话。</em>
  </p>
  <div class="chips">
    <span class="chip">前置 <b>专题一 · 专题二</b></span>
    <span class="chip">读法 <b>从头顺着读</b></span>
    <span class="chip">要推导与实测 <b>看 L300</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

<div class="wrap">''' + FAM.nav("topic-03.html") + '''</div>

''', BODY, '''

<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    硬件背景在 <a href="topic-02.html">专题二 · TPU 与 GPU</a>　·
    推导 · 消融 · 一手出处在 <a href="topic-03-L300.html">本讲 L300 完整版</a><br>
    本页由 <code>Courses/tools/topic03-build-L200.py</code> 生成 ——&nbsp;
    <b>正文写在那个脚本里</b>。
  </p>
</div>

</body></html>''']

_html = "\n".join(out)

for ph, (what, figs, asks) in PLAN.items():
    assert ph in _html, "正文里没有 %s —— 章写完了要连这一行一起删" % ph
    _html = _html.replace(ph, todo(what, figs, asks))

_html = P.place_table(_html)
_html = P.place_figs(_html, FIGS)
P.finish(_html, OUT, SECTIONS, "topic-03.html")
