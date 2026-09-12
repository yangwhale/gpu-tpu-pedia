# -*- coding: utf-8 -*-
"""专题三 · 注意力演进 —— 教材（HTML 就是源）。

════════════════════════════════════════════════════════════════
⛔⛔ 2026-09-04 改过一次路线，别再改回去
════════════════════════════════════════════════════════════════
我一度把这一讲做成「md 是源、脚本转 HTML」。**现场明确否掉了**：

    「没有人让你把 MD 转成 HTML。那个 MD 只是一个简要的大纲，
      你也不用次次都去更新 MD。我们要写的是 HTML ——
      什么东西你都直接写 HTML，HTML 才是教材和讲义。」

所以现在的规矩跟专题一、二一致：

  · `Courses/专题03-注意力演进.md`  = **简要大纲**，随手记，不必与页面同步
  · 这个文件里的 BODY              = **教材本体**，改内容改这里

⛔ 不要再加 md→HTML 的转换器（`md2course.py` 已删）。理由不是它做不到，
   是它把「大纲」和「成品」绑成了同一个载体 ——
   大纲要能随手涂改，成品要能精细排版，**这两件事的自由度本来就不一样**。

════════════════════════════════════════════════════════════════
状态：第一节是成品，其余是把大纲搬进来的骨架
════════════════════════════════════════════════════════════════
第一节（FlashAttention）按成品质量写完，带我们自己在 v7 上的块大小扫描
与三层效率天花板。其余各节内容都在、数都核过出处，但没有逐字打磨。

⭐ 往下写就直接在 BODY 里写。想加折叠写 <details>，想加图写 <figure>
   —— **这里没有语法天花板。**
⛔ BODY 里有 % 号（百分比），所以它**只能用普通字符串**，
   不要在它上面做 %-格式化。
"""
import io
import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-02-L300.html")
OUT = os.path.join(WEB, "topic-03.html")

_src = io.open(CSS_SRC, encoding="utf-8").read()
# ⛔⛔ 2026-09-07 修一个**静默了很久**的 bug：整个 <head> 是从专题二 L300 搬的，
#    原先靠一句 `head.replace("<title>TPU 与 GPU", "<title>注意力演进")` 改标题。
#    **那个模式根本匹配不上** ——&nbsp;L300 的真实标题是
#    `<title>专题二 · TPU 与 GPU（L300 · 完整版）</title>`，中间多了「专题二 · 」。
#    于是这一页在浏览器标签上一直显示「专题二 · TPU 与 GPU（L300 · 完整版）」，
#    og:title / og:description / og:url / og:image 也全是专题二的 ——&nbsp;
#    **分享出去的卡片标题、摘要、跳转全指向另一讲。**
#
# ⭐⭐ 形状：**`str.replace` 匹配不上时是静默的**，不报错、不返回失败标志，
#    就是原样返回。**「改了」和「没改成」在代码里长得一模一样。**
#    ⛔ 判据：**任何「替换模板里某一段」的操作，都必须断言它真的改了。**
#    下面 `_sub()` 就是干这个的 —— 换不到直接让构建挂掉。
#
# 📌 顺带发现 `img/og-topic-02.jpg` **这个文件根本不存在**（img/ 里只有
#    og-topic-01.jpg）。所以那条 og:image 是死链，专题二 L300 自己也带着。
#    这一页直接把 og:image 摘掉 ——&nbsp;**没有图，好过指一张 404 的图**。


def _sub(text, pattern, repl, what):
    """替换 + 断言真的替换了。⛔ 不要退回裸的 str.replace。"""
    new, n = re.subn(pattern, repl, text, count=1)
    assert n == 1, "改不动 %s —— 模板变了？（模式：%s）" % (what, pattern)
    return new


head = _src[:_src.index("</style>") + len("</style>")]
head = _sub(head, r"<title>.*?</title>", "<title>专题三 · 注意力演进</title>", "<title>")
head = _sub(head, r'<meta property="og:title" content="[^"]*">',
            '<meta property="og:title" content="注意力演进 · 从 RNN 的一个补丁，到今天各家的混合配比">', "og:title")
head = _sub(head, r'<meta property="og:description" content="[^"]*">',
            '<meta property="og:description" content="名词多到像各搞各的，但只有三个旋钮可以拧 —— MLA、GQA、SWA、DSA、GDN、KDA 全部收进同一把尺子，每一条都回到一手 config 核过。">', "og:description")
head = _sub(head, r'<meta property="og:url" content="[^"]*">',
            '<meta property="og:url" content="https://gist.higcp.com/Courses/WebPages/topic-03.html">', "og:url")
# ⛔ og:image 曾经指向一个不存在的文件（img/og-topic-02.jpg），整条摘掉。
# ⭐⭐ 这一条**故意不用 _sub**：它是「有就清掉」，不是「必须换成什么」。
#   2026-09-07 实测：源模板那条已经先修好了，于是这里没得摘 —— _sub 当场断言失败。
#   ⛔ 判据：**「必须改到」用断言，「有就清理」不用。** 把两者混在一起，
#     护栏会在上游修好之后反过来把构建搞挂。
head = re.sub(r'\s*<meta property="og:image"[^>]*>(\s*<meta property="og:image:(width|height)"[^>]*>)*',
            "", head)
# ⛔⛔ 2026-09-08：查残留之前**先把注释剥掉**。
#   起因：我在 port-microscope 注入的 CSS 里写了一句注释，正文提到了
#   「topic-02-L300.html」这个文件名 —— 这条断言当场把整个 build 挂掉。
#   ⭐ 判据：**护栏要查的是元数据（title / og:*），不是碰巧提到兄弟文件的散文。**
#     注释是写给人看的，把它算进「残留」是把护栏的口径放得太宽。
_probe = re.sub(r"/\*.*?\*/|<!--.*?-->", "", head, flags=re.S)
assert "TPU 与 GPU" not in _probe and "topic-02" not in _probe, "head 里还有专题二的残留"
# 跟专题八同一个补丁：这份 CSS 的 p 没有 margin-top，ul 后面紧跟的段落会贴上去
head += """
<style>
ul + p, ol + p, ul + div.note, ol + div.note, table + p { margin-top: 14px }
/* ⭐ pre / pre code 的样式**已经修在 L300 的 CSS 源里**了（2026-09-04），
   这里不再重复一份 —— 重复的 CSS 跟重复的正文是同一类问题。 */
h4 { margin:18px 0 6px; font-size:15px }

/* ⭐ 2026-09-08：提示框去彩底那段**已经改到共用 CSS 源里了**
   （topic-01.html 与 topic-02-L300.html），本讲不再单独覆盖 ——
   ⛔ 同一条规则留两份，迟早只改一份。 */
</style>"""

# ⭐ 路线目录：节号与标题写死在这儿，跟正文里的 <section id> 对齐。
# ⛔ 加节 / 改标题时**两处一起改** —— 这是手写 HTML 的代价，认了。
# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-07 立的规矩 ——&nbsp;**这条管整门课，不只是专题三**
# ══════════════════════════════════════════════════════════════════
# 现场原话：「现在是立规矩的时候，不着急往后写，先把第一节搞明白。
#            这些乱七八糟的字不应该折叠起来吗？」
#
# 【规矩】**教材正文只留三样：一句 highlight、一张图、一句图注。**
#
#   1. 任何超过两三句的解释、清单、推导链、出处 ——&nbsp;
#      **要么折叠（`<details class="aside">`），要么搬进讲义。**
#   2. **判据：关着所有折叠从头读一遍，正文必须依然通顺、完整、不缺前提。**
#      折叠里装的是「想深挖的人才要的」，不是「读者必须知道但我懒得排版的」。
#   3. summary 要写清**里面是什么、值不值得展开**。
#      ⛔ 不许写「详情」「更多」「展开看看」——&nbsp;那等于让读者赌一把。
#   4. 讲述性的内容（怎么讲、讲多久、现场怎么收口、不许展开什么）
#      **一律进讲义**，教材里一个字都不留。
#
# ⭐ 为什么这条值钱：教材是**给学员自己读的**，而人读长文默认是扫的。
#   一屏三段以上的解释文字，读者不会读，只会跳 ——&nbsp;
#   **于是那些字既没被读到，又把图挤下去了，两头亏。**
#   折叠不是把内容藏起来，是**把「必须读」和「想读才读」分开**。
#
# ⛔ 新增章节前先回来读这一段。这门课后面每一节都按它写。
# ⛔⛔⛔ 2026-09-08 抓到的一类**体检查不出来**的污染，写在这儿当规矩。
#
#   两次整体重编号（插 §零 RNN、插 §一 MHA）都用 `§(\d+)\.` 机械 +1。
#   ⭐ 但这份文档里的「§X.Y」**有两种含义**：
#        ① 本课的小节号 ——&nbsp;该跟着改
#        ② **被引论文自己的小节号** ——&nbsp;⛔ 绝对不能改
#   结果 6 处论文节号被 +2：DeepSeek-V3 的 sec.2.1/4.2 变成了 §4.1/§6.2、
#   Kimi K3 的 sec.2.1.2 变成 §4.1.2、DeepSeek-V4 的 sec.2.3 变成 §4.3……
#
# ⭐⭐ **最毒的地方：污染后的号码恰好都落在真实存在的本课小节上。**
#   于是跨节指针体检**全绿** ——&nbsp;它只查「这个号存不存在」，
#   查不了「这个号该不该是本课的号」。读者按图索骥会翻到一节毫不相干的内容。
#
# 📌 现在的规矩：**论文小节号一律写成 `sec. X.Y`，本课小节号才用 `§X.Y`。**
#   ⛔ 以后再重编号，只动 `§`，`sec.` 一个字都别碰。
SECTIONS = [
    # ⭐ 2026-09-08：在 §零 之后插入新的 §一「MHA」，其余整体后移一位。
    # ⛔ 这是第二次整体重编号了。两次都靠 topic02-lint-xref.py 兜底 ——
    #   ⚠️ 而它**只查渲染后的 HTML**，所以 fig 脚本里的 §X.Y 也得一起扫
    #   （第一次就漏在那儿）。改节号 = build 脚本 ＋ 全部 fig 脚本一起改。
    ("s零", "零", "起点：RNN —— 被注意力补的那个东西"),
    ("s一", "一", "MHA —— 把循环拿掉，代价是什么"),
    ("s二", "二", "一切从长上下文说起 —— 两条独立的动机"),
    ("s三", "三", "FlashAttention —— 已经是标配，所以这一讲不展开它"),
    ("s四", "四", "骨架：三个旋钮，是同一个账本的三个面"),
    ("s五", "五", "旋钮①：让每一份更小"),
    ("s六", "六", "旋钮②：KV 照存，但每步只读一部分"),
    ("s七", "七", "旋钮③：换回一个固定大小的状态 —— 线性注意力"),
    ("s八", "八", "旋钮④（其实是元旋钮）：混合"),
    ("s九", "九", "代价：没有免费的午餐"),
    ("s十", "十", "落到硬件（本专题的落点）"),
    ("s十一", "十一", "收尾：把谱系放回时间线"),
    ("s十二", "十二", "这个专题明确不讲"),
]
# ⛔ 2026-09-07 删掉了「这一讲的路线」那个目录块。现场：「这一小部分也不用，
#   看上去这个风格跟专题二和专题一都不一样，你保持一致。」
#   ——&nbsp;专题一、二都是 hero 之后**直接进正文**，没有目录。
# ⭐ 判据：**一致性本身就是内容的一部分。** 同一套课里某一讲多长一块，
#   读者第一反应不是「这里更贴心」，是「这一讲跟别的不一样」——&nbsp;
#   而那个疑问会一直挂着，直到他确认没有别的差异为止。
# 📌 SECTIONS 仍然要留着：跨节指针体检（topic02-lint-xref.py）靠它对小节号。


# ══════════════════════════════════════════════════════════════════
# 正文 —— **这里就是教材本体，直接改**
# ⛔ 不要去改 md 再生成回来；md 是大纲，这里是成品。
# ══════════════════════════════════════════════════════════════════
BODY = '''<section id="x1"><div class="wrap"><div class="stn"><h2>这个专题在讲一件什么事</h2></div>

<!-- ⭐⭐⭐ 2026-09-08 换立意。原来的开篇是「名词很多，但只有三个旋钮」——
     那是**一把尺子，不是一个故事**。尺子能让人认名词，
     但回答不了「为什么这六年非得这么走」。
     现场原话：「先把骨架打好，把故事线捋清楚了 —— 这是一个什么样的故事？
     怎么样向前推进的？每一次变革都是图啥？带来了什么？以及对现在这个
     agentic 的意义 —— **长上下文才带来了智能，原来 2K 的上下文玩个毛？**」
     ⛔ 三个旋钮那把尺子没有丢，它降级成了「故事的骨架」，在 §四 才登场。 -->

<p><b>2020 年的 GPT-3，上下文是 2048 个 token。</b>那时候的模型是个很聪明的<b>补全器</b>
  ——&nbsp;你给它一段话，它接得很好。</p>
<p>今天你让它做的事完全不一样了：<b>读完整个代码库再改一处 bug</b>、
  <b>连着跑几十轮工具调用</b>、<b>记住整场对话里你反复改过的主意</b>。</p>

<div class="note ok"><p>⭐⭐ <b>上下文长度就是 agent 的工作记忆。</b>
  记不住，就什么都干不成 ——&nbsp;<b>2K 的上下文，连一个文件都读不完。</b><br>
  ⛔ 而<b>每加长一分上下文，KV cache 就线性涨一分</b>。<br>
  ⭐ <b>所以这六年注意力的全部演化，是为了让「记得住」这件事付得起。</b></p></div>

<!-- ⭐ 2026-09-12 加这一句。口述一遍之后发现的：
     「只有一个账本」那句原来只出现在**骨架图之后**（作为图的落点），
     可它其实是<strong>导航</strong> —— 台下需要在听任何内容之前就拿到这个框，
     否则前三节会被当成三个并列的技术名词听。
     ⛔ 但不照抄那一段：两处写同一段话必然漂。
        这里只放<strong>承诺</strong>（一句、不带张量形状）；
        骨架图后面那处仍是<strong>兑现</strong>（带 S 在哪、三个面各是什么）。
     ⭐ 判据：**同一句话可以出现两次，前提是两次的职责不同。** -->
<p class="landing">⭐⭐ 先把这一讲的框给你 ——&nbsp;<b>从头到尾只有一个账本：KV cache。</b>
  <em>后面那三个旋钮，是同一个账本的三个面。</em>
  <br>（三个面分别是什么，看完下面这张骨架图再说。）</p>

<!-- ⭐⭐ 2026-09-12 加两道课前题（现场点的）。样式 .guess/.opts/.oplab 与
     劝阻用的 details.more.preclass 都是从专题二 L300 的 <style> 继承来的，
     本页不另写 CSS。JS 也照搬那一份，三条坑都已经在那边踩平：
       ① 按 .guess 逐个绑定，别用 id（页面上会有第二道题）
       ② 等 DOMContentLoaded（内联脚本只看得见它上面的 DOM）
       ③ 以 .opts 为一组，组内互斥、组间独立，**所有组都选过才展开答案**
          —— 配对选项会泄题：认出一半，另一半自动跟着定了。
     ⛔⛔ 第二题 (b) 不给具体数字：**VMEM 带宽是未公开规格**，
        而屋脊点 × 算力 = 带宽，给了屋脊点等于把那个数反推出来。
        这里只到「落在几十这一档」为止 —— 跟专题二的公开边界严格一致。
        ⭐ 而且这恰好是个好考点：它就是专题二教的那四句问法之一。 -->
<details class="more preclass">
  <summary>⛔ 课前勿点 ——&nbsp;开讲前的两道热身题<span class="why">现场会一起做；提前看答案＝自己剧透，这两道题就废了</span></summary>
  <div class="body">

  <div class="guess">
    <h3>第一题 · 一个用户的 KV cache，到底有多大</h3>
    <p class="q"><b>DeepSeek V3</b>（671B，61 层，128 个注意力头，head_dim 128），
      <b>128K 上下文、单个用户、bf16 存</b>。<br>
      <span class="qs">同样这个形状，换四种注意力，KV cache 各是多大？
      <b>四行分开选，各选各的。</b></span></p>

    <div class="oplab">(a) 最朴素的 <b>MHA</b>（128 个 KV 头）</div>
    <div class="opts">
      <button data-g="0">61 GiB</button>
      <button data-g="1">122 GiB</button>
      <button data-g="2" data-right>488 GiB</button>
      <button data-g="3">976 GiB</button>
    </div>

    <div class="oplab">(b) <b>GQA-8</b>（KV 头砍到 8）</div>
    <div class="opts">
      <button data-g="0">3.8 GiB</button>
      <button data-g="1">8.6 GiB</button>
      <button data-g="2" data-right>30.5 GiB</button>
      <button data-g="3">61 GiB</button>
    </div>

    <div class="oplab">(c) <b>MQA</b>（KV 头砍到 1）</div>
    <div class="opts">
      <button data-g="0" data-right>3.8 GiB</button>
      <button data-g="1">8.6 GiB</button>
      <button data-g="2">15.3 GiB</button>
      <button data-g="3">30.5 GiB</button>
    </div>

    <div class="oplab">(d) <b>MLA</b>（V3 真实用的方案）</div>
    <div class="opts">
      <button data-g="0">3.8 GiB</button>
      <button data-g="1" data-right>8.6 GiB</button>
      <button data-g="2">30.5 GiB</button>
      <button data-g="3">61 GiB</button>
    </div>

    <div class="rev">
      <p><b>488　/　30.5　/　3.8　/　8.6　GiB。</b><br>
        <span class="qs">算法就一条：<b>每 token 每层要留下几个数</b>，
        乘 61 层、乘 2 字节、乘 131,072 个 token。<br>
        MHA ＝ 2×128×128 ＝ 32,768 →&nbsp;<b>488 GiB</b>；
        GQA-8 ＝ 2×8×128 ＝ 2,048 →&nbsp;<b>30.5</b>（16×）；
        MQA ＝ 2×1×128 ＝ 256 →&nbsp;<b>3.8</b>（128×）；
        MLA ＝ 压缩维 512 ＋ RoPE 64 ＝ 576 →&nbsp;<b>8.6</b>（56.9×）。</span></p>
      <p><b>⭐⭐ 这道题真正的题眼在 (c) 和 (d) 的大小关系：</b>
        <span class="qs"><b>MQA 只要 3.8 GiB，比 MLA 的 8.6 还小 2.25 倍。</b>
        <b>MLA 并不是最省的那个。</b><br>
        ⭐ 所以这一支的目标从来不是「谁存得最少」——&nbsp;
        MQA 早在 2019 年就把它压到头了，代价是<b>质量掉得厉害</b>。
        <b>真正要比的是「同样一份字节，换回多少能力」。</b>
        <em>这正是第五节要讲的那条线。</em></span></p>
      <p style="margin-bottom:0"><b>⚠️ 还有一个口径要说清：</b>
        <span class="qs">488 GiB 是「<b>假如 V3 用 MHA</b>」的<b>反事实</b>数字，
        不是 V3 的实测值 ——&nbsp;V3 从第一天就是 MLA。
        而且这里沿用了 V3 论文比较表的口径（K、V 都按 head_dim=128 算）；
        <b>V3 真实的 K 每头是 128+64＝192 维，严格算这个基线还会更大一点。</b></span></p>
    </div>
  </div>

  <div class="guess" style="margin-top:22px">
    <h3>第二题 · 那条「算得过来还是搬得过来」的线</h3>
    <p class="q">上一讲那把尺子：<b>算力 ÷ 带宽</b> ——&nbsp;
      每从内存搬一个字节，这台机器配套能算多少次。<br>
      <span class="qs">在 <b>TPU v7</b> 上，<b>两层各是多少？两行分开选。</b></span></p>

    <div class="oplab">(a) 对 <b>HBM</b>（片外）</div>
    <div class="opts">
      <button data-g="0">78</button>
      <button data-g="1">156</button>
      <button data-g="2" data-right>约 313</button>
      <button data-g="3">约 626</button>
    </div>

    <div class="oplab">(b) 对 <b>VMEM</b>（片上）</div>
    <div class="opts">
      <button data-g="0">跟 HBM 一样，约 313</button>
      <button data-g="1">比 HBM <b>高</b>，约 3,000</button>
      <button data-g="2" data-right>比 HBM <b>低</b>一个量级 ——&nbsp;几十这一档</button>
      <button data-g="3">片上没有「屋脊点」这回事</button>
    </div>

    <div class="rev">
      <p><b>(a) 约 313。(b) 比 HBM 低一个量级，落在几十这一档。</b><br>
        <span class="qs">(a) ＝ 2,307 TFLOP/s ÷ 7.37 TB/s。
        <b>这个数不是本讲新造的，它就是<a href="topic-02.html">专题二</a>整整一节在立的那条屋脊线。</b></span></p>
      <p><b>⭐ (b) 最容易选反，而选反的人通常是把「快」和「门槛高」搞混了：</b>
        <span class="qs">片上更快，所以<b>分母变大</b> ——&nbsp;
        同一个分子除以更大的分母，<b>商只会更小</b>。
        <b>越靠近计算，这条线越低。</b>
        <em>门槛低意味着：同一个算子挪到片上以后，更容易变成算力受限。</em></span></p>
      <p style="margin-bottom:0"><b>⛔ 而 (b) 为什么只给量级、不给数 ——&nbsp;这才是这道题最想教的一件事：</b>
        <span class="qs"><b>VMEM 的带宽官方没有公开。</b>
        而屋脊点乘以算力就等于带宽 ——&nbsp;<b>给出一个精确的屋脊点，
        等于把那个没公开的数反推出来。</b>所以我们到「几十这一档」为止，<b>不往下猜</b>。<br>
        ⭐ 这正是<a href="topic-02.html">专题二</a>第 6 节那四句问法里的一句：
        <b>先问这个数的出处和口径，再用它。</b>
        <em>而那四句不是拿来审别人材料的 ——&nbsp;是先拿来审自己的。</em></span></p>
    </div>
  </div>

  </div>
</details>
<script>
/* 与专题二同一份实现；三条坑（别用 id、等 DOMContentLoaded、按 .opts 分组）
   的来由写在上面那段注释里。 */
(function(){
  function bind(){
  document.querySelectorAll('.guess').forEach(function(box){
    var rev=box.querySelector('.rev'); if(!rev) return;
    var groups=box.querySelectorAll('.opts');
    var rst=document.createElement('div');
    rst.className='rst';
    rst.innerHTML='<button type="button">\u21ba 重来</button>';
    rst.firstChild.addEventListener('click', function(){
      box.querySelectorAll('button').forEach(function(x){x.classList.remove('picked','right');});
      rev.classList.remove('on');
      box.scrollIntoView({behavior:'smooth', block:'nearest'});
    });
    rev.insertBefore(rst, rev.firstChild);
    groups.forEach(function(g){
      g.querySelectorAll('button').forEach(function(b){
        b.addEventListener('click', function(){
          g.querySelectorAll('button').forEach(function(x){x.classList.remove('picked','right');});
          b.classList.add('picked');
          var r=g.querySelector('[data-right]'); if(r) r.classList.add('right');
          var done=true;
          groups.forEach(function(gg){ if(!gg.querySelector('.picked')) done=false; });
          if(done){ rev.classList.add('on'); rev.scrollIntoView({behavior:'smooth', block:'nearest'}); }
        });
      });
    });
  });
  }
  if(document.readyState==='loading') addEventListener('DOMContentLoaded', bind);
  else bind();
})();
</script>

<p>下面这张是<b>整个专题的骨架</b>。六个阶段，每一段只问三件事：
  <b>图啥、带来了什么、欠下了什么</b>。</p>
__FIG_ARC__

<div class="note info"><p>⭐ <b>这一讲从头到尾只有一个账本：KV cache。</b>
  在张量形状里找 <code>S</code>（KV 长度）——&nbsp;<b>全图只有 K 和 V 两处带它</b>，
  那就是唯一需要跨 token 留下来的东西。<br>
  三个旋钮是同一个账本的三个面：<b>① 每份多大　② 每步读多少　③ 干脆别让它变长</b>。
  <b>看到任何一个新名词，先问它在拧哪一面。</b></p></div>

<p>骨架之后，先看一眼<b>全景与全部证据</b>：这些名词是什么时候、按什么顺序冒出来的，
  以及今天各家<b>实际上是怎么配的</b>、<b>每一家的 KV cache 到底多大</b>。</p>
__FIG_CHRONICLE__
__TABLE_MODELS__
<!-- ⛔ 图上所有型号与配比都是当天现搜的公开信息 —— **这张图会过时**，
     而且过时得比正文快得多。约定：**每次开课前只重跑这一张图的调研，正文不动。** -->
<p class="sub">⭐ <b>这一整块最该带走的一句</b>：<b>MiniMax 一家、三代模型，
  把三个旋钮各拧了一遍</b>（线性 → 退回全注意力 → 稀疏），而且每次转向都公开写了理由
  ——&nbsp;<em>「三个旋钮」这个框架不是我们归纳出来的，是有人真的一个一个试过去了。</em></p>
<hr>
</div></section>
<section id="s零"><div class="wrap"><div class="stn"><span class="badge">第 零 节</span><h2>起点：RNN ——&nbsp;被注意力补的那个东西</h2></div>

<div class="note danger"><p>⭐⭐ 先给一个反直觉的事实：你现在用的每一个大模型，在往外吐每一个字的时候，
  都退回成了 1990 年那条链的形状。<br>
  Transformer 赢在<b>训练能并行</b>。可生成的时候，它一个 token 一个 token 地走，
  每走一步都要把全部权重从显存里搬一遍 ——&nbsp;<b>这跟 RNN 一模一样</b>。<br>
  ⛔ <b>它没有治好 RNN 的病，它只是把病从训练挪到了推理</b>；而且挪过去之后更重。</p></div>

<p>所以这一节<b>不是背景介绍，是本专题的舞台说明</b>。<b>四个问题，四张图。</b></p>

<h3>0.1 它是什么、怎么算</h3>
<p>序列有先后，所以得有个东西把历史带下去。RNN 的答案是带一个固定大小的状态向量 <code>h</code>
  ——&nbsp;<b>全部设计就这一句</b>。</p>
__FIG_RNN_UNROLL__
<h3>0.2 为什么在加速器上快不起来</h3>
<p>⛔ 先别去比总计算量。<code>O(n·d²)</code> 和 <code>O(n²·d)</code> 谁大，取决于 <code>n</code> 和 <code>d</code> 谁大；
  Vaswani 原文说的是 <code>n &lt; d</code> 时自注意力更快，而那正是当年的常态。
  <b>固定不变的是另一列 ——&nbsp;串行步数。</b></p>
<table>
<thead><tr><th>层的类型</th><th>每层计算量</th><th><b>串行步数</b></th><th>两个位置之间的最长路径</th></tr></thead><tbody>
<tr><td>自注意力</td><td>O(n² · d)</td><td><b>O(1)</b></td><td><b>O(1)</b></td></tr>
<tr><td>循环（RNN）</td><td>O(n · d²)</td><td><b>O(n)</b></td><td><b>O(n)</b></td></tr>
<tr><td>卷积</td><td>O(k · n · d²)</td><td>O(1)</td><td>O(log_k n)</td></tr>
</tbody></table>
<p class="sub">⭐ 这张表是 <b>Transformer 作者自己算的</b>（arXiv 1706.03762 表 1）。
  <b>中间那一列就是全部答案。</b></p>
__FIG_RNN_HW__
<div class="note danger"><p>⛔ 两头堵死：batch 是它唯一的算术强度来源，
  而 Vaswani 引言那句原话说的正是另一头 ——&nbsp;<em>「memory constraints limit batching
  across examples」</em>：<b>序列一长，显存就不让你把 batch 开大。</b></p></div>

<h3>0.3 解码时，Transformer 又变回了这个形状</h3>
<p><b>这是本节的落点，也是整个专题的舞台。</b></p>
__FIG_RNN_DECODE__
<div class="note info"><p>⭐⭐ Ⓐ 和 Ⓒ 都是「一步一个，每步搬一遍权重」。
  唯一的区别是每步还得额外搬什么：<br>
  RNN 搬的是一个固定大小的状态；Transformer 搬的是一路线性变长的 KV cache
  ——&nbsp;<b>128K 时它能比权重本身还大</b>（下一节算给你看）。<br>
  ⭐ 后面三个旋钮拧的全是同一件事：让这一行每步要搬的东西变小。</p></div>

<h3>0.4 三个痛点，各自通向哪</h3>
__FIG_RNN_PAIN__
<div class="note warn"><p>⚠️ 一个常见的张冠李戴：「固定长度向量是瓶颈」不是 Sutskever 说的。
  他那篇只是描述做法（映射到「a vector of a fixed dimensionality」）；
  「这是个瓶颈」是 Bahdanau 那篇的原话（arXiv 1409.0473：
  <em>「we conjecture that the use of a fixed-length vector is a bottleneck」</em>）。
  <b>别把后人的批评安到原作者头上。</b></p></div>

<details class="aside"><summary>📌 这一节的出处清单（全部一手核过）</summary>
<ul>
<li><b>Elman 1990</b>《Finding Structure in Time》——&nbsp;context units「copied … on a
  one-for-one basis, with fixed weight of 1.0」。</li>
<li><b>Bengio, Simard, Frasconi 1994</b>——&nbsp;梯度消失；<b>Hochreiter &amp; Schmidhuber 1997</b>——&nbsp;LSTM。</li>
<li><b>Bahdanau et al. 2014</b>（arXiv 1409.0473）——&nbsp;「fixed-length vector is a bottleneck」。</li>
<li><b>Vaswani et al. 2017</b>（arXiv 1706.03762）——&nbsp;引言「This <b>inherently sequential</b>
  nature precludes parallelization within training examples…」＋ 表 1 三列。</li>
<li><b>NVIDIA《Recurrent Layers User's Guide》</b>——&nbsp;「a GEMM with <b>one dimension of one</b>」；
  「can combine these GEMMs over the minibatch size, <b>but not over different sequence steps</b>」。</li>
<li><b>Martin &amp; Cundy 2018</b>（arXiv 1709.04057）——&nbsp;非线性依赖挡住并行，
  <b>只有线性依赖能用 parallel scan 扫</b>，实测最高 9× 加速。</li>
<li>拐点 313 FLOP/byte ＝ v7 官方每芯片 FP8 4614 TFLOP/s（BF16 取一半 2307）÷ 官方 HBM 7.37 TB/s。
  <br>⭐ <b>这个 313 不是本讲新造的数 ——&#160;它就是<a href="topic-02.html">专题二</a>
  整整一节在立的那条屋脊线</b>：<em>每从显存搬一个字节，这台机器配套能算 313 次。</em>
  <b>那边花了十几分钟把它立起来，这里一句话就能用。</b>
  <br><em>（2026-09-12 补：原来这里只有算式、没点名出处，
  于是这个数在两讲之间是断的 ——&#160;台下不会自己把它接上。）</em></li>
</ul></details>

<h3>0.5 于是下一节</h3>
<p>RNN 疼在三处：算不快、记不住、装不下。
  注意力最早只解决了第三条，<b>而且是作为 RNN 的一个附件出现的</b>。</p>
<p>2017 年有人问：<b>既然这个附件这么好使，能不能把 RNN 整个扔掉，只留附件？</b></p>
<hr>
</div></section>

<section id="s一"><div class="wrap"><div class="stn"><span class="badge">第 一 节</span><h2>MHA ——&nbsp;把循环拿掉，代价是什么</h2></div>

<p>2017 年有人问：<b>既然那个补丁这么好使，能不能把 RNN 整个扔掉，只留补丁？</b></p>
<div class="note ok"><p>⭐ 这一节要交代三件事，各配一张图：
  换掉了什么（一根横箭头 → 一张 n × n 的表）、一层里到底在算什么
  （检索三件套，外加那个 √d 是怎么来的）、多头在多什么（不是堆算力）。<br>
  ⭐⭐ 落点只有一句：<b>KV cache 就是在这一节出生的</b> ——&nbsp;§零 图三 Ⓒ 那一行，
  病根在这里。</p></div>

<h3>1.1 换掉了什么</h3>
__FIG_MHA_SWAP__
<div class="note info"><p>⭐ §零 那张表（Vaswani 表 1）的「怎么做到的」就在这张图里：
  串行步数 O(n) → O(1)、最长路径 O(n) → O(1)，代价是要算的格子从 <code>n</code> 变成
  <code>n²</code>。<br>
  ⛔ 而更要紧的是另一件事：状态没了。RNN 那个固定大小的 <code>h</code>，
  被换成了「<b>把所有历史原封不动留着</b>」。</p></div>

<h3>1.2 一层里到底在算什么</h3>
__FIG_MHA_QKV__
<div class="note warn"><p>⚠️ 「query / key / value」不是我们编的比喻，是论文自己的措辞：
  <em>「mapping a query and a set of key-value pairs to an output … the output is a
  weighted sum of the values」</em>。<br>
  ⭐ 而那个 <code>√d_k</code> 也不是玄学 ——&nbsp;推导链在论文脚注里，两行就讲得完
  （见图中蓝框）：<b>打分的方差随维度线性长大，而 softmax 只认绝对数值。</b></p></div>

<h3>1.3 多头在多什么</h3>
__FIG_MHA_HEADS__
<div class="note danger"><p>⛔ 多头的代价，正好是本专题的题眼：每个头都要自己的 K 和 V。<br>
  ⭐⭐ 所以后面第一个旋钮第一刀砍的就是这里 ——&nbsp;<b>MQA 让所有头共用 1 组 KV、
  GQA 折中成几组</b>，砍的正是多头在这一步乘上去的那个 8。</p></div>

<h3>1.4 把形状标出来：本专题的主线图</h3>
<p>上面三张图讲的是<b>想法</b>。把一层 Transformer 的<b>每一步张量形状</b>都标出来之后，
  <b>这一节的账会变得可以用眼睛读</b> ——&nbsp;在形状里找那个会越变越长的维度就行。</p>
__FIG_TX_BASE__

<h3>1.5 于是下一节</h3>
<div class="note info"><p>⭐⭐ <b>这个形状的代价，2019 年就被点名了。</b>
  Shazeer 那篇（就是 MQA 的出处）摘要里写着：训练很快，因为序列方向可以并行；
  <em>但「incremental inference … is often slow, due to the <b>memory-bandwidth cost of
  repeatedly loading the large "keys" and "values" tensors</b>」</em>。<br>
  ⭐ 这句话把 §零 图三 Ⓒ 那一行，从我们的推论变成了原文 ——&nbsp;
  2017 年造出这个形状，<b>2019 年就有人把它命名成问题了</b>。</p></div>
<p>下一节先把这笔账<b>算成具体的字节数</b>，再看它为什么必须省。</p>
<hr>
</div></section>
<section id="s二"><div class="wrap"><div class="stn"><span class="badge">第 二 节</span><h2>一切从长上下文说起 —— 两条独立的动机</h2></div>
<p><b>这两条要分开讲。</b> 它们指向同一批技术，但出发点完全不同， 混在一起讲就变成了名词罗列。</p>
<h3>2.1 线索 A · 硬件账算不过来</h3>
<p>上下文一长，<b>KV cache 就爆</b>。</p>
<p>它的增长是双重的：<b>随序列长度线性增长，又随层数线性增长</b>。 而且它跟参数不一样 —— 参数是固定的一次性开销，KV cache 是随用户输入长度涨的。</p>
<p>用一个大家已经熟悉的形状来砸体感：<b>就用专题一那个 V3</b> （61 层、hidden 7168、128 个头、每头 128 维）。 假设它用的是最朴素的 MHA，128K 上下文，bf16 存：</p>
<pre><code>每 token 每层    2 × n_h × d_h  = 2 × 128 × 128 = 32,768 个数
每 token 全模型  × 61 层                        = 1,998,848 个数
                × 2 字节                        = 3,997,696 B = 3.81 MiB
128K token       × 131,072                      = 523,986,010,112 B
                                                 = 488 GiB</code></pre>
<p><b>一个用户、一段 128K 的输入，KV cache 488 GiB。</b></p>
<p>摆三个对照，让这个数字站住：</p>
<table>
<thead><tr><th>对照物</th><th>数值</th><th>于是</th></tr></thead><tbody>
<tr><td>一块 TPU v7 device 的 HBM</td><td>94.74 GB ≈ <b>88.23 GiB</b></td><td><b>一个用户的 KV 就要 5.5 个 device 装</b></td></tr>
<tr><td>V3 的全部权重（671 B × 2 B）</td><td>1.342 TB ≈ <b>1250 GiB</b></td><td>单用户占权重的 <b>39%</b>；<b>三个并发用户就超过权重本身</b></td></tr>
<tr><td>同样形状换成 GQA-8</td><td><b>30.50 GiB</b></td><td>16 倍，但还是装不进一块卡</td></tr>
<tr><td>同样形状换成 MLA（V3 真实方案）</td><td><b>8.58 GiB</b></td><td><b>56.9 倍</b></td></tr>
</tbody></table>
<div class="note warn"><p>⚠️ <b>这个 488 GiB 是「假如 V3 用 MHA」的反事实数字，不是 V3 的实测值。</b> 而且这里沿用了 V3 论文比较表的口径（K、V 都按 d_h=128 算）； V3 真实的 K 每头是 128+64=192 维，所以严格算 MHA 基线还会更大一点。 讲的时候要说清楚这是自己按公式推的、口径是什么 —— 这门课自己的规矩：推出来的数字必须带推导链。</p></div>
<p>推理场景更狠：并发用户越多，KV cache 线性叠加。 权重是所有用户共享的一份，KV cache 是每人一份。 所以 KV cache 不是显存里的一项开销，<b>它直接决定了你能同时服务多少人</b> —— 这一条到<a href="专题06-推理.md">专题六</a>会变成 batch size 的硬上限。</p>
<div class="note info"><p>这一条是<b>工程账</b>：不解决它，长上下文根本上不了线。</p></div>
<h3>2.2 线索 B · 信息本身不需要那么多</h3>
<p>这条更有意思，也更少被讲。</p>
<p><b>一个 128K 的序列，真的需要 128K 份独立的 KV 吗？</b></p>
<p>几个可以摆出来的观察：</p>
<ul><li><b>注意力矩阵实测是极其稀疏的</b> —— 绝大部分权重集中在很少的位置上， 剩下的近乎为零。那么把那些近零的算出来，算的是什么？</li><li><b>Attention sink</b> —— 模型会把大量注意力"停放"在序列最开头的几个 token 上， 跟内容无关。这说明注意力权重里有一部分根本不是在做检索</li><li><b>远近有别</b> —— 邻近几十个 token 的关注是密集、细粒度的； 几万 token 之外的关注是稀疏、粗粒度的。 <b>凭什么用同一套精度去处理这两种？</b></li><li>于是：<b>压缩</b>（远处的多个 token 合并成一个）、 <b>稀疏</b>（只挑相关的看）、<b>分层</b>（近处精细远处粗糙）都变得合理</li></ul>
<div class="note info"><p>这一条是<b>信息账</b>：即使显存无限，把全部算力花在一个高度冗余的矩阵上也是浪费。</p></div>
<h3>2.3 两条线的关系</h3>
<p><b>A 说「必须省」，B 说「可以省而不太亏」。</b></p>
<p>这两句话缺一不可：</p>
<ul><li><b>只有 A</b>，你得到的是一堆有损压缩的权宜之计，效果掉了只能认</li><li><b>只有 B</b>，你没有动力去付 kernel 那么难写的代价</li><li><b>两条合起来</b>，才解释了为什么这个方向在过去三年里投入了这么多人 —— 它同时是一件<b>不得不做</b>和<b>做了不太亏</b>的事</li></ul>
<p><b>所有变体都活在这两条线的交汇处。</b> 一个变体好不好，就看它在 「省了多少（A）」和「亏了多少（B）」之间落在哪。</p>
<hr>
</div></section>
<section id="s三"><div class="wrap"><div class="stn"><span class="badge">第 三 节</span><h2>FlashAttention ——&nbsp;已经是标配，所以这一讲不展开它</h2></div>

<!-- ⛔⛔ 2026-09-08 第二次降级，这次连标题都换了。现场原话：
       「FlashAttention 现在已经是标配了，它在各种 attention 模式下都是一样的，
         所以在这一刻里边应该没有什么讨论的意义。」
     ⭐ 判据：**一个在所有分支上取值都相同的变量，对这一讲没有解释力。**
       它既不区分 MLA 和 GQA，也不区分稀疏和线性 —— 谁都要用，谁用了都一样。
       ⛔ 留一节讲它，等于在一根不变的轴上花掉读者十分钟。
     📌 原标题是「先解决一个误会：那个矩阵从来没被存下来过」——&nbsp;
       同一轮里那个「误会」本身也不讲了（见下面这一段的注释）。 -->

<div class="note ok"><p>⭐ <b>它跟这一讲要讨论的东西不是一类。</b>
  三个旋钮改的是<b>算什么</b>（模型变了，通常要重训）；
  FlashAttention <b>一个字都不改数学</b>，它改的是<b>怎么算</b>
  ——&nbsp;<b>跟谁都能叠，而且必须叠</b>。<br>
  📌 <b>正因为它跟谁都能叠、叠上去效果都一样，它在这一讲里没有区分度</b>：
  它既不区分 MLA 和 GQA，也不区分稀疏和线性。<b>所以本节只留这一句。</b></p></div>

<p class="sub">⭐ 它做的事一句话：<b>让 softmax 前后那一步不落到显存</b>，
  只在片上一块一块地过。<b>这也是为什么本讲从头到尾不讨论「那个中间矩阵有多大」</b>
  ——&nbsp;<b>它根本不占显存。这一讲的账本只有一个：KV cache。</b></p>

<details class="aside"><summary>🔬 <b>课外：FlashAttention 深潜</b>
——&nbsp;它怎么做到不落地、块该开多大、以及
<b>为什么融合完还是只跑到 35%</b><em>（硬件细节，写 kernel 的人才需要）</em></summary>

<h3>3.2 它删掉的是哪一项</h3>
<p>朴素写法是<b>三步</b>：① 算 <code>S = QKᵀ</code> ② 对 S 做 softmax 得 P ③ 算 <code>O = PV</code>。</p>
<p>三步之间，那个 L×L 的 S 每次都要<b>落一趟 HBM</b>： 写 S、读 S、写 P、读 P —— <b>三个步骤，四趟。</b></p>
<p>而这个 S 有多大：128K 上下文、单个头就是 <code>131072² × 2 B = 32 GiB</code>，一层 128 个头。</p>
<p>FlashAttention 把三步融成一个 kernel：S 分块在片上算完， softmax 用在线归约边走边更新最大值和求和项，<b>S 整项消失</b>。</p>
<p>⭐ 所以三笔账的形状是所有算子融合共有的： <b>FLOPs 一分不省，省的全在「中间产物不落地」这一行。</b></p>
<h3>3.2b ⭐ 那张经典图：内循环、外循环 —— 以及 FA2 为什么把它掉了个个儿</h3>
<p>FlashAttention 论文那张图，画的是<b>两层循环</b>： HBM 里躺着 Q、K、V、O，SRAM 是旁边一个小方块； 外层循环搬一块进 SRAM，内层循环扫过另一边的所有块。</p>
<p><b>⚠️ 但那张图画的是第一版的顺序，而第二版把它换了过来 ——&nbsp; 这个「换」本身，比图更值得讲。</b></p>
<p><b>第一版：外层是 K／V，内层是 Q。</b></p>
<pre><code>for j in K/V 的每一块:              # 外层
    把 K_j, V_j 搬进 SRAM
    for i in Q 的每一块:            # 内层
        把 Q_i、O_i、m_i、ℓ_i 从 HBM 读进来
        算这一格，更新 O_i、m_i、ℓ_i
        再把 O_i、m_i、ℓ_i 写回 HBM</code></pre>
<p>⛔ 看内层那三行：<code>O_i</code> 和那两个统计量每一轮外循环都要读进来、写回去一次。 为什么躲不掉？——&nbsp;因为<b>下一块 K／V 还会碰到同一个 Q 块</b>， 它的输出没算完，只能先存回去。</p>
<p><b>第二版：把两层对调。</b></p>
<pre><code>for i in Q 的每一块:                # 外层 ← 换成了 Q
    把 Q_i 搬进 SRAM，O_i、m_i、ℓ_i 就地清零
    for j in K/V 的每一块:          # 内层
        搬 K_j, V_j 进来，算，就地累加进 O_i
    整个内层跑完，才把 O_i 写出去一次</code></pre>
<p>⭐ <b>两个后果，都很硬：</b></p>
<ol><li><b><code>O_i</code> 和那两个统计量在整个内层循环里一直待在片上，一次都不落 HBM。</b></li><li><b>不同的 Q 块之间彻底独立</b> ——&nbsp; 可以直接铺到几百个执行单元上，<b>互相不用通信</b>。</li></ol>
<div class="note ok"><p>⭐ 一句话记住它： 外循环放谁，谁的中间状态就不用来回搬。 而 attention 里「需要被累加到最后」的是 <code>O</code>，所以<b>外循环必须放 Q</b>。</p></div>
<h3>3.2c 同一个原则，在 warp 这一层又用了一遍</h3>
<p>这件事在<b>块内部</b>还发生了第二次 ——&nbsp;而且这一次是官方原话：</p>
<ul><li><b>第一版</b>：把 <b>K 和 V 切给 4 个 warp</b>，Q 大家共用（叫 sliced-K）。 ⛔ 于是每个 warp 都得<b>把中间结果写进共享内存、同步、再加起来</b> ——&nbsp; 这些读写拖慢了前向。</li><li><b>第二版</b>：反过来，把 <b>Q 切给 4 个 warp</b>，K 和 V 大家共用。 ⭐ 每个 warp 算出自己那一片 <code>QKᵀ</code>，直接乘共享的那片 V 就得到自己那片输出 ——&nbsp;<b>warp 之间完全不需要通信。</b></li></ul>
<div class="note warn"><p>⭐⭐ <b>两个尺度，同一条判据</b>： 切「要被累加的那一维」就得合；切「各自独立出结果的那一维」就不用合。 attention 里前者是 K／V（它们在求和号里面），后者是 Q（每行输出各管各的）。</p>
<p>⚠️ 这条判据的用处远不止 attention ——&nbsp; <b>任何融合 kernel 在分工时，先问一句「我切的这一维在不在求和号里」。</b></p></div>
<h3>3.3 ⚠️ 在线 softmax 不是免费的 —— 它是后面 35% 那个数的根</h3>
<p>softmax 要减最大值才数值稳定，而最大值要看完整行才知道。 在线归约的做法是：<b>每来一块就更新一次 running max 和 running sum， 并把已经累好的输出按比例重标定一次。</b></p>
<div class="note danger"><span class="t">⛔ 这里有个几乎人人都会踩的误会：压力<b>不在</b>那两个 running 值上</span><p>「在线 softmax 只要存一个最大值和一个求和项」——&nbsp;<b>这句话是对的， 而且它们确实很小</b>：每行各一个标量。一个 <code>bq = 512</code> 的 Q 块， 两个加起来也就 <b>4 KB</b> 量级。</p>
<p>真正占地方的是 <code>S = Q@Kᵀ</code> 那一整块。 它的形状是 <code>[bq, bkv]</code> ——&nbsp;<code>512 × 512</code> 的 fp32 就是 1 MB， <code>2048 × 2048</code> 是 16 MB。<b>比那两个 running 值大两三个数量级。</b></p>
<p>而且它<b>不是存一次就完了</b>：每来一个 KV 块就重新生成一整块。</p></div>
<p><b>所以卡住的是 <code>S</code> 的生命周期，不是那两个数。</b></p>
<p>一个 KV 块内，<code>S</code> 要连着走完四步才能扔： <b>① MXU 产出 → ② 沿着行求最大值 → ③ 减掉它再取指数 → ④ 喂给第二个矩阵乘。</b></p>
<p>这四步里它一直是活的。 ②③ 是向量单元的活，而 ① ④ 是矩阵单元的活 ——&nbsp; <b>矩阵单元想开始下一块，可它的输出还被 ②③ 占着。</b></p>
<p>⭐ 再加上流水线：要让下一块的矩阵乘和这一块的向量运算重叠， <b>就得同时留住不止一块 <code>S</code>。</b> 于是「一块」变成「好几块」。</p>
<div class="note info"><p>📌 ② 那一步在 TPU 上还额外贵一点：行方向的最大值是一次<b>跨 lane 归约</b>， 而 lane 正是硬件那 128 的方向。<b>归约越慢，<code>S</code> 活得越久。</b></p></div>
<p><b>这条链在 1.6 会变成一个具体的数字。</b></p>
<h3>3.4 块开多大 —— GPU 那边是一堵墙</h3>
<p>块大小不是调着玩的旋钮，它由片上暂存的容量直接顶死。</p>
<p><b>GPU 侧</b>：一个线程块最多拿 <b>227 KiB</b> 共享内存。 装三块 <code>128×128</code> 的 bf16 tile（Q／K／V）就是 <code>3 × 128 × 128 × 2 B = 96 KiB</code>， 再留双缓冲，<b>基本到顶</b>。</p>
<p><b>而块开不大，代价是 K／V 被重复读</b> —— 重读次数 ≈ 序列长度 ÷ Q 块大小。 所以那 227 KiB 不只是「装不下」，它通过块大小间接决定了 HBM 流量。</p>
<div class="note warn"><p>⚠️ 顺带拆掉一个常见误解：<b>标准 FlashAttention 前向<u>不需要跨块归并</u>。</b> 每个执行单元拿走一个 Q 块，自己走完整条 KV 循环，在线 softmax 在块内就闭合了。 需要合的只有两种：<b>KV 也被切开时</b>（长上下文解码那类做法， 要跨切片重新对齐最大值和求和项），以及<b>反向</b>（对 Q 的梯度要跨块累加）。 <b>所以 GPU 的代价在「分」，不在「合」。</b></p></div>
<h3>3.5 ⭐ TPU 那边是三堵墙 —— 而且「块越大越好」是错的</h3>
<p>TPU 侧对应的是 <b>Splash Attention</b>。它的灶台看着大得多 （Ironwood 每个 TensorCore <b>64 MB VMEM</b>，一颗 chip 两个核 = 128 MB）， <b>但块反而不能随便开大。</b></p>
<p>JAX 里 Splash Attention 的<b>默认块是 128 × 128</b>， 而且源码里挂着一句 <code>TODO</code>：「以后按启发式选更好的参数」。</p>
<p><b>我们自己在 Hunyuan3-295B 上扫过这个参数</b>（<code>seq = 4096</code>，v7 64 芯片， 完整数据见 <a href="../tpu/Hunyuan3-295B-Pretraining/TUNING-v7.md"><code>tpu/Hunyuan3-295B-Pretraining/TUNING-v7.md</code></a>）：</p>
<table>
<thead><tr><th>块大小</th><th>KV 方向切出几块</th><th>结果</th></tr></thead><tbody>
<tr><td><b>2048（甜点）</b></td><td><b>2 块</b></td><td><b>228.4 TFLOP/s/device</b></td></tr>
<tr><td>4096</td><td>1 块</td><td><b>VMEM 直接爆</b>（爆在反向）</td></tr>
<tr><td>4096（compute 压回 2048）</td><td>1 块</td><td><b>−11.5%</b></td></tr>
<tr><td>512（照抄官方长上下文配置）</td><td>8 块</td><td><b>−1.0%</b></td></tr>
</tbody></table>
<p><b>三堵墙，方向各不相同：</b></p>
<ol><li><b>往上是容量墙</b> —— 再开大一档就装不下，<b>而且先爆的是反向那一侧</b></li><li><b>往上还有第二堵，常常比容量更早撞到：并行度</b> —— 块一大，KV 方向只切得出一块，<b>那一维的流水直接塌掉</b>。 ⭐ <b>这一堵最反直觉：装得下，却更慢。</b></li><li><b>往下是碎块开销</b> —— 每块的固定开销（mask 检查、running max/sum 更新、 pipeline stage 切换）摊不动</li></ol>
<div class="note warn"><span class="t">⚠️ 一条可迁移的教训：块大小要看 <code>block / seq</code> 的比例，不是绝对值</span><p>那个 512 不是我们瞎试的，是官方 tpu7x benchmark 里的值 —— 但那份配置是给 <code>max_target_length = 131072</code> 调的： 512 相对 131072 是 1/256；到我们 <code>seq = 4096</code>，512 就成了 1/8。 <b>跨序列长度照抄配置会反向优化。</b></p>
<p>我们记下的经验规则是「块 ≈ seq/2」，可证伪版本是 「换 <code>seq = 8192</code> 时最优块应当变成 4096」—— ⚠️ <b>这一条尚未验证。</b></p></div>
<p>⭐ 所以两边的墙不是「硬件 vs 调参」，是一面 vs 三面： GPU 那边容量一堵墙顶死，方向反倒清楚 —— 能开多大就开多大； <b>TPU 那边最优往往不在最大处</b>，得在三面之间找那个点。</p>
<h3>3.6 ⭐ 融合之后，它还是只跑到 35% —— 三层原因，都不是配置问题</h3>
<p>这是这一节最该带走的一段：FlashAttention 不是终点。 我们在 v7 上量到 splash attention <b>占 23% 的时间、效率只有 35.5%，全场最低</b>。</p>
<p>① 记账口径：报出来的那个百分比，分子是虚高的。 XLA 给 splash 记的 FLOP 是不折 causal 的全量 <code>4·b·s²·h·d</code>。 所以 32.8–39.0% 这个区间是拿虚高的分子算出来的，<b>真实执行效率比它更低</b>。</p>
<div class="note warn"><p>⚠️ causal 跳过上三角是<b>节省</b>，不是又一道要乘上去的折扣 —— 它只造成记账错位。这两件事最容易混。</p></div>
<p><b>② 形状锁死 50%：MXU 是 256×256，而 <code>head_dim = 128</code> 只吃得下一半。</b></p>
<table>
<thead><tr><th>matmul</th><th>形状</th><th>浪费在哪</th></tr></thead><tbody>
<tr><td><code>QKᵀ</code></td><td><code>[q_len, 128] @ [128, kv_len]</code></td><td><b>收缩维</b>只有 128 → 废一半</td></tr>
<tr><td><code>PV</code></td><td><code>[q_len, kv_len] @ [kv_len, 128]</code></td><td><b>输出维</b>只有 128 → 废一半</td></tr>
</tbody></table>
<p>两个矩阵乘各撞一次，所以整个算子的 MXU 利用率封顶 50%。
<span class="sub">（⭐ 为什么「输出维只有 128」也会浪费一半，
以及为什么这种浪费能救而收缩维那种救不了 ——&nbsp;
机制在 <a href="topic-02.html">专题二 3.4</a>。）</span> Google 侧的结论是明确的：<code>head_dim = 128</code> 时 MXU 利用率无法超过 50%， <b>没有办法绕过</b>。</p>
<p><b>③ 为什么连 50% 都到不了 —— 回到 1.3 那条链。</b> 持有 <code>Q@K</code> 输出（也就是<b>整块 <code>S</code></b>，不是那两个 running 值）的寄存器， 在最大值和减法完成前不能释放，于是不断堆积 → 寄存器压力 → <b>spill 到 VMEM</b> → MXU 停在等数据载回。 <b>一句话：VPU 跟不上 MXU。</b></p>
<div class="note warn"><p>⭐ 这也解释了 1.5 那三堵墙里最反直觉的一堵为什么存在： <b>块开大 → <code>S</code> 那一块跟着变大 → 生命周期更长、更容易 spill。</b> 「装得下」和「跑得快」在这里是两回事。</p>
<p>⭐ <b>三层叠起来的结论很硬</b>： 形状锁死一半、寄存器压到 35%、记账口径还让它看着比实际好看。 <b>这三层没有一层是配置能救的</b> —— 要么改 head_dim，要么改 kernel 的数据流。 （我们试过的那条出路是把矩阵乘全转置，见 TUNING-v7 的附录。）</p>
<h3>⚠️ 还有一条方法论：同一份 profile，不同工具页的百分比不可混用</h3>
<p>「HBM 受限占多少」—— 一个工具页说 <b>35.6%</b>，另一个说 <b>19.5%</b>。 破案的钥匙是两者的 self-time 合计正好差 <b>2.00 倍</b>， 而 v7 恰好是 <b>2 device/chip</b>。分母不是同一个东西，分子上的百分比自然对不上。 ⇒ 判瓶颈用 roofline 那一页，归因到算子用 op stats 那一页； <b>引用任何百分比都要写清出自哪个工具页。</b></p></div>
__FIG_TX_FA__
</details>

<h3>3.7 这一节留下的那句话</h3>
<p><b>「怎么算」这条路，到这里基本走到头了。</b></p>
<p>中间产物已经不落地，块大小已经贴着三堵墙，而算子仍然只跑到三成多 —— <b>剩下的空间不在「怎么算」里，只能去改「算什么」。</b></p>
<p>⭐ <b>下一节的三个旋钮，就是「改算什么」的全部可能位置。</b></p>
<hr>
</div></section>
<section id="s四"><div class="wrap"><div class="stn"><span class="badge">第 四 节</span><h2>骨架：三个旋钮，是同一个账本的三个面</h2></div>
<p>这是这个专题的骨架。把所有名词收进一张表：</p>
<table>
<thead><tr><th>旋钮</th><th>在改什么</th><th>代表</th></tr></thead><tbody>
<tr><td><b>① 每个 token 存多少</b></td><td>减少 KV 的<b>份数</b>或<b>维度</b></td><td>MQA → GQA → <b>MLA</b> → Gated MLA</td></tr>
<tr><td><b>② 每个 query 看多少</b></td><td>限制<b>范围</b>或<b>动态挑选</b></td><td><b>SWA</b> · NSA · <b>DSA</b> · <b>CSA / HCA</b></td></tr>
<tr><td><b>③ 换一套数学</b></td><td>用<b>固定大小的状态</b>代替不断变长的 KV</td><td>线性注意力：DeltaNet → <b>GDN</b> → <b>KDA</b></td></tr>
<tr><td><b>①+②+③ 混着来</b></td><td>不同层用不同方案</td><td><b>Hybrid</b>：V4 的 CSA+HCA、K3 的 KDA+Gated MLA</td></tr>
</tbody></table>
<div class="note ok"><p>⭐ 前两个旋钮省的是同一样东西的两个不同侧面： 旋钮 ① 让每份 KV 更小，旋钮 ② 让读的份数更少。<b>它们可以叠加。</b></p>
<p>⭐ <b>旋钮 ③ 是换赛道</b>：它不再有"随长度增长的 KV"这个概念， 代价是把无损的检索换成了有损的状态压缩。</p></div>
<h3>⭐ 为什么恰好是三个</h3>
<p>不是凑出来的。回到注意力那个式子，一个 query 要做的事只有三步：</p>
<pre><code>①  从每个位置取出一份 K、一份 V        ← 存什么、存多大
②  跟哪些位置算，然后加权求和           ← 求和的范围有多大
③  用 softmax(qKᵀ/√d) 做这个加权        ← 用哪种运算</code></pre>
<p>三个旋钮就是这三步各自可以动的地方。 除此之外没有第四个位置可以动 —— 除非你不改算什么、只改怎么算，那就是 FlashAttention —— 那是<b>第一节</b>刚讲完的事。</p>
<h3>一张名词收纳表</h3>
<p>这张表建议做成板书，讲完每个支线回来填一格：</p>
<table>
<thead><tr><th>名词</th><th>出处</th><th>旋钮</th><th>一句话</th></tr></thead><tbody>
<tr><td>MQA</td><td>Shazeer, arXiv 1911.02150</td><td>①</td><td>所有头共用一份 K/V</td></tr>
<tr><td>GQA</td><td>Ainslie 等, arXiv 2305.13245</td><td>①</td><td>分组共用，MQA 与 MHA 之间的连续旋钮</td></tr>
<tr><td><b>MLA</b></td><td>DeepSeek-V2 / V3, arXiv 2412.19437</td><td>①</td><td>KV 压成 512 维隐向量，用时再升回 128 头</td></tr>
<tr><td>Gated MLA</td><td>Kimi K3, arXiv 2607.24653</td><td>①</td><td>MLA 输出端加一个全秩门控</td></tr>
<tr><td><b>SWA</b></td><td>Mistral 7B, arXiv 2310.06825</td><td>②</td><td>只看前面固定窗口（Mistral 是 4096）</td></tr>
<tr><td>Attention sink</td><td>StreamingLLM, arXiv 2309.17453</td><td>—（现象）</td><td>开头几个 token 被当作"停车位"，扔了就崩</td></tr>
<tr><td><b>NSA</b></td><td>arXiv 2502.11089</td><td>②</td><td>压缩 / 选择 / 滑窗三条支路，门控融合，<b>训练时就用</b></td></tr>
<tr><td><b>DSA</b></td><td>DeepSeek-V3.2, arXiv 2512.02556</td><td>②</td><td>Lightning Indexer 给每个 query 挑 top-k</td></tr>
<tr><td><b>CSA / HCA</b></td><td>DeepSeek-V4, arXiv 2606.19348</td><td>①+②</td><td>先把 KV 按块压缩，再稀疏挑选；两档压缩率混排</td></tr>
<tr><td>DeltaNet</td><td>起源 Schlag 等 arXiv 2102.11174；可并行化 arXiv 2406.06484</td><td>③</td><td>状态更新用 delta rule：擦掉旧的再写新的</td></tr>
<tr><td><b>GDN</b>（Gated DeltaNet）</td><td>arXiv 2412.06464</td><td>③</td><td>在 delta rule 上加遗忘门</td></tr>
<tr><td><b>KDA</b></td><td>Kimi Linear, arXiv 2510.26692</td><td>③</td><td>遗忘门从标量升级成 <b>per-channel</b> 向量</td></tr>
<tr><td>FlashAttention</td><td>arXiv 2205.14135</td><td><b>不是旋钮</b></td><td>数学一个字不改，只改访存顺序</td></tr>
</tbody></table>
<hr>
</div></section>
<section id="s五"><div class="wrap"><div class="stn"><span class="badge">第 五 节</span><h2>旋钮①：让每一份更小</h2></div>
__FIG_TX_K1__
<h3>5.1 MHA → MQA → GQA：一个连续旋钮</h3>
<ul><li><b>MHA</b>：每个头各存一份 K/V。128 个头就是 128 份</li><li><b>MQA</b>：所有头共用同一份 K/V。省 128 倍，但<b>质量掉</b></li><li><b>GQA</b>：分成 g 组，组内共用。<code>g = n_h</code> 退化成 MHA，<code>g = 1</code> 退化成 MQA</li></ul>
<p>⭐ 值得强调的是「GQA 是一个连续旋钮」这件事本身 —— 它不是一个新机制，是把 MHA 和 MQA 之间的空白填上，让你可以按需要选一个点。 这门课后面会反复见到这个套路：<b>把一个二选一变成一个可调的连续量。</b></p>
<p>代价说清楚：省的是 KV 的<b>份数</b>，赔的是<b>表达能力</b> —— 本来 128 个头可以各自关注不同的东西，现在被迫共享。</p>
<h3>5.2 MLA：不砍头，改成低秩压缩</h3>
<ul><li>KV 不再按头存，而是压成<b>一个 512 维的隐向量</b>（<code>d_c = 512</code>）， 用的时候再用上投影矩阵升回 128 个头</li><li>位置信息<b>单独走 64 维一路 RoPE</b>（<code>d_h^R = 64</code>）</li><li>所以每 token 每层只存 <b>512 + 64 = 576</b> 个数 —— 对照 MHA 的 32,768 个，<b>56.9 倍</b></li><li>出处：V3 论文 <b>sec. 4.2</b> 超参一节，<code>n_h=128, d_h=128, d_c=512, d_h^R=64, 61 层</code></li></ul>
<p><b>代价：用计算换显存。</b> 多了一对降维/升维的矩阵乘。 这句话在<a href="专题04-反向与优化器.md">专题四</a> §3.2 会被再打一个折扣 —— ⚠️ <b>MLA 的压缩在训练前向里其实不生效</b>（K/V 会被解压出来算）， 它省的是<b>推理时的 KV cache</b>，不是训练时的激活。这是一个非常常见的误解。</p>
<div class="note warn"><p>⚠️ <b>落到硬件上有个反直觉的后果</b>：MLA 在推理时可以把上投影矩阵"吸收"进 query 那一侧，从而改变整个计算的形状。同一个数学式子有多种算法实现， 选哪种取决于是 prefill 还是 decode。 这一条留到<a href="专题06-推理.md">专题六</a>。</p></div>
<h3>5.3 ⚠️ 为什么 RoPE 必须单独走一路</h3>
<p>这是理解 MLA 的关键一步，也是最容易讲糊的一步。<b>值得花两分钟。</b></p>
<p>MLA 想做的事是：<b>把上投影矩阵吸收掉，让推理时只需要读那个 512 维的隐向量。</b> 数学上，<code>qᵀ (W_UK c)</code> 可以重写成 <code>(W_UKᵀ q)ᵀ c</code> —— 上投影跑到 q 那边去了， 于是 K 根本不用真的解压出来。</p>
<p><b>但 RoPE 一插进来这个重写就不成立了。</b> RoPE 是一个跟位置有关的旋转， 它作用在解压之后的 K 上；旋转矩阵夹在中间，<code>W_UK</code> 和 <code>c</code> 就分不开了。</p>
<p>所以 MLA 的解法是把这两件事拆开走两条路： 一路 512 维不带位置、可以被吸收；另一路 64 维专门扛 RoPE、老老实实存着。 <b>576 = 512（可吸收）+ 64（不可吸收）。</b></p>
<div class="note ok"><p>⭐ 这个「因为要保留某个代数变换，所以把功能拆成两路」的动作， 在后面还会以别的面貌出现（V4 的部分 RoPE、K3 的 NoPE）。 <b>值得当成一个套路记住，而不是当成 MLA 的一个实现细节。</b></p></div>
<h3>5.4 Gated MLA</h3>
<ul><li>在 MLA 的输出端加一个<b>门控</b>：<code>gate = σ(W_g x)</code>，逐元素乘在注意力输出上</li><li>Kimi K3 用的是<b>全秩</b>门控矩阵（K2 那代是低秩的）</li><li>出处：K3 技术报告 <b>sec. 2.1.2</b></li></ul>
<p>它的作用不是省显存 —— <b>门控不减少任何 KV</b> —— 而是让模型能学会「这一层这个位置，注意力的输出干脆不要」。 放在旋钮 ① 里是因为它改的是 MLA 这一支的形状，但要讲清楚它<b>省的不是显存</b>。</p>
<hr>
</div></section>
<section id="s六"><div class="wrap"><div class="stn"><span class="badge">第 六 节</span><h2>旋钮②：KV 照存，但每步只读一部分</h2></div>
__FIG_TX_K2__
<h3>6.1 SWA（滑动窗口）</h3>
<ul><li>每个 token 只看自己前面固定窗口内的（Mistral 7B：<b>4096</b>）</li><li>最简单，KV cache 从随长度增长变成<b>常数</b></li><li>代价很硬：<b>长距离信息只能靠层层传递间接到达</b> —— 第 1 层看 4K，第 2 层能间接摸到 8K，要跨 128K 得堆 32 层</li><li>所以一般<b>不单用</b>，跟全注意力混排（见第六节）</li></ul>
<h3>6.2 Attention sink —— 一个现象，不是一个方案</h3>
<p><b>这一段是全课少有的"实验发现改变了工程做法"的例子，值得单独讲。</b></p>
<p>StreamingLLM（arXiv 2309.17453）的观察：如果你朴素地做滑动窗口， 把开头那几个 token 的 KV 也滑掉，<b>模型立刻崩</b>。</p>
<p>而它们发现的解法简单到有点荒谬：<b>留住最开头 4 个 token 的 KV 就够了</b> （论文原话：<em>"with just 4 initial tokens sufficing"</em>）， 再加上滑动窗口，Llama-2 / MPT / Falcon / Pythia 就能稳定处理到 <b>400 万 token</b>， 比"滑窗 + 重算"这个可用基线快 <b>22.2×</b>。</p>
<p>为什么？因为 softmax 强制所有权重加起来等于 1 —— 模型有时候什么都不想看， 但它没有"弃权"这个选项，于是它学会了把多余的注意力倾倒在开头几个位置上。 那几个 token 不是在传递信息，<b>它们是停车位</b>。</p>
<div class="note ok"><p>⭐ <b>两个层次的教训，都要讲：</b></p>
<ol><li>工程上：任何"扔掉一部分 KV"的方案，都必须先问「有没有扔掉停车位」</li><li>方法上：这个 bug 从公式上完全看不出来，<b>只有把注意力矩阵画出来看才发现</b>。 这是这门课反复要说的那件事 —— <b>量出来的和算出来的，是两回事</b></li></ol>
<p>后续：DeepSeek-V4 干脆给每个头加了一个<b>可学习的 sink logit</b>， 直接加进 softmax 分母里，于是这一行的注意力总和<b>可以小于 1，甚至接近 0</b>。 把一个模型被迫发明的 hack，变成了架构里的一等公民。</p></div>
<h3>6.3 NSA —— 「native」是什么意思</h3>
<ul><li>三条支路并行：<b>压缩</b>（粗看全部）/ <b>选择</b>（细看挑出来的）/ <b>滑窗</b>（细看邻近）， 再用一个小 MLP + sigmoid 出的门控分数融合</li><li>效果：64k 长度下，<b>解码快 11.6×、前向快 9.0×、反向快 6.0×</b>（arXiv 2502.11089）</li></ul>
<p><b>「native」的意思是训练时就这么做，不是训练完再加的推理优化。</b> 这个区别很重要，值得单独说三十秒：</p>
<table>
<thead><tr><th></th><th>推理期稀疏</th><th>训练期稀疏（native）</th></tr></thead><tbody>
<tr><td>模型知不知道自己会被稀疏</td><td><b>不知道</b></td><td>知道，权重是在稀疏条件下学出来的</td></tr>
<tr><td>掉点</td><td>有，且难预测</td><td>小得多，甚至能反超</td></tr>
<tr><td>能不能省训练成本</td><td><b>不能</b></td><td>能（上面那个 6.0× 反向）</td></tr>
<tr><td>代价</td><td>无，随时可开关</td><td><b>要重训</b>，没法给已有模型打补丁</td></tr>
</tbody></table>
<div class="note ok"><p>⭐ 这张表解释了后面一个反复出现的现象： <b>为什么这些新注意力方案总是跟新模型一起发布，而不是作为一个推理框架的开关。</b></p></div>
<h3>6.4 DSA（DeepSeek-V3.2）—— Lightning Indexer</h3>
<ul><li><b>Lightning Indexer</b>：一个极轻量的打分网络，为每个 query 挑出 top-k 个 KV</li><li>打分式子 <code>I(t,s) = Σ_j w_j · ReLU(q_j · k_s)</code> —— ⭐ <b>用 ReLU 不用 softmax，纯粹是为了吞吐</b>，论文自己这么说的</li><li>头很少，而且跑在 <b>FP8</b> 上</li><li><b>k = 2048</b>。128K 上下文下，参与主注意力的从 128K 降到 2K —— <b>64 倍</b></li><li>复杂度从 O(L²) 变成 <b>O(L·k)</b></li><li>出处：arXiv 2512.02556</li></ul>
<p>⚠️ <b>两个必须一起讲的细节，不然这个方案听起来像免费午餐：</b></p>
<ol><li><b>indexer 自己还是 O(L²)。</b> 每个 query 要给所有 KV 打分才能挑 top-k。 省下来的是「主注意力的 O(L²)」，换来的是「一个便宜得多的 O(L²)」。 <b>省的是常数，不是阶。</b></li><li><b>它需要一个专门的训练阶段。</b> 先冻住除 indexer 之外的全部参数， 让 indexer 去<b>拟合主注意力自己的分布</b>（把各头注意力加起来、L1 归一化， 当作 indexer 的学习目标）。<b>稀疏是学出来的，不是规则定出来的。</b></li></ol>
<h3>6.5 ⭐ CSA + HCA（DeepSeek-V4）—— 压缩率按距离分层</h3>
<p>这是线索 B「远近有别」最直接的体现，也是这一节的高潮。</p>
<p><b>CSA</b>（Compressed Sparse Attention）</p>
<ul><li>每 <b>m = 4</b> 个 token 的 KV 压成 1 个 entry，然后在<b>压缩后的 entry 上</b>做 DSA</li><li>V4-Pro top-k = 1024，V4-Flash top-k = 512</li><li>压缩不是简单平均：用了<b>两套交错的压缩序列</b>（互相错开半个块）， 各带一组可学习的位置偏置，在 2m 个元素上做 softmax。 ⭐ <b>交错是为了不让块边界成为信息断层</b> —— 一个 token 总在某一套里落在块中间</li></ul>
<p><b>HCA</b>（Heavily Compressed Attention）</p>
<ul><li>压缩率 <b>m′ = 128</b>，远大于 CSA 的 4</li><li>而且<b>不做稀疏</b>：压完之后在全部 entry 上做<b>密集</b>注意力</li><li>⭐ <b>逻辑很干净</b>：都压了 128 倍了，剩下的条目本来就没几个，再挑就没意义了</li></ul>
<p><b>两者混排</b>：V4-Pro 前两层用 HCA，之后 HCA 与 CSA 交替； V4-Flash 前两层用纯 SWA。</p>
<p><b>效果</b>（都是 V4 技术报告自己给的）：</p>
<table>
<thead><tr><th>口径</th><th>数字</th></tr></thead><tbody>
<tr><td>1M 上下文，vs DeepSeek-V3.2</td><td>单 token 推理 FLOPs <b>27%</b>、KV cache <b>10%</b>（V4-Pro）</td></tr>
<tr><td>同上，V4-Flash</td><td>FLOPs <b>10%</b>、KV cache <b>7%</b></td></tr>
<tr><td>以 BF16 GQA-8（head dim 128）为基线，1M 上下文</td><td>KV cache 降到约 <b>2%</b></td></tr>
</tbody></table>
<p>⭐ 最后那个 2% 值得停一下算给学生看：同样形状的 GQA-8 在 1M 下是 244 GiB， 2% 就是 不到 5 GiB —— 一百万 token 的上下文，KV 装得进一块卡的零头。 对照本节开头那个 MHA 的 488 GiB（那还只是 128K）。<b>这就是三年的进展。</b></p>
<div class="note warn"><p>⚠️ 这里还叠了一层跟注意力机制无关的优化：<b>KV 混合精度存储</b> —— RoPE 那几维用 BF16，其余用 FP8，光这一项就"近乎减半"。 报告一个总收益的时候，要能拆出哪几分是机制带来的、哪几分是精度带来的。</p></div>
<h3>6.5b ⭐ 第二阶段：索引本身成了开销 —— IndexShare 与 IndexCache</h3>
<p>前面四个方案（NSA / DSA / CSA+HCA）都有一个共同的零件：一个决定"该看哪几块"的索引器。 §5.4 讲 DSA 的时候它叫 Lightning Indexer。到这一步为止，所有心思都花在<b>让每个 query 少看几块</b>上。</p>
<p><b>但索引器自己也要算。</b> 它要为<b>每一层、每一个 token</b>，跟全部历史块打一次分。稀疏注意力把主体那部分省下来之后， 这笔原本不起眼的账就浮上来了 —— 尤其在 1M 上下文下，历史块本身就有几万个。</p>
<p>2026 年年中，两家公司几乎同时给出了同一个答案：<b>别每层都重新算一遍"该看谁"。</b></p>
<table>
<thead><tr><th>模型</th><th>叫法</th><th>做法</th><th>官方给的收益</th></tr></thead><tbody>
<tr><td><b>GLM-5.2</b>（智谱，2026-06-16，744B）</td><td><b>IndexShare</b></td><td>每四个稀疏注意力层<b>共用同一个索引器</b></td><td>1M 上下文下每 token FLOPs 降 <b>2.9×</b></td></tr>
<tr><td><b>混元 Hy4-preview</b>（腾讯，2026-08-28，770B/49B）</td><td><b>IndexCache</b></td><td>同上：跨层复用稀疏索引</td><td>未单独给数</td></tr>
</tbody></table>
<div class="note ok"><p>⭐ 这两家撞了同一个想法，而且证据不用查博客 —— <b>打开两份 config 就看得见</b>。 它们的 <code>indexer_types</code> 字段都是同一个循环：</p>
<pre><code>GLM-5.2  ： full, full, full, shared, shared, shared, full, shared, shared, shared, ...
混元 Hy4 ： full, full,       shared, shared, shared, full, shared, shared, shared, ...
                              └──────── 每 4 层里，只有 1 层自己算索引 ────────┘</code></pre>
<p>GLM-5.2 的 config 里还有一个 <code>index_topk_freq: 4</code> 直接把这个 4 写了出来。</p></div>
<p>⭐ <b>为什么这一小节值得单独留一块地方</b>：它是一个<b>优化制造出新的被优化对象</b>的干净例子。 稀疏注意力是为了省 attention 的账而来的；省成了，于是索引 —— 它原本只是这个方案的附属零件 —— 变成了新的大头，再被优化一轮。这一节讲的三个旋钮都会经历这一步，不只是稀疏这一支。</p>
<div class="note warn"><p>⚠️ <b>这不是白拿的，代价要说出来。</b> 官方博客只报了省下来的 FLOPs。 但共享索引器意味着<b>这四层被迫看同一批块</b> —— 它们不能各自挑各自的。 这是表达力上的一次让步：原本每一层可以按自己那一层的语义去决定关注哪里，现在四层绑在一起。</p>
<p>📌 口径：上面这句是从机制推出来的，不是引用 —— 索引器共享，选出的 top-k 集合当然就一样。 两家都没有公开这项让步对质量的影响有多大。<b>看到"降 2.9× FLOPs"这种数字，先找它换走了什么。</b></p></div>
<h3>6.6 ⭐ 这一支的共同结构</h3>
<p>讲完四个方案，回头把它们叠在一起，会看到同一个骨架：</p>
<pre><code>                 ┌─ 一条"粗看"的路：压缩 / 全局，保证不漏
一个 query 的注意力 ┼─ 一条"细看"的路：挑出来的 top-k，保证准
                 └─ 一条"近处"的路：滑动窗口，保证局部连贯</code></pre>
<ul><li><b>NSA</b>：三条都有，显式的三支路 + 门控融合</li><li><b>DSA</b>：主要是中间那条（top-k），配一点局部</li><li><b>CSA/HCA</b>：HCA 是粗看，CSA 是细看，另外还挂了一条 n_win=128 的滑窗支路</li><li><b>SWA 单用</b>：只有第三条 —— 所以它单用不行</li></ul>
<div class="note ok"><p>⭐ <b>看到一个新的稀疏注意力方案，先问它这三条路各占什么位置。</b> 这比记住它叫什么有用得多。</p></div>
<hr>
</div></section>
<section id="s七"><div class="wrap"><div class="stn"><span class="badge">第 七 节</span><h2>旋钮③：换回一个固定大小的状态 —— 线性注意力</h2></div>
__FIG_TX_K3__
<h3>7.1 基本换法</h3>
<p>softmax 注意力必须把所有 K 都留着，是因为 softmax 的分母要对<b>所有位置</b>求和 —— 你没法提前把它们合并。</p>
<p><b>把 softmax 去掉</b>（换成某个可分解的核函数），求和就可以重排：</p>
<pre><code>softmax 版： out_t = Σ_{s≤t} softmax(q_t·k_s) v_s      ← 必须留下所有 (k_s, v_s)
线性版：     S_t   = S_{t-1} + k_t v_tᵀ                ← 一个固定大小的状态
            out_t = q_t S_t</code></pre>
<p>于是：</p>
<ul><li>复杂度 O(L²·d) → <b>O(L·d²)</b>，对长序列是数量级的差别</li><li><b>没有随长度增长的 KV cache</b> —— 只有一个 <code>d_k × d_v</code> 的状态矩阵</li><li>推理时它就是一个 <b>RNN</b>：读一个 token、更新一次状态、吐一个输出</li></ul>
<p><b>代价说死</b>：状态大小固定 → <b>信息必然有损</b>。 序列越长，往同一个矩阵里塞的东西越多，长程精确检索（"第 30 万字提到的那个电话号码"） 会力不从心。这不是实现问题，是这个换法的性质。</p>
<h3>7.2 谱系：四步，每一步都在修上一步的一个具体毛病</h3>
<table>
<thead><tr><th>步</th><th>名字</th><th>补了什么</th><th>出处</th></tr></thead><tbody>
<tr><td>1</td><td>朴素线性注意力</td><td>把 softmax 拆掉，换来固定状态</td><td>Katharopoulos 等, arXiv 2006.16236</td></tr>
<tr><td>2</td><td><b>delta rule</b></td><td>状态只加不减 → 写满就糊。改成「<b>先擦掉旧的，再写新的</b>」</td><td>Schlag 等, arXiv <b>2102.11174</b>（2021）</td></tr>
<tr><td>3</td><td><b>GDN</b>（Gated DeltaNet）</td><td>delta rule 不会主动遗忘 → 加一个<b>遗忘门</b>，让陈年旧事自然衰减</td><td>arXiv <b>2412.06464</b>（2024-12）</td></tr>
<tr><td>4</td><td><b>KDA</b></td><td>遗忘门是个<b>标量</b>，全部通道同一个速度 → 升级成 <b>per-channel 向量</b></td><td>Kimi Linear, arXiv <b>2510.26692</b>（2025-10）</td></tr>
</tbody></table>
<p>⚠️ <b>两处容易记错的出处，讲的时候要说对：</b></p>
<ul><li><b>delta rule 不是 2024 年的东西</b>，是 Schlag 等 2021 年那篇 <em>Linear Transformers Are Secretly Fast Weight Programmers</em>（可上溯到 1990 年代的 fast weight programmer）。2024 年那篇（arXiv 2406.06484）做的是 <b>把它并行化</b>，这是另一件事、也是很关键的一件事 —— 见 §6.4</li><li><b>KDA 是 2025 年 10 月的 Kimi Linear</b>，不是 2026 年</li></ul>
<p>⭐ <b>第 4 步为什么值得单独说</b>：标量遗忘门意味着「整个状态一起变旧」。 per-channel 意味着<b>不同特征维度可以有各自的遗忘速度</b> —— 有些通道记语法（该快忘），有些记实体名（该慢忘）。 KDA 还把它做成了一个特殊的 <b>DPLR（对角 + 低秩）</b>形式， 正是因为这个特殊形式，才配得出一个比通用 DPLR 便宜得多的分块并行算法。 表达力和可算性是一起设计的，不是先设计再优化。</p>
<h3>7.2b ⭐⭐ Mamba 在哪？——&nbsp;它不是另一支，它是这一支的祖宗</h3>
<p>讲到这里一定会有人问：<b>那 Mamba 呢？SSM 那一路怎么没提？</b> 这个问题问得对，而答案比「另开一支」有意思得多。</p>
<div class="note ok"><p>⭐ 最硬的一条证据，不用推：Gated DeltaNet 那篇论文的标题就叫 <em>《Gated Delta Networks: Improving Mamba2 with Delta Rule》</em>（arXiv 2412.06464）。 而千问 Qwen3-Next 和 Qwen3.5 用的就是 GDN。 <b>所以千问那一支的祖宗，字面意义上就是 Mamba-2。</b></p></div>
<p>把它们放到同一个式子里看，一切就清楚了。上面 §6.1 那个递推稍微写全一点：</p>
<pre><code>S_t = A_t · S_{t-1} + v_t k_tᵀ        ← 所有这些方法都长这样

区别只有一处：允许 A_t 长什么样。</code></pre>
<table>
<thead><tr><th>方法</th><th>A_t 的结构</th><th>它解决了上一步的什么毛病</th></tr></thead><tbody>
<tr><td>朴素线性注意力（2020）</td><td><b>I</b>（单位阵）</td><td>——&nbsp;只加不减，写满就糊</td></tr>
<tr><td>RetNet / GLA</td><td>标量 γ 或对角 <b>Diag(γ)</b> 衰减</td><td>让陈年旧事自己淡出</td></tr>
<tr><td>Mamba（2023）</td><td>对角，而且<b>依赖输入</b>（selective）</td><td>衰减速度由内容决定，不是固定的</td></tr>
<tr><td><b>Mamba-2</b>（2024-05）</td><td><b>A ＝ a<sub>t</sub>·I</b>（标量×单位阵）</td><td>⭐ <b>故意退回最简形式</b>——正因为退了，才证得出跟线性注意力<b>对偶</b>（SSD），才能把递推写成矩阵乘、吃上 Tensor Core</td></tr>
<tr><td>DeltaNet（2021 / 2024 并行化）</td><td><b>I − β k kᵀ</b>（单位阵减秩一）</td><td>先擦掉旧的再写新的，而不是硬加</td></tr>
<tr><td><b>GDN</b>（2024-12）</td><td><b>α(I − β k kᵀ)</b></td><td>把 Mamba-2 的门控 ＋ DeltaNet 的擦除<b>合到一起</b></td></tr>
<tr><td><b>KDA</b>（2025-10）</td><td><b>Diag(α)(I − β k kᵀ)</b></td><td>门从标量升成<b>逐通道</b>——不同特征各有各的遗忘速度</td></tr>
<tr><td>Mamba-3（2026-03）</td><td>面向<b>推理</b>重新设计</td><td>前几代都在优化训练，这一代优化部署</td></tr>
</tbody></table>
<p>⭐ <b>Mamba-2 那篇论文的题目更直白</b>：<em>《Transformers are SSMs》</em>（arXiv 2405.21060）。 它证明的正是 selective SSM 和「带结构化掩码的线性注意力」是同一件事的两种写法——这就是 SSD（State Space Duality）。</p>
<div class="note warn"><p>⛔ <b>贯穿这张表的，是一条硬约束：A<sub>t</sub> 必须有足够的结构，才能做分块并行。</b></p>
<ul><li><b>全矩阵 A</b>：表达力最强，但没有高效的并行扫描——只能一步一步递推，<b>并行度直接归零</b></li><li><b>对角</b>：最容易并行</li><li><b>单位阵减秩一</b>：要靠 2024 年那篇「可并行 DeltaNet」的技巧才算得动</li><li><b>DPLR（对角 ＋ 低秩）</b>：KDA 专门设计了一个<b>特化版本</b>，才能把它压进 Tensor Core</li></ul>
<p>⭐ 所以这张表不是「表达力越来越强」的单调故事。<b>每一步都在「A 能多复杂」和「还算不算得动」之间重新划线</b>—— 这正是 §6.2 那句「表达力和可算性是一起设计的」的完整版。</p></div>
<div class="note ok"><p>⭐⭐ <b>回到最初那个问题：Mamba 缺了吗？</b> 没缺——它在表里，只是我们一直用它下游的名字（GDN、KDA）在叫它。 编年史那张图的第三条泳道里，Mamba、Mamba-2、Mamba-3 现在都标上了，<b>跟 DeltaNet 一族在同一条线上</b>，因为它们本来就是。</p></div>
<h3>7.3 ⚠️ 它不是「更快的 attention」，是另一个模型</h3>
<p>这是本节最重要的一句话，也是最容易被听众误解的一句。</p>
<p>三个旋钮里，只有旋钮 ③ <b>改变了模型能表达什么</b>：</p>
<ul><li>旋钮 ①②：改的是<b>存法</b>和<b>看多少</b>，理论上你还能指着某个历史 token 说 "注意力权重在这儿"。<b>检索是显式的</b></li><li>旋钮 ③：历史被<b>碾进了一个矩阵</b>。你无法指着状态里的某一块说 "这是第 3 万个 token"。<b>检索是隐式的、有损的</b></li></ul>
<p>所以：</p>
<ul><li>它<b>不能</b>给已有模型打补丁，必须<b>从头训</b></li><li>评测上要特别看 <b>needle-in-a-haystack 这类精确检索任务</b>， 平均分好看不代表这一类不塌</li><li>也正因如此，<b>几乎没有人纯用线性注意力</b> —— 全都是混合（第五节）</li></ul>
<h3>7.4 硬件视角：串行的状态怎么榨出并行度</h3>
<p><b>这一段是我们的角度，别人的课不会这么讲。</b></p>
<p><code>S_t = S_{t-1} + ...</code> 是<b>逐 token 串行</b>的。而加速器要的是宽而规整的并行。 训练一个 100 万 token 的序列，串行跑一百万步 —— 直接不用想。</p>
<p>解法叫 <b>chunkwise parallel</b>：<b>块内并行、块间串行</b>。 把序列切成长为 C 的块，块内用矩阵乘一次算完，块与块之间才传递状态。 于是并行度从 1 变成 C，串行步数从 L 变成 L/C。</p>
<p>⭐ <b>C 怎么选，是一个纯硬件问题</b>：</p>
<ul><li>C 太小 → 串行步数多，而且每步的矩阵乘太瘦，算力吃不满</li><li>C 太大 → 块内那个中间矩阵放不进片上内存（SRAM / VMEM），要往 HBM 上倒</li><li><b>所以 chunk 大小是被片上内存容量顶死的</b> —— 跟<a href="专题01-一个-Token-的一生.md">专题一</a> splash attention 的块大小是同一类问题</li></ul>
<p><b>这条路有多难走，看两个信号：</b></p>
<ul><li>K3 为了 KDA 专门写了 CUTLASS 的 <b>FlashKDA</b> kernel， 因为块内并行和块间串行<b>交替进行时 SM 会空转</b></li><li>长序列训练还要 <b>KDA Context Parallelism</b> —— 标准的序列并行做法是把各段 局部结果直接相加，<b>但这对 KDA 不成立</b>，因为 delta rule 的状态更新是 token 相关的矩阵连乘，前一段的影响不是简单的加法</li></ul>
<div class="note ok"><p>⭐ 一句话收：<b>线性注意力在纸上是 O(L)，在硬件上是一场 kernel 战争。</b> 我们在 TPU 上写过 KDA kernel，这里有一手的坑可以摆。</p></div>
<hr>
</div></section>
<section id="s八"><div class="wrap"><div class="stn"><span class="badge">第 八 节</span><h2>旋钮④（其实是元旋钮）：混合</h2></div>
<h3>8.1 为什么混合几乎是唯一的答案</h3>
<p>单用任何一个旋钮都有一个致命短板：</p>
<table>
<thead><tr><th>单用</th><th>短板</th></tr></thead><tbody>
<tr><td>SWA</td><td>跨不了长距离</td></tr>
<tr><td>纯线性</td><td>精确检索塌</td></tr>
<tr><td>纯全注意力</td><td>KV 和 FLOPs 都爆</td></tr>
</tbody></table>
<p>混合的逻辑很朴素：全局层负责精确长程检索，线性/窗口层负责局部与效率，各司其职。 关键在于全局层<b>不需要很多</b> —— 只要有几层能做无损检索， 信息就能沿着残差流传给其余层用。</p>
<h3>8.2 配比：3:1 是怎么来的，以及它不是定律</h3>
<table>
<thead><tr><th>模型</th><th>配比</th><th>出处</th></tr></thead><tbody>
<tr><td>Kimi Linear</td><td>KDA : 全注意力 MLA = <b>3 : 1</b>（48B 总 / 3B 激活）</td><td>arXiv 2510.26692</td></tr>
<tr><td><b>Kimi K3</b></td><td>KDA : Gated MLA = <b>3 : 1</b>（<b>69 KDA + 24 MLA = 93 层</b>）</td><td>arXiv 2607.24653 表 1</td></tr>
<tr><td>Ling-3.0-tiny</td><td>KDA : MLA = 3 : 1</td><td>模型卡</td></tr>
<tr><td>Ling-3.0-flash</td><td>KDA : MLA = <b>5 : 1</b></td><td>模型卡</td></tr>
<tr><td>一篇系统性消融</td><td>建议区间 <b>3:1 ～ 6:1</b></td><td>arXiv 2507.06457</td></tr>
</tbody></table>
<p>⭐ <b>三件事要讲清楚：</b></p>
<ol><li><b>3:1 是消融出来的，不是推出来的。</b> Kimi Linear 的消融里， <b>0:1（纯全注意力）反而表现不好</b> —— 这个结果比"3:1 最好"更有意思： <b>加线性层不只是省钱，它可能还带来了别的东西</b></li><li><b>同一家不同规模就换了配比</b>（Ling 的 tiny 3:1 / flash 5:1）—— <b>配比是超参，跟规模和数据有关，不要背下来当常识</b></li><li><b>区间比点值可信。</b> 记 "3:1 到 6:1 这个量级" 就够了</li></ol>
<h3>8.3 ⭐ K3 的 NoPE —— 混合带来的一个意外红利</h3>
<p>这是全节最漂亮的一处，值得留三分钟。</p>
<p>K3 的全注意力（Gated MLA）层<b>完全不加位置编码</b>（NoPE）： 没有 RoPE，没有 YaRN，什么都没有。</p>
<p>为什么敢这么做？ 因为它们中间夹着的 KDA 层， 本身就是靠递归的衰减和门控在编码顺序 —— 一个天然带时序的算子。 <b>位置信息由线性层提供，全注意力层只管检索。</b></p>
<p>三个后果，一个比一个实在：</p>
<ol><li><b>不用调 RoPE 外推。</b> 模型直接外推到 1M，不需要任何位置编码的重标定 —— 长上下文扩展里最烦人的一块调参，直接消失了</li><li><b>MLA 层在推理时可以退化成纯 MQA。</b> 位置编码没了， §5.3 里那条"不可吸收的 64 维"也就不存在了 —— <b>上投影可以完全吸收</b></li><li><b>KV cache 最多降 75%</b>（Kimi Linear 的数字）， 1M 上下文下 TPOT 从 11.48 ms 降到 1.84 ms，<b>6.3×</b></li></ol>
<div class="note ok"><p>⭐ <b>这才是"混合"真正的意思</b>：不是"两个方案各跑一半凑合用"， 而是让每一层只做自己擅长的事，然后把别人不用做的事一并省掉。 一个架构选择（混合）解开了另一个看起来完全无关的约束（位置编码）。 这门课想教的就是这种"看见约束之间的连接"的能力。</p></div>
<h3>8.4 ⭐ 各家速查：你日常在用的那些模型，注意力到底是什么</h3>
<p>三个旋钮到这里就拆完了。这一小节反过来 —— <b>按公司排一遍，看每一家实际拧的是哪个旋钮</b>。 都是能在公开 config 或官方博客里查到的，信息截至 <b>2026-09-07</b>。</p>
<table>
<thead><tr><th>家</th><th>代表型号</th><th>拧的是哪个旋钮</th><th>配比 / 形态</th></tr></thead><tbody>
<tr><td rowspan="2">阿里 千问</td><td>Qwen3-Next（80B/3B）</td><td>③ 线性（Gated DeltaNet）</td><td><b>3 : 1</b></td></tr>
<tr><td>Qwen3.5（0.8B–397B）</td><td>③ 线性</td><td><b>3 : 1</b>，全家族统一</td></tr>
<tr><td rowspan="2">月之暗面 Kimi</td><td>Kimi Linear（48B/3B）</td><td>③ 线性（KDA）</td><td><b>3 : 1</b></td></tr>
<tr><td>Kimi K3（2.8T）</td><td>③ 线性 ＋ NoPE</td><td>93 层 ＝ <b>69 KDA ＋ 24 Gated MLA</b></td></tr>
<tr><td rowspan="2">蚂蚁 百灵 Ling</td><td>Ling 2.6</td><td>③ 线性（Lightning）</td><td><b>7 : 1</b></td></tr>
<tr><td>Ling-3.0-flash（124B/5.1B）</td><td>③ 线性（KDA）</td><td><b>5 : 1</b> ＝ 35 KDA ＋ 7 MLA</td></tr>
<tr><td rowspan="2">小米 MiMo</td><td>MiMo-V2-Flash</td><td>② 稀疏（SWA，窗口 128）</td><td><b>5 : 1</b></td></tr>
<tr><td>MiMo-V2.5-Pro</td><td>② 稀疏（SWA，窗口 128）</td><td><b>6 : 1</b></td></tr>
<tr><td rowspan="2">DeepSeek</td><td>V3.2</td><td>② 稀疏（DSA）</td><td>层内稀疏</td></tr>
<tr><td>V4</td><td>② 稀疏（CSA ＋ HCA）</td><td>层内稀疏，按距离分层压缩</td></tr>
<tr><td>MiniMax</td><td>01 → M2 → M3</td><td>③ → 退回全注意力 → ②</td><td>7 : 1 → 纯全 → 层内稀疏</td></tr>
</tbody></table>
<p>下面两家单独展开 —— 它们的轨迹在公开 config 里看得特别清楚，而且刚好是<b>两种完全不同的走法</b>。</p>
<p><b>① 腾讯混元 —— 跳过线性那一支，直接进稀疏</b></p>
<table>
<thead><tr><th>看哪一项</th><th>Hy3（295B/21B）</th><th>Hy4-preview（770B/49B）</th></tr></thead><tbody>
<tr><td>发布</td><td>preview 2026-04-23，正式版 2026-07-06</td><td>2026-08-28</td></tr>
<tr><td>层数</td><td><b>80</b></td><td><b>78</b></td></tr>
<tr><td>注意力</td><td><b>纯 GQA-8</b>（64 头 / 8 KV 头，head dim 128）</td><td><b>全部 78 层都是 Gated DSA</b>（<code>layer_types</code> 全为 <code>deepseek_sparse_attention</code>）</td></tr>
<tr><td>混合</td><td><b>没有</b> —— 不掺线性，不掺稀疏</td><td>不是层间混合，是<b>层内稀疏</b>；索引器 32 头 × 128 维，top-k <b>2048</b></td></tr>
<tr><td>上下文</td><td>256K</td><td><b>1M</b></td></tr>
<tr><td>另外</td><td>192 专家 ＋ 1 共享，top-8</td><td>256 ＋ 1 共享 top-8；<b>iHC</b>（4 条残差流）；<code>gated_mla</code>；IndexCache</td></tr>
</tbody></table>
<div class="note ok"><p>⭐ Hy3 那一列不是查来的，是读我们自己仓库里那份 config 数出来的 —— <code>tpu/Hunyuan3-295B-Pretraining/</code> 底下就有。<b>80 层里没有一层线性、没有一层稀疏。</b></p>
<p>这跟"腾讯评估过线性注意力但最终没上"的说法对得上，而且是一手证据。 然后 Hy4 一步跨到全层稀疏 —— 它整个跳过了线性这一支。 对照 §7.2 那张配比表：<b>混合不是唯一解，只是最多人选的那个解。</b></p></div>
<p><b>② 智谱 GLM —— 半年之内走完三步，而且步步可查</b></p>
<table>
<thead><tr><th>版本</th><th>时间</th><th>注意力</th><th>这一步新增了什么</th></tr></thead><tbody>
<tr><td>GLM-5（355B–744B）</td><td>2026-02-12</td><td>MLA ＋ <b>DSA</b></td><td>智谱第一次上稀疏</td></tr>
<tr><td>GLM-5.2（744B）</td><td>2026-06-16</td><td>MLA ＋ DSA ＋ <b>IndexShare</b></td><td>每四个稀疏层共用一个索引器（见 §6.5b），1M 下省 <b>2.9×</b> FLOPs</td></tr>
<tr><td><b>GLM-5.3-Flash</b>（321B/18B）</td><td>2026-08-26</td><td><b>KDA 线性 ＋ NoPE 稀疏 MLA</b></td><td>⭐ GLM 家族<b>第一次把线性和稀疏放进同一个模型</b>；原生多模态</td></tr>
</tbody></table>
<p>GLM-5.3-Flash 的 <code>layer_types</code> 是一个干净的四层循环：</p>
<pre><code>linear, linear, linear, deepseek_sparse_attention,   ← 重复 11 次
linear                                               ← 第 45 层多出来的一层

45 层 = 34 层 KDA + 11 层稀疏 MLA        循环配比 3 : 1</code></pre>
<div class="note ok"><p>⭐ 为什么单说这一个型号：它是本课整张配比表里<b>唯一一个两种便宜法同时上</b>的模型。</p>
<p>看清楚它的两层各是什么：便宜的那层是线性（KDA），而它配的那层"贵的"—— 本身已经是稀疏的了。 别的混合模型是"线性配全注意力"，它是"线性配稀疏"。 <b>三个旋钮不是三选一，是可以叠着拧的。</b></p>
<p>而且配比正好落回 <b>3 : 1</b>，跟 Kimi Linear、Qwen3.5 一样 —— 换了一家公司、换了搭档层的类型，配比还是那个区间。 这是 §7.2 那句"3:1 到 6:1 这个量级"到目前为止最强的一个旁证。</p></div>
<hr>
</div></section>
<section id="s九"><div class="wrap"><div class="stn"><span class="badge">第 九 节</span><h2>代价：没有免费的午餐</h2></div>
<p>一张表把所有方案摆在一起：</p>
<table>
<thead><tr><th></th><th>KV 显存</th><th>计算量</th><th>长程质量</th><th>kernel 复杂度</th><th>能否给已有模型打补丁</th></tr></thead><tbody>
<tr><td>MHA</td><td>基准</td><td>基准</td><td>基准</td><td>简单</td><td>—</td></tr>
<tr><td>GQA</td><td>↓↓</td><td>—</td><td>↓</td><td>简单</td><td>需微调</td></tr>
<tr><td>MLA</td><td>↓↓↓</td><td>↑（训练时）</td><td>≈</td><td>中</td><td>不能</td></tr>
<tr><td>SWA</td><td>↓↓↓</td><td>↓↓</td><td>↓↓↓</td><td>简单</td><td>勉强（要留 sink）</td></tr>
<tr><td>DSA</td><td>↓</td><td>↓↓↓</td><td>≈</td><td><b>高</b></td><td>需专门训练阶段</td></tr>
<tr><td>CSA/HCA</td><td>↓↓↓</td><td>↓↓↓</td><td>≈</td><td><b>很高</b></td><td>不能</td></tr>
<tr><td>线性（KDA 等）</td><td><b>无 KV</b></td><td>↓↓↓</td><td>↓↓</td><td><b>很高</b></td><td><b>不能，必须从头训</b></td></tr>
</tbody></table>
<p>要讲透的<b>四个</b>取舍：</p>
<ol><li><b>省显存 ≠ 省计算。</b> MLA 省了显存但增加了计算；DSA 省了计算但 KV 还在那儿。 <b>这两个是不同的资源，问"省了多少"之前先问"省的是哪一样"</b></li><li><b>训练时省和推理时省是两回事。</b> MLA 的压缩在训练前向里不生效； NSA 的 native 意味着训练时也省。<b>一个方案属于哪一类，直接决定它能不能被采用</b></li><li><b>不规则访存的代价常常被低估。</b> 稀疏方案在纸面上是 64 倍， 落到硬件上是 gather、是不连续访问、是 kernel 难写 —— <b>实际加速比远小于理论值</b>。这一点我们有实测可以摆</li><li>⭐ <b>收益是有天花板的，因为注意力只是账单的一部分。</b> 把注意力砍到 0，剩下的 MoE、MLP、通信一分钱没省。 <a href="专题01-一个-Token-的一生.md">专题一</a>那条曲线说得很清楚：短上下文下注意力只占 12%， <b>这时候你把它优化到极致，端到端也就快 10%</b>。 <b>所有这些技术的价值都随上下文长度而涨</b> —— 讲的时候必须带上"在多长的上下文下"， 不然那些倍数全是耍流氓</li></ol>
<hr>
</div></section>
<section id="s十"><div class="wrap"><div class="stn"><span class="badge">第 十 节</span><h2>落到硬件（本专题的落点）</h2></div>
<p>回到全课那条主线：<b>每一个变体都是被硬件逼出来的，也都对硬件提出了新要求。</b></p>
<table>
<thead><tr><th>变体</th><th>它假设了什么硬件条件</th><th>条件不成立会怎样</th></tr></thead><tbody>
<tr><td>MLA</td><td>算力相对充裕、显存相对紧张</td><td>算力紧张的机器上，用计算换显存这笔交易不划算</td></tr>
<tr><td>稀疏（DSA/NSA/CSA）</td><td>gather 不太贵</td><td><b>对规整访存友好的加速器反而吃亏</b> —— 纸面 64 倍拿不到</td></tr>
<tr><td>线性（KDA）</td><td>片上内存够放下 chunk 的中间量</td><td>chunk 被迫调小 → 并行度掉 → 优势被吃掉</td></tr>
<tr><td>长上下文 + MoE 同时上</td><td>HBM 带宽够两边分</td><td>all-to-all 与 KV cache <b>抢同一份带宽</b></td></tr>
</tbody></table>
<div class="note info"><p>一句话收尾：注意力的变体史，就是一部 「在显存、算力、访存规整度三者之间反复搬家」的历史。 早期搬显存（MQA/GQA/MLA），中期搬算力（稀疏）， 现在在搬访存规整度（chunk 化的线性注意力）—— <b>而访存规整度是最难搬的那一样。</b></p></div>
<hr>
</div></section>
<section id="s十一"><div class="wrap"><div class="stn"><span class="badge">第 十一 节</span><h2>收尾：把谱系放回时间线</h2></div>
<p><b>前面那些节讲的是谱系（可迁移的判断框架），这一节是时间线（记忆的挂钩）。</b> 顺序不能反 —— 先给框架，时间线才有意义；先给时间线，框架就变成了流水账。</p>
<p>板书画一条线，把上面所有名词按年份钉上去：</p>
<pre><code>2020  线性注意力          「softmax 拆了会怎样」        —— 想法有了，效果不行
2021  delta rule          「状态要能擦除」              —— 修了第一个毛病
2022  FlashAttention      「矩阵根本不用存」            —— 不改数学的那一支分岔
2023  GQA / SWA / sink    「工程上先把它压下去」        —— 三个简单办法同年出现
2024  MLA / GDN           「低秩压缩」「遗忘门」        —— 两条支线各自成熟
2025  NSA / DSA / KDA     「稀疏要在训练时就用」        —— 从推理补丁变成架构
2026  CSA+HCA / K3        「按距离分层」「混合 + NoPE」  —— 组合拳，1M 成为常规</code></pre>
<p>⭐ <b>让学生自己读出这条线的形状</b>（讲师不要先说出答案）：</p>
<ol><li><b>前半段是单点突破，后半段全是组合。</b> 2025 年之后没有哪个模型只用一招</li><li><b>"推理期的补丁"逐年变成"训练期的架构"</b> —— NSA 的 native、DSA 的训练阶段、 K3 的从头混合训练，是同一个趋势的三次出现</li><li><b>每一步都是在修上一步暴露出来的具体毛病</b>，不是凭空发明。 所以<b>下一步大概率也是在修今天这批方案暴露的毛病</b> —— 那么今天这批的毛病是什么？（留给学生，也留给下一版课件）</li></ol>
<hr>
</div></section>
<section id="s十二"><div class="wrap"><div class="stn"><span class="badge">第 十二 节</span><h2>这个专题明确不讲</h2></div>
<ul><li><b>MLA 逐步的矩阵推导</b> → 在<a href="专题01-一个-Token-的一生.md">专题一</a>第 2 步。 ⭐ <b>分工是这样定的</b>：<b>专题一讲"V3 这一个模型里它怎么算"， 这里讲"为什么会有它、它在谱系里站哪、它换走了什么"。</b> 这里只用一张图复述结论（576 = 512 + 64），<b>不重讲推导</b> —— 重讲会占掉 5 分钟，而这 5 分钟买不到任何新东西</li><li><b>注意力的 kernel 怎么写</b> → 实现细节在<a href="专题07-性能调优与工具链.md">专题七</a></li><li><b>序列并行 / Context Parallelism 怎么切</b> → <a href="专题05-并行策略.md">专题五</a>。 这里只说"线性注意力的 CP 跟标准 CP 不一样"，不展开</li><li><b>prefill / decode 的形状差异</b> → <a href="专题06-推理.md">专题六</a></li><li><b>各家模型的完整参数表</b> → <a href="专题09-最新开源模型对比.md">专题九</a>。 这里给的每个数字都只为说明一个机制，<b>不做横向评测</b></li><li><b>Mamba / SSM 那一支</b> —— 跟线性注意力是近亲，但它自成体系。 这门课的听众用不上，<b>明确不讲</b>（问到就说一句"同一个思路的另一个分支"）</li></ul>
<hr>
</div></section>
<!-- ⭐⭐ 2026-09-07 定性。现场问：「后面那些内容是先放在那，一边写一边提取，
     吸收完就没了？还是现在就砍掉？」——&nbsp;**这两块是两种东西，不该一起处置。**

     【这一块 · 出处清单】**不是待吸收的草稿，是参考书目，永久留着。**
     ⭐ 判据：这门课到处在下硬断言（层数、头数、KV 多大、谁比谁快几倍），
       **一份敢下硬断言的材料，就必须随时能把出处摊出来。**
       它不会随着章节写完而「被吸收掉」——&nbsp;章节写得越多，它越该长。
     ⛔ 但按新规矩它不能裸露在正文里：**学员扫到一屏 arXiv 号只会觉得这份材料没写完。**
       所以 ① 改名（「素材」听起来像半成品，「出处」是成品该有的东西）
            ② 折叠（想验的人点开，不想验的人看不见）
            ③ ⭐ 以后写新章节时，**正文里就地标出处，这里同步补一行** —— 两边都要有。 -->
<section id="x13"><div class="wrap">
<details class="aside"><summary>📚 <b>这一讲的出处清单</b>
<em>（每个机制配它的一手论文；想复核任何一个数字都从这儿进）</em></summary>

<table>
<thead><tr><th>要什么</th><th>在哪</th></tr></thead><tbody>
<tr><td>MQA / GQA</td><td>arXiv <b>1911.02150</b> / <b>2305.13245</b></td></tr>
<tr><td>MLA</td><td>DeepSeek-V3, arXiv <b>2412.19437</b> <b>sec. 2.1 + 4.2</b>（超参那段给了 <code>n_h/d_h/d_c/d_h^R</code> 的准确值）</td></tr>
<tr><td>Gated MLA / K3 全貌</td><td>Kimi K3, arXiv <b>2607.24653</b> <b>sec. 2.1.2</b>、表 1（93 层 / 69 KDA + 24 MLA / 2.78T-104.2B）</td></tr>
<tr><td>SWA</td><td>Mistral 7B, arXiv <b>2310.06825</b>（窗口 4096）</td></tr>
<tr><td>Attention sink</td><td>StreamingLLM, arXiv <b>2309.17453</b>（4 个 token / 400 万 / 22.2×）</td></tr>
<tr><td>NSA</td><td>arXiv <b>2502.11089</b>（三支路 + 门控；64k 下 11.6× / 9.0× / 6.0×）</td></tr>
<tr><td>DSA + Lightning Indexer</td><td>DeepSeek-V3.2, arXiv <b>2512.02556</b>（ReLU 打分 / FP8 / k=2048 / 稠密预热阶段）。<b>我们有一手实测</b></td></tr>
<tr><td>CSA / HCA</td><td>DeepSeek-V4, arXiv <b>2606.19348</b> <b>sec. 2.3 + 2.3.4</b>（m=4 / m′=128 / top-k / 27%·10% / 2%）</td></tr>
<tr><td>线性注意力谱系</td><td><b>2006.16236</b>（线性）→ <b>2102.11174</b>（delta rule, 2021）→ <b>2406.06484</b>（可并行化）→ <b>2412.06464</b>（GDN）→ <b>2510.26692</b>（KDA）</td></tr>
<tr><td>混合配比</td><td><b>2510.26692</b>（3:1 + 消融）、Ling-3.0 模型卡（3:1 / 5:1）、<b>2507.06457</b>（建议 3:1～6:1）</td></tr>
<tr><td>FlashAttention</td><td>arXiv <b>2205.14135</b> + TPU 侧 Splash Attention 实测（<code>tpu/</code> 下多处）</td></tr>
<tr><td><b>MHA 本体</b>（§一）</td><td>Vaswani et al. 2017, arXiv <b>1706.03762</b> ——&nbsp;<b>sec. 3.2 / 3.2.1 / 3.2.2 / 3.2.3</b> ＋ <b>表 1</b>；四条原话见下方折叠</td></tr>
<tr><td><b>KV cache 被点名成瓶颈</b></td><td>Shazeer 2019, arXiv <b>1911.02150</b>（MQA 那篇）——&nbsp;<b>「memory-bandwidth cost of repeatedly loading the large keys and values tensors」</b></td></tr>
<tr><td><b>RNN 一支</b>（§零）</td><td>Elman 1990《Finding Structure in Time》；Bengio, Simard, Frasconi 1994；Hochreiter &amp; Schmidhuber 1997；Cho et al. 2014；Bahdanau et al. 2014, arXiv <b>1409.0473</b></td></tr>
<tr><td><b>只有线性依赖才扫得动</b></td><td>Martin &amp; Cundy 2018, arXiv <b>1709.04057</b>（ICLR'18）——&nbsp;实测最高 9× 加速</td></tr>
<tr><td><b>RNN 在硬件上为什么慢</b></td><td>NVIDIA《Recurrent Layers User's Guide》——&nbsp;<b>「a GEMM with one dimension of one」</b>、<b>「can combine these GEMMs over the minibatch size, but not over different sequence steps」</b></td></tr>
<tr><td>我们自己的 kernel 实战</td><td>Tokamax KDA kernel、<code>tpu/</code> 下 DSA 相关</td></tr>
<tr><td>§2.1 那张 KV cache 对照表</td><td><b>自己按公式推的</b>：<code>2·n_h·d_h·L</code> 与 <code>(d_c+d_h^R)·L</code>，输入全部来自 V3 论文 <b>sec. 4.2</b>。<b>口径（K/V 都按 d_h=128）要在讲的时候声明</b></td></tr>
</tbody></table>

<details class="aside"><summary>🔍 <b>这一讲最吃劲的四条，把原话摆出来</b>
<em>（省得读者去翻论文对措辞）</em></summary>
<table>
<thead><tr><th>这一讲怎么讲的</th><th>论文原话</th></tr></thead><tbody>
<tr><td>「query 问、key 挂牌、value 是货」<b>不是我们编的比喻</b></td>
<td><em>「mapping a <b>query</b> and a set of <b>key-value pairs</b> to an output …
the output is computed as a <b>weighted sum of the values</b>, where the weight
assigned to each value is computed by a <b>compatibility function of the query
with the corresponding key</b>.」</em>（sec. 3.2）</td></tr>
<tr><td>√d_k <b>别只说「防止 softmax 饱和」</b>，那是结论不是理由</td>
<td><em>「for large values of d_k, the dot products <b>grow large in magnitude, pushing
the softmax function into regions where it has extremely small gradients</b>.」</em>
（sec. 3.2.1）<br>⭐ 理由在<b>脚注 4</b>：q、k 各维独立、均值 0、方差 1 时，
q·k <b>均值 0、方差 d_k</b> ——&nbsp;标准差就是 √d_k。</td></tr>
<tr><td>多头<b>不是不够用，是会把该分开的关注平均掉</b></td>
<td><em>「jointly attend to information from different representation subspaces at
different positions. <b>With a single attention head, averaging inhibits this.</b>」</em>
（sec. 3.2.2；h = 8，d_k = d_v = d_model/h = 64）</td></tr>
<tr><td><b>§零 图三 Ⓒ 那一行不是我们的推论</b>，是原文</td>
<td><em>「training these layers is generally fast and simple, due to parallelizability
across the length of the sequence, <b>incremental inference (where such parallelization
is impossible) is often slow, due to the memory-bandwidth cost of repeatedly loading
the large "keys" and "values" tensors</b>.」</em>（Shazeer 2019 摘要）<br>
⭐ 2017 年造出这个形状，<b>2019 年就有人把它命名成问题了</b>；那篇给的解法 MQA
正是本讲模型表的第二行。</td></tr>
</tbody></table>
</details>

<p class="sub">📌 <b>记号约定</b>：<code>§X.Y</code> 是<b>本课</b>的小节号，
<code>sec. X.Y</code> 是<b>被引论文自己</b>的小节号。
⛔ 两者曾经用同一个记号，结果整体重编号时把论文的节号也改了 ——&nbsp;
而且改完恰好落在真实存在的本课小节上，体检全绿。</p>
<hr>
</details>
</div></section>
<!-- ⛔ 2026-09-07 把「还没想清楚的」整块搬进了讲义（topic03-build-lecture.py）。
     ⭐ 判据就是刚立的规矩第 4 条：**讲述性 / 作者视角的内容一律进讲义，教材一个字不留。**
       「要不要把 GQA-8 那一行也画进图里」——&nbsp;学员读到这句什么也学不到，
       只会读到一个信号：**这份材料还没做完。**
     ⛔ 而它又不能删：那三条是真的没想清楚，删了就是假装想清楚了。
       所以是**搬**不是删 —— 搬到只有讲课的人会看的那一份里。 -->'''

out = [head, '''
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic03-build.py 生成。
     **正文写在那个脚本的 BODY 常量里** —— 改内容改那里，别改这个产物，
     也别去改 md 再生成回来（md 只是大纲，2026-09-04 定的）。 -->

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


''', BODY, '''

<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    硬件背景在 <a href="topic-02.html">专题二 · TPU 与 GPU</a>　·
    量化那一支在 <a href="topic-08.html">专题八 · 精度与量化</a>　·
    <a href="topic-03-lecture.html">📝 讲义</a></p>
  <p style="color:var(--gray);font-size:13px">
    本页由 <code>Courses/tools/topic03-build.py</code> 生成 ——&nbsp;
    <b>正文就写在那个脚本的 BODY 里</b>（md 只是大纲）。本目录采用 CC BY-NC-SA 4.0。</p>
</div>

</body></html>''']

# ⭐ 图是外部脚本的产物：**页面里的 SVG 是产物，脚本才是源。**
#   跟专题二那套规矩一致 —— 改图只改 topic03-fig-chronicle.py，然后重跑本脚本。
# ⛔ 这张表原先只有一张图、写死在代码里。加主线图那五张时改成了表驱动 ——
#    否则同一段「读文件 → assert → 拼 figure → 换占位符」要抄六遍，
#    抄的时候漏一处不会报错，只是那张图不见了。
FIGS = {
    # ⛔ 2026-09-07 把这张图的图注**整段删掉了**（现在是空串 ——&nbsp;下面的循环
    #    见到空串就不出 <figcaption>）。现场判断：「这些内容都没有什么用。」
    # ⭐ 它确实已经没用了，而且原因值得记：那段图注还在讲「上半 · 编年史」和
    #    「下半 · 一个循环里都有哪些层」——&nbsp;可**「下半」早就不在这张图里了**，
    #    39 行的模型表已经拆成图下面那张可排序 HTML 表。图注在给一张
    #    **不存在的图**做导读，而且它读起来完全通顺，所以一直没人发现。
    # ⭐⭐ 形状：**图注是图的下游，图改了它不会自己跟着动，也不会报错。**
    #    这跟「教材改了讲义不跟」「aria-label 还在描述被拆走的半张图」是同一个病。
    # 📌 那段图注里唯一有信息量的一句「一格 ＝ 一层 / 整条一色 ＝ 每层同构」
    #    没有丢：它在表头的 hint（topic03-table-models.py）和表下的落点里都有。
    "__FIG_CHRONICLE__": ("fig-chronicle", "fig3-chronicle.svg",
        'topic03-fig-chronicle.py', ''),

    # ── 总纲：这是一个什么故事 ─────────────────────────────────────
    "__FIG_ARC__": ("fig-arc", "fig3-arc.svg", 'topic03-fig-arc.py',
        '⭐ <b>整个专题的骨架。</b>六段，每段只问三件事：图啥、带来了什么、欠下了什么。'
        '<b>底下那两条数是这一讲真正的落点</b> —— 能跑的长度涨了，同一长度下要付的钱降了，'
        '<b>两件事同时发生，才有今天的 agent。</b>'),

    # ── §零 起点：RNN 的三张图 ──────────────────────────────────────
    # ⭐ 三张各答一个问题：怎么算 / 为什么慢 / 三个痛点通向哪。
    #   ⛔ 别合并 —— 第一张是语义、第二张是性能，两种坐标系。
    "__FIG_RNN_UNROLL__": ("fig-rnn-unroll", "fig3-rnn-unroll.svg",
        'topic03-fig-rnn.py',
        '⭐ <b>同一个东西的两种画法。</b>看完只要带走一件事：'
        '<b>竖着的箭头一直可以并行，被卡住的只有横着那一根</b> —— '
        '<b>后面所有「让它变快」的努力，动的都只是那一根。</b>'),

    "__FIG_RNN_HW__": ("fig-rnn-hw", "fig3-rnn-hw.svg",
        'topic03-fig-rnn.py',
        '⭐ <b>这张图要带走的是那个「跟 d 无关」</b>：不管隐藏维是 512 还是 8192，'
        '<b>RNN 每一步的算术强度就等于 batch size</b>。'
        '<b>它意味着「把模型做小」根本救不了 RNN。</b>'),

    "__FIG_RNN_DECODE__": ("fig-rnn-decode", "fig3-rnn-decode.svg",
        'topic03-fig-rnn.py',
        '⭐⭐ <b>本节的落点。</b>Ⓐ 和 Ⓒ 是同一个形状 —— 一步一个 token，每步把权重搬一遍。'
        '<b>区别只在每步还得额外搬什么：RNN 是一个固定大小的状态，'
        'Transformer 是一路线性变长的 KV cache。</b>'
        '<b>整个专题三都发生在 Ⓒ 这一行上。</b>'),

    "__FIG_RNN_PAIN__": ("fig-rnn-pain", "fig3-rnn-pain.svg",
        'topic03-fig-rnn.py',
        '⭐ <b>这一页是整个专题的路标</b>：后面每一个变体都能追回到这三条里的某一条。'
        '<b>最底下那一格是主脊</b> —— 第①条后来被反着又走了一遍，'
        '而「必须线性到能被 scan」就是线性注意力那些公式的由来。'),

    # ── §一 MHA 的三张图 ───────────────────────────────────────────
    "__FIG_MHA_SWAP__": ("fig-mha-swap", "fig3-mha-swap.svg", 'topic03-fig-mha.py',
        '⭐ <b>左边是 §零 那条链，右边是它的替代品。</b>'
        '看完带走一件事：<b>要算的格子从 n 变成了 n²，而那张表就是后面反复出现的「注意力矩阵」本人</b>。'),

    "__FIG_MHA_QKV__": ("fig-mha-qkv", "fig3-mha-qkv.svg", 'topic03-fig-mha.py',
        '⭐ <b>四步里第一步的形状是 [n × n]</b> —— 整层里唯一一个随长度平方长大的东西。'
        '<b>而 K 和 V 要留着给下一个 token 用，留下来的那两份就叫 KV cache。</b>'),

    "__FIG_MHA_HEADS__": ("fig-mha-heads", "fig3-mha-heads.svg", 'topic03-fig-mha.py',
        '⭐ <b>8 × 64 = 512：多头是「切开」不是「加倍」</b>，总维度和计算量都没变。'
        '<b>唯一被乘上去的是 KV</b> —— 下一节起要砍的就是它。'),

    # ── 贯穿全篇的主线图：同一张图画五遍，每次只点亮被改动的那一处 ──────
    # ⭐ 这组图的教学装置在于「五张除了高亮处完全一样」，所以图注也要一致地
    #    提醒读者「这是同一张图」。⛔ 图注里不许出现「上一张 / 下一张」——
    #    这些 SVG 是共用资产，方位词换个文档就指错，而且不报错。
    "__FIG_TX_BASE__": ("fig-tx-base", "fig3-tx-base.svg",
        'topic03-fig-transformer.py',
        '⭐ <b>本专题的主线图</b>：一层 Transformer，每一步的张量形状都标出来。'
        '<b>先只看一件事 —— 在形状里找 <code>S</code>（KV 长度）：全图只有两处带它，'
        'K 和 V 的输出。</b>'
        '那就是唯一需要跨 token 留下来的东西 —— <b>KV cache</b>。'
        '三个旋钮各是一种跟它较劲的方式。'
        '<span class="sub">图式借自 How to Scale Your Model，本图为重画。</span>'),

    "__FIG_TX_K1__": ("fig-tx-k1", "fig3-tx-k1.svg",
        'topic03-fig-transformer.py',
        '⭐ <b>同一张主线图，只点亮旋钮① 动到的地方</b> —— 产生 K 和 V 的那两条支路。'
        '<b>注意那个平方大的矩阵一点没动</b>：这个旋钮治的是显存墙，'
        '<b>治不了 O(N²) 的计算量</b>。'),

    "__FIG_TX_K2__": ("fig-tx-k2", "fig3-tx-k2.svg",
        'topic03-fig-transformer.py',
        '⭐ <b>同一张主线图，只点亮旋钮② 动到的地方</b> —— Q·Kᵀ 和它后面那张 mask。'
        '<b>矩阵的形状一点没变</b>，变的是里面有多少格子真的要算。'
        '⛔ 所以它跟旋钮① <b>正交</b>，两个可以同时上。'),

    "__FIG_TX_K3__": ("fig-tx-k3", "fig3-tx-k3.svg",
        'topic03-fig-transformer.py',
        '⛔ <b>这一张不是高亮，是替换</b>：原来那四格（Q·Kᵀ → mask → softmax → S·V）'
        '整个没了，换成点亮的两格。'
        '⭐⭐ <b>不用听解释，读输出形状就够 —— 点亮那格输出 <code>BKHH</code>，'
        'S 不见了。</b>状态大小只跟头维有关，跟序列多长无关。'),

    "__FIG_TX_FA__": ("fig-tx-fa", "fig3-tx-fa.svg",
        'topic03-fig-transformer.py',
        '⭐ <b>同一张主线图，而点亮的三格跟底图一模一样</b>：同样的算子、同样的形状、'
        '同样的 FLOPs。<b>FlashAttention 不是第四个旋钮</b> —— '
        '三个旋钮改的是这张图，它改的是这张图<b>怎么跑</b>。'),
}

_html = "\n".join(out)

# ⭐ 模型表是 HTML 不是 SVG（表头可排序），所以不走 FIGS 那条 figure 通道。
# ⛔ 硬失败：表缺了宁可构建挂掉，也不要悄悄出一份没有编年史表的教材。
_tbl = os.path.join(HERE, "fig3-models-table.html")
assert os.path.isfile(_tbl), "缺 fig3-models-table.html —— 先跑 `python3 topic03-table-models.py`"
assert "__TABLE_MODELS__" in _html, "正文里没有 __TABLE_MODELS__ 占位符"
_html = _html.replace("__TABLE_MODELS__",
                      '<div class="wrap">%s</div>'
                      % io.open(_tbl, encoding="utf-8").read())

for ph, (fid, fn, src, cap) in FIGS.items():
    fp = os.path.join(HERE, fn)
    # ⛔ 硬失败：图缺了宁可构建挂掉，也不要悄悄出一份少图的教材。
    assert os.path.isfile(fp), "缺 %s —— 先跑 `python3 %s`" % (fn, src)
    assert ph in _html, "正文里没有占位符 %s —— 加图忘了插锚点？" % ph
    svg = io.open(fp, encoding="utf-8").read().strip()
    # ⛔ 图注为空串时**不要出空的 <figcaption>** —— 它不显示文字，但照样吃
    #    figcaption 的 margin/padding，图底下会多出一段说不清来路的空白。
    _html = _html.replace(
        ph, '<figure class="fbox fwide" id="%s">%s%s</figure>'
            % (fid, svg, '<figcaption>%s</figcaption>' % cap if cap else ''))
assert "__FIG_" not in _html, "还有图占位符没被替换掉"
io.open(OUT, "w", encoding="utf-8").write(_html)
print("ok  topic-03.html  %s 字符 · %d 节"
      % (format(os.path.getsize(OUT), ","), len(SECTIONS)))
