# -*- coding: utf-8 -*-
"""专题三 · 注意力演进 —— 教材（HTML 就是源）。

════════════════════════════════════════════════════════════════
⭐⭐⭐ 这一课的第一条规矩：**多画图，少说话**（2026-09-12 立，常驻）
════════════════════════════════════════════════════════════════
现场原话：「你永远都要保持『多画图、少说话』的这个教材。」

**这不是一次性的精简要求，是往后每一次改动都要过的一道闸。**

⭐ 落到笔上就是三个动作：

  ① **写任何一段正文之前，先问：这句话图里有没有？**
     有 —— 就别写第二遍。图注和落点带**是图的一部分**，
     不是「图外面还得再讲一次」的理由。
     ⛔ 2026-09-12 现场实例：我画完注意力流那张图，转手又在图下面
        写了 134 字注解 —— 三句话在图里那两条落点带上**一字不差地都有**。
        整条删掉。**画图的人最容易犯这个错，因为他刚想过一遍。**

  ② **留下来的正文，只写「图画不出来的东西」。**
     图擅长画「多了什么、怎么流、谁比谁大」；
     画不出「少了什么」「为什么当年这么选」「这句话后面还要用」。
     ⛔ 实例：§1.1 那条注解原来前半段在复述图上的 O(n)→O(1)，
        删；只留「**状态也没了**」——&nbsp;那是图画不出来的。

  ③ **h3 标题就是目录，不要再写一段散文版目录。**
     ⛔ 实例：§一 开头原来有 88 字的「这一节要交代三件事，各配一张图……」，
        而下面三个 h3 写的就是这三件事。删，只留落点一句。

📌 体检口径（想量一量的时候用）：
   §一 这一轮从 **1,031 汉字 → 818 汉字**，图 5 张没动 ——&nbsp;
   **砍掉的全是「图已经说过的」，一条信息都没少。**

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
import re
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

/* ── 图下面折起来的「出处与口径」（2026-09-14）────────────────
   ⭐ 收起来的时候只是一行灰字，跟图注拉开一档；展开才是那一堆脚注。
   ⛔ 宽度跟着 figure 走（figure 是 .fwide，最宽 1760px），但**正文限宽在
     1080px** —— 出处是拿来读的散文，不是图，所以这里自己收回版心，
     否则一行会拉到一米八长，眼睛跟不回来。 */
.figsrc{max-width:1080px;margin:6px auto 0;border-top:1px dashed var(--line);
        font-size:13.5px;color:var(--gray)}
/* ⭐ 跟 details.aside 用同一套三角记号，不另发明一种 —— 同一页里两种折叠
   长两个样子，读者要分别学一次。 */
.figsrc>summary{cursor:pointer;list-style:none;padding:8px 2px;
                font:600 12.5px/1.4 var(--mono);color:#9aa0a6;
                letter-spacing:.02em}
.figsrc>summary::-webkit-details-marker{display:none}
.figsrc>summary::before{content:"▸ "}
.figsrc[open]>summary::before{content:"▾ "}
.figsrc>summary:hover{color:var(--ink)}
.figsrc p{margin:0 0 7px;line-height:1.7}
.figsrc p:first-of-type{margin-top:4px}
.figsrc b{color:var(--ink);font-weight:600}
@media print{.figsrc{font-size:11px}}
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
    # ⭐⭐ 2026-09-14 改了 §二 / §三 两个标题。⛔ 一个字的正文都没动 ——
    #   问题从头到尾只在标题上，而这恰恰最难自己发现：
    #     · §二 原叫「一切从长上下文说起」。可它排在 RNN 和 MHA 两段历史**之后**，
    #       「一切从…说起」是**开场白的句式**，读者读到这儿会以为故事重启了一次。
    #       它真正在回答的是**「为什么是现在」**（这个形状 2017 年就有了，
    #       2019 年 MQA 那篇就点过名，全行业动手却是 2024 年以后）——
    #       节内 R20 那段注释早就写明了这一点，只是标题一直没跟上。
    #     · §三（FlashAttention）**2026-09-14 R60 整节删除** ——&nbsp;现场判定
    #       「专题二里边已经讲过了」。⛔ 但它里面有一句是**整篇的转轴**：
    #       3.7 的「『怎么算』到头了，剩下只能改『算什么』」正是三个旋钮的出处。
    #       ⭐ 所以删的是讲解，**那句转轴搬到了 §四 开头**（连同跳号说明）。
    #       ⚠️ 判据：**删一节之前先问它有没有在替别人承重。**
    #         删掉承重句，后面那节会变成「凭空冒出三个旋钮」，而且不报错。
    #   ⛔ §三 空出来的号**不回填也不重编** —— 见下面那条「不动节号」。
    #   ⭐⭐ 判据：**标题要说这一节在故事里干什么，不是说它值不值得读。**
    #     「不展开 X」是写作者的排期，不是读者的路标。
    #   ⛔ 不动节号 —— 上面写了，重编号要连全部 fig 脚本一起改，代价远大于收益。
    ("s零", "零", "起点：RNN —— 被注意力补的那个东西"),
    ("s一", "一", "MHA —— 把循环拿掉，代价是什么"),
    ("s二", "二", "为什么是现在 —— 长上下文的两条独立动机"),
    ("s四", "四", "骨架：三个旋钮，是同一个账本的三个面"),
    ("s五", "五", "旋钮①：让每一份更小"),
    ("s六", "六", "旋钮②：KV 照存，但每步只读一部分"),
    ("s七", "七", "旋钮③：换回一个固定大小的状态 —— 线性注意力"),
    ("s八", "八", "元旋钮 ⊕：混合"),
    ("s九", "九", "代价：没有免费的午餐"),
    ("s十", "十", "落到硬件（本专题的落点）"),
    ("s十一", "十一", "收尾：把谱系放回时间线"),
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
<p>今天你让它做的事完全不一样了：读完整个代码库再改一处 bug、
  连着跑几十轮工具调用、<b>记住整场对话里你反复改过的主意</b>。</p>

<div class="note ok"><p>⭐⭐ <b>上下文长度就是 agent 的工作记忆。</b>
  记不住，就什么都干不成 ——&nbsp;2K 的上下文，连一个文件都读不完。<br>
  ⛔ 而每加长一分上下文，<b>KV cache</b>（模型每吐一个字都要回看前面所有字，于是把每个字算出来的 K、V 存着不重算 ——&nbsp;存下来的这堆就叫它）<b>就线性涨一分</b>。<br>
  ⭐ 所以这六年注意力的全部演化，是为了让「记得住」这件事<b>付得起</b>。</p></div>

<!-- ⭐ 2026-09-12 加这一句。口述一遍之后发现的：
     「只有一个账本」那句原来只出现在**骨架图之后**（作为图的落点），
     可它其实是<strong>导航</strong> —— 台下需要在听任何内容之前就拿到这个框，
     否则前三节会被当成三个并列的技术名词听。
     ⛔ 但不照抄那一段：两处写同一段话必然漂。
        这里只放<strong>承诺</strong>（一句、不带张量形状）；
        骨架图后面那处仍是<strong>兑现</strong>（带 S 在哪、三个面各是什么）。
     ⭐ 判据：**同一句话可以出现两次，前提是两次的职责不同。** -->
<p class="landing">⭐⭐ 先把这一讲的框给你 ——&nbsp;<b>前半程只有一个账本：KV cache。</b>
  <em>后面那三个旋钮，是同一个账本的三个面。</em>
  <br>（三个面分别是什么，看完下面这张骨架图再说。）
  <br><em>⚠️ 说「前半程」是认真的：<b>到旋钮② 这个账本就不够用了</b>
  ——&nbsp;DSA 省的是「读多少」，KV 一个字节都没少。
  <b>从那里起账本分成三样</b>（显存 / 算力 / 访存规整度），
  <a href="#s十">§十</a> 收在那三样上。</em></p>

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
<style>
/* ⭐ 2026-09-12 新造的 .hint —— 判据：**自己新造的类名，落地前先确认它真有样式**。
   放行内 <style> 而不是塞进公共 CSS：它只服务这两道题，没必要全站背着。 */
.guess .hint{display:none;margin-top:14px;padding:11px 16px;border-radius:10px;
  background:#fce8e6;border:1px solid #f3c0bb;color:#a50e0e;font-weight:700;font-size:15px}
.guess .hint.on{display:block}
</style>
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
        MLA ＝ 压缩维 512 ＋ RoPE 64 ＝ 576 →&nbsp;<b>8.6</b>（56.9×）。<br>
        ⭐⭐ <b>488 GiB 到底是多大？</b>换算成机器就有感觉了：按专题二推的
        可分配口径（<b>94.74 GiB／device</b>）算，488 ÷ 94.74 ＝
        <b>5.2 个 device</b>，也就是<b>约 2.6 块 v7 芯片</b>。<br>
        <b>一个用户、一段输入，就要把两块半芯片的 HBM 整个拿来放 KV
        ——&nbsp;而模型权重还一个字节都没放进去。</b><br>
        📌 <b>RoPE</b>（旋转位置编码）后面会反复出现，这里先把它的职责摆正：
        <b>「谁在前谁在后」并不归它管。</b>那件事是<b>因果掩码</b>白送的
        ——&nbsp;每个位置只看得见自己左边，多堆几层就能数出自己前面有几个人
        （<b>NoPE</b>，arXiv 2305.19466，NeurIPS 2023 ——&nbsp;
        直接证明了 decoder-only 不加任何显式位置编码也学得会顺序）。<br>
        ⭐ RoPE 真正加进来的是另一样东西：<b>「我跟它差几格」</b>。
        给每个位置的 Q 和 K <b>按它的位置转一个角度</b>，点积时两个绝对角度相减，
        <b>相对距离就直接出现在打分里</b>——&nbsp;零参数，而且每一层都能用，
        不必靠堆层去数。
        ⭐ 它只改 Q/K，不改 V。<b>后面 §五会讲它给 MLA 惹的麻烦</b>——&nbsp;
        上面那 64 维的成本，买的就是这一件事。</span></p>
      <p>⭐⭐ 这道题真正的题眼在 <b>(c) 和 (d) 的大小关系</b>：
        <span class="qs">MQA 只要 3.8 GiB，比 MLA 的 8.6 还小 2.25 倍。
        <b>MLA 并不是最省的那个。</b><br>
        ⭐ 所以这一支的目标从来不是「谁存得最少」——&nbsp;
        MQA 早在 2019 年就把它压到头了，代价是<b>质量掉得厉害</b>。
        真正要比的是「同样一份字节，换回多少能力」。
        <em>这正是第五节要讲的那条线。</em></span></p>
      <p style="margin-bottom:0"><b>⚠️ 还有一个口径要说清：</b>
        <span class="qs">488 GiB 是「假如 V3 用 MHA」的<b>反事实</b>数字，
        不是 V3 的实测值 ——&nbsp;V3 从第一天就是 MLA。
        而且这里沿用了 V3 论文比较表的口径（K、V 都按 head_dim=128 算）；
        V3 真实的 K 每头是 128+64＝192 维，严格算这个基线还会更大一点。</span></p>
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
        这个数不是本讲新造的，它就是<a href="topic-02.html">专题二</a>整整一节在立的那条屋脊线。</span></p>
      <p><b>⭐ (b) 最容易选反，而选反的人通常是把「快」和「门槛高」搞混了：</b>
        <span class="qs">片上更快，所以<b>分母变大</b> ——&nbsp;
        同一个分子除以更大的分母，<b>商只会更小</b>。
        <b>越靠近计算，这条线越低。</b>
        <em>门槛低意味着：同一个算子挪到片上以后，更容易变成算力受限。</em></span></p>
      <p style="margin-bottom:0">⛔ 而 (b) 为什么只给量级、不给数 ——&nbsp;<b>这才是这道题最想教的一件事</b>：
        <span class="qs"><b>VMEM 的带宽官方没有公开。</b>
        而屋脊点乘以算力就等于带宽 ——&nbsp;给出一个精确的屋脊点，
        等于把那个没公开的数反推出来。所以我们到「几十这一档」为止，<b>不往下猜</b>。<br>
        ⭐ 这正是<a href="topic-02.html">专题二</a>第 6 节那四句问法里的一句：
        先问这个数的出处和口径，再用它。
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
      var hh=box.querySelector('.hint');
      if(hh){ hh.textContent=''; hh.classList.remove('on'); }   /* ⛔ 别忘了连提示一起清 */
      box.scrollIntoView({behavior:'smooth', block:'nearest'});
    });
    rev.insertBefore(rst, rev.firstChild);
    /* ⛔⛔ 2026-09-12 改揭晓规则（现场点的）：
         ① **全对才开**；② 全选了但没全对 → 报「N 对 M 错，继续努力」，不揭晓。
       ⭐ 连带必须改的一处：原来一点就给正确答案加 .right（绿框）——
         那等于**第一次点击就把答案送出去了**，「全对才开」根本无从谈起。
         现在 .right 只在**全对那一刻**才加。
       ⚠️ 判据：**改揭晓条件时要顺着查一遍「还有什么地方也在泄答案」** ——
         按钮配色、aria、title 都算。这里就差点漏掉 .right。 */
    var hint=document.createElement('div');
    hint.className='hint';
    box.insertBefore(hint, rev);
    function grade(){
      var done=true, ok=0;
      groups.forEach(function(gg){
        var p=gg.querySelector('.picked');
        if(!p){ done=false; return; }
        if(p.hasAttribute('data-right')) ok++;
      });
      if(!done){ hint.textContent=''; hint.classList.remove('on'); return; }
      var bad=groups.length-ok;
      if(bad===0){
        hint.textContent=''; hint.classList.remove('on');
        box.querySelectorAll('[data-right]').forEach(function(r){ r.classList.add('right'); });
        rev.classList.add('on');
        rev.scrollIntoView({behavior:'smooth', block:'nearest'});
      }else{
        hint.textContent='\u26a0\ufe0f ' + ok + ' 对 ' + bad + ' 错，继续努力';
        hint.classList.add('on');
        rev.classList.remove('on');
        box.querySelectorAll('.right').forEach(function(r){ r.classList.remove('right'); });
      }
    }
    groups.forEach(function(g){
      g.querySelectorAll('button').forEach(function(b){
        b.addEventListener('click', function(){
          g.querySelectorAll('button').forEach(function(x){x.classList.remove('picked','right');});
          b.classList.add('picked');
          grade();
        });
      });
    });
  });
  }
  if(document.readyState==='loading') addEventListener('DOMContentLoaded', bind);
  else bind();
})();
</script>

__FIG_ARC__

<div class="note info"><p>⭐ 这一讲的前半程<b>只有一个账本：KV cache</b>。
  在张量形状里找 <code>S</code>（KV 长度）——&nbsp;全图只有 K 和 V 两处「留得下来」的带它（注意力矩阵那一处也带 S，但它<b>算完就扔</b>，不进 KV cache），
  那就是唯一需要跨 token 留下来的东西。<br>
  三个旋钮是同一个账本的三个面：<b>① 每份多大　② 每步读多少　③ 干脆别让它变长</b>。
  看到任何一个新名词，先问它在拧哪一面。</p></div>

<p>骨架之后，先看一眼<b>全景与全部证据</b>：这些名词是什么时候、按什么顺序冒出来的，
  以及今天各家<b>实际上是怎么配的</b>、<b>每一家的 KV cache 到底多大</b>。</p>
__FIG_CHRONICLE__
__TABLE_MODELS__
<!-- ⛔ 图上所有型号与配比都是当天现搜的公开信息 —— **这张图会过时**，
     而且过时得比正文快得多。约定：**每次开课前只重跑这一张图的调研，正文不动。** -->
<p class="sub">⭐ <b>这一整块最该带走的一句</b>：<b>MiniMax 一家、三代模型，
  把旋钮③和②各试了一遍，中间还退回过基线</b>（线性 → 退回全注意力 → 稀疏），而且每次转向都公开写了理由
  ——&nbsp;<em>「三个旋钮」这个框架不是我们归纳出来的，是有人真的一个一个试过去了。</em></p>
<hr>
</div></section>
<section id="s零"><div class="wrap"><div class="stn"><span class="badge">第 零 节</span><h2>起点：RNN ——&nbsp;被注意力补的那个东西</h2></div>

<div class="note danger"><p>先给一个反直觉的事实：你现在用的每一个大模型，在往外吐每一个字的时候，
  都退回成了 1990 年那条链的形状。<br>
  Transformer 赢在<b>训练能并行</b>。可生成的时候，它一个 token 一个 token 地走，
  每走一步都要把全部权重从显存里搬一遍 ——&nbsp;<b>这跟 RNN 一模一样</b>。<br>
  ⛔ <b>它没有治好 RNN 的病，它只是把病从训练挪到了推理</b>；而且挪过去之后更重。</p></div>

<!-- ⭐⭐ 2026-09-14 故事线审计 R25。审计第五条：**两个开场钩子讲的不是同一个故事。**
     封面是「2048 → 100 万，512 倍」（讲**容量**），
     §零 开场是「你用的每个模型吐字时都退回成 1990 年那条链」（讲**形状**）。
     ⛔ 两个单看都好，但读者看完封面以为要讲「怎么变长的」，翻页看到的是 RNN。
     ⭐ 修法不是删掉一个，是**把它们接起来**。 -->
<p>⭐ 这一节跟封面那 512 倍是什么关系？——&nbsp;封面问的是「<b>能记多长</b>」，
  这一节问的是「<b>它记东西的形状是什么</b>」。
  <em>两者是同一件事的两面：正因为那个形状会一路变长，「记得更长」才会贵得离谱。
  要涨 512 倍，就得先看清要涨的是什么。</em></p>
<p>所以这一节<b>不是背景介绍，是本专题的舞台说明</b>。<b>四个问题，四张图。</b></p>

<h3>0.1　先把它是什么说清楚：一个固定大小的状态</h3>
<p>序列有先后，所以得有个东西把历史带下去。RNN 的答案是带一个固定大小的状态向量 <code>h</code>
  ——&nbsp;<b>全部设计就这一句</b>。</p>
__FIG_RNN_UNROLL__
<h3>0.2　关键的是哪一列：为什么它在加速器上快不起来</h3>
<p>⛔ 先别去比总计算量。<code>O(n·d²)</code> 和 <code>O(n²·d)</code> 谁大，取决于 <code>n</code> 和 <code>d</code> 谁大；
  Vaswani 原文说的是 <code>n &lt; d</code> 时自注意力更快，而那正是当年的常态。
  <b>固定不变的是另一列 ——&nbsp;串行步数。</b></p>
<table>
<thead><tr><th>层的类型</th><th>每层计算量</th><th><b>串行步数</b></th><th>两个位置之间的最长路径</th></tr></thead><tbody>
<tr><td>自注意力</td><td>O(n² · d)</td><td><b>O(1)</b></td><td><b>O(1)</b></td></tr>
<tr><td>循环（RNN）</td><td>O(n · d²)</td><td><b>O(n)</b></td><td><b>O(n)</b></td></tr>
<tr><td>卷积</td><td>O(k · n · d²)</td><td>O(1)</td><td>O(log_k n)</td></tr>
</tbody></table>
<p class="sub">这张表是 <b>Transformer 作者自己算的</b>（arXiv 1706.03762 表 1）。
  <b>中间那一列就是全部答案。</b></p>
__FIG_RNN_HW__
<div class="note danger"><p>⛔ 两头堵死：batch 是它唯一的算术强度来源，
  而 Vaswani 引言那句原话说的正是另一头 ——&nbsp;<em>「memory constraints limit batching
  across examples」</em>：<b>序列一长，显存就不让你把 batch 开大。</b></p></div>

<h3>0.3　本节高潮：解码时，Transformer 又变回了这个形状</h3>
<p>📌 <b>两个词先说清，后面一直要用</b>：把整段输入<b>一次算完</b>叫
  <b>prefill</b>（预填充）；之后<b>一个一个往外吐</b>叫 <b>decode</b>（解码）。
  <em>⭐ 这一讲后面有<b>三处</b>结论在这两个阶段是<b>相反</b>的 ——&nbsp;
  看到一个「省了多少」，先问它说的是哪个阶段。</em></p>
<div class="note info"><p><b>先把那三处列出来，读到时你会认出它们</b>
  ——&nbsp;<em>（这一栏是<b>路标</b>，现在不用懂，读到那儿回头看一眼就行）</em></p>
<table>
<thead><tr><th>旋钮</th><th>prefill 这边</th><th>decode 那边</th><th>在哪一节</th></tr></thead><tbody>
<tr><td><b>① MLA</b></td><td>⛔ <b>压缩不生效</b>：训练与 prefill 的前向要把 KV 解压出来算</td>
  <td><b>省得最狠</b>：只读那 576 维</td><td><a href="#s五">§五</a></td></tr>
<tr><td><b>② 稀疏</b></td><td>⛔ <b>可能一点不省</b>：要先算出注意力图才知道挑谁
  ——&nbsp;NSA 论文 §2 的第一个坑说的就是这个</td>
  <td>每步只读 k 条</td><td>§6.3b</td></tr>
<tr><td><b>③ 线性</b></td><td>要靠<b>分块</b>才榨得出并行度（块内并行、块间串行）</td>
  <td><b>就是一条纯递推</b>，每步只碰那块固定大小的板子</td>
  <td>§7.4</td></tr>
</tbody></table>
<p>⛔ <b>所以「省了 N 倍」这句话，不带阶段就是半句话。</b>
  <em><a href="#s九">§九</a>那张代价表专门有一列「⭐ 省在哪个阶段」，就是为了逼出这一问。</em></p></div>
__FIG_RNN_DECODE__
<div class="note info"><p>Ⓐ 和 Ⓒ 都是「一步一个，每步搬一遍权重」。
  唯一的区别是每步还得额外搬什么：<br>
  RNN 搬的是一个固定大小的状态；Transformer 搬的是一路线性变长的 KV cache
  ——&nbsp;<b>128K 时它能比权重本身还大</b>（下一节算给你看）。<br>
  ⭐ 后面三个旋钮拧的全是同一件事：让这一行每步要搬的东西变小。</p></div>

<h3>0.4　三个痛点，各自通向哪</h3>
__FIG_RNN_PAIN__
<div class="note warn"><p>⚠️ 一个常见的张冠李戴：「固定长度向量是瓶颈」不是 Sutskever 说的。
  他那篇只是描述做法（映射到「a vector of a fixed dimensionality」）；
  「这是个瓶颈」是 Bahdanau 那篇的原话（arXiv 1409.0473：
  <em>「we conjecture that the use of a fixed-length vector is a bottleneck」</em>）。
  <b>别把后人的批评安到原作者头上。</b></p></div>

<h3>0.5　本节落点</h3>
<div class="note ok"><p><b>这一节要留下的只有一句</b>：
  <b>解码那一行，从 1990 年到今天，形状没变过。</b></p>
<p>变的只是<b>每一步额外要搬的那个东西</b>：RNN 搬一个固定大小的状态，
  Transformer 搬一路变长的 KV cache。
  <em>后面三个旋钮拧的全是它 ——&nbsp;所以这一节不是背景介绍，
  是<b>本专题的舞台说明</b>。</em></p></div>

<details class="aside"><summary>📌 这一节的出处清单（全部一手核过）</summary>
<ul>
<li><b>Elman 1990</b>《Finding Structure in Time》——&nbsp;context units「copied … on a
  one-for-one basis, with fixed weight of 1.0」。</li>
<li><b>Bengio, Simard, Frasconi 1994</b>——&nbsp;梯度消失；<b>Hochreiter &amp; Schmidhuber 1997</b>——&nbsp;LSTM。</li>
<li><b>Bahdanau et al. 2014</b>（arXiv 1409.0473）——&nbsp;「fixed-length vector is a bottleneck」。</li>
<li><b>Vaswani et al. 2017</b>（arXiv 1706.03762）——&nbsp;引言「This <b>inherently sequential</b>
  nature precludes parallelization within training examples…」＋ 表 1 三列。</li>
<li><b><a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-recurrent/" target="_blank" rel="noopener">NVIDIA《Recurrent Layers User's Guide》</a></b>——&nbsp;「a GEMM with <b>one dimension of one</b>」；
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

<!-- ⛔ 2026-09-12 二轮学生审稿：这里原来有个「0.5 于是下一节」小节，
     它那句「2017 年有人问：既然这个附件这么好使……」跟 §一 的开篇**是同一句**，
     只把「附件」换成了「补丁」。⭐ 而「补丁」在 §一 里又被用于第二个意思
     （多头是「平均带来损失」的补丁）—— 同一个词在一讲里两个所指。
     → 整段删掉（§一 的标题和首句已经接住了），「补丁」只保留一个所指。 -->
<hr>
</div></section>

<section id="s一"><div class="wrap"><div class="stn"><span class="badge">第 一 节</span><h2>MHA ——&nbsp;把循环拿掉，代价是什么</h2></div>

<p>RNN 疼在三处：算不快、记不住、装不下。注意力最早只解决了第三条，而且是作为 RNN 的一个附件出现的。于是 2017 年有人问：既然这个附件这么好使，<b>能不能把 RNN 整个扔掉，只留附件？</b></p>
<!-- ⛔ 2026-09-12 砍掉原来那段 88 字的「这一节要交代三件事，各配一张图……」：
     **下面三个 h3 标题本身就是目录**，那段只是它的散文版重复。
     只留落点一句 —— 落点是标题给不了的。 -->
<div class="note ok"><p><b>这一节要回答的就一件事：KV cache 是从哪儿来的</b>
  ——&nbsp;<a href="#s零">§零</a>那张「解码时又变回 RNN」图上 Ⓒ 那一行的病根，在这一节。</p></div>

<!-- ⭐⭐ 2026-09-13 夜间 R7。现场定的新方针：
     「讲注意力的重点是**这个东西是怎么思考的、怎么被发明出来的、为什么可行**。」
     ⛔ 这一节原来直接从「2017 年有人问」开始 —— 那是**结果**，不是**过程**。
     ⭐ 补一张 1.0：注意力不是谁一次想出来的，是被三次「这儿不对」推出来的；
       而且第二刀（加性 → 点积）是这门课「硬件反过来决定公式」的第一个例子，
       还是论文原话。 -->
<h3>1.0　它是怎么被发明出来的</h3>
__FIG_ATTN_INVENTED__
<p>⭐ <b>读论文的姿势</b>：
  <b>看一个机制，先找它在修上一版的哪一个具体毛病。</b>
  <em>「有人灵光一闪设计了注意力」什么也教不了你；
  「一个固定长度向量装不下长句子，于是让解码器自己去源句里软性地找」
  ——&nbsp;<b>这个你下次能照着用。</b></em></p>

<h3>1.1　先看它换掉了什么</h3>
__FIG_MHA_SWAP__
<!-- ⛔ 2026-09-12：前半句（O(n)→O(1)、格子 n→n²）图上就写着，删。
     留下的这句**图画不出来**：图能画「多了什么」，画不出「少了什么」。 -->
<div class="note danger"><p>⛔ 被换掉的不只是那根箭头 ——&nbsp;<b>状态也没了</b>。
  RNN 那个固定大小的 <code>h</code>，被换成了「把所有历史原封不动留着」。
  <em>这一句后面每一节都要用到。</em></p></div>

<!-- ⭐⭐ 2026-09-12 加 1.1b。现场要求：「借机先把 Transformer 讲了 ——
     不用讲 MLP，就讲 attention；讲 Attention is all you need 到底为啥；
     讲清楚信息到底怎么汇到最后一个 token 上去。**以画图为主，别写太多字。**」
     ⭐ 位置是刻意的：**直觉放在机械之前**。
       1.1 说「换成了什么结构」，1.1b 说「信息怎么流」，1.2 才说「形状和公式」。
       原来 1.1 之后直接进 1.2，读者是先看到四步和 √d、再自己去悟为什么。
     ⛔ 正文只留三句 —— 这一节的讲述全在图里。 -->
<h3>1.1b　那它到底怎么把信息传过去的</h3>
<div class="note info"><p>📌 <b>先认一个后面要用两次的单位：困惑度（perplexity）</b>。
  它衡量「模型对下一个词有多没把握」——&nbsp;<b>越低越好</b>。
  粗略地讲，困惑度 30 就是「大约在 30 个词之间犹豫」。</p>
<p>⚠️ <b>它只在同一个基准、同一份数据上可比</b> ——&nbsp;
  所以后面 <a href="#s五">§五</a>那张表的 30 上下，和 <a href="#s六">§六</a>那张表的 5 上下，
  <b>不能互相比</b>，只能各自看组内差多少。</p></div>
__FIG_MHA_FLOW__
<!-- ⛔⛔ 2026-09-12：这里原来有一条 134 字的注解，写着「一层注意力 ＝ 全员报价
     按需取货」「每个位置同时在做」「那个标题最常被读反」——&nbsp;
     **而这三句，在图里那两条落点带上一字不差地都有。**
     ⭐ 是我画完图之后又把图里的话抄了一遍。整条删掉。
     ⭐⭐ 判据（本讲常驻规矩，见文件头）：
       **写正文之前先问一句「这句话图里有没有」。有，就别写第二遍。**
       图注和落点带是图的一部分，不是「图外面还得再说一次」的理由。 -->
<h3>1.2　拆开看：一层里到底在算什么</h3>
__FIG_MHA_QKV__
<div class="note warn"><p>⚠️ 「query / key / value」不是我们编的比喻，是论文自己的措辞：
  <em>「mapping a query and a set of key-value pairs to an output … the output is a
  weighted sum of the values」</em>。<br>
  <!-- ⛔ 2026-09-12 按「多画图少说话」砍：原来这里把图里蓝框那段推导
       又复述了一遍（方差随维度线性长大 / softmax 只认绝对数值）。删。
       只留一句指路 ＋ 一句**结论的重新命名** —— 后者图里没有，而且一说就记住。 -->
  那个 <code>√d_k</code> 的<b>推导链在图里的蓝框</b>（论文脚注 4，两行）。<br>
  <b>说白了：它就是一个标准差。</b>
  <em>除掉它只做一件事 ——&nbsp;把打分的方差拉回 1。</em></p></div>

<!-- ⭐⭐ 2026-09-13 夜间 R8。两个「为什么不那样做」——
     这一讲讲了 softmax 怎么用，**从没讲为什么非它不可**；
     也讲了 q/k/v 三件套，**没讲为什么 k 和 v 不能是同一个**。
     ⭐⭐ 第二格是意外收获：它一句话串起了 §六 ——
       DSA 的索引器敢换 ReLU，是因为**它只排序、不加权平均**。 -->
__FIG_WHY_SOFTMAX__
<h3>1.2b　那为什么非得是 softmax，k 和 v 又为什么不能是同一个</h3>
<p>⭐ 这条判据的<b>通用形式</b>：
  看到一个设计的副作用，先回头看它的优点是靠哪条性质换来的 ——&nbsp;十有八九是同一条。
  <em>softmax 那条「一行加起来等于 1」，在这里让加权平均的尺度稳定（优点），
  到 §六 就是 attention sink 的成因（副作用）。
  而 softmax-off-by-one 这个补丁，本质就是把这一条放松掉。</em></p>

<h3>1.3　多头在多什么</h3>
__FIG_MHA_HEADS__
<!-- ⛔⛔ 2026-09-12：这里原来那条注解，跟图里那条落点带**连标题都一样**
     （「多头的代价，正好是本专题的题眼」），三句话在图上一句不少。删。
     ⭐ 这是本讲**第二次**抓到同型重复（上一次是 1.1b 那 134 字）——&nbsp;
       判据见文件头第一条：**写正文前先问「这句话图里有没有」。** -->

<div class="note info"><p>📌 <b>先回答一个很自然的问题：存不下，为什么不能每步重算？</b></p>
<p>因为重算的代价不是「再算一遍」，是<b>每一步都把整段历史重算一遍</b>。
  生成第 <code>n</code> 个 token 时，如果不存 K/V，就得拿前面 <code>n−1</code> 个 token
  重新过一遍投影 ——&nbsp;而下一步又要重来。
  整段生成的总开销<b>从正比于 N² 变成正比于 N³</b>。</p>
<p>⭐ 所以这不是「省一点」的优化，是<b>能不能用</b>的分界。
  <em>KV cache 是拿显存换时间，而这一讲全部的账，
  都是在算这笔交换到底有多贵。</em></p></div>

<!-- ⭐⭐⭐ 2026-09-13 夜间 R14。六份调研报告一致指到同一个空白：
     讲义讲完「八乘六十四等于五百一十二」，台下立刻会想的那个问题
     ——&nbsp;**128 个数凭什么记得住上万个 token** ——&nbsp;全网没人回答。
     ⭐ 而它一旦回答了，能同时接住本讲三处：为什么不是 8 维、
       为什么 MLA 压到 512 不死、为什么长上下文会越读越糊。 -->
<h3>1.3b　顺便回答一个立刻会想到的问题：128 个数，凭什么记得住上万个 token</h3>

<p>上一段说多头买到的是<b>分辨率</b>：一个头只管 128 维。
  那问题就很自然了 ——&nbsp;128 个数，<b>怎么可能</b>记得住一整篇文章里上万个词的区别？</p>

__FIG_CAPACITY__

<div class="note ok"><p>★ <b>答案分两层，而第二层才是有用的那层</b></p>
<p><b>第一层是硬上限</b>：要求「谁跟谁都不沾边」（两两垂直），
  128 维里最多只能立 <b>128 根</b> 杆子 ——&nbsp;第 129 根一定能被前面那些拼出来。
  <em>如果模型真按这个标准办事，一个头一辈子只认得 128 个概念。</em></p>
<p><b>第二层是模型实际在做的</b>：把标准放宽成「<b>差不多</b>不沾边」。
  图②是当场算的 ——&nbsp;128 维里随便丢 <b>一万根</b>，最挤的那一对仍然差着 <b>61°</b>
  才会重合。<b>根数涨了 78 倍，最挤的那一对才挤了 9°。</b></p>
<p>⭐ 所以容量不是 128，是几万。<em>这笔便宜不是白占的 ——&nbsp;
  「差不多」的意思就是「有点像」，而有点像会漏票。</em></p></div>

<p><b>漏票有多少？图③把它算成了两个数</b>，这两个数放在一起才是重点：</p>

<ol><li><b>单个干扰项小到画不出来</b> ——&nbsp;只有正主的 <code>0.0117%</code>。
  完全可以忽略。</li>
<li><b>但「可以忽略」不能乘以一万。</b>一万个加起来是正主的 1.17 倍，
  <b>正主只剩 46%</b>。</li></ol>

<div class="note warn"><p>⚠️ <b>这就是「记混」，它不是模型偷懒，是 softmax 的直接后果</b></p>
<p>生活里的同一件事：<em>一个人在台下小声嘀咕，你听不见；
  一万个人同时小声嘀咕，台上的人就喊不过了。</em></p>
<p>⛔ 而长上下文是<b>两头夹击</b>：嘀咕的人变多（图③那根轴），
  <b>而且最吵的那一个还离得更近了</b>（图②那根轴，夹角在变小）。
  <em>两根轴同时往坏的方向走 ——&nbsp;这就是为什么上下文一长，模型就开始「串台」。</em></p></div>

<div class="note info"><p>📌 <b>这一小节后面要用三次，先在这里打个招呼</b></p>
<p>① <b>为什么每个头是 128 维，不是 8 维</b> ——&nbsp;容量不是线性的。
  8 维里丢 1000 根，最挤的一对只剩 12°，那两根基本就是同一根。</p>
<p>② <b>为什么 MLA 敢压到 512 维</b>（§五）——&nbsp;因为它要的从来不是「完全正交」。
  <em>「赌的那段」赌的就是这件事：压完之后大家<b>还够不像</b>。</em></p>
<p>③ <b>为什么稀疏注意力不只是省钱</b>（§六）——&nbsp;如果记混是干扰项太多造成的，
  那<b>少看几个反而可能更准</b>。<em>降噪是它的正面效果，不是省钱的副作用。</em></p></div>

<h3>1.4　本节压轴：把形状标出来（这张图后面一直在用）</h3>
<p>前面讲的是想法。<b>标上形状之后，这笔账可以用眼睛读</b>
  ——&nbsp;<em>在形状里找那个会越变越长的维度就行。</em></p>
__FIG_TX_BASE__
<!-- ⭐⭐⭐ 2026-09-13 现场拍板「要搬」：主线图右边那条注解栏**整段搬到这里**。
     现场判据：「图是图，字是字。」——&nbsp;十几行解释本来就是正文的活。
     ⭐ 搬过来还白捡三样：**能搜索、能复制、手机上会自动折行** ——&nbsp;
       这三样在 SVG 里一样都没有，而且在那儿还得用 11px 去换。
     ⛔ 改图的时候不要把这几段又塞回 SVG。 -->
<div class="note info"><p>★ <b>先只看一件事：图上什么东西需要留到下一个 token</b></p>
<p>在形状里找 <code>S</code>（KV 长度）——&nbsp;只有两处「留得下来」的带它：
  <b>K 的输出 BSKH</b> 和 <b>V 的输出 BSKH</b>。
  S 是唯一会随对话越变越长的那一维。</p>
<p>⚠️ 图上带 <code>S</code> 的形状不止两处（K/V 的进出、两个 matmul 的操作数都带）——&nbsp;
  <b>「带 S」和「要跨 token 留下来」是两件事</b>：
  中间那些带 S 的算完就扔（FlashAttention 连物化都不物化）。
  <em>→ 唯一要跨 token 留下来的，是 K 和 V 的输出。</em></p>
<!-- ⭐⭐⭐ 2026-09-14 故事线审计 R24。审计发现的第三条：
     **主角一直在场，却从没被命名。**
     §1.4「在形状里找那个 S」· §四「三个旋钮拧的全是它」·
     §七「第一次让 S 从形状里消失」—— 这条线**是完整的**，已经埋了三处，
     ⛔ 但从没被明说成一条线，读者要自己把三处连起来才看得见。
     ⭐ 这里给它命名，§七 点它退场，§11.1 回收。 -->
<p>⭐⭐ <b>给它起个名字吧 ——&nbsp;这一讲的主角就是这个 <code>S</code>。</b></p>
<p><em>后面每一节，你其实只要盯着它一件事 ——&nbsp;这一招，
  是让 S 前面的<b>系数变小</b>（旋钮①）、让每步<b>读到的 S 变少</b>（旋钮②），
  还是干脆让 S <b>从形状里消失</b>（旋钮③）？</em></p>
<p>⭐ <b>这三问就是全课的骨架。</b>
  <em>名词有几十个，但它们全都只在回答这三问中的一个。</em></p>
<p>⭐⭐ 整个专题三，就是在跟这一份 KV cache 较劲：
  <b>①</b> 让每一份更小（改产生 K/V 的那两条支路）·
  <b>②</b> KV 照存但每步只读一部分（改 mask 那一格）·
  <b>③</b> 换成一个固定大小的状态（换一套数学，<code>S</code> 直接消失）。</p></div>

<details class="aside"><summary>📐 <b>把字母换成数字 ——&nbsp;那份要留下来的到底有多大</b>
<em>（这笔账后面四张图一直挂着；想自己核的人点开）</em></summary>
<p>取序列 128K、BF16、batch 1，把带 <code>S</code> 的那两处换成实际占多少：</p>
<table>
<thead><tr><th>算到哪一步</th><th>多大</th><th>说明</th></tr></thead><tbody>
<tr><td>每层 K ＋ V</td><td><b>8 GiB</b></td><td>2 × S × K头 × H × 2 B（MHA，128 头）</td></tr>
<tr><td>× 61 层</td><td><b>488 GiB</b></td><td>一个用户、一段输入</td></tr>
<tr><td>对照：整个模型的权重</td><td><b>625 GiB</b></td><td>671B，原生 FP8，1 B/参数</td></tr>
<tr><td>⭐ 于是</td><td><b>78%</b></td><td>单用户就占 78%；<b>两个并发，KV 就超过权重本身</b></td></tr>
</tbody></table>
<div class="note danger"><p>⚠️ <b>这一栏在训练里是空的。</b>
  K/V 是算完就扔的激活，不跨 step 留 ——&nbsp;
  全讲所有倍数都只对推理成立。</p></div>
<div class="note ok"><p>⭐⭐ 权重是所有人共享的一份，KV cache 是每人一份 ——&nbsp;
  所以它决定的不是「装不装得下」，是<b>能同时服务多少人</b>。</p></div>
</details>

<h3>1.5　本节落点</h3>
<!-- ⛔ 2026-09-12：Shazeer 那段引文与「2019 年就被点名」这句，
     图里第二条落点带已经完整写着（含英文原文）。删，只留下面那句指路。 -->
<div class="note ok"><p><b>这一节要留下的只有一句</b>：
  <b>KV cache 就是在这里出生的。</b></p>
<p>把循环换成一张 n×n 的表，换来了训练能并行；
  <em>代价是解码时每一步都要把<b>之前所有 token 的 K 和 V</b> 重新读一遍 ——&nbsp;
  而这份东西<b>会随对话一直长下去</b>。</em></p>
<p>⭐ <b>下一节把这笔账算成具体的字节数</b>，再看它为什么必须省。</p></div>
<hr>
</div></section>
<section id="s二"><div class="wrap"><div class="stn"><span class="badge">第 二 节</span><h2>为什么是现在 —— 长上下文的两条独立动机</h2></div>
<!-- ⛔⛔ 2026-09-12：2.1 / 2.2 / 2.3 原来是三段纯散文（合计 887 汉字、**0 张图**），
     是全讲最严重的一处「说话多、画图少」。
     ⭐ 而它们全都是**结构**不是叙述 —— 一条推导阶梯、四个对照、三个观察、
       一个「缺一不可」的交汇。**结构就该画。**
     现在整节压成：两句引子 ＋ 一张图 ＋ 两句图给不了的落点。
     ⛔ 改这一段时切过一次头：按「2.1 的 h3 到下一个 h3」切，
       **把夹在中间的 §三 节标题一起切掉了**（页面上第三节整个消失，构建不报错）。
       ⭐ 判据：**切之前先看这中间还夹着什么** —— 这次改成按行号切，
         并对首尾两行做断言。 -->
<!-- ⭐⭐⭐ 2026-09-13 夜间 R20。这一节讲「两条动机」，但少了它前面那一问：
     **为什么是现在？** 这个形状 2017 年就造出来了，2019 年 MQA 那篇就指着它
     说是问题，可全行业动手改是 2024 年以后 —— 中间那几年技术一个字没变。
     ⭐ 装置偷自知乎 姜富春：**同一个模型、只把 batch 和上下文拧一下，
       瓶颈就从「参数」换成了「KV cache」。** 本图换成本讲一直在用的 V3 口径重算。 -->
<h3>2.0　先回答一个更前面的问题：为什么是现在</h3>

<p>Transformer 是 <b>2017</b> 年的东西。KV cache 会涨这件事，
  <b>2019 年 MQA 那篇论文的摘要里就写着</b>。
  <em>那为什么全行业真正动手改注意力，是 2024 年以后？</em></p>

__FIG_FLIP__

<div class="note ok"><p>★ <b>因为变的不是技术，是工作负载</b></p>
<p>同一个模型（V3），<b>权重那一段一个字节没动</b>，都是 625 GiB。
  只把两个东西拧了一下 ——&nbsp;上下文从 4K 到 128K，同时服务的人从 1 个到 64 个：</p>
<p>KV cache 从 <code>0.27 GiB</code> 涨到 <code>549 GiB</code>，
  占总量从 <b>0.04%</b> 变成 <b>46.8%</b>。<em>主角换人了。</em></p>
<p>⭐⭐ 而这两个数<b>是相乘的，而且乘的是同一项</b>
  ——&nbsp;这就是为什么它不是慢慢变严重，而是突然变成了首要问题。</p></div>

<h3>2.1 ＋ 2.2 ＋ 2.3　两条线，一个交汇处</h3>
<p><b>这两条必须分开讲。</b>它们指向同一批技术，<b>但出发点完全不同</b>
  ——&nbsp;<em>混在一起讲，就成了名词罗列。</em></p>
__FIG_MOTIVES__

<!-- ⭐⭐⭐ 2026-09-13 夜间 R9。上面那张给的是**直觉版**的信息账
     （实测稀疏、远近有别）。这一张给**严格版**：
       · 两点互信息随距离幂律衰减（Lin ＆ Tegmark：马尔可夫是指数衰减）
       · 双部互信息随长度幂律**增长**
       · L2M 条件：状态维度必须至少同阶增长
     ⭐⭐ 它顺手回答了一个后面才会问的问题：**为什么第三个旋钮必须混着用。**
       那不是工程经验，是有理论下界的。 -->
__FIG_INFO_LAW__
<h3>2.3b　严格版本：两半之间的互信息</h3>
<p>⭐ 还要补一句改口：
  <b>「远处可以少看」不等于「远处不重要」。</b>
  <em>两半之间的互信息是<b>随长度增长</b>的 ——&nbsp;只是增长得慢。
  所以正确的说法是：<b>Transformer 那条线性增长的 KV 是「供给过量」，
  而这条幂律是「实际需求」</b>；
  前两个旋钮做的事，是在不掉到需求线以下的前提下<b>把过量的常数压小</b>。</em></p>

<div class="note danger"><p>⛔ <b>图上那 488 GiB 是「假如 V3 用 MHA」的反事实值</b>，
  不是 V3 的实测 ——&nbsp;<b>V3 从第一天就是 MLA</b>。
  <em>这门课自己的规矩：推出来的数必须带推导链和口径，图里两样都写了。</em></p></div>
<p>⭐ 还有一句图上画不出来、后面却要反复用：
  <b>KV cache 不是显存里的一项开销，它直接决定你能同时服务多少人</b>
  ——&nbsp;<em>权重所有人共享一份，KV cache 每人一份。这一条到<b>专题六</b>
  会变成 batch size 的硬上限。</em></p>
</div></section>
<section id="s四"><div class="wrap"><div class="stn"><span class="badge">第 四 节</span><h2>骨架：三个旋钮，是同一个账本的三个面</h2></div>
<!-- ⭐⭐⭐ 2026-09-14 R60。§三（FlashAttention）整节删除，现场判定
     「专题二里边已经讲过了」——&nbsp;确实如此：专题二 §三 整节就是拿
     FlashAttention 当主角走完的（怎么融、算术强度、屋脊线）。
     ⛔ 但 §三 的 3.7 是**整篇的转轴**：三个旋钮成立的前提就是
       「『怎么算』这条路已经到头」。删掉它，下面就变成凭空冒出三个旋钮。
     ⭐ 所以这里只留**那句转轴 ＋ 一个指针**，不重讲任何机制。 -->
<div class="note ok"><p>⭐⭐ <b>先接上一句话，它是下面整套骨架的前提。</b></p>
<p><b>「怎么算」这条路，已经走到头了。</b>FlashAttention 那一套 ——&nbsp;
  数学一个字不改，只把中间产物留在片上、一块一块地算 ——&nbsp;
  今天<b>是地板不是选项</b>，而且已经调到头了。
  <em>它怎么做到的、块大小怎么选、为什么融合完实测还是只跑到三成多，
  <a href="topic-02.html">专题二</a>整整一节就是拿它当主角走完的，
  <b>本讲不重复</b>。</em></p>
<p>⭐ <b>所以剩下的空间不在「怎么算」里，只能去改「算什么」——&nbsp;
  下面这三个旋钮，就是「改算什么」的全部可能位置。</b></p>
<p style="margin-top:6px"><em>📌 <b>节号从 §二 直接跳到 §四</b>：原 §三 讲的就是
  FlashAttention，整节已移交<a href="topic-02.html">专题二</a>。
  ⛔ 空出来的号不回填 ——&nbsp;回填要连全部图脚本里的「§X.Y」一起改，
  代价远大于收益。</em></p></div>
<h3>4.1　三个旋钮</h3>
<p>这是这个专题的骨架。把所有名词收进一张表：</p>
__FIG_KNOBS__
<!-- ⭐⭐⭐ 2026-09-13 夜间 R10。前九轮里「事后压 vs 从头按压缩训」这条对立
     **出现了四次**，每次在不同的技术分支上。到第四次它就不是巧合了 ——
     它是一个**跟三个旋钮正交的轴**，该被扶正到骨架里。
       · 三个旋钮回答「**改什么**」
       · 这条暗线回答「**什么时候改**」
     ⚠️ 图里带了一条必须讲的口径：事后/native 说的是**这次被怎么用**，
       不是方法本身的属性（GQA 两列都待过）。 -->
__FIG_WHEN_AXIS__
<div class="note warn"><p>⚠️ 这里一定会被问：「那 KV 量化算不算第四个旋钮？」
  ——&nbsp;<b>不算</b> —— 它跟这三步正交。
  <em>三个旋钮管的是<b>存几个数、读几个数</b>；量化管的是<b>每个数用几个 bit</b>。
  两者可以任意组合（DeepSeek-V4 就是稀疏 ＋ KV 混合精度一起上）。
  <b>精度那一维整个归<a href="专题08-低精度.md">专题八</a>。</b></em></p>
<p>📌 所以「没有第四个位置」这句话的准确版本是：
  在「<b>要不要留、留多少、读多少</b>」这件事上没有第四个位置；
  <b>「每个数多大」是另一根轴。</b></p></div>
<!-- ⛔⛔ 2026-09-12 二轮学生审稿，这一块挨了三刀：
     ① 标题写「第二个轴」，可底下那张四行表讲的全是**第一个轴**（三个旋钮）；
     ② 「①和②可以叠加…③是换赛道」两句与 fig-knobs 的落点带**逐字相同**；
     ③ 真正致命的：这条「轴」在**旋钮③ 上一行都没有**，也没承认它不适用 ——
        四行里四行都在 ①②。⭐ 一个只覆盖三分之二的东西，不能叫「轴」。
     → 表挪回 §四 主干（它本来就是名词收纳表）；重复的两句删；
       「第二个轴」降级成「一条在 ①② 上反复出现的判据」，
       并**把旋钮③ 那一格的空白如实标出来**。 -->
<table>
<thead><tr><th>旋钮</th><th>在改什么</th><th>代表</th></tr></thead><tbody>
<tr><td><b>① 每个 token 存多少</b></td><td>减少 KV 的<b>份数</b>或<b>维度</b></td><td>MQA → GQA → <b>MLA</b> → Gated MLA</td></tr>
<tr><td><b>② 每个 query 看多少</b></td><td>限制<b>范围</b>或<b>动态挑选</b></td><td><b>SWA</b> · NSA · <b>DSA</b> · <b>CSA / HCA</b>（⚠️ CSA 同时也在拧 ①，见 §6.4c）</td></tr>
<tr><td><b>③ 换一套数学</b></td><td>用<b>固定大小的状态</b>代替不断变长的 KV</td><td>线性注意力：DeltaNet → <b>GDN</b> → <b>KDA</b></td></tr>
<tr><td><b>①+②+③ 混着来</b></td><td>不同层用不同方案</td><td><b>Hybrid</b>：V4 的 CSA+HCA、K3 的 KDA+Gated MLA</td></tr>
</tbody></table>

<div class="note danger"><p>⛔⛔ <b>账本在这里交棒。</b></p>
<p>开场那句「只有一个账本：KV cache」到这里为止<b>还成立</b>，
  但<b>拧下去就不成立了</b>：</p>
<ul>
<li><b>旋钮①</b> 确实在改这个账本 ——&nbsp;每份更小，488 直接变小。</li>
<li><b>旋钮②</b> <b>一个字节都不省</b>。DSA 的 KV 全都存着，只是不读
  （<a href="#s九">§九</a> 那张表里这一格写的就是「—」不是「↓」）。
  它改的是<b>算力和访存</b>。</li>
<li><b>旋钮③</b> 把这个账本<b>整个作废</b> ——&nbsp;S 从张量形状里消失了。</li>
</ul>
<p>⭐ <b>所以「一个账本」是一个极好的开场钩子，但它不是全讲那句话。</b>
  <em>它的职责是 §零–§五 的<b>记账装置</b>：让你在最初那段有个具体的东西可以盯。
  从这一节起，账分成三样 ——&nbsp;<b>显存、算力、访存规整度</b>，
  <a href="#s十">§十</a> 收在那三样上，那才是全讲唯一的落点。</em></p>
<p>⚠️ <b>这一句不能跳过</b>：带着「省 KV」这一个念头读完 §六 和 §七，
  到 §十 会发现对不上 ——&nbsp;而那时候很容易以为是自己没看懂。</p></div>

<h3>4.2b　一条反复出现的判据：事后压，还是从头按压缩训</h3>
<p>⭐ 这张地图的用法：拿到一个新名字，先把它放进某一格。
  <em>放得进去的，它的优点和代价你已经知道了，不用细看；
  <b>放不进去的，才值得你花时间</b> ——&nbsp;那才是真正的新东西。</em></p>
<div class="note warn"><p>⚠️ 这一条曾经被本讲写成「第二个轴」，现在降级了 ——&nbsp;
  因为它撑不起「轴」这个词。
  <em>上面那张图四行全落在<b>旋钮①②</b>上；<b>旋钮③ 一行都没有</b>。</em></p>
<p>📌 旋钮③ 那一格为什么空着，值得如实说：本讲那张 44 行表里，
  所有线性／混合模型都是 <b>native</b> 的（从第一天就按这个结构训）。
  「把一个训好的 Transformer <b>事后</b>线性化」是有人在做的一支，
  但本讲没有核过它的数，所以这一格留空 ——&nbsp;
  <em>⭐ 留空比硬凑一行好：<b>空格是可以被后来的人填上的，凑出来的行只会被当成事实背下去。</b></em></p>
<p>⭐ 所以它的准确身份是：<b>一条在旋钮①②上反复出现四次的判据</b>，
  而不是跟三个旋钮正交的第二根轴。<em>作为判据它很好用，作为轴是虚的。</em></p></div>

<!-- ⛔ 2026-09-12：这里原来是「为什么恰好是三个」那段散文（三步的 <pre> 清单 ＋
     「没有第四个位置」＋ FlashAttention 的例外）。**上面那张图一条不少地画了**，
     而且图能表达「三步与三个旋钮一一正对」这个关系 —— 文字表达不了。整段删。
     ⭐ 判据见文件头第一条：写正文前先问「这句话图里有没有」。 -->
<h3>4.3　一张名词收纳表</h3>
<p>这张表可以边读边填 ——&nbsp;每读完一个支线回来补一格：</p>
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
<tr><td><b>CSA / HCA</b></td><td>DeepSeek-V4, arXiv 2606.19348</td><td><b>①（token 维）＋②</b></td><td>先把 KV 按块压缩（＝沿 token 维压，见 §6.4c），再稀疏挑选；两档压缩率混排</td></tr>
<tr><td>DeltaNet</td><td>起源 Schlag 等 arXiv 2102.11174；可并行化 arXiv 2406.06484</td><td>③</td><td>状态更新用 delta rule：擦掉旧的再写新的</td></tr>
<tr><td><b>GDN</b>（Gated DeltaNet）</td><td>arXiv 2412.06464</td><td>③</td><td>在 delta rule 上加遗忘门</td></tr>
<tr><td><b>KDA</b></td><td>Kimi Linear, arXiv 2510.26692</td><td>③</td><td>遗忘门从标量升级成 <b>per-channel</b> 向量</td></tr>
<tr><td>FlashAttention</td><td>arXiv 2205.14135</td><td><b>不是旋钮</b></td><td>数学一个字不改，只改访存顺序</td></tr>
<tr><td><b>RoPE</b></td><td>RoFormer, arXiv 2104.09864</td><td><b>不是旋钮</b></td><td>把位置变成转角，让打分直接带上「差几格」。<b>它管距离，不管谁在前</b>；给 MLA 惹出 decoupled 那一路（§5.3）</td></tr>
<tr><td><b>NoPE</b></td><td>arXiv 2305.19466（NeurIPS 2023）</td><td><b>不是旋钮</b></td><td>decoder-only 靠因果掩码就能学会顺序 ——&nbsp;显式位置编码不是必需品（§8.3 混合架构因此能整层不放 RoPE）</td></tr>
</tbody></table>
<hr>
</div></section>
<section id="s五"><div class="wrap"><div class="stn"><span class="badge">第 五 节</span><h2>旋钮①：让每一份更小</h2></div>
__FIG_TX_K1__
<!-- ⭐⭐⭐ 2026-09-13 现场拍板「要搬」：主线图右边那条注解栏**整段搬到这里**。
     现场判据：「图是图，字是字。」——&nbsp;十几行解释本来就是正文的活。
     ⭐ 搬过来还白捡三样：**能搜索、能复制、手机上会自动折行** ——&nbsp;
       这三样在 SVG 里一样都没有，而且在那儿还得用 11px 去换。
     ⛔ 改图的时候不要把这几段又塞回 SVG。 -->
<div class="note info"><p>① <b>三种改法，都只改这两格</b></p>
<table>
<thead><tr><th>改法</th><th>动了什么</th></tr></thead><tbody>
<tr><td><b>MQA</b></td><td>K 从多个头砍到 1 个（K＝1）</td></tr>
<tr><td><b>GQA</b></td><td>砍到几个，多个 query 头共用一个 KV 头（G＝N∕K）</td></tr>
<tr><td><b>MLA</b></td><td>不砍头，改成先压到一个低秩的 <code>c</code>，用时再上投影</td></tr>
</tbody></table>
<p>⭐ 看形状就知道省在哪：KV cache 存的是 <code>BSKH</code>，
  里面那个 <code>K</code> 变小，缓存就等比变小 ——&nbsp;就这么直接。</p></div>
<div class="note warn"><p>⚠️ <b>它不省核心那两个 matmul 的 FLOPs。</b>
  Q·Kᵀ 出来的还是 <code>BTSKG</code>，该算的乘加一次不少 ——&nbsp;
  GQA 靠的是把 KV 头广播开再算。</p>
<p>⭐ 但<b>训练侧它省</b>：W_K / W_V 跟着缩 ——&nbsp;
  参数、优化器状态、投影的 FLOPs 三样都省。</p>
<p>📌 MLA 那条另有一处麻烦：RoPE 必须单独走一路，
  因为上投影吸收不了带位置旋转的那几维（见 <a href="#s五">§五</a>）。</p></div>
<!-- ⛔⛔ 2026-09-12 按「多画图少说话」重排 §五。原来 5.1–5.4 是四段散文（824 汉字）。
     ⭐ 新图 fig3-knob1 吸收了其中三块：
       · 四种存法的对照（含「MQA 比 MLA 还小」那个题眼）→ 左栏
       · 5.3 那三步代数（能吸收 / 被 R 夹住 / 拆两路）→ 右栏。
         这一块尤其该画：讲义写着它「最容易讲糊」，
         而它本质是「一个重写成不成立」—— 图能直接把「成不成立」摆出来。
       · MLA 的代价、训练／推理口径、Gated MLA 不省显存 → 落点带
     正文只留**图给不了的三样**：GQA 那个「连续旋钮」的方法论、
     MLA 的超参出处、以及那条留给专题六的硬件伏笔。
     ⛔ 按行号切并对首尾断言 —— 上一轮按内容切，切掉过 §三 的节标题。 -->
<!-- ⭐⭐⭐ 2026-09-13 新增。本课原来对 RoPE 只有「寄快递」一个比喻，
     而那个比喻回答的是「为什么 MLA 吸收不了它」——
     ⛔ 「把位置变成转角，为什么点积就自动带上了相对距离」
       这个最基础的问题，我们一张图都没有。 -->
__FIG_ROPE__
<!-- ⭐⭐⭐ 2026-09-14 R59 加。§五 手上已经有三张讲这四种存法的图
     （knob1 散点 / copy-matrix 矩阵 / absorb 代数搬家）——&nbsp;
     ⛔ 但它们**都从第二步起讲**，各自都默认读者已经知道「一份」是什么、
       谁在跟谁共用、以及 MLA 存的根本不是 K/V。
     ⭐ 这张补的是第一步：一张零公式零数字的接线图，只回答「省在哪」。
       判据在最下面那排行李箱 —— 算力四家几乎一样，
       差别全在「跨 token 得一直留着多少」。 -->
__FIG_WIRING__

__FIG_KNOB1__
<!-- ⭐⭐ 2026-09-13 夜间 R1 加。现场原话：
     「像 MLA 这种东西，它为什么能压缩？这里边跟信息论有关的东西，对吧？」
     ⛔ 原来这一节只回答了「MLA 是什么」和「省多少」，**从没回答「凭什么」**。
     ⭐ 判据：**一个数字告诉你结果，一条推理链才留得下来。**
       56.9× 这个数学生记不住；「哪一段白送、哪一段是赌的」这个问法会跟他一辈子。
     图放在四种存法对照**之后** —— 先知道它做了什么，再问它凭什么。 -->
__FIG_MLA_WHY__
<!-- ⭐⭐ 2026-09-13 夜间 R2。上一张答「MLA 凭什么敢压」，
     这一张答**它的对手为什么更窄反而更差**，外加一个意外收获：
     ⭐ MQA 论文 §2.4 自己就列了一张「怎么少搬 K/V」的选项单，
       今天三个旋钮里的两个都在那张单子上 ——
       **这门课的骨架不是我们事后归纳的，是当事人自己列的。** -->
__FIG_MQA_WHY__
<h3>5.0b　那为什么更窄的 MQA 反而输了</h3>
<p>⭐ 图里那三行困惑度值得单独记一下：<b>MQA 和「真的只剩一个头」缓存的 K/V 一样多</b>，
  差别只在 query 侧还留不留 8 个不同的问法 ——&nbsp;就这一点差了整整 1.0。
  <em>所以后面看任何一个「压缩」方案，都要分清它压掉的是<b>体积</b>还是<b>自由度</b>：
  <b>压体积通常还好，压自由度很贵。</b></em></p>

<!-- ⭐⭐⭐ 2026-09-13 新增。「白送的那段 vs 赌的那段」原来只是**嘴上说的**。
     planetbanatt 用 Manim 把左边那栏（秩够 → 拆开再复原、数字一模一样）画出来了，
     ⛔ 但**右边那栏全网没人画**。这张图两栏都画，都用真数字，让读者自己核。 -->
__FIG_LOWRANK__

<h3>5.0c　为什么它能压 —— 一笔信息账</h3>
<p>⭐ <b>这条推理链本身是可迁移的</b>。
  <em>看任何一个压缩方案 ——&nbsp;KV、权重、激活、梯度 ——&nbsp;都先把它拆成两段问：
  <b>哪一段是「表示冗余」白送的，哪一段是「赌它低秩／稀疏／可近似」赌来的？</b>
  白送的那段不用做实验，赌的那段必须看掉点。</em></p>

<h3>5.1 ＋ 5.2 ＋ 5.3 ＋ 5.4　从砍头到压缩</h3>
<p>⭐ 这里真正值得单独说一句的，是「GQA 是一个连续旋钮」这件事本身
  ——&nbsp;它不是一个新机制，是把 MHA 和 MQA 之间的空白填上，
  让你可以按需要选一个点。
  <em>这门课后面会反复见到这个套路：<b>把一个二选一变成一个可调的连续量。</b></em></p>
<p>⭐⭐ 而这个旋钮之所以是<b>连续</b>的，理由比「取个中间值」深一层 ——&nbsp;
  <b>MHA、MQA、GQA 之间差的，只是同一个位置上同一块矩阵里写了什么</b>。
  把这块矩阵一格一格画出来，MLA 也会自己落到同一条轴上。</p>
__FIG_COPY_MATRIX__
<div class="note ok"><p>⭐ <b>这张图要破的是一个很顺口、但区分不了事情的说法：</b>
  「MLA 就是给 KV 做低秩分解」。</p>
<p>低秩这个描述<b>没错</b>，但它<b>区分不了 GQA 和 MLA</b> ——&nbsp;
  把 GQA 所有的 K、V 叠在一起，<b>GQA 本身就是一次低秩投影</b>。
  苏剑林的原话是：「<em>笔者认为低秩投影这个角度并不贴近本质……
  MLA的本质改进不是低秩投影，而是低秩投影之后的工作。</em>」</p>
<p>⭐⭐ <b>低秩之后那一步，才是分界线。</b>
  GQA 用「分割 ＋ 复制」把 c 凑成各头要的 K、V；
  而分割和复制本身就是线性变换 ——&nbsp;
  它们对应的就是图里那块只有 8 个 1、其余全是 0、而且不可训练的矩阵。
  <b>MLA 做的事，就是把这块写死的矩阵松开，让它学。</b></p></div>
<div class="note warn"><p>⚠️ <b>引这一段时，两处归属要说准。</b></p>
<p>① 苏剑林写的是「<b>MLA 被视为 GQA 的一般化</b>」，
  <b>不是</b>「MHA / MQA / GQA 都是 MLA 的特例」 ——&nbsp;
  后一句是网上转述时放大出来的，别挂在他名下。</p>
<p>② 他也<b>没有否认</b> MLA 是低秩分解（他在另一篇里还写过
  「从 MHA 的角度看，MLA 是给 K、V 加了 rank=512 的 LoRA」）。
  准确的说法是上面那句：<b>低秩这个描述没错，只是它区分不了这两个</b>。</p>
<p>③ 他给的条件 <code>d_c = g(d_k+d_v) &lt; d</code> 在 <b>MHA 那一端会失效</b> ——&nbsp;
  g 取到 h 时 d_c 就等于 K、V 本身的总宽，<b>根本没压</b>，
  图里那块矩阵也退化成单位阵。<em>所以「同一条轴」说的是那块矩阵的位置与形状，
  不是说四个成员互为特例。</em></p></div>
<p>MLA 的超参出自 <b>V3 论文 §4.2</b>：
  <code>n_h=128, d_h=128, d_c=512, d_h^R=64, 61 层</code>
  ——&nbsp;<em>图上那 576 就是 512 ＋ 64。</em></p>
<div class="note danger"><p>⛔⛔ <b>MLA 最大的<u>推理</u>部署坑：它在张量并行下会退化（⭐ 2026-09-13 补）</b></p>
<p>⚠️ <b>先把作用域说死：下面讲的<u>只是推理</u>。</b><em>训练侧没有这个问题 ——&nbsp;训练<b>没有 cache 要复制</b>，W<sup>UK</sup>/W<sup>UV</sup> 都是按头切的，TP 照切；被复制的只有那个 576 维的<b>隐向量激活</b>，跟 7168 维的 hidden 比可以忽略。</em></p>
<p>GQA / MHA 的 KV 是<b>按头切</b>的，TP=8 就每张卡各存八分之一，天然可分。
  <b>而 MLA 的 KV 是一份 576 维的隐向量，根本不按头分</b> ——&nbsp;于是只剩三条路：</p>
<ul>
<li><b>每张卡各存一份</b>（复制）——&nbsp;TP=8 就等于把省下来的 57 倍<b>当场还回去 8 倍</b>。</li>
<li><b>attention 那段改成数据并行</b>（DP attention，SGLang 给 DeepSeek 就是这么做的）
  ——&nbsp;代价是 attention 和 MoE 两段的并行策略<b>不一致</b>，中间得插 all-gather。</li>
<li><b>只切 query 头</b>，KV 仍然复制 ——&nbsp;省了计算，没省显存。</li>
</ul>
<p>⭐ 这条值得单独记，因为它是本讲主线的又一个例子：
  一个在单卡上很漂亮的数学结构（把 K、V 合进一个共享隐向量），
  到了多卡上恰恰因为「不按头分」而失去了最自然的切法。
  <em>⛔ 所以引用「MLA 省 57 倍」的时候，要带一句「在什么并行配置下」。</em></p>
<p>📌 口径：三条做法是公开实现里能看到的；各自的具体开销本课没有实测。⚠️ 再强调一次：这三条<b>全是推理侧</b>的。</p></div>

<!-- ⭐⭐⭐ 2026-09-13 夜间 R15。这一节里「吸收」这个词出现了四次，
     **一次也没说它是什么** —— 最接近的一处是 fig3-knob1 里一句括号注
     「（吸收进 q 那一侧）」，对没学过的人等于没说。而讲义自己写着
     这一块「最容易讲糊」。⛔ 原来这里那条 warn 直接把它推给了专题六，
     可它是 MLA **能不能省**的前提，推掉之后这一节就缺了一环。
     ⭐ 它其实是本讲已经立起来的装置第三次出场：把括号挪个位置。 -->
<h3>5.4b　「吸收」到底是什么 ——&nbsp;这是 MLA 能省下来的前提</h3>

<!-- ⛔ 这段原来还带着「如果真要拆……论文原话 must recompute the keys」三句，
     被「图前预告过长」的 lint 抓到 —— 而那三句图里 ① 格一字不差地画着。
     ⭐ 判据（文件头第一条）：**写正文前先问「这句话图里有没有」。** -->
<p>前面一直在说 MLA 把 K、V 压成一个 576 维的隐向量存起来。
  那生成下一个词的时候，不是还得把它们<b>拆回来</b>才能比对吗？</p>

__FIG_ABSORB__

<div class="note ok"><p>★ <b>不用拆。把括号挪一下就行</b></p>
<p><code>qᵀ (W<sup>UK</sup> c) ＝ (W<sup>UK</sup>ᵀ q)ᵀ c</code>
  ——&nbsp;<b>同一个乘法，只是括号换了个位置。</b></p>
<p>左边括号在 <code>c</code> 那侧：缓存里有几个 <code>c</code> 就得算几次。
  右边括号在 <code>q</code> 那侧：<b>一步只有一个 q，所以只算一次</b>，
  而缓存里的压缩包<b>一个都不用拆</b>。</p>
<p>🏠 <b>生活版就一句话</b>：<em>与其把一万本外文书全翻译过来，
  不如把你的搜索词翻译过去。</em></p>
<p>⭐ V 那一侧同理 ——&nbsp;<code>W<sup>UV</sup></code> 可以吸进输出投影
  <code>W<sup>O</sup></code>。所以 K 和 V 两边都不用拆。</p></div>

<div class="note warn"><p>⚠️ <b>省多少？算出来的答案跟直觉不一样</b></p>
<p>直觉会说「解压从 S 次变成 1 次，所以省 S 倍」。<b>不对</b>
  ——&nbsp;吸收之后每个 token 的点积从 128 维变成了 512 维，
  <b>这一头贵了 4 倍</b>，在把省下的吃回去。</p>
<p>两笔加起来算（图③），省的倍数<b>有个上限，而这个上限正好是每头维度
  <code>d_h = 128</code></b>：解压一个 token 要 <code>d_h×d_c</code> 次乘加，
  而点积只要 <code>d_c</code> 次，两者的比就是 <code>d_h</code>。
  <em>S 再长也过不去这个数。</em></p>
<p>📌 口径：这笔账是本课自己算的，<b>只数乘加，没算访存</b>
  ——&nbsp;真机上访存往往才是瓶颈，所以它是个下界不是实测。</p></div>

<div class="note danger"><p>⛔ <b>两条定律，一条允许、一条禁止 ——&nbsp;MLA 最难的两件事都在这儿</b></p>
<p><b>结合律允许你挪括号。</b>上面那一步靠的就是它。
  <em>论文原话：<code>due to the associative law of matrix multiplication, we
  can absorb W<sup>UK</sup> into W<sup>UQ</sup>, and W<sup>UV</sup> into
  W<sup>O</sup></code>。</em></p>
<p><b>但交换律不成立。</b>RoPE 会往 <code>q</code> 和 <code>W<sup>UK</sup></code>
  中间塞进一个跟位置有关的旋转矩阵，而<b>夹在中间的东西挪不出去</b>。
  <em>原话：<code>a RoPE matrix … will lie between W<sup>Q</sup> and
  W<sup>UK</sup> and matrix multiplication does not obey a commutative
  law</code>。</em></p>
<!-- ⭐⭐⭐ 2026-09-14 R44 补的一层。上面那两段是**论文自己的说法**，没错，
     但「交换律不成立」是结论不是原因 —— 因为吸收这一步本来就只用结合律。
     真正预乘不出来的是 R 带着的那个下标。推导链写在 fig3-knob1 的脚本头部。 -->
<p>⭐⭐ <b>再往下追一层：卡住的不是「中间有东西」，是那东西<u>带下标</u>。</b>
  假如塞在中间的是一块<b>固定</b>的矩阵 <code>M</code>，那
  <code>W<sup>UQ</sup>ᵀ M W<sup>UK</sup></code> 照样能预乘成<b>一个</b>矩阵，
  吸收完全成立 ——&nbsp;<b>这一步只用到结合律，压根没要交换律。</b></p>
<p>可 RoPE 是<b>相对</b>的：<code>R<sub>t</sub>ᵀR<sub>j</sub> ＝
  R<sub>j−t</sub></code> ——&nbsp;<b>每一对 (query, key) 对应一个不同的矩阵</b>。
  要预乘就得预乘出<b>一整套</b>，有多少种相对距离就有多少个。
  <b>这才是真做不到的那一步。</b></p>
<p>⭐ 而 5.3 那张图画的就是解法：<b>把带位置的那一小块单独拎出来走 64 维一路</b>
  ——&nbsp;<b>宽的那条轨上没有闸，紫块照旧搬得走。</b></p></div>

<div class="note info"><p>📌 <b>同一个把戏，本讲这是第三次出场</b></p>
<p><b>§七 线性注意力</b>：把括号从 <code>(QKᵀ)V</code> 挪成
  <code>Q(KᵀV)</code> ——&nbsp;那个句长×句长的大方块就不用建了。<br>
  <b>§五 MLA 吸收</b>（这一小节）：把括号从 <code>qᵀ(W<sup>UK</sup>c)</code>
  挪成 <code>(W<sup>UK</sup>ᵀq)ᵀc</code> ——&nbsp;压缩包就不用拆了。</p>
<p>⭐ 两次是同一个数学恒等式，而且都被同一类东西挡过：
  §七被因果 mask 挡住，这里被 RoPE 挡住。
  <em><b>挡住结合律的，永远是「中间被塞了个东西」。</b></em></p></div>

<div class="note warn"><p>⚠️ <b>作用域</b>：吸收<b>只在 decode 用得上</b>
  ——&nbsp;prefill 时一批里有很多个 <code>q</code>，「只变换一次」这个便宜就没了
  （这正是本节开头那张表里「压缩不生效」那一行的意思）。</p>
<p>⭐ 于是同一个数学式子有了两种算法实现，<b>选哪种取决于是 prefill 还是 decode</b>
  ——&nbsp;怎么在一个引擎里同时装下两套，留到<a href="专题06-推理.md">专题六</a>。</p></div>

<!-- ⭐⭐⭐ 2026-09-14 R46 新增。§五 到这里为止**全是「存多少」的算账** ——
     4.571 白送、12.4 赌出来、56.9×、576 对 4096、一张卡换 64 个人。
     ⛔ 而学生一定会问的那个问题「压这么狠会不会变笨」，本讲**一次实测都没给**。
     这一小节补的就是那一维，而且它的答案反过来打了 §5.3 一巴掌。
     ⚠️ 位置必须在 5.4b 之后 —— 它讲的是 5.3 那条窄轨的**再评价**，
       读者得先知道那条窄轨是被什么逼出来的。 -->
<h3>5.4c　那条「无奈」的窄轨，可能才是 MLA 好的原因</h3>

<p>§五 到这儿把「<b>存多少</b>」算干净了。但有个问题一直悬着 ——
  <b>压这么狠，模型会不会变笨？</b>
  前面每一笔账数的都是字节，<b>质量那一维一次实测都没出现过</b>。</p>

__FIG_MLA_CREDIT__

<div class="note danger"><p>⛔ <b>这组数反转了本讲自己刚说过的一句话</b></p>
<p>§5.3 讲到 RoPE 那条 64 维窄轨时，本课用的词是「<b>被赶到一条窄轨上</b>」
  ——&nbsp;把它当成被代数逼出来的<b>妥协</b>。
  可这组消融指的是反方向：同样 512 的 KV Cache，光把 256 拆成 192＋64、
  只给那 64 维加 RoPE，loss 就从 2.720 掉到 2.711。</p>
<p>⭐ <b>那可能不是妥协的代价，而是这个设计顺手做对的一件事。</b>
  <em>原作者的说法是「看似无奈的设计，极有可能是它效果优异的关键原因」。</em></p></div>

<div class="note info"><p>📌 <b>三条猜测，图里只画了两条</b></p>
<p>原文一共提了三个可能的功臣：<b>head_dims</b>、<b>Partial RoPE</b>、
  <b>KV-Shared</b>（K 和 V 共享大部分维度）。前两条图里都有受控对照，
  <b>第三条本课只提，不画、不给数</b> ——&nbsp;它要跟 RoPE 兼容得额外引入
  一套新的位置编码，实验设计绕得多，而原文自己对它的措辞也最保守：
  <em>「应该也有一定作用」</em>。</p></div>

<div class="note q"><p>❓ <b>顺手想一步</b>：如果主因真是 head_dims，
  那想让普通 GQA 追平 MLA，<b>该从哪个数字改起、改到多少</b>？</p>
<p>原文给的答复很具体：<em>「head_dims 应该要 192 起步了，并辅以 Partial RoPE」</em>
  ——&nbsp;<b>注意这是一条能直接写进配置文件的结论</b>，
  而它是从上面那两级台阶读出来的，不是从哪篇论文的摘要抄的。</p></div>
<hr>
</div></section>
<section id="s六"><div class="wrap"><div class="stn"><span class="badge">第 六 节</span><h2>旋钮②：KV 照存，但每步只读一部分</h2></div>
__FIG_TX_K2__
<!-- ⭐⭐⭐ 2026-09-13 现场拍板「要搬」：主线图右边那条注解栏**整段搬到这里**。
     现场判据：「图是图，字是字。」——&nbsp;十几行解释本来就是正文的活。
     ⭐ 搬过来还白捡三样：**能搜索、能复制、手机上会自动折行** ——&nbsp;
       这三样在 SVG 里一样都没有，而且在那儿还得用 11px 去换。
     ⛔ 改图的时候不要把这几段又塞回 SVG。 -->
<div class="note info"><p>② <b>稀疏，本质上就是换一张 mask</b></p>
<p>标准因果注意力的 mask 是一个下三角 ——&nbsp;看全部历史。
  ⭐ 所谓稀疏，就是把这张 mask 换成别的形状：</p>
<table>
<thead><tr><th>方案</th><th>换成什么形状</th></tr></thead><tbody>
<tr><td><b>SWA</b></td><td>只留主对角线附近一条带 ——&nbsp;只看最近 W 个</td></tr>
<tr><td><b>NSA</b></td><td>三条路并存：压缩看全局 ＋ top-k 挑重点 ＋ 滑窗看近处</td></tr>
<tr><td><b>DSA</b></td><td>拿一个轻量索引器先打分，只留 top-k 那几块</td></tr>
<tr><td><b>CSA</b></td><td>先把每 4 个 token 压成 1 个 entry，在压缩后的格上挑</td></tr>
</tbody></table>
<p>⭐ <b>它跟旋钮① 是正交的</b>：一个改 KV 存多少，一个改这张 mask ——&nbsp;
  所以两个可以同时上（GLM-5 就是）。</p></div>
<div class="note warn"><p>⚠️ <b>纸面省下的 FLOPs，要 kernel 跟上了才算数。</b>
  不规则的 mask 对硬件不友好 ——&nbsp;这是这一支真正的门槛，
  §六 那几家推理期选择器栽的就是这一跤。</p></div>
<!-- ⛔⛔ 2026-09-12 按「多画图少说话」重排 §六。原来 6.1–6.6 是六段散文
     ＋ 三张表 ＋ 一幅 ASCII 画，合计 2,065 汉字，是全讲最长的一节。
     ⭐⭐ 这一节自己写着一句话，直接论证了该怎么改：
       「attention sink 那个 bug **从公式上完全看不出来，
         只有把注意力矩阵画出来看才发现**。」——&nbsp;那就把五种读法全画成 mask。
     新图 fig3-knob2 吸收了：五个方案的读取形状、三条路的共同结构
     （原来是 <pre> 里的 ASCII 画 ——&nbsp;**ASCII 画本来就是「想画图但手边
     只有文本」的产物**）、attention sink 的完整故事、以及 2% 那笔账。
     正文只留**图给不了的三样**：native 那张对照表（它是表，表就该是表）、
     DSA 的打分式子、以及 6.5b 索引本身成了开销那一段。
     ⛔ 按行号切 ＋ 首尾断言。 -->
__FIG_KNOB2__
<h3>6.1 ～ 6.6　从「砍成一条带」到「先压再挑」</h3>
<!-- ⭐⭐ 2026-09-13 夜间 R6。滑窗这一支原来只在 mask 图里占一格，
     而它其实藏着这一讲**最好讲的一个故事**：
       凭什么敢砍（层数是免费的射程）→ 砍了为什么会崩（5158 对 5.40）
       → 真正的原因（softmax 要求一行加起来等于 1）。
     ⭐ 「把那四个 token 换成换行符照样管用」是判决性实验 ——
       它一句话证明了起作用的是**位置**不是语义。 -->
__FIG_SWA_WHY__
<h4>6.1b　滑窗：凭什么敢砍，砍了为什么会崩</h4>
<!-- ⛔ 2026-09-12：这里原来把 fig-swa-why 落点带的两句（「归一化约束逼出来的」
     ＋「量化里那批 outlier」）**逐字复述**了一遍。图上有，正文就不写。 -->
<p>⭐ 一条<b>课程内的接线</b>：
  图上那句「量化里那批 outlier 跟这是同一件事」不是顺口一提 ——&nbsp;
  <b>那批 outlier 在 <a href="topic-08.html">专题八</a> 有专门一节</b>。
  <em>到那儿你会再遇到同一个形状：一个「毫无道理却极其稳定」的现象，
  背后是一条你没注意到的守恒约束。<b>同一个形状分在两讲里各讲一遍，
  比在一讲里说两遍有用。</b></em></p>

<p>⭐ 图上五张 mask 从左到右，就是这一支的演进：
  <b>砍成一条带（SWA）→ 补回停车位（sink）→ 学着挑（NSA / DSA）→ 先压再挑（CSA/HCA）</b>。</p>

<!-- ⭐⭐⭐ 2026-09-13 夜间 R18。调研 agent 把这一处点成第一名，理由很硬：
     **这是全课唯一一个能把「抽象危害」变成看得见的东西的装置。**
     ⛔ 上面那张图只讲了 sink 的**成因**（不许弃权 → 造一个弃权用的候选人），
       **从没讲它有什么用** —— 于是读者会以为它纯粹是个病灶。
     ⭐ 真相反过来：**那张弃权票不是浪费掉的，它是刹车片。** -->
<h4>6.1c　那张弃权票有什么用 ——&nbsp;它是刹车片</h4>

<p>上面说了 sink 是<b>怎么来的</b>。但还有半句没说：
  <b>它有什么用？</b>——&nbsp;<em>如果它纯粹是个副作用，
  StreamingLLM 特意把它留着就只是在打补丁。而事实不是这样。</em></p>

__FIG_OVERMIX__

<div class="note ok"><p>★ <b>机制是两半，第二半最常被漏掉</b></p>
<p>① <b>吸得多</b> ——&nbsp;它把绝大部分注意力吸到自己身上
  （Llama 405B 里将近 <b>80% 的注意力</b>落在第一个 token 上），别人分到的就少了。<br>
  ② <b>吐得少</b> ——&nbsp;⭐ <b>而它的 value 几乎是零。</b>
  所以吸走的那一大块<b>不带任何内容回来</b>。</p>
<p>⛔ <b>只讲第一半是讲不通的</b>：如果它吸走的注意力照样带内容回来，
  那信息一样会混，只是换了条路。<em>论文管这个叫 <code>approximate no-op</code>
  ——&nbsp;一个「几乎什么都不做」的去处。</em></p></div>

<div class="note info"><p>📌 <b>于是前面那句口径要补一半</b></p>
<p>「softmax 不许弃权，模型就自己造了一个弃权用的候选人」——&nbsp;<b>那只说到成因</b>。
  补上的这一半是：<b>那张弃权票不是浪费掉的，它是刹车片。</b></p>
<p>⭐ 所以 StreamingLLM 为什么砍掉开头几个 token 模型就崩
  ——&nbsp;<b>不是丢了信息，是刹车没了。</b>
  <em>（那几个 token 本来就没什么内容，这正是它们能当 sink 的原因。）</em></p></div>

<div class="note danger"><p>⛔ <b>两个 80%，长得一样，意思完全不同</b></p>
<p>「Llama 405B 里将近 <b>80% 的注意力</b>落在第一个 token 上」——&nbsp;这是<b>权重占比</b>。<br>
  「LLaMa 3.1 405B 里有 <b>80% 的注意力头</b>形成了强 sink」——&nbsp;这是<b>头的比例</b>
  （判据是阈值 ε=0.8）。</p>
<p>⭐ 两句都出自同一篇论文，数字一样、含义毫不相干。
  <em>引的时候说串了，懂行的人一听就知道。</em></p></div>

<!-- ⭐⭐ 2026-09-13 夜间 R3。现场原话：「DSA 他又是怎么想的？
     为什么选 top 2048 就足够？」
     ⛔ 这一节原来只说了 DSA「学着挑」，**没说它凭什么挑得准**。
     ⭐ 拆成三个问题是这一张图的全部价值：
       「能不能少看」是**经验事实**（H2O 量出来的 95% 稀疏）；
       「怎么知道该看谁」才是 DSA 解决的那个，而且是个**鸡生蛋**；
       「那个便宜的复制品凭什么算得动」是工程。
     ⛔ 2048 这个数**论文没给消融** —— 图里如实写了能说和不能说的。 -->
__FIG_DSA_WHY__
<!-- ⭐⭐⭐ 2026-09-13 新增。本课原来只讲了**一种**破法（DSA 的师徒），
     而实际上有三条真正不同的路，第三条（CSA 的降维打击）正好是 V4 相对
     V3.2 的新东西 —— 我们现在把 CSA 讲成「压缩的两个方向」，
     ⛔ 没把它跟鸡生蛋挂上钩，可 CSA 最聪明的地方恰恰在这里。 -->
__FIG_CHICKEN__

<h4>6.2b　「怎么知道该看谁」是个鸡生蛋 —— DSA 让真注意力当老师</h4>
<p>⭐ 这条套路值得起个名字：<b>要省掉一个贵的东西，先让它自己说出答案，
  再训一个便宜的去复制那个答案。</b>
  <em>DSA 的索引器就是这么来的 ——&nbsp;它<b>不是一个猜谁重要的启发式</b>，
  它是主注意力分布的一个廉价复制品（跨头求和、L1 归一、KL 对齐）。
  这个套路在推理优化里到处都是，值得单独记住。</em></p>

<!-- ⭐⭐ 2026-09-13 夜间 R4。上一张答 DSA 怎么挑，这一张答
     **NSA 为什么长成三条路** —— 答案是「被四个坑逼出来的」，
     而那四个坑是论文 §2 自己列的，不是我们归纳的。
     ⭐ 这一张顺带把本讲「落到硬件」那条主线提前落地了一次：
       **计算稀疏 ≠ 访存稀疏** —— 按块选、组内共享，都是为访存不是为精度。
     ⚠️ 图里如实并列了两处打架的稀疏度口径（95% vs top20%→70%）。 -->
__FIG_NSA_WHY__
<h4>6.3b　NSA 的三条路，是被四个坑逼出来的</h4>
<p>⭐ 这条判据的<b>用法</b>：
  <b>看任何一篇讲稀疏的文章，先问「它省的是 FLOPs 还是字节」。</b>
  <em>省 FLOPs 谁都会 ——&nbsp;在纸上少算 90% 的格子而已；
  但只要那些格子散落在显存各处、或者同组的头各挑各的，<b>要搬的字节一点没少</b>。
  NSA 的两个看起来很朴素的决定（按块选、组内共享），<b>都是为访存，不是为精度</b>。</em></p>

<h4>6.3c　唯一必须单独拎出来的一张表：「训练期稀疏」和「推理期稀疏」不是一回事</h4>
<p>NSA 的「<b>native</b>」指的就是<b>训练时就这么做</b>，不是训练完再加的推理优化。
  <em>这个区别值得单独说三十秒 ——&nbsp;它解释了后面一个反复出现的现象。</em></p>
<table>
<thead><tr><th></th><th>推理期稀疏</th><th>训练期稀疏（native）</th></tr></thead><tbody>
<tr><td>模型知不知道自己会被稀疏</td><td>不知道</td><td><b>知道</b>，权重是在稀疏条件下学出来的</td></tr>
<tr><td>掉点</td><td>有，且难预测</td><td><b>小得多</b>，甚至能反超</td></tr>
<tr><td>能不能省训练成本</td><td>不能</td><td><b>能</b>（NSA 报的反向 6.0×）</td></tr>
<tr><td>代价</td><td>无，随时可开关</td><td><b>要重训</b>，没法给已有模型打补丁</td></tr>
</tbody></table>
<div class="note ok"><p>⭐ 这张表解释了为什么这些新注意力方案总是跟新模型一起发布，
  而不是作为一个推理框架的开关。</p></div>

<!-- ⭐⭐ 2026-09-13 夜间 R5。现场问：「CSA 又为什么把四个头压成一个？
     HCA 又图啥？那它俩之间为什么穿插着摆？」
     ⛔ 第一件事是把问题本身校正：**CSA 压的不是头，是 token。**
     ⭐⭐ 而这个口误恰恰指向这一张图最该留下的东西 ——
       **压缩有两个正交方向**：竖着压（一个 token 存多少）是旋钮①，
       横着压（几个 token 合一条）才是 CSA/HCA，再加「读哪几条」是旋钮②。
     ⚠️ 「两种失效模式互补所以交错」是从定义推出的解释，图里标了口径。 -->
__FIG_CSA_WHY__
<h4>6.4c　CSA 压的是 token 不是头 —— 压缩的两个方向</h4>
<!-- ⛔ 2026-09-12：「先问它动了哪几个轴」这句 fig-csa-why 的落点带上已经有了。 -->
<p>⭐ 那这三个轴<b>跟三个旋钮怎么对上</b>？——&nbsp;
  而这一处全讲原先有三种说法，必须在这儿定死一个：</p>
<div class="note ok"><p>⭐⭐ 「横着压」不是第四个旋钮，它是<b><u>旋钮① 的第二个方向</u></b>。
  <em>旋钮① 问的是「每个 token 留多少字节」，而这件事有两个压法：
  <b>沿特征维压</b>（MQA/GQA/MLA ——&nbsp;<a href="#s五">§五</a>讲的全是这个方向）和
  <b>沿 token 维压</b>（CSA：几个 token 合成一条，于是平均每 token 也变小了）。
  两个方向都在同一个旋钮上，因为它们改的是同一件事：那份要留下来的有多大。</em></p>
<p>⛔ <b>所以三处口径统一成这一条</b>：§四 的名词收纳表写「旋钮①（token 维）＋ 旋钮②」，
  编年史那条泳道同理。<em>⚠️ 而「一个 query 只做三步」这个封闭性论证<b>不受影响</b> ——&nbsp;
  横着压改的仍然是第一步（要留什么），只是换了个维度下刀。</em></p></div>
<p><em>三个轴互不相干，所以可以同时拧 ——&nbsp;
  DeepSeek-V4 就是三个一起拧：MLA ＋ CSA/HCA ＋ DSA。</em></p>

<!-- ⭐ 2026-09-12 补回 6.5b。上一步重排 §六 时，注释里写了「保留 6.5b」，
     **实际的切片却把它一起切掉了** —— 说的和做的不一致，靠事后核对图数
     和小节清单才发现。
     ⭐ 判据：**注释写「保留 X」不等于保留了 X；改完要按清单逐项回点。**
     它必须留：索引器自身的开销这一层，图里没有，而且它是这一支的**第二阶段**。 -->
<h3>6.5b ⭐ 第二阶段：索引本身成了开销 —— IndexShare 与 IndexCache</h3>
<p>前面四个方案（NSA / DSA / CSA+HCA）都有一个共同的零件：一个决定"该看哪几块"的索引器。 §6.2b 讲 DSA 的时候它叫 Lightning Indexer。到这一步为止，所有心思都花在<b>让每个 query 少看几块</b>上。</p>
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
<p>📌 口径，分清能推到哪一步：「四层被迫看同一批块」<b>是能推的</b>
  ——&nbsp;索引器共享，选出的 top-k 集合当然就一样，这一步不需要实验。
  但「这会不会掉点、掉多少」推不出来，而且两家都没公开。
  <em>⚠️ 所以这里说的是<b>「这是一次表达力上的让步」，不是「它让效果变差了」</b>
  ——&nbsp;两句话差得很远，而且它给出 2.9× 的同时并没有报质量回退，
  也可能这一让步在实测上几乎无损。</em></p>
<p>⭐ <b>可带走的那一句只到这里为止</b>：看到「降 2.9× FLOPs」这种数字，
  <b>先去找它在结构上放弃了什么自由度</b> ——&nbsp;
  <em>找到了不等于代价大，但没找到就说明你还没看懂它省在哪。</em></p></div>

<div class="note info"><p>📌 DSA 的打分式子值得念一遍：
  <code>I(t,s) = Σ_j w_j · ReLU(q_j · k_s)</code> ——&nbsp;
  ⭐ 用 ReLU 不用 softmax，<b>纯粹是为了吞吐</b>，论文自己这么说的。
  索引器 64 头（主注意力 128 头），而且跑在 FP8 上；<code>k = 2048</code>。</p></div>

<!-- ⭐⭐ 2026-09-14 故事线审计 R25。审计第四条：**钩子只写了一半。**
     §一末、§三末都有很好的段尾钩子，证明作者会写 —— 但没系统地写。
     ⛔ 这一节原来是拿一串超参收尾（64 头、FP8、k=2048）。
       **用参数清单收尾的那一节，读者会在这里放下。** -->
<p>⭐ 段尾：两个旋钮拧完了，回头看一眼它们的共同点 ——&nbsp;
  旋钮①② 都还在<b>跟那张表打交道</b>：一个让每格更小，一个让读的格子更少，
  <em>但那张会一路变长的表，始终都在。</em></p>
<p>⭐⭐ 下一节是唯一一个不跟它讨价还价的 ——&nbsp;<b>它直接把表拿走。</b>
  <em>而拿走之后换来的，不是一个更快的注意力，是另一个模型。</em></p>

<hr>
</div></section>
<section id="s七"><div class="wrap"><div class="stn"><span class="badge">第 七 节</span><h2>旋钮③：换回一个固定大小的状态 —— 线性注意力</h2></div>
__FIG_TX_K3__
<!-- ⭐⭐⭐ 2026-09-13 现场拍板「要搬」：主线图右边那条注解栏**整段搬到这里**。
     现场判据：「图是图，字是字。」——&nbsp;十几行解释本来就是正文的活。
     ⭐ 搬过来还白捡三样：**能搜索、能复制、手机上会自动折行** ——&nbsp;
       这三样在 SVG 里一样都没有，而且在那儿还得用 11px 去换。
     ⛔ 改图的时候不要把这几段又塞回 SVG。 -->
<div class="note info"><p>③ <b>不用听解释 ——&nbsp;读输出形状就够了</b></p>
<p>softmax 的分母要对所有位置求和，所以它<b>锁死了乘法顺序</b>：
  必须先 Q·Kᵀ（于是必须造出那个平方大的矩阵），再乘 V。
  ⭐ 把 softmax 拿掉，乘法就可以重新结合：先 KᵀV，再乘 Q。</p>
<p>⭐⭐ 看点亮那格的输出形状：<code>BKHH</code> ——&nbsp;<b>S 不见了。</b>
  状态大小只跟头维 H 有关，跟序列多长无关。
  <em>这就是 O(N²) → O(N) 的全部内容，写在形状里，不用相信谁。</em></p></div>
<div class="note danger"><p>⛔ <b>但代价也写在同一格里。</b>
  因果版不能真的这么一乘 ——&nbsp;状态要按 t 一步步累加，于是串行回来了；
  分块并行（chunkwise）就是为了把并行度再找回来（见 <a href="#s七">§7.4</a>）。</p>
<p>⚠️ <b>它不是「更快的 attention」，是另一个模型。</b>
  固定大小的状态 →&nbsp;信息必然有损，长程精确检索会力不从心。</p></div>
<!-- ⭐⭐⭐ 2026-09-13 夜间 R11。现场给了一个比喻并说它「非常重要」：
     「一个记事板，固定大小。一种是疯狂往里写、后写的覆盖先写的；
       一种是先把先写的叉掉，再写后写的；一种是选择性地叉掉。」
     ⭐ 这三句正好是这一支的三代（纯线性 → delta rule → 门控）。
       这张图做的事是：**把「叉掉」钉到式子里的一个确切位置上** ——
       (I − β k kᵀ) 就是它。
     ⭐⭐ 顺带给出第三个读法：delta rule 等价于对 ½‖Sk−v‖² 做一步 SGD，
       于是状态不再是缓存，而是**一个边跑边被训练的小模型**。 -->
<!-- ⭐⭐⭐ 2026-09-13 新增。调研全网之后发现的**本课最大的一个缺口**：
     「去掉 softmax，乘法就可以重新结合」是旋钮③ 的立身之本，
     而我们原来**一张图都没有** —— 只有一句话和一个结果（BKHH，S 不见了）。
     ⛔ 结果不等于动作：学生看得到 S 消失了，看不到是哪一步让它消失的。 -->
__FIG_ASSOC__

__FIG_NOTEPAD__
<h3>7.0　一块固定大小的记事板</h3>
<!-- ⛔ 2026-09-12：这句提问法 fig-notepad 的落点带上已经有了，此处不再复述。 -->
<p>⭐ 它换的东西比前两个旋钮都大：
  <em>旋钮①②改的是那张 n×n 的表怎么存、怎么读，表本身一直都在；
  旋钮③ 是<b>第一次把那张表整个拿掉</b>，换成一块固定大小的板子。
  ——&nbsp;所以它是唯一一个<b>让 S 从张量形状里消失</b>的旋钮，
  也是唯一一个换了数学的。</em></p>
<p>⭐⭐ <b>§1.4 认的那个主角，在这一节退场了。</b>
  <em>前两个旋钮一直在跟 S 讨价还价 ——&nbsp;少存一点、少读一点；
  这一个直接把它请出了张量形状。代价你马上会看到：
  它换来的是另一个模型，不是一个更快的注意力。</em></p>

<!-- ⭐⭐⭐ 2026-09-13 夜间 R19。调研 agent 的结论很直接：
     **「先擦后写」这个动作，全网没有一张画好的图。**
     Songlin Yang 只给了 Householder 镜面（而且是 β=2 的完整反射，
     跟实际用的 β∈(0,1] 部分擦除对不上）；中文里讲得最透的那篇 KDA 长文
     **一张图都没有**。⭐ 所以这一张是本讲最可能的差异点。
     ⛔ 不要跟 fig-notepad 混：那张讲的是「三种写法」这个**比喻本身**，
       这一张是**把其中「擦」那个动作拆开**，并且一直拆到 KDA 的逐通道门。 -->
<h3>7.0b　把那个「擦」的动作拆开看 ——&nbsp;查 · 擦 · 写</h3>

<p>上面那块记事板有三种写法，区别都在<b>擦不擦、怎么擦</b>。
  <em>这一小节就把那个动作拆成三步来看 ——&nbsp;
  苏剑林给了它最好的中文名字：<b>除旧迎新</b>。</em></p>

__FIG_ERASE__

<div class="note ok"><p>★ <b>一句话收住这一支的全部改进史</b></p>
<p><b>decay 会忘但不会改，delta rule 会改但不会忘。</b>
  ——&nbsp;<em>所以自然的下一步就是两个拼起来，那就是 Gated DeltaNet；
  而 KDA 只在它之上改了一处：<b>调光器从一个总开关变成每格一个</b>。</em></p>
<p>🏠 为什么要分开调？「你现在在写哪门编程语言」这条<b>该留很久</b>；
  「刚离开的那个函数里的变量名」<b>可以马上忘掉</b>。
  <em>一个总开关做不到这件事。</em></p></div>

<div class="note warn"><p>⚠️ <b>抽屉那个画面有一处不诚实，这里说破</b></p>
<p>真实的「地址」不是一格一格的抽屉，是<b>连续的方向</b>；擦也是按比例擦
  （<code>β</code> 决定擦多干净），而且<b>会顺带擦到相近的地址</b>。</p>
<p>⭐⭐ 这正好回指 <a href="#s一">§1.3b</a>：那里算过，128 维里塞一万个方向，
  最挤的一对还差 60 度 ——&nbsp;<b>「差不多不像」的代价，在这里就变成「擦串了」</b>。
  <em>所以「定点擦」是个近似，不是真的只动一格。板子越满，擦得越串。</em></p></div>

<div class="note info"><p>📌 <b>为什么固定大小的板子一定会坏 ——&nbsp;一软一硬两句</b></p>
<p><b>硬的那句（可以验算）</b>：<code>d</code> 维空间里最多只能有 <code>d</code>
  个互相正交的方向。板子一满，新记录就只能挤在别人旁边。<br>
  <b>软的那句（会被记住）</b>：<em>「记忆的敌人不是时间，是别的记忆。」</em>
  ——&nbsp;你忘掉一个电话号码，不是因为时间久，是因为你又记了新的。</p>
<p>⭐ 一软一硬配在一起，比任何一句单独说都有用：
  <b>诗给画面，数给它一个可以验算的身体。</b>
  <em>（后一句出自 Eagleman《Livewired》，经 Songlin Yang 的 DeltaNet 博客引用。）</em></p></div>

<h3>7.1 基本换法（上面那句「读形状就够了」的推导版）</h3>
<div class="note info"><p>📌 <b>这一节的记号约定</b>（⭐ 2026-09-13 补 ——&nbsp;
  原先三处朝向不一致，数学背景的读者第一眼就卡住）：
  状态 <code>S ∈ ℝ^(d_v × d_k)</code>；转移矩阵 <code>A_t</code> 一律<b>右乘</b>
  （<code>S_t = S_{t-1} · A_t + v_t k_tᵀ</code>）；读出写作 <code>S_t · q_t</code>。
  <em>A_t 是 d_k×d_k ——&nbsp;<b>只有右乘，维度才对得上。</b></em></p></div>
<p><b>把 softmax 去掉</b>（换成某个可分解的核函数），求和就可以重排：</p>
<pre><code>softmax 版： out_t = Σ_{s≤t} softmax(q_t·k_s) v_s      ← 必须留下所有 (k_s, v_s)
线性版：     S_t   = S_{t-1} + v_t k_tᵀ                ← 一个固定大小的状态
            out_t = S_t · q_t</code></pre>
<p>于是：</p>
<ul><li>复杂度 O(L²·d) → <b>O(L·d²)</b>，对长序列是数量级的差别</li><li><b>没有随长度增长的 KV cache</b> —— 只有一个 <code>d_k × d_v</code> 的状态矩阵</li><li>推理时它就是一个 <b>RNN</b>：读一个 token、更新一次状态、吐一个输出</li></ul>
<p><b>代价说死</b>：状态大小固定 → <b>信息必然有损</b>。 序列越长，往同一个矩阵里塞的东西越多，长程精确检索（"第 30 万字提到的那个电话号码"） 会力不从心。这不是实现问题，是这个换法的性质。</p>
<!-- ⛔⛔ 2026-09-13 夜间 R12 重排 7.2 ＋ 7.2b。
     原来这里是**三张表 ＋ 大段散文**（含两张 A_t 结构表），
     而它们讲的其实只有一件事：**那个矩阵长什么样**。
     ⭐⭐ 这是全讲最该画、却一直没画的地方 —— 矩阵结构画出来一眼就懂，
       用表格讲是在浪费读者的眼睛。
     新图 fig3-at-gallery 吸收了：八种 A_t 的结构与年份、每一步修了什么、
     各自的并行代价、Mamba 那两条字面证据。
     正文只留**图给不了的两样**：两处容易记错的出处，
     以及 KDA 那个 DPLR 特化为什么值得单独说。
     ⛔ 按行号切并对首尾断言。 -->
<h3>7.2 ＋ 7.2b　A_t 的形状决定了一切</h3>
__FIG_AT_GALLERY__
<p>⭐ 一条<b>读法</b>：
  看到一个新的线性注意力，先把它的 A 写出来，再问「这个形状还能不能分块并行」。
  <em>两个问题的答案一配对，你就知道它会不会活下来 ——&nbsp;
  <b>表达力和可算性是一起设计的，不是先设计再优化。</b></em></p>

<div class="note warn"><p>⚠️ <b>两处容易记错的出处，讲的时候要说对：</b></p>
<ul><li><b>delta rule 不是 2024 年的东西</b>，是 Schlag 等 2021 年那篇
  <em>Linear Transformers Are Secretly Fast Weight Programmers</em>
  （可上溯到 1990 年代的 fast weight programmer）。
  2024 年那篇（arXiv 2406.06484）做的是<b>把它并行化</b> ——&nbsp;
  这是另一件事，而且是很关键的一件事。</li>
<li><b>KDA 是 2025 年 10 月的 Kimi Linear</b>，不是 2026 年。</li></ul></div>

<div class="note ok"><p>⭐ 图上最后那两格为什么值得单独说一句：
  KDA 把逐通道的门做成了一个特殊的 <b>DPLR（对角 ＋ 低秩）</b>形式 ——&nbsp;
  正是因为这个特殊形式，才配得出一个比通用 DPLR 便宜得多的分块并行算法。
  <em>它不是「先设计一个强的，再去优化」，是<b>一边看着能不能算得动，一边设计</b>。</em></p></div>

<!-- ⭐⭐ 2026-09-14 夜间 R32 新增 7.2c。
     ⛔ 本讲此前提过一次「对偶」——&nbsp;只在 fig3-at-gallery 的图注里：
       「正因为退了，才证得出跟线性注意力的对偶，才能吃上 Tensor Core」。
     对偶是什么，全讲一个字都没说过，等于抛了个术语就走。
     ⭐ 新图 fig3-duality 把它补上，并且顺手把三样东西接成一条线：
       ① 7.1 的递推读法 ② 7.4 的矩阵读法 ③「换括号」这条暗线
         （Mamba-2 原文管它叫 "a different contraction ordering"）。
     ⭐⭐ 还给 7.4 那根竖条一个准确的高度：半可分的秩上界 N ＝ 状态维度。
     ⛔ 图里那个等号是脚本用 numpy 当场验的，不是画上去的修辞。 -->
<h3>7.2c　对偶：同一个东西的两种读法</h3>
<p>上面那张图的落点里藏了一个没解释过的词：
  「正因为退了，才证得出跟线性注意力的<b>对偶</b>」。
  <em>对偶是什么？这一格补上 ——&nbsp;
  它不是个术语，是个读论文的技巧。</em></p>
__FIG_DUALITY__

<div class="note ok"><p>⭐ 这一格真正有用的地方，是让你会读论文了：
  一篇自称 SSM 的和一篇自称 linear attention 的，
  很可能在讲同一件事 ——&nbsp;只是一个从递推那头写，一个从矩阵那头写。
  <em>拿到新论文先问一句：<b>它的那张 L 长什么样？</b>
  这个问题能穿过命名，直接问到结构。</em></p></div>

<div class="note warn"><p>⚠️ <b>别把这条对偶讲成「所以它们都一样」。</b>
  框架相同不代表模型相同 ——&nbsp;L 换一张，<b>表达力和可算性都跟着变</b>，
  这恰恰是 7.2 那条演化线在折腾的全部内容。
  <em>对偶说的是「能不能换个顺序算」，不是「算出来的东西一样」。</em></p></div>

<h3>7.3 ⚠️ 它不是「更快的 attention」，是另一个模型</h3>
<p>三个旋钮里，只有旋钮 ③ <b>改变了模型能表达什么</b>：</p>
<ul><li>旋钮 ①②：改的是<b>存法</b>和<b>看多少</b>，理论上你还能指着某个历史 token 说 "注意力权重在这儿"。<b>检索是显式的</b></li><li>旋钮 ③：历史被<b>碾进了一个矩阵</b>。你无法指着状态里的某一块说 "这是第 3 万个 token"。<b>检索是隐式的、有损的</b></li></ul>
<ul><li>它<b>不能</b>给已有模型打补丁，必须<b>从头训</b></li><li>评测上要特别看 <b>needle-in-a-haystack 这类精确检索任务</b>， 平均分好看不代表这一类不塌</li><li>也正因如此，<b>几乎没有人纯用线性注意力</b> —— 全都是混合（<b>第八节</b>）</li></ul>
<!-- ⛔ 2026-09-13 夜间 R13 重排 7.4。原来是**纯散文**，
     而它讲的是一件彻底图形化的事：把序列切块、块内并行、块间串行。
     ⭐ 画出来之后，「并行度 1→C、串行步数 L→L/C」这句话不需要解释了。
     正文只留图给不了的一样：那句「它不是更快的 attention」的硬件版收口。 -->
<!-- ⭐⭐ 2026-09-14 夜间 R30：7.4 原来只有「硬件上怎么排」这半边。
     ⛔ 而 fig3-assoc 的图注早就写了「因果 mask 挡住结合律，所以真实实现是分块：
       块内按左边算，块间才用右边（见 §7.4）」——&nbsp;**这句话本讲从没画过，
       §7.4 也从没答过**。读者到这儿只知道「要分块」，不知道「凭什么能分」。
     ⭐ 于是 7.4 补成两半：先答为什么可以（新图，代数），再答硬件上怎么排（原图）。
     标题同步改，别让一张代数图挂在「硬件视角」下面。 -->
<h3>7.4 分块：凭什么可以，以及硬件上怎么排</h3>
<p>先补上<a href="#s四">前面</a>欠下的那一步。我们说过换括号能把平方变成线性，
  也说过因果 mask 会把这个换括号挡住 ——&nbsp;
  那为什么<b>切成块之后又能换回来</b>？</p>
<p>⭐ 答案不在「块小所以算得动」，而在一件把矩阵画出来就看得见的事上。</p>
__FIG_TWO_BRACKETS__
<div class="note ok"><p>⭐⭐ <b>这张图真正的落点，是一句反直觉的话：</b>
  <b>被 mask 挡住的，从来只有对角块。</b></p>
<p>块间那些块里<b>根本没有 mask</b> ——&nbsp;
  第 r 块里的每个 token，都能读前面每一块里的每个 token，没有谁被挡。
  <em>既然那里从来没有逐元素乘挡路，<b>结合律在那儿也就从来没被挡住过</b>，
  自然可以把整块压成一个与句长无关的状态。</em></p>
<p>⭐ 这也解释了图上那笔账：句长翻倍，整张表的有效格子涨 3.85 倍，
  而<b>真正要一格一格算的只涨 2 倍</b> ——&nbsp;正好线性。
  <em>对角块那一部分<b>不会随句长消失</b>，它只是从平方变成了线性。</em></p></div>
<div class="note warn"><p>⚠️ <b>一处别讲过头。</b>Mamba-2 的作者在同一段里紧接着写了：
  被挡住的是<b>「结合律」这个特例，不是「重排」本身</b> ——&nbsp;
  结合律只是张量缩并顺序的一个特例，换成更一般的缩并顺序，
  mask 是能被吸收进去的。</p>
<p><em>所以准确的说法是：<b>分块是通常实现走的那条路，不是数学上的唯一解。</b>
  ——&nbsp;⭐ 这跟本讲反复出现的那条纪律是同一件事：
  「做不到」和「这条路上做不到」是两句话。</em></p></div>
<p>知道了凭什么能分，剩下的就是<b>硬件上怎么排</b>：</p>
__FIG_CHUNKWISE__
<p>⭐ 这一条可以<b>迁移</b>出去：
  <b>块大小被片上内存顶死</b>，这跟 splash attention 的块大小是<b>同一类问题</b>
  ——&nbsp;<em>那边我们实测扫过一轮：<b>最优块是个绝对值，不随序列长度缩放</b>
  （seq 从 4096 拉到 16384，最优块都是 2048）。换 seq 不用重扫，换硬件才要。
  ⭐ 块大小那条线整个归<a href="topic-02.html">专题二</a>。</em></p>

<hr>
</div></section>
<section id="s八"><div class="wrap"><div class="stn"><span class="badge">第 八 节</span><h2>元旋钮 ⊕：混合</h2></div>
<!-- ⭐ 2026-09-13 夜间 R14。§八 原来是**五张表** —— 表该留（型号对配比
     本来就是表），但缺一张**把结论摆出来**的图。
     ⭐⭐ 新图最重要的一格是：§二 那条 L2M 条件把「纯线性」那一头**直接判死**，
       于是「为什么必须混合」从工程经验变成了有下界的结论。 -->
__FIG_HYBRID__
<h3>8.1 为什么混合几乎是唯一的答案</h3>
<p>单用任何一个旋钮都有一个致命短板：</p>
<table>
<thead><tr><th>单用</th><th>短板</th></tr></thead><tbody>
<tr><td>SWA</td><td>跨不了长距离</td></tr>
<tr><td>纯线性</td><td>精确检索塌</td></tr>
<tr><td>纯全注意力</td><td>KV 和 FLOPs 都爆</td></tr>
</tbody></table>
<p>混合的逻辑很朴素：全局层负责精确长程检索，线性/窗口层负责局部与效率，各司其职。 关键在于全局层<b>不需要很多</b> —— 只要有几层能做无损检索， 信息就能沿着残差流传给其余层用。</p>
<div class="note info"><p>📌 <b>「残差流」是什么</b>：主线图上每一层都有两处「＋ 残差」——&nbsp;
  每一层不是把上一层的结果换掉，是<b>在它上面「加一笔」</b>。
  于是从第一层到最后一层，有一条一路贯通、只被不断加料的通道，
  这条通道就叫残差流。</p>
<p>⭐ <b>这正是「资深不用配很多」成立的原因</b>：
  某一层全注意力查到的东西，被加进残差流之后，
后面<b>每一层都读得到</b> ——&nbsp;不需要每层都自己再查一遍。</p></div>
<h3>8.2 配比：3:1 是怎么来的，以及它不是定律</h3>
<table>
<thead><tr><th>模型</th><th>配比</th><th>出处</th></tr></thead><tbody>
<tr><td>Kimi Linear</td><td><b>27 层 ＝ 每 4 层一个全注意力 ＋ 末层再补一个</b>，实际层数 KDA : MLA = <b>20 : 7</b>。<em>⚠️ 循环配比是 3:1，但一除是 2.857 ——&nbsp;跟下面 K3 是同一回事，别写成 3:1</em></td><td>arXiv 2510.26692 ＋ HF config<br><code>full_attn_layers</code></td></tr>
<tr><td><b>Kimi K3</b></td><td><b>93 层 ＝ 23 × (3 KDA ＋ 1 Gated MLA) ＋ 1 MLA</b> ——&nbsp;循环配比 <b>3 : 1</b>，实际层数 69 : 24。<em>⚠️ 别写成「69 : 24 = 3 : 1」，一除就是 2.875 ——&nbsp;<b>末层补一个不是 K3 的花样，上面 Kimi Linear 也是这么排的</b></em></td><td>arXiv 2607.24653 表 1 ＋ sec. 2.1</td></tr>
<tr><td>Ling-3.0-tiny</td><td>KDA : MLA = 3 : 1</td><td>模型卡</td></tr>
<tr><td>Ling-3.0-flash</td><td>KDA : MLA = <b>5 : 1</b></td><td>模型卡</td></tr>
<tr><td>一篇系统性消融</td><td>建议区间 <b>3:1 ～ 6:1</b></td><td>arXiv 2507.06457</td></tr>
</tbody></table>
<p>⭐ <b>三件事要讲清楚：</b></p>
<ol><li><b>3:1 是消融出来的，不是推出来的。</b> Kimi Linear 的消融里， <b>0:1（纯全注意力）反而不是最好的</b> —— 这个结果比"3:1 最好"更有意思： 加线性层不只是省钱，它可能还带来了别的东西</li><li><b>同一家不同规模就换了配比</b>（Ling 的 tiny 3:1 / flash 5:1）—— <b>配比是超参，跟规模和数据有关，不要背下来当常识</b></li><li><b>区间比点值可信。</b> 记 "3:1 到 6:1 这个量级" 就够了</li></ol>
<!-- ⛔ 2026-09-14 R28 改：上面第 1 条原来写着「⚠️ 原文只有这一句定性描述，
     **没公开数值**」——&nbsp;**这是错的**。arXiv 2510.26692v2 §5.2 Table 1
     把五个配比的训练 / 验证 PPL 全列出来了（3:1 9.23/5.65、0:1 9.45/5.77、
     1:1 9.29/5.66、7:1 9.23/5.70、15:1 9.34/5.82）。
     ⭐ 而且数值一摆出来，原来那句「0:1 反而表现不好」方向对但排名说重了：
     0:1 排第四，最差的是 15:1。措辞已改成「不是最好的」，下面补图。 -->
__FIG_RATIO_GRID__
<div class="note info"><p>⭐⭐ <b>顺着这张图，把上面第 1 条再往前推一格。</b>
  那五个配比看起来像是精心挑的，其实是<b>被 16 除出来的</b> ——&nbsp;
  消融模型只有 <b>16 层</b>，配比 r:1 要摆得匀，每组 (r+1) 层就必须整除 16；
  16 的约数只有 1、2、4、8、16，于是 r 只能取 <b>0、1、3、7、15</b>，
  <b>正好就是表里那五行</b>。</p>
<p>⛔ 所以 <b>4:1 从来没被试过</b>（它要 5 层一组，16 层摆到第 15 层多出一层），
  <b>2:1 也没有</b>。「为什么是 3 不是 4」这个问题，<b>那张表问不出来</b>。</p>
<p><em>⚠️ 这是我们对那张表做的<b>算术观察</b>，不是论文给的理由 ——&nbsp;
  原文没有解释为什么选这五个配比。</em></p></div>
<div class="note warn"><p>⚠️ 还有一个口径问题，跟 <a href="#s九">§9.1</a> 那条判据是同一件事。
  这张消融表量的是 <b>PPL</b>，而全场五个配比的验证 PPL 只从 5.65 到 5.82，
  <b>一共差 3%</b>。</p>
<p>另一篇 340M / 1.3B 的系统性消融（arXiv 2507.06457）把两个口径分开画，
  结论是<b>一条平的、一条涨的</b>：语言建模分各架构都挤在 0.55～0.57、几乎不受配比影响；
  而召回（RULER）从纯线性的 0.1～0.35 一路涨到全注意力基线约 0.42，
  <b>多数架构在 3:1 追平或超过</b>。它给这个结论起的标题是
  「<b>决定配比的是召回，不是困惑度</b>」。</p>
<p>⭐ 换句话说：<b>一个「最优配比」也得说清它是按哪个指标最优的</b> ——&nbsp;
  跟 <a href="#s九">§9.1</a> 说「一个倍数得说清它是哪一样的倍数」是同一条纪律。</p></div>

<!-- ⭐⭐ 2026-09-14 夜间 R31 新增 8.2b。R28 画完配比消融表之后留了个悬案：
     K3 是「93 层里 24 层全注意力」——&nbsp;那守恒的到底是**比值**还是**绝对层数**？
     ⛔ 这不是学究问题：§8.1 那句「全局层不需要很多」**按字面读就是一个可证伪的预测**
       （绝对数与深度无关）。去数 14 个模型的 config.json，这个读法被否掉了。
     ⛔⛔ 顺带查出课程自己的一处漏：Kimi Linear 那行只写「3:1」，而它跟 K3
       一样是「20:7 = 2.857」，同一个警告课程只给了 K3。表已改。 -->
<h3>8.2b 守恒的是比值，还是全注意力的层数</h3>
<p>上面那张消融表只在<b>一个深度</b>上扫配比，所以它答不了一个更基本的问题：
  <b>配比 3:1 是超参，还是「几层全注意力」才是超参？</b></p>
<p>⛔ 这个问题有实际后果。<a href="#s八">§8.1</a> 写过「全局层不需要很多 ——&nbsp;
  只要有几层能做无损检索，信息就能沿残差流传给其余层用」。
  <em>这句话按字面读，预测的是<b>绝对条数与深度无关</b>：模型越深，比值就该越大。
  它是可以被证伪的 ——&nbsp;去数配置文件就行。</em></p>
__FIG_RATIO_OR_COUNT__
<div class="note danger"><p>⛔ <b>数完之后：那个字面读法是错的。</b></p>
<p>Qwen3.5 七个尺寸（深度 24 → 64）的 <code>layer_types</code> 逐层写死，
  <b>恰好 3:1 一次不差</b>；Kimi 自己更是最干净的反证 ——&nbsp;
  Kimi Linear 27 层 / 7 条，K3 93 层 / 24 条，
  <b>绝对数涨了 3.4 倍，比值纹丝不动</b>。</p>
<p><em>所以「不需要很多」的准确意思是<b>占比低</b>（各家落在 1/4 到 1/10 之间），
  <b>不是绝对条数少</b>。反过来说：<b>模型越深，你要付的全注意力层就越多</b> ——&nbsp;
  混合省下的是一个固定比例，不是「越深越划算」。</em></p></div>
<div class="note ok"><p>⭐ <b>但真实规则是两条叠加，不是单一的「按比例」：</b></p>
<p>① <b>主体按固定比例铺</b>（3:1、7:1、每 10 层一个）——&nbsp;这部分随深度线性涨；
  ② 外加几个<b>按「位置」钉死</b>的全局层 ——&nbsp;这部分是常数。</p>
<p>Kimi 两个模型都是「每 4 层一个 ＋ <b>末层必为全局</b>」。
  <em>正是末层那一个额外的，把实际比值从 3.0 压下来一点点：
  27 层时是 2.857，93 层时被摊薄到 2.875 ——&nbsp;
  <b>表里那两个别扭的小数，是这么来的。</b></em></p></div>
<div class="note warn"><p>⚠️ <b>假说 B 也不是全无依据，但依据不在「数量」上。</b>
  Hymba 的全局层只有<b>首 / 中 / 末</b>三层，理论上多深都是 3 ——&nbsp;
  但它是<b>按位置</b>定的，不是按数量定的，这是两回事。</p>
<p><em>⛔ 还有一个反面参照：<b>MiniMax-M2 干脆退回了全注意力</b>
  （62 层全是 full attention）。<b>混合不是一条只进不退的路。</b></em></p></div>
<!-- ⛔ 2026-09-12 二轮学生审稿：这一小节原来叫「K3 的 NoPE」，**记错了发明人**。
     Kimi Linear（arXiv 2510.26692）原文就写着「we apply NoPE to all full
     attention (MLA) layers」，而 K3 sec. 2.1.2 自己说的是「follows the hybrid
     design of Kimi Linear and applies NoPE to all MLA layers」。
     ⭐ 而且 §8.4 速查表里 Kimi Linear 那一行也漏了 NoPE —— 表跟着一起错。 -->
<h3>8.3 ⭐ 从 Kimi Linear 到 K3 的 NoPE —— 混合带来的一个意外红利</h3>

<p><b>Kimi Linear 就已经这么做了</b>，K3 只是照搬：全注意力（Gated MLA）层<b>完全不加位置编码</b>（NoPE）—— 没有 RoPE，没有 YaRN，什么都没有。</p>

<!-- ⭐⭐⭐ 2026-09-13 夜间 R17。这一小节原来全是文字，而它其实是
     **fig3-absorb（§5.4b）那张图的续集** —— 那张说「R 夹在中间挡住吸收」，
     这一节说「混合之后那个 R 干脆没了」。⭐ 两张连着读，收益比各自大得多。
     ⛔ 并且图里守住了 §8.3 正文刚修过的那条边界：
       **75% 是配比的功劳，不是 NoPE 的**，别让图再错一遍。 -->
__FIG_NOPE__
<p>为什么敢这么做？ 因为它们中间夹着的 KDA 层， 本身就是靠递归的衰减和门控在编码顺序 —— 一个天然带时序的算子。 <b>位置信息由线性层提供，全注意力层只管检索。</b></p>
<p>两个后果，一个比一个实在（⚠️ 原先这里写「三个」——&nbsp;第三条已经在下面那个框里撤回了，数字忘了跟着改）：</p>
<ol><li><b>不用调 RoPE 外推。</b> 模型直接外推到 1M，不需要任何位置编码的重标定 —— 长上下文扩展里最烦人的一块调参，直接消失了</li><li><b>MLA 层在推理时可以退化成纯 MQA。</b> 位置编码没了， §5.3 里那条"不可吸收的 64 维"也就不存在了 —— <b>上投影可以完全吸收</b></li></ol>
<div class="note warn"><p>⛔ <b>这里原先还列了第三条「KV cache 最多降 75%」——&nbsp;那一条不是 NoPE 的功劳。</b>
  75% 来自 <b>3:1 的配比</b>（四层里只有一层是全注意力），<b>跟加不加位置编码无关</b>。
  <em>⭐ 这个错误值得留在页面上：<a href="#s八">§八那张配比图</a>的 ⚠️ 注早就写对了，
  而正文没跟着改 ——&nbsp;<b>图改对了不等于文改对了，同一个事实有两个落点就会有两个版本。</b></em>
  1M 下 TPOT 从 11.48 ms 降到 1.84 ms（<b>6.3×</b>）这个数仍然成立，它记在配比头上。</p></div>
<div class="note ok"><p>⭐ <b>这才是"混合"真正的意思</b>：不是"两个方案各跑一半凑合用"， 而是让每一层只做自己擅长的事，然后把别人不用做的事一并省掉。 一个架构选择（混合）解开了另一个看起来完全无关的约束（位置编码）。 这门课想教的就是这种"看见约束之间的连接"的能力。</p></div>
<h3>8.4 ⭐ 各家速查：你日常在用的那些模型，注意力到底是什么</h3>
<p>三个旋钮到这里就拆完了。这一小节反过来 —— <b>按公司排一遍，看每一家实际拧的是哪个旋钮</b>。 都是能在公开 config 或官方博客里查到的，信息截至 <b>2026-09-07</b>。</p>
<table>
<thead><tr><th>家</th><th>代表型号</th><th>拧的是哪个旋钮</th><th>配比 / 形态</th></tr></thead><tbody>
<tr><td rowspan="2">阿里 千问</td><td>Qwen3-Next（80B/3B）</td><td>③ 线性（Gated DeltaNet）</td><td><b>3 : 1</b></td></tr>
<tr><td>Qwen3.5（0.8B–397B）</td><td>③ 线性</td><td><b>3 : 1</b>，全家族统一</td></tr>
<tr><td rowspan="2">月之暗面 Kimi</td><td>Kimi Linear（48B/3B）</td><td>③ 线性（KDA）</td><td><b>3 : 1</b></td></tr>
<tr><td>Kimi K3（2.8T）</td><td>③ 线性 ＋ NoPE（沿用 Kimi Linear）</td><td><b>93 层 ＝ 23 × (3 KDA ＋ 1 MLA) ＋ 1 MLA</b></td></tr>
<tr><td rowspan="2">蚂蚁 百灵 Ling</td><td>Ling 2.6</td><td>③ 线性（Lightning）</td><td><b>7 : 1</b></td></tr>
<tr><td>Ling-3.0-flash（124B/5.1B）</td><td>③ 线性（KDA）</td><td><b>5 : 1</b> ＝ 35 KDA ＋ 7 MLA</td></tr>
<tr><td rowspan="2">小米 MiMo</td><td>MiMo-V2-Flash</td><td>② 稀疏（SWA，窗口 128）</td><td><b>5 : 1</b></td></tr>
<tr><td>MiMo-V2.5-Pro</td><td>② 稀疏（SWA，窗口 128）</td><td><b>6 : 1</b></td></tr>
<tr><td rowspan="2">DeepSeek</td><td>V3.2</td><td>② 稀疏（DSA）</td><td>层内稀疏</td></tr>
<tr><td>V4</td><td>② 稀疏（CSA ＋ HCA）</td><td>层内稀疏，按距离分层压缩</td></tr>
<tr><td>MiniMax</td><td>01 → M2 → M3</td><td>③ → 退回基线 → ②</td><td>7 : 1 → 纯全 → 层内稀疏</td></tr>
</tbody></table>
<!-- ⛔ 2026-09-12 二轮学生审稿：这两张展开表与开篇编年史图「三家公司」那格
     几乎同内容同结论，而 §八 全部预算只有 4 分钟、讲义 §八 讲稿**一个字都没提 8.4**。
     ⭐ 但它们里面有两样是别处没有的：① Hy3 那一列是**读我们自己仓库的 config 数出来的**
       （全课唯一的一手反例：80 层里没有一层线性、没有一层稀疏）；
       ② GLM-5.3-Flash 是整张表里**唯一一个两种便宜法同时上**的模型。
     → 主线上只留这两句落点，表整体折叠 —— 按本课的规矩：
       **折叠里放「想深挖的人才要的」，关着折叠读正文必须依然通顺。** -->
<div class="note ok"><p>⭐ <b>这张速查表里有两个型号值得单独点名，各代表一种「不走大路」的走法</b>：</p>
<ul>
<li>腾讯 Hy3 ——&nbsp;<b>混合不是唯一解</b>。
  80 层里<b>没有一层线性、没有一层稀疏</b>，纯 GQA-8 做到 256K。
  <em>⭐ 这一列不是查来的，是读我们自己仓库里那份 config 数出来的
  （<code>tpu/Hunyuan3-295B-Pretraining/</code>）——&nbsp;全课唯一的一手反例。
  然后 Hy4 一步跨到全层稀疏，整个跳过了线性这一支。</em></li>
<li><b>智谱 GLM-5.3-Flash ——&nbsp;三个旋钮可以叠着拧。</b>
  它是整张表里<b>唯一一个「线性配稀疏」</b>的：便宜的那层是 KDA，
  而它配的那层「贵的」<b>本身已经是稀疏的</b>（NoPE 稀疏 MLA）。
  <em>⭐ 而配比还是落回 <b>3 : 1</b> ——&nbsp;换了公司、换了搭档层的类型，区间不变。</em></li>
</ul></div>

<details class="aside"><summary>📊 <b>这两家的完整轨迹表</b>
<em>（层数、config 字段、逐版本对照 ——&nbsp;想自己核的人点开；讲课时不展开）</em></summary>
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
<p><b>② 智谱 GLM —— 半年之内走完三步，而且步步可查</b></p>
<table>
<thead><tr><th>版本</th><th>时间</th><th>注意力</th><th>这一步新增了什么</th></tr></thead><tbody>
<tr><td>GLM-5（355B–744B）</td><td>2026-02-12</td><td>MLA ＋ <b>DSA</b></td><td>智谱第一次上稀疏</td></tr>
<tr><td>GLM-5.2（744B）</td><td>2026-06-16</td><td>MLA ＋ DSA ＋ <b>IndexShare</b></td><td>每四个稀疏层共用一个索引器（见 §6.5b），1M 下省 <b>2.9×</b> FLOPs</td></tr>
<tr><td><b>GLM-5.3-Flash</b>（320B/18B）</td><td>2026-08-26</td><td><b>KDA 线性 ＋ NoPE 稀疏 MLA</b></td><td>⭐ GLM 家族<b>第一次把线性和稀疏放进同一个模型</b>；原生多模态</td></tr>
</tbody></table>
<p>GLM-5.3-Flash 的 <code>layer_types</code> 是一个干净的四层循环：</p>
<pre><code>linear, linear, linear, deepseek_sparse_attention,   ← 重复 11 次
linear                                               ← 第 45 层多出来的一层

45 层 = 34 层 KDA + 11 层稀疏 MLA        循环配比 3 : 1</code></pre>
</details>
<!-- ⭐⭐⭐ 2026-09-12 二轮学生审稿提的一条，我认为是全部反馈里
     **最值钱的一条增量**：这一讲反复教「拿到一个新名字先把它放进某一格」，
     但**从头到尾没有一次让读者自己放** —— 所有名字都是作者放好的。
     课前那两道题又都是算数题。
     ⭐ 判据：**一个「姿势」如果没被练过一次，它就只是一句口号。**
     ⛔ 出题原则：三道题的答案**必须都能在本讲内部核到**，
       不引入任何本讲没核过的机制（否则答案本身就是新的未验证断言）。 -->
<h3>8.5 ⭐ 六十秒课后题：这一讲唯一一次让你自己动手</h3>
<p>下面三个机制，本讲<b>都提到过但没有明确归位</b>。
  每一个只回答三问，<b>各一个词</b>：</p>
<ol>
<li><b>它在拧哪个旋钮？</b>（①每份多大 / ②每步读多少 / ③换数学 / 都不是）</li>
<li><b>事后，还是 native？</b></li>
<li><b>它省的是字节，还是 FLOPs？</b></li>
</ol>
<table>
<thead><tr><th>题</th><th>机制</th><th>本讲在哪儿提过</th></tr></thead><tbody>
<tr><td><b>A</b></td><td><b>CLA</b> ——&nbsp;每 2 层共享同一份 KV</td><td>44 行表里有一行</td></tr>
<tr><td><b>B</b></td><td><b>KV 量化</b> ——&nbsp;每个数从 16 bit 降到 8 bit</td><td><a href="#s四">§四</a> 那个 ⚠️ 框</td></tr>
<tr><td><b>C</b></td><td><b>IndexShare</b> ——&nbsp;每四个稀疏层共用一个索引器</td><td>§6.5b</td></tr>
</tbody></table>
<details class="aside"><summary>✅ <b>对答案</b>
<em>（⛔ 先自己写下九个词再点开 ——&nbsp;看着答案想「我本来也这么想」是没有用的）</em></summary>
<table>
<thead><tr><th></th><th>哪个旋钮</th><th>事后 / native</th><th>省字节还是 FLOPs</th><th>⭐ 这题在考什么</th></tr></thead><tbody>
<tr><td><b>A · CLA</b></td><td><b>①</b>（少存几份）</td><td>native</td><td><b>字节</b></td>
  <td>旋钮① 的第三招：MQA 砍头数、MLA 压维度、<b>CLA 减层数</b> ——&nbsp;
  三招都在回答「那份要留下来的有多大」</td></tr>
<tr><td><b>B · KV 量化</b></td><td><b>都不是</b></td><td>两种都有</td><td><b>字节</b></td>
  <td>⛔ <b>这是个陷阱题</b>，而且是<a href="#s四">§四</a>主动说掉的那个：
  三个旋钮管「存几个数、读几个数」，量化管「每个数几个 bit」——&nbsp;<b>它正交</b></td></tr>
<tr><td><b>C · IndexShare</b></td><td><b>都不是</b></td><td>native</td><td><b>FLOPs</b></td>
  <td>⭐ 最难的一道：它不改「读哪些」，<b>它省的是旋钮② 自己的开销</b>
  （索引器）——&nbsp;所以它是<b>旋钮上的优化，不是旋钮</b>。
  ⚠️ KV 一个字节都没少</td></tr>
</tbody></table>
<p>⭐⭐ 三道里有两道答案是「都不是」——&nbsp;这是故意的。
  <em>一张分类表真正的用处不是「什么都装得下」，是让装不进去的东西显形。
  §4.2b 那句话反过来说一遍就是：
  <b>放不进去的，才值得你花时间。</b></em></p>
</details>
<hr>
</div></section>
<section id="s九"><div class="wrap"><div class="stn"><span class="badge">第 九 节</span><h2>代价：没有免费的午餐</h2></div>
<p>一张表把所有方案摆在一起：</p>
<table>
<thead><tr><th>方案</th><th>KV 显存</th><th>计算量</th><th>⭐ 省在哪个阶段</th><th>长程质量</th><th>kernel 复杂度</th><th>能否给已有模型打补丁</th></tr></thead><tbody>
<tr><td>MHA</td><td>基准</td><td>基准</td><td>—</td><td>基准</td><td>简单</td><td>—</td></tr>
<tr><td>GQA</td><td>↓↓</td><td>—</td><td><b>decode</b>（省带宽）</td><td>↓</td><td>简单</td><td>需微调</td></tr>
<tr><td>MLA</td><td>↓↓↓</td><td>↑（训练时）</td><td><b>decode</b>（训练前向反而更贵）</td><td>≈</td><td>中</td><td>不能</td></tr>
<tr><td>SWA</td><td>↓↓↓</td><td>↓↓</td><td>两边都省</td><td>↓↓↓</td><td>简单</td><td>勉强（要留 sink）</td></tr>
<tr><td>DSA</td><td><b>—</b>（KV 全存，只是不读）</td><td>↓↓↓</td><td><b>prefill</b> 为主（decode 省的是读）</td><td>≈</td><td><b>高</b></td><td>需专门训练阶段</td></tr>
<tr><td>CSA/HCA</td><td>↓↓↓</td><td>↓↓↓</td><td>两边都省</td><td>≈</td><td><b>很高</b></td><td>不能</td></tr>
<tr><td>线性（KDA 等）</td><td><b>无 KV</b>，但有固定状态</td><td>↓↓↓</td><td><b>decode</b>（prefill 要 chunk 化才不亏）</td><td>↓↓</td><td><b>很高</b></td><td><b>不能，必须从头训</b></td></tr>
</tbody></table>
<!-- ⭐⭐⭐ 2026-09-14 R26。这一节原来**只有上面这张表**，而整张表都是 ↓↓↓。
     ⛔ 这门课自己的规矩是「问『省了多少』之前先问『省的是哪一样』」——
        箭头恰恰是**回答不了这个问题**的东西：↓↓↓ 既不说省的是哪一样，
        也不说省了多少。⭐ 于是这里把同一批方案一路换算到**毫秒**。
     ⭐ 顺带把 §二 那张图欠下的账还了 —— 它的图注明写着「只算了装得下装不下，
        **没算带宽**」。带宽这一笔，就在这儿。 -->
<h3>9.1 把 ↓↓↓ 换成一笔能自己核的账</h3>
<p>上面这张表每一格都是箭头。箭头能排序，<b>但它回答不了「省的是哪一样」</b> ——
  而那正是本节唯一想说死的那条。所以把同一批方案，换成一个能拿计算器核的问题：
  <b>吐一个字，要从 HBM 上搬多少字节？</b></p>
__FIG_PERSTEP__
<p>三行收口，都能自己验算：</p>
<ul>
<li><b>省显存 ≠ 省时间。</b> MLA 对 MHA，显存 <b>56.9×</b>，
  单用户 decode 只有 <b>7.08×</b>。<em>⭐ 那 8 倍不是凭空少掉的，
  它是在<b>两步</b>里被吃掉的 ——&nbsp;跟 §五 把 56.9 拆成「4.571 白送 × 12.4 赌出来」
  是同一个手法：</em>
  <ul>
  <li><b>56.9×</b> ——&nbsp;显存之比（KV 488 ÷ 8.58）</li>
  <li><b>÷ 4.69　→　12.14×</b> ——&nbsp;每步还要读一份<b>所有人共享</b>的权重
    （34.46 GiB），它把比例摊薄了</li>
  <li><b>÷ 1.71　→　7.08×</b> ——&nbsp;MHA 装不下，<b>被迫用 12 张 device 而不是 7 张</b>；
    它拿显存换来的卡，顺手也把带宽换来了（1.71 ＝ 12 ÷ 7）</li>
  </ul>
  <em>⭐ 所以 MLA 真正省下的不是时间，<b>是那 5 张卡</b> ——&nbsp;
  而它们可以拿去服务别人。</em></li>
<li><b>省读 ≠ 省存。</b> DSA 那根柱子一个字节没少存，它只是<b>每步不读</b> ——&nbsp;
  <em>所以它改变高度，不改变「要几张卡」。</em></li>
<li>⭐⭐ <b>瓶颈是会搬家的。</b> MHA 时代它在 KV（占一步的 <b>93.4%</b>）；
  MLA 之后它搬到了权重（<b>80.1%</b>）。
  <em>对着上一个瓶颈继续优化，是这一行最常见的浪费。</em></li>
</ul>
<div class="note warn"><p>⚠️ <b>这笔账只成立在 batch ＝ 1 上。</b>
  权重那一段是<b>所有人分摊</b>的，KV 那一段<b>不摊</b> ——&nbsp;
  人一多，前者被摊薄、后者成倍长，<b>画面会翻回 KV 主导</b>。</p>
<p>⭐ 这正好是 <a href="#s二">§二那张图</a>的另一面：那里是<b>把人加上去</b>，
  让 KV 变成主角；这里是<b>只留一个人</b>，让权重变成主角。
  <em>同一个模型，问法不同，答案就不同 ——&nbsp;这本身就是本节的主题。</em></p></div>
<!-- ⛔ 2026-09-13 夜间 R18：这四条原来在这里用 <ol> 写了一遍，
     而 R15 那张 fig3-landing 的第二格**已经把它们画出来了**（连第四条那个
     12% / 10% 的账都画了）。⭐ 同一件事两处各讲一遍，正是本课一直在修的毛病。
     → 正文压成一行指针，四条的完整版看 §十 那张图。 -->
<div class="note info"><p>⭐ 这张表要配着 <a href="#s十">§十那张落点图</a> 的第二格看
  ——&nbsp;那里把<b>四个最容易被低估的取舍</b>画在了一起：
  <b>省显存 ≠ 省计算</b> · <b>训练时省 ≠ 推理时省</b> · 不规则访存的代价常被低估 ·
  收益有天花板。</p>
<p>⭐ 这一节唯一要在这儿说死的是<b>提问顺序</b>：
  <b>问「省了多少」之前，先问「省的是哪一样」。</b>
  <em>这张表的每一列，就是一样不同的资源。</em></p></div>
<hr>
</div></section>
<section id="s十"><div class="wrap"><div class="stn"><span class="badge">第 十 节</span><h2>落到硬件（本专题的落点）</h2></div>
<h3>10.1　三种资源之间的搬家史</h3>
<p>回到全课那条主线：<b>每一个变体都是被硬件逼出来的，也都对硬件提出了新要求。</b></p>
<!-- ⭐⭐⭐ 2026-09-13 夜间 R15。这是**整个专题的落点**，
     而它原来只有两张表和一段散文。它该是一张图 ——
     因为它讲的本来就是**一个空间里的移动**：
     在显存 / 算力 / 访存规整度三者之间反复搬家。
     ⭐ 第三格那条防骗判据是这一讲最该带走的一句：
       注意力只是账单的一部分，任何倍数**必须带上「在多长的上下文下」**。 -->
__FIG_LANDING__
<!-- ⭐ 2026-09-14 R26 接线。图里那条防骗判据只说了一半（「在多长的上下文下」），
     而 §9.1 刚刚算出了另一半：同一个 MLA，显存 56.9×、时间 7.08×，差了整整八倍。
     ⛔ 这里只放指针 + 那一个对照，不复述图 —— 「图给得了的，正文不复述」。 -->
<div class="note ok"><p>⭐ <a href="#s九">§9.1</a> 刚给这条判据补上了<b>另一半</b>：
  一个倍数除了要带「<b>在多长的上下文下</b>」，还得说清<b>它是哪一样的倍数</b>。</p>
<p><em>同一个 MLA：<b>显存 56.9×</b>，<b>单用户 decode 的时间只有 7.08×</b>。
  两个数都对，而它们差了整整<b>八倍</b> ——&nbsp;
  报哪一个，取决于你想让听的人以为你省了多少。</em></p></div>

<table>
<thead><tr><th>变体</th><th>它假设了什么硬件条件</th><th>条件不成立会怎样</th></tr></thead><tbody>
<tr><td>MLA</td><td>算力相对充裕、显存相对紧张</td><td>算力紧张的机器上，用计算换显存这笔交易不划算</td></tr>
<tr><td>稀疏（DSA/NSA/CSA）</td><td>gather 不太贵</td><td><b>对规整访存友好的加速器反而吃亏</b> —— 纸面 64 倍拿不到</td></tr>
<tr><td>线性（KDA）</td><td>片上内存够放下 chunk 的中间量</td><td>chunk 被迫调小 → 并行度掉 → 优势被吃掉</td></tr>
<tr><td>长上下文 + MoE 同时上</td><td>HBM 带宽够两边分</td><td>all-to-all 与 KV cache <b>抢同一份带宽</b></td></tr>
</tbody></table>
<!-- ⛔ 2026-09-12 二轮学生审稿（叙事与取舍那位）：这里原来有一段
     「一句话收尾」散文，**跟 fig3-landing 的落点带逐字相同**。
     ⭐ 判据沿用本讲已经用过两次的那条：**图给得了的，正文不复述。**
     → 删散文，留图。 -->

<h3>10.2 ⭐⭐ 上面那张表的第二行，落到我们自己的机器上是什么样</h3>
<p>那张表里「稀疏（DSA/NSA/CSA）假设 gather 不太贵」这一行，
  <b>在 TPU 上就是一整个工程战场</b>。而这恰好是这门课<b>唯一有资格讲、别人讲不了</b>的部分 ——
  所以它值得单独占两张图。</p>
<!-- ⭐⭐⭐ 2026-09-12 TPU 轮 R27。现场点的题：
     「注意力在 TPU 上跑有没有困难？哪些本来是给 GPU 设计的、搬过来需要克服？」
     ⭐ 先摆**结构性的错配**，不摆清楚，后面那些 kernel 技巧看起来就只是一堆技巧。 -->
__FIG_TPU_GAP__

<div class="note warn"><p>⚠️ 先把「谁是给谁设计的」说清楚，免得听成 TPU 的黑历史。
  这一讲从 §四 到 §八 讲的每一个机制，
  它们的第一版 kernel 全部是在 GPU 上写出来的 ——&nbsp;
  FlashAttention、PagedAttention、NSA 的三支路、DSA 的 indexer，无一例外。
  所以「搬到 TPU 上有难度」不是 TPU 的缺陷，是这批机制自带的一条硬件假设：
  <b>随手 gather 不太贵</b>。
  <em>⭐ 而这条假设，正是 TPU 为了换取规整访存下的高效率而主动放弃的。</em></p></div>

<h3>10.3 ⭐⭐ 那怎么克服 —— 三招，以及两个一定会被问到的问题</h3>
<!-- ⭐⭐⭐ 2026-09-12 TPU 轮 R28–R31。这张图要回答的是现场点名的三问：
     ① 不连续的 KV gather 怎么办（DMA 调度）
     ② 运行时那个「取哪 2048 条」的决定，成本是什么？能不能全在卡上算？
        要不要发回 host？
     ③ SparseCore 能不能帮上忙？
     ⛔ 第三问必须**诚实地留在「看起来对但还没被公开验证」**上 ——
       公开的那套 TPU 生产注意力 kernel（RPA）走的是 TensorCore + Pallas/Mosaic，
       **不是 SparseCore**。 -->
__FIG_TPU_FIX__

<div class="note ok"><p>⭐⭐ <b>「能不能在卡里边完全算完、要不要发回 CPU」这一问，答案是分两层的</b> ——&nbsp;
  <b>别答成一个字。</b></p>
<table>
<thead><tr><th>哪一层的决定</th><th>谁来算</th><th>频率</th><th>为什么放在这一层</th></tr></thead><tbody>
<tr><td><b>批次级</b>：这一步有哪些请求、各自多长、页表长什么样</td><td><b>host CPU</b>（服务框架）</td><td>每步一次</td><td>它本来就是调度器的产物，而且一步只算一次，摊到几千个 token 上可以忽略</td></tr>
<tr><td><b>token 级</b>：这个 query 要读哪 2048 条、对应哪些 HBM 地址</td><td><b>卡上的标量单元</b></td><td>每 token 每层</td><td><b>发回 host 是不可能的</b> ——&nbsp;一次 PCIe 往返以微秒计，而这一步的预算是几十微秒</td></tr>
</tbody></table>
<p>⭐ 所以准确的说法是：<b>top-k 那个「决定」不出卡；出卡的只有本来就在 host 上的批次级元信息。</b>
  <em>⛔ 不要说成「全在卡上算」——&nbsp;页表是 host 给的；也不要说成「要发回 CPU」——&nbsp;
  逐 token 的地址计算发回去一次就废了。</em></p></div>

<div class="note info"><p>⭐⭐ <b>跨层共享那一支（§6.5b 的 IndexShare / IndexCache），
  在 TPU 上比在 GPU 上更值钱</b> ——&nbsp;这是一条本课的推导，写清楚它多省的是什么：</p>
<ul>
<li><b>GPU 上省的是</b>：indexer 那部分 FLOPs（GLM-5.2 报 1M 下每 token 降 <b>2.9×</b>，见 §6.5b）</li>
<li><b>TPU 上还额外省三样</b>：① 动态元信息的标量计算<b>只做一次</b>，后面几层直接复用；
  ② 几层的 gather 模式<b>完全相同</b>，DMA 描述符可以重用，不必每层重编一遍；
  ③ <b>动态决定的「次数」本身降了四倍</b></li>
</ul>
<p>⭐ 最后那条是这一节真正想留下的判据：
  <b>在一台 static-first 的机器上，动态性的<em>次数</em>本身就是成本 ——&nbsp;
  不只是每次动态有多贵。</b>
  <em>⚠️ 「TPU 上额外更值钱」是本课从 RPA 那篇描述的机制推出来的，
  <b>没有公开的对照实测</b>；GLM-5.2 的 2.9× 是 FLOPs 口径、且不是在 TPU 上测的。</em></p></div>

<!-- ⭐⭐ 2026-09-14 故事线审计 R25：这一节原来以「一条判据 ＋ 一条免责声明」收尾。
     ⛔ 免责声明该留（它是诚实），但**不该是这一节留在读者脑子里的最后一句**。 -->
<p>⭐ <b>段尾</b>：这一节把全部机制放回了它们出生的那台机器上 ——&nbsp;
  <em>你会发现<b>同一个聪明办法，换台机器就要重新算一遍值不值</b>。</em></p>
<p>⭐⭐ <b>下一节只剩最后一件事：回到开场那句话，
  把我们许下的那笔账算完。</b></p>
<hr>
</div></section>
<section id="s十一"><div class="wrap"><div class="stn"><span class="badge">第 十一 节</span><h2>收尾：把谱系放回时间线</h2></div>
<p>前面那些节讲的是<b>谱系</b>（可迁移的判断框架），这一节是<b>时间线</b>（记忆的挂钩）。 顺序不能反 —— 先给框架，时间线才有意义；先给时间线，框架就变成了流水账。</p>

<!-- ⛔⛔ 2026-09-13 夜间 R18。这里原来是一份 **ASCII 画的时间线** ——
     而**同一条线在开篇已经有图版**（fig3-chronicle）。
     ⭐ 两处各存一份，而且 ASCII 版正是本课自己判定要淘汰的形式
       （「想画图但手边只有文本」的产物）。
     → 删掉副本，指回那张图；这一节只留**图给不了的东西**：
       「让学生自己读出这条线的形状」这个**教法**。 -->
<div class="note ok"><p>🖥 <b>回到开篇那张编年史图</b>（三条泳道那张）——&nbsp;
  这一节不需要新画面，需要的是一个<b>新问法</b>。</p></div>
<p>⭐ <b>试着自己先从那条线上读出三件事，再往下看</b>：</p>
<ol><li><b>前半段是单点突破，后半段全是组合。</b> 2025 年之后没有哪个模型只用一招</li><li><b>"推理期的补丁"逐年变成"训练期的架构"</b> —— NSA 的 native、DSA 的训练阶段、 K3 的从头混合训练，是同一个趋势的三次出现</li><li>每一步都是在<b>修上一步暴露出来的具体毛病</b>，不是凭空发明。 所以下一步大概率也是在修今天这批方案暴露的毛病 —— 那么今天这批的毛病是什么？（留给学生，也留给下一版课件）</li></ol>

<!-- ⭐⭐⭐ 2026-09-14 故事线审计 R22。审计抓到的第一条、也是最硬的一条：
     **封面立了一个极其具体的承诺 —— 那 512 倍 —— 而它在正文里出现 0 次。**
     ⛔ 这不是少写了一段，是**故事结构上的失约**：
       观众带着这个问题进场，散场时没人把答案给他。挂在墙上的枪，一整堂课没响。
     ⭐ 这一小节就是那一枪，而且是用本课自己的提问顺序打的（§九 那条）。 -->
<h3>11.1　最后一件事：回到封面那 512 倍</h3>

<p>开场我们说：<b>2020 年的 GPT-3 记 2048 个 token，今天的模型记 100 万</b>
  ——&nbsp;这一讲讲的就是这 <b>512 倍</b> 是怎么换来的。
  <em>现在把这笔账算完。</em></p>

__FIG_GUN__

<div class="note ok"><p>★ <b>那 512 倍，是这么换来的</b></p>
<p><b>不是内存变大了。</b>显存这一笔，是把每个 token 的开销<b>压了 227.5 倍</b>
  （旋钮① 的 56.9× × 混合 3:1 的 4×）——&nbsp;
  要还 512 倍，还上了 227.5 倍，<b>剩下那 2.25 倍，才是真正多买的硬件</b>。</p>
<p><b>也不是带宽变快了。</b>带宽这一笔，是干脆不读了
  ——&nbsp;稀疏注意力每步固定只看 2048 个，
  <b>上下文涨了 512 倍，它一个都没多读。</b></p>
<p>⭐⭐ 一句话收：<b>这六年真正变的不是机器，
  是「一个 token 到底该花多少钱」这件事被重新定价了。</b></p></div>

<div class="note ok"><p>★ <b>还有一条线，到这儿也该合上了</b></p>
<p><a href="#s一">§1.4</a> 我们认了一个主角 ——&nbsp;<b>那个 <code>S</code></b>，
  张量形状里唯一会越变越长的一维。</p>
<p>之后每一节做的都是同一件事：旋钮① 让 S 前面的<b>系数变小</b>，
  旋钮② 让每步<b>读到的 S 变少</b>，旋钮③ 干脆让 S <b>从形状里消失</b>。
  <em>混合是把三个答案摆在不同的层上。</em></p>
<p>⭐⭐ 所以下次再看到一个没听过的注意力名字，你只需要问它一句：
  <b>你是在跟这个 S 讨价还价，还是打算把它请出去？</b>
  <em>——&nbsp;答得上来，它就已经在你这张地图上了。</em></p></div>

<div class="note info"><p>📌 <b>顺手看一眼这笔账是怎么算出来的 ——&nbsp;它就是这门课的方法</b></p>
<p>它不是一笔账，<b>是两笔</b>：显存账问「一个人要占多少」，带宽账问「每走一步要读多少」。
  而三个旋钮<b>各还各的那一笔</b> ——&nbsp;
  旋钮① 只还显存，旋钮② 只还带宽（<b>它一个字节都不省显存</b>），
  旋钮③ 两笔一起还。</p>
<p>⭐ 所以 <a href="#s九">§九</a> 那句「问『省了多少』之前，先问『省的是哪一样』」
  不是一句方法论口号 ——&nbsp;<em><b>不分开问，这两笔账根本对不上。</b></em></p></div>
<hr>
</div></section>
<!-- ⭐⭐⭐ 2026-09-14 故事线审计 R23：这一块原来是**第十二节**，于是整堂课的
     最后一节是「我们不讲什么」，最后一段是出处清单的说明 ——
     ⛔ **等于电影最后一幕是片尾免责声明。**
     ⭐ 内容该留（它防跑题、也是对读者的交代），但**不该是最后一句话**。
     → 降级成折叠附录，跟出处清单并排。课现在结束在 11.1 那一枪上。
     ⚠️ 目录是从「第 N 节」的 section 结构自动抓的，所以它会自动从目录里消失
       —— 这正是想要的，顺手也把节表里那一行删了。 -->
<section id="s十二"><div class="wrap">
<details class="aside"><summary>📎 <b>附：这个专题明确不讲什么（以及为什么）</b>
<em>（想知道边界在哪、或者在找某个主题去了哪一讲，点开）</em></summary>
<ul><li><b>MLA 逐步的矩阵推导</b> → 在<a href="专题01-一个-Token-的一生.md">专题一</a>第 2 步。 ⭐ 分工是这样定的：专题一讲"V3 这一个模型里它怎么算"， 这里讲"为什么会有它、它在谱系里站哪、它换走了什么"。 这里只用一张图复述结论（576 = 512 + 64），不重讲推导 —— 重讲会占掉 5 分钟，而这 5 分钟买不到任何新东西</li><li><b>注意力的 kernel 怎么写</b> → 实现细节在<a href="专题07-性能调优与工具链.md">专题七</a></li><li><b>序列并行 / Context Parallelism 怎么切</b> → <a href="专题05-并行策略.md">专题五</a>。 这里只说"线性注意力的 CP 跟标准 CP 不一样"，不展开</li><li><b>prefill / decode 的形状差异</b> → <a href="专题06-推理.md">专题六</a></li><li><b>各家模型的完整参数表</b> → <a href="专题09-最新开源模型对比.md">专题九</a>。 这里给的每个数字都只为说明一个机制，<b>不做横向评测</b></li><li><b>Mamba / SSM 那一支</b> —— 跟线性注意力是近亲，但它自成体系。 这门课的听众用不上，<b>明确不讲</b>（问到就说一句"同一个思路的另一个分支"）</li></ul>
</details>
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

<!-- ⭐⭐ 2026-09-13 现场点的：「引用的那些论文得在教材里边，把可点击的 link
     都放里边，有愿意多学的人可以去点开看。」
     ⛔ 做法是**后处理自动加**，不是手写（course_links.py）——
       这一讲有几十处 arXiv 编号，而且每加一个机制就会多几处；
       ⭐ 手写等于每次都要记得加，而**忘了不报错**。
     ⭐ 图里的出处也一并处理了：那边用 SVG 自己的链接元素
       （HTML 的链接进不去 SVG 的文本流），
       并且加了下划线 —— SVG 的链接不会自动变色，不标出来没人知道它能点。 -->
<div class="note ok"><p>⭐ <b>整页所有 arXiv 编号都是可点的</b>
  ——&nbsp;正文里的、表里的、连图上那些小字出处，点一下直接开论文。
  <em>（图里的链接带下划线 ——&nbsp;SVG 的链接不像网页那样自动变蓝，
  所以特意标出来。）</em></p>
<p>⚠️ <b>几个不是 arXiv 的，在表里单独挂了链接</b>：
  DeepSeek-V3.2-Exp 技术报告（GitHub）、NVIDIA 的 RNN 性能指南、
  张量形状记号沿用的 <a href="https://jax-ml.github.io/scaling-book/" target="_blank" rel="noopener">How to Scale Your Model</a>。
  <em>各家模型的 config 没挂链接 ——&nbsp;它们在各自的 Hugging Face 仓库里，
  版本会动，<b>写死一个链接迟早指到改过的那一版</b>。</em></p></div>

<table>
<thead><tr><th>要什么</th><th>在哪</th></tr></thead><tbody>
<tr><td>MQA / GQA</td><td>arXiv <b>1911.02150</b> / <b>2305.13245</b></td></tr>
<tr><td>MLA</td><td>DeepSeek-V3, arXiv <b>2412.19437</b> <b>sec. 2.1 + 4.2</b>（超参那段给了 <code>n_h/d_h/d_c/d_h^R</code> 的准确值）</td></tr>
<tr><td>Gated MLA / K3 全貌</td><td>Kimi K3, arXiv <b>2607.24653</b> <b>sec. 2.1.2</b>、表 1（93 层 / 69 KDA + 24 MLA / 2.78T-104.2B）</td></tr>
<tr><td>SWA</td><td>Mistral 7B, arXiv <b>2310.06825</b>（窗口 4096）</td></tr>
<tr><td>Attention sink</td><td>StreamingLLM, arXiv <b>2309.17453</b>（4 个 token / 400 万 / 22.2×）</td></tr>
<tr><td>NSA</td><td>arXiv <b>2502.11089</b>（三支路 + 门控；64k 下 11.6× / 9.0× / 6.0×）</td></tr>
<tr><td>DSA + Lightning Indexer</td><td><b><a href="https://github.com/deepseek-ai/DeepSeek-V3.2-Exp" target="_blank" rel="noopener">DeepSeek-V3.2-Exp 技术报告</a> sec. 1–2.1</b>（ReLU 打分 / FP8 / k=2048 / 稠密预热阶段 / KL 对齐）——&nbsp;⚠️ <b>不是后来那篇 arXiv 2512.02556</b>（《DeepSeek-V3.2》），两者节号对不上；本讲图里核的数全部来自 Exp 那份。<b>我们有一手实测</b></td></tr>
<tr><td>CSA / HCA</td><td>DeepSeek-V4, arXiv <b>2606.19348</b> <b>sec. 2.3 + 2.3.4</b>（m=4 / m′=128 / top-k / 27%·10% / 2%）</td></tr>
<tr><td>线性注意力谱系</td><td><b>2006.16236</b>（线性）→ <b>2102.11174</b>（delta rule, 2021）→ <b>2406.06484</b>（可并行化）→ <b>2412.06464</b>（GDN）→ <b>2510.26692</b>（KDA）</td></tr>
<tr><td>混合配比</td><td><b>2510.26692</b>（3:1 + 消融）、Ling-3.0 模型卡（3:1 / 5:1）、<b>2507.06457</b>（建议 3:1～6:1）</td></tr>
<tr><td>FlashAttention</td><td>arXiv <b>2205.14135</b> + TPU 侧 Splash Attention 实测（<code>tpu/</code> 下多处）</td></tr>
<tr><td><b>MHA 本体</b>（§一）</td><td>Vaswani et al. 2017, arXiv <b>1706.03762</b> ——&nbsp;<b>sec. 3.2 / 3.2.1 / 3.2.2 / 3.2.3</b> ＋ <b>表 1</b>；四条原话见下方折叠</td></tr>
<tr><td><b>KV cache 被点名成瓶颈</b></td><td>Shazeer 2019, arXiv <b>1911.02150</b>（MQA 那篇）——&nbsp;<b>「memory-bandwidth cost of repeatedly loading the large keys and values tensors」</b></td></tr>
<tr><td><b>RNN 一支</b>（§零）</td><td>Elman 1990《Finding Structure in Time》；Bengio, Simard, Frasconi 1994；Hochreiter &amp; Schmidhuber 1997；Cho et al. 2014；Bahdanau et al. 2014, arXiv <b>1409.0473</b></td></tr>
<tr><td><b>只有线性依赖才扫得动</b></td><td>Martin &amp; Cundy 2018, arXiv <b>1709.04057</b>（ICLR'18）——&nbsp;实测最高 9× 加速</td></tr>
<tr><td><b>RNN 在硬件上为什么慢</b></td><td><a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-recurrent/" target="_blank" rel="noopener">NVIDIA《Recurrent Layers User's Guide》</a>——&nbsp;<b>「a GEMM with one dimension of one」</b>、<b>「can combine these GEMMs over the minibatch size, but not over different sequence steps」</b></td></tr>
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
<tr><td><b>§零那张「解码时又变回 RNN」图 Ⓒ 那一行不是我们的推论</b>，是原文</td>
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
    2020 年的 GPT-3 只能记住 <b>2048</b> 个 token；今天的模型记 <b>100 万</b>。<br>
    <em>——&nbsp;这六年的注意力演进，讲的就是这 <b>512 倍</b> 是怎么换来的。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    MLA、GQA、SWA、DSA、NSA、CSA、DeltaNet、GDN、KDA……
    名词多到像各搞各的，<b>但只有三个旋钮可以拧</b>。
    这一讲不按名字讲，按<b>它们各自是怎么被逼出来的</b>讲 ——&nbsp;
    每一个都回到一手论文与 config 核过。
  </p>
  <div class="chips">
    <span class="chip">前置 <b>专题一 · 专题二</b></span>
    <span class="chip">主线 <b>三个旋钮</b> ＋ 一条硬件线</span>
    <span class="chip">含 <b>我们自己的 v7 实测</b></span>
    <span class="chip">⏱ <b>70′ / 113.5′ / 136.5′ 三档</b></span>
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
        '<b>它意味着「把模型做小」根本救不了 RNN。</b>'
        '⭐⭐ <b>然后看最下面那根横轴 —— 那是「两头堵死」的图形版</b>：'
        '最粗的那根竖线是地板（313，站右边才算喂饱），上面两个蓝三角是显存的天花板'
        '（<b>序列一长它往左走</b>），而它们之间那块<b>画满斜线的地方</b>就是'
        '<b>「硬件要你站进来，显存不让你进来」</b>。'
        '⚠️ 三角的位置是示意、图上没标数字：<b>这张图只承诺方向，不承诺数值</b>。'
        '⛔ 横轴是线性的不是对数 —— 换成 log，batch 1 和 313 会挤在一起，'
        '<b>整张图唯一的论点当场就看不见了</b>。'),

    "__FIG_RNN_DECODE__": ("fig-rnn-decode", "fig3-rnn-decode.svg",
        'topic03-fig-rnn.py',
        '⭐⭐ <b>本节的落点，看最后那一格就够。</b>上面那排 W 两边共用，'
        '所以它区分不了任何事；<b>真正分开两条路的只有下面那两排条的宽度</b> ——'
        '红的从头到尾一样宽（状态 h 跟上下文多长无关），'
        '紫的每一步都比上一步长，到 tn 直接冲出格子（KV cache）。'
        '<b>整个专题三都发生在紫色这一排上。</b>'),

    "__FIG_RNN_PAIN__": ("fig-rnn-pain", "fig3-rnn-pain.svg",
        'topic03-fig-rnn.py',
        '⭐ <b>这一页是整个专题的路标</b>：后面每一个变体都能追回到这三条里的某一条。'
        '<b>最底下那一格是主脊</b> —— 第①条后来被反着又走了一遍，'
        '而「必须线性到能被 scan」就是线性注意力那些公式的由来。'),

    # ── §六 旋钮②（2026-09-12 加）─────────────────────────────────
    # ⭐⭐ 这一节自己写着一句话，直接论证了这张图该存在：
    #   「attention sink 那个 bug **从公式上完全看不出来，
    #     只有把注意力矩阵画出来看才发现**。」
    #   ⛔ 五种方案用文字并列讲，听完只记住五个名词；画成五张 mask 并排，
    #     **亲缘关系和演进方向一眼就出来**：砍成一条带 → 补回停车位
    #     → 学着挑 → 先压再挑。
    "__FIG_KNOB2__": ("fig-knob2", "fig3-knob2.svg", 'topic03-fig-knob2.py',
        '⭐⭐ <b>五张 mask 并排，这一支的共同结构就出来了：先用一个便宜得多的办法'
        '决定「看哪些」，再只对那些做主注意力。</b>'
        '<em>区别只在那个「便宜的办法」是写死的规则，还是学出来的。</em>'),

    # ── §10.2 / §10.3 落到 TPU（2026-09-12 TPU 轮 R27–R31 加）──────
    "__FIG_TPU_GAP__": ("fig-tpu-gap", "fig3-tpu-gap.svg",
        'topic03-fig-tpu-gap.py',
        '⭐⭐ <b>先看清楚是哪两件事对不上：TPU 的三条硬约束（静态形状、tiled 粗粒度布局、'
        '偏好规整访存）对上现代注意力的三个动态性来源（ragged、分页 KV、运行时 top-k）。</b>'
        '<em>不摆清楚这个错配，后面那些 kernel 技巧看起来就只是一堆技巧。</em>'),

    "__FIG_TPU_FIX__": ("fig-tpu-fix", "fig3-tpu-fix.svg",
        'topic03-fig-tpu-fix.py',
        '⭐⭐ <b>三招的共同形状：把「一个动态」换成「一批静态」。</b>'
        '<b>而运行时那个 top-k 决定不用出卡 —— 它用的是 FlashAttention 阶段本来就闲着的标量单元。</b>'
        '<em>SparseCore 架构上正对口（它天生支持数据相关的控制流与访存），'
        '但公开的那套 TPU 生产注意力 kernel 用的还是 TensorCore —— 这条留在「看起来对但未被公开验证」。</em>'),

    # ── §九＋§十 落点（2026-09-13 夜间 R15 加）─────────────────────
    "__FIG_LANDING__": ("fig-landing", "fig3-landing.svg", 'topic03-fig-landing.py',
        '⭐⭐ <b>注意力的变体史，是一部在显存 / 算力 / 访存规整度之间反复搬家的历史 —— '
        '而顺序不是随机的：先搬能算的，最后才搬算不出来的。</b>'
        '<b>右边那格是全课最该带走的一句：注意力只是账单的一部分，'
        '任何一个倍数都必须带上「在多长的上下文下」。</b>'),

    # ── §八 混合（2026-09-13 夜间 R14 加）──────────────────────────
    "__FIG_HYBRID__": ("fig-hybrid", "fig3-hybrid.svg", 'topic03-fig-hybrid.py',
        '⭐⭐ <b>配比是一条轴，而两头都不好。</b>'
        # ⛔ 2026-09-14：这里原来跟着图一起写「两端各 1/8 一个都没有」——**假的**，
        #   轴是 1:1→8:1，左端 1:1 站着两个点。图和讲义在 R48 都改了，
        #   **这条图注漏了** ——&nbsp;⭐ 一句话住在三个地方（图 / 图注 / 讲稿），
        #   改一处不叫改完，得三处一起查。
        '<b>中间那格一个点就是一个模型</b> —— 14 个点堆起来，'
        '<b>最高的一摞在 3:1（十四家占五家）</b>、往右没有一家超过 7:1，'
        '都不用数；点的颜色只编码一件事：<b>便宜的那层是线性，还是滑窗 / 局部</b>'
        '（按 <a href="#s八">§8.4</a> 逐家核过的旋钮分的）。'
        '<b>最左边 1:1 那两个清一色是滑窗族 —— 做线性混合的没有一家敢一比一。</b>'
        '<b>纯线性那一头不是「实测不行」—— §二 那条 L2M 条件说它的状态必须随长度变大。</b>'
        '<em>⚠️ 但「变大」的办法不止混合一种：论文自己给的另一条是按长度整个放大模型。'
        '混合是工程上选的那条，不是定理逼出来的那条。</em>'
        '<em>而左端那个反直觉结果更值得讲：Kimi 的消融里 0:1（纯全注意力）'
        '反而表现不好 —— 加线性层不只是省钱。'
        '那张消融表长什么样、它又<b>问不出</b>什么，见 <a href="#s八">§8.2</a>。</em>'),

    # ── §5 那块「复制矩阵」（2026-09-14 R29 加）────────────────────
    # ⭐ 调研结论：苏剑林三篇正文零配图，TransMLA Figure 1 只画了 repeat
    #   这个**操作**。把**矩阵本身**一格一格画出来，全网这一格是空的。
    "__FIG_COPY_MATRIX__": ("fig-copy-matrix", "fig3-copy-matrix.svg",
        'topic03-fig-copy-matrix.py',
        '⭐⭐ <b>「MLA 就是给 KV 做低秩分解」这句话没错，但它区分不了 GQA 和 MLA '
        '—— 因为 GQA 也是低秩投影。</b>'
        '<b>分界线在低秩<em>之后</em>那一步：GQA 用「分割 ＋ 复制」把 c 凑成各头的 K，'
        '而分割和复制本身就是线性变换 —— 它对应的就是图里那块'
        '<em>只有 8 个 1、其余全为 0、而且不可训练</em>的矩阵。'
        'MLA 只是把同一个位置、同一个形状的矩阵松开让它学。</b>'
        '<em>⭐ 四个成员在图里列数完全相同（都是 8），只有行数（＝ cache 宽度）'
        '和格子内容不同：MHA ＝ 单位阵、MQA ＝ 一份抄满、GQA-2 与 MLA 同为 4×8。</em>'
        '<em>⛔ 但矩阵一松开，各头的 K 又各不相同，cache 会涨回 MHA 那么大 —— '
        '所以必须接 <a href="#s五">§5.4b</a> 的「吸收」。这张图是它的前提，不是替代。</em>'),

    # ── §8.2 那张消融表（2026-09-14 R28 加）────────────────────────
    # ⛔ 这张图是为了修本节自己的一处事实错误而画的：原文写着论文
    #    「没公开数值」，而 arXiv 2510.26692v2 Table 1 把五个配比的
    #    训练 / 验证 PPL 全列出来了。详见图脚本头部。
    "__FIG_RATIO_GRID__": ("fig-ratio-grid", "fig3-ratio-grid.svg",
        'topic03-fig-ratio-grid.py',
        '⭐⭐ <b>「3:1 最好」底下是一张五行的表，数值全都公开了 —— '
        '而把它画出来之后，多出来三件原文那句定性描述看不到的事。</b>'
        '<b>一：纯全注意力那一头（0:1）排第四，<em>不是最差</em>；最差的是另一头的 15:1。</b>'
        '<b>二：3:1 和 7:1 的<em>训练</em> PPL 完全相同（都是 9.23），'
        '要到验证集上才分得开。</b>'
        '⭐⭐ <b>三：这五个配比不是挑出来的，是 16 除出来的 —— '
        '消融模型只有 16 层，每组 (r+1) 层必须整除 16，'
        '所以 r 只能取 0 / 1 / 3 / 7 / 15。'
        '「为什么是 3 不是 4」这个问题，那张表<em>问不出来</em>。</b>'
        '<em>⚠️ 「(r+1) 必须整除 16」是本课对该表做的算术观察，'
        '不是论文给的理由 —— 论文没有解释为什么选这五个。</em>'),

    # ── §8.2b 守恒的是比值还是层数（2026-09-14 夜间 R31 加）────────
    # ⛔ 这一格答的是 R28 画完消融表留下的悬案，顺带证伪了 §8.1 那句话的字面读法。
    #   数据全部来自 14 个模型 config.json 的逐层字段，相关系数脚本当场算。
    "__FIG_RATIO_OR_COUNT__": ("fig-ratio-or-count", "fig3-ratio-or-count.svg",
        'topic03-fig-ratio-or-count.py',
        '⭐⭐ <b>把「几层就够」当成一个可证伪的预测，然后去数 14 个模型的 '
        'config.json。</b><b>点全落在 3:1 那条斜线上，不落在那条平线上 ——&nbsp;'
        '守恒的是比值，绝对条数随深度线性涨。</b>'
        '<em>⭐ 更准的说法是两条叠加：主体按比例铺，外加几个按「位置」钉死的'
        '全局层（Kimi 的「末层必为全局」）——&nbsp;后者正好解释了 2.857 和 2.875 '
        '这两个别扭的小数。</em>'),

    # ── §7.4 前半：凭什么可以分块（2026-09-14 夜间 R30 加）─────────
    # ⛔ 这一格补的是 fig3-assoc 图注里那句「所以真实实现是分块」——
    #   全讲说了三次「mask 挡住结合律」，一次也没画过它是怎么被绕开的。
    # ⭐ 本图自己的装置：**用「有没有画出格子」当编码** —— 对角块一格一格
    #   （真的物化了 C×C），块间一整块 ＋ 一根竖条（压成状态，根本没物化）。
    #   调研过的四张同题图（GLA Fig3 / Mamba-2 Fig5 / snowchord / rudrite）
    #   都没这么干，详见脚本头。
    "__FIG_TWO_BRACKETS__": ("fig-two-brackets", "fig3-two-brackets.svg",
        'topic03-fig-two-brackets.py',
        '⭐⭐ <b>同一张矩阵画了两遍：不分块时右括号插不进去，'
        '切成块之后两种括号同框。</b>'
        '<b>关键不是「块小所以算得动」——&nbsp;是块间那些块里根本没有 mask，'
        '结合律在那儿从来没被挡住过。</b>'
        '<em>⚠️ 被挡住的是「结合律」这个特例，不是「重排」本身：'
        '换成更一般的张量缩并顺序，mask 是能被吸收进去的。'
        '分块是通常实现走的路，不是数学上的唯一解。</em>'),

    # ── §7.4 chunkwise（2026-09-13 夜间 R13 加）────────────────────
    "__FIG_CHUNKWISE__": ("fig-chunkwise", "fig3-chunkwise.svg",
        'topic03-fig-chunkwise.py',
        '⭐⭐ <b>第三个旋钮本来是为了省 —— 把平方降成线性；结果先丢掉的是并行度。</b>'
        '<b>解法是块内并行、块间串行：并行度从 1 变成 C，串行步数从 L 变成 L/C，'
        '而且数学一点没改。</b>'
        '<em>块长两头被夹住 —— 小了算力吃不满，大了放不进片上内存。</em>'),


    # ── §七 A_t 形状谱系（2026-09-13 夜间 R12 加）──────────────────
    # ⛔ 这一格换掉了 7.2/7.2b 的两张表 —— 它们讲的其实是「矩阵长什么样」。
    "__FIG_AT_GALLERY__": ("fig-at-gallery", "fig3-at-gallery.svg",
        'topic03-fig-at-gallery.py',
        '⭐⭐ <b>所有这些方法长成同一个递推，区别只有一处：允许 A_t 长什么样。</b>'
        '<b>八种形状并排，谱系、亲缘、代价一眼全在。</b>'
        '<em>最能说明问题的是 Mamba-2 主动往回退到最简的 a·I —— '
        '正因为退了，才证得出跟线性注意力的对偶，才能吃上 Tensor Core。</em>'),


    "__FIG_DUALITY__": ("fig-duality", "fig3-duality.svg",
        'topic03-fig-duality.py',
        '⭐⭐ <b>递推读法和矩阵读法是同一个计算，差别只在先算哪一步。</b>'
        'Mamba-2 管这个叫 “a different contraction ordering” —— '
        '也就是本讲反复出现的「换括号」。'
        'M ＝（Q Kᵀ）∘ L 这个骨架下，<b>你换的从来只有 L</b>。'
        '<em>图里那个等号是脚本用 numpy 当场验的 —— '
        '两条路算出来逐元素相等，不是打比方。</em>'),

    # ── §七 记事板比喻（2026-09-13 夜间 R11 加）────────────────────
    # ⭐ 这个比喻是现场给的，而且现场说它「对理解非常重要」。
    #   图的任务不是复述比喻，是**把它钉到式子上**。
    # ⭐ 装置偷自 Google Research 的 Performer 博客：**括号画成彩色虚线框、
    #   矩阵按真实比例画**。那个 S×S 的大方块「没被建出来」是看出来的。
    "__FIG_CHICKEN__": ("fig-chicken", "fig3-chicken.svg",
        'topic03-fig-chicken.py',
        '⭐⭐ <b>三条破法是真正不同的三条</b>：师徒（训一个便宜的学生学老师的排序）·'
        '一份算两用（压缩分支的分数本来就要算，直接当路由）·'
        '降维打击（先把序列压短 4 倍，鸡生蛋没破但那只鸡小了 4 倍）。'
        '<em>⭐ 顺着这个死结往下一步，「为什么必须整块取」就不用单独讲了 ——'
        '便宜的近似打分只能按块做，整块取是这个死结的直接推论。</em>'),
    # ── §1.3b 一个头装得下多少（2026-09-13 夜间 R14 加）────────────
    # ⭐ 这是六份调研报告里唯一一条**四个视角都提到**的空白。
    #   ⛔ 图里每一个数字都是脚本当场算的（numpy + 断言），一个都不是引来的 ——
    #     因为这套说法在别处只有定性描述，没人给数。
    "__FIG_CAPACITY__": ("fig-capacity", "fig3-capacity.svg",
        'topic03-fig-capacity.py',
        '⭐⭐ <b>这张图的重点在②和③的对照</b>：② 说容量几乎是白捡的'
        '（根数涨 78 倍，最挤的一对才挤 9°），③ 说这笔便宜的价钱是串扰'
        '——&nbsp;<b>而价钱是按数量收的</b>。'
        '<em>⚠️ ③ 的 logit = √d·cos 是推导来的不是测来的，读趋势别读绝对值。</em>'),
    "__FIG_LOWRANK__": ("fig-lowrank", "fig3-lowrank.svg",
        'topic03-fig-lowrank.py',
        '⭐⭐ <b>两栏都是真数字，你可以自己核</b> —— 秩够的时候拆开再复原，'
        '每一个数一模一样（白送）；秩不够硬压，红格子就是差出来的（赌）。'
        '<em>⚠️ 右栏那个误差是 SVD 给的理论下界，不是「没调好」。</em>'),
    "__FIG_ASSOC__": ("fig-assoc", "fig3-assoc.svg",
        'topic03-fig-assoc.py',
        '⭐⭐ <b>同一个乘法，只是把括号挪了个位置</b> —— 左边被迫造出一个'
        '句长×句长的大方块，右边中间那块只有头维×头维，<b>跟句子多长完全无关</b>。'
        '<em>⛔ 但因果 mask 会把这个重排挡住 —— 逐元素乘那张下三角，'
        '恰恰就是挡住你用结合律的东西。所以真实实现是分块：块内按左边算，'
        '块间才用右边（见 §7.4）。</em>'),
    "__FIG_NOTEPAD__": ("fig-notepad", "fig3-notepad.svg",
        'topic03-fig-notepad.py',
        '⭐⭐ <b>三种写法就是这一支的三代：疯狂往里写（纯加，L &gt; d 必撞车）→ '
        '先把这个方向上的旧内容叉掉再写（delta rule）→ 选择性地叉（门控）。</b>'
        '<b>那个 (I − β k kᵀ) 就是「叉掉」。</b>'
        '<em>再换个角度看，它等价于对 ½‖Sk−v‖² 做一步 SGD —— '
        '状态不再是缓存，是一个边跑边被训练的小模型。</em>'),

    # ── §四 第二个轴：什么时候改（2026-09-13 夜间 R10 加）──────────
    "__FIG_WHEN_AXIS__": ("fig-when-axis", "fig3-when-axis.svg",
        'topic03-fig-when-axis.py',
        '⭐⭐ <b>三个旋钮说的是「改什么」，这一张说的是「什么时候改」—— 两个轴正交。</b>'
        '<b>事后那一列便宜、可开关，但天花板就是「接近原来那个」；'
        'native 那一列要重训，却能持平甚至超过。</b>'
        '<em>⚠️ 注意：这说的是方法「这次被怎么用」，不是它本身的属性。</em>'),

    # ── §二 信息账的严格版本（2026-09-13 夜间 R9 加）───────────────
    # ⭐⭐⭐ 全讲理论上最重要的一张：把「远处可以少看」换成一条能算的规律，
    #   而且**提前回答了 §八「为什么必须混合」**。
    "__FIG_INFO_LAW__": ("fig-info-law", "fig3-info-law.svg",
        'topic03-fig-info-law.py',
        '⭐⭐ <b>两点互信息随距离幂律衰减（所以语言不能用马尔可夫近似），'
        '而两半之间的双部互信息随长度幂律增长。</b>'
        '<b>L2M 条件：模型历史状态的维度必须至少以同样的幂律增长 —— '
        '这就是为什么固定大小状态那一支必须混着用。</b>'),

    # ── §一 为什么是 softmax / K V 为什么分家（2026-09-13 夜间 R8）──
    "__FIG_WHY_SOFTMAX__": ("fig-why-softmax", "fig3-why-softmax.svg",
        'topic03-fig-why-softmax.py',
        '⭐⭐ <b>softmax 的三条要求（非负、和为 1、可导）全是「加权平均」逼出来的。</b>'
        '<b>所以 DSA 的索引器敢换成 ReLU —— 它只拿分数去排序，不做加权平均。</b>'
        '<em>右边：k 是书脊、v 是内容，检索天生非对称 —— 按 A 去找，取回 B。</em>'),

    # ── §一 注意力是怎么被发明出来的（2026-09-13 夜间 R7 加）───────
    "__FIG_ATTN_INVENTED__": ("fig-attn-invented", "fig3-attn-invented.svg",
        'topic03-fig-attn-invented.py',
        '⭐⭐ <b>三步，每一步都在修上一步的一个具体毛病：一个向量装不下 → 软对齐；'
        '小网络打分太慢 → 点积；既然能直连 → 扔掉循环，而扔掉之后分辨率变糙 → 多头补回来。</b>'
        '<em>中间那格是本课主线第一次露面 —— 点积胜出不是因为它更准，'
        '是因为它能写成矩阵乘，这是论文原话。</em>'),

    # ── §六 滑窗与 attention sink（2026-09-13 夜间 R6 加）──────────
    "__FIG_SWA_WHY__": ("fig-swa-why", "fig3-swa-why.svg", 'topic03-fig-swa-why.py',
        '⭐⭐ <b>一个侦探故事：先有办法（层数是免费的射程，32 层 × 4096 = 131,072），'
        '再出事故（困惑度 5.40 → 5158），最后才找到原因（softmax 要求一行加起来等于 1）。</b>'
        '<em>判决性实验在中间那格：把最前面四个 token 换成换行符，照样管用 —— '
        '起作用的是位置，不是语义。</em>'),

    # ── §六 CSA/HCA 压的是什么（2026-09-13 夜间 R5 加）────────────
    "__FIG_CSA_WHY__": ("fig-csa-why", "fig3-csa-why.svg", 'topic03-fig-csa-why.py',
        '⭐⭐ <b>压缩有两个互不相干的方向：竖着压（一个 token 存多少个数）是旋钮①，'
        '横着压（几个 token 合成一条）才是 CSA/HCA。</b>'
        '<b>CSA 压得轻＋挑，看得细但会漏；HCA 压得狠＋全看，不漏但看得粗 —— '
        '两种坏法正好相反。</b>'
        '<em>⚠️ 「所以交错摆着互相兜底」是一个讲得通的解释，'
        '论文只说了采用交错配置、没给理由 —— 讲的时候别说成设计意图。</em>'),

    # ── §六 NSA 三条路被什么逼出来的（2026-09-13 夜间 R4 加）───────
    "__FIG_NSA_WHY__": ("fig-nsa-why", "fig3-nsa-why.svg", 'topic03-fig-nsa-why.py',
        '⭐⭐ <b>NSA 的三条分支不是设计出来的，是被四个坑逼出来的 —— '
        '而那四个坑是论文 §2 自己列的。</b>'
        '<b>最值钱的一条：计算稀疏 ≠ 访存稀疏。</b>'
        '<em>按块选、组内共享，都是为了访存，不是为了精度。</em>'),

    # ── §六 DSA 凭什么只看 2048 个（2026-09-13 夜间 R3 加）─────────
    "__FIG_DSA_WHY__": ("fig-dsa-why", "fig3-dsa-why.svg", 'topic03-fig-dsa-why.py',
        '⭐⭐ <b>「能不能少看」是经验事实（H2O 量到注意力矩阵 95% 以上稀疏）；'
        '「怎么知道该看谁」才是难点 —— 要判断谁重要得先算注意力分数，'
        '而那正是你想省掉的东西。</b>'
        '<em>DSA 的解法不是猜，是让真注意力当老师：冻住模型、保持密集，'
        '训一个便宜的索引器去 KL 拟合主注意力的分布。</em>'),

    # ── §五 砍头这一支怎么想出来的（2026-09-13 夜间 R2 加）────────
    "__FIG_MQA_WHY__": ("fig-mqa-why", "fig3-mqa-why.svg", 'topic03-fig-mqa-why.py',
        '⭐⭐ <b>最下面那一格只有两根轴，先看它们是垂直的</b>：'
        '横轴上那两个点是 2019 年 MQA 论文列给别人的选项（后来叫滑窗、叫稀疏 / 压缩，'
        '也就是今天的<b>旋钮②</b>），纵轴上那条是它自己走的（<b>旋钮①</b>）。'
        '<b>原点那个小方角是全图的论点 —— 而「正交」是论文原话，不是我们事后安的词。</b>'
        '左边那个灰虚线框（限制序列长度）<b>故意画在平面外</b>：它也在那张单子上，'
        '但它不是旋钮，硬塞到某根轴上就是撒谎。'
        '<b>中间那三行是「自由度不是宽度」的论文自带证据：同样只缓存一份 K/V，'
        '保留 8 个 query 头能把困惑度拉回整整 1.0。</b>'),

    # ── §五 MLA 为什么能压缩（2026-09-13 夜间 R1 加）──────────────
    # ⭐ 跟 fig3-knob1 分工清楚：那张答「是什么、省多少」，这张答「凭什么」。
    #   ⛔ 别把两张合并 —— 合并之后「凭什么」一定会被压成一行小字，
    #     而它才是这一节真正教的东西。
    "__FIG_MLA_WHY__": ("fig-mla-why", "fig3-mla-why.svg", 'topic03-fig-mla-why.py',
        '⭐⭐ <b>32,768 个数是从一个 7,168 维的 h 算出来的 —— 一次确定性映射不造信息，'
        '所以 4.57 倍是白送的。</b>'
        '<b>真正的赌注是后面那步：7,168 压到 576，等于强制 K 投影的秩 ≤ 512。</b>'
        '<em>而敢赌是因为「K/V 本来就低秩」在 MLA 之前已经被量过了。</em>'),

    # ── §五 旋钮①（2026-09-12 加）─────────────────────────────────
    # ⭐ 两件事一张图：四种存法的对照（题眼是 MQA 比 MLA 还小），
    #   以及「RoPE 为什么必须单独走一路」—— 讲义写着那是「最容易讲糊的一步」，
    #   而它本质是**一个代数重写成不成立**，画出来比说出来清楚得多。
    # ⭐ 装置来自两篇顶级材料**各一半，而且没人把它们拼起来**：
    #   Fleetwood 画了二进制计数器动画却停在「正弦是它的连续版」；
    #   苏剑林走到了「RoPE ＝ 位置的 β 进制写法」却一张图没画。
    "__FIG_ROPE__": ("fig-rope", "fig3-rope.svg", 'topic03-fig-rope.py',
        '⭐⭐ <b>一个位置 ＝ 一排里程表转盘的读数</b>：最右边转得飞快，往左逐级变慢。'
        '两个位置各自转完之后，<b>每个盘上你只看得出两根针差多少，看不出各自转到哪儿</b>'
        ' —— 这就是「点积自动带相对距离」的全部内容。'
        '<em>⭐ 顺带把长文本那三种做法一句话各自归位：直接外推＝硬往超刻度处读；'
        '内插＝每格走半格；NTK-aware＝换一个进制。</em>'),
    # ── §5.4b 吸收（2026-09-13 夜间 R15 加）───────────────────────
    # ⭐ 全网讲 MLA 的文章几乎都跳过这一步，或者只写一行公式。
    #   ⛔ 而它是 MLA「能不能真省下来」的前提 —— 不吸收，压缩等于白压。
    "__FIG_ABSORB__": ("fig-absorb", "fig3-absorb.svg",
        'topic03-fig-absorb.py',
        '⭐⭐ <b>盯住那块紫色的 W<sup>UK</sup></b>：①里它站在缓存那一侧，'
        '②里它<b>搬到了 q 那一侧</b> ——&nbsp;整张图讲的就是这一次搬家。'
        '<em>⚠️ ③ 那个「上限 128×」是本课自己算的，只数乘加没算访存。</em>'),
    # ── §5.4c 功劳记给谁（2026-09-14 夜间 R46 加）───────────────────
    # ⭐ 跟 fig3-mla-why 分工：那张答「凭什么敢把 K/V 压成低秩」，
    #   这张答「压完了效果好，功劳到底是不是低秩的」——&nbsp;结论是**多半不是**。
    # ⛔ 别把它并进 fig3-mla-why。两张图的结论方向相反，合在一起会互相抵消：
    #   一张在论证「敢赌」，一张在说「赢的钱可能不是从这儿来的」。
    "__FIG_MLA_CREDIT__": ("fig-mla-credit", "fig3-mla-credit.svg",
        'topic03-fig-mla-credit.py',
        '⭐⭐ <b>盯住那条紫色虚线（MLA 的水位）和三级台阶</b>：KV Cache 全程钉死 512，'
        '只把 head_dims 放宽、再拆一小段给 RoPE，<b>第二级就已经踩到线下去了</b>。'
        '<em>⚠️ ~900M dense、16B tokens 的小规模消融 ——&nbsp;方向可信，幅度别往 671B 搬。</em>'),
    # ── §五 开篇 接线图（2026-09-14 R59 加）───────────────────────
    # ⭐ 跟紧随其后的 fig3-knob1 分工：这张答「省在哪」（零数字，看形状），
    #   那张答「省多少、值不值」（散点，每个数都核过）。
    # ⛔ 别往这张图里加数 —— 一出现数字读者就开始算，就不看形状了。
    "__FIG_WIRING__": ("fig-wiring", "fig3-wiring.svg",
        'topic03-fig-wiring.py',
        '⭐⭐ <b>只看最下面那排行李箱</b>：四家该做的乘加几乎一样多，'
        '差别全在「这段对话得一直背着多少」。'
        '<em>⭐ MLA 那格多看两眼 ——&nbsp;<b>箱子里那个不是 K/V</b>，'
        '虚线的那几个是用的时候现场展开、用完就扔的。</em>'),
    "__FIG_KNOB1__": ("fig-knob1", "fig3-knob1.svg", 'topic03-fig-knob1.py',
        '⭐⭐ <b>左边四种存法摆在同一形状下：MQA 反而比 MLA 还小 2.25 倍。</b>'
        '<b>所以这一支比的不是「谁存得最少」，是「同样一份字节换回多少能力」。</b>'
        '<em>右边那三行代数，就是 MLA 为什么非要把 RoPE 拆出去单走一路。</em>'),

    # ── §四 三个旋钮：全课骨架（2026-09-12 加）─────────────────────
    # ⛔ 讲义写着这一节「不许压，一压后面全散」，而它原来 **0 张图**。
    #   ⭐ 图要证明的是「**为什么恰好是三个**」，不是罗列三类方法 ——
    #     所以左边摆「一个 query 要做的三步」，右边一一正对。
    # ── §3.3 在线 softmax（2026-09-13 夜间 R16 加）─────────────────
    # ⭐ 装置沿用 fig3-lowrank：**让读者自己核**。两条路用同一组真数字各算一遍，
    #   脚本断言**精确相等**（差恒为 0，不是「误差很小」）——
    #   ⛔ 写成「误差 < 1e-6」读者就有理由怀疑它是个近似，而它不是。
    # ── §8.3 NoPE（2026-09-13 夜间 R17 加）─────────────────────────
    # ⭐ 这是 fig3-absorb 的续集。⛔ 别把它当成「又一个技巧」——
    #   它要教的是**约束之间有连接**：改层怎么配比，能让位置编码那条约束整个消失。
    # ── §6.1c 红墨水（2026-09-13 夜间 R18 加）─────────────────────
    # ⭐ 装置偷自 Barbero Figure 1（左右对照的 token×层 网格），
    #   ⛔ 但原图是**示意图**且漏了 ‖v‖≈0 那一笔 —— 本图两样都补上，并把数算出来。
    # ── §7.0b 查/擦/写（2026-09-13 夜间 R19 加）───────────────────
    # ⭐ 全网没有一张把「先擦后写」画好的图 —— 这张是本讲最可能的差异点。
    # ⛔ 图里所有比喻**都不是本课原创**，脚本头部与出处行里逐条记了来源。
    # ── §2.0 瓶颈翻转（2026-09-13 夜间 R20 加）────────────────────
    # ⭐ 装置的关键在**「权重那一段两根柱子必须像素级相同」** ——
    #   读者要看到的是「什么都没改，主角却换了人」。高度稍有不同这张图就废了。
    # ── 收尾 11.1 把枪打响（2026-09-14 故事线审计 R22 加）──────────
    # ⛔ 封面立的那个「512 倍」承诺，正文里一次都没兑现过 —— 这是结构性失约。
    # ⭐ 而且这一枪必须用本课自己的提问顺序打：**两笔账，三个旋钮各还各的**。
    "__FIG_GUN__": ("fig-gun", "fig3-gun.svg", 'topic03-fig-gun.py',
        '⭐⭐ <b>两个 512 为什么会相等</b>：GPT-3 的上下文长度和 DSA 的 k '
        '<b>恰好都是 2048</b> ——&nbsp;两条不相干的设定撞成一个数。'
        '<em>⚠️ 这是巧合不是规律，但它让这笔账好记得多。</em>'),
    "__FIG_FLIP__": ("fig-flip", "fig3-flip.svg", 'topic03-fig-flip.py',
        '⭐⭐ <b>只盯灰色那一段：两根柱子里它完全一样。</b>'
        '涨的全在红色那一段。'
        '<em>⚠️ 「要几张卡」只算了装得下装不下，<b>没算带宽</b> ——&nbsp;'
        '带宽那一笔在 <a href="#s九">§9.1</a> 还；'
        '换别的硬件只是刻度变，结论不变。</em>'),
    # ⭐ 2026-09-14 R26。§九 原来只有一张全是 ↓↓↓ 的表 —— 而箭头恰恰答不了
    #   本课自己立的那条「省的是哪一样」。这张图把它一路换算到毫秒。
    #   ⛔ 它**不引入任何新数**，全是前面核过的数做除法，读者可以逐格核。
    "__FIG_PERSTEP__": ("fig-perstep", "fig3-perstep.svg",
        'topic03-fig-perstep.py',
        '⭐⭐ <b>先只看灰段：五根柱子里它一模一样</b>（每步都要读那 34.46 GiB '
        '激活权重）。差别全在红段 ——&nbsp;而 MHA 那根红段是灰段的 <b>14 倍</b>。'
        '<em>⚠️ 这是 batch ＝ 1 的账，也是<b>只算 HBM 读的下界</b>；'
        '人一多，灰段被摊薄、红段不摊，画面会翻回去。</em>'),
    "__FIG_ERASE__": ("fig-erase", "fig3-erase.svg", 'topic03-fig-erase.py',
        '⭐⭐ <b>②那五行只看颜色就够</b>：格子全满＝只会写，五格一起淡＝只会忘，'
        '单格掏空＝只会改，<b>又淡又掏＝两个都会</b>，'
        '<b>五格淡得不一样快＝逐通道</b>。'
        '<em>⚠️ 抽屉是离散的，真实地址是连续方向 ——&nbsp;'
        '所以「定点擦」其实是个近似，落点带里说破了。</em>'),
    "__FIG_OVERMIX__": ("fig-overmix", "fig3-overmix.svg",
        'topic03-fig-overmix.py',
        '⭐⭐ <b>只读一件事：最后一行有多红。</b>左边四层之后旁观的词被推动得'
        '<b>跟当事人一样多</b>，右边只有约四分之一。'
        '<em>⚠️ 两张网格是本课的线性简化模拟，不是实测 ——&nbsp;'
        'Barbero 的 Figure 1 本身也是示意图。</em>'),
    "__FIG_NOPE__": ("fig-nope", "fig3-nope.svg", 'topic03-fig-nope.py',
        '⭐⭐ <b>③ 那三条本来看着互不相干</b>，却是同一个动作带来的。'
        '<em>⛔ 注意措辞：这不是「绕过了 R」，是<b>让 R 根本不用存在</b> ——&nbsp;'
        '被绕过和不存在，是两件事。</em>'),
    "__FIG_KNOBS__": ("fig-knobs", "fig3-knobs.svg", 'topic03-fig-knobs.py',
        '⭐⭐ <b>三个旋钮不是凑出来的：一个 query 只做三步，一步一个位置可动。</b>'
        '<b>这张图要留下的是「没有第四个位置」这个封闭感</b> —— '
        '<em>而 FlashAttention 不改算什么、只改怎么算，所以它不在这三条里。</em>'),

    # ── §二 两条动机线（2026-09-12 加）───────────────────────────────
    # ⛔ 这一节原来 **887 汉字、0 张图**，是全讲最严重的一处「说话多画图少」。
    "__FIG_MOTIVES__": ("fig-motives", "fig3-motives.svg", 'topic03-fig-motives.py',
        '⭐ <b>左边是硬件账（必须省），右边是信息账（可以省而不太亏）。</b>'
        '<b>两条完全独立，却在同一个地方交汇 —— 所有变体都活在那儿。</b>'
        '<em>判一个变体好不好，就看它在「省了多少」和「亏了多少」之间落在哪。</em>'),

    # ── §一 MHA 的三张图 ───────────────────────────────────────────
    "__FIG_MHA_SWAP__": ("fig-mha-swap", "fig3-mha-swap.svg", 'topic03-fig-mha.py',
        '⭐ <b>左边是 §零 那条链，右边是它的替代品。</b>'
        '看完带走一件事：<b>要算的格子从 n 变成了 n²，而那张表就是后面反复出现的「注意力矩阵」本人</b>。'),

    # ⭐ 2026-09-12 加。⛔ 跟 QKV 那张分工：这张答「信息怎么流」（直觉），
    #   那张答「一层里在算什么」（形状、四步、√d，机械）。
    #   判据：**一张图只回答一个问题；「顺便也讲讲」就是它开始讲不清的时候。**
    "__FIG_MHA_FLOW__": ("fig-mha-flow", "fig3-mha-flow.svg", 'topic03-fig-mha.py',
        '⭐ <b>跟着最后那个 token 走一遍：它提问（q）、每个位置报价（k）、'
        '按权重把所有人的货（v）加起来。</b>'
        '<b>一步之内够到全场，而且六个位置同时在做</b> —— '
        '这就是那句标题的意思：<b>混合信息这件事，只要注意力就够了。</b>'),

    "__FIG_MHA_QKV__": ("fig-mha-qkv", "fig3-mha-qkv.svg", 'topic03-fig-mha.py',
        '⭐⭐ <b>看第二格那两根横条就够了 —— 它们破的是全课最常见的那个误解</b>：'
        '注意力<b>不是「找出最像的那一个」</b>，而是<b>把五家的货按百分比混成一碗</b>；'
        '而且<b>这一整条的长度永远是 100%</b> —— softmax 不许有人弃权'
        '（<a href="#s六">§六</a> 讲 attention sink 时会回到这一句）。'
        '<b>左右两根条唯一的差别就是除没除 √d_k</b>：不除，整条被一家吃掉、'
        '模型只看得见「抽屉」；除了，才混得出「放进 ＋ 抽屉」。'
        '⭐ 所以 √d_k 决定的不只是梯度，是<b>这一层到底在「挑一家」还是在「混一碗」</b>。'
        '<b>而牌子和货（K 和 V）要留着给下一个 token 用，留下来的那两份就叫 KV cache。</b>'),

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
        '<b>先只看一件事 —— 在形状里找 <code>S</code>（KV 长度）：只有两处「留得下来」的带它，'
        'K 和 V 的输出。</b>'
        '那就是唯一需要跨 token 留下来的东西 —— <b>KV cache</b>。'
        '三个旋钮各是一种跟它较劲的方式。'
        '<span class="sub">图式借自 <a href="https://jax-ml.github.io/scaling-book/" target="_blank" rel="noopener">How to Scale Your Model</a>，本图为重画。</span>'),

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
    # ⭐⭐ 2026-09-14 现场原话：「整个文档里边有很多这种小字，并且是半透明的，
    #   其实并不重要，就是需要收着，折起来。」——&nbsp;说的就是每张图底下那四到
    #   八行「出处与口径」。它现在不画进 SVG 了（见 topic03_draw.src），
    #   旁落一份 .src.html，在这儿包成一个默认收起的 <details>。
    # ⛔ 折起来 ≠ 删掉：出处是「查过就要留下痕迹」那条规矩的另一半，
    #   它必须一直在、一直可查，只是不该默认占版面。
    sfp = fp[:-4] + ".src.html"
    note = ('<details class="figsrc"><summary>出处与口径</summary>%s</details>'
            % io.open(sfp, encoding="utf-8").read()) if os.path.isfile(sfp) else ''
    # ⛔ 图注为空串时**不要出空的 <figcaption>** —— 它不显示文字，但照样吃
    #    figcaption 的 margin/padding，图底下会多出一段说不清来路的空白。
    _html = _html.replace(
        ph, '<figure class="fbox fwide" id="%s">%s%s%s</figure>'
            % (fid, svg, '<figcaption>%s</figcaption>' % cap if cap else '',
               note))
assert "__FIG_" not in _html, "还有图占位符没被替换掉"
# ══════════════════════════════════════════════════════════════════
# ⭐⭐ 2026-09-13 学生审稿：全文 0 个锚链接 —— 每一句「见 §X.Y」都要
#   手动往回滚 80 万字符的页面。新手那位说他「真的滚回去找过，找不到才发现是错引」。
#   ⭐ 这里做一次后处理：
#     ① 给每个 <h3>/<h4> 自动加 id（按它开头的小节号）
#     ② 把正文里的「§X.Y」替换成指向它的 <a>
#   ⛔ 只替换**真实存在**的号 —— 指不到的保持原样，
#     这样它们在页面上仍然是纯文本，而 xref 体检照样能抓出来。
def _anchorize(html):
    import re as _re
    ids = {}

    def _mark(m):
        tag, attrs, body = m.group(1), m.group(2), m.group(3)
        plain = _re.sub(r"<[^>]+>", "", body)
        # ⛔ 合并标题（「5.1 ＋ 5.2 ＋ 5.3 ＋ 5.4」）要**把整串号都登记上** ——
        #   只认第一个的话，指向 5.3 的引用就锚不上。
        #   ⭐ 这跟 topic02-lint-xref 里那条是同一个坑，那边今晚刚修过一次。
        run = _re.match(r"\s*((?:\d+\.\d+[a-z]?)(?:\s*[＋+、，,～~]\s*\d+\.\d+[a-z]?)*)",
                        plain)
        if not run or "id=" in attrs:
            return m.group(0)
        nums = _re.findall(r"\d+\.\d+[a-z]?", run.group(1))
        sid = "s" + nums[0].replace(".", "-")
        for nm in nums:
            ids[nm] = sid
        return "<%s%s id=\"%s\">%s</%s>" % (tag, attrs, sid, body, tag)

    html = _re.sub(r"<(h[34])([^>]*)>(.*?)</\1>", _mark, html, flags=_re.S)

    def _link(m):
        num = m.group(1)
        if num not in ids:
            return m.group(0)          # 指不到的不动，留给 xref 体检去报
        return '<a href="#%s">§%s</a>' % (ids[num], num)

    # ⛔⛔ 2026-09-13：上面这行注释写着「也不碰已经在 <a> 里的」——&nbsp;**它没做**。
    #   于是手写的 `§6.5b` 被再套一层，产出 8 处**嵌套 <a>**
    #   （非法 HTML，浏览器会悄悄拆掉，点击行为不可预期）。
    # ⭐ 判据（本仓库第 N 次遇到）：**注释写「不碰 X」不等于真的没碰。**
    #   —— 而且这是自己给自己写的注释，最容易被当成已经成立的前提。
    parts = _re.split(r"(<svg.*?</svg>)", html, flags=_re.S)
    for i in range(0, len(parts), 2):
        inner = _re.split(r"(<a\b.*?</a>)", parts[i], flags=_re.S)
        for j in range(0, len(inner), 2):
            inner[j] = _re.sub(r"§(\d+\.\d+[a-z]?)", _link, inner[j])
        parts[i] = "".join(inner)
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-13 麻瓜视角审稿里唯一一条「我真的想关掉页面」：
#   「1MB 单页 ＋ 36 张图 ＋ 十几处内部跳转，**跳过去就回不来了**。」
#   而 CSS 里 .progress / .totop 早就写好了，body 里一次都没用上 ——
#   ⛔ **写好了没接上，等于没写**，而且比没写更难发现（看代码像是有）。
# ⭐ 目录**从 SECTIONS 里长出来**，不手写 —— 手写那份一定会跟节标题漂。
# ══════════════════════════════════════════════════════════════════
def _build_nav(html):
    secs = re.findall(
        r'<section id="(s[^"]+)"><div class="wrap"><div class="stn">'
        r'<span class="badge">第 (\S+) 节</span><h2>(.*?)</h2>', html, re.S)
    if not secs:
        return html
    # ⭐⭐ 2026-09-14：去掉条目前面的「零 一 二 三…」。
    #   ⛔ 它在这一条横带里**没有一个读者用得上**：目录本来就是从左到右排的，
    #     顺序靠位置已经说清楚了，汉字数字只是把每个条目撑宽一截、
    #     还跟标题抢视觉重量（原来它是 <b> 且比标题更黑）。
    #   ⭐ 节号没有丢 —— 挪进 title= 悬停提示。**要用的时候还在，不占版面。**
    items = []
    for sid, num, title in secs:
        t = re.sub(r"<[^>]+>", "", title).replace("&nbsp;", " ")
        full = re.split(r"——|:|：", t)[0].strip()
        items.append('<a href="#%s" title="第 %s 节 · %s">%s</a>'
                     % (sid, num, full, full[:16]))
    # ⛔⛔ 2026-09-14 类名撞车：这条吸顶带原来也叫 `.toc`，而**基础 CSS 里早就有一个
    #   `.toc`** —— 那是 2026-09-07 删掉的「这一讲的路线」方框目录留下的规则
    #   （`.toc{background:var(--bg2);border-radius;padding:22px 26px;margin:24px 0}`）。
    #   CSS_NAV 排在后面，所以 background 被盖住了、**看不出撞了**；
    #   但 padding / margin / border-radius 没人覆盖，**一路漏进来**：
    #   一条 44px 的横带被撑成 **88px**，上下各白 22px，左右还多 10px。
    #   ⛔ 那条基础规则**不能删** —— 它是从 topic-02-L300 的 CSS 整份抄来的，
    #     专题一现在还在用那个方框目录。所以改名的是我这边。
    #   ⭐⭐ 判据：**新控件复用老类名，冲突只会在「没被覆盖到的那几个属性」上显形。**
    #     背景色这种一眼能看见的反而是安全的 —— 危险的是盒模型：
    #     它不改颜色、不报错，只是让尺寸对不上，而你会以为是自己的 padding 写错了。
    nav = ('<div class="progress"><i></i></div>\n'
           '<nav class="tocbar" id="tocbar"><div class="tocbar-in">'
           '<span class="tocbar-lb">目录</span>' + "".join(items) + '</div></nav>\n')
    # ⛔⛔ 2026-09-14：这个「回到顶部」**一次都没工作过**，而且它看起来完全正常 ——
    #   按钮画出来了、.on 加上了、点得到（elementFromPoint 就是它自己）、
    #   控制台一条报错都没有，**只是页面纹丝不动**。
    #   根因：原来写的是裸 `scrollTo({top:0,…})`。行内 onclick 的作用域链是
    #     元素 → 表单 → document → window，而 **`Element.prototype.scrollTo` 是存在的** ——
    #   于是它解析成「滚动这个按钮自己」，按钮没有溢出内容，静默无操作。
    #   ⭐⭐ 判据：**行内事件处理器里调任何滚动 / 焦点 / 尺寸类方法，一律写全 `window.`。**
    #     这类名字在 Element 上往往也有一份同名的，抢在 window 前面被找到，
    #     而且**不报错**——它是个合法调用，只是作用在错的对象上。
    #   ⭐ 同一个坑的另一半：下面那个局部变量原来叫 `top`（＝ `window.top`）。
    #     IIFE 里侥幸没出事，但同一族的影子命名，顺手改成 `btn`。
    tail = ('<button class="totop" id="totop" title="回到顶部"'
            ' onclick="window.scrollTo({top:0,behavior:\'smooth\'})">↑</button>\n'
            '<script>(function(){\n'
            ' var bar=document.querySelector(".progress i"),'
            ' btn=document.getElementById("totop"),'
            ' toc=document.getElementById("tocbar"),'
            ' ls=[].slice.call(toc.querySelectorAll("a"));\n'
            ' function tick(){\n'
            '  var h=document.documentElement,'
            '  p=h.scrollTop/(h.scrollHeight-h.clientHeight||1);\n'
            '  bar.style.width=(p*100).toFixed(1)+"%";\n'
            '  btn.className="totop"+(h.scrollTop>600?" on":"");\n'
            '  var cur=null;\n'
            '  ls.forEach(function(a){var e=document.querySelector(a.getAttribute("href"));\n'
            '    if(e&&e.getBoundingClientRect().top<140)cur=a;});\n'
            '  ls.forEach(function(a){a.className=(a===cur?"on":"");});\n'
            '  if(cur&&cur.offsetLeft-toc.scrollLeft>toc.clientWidth-220)\n'
            '    toc.scrollLeft=cur.offsetLeft-120;\n'
            ' }\n'
            ' addEventListener("scroll",tick,{passive:true});tick();\n'
            # ⛔ 折叠起来的 <details> **打印出来只剩一行标题** —— 出处与口径
            #   是「查过就要留痕」那条规矩的落点，不能因为默认收起就印不出来。
            #   ⭐ CSS 做不到强行展开（那是 UA 行为，display 覆盖不了），
            #     所以挂 beforeprint / afterprint，印完原样收回去。
            ' var SR=".figsrc";\n'
            ' addEventListener("beforeprint",function(){\n'
            '  document.querySelectorAll(SR).forEach(function(d){\n'
            '   d.dataset.wasopen=d.open?"1":"";d.open=true;});});\n'
            ' addEventListener("afterprint",function(){\n'
            '  document.querySelectorAll(SR).forEach(function(d){\n'
            '   d.open=d.dataset.wasopen==="1";});});\n'
            '})();</script>\n')
    html = html.replace("</head>", CSS_NAV + "</head>", 1)
    html = re.sub(r"(<body[^>]*>)", r"\1\n" + nav, html, count=1)
    return html.replace("</body>", tail + "</body>", 1)


CSS_NAV = """<style>
/* ⭐ 吸顶目录：html 的 scroll-padding-top 本来就留了 96px 给它 ——
   那条规则之前是空转的（顶上什么都没有，点内链只是白空一截）。 */
.tocbar{position:sticky;top:0;z-index:58;background:rgba(255,255,255,.96);
     backdrop-filter:saturate(1.6) blur(8px);border-bottom:1px solid #e8eaed;
     overflow-x:auto;scrollbar-width:none}
.tocbar::-webkit-scrollbar{display:none}
.tocbar-in{display:flex;gap:6px;align-items:center;white-space:nowrap;
        max-width:1760px;margin:0 auto;padding:8px 16px}
.tocbar-lb{font:600 12px var(--mono);color:#9aa0a6;padding-right:6px;flex:none}
/* ⭐ 2026-09-14 每个条目给一个真的框。原来只有 padding ＋ 悬停底色 ——
   不悬停的时候它就是一排裸字，看不出「这是可以点的东西」，
   条目之间也只靠 2px 间距分隔，读起来是一长串。
   ⛔ 框要用 1px 实线 ＋ 胶囊圆角，不要用阴影：吸顶条只有 40 多像素高，
     阴影在这个尺度上只会糊成一片灰。 */
.tocbar a{font:500 13px/1.2 var(--mono);color:#5f6368;text-decoration:none;
       padding:5px 11px;border:1px solid #dadce0;border-radius:999px;
       background:#fff;flex:none;transition:background .12s,border-color .12s}
.tocbar a:hover{background:#f1f3f4;border-color:#bdc1c6;color:#202124}
.tocbar a.on{background:#e8f0fe;border-color:#aecbfa;color:#174ea6;font-weight:600}
@media print{.tocbar{display:none}}
</style>
"""

_html = _anchorize(_html)
_html = _build_nav(_html)

# ⭐⭐ 2026-09-13 现场点的：「引用的那些论文得在教材里边，把可点击的 link 都放里边，
#   有愿意多学的人可以去点开看。」→ arXiv 编号**自动**变成链接，见 course_links.py。
#   ⛔ 用后处理不用手写 <a>：这一讲有几十处，手写等于每次都要记得加，而忘了不报错。
import course_links as _CL
_html = _CL.linkify_arxiv(_html)
io.open(OUT, "w", encoding="utf-8").write(_html)
print("ok  topic-03.html  %s 字符 · %d 节 · %d 个论文链接"
      % (format(os.path.getsize(OUT), ","), len(SECTIONS), _CL.count(_html)))
