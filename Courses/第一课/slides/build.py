#!/usr/bin/env python3
"""生成第一课的 HTML 幻灯片。

  python3 build.py              → 输出 index.html（图用相对路径，本地打开可看）
  python3 build.py --bundle DIR → 连图一起拷进 DIR，用于发布

幻灯片内容在下面的 SLIDES 列表里，一页一个 dict。改内容改这里，别改 HTML。
每页的 kind 决定版式：

  cover    封面
  section  节扉页（大号数字 + 标题）
  image    一张图占主体 + 标题 + 一句旁白        ← 最常用
  bullets  标题 + 要点列表
  table    标题 + 表格
  quote    整页一句话（用来放钩子和金句）
  split    左文右图
  end      结尾
"""
import argparse, base64, html, pathlib, re, shutil, sys

IMG_SRC = pathlib.Path(__file__).resolve().parent.parent / "教学材料" / "3b1b图"

# ─────────────────────────────────────────────────────────────────
# 内容
# ─────────────────────────────────────────────────────────────────
SLIDES = [
 dict(kind="cover", title="模型架构", sub="第一课 · 从「猜下一个词」到 1750 亿参数",
      meta="加速器系统课程 · 约 60 分钟"),

 dict(kind="bullets", title="这一小时讲什么", items=[
      "**1 · 一台会接话的机器** — LLM 到底是什么　*10 min*",
      "**2 · 词变成向量** — 方向是有含义的　*10 min*",
      "**3 · Attention** — 让词互相通气　*15 min*",
      "**4 · MLP** — 事实存在这里　*15 min*",
      "**5 · 参数账** — 1750 亿怎么摞出来　*10 min*"],
      foot="不讲硬件、不讲怎么训、不讲数据从哪来。只讲模型本身长什么样，以及为什么长成这样。"),

 # ── 1 ────────────────────────────────────────────────────────
 dict(kind="section", num="1", title="一台会接话的机器", sub="LLM 到底是什么"),

 dict(kind="quote", text="你翻出一页剧本。人说的话都在，**AI 的回答被撕掉了**。\n\n"
      "你手上有一台机器，只会做一件事：给它一段文字，它告诉你**下一个词最可能是什么**。\n\n"
      "你会怎么补？", accent="blue"),

 dict(kind="image", img="SimpleAutogregression.png", title="接龙",
      cap="猜一个词 → 接回输入 → 再猜下一个。**这就是你跟 ChatGPT 说话时背后发生的全部事情。**"),

 dict(kind="quote", text="大语言模型是一个\n**预测下一个词**的\n非常复杂的数学函数。", accent="yellow"),

 dict(kind="image", img="AnnotateNextWord.png", title="它给的不是答案，是一整张概率表",
      cap="喂进「中国的首都是」，它不吐出「北京」，而是吐出**所有可能的词各自一个概率**，然后抽签。"),

 dict(kind="bullets", title="而且它故意不总选最可能的那个", items=[
      "**全取最大值，文字会非常僵硬** — 真实语言里总有不那么「标准」的用词",
      "所以按概率随机抽，允许偶尔选一个不那么可能的词",
      "**模型本身是完全确定的** — 同样输入，同样的概率分布",
      "**但你每次问同一个问题，答案不一样** — 不确定性在「抽签」这一步"],
      foot="这个抽签的手气可以调，那个旋钮叫 temperature —— 第 5 节回来说。"),

 dict(kind="image", img="TweakedMachine.png", title="「大」在哪：参数就是旋钮",
      cap="GPT-3 有 **1750 亿**个旋钮。**没有任何人手动设过其中任何一个。**"),

 dict(kind="bullets", title="那它们是怎么来的", items=[
      "一开始全是随机的 — 那时模型输出的是彻底的乱码",
      "拿一段真实文字，**把最后一个词遮住**，让模型猜",
      "跟真答案比，**微调所有旋钮**：真答案概率高一点点，其它低一点点",
      "重复 — **几万亿次**"],
      foot="一个人要读完 GPT-3 的训练数据，得不吃不喝不睡连读 2600 多年。"),

 dict(kind="image", img="DistinguishWeightsAndData.png", title="两种东西，别混在一起",
      cap="**权重**（蓝/红）是训练学来的脑子，训完就固定；**数据**（灰）是你这次喂进去的文字。"
          "看任何架构图，先分清哪些是学来的、哪些是流过去的。"),

 dict(kind="image", img="AthleteCompletion.png", title="顺带看一件怪事",
      cap="它不只会接话，还**记得住事实**。「Michael Jordan 打的运动是」→「篮球」。\n"
          "**这条知识存在哪儿？存成什么样子？** 第 4 节回来回答。"),

 dict(kind="bullets", title="最后：为什么是 Transformer", items=[
      "2017 年之前的语言模型**一个词一个词地读** — 第 100 个词得等前 99 个",
      "**它没法并行**",
      "Transformer 把整段一次全吞进去，**所有位置同时处理**"],
      foot="Transformer 赢，不是因为它更聪明，而是因为它能把活儿摊开给成千上万个计算单元同时干。"),

 dict(kind="quote", text="🔑 **模型架构的选择，\n从一开始就被硬件的形状塑造着。**\n\n"
      "为什么注意力有那么多变体、为什么 MoE 会流行、为什么大家都在改 KV cache —— "
      "追到底，答案常常不在数学里，在硬件里。", accent="blue"),

 # ── 2 ────────────────────────────────────────────────────────
 dict(kind="section", num="2", title="词变成向量", sub="方向是有含义的"),

 dict(kind="quote", text="计算机只会算数。\n\n「猫」这个字怎么变成能做乘法的东西？\n\n"
      "而且变完之后 —— 为什么 **向量(国王) − 向量(男人) + 向量(女人) ≈ 向量(女王)**？",
      accent="yellow"),

 dict(kind="image", img="DiscussTokenization.png", title="第一步：切碎成 token",
      cap="不按字母（序列太长，attention 是平方开销）；不按单词（词表装不下，遇生词抓瞎）。\n"
          "**常见词整块留，罕见词拆成常见碎片。**"),

 dict(kind="image", img="IntroduceEmbeddingMatrix.png", title="第二步：查表变向量",
      cap="GPT-3：词表 **50,257**，每个向量 **12,288** 维 → 这张表 **6.18 亿**个数，全是训练出来的。"),

 dict(kind="image", img="ThreeDSpaceExample.png", title="词落在一个高维空间里",
      cap="12,288 维画不出来，切个三维片看个意思。"),

 dict(kind="image", img="ManyIdeasManyDirections.png", title="关键直觉：方向是有含义的",
      cap="**性别方向**：woman − man 加到 king 上 ≈ queen。\n"
          "**复数方向**：cats − cat，拿它跟 one/two/three/four 做点积，值单调上升。\n"
          "没人教它这么做 —— 这样组织信息对「猜下一个词」有利。"),

 dict(kind="image", img="DotProducts.png", title="记住这个运算：点积",
      cap="同向为**正**、垂直为**零**、反向为**负**。计算上就是对应位置相乘再全加起来。\n"
          "**下一节整个 attention，核心就是这一个运算。**"),

 dict(kind="quote", text="此刻，每个向量**只知道自己**。\n\n"
      "「bank」查出来的向量，旁边是 river 还是 money，**完全一样**。\n\n"
      "让它们互相认识 —— 下一节。", accent="grey"),

 # ── 3 ────────────────────────────────────────────────────────
 dict(kind="section", num="3", title="Attention", sub="让词互相通气"),

 dict(kind="quote", text="I sat on the river **bank**.\n\nI went to the **bank** to withdraw money.\n\n"
      "同一个词，同一个向量。**显然不够。**", accent="red"),

 dict(kind="image", img="AttentionPatterns.png", title="先看结果长什么样",
      cap="句子：a fluffy blue creature roamed the verdant forest。\n"
          "creature 那一列：fluffy **0.42**、blue **0.58** —— 「我该从这两个词各拿多少信息」。\n"
          "**attention 的全部输出就是一堆增量 ΔE。**"),

 dict(kind="table", title="三个矩阵：一个招聘的比喻",
      head=["名字", "记号", "在问什么"],
      rows=[["**Query** 查询", "Q", "我这个岗位在找什么样的人？"],
            ["**Key** 键", "K", "我是什么样的人？"],
            ["**Value** 值", "V", "如果配上了，我能交给你什么？"]],
      foot="这三个矩阵里的参数全是学出来的 —— 没人指定「第 3 个头负责找形容词」。"),

 dict(kind="split", img="QueryMap.png", title="Query：我在找什么",
      body="creature 这个位置的 Query，相当于在问：\n\n"
           "**「我是个名词。有没有形容词在描述我？」**\n\n"
           "这个「问题」不是人写的，是 W_Q 矩阵里的参数**学出来的**。"),

 dict(kind="split", img="KeyMap.png", title="Key：我是什么",
      body="同时每个位置算一个 Key 向量，相当于在回答：\n\n**「我是一个形容词。」**\n\n"
           "然后 Q 和 K 做**点积** —— 大就是相关，小或负就是不相关。\n\n"
           "再对每列做 softmax，变成加起来为 1 的权重。"),

 dict(kind="image", img="DescribeAttentionEquation.png", title="那条著名的公式",
      cap="**QKᵀ** 每个 Q 跟每个 K 点积　·　**÷√dₖ** 防止 softmax 饱和　·　"
          "**softmax** 变成权重　·　**×V** 加权求和\n"
          "**没有一个「聪明」的部件，全是点积、缩放、归一化、加权求和。**"),

 dict(kind="image", img="ShowMasking.png", title="一条铁律：不许偷看未来",
      cap="一次算完整句是 Transformer 能并行的原因，但第 3 个词看见第 5 个词就成了抄答案。\n"
          "做法：softmax 之前把右上三角设成负无穷。这叫 **causal mask**。"),

 dict(kind="bullets", title="多头：一次问很多个问题", items=[
      "形容词在修饰哪个名词？　代词指的是谁？　这个动词的宾语是什么？",
      "**一个头只能问一个问题** → 并排放很多个，各有独立的 Q/K/V",
      "GPT-3：每层 **96 个头**，共 **96 层**"],
      foot="⚠️ 最要紧的一句：attention 不产生任何新知识，它只做信息的搬运和路由。"),

 # ── 4 ────────────────────────────────────────────────────────
 dict(kind="section", num="4", title="MLP", sub="事实存在这里"),

 dict(kind="quote", text="回到第 1 节那个问题：\n\n"
      "「Michael Jordan 打篮球」这条知识，\n**在那 1750 亿个数字里，具体存在哪儿？**",
      accent="yellow"),

 dict(kind="table", title="它跟 Attention 有多不一样",
      head=["", "Attention", "MLP"],
      rows=[["向量之间", "**互相通气**", "**完全不通气，各走各的**"],
            ["在干什么", "搬运信息", "存放并检索事实"],
            ["计算形状", "长度的平方项", "一堆独立的矩阵乘"]],
      foot="第二行是关键：搞懂一个向量发生了什么，就等于搞懂了全部。"),

 dict(kind="image", img="MLPIcon.png", title="招牌形状：中间胖，两头瘦",
      cap="中间层是两头的 **4 倍宽**。"),

 dict(kind="image", img="BreakDownThreeSteps.png", title="三步走",
      cap="**升维**（×4）→ **非线性**（ReLU）→ **降维**（变回来），结果加回原向量。\n"
          "结构上简单得令人失望。难的是理解它在干什么。"),

 dict(kind="table", title="第一步：升维矩阵在提问",
      head=["输入", "跟「Michael+Jordan」方向的点积", "加偏置 −1"],
      rows=[["**Michael Jordan**", "2", "**+1**"],
            ["Michael Phelps", "1", "0"],
            ["Alexis Jordan", "1", "0"],
            ["无关的东西", "≤ 0", "负数"]],
      foot="矩阵有多少行，就在同时问多少个这样的问题。GPT-3 是 49,152 行（= 12,288 × 4，取整倍数对硬件友好）。"),

 dict(kind="image", img="NonlinearityOfLanguage.png", title="第二步：ReLU 把它变成干脆的「是」",
      cap="负数压成 0，正数原样通过。于是只有完整的 Michael Jordan 输出 1，其它全是 0。\n"
          "**本质上是一个 AND 门**：名叫 Michael **并且** 姓 Jordan，才点亮。"),

 dict(kind="image", img="BasicMLPWalkThrough.png", title="第三步：降维矩阵在作答",
      cap="把矩阵的每**列**看成一个方向。那个神经元亮了 → 「篮球」这个方向被加进结果。\n"
          "**升维矩阵的一行负责问，降维矩阵的对应列负责答。**"),

 dict(kind="quote", text="⚠️ 但真实模型不是这样的。\n\n"
      "**证据表明：单个神经元几乎从不对应一个干净的概念。**", accent="red"),

 dict(kind="image", img="ShowAngleRange.png", title="先问一个几何问题",
      cap="N 维空间里最多能放几个**互相垂直**的方向？ → N 个，这是「维度」的定义。\n"
          "那如果放宽到 **89°–91° 也算**呢？"),

 dict(kind="image", img="Superposition.png", title="叠加",
      cap="**能塞进去的方向数随维度指数增长**（Johnson-Lindenstrauss）。\n"
          "85° 容差下：**12,288 维 → 400 亿个以上**；116,000 维 → 超过 10¹⁰⁰。"),

 dict(kind="bullets", title="于是这一条同时解释了两件事", items=[
      "**① 为什么模型这么难解释** — 一个神经元同时参与很多个特征的编码，"
      "所以它看起来对一堆不相干的东西都有反应",
      "**② 为什么放大收益这么大** — 维度翻 10 倍，能装的独立概念**远不止**翻 10 倍"],
      foot="一个特征不是「某个神经元亮了」，而是「一组神经元以某种特定组合亮了」。"),

 dict(kind="quote", text="**关联在 Attention，记忆在 MLP。**\n\n"
      "一个负责路由，一个负责查表。\nGPT-3 里这样交替 96 次。", accent="green"),

 # ── 5 ────────────────────────────────────────────────────────
 dict(kind="section", num="5", title="参数账", sub="1750 亿是怎么摞出来的"),

 dict(kind="quote", text="零件都看完了。现在数一数。\n\n"
      "**先猜：Attention 和 MLP，哪个参数更多？**", accent="yellow"),

 dict(kind="image", img="CountMatrixParameters.png", title="Attention 的参数怎么数",
      cap="一个头四个矩阵 ≈ **630 万** → 一层 96 个头 ≈ **6 亿** → 96 层 ≈ **580 亿**"),

 dict(kind="table", title="结果", head=["部件", "参数量", "占比"],
      rows=[["嵌入矩阵 W_E", "6.18 亿", "0.35%"],
            ["反嵌入矩阵 W_U", "6.18 亿", "0.35%"],
            ["**Attention（全部）**", "**约 580 亿**", "**33%**"],
            ["**MLP（全部）**", "**约 1160 亿**", "**66%**"],
            ["**合计**", "**约 1750 亿**", ""]],
      foot="「Attention 得到了所有的注意力，但大多数参数在它旁边那些块里。」"),

 dict(kind="image", img="ShowGPT3Numbers.png", title="两个反直觉",
      cap="**① 大头在 MLP，是 attention 的两倍。**\n"
          "**② 那 1160 亿正是「事实」存放的地方** —— 第 4 节和这一节在这里合上了。\n"
          "顺带解释了为什么 MoE 稀疏化的是 MLP：大头在这儿，而且它天然是一堆独立的问题。"),

 dict(kind="image", img="SoftmaxBreakdown.png", title="最后一步：变回文字",
      cap="最后那个向量 → 反嵌入矩阵 → logits → **softmax** → 概率分布。\n"
          "**温度**大则分布变平（更有创意也更容易胡说），小则变尖，为 0 就退化成确定性输出。\n"
          "**这就闭上了第 1 节那个环。**"),

 dict(kind="quote", text="1750 亿参数 × 2 字节（bf16）= **350 GB**　*只是权重*\n\n"
      "训练还要优化器状态，约 12 字节/参数 ≈ **2.1 TB**\n\n"
      "而一块 **TPU v7** 的一个 device：**94.74 GB**", accent="red"),

 dict(kind="quote", text="**光是把它放下，就需要几十个 device。**\n\n"
      "装不下就得切开；一切开就得通信；一通信，通信本身可能比计算还慢。\n\n"
      "**那这笔账到底该怎么算？** → **第二课**", accent="blue"),

 dict(kind="bullets", title="第一课 · 五句话", items=[
      "**LLM 是一个猜下一个词的函数** — Transformer 赢在能并行，架构从一开始就被硬件塑造",
      "**方向是有含义的** — 点积衡量对齐，是后面一切的基础运算",
      "**Attention 让向量互相通气** — Q 问、K 答、V 交付；只搬运信息，不存储知识",
      "**MLP 存事实** — 升维提问、ReLU 判是否、降维作答；但特征是方向的组合（叠加）",
      "**三分之二的参数在 MLP** — 1750 亿 = 350 GB = 装不进任何一块卡"]),

 dict(kind="end", title="课后三题", items=[
      "**纸笔** — 词表 128,000、维度 4,096、32 层、MLP 4 倍宽：嵌入矩阵多少参数？全部 MLP 多少？后者是前者几倍？",
      "**十行代码** — 跑个小模型打印概率分布，temperature 从 0 调到 2，看分布怎么变",
      "**观察** — 找句有歧义的话换上下文，看预测怎么变 —— 你正在亲眼看见 attention 在起作用"],
      foot="想把零件全手写一遍：Stanford CS336 Assignment 1 Section 3，有测试套件判分。"),
]

# ─────────────────────────────────────────────────────────────────
# 渲染
# ─────────────────────────────────────────────────────────────────
def rich(s: str) -> str:
    """**粗体** → <b>，*斜体* → <em>，换行 → <br>"""
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r'<em>\1</em>', s)
    return s.replace("\n", "<br>")


def render(s: dict, i: int, n: int, prefix: str) -> str:
    k = s["kind"]
    img = lambda: f'{prefix}{s["img"]}'
    body = ""

    if k == "cover":
        body = (f'<div class="cover"><div class="dots"><i></i><i></i><i></i><i></i></div>'
                f'<h1>{rich(s["title"])}</h1><p class="sub">{rich(s["sub"])}</p>'
                f'<p class="meta">{rich(s["meta"])}</p></div>')
    elif k == "section":
        body = (f'<div class="section"><span class="num">{s["num"]}</span>'
                f'<div><h2>{rich(s["title"])}</h2><p class="sub">{rich(s["sub"])}</p></div></div>')
    elif k == "image":
        body = (f'<h3>{rich(s["title"])}</h3>'
                f'<div class="figwrap"><img src="{img()}" alt=""></div>'
                f'<p class="cap">{rich(s["cap"])}</p>')
    elif k == "split":
        body = (f'<h3>{rich(s["title"])}</h3><div class="split">'
                f'<div class="sbody">{rich(s["body"])}</div>'
                f'<div class="figwrap"><img src="{img()}" alt=""></div></div>')
    elif k == "bullets":
        li = "".join(f"<li>{rich(x)}</li>" for x in s["items"])
        body = f'<h3>{rich(s["title"])}</h3><ul>{li}</ul>'
    elif k == "table":
        th = "".join(f"<th>{rich(x)}</th>" for x in s["head"])
        tr = "".join("<tr>" + "".join(f"<td>{rich(c)}</td>" for c in r) + "</tr>"
                     for r in s["rows"])
        body = (f'<h3>{rich(s["title"])}</h3>'
                f'<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>')
    elif k == "quote":
        body = f'<div class="quote a-{s.get("accent","blue")}">{rich(s["text"])}</div>'
    elif k == "end":
        li = "".join(f"<li>{rich(x)}</li>" for x in s["items"])
        body = f'<h3>{rich(s["title"])}</h3><ul>{li}</ul>'

    if s.get("foot"):
        body += f'<p class="foot">{rich(s["foot"])}</p>'
    return (f'<section class="slide k-{k}" id="s{i}">{body}'
            f'<span class="pn">{i} / {n}</span></section>')


CSS = """
:root{--bg:#0b0c10;--fg:#e9eaee;--dim:#9aa0aa;--line:#23252c;
      --blue:#8ab4f8;--red:#f28b82;--yellow:#fdd663;--green:#81c995;}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:#000;color:var(--fg);
  font-family:"Noto Sans CJK SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",
              -apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
#deck{width:100vw;height:100vh;overflow:hidden;position:relative}
.slide{position:absolute;inset:0;width:100%;height:100%;
  background:var(--bg);padding:4.2vh 5vw 6vh;
  display:none;flex-direction:column;justify-content:center;gap:2.2vh}
.slide.on{display:flex}
h1{font-size:6.4vh;font-weight:700;letter-spacing:-.02em}
h2{font-size:5.2vh;font-weight:700;letter-spacing:-.02em}
h3{font-size:3.5vh;font-weight:600;letter-spacing:-.01em;flex:0 0 auto}
b{color:#fff;font-weight:650}
em{color:var(--dim);font-style:normal;font-size:.86em}
.sub{color:var(--dim);font-size:2.7vh;margin-top:1.4vh}
.meta{color:#6b7280;font-size:2vh;margin-top:3.5vh}
/* cover */
.cover{height:100%;display:flex;flex-direction:column;justify-content:center}
.dots{display:flex;gap:1.1vh;margin-bottom:4vh}
.dots i{width:1.5vh;height:1.5vh;border-radius:50%}
.dots i:nth-child(1){background:var(--blue)}.dots i:nth-child(2){background:var(--red)}
.dots i:nth-child(3){background:var(--yellow)}.dots i:nth-child(4){background:var(--green)}
/* section */
.section{height:100%;display:flex;align-items:center;gap:4vw}
.num{font-size:22vh;font-weight:800;line-height:.8;color:#1b1d24;letter-spacing:-.05em}
/* image */
.figwrap{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center}
.figwrap img{max-width:100%;max-height:100%;object-fit:contain;
  border-radius:10px;background:#000}
.cap{color:var(--dim);font-size:2.15vh;line-height:1.65;flex:0 0 auto}
/* split */
.split{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:1fr 1.15fr;
  gap:3vw;align-items:center}
.sbody{font-size:2.5vh;line-height:1.8;color:#c8ccd4}
/* bullets */
ul{list-style:none;display:flex;flex-direction:column;gap:2.1vh;
   font-size:2.6vh;line-height:1.6;color:#c8ccd4}
li{padding-left:2.6vh;position:relative}
li:before{content:"";position:absolute;left:0;top:.85em;width:.9vh;height:.9vh;
  border-radius:50%;background:var(--blue)}
/* table */
table{width:100%;border-collapse:collapse;font-size:2.35vh}
th{text-align:left;color:var(--dim);font-weight:500;font-size:2vh;
   padding:1.5vh 1.6vw;border-bottom:1px solid var(--line)}
td{padding:1.7vh 1.6vw;border-bottom:1px solid var(--line);color:#c8ccd4}
tbody tr:last-child td{border-bottom:none}
/* quote */
.quote{font-size:4vh;line-height:1.55;font-weight:500;padding-left:2.6vw;
  border-left:5px solid var(--blue);color:#dfe2e8}
.a-red{border-color:var(--red)}.a-yellow{border-color:var(--yellow)}
.a-green{border-color:var(--green)}.a-grey{border-color:#4b5563}
/* foot */
.foot{color:#7c828c;font-size:1.95vh;line-height:1.6;
  border-top:1px solid var(--line);padding-top:1.8vh;flex:0 0 auto}
.pn{position:absolute;right:2.4vw;bottom:2.4vh;color:#3d4149;font-size:1.7vh;
  font-variant-numeric:tabular-nums}
/* progress */
#bar{position:fixed;left:0;top:0;height:2px;background:var(--blue);z-index:9;
  transition:width .18s ease}
#help{position:fixed;left:2.4vw;bottom:2.4vh;color:#3d4149;font-size:1.6vh;z-index:9}
@media print{
  html,body,#deck{width:auto;height:auto;overflow:visible}
  .slide{display:flex!important;position:relative;page-break-after:always;
    width:100vw;height:100vh}
  #bar,#help{display:none}
}
"""

JS = """
const S=[...document.querySelectorAll('.slide')];let i=0;
const bar=document.getElementById('bar');
function go(n){i=Math.max(0,Math.min(S.length-1,n));
  S.forEach((s,k)=>s.classList.toggle('on',k===i));
  bar.style.width=((i+1)/S.length*100)+'%';
  location.hash=i?('#'+(i+1)):'';}
addEventListener('keydown',e=>{
  if(['ArrowRight','ArrowDown',' ','PageDown','n','j'].includes(e.key)){go(i+1);e.preventDefault();}
  if(['ArrowLeft','ArrowUp','PageUp','p','k'].includes(e.key)){go(i-1);e.preventDefault();}
  if(e.key==='Home')go(0); if(e.key==='End')go(S.length-1);});
addEventListener('click',e=>{if(!e.target.closest('a'))go(i+(e.clientX>innerWidth*0.32?1:-1));});
addEventListener('hashchange',()=>{const n=(parseInt(location.hash.slice(1))||1)-1;if(n!==i)go(n);});
go(Math.max(0,(parseInt(location.hash.slice(1))||1)-1));
window.__go=go;
"""


def build(prefix: str) -> str:
    n = len(SLIDES)
    slides = "\n".join(render(s, k + 1, n, prefix) for k, s in enumerate(SLIDES))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一课 · 模型架构</title>
<meta property="og:title" content="第一课 · 模型架构">
<meta property="og:description" content="从「猜下一个词」到 1750 亿参数 · 约 60 分钟">
<meta name="theme-color" content="#0b0c10">
<style>{CSS}</style></head>
<body><div id="bar"></div><div id="deck">
{slides}
</div><div id="help">← → 翻页 · 点击左/右半屏 · 共 {n} 页</div>
<script>{JS}</script></body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="连图一起输出到这个目录（用于发布）")
    a = ap.parse_args()

    if a.bundle:
        out = pathlib.Path(a.bundle); (out / "img").mkdir(parents=True, exist_ok=True)
        used = {s["img"] for s in SLIDES if "img" in s}
        for f in used:
            shutil.copy(IMG_SRC / f, out / "img" / f)
        (out / "index.html").write_text(build("img/"), encoding="utf-8")
        print(f"{out}/index.html　{len(SLIDES)} 页，{len(used)} 张图")
    else:
        out = pathlib.Path(__file__).resolve().parent / "index.html"
        out.write_text(build("../教学材料/3b1b图/"), encoding="utf-8")
        print(f"{out}　{len(SLIDES)} 页")
