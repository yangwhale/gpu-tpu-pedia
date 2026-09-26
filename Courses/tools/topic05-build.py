import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# -*- coding: utf-8 -*-
"""专题五 · 并行策略 —— 教材。

════════════════════════════════════════════════════════════════
主线（2026-09-24 定，R7 改口）：一刀一刀接力，每一刀都在补前面没管到的那一块
════════════════════════════════════════════════════════════════
    零  一张卡装不下 → 得切；一句话：用一种通信，换一份显存或一份算力
    一  先认识五种通信（集合通信）—— 后面每一刀多出来的都是这里的某一种
    二  第一刀 切数据：DP → ZeRO → FSDP（AllReduce 拆两半：ZeRO-1 白送，ZeRO-3 多付一半）
    三  第二刀 切权重：TP / PP
    四  第三刀 切专家：EP；attention 和专家各配各的（Folding / DEP / TEP）
    五  第四刀 切序列：训练切激活（CP），推理切 KV（DCP）
    六  第五刀 不切张量，切工作：PD 分离 / AFD
    七  摆到机器上：频率高的绑快链路；五步怎么选；strong / weak scaling
    八  全景地图（走完五刀回头看的总复习）

⭐ 全景原来放在最前面，主线定下来之后挪到最后：学员没认识通信、没走过五刀，
   一上来看一张几十行的表只会记名字。走完再看，一行一行都有来处。

════════════════════════════════════════════════════════════════
写完的和没写的混在一起，而且必须看得出来
════════════════════════════════════════════════════════════════
九节全部写完（2026-09-24）。
⛔ 编出来的内容看着最合理，也最难被自己发现（第一原则）。

几条核对过的事实（出处在文末台账）：
  · 各集合通信每卡发出的量：NCCL nccl-tests 的 PERFORMANCE.md 给的 busbw 修正系数
    —— AllReduce 2(n−1)/n，AllGather / ReduceScatter / AllToAll (n−1)/n，Broadcast / Reduce 1。
  · 环形 ReduceScatter 的逐步推演参考 wanghonglei《分布式深度学习集体通信原语——从零到精通》
    （2026-06-27），原文已存 my-wiki-v2/raw/articles/。图里的每一步是脚本现算并断言的。
  · TEP / DEP 是推理侧叫法（TRT-LLM blog26 原文定义），D 是 attention DP；
    **不能等同 Megatron 的 ETP / EDP**。
  · vLLM 的 PCP 扩大 world size、DCP 不扩大（vllm/config/parallel.py docstring）。

⭐ 大纲在 `Courses/专题05-并行策略.md`，只是计划，不必与页面同步。
"""
import io
import json
import os
import re

import topic03_page as P
import course_ai_trainer as AIT
import topic05_quiz as QZ          # 开场热身题；数字从 topic05_numbers 取
import topic05_numbers as NB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "WebPages", "topic-05.html")

head = P.make_head(
    "专题五 · 并行策略",
    "并行策略 · 用一种通信，换一份显存或一份算力",
    "先认识五种集合通信，再一刀一刀地切：数据、权重、专家、序列，最后连工作一起拆开。",
    "topic-05.html")
head += """
<style>
.kind { display:inline-block; font-size:12px; font-weight:600; padding:0 7px;
        border-radius:9px; margin-right:4px; white-space:nowrap }
.k-d { background:#e8f0fe; color:#174ea6 }
.k-s { background:#e6f4ea; color:#0d652d }
.k-m { background:#fef7e0; color:#8a4b00 }
.k-x { background:#f3e8fd; color:#681da8 }
.k-new { background:#fce8e6; color:#a50e0e }
.animgrid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; margin:14px 0;
             width:min(1408px, 94vw); position:relative; left:50%; transform:translateX(-50%) }
/* ⭐ 2026-09-25 L12：原来被正文版心限在约 1032px，每格动画只有 509px 宽、方块上的字约 10px；放到跟静态图一样宽 */
.animgrid figure { margin:0; min-width:0 }
.animcell figcaption { font-size:14px; color:var(--gray); margin-top:6px; line-height:1.6 }
.animgrid video { width:100%; border-radius:8px; display:block }
@media (max-width:900px) { .animgrid { grid-template-columns:1fr } }
.lecbtn { display:inline-block; margin-left:14px; font-size:13px; padding:4px 12px; border-radius:14px;
          background:#fff; border:1px solid var(--line); color:var(--blue); text-decoration:none }
.lecbtn:hover { background:var(--blue-l); border-color:var(--blue) }
</style>"""

D = '<span class="kind k-d">数据</span>'
S = '<span class="kind k-s">序列</span>'
M = '<span class="kind k-m">模型</span>'
X = '<span class="kind k-x">解耦</span>'
NEW = '<span class="kind k-new">新</span>'

# 开场热身题的样式单独追加一段 <style>，不去猜往第几个 </style> 前面插（专题四踩过：head 里有好几个）
_QUIZ_HTML, _QUIZ_CSS = QZ.build()
head += "<style>" + _QUIZ_CSS + "</style>\n"

SECTIONS = [
    ("s零", "零", "一张卡装不下"),
    ("s一", "一", "先认识五种通信"),
    ("s二", "二", "第一刀：切数据"),
    ("s三", "三", "第二刀：切权重"),
    ("s四", "四", "第三刀：切专家"),
    ("s五", "五", "第四刀：切序列"),
    ("s六", "六", "第五刀：不切张量，切工作"),
    ("s七", "七", "摆到机器上"),
    ("s八", "八", "全景：今天所有的并行方式"),
    ("s九", "九", "出处台账"),
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
  <div class="crumb"><a href="index.html">加速器系统课程</a> ／ 主线 ／ 专题五
    ／ <b>并行策略</b>
    <a class="lecbtn" href="topic-05-lecture.html">📝 讲义（授课稿）</a></div>
  <h1>并行策略</h1>
  <div class="hook">
    一个模型装不进一块卡，就得切开分到很多卡上。<br>
    <em>每切一刀，就在那一维上多出一种通信。用一种通信，换一份显存或一份算力。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    先认识五种通信，再一刀一刀地切：数据、权重、专家、序列，最后连工作一起拆开。
    每一刀都在补前面没管到的那一块。</p>
  <div class="chips">
    <span class="chip">前置 <b>专题四</b>（那张 16 字节的账）</span>
    <span class="chip">口径 <b>截至 2026-09</b></span>
    <span class="chip">⏱ <b>讲约 60 分钟</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

''' + AIT.note("topic-05-lecture.html") + '''
<div class="wrap">
  <p style="color:var(--gray)">每个数字的出处在文末台账；自己推出来的数，都标了「本课推导」。</p>
</div>
'''

BODY = sec("s零", "零", "一张卡装不下") + '''
__QUIZ__
  <p class="lead">专题四把账算完了：一个 6,710 亿参数的模型，按每参数 16 字节算（权重 2 ＋ 梯度 2 ＋ 优化器状态 12），
    光常驻的训练状态就要 9.76 TiB，约合一万 GB。最大的一块卡显存也就两三百 GB，光放下就要三四十到五六十块卡，还没开始算。
    <b>一块卡装不下，就得切。问题是沿哪一维切。</b></p>
__FIG_WHY_CUT__

  <p>一个训练中的张量能下刀的维度有好几个：batch、序列、隐藏维、层、专家。这一讲从头到尾只讲一件事：</p>

  <div class="note ok"><span class="t">一句话</span>
    <b>用一种通信，换一份显存或一份算力。</b><br>
    选并行策略，就是在选你愿意付哪一种通信、付多频繁、放在哪根线上。</div>

  <p>这一讲按一条接力线走：<b>切一刀，付一笔通信；切完发现还有一块没管到，再切一刀。</b></p>
__FIG_ROADMAP__
  <p>量每一刀用两把尺子：它多出来的通信<b>有多频繁</b>（决定放哪根线），每搬一个字节<b>换来多少计算</b>（决定会不会被拖住，第三节）。</p>

  <details class="foldfig"><summary><b>遇到不认识的词，回这儿查</b>：张量、激活、注意力头、MoE、KV cache、prefill 与 decode、Q/K/V、MLA、显存、ICI、「几路」……（专题一、三讲过，这里各一句话）</summary>
  <p style="line-height:1.9"><b>token 与 batch</b>：token 是模型处理文字的最小单位，大致一个字；batch 是一步喂进去的那一批数据，本讲常按 token 数算。<br>
    <b>张量与隐藏维</b>：模型里流动的数据都是多维数组，叫张量；每个 token 用一串数表示，这串数的长度就是隐藏维（V3 是 7,168）。<br>
    <b>激活</b>：前向每一层算出来的中间结果。反向时还要用，所以得先存着。<br>
    <b>注意力头</b>：注意力被拆成几十份并排的小注意力，每一份叫一个头，各算各的。<br>
    <b>micro-batch</b>：把一步要算的一批样本再切成几小份，一份一份地过。<br>
    <b>V3</b>：本讲反复拿来举例的 DeepSeek-V3，6,710 亿参数的 MoE 模型。<br>
    <b>Q、K、V</b>：注意力里每个 token 算出三样东西：Q 是「我要找什么」，K 是「我是什么」，V 是「我带着什么内容」。
    新 token 拿自己的 Q 去和前面所有 token 的 K 比，按相似度把它们的 V 加权合起来。<br>
    <b>查询头与 KV 头</b>：注意力分很多个头，每个头都有自己的 Q；K 和 V 的头可以更少，几个查询头共用一组 KV（极端情况只有一个 KV 头）。推理时缓存的是 KV，所以 KV 头的个数决定了 KV cache 能不能按头切开。<br>
    <b>softmax</b>：注意力给前面每个 token 打分，再把分数归一化成加起来等于 1 的权重，这一步叫 softmax。<br>
    <b>MLA</b>：DeepSeek 的一种注意力，把 K 和 V 压成一小份所有头共用，所以 KV cache 小得多，但也只剩「一个头」可切。<br>
    <b>节点</b>：一台机器，里面通常有 4 到 8 张卡，机器内部走高速互联（NVIDIA 的叫 NVLink）。<br>
    <b>显存（HBM）</b>：卡上自带的高速内存，模型和中间结果都得放在这儿。<br>
    <b>算力和带宽</b>：算力是每秒能算多少次，带宽是每秒能搬多少字节。一件活要是搬得多算得少，就是「吃带宽」，反过来是「吃算力」。<br>
    <b>ICI 和切片</b>：TPU 芯片之间的专用快线叫 ICI；用 ICI 连在一起的一整块芯片叫一个切片，切片和切片之间走普通的数据中心网络。<br>
    <b>device 和轴</b>：v7 的一颗芯片，对软件显示成 2 个 device，并行度按 device 数；切片里的芯片排成三维网格，x、y、z 三个方向各叫一根轴。<br>
    <b>rank</b>：参与并行的一个进程，通常就对应一张卡（TPU 上是一个 device）。<br>
    <b>v7 与 v7x</b>：同一代 TPU（Ironwood）的两种叫法，本讲混用。<br>
    <b>几路</b>：「TP 8 路」就是 8 张卡一组做 TP。写成 TP4 × DP2，意思是每 4 张卡一组做 TP，这样的组有 2 份做数据并行，一共 4 × 2 ＝ 8 张卡。<br>
    <b>MoE 与专家</b>：模型里有很多组并排的前馈层，叫专家；每个 token 由一个小的路由器挑出其中几个去算，其余的不碰。
    所以参数很多，每个 token 实际用到的却很少。<br>
    <b>KV cache</b>：生成文字时，每个已经处理过的 token 都留下一份 K 和 V，后面的 token 算注意力时要回头看它们。
    这份东西要一直存着，上下文越长越大。<br>
    <b>prefill 和 decode</b>：推理分两段。prefill 一口气读完整个 prompt，算出第一个字；decode 之后每一步只出一个字。
    decode 这一步很轻，所以要把很多请求拼成一批一起跑：<b>权重读一遍，一批请求一起用</b>，batch 越大越划算。</p></details>
</div></section>

''' + sec("s一", "一", "先认识五种通信") + '''
  <p class="lead">后面每一刀都会多出一种通信。这一节先把它们认全：日常用到的是五种，AllReduce、AllGather、ReduceScatter、AllToAll，再加最朴素的一对一收发。
    名字看着多，但每一个都只回答两个问题：<b>谁发给谁</b>；数据到了之后是<b>拼起来、加起来，还是原样放着</b>。</p>

  <h3>1.1　先学会看图</h3>
  <p>一组卡按同一个规则一起发、一起收，叫<b>集合通信</b>。下面几张图用同一套画法，只学一次：</p>
  <ul>
    <li>四张卡一张一个颜色（卡 0 蓝、1 橙、2 绿、3 紫），每张卡的数据切成四块。</li>
    <li>动画是黑底，<b>一列是一张卡</b>（静态图里一行是一张卡）。</li>
  </ul>

  <h3>1.2　一个人对所有人：四个基本动作</h3>
  <p>把四张卡想成四个同学，卡 0 是班长。</p>
__FIG_COLL_1N__
<details class="foldfig"><summary><b>看它们动起来</b>：四支小动画，一步一停</summary>
<div class="animgrid"><figure class="animcell" id="anim-broadcast"><video src="media/topic05-broadcast.mp4" autoplay loop muted playsinline aria-label="Broadcast 广播 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：Broadcast 广播：一份 → 人人一份。字幕：卡 0 的整份数据，复制给每一个人。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>Broadcast 广播</b>：卡 0 的整份数据，复制给每一个人<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-scatter"><video src="media/topic05-scatter.mp4" autoplay loop muted playsinline aria-label="Scatter 分发 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：Scatter 分发：一份拆开 → 一人一块。字幕：卡 0 把第 j 块发给卡 j，自己只留第 0 块。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>Scatter 分发</b>：卡 0 把第 j 块发给卡 j，自己只留第 0 块<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-gather"><video src="media/topic05-gather.mp4" autoplay loop muted playsinline aria-label="Gather 收集 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：Gather 收集：一人一块 → 拼成一份。字幕：每人把自己那块交给卡 0，卡 0 按顺序拼起来。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>Gather 收集</b>：每人把自己那块交给卡 0，卡 0 按顺序拼起来<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-reduce"><video src="media/topic05-reduce.mp4" autoplay loop muted playsinline aria-label="Reduce 归约 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：Reduce 归约：人人一份 → 加成一份。字幕：每人把整份交给卡 0，卡 0 逐块相加。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>Reduce 归约</b>：每人把整份交给卡 0，卡 0 逐块相加<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure></div>
</details>
  <p>这四个都有一个「班长」，数据全压在它一条线上，卡一多就堵。所以训练里天天跑的是下面那组。</p>
  <details class="foldfig"><summary><b>细一点</b>：班长到底怎么堵、通信库怎么缓解</summary>
  <p>按最朴素的做法（班长挨个发、挨个收），广播和归约的班长要扛下全部流量，卡越多越堵；
    通信库会把它们排成一条链接力传，让班长只发或只收一份。收集和分发的班长省不掉那份量，它手里本来就是 n 份不同的东西：总量不随卡数涨，但全压在它一条线上。</p></details>

  <h3>1.3　人人对人人：训练里天天在跑的四个</h3>
__FIG_COLL_NN__
<div class="animgrid"><figure class="animcell" id="anim-allgather"><video src="media/topic05-allgather.mp4" autoplay loop muted playsinline aria-label="AllGather 全收集 动画，按环一步一步走。四张卡排成一排，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。卡名之间有往右的箭头，卡片下面一条虚线车道表示卡 3 绕回卡 0。开始时卡 k 只有第 k 块。三步里，每一步四张卡同时把上一步刚拿到的那块发给右边的人，卡 3 那块从右边出去、沿车道绕回卡 0，落地后停住。三步之后人人一整份，只拼不加。最后画面复位。"></video><figcaption><b>AllGather 全收集</b>：按环转三步，每步人人把刚拿到的那块传给右边，只拼不加<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-reducescatter"><video src="media/topic05-reducescatter.mp4" autoplay loop muted playsinline aria-label="ReduceScatter 归约分散 动画，按环一步一步走。四张卡排成一排，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。卡名之间有往右的箭头，卡片下面一条虚线车道表示卡 3 绕回卡 0。开始时人人一整份。三步里，每一步四张卡同时往右发一块，收到的加到自己那块上，条纹多一种颜色，卡 3 那块沿车道绕回卡 0，落地后停住。三步之后卡 k 恰好握着第 k 块的完整总和，其余中间结果淡出。最后画面复位。"></video><figcaption><b>ReduceScatter 归约分散</b>：按环转三步，每步人人往右发一块、收的人加上；三步后卡 j 握着第 j 块的总和<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-allreduce"><video src="media/topic05-allreduce.mp4" autoplay loop muted playsinline aria-label="AllReduce 全归约 动画，按环一步一步走，跟 1.5 的环是同一支。四张卡排成一排，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。卡名之间有往右的箭头，卡片下面一条虚线车道表示卡 3 绕回卡 0。前三步是 ReduceScatter：人人往右发一块，收的人加上；三步后每张卡握着一块完整总和，其余中间结果淡出。后三步是 AllGather：把总和接着往右传，只替换不相加。每一步落地后停住。最后人人一份总和，画面复位。"></video><figcaption><b>AllReduce 全归约</b>：同一个环转两圈，先加三步（ReduceScatter）、再拼三步（AllGather）<span class="sub">（8 秒无声循环，Manim 渲染。）</span></figcaption></figure></div>
  <p>All ＝「人人都拿到结果」：AllGather 是收集完发给每个人，ReduceScatter 是归约完切开、一人一块，AllReduce 是归约完发给每个人；AllToAll 独一份，不加也不拼。</p>

  <h3>1.4　AllReduce 可以拆成两半</h3>
__FIG_AR_SPLIT__
  <p>拆开以后，中间还能塞进别的动作。后面好几种并行白捡的便宜全从这里来：ZeRO（第二节）、张量并行配序列并行（第三节）。</p>

  <h3>1.5　没有班长，怎么做到的：环</h3>
  <p>班长模式下卡 0 要收 n−1 份、发 n−1 份，成了全场的瓶颈。换个办法：首尾相连排成一圈，谁都不当班长。</p>
__FIG_RING__
<figure class="fbox fwide" id="anim-ring">
<video src="media/topic05-ring.mp4" autoplay loop muted playsinline
       aria-label="环形 AllReduce 动画。四张卡排成一排，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。卡名之间有往右的箭头，卡片下面一条虚线车道表示卡 3 绕回卡 0。字幕依次是：ReduceScatter，每一步人人同时往右发一块，收到的加到自己那块上；第 1、2、3 步各自写明 0→1、1→2、2→3、卡 3 从右边绕回卡 0，落地后写「第几步完成：粗框那块又多加进了一个人」并停住。三步之后每张卡恰好握着一块完整总和，其余中间结果淡出。接着 AllGather，把总和往右再传三步，只替换不相加。最后人人一份总和，画面复位。"></video>
<figcaption>同一个环转两圈：前三步<b>加</b>（ReduceScatter），后三步<b>拼</b>（AllGather），合起来就是一次 AllReduce。卡 3 发出的那块从右边出去、绕回卡 0。
  <span class="sub">（15 秒无声循环，Manim 渲染。）</span></figure>
  <p>这个环，其实就是把 AllReduce 拼成了 ReduceScatter ＋ AllGather。换一种拼法比一比，就知道它为什么赢：</p>
__FIG_AR_TWO_WAYS__
  <p>代价是步数跟着卡数涨：数据小时比的是一步步的等待，环反而吃亏，所以通信库会按数据大小自己挑算法。</p>
  <details class="foldfig"><summary><b>细一点</b>：环形和树形怎么挑、NVSwitch 在交换机里做加法、TPU 的环面</summary>
  <p>数据很大时比的是带宽，环几乎是最优的；数据很小时比的是一步一步的等待，步数多反而吃亏。
    所以 NCCL 这类通信库会按消息大小，在环形、树形等几种算法之间自己挑。
    NVLink 交换机（NVSwitch）还能在交换机里直接做加法，每张卡发出去的又能少将近一半。</p>
  <p>TPU 这边更直接：芯片之间的 ICI 本身就连成环面，切片够大（每一维都是 4 的整倍数）时，每一维天然就是一个首尾相接的环；
    更小的切片某一维只是一条线，环要在线上折返，带宽约减半。</p></details>

  <h3>1.6　AllToAll：每人给每人一份不一样的</h3>
__FIG_A2A__
<figure class="fbox fwide" id="anim-a2a">
<video src="media/topic05-a2a.mp4" autoplay loop muted playsinline
       aria-label="AllToAll 动画。四张卡，每张卡分「寄出」「收到」两列；块的名字是谁出的加要寄给谁，A1 是卡 0 出的、要寄给卡 1，颜色表示出自哪张卡。先把自己留给自己的四块挪到右列，不走网络。第 1 步人人寄给右边第 1 个人，第 2 步直接寄给右边第 2 个人，第 3 步寄给右边第 3 个人，到头的从右边出去、沿卡片下面的车道绕回来；每一步落地后停住。三步之后卡 j 的右列收齐了四个人寄给它的那一份。字幕提示专家算完还要原路寄回一次，MoE 每层两次 AllToAll。最后画面复位。"></video>
<figcaption>三步寄完：第 s 步人人直接寄给右边第 s 个人，到头的绕回来；专家算完还要原路寄回一次。
  <span class="sub">（8 秒无声循环，Manim 渲染。）</span></figure>
  <p>网络最怕它：每一份都是写给某一个人的私信，路上没法像 AllReduce 那样边走边合并；排不成只跟邻居说话的环，任意两张卡之间都有东西要走，拼的是整个网络的横截面有多宽；而且每份多大，要等模型算到这一层才知道。
    专家并行（第四节）、Ulysses（第五节）都用它。</p>

  <h3>1.7　一张表收住</h3>
  <p>八个名字里带班长的四个只是积木，日常用的是五种（最后一种一对一收发，第三节切层时用）。给会后查：</p>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">「每卡发出」按环形算法算；S 是一整份数据（AllGather 指拼好之后那份，ReduceScatter 指加之前那份）；Broadcast／Reduce 按链式接力算。</caption>
    <tr><th>通信</th><th>做什么</th><th>每卡发出</th><th>后面谁在用</th></tr>
    <tr><td>AllReduce</td><td>加完，人人一份</td><td>2(n−1)/n · S</td><td>数据并行同步梯度；张量并行每层前向两次、反向两次</td></tr>
    <tr><td>ReduceScatter</td><td>加完，各拿一块</td><td>(n−1)/n · S</td><td>FSDP 反向分梯度；张量并行配序列并行</td></tr>
    <tr><td>AllGather</td><td>拼完，人人一份</td><td>(n−1)/n · S</td><td>FSDP 前向拼权重；张量并行配序列并行；推理切 KV 时收齐 Q</td></tr>
    <tr><td>AllToAll</td><td>只换位置</td><td>(n−1)/n · S</td><td>专家并行派发和收回 token；Ulysses</td></tr>
    <tr><td>Broadcast / Reduce</td><td>一对多 / 多对一</td><td>S</td><td>分发初始权重、汇总指标这类场合</td></tr>
    <tr><td>Send / Recv</td><td>一对一</td><td>看传多少</td><td>流水线并行在 stage 之间传激活；Ring Attention 沿环传 KV</td></tr>
  </table>

  <div class="note ok"><span class="t">这一节只要带走两件事</span>
    <b>① AllReduce 可以拆成 ReduceScatter 和 AllGather 两半</b>，后面好几刀都靠它白捡便宜。<br>
    <b>② AllToAll 是唯一一个人人对人人发不同数据的</b>，专家并行离不开它，网络也最怕它。<br>
    后面五刀，每一刀多出来的通信都是这张表里的某一行。话认全了，第一刀：切数据。</div>
  <details class="foldfig"><summary><b>考考自己</b>：四张卡做一次 AllReduce：环形做法下每张卡一共发出几份数据？最朴素的班长做法下，卡 0 要发几份？</summary>
  <p>环：ReduceScatter 3 步各发 1/4 份，AllGather 再 3 步，一共 1.5 份；班长：卡 0 发 3 份。卡越多差得越远。</p></details>
</div></section>

''' + sec("s二", "二", "第一刀：切数据") + '''
  <p class="lead">最朴素的一刀是切数据：每张卡算不同的样本。
    <b>可它一开始根本没解决「装不下」</b>，要一路削到 FSDP 才解决 —— 而最后那一步是要付钱的。</p>

  <h3>2.1　数据并行：每张卡一整份模型</h3>
  <p>每张卡放一整份模型、各算一批样本；反向算完做一次 AllReduce 求平均，再一起更新。
    一步只通信一次（频率最低，量却不小：V3 这么大的模型要是纯数据并行，每卡一步约发 2.7 TB）。<b>每张卡还是存一整份 16 字节／参数 —— 装不下一点没变。</b></p>
  <p>打个比方：V3 的全部权重是一整套四库全书，分放在 61 间书房里，一间一层。按 V3 的配置有 128 个阁（一个阁就是一张卡），数据并行等于每个阁都藏一整套。下面一节一节把重复的清掉。</p>

  <h3>2.2　ZeRO：越闲的越先削</h3>
  <p>先认清那三块：<b>权重</b>是模型本身；<b>梯度</b>是这一步算出来的「每个参数该往哪改、改多少」；<b>优化器状态</b>是优化器替每个参数记的账（主权重、两个动量），它最大。下文的 Ψ 就是参数个数。</p>
  <p class="sub">口径：本课按 ZeRO 论文算，优化器状态指<b>跨步常驻、要存进 checkpoint</b> 的那几样；梯度每步重算、用完清零，单独算一块（所以 ZeRO 才能把「切梯度」单列一级）。Megatron 里累加梯度的精度（<code>--main-grads-dtype</code>）跟主权重、两个动量的精度归在同一组「精度感知优化器」参数里配，所以也有人把它算进优化器。</p>
__FIG_ZERO_MEM__
  <p>16 字节里，12 个是优化器状态 —— 开场第二问的答案：最大的是它，先削它。巧的是，它也是最闲的那块：</p>
__FIG_ZERO_BUSY__
  <p><b>ZeRO-1</b> 切优化器状态，<b>ZeRO-2</b> 再切梯度，<b>ZeRO-3</b> 连权重也切。每张卡只长期存自己负责的那 1/n。
    ZeRO-1 顺手还省了一笔计算：原来每张卡都把全部参数更新一遍，现在各更新各的 1/n。</p>
  <p>前两级为什么不多花一个字节的通信？就是 §1.4 那个等式：</p>
__FIG_ZERO_STEP__
__Q3_ANSWER__
  <details class="foldfig"><summary><b>细一点</b>：动量为什么压得了、主权重为什么压不得 —— 一把刻度尺</summary>
__FIG_BF16_BETA__
  </details>

  <h3>2.3　ZeRO-3 ＝ FSDP：最后那 2 个字节要付 50%</h3>
  <p>削完前两级，每参数还剩 2 字节的权重。V3 按 128 路算，每卡仍要 1.29 TiB（约 1,415 GB，一张卡才两三百 GB），照样装不下。
    ZeRO-2 加再多卡也降不下去了：剩下的正是人人一整份的权重。只靠数据并行这一刀，大模型只能走到 ZeRO-3，也就是 PyTorch 里的 <b>FSDP</b>：每层要算之前先 AllGather 拼回这一层，算完就扔。</p>
  <p>回到藏书楼。FSDP 切的时候<b>根本不管书是什么</b>：把一间书房的书一页页排成一长条，按长度等分成 128 段，每阁拿一段。切口常常落在一本书中间，所以每阁手里那段，可能是一本书的后半截接着另一本的开头 —— <b>单拿这一段谁也读不了</b>。
    要用这一间，就得把其余 127 段都<b>抄</b>过来，凑回整间（AllGather），读完把抄本扔掉，只留自己那段；反向算完，每一段的修改意见交回管那段的阁合并（ReduceScatter）。</p>
__FIG_SIKU__
  <p>注意跟后面 EP 的区别：EP 按整架分，每架都是一个完整专家，单独就能算；FSDP 切的是碎片，为的是通信好切块，<b>拼回来之前没用</b>。</p>
__FIG_FSDP_STEP__
<figure class="fbox fwide" id="anim-fsdp">
<video src="media/topic05-fsdp.mp4" autoplay loop muted playsinline
       aria-label="FSDP 一步的动画。四张卡，每张卡三层，每层只长期持有自己那一段（四分之一）。标题：FSDP：每层用之前把别人那几段复制过来，用完就扔。字幕一：前向：每层先 AllGather，把别人那几段复制过来拼成整层，算完扔掉复制件。第 1、2、3 层依次：别人那三段各复制一份飞进来拼成整层（原件留在原处），整层亮一下，再把复制来的三段扔掉。底部计数一次次加一。字幕二：反向：扔掉的权重要再拼一次；算出的梯度 ReduceScatter 给各自的主人。从第 3 层往回：再拼一次整层、亮一下、扔掉，然后黄色的梯度小块飞回各自的主人。字幕三：每层三次：前向拼一次，反向再拼一次、散一次。计数停在 9。最后复位。"></video>
<figcaption>整层只在用的那一刻出现；平时每张卡只拿着每层的四分之一。
  <span class="sub">（14 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>通信是 3Ψ，<b>数据并行的 1.5 倍</b>；换来的是每卡常驻从 9.76 TiB 降到 78.11 GiB（约 84 GB，还不含激活）。可 V3 用的 H800 一张才 80 GB，放进去就满了 —— 光切数据还不够，后面几刀就是这么来的。</p>
  <div class="note warn"><span class="t">⚠️ 哪几级白送，哪一级要付钱</span>
    ZeRO-1、ZeRO-2 白送（切 micro-batch 时只剩 ZeRO-1，原因见「ZeRO 的一步」那张图的出处）；ZeRO-3 多付一半，对大模型来说非付不可。</div>

  <h3>2.4　这一刀能放多远：看频率</h3>
  <p>数据并行一步说一次话，能放在慢链路上；FSDP 每层都要说，得放在快线里。</p>
  <details class="foldfig"><summary><b>细一点</b>：数据并行、FSDP、HSDP、DiLoCo 各能放多远</summary>
  <ul>
    <li><b>数据并行、ZeRO-1</b>：一步通信一次，可以放在慢链路上，最远能横跨数据中心。</li>
    <li><b>FSDP</b>：每一层都要拼一次权重，一步下来多出成百上千次（第七节有个数），得放在高带宽域里。</li>
    <li><b>HSDP</b> 是两者的折中：<b>机内 FSDP、机间数据并行</b>。高频的拼权重留在机内，跨机只剩梯度同步，量也除以了机内的分片数。</li>
    <li><b>DiLoCo</b> 再往前一步：每个副本先自己走几百步再同步一次，专门为跨数据中心训练设计。</li>
  </ul></details>

  <h3>2.5　这一刀留下的问题</h3>
  <p>FSDP 每一步搬的是<b>权重</b>，搬多少只跟参数量有关，<b>跟 batch 无关</b>；
    而每一步要算多少，跟每张卡分到的 token 数成正比。</p>
  <p>一次喂进去的总 batch 又不能跟着卡数无限加，加太大模型反而学不好；所以卡越多，每卡分到的越少。
    每张卡的 batch 一小，算得少、搬得一样多，时间就被搬权重吃掉了。
    <b>要继续加卡，又不想让每张卡越算越少，还能怎么切？</b></p>
  <p>还有一句话能帮你分清下一刀：FSDP <b>形式上</b>把模型切开了，<b>实质</b>还是数据并行 —— 算的那一刻，权重已经拼回一整份，每张卡算的是自己那批数据。
    下一刀要的是：算的时候，权重也不拼回来。</p>
  <details class="foldfig"><summary><b>考考自己</b>：ZeRO 三级里，哪两级不多花一个字节的通信？为什么先削优化器状态？</summary>
  <p>ZeRO-1、ZeRO-2 不多花（AllReduce 本来就能拆成 RS＋AG，中间那步本地更新不用传）。优化器状态最大，又最闲，只在更新那一下用。</p></details>
</div></section>

''' + sec("s三", "三", "第二刀：切权重") + '''
  <p class="lead">FSDP 每一层都要把整层权重拼回来。只要每张卡的 batch 够大，这笔搬运能藏在计算后面；
    <b>batch 一小，就藏不住了。</b>第二刀不再拼权重，直接把权重切开：几张卡合起来算同一批 token，每张卡要处理的 token 数不会随加卡被摊薄。
    这一节会连撞三堵墙：batch 一小，FSDP 被拖住，换 TP；TP 出不了一台机器，换 PP；PP 又有人闲着等。</p>

  <h3>3.1　FSDP 的尽头：搬一个字节，换来多少计算</h3>
  <p>先问一个朴素的问题：<b>一份数据搬过来一次，能被用几回？</b>用的回数多，搬的那点时间就藏得住；只用一回，卡就在等搬运。
    换成数字，就是一个比值：<b>每在网络上搬一个字节，能换来多少次计算</b>。它要高过硬件自己的比值（每秒算多少次 ÷ 每秒搬多少字节），否则卡就在干等。</p>
  <p>FSDP 这笔账很干净（⚠️ 推导，按稠密模型算）：一步搬约 6Ψ 字节（前向、反向各拼一次 bf16 权重，再分一次梯度，各约 2Ψ），
    算 6ΨT 次（专题四「每个 token 约 6 倍参数量次运算」，T 是每张卡分到的 token 数）。<b>两者一除，正好等于 T</b>，跟模型多大没关系。
    更直观的看法：一个参数占 2 字节，每个 token 拿它做一次乘加，正好 2 次运算。所以送来一个参数，这张卡上有几个 token，它就被用几回。</p>
__FIG_INTENSITY__
  <p>红虚线约 3,845：<b>每颗芯片一步分不到三千八百多个 token，FSDP 就被拖住</b>。</p>
  <p>分母要当心：<b>该用你在自己网络上实测的带宽，不是规格表上的数</b>。门槛 ＝ 每秒能算多少次 ÷ 每秒实际拼得动多少字节的权重。
    测法是真跑一次 AllGather（GPU 上用 nccl-tests），看它报的 busbw。它按 AllGather 的口径算：数据量 ÷ 用时 × (n−1)/n，扣掉了自己那份本来就有、不用收的部分，正好对上每张卡每秒实际收到的字节。
    实测只跑到标称的七成，门槛就从 3,845 抬到约 5,493（图上细虚线，七成是假设的）。FSDP 那根轴要是只占几根链路、还跟别的轴抢线，这个数还得往上走。</p>
  <details class="foldfig"><summary><b>细一点</b>：3,845 怎么来的，什么情况下门槛更高</summary>
  <p>红虚线 3,845 ＝ v7 每芯片 2,307 TFLOP/s（bf16）÷ 每芯片发出约 0.6 TB/s（常说的 1.2 TB/s 是收发合计；⚠️ 这个拆分是推算，见台账）。
    这还是最乐观的线：只走一根轴，高约 3 倍；
    MoE 只靠 FSDP、不配专家并行，搬的是全部参数、算的只有被选中的那部分，再乘约 18 倍（V3 的 6,710 亿 ÷ 370 亿）。</p></details>

  <h3>3.2　TP：切进矩阵内部</h3>
  <p>张量并行把一层的权重矩阵本身切开，每张卡只存、只算其中一块。英伟达 Megatron-LM 的切法：</p>
__FIG_TP_MLP__
<figure class="fbox fwide" id="anim-tpsplit">
<video src="media/topic05-tpsplit.mp4" autoplay loop muted playsinline
       aria-label="张量并行切一个 MLP 的动画。两张卡，卡 0 蓝、卡 1 橙，各有完整的 X（灰）；W1 画成宽矩阵竖切一刀、W2 画成高矩阵横切一刀，每张卡只亮自己那一半。标题：张量并行切一个 MLP：Y ＝ GeLU(X·W1)·W2。字幕一：① W1 按列切：每张卡算出中间结果的一半。字幕二：② 激活函数逐元素算：各算各的，这一段没有任何通信。字幕三：③ W2 按行切：每张卡只得到 Y 的一个部分和。字幕四：④ AllReduce：两份部分和相加，两张卡都拿到完整的 Y（中间一个绿框标 AllReduce，Y 是灰色的完整副本）。字幕五：整个 MLP 只在最后通信一次 —— 代价是每一层都有这一次。最后复位。"></video>
<figcaption>通信只在最后那一下；可每一层都有这一下。
  <span class="sub">（9 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>两个容易绕晕的地方。<b>一、每张卡两块都有份</b>：卡 0 拿上投影（UP）的一半，也拿下投影（DOWN）的一半，不是一张卡管 UP、另一张管 DOWN（那是按层切，后面讲的 PP）。
    两块之间不通信，DOWN 算完才把部分和加一次，传的是激活。attention 那块同理：按头切，出口加一次。所以一层前向两次 AllReduce，反向再两次。</p>
  <p><b>二、「横」「竖」只是画法</b>：本课把 W1 画成「输入维 × 输出维」（X·W1），第一刀切的是输出维，看着是竖切；PyTorch 的 nn.Linear 把权重存成「输出维 × 输入维」，同一刀在它那里就是横切。
    所以别记横竖，记<b>第一块切输出维，第二块切输入维</b>。Megatron 按 X·A 的画法起名，叫列并行（ColumnParallelLinear）和行并行（RowParallelLinear）。</p>
  <p>顺序不能反过来，因为两次矩阵乘中间夹着一个不是线性的 GeLU：</p>
__FIG_TP_ORDER__

  <h3>3.3　SP：TP 的搭档</h3>
  <p>TP 切不到的那几段小运算，由它的搭档 SP 沿序列切开：<b>通信量不变，省激活显存</b>。别被名字骗了，它不是第四刀（§8.8）。</p>
  <details class="foldfig"><summary><b>细一点</b>：SP 切的是哪几段、怎么做到通信量不变</summary>
  <p>TP 切不到的是逐个 token 的小运算（归一化、dropout、残差相加），那几段每张卡都存着一整份激活。
    Megatron 的序列并行（SP）把它们沿序列切开，顺手把每次 AllReduce 拆成两半：进 TP 那块之前 AllGather 拼齐序列，出来时 ReduceScatter 切回。
    就是第一节那个等式，所以通信量跟原来一样。</p></details>

  <h3>3.4　TP 的上限：跟 batch 无关</h3>
  <p>TP 送的是「半成品」（激活）：token 多一倍，要送的和要算的都多一倍，<b>batch 在账里约掉了</b>。
    FSDP 送的权重像工具，能提前送（算这一层时就把下一层的拿过来）；TP 送的半成品是刚算出来的，默认只能等它算完再送，所以 TP 的通信很难藏进计算里。
    剩下的只有隐藏维和 TP 度数：每字节换来的计算约 4.5 × 隐藏维 ÷ TP 度数（⚠️ 推导，稠密层）。
    V3 的隐藏维 7,168，TP 8 路约 4,032。图上那条红线是<b>最乐观</b>的画法，所以看着刚好贴线（7,168 × 4.5 ÷ 3,845 ≈ 8.4）；真实的红线还要更高，TP 8 路其实已经在线下（原因在下面「细一点」）。</p>
  <p>所以 <b>batch 小多用 TP，batch 大多用 FSDP</b>：图上左边蓝线还没爬过红线，靠橙线；右边爬过去了，交给 FSDP。
    前提是 TP 那条线本身在红线上面。像 V3 这样两条都贴着或落在线下，TP 本身就不划算（§3.7）。</p>
  <details class="foldfig"><summary><b>细一点</b>：为什么说实际已经在线下；TP 度数还受哪两条约束</summary>
  <ul>
    <li>① 3,845 假设三根轴都用满、而且每根轴首尾连成环；TP 8 路只有 4 颗芯片（v7 一颗芯片算 2 个 device），两个折扣叠在一起：切片太小成不了环（§1.5），又只用得上一两根轴（§3.1），实际门槛高好几倍。</li>
    <li>② Megatron 的 TP 通信默认要等它做完才能往下算，藏不进计算里。</li>
    <li>注意力头要按整个分：Megatron 要求查询头数能被 TP 度数整除（KV 头更少的模型，KV 头数和 TP 只要一个能整除另一个，TP 更大时 KV 就复制）。</li>
    <li>TP 每层都要通信，只能待在最快的那一圈互联里（GPU 上一般不出一个 NVLink 域）。</li>
  </ul></details>

  <h3>3.5　FSDP 和 TP 一起用：二维切</h3>
  <p>两把刀能并存，大模型训练里也常常一起用。关键是<b>切在不同的维上</b>：把卡排成一张表，一行是一组 TP，一列是一组 FSDP。TP 按输出维把权重切成几条，FSDP 再把每一条按输入维劈开。</p>
__FIG_FSDP_TP__
  <p>算一层时，先在<b>列</b>里 AllGather，把自己那一条拼齐。拼回的只是整矩阵的 1/TP，不是整个矩阵。再各算各的，最后在<b>行</b>里把部分和加一次。反向时，梯度按列 ReduceScatter 回各自的主人。</p>
  <p>账也跟着变：同一行的几张卡用的是同一批 token，每张卡却只拼 1/TP 条，所以 FSDP 那根轴的门槛按<b>一行的 token 数</b>算。TP 4 路时，摊到每张卡只要约 961 个（⚠️ 推导，没按各轴的实测带宽修正）。
    这才是「batch 小多用 TP」的真正意思：<b>TP 把一行的 token 借给了 FSDP</b>。代价是 TP 那根轴每层都要对账，得占最快的线。</p>

  <h3>3.6　PP：按层切</h3>
  <p>要横跨很多台机器、走慢线，又不想像 FSDP 那样每层搬权重，就按层切。TP 是<b>切宽</b>，PP 是<b>切深</b>：</p>
__FIG_WIDE_DEEP__
  <p>段与段之间只在边界上用一对一收发传激活，是所有刀里通信最少的，所以它<b>能跨到慢线上</b>。
    可朴素地切，就是「4 张卡的显存、1 张卡的速度」：同一时刻只有一段在干活。于是把一批数据切成几份 micro-batch 轮着灌进去，像装修队：
    泥瓦、水电、油漆三队一户一户往下做，第一户开工时后两队只能在楼道里等，最后一户只剩油漆在干。等的那段，叫<b>气泡</b>：</p>
__FIG_PP_BUBBLE__
<figure class="fbox fwide" id="anim-pipeline">
<video src="media/topic05-pipeline.mp4" autoplay loop muted playsinline
       aria-label="流水线并行时间表的动画。四个 stage，时间轴一格一格长出来，蓝色是前向、绿色是反向、深灰是空等的气泡。标题：流水线并行：灰色是气泡，每个 stage 都在空等的时间。字幕一：先用 4 个 micro-batch：气泡 ÷ 理想计算时间 ＝ (4−1) ÷ 4 ＝ 3/4。字幕二：总量不变、切成 8 份（每份的格子窄一半）：气泡 ＝ (4−1) ÷ 8 ＝ 3/8，灰色真的短了一半。字幕三：气泡只能摊薄、不能消灭：份数越多越省，可每份太小，卡就吃不饱。最后复位。"></video>
<figcaption>micro-batch 从 4 个加到 8 个，灰色的气泡跟着缩一半。
  <span class="sub">（14 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>气泡跟理想计算时间之比是 (p−1) ÷ m，p 是段数，m 是 micro-batch 个数。份数越多越省，可每份太小，卡就吃不饱。</p>
  <details class="foldfig"><summary><b>细一点</b>：激活为什么越攒越多；三种常见的改进</summary>
  <p>每一行空白的总长都一样，是 p−1 个前向加 p−1 个反向。在最早的 GPipe 调度里，每一份的激活都要攒着等反向，份数越多攒得越多；
    1F1B 调度把同时在路上的份数限制在 p 以内，气泡一样大，省的是激活显存 —— 这也是个反转：PP 把参数切开了，每张卡的激活却没少多少，因为每段都得替好几个 micro-batch 攒着。</p>
  <p>VPP 让每张卡负责几段不连续的层，气泡再除以每卡的段数；
    Zero Bubble 把反向拆成「算输入的梯度」和「算权重的梯度」，后者不急，挪去填空；
    DeepSeek-V3 用的 DualPipe 从流水线两头同时往里灌，更要紧的是把一对前向和反向的计算，跟专家并行的通信叠在一起藏掉（技术报告 sec. 3.2.1）。</p></details>

  <h3>3.7　这一刀留下的问题</h3>
  <p>还记得开场那行配置吗？DeepSeek-V3 里<b>没有 TP</b>。报告自己的解释是显存优化做得够细，用不着代价高的 TP。
    再看它的参数都在哪：几乎全在一种叫「专家」的窄矩阵里。专家本来就窄，再往里切，每份更小，要搬的激活却一点不少。<b>真正该切的，是「专家」这一维。</b>这是第三刀。</p>
  <details class="foldfig"><summary><b>考考自己</b>：每张卡分到的 token 很少时，FSDP 和 TP 哪个先被通信拖住？TP 为什么出不了一台机器？</summary>
  <p>FSDP：它每搬一个字节换来的计算就等于每卡 token 数，token 少就藏不住。TP 每一层都要对账，而且半成品要等算完才能送，藏不进计算，只能待在最快的那圈线里。</p></details>
</div></section>

''' + sec("s四", "四", "第三刀：切专家") + '''
  <p class="lead">MoE 模型的参数几乎全在专家里。第三刀就切这一维：把不同的专家放到不同的卡上。
    <b>它多出来的通信只有一种，AllToAll；可它也带来了一个别的刀里最严重的病。</b></p>

  <h3>4.1　为什么该切专家</h3>
  <p>V3 除了最前面 3 层，其余 58 层都是 MoE 层：每层 256 个路由专家，每个 token 由路由器挑其中 8 个去算；另有 1 个人人都要过的共享专家。</p>
__FIG_MOE_PARAMS__
  <p>256 个路由专家 × 58 层 ≈ 6,539 亿参数，<b>占全部的约 97%</b>。换句话说，TP 那把刀要切的是「一个很宽的矩阵」，而 V3 里真正占地方的是「很多个窄矩阵」。
    对后者，最省事的切法不是把每一个都劈开，而是<b>把它们整个分给不同的卡</b>：搬病人，不劈医生。图底那行就是 V3 训练的真实配置：PP16 ＋ EP64，一路 TP 都没用。</p>
  <p>回到第二节那座藏书楼：EP 是把每间的 256 个专家书架<b>整架</b>分给 64 个阁，每阁 4 架；PP 是把 61 间书房按顺序分成 16 段。都是按整件分，拿到手就能用 —— 跟 FSDP 切碎了再拼正好相反。</p>

  <h3>4.2　EP：token 飞去专家那里</h3>
  <p>专家并行（EP）的通信只在 MoE 层里发生，就是 1.6 讲的 AllToAll，网络最怕的那种，每层两次：
    <b>派发</b>，把每个 token 送到它选中的专家所在的卡；<b>合并</b>，算完再送回原来的卡。</p>
<figure class="fbox fwide" id="anim-ep">
<video src="media/topic05-ep.mp4" autoplay loop muted playsinline
       aria-label="专家并行的动画。四张卡，每张卡上方 4 个 token（颜色表示来自哪张卡），下方 2 个专家，共 8 个专家。标题：专家并行：token 飞到专家那里，算完再飞回来。字幕一：每个 token 由路由挑一个专家（真实的 V3 每个 token 挑 8 个）。字幕二：派发（AllToAll）：token 飞到专家所在的卡，在专家门口排队。专家 0 门口排了 7 个，其他专家 1 到 2 个。字幕三：专家 0 排了 7 个，别的专家只有 1 到 2 个：它算完之前，大家都得等。字幕四：合并（AllToAll）：算完再送回原来的卡。字幕五：发给谁由数据决定，负载天生不均 —— 这是专家并行最重的病。最后 token 回到原位。"></video>
<figcaption>派发、排队、合并。那一根排得最高的队，决定了所有卡什么时候能往下走。token 的颜色表示它从哪张卡出发，每张卡下方两个框是它的专家。
  <span class="sub">（10 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>V3 为了压住这两次 AllToAll，做了两件事：<b>每个 token 最多去 4 台机器</b>，跨机每台只发一份、到了再在机器里分；<b>派发压成 FP8</b>，合并还用 BF16。</p>
__FIG_EP_ROUTE__
  <p>这是训练和 prefill 的做法；decode 追求低延迟时改成按专家逐个直发（跳过机器里的转发，每个专家各收一份），份数就跟着选的专家数涨了。</p>

  <h3>4.3　EP 最重的病：负载由数据决定</h3>
  <p>别的刀负载均不均，大多事先就知道。<b>EP 的不均最严重</b>：
    哪个专家忙、哪个专家闲，要等路由算完才知道，而且每一批数据都不一样。
    最忙的那个专家算完之前，所有人都得等它。它还会越滚越大：越受欢迎的专家分到的 token 越多、学得越好、就越受欢迎，其余的专家荒掉 ——
    花一个大模型的显存，训出一个小模型。所以这既是速度病，也是质量病。治法分两头：</p>
__FIG_EP_BIAS__
  <ul>
    <li><b>训练时</b>：让路由本身尽量均匀。以前靠在训练目标里加一项「辅助损失」罚不均，但会拖累模型效果。
      打个比方，老办法是罚分诊员，逼他改诊断；V3 是在门口挂个号牌，只管排号、不管诊断。具体是：路由给每个专家打分、挑最高的 8 个，偏置是加在分上的一个数。每步结束看一眼整批的负载，超载的专家偏置减 0.001，欠载的加 0.001；偏置只影响挑谁，不改算出来的权重。
      辅助损失只留一个 0.0001 的兜底，防止单条序列里极端不均。因为够均匀，V3 训练时<b>一个 token 都没丢</b>：不设容量上限，不均就直接变成等待。</li>
    <li><b>推理时</b>：把热门专家多复制几份，摊到不同的卡上，也就是 EPLB（专家并行负载均衡器）和冗余专家。</li>
  </ul>

  <h3>4.4　attention 和专家，各配各的</h3>
  <p>一层 Transformer 里，attention 和专家是两种完全不同的形状（对不是 MoE 的那部分来说，EP 那一维其实就是数据并行）：attention 的负担跟序列和 KV 有关，
    专家的负担是那一大堆参数。<b>所以同一批卡，在这两部分可以用两套切法。</b></p>
__FIG_FOLD__
  <p>挑哪种配法，差别大到什么程度？我们自己测过一次：</p>
  <details class="foldfig"><summary><b>先认两个简称</b>：TEP、DEP</summary>
  <p>TEP 是 attention 用 TP、专家用 EP；DEP 是 attention 用数据并行、专家用 EP；字母后的数字是总卡数，DEP8 ＝ 8 张卡。
    上图左边那种配法，算两者中间。别和 Megatron 的 ETP／EDP 混，见 §8.6。</p></details>
  <div class="note ok"><span class="t">一次实测：换一种切法，每张卡的吞吐翻一倍</span>
    GB300 上跑 DeepSeek-V4-Pro（vLLM），decode 从 TP4 换成 DEP8（attention 数据并行 8 路、专家 EP8），同样并发下<b>每张卡的吞吐是调完参的 TP4 的 2.09 倍</b>。
    最属于「换切法」的一笔在 attention 那一半：KV 不再在 4 张卡上各存一份（§5.4 讲为什么）。attention 权重虽然每张卡要存一份，但在 MoE 模型里只占几个百分点。注意上面的图画的是 V3（每卡 32 个专家），实测用的是 V4-Pro。<br>
    <em>完整的账（卡数、并发、首字延迟）在 §7.3。</em></div>

  <h3>4.5　这一刀留下的问题</h3>
  <p>前三刀切的是 batch、权重、专家，都没碰过「序列」这一维。可上下文一长，训练时的激活、推理时的 KV cache，都跟着序列长度往上涨 ——
    <b>涨到一条样本都放不进一张卡时，怎么办？</b></p>
  <details class="foldfig"><summary><b>考考自己</b>：专家并行最重的病是什么？V3 用什么办法治，为什么不拖累模型学东西？</summary>
  <p>负载不均：最忙的专家决定所有人等多久。V3 给每个专家的分数挂一个偏置，超载减、欠载加；偏置只影响挑谁，不改算出来的权重。</p></details>
</div></section>

''' + sec("s五", "五", "第四刀：切序列") + '''
  <p class="lead"><b>一条样本自己放不进一张卡，那就把它沿序列切开。</b>这是第四刀。训练和推理切的东西不一样，分开讲。</p>

  <h3>5.1　一条样本为什么会放不下</h3>
  <ul>
    <li><b>训练</b>：每层要存的激活至少跟序列长度成正比；不用 FlashAttention 时还有一项跟长度的平方成正比（Megatron 序列并行论文的式 1）。</li>
    <li><b>推理</b>：KV cache 每个 token 都要存一份。V3 的 MLA 每 token 存一份压缩后的 KV（512 维）加一小段位置信息（64 维），(512 ＋ 64) × 61 层 × 2 字节（BF16）＝ 70,272 字节；再乘 131,072 个 token，
      <b>一个 128K 的请求就是约 8.58 GiB</b>，差不多是一张 80 GB 显卡的九分之一，这还只是一个请求。把 KV 想成每个字留下的「笔记」，后面的字都要回头翻它。</li>
  </ul>

  <h3>5.2　训练：CP 把激活沿序列切开</h3>
  <p>上下文并行（CP）把一条长序列切成几段，每张卡只存自己那段的激活。
    难点只在注意力：每个 token 要看到它前面所有的 token，而那些 token 在别的卡上。两种办法：</p>
  <ul>
    <li><b>Ring Attention</b>：把 KV 想成每个 token 留下的「笔记」，Q 是新 token 手里的「提问」。Q 不动，KV 沿环一段段传，每一步算手上这一对，同时把 KV 传给下一张。
      这就是 1.5 那个环：传下一段时正在算这一段，通信藏在计算后面。</li>
    <li><b>Ulysses</b>：先说一个词：注意力被拆成几十个并排的「头」，各看各的。Ulysses 在注意力前后各做一次 AllToAll（§1.6 那个转置），在「按序列切」和「按头切」之间来回换。
      换成按头切以后，每张卡手里是全部 token、但只有几个头，注意力就能在本卡算完，算完再换回去。
      每张卡的通信量在序列长度和卡数同比放大时保持不变；代价是并行度不能超过注意力头数：换过去以后每张卡至少要拿一个完整的头；KV 头比查询头少的模型（GQA），卡在 KV 头数上。</li>
  </ul>

<figure class="fbox fwide" id="anim-ringattn">
<video src="media/topic05-ringattention.mp4" autoplay loop muted playsinline
       aria-label="Ring Attention 动画。四张卡，每张卡左边固定一段 Q（Q0 到 Q3），旁边一段 KV。右边是 4 乘 4 的注意力块网格，行是哪张卡的 Q，列是哪段 KV。标题：Ring Attention：Q 不动，KV 沿环传。字幕一：每张卡固定一段 Q；每一步算手上这对（Q, KV），同时把 KV 传给下一张。四步里 KV 段一格格往下传，卡 3 的传回卡 0，网格每行逐格填满。字幕二：转完一圈：每张卡都跟所有 KV 算过了，自己那一行填满。字幕三：关键：传下一块的时候正在算这一块，通信藏在计算后面。最后复位。"></video>
<figcaption>Q 留在原地，KV 沿环转一圈，每张卡把自己那一行填满。
  <span class="sub">（9 秒无声循环，Manim 渲染。画的是不带因果掩码的情形。）</span></figcaption></figure>
<figure class="fbox fwide" id="anim-ulysses">
<video src="media/topic05-ulysses.mp4" autoplay loop muted playsinline
       aria-label="Ulysses 动画。标题：Ulysses：用两次 AllToAll，在「按序列切」和「按头切」之间换。四张卡，每张卡四格，颜色表示哪一段序列，格子里写「第 k 段 · 头 j」。字幕一：开始：每张卡拿一段序列，这一段的全部头都在（颜色 ＝ 哪一段）。字幕二：第一次 AllToAll：第 j 个头的那一格，送到卡 j。十六格同时飞到新位置，每张卡变成四种颜色、同一个头。字幕三：现在每张卡：全部序列、一个头 —— 这个头的注意力在本卡就能算完。四张卡外框亮黄一下。字幕四：第二次 AllToAll：算完再换回按序列切，接着往下走。十六格飞回原位。字幕五：代价：每层两次 AllToAll；卡数不能超过头数。"></video>
<figcaption>一次转置，每张卡就有了一个头的全部序列，注意力不用再问别人。
  <span class="sub">（10 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>一句话记住两者：<b>Ring 是人不动、笔记转圈；Ulysses 是换个切法，每人拿全部笔记的几个头。</b>
    两者可以叠起来用（USP：卡排成二维，一个维度走环，另一个维度走 AllToAll），Megatron 的 CP 也支持分层组合。</p>

  <h3>5.3　causal 带来的不均</h3>
  <p>生成式模型的注意力有因果掩码：每个 token 只看前面的。于是越靠后的段算得越多，顺序切会让最后一张卡累死。
    解法很简单：轻的配重的，一张卡拿一轻一重两块，叫之字形切，见下图。</p>
__FIG_CP_ZIGZAG__

  <h3>5.4　推理：KV cache 被 TP 复制了</h3>
  <p>长上下文、高并发的推理（decode 阶段）里，KV 常常是最大的一块。先记住一句：<b>TP 按头切，DCP 按字切</b>，下面两段就是这句话的展开。TP 在注意力里是按头分卡的：V3 的查询头有 128 个，分得开；MLA 把 KV 压成所有头共用的一份，只剩一个头，切不开。</p>
  <p>很多模型把 KV 头定成 8 个，TP 8 路时每张卡正好分一组（Llama 2 70B 就是 8 个 KV 头）。一般地，TP 一旦超过 KV 头数 H，KV 就切不开了：每个头被复制 TP ÷ H 次（vLLM 文档原话：duplicated tp_size / H times）。
    MLA 模型只有一个 KV 头，TP 8 路就是每张卡一整份、8 份一模一样的 KV。4.4 里 V4-Pro 在 TP4 上白存 3 份，毛病就在这儿。</p>
__FIG_KV_DUP__
  <p><b>DCP（decode 上下文并行）</b>让 KV 按 token 轮流存到几张卡上，用的还是原来那几张卡。按 token 轮流而不是一段一段地存，是因为笔记一直在长：新来的字也会均摊到每张卡上。</p>
<figure class="fbox fwide" id="anim-dcp">
<video src="media/topic05-decodecp.mp4" autoplay loop muted playsinline
       aria-label="DCP 动画。四张卡。标题：DCP：decode 时 KV 按 token 轮流存到各张卡。字幕一：只开 TP：笔记切不开，每张卡都存全部 12 个 token 的 KV（四张卡各一块红色大块）。字幕二：DCP：每生成一个 token，它的 KV 存到第 (token 号 mod 4) 张卡上。token 0 到 11 依次落到卡 0、1、2、3 轮转。字幕三：12 个 token，每张卡只存 3 个的 KV：同样的卡，能装 4 倍的笔记。字幕四：算注意力：新 token 的 Q 发给所有卡，各自在自己那份 KV 上算。字幕五：四份部分结果合并成一份：每层多几次通信，换回 4 倍的 KV 空间。最后复位。"></video>
<figcaption>KV 轮流落到四张卡上；每一层多几次通信，把 Q 收齐、把结果合起来。
  <span class="sub">（11 秒无声循环，Manim 渲染。先放只开 TP 时的样子做对照。）</span></figcaption></figure>
  <p>算注意力时，新 token 的 Q 发给所有卡，各自在自己那份 KV 上算，再把几份部分结果合并。注意力本身就是按分数加权平均，所以合并不能各除各的，要「先别除」：</p>
__FIG_SOFTMAX_MERGE__
  <p>每层多付三次通信（vLLM 默认实现：收齐 Q、交换归一化分母、合并输出），换回被 TP 白白复制掉的那几份 KV。
    DCP 的度数最多开到 TP ÷ KV 头数，也就是刚好把复制的那几份收回来。<b>用一种通信，换一份显存</b>，又一次。</p>

  <h3>5.5　prefill 那边也有一种</h3>
  <p>DCP 管的是 decode。prefill 那边要把一个长 prompt 切开、让第一个字早点出来，叫 <b>PCP</b>（prefill 上下文并行）。
    它是在 TP 之外再加一维，所以<b>PCP 加卡，DCP 不加卡</b>；它只管算得快，不管 KV 装不装得下。
    下一节把 prefill 和 decode 拆到两批机器上以后，它们正好可以一边一个（本课的归纳）。</p>

  <h3>5.6　这一刀留下的问题</h3>
  <p>说到这儿，prefill 和 decode 已经各要各的切法了：一个吃算力、要把 prompt 切开；一个吃带宽、要把 KV 摊开。
    <b>硬塞在同一批卡上，谁都配不好。</b>那就干脆别切张量了，把这两种活拆到不同的机器上。这是第五刀。</p>
  <details class="foldfig"><summary><b>考考自己</b>：两张卡各自算出注意力的一部分结果，能不能直接取平均？</summary>
  <p>不能。要「先别除」：每张卡交出分子和分母，分子加分子、分母加分母，最后只除一次。</p></details>
</div></section>

''' + sec("s六", "六", "第五刀：不切张量，切工作") + '''
  <p class="lead">前四刀切的都是张量：batch、权重、专家、序列。第五刀换个思路 ——&nbsp;
    <b>一个请求里本来就有性质完全不同的几段活，把它们拆到不同的机器上</b>，每一边再挑自己的切法。</p>

  <h3>6.1　prefill 和 decode 为什么会打架</h3>
  <p>一个请求分两段。<b>prefill</b> 一口气吞下整个 prompt，几千个 token 一起过矩阵乘，吃的是算力；
    <b>decode</b> 每一步只出一个 token，却要把全部权重和 KV 从显存里读一遍，吃的是带宽。</p>
  <p>你正看着回答一个字一个字往外吐，别人丢进来一本一百页的说明书，你的字就停了。语音助手更明显：说一句停一段，平均速度再快也觉得卡。
    这就是两种活挤在同一批卡上：引擎每一步都要决定先干哪个。长 prompt 一来，它的 prefill 要占好几步，
    这几步里<b>所有正在出字的请求都得等</b>：新请求的首字延迟（TTFT）和老请求的出字间隔（TPOT）一起变差。</p>
  <p>decode 还有个脾气：每一步都要把权重从显存读一遍，读一次只够这一批用一次。一批要多大才不白读？</p>
__FIG_DECODE_AI__
  <p>模型这边也在帮忙凑：MLA 把 KV 压小，一张卡装得下更多请求，一批就大。MegaScale-Infer 论文在 A100 上算过同一笔账：一批至少 156 个请求，Mixtral 每个专家只分到 39 个。</p>
__FIG_PD__
<figure class="fbox fwide" id="anim-pd">
<video src="media/topic05-pd.mp4" autoplay loop muted playsinline
       aria-label="PD 分离动画。标题：PD 分离：prefill 和 decode 拆到两批机器上。三条车道：放在一起、prefill 机器、decode 机器。字幕一：放在一起：大家一步一步 decode，每格出一个字。上面一条车道绿格一格一格长出来，第 5 步落下一个长橙块「新请求的 prefill」，下面标红「这 5 步没人出字」，之后绿格继续。字幕二：拆开：prefill 在自己的机器上跑，decode 那边一步不停。decode 车道 16 个绿格连续长出，prefill 车道同时跑完橙块，一支蓝色箭头「KV 传过去」落到 decode 车道。字幕三：放在一起：decode 被截走 5 步；拆开：一格不断（多用了一批 prefill 机器）。字幕四：代价：多一趟 KV 传输；按我们 v7x 那套的带宽估算约 100 毫秒。最后复位。"></video>
<figcaption>同样 16 步，上面那条被长 prefill 截走了 5 步；下面那条一格没少。
  <span class="sub">（11 秒无声循环，Manim 渲染。格数是示意，不是实测时序。）</span></figcaption></figure>
  <p>分块 prefill（chunked prefill）能缓解：把长 prompt 切成小块，每一步跟 decode 拼着跑。
    但块切小了 prefill 自己变慢（每一块都要把前面的 KV 重读一遍），两种活还在抢同一批卡，而且<b>只能用同一套并行方式</b>。PD 分离干脆把它们拆开：
    prefill 机器只做 prefill，算完把 KV cache 交给 decode 机器，decode 机器只管出字。</p>

  <h3>6.2　代价：一趟 KV 传输</h3>
  <p>我们在 TPU v7x 上搭过一套 1P1D（Qwen3-Coder-480B，一台 v7x-8 做 prefill、一台做 decode），KV 从 prefill 那台的显存走到 decode 那台的显存：</p>
__FIG_KV_TRIP__
  <details class="foldfig"><summary><b>细一点</b>：三段各怎么估的</summary>
  <table>
    <tr><th>段</th><th>路径</th><th>估算</th></tr>
    <tr><td>①</td><td>HBM → 本机内存（PCIe）</td><td>约 10 ms</td></tr>
    <tr><td>②</td><td>本机内存 → 对方内存（数据中心网络，按单颗芯片的 100 Gbps 份额保守估）</td><td>约 80 ms</td></tr>
    <tr><td>③</td><td>对方内存 → HBM（PCIe），接进 decode 的 KV 池</td><td>约 10 ms</td></tr>
    <tr><td></td><td><b>合计</b></td><td><b>约 100 ms</b></td></tr>
  </table>
  <p>中间那段：8K prompt 的 KV 是 2（K、V 各一份）× 62 层 × 8 个 KV 头 × 128（每头维度）× 8,192 × 1 字节（FP8）≈ 1.04 GB，
    100 Gbps 就是每秒 12.5 GB，走一趟约 83 ms。另有一个 CPU 上的转发代理负责把请求分给两边。</p></details>
  <p>这一趟用的是第一节那个一对一收发，而且跨机器、走数据中心网络这根慢线；它敢跨出去，是因为一个请求只传一次。
    从这一刀往后，调度的中心其实是那份笔记：KV 跟着请求搬家。Kimi 的 Mooncake 干脆把整套推理系统叫作「以 KV cache 为中心」。</p>
  <p>反过来说，请求都很短、量也不大的时候，拆开多出来的这趟传输和两套机器就不一定划算，放在一起、用分块 prefill 缓解就够了。</p>

  <h3>6.3　两边各配几台</h3>
  <p>拆开之后多了一个旋钮：prefill 和 decode 的机器配比。思路是让两边差不多同时忙满，DistServe 论文给过一道很干净的算术：</p>
__FIG_PD_RATIO__
  <p>经验上<b>长 prompt 的业务配 2P:1D，长输出的业务配 1P:2D</b>。</p>
  <details class="foldfig"><summary><b>细一点</b>：论文里最多能多服务几倍</summary>
  <p>DistServe 论文把配比和各自的并行方式一起搜，跟不拆开的系统比，同样的延迟要求下最多能多服务 7.4 倍的请求（对 DeepSpeed-MII），或者把延迟要求收紧最多 12.6 倍（对 vLLM）（OPT 系列模型，以 90% 请求达标为准）。上图那个「翻一倍多」是论文里一个单卡算例，7.4 倍是搜遍配比和切法之后的最好结果。</p></details>

  <h3>6.4　两边各挑各的切法</h3>
  <p>这才是拆开的真正收益：<b>两边不再被迫用同一套并行方式</b>。</p>
  <ul>
    <li><b>prefill 机器</b>：可以上 PCP，把一个长 prompt 切到几张卡上一起算，第一个字出得快，它要另外加卡（§5.5）。</li>
    <li><b>decode 机器</b>：可以上 DCP，把 KV 摊到几张卡上装更多请求（§5.4）；还可以把专家铺到更多卡上（Wide-EP）：卡多了，送 token 来的请求也多，每个专家分到的 batch 就更大。</li>
    <li><b>MoE 模型</b>：常见的写法是一边 TEP（attention 用 TP）、一边 DEP（attention 用数据并行），但<b>哪边用哪个没有定式</b>，要看模型和负载（§8.6）。</li>
  </ul>

  <h3>6.5　AFD：attention 和专家分到两组机器</h3>
  <p>decode 这边还能再拆。attention 要读每个请求自己的 KV，跟请求绑定；专家不管 token 来自谁，只要 batch 够大：decode 每步都要把专家权重读一遍，来的 token 越多，这一遍越值（§6.1）。
    <b>AFD</b>（Attention-FFN 分离）把两者放到两组机器上：M 台只算 attention，N 台只放专家。</p>
__FIG_AFD__
  <p>每一层都要把 token 从 attention 那边发给专家（M → N），算完再收回来（N → M）。
    为了不让这一来一回拖慢，把一批请求切成几个小批轮着跑：这一个在算 attention，另一个正好在算专家，还有一个在路上。
    MegaScale-Infer 论文算过，要藏住通信至少得三个小批，通信慢的时候要四个；它报告每 GPU 吞吐最高提升 1.90 倍。</p>
  <details class="foldfig"><summary><b>谁在用</b></summary><p>字节的 MegaScale-Infer、阶跃的 Step-3 都是这个路子；vLLM 在 2026 年 7 月出了实验性插件。</p></details>

  <h3>6.6　多模态：编码器也单独放</h3>
  <p>同样的思路还能往前推一段：多模态模型的视觉编码器单独部署，算好的 embedding 再传给语言模型（Encoder 分离，
    跟 PD 连起来常写作 EPD）。训练里对应的是多模块异构并行：编码器和语言模型各用一套并行方式。</p>

  <h3>6.7　这一刀留下的问题</h3>
  <p>五刀讲完了，每一刀都有自己的通信和适用场景。真到一个集群上，它们要同时存在：
    <b>谁放在同一台机器里、谁跨机器、先定哪一刀的度数？</b></p>
  <details class="foldfig"><summary><b>考考自己</b>：把 prefill 和 decode 拆到两批机器上，还要多搬一趟 KV，为什么反而省卡？</summary>
  <p>两种活不再互相拖，各自都能接满：论文的例子里 2 台 prefill ＋ 1 台 decode 每秒接 10 个请求，混着做要 7 张卡。那趟 KV 只占 prefill 的一小截。</p></details>
</div></section>

''' + sec("s七", "七", "摆到机器上") + '''
  <p class="lead">一套真实配置里常常有三四种并行叠在一起。剩下的问题只有一个：<b>哪一刀放在哪根线上</b>。摆对了，前面每一刀省下的都是真省；摆错了，全被网络吃回去。</p>

  <h3>7.1　线有快有慢，刀有勤有懒</h3>
  <p>一个集群里的线不是一样快的。GPU 这边，GB300 NVL72 把 72 块卡连成一个 NVLink 域，域里每块卡 1.8 TB/s（收发合计，下同）；
    出了这个域只能走网卡，每块卡 200 GB/s，差 9 倍。TPU 这边，一个切片里的芯片走 ICI（够大的切片连成 3D 环面，见 §1.5），
    跨切片走数据中心网络，同口径比慢约 50 倍（换算见台账），比 GPU 那边的 9 倍悬殊得多。</p>
  <p>刀也不是一样勤的。把每一刀一步要通信几次数一遍（本课推导，示意配置），差出三个数量级：</p>
__FIG_FREQ__
  <p>先别往下看：TP、PP、DP 三个，谁该坐快线？把集群想成一栋写字楼就好猜了：</p>
__FIG_OFFICE__
  <p>于是有一条默认的摆法：<b>每一层都要说话的 TP、EP、FSDP、CP 先往最快的那一圈里放</b>；PP 只在段边界说话，DP 一步只说一次，它们去跨慢线。
    但光数次数不够，还要看<b>它能不能跟计算叠起来</b>。TP 叠不起来（每一块末尾那次 AllReduce 不做完，下一块就没法开始），所以必须待在快线里，度数上限就是那一圈的大小。</p>
  <p>EP 和 FSDP 能靠提前发、边算边传藏住一部分，就有人让它们跨出去：V3 的 EP 64 就横跨 8 台机器 —— H800 机内外只差 3 到 4 倍，远没有 TPU 悬殊；Llama 3 把 FSDP 放在了最外层。
    DP 那一次量虽然最大，但一步只有一次，而且反向从最后一层往前算，后面几层的梯度一算好就能先传。
    第六节那趟 KV 也一样，一个请求只传一次，所以敢走数据中心网络这根慢线（指 TPU v7x 那套）。</p>

  <h3>7.2　摆错一次是什么样</h3>
  <p>同样 8 张卡、同样 TP4 × DP2，只改 TP 组放在哪。下面的时间是示意：快线传一次 1 格，慢线 9 格。</p>
<figure class="fbox fwide" id="anim-meshmap">
<video src="media/topic05-meshmap.mp4" autoplay loop muted playsinline
       aria-label="TP 组摆在哪的动画。两台机器各 4 张卡，中间一条虚线是机器之间的慢线。摆法一：机器 0 的 4 张卡是蓝色 TP 组、机器 1 的是橙色 TP 组，每组的环画成同色箭头。第 1 轮 TP，每组沿自己的环传一格，全在机器里，快线 1 格，停住；后面 7 轮一模一样，快进；最后 1 次 DP，两个组里对应的卡交换梯度，要过慢线，9 格，停住，一共 17 格。摆法二：上排 4 张卡是蓝组、下排 4 张是橙组，环横跨两台机器，过慢线的两段画成红箭头，其中一段从机器上方或下方绕回。第 1 轮 TP 要等最慢的那段，9 格，停住；后 7 轮快进；DP 那一对在同一台机器里，1 格，一共 73 格。最后字幕：17 格对 73 格，同样的卡，慢 4.3 倍（示意）。画面复位。"></video>
<figcaption>一轮通信要等最慢的那一段传完，所以摆法二每一轮都按慢线算。红箭头就是过慢线的那几段。
  <span class="sub">（15 秒无声循环，Manim 渲染。一步 8 轮 TP、1 次 DP，快慢 1 : 9 是示意。）</span></figure>
  <p>同一个颜色就是一个 TP 组；摆法二里上排、下排各是一组。一步是 8 轮 TP 加 1 次 DP：摆法一 8 × 1 ＋ 9 ＝ 17 格，摆法二 8 × 9 ＋ 1 ＝ 73 格（DP 那一对卡在同一台机器里，只花 1 格）。</p>
  <p>放到真实集群上，这就是为什么 TP 一般不出一台 8 卡机器。GB300 把 NVLink 域扩到一整柜，
    专家并行才敢往大了开（TP 受头数和矩阵效率限制，在整柜上也很少超过 8 或 16）：<b>快线那一圈画多大，这几刀就能切多深。</b></p>

  <h3>7.3　先选对切法，再调参数</h3>
  <p>摆法和切法选错了，参数调得再细也只是在错的天花板下面打转。我们在 GB300 上跑 DeepSeek-V4-Pro 时撞上过一次。
    图里的 1P1D 是一台 prefill 机器配一台 decode 机器，3P 是三台 prefill；DEP8 是 decode 那边 attention 每张卡各管一批请求、专家铺在 8 张卡上：</p>
__FIG_TOPO__
  <p>TP4 decode 上能调的都调了（去掉 eager 模式、加 prefill 机器、调并发），总吞吐涨了四成五 ——
    可这是拿多一倍的卡换来的，每卡反而掉了，出字间隔一直在 50 ms 上下。
    换成 DEP8 那一步，同样的并发下出字间隔降到原来的四分之一，首字延迟也少了一半多。</p>
  <p>真正属于「换切法」的那笔账，是 KV 不再在 4 张卡上各存一份（§5.4）；decode 从 4 张卡加到 8 张，也把专家摊薄了一半。
    两笔都换成了更大的 batch。这 8 张卡横跨两个计算托盘，但还在同一柜的 NVLink 快线里。<b>先问切法对不对，再动参数。</b></p>
  <details class="foldfig"><summary><b>这一段的细账</b>：每个数、口径，以及原始记录里没归因的地方</summary>
  <ul>
    <li>去掉 eager 模式（即打开 CUDA Graph）只多 2.6%；加 prefill 机器、调并发，总数从 14,563 涨到 21,100（+45%），出字间隔始终在 46.8–53 ms。</li>
    <li>换 DEP8，同样并发 512：出字间隔 46.8 ms → 11.8 ms，首字延迟 55.8 秒 → 22.8 秒。同一套在并发 256 时首字只要 7.8 秒，并发 512 下多出来的大部分是排队。</li>
    <li>出字间隔为什么同时降下来，原始记录里没有拆开归因。</li>
    <li>GB300 一个计算托盘 4 块卡；prefill 到 decode 的 KV 也走柜内 NVLink，不走网卡。并发是排队的请求数，batch 是同时在算的请求数。</li>
    <li>图里后两行都是并发 512，第一行是原始脚本的并发 256。DEP8 把并发拉到 1,536，总量能到 65,132（每卡 2.47 倍），但那时首字要等 95 秒，prefill 又成了瓶颈。</li>
    <li>吞吐是 prompt 和输出 token 加在一起算的。跟最早的原始脚本（每卡 1,820）比，DEP8 每卡约 1.5 倍。</li>
  </ul></details>

  <h3>7.4　五步怎么选</h3>
  <p>把前面几节串起来，给一个模型挑并行方式，像闯五关，每一关一个是非题：</p>
  <ol>
    <li><b>第 1 关　装得下吗？</b>装不下：权重和优化器状态摊不开就上 FSDP ／ ZeRO（第二节）；MoE 的专家太多就上 EP（第四节）；
      推理时 KV 放不下就上 DCP（第五节）。</li>
    <li><b>第 2 关　快线那一圈有多大？</b>TP、EP 的度数别超过它（8 卡一台的机器就是 8，GB300 一柜是 72，TPU 看切片）。</li>
    <li><b>第 3 关　序列长吗？</b>训练长上下文加 CP，推理长 prompt 加 PCP。</li>
    <li><b>第 4 关　还不够，或者非跨很慢的线不可？</b>这时才上 PP：它只传段边界的激活，量小、频率低，能跨慢线，代价是气泡（第三节）。</li>
    <li><b>第 5 关　还剩卡吗？</b>全给 DP：一步只通信一次，最便宜。</li>
  </ol>
  <p>拿两个真实配置对一遍。读这张表，先看两个「不用」：V3 在 8 卡一台的 H800 上不用 TP；混元 3 在 TPU 上不用 EP。表里「DP 4 × FSDP 128」这类写法按乘法读：两种切法各切几份，乘起来就是总共多少份。</p>
  <table>
    <tr><th>步骤</th><th>混元 3（295B MoE，TPU v7，256 芯片）</th><th>DeepSeek-V3（671B MoE，2,048 块 H800）</th></tr>
    <tr><td>① 装得下</td><td>FSDP 128（专家权重也由 FSDP 切）：再窄就爆显存</td><td>ZeRO-1 摊优化器状态；EP 64 摊专家</td></tr>
    <tr><td>② 快线多大</td><td>256 芯片在同一个切片里，全是快线，FSDP 已经能把专家也摊开，<b>不用 EP</b>；小规模试过一次 EP，吞吐反而掉了（见表下注）</td>
      <td>8 卡一台，<b>不用 TP</b>；EP 64 横跨 8 台机器，跨机那段是慢线，是个例外。为了少走慢线，规定每个 token 最多发到 4 台机器</td></tr>
    <tr><td>③ 序列</td><td>4K ／ 8K，不用 CP</td><td>预训练 4K，报告里没有 CP</td></tr>
    <tr><td>④ PP</td><td>一个切片放得下，不用</td><td>PP 16，用 DualPipe 把通信叠进计算</td></tr>
    <tr><td>⑤ DP</td><td>剩下的 4 倍全给 DP：DP 4 × FSDP 128，这是默认配方（v7 一颗芯片算 2 个 device，并行度按 device 数：256 芯片 ＝ 512 个）。
      最好成绩是把 FSDP 加宽到 256、用省下的显存把每个 device 的 batch 从 12 推到 16，比默认配方高 3%；加宽本身只慢不到 1%，收益全来自推大的 batch</td><td>剩下的给 ZeRO-1 的数据并行</td></tr>
  </table>
  <p>同一套五步，两个模型走出来的配置几乎没有重合，<b>因为两台机器的快线长得不一样</b>。
    这也是为什么别人家的并行配置不能照抄：先看自己的线。</p>
  <p><em>表下注：混元 3 那次 EP 实测是在 16 芯片上开 4 路，吞吐掉了 71%（只跑了一次），batch 同时减半，有混杂；换一种配法掉 37%。原始记录归因于环面上 all-to-all 要多跳，但 4 路时这笔账不大，原因还要再查。
    V3 那一列按乘法读：PP 16 × 数据并行 128 ＝ 2,048；EP 64 不另占卡，是在那 128 路数据并行里切专家（本课推导）。</em></p>

  <h3>7.5　怎么评：加卡之后掉了多少</h3>
  <p>配好之后要回答一个问题：卡加上去，每张卡的效率还剩多少。有两种量法：</p>
  <ul>
    <li><b>strong scaling</b>：总活不变，卡加倍，看时间能不能减半。每张卡分到的活越来越少，固定开销的占比越来越大，
      迟早撞墙（Amdahl 定律说的就是这个：不随卡数变小的那部分，决定了加卡的上限）。</li>
    <li><b>weak scaling</b>：每张卡的活不变，卡和总活一起加倍，看每张卡的速度掉不掉。
      训练大模型常常先走这一种：卡多了，global batch 也跟着加。但 global batch 有收敛允许的上限（临界 batch size：batch 再加大，需要的训练步数不再按比例减少），到了上限就只能 strong scaling。</li>
  </ul>
  <p>衡量的数叫扩展效率：加卡后每卡的吞吐 ÷ 加卡前每卡的吞吐。
    我们没有干净的 strong scaling 实测；最接近的是同一个 DP 4 × FSDP 128，每个 device 的 batch 从 12 降到 8，每芯片从 580 掉到 453：每张卡的活一少就吃亏。</p>
  <p>下面是我们自己的两组实测，<b>只能组内比</b>。左边是一次 weak scaling：卡 ×4、global batch ×4。右边比的是同样的卡、FSDP 铺多宽。</p>
__FIG_SCALE__
  <p>左边那组说明 DP 方向几乎是白送的：4 个 64 芯片的组之间，一步只有一次梯度 all-reduce，
    按 v7 的 ICI 粗算，就算几组共用链路、打个折扣，也占不到一步 23.5 秒的百分之一（本课推导）。
    不过这 4 组在同一个切片里，DP 走的是 ICI，这个结论不能直接推到跨切片。</p>
  <p>右边那组才是要小心的：卡没变、batch 没变，只是把 FSDP 从四分之一的芯片铺到全部芯片，每拼一次权重要走的步数多了、每块数据更小，延迟摊不掉，就少了 11%。FSDP 再窄（64 份、32 份）每份权重太大，加上激活就放不下了。
    <b>所以在 batch 还能加的时候，加卡要连 batch 一起加，多出来的卡优先当副本。</b></p>

  <h3>7.6　这一节留下的问题</h3>
  <p>到这里，每一种并行都有了来处：它切什么、多出什么通信、该放在哪根线上。
    下一节把它们摊在一张全景图和几张表上，那些 TEP8、DEP16、DP4 × FSDP128 的写法，就都读得懂了。</p>
  <details class="foldfig"><summary><b>考考自己</b>：TP、PP、DP 三个，谁最该坐最快的那圈线？谁可以去跨慢线？</summary>
  <p>TP：每层都要对账、又藏不住。PP 只在段边界递一次，能跨机器；DP 一步只说一次，最远能跨数据中心。</p></details>
</div></section>

''' + sec("s八", "八", "全景：今天所有的并行方式") + '''


  <p class="lead">名字很多，DP、FSDP、TP、SP、CP、PP、EP，推理那边还有 DCP、PCP、DEP、TEP……
    <b>但归起类来只有四种。</b>前面五刀里，切权重和切专家都算模型并行。先把类认清楚，名字就好记了。</p>

  <h3>8.1　四类刀法</h3>
  <p>判据只有一个：<b>看它切的是什么。</b>先看一张图，再看表：</p>
__FIG_PANO__
  <table>
    <tr><th>刀法</th><th>切的是什么</th><th>每张卡手里有什么</th><th>典型</th></tr>
    <tr><td>''' + D + '''<b>数据并行</b></td><td>batch，或者说请求</td>
      <td>算不同的样本，模型在逻辑上是完整的</td><td>DP、FSDP、Attention DP</td></tr>
    <tr><td>''' + S + '''<b>序列并行</b></td><td>一条样本内部的 token</td>
      <td>同一条样本的一段；训练时是一段激活，推理时是一段 KV cache</td><td>CP、Ulysses、DCP</td></tr>
    <tr><td>''' + M + '''<b>模型并行</b></td><td>权重</td>
      <td>只有模型的一部分：矩阵的几列、几层、几个专家</td><td>TP、PP、EP</td></tr>
    <tr><td>''' + X + '''<b>解耦</b></td><td>工作的阶段或模块</td>
      <td>只干一种活：只做 prefill，或者只算专家</td><td>PD 分离、AFD</td></tr>
  </table>

  <ul>
    <li><b>FSDP 算数据并行，不算模型并行。</b>它确实把权重分片存着，
      但每一层算之前都会把完整权重临时拼回来，每张卡算的还是自己那份样本。
      <em>它省的是显存，不改变「谁算什么」。</em></li>
    <li><b>序列并行单独成一类。</b>严格说它也是在切数据（切一条样本内部），
      但它专门对付长上下文，通信方式也跟 DP 完全不同，放一起反而讲不清。</li>
  </ul>
  <p>下面几张表就按这四类排。「新」表示 2025–26 年才出现或才普及。
    <b>这四张是查阅用的，第一次读可以直接跳到 8.6</b>；表里有不少前面没讲过的名字，用到时再回来翻。</p>

  <h3>8.2　数据并行类</h3>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>DP / DDP</td><td>batch，模型整份复制</td><td>线性扩吞吐</td>
      <td>训练：梯度 all-reduce，每个 step 一次。推理：没有，前面靠 router 分流</td><td>训 · 推</td></tr>
    <tr><td>ZeRO-1 / 2 / 3</td><td>依次多切一样：优化器状态 → 梯度 → 参数</td>
      <td>DP 每张卡存一份完整状态，是纯冗余</td><td>reduce-scatter ＋ all-gather；ZeRO-1／2 通信与 DP 相同（ZeRO-2 切 micro-batch 时每份都要 RS 一次）；ZeRO-3 每层都要把参数 all-gather 回来</td><td>训</td></tr>
    <tr><td>FSDP / FSDP2</td><td>就是 ZeRO-3，PyTorch 原生版本</td>
      <td>通信量是 DP 的 1.5 倍，显存却随卡数线性下降</td><td>前向 all-gather 权重，反向再 all-gather 一次、再 reduce-scatter 梯度</td><td>训</td></tr>
    <tr><td>HSDP</td><td>机内分片，机间复制（例：混元 3 的 DP 4 × FSDP 128，不过它整个在一个切片里）</td><td>把最频繁的 all-gather 圈在高带宽域里</td>
      <td>机内 AG / RS，机间 all-reduce</td><td>训</td></tr>
    <tr><td>ZeRO++</td><td>ZeRO-3 上再加三招</td><td>跨节点通信太贵</td>
      <td>权重量化成 INT8 再 all-gather；节点内多存一份参数；梯度也量化</td><td>训</td></tr>
    <tr><td>跨 DCN 的 DP</td><td>DP 横跨多个 slice 或数据中心</td>
      <td>DP 每个 step 才通信一次，最适合放上慢链路（PP 也能）</td><td>DCN 上的梯度 all-reduce</td><td>训</td></tr>
    <tr><td>DiLoCo ''' + NEW + '''</td><td>每个副本先自己走几百步，再同步一次</td>
      <td>跨数据中心训练：论文里每 500 步才同步一次，通信次数降到几百分之一</td><td>一次外层梯度同步（Streaming 版分片轮流同步，跟计算重叠）</td><td>训</td></tr>
    <tr><td>Attention DP ''' + NEW + '''</td><td><b>只在 attention 层</b>按请求切，专家层另配</td>
      <td>MLA 只有 1 个 KV 头，开 TP8 会把 KV 复制 8 份；改成 DP，每张卡只存自己那批请求的 KV</td>
      <td>后面接 EP 时是 all-to-all；没活的卡也得跑一遍空前向陪着同步</td><td>推</td></tr>
  </table>
  <div class="note warn"><span class="t">⚠️ 同一个「DP」，在 MoE 推理里意思变了</span>
    dense 模型上开 DP，是几份互不相干的副本。MoE 模型在 vLLM / SGLang 里开 DP，
    实际开的是 <b>Attention DP</b>：专家层还是大家一起跑，所以每一步所有 rank 都要同步。
    <br><em>vLLM 还有一个坑：只开 <code>-dp</code> 不开 <code>-ep</code>，专家层走的是 TP 而不是 EP。</em></div>

  <h3>8.3　序列并行类</h3>
  <p>一条样本太长，一张卡放不下它，就沿序列切开。<b>这一类在训练和推理里的动机完全不一样</b>：
    训练切的是激活，推理切的是 KV cache。所以分两张表。</p>

  <p><b>训练侧（以及推理的 prefill）：切激活</b></p>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>Megatron SP</td><td>只切 LayerNorm、Dropout、残差这几段的激活，<b>必须跟 TP 搭配</b>（按「切什么」归到这一列；按用途它是 TP 的搭档，见 §3.3）</td>
      <td>TP 切不到的那部分激活，每张卡都存了一整份</td><td>把 TP 的 all-reduce 拆成 all-gather ＋ reduce-scatter，总量不变</td><td>训 · 推</td></tr>
    <tr><td>CP（Context Parallel）</td><td><b>所有</b>激活沿序列切</td><td>长上下文训练，上下文到几十 K 以上基本都要用</td>
      <td>ring 传 KV，或者 all-to-all，或者 all-gather，可以分层组合</td><td>训</td></tr>
    <tr><td>Ulysses</td><td>序列切和 head 切来回转换</td><td>长序列：序列长度和卡数同比放大时，每张卡的通信量不变</td>
      <td>attention 前后各一次 all-to-all；<b>度数不能超过 head 数</b>（GQA 模型卡在 KV 头数上）</td><td>训</td></tr>
    <tr><td>Ring Attention</td><td>KV 分块沿一个环传递</td><td>度数不受 head 数限制</td><td>环上点对点传递，跟计算重叠</td><td>训 · 推</td></tr>
    <tr><td>Striped / USP / 2D-Attention</td><td>改 token 的分配方式；Ulysses 和 Ring 叠成二维</td>
      <td>ring 在 causal mask 下各卡负载不均；两者的限制互补</td><td>同上两种的组合</td><td>训</td></tr>
    <tr><td>PCP（Prefill CP） ''' + NEW + '''</td><td>prefill 阶段的 query token</td>
      <td>长 prompt 的首字延迟（TTFT）</td><td>all-gather KV，或者 ring</td><td>推</td></tr>
    <tr><td>Chunked PP ''' + NEW + '''</td><td>长 prompt 切成块，依次灌进流水线</td>
      <td>prefill 的启动延迟只跟第一块成正比</td><td>流水线的点对点</td><td>推</td></tr>
  </table>

  <p><b>推理 decode 侧：切 KV cache</b>。这是 2025–26 年最热的一块</p>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>DCP（Decode CP） ''' + NEW + '''</td><td>KV cache 的序列维，按 token 轮流存到各卡</td>
      <td>TP 一旦超过 KV 头数，KV 就开始复制。MLA 只有 1 个头，TP8 就是 8 份一模一样的 KV。
        DCP 把这份冗余变回容量</td>
      <td>每层三次（vLLM 默认实现）：all-gather 收齐 Q、交换 LSE、合并输出</td><td>推</td></tr>
    <tr><td>Helix ''' + NEW + '''</td><td>同一组卡在一层里换两次布局：attention 按 KV 序列切（再叠一维不超过 KV 头数的 TP），FFN 按 TP × EP 切</td>
      <td>百万 token 级的 decode：读 KV 和读权重两件事都要摊开</td><td>attention 后一次 all-to-all 交换部分结果；FFN 段照样有 TP 的 all-reduce 或 EP 的 all-to-all</td><td>推</td></tr>
    <tr><td>MaxText <code>context_autoregressive</code></td><td>decode 时 KV 沿序列切，FFN 按专家切</td>
      <td>同上，TPU 上的做法</td><td>XLA 自动插入</td><td>推</td></tr>
  </table>



  <h3>8.4　模型并行类</h3>
  <p>这一类真正在切权重。按下刀的位置再分三种：矩阵内部、层、专家。</p>

  <p><b>切矩阵内部（张量并行）</b></p>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>TP</td><td>矩阵先按列切、再按行切，两两配对</td><td>单层的权重或计算放不下</td>
      <td>每层前向两次、反向两次 all-reduce，<b>频率极高</b>，所以只能待在最快那一圈（NVLink 域、同一切片的 ICI）；v7 上 TP8 只有 4 颗芯片，已经在线下</td><td>训 · 推</td></tr>
    <tr><td>2D / 2.5D / 3D TP</td><td>把矩阵切成网格</td><td>1D TP 的通信随度数上涨</td>
      <td>沿网格的行、列广播和归约</td><td>训，<b>已基本不用</b></td></tr>
    <tr><td>GTP ''' + NEW + '''</td><td>在 TP 轴上再把权重切一层，用的时候再收回来</td>
      <td>Megatron 文档原话：GTP_remat is an implementation of ZeRO-3。只不过它切在模型并行轴上</td>
      <td>逐个权重异步 all-gather，梯度 reduce-scatter</td><td>训，实验性</td></tr>
  </table>

  <p><b>切层（流水线并行）</b></p>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>PP</td><td>层，分成若干 stage</td><td>模型太深，或者跨机只剩低带宽链路</td>
      <td>stage 边界的点对点，量最小；代价是<b>气泡</b></td><td>训 · 推</td></tr>
    <tr><td>VPP（交错 1F1B）</td><td>每张卡放几段不连续的层</td><td>把气泡缩小</td><td>点对点的次数变多</td><td>训</td></tr>
    <tr><td>Zero Bubble ''' + NEW + '''</td><td>把反向拆成两半：激活的梯度和权重的梯度</td>
      <td>拿权重梯度那一半去填气泡</td><td>点对点</td><td>训</td></tr>
    <tr><td>DualPipe</td><td>两个方向同时灌流水线</td><td>V3 的专家通信太重，要把前反向的计算和通信完全叠起来</td>
      <td>点对点，跟 EP 的 all-to-all 叠在一起</td><td>训</td></tr>
  </table>

  <p><b>切专家（MoE）</b></p>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>EP</td><td>不同专家放到不同的卡上</td><td>专家总参数太大</td>
      <td>token 发出去、算完收回来，两次 all-to-all。<b>发给谁由数据决定</b>，所以负载会不均，这是 EP 最重的病</td><td>训 · 推</td></tr>
    <tr><td>Wide-EP ''' + NEW + '''</td><td>EP 铺到 32、64 甚至更多张卡</td>
      <td>每张卡只放几个专家，decode 时每个专家分到的 batch 就大了</td><td>大规模 all-to-all，要专门的通信库</td><td>推</td></tr>
    <tr><td>ETP</td><td>单个专家内部再做 TP</td><td>单个专家本身太大；细粒度 MoE 一般设成 1</td><td>专家内部 all-reduce</td><td>训</td></tr>
    <tr><td>EDP</td><td>专家侧剩下的那一维数据并行，由别的维度推出来</td><td>专家梯度要在哪些卡之间求和</td><td>专家梯度 all-reduce</td><td>训</td></tr>
    <tr><td>Parallel Folding ''' + NEW + '''</td><td>attention 层和专家层各用一套切法，铺在同一批卡上</td>
      <td>打破「EP 不能超过 DP」的老限制。比如 attention 用 TP4 · CP2 · DP8，专家可以直接 EP64</td>
      <td>两套通信组</td><td>训</td></tr>
  </table>

  <div class="note ok"><span class="t">MoE 层有自己的一套并行维度</span>
    训练时 Megatron 把 attention 层记作 TP × CP × DP × PP，把专家层记作 ETP × EP × EDP × PP，
    <b>两套是分开配的</b>。推理时同样如此，只是换了一套名字，见 §8.6。
    <br>这是 MoE 模型调并行时自由度最大、也最容易配错的地方。</div>

  <h3>8.5　解耦类</h3>
  <p>前三种刀法切的都是张量。这一类切的是<b>工作</b>：把不同性质的活拆到不同的机器上，
    每一边再挑自己的并行方式。</p>
  <table>
    <tr><th>名称</th><th>拆什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>PD 分离</td><td>prefill 和 decode 放到不同机器</td>
      <td>prefill 吃算力，decode 吃带宽，放在一起互相拖累首字延迟和出字速度</td><td>KV cache 跨机传输</td><td>推</td></tr>
    <tr><td>AFD ''' + NEW + '''</td><td>attention 放一组机器，专家放另一组</td>
      <td>专家那边汇集好几组 attention 的 token，把专家的 batch 做大；两边的机器配比也能分开调</td>
      <td>每层一来一回：attention → 专家发一次，专家 → attention 收一次，要切三四个小批把它藏起来</td><td>推</td></tr>
    <tr><td>Encoder 分离</td><td>多模态模型的视觉编码器单独部署</td><td>编码器和语言模型抢资源</td><td>embedding 传输</td><td>推</td></tr>
    <tr><td>多模块异构并行</td><td>编码器和语言模型各用一套并行方式</td><td>多模态训练里两部分的形状差太远</td><td>两套网格之间的桥接通信</td><td>训</td></tr>
  </table>

  <h3>8.6　组合简称：TEP 和 DEP</h3>
  <p>推理这边，MoE 模型的并行配置通常用一个简称加一个数字说完，比如 TEP8、DEP16。
    <b>前一个字母说 attention 怎么切，后面的 EP 说专家怎么切。</b></p>
  <table>
    <tr><th>简称</th><th>attention 层</th><th>专家层</th></tr>
    <tr><td>TEP&lt;N&gt;</td><td>TP，N 张卡一起算</td><td>EP，铺在同样 N 张卡上</td></tr>
    <tr><td>DEP&lt;N&gt;</td><td>Attention DP，每张卡自己管一批请求</td><td>EP，铺在同样 N 张卡上</td></tr>
  </table>
  <p>这是 TensorRT-LLM 的原文定义：<em>TEP&lt;N&gt; shards both attention (TP) and experts (EP) across N ranks.
    DEP&lt;N&gt; keeps attention data-parallel (ADP) while distributing experts across N ranks.</em>
    vLLM 的用法与此一致。</p>
  <p>PD 分离时两边经常各选一种，但<b>哪边用哪种没有定式</b>，要看模型和负载：
    vLLM 部署 Kimi K3 用的是 <b>TEP8 做 prefill、DEP16 做 decode</b>；
    TensorRT-LLM 那篇文章举的例子却是 <b>DEP4 做 prefill、TEP8 做 decode</b>。</p>

  <div class="note danger"><span class="t">⛔ TEP / DEP 不是 ETP / EDP</span>
    名字只差一个字母的顺序，说的却是两件事。<br>
    <b>TEP / DEP</b> 是推理侧的叫法，说的是 <b>attention 怎么切</b>，DEP 里的 D 是 Attention DP。<br>
    <b>ETP / EDP</b> 是 Megatron 训练侧的叫法，说的是<b>专家内部</b>的 TP 度数，以及专家梯度在哪些卡之间求和。<br>
    Megatron 仓库里搜 TEP / DEP，结果是零。所以这两套名字<b>并列记，不要互相换算</b>。</div>

  <p>其它常听到的组合说法：</p>
  <ul>
    <li><b>3D / 4D / 5D 并行</b>：只是说「同时用了几种」，<b>各家算哪几维不一样</b>。
      torchtitan 把 FSDP ＋ TP ＋ CP ＋ PP 叫 4D，Megatron 的 Parallel Folding 论文把 TP / EP / CP / DP / PP 叫 5D。
      引用时要把具体是哪几维说出来。</li>
    <li><b>TRT-LLM 的「Hybrid ETP」</b>：指专家层 TP 和 EP 混用，跟 Megatron 的 ETP 又不是一回事。</li>
  </ul>

  <h3>8.7　常被当成并行、其实不是的</h3>
  <p>判据：<b>它有没有多切出一维。</b>没有的，就不是新的并行方式。</p>
  <table>
    <tr><th>名称</th><th>它实际是什么</th></tr>
    <tr><td>TBO / SBO / DBO</td><td>把通信藏到计算后面：两批数据错开跑，一批在通信时另一批在计算</td></tr>
    <tr><td>Async TP、TP 通信重叠、collective matmul</td><td>还是 TP，只是把通信和矩阵乘切成小块交错着做</td></tr>
    <tr><td>EPLB、冗余专家</td><td>负载均衡：把热门专家多复制几份再重新摆放。修的是 EP 的病，本身不切新维度</td></tr>
    <tr><td>Offload、重计算</td><td>拿时间换显存，或者拿主机内存换显存</td></tr>
    <tr><td>分块 prefill（chunked prefill）</td><td>推理调度：把长 prompt 切成小块跟 decode 拼着跑，缓解互相卡住（§6.1）；别跟训练里的 Chunked PP 混</td></tr>
  </table>

  <h3>8.8　同名不同义</h3>
  <table>
    <tr><th>词</th><th>在不同地方的意思</th></tr>
    <tr><td>SP</td><td>至少三种：Megatron 的 SP（只切 LayerNorm 那几段，跟着 TP）；DeepSpeed 说的 SP（指 Ulysses）；
      以及泛指一切切序列的做法。Megatron 里「所有激活都切」叫 CP</td></tr>
    <tr><td>CP</td><td>训练里指切激活。vLLM 把它拆成两个开关，对卡数的作用相反：PCP 加卡，DCP 不加卡；功能也不同，一个压首字延迟，一个扩 KV 容量</td></tr>
    <tr><td>DP</td><td>dense 模型上是独立副本；MoE 推理里其实是 Attention DP，每一步都要同步</td></tr>
    <tr><td>ETP</td><td>Megatron 指专家内部的 TP；TensorRT-LLM 的 Hybrid ETP 指专家层 TP 和 EP 混用</td></tr>
    <tr><td>hierarchical</td><td>ZeRO++ 的分层分片、Megatron 的分层 CP，是两件不同的事</td></tr>
    <tr><td>卡／芯片／device</td><td>GPU 上一张卡就是一个 device；TPU v7 一颗芯片对软件显示成 2 个 device，并行度按 device 数</td></tr>
    <tr><td>带宽</td><td>有的按单向报、有的按收发合计报：本讲的 1.8 TB/s（NVLink）、1.2 TB/s（ICI）是收发合计，800 Gb/s、100 Gbps（网卡）是单向</td></tr>
    <tr><td>并发与 batch</td><td>并发是在排队的请求数，batch 是同时在算的请求数</td></tr>
  </table>




  <h3>8.9　带走三句话</h3>
  <div class="note ok"><span class="t">以后遇到一个没见过的并行名字，先问三句</span>
    <b>① 它切的是什么？</b>（数据、序列、模型，还是把活拆开）<br>
    <b>② 它多付哪种通信？</b>（第一节那五种之一）<br>
    <b>③ 这种通信该放哪根线？</b>（一步里说得越勤，越要靠里；第七节那栋写字楼）</div>
  <p>拿开场那个 V3 试一次：<b>PP 16、EP 64、ZeRO-1 数据并行</b>（技术报告 sec. 3.2）。
    PP 是切深，只在段边界收发，敢跨机器；EP 是切专家，每层两次 AllToAll，所以规定每个 token 最多去 4 台机器；ZeRO-1 是切数据，一步一次 ReduceScatter ＋ AllGather，最便宜。
    三刀、三种通信、三根线 —— 开场看着像天书的一行配置，现在读得懂了。</p>
  <p class="landing">⭐ 回到开头那个问题：从哪儿下刀。五刀切的东西各不一样，付的账却是同一种：<b>拿一种通信，换一份显存或一份算力</b>。
    所以选并行策略，说到底是在选你愿意付哪一种通信、付多频繁、放在哪根线上。</p>
  <p>最后一句也是第三节和第六节那把尺子：<b>硬件线是算力 ÷ 带宽，负载线由你的切法决定</b>。这一讲的每一刀，都是在让某条负载线爬到硬件线上面去。</p>
  <details class="foldfig"><summary><b>会后两道小题</b>（答案都在前面，自己先想；第一到第七节末尾还各有一道「考考自己」）</summary>
  <p>① 一个 MoE 模型要在一柜 GB300（72 块卡、同一个 NVLink 快线圈）上做 decode，attention 那一半和专家那一半，你会各怎么切？先答三问，再对照 7.3。<br>
    ② TEP8 和 Megatron 的 ETP8，名字只差一个字母顺序，意思有什么不同？先自己查，再看 8.6 那个红框。</p></details>
</div></section>

''' + sec("s九", "九", "出处台账") + '''
  <p class="lead">按「结论 ← 材料」排。第八节 2026-09-23 核对，第一节 2026-09-24 核对，其余各节 2026-09-25 经十轮评审与两遍试讲复核。</p>
  <table>
    <tr><th>结论</th><th>材料</th></tr>
    <tr><td>各集合通信每卡发出的量（第一节的表）</td><td>NVIDIA/nccl-tests：doc/PERFORMANCE.md 的 bus bandwidth 修正系数：AllReduce 2(n−1)/n，ReduceScatter / AllGather / AlltoAll (n−1)/n，Broadcast / Reduce 1</td></tr>
    <tr><td>环形 ReduceScatter 的逐步推演、班长模式</td><td>wanghonglei《分布式深度学习集体通信原语——从零到精通》（2026-06-27）第 1–2 章；图里每一步由脚本按调度现算并断言。块号比原文挪了一位，让卡 k 最后拿第 k 块</td></tr>
    <tr><td>开场题：常规混合精度每参数 16 字节；V3 实际把 m、v 存成 bf16，主权重与累积梯度留 fp32</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 sec. 3.3.3（原文：用 BF16 代替 FP32 追踪 AdamW 一、二阶矩，「未观察到性能退化」；主权重与用于 batch 累积的梯度仍保留 FP32）；常规做法里优化器状态与参数同精度（PyTorch 默认）；Megatron Core 需显式开启 --use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16（Megatron Core MoE 文档）；Megatron 的「精度感知优化器」参数组里并列 --main-grads-dtype、--main-params-dtype、--exp-avg-dtype、--exp-avg-sq-dtype（arguments.py），本课口径仍把梯度单列；V3 每参数约 14 字节为本课推导（计算用权重的精度报告未写，按 BF16 计）；bf16 训练时 Megatron 默认把梯度累加与 all-reduce 放 fp32（megatron/training/arguments.py：「bfloat16 requires gradient accumulation and all-reduce to be done in fp32」），此时梯度 4 字节、每参数 18 字节</td></tr>
    <tr><td>V3 的 FP8 方案整体验证：损失相对误差低于 0.25%</td><td>同上 sec. 3.3 与附录 B.1：约 16B、230B 两个规模各训约一万亿 token，整套 FP8 方案（含 bf16 优化器状态）对 BF16 基线；报告未单独消融「动量用 bf16」这一项</td></tr>
    <tr><td>动量能不能用 bf16，要看 β₂</td><td>V3 的 AdamW β₁＝0.9、β₂＝0.95（技术报告 sec. 4.2）；PyTorch AdamW 默认 β₂＝0.999。⚠️ 本课推导：bf16 有效位 8 位，相邻两数的相对间隔 2⁻⁷～2⁻⁸，四舍五入丢掉小于半个间隔（最小约 0.2%）的改动；β₂＝0.999 时新值每步只占 0.1%。独立证据：Dettmers 等 arXiv 2110.02861，分块量化的 8 比特优化器状态能追平 32 比特</td></tr>
    <tr><td>V3 全部训练 278.8 万 H800 卡时；一块卡约 318 年</td><td>DeepSeek-V3 技术报告摘要（含预训练、长上下文扩展与后训练）；318 年 ＝ 2.788M ÷ 8,760 小时，本课推导</td></tr>
    <tr><td>ZeRO 的本地更新不需要通信；梯度裁剪要一次标量 AllReduce</td><td>Adam 逐元素更新（Kingma 与 Ba，arXiv 1412.6980 算法 1）；梯度裁剪按全体梯度的范数（Pascanu 等 arXiv 1211.5063），分片时各卡算局部平方和再 AllReduce（本课归纳）</td></tr>
    <tr><td>ZeRO 各级的显存与通信（第二节）</td><td>Rajbhandari 等，ZeRO，arXiv 1910.02054 sec. 5、sec. 7：Pos、Pos+g 通信量与数据并行相同（2Ψ），Pos+g+p 最多 1.5 倍；显存 16Ψ → 16Ψ/Nd</td></tr>
    <tr><td>TP 的切法与通信次数；SP 不增通信（第三节）</td><td>Megatron-LM arXiv 1909.08053 sec. 3（前向 2 次、反向 2 次 all-reduce）；arXiv 2205.05198 sec. 4.2.2（AG＋RS 替代 all-reduce，无额外通信）；查询头数须被 TP 整除、KV 组数与 TP 互为倍数或约数：megatron/core/transformer/transformer_config.py 的校验</td></tr>
    <tr><td>PP 气泡 (p−1)/m；交错式除以 v</td><td>Narayanan 等 arXiv 2104.04473 sec. 2.2.1–2.2.2；Zero Bubble arXiv 2401.10241；DualPipe README</td></tr>
    <tr><td>V3 训练并行配置；参数分布</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 sec. 3.2（16 路 PP、64 路 EP、ZeRO-1，不用 TP）；config.json（61 层、前 3 层 dense、256 专家、moe_intermediate_size 2048、hidden 7168）</td></tr>
    <tr><td>每字节换多少计算、v7 硬件线约 3,845</td><td>⚠️ 本课推导（稠密近似、完全重叠）；v7 2,307 TFLOP/s bf16；官方给每芯片 ICI 1,200 GB/s，另给 200 GB/s 一档；把它理解成每条链路收发合计，「6 条链路 × 200、发出方向 600」才对得上，这是推导（按 scaling book 单链路单向 9e10 算约 540，硬件线约 4,270，所以取 3,800–4,300 区间）；按 device 口径同样约 3,845（一颗芯片的两个 device 共用链路，算力和带宽一起减半）（wiki ici-dcn、Inferact 博客规格表）。2026-09-25 更正：旧版误用 1,200 得出 1,922；TP 那条线的 4.5 系数按标准注意力加 4 倍宽 MLP 推出（V3 实际是 MLA 加 MoE，只作示意）</td></tr>
    <tr><td>V3 的 EP 细节：最多 4 节点、FP8 派发 BF16 合并、无辅助损失的负载均衡</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 sec. 2.1.2、sec. 3.2.2、sec. 3.3.3；每 token 跨节点派发 ≈ 28.7 KB 为本课推导</td></tr>
    <tr><td>V3 负载均衡的超参与「不丢 token」；NVLink ／ IB 3.2 倍、同通信量最多 13 个专家</td><td>同一份报告 sec. 4.2（偏置更新步长 γ ＝ 0.001，最后 500B token 置 0；序列级平衡损失 α ＝ 0.0001；M ＝ 4、64 卡 8 台）、sec. 2.1.2（No Token-Dropping）、sec. 3.2.2（160 ／ 50 GB/s，4 × 3.2 ≈ 13）</td></tr>
    <tr><td>Parallel Folding 的例子</td><td>Megatron-Core megatron/core/transformer/moe/README.md；arXiv 2504.14960</td></tr>
    <tr><td>GB300 上 TP4 → DEP8：同并发 512 总量 2.61 倍、每卡 2.09 倍（DEP8 并发 1,536 时每卡 2.47 倍、TTFT 95 s）；调参 +45%</td><td>本课程作者实测：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/README.md 与 VLLM-V4PRO-RUNBOOK.md（TP4 decode 14,563 → 调参后 21,100，16 GPU、每卡 1,319；DEP8 65,132，20 GPU、每卡 3,257 tok/s）</td></tr>
    <tr><td>激活随序列长度增长</td><td>Korthikanti 等 arXiv 2205.05198 式 (1)：每层 sbh(34 ＋ 5as/h)</td></tr>
    <tr><td>Ring Attention；Ulysses 通信量恒定、并行度不超过头数</td><td>arXiv 2310.01889；arXiv 2309.14509 sec. 3.2（4Nh/P，N 与 P 同比放大时不变）；头数上限见 USP arXiv 2405.07719 sec. 3</td></tr>
    <tr><td>CP 的之字形切法</td><td>Megatron-LM megatron/core/utils.py（2×cp 块，rank r 拿第 r 与 2·cp−r−1 块）；docs/user-guide/features/context_parallel.md</td></tr>
    <tr><td>KV 被 TP 复制 tp/H 次；DCP 复用 TP rank</td><td>vLLM context parallel 部署文档；vllm/config/parallel.py docstring。V3 每 token KV 70,272 字节按 config.json 现算。config 里 num_key_value_heads=128 是 MLA 解压后的头数，推理缓存的是压缩后那一份，vLLM 当作 1 个 KV 头</td></tr>
    <tr><td>PD 分离的动机与收益（第六节）</td><td>DistServe arXiv 2401.09670（prefill 偏算力、decode 受带宽约束；7.4 倍请求是对 DeepSpeed-MII，12.6 倍更紧的 SLO 是对 vLLM）</td></tr>
    <tr><td>v7x 上 1P1D 的 KV 三段约 100 ms（带宽估算）；2P:1D ／ 1P:2D</td><td>本课程作者的部署记录（KV 用时为按带宽估算，非计时）：wiki qwen3-coder-480b-pd-disagg-tpuv7x-20260425。8K KV ≈ 1.04 GB、过 100 Gbps 约 83 ms 为本课推导（Qwen3-Coder config：62 层、8 个 KV 头、head_dim 128）</td></tr>
    <tr><td>每一刀每步的通信次数（第七节）</td><td>⚠️ 本课推导（60 层、8 个 micro-batch 的示意配置）：TP 每层 4 次（arXiv 1909.08053 sec. 3），FSDP 每层 3 次（arXiv 1910.02054 sec. 7），EP ／ PP ／ DP 按调度数出</td></tr>
    <tr><td>GB300 NVLink 1.8 TB/s ／ 每 GPU 800 Gb/s 网卡；9 倍</td><td>wiki sources/nvidia-gpu-comparison-20260311、analyses/gb300-a4x-max-network-congestion-control（A4X Max 每节点 4 GPU、4 × CX-8 800 Gb/s）；9 倍为本课按双向口径换算</td></tr>
    <tr><td>混元 3 的 scaling 与五步对照</td><td>本课程作者实测：gpu-tpu-pedia tpu/Hunyuan3-295B-Pretraining/TUNING-v7 sec. 3.7（五种分法 404 ／ 450 ／ 453 ／ OOM ／ OOM）、sec. 4.1（64 与 256 芯片同为 580）、sec. 3.6（DP2 × FSDP256、pdbs 16 得 599）、EP 4 路在 16 芯片上 −71%（单次）。sec. 4.1 一步 23.54 s；组间 all-reduce 占不到百分之一为本课推导（原文 12 ms 的算式前后不一致）；换配法 EP 掉 37% 与 FSDP 加宽只慢不到 1%（450 对 453）见 TUNING-v7 sec. 3.7</td></tr>
    <tr><td>DCP 每层三次通信</td><td>vllm/config/parallel.py 中 dcp_comm_backend 的 docstring（默认 ag_rs 每层 3 次 NCCL 调用，a2a 后端 2 次）</td></tr>
    <tr><td>ZeRO-2 切 micro-batch 不再白送；配 PP 选 ZeRO-1</td><td>DeepSpeed 文档：流水线并行不兼容 ZeRO-2／3；V3 技术报告 sec. 3.2（ZeRO-1）；Megatron distributed optimizer</td></tr>
    <tr><td>TP 每字节换多少计算 ≈ 4.5 × 隐藏维 ÷ TP、V3 TP8 ≈ 4,032</td><td>⚠️ 本课推导：每层 4 次 AllReduce 各发约 4Th 字节、算 72h²T／n（标准注意力＋4 倍宽 MLP），对 V3 只作示意</td></tr>
    <tr><td>decode 时 EP 按专家逐个直发</td><td>DeepEP 低延迟模式（deepseek-ai/DeepEP README）；V3 技术报告 sec. 3.4</td></tr>
    <tr><td>V3 的集群与每 token 最多 4 节点</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 sec. 3.1（2,048 块 H800、节点内 NVLink、节点间 IB）、sec. 2.1.2、sec. 3.2</td></tr>
    <tr><td>strong ／ weak scaling</td><td>Amdahl 1967；Gustafson 1988（Reevaluating Amdahl's Law）；临界 batch size：arXiv 1812.06162</td></tr>
    <tr><td>NVSwitch 在交换机里做加法（NVLS）</td><td>NCCL NVLS 算法（NVIDIA NCCL 文档）；「少将近一半」为按每卡发出约 S 对 2(n−1)/n·S 的本课推导</td></tr>
    <tr><td>TPU 切片何时首尾成环、小切片带宽约减半</td><td>How to Scale Your Model（scaling book）TPU 章节：只有整 cube（4 的倍数）才有环回；Google Cloud TPU7x 拓扑文档</td></tr>
    <tr><td>Llama 3 把 FSDP 放在最外层</td><td>Llama 3 技术报告 arXiv 2407.21783 sec. 3.3.2（并行维度顺序 [TP, CP, PP, DP]）</td></tr>
    <tr><td>V3 每 token 激活 370 亿参数、预训练 4K</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 摘要与 sec. 4.1</td></tr>
    <tr><td>torchtitan 的「4D」</td><td>pytorch/torchtitan README（FSDP2 ＋ TP ＋ PP ＋ CP）</td></tr>
    <tr><td>「最大的一块卡显存也就两三百 GB」</td><td>NVIDIA B300 每卡 288 GB HBM3e；Google TPU7x 每芯片 192 GiB HBM（源码写十进制 206 × 10⁹ 字节 ≈ 191.85 GiB）；9.76 TiB ≈ 10,700 GB（十进制）</td></tr>
    <tr><td>DCN 每芯片 100 Gbps；跨切片同口径慢约 50 倍</td><td>Google Cloud TPU7x 文档（第五轮 TPU 专家评审核对）；100 Gbps 按单向计、×2 得双向约 25 GB/s，对 ICI 双向 1,200 GB/s 约 48 倍，为本课推导（只按单向比则约 96 倍）</td></tr>
    <tr><td>TEP / DEP 的定义</td><td>TensorRT-LLM tech blog 26（DeepSeek V4 on Blackwell）原文；vLLM Kimi K3 blog（2026-07-27）</td></tr>
    <tr><td>Megatron 里没有 TEP / DEP；ETP / EDP / Parallel Folding</td>
      <td>NVIDIA/Megatron-LM main：megatron/core/transformer/moe/README.md；论文 arXiv 2504.14960</td></tr>
    <tr><td>PCP 加卡、DCP 不加卡、DCP 复用 TP rank</td><td>vllm-project/vllm main：vllm/config/parallel.py 的 docstring；vLLM context parallel 部署文档</td></tr>
    <tr><td>Attention DP 的动机与开关</td><td>SGLang DP / DPA 指南；TensorRT-LLM docs/source/features/parallel-strategy.md；vLLM data parallel 部署文档</td></tr>
    <tr><td>Helix</td><td>arXiv 2507.07120；TensorRT-LLM tech blog 22</td></tr>
    <tr><td>GTP 原话</td><td>Megatron-LM docs/api-guide/core/generalized_tensor_parallel.md</td></tr>
    <tr><td>Zero Bubble / DualPipe</td><td>arXiv 2401.10241；github.com/deepseek-ai/DualPipe</td></tr>
    <tr><td>DiLoCo</td><td>arXiv 2311.08105；Streaming DiLoCo arXiv 2501.18512；MaxText 的 diloco mesh 轴</td></tr>
    <tr><td>AFD</td><td>MegaScale-Infer arXiv 2504.02263；Step-3 arXiv 2507.19427；vLLM AFD Plugin blog（2026-07-23）</td></tr>
    <tr><td>Chunked PP</td><td>LMSYS blog（2026-01-15）；Mooncake arXiv 2407.00079</td></tr>
    <tr><td>MaxText 的并行轴</td><td>AI-Hypercomputer/maxtext：src/maxtext/configs/base.yml、docs/guides/optimization/sharding.md</td></tr>
  </table>
</div></section>


'''

FOOT = '''
<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    上一讲 <a href="topic-04.html">专题四 · 反向与优化器</a>　·
    想自己讲一遍？看 <a href="topic-05-lecture.html">本讲讲义（授课稿）</a><br>
    本页由 <code>Courses/tools/topic05-build.py</code> 生成 ——&nbsp;<b>正文写在那个脚本里</b>。</p>
</div>

</body></html>'''

FIGS = {
    "__FIG_INTENSITY__": ("fig-intensity", "fig5-intensity.svg", "topic05-fig-tp.py",
        '<b>FSDP 那条斜线看 batch，TP 那几条平线看隐藏维。</b><br>'
        '<em>低于红色虚线，就是算得没有搬得快。</em>'),
    "__FIG_TP_MLP__": ("fig-tp-mlp", "fig5-tp-mlp.svg", "topic05-fig-tp.py",
        '<b>列切接行切，中间结果不出卡。</b><br>'
        '<em>这是 Megatron-LM 的切法，attention 按头切也是同一个思路。</em>'),
    "__FIG_TP_ORDER__": ("fig-tp-order", "fig5-tp-order.svg", "topic05-fig-tp.py",
        '<b>一个元素、两张卡、两种顺序。</b><br>'
        '<em>反过来切，GeLU 之前就得先通信一次。</em>'),
    "__FIG_WIDE_DEEP__": ("fig-wide-deep", "fig5-wide-deep.svg", "topic05-fig-tp.py",
        '<b>红点就是通信：左边每层一个，右边只有三个。</b><br>'
        '<em>各家对「横切、纵切」的叫法正好相反，本课只说切宽、切深。</em>'),
    "__FIG_PP_BUBBLE__": ("fig-pp-bubble", "fig5-pp-bubble.svg", "topic05-fig-tp.py",
        '<b>浅灰色就是气泡：这一段在干等。</b><br>'
        '<em>4 段 8 个 micro-batch，气泡是理想计算时间的 3/8。</em>'),
    "__FIG_MOE_PARAMS__": ("fig-moe-params", "fig5-moe-params.svg", "topic05-fig-ep.py",
        '<b>那一小截灰色，是注意力、共享专家、稠密 MLP 和词表加起来的全部。</b><br>'
        '<em>路由专家的份额由 config.json 的尺寸现算。</em>'),
    "__FIG_EP_ROUTE__": ("fig-ep-route", "fig5-ep-route.svg", "topic05-fig-ep.py",
        '<b>同一个 token，左边跨机 8 份，右边 4 份、每份还小一半。</b><br>'
        '<em>机器里那几根绿色短弧几乎是白送的。</em>'),
    "__FIG_EP_BIAS__": ("fig-ep-bias", "fig5-ep-bias.svg", "topic05-fig-ep.py",
        '<b>红虚线是平均：柱子越齐，所有人等得越少。</b><br>'
        '<em>示意模拟，8 选 2、调得比 V3 快，为了几十步就看得出来。</em>'),
    "__FIG_FOLD__": ("fig-fold", "fig5-fold.svg", "topic05-fig-ep.py",
        '<b>左右两边是同样的 8 张卡。</b><br>'
        '<em>进 attention 时按 TP 组干活，进专家层时每张卡管 32 个专家。</em>'),
    "__FIG_CP_ZIGZAG__": ("fig-cp-zigzag", "fig5-cp-zigzag.svg", "topic05-fig-seq.py",
        '<b>同样 8 块，换一种分法，最忙和最闲从差好几倍变成一样忙。</b><br>'
        '<em>每格数由脚本按因果掩码现算。</em>'),
    "__FIG_SOFTMAX_MERGE__": ("fig-softmax-merge", "fig5-softmax-merge.svg", "topic05-fig-seq.py",
        '<b>两个平均数不能再平均，得带上各自的「人数」。</b><br>'
        '<em>分子分母各自交上来，最后只除一次。</em>'),
    "__FIG_KV_DUP__": ("fig-kv-dup", "fig5-kv-dup.svg", "topic05-fig-seq.py",
        '<b>红色那 7 份，存的是一模一样的东西。</b><br>'
        '<em>KV 尺寸取自 V3 的 config.json。</em>'),
    "__FIG_PD__": ("fig-pd", "fig5-pd.svg", "topic05-fig-pd.py",
        '<b>上面那条被截走的几格，就是拆开要换回来的东西。</b><br>'
        '<em>下面那条多用了一批 prefill 机器，比的不是谁出字多，是出字断不断。时间线是示意；100 ms 是按我们 v7x 那套的带宽估算的。</em>'),
    "__FIG_DECODE_AI__": ("fig-decode-ai", "fig5-decode-ai.svg", "topic05-fig-pd.py",
        '<b>蓝线看一批有多少请求，橙线还要再除以 32。</b><br>'
        '<em>跟第三节那张「搬一个字节换多少计算」是同一把尺子，只是这回搬的是显存。</em>'),
    "__FIG_KV_TRIP__": ("fig-kv-trip", "fig5-kv-trip.svg", "topic05-fig-pd.py",
        '<b>下面那条细细的三色条，就是拆开要付的全部代价。</b><br>'
        '<em>放大 10 倍那行才看得清三段。</em>'),
    "__FIG_PD_RATIO__": ("fig-pd-ratio", "fig5-pd-ratio.svg", "topic05-fig-pd.py",
        '<b>多用一种机器，反而省卡。</b><br>'
        '<em>每张卡接的请求翻了一倍多，是因为谁也不再拖谁。</em>'),
    "__FIG_AFD__": ("fig-afd", "fig5-afd.svg", "topic05-fig-pd.py",
        '<b>拆开的不是张量，是一层里的两种活。</b><br>'
        '<em>机器数和格子都是示意。</em>'),
    "__FIG_OFFICE__": ("fig-office", "fig5-office.svg", "topic05-fig-map.py",
        '<b>合写合同得同桌，递稿子隔层楼也行，对账可以在别的城市。</b><br>'
        '<em>圈的大小不代表带宽比例，只表示由里到外越来越慢。</em>'),
    "__FIG_FREQ__": ("fig-freq", "fig5-freq.svg", "topic05-fig-map.py",
        '<b>次数差三个数量级，线速差一个数量级，两边一对就是摆法。</b><br>'
        '<em>次数是示意配置下的推导；带宽取 GB300 每块 GPU 的双向值。</em>'),
    "__FIG_TOPO__": ("fig-topo", "fig5-topo.svg", "topic05-fig-map.py",
        '<b>比吞吐要按每张卡比：三次实测用的卡数是 8、16、20。</b><br>'
        '<em>本课程作者实测，4K 进 1K 出。</em>'),
    "__FIG_SCALE__": ("fig-scale", "fig5-scale.svg", "topic05-fig-map.py",
        '<b>同样多的卡，当副本用和摊薄了用，差 11%。</b><br>'
        '<em>本课程作者实测；左右两组每卡 batch 不同，只在组内比。</em>'),
    "__FIG_PANO__": ("fig-panorama", "fig5-panorama.svg", "topic05-fig-map.py",
        '<b>名字再多，先问它在哪一列、圆点是实是空。</b><br>'
        '<em>圆点是默认摆法，例外见第七节。</em>'),
    "__FIG_ZERO_MEM__": ("fig-zero-mem", "fig5-zero-mem.svg", "topic05-fig-zero.py",
        '<b>16 字节里，优化器状态独占 12 个 —— 所以先削它。</b><br>'
        '<em>ZeRO-3 那条细得要放大才看得清：16 个字节切成 128 份，每份只剩 0.125。</em>'),
    "__FIG_ROADMAP__": ("fig-roadmap", "fig5-roadmap.svg", "topic05-fig-roadmap.py",
        '<b>红字是剧情：每一刀留下的问题，就是下一节的开头。</b><br>'
        '<em>每一节最后一小节「这一刀留下的问题」，就是这里的一格红字。</em>'),
    "__FIG_WHY_CUT__": ("fig-why-cut", "fig5-why-cut.svg", "topic05-fig-why.py",
        '<b>左边是放不下，右边是算不完。</b><br>'
        '<em>一格一块卡；右边两根条用的是同一把尺子。</em>'),
    "__FIG_ZERO_BUSY__": ("fig-zero-busy", "fig5-zero-busy.svg", "topic05-fig-zero.py",
        '<b>虚线那一长条，就是 ZeRO-1 白捡的地方。</b><br>'
        '<em>时间轴是示意，只画每块东西什么时候被读写。</em>'),
    "__FIG_ZERO_STEP__": ("fig-zero-step", "fig5-zero-step.svg", "topic05-fig-zero.py",
        '<b>① ReduceScatter、③ AllGather 拼起来就是原来那次 AllReduce。</b><br>'
        '<em>多出来的只是中间那一格：每张卡只更新自己那一段。</em>'),
    "__FIG_BF16_BETA__": ("fig-bf16-beta", "fig5-bf16-beta.svg", "topic05-fig-zero.py",
        '<b>绿的那一步挪过了几格，红的那一步没挪过半格。</b><br>'
        '<em>舍入按 bf16 真算；假设新来的 g² 是当前 v 的 2 倍。</em>'),
    "__FIG_SIKU__": ("fig-siku", "fig5-siku.svg", "topic05-fig-zero.py",
        '<b>单独一段读不了：这是 FSDP 跟 EP、PP 最大的区别。</b><br>'
        '<em>阁数按 V3 的真实配置：数据并行 128、EP 64、PP 16。</em>'),
    "__FIG_FSDP_TP__": ("fig-fsdp-tp", "fig5-fsdp-tp.svg", "topic05-fig-tp.py",
        '<b>FSDP 在列里拼，TP 在行里加；拼回来的只是自己那一条。</b><br>'
        '<em>切法对照 Scaling Book 训练篇；961 是本课推导。</em>'),
    "__FIG_FSDP_STEP__": ("fig-fsdp-step", "fig5-fsdp-step.svg", "topic05-fig-zero.py",
        '<b>数据并行一层做两次通信，FSDP 做三次。</b><br>'
        '<em>多出来的那次 AllGather，是反向时把前向扔掉的权重再拼回来。</em>'),
    "__FIG_COLL_1N__": ("fig-coll-1n", "fig5-coll-1n.svg", "topic05-fig-coll.py",
        '<b>四个有班长的动作：广播、分发、收集、归约。</b><br>'
        '<em>收集和归约只差一个动作：数据到了之后是拼起来，还是加起来。</em>'),
    "__FIG_COLL_NN__": ("fig-coll-nn", "fig5-coll-nn.svg", "topic05-fig-coll.py",
        '<b>名字带 All 的，结果人人一份。</b><br>'
        '<em>AllToAll 是另一回事：不加也不拼，只换位置。</em>'),
    "__FIG_AR_SPLIT__": ("fig-ar-split", "fig5-ar-split.svg", "topic05-fig-coll.py",
        '<b>先加后拼，跟一步到位的结果完全一样。</b><br>'
        '<em>拆开之后，两半可以放在不同的时间点去做。</em>'),
    "__FIG_RING__": ("fig-ring", "fig5-ring.svg", "topic05-fig-coll.py",
        '<b>盯着条纹看：每一步，每张卡都有一块多加进一个人。</b><br>'
        '<em>一行一张卡；「右边」就是下一号卡，卡 3 的右边是卡 0。条纹看不清加了谁，就看下面那支动画，一步一停。</em>'),
    "__FIG_AR_TWO_WAYS__": ("fig-ar-two-ways", "fig5-ar-two-ways.svg", "topic05-fig-coll.py",
        '<b>结果一样，差的是谁在干活。</b><br>'
        '<em>右边那张小表：班长那列跟着卡数涨，环那列永远不到 2。</em>'),
    "__FIG_A2A__": ("fig-a2a", "fig5-a2a.svg", "topic05-fig-coll.py",
        '<b>左边一行是「我要发给谁」，右边一行是「谁发给了我」。</b><br>'
        '<em>每一格多大，在 MoE 里要等路由算完才知道。</em>'),
}

_html = head + HERO + BODY + FOOT
_html = P.place_figs(_html, FIGS)
_html = _html.replace("__QUIZ__", _QUIZ_HTML).replace("__Q3_ANSWER__", QZ.answers_html())
_leak = sorted(set(re.findall(r"__[A-Z][A-Z_0-9]*__", _html)))
assert not _leak, "占位符没落地：%s" % "、".join(_leak)
for _tag in ("h2", "h3", "section", "div"):
    _o = len(re.findall(r"<%s[ >]" % _tag, _html))
    _c = _html.count("</%s>" % _tag)
    assert _o == _c, "<%s> 开 %d 个、闭 %d 个" % (_tag, _o, _c)
# ⭐ 2026-09-25 对照构建手册补的锁：正文里手打的关键数，必须在对应的图里也出现（图是脚本现算的）。
#   图脚本常量一改，正文不跟着改就当场失败 —— 否则正文会静默过期。比较时去掉千分位逗号。
_NUM_LOCK = [("3,845", "fig5-intensity.svg"), ("4,032", "fig5-intensity.svg"),
             ("9.76 TiB", "fig5-zero-mem.svg"), ("1.29 TiB", "fig5-zero-mem.svg"), ("78.11 GiB", "fig5-zero-mem.svg"),
             ("8.58 GiB", "fig5-kv-dup.svg"), ("97%", "fig5-moe-params.svg"), ("6,539", "fig5-moe-params.svg"),
             ("1,319", "fig5-topo.svg"), ("1,820", "fig5-topo.svg"), ("2.09", "fig5-topo.svg"),
             ("961", "fig5-fsdp-tp.svg"), ("5,493", "fig5-intensity.svg"), ("580", "fig5-scale.svg"), ("453", "fig5-scale.svg"), ("404", "fig5-scale.svg")]
for _n, _svg in _NUM_LOCK:
    assert _n in BODY, "正文里已经没有 %s 了 —— 从 _NUM_LOCK 里删掉这一条" % _n
    _g = io.open(os.path.join(HERE, _svg), encoding="utf-8").read().replace(",", "")
    assert _n.replace(",", "") in _g, "正文写 %s，而 %s 里没有这个数 —— 图脚本改了，正文没跟上" % (_n, _svg)
# ⭐⭐ 2026-09-25 现场：「那个动图你每跳一步就停一下，我们琢磨一下再跳第二步。」
#   动画脚本在每一步落地后打点，写进 manim/steps/<Scene>.json；这里把时刻挂到 <video data-pauses>，
#   页面脚本在这些时刻自动暂停，出一个「下一步」按钮。勾「连续播放」就不停。
#   图注里的「N 秒」也从同一份 json 取，免得跟实测时长对不上（check-loop 会查）。
STEP_VIDEOS = {"topic05-ring.mp4": "Ring", "topic05-allreduce.mp4": "AllReduce",
               "topic05-reducescatter.mp4": "ReduceScatter", "topic05-allgather.mp4": "AllGather",
               "topic05-a2a.mp4": "AllToAll", "topic05-broadcast.mp4": "Broadcast",
               "topic05-scatter.mp4": "Scatter", "topic05-gather.mp4": "Gather", "topic05-reduce.mp4": "Reduce",
               "topic05-meshmap.mp4": "MeshMap"}


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
        fig2, n = re.subn(r"（\d+ 秒无声循环，(.*?)Manim 渲染", lambda m: "（%d 秒，Manim 渲染；点「开始」一步一步看，勾「连续播放」就无缝循环%s" % (round(st["duration"]), m.group(1).rstrip("，；")), fig, flags=re.S)
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
document.querySelectorAll('video[data-pauses]').forEach(function(v){
  var P=v.dataset.pauses.split(',').map(Number), i=0, last=0, auto=false;
  var bar=document.createElement('div'); bar.className='stepbar';
  bar.innerHTML='<button type="button" class="nx"></button><button type="button" class="rs gh">⟲ 重来</button>'+
                '<span></span><label><input type="checkbox"> 连续播放</label>';
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
  v.removeAttribute('autoplay'); toStart();
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
P.finish(_html, OUT, SECTIONS, "topic-05.html")
