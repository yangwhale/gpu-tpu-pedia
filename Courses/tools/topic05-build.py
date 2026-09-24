import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# -*- coding: utf-8 -*-
"""专题五 · 并行策略 —— 教材。

════════════════════════════════════════════════════════════════
主线（2026-09-24 定）：一刀一刀接力，每一刀都是上一刀留下的问题逼出来的
════════════════════════════════════════════════════════════════
    零  一张卡装不下 → 得切；一句话：用一种通信，换一份显存或一份算力
    一  先认识五种通信（集合通信）—— 后面每一刀多出来的都是这里的某一种
    二  第一刀 切数据：DP → FSDP（AllReduce 拆两半，白送）
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
第一节（集合通信）、第二节（切数据）与第八节（全景）是写完的；其余是 🚧 大纲，不补写。
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
import os
import re

import topic03_page as P

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
.animgrid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; max-width:1760px; margin:14px auto }
.animgrid figure { margin:0; min-width:0 }
.animcell figcaption { font-size:14px; color:var(--gray); margin-top:6px; line-height:1.6 }
.animgrid video { width:100%; border-radius:8px; display:block }
@media (max-width:900px) { .animgrid { grid-template-columns:1fr } }
</style>"""

D = '<span class="kind k-d">数据</span>'
S = '<span class="kind k-s">序列</span>'
M = '<span class="kind k-m">模型</span>'
X = '<span class="kind k-x">解耦</span>'
NEW = '<span class="kind k-new">新</span>'

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
    ／ <b>并行策略</b></div>
  <h1>并行策略</h1>
  <div class="hook">
    一个模型装不进一块卡，就得切开分到很多卡上。<br>
    <em>每切一刀，就在那一维上多出一种通信。用一种通信，换一份显存或一份算力。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    先认识五种通信，再一刀一刀地切：数据、权重、专家、序列，最后连工作一起拆开。
    每一刀都是上一刀留下的问题逼出来的。</p>
  <div class="chips">
    <span class="chip">前置 <b>专题四</b>（那张 16 字节的账）</span>
    <span class="chip">口径 <b>截至 2026-09</b></span>
    <span class="chip">⏱ <b>约 1 小时</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

<div class="wrap">
  <div class="note warn"><span class="t">🚧 这一讲写了三节</span>
    <b>第一节（集合通信）、第二节（切数据）和第八节（全景）是写完的</b>，出处在文末台账。<br>
    第三到第七节还是大纲，页面上按原样列出，没有补写。</div>
</div>
'''

BODY = sec("s零", "零", "一张卡装不下") + '''
  <p class="lead">专题四把账算完了：一个 6,710 亿参数的模型，按每参数 16 字节算，
    光常驻的训练状态就要 9.76 TiB。<b>一块卡装不下，就得切。问题是沿哪一维切。</b></p>

  <p>一个训练中的张量有好几个维度可以下刀：batch、序列、隐藏维、层、专家。
    每切一刀，就在那一维上产生一种通信。所以这一讲从头到尾只讲一件事：</p>

  <div class="note ok"><span class="t">一句话</span>
    <b>用一种通信，换一份显存或一份算力。</b><br>
    选并行策略，就是在选你愿意付哪一种通信、付多频繁。</div>

  <p>这一讲按一条接力线走。<b>每一刀都是上一刀留下的问题逼出来的</b>：</p>
  <ol>
    <li><b>先认识五种通信。</b>后面每一刀多出来的，都是其中某一种。</li>
    <li><b>第一刀，切数据。</b>最朴素，但每张卡还是存一整份模型，于是有了 FSDP。</li>
    <li><b>第二刀，切权重。</b>FSDP 每一层都要把整层权重拼回来，batch 一小就被搬权重拖垮，于是切进矩阵、切开层。</li>
    <li><b>第三刀，切专家。</b>MoE 的专家太多；而且 attention 和专家是两种形状，得各配各的。</li>
    <li><b>第四刀，切序列。</b>上下文一长，训练时激活爆，推理时 KV cache 爆。</li>
    <li><b>第五刀，不切张量，切工作。</b>prefill 和 decode 分开，attention 和专家分开。</li>
    <li><b>最后摆到真机器上</b>，再用一张全景地图把走过的路收一遍。</li>
  </ol>

  <p>整条线只用一把尺子量：<b>这一刀多出来的通信有多频繁，它就只能放在多快的链路上。</b></p>
</div></section>

''' + sec("s一", "一", "先认识五种通信") + '''
  <p class="lead">后面每一刀都会多出一种通信。这一节先把它们认全。
    名字看着多，但每一个都只回答两个问题：<b>谁发给谁</b>；数据到了之后是<b>拼起来、加起来，还是原样放着</b>。</p>

  <h3>1.1　所有集合通信，拆到底只有「发」和「收」</h3>
  <p>一组卡按同一个规则一起发、一起收，叫<b>集合通信</b>（collective communication）。
    拆到最底层，每一种都只是很多次「一张卡发、另一张卡收」按某个顺序排起来。</p>
  <p>下面几张图用同一套画法，只学一次：</p>
  <ul>
    <li>四张卡，<b>一张一个颜色</b>：卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫。</li>
    <li>每张卡的数据切成四块，一块一个小方格。</li>
    <li><b>加过的块画成竖条纹</b>，条纹是哪几种颜色，就是哪几张卡的数加在了一起。</li>
    <li>虚线框表示这里没有数据。</li>
  </ul>

  <h3>1.2　一个人对所有人：四个基本动作</h3>
__FIG_COLL_1N__
<div class="animgrid"><figure class="animcell" id="anim-broadcast"><video src="media/topic05-broadcast.mp4" autoplay loop muted playsinline aria-label="Broadcast 广播 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：Broadcast 广播。字幕：卡 0 的整份数据，复制给每一个人。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>Broadcast 广播</b>：卡 0 的整份数据，复制给每一个人<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-scatter"><video src="media/topic05-scatter.mp4" autoplay loop muted playsinline aria-label="Scatter 分发 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：Scatter 分发。字幕：卡 0 把第 j 块发给卡 j，自己只留第 0 块。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>Scatter 分发</b>：卡 0 把第 j 块发给卡 j，自己只留第 0 块<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-gather"><video src="media/topic05-gather.mp4" autoplay loop muted playsinline aria-label="Gather 收集 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：Gather 收集。字幕：每人把自己那块交给卡 0，卡 0 按顺序拼起来。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>Gather 收集</b>：每人把自己那块交给卡 0，卡 0 按顺序拼起来<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-reduce"><video src="media/topic05-reduce.mp4" autoplay loop muted playsinline aria-label="Reduce 归约 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：Reduce 归约。字幕：每人把整份交给卡 0，卡 0 逐块相加。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>Reduce 归约</b>：每人把整份交给卡 0，卡 0 逐块相加<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure></div>
  <p>这四个都有一个「班长」：所有数据要么从它那里发出去，要么都往它那里送。
    班长那一条线要扛下全部流量，卡越多越堵。</p>

  <h3>1.3　人人对人人：训练里天天在跑的四个</h3>
__FIG_COLL_NN__
<div class="animgrid"><figure class="animcell" id="anim-allgather"><video src="media/topic05-allgather.mp4" autoplay loop muted playsinline aria-label="AllGather 全收集 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：AllGather 全收集。字幕：每人把自己那块发给所有人：只拼，不加。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>AllGather 全收集</b>：每人把自己那块发给所有人：只拼，不加<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-reducescatter"><video src="media/topic05-reducescatter.mp4" autoplay loop muted playsinline aria-label="ReduceScatter 归约分散 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：ReduceScatter 归约分散。字幕：第 j 块全部送到卡 j 加起来：先加，再分。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>ReduceScatter 归约分散</b>：第 j 块全部送到卡 j 加起来：先加，再分<span class="sub">（5 秒无声循环，Manim 渲染。）</span></figcaption></figure><figure class="animcell" id="anim-allreduce"><video src="media/topic05-allreduce.mp4" autoplay loop muted playsinline aria-label="AllReduce 全归约 动画。四张卡，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，虚线框是空位，条纹块是加过的。标题：AllReduce 全归约。字幕：① ReduceScatter 各拿一块总和；② AllGather 总和发给所有人。块从发送的卡飞到接收的卡，最后画面复位到开始的样子。"></video><figcaption><b>AllReduce 全归约</b>：① ReduceScatter 各拿一块总和；② AllGather 总和发给所有人<span class="sub">（8 秒无声循环，Manim 渲染。）</span></figcaption></figure></div>
  <p>All 就是「人人都拿到结果」。拿上一组对照着看：</p>
  <ul>
    <li><b>AllGather</b> ＝ Gather，再把拼好的结果发给每个人。</li>
    <li><b>AllReduce</b> ＝ Reduce，再把加好的结果发给每个人。</li>
    <li><b>ReduceScatter</b> ＝ Reduce，再把结果切开，一人一块。</li>
    <li><b>AllToAll</b> 独一份：每一对卡之间各传一份专属的数据，不加也不拼。</li>
  </ul>

  <h3>1.4　AllReduce 可以拆成两半</h3>
__FIG_AR_SPLIT__
  <p>这两半各自单独拿出来，就是两种有用的通信。<b>后面好几种并行白捡的便宜，全从这里来</b>：</p>
  <ul>
    <li><b>FSDP</b>：数据并行同步梯度那一次 AllReduce，一半挪到反向去分梯度，一半挪到前向去拼权重。
      账面上一个字节没多花，每张卡却只需要长期存 1/n 的权重。第二节细讲。</li>
    <li>张量并行每层要做 AllReduce；配上序列并行时，也是把它拆成这两半，分别挪到不同位置。第三节细讲。</li>
  </ul>

  <h3>1.5　没有班长，怎么做到的：环</h3>
  <p>最直接的 AllReduce 是班长模式：大家先把数据交给卡 0 加起来，卡 0 再广播回去。
    卡 0 要收 n−1 份、发 n−1 份，卡一多，它那条线就成了全场的瓶颈。</p>
  <p>换个办法：把卡首尾相连排成一圈，谁都不当班长。</p>
__FIG_RING__
<figure class="fbox fwide" id="anim-ring">
<video src="media/topic05-ring.mp4" autoplay loop muted playsinline
       aria-label="环形 AllReduce 动画。四张卡排成一排，卡 0 蓝、卡 1 橙、卡 2 绿、卡 3 紫，每张卡四块，箭头从卡 0 依次指向卡 3，再从卡 3 绕回卡 0。字幕一：ReduceScatter：每人往右发一块，收到的加到自己那块上。三步里，每一步四张卡同时把一块飞给右边的邻居，落地后那一块多出一种颜色的条纹。字幕二：三步之后：每张卡恰好握着一块完整总和。字幕三：AllGather：把总和接着往右传，只替换，不相加。再转三步，完整总和一块块复制过去。字幕四：人人一份总和 ＝ AllReduce ＝ ReduceScatter ＋ AllGather。最后画面复位到四张卡的初始状态。"></video>
<figcaption>同一个环转两圈：前三步<b>加</b>（ReduceScatter），后三步<b>拼</b>（AllGather），合起来就是一次 AllReduce。
  <span class="sub">（15 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>ReduceScatter 转 n−1 步、AllGather 再转 n−1 步，每张卡一共发出 2(n−1)/n 份数据。
    卡再多，也不到两整份。</p>
  <p>代价是步数跟着卡数涨。数据很大时，比的是带宽，环几乎是最优的；
    数据很小时，比的是一步一步的等待，步数多反而吃亏。
    所以 NCCL 这类通信库会按消息大小，在环形、树形等几种算法之间自己挑。</p>
  <p><em>TPU 这边更直接：芯片之间的 ICI 本身就连成环面，每一维天然就是一个环。</em></p>

  <h3>1.6　AllToAll：每人给每人一份不一样的</h3>
__FIG_A2A__
<figure class="fbox fwide" id="anim-a2a">
<video src="media/topic05-a2a.mp4" autoplay loop muted playsinline
       aria-label="AllToAll 动画。四张卡各有四块，颜色表示出自哪张卡，对角线上的四块画粗框。字幕一：派发：卡 k 的第 j 块 → 发给卡 j（粗框是自己留给自己的，不走网络）。十六块同时飞到新位置，卡 k 的第 j 块落到卡 j 的第 k 行。字幕二：卡 j 收齐了四个人给它的那一份 —— 一张表转置了一次。字幕三：专家算完，再转置一次送回去 —— MoE 每层两次 AllToAll。十六块原路飞回，画面回到开始的样子。"></video>
<figcaption>派发过去、送回来，正好是专家并行每层的两次 AllToAll。
  <span class="sub">（8 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>它是这五种里<b>唯一一个「人人对人人发不同数据」的</b>。
    专家并行派发 token 要用它（每层两次：发过去、送回来）；切序列时的 Ulysses 也用它，
    在「按序列切」和「按头切」之间来回换。</p>

  <h3>1.7　一张表收住</h3>
  <p>「每卡发出」一列按最优算法算，S 是一整份数据的大小（AllGather 指拼好之后那一整份，
    ReduceScatter 指加之前那一整份）。系数就是 NCCL 官方测试工具换算总线带宽时用的那一组。</p>
  <table>
    <tr><th>通信</th><th>做什么</th><th>每卡发出</th><th>后面谁在用</th></tr>
    <tr><td>AllReduce</td><td>加完，人人一份</td><td>2(n−1)/n · S</td><td>数据并行同步梯度；张量并行每层两次</td></tr>
    <tr><td>ReduceScatter</td><td>加完，各拿一块</td><td>(n−1)/n · S</td><td>FSDP 反向分梯度；张量并行配序列并行</td></tr>
    <tr><td>AllGather</td><td>拼完，人人一份</td><td>(n−1)/n · S</td><td>FSDP 前向拼权重；张量并行配序列并行；推理切 KV 时收齐 Q</td></tr>
    <tr><td>AllToAll</td><td>只换位置</td><td>(n−1)/n · S</td><td>专家并行派发和收回 token；Ulysses</td></tr>
    <tr><td>Broadcast / Reduce</td><td>一对多 / 多对一</td><td>S</td><td>分发初始权重、汇总指标这类场合</td></tr>
    <tr><td>Send / Recv</td><td>一对一</td><td>看传多少</td><td>流水线并行在 stage 之间传激活；Ring Attention 沿环传 KV</td></tr>
  </table>

  <div class="note ok"><span class="t">这一节只要带走两件事</span>
    <b>① AllReduce 可以拆成 ReduceScatter 和 AllGather 两半</b>，后面好几刀都靠它白捡便宜。<br>
    <b>② AllToAll 是唯一一个人人对人人发不同数据的</b>，专家并行离不开它，网络也最怕它。<br>
    后面五刀，每一刀多出来的通信都是这张表里的某一行。</div>
</div></section>

''' + sec("s二", "二", "第一刀：切数据") + '''
  <p class="lead">最朴素的一刀是切数据：每张卡算不同的样本。
    <b>可它一开始根本没解决「装不下」</b>，要一路削到 FSDP 才解决 —— 而最后那一步是要付钱的。</p>

  <h3>2.1　数据并行：每张卡一整份模型</h3>
  <p>n 张卡，每张卡都放一整份模型，各算各的一批样本。反向算完，每张卡手里是<b>自己那批样本的梯度</b>，
    大家的不一样，所以要做一次 AllReduce 求平均，然后每张卡用同一份梯度更新同一份权重。</p>
  <ul>
    <li><b>通信</b>：每一步只有这一次 AllReduce，量是 2Ψ（Ψ 是参数个数）。</li>
    <li><b>频率</b>：一步一次。这是后面所有刀里最低的，所以它最能忍受慢链路。</li>
    <li><b>问题</b>：每张卡还是存一整份 16 字节／参数。V3 就是每张卡 9.76 TiB，<b>装不下的问题一点没变</b>，
      而且 n 张卡存了 n 份一模一样的东西。</li>
  </ul>

  <h3>2.2　ZeRO：按大小顺序，一级一级削</h3>
__FIG_ZERO_MEM__
  <p>16 字节里，12 个是优化器状态。ZeRO 按大小顺序一级一级削：</p>
  <ul>
    <li><b>ZeRO-1</b> 切优化器状态：每张卡只管 1/n 的参数，只存这 1/n 的状态。</li>
    <li><b>ZeRO-2</b> 再切梯度：每张卡只需要自己负责那 1/n 参数的梯度。</li>
    <li><b>ZeRO-3</b> 连权重也切：每张卡只长期存 1/n 的权重。</li>
  </ul>
  <p>前两级为什么不多花一个字节的通信？就是第一节那个等式。数据并行那次 AllReduce 拆开来做：
    先 ReduceScatter，每张卡正好拿到自己负责那 1/n 的梯度总和，就地更新那 1/n 的参数；
    再 AllGather，把更新好的权重拼回给每个人。<b>通信还是 2Ψ，显存却不用再存别人的状态和梯度了。</b></p>

  <h3>2.3　ZeRO-3 ＝ FSDP：最后那 2 个字节要付 50%</h3>
  <p>削完前两级，每参数还剩 2 字节的权重。V3 按 1,024 路算，每卡仍要 1.23 TiB，照样装不下。
    所以大模型只能走到 ZeRO-3，也就是 PyTorch 里的 <b>FSDP</b>：每层要算之前先 AllGather 拼回这一层，算完就扔。</p>
__FIG_FSDP_STEP__
  <p>代价在反向：前向扔掉的权重，反向还得再拼一次。所以一层一步要做 AG、AG、RS 三次，
    通信是 3Ψ，<b>数据并行的 1.5 倍</b>（ZeRO 原论文 §7 的结论）。换来的是每卡常驻从 9.76 TiB 降到 9.76 GiB。</p>
  <div class="note warn"><span class="t">⚠️ 「FSDP 白送」这句话要说准</span>
    白送的是 ZeRO-1、ZeRO-2：通信一个字节不多，已经削掉 16 字节里的 14 个。<br>
    ZeRO-3 不白送：最后那 2 个字节，要多付一半的通信。只是对大模型来说，这笔钱非付不可。</div>

  <h3>2.4　这一刀能放多远：看频率</h3>
  <ul>
    <li><b>数据并行、ZeRO-1、ZeRO-2</b>：一步通信一次，可以放在慢链路上，最远能横跨数据中心。</li>
    <li><b>FSDP</b>：每一层都要拼一次权重，频率一下子高了几十倍，得放在高带宽域里。</li>
    <li><b>HSDP</b> 是两者的折中：<b>机内 FSDP、机间数据并行</b>。高频的拼权重留在机内，跨机只剩一步一次的梯度同步。</li>
    <li><b>DiLoCo</b> 再往前一步：每个副本先自己走几百步再同步一次，专门为跨数据中心训练设计。</li>
  </ul>

  <h3>2.5　这一刀留下的问题</h3>
  <p>FSDP 每一步搬的是<b>权重</b>，搬多少只跟参数量有关，<b>跟 batch 无关</b>；
    而每一步要算多少，跟每张卡分到的 token 数成正比。</p>
  <p>所以每张卡的 batch 一小，算得少、搬得一样多，时间就被搬权重吃掉了。
    要继续加卡、又不能让每张卡的 batch 变小，就得换一种切法：<b>不再拼整层权重，直接把权重切开</b>。这是第二刀。</p>
</div></section>

''' + sec("s三", "三", "第二刀：切权重") + todo([
    "FSDP 的尽头：每一层都要把整层权重拼回来；<b>FSDP 搬权重、TP 搬激活</b>，batch 小时搬权重吃光时间",
    "<b>TP</b>：切进矩阵，每层两次 AllReduce，频率极高 → 只能待在 NVLink / ICI 一跳之内；度数受头数限制",
    "<b>SP（Megatron）</b>：TP 的搭档，把 AllReduce 拆成 AG ＋ RS，顺手切掉 LayerNorm 那几段的激活",
    "<b>PP / VPP</b>：按层切，通信最少，代价是气泡；V3 前 3 层 dense、后 58 层 MoE 怎么切才平衡",
]) + '''
</div></section>

''' + sec("s四", "四", "第三刀：切专家") + todo([
    "<b>EP</b>：专家放到不同卡上，每层两次 AllToAll",
    "EP 独有的病：<b>发给谁由数据决定</b>，负载天生不均（EPLB、冗余专家）",
    "一个模型里 attention 和专家是两种形状 → <b>各配各的</b>：训练叫 Parallel Folding（ETP / EP / EDP），推理叫 DEP / TEP",
    "Wide-EP：把 EP 铺到几十张卡，decode 时每个专家的 batch 才做得大",
]) + '''
</div></section>

''' + sec("s五", "五", "第四刀：切序列") + todo([
    "训练：激活沿序列爆 → <b>CP</b>（ring / Ulysses / 两者叠加）；为什么 128K 以上绕不开",
    "推理：KV cache 沿序列爆 → <b>DCP</b>；TP 超过 KV 头数时 KV 被复制，DCP 用一次合并通信把冗余换回容量",
    "PCP：prefill 端切长 prompt，压首字延迟；PD 分离时和 DCP 一边一个",
]) + '''
</div></section>

''' + sec("s六", "六", "第五刀：不切张量，切工作") + todo([
    "<b>PD 分离</b>：prefill 吃算力、decode 吃带宽，放在一起互相拖累；中间靠传 KV cache 衔接",
    "<b>AFD</b>：attention 和专家分到两组机器，把专家的 batch 做大",
    "两边各自再挑并行方式（比如 P 端 TEP、D 端 DEP，或者反过来）",
]) + '''
</div></section>

''' + sec("s七", "七", "摆到机器上") + todo([
    "<b>mesh</b>：把设备排成多维网格，每个并行维绑一根轴；高频的绑最快的轴，低频的绑最慢的",
    "我们的实测：换拓扑和调参数，收益不在一个量级",
    "<b>怎么选</b>：先装得下（FSDP / EP）→ 看高带宽域多大（TP / EP 上限）→ 看序列多长（CP / DCP）→ PP 最后 → DP 兜底",
    "<b>怎么评</b>：strong scaling 与 weak scaling",
]) + '''
</div></section>

''' + sec("s八", "八", "全景：今天所有的并行方式") + '''


  <p class="lead">名字很多，DP、FSDP、TP、SP、CP、PP、EP，推理那边还有 DCP、PCP、DEP、TEP……
    <b>但刀法只有四种。</b>先把刀法认清楚，名字就好记了。</p>

  <h3>8.1　四种刀法</h3>
  <p>判据只有一个：<b>看它切的是什么。</b></p>
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

  <p>有两个地方要多说一句。</p>
  <ul>
    <li><b>FSDP 算数据并行，不算模型并行。</b>它确实把权重分片存着，
      但每一层算之前都会把完整权重临时拼回来，每张卡算的还是自己那份样本。
      <em>它省的是显存，不改变「谁算什么」。</em></li>
    <li><b>序列并行单独成一类。</b>严格说它也是在切数据（切一条样本内部），
      但它专门对付长上下文，通信方式也跟 DP 完全不同，放一起反而讲不清。</li>
  </ul>
  <p>下面四张表就按这四种刀法排。「新」表示 2025–26 年才出现或才普及。</p>

  <h3>8.2　数据并行类</h3>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>DP / DDP</td><td>batch，模型整份复制</td><td>线性扩吞吐</td>
      <td>训练：梯度 all-reduce，每个 step 一次。推理：没有，前面靠 router 分流</td><td>训 · 推</td></tr>
    <tr><td>ZeRO-1 / 2 / 3</td><td>依次多切一样：优化器状态 → 梯度 → 参数</td>
      <td>DP 每张卡存一份完整状态，是纯冗余</td><td>reduce-scatter ＋ all-gather；ZeRO-3 每层都要把参数 all-gather 回来</td><td>训</td></tr>
    <tr><td>FSDP / FSDP2</td><td>就是 ZeRO-3，PyTorch 原生版本</td>
      <td>通信量和 DP 同一个量级，显存却随卡数线性下降</td><td>前向 all-gather 权重，反向 reduce-scatter 梯度</td><td>训</td></tr>
    <tr><td>HSDP</td><td>机内分片，机间复制</td><td>把最频繁的 all-gather 圈在高带宽域里</td>
      <td>机内 AG / RS，机间 all-reduce</td><td>训</td></tr>
    <tr><td>ZeRO++</td><td>ZeRO-3 上再加三招</td><td>跨节点通信太贵</td>
      <td>权重量化成 INT8 再 all-gather；节点内多存一份参数；梯度也量化</td><td>训</td></tr>
    <tr><td>跨 DCN 的 DP</td><td>DP 横跨多个 slice 或数据中心</td>
      <td>DP 每个 step 才通信一次，是唯一放得上慢链路的维度</td><td>DCN 上的梯度 all-reduce</td><td>训</td></tr>
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
    <tr><td>Megatron SP</td><td>只切 LayerNorm、Dropout、残差这几段的激活，<b>必须跟 TP 搭配</b></td>
      <td>TP 切不到的那部分激活，每张卡都存了一整份</td><td>把 TP 的 all-reduce 拆成 all-gather ＋ reduce-scatter，总量不变</td><td>训 · 推</td></tr>
    <tr><td>CP（Context Parallel）</td><td><b>所有</b>激活沿序列切</td><td>长上下文训练，128K 以上基本绕不开</td>
      <td>ring 传 KV，或者 all-to-all，或者 all-gather，可以分层组合</td><td>训</td></tr>
    <tr><td>Ulysses</td><td>序列切和 head 切来回转换</td><td>长序列：序列长度和卡数同比放大时，每张卡的通信量不变</td>
      <td>attention 前后各一次 all-to-all；<b>度数不能超过 head 数</b></td><td>训</td></tr>
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
      <td>先把 Q 收齐，各卡在自己那段 KV 上算 attention，再带着 LSE 合并结果</td><td>推</td></tr>
    <tr><td>Helix ''' + NEW + '''</td><td>同一组卡在一层里换两次布局：attention 按 KV 切，FFN 按 TP × EP 切</td>
      <td>百万 token 级的 decode：读 KV 和读权重两件事都要摊开</td><td>一次 all-to-all 交换部分结果</td><td>推</td></tr>
    <tr><td>MaxText <code>context_autoregressive</code></td><td>decode 时 KV 沿序列切，FFN 按专家切</td>
      <td>同上，TPU 上的做法</td><td>XLA 自动插入</td><td>推</td></tr>
  </table>

  <div class="note ok"><span class="t">PD 分离的时候，PCP 和 DCP 正好一边一个</span>
    <b>P 节点用 PCP</b>：把长 prompt 切开，压首字延迟。<br>
    <b>D 节点用 DCP</b>：把 KV 摊开，让 decode 装得下更多、更长的请求。<br>
    两者对卡数的作用也正好相反：<b>PCP 要加卡，DCP 不加卡</b>，直接复用 TP 那几张卡。
    <br><em>这是 vLLM 源码里的原话：PCP expands the process world size；DCP does not expand
    the process world size, without PCP it reuses TP ranks。</em></div>

  <p>DCP 也是这一讲那句话最好的新例子。<b>它付出的是每层一次合并通信，换回来的是被 TP 白白复制掉的那 7/8 份 KV。</b>
    用一种通信，换一份显存。</p>

  <h3>8.4　模型并行类</h3>
  <p>这一类真正在切权重。按下刀的位置再分三种：矩阵内部、层、专家。</p>

  <p><b>切矩阵内部（张量并行）</b></p>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>TP</td><td>矩阵先按列切、再按行切，两两配对</td><td>单层的权重或计算放不下</td>
      <td>每层两次 all-reduce，<b>频率极高</b>，所以只能待在 NVLink 或 ICI 一跳之内</td><td>训 · 推</td></tr>
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
      <td>token 发出去、算完收回来，两次 all-to-all。<b>发给谁由数据决定</b>，所以负载会不均，这是 EP 独有的病</td><td>训 · 推</td></tr>
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
    <b>两套是分开配的</b>。推理时同样如此，只是换了一套名字，见 8.6。
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
      <td>attention → 专家发一次，专家 → attention 收一次，要切小批次把它藏起来</td><td>推</td></tr>
    <tr><td>Encoder 分离</td><td>多模态模型的视觉编码器单独部署</td><td>编码器和语言模型抢资源</td><td>embedding 传输</td><td>推</td></tr>
    <tr><td>多模块异构并行</td><td>编码器和语言模型各用一套并行方式</td><td>多模态训练里两部分的形状差太远</td><td>两套网格之间的桥接通信</td><td>训</td></tr>
  </table>
  <p><em>AFD 目前的代表是字节的 MegaScale-Infer 和阶跃的 Step-3，vLLM 在 2026 年 7 月出了实验性插件。</em></p>

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
    vLLM 和 NVIDIA Dynamo 的用法一致。</p>
  <p>PD 分离时两边经常各选一种，但<b>哪边用哪种没有定式</b>，要看模型和负载：
    vLLM 部署 Kimi K3 用的是 <b>TEP8 做 prefill、DEP16 做 decode</b>；
    TensorRT-LLM 那篇文章举的例子却是 <b>DEP4 做 prefill、TEP8 做 decode</b>。
    <em>这正是第七节要讲的「怎么选」，这里先记住两个名字各是什么。</em></p>

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
  </table>

  <h3>8.8　同名不同义</h3>
  <table>
    <tr><th>词</th><th>在不同地方的意思</th></tr>
    <tr><td>SP</td><td>至少三种：Megatron 的 SP（只切 LayerNorm 那几段，跟着 TP）；DeepSpeed 说的 SP（指 Ulysses）；
      以及泛指一切切序列的做法。Megatron 里「所有激活都切」叫 CP</td></tr>
    <tr><td>CP</td><td>训练里指切激活。vLLM 把它拆成两个作用相反的开关：PCP 加卡，DCP 不加卡</td></tr>
    <tr><td>DP</td><td>dense 模型上是独立副本；MoE 推理里其实是 Attention DP，每一步都要同步</td></tr>
    <tr><td>ETP</td><td>Megatron 指专家内部的 TP；TensorRT-LLM 的 Hybrid ETP 指专家层 TP 和 EP 混用</td></tr>
    <tr><td>hierarchical</td><td>ZeRO++ 的分层分片、Megatron 的分层 DP、Megatron 的分层 CP，是三件不同的事</td></tr>
  </table>

</div></section>

''' + sec("s九", "九", "出处台账") + '''
  <p class="lead">按「结论 ← 材料」排。第八节 2026-09-23 核对，第一节 2026-09-24 核对。</p>
  <table>
    <tr><th>结论</th><th>材料</th></tr>
    <tr><td>各集合通信每卡发出的量（第一节的表）</td><td>NVIDIA/nccl-tests：doc/PERFORMANCE.md 的 bus bandwidth 修正系数：AllReduce 2(n−1)/n，ReduceScatter / AllGather / AlltoAll (n−1)/n，Broadcast / Reduce 1</td></tr>
    <tr><td>环形 ReduceScatter 的逐步推演、班长模式</td><td>wanghonglei《分布式深度学习集体通信原语——从零到精通》（2026-06-27）第 1–2 章；图里每一步由脚本按调度现算并断言。块号比原文挪了一位，让卡 k 最后拿第 k 块</td></tr>
    <tr><td>ZeRO 各级的显存与通信（第二节）</td><td>Rajbhandari 等，ZeRO，arXiv 1910.02054 §5、§7：Pos、Pos+g 通信量与数据并行相同（2Ψ），Pos+g+p 最多 1.5 倍；显存 16Ψ → 16Ψ/Nd</td></tr>
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
    上一讲 <a href="topic-04.html">专题四 · 反向与优化器</a><br>
    本页由 <code>Courses/tools/topic05-build.py</code> 生成 ——&nbsp;<b>正文写在那个脚本里</b>。</p>
</div>

</body></html>'''

FIGS = {
    "__FIG_ZERO_MEM__": ("fig-zero-mem", "fig5-zero-mem.svg", "topic05-fig-zero.py",
        '<b>16 字节里，优化器状态独占 12 个 —— 所以先削它。</b><br>'
        '<em>ZeRO-3 那条短到几乎看不见 —— 每卡从 9.76 TiB 降到 9.76 GiB，正好除以 1,024。</em>'),
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
        '<em>每一步是脚本按调度现算的，不是手摆的。</em>'),
    "__FIG_A2A__": ("fig-a2a", "fig5-a2a.svg", "topic05-fig-coll.py",
        '<b>左边一行是「我要发给谁」，右边一行是「谁发给了我」。</b><br>'
        '<em>每一格多大，在 MoE 里要等路由算完才知道。</em>'),
}

_html = head + HERO + BODY + FOOT
_html = P.place_figs(_html, FIGS)
_leak = sorted(set(re.findall(r"__[A-Z][A-Z_0-9]*__", _html)))
assert not _leak, "占位符没落地：%s" % "、".join(_leak)
for _tag in ("h2", "h3", "section", "div"):
    _o = len(re.findall(r"<%s[ >]" % _tag, _html))
    _c = _html.count("</%s>" % _tag)
    assert _o == _c, "<%s> 开 %d 个、闭 %d 个" % (_tag, _o, _c)
_html = P.add_figonly_toggle(_html)
P.finish(_html, OUT, SECTIONS, "topic-05.html")
