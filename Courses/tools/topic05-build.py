import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# -*- coding: utf-8 -*-
"""专题五 · 并行策略 —— 教材。

════════════════════════════════════════════════════════════════
主线（2026-09-24 定）：一刀一刀接力，每一刀都是上一刀留下的问题逼出来的
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
import os
import re

import topic03_page as P
import course_ai_trainer as AIT

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
.lecbtn { display:inline-block; margin-left:14px; font-size:13px; padding:4px 12px; border-radius:14px;
          background:#fff; border:1px solid var(--line); color:var(--blue); text-decoration:none }
.lecbtn:hover { background:var(--blue-l); border-color:var(--blue) }
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
    <span class="chip">⏱ <b>讲约 48 分钟</b></span>
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
  <p class="lead">专题四把账算完了：一个 6,710 亿参数的模型，按每参数 16 字节算，
    光常驻的训练状态就要 9.76 TiB。<b>一块卡装不下，就得切。问题是沿哪一维切。</b></p>

  <p>一个训练中的张量有好几个维度可以下刀：batch、序列、隐藏维、层、专家。
    每切一刀，就在那一维上产生一种通信。所以这一讲从头到尾只讲一件事：</p>

  <div class="note ok"><span class="t">一句话</span>
    <b>用一种通信，换一份显存或一份算力。</b><br>
    选并行策略，就是在选你愿意付哪一种通信、付多频繁。</div>

  <p>这一讲按一条接力线走。<b>每一刀都在补前面没管到的那一块</b>：前一刀撑不住了，或者模型换了形状、上下文变长了，就得换一刀。</p>
  <ol>
    <li><b>先认识五种通信。</b>后面每一刀多出来的，都是其中某一种。</li>
    <li><b>第一刀，切数据。</b>最朴素，但每张卡还是存一整份模型，于是有了 FSDP（全分片数据并行：连模型本身也分片存）。</li>
    <li><b>第二刀，切权重。</b>FSDP 每一层都要把整层权重拼回来，batch 一小就被搬权重拖垮，于是切进矩阵、切开层。</li>
    <li><b>第三刀，切专家。</b>模型换了形状：MoE 的参数几乎全在一堆窄窄的专家里，TP 不对路了；而且 attention 和专家是两种形状，得各配各的。</li>
    <li><b>第四刀，切序列。</b>前三刀都没碰过的一维：上下文一长，训练时激活爆，推理时 KV cache 爆。</li>
    <li><b>第五刀，不切张量，切工作。</b>prefill 和 decode 分开，attention 和专家分开。</li>
    <li><b>最后摆到真机器上</b>，再用一张全景表把走过的路收一遍。</li>
  </ol>

  <p>整条线主要用一把尺子量：<b>这一刀多出来的通信有多频繁，它就只能放在多快的链路上。</b>
    第三节还会拿出一把配套的：每搬一个字节换来多少计算。频率决定它放哪根线，这一把决定它会不会被拖住。</p>

  <details class="foldfig"><summary><b>遇到不认识的词，回这儿查</b>：张量、激活、注意力头、MoE、KV cache、prefill 与 decode、Q/K/V、MLA、显存、ICI、「几路」……（专题一、三讲过，这里各一句话）</summary>
  <p style="line-height:1.9"><b>张量与隐藏维</b>：模型里流动的数据都是多维数组，叫张量；每个 token 用一串数表示，这串数的长度就是隐藏维（V3 是 7,168）。<br>
    <b>激活</b>：前向每一层算出来的中间结果。反向时还要用，所以得先存着。<br>
    <b>注意力头</b>：注意力被拆成几十份并排的小注意力，每一份叫一个头，各算各的。<br>
    <b>micro-batch</b>：把一步要算的一批样本再切成几小份，一份一份地过。<br>
    <b>V3</b>：本讲反复拿来举例的 DeepSeek-V3，6,710 亿参数的 MoE 模型。<br>
    <b>Q、K、V</b>：注意力里每个 token 算出三样东西：Q 是「我要找什么」，K 是「我是什么」，V 是「我带着什么内容」。
    新 token 拿自己的 Q 去和前面所有 token 的 K 比，按相似度把它们的 V 加权合起来。<br>
    <b>MLA</b>：DeepSeek 的一种注意力，把 K 和 V 压成一小份所有头共用，所以 KV cache 小得多，但也只剩「一个头」可切。<br>
    <b>节点</b>：一台机器，里面通常有 4 到 8 张卡，机器内部走高速互联（NVIDIA 的叫 NVLink）。<br>
    <b>显存（HBM）</b>：卡上自带的高速内存，模型和中间结果都得放在这儿。<br>
    <b>算力和带宽</b>：算力是每秒能算多少次，带宽是每秒能搬多少字节。一件活要是搬得多算得少，就是「吃带宽」，反过来是「吃算力」。<br>
    <b>ICI 和切片</b>：TPU 芯片之间的专用快线叫 ICI；用 ICI 连在一起的一整块芯片叫一个切片，切片和切片之间走普通的数据中心网络。<br>
    <b>device 和轴</b>：v7 的一颗芯片，对软件显示成 2 个 device，并行度按 device 数；切片里的芯片排成三维网格，x、y、z 三个方向各叫一根轴。<br>
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

  <h3>1.1　所有集合通信，拆到底只有「发」和「收」</h3>
  <p>一组卡按同一个规则一起发、一起收，叫<b>集合通信</b>（collective communication）。
</p>
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
    按最朴素的做法（班长挨个发、挨个收），班长那一条线要扛下全部流量，卡越多越堵。
    通信库会把广播、归约排成一条链接力传，让班长只发或只收一份；收集和分发的班长省不掉那份量，它手里本来就是 n 份不同的东西（总量倒是不随卡数涨）。</p>

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
    <li><b>ZeRO ／ FSDP</b>：数据并行同步梯度那一次 AllReduce 拆开：ReduceScatter 让每张卡只拿自己那 1/n 梯度的总和、就地更新那 1/n 参数；
      AllGather 拼回来的，是更新好的新权重。账面上一个字节没多花，每张卡却只需要存 1/n 的梯度和优化器状态；
      连权重也只存 1/n，就要多付一半。第二节细讲。</li>
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
    所以 NCCL 这类通信库会按消息大小，在环形、树形等几种算法之间自己挑。
    NVLink 交换机（NVSwitch）还能在交换机里直接做加法，每张卡发出去的又能少将近一半。</p>
  <p><em>TPU 这边更直接：芯片之间的 ICI 本身就连成环面，切片够大（每一维都是 4 的整倍数）时，每一维天然就是一个首尾相接的环；
    更小的切片某一维只是一条线，环要在线上折返，带宽约减半。</em></p>

  <h3>1.6　AllToAll：每人给每人一份不一样的</h3>
__FIG_A2A__
<figure class="fbox fwide" id="anim-a2a">
<video src="media/topic05-a2a.mp4" autoplay loop muted playsinline
       aria-label="AllToAll 动画。四张卡各有四块，颜色表示出自哪张卡，对角线上的四块画粗框。字幕一：派发：卡 k 的第 j 块 → 发给卡 j（粗框是自己留给自己的，不走网络）。十六块同时飞到新位置，卡 k 的第 j 块落到卡 j 的第 k 行。字幕二：卡 j 收齐了四个人给它的那一份 —— 一张表转置了一次。字幕三：专家算完，再转置一次送回去 —— MoE 每层两次 AllToAll。十六块原路飞回，画面回到开始的样子。"></video>
<figcaption>派发过去、送回来，正好是专家并行每层的两次 AllToAll。
  <span class="sub">（8 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>专家并行派发 token 要用它；切序列时的 Ulysses 也用它，
    在「按序列切」和「按头切」之间来回换。</p>

  <h3>1.7　一张表收住</h3>
  <p>「每卡发出」一列按点对点链路上的最优算法（环）算，S 是一整份数据的大小（AllGather 指拼好之后那一整份，
    ReduceScatter 指加之前那一整份）；Broadcast ／ Reduce 那一行按链式接力算。</p>
  <table>
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
    如果只靠数据并行这一刀，大模型只能走到 ZeRO-3，也就是 PyTorch 里的 <b>FSDP</b>：每层要算之前先 AllGather 拼回这一层，算完就扔。</p>
__FIG_FSDP_STEP__
<figure class="fbox fwide" id="anim-fsdp">
<video src="media/topic05-fsdp.mp4" autoplay loop muted playsinline
       aria-label="FSDP 一步的动画。四张卡，每张卡三层，每层只长期持有自己那一段（四分之一）。标题：FSDP：每层用之前借回来，用完就还。字幕一：前向：每层先 AllGather 拼回整层，算完只留自己那一段。第 1、2、3 层依次：别人的三段飞进来拼成整层，整层亮一下，再把借来的三段扔掉。底部计数一次次加一。字幕二：反向：扔掉的权重要再拼一次；算出的梯度 ReduceScatter 给各自的主人。从第 3 层往回：再拼一次整层、亮一下、扔掉，然后黄色的梯度小块飞回各自的主人。字幕三：每层三次：前向拼一次，反向再拼一次、散一次。计数停在 9。最后复位。"></video>
<figcaption>整层只在用的那一刻出现；平时每张卡只拿着每层的四分之一。
  <span class="sub">（14 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>代价在反向：前向扔掉的权重，反向还得再拼一次。所以一层一步要做 AG、AG、RS 三次，
    通信是 3Ψ，<b>数据并行的 1.5 倍</b>（ZeRO 原论文 §7 的结论）。换来的是每卡常驻从 9.76 TiB 降到 9.76 GiB。</p>
  <div class="note warn"><span class="t">⚠️ 「FSDP 白送」这句话要说准</span>
    一步只同步一次时，ZeRO-1、ZeRO-2 白送：通信一个字节不多，已经削掉 16 字节里的 14 个。<br>
    一步切成几个 micro-batch 时，只剩 ZeRO-1 白送：ZeRO-2 手里没有完整梯度，每个 micro-batch 都得分一次。所以配流水线时，V3、Megatron 都选 ZeRO-1。<br>
    ZeRO-3 不白送：最后那 2 个字节，要多付一半的通信。只是对大模型来说，这笔钱非付不可。</div>

  <h3>2.4　这一刀能放多远：看频率</h3>
  <ul>
    <li><b>数据并行、ZeRO-1</b>：一步通信一次，可以放在慢链路上，最远能横跨数据中心（ZeRO-2 只在不切 micro-batch 时才如此）。</li>
    <li><b>FSDP</b>：每一层都要拼一次权重，一步下来多出成百上千次（第七节有个数），得放在高带宽域里。</li>
    <li><b>HSDP</b> 是两者的折中：<b>机内 FSDP、机间数据并行</b>。高频的拼权重留在机内，跨机只剩梯度同步，量也除以了机内的分片数。</li>
    <li><b>DiLoCo</b> 再往前一步：每个副本先自己走几百步再同步一次，专门为跨数据中心训练设计。</li>
  </ul>

  <h3>2.5　这一刀留下的问题</h3>
  <p>FSDP 每一步搬的是<b>权重</b>，搬多少只跟参数量有关，<b>跟 batch 无关</b>；
    而每一步要算多少，跟每张卡分到的 token 数成正比。</p>
  <p>所以每张卡的 batch 一小，算得少、搬得一样多，时间就被搬权重吃掉了。
    要继续加卡、又不能让每张卡的 batch 变小，就得换一种切法：<b>不再拼整层权重，直接把权重切开</b>。
    这样几张卡合起来算同一批 token，每张卡要处理的 token 数不会随加卡被摊薄。这是第二刀。</p>
</div></section>

''' + sec("s三", "三", "第二刀：切权重") + '''
  <p class="lead">FSDP 每一层都要把整层权重拼回来。只要每张卡的 batch 够大，这笔搬运能藏在计算后面；
    <b>batch 一小，就藏不住了。</b>第二刀不再拼权重，直接把权重切开。</p>

  <h3>3.1　FSDP 的尽头：搬一个字节，换来多少计算</h3>
  <p>想象砌墙：师傅砌一块砖两秒，小工送一块砖要五秒，师傅再快也得等。每张卡就是师傅，网络就是小工。
    所以判断一刀会不会被通信拖住，只看一个比值：<b>每在网络上搬一个字节，能换来多少次计算</b>。
    这个比值要高过硬件自己的比值（每秒能算多少次 ÷ 每秒能搬多少字节），否则师傅就在干等。</p>
  <p>FSDP 这笔账很干净（⚠️ 推导，按稠密模型算）：一步搬的是两次拼权重、一次分梯度，约 6Ψ 字节（Ψ 还是参数个数，bf16 每个 2 字节）；
    一步算的是 6ΨT 次（专题四那条「训练每个 token 约 6 倍参数量次运算」），T 是每张卡分到的 token 数。<b>两者一除，正好等于 T</b>，跟模型多大没关系。</p>
__FIG_INTENSITY__
  <p>TPU v7 每芯片每秒能算 2,307 T 次（bf16）。芯片之间的 ICI 每颗芯片每秒能发出约 0.6 TB
    （常说的 1.2 TB/s 是收发两个方向加起来的；上面搬的字节是「每卡发出」的量，只能跟发出那一半比，推算见台账）。
    硬件的比值约 3,845。所以 <b>每张卡每步少于约 3,845 个 token，FSDP 就被搬权重拖住了</b>。
    而加卡时总 batch 往往不能跟着涨，每张卡分到的只会越来越少。</p>
  <p>这还是最乐观的算法。两种情况门槛更高：只走一根轴，高约 3 倍；MoE 只靠 FSDP、不配专家并行，
    搬的是全部参数、算的只有被选中的那一小部分，还要再乘约 18 倍（V3 的 6,710 亿 ÷ 370 亿）。</p>

  <h3>3.2　TP：切进矩阵内部</h3>
  <p>张量并行把一层的权重矩阵本身切开，每张卡只存、只算其中一块。Megatron-LM 的切法里藏着一个巧思：</p>
__FIG_TP_MLP__
<figure class="fbox fwide" id="anim-tpsplit">
<video src="media/topic05-tpsplit.mp4" autoplay loop muted playsinline
       aria-label="张量并行切一个 MLP 的动画。两张卡，卡 0 蓝、卡 1 橙，各有完整的 X（灰）；W1 画成宽矩阵竖切一刀、W2 画成高矩阵横切一刀，每张卡只亮自己那一半。标题：张量并行切一个 MLP：Y ＝ GeLU(X·W1)·W2。字幕一：① W1 按列切：每张卡算出中间结果的一半。字幕二：② 激活函数逐元素算：各算各的，这一段没有任何通信。字幕三：③ W2 按行切：每张卡只得到 Y 的一个部分和。字幕四：④ AllReduce：两份部分和相加，两张卡都拿到完整的 Y（中间一个绿框标 AllReduce，Y 是灰色的完整副本）。字幕五：整个 MLP 只在最后通信一次 —— 代价是每一层都有这一次。最后复位。"></video>
<figcaption>通信只在最后那一下；可每一层都有这一下。
  <span class="sub">（9 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>关键在切的方向。按列切第一块，每张卡手里是中间结果的某几列；激活函数只看单个元素，不需要别人的那几列；
    按行切第二块，正好只用到自己这几列。所以通信被整个推到了出口。
    代价也在这儿：<b>这一下每层都要来</b>。一层 Transformer 里 attention 和 MLP 各一下，所以前向两下、反向两下。</p>

  <h3>3.3　SP：TP 的搭档</h3>
  <p>TP 切不到的那几段，是逐个 token 做的小运算（归一化、dropout、残差相加），每张卡都存着一整份激活。
    Megatron 的序列并行（SP）把这几段沿序列切开（名字里有「序列」，但它是 TP 的搭档，跟第四刀切序列不是一回事，见 8.8），顺手把每次 AllReduce 拆成 AllGather ＋ ReduceScatter
    ——&nbsp;就是第一节那个等式，<b>通信量跟原来一样</b>，省下的是激活显存。
    （为了连这份激活也只存一段，反向时要再拼一次，但可以跟算权重梯度叠在一起藏掉。）</p>

  <h3>3.4　TP 的上限：跟 batch 无关</h3>
  <p>TP 搬的是激活，一层搬多少跟 token 数成正比，算多少也跟 token 数成正比，<b>batch 在账里约掉了</b>。
    剩下的只有隐藏维和 TP 度数：每字节换来的计算约 4.5 × 隐藏维 ÷ TP 度数（⚠️ 推导，稠密层）。</p>
  <ul>
    <li>V3 的隐藏维 7,168：按最乐观的硬件线，TP 8 路约 4,032，刚好贴线（7,168 × 4.5 ÷ 3,845 ≈ 8.4；4.5 这个系数按标准注意力加 4 倍宽 MLP 推出，V3 实际是 MLA 加 MoE，这里只作示意）。
      可 3,845 假设 TP 组占满三根带环回的轴，TP 8 路只有 4 颗芯片（v7 一颗芯片算 2 个 device）、用不满，实际门槛要高好几倍，已经在线下了；
      再加上 Megatron 的 TP 通信默认在关键路径上藏不住，实际更紧。</li>
    <li>另外两条约束：注意力头要按整个分，Megatron 要求查询头数能被 TP 度数整除（KV 头更少的模型，KV 头数和 TP 只要一个能整除另一个，TP 更大时 KV 就复制）；TP 每层都要通信，<b>只能待在最快的那一圈互联里</b>。</li>
  </ul>
  <p>所以 batch 小的时候多用 TP，batch 大的时候多用 FSDP。两把刀各管一边。</p>

  <h3>3.5　PP：按层切</h3>
  <p>TP 每一层都要通信，出不了最快的那一圈互联。要横跨很多台机器、走慢线，又不想像 FSDP 那样每层搬权重，就按层切。
    流水线并行把模型按层切成几段，每张卡负责一段，段与段之间只在边界上点对点传激活，
    是所有刀里通信最少的，所以它<b>能跨到慢线上</b>，横跨很多台机器也吃得消。代价是<b>气泡</b>：</p>
__FIG_PP_BUBBLE__
<figure class="fbox fwide" id="anim-pipeline">
<video src="media/topic05-pipeline.mp4" autoplay loop muted playsinline
       aria-label="流水线并行时间表的动画。四个 stage，时间轴一格一格长出来，蓝色是前向、绿色是反向、深灰是空等的气泡。标题：流水线并行：灰色是气泡，每个 stage 都在空等的时间。字幕一：先用 4 个 micro-batch：气泡 ÷ 理想计算时间 ＝ (4−1) ÷ 4 ＝ 3/4。字幕二：总量不变、切成 8 份（每份的格子窄一半）：气泡 ＝ (4−1) ÷ 8 ＝ 3/8，灰色真的短了一半。字幕三：气泡只能摊薄、不能消灭：份数越多越省，可每份太小，卡就吃不饱。最后复位。"></video>
<figcaption>micro-batch 从 4 个加到 8 个，灰色的气泡跟着缩一半。
  <span class="sub">（14 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>开头要等后面几段灌满，结尾要等前面几段排空。气泡跟理想计算时间之比是 (p−1) ÷ m，
    p 是段数，m 是 micro-batch 个数。micro-batch 越多越省；可在最早的 GPipe 调度里，每一份的激活都要攒着等反向，份数越多攒得越多
    （1F1B 调度把同时在路上的份数限制在 p 以内）。
    三种常见的改进：VPP 让每张卡负责几段不连续的层，气泡再除以每卡的段数；
    Zero Bubble 把反向拆成「算输入的梯度」和「算权重的梯度」，后者不急，挪去填空；
    DeepSeek-V3 用的 DualPipe 从流水线两头同时往里灌，更要紧的是把一对前向和反向的计算，跟专家并行的通信叠在一起藏掉（技术报告 §3.2.1）。</p>

  <h3>3.6　这一刀留下的问题</h3>
  <p>DeepSeek-V3 训练时<b>一点 TP 都没用</b>：16 路 PP、64 路专家并行、ZeRO-1 数据并行（技术报告 §3.2）。
    报告自己的解释是显存优化做得够细，用不着代价高的 TP。模型已经被 PP 和专家并行切开，每张卡上那一份放得下，数据并行那一维 ZeRO-1 就够了。
    再从参数的分布看（本课解读）：每个专家 3 × 7,168 × 2,048 ≈ 4,400 万参数，256 个专家 × 58 个 MoE 层 ≈ 6,539 亿，
    <b>占全部参数的约 97%</b>。专家本来就窄，再往里切，每份更小，要搬的激活却一点不少；真正该切的，是「专家」这一维。这是第三刀。</p>
</div></section>

''' + sec("s四", "四", "第三刀：切专家") + '''
  <p class="lead">MoE 模型的参数几乎全在专家里。第三刀就切这一维：把不同的专家放到不同的卡上。
    <b>它多出来的通信只有一种，AllToAll；可它也带来了一个别的刀里最严重的病。</b></p>

  <h3>4.1　为什么该切专家</h3>
__FIG_MOE_PARAMS__
  <p>换句话说，TP 那把刀要切的是「一个很宽的矩阵」，而 V3 里真正占地方的是「很多个窄矩阵」。
    对后者，最省事的切法不是把每一个都劈开，而是<b>把它们整个分给不同的卡</b>。</p>

  <h3>4.2　EP：token 飞去专家那里</h3>
  <p>专家并行（EP）的通信只在 MoE 层里发生，每层两次 AllToAll：
    <b>派发</b>，把每个 token 送到它选中的专家所在的卡；<b>合并</b>，算完再送回原来的卡。</p>
<figure class="fbox fwide" id="anim-ep">
<video src="media/topic05-ep.mp4" autoplay loop muted playsinline
       aria-label="专家并行的动画。四张卡，每张卡上方 4 个 token（颜色表示来自哪张卡），下方 2 个专家，共 8 个专家。标题：专家并行：token 飞到专家那里，算完再飞回来。字幕一：每个 token 由路由挑一个专家（真实的 V3 每个 token 挑 8 个）。字幕二：派发（AllToAll）：token 飞到专家所在的卡，在专家门口排队。专家 0 门口排了 7 个，其他专家 1 到 2 个。字幕三：专家 0 排了 7 个，别的专家只有 1 到 2 个：它算完之前，大家都得等。字幕四：合并（AllToAll）：算完再送回原来的卡。字幕五：发给谁由数据决定，负载天生不均 —— 这是专家并行最重的病。最后 token 回到原位。"></video>
<figcaption>派发、排队、合并。那一根排得最高的队，决定了所有卡什么时候能往下走。
  <span class="sub">（10 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>V3 为了压住这两次 AllToAll，做了两件事（技术报告 §2.1.2、§3.2.2、§3.3.3）：</p>
  <ul>
    <li><b>限制跨节点</b>：每个 token 最多发往 4 个节点。先走节点间网络发到目标节点，再走节点内的 NVLink 转给真正持有专家的卡。
      同一台机器上就算有好几个它要找的专家，跨机也只发一份，到了再在机器里分。</li>
    <li><b>派发用 FP8，合并用 BF16</b>：派发那一趟的字节数直接减半。</li>
  </ul>
  <p>按这个算（⚠️ 推导）：每个 token 跨节点派发最多 4 份 × 7,168 字节（隐藏维 7,168 个数，FP8 每个 1 字节）≈ 28.7 KB，跟它选了几个专家无关，只跟去了几个节点有关。</p>
  <p>这是训练和 prefill 的做法；decode 追求低延迟时改成按专家逐个直发，份数就跟着选的专家数涨了。</p>

  <h3>4.3　EP 最重的病：负载由数据决定</h3>
  <p>别的刀负载均不均，大多事先就知道。<b>EP 的不均最严重</b>：
    哪个专家忙、哪个专家闲，要等路由算完才知道，而且每一批数据都不一样。
    最忙的那个专家算完之前，所有人都得等它。治法分两头：</p>
  <ul>
    <li><b>训练时</b>：让路由本身尽量均匀。以前靠在训练目标里加一项「辅助损失」罚不均，但会拖累模型效果。
      V3 改用偏置：哪个专家负载高就把它的偏置调低，低了就调高；偏置只影响挑谁，不改算出来的权重。辅助损失只留一个极小的兜底。</li>
    <li><b>推理时</b>：把热门专家多复制几份，摊到不同的卡上，也就是 EPLB 和冗余专家。</li>
  </ul>

  <h3>4.4　attention 和专家，各配各的</h3>
  <p>一层 Transformer 里，attention 和专家是两种完全不同的形状：attention 的负担跟序列和 KV 有关，
    专家的负担是那一大堆参数。<b>所以同一批卡，在这两部分可以用两套切法。</b></p>
__FIG_FOLD__
  <p>推理那边的简称：TEP 是 attention 用 TP、专家用 EP；DEP 是 attention 用数据并行、专家用 EP。
    挑哪个差别大到什么程度，我们自己测过一次：</p>
  <div class="note ok"><span class="t">一次实测：换一种切法，每张卡的吞吐翻一倍（跟调完参的 TP4 比；卡数也变了，见 7.3）</span>
    GB300 上跑 DeepSeek-V4-Pro（vLLM），decode 从 TP4 换成 DEP8（attention 数据并行 8 路、专家 EP8），同样并发下<b>每张卡的吞吐是调完参的 TP4 的 2.09 倍</b>。
    其中最属于「换切法」的一笔在 attention 那一半：V4-Pro 的 KV 只有一个头，TP 切不开，只能在 4 张卡上各复制一份；改成数据并行后，每张卡只存自己那批请求的 KV。
    attention 权重虽然每张卡要存一份，但在 MoE 模型里只占几个百分点。<br>
    <em>完整的账（卡数、并发、首字延迟）在 7.3。KV 这件事，下一刀专门讲。</em></div>

  <h3>4.5　这一刀留下的问题</h3>
  <p>前三刀都没碰过「序列」这一维。可上下文一长，训练时的激活、推理时的 KV cache，都跟着序列长度往上涨。
    <b>一条样本本身就放不进一张卡了</b>，只能把它切开。这是第四刀。</p>
</div></section>

''' + sec("s五", "五", "第四刀：切序列") + '''
  <p class="lead">前三刀切的是 batch、权重、专家，从没碰过「一条样本」本身。
    <b>上下文一长，一条样本自己就放不进一张卡了。</b>第四刀沿序列切。训练和推理切的东西不一样，分开讲。</p>

  <h3>5.1　一条样本为什么会放不下</h3>
  <ul>
    <li><b>训练</b>：每层要存的激活至少跟序列长度成正比；不用 FlashAttention 时还有一项跟长度的平方成正比（Megatron 序列并行论文的式 1）。</li>
    <li><b>推理</b>：KV cache 每个 token 都要存一份。V3 的 MLA 每 token 存一份压缩后的 KV（512 维）加一小段位置信息（64 维），(512 ＋ 64) × 61 层 × 2 字节 ＝ 70,272 字节，
      <b>一个 128K 的请求就是约 8.58 GiB</b>。</li>
  </ul>

  <h3>5.2　训练：CP 把激活沿序列切开</h3>
  <p>上下文并行（CP）把一条长序列切成几段，每张卡只存自己那段的激活。
    难点只在注意力：每个 token 要看到它前面所有的 token，而那些 token 在别的卡上。两种办法：</p>
  <ul>
    <li><b>Ring Attention</b>：Q 不动，KV 沿环一段段传，每一步算手上这一对，同时把 KV 传给下一张。</li>
    <li><b>Ulysses</b>：注意力前后各做一次 AllToAll，在「按序列切」和「按头切」之间来回换。
      换成按头切以后，每张卡手里是全部 token、但只有几个头，注意力就能在本卡算完，算完再换回去。
      每张卡的通信量在序列长度和卡数同比放大时保持不变；代价是并行度不能超过注意力头数（有分组的模型卡在 KV 头数上）。</li>
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
  <p>两者可以叠起来用（USP：一个方向走环，一个方向走 AllToAll），Megatron 的 CP 也支持分层组合。</p>

  <h3>5.3　causal 带来的不均</h3>
  <p>生成式模型的注意力有因果掩码：每个 token 只看前面的。于是越靠后的段算得越多，顺序切会让最后一张卡累死。</p>
__FIG_CP_ZIGZAG__

  <h3>5.4　推理：KV cache 被 TP 复制了</h3>
  <p>推理时 KV 常常是最大的一块。而 TP 一旦超过 KV 头数 H，KV 就切不开了：每个头被复制 TP ÷ H 次（vLLM 文档原话：duplicated tp_size / H times）。
    MLA 模型只有一个 KV 头，TP 8 路就是每张卡一整份、8 份一模一样的 KV。</p>
__FIG_KV_DUP__
  <p><b>DCP（decode 上下文并行）</b>让 KV 按 token 轮流存到几张卡上，用的还是原来那几张卡：</p>
<figure class="fbox fwide" id="anim-dcp">
<video src="media/topic05-decodecp.mp4" autoplay loop muted playsinline
       aria-label="DCP 动画。四张卡。标题：DCP：decode 时 KV 按 token 轮流存到各张卡。字幕一：每生成一个 token，它的 KV 存到第 (token 号 mod 4) 张卡上。token 0 到 11 依次落到卡 0、1、2、3 轮转。字幕二：12 个 token，每张卡只存 3 个的 KV：容量是原来的 4 倍。字幕三：算注意力：新 token 的 Q 发给所有卡，各自在自己那份 KV 上算。字幕四：四份部分结果合并成一份：每层多几次通信，换回 4 倍的 KV 空间。最后复位。"></video>
<figcaption>KV 轮流落到四张卡上；每一层多几次通信，把 Q 收齐、把结果合起来。
  <span class="sub">（9 秒无声循环，Manim 渲染。）</span></figcaption></figure>
  <p>算注意力时，新 token 的 Q 发给所有卡，各自在自己那份 KV 上算，再把几份部分结果合并（每份带着自己的归一化分母，记作 LSE，才能合得对）。
    每层多付三次通信（vLLM 默认实现：收齐 Q、交换归一化分母、合并输出），换回被 TP 白白复制掉的那几份 KV。
    DCP 的度数最多开到 TP ÷ KV 头数，也就是刚好把复制的那几份收回来。<b>用一种通信，换一份显存</b>，又一次。</p>

  <h3>5.5　prefill 那边也有一种</h3>
  <p>DCP 管的是 decode。prefill 那边要把一个长 prompt 切开、让第一个字早点出来，叫 <b>PCP</b>（prefill 上下文并行）。
    它是在 TP 之外再加一维，所以<b>PCP 加卡，DCP 不加卡</b>；它只管算得快，不管 KV 装不装得下。
    下一节把 prefill 和 decode 拆到两批机器上以后，它们正好可以一边一个（本课的归纳）。</p>

  <h3>5.6　这一刀留下的问题</h3>
  <p>说到这儿，prefill 和 decode 已经各要各的切法了：一个吃算力、要把 prompt 切开；一个吃带宽、要把 KV 摊开。
    <b>硬塞在同一批卡上，谁都配不好。</b>那就干脆别切张量了，把这两种活拆到不同的机器上。这是第五刀。</p>
</div></section>

''' + sec("s六", "六", "第五刀：不切张量，切工作") + '''
  <p class="lead">前四刀切的都是张量：batch、权重、专家、序列。第五刀换个思路 ——&nbsp;
    <b>一个请求里本来就有性质完全不同的几段活，把它们拆到不同的机器上</b>，每一边再挑自己的切法。</p>

  <h3>6.1　prefill 和 decode 为什么会打架</h3>
  <p>一个请求分两段。<b>prefill</b> 一口气吞下整个 prompt，几千个 token 一起过矩阵乘，吃的是算力；
    <b>decode</b> 每一步只出一个 token，却要把全部权重和 KV 从显存里读一遍，吃的是带宽。</p>
  <p>它们挤在同一批卡上时，引擎每一步都要决定先干哪个。长 prompt 一来，它的 prefill 要占好几步，
    这几步里<b>所有正在出字的请求都得等</b>：新请求的首字延迟（TTFT）和老请求的出字间隔（TPOT）一起变差。</p>
  <p>decode 还有个脾气：它每一步都要把权重读一遍，靠把很多请求拼成一大批来摊薄这笔成本，batch 越大越划算。后面 DEP、AFD、Wide-EP 这些做法，说到底都是在想办法把 decode 的 batch 做大。</p>
__FIG_PD__
<figure class="fbox fwide" id="anim-pd">
<video src="media/topic05-pd.mp4" autoplay loop muted playsinline
       aria-label="PD 分离动画。标题：PD 分离：prefill 和 decode 拆到两批机器上。三条车道：放在一起、prefill 机器、decode 机器。字幕一：放在一起：大家一步一步 decode，每格出一个字。上面一条车道绿格一格一格长出来，第 5 步落下一个长橙块「新请求的 prefill」，下面标红「这 5 步没人出字」，之后绿格继续。字幕二：拆开：prefill 在自己的机器上跑，decode 那边一步不停。decode 车道 16 个绿格连续长出，prefill 车道同时跑完橙块，一支蓝色箭头「KV 传过去」落到 decode 车道。字幕三：放在一起：decode 被截走 5 步；拆开：一格不断（多用了一批 prefill 机器）。字幕四：代价：多一趟 KV 传输；按我们 v7x 那套的带宽估算约 100 毫秒。最后复位。"></video>
<figcaption>同样 16 步，上面那条被长 prefill 截走了 5 步；下面那条一格没少。
  <span class="sub">（11 秒无声循环，Manim 渲染。格数是示意，不是实测时序。）</span></figcaption></figure>
  <p>分块 prefill（chunked prefill）能缓解：把长 prompt 切成小块，每一步跟 decode 拼着跑。
    但两种活还在抢同一批卡，而且<b>只能用同一套并行方式</b>。PD 分离干脆把它们拆开：
    prefill 机器只做 prefill，算完把 KV cache 交给 decode 机器，decode 机器只管出字。</p>

  <h3>6.2　代价：一趟 KV 传输</h3>
  <p>我们在 TPU v7x 上搭过一套 1P1D（Qwen3-Coder-480B，一台 v7x-8 做 prefill、一台做 decode、再加一个 CPU 上的转发代理），
    KV 从 prefill 那台的显存走到 decode 那台的显存。结论先说：按带宽估算一趟约 100 ms，占一次 prefill 的 5–10%，<b>网络不是瓶颈</b>。细账是三段：</p>
  <table>
    <tr><th>段</th><th>路径</th><th>估算</th></tr>
    <tr><td>①</td><td>HBM → 本机内存（PCIe）</td><td>约 10 ms</td></tr>
    <tr><td>②</td><td>本机内存 → 对方内存（数据中心网络，按单颗芯片的 100 Gbps 份额保守估）</td><td>约 80 ms</td></tr>
    <tr><td>③</td><td>对方内存 → HBM（PCIe），接进 decode 的 KV 池</td><td>约 10 ms</td></tr>
    <tr><td></td><td><b>合计</b></td><td><b>约 100 ms</b></td></tr>
  </table>
  <p>对一下账：8K prompt 的 KV 是 2 × 62 层 × 8 个 KV 头 × 128 × 8,192 × 1 字节（FP8）≈ 1.04 GB，
    100 Gbps 就是每秒 12.5 GB，走一趟约 83 ms，中间那段就是这么来的。
    这个模型 KV 头少（8 个）又用 FP8 存，KV 本来就小；换模型、换网络，要重算。</p>
  <p>反过来说，请求都很短、量也不大的时候，拆开多出来的这趟传输和两套机器就不一定划算，放在一起、用分块 prefill 缓解就够了。</p>

  <h3>6.3　两边各配几台</h3>
  <p>拆开之后多了一个旋钮：prefill 和 decode 的机器配比。思路是让两边差不多同时忙满：
    先量一台 prefill 机器每秒能吞多少 prompt token、一台 decode 机器（一整批请求一起跑）每秒能吐多少 token，
    再按业务里输入和输出的长度比去配。经验上<b>长 prompt 的业务配 2P:1D，长输出的业务配 1P:2D</b>。
    DistServe 论文把配比和各自的并行方式一起搜，跟 vLLM 等不拆开的系统比，同样的延迟要求下最多能多服务 7.4 倍的请求，或者把延迟要求收紧最多 12.6 倍（OPT 系列模型，以 90% 请求达标为准）。</p>

  <h3>6.4　两边各挑各的切法</h3>
  <p>这才是拆开的真正收益：<b>两边不再被迫用同一套并行方式</b>。</p>
  <ul>
    <li><b>prefill 机器</b>：可以上 PCP，把一个长 prompt 切到几张卡上一起算，第一个字出得快（5.5）。</li>
    <li><b>decode 机器</b>：可以上 DCP，把 KV 摊到几张卡上装更多请求（5.4）；还可以把专家铺到更多卡上（Wide-EP），每个专家分到的 batch 更大。</li>
    <li><b>MoE 模型</b>：常见的写法是一边 TEP（attention 用 TP）、一边 DEP（attention 用数据并行），但<b>哪边用哪个没有定式</b>，要看模型和负载（8.6）。</li>
  </ul>

  <h3>6.5　AFD：attention 和专家分到两组机器</h3>
  <p>decode 这边还能再拆。attention 要读每个请求自己的 KV，跟请求绑定；专家不管 token 来自谁，只要 batch 够大。
    <b>AFD</b>（Attention-FFN 分离）把两者放到两组机器上：M 台只算 attention，N 台只放专家。</p>
__FIG_AFD__
  <p>每一层都要把 token 从 attention 那边发给专家（M → N），算完再收回来（N → M）。
    为了不让这一来一回拖慢，把一批请求切成几个小批轮着跑：这一个在算 attention，另一个正好在算专家，还有一个在路上。
    MegaScale-Infer 论文算过，要藏住通信至少得三个小批，通信慢的时候要四个。
    字节的 MegaScale-Infer 报告每 GPU 吞吐最高提升 1.90 倍；阶跃的 Step-3 也是这个路子；
    vLLM 在 2026 年 7 月出了实验性插件。</p>

  <h3>6.6　多模态：编码器也单独放</h3>
  <p>同样的思路还能往前推一段：多模态模型的视觉编码器单独部署，算好的 embedding 再传给语言模型（Encoder 分离，
    跟 PD 连起来常写作 EPD）。训练里对应的是多模块异构并行：编码器和语言模型各用一套并行方式。</p>

  <h3>6.7　这一刀留下的问题</h3>
  <p>五刀讲完了，每一刀都有自己的通信和适用场景。真到一个集群上，它们要同时存在：
    谁放在同一台机器里、谁跨机器、先定哪一刀的度数？<b>摆错了位置，前面每一刀省下来的都会被网络吃回去。</b>这是第七节。</p>
</div></section>

''' + sec("s七", "七", "摆到机器上") + '''
  <p class="lead">五刀都讲完了，真到一个集群上它们是同时存在的：一套配置里常常有三四种并行叠在一起。
    剩下的问题只有一个：<b>哪一刀放在哪根线上</b>。摆对了，前面每一刀省下的都是真省；摆错了，全被网络吃回去。</p>

  <h3>7.1　线有快有慢，刀有勤有懒</h3>
  <p>一个集群里的线不是一样快的。GPU 这边，GB300 NVL72 把 72 块卡连成一个 NVLink 域，域里每块卡 1.8 TB/s（收发合计，下同）；
    出了这个域只能走网卡，每块卡 200 GB/s，差 9 倍。TPU 这边，一个切片里的芯片走 ICI（够大的切片连成 3D 环面，见 1.5），
    跨切片走数据中心网络，同口径比慢约 50 倍，比 GPU 那边的 9 倍悬殊得多。</p>
  <p>刀也不是一样勤的。把每一刀一步要通信几次数一遍（本课推导，示意配置），差出三个数量级：</p>
__FIG_FREQ__
  <p>于是有一条默认的摆法：<b>每一层都要说话的 TP、EP、FSDP、CP 先往最快的那一圈里放</b>；PP 只在段边界说话，DP 一步只说一次，它们去跨慢线。
    但光数次数不够，还要看<b>它能不能跟计算叠起来</b>。TP 叠不起来，所以必须待在快线里，度数上限就是那一圈的大小。
    EP 和 FSDP 能靠提前发、边算边传藏住一部分，就有人让它们跨出去：V3 的 EP 64 就横跨 8 台机器
    （V3 报告里 H800 机内 NVLink 实际约 160 GB/s、机间 IB 50 GB/s，只差约 3 倍，技术报告 §3.2.2；按规格同口径本课推算约 4 倍），Llama 3 把 FSDP 放在了最外层。
    （TP 叠不起来，是因为每一块末尾那次 AllReduce 不做完，下一块就没法开始。）
    DP 那一次量虽然最大，但一步只有一次，而且反向从最后一层往前算，后面几层的梯度一算好就能先传。</p>

  <h3>7.2　摆错一次是什么样</h3>
  <p>同样 8 张卡、同样 TP4 × DP2，只改 TP 组放在哪。下面的时间是示意：快线传一次 1 格，慢线 9 格。</p>
<figure class="fbox fwide" id="anim-meshmap">
<video src="media/topic05-meshmap.mp4" autoplay loop muted playsinline
       aria-label="TP 组摆放动画。标题：同样 8 张卡、TP4 × DP2：TP 组摆在哪，差好几倍。两台机器各 4 张卡，中间一条虚线标着「慢线：机器之间」。字幕一：摆法一：一个 TP 组就在一台机器里，DP 才过慢线。机器 0 的 4 张卡涂蓝、机器 1 的涂橙，白点在机器内沿环传 8 轮，最后绿点过慢线一次，左下角计时：摆法一：这一步用了 17 格。字幕二：摆法二：TP 组横跨两台机器，每一轮都要过慢线。每台机器上下两排分属两个组，白点每一轮都要穿过慢线，右下角计时一格一格跳到 73 格，标红。字幕三：17 格对 73 格：同样的卡，慢 4.3 倍（示意）。最后复位。"></video>
<figcaption>一轮通信要等最慢的那一段传完，所以摆法二每一轮都按慢线算。
  <span class="sub">（15 秒无声循环，Manim 渲染。一步 8 轮 TP、1 次 DP，快慢 1 : 9 是示意。）</span></figcaption></figure>
  <p>放到真实集群上，这就是为什么 TP 一般不出一台 8 卡机器。GB300 把 NVLink 域扩到一整柜，
    专家并行才敢往大了开（TP 受头数和矩阵效率限制，在整柜上也很少超过 8 或 16）：<b>快线那一圈画多大，这几刀就能切多深。</b></p>

  <h3>7.3　先选对切法，再调参数</h3>
  <p>摆法和切法选错了，参数调得再细也只是在错的天花板下面打转。我们在 GB300 上跑 DeepSeek-V4-Pro 时撞上过一次：</p>
__FIG_TOPO__
  <p>TP4 decode 上能调的都调了：去掉 eager 模式只多 2.6%，加 prefill 机器、调并发，总数从 14,563 涨到 21,100。
    可这 45% 是拿多一倍的卡换来的，出字间隔始终钉在 46.8–53 ms。换成 DEP8 那一步，同样并发 512 下出字间隔从 46.8 ms 降到 11.8 ms，首字延迟从 55.8 秒降到 22.8 秒。
    真正属于「换切法」的那笔账，是 KV 不再在 4 张卡上各存一份（切序列那一节讲的毛病）；decode 从 4 张卡加到 8 张，也把专家摊薄了一半。两笔都换成了更大的 batch。
    （出字间隔为什么同时降下来，原始记录里没有拆开归因。）<b>先问切法对不对，再动参数。</b></p>
  <p><em>口径提醒：图里后两行都是并发 512，第一行是原始脚本的并发 256。DEP8 把并发拉到 1,536 总量能到 65,132（每卡 2.47 倍），但那时首字要等 95 秒，prefill 又成了瓶颈。
    这里的吞吐是 prompt 和输出 token 加在一起算的。</em></p>

  <h3>7.4　五步怎么选</h3>
  <p>把前面几节串起来，给一个模型挑并行方式，大致按这个顺序：</p>
  <ol>
    <li><b>先装得下</b>：权重和优化器状态摊不开就上 FSDP ／ ZeRO（第二节）；MoE 的专家太多就上 EP（第四节）；
      推理时 KV 放不下就上 DCP（第五节）。</li>
    <li><b>再看快线那一圈有多大</b>：TP、EP 的度数别超过它（8 卡一台的机器就是 8，GB300 一柜是 72，TPU 看切片）。</li>
    <li><b>再看序列多长</b>：训练长上下文加 CP，推理长 prompt 加 PCP。</li>
    <li><b>还不够，或者必须跨很慢的线，才上 PP</b>：它只传段边界的激活，量小、频率低，能跨慢线，代价是气泡（第三节）。</li>
    <li><b>剩下的卡全给 DP</b>：一步只通信一次，最便宜。</li>
  </ol>
  <p>拿两个真实配置对一遍。表里「DP 4 × FSDP 128」这类写法按乘法读：两种切法各切几份，乘起来就是总共多少份。</p>
  <table>
    <tr><th>步骤</th><th>混元 3（295B MoE，TPU v7，256 芯片）</th><th>DeepSeek-V3（671B MoE，2,048 块 H800）</th></tr>
    <tr><td>① 装得下</td><td>FSDP 128（专家权重也由 FSDP 切）：再窄就爆显存</td><td>ZeRO-1 摊优化器状态；EP 64 摊专家</td></tr>
    <tr><td>② 快线多大</td><td>切片内全是 ICI；但在 16 芯片上试 EP 4 路，吞吐反而掉 71%（那次 batch 减半，有混杂；换一种配法是掉 37%），<b>不用 EP</b>。原文归因于环面上 all-to-all 要多跳，但 4 路时这笔账其实不大，原因还要再查</td>
      <td>8 卡一台，<b>不用 TP</b>；EP 64 横跨 8 台机器，跨机那段是慢线，是个例外。为了少走慢线，规定每个 token 最多发到 4 台机器</td></tr>
    <tr><td>③ 序列</td><td>4K ／ 8K，不用 CP</td><td>预训练 4K，报告里没有 CP</td></tr>
    <tr><td>④ PP</td><td>一个切片放得下，不用</td><td>PP 16，用 DualPipe 把通信叠进计算</td></tr>
    <tr><td>⑤ DP</td><td>剩下的 4 倍全给 DP：DP 4 × FSDP 128，这是默认配方（v7 一颗芯片算 2 个 device，并行度按 device 数：256 芯片 ＝ 512 个）。
      最好成绩是把 FSDP 加宽到 256、用省下的显存把每卡 batch 从 12 推到 16，比默认配方高 3%；batch 不变只加宽 FSDP 会变慢（见 7.5）</td><td>剩下的给 ZeRO-1 的数据并行</td></tr>
  </table>
  <p>同一套五步，两个模型走出来的配置几乎没有重合，<b>因为两台机器的快线长得不一样</b>。
    这也是为什么别人家的并行配置不能照抄：先看自己的线。</p>

  <h3>7.5　怎么评：加卡之后掉了多少</h3>
  <p>配好之后要回答一个问题：卡加上去，每张卡的效率还剩多少。有两种量法：</p>
  <ul>
    <li><b>strong scaling</b>：总活不变，卡加倍，看时间能不能减半。每张卡分到的活越来越少，固定开销的占比越来越大，
      迟早撞墙。</li>
    <li><b>weak scaling</b>：每张卡的活不变，卡和总活一起加倍，看每张卡的速度掉不掉（Gustafson 定律的视角）。
      训练大模型常常先走这一种：卡多了，global batch 也跟着加。但 global batch 有收敛允许的上限（临界 batch size），到了上限就只能 strong scaling。</li>
  </ul>
  <p>下面是我们自己的两组实测。左右两组的每卡 batch 不同，<b>只能组内比</b>。右边那组每卡活不变，变的只是 FSDP 组的大小。</p>
__FIG_SCALE__
  <p>左边那组说明 DP 方向几乎是白送的：4 个 64 芯片的组之间，一步只有一次梯度 all-reduce，
    按 v7 的 ICI 粗算，就算几组共用链路、打个折扣，也占不到一步 23.5 秒的百分之一（本课推算）。
    右边那组才是要小心的：卡没变、batch 没变，只是把 FSDP 从四分之一的芯片铺到全部芯片，每拼一次权重要走的步数多了、每块数据更小，延迟摊不掉，就少了 11%。FSDP 再窄（64 份、32 份）每份权重太大，加上激活就放不下了。
    <b>所以在 batch 还能加的时候，加卡要连 batch 一起加，多出来的卡优先当副本。</b></p>

  <h3>7.6　这一节留下的问题</h3>
  <p>到这里，每一种并行都有了来处：它切什么、多出什么通信、该放在哪根线上。
    下一节把它们摊在一张全景图和几张表上，那些 TEP8、DEP16、DP4 × FSDP128 的写法，就都读得懂了。</p>
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
      <td>DP 每张卡存一份完整状态，是纯冗余</td><td>reduce-scatter ＋ all-gather；ZeRO-3 每层都要把参数 all-gather 回来</td><td>训</td></tr>
    <tr><td>FSDP / FSDP2</td><td>就是 ZeRO-3，PyTorch 原生版本</td>
      <td>通信量是 DP 的 1.5 倍，显存却随卡数线性下降</td><td>前向 all-gather 权重，反向再 all-gather 一次、再 reduce-scatter 梯度</td><td>训</td></tr>
    <tr><td>HSDP</td><td>机内分片，机间复制</td><td>把最频繁的 all-gather 圈在高带宽域里</td>
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
    <tr><td>Megatron SP</td><td>只切 LayerNorm、Dropout、残差这几段的激活，<b>必须跟 TP 搭配</b>（按「切什么」归到这一列；按用途它是 TP 的搭档，见 3.3）</td>
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
      <td>每层前向两次、反向两次 all-reduce，<b>频率极高</b>，所以只能待在 NVLink 或 ICI 一跳之内</td><td>训 · 推</td></tr>
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
  </table>

  <h3>8.8　同名不同义</h3>
  <table>
    <tr><th>词</th><th>在不同地方的意思</th></tr>
    <tr><td>SP</td><td>至少三种：Megatron 的 SP（只切 LayerNorm 那几段，跟着 TP）；DeepSpeed 说的 SP（指 Ulysses）；
      以及泛指一切切序列的做法。Megatron 里「所有激活都切」叫 CP</td></tr>
    <tr><td>CP</td><td>训练里指切激活。vLLM 把它拆成两个开关，对卡数的作用相反：PCP 加卡，DCP 不加卡；功能也不同，一个压首字延迟，一个扩 KV 容量</td></tr>
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
    <tr><td>TP 的切法与通信次数；SP 不增通信（第三节）</td><td>Megatron-LM arXiv 1909.08053 §3（前向 2 次、反向 2 次 all-reduce）；arXiv 2205.05198 §4.2.2（AG＋RS 替代 all-reduce，无额外通信）；头数须被 TP 整除：megatron/core/transformer/transformer_config.py 的校验</td></tr>
    <tr><td>PP 气泡 (p−1)/m；交错式除以 v</td><td>Narayanan 等 arXiv 2104.04473 §2.2.1–2.2.2；Zero Bubble arXiv 2401.10241；DualPipe README</td></tr>
    <tr><td>V3 训练并行配置；参数分布</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 §3.2（16 路 PP、64 路 EP、ZeRO-1，不用 TP）；config.json（61 层、前 3 层 dense、256 专家、moe_intermediate_size 2048、hidden 7168）</td></tr>
    <tr><td>每字节换多少计算、v7 硬件线约 3,845</td><td>⚠️ 本课推导（稠密近似、完全重叠）；v7 2,307 TFLOP/s bf16；官方给每芯片 ICI 1,200 GB/s，另给 200 GB/s 一档；把它理解成每条链路收发合计，「6 条链路 × 200、发出方向 600」才对得上，这是推导（按 scaling book 单链路单向 9e10 算约 540，硬件线约 4,270，所以取 3,800–4,300 区间）；按 device 口径同样约 3,845（一颗芯片的两个 device 共用链路，算力和带宽一起减半）（wiki ici-dcn、Inferact 博客规格表）。2026-09-25 更正：旧版误用 1,200 得出 1,922</td></tr>
    <tr><td>V3 的 EP 细节：最多 4 节点、FP8 派发 BF16 合并、无辅助损失的负载均衡</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 §2.1.2、§3.2.2、§3.3.3；每 token 跨节点派发 ≈ 28.7 KB 为本课推导</td></tr>
    <tr><td>Parallel Folding 的例子</td><td>Megatron-Core megatron/core/transformer/moe/README.md；arXiv 2504.14960</td></tr>
    <tr><td>GB300 上 TP4 → DEP8：同并发 512 总量 2.61 倍、每卡 2.09 倍（DEP8 并发 1,536 时每卡 2.47 倍、TTFT 95 s）；调参 +45%</td><td>本课程作者实测：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/README.md 与 VLLM-V4PRO-RUNBOOK.md（TP4 decode 14,563 → 调参后 21,100，16 GPU、每卡 1,319；DEP8 65,132，20 GPU、每卡 3,257 tok/s）</td></tr>
    <tr><td>激活随序列长度增长</td><td>Korthikanti 等 arXiv 2205.05198 式 (1)：每层 sbh(34 ＋ 5as/h)</td></tr>
    <tr><td>Ring Attention；Ulysses 通信量恒定、并行度不超过头数</td><td>arXiv 2310.01889；arXiv 2309.14509 §3.2（4Nh/P，N 与 P 同比放大时不变）；头数上限见 USP arXiv 2405.07719 §3</td></tr>
    <tr><td>CP 的之字形切法</td><td>Megatron-LM megatron/core/utils.py（2×cp 块，rank r 拿第 r 与 2·cp−r−1 块）；docs/user-guide/features/context_parallel.md</td></tr>
    <tr><td>KV 被 TP 复制 tp/H 次；DCP 复用 TP rank</td><td>vLLM context parallel 部署文档；vllm/config/parallel.py docstring。V3 每 token KV 70,272 字节按 config.json 现算</td></tr>
    <tr><td>PD 分离的动机与收益（第六节）</td><td>DistServe arXiv 2401.09670（prefill 偏算力、decode 受带宽约束；7.4 倍请求或 12.6 倍更紧的 SLO）</td></tr>
    <tr><td>v7x 上 1P1D 的 KV 三段约 100 ms（带宽估算）；2P:1D ／ 1P:2D</td><td>本课程作者的部署记录（KV 用时为按带宽估算，非计时）：wiki qwen3-coder-480b-pd-disagg-tpuv7x-20260425。8K KV ≈ 1.04 GB、过 100 Gbps 约 83 ms 为本课推导（Qwen3-Coder config：62 层、8 个 KV 头、head_dim 128）</td></tr>
    <tr><td>每一刀每步的通信次数（第七节）</td><td>⚠️ 本课推导（60 层、8 个 micro-batch 的示意配置）：TP 每层 4 次（arXiv 1909.08053 §3），FSDP 每层 3 次（arXiv 1910.02054 §7），EP ／ PP ／ DP 按调度数出</td></tr>
    <tr><td>GB300 NVLink 1.8 TB/s ／ 每 GPU 800 Gb/s 网卡；9 倍</td><td>wiki sources/nvidia-gpu-comparison-20260311、analyses/gb300-a4x-max-network-congestion-control（A4X Max 每节点 4 GPU、4 × CX-8 800 Gb/s）；9 倍为本课按双向口径换算</td></tr>
    <tr><td>混元 3 的 scaling 与五步对照</td><td>本课程作者实测：gpu-tpu-pedia tpu/Hunyuan3-295B-Pretraining/TUNING-v7 §3.7（五种分法 404 ／ 450 ／ 453 ／ OOM ／ OOM）、§4.1（64 与 256 芯片同为 580）、§3.6（DP2 × FSDP256、pdbs 16 得 599）、EP 4 路在 16 芯片上 −71%（单次）。组间 all-reduce 十几到二十几毫秒为本课推算（原文 12 ms 的算式前后不一致）</td></tr>
    <tr><td>V3 的集群与每 token 最多 4 节点</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 §3.1（2,048 块 H800、节点内 NVLink、节点间 IB）、§2.1.2、§3.2</td></tr>
    <tr><td>strong ／ weak scaling</td><td>Amdahl 1967；Gustafson 1988（Reevaluating Amdahl's Law）；临界 batch size：arXiv 1812.06162</td></tr>
    <tr><td>NVSwitch 在交换机里做加法（NVLS）</td><td>NCCL NVLS 算法（NVIDIA NCCL 文档）；「少将近一半」为按每卡发出约 S 对 2(n−1)/n·S 的本课推算</td></tr>
    <tr><td>TPU 切片何时首尾成环、小切片带宽约减半</td><td>How to Scale Your Model（scaling book）TPU 章节：只有整 cube（4 的倍数）才有环回；Google Cloud TPU7x 拓扑文档</td></tr>
    <tr><td>Llama 3 把 FSDP 放在最外层</td><td>Llama 3 技术报告 arXiv 2407.21783 §3.3.2（并行维度顺序 [TP, CP, PP, DP]）</td></tr>
    <tr><td>V3 每 token 激活 370 亿参数、预训练 4K</td><td>DeepSeek-V3 技术报告 arXiv 2412.19437 摘要与 §4.1</td></tr>
    <tr><td>torchtitan 的「4D」</td><td>pytorch/torchtitan README（FSDP2 ＋ TP ＋ PP ＋ CP）</td></tr>
    <tr><td>「最大的一块卡显存也就两三百 GB」</td><td>NVIDIA B300 每卡 288 GB HBM3e；Google TPU7x 每芯片 192 GB HBM（官方规格）；9.76 TiB ≈ 10,700 GB（十进制）</td></tr>
    <tr><td>DCN 每芯片 100 Gbps；跨切片同口径慢约 50 倍</td><td>Google Cloud TPU7x 文档（第五轮 TPU 专家评审核对）；100 Gbps 按单向计、×2 得双向约 25 GB/s，对 ICI 双向 1,200 GB/s 约 48 倍，为本课换算（只按单向比则约 96 倍）</td></tr>
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
    "__FIG_PP_BUBBLE__": ("fig-pp-bubble", "fig5-pp-bubble.svg", "topic05-fig-tp.py",
        '<b>浅灰色就是气泡：这一段在干等。</b><br>'
        '<em>4 段 8 个 micro-batch，气泡是理想计算时间的 3/8。</em>'),
    "__FIG_MOE_PARAMS__": ("fig-moe-params", "fig5-moe-params.svg", "topic05-fig-ep.py",
        '<b>那一小截灰色，是注意力、共享专家、稠密 MLP 和词表加起来的全部。</b><br>'
        '<em>路由专家的份额由 config.json 的尺寸现算。</em>'),
    "__FIG_FOLD__": ("fig-fold", "fig5-fold.svg", "topic05-fig-ep.py",
        '<b>左右两边是同样的 8 张卡。</b><br>'
        '<em>进 attention 时按 TP 组干活，进专家层时每张卡管 32 个专家。</em>'),
    "__FIG_CP_ZIGZAG__": ("fig-cp-zigzag", "fig5-cp-zigzag.svg", "topic05-fig-seq.py",
        '<b>同样 8 块，换一种分法，最忙和最闲从差好几倍变成一样忙。</b><br>'
        '<em>每格数由脚本按因果掩码现算。</em>'),
    "__FIG_KV_DUP__": ("fig-kv-dup", "fig5-kv-dup.svg", "topic05-fig-seq.py",
        '<b>红色那 7 份，存的是一模一样的东西。</b><br>'
        '<em>KV 尺寸取自 V3 的 config.json。</em>'),
    "__FIG_PD__": ("fig-pd", "fig5-pd.svg", "topic05-fig-pd.py",
        '<b>上面那条被截走的几格，就是拆开要换回来的东西。</b><br>'
        '<em>时间线是示意；KV 传输的 100 ms 是按我们 v7x 那套的带宽估算的。</em>'),
    "__FIG_AFD__": ("fig-afd", "fig5-afd.svg", "topic05-fig-pd.py",
        '<b>拆开的不是张量，是一层里的两种活。</b><br>'
        '<em>机器数和格子都是示意。</em>'),
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
        '<em>每一步谁发给谁、加到哪一块，由脚本按环形调度现算。</em>'),
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
