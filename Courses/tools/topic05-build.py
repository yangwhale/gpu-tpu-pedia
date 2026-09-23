import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# -*- coding: utf-8 -*-
"""专题五 · 并行策略 —— 教材。

════════════════════════════════════════════════════════════════
这个文件的状态：写完的和没写的混在一起，而且必须看得出来
════════════════════════════════════════════════════════════════
2026-09-23 开工。第一步的要求是「先把所有并行方式列全，不分训练推理，
新的别漏」—— 那一轮调研直接写成了 §1（全景），**这一节是写完的**。
其余各节还是大纲，照专题八的做法渲染成 🚧 占位，不补写。

⛔ 不要为了「看起来完整」去补写占位节 —— 编出来的内容看着最合理，
   也最难被自己发现（第一原则）。

§1 里每一个参数名、每一句定义，都对过源码或官方文档（出处在文末台账）。
几条值得记住的核对结果：
  · TEP / DEP 是推理侧叫法（TRT-LLM blog26 原文定义），D 是 attention DP；
    Megatron 仓库里 grep 为零。**不能等同 ETP / EDP** —— 大纲原来那句是错的。
  · vLLM 的 PCP 扩大 world size、DCP 不扩大（复用 TP rank）——
    vllm/config/parallel.py 的 docstring 原话。
  · 「DTP」这个缩写没有在任何框架文档里查到，所以这一页不列。

⭐ 大纲在 `Courses/专题05-并行策略.md`，只是计划，不必与页面同步。

CSS 从专题二 L300 抄（同专题八，见 topic08-build.py 文件头）。
"""
import io
import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "WebPages", "topic-02-L300.html")
OUT = os.path.join(HERE, "..", "WebPages", "topic-05.html")

_src = io.open(SRC, encoding="utf-8").read()


def _sub(text, pattern, repl, what):
    """替换 + 断言真的替换了（str.replace 匹配不上是静默的，见 topic08-build.py）。"""
    new, n = re.subn(pattern, repl, text, count=1)
    assert n == 1, "改不动 %s —— 模板变了？（模式：%s）" % (what, pattern)
    return new


_head = _src[:_src.index("</style>") + len("</style>")]
_head = _sub(_head, r"<title>.*?</title>", "<title>专题五 · 并行策略</title>", "<title>")
_head = _sub(_head, r'<meta property="og:title" content="[^"]*">',
             '<meta property="og:title" content="并行策略 · 用一种通信，换一份显存或一份算力">', "og:title")
_head = _sub(_head, r'<meta property="og:description" content="[^"]*">',
             '<meta property="og:description" content="今天所有的并行方式，按四种刀法排成一张图：切数据、切序列、切权重、拆阶段。">',
             "og:description")
_head = _sub(_head, r'<meta property="og:url" content="[^"]*">',
             '<meta property="og:url" content="https://gist.higcp.com/Courses/WebPages/topic-05.html">', "og:url")
_head = re.sub(r'\s*<meta property="og:image"[^>]*>(\s*<meta property="og:image:(width|height)"[^>]*>)*',
               "", _head)
_probe = re.sub(r"/\*.*?\*/|<!--.*?-->", "", _head, flags=re.S)
assert "TPU 与 GPU" not in _probe and "topic-02" not in _probe, "head 里还有专题二的残留"

_head += """
<style>
ul + p, ul + div.note { margin-top: 14px }
.kind { display:inline-block; font-size:12px; font-weight:600; padding:0 7px;
        border-radius:9px; margin-right:4px; white-space:nowrap }
.k-d { background:#e8f0fe; color:#174ea6 }
.k-s { background:#e6f4ea; color:#0d652d }
.k-m { background:#fef7e0; color:#8a4b00 }
.k-x { background:#f3e8fd; color:#681da8 }
.k-new { background:#fce8e6; color:#a50e0e }
td .where { color:var(--gray); font-size:13px }
</style>"""

D = '<span class="kind k-d">数据</span>'
S = '<span class="kind k-s">序列</span>'
M = '<span class="kind k-m">模型</span>'
X = '<span class="kind k-x">解耦</span>'
NEW = '<span class="kind k-new">新</span>'

out = []
a = out.append

a(_head)
a('''
</head>
<body>

<!-- ══════════ HERO ══════════ -->
<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">加速器系统课程</a> ／ 主线 ／ 专题五
    ／ <b>并行策略</b></div>
  <h1>并行策略</h1>
  <div class="hook">
    一个模型装不进一块卡，就得切开分到很多卡上。<br>
    <em>每切一刀，就在那一维上多出一种通信 —— 用一种通信，换一份显存或一份算力。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    这一讲先把今天用得上的并行方式全部摆出来，训练和推理放在一起，
    再回头讲每一种什么时候该用、什么时候会被通信拖垮。</p>
  <div class="chips">
    <span class="chip">前置 <b>专题四</b>（那张 16 字节的账）</span>
    <span class="chip">口径 <b>截至 2026-09</b></span>
    <span class="chip">⏱ <b>约 1 小时</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

<div class="wrap">
  <div class="note warn"><span class="t">🚧 这一讲刚开工</span>
    <b>§1（全景）是写完的</b>，每个参数名和定义都对过源码或官方文档，出处在文末。<br>
    <b>§2 以后还是大纲</b>，页面上按原样列出，没有补写。</div>
</div>

<section id="s0"><div class="wrap">
  <div class="stn"><span class="badge">第 0 节</span><h2>这一讲要回答的一个问题</h2></div>

  <p class="lead">专题四把账算完了：一个 6,710 亿参数的模型，按每参数 16 字节算，光训练状态就要约 10.7 TB。
    <b>一块卡装不下，就得切。问题是沿哪一维切。</b></p>

  <p>一个训练中的张量有好几个维度可以下刀：batch、序列、隐藏维、层、专家。
    每切一刀，就在那一维上产生一种通信。所以这一讲从头到尾只讲一件事：</p>

  <div class="note ok"><span class="t">一句话</span>
    <b>用一种通信，换一份显存或一份算力。</b><br>
    选并行策略，就是在选你愿意付哪一种通信、付多频繁。</div>

  <p><b>目标</b>：看到任何一个并行方案，包括还没出现的，都能说出它切的是哪一维、
    多出来的是哪种通信、每一步发生几次，以及它能不能放到慢链路上。</p>
</div></section>

<section id="s1"><div class="wrap">
  <div class="stn"><span class="badge">第 1 节</span><h2>全景：今天所有的并行方式</h2></div>

  <p class="lead">名字很多，DP、FSDP、TP、SP、CP、PP、EP，推理那边还有 DCP、PCP、DEP、TEP……
    <b>但刀法只有四种。</b>先把刀法认清楚，名字就好记了。</p>

  <h3 id="s1-1">1.1 · 四种刀法</h3>
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

  <h3 id="s1-2">1.2 · 数据并行类</h3>
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

  <h3 id="s1-3">1.3 · 序列并行类</h3>
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

  <p><b>推理 decode 侧：切 KV cache</b> —— 这是 2025–26 年最热的一块</p>
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

  <h3 id="s1-4">1.4 · 模型并行类</h3>
  <p>这一类真正在切权重。按下刀的位置再分三种：矩阵内部、层、专家。</p>

  <p><b>切矩阵内部（张量并行）</b></p>
  <table>
    <tr><th>名称</th><th>切什么</th><th>解决什么</th><th>多出来的通信</th><th>场景</th></tr>
    <tr><td>TP</td><td>矩阵先按列切、再按行切，两两配对</td><td>单层的权重或计算放不下</td>
      <td>每层两次 all-reduce，<b>频率极高</b>，所以只能待在 NVLink 或 ICI 一跳之内</td><td>训 · 推</td></tr>
    <tr><td>2D / 2.5D / 3D TP</td><td>把矩阵切成网格</td><td>1D TP 的通信随度数上涨</td>
      <td>沿网格的行、列广播和归约</td><td>训，<b>已基本不用</b></td></tr>
    <tr><td>GTP ''' + NEW + '''</td><td>在 TP 轴上再把权重切一层，用的时候再收回来</td>
      <td>Megatron 文档原话：GTP_remat is an implementation of ZeRO-3 —— 只不过切在模型并行轴上</td>
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
    <b>两套是分开配的</b>。推理时同样如此，只是换了一套名字，见 1.6。
    <br>这是 MoE 模型调并行时自由度最大、也最容易配错的地方。</div>

  <h3 id="s1-5">1.5 · 解耦类</h3>
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

  <h3 id="s1-6">1.6 · 组合简称：TEP 和 DEP</h3>
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
    <em>这正是 §6 要讲的「怎么选」—— 这里先记住两个名字各是什么。</em></p>

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

  <h3 id="s1-7">1.7 · 常被当成并行、其实不是的</h3>
  <p>判据：<b>它有没有多切出一维。</b>没有的，就不是新的并行方式。</p>
  <table>
    <tr><th>名称</th><th>它实际是什么</th></tr>
    <tr><td>TBO / SBO / DBO</td><td>把通信藏到计算后面：两批数据错开跑，一批在通信时另一批在计算</td></tr>
    <tr><td>Async TP、TP 通信重叠、collective matmul</td><td>还是 TP，只是把通信和矩阵乘切成小块交错着做</td></tr>
    <tr><td>EPLB、冗余专家</td><td>负载均衡：把热门专家多复制几份再重新摆放。修的是 EP 的病，本身不切新维度</td></tr>
    <tr><td>Offload、重计算</td><td>拿时间换显存，或者拿主机内存换显存</td></tr>
  </table>

  <h3 id="s1-8">1.8 · 同名不同义</h3>
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

<section id="s2"><div class="wrap">
  <div class="stn"><span class="badge">第 2 节</span><h2>🚧 什么时候真的会被通信拖累</h2></div>
  <div class="note warn"><span class="t">🚧 这一节还是大纲</span>下面是计划要讲的东西，还没有展开。</div>
  <ul>
    <li><b>判据还是那道除法，换个分母</b>：算力 ÷ 这一维用的那条链路的带宽</li>
    <li><b>AllReduce = AllGather ＋ ReduceScatter</b>，所以 FSDP 的通信量跟 DP 一样 —— DP 跑得动，就该换 FSDP</li>
    <li><b>FSDP 搬权重，TP 搬激活</b>：批次小多用 TP，批次大多用 FSDP</li>
    <li>TP 有一个跟批次无关的硬上限</li>
  </ul>
</div></section>

<section id="s3"><div class="wrap">
  <div class="stn"><span class="badge">第 3 节</span><h2>🚧 逐个讲</h2></div>
  <div class="note warn"><span class="t">🚧 这一节还是大纲</span>
    §1 已经把每一种是什么、切哪一维讲完了。这一节要讲的是每一种<b>怎么落地、踩什么坑</b>。</div>
  <ul>
    <li><b>DP → FSDP / HSDP</b>：为什么这笔买卖通常划算</li>
    <li><b>TP</b>：度数受头数和高带宽域大小两头限制；跨机做 TP 基本等于自杀</li>
    <li><b>SP 与 CP</b>：这两个最容易混；为什么 128K 以上必须上 CP；MLA 下 CP 的特殊性</li>
    <li><b>PP 与 VPP</b>：气泡怎么算；V3 前 3 层 dense、后 58 层 MoE，怎么切才平衡</li>
    <li><b>EP 家族</b>：负载不均为什么是 EP 独有的病；训练和推理两套名字怎么对上</li>
    <li><b>推理侧</b>：Attention DP、DCP、PD 分离下两边各怎么配</li>
  </ul>
</div></section>

<section id="s4"><div class="wrap">
  <div class="stn"><span class="badge">第 4 节</span><h2>🚧 组合：mesh 与拓扑映射</h2></div>
  <div class="note warn"><span class="t">🚧 这一节还是大纲</span>下面是计划要讲的东西，还没有展开。</div>
  <ul>
    <li><b>mesh 是什么</b>：把物理设备排成多维网格，每个逻辑并行维绑一根轴</li>
    <li><b>映射决定成败</b>：高频通信（TP）绑最快的轴，低频（DP）绑最慢的轴</li>
    <li>我们的实测：换拓扑和调参数，收益不在一个量级</li>
    <li>并行度不是连续旋钮，是一组离散、互相约束的选择</li>
  </ul>
</div></section>

<section id="s5"><div class="wrap">
  <div class="stn"><span class="badge">第 5 节</span><h2>🚧 评价：strong scaling 与 weak scaling</h2></div>
  <div class="note warn"><span class="t">🚧 这一节还是大纲</span>下面是计划要讲的东西，还没有展开。</div>
  <ul>
    <li><b>Strong scaling</b>：问题规模固定、加卡求快 —— 每卡活变少，通信占比必然上升</li>
    <li><b>Weak scaling</b>：每卡活固定、加卡同时放大问题 —— 通信占比大致不变</li>
  </ul>
</div></section>

<section id="s6"><div class="wrap">
  <div class="stn"><span class="badge">第 6 节</span><h2>🚧 实际该怎么选</h2></div>
  <div class="note warn"><span class="t">🚧 这一节还是大纲</span>下面是计划要讲的东西，还没有展开。</div>
  <ol>
    <li><b>先看装不装得下</b>：FSDP / EP 先上到装得下为止</li>
    <li><b>再看高带宽域有多大</b>：它决定 TP / EP 的度数上限</li>
    <li><b>再看序列多长</b>：超过某个长度必须上 CP（推理看 DCP）</li>
    <li><b>PP 通常最后考虑</b>：气泡是纯损失</li>
    <li><b>DP 兜底</b>：剩下的规模用它铺开，可以铺到 DCN 上</li>
  </ol>
</div></section>

<section id="src"><div class="wrap">
  <div class="stn"><span class="badge">出处</span><h2>§1 的出处台账</h2></div>
  <p class="lead">按「结论 ← 材料」排。全部在 2026-09-23 核对。</p>
  <table>
    <tr><th>结论</th><th>材料</th></tr>
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

<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    上一讲 <a href="topic-04.html">专题四 · 反向与优化器</a></p>
</div>

</body></html>''')

import course_links as _CL
_html = _CL.linkify_arxiv("\n".join(out))
io.open(OUT, "w", encoding="utf-8").write(_html)
print("ok  topic-05.html  %s 字符" % format(os.path.getsize(OUT), ","))
