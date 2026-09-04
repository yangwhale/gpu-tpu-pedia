# -*- coding: utf-8 -*-
"""专题八 · 精度与量化 —— 教材。

════════════════════════════════════════════════════════════════
⭐ 这一讲为什么会在轮到它之前就开工
════════════════════════════════════════════════════════════════
2026-09-04 Chris 立的规矩（原话）：

    「以后我们不是说『轮到哪个专题才写』，而是在日常写前面专题的时候，
      如果有什么对后面有帮助的材料，就先放在后面的专题里，
      省得等到时候轮到那个专题的时候，再去收集材料。」

这一讲是这条规矩的**第一个产物**：讲专题二 §3.5「各自多出来的那一块」时，
他连问六轮 NVFP4／MXFP4，攒出一万两千字带出处的内容 ——&nbsp;
而它的主题归属明明是**这一讲**，不是「TPU vs GPU」。
所以整块迁过来，专题二那边只留一句指针。

════════════════════════════════════════════════════════════════
⛔ 这个文件的状态：**已写的和没写的混在一起，而且必须看得出来**
════════════════════════════════════════════════════════════════
§3、§4 的一半、§5 是**写完的**（有出处、经过六轮追问打磨）。
§0 §1 §2 §6 还是**大纲**。

⛔ 不要为了「看起来完整」去补写那几节 ——&nbsp;
   编出来的内容看着最合理，也最难被自己发现（第一原则）。
   没写的就渲染成 🚧 占位，把大纲条目原样列出来，让读者一眼看见边界。

⭐ 大纲在 `Courses/专题08-精度与量化.md`，那份是**计划**；
   这份是**成品**。两边的 🚧 标记要一致 —— 改了一边记得对一下另一边。

════════════════════════════════════════════════════════════════
CSS 从专题二 L300 抄
════════════════════════════════════════════════════════════════
跟 L200 的做法一样：整个 <head> 连 CSS 一起搬。理由同样是
「同一个职责不要两个载体」——&nbsp;各留一份必然走散。
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "WebPages", "topic-02-L300.html")
OUT = os.path.join(HERE, "..", "WebPages", "topic-08.html")

_src = io.open(SRC, encoding="utf-8").read()

# 从专题二 L300 抽 <head>（含 CSS），把标题换掉
_head = _src[:_src.index("</style>") + len("</style>")]
_head = _head.replace("<title>TPU 与 GPU", "<title>精度与量化", 1)

# ⛔ 专题二那份 CSS 里 p 只有 margin-bottom，没有 margin-top，而 li 是 flex。
#    于是紧跟在 </ul> 后面的段落会**贴到最后一个 bullet 上**，读起来像是
#    那条 bullet 的一部分 —— 专题二用不着这条是因为它极少 ul 后接 p，
#    这一讲（大纲占位多）到处都是。补一条，不去动专题二那份。
_head += """
<style>
ul + p, ul + div.note { margin-top: 14px }
</style>"""

out = []
a = out.append

a(_head)
a('''
</head>
<body>

<!-- ══════════ HERO ══════════ -->
<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">加速器系统课程</a> ／ 深入 ／ 专题八
    ／ <b>精度与量化</b>
    <a class="lecbtn" href="topic-08-lecture.html">📝 讲义（授课稿）</a></div>
  <h1>精度与量化</h1>
  <div class="en">Precision and Quantization</div>
  <div class="hook">
    每往下压一档精度，就省一半的显存和带宽。<br>
    <em>代价是数值上的安全边际 ——&nbsp;这一课讲这条边界具体在哪。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    「压精度」不是一个开关，是一整套决策：哪些张量能压、缩放因子怎么算、
    训练和推理为什么是两件事。<b>而这一讲风险最高的地方在于 ——&nbsp;
    量化出问题往往不报错。</b>
  </p>
  <div class="chips">
    <span class="chip">前置 <b>专题四 · 专题六</b></span>
    <span class="chip">硬件侧 <b>Blackwell (B200)</b></span>
    <span class="chip">规格来源 <b>全部公开可核</b></span>
    <span class="chip">⏱ <b>约 1 小时</b></span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

<div class="wrap">
  <div class="note warn"><span class="t">🚧 这一讲写了一半，而且哪一半写了是标出来的</span>
    <b>§3（缩放）、§4 的后半（FP4 训练）、§5（硬件视角）是写完的</b>
    ——&nbsp;<em>带完整出处，见文末的出处台账。</em><br>
    <b>§0 §1 §2 §6 还是大纲</b>，页面上按原样列出，<b>没有补写</b>。<br>
    <em>⭐ 之所以先写中间这三节：它们是讲<b>专题二</b>的时候问出来的 ——&nbsp;
    材料在讨论中长出来的那一刻就写进它该属于的这一讲，不等轮到它再去收集。</em></div>
</div>

<section id="s0"><div class="wrap">
  <div class="stn"><span class="badge">第 0 节</span><h2>这一讲要回答的一个问题</h2></div>

  <p class="lead">FP8 是 DeepSeek V3 的三大贡献之一，FP4 权重已经出现在更新的模型上。
    <b>所有人都在往下压精度。</b></p>

  <p>但「压精度」不是一个开关，它是一整套决策：</p>
  <ul>
    <li>哪些张量可以压、哪些必须保住？</li>
    <li>缩放因子怎么算？<b>per-tensor 还是 per-block？</b></li>
    <li>训练时压和推理时压，<b>是完全不同的两件事</b></li>
    <li><b>压过头会怎样？</b>——&nbsp;<em>不是「精度略降」，是收敛直接毁掉。</em></li>
  </ul>

  <p><b>目标</b>：看到一个量化方案，<b>能说出它在哪里省、风险在哪、
    以及怎么验证它没把模型搞坏。</b></p>

  <div class="note ok"><span class="t">⭐ 这一讲分两半，而且分界线是可验证的</span>
    <b>训练侧和推理侧要分开讲</b> ——&nbsp;依据不是「目标不同」这种软理由，
    而是一个结构差异：<br>
    <b>同一个 FP4 格式，推理侧权重按 <u>1×16</u> 一维分块，
    训练侧权重按 <u>16×16</u> 二维分块。</b><br>
    <em>原因在 §4 ——&nbsp;而且那个原因跟精度无关。</em></div>
</div></section>

<section id="s1"><div class="wrap">
  <div class="stn"><span class="badge">第 1 节</span><h2>🚧 为什么低精度能行</h2></div>

  <div class="note warn"><span class="t">🚧 这一节还是大纲</span>
    下面是计划要讲的东西，<b>还没有展开</b>。</div>

  <ul>
    <li>神经网络对<b>单个数</b>的精度并不敏感 ——&nbsp;它敏感的是<b>统计上的分布</b></li>
    <li>训练本身就带噪声（随机初始化、数据顺序、dropout），
      量化误差在很多地方可以被当成又一种噪声</li>
    <li><b>但有些地方不行</b> ——&nbsp;那些地方就是这一讲要找出来的</li>
  </ul>

  <p><b>两个收益，性质不同</b>（这一条要讲透，否则后面分不清训练和推理）：</p>
  <ul>
    <li><b>算力</b>：低精度矩阵乘吞吐更高 ——&nbsp;受益的是<b>算力受限</b>的场景（训练、prefill）</li>
    <li><b>访存</b>：读写的字节直接减半 ——&nbsp;⭐ 受益的是<b>带宽受限</b>的场景（decode）</li>
  </ul>
  <p><em>所以量化在推理侧的收益比训练侧直接得多：
    decode 读一遍权重就是全部开销，<b>字节减半近似就是快一倍</b>。</em></p>
</div></section>

<section id="s2"><div class="wrap">
  <div class="stn"><span class="badge">第 2 节</span><h2>🚧 格式：这些名字到底在说什么</h2></div>

  <div class="note warn"><span class="t">🚧 这一节还是大纲（FP4 那几行已经在 §3 讲透）</span>
    位布局图还没画 ——&nbsp;<b>BF16 vs FP16 那个对比不画图说不清</b>，已列入待办。</div>

  <p><b>要立的一条主线</b>：<b>范围（指数位）比精度（尾数位）重要。</b></p>
  <ul>
    <li><b>BF16 为什么赢了 FP16</b>：它牺牲尾数保住指数位 ——&nbsp;
      <em>动态范围跟 FP32 一样宽，只是精度粗。</em>
      <b>溢出是致命的，精度粗一点只是噪声。</b></li>
    <li>⭐⭐ <b>而这条规律会原样再来一次</b>：MXFP4 的 scale 选了纯指数的
      <code>E8M0</code>（范围极大、精度极差），NVFP4 选了 <code>E4M3</code>
      （有尾数、范围只到 448）——&nbsp;<b>同一个取舍，换了个位置又演一遍。</b>
      <em>见 §3。</em></li>
    <li>还要讲：<b>subnormal</b> 与 flush-to-zero；累加为什么通常仍是 FP32</li>
  </ul>
</div></section>

<section id="s3"><div class="wrap">
  <div class="stn"><span class="badge">第 3 节</span><h2>⭐ 缩放这件事</h2></div>

  <p class="lead">这一节是整讲的核心。<b>从一个问题往下长：四个 bit 只能表示八个数，
    这怎么可能算得准？</b></p>

  <div class="note info"><span class="t">这一节的走法</span>
    <b>① 为什么四个 bit 够用</b>（靠 scale）→&nbsp;
    <b>② scale 要多密</b>（粒度阶梯）→&nbsp;
    <b>③ 命名碰撞</b>（「块」不蕴含二维）→&nbsp;
    <b>④ 两个格式差在哪</b>（scale 的类型）→&nbsp;
    <b>⑤ 两级 scale 各管什么</b>（外层管范围）→&nbsp;
    <b>⑥ 那 16 个数怎么排</b>（一维，不是方块）。
    <br><em>⭐ 只想拿一句话走：<b>四个 bit 只负责形状，量级由 scale 给；
    scale 有多密，决定这个格式能不能用。</b></em></div>
''')

# ── §3：从专题二迁过来的「缩放」那一段 ─────────────────────────
a(io.open(os.path.join(HERE, "topic08-seg-scaling.html"), encoding="utf-8").read())

a('''
</div></section>

<section id="s4"><div class="wrap">
  <div class="stn"><span class="badge">第 4 节</span><h2>训练侧：混合精度混了什么</h2></div>

  <div class="note warn"><span class="t">🚧 前半是大纲，后半（FP4 训练）已写完</span>
    混合精度、主权重、FP8 的 scaling、QAG ——&nbsp;<b>这四块还没展开</b>。<br>
    <b>「一个线性层三个矩阵乘」和「16×16 与链式法则」是写完的</b>，在下面。</div>

  <p><b>先回到专题四那张 16 字节的表</b>，从精度角度重看一遍：</p>
  <ul>
    <li><b>前向／反向用 bf16</b> ——&nbsp;省算力省带宽</li>
    <li><b>主权重必须用 fp32</b> ——&nbsp;⚠️ bf16 只有 7 位尾数，
      训练后期的小更新量加上去会被<b>直接舍掉</b>，等于没更新</li>
    <li><b>优化器状态、累加</b> ——&nbsp;同理，fp32</li>
  </ul>

  <div class="note warn"><span class="t">🚧 还要讲（都是我们自己踩过的）</span>
    <b>· FP8 训练的 scaling 怎么定</b>（per-tensor／per-block／动态 vs 静态；
    V3 用 128×128 分块 ——&nbsp;<em>正好跟下面那个 16×16 是同一路子</em>）<br>
    <b>· ⚠️ <code>fixed</code> 固定缩放会毁掉收敛</b>，不是「精度略降」，是训坏<br>
    <b>· ⭐ QAG：量化后再 all-gather</b> ——&nbsp;五道锁、只开一个会<b>静默失效</b>、
    实测 0.88× → 1.05×</div>
''')

# ── §4 后半：量化算式 ＋ 三个 GEMM ＋ 16×16 ──────────────────
a(io.open(os.path.join(HERE, "topic08-seg-training.html"), encoding="utf-8").read())

a('''
</div></section>

<section id="s5"><div class="wrap">
  <div class="stn"><span class="badge">第 5 节</span><h2>⭐ 硬件视角：Blackwell 到底加了什么</h2></div>

  <p class="lead">这一节接<a href="topic-02.html">专题二</a>。
    <b>「硬件支持 FP4」这句话，落到硅上到底是什么？</b></p>
''')

a(io.open(os.path.join(HERE, "topic08-seg-hardware.html"), encoding="utf-8").read())

a('''
  <div class="note warn"><span class="t">🚧 这一节还差两块</span>
    <b>· 各代硬件支持到哪一档</b>（哪一代开始有 FP8、有 FP4）<br>
    <b>· ⚠️ 低精度的收益有没有真落到 MXU 上</b> ——&nbsp;
    有时候编译器又把它转回去了，<b>profile 里能看出来</b></div>
</div></section>

<section id="s6"><div class="wrap">
  <div class="stn"><span class="badge">第 6 节</span><h2>🚧 一条贯穿的教训：沉默的失败</h2></div>

  <div class="note warn"><span class="t">🚧 这一节还是大纲</span>
    但它是这一讲<b>风险最高</b>的一节，不能删。</div>

  <p><b>量化出问题往往不报错。</b>已经攒下三类，一类比一类隐蔽：</p>
  <ul>
    <li><b>① 编译过了、跑起来了、loss 也在降</b> ——&nbsp;
      但降得比该有的慢，或者后期崩</li>
    <li><b>② 开关没生效也不报错</b> ——&nbsp;QAG 那五道锁只开一把就是这样</li>
    <li>⭐ <b>③ 链式法则被打破</b>（<code>w<sub>fprop</sub> ≠ w<sub>bprop</sub></code>）
      ——&nbsp;<em>它连「开关没生效」都算不上：<b>一切正常，只是你在给另一个函数求导。</b></em>
      <b>见 §4。</b></li>
  </ul>

  <p><b>所以：每一次精度改动都必须有对照实验，而且要跑足够长才看得出差别。</b></p>

  <div class="note danger"><span class="t">⛔ 跟专题七那条是同一类问题</span>
    「测试全绿不是证据」——&nbsp;<b>在数值这件事上，「没报错」什么都不能说明。</b></div>
</div></section>

<section id="src"><div class="wrap">
  <div class="stn"><span class="badge">出处</span><h2>已写部分的出处台账</h2></div>

  <p class="lead">按<b>「结论 ← 材料」</b>排，不按书目排 ——&nbsp;
    <em>要回查的人手里拿的是结论，不是书单。</em>
    <b>凡是推的都单独标了。</b></p>
''')

a(io.open(os.path.join(HERE, "topic08-seg-sources.html"), encoding="utf-8").read())

a('''
</div></section>

<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    这一讲的硬件背景在 <a href="topic-02.html">专题二 · TPU 与 GPU</a>　·
    <a href="topic-08-lecture.html">📝 讲义</a></p>
</div>

</body></html>''')

io.open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("ok  topic-08.html  %s 字符" % format(os.path.getsize(OUT), ","))
