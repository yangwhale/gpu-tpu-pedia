# -*- coding: utf-8 -*-
"""专题六 · 第十一节「投机解码：搬一趟，出好几个字」。图在 topic06-fig-spec.py。

出处（行号见图脚本 docstring）：K3 两本 runbook 与 PD-BACKLOG 在 gpu/inference/a4x-max/kimi-k3/；
V4 两本在 gpu/inference/a4x-max/deepseek-v4/；B200 在 gpu/inference/a4/deepseek-v4/README.vllm.md。
"""

BODY = '''
  <p class="lead">第三到十节大多在做大 batch；可交互档限住了 batch，推理模型又一答几千字（0.1），「搬一趟权重只出一个字」就成了硬上限。
    投机解码先便宜地猜几个字，再让大模型搬一趟验完，<b>猜对几个就出几个</b>。</p>

  <h3 id="s11-1">11.1　先猜后验：收益账</h3>
  <p>大模型把草稿当一段输入，一步算出每个位置该写的字，逐个对照。</p>
__FIG_S11_STEP__
<figure class="fbox fwide" id="anim-spec">
<video src="media/topic06-spec.mp4" autoplay loop muted playsinline
       aria-label="投机解码一步的动画。标题：投机解码：小模型先猜 4 个，大模型一趟验完。三行，分别标着已写下、小模型猜、大模型验；底下一条用时条，图例写：紫＝小模型猜一个（便宜），灰＝大模型搬一趟权重（贵）。开始时已写下「床前明月」。字幕依次是：第 1 步：小模型一个接一个猜 4 个字，每个都很快（猜出「光疑似地」，用时条多 4 小格紫色）。大模型搬一趟权重，同时算出这 5 个位置它自己会写的字（一个蓝框标着「一趟」，里面是「光、疑、是」和两个问号，用时条多一大格灰色）。从左往右逐个比：一样就收下（光、疑变绿）。第 3 个不一样：换成大模型自己的字，后面的全丢掉（「似」变红，「是」被黄框圈出，后面变暗；小字注：？＝前面那个字已经错了，这个位置算了也作废）。「光疑是」飞上去，第 1 步完成：猜对 2 个 ＋ 大模型补 1 个 ＝ 出 3 个字，停住。第 2 步：小模型猜「地上霜举」，大模型一趟算出「地上霜举头」；4 个全对；第 5 个位置大模型本来就算了，白送 1 个。「地上霜举头」飞上去，第 2 步完成：猜对 4 个 ＋ 白送 1 个 ＝ 出 5 个字，停住。最后：猜对几个就出几个，再加大模型自己的 1 个：每步至少出 1 个。画面复位。"></video>
<figcaption>看碰到第一个错字之后，大模型怎么接手。
  <span class="sub">（23 秒无声循环，Manim 渲染。示意，按温度 0 贪心比对。）</span></figcaption></figure>
  <p>用标准拒绝采样（按概率决定每个草稿字收不收）时，投机解码只改速度，不改输出分布。</p>

  <p><b>收益账：接受长度 ÷ 一步变慢了多少。</b>每步平均出几个字叫接受长度。小 batch 时一步主要在读权重，多验几个 token 本该几乎不加时间。</p>
__FIG_S11_BYTES__
  <p>MoE 不白送：每个 token 挑自己的专家，验得越多读的专家越多（图左；估算，偏高）。</p>
  <p>图右是 K3 在 GB300 上的实测：一步慢了一截，接受长度却大得多，相除仍是好几倍。</p>

  <h3 id="s11-2">11.2　收益会缩，温度会骗人</h3>
  <p><b>收益会缩：并发、上下文、DEP。</b>并发一高算力先满，草稿里每个 token 都得真算。
    K3 输出吞吐对不投机的倍数，并发从 1 加到 64 一路往下缩。并发 1 的倍数也低于 11.1 的每字提速，因为端到端含不受加速的 prefill。</p>
__FIG_S11_CONC__
  <p>上下文变长也会缩（图最右柱）。
    验证本不该因此变贵：不开投机时每字间隔只从 11.1 涨到 11.4 ms，多验的位置又共用同一份 KV。
    更可能是接受长度掉了，或草稿读长 KV 变慢（没实测，本课推导）。</p>
  <p>DEP（第六节）本为大 batch 而设，高并发时算力已满，投机成了纯开销：B200 上 DeepSeek-V4-Flash 用 vLLM、DEP8 开 MTP，十档并发全变慢（−3% 到 −37%），并发 1 也慢 12%。</p>
  <p>峰值和端到端还可能反向：GB300 上 V4-Pro 把 MTP 从 1 步加到 2 步，decode 峰值掉 13%，端到端反涨 28%。真实负载 batch 多半没那么大，白送的那部分还在。</p>

  <p><b>温度：压测怎么发，结果就怎么变。</b>草稿的抽样要和请求的温度对上。贪心草稿（相当于温度 0）遇上温度 1.0 的随机采样，理论接受率就是大模型给这个 token 的概率，会降但不归零。</p>
__FIG_S11_TEMP__
  <p>GB300 上 V4-Pro 用的 DSpark 草稿（11.3）是贪心的，服务端默认温度 1.0。
    压测工具不发温度，就按 1.0 采样。
    实测接受率（图）远低于理论值：它是收下的 ÷ 猜的，每步猜 7 个，第 1 个被拒，后 6 个跟着作废（本课推导）。</p>
  <div class="note warn"><span class="t">接受率掉了不会报错</span>
    它只让 TPOT 慢慢变差。线上要把接受率和接受长度当一级指标告警；跨框架比先对齐两边发不发 temperature。</div>

  <h3 id="s11-3">11.3　草稿从哪来</h3>
  <p>各家报的倍数口径不同，不能横比。</p>
  <ul>
    <li><b>2022-11</b>：Google 提出（DeepMind 2023-02 独立提出），独立小模型当草稿，报单请求速度是不投机的 2–3 倍。</li>
    <li><b>2024-01</b>：Medusa 在大模型上加几个解码头，报速度是不投机的 2.2–3.6 倍；EAGLE 用大模型的隐状态（出字前那层的向量）猜，是不投机的 2.7–3.5 倍。</li>
    <li><b>2024-12</b>：DeepSeek-V3 的 MTP 第 2 个字接受率 85–90%，速度约是不开 MTP 的 1.8 倍；草稿训进主模型，这招从此成了主流。</li>
    <li><b>2025-03</b>：EAGLE-3 融合大模型浅、中、深三层的状态，单请求速度最多是不投机的 6.5 倍，batch 64 时吞吐仍是不投机的 1.38 倍（SGLang）。</li>
    <li><b>2026-02</b>：DFlash 的小草稿一次前向并行出一整块（块扩散），论文报比 EAGLE-3 的加速比最多再高 2.5 倍。</li>
    <li><b>2026-07</b>：DeepSeek 的 DSpark 拿 DFlash 当骨干，补一个顺序小头，再按置信度定每步验几个；摘要报吞吐相同时单用户生成比线上 MTP-1 快 60–85%。</li>
  </ul>
__FIG_S11_TREE__
  <p>还有一支不要草稿模型：n-gram（也叫 Prompt Lookup）拿已经写出的最后几个字，到提示词里找一样的片段，把后面几个字照抄成草稿。</p>
__FIG_S11_NGRAM__
  <p>vLLM 在 4 张 H100 上跑 Llama 3 70B、每秒 1 个请求，摘要数据集上最多快 2.8 倍；同条件换小草稿模型跑 ShareGPT 对话是 1.5 倍。开法：vLLM 的 <code>speculative_config</code> 里 <code>method</code> 写 <code>ngram</code>，SGLang 是 <code>--speculative-algorithm NGRAM</code>。</p>
  <p><b>MTP、DFlash、DSpark 是一条线上的三步。</b>三者都是贴着大模型隐状态猜字的小草稿，差在怎么出字。MTP 和 EAGLE 一个字一个字猜，猜 5 个跑 5 趟；DFlash 一趟出整块，猜 5 个和猜 1 个差不多贵。</p>
__FIG_S11_THREE__
  <p>DFlash 还挪了喂隐状态的位置：EAGLE-3 只在草稿入口喂，草稿一深信号就淡；DFlash 注入草稿每一层的 KV（LMSYS 博客）。2026 年已上 vLLM、SGLang，B300 上单并发速度是不投机的 3.0–5.8 倍（Gemma-4 31B，NVIDIA 博客）。</p>
  <p><b>一趟出整块的病：越往块尾越不准。</b>每个位置各猜各的，看不到同块前一位到底选了哪个字。上文既能接「of course」又能接「no problem」时，它会拼出「of problem」。DSpark 论文里，EAGLE-3 的逐位接受率持平，DFlash 一路往下掉。</p>
  <p><b>DSpark 第一刀补草稿。</b>DFlash 照旧一趟出整块，上面加一个很轻的顺序小头，按前一位选的字调下一位的分数：选了「of」，就抬高「course」。Qwen3 4B–14B 上平均接受长度比 DFlash 高 16–18%，比 EAGLE-3 高 27–31%；batch 128 时草稿从 4 个加到 16 个，整轮只比 DFlash 慢 0.2–1.3%。</p>
  <p><b>第二刀补验证，这才是它线上好用的主因。</b>另一个小头估每一位被收下的概率，连乘成前缀存活概率。调度器把整批请求的候选前缀按存活概率排队，再查引擎吞吐表，定这一步总共验多少。负载轻多验，负载重少验，没把握的字不占名额。</p>
  <p>这正好解了 11.2 的坑：固定猜 3 个、5 个的 MTP 在高并发下验证太贵、拉低总吞吐，所以 DeepSeek 线上一直只开 MTP-1（每次验 2 个字）。换成最多猜 5 个的 DSpark，并发不到 200 时每请求验到约 4–6 个；V4 预览版上线两周后，它替下了 MTP-1。</p>
  <p>它在 V4 上占的正是 MTP 的位子：vLLM 按 <code>mtp.0–2</code> 的名字从主模型 checkpoint 读它的权重。所以说 DSpark 是 DeepSeek 线上 MTP 的接班人，出字方法却走 DFlash 那一路。</p>

  <h3 id="s11-4">11.4　接进系统的代价</h3>
  <ul>
    <li><b>草稿要占显存。</b>K3 开 DSpark 后 KV 池从 787,504 token 缩到 637,504，少了 19%。</li>
    <li><b>引擎默认值会跟着变。</b>K3 的 SGLang 配方开投机又没写 max-running-requests，它就被重设成 48，batch 上限悄悄压低。</li>
    <li><b>PD 碰上投机解码。</b>刚从 prefill 交接来的请求只算 1 个 token，别人是 1＋N（N 个草稿），形状不齐就用不上 CUDA Graph（一串 kernel 录好一次发出，见 <a href="#s12-3">12.3</a>）；decode 端补齐到 1＋N 后，TPOT 约 40 ms 降到 22 ms（vLLM GLM-5.2 PD 博客）。</li>
  </ul>

  <div class="note ok"><span class="t">回到主线</span>
    <b>投机解码让搬一趟权重换来好几个字</b>。batch 小时赚得最多；batch 一大算力吃满，收益就缩，甚至倒亏（11.2）。
    留下的问题：batch 大时投机帮不上，那还能省什么（第十二节）？</div>
'''

FIGS = {
    "__FIG_S11_STEP__": ("fig-s11-step", "fig6-spec-step.svg", "topic06-fig-spec.py",
        '<b>对齐上下两行的灰条：投机这行少读一遍权重，出的字反而多。</b><br>'
        '<em>示意时间线，格子长短不按比例。</em>'),
    "__FIG_S11_BYTES__": ("fig-s11-bytes", "fig6-spec-bytes.svg", "topic06-fig-spec.py",
        '<b>看左边：稠密部分验几个都只读一遍，专家却越验越多。</b><br><em>左为本课推导（FP8、均匀路由）；右为 K3 在 GB300 上 bs=1 的实测时间账。</em>'),
    "__FIG_S11_CONC__": ("fig-s11-conc", "fig6-spec-conc.svg", "topic06-fig-spec.py",
        '<b>看柱子往右怎么缩。</b><br><em>K3 输出 tok/s 总量；B200 与 V4-Pro 的对照见出处。</em>'),
    "__FIG_S11_TEMP__": ("fig-s11-temp", "fig6-spec-temp.svg", "topic06-fig-spec.py",
        '<b>看温度 1.0 那侧：草稿猜对也常被判错。</b><br><em>V4-Pro 为 4K/1K 并发 256 实测；五个候选字是举例。</em>'),
    "__FIG_S11_NGRAM__": ("fig-s11-ngram", "fig6-spec-ngram.svg", "topic06-fig-spec.py",
        '<b>顺着 ① 到 ④：橙色对上，绿色照抄，红色被拒处换成蓝色。</b><br><em>示例自拟，只说明机制。</em>'),
    "__FIG_S11_THREE__": ("fig-s11-three", "fig6-spec-three.svg", "topic06-fig-spec.py",
        '<b>从上往下三行：串行猜、一趟出整块、一趟出整块再补两刀。</b><br><em>例句和概率是示意。</em>'),
    "__FIG_S11_TREE__": ("fig-s11-tree", "fig6-spec-tree.svg", "topic06-fig-spec.py",
        '<b>看草稿从哪来。</b>'),
}

NUM_LOCK = []
