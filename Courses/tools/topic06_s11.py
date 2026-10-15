# -*- coding: utf-8 -*-
"""专题六 · 第十一节「投机解码：搬一趟，出好几个字」。图在 topic06-fig-spec.py。

出处（行号见图脚本 docstring）：K3 两本 runbook 与 PD-BACKLOG 在 gpu/inference/a4x-max/kimi-k3/；
V4 两本在 gpu/inference/a4x-max/deepseek-v4/；B200 在 gpu/inference/a4/deepseek-v4/README.vllm.md。
"""

BODY = '''
  <p class="lead">decode 一步把权重搬一趟，只换来一个字。
    投机解码先便宜地猜几个字，再让大模型搬一趟、一次验完，<b>猜对几个就出几个</b>。
    这招是 2022 年底到 2023 年初 Google 和 DeepMind 各自提出的，2024 年底 DeepSeek-V3 把它训进了模型。</p>

  <h3 id="s11-1">11.1　先猜后验</h3>
  <p>大模型把草稿当成一段输入，一步算出每个位置自己想出的字，和草稿逐个对。</p>
__FIG_S11_STEP__
<figure class="fbox fwide" id="anim-spec">
<video src="media/topic06-spec.mp4" autoplay loop muted playsinline
       aria-label="投机解码一步的动画。标题：投机解码：小模型先猜 4 个，大模型一趟验完。三行，分别标着已写下、小模型猜、大模型验；底下一条用时条，图例写：紫＝小模型猜一个（便宜），灰＝大模型搬一趟权重（贵）。开始时已写下「床前明月」。字幕依次是：第 1 步：小模型一个接一个猜 4 个字，每个都很快（猜出「光疑似地」，用时条多 4 小格紫色）。大模型搬一趟权重，同时算出这 5 个位置它自己会写的字（一个蓝框标着「一趟」，里面是「光、疑、是」和两个问号，用时条多一大格灰色）。从左往右逐个比：一样就收下（光、疑变绿）。第 3 个不一样：换成大模型自己的字，后面的全丢掉（「似」变红，「是」被黄框圈出，后面变暗；小字注：？＝前面那个字已经错了，这个位置算了也作废）。「光疑是」飞上去，第 1 步完成：猜对 2 个 ＋ 大模型补 1 个 ＝ 出 3 个字，停住。第 2 步：小模型猜「地上霜举」，大模型一趟算出「地上霜举头」；4 个全对；第 5 个位置大模型本来就算了，白送 1 个。「地上霜举头」飞上去，第 2 步完成：猜对 4 个 ＋ 白送 1 个 ＝ 出 5 个字，停住。最后：猜对几个就出几个，再加大模型自己的 1 个：每步至少出 1 个。画面复位。"></video>
<figcaption>看碰到第一个错字之后，大模型怎么接手。
  <span class="sub">（23 秒无声循环，Manim 渲染。示意，按温度 0 贪心比对。）</span></figcaption></figure>
  <p>按标准拒绝采样（按概率决定每个草稿字收不收）实现时，投机解码只改速度，输出分布不变。</p>

  <h3 id="s11-2">11.2　收益账：接受长度 ÷ 一步变慢了多少</h3>
  <p>每步平均出几个字叫接受长度。小 batch 时一步主要在读权重，多验几个 token 本该几乎不加时间。</p>
__FIG_S11_BYTES__
  <p>MoE 不白送：每个 token 挑自己的专家，按 V3 的结构算，验 8 个要读的字节是验 1 个的 4.4 倍（本课推导，均匀路由，偏高）。</p>
  <p>Kimi K3 在 GB300 上实测，开投机后每字间隔从 11.1 ms 降到 2.31 ms。
    每步接受 6.96 个，一步慢了 1.45 倍，净赚 4.81 倍。</p>

  <h3 id="s11-3">11.3　收益会缩：并发、上下文、DEP</h3>
  <p>并发一高，计算单元先忙满，草稿里的每个 token 都得真算一遍。
    K3 输出吞吐的加速比从并发 1 的 4.22× 降到并发 64 的 1.74×。它低于 4.81 倍，因为端到端含不受加速的 prefill。</p>
__FIG_S11_CONC__
  <p>上下文变长也会缩：并发 1 时 prompt 从 4K 拉到 32K，加速比掉到 1.32×。
    不开投机时每字间隔只从 11.1 ms 涨到 11.4 ms，多读的 KV 不怎么费时。</p>
  <p>慢在验证本身：K3 的全注意力是 MLA，所有 query 头共用一个 KV 头，每读 1 字节本来就要算很多次；一次验 7 个字、上下文又长，就撞上算力墙。草稿读长 KV、接受率下降也有份，各占多少没拆开测（本课推导）。</p>
  <p>DEP（第六节）本来就是给大 batch 配的。B200 上 DeepSeek-V4-Flash 用 vLLM、DEP8 开 MTP，十档并发全变慢，−3% 到 −37%。</p>
  <p>峰值和端到端还可能反向：GB300 上 V4-Pro 把 MTP 从 1 步加到 2 步，decode 峰值掉 13%，端到端反涨 28%。真实负载下 batch 多数时间没那么大，读权重的便车还在。</p>

  <h3 id="s11-4">11.4　温度：压测怎么发，结果就怎么变</h3>
  <p>草稿怎么抽样，要和请求的温度对上。贪心草稿遇上温度 1.0 的随机采样，理论接受率等于大模型给这个 token 的概率：会下降，但不归零。</p>
__FIG_S11_TEMP__
  <p>GB300 上 V4-Pro 的 DSpark 草稿是贪心的，服务端默认温度 1.0。
    压测工具不发温度，接受率只剩 1.16%，输出 944 tok/s；发温度 0 是 34%、2,911 tok/s，差 3.1 倍。</p>
  <p>1.16% 比理论值还低得多：它是收下的 ÷ 猜的，每步猜 7 个，第 1 个被拒，后 6 个跟着作废（本课推导，各因素没拆）。</p>
  <div class="note warn"><span class="t">接受率掉了不会报错</span>
    它只让 TPOT 慢慢变差。线上要把接受率和接受长度当一级指标告警；跨框架比，先对齐两边发不发 temperature。</div>

  <h3 id="s11-5">11.5　草稿从哪来</h3>
  <p>DeepSeek-V3 的 MTP 是挂在主干后面的一整层 transformer，共享词表和输出头。
    EAGLE-3（2025-03）融合大模型浅、中、深三层的状态，batch 64 时吞吐仍高 1.38 倍（SGLang）。最新一步是 2026 年 DeepSeek 的 DSpark：半自回归（普通 decode 每个字要等前一个，它一次出一小段），按置信度决定验几个。</p>
__FIG_S11_TREE__
  <p>不要模型的那一支在改代码这类重复多的场景命中最多。扩散草稿（DiffuSpec、DFlash）一次并行出一整块，是最新的一支。</p>

  <h3 id="s11-6">11.6　接进系统的代价</h3>
  <ul>
    <li><b>草稿要占显存。</b>K3 开 DSpark 后 KV 池从 787,504 个 token 缩到 637,504 个，少了 19%。</li>
    <li><b>引擎默认值会跟着变。</b>K3 的 SGLang 配方里，开投机又没写 max-running-requests 时，它被重设成 48，batch 上限悄悄压低。</li>
    <li><b>PD 碰上投机解码。</b>刚交接过来的请求只算 1 个 token，别人是 1＋N，批次形状不齐就用不上 CUDA Graph（9.3 讲过）；decode 端补齐到 1＋N 后，TPOT 约 40 ms 降到 22 ms（vLLM GLM-5.2 PD 博客）。</li>
  </ul>

  <div class="note ok"><span class="t">回到主线</span>
    <b>投机解码让搬一趟权重换来好几个字</b>。batch 小、读权重占大头时赚得最多；batch 一大、算力吃满，它就从白送变成纯开销。</div>
'''

FIGS = {
    "__FIG_S11_STEP__": ("fig-s11-step", "fig6-spec-step.svg", "topic06-fig-spec.py",
        '<b>猜错只浪费草稿那点功夫，每轮至少前进 1 个。</b><br>'
        '<em>示意时间线，格子长短不按比例。</em>'),
    "__FIG_S11_BYTES__": ("fig-s11-bytes", "fig6-spec-bytes.svg", "topic06-fig-spec.py",
        '<b>看左边：稠密部分验几个都只读一遍，专家却越验越多。</b><br><em>左为本课推导（FP8、均匀路由）；右为 K3 在 GB300 上 bs=1 的实测时间账。</em>'),
    "__FIG_S11_CONC__": ("fig-s11-conc", "fig6-spec-conc.svg", "topic06-fig-spec.py",
        '<b>看柱子往右怎么缩。</b><br><em>K3 输出 tok/s 总量；B200 与 V4-Pro 的对照见出处。</em>'),
    "__FIG_S11_TEMP__": ("fig-s11-temp", "fig6-spec-temp.svg", "topic06-fig-spec.py",
        '<b>看温度 1.0 那侧：草稿猜对也常被判错。</b><br><em>V4-Pro 为 4K/1K 并发 256 实测；五个候选字是举例。</em>'),
    "__FIG_S11_TREE__": ("fig-s11-tree", "fig6-spec-tree.svg", "topic06-fig-spec.py",
        '<b>草稿一路往大模型身上长；另有一支不要草稿模型，直接从提示里抄。</b><br><em>各家加速倍数不可横比，图上不画。</em>'),
}

NUM_LOCK = [
    ("6.96", "fig6-spec-bytes.svg"),
    ("1.45", "fig6-spec-bytes.svg"),
    ("4.81", "fig6-spec-bytes.svg"),
    ("4.4", "fig6-spec-bytes.svg"),
    ("4.22", "fig6-spec-conc.svg"),
    ("1.74", "fig6-spec-conc.svg"),
    ("1.32", "fig6-spec-conc.svg"),
    ("1.16%", "fig6-spec-temp.svg"),
    ("2,911", "fig6-spec-temp.svg"),
    ("944", "fig6-spec-temp.svg"),
]
