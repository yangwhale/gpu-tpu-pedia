# -*- coding: utf-8 -*-
"""专题六 · 第七节「切序列：DCP 和 PCP」。图在 topic06-fig-cp.py；另复用专题五三张图。"""

BODY = '''
  <p class="lead">第六节切的是头和请求，一条序列变长时还剩两件事。DEP 起不来时（如 6.1 的 K3）只能用 TP，上下文一长，KV 被复制几份，卡很快就满；prompt 一长，第一个字就要等很久。
    <b>DCP 管前一件，PCP 管后一件，两者都是沿序列切。</b></p>

  <h3 id="s7-1">7.1　DCP：只在长上下文赚，开之前先知道坑</h3>
  <p><a href="topic-05.html#s5-4">专题五 §5.4</a> 讲过：KV 头比 TP 路数少时，DCP 把 KV 按 token 轮流存到这几张卡上，合并时再统一除 softmax 的分母。轮流放而不是前一半后一半，是因为新字不停写进来，轮流放各卡才能一直一样多（本课的解释）。</p>
__FIG_S7_KVDUP_T5__
__FIG_S7_MERGE_T5__
  <p>代价是每层多几次通信（vLLM 默认后端每层 3 次），度数（切成几份，DCP8 就是 8 份）最多开到 TP ÷ KV 头数。</p>
  <details class="foldfig"><summary><b>每层那 3 次是什么</b>（vLLM 默认 <code>ag_rs</code> 后端）</summary>
  <ul>
    <li><b>① all-gather 提问</b>：权重按头切，每卡只投影出自己那几个头，可本地那段 KV 所有头都要查，先收齐全部头。</li>
    <li><b>② all-gather LSE</b>：kernel 在本地已经除过分母，只留下 log-sum-exp（每人每头一个数），收齐后各卡算出自己该占的比例。所以「先别除」是课堂说法，实现是「除了，但把分母记着」。</li>
    <li><b>③ reduce-scatter 输出</b>：相加得完整结果，按头分回各卡，接上按头切的输出投影。</li>
  </ul>
  <p>代价主要是等不是量：V3 61 层 MLA × 3 ＝ 每 token 多 183 次集合通信，K3 24 层 × 3 ＝ 72 次（本课按层数乘）。<code>a2a</code> 后端把 ②③ 并成一次 all-to-all（源码注释：MLA 每层 3 次降到 2 次）；<code>dcp_q_replicate</code> 每卡放一整份提问投影省掉 ①，代价是重复算。两者没有公开实测。</p></details>

  <p><b>DCP 只在长上下文赚。</b>Kimi K3 在 GB300 上，<b>128K 输入时 DCP8 大赚</b>；输入 4K 时各档并发都亏。</p>
__FIG_S7_GAIN__
  <ul>
    <li><b>为什么长了才赚：TP8 放不下，只能少收请求</b>。128K 乘 8 并发要 100 多万 token，TP8 默认配置实测只装得下约 79 万（这组没开投机，开了还少两成）；官方说 DCP8 把整组卡能存的 KV 从 1.5M token 撑到 12.2M。差别落在首字上：首字中位数 TP8 调得最好的一档要半分钟（32.1 s，默认档 39.3 s），DCP8 是 2.3 s（都含排队）。</li>
    <li><b>为什么短了亏</b>：KV 本来放得下，DCP 只剩代价。runbook 发现 4K 亏得最多的 −26%，正好等于关掉 symm-mem（symmetric memory：各卡在显存里留一块同形的区域，互相直接写对方那块，给小消息的 all-reduce 省掉来回握手；K3 一个人用时开它 87.8、关它 64.9 tok/s）的损失，而开 DCP 时 symm-mem 用不了：SGLang 对 K3 一开 DCP 就强制关它，理由是 decode 的 CUDA Graph 会算错（模型设置，不是 DCP 本身的限制）。</li>
    <li><b>看的是总量</b>：同是 8K，并发 8 还亏，并发 32 已赚。盈亏点看「输入长度 × 并发」压出的 KV 总量。</li>
    <li><b>每个字的间隔：DCP 越长越不吃亏，可起点高一截</b>。TP8 每张卡都读这条请求的整份 KV，DCP8 每张卡只读 1/8，代价是每层多几次通信。并发 1、输入从 4K 拉到 128K，TP8 每 token 跟着涨（11.1 → 12.0 ms），DCP8 几乎不动（15.2 → 15.3 ms）。高出的那约 4 ms 基本是 symm-mem 被强制关掉的损失：同样 4K、关掉 symm-mem 的 TP8 是 15.1 ms（本课按两行数推）。K3 只有 24 层 MLA，所以它主要赚在装得下。</li>
    <li><b>官方口径是容量不是速度</b>：官方大规模预设 Peak Capacity（加 <code>--dcp-size 8</code>）写的是并发上限 +72%、引擎吞吐不变、ITL ×1.8。别说成「DCP 让每个字更快」。</li>
  </ul>
  <p><b>多长时注意力算得比专家多？</b>拿 Qwen3.5-397B 算一笔（本课推导）：</p>
  <ul>
    <li><b>专家不随长度变</b>：每 token 每层 10 个路由专家 ＋ 1 个共享专家，每个 3 个 4096 × 1024 矩阵，约 2.77 亿次浮点运算。</li>
    <li><b>注意力随长度涨</b>：32 个查询头 × 256 维，打分、加权两步，每 token 每层约 32,768 × 上下文长度。</li>
    <li><b>交叉点</b>：单看一层全注意力，decode 约 8.4K，prefill 因果平均只看一半，约 16.9K。60 层里只有 15 层全注意力，全模型再乘 4：decode 约 34K、prefill 约 68K。线性注意力层每 token 算量固定，不进这笔账。</li>
    <li><b>它不是开 CP 的门槛</b>：decode 拼的是读字节，专家权重整批共用、KV 各读各的，注意力早就是大头。DCP 看 KV 放不放得下，PCP 看首字等不等得起（7.2）。</li>
  </ul>
  <p><b>名字按阶段起。</b>DCP／PCP 是推理框架里的叫法。Meta 论文（arXiv 2411.01783）按传什么叫 pass-KV／pass-Q：完整 prefill 用 pass-KV，decode 和缓存命中高的 partial prefill 用 pass-Q。NVIDIA Helix 论文（arXiv 2507.07120）管 decode 时按序列切 KV 的那一派叫 KV 并行（KVP）。</p>

  <p><b>开 DCP 之前要知道的坑。</b></p>
  <ul>
    <li><b>缓存层没跟上</b>：SGLang 的 HiCache（4.4）开 L3（远端存储），或 L1＋L2 叠投机解码，就得退回纯 TP。</li>
    <li><b>状态池它不管</b>：K3 的 KDA 状态池 DCP 不切（6.1）。</li>
    <li><b>官方 PD 分离的吞吐数反而是 TP8 略高</b>：1P1D、decode 用 TP8 是 2,808 tok/s，2P2D、decode 用 DCP8 是 2,633（P:D 都是 1:1；分母和输入长度官方都没写，不能和 7.1 比）。</li>
  </ul>

  <h3 id="s7-2">7.2　PCP：超长 prompt 的首字延迟</h3>
  <p>PCP 在 prefill 阶段沿序列切，<b>让几台机器合算同一个 prompt</b>（<a href="topic-05.html#s5-5">专题五 §5.5</a>）。</p>
__FIG_S7_PCPDCP_T5__
  <p>Meta 2024 年 11 月的论文把训练侧的切法搬到推理 prefill：一百多张卡合算一个 prompt，一百万 token 一分多钟算完（128 张 H100、Llama3 405B，1M token 77 s，128K 3.8 s，并行效率 93%）。用的是 <b>Ring</b>（2023-10）：各卡拿一段，别人那段沿环一站站传来，边传边算。整段 prefill 传 K、V（GQA 下它小），新 token 很少时改传 Q，结果都和不切一样。 为了让因果三角各卡活一样多，Meta 把序列切成 2N 块、首尾配对分给 N 张卡；DCP 则按 token 轮流放。</p>
  <p>同一篇里，1 到 8 台之间机器数翻倍，prefill 延迟约减半；再翻到 16 台就打了折扣，只降约 35%（8 台做 128K 5.85 s，16 台 3.8 s，降幅本课推导）。跨机切 TP 就不行：2 台时 TP16 比 CP2 慢约 15%，8 台慢约 100%。原文的解释：TP 每层都要走跨机带宽，而它比机内低得多。</p>
  <p>另有 Ulysses（DeepSpeed，2023-09）用 AllToAll（每张卡给其余每张卡各发一份）把按序列切换成按头切，切的份数不能超过头数。</p>

  <p><b>PCP 的两个正确性陷阱。</b>PCP 出错往往不报错，形状全对、数不对。</p>
  <ul>
    <li><b>少算会「更快」</b>：某个 rank 漏了别段的 K、V，算量少了，速度自然好看。快得反常，先核输出。</li>
    <li><b>线性注意力的状态要接力</b>：softmax 注意力没有跨段状态，拿齐 K、V 就能算对。GDN、KDA 这类层带着一份随 token 更新的状态，切开后每个 rank 若从空状态起步，就全错了。</li>
  </ul>
__FIG_S7_LINEAR__
  <p>修法是把前一段的末状态传过来；或者利用结合律：各段并行算好自己对状态的映射，再从头合并出每段的起始状态。</p>

  <h3 id="s7-3">7.3　并排看：副本让出来，多放 token</h3>
  <p>把第六、七节的切法摆进同一本账：V3，8 颗 v7（16 个 device，v7x-16），上下文 128K。每格按 device 算，口径同 4.1、10.2：每 device 可用 94.75 GiB，留一成给激活值，剩 85.3 GiB；权重 FP8 每参数 1 字节，KV 按 bf16，一条 128K 是 8.58 GiB。</p>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">本课推导，纸面账。非专家权重 ＝ 总参数 671.0B 减去 58 层 × 256 个路由专家（653.9B）＝ 17.1B；每层参数分项见 0.2 的图。实测 FP8 专家比均摊更占地方（10.2），真机余量只会更少。</caption>
    <tr><th>切法</th><th>权重／device</th><th>剩给 KV</th><th>一条 128K 在每 device 占</th><th>16 个 device 装几条</th></tr>
    <tr><td>DP：整套复制</td><td>625 GiB</td><td>放不下</td><td>—（一份权重就超了）</td><td>0</td></tr>
    <tr><td>TP16：注意力、专家都切 16 片</td><td>39.1 GiB（全切）</td><td>46.2 GiB</td><td>8.58 GiB（每 device 一整份）</td><td>5</td></tr>
    <tr><td>TEP16：注意力 TP、专家 EP</td><td>39.1 GiB（全切）</td><td>46.2 GiB</td><td>8.58 GiB（每 device 一整份）</td><td>5</td></tr>
    <tr><td>DEP16：注意力 DP、专家 EP</td><td>54.0 GiB（非专家整份 15.9 ＋ 路由专家 1/16 是 38.1）</td><td>31.3 GiB</td><td>8.58 GiB（只在自己那个 device）</td><td>每 device 3 条，共 48</td></tr>
    <tr><td>TEP16 ＋ DCP16</td><td>39.1 GiB（全切）</td><td>46.2 GiB</td><td>0.54 GiB（按 token 切 16 份）</td><td>86</td></tr>
  </table>
  <p>读的一侧取装满时的 batch，每层碰到几个专家按 1.2 的均匀路由算；只算从 HBM 读的字节，下限按每 device 3.685 TB/s（每芯片 7.37 ÷ 2）。</p>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">本课推导。读权重取 MoE 层（第 4 到 61 层），每 device 每层；一步下限含 61 层与输出层，不含通信和计算。通信次数：TP 前向每层 2 次 AllReduce（Megatron-LM，arXiv 1909.08053）；TEP 同 2 次（专题五 §4.4 的图）；DEP 专家前后各 1 次 all-to-all（6.4）；DCP 另加每层 3 次（vLLM 默认后端，7.1）。</caption>
    <tr><th>切法</th><th>整套 batch</th><th>每层读权重</th><th>每层读 KV</th><th>一步读完下限</th><th>每层通信</th><th>代价／适用</th></tr>
    <tr><td>TP16</td><td>5</td><td>118 MB：注意力 1/16 ＋ 碰到的 37.6 个路由专家和共享专家各 1/16</td><td>755 MB：5 条全读</td><td>14.4 ms</td><td>2 次 AllReduce</td><td>单人出字快；KV 复制 16 份，长上下文装不下人</td></tr>
    <tr><td>TEP16</td><td>5</td><td>118 MB（平均）：注意力 1/16 ＋ 本卡被碰到的整块专家</td><td>755 MB：5 条全读</td><td>14.4 ms</td><td>2 次 AllReduce</td><td>同 TP16；专家整块，热门专家那张卡要多读，大家等它</td></tr>
    <tr><td>DEP16</td><td>48</td><td>784 MB：整份注意力 187 ＋ 本卡 12.5 个路由专家 ＋ 共享专家</td><td>453 MB：只读自己 3 条</td><td>20.6 ms</td><td>2 次 all-to-all（前 3 层不用）</td><td>KV 不复制，人多时最省；每 rank 一份状态，K3 这类起不来（6.1）</td></tr>
    <tr><td>TEP16 ＋ DCP16</td><td>86</td><td>673 MB：注意力 1/16 ＋ 本卡 15 个整块专家</td><td>812 MB：86 条各读 1/16</td><td>24.1 ms</td><td>2 次 AllReduce ＋ 3 次</td><td>装得最多；通信最勤，只在长上下文划算（7.1）</td></tr>
  </table>
  <ul>
    <li><b>TP 和 TEP 只差在专家</b>（切片还是整块）：MLA 只有一份 KV，按头切不开，每个 device 都存、都读全部 5 条（6.1）。</li>
    <li><b>DEP 多背 14.9 GiB 非专家权重</b>（54.0 − 39.1，比全切多出来的那份；表里 15.9 GiB 是非专家权重整份），换来 KV 不复制：48 条，以 TP16 的 5 条为基线约 9.6 倍；代价是每层整读一遍注意力权重。</li>
    <li><b>DCP 每卡只存、只读每条的 1/16</b>，装 86 条，代价是每层多 3 次通信。DEP 上叠不了：度数最多 TP ÷ KV 头数，DEP 的 TP 是 1（7.1）。</li>
    <li><b>下限只看了读</b>，谁真赢要把通信那一列也算进去。</li>
    <li><b>和 10.2 对账</b>：那张图说 8 颗装得下 128K × 64，前提是 KV 不复制、非专家权重不重复。DEP 只到 48 条，TEP16 ＋ DCP16 到 86 才兑现。</li>
  </ul>

  <p><b>落点：副本让出来，多放 token。</b>DCP 让出 KV 副本，<b>同样的显存多放 token，这一趟就分给了更多 token；</b>PCP 买的是首字延迟。留下的问题：一张卡上同时出字的人越多，长 prompt 进门那一步耽误的人也越多（第八节）。</p>

  <details class="foldfig"><summary><b>考考自己</b>：4K 输入、并发 8 的线上服务，要不要开 DCP8？换成 128K 的 agent 长对话呢？</summary>
  <p>4K 不开：KV 放得下，DCP 只剩代价（7.1）。128K 开：TP8 放不下，只能少收请求、排长队。</p></details>
'''

FIGS = {
    "__FIG_S7_KVDUP_T5__": ("fig-s7-kv-dup-t5", "fig5-kv-dup.svg", "topic05-fig-seq.py",
        '<b>同一个请求，TP8 存 8 份，DCP8 合起来只存 1 份。</b><br><em>专题五原图，以 V3 一个 128K 请求为例。</em>'),
    "__FIG_S7_MERGE_T5__": ("fig-s7-softmax-merge-t5", "fig5-softmax-merge.svg", "topic05-fig-seq.py",
        '<b>分块算注意力，合并时先别除。</b><br><em>每张卡除了结果，还要交出自己的分母。</em>'),
    "__FIG_S7_GAIN__": ("fig-s7-dcp-gain", "fig6-dcp-gain.svg", "topic06-fig-cp.py",
        '<b>看突出画的并发 8 那条在哪个长度翻上去。</b><br>'
        '<em>输出吞吐，取预热后那轮；TP8 取调过参的最好一档。纵轴为对数刻度。</em>'),
    "__FIG_S7_PCPDCP_T5__": ("fig-s7-pcp-dcp-t5", "fig5-pcp-dcp.svg", "topic05-fig-seq.py",
        '<b>PCP 切 prefill 的计算，DCP 切 decode 的 KV。</b><br><em>专题五原图。</em>'),
    "__FIG_S7_LINEAR__": ("fig-s7-pcp-linear", "fig6-pcp-linear.svg", "topic06-fig-cp.py",
        '<b>各扫各的，锯齿；接力扫，平滑。</b><br>'
        '<em>本课自造的玩具递推，真实的状态是矩阵、带门控，递推形状相同。</em>'),
}

NUM_LOCK = [
    ("26%", "fig6-dcp-gain.svg"),
]

# ── 7.3 两张表的数（本课推导；改表先改这里，跑不过就说明表和推导对不上）──────
def _t76():
    GIB, MB = 1024 ** 3, 1e6
    H, D = 7168, 16                                   # V3 hidden；8 颗 v7 ＝ 16 个 device
    ATTN, DENSE, EXP = 187105280, 3 * H * 18432, 3 * H * 2048   # 每层参数，推导见 topic06-fig-layerwalk.py
    ROUTER, LM = 256 * H, 129280 * H
    TOTAL = 61 * ATTN + 3 * DENSE + 58 * (257 * EXP + ROUTER) + 2 * LM
    ROUTED = 58 * 256 * EXP
    NONR = TOTAL - ROUTED
    USE = 94.75 * GIB * 0.9                           # 每 device 可用 94.75 GiB，留一成（4.1、10.2 同口径）
    SEQ, SEQL = 70272 * 131072, 1152 * 131072          # 一条 128K：全层／每层
    assert round(TOTAL / GIB) == 625 and round(NONR / 1e9, 1) == 17.1 and round(NONR / GIB, 1) == 15.9
    w_tp, w_dep = TOTAL / D, NONR + ROUTED / D
    assert (round(w_tp / GIB, 1), round(w_dep / GIB, 1), round(ROUTED / D / GIB, 1)) == (39.1, 54.0, 38.1)
    p_tp, p_dep = USE - w_tp, USE - w_dep
    assert (round(USE / GIB, 1), round(p_tp / GIB, 1), round(p_dep / GIB, 1)) == (85.3, 46.2, 31.3)
    n_tp, n_dep, n_dcp = int(p_tp // SEQ), int(p_dep // SEQ) * D, int(D * p_tp // SEQ)
    assert (n_tp, n_dep, n_dcp) == (5, 48, 86) and round(SEQ / D / GIB, 2) == 0.54
    assert round(n_dep / n_tp, 1) == 9.6 and round((w_dep - w_tp) / GIB, 1) == 14.9
    hit = lambda b: 256 * (1 - (1 - 8 / 256) ** b)   # 1.2 的均匀路由
    BW = 7.37e12 / 2
    rows = {}
    for k, b in (("tp", n_tp), ("dep", n_dep), ("dcp", n_dcp)):
        if k == "dep":
            a, m, d, kv, lm = ATTN, hit(b) / D * EXP + EXP + ROUTER, DENSE, b / D * SEQL, LM
        else:
            a, m, d, lm = ATTN / D, hit(b) / D * EXP + EXP / D + ROUTER / D, DENSE / D, LM / D
            kv = b * SEQL if k == "tp" else b * SEQL / D
        step = 61 * (a + kv) + 3 * d + 58 * m + lm
        rows[k] = (round(hit(b), 1), round((a + m) / MB), round(kv / MB), round(step / BW * 1e3, 1))
    assert rows == {"tp": (37.6, 118, 755, 14.4), "dep": (200.2, 784, 453, 20.6),
                    "dcp": (239.3, 673, 812, 24.1)}, rows
    assert round(hit(n_dep) / D, 1) == 12.5 and round(hit(n_dcp) / D) == 15


_t76()
