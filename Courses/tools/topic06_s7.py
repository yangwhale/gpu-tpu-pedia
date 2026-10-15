# -*- coding: utf-8 -*-
"""专题六 · 第七节「切序列：DCP 和 PCP」。图在 topic06-fig-cp.py；另复用专题五三张图。"""

BODY = '''
  <p class="lead">第六节切的是头和请求，一条序列变长时还剩两件事。DEP 起不来时（如 6.3 的 K3）只能用 TP，上下文一长，KV 被复制几份，卡很快就满；prompt 一长，第一个字就要等很久。
    <b>DCP 管前一件，PCP 管后一件，两者都是沿序列切。</b></p>

  <h3 id="s7-1">7.1　一句回顾</h3>
  <p><a href="topic-05.html#s5-4">专题五 §5.4</a> 讲过：KV 头比 TP 路数少时，DCP 把 KV 按 token 轮流存到这几张卡上，合并时再统一除 softmax 的分母。</p>
__FIG_S7_KVDUP_T5__
__FIG_S7_MERGE_T5__
  <p>代价是每层多几次通信（vLLM 默认后端每层 3 次），度数（切成几份，DCP8 就是 8 份）最多开到 TP ÷ KV 头数。</p>

  <h3 id="s7-2">7.2　DCP 只在长上下文赚</h3>
  <p>Kimi K3 在 GB300 上，<b>128K 输入时 DCP8 大赚</b>；输入 4K 时各档并发都亏。</p>
__FIG_S7_GAIN__
  <ul>
    <li><b>为什么长了才赚</b>：官方说 DCP8 把整组卡能存的 KV 从 1.5M token 撑到 12.2M。实测 TP8 默认配置只装得下约 79 万 token（这组没开投机，开了还少两成），128K 乘 8 并发要 100 多万，放不下，只能少收请求，128K、8 并发下首字延迟中位数 32.1 s，DCP8 是 2.3 s（含排队）。</li>
    <li><b>为什么短了亏</b>：KV 本来放得下，DCP 只剩代价。runbook 发现 4K 亏得最多的 −26%，正好等于关掉 symm-mem（symmetric memory，一种加速卡间通信的显存用法）的损失，而开 DCP 时 symm-mem 用不了。</li>
    <li><b>看的是总量</b>：同是 8K，并发 8 还亏，并发 32 已赚。盈亏点看「输入长度 × 并发」压出的 KV 总量。</li>
  </ul>

  <h3 id="s7-3">7.3　开 DCP 之前要知道的坑</h3>
  <ul>
    <li><b>缓存层没跟上</b>：SGLang 的 HiCache（3.6）开 L3（远端存储），或 L1＋L2 叠投机解码，就得退回纯 TP。</li>
    <li><b>状态池它不管</b>：K3 的 KDA 状态池 DCP 不切（6.3）。</li>
    <li><b>官方 PD 分离的吞吐数反而是 TP8 略高</b>：1P1D、decode 用 TP8 是 2,808 tok/s，2P2D、decode 用 DCP8 是 2,633（P:D 都是 1:1；分母和输入长度官方都没写，不能和 7.2 比）。</li>
  </ul>

  <h3 id="s7-4">7.4　PCP：超长 prompt 的首字延迟</h3>
  <p>PCP 在 prefill 阶段沿序列切，<b>让几台机器合算同一个 prompt</b>（<a href="topic-05.html#s5-5">专题五 §5.5</a>）。</p>
__FIG_S7_PCPDCP_T5__
  <p>Meta 2024 年 11 月的论文把训练侧的切法搬到推理 prefill，用 128 张 H100 给 Llama3 405B 做 prefill：1M token 77 s，128K 3.8 s，并行效率 93%。Meta 用的是 <b>Ring</b>（2023-10）：各卡拿一段，别人那段沿环一站站传来，边传边算；传 K、V 还是传 Q，看谁的通信藏得进计算：整段 prefill 传 K、V（GQA 下它小），新 token 很少时传 Q，结果都和不切一样。</p>
  <p>另有 Ulysses（DeepSpeed，2023-09）用 AllToAll（每张卡给其余每张卡各发一份）把按序列切换成按头切，切的份数不能超过头数。</p>

  <h3 id="s7-5">7.5　PCP 的两个正确性陷阱</h3>
  <p>PCP 出错往往不报错，形状全对、数不对。</p>
  <ul>
    <li><b>少算会「更快」</b>：某个 rank 漏了别段的 K、V，算量少了，速度自然好看。快得反常，先核输出。</li>
    <li><b>线性注意力的状态要接力</b>：softmax 注意力没有跨段状态，拿齐 K、V 就能算对。GDN、KDA 这类层带着一份随 token 更新的状态，切开后每个 rank 若从空状态起步，就全错了。</li>
  </ul>
__FIG_S7_LINEAR__
  <p>修法是把前一段的末状态传过来；或者利用结合律：各段并行算好自己对状态的映射，再从头合并出每段的起始状态。</p>

  <h3 id="s7-6">7.6　落点：副本让出来，多放 token</h3>
  <p>DCP 让出 KV 副本，<b>同样的显存多放 token，这一趟就分给了更多 token；</b>PCP 买的是首字延迟。</p>

  <details class="foldfig"><summary><b>考考自己</b>：4K 输入、并发 8 的线上服务，要不要开 DCP8？换成 128K 的 agent 长对话呢？</summary>
  <p>4K 不开：KV 放得下，DCP 只剩代价（7.2）。128K 开：TP8 放不下，只能少收请求、排长队。</p></details>
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
