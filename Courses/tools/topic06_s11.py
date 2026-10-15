# -*- coding: utf-8 -*-
"""专题六 · 第十一节「投机解码：搬一趟，出好几个字」。图在 topic06-fig-spec.py。

出处（行号见图脚本 docstring）：K3 两本 runbook 与 PD-BACKLOG 在 gpu/inference/a4x-max/kimi-k3/；
V4 两本在 gpu/inference/a4x-max/deepseek-v4/；B200 在 gpu/inference/a4/deepseek-v4/README.vllm.md。
"""

BODY = '''
  <p class="lead">decode 一步把权重搬一趟，只换来一个字。
    <b>投机解码先便宜地猜几个字，再让大模型搬一趟、一次验完，猜对几个就出几个。</b></p>

  <h3 id="s11-1">11.1　先猜后验</h3>
  <p>草稿可以来自小模型、挂在大模型上的预测头，或者干脆从 prompt 里抄。
    大模型把草稿当成一段输入，一步算出每个位置自己想出的字，和草稿逐个对。</p>
__FIG_S11_STEP__
  <p>收下第 i 个草稿的概率是 min(1, p/q)，p 是大模型给的概率、q 是草稿给的。
    被拒时从 p 减 q 剩下的那部分里重抽一个，整体输出分布和大模型自己一个个生成完全一样（Leviathan 等 2022；Chen 等 2023）。</p>
  <p>所以投机解码是无损的：它改的是速度，不改答案的分布。
    温度为 0 时这条规则退化成「草稿和大模型的贪心选择一致才收下」。</p>

  <h3 id="s11-2">11.2　收益账：接受长度 ÷ 一步变慢了多少</h3>
  <p>每步平均出几个字叫接受长度。加速比约等于接受长度除以一步变慢的倍数。
    小 batch 时一步的时间主要花在读权重上，多验证几个 token 本该几乎不加时间。</p>
__FIG_S11_BYTES__
  <p>稠密部分确实如此：注意力和共享权重验证几个都只读一遍。
    MoE 不白送，每个 token 挑自己的专家，按 V3 的结构粗算，验证 8 个要读的字节是验证 1 个的 4.4 倍（本课推导，均匀路由，偏高）。</p>
  <p>Kimi K3 在 GB300 上实测，开投机后每字间隔从 11.1 ms 降到 2.31 ms。
    每步接受 6.96 个，一步其实慢了 1.45 倍，净赚 4.81 倍（本课推导：2.31 × 6.96 ≈ 16.1 ms）。</p>

  <h3 id="s11-3">11.3　收益会缩：并发、上下文、EP</h3>
  <p>并发一高，计算单元先忙满，草稿里的每个 token 都得真算一遍。
    K3 输出吞吐的加速比从并发 1 的 4.22× 降到并发 64 的 1.74×。</p>
__FIG_S11_CONC__
  <p>上下文变长也会缩。同样并发 1，K3 的 prompt 从 4K 拉到 32K，加速比从 4.22× 掉到 1.32×；注意力读 KV 的时间变长了，草稿省下的那部分占比就小了。</p>
  <p>EP+DP 本来就是给大 batch 配的。B200 上 DeepSeek-V4-Flash 用 vLLM、EP+DP8 开 MTP，十档并发全部变慢，从 −3% 到 −37%。</p>
  <p>峰值和端到端也可能方向相反。GB300 上 V4-Pro 把 MTP 从 1 步加到 2 步，decode 峰值掉 13%，端到端反而涨 28%：真实负载下 batch 多数时间不在顶点，那里还是读权重的便车在起作用。</p>

  <h3 id="s11-4">11.4　温度：压测怎么发，结果就怎么变</h3>
  <p>草稿怎么抽样，要和请求的温度对上。贪心草稿遇上温度 1.0 的随机采样，几乎每个都会被拒。</p>
__FIG_S11_TEMP__
  <p>GB300 上 V4-Pro 的 DSpark 草稿是贪心的，服务端默认温度 1.0。
    压测工具不发温度，接受率只有 1.16%、输出 944 tok/s；发温度 0 是 34%、2,911 tok/s，差 3.1 倍。</p>
  <p>K3 的草稿按概率抽，官方就用温度 1.0 压测，接受长度在代码类任务约 4.73、创作类约 2.61。</p>
  <div class="note warn"><span class="t">接受率掉了不会报错</span>
    它只让 TPOT 慢慢变差。线上要把接受率和接受长度当一级指标告警；跨框架比投机解码，先对齐两边发不发 temperature。</div>

  <h3 id="s11-5">11.5　草稿从哪来</h3>
  <p>谱系一条主线是草稿越来越贴近大模型：独立小模型、草稿树（SpecInfer）、预测头（Medusa）、读隐状态（EAGLE）、训进模型的 MTP（DeepSeek-V3），到 2026 年的 DSpark。</p>
__FIG_S11_TREE__
  <p>另一支不要模型。Prompt Lookup 从 prompt 里抄，SuffixDecoding 查历史输出，在重复率高的场景命中最多，比如改代码、长文摘要、agent 循环调用工具。
    扩散草稿（DiffuSpec、DFlash）一次并行出一整块，是最新的一支。</p>

  <h3 id="s11-6">11.6　接进系统的代价</h3>
  <p>草稿要占显存。K3 开 DSpark 后 KV 池从 787,504 个 token 缩到 637,504 个，少了 19%。</p>
  <p>引擎的默认值也会跟着变。K3 的 SGLang 配方里，开投机又没写 max-running-requests 时，它会被重设成 48；batch 上限被悄悄压低，吞吐测出来就不对。</p>
  <p>线性注意力要能回滚。KDA 这类层把历史压成一个状态，草稿被拒就得退回去；K3 的做法是只存草稿窗口的原始输入、拒了就重放，窗口占用从 512 KB 降到约 16 KB。</p>
  <p>EP 的通信缓冲按 token 数开。SGLang 跑 V4-Pro 时要满足 max-running-requests × 草稿数 ≤ DeepEP 每个 rank 的派发上限，否则开不起来。</p>
  <p>还有 kernel 和编译。验证一步是「一个请求带好几个 token」的新形状，prefill、decode、验证三套注意力后端要一起选对；PD 分离下混合 batch 还会掉出整图 CUDA Graph。</p>

  <div class="note ok"><span class="t">回到主线</span>
    <b>投机解码让搬一趟权重换来好几个字</b>。batch 小、读权重占大头时赚得最多；batch 一大、算力吃满，它就从白送变成纯开销。</div>
'''

FIGS = {
    "__FIG_S11_STEP__": ("fig-s11-step", "fig6-spec-step.svg", "topic06-fig-spec.py",
        '<b>猜错只浪费草稿那点功夫，每轮至少前进 1 个。</b><br>'
        '<em>示意时间线，格子长短不按比例。</em>'),
    "__FIG_S11_BYTES__": ("fig-s11-bytes", "fig6-spec-bytes.svg", "topic06-fig-spec.py",
        '<b>稠密部分验几个都只读一遍，专家却越验越多；K3 每步多花 45% 的时间，换来 6.96 个字。</b><br>'
        '<em>左为本课推导（FP8、均匀路由）；右图红点为 GB300 实测、bs=1。</em>'),
    "__FIG_S11_CONC__": ("fig-s11-conc", "fig6-spec-conc.svg", "topic06-fig-spec.py",
        '<b>并发越高、上下文越长，收益越小；EP+DP 高吞吐配置下全部变负。</b><br>'
        '<em>三块面板硬件和模型都不同，只看各自的趋势；K3 是输出 tok/s 总量。</em>'),
    "__FIG_S11_TEMP__": ("fig-s11-temp", "fig6-spec-temp.svg", "topic06-fig-spec.py",
        '<b>同一套服务，只差请求里带不带温度 0，吞吐差 3.1 倍。</b><br>'
        '<em>V4-Pro 为 4K/1K 并发 256 实测；K3 接受长度为官方参考值。</em>'),
    "__FIG_S11_TREE__": ("fig-s11-tree", "fig6-spec-tree.svg", "topic06-fig-spec.py",
        '<b>草稿一路往大模型身上长，另有不要模型和扩散两支。</b><br>'
        '<em>年份按 arXiv 首版；各家加速倍数不可横比，图上不画。</em>'),
}

NUM_LOCK = [
    ("6.96", "fig6-spec-bytes.svg"),
    ("1.45", "fig6-spec-bytes.svg"),
    ("4.81", "fig6-spec-bytes.svg"),
    ("4.4", "fig6-spec-bytes.svg"),
    ("4.22", "fig6-spec-conc.svg"),
    ("1.74", "fig6-spec-conc.svg"),
    ("1.32", "fig6-spec-conc.svg"),
    ("37%", "fig6-spec-conc.svg"),
    ("1.16%", "fig6-spec-temp.svg"),
    ("2,911", "fig6-spec-temp.svg"),
    ("944", "fig6-spec-temp.svg"),
    ("4.73", "fig6-spec-temp.svg"),
    ("2.61", "fig6-spec-temp.svg"),
]
