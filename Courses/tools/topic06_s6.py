# -*- coding: utf-8 -*-
"""专题六 · 第六节「切专家与注意力：TEP 和 DEP」。图在 topic06-fig-dep.py；另复用专题五两张图。"""

BODY = '''
  <p class="lead">2025 年初 DeepSeek 公开大 EP 部署，每个 H800 节点 decode 约 1.48 万 token/s；两个月后 SGLang 公开复现，对同卡数 TP 最多快 5 倍。专家那一半大家都用 EP，分歧全在注意力：几张卡合算同一批请求，还是各算各的。
    <b>选哪个，看模型有几个 KV 头、你要多大的 batch。</b></p>

  <h3 id="s6-1">6.1　一句回顾</h3>
  <p><a href="topic-05.html#s4-4">专题五 §4.4</a> 讲过：同一批卡，进注意力和进专家时可以用两套切法。
    <b>TEP</b> 是注意力用 TP、专家用 EP；<b>DEP</b> 是注意力用数据并行、专家用 EP。</p>
__FIG_S6_FOLD_T5__
__FIG_S6_TEPDEP_T5__

  <h3 id="s6-2">6.2　判据：KV 头数和 batch</h3>
  <p>TP 按头切注意力。KV 头只有 1 个时切不开，<b>TP 开几路，每个请求的 KV 就存几份</b>。DEP 各存各的，省下的地方用来多收请求。</p>
__FIG_S6_KV__
  <p>DeepSeek-V4-Pro 不是 MLA，它的注意力是 CSA 与 HCA 交错，但每层同样只存一条所有头共享的 KV（<code>num_key_value_heads: 1</code>）。
    所以 TP4 decode 对它是零节省，换成 dep8（TP1 ＋ 注意力 DP8 ＋ EP8）后，同样并发 512，每卡吞吐从 1,319 涨到 2,758，是 <b>2.09 倍</b>。</p>
  <ul>
    <li><b>口径</b>：吞吐是 prompt 加输出的总 token（21,100 和 55,153），分母含 prefill 的卡（16 张和 20 张）。</li>
    <li><b>反过来</b>：8 个 KV 头的 GQA 模型开 TEP8，每卡正好一个头，没有副本，DEP 在 KV 上就不占便宜。</li>
    <li><b>其他模型照这个算</b>：MLA 模型开 TEP8 是 8 倍；Qwen3.5-397B 的全注意力层有 2 个 KV 头，TEP8 是 4 倍。</li>
  </ul>

  <h3 id="s6-3">6.3　DEP 的代价：每个 rank 都得自己备一份</h3>
  <p>DEP 让每个 rank 像一台小推理机：注意力权重、调度器、请求状态各有一份。<b>状态按请求开、又不随 DP 切的东西，DEP 一宽就成倍涨。</b></p>
  <ul>
    <li><b>K3 上 DEP8 起不来</b>：Kimi K3 的 93 层里有 69 层 KDA（一种线性注意力），每个请求一块固定大小的状态。DEP8 让每个 rank 复制一份完整的状态池，两次加载都 OOM；runbook 认为这正是官方在 K3 上改推 TP 和 DCP 的原因。</li>
    <li><b>TP 反而切得开</b>：KDA 的状态按头存，一个头一块。TP 按头切，每张卡只存自己那几个头的状态；DP、EP、DCP 都不动头这一维（本课推导）。同一份 runbook 引官方的话也是这样：能改每卡账单的只有注意力的 TP 宽度。</li>
    <li><b>每 rank 的 batch 会撞上实现上限</b>：SGLang 跑 V4-Pro dep8 时，一个注意力 kernel 写死每 rank 最多 1,024 个请求。实测 batch 到 999 时吞吐已不再涨，算力先到顶，上限没真正卡住。</li>
    <li><b>忙闲不均</b>：各 rank 的请求长短不一，专家那一步要等最慢的 rank；没请求的 rank 也得陪跑一遍空前向（专题五 §4.4）。</li>
  </ul>

  <h3 id="s6-4">6.4　DEP 放得下了，就得有人喂</h3>
  <p>decode 卡每卡放得下的请求一多，瓶颈就挪到 prefill。<b>dep8 不动，prefill 从 1 台加到 14 台，每块 decode 卡每秒出的字从 307 涨到 3,321</b>，几乎一条直线。</p>
__FIG_S6_FEED__
  <p>出字间隔从 1 台 prefill 时的 5.4 ms 慢慢涨到 12 台时的 14.3 ms。一步这么快，是因为每步手里的请求少；每多一台 prefill 吞吐就跟着涨，说明 decode 一直在等料，不是自己算不过来。同一套机器换成 TP4 decode，瓶颈就在 decode 那边：加第 4 台 prefill，总吞吐 21,100 → 20,479，纹丝不动。</p>
  <div class="note warn"><span class="t">拓扑一换，瓶颈换边</span>
    「prefill 加到 3 台就够」是 TP4 decode 的结论，搬到 DEP8 上就错了。每换一次拓扑，都要重新扫一遍 P:D 配比（第九节）。</div>

  <h3 id="s6-5">6.5　TPU 上也一样，而且要先读懂参数名</h3>
  <p>tpu-inference 的一个公开 PR，把带线性注意力的混合模型也接上了正确的 DP 注意力。
    Qwen3.5-397B 在 v7x-8 上，<b>1K 进 8K 出、并发 512 时，DP=8 比 TP=8 的输出吞吐多 62%</b>；8K 进 1K 出时多 40%。</p>
__FIG_S6_TPU__
  <p>原因的一种解读：Qwen3.5 的全注意力层是 2 个 KV 头，8 路 TP 下同一份 KV 要在 4 个 device 上各放一份（本课推导，PR 没有单独归因）。1K 进 8K 出、并发 64 时两者打平：KV 放得下，复制就不要紧。8K 进 1K 出在并发 64 已多 14%，长 prompt 的 KV 更早把显存挤满。</p>
  <p>TPU 上读启动参数要多一步。tpu-inference 跑 DeepSeek-R1 的命令里写着 <code>--tensor-parallel-size 8</code>，
    <b>那其实是总 device 数</b>（v7x-8 是 4 颗芯片、8 个 device）；真正的切法在 additional-config 里：TP=1、开 DP 注意力、EP=8，也就是 DEP8。</p>

  <h3 id="s6-6">6.6　两件顺手的事</h3>
  <ul>
    <li><b>名字别照字面套</b>：TEP／DEP 是 TRT-LLM 的叫法，不等于 Megatron 的 ETP／EDP。连「prefill 用哪个、decode 用哪个」，vLLM 部署 K3 和 TRT-LLM 的例子都正好相反（<a href="topic-05.html#s8-6">专题五 §8.6</a>）。</li>
    <li><b>EPLB</b>：DEP 下专家那一步要等最忙的那张卡，热门专家多复制几份就是 EPLB（<a href="topic-05.html#s4-3">专题五 §4.3</a>）。</li>
  </ul>
  <p>回到全课那句话：DEP 没让每一趟搬得更少，它让同样的显存放下更多请求，<b>这一趟分给了更多 token</b>。</p>

  <details class="foldfig"><summary><b>考考自己</b>：一个模型有 4 个 KV 头，8 张卡开 TEP8，换 DEP8 后 KV 容量最多涨几倍？什么情况下涨了也没用？</summary>
  <p>8 ÷ 4 ＝ 2 倍。prefill 喂不上、或 batch 已经大到算力先到顶时，放得下更多请求也换不来吞吐。</p></details>
'''

FIGS = {
    "__FIG_S6_FOLD_T5__": ("fig-s6-fold-t5", "fig5-fold.svg", "topic05-fig-ep.py",
        '<b>同样 8 张卡：注意力按 TP 组算，专家层每卡管 32 个专家。</b><br><em>专题五原图，以 V3 为例。</em>'),
    "__FIG_S6_TEPDEP_T5__": ("fig-s6-tep-dep-t5", "fig5-tep-dep.svg", "topic05-fig-ep.py",
        '<b>专家那一半一样，差别全在注意力：合算一批，还是各算各的。</b><br><em>专题五原图。</em>'),
    "__FIG_S6_KV__": ("fig-s6-dep-kv", "fig6-dep-kv.svg", "topic06-fig-dep.py",
        '<b>只有 1 个 KV 头时，TEP4 的 32 格里有 24 格是副本；DEP8 放下 4 倍的请求，实测每卡吞吐 2.09 倍。</b><br>'
        '<em>每卡 4 个槽是示意；倍数为本课推导。实测为 GB300、vLLM、总吞吐、分母含 prefill 卡。</em>'),
    "__FIG_S6_FEED__": ("fig-s6-dep-feed", "fig6-dep-feed.svg", "topic06-fig-dep.py",
        '<b>decode 卡数不变，吞吐跟着 prefill 台数走。</b><br>'
        '<em>纵轴是输出吞吐 ÷ 8 块 decode 卡；8K 进 1K 出，并发 1,024。16 台那一点撞上失效模式，数不完整。</em>'),
    "__FIG_S6_TPU__": ("fig-s6-dep-tpu", "fig6-dep-tpu.svg", "topic06-fig-dep.py",
        '<b>并发越高，DP 注意力赢得越多。</b><br>'
        '<em>输出吞吐；折线两端标的是 TP → DP 的绝对值。两种配置专家都走 EP，只换注意力的切法。</em>'),
}

NUM_LOCK = [
    ("1,319", "fig6-dep-kv.svg"), ("2,758", "fig6-dep-kv.svg"), ("2.09", "fig6-dep-kv.svg"),
    ("307", "fig6-dep-feed.svg"), ("3,321", "fig6-dep-feed.svg"),
    ("62%", "fig6-dep-tpu.svg"), ("40%", "fig6-dep-tpu.svg"),
]
