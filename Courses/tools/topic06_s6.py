# -*- coding: utf-8 -*-
"""专题六 · 第六节「切专家与注意力：TEP 和 DEP」。图在 topic06-fig-dep.py；另复用专题五两张图。"""

BODY = '''
  <p class="lead">专家那一半大家都用 EP，分歧全在注意力：几张卡合算同一批请求，还是各算各的。2025 年初 DeepSeek 公开的大 EP 部署走的是后一条，两个月后 SGLang 复现，对同卡数 TP 最多快 5 倍。
    <b>选哪个，看模型有几个 KV 头、你要多大的 batch。</b></p>

  <h3 id="s6-1">6.1　一句回顾</h3>
  <p><a href="topic-05.html#s4-4">专题五 §4.4</a> 讲过：同一批卡，进注意力和进专家时可以用两套切法。
    <b>TEP</b> 是注意力用 TP、专家用 EP；<b>DEP</b> 是注意力用数据并行、专家用 EP。</p>
__FIG_S6_FOLD_T5__
__FIG_S6_TEPDEP_T5__

  <h3 id="s6-2">6.2　判据：KV 头数和 batch</h3>
  <p>TP 按头切注意力。KV 头只有 1 个时切不开，<b>TP 开几路，每个请求的 KV 就存几份，每一步也读几遍</b>。DEP 各存各的、各读各的。</p>
  <p>DeepSeek-V4-Pro 的注意力是 CSA 与 HCA 交错，不是 MLA，但每层同样只存一条所有头共享的 KV（<code>num_key_value_heads: 1</code>）。
    同样并发 512，decode 从 TP4 换成 dep8（TP1 ＋ 注意力 DP8 ＋ EP8），每卡吞吐从 1,319 涨到 2,758，是 <b>2.09 倍</b>。</p>
  <p>原始记录把功劳记给容量：副本让出的显存多收了请求。拿 Little 定律核一下（在飞请求 ＝ 每秒输出 × 出字间隔）：</p>
__FIG_S6_KV__
  <p>容量没帮上忙，<b>变的是步时</b>：TP4 的 4 张卡每步各读一遍整份 KV，每层还多一次 AllReduce。这个归因是本课推导，runbook 没拆分。</p>
  <ul>
    <li><b>口径</b>：2.09 倍是 prompt 加输出的总 token 除以全部卡（16 张和 20 张，各含 12 张 prefill 卡）。decode 卡同时从 4 张翻到 8 张，只算 decode 卡的输出是 1.31 倍。</li>
    <li><b>换个模型</b>：MLA 开 TEP8 是 8 份；Qwen3.5-397B 全注意力层 2 个 KV 头，是 4 份；8 个 KV 头的 GQA 每卡正好一个头，DEP 在 KV 上不占便宜。</li>
  </ul>

  <h3 id="s6-3">6.3　DEP 的代价：每个 rank 都得自己备一份</h3>
  <p>DEP 下每个 rank 像一台小推理机，注意力权重、调度器、请求状态各一份。<b>状态按请求开、又不随 DP 切的东西，DEP 一宽就成倍涨。</b></p>
  <ul>
    <li><b>K3 上 DEP8 起不来</b>：Kimi K3 的 93 层里有 69 层 KDA（线性注意力），每个请求一块固定状态。DEP8 下每个 rank 复制整个状态池，两次加载都 OOM；runbook 认为官方因此改推 TP 和 DCP（把一个请求的 KV 按 token 分存到几张卡上，见<a href="#s7-1">第七节</a>）。</li>
    <li><b>TP 反而切得开</b>：KDA 状态按头存，TP 按头切，每张卡只存自己那几个头；DP、EP、DCP 都不动头这一维（本课推导，与 runbook 所引官方说法一致）。</li>
    <li><b>每 rank 的 batch 会撞上实现上限</b>：SGLang 跑 V4-Pro dep8 时，一个注意力 kernel 写死每 rank 最多 1,024 个请求。实测 batch 到 999 时算力先到顶，上限没真正卡住。</li>
    <li><b>忙闲不均</b>：各 rank 的请求长短不一，专家那一步要等最慢的 rank；没请求的 rank 也得陪跑空前向（专题五 §4.4）。热门专家多复制几份就是 EPLB（<a href="topic-05.html#s4-3">专题五 §4.3</a>）。</li>
  </ul>

  <h3 id="s6-4">6.4　DEP 吃得快了，就得有人喂</h3>
  <p>decode 一步变短，瓶颈就挪到 prefill：dep8 不动，每加一台 prefill，decode 的吞吐就跟着涨，曲线见 <a href="#s9-4">9.4</a>。同一套机器换成 TP4 decode，加第 4 台 prefill 总吞吐 21,100 → 20,479，纹丝不动。<b>拓扑一换，瓶颈换边，P:D 配比要重扫。</b></p>

  <h3 id="s6-5">6.5　TPU 上也一样，而且要先读懂参数名</h3>
  <p>tpu-inference 的一个公开 PR，把带线性注意力的混合模型也接上了正确的 DP 注意力。
    Qwen3.5-397B 在 v7x-8 上，<b>1K 进 8K 出、并发 512 时，DP=8 比 TP=8 的输出吞吐多 62%</b>；8K 进 1K 出时多 40%。</p>
__FIG_S6_TPU__
  <p>一种解读：全注意力层 2 个 KV 头，8 路 TP 下同一份 KV 在 4 个 device 上各放一份（本课推导，PR 没归因）。1K 进 8K 出、并发 64 时打平：KV 放得下，复制不要紧；8K 进 1K 出在并发 64 已多 14%，长 prompt 更早挤满显存。</p>
  <p>TPU 上读启动参数要多一步：tpu-inference 跑 DeepSeek-R1 的命令写着 <code>--tensor-parallel-size 8</code>，
    <b>那其实是总 device 数</b>；真正的切法在 additional-config 里：TP=1、开 DP 注意力、EP=8，也就是 DEP8。</p>

  <h3 id="s6-6">6.6　名字别照字面套</h3>
  <p>TEP／DEP 是 TRT-LLM（NVIDIA 的推理引擎）的叫法，不等于 Megatron（NVIDIA 的训练框架）的 ETP／EDP，那是专家层内部再分的 TP 组和数据并行组。连「prefill 用哪个、decode 用哪个」，vLLM 部署 K3 和 TRT-LLM 的例子都正好相反（<a href="topic-05.html#s8-6">专题五 §8.6</a>）。</p>

  <h3 id="s6-7">6.7　大 EP 为什么省</h3>
  <p>DeepSeek 的 decode 把专家铺到 144 张卡（18 个节点），每张卡只放 2 个路由专家加 1 个共享专家：</p>
  <ul>
    <li><b>读得少</b>：每张卡只读几个专家，一步更快，空出的显存给 KV。</li>
    <li><b>车坐得满</b>：上百张卡的请求汇到同一个专家，每个专家每步分到的行数往拐点（5.5）靠。</li>
    <li><b>通信藏起来</b>：把每个 token 送到它选中专家所在的卡，这一步 all-to-all 用 DeepEP（DeepSeek 开源的专家通信库）；一批再拆成两个微批，一个传、一个算。</li>
  </ul>
  <p>所以专家那头，大 EP 让一趟分给更多行；注意力那头，DEP 省掉每步白读的 KV 副本。</p>

  <details class="foldfig"><summary><b>考考自己</b>：一个模型有 4 个 KV 头，8 张卡开 TEP8，换 DEP8 后 KV 容量最多涨几倍？什么情况下涨了也没用？</summary>
  <p>8 ÷ 4 ＝ 2 倍。prefill 喂不上、或 batch 已经大到算力先到顶时，放得下也换不来吞吐；6.2 的 V4-Pro 就没用上这份容量。</p></details>
'''

FIGS = {
    "__FIG_S6_FOLD_T5__": ("fig-s6-fold-t5", "fig5-fold.svg", "topic05-fig-ep.py",
        '<b>同样 8 张卡：注意力按 TP 组算，专家层每卡管 32 个专家。</b><br><em>专题五原图，以 V3 为例。</em>'),
    "__FIG_S6_TEPDEP_T5__": ("fig-s6-tep-dep-t5", "fig5-tep-dep.svg", "topic05-fig-ep.py",
        '<b>专家那一半一样，差别全在注意力：合算一批，还是各算各的。</b><br><em>专题五原图。</em>'),
    "__FIG_S6_KV__": ("fig-s6-dep-kv", "fig6-dep-kv.svg", "topic06-fig-dep.py",
        '<b>在飞的请求没变多，每一步却快了 4 倍：TP4 每张卡都在读同一份 KV。</b><br>'
        '<em>实测为 GB300、vLLM、并发 512；在飞请求按 Little 定律推导，条长是示意。</em>'),
    "__FIG_S6_TPU__": ("fig-s6-dep-tpu", "fig6-dep-tpu.svg", "topic06-fig-dep.py",
        '<b>并发越高，DP 注意力赢得越多。</b><br>'
        '<em>输出吞吐；折线两端标的是 TP → DP 的绝对值。两种配置专家都走 EP，只换注意力的切法。</em>'),
}

NUM_LOCK = [
    ("1,319", "fig6-dep-kv.svg"), ("2,758", "fig6-dep-kv.svg"), ("2.09", "fig6-dep-kv.svg"),
    ("1.31", "fig6-dep-kv.svg"),
    ("62%", "fig6-dep-tpu.svg"), ("40%", "fig6-dep-tpu.svg"),
]
