# -*- coding: utf-8 -*-
"""专题六 · 第六节「切专家与注意力：TEP 和 DEP」。图在 topic06-fig-dep.py；另复用专题五两张图。"""

BODY = '''
  <p class="lead">V3 一张卡装不下；切多卡时专家都用 EP，分歧在注意力：几张卡合算一批请求，还是各算各的。2024 年 12 月 SGLang 给 DeepSeek 模型开了 DP attention，decode 吞吐最多是原来的 1.9 倍；2025 年初 DeepSeek 公开的大 EP 部署也走这条。SGLang 用 96 张 H100 复现大 EP，输出吞吐最多是同卡数纯 TP 的 5 倍，但含<a href="#s8-1">第八节</a>的 PD 分离。
    <b>选哪个，看模型有几个 KV 头、要多大 batch。</b></p>

  <h3 id="s6-1">6.1　一句回顾</h3>
  <p><a href="topic-05.html#s4-4">专题五 §4.4</a> 讲过：同一批卡进注意力和进专家时可用两套切法。
    <b>TEP</b> 是注意力用 TP、专家用 EP；<b>DEP</b> 是注意力用数据并行、专家用 EP。</p>
__FIG_S6_FOLD_T5__
__FIG_S6_TEPDEP_T5__

  <h3 id="s6-2">6.2　判据：KV 头数和 batch</h3>
  <p>TP 按头切注意力。KV 头只有 1 个时切不开，<b>TP 开几路，每个请求的 KV 就存几份，每步也读几遍</b>。DEP 各存各的、各读各的。</p>
  <p>DeepSeek-V4-Pro 不是 MLA，但每层同样只存一条所有头共享的 KV（<code>num_key_value_heads: 1</code>）。
    同样并发 512，decode 从 TP4 换成 dep8（TP1 ＋ 注意力 DP8 ＋ EP8），每卡吞吐从 1,319 涨到 2,758，是 <b>2.09 倍</b>。</p>
  <p>原记录归功于容量：副本让出的显存多收了请求。用 Little 定律（4.3）核一下：</p>
__FIG_S6_KV__
  <p><b>变的是步时</b>：除了图里的重复读和 AllReduce，EP 从 4 路宽到 8 路，每卡读的专家也减半（三项各占多少 runbook 没拆，本课推导）。</p>
  <ul>
    <li><b>换模型</b>：MLA 开 TEP8，KV 存 8 份；Qwen3.5-397B 全注意力层 2 个 KV 头，存 4 份；8 个 KV 头的 GQA 每卡正好一个头，DEP 在 KV 上不占便宜。</li>
  </ul>

  <h3 id="s6-3">6.3　DEP 的代价：每个 rank 都得自己备一份</h3>
  <p>DEP 下每个 rank 像台小推理机，注意力权重、调度器、请求状态各一份。<b>状态按请求开、又不随 DP 切的东西，DEP 一宽就成倍涨。</b></p>
  <ul>
    <li><b>K3 上 DEP8 起不来</b>：Kimi K3 的 93 层里 69 层是 KDA（线性注意力），每个请求一块固定状态。DEP8 下每个 rank 复制整个状态池，两次加载都 OOM；runbook 认为官方因此改推 TP 和 DCP（<a href="#s7-1">第七节</a>）。</li>
    <li><b>TP 反而切得开</b>：KDA 状态按头存，TP 按头切，每卡只存自己的头；DP、EP、DCP 都不动头这一维（本课推导，与官方说法一致）。</li>
    <li><b>忙闲不均</b>：专家那步等最慢的 rank，闲着的 rank 也得陪跑空前向（专题五 §4.4）；热门专家多复制几份是 EPLB（<a href="topic-05.html#s4-3">专题五 §4.3</a>）。</li>
  </ul>

  <h3 id="s6-4">6.4　DEP 吃得快了，就得有人喂</h3>
  <p>decode 步短了，瓶颈挪到 prefill：dep8 下每加一台 prefill，decode 吞吐就涨（<a href="#s9-4">9.4</a>）；换成 TP4 decode，加第 4 台 prefill，总吞吐 21,100 → 20,479，纹丝不动。<b>拓扑一换，瓶颈换边，P:D 配比要重扫。</b></p>

  <h3 id="s6-5">6.5　TPU 上也一样，而且要先读懂参数名</h3>
  <p>tpu-inference 一个公开 PR 给带线性注意力的混合模型接上了 DP 注意力。
    Qwen3.5-397B 在 v7x-8 上，<b>1K 进 8K 出、并发 512 时，DP=8 比 TP=8 的输出吞吐多 62%</b>。</p>
__FIG_S6_TPU__
  <p>可能是 8 路 TP 把全注意力层的 KV 复制了 4 份（6.2，本课推导，PR 没归因），也可能 TP 基线撞了参数上限：TP 从并发 256 到 512 只涨 3%（4,130 → 4,266），PR 又没给 TP 的完整命令。并发 64 时 1K 进 8K 出打平，KV 放得下，复制无妨；8K 进 1K 出已多 14%，长 prompt 更早挤满显存。</p>
  <p>tpu-inference 跑 DeepSeek-R1 的命令写着 <code>--tensor-parallel-size 8</code>，
    <b>那其实是总 device 数</b>；真正的切法在 additional-config 里：TP=1、开 DP 注意力、EP=8，即 DEP8。</p>

  <h3 id="s6-6">6.6　名字别照字面套</h3>
  <p>TEP／DEP 是 TRT-LLM 的叫法，不等于 Megatron（训练框架）的 ETP／EDP，后者指专家层内再分的 TP 组和 DP 组。prefill、decode 各用哪个，vLLM 部署 K3 和 TRT-LLM 的例子正好相反（<a href="topic-05.html#s8-6">专题五 §8.6</a>）。</p>

  <h3 id="s6-7">6.7　大 EP 为什么省</h3>
  <p>DeepSeek 的 decode 把专家铺到 144 张卡（18 个节点），每卡只放 2 个路由专家加 1 个共享专家：</p>
  <ul>
    <li><b>读得少</b>：每卡只读几个专家，一步更快，显存空给 KV。</li>
    <li><b>车坐得满</b>：上百张卡的请求汇到同一个专家，每步行数从 1.4 的平均 8 行往上涨，朝屋脊点靠近。</li>
    <li><b>通信藏起来</b>：把 token 送到所选专家所在的卡，这步 all-to-all 用 DeepEP（DeepSeek 开源）；一批拆成两个微批，一个传、一个算。</li>
  </ul>

  <details class="foldfig"><summary><b>考考自己</b>：一个模型有 4 个 KV 头，8 张卡开 TEP8，换 DEP8 后 KV 容量最多涨几倍？什么情况下涨了也没用？</summary>
  <p>8 ÷ 4 ＝ 2 倍。prefill 喂不上、或 batch 大到算力先到顶时，放得下也换不来吞吐，6.2 的 V4-Pro 就是。</p></details>
'''

FIGS = {
    "__FIG_S6_FOLD_T5__": ("fig-s6-fold-t5", "fig5-fold.svg", "topic05-fig-ep.py",
        '<b>同样 8 张卡：注意力按 TP 组算，专家层每卡管 32 个专家。</b><br><em>专题五原图，以 V3 为例。</em>'),
    "__FIG_S6_TEPDEP_T5__": ("fig-s6-tep-dep-t5", "fig5-tep-dep.svg", "topic05-fig-ep.py",
        '<b>只看注意力那一半。</b><br><em>专题五原图。</em>'),
    "__FIG_S6_KV__": ("fig-s6-dep-kv", "fig6-dep-kv.svg", "topic06-fig-dep.py",
        '<b>对比两列：在飞请求没变多，步时只剩约四分之一。</b><br>'
        '<em>实测为 GB300、vLLM、并发 512；在飞请求按 Little 定律推导，条长是示意。</em>'),
    "__FIG_S6_TPU__": ("fig-s6-dep-tpu", "fig6-dep-tpu.svg", "topic06-fig-dep.py",
        '<b>两条线在 256 到 512 之间交叉：decode 为主的那条后来居上。</b><br>'
        '<em>输出吞吐；折线两端标 TP → DP 的绝对值。专家都走 EP，只换注意力的切法。</em>'),
}

NUM_LOCK = [
    ("1,319", "fig6-dep-kv.svg"), ("2,758", "fig6-dep-kv.svg"), ("2.09", "fig6-dep-kv.svg"),
    ("1.31", "fig6-dep-kv.svg"),
    ("62%", "fig6-dep-tpu.svg"), ("40%", "fig6-dep-tpu.svg"),
]
