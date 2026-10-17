# -*- coding: utf-8 -*-
"""专题六 · 第十三节「新架构出难题：状态没法按 token 切」。混合模型把前十二节的一个默认前提拆掉了。"""

BODY = '''
  <p class="lead">前十二节都默认一件事：每个 token 留一份一样大的 KV，想截在哪就截在哪，想按块复用、按块搬都行。
    混合模型（4.1）里的线性层不是这样：<b>一条序列只有一份状态，整段压进去，截不开。</b></p>

  <h3 id="s13-1">13.1　前提塌了：三种层，能从哪儿接着算</h3>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">「能从哪儿接着算」的三分法取自 LMSYS 2026-08 的统一前缀树博客。</caption>
    <tr><th>层</th><th>存什么</th><th>新请求能从哪儿接着算</th></tr>
    <tr><td>全注意力</td><td>每个 token 一份 KV</td><td>共享前缀的任意位置</td></tr>
    <tr><td>滑窗</td><td>只留最近一个窗口的 KV</td><td>要有完整的最后一个窗口</td></tr>
    <tr><td>线性（GDN、Mamba、KDA）</td><td>每条序列一份固定大小的状态，原地更新</td><td>只能是正好存过状态的位置</td></tr>
  </table>
  <p>线性层的状态每进一个 token 就改写一次，回不到更短的前缀。PyTorch 博客（2025-12）把难处归成三条：退不回去；一份比单个 token 的 KV 大几个数量级；要么整份能用，要么不能用。</p>

  <h3 id="s13-2">13.2　前缀缓存只能改存快照</h3>
  <p>截不开，就在一些位置把整份状态拷一份存下来，叫快照。新请求沿前缀匹配，退到不超过共享前缀的最近一份快照，从那儿接着算，中间那段重算。</p>
__FIG_S13_SNAP__
  <p>命中后先把快照拷到这个请求自己的槽里再往下改：状态是原地更新的，直接在共享的那份上算，别人的就坏了。</p>
  <p>全注意力层的 KV 还是按块存，快照只是线性层多出来的一份。两类东西的复用边界不同，SGLang 2026 年把它们挂进同一棵前缀树，各按自己的边界判命中。</p>

  <h3 id="s13-3">13.3　隔多久存一份</h3>
  <ul>
    <li><b>SGLang：每 256 个 token 一份。</b>decode 时序列长度整除 <code>--mamba-track-interval</code>（默认 256）就存；prefill 的落点按 chunk 和 page 的整数倍对齐。<code>--mamba-max-states-per-path</code> 可再限一条路径上留几份，超了先删最浅的。</li>
    <li><b>vLLM：只在块边界存。</b>开前缀缓存时 <code>mamba_cache_mode</code> 自动设为 <code>align</code>，命中也只能从块边界接着算。<code>prefix_cache_retention_interval</code> 默认 0，只留必要的几份（重放边界、共享前缀分叉处）；设正值才按间隔多留。</li>
    <li><b>块被撑大了。</b>vLLM 所有层共用一种页大小，全注意力的块要放大到一页装得下一份状态。官方博客的 Nemotron-Nano-12B-v2 例子里，一块变成 672 个 token，前缀命中的粒度跟着变粗。</li>
  </ul>

  <h3 id="s13-4">13.4　代价：一份快照顶几千个 token</h3>
  <p>拿 Qwen3.5-397B 算（本课推导，按 config）：15 层全注意力，每 token 的 KV 是 30 KiB；45 层 GDN，每层状态 4 MiB（fp32）加卷积状态 72 KiB，一份快照约 183 MiB。4.1 表里的 180 MiB 只算了前一项。</p>
  <p>183 MiB 约等于 6,250 个 token 的 KV。按 SGLang 默认每 256 个 token 存一份，快照占的显存约是这段 KV 的 24 倍（183 ÷ 7.5）。</p>
  <ul>
    <li><b>存得密</b>：命中更准，显存被快照吃掉，能留的前缀变少。</li>
    <li><b>存得疏</b>：省显存，命中后平均要多重算约半个间隔（本课推导）。</li>
    <li><b>把快照压小</b>：SGLang 的 <code>--enable-int8-mamba-checkpoint</code>（默认关）存 int8，说明写能多缓存约一倍前缀。选 int8 不选 fp8：状态值分布近乎均匀，fp8 的指数位用不上；而且只在命中时还原一次，误差不会在递推里累积。</li>
  </ul>

  <h3 id="s13-5">13.5　同一个前提，前面已经撞了四次</h3>
  <ul>
    <li><b>前缀复用</b>（<a href="#s4-5">4.5</a>）：tpu-inference 跑 Qwen3.5 得关掉前缀缓存，否则断言出错；看样子是还没接上这套快照（本课推测）。</li>
    <li><b>DEP</b>（<a href="#s6-3">6.3</a>）：状态按请求开、不随 DP 切，K3 每个 rank 存整池状态，DEP8 起不来。</li>
    <li><b>PCP</b>（<a href="#s7-5">7.5</a>）：序列切开后状态要从前一段接力，不能从空状态起步。</li>
    <li><b>PD 分离</b>（<a href="#s8-5">8.5</a>）：状态已吃下整段 prompt，尾巴不能让 decode 端重算，只能整份传。</li>
  </ul>
  <p>GPU 引擎这两年为此补了成套机制。TPU 上把一个混合模型真正跑起来，要过哪几关，是下一节。</p>
'''

FIGS = {
    "__FIG_S13_SNAP__": ("fig-s13-snap", "fig6-snapshot.svg", "topic06-fig-snapshot.py",
        '<b>线性层只能退到存档点；一份存档约 183 MiB，是 256 个 token 的 KV 的 24 倍。</b><br>'
        '<em>Qwen3.5-397B，按 config 推算（本课推导）。右下假设一条路径上的快照全留着，实际会被驱逐或限数。</em>'),
}

NUM_LOCK = []
