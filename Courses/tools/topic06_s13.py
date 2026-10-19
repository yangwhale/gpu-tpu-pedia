# -*- coding: utf-8 -*-
"""专题六 · 第十三节「新架构出难题：状态没法按 token 切」。混合模型把前十二节的一个默认前提拆掉了。"""

BODY = '''
  <p class="lead">前十二节都默认一件事：每个 token 留一份一样大的 KV，想截在哪就截在哪，想按块复用、按块搬都行。
    混合模型（4.1）里的线性层不是这样：<b>一条序列只有一份状态，整段压进去，截不开。</b></p>

  <h3 id="s13-1">13.1　前提塌了：三种层，前缀缓存只能存快照</h3>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">「能从哪儿接着算」的三分法取自 LMSYS 2026-08 的统一前缀树博客。</caption>
    <tr><th>层</th><th>存什么</th><th>新请求能从哪儿接着算</th></tr>
    <tr><td>全注意力</td><td>每个 token 一份 KV</td><td>共享前缀的任意位置</td></tr>
    <tr><td>滑窗</td><td>只留最近一个窗口的 KV</td><td>要有完整的最后一个窗口</td></tr>
    <tr><td>线性（GDN、Mamba、KDA）</td><td>每条序列一份固定大小的状态，原地更新</td><td>只能是正好存过状态的位置</td></tr>
  </table>
  <p>线性层的状态每进一个 token 就改写一次，回不到更短的前缀。PyTorch 博客（2025-12）把难处归成三条：退不回去；一份比单个 token 的 KV 大几个数量级；要么整份能用，要么不能用。</p>

  <p><b>前缀缓存只能改存快照。</b>截不开，就在一些位置把整份状态拷一份存下来，叫快照。新请求沿前缀匹配，退到不超过共享前缀的最近一份快照，从那儿接着算，中间那段重算。</p>
__FIG_S13_SNAP__
  <p>命中后先把快照拷到这个请求自己的槽里再往下改：状态是原地更新的，直接在共享的那份上算，别人的就坏了。</p>
  <p>全注意力层的 KV 还是按块存，快照只是线性层多出来的一份。两类东西的复用边界不同，SGLang 2026 年把它们挂进同一棵前缀树，各按自己的边界判命中。</p>

  <h3 id="s13-2">13.2　生产里怎么存：两个池，只留要紧的几份</h3>
  <p><b>一份快照顶几千个 token。</b>拿 Qwen3.5-397B 算（本课推导，按 config）：15 层全注意力，每 token 的 KV 是 30 KiB；45 层 GDN，每层状态 4 MiB（fp32）加卷积状态 72 KiB，一份约 183 MiB，约等于 6,250 个 token 的 KV。4.1 表里的 180 MiB 只算了前一项。</p>
  <p>K3 更大（本课推导，按公开 config、状态按 fp32 计）：69 层 KDA，每层 96 头 × 128 × 128，一份快照约 430 MiB；24 层 MLA 每 token 只存 576 维，约 27 KiB。一份快照约顶 1.6 万个 token 的 KV。</p>

  <p><b>所以生产不每段都存，只留要紧的几份。</b>vLLM 的 K3 博客明说：每个位置都存「贵得没法用」。它留三种：</p>
  <ul>
    <li><b>prompt 末尾必留</b>，自动识别。多轮对话下一轮正好从这里接着算。</li>
    <li><b>隔一大段留一份</b>，例如每 32K 一份，用 <code>VLLM_PREFIX_CACHE_RETENTION_INTERVAL</code> 设；设 0 就只留 prompt 末尾。</li>
    <li><b>第二次撞见才存</b>（Marconi，MLSys '25）：系统提示、代码仓库这类共享前缀不在 prompt 边界上，第一次只记下来，第二次确认真被共享，才在分叉处存一份。</li>
  </ul>
  <p>SGLang 是另一种管法：decode 每 256 个 token 记一份（<code>--mamba-track-interval</code>），一条路径留几份默认不封顶；真正的上限是池子大小，满了按 LRU 删快照，KV 留着。新请求的 KV 命中比快照远，就在分叉处补记一份，下回直接命中。</p>
  <p><b>摊到一个请求上就不吓人了</b>（本课推导）：128K 的前缀，Qwen3.5 的 KV 约 3.75 GiB，末尾一份快照 183 MiB，约 5%；每 32K 再留一份，四份约 0.72 GiB，约 19%。吓人的是每 256 个都留：512 份、约 92 GiB。</p>

  <p><b>两种东西，两种管法。</b>KV 按 token 长、按块存；快照按请求、按份存。</p>
  <ul>
    <li><b>SGLang 分两个池</b>：KV 池按 token，Mamba 池按份，比例用 <code>--mamba-full-memory-ratio</code> 定；弹性池让一边闲时让给另一边。淘汰走两条 LRU：KV 只能从叶子往根删，快照哪个节点上的都能单独删。</li>
    <li><b>vLLM 一个管理器、一种页</b>：全注意力的块放大到一页装得下一份状态。PyTorch 博客的 Nemotron-Nano-12B-v2 例子里一块 672 个 token，命中粒度跟着变粗。K3 起把物理块、调度对齐、前缀匹配单位拆开，快照能登记在大块里的细边界上。</li>
    <li><b>往下分层</b>：显存放不下，KV 和快照一起落到 CPU 内存、再到外部存储（SGLang HiCache；vLLM 接 Mooncake 或 CPU 卸载），命中再搬回来。</li>
    <li><b>把快照压小</b>：SGLang 的 <code>--enable-int8-mamba-checkpoint</code>（默认关）存 int8，说明写能多缓存约一倍前缀。选 int8 不选 fp8：状态值分布近乎均匀，fp8 的指数位用不上；而且只在命中时还原一次，误差不会在递推里累积。</li>
  </ul>

  <p><b>同一个前提，前面已经撞了四次。</b></p>
  <ul>
    <li><b>前缀复用</b>（<a href="#s4-3">4.3</a>）：tpu-inference 跑 Qwen3.5 得关掉前缀缓存（<code>--no-enable-prefix-caching</code>，<code>tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:95</code>），否则断言出错；看样子是还没接上这套快照（本课推测）。</li>
    <li><b>DEP</b>（<a href="#s6-1">6.1</a>）：状态按请求开、不随 DP 切，K3 每个 rank 存整池状态，DEP8 起不来。</li>
    <li><b>PCP</b>（<a href="#s7-2">7.2</a>）：序列切开后状态要从前一段接力，不能从空状态起步。</li>
    <li><b>PD 分离</b>（<a href="#s8-3">8.3</a>）：状态已吃下整段 prompt，尾巴不能让 decode 端重算，只能整份传。</li>
  </ul>
  <p>GPU 引擎这两年为此补了成套机制。TPU 上要把一个混合模型真正跑起来，还得一关一关过（第十四节）。</p>
'''

FIGS = {
    "__FIG_S13_SNAP__": ("fig-s13-snap", "fig6-snapshot.svg", "topic06-fig-snapshot.py",
        '<b>线性层只能退到存档点；一份存档约 183 MiB，顶 6,250 个 token 的 KV；128K 前缀只留末尾一份，才占 KV 的 5%。</b><br>'
        '<em>Qwen3.5-397B，按 config 推算（本课推导）。右下比的是 128K 前缀存几份快照；末尾必留、每 32K 一份取自 vLLM 的留存策略。</em>'),
}

NUM_LOCK = []
