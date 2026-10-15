# -*- coding: utf-8 -*-
"""专题六 · 第三节「KV cache：账、池、复用」。图在 topic06-fig-kv.py。"""

BODY = '''
  <p class="lead">每条在跑的序列，都要在显存里单独占一块 KV，这块地方不能和别人共用。
    <b>KV 池装得下几条序列，decode 每一趟搬运最多就分给几个人。</b></p>

  <h3 id="s3-1">3.1　一个 token 留下多少字节：四种记法</h3>
  <p>KV cache 是每个处理过的 token 在每一层留下的 K 和 V，后面的 token 算注意力时要回头读。
    一个 token 留多少字节，由注意力怎么存 K、V 决定（专题三逐项算过）。</p>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">全部按 bf16（2 字节）、所有层加起来算；实际部署常把 KV 存成 fp8，字节再减半。字节数取自各模型 config.json，本课推导。</caption>
    <tr><th>记法</th><th>每层存什么</th><th>每 token 字节</th><th>例子</th></tr>
    <tr><td>MHA</td><td>每个头各存一份 K、V</td><td>3,997,696</td><td>V3 的形状假如用 MHA</td></tr>
    <tr><td>GQA</td><td>几个头共用一组 K、V</td><td>327,680</td><td>Llama-3.1-70B：8 组 × 128 维 × 80 层</td></tr>
    <tr><td>MLA</td><td>一条压缩向量，所有头从它解出 K、V</td><td>70,272</td><td>DeepSeek-V3：(512＋64) 维 × 61 层</td></tr>
    <tr><td>线性注意力</td><td>不按 token 存，每条序列一个大小固定的状态</td><td>0（另交固定状态）</td><td>Qwen3.5：45 层线性层每条序列 180 MiB，另 15 层全注意力 30,720 B/token</td></tr>
  </table>
  <p>DeepSeek-V4 不在这张表的任何一行：它每层只存一条所有头共用的 KV，再按层压缩 4 倍或 128 倍（CSA／HCA），3.4 节讲它的两个池。</p>

  <h3 id="s3-2">3.2　batch 上限 ＝ KV 池 ÷ 每条序列的 KV</h3>
  <p>一颗 v7 芯片 192 GB，放下 V3 的八分之一 FP8 权重，再留一成余量，剩 88.9 GB 给 KV。
    V3 的每个 token 占 70,272 字节，4K 上下文能装 308 条，32K 只剩 38 条。</p>
__FIG_S3_BATCH__
  <ul>
    <li><b>上下文翻几倍，人数就除几倍。</b>KV 和上下文长度成正比，池子是固定的。</li>
    <li><b>记法决定起点。</b>同样 4K，GQA 每 token 327,680 字节，只装得下 66 条，不到 MLA 的四分之一。</li>
    <li><b>线性注意力是一笔固定成本。</b>短上下文时吃亏，长上下文时占便宜。</li>
  </ul>
  <p>人数少了，同一趟权重搬运分给的 token 也少了。这就是长上下文贵的第一个原因。</p>

  <h3 id="s3-3">3.3　PagedAttention：像操作系统那样分页</h3>
  <p>早先的做法按最大长度给每条序列预留一整段连续显存，没写到的部分全浪费。
    PagedAttention 把 KV 切成固定大小的块，每条序列用一张页表记下「第几块住在哪个物理块」。</p>
__FIG_S3_PAGED__
  <ul>
    <li><b>按需领块。</b>序列长一点就多领一块，浪费只剩最后一块没写满的尾巴。vLLM 论文测过，旧系统 KV 显存里真正存着 token 的只有 20.4%–38.2%，分页后到 96.3%。</li>
    <li><b>共享与写时复制。</b>并行采样的几条序列共用 prompt 的块，各记一个引用数；谁要往共享块里写，先给自己复制一份。</li>
    <li><b>块大小跟 kernel 一起定。</b>attention kernel 每读一块查一次页表。TPU 上 Qwen3.5 把块从 16 个 token 调到 256，512 并发时 decode 一步从 428 µs 降到 283 µs。</li>
  </ul>

  <h3 id="s3-4">3.4　混合模型有两个池，预算要分</h3>
  <p>DeepSeek-V4 每层有两种 KV：滑窗那份只留最近 128 个 token，另一份存压缩后的整段。
    SGLang 给它们开两个池，共用一份预算，用 <code>--swa-full-tokens-ratio</code> 定两边怎么分。</p>
__FIG_S3_POOL__
  <p>4K 输入时，比例 0.20 让全长池先满，每张 decode GPU 只跑出 6,887 tok/s；调到 0.15 两边一起满，到 10,614，高 54%。
    再多给显存、多装 13% 的人，只到 10,704：每张卡近 900 条之后，瓶颈已经从装不下换成了算不动。</p>

  <div class="note warn"><span class="t">⚠️ 池子算错不一定报错</span>
    tpu-inference 跑 Qwen3.5 时，vLLM 的混合分配器默认 4 层共用一块 KV 张量，而 TPU 上每层必须各存一份。
    结果块编号的池子大了约 3.5 倍，编号越界后被 JAX 的 <code>dynamic_update_slice_in_dim</code> <b>静默截断</b>到边界上，多个请求写进同一块。
    表面症状是胡言乱语、OOM 或引擎崩溃，看不出是 KV 池的账错了（<code>tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:77</code>）。</div>

  <h3 id="s3-5">3.5　前缀复用：算过的不再算</h3>
  <p>很多请求开头一样：同一段 system prompt、同一份文档、多轮对话的前几轮。
    把算好的 KV 留着，下一个请求命中前缀就直接接着算，prefill 省掉一大段。</p>
  <ul>
    <li><b>RadixAttention。</b>SGLang 把所有缓存的前缀存成一棵基数树，新请求沿树往下匹配最长前缀，满了按最近最少使用淘汰。</li>
    <li><b>命中率可以很高。</b>DeepSeek 公开过 24 小时的线上统计：6,080 亿输入 token 里 56.3% 命中了存在磁盘上的 KV。</li>
    <li><b>线性注意力和滑窗难复用。</b>线性层的状态是整段压出来的，截不回任意位置；滑窗层早先位置的 KV 已经丢了。想复用，就得在块边界另存一份状态快照。</li>
  </ul>
  <p>tpu-inference 跑 Qwen3.5 目前要显式关掉前缀缓存（<code>--no-enable-prefix-caching</code>），否则触发一个断言错误（同一 README:95）。</p>

  <h3 id="s3-6">3.6　分层卸载：搬回来，还是重算</h3>
  <p>HBM 放不下的 KV 可以往下放：主机内存、本机磁盘、远端存储，一层比一层大、一层比一层慢。
    命中之后要搬回来，搬回来比重算慢就白存了。</p>
  <div class="note ok"><span class="t">盈亏带宽</span>
    搬回 ＝ 每 token KV 字节 × L ÷ 带宽；重算 ≈ 每 token FLOP × L ÷ 有效算力。<br>
    两者相等时：<b>带宽 ＝ 每 token KV 字节 × 有效算力 ÷ 每 token FLOP</b>。链路比它慢，就不如重算。</div>
__FIG_S3_OFFLOAD__
  <p>V3 按 v7 FP8 峰值的一半算，盈亏带宽约 2.19 GB/s；GQA 的 Llama-3.1-70B 每个 FLOP 对应的 KV 多，要 5.35 GB/s。
    前缀越长，注意力的平方项让重算越贵，1 GB/s 的慢链路过了约 35K token 也能赢。</p>
  <ul>
    <li><b>llm-d</b>（开源的 Kubernetes 推理调度）在 TPU7x 上测过 Qwen3-32B 把 KV 卸到主机内存：平均 TTFT 0.98 s → 0.56 s，P90 2.1 s → 0.5 s，吞吐 +8.9%。它的排障说明里也写着另一面：TTFT 比不卸载还差，就是搬回来比重算慢。</li>
    <li><b>Mooncake Store</b>：开源的分布式 KV 存储，把集群各机的内存和 SSD 拼成一个池，机器之间用 RDMA 互取。</li>
    <li><b>LMCache</b>：开源的 KV 缓存层，接在 vLLM、SGLang 下面，往主机内存、本地盘、远端存储里放 KV。</li>
    <li><b>TPU Sync</b>：Google 开源的 TPU 侧 KV 编排，做多层存储和主机内存卸载，README 写明还在开发中、不建议通用。</li>
  </ul>

  <div class="note ok"><span class="t">这一节只要带走两件事</span>
    <b>① KV 一人一份，它定了 batch 上限，也就定了一趟搬运最多分给几个人。</b>记法、分页、两池分账，都是在同样的显存里多装几条序列。<br>
    <b>② 复用和卸载是少算、少搬。</b>命中的前缀不再 prefill；放到下层的 KV 要算清搬回来是不是比重算快。</div>

  <details class="foldfig"><summary><b>考考自己</b>：一颗 v7 芯片的 KV 池是 88.9 GB，V3 跑 64K 上下文能装几条？要是把 KV 存成 fp8 呢？</summary>
  <p>每条 70,272 × 65,536 ≈ 4.6 GB，装 19 条。fp8 字节减半，装 38 条。</p></details>
'''

FIGS = {
    "__FIG_S3_BATCH__": ("fig-s3-batch", "fig6-kv-batch.svg", "topic06-fig-kv.py",
        '<b>每条线往右一格，能装的人就少一半；红线到 32K 就一条都装不下了。</b><br>'
        '<em>一颗 v7 芯片，V3 权重 FP8 铺 8 颗芯片，可用显存 90%；KV 按 bf16、attention 数据并行。本课推导。</em>'),
    "__FIG_S3_PAGED__": ("fig-s3-paged", "fig6-paged.svg", "topic06-fig-kv.py",
        '<b>绿色两块两条序列共用，橙色那块是 B 要改时复制出来的。</b><br>'
        '<em>每块 4 个 token 只为画得下；机制见 vLLM 论文（arXiv 2309.06180）。</em>'),
    "__FIG_S3_POOL__": ("fig-s3-pool", "fig6-swa-ratio.svg", "topic06-fig-kv.py",
        '<b>只有红条一边满时，另一边的预算在空转；两边都红才最快。</b><br>'
        '<em>V4-Pro 在 GB300 上的 SGLang 实测，按单卡计，输入 4K／8K、输出 1K，关前缀缓存。</em>'),
    "__FIG_S3_OFFLOAD__": ("fig-s3-offload", "fig6-offload.svg", "topic06-fig-kv.py",
        '<b>绿线始终在红线下面：链路够快，命中多长都该搬。</b><br>'
        '<em>每芯片口径；有效算力按 FP8 峰值 50% 假设，两档带宽是示意值。本课推导。</em>'),
}

NUM_LOCK = [
    ("88.9", "fig6-kv-batch.svg"),
    ("70,272", "fig6-kv-batch.svg"),
    ("308", "fig6-kv-batch.svg"),
    ("38", "fig6-kv-batch.svg"),
    ("327,680", "fig6-kv-batch.svg"),
    ("66", "fig6-kv-batch.svg"),
    ("428", "fig6-paged.svg"),
    ("283", "fig6-paged.svg"),
    ("6,887", "fig6-swa-ratio.svg"),
    ("10,614", "fig6-swa-ratio.svg"),
    ("10,704", "fig6-swa-ratio.svg"),
    ("54%", "fig6-swa-ratio.svg"),
    ("2.19", "fig6-offload.svg"),
    ("5.35", "fig6-offload.svg"),
    ("35K", "fig6-offload.svg"),
]
