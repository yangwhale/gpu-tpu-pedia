# -*- coding: utf-8 -*-
"""专题六 · 第三节「KV cache：账、池、复用」。图在 topic06-fig-kv.py。"""

BODY = '''
  <p class="lead">尺子有了，回到第一节那一趟：每条在跑的序列都在显存里单独占一块 KV。
    <b>KV 池装得下几条序列，decode 每一趟搬运最多就分给几个人。</b>
    2023 年起的几招都在同样的显存里多装人、少重算。</p>

  <h3 id="s3-1">3.1　一个 token 留下多少字节：四种记法</h3>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">按 bf16、所有层合计；存 fp8 约减半（MLA 见 12.7）。字节数取自各模型 config.json，本课推导；Qwen3.5 线性层状态按 fp32 算（config 的 mamba_ssm_dtype）；MHA 一行按 DeepSeek-V2 论文的口径（K、V 各 128 维 × 128 头）。</caption>
    <tr><th>记法</th><th>每层存什么</th><th>每 token 字节</th><th>例子</th></tr>
    <tr><td>MHA（2017）</td><td>每个头各存一份 K、V</td><td>3,997,696</td><td>V3 的形状假如用 MHA</td></tr>
    <tr><td>GQA（2023）</td><td>几个头共用一组 K、V</td><td>327,680</td><td>Llama-3.1-70B：8 组 × 128 维 × 80 层</td></tr>
    <tr><td>MLA（2024）</td><td>一条压缩向量，所有头从它解出 K、V</td><td>70,272</td><td>DeepSeek-V3：(512 维压缩 KV＋64 维位置编码那截) × 61 层（不含 MTP 层）</td></tr>
    <tr><td>线性注意力</td><td>不按 token 存，每条序列一个大小固定的状态</td><td>0（另交固定状态）</td><td>Qwen3.5：45 层线性层每条序列 180 MiB，另 15 层全注意力 30,720 B/token</td></tr>
  </table>
  <p>DeepSeek 自家这条线后来又改了三代，省的地方不一样：看黄框落在哪一排。V4 的两个池怎么分预算见 3.4。</p>
__FIG_S3_LINEAGE__

  <h3 id="s3-2">3.2　batch 上限 ＝ KV 池 ÷ 每条序列的 KV</h3>
  <p>一颗 v7 芯片扣掉一成激活值和 1/8 份 V3 FP8 权重，剩 92.4 GiB 给 KV（拆法见图，口径同 10.5）。</p>
  <p>KV 不切：注意力走数据并行（DP attention），每条序列的 KV 只住一个 device（第六节）。这里按权重全切粗算，实际非专家权重每个 device 各要一份，池子更小。</p>
__FIG_S3_BATCH__
  <p>人少了，一趟搬运分给的 token 也少：这是长上下文贵的第一个原因。</p>

  <h3 id="s3-3">3.3　PagedAttention：像操作系统那样分页</h3>
  <p>早先按最大长度给每条序列预留一整段连续显存。PagedAttention 把 KV 切成固定大小的块，用页表记每块住在哪。</p>
__FIG_S3_PAGED__
  <ul>
    <li><b>按需领块，</b>浪费只剩最后一块的空尾巴。vLLM 论文测得旧系统 KV 显存利用率 20.4%–38.2%，分页后 96.3%；同样延迟下吞吐是 FasterTransformer 和 Orca 的 2–4 倍（SOSP 2023，Orca 为作者复现版）。</li>
  </ul>
<figure class="fbox fwide" id="anim-paged">
<video src="media/topic06-paged.mp4" autoplay loop muted playsinline
       aria-label="分页 KV 动画。标题：分页 KV：写满一块才领下一块，用完立刻还。左边是 16 块显存，标着块 0 到块 15，每块 4 个 token 格；右边是页表，四行请求 1 到请求 4，每行列出它的第几块住在哪个物理块；底下写空闲块数和第几步，右下角写：每种颜色一个请求；灰框＝空闲块。字幕依次是：KV 显存切成固定大小的块，每块装 4 个 token（示意）。三个请求进来、边生成边长：写满一块，才去空闲池领一块新的。领到哪块算哪块：一个请求的块东一块西一块，页表记住先后。请求 2 结束：它的 2 块立刻还回空闲池（页表上请求 2 那行写：已结束：块已还回）。新请求 4 进来，马上用上刚还回来的那 2 块（块 2 和块 5）。最后第 12 步，页表是：请求 1 住块 0、1、6、8、11，请求 3 住块 3、4、7、9、12，请求 4 住块 2、5、10，空闲块 3。最后两行字幕：浪费只剩每个请求最后一块的空尾巴（此刻共 8 格）；对照：按最大长度连续预留，没写到的位置也一直占着。画面复位。"></video>
<figcaption>请求一结束块就还回去，下一个人马上用上。
  <span class="sub">（18 秒无声循环，Manim 渲染。块大小、块数、请求长度都是示意。）</span></figcaption></figure>

  <h3 id="s3-4">3.4　混合模型有两个池，预算要分</h3>
  <p>V4 的滑窗池存最近 128 个 token 的原始 KV，全长池存压缩后的整段。
    SGLang 开两个池共用一份预算，用 <code>--swa-full-tokens-ratio</code> 定怎么分。</p>
__FIG_S3_POOL__
  <p>两边一起满之后再多装人，吞吐几乎不涨（4.3）：瓶颈从装不下换成了算不动。</p>

  <div class="note warn"><span class="t">⚠️ 池子算错不一定报错</span>
    tpu-inference 跑 Qwen3.5 时块编号算多了约 3.5 倍，越界的编号被 JAX <b>静默截断</b>，多个请求写进同一块；症状是胡言乱语或崩溃（<code>tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:77</code>）。</div>

  <h3 id="s3-5">3.5　前缀复用：算过的不再算</h3>
  <p>很多请求开头一样（同一段 system prompt、多轮对话的前几轮），这段的 KV 留着，下一个直接接着算。</p>
  <ul>
    <li><b>RadixAttention。</b>SGLang 把缓存的前缀存成一棵基数树（开头相同的共用一段树枝），新请求沿树匹配最长前缀；论文报吞吐最多是当时其他系统的 6.4 倍（整套运行时的结果）。</li>
    <li><b>默认就开。</b>vLLM 早先的前缀缓存命中率低时反而拖慢，默认关；2025 年 1 月 V1 改了数据结构，命中率为 0 也只掉不到 1%，才默认打开。</li>
    <li><b>命中率可以很高。</b>DeepSeek 线上过半输入命中磁盘上的 KV。命中部分按十分之一计价（DeepSeek API 文档 2024-08）。</li>
  </ul>
__FIG_S3_DISK__
<figure class="fbox fwide" id="anim-kvstore">
<video src="media/topic06-kvstore.mp4" autoplay loop muted playsinline
       aria-label="KV 当存储动画。标题：KV 当存储：先找，找到就搬回，都没有才重算。中间三层，从上到下是 HBM（小、最快，3 格）、主机内存（大一些、慢一些，5 格）、SSD（最大、最慢，7 格），每格一块带字母的前缀 KV；右上角写：每层从左到右：最近用过 → 最久没用。开始时 HBM 里是 A、B、C，主机内存里是 D、E、F、G，SSD 里是 H、I、J、K。底下是首字要等多久的四行条。第一句字幕：每块是一段算好的前缀 KV；新请求先看自己的前缀存在哪一层。请求 1，前缀 A：HBM 命中；字幕：前缀 A 就在 HBM：直接用，接着往后算；首字条最短，写 HBM 里就有。请求 2，前缀 D：HBM 没有，主机内存命中；字幕：前缀 D 在主机内存：拉回 HBM，比重算 prefill 快；首字条写从主机内存拉回。D 放进 HBM 最前面，C 被挤出框外；字幕：HBM 满了：最久没用的 C 往下放到主机内存。请求 3，前缀 H：HBM、主机内存都没有，SSD 命中；字幕：前缀 H 在 SSD：拉回来也比重算快（DeepSeek：首字 13 s → 0.5 s）；首字条写从 SSD 拉回；H 进 HBM，B 被放到主机内存。请求 4，前缀 L：三层都没有；字幕：前缀 L 哪层都没有：只能重算 prefill；橙色的重算条慢慢长到最长，写重算 prefill，同时算出一块新的 L。L 进 HBM 最前面；字幕：HBM 满了，A 往下放；主机内存也满了，G 再推到 SSD。最后一句字幕：先找、再搬、最后才算；显存满了，冷的 KV 一层层往下放。底注：块数、层的大小都是示意；首字条只有 SSD 与重算两条按 DeepSeek 实测比例画。画面复位。"></video>
<figcaption>看每个请求往下找了几层，以及显存满时谁被挤下去。
  <span class="sub">（27 秒无声循环，Manim 渲染。层的容量、块的摆放和四个请求是示意，命中哪层由 LRU 现模拟。）</span></figcaption></figure>
  <p><b>线性注意力和滑窗难复用：</b>线性层的状态是整段压出来的，截不回任意位置；滑窗层早先的 KV 已经丢了。vLLM、SGLang 于是在块边界另存状态快照；tpu-inference 跑 Qwen3.5 得显式关掉（<code>--no-enable-prefix-caching</code>），否则断言错误（同一 README:95）。</p>

  <h3 id="s3-6">3.6　分层卸载：搬回来，还是重算</h3>
  <p>KV 还能往下放到主机内存、磁盘乃至远端存储。</p>
  <div class="note ok"><span class="t">盈亏带宽</span>
    搬回 ＝ 每 token KV 字节 × L ÷ 带宽；重算 ≈ 每 token FLOP × L ÷ 有效算力。L 是命中的前缀长度，有效算力是峰值乘利用率。<br>
    两者相等时：<b>带宽 ＝ 每 token KV 字节 × 有效算力 ÷ 每 token FLOP</b>。链路比它慢，就不如重算。<br>
    公式只算线性部分，注意力那项随 L² 涨，越长越该下放（图里算进了它）。负载高时重算挤占别人的算力、搬回能和计算重叠，盈亏点再往慢链路移。</div>
__FIG_S3_OFFLOAD__
  <p>V3 只要 2.19 GB/s；GQA 的 Llama-3.1-70B 每 FLOP 的 KV 多，链路至少 5.35 GB/s 才划算。</p>
  <ul>
    <li><b>llm-d</b> 在 TPU7x 上把 Qwen3-32B 的 KV 卸到主机内存：平均 TTFT 0.98 s → 0.56 s，P90 2.1 s → 0.5 s，输入输出合计吞吐 +8.9%（<code>llm-d/llm-d@accc6c5:guides/tiered-prefix-cache/README.md</code>）。</li>
    <li><b>Mooncake Store</b>：集群各机的内存和 SSD 拼成一个 KV 池，机器间用 RDMA（网卡直接读写对方内存，不经 CPU）互取。</li>
    <li><b>LMCache</b>：接在 vLLM、SGLang 下分层存 KV；论文（2025-10）报多轮问答、文档分析下吞吐最多是单跑 vLLM 的 15 倍；TPU 上的同类工具见 8.3。</li>
    <li><b>SGLang HiCache</b>：2025 年 9 月把分层缓存做进引擎（L1 显存、L2 主机内存、L3 远端存储）；官方博客报长上下文和多轮对话下吞吐最多 6 倍、首字最多降 80%（对全部重算）。vLLM 0.11 也自带了 CPU 下放（OffloadingConnector），下放成了两大引擎的标配。</li>
  </ul>

  <div class="note ok"><span class="t">这一节只要带走两件事</span>
    <b>① KV 池定了 batch 上限。</b><br>
    <b>② 复用和卸载是少算、少搬。</b>下放之前算清搬回来是不是比重算快。</div>

  <details class="foldfig"><summary><b>考考自己</b>：一颗 v7 芯片的 KV 池是 92.4 GiB，V3 跑 64K 上下文能装几条？要是把 KV 存成 fp8 呢？</summary>
  <p>每条 70,272 × 65,536 字节 ≈ 4.29 GiB，装 21 条。存 FP8 按 12.7 的布局每 token 40,016 字节，约装 37 条（严格减半才是 43 条）。</p></details>
'''

FIGS = {
    "__FIG_S3_LINEAGE__": ("fig-s3-lineage", "fig6-kv-lineage.svg", "topic06-fig-kvlineage.py",
        '<b>只有 DSA 那一列是下排变少，后两列变少的是上排。</b><br>'
        '<em>方块示意，不按比例；各数出处见文末台账（V3.2-Exp、V4 技术报告与 V4.1-Flash 模型卡）。</em>'),
    "__FIG_S3_BATCH__": ("fig-s3-batch", "fig6-kv-batch.svg", "topic06-fig-kv.py",
        '<b>纯注意力三条线往右一格人数减半；紫线先交固定状态，少得慢；红线 32K 起装不下。</b><br>'
        '<em>口径同正文；KV 按 bf16、attention 数据并行。本课推导。</em>'),
    "__FIG_S3_PAGED__": ("fig-s3-paged", "fig6-paged.svg", "topic06-fig-kv.py",
        '<b>绿色两块两条序列共用，橙色那块是 B 要改时复制的。</b><br>'
        '<em>每块 4 个 token 只为画得下；机制见 vLLM 论文（arXiv 2309.06180）。</em>'),
    "__FIG_S3_POOL__": ("fig-s3-pool", "fig6-swa-ratio.svg", "topic06-fig-kv.py",
        '<b>隔板挪到两池一起快满时最快；一边先满，另一边就空转。</b><br>'
        '<em>V4-Pro 在 GB300 上的 SGLang 实测，按单卡计；前两格输入 4K、第三格 8K，输出 1K，关前缀缓存。</em>'),
    "__FIG_S3_DISK__": ("fig-s3-disk", "fig6-disk-cache.svg", "topic06-fig-kv.py",
        '<b>上栏是一条请求能省多少，下栏是一天的输入里多少能这样省。</b><br>'
        '<em>首字延迟出自 DeepSeek API 文档 2024-08，命中比例出自 open-infra-index 2025-02，两次统计不同时；1/26 为本课推导。</em>'),
    "__FIG_S3_OFFLOAD__": ("fig-s3-offload", "fig6-offload.svg", "topic06-fig-kv.py",
        '<b>绿线始终在红线下面：链路够快，多长都该搬。</b><br>'
        '<em>灰虚线是盈亏带宽 2.19 GB/s；每芯片口径；有效算力按 FP8 峰值 50% 假设，两档带宽是示意值。本课推导。</em>'),
}

NUM_LOCK = [
    ("92.4", "fig6-kv-batch.svg"),
    ("70,272", "fig6-kv-batch.svg"),
    ("327,680", "fig6-kv-batch.svg"),
]
