# -*- coding: utf-8 -*-
"""专题六 · 第五节「单卡里的两块硬骨头：decode 注意力与 MoE」。图全在 topic06-fig-decode.py。"""

BODY = '''
  <p class="lead">座位排满以后，剩下的是每一步本身。一张卡上的 decode 有两块最难啃：注意力读的是每个人自己的 KV（5.1–5.4），MoE 的每个专家只分到几行（5.5）。<b>两块都是搬运问题。</b>最后补一条小 AllReduce（把各卡的部分和加起来再发回每张卡，5.6），它卡的是延迟。</p>

  <h3 id="s5-1">5.1　注意力：攒 batch 救不了它</h3>
  <p>注意力拆成几十个并行的「头」：查询头负责发问，KV 头存着被查的历史。decode 注意力每读 1 字节只做 G 次运算（G ＝ 每个 KV 头带几个查询头），推导见<a href="topic-03.html#s3-1">专题三 §3.1</a>。Llama-3.1-70B 64 个查询头配 8 个 KV 头，G＝8，离 v7 的屋脊 313 差了近 40 倍。</p>
__FIG_S5_AMORTIZE__
  <p>batch 小、序列长时，一条序列的注意力只占一个计算单元，其余闲着。
    Flash-Decoding（斯坦福 CRFM 博客，2023-10）把它的 KV 切成几段同时读，最后合并（<a href="#s7-1">7.1</a> 有图）。博客报长上下文下 decode 最多提速 8 倍。</p>
  <p>名字来自 2022 年的 FlashAttention：把注意力切块放进片上 SRAM 算，显存从随长度平方涨变成线性涨，长上下文这才跑得起来。</p>

  <h3 id="s5-2">5.2　一个 kernel 装下 prefill 和 decode</h3>
  <p>同一步常混着 prefill 和 decode。RPA（Ragged Paged Attention，按页存 KV、一批里长短不齐也能一起算的注意力 kernel）把两者放进同一个 kernel，按这一批的成分挑三份预编译版本之一：纯 decode、纯 prefill、混合（arXiv 2604.15464 §3.4–3.5）。GPU 上做同一件事的是 FlashInfer（MLSys 2025）：按页存 KV、按需生成 kernel 变体，已是 SGLang、vLLM 的注意力后端之一。</p>
  <p>它在 TPU7x 上跑 Llama 3 8B，decode 最多用到峰值带宽的 86%，prefill 最多用到峰值算力的 73%，三招拆解见<a href="topic-03.html#s9-2">专题三 §9.2</a>。</p>
  <p>每份版本各配一组块大小，这是最便宜的旋钮：块太小，向量单元反复停下来算地址。Qwen3.5-397B 在 TPU7x 上的公开调优（Google 开发者博客）把 KV 页从 16 个 token 调到 256 个，并发 512 时这个 kernel 的 decode 一步从 428 µs 降到 283 µs（kernel 级，不是整步）。</p>

  <h3 id="s5-3">5.3　布局：窄维被补成整块</h3>
  <p>TPU 的显存按 4 KiB 一块的 tile 存：最内层（内存里连续排的那一维）补齐到 128 格，次内层 bf16 补齐到 16 行、fp8 补齐到 32 行。</p>
  <p>KV 头只有 1–2 个，若把头数放进这两维，就被补成整块，多出的全是空位。</p>
__FIG_S5_TILE__
  <p>RPA 的解法是加一个打包维、把 K 和 V 沿头维并起来，逼编译器选最小的 tile（论文 3.1 节）。量化压不掉空位。</p>

  <h3 id="s5-4">5.4　V3 的 MLA：一个阶段一套算法</h3>
  <p><a href="topic-03.html#s5-2">专题三 §5.2</a> 推过「吸收」：把投影并进查询，128 个头直接对同一条 576 维潜向量算，K/V 不展开。那是 decode 用的，prefill 呢？</p>
__FIG_S5_MLA__
  <p>decode 缺带宽，用吸收；prefill 缺算力，要的是少算。每个头对每个历史 token，吸收要和 576 维的潜向量做点积、再按其中 512 维加权求和，共 1,088 次乘加；展开后每头 K 192 维、V 128 维，共 320 次，少 3.4 倍（V3 config 推导）。展开多读的字节，prefill 有富余带宽兜着。</p>

  <h3 id="s5-5">5.5　decode 的 MoE：全算比排序便宜</h3>
  <p>1.4 算过，decode 时每个专家平均只分到 8 行。按专家排序、搬行、再搬回来，比这点矩阵乘还贵。</p>
  <p>换个算法：让所有 token 都过这个专家，没选中的乘 0。</p>
__FIG_S5_DENSE__
  <p>搬行本身也能这样换：tpu-inference#2674 在 batch 小时把排序和还原换成「one-hot 矩阵 × 隐状态」，一次规整的矩阵乘代替按下标取行。Qwen3.5 并发 64 实测总吞吐 +14.5%（1K 进 8K 出）、+7.3%（8K 进 1K 出）；合入后默认关，要用环境变量开。</p>

  <h3 id="s5-6">5.6　小 AllReduce 是延迟问题</h3>
  <p>batch 1 时，每层 TP 的 AllReduce 只有 hidden × 2 字节（hidden 见 1.1）。</p>
__FIG_S5_ALLREDUCE__
  <p>消息这么小，时间花在启动和互相等待上，加带宽救不回来，要省的是同步次数（arXiv 2607.16100）。</p>

  <p class="landing">⭐ 从 2022 年的 FlashAttention 到 2026 年的 RPA，单卡 kernel 一直在做同一件事：<b>「少搬」从布局开始。</b>补齐的空位、展开的 K/V、搬来搬去的行都是没人要的字节，先去掉它们，再谈量化和加卡。</p>
'''

FIGS = {
    "__FIG_S5_AMORTIZE__": ("fig-s5-amortize", "fig6-s5-amortize.svg", "topic06-fig-decode.py",
        '<b>batch 越大橙段越高：过了约 53，一步的字节大半是 KV。</b>'
        '<br><em>只记读显存的字节，没算激活值和 logits（给词表里每个字打的分）。</em>'),
    "__FIG_S5_TILE__": ("fig-s5-tile", "fig6-s5-tile.svg", "topic06-fig-decode.py",
        '<b>同一份 KV，看三种摆法各占多大地方。</b>'
        '<br><em>按默认 tile 推导；编译器对极窄的数组有时改用更小的 tile，见出处。</em>'),
    "__FIG_S5_MLA__": ("fig-s5-mla", "fig6-s5-mla.svg", "topic06-fig-decode.py",
        '<b>吸收每字节 242 次运算，给 decode；展开每字节 1 次，给 prefill。</b>'
        '<br><em>乘加按 2 次运算计，bf16；没算把查询吸收进去、把输出投影回来的那两次矩阵乘。</em>'),
    "__FIG_S5_DENSE__": ("fig-s5-dense", "fig6-s5-dense.svg", "topic06-fig-decode.py",
        '<b>比两根条：全算短于读权重，就藏得住。</b>'
        '<br><em>DeepSeek-V3 一个路由专家，TPU v7 单 device，按峰值带宽和峰值算力推导。</em>'),
    "__FIG_S5_ALLREDUCE__": ("fig-s5-allreduce", "fig6-s5-allreduce.svg", "topic06-fig-decode.py",
        '<b>14 KiB：带宽项约 24 ns；小消息实测平均 11.0 µs。</b>'
        '<br><em>4 张 GB200，小消息平均延迟（论文实测）；带宽项为本课推导，对数轴。</em>'),
}

NUM_LOCK = [
]
