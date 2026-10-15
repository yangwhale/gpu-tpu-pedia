# -*- coding: utf-8 -*-
"""专题六 · 第五节「单卡里的两块硬骨头：decode 注意力与 MoE」。图全在 topic06-fig-decode.py。"""

BODY = '''
  <p class="lead">一张卡上的 decode 有两块最难啃：注意力读的是每个人自己的 KV（5.1–5.4），MoE 的每个专家只分到几行（5.5）。<b>两块都是搬运问题。</b>最后补一条小 AllReduce（5.6），它卡的是延迟。</p>
  <p>注意力这边两招：2023 年 Flash-Decoding 把一条长序列切开并行，超长序列最多快 8 倍（5.1）；2024 年 DeepSeek-V2 的 MLA 从结构上把 KV 压薄，V3 每 token 70,272 B，约为同形状 MHA 的 1/57（3.1 表）。</p>

  <h3 id="s5-1">5.1　注意力：攒 batch 救不了它</h3>
  <p>注意力拆成几十个并行的「头」：查询头负责发问，KV 头存着被查的历史。decode 注意力每读 1 字节只做 G 次运算（G ＝ 每个 KV 头带几个查询头），推导见<a href="topic-03.html#s3-1">专题三 §3.1</a>。Llama-3.1-70B 64 个查询头配 8 个 KV 头，G＝8，离 v7 的屋脊 313 差了近 40 倍。</p>
  <p>攒 batch 分摊的是权重：读一遍，所有人共用。KV 是每条序列自己的历史，batch 翻倍它也翻倍。</p>
__FIG_S5_AMORTIZE__
  <p>batch 过了 53，一步的字节大半是 KV，注意力就成了主角。</p>
  <p>batch 小、序列长时还有一个毛病：一条序列的注意力只占一个计算单元，其余的闲着。
    Flash-Decoding（斯坦福 CRFM 博客，2023-10）把这条序列的 KV 切成几段，分给多个计算单元同时读，每段交出部分结果和自己的分母（softmax 把各分数取指数再除以总和，这个总和就是分母），最后把分母加齐了统一除，为什么不差一分见 <a href="#s7-1">7.1</a>。</p>

  <h3 id="s5-2">5.2　一个 kernel 装下 prefill 和 decode</h3>
  <p>同一步常混着 prefill 和 decode。RPA（Ragged Paged Attention，按页存 KV、一批里长短不齐也能一起算的注意力 kernel）把两者放进同一个 kernel，按这一批的成分分派到三份预先编译的版本：纯 decode、纯 prefill、两者混合（论文 arXiv 2604.15464 的 3.4、3.5 节）。</p>
  <p>它在 TPU7x 上跑 Llama 3 8B，decode 用到峰值带宽的 86%，prefill 用到峰值算力的 73%，三招拆解见<a href="topic-03.html#s9-2">专题三 §9.2</a>。</p>
  <p>每份版本各配一组块大小，这是最便宜的一档旋钮。Qwen3.5-397B 在 TPU7x 上的公开调优实录（Google 开发者博客）里，KV 页从 16 个 token 调到 256 个，并发 512 时 RPA 这个 kernel 的 decode 一步从 428 µs 降到 283 µs（原文称 kernel 级提速，不是整步）。块太小，向量单元反复停下来算地址。</p>

  <h3 id="s5-3">5.3　布局：窄维被补成整块</h3>
  <p>TPU 的显存按 4 KiB 一块的 tile 存：最内层（内存里连续排的那一维）补齐到 128 格，次内层 bf16 补齐到 16 行、fp8 补齐到 32 行。</p>
  <p>KV 头只有 1–2 个时，要是把头数放进这两维，它就被补成整块，多出来的全是空位。</p>
__FIG_S5_TILE__
  <p>RPA 的解法是加一个打包维、把 K 和 V 沿头维并起来，逼编译器选最小的 tile（论文 3.1 节）。</p>
  <p>换成 FP8：A 降到 32 KiB，仍是 64 倍；B 还是 8 KiB；只有 C 跟着减半到 512 B。</p>

  <h3 id="s5-4">5.4　V3 的 MLA：一个阶段一套算法</h3>
  <p><a href="topic-03.html#s5-2">专题三 §5.2</a> 推过「吸收」：把投影并进查询，128 个头直接对着同一条 576 维的潜向量算，K/V 不展开。那里留了一句「两套怎么装进一个引擎，留到专题六」。</p>
__FIG_S5_MLA__
  <p>答案看每个阶段缺什么：decode 缺带宽，用吸收；prefill 缺算力，要的是少算。每个头对每个历史 token，吸收要和 576 维的潜向量做点积、再按其中 512 维加权求和，共 1,088 次乘加；展开后每头 K 192 维、V 128 维，共 320 次，少 3.4 倍（V3 config 推导）。展开多读的字节，prefill 有富余带宽兜着。所以跑 V3 的 MLA，引擎里两套 kernel 都留着，按阶段选。</p>

  <h3 id="s5-5">5.5　decode 的 MoE：全算比排序便宜</h3>
  <p>decode 一批 256 个 token、每个选 8 个专家，DeepSeek-V3 的 256 个专家平均每个只分到 8 行。按专家排序、搬行、再搬回来，比这点矩阵乘还贵。</p>
  <p>换个算法：让所有 token 都过这个专家，没选中的乘 0。只要「全算」比「读一遍权重」快，它就藏在搬运的时间里。</p>
__FIG_S5_DENSE__
  <p>同一个思路也能用在搬行本身。tpu-inference#2674 在 batch 小时把排序和还原都换成「one-hot 矩阵 × 隐状态」，用一次规整的矩阵乘代替按下标取行。PR 里 Qwen3.5 并发 64 的实测：1K 进 8K 出总吞吐 +14.5%，8K 进 1K 出 +7.3%。合入后默认关着，要用环境变量设阈值。</p>

  <h3 id="s5-6">5.6　小 AllReduce 是延迟问题</h3>
  <p>batch 1 时，每层 TP 的 AllReduce 只有 hidden（每个字的向量长度，V3 是 7,168）× 2 字节，按带宽算几乎不花时间。</p>
__FIG_S5_ALLREDUCE__
  <p>图里灰条以外的三条出自论文 arXiv 2607.16100。小 decode 的同步要省次数，加带宽没用，第六节比 TEP 和 DEP 时会用到这条。</p>

  <p class="landing">⭐ 5.3 到 5.5 指向同一件事：<b>「少搬」从布局开始。</b>补齐的空位、展开的 K/V、搬来搬去的行都是没人要的字节，先去掉它们，再谈量化和加卡。</p>
'''

FIGS = {
    "__FIG_S5_AMORTIZE__": ("fig-s5-amortize", "fig6-s5-amortize.svg", "topic06-fig-decode.py",
        '<b>攒 batch 只分摊权重；KV 跟着 batch 线性长，注意力每字节始终只做 8 次运算。</b>'
        '<br><em>只记读显存的字节，没算激活值和 logits（给词表里每个字打的分）。</em>'),
    "__FIG_S5_TILE__": ("fig-s5-tile", "fig6-s5-tile.svg", "topic06-fig-decode.py",
        '<b>同一份 KV 装错箱就占 8～64 倍的地方；B 那种摆法换成 FP8 一个字节都不省。</b>'
        '<br><em>按默认 tile 推导；编译器对极窄的数组有时改用更小的 tile，见出处。</em>'),
    "__FIG_S5_MLA__": ("fig-s5-mla", "fig6-s5-mla.svg", "topic06-fig-decode.py",
        '<b>吸收：一条压缩 KV 128 个头共读，每字节 242 次运算，decode 用；展开：先复制 128 份，每字节 1 次，prefill 用。</b>'
        '<br><em>乘加按 2 次运算计，bf16；没算把查询吸收进去、把输出投影回来的那两次矩阵乘。</em>'),
    "__FIG_S5_DENSE__": ("fig-s5-dense", "fig6-s5-dense.svg", "topic06-fig-decode.py",
        '<b>一批 256 个 token 全过一遍专家，仍比读一遍它的权重快。</b>'
        '<br><em>DeepSeek-V3 一个路由专家，TPU v7 单 device，按峰值带宽和峰值算力推导。</em>'),
    "__FIG_S5_ALLREDUCE__": ("fig-s5-allreduce", "fig6-s5-allreduce.svg", "topic06-fig-decode.py",
        '<b>14 KiB 的 AllReduce，搬数据约 24 ns，实测 11.0 µs。</b>'
        '<br><em>4 张 GB200，小消息平均延迟（论文实测）；带宽项为本课推导，对数轴。</em>'),
}

NUM_LOCK = [
    ("8 KiB", "fig6-s5-tile.svg"),
]
