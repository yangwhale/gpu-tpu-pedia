# -*- coding: utf-8 -*-
"""专题六 · 第十二节「少搬字节：量化与 kernel 融合」。图在 topic06-fig-quant.py。"""

BODY = '''
  <p class="lead">上一节让搬一趟出更多字，这一节让每趟少搬、少停。
    <b>量化</b>让权重和 KV 占更少字节，<b>kernel 融合</b>去掉交接时的空等。
    2022 年底的 GPTQ 只压权重、算时还原，3 bit 的 OPT-175B 在 1 张 A100 上逐字速度是 5 张卡跑 FP16 的 3.25 倍；同年 Hopper 的张量核（GPU 里做矩阵乘的单元）直接算 FP8，DeepSeek-V3 连训练都用 FP8；到 Blackwell，4 bit 浮点（NVFP4）也进了张量核。</p>

  <h3 id="s12-1">12.1　位数减半，下限减半</h3>
  <p>batch 小时一步几乎全在读权重：8 颗 TPU v7 分着读 V3 的激活参数，FP8 的下限正是 1.2 节单芯片 5 毫秒的八分之一（本课推导，只算权重）。</p>
__FIG_S12_BYTES__
  <p>实测接近这个比例：vLLM 2024 年 7 月接入 FP8 权重，稠密的 Llama 3 70B 每字间隔最多降到 BF16 的 1/2，MoE 的 Mixtral 8x7B 最多降到 BF16 的 1/1.6。MoE 赚得少，原文没给原因；本课推测是只读选中的专家，读权重在一步里占比本就低。</p>
  <p>省下的显存多放 KV、多接请求（3.2），一趟搬运分给更多人。</p>

  <h3 id="s12-2">12.2　同叫 4 bit，差在缩放</h3>
  <p>4 bit 只有 16 个编码，必须几个数共用一个缩放倍数；MXFP4 和 NVFP4 差在块大小和缩放格式。</p>
__FIG_S12_MX__
  <p>块越小越贴合局部数值，精度越好，字节越多。格式细节见专题八 <a href="topic-08.html#s2">§2</a>。</p>

  <h3 id="s12-3">12.3　低精度直接训出来</h3>
  <p>2025 年的 Kimi K2 Thinking 在后训练做量化感知训练（QAT，训练时就模拟低精度），MoE 部分原生 INT4 权重，模型卡称低延迟模式生成约快 2 倍，没写对照精度。</p>
  <p>DeepSeek-V4 模型卡写明，MoE 专家用 FP4，其余大部分用 FP8。专家占参数的绝大部分，也是 decode 读得最多的部分，压它省得最多。</p>

  <h3 id="s12-4">12.4　FP4 先是被容量逼的</h3>
  <p>DeepSeek-R1 放到 TPU v7x-8 上，专家权重用 FP8 时每个 device 放不下，换 FP4 才腾出余量。</p>
__FIG_S12_CAP__
  <p>但 v7 公开规格只列 BF16 和 FP8 峰值，推测 FP4 只能当权重存，算前在片上解成 FP8 或 BF16。字节省了，多了转换，净收益只能实测。</p>

  <h3 id="s12-5">12.5　INT4 不是另一种 FP4</h3>
  <p>R1 的 FP4（E2M1）是浮点，值在 0 附近密、远处疏，靠块缩放对齐量级；Kimi K2.6 的 INT4 是 −8 到 7 等距排开的整数。</p>
__FIG_S12_R1__
  <p>K2.6 的记录里，加载时漏了图里「减 8」一步，模型输出崩成重复的乱词。</p>

  <h3 id="s12-6">12.6　kernel 融合：边界上不再干等</h3>
  <p>一次前向（模型从头算到尾一遍）要发上百个 kernel。每个 kernel 收尾、下一个开场时，没人在读权重。</p>
__FIG_S12_FUSE__
  <p>生产里先上的是 CUDA Graph 和 PDL：前者省掉 CPU 逐个发射 kernel 的开销，后者让下一个 kernel 提前准备。
    megakernel 把整次前向写成一个 kernel，下一段权重在上一段计算时就开始读。</p>
  <p>Hazy Research（2025-05）实测，vLLM、SGLang 最多用到 H100 带宽的 50%，megakernel 到 78%。模型和 batch 越小，空等占比越大，赚得越多。</p>
  <p>TPU 这边，Inferact 2026-09 用 Pallas（JAX 里手写 TPU kernel 的语言）写了 megakernel。16 颗 TPU v7 跑 Kimi K3，单用户超过每秒 700 token；16 张 GB200（vLLM）是 452 token。两个数都开了投机，按接受长度 6 换算。
    不开投机时，K3 和 Qwen 3.8 27B 在 batch 1 到 8 约是 GB200 的 1.4–2 倍（<a href="https://inferact.ai/blog/tpu-megakernels">Inferact 博客</a>）。</p>

  <h3 id="s12-kv">12.7　KV 也能量化</h3>
  <p>KV 存 FP8，每个数从 2 字节变 1 字节；vLLM 和 SGLang 都是一个 <code>--kv-cache-dtype</code> 参数。
    K3 官方高吞吐配方列为必开：BF16 的 KV 装不下每副本 128 个请求。</p>
  <p>缩放给多细看 checkpoint：vLLM 只设 <code>fp8</code> 不校准时，缩放全取 1.0；用 llm-compressor 拿数据校准，可以每个 K、V 张量一个，或每个 KV 头一个（后者目前只有 FlashAttention 后端支持）。开了 FP8 KV 掉精度，先查有没有校准过的缩放。</p>
  <p>实际布局不是严格一半。V3.2 每层还多存一条索引键（indexer key，3.1 那个小打分器挑 token 时读的），它本来就存 FP8，换了也不变小，所以同显存多接的请求比只算 MLA 时少。</p>
__FIG_S12_KVLAYOUT__
  <p>4 bit 已经落地：NVIDIA 2025-12 在 Blackwell 上放出 NVFP4 KV，比 FP8 KV 再省约一半，基准上精度损失不到 1%（NVIDIA 博客）。算注意力前仍解回 FP8，省的是容量和读的字节。同样显存多存了前缀，Qwen3-Coder-480B 上命中率比 FP8 KV 高 20%，首字延迟最多降到其约三分之一。</p>
  <p>2026-09 的 DeepSeek-V4.1-Flash 主 KV 原生存 FP4（每 16 个数一个缩放），FlashMLA 同时放出读它的 kernel，比原先的 FP8 KV 小 45%。</p>

  <div class="note ok"><span class="t">回到主线</span>
    <b>量化是少搬，融合是去掉空等</b>，两者都能和投机解码叠用：Inferact 那个数来自 megakernel 加 DSpark 投机，博客没提量化。</div>
'''

FIGS = {
    "__FIG_S12_KVLAYOUT__": ("fig-s12-kvlayout", "fig6-quant-kvlayout.svg", "topic06-fig-quant.py",
        '<b>看下面那条的 MLA 部分比红虚线多出的那截：没压的位置编码，加上缩放。</b><br>'
        '<em>FlashMLA 给 DeepSeek-V3.2 的布局，加上 indexer key，条长按字节比例；倍数为本课推导。</em>'),
    "__FIG_S12_BYTES__": ("fig-s12-bytes", "fig6-quant-bytes.svg", "topic06-fig-quant.py",
        '<b>看下限随位数怎么缩。</b><br>'
        '<em>本课推导：V3 激活参数、8 颗 v7、batch 1、带宽跑满，不含 KV 与通信。</em>'),
    "__FIG_S12_MX__": ("fig-s12-mx", "fig6-quant-mx.svg", "topic06-fig-quant.py",
        '<b>差在块大小和缩放格式。</b><br>'
        '<em>出自 OCP MX v1.0 规范与 NVIDIA NVFP4 博客。</em>'),
    "__FIG_S12_R1__": ("fig-s12-r1", "fig6-quant-r1.svg", "topic06-fig-quant.py",
        '<b>跟着一个权重走：存时加 8，读时不减，就差 8。</b><br><em>示意。</em>'),
    "__FIG_S12_CAP__": ("fig-s12-cap", "fig6-quant-cap.svg", "topic06-fig-quant.py",
        '<b>看虚线右边那截淡红：换成 GiB 读，越界更多。</b><br><em>README 原值，每 device；两种单位都超。</em>'),
    "__FIG_S12_FUSE__": ("fig-s12-fuse", "fig6-quant-fuse.svg", "topic06-fig-quant.py",
        '<b>每次 kernel 交接都要空等，加起来四成多；megakernel 把它捡回来。</b><br><em>时间轴按比例画。</em>'),
}

NUM_LOCK = []
