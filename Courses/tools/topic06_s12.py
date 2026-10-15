# -*- coding: utf-8 -*-
"""专题六 · 第十二节「少搬字节：量化与 kernel 融合」。图在 topic06-fig-quant.py。"""

BODY = '''
  <p class="lead">上一节让搬一趟出更多字，这一节让每一趟少搬一点、少停一会。
    <b>量化</b>让每个权重和 KV 占更少的字节，<b>kernel 融合</b>去掉 kernel 交接时的空等。
    这条路先走到 4 bit：2022 年底的 GPTQ、2023 年的 AWQ 只压权重，算时再还原。2024 年底 DeepSeek-V3 直接用 FP8 训练、按 FP8 发布；到 Blackwell，NVFP4 让 4 bit 浮点直接进了张量核（GPU 上专做矩阵乘的单元）。</p>

  <h3 id="s12-1">12.1　位数减半，下限减半</h3>
  <p>batch 小时，一步的时间几乎全花在读权重上。
    DeepSeek-V3 每个 token 激活 37B 参数，放在 8 颗 TPU v7 上分着读，BF16 一步至少 1.26 ms，FP8 0.63 ms，FP4 约 0.33 ms（本课推导，只算权重）。FP8 这个数是 1.2 节按一颗芯片算的 5 毫秒的八分之一。</p>
__FIG_S12_BYTES__
  <p>省下的显存能多放 KV、多接请求，顺带走了第四节分摊的路。</p>

  <h3 id="s12-2">12.2　同叫 4 bit，差在缩放</h3>
  <p>4 bit 只有 16 个编码，必须几个数共用一个缩放。
    MXFP4 每 32 个数共用一个 E8M0 缩放，平均 4.25 bit；NVFP4 每 16 个数共用一个 E4M3 缩放，平均 4.5 bit。</p>
__FIG_S12_MX__
  <p>块越小越贴合局部数值，精度越好，字节也越多。格式的细节见专题八 <a href="topic-08.html#s2">§2</a>。</p>

  <h3 id="s12-3">12.3　低精度直接训出来</h3>
  <p>发布时就是低精度，正在变成默认做法。
    Kimi K2 Thinking 在后训练阶段做量化感知训练（QAT），MoE 部分原生 INT4 权重，模型卡称生成速度约提升 2 倍。</p>
  <p>DeepSeek-V4 的模型卡写明，MoE 专家参数用 FP4，其余大部分用 FP8。
    专家占了参数的绝大部分，也是 decode 时读得最多的那部分。</p>
  <p>QAT 让模型训练时就适应舍入误差，部署方也不用再量化一遍。</p>

  <h3 id="s12-4">12.4　FP4 先是被容量逼的</h3>
  <p>DeepSeek-R1 放到 TPU v7x-8 上，专家权重用 FP8 时每个 device 要约 101.8 GB（README 原文单位，按 GiB 读也超），超过可用的 94.75 GiB；转成 FP4 后专家约 60.9 GB，连同注意力约 70 GB，才放得下。</p>
  <p>可 v7 的公开规格只列 BF16 和 FP8 峰值，没写 FP4 算力；按公开资料 FP4 只能当权重存，算之前在片上解成 FP8 或 BF16。字节省下了，转换的功夫加上了，净收益只能实测。</p>

  <h3 id="s12-5">12.5　INT4 不是另一种 FP4</h3>
  <p>R1 的 FP4（E2M1）是浮点，值在 0 附近密、远处疏，靠块缩放对齐量级；Kimi K2.6 的 INT4 是 −8 到 7 等距排开的整数。</p>
__FIG_S12_R1__
  <p>INT4 的读法也得对上。K2.6 的记录里，加载时把存成 0 到 15 的码直接当有符号数读、没先减 8，模型输出崩成重复的乱词。</p>

  <h3 id="s12-6">12.6　kernel 融合：边界上不再干等</h3>
  <p>一次前向要发上百个 kernel。每个 kernel 收尾、下一个开场时，没人在读权重。</p>
__FIG_S12_FUSE__
  <p>Hazy Research 2025-05 算过：Llama-3.2-1B 在 H100 上光读权重的上限是每秒约 1,350 次前向；每层 7 个 kernel、16 层、每个边界空等约 5 µs，就只剩约 770 次。</p>
  <p>生产里先上的是 CUDA Graph 和 PDL：前者省掉 CPU 逐个发射 kernel 的开销，后者让下一个 kernel 在上一个收尾前先做准备。
    megakernel 把整次前向写成一个 kernel，下一段的权重在上一段计算时就开始读。</p>
  <p>Hazy 的实测里，vLLM、SGLang 最多用到 H100 带宽的 50%，megakernel 到 78%，比 vLLM 快约 2.5 倍、比 SGLang 快 1.5 倍多。模型越小、batch 越小，空等占比越大，这招赚得越多。
    Mirage MPK（2025-06 开源，论文 2025-12）让编译器自动把模型变成一个 megakernel，论文报的端到端延迟最多降 1.7 倍。</p>
  <p>TPU 这边，Inferact 2026-09 用 Pallas（JAX 里手写 TPU kernel 的语言）写了 megakernel。16 颗 TPU v7 跑 Kimi K3，单用户超过每秒 700 token；对照的 16 张 GB200（vLLM）是 452 token，两边都按接受长度 6（每步平均收下 6 个字，见 11.2）换算。
    它一步约 8.5 ms，下一层的权重提前搬进片上内存，整套从零编译不到 90 秒；不开投机时，batch 1 到 8 约是 GB200 的 1.4–2 倍（<a href="https://inferact.ai/blog/tpu-megakernels">Inferact 博客</a>）。</p>

  <h3 id="s12-kv">12.7　KV 也能量化</h3>
  <p>KV 存成 FP8，每 token 的字节约减半，同样显存多装近一倍请求；vLLM 和 SGLang 都是一个 <code>--kv-cache-dtype</code> 参数（K3 的 SGLang 配方写 fp8_e4m3）。
    K3 官方的高吞吐配方把它列为必开：BF16 的 KV 装不下每副本 128 个请求。</p>
  <p>实际布局不是严格一半。FlashMLA 给 DeepSeek-V3.2 的 FP8 KV 每 token 每层 656 B：512 个 FP8 值、4 个 FP32 缩放，64 维位置编码（告诉模型每个字排第几）部分为了精度留在 BF16。
    BF16 是 1,152 B，656 B 是它的 57%，同样显存多装约 1.76 倍（本课推导）。</p>

  <div class="note ok"><span class="t">回到主线</span>
    <b>量化是少搬，融合是去掉空等</b>：一个让权重和 KV 更小，一个让读权重的流水线不断。它们和投机解码能叠在一起用：Inferact 那个数是 megakernel 加 DSpark 投机一起换来的，博客没提量化。</div>
'''

FIGS = {
    "__FIG_S12_BYTES__": ("fig-s12-bytes", "fig6-quant-bytes.svg", "topic06-fig-quant.py",
        '<b>FP4 一步的搬运下限只有 BF16 的四分之一多一点。</b><br>'
        '<em>本课推导：V3 激活参数、8 颗 v7、batch 1、带宽跑满，不含 KV 与通信。</em>'),
    "__FIG_S12_MX__": ("fig-s12-mx", "fig6-quant-mx.svg", "topic06-fig-quant.py",
        '<b>4.25 和 4.5 的差别全在块大小：32 个一块还是 16 个一块。</b><br>'
        '<em>出自 OCP MX v1.0 规范与 NVIDIA NVFP4 博客。</em>'),
    "__FIG_S12_R1__": ("fig-s12-r1", "fig6-quant-r1.svg", "topic06-fig-quant.py",
        '<b>INT4 存的时候加了 8；读时不减、直接当有符号数读，每个权重都差 8。</b><br><em>示意；12.4 里 R1 的占用数字和单位见出处。</em>'),
    "__FIG_S12_FUSE__": ("fig-s12-fuse", "fig6-quant-fuse.svg", "topic06-fig-quant.py",
        '<b>每次 kernel 交接都要空等，加起来四成多；megakernel 把它捡回来。</b><br><em>时间轴按比例画。</em>'),
}

NUM_LOCK = [
    ("1.26", "fig6-quant-bytes.svg"),
    ("0.63", "fig6-quant-bytes.svg"),
    ("0.33", "fig6-quant-bytes.svg"),
    ("4.25", "fig6-quant-mx.svg"),
    ("4.5", "fig6-quant-mx.svg"),
    ("1,350", "fig6-quant-fuse.svg"),
    ("770", "fig6-quant-fuse.svg"),
]
