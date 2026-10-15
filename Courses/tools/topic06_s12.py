# -*- coding: utf-8 -*-
"""专题六 · 第十二节「少搬字节：量化与 kernel 融合」。图在 topic06-fig-quant.py。"""

BODY = '''
  <p class="lead">2022 年底 GPTQ、2023 年 AWQ 把权重压到 4 bit；到 Blackwell，NVFP4 让 4 bit 浮点直接进了张量核。上一节让搬一趟出更多字，这一节让每一趟少搬一点。
    <b>量化</b>让每个权重占更少的字节，<b>kernel 融合</b>让读权重的流水线不在 kernel 边界上停。</p>

  <h3 id="s12-1">12.1　位数减半，下限减半</h3>
  <p>batch 小时，一步的时间几乎全花在读权重上。
    DeepSeek-V3 每个 token 激活 37B 参数，放在 8 颗 TPU v7 上，BF16 一步至少 1.26 ms，FP8 0.63 ms，FP4 约 0.33 ms（本课推导，只算权重）。</p>
__FIG_S12_BYTES__
  <p>省下的显存可以多放 KV、多接请求，等于顺带走了第四节分摊的那条路。</p>

  <h3 id="s12-2">12.2　同叫 4 bit，差在缩放</h3>
  <p>4 bit 只有 16 个编码（FP4 的 E2M1 格式里 ±0 是两个编码，实际只有 15 个值），必须几个数共用一个缩放。
    MXFP4 每 32 个数共用一个 E8M0 缩放（8 位全是指数、没有尾数），平均 4.25 bit；NVFP4 每 16 个数共用一个 E4M3 缩放（4 位指数、3 位尾数的 FP8），平均 4.5 bit。</p>
__FIG_S12_MX__
  <p>块越小越贴合局部数值，精度越好，字节也越多。NVFP4 的缩放（E4M3）还能取非 2 的幂，MXFP4（E8M0）只能取 2 的幂。格式的细节见专题八 <a href="topic-08.html#s2">§2</a>。</p>

  <h3 id="s12-3">12.3　低精度直接训出来</h3>
  <p>发布时就是低精度，正在变成默认做法。
    Kimi K2 Thinking 在后训练阶段做量化感知训练（QAT），MoE 部分原生 INT4 权重，模型卡称生成速度约提升 2 倍。</p>
  <p>DeepSeek-V4 的模型卡写明，MoE 专家参数用 FP4，其余大部分用 FP8。
    专家占了参数的绝大部分，也是 decode 时读得最多的那部分。</p>
  <p>QAT 让模型在训练时就适应舍入误差，比训完再压精度更稳；部署方拿到的就是最终格式，不用自己再量化一遍。</p>

  <h3 id="s12-4">12.4　FP4 先是被容量逼的</h3>
  <p>DeepSeek-R1 放到 TPU v7x-8 上，专家权重用 FP8 时每个 device 要约 101.8 GB，超过约 94.75 GB 的上限。
    转成 FP4 后专家约 60.9 GB，连同注意力约 70 GB，才放得下（R1 FP4 README 实测值；README 写 GB，日志实为 GiB，按哪个单位都放不下）。</p>
__FIG_S12_R1__
  <p>同叫 4 bit，FP4 和 INT4 是两回事。R1 用的 FP4（E2M1）是浮点：1 位符号、2 位指数、1 位尾数，值在 0 附近密、远处疏，靠块缩放对齐量级。Kimi K2.6 用的 INT4 是整数，−8 到 7 等距排开。</p>
  <p>INT4 的读法也得对上。对称 INT4 在文件里整体加 8 存成 0 到 15，加载时必须减 8，少减这一次，读出的数就全错位。K2.6 的记录里出过这个错，模型输出崩成重复的乱词。</p>
  <p>格式还要和硬件对上。v7 的矩阵单元没有 FP4 通路（公开规格只列 BF16 和 FP8 峰值），FP4 只能当权重存，算之前在片上先解成 FP8 或 BF16；字节省下了，转换的功夫却加上了，净收益只能实测。</p>
  <p>KV 也能存 FP8，但要先看布局：窄维（比如只有 1–2 个 KV 头的那一维，比 TPU 内存 tile 的行数还少）被补成整块时，FP8 只是让空位变多（§5.3）。</p>

  <h3 id="s12-5">12.5　kernel 融合：边界上不再干等</h3>
  <p>一次前向要发上百个 kernel。每个 kernel 收尾、下一个开场时，没人在读权重。</p>
__FIG_S12_FUSE__
  <p>Hazy Research 算过：Llama-3.2-1B 在 H100 上光读权重的上限是每秒约 1,350 次前向；每层 7 个 kernel、16 层、每个边界空等约 5 µs，就只剩约 770 次。</p>
  <p>CUDA Graph 只省掉 CPU 发射那一侧的开销，GPU 上 kernel 交接时的空当还在。
    megakernel 把整次前向写成一个 kernel，下一段的权重在上一段计算时就开始读。</p>
  <p>Hazy 的实测里，vLLM、SGLang 最多用到 H100 带宽的 50%，megakernel 到 78%，比 vLLM 快约 2.5 倍、比 SGLang 快 1.5 倍多。
    Mirage MPK 让编译器自动把模型变成一个 megakernel，论文报的端到端延迟最多降 1.7 倍。</p>
  <p>TPU 这边，Inferact 用 Pallas 写了 megakernel。16 颗 TPU v7 跑 Kimi K3，单用户超过每秒 700 token；对照的 16 张 GB200 是 452。
    它一步约 8.5 ms、每步接受 3 到 6 个，按 6 个算超过每秒 700；下一层的权重提前搬进片上内存；整套从零编译用时不到 90 秒（<a href="https://inferact.ai/blog/tpu-megakernels">Inferact 博客</a>）。</p>

  <div class="note ok"><span class="t">回到主线</span>
    <b>量化和融合都是少搬</b>：一个让每个权重更小，一个让读权重的流水线不断。它们和投机解码能叠在一起用：Inferact 那个数是 megakernel 加 DSpark 投机一起换来的，博客没提量化。</div>
'''

FIGS = {
    "__FIG_S12_BYTES__": ("fig-s12-bytes", "fig6-quant-bytes.svg", "topic06-fig-quant.py",
        '<b>FP4 一步的搬运下限只有 BF16 的四分之一多一点。</b><br>'
        '<em>本课推导：V3 激活参数、8 颗 v7、batch 1、带宽跑满，不含 KV 与通信。</em>'),
    "__FIG_S12_MX__": ("fig-s12-mx", "fig6-quant-mx.svg", "topic06-fig-quant.py",
        '<b>4.25 和 4.5 的差别全在块大小：32 个一块还是 16 个一块。</b><br>'
        '<em>出自 OCP MX v1.0 规范与 NVIDIA NVFP4 博客。</em>'),
    "__FIG_S12_R1__": ("fig-s12-r1", "fig6-quant-r1.svg", "topic06-fig-quant.py",
        '<b>INT4 存的时候加 8，读的时候少减一次，每个权重都偏大 8。</b><br><em>示意；FP8 放不放得下的占用数字见出处。</em>'),
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
