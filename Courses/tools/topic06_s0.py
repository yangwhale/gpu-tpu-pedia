# -*- coding: utf-8 -*-
"""专题六 · 第零节「一次请求走一遍」。图在 topic06-fig-intro.py。"""

BODY = '''
  <p class="lead">同一个模型，吞 prompt 那一段吃算力，往外吐字那一段吃带宽。
    <b>一次请求里，前后两段要的硬件正好相反。</b></p>

  <h3 id="s0-1">0.1　2022 年以来，推理怎么变便宜</h3>
  <p>2022 年的推理服务，一批请求要等最长的那个写完才一起退出；KV（已读内容的缓存）按最大长度预留，显存里真正存着 token（大致一个字或半个词，本课常直接叫“字”）的只有两到四成。</p>
__FIG_S0_STORY__
  <ul>
    <li><b>先补调度和显存（2022–2023，第三、四节）。</b>Orca（逐步调度）让每一步都能有人进出，PagedAttention（KV 分页）腾出空位给新请求。</li>
    <li><b>同时把 KV 压薄（2023 起，第三节、<a href="#s12-kv">12.7</a>）。</b>GQA（多个头共用 KV）、MLA（KV 压成一条）从结构上少存，部署时再存成 FP8（每个数 8 位）。</li>
    <li><b>再补「一趟只出一个字」（2022 底–2024，第十一节）。</b>投机解码（先猜后验）、MTP（模型自带猜字层）让大模型搬一趟权重验好几个字。</li>
    <li><b>同时压权重（2022–2025，第十二节）。</b>每个数从 16 bit 一路压到 4 bit。</li>
    <li><b>然后拆开放（2023 底–2025，第六至九节）。</b>prefill（吞 prompt）和 decode（吐字）分家，MoE（每字只走几个专家）的专家分到许多张卡上（大 EP，专家并行）。</li>
    <li><b>同时把 KV 当成存储（2023 底–2025，3.5–3.6）。</b>算过的前缀留着复用，集群的内存和 SSD 拼成 KV 池。DeepSeek 线上过半的输入不用重算。</li>
  </ul>
  <p>GQA、MLA 是模型侧的进步，图里用虚线卡标出，不记到系统头上；全课的账也建在模型上：DeepSeek-V3（2024 年 12 月）用 MoE，总参数 6,710 亿，每个 token 激活的参数约 370 亿。
    同等 GPT-3.5 水平的推理价格，2022 年 11 月到 2024 年 10 月降了 280 多倍（斯坦福 AI Index 2025）；报告把它主要归给越来越强的小模型，硬件另算，本课认为上面这条路也在里面。</p>
  <h3 id="s0-2">0.2　一个请求的一生</h3>
__FIG_S0_LIFE__
  <p>用户只感觉得到两个数：等第一个字等了多久（TTFT），之后每个字隔多久（TPOT）。
    前者由 prefill 定，后者由 decode 定，口径见第二节。</p>

  <h3 id="s0-3">0.3　全课一句话</h3>
  <div class="note ok"><span class="t">一句话</span>
    <b>decode 每一步都要把权重和 KV 从显存里搬一趟。推理的每一招，要么让这一趟分给更多 token，要么干脆少搬。</b><br>
    还有一类招数不碰这一趟本身：把吞 prompt 和吐字拆开，各放到合适的机器上。</div>
__FIG_S0_MAP__

  <details class="foldfig"><summary><b>遇到不认识的词，回这儿查</b>：prefill、decode、KV cache、TTFT、TPOT、吞吐、屋脊点、chip 与 device、PD 分离、投机解码……（每条一句话）</summary>
  <ul style="line-height:1.9">
    <li><b>prefill</b>：把整段 prompt 一次喂进模型算完，顺手产出第一个输出 token。</li>
    <li><b>decode</b>：之后每一步只往外出一个 token，出完接着喂回去算下一个。</li>
    <li><b>KV cache</b>：每个处理过的 token 在每一层留下的 K 和 V，后面的 token 算注意力时要回头读它们，所以一直存在显存里（专题三）。</li>
    <li><b>TTFT</b>：从请求发出到收到第一个输出 token 的时间；含不含排队要写清。</li>
    <li><b>TPOT 与 ITL</b>：TPOT 是一个请求除第一个 token 外平均每个 token 花的时间；ITL 是相邻两个 token 之间的每一次间隔，看 p99 时用它。</li>
    <li><b>batch 与并发</b>：并发是同时在服务的请求数；batch 是引擎这一步实际拼在一起算的请求（或 token）数，通常不超过并发。</li>
    <li><b>吞吐</b>：每秒出多少 token；要写清总量还是每卡、分母含不含 prefill 卡、只数输出还是输入加输出。</li>
    <li><b>每 token 激活的参数</b>：MoE 模型每个 token 只经过挑中的几个专家，真正参与计算的那部分参数。和「激活值」（每层算出来的中间结果）不是一回事。</li>
    <li><b>HBM 带宽</b>：芯片每秒能从自带显存读写多少字节；TPU v7 每芯片 7.37 TB/s。</li>
    <li><b>屋脊点</b>：峰值算力 ÷ 显存带宽，即每读 1 字节至少要做多少次运算才不被带宽拖住；v7 按 bf16 算约 313（专题二）。</li>
    <li><b>chip 与 device</b>：一颗 TPU v7 芯片对软件显示成 2 个 device；本课的规格按每芯片写，并行度按 device 数。</li>
    <li><b>PD 分离</b>：prefill 和 decode 放到两批不同的机器上跑，中间把 KV cache 传过去（专题五第六节）。</li>
    <li><b>投机解码</b>：先用便宜的办法猜出后面几个 token，再让大模型一步把它们一起验掉，猜对几个就一次出几个（第十一节）。</li>
    <li><b>MTP</b>：多 token 预测，训练时在主干后面多挂一层 transformer（共享词表和输出头），去猜再下一个字，推理时拿它当投机解码的草稿（第十一节）。</li>
    <li><b>MXU</b>：TPU 里做矩阵乘的单元，v7 上是 256×256 的一块；喂进去的行数不够，它就空转。</li>
    <li><b>TP、EP、DP、DCP</b>：把模型分到多张卡的四种切法。TP 把一层的矩阵切开分给几张卡；EP 把不同专家放到不同卡上；DP 整套复制几份、各接各的请求；DCP 把一条长序列的 KV 按 token 分到几张卡（第七节）。前三种专题五逐个讲过。</li>
    <li><b>TEP、DEP、dep8</b>：MoE 推理的两种切法，第六节细讲。TEP 是注意力按 TP 切，DEP 是注意力按数据并行切、每张卡带自己的一批人；dep8 指 8 路 DEP。</li>
    <li><b>rank</b>：参与并行的一个进程，通常一张卡（或一个 device）一个。</li>
    <li><b>NVLink 域</b>：一组用 NVLink 直连、能互读显存的 GPU；GB200／GB300 NVL72 一柜 72 块是一个域，出了域就走慢得多的网卡。</li>
    <li><b>切片</b>：一组用 ICI 直连的 TPU 芯片，档位是 4、8、16、32……颗；切片之间走数据中心网络（DCN）。</li>
    <li><b>v7 与 v7x</b>：同一款芯片（TPU v7，Ironwood），云上和代码里写作 v7x；切片名按 device 数，v7x-8 是 4 颗芯片。</li></ul></details>

  <details class="foldfig"><summary><b>出场角色</b>：本课反复出现的模型、硬件和软件，各一句话</summary>
  <ul style="line-height:1.9">
    <li><b>DeepSeek-V3 ／ R1</b>：MoE，注意力用 MLA；R1 是同一架构的推理版。本课的账大多拿 V3 算。</li>
    <li><b>DeepSeek-V4-Pro ／ V4-Flash</b>：DeepSeek 的下一代 MoE；本课的 GPU 实测大多来自 V4-Pro 在 GB300 上的部署记录。</li>
    <li><b>Kimi K3 ／ K2.6</b>：Moonshot AI 的 MoE；K3 的注意力里混了 KDA 线性注意力层。</li>
    <li><b>Qwen3.5-397B</b>：通义千问的混合模型，60 层里 45 层是 GDN 线性注意力，只有 15 层存传统 KV。</li>
    <li><b>其他模型</b>：Gemma4-31B、Qwen3-Coder-480B、GLM-5.2、MiMo-V2-Flash、MiniMax-M2.5 只在个别实测里出现。</li>
    <li><b>TPU v7</b>：芯片之间用 ICI 直连成 3D torus。</li>
    <li><b>GB200 ／ GB300</b>：NVIDIA Blackwell 代的整柜系统（NVL72）。H100、H200、H800、H20 是上一代 Hopper，后两种是带宽或算力受限的出口版本。</li>
    <li><b>vLLM、SGLang</b>：两大开源推理引擎；tpu-inference 是 vLLM 在 TPU 上的后端。</li>
    <li><b>Dynamo、llm-d</b>：管多台机器的编排层，负责路由、PD 分离和扩缩容；前者出自 NVIDIA，后者跑在 Kubernetes 上。</li>
    <li><b>Mooncake、NIXL、LMCache</b>：搬 KV 和存 KV 的库，第三、八节细讲。</li></ul></details>
'''

FIGS = {
    "__FIG_S0_STORY__": ("fig-s0-story", "fig6-story.svg", "topic06-fig-story.py",
        '<b>每张卡是一步：卡上写原文报告的提升和基线，没有可对照的倍数就写做法。</b><br>'
        '后一个的基线常常就是前一个，所以这些倍数不能连乘。'),
    "__FIG_S0_LIFE__": ("fig-s0-life", "fig6-life.svg", "topic06-fig-intro.py",
        '<b>红点是第一个字，之后每个绿格出一个字；紫色阶梯每一步都在长。</b><br>'
        '<em>示意时间线，长短不按比例。</em>'),
    "__FIG_S0_MAP__": ("fig-s0-map", "fig6-map.svg", "topic06-fig-intro.py",
        '<b>每出一个字搬一趟；三根杠杆：车上多坐人、货变轻、开两个仓库。</b><br>'
        '<em>归类是本课的分法，标签是讲这一招的节。</em>'),
}

NUM_LOCK = []
