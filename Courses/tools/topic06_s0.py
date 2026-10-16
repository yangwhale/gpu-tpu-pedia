# -*- coding: utf-8 -*-
"""专题六 · 第零节「一次请求走一遍」。图在 topic06-fig-intro.py。"""

BODY = '''
  <p class="lead">同一个模型，吞 prompt 那一段（prefill）吃算力，往外吐字那一段（decode）吃带宽。
    <b>一次请求里，前后两段要的硬件正好相反。</b></p>

  <h3 id="s0-1">0.1　2022 年以来，推理怎么变便宜</h3>
  <p>2022 年的推理服务，一批请求要等最长的那个写完才一起退出。Orca 的连续批让每一步都能有人进出，同延迟下吞吐是老办法 FasterTransformer 静态批的 36.9 倍。可座位数由 KV（已读内容的缓存）定：KV 按最大长度预留，显存里真正存着 token（大致一个字，本课常叫“字”）的只有两到近四成（3.3）。</p>
  <p>2023 年 PagedAttention 把 KV 分页，利用率到 96%。浪费省掉以后，坐几个人就看 KV 池装得下几条 KV，偏偏上下文一路变长：2023 年 11 月 GPT-4 Turbo 开到 128K，2024 年 2 月 Gemini 1.5 Pro 预览 1M。模型侧用 GQA、MLA 把每条压薄；系统侧让相同的开头不再重算（SGLang 的 RadixAttention），存不下的往内存、SSD 放（Mooncake，给 Kimi 线上用）。</p>
  <p>座位坐满后，长 prompt 进门那一步会卡住所有正在出字的人。2024 年 Sarathi-Serve 把 prefill 切块塞进 decode 的步里，可这只摊平了打扰；Splitwise、DistServe 干脆把两段分到两批机器（PD 分离）。代价是每个请求多传一趟 KV，还多了一份 prefill 与 decode 的机器配比要调。</p>
  <p>DeepSeek-V3 这类 MoE（每个字只走几个专家）又把拼好的 batch 拆散：256 个专家选 8 个，每个专家只分到几行。SGLang 的 DP attention、DeepSeek 的大 EP 把专家铺到上百张卡上，凑够行数。随之而来的是每个 rank 要自备一份状态，K3 上 DEP8 直接显存不足；decode 快了，prefill 又喂不上。</p>
  <p>2024 年 9 月的 o1、2025 年 1 月的 DeepSeek-R1 一题要写几百到几千个思考 token（R1 论文），交互档（同样的出字速度）又限住 batch，“读一趟权重出一个字”成了硬上限。投机解码、MTP 让一趟验好几个字；剩下能省的是字节，FP8、FP4 和合并 kernel（芯片上调起的一个计算函数）接着压。同一套 GB300（NVIDIA 整柜 GPU）跑 DeepSeek-V4-Pro，光改软件，同一交互档每卡吞吐已是首发日（Day-0）的 5.1 倍（含 MTP）。</p>
__FIG_S0_CHAIN__
__FIG_S0_TWOLONG__
  <p>全课的账建在 DeepSeek-V3（2024 年 12 月）上：MoE，总参数 6,710 亿，每个 token 激活的参数约 370 亿。
    同等 GPT-3.5 水平的推理这两年便宜得很快（斯坦福 AI Index 2025 按市场价统计，本课不引价格口径的数），报告主要归给更强的小模型；系统这一侧能不能单独拆出来，14.3 再说。</p>
  <p>这几年的引擎主要是 vLLM 和 SGLang 两家轮流领先。SGLang 2024 年 1 月首发时，博客写明用了 vLLM 的部分代码；一年后 vLLM 重写成 V1，博客反过来说受 SGLang 等引擎显著影响。分块 prefill、接 Mooncake 是 vLLM 先；MLA 优化、MTP、EAGLE-3、大 EP 是 SGLang 先。更多部件两家从同一个货架上拿：FlashInfer、xgrammar、DeepEP 与 FlashMLA、Mooncake、NIXL。</p>
__FIG_S0_ENGINES__
  <h3 id="s0-2">0.2　一个请求的一生</h3>
__FIG_S0_LIFE__
  <p>用户只感觉得到两个数：等第一个字多久（TTFT），之后每个字隔多久（TPOT）。前者由 prefill 定，后者由 decode 定（第二节）。</p>

  <h3 id="s0-3">0.3　全课一句话</h3>
  <div class="note ok"><span class="t">一句话</span>
    <b>decode 每一步都要把权重和 KV 从显存里搬一趟。推理的每一招，要么让这一趟分给更多 token，要么干脆少搬。</b><br>
    还有一类不碰这一趟：把两段拆开，各放合适的机器。</div>
__FIG_S0_MAP__

  <details class="foldfig"><summary><b>遇到不认识的词，回这儿查</b></summary>
  <ul style="line-height:1.9">
    <li><b>prefill</b>：整段 prompt 一次算完，顺手产出第一个输出 token。</li>
    <li><b>decode</b>：之后每步只出一个 token，再喂回去算下一个。</li>
    <li><b>KV cache</b>：每个处理过的 token 在每层留下的 K 和 V，后面的 token 算注意力要回头读，所以常驻显存（专题三）。</li>
    <li><b>TTFT</b>：请求发出到收到第一个输出 token 的时间；含不含排队要写清。</li>
    <li><b>TPOT 与 ITL</b>：TPOT 是首字之后平均每个 token 的时间；ITL 是相邻两个 token 的间隔，看 p99 用它。</li>
    <li><b>goodput</b>：满足延迟要求的那部分吞吐（第二节）。</li>
    <li><b>交互档</b>：按每个用户看到的出字速度分档；只在同一档里比每卡吞吐（第二节）。</li>
    <li><b>Little 定律</b>：系统里的人数 ＝ 到达速率 × 每人停留时间（第四节）。</li>
    <li><b>connector</b>：推理引擎里在 prefill、decode 两端收发 KV 的插件（第八节）。</li>
    <li><b>batch 与并发</b>：并发是同时在服务的请求数；batch 是引擎这一步实际拼在一起算的数，通常不超过并发。</li>
    <li><b>吞吐</b>：每秒出多少 token；要写清总量还是每卡、分母含不含 prefill 卡、分子含不含输入。</li>
    <li><b>每 token 激活的参数</b>：MoE 每个 token 只经过挑中的几个专家，这部分参数才参与计算；不是「激活值」（每层的中间结果）。</li>
    <li><b>HBM 带宽</b>：芯片每秒从自带显存读写的字节数；TPU v7 每芯片 7.37 TB/s。</li>
    <li><b>屋脊点</b>：峰值算力 ÷ 显存带宽，即每读 1 字节至少做几次运算才不被带宽拖住；v7 bf16 约 313（专题二）。</li>
    <li><b>chip 与 device</b>：一颗 TPU v7 芯片对软件显示成 2 个 device；规格按芯片写，并行度和切片名按 device 数（v7x-8 是 4 颗芯片）。</li>
    <li><b>PD 分离</b>：prefill 和 decode 分到不同机器，中间传 KV cache（专题五第六节）。</li>
    <li><b>投机解码</b>：先便宜地猜后面几个 token，大模型一步一起验，猜对几个就出几个（第十一节）。</li>
    <li><b>MTP</b>：多 token 预测，训练时在主干后多挂一层 transformer 猜再下一个字，推理时当投机解码的草稿（第十一节）。</li>
    <li><b>MXU</b>：TPU 里做矩阵乘的单元，v7 上 256×256；喂的行数不够就空转。</li>
    <li><b>TP、EP、DP、DCP</b>：四种分卡切法：TP 切开一层的矩阵；EP 把不同专家放不同卡；DP 整套复制、各接各的请求；DCP 把一条长序列的 KV 按 token 分到几张卡（第七节）。</li>
    <li><b>TEP、DEP、dep8</b>：MoE 推理的两种切法（第六节）：TEP 注意力按 TP 切；DEP 注意力按数据并行切，每卡带自己的一批人；dep8 即 8 路 DEP。</li>
    <li><b>rank</b>：参与并行的一个进程，通常一卡一个。</li>
    <li><b>NVLink 域</b>：用 NVLink 直连、能互读显存的一组 GPU；NVL72 一柜 72 块是一个域，出域走慢得多的网卡。</li>
    <li><b>切片</b>：ICI 直连的一组 TPU 芯片；切片之间走数据中心网络（DCN）。</li>
    <li><b>v7 与 v7x</b>：同一款芯片（Ironwood），云上和代码里写 v7x。</li></ul></details>

  <details class="foldfig"><summary><b>出场角色</b>：本课反复出现的模型、硬件和软件</summary>
  <ul style="line-height:1.9">
    <li><b>DeepSeek-V3 ／ R1</b>：MoE，注意力用 MLA；R1 是同架构的推理版。</li>
    <li><b>DeepSeek-V4-Pro ／ V4-Flash</b>：DeepSeek 2026-04 的 MoE；本课 GPU 实测大多来自 V4-Pro 在 GB300 上的部署记录。</li>
    <li><b>Kimi K3 ／ K2.6</b>：Moonshot AI 的 MoE；K3 的注意力里混了 KDA（Kimi Delta Attention）线性注意力层。</li>
    <li><b>Qwen3.5-397B</b>：通义千问的混合模型，60 层里 45 层是 GDN（Gated DeltaNet）线性注意力，只有 15 层存传统 KV。</li>
    <li><b>TPU v7</b>：芯片间用 ICI 直连成 3D torus。</li>
    <li><b>GB200 ／ GB300</b>：NVIDIA Blackwell 代整柜系统（NVL72）。H100、H200、H800、H20 是上一代 Hopper，后两种是出口受限版。</li>
    <li><b>vLLM、SGLang</b>：两大开源推理引擎；tpu-inference 是 vLLM 在 TPU 上的后端。</li>
    <li><b>Dynamo、llm-d</b>：多机编排层，管路由、PD 分离和扩缩容；前者出自 NVIDIA，后者跑在 Kubernetes 上。</li>
    <li><b>Mooncake、NIXL、LMCache</b>：搬 KV、存 KV 的库（第三、八节）。</li></ul></details>
'''

FIGS = {
    "__FIG_S0_CHAIN__": ("fig-s0-chain", "fig6-chain.svg", "topic06-fig-chain.py",
        '<b>红字是下一环的开头：每一招解决一个问题，也留下一个。</b><br>'
        '<em>倍数各对各的基线，所以只能一环一环看。</em>'),
    "__FIG_S0_ENGINES__": ("fig-s0-engines", "fig6-engines.svg", "topic06-fig-engines.py",
        '<b>谁先做、谁跟进：蓝色小条是在跟 vLLM，红色小条是在跟 SGLang。</b><br>'
        '<em>间隔按两边日期推算，往下取整；只凭推测的借鉴一律不标。</em>'),
    "__FIG_S0_TWOLONG__": ("fig-s0-twolong", "fig6-twolong.svg", "topic06-fig-story.py",
        '<b>深紫只在 ② 变厚，③ 多出来的全是浅紫。</b><br>'
        '<em>格子数不是实测，只为三块能放在一起比。</em>'),
    "__FIG_S0_LIFE__": ("fig-s0-life", "fig6-life.svg", "topic06-fig-intro.py",
        '<b>红点是第一个字，之后每个绿格出一个字；紫色阶梯每一步都在长。</b><br>'
        '<em>示意时间线，长短不按比例。</em>'),
    "__FIG_S0_MAP__": ("fig-s0-map", "fig6-map.svg", "topic06-fig-intro.py",
        '<b>蓝色那组节最多：第四到第七节加第十一节，都在让一趟多坐几个人。</b><br>'
        '<em>归类是本课的分法，标签是讲这一招的节。</em>'),
}

NUM_LOCK = [
    ("36.9", "fig6-chain.svg"),
    ("5.1", "fig6-chain.svg"),
    ("96%", "fig6-chain.svg"),
]
