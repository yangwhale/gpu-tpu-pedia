# -*- coding: utf-8 -*-
"""专题六 · 第零节「一次请求走一遍」。图在 topic06-fig-intro.py。"""

BODY = '''
  <p class="lead">同一个模型，吞 prompt 那一段（prefill）吃算力，往外吐字那一段（decode）吃带宽。
    <b>一次请求里，前后两段要的硬件正好相反。</b></p>

  <h3 id="s0-1">0.1　2022 年以来，推理怎么变便宜</h3>
  <p>2022 年的推理服务，一批请求要等最长的那个写完才一起退出；KV（已读内容的缓存）按最大长度预留，显存里真正存着 token（大致一个字，本课常叫“字”）的只有两到四成（3.3）。</p>
  <p>上下文和输出也一路变长：2023 年 11 月 GPT-4 Turbo 开到 128K，2024 年 2 月 Gemini 1.5 Pro 预览 1M，KV 随长度线性涨；2024 年 9 月的 OpenAI o1、2025 年 1 月的 DeepSeek-R1 先想再答，一题要写几百到几千个思考 token（R1 论文）。</p>
__FIG_S0_TWOLONG__
__FIG_S0_STORY__
  <ul>
    <li><b>先补调度和显存（泳道：调度、KV 与注意力；第三、四节）。</b>Orca（逐步调度，又叫连续批）让每一步都能有人进出，同延迟吞吐是老办法 FasterTransformer 静态批的 36.9 倍；PagedAttention（KV 分页）腾出空位；分块 prefill 把长 prompt 切开塞进 decode 的步里。</li>
    <li><b>同时把 KV 压薄（泳道：KV 与注意力；第三节、<a href="#s12-kv">12.7</a>）。</b>GQA（多个头共用 KV）、MLA（KV 压成一条）从结构上少存，V3 每 token 的 KV 约是同形状 MHA（每个头各存一份）的 1/57（本课推导）；部署时再把 KV 存成 8 bit（FP8）。</li>
    <li><b>KV 当成存储（泳道：复用与下放；3.5–3.6、第八节）。</b>先在单机里复用前缀（请求间相同的开头），SGLang（含 RadixAttention）吞吐最多是当时其他系统的 6.4 倍；Mooncake 再把整个集群的内存和 SSD 拼成 KV 池，给 Kimi 线上用。</li>
    <li><b>再拆开放（泳道：拆开放；第六至九节）。</b>prefill 和 decode 分家（PD 分离），DistServe 同等延迟要求下能接的请求是不拆的 7.4 倍；MoE（每个字只走几个专家）的专家再分到许多张卡上（大 EP）。</li>
    <li><b>补「一趟只出一个字」（泳道：解码提速；第十一节）。</b>投机解码、MTP（模型自带的猜字头）让大模型读一趟权重验好几个字。投机解码 2022 年底就有，单请求速度是不投机的 2–3 倍，输出变长后才成主流。</li>
    <li><b>同时压权重（泳道：少搬字节；第十二节）。</b>每个数从 16 bit 一路压到 4 bit，8 bit 的 FP8 权重让 Llama 3 70B 每字间隔最多降到 16 bit BF16 的 1/2。kernel（芯片上每次调起的一个计算函数）也越融越少。</li>
    <li><b>最后看叠起来有多大（第二、十四节）。</b>同一套 GB300（NVIDIA 整柜 GPU）跑 DeepSeek-V4-Pro，光改软件，同一交互档（同样的出字速度）每卡吞吐是首发日的 5.1 倍（含 MTP，见 2.2）。</li>
  </ul>
  <p>全课的账建在 DeepSeek-V3（2024 年 12 月）上：MoE，总参数 6,710 亿，每个 token 激活的参数约 370 亿。
    同等 GPT-3.5 水平的推理价格，2024 年 10 月已不到 2022 年 11 月的 1/280（斯坦福 AI Index 2025），系统这一侧能不能单独拆出来，14.3 再说。</p>
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
    "__FIG_S0_STORY__": ("fig-s0-story", "fig6-story.svg", "topic06-fig-story.py",
        '<b>每张卡是一步：写原文报告的提升和基线，没有倍数就写做法。</b><br>'
        '后一个的基线常是前一个，所以这些倍数不能连乘。'),
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

NUM_LOCK = []
