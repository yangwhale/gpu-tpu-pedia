# -*- coding: utf-8 -*-
"""专题六 · 第十四节「落地：把 Qwen3.5-397B 在 TPU 上跑起来」。

三块：TPU 跟 GPU 哪里不一样 → 这个模型落到 TPU 要特殊处理哪几处 → 怎么跑快。
主线是一台 v7x-8 上、vLLM 的 PyTorch TPU 后端（写作时尚未公开），
数字一律以起点那天 ＝ 1.00 指数化；Kimi K3 做两处对照。
只讲做法，不讲排障。
"""

BODY = '''
  <p class="lead">前面十三节的招数，这一节落到一个模型、一种芯片上。主线是 Qwen3.5-397B（FP8）在一台 v7x-8 上：4 颗芯片、8 个 TensorCore；60 层里 45 层线性注意力（GDN）、15 层全注意力，每层 512 个专家选 10 个。栈是 vLLM 的 PyTorch TPU 后端，写作时尚未公开，倍数以起点那天 ＝ 1。<b>三个问题：TPU 跟 GPU 哪里不一样，这个模型哪几处要特殊处理，怎么跑快。</b></p>
__FIG_S14_DIFF__

  <h3 id="s14-1">14.1　TPU 跟 GPU 哪里不一样</h3>
  <p><b>形状：GPU 录图只是加速，TPU 不编就不能跑。</b>GPU 上 kernel 逐个发，CUDA Graph 录下来省发射开销，档外的批照样跑（<a href="#s12-3">12.3</a>）。XLA 按静态形状整图编译（专题二<a href="topic-02-L300.html#s5">「谁做决定」</a>），一批的 token 总数补到预编译的几档；没编过的档，第一个撞上的请求要等 60–90 s，之后同形状不到 1 s。</p>
  <p>所以档位写进启动参数、开服前编好，SGLang-JAX 的写法是 <code>--precompile-bs-paddings</code> 加 <code>--precompile-token-paddings</code>；长短不一的请求由 RPA 用一个 kernel 吃下（<a href="#s5-1">5.1</a>）。按 tpu-inference 公开 README 的配置，冷启动约 7 分钟：加载权重、MoE 重新量化、初始化 KV 池。</p>
  <p>档位一多，编译就成了开服的大头。同一形状在每个 rank 上是同一张图，所以档位分给不同 rank 编，产物经图指纹校验，其余 rank 直接加载。一篇公开工程博客在 Hy4（TP16）上连同几项编译期小改动，冷编译 95.1 分钟 → 约 14 分钟；放宽到跨 DP 组回放，Qwen3.5 DP8 从权重加载完到可服务 741 s → 270 s。</p>
  <p><b>通信：没有全互连，XLA 也未必替你藏。</b>GPU 机内 8 卡 NVLink 全互连、机间走 IB，MoE 收发有 DeepEP 点对点（<a href="#s6-4">6.4</a>）。v7 芯片间是 ICI 直连的 3D 环面，一颗只跟 6 个邻居直连，擅长 all-gather、reduce-scatter 这类规整的集合通信。XLA 排程未必把通信藏到计算后面，想藏就得自己写 Pallas。</p>
  <p><b>片上：VMEM 和 SMEM 由 kernel 自己管。</b>每个 TensorCore 有 64 MiB VMEM 放正在算的块，1 MiB SMEM 放标量和索引；Pallas kernel 决定什么时候把什么搬进来。显存按 tile 存，最内两维都要补齐，FP8 的次内层补到 32 行（<a href="#s5-1">5.1</a>），窄维落在那里就补出一片空位。</p>
  <p><b>数据格式：v7 公开规格只列 BF16 和 FP8。</b>Blackwell 的张量核直接算 FP4；v7 上权重和 KV 存 FP8，更窄的格式要在向量单元上解包，代价见 14.3 的 K3。</p>

  <h3 id="s14-2">14.2　这个模型落到 TPU，哪几处要特殊处理</h3>
  <p><b>切法：注意力分请求，MoE 分专家。</b>全注意力层只有 2 个 KV 头，8 路 TP 切不开，每份 KV 要复制 4 份（<a href="#s6-1">6.1</a>）。所以两段各切各的：注意力用 DP，MoE 用 EP 切成 8 份，中间 all-gather 收齐 token、算完 reduce-scatter 发回（<a href="#s6-2">6.2</a>），正是 ICI 擅长的那类通信。prefill 用 DP8；decode 用 DP4×TP2，每个 rank 正好分到 1 个 KV 头。</p>
  <p><b>缓存：KV 和线性层状态共用一个池。</b>KV 随上下文变长，GDN 状态每条序列固定约 189 MB（FP32）。分两个池各自预留，并发和长度一变总有一个闲着；统一池里两者共用可互换的页、按需领取，前缀缓存也走这个池（<a href="#s13-2">13.2</a>）。</p>
  <p><b>GDN 状态：换视图，不搬数据。</b>GDN kernel 按 tile 算，要的状态布局跟池里存的不一样。每个 tile 都重排一遍最费；在引用上换视图（bitcast 加 reshape），只改元数据、不搬字节，指令包 366 → 87。</p>
  <p><b>切序列时，线性层按头切。</b>开 PCP 把一条 prompt 摊到 8 个 device 上，全注意力层按 token 切没问题；线性层的状态要从头接力扫，按 token 切每段都得从空状态起步。所以线性层改按头切：每个 rank 拿几个头，在完整序列上扫完再写回（<a href="#s7-2">7.2</a>）。</p>
  <p><b>KV：序列维放最内层。</b>KV 存 FP8；头只有 1–2 个，头维若落在最内两维上，就被补成整块，省下的字节又补了回去。让序列维放最内层，FP8 才真省字节（<a href="#s5-1">5.1</a>）。线性层状态则用 BF16 存、计算时升回 FP32。</p>
  <p><b>MoE：整层一个 Pallas 程序。</b>MoE 占 prefill 约八成时间，XLA 又不把 reduce-scatter 藏到矩阵乘后面。写成一个 Pallas 程序后，kernel 自己选专家、把每行推给持有它的 rank，通信藏到矩阵乘后面。路由表放 1 MiB 的 SMEM，token 一多装不下，就按窗口分批流进去。</p>
  <div class="note"><span class="t">K3 对照①</span>K3 的部署手册写明：69 层 KDA 的状态池，DP、EP、DCP 都不切它（DCP 只按 token 位置切 MLA 的 KV）。DEP8 下每个 rank 要存一整池 KDA 状态，装不下，所以 K3 用 TP（<a href="#s6-1">6.1</a>）。Qwen3.5 的统一池随 DP 切，DP=8 实测跑通。</div>

  <h3 id="s14-3">14.3　怎么跑快</h3>
  <p>砍的理由跟前面几节一样：<b>HBM 在空等，通信露在外面，布局把窄维补成整块。</b>最大的几刀，都是手写 Pallas 把调度权拿回来。</p>
  <p><b>prefill 四刀，全砍在 MoE。</b>先在片上按目的地累加；再把通信切成小块，想让它跟计算叠起来（XLA 实际仍没叠上）；接着整层融成一个程序；最后修剪 kernel 本身。</p>
__FIG_S13_LADDER__
  <p>整层融合是最大一级：改前 reduce-scatter 约占两成时间，切了块也没东西和它重叠，融合后才藏到矩阵乘后面。四级连乘约 1.58，同期实测约 1.59，只差半个百分点。拆得这么干净，靠每刀先合进代码、默认关、用环境变量打开，开关前后天然一组 A/B（<a href="#s10-2">10.2</a>）。</p>
  <p><b>decode 三刀，砍布局和小 batch</b>：KV 序列维放最内层（14.2）；小 batch 的 MoE 不排序，每个被选中的专家只读一次，daily 涨约一成二（<a href="#s5-2">5.2</a>）；线性层状态存 BF16，KV 池多装约一成二，同时在跑的请求 208 → 232，精度验收还没收尾。</p>
  <div class="note"><span class="t">K3 对照②</span>K3 只追 1 到 8 条，一步几乎全在读权重，按 config 推一个 token 读约 158 GB（本课推导，<a href="#s1-1">1.1</a>）。Inferact 把 92 个 MoE 层装进一次 Pallas 调用，跑在 16 颗 v7 上。挑三刀：
  <ul>
    <li><b>层循环搬进 kernel</b>：第 N 层还在算 MoE，第 N+1 层的权重已在路上，<a href="#s12-3">12.3</a> 那张融合图里 kernel 交接时的空等，就是这么捡回来的。</li>
    <li><b>VMEM 分时复用</b>：两种投影加 4 个专家槽各占一块要 90.67 MB，放不进 64 MiB；三者时间上不重叠，共用 31.29 MB 就够（本课按公开代码推导）。</li>
    <li><b>MXFP4 换 FP8，多读反而快</b>：MXFP4 每 32 个元素要在向量单元上解包缩放一次，FP8 每 256 个才一次；字节多 1.60 倍（本课推导），可 batch 1 带宽只用到五成上下，卡住的是向量单元（<a href="#s12-1">12.1</a>）。</li>
  </ul>
  另几刀：专家槽轮转；集合通信按 2×2×4 连线手写，换拓扑就得重写；兼做投机验证器。整个 megakernel 从零编译不到 90 s，14.1 那笔编译账便宜得多。</div>
__FIG_S13_MAP__
  <p><b>拆开不一定更快。</b>PCP8 收尾时吞吐是 DP8 的约 0.89 倍，省的是长 prompt 的首 token 延迟（本课推测，<a href="#s7-2">7.2</a>）；tpu-inference 上拆 PD，每请求要搬约 349 MB（KV 加线性层状态，<a href="#s8-3">8.3</a>），单并发 0.95 倍。一台装得下就先别拆（<a href="#s8-1">8.1</a>）。</p>
  <p>多副本时，前缀缓存只在本副本有效，KV 感知路由把同前缀的请求送到存着它的副本；SGLang、NVIDIA Dynamo、llm-d、GKE Inference Gateway 都做了。选栈：JetStream 已停止维护；tpu-inference 是 vLLM 的 JAX 后端，SGLang-JAX 是 SGLang 的 JAX 实现。先看模型在不在支持列表里，再看第十节的拓扑搭不搭得出来。</p>

  <p class="landing">⭐ 同一种芯片，刀落在哪取决于 batch：Qwen3.5 并发几百，刀落在通信和形状；K3 只有 1 到 8 条，刀落在 HBM 读写。<b>先看 batch 落在 <a href="#s1-1">1.1</a> 的哪一侧，再决定刀往哪砍。</b>接手一个服务的第一天怎么问，收成第十五节的八问。</p>
'''

FIGS = {
    "__FIG_S14_DIFF__": ("fig-s14-diff", "fig6-s14-diff.svg", "topic06-fig-climb.py",
        '<b>看右列：每条变了的规矩，都对应这个模型的一种写法。</b><br><em>GPU 一列以 Blackwell 和开源栈为参照；v7 一列依据公开规格与博客，各条出处见正文。</em>'),
    "__FIG_S13_LADDER__": ("fig-s13-ladder", "fig6-s13-ladder.svg", "topic06-fig-landing.py",
        '<b>看绿柱比左右两根高出多少：通信藏到矩阵乘后面，是一次就顶前两刀加起来的一级。</b><br><em>本课据私有实测指数化，起点那天 ＝ 1.00；只画 prefill，decode 中途换过口径，连不成一条梯子。</em>'),
    "__FIG_S13_MAP__": ("fig-s13-map", "fig6-s13-map.svg", "topic06-fig-landing.py",
        '<b>看实心点扎堆在哪一列：上表在通信和形状，下表在 HBM 读写。</b><br><em>定性归类，点的大小不代表收益；K3 一侧依据 Inferact 公开博客与代码。</em>'),
}

NUM_LOCK = [
]
