# -*- coding: utf-8 -*-
"""把八张图组装成一份独立文档 —— 视觉与结构对标 TPU 那份芯片架构页。

用法：python3 build_doc.py /path/to/out.html
图由各 fig_g*.py 现场生成后内联进 HTML（不引外部文件，单页自包含）。
"""
import io, os, sys, importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FIGS = ["fig_g1_chip", "fig_g2_sm", "fig_g3_hierarchy", "fig_g4_mma",
        "fig_g5_blockscale", "fig_g6_memory", "fig_g7_latency", "fig_g8_scale"]

CSS = io.open(os.path.join(HERE, "style.css"), encoding="utf-8").read()
FAVICON = ("<link rel=\"icon\" href=\"data:image/svg+xml,"
           "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
           "<text y='.9em' font-size='90'>🦀</text></svg>\">")


def svg(i):
    return importlib.import_module(FIGS[i - 1]).build()


def fig(i, cap):
    return (f'<figure class="figwide" id="g{i}">\n{svg(i)}\n'
            f'<figcaption>{cap}</figcaption>\n</figure>')


NAV = [("s0", "读之前"), ("s1", "1 全景"), ("s2", "2 一个 SM"), ("s3", "3 线程层级"),
       ("s4", "4 Tensor Core"), ("s5", "5 块量化"), ("s6", "6 内存层级"),
       ("s7", "7 延迟"), ("s8", "8 同尺度对照"), ("s9", "来源等级"),
       ("s10", "还没查实")]


def build():
    P = []
    a = P.append

    a(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU 显微镜 —— 一颗 B200 从封装拆到 warp</title>
{FAVICON}
{CSS}</head><body>

<header><div class="wrap">
<span class="eyebrow">🔬 对标 TPU 芯片架构图的标准重画一遍 GPU</span>
<h1>GPU 显微镜 —— 一颗 B200 从封装拆到 warp</h1>
<p class="lede">每个单元到底多大、彼此之间怎么连、谁在一个 SM 里、谁算一个 warp、
一个数从 HBM 走到乘加单元要经过几站 —— 全部拆开画出来，并在每一处跟 TPU v7 对照。
八张图，每一张都能单独拿去讲。</p>
<div class="meta">
<span><b>八张图</b> · 全部为可讲课密度</span>
<span><b>来源分三级</b>：官方 / 第三方 / 本文推导，图上逐处标注</span>
<span><b>TPU 侧只用公开数字</b>，未公开的如实留灰</span>
</div>
</div></header>

<nav><div class="wrap">''')
    a("".join(f'<a href="#{i}">{t}</a>' for i, t in NAV))
    a('</div></nav>\n<div class="wrap">')

    # ── §0 ──────────────────────────────────────────────────────────
    a('''
<section id="s0">
<h2><span class="n">0</span>读之前：这份文档怎么区分「查到的」和「算出来的」</h2>
<p class="sub">同一张图上会同时出现三种可信度的数字，混在一起就没法用来讲课了。</p>

<div class="grid g3">
<div class="stat b"><div class="k">官方</div><div class="v">直接引用</div>
<div class="d">NVIDIA 开发者博客、Blackwell 调优指南、PTX ISA 9.3、
Google Cloud TPU 文档。图上不加任何标记。</div></div>
<div class="stat y"><div class="k">第三方</div><div class="v">标灰</div>
<div class="d">独立评测与论文。图上一律用<b>灰色小字</b>并注明「第三方」——
它们大多是实测值，跟官方口径不一定完全一致。</div></div>
<div class="stat r"><div class="k">本文推导</div><div class="v">给出推导链</div>
<div class="d">官方没公布、但能从公开数字算出来的。图上标「推导值」，
并在正文里把每一步写清楚，让你能自己复核。</div></div>
</div>

<div class="note info"><span class="t">关于 TPU 那一侧</span>
这份文档里凡是 TPU 的数字，<b>只用公开来源</b>：Cloud TPU 官方文档、JAX 生态的公开工具、
以及 IEEE Micro 2021 那篇 TPUv2/v3 设计论文。片上暂存的容量与带宽官方没有公开，
所以图上就是灰色虚线框加「官方未公开」，<b>不猜、不填</b>。
少几个数字不影响这份文档要讲的事 —— 结构和搬运方式才是重点。</div>

<div class="note warn"><span class="t">一处必须提前说明的口径</span>
NVIDIA 的 B200 是<b>两个 die 对软件呈现为一个 GPU</b>；TPU v7 是
<b>两个 chiplet 对软件呈现为两个独立 device</b>。同样的封装形态，软件视图完全相反。
所以下文凡是拿两者比总量的地方，都以<b>一个完整封装</b>为单位 ——
一颗 B200 对一颗 v7 chip（＝ 2 个 device）。这一点不统一，后面所有数字都会差两倍。</div>
</section>''')

    # ── §1 ──────────────────────────────────────────────────────────
    a('''
<section id="s1">
<h2><span class="n">1</span>先看全景：一颗 B200 里有些什么</h2>
<p class="sub">封装 → die → SM 阵列 → L2 → HBM → 对外互联，一层层往里剥。</p>''')
    a(fig(1, "<b>图 G-1　一颗 B200 全景。</b>每个小格是一个 SM，红色虚线的 6 个是出厂禁用的 —— "
             "148 ＝ (80 − 6) × 2 就是这么来的，禁用是为了提良率。"
             "中间那条 NV-HBI 是两个 die 之间的一致性总线，10 TB/s；"
             "正因为有它，两个 die 才能对软件装成一个 GPU。"
             "GPC 的分组存疑（官方 Blackwell Ultra 说 8 个，第三方报 B200 是 10 个），所以图上不画 GPC 边界。"))
    a('''
<p>这张图上最值得停一下的是<b>「两个 die 对软件是一个 GPU」</b>这件事。它不是免费的：
L2 被切成两个分区，本分区实测约 21 TB/s，跨到对面 die 掉到 16.8 TB/s，延迟也变高。
也就是说「一个 GPU」这个抽象底下藏着一条你看不见、但会影响性能的缝。</p>
<p>TPU v7 在同一个问题上做了<b>相反</b>的选择：同样是双 chiplet，它直接暴露成两个独立 device，
各有各的地址空间。缝还是那条缝，区别只在于<b>由谁来面对它</b> ——
NVIDIA 让硬件扛下来（代价是跨 die 访问悄悄变慢），Google 让软件扛（代价是你必须自己切分）。</p>
</section>''')

    # ── §2 ──────────────────────────────────────────────────────────
    a('''
<section id="s2">
<h2><span class="n">2</span>把一个 SM 拆开 —— 这张是全篇的核心</h2>
<p class="sub">四个处理块、128 个 CUDA Core、4 个 Tensor Core、64 个 warp 槽，一个不落地画出来。</p>''')
    a(fig(2, "<b>图 G-2　一个 SM 的显微镜展开。</b>一个 SM 分成四个对称的处理块（sub-core），"
             "每块自带 L0 指令缓存、一个 warp 调度器、16 个 warp 槽、64 KiB 寄存器、"
             "32 个 CUDA Core 和 1 个 Tensor Core。TMEM 也按 warp 号切成四份 —— "
             "warp k 只能碰 lane 32k 到 32k+31，这是 PTX 手册明写的硬约束。"
             "LD/ST 与 SFU 的个数用灰色标出：Blackwell 官方框图没有标数，图上沿用 Hopper 的画法。"))
    a('''
<h3>那个「1,024 乘加 / 周期」是怎么算出来的</h3>
<p>NVIDIA 从不公布「一个 Tensor Core 每周期做多少次乘加」。但这个数可以从
<b>峰值 ÷ SM 数 ÷ 时钟</b> 反推出来，而且能用两个已知世代来验证这套算法对不对：</p>
<div class="tw"><table>
<thead><tr><th>芯片</th><th class="n">峰值 FP16（稠密）</th><th class="n">SM</th>
<th class="n">时钟</th><th class="n">FLOP / 周期 / SM</th><th class="n">最近的 2ⁿ</th><th class="n">偏差</th></tr></thead>
<tbody>
<tr class="hit"><td>A100 SXM</td><td class="n">312 TF</td><td class="n">108</td><td class="n">1.410 GHz</td>
<td class="n">2,048.9</td><td class="n">2,048</td><td class="n">+0.04%</td></tr>
<tr class="hit"><td>H100 SXM</td><td class="n">989.4 TF</td><td class="n">132</td><td class="n">1.830 GHz</td>
<td class="n">4,095.9</td><td class="n">4,096</td><td class="n">−0.00%</td></tr>
<tr><td><b>B200</b></td><td class="n">2,250 TF</td><td class="n">148</td><td class="n">1.830 GHz</td>
<td class="n"><b>8,307.5</b></td><td class="n"><b>8,192</b></td><td class="n">+1.41%</td></tr>
</tbody></table></div>
<p>前两代几乎分毫不差地落在 2 的整数次幂上 —— 这说明这套算法本身是对的，
不是凑出来的。于是 Blackwell 那一行的 <b>8,192 FLOP / 周期 / SM</b> 就可以放心用：</p>
<pre><code>8,192 FLOP/周期/SM  ÷  4 个 Tensor Core  ÷  2（一次乘加算 2 个 FLOP）
                    =  1,024 次乘加 / 周期 / Tensor Core

对照：A100 是 256，H100 是 512 —— 每代翻倍，很干净。</code></pre>
<p>B200 那 <b>1.41%</b> 的缺口是诚实留着的：它来自时钟。1.83 GHz 是第三方给的 boost 频率，
NVIDIA 没有公布确切值。反过来说，如果每 SM 确实是 8,192，那么真实时钟应该在 1.856 GHz 左右。
<b>这个缺口没法用现有公开数字消掉，所以就让它留在那儿</b>，不去凑一个好看的数。</p>
</section>''')

    # ── §3 ──────────────────────────────────────────────────────────
    a('''
<section id="s3">
<h2><span class="n">3</span>谁是一个 warp、谁在一个 SM 里 —— 每层抽象钉在哪块硅上</h2>
<p class="sub">CUDA 那六层不是纯软件约定，每一层都精确对应一条硬件边界。</p>''')
    a(fig(3, "<b>图 G-3　线程层级 ↔ 硬件归属。</b>左边是软件写出来的抽象层，"
             "中间是它被钉死在哪一级硬件上，右边是这一层内部能共享什么、靠什么同步。"
             "最下面那张表把同样的问题问一遍 TPU —— 会发现 TPU 少掉的那几层，"
             "恰好就是 GPU 用来藏延迟的那几层。"))
    a('''
<p>这里有个容易被忽略的因果：<b>「线程块整块钉死在一个 SM 上、落下就不迁走」</b>
不是实现上的偷懒，而是共享内存这个抽象能成立的<b>前提</b>。
共享内存是 SM 内部的物理 SRAM，块要是能迁走，这块内存的语义就没法定义了。</p>
<p>再往上，Hopper 引入的 cluster 之所以出现，是因为 148 个 SM 之间除了 L2 之外
再没有别的快捷通道 —— cluster 相当于在「一个 SM」和「整颗 GPU」之间硬插了一层，
让同一个 GPC 内的几个 SM 能互相读写共享内存。
<b>TPU 完全不需要这一层</b>：一个 chip 只有 2 个 TensorCore，本来就不存在「一组核如何协同」的问题。
核少反而省掉一整层抽象，这是规模带来的差别，不是设计水平的差别。</p>
</section>''')

    # ── §4 ──────────────────────────────────────────────────────────
    a('''
<section id="s4">
<h2><span class="n">4</span>Tensor Core 一次能吃多大一块矩阵</h2>
<p class="sub">四代演进＋一张真实比例的叠图。专治「GPU 是不是也要喂 128×128」这个误解。</p>''')
    a(fig(4, "<b>图 G-4　四代 MMA 指令与真实比例叠图。</b>上半部分是「一次 MMA 动员多少个线程」"
             "从 Volta 的 8 个一路涨到 Blackwell 的 256 个；"
             "下半部分把三条指令的输出矩阵按<b>同一个比例</b>叠在 TPU 一个 MXU 的 256×256 上 —— "
             "每个元素 1.17 像素，没有任何缩放作弊。左上角那个圈住的小蓝点就是 "
             "<code>mma.sync</code> 的 16×8，只有 19 × 9 像素。"))
    a('''
<p>四代过去，变的是「一次动员多少线程」，<b>不变的是收缩维 K —— 一直是 16</b>
（fp16/bf16 是 K=16，fp8 是 K=32，fp4 是 K=64，换算成位宽全是同样的 16 个 32-bit 字）。
而 TPU 的 MXU 收缩边是 256，差 16 倍。</p>
<div class="note ok"><span class="t">这就是为什么 <code>head_dim=128</code> 打 TPU 却不打 GPU</span>
128 撞上 TPU 的 256 收缩边，只能喂满一半；而 128 是 16 的整数倍，对 GPU 来说喂得满满当当。
<b>同一个模型配置，在两边的「浪费」根本不在同一个位置。</b>
这也是调优经验不能直接跨平台搬的根本原因之一。</div>
</section>''')

    # ── §5 ──────────────────────────────────────────────────────────
    a('''
<section id="s5">
<h2><span class="n">5</span>块量化：GPU 把「多细的一撮数共享一个缩放因子」做进了硬件</h2>
<p class="sub">这一节的结论跟大多数人的直觉相反，值得单独拿出来讲。</p>''')
    a(fig(5, "<b>图 G-5　三种缩放粒度，以及它们各自挂在哪条指令上。</b>"
             "同样 64 个数：整个张量共享一个缩放因子（最粗）、每 32 个一组（MX 标准）、"
             "每 16 个一组（NVFP4，粒度是 MX 的两倍细）。"
             "右下角灰色虚线框是本文<b>未查实</b>的部分，如实标出来。"))
    a('''
<p>常见的说法是「GPU 在往 TPU 靠：也开始搞大矩阵单元了」。<b>这个说法只对了一半。</b>
Blackwell 确实加了一条更粗的快车道（<code>tcgen05.mma</code>，两个 SM 配对做同一次 MMA），
但它<b>没有把细的那条路砍掉</b> —— 恰恰相反，最细的那条 warp 级
<code>mma.sync.aligned.m16n8k32.block_scale</code> 是从 Ampere 留下来的老指令，
Blackwell 不但没砍，还专门给它加了块缩放能力。</p>
<p>所以正确的说法是：<b>GPU 现在粗细两条路并存，TPU 只有粗的那一条。</b>
细粒度不是历史包袱，它换来的是表达力 —— 「每 16 个元素配一个缩放因子」这种事，
需要计算单元本身就认得「16 个元素」这个粒度。收缩边是 256 的阵列，天然做不了这件事。</p>
<div class="note q"><span class="t">这里没有下结论的地方</span>
TPU v7 有没有硬件级的分块缩放通路，<b>本文没有查实</b>。
公开的 JAX 规格表只列了 bf16 和 fp8 两档峰值，没有任何 MX / NVFP4 类格式的条目，
也没有说明 MXU 内部是否有这条通路。<b>能确定的只有一条：MXU 的收缩边是 256，比 GPU 的 K=16 粗 16 倍。</b></div>
</section>''')

    # ── §6 ──────────────────────────────────────────────────────────
    a('''
<section id="s6">
<h2><span class="n">6</span>一个数走完全程：从 HBM 到乘加单元，中间几站</h2>
<p class="sub">这一节刻意不比带宽大小 —— 比的是「每一站由谁负责搬」。</p>''')
    a(fig(6, "<b>图 G-6　两条链并排。</b>蓝色的站是<b>硬件自动管</b>（有 tag、会 miss），"
             "紫色的站是<b>软件／编译器显式管</b>（不会 miss，也没有兜底）。"
             "GPU 五站，中间两站是缓存；TPU 四站，中间那站是暂存。"
             "TPU 的片上容量与带宽官方未公开，所以那两格是灰色虚线框 —— 不填数字。"))
    a('''
<p>为什么不画一张带宽柱状图？因为那样会变成一半实数、一半灰框，看着像对比，其实什么也没比。
更重要的是，<b>这两条链真正的差别不在快慢，在「谁负责知道数在哪」</b>：</p>
<div class="grid g2">
<div class="card"><h4>GPU 的 L1 / L2 是<b>缓存</b></h4>
<p style="margin:0">你只管发访存指令，命中不命中它自己处理。代价是要存 tag、要做替换、
要维护一致性 —— 相当一部分硅面积和功耗花在「猜你接下来要什么」上，而且<b>时间不可预测</b>。</p></div>
<div class="card"><h4>TPU 的 VMEM 是<b>暂存</b></h4>
<p style="margin:0">编译器显式发 DMA 把数据搬进来。没有 tag、没有 miss、时间可预测 ——
代价是编译器算错了就是真的慢，<b>没有兜底</b>。</p></div>
</div>
<div class="note info"><span class="t">TMEM 是个值得注意的信号</span>
Blackwell 新加的 TMEM，是一块<b>只给 Tensor Core 用、由指令显式搬进搬出、不参与缓存机制</b>的
片上 SRAM —— 这个描述几乎就是 TPU 的 VMEM。
在矩阵这条路上，「让硬件猜」的收益越来越小，不如把控制权交回给编译器和 kernel 作者。
<b>方向是清楚的。</b></div>
</section>''')

    # ── §7 ──────────────────────────────────────────────────────────
    a('''
<section id="s7">
<h2><span class="n">7</span>延迟怎么被藏起来 —— 一边靠换人，一边靠排班</h2>
<p class="sub">取一次数要几百个周期，两边都躲不掉。区别在于用什么盖住它。</p>''')
    a(fig(7, "<b>图 G-7　两种延迟隐藏机制。</b>左边每一行是一个 warp，"
             "调度器每一拍只能发一条指令；六个 warp 全在等数的那两拍就是<b>气泡</b>。"
             "右边四条泳道是四个<b>不同的物理单元</b>，本来就能同时动 —— 一个气泡都没有。"
             "下方的 VLIW 槽位构成出自 IEEE Micro 2021 那篇 TPUv2/v3 设计论文，"
             "<b>是 v2/v3 的公开数字，v7 官方没有公布</b>。"))
    a('''
<p>把两边的机制说透，只要一句话：<b>GPU 是时间复用（很多 warp 轮流用同一个单元），
TPU 是空间并行（不同单元同时开工）</b>。这一个选择，几乎决定了两边硬件长什么样。</p>
<p>GPU 这边，因为有 cache，延迟就不是常数；不是常数，编译期就排不了班。
于是只能准备一大批随时能顶上的替补 —— 这直接解释了 SM 为什么长成图 G-2 那个样子：
<b>256 KiB 的寄存器堆</b>（装所有人的现场）、<b>64 个 warp 槽</b>（装人）、
<b>4 个独立调度器</b>（挑人）。这些硅一次乘法都不算，它们的全部工作就是盖住等待。</p>
<p>TPU 那边，没有 cache，延迟就是常数：标量个位数周期、向量几十、矩阵几百，
全都是编译期已知的。既然是常数，编译器就能把 WAIT 放在正好那一拍。
省下的替补、寄存器堆和调度器，面积<b>全给了乘加阵列</b> —— 这一点在下一节会直接看到后果。</p>
<div class="note warn"><span class="t">这套机制的代价，两边都不小</span>
GPU 这边：你在 kernel 里多用一个寄存器，就少一个能替它顶班的 warp。
循环展开、把中间结果留在寄存器里 —— 这些让单个 warp 变快的手段，同时在削弱藏延迟的能力。
<b>这就是 CUDA 调优里那条最基本的取舍。</b><br>
TPU 这边：WAIT 放早了纯空转，放晚了 MXU 干等，而且<b>没有别的 warp 能顶上来</b>。
编译器必须算准 —— 算不准就是真的慢。</div>
</section>''')

    # ── §8 ──────────────────────────────────────────────────────────
    a('''
<section id="s8">
<h2><span class="n">8</span>压轴：把两颗芯片放在同一把尺子上</h2>
<p class="sub">前面七张图的结论，最后收在这一张上。</p>''')
    a(fig(8, "<b>图 G-8　等面积对照。</b>两个方块的<b>面积</b>按每周期乘加单元总数等比画，"
             "内部按<b>真实单元数</b>切分。左边 592 个小格，右边 4 个大格 —— "
             "总量只差 1.16 倍，切分粒度差 128 倍。"))
    a('''
<p>这张图是全文的落点。左边一颗 B200 有 592 个 Tensor Core，每个每周期 1,024 次乘加；
右边一颗 TPU v7 chip 只有 4 个 MXU，每个每周期 131,072 次。
<b>总量几乎一样（1.16×），单元粒度差整整 128 倍。</b></p>
<p>更有意思的是它能自我验证：两边<b>公布的峰值</b>只差 3%（2,250 对 2,307 TFLOPS），
而按上面这套口径算出来的乘加总数差 16% —— 剩下的差额只能来自时钟。
两条互相独立的账能对上量级，说明「每 cell 双发」这个推导没有跑偏。</p>
<div class="note bad"><span class="t">别把这张图读成排名</span>
总量接近、峰值接近，说明两家在同一代工艺上做出的<b>算力密度是可比的</b>。
真正的分歧在前面几张图里：<b>谁来知道数在哪</b>（§6）、<b>谁来盖住延迟</b>（§7）、
<b>切多细</b>（本节）。这三个选择互相咬合 —— 换掉任何一个，另外两个都得跟着换。
细粒度买来的是「什么都能跑」，粗粒度买来的是「几乎没有控制开销」，
<b>各自的代价都写在图上了</b>。</div>
</section>''')

    # ── §9 来源等级 ─────────────────────────────────────────────────
    a('''
<section id="s9">
<h2><span class="n">✓</span>来源等级：每个数字是从哪来的</h2>
<p class="sub">照着这张表就能自己复核，不必相信本文。</p>
<div class="tw"><table>
<thead><tr><th>等级</th><th>用在哪些数字上</th><th>具体来源</th></tr></thead>
<tbody>
<tr><td><span class="pill b">官方</span></td>
<td>SM 内部构成、TMEM 尺寸与 lane 约束、PTX 指令与限定符、公布峰值、HBM 容量与带宽、
NVLink、warp / 寄存器 / 共享内存的各项上限</td>
<td>NVIDIA 开发者博客（Blackwell 架构）、Blackwell 调优指南、
PTX ISA 9.3、Google Cloud TPU 官方文档</td></tr>
<tr><td><span class="pill">第三方</span></td>
<td>L2 分区带宽与延迟、L1 命中周期、每 die 启用 SM 数、boost 时钟、
处理块（sub-core）的四分结构、TPU VLIW 槽位构成</td>
<td>chipsandcheese 实测、TechPowerUp 规格库、arXiv 2507.10789、
IEEE Micro 2021《The Design Process for Google's Training Chips: TPUv2 and TPUv3》</td></tr>
<tr class="hit"><td><span class="pill g">本文推导</span></td>
<td>每个 Tensor Core 1,024 乘加 / 周期；每个 MXU cell 每周期做 2 次 bf16 乘加</td>
<td>本文 §2 与 §8，推导链已完整写出，并用 A100 / H100 两代交叉验证</td></tr>
</tbody></table></div>
</section>''')

    # ── §10 还没查实 ────────────────────────────────────────────────
    a('''
<section id="s10">
<h2><span class="n">?</span>还没查实的 —— 这份清单本身也是内容</h2>
<p class="sub">写清楚哪里不知道，比把每一格都填满有用。</p>
<div class="tw"><table>
<thead><tr><th>问题</th><th>目前的状态</th><th>影响到哪张图</th></tr></thead>
<tbody>
<tr><td>B200 到底几个 GPC</td>
<td>官方 Blackwell Ultra 页面说 8 个 GPC / 160 SM；第三方报 B200 是 10 个。<b>存疑</b></td>
<td>G-1（因此图上不画 GPC 边界）</td></tr>
<tr><td>每个处理块的 LD/ST 与 SFU 各几个</td>
<td>沿用 Hopper 框图的 ×8 / ×4，<b>Blackwell 官方图没有标数</b></td><td>G-2（已标灰）</td></tr>
<tr><td>Blackwell 共享内存的 bank 数</td>
<td>历史上一直是 32 bank × 4 B，未见 Blackwell 明确重述，<b>本文不复述</b></td><td>G-2 / G-6</td></tr>
<tr><td>B200 的确切 boost 时钟</td>
<td>1.83 GHz 出自第三方；用它反推每 SM 有 <b>1.41% 缺口</b>，官方未公布确切值</td>
<td>G-2 的推导表（缺口已如实留着）</td></tr>
<tr><td>TPU v7 有没有硬件块缩放通路</td>
<td>公开 JAX 规格表只有 bf16 / fp8 两档峰值，<b>没有任何 MX / NVFP4 条目</b>。查到之前不下结论</td>
<td>G-5（灰色虚线框）</td></tr>
<tr><td>TPU v7 的 VMEM 容量与带宽</td>
<td><b>官方未公开</b></td><td>G-6（灰色虚线框，不填数字）</td></tr>
<tr><td>TPU v7 的 VLIW 槽位构成</td>
<td>本文用的是 v2/v3 论文的公开数字（2 标量 + 4 向量 + 2 矩阵 + 1 杂项 + 6 立即数），
<b>v7 官方没有公布</b>，不往 v7 上套</td><td>G-7（图上已标代次）</td></tr>
<tr><td>TPU v7 的时钟</td>
<td><b>官方未公开。</b>本文不给数字 —— 只说峰值与乘加总数两笔账能对上量级</td><td>G-8</td></tr>
</tbody></table></div>

<div class="note info"><span class="t">下一步</span>
这八张图画完就是为了移植进教学课件的专题二。移植时图不用改 ——
本文从一开始就是按<b>公开可讲</b>的标准画的，没有任何需要脱敏的数字。</div>
</section>

<footer>八张图均由脚本生成（<code>Courses/tools/gpu-micro/</code>），改数据重跑即可，不用手改 SVG。
每张图都经过渲染 → 截图 → 目视核对的循环，确认无遮挡、无溢出。</footer>
</div></body></html>''')
    return "\n".join(P)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gpu-micro.html"
    html = build()
    io.open(out, "w", encoding="utf-8").write(html)
    print("ok", out, len(html), "chars")
