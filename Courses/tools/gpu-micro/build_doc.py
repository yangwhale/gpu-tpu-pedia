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
<span><b>TPU 侧只用公开数字</b>，查不到的如实留灰</span>
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
             "GPC 分组图上<b>不画</b>：官方 Blackwell Ultra 页面给 8 个 GPC，第三方对 B200 报 10 个，"
             "而 8 × 20 和 10 × 16 都等于 160 个物理 SM —— 算术两边都成立，定不下来的事就不画。"))
    a('''
<p>这张图上最值得停一下的是<b>「两个 die 对软件是一个 GPU」</b>这件事。它不是免费的：
L2 被切成四个分区（每个 die 两个，Hopper 的两倍），本分区实测约 21 TB/s，
跨到对面 die 掉到 16.8 TB/s，延迟也变高。
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
<h3>那个「1,024 乘加 / 周期」是怎么来的</h3>
<p>先说清楚出处，因为这里有个容易走弯路的地方：<b>这个数并不需要推导，官方白皮书里就有。</b>
A100 那份架构白皮书写得很直白 ——「Volta 和 Turing 每 SM 有八个 Tensor Core，每个每周期做 64 次
FP16/FP32 融合乘加；A100 的第三代 Tensor Core 每个做 <b>256 次</b>，每 SM 四个，
合起来 <b>1,024 次稠密 FP16/FP32 乘加 / 周期</b>」。Hopper 官方博客接着说自己
「clock-for-clock 每 SM 的稠密矩阵吞吐是 A100 的两倍」，于是 H100 是 512。</p>
<p>那还推导什么？——<b>推导是用来验证口径的。</b>把「峰值 ÷ SM 数 ÷ 时钟」这套算法
拿到两代<b>已经有官方答案</b>的芯片上跑一遍，如果能还原出官方数字，就说明这套算法
没有偷偷混进稀疏、混进别的精度、或者用错了时钟；那么把它用在<b>官方没有明说的
B200</b> 上，才是可信的：</p>
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
<p>A100 那行算出 2,048.9 FLOP/周期/SM ＝ <b>1,024.4 次乘加</b>，官方原文是 1,024，
误差 0.04%；H100 那行算出 2,048 次乘加，正好是官方说的「A100 的两倍」。
<b>两代都还原成功，口径没跑偏</b>，于是 Blackwell 那一行的
<b>8,192 FLOP / 周期 / SM</b> 才可以放心用：</p>
<pre><code>8,192 FLOP/周期/SM  ÷  4 个 Tensor Core  ÷  2（一次乘加算 2 个 FLOP）
                    =  1,024 次乘加 / 周期 / Tensor Core

对照：Volta/Turing 是 64（每 SM 八个），A100 是 256，H100 是 512 —— 每代翻倍，很干净。</code></pre>
<p>这个 1,024 有第三方旁证：康奈尔的 GPU 教程写「一个 B200 的 Tensor Core 每周期最多做
1,024 次半精度 FMA，所以一个 SM 里的四个合起来做 4,096 次」—— 和上面推出来的
8,192 FLOP（＝4,096 次乘加）完全一致。</p>
<p>B200 那 <b>1.41%</b> 的缺口是诚实留着的：它来自时钟。1.83 GHz 是第三方给的 boost 频率，
NVIDIA 没有公布确切值。反过来说，如果每 SM 确实是 8,192，那么真实时钟应该在 1.856 GHz 左右。
而且第三方之间本身就不一致 —— 另一家的 B200 实测把 L1 命中报成「19.6 ns ＝ 39 周期」，
反推出来约 1.99 GHz。<b>这个缺口没法用现有公开数字消掉，所以就让它留在那儿</b>，
不去凑一个好看的数。注意它比前两代 0.04% / 0.00% 的余量大了一到两个数量级，
性质不同：前两代是「算法对上了」，这一代是「时钟不确定」。</p>
</section>''')

    # ── §3 ──────────────────────────────────────────────────────────
    a('''
<section id="s3">
<h2><span class="n">3</span>谁是一个 warp、谁在一个 SM 里 —— 每层抽象钉在哪块硅上</h2>
<p class="sub">CUDA 那六层不是纯软件约定，每一层都精确对应一条硬件边界。</p>''')
    a(fig(3, "<b>图 G-3　线程层级 ↔ 硬件归属。</b>看这张图请<b>竖着看中间那列</b>："
             "六层抽象里没有一层是纯软件约定，每一层都有一条对应的硬件边界。"
             "最下面那张 TPU 对照表里有三行写着「没有对应物」——"
             "<b>缺的那三层恰好都是 GPU 用来藏延迟的</b>，这不是巧合，第 7 节会回到这件事。"))
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
<p class="sub">五代演进＋一张真实比例的叠图。专治「GPU 是不是也要喂 128×128」这个误解。</p>''')
    a(fig(4, "<b>图 G-4　五代 MMA 指令与真实比例叠图。</b>上半部分是「一次 MMA 动员多少个线程」"
             "从 Volta 的 8 个一路涨到 Blackwell 的 256 个，"
             "每一行的目标架构（sm_70 / sm_75 / sm_80 / sm_90a / sm_100a）都摘自 PTX ISA 的 Target ISA Notes；"
             "下半部分把三条指令的输出矩阵按<b>同一个比例</b>叠在 TPU 一个 MXU 的 256×256 上 —— "
             "每个元素 1.17 像素，没有任何缩放作弊。三块<b>共用左上角，是嵌套不是并排</b>；"
             "左上角圈住的小蓝点就是 <code>mma.sync</code> 的 16×8，宽 9 × 高 19 像素。"))
    a('''
<p>五代过去，变的是「一次动员多少线程」：8 → 32 → 32 → 128 → 256。
<b>而收缩维 K 在 Ampere 定到 16 之后就再没动过</b>
（fp16/bf16 是 K=16，fp8 是 K=32，fp4 是 K=64，换算成位宽全是同样的 256 bit ＝ 8 个 32-bit 字）。
而 TPU 的 MXU 收缩边是 256，一条指令的粒度差 16 倍。</p>
<p><b>但这 16 倍不能读成「GPU 只能算 16 深」。</b>GPU 连发多条 K=16，累加器一直待在 TMEM 里不落地，
深度照样累得上去 —— 差的是一条指令吃进去的<b>粒度</b>，不是能力上限。
粒度粗的一方省下的是取指、译码和控制开销，粒度细的一方买到的是「什么形状都喂得满」。</p>
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
             "两张指令卡右上角的 <code>sm_100a</code> / <code>sm_120a</code> 是要点 —— "
             "<b>B200 是 sm_100a，用不了 warp 级那条</b>。"))
    a('''
<p>常见的说法是「GPU 在往 TPU 靠：也开始搞大矩阵单元了」。这个说法确实只对了一半，
但<b>不对的地方跟大多数人想的不一样</b> —— 包括本文第一版在内。</p>
<p>Blackwell 这一代有两条块缩放通路，PTX 手册把它们的目标架构写得清清楚楚：</p>
<div class="tw"><table>
<thead><tr><th>通路</th><th>动员多少线程</th><th>要求的目标架构</th><th>哪些卡</th></tr></thead>
<tbody>
<tr><td><code>tcgen05.mma … block_scale</code></td><td>两个 SM 配对</td>
<td class="n"><b>sm_100a</b></td><td>B200 / GB200（数据中心）</td></tr>
<tr><td><code>mma.sync … block_scale</code></td><td>一个 warp（32 线程）</td>
<td class="n"><b>sm_120a</b></td><td>RTX 50 / RTX PRO（消费级）</td></tr>
</tbody></table></div>
<p><b>B200 是 sm_100a。</b>也就是说，那条最细的 warp 级块缩放指令，
<r>B200 根本用不了</r> —— 它只存在于消费级那颗 die 上。数据中心这颗
能做的 fp8/fp6/fp4 矩阵乘依然有（<code>.kind::f8f6f4</code>），
但<b>带块缩放的只有 tcgen05 那条粗路</b>。</p>
<div class="note warn"><span class="t">所以正确的说法是</span>
不是「GPU 粗细两条路并存」，而是 <b>NVIDIA 把粗细两条路拆到了两颗不同的 die 上</b>：
消费级那颗留了细粒度的 warp 级通路，数据中心这颗只留了粗的。
换句话说，<b>在块量化这件事上，B200 反而比消费级 Blackwell 更像 TPU</b> ——
两者都只有「一大块一起算」这一条路。</div>
<p>为什么会这样分？细粒度换来的是表达力 —— 「每 16 个元素配一个缩放因子」这种事，
需要计算单元本身就认得「16 个元素」这个粒度，而这套控制通路是要占硅面积的。
消费级卡跑的是形状零碎的推理负载，值得为它掏这个面积；数据中心卡跑的是大矩阵，
把面积让给乘加阵列更划算。<b>同一代架构、同一个功能，按负载形状做了取舍</b> ——
这比「GPU 什么都要」更接近真相。</p>
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
    a(fig(6, "<b>图 G-6　两条链并排。</b>看这张图只需要数一件事：<b>一条链上有几个蓝色的站</b>。"
             "蓝＝硬件自动管（有 tag、会 miss），紫＝软件／编译器显式管（不会 miss，也没有兜底）。"
             "GPU 那条链上「L1＋共享内存」那一格是<b>左右分色</b>的 —— 同一块 SRAM，"
             "一半当缓存、一半当暂存，这一格本身就是本节结论的缩影。"
             "TPU 的 VMEM 容量取自 JAX 开源代码（64 MiB / core），<b>带宽官方没公开</b>，"
             "所以图上只给容量不给带宽；向量寄存器的数量同样未公开，那一格是灰虚框。"))
    a('''
<p>为什么不画一张带宽柱状图？<b>因为那张图会撒谎。</b>GPU 侧的每一站都有第三方实测数字，
TPU 侧只有 HBM 那一站是公开的 —— 画出来就是一半实柱、一半灰框，看着像对比，
其实只是在比「谁的资料公开得多」。缓存 vs 暂存的区别见图上那三张卡片，这里不重复。</p>
<p>倒是有一件图上塞不下、但值得单独说的事：<b>这两种设计对「程序员该操心什么」的要求完全不同。</b>
写 CUDA kernel，你操心的是访问模式（合并访存、bank conflict、复用距离）——
数在哪由硬件负责，你只负责让它猜得准。写 TPU kernel（或者说，让 XLA 替你写），
操心的是<b>切分</b>：这块权重能不能整块放进 VMEM、DMA 要提前几步发。
<b>同一个性能问题，在两边甚至不属于同一个知识门类。</b></p>
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
TPU 是空间并行（不同单元同时开工）</b>。这一个选择，几乎决定了两边硬件长什么样 ——
图上那三张卡片把这条因果链列全了，这里只补一件卡片上说不下的事。</p>
<p><b>「藏延迟」这件事在两边的账本上，记在完全不同的科目下。</b>
GPU 那 256 KiB 寄存器堆、64 个 warp 槽、4 个调度器，是<b>硬件成本</b> ——
它们一次乘法都不算，芯片流片那天就已经付掉了，你用不用得上都在那里。
TPU 把同样的活交给编译器，那是<b>编译期成本</b> —— 芯片上一分硅都不占，
但每换一个模型形状就要重新付一次，而且付不起的时候没有退路（编译器排错了就是空转）。</p>
<p>所以「哪种更好」这个问题问错了。真正的问题是：<b>你的工作负载形状变不变。</b>
形状天天变、什么都要跑，那笔硬件成本就摊得开；形状固定、就跑那几个大矩阵，
那笔硬件成本就是纯浪费 —— 而省下来的面积，下一节会看到它去了哪里。</p>
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
             "内部按单元数切分。<b>先比外框，再比里面的格子</b> —— 外框几乎一样大（1.16×），"
             "格子差 128 倍，这个反差就是全文的落点。右边 MXU 里的网点只表示「密」，"
             "真按 256×256 画每个 cell 只有 0.27 像素。"
             "右下角绿框是右边这一列的官方出处 —— 这一格来回改了三版，返工过程写在正文里。"))
    a('''
<p>这张图是全文的落点。左边一颗 B200 有 592 个 Tensor Core，每个每周期 1,024 次乘加，
合计 606,208；右边一颗 TPU v7 chip 是 4 个 MXU，每个 131,072，合计 524,288。
<b>总量几乎一样（1.16×），而单元粒度差 128 倍。</b></p>

<h3>这一格来回改了三版 —— 过程比结论值钱</h3>
<p>这是全文最该讲给学生听的一段，因为它示范了一种<b>最难自己发现的错误</b>：
答案对了，前提是错的。</p>
<pre><code>v1   写「4 个 MXU、每 cell 每周期 2 次乘加」
     → 结论对，但当时手上没有任何证据，纯粹是为了把 2,307 凑平而发明的

v2   拿 v5e / v6e 当验尸台，判定「每 cell 只有 1 次乘加」，改成 8 个 MXU
     → 反而错了

v3   官方数字出现，回到 4 个 MXU、每 cell 2 次乘加</code></pre>
<p><b>v2 是怎么错的</b>，值得一步步看。当时的算式是这样：</p>
<pre><code>v5e   4 MXU × 128×128 = 65,536 cell   × 1 MAC × 2 FLOP @ 1.50 GHz  →  196.6 TF   官方 197  ✓
v6e   4 MXU × 256×256 = 262,144 cell  × 1 MAC × 2 FLOP @ 1.75 GHz  →  917.5 TF   官方 918  ✓</code></pre>
<p>两行都对得上，看起来铁证如山。但<b>官方 v6e 文档白纸黑字写着「每个 TensorCore 有
2 个 MXU」，而 v6e 每颗芯片只有一个 TensorCore</b> —— 也就是 2 个 MXU，不是 4 个。
我把 MXU 数写多了一倍，又把每 cell 的乘加数写少了一倍，<b>两个错方向相反、各差一倍，
乘出来的 918 完全正确</b>。</p>
<div class="note bad"><span class="t">这类错误为什么最危险</span>
<b>一个对得上的答案，掩护了两个错误的前提。</b>而且它还骗过了一次刻意设计的证伪 ——
验尸台这个方法本身没问题，问题是<b>喂给它的参数也是我自己填的</b>。
所以「用已知样本反算」这招要真正有效，<b>被反算的那几个输入必须逐个有独立出处</b>，
不能有任何一个是顺手写下的。</div>

<h3>官方后来把这件事说死了</h3>
<p>Google 的工程博客在讲 Ironwood 调优时，逐字给出了这个乘式：</p>
<pre><code>262,144 FLOP / cycle / MXU  ×  2.2 GHz  ×  4 MXU  =  2,307 TFLOPS

→ 262,144 ÷ 2 ÷ (256 × 256) = 2      即每个 cell 每周期做 2 次乘加
→ 4 MXU × 65,536 cell × 2 MAC        = 524,288 乘加 / 周期 / chip</code></pre>
<p>三个输入（每 MXU 的 FLOP/周期、时钟、MXU 个数）<b>全部是官方给的</b>，
乘出来精确等于官方峰值。这一列不再是推导。</p>
<p>回代到 v6e 也立刻自洽：<code>2 MXU × 262,144 × f = 918 TF</code> → <b>f ≈ 1.75 GHz</b>，
正好和 v5p 同频，也刚好解释了官方宣称的「v6e 比 v5e 快 4.7 倍」＝
MXU 吞吐 4 倍 × 时钟 1.17 倍。<span class="g">（1.75 GHz 这个数是本文推导的，
不是官方值，因为官方从未公布 v6e 的时钟。）</span></p>

<h3>真正的结论：这个常数是分代的</h3>
<p>128×128 的那几代 —— v3、v4、v5e、v5p —— 全都还原成<b>每 cell 每周期 1 次乘加</b>，
其中 v3 的 940 MHz 和 v4 的 1,050 MHz 还是官方论文里的硬锚点。
到 256×256 这一代（v6e、v7），变成了 <b>2 次</b>。</p>
<p>所以 v2 的错误<b>不是算错，是把在老架构上验过的常数默认搬到了新架构上</b>。
这正是硬件材料里最容易翻车的一类断言 —— 它听起来像常识，因为它<b>曾经</b>是。</p>
<div class="note info"><span class="t">至于「2 次乘加」在硅上是怎么实现的</span>
公开资料<b>回答不了</b>：是每个 cell 里真的放了两个乘法器，还是这个 256×256 是逻辑视图、
物理上更大，官方从未说明。<b>本文只主张「每周期 262,144 FLOP」这个可观测量，
不主张微架构。</b></div>

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
<td>chipsandcheese 的 B200 实测（L1 39 周期、L2 21 / 16.8 TB/s）、TechPowerUp 规格库、
arXiv 2512.02189《Microbenchmarking NVIDIA's Blackwell Architecture》（B200 微基准，
TMEM 与 L2 分区数出自这里）、康奈尔 GPU 教程、
IEEE Micro 2021《The Design Process for Google's Training Chips: TPUv2 and TPUv3》。
<b>本文第一版曾引 arXiv 2507.10789 给 B200 的 L1/L2 背书，那篇测的是消费级 GB203，已撤换。</b></td></tr>
<tr class="hit"><td><span class="pill g">本文推导</span></td>
<td>B200 每个 Tensor Core 1,024 乘加 / 周期（A100 的 256 是<b>官方白皮书原文</b>，
B200 这个是推的，另有康奈尔教程旁证）；TPU v7 那一列<b>改为官方</b> ——
4 MXU × 262,144 FLOP/周期 × 2.2 GHz 三个数都出自 Google 工程博客</td>
<td>§2 用 A100 / H100 两代官方数字验证口径；§8 用 v5e / v6e 两代验证。
<b>「每 cell 双发」被撤回过一次、又被官方数字改回来了</b> ——
三版返工的完整过程写在 §8，那一段本身就是教学材料</td></tr>
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
<td>§5 文末（G-5 已不画灰框，结论留在正文）</td></tr>
<tr><td>TPU v7 的 VMEM / SMEM <b>带宽</b></td>
<td><b>官方未公开</b>。容量不一样 —— JAX 开源代码里写着 VMEM 64 MiB / core、
SMEM 1 MiB / core，所以 G-3 给了容量数字并注明出处；<b>带宽是真的查不到</b></td>
<td>G-6（灰色虚线框，不填带宽）</td></tr>
<tr><td>TPU v7 的 VLIW 槽位构成</td>
<td>本文用的是 v2/v3 论文的公开数字（2 标量 + 4 向量 + 2 矩阵 + 1 杂项 + 6 立即数），
<b>v7 官方没有公布</b>，不往 v7 上套</td><td>G-7（图上已标代次）</td></tr>
<tr><td>TPU v7 的时钟</td>
<td><b>这一条已经查到了，不再是开放问题</b>：Google 工程博客给出 v7 TensorCore
<b>2.2 GHz</b>，并连带给出 262,144 FLOP/周期/MXU 与 4 MXU/chip。
剩下真正查不到的是 <b>v6e 与 v5e 的时钟</b>（只有非产品文档给过 1.75 / 1.5 GHz），
以及 256×256 那「2 次乘加」在硅上如何实现</td><td>G-8 / §8</td></tr>
<tr><td>B200 的确切 boost 时钟</td>
<td>NVIDIA 没公布，<b>第三方之间也不一致</b>：TechPowerUp 给 1.83 GHz，
chipsandcheese 的 L1 实测（19.6 ns ＝ 39 周期）隐含约 1.99 GHz。
这就是 §2 那 1.41% 缺口的来源</td><td>G-2 / §2</td></tr>
</tbody></table></div>

<div class="note info"><span class="t">下一步</span>
这八张图画完就是为了移植进教学课件的专题二。<b>移植时图不用脱敏</b> ——
全文只用公开来源，没有任何需要抹掉的数字。<br>
但这一版是<b>返工过的</b>：两轮独立审查（一轮只查事实、一轮只看讲课效果）
推翻了第一版的三处结论 —— §5 整节的指令前提（B200 是 sm_100a，用不了那条 warp 级块缩放）、
§8 的「每 cell 双发」（当时被 v5e / v6e 证伪 —— <b>而这次证伪本身后来又被官方数字推翻了，
见下</b>）、以及 §2 那句「NVIDIA 从不公布」
（官方白皮书里白纸黑字写着）。<b>三处都在正文里留了返工记录，没有悄悄改掉</b>。<br>
<b>第三轮（事实核查）又推翻了第二轮的一个结论</b>：Google 工程博客给出了 v7 的
2.2 GHz 与 262,144 FLOP/周期/MXU，证明 256×256 的 MXU <b>确实是每 cell 每周期 2 次乘加</b>，
第二轮那次「证伪」用错了 v6e 的 MXU 个数（官方是 2 个，我写成 4 个），
两个反向的错误互相抵消、算出了正确的 918，于是掩护了错误的前提。
§8 的落点因此从「差 64 倍」改成 <b>「差 128 倍」</b>，整段返工过程按三版顺序完整写在 §8 ——
错在哪、怎么被发现的，比改对之后的结论更值得讲给学生听。<br>
同两轮审查还改掉了一批<b>只有画出来才看得见</b>的毛病，一并记在这里：
G-7 的图例有三个色块跟图上实际颜色对不上（图例和格子是两条独立的绘制路径，
现在图例直接调画格子的那支笔，从结构上不可能再漂移）；
G-7 两条合成条的<b>判据本来不一样却并排放着</b>，容易被读成打分，现在加了通栏说明；
G-2 把同一段说明誊了四遍（四个处理块确实一模一样，但那该由形状来说，不是把字写四遍）；
G-2 的「两个 SM 配对」原本画在处理块盒子<b>里面</b>，层级错位；
G-4 的三块叠图用半透明填充叠在同一个角上，渲染出来是三条认不出的色带，改成只描边；
G-4 补回了漏掉的 Turing（第 2 代），并按 PTX ISA 修正了 <code>tcgen05.mma</code> 在
<code>cta_group::2</code> 下 M 与 N 步长都翻倍这件事；
G-1 的 PCIe 卡片用了灰框，而灰色在那张图里的含义是「非官方来源」。
另外全套图的 <code>xs</code> / <code>xxs</code> 字号上调了 1 px —— 这份材料是投影讲课用的。</div>
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
