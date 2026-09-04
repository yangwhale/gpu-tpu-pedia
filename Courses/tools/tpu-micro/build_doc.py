# -*- coding: utf-8 -*-
"""TPU 显微镜 —— 把八张图组装成一份自包含的 HTML。

    python3 build_doc.py out.html --mode internal     # 默认
    python3 build_doc.py out.html --mode public       # 过闸门 + 出厂自检

**一份稿子，两种输出。** 所有内部信息走 gate.py，不复制第二份稿子 ——
理由写在 gate.py 的模块注释里。公开版构建完会自动跑 lint_public()，
命中任何禁字直接非零退出，不给「忘了删」留机会。

图由各 fig_t*.py 现场生成后内联进 HTML。还没画完的图会渲染成一个占位条，
这样每加一张图都能立刻构建出来看整体，不用等八张全齐。
"""
import io, os, sys, importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate

FIGS = ["fig_t1_chip", "fig_t2_core", "fig_t3_hierarchy", "fig_t4_mxu",
        "fig_t5_sparsecore", "fig_t6_datapath", "fig_t7_vliw", "fig_t8_pod"]

CSS = io.open(os.path.join(HERE, "style.css"), encoding="utf-8").read()
FAVICON = ("<link rel=\"icon\" href=\"data:image/svg+xml,"
           "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
           "<text y='.9em' font-size='90'>🦀</text></svg>\">")


def svg(i):
    name = FIGS[i - 1]
    if not os.path.exists(os.path.join(HERE, name + ".py")):
        return ('<div class="pending">图 T-%d（<code>%s</code>）还在画 —— '
                '这份文档是一张一张长出来的，占位条会随构建自动消失。</div>' % (i, name))
    return importlib.import_module(name).build()


def fig(i, cap):
    return (f'<figure class="figwide" id="t{i}">\n{svg(i)}\n'
            f'<figcaption>{cap}</figcaption>\n</figure>')


NAV = [("s0", "读之前"), ("s1", "1 全景"), ("s2", "2 一个核"), ("s3", "3 层级"),
       ("s4", "4 MXU"), ("s5", "5 SparseCore"), ("s6", "6 一个数走完全程"),
       ("s7", "7 VLIW"), ("s8", "8 到一个 pod"), ("s9", "来源等级"),
       ("s10", "还没查实")]


def build():
    P = []
    a = P.append
    pub = gate.is_public()

    a(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TPU 显微镜 —— 一颗 v7 从封装拆到 lane</title>
{FAVICON}
{CSS}
<style>.pending{{border:2px dashed #dadce0;border-radius:10px;padding:28px;
text-align:center;color:#5f6368;font-size:14px;background:#f8f9fa}}</style>
</head><body>

<header><div class="wrap">
<span class="eyebrow">🔬 与《GPU 显微镜》同一套标准，反过来拆一遍 TPU</span>
<h1>TPU 显微镜 —— 一颗 v7 从封装拆到 lane</h1>
<p class="lede">每个单元到底多大、彼此之间怎么连、谁算一个 lane、
一个数从 HBM 走到乘加阵列要经过几站、为什么这颗芯片<b>天生是拿来拼成一片的</b> ——
全部拆开画出来，并在每一处跟 B200 对照。八张图，每一张都能单独拿去讲。</p>
<div class="meta">
<span><b>八张图</b> · 全部为可讲课密度</span>
<span><b>来源分四级</b>：官方 / 第三方 / 本文推导 / 本文实测，图上逐处标注</span>
<span>{"<b>公开版</b> · 内部信息已按闸门过滤" if pub else "<b>内部版</b> · 含内部来源的数字"}</span>
</div>
</div></header>

<nav><div class="wrap">''')
    a("".join(f'<a href="#{i}">{t}</a>' for i, t in NAV))
    a('</div></nav>\n<div class="wrap">')

    # ── §0 ──────────────────────────────────────────────────────────
    a(f'''
<section id="s0">
<h2><span class="n">0</span>读之前：三件必须先说清的事</h2>
<p class="sub">口径、来源等级、以及这份文档跟原来那份 embedding 专题的关系。</p>

<div class="note warn"><span class="t">一、chip 和 device 的比例是 1 : 2，这是全文最容易翻车的地方</span>
一颗 TPU v7 <b>封装</b>里是两个 chiplet，而它们<b>如实暴露成两个独立的 JAX device</b>。
所有框架日志、<code>jax.devices()</code>、GKE 机型名里的数字，<b>按 device 算</b>。
所以：<code>tpu7x-128</code> 是 <b>64 颗芯片</b>；日志里的「1,153 TFLOP/s per device」
正是官方 2,307 的一半，不是掉了性能。算 MFU 时分母要么全用 2,307（按 chip），
要么全用 1,153.5（按 device）—— <b>混用会让结论直接差两倍</b>。</div>

<div class="grid g4">
<div class="stat b"><div class="k">官方</div><div class="v">直接引用</div>
<div class="d">Cloud TPU 产品文档、Google 官方博客与工程博客、JAX 开源代码、
以及 Google 自己发表的论文（IEEE Micro 2021 的 TPUv2/v3、ISCA 2023 的 TPU v4）。图上不加标记。</div></div>
<div class="stat y"><div class="k">第三方</div><div class="v">标灰</div>
<div class="d">非产品文档的来源 —— 包括 Google 作者写的教科书式博客。
它们看着很权威，<b>但本文实测到其中至少一条是错的</b>（见 §4），所以一律降级标灰。</div></div>
<div class="stat r"><div class="k">本文推导</div><div class="v">给出推导链</div>
<div class="d">官方没公布、但能从公开数字算出来的。图上标「推导值」，
正文把每一步写清楚，并且<b>尽量拿一代四个变量全公开的老芯片当验尸台</b>先验证公式。</div></div>
<div class="stat"><div class="k">本文实测</div><div class="v">说清怎么测的</div>
<div class="d">我们自己在 v7 上跑开源模型量出来的，任何文档里都查不到。
判断可不可信只能看<b>实验是怎么做的</b> —— 所以每一条都连着模型、尺寸和口径一起给（§9）。</div></div>
</div>

<div class="note info"><span class="t">二、这份文档和原来那份 embedding 专题的关系</span>
原来那份是<b>单点深挖</b>：把「embedding 查表在 v7 上到底怎么执行」这一件事挖到底。
这份把镜头拉远成<b>通用架构</b> —— embedding / SparseCore 收缩成其中一节（§5）。
它仍然是全文最好的具体案例，但不再是主线。<b>主线是：这颗芯片的每一层结构，
是为了回答「怎么把一个大矩阵乘做快」还是「怎么把上万颗芯片连成一台机器」。</b></div>

{"" if pub else '''<div class="note danger"><span class="t">三、这一版是内部版</span>
下面有一部分数字来自内部资料，<b>不要外发、不要进公开课件</b>：
<b>官方未公开的硬件规格</b>（片上带宽、时钟档位、核间通道、ISA 细节）和<b>内部代号</b>。
公开版由同一份稿子构建，构建时闸门会把这些条目删掉或换成「官方未公开」，并跑一遍禁字自检。
<br><br><b>不在此列的是「本文实测」那一类</b> —— 跑的是开源模型，量的是我们自己的结果，
两版都保留（见 §9）。</div>'''}
</section>''')

    # ── §1 ──────────────────────────────────────────────────────────
    a('''
<section id="s1">
<h2><span class="n">1</span>先看全景：一颗 v7 chip 里有些什么</h2>
<p class="sub">封装 → 两个 die → TensorCore ＋ SparseCore → HBM → 六个对外出口。</p>''')
    a(fig(1, "<b>图 T-1　一颗 TPU v7 全景。</b>两个 chiplet 结构完全相同，"
             "所以说明文字<b>只写在 die 0 上</b> —— 对称该由形状来表达，不该把字誊两遍。"
             "注意 HBM 那两块：96 GiB 各归各的 die，<b>两套地址空间互不可见</b>，"
             "die-to-die 互连只是让跨 die 搬运比出封装便宜，并没有把它们缝成一个 device。"))
    a('''
<p>这张图最值得停下来的一处，是它和 B200 <b>正好相反</b>的那个选择。</p>
<p>两家都在同一个物理事实面前：一颗 die 做不到想要的算力，只能上双 die 封装；
而双 die 之间的通路<b>必然</b>比 die 内部慢。区别只在于<b>把这条缝交给谁</b>。</p>
<p>NVIDIA 用 NV-HBI 这条一致性总线把两个 die 缝起来，对软件<b>装成一个 GPU</b>。
好处是你什么都不用改；代价是缝还在 —— L2 跨 die 访问从约 21 TB/s 掉到 16.8 TB/s，
而这件事<b>不会出现在任何 API 里</b>，只会出现在你的性能曲线里。</p>
<p>Google 选了另一头：<b>如实暴露成两个 device</b>。代价是你必须自己决定怎么切分，
好处是<b>缝在明处</b> —— 一旦你写下了分片策略，跨不跨 die 就是你自己写的，
不会有一条看不见的慢路径在背后偷走性能。</p>
<p>这不是谁对谁错，是两种成本转移。<b>但它解释了后面七张图里几乎所有的设计差异</b>：
当你决定把并行性交给软件之后，硬件里那些专门用来「自动藏起复杂度」的部件
—— 调度器、乱序、大寄存器堆、多级缓存 —— 就都可以省掉。省下的面积去了哪里，
就是下一张图的内容。</p>
</section>''')

    # ── §2 ──────────────────────────────────────────────────────────
    a('''
<section id="s2">
<h2><span class="n">2</span>把一个 TensorCore 拆开 —— 省掉的部件比留下的更说明问题</h2>
<p class="sub">2 个 MXU、1 个 VPU、标量单元、VMEM、SMEM，一个不落地画出来。</p>''')
    a(fig(2, "<b>图 T-2　一个 TensorCore 的显微镜展开。</b>左边是留下的部件，"
             "右边是<b>一个 GPU SM 里有、而这里没有</b>的五样东西。"
             "这张图的信息量主要在右边 —— 一颗芯片的性格，往往是被它<b>不做</b>的事定义的。"))
    a('''
<p>数一遍留下的：2 个 MXU、1 个 VPU、2 个跨 lane 单元、一个标量单元、
64 MiB VMEM、1 MiB SMEM。<b>就这些。</b></p>
<p>再数一遍没有的：没有 warp 调度器，没有几十份并发上下文，没有 256 KB 那种规模的寄存器堆，
没有会自动填充的 L1，没有记分板和乱序发射。这五样在 GPU 的 SM 里加起来占了相当可观的一片硅，
而它们的<b>共同职能只有一个</b> —— 在运行时动态地藏住延迟和分支。</p>

<div class="note info"><span class="t">这五样东西是一根绳上的</span>
它们不是五个独立的取舍，是<b>同一个决定的五个后果</b>。
一旦你决定「运行时要能换一个任务跑」，就必须同时驻留几十份上下文（要大寄存器堆）、
必须有东西来挑下一个（要调度器）、必须知道谁的数据到了（要记分板）、
必须让访存不至于把流水线卡死（要自动缓存）。
反过来，一旦你决定<b>不在运行时换任务</b>，这五样<b>同时</b>失去存在理由。
这就是为什么 TPU 的核看起来「少了很多东西」却依然自洽 —— 它不是被砍出来的，是另一条线上长出来的。</div>

<p>省下的面积去了哪里，图上给了一个直接的对照：<b>一颗 TPU v7 chip 上的 VMEM 是 128 MiB</b>
（两个核各 64 MiB），而一整颗 B200 的共享内存合计约 34 MB。
差别不只在容量 —— 更在于<b>那 128 MiB 的每一个字节都由编译器显式安排</b>，
不是硬件按访问历史猜着填的。§6 会把这件事讲透。</p>

<!-- ⛔ 2026-09-01 补。上面那段比的是「软件可控暂存」这一个口径，
     但**口径原来没写出来**，于是很自然会被读成「TPU 片上存储更多」——
     而那句话是**假的**：B200 片上 SRAM 总量 231 MiB，v7 只有 134 MiB。
     不写口径就等于默许读者按总量读，所以这里必须把反方向的数也摆出来。 -->
<div class="note warn"><span class="t">⚠️ 上面比的是「软件可控暂存」这一个口径 —— 换个口径结论会反过来</span>
<b>比片上 SRAM 总量，B200 是赢的</b>：寄存器堆 37 MiB ＋ L1／共享 37 MiB
＋ TMEM 37 MiB ＋ L2 126 MB（≈ 120 MiB）＝ <b>约 231 MiB</b>；
TPU v7 一颗 chip 是 VMEM 128 ＋ SMEM 2 ＋ 累加器 4 ＝ <b>约 134 MiB</b>。
<br>——&nbsp;<b>所以不能说「TPU 片上存储更多」，只能说「TPU 软件说了算的那部分更多」</b>
（130 对 37，约 3.5 倍）。B200 总量领先的那近 100 MiB 几乎全在 L2 里，
而 L2 恰恰是<b>唯一一格你没法显式安排的空间</b>。
<em>再收窄一档：单个计算单元能当成一整块连续工作台用的，是 227 KiB 对 64 MiB —— 差 289 倍。
这一档才是决定「一次融合能融多大」的那个数。</em>
<br><span class="sub">⚠️ 但要读准：这是<b>上限比上限</b>。227 KiB 是<b>一个线程块</b>的上限，
而那个 SM 上有 4 个 Tensor Core、还可能驻着别的块；64 MiB 那边也要跟<b>权重预取</b>分账
（由 <code>xla_tpu_scoped_vmem_limit_kib</code> 控制）。<b>两边的「独占」都有水分。</b></span></div>

<p>还有一处容易被略过：<b>标量单元在这颗芯片上比你以为的重要得多。</b>
它不参与矩阵乘，但 DMA 的描述符是它写出来的 —— 也就是说「什么时候搬、搬多少、搬到哪」
这件事，是<b>标量指令流里的一条指令</b>，不是硬件后台自动发生的行为。
一颗以矩阵乘为业的芯片，把「安排搬运」这件事放在了标量单元上，
这个安排在 §6 和 §7 会各出现一次。</p>
</section>''')

    # ── §3 ──────────────────────────────────────────────────────────
    a('''
<section id="s3">
<h2><span class="n">3</span>并行层级：lane / sublane 分别对应硬件上的什么</h2>
<p class="sub">和 GPU 的 thread / warp / block / grid 逐层对照 —— 层数少得多，少掉的那几层正是关键。</p>''')
    a(fig(3, "<b>图 T-3　并行层级逐层对照。</b>这张表刻意<b>按问题对齐，不按名词对齐</b> —— "
             "左边一列是问题，中右两列各自回答。名词对名词地排会得到一张翻译表，"
             "而按问题排，<b>有两行 TPU 那一栏是空的</b>，那才是要看的东西。"))
    a('''
<p>七个问题里，TPU 有两个答不出来：<b>「共享一块暂存的是哪一组」</b>和
<b>「运行时谁决定接下来跑哪个」</b>。</p>
<p>这两行不是随便哪两行。GPU 的 thread block 是一个<b>运行时</b>概念 ——
有几个 block、落在哪个 SM 上，启动的时候才知道；而 warp 调度器则是
<b>延迟出现之后</b>才发挥作用的东西。它们俩合起来，就是 GPU 藏延迟的全部机制。
TPU 把这两层都拿掉了，于是也就把「运行时还能补救」这条路一起拿掉了。</p>

<div class="note warn"><span class="t">最实用的一句话：两边的调优直觉不能互相搬运</span>
<b>GPU 的层级是「执行体的层级」</b> —— thread、warp、block 说的都是<b>谁在跑</b>；
<b>TPU 的层级是「数据形状的层级」</b> —— lane、sublane、tile、slice 说的都是<b>数据被切成什么样</b>。
所以 GPU 的性能问题多半出在「占用率」上（同时驻留的 warp 够不够多、能不能把延迟盖住），
TPU 的性能问题多半出在「形状」上（矩阵维度对不对齐、切片切得均不均匀）。
一个熟练的 CUDA 工程师第一次调 TPU，最常见的挫败感就来自这里：
<b>他熟悉的那套旋钮在这颗芯片上根本不存在。</b></div>

<p>顺带解释了一件常被问到的事：<b>为什么在 TPU 上「跑一个不规则的算法」这么别扭。</b>
不是编译器不肯，是<b>硬件的层级里根本没有一个能承载「不规则」的单位</b> ——
最小的可指名单位是「向量里的一格」，而它没有程序计数器，也不能走自己的分支。
你能表达的最小的「不同」，是让数据的<b>形状</b>不同，不是让某一格的<b>行为</b>不同。</p>
</section>''')

    # ── §4 ──────────────────────────────────────────────────────────
    a('''
<section id="s4">
<h2><span class="n">4</span>MXU：256×256 到底意味着什么</h2>
<p class="sub">脉动阵列里数是怎么流的，以及那个「每 cell 每周期几个乘加」的常数 —— 这一节还附带一个我自己犯过的错。</p>''')
    a(fig(4, "<b>图 T-4　MXU 脉动阵列的一个快照。</b>颜色深浅表示这个单元<b>什么时候开始干活</b>，"
             "同一条反对角线上的单元在同一拍工作 —— 这就是「脉动」两个字的字面意思。"
             "右边三张卡分别讲：峰值那条链是官方给的、256 是粒度不是上限、"
             "以及一个真实到扎心的例子（<code>head_dim=128</code>）。"))
    a('''
<p>先说结论，因为它反直觉：<b>K 比 256 大一点不亏，K 比 256 小才是真亏。</b></p>
<p>阵列的收缩边是物理的 256 行。K ＝ 1,024 就是走 4 趟，累加器一直不落地，
效率跟一趟算完几乎一样；而 K ＝ 128 时，<b>有一半的行在空转</b>，
并且 —— 这才是关键 —— <b>没有办法把别的活塞进那一半</b>。
回到 §3：TPU 没有「换一个跑」这一层，所以空着就是空着。</p>

<div class="note bad"><span class="t">一个真实例子：注意力的 head_dim ＝ 128</span>
多头注意力里，<code>QK&#7488;</code> 的收缩维和 <code>PV</code> 的输出维都等于 <code>head_dim</code>。
<b><code>head_dim</code> ＝ 128 时，这两处各只喂满 256 的一半</b> —— 纯几何，
不需要任何内部信息就能推出上限是 50%。
<br><br><b>实测佐证：</b>把 Qwen3-30B 的注意力从 32 头 × 128
改成 16 头 × 256（参数量和 FLOP 完全不变），MFU 在 8K / 16K / 32K 序列长度上
分别提升 <b>21% / 32% / 46%</b>。<br><br>
<b>结论不是「TPU 不适合注意力」</b>，是<b>模型配置和硬件的收缩边要一起选</b>。
同样的改动搬到 GPU 上收益接近于零，因为 GPU 的收缩维是 16，
而 128 是 16 的整整 8 倍 —— <b>同一个模型配置，在两边的「浪费」完全不在一个位置。</b></div>

<h3>插一段：我在这个常数上错了两次，而第二次错得更值得讲</h3>
<p>那条峰值公式 —— <code>262,144 FLOP / 周期 / MXU × 2.2 GHz × 4 个 MXU ＝ 2,307 TFLOP/s</code>
—— 是 Google 工程博客逐字给出的。但在找到这条公式之前，我先后写下过两个版本，
<b>两个都是错的，而第二个错误比第一个危险得多</b>。</p>

<div class="tw"><table>
<thead><tr><th style="width:56px">版本</th><th>当时写的</th><th>问题出在哪</th></tr></thead>
<tbody>
<tr><td><b>v1</b></td><td>4 个 MXU、每个 cell 每周期双发</td>
<td>结论其实是对的，<b>但当时没有任何证据</b> —— 它是为了凑平 2,307 这个已知答案倒推出来的</td></tr>
<tr class="miss"><td><b>v2</b></td><td>拿 v5e / v6e 当验尸台反算，判定「每 cell 一次乘加」，于是改成 8 个 MXU</td>
<td><b>反而错了。</b>反算用的输入里，「v6e 有几个 MXU」这一项是我自己填的，没有独立出处</td></tr>
<tr class="hit"><td><b>v3</b></td><td>官方公式出现，回到 4 个 MXU、每 cell 2 次乘加</td>
<td>—</td></tr>
</tbody></table></div>

<p>v2 那次的错法值得单独说。我用的方法本身是对的：<b>找一代四个变量全公开的老芯片，
拿它反算公式，验证通过了再套到新芯片上。</b>问题在于 v6e 那次反算里，
我算的是「4 个 MXU × 256×256 × 每 cell 1 次 @ 1.75 GHz ＝ 917.5 TFLOP/s ✓」——
数字完美对上了官方的 918。</p>
<p>但官方文档白纸黑字写着「每个 TensorCore 有 <b>2 个</b> MXU」，而 v6e 是 1 个 TensorCore/chip，
所以是 <b>2 个 MXU</b>，不是 4 个。正确的算式是
「2 个 MXU × 262,144 FLOP/周期 × 1.75 GHz ＝ 918 ✓」—— 同样对得上。</p>

<div class="note bad"><span class="t">为什么这类错误最危险：一个对得上的答案，掩护了两个错误的前提</span>
MXU 个数错了 2 倍（4 应为 2），每 cell 的乘加数也错了 2 倍（1 应为 2），
<b>两个 2 倍方向相反，乘起来正好抵消</b>。于是我得到一个和官方完全吻合的数字，
并且拿这个「验证通过」去否定了原本正确的 v1。
<br><br>
教训不是「要小心」，而是一条可执行的规则：<b>「用已知样本反算」这招要真正有效，
被反算的那几个输入必须<u>逐个</u>有独立出处。</b>
只要其中任何一个是你自己填进去的，这个方法就从「验证」退化成了「凑答案」——
而凑出来的答案看起来和真的一模一样。</div>

<p>还有一件事图上写了、这里再强调一遍：<b>「每 cell 2 次乘加」在硅上到底怎么实现，
公开资料答不了。</b>是每个 cell 里真放了两个乘法器，还是 256×256 只是一个逻辑视图、
底下另有排布？我不知道，所以本文不猜。<b>这一条会出现在 §10。</b></p>
</section>''')

    # ── §5 ──────────────────────────────────────────────────────────
    a('''
<section id="s5">
<h2><span class="n">5</span>SparseCore：第二种核，以及它到底在干什么</h2>
<p class="sub">原来那份 embedding 专题的核心结论，压缩成一节 —— 包括那个最反直觉的一条。</p>''')
    a(fig(5, "<b>图 T-5　SparseCore 与它在芯片里的归属。</b>左边那张有向图是重点："
             "两颗核之间的数据流是<b>单向</b>的。右下角那张卡则是全节的落点 ——"
             "在我们真实跑的生产任务里，SparseCore <b>一次表都没查过</b>。"))
    a('''
<p>关于 SparseCore，流传最广的一句话是「TPU 上有个专门查 embedding 表的核」。
这句话不算错，但它会让你在<b>看自己的 profile 的时候彻底看错</b>。</p>

<p>先把结构说清楚：一颗 chip 上有 <b>4 个 SparseCore</b>（每个 device 2 个），
每个由 1 个标量子核 ＋ 16 个向量子核组成，lane 宽 16，私有 SRAM 512 KiB。
它<b>没有 MXU</b>，做不了矩阵乘。它能做的是 <code>cumsum</code>、<code>sort_key_val</code>、
<code>fetch_and_add</code>、<code>addupdate_scatter</code> 这一类
<b>不规则访存 ＋ 归约</b>的活 —— 这些在开源的 Pallas SparseCore 接口里能直接看到。</p>

<div class="note info"><span class="t">最容易讲错的一条：「细 128 倍」指的是 tile 形状，不是 DMA 更快</span>
两颗核<b>共用同一个 HBM 控制器</b>，DMA 的通道宽度都是 32 B，
<b>SparseCore 并没有一个更快的搬运器</b>。差别在于最小可寻址的一块：
TensorCore 的 tile 是 <code>(8,128)</code> ＝ 4,096 B，SparseCore 是 <code>(8,)</code> ＝ 32 B。
散落在一张大词表里的几百行，用 4 KB 的块去取，取回来的绝大部分都会被扔掉 ——
<b>省的是无效流量，不是提高了带宽上限。</b></div>

<h3>那生产里的 embedding 到底怎么跑的？</h3>
<p>MaxText 里有一个开关叫 <code>use_iota_embed</code>。打开时，「查表」被写成
<b>「把 token id 展成 one-hot，再和整张词表做一次矩阵乘」</b> ——
一条 <code>dot</code>，跑在 MXU 上。关掉才是真的 <code>gather</code>。</p>
<p><b>上游的默认值是关的</b>（<code>configs/base.yml</code> 里写 <code>false</code>），
但仓库里 <b>34 份配置显式打开、0 份显式关闭</b> —— 连给 GPU 用的那几份也是打开的。
所以判断某一次跑走了哪条路，<b>只能看那一次实际生效的配置，不能看模型的 yml</b>。</p>

<div class="note bad"><span class="t">别说「反正 MXU 闲着，用算力换带宽很划算」—— 算术直接否掉了它</span>
Hunyuan3 那个尺寸下（M＝7×4096，K＝120,832，N＝4096），这条 matmul 是
<b>28.38 TFLOP/device</b>，占一步的 <b>0.62%</b>，约 24.6 ms；
同样的事用 gather 只要约 1.0 ms —— <b>matmul 慢约 24 倍</b>。
它不省时间，它<b>花</b>时间。这个选择成立的唯一理由是<b>分母够大</b>。
<br><br>真实动机我<b>没查实</b>，最强的候选是反向传播 —— scatter-add 在分片下不好做，
而 matmul 的反向天然还是 matmul。但这只是候选，<b>不写成结论</b>。</div>

<p>至于 SparseCore 为什么没被派去干这件事：看的是<b>同一批里「重复取同一行」的程度</b>。
推荐系统那边少量热行被反复命中，重复率极高，那是 SparseCore 的主场；
而语言模型的一个 batch 里几乎每个位置取的都是不同的词
（一批 131,072 个 token，词表 129,280 行，重合极少）——
落在<b>最不划算的那一端</b>，编译器于是根本没把它派过去。''' +
gate.I('''
<p>⛔ 内部：那条判据叫 <b>duplication factor</b> ＝ 这一批取了多少个索引 ÷
<b>其中不重复的行数</b>。<b>公式本身不进对外材料</b>（出处是内部资料）；
对外只说上面那句白话。</p>''', why="判据公式出自内部资料") + '''

<p><b>所以这一节真正的收获不是「SparseCore 是什么」，是「不要用宣传口径去解释自己的 profile」。</b>
那颗核确实存在、确实为稀疏访问而生，但在<b>你手上这个任务</b>里它可能在干完全不同的事
（比如集合通信卸载），甚至什么都没干。<b>看 trace，不要看宣传页。</b></p>
</section>''')

    # ── §6 ──────────────────────────────────────────────────────────
    a('''
<section id="s6">
<h2><span class="n">6</span>一个数走完全程：HBM → VMEM → MXU → 累加器</h2>
<p class="sub">每一站的容量、谁负责搬、以及「暂存」和「缓存」到底差在哪。</p>''')
    a(fig(6, "<b>图 T-6　一个数从 HBM 到乘加单元要经过几站。</b>"
             "每一站只问一个问题：<b>这一步是谁决定的</b>。"
             "从头到尾没有一站的答案是「硬件自己看着办」—— 这就是全图的落点。"))
    a('''
<p>几乎每份 TPU 材料都会列容量和带宽。但真正决定你写代码时会撞上什么的，
是图上那一整行 <b>「谁决定搬」</b>。</p>

<div class="note warn"><span class="t">VMEM 不是缓存，这不是措辞讲究，是两种不同的机器</span>
<div class="tw" style="margin-top:12px"><table>
<thead><tr><th style="width:200px">问题</th><th>GPU 的 L1 / L2（缓存）</th><th>TPU 的 VMEM（暂存）</th></tr></thead>
<tbody>
<tr><td>放什么进去</td><td>硬件按访问历史猜</td><td>编译器写死</td></tr>
<tr><td>有没有命中率</td><td>有，而且是主要调优指标</td><td><b>没有这个概念</b></td></tr>
<tr><td>猜错 / 排错的后果</td><td>变慢（多跑一趟内存）</td><td><b>停住（没有别的活可切）</b></td></tr>
<tr><td>你能控制到什么程度</td><td>间接：改访问顺序去哄它</td><td>直接：改分片和 tile 形状</td></tr>
</tbody></table></div></div>

<p>把这张小表念一遍就明白：<b>TPU 上不存在「缓存没命中」这种性能问题</b>。
不是因为它命中率高，是因为根本没有「命中」这个事件 —— 数据要么按班表到了，要么没到。
没到就是编译器排错了，而排错了没有兜底。</p>

<p>再说一处很多人第一次看会愣一下的：<b>搬运不是后台自动发生的</b>。
HBM ↔ VMEM 的每一次搬运都由 DMA 完成，而 DMA 的描述符是<b>标量单元写出来的</b>。
也就是说「什么时候搬、搬多少、搬到哪」跟乘加指令一样，<b>占着同一条指令流里的槽位</b>。
这解释了 §2 里那个看起来很怪的设计：为什么标量单元在一颗以矩阵乘为业的芯片上还这么重要
—— <b>它不算数，它安排搬运。</b></p>

<p>''' + gate.IP("带宽落差是这条通路的真正约束：<b>VMEM 约 34,428 GiB/s，"
                 "是 HBM 那 3,433 GiB/s 的整整 10 倍。</b>所以「尽量让数在 VMEM 里多待一会儿」"
                 "不是风格建议，是一个整整一位数的差距 —— 也是绝大多数 TPU 调优工作的实际内容。",
                 "带宽落差是这条通路的真正约束：片上暂存的读写带宽比 HBM 高<b>约一个数量级</b>"
                 "（具体数值官方未公开，这里只给量级）。所以「尽量让数在暂存里多待一会儿」"
                 "不是风格建议 —— 它是绝大多数 TPU 调优工作的实际内容。",
                 why="片上带宽属于未公开规格") + '''</p>

<p>最后一站顺手给一个能自己验的推导：累加器是 <b>128 个、形状 (8, 256)、32 bit</b>，
所以 128 × 8 × 256 × 4 B ＝ <b>1,048,576 B</b>，<b>正好 1 MiB / MXU</b>。
这类能对上整数的推导值得多做几次 —— 它是检查自己有没有把口径搞错的最便宜的办法。</p>
</section>''')

    # ── §7 ──────────────────────────────────────────────────────────
    a('''
<section id="s7">
<h2><span class="n">7</span>延迟被藏到哪里去了：VLIW 与编译期排班</h2>
<p class="sub">GPU 靠切换 warp 藏延迟，TPU 靠编译器提前排好 —— 代价是排错了没有兜底。</p>''')
    a(fig(7, "<b>图 T-7　VLIW 一拍多槽与编译期排班。</b>"
             "上面那一排是<b>公开论文里 TPU v2/v3 的 bundle 构成</b>（v7 的官方没有公开）；"
             "下面的甘特图是<b>示意，不是实测 trace</b> —— 它演示的是排班这件事的形状。"
             "灰色格子表示那一拍那个槽真的空着。"))
    a('''
<p>VLIW 的意思很朴素：<b>一拍要干的所有事，打成一包</b>。包里给标量、向量、矩阵、
杂项各留了固定的槽，谁往哪个槽里填、在第几拍填 —— 全部写死在指令流里。
没有乱序，没有记分板，没有「运行时再看」。</p>

<p>看甘特图的时候要盯两处。第一处是<b>第 1–6 拍的向量槽</b>：那里塞的是
<b>上一块</b>的后处理，和这次搬运没有依赖关系，所以能被提前挪过来盖住 DMA 的延迟。
<b>这就是编译器「藏延迟」的全部手段 —— 找独立的活。</b>
第二处是<b>第 14 拍之后</b>：独立的活用完了，所有槽都空着，
而且<b>不会有任何东西自动顶上来</b>，因为没有第二个线程可切。</p>

<div class="note ok"><span class="t">把 §2 那句话补完整</span>
§2 说 TPU「省下了调度器、寄存器堆、记分板那片硅」。这张图是那句话的另一半：
<b>省下来的东西并没有消失，它被搬到了编译期。</b>
调度这件事总得有人做 —— 区别只在于是硬件<b>每次运行时重做一遍</b>，还是编译器<b>一次做完</b>。
所以「TPU 更简单」是个误解：它不是把复杂度删掉了，是<b>把复杂度换了个地方放</b> ——
从硅上换到了编译器里，也从运行时换到了你写模型配置的那一刻。</div>

<p>这个取舍在日常里长成三件事，而这三件事其实是同一件：</p>
<ol>
<li><b>形状一不规则，性能就掉得很难看。</b>batch 小、序列不齐、专家路由不均 ——
这几种情况的共同点都是「后一步紧跟着前一步」，空拍没东西填。</li>
<li><b>编译很慢，而且慢得有道理。</b>v7 上一次 XLA 编译常见 <b>10–17 分钟</b>。
它不是在「翻译」，是在<b>替硬件把整张班表排完</b>；GPU 那边这件事是每次运行时由调度器现做的，
成本摊在了跑的时候。</li>
<li><b>但跑起来非常稳。</b>同一个形状重复跑，步时几乎没有抖动 —— 因为根本没有运行时决策可抖。
<b>可预测性是这个设计买来的东西，不是附带效果</b>：容量规划和性能回归检测在 TPU 上都因此简单不少。</li>
</ol>
</section>''')

    # ── §8 ──────────────────────────────────────────────────────────
    a('''
<section id="s8">
<h2><span class="n">8</span>从一颗到一个 pod —— 全文的落点</h2>
<p class="sub">ICI、3D 环面、cube、slice：为什么说 scale-out 是这颗芯片的第一性设计。</p>''')
    a(fig(8, "<b>图 T-8　从一颗 chip 到 9,216 颗。</b>主图是一根<b>对数刻度</b>的横轴 ——"
             "因为这件事的关键不是「谁能连更多」，而是<b>在哪个规模上你被迫换一套编程模型</b>。"
             "线性轴会把 72 和 9,216 压成一个点和一条线，那一刀就看不见了。"))
    a('''
<p>每颗 v7 有 <b>6 条 ICI 物理链路</b>，对应三维的正负方向，接成 3D 环面。
「环面」而不是「网格」的差别是实打实的：4×4×4 如果只是网格，最远要走 3+3+3 ＝ 9 跳；
接成环面之后是 2+2+2 ＝ <b>6 跳</b>。规模越大差得越多，而这直接决定了 all-reduce 的最坏时延。</p>
<p><b>但环面不是白拿的。</b>它要求切片在物理上必须是连续的一块立方体 ——
所以在 TPU 上你申请的不是「64 颗芯片」，是「一个 4×4×4」。
<b>形状本身是调度的一部分</b>，这一点在 §3 那张表里就已经埋下了伏笔。</p>

<div class="note ok"><span class="t">128 倍的差距不在带宽上，在「不换协议能连多远」上</span>
B200 的 NVLink 域是 <b>72 颗</b>，再往外就得换 InfiniBand 或以太网 ——
带宽掉一个量级，而且<b>集合通信要重写</b>。
TPU 的 3D 环面一路铺到 <b>9,216 颗</b>，全程同一套 ICI，
这两个数量级之间<b>通信代码一个字都不用改</b>。9,216 ÷ 72 ＝ <b>128</b>。
<br><br>
<b>但也别夸大。</b>单颗算力两边几乎打平（<b>HGX B200</b> 约 2,250 TFLOP/s BF16 dense，TPU v7 一颗 chip 2,307 —— <b>换成 NVL72 里那颗 GB200 是 2,500，反过来略高</b>），
而且在 72 颗以内，<b>NVLink 的每颗带宽还更高</b>（1.8 TB/s 对 1.2 TB/s）。
这里比的是<b>拓扑能延展多远</b>，不是单芯片谁快。</div>

<h3>两份文档合起来的那一句话</h3>
<p>《GPU 显微镜》的最后一张图讲的是：一颗 B200 里有 <b>592 个 Tensor Core</b>，
而一颗 TPU v7 里只有 <b>4 个 MXU</b> —— 同样一块矩阵乘的活，两边被切成的份数差 <b>128 倍</b>。
<b>GPU 的协调主要发生在芯片内部。</b></p>
<p>这张图讲的是另一半：TPU 一颗芯片里只有两个核要协调，
但不换协议能一路连到 9,216 颗，而 GPU 在 72 颗上就得换。
<b>TPU 的协调主要发生在芯片之间。</b></p>
<p>两个「128 倍」出现在完全不同的位置，这不是巧合，是同一个设计取向的两个侧面：
<b>一边把复杂度收进一颗芯片里由硬件消化，另一边把复杂度摊到芯片之间由软件面对。</b>
八张图从头到尾说的都是这同一件事 —— <b>不是谁更强，是把同一份复杂度放在了不同的地方。</b></p>
</section>''')

    # ── §9 来源等级总表 ─────────────────────────────────────────────
    a('''
<section id="s9">
<h2><span class="n">9</span>来源等级总表</h2>
<p class="sub">逐条列出每个数字来自哪里 —— 这样你可以只信你愿意信的那几行。</p>

<p>四级的划分标准写在 §0。这里只补一句：<b>「第三方」不等于「不可信」，
但它确实被降级了</b> —— 因为本文实测到其中至少一条是错的（那本流传很广的 TPU 教科书式博客
把 256×256 说成 131,072 FLOP/周期，比官方的 262,144 少一半，本文不采用）。</p>

<h3>官方 —— 可直接引用</h3>
<div class="tw"><table>
<thead><tr><th style="width:260px">事实</th><th style="width:210px">值</th><th>出处</th></tr></thead>
<tbody>
<tr><td>峰值那条完整算式</td><td>262,144 × 2.2 GHz × 4 ＝ 2,307 TF/s</td>
<td>Google 工程博客（三个数逐字给出，乘出来精确等于官方峰值）</td></tr>
<tr><td>MXU 尺寸 / 每核个数</td><td>256×256 · 2 个 / TensorCore</td><td>JAX 开源代码</td></tr>
<tr><td>累加器</td><td>128 个 · (8,256) · 32 bit</td><td>JAX 开源代码</td></tr>
<tr><td>lane / sublane</td><td>128 / 8</td><td>JAX 开源代码</td></tr>
<tr><td>VMEM / SMEM 容量</td><td>64 MiB / 1 MiB 每 core</td><td>JAX 开源代码</td></tr>
<tr><td>HBM 容量与带宽</td><td>192 GiB · 7,380 GB/s · 8 stack</td>
<td>Cloud TPU 产品文档（表头写的是 GiB）</td></tr>
<tr><td>SparseCore 内部构成</td><td>16 子核 · lane 16 · 512 KiB · 粒度 32 B</td>
<td>JAX 开源代码</td></tr>
<tr><td>SparseCore 的指令面</td><td>cumsum / sort / scatter …</td>
<td>开源 Pallas SparseCore 接口（其中<b>没有</b>任何矩阵乘原语）</td></tr>
<tr><td>ICI 链路数与总带宽</td><td>6 条 · 1,200 GB/s 每 chip</td>
<td>Cloud TPU 产品文档（<b>措辞自相矛盾，见 §10</b>）</td></tr>
<tr><td>3D 环面 / cube / pod</td><td>4×4×4 ＝ 64 · pod 9,216</td><td>Cloud TPU 产品文档</td></tr>
<tr><td>一台主机挂几颗</td><td>4 chips · 224 vCPU · 960 GB</td><td>Cloud TPU 产品文档</td></tr>
<tr><td>VLIW bundle 构成</td><td>322 bit ＝ 2＋4＋2＋1 ＋ 6 立即数</td>
<td>IEEE Micro 2021（<b>v2 / v3，不是 v7</b>）</td></tr>
<tr><td>老代次时钟</td><td>v3 940 MHz · v4 1,050 MHz</td><td>ISCA 2023 论文</td></tr>
<tr><td><code>use_iota_embed</code> 的默认值与用法</td><td>默认 false，34 份配置显式开</td>
<td>MaxText 开源仓库</td></tr>
</tbody></table></div>

<h3>第三方 —— 一律标灰</h3>
<div class="tw"><table>
<thead><tr><th style="width:260px">事实</th><th style="width:210px">值</th><th>为什么降级</th></tr></thead>
<tbody>
<tr class="dim"><td>B200 的 SM 数与每 SM 构成</td><td>148 SM × 4 Tensor Core</td>
<td>非官方拆解；本文只用它做数量级对照</td></tr>
<tr class="dim"><td>B200 共享内存合计</td><td>约 34 MB</td><td>同上</td></tr>
<tr class="dim"><td>B200 峰值与 NVLink</td><td>约 2,250 TF/s BF16 dense · 1.8 TB/s</td>
<td>厂商规格页；<b>这是 HGX B200 的口径</b>，NVL72 里的 GB200 为 2,500 —— 用前先确认是哪个 SKU</td></tr>
<tr class="miss"><td>「256×256 每周期 131,072 FLOP」</td><td>131,072</td>
<td><b>与官方的 262,144 差 2 倍。本文不采用，并在此明确标出。</b></td></tr>
</tbody></table></div>

<h3>本文推导 —— 每条都给推导链</h3>
<div class="tw"><table>
<thead><tr><th style="width:260px">结论</th><th>怎么算出来的</th></tr></thead>
<tbody>
<tr><td>每个 cell 每周期 <b>2 次</b>乘加</td>
<td>262,144 FLOP ÷ 2（一次乘加算 2 FLOP）÷ (256×256) ＝ 2。
<b>并且这是新架构才有的</b> —— 128×128 的 v3 / v4 / v5e / v5p 全部还原成每 cell 1 次</td></tr>
<tr><td>v6e 反算校验</td>
<td>2 个 MXU × 262,144 × 1.75 GHz ＝ <b>918 TF/s</b>，对上官方 918。
<b>这条校验的每一个输入都有独立出处</b>（MXU 个数来自官方文档，不是我填的）—— 见 §4 那段自陈</td></tr>
<tr><td>累加器合计 1 MiB / MXU</td><td>128 × 8 × 256 × 4 B ＝ 1,048,576 B</td></tr>
<tr><td>环面直径 6 跳</td><td>4×4×4 网格 3+3+3 ＝ 9；环面 ⌊4/2⌋×3 ＝ 6</td></tr>
<tr><td>「128 倍」（拓扑）</td><td>9,216 ÷ 72 ＝ 128</td></tr>
<tr><td>SparseCore 4 / chip</td>
<td>JAX 那张表在 per-device 口径下读出 2，而一颗 chip ＝ 2 个 device ⇒ 4。
<b>注意：该字段在不同代次之间口径不一致，不能只拿它推口径</b></td></tr>
<tr><td>官方博客的「pod HBM 1.77 PB」是单位混用</td>
<td>9,216 × 192 ＝ 1,769,472 —— <b>恰好等于把 GiB 当成 GB 直接乘</b>。
按十进制算应为约 1.90 PB。这不是两个矛盾的数，是同一个数的两种单位写法</td></tr>
</tbody></table></div>

<h3>本文实测 —— 我们自己在 v7 上跑出来的</h3>
<p>下面这几条<b>不是从任何文档里查来的</b>，是在 v7 上跑开源模型量出来的。
所以它们既不属于「官方」也不属于「第三方」—— 判断它们可不可信，
唯一的依据是<b>这份材料自己说了实验是怎么做的</b>。</p>
<div class="tw"><table>
<thead><tr><th style="width:260px">数字</th><th style="width:210px">值</th><th>怎么测的 / 怎么读</th></tr></thead>
<tbody>
<tr><td>注意力换形状后的 MFU 提升<br>（§4）</td><td>21% / 32% / 46%</td>
<td>Qwen3-30B，把注意力从 32 头 × 128 改成 16 头 × 256，<b>参数量和 FLOP 完全不变</b>，
只是让收缩维正好填满 MXU 的 256。三个数分别对应 8K / 16K / 32K 序列长度 ——
<b>序列越长收益越大</b>，因为注意力在整步里的占比越来越高</td></tr>
<tr><td>one-hot embedding 的代价<br>（§5）</td><td>28.38 TFLOP/device · 0.62% · 24.6 ms</td>
<td>Hunyuan3 那个尺寸（M＝7×4096，K＝120,832，N＝4096）。
同样的事用 <code>gather</code> 约 1.0 ms —— <b>matmul 慢约 24 倍</b>。
<b>请连着分母一起读</b>：它占一步只有 0.62%，所以「能接受」，不是「更快」</td></tr>
<tr><td>embedding 的取行重复度<br>（§5）</td><td>一批 131,072 个 token · 词表 129,280 行</td>
<td>语言模型几乎每个位置取的都是不同的词，<b>重合极少</b> —— 这正是 SparseCore
<b>最不划算</b>的那一端。⚠️ 本页只给这个观察，<b>不给判据的具体算式</b></td></tr>
<tr><td>XLA 编译耗时<br>（§7）</td><td>10–17 分钟</td>
<td>v7 上大模型一次编译的常见区间。<b>它和硬件规格无关，跟模型大小、
切片规模、XLA 版本都有关</b>，所以只当量级读，不要当基准</td></tr>
</tbody></table></div>
''' + gate.I('''
<h3>内部来源 —— 仅内部版可见</h3>
<div class="note bad"><span class="t">下面这些条目在公开版里已被闸门删除或替换</span>
<div class="tw" style="margin-top:12px"><table>
<thead><tr><th style="width:320px">条目</th><th>为什么算内部</th></tr></thead>
<tbody>
<tr><td>内部代号（v7x ＝ Ghostfish 等）</td><td>内部命名</td></tr>
<tr><td>TensorCore 默认 1.9 GHz、p-state 1.6–2.2</td><td>时钟档位</td></tr>
<tr><td>VMEM 带宽 34,428 GiB/s</td><td>片上带宽</td></tr>
<tr><td>主机接口 119.2 GiB/s</td><td>片上互联带宽</td></tr>
<tr><td>核间 DMA 通道清单与缺失的方向</td><td>内部接口定义</td></tr>
<tr><td>SparseCore 1.75 GHz / 5.85 TF/s</td><td>时钟与未公开峰值</td></tr>
<tr><td>ISA 层：向量寄存器个数、bundle 字节宽度</td><td>内部 ISA 定义，且未确认对应代次</td></tr>
</tbody></table></div></div>''', why="内部条目清单") + '''
</section>''')

    # ── §10 还没查实的 ──────────────────────────────────────────────
    a('''
<section id="s10">
<h2><span class="n">10</span>还没查实的</h2>
<p class="sub">查不到就写查不到 —— 这一节是这份文档最该被信任的部分。</p>

<p>下面每一条都是我<b>试着查了、没查到</b>的。它们没有出现在前面任何一张图的数字里，
出现的地方一律画成灰色虚线或直接写「查不到」。</p>

<h3>一、公开资料里找不到的</h3>
<div class="tw"><table>
<thead><tr><th style="width:300px">问题</th><th>现状</th></tr></thead>
<tbody>
<tr><td>v6e / v5e / v5p 的官方时钟</td>
<td>只有 v3（940 MHz）和 v4（1,050 MHz）有论文出处；v7 的 2.2 GHz 来自工程博客。
中间那几代<b>没找到可引用的官方数字</b> —— 所以本文不做「逐代时钟趋势」这种图</td></tr>
<tr><td>v7 向量寄存器的形状和个数</td>
<td>v2 / v3 论文给的是每 sublane 32 深。<b>v7 这一层官方没有公开</b>，
本文不拿旧代的数字顶替（§6 那一站因此画成虚线框）</td></tr>
<tr><td>v7 的 VLIW bundle 宽度与槽位构成</td>
<td>只有 v2 / v3 的 322 bit 是公开的。图上明确标了代次</td></tr>
<tr><td>「每 cell 2 次乘加」在硅上怎么实现</td>
<td>是每个 cell 真放了两个乘法器？还是 256×256 只是逻辑视图？<b>公开资料答不了，本文不猜</b></td></tr>
<tr><td>v7 的 INT8 / INT4 峰值</td>
<td>JAX 那张表里这两项都是 <b>0</b>。但「0」到底是「不支持」还是「没填」，
<b>我没能确认</b> —— 所以本文全篇不提 v7 的整数峰值</td></tr>
<tr><td>die 面积 / 晶体管数 / TDP / 制程</td><td>官方一项都没公开</td></tr>
<tr><td>一个作业能调度到的最大切片</td>
<td>物理上 9,216 连成一个环面是官方数字，但「一次能拿到多大一块」取决于调度系统，
<b>公开资料里没有可引用的上限</b></td></tr>
<tr><td>边缘切片能不能拿到环绕链路</td>
<td>也就是「环面在多大规模上会退化成非环」。公开资料同样没说。<b>不猜</b></td></tr>
</tbody></table></div>

<h3>二、官方资料自己对不上的</h3>
<p>这类比「查不到」更麻烦 —— 它会让你以为自己查到了。</p>
<div class="tw"><table>
<thead><tr><th style="width:300px">矛盾</th><th>本文怎么处理</th></tr></thead>
<tbody>
<tr class="miss"><td>ICI：正文写「每<b>轴</b>双向 200 GB/s」，
但 3 轴 × 200 ＝ 600，对不上同一页表格里的 1,200</td>
<td>只有读成「每条<b>链路</b> 200」才自洽（6 条 × 200 ＝ 1,200）。
<b>本文按这个读法画，并在图上写明原文是另一种措辞</b></td></tr>
<tr class="miss"><td>「pod」有两个官方定义：9,216 颗 / 256 颗</td>
<td>两个都是官方说法，没有一处说明哪个作准。
<b>本文一律指 9,216 那个，并且每次都把数字写出来</b></td></tr>
<tr class="miss"><td>HBM 同一页里既写 GiB 又写 GB</td>
<td>表头写的是 GiB。<b>对外一律说 192 GiB</b>，不说 206 GB
（那是同一个容量的十进制写法，业界没人这么报）</td></tr>
<tr class="miss"><td>SparseCore 私有 SRAM：512 KiB / 子核 vs 公开 v4 材料的 2.5 MB / SC</td>
<td>差一个量级，<b>我没能判定这两个是不是在说同一个东西</b>。
本文只用 JAX 开源代码里的 512 KiB，并标明它的口径是「每子核」</td></tr>
</tbody></table></div>

<div class="note q"><span class="t">为什么把这一节放进讲课材料</span>
一份材料的可信度，不取决于它讲对了多少，取决于<b>它有没有能力说「这个我不知道」</b>。
上面这十几条如果强行填上一个「看起来合理」的数字，整份文档会显得更完整、更好看，
而且<b>几乎没有人会当场发现</b> —— 这正是它危险的地方。
<br><br>
§4 那段自陈是这条原则的最好注脚：我一度用一个<b>算得完全正确的结果</b>，
去支撑两个错误的前提。<b>能对上的答案不等于对的推理</b>；
唯一的解法是把每一个输入都追到独立出处，追不到就如实标出来。</div>
</section>''')

    a('''
<footer>八张图均由脚本生成（<code>Courses/tools/tpu-micro/</code>），改数据重跑即可，不用手改 SVG。
每张图都经过渲染 → 截图 → 目视核对的循环。内部版与公开版由<b>同一份稿子</b>构建，
过滤规则见 <code>gate.py</code>。</footer>
</div></body></html>''')
    return "\n".join(P)


if __name__ == "__main__":
    args = [x for x in sys.argv[1:]]
    mode = "internal"
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        del args[i:i + 2]
    out = args[0] if args else f"/tmp/tpu-micro-{mode}.html"

    gate.set_mode(mode)
    html = build()
    io.open(out, "w", encoding="utf-8").write(html)
    print(f"ok  {out}  {len(html):,} chars  mode={mode}")

    if mode == "public":
        hits = gate.lint_public(html)
        if hits:
            print("\n❌ 出厂自检没过 —— 公开版里出现了这些字样：")
            for w, d in hits:
                print(f"   · {w}   （{d}）")
            sys.exit(1)
        print("✅ 出厂自检通过（bug 号 / go 链接 / 内部域名 / 内部路径 / 内部来源署名 全部为零）")
    print()
    print(gate.audit_report())
