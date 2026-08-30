# -*- coding: utf-8 -*-
"""把《GPU 显微镜》《TPU 显微镜》的十六张图移植进专题二。

    python3 topic02-port-microscope.py

**为什么要有这个脚本，而不是手动粘贴。** 两份显微镜文档还在改，
图是脚本现场生成的。手动粘一次，下次改了图就得记着再粘一遍 ——
迟早忘，然后课件上挂的是旧图。这个脚本可以反复跑：
它用一对注释标记圈出自己写过的区域，重跑时整段换掉。

三处必须由脚本处理、手动很容易做错的地方：

1. **`<defs>` 只能出现一次。** 每张图自带一份 `<defs>`，里面有 `id="aB"`
   这些箭头 marker。十六份粘进同一个页面 = 十六个重复 id，
   `url(#aB)` 只会命中第一个。这里把 defs 抽出来在页面顶部放一份。
2. **宽度要改成百分比。** 图是按 1400 px 画的，`width="1400"` 会撑破版心。
   但换成 100% 塞进 1080 px 的版心，11 px 的字会缩到 8 px —— 讲课看不清。
   所以配一个 `.fwide`，让这十六张图突破版心、按接近原生的宽度渲染。
3. **TPU 侧必须用公开模式导出。** 这个仓库是公开的。
   脚本里写死 `gate.set_mode("public")`，并在最后跑一遍禁字自检。
"""
import io, os, re, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "..", "WebPages", "topic-02.html")

BEG = "<!-- ▼▼ 由 topic02-port-microscope.py 生成，不要手改 ▼▼ -->"
END = "<!-- ▲▲ 生成区结束 ▲▲ -->"

GPU_URL = "gpu-microscope.html"
TPU_URL = "tpu-microscope.html"


# ══════════════════════════════════════════════════════════════════════
# 一、把十六张图取出来
# ══════════════════════════════════════════════════════════════════════
GPU = [(1, "chip"), (2, "sm"), (3, "hierarchy"), (4, "mma"),
       (5, "blockscale"), (6, "memory"), (7, "latency"), (8, "scale")]
TPU = [(1, "chip"), (2, "core"), (3, "hierarchy"), (4, "mxu"),
       (5, "sparsecore"), (6, "datapath"), (7, "vliw"), (8, "pod")]


def _harvest():
    """两棵树各有一个同名的 common.py，同进程 import 会打架 —— 各起一个子进程。"""
    svg = {}
    code = (
        "import sys, importlib\n"
        "sys.path.insert(0, '.')\n"
        "%s\n"
        "for k, mod in %r:\n"
        "    open('/tmp/port/' + k + '.svg', 'w').write("
        "        importlib.import_module(mod).build())\n")
    jobs = [("gpu-micro", "", [("g%d" % n, "fig_g%d_%s" % (n, s)) for n, s in GPU]),
            ("tpu-micro", "import gate; gate.set_mode('public')",
             [("t%d" % n, "fig_t%d_%s" % (n, s)) for n, s in TPU])]
    os.makedirs("/tmp/port", exist_ok=True)
    for tree, pre, mods in jobs:
        subprocess.run([sys.executable, "-c", code % (pre, mods)],
                       cwd=os.path.join(HERE, tree), check=True)
    for k in [j[0] for job in jobs for j in job[2]]:
        svg[k] = io.open("/tmp/port/%s.svg" % k, encoding="utf-8").read()
    return svg


_DEFS_RE = re.compile(r"<defs>.*?</defs>", re.S)
_SIZE_RE = re.compile(r'width="\d+" height="\d+" ')

# 「第 N / 8 张」是显微镜文档里的页码，搬到课件里就成了噪声 ——
# 听众会去找「另外七张在哪」。徽章由 Fig.title() 画成「一个 rect 紧跟一个 text」，
# 所以连着删。**只删页码**：G-4 的「PTX ISA 9.3」、G-1 的「灰字＝官方未标」
# 那种是图例，必须留着。
_PAGE_BADGE = re.compile(
    r'<rect[^>]*\sy="10"[^>]*rx="11"[^>]*/>\s*'
    r'<text[^>]*\sy="25"[^>]*>第 \d+ / 8 张</text>\s*')


_ID_RE = re.compile(r'\sid="([^"]+)"')


def _split(key, raw):
    """→ (公共 defs, 去掉 defs／页码徽章、宽度百分比化、id 加过前缀的 svg)"""
    m = _DEFS_RE.search(raw)
    defs = m.group(0)
    body = _DEFS_RE.sub("", raw, count=1)
    body = _SIZE_RE.sub('width="100%" ', body, count=1)
    body = _PAGE_BADGE.sub("", body)

    # 头部那份 defs（三个箭头 marker）十六张图完全一致，抽成一份共用 ——
    # main() 里有断言兜着。**但图自己在正文里再开的 <defs> 就不一定了。**
    #
    # 实例：fig_t1_chip 和 fig_g8_scale 各自定义了一个叫 mxucell 的 pattern。
    # 各自成篇时毫不相干，合进同一个页面，`url(#mxucell)` 就只命中文档里的第一个 ——
    # 后一张图的网格会**悄悄换成前一张的颜色**。渲染不报错，浏览器不警告，
    # 只是画错，而且要盯着两张图对比才看得出来。
    #
    # 与其回去改那两个图的源码（只修得掉今天这一个，明天新加一张图照撞），
    # 不如在这里给每张图的非共用 id 统统加上图名前缀，
    # 让「两张图撞 id」这件事**结构上不可能发生**。
    shared = set(_ID_RE.findall(defs))
    for i in sorted(set(_ID_RE.findall(body)) - shared, key=len, reverse=True):
        body = (body.replace('id="%s"' % i, 'id="%s-%s"' % (key, i))
                    .replace('url(#%s)' % i, 'url(#%s-%s)' % (key, i))
                    .replace('href="#%s"' % i, 'href="#%s-%s"' % (key, i)))
    return defs, body.replace("\n\n", "\n")


# ══════════════════════════════════════════════════════════════════════
# 二、图注 —— 移植过来要重写，不能照抄显微镜里的那一句
#
# 原文那句是给「读文档的人」写的；这里是给「听课的人」写的，
# 所以每一句都必须挂回专题二自己的主线（放得下 / 算得动 / 卡间说话 / 谁做决定），
# 而不是复述图上已经有的信息。
# ══════════════════════════════════════════════════════════════════════
CAP = {
 "g1": "<b>一颗 B200 的全景。</b>两颗 die 用 NV-HBI 缝成「一个 GPU」—— "
       "对软件完全透明，代价是那道缝还在：L2 跨 die 从约 21 TB/s 掉到 16.8 TB/s，"
       "<b>而这件事不出现在任何 API 里</b>。",
 "t1": "<b>一颗 TPU v7 的全景，同样两颗 die。</b>但它<b>如实暴露成两个 device</b> —— "
       "缝在明处。<b>这两张图并排看，就是第 0 节那两条出身的第一次具体化</b>："
       "一边把复杂度藏起来，一边把它交给你。",
 "g3": "<b>GPU 的四层：thread → warp → block → grid。</b>"
       "注意 block 和 warp 调度器都是<b>运行时</b>概念 —— 有几个 block、落在哪个 SM，"
       "启动才知道。这两层就是 GPU 藏延迟的全部本钱。",
 "t3": "<b>TPU 这边刻意按问题对齐，不按名词对齐。</b>"
       "名词对名词能排出一张漂亮的翻译表，按问题排才看得见<b>有两行 TPU 那一栏是空的</b>。"
       "空的那两行正是 GPU 的 block 和 warp 调度器。",
 "g6": "<b>B200 侧五站，中间两站是缓存。</b>"
       "命中不命中要跑起来才知道 —— 这就是 2.1 那五个红框的物理来源。",
 "t6": "<b>v7 侧四站，中间那站不是缓存，是暂存。</b>每一站只问一个问题："
       "<b>这一步是谁决定的</b>。从头到尾没有一站的答案是「硬件自己看着办」—— "
       "这张图是 2.1 里「TPU 侧 0 个红框」的逐站展开。",
 "g2": "<b>一个 SM 拆开。</b>把面积清单念一遍：warp 调度器、256 KB 寄存器堆、"
       "228 KB L1／共享内存、记分板。<b>这些部件加起来只干一件事 —— "
       "在运行时动态地藏住延迟和分支。</b>",
 "t2": "<b>一个 TensorCore 拆开，重点在右边那一栏。</b>"
       "上面那五样在这里<b>一样都没有</b>。而它们不是五个独立的取舍，是同一个决定的五个后果："
       "一旦你决定<b>不在运行时换任务</b>，这五样同时失去存在理由。",
 "g4": "<b>Tensor Core 一条指令吃多大一块。</b>"
       "GPU 的收缩维是 <code>16</code> —— 记住这个数，下一张图要用它。",
 "t4": "<b>MXU 是 256×256，收缩维比 GPU 大 16 倍。</b>"
       "所以同一个 <code>head_dim = 128</code>：在 GPU 上是 16 的 8 倍，切八条指令、不浪费；"
       "在 TPU 上只喂满 256 的一半，<b>而空着的那一半没有别的活能顶上来</b>。"
       "<b>同一个模型配置，两边的「浪费」完全不在一个位置</b> —— 这是 2.5 那条反直觉的硬件根源。",
 "g5": "<b>块量化：多细的一撮数共享一个缩放因子。</b>"
       "Blackwell 把它做进了硬件，但两条通路<b>不在同一颗 die 上</b> —— "
       "所以「B200 支持 FP4」这句话要看你说的是哪一条。",
 "t5": "<b>TPU 这边的第二种核。</b>它<b>没有矩阵乘单元</b>，"
       "干的是不规则访存加归约。但要小心：<b>「专门查 embedding 表的核」这句话，"
       "会让你看自己的 profile 时彻底看错</b> —— 生产任务里它可能一次表都没查过。",
 "g8": "<b>NVLink 域到 72 颗就是边界。</b>再往外换 InfiniBand 或以太网，"
       "带宽掉一个量级，<b>而且集合通信要重写</b>。",
 "t8": "<b>3D 环面一路铺到 9,216 颗，全程同一套 ICI。</b>"
       "横轴是<b>对数刻度</b> —— 关键不是谁能连更多，是<b>在哪个规模上你被迫换一套编程模型</b>。"
       "9,216 ÷ 72 ＝ <b>128</b>。但别夸大：单颗算力两边几乎打平，"
       "72 颗以内 NVLink 每颗带宽还更高。<b>这里比的是拓扑能延展多远。</b>",
 "g7": "<b>GPU 怎么藏延迟：一个 warp 卡住就换下一个。</b>"
       "藏得住的前提是<b>同时驻留的 warp 够多</b> —— 这就是「占用率」这个词的全部含义。",
 "t7": "<b>TPU 怎么藏延迟：编译器提前把班排好。</b>"
       "灰格子是真的空着，而且<b>不会有任何东西自动顶上来</b>。"
       "上面那排 bundle 构成是<b>公开论文里 v2/v3 的</b>，下面的甘特图是<b>示意，不是实测 trace</b>。",
}


def figure(key, svg_body):
    return ('<figure class="fbox fwide" id="ms-%s">\n%s\n'
            '<figcaption>%s　<span class="msfrom">—— 出自%s</span></figcaption>\n'
            '</figure>' % (key, svg_body, CAP[key],
                           '《<a href="%s">GPU 显微镜</a>》' % GPU_URL if key[0] == "g"
                           else '《<a href="%s">TPU 显微镜</a>》' % TPU_URL))


# ══════════════════════════════════════════════════════════════════════
# 三、正文
# ══════════════════════════════════════════════════════════════════════
def sections(F):
    """F(key) → 一个完整的 <figure> 块。

    **写旁白的唯一规矩：不复述图里已经有的话。**
    这十六张图本来就是按「可讲课密度」画的 —— 该说的都在图上。
    第一版旁白把 T-7 图里那两个框、T-8 图里的环面段几乎逐句抄了一遍，
    渲染出来才发现同一段话在同一屏里出现两次。
    所以这里每张图后面只回答一个问题：**它和本课别的部分怎么接。**
    """
    s = []
    a = s.append

    # ── §3 计算单元 ──────────────────────────────────────────────
    a('''
<section><div class="wrap">
  <div class="stn big"><span class="badge">第 3 节</span><h2>拆开看：同一份复杂度，两边放在了不同的地方</h2></div>
  <p class="lead">第 2 节数的是<b>决策点</b>，这一节数的是<b>份数</b>。
    同样一块矩阵乘的活，两边被切成的份数差 <b>128 倍</b> ——
    这个数字一旦记住，后面「为什么这个优化在那边不管用」大半都能自己推出来。</p>

  <div class="note info"><span class="t">下面十六张图的读法</span>
    这一节起的图都搬自两份《显微镜》文档，<b>图上的「§N」指的是那两份文档里的节号，
    不是本课的节号</b>。每张图右下角都标了出处，点进去就是对应那一节。
    <b>建议按顺序讲</b>：从整颗芯片开始，一层层往里，最后落到「谁做决定」。</div>

  <h3>3.1　先看整颗，再谈内部</h3>
  <p>两颗芯片<b>都是双 die 封装</b>，这是先讲全景的理由 ——
    如果直接跳进核内部，会漏掉一个只在封装层面才看得见的差别。先看 B200：</p>''')
    a(F("g1"))
    a('''
  <p>再看 v7。<b>物理构造几乎一样，对软件的呈现完全相反。</b></p>''')
    a(F("t1"))
    a('''
  <div class="note info"><span class="t">这一对图是全课的缩影，值得在这儿多停一分钟</span>
    图上已经把「一边缝在暗处、一边缝在明处」说清楚了。要补的是<b>它意味着什么</b>：
    <br><b>藏得好的时候你省事，藏漏了的时候你连查都不知道从哪查起。</b>
    B200 那道缝不出现在任何 API 里，所以它只会表现为「这个 kernel 莫名其妙慢了」；
    v7 那道缝写在脸上，麻烦但可查。
    <br><br><b>后面每一层都会再遇到这个对子一次</b> ——
    核内部、层级表、访存路径，全是同一个选择在不同尺度上的复现。
    <b>如果这门课只让听众记住一件事，就是这一件。</b></div>

  <h3>3.2　把一个核拆开</h3>
  <p>往里一层。<b>不要从参数表开始，从「有什么」和「没有什么」开始。</b>
    先看 GPU 这边留下了哪些部件 ——</p>''')
    a(F("g2"))
    a('''
  <p>再看 TPU 这边。<b>这张图的信息量在右边那一栏</b>，上面那些部件在这里一样都没有。</p>''')
    a(F("t2"))
    a('''
  <div class="note info"><span class="t">接回第 2 节：这不是「砍配置」，是同一个决定的连锁反应</span>
    右边那五样看着像五个独立的取舍，其实是一个：<b>要不要在运行时换任务跑</b>。
    答「要」，就必须同时驻留几十份上下文、必须有东西挑下一个、必须知道谁的数据到了 ——
    五样缺一不可。答「不要」，五样<b>同时</b>失去存在理由。
    <br><br>而第 2 节那五个红框，就是答「要」之后的必然产物。
    <b>「有没有 cache」和「有没有 warp 调度器」不是两件事，是同一个决定的两个面。</b></div>

  <h3>3.3　并行层级：哪几层是运行时才定的</h3>
  <p>上一节说的「运行时换任务」，具体是在哪一层发生的？把两边的层级摆开对照。
    <b>GPU 这边有四层</b>，注意哪几层是启动之后才知道的 ——</p>''')
    a(F("g3"))
    a('''
  <p>TPU 这边<b>刻意按问题排，不按名词排</b>。名词对名词能排出一张漂亮的翻译表，
    按问题排才看得见<b>有两行是空的</b>。</p>''')
    a(F("t3"))
    a('''
  <div class="note warn"><span class="t">这张表最容易被听众误用的方式：拿它当翻译词典</span>
    图上标出的两个空格，正是上一节 TPU 核里少掉的那批部件 —— 同一件事的第二个视角。
    但接下来几乎每次都会有人问：<b>「那 TPU 上的 warp 到底是什么？」</b>
    <br><br><b>这个问题要正面挡回去：它没有对应物。</b>
    那一层解决的问题（运行时挑谁跑）在 TPU 上根本不存在，
    硬给它找一个对应物，等于<b>给自己造一个不存在的心智模型</b> ——
    然后你会用它去解释 profile，然后每一次都解释错。
    <b>跨平台学习翻车，八成翻在这儿，不是翻在记错参数。</b></div>

  <h3>3.4　一条指令吃多大一块 —— 这里出那个 128 倍</h3>
  <p>再往里一层，到指令。两边都有专门的矩阵乘单元，要问的只有一句：
    <b>一条指令一次吃进去的矩阵有多大。</b>先看 GPU，记住它的收缩维。</p>''')
    a(F("g4"))
    a('''
  <p>TPU 这边的收缩边是 <b>256</b>，正好是上面那个数的 16 倍。
    图上第三张卡把 <code>head_dim = 128</code> 那个例子算完了。</p>''')
    a(F("t4"))
    a('''
  <div class="note danger"><span class="t">⭐ 这一对图直接回答了第 2 节结尾那条反直觉</span>
    第 2 节讲 <code>head_dim = 128</code> 在 TPU 上浪费 1/2 时，只说了「MXU 是 256×256」。
    <b>并排看这两张图，才知道这句话在 GPU 上根本不成立</b> ——
    GPU 的收缩维是 16，128 是它的整整 8 倍，切八条指令，<b>一点不浪费</b>。
    <br><br><b>所以「head_dim 要对齐硬件」这条建议是有前提的</b>：
    同一个模型配置，两边的浪费根本不在一个位置。
    在 GPU 上调它收益接近于零，在 TPU 上是实打实的。
    <b>这也是为什么跨平台照搬调优经验特别容易翻车。</b></div>

  <h3>3.5　各自多出来的那一块</h3>
  <p>两边都在主计算单元之外挂了点别的东西，<b>但挂的方向正好相反</b>：
    GPU 往「更细的数据类型」走 ——</p>''')
    a(F("g5"))
    a('''
  <p>TPU 往「更不规则的访存」走。</p>''')
    a(F("t5"))
    a('''
  <div class="note warn"><span class="t">这两块「多出来的」有个共同点，讲的时候值得点一句</span>
    它们都<b>不是给通用计算用的</b>，而且都容易被一句宣传话带偏：
    <br>· 「B200 支持 FP4」—— 但两条通路<b>不在同一颗 die 上</b>，
      你说的是 warp 级那条还是双 SM 那条，能不能用完全不同；
    <br>· 「TPU 有个专门查 embedding 表的核」—— 但判据是 duplication factor，
      语言模型落在<b>最不划算的那一端</b>，编译器根本没往那儿派。
    <br><br><b>共同的教训是同一条：看 trace，不要看宣传页。</b>
    这两个坑在课上被问到的频率极高，而且都是「说法没错、但用它解释自己的 profile 就会全错」。</div>

  <h3>3.6　最后把一个数从内存送到计算单元</h3>
  <p>前面五小节拆的都是<b>部件</b>，这一小节走的是<b>路径</b> ——
    一个数从 HBM 出发到进 MXU／Tensor Core，中间要停几站，每一站是谁在做决定。
    GPU 侧五站，<b>中间两站是缓存</b>：</p>''')
    a(F("g6"))
    a('''
  <p>TPU 侧四站，<b>中间那站不是缓存，是暂存</b> —— 一字之差，性质完全不同。</p>''')
    a(F("t6"))
    a('''
  <div class="note ok"><span class="t">这两张图的讲法：不要念站名，逐站问同一个问题</span>
    站名念一遍听众记不住，也没意义。<b>正确的讲法是从左到右逐站问：这一步是谁决定的。</b>
    问完会得到一个很干净的结果 ——
    <b>GPU 侧有两站答「硬件，运行时才知道」，TPU 侧一站都没有。</b>
    <br><br>而这正是第 2 节里「GPU 五个红框、TPU 零个」的来历：
    <b>那五个红框不是随手圈的，就是沿着这条路径一站一站数出来的。</b>
    到这里第 2 节那张图才算真的讲完了。</div>
</div></section>''')

    # ── §4 互联 ──────────────────────────────────────────────────
    a('''
<section><div class="wrap">
  <div class="stn big"><span class="badge">第 4 节</span><h2>卡间说话：连得多远，比连得多快更要紧</h2></div>
  <p class="lead">这一节只有一对图，但它是整门课里<b>差距最大的一处</b>。
    而且要先把话说公道：<b>差的不是带宽。</b></p>
  <p>先看 GPU 这边的边界在哪 ——</p>''')
    a(F("g8"))
    a('''
  <p>再看 TPU 这边能铺多远。<b>务必提醒听众横轴是对数刻度</b> ——
    线性轴会把 72 和 9,216 压成一个点和一条线，那一刀就看不见了。</p>''')
    a(F("t8"))
    a('''
  <div class="note ok"><span class="t">讲这一对图时，最容易被听众抓住的一个反问</span>
    「那不就是 TPU 互联更强吗？」—— <b>不是。</b>单颗算力两边几乎打平
    （B200 约 2,250 TFLOP/s BF16，v7 一颗 chip 2,307），
    而且在 72 颗以内，<b>NVLink 的每颗带宽还更高</b>（1.8 TB/s 对 1.2 TB/s）。
    <br><br>差别只出现在<b>越过 72 颗那一刻</b>：一边要换协议、换传输层、
    <b>还要重写集合通信</b>，另一边通信代码一个字不动。
    <b>所以这 128 倍衡量的是「拓扑能延展多远」，不是「谁跑得快」</b> ——
    这个区分不讲清楚，整节课就会被听成厂商对比。</div>
</div></section>''')

    # ── §5 范式 ──────────────────────────────────────────────────
    a('''
<section><div class="wrap">
  <div class="stn big"><span class="badge">第 5 节</span><h2>谁做决定：运行时，还是编译期</h2></div>
  <p class="lead">前面四节每一节都指向同一个分岔口，这一节把它正面画出来。
    <b>延迟是物理事实，两边都躲不掉，区别只在于谁负责把它盖住。</b></p>
  <p>GPU 的办法是<b>换一个跑</b> ——</p>''')
    a(F("g7"))
    a('''
  <p>TPU 的办法是<b>提前把班排好</b>。灰格子是真的空着，
    而且不会有任何东西自动顶上来。</p>''')
    a(F("t7"))
    a('''
  <div class="note danger"><span class="t">⭐ 全课的落点：两个「128 倍」出现在完全不同的位置</span>
    <b>第 3 节那个 128 倍在芯片内部</b>：一颗 B200 里有 592 个 Tensor Core，
    一颗 v7 里只有 4 个 MXU。<b>GPU 的协调主要发生在芯片内部。</b>
    <br><b>第 4 节那个 128 倍在芯片之间</b>：GPU 72 颗就得换协议，TPU 一路 9,216 颗。
    <b>TPU 的协调主要发生在芯片之间。</b>
    <br><br>这不是巧合，是同一个设计取向的两个侧面：
    <b>一边把复杂度收进一颗芯片里由硬件消化，另一边把复杂度摊到芯片之间由软件面对。</b>
    <br>整门课从头到尾说的都是这同一件事 ——
    <b>不是谁更强，是把同一份复杂度放在了不同的地方。</b></div>
</div></section>''')

    # ── 深入阅读 ─────────────────────────────────────────────────
    a('''
<section><div class="wrap">
  <div class="stn"><span class="badge">深入</span><h2>想再往下挖一层的同学</h2></div>
  <p class="lead">上面这十六张图都搬自两份「显微镜」文档。
    那两份把镜头拉得更近 —— <b>每一个数字都标了来源等级，查不到的地方明写「查不到」</b>。
    课上讲不完的推导链，以及作者自己犯过又改回来的两次错，都在里面。</p>
  <div class="msdeep">
    <a class="mscard" href="''' + GPU_URL + '''">
      <b>🔬 GPU 显微镜</b>
      <span>一颗 B200 从封装拆到 thread。八张图：双 die 与那道缝、SM 内部、
        四层并行、Tensor Core、块量化、内存通路、延迟隐藏、扩展边界。</span></a>
    <a class="mscard" href="''' + TPU_URL + '''">
      <b>🔬 TPU 显微镜</b>
      <span>一颗 v7 从封装拆到 lane。同一套标准反过来拆一遍，
        外加一张来源等级总表，和一整节「还没查实的」。</span></a>
  </div>
  <div class="note q"><span class="t">为什么值得花时间看那两份</span>
    它们最有价值的部分不是图，是最后两节：逐条列出每个数字来自官方文档、第三方、
    还是本文推导，以及<b>官方资料自己对不上的四处</b>
    （ICI 每轴还是每链路、pod 的两个定义、GiB 和 GB 混用、SparseCore SRAM 的口径）。
    <br><br>还有一段作者的自陈：他曾用一个<b>算得完全正确的结果</b>去支撑两个错误的前提 ——
    MXU 个数错了 2 倍、每 cell 乘加数也错了 2 倍，方向相反正好抵消，
    于是得到一个和官方完全吻合的数字，还拿这个「验证通过」否定了原本正确的版本。
    <br><b>能对上的答案不等于对的推理。</b>这条教训比里面任何一个参数都值钱。</div>
</div></section>''')
    return "\n".join(s)


# ══════════════════════════════════════════════════════════════════════
CSS = '''
/* ---- 显微镜移植图（由 topic02-port-microscope.py 注入）---- */
/* 图是按 1400 px 画的。塞进 1080 px 的版心，11 px 的字会缩到 8 px，
   讲课根本看不清 —— 所以让它们突破版心，按接近原生的宽度渲染。 */
.fwide{margin-left:50%;transform:translateX(-50%);
  width:min(1440px,calc(100vw - 32px));max-width:none;overflow-x:auto}
.fwide svg{display:block}
.msfrom{color:var(--gray);font-size:12.5px}
.msdeep{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}
@media(max-width:860px){.msdeep{grid-template-columns:1fr}}
.mscard{display:block;padding:20px 22px;border:1px solid var(--line);
  border-radius:var(--radius);background:var(--bg2);text-decoration:none;
  color:var(--ink);transition:border-color .15s,box-shadow .15s}
.mscard:hover{border-color:var(--blue);box-shadow:0 2px 12px rgba(26,115,232,.12)}
.mscard > b{display:block;font-size:16px;margin-bottom:6px}
.mscard > span{display:block;color:var(--gray);font-size:13.5px;line-height:1.65}
'''

TODO = re.compile(
    r'<section><div class="wrap">\s*<div class="note info">'
    r'<span class="t">🚧 后面还在写</span>.*?</section>', re.S)


def main():
    svg = _harvest()
    parts = {k: _split(k, v) for k, v in svg.items()}
    defs = parts["g1"][0]
    for k, (d, _) in parts.items():                 # 十六份 defs 必须完全一致，
        assert d == defs, "defs 不一致：" + k        # 否则抽成一份会悄悄改掉某张图
    F = lambda k: figure(k, parts[k][1])

    html = io.open(PAGE, encoding="utf-8").read()

    if CSS not in html:
        html = html.replace("</style>", CSS + "</style>", 1)

    block = "\n".join([
        BEG,
        '<svg width="0" height="0" style="position:absolute" aria-hidden="true">',
        defs, "</svg>",
        sections(F),
        END])

    # 幂等：第一次吃掉「🚧 后面还在写」那一段，之后拿生成块自己当锚点。
    # 不能先删再找 —— 上一版就是这么写的，删完锚点就没了，第二次跑必炸。
    old = re.compile(re.escape(BEG) + ".*?" + re.escape(END), re.S)
    if old.search(html):
        html = old.sub(lambda _: block, html, count=1)
    else:
        assert TODO.search(html), "找不到「🚧 后面还在写」那一段，页面结构变了"
        html = TODO.sub(lambda _: block, html, count=1)

    # ⚠️ 两道自检都必须跑在**写盘之前**。
    # 上一版是先写后检 —— 那样内部词一旦漏进来，等 assert 喊出声时，
    # 它已经落在这个**公开仓库**的文件里了，接下来只要有人 commit 就出去了。
    # 「检查失败」和「文件没被污染」得是同一件事，不能分成两步。
    bad = [w for w in ("Ghostfish", "34,428", "119.2 GiB", "内部设备表",
                       "cc.higcp.com", "go/") if w in html]
    assert not bad, "公开页面里出现内部词，已中止写盘：%s" % bad

    # 数**页面上真有的**，不是数采集到的。这两个数一度差了 6 ——
    # sections() 里漏掉了六张图，而这行照样报「移植 16 张图」，
    # 于是自检看着全绿、实际漏了三分之一。**报数只能报最终产物。**
    placed = set(re.findall(r'<figure class="fbox fwide" id="ms-(\w+)"', html))
    miss = sorted(set(parts) - placed)
    assert not miss, "这几张采集了却没摆上页面：%s" % miss

    io.open(PAGE, "w", encoding="utf-8").write(html)
    print("ok  topic-02.html  %s 字符  移植 %d 张图  禁字自检通过"
          % (format(len(html), ","), len(placed)))


if __name__ == "__main__":
    main()
