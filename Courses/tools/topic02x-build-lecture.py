# -*- coding: utf-8 -*-
"""专题二 外传 · **讲义（授课稿）** —— L100，20 分钟。

════════════════════════════════════════════════════════════════
⭐ 这份文件是「站在台上要说的话」，不是教材的复述
════════════════════════════════════════════════════════════════
现场原话：「一边画一边给我从头到尾的讲解，这个课程，就是你给学员讲的内容」。

所以每一讲只写四样，多一样都不要：
  · **这一讲要留下什么** ——&nbsp;一句话。没有它就别开口
  · **讲稿** ——&nbsp;口语，能照着念；配「板书」提示滚到哪张图、停在哪
  · **别讲什么** ——&nbsp;L100 二十分钟，跑偏一次就回不来了
  · **可能被问到什么**

⛔ 体例照专题一 / 专题二讲义（CSS 直接抽它们那份），别另起一套。
⛔ 分钟数是**计划值**。讲过一遍之后回来改成实测 ——&nbsp;
   ⚠️ 专题二那份栽过：同一个事实在页面和控制台有两套算法，改了一处另一处不动。
   这里只有 LEC 一个来源，页头、侧栏、合计全从它算。
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-01-lecture.html")
OUT = os.path.join(WEB, "topic-02x-lecture.html")
DECK = "topic-02x.html"


def _css():
    h = io.open(CSS_SRC, encoding="utf-8").read()
    i = h.index("<style>")
    return h[i:h.index("</style>") + len("</style>")]


LEC = []


def lec(i, no, title, mins, toc, what, land, html, grp=None, opt=False):
    """opt=True ＝ **可跳的岔路**，不计进主线时长。
    ⛔ 时长护栏（文件末尾那句 assert）只数主线 ——&nbsp;
      把可跳的也算进去，会逼着你去砍主线里真正该讲的东西。"""
    LEC.append(dict(id=i, no=no, title=title, min=mins, toc=toc, what=what,
                    land=land, html=html, grp=grp, opt=opt))


# ══════════════════════════════════════════════════════════════════
lec("X0", "开场", "这一讲只回答一个问题", 2, "开场：只回答一个问题",
    what="把「屋脊点」这把尺子接回来，并**当场声明本讲不比 benchmark**",
    land="台下知道接下来二十分钟在追什么：**什么样的活凑得够 560**",
    grp="开场 · 2 分钟", html='''
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>选芯片不是选更强的那颗，是先看你的活落在哪一边。</b>
    这句话到最后一分钟才说得出口，但从第一句话起就在为它铺路。
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">课件停在首屏，先不往下滚。</span>

    <p>这一讲二十分钟，是专题二的外传。<b>主角是一颗芯片，不是一个模型。</b></p>

    <p>先把专题二那把尺子拿回来。我们当时用<b>算力除以显存带宽</b>，
      得到一句话：<em>「每搬一个字节，这台机器本来能算多少次。」</em>
      那一讲算完 B200 和 TPU v7，结论是<b>两家旗舰几乎一模一样</b>。</p>

    <p><b>今天我们把同一把尺子多量两颗</b> —— 其中一颗，
      当场就把那个「都一样」给破了。</p>

    <span class="pause">⭐ <b>这里停一下</b>，别急着翻。
      让他们带着「哪一颗、破成什么样」这个问题往下看，
      <em>比你直接把 560 报出来强得多。</em></span>

    <p>开始之前先说清一件事，免得后面被问：
      <b>这二十分钟我不会给你们比 benchmark，一个性能数都没有。</b>
      屋脊点是个<b>结构量</b> —— 它只说这台机器的胃口有多大，
      <em>一个字都没说谁跑得快。</em></p>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="dont">
    <b>别在这儿解释 roofline。</b>画那张折线图要五分钟，
    而这一讲只需要「一根轴上四个点」。<br>
    <b>别提扩散模型。</b>它到第十五分钟才出场 —— 现在提，
    台下会以为这是一堂讲模型的课。
  </div>
''')

lec("X1", "一", "同一把尺子，量四颗芯片", 3, "一：四颗芯片，一个异类",
    what="图 X-1：H100 295 · B200 312 · v7 313 · **v6e 560**",
    land="**「算力强、显存弱」不是形容词，就是这个 560**",
    grp="硬件 · 15 分钟", html='''
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>v6e 要求你每搬一个字节，比 H100 多算一倍的次数。</b>
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到<b>图 X-1</b>，整张露出来。</span>

    <p>四颗芯片，同一道除法。<b>先看上面三条。</b>
      H100 是 295，B200 是 312，TPU v7 是 313 ——
      <em>三颗挤在一线上，这正是专题二那句「按参数表，这是同一类芯片」。</em></p>

    <span class="pause">⭐ 这时候<b>手指着那条虚线停两秒</b>，
      让他们先把「三颗重合」这件事看进去，再看第四条。</span>

    <p><b>然后是 v6e：560。</b>差不多是前三颗的两倍。</p>

    <p>这个数怎么来的，就在左边那一列：<code>918 ÷ 1.638</code>。
      跟 H100 逐项摆开 —— <b>算力 918 对 989.5，是它的 93%，基本打平；
      带宽 1,638 对 3,350，只有 49%；容量 32 GB 对 80 GB，只有 40%。</b></p>

    <p><b>分子几乎没动，分母砍掉一半，门槛就翻了一倍。</b>
      ——&nbsp;<em>「算力强、显存弱」这句话，唯一精确的说法就是这个 560。</em></p>

    <p>它对负载提了个要求：<b>每搬一个字节，你得给它凑够 560 次运算，它才不闲着。</b>
      凑不够的活，它比谁都亏；凑得够的活，它比谁都划算。
      <em>这二十分钟剩下的时间，就是在回答「什么样的活凑得够」。</em></p>
  </div>

  <h3 class="sec">可能被问到</h3>
  <div class="qa">
    <p><b>Q：H100 不是 1,979 TFLOPS 吗？</b><br>
      数据表上那个<b>带稀疏</b>，脚注写着。稠密是它的一半，989.5。
      <em>这一讲四颗全取稠密 bf16，口径一致才能比。</em></p>
    <p><b>Q：为什么不算 FP8 / Int8？</b><br>
      换精度两边都会变，比值不一定同步动。<b>先把一个口径讲透，比铺四个口径有用。</b></p>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="dont">
    <b>别在这儿讲「所以 v6e 更适合扩散」。</b>
    现在说，它就只是一句断言 —— <b>让它到第十七分钟自己掉出来。</b>
  </div>
''')

lec("X2", "二", "把 v6e 拆开", 4, "二：v6e 显微镜",
    what="图 X-2：1 颗 ＝ 1 个核 ＝ 1 个 device · 128 MiB 暂存 · 那道窄门",
    land="**560 的分母，就是图底下那根管子**",
    html='''
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>算的地方很大，进料的门很窄。</b>这两件事在同一张图上，一眼能看见。
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到<b>图 X-2</b>。先只看芯片那个大框，
      <b>别急着往下看那根红管子。</b></span>

    <p>一颗 v6e 里面就一个核。<b>说清楚这一点很重要</b> ——
      专题二讲 v7 的时候我们反复强调「所有容量除以 2 才是你的额度」，
      <em>因为 v7 一颗芯片对软件是两个 device。</em>
      <b>v6e 没有这回事：一颗 ＝ 一个核 ＝ 一个 device，不用除。</b></p>

    <p>核里面，<b>算力全压在那两块 256×256 的方阵里</b>。
      旁边是向量单元，归一化、激活、softmax 走那儿；再旁边是标量单元，发指令、发搬运。</p>

    <p>再看橙色那两块 —— <b>片上一整块 128 MiB 的暂存</b>。
      注意括号里那句：<em>它不是缓存。</em>
      放什么、什么时候放，全是编译期写死的。</p>

    <span class="pause">⭐ 这里值得多说一句：
      <b>v7 是「两个核各 64 MiB」，v6e 是「一个核独占 128 MiB」。</b>
      每颗芯片总量一样，但 v6e 不用把一张大张量切成两半。
      ——&nbsp;<em>这一点到第十五分钟讲扩散的时候会回来。</em></span>

    <span class="board">⭐ <b>现在往下滚，露出那根红色的细颈。</b></span>

    <p>这就是全图的重点。<b>芯片那么宽，通往片外的口只有这么细。</b>
      1,638 GB/s —— 整颗芯片的数据，都得从这儿过。</p>

    <p><b>560 的分母就是它。</b>分子那两块方阵没缩水，分母被砍掉一半，
      于是「每搬一个字节得算多少次才不亏」这个门槛，就抬到了别人的两倍。</p>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="dont">
    <b>右栏那五行对外规格（ICI 口、拓扑、Pod）现在只念一遍，不展开。</b>
    它们看着像减配，<b>而解释要等到第十二分钟那张图</b> ——
    提前解释会把这一节的落点冲掉。<br>
    <b>SparseCore 一句话带过</b>：「旁边还蹲着两个协处理器，一会儿单独说。」
  </div>
''')

lec("X3", "三", "把 H100 按同样的画法拆开", 3, "三：H100 显微镜",
    what="图 X-3：528 个小单元 · 多一层硬件管的 L2 · 门宽一倍",
    land="**两颗芯片在同一件事上做了相反的选择，而各自都是自洽的**",
    html='''
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>问题从来不是谁更强，是你手上的活属于哪一种。</b>
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到<b>图 X-3</b>。
      ⭐ <b>先说一句：这张跟上一张是同一套画法、同一个比例尺。</b></span>

    <p>三处不一样，一处一处看。</p>

    <p><b>第一，算力怎么摆。</b>v6e 是两块大方阵；
      H100 是 132 个 SM，每个里面 4 个 Tensor Core ——<b>一共 528 个小单元</b>。
      <em>一个是「一次吞一大块」，一个是「同时应付很多小块」。</em></p>

    <p><b>第二，片上谁做主。</b>H100 每个 SM 有 256 KB 的 L1 加共享内存，
      这层跟 v6e 的 VMEM 性质相近；<b>但它下面还有一整层 50 MB 的 L2，
      而那一层是硬件自动管的</b> —— 程序管不着它留什么、赶走什么。</p>

    <span class="pause">⭐ <b>这一条要停一下。</b>
      v6e <b>完全没有这一层</b>，JAX 源码里它的 CMEM 直接写着 0。
      ——&nbsp;<em>片上放什么，那边全部由编译期决定，没有后手。</em></span>

    <p><b>第三，门有多宽。</b>3,350 对 1,638。
      <b>图上这道门的宽度是按带宽等比画的，两张图可以直接比。</b></p>

    <span class="board">⭐ 指一下右下角<b>那个反直觉的账</b>。</span>

    <p>H100 全部片上内存加起来大约 84 MB；<b>v6e 光 VMEM 一项就是 128 MiB，
      约 134 MB。</b>——&nbsp;<em>门窄，但屋里的台面更大。这两件事是配套的，不是矛盾的。</em></p>

    <p>最后把三条收一句。<b>H100 摊成很多小单元、再压一层硬件缓存，
      是为了应付「我不知道你要跑什么」；v6e 压成两块大方阵、片上全交给编译器，
      是为了吃透「我早就知道你要跑什么」。</b></p>
    <p><em>两边各自都是自洽的。所以问题从来不是谁更强 —— 是你手上的活属于哪一种。</em></p>
  </div>

  <h3 class="sec">可能被问到</h3>
  <div class="qa">
    <p><b>Q：没有 L2 不是很吃亏吗？</b><br>
      <b>看你跑什么。</b>缓存是给「猜」用的 —— 你不知道下一次要什么，才需要它留一手。
      <em>形状固定、访问能提前排的活，那一层就是白占面积。</em>
      这正是下一段要讲的。</p>
  </div>
''')

lec("X6", "四", "一颗装得下吗", 2, "四：为什么 Pod 只有 256 颗",
    what="图 X-6：扩散在「一颗到几颗」，LLM 在「几十颗」",
    land="**那三条看着像减配的规格，是配套的，不是缩水**",
    html='''
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>v6e 不打「一个模型摊在几千颗上」那场仗 —— 不打，就不用付那个成本。</b>
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到<b>图 X-6</b>。</span>

    <p>回到刚才那三条我说「一会儿再讲」的规格：<b>只有 4 个 ICI 口、
      二维环面、一个 Pod 只有 256 颗。</b>对着 v7 的 6 个口、三维、九千多颗，
      <em>看着像减配。</em></p>

    <p><b>把模型的体积摆出来就说得通了。</b>
      这条轴上是 bf16 的权重体积。扩散这一族 —— SDXL 7 GB、
      HunyuanVideo 16.6 GB、FLUX 24 GB、Wan2.1 28 GB ——
      <b>都在一颗到两颗的量级上。</b></p>

    <span class="pause">⛔ <b>这里千万别说「所以都装得进一颗」。</b>
      指着 Wan2.1 那行：28 GB 已经贴着 32 GB 的边，<b>激活只剩 4 GB 余量</b>；
      再指 Wan2.2：总权重 54 GB，<b>直接超线，两颗起步。</b>
      ——&nbsp;<em>一张结论过于整齐的图，台下第一时间就会怀疑它。
      主动把不整齐的地方点出来，反而是加分的。</em></span>

    <p>再看下面两条灰的：Qwen3.5-397B 是 794 GB，DeepSeek-V3 是 1,342 GB。
      <b>差了一个半到两个数量级。</b></p>

    <p>所以那三条不是减配，是<b>配套</b>。<em>不打那场仗，就不用付那个成本。</em></p>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="dont">
    <b>别展开 MoE 的激活参数是怎么回事。</b>Wan2.2 那行标了「总 27B / 每步激活 14B」，
    <b>念一遍就走</b> —— 展开是十分钟，而这一节只有三分钟。
  </div>
''')

lec("X78", "五", "补两样：SparseCore 与二维环面", 2, "五：补两样（可跳）",
    what="图 X-7、X-8：芯片上还有什么、4 个口怎么连成 256 颗",
    land="**SparseCore 有两个职责，跨卡的时候一直在干活**", opt=True,
    html='''
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>SparseCore 不只是「给推荐系统用的」——&nbsp;它还负责把集合通信从 TensorCore 手里接过去。</b>
    <em>时间紧可以整段跳过，但这一条别讲错。</em>
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到<b>图 X-7</b>。</span>

    <p>芯片上除了那个大核，还蹲着两个 SparseCore。<b>它们有两个职责。</b></p>

    <p><b>第一个</b>，架构文档里写的：加速稀疏运算，主用途是重 embedding 的推荐模型。
      <em>——&nbsp;扩散这条链路上没有大表，所以这条路它不怎么走。</em></p>

    <p><b>第二个，也是这一讲真正相关的：集合通信卸载。</b>
      官方性能指南原话 —— <em>把 all-reduce 这类集合通信卸载到 SparseCore，
      这些操作不占 MXU，可以在它上面执行，同时 TensorCore 继续算。</em>
      All-Gather 和 Reduce-Scatter 也能卸，由 XLA flag 开关。</p>

    <span class="pause">⭐ <b>这一句要说清楚：只要跨卡，它就在干活。</b>
      <em>扩散模型一旦切开 —— 模型放不下、要切序列、或者想缩短单张出图时间 ——
      all-gather 就一大堆。</em>
      ——&nbsp;<b>而对一颗「算力强、显存弱」的芯片，把通信从关键路径上挪开是格外划算的。</b></span>

    <span class="board">滚到<b>图 X-8</b>。</span>

    <p>4 个 ICI 口，连上下左右，<b>边缘绕回对面 —— 所以叫环面，不叫网格。</b>
      16 乘 16 就是一个 Pod 的 256 颗。</p>

    <p><b>最远 16 跳</b>，能当场推：环面每一维最多绕半圈，16 除以 2 是 8，两维相加 16。
      <em>对照 v7 的 4×4×4，每维 2 跳，三维相加 6 跳。</em></p>

    <p><b>走不走这条路，取决于你切不切模型。</b>
      纯数据并行 —— 一颗一张图 —— 通信接近零，跳数无所谓；
      <b>一旦切开，拓扑就开始要钱了。</b></p>

    <p><em>所以这一张的落点不是「16 跳没关系」，是：
      <b>v6e 把互联做到「够扩散这一族用」的档位，而不是「够一个模型摊在几千颗上」的档位。
      够用，不是不用。</b></em></p>
  </div>

  <h3 class="sec">⛔ 这里我讲错过一次，值得原样告诉学员</h3>
  <div class="dont">
    <b>这两张图的初版我都写过头了。</b><br>
    X-7 写的是「扩散模型用不到 SparseCore」——&nbsp;<b>我把架构文档里那句
    「a primary use case（主用途）是推荐模型」当成了用途的全集。</b>
    <em>「a primary use case」这种措辞本身就在告诉你「还有别的」。</em><br>
    X-8 写的是「几乎没人走这条路」——&nbsp;<b>那只说对了纯数据并行那一种用法。</b><br>
    ⭐ <b>两次都是同一个形状：把「一种常见情形」讲成了「唯一情形」，
    而且都给它配了一条看起来很像样的推导链。</b>
    <em>链子是对的，前提漏了一半 —— 于是整条链推向了错的地方。</em>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="dont">
    <b>SparseCore 的内部数据流一个字都不要讲。</b>gather / scatter 怎么排是半小时的东西。<br>
    ⚠️ <b>集合通信卸载那批材料主要是 v7x / 第四代 SparseCore 的语境</b>，
    Trillium 是第三代 ——&nbsp;<b>被问到 v6e 上成熟度如何，就答「我们没实测」。</b>
  </div>
''')

lec("X5", "六", "扩散一步在干什么", 3, "六：扩散是哪一种活",
    what="图 X-5：没有 KV cache · 形状不变 · 同一段跑五十遍",
    land="**编译期能知道的，扩散全都提前告诉你了**",
    grp="模型 · 5 分钟", html='''
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>扩散是「我早就知道你要跑什么」的极端情形。</b>
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到<b>图 X-5</b>，两栏一起露出来。</span>

    <p><b>先看左边，自回归 LLM。</b>一个 token 一个 token 往外蹦，
      每一步都要带上前面所有的 KV —— <em>那条红条一直在长。</em>
      <b>形状每一步都在变，编译好的那一份，下一步就不合用了。</b></p>

    <p><b>再看右边，扩散。</b>整张图一起动，走五十步。
      每一步只吃上一步的结果，<b>传过去的东西大小不变</b>。
      ——&nbsp;<em>没有 KV cache 这回事，它根本没有「历史」这个概念。</em></p>

    <span class="board">指下面那三行对照表。</span>

    <p>三条：<b>步与步之间传什么</b> —— 一边是越来越长的 KV，一边是大小不变的 latent。
      <b>张量形状</b> —— 一边每步都变，一边从头到尾一个形状。
      <b>同一段计算跑几次</b> —— 一边每步都得重排，一边原样跑五十遍。</p>

    <span class="pause">⭐ <b>这里把它接回第九分钟那张 H100 图。</b>
      我们说过 v6e 片上没有硬件缓存兜底，全靠编译期。
      <em>那个设计有个前提：形状得固定、计划得排得出来。</em>
      ——&nbsp;<b>而扩散把这个前提喂得满满的。</b></span>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="dont">
    <b>不要讲加噪去噪、不要讲 latent、不要讲 VAE、不要讲 DiT。</b>
    台下要的是「它对硬件提了什么要求」，不是「它为什么能生成图片」。<br>
    <em>那一半是专题十一，一小时的深潜。这里三分钟，只讲三个后果。</em>
  </div>
''')

lec("X4", "七", "把活放回那两条线上", 2, "七：收口",
    what="图 X-4：decode 在左边，prefill 和扩散甩到右边两个数量级",
    land="**不是它跑扩散更快，是扩散把它的短板挡在了瓶颈之外**",
    html='''
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>短板不参与，长板打平 —— 这就是「合适」的全部含义。</b>
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到<b>图 X-4</b>。</span>

    <p>一条规则就够：<b>算术强度约等于，同一份权重被多少个「位置」共用。</b>
      权重从 HBM 搬上来是固定开销，服务的位置越多，这笔搬运摊得越薄。</p>

    <p><b>看两条红的。</b>decode batch 1 是 1，batch 64 是 64 ——
      <em>离那两条线还差着两个数量级，两颗芯片都在饿着。</em></p>

    <p><b>再看三条右边的。</b>prefill 8K 是 8,192，文生图约 16,000，
      视频再多一到两个数量级。<b>全都甩开那两条线一个半数量级以上。</b></p>

    <span class="pause">⭐ <b>这里是全讲的落点，慢一点说。</b></span>

    <p><b>所以「v6e 适合扩散」的精确说法是这个 —— 不是它跑扩散更快。</b>
      是扩散把它的短板<b>挡在了瓶颈之外</b>：强度上万，那根窄管子根本没成为瓶颈。
      <b>于是只剩下没被挡住的那一项在起作用：算力 918 对 989.5，是 H100 的 93%。</b></p>

    <p><em>短板不参与，长板打平。这就是「合适」的全部含义。</em></p>

    <p><b>反过来也成立。</b>batch 小的 decode 落在最左边，那里比的全是带宽，
      而 v6e 只有 H100 的 49% —— <b>同一颗芯片，在那种活上就是最吃亏的。</b></p>

    <p>还有一条顺带的：<b>prefill 重、decode 轻的任务，跟扩散落在轴上的同一边。</b>
      <em>道理一模一样，不用重讲。</em></p>
  </div>

  <h3 class="sec">可能被问到</h3>
  <div class="qa">
    <p><b>Q：那实际跑起来到底谁快？</b><br>
      <b>这一讲不回答这个，而且是故意的。</b>快慢要实测，实测要连口径一起给 ——
      同一个模型、同一个精度、同一套 kernel 成熟度。<em>结构匹配和性能是两件事，
      需要的证据不一样。</em></p>
  </div>
''')

lec("X9", "八", "落点", 1, "八：落点",
    what="一句话收",
    land="**从布局图上就能提前看出来，不用等跑完**",
    grp="收尾 · 1 分钟", html='''
  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到最后一屏。</span>

    <p><b>选芯片不是选「更强的那颗」，是先看你的活落在那根轴的哪一边。</b></p>

    <p>落右边 —— 扩散、prefill、大 batch —— 比的是算力，v6e 打平。<br>
      落左边 —— 小 batch 的 decode —— 比的是带宽，v6e 吃亏。</p>

    <p><b>而这两件事，从两颗芯片的布局图上就能提前看出来。</b>
      <em>不用等跑完。</em></p>

    <span class="pause">⭐ <b>说完这句就停。</b>不要再补一句总结 ——
      这一讲二十分钟只有一个落点，多说一句就摊薄了。</span>
  </div>
''')


def _toc():
    out = []
    for L in LEC:
        if L["grp"]:
            out.append('<li class="grp">%s</li>' % L["grp"])
        out.append('<li><a href="#%s"><span class="n">%s</span>%s'
                   '<span class="m">%d′%s</span></a></li>'
                   % (L["id"], L["no"], L["toc"], L["min"],
                      " 可跳" if L["opt"] else ""))
    return "\n".join(out)


def _md(t):
    """把 what / land 里顺手写的 **粗体** 转成真的 <b>。
    ⛔ 2026-09-09 栽过：这是 HTML 不是 markdown，星号会**原样印在页面上**。
      ⭐ 判据：**任何「顺手用 markdown 语法」的字段都得有一道转换**，
        否则它长得像格式、渲染出来是乱码字符 ——&nbsp;而且不报错。"""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


def _secs():
    out = []
    for L in LEC:
        out.append(
            '<section class="lec" id="%s">\n'
            '  <h2><span class="no">%s</span>　%s　<span class="min">%d 分钟</span></h2>\n'
            '  <div class="meta2"><b>讲什么</b>：%s<br><b>落点</b>：%s</div>\n'
            '%s\n</section>' % (L["id"], L["no"], L["title"], L["min"],
                                _md(L["what"]), _md(L["land"]), L["html"]))
    return "\n".join(out)


def main():
    # ⭐ 只数主线。可跳的那一节单独报，别混进来。
    total = sum(L["min"] for L in LEC if not L["opt"])
    optm = sum(L["min"] for L in LEC if L["opt"])
    html = ('''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>专题二 外传 · 讲义（授课稿）</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#129408;</text></svg>">
<meta property="og:title" content="v6e 与扩散模型 · L100 讲义">
<meta property="og:description" content="老师的草稿：逐字讲稿、该指哪张图、别讲什么、可能被问到什么。二十分钟。">
<meta name="theme-color" content="#5f6368">
''' + _css() + '''
<style>
/* ⭐ 只补三条，其余全走专题一讲义那份 CSS —— 别在这儿另立一套观感。 */
.lec h2 .no{display:inline-block;min-width:2.4em;color:#5f6368;font-weight:600}
.lec h2 .min{font-size:14px;font-weight:400;color:#80868b}
.lec .meta2{font-size:14px;color:#5f6368;line-height:1.9;margin:6px 0 14px}
</style>
</head>
<body>
<!-- ⛔ 本文件由 Courses/tools/topic02x-build-lecture.py 生成，不要手改。
     加一讲 ＝ 往那个脚本的 LEC 里加一条，侧栏与合计会自动跟上。 -->
<div class="top"><div class="in">
  <span class="who">专题二 · <b>外传：v6e 与扩散模型</b>　<span style="opacity:.6">L100</span></span>
  <nav class="tabs">
    <a href="''' + DECK + '''">课件</a>
    <a href="topic-02x-lecture.html" class="on">讲义</a>
    <a href="topic-02.html">专题二 L200</a>
  </nav>
  <span class="meta">主线 ''' + str(total) + ''' 分钟　可跳 ''' + str(optm) + ''' 分钟</span>
</div></div>
<div class="shell">
<aside class="side">
  <h4>讲次</h4>
  <ol id="toc">
''' + _toc() + '''
  </ol>
</aside>
<main>
''' + _secs() + '''
</main>
</div>
</body>
</html>
''')

    import sys
    sys.path.insert(0, os.path.join(HERE, "tpu-micro"))
    from gate import lint_public
    bad = lint_public(html)
    assert not bad, "公开页面里出现内部词，已中止写盘：%s" % bad
    # 侧栏每一条都要落到真 section 上 —— 锚点写错是静默的
    ids = set(re.findall(r'<section class="lec" id="(\w+)"', html))
    hrefs = set(re.findall(r'<li><a href="#(\w+)"', html))
    assert hrefs == ids, "侧栏和正文对不上：侧栏多 %s，正文多 %s" % (
        sorted(hrefs - ids), sorted(ids - hrefs))
    assert total == 20, "主线合计 %d 分钟 —— 目标 20，超了就砍，少了就补" % total

    io.open(OUT, "w", encoding="utf-8").write(html)
    print("ok  topic-02x-lecture.html  %s 字符　%d 讲　主线 %d 分钟"
          "（另有可跳 %d 分钟）"
          % (format(len(html), ","), len(LEC), total, optm))


if __name__ == "__main__":
    main()
