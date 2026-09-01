# -*- coding: utf-8 -*-
"""专题二 L200 · 教师讲义（授课稿）生成器。

════════════════════════════════════════════════════════════════
这份东西是什么
════════════════════════════════════════════════════════════════
课件（topic-02-L200.html）是**给学员看的**；这一份是**老师的草稿** ——
逐字讲稿、该滚到哪张图、真会被问到什么、以及最要紧的「别讲什么」。

版式和 CSS 整套沿用专题一的讲义（topic-01-lecture.html）：`_css()` 直接
从那个文件里把 <style> 块原样抽过来。**不要在这里另写一份 CSS** ——
两份讲义长得不一样，翻起来会以为是两套材料。

════════════════════════════════════════════════════════════════
为什么用脚本生成，而不是像专题一那样手写 HTML
════════════════════════════════════════════════════════════════
因为这份讲义是**一节一节攒出来的**（讲一节、打磨一节、记一节），
而每加一节就有三样东西必须跟着改：

  ① 左边侧栏那个讲次列表
  ② 「已写 N 讲」的计数
  ③ 未写的那几讲要显示成灰的、不可点

这三样全都是**从讲次表推出来的**，不是独立信息。手写就意味着每次都要
记得同步三个地方 —— 而只要漏一次，侧栏就开始说谎，且不会有任何报错。
（专题一那份的顶栏计数就是靠一段 JS 现算的，正是同一个动机。）

所以这里把讲次表 `LEC` 立成唯一来源，侧栏和计数全从它生成。
**加一讲 = 往 LEC 里加一个 dict，别的都不用动。**

════════════════════════════════════════════════════════════════
写讲稿的三条规矩
════════════════════════════════════════════════════════════════
① **讲稿是说出来的话，不是写出来的话。** 短句、口语、能照着念。
   `<em>` 是黄底强调 —— 只给「必须原样说出口」的那几句，滥用就失效了。
② **`别讲什么` 比 `讲稿` 更值钱。** 一小时讲不完的原因从来不是讲得慢，
   是跑偏。每一讲都要写清楚这一节最容易被带走的方向。
③ **`可能被问到` 里答不上来的，就写「这个我不确定」。** 讲义里编一个
   答案，等于在现场埋一颗雷 —— 那时候你会照着念。
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-01-lecture.html")
OUT = os.path.join(WEB, "topic-02-L200-lecture.html")

DECK = "topic-02-L200.html"        # 学员看的课件（L200）
FULL = "topic-02.html"             # 完整版 L300


def _css():
    """把专题一讲义的整个 <style> 块原样抽过来（含 :root 变量）。"""
    h = io.open(CSS_SRC, encoding="utf-8").read()
    i = h.index("<style>")
    return h[i:h.index("</style>") + len("</style>")]


# ══════════════════════════════════════════════════════════════════════
# 讲次表 —— 唯一来源
# ══════════════════════════════════════════════════════════════════════
# 字段：
#   id     锚点，跟课件的节号对齐（课件 §3 → 讲义 L3），方便边讲边对
#   no     侧栏上显示的编号
#   title  讲次标题
#   min    分钟数。⚠️ 这是**计划**不是实测，讲过之后回来改成真实值
#   toc    侧栏里的短标题（长了会换行，侧栏很窄）
#   grp    非空则在它前面插一条分组横线
#   html   正文。**留空 = 这一讲还没写**，会自动渲染成 todo 占位块，
#          侧栏那条也会自动变灰（靠页面底部那段 JS 认 .todo-box）
LEC = []


def lec(id, no, title, min, toc, html="", grp=""):
    LEC.append(dict(id=id, no=no, title=title, min=min, toc=toc, html=html, grp=grp))


# ──────────────────────────────────────────────────────────────────────
# 第 0 讲
# ──────────────────────────────────────────────────────────────────────
lec("L0", "00", "开场：两条出身，一条主线", 4, "开场：两条出身，一条主线",
    grp="开场 · 4 分钟", html='''
  <div class="goal">
    <span class="k">这一讲要留下什么</span>
    <b>两边的差别不是十几处，是一处。</b>
    那一处是「把一块数据从 HBM 搬进片上，<b>谁来安排</b>」——&nbsp;
    一边交给运行时，一边交给编译期。整节课其余部分都是在验证这一句。
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">课件停在首屏，先不往下滚。</span>

    <p>好，我们开始。这一讲叫「TPU 与 GPU」，但我先说清楚，
    <em>这门课不给你念参数表。</em>参数表恰恰是我们第一个要拆穿的东西。</p>

    <p>整节课就<b>两个问题，一条路线</b>。第一个问题，这两块硬件在硬件层面
    到底哪儿不一样。第二个问题，我们拿同一个算子，从头到尾跑一遍，
    看那些不一样<b>分别在哪一步冒出来</b>。那个算子就是 FlashAttention。</p>

    <span class="board">往下滚一屏，停在第 0 节那张地图（figC）。
      这张图后面会反复回来指，先让它在屏幕上多待一会儿。</span>

    <p>现在看第一张图。这张图有点特殊 ——&nbsp;
    <b>它既是我们的目录，也是全课唯一的前提。</b></p>

    <p><b>先看上半部分。</b>左边 GPU，出身是图形处理器。它从第一天起要伺候的，
    就是一大堆<b>互不相干的小任务</b> ——&nbsp;形状不知道，访问模式不知道，
    控制流也不知道。所以它的芯片面积，<b>大半花在「应付各种情况」上</b>。
    cache 层次、warp 调度器、海量并发线程，全是为了这个。</p>

    <p>右边 TPU，出身是<b>为矩阵乘定做的专用芯片</b>。它从第一天就只服务神经网络。
    形状是规则的，访问是可预测的，控制流是静态的。所以省下来的面积，
    <b>全给了计算单元</b>。没有 cache，布局在编译期就定死，换来一个巨大的 MXU。</p>

    <span class="pause">停一拍。下面这两句是这一讲要背下来的。</span>

    <p><em>一个是「不知道你要跑什么，所以处处留一手」；
    另一个是「早就知道你要跑什么，所以一手都不留」。</em></p>

    <p><b>再看下半部分。</b>那是我们这一小时要走的六站。出发前先算一笔账，
    中间四站往下拆，最后一站拿到两个数看能不能比。</p>

    <p>中间有一个格子是<b>红的、加粗描边</b> ——&nbsp;那是<b>第 2 节，分岔点</b>。
    我特意用颜色把它顶出来了。</p>

    <span class="board">手指或鼠标点一下那个红格子，再往下划到最底部那个红框。
      这两处是同一件事的两个位置，要连着指。</span>

    <p>因为这门课真正的结论，在最底下那个红框里。<em>两边的差别，不是十几处，是一处。</em>
    就是把一块数据从 HBM 搬进片上的时候，<b>谁来安排</b>。
    GPU 那边有 cache 在运行时替你猜；TPU 那边没有，编译期就排死了。</p>

    <p>形状要对齐到 8 乘 128、一条指令吃多大、kernel 谁来写、
    连一台机器要付什么代价 ——&nbsp;<b>全都是从这一处长出来的</b>。</p>

    <span class="pause">这里语速放慢。这句话是整节课的论点，
      后面每一节都在给它找证据。说太快，学员会当成一句口号听过去。</span>

    <p>开课前还有一句话我要说死：<em>这门课不比谁强。</em>
    两条出身<b>各自都是当年的正确决定，代价也各自都还在</b>。
    所以整节课你不会听到「谁更好」这种句子，
    只会听到<b>「这件事在这一边由谁决定」</b>。</p>

    <p>好，地图立完了。下面从第 1 节开始 ——&nbsp;先架量具。</p>
  </div>

  <h3 class="sec">可能被问到</h3>
  <div class="qa">
    <details><summary>「差别只有一处」是不是说得太绝对了？明明处处都不一样。</summary>
      <div class="a">
        <p><b>现象上确实是十几处，「一处」说的是因果上的一处。</b>
        cache、形状对齐、kernel 谁写、互联怎么连 ——&nbsp;这些都是真的不一样，
        但它们<b>不是并列的十几个事实</b>，是同一个决定往下长出来的十几个后果。</p>
        <p>这句话在开场是<b>一个论点，不是一个已证的结论</b>。
        可以坦白说：「你现在完全可以不信，第 3 节和第 5 节我会分别验证它。」
        ——&nbsp;<b>这样比硬撑更有说服力</b>，而且给了他们一个听下去的理由。</p>
      </div></details>

    <details><summary>TPU 真的一点 cache 都没有？那 VMEM 是什么？</summary>
      <div class="a">
        <p><b>准确的说法是：TPU 没有「硬件自动管的缓存」那一层。</b>
        片上存储是有的，而且不小 ——&nbsp;但它是<b>软件管理的 scratchpad</b>：
        什么时候搬进来、放哪儿、什么时候倒掉，全部由编译器在编译期写死，
        运行时没有任何东西会替你做决定。</p>
        <p>这个区别正是本课的主线，所以<b>被问到是好事</b>，顺势就能接：
        「对，缺的不是容量，是『谁做决定』。」<br>
        ⚠️ 但<b>别在开场展开讲 VMEM 有多大、分几块</b> ——&nbsp;第 3 节有专门的图。</p>
      </div></details>

    <details><summary>为什么拿 FlashAttention 当主角？换个 GEMM 或者跑一整个模型不行吗？</summary>
      <div class="a">
        <p><b>因为它同时满足两个很难同时满足的条件。</b></p>
        <p>一是<b>单卡就跑得完</b> ——&nbsp;不用把卡间通信拽进来，
        故事能一条线走到底（这也是第 4 节被画成岔路的原因）。</p>
        <p>二是它<b>一次压到四件事</b>：访存、形状、片上容量、kernel 怎么写。
        纯 GEMM 太干净，压不到后面三件；跑一整个模型又太杂，
        每一步都在换话题，最后什么都没讲透。</p>
      </div></details>

    <details><summary>第 4 节为什么画成岔路？卡间通信不重要吗？</summary>
      <div class="a">
        <p><b>重要，但它不在这条主线上。</b>判据很硬：
        第 4 节的正文里 FlashAttention 出现 <b>0 次</b>（grep 过的）——&nbsp;
        它单卡就跑完了，<b>根本撞不到卡间</b>。</p>
        <p>与其让人读到那儿觉得「怎么突然跑题了」，不如<b>在地图上就标成岔路</b>。
        L200 里这一节是折叠的，主线时间不够可以整节跳过，
        对第 5 节完全没有影响。</p>
      </div></details>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="warn">
    <span class="k">这一讲最容易跑偏的四个方向</span>
    <b>① 别在开场比参数。</b> 算力、带宽、SRAM 那三组数是<b>第 1 节的包袱</b> ——&nbsp;
    「两边几乎一样」这个意外，得等到量具架好了才炸得响。开场先说了，第 1 节就白讲了。<br>
    <b>② 别讲两家的历史年表。</b> 出身只需要两句话（「不知道你要跑什么」／「早就知道」），
    多一句都是在花时间。<br>
    <b>③ 别在这儿解释 MXU 是什么、warp 是什么。</b> 第 3 节有八张显微镜图专门拆。
    <b>开场做术语扫盲是最贵的一种跑偏</b> ——&nbsp;一开口就是十分钟。<br>
    <b>④ 别说「谁更好」。</b> 一旦有人把话题带到选型，直接用课件上那句挡回去：
    「这门课只回答『这件事由谁决定』。」
  </div>
  <div class="tip">
    <span class="k">节奏提示</span>
    这一讲<b>四分钟必须走完</b>，它是地图不是内容。
    真正让人坐直的时刻在第 1 节末尾（<b>312.6 对 312.5</b>），
    <b>早点走到那儿。</b><br>
    如果开场发现台下背景很杂，<b>宁可砍掉出身那两段的细节，也要把最底下那个红框念完</b> ——&nbsp;
    没有那句话，后面每一节都会变成孤立的知识点。
  </div>
''')

# ──────────────────────────────────────────────────────────────────────
# 以下待写。⚠️ 分钟数是**计划值**，讲过之后回来改成实测。
# 主线合计 59 分钟（第 4 节 3 分钟是可跳的岔路，不计入）。
# ──────────────────────────────────────────────────────────────────────
lec("L1", "01", "先架量具：两边的胃口一样大", 4, "先架量具：胃口一样大",
    grp="立尺子 · 4 分钟", html='''
  <div class="goal">
    <span class="k">这一讲要留下什么</span>
    <b>312。</b>每从 HBM 搬 1 个字节，硬件配套供得起约 312 次 BF16 运算。
    <b>这一节不给任何结论，只交一把尺子</b> ——&nbsp;后面每个算子都要拿它量一下。
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到第 1 节。这一节只有两张图，节奏要快。</span>

    <p>上一段我说了「差别只有一处」。<em>但在比任何东西之前，得先有一把尺子。</em>
    这一节就干这一件事 ——&nbsp;<b>架尺子，不下结论</b>。</p>

    <p>两边的官方规格表都给了两个数：<b>每秒能算多少次</b>，和<b>每秒能搬多少字节</b>。
    我们把它们除一下。这个比值的意思很实在 ——&nbsp;
    <b>每搬一个字节，这块硬件配套能算几次</b>。</p>

    <span class="board">停在那张两栏对照图（fig1-4）。<b>不要急着往下滚</b>，
      让数字在屏幕上待一会儿。</span>

    <p>左边 TPU v7，一颗芯片 <b>2,307</b> TFLOPS，带宽 <b>7.38</b> TB 每秒。
    除出来 <b>312.6</b>。</p>

    <p>右边 GB200，NVL72 里那一颗，<b>2,500</b> TFLOPS 配 <b>8.0</b> TB 每秒。
    除出来 <b>312.5</b>。</p>

    <span class="pause">这里停死。数字念完不要马上解释，
      让台下自己反应过来这两个数是一样的。<b>这是第 1 节唯一的爆点。</b></span>

    <p><em>两家公司，两套架构，两种设计哲学 —— 比值撞在小数点后一位上。</em>
    换成 FP8 也一样，625.2 对 625.0。相对差都是 <b>0.03%</b>。</p>

    <p>但接下来这句<b>必须跟着说，不能省</b>：<em>「都落在 312」是真的，
    「为什么都落在 312」是我的推断，没有任何一家这么解释过。</em>
    而且它<b>挑型号</b> ——&nbsp;换成 HGX 板上那颗 B200，就是 281.3，差 11%。</p>

    <p>所以我们后面每次拿 312 当判据，用的都是<b>那个带限定的版本</b>。
    <b>往外引的时候不能把限定甩掉只留数字。</b></p>

    <span class="board">往下滚到三格那张图（fig1-5）。</span>

    <p>比值一样，还有一种可能：<b>分子分母凑巧成了比例</b>。
    所以我们把两边完整的九行摊开再验一遍。</p>

    <p>但九行并排有个坏处 ——&nbsp;<b>每一行看起来一样重</b>。
    所以这张图先替我们按「值不值得花时间」压成了三格。</p>

    <p>中间那格，<b>算力差 8%，带宽差 8%，片上 SRAM 差 6%</b>。
    ——&nbsp;<em>不只是比值一样，分子和分母各自也几乎一样。</em>
    凑巧成比例这条路，堵死了。</p>

    <span class="pause">这句是这一节真正的收获，
      <b>它要两张图连起来看才成立</b>，单看任何一张都得不到。</span>

    <p>左边那一格才是我们要讲的东西：<b>各缺对方整整一层</b>。
    TPU 没有硬件自动管的片上缓存那一层，GPU 没有可编程的专用协处理器那一层。
    <b>缺的不是容量，是「谁做决定」。</b></p>

    <p>右边那格的完整九行表折在图下面，<b>那是回查用的手册，现在不展开</b>。</p>

    <p>好，尺子架完了，两个刻度：<b>312 这个数</b>，和<b>三格这个筛子</b>。
    下面第 2 节，我们去看那个分岔点长什么样。</p>
  </div>

  <h3 class="sec">可能被问到</h3>
  <div class="qa">
    <details><summary>这个 312 是 BF16 的，跑 FP8 呢？判据要换吗？</summary>
      <div class="a">
        <p><b>要换，换成 625。</b>图上两行都算了：TPU 625.2，GB200 625.0。</p>
        <p>道理很简单：<b>算力翻倍，带宽没变</b>，所以比值也翻倍 ——&nbsp;
        而且两边都翻倍，所以「撞在一起」这件事在 FP8 上依然成立。</p>
        <p><b>判据要按你实际跑的精度取。</b>拿 BF16 的 312 去判一个 FP8 的算子，
        会把本来算力受限的判成带宽受限。</p>
      </div></details>

    <details><summary>为什么 TPU 那栏写「每 chip」？我看日志里的数对不上。</summary>
      <div class="a">
        <p><b>因为 v7 是一颗 chip 里有两个 device，而框架日志一律按 device 报。</b>
        所以看日志换算：<b>per-chip = 日志上的 per-device × 2</b>。</p>
        <p>这里两边都按「一颗封装」的口径比 ——&nbsp;TPU 一个 chip 对 GB200 一颗 GPU，
        这样才是同一个东西在比。<b>口径写在图里那一行小字上，讲的时候可以指一下。</b></p>
        <p>⚠️ 这个 1 比 2 还会影响 batch：<code>per_device_batch_size</code> 里的
        「device」在 v7 上是<b>半颗芯片</b>。但<b>这里不要展开</b>，是另一门课的事。</p>
      </div></details>

    <details><summary>算力除带宽这个数，有正式名字吗？</summary>
      <div class="a">
        <p>有 ——&nbsp;它是 <b>Roofline 模型里屋脊点的横坐标</b>，也叫机器平衡点。
        算子那一头对应的量叫<b>算术强度</b>，单位一样，所以能直接比大小。</p>
        <p><b>提一句名字就走，别画 Roofline 图。</b>画那张图要五分钟，
        而第 2 节那张访存图讲同一件事讲得更具体。</p>
      </div></details>

    <details><summary>两家参数为什么会这么像？是互相抄的吗？</summary>
      <div class="a">
        <p><b>这个我不确定，没有任何一家公开解释过。</b>直说就行。</p>
        <p>可以给一个<b>标明是推断</b>的说法：两边都在拿 HBM 带宽去配矩阵算力，
        用的还是同一代 HBM，比值撞在一起并不奇怪。<b>但这是我倒推的，不是出处。</b></p>
        <p>⚠️ 这里<b>特别容易顺口讲成因果</b> ——&nbsp;「两边被同一批负载钉在同一个比值上」
        这种话听着很漂亮，但它是从两个数字反推的，而且换个 SKU 就不成立了（281.3）。</p>
      </div></details>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="warn">
    <span class="k">这一讲最容易跑偏的四个方向</span>
    <b>① 别展开 Roofline。</b> 被问到就报个名字，<b>不要画屋脊图</b> ——&nbsp;
    五分钟起步，而且第 2 节有更具体的讲法。<br>
    <b>② 别把九行表摊开逐行念。</b> 它在课件里是<b>折起来的</b>，就是为了不念。
    主线记三格，九行是回查用的。<br>
    <b>③ 别在 SparseCore、CMEM 上停留。</b> 第 1 节只需要说「各缺对方一层」，
    <b>那一层具体是什么，第 3 节有专门的图</b>。<br>
    <b>④ 别把 312 讲成「TPU 和 GPU 是一样的」。</b> 这一节证明的是
    <b>「参数表分不出它们」</b>，不是「它们一样」 ——&nbsp;这两句差得很远，
    而且下一节就要打脸第一句。
  </div>
  <div class="tip">
    <span class="k">节奏提示</span>
    <b>四分钟，两张图。</b>时间几乎全砸在第一张的那个停顿上 ——&nbsp;
    <b>312.6 对 312.5 念完之后，留住三十秒不说话</b>，让台下自己发现。
    这是全课第一个「坐直」的时刻，赶过去就没有了。<br>
    第二张图相对快：只指中间那格的三个百分比，和左边那格的「两个空」，
    <b>右边那格不用讲</b>（它是第一张图的复述）。
  </div>
''')
lec("L2", "02", "⭐ 分岔在这里：一个有 cache，一个没有", 7, "⭐ 分岔在这里",
    grp="分岔点 · 7 分钟")
lec("L3", "03", "拆开看：形状对不对得齐，以及并排走完全程", 24,
    "⭐⭐ 拆开看 + FlashAttention 走一遍", grp="拆开看 · 24 分钟（全课最重）")
lec("L4", "04", "岔路：一张卡装不下的时候", 3, "岔路：一张卡装不下（可跳）",
    grp="岔路 · 3 分钟（时间不够整节跳过）")
lec("L5", "05", "⭐ 分岔的本质：运行时，还是编译期", 9, "⭐ 分岔的本质",
    grp="收束 · 9 分钟")
lec("L6", "06", "跑完之后：拿到两个数，能不能比", 7, "跑完之后：能不能比",
    grp="落地 · 11 分钟")
lec("L9", "09", "那到底各自擅长什么", 4, "收尾：各自擅长什么")


# ══════════════════════════════════════════════════════════════════════
# 组装
# ══════════════════════════════════════════════════════════════════════
TODO = '''
  <div class="todo-box">
    <b>还没写。</b>这一讲要等实际讲过一遍之后再落笔 ——&nbsp;
    讲义的价值在于记下<b>现场发现讲不通的地方</b>，
    照着课件先编一份出来，等于把课件抄了一遍，没有用。<br>
    课件对应的是 <a href="%s#%s">L200 第 %s 节</a>。
  </div>
'''


def _toc():
    out = []
    for L in LEC:
        if L["grp"]:
            out.append('    <div class="grp">%s</div>' % L["grp"])
        out.append('    <li><a href="#%s"><i>%s</i><span>%s</span></a></li>'
                   % (L["id"], L["no"], L["toc"]))
    return "\n".join(out)


def _sections():
    out = []
    for L in LEC:
        sec = L["id"][1:]                       # L3 → 3，跟课件的 §3 对齐
        body = L["html"] or TODO % (DECK, "s" + sec, sec)
        out.append('''
<section class="lec" id="%s">
  <div class="lh"><span class="no">%s</span><h2>%s</h2>
    <span class="t">⏱ %s 分钟</span></div>
%s</section>''' % (L["id"], L["no"], L["title"], L["min"], body))
    return "\n".join(out)


def main():
    html = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>专题二 L200 · 讲义（授课稿）</title>
<!-- 🦀 CloseCrab 标识。刻意不放图片文件 —— 内联一个 SVG，让每台机器
     用自己的 emoji 字体现画；烤成 PNG 会把某一家的字形固定下来。 -->
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#129408;</text></svg>">
<link rel="apple-touch-icon" href="../../apple-touch-icon.png">
<meta property="og:title" content="TPU 与 GPU · L200 讲义">
<meta property="og:description" content="老师的草稿：逐字讲稿、该指哪张图、可能被问到什么、别讲什么。">
<meta name="theme-color" content="#5f6368">
''' + _css() + '''
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic02-build-L200-lecture.py 生成，不要手改。
     加一讲 = 往那个脚本的 LEC 里加一个 dict，侧栏和计数会自动跟上。 -->

<div class="top"><div class="in">
  <span class="who">专题二 · <b>TPU 与 GPU</b>　<span style="opacity:.6">L200</span></span>
  <nav class="tabs">
    <a href="''' + DECK + '''">课件 L200</a>
    <a href="topic-02-L200-lecture.html" class="on">讲义</a>
    <a href="''' + FULL + '''">完整版 L300</a>
  </nav>
  <span class="meta" id="meta"></span>
</div></div>

<div class="shell">

<aside class="side">
  <h4>讲次</h4>
  <ol id="toc">
''' + _toc() + '''
  </ol>
</aside>

<main>

<div class="plan">
  <h2>这份讲义怎么用</h2>
  <p>这是<b>老师的草稿</b>，不是给学员的材料。学员看
    <a href="''' + DECK + '''">课件 L200</a>，你看这一份。<br>
    课程作者 <b>Chris Yang</b> · Google Cloud AI Infra 架构师。</p>
  <table>
    <tbody>
      <tr><td>🎯</td><td><b>这一讲要留下什么</b>　讲完之后学员脑子里应该剩的那一句话。
        <b>只有一句</b> —— 讲散了就是没讲到。</td></tr>
      <tr><td>🗣</td><td><b>讲稿</b>　接近逐字，可以照着念。<em>黄底</em>的是必须说出口的原话，
        其余按自己的语感改。</td></tr>
      <tr><td>🖥</td><td><b>屏幕</b>　这一段该滚到课件的哪里、该指哪张图的哪一格。</td></tr>
      <tr><td>❓</td><td><b>可能被问到</b>　真实会被问的问题 + 答法。
        <b>答不上来的就写「这个我不确定」</b> —— 讲义里编一个，现场你会照着念。</td></tr>
      <tr><td>⚠️</td><td><b>别讲什么</b>　这一讲最容易跑偏的方向。<b>时间就是被这些吃掉的。</b></td></tr>
    </tbody>
  </table>

  <h3>一小时怎么分</h3>
  <p>L200 全篇 33 张图。下面这张表是<b>计划</b>，不是实测 ——&nbsp;
    每讲过一遍，回到生成脚本里把真实分钟数改回去。</p>
  <table class="sched">
    <thead><tr><th>分钟</th><th>讲什么</th><th>这一段的落点</th></tr></thead>
    <tbody>
      <tr><td>0–4</td><td><b>开场</b>：两条出身 ＋ 那张地图</td>
        <td>把「差别只有一处」这个论点立住。<b>它现在只是论点，不是结论</b></td></tr>
      <tr><td>4–8</td><td><b>第 1 节 · 架量具</b>：算力 ÷ 带宽，两边撞在同一个数上</td>
        <td>第一个「坐直」的时刻：<b>312.6 对 312.5</b>。参数表在这里被拆穿</td></tr>
      <tr><td>8–15</td><td>⭐ <b>第 2 节 · 分岔点</b>：同一次访存，一边有 cache，一边没有</td>
        <td>论点第一次变成可看见的东西</td></tr>
      <tr><td>15–39</td><td>⭐⭐ <b>第 3 节 · 拆开看</b>：整颗 → 一个核 → 一条指令 →
        <b>FlashAttention 两边并排走完一遍</b></td>
        <td><b>全课最重的 24 分钟，14 张图。</b>讲不完别的可以砍，这段不能砍</td></tr>
      <tr><td>（39–42）</td><td><b>第 4 节 · 岔路</b>：一张卡装不下的时候</td>
        <td><b>时间不够整节跳过</b>，对第 5 节零影响。课件里它本来就是折叠的</td></tr>
      <tr><td>42–51</td><td>⭐ <b>第 5 节 · 分岔的本质</b>：运行时，还是编译期</td>
        <td>开场那个论点在这里<b>收口</b> —— 三次出场合成一件事</td></tr>
      <tr><td>51–58</td><td><b>第 6 节 · 跑完之后</b>：两个数摆上桌，能不能比</td>
        <td>产出不是那个倍数，是<b>「这个数往哪边都有理由」</b></td></tr>
      <tr><td>58–62</td><td><b>第 9 节 · 收尾</b>：那到底各自擅长什么</td>
        <td>回到最开始那句：不比谁强，只说由谁决定</td></tr>
    </tbody>
  </table>
  <h3>讲不完的时候先砍哪个</h3>
  <p>顺序是固定的，<b>临场别现想</b>：①&nbsp;第 4 节整节跳过（−3）；
    ②&nbsp;第 6 节只留那张两个数的实测图，边界那张压成一句（−4）；
    ③&nbsp;第 3 节前三小节各砍一张图（−4）。<br>
    <b>⛔ 第 3.6（FlashAttention 并排走那一段）和第 5 节，任何情况下都不能砍</b> ——&nbsp;
    砍掉它们，这节课就退回成「两块硬件参数对比」，
    而那正是开场第一句话说了不做的事。</p>
</div>
''' + _sections() + '''

<footer>
  <p>讲义随课上的实际反应改。<b>发现哪一讲现场讲不通，改生成脚本
    <code>Courses/tools/topic02-build-L200-lecture.py</code> 里的 <code>LEC</code>，
    别改课件</b> —— 课件是学员的材料，讲不通是讲法的问题，不一定是材料的问题。</p>
</footer>

</main>
</div>

<script>
// 侧栏跟随滚动高亮
const secs = [...document.querySelectorAll(".lec")];
const links = new Map([...document.querySelectorAll(".side a")].map(a=>[a.getAttribute("href").slice(1), a]));
const io = new IntersectionObserver(es=>{
  es.forEach(e=>{
    const a = links.get(e.target.id); if(!a) return;
    if(e.isIntersecting){ links.forEach(x=>x.classList.remove("on")); a.classList.add("on"); }
  });
}, {rootMargin:"-78px 0px -70% 0px", threshold:0});
secs.forEach(s=>io.observe(s));
// 未写完的讲次淡显
document.querySelectorAll(".lec").forEach(s=>{
  if(s.querySelector(".todo-box")) links.get(s.id)?.classList.add("todo");
});
// 顶栏统计
const done = secs.filter(s=>!s.querySelector(".todo-box")).length;
const mins = [...document.querySelectorAll('.lh .t')]
  .reduce((s,e)=>s+parseFloat((e.textContent.match(/[0-9.]+/)||[0])[0]), 0);
document.getElementById("meta").textContent =
  `${secs.length} 讲 · 已写 ${done} · 合计 ${mins} 分钟`;
</script>
</body>
</html>
'''

    # ── 自检（全部跑在写盘之前，理由见 topic02-build-L200.py 同一位置）──
    import sys
    sys.path.insert(0, os.path.join(HERE, "tpu-micro"))
    from gate import lint_public
    bad = lint_public(html)
    assert not bad, "公开页面里出现内部词，已中止写盘：%s" % bad

    # 侧栏每一条都必须能落到一个真的 section 上。锚点写错是**静默**的 ——
    # 点了没反应，不会报错，而侧栏是这份讲义唯一的导航。
    ids = set(re.findall(r'<section class="lec" id="(\w+)"', html))
    hrefs = set(re.findall(r'<li><a href="#(\w+)"', html))
    assert hrefs == ids, "侧栏和正文对不上：侧栏多 %s，正文多 %s" % (
        sorted(hrefs - ids), sorted(ids - hrefs))

    io.open(OUT, "w", encoding="utf-8").write(html)
    done = sum(1 for L in LEC if L["html"])
    main_min = sum(L["min"] for L in LEC if L["id"] != "L4")
    print("ok  topic-02-L200-lecture.html  %s 字符" % format(len(html), ","))
    print("    %d 讲，已写 %d 讲　主线 %d 分钟（第 4 节 %d 分钟可跳，不计入）"
          % (len(LEC), done, main_min, next(L["min"] for L in LEC if L["id"] == "L4")))


if __name__ == "__main__":
    main()
