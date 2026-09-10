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


# ⭐ 逐字稿块的样式。⛔ 自己新造的 class 一定要配样式 ——&nbsp;
#   没样式的 class 完全合法，只是**安静地什么都不做**（2026-09-09 刚在 .secno 上栽过）。
#   讲稿是准备时读的（蓝），逐字稿是临场照着念的（绿），配色分开。
_ORAL_CSS = """
details.oral{border:1px solid #a8dab5;border-left:4px solid #1e8e3e;border-radius:8px;
  background:#f6fbf7;margin:16px 0;overflow:hidden}
details.oral>summary{cursor:pointer;list-style:none;padding:10px 16px;
  font-weight:700;font-size:14.5px;color:#0d652d;user-select:none}
details.oral>summary::-webkit-details-marker{display:none}
details.oral>summary::before{content:"\\25b8";display:inline-block;margin-right:8px;
  font-size:11px;color:#1e8e3e;transition:transform .18s}
details.oral[open]>summary::before{transform:rotate(90deg)}
details.oral[open]>summary{border-bottom:1px solid #d7ecdd}
details.oral .body{padding:6px 20px 16px}
details.oral .body p{margin:12px 0;font-size:15.5px;line-height:1.95}
details.oral .body b{color:#0d652d}
details.oral .body em{color:#5f6368;font-style:normal}
"""


def _css():
    h = io.open(CSS_SRC, encoding="utf-8").read()
    i = h.index("<style>")
    css = h[i:h.index("</style>")]
    # ⛔ 断言样式真的进去了 ——&nbsp;上一版我把 CSS 插错了地方，构建照样成功，
    #   页面上逐字稿块只是没有边框，肉眼很难第一时间发现。
    out = css + _ORAL_CSS + "</style>"
    assert "details.oral{" in out
    return out


LEC = []


def lec(i, no, title, mins, toc, what, land, html, grp=None, opt=False):
    """opt=True ＝ **可跳的岔路**，不计进主线时长。
    ⛔ 时长护栏（文件末尾那句 assert）只数主线 ——&nbsp;
      把可跳的也算进去，会逼着你去砍主线里真正该讲的东西。"""
    LEC.append(dict(id=i, no=no, title=title, min=mins, toc=toc, what=what,
                    land=land, html=html, grp=grp, opt=opt))


# ══════════════════════════════════════════════════════════════════
lec("X0", "开场", "这十九分钟要回答两件事", 1, "开场：两件事",
    what="接回屋脊点这把尺子，声明不比 benchmark，并**预告后半场是我们自己跑过的**",
    land="台下知道这不只是一堂讲道理的课 —— **后八分钟全是实活**",
    grp="判据 · 11 分钟", html="""
  <div class="goal"><span class="k">这一讲要留下什么</span>
    <b>前十分钟：一条线 ＋ 两颗芯片拆开，判你的活配不配，而且跑之前就能判。</b><br>
    <b>后八分钟：配了之后我们是怎么落地的 —— 三段怎么拆，十个模型怎么跑通的。</b>
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">课件停在首屏。</span>

    <p>这一讲十九分钟，专题二的外传。<b>分两半。</b></p>

    <p>前一半讲<b>判据</b>：一条线，判你手上的活配不配这颗芯片 ——
      <em>而且这件事<b>跑之前就能判</b>，不需要你先给我机器。</em></p>

    <p>后一半讲<b>我们做了什么</b>：这类模型的流水线为什么能拆成三段部署、
      拆完各放哪台机器，以及<b>五个月里十个模型是怎么一个个跑通的</b>。</p>

    <span class="pause">⭐ <b>这句要说出口</b>：「后一半不是推的，是我们自己趟出来的。」<br>
      <em>台下坐的是要做决定的人 —— 他们对「我们试过」的兴趣，远大于对公式的兴趣。</em></span>

    <p>开始之前先说清一件事：<b>这十九分钟我不给你们比 benchmark，一个快慢数都没有。</b>
      屋脊点是个<b>结构量</b>，它只说这台机器的胃口有多大。</p>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="dont">
    <b>别在这儿解释 roofline。</b>画那张折线图要五分钟，这一讲只需要「一根轴上四个点」。<br>
    <b>别铺垫太久。</b>开场只有一分钟 —— 说完两件事就往下走。
  </div>
""")

lec("X1", "一", "一把尺子量四颗，再把两颗拆开", 5, "一：那条线 ＋ 两颗芯片",
    what="给出那条线，并强调它**纯用规格算、不需要实测**",
    land="**560 对 295 —— 「算力强、显存弱」不是形容词，就是这个数**",
    grp="判据 · 11 分钟", html="""
  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到 <b>图 X-1</b>，四颗芯片一根轴。</span>

    <p>专题二那把尺子：<b>算力除以显存带宽</b>。它的意思是
      <em>「每从显存搬一个字节，这台机器本来能算多少次」。</em></p>

    <p>三颗旗舰挤在 295 到 313 —— <b>H100、B200、TPU v7，胃口差不多。</b>
      而 TPU v6e 是 <b>560</b>。</p>

    <span class="pause">⭐ 停一下，让这个数落地。<br>
      「算力强、显存弱」这六个字，在这儿变成了一个可以写进邮件的数字。</span>

    <p>拆开看就是三个比值：跟 H100 比，<b>算力 93%、带宽只有 49%、容量只有 40%</b>。
      分子基本没动，分母砍一半，商自然涨到两倍。</p>

    <p><b>这条线最值钱的地方是：它纯用官方规格算，一个实测都不需要。</b>
      <em>也就是说 —— 客户还在犹豫要不要给你机器的时候，你已经能下判断了。</em></p>
  </div>

  <h3 class="sec">怎么用它</h3>
  <div class="dont">
    把客户的负载也算出一个<b>强度</b>，跟这条线一比：
    <b>落右边＝算力受限，落左边＝带宽受限。</b>就这一句。<br>
    <b>别展开推导。</b>怎么除出来的、H100 那个 989.5 为什么不是 1979 —— 全在 L200。
  </div>

  <h3 class="sec">接着讲两张显微镜图（图 X-2 / X-3）</h3>
  <div class="say">
    <span class="board">滚到 <b>图 X-2</b>，一颗 v6e。</span>

    <p><b>这个 560 是从什么样的硅片里长出来的？</b>拆开看。</p>

    <p>v6e：算力<b>集中</b>在两个 256 乘 256 的大方阵里；片上有一整块 128 MiB 的暂存，
      <b>全部由编译器安排，没有硬件缓存兜底</b>；通往片外那道门 1,638 GB/s。</p>

    <span class="board">⚠️ <b>图 X-3（H100）2026-09-10 起默认折起</b>
      （现场原话：「这个不重要」）。<b>折叠条上已经写着
      528 / 硬件自动管的 L2 / 门宽一倍</b> ——&nbsp;
      <b>念折叠条就够，不必点开。</b></span>

    <p>H100：算力<b>摊成 528 个</b>小单元；片上多了一整层<b>硬件自动管</b>的 L2；门宽一倍。</p>

    <span class="pause">⭐ <b>只有下面这句需要点开图 X-3</b>：那道门在两张图里是
      <b>按带宽等比画的</b>，可以直接叠着比 —— <em>台下会自己看出来那是一半。</em><br>
      ⏱ <b>时间紧就整句跳过、不点开</b>：对照的三样折叠条上都有了，
      少的只是「看一眼」这个动作。<em>判据：折的是图，不是结论。</em></span>

    <p>三个问题，两边给了<b>相反</b>的答案。而且各自都自洽：<br>
      <b>H100 那一列全部指向「负载未知」</b> —— 摊开、缓存兜底、门开大，
      <em>三个都是为不确定性买的保险</em>。<br>
      <b>v6e 那一列全部指向「负载已知」</b> —— 集中、编译期写死、门只开够用，
      <em>三个都是把保险费省下来换算力密度</em>。</p>

    <p>所以<b>问题从来不是谁更强，是你手上的活属于哪一种。</b></p>
  </div>

  <h3 class="sec">⚠️ 顺带埋一个后面要用的钩子</h3>
  <div class="dont">
    v6e 那个 <b>CMEM ＝ 0</b> 是有代价的：<b>编译器排错了就没有后手</b>。
    形状动态、访存模式运行时才知道的负载，在这套设计上会很难受。<br>
    ⭐ <b>说完这句就走，别展开</b> —— 第四节讲 VAE「编译 80 秒、跑 1 秒」的时候，
    台下会自己接上。
  </div>
""")

lec("X9X4", "二", "那你的活落在哪一边", 3, "二：算一遍，放回轴上",
    what="拿 Wan2.1 **官方 config** 当场算，再把四类负载放回轴上",
    land="**强度 75,600 是门槛的 135 倍 → 短板被挡在瓶颈之外，只剩算力那 93% 在起作用**",
    grp="判据 · 11 分钟", html="""
  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到 <b>图 X-9</b>。</span>

    <p>不引用别人的结论，拿 Wan2.1 的<b>官方 config</b> 当场算。</p>

    <p>720P、81 帧，过 VAE 压缩、再 patch 化，得到 <b>75,600 个 token</b>。
      对照一下：<b>LLM 解一步只处理 1 个 token。</b></p>

    <p>再看 config 里一个字段：<code>window_size = (−1, −1)</code> ——
      <b>不开滑窗，全局注意力。</b>于是算力的<b>七成压在 N² 那一项上</b>，
      而那是纯矩阵乘。</p>

    <span class="pause">⭐ 这里值得多说一句<b>我们怎么确认自己没算错</b>：<br>
      同一份 config 把参数量数出来 —— 每层 351.3 M，四十层<b>14.05 B</b>，
      <em>正好对上官方标称的 14B。</em><br>
      <b>参数量能对上，说明我们对这一层里有哪些矩阵的理解是对的。</b></span>

    <span class="board">滚到 <b>图 X-4</b>，那根强度轴。</span>

    <p>强度约等于 <b>75,600</b>，是那条 560 的 <b>135 倍</b>。<b>带宽根本没参与。</b></p>

    <p>所以「v6e 适合扩散」的<b>精确说法</b>是：
      <em>不是它跑得更快</em> —— 是扩散把它的两个短板<b>挡在了瓶颈之外</b>，
      于是只剩没被挡住的那一项在起作用：<b>算力 918 对 989.5，93%。</b></p>

    <p><b>短板不参与，长板打平。</b>这就是「合适」的全部含义。</p>
  </div>

  <h3 class="sec">⛔ 这两条一定要主动说</h3>
  <div class="dont">
    <b>① 反过来也成立：</b>小 batch 的 decode 落在轴最左边，那里比的全是带宽，
    v6e 只有 49% —— <b>同一颗芯片，那种活它最吃亏。</b>
    主动说这条，客户才信你前面那句。<br>
    <b>② 落在算力侧不等于算力吃满了：</b>我们在 <b>Wan2.2</b> 上抓的 profile ——&nbsp;
    <b>Splash Attention 算子 MFU 37%</b>、<b>整模型 34%</b>（基线 12%），
    卡在 head_dim 128 对 MXU 的 256。<b>屋顶线说瓶颈在哪一侧，不说那一侧用得好不好。</b><br>
    ⛔ <b>口径要说准</b>（2026-09-09 审计改）：原来写「实测 Wan 的 MFU 37%」——&nbsp;
    可这一节从头到尾在算 <b>Wan2.1</b>，而 37% 出自 <b>Wan2.2</b> 的优化指南，
    <b>而且它是那一个算子的数，不是整模型的</b>。被问到「整体还是算子」要答得出来。
  </div>
""")

lec("X6", "三", "装得下吗 ——&nbsp;要算权重加激活（⏱ 主线可跳）", 2, "三：装得下吗（权重＋激活）",
    what="把权重体积摆到 32 GB 这条线上，给出「摊开还是多放几路」的判据",
    land="**权重决定装不装得进，激活决定跑不跑得动 —— 而扩散是激活说了算**",
    grp="判据 · 11 分钟", html="""
  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到 <b>图 X-6</b>。</span>

    <p>扩散这一族的权重在 <b>7 到 54 GB</b>；同一根轴上，
      Qwen3.5-397B 是 794 GB、DeepSeek-V3 是 1,342 GB ——&nbsp;<b>差一个半到两个数量级。</b></p>

    <p>一台 v6e-8 是八颗乘 32 GB。所以同一台机器有两种用法：</p>
    <p><b>Wan2.1 的 28 GB</b> 塞单颗只剩 4 GB 余量 → <b>TP 摊到八颗</b>，每颗 3.5 GB。<br>
      <b>SDXL 只有 7 GB</b> → 一颗装得下，那就<b>开八路各生成各的</b>，我们实测 2.40 张每秒。</p>

    <p>一句话：<b>装不下就把一个模型摊开，装得下就多放几路。</b></p>
  </div>

  <h3 class="sec">⛔ 但这一节真正要讲的是下半张图</h3>
  <div class="say">
    <span class="board">滚到图 X-6 下半部分那张预算表。</span>

    <p>上面那根轴<b>只画了权重</b>。而扩散模型真正吃显存的<b>不是权重，是激活</b>。</p>

    <p>所以「28 GB 小于 32 GB 所以能跑」这句话 —— <b>分子分母都错。</b></p>

    <p><b>分母错</b>：32 GB 是标称，不是预算。真跑起来权重和运行时先占掉一大块。
      我们自己撞 OOM 那次，XLA 报的原话是
      <em>「There are 13.10G free」</em> —— <b>只剩十三点一个 G。</b></p>

    <p><b>分子也错</b>：要放进去的不只是权重，还有<b>峰值激活</b>。
      同一条报错里前半句是 <em>「Attempting to reserve 19.00G」</em> ——&nbsp;
      <b>CogVideoX 的 VAE 解码一步就要十九个 G，而这个模型的权重才十个 G。</b>
      <b>激活是权重的近两倍。</b></p>

    <span class="pause">⭐⭐ <b>这里给最有说服力的一句</b>：
      激活随分辨率和帧数涨，<b>而权重一个字节都不涨</b>。<br>
      <em>Wan2.1 的 480P 跑得动、720P OOM —— <b>用的是同一份权重。</b></em></span>

    <p>而且解法不是换更大的卡：逐帧解码加共享缓存，把那十九个 G 压到十三以下 ——&nbsp;
      <b>改的是实现。</b></p>

    <p>一句话收口：<b>权重决定装不装得进，激活决定跑不跑得动。</b></p>
  </div>

  <h3 class="sec">别讲什么</h3>
  <div class="dont">
    <b>别把上半张图讲成「扩散都装得进一颗」</b> —— 那两条踩线和超线的是特意画的，
    而且那根轴只算权重。<br>
    <b>别展开 Pod 拓扑</b>，也别展开逐帧解码怎么实现 —— 都在 L200 和专题十一。
  </div>
""", opt=True)   # ⏱ 2026-09-10 现场：这一节在故事线上没作用，改成主线可跳

lec("X10", "四", "仓库里为什么同一个模型有两份例子", 3, "四：切口露在外面",
    what="讲清三阶段那份例子存在的理由，并给出**验证一份 latent 的三步自检**",
    land="**「看文件大小」不算验证 —— shape、dtype、字节数，三步都要读头**",
    grp="我们做了什么 · 8 分钟", html="""
  <div class="goal"><span class="k">从这里开始换气口</span>
    前面三节讲的是<b>道理</b>。<b>从这一节起，全是我们自己跑出来的东西。</b><br>
    ⭐ 明确说一句「下面这些是我们趟出来的」——&nbsp;台下的注意力会明显回来。
  </div>

  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到 <b>图 X-10</b>。</span>

    <p>打开我们仓库里任何一个扩散模型的目录，都会看到<b>两份例子</b>：
      一个一体化脚本，和一个 staged 目录。</p>

    <p><b>后者存在的理由不是「跑得更快」——&nbsp;是它把切口露在外面。</b></p>

    <p>一体化那份跑得快，适合验证一次改动、量一次端到端。
      <b>但你看不见中间态</b>：文本 embedding、latent 全在内存里，跑完就没了。
      <em>一旦出问题 —— 视频全黑、出 NaN、动作快进 —— 你没有任何中间产物可看。</em></p>

    <p>三阶段那份把这一切摊开：三段之间只交接<b>三个文件</b> ——&nbsp;
      两个 safetensors 加一份 config。<b>只认文件，不认进程。</b></p>

    <span class="pause">⭐ <b>这句可以停一下</b>：既然它们只认文件不认进程，
      <em>那三段本来就能跑在三台机器上</em> —— 下一节讲各该放哪台。</span>
  </div>

  <h3 class="sec">⭐ 这一节真正要教会的：拿到一份 latent，怎么验</h3>
  <div class="say">
    <p><b>① shape 对不对</b> —— 由分辨率直接推：
      帧 (81−1)÷4+1 得 21，高 720÷8 得 90，宽 1280÷8 得 160，通道 16。
      期望是 <b>[1, 16, 21, 90, 160]</b>。<b>对不上就别往下跑</b>，
      后面只会得到全黑或 NaN。</p>

    <p><b>② dtype 在哪看</b> —— 读 safetensors 头的 dtype 字段，
      <b>再读 metadata 里的 dtype_info</b>。
      <em>⭐ 这两者可能不一样</em>：Wan 的 embedding <b>盘上是 F32，
      而 dtype_info 写着原始是 bfloat16</b> —— 保存时转的，加载时按那条恢复。</p>

    <p><b>③ 字节数对不对</b> —— 16×21×90×160×4 再加 272 字节的头，
      等于 <b>19,353,872</b>，跟文件<b>一个字节不差</b>。</p>

    <span class="pause">⭐⭐ <b>这里给一个反例，整节就立住了</b>：<br>
      Wan2.1 的 latents 是 F32、一千九百三十五万字节；
      CogVideoX 的 latents <b>形状一模一样</b>，但 BF16、九百六十七万字节。<br>
      <em><b>大小差一倍，形状完全相同</b> —— 所以「看文件大小」既不能证明 shape 对，
      也不能反推 dtype。<b>只能读头。</b></em></span>
  </div>

  <h3 class="sec">⛔ 这张图我推翻重写过一次 —— 值得当众说</h3>
  <div class="dont">
    初版我画的是「数据体积对数轴」，最粗的一根柱子标着
    <b>457 GB —— 注意力分数矩阵</b>，旁边注了一句「从不落地」。<br>
    <b>那是错的</b>：那个矩阵<b>从来没有被物化出来过</b>，
    Flash / Splash Attention 是分块算的，算完即弃，HBM 里根本不存在这么一块。<br>
    ⭐ 而「从不落地」那句注解<b>救不了它</b>：把一个不存在的量画成「管子最粗处」，
    <b>整张图的比例尺就锚在了虚构上</b> —— 读者记住的是柱子，不是旁边那行小字。
    <b>注解抵消不了图形本身的断言。</b>
  </div>
""")

lec("X11", "五", "拆完怎么摆", 3, "五：三段各放哪台机器",
    what="三段的资源画像完全不同 —— 这才是分开部署的真正理由；并给一个反例",
    land="**吃算力的、吃显存的、几乎不吃的，塞一台机器必然有人吃不饱有人撑着**",
    grp="我们做了什么 · 8 分钟", html="""
  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到 <b>图 X-11</b>。</span>

    <p>切得动只是可行性。<b>真正的收益来自另一件事：这三段吃的根本不是同一种资源。</b></p>

    <p><b>文本编码</b>几乎不吃算力，占全程 1.3%。<br>
      <b>DiT 去噪吃算力</b>，占 98.3% —— 前面算的那三亿亿次浮点全在这儿。<br>
      <b>VAE 解码吃显存峰值</b>，把四百八十万个数展开成两亿两千万个，却只占 0.4%。</p>

    <span class="pause">⭐ VAE 还有个特别的形状值得单说：<b>预热八十秒，之后每次一秒 —— 八十倍差。</b><br>
      <em>这种东西<b>天生该做成常驻服务</b>：编译好放着，八十秒摊到上万次调用等于零。
      而一体化脚本每起一次进程就得重付一遍。</em></span>

    <p>所以左边那种「三段全塞一台机器」的问题不是慢，是
      <b>按最馋的那一段配机器，另外两段的钱就白付了</b>。</p>
  </div>

  <h3 class="sec">⭐ 这个反例一定要讲 —— 它让整页可信</h3>
  <div class="dont">
    看完这张图最容易得出的结论是<b>「文本编码那么轻，放 CPU 就行」</b>。
    在 Wan2.1 上成立 —— 它用 T5，那一段 <b>3 秒</b>。<br>
    <b>但 Flux.2 当场推翻它</b>：它用 Mistral3，<b>放 CPU 要 30 秒</b>，
    而它的 DiT 在 TPU 上跑完五十步只要 <b>13.5 秒</b> ——&nbsp;
    <b>那个「轻量」的第一段，反而是全程最慢的一段。</b><br>
    ⭐ 判据：<b>不是「哪一段天生该放哪」，是先量一量它在目标硬件上要多久。</b>
    而分段的价值恰恰在这里 —— <b>拆开之后，每一段的账才第一次能单独算清楚。</b>
  </div>
""")

lec("X12", "六", "这套说法我们自己验过", 2, "六：十个模型，五个月",
    what="十个模型的时间线，落点在**提交数从 73 掉到 2**",
    land="**接第十一个模型的成本，已经不是前十个的量级了**",
    grp="我们做了什么 · 8 分钟", html="""
  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到 <b>图 X-12</b>。</span>

    <p>前面五节讲的都是<b>应该怎样</b>。这一页讲<b>实际怎样</b>。</p>

    <p>十个模型，五个月。视频这边 HunyuanVideo、Wan 2.1、Wan 2.2 图生视频、CogVideoX；
      图像这边 SDXL、Flux.1、Flux.2、S3Diff 超分、Real-ESRGAN；
      外加一套 ComfyUI 节点，让不写代码的人也能用。</p>

    <p><b>每一行的起止和提交数都是 git 提交历史直接数出来的，不是凭印象写的。</b></p>

    <span class="pause">⭐ <b>这一页真正的落点不是「我们做了十个」。</b><br>
      看提交数：HunyuanVideo <b>73</b> 次、CogVideoX 49、Wan2.1 44 ——&nbsp;
      而 SDXL <b>9</b> 次、Real-ESRGAN <b>3</b> 次、Flux.1 <b>2</b> 次。</span>

    <p>差别<b>不在模型难度</b> —— SDXL 和 Flux.1 都不比 CogVideoX 简单。
      差别在于：<b>前面几个是在发明方法，后面几个是在套用方法。</b></p>

    <p>所以落点是这一句：<b>接第十一个模型的成本，已经不是前十个的量级了。</b>
      <em>这才是一条工程路线走通的标志。</em></p>

    <p>顺带说，这十个盖了<b>五种架构</b>：DiT、MMDiT、UNet、MoE、纯卷积。
      最后那个 Real-ESRGAN 是特意留的 —— 它是唯一一个<b>非 Transformer、非扩散</b>的，
      放它进来就是<b>看这套框架在完全不同的架构上还成不成立</b>。
      <b>答案是成立的。</b></p>
  </div>

  <h3 class="sec">⚠️ 口径</h3>
  <div class="dont">
    <b>提交数只反映改动次数，不等于工作量或难度。</b>
    这里只用它做「发明 vs 套用」的量级判断，
    <b>不要拿它比较任意两个模型谁更难</b> —— 会被当场问穿。
  </div>
""")

lec("XEND", "收尾", "带走三句", 0, "收尾：带走三句",
    what="收口，并把往下挖的路指清楚",
    land="**选芯片先看活落在哪边；腰细所以能拆；这套我们跑过五个月**",
    grp="收尾", html="""
  <h3 class="sec">讲稿</h3>
  <div class="say">
    <span class="board">滚到收尾。</span>

    <p><b>第一句：</b>选芯片不是选更强的那颗，是先看你的活落在那根轴的哪一边 ——&nbsp;
      <b>而这件事跑之前就能判。</b></p>

    <p><b>第二句：</b>扩散的流水线有一道很细的腰，只有 19.4 兆，
      所以三段可以拆到不同机器上各配各的资源 —— <b>切口成本万分之零点七。</b></p>

    <p><b>第三句：</b>这套说法我们在十个模型上跑了五个月，
      而提交数从 73 掉到 2 —— <b>接下一个模型的成本，已经不是当初的量级了。</b></p>

    <span class="pause">⭐ 最后一句留给对面：<b>「你手上那个模型，我们可以先把它放到这根轴上看看。」</b><br>
      <em>——&nbsp;不需要他先给机器，这是这一讲最实用的一个副产品。</em></span>
  </div>

  <h3 class="sec">往下挖的路</h3>
  <div class="dont">
    <b>L200 精讲版</b>：每一步的推导、反例与中间量，外加<b>两颗芯片的显微镜级拆解</b>
    （v6e / H100 逐项对照、SparseCore 的两个职责、二维环面）——&nbsp;
    这些本讲全砍了，收尾那张对照表标了它们在 L200 的位置。<br>
    <b>专题十一</b>：扩散模型本身的原理。
  </div>
""")


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


from topic02x_oral import ORAL                    # 逐字稿，单独一份，见该文件头


def _oral(i):
    """渲染那一节的逐字稿。⛔ 缺一节直接断言失败 ——&nbsp;
    「讲义里某一节没有逐字稿」不报错也不难看，只在临场翻到那一页时才发现。"""
    assert i in ORAL, "%s 这一节没写逐字稿 —— 见 topic02x_oral.py" % i
    return ('<details class="oral"><summary>🎙 逐字稿 ——&nbsp;可以直接照着念</summary>'
            '<div class="body">%s</div></details>' % ORAL[i])


def _secs():
    out = []
    for L in LEC:
        out.append(
            '<section class="lec" id="%s">\n'
            '  <h2><span class="no">%s</span>　%s　<span class="min">%d 分钟</span></h2>\n'
            '  <div class="meta2"><b>讲什么</b>：%s<br><b>落点</b>：%s</div>\n'
            '%s\n%s\n</section>' % (L["id"], L["no"], L["title"], L["min"],
                                    _md(L["what"]), _md(L["land"]), L["html"],
                                    _oral(L["id"])))
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
    assert set(ORAL) == {L["id"] for L in LEC}, \
        "逐字稿与讲义章节对不上：多 %s，少 %s" % (\
            set(ORAL) - {L["id"] for L in LEC}, {L["id"] for L in LEC} - set(ORAL))
    # ⛔ 2026-09-10 现场：「一颗装得下吗这个章节在整个故事线上没什么用，
    #   把这个章节折起来，15 分钟不讲这个。」→ X6 改成 opt=True。
    # ⚠️ 他说的 15 是约数：19 − 2 ＝ 17。**如实记 17，不去凑那个数。**
    assert total == 17, "主线合计 %d 分钟 —— 目标 17（X6 已改可跳），超了就砍" % total

    io.open(OUT, "w", encoding="utf-8").write(html)
    print("ok  topic-02x-lecture.html  %s 字符　%d 讲　主线 %d 分钟"
          "（另有可跳 %d 分钟）"
          % (format(len(html), ","), len(LEC), total, optm))


if __name__ == "__main__":
    main()
