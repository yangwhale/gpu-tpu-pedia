# -*- coding: utf-8 -*-
"""专题八 · 精度与量化 —— 教师讲义（授课稿）。

════════════════════════════════════════════════════════════════
跟专题二那份讲义的关系
════════════════════════════════════════════════════════════════
CSS 同样从**专题一讲义**里抽（`_css()`），三份讲义长一个样。
⛔ 不要在这里另写 CSS。

但**没有沿用专题二那套 LEC 机制**，理由很实在：那套机制的价值在于
「侧栏 / 计数 / 灰置」三样从一张讲次表推出来，而它是为**十几讲、
一讲一讲攒**准备的。这一讲目前只有三节写完、四节还是大纲，
上那套机制等于先搭脚手架再盖一间平房。

⭐ **等这一讲写到六节以上、需要排时间轴的时候，再换成 LEC。**
   换的时候直接抄 topic02-build-L200-lecture.py 的 LEC / _toc / _sched。

════════════════════════════════════════════════════════════════
⛔ 这份讲义的边界：没写的节，讲义里也不许编
════════════════════════════════════════════════════════════════
课件（topic-08.html）里 🚧 的那几节，这里同样只给「打算怎么讲」，
**不给逐字稿** —— 讲义里编一段，现场你会照着念。
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-01-lecture.html")
OUT = os.path.join(WEB, "topic-08-lecture.html")
DECK = "topic-08.html"


def _css():
    h = io.open(CSS_SRC, encoding="utf-8").read()
    i = h.index("<style>")
    return h[i:h.index("</style>") + len("</style>")]


html = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>专题八 · 讲义（授课稿）</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#129408;</text></svg>">
<meta property="og:title" content="精度与量化 · 讲义">
<meta property="og:description" content="老师的草稿：逐字讲稿、该指哪张图、可能被问到什么、别讲什么。">
<meta name="theme-color" content="#5f6368">
''' + _css() + '''
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic08-build-lecture.py 生成，不要手改。 -->

<div class="top"><div class="in">
  <span class="who">专题八 · <b>精度与量化</b></span>
  <nav class="tabs">
    <a href="''' + DECK + '''">课件</a>
    <a href="topic-08-lecture.html" class="on">讲义</a>
    <a href="topic-02.html">专题二（硬件背景）</a>
  </nav>
</div></div>

<div class="shell">
<main>

<div class="plan">
  <h2>这份讲义怎么用</h2>
  <p>这是<b>老师的草稿</b>，不是给学员的材料。学员看
    <a href="''' + DECK + '''">课件</a>，你看这一份。<br>
    课程作者 <b>Chris Yang</b> · Google Cloud AI Infra 架构师。</p>
  <table><tbody>
    <tr><td>🎯</td><td><b>这一讲要留下什么</b>　讲完之后学员脑子里应该剩的那一句话。</td></tr>
    <tr><td>🗣</td><td><b>讲稿</b>　接近逐字，可以照着念。<em>黄底</em>的是必须说出口的原话。</td></tr>
    <tr><td>🖥</td><td><b>屏幕</b>　这一段该滚到课件的哪里。</td></tr>
    <tr><td>⚠️</td><td><b>别讲什么</b>　这一讲最容易跑偏的方向。</td></tr>
  </tbody></table>
</div>

<div class="warn">
  <span class="k">🚧 这份讲义跟课件一样，只写完了中间三节</span>
  <b>§3 缩放、§4 后半 FP4 训练、§5 硬件视角</b> ——&nbsp;这三节有逐字稿。<br>
  <b>§0 §1 §2 §6 只有「打算怎么讲」</b>，<b>没有逐字稿</b>。
  ⛔ <em>讲义里编一段，现场你会照着念 ——&nbsp;所以宁可空着。</em>
</div>

<h3 class="sec">这一讲要留下的一句话</h3>
<div class="say">
  <span class="pause">整讲只留这一句：
    <b>「四个 bit 只负责形状，量级由外面那层 scale 给 ——&nbsp;
    所以量化的全部难点，都在<u>那个 scale 怎么定</u>。」</b></span>

  <p><em>后面每一节都是它的一个侧面</em>：
    <b>scale 要多密</b>（§3）、<b>训练和推理为什么要定得不一样</b>（§4）、
    <b>硬件替你做了哪一半</b>（§5）、<b>定错了为什么不报错</b>（§6）。</p>
</div>

<h3 class="sec">🚧 讲稿 · 0 开场</h3>
<div class="say">
  <span class="board">停在首屏。</span>
  <p><b>打算这么开</b>：所有人都在往下压精度 ——&nbsp;
    但「压精度」不是一个开关，是一整套决策。<br>
    <em>然后直接抛出那个可验证的分界线：</em>
    <b>同一个 FP4，推理侧权重按 1×16 分块，训练侧按 16×16。</b>
    <b>先不解释为什么</b> ——&nbsp;那是 §4 的悬念。</p>
  <span class="pause">⛔ <b>别在开场解释 16×16</b>。
    <em>它的理由（链式法则）需要先建立「一个线性层三个矩阵乘」那个图景，
    开场没有那个铺垫，说了等于白说。</em></span>
</div>

<h3 class="sec">🚧 讲稿 · 1 为什么低精度能行 · 2 格式</h3>
<div class="say">
  <span class="board">课件 §1 §2。</span>
  <p><b>只有提纲，没有逐字稿。</b>要立的两件事：</p>
  <p>① <b>两个收益性质不同</b> ——&nbsp;算力受益的是训练和 prefill，
    访存受益的是 decode。<em>这条不讲透，后面分不清训练和推理。</em></p>
  <p>② <b>范围比精度重要</b> ——&nbsp;BF16 牺牲尾数保指数，所以它赢了 FP16。</p>
  <span class="pause">⭐ <b>②那条一定要留个钩子</b>：
    <b>「记住这个取舍，等下讲 FP4 的 scale 时它会原样再来一次。」</b>
    <em>MXFP4 选纯指数的 E8M0、NVFP4 选有尾数的 E4M3 ——&nbsp;
    同一个取舍换个位置又演一遍。<b>这是这一讲最好的一处呼应，别浪费。</b></em></span>
</div>

<h3 class="sec">讲稿 · 3 缩放这件事</h3>
<div class="say">
  <span class="board">课件 §3。<b>这一节是整讲的核心，给足时间。</b></span>

  <p>先把问题问出来：<b>E2M1 一共四个 bit，能表示的绝对值只有八个
    ——&nbsp;这怎么可能算得准？</b></p>

  <span class="pause">停一拍，答案要慢说：
    <b>「答案不在那四个 bit 上，在<u>外面套的那层 scale</u>。
    四个 bit 只负责形状，量级由 scale 给。」</b></span>

  <p><b>然后立刻给尺度感</b>，别停在抽象：拿 7,168 × 18,432 一层权重走一遍
    ——&nbsp;<b>per-tensor 是一亿三千万个数共用一个，
    NVFP4 是每 16 个一个，那 7,168 切成 448 段。</b>
    <em>两端差八百多万倍。</em></p>

  <span class="pause">⚠️ <b>这里必然有人混</b>，主动拆开：
    <b>「一个 tensor 是整块矩阵，不是它的某一条边 ——&nbsp;
    按 7,168 那一档叫 per-channel，不叫 per-tensor。」</b></span>

  <p><b>命名碰撞也要主动说</b>，不然学员在四份资料里看到四个名字会以为是四件事：
    <em>micro-tensor scaling ／ microscaling(MX) ／ block scaling ／ 块量化，
    再加上 sub-channel、group-wise ——&nbsp;<b>全是同一件事。</b></em></p>

  <span class="pause">⛔ <b>中文这里特别容易踩，这句要说</b>：
    <b>「『块』不蕴含二维。NVIDIA 自己给 block quantization 的定义
    就是沿一个轴分块。」</b></span>

  <p><b>两个格式的差别只在 scale 上</b>：
    <em>MXFP4 每 32 个一个 E8M0（纯指数，只能是 2 的幂）；
    NVFP4 每 16 个一个 E4M3（有尾数），外面再套一个整张量的 FP32。</em></p>

  <span class="pause">⭐ <b>两级为什么存在，这句是关键，慢慢说</b>：
    <b>「不是因为两级更高级 ——&nbsp;是因为 E4M3 最大只到 448，
    盖不住整张量的量级跨度，所以要先用一个 FP32 把张量搬进它的量程。
    外层管<u>范围</u>，内层管<u>局部量级</u>。」</b><br>
    <em>接着反过来说：MXFP4 的 E8M0 范围大得离谱，所以它<b>不需要外层</b>。</em></span>

  <p><b>最后一步：那 16 个数怎么排。</b>
    <em>直接读指令的分片形状 ——&nbsp;A 片 16×64，SFA 片 16×4，
    64 ÷ 16 ＝ 4。<b>一行切四段，完全不跨行。</b></em></p>

  <span class="pause">⭐ <b>这一节的落点，必须说出口</b>：
    <b>「它不需要『相邻的数值相近』这个假设 ——&nbsp;
    只需要这 16 个数的量级别差太远。这是统计性质，不是语义性质。」</b><br>
    <em>⛔ 按「相邻相近」去理解，会推出一堆错的优化直觉。</em></span>
</div>

<h3 class="sec">讲稿 · 4 训练侧 —— 16×16 与链式法则</h3>
<div class="say">
  <span class="board">课件 §4。<b>前半（混合精度、FP8 scaling、QAG）还没写，
    先按提纲带过；后半这一段有逐字稿。</b></span>

  <span class="pause">⛔ <b>先建图景，别急着讲 16×16</b>：
    <b>「一个线性层其实是三个矩阵乘：前向 y ＝ w·x，
    反向之一 ∂x ＝ wᵀ·∂y，反向之二 ∂w ＝ ∂y·xᵀ。
    注意第二个 ——&nbsp;<u>同一个权重，被转置着又用了一次</u>。」</b><br>
    <em>实测：不给这个图景，后面整段听不懂。</em></span>

  <p>然后问：<b>缩放必须沿点积那一维 ——&nbsp;可反向把点积维换了，怎么办？</b></p>

  <p><em>同一个权重，前向沿行切块量化，反向沿列切块量化，
    <b>得到两份不一样的量化结果</b>。</em></p>

  <span class="pause">⭐ <b>这句是这一节的核心，慢，且要说完整</b>：
    <b>「这不是精度差一点，这是<u>链式法则被打破</u> ——&nbsp;
    反向传播算的梯度，对应的已经不是前向那个函数了。你在给另一个函数求导。」</b></span>

  <p><b>于是有了 16×16</b>：<em>一个正方形的块，<b>转置之后还是同一个块</b>，
    前向反向拿到同一份量化权重。</em></p>

  <span class="pause">⛔ <b>这一句必须说，否则学员会记反</b>：
    <b>「16×16 是 256 个数共用一个 scale，<u>比 1×16 粗十六倍</u>。
    它更粗，却更好 ——&nbsp;这里赢的不是精度，是<u>一致性</u>。」</b><br>
    <em>⛔ 千万别讲成「二维更细所以更准」。</em></span>

  <p><b>硬件什么都没多干</b>：<em>Tensor Core 只认沿点积维的一维 scale，
    所以软件算出那个 16×16 的 scale 之后，<b>复制十六份</b>喂进去。
    硬件不知道有二维这回事。</em>
    <b>所以二维不省存储 ——&nbsp;买的是「转置前后是同一份」。</b></p>

  <span class="pause">⚠️ <b>有人会拿论文里的 4 来问</b>，提前拆开：
    <b>「论文里那个 d ＝ 4 是<u>随机 Hadamard 变换的矩阵尺寸</u>，
    不是块大小。结论是 16 比 4 收敛好、跟 128 差不多。」</b></span>
</div>

<h3 class="sec">讲稿 · 5 硬件视角</h3>
<div class="say">
  <span class="board">课件 §5。<b>这一节接专题二，可以快，但三个点要清楚。</b></span>

  <span class="pause">开门见山，别绕：
    <b>「『硬件支持 FP4』加的是三样，<u>没有协处理器</u>：
    乘法阵列认四 bit、MMA 通路里内嵌一级乘 scale、通用核多了几条转换指令。」</b></span>

  <p><b>然后甩那个反例</b>，它比任何解释都有力：
    <em>桌面那颗 GB10 <b>有 FP4 的 Tensor Core 指令，却缺那条转换指令</b>
    ——&nbsp;能拿 FP4 算矩阵乘，却没法把数转成 FP4。</em>
    <b>同一颗芯片上一个有一个没有 ——&nbsp;所以它们是两块独立的硬件。</b></p>

  <span class="board">⚠️ 有人追出处就说清楚：<b>这条出自 NVIDIA 开发者论坛的实测报告，
    不是官方规格书</b>。<em>只用它支撑定性判断，不支撑任何数字。</em></span>

  <p><b>一句话总结分工</b>：<em>硬件负责「用」scale，软件负责「算」scale。</em></p>

  <span class="pause">⛔ <b>这两句别说错</b>：<br>
    <b>「scale 不是放进 Tensor Core 里，是放进 TMEM ——&nbsp;
    TMEM 是 SM 上的一块内存，每 SM 256 KiB。」</b><br>
    <b>「乘 scale 不额外消耗算力，但这不等于不花钱。」</b></span>

  <p><b>三笔代价，都不在 FLOPs 头上</b>：
    <em>① 占 TMEM ——&nbsp;<b>一个 128×256 的 FP32 累加器就吃掉一半，
    真正卡住 tile 能开多大的往往是这一笔</b>；
    ② 多一路搬运，而且必须摆成交错布局；
    ③ 量化那一头是真花时间的 ——&nbsp;<b>只比 MMA 就会把它漏掉。</b></em></p>

  <span class="board">⚠️ 顺带澄清一个名字：<b>「Transformer Engine」不是物理部件</b>，
    芯片版图上找不到它。<em>被问到再说，不必主动展开。</em></span>
</div>

<h3 class="sec">🚧 讲稿 · 6 沉默的失败</h3>
<div class="say">
  <span class="board">课件 §6。<b>只有提纲。但这是全讲风险最高的一节，不能砍。</b></span>
  <p><b>三类，一类比一类隐蔽</b>：
    <em>① 跑起来了、loss 也在降，只是降得慢或后期崩；
    ② 开关没生效也不报错（QAG 那五道锁）；
    ③ <b>链式法则被打破</b> ——&nbsp;连「开关没生效」都算不上，
    一切正常，只是你在给另一个函数求导。</em></p>
  <span class="pause">收口那句：
    <b>「在数值这件事上，『没报错』什么都不能说明。」</b></span>
</div>

<h3 class="sec">别讲什么</h3>
<div class="warn">
  <span class="k">这一讲最容易跑偏的四个方向</span>
  <b>① 别把 §3 讲成格式罗列。</b> 那一节的主线是「scale 要多密」，
  格式差异只是它的一个切面。<br>
  <b>② 别在 §4 讲成「二维更准」。</b> ⛔ 它更粗，赢在一致性 ——&nbsp;
  这是全讲最容易记反的一句。<br>
  <b>③ 别把出处台账念出来。</b> 那是给课后回查用的，
  <em>台上只在被追问时报一个来源名。</em><br>
  <b>④ 别越界讲 TPU 那一侧。</b> ⚠️ 目前写完的内容<b>全是 Blackwell</b>；
  TPU 的低精度是另一套（QAG 在专题二那条线上），<b>这一讲还没并排写</b> ——&nbsp;
  <em>被问到就直说「这一讲这一版只讲了 NVIDIA 侧」。</em>
</div>

</main>
</div>
</body></html>'''

io.open(OUT, "w", encoding="utf-8").write(html)
print("ok  topic-08-lecture.html  %s 字符" % format(os.path.getsize(OUT), ","))
