# -*- coding: utf-8 -*-
"""专题三 · 注意力演进 —— 教师讲义（授课稿）。

CSS 沿用专题一讲义（跟专题二、专题八三份一致）。⛔ 不要另写 CSS。

════════════════════════════════════════════════════════════════
⛔ 边界：课件里 🚧 的那几节，这里同样只给「打算怎么讲」
════════════════════════════════════════════════════════════════
第一节（FlashAttention）有逐字稿 ——&nbsp;它在课件里是成品。
其余各节课件本身还是「已展开的大纲」，**讲义里不许编逐字稿**：
编一段，现场你会照着念。

⭐ 这一讲的讲义有个别人没有的难点：**第一节又长又硬，而它排在最前面。**
   讲砸的方式是把它讲成「FlashAttention 科普」——&nbsp;
   台下坐的是有经验的人，他们要的是**天花板在哪**，不是原理。
   所以讲义里反复提醒：**原理快走，天花板慢走。**
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-01-lecture.html")
OUT = os.path.join(WEB, "topic-03-lecture.html")
DECK = "topic-03.html"


def _css():
    h = io.open(CSS_SRC, encoding="utf-8").read()
    i = h.index("<style>")
    return h[i:h.index("</style>") + len("</style>")]


html = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>专题三 · 讲义（授课稿）</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#129408;</text></svg>">
<meta property="og:title" content="注意力演进 · 讲义">
<meta name="theme-color" content="#5f6368">
''' + _css() + '''
</head>
<body>

<!-- ⛔ 这个文件由 Courses/tools/topic03-build-lecture.py 生成，不要手改。 -->

<div class="top"><div class="in">
  <span class="who">专题三 · <b>注意力演进</b></span>
  <nav class="tabs">
    <a href="''' + DECK + '''">课件</a>
    <a href="topic-03-lecture.html" class="on">讲义</a>
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
  <span class="k">🚧 只有第一节有逐字稿</span>
  <b>第一节（FlashAttention）在课件里是成品</b>，讲义给足。<br>
  <b>其余各节课件本身还是「已展开的大纲」</b> ——&nbsp;
  这里只给「打算怎么讲」，<b>不编逐字稿</b>。
  ⛔ <em>编一段，现场你会照着念。</em>
</div>

<h3 class="sec">这一讲要留下的一句话</h3>
<div class="say">
  <span class="pause">整讲只留这一句：
    <b>「名词一年换一批，尺子不换 ——&nbsp;看到任何一个新变体，
    先问它在拧哪个旋钮、省了什么、赔了什么。」</b></span>

  <p>⭐ <b>而在拧旋钮之前，得先把另一条路走完</b>：
    <em>「怎么算」这条路已经到头了（第一节），
    <b>所以剩下的省法只能去改「算什么」</b>（第二节起）。</em>
    <b>这个顺序是这一讲的骨架，不要打乱。</b></p>
</div>

<h3 class="sec">讲稿 · 1 FlashAttention（唯一有逐字稿的一节）</h3>
<div class="say">
  <span class="board">课件第一节。<b>这一节占 12 分钟，是全讲最长的一节。</b></span>

  <span class="pause">⛔ <b>先立规矩，这一节最容易讲砸的方式是把它讲成科普。</b><br>
    <b>原理快走，天花板慢走。</b>
    <em>台下要的是「它的极限在哪」，不是「它怎么工作」——&nbsp;
    后者他们多半已经知道。</em></span>

  <p><b>开口一句话定位</b>：<em>三个旋钮改的是<b>算什么</b>，
    FlashAttention 一个字都不改数学，它改的是<b>怎么算</b>。</em></p>

  <p><b>然后快走原理</b>：朴素写法三步，中间那个 L×L 的 S 每一步都要落一趟 HBM，
    三步四趟；128K 单头就是 32 GiB。<b>融成一个 kernel，S 整项消失。</b><br>
    <em>⭐ 顺带把可迁移的那句说出口：<b>FLOPs 一分不省，省的全在
    「中间产物不落地」这一行</b> ——&nbsp;这是所有算子融合共有的形状。</em></p>

  <span class="board">滚到 1.2b 那两段伪代码。<b>这里开始慢。</b></span>

  <p><b>那张经典图画的是第一版的循环顺序</b>：外层 K／V，内层 Q。
    <em>指着内层那三行说：<b>O 和那两个统计量每一轮外循环都要读进来、写回去一次。</b></em><br>
    <b>为什么躲不掉？</b>——&nbsp;<em>因为下一块 K／V 还会碰到同一个 Q 块。</em></p>

  <p><b>第二版把两层对调</b>，两个后果：
    <em>① O 整个内循环待在片上，一次都不落 HBM；
    ② 不同 Q 块彻底独立，铺到几百个执行单元上互不通信。</em></p>

  <span class="pause">⭐ 这句要说出口，它是本节最可迁移的一句：
    <b>「外循环放谁，谁的中间状态就不用来回搬。」</b></span>

  <p><b>接着讲 warp 那一层</b> ——&nbsp;<em>同一个原则又用了一遍：
    第一版切 K／V，warp 之间要写共享内存、同步、再加起来；
    第二版切 Q，<b>warp 之间完全不用通信</b>。</em></p>

  <span class="pause">⭐⭐ 两个尺度合成一条判据，慢慢说：
    <b>「切『要被累加的那一维』就得合；切『各自独立出结果的那一维』就不用合。」</b><br>
    <em>再补一句它的适用面：<b>任何融合 kernel 分工时，
    先问『我切的这一维在不在求和号里』。</b></em></span>

  <span class="board">滚到 1.3。</span>

  <span class="pause">⛔ <b>这里一定会有人在心里嘀咕「不就存两个数吗」</b> ——&nbsp;
    <b>主动拆开，别等人问</b>：<br>
    <b>「那两个 running 值确实很小，每行各一个标量。
    真正占地方的是 S 那一整块 ——&nbsp;它是 [bq, bkv]，
    512 见方就是 1 MB，2048 见方是 16 MB。」</b></span>

  <p><b>然后讲生命周期</b>：<em>S 要连着走完「产出 → 求最大值 → 减完取指数 →
    喂给第二个矩阵乘」四步才能扔，而中间两步是向量单元的活、两头是矩阵单元的活
    ——&nbsp;<b>矩阵单元想开工，输出却还被向量单元占着</b>。</em><br>
    <b>再加流水线要重叠，就得同时留住好几块 S。</b></p>

  <span class="board">滚到 1.5 那张块大小的表。<b>这是我们自己的数，可以慢慢讲。</b></span>

  <p><b>三堵墙，一堵一堵指</b>：<em>往上撞容量（先爆反向）；
    往上还先撞<b>并行度</b>（KV 方向只剩一块，流水塌）；往下撞碎块开销。</em></p>

  <span class="pause">⭐ 最反直觉的那句，停一下：
    <b>「装得下，却更慢。」</b><br>
    <em>然后给可迁移的教训：<b>块要看 block/seq 的比例，不是绝对值 ——&nbsp;
    跨序列长度照抄配置会反向优化。</b></em>
    ⚠️ <b>「块 ≈ seq/2」那条要说清是「尚未验证」</b>，别当定律讲。</span>

  <span class="board">滚到 1.6。<b>这一段是这一节真正的分量，给足时间。</b></span>

  <p><b>三层，顺着讲</b>：
    <em>① 记账口径虚高（FLOP 记的是不折 causal 的全量）；
    ② 形状锁死 50%（<code>head_dim=128</code> 撞 256 的 MXU，两个矩阵乘各撞一次）；
    ③ 寄存器生命周期把 50% 压到 35%（回指 1.3）。</em></p>

  <span class="pause">⛔ 这句必须说，否则学员会以为「调调参就能好」：
    <b>「三层没有一层是配置能救的 ——&nbsp;要么改 head_dim，要么改 kernel 的数据流。」</b></span>

  <span class="pause">收口那句，它是通往第二节的桥：
    <b>「『怎么算』这条路到这里基本走到头了。中间产物已经不落地、
    块大小已经贴着三堵墙，而算子仍然只跑到三成多 ——&nbsp;
    剩下的空间只能去改『算什么』。」</b></span>
</div>

<h3 class="sec">🚧 讲稿 · 2 起：三个旋钮与各家的选择</h3>
<div class="say">
  <span class="board">课件第二节起。<b>只有提纲，没有逐字稿。</b></span>
  <p><b>骨架</b>：<em>先给三个旋钮那张收纳表（每个token存多少 / 每个query看多少 /
    换一套数学），把所有名词一次收进去；再逐支展开；
    最后用代价对照表和时间线收尾。</em></p>
  <span class="pause">⭐ 讲这几节时反复回到同一个动作：
    <b>「拿到一个新名字，先判它在哪一格。」</b>
    <em>这一讲的价值是那把尺子，不是那些名字。</em></span>
</div>

<h3 class="sec">别讲什么</h3>
<div class="warn">
  <span class="k">这一讲最容易跑偏的四个方向</span>
  <b>① 别把第一节讲成 FlashAttention 科普。</b>
  ⛔ 原理快走、天花板慢走 ——&nbsp;<em>台下多半已经知道它怎么工作。</em><br>
  <b>② 别把「不落 HBM」说成「省了计算」。</b>
  <em>FLOPs 一分不省，这句说错，后面屋脊线那套判据全歪。</em><br>
  <b>③ 别把块大小讲成「调参技巧」。</b>
  <em>它是三堵墙夹出来的，而且最反直觉的一堵是并行度不是容量。</em><br>
  <b>④ 别在第二节起逐个念名词。</b>
  ⚠️ <b>名词一年换一批</b> ——&nbsp;<em>讲不完是正常的，讲不出尺子才是失败。</em>
</div>

</main>
</div>
</body></html>'''

io.open(OUT, "w", encoding="utf-8").write(html)
print("ok  topic-03-lecture.html  %s 字符" % format(os.path.getsize(OUT), ","))
