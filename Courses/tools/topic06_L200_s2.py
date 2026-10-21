# -*- coding: utf-8 -*-
"""专题六 L200 · 第二节「那多拉几个人不就行了？batch 能开多大？」
事实全部取自 L300 2.1、3.1–3.3（topic06_s2.py、topic06_s3.py）；图复用 L300。
⛔ 5.1 倍留到第九节揭晓，这里不报 2,200 → 11,200 的倍数。"""
import topic06_common as C

BODY = '''
  <p class="lead">要多拉人，可人不是一起上车、一起下车的：有人问一句就走，有人要写三千字。
    <b>车怎么排，决定一趟能坐满几个人；座位有多少，决定最多坐几个；坐得越满，每个人越慢。</b></p>

  <h3 id="l2-1">老办法：凑满一车，等最慢的那个</h3>
  <ul>
    <li><b>静态批。</b>一批人一起进出，先答完的座位空着，车照样开。</li>
    <li><b>卡在接口。</b>老引擎只有「整批跑完」一个按钮，没法中途换人。</li>
  </ul>

  <h3 id="l2-2">每站都停：每一步都重新排一次</h3>
__L2_ANIM__
  <p>Orca（OSDI 2022）第一招：<b>每走一步就停下来看一眼</b>。
    后来叫连续批（continuous batching），vLLM、SGLang 都这样排。</p>
  <div class="note info"><span class="t">还是那辆公交车</span>
    静态批像包车，凑满一车一起到站；连续批才是公交：每站都停，到站的下，等车的上。</div>
  <ul>
    <li><b>为什么拼得起来（Orca 第二招）。</b>乘权重不管 token 是谁的，所有人接成一长条一起乘；只有注意力（回头翻笔记）按人分开。</li>
    <li><b>新人上车不堵门。</b>长 prompt 切成小块，每步带一块（chunked prefill），车上出字的人不被卡住。</li>
  </ul>

  <h3 id="l2-3">座位就是 KV：多给座位，不一定多出字</h3>
  <p><b>KV 池装得下几个人，就有几个座位。</b>Little 定律：每卡吞吐 ≈ 座位数 ÷ 两字间隔，座位一满，吞吐就到顶。</p>
  <p>那多给座位呢？两字间隔不会原地不动：</p>
__L2_SEATS__
  <p>权重早已摊薄，新上车的人带来的全是 KV，每一步就长了。</p>

  <h3 id="l2-4">所以 batch 开多大，要先问「每人要多快」</h3>
__L2_PARETO__
  <ul>
    <li><b>同一套系统，一条曲线。</b>每人每秒 68 字，每卡 9,759；要 173 字，只剩 895。</li>
    <li><b>先定档，再比吞吐。</b>「每人每秒几个字」叫交互档；别人报的大数，可能只是站得更左。</li>
    <li><b>线上真有人选左边。</b>DeepSeek 线上每个用户只有 20–22 字每秒，换来每节点（8 卡）约 14,800；硬件和模型都不同，别和上面的每卡数比。</li>
  </ul>

  <div class="note ok"><span class="t">这一节带走</span>
    <b>连续批让座位不空着；座位数由 KV 定；人越多，每人越慢。</b>
    这一步进步多少：Orca 论文里，同样每字 190 毫秒的延迟，GPT-3 175B 每秒能服务的请求是 FasterTransformer 静态批的 36.9 倍。<br>
    座位既然是 KV，下一节就问：每人那块 KV 到底多大，能不能小一点？<br>
    <em>想深挖：完整版第三节。</em></div>
'''


ANIM = '''<figure class="fbox fwide" id="anim-contbatch">
<video src="media/topic06-contbatch.mp4" autoplay loop muted playsinline
       aria-label="静态批与连续批动画。上下两组各 4 个槽位；上组凑满一批、等最长的做完，下组每一步都重新排、谁做完谁让位。每一列是一步，灰格是槽位空转。最后：同样 15 步，静态批做完 8 个（空转 23 格），连续批做完 14 个（空转 0 格）。"></video>
<figcaption><b>比两组的灰格：上面越走越多，下面一格没有。</b>
  <span class="sub">（18 秒无声循环。请求长度是示意，每步按等长算。）</span></figcaption></figure>'''

BODY = BODY.replace("__L2_ANIM__", ANIM)

FIGS = {
    "__L2_SEATS__": ("fig-seats", "fig6-seats.svg", "topic06-fig-seats.py",
        '<b>先比最上和最下两根：人多了一成多，吞吐几乎没动。</b><br>'
        '<em>同一组实测，只调了 KV 池的显存。</em>'),
    "__L2_PARETO__": C.l300_fig("fig-s2-pareto",
        '<b>沿着绿线往左走，每卡吞吐往上涨，每个人拿到的字变慢。</b><br>'
        '<em>只有绿色三点是同一条曲线；其余点分属不同框架和配置。</em>'),
}

NUM_LOCK = []
