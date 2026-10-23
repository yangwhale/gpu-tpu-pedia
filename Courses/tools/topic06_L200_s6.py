# -*- coding: utf-8 -*-
"""专题六 L200 · 第六节「PD 分离：多搬一趟 KV，凭什么接的人更多？」
事实全部取自 L300 8.1、8.2、8.4、9.1、9.2（topic06_s8.py、topic06_s9.py）和讲义第八、九节（topic06_lec_4.py）；图和动画复用 L300。
⭐ 主线是「两种活互相拖」：一个灶台备菜、一个灶台出菜（跟专题五同一个比方）；分开后两边各按自己的上限开批，谁也不等谁。
⛔ 5.1 倍留到第九节揭晓。"""
import topic06_common as C

BODY = '''
  <p class="lead">上一节的洞：prefill 和 decode 还挤在同一批卡上，每卡装的人越多，一个长 prompt 进门耽误的人越多。
    PD 分离把两种活拆到两批机器上，代价是每个请求多搬一趟 KV。<b>多搬一趟，凭什么接的人反而更多？</b></p>

  <h3 id="l6-1">两种活互相拖</h3>
__L6_ANIM__
  <div class="note info"><span class="t">一个厨房：备菜和出菜</span>
    毛病不在一趟搬多少，在两种活抢同一个灶台。prefill 是备菜，整段 prompt 一次备完；decode 是出菜，一盘一盘往外端，一次多端几盘才划算。突然来一桌大单、一大堆菜要备，出菜全停。<br>
    <b>拆开就是两个灶台，备好的料（KV）递过去一趟。</b></div>
  <ul>
    <li><b>分块 prefill 只是摊平。</b>长 prompt 切几段夹进去，每步少等一点，两种活还在抢同一批卡。</li>
    <li><b>拆开不改 decode 每一趟搬多少。</b>只保证这一趟不被打断，才敢把批攒满。</li>
  </ul>

  <h3 id="l6-2">凭什么接的人更多：各按各的上限</h3>
__L6_RATIO__
  <ul>
    <li><b>为什么只有 1.6</b>：延迟要求其实是两条，首字别等太久、字距别卡；混在一起两头都得防，只能少收。</li>
    <li><b>两边都跑到自己的上限</b>：两张 prefill 配一张 decode，3 张卡接 10 个；混着做 7 张才接约 11 个，每卡翻一倍多。</li>
  </ul>
  <p>DistServe 报告每卡请求率最多是不拆的 7.4 倍（九成以上请求守住延迟）。Mooncake 做进 Kimi 线上，NVIDIA Dynamo 2026 年 3 月出 1.0 版，PD 分离成了标准件。</p>

  <h3 id="l6-3">多搬的那一趟有多贵</h3>
__L6_TRIP__
  <ul>
    <li><b>很便宜，是个零头。</b>Qwen3-Coder-480B 一个 8K prompt 递 KV 约 100 ms，只占 prefill 的 5–10%（按带宽估算）。</li>
    <li><b>单趟便宜，不等于总量便宜。</b>一台 prefill 每秒外送一串请求的 KV，一条专线并不了几台（下图）。KV 薄的才敢把 prefill 放到别的机房；KV 厚的，别出 NVLink 域。</li>
  </ul>
__L6_PRFAAS__

  <h3 id="l6-4">拆开之后：各怎么切、配几台、什么时候别拆</h3>
  <div class="note info"><span class="t">两边各怎么切：prefill 窄而多份，decode 宽而一份</span>
    我们自己在 GB300 上用 SGLang 搭的 V4-Pro，16 台机器，<b>两头都是 DEP</b>（注意力各卡带自己那批人，专家按 EP 分），差在宽度和份数：
    <ul>
      <li><b>prefill：14 份 dep4。</b>一台机器 4 张卡一组，组和组互不说话。prefill 卡在算，一个长 prompt 自己就喂饱算力，不用凑人；组小，跨机通信少；要接更多，多开几份。</li>
      <li><b>decode：1 份 dep8。</b>横跨两台机器 8 张卡。decode 卡在搬，得凑大批，专家权重读一遍才划算；EP 开宽，每个专家攒到的人更多；注意力走 DP，V4-Pro 只有 1 个 KV 头，每人的 KV 只存一份不复印。</li>
      <li><b>宽了要有人喂。</b>decode 宽到 dep16，prefill 只剩 12 台、喂不饱，每张 decode 卡输出反掉四成（第四节）。</li>
      <li><b>不只我们这么配。</b>SGLang 在 GB200 NVL72 上的公开实测也是这个形状：decode 48 卡一个实例（DP 注意力＋EP48），prefill 每实例 2–4 卡（完整版 8.5）。</li>
      <li><b>换模型，组合会变。</b>DEP 起不来的，decode 退回 TEP 再叠 DCP（K3，第五节）；首字太慢的，prefill 可以加 PCP。这两条是按前两节推的配法，我们这套部署没用到。</li>
    </ul></div>
  <ul>
    <li><b>配比先算起点，再扫。</b>起点 ≈ decode 每秒完成的请求数 × 输入长度 ÷ 一个 prefill 每秒吞的 token。V4-Pro（GB300、8 张 decode 卡）上，公式、官方配方、我们实扫三个数都不一样（约 9.8、8、14 个）；14 加到 16 只涨 2%，说明已喂饱。拓扑一换就重扫。</li>
    <li><b>再加一台 prefill，买的是首字。</b>每个请求端到端约 65 s，首字省下的 0.67 s 摊进去看不出，总输出不涨。</li>
  </ul>
__L6_2P1D__
  <ul>
    <li><b>两种活本来不打架，拆开是白付代价。</b>Qwen3-Coder 在两台 v7x-8 上单并发，拆成 1P1D 后首字中位数 95 → 281 ms，输出只多 12%。</li>
    <li><b>别照搬别人的配比</b>，拿自己的负载量。</li>
  </ul>

  <div class="note ok"><span class="t">这一节带走</span>
    <b>PD 分离不让 decode 少搬，只让它不被打断；两边各按上限开批。多递的那趟 KV 一个请求只传一次，单趟很便宜；拆开后两边还能各挑切法。</b><br>
    这一步进步多少：Mooncake 的 FAST'25 正式版报告，比此前系统多接 115%（A800）、107%（H800）请求。<br>
    可拆不拆，decode 还是一步只出一个字。下一节：能不能一趟出好几个字？<br>
    <em>想深挖：完整版第八、九节。</em></div>
'''

ANIM = '''<figure class="fbox fwide" id="anim-pdsplit">
<video src="media/topic06-pdsplit.mp4" autoplay loop muted playsinline
       aria-label="动画：上下两组时间轴，一格是一步 decode 的时间。上组是混部，下组是 PD 分离。前 4 步两组一样，请求 A、B、C 每步各出一个字。接着来了一个长 prompt，橙色 prefill 占了 5 格：上组它占住整台机器，A、B、C 这 5 格一个字都没出，被红框圈出；下组它在 prefill 机上算，decode 机照常每步出字。算完后下组一支紫色箭头「传 KV」从 prefill 机指向 decode 机，新请求下一步加入 decode 批。结尾：拆开后字距始终均匀，代价是多一批机器、每个请求多传一趟 KV。"></video>
<figcaption><b>先看上组的红框，再看下组那支紫箭头：一个是整批人在等，一个是多搬一趟。</b>
  <span class="sub">（22 秒无声循环，Manim 渲染。时间格只表相对长短，不是实测毫秒；传 KV 画得比 prefill 短，量级见下文。）</span></figcaption></figure>'''

BODY = BODY.replace("__L6_ANIM__", ANIM)

FIGS = {
    "__L6_RATIO__": C.l300_fig("fig-s9-pd-ratio-t5",
        '<b>上排七张卡都「两样都做」，下排三张卡各干一样。</b><br>'
        '<em>专题五原图；请求率取自论文里的例子。</em>'),
    "__L6_TRIP__": C.l300_fig("fig-s8-kv-trip-t5",
        '<b>先看上面那根一到两秒的长条，再看下面那根三色细条。</b><br>'
        '<em>KV 用时按带宽估算，不是计时。</em>'),
    "__L6_PRFAAS__": C.l300_fig("fig-s8-prfaas",
        '<b>先看两根橙柱只够并 1–2 台，再看两根绿柱能并 12–21 台。</b><br>'
        '<em>32K 输入，一台 8×H200 每秒外送的 KV；「够并几台」是 100 Gb/s 除以每台外送，本课推导。</em>'),
    "__L6_2P1D__": C.l300_fig("fig-s9-2p1d",
        '<b>左边总输出几乎没动、每卡少了三成；右边两组首字都在降。</b><br>'
        '<em>每卡 ＝ 总输出 ÷ 全部卡数；首字是客户端测的，含排队。</em>'),
}

NUM_LOCK = []
