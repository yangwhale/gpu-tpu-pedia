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
    <li><b>拆开还能各挑切法</b>：decode 开大 EP、开 DCP（第四、五节）。</li>
  </ul>
  <p>DistServe 报告每卡请求率最多是不拆的 7.4 倍（九成以上请求守住延迟）。Mooncake 做进 Kimi 线上，NVIDIA Dynamo 2026 年 3 月出 1.0 版，PD 分离成了标准件。</p>

  <h3 id="l6-3">多搬的那一趟有多贵</h3>
__L6_TRIP__
  <ul>
    <li><b>很便宜。</b>Qwen3-Coder-480B 一个 8K prompt 约 100 ms，只占 prefill 的 5–10%（按带宽估算）。</li>
    <li><b>单趟便宜，不等于总量便宜。</b>一台 prefill 每秒外送一串请求的 KV，一条专线并不了几台（下图）。KV 厚的，别出 NVLink 域。</li>
    <li><b>前提是网络跑在设计带宽上。</b>vLLM 在 GB300 上跑 V4-Pro，配置漏了，绕道主机内存，每秒只走约 200 MB，一趟 5 秒多；配对后快两个数量级。配错不报错，只是首字慢。</li>
  </ul>
__L6_PRFAAS__
  <p><b>上线前两件事：先量 KV 传输带宽，再比字。</b>temperature 设 0，拆与不拆逐 token 对：前几个就分叉多半是 bug，几百个后才分叉可能是浮点差，分不清降到并发 1 再比。PD 出错常是能跑、通顺，但和不拆时不同。</p>

  <h3 id="l6-4">所以呢：配几台，什么时候别拆</h3>
  <ul>
    <li><b>配比先算起点，再扫。</b>起点 ≈ decode 每秒完成的请求数 × 输入长度 ÷ 一个 prefill 每秒吞的 token。V4-Pro（GB300、8 张 decode 卡）算出约 9.8 个，官方配方 8 个，runbook 扫出 14 个；14 加到 16，每张 decode 卡输出只涨 2%，已喂饱。</li>
    <li><b>再加一台 prefill，买的是首字。</b>每个请求端到端约 65 s，首字省下的 0.67 s 摊进去看不出，总输出不涨。</li>
  </ul>
__L6_2P1D__
  <ul>
    <li><b>两种活本来不打架，拆开是白付代价。</b>Qwen3-Coder 在两台 v7x-8 上单并发，拆成 1P1D 后首字中位数 95 → 281 ms，输出只多 12%。</li>
    <li><b>别照搬别人的配比</b>，拿自己的负载量。</li>
  </ul>

  <div class="note ok"><span class="t">这一节带走</span>
    <b>PD 分离不让 decode 少搬，只让它不被打断；两边各按上限开批。多搬的 KV 一个请求只传一次，前提是网络跑在设计带宽上。</b><br>
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
        '<em>KV 用时按带宽估算，不是计时；图底那句的「第三节」指专题五。</em>'),
    "__L6_PRFAAS__": C.l300_fig("fig-s8-prfaas",
        '<b>先看两根橙柱只够并 1–2 台，再看两根绿柱能并 12–21 台。</b><br>'
        '<em>32K 输入，一台 8×H200 每秒外送的 KV；「够并几台」是 100 Gbps 除以每台外送，L300 的推导。</em>'),
    "__L6_2P1D__": C.l300_fig("fig-s9-2p1d",
        '<b>左边总输出几乎没动、每卡少了三成；右边两组首字都在降。</b><br>'
        '<em>每卡 ＝ 总输出 ÷ 全部卡数；首字是客户端测的，含排队。</em>'),
}

NUM_LOCK = []
