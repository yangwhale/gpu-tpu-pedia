# -*- coding: utf-8 -*-
"""专题六 · 第十节「先选拓扑，再调参数」。"""

BODY = '''
  <p class="lead">第六到九节给出了一串拓扑，每换一个，调好的参数就得重找。<b>换一次拓扑，抵得上把参数调遍。</b></p>

  <h3 id="s10-1">10.1　回顾：线有快慢，切法先定</h3>
  <p>专题五 <a href="topic-05.html#s7-1">§7.1</a> 讲过：通信勤的切法要放在快线里，懒的才能跨慢线。</p>
__FIG_FREQ_T5__

  <h3 id="s10-2">10.2　「+45% 对 +347%」：先看分母</h3>
  <p>这两个数出自 V4-Pro 的部署记录，都是<b>总吞吐</b>，可图里三档配置用的卡数各不相同。</p>
__FIG_S10_LENS__
  <p>拆开看，+45% 里大半是加卡：prefill 从 1 台加到 3 台，卡数翻倍，每卡反而变少。卡数不变、只调并发的那一步才是纯调参。</p>
  <p>换拓扑在两种口径下都赢：按每卡算，换 DEP 是 TP4 的 2.09 倍（第六节拆过为什么），只调并发是 +28%。dep8 再拉高并发时每卡最高，但首字 95 s，已出了交互档。</p>

  <h3 id="s10-3">10.3　拓扑约束是硬的</h3>
  <p>16 个 device 跑 TP16，机间走 DCN 还是 ICI，结果差出一截。即便走 ICI，两台的总输出 513.8 tok/s 也不如单机 602。</p>
__FIG_S10_WIRE__
  <p>TP 每层都要把各卡的部分和加起来，跨慢线就层层要等；原记录建议开两份单机做 DP。</p>
  <p>GPU 上同理。V4-Pro 的 SGLang 部署要求 16 个节点都在同一个 NVL72 域：PD 之间传 KV 走 NVLink，跨域退化成 RDMA，原文只说「显著变慢」。</p>

  <h3 id="s10-4">10.4　TPU 上，device 编号就是拓扑</h3>
  <p>JAX 的 mesh 是一张 device 列表排成的网格。<b>列表怎么排，决定了沿哪一轴通信时哪几个 device 一组。</b></p>
__FIG_S10_RANK__
  <p>所以 mesh 交给 <code>mesh_utils.create_device_mesh</code> 按物理位置生成，跨切片用 <code>create_hybrid_device_mesh</code> 把 ICI 和 DCN 分成两层。</p>

  <h3 id="s10-5">10.5　最小切片怎么定</h3>
  <p>一个引擎最少几颗芯片，先算纸面账：权重 ＋ KV ＋ 激活值（中间结果）的显存。拿 DeepSeek-V3 在 TPU v7 上算（口径同 4.2）。</p>
__FIG_S10_SLICE__
  <p>到 128K、256 并发，KV 约 2,196 GiB，是权重的 3.5 倍。要装下只有三条路：加芯片；KV 存 FP8（第十二节末尾）；冷 KV 下放主机内存（4.6）。DCP 帮不上：这里 KV 本就只存一份（DP 注意力），DCP 只把一条序列摊开，而一条 128K 才 8.58 GiB。</p>
  <p>纸面最少 4 颗，实测却放不下：R1（和 V3 同结构）的 FP8 专家每个 device 要约 101.8 GB，超出 94.75 GiB 上限，比按 8 份均摊多约 20 GB（原记录没拆原因），v7x-8 上实际用 FP4（§12.4）。</p>
  <div class="note warn"><span class="t">本课推导的边界</span>纸面账只给下限，激活值预留 10% 也是假设；用一次真实启动看实测的 HBM 峰值。</div>

  <h3 id="s10-6">10.6　容量别拿小并发估</h3>
  <p>R1 在 GB300 上的部署记录先估了这套 DEP8 decode 的容量，起初却只拿并发 8 去量。</p>
__FIG_S10_CONC__

  <h3 id="s10-7">10.7　对照实验的纪律</h3>
  <ul>
    <li><b>先固定拓扑，再扫参数。</b>拓扑一变瓶颈就换边（6.4），错拓扑上调出的最优参数搬不到对的拓扑上。</li>
    <li><b>一次只动一个变量。</b>同时改并发和 prefill 台数，涨了说不清是谁的功劳。</li>
  </ul>

  <h3 id="s10-8">10.8　五步选法落到两个例子</h3>
  <p>专题五 <a href="topic-05.html#s7-4">§7.4</a> 的五关，推理时照样走一遍。PP 是流水线并行：按层切给几组卡。</p>
__FIG_S10_GATES__

  <p class="landing">⭐ 拓扑定了那一趟搬运走哪条线、能分给多少人，参数只在这个上限底下挪动。<b>先选拓扑，再调参数；比较时先看分母。</b>留下的问题：这些都靠把 batch 做大，交互档压住 batch 时，一趟还是只出一个字（第十一节）。</p>
'''

FIGS = {
    "__FIG_FREQ_T5__": ("fig-freq-t5", "fig5-freq.svg", "topic05-fig-map.py",
        '<b>通信越勤的切法，越要放在快线里。</b><br><em>专题五原图，复用。这是训练口径；推理只有前向，FSDP 和梯度那几行不适用。</em>'),
    "__FIG_S10_LENS__": ("fig-s10-lens", "fig6-s10-lens.svg", "topic06-fig-topo.py",
        '<b>左右两栏只换了分母，看「调参」那一步。</b><br><em>卡数含 prefill 与 decode。</em>'),
    "__FIG_S10_WIRE__": ("fig-s10-wire", "fig6-s10-wire.svg", "topic06-fig-topo.py",
        '<b>device 数相同时，决定快慢的是那条线。</b><br><em>口径：只数输出 token；负数是比单台机器慢多少。右图：同样两台 v7x-8，机间走 DCN 还是编进同一个 ICI 切片。</em>'),
    "__FIG_S10_RANK__": ("fig-s10-rank", "fig6-s10-rank.svg", "topic06-fig-topo.py",
        '<b>看编号顺序怎么决定环的走法。</b><br><em>真实 v7 是 3D torus。</em>'),
    "__FIG_S10_SLICE__": ("fig-s10-slice", "fig6-s10-slice.svg", "topic06-fig-topo.py",
        '<b>六种配置里四种落在 8 颗；最长最满那行跳过 16，要取 32。</b><br><em>KV 按 bf16 计；纸面下限。</em>'),
    "__FIG_S10_GATES__": ("fig-s10-gates", "fig6-s10-gates.svg", "topic06-fig-topo.py",
        '<b>没有哪列更对：同一道关，硬件和模型变了，答案跟着变。</b><br><em>每格是本课按部署记录归纳的结论。</em>'),
    "__FIG_S10_CONC__": ("fig-s10-conc", "fig6-s10-conc.svg", "topic06-fig-topo.py",
        '<b>看红点离灰虚线多远，再看虚线右边那段线有多平。</b><br><em>口径：并发是压测客户端同时挂着的请求数，不是服务端实际批大小。</em>'),
}

NUM_LOCK = [
    ("+28%", "fig6-s10-lens.svg"),
]
