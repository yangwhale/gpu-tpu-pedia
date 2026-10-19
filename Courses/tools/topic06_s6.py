# -*- coding: utf-8 -*-
"""专题六 · 第六节「切专家与注意力：TEP 和 DEP」。图在 topic06-fig-dep.py 和 topic06-fig-bigep.py（大 EP 两张）；动画 BigEP 在 manim/topic06-anim-serve.py；另复用专题五两张图。"""

BODY = '''
  <p class="lead">专家要多凑几行，就得铺到更多卡上，何况 V3 一张卡本来就装不下。切多卡时专家都用 EP，分歧在注意力：几张卡合算一批请求，还是各算各的。
    <b>选哪个，看模型有几个 KV 头、要多大 batch。</b></p>

  <h3 id="s6-1">6.1　注意力怎么切：TP 还是 DEP</h3>
  <p><a href="topic-05.html#s4-4">专题五 §4.4</a> 讲过：同一批卡进注意力和进专家时可用两套切法。
    <b>TEP</b> 是注意力用 TP、专家用 EP；<b>DEP</b> 是注意力用数据并行、专家用 EP。</p>
__FIG_S6_FOLD_T5__
__FIG_S6_TEPDEP_T5__

  <p><b>判据：KV 头数和 batch。</b>TP 按头切注意力。KV 头只有 1 个时切不开，<b>TP 开几路，每个请求的 KV 就存几份，每步也读几遍</b>。DEP 各存各的、各读各的。</p>
  <p>DeepSeek-V4-Pro 不是 MLA，但每层同样只存一条所有头共享的 KV（<code>num_key_value_heads: 1</code>）。
    同样并发 512，decode 从 TP4 换成 dep8（TP1 ＋ 注意力 DP8 ＋ EP8），每卡吞吐<b>翻了一倍多</b>。</p>
  <p>原记录归功于容量：副本让出的显存多收了请求。用 Little 定律（3.3）核一下：</p>
__FIG_S6_KV__
  <p><b>变的是步时</b>：除了图里的重复读和 AllReduce，EP 从 4 路宽到 8 路，每卡读的专家也减半（三项各占多少 runbook 没拆，本课推导）。</p>
  <p><b>换模型</b>：MLA 开 TEP8，KV 存 8 份；Qwen3.5-397B 全注意力层 2 个 KV 头，存 4 份；8 个 KV 头的 GQA 每卡正好一个头，DEP 在 KV 上不占便宜。</p>

  <p><b>DEP 的代价：每个 rank 都得自己备一份。</b>DEP 下每个 rank 是台小推理机，注意力权重、调度器、状态各一份。<b>按请求开、又不随 DP 切的状态，DEP 一宽就成倍涨。</b></p>
__FIG_S6_STATE__
  <ul>
    <li><b>K3 上 DEP8 起不来</b>：每 rank 存整池 KDA 状态，加载即显存不足（OOM）；官方改推 TP、DCP（<a href="#s7-1">第七节</a>）。</li>
    <li><b>忙闲不均</b>：专家那步等最慢 rank，闲的陪跑空前向；热门专家多备几份是 EPLB（<a href="#s6-6">6.6</a>）。</li>
  </ul>

  <p><b>DEP 吃得快了，就得有人喂。</b>decode 步短了，瓶颈挪到 prefill：dep8 下每加一台 prefill，decode 吞吐就涨（<a href="#s9-2">9.2</a>）；换成 TP4 decode，加第 4 台 prefill，总吞吐 21,100 → 20,479，纹丝不动。<b>拓扑一换，瓶颈换边，P:D 配比要重扫。</b></p>

  <h3 id="s6-2">6.2　TPU 上也一样，先读懂参数名</h3>
  <p>tpu-inference 一个公开 PR 给带线性注意力的混合模型接上了 DP 注意力。
    Qwen3.5-397B 在 v7x-8 上比了 DP=8 和 TP=8：</p>
__FIG_S6_TPU__
  <p>decode 为主那条在 512 猛涨，原因可能是 8 路 TP 把全注意力层的 KV 复制了 4 份（6.1，本课推导，PR 没归因），也可能 TP 基线撞了参数上限：TP 从并发 256 到 512 只涨 3%（4,130 → 4,266），PR 又没给 TP 的完整命令。并发 64 时 decode 为主那条打平：KV 放得下，复制无妨；prefill 为主那条已领先，长 prompt 更早挤满显存。这套切法定下之后，MoE 的通信路径还要一刀刀砍，见 <a href="#s14-3">14.3</a>。</p>
  <p>tpu-inference 跑 DeepSeek-R1 的命令写着 <code>--tensor-parallel-size 8</code>，
    <b>那其实是总 device 数</b>；真正的切法在 additional-config 里：TP=1、开 DP 注意力、EP=8，即 DEP8。</p>

  <p><b>名字别照字面套。</b>TEP／DEP 是 TRT-LLM 的叫法，不等于 Megatron（训练框架）的 ETP／EDP，后者指专家层内再分的 TP 组和 DP 组。prefill、decode 各用哪个，vLLM 部署 K3 和 TRT-LLM 的例子正好相反（<a href="topic-05.html#s8-6">专题五 §8.6</a>）。</p>

  <h3 id="s6-3">6.3　大 EP：是什么，为什么省</h3>
  <p>EP 是把专家分给不同的卡（<a href="topic-05.html#s4-4">专题五 §4.4</a>）。一台 8 卡机器里分完就是 EP8，token 走机内 NVLink。<b>出了一台机器还在分，就是大 EP</b>；TensorRT-LLM 划的线是 EP 超过 8。注意力不跟着切专家，各卡各管自己那批请求，所以大 EP 就是放大版的 DEP。</p>
__FIG_S6_BIGEP_WHAT__
  <ul>
    <li><b>DeepSeek 线上 decode</b>：EP144，18 个节点；256 个专家加 32 份冗余共 288 份，每卡 2 个路由专家加 1 个共享专家。</li>
    <li><b>DeepSeek 线上 prefill</b>：EP32，4 个节点，每卡 9 个路由专家加 1 个共享专家。</li>
    <li><b>两份资料对不上</b>：V3 技术报告写的 decode 是 EP320、40 个节点、每卡 1 个专家；上面是 DeepSeek 开源周 Day 6（2025-03-01）公布的线上配置。</li>
  </ul>

  <p><b>大 EP 为什么省。</b>DeepSeek 自己的说法两句话：EP 把批做大，矩阵算得更有效率；每卡只管一小撮专家，访存少，延迟低。V3 每层 256 个专家只激活 8 个，<b>不铺大，总批凑不起来</b>。动画把这条链拆开：</p>
<figure class="fbox fwide" id="anim-bigep">
<video src="media/topic06-bigep.mp4" autoplay loop muted playsinline
       aria-label="动画：上下两条泳道，上面 EP8、8 张卡，一张卡里摆着 32 个路由专家的小格子；下面 EP144、144 张卡，一张卡里只有 2 个大格子。四列依次长出条形：每个专家分到几行，EP8 是 8 行、EP144 约 128 行；每层读多少权重，1.41 GB 对 88 MB；接着标出每字节用 8 次对 128 次；路由专家占显存，EP8 的 81.7 GB 越过 80 GB 的红线，EP144 只有 5.1 GB，旁边写着剩下装 KV。最后一拍，下面那张卡的 token 跑去其余 143 张卡再跑回来，字幕说代价是每层都要跨机器去一趟、回一趟。结尾打出：专家铺开，每卡读得少，读一次用得多。"></video>
<figcaption>上下两条泳道比同一列：行数越多，每卡读得越少。
  <span class="sub">（24 秒无声循环，Manim 渲染。专家大小、层数照 V3 配置；每卡 32 个请求、负载均匀是示意。）</span></figcaption></figure>
  <ul>
    <li><b>车坐得满</b>：每份专家分到的行数 ＝ 卡数 × 每卡请求数 × 8 ÷ 专家份数（本课推导，负载均匀时）。每卡 32 个请求，EP8 是 8 行，正是 1.2 节算的平均数；EP144 涨到约 128 行。</li>
    <li><b>读得少</b>：V3 一个路由专家约 44 MB（3 × 7168 × 2048，FP8）。每层每卡要读的专家权重从 1.41 GB 降到 88 MB，读进来的每个字节从用 8 次变成用 128 次。</li>
    <li><b>腾出显存</b>：光放路由专家，EP8 每卡要 81.7 GB，80 GB 的卡根本放不下；EP144 只要 5.1 GB，剩下的装 KV，批还能再大。</li>
  </ul>
  <p>实测：SGLang 2024 年 12 月先给 DeepSeek 模型开了 DP attention，decode 吞吐最多是原来的 1.9 倍；后来用 96 张 H100 复现大 EP，输出最多是同卡数纯 TP 的 5 倍，但含<a href="#s8-1">第八节</a>的 PD 分离。</p>

  <h3 id="s6-4">6.4　大 EP 的代价：网络、大批、一卡出事全组陪着</h3>
  <p>大 EP 每层传两趟：先把 token 分发到专家所在的卡（FP8），算完再汇回（BF16）。按 V3 的 7168 维、选 8 个专家，每个 token 最多发出约 57 KB、收回约 115 KB（本课推导，上限）。卡越多，每趟越慢：</p>
__FIG_S6_BIGEP_NET__
  <ul>
    <li><b>DeepSeek 跑在普通 8 卡机器上</b>：机内 NVLink，全集群 InfiniBand 全互连（DeepEP 的测试机每卡一张 400 Gb 网卡）；decode 用 IBGDA，由 GPU 直接发起 RDMA，不经 CPU。</li>
    <li><b>开源栈写明的门槛</b>：llm-d 要求 InfiniBand 或 RoCE 全互连、GPU 发起 RDMA；vLLM 文档的排障一节专门列了 IBGDA 初始化失败。</li>
    <li><b>少跨机器</b>：V3 训练时就限定每个 token 最多去 4 个节点，先过 IB 到目标节点，再走 NVLink 转发给机内各卡；同一节点挑中几个专家，IB 上只走一份。共享专家每卡一份，本地算，不用发。</li>
    <li><b>量不大，难在延迟</b>：一个 token 一步最多发收约 172 KB × 58 层 ≈ 10 MB（本课推导），比它 4K 上下文一步要读的 288 MB KV 小一个多量级。难的是每层都要等一去一回，出一个字要等 116 次。</li>
    <li><b>NVL72 换来的是余量</b>：72 张卡一个 NVLink 域，EP 铺到几十张卡也不出柜、不走网卡。这份余量怎么用，见本小节末尾。</li>
  </ul>
  <p><b>TPU</b>：Ironwood 64 颗一组，用 ICI 直连成 3D 环面，每颗只连 6 个邻居，远处的芯片要经中间芯片转发（本课推导）；组之间靠光交换拼成最多 9,216 颗的系统，更大就得走 DCN。MaxText 有 <code>dcn_expert_parallelism</code> 这个配置项，但跨 DCN 推荐的只有数据并行。</p>
  <p>公开的 TPU 推理目前是 DEP8，专家那步用 all-gather 加 reduce-scatter，不是 all-to-all；跨切片的大 EP 推理还没有公开结果。</p>

  <p><b>代价：要大批，要藏通信，一张卡出事全组陪着。</b></p>
  <ul>
    <li><b>通信露在外面</b>：EP256 一去一回约 554 µs，乘 58 个 MoE 层约 32 ms（本课推导，不重叠的上限）。不藏起来，省下的读权重时间全赔进去，所以要双微批（6.5）。</li>
    <li><b>批必须够大</b>：行数靠总批撑，批小了凑不满车，传的开销照付。SGLang 实测 decode 每卡 32 个 token 时，双微批反而慢 27%。</li>
    <li><b>等最慢的那张卡</b>：DeepSeek 原话，一张卡过载就拖慢整个系统、其余卡空等；热门专家多备几份是 EPLB（6.6）。</li>
    <li><b>爆炸半径大</b>：一个实例上百张卡。Anyscale 指出，一个 DP rank 出故障，整个 DP 组都得下线，最多 128 张卡；TensorRT-LLM 也提醒，没有容错就会抬高线上故障率。</li>
    <li><b>两套 kernel 不能混</b>：DeepEP 的常规模式（吞吐优先）和低延迟模式不能在同一个通信组里同时用，SGLang 因此必须 PD 分离；decode 批小、kernel 启动开销显眼，要靠 CUDA Graph，而常规模式用不了它；TensorRT-LLM 也因 DeepEP 并非所有场景都兼容 CUDA Graph，在 GB200 上自写了通信 kernel。</li>
  </ul>

  <p><b>整柜当一个 EP 域：SGLang 在 GB200 NVL72 上怎么还这几笔账。</b>两篇博客跑的都是 DeepSeek V3/R1、2K 输入，PD 分离（<a href="#s8-1">第八节</a>）。decode 用 48 张卡，命令写 <code>--tp-size 48 --dp-size 48 --enable-dp-attention</code>，其实是 48 路 DP 注意力加 EP48，和 6.2 一样别照字面读。</p>
  <ul>
    <li><b>网络</b>：DeepEP 加了纯 NVLink 模式，把别的卡的显存映射进本卡的虚拟地址空间。延迟低到双微批可以关掉：整批算，kernel 更快；也不会再出现通信比计算长、重叠了还要空等的浪费。</li>
    <li><b>大批</b>：decode 每卡一次跑 768 个请求，48 卡共 36,864 个（复现命令）。原文把「显存大、KV 放得多、批更大」列为提速原因之一；batch 消融也是越大吞吐越高。</li>
    <li><b>再挤 KV</b>：第二篇把 KV 存成 FP8，decode 注意力的访存约减半，KV 里也多装了 token。batch 选到 KV 基本占满：2K 输入 1408、4K 输入 768；2K 输入从 1408 降到 768，性能约掉一成。</li>
    <li><b>少传</b>：专家改用 NVFP4，DeepEP 发出前就量化成 NVFP4，分发流量比 FP8 再减半；汇回和专家的下投影、共享专家细粒度重叠（6.5）。</li>
    <li><b>反着缩</b>：prefill 不往大铺。显存不够时，借 GB200 CPU 与 GPU 之间 900 GB/s（双向）把权重卸到主机内存、提前取回；每个 prefill 实例只用 4 张卡（原精度）或 2 张（低精度）。原文列的好处：通信少、出事波及的卡少、少等最慢那张；最优 EP 大小要看 kernel 和模型。</li>
  </ul>
  <p><b>换来多少（每卡，对 H100 上的大 EP 部署）</b>：第一篇 decode 7,583 tok/s，2.7 倍；第二篇用 FP8 注意力加 NVFP4 专家，prefill 26,156、decode 13,386 tok/s，3.8 倍和 4.8 倍，原精度也有 18,471 和 9,087。</p>
  <p>这些倍数混着显存、精度、kernel、PD 分离，不全是大 EP 的功劳；decode 是先压慢、攒满请求再放开测的。H100 基线有一组开了模拟 MTP、GB200 没开；原文还说许多 kernel 没吃满带宽和算力。</p>

  <h3 id="s6-5">6.5　双微批：一批拆两半，一个传、一个算</h3>
  <p>大 EP 每层要传两趟：先把 token 分发到专家所在的卡，算完再汇回。传的时候，计算单元空等。DeepSeek 把一批拆成两个微批：A 在传，B 就在算。SGLang 叫它 TBO，vLLM 叫 DBO，都是这一招。</p>
  __FIG_S6_TBO__
  <ul>
    <li><b>prefill 最赚</b>：SGLang 实测，每卡 token 数相同时快 27%～35%；拆开还省显存，每卡能塞 16,384 个 token，比不拆时调到最好再快 40.5%。</li>
    <li><b>decode 看批多大</b>：每卡 256 个 token 时快 25.5%，32 个时反而慢 27%，半批太小，kernel 吃不饱。vLLM 默认一批纯 decode 不超过 32 个 token 就不拆。</li>
    <li><b>decode 拆得更细</b>：各步长短不一，DeepSeek 把注意力再切成两段，排成 5 段流水。</li>
    <li><b>各自仍是串行</b>：一个微批内部，路由、分发、专家、汇回一环扣一环。重叠靠两个微批错开半拍，错着就跨了层：A 还在收第 N 层，B 已在算第 N＋1 层的注意力。</li>
    <li><b>decode 的通信不占计算单元</b>：DeepSeek 的 profile 说明，RDMA 发出后 SM 全部还给计算，算完再等通信收尾；prefill 的通信要占 SM。</li>
    <li><b>权重不多存，但多读一遍</b>：两个微批用同一份常驻 HBM 的权重，可各自做矩阵乘时都要把这层权重流一遍，一步读两次（本课推导）。EP144 每卡每层专家 88 MB，读两遍不贵；每卡 32 个专家的 1.41 GB 就肉疼（6.3）。</li>
    <li><b>NVL72 上换细粒度重叠</b>：通信变短后，拆半批反而亏（6.4 末尾）。SGLang 第二篇不拆批，只把汇回通信和专家的下投影 GEMM、共享专家叠在一起：在 GEMM 里埋信号，通信不必等整个 GEMM 结束。</li>
    <li><b>单批也能跨层藏</b>：本层在算时，先把下一层权重往片上搬。这不靠拆批，<a href="#s14-3">14.3</a> 里 K3 的 megakernel 就让下一层权重提前在路上。</li>
  </ul>

  <h3 id="s6-6">6.6　EPLB：热门专家多备几份</h3>
  <p>哪个专家热门，要等路由算完才知道（<a href="topic-05.html#s4-3">专题五 §4.3</a>）。专家那一步等最慢的卡，热门专家所在的卡拖住所有人。EPLB（专家并行负载均衡器，DeepSeek 开源）按过去一段的负载统计，把热门专家多复制几份放到别的卡上，再重新排座位。</p>
  __FIG_S6_EPLB__
  <ul>
    <li><b>DeepSeek V3 怎么配</b>：prefill 多备 32 个冗余专家，每卡原有 8 个再加 1 个；按线上统计定期重算，例如每 10 分钟。</li>
    <li><b>卡越多越要紧</b>：卡越多越不均。SGLang 在大规模 EP 上测，开 EPLB 后 prefill 吞吐是不开的 1.49 倍，decode 是 2.54 倍；作者提醒效果看负载分布，线上要拿真实流量重测。</li>
    <li><b>顺带松开卡数</b>：256 个专家只能均分给 2 的幂张卡；加 32 份冗余凑成 288，12 张、72 张也分得开。</li>
    <li><b>开关</b>：SGLang 是 <code>--enable-eplb</code> 加 <code>--ep-num-redundant-experts</code>；vLLM 是 <code>--enable-eplb</code>，冗余份数和重排间隔写在 <code>--eplb-config</code>，默认每 3,000 步按最近 1,000 步的统计重排。SGLang 在 GB200 上的复现命令就带着 <code>--ep-num-redundant-experts 32</code>。</li>
  </ul>

  <div class="note ok"><span class="t">回到主线</span>
    <b>DEP 省下重复的 KV，大 EP 把专家那趟车坐满。</b>留下的问题：每个 rank 自备一份状态，K3 这类模型 DEP 起不来，退回 TP 后 KV 又被复制（第七节）。</div>

  <details class="foldfig"><summary><b>考考自己</b>：一个模型有 4 个 KV 头，8 张卡开 TEP8，换 DEP8 后 KV 容量最多是原来的几倍？什么情况下涨了也没用？</summary>
  <p>8 ÷ 4 ＝ 2，是 TEP8 的 2 倍。prefill 喂不上、或 batch 大到算力先到顶时，放得下也换不来吞吐，6.1 的 V4-Pro 就是。</p></details>
'''

FIGS = {
    "__FIG_S6_FOLD_T5__": ("fig-s6-fold-t5", "fig5-fold.svg", "topic05-fig-ep.py",
        '<b>同样 8 张卡：注意力按 TP 组算，专家层每卡管 32 个专家。</b><br><em>专题五原图，以 V3 为例。</em>'),
    "__FIG_S6_TEPDEP_T5__": ("fig-s6-tep-dep-t5", "fig5-tep-dep.svg", "topic05-fig-ep.py",
        '<b>只看注意力那一半。</b><br><em>专题五原图。</em>'),
    "__FIG_S6_KV__": ("fig-s6-dep-kv", "fig6-dep-kv.svg", "topic06-fig-dep.py",
        '<b>对比两列：在飞请求没变多，步时只剩约四分之一。</b><br>'
        '<em>实测为 GB300、vLLM、并发 512；在飞请求按 Little 定律推导，条长是示意。</em>'),
    "__FIG_S6_STATE__": ("fig-s6-dep-state", "fig6-dep-state.svg", "topic06-fig-dep.py",
        '<b>比同一位置的两张卡：左边格子全满，右边每卡只亮两列。</b><br>'
        '<em>格子是示意，头数、槽数不按实际；「切头只有 TP」为本课推导。</em>'),
    "__FIG_S6_BIGEP_WHAT__": ("fig-s6-bigep-what", "fig6-bigep-what.svg", "topic06-fig-bigep.py",
        '<b>比右边那张放大的卡：EP8 每卡 32 个专家，EP144 每卡只剩 2 个。</b><br>'
        '<em>机器、卡、专家份数照 DeepSeek 公开配置；机内摆法是示意。</em>'),
    "__FIG_S6_BIGEP_NET__": ("fig-s6-bigep-net", "fig6-bigep-net.svg", "topic06-fig-bigep.py",
        '<b>看左图横轴：卡数涨 32 倍，延迟只涨两到三倍，但每层都要付。</b><br>'
        '<em>32 ms 为本课推导的不重叠上限；三张卡片的规格来自各家公开资料。</em>'),
    "__FIG_S6_TBO__": ("fig-s6-tbo", "fig6-tbo.svg", "topic06-fig-dep.py",
        '<b>看「算」那一行：上面有空档，下面排满。</b><br>'
        '<em>时长是示意，省下的比例不按实测；实测 prefill 快 27%～35%。</em>'),
    "__FIG_S6_EPLB__": ("fig-s6-eplb", "fig6-eplb.svg", "topic06-fig-dep.py",
        '<b>看红虚线：从 70 降到 52.5，总量没变。</b><br>'
        '<em>8 个专家、4 张卡，负载数字是示意，不按实测。</em>'),
    "__FIG_S6_TPU__": ("fig-s6-dep-tpu", "fig6-dep-tpu.svg", "topic06-fig-dep.py",
        '<b>两条线在 256 到 512 之间交叉：decode 为主的那条后来居上。</b><br>'
        '<em>输出吞吐；折线两端标 TP → DP 的绝对值。专家都走 EP，只换注意力的切法。</em>'),
}

NUM_LOCK = []
