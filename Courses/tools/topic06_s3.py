# -*- coding: utf-8 -*-
"""专题六 · 第三节「批处理与调度：一趟分给更多人」。图在 topic06-fig-batch.py。

⛔ 数字全部来自公开论文、官方文档／博客或本仓库公开 runbook，行号写在图脚本的 docstring 里。
   vLLM 抢占那句引自 docs.vllm.ai 的 Optimization and Tuning 页。
"""

BODY = '''
  <p class="lead">第一节算过：decode 每一步把权重搬一遍，同一趟坐的人越多，每人摊到的越少。最早的服务一批人一起进、一起出，先答完的人座位空着，等最慢的那个。
    <b>调度决定这一趟能坐满几个人。</b></p>

  <h3 id="s3-1">3.1　每一步都重新排一次</h3>
__FIG_S3_GANTT__
<figure class="fbox fwide" id="anim-contbatch">
<video src="media/topic06-contbatch.mp4" autoplay loop muted playsinline
       aria-label="静态批与连续批动画。标题：同样 4 个槽位、同一串请求：两种排法。上下两组各 4 个槽位，标着槽 1 到槽 4；上组叫「静态批：凑满一批，等最长的做完」，下组叫「连续批：每一步都重新排，谁做完谁让位」。每一列是一步，每个请求一种颜色，字母 A 到 O 标在它的第一格；灰格是槽位空转。一条黄色竖线随时间往右走，右边两个计数器写「做完几个」，底下写「已走几步」，右下角写「请求长度是示意」。字幕依次是：16 个请求排队，长短不一；每一列是一步 decode，每一步都要搬一趟权重。上面：短的早做完了，槽位也得空等最长的那个（灰格＝这一趟白搬）。下面：谁做完，下一步就换排队的下一个进来，槽位一格不空。最后：同样 15 步：静态批做完 8 个（空转 23 格），连续批做完 14 个（空转 0 格）。画面复位。"></video>
<figcaption>动画里队列一直不空，看两组各做完几个。
  <span class="sub">（18 秒无声循环，Manim 渲染。请求长度是示意，每步按等长算，没画进门那步的 prefill，3.2 那支补上。）</span></figcaption></figure>
  <p>Orca 是 2022 年 OSDI 上的一篇论文，也是论文里做的那套服务系统。今天没人直接跑它，跑的是它的招：论文叫迭代级调度，业界后来通称连续批（continuous batching），vLLM、SGLang 都按步调度。同延迟下，吞吐是 FasterTransformer（NVIDIA 早年的推理库）静态批的 36.9 倍（GPT-3 175B：每 token 中位延迟都压在 190 ms，每秒服务 0.185 → 6.81 个请求）。</p>
  <ul>
    <li><b>原来卡在哪：</b>老引擎对外只有「整批跑完」一个接口，调度器看不见中间的步；一批摞成「人数 × 最长长度」的方块，短的补零，中途换不掉一行。</li>
    <li><b>Orca 第一招，只跑一步就交回：</b>每步结束调度器都看一次，写完的马上返回、让座，排队的装得下就补进来。</li>
    <li><b>第二招，选择性批处理：</b>矩阵乘、归一化不管 token 属于谁，就把这一步所有人的 token 接成一长条，一起乘权重；只有注意力按人切开，各读各的 KV。</li>
    <li><b>挂不同 LoRA 的人也能拼：</b>基座权重照样共享一趟，只把各人的小矩阵另算。S-LoRA（2023）同时服务几千个适配器，吞吐最多是 HF PEFT、vLLM 简单做法的 4 倍；vLLM 用 <code>--enable-lora</code>、<code>--max-loras</code> 开。</li>
  </ul>

  <h3 id="s3-2">3.2　新人进门：chunked prefill，一步里装了什么</h3>
__FIG_S3_CHUNK__
  <p>新请求的长 prompt 一口吞下，那一步就特别长，正在出字的人都卡一下。</p>
  <p>Sarathi（arXiv 2308.16369）和 Sarathi-Serve（arXiv 2403.02310）把 prompt 切成小块，每步只带一块和 decode 拼着走，哪步都不会被拉长；延迟达标时能接的请求是 vLLM 的 2.6–3.7 倍。
    引擎里 vLLM 2024 年 4 月先合入，SGLang 同年 7 月跟进；vLLM V1 默认打开，decode 先上车；每步塞多少 token 由 max_num_batched_tokens 控制。</p>
  <p>TPU v7x 上跑 Gemma4-31B 的公开记录把它设成 16,384，256K 的 prompt 切成 16 块。</p>
  <ul>
    <li><b>块多大，看每步的 token 预算。</b>decode 的人先占位，每人一个 token，剩下的额度才给 prefill，所以块长随 decode 人数浮动。vLLM、SGLang 的默认都按显存分档，大显存卡 16,384，中档 8,192。预算小，出字间隔稳，首字晚；预算大则相反。</li>
    <li><b>后面的块要读前面全部的 KV，算得更慢。</b>矩阵乘部分每块一样重；注意力部分要和前面所有 token 配对，随位置线性涨。V3 粗算（本课推导）：线性部分每 token 约 740 亿次运算（激活 37B × 2）；注意力每多看一个历史 token 多约 500 万次（2 ×（192＋128）× 128 头 × 61 层）。历史超过约 1.5 万 token 后注意力反超；128K 的 prompt 切 8K 一块，末块约是首块的 7 倍，那一步也随之变长。</li>
    <li><b>切块不会多读 KV。</b>FlashAttention 本来就把 query 切成一百多个 token 的小块，每个小块各扫一遍前面的 KV；切不切块，扫的次数和内容都一样，只是分到不同的步里。注意力的总运算量也不变。</li>
    <li><b>块不能太小，代价在权重。</b>每走一步，权重都要从 HBM 读一遍。H100 每字节带宽配约 300 次运算（989 TFLOPS ÷ 3.35 TB/s），bf16 下一步至少约 300 个 token 矩阵乘才喂得饱（本课推导）。MoE 更苛刻：V3 每个 token 只去 256 个专家里的 8 个，要每个专家凑够 300 个，一步得有 9 千多个 token。默认预算落在 8K–16K，与此吻合。</li>
  </ul>

  <p><b>一步里装了什么：新人怎么接上。</b></p>
__FIG_S3_STEPMIX__
<figure class="fbox fwide" id="anim-stepmix">
<video src="media/topic06-stepmix.mp4" autoplay loop muted playsinline
       aria-label="一步里装了什么的动画。标题：一步里装了什么：新人怎么接上。左边六行标签：在车上、排队、这一步、乘权重、注意力、出字。车上一开始有 A 还差 9 字、B 还差 9 字、C 还差 9 字、D 还差 1 字；门口排着 E prompt 13、F prompt 4。「这一步」是 10 个空格子。底部两行：实心＝decode：每人 1 格、出 1 字；浅色＝prompt 块：最后一块那步出首字；每步预算 10 格、每格 1 个 token（真实部署是 8K 到 16K）；人数和长度都是示意。每一步先从车上飞下实心格，再从门口飞下浅色格，然后整条下面出现「矩阵乘：不管谁是谁」，每人一段下划线标着读多少 KV，最底下标出字、首字或不出字。字幕依次是：第 1 步：车上 4 人，各占 1 格 decode；剩 6 格切给 E：13 个 prompt 先算 6 个；整条 10 格拼成一个矩阵，一起乘权重；只有注意力分开算：各读各的 KV；第 1 步完成：D 出完最后一字下车；E 这步不出字。第 2 步：车上 3 人，各占 1 格 decode；剩 7 格：E 剩下的 7 个正好装完；F 这步排不上；第 2 步完成：E 最后一块当步出首字，下一步直接 decode。第 3 步：车上 4 人，各占 1 格 decode；剩 6 格：F 的 4 个一块装完，空着 2 格；第 3 步完成：F 也当步出首字；没人在门口等着上车。第 4 步：车上 5 人，各占 1 格 decode；没人排队：空着 5 格。预算是上限，不必塞满；第 4 步完成：5 人各出 1 字，车上始终没断。最后：一步 ＝ 一趟前向：decode 先占位，剩下的格子切给 prompt。画面复位。"></video>
<figcaption>盯住 E：从门口到车上，中间没有停靠。
  <span class="sub">（42 秒无声循环，Manim 渲染。规则照 vLLM V1；预算、人数、长度是示意，格子由脚本模拟。）</span></figcaption></figure>
  <p>一步就是一趟前向。每步开头调度器排一次：车上的人各占一格，剩下的格子按排队顺序切给新 prompt（3.2）。拼好的这一行一起乘权重，只有注意力各读各的 KV（<a href="#s3-1">3.1</a> 的选择性批处理）。</p>
  <ul>
    <li><b>新人不用「等着上车」。</b>prompt 最后一块算完那一步，前向末尾已经得到下一个字的概率，首字当步采样。下一步它就是一格普通 decode，中间没有停靠站。</li>
    <li><b>真要等，是 KV 池满了。</b>格子有剩、KV 装不下，新人只能在门口排；连车上的人都装不下，就按 3.4 请人下车。</li>
    <li><b>只有 PD 分离才有交接。</b>prefill 机器出了首字，KV 要整段搬到 decode 机器，搬完才接着出字（第八节）。混部时 KV 本来就在同一个池里，不用搬。</li>
    <li><b>门口一步只进一两个，车上怎么坐满三百人？</b>prefill 一个人只占一步，decode 一个人要坐几百上千步。设每人输出 1,000 字，要让车上一直有 300 人，平均每 3 步补进 1 人就够；8K 的预算一步能进两条 4K。门口不是瓶颈，KV 池才是（下一条）。</li>
    <li><b>预算里一格算一格，搬的 KV 不一样。</b>矩阵乘那部分，每格的运算一样、权重共用一趟，decode 搭车几乎白搭。注意力那部分不一样：prefill 的 KV 块读上芯片，一组一百多个 query 一起用（3.2）；decode 一块只喂一个人。两条 4K prefill 加 300 人 4K 上下文的 decode，同一步里前者读约 9.5 GB 的 KV，后者约 86 GB；摊到每格差约 250 倍（本课推导）。预算是调度旋钮，不是耗时模型。</li>
    <li><b>混着跑能凑够屋脊点，吐字还是看 decode 人数。</b>1.1 那约 343 行，数的是这一步乘权重的总行数，prefill 块的几千行也算：decode 只有几十人，这一步也过线。可 prefill 那几千行不出字，每步出几个字还是车上几个人；decode 的注意力照样卡带宽。</li>
    <li><b>引擎不全都混。</b>vLLM V1 照图里这样混。SGLang 默认一步只跑一种，要么一块 prefill，要么全员 decode，开 <code>--enable-mixed-chunk</code> 才混。不混也不用等：prefill 那步照样出首字，下一个 decode 步就带上它。</li>
    <li><b>TPU 上同一套规则。</b>长短不一的 prefill 块和 decode 进同一个注意力 kernel（RPA，<a href="#s5-1">5.1</a>）；这一步的格数也要落在预编译的档位上，差的补空（3.3 末尾）。</li>
  </ul>

  <h3 id="s3-3">3.3　槽位满了以后</h3>
__FIG_S3_CONC__
  <p>槽位就是 KV 池装得下几个人。Little 定律（见词表）：批里的人数 ≈ 吞吐 × TPOT，反过来<b>吞吐 ≈ 槽位数 × 1000 ÷ TPOT(ms)</b>。</p>
  <p>GB300 上 V4-Pro 的 dep8（第六节）实测每张 decode 卡 9,032 tok/s、TPOT 60 ms（<code>gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:794-801</code>，ISL 8K 的基线）。
    客户端并发 8,400 路，进 decode 批的只有约一半，其余在 prefill 端排队。</p>
  <details class="foldfig"><summary><b>Little 定律核一遍</b>：批里到底几个人？</summary>
  <p>9,032 × 8 张 decode 卡 × 0.060 s 反推 decode 批里约 4,335 个请求，每卡约 542。</p></details>
  <p>槽位也不白给。同一组实测把两个 KV 池的隔板挪一下（4.2），每卡多装 13% 的人（886 → 999），端到端吞吐只多 0.8%，TPOT 从 77.7 拖到 85.3 ms：人多了，每步就长了。</p>
  <div class="note warn"><span class="t">TPU 上多一层：batch 大小要落在档位上</span>
    XLA 按静态形状编译，batch 大小也是形状：人数不在预编译的档位上就补到最近一档，补的行照样计算（<a href="topic-02.html#s5">专题二「运行时，还是编译期」</a>；选档见 <a href="#s14-1">14.1</a>）。</div>

  <p><b>CPU 那几毫秒也要藏起来。</b></p>
__FIG_S3_OVERLAP__
  <p>每步之前 CPU 要挑人、拼 batch、准备输入；串着走，加速器每步都要等它。</p>
  <p>GPU 上，2024 年底 SGLang 的零开销调度把下一批的 CPU 准备和这一批的 GPU 计算重叠，吞吐是原来的 1.1 倍。2025 年 vLLM V1 吞吐最多是 V0 的 1.7 倍，原因之一是分词、回传等 CPU 活挪到了别的进程。</p>

  <h3 id="s3-4">3.4　显存不够时踢谁</h3>
  <p>批开大了，KV 可能中途装不下，只能先请出去几个。回来两种办法：KV 换到主机内存再搬回（swap），或丢掉、重算 prefill（recompute）。</p>
  <p>vLLM V1 默认重算，文档说 V1 里重算开销更低。</p>

  <div class="note ok"><span class="t">这一节只要带走一件事</span>
    <b>逐步调度是「一趟分给更多人」的地基。</b>槽位满了，吞吐就钉在槽位数 × 1000 ÷ TPOT 上，再加并发只加排队。
    chunked prefill 只摊平打扰，拆到两批机器上是第八节的事。
    留下的问题：槽位数由显存里的 KV 定，而 Orca 给每人按最大长度预留一整段 KV，实际只用上两到近四成（第四节）。</div>

  <details class="foldfig"><summary><b>考考自己</b>：TPOT 稳定在 50 ms，想让每张卡出 10,000 tok/s，每张卡的批里至少要同时坐多少人？显存只装得下 300 人时会怎样？</summary>
  <p>10,000 × 0.05 ＝ 500 人。只装得下 300 人，吞吐钉在约 300 × 1000 ÷ 50 ＝ 6,000；
    要么少搬（<a href="#s12-4">第十二节末尾的 FP8 KV</a> 让每条 KV 约减到 61%，含索引键），要么换个交互档。</p></details>
'''

FIGS = {
    "__FIG_S3_GANTT__": ("fig-s3-gantt", "fig6-gantt.svg", "topic06-fig-batch.py",
        '<b>同样 37 格活，看两种排法各搬几趟。</b><br>'
        '<em>示意：请求长度是编排的，步数和空转格数由脚本模拟。</em>'),
    "__FIG_S3_CHUNK__": ("fig-s3-chunk", "fig6-chunk.svg", "topic06-fig-batch.py",
        '<b>看正在出字的人最长间隔，和新人首字晚了几格。</b><br>'
        '<em>示意单位；混合步按相加计时，偏保守。</em>'),
    "__FIG_S3_STEPMIX__": ("fig-s3-stepmix", "fig6-stepmix.svg", "topic06-fig-batch.py",
        '<b>每一行先排实心，剩下的才轮到浅色。</b><br>'
        '<em>示意：规则照 vLLM V1 文档；预算、人数、prompt 长度是编排的，格子由脚本模拟。</em>'),
    "__FIG_S3_CONC__": ("fig-s3-conc", "fig6-conc.svg", "topic06-fig-batch.py",
        '<b>左边绿柱几乎一样高，右边红柱越来越高。</b><br>'
        '<em>GB300 上 SGLang 跑 V4-Pro 的公开实测；TTFT 只记三档；排队人数按比例示意；并发按每个前端计，共 14 个。</em>'),
    "__FIG_S3_OVERLAP__": ("fig-s3-overlap", "fig6-overlap.svg", "topic06-fig-batch.py",
        '<b>比两行 TPU：上面的「空等」格在下面没了。</b><br>'
        '<em>时间线是示意，只有空档 12 ms 与 38 µs 是 SGLang-JAX 原文数字。</em>'),
}

NUM_LOCK = []
