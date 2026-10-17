# -*- coding: utf-8 -*-
"""专题六 · 第三节「批处理与调度：一趟分给更多人」。图在 topic06-fig-batch.py。

⛔ 数字全部来自公开论文、官方文档／博客或本仓库公开 runbook，行号写在图脚本的 docstring 里。
   vLLM 抢占那句引自 docs.vllm.ai 的 Optimization and Tuning 页。
"""

BODY = '''
  <p class="lead">第一节算过：decode 每一步把权重搬一遍，同一趟坐的人越多，每人摊到的越少。最早的服务一批人一起进、一起出，先答完的人座位空着，等最慢的那个。
    <b>调度决定这一趟能坐满几个人。</b>两招：2022 年 Orca 让每一步都能有人进出，同延迟吞吐是 FasterTransformer（NVIDIA 早年的推理库）的 36.9 倍（GPT-3 175B）；2024 年 Sarathi-Serve 把长 prompt 切块塞进 decode 的步里，延迟达标时能接的请求是 vLLM 的 2.6–3.7 倍。</p>

  <h3 id="s3-1">3.1　每一步都重新排一次</h3>
__FIG_S3_GANTT__
<figure class="fbox fwide" id="anim-contbatch">
<video src="media/topic06-contbatch.mp4" autoplay loop muted playsinline
       aria-label="静态批与连续批动画。标题：同样 4 个槽位、同一串请求：两种排法。上下两组各 4 个槽位，标着槽 1 到槽 4；上组叫「静态批：凑满一批，等最长的做完」，下组叫「连续批：每一步都重新排，谁做完谁让位」。每一列是一步，每个请求一种颜色，字母 A 到 O 标在它的第一格；灰格是槽位空转。一条黄色竖线随时间往右走，右边两个计数器写「做完几个」，底下写「已走几步」，右下角写「请求长度是示意」。字幕依次是：16 个请求排队，长短不一；每一列是一步 decode，每一步都要搬一趟权重。上面：短的早做完了，槽位也得空等最长的那个（灰格＝这一趟白搬）。下面：谁做完，下一步就换排队的下一个进来，槽位一格不空。最后：同样 15 步：静态批做完 8 个（空转 23 格），连续批做完 14 个（空转 0 格）。画面复位。"></video>
<figcaption>动画里队列一直不空，看两组各做完几个。
  <span class="sub">（18 秒无声循环，Manim 渲染。请求长度是示意，每步按等长算，没画进门那步的 prefill。）</span></figcaption></figure>
  <p>Orca 是 2022 年 OSDI 上的一篇论文，也是论文里做的那套服务系统。今天没人直接跑它，跑的是它的招：论文叫迭代级调度，业界后来通称连续批（continuous batching），vLLM、SGLang 都按步调度。</p>
  <ul>
    <li><b>原来卡在哪：</b>老引擎对外只有「整批跑完」一个接口，调度器看不见中间的步；一批摞成「人数 × 最长长度」的方块，短的补零，中途换不掉一行。</li>
    <li><b>Orca 第一招，只跑一步就交回：</b>每步结束调度器都看一次，写完的马上返回、让座，排队的装得下就补进来。</li>
    <li><b>第二招，选择性批处理：</b>矩阵乘、归一化不管 token 属于谁，就把这一步所有人的 token 接成一长条，一起乘权重；只有注意力按人切开，各读各的 KV。</li>
    <li><b>挂不同 LoRA 的人也能拼：</b>基座权重照样共享一趟，只把各人的小矩阵另算。S-LoRA（2023）同时服务几千个适配器，吞吐最多是 HF PEFT、vLLM 简单做法的 4 倍；vLLM 用 <code>--enable-lora</code>、<code>--max-loras</code> 开。</li>
    <li><b>还缺一半：</b>Orca 的 KV 仍按最大长度给每人预留连续一段，4.3 里利用率 20.4%–38.2% 的旧系统就是按它复现的。PagedAttention 改成按块领，配上按页表读的注意力 kernel，长短不一的一批一次算完，座位才多起来。</li>
  </ul>

  <h3 id="s3-2">3.2　新人进门那一步：chunked prefill</h3>
__FIG_S3_CHUNK__
  <p>新请求的长 prompt 一口吞下，那一步就特别长，正在出字的人都卡一下。</p>
  <p>Sarathi（arXiv 2308.16369）和 Sarathi-Serve（arXiv 2403.02310）把 prompt 切成小块，每步只带一块和 decode 拼着走，哪步都不会被拉长。
    引擎里 vLLM 2024 年 4 月先合入，SGLang 同年 7 月跟进；vLLM V1 默认打开，decode 先上车；每步塞多少 token 由 max_num_batched_tokens 控制。</p>
  <p>TPU v7x 上跑 Gemma4-31B 的公开记录把它设成 16,384，256K 的 prompt 切成 16 块。</p>
  <ul>
    <li><b>块多大，看每步的 token 预算。</b>decode 的人先占位，每人一个 token，剩下的额度才给 prefill，所以块长随 decode 人数浮动。vLLM 按显存定默认：160 GB 以上的卡 16,384；70 GB 以上（A100 除外，调大反而掉吞吐）起 API 服务 8,192；更小的卡 2,048。SGLang 也按显存分档：H100 一档 8,192，B200 一档 16,384。预算小，出字间隔稳，首字晚；预算大则相反。</li>
    <li><b>后面的块要读前面全部的 KV，算得更慢。</b>矩阵乘部分每块一样重；注意力部分要和前面所有 token 配对，随位置线性涨。V3 粗算（本课推导）：线性部分每 token 约 740 亿次运算（激活 37B × 2）；注意力每多看一个历史 token 多约 500 万次（2 ×（192＋128）× 128 头 × 61 层）。历史超过约 1.5 万 token 后注意力反超；128K 的 prompt 切 8K 一块，末块约是首块的 7 倍，那一步也随之变长。</li>
    <li><b>切块不会多读 KV。</b>FlashAttention 本来就把 query 切成一百多个 token 的小块，每个小块各扫一遍前面的 KV；切不切块，扫的次数和内容都一样，只是分到不同的步里。注意力的总运算量也不变。</li>
    <li><b>块不能太小，代价在权重。</b>每走一步，权重都要从 HBM 读一遍。H100 每字节带宽配约 300 次运算（989 TFLOPS ÷ 3.35 TB/s），bf16 下一步至少约 300 个 token 矩阵乘才喂得饱（本课推导）。MoE 更苛刻：V3 每个 token 只去 256 个专家里的 8 个，要每个专家凑够 300 个，一步得有 9 千多个 token。默认预算落在 8K–16K，与此吻合。</li>
  </ul>

  <h3 id="s3-3">3.3　槽位满了以后</h3>
__FIG_S3_CONC__
  <p>槽位就是 KV 池装得下几个人。Little 定律（见词表）：批里的人数 ≈ 吞吐 × TPOT，反过来<b>吞吐 ≈ 槽位数 × 1000 ÷ TPOT(ms)</b>。</p>
  <p>GB300 上 V4-Pro 的 dep8（第六节）实测每张 decode 卡 9,032 tok/s、TPOT 60 ms（<code>gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:794-801</code>，ISL 8K 的基线）。
    客户端并发 8,400 路，进 decode 批的只有约一半，其余在 prefill 端排队。</p>
  <details class="foldfig"><summary><b>Little 定律核一遍</b>：批里到底几个人？</summary>
  <p>9,032 × 8 张 decode 卡 × 0.060 s 反推 decode 批里约 4,335 个请求，每卡约 542。</p></details>
  <p>槽位也不白给。同一组实测把两个 KV 池的隔板挪一下（4.4），每卡多装 13% 的人（886 → 999），端到端吞吐只多 0.8%，TPOT 从 77.7 拖到 85.3 ms：人多了，每步就长了。</p>
  <div class="note warn"><span class="t">TPU 上多一层：batch 大小要落在档位上</span>
    XLA 按静态形状编译，batch 大小也是形状：人数不在预编译的档位上就补到最近一档，补的行照样计算（<a href="topic-02.html#s5">专题二「运行时，还是编译期」</a>；选档见 <a href="#s14-2">14.2</a>）。</div>

  <h3 id="s3-4">3.4　CPU 那几毫秒也要藏起来</h3>
__FIG_S3_OVERLAP__
  <p>每步之前 CPU 要挑人、拼 batch、准备输入；串着走，加速器每步都要等它。</p>
  <p>GPU 上，2024 年底 SGLang 的零开销调度把下一批的 CPU 准备和这一批的 GPU 计算重叠，吞吐是原来的 1.1 倍。2025 年 vLLM V1 吞吐最多是 V0 的 1.7 倍，原因之一是分词、回传等 CPU 活挪到了别的进程。</p>

  <h3 id="s3-5">3.5　显存不够时踢谁</h3>
  <p>批开大了，KV 可能中途装不下，只能先请出去几个。回来两种办法：KV 换到主机内存再搬回（swap），或丢掉、重算 prefill（recompute）。</p>
  <p>vLLM V1 默认重算，文档说 V1 里重算开销更低。</p>

  <div class="note ok"><span class="t">这一节只要带走一件事</span>
    <b>逐步调度是「一趟分给更多人」的地基。</b>槽位满了，吞吐就钉在槽位数 × 1000 ÷ TPOT 上，再加并发只加排队。
    chunked prefill 只摊平打扰，拆到两批机器上是第八节的事。
    留下的问题：槽位数由显存里的 KV 定，而 Orca 给每人按最大长度预留一整段 KV，实际只用上两到近四成（第四节）。</div>

  <details class="foldfig"><summary><b>考考自己</b>：TPOT 稳定在 50 ms，想让每张卡出 10,000 tok/s，每张卡的批里至少要同时坐多少人？显存只装得下 300 人时会怎样？</summary>
  <p>10,000 × 0.05 ＝ 500 人。只装得下 300 人，吞吐钉在约 300 × 1000 ÷ 50 ＝ 6,000；
    要么少搬（<a href="#s12-kv">第十二节末尾的 FP8 KV</a> 让每条 KV 约减到 61%，含索引键），要么换个交互档。</p></details>
'''

FIGS = {
    "__FIG_S3_GANTT__": ("fig-s3-gantt", "fig6-gantt.svg", "topic06-fig-batch.py",
        '<b>同样 37 格活，看两种排法各搬几趟。</b><br>'
        '<em>示意：请求长度是编排的，步数和空转格数由脚本模拟。</em>'),
    "__FIG_S3_CHUNK__": ("fig-s3-chunk", "fig6-chunk.svg", "topic06-fig-batch.py",
        '<b>看正在出字的人最长间隔，和新人首字晚了几格。</b><br>'
        '<em>示意单位；混合步按相加计时，偏保守。</em>'),
    "__FIG_S3_CONC__": ("fig-s3-conc", "fig6-conc.svg", "topic06-fig-batch.py",
        '<b>左边绿柱几乎一样高，右边红柱越来越高。</b><br>'
        '<em>GB300 上 SGLang 跑 V4-Pro 的公开实测；TTFT 只记三档；排队人数按比例示意；并发按每个前端计，共 14 个。</em>'),
    "__FIG_S3_OVERLAP__": ("fig-s3-overlap", "fig6-overlap.svg", "topic06-fig-batch.py",
        '<b>比两行 TPU：上面的「空等」格在下面没了。</b><br>'
        '<em>时间线是示意，只有空档 12 ms 与 38 µs 是 SGLang-JAX 原文数字。</em>'),
}

NUM_LOCK = []
