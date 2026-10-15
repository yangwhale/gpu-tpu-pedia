# -*- coding: utf-8 -*-
"""专题六 · 第四节「批处理与调度：一趟分给更多人」。图在 topic06-fig-batch.py。

⛔ 数字全部来自公开论文、官方文档／博客或本仓库公开 runbook，行号写在图脚本的 docstring 里。
   vLLM 抢占那句引自 docs.vllm.ai 的 Optimization and Tuning 页。
"""

BODY = '''
  <p class="lead">decode 每一步都搬一趟权重，同一步拼进来的人越多，每人分摊越少。
    <b>调度决定这一趟能坐满几个人。</b>两招：2022 年 Orca 让每一步都能有人进出，同延迟吞吐是 FasterTransformer（NVIDIA 早年的推理库）的 36.9 倍；2024 年 Sarathi-Serve 把长 prompt 切块塞进 decode 的步里，达标容量是 vLLM 的 2.6–5.6 倍。</p>

  <h3 id="s4-1">4.1　每一步都重新排一次</h3>
__FIG_S4_GANTT__
  <p>静态 batch 凑满一批一起跑，最慢的跑完才整批返回；先跑完的槽位空着，每步照样搬一趟权重。</p>
<figure class="fbox fwide" id="anim-contbatch">
<video src="media/topic06-contbatch.mp4" autoplay loop muted playsinline
       aria-label="静态批与连续批动画。标题：同样 4 个槽位、同一串请求：两种排法。上下两组各 4 个槽位，标着槽 1 到槽 4；上组叫「静态批：凑满一批，等最长的做完」，下组叫「连续批：每一步都重新排，谁做完谁让位」。每一列是一步，每个请求一种颜色，字母 A 到 O 标在它的第一格；灰格是槽位空转。一条黄色竖线随时间往右走，右边两个计数器写「做完几个」，底下写「已走几步」，右下角写「请求长度是示意」。字幕依次是：16 个请求排队，长短不一；每一列是一步 decode，每一步都要搬一趟权重。上面：短的早做完了，槽位也得空等最长的那个（灰格＝这一趟白搬）。下面：谁做完，下一步就换排队的下一个进来，槽位一格不空。最后：同样 15 步：静态批做完 8 个（空转 23 格），连续批做完 14 个（空转 0 格）。画面复位。"></video>
<figcaption>动画里队列一直不空，看两组各做完几个。
  <span class="sub">（18 秒无声循环，Manim 渲染。请求长度是示意，每步按等长算，没画进门那步的 prefill。）</span></figcaption></figure>
  <p>Orca（OSDI'22）改成每步结束都看一次：谁跑完谁让位，排队的人下一步补进来。今天的 vLLM、SGLang 都这么调度。</p>

  <h3 id="s4-2">4.2　新人进门那一步：chunked prefill</h3>
__FIG_S4_CHUNK__
  <p>新请求要在某一步里做 prefill。长 prompt 一口吞下，那一步就特别长，正在出字的人都卡一下。</p>
  <p>Sarathi（arXiv 2308.16369）和 Sarathi-Serve（arXiv 2403.02310）把 prompt 切成小块，每步带一块，和 decode 拼在一起。
    vLLM V1 默认打开，decode 先上车；每步塞多少 token 由 max_num_batched_tokens 控制。</p>
  <p>TPU v7x 上跑 Gemma4-31B 的公开记录把它设成 16,384，256K 的 prompt 切成 16 块。</p>

  <h3 id="s4-3">4.3　槽位满了以后</h3>
__FIG_S4_CONC__
  <p>槽位就是 KV 池装得下几个人。Little 定律：批里的人数 ≈ 吞吐 × TPOT，反过来<b>吞吐 ≈ 槽位数 × 1000 ÷ TPOT(ms)</b>。</p>
  <p>GB300 上 V4-Pro 的 dep8（8 路 DEP，第六节讲）实测每张 decode 卡 9,032 tok/s、TPOT 60 ms（<code>gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:794-801</code>，ISL 8K 的基线）。
    客户端并发 8,400 路，进 decode 批的只有约一半，其余在 prefill 那边排队。</p>
  <details class="foldfig"><summary><b>Little 定律核一遍</b>：批里到底几个人？</summary>
  <p>9,032 × 8 × 0.060 s 反推 decode 批里约 4,335 个请求，每卡约 540。</p></details>
  <p>槽位也不白给。3.4 那组每卡多装 13% 的人（886 → 999），端到端只多 0.8%，TPOT 从 77.7 拖到 85.3 ms：人多了，每步就长了。</p>
  <div class="note warn"><span class="t">TPU 上多一层：batch 大小要落在档位上</span>
    XLA（JAX 背后的编译器）按静态形状编译，batch 大小也是形状。SGLang-JAX 为几档 batch 提前编译，
    人数不在档位上就补到最近一档，补的行照样计算（<a href="topic-02.html#s5">专题二「运行时，还是编译期」</a>）。</div>

  <h3 id="s4-4">4.4　CPU 那几毫秒也要藏起来</h3>
__FIG_S4_OVERLAP__
  <p>每步之前 CPU 要挑人、拼 batch、准备输入；串着走，加速器每步都要等它。</p>

  <h3 id="s4-5">4.5　显存不够时踢谁</h3>
  <p>批开大了，KV 可能中途装不下，只能先请出去几个。回来两种办法：KV 换到主机内存再搬回（swap），或丢掉、重算 prefill（recompute）。</p>
  <p>vLLM 文档的原话是：In vLLM V1, the default preemption mode is RECOMPUTE rather than SWAP, as recomputation has lower overhead in the V1 architecture。</p>

  <div class="note ok"><span class="t">这一节只要带走一件事</span>
    <b>逐步调度是「一趟分给更多人」的地基。</b>槽位满了，吞吐就钉在槽位数 × 1000 ÷ TPOT 上，再加并发只加排队。
    chunked prefill 只摊平打扰，拆开是第八节的事。</div>

  <details class="foldfig"><summary><b>考考自己</b>：TPOT 稳定在 50 ms，想让每张卡出 10,000 tok/s，每张卡的批里至少要同时坐多少人？显存只装得下 300 人时会怎样？</summary>
  <p>10,000 × 0.05 ＝ 500 人。只装得下 300 人，吞吐钉在约 300 × 1000 ÷ 50 ＝ 6,000，再加并发只让首字等待变长；
    要么少搬（<a href="#s12-kv">第十二节末尾的 FP8 KV</a> 让每条 KV 约减半），要么换个交互档。</p></details>
'''

FIGS = {
    "__FIG_S4_GANTT__": ("fig-s4-gantt", "fig6-gantt.svg", "topic06-fig-batch.py",
        '<b>同样 37 格活，看两种排法各搬几趟。</b><br>'
        '<em>示意：请求长度是编排的，步数和空转格数由脚本模拟。</em>'),
    "__FIG_S4_CHUNK__": ("fig-s4-chunk", "fig6-chunk.svg", "topic06-fig-batch.py",
        '<b>看正在出字的人最长间隔，和新人首字晚了几格。</b><br>'
        '<em>示意单位；混合步按相加计时，偏保守。</em>'),
    "__FIG_S4_CONC__": ("fig-s4-conc", "fig6-conc.svg", "topic06-fig-batch.py",
        '<b>座位满了，翻台速度不变，多来的人只在门口排队。</b><br>'
        '<em>GB300 上 SGLang 跑 V4-Pro 的公开实测；TTFT 只记三档；排队人数按比例示意。</em>'),
    "__FIG_S4_OVERLAP__": ("fig-s4-overlap", "fig6-overlap.svg", "topic06-fig-batch.py",
        '<b>CPU 提前排好下一批，加速器不再等。</b><br>'
        '<em>时间线是示意，只有空档 12 ms 与 38 µs 是原文数字。</em>'),
}

NUM_LOCK = []
