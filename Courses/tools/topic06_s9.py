# -*- coding: utf-8 -*-
"""专题六 · 第九节「配几台、拆不拆：P:D 配比、AFD、EPD」。

新图在 topic06-fig-ratio.py（fig6-2p1d / prefill-sweep / decide / afd-ascend）；
另复用专题五 §6.3、§6.5 的两张图（topic05-fig-pd.py，id 加 -t5）。
"""

BODY = '''
  <p class="lead">拆开以后多出一个旋钮：P、D 各配几台。在它前面还有一问：这个负载该不该拆。
    <b>都要拿自己的负载去量，别照搬别人的配比。</b></p>

  <h3 id="s9-1">9.1　配比：先算一个起点</h3>
  <p><a href="topic-05.html#s6-3">专题五 §6.3</a> 用 DistServe 的例子讲过：两边按各自上限配。</p>
  <p>混着做为什么只到 1.6：DistServe 的延迟要求是两条，首字（TTFT）和字距（TPOT）。两段混在一起互相干扰，要求卡得紧时只能顾一头，要么多堆卡才能两头都守住。</p>
__FIG_S9_RATIO_T5__
  <p>真实部署要把「一张接几个」换成量出来的吞吐。</p>

  <p><b>配比公式：先算一个起点。</b><b>prefill 个数 ≈ decode 每秒完成的请求数 × 输入长度 ÷ 一个 prefill 每秒能吞的 token。</b></p>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">DeepSeek-V4-Pro，SGLang，GB300，8K 进 1K 出，每张 decode 卡 11,200 output tok/s 的官方点（分母按 output ÷ decode 卡读，见 2.1；若是全部卡，此表不成立）；推导见 gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:1082–1091。</caption>
    <tr><th>步骤</th><th>算式</th><th>结果</th></tr>
    <tr><td>decode 每秒完成</td><td>8 卡 × 11,200 ÷ 1,024 个输出 token</td><td>约 87.5 个请求</td></tr>
    <tr><td>prefill 要供的料</td><td>87.5 × 8,192 个输入 token</td><td>约 71.7 万 tok/s</td></tr>
    <tr><td>一个 prefill（4 卡）</td><td>4 × 每卡 18,200 input tok/s（官方的 prefill 单卡吞吐）</td><td>约 7.28 万 tok/s</td></tr>
    <tr><td>要几个</td><td>71.7 ÷ 7.28</td><td>约 9.8</td></tr>
  </table>
  <p>同一个部署有三个 prefill 个数：</p>
  <ul>
    <li><b>9.8 个</b>：公式的起点。runbook 实测 prefill 单卡峰值只有 15,196（官方的 83%），照实测还要更多。</li>
    <li><b>8 个</b>：官方高并发配方（high-conc-8p1d）的取值。</li>
    <li><b>14 个</b>：runbook 早期自己扫的，14 到 16 只涨 2%，已喂饱，于是定为默认（每张 decode 卡输出：8 个 prefill 时 6,788 tok/s，14 个时 8,809，16 个时 8,993；runbook:43、1117）。</li>
  </ul>
  <p>公式定起点，上下扫，取每卡输出不再涨的那一档。</p>
  <p><b>线上流量一直在变，配比也不能定死。</b>DeepSeek V4.1-Flash 线上让两边各自扩缩：prefill 按没命中缓存的 token 数算要几个实例，相当于把公式里的输入长度换成没命中的那部分；decode 按正在算的请求数算。都设上下界和限速；配额占用高于 0.8 不许缩容，池里另留空闲机器当扩容库存。</p>

  <p><b>多一台 prefill，换来的是首字。</b>一份 B200 上的 DeepSeek-V3 报告，在 1P1D（1 台 prefill 配 1 台 decode）上加一台 prefill。</p>
__FIG_S9_2P1D__
  <p><b>同一组数，按每卡吞吐看是亏，按 TTFT 看是赚。</b></p>
  <ul>
    <li><b>口径</b>：TTFT 是客户端测的，含排队；两边都关了 CUDA Graph，TPOT 偏高，不比。</li>
    <li><b>为什么吞吐不涨</b>：固定并发压测下每个请求从发出到收完约 65 s（报告的平均端到端延迟），平均首字省下的 0.67 s 摊进去几乎看不出。</li>
  </ul>

  <h3 id="s9-2">9.2　换个分母，不拆反而更好的时候</h3>
  <p>vLLM 跑 V4-Pro，decode 固定 8 张卡，prefill 从 1 个加到 14 个。</p>
__FIG_S9_SWEEP__
  <p>蓝线只除 decode 卡，是不少 PD 报告的默认口径（2.1）；橙线把 prefill 的卡也算进分母，12 个以后就不涨了。曲线是 2026-07-24 的历史实测，只看趋势。</p>

  <p><b>不拆反而更好的时候。</b><b>prefill 和 decode 本来没怎么打架时，拆开是白付代价。</b>图是 MiMo-V2-Flash 在 TPU 上的三种摆法：</p>
__FIG_S9_SPLIT__
  <p>Qwen3-Coder（两台 v7x-8，1K 进 1K 出）单并发拆成 1P1D，首字中位数 95 → 281 ms，输出只多 12%：单并发时两段本不抢资源。</p>
  <details class="foldfig"><summary><b>出处与口径</b>：两组 TPU 例子</summary>
  <p>MiMo：tpu/sglang-jax/mimo/README.md:8–12、81–88，每 pod 4 颗芯片，request_rate=inf，total tok/s 含输入。README 写 1P1D 比不拆高「约 6.8%」，按表算是 6.7%；2P1D 每 pod 5.10K 对不拆的 6.39K（本课推导）。</p>
  <p>Qwen3-Coder：tpu/tpu-inference/Qwen3-Coder-480B/README.md:1005–1021，TTFT 与 TPOT 都是中位数。两组都是单点测量。</p></details>

  <h3 id="s9-3">9.3　决策树：三个问题</h3>
  <p>答「否」就停。分块 prefill 见第三节，KV 薄不薄见 8.4。</p>
__FIG_S9_DECIDE__

  <h3 id="s9-4">9.4　再拆一层：AFD 与 EPD</h3>
  <p><a href="topic-05.html#s6-5">专题五 §6.5</a> 讲过：decode 里的注意力和专家也能拆到两批机器上，多台注意力机喂同一批专家，专家那边就凑得出大 batch。</p>
__FIG_S9_AFD_T5__
  <p>字节 2025 年的 MegaScale-Infer 报告每卡吞吐最多是对照系统的 1.90 倍；阶跃星辰的 Step-3 在 Hopper 上、50 ms TPOT、4K 上下文的约束下，峰值每卡 4,039 tok/s；论文拿来对照的 DeepSeek-V3 是 2,324，那是 DeepSeek 自己公开的 H800 profile 峰值，不是 Step-3 重测的。模型不同，Step-3 这档还用了 FP8 注意力，差距不能全记给 AFD。TPU 上本课没找到公开的 AFD 实现。</p>
  <p>开源实现没这么顺：vLLM 的 AFD 是外部插件，主仓库的提案（RFC）已因不活跃被关闭；插件博客在 Ascend 910C 上的数，配比一错就比不拆还慢。</p>
__FIG_S9_AFD_ASCEND__
  <ul>
    <li><b>配比是关键</b>：作者说绿柱那组的专家侧仍有余力，再加注意力可能还涨。</li>
    <li><b>只能看趋势</b>：实验把专家路由换成强制均衡，还用小部署模拟大规模。</li>
  </ul>
  <details class="foldfig"><summary><b>出处</b>：AFD 的几个数</summary>
  <p>MegaScale-Infer：arXiv 2504.02263（对照的是当时最好的方案）。Step-3：arXiv 2507.19427 表 8（4K 上下文、FP8、不开 MTP；4,039 是峰值那一分钟，长期平均 3,910）。vLLM AFD 插件：vllm.ai/blog/2026-07-23-vllm-afd-plugin、github.com/vllm-project/afd-plugin；RFC #22799 于 2026-05-31 被机器人关闭。</p></details>

  <p><b>EPD：多模态把编码也拆出去。</b>多模态模型在 prefill 前要把图片、视频编码成 token，图一多就和 prefill 抢资源。</p>
  <ul>
    <li><b>做法</b>：编码器单独一批机器，视觉 token 交给 prefill，KV 再交给 decode。</li>
    <li><b>论文的数</b>：arXiv 2501.05460 对比不拆的系统，报告峰值显存最多降到 1/15，batch 最多是原来的 22 倍，TTFT 最多降 71%。</li>
    <li><b>什么时候值得</b>：请求带很多图或大图时；纯文本用不上（专题五 §6.6）。</li>
  </ul>

  <div class="note ok"><span class="t">回到主线</span>
    AFD 让每个专家一趟分到更多 token。拆不拆、怎么配，看负载，不看潮流。
    留下的问题：DEP、PD、AFD 摆在一起，先定哪种拓扑、再调哪些参数（第十节）？</div>

'''

FIGS = {
    "__FIG_S9_RATIO_T5__": ("fig-s9-pd-ratio-t5", "fig5-pd-ratio.svg", "topic05-fig-pd.py",
        '<b>两边按上限配，谁也不等谁。</b><br><em>专题五原图，DistServe 论文的 OPT 例子。</em>'),
    "__FIG_S9_2P1D__": ("fig-s9-2p1d", "fig6-2p1d.svg", "topic06-fig-ratio.py",
        '<b>左边一高一低，右边两组都在降。</b><br><em>B200，并发 128；每卡 ＝ 总输出 ÷ 全部卡数。</em>'),
    "__FIG_S9_SWEEP__": ("fig-s9-prefill-sweep", "fig6-prefill-sweep.svg", "topic06-fig-ratio.py",
        '<b>同一组实测，两条线。</b><br><em>蓝线是 runbook 原口径，橙线是本课换了分母重算的。</em>'),
    "__FIG_S9_SPLIT__": ("fig-s9-split-pod", "fig6-split-pod.svg", "topic06-fig-ratio.py",
        '<b>总量看 2P1D 最高；按 pod 算，只有 batch 128 的 1P1D 赢一点。</b><br><em>README 原值；每 pod 吞吐为本课推导（总吞吐 ÷ pod 数）。</em>'),
    "__FIG_S9_DECIDE__": ("fig-s9-decide", "fig6-decide.svg", "topic06-fig-ratio.py",
        '<b>灰框是先做的一步，蓝框才是那三问。</b><br><em>阈值没画，要拿自己的负载量。</em>'),
    "__FIG_S9_AFD_T5__": ("fig-s9-afd-t5", "fig5-afd.svg", "topic05-fig-pd.py",
        '<b>一层里的两种活，分到两批机器。</b><br><em>专题五原图，机器数是示意。</em>'),
    "__FIG_S9_AFD_ASCEND__": ("fig-s9-afd-ascend", "fig6-afd-ascend.svg", "topic06-fig-ratio.py",
        '<b>同样拆开，橙柱亏、绿柱赚。</b><br><em>每 die 吞吐的分母是部署的全部 die。</em>'),
}

NUM_LOCK = [
    ("10.4", "fig6-prefill-sweep.svg"),
]
