# -*- coding: utf-8 -*-
"""专题六 · 第九节「配几台、拆不拆：P:D 配比、AFD、EPD」。

新图在 topic06-fig-ratio.py（fig6-2p1d / prefill-sweep / decide / afd-ascend）；
另复用专题五 §6.3、§6.5 的两张图（topic05-fig-pd.py，id 加 -t5）。
"""

BODY = '''
  <p class="lead">Dynamo、llm-d 这类框架出来以后，PD 配比成了上线要填的参数。要回答两件事：P、D 各配几台；这个负载该不该拆。
    <b>两件都要拿自己的负载去量，别照搬别人的配比。</b></p>

  <h3 id="s9-1">9.1　回顾：配比从哪来</h3>
  <p><a href="topic-05.html#s6-3">专题五 §6.3</a> 用 DistServe 的例子讲过：两边各有一张卡能接的上限，按上限配到差不多同时忙满。</p>
__FIG_S9_RATIO_T5__
  <p>真实部署里，要把「一张接几个」换成量出来的吞吐。</p>

  <h3 id="s9-2">9.2　配比公式：先算一个起点</h3>
  <p><b>prefill 个数 ≈ decode 每秒完成的请求数 × 输入长度 ÷ 一个 prefill 每秒能吞的 token。</b>
    分子是 decode 每秒要的料，分母是一台 prefill 每秒能备的。</p>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">DeepSeek-V4-Pro，SGLang，GB300，8K 进 1K 出，每张 decode 卡 11,200 output tok/s 的官方点（官方未写明分母，按 output ÷ decode 卡读；若是全部卡，此表不成立）；推导见 gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:1082–1091。</caption>
    <tr><th>步骤</th><th>算式</th><th>结果</th></tr>
    <tr><td>decode 每秒完成</td><td>8 卡 × 11,200 ÷ 1,024 个输出 token</td><td>约 87.5 个请求</td></tr>
    <tr><td>prefill 要供的料</td><td>87.5 × 8,192 个输入 token</td><td>约 71.7 万 tok/s</td></tr>
    <tr><td>一个 prefill（4 卡）</td><td>4 × 每卡 18,200 input tok/s（官方的 prefill 单卡吞吐）</td><td>约 7.28 万 tok/s</td></tr>
    <tr><td>要几个</td><td>71.7 ÷ 7.28</td><td>约 9.8</td></tr>
  </table>
  <p>同一个部署有三个 prefill 个数：</p>
  <ul>
    <li><b>9.8 个</b>：上表公式的起点。runbook 实测 prefill 单卡峰值只有 15,196（官方的 83%），照实测还要更多。</li>
    <li><b>8 个</b>：官方高并发配方（high-conc-8p1d）的取值，比公式少。</li>
    <li><b>14 个</b>：runbook 早期自己扫的（历史数据）：每张 decode 卡输出 8 个时 6,788，14 个时 8,809，16 个时 8,993；14 到 16 只涨 2%，已喂饱，于是定为默认（runbook:43、1117）。</li>
  </ul>
  <p>自己上线时，用公式算起点，在起点上下扫，取每卡输出不再涨的那一档。</p>

  <h3 id="s9-3">9.3　多一台 prefill，换来的是首字</h3>
  <p>一份 B200 上的 DeepSeek-V3 报告，在 1P1D（1 台 prefill 配 1 台 decode）上再加一台 prefill：总输出只从 1,118.8 到 1,159.0 tok/s，每张卡反倒从 69.9 掉到 48.3。</p>
__FIG_S9_2P1D__
  <p>用户那头变化却大：平均 TTFT 少了 28%，P99 TTFT 从 12.07 s 降到 6.23 s。<b>同一组数，按每卡吞吐看是亏，按 TTFT 的 SLO（对用户承诺的延迟上限）看是赚。</b></p>
  <ul>
    <li><b>口径</b>：TTFT 是客户端测的，含排队；两边都关了 CUDA Graph（把一串 GPU 调用录好、整段重放），TPOT 偏高，不比。</li>
    <li><b>为什么吞吐不涨</b>：瓶颈在 decode，排队的请求只是开始得早。</li>
  </ul>

  <h3 id="s9-4">9.4　换个分母，10.4 倍变 2.2 倍</h3>
  <p>vLLM 跑 V4-Pro，decode 固定 8 张卡，prefill 从 1 个加到 14 个。按 decode 卡算，12 个 prefill 时每卡输出从 307 涨到 3,187。</p>
__FIG_S9_SWEEP__
  <p>只除 decode 卡是不少 PD 报告的默认口径，藏起了 prefill 的成本；把 prefill 的卡也算进分母，只从 204 到 455，2.2 倍。<b>看到每卡吞吐，先问分母里有没有 prefill 的卡。</b>曲线是 2026-07-24 的历史实测，看趋势。</p>

  <h3 id="s9-5">9.5　不拆反而更好的时候</h3>
  <p><b>prefill 和 decode 本来没怎么打架时，拆开是白付代价。</b></p>
  <table>
    <tr><th>负载</th><th>不拆</th><th>拆</th><th>怎么读</th></tr>
    <tr><td>MiMo-V2-Flash，TPU v7x，16K 进 4K 出，batch 64</td><td>两个 pod 各跑一份：11.70K total tok/s</td><td>1P1D 10.55K；2P1D 用三个 pod 也只有 10.83K</td><td>并发不高，拆了亏</td></tr>
    <tr><td>同上，batch 128</td><td>12.78K</td><td>1P1D 13.64K；2P1D 用三个 pod 15.31K</td><td>同样两个 pod，1P1D 多 6.7%；2P1D 赢在多一个 pod，按 pod 算低 20%</td></tr>
    <tr><td>Qwen3-Coder-480B，两台 v7x-8，1K 进 1K 出，单并发</td><td>一台：TTFT 中位数 95 ms</td><td>1P1D：281 ms</td><td>多一趟传输；两倍机器，输出只多 12%</td></tr>
  </table>
  <p><b>拆不拆跟着负载走，同一个模型可以两种答案。</b></p>
  <details class="foldfig"><summary><b>出处与口径</b>：两组 TPU 例子</summary>
  <p>MiMo：tpu/sglang-jax/mimo/README.md:8–12、81–88，每 pod 4 颗芯片，request_rate=inf，total tok/s 含输入。README 写 1P1D 比不拆高「约 6.8%」，按表算是 6.7%；2P1D 每 pod 5.10K 对不拆的 6.39K（本课推导）。</p>
  <p>Qwen3-Coder：tpu/tpu-inference/Qwen3-Coder-480B/README.md:1005–1021，TTFT 与 TPOT 都是中位数。两组都是单点测量。</p></details>

  <h3 id="s9-6">9.6　决策树：三个问题</h3>
  <p>答「否」就停。分块 prefill 见第四节，KV 薄不薄见 8.8。</p>
__FIG_S9_DECIDE__

  <h3 id="s9-7">9.7　AFD：再拆一层</h3>
  <p><a href="topic-05.html#s6-5">专题五 §6.5</a> 讲过：decode 里的注意力和专家也能拆到两批机器上，每一层来回一趟，靠几个小批轮流跑把通信藏住。</p>
__FIG_S9_AFD_T5__
  <p>字节的 MegaScale-Infer 报告每卡吞吐最多高 1.90 倍；阶跃星辰的 Step-3 在 Hopper 上、50 ms TPOT 的约束下做到每卡 4,039 tok/s，同条件的 DeepSeek-V3 是 2,324（Step-3 论文算的）。模型不同，差距不能全记给 AFD。</p>
  <p>开源实现没这么顺。vLLM 的 AFD 是外部插件，主仓库 RFC 已因不活跃被关闭；插件博客在 Ascend 910C 上的数，配比一错就比不拆还慢。</p>
__FIG_S9_AFD_ASCEND__
  <ul>
    <li><b>配比是关键</b>：16K 输入时 48 个注意力配 16 个专家，每 die（芯片上的一块计算单元）220.3，比不拆少 5.3%；注意力加到 64 个，涨到 258.9。作者说专家侧还有余力，注意力再加可能还涨。</li>
    <li><b>只能看趋势</b>：实验把专家路由换成强制均衡，还用小部署模拟大规模。</li>
    <li><b>TPU 侧</b>：没找到公开实现或实测。</li>
  </ul>
  <details class="foldfig"><summary><b>出处</b>：AFD 的几个数</summary>
  <p>MegaScale-Infer：arXiv 2504.02263。Step-3：arXiv 2507.19427（4K 上下文、FP8、不开 MTP）。vLLM AFD 插件：vllm.ai/blog/2026-07-23-vllm-afd-plugin、github.com/vllm-project/afd-plugin；RFC #22799 于 2026-05-31 被机器人关闭。</p></details>

  <h3 id="s9-8">9.8　EPD：多模态把编码也拆出去</h3>
  <p>多模态模型在 prefill 前要把图片、视频编码成 token，图一多就和 prefill 抢显存和算力。</p>
  <ul>
    <li><b>做法</b>：编码器单独一批机器，视觉 token 交给 prefill，prefill 再把 KV 交给 decode。</li>
    <li><b>论文的数</b>：arXiv 2501.05460 报告峰值显存最多降 15 倍，batch 最多大 22 倍，TTFT 最多降 71%。</li>
    <li><b>什么时候值得</b>：请求带很多图或大图时；纯文本用不上（专题五 §6.6）。</li>
  </ul>

  <h3 id="s9-9">9.9　落点</h3>
  <div class="note ok"><span class="t">回到主线</span>
    拆开是为了让这一趟不再等 prefill，两边切法各选各的；AFD 让每个专家一趟分到更多 token。
    拆不拆、怎么配，看负载，不看潮流。</div>

  <details class="foldfig"><summary><b>考考自己</b>：一份报告说「加到 12 个 prefill，每卡吞吐涨了 10 倍」。你先问什么？</summary>
  <p>先问分母：除的是 decode 卡还是全部卡。V4-Pro 那组，12 个 prefill 按 decode 卡算涨 10 倍多，按全部卡只涨 2.2 倍。</p></details>
'''

FIGS = {
    "__FIG_S9_RATIO_T5__": ("fig-s9-pd-ratio-t5", "fig5-pd-ratio.svg", "topic05-fig-pd.py",
        '<b>两边按上限配，谁也不等谁。</b><br><em>专题五原图，DistServe 论文的 OPT 例子。</em>'),
    "__FIG_S9_2P1D__": ("fig-s9-2p1d", "fig6-2p1d.svg", "topic06-fig-ratio.py",
        '<b>左边一高一低，右边两组都在降。</b><br><em>B200，并发 128；每卡 ＝ 总输出 ÷ 全部卡数。</em>'),
    "__FIG_S9_SWEEP__": ("fig-s9-prefill-sweep", "fig6-prefill-sweep.svg", "topic06-fig-ratio.py",
        '<b>同一组实测，两条线。</b><br><em>蓝线是 runbook 原口径，橙线是本课换了分母重算的。</em>'),
    "__FIG_S9_DECIDE__": ("fig-s9-decide", "fig6-decide.svg", "topic06-fig-ratio.py",
        '<b>三问，从上往下。</b><br><em>阈值没画，要拿自己的负载量。</em>'),
    "__FIG_S9_AFD_T5__": ("fig-s9-afd-t5", "fig5-afd.svg", "topic05-fig-pd.py",
        '<b>一层里的两种活，分到两批机器。</b><br><em>专题五原图，机器数是示意。</em>'),
    "__FIG_S9_AFD_ASCEND__": ("fig-s9-afd-ascend", "fig6-afd-ascend.svg", "topic06-fig-ratio.py",
        '<b>同样拆开，橙柱亏、绿柱赚。</b><br><em>每 die 吞吐的分母是部署的全部 die。</em>'),
}

NUM_LOCK = [
    ("1,159.0", "fig6-2p1d.svg"), ("48.3", "fig6-2p1d.svg"), ("6.23", "fig6-2p1d.svg"),
    ("3,187", "fig6-prefill-sweep.svg"), ("10.4", "fig6-prefill-sweep.svg"), ("455", "fig6-prefill-sweep.svg"),
    ("258.9", "fig6-afd-ascend.svg"), ("220.3", "fig6-afd-ascend.svg"),
]
