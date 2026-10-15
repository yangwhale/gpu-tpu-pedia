# -*- coding: utf-8 -*-
"""专题六 · 第二节「先校准尺子：指标与口径」。图在 topic06-fig-metrics.py。

⛔ 数字全部来自公开出处或本仓库公开 runbook，行号写在图脚本的 docstring 和各图的「出处与口径」里。
   DeepSeek 成本那段出自 deepseek-ai/open-infra-index 的 202502OpenSourceWeek/day_6 文档。
"""

BODY = '''
  <p class="lead">同一次测量，按一种算法是每卡 5,270，按另一种是 11,880。
    <b>先把尺子对齐，后面每一节的数才能放在一起比。</b></p>

  <h3 id="s2-1">2.1　四个时间指标，各量哪一段</h3>
__FIG_S2_METRICS__
  <p>TTFT 是进门到第一个字。压测时它往往大半是排队：同一个 Gemma4-31B，并发 1 时 86 ms，并发 256 时 7,756 ms。</p>
  <p>TPOT 是首字之后平均每个字的间隔，每个请求一个数；每用户 tok/s 约等于 1000 ÷ TPOT(ms)。
    ITL 是逐次间隔，开了投机解码时一次可能吐两三个字，看 p99 要看它。</p>
  <div class="note warn"><span class="t">同一个词，两家定义</span>
    NVIDIA NIM 把 ITL 定义成 (e2e − TTFT) ÷ (字数 − 1)，并注明「也叫 TPOT」；vLLM 的 ITL 是逐 chunk 的间隔列表。
    拿两家工具的数对比之前，先查定义。</div>

  <h3 id="s2-2">2.2　吞吐和交互性是一条曲线</h3>
__FIG_S2_PARETO__
  <p>同一条 vLLM 曲线上，每用户 68 tok/s 时每卡 9,759，提到 173 时只剩 895。
    让每个人等久一点，同一趟就能分给更多人，每卡吞吐自然上去。</p>
  <p>所以比较的顺序是：先定交互档，再在这一档上比吞吐。PyTorch 博客报的 11,200 就钉在每用户约 50 tok/s 这一档上。</p>
  <p>DistServe（arXiv 2401.09670）把这件事做成一个指标：goodput，只数同时满足 TTFT 和 TPOT 目标的请求。
    按这个尺子，它报告能多服务 7.4 倍的请求，或者扛住紧 12.6 倍的延迟目标。</p>

  <h3 id="s2-3">2.3　同一次测量，四把尺子</h3>
__FIG_S2_RULERS__
  <p>分子只数输出还是输入输出合计，分母只数 decode 卡还是全部卡，排列组合就是四个数。
    InferenceX 的两个字段各占一个：output ÷ decode GPU 是 5,270，(输入＋输出) ÷ 全部 GPU 是 11,880。</p>
  <div class="note ok"><span class="t">看到一个百分比或每卡吞吐，先问两句</span>
    <b>一，分母是谁？</b>只数 decode 卡，就把 prefill 的成本藏起来了。<br>
    <b>二，每卡还是总量，卡按什么数？</b>一颗 TPU v7 芯片对软件显示成 2 个 device，按 device 报的每卡数只有按芯片报的一半。</div>

  <h3 id="s2-4">2.4　尺子以外的四个坑</h3>
__FIG_S2_TRAPS__
  <ul>
    <li><b>假天花板</b>：并发扫描在 9,200–9,400 一带涨不动，当时以为到顶了。只改 KV 池的划分比例，同为 4K 输入的端到端从 6,887 涨到 10,614，再调显存比例到 10,704。扫出来的上界只属于那组参数。</li>
    <li><b>峰值和稳态</b>：引擎自报的峰值是某一刻的最高值。同一轮里峰值 12,070、端到端 10,704。</li>
    <li><b>温度</b>：草稿模型按 greedy 猜，服务端却按默认温度 1.0 采样，接受率只有 1.16%；改成温度 0 是 34%，输出 2,911 对 944。草稿按概率采样时没有这个坑。</li>
    <li><b>预热和闭环</b>：第一轮要编译、要热缓存，报第二轮。限并发的闭环压测比开环低 13%，对标要开环。</li>
  </ul>

  <h3 id="s2-5">2.5　落到钱上</h3>
  <p>DeepSeek 公布过一天的账：平均 226.75 个 H800 节点，按每卡每小时 2 美元算，一天成本 87,072 美元。
    如果全部按 R1 定价收费，理论日收入 562,027 美元，成本利润率 545%。</p>
  <p>它自己也写明实际收入低得多：V3 更便宜，网页和 App 免费，夜间打折。
    同一份文档里，每个节点 decode 时每秒约出 14,800 个字，平均每用户 20–22 tok/s，这又是一个交互档的选择。</p>

  <div class="note ok"><span class="t">这一节只要带走一件事</span>
    报一个吞吐，要连同五样一起报：<b>分子分母、交互档、峰值还是稳态、温度和采样、有没有预热和是不是开环。</b>
    缺一样，两个数的差距可能全是口径。</div>

  <details class="foldfig"><summary><b>考考自己</b>：一份材料写「每卡 12,000 tok/s，超过官方」。你要先问哪三件事？</summary>
  <p>分母只算 decode 卡还是全部卡、分子含不含输入；在每用户多少 tok/s 的档位上测的；是峰值还是端到端稳态。
    放在 TPU 上还要再问一句：卡按芯片数还是 device 数。</p></details>
'''

FIGS = {
    "__FIG_S2_METRICS__": ("fig-s2-metrics", "fig6-metrics.svg", "topic06-fig-metrics.py",
        '<b>TTFT 含排队，TPOT 是平均，ITL 是逐次。</b><br>'
        '<em>时间线是示意；下方锚点是 TPU v7x 上 Gemma4-31B 的实测。</em>'),
    "__FIG_S2_PARETO__": ("fig-s2-pareto", "fig6-pareto.svg", "topic06-fig-metrics.py",
        '<b>沿着曲线往左走，每卡吞吐就往上涨。</b><br>'
        '<em>只有绿色三点是同一条曲线；其余点分属不同框架和配置。</em>'),
    "__FIG_S2_RULERS__": ("fig-s2-rulers", "fig6-rulers.svg", "topic06-fig-metrics.py",
        '<b>同一组原始数，最大和最小差 36 倍。</b><br>'
        '<em>灰色两条是本课推导，没有工具这样报。</em>'),
    "__FIG_S2_TRAPS__": ("fig-s2-traps", "fig6-traps.svg", "topic06-fig-metrics.py",
        '<b>温度那一格差得最多：接受率差 29 倍。</b><br>'
        '<em>假天花板一格同为 4K 输入；其余各格是独立的几组实测。</em>'),
}

NUM_LOCK = [
    ("7,756", "fig6-metrics.svg"),
    ("9,759", "fig6-pareto.svg"),
    ("11,200", "fig6-pareto.svg"),
    ("5,270", "fig6-rulers.svg"),
    ("11,880", "fig6-rulers.svg"),
    ("10,614", "fig6-traps.svg"),
    ("10,704", "fig6-traps.svg"),
    ("12,070", "fig6-traps.svg"),
    ("1.16%", "fig6-traps.svg"),
    ("2,911", "fig6-traps.svg"),
]
