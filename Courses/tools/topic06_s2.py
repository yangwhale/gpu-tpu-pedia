# -*- coding: utf-8 -*-
"""专题六 · 第二节「先校准尺子：指标与口径」。图在 topic06-fig-metrics.py。

⛔ 数字全部来自公开出处或本仓库公开 runbook，行号写在图脚本的 docstring 和各图的「出处与口径」里。
   DeepSeek 成本那段出自 deepseek-ai/open-infra-index 的 202502OpenSourceWeek/day_6 文档。
"""

BODY = '''
  <p class="lead">要说清后面每一招省了多少，先得有把尺子。可同一次测量，换一种数法，每卡可以是 5,270，也可以是 11,880。
    <b>先把尺子对齐，后面每一节的数才能放在一起比。</b></p>

  <h3 id="s2-1">2.1　四个时间指标，各量哪一段</h3>
__FIG_S2_METRICS__
  <p>TTFT 压测时往往大半是排队：同一个 Gemma4-31B，并发 1 时 86 ms，并发 256 时 7,756 ms。</p>
    <p>同一个词两家定义可能不同：NVIDIA NIM 的 ITL 是平均值，注明「也叫 TPOT」；vLLM 的 ITL 是逐次间隔的列表。</p>

  <h3 id="s2-2">2.2　吞吐和交互性是一条曲线</h3>
__FIG_S2_PARETO__
  <p>每人等久一点，同一趟就分给更多人。所以先定交互档，再在这一档上比吞吐。GB300 跑 V4-Pro，PyTorch 博客的 11,200 和模型首发当天的约 2,200 同在每用户约 50 tok/s 这一档。</p>
  <p>DistServe（2024，arXiv 2401.09670）的 goodput：规定比例（如 90%）的请求同时满足 TTFT 和 TPOT 目标时，每张卡能扛的最大请求速率。</p>

  <h3 id="s2-3">2.3　同一次测量，四把尺子</h3>
__FIG_S2_RULERS__
  <p>PD 分离（第八节）下分子分母各有两种数法（InferenceX 是 SemiAnalysis 的公开基准）。</p>

  <h3 id="s2-4">2.4　尺子以外的四个坑</h3>
__FIG_S2_TRAPS__
  <ul>
    <li><b>上限依赖负载</b>：换负载或旋钮，上限就变：3.4 只挪了一块隔板，每卡就涨 54%。</li>
    <li><b>峰值和稳态</b>：引擎自报的峰值是某一刻的最高值，比整轮端到端高一截；对标报端到端。</li>
    <li><b>温度</b>：选字的随机程度，0 即永远挑最可能的字。压测和线上不对齐，投机解码的接受率会差一大截（11.4）。</li>
    <li><b>预热和闭环</b>：第一轮要编译、热缓存，报第二轮。闭环是「答完一个才发下一个」，开环按固定节奏发；这组实测闭环低 13%，对标要用同一种。</li>
  </ul>

  <h3 id="s2-5">2.5　落到钱上</h3>
  <p>DeepSeek 公布过一天的账：全按 R1 定价收费，理论成本利润率 545%，它也写明实际收入低得多。
    同一文档里，decode 时每节点输出约 14,800 tok/s，每用户只有 20–22 tok/s：便宜是拿交互档换来的。</p>
  <p>换算到每个字：每百万输出 token 的成本 ＝ 卡时价 ÷（每卡 tok/s × 3,600）× 10⁶。卡时价不变，2.2 那一档每卡从约 2,200 到约 11,200，每个字的成本就降到约五分之一（本课推导）。</p>

  <div class="note ok"><span class="t">这一节只要带走一件事</span>
    报一个吞吐，要连同五样一起报：<b>分子分母、交互档、峰值还是稳态、温度和采样、有没有预热和是不是开环。</b>
    缺一样，差距可能全是口径。</div>

  <details class="foldfig"><summary><b>考考自己</b>：一份材料写「每卡 12,000 tok/s，超过官方」。你要先问哪三件事？</summary>
  <p>分母只算 decode 卡还是全部、分子含不含输入；在每用户多少 tok/s 的档位测的；峰值还是稳态。TPU 上再问：卡按芯片还是 device 数。</p></details>
'''

FIGS = {
    "__FIG_S2_METRICS__": ("fig-s2-metrics", "fig6-metrics.svg", "topic06-fig-metrics.py",
        '<b>比三根横杠的起点：只有 TTFT 从进门算起。</b><br>'
        '<em>时间线是示意，绿块宽度按字数画。</em>'),
    "__FIG_S2_PARETO__": ("fig-s2-pareto", "fig6-pareto.svg", "topic06-fig-metrics.py",
        '<b>沿着曲线往左走，每卡吞吐就往上涨。</b><br>'
        '<em>只有绿色三点是同一条曲线；其余点分属不同框架和配置。</em>'),
    "__FIG_S2_RULERS__": ("fig-s2-rulers", "fig6-rulers.svg", "topic06-fig-metrics.py",
        '<b>先比框住的格子再看结果条：分子框得越多、分母框得越少，数越大。</b><br>'
        '<em>虚线两条是本课推导，没有工具这样报。</em>'),
    "__FIG_S2_TRAPS__": ("fig-s2-traps", "fig6-traps.svg", "topic06-fig-metrics.py",
        '<b>左边的坑让数虚高，右边的坑让数虚低。</b><br>'
        '<em>两格各是独立实测。</em>'),
}

NUM_LOCK = [
    ("11,200", "fig6-pareto.svg"),
    ("5,270", "fig6-rulers.svg"),
    ("11,880", "fig6-rulers.svg"),
]
