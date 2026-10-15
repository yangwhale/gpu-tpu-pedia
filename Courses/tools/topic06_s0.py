# -*- coding: utf-8 -*-
"""专题六 · 第零节「一次请求走一遍」。图在 topic06-fig-intro.py。"""

BODY = '''
  <p class="lead">同一个模型，吞 prompt 那一段吃算力，往外吐字那一段吃带宽。
    <b>一次请求里，前后两段要的硬件正好相反。</b></p>

  <h3 id="s0-1">0.1　一个请求的一生</h3>
__FIG_S0_LIFE__
  <p>用户只感觉得到两个数：等第一个字等了多久（TTFT），之后每个字隔多久（TPOT）。
    这两个数分别由 prefill 和 decode 决定，第二节再把它们的口径校准。</p>

  <h3 id="s0-2">0.2　全课一句话</h3>
  <div class="note ok"><span class="t">一句话</span>
    <b>decode 每一步都要把权重和 KV 搬一趟。推理的每一招，要么让这一趟分给更多 token，要么干脆少搬。</b><br>
    还有一类招数不碰这一趟本身：把吞 prompt 和吐字拆开，各放到合适的机器上。</div>
__FIG_S0_MAP__
  <p>第一节先把这一趟的账算清楚：搬多少字节、花多少时间、算力闲了多少。</p>

  <details class="foldfig"><summary><b>遇到不认识的词，回这儿查</b>：prefill、decode、KV cache、TTFT、TPOT、吞吐、屋脊点、chip 与 device、PD 分离、投机解码……（每条一句话）</summary>
  <p style="line-height:1.9"><b>prefill</b>：把整段 prompt 一次喂进模型算完，顺手产出第一个输出 token。<br>
    <b>decode</b>：之后每一步只往外出一个 token，出完接着喂回去算下一个。<br>
    <b>KV cache</b>：每个处理过的 token 在每一层留下的 K 和 V，后面的 token 算注意力时要回头读它们，所以一直存在显存里（专题三）。<br>
    <b>TTFT</b>：从请求发出到收到第一个输出 token 的时间；含不含排队要写清。<br>
    <b>TPOT 与 ITL</b>：TPOT 是一个请求除第一个 token 外平均每个 token 花的时间；ITL 是相邻两个 token 之间的每一次间隔，看 p99 时用它。<br>
    <b>batch 与并发</b>：并发是同时在服务的请求数；batch 是引擎这一步实际拼在一起算的请求（或 token）数，通常不超过并发。<br>
    <b>吞吐</b>：每秒出多少 token。要写清是总量还是每卡（每卡 ＝ 总量 ÷ 卡数，分母含不含 prefill 的卡），以及只数输出还是输入输出合计。<br>
    <b>MoE 激活参数</b>：MoE 模型每个 token 只经过挑中的几个专家，真正参与计算的那部分参数；DeepSeek-V3 总共 6,710 亿，每个 token 激活约 370 亿。<br>
    <b>HBM 带宽</b>：芯片每秒能从自带显存读写多少字节；TPU v7 每芯片 7.37 TB/s。<br>
    <b>屋脊点</b>：峰值算力 ÷ 显存带宽，即每读 1 字节至少要做多少次运算才不被带宽拖住；v7 按 bf16 算约 313（专题二）。<br>
    <b>chip 与 device</b>：一颗 TPU v7 芯片对软件显示成 2 个 device；本课的规格按每芯片写，并行度按 device 数。<br>
    <b>PD 分离</b>：prefill 和 decode 放到两批不同的机器上跑，中间把 KV cache 传过去（专题五第六节）。<br>
    <b>投机解码</b>：先用便宜的办法猜出后面几个 token，再让大模型一步把它们一起验掉，猜对几个就一次出几个。</p></details>
'''

FIGS = {
    "__FIG_S0_LIFE__": ("fig-s0-life", "fig6-life.svg", "topic06-fig-intro.py",
        '<b>红点是第一个字，之后每个绿格出一个字；紫色阶梯每一步都在长。</b><br>'
        '<em>示意时间线，长短不按比例。</em>'),
    "__FIG_S0_MAP__": ("fig-s0-map", "fig6-map.svg", "topic06-fig-intro.py",
        '<b>蓝、绿两栏直接改这一趟，橙栏换个地方搬。</b><br>'
        '<em>归类是本课的分法，不是业界通行的划分。</em>'),
}

NUM_LOCK = []
