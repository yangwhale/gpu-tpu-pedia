# -*- coding: utf-8 -*-
"""专题六 · 第一节「两个阶段：prefill 吃算力，decode 吃带宽」。图在 topic06-fig-phases.py。"""

BODY = '''
  <p class="lead">prefill 一次喂几千行，decode 一次每人只喂一行。
    <b>行数差了三个数量级，瓶颈就从算力换到了显存带宽。</b></p>

  <h3 id="s1-1">1.1　一次喂几行，决定卡在哪</h3>
  <p>屋脊点是专题二那把尺：v7 每读 1 字节要做 313 次运算，算力才不闲着。
    矩阵乘一次喂进 M 行，强度大致跟着 M 涨。</p>
__FIG_S1_AI__
  <p>decode 拼 256 个人也只有 239，要到 M ≈ 343 才过线。
    prefill 一段 4K prompt 就是 1,911，稳稳卡在算力上。</p>

  <h3 id="s1-2">1.2　一个人解码一步：5 毫秒在读，0.016 毫秒在算</h3>
  <p>DeepSeek-V3 每个字激活约 370 亿参数，FP8 存，一步要读 37 GB。
    按一颗 v7 的规格算，账是这样的：</p>
__FIG_S1_B1__
  <p>读完要 5.02 ms，算完只要 0.016 ms，算力利用率 0.32%。
    这就是 decode 的字节下限：步时离它越近，说明带宽用得越满。</p>

  <h3 id="s1-3">1.3　上下文一长，KV 接过主角</h3>
  <p>权重一次读完能分给一整批人，KV 却是每人自己的历史。
    拼的人越多、上下文越长，KV 在这一趟里的份量越大。</p>
__FIG_S1_WKV__
  <p>128K 时每个字要读这个人自己的 9.21 GB KV（MLA 已经压过一轮，专题三）。
    256 人一起跑，KV 占到 78%，光靠拼 batch 已经摊不动。</p>

  <h3 id="s1-4">1.4　MoE 把拼好的 batch 又拆散</h3>
  <p>拼 batch 是为了让一块权重多分几行。
    可 MoE 每人只挑 8 个专家，256 人进来，到每个专家手里平均只剩 8 行。</p>
__FIG_S1_MOE__
  <p>v7 的 MXU 一边是 256，8 行只喂满 1/32。
    专家怎么切、怎么把各卡的人收到一起，接着专题一的 MoE 往下讲（第五、六节）。</p>

  <div class="note ok"><span class="t">这一节只要带走一件事</span>
    <b>prefill 卡算力，decode 卡带宽；decode 每一步都要把权重和 KV 搬一趟。</b><br>
    后面每一节，要么把这一趟分给更多 token，要么让它少搬几个字节，要么把两段拆开各放各的机器。</div>
'''

FIGS = {
    "__FIG_S1_AI__": ("fig-s1-ai", "fig6-ai.svg", "topic06-fig-phases.py",
        '<b>绿点全在红线下面，橙点在上面。</b><br>'
        '<em>7168×7168 bf16 矩阵乘，读权重、读输入、写输出都算进字节；本课推导。</em>'),
    "__FIG_S1_B1__": ("fig-s1-b1", "fig6-b1.svg", "topic06-fig-phases.py",
        '<b>橙条细到几乎看不见。</b><br>'
        '<em>理想下限：只算权重，不算 KV、通信和 kernel 开销；按每芯片规格，摊到多颗上比例不变。</em>'),
    "__FIG_S1_WKV__": ("fig-s1-wkv", "fig6-wkv.svg", "topic06-fig-phases.py",
        '<b>蓝段随人数缩短，紫段只随上下文变长。</b><br>'
        '<em>FP8 权重、bf16 KV，假设路由均匀、人人上下文一样长；不含激活与通信。</em>'),
    "__FIG_S1_MOE__": ("fig-s1-moe", "fig6-moe-rows.svg", "topic06-fig-phases.py",
        '<b>同一批人，左边填满一边，右边只剩一条细线。</b><br>'
        '<em>8 是均匀路由下的平均数；共享专家不受影响。</em>'),
}

NUM_LOCK = [
    ("313", "fig6-ai.svg"),
    ("239", "fig6-ai.svg"),
    ("343", "fig6-ai.svg"),
    ("1,911", "fig6-ai.svg"),
    ("5.02", "fig6-b1.svg"),
    ("0.016", "fig6-b1.svg"),
    ("0.32%", "fig6-b1.svg"),
    ("9.21", "fig6-wkv.svg"),
    ("78%", "fig6-wkv.svg"),
    ("1/32", "fig6-moe-rows.svg"),
]
