# -*- coding: utf-8 -*-
"""专题六 · 第一节「两个阶段：prefill 吃算力，decode 吃带宽」。图在 topic06-fig-phases.py。"""

BODY = '''
  <p class="lead">训练、prefill、decode 摆在一起比：prefill 一次喂几千个 token（矩阵乘里每个 token 占一行），decode 每人每步只喂一个。
    <b>行数差了三个数量级，瓶颈就从算力换到了显存带宽。</b>0.1 里的每一招都冲着这个事实。</p>
  <table>
    <caption class="sub" style="caption-side:bottom;text-align:left">训练每步约 6 次运算／参数／token，前向占 2 次，反向约 4 次；推理只有前向。</caption>
    <tr><th></th><th>训练</th><th>prefill</th><th>decode</th></tr>
    <tr><td>一步喂进去的</td><td>整批完整序列</td><td>一段 prompt，几千行</td><td>每人一行</td></tr>
    <tr><td>算什么</td><td>前向＋反向＋更新</td><td>只有前向</td><td>只有前向</td></tr>
    <tr><td>序列能不能并行</td><td>能：答案已知，整段一起算</td><td>能：prompt 已知</td><td>不能：下一个字要等上一个</td></tr>
    <tr><td>卡在哪</td><td>算力、通信</td><td>算力</td><td>显存带宽</td></tr>
    <tr><td>显存装什么</td><td>权重、梯度、优化器状态</td><td colspan="2">权重＋每人的 KV</td></tr>
    <tr><td>看什么</td><td>吞吐</td><td>首字延迟</td><td>字间隔，以及同时服务多少人</td></tr>
  </table>
  <p><b>prefill 就是训练的前向。</b>同样是大矩阵乘、吃算力，专题五那套切法和 kernel 大多能直接搬过来。</p>
  <p><b>decode 是另一个故事。</b>训练时答案已知，整段序列一起算；decode 每一步只能出一个字，却要把全部权重和每人的 KV 读一遍。请求又随时到、长短不一，还要守住延迟。一颗很贵的芯片，大半时间在等搬运。</p>

  <h3 id="s1-1">1.1　一次喂几行，决定卡在哪</h3>
  <p>v7 按 bf16（每个数 2 字节）算，每读 1 字节要做满 313 次运算才不被带宽拖住，这叫屋脊点。
    矩阵乘一次喂进 M 行，每个权重就要做 M 次乘加（2M 次运算），而它只占 2 字节：只算读权重，每字节的运算次数（强度）就是 2M ÷ 2 ＝ M。</p>
__FIG_S1_AI__
  <p>曲线比 M 低一点，是因为输入、输出也要读写；M 越大它们越多，强度最多到 hidden（每个字的向量长度，V3 是 7,168）的一半。</p>

  <h3 id="s1-2">1.2　一个人解码一步：5 毫秒在读，0.016 毫秒在算</h3>
  <p>decode 每出一个字，都要把 V3 的 370 亿激活参数从显存读一遍；按 FP8（每个 1 字节）存，就是 37 GB。</p>
__FIG_S1_B1__
  <p>算的一侧按 W8A8（权重和激活值都用 8 位）计。这 5 毫秒是每步用时的下限，实测离它越近，带宽用得越满。可这一趟只出了一个字。</p>

  <h3 id="s1-3">1.3　上下文一长，KV 接过主角</h3>
  <p>权重一次读完能分给一整批人，KV 却是每人自己的历史。</p>
__FIG_S1_WKV__
  <p>这还是 MLA 压过的 KV（专题五 128K 那份 8.58 GiB）。</p>

  <h3 id="s1-4">1.4　MoE 把拼好的 batch 又拆散</h3>
  <p>拼 batch，就是让读一次的权重多喂几行。
    可 V3 每层有 256 个路由专家，每人只挑 8 个；256 人进来，每个专家平均只分到 8 行，按 1.1 强度也只有约 8。</p>
__FIG_S1_MOE__

  <div class="note ok"><span class="t">这一节只要带走一件事</span>
    <b>prefill 卡算力，decode 卡带宽；decode 每一步都要把权重和 KV 搬一趟。</b>
    要让这一趟分给更多人，先得让批一直坐满、装下更多人的 KV（第三、四节），再把被 MoE 拆散的行收回来（第五、六节）。</div>
'''

FIGS = {
    "__FIG_S1_AI__": ("fig-s1-ai", "fig6-ai.svg", "topic06-fig-phases.py",
        '<b>绿点全在红线下面，橙点在上面。</b><br>'
        '<em>读权重、读输入、写输出都算进字节；本课推导。</em>'),
    "__FIG_S1_B1__": ("fig-s1-b1", "fig6-b1.svg", "topic06-fig-phases.py",
        '<b>橙条细到几乎看不见。</b><br>'
        '<em>理想下限：只算权重，不算 KV、通信和 kernel 开销；按每芯片规格，摊到多颗上比例不变。</em>'),
    "__FIG_S1_WKV__": ("fig-s1-wkv", "fig6-wkv.svg", "topic06-fig-phases.py",
        '<b>蓝段随人数缩短，紫段只随上下文变长。</b><br>'
        '<em>64 人时 87% 的专家被碰到（一个专家没人挑的概率 (1 − 8/256)<sup>64</sup> ≈ 13%），所以每人摊 9.14 GB，不是 37 ÷ 64。路由均匀、上下文等长；不含激活值与通信。</em>'),
    "__FIG_S1_MOE__": ("fig-s1-moe", "fig6-moe-rows.svg", "topic06-fig-phases.py",
        '<b>左边填满一边，右边只剩一条细线。</b><br>'
        '<em>8 是均匀路由下的平均数；共享专家不受影响。</em>'),
}

NUM_LOCK = [
    ("313", "fig6-ai.svg"),
]
