# -*- coding: utf-8 -*-
r"""专题三 · §六「旋钮②：每步只读一部分」的图（2026-09-12 加）。

⭐⭐ 这一节自己写着一句话，直接论证了这张图该存在：
   「（attention sink）这个 bug **从公式上完全看不出来，
     只有把注意力矩阵画出来看才发现**。」
   ——&nbsp;那就把这一支的五种读法，**全部画成 mask**。

⛔ 这五种方案用文字并列讲，听完只会记住五个名词；
   画成五张 mask 并排，**它们的亲缘关系和演进方向一眼就出来**：
   从「砍成一条带」→「补回停车位」→「学着挑」→「先压再挑」。

📌 数字当场算并断言；效果数据全部标了出处与口径。
"""
from topic03_draw import (Fig, wpx, _sz,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
GiB = 2 ** 30
N = 16          # mask 边长（格）
C = 15.0        # 格子像素（原 8.5 ——&nbsp;一行挤五张的时候只能这么小）


def main():
    # 1M 上下文下 GQA-8 的 KV，用来兑现「2%」那笔账
    gqa_1m = 2 * 8 * 128 * 61 * 2 * 1048576 / GiB
    assert 230 < gqa_1m < 250, gqa_1m

    f = Fig(W, "旋钮二的五种读法画成五张注意力 mask：全注意力、滑动窗口、"
               "滑窗加 sink、DSA 学着挑 top-k、CSA 先压缩再挑。"
               "它们的共同结构是先用一个便宜的办法决定看哪些，再只对那些做主注意力")
    f.marks = set()
    y = f.header(
        '旋钮 ② 每步只读一部分 ——&#160;'
        '<tspan font-weight="700">五种读法，画成五张 mask 就看明白了</tspan>',
        "⭐ 这一支的历史本身就说明该画图：attention sink 那个 bug 从公式上看不出来，"
        "是把矩阵画出来才发现的")

    MW = 440
    # ⛔⛔ 麻瓜视角：「怎么读这张图」的钥匙原来在**五张图全部之下**（y≈863）。
    #   读者先按 16×16 的格子读出一个错的密度，翻到底才被告知格子只是示意。
    # ⭐ 提到图前一行 —— 落点带里那两段长解释留在下面，这里只要一句。
    f.t(0, y + 22, "⚠️ 先说怎么读："
        "<tspan font-weight=\"700\">格子是示意</tspan>（16×16 画不出 1.56% 那种量级），"
        "每张下面那条<tspan font-weight=\"700\">细带才是真比例</tspan> ——　"
        "读格子看形状，读细带看狠不狠。", RD, size=17, w=1396)
    TOP = y + 58
    # 128K 上下文下**真正被读到**的那一份占全部历史的比例。
    # ⚠️ CSA 那格留空：V4 报的 2% 是 1M 下的**存储**口径、基线也不是这里的全注意力，
    #   ⭐ 与其凑一个可比的假数，不如把「不可比」写在图上。
    REAL = [
        (1.0, "真实比例：100%（基线）"),
        (4096 / 131072.0, "真实比例：3.1%（窗口 4096 / 128K）"),
        (4100 / 131072.0, "真实比例：3.1% ＋ 4 个 sink"),
        (2048 / 131072.0, "真实比例：1.56%（k=2048 / 128K）"),
        (0.0, "⚠️ 口径不同，不能并排比（见下）"),
    ]
    MASKS = [
        ("全注意力", GY, "基线：下三角全算", "O(L²)，KV 随长度线性涨",
         lambda r, c: c <= r),
        ("SWA 滑动窗口", OR, "只看前面固定窗口（Mistral 4096）",
         "⛔ 跨 128K 要堆 32 层才摸得到",
         lambda r, c: c <= r and r - c < 4),
        ("＋ Attention sink", RD, "滑窗 ＋ 留住最开头 4 个",
         "⭐ 只留 4 个就够，扔了立刻崩",
         lambda r, c: (c <= r and r - c < 4) or c < 4),
        ("DSA 学着挑", BL, "Indexer 给每个 query 挑 top-k",
         "128K → 2K，64 倍；k=2048",
         lambda r, c: c <= r and (r - c < 2 or c < 1 or (c * 7 + r * 3) % 11 == 0)),
        ("CSA 先压再挑", PU, "每 4 个 token 压成 1 个，再在压缩后挑",
         "V4：1M 下 KV 降到约 2%",
         lambda r, c: c <= r and (r - c < 3
                                  or (c % 4 == 0 and (c * 5 + r) % 9 < 5))),
    ]
    # ⛔ 原来五张并排一行（每张 250px）。⭐ 改成 3＋2 两行，每张 440px ——
    #   格子从 8.5px 变成 15px，标签从 11px 变成 16–19px。
    for i, (nm, col, how, cost, fn) in enumerate(MASKS):
        x = (i % 3) * (MW + 40)
        ROWH = 372
        ty = TOP + (i // 3) * ROWH
        f.t(x, ty, nm, col, bold=True, size=_sz(19), cls="svglbl")
        f.t(x, ty + 24, how, GY, size=15, w=MW)
        gy_ = ty + 38
        for r in range(N):
            for c in range(N):
                if c > r:
                    continue
                on = fn(r, c)
                f.box(x + c * C, gy_ + r * C, C - 1.2, C - 1.2,
                      col if on else "#eef1f3", "none", 1)
        f.t(x, gy_ + N * C + 24, cost, col, size=16, w=MW)
        # ── 按真实比例的一条细带（16×16 的格子撑不住 1.56% 这种量级）──
        fr, note = REAL[i]
        by_ = gy_ + N * C + 30
        f.box(x, by_, MW - 12, 9, "#fff", LINE, 2)
        f.box(x + 0.8, by_ + 0.8, max(0.8, (MW - 13.6) * fr), 7.4, col,
              "none", 2)
        f.t(x, by_ + 28, note, GY2, size=14, w=MW)

    yy = TOP + 372 + 300 + 56

    # ⭐ 2026-09-12：这一条原来写的是「先用便宜办法决定看哪些」——&nbsp;对，但泛。
    #   教材 6.6 那个「三条路」框架更准，而且它原来是 <pre> 里的 ASCII 画 ——
    #   ⭐ **ASCII 画本来就是「想画图但手边只有文本」的产物**，搬进真图里。
    yy = f.band(yy, "warn", "那条细带的口径（上面那句的完整版）", [
        "16×16 的格子<tspan font-weight=\"700\">撑不住 1.56% 这种量级</tspan>："
        "画得出来的最细的窗口也有 1/16 ＝ 6%，而真实是 1.56%～3.1%。"
        "⭐ 所以每张下面加了一条<tspan font-weight=\"700\">按真比例画的细带</tspan>。",
        "⛔ 读格子看<tspan font-weight=\"700\">形状</tspan>（哪里有、哪里没有），"
        "读细带看<tspan font-weight=\"700\">狠不狠</tspan>。"
        "⚠️ CSA 那条空着是故意的：V4 报的 2% 是 <tspan font-weight=\"700\">1M 下的存储量</tspan>，"
        "跟前四条的「128K 下读多少」<tspan font-weight=\"700\">不是一个口径</tspan> ——&#160;"
        "凑一个可比的假数还不如把「不可比」写在图上。",
    ], fold=True)

    yy = f.band(yy + 14, "info", "五张 mask 摆在一起，这一支的共同结构就出来了", [
        '<tspan font-weight="700">同一个骨架，三条路</tspan>：'
        '<tspan font-weight="700">粗看</tspan>（压缩／全局，保证不漏）＋'
        '<tspan font-weight="700">细看</tspan>（挑出来的 top-k，保证准）＋'
        '<tspan font-weight="700">近处</tspan>（滑动窗口，保证局部连贯）。',
        'NSA 三条都有（显式三支路 ＋ 门控融合）；DSA 主要是<tspan font-weight="700">细看</tspan>配一点局部；'
        'CSA/HCA ——&#160;HCA 粗看、CSA 细看，另挂一条滑窗；'
        '<tspan font-weight="700">SWA 单用只有第三条 ——&#160;所以它单用不行</tspan>。',
        '⭐⭐ <tspan font-weight="700">看到一个新的稀疏注意力方案，先问它这三条路各占什么位置</tspan>'
        '——&#160;<tspan font-weight="700">这比记住它叫什么有用得多。</tspan>',
        '⛔ <tspan font-weight="700">不是免费午餐</tspan>：DSA 的 indexer '
        '<tspan font-weight="700">自己仍然是 O(L²)</tspan> ——&#160;每个 query 要给所有 KV 打分才能挑 top-k。'
        '<tspan font-weight="700">省的是常数，不是阶。</tspan>',
        '⛔ 而且它<tspan font-weight="700">需要专门的训练阶段</tspan>：冻住其余参数，'
        '让 indexer 去<tspan font-weight="700">拟合主注意力自己的分布</tspan>'
        '——&#160;<tspan font-weight="700">稀疏是学出来的，不是规则定出来的</tspan>。'])

    yy = f.band(yy + 14, "bad", "⭐⭐ Attention sink：一个「只有画出来才看得见」的 bug", [
        '朴素滑窗把开头几个 token 一起滑掉，<tspan font-weight="700">模型立刻崩</tspan>。'
        '而解法简单到荒谬 ——&#160;<tspan font-weight="700">留住最开头 4 个就够</tspan>'
        '（原话 “with just 4 initial tokens sufficing”）。',
        '为什么？<tspan font-weight="700">softmax 强制所有权重加起来等于 1</tspan> ——&#160;'
        '模型有时什么都不想看，<tspan font-weight="700">却没有「弃权」这个选项</tspan>，'
        '于是学会把多余的注意力<tspan font-weight="700">倾倒在开头几个位置</tspan>。'
        '<tspan font-weight="700">那几个 token 不是在传信息，是停车位。</tspan>',
        '⭐ 两层教训：<tspan font-weight="700">工程上</tspan>，任何「扔掉一部分 KV」的方案'
        '都要先问<tspan font-weight="700">有没有扔掉停车位</tspan>；'
        '<tspan font-weight="700">方法上</tspan>，这个 bug 公式里完全看不出来 ——&#160;'
        '<tspan font-weight="700">量出来的和算出来的，是两回事。</tspan>',
        '📌 后续：V4 干脆给每个头加了一个<tspan font-weight="700">可学习的 sink logit</tspan>，'
        '直接进 softmax 分母 ——&#160;于是这一行的注意力总和<tspan font-weight="700">'
        '可以小于 1、甚至接近 0</tspan>。<tspan font-weight="700">'
        '把模型被迫发明的 hack，变成了架构里的一等公民。</tspan>'])

    yy = f.band(yy + 14, "ok", "这三年走了多远 ——&#160;把 2% 那个数算给学生看", [
        # ⛔ 这一行里有**字面的百分号**（「约 2%」）。用 % 格式化时它会被当成
        #   格式符 —— 2026-09-12 当场报 TypeError。⭐ 改用 .format()，
        #   判据：**正文里带 % 的字符串，别用 % 格式化。**
        ('同样形状的 <tspan font-weight="700">GQA-8 在 1M 上下文下是 {0:.0f} GiB</tspan>；'
         'V4 报的 <tspan font-weight="700">约 2%</tspan> 就是'
         '<tspan font-weight="700">不到 5 GiB</tspan> ——&#160;'
         '<tspan font-weight="700">一百万 token 的上下文，KV 装得进一块卡的零头。</tspan>'
         ).format(gqa_1m),
        '⭐ 对照本讲开头那个 <tspan font-weight="700">MHA 的 488 GiB ——&#160;而那还只是 128K</tspan>。'
        '<tspan font-weight="700">这就是三年的进展。</tspan>',
        '⚠️ 这个 2% 的基线是<tspan font-weight="700">同形状的 GQA-8</tspan>；'
        '那张 CSA 图给的「10% 的 KV」比的是 <tspan font-weight="700">V3.2</tspan> ——&#160;'
        '<tspan font-weight="700">两个数都对，对的是各自的基线</tspan>，别并排比。',
        '⚠️ 而且报总收益时要能<tspan font-weight="700">拆开</tspan>：V4 那一版还叠了一层'
        '<tspan font-weight="700">跟注意力机制无关</tspan>的优化 ——&#160;'
        'KV 混合精度（RoPE 那几维 BF16、其余 FP8），<tspan font-weight="700">光这一项就近乎减半</tspan>。'])

    yy = f.src(yy + 16,
               ('mask 图案为<tspan font-weight="700">示意</tspan>，用来表达各方案的读取形状，'
                '不是实测注意力分布；GQA-8 在 1M 下的 {0:.0f} GiB 由公式当场算出（脚本带断言）'
                ).format(gqa_1m),
               "SWA：Mistral 7B arXiv 2310.06825　sink：StreamingLLM arXiv 2309.17453　"
               "NSA：arXiv 2502.11089　DSA：DeepSeek-V3.2 arXiv 2512.02556　"
               "CSA/HCA：DeepSeek-V4 arXiv 2606.19348")
    f.save("fig3-knob2.svg", yy + 6)


main()
