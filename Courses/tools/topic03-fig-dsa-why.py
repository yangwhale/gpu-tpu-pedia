# -*- coding: utf-8 -*-
r"""专题三 · §六「凭什么只看 2048 个就够」（2026-09-13 夜间 20 轮 · R3）。

⭐⭐ 现场原话：「DSA 他又是怎么想的？为什么选 top 2048 就足够？
   太远的那个序列距离，也未必都跟你有关系。」

三格，回答的其实是**三个不同的问题**（这一节最容易混为一谈）：

  ① **「能不能只看一小部分」** ——&nbsp;这是个**经验事实**，而且先于 DSA：
     H2O（2023）量到密集训出来的模型，推理时注意力矩阵 **95% 以上是稀疏的**，
     累计注意力分数服从**幂律**。少数位置吃掉绝大部分权重，长尾每个都接近 0。

  ② **「那怎么知道该看谁」** ——&nbsp;这才是难点，而且是个**鸡生蛋**：
     要判断谁重要，得先算注意力分数；而那正是你想省掉的东西。
     ⭐⭐ DSA 的解法不是设计一个启发式去猜，是**让真注意力当老师**：
     热身阶段保持密集、冻住全模型，只训索引器去拟合
     「主注意力跨头求和再归一」的那个分布，损失就是 KL。

  ③ **「这个便宜的老师复制品，凭什么算得动」** ——&nbsp;头少、ReLU、FP8；
     外加两个干净的设计决策（梯度断开、稀疏期只在选中集合上算 KL），
     以及一条硬件约束：**它必须搭 MQA 模式**，否则 kernel 上不划算 ——
     ⭐ 旋钮①和旋钮②在这里被硬件绑在了一起。

⛔ 诚实口径：**论文没有给 k 的消融。** 2048 这个具体数字，
   能说的只有「128K 下占 1.6%」和「短于 2048 时 top-k 等于全选」。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        """⛔ 面板内容溢出底边不会报错、版面体检也看不见 —— 只能自己断言。"""
        assert y <= y0 + ph - 6, "%s 的内容到 %d，面板底边在 %d" % (who, y, y0 + ph)

    K, CTX = 2048, 131072
    frac = K * 100.0 / CTX
    warm = 1000 * 16 * CTX / 1e9              # 热身：1000 步 × 16 条 × 128K
    spars = 15000 * 480 * CTX / 1e9           # 稀疏期：15000 步 × 480 条 × 128K
    assert abs(frac - 1.5625) < 1e-6
    assert abs(warm - 2.1) < .05              # 论文写 2.1B —— 乘出来对得上
    assert abs(spars - 943.7) < .1            # 论文写 943.7B —— 也对得上

    f = Fig(W, "DSA 凭什么只看 2048 个就够：先是「注意力本来就 95% 稀疏」这个"
               "经验事实，再是「怎么知道该看谁」这个鸡生蛋问题，"
               "DSA 用让真注意力当老师的办法解决它")
    f.marks = set()
    y0 = f.header(
        "凭什么「只看 2048 个」就够　——　三个问题，别混在一起",
        "⭐ 「能不能少看」是经验事实；"
        "<tspan font-weight=\"700\">「怎么知道该看谁」才是 DSA 真正解决的那个</tspan>",
        [(GR, "先于 DSA 的经验事实"), (RD, "鸡生蛋的难点"),
         (BL, "让真注意力当老师"), (PU, "设计决策"), (OR, "硬件约束")])

    ph = 412

    # ══ ① 能不能只看一小部分 ════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 能不能只看一小部分", GR,
                 sub="经验事实，先于 DSA")

    yy = py + 26
    f.t(x + 24, yy, "H2O（2023）量了一批密集训出来的模型：", GY, size=12,
        w=pw - 48)
    yy += 24
    f.box(x + 24, yy, pw - 48, 78, "#fff", GR, 8)
    f.box(x + 24, yy, 4, 78, GR, GR, 2)
    f.box(x + 26, yy, 3, 78, "#fff", "#fff", 0)
    f.t(x + 42, yy + 25, "推理时，注意力矩阵 <tspan font-weight=\"700\">95% 以上是稀疏的</tspan>", GR,
        True, 12.5)
    f.t(x + 42, yy + 47, "累计注意力分数服从<tspan font-weight=\"700\">幂律分布</tspan>", GY, size=11.5)
    f.t(x + 42, yy + 67, "→ 论文结论：5% 的 KV 就够解出同一个 token", GY2,
        size=11)
    yy += 92

    # 幂律小图：少数几根高柱 ＋ 一条贴地长尾
    f.t(x + 24, yy, "幂律长什么样", GY, True, 12)
    yy += 10
    bx, bw, bh = x + 24, pw - 48, 68
    f.box(bx, yy, bw, bh, "#fff", LINE, 6)
    import math
    n = 46
    for i in range(n):
        v = 1.0 / (1 + i) ** 0.85
        h = max(1.4, v * (bh - 16))
        col = GR if i < 3 else (LINE2 if i > 8 else "#a8dab5")
        f.box(bx + 8 + i * (bw - 16) / float(n), yy + bh - 8 - h,
              (bw - 16) / float(n) - 1.2, h, col, "none", 1)
    f.t(bx + 8, yy + bh + 16, "前几个吃掉大半", GR, size=11)
    f.t(bx + bw - 8, yy + bh + 16, "长尾每个都接近 0", GY2, size=11,
        anchor="end")
    yy += bh + 30

    f.box(x + 24, yy, pw - 48, 48, "#fff", LINE, 8)
    f.t(x + 40, yy + 21, "DSA 取 k = 2,048；128K 上下文下 ——", GY, size=11.5)
    f.t(x + 40, yy + 39, "占全部历史的 %.2f%%" % frac, GR, True, 12.5)

    fits(yy + 48, y0, ph, "①")

    # ══ ② 那怎么知道该看谁 ══════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 那怎么知道该看谁", RD,
                 sub="这才是难点：一个鸡生蛋")

    yy = py + 26
    f.box(x + 24, yy, pw - 48, 62, "#fff", RD, 8)
    f.box(x + 24, yy, 4, 62, RD, RD, 2)
    f.box(x + 26, yy, 3, 62, "#fff", "#fff", 0)
    f.t(x + 42, yy + 24, "要判断谁重要，得先算出注意力分数 ——", RD, True, 12.5)
    f.t(x + 42, yy + 46, "而那<tspan font-weight=\"700\">正是你想省掉的东西</tspan>。",
        GY, size=11.5)
    yy += 78

    f.t(x + 24, yy, "⭐⭐ DSA 的解法：不猜，让真注意力当老师",
        BL, True, 13.5, cls="svglbl")
    yy += 26

    STEP = [
        ("热身阶段", "<tspan font-weight=\"700\">保持密集</tspan>注意力，冻住全模型，只训索引器"),
        ("造一个目标", "主注意力分数<tspan font-weight=\"700\">跨所有头求和</tspan>，再 L1 归一 → p"),
        ("损失", "KL( p ‖ softmax(索引分数) ) —— 就这一项"),
        ("代价", "1,000 步 · %.1fB token，<tspan font-weight=\"700\">然后就可以开稀疏了</tspan>" % warm),
    ]
    for i, (lab, txt) in enumerate(STEP):
        f.box(x + 24, yy, pw - 48, 46, "#fff", BL if i < 3 else LINE, 8)
        f.t(x + 40, yy + 20, lab, BL if i < 3 else GY, True, 12)
        f.t(x + 40, yy + 38, txt, GY, size=11.5)
        yy += 54

    yy += 2
    f.t(x + 24, yy, "⭐ 判据：不是设计启发式去猜谁重要，", INK, True, 12.5,
        w=pw - 48)
    f.t(x + 24, yy + 21, "是让那个贵的东西自己说出答案，再训个便宜的去复制。",
        INK, True, 12.5, w=pw - 48)

    fits(yy + 21, y0, ph, "②")

    # ══ ③ 这个复制品凭什么算得动 ════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 这个复制品凭什么算得动", PU,
                 sub="三个便宜 ＋ 两个决策 ＋ 一条硬约束")

    yy = py + 24
    bx = x + 24
    for lab, sub in [("头很少", "不是 128 个"), ("ReLU", "不用 softmax"),
                     ("FP8", "半个字节的事")]:
        f.cell(bx, yy, 132, 46, lab, sub, GR)
        bx += 138
    yy += 58
    f.t(x + 24, yy, "⭐ ReLU 是<tspan font-weight=\"700\">为吞吐</tspan>选的，论文自己这么说的", GY2,
        size=11, w=pw - 48)
    yy += 22

    for lab, txt in [
        ("决策一　梯度断开",
         "索引器只由 KL 训，主模型只由语言建模 loss 训"),
        ("决策二　稀疏期只在选中集合上算 KL",
         "已经稀疏之后，它只需要在自己选出的那批里排对序"),
    ]:
        f.box(x + 24, yy, pw - 48, 50, "#fff", PU, 8)
        f.t(x + 40, yy + 21, lab, PU, True, 12)
        f.t(x + 40, yy + 39, txt, GY, size=11.5)
        yy += 58

    yy += 4
    f.box(x + 24, yy, pw - 48, 90, "#fff", OR, 8)
    f.box(x + 24, yy, 4, 90, OR, OR, 2)
    f.box(x + 26, yy, 3, 90, "#fff", "#fff", 0)
    f.t(x + 42, yy + 24, "硬约束　它必须搭 MQA 模式的 MLA", OR, True, 12.5)
    f.t(x + 42, yy + 46, "kernel 上，一条 KV 必须被多个 query 共享才划算",
        GY, size=11.5)
    f.t(x + 42, yy + 68,
        "⭐ 旋钮①和旋钮②，在这里被<tspan font-weight=\"700\">硬件</tspan>绑在了一起", GY, size=11.5)

    fits(yy + 90, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "warn", "那 2048 这个数到底怎么来的 —— 老实说：论文没给消融", [
        ("能说的只有三件事：① 128K 下它占 <tspan font-weight=\"700\">{:.2f}%</tspan>；"
         "② H2O 量到的稀疏度是 95% 以上，"
         "<tspan font-weight=\"700\">1.6% 比那还狠 ——&#160;它靠的是「挑得准」不是「挑得多」</tspan>；"
         ).format(frac),
        "③ <tspan font-weight=\"700\">历史不足 2048 时，top-2048 就是全选</tspan>"
        "——&#160;这是定义直接推出来的，所以短上下文下 DSA 就是普通 MLA，"
        "<tspan font-weight=\"700\">稀疏只在长上下文才启动</tspan>。",
        "⚠️ 别把「95% 稀疏」当普适常数 ——&#160;那是 H2O 在它测的那批模型上量的。",
    ])

    yy = f.band(yy + 14, "ok", "暗线第三次出现：事后压 vs 从头按压缩训", [
        "H2O 是<tspan font-weight=\"700\">事后</tspan>的 ——&#160;"
        "一个 KV 驱逐策略，不改训练，随时可开关。",
        "DSA 是<tspan font-weight=\"700\">从头</tspan>的 ——&#160;"
        "热身 %.1fB token ＋ 稀疏期 <tspan font-weight=\"700\">%.1fB token</tspan> "
        "的继续训练，模型是在「我会被稀疏」这个前提下学出来的。"
        % (warm, spars),
        "⭐ 三次了：Eigen Attention 对 MLA、GQA 对 MLA、H2O 对 DSA。"
        "<tspan font-weight=\"700\">同一个对立，三个不同的技术分支。</tspan>",
    ])

    yy = f.src(yy + 16,
               "① 出自 H2O：Zhang 等 arXiv 2306.14048（「over 95% sparse」"
               "与累计注意力分数的幂律分布，均为原文表述）",
               "②③ 出自 DeepSeek-V3.2-Exp 技术报告 §1–§2.1："
               "索引式 I(t,s)=Σ w·ReLU(q·k)、KL 对齐、梯度 detach、k=2048、MQA 模式",
               "两个 token 数由「步数 × 每步序列数 × 128K」当场乘出并断言，"
               "与报告写的 2.1B / 943.7B 一致")
    f.save("fig3-dsa-why.svg", yy + 6)


main()
