# -*- coding: utf-8 -*-
r"""专题三 · §五「RoPE：把位置变成一个转角」

⭐⭐⭐ 2026-09-13 新画。本课原来对 RoPE 只有一个「寄快递」的比喻，
   而那个比喻回答的是「为什么 MLA 吸收不了它」——&nbsp;
   ⛔ **「把位置变成旋转角，为什么点积就自动带上了相对距离」
     这个最基础的问题，我们一张图都没有。**

📌 装置来自两篇顶级材料**各一半，而且没人把它们拼起来**：
   · Fleetwood（huggingface.co/blog/designing-positional-encoding）
     画了**二进制计数器动画**（低位飞快翻、高位几乎不动），
     用来说明「正弦是二进制计数器的连续版」——&nbsp;停在这儿了。
   · 苏剑林（kexue.fm/archives/9675）走到了
     **「RoPE ＝ 位置的 β 进制写法」**，于是外推 / 内插 / NTK 一句话各自归位
     ——&nbsp;⛔ 但他全文是公式，**一张图没画**。
   ⭐ 这张图把两半拼起来：**一排里程表转盘。**

📐 那条等式本课自己推了一遍，不靠转述（脚本里断言）：
   RoPE 第 m 对维度的角速度 θ_m ＝ 10000^(-2(m-1)/d) ＝ 1/β^(m-1)，
   其中 β ＝ 10000^(2/d)。于是 n·θ_m ＝ n / β^(m-1) ——&nbsp;
   **跟 β 进制取第 m 位时的除数完全一样。** d=128 → 64 对维度 ＝ 64 位数。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    d = 128
    beta = 10000 ** (2.0 / d)
    for m in range(1, 8):                      # ⛔ 当场验，别信转述
        assert abs(10000 ** (-2.0 * (m - 1) / d) - 1.0 / beta ** (m - 1)) < 1e-12
    assert d // 2 == 64

    f = Fig(W, "RoPE 就是把位置写成一排里程表转盘：最右边转得飞快，"
               "往左逐级变慢；两个位置各自转完之后，每个转盘上的夹角只跟"
               "它们的距离有关，跟各自在哪完全无关")
    f.marks = set()
    y0 = f.header(
        "RoPE ——　<tspan font-weight=\"700\">把位置写成一排里程表转盘</tspan>",
        "⭐ 这张图回答三件事：<tspan font-weight=\"700\">它到底负责什么</tspan>、"
        "<tspan font-weight=\"700\">为什么点积自动带上了相对距离</tspan>、"
        "以及<tspan font-weight=\"700\">凭什么能外推到 1M</tspan>",
        [(RD, "掩码管顺序"), (BL, "位置 m"), (OR, "位置 n"),
         (PU, "夹角＝m−n"), (GR, "三种改法")])

    # ══════════ ① 先摆正职责：顺序不归 RoPE 管 ══════════════════
    # ⛔⛔⛔ 2026-09-14 R59 新增。现场纠正的一条，**几乎所有科普都讲错**：
    #   拿「你打我 / 我打你」来论证「所以必须有位置编码」——&nbsp;讲反了。
    #   那个例子成立的前提是**没有掩码**（双向 encoder）。
    # ⭐ 推导链（不靠转述，逐步可验）：
    #   · 无掩码：o_i ＝ Σ_j softmax(q_i·k_j)·v_j ——&nbsp;**对 j 求和**。
    #     求和不认顺序，所以把输入重排，输出只是跟着重排 → 置换等变。
    #   · 有因果掩码：o_i 只对 j ≤ i 求和，于是它依赖的是**前缀**而不是全集。
    #     「你打我」的第 2 格看见 {你,打}；「我打你」的第 2 格看见 {我,打}
    #     ——&nbsp;**两个前缀不同，这一格的输出就已经不同了**；
    #     再上一层，句尾那一格读到的东西就带上了顺序。
    #   · 这正是 NoPE（arXiv 2305.19466, NeurIPS 2023）的结论：
    #     decoder-only 不加任何显式位置编码也学得会顺序。
    # ⭐⭐ 所以 RoPE 买的不是「顺序」，是**「差几格」直接进打分**。
    #   §5.3 里 MLA 为它单独留的那 64 维，成本对应的就是这一件事。
    PH0 = 300
    ay = f.panel(0, y0, W, PH0,
                 "① 先摆正职责 ——　<tspan font-weight=\"700\">"
                 "「谁在前谁在后」并不归 RoPE 管</tspan>", RD,
                 sub="这一格几乎所有讲法都拧了：那个经典例子证明的是"
                     "「没有掩码时分不出」，不是「所以必须有位置编码」")
    TK = ("#fff5f5", "#f1f8f4")

    def sent(x, y, chars, col, tint):
        for i, c in enumerate(chars):
            f.box(x + i * 66, y, 56, 42, tint, col, 8)
            f.t(x + i * 66 + 28, y + 28, c, col, True, 21, "middle")

    # —— 左：没有掩码（双向）→ 真的分不出 ——
    f.box(18, ay + 6, 664, 254, "#fff", RD, 10)
    f.t(42, ay + 40, "没有掩码时（双向 encoder）", RD, True, 20)
    sent(46, ay + 60, "你打我", RD, TK[0])
    sent(46, ay + 122, "我打你", RD, TK[0])
    f.box(330, ay + 90, 78, 46, "#fff", GY2, 8)
    f.t(369, ay + 119, "同一个 o", GY, True, 16, "middle")
    f.line(250, ay + 81, 328, ay + 104, GY2, 1.6)
    f.line(250, ay + 143, 328, ay + 122, GY2, 1.6)
    f.t(424, ay + 105, "⛔ 两句的输出", RD, True, 17)
    f.t(424, ay + 129, "一模一样", RD, True, 17)
    f.box(42, ay + 182, 616, 62, BG2, LINE2, 8)
    f.t(60, ay + 208, "因为 o ＝ <tspan font-weight=\"700\">对所有位置求和</tspan>"
        "，而求和不认顺序 ——", GY, size=16, w=580)
    f.t(60, ay + 232, "把输入重排，输出只是跟着重排。这才是那个例子真正说明的事。",
        GY, size=16, w=580)

    # —— 右：有因果掩码 → 顺序已经在里面了 ——
    f.box(718, ay + 6, 664, 254, "#fff", GR, 10)
    f.t(742, ay + 40, "有因果掩码时（今天的 decoder-only LLM）", GR, True, 20)
    sent(746, ay + 60, "你打我", GR, TK[1])
    sent(746, ay + 122, "我打你", GR, TK[1])
    for yy_ in (ay + 54, ay + 116):            # 框住「第 2 格看得见的全部」
        f.box(740, yy_, 134, 54, "none", GR, 9, sw=2.2)
    # ⛔ x 不能小于 950：三个字块从 746 起、每格 66，最后一块右沿在 934。
    f.t(956, ay + 84, "第 2 格看得见 ＝ {你, 打}", GY, size=16)
    f.t(956, ay + 146, "第 2 格看得见 ＝ {我, 打}", GY, size=16)
    f.t(956, ay + 115, "⭐ 已经不一样了", GR, True, 17)
    f.box(742, ay + 182, 616, 62, BG2, LINE2, 8)
    f.t(760, ay + 208, "每个位置只对<tspan font-weight=\"700\">自己的前缀</tspan>"
        "求和，两句的前缀不同 → 中间那一格就已经分开；", GY, size=16, w=580)
    f.t(760, ay + 232, "再上一层，顺序就传到了句尾。<tspan font-weight=\"700\">"
        "NoPE（arXiv 2305.19466）证明的就是这件事。</tspan>", GY, size=16, w=580)
    y0 = y0 + PH0 + 18

    def dial(cx, cy, r, ang, col, tint="#fff", sw=2.0):
        f.p.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" '
                   'stroke="%s" stroke-width="1.6"/>' % (cx, cy, r, tint, GY2))
        f.line(cx, cy, cx + r * 0.78 * math.sin(ang),
               cy - r * 0.78 * math.cos(ang), col, sw, arrow=False)

    # ══════════ ② 一排转盘 ══════════════════════════════════════
    PH = 452
    py = f.panel(0, y0, W, PH, "② 一个位置 ＝ 一排转盘的读数", INK,
                 sub="最右边转得飞快，往左逐级变慢 ——　跟里程表一模一样")
    ay = py + 34
    NDIAL = 7
    R = 34
    for k in range(NDIAL):
        cx = 130 + k * 170
        speed = 1.0 / beta ** (k * 9)          # 每隔 9 对取一个，好看出快慢
        dial(cx, ay + 54, R, 3 * speed, BL)
        f.t(cx, ay + 116, "第 %d 对" % (k * 9 + 1), GY2, size=15, anchor="middle")
        f.t(cx, ay + 140, "转速 1/β^%d" % (k * 9), GY2, size=14, anchor="middle")
    f.t(130 - R - 8, ay + 172, "← 快（每个位置都转一大格）", GY, size=17)
    f.t(130 + 6 * 170 + R + 8, ay + 172, "慢（几万个位置才转一圈）→", GY,
        size=17, anchor="end")
    f.box(56, ay + 192, 1288, 80, "#e8f0fe", BL, 10)
    f.t(80, ay + 222, "⭐ 这排转盘就是 <tspan font-weight=\"700\">位置 n 的 "
        "β 进制写法</tspan>（β ＝ 10000^(2/d) ≈ %.3f）" % beta, GY, size=17,
        w=1240)
    f.t(80, ay + 250, "d=128 就是 <tspan font-weight=\"700\">64 位数</tspan>；"
        "第 m 位的除数 β^(m-1)，<tspan font-weight=\"700\">正是第 m 对维度的"
        "转速</tspan>。", GY, size=17, w=1240)

    # ⭐⭐ 2026-09-14 R59 新增：快盘 / 慢盘各自被拿去干什么。
    #   ⛔ 顺手先拆掉一个流传最广的说法：「RoPE 有用是因为越远打分越低」。
    #     arXiv 2410.06205（Barbero et al.）开篇就是冲这句去的 ——&nbsp;
    #     原文 "A common belief is that RoPE is useful because it helps to
    #     decay token dependency as relative distance increases. …
    #     we argue that this is unlikely to be the core reason."
    #   ⭐ 它扒的是训练好的 **Gemma 7B** 的内部：
    #     · 最高频（转得最快那几个盘）→ 被拿去构造稳健的「位置型」注意力模式
    #     · 最低频（转得最慢那几个盘）→ 用得最多，**论文推测**是在携带语义
    #     ⚠️ 「携带语义」原文写的是 "we suspect"，这里必须照样写成推测。
    #   📌 为什么值得画进这一格：这张图本来就把 64 对维度画成了快慢不同的转盘，
    #     这条发现等于给「为什么要有快慢之分」补上了实测层面的答案 ——&nbsp;
    #     快盘当尺子、慢盘当载货位，而不是「一起制造一个衰减」。
    f.box(56, ay + 284, 1288, 104, "#fff", GY2, 10)
    f.t(80, ay + 314, "⛔ 顺带拆一个流传最广的说法：<tspan font-weight=\"700\">"
        "「RoPE 有用是因为越远打分越低」——　这不是它起作用的原因</tspan>"
        "（arXiv 2410.06205 专门论证了这点：RoPE 并不单调衰减）。", GY,
        size=17, w=1240)
    f.t(80, ay + 344, "⭐⭐ 那篇论文扒开训练好的 Gemma 7B 看到的是<tspan "
        "font-weight=\"700\">分工</tspan>：<tspan font-weight=\"700\">最快的那几个盘"
        "</tspan>被拿去搭稳定的位置型注意力（「盯住我前面第几个」），", GY,
        size=17, w=1240)
    f.t(80, ay + 372, "<tspan font-weight=\"700\">最慢的那几个盘</tspan>用得最多 ——　"
        "它们几乎不随位置变，<tspan font-weight=\"700\">论文推测</tspan>是被模型"
        "腾出来携带语义。⭐ 快盘当尺子，慢盘当载货位。", GY, size=17, w=1240)

    # ══════════ ③ 为什么点积只认距离 ════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 322
    py2 = f.panel(0, y1, W, PH2,
                  "③ 为什么点积自动带上了相对距离 ——　看夹角就行", PU,
                  sub="两排转盘叠起来，每个盘上的夹角都是 (m−n)×转速")
    by = py2 + 40
    for k in range(5):
        cx = 180 + k * 210
        sp = 1.0 / beta ** (k * 12)
        am, an = 12 * sp, 5 * sp
        f.p.append('<circle cx="%.1f" cy="%.1f" r="48" fill="#fff" '
                   'stroke="%s" stroke-width="1.6"/>' % (cx, by + 56, GY2))
        f.line(cx, by + 56, cx + 38 * math.sin(am), by + 56 - 38 * math.cos(am),
               BL, 2.6, arrow=False)
        f.line(cx, by + 56, cx + 38 * math.sin(an), by + 56 - 38 * math.cos(an),
               OR, 2.6, arrow=False)
        f.t(cx, by + 126, "夹角 ＝ (m−n)×转速", PU, True, 15, "middle")
    f.t(180 - 58, by - 10, "蓝＝位置 m 转过的角　·　橙＝位置 n 转过的角", GY,
        size=17)
    f.box(56, by + 146, 1288, 96, "#f3e8fd", PU, 10)
    f.t(80, by + 178, "⭐⭐ 每个盘上你只看得出<tspan font-weight=\"700\">"
        "两根针差多少</tspan> ——　看不出各自转到了哪儿。", PU, True, 17)
    f.t(80, by + 208, "而点积 <tspan font-weight=\"700\">a·b ＝ |a||b|cos θ</tspan>"
        " 只吃夹角和长度：同转一个角，两样都没变。", GY, size=17, w=1240)
    f.t(80, by + 234, "⭐ 所以绝对位置被转掉了，<tspan font-weight=\"700\">"
        "留下来的只有 m−n</tspan>。", GY, size=17, w=1240)

    # ══════════ ④ 长文本三种改法 ════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 340
    py3 = f.panel(0, y2, W, PH3,
                  "④ 于是长文本那三种做法，一句话各自归位", GR,
                  sub="同一排转盘，三种改法")
    ey = py3 + 30
    WAYS = [
        (RD, "直接外推", "转盘一格不改", "硬往超出刻度的地方读",
         "⛔ 最慢那个盘从没转到过那儿，模型没见过", "none"),
        (OR, "位置内插（PI）", "每格改成走半格", "整排一起放慢",
         "⛔ 最快那个盘现在分不清相邻两个位置了", "slow"),
        (GR, "NTK-aware", "换一个进制", "β 变大：快盘几乎不动，慢盘明显变慢",
         "⭐ 一句话：高频外推、低频内插", "base"),
    ]
    for i, (col, name, a, b_, note, kind) in enumerate(WAYS):
        x = 30 + i * 452
        f.box(x, ey, 428, 268, "#fff", col, 10)
        f.box(x, ey, 428, 5, col, col, 3)
        f.box(x, ey + 3, 428, 5, "#fff", "#fff", 0)
        f.t(x + 22, ey + 42, name, col, True, 23)
        f.t(x + 22, ey + 70, a, GY, size=17)
        for k in range(4):
            cx = x + 70 + k * 90
            base = 1.0 / beta ** (k * 16)
            sp = base if kind == "none" else (base * 0.5 if kind == "slow"
                                              else base ** 1.25)
            dial(cx, ey + 130, 28, 9 * sp, col)
        f.t(x + 22, ey + 184, b_, col, True, 17, w=384)
        f.box(x + 22, ey + 200, 384, 52, BG2, LINE2, 8)
        f.t(x + 36, ey + 230, note, GY, size=16, w=356)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这张图跟「寄快递」不冲突，是同一件事的两个切面", [
        "<tspan font-weight=\"700\">寄快递</tspan>回答的是"
        "「<tspan font-weight=\"700\">为什么 MLA 吸收不了它</tspan>」——&#160;"
        "那个旋转矩阵夹在上投影和隐向量中间，拆不开。",
        "<tspan font-weight=\"700\">里程表</tspan>回答的是"
        "「<tspan font-weight=\"700\">为什么点积自动带相对距离</tspan>」和"
        "「<tspan font-weight=\"700\">凭什么能外推</tspan>」。",
        "⭐ 两个一起看，<a href=\"#s五\">§五</a>那条 decoupled RoPE "
        "（让带位置的那几维单独走一路）就不是一个补丁，而是唯一的出路。",
    ])
    yy = f.src(yy + 16,
               "①「顺序归因果掩码，不归位置编码」：NoPE，arXiv 2305.19466"
               "（NeurIPS 2023）——&#160;证明 decoder-only 不加任何显式位置编码"
               "也能学会顺序。<tspan font-weight=\"700\">⛔ 常见讲法拿"
               "「你打我／我打你」论证「所以必须有位置编码」，是把前提"
               "（没有掩码的双向模型）漏掉了。</tspan>",
               "②「RoPE 不是靠越远越衰减起作用」＋「快盘做位置、慢盘携带语义」："
               "Barbero et al.，arXiv 2410.06205 ——&#160;对训练好的 Gemma 7B "
               "做的内部分析；⚠️ 其中「携带语义」原文是 we suspect，"
               "<tspan font-weight=\"700\">这里照样只当推测</tspan>。",
               "「RoPE ＝ 位置的 β 进制写法」以及外推／内插／NTK 的统一解释，"
               "出自苏剑林 kexue.fm/archives/9675（⚠️ 原文是推导，没有图）",
               "「二进制计数器 → 正弦」的动画装置出自 Fleetwood "
               "huggingface.co/blog/designing-positional-encoding；"
               "「点积只吃夹角和长度」出自 EleutherAI 的 RoPE 博客",
               "📐 θ_m ＝ 10000^(−2(m−1)/d) ＝ 1/β^(m−1) 这条等式"
               "<tspan font-weight=\"700\">由本脚本当场验证并断言</tspan>，不是转述；"
               "RoPE 原始出处 RoFormer arXiv 2104.09864")
    f.save("fig3-rope.svg", yy + 6)


main()
