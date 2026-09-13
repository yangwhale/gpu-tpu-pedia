# -*- coding: utf-8 -*-
r"""专题三 · §3.2b ＋ §3.2c「内外循环对调」

⭐⭐⭐ 2026-09-14 夜间 R33 新画。补的缺口很直白：
   §3.2b 的标题叫「⭐ <b>那张经典图</b>：内循环、外循环 ——&#160;
   以及 FA2 为什么把它掉了个个儿」，
   ⛔ 而这一节**一张图都没有** ——&#160;只有一段散文和两块伪代码。
   一个讲「那张图」的小节自己不给图，是全讲最刺眼的一处。

═══ 这张图跟别人画的不一样在哪 ═══
⚠️ 满网的 FA 图（含 FA1 Figure 1 本身）画的都是**遍历顺序**：
   谁在外圈转、块怎么被扫过。
⭐ 本图画的是**后果**：换一个顺序，四个张量各自要被搬多少趟。
⭐⭐ 装置是「**两个标记在两版之间互换位置**」——
   左上蓝点＝这一格要载一次 K/V；右下橙点＝这一格要搬一次 O。
   FA1：蓝点每列一个、橙点每格都有；FA2：正好反过来。
   ⛔ 初版把橙点画在网格下方一排，36 个点重叠成 8 个，
     看上去跟 FA2 的 8 个一模一样 ——&#160;等于用图说了句假话。改成画进格子里。

═══ ⛔ 本图最重要的一个数 ═══
课程正文（和绝大多数讲解）强调的是：
  「O_i 和那两个统计量在整个内层循环里一直待在片上，一次都不落 HBM。」
这句话对，但**只数 O 会给人一个错觉**：
  ▸ 只数 O：FA1 要 4160 次块读写，FA2 只要 64 次 ——&#160;看着是 65 倍。
  ▸ 四个张量一起数：FA1 6368，FA2 4288 ——&#160;**只有 1.49 倍**。
⭐⭐ 差额去哪了？**FA2 把省下的 O 流量，又用 K/V 的重读还了回去** ——
   K/V 在 FA1 里待在外圈、只读一趟；到 FA2 就跑进内圈，
   每个 Q 块都要把全部 K/V 重新走一遍。
⛔ 所以「换循环顺序」换来的**不主要是 HBM 流量**。

⭐⭐⭐ 而这个判断**跟论文自己的说法是一致的**（这一点是核过才敢写的）：
   FA2 全文**没有一句** prose 说「对调是为了让 O 不落 HBM」——
   它给的两条理由是 ① 减少 non-matmul FLOP（§3.1）
   ② occupancy / thread block 之间无需通信（§3.2）。
   ⛔ 而且 FA2 **没有 IO-complexity 定理**，整个 IO 分析 defer 给了 FA1。
   所以本图那个 1.49 倍**必须标成「本课自己按两份 Algorithm 1 数出来的」**，
   不能写成论文报的数。图上和图注都标了。

⚠️ 口径写死（图上也写了）：
  - 只数 Q / K / V / O 四个 [L, d] 张量的**块级** HBM 读写；
    m、ℓ、L 是 B 维小向量，忽略。
  - L = 8192、块 128 ×&#160;128（块长照 §3.4 的 GPU 侧口径），故 Tr = Tc = 64。
  - **默认按 causal 数**（上三角整块跳过），与本讲 §3.6 ① 的记账纪律一致；
    脚本同时算了非 causal 版，两者比值 1.485 / 1.492，**结论不依赖这个选择**。
  - ⚠️ 这是**按 HBM 读写次数**数的理想账：真机上 L2 会吃掉一部分 K/V 重读，
    所以 1.49 倍是**偏乐观那一侧**，不是实测。图上标明了。
⛔ 图里每个数都是本脚本当场数的（两层循环真的跑了一遍），不是抄来的。

═══ 逐字引文（全部核自 arXiv e-print 源码）═══
- FA1 Algorithm 1（arXiv 2205.14135 §3.1）外层 K/V、内层 Q，逐字：
    "for 1 ≤ j ≤ T_c do / Load K_j, V_j from HBM to on-chip SRAM. /
     for 1 ≤ i ≤ T_r do / Load Q_i, O_i, ℓ_i, m_i from HBM to on-chip SRAM."
  而 O_i、ℓ_i、m_i 的写回在**内层循环里**（line 12–13）：
    "Write O_i ← ... to HBM." / "Write ℓ_i ← ℓ_i^new, m_i ← m_i^new to HBM."
- FA1 Figure 1 caption（逐字，本图的「遍历顺序」对照对象）：
    "In the outer loop (red arrows), FlashAttention loops through blocks of
     the K and V matrices and loads them to fast on-chip SRAM. In each block,
     FlashAttention loops over blocks of Q matrix (blue arrows)..."
- FA2 Algorithm 1（arXiv 2307.08691 §3.1.1）外层 Q、内层 K/V，
  O_i 与 L_i 在内层结束后各写一次（line 14–15）。
- ⛔ 宣布「换过来」那句话**不在 §3.1，在 §3.2 "Parallelism"**，逐字：
    "These ideas of swapping the order of the loop (outer loop over row blocks
     and inner loop over column blocks, instead of the other way round in the
     original FlashAttention paper), as well as parallelizing over the sequence
     length dimension were first suggested and implemented by Phil Tillet in
     the Triton implementation."
  ⭐ 归功对象是 **Phil Tillet 的 Triton 实现**，不是 FA1 的 backward。
  ⛔ 事实上 FA1 的 backward（Algorithm 4）**也是** K/V 外层，
    FA2 的 backward（Algorithm 2）**依然**是 K/V 外层 ——&#160;**只对调了 forward**。
- FA2 并行那条理由逐字（§3.2）：
    "We see that the outer loop (over sequence length) is embarrassingly
     parallel, and we schedule them on different thread blocks that do not need
     to communicate with each other. ... The increased parallelism over sequence
     length helps improve occupancy ... when the batch size and number of heads
     are small."
- warp 层（全部出自 FA2 §3.3，⛔ **FA1 论文里 "warp" 出现 0 次**）：
    "For each block, FlashAttention splits K and V across 4 warps while keeping
     Q accessible by all warps. ... This is referred to as the “split-K” scheme."
    "However, this is inefficient since all warps need to write their
     intermediate results out to shared memory, synchronize, then add up the
     intermediate results. These shared memory reads/writes slow down the
     forward pass in FlashAttention."
    "In FlashAttention-2, we instead split Q across 4 warps while keeping K and
     V accessible by all warps. ... There is no need for communication between
     warps."
  ⛔⛔ 原词是 **"split-K"**（论文里带引号带连字符）。
    「sliced-K」是二手叫法，两篇论文全文**零次**。课程原来写错了，本轮一并改。
- 数字（⚠️ 三个口径别串，各自带硬件与场景）：
    "FlashAttention-2 is 1.7-3.0× faster than FlashAttention ... reaches up to
     230 TFLOPs/s, 73% of the theoretical maximum TFLOPs/s on A100 GPUs."
     （attention microbenchmark，A100 80GB SXM4）
    Table 1 caption: "FlashAttention-2 reaches up to 225 TFLOPs/s
     (72% model FLOPs utilization)."（端到端训练，8×A100 80GB SXM）
  ⛔ 73% **不是 MFU**；唯一标 MFU 的是 72%。
  ⚠️ 2026-09-14：这两个数原本写进了图的出处栏，被 lint 抓到 ——
    「MFU」这个词的定义在本讲 §3.6，而本图在 §3.2b，**比定义早**。
    图上那行已改成不提这个词；口径纪律指回 §3.6b。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2)

W_ = 1400

SEQ = 8192
BLK = 128
TR = TC = SEQ // BLK          # 64
assert TR == 64 and TC == 64


def count(tr, tc, causal):
    """真的把两层循环跑一遍，数块级 HBM 读写次数。"""
    pairs = 0
    for i in range(tr):
        for j in range(tc):
            if causal and j > i:      # causal：上三角整块跳过
                continue
            pairs += 1
    # FA1：外层 j 走 K/V，内层 i 走 Q。
    #   每个 (i, j) 都要 load Q_i、load O_i、store O_i；K/V 每块只 load 一次。
    fa1 = {"Q": pairs, "KV": tc * 2, "O": pairs * 2}
    # FA2：外层 i 走 Q，内层 j 走 K/V。
    #   每个 (i, j) 都要 load K_j、load V_j；Q 每块 load 一次，O 每块 store 一次。
    fa2 = {"Q": tr, "KV": pairs * 2, "O": tr}
    return fa1, fa2, pairs


C1, C2, PAIRS = count(TR, TC, causal=True)
T1, T2 = sum(C1.values()), sum(C2.values())
assert PAIRS == TR * (TR + 1) // 2 == 2080
assert (T1, T2) == (6368, 4288), (T1, T2)
RATIO = T1 / float(T2)
assert 1.48 < RATIO < 1.49

# 只数 O 会看到什么 —— 这正是要破的那个错觉
O_ONLY = C1["O"] / float(C2["O"])
assert C1["O"] == 4160 and C2["O"] == 64 and abs(O_ONLY - 65.0) < 1e-9

# 非 causal 版：证明结论不挑口径
_N1, _N2, _ = count(TR, TC, causal=False)
_R_NC = sum(_N1.values()) / float(sum(_N2.values()))
assert abs(_R_NC - 1.492) < 0.001, _R_NC
assert abs(_R_NC - RATIO) < 0.01, "换口径结论就变的话，这张图不能画"

# 并行度：FA1 只能按 batch×head 铺；FA2 多出 Q 块这一维
PAR_EXTRA = TR
assert PAR_EXTRA == 64

G = 32          # 网格格边长
NB = 8          # 图上示意用 8×8，不是真实的 64×64（图上标明）
DOT = 7


def main():
    f = Fig(W_, "FlashAttention 把内外循环对调之后：同一张块网格，两种走法；"
               "四个张量各自要被搬多少趟；以及同一条判据在 warp 这一层又用了一遍")
    f.marks = set()
    y0 = f.header(
        "内外循环对调：省下的到底是什么",
        "满网的图画的都是<tspan font-weight=\"700\">遍历顺序</tspan>；"
        "这张画的是<tspan font-weight=\"700\">后果</tspan> ——&#160;"
        "把四个张量分开称一次重",
        [(BL, "载一次 K/V"), (OR, "搬一次 O"), (GR, "FA2")])

    # ══════════ ① 同一张网格，两种走法 ═════════════════════════════
    PH1 = 482
    top = f.panel(0, y0, W_, PH1, "① 同一张块网格，两种走法", PU,
                  sub="行 ＝ Q 的块，列 ＝ K/V 的块；浅灰 ＝ causal 整块跳过")

    def grid(x, y, outer_q, col, title, sub):
        f.t(x + NB * G / 2.0, y - 44, title, col, True, 18, "middle")
        f.t(x + NB * G / 2.0, y - 22, sub, GY, False, 14, "middle")
        f.t(x + NB * G / 2.0, y - 4, "K / V 的块 →", GY2, False, 13, "middle")
        f.t(x - 12, y + NB * G / 2.0, "Q 的块", BL, True, 14, "end")
        f.t(x - 12, y + NB * G / 2.0 + 20, "↓", BL, True, 14, "end")
        for i in range(NB):
            for j in range(NB):
                cx, cy = x + j * G, y + i * G
                if j > i:
                    f.box(cx, cy, G - 2, G - 2, "#f4f6f8", "#fff", 2, 1)
                    continue
                f.box(cx, cy, G - 2, G - 2, "#fbfcfe", "#dde4ec", 2, 1)
                # 左上蓝点：这一格要从 HBM 载一次 K/V
                #   FA1 的 K/V 在外圈 → 只有每列第一次进入时载（对角那一格）
                if outer_q or i == j:
                    f.box(cx + 4, cy + 4, DOT, DOT, BL, "none", 2)
                # 右下橙点：这一格要搬一次 O
                #   FA2 的 O 留在片上 → 只有走到这一行最后一格才写出去
                if (not outer_q) or j == i:
                    f.box(cx + G - 2 - DOT - 4, cy + G - 2 - DOT - 4,
                          DOT, DOT, OR, "none", 2)
        # 遍历轨迹：FA1 沿列走，FA2 沿行走
        for k in range(NB):
            if outer_q:
                f.line(x + 3, y + k * G + G / 2 - 1,
                       x + k * G + G - 5, y + k * G + G / 2 - 1, "#9ac7a8", 1.2)
            else:
                # ⚠️ 轨迹线要让位给格子里那两个点 —— 压细、走浅色，
                #   否则一整列的橙点被紫线穿成一条，数不清。
                f.line(x + k * G + G / 2 - 1, y + k * G + 3,
                       x + k * G + G / 2 - 1, y + NB * G - 5, "#c9b6d8", 1.2)
        return y + NB * G

    GX1, GX2 = 150, 830
    GY_ = top + 96
    bot = grid(GX1, GY_, False, PU, "FA1：外层 K/V，内层 Q",
               "一列一列地走")
    grid(GX2, GY_, True, GR, "FA2：外层 Q，内层 K/V", "一行一行地走")
    f.t(700, GY_ + NB * G / 2.0, "vs", GY2, True, 24, "middle")

    for gx, s1, s2 in ((GX1, "K/V 每列只载一次（%d 次）" % TC,
                        "O 每一格都要读回来再写回去"),
                       (GX2, "K/V 每一格都要重新载一次",
                        "O 每行只写一次（%d 次）" % TR)):
        f.box(gx + 4, bot + 12, DOT, DOT, BL, "none", 2)
        f.t(gx + 20, bot + 20, s1, GY, False, 14)
        f.box(gx + 4, bot + 34, DOT, DOT, OR, "none", 2)
        f.t(gx + 20, bot + 42, s2, GY, False, 14)

    f.lines(60, bot + 72, 1280, [
        "⭐ <tspan font-weight=\"700\">两个点在两版之间正好换了位置</tspan> ——&#160;"
        "这就是这一格要说的全部。FA1 躲不掉那些橙点，是因为它外层走 K/V："
        "<tspan font-weight=\"700\">下一列还会碰到同一个 Q 块</tspan>，",
        "这块的输出没算完，只能先存回去。FA2 外层走 Q："
        "<tspan font-weight=\"700\">一整行走完，这块输出就彻底定稿了</tspan>。"
        "⚠️ 图上画 8×8 是示意，真实是 64×64（L ＝ 8192、块 128）。",
    ], size=15, lh=25)

    # ══════════ ② 分开称重 ═══════════════════════════════════════
    y1 = y0 + PH1 + 20
    PH2 = 430
    top = f.panel(0, y1, W_, PH2,
                  "② 那到底省了多少 ——&#160;四个张量分开称一次", OR,
                  sub="⚠️ 本课自己按两份 Algorithm 1 数的，论文没给过这个数")

    ROWS = [("Q", C1["Q"], C2["Q"]),
            ("K、V", C1["KV"], C2["KV"]),
            ("O（读＋写）", C1["O"], C2["O"])]
    MAXV = max(max(a, b) for _, a, b in ROWS)
    BX, BW = 300, 560
    by = top + 54
    for name, v1, v2 in ROWS:
        f.t(BX - 16, by + 16, name, INK, True, 16, "end")
        w1 = max(3, int(BW * v1 / float(MAXV)))
        w2 = max(3, int(BW * v2 / float(MAXV)))
        f.box(BX, by, w1, 20, "#efe9f6", PU, 3, 1.2)
        f.t(BX + w1 + 10, by + 15, "FA1　%d" % v1, PU, True, 15)
        f.box(BX, by + 26, w2, 20, "#e4f1e7", GR, 3, 1.2)
        f.t(BX + w2 + 10, by + 41, "FA2　%d" % v2, GR, True, 15)
        by += 70
    f.line(BX - 170, by - 10, BX + BW + 180, by - 10, LINE2, 1.2, arrow=False)
    f.t(BX - 16, by + 18, "合计", INK, True, 17, "end")
    f.t(BX, by + 18, "FA1 %d　　FA2 %d　　→　只差 %.2f 倍"
        % (T1, T2, RATIO), RD, True, 18)

    f.lines(60, by + 56, 1280, [
        "⛔ <tspan font-weight=\"700\">只看 O 那一行，会得出一个错的量级</tspan>："
        "%d 对 %d，是 <tspan font-weight=\"700\">%.0f 倍</tspan>。"
        "「一次都不落 HBM」这句话本身没错，"
        "<tspan font-weight=\"700\">但拿它当总账就错了</tspan>。"
        % (C1["O"], C2["O"], O_ONLY),
        "⭐ 因为 <tspan font-weight=\"700\">FA2 把省下的 O 流量，"
        "又用 K/V 的重读还了回去</tspan>："
        "K/V 在 FA1 里待在外圈、只读一趟；到 FA2 跑进内圈，"
        "每个 Q 块都要把全部 K/V 重新走一遍。",
        "⚠️ 换成非 causal 口径这个比值是 %.3f，"
        "<tspan font-weight=\"700\">结论不挑口径</tspan>；"
        "而且这是按 HBM 读写次数数的理想账，真机上 L2 会吃掉一部分 K/V 重读 ——&#160;"
        "%.2f 倍是<tspan font-weight=\"700\">偏乐观那一侧</tspan>。"
        % (_R_NC, RATIO),
    ], size=15, lh=25)

    # ══════════ ③ warp 层：同一条判据再用一遍 ═══════════════════════
    y2 = y1 + PH2 + 20
    PH3 = 386
    top = f.panel(0, y2, W_, PH3,
                  "③ 同一条判据，在 warp 这一层又用了一遍", GR,
                  sub="一个 thread block 内部，4 个 warp 怎么分活")

    def warps(x, y, split_q, col, title):
        f.t(x + 100, y - 30, title, col, True, 18, "middle")
        # 输出块：左边是 4 份部分和叠在一起，右边是 4 条各自定稿的横条
        f.box(x, y, 200, 160, "#fff", LINE, 6, 1.2)
        f.t(x + 100, y - 6, "输出块 O_i", GY2, False, 13, "middle")
        for k in range(4):
            if split_q:
                f.box(x + 6, y + 6 + k * 38, 188, 32,
                      ["#e4f1e7", "#d8ebdd", "#e4f1e7", "#d8ebdd"][k],
                      GR, 4, 1.2)
                f.t(x + 100, y + 27 + k * 38, "warp %d 的那一片（定稿）" % k,
                    GR, True, 13, "middle")
            else:
                f.box(x + 6 + k * 5, y + 6 + k * 5, 176, 142,
                      "none", PU, 4, 1.2, dash="4,3")
        if not split_q:
            f.t(x + 100, y + 86, "4 份部分和", PU, True, 15, "middle")
            f.t(x + 100, y + 108, "谁都不完整", PU, False, 13, "middle")
        return y + 160

    WY = top + 62
    wb = warps(80, WY, False, PU, "FA1：切 K／V（论文原词 “split-K”）")
    warps(760, WY, True, GR, "FA2：切 Q")

    f.lines(80, wb + 28, 560, [
        "4 个 warp 各持 K/V 的一片，<tspan font-weight=\"700\">Q 大家共用</tspan>。",
        "每个 warp 只算得出<tspan font-weight=\"700\">一份部分和</tspan> ——",
        "必须写进共享内存、同步、再加起来。",
        "⛔ 论文原话：这些读写<tspan font-weight=\"700\">拖慢了前向</tspan>。",
    ], size=15, lh=25)
    f.lines(760, wb + 28, 560, [
        "4 个 warp 各持 Q 的一片，<tspan font-weight=\"700\">K/V 大家共用</tspan>。",
        "每个 warp 直接算出<tspan font-weight=\"700\">自己那一片完整输出</tspan> ——",
        "<tspan font-weight=\"700\">warp 之间完全不需要通信。</tspan>",
        "⭐ 跟上面 Q 块之间互不通信，是同一件事。",
    ], size=15, lh=25)

    yy = f.band(y2 + PH3 + 2, "ok", "一条判据，两个尺度通用", [
        "<tspan font-weight=\"700\">切「要被累加的那一维」就得合；"
        "切「各自独立出结果的那一维」就不用合。</tspan>"
        "attention 里前者是 K／V（它们在求和号里面），后者是 Q（每行输出各管各的）。",
        "⭐ 所以 thread block 那一层和 warp 那一层做的是同一个动作 ——&#160;"
        "<tspan font-weight=\"700\">把外圈让给 Q</tspan>。"
        "这条判据的用处远不止 attention："
        "任何融合 kernel 分工前，先问一句「我切的这一维在不在求和号里」。",
    ])

    yy = f.band(yy + 12, "warn", "三处别讲过头", [
        "① <tspan font-weight=\"700\">论文从没说过「对调是为了让 O 不落 HBM」</tspan>。"
        "它给的理由是 ① 减少 non-matmul FLOP ② occupancy ——&#160;"
        "本图那个 1.49 倍是<tspan font-weight=\"700\">本课自己数的</tspan>，"
        "FA2 连 IO-complexity 定理都没有。",
        "② <tspan font-weight=\"700\">只对调了 forward</tspan>。"
        "FA1 的 backward 和 FA2 的 backward "
        "<tspan font-weight=\"700\">都还是 K/V 在外层</tspan>。",
        "③ 这个顺序<tspan font-weight=\"700\">不是 FA2 首创</tspan>：论文自己写的是"
        "“first suggested and implemented by "
        "<tspan font-weight=\"700\">Phil Tillet</tspan> in the Triton "
        "implementation”。",
    ])

    yy = f.src(yy + 14,
               "📌 FA1 arXiv 2205.14135 §3.1 Algorithm 1（外层 K/V、内层 Q；"
               "O_i、ℓ_i、m_i 的写回在内层循环里，line 12–13）；"
               "Figure 1 caption 逐字「In the outer loop (red arrows) ... "
               "loops through blocks of the K and V matrices」。",
               "📌 FA2 arXiv 2307.08691 §3.1.1 Algorithm 1（外层 Q、内层 K/V，"
               "O_i 与 L_i 在内层结束后各写一次）；宣布对调那句在 §3.2 "
               "“Parallelism”，并注明归功于 Phil Tillet 的 Triton 实现。"
               "warp 层全部出自 §3.3 与 Figure 3 ——&#160;"
               "⚠️ FA1 论文里 “warp” 出现 0 次，那段是 FA2 的回溯。",
               "⛔ 原词是 “split-K”，不是坊间常见的 “sliced-K”（两篇全文零次）。"
               "⚠️ 本图<tspan font-weight=\"700\">不引用任何加速比</tspan> ——&#160;"
               "FA2 论文里那几个百分比分属不同口径（"
               "attention 单算子 vs 端到端训练、不同硬件配置），"
               "混用会得出错的结论，本讲 §3.6b 专门讲过这条纪律。")

    f.save("fig3-loop-order.svg", yy + 10)


if __name__ == "__main__":
    main()
