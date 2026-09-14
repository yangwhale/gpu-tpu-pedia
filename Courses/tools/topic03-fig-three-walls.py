# -*- coding: utf-8 -*-
r"""专题三 · §3.5「三堵墙，方向各不相同 —— 而且甜点是个绝对值」

⭐⭐⭐ 2026-09-14 夜间 R34 新画。这一格是**先审出错才画的**，顺序反过来了：
   §3.5 原本是一张表加一段话、没有图，而它是全讲**标 ⭐ 最多**的一节。
   按第一原则去核 `tpu/Hunyuan3-295B-Pretraining/TUNING-v7.md`，一节里核出 7 处，
   **错得最狠的恰好是那句被框成「⚠️ 一条可迁移的教训」的话**。

═══ 审计结论（每条都对过源文件行号） ═══
① 「v7 64 芯片」    → 消融头写的是 **16 chip / 20 层 / pdbs 8 / seq 4096**。
② 「2048 甜点 = 228.4」→ 228.4 是 run **S1 = 全 2048 ＋ use_max_logit_estimate=30**。
   纯块大小的基线是 **B1 = 223.6**。⛔ 把另一个开关的 +2.1% 算进了块大小的功劳。
③ 「4096 → VMEM 直接爆（爆在反向）」→ splash 侧源文件只写「往上撞 VMEM 墙」，
   **没有反向的说法，也没有 OOM 现场**。明写「4096 直接 OOM」的是 §3.4.4 的
   **MoE ragged-dot `tile_k`** —— ⛔ 另一个 kernel，张冠李戴。
④ 「compute 压回 2048 → −11.5%」→ 全项目 `11.5%` 只有一处：
   `23% × 50% = 11.5%`，是**「把 forward 优化到无限快」的端到端收益天花板**。
   ⛔ 一个收益上限被搬成了一个实测跌幅，而且那个 run 根本不存在。
⑤ 「512 → −1.0%」→ 对应 run **S2**，改的是**整套官方非均匀布局**，
   `sa_block_kv_compute=512` 只是其中一个字段。归因方向成立，口径要限定。
⑥ 「块大小要看 block/seq 的**比例**，不是绝对值」→ ⛔ **正好讲反了。**
⑦ 「块 ≈ seq/2 …… ⚠️ 这一条尚未验证」→ ⛔ **已经验过，而且被证伪。**

═══ ⑥⑦ 是怎么错的 —— 这一条比错误本身更值得记 ═══
源文件**自己前后矛盾**，而且矛盾隔着两千行：
  · 第 634 行「方法论教训 ①」：看 `block/seq` 的比例，不是绝对值。
    —— 它是从**一个数据点**（照抄 512，−1.0%）反推出来的。
  · 第 2803 行「四条可复用结论 ①」：**`block=2048` 是硬件甜点，与 seq 无关。**
    seq 4096 和 16384 上最优块**都是 2048**（分别 = seq/2 和 seq/8），
    「共同点是**绝对值不是比例**」。—— 它背后是**一次直接实验**。
⭐ 后一轮（2026-08-09）用 seq 16384 把前一轮的规则打掉了，课件读到前一半就停了。
⛔ 判据：**同一份长文档里，晚出现的结论可能推翻早出现的。**
  引一句「教训」之前，先搜一遍全文还有没有同主题的第二段。

═══ 那这一节到底该怎么讲 ═══
⭐ 更正之后反而讲得通了，因为「绝对值」是**可以从 kernel 结构推出来的**：
  片上工作集 = Q[b×d] ＋ K[b×d] ＋ V[b×d] ＋ S[b×b]。**这四项里一个 seq 都没有。**
  seq 只决定你要绕几趟（seq ÷ b）。所以甜点由 VMEM 顶死，而 VMEM 不知道你的 seq。
⭐⭐ 顺手掉出一个本讲**完全没讲过**的闭环：b = seq 时「KV 方向切 1 块」＝ 压根没分块，
  S 就是整张注意力矩阵 —— **你回到了 §3.1 那个 FlashAttention 要解决的原始问题**。
  所以那堵容量墙不是巧合，它就是 §3.1 那堵墙本人。

═══ 画法 ═══
① 一根块大小轴，三堵墙从**不同方向**压进来（左一右二），剩下的缝正对 2048。
   ⛔ 没测过的点（1024）用虚线段画，不跟实测的实线混。
② 判决格：seq 4096 与 seq 16384 两条轴**按同一个绝对像素尺**画，
   两个最优点因此**竖直对齐** —— 比例规则若成立，它们应该散开。
   比例规则对 seq 16384 的预测（8192）画成空心虚线圈 ＋ ✗。
③ 根因格：两块 VMEM，S 块**按面积等比**画（块翻倍 → S 翻四倍）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topic03_draw import (Fig, BL, OR, GR, RD, GY, GY2, PU, INK, LINE, wpx)   # noqa: E402

W_ = 1400


def B(s):
    """加粗。⛔ 图里不能写 markdown 的 `**` ——&#160;SVG 不渲染，会原样印出来
    （topic03_draw 里有断言挡着）。wrap_rich 会正确处理跨行的 tspan。"""
    return '<tspan font-weight="700">%s</tspan>' % s


# ══ 一、把要画的数当场算一遍 ══════════════════════════════════════
# ⛔ 这一格是因为「引了没核的数」才存在的，所以它自己的数一个都不能靠记。

MIB = 1024 ** 2
D_HEAD = 128        # head_dim=128，源文件 §B.7 结论 ③ 的口径
EB = 2              # bf16 输入
AB = 4              # ⭐ MXU 在 fp32 里累加，所以 S 块是 4 字节不是 2


def s_tile(b):
    """S 块 = [b_q, b_kv]，**面积随块大小平方增长** —— 容量墙的来源。"""
    return b * b * AB


def qkv_tile(b):
    """Q/K/V 三块都是 [b, d]，**线性**增长。量级上打不过 S。"""
    return 3 * b * D_HEAD * EB


assert s_tile(2048) == 16 * MIB
assert s_tile(4096) == 64 * MIB
assert s_tile(4096) == 4 * s_tile(2048), "块翻倍 S 要翻四倍，不然这张图的面积是骗人的"
assert qkv_tile(2048) == 1536 * 1024            # 1.5 MiB —— 比 S 小一个数量级还多
assert s_tile(2048) // qkv_tile(2048) >= 10

VMEM = 64 * MIB     # 课件 §3.5 已述：Ironwood 每个 TensorCore 64 MB VMEM
# ⭐ 这张图最硬的一个点：块开到 4096，光 S 一块就正好把整个 VMEM 占满。
assert s_tile(4096) == VMEM

# ── 消融实测（源文件「消融实测」表，16 chip / 20 层 / pdbs 8 / seq 4096）──
B1, S1, S2 = 223.6, 228.4, 221.3          # 全2048 ／ 全2048+max_logit=30 ／ 官方非均匀
assert abs((S1 - B1) / B1 * 100 - 2.1) < 0.06, "S1 相对 B1 应是 +2.1%"
assert abs((S2 - B1) / B1 * 100 + 1.0) < 0.06, "S2 相对 B1 应是 −1.0%"
# ⛔ 课件原来把 228.4 当成「块大小 2048」的成绩 —— 它里面含着 max_logit 那 +2.1%。
assert S1 != B1, "228.4 与 223.6 不是同一个 run，这正是审计第 ② 条"

# ── 判决格：最优块在两个 seq 上都是 2048 ──────────────────────────
OPT = 2048
assert OPT * 2 == 4096 and OPT * 8 == 16384       # 分别 = seq/2 与 seq/8
PRED = 16384 // 2                                  # 「比例规则」对 seq 16384 的预测
assert PRED == 8192 and PRED != OPT, "比例规则若成立，最优块应当跟着 seq 走"

# ── 轴刻度：seq 4096 时 KV 方向切出几块 ──────────────────────────
BLOCKS = [512, 1024, 2048, 4096]
NCHUNK = {b: 4096 // b for b in BLOCKS}
assert [NCHUNK[b] for b in BLOCKS] == [8, 4, 2, 1]


def fmt_mib(n):
    return "%d MiB" % (n // MIB)


# ══ 二、画 ════════════════════════════════════════════════════════
def main():
    f = Fig(W_, "Splash attention 块大小的三堵墙，以及最优块为什么是个绝对值")
    f.header("三堵墙，方向各不相同 —— 而且甜点是个" + B("绝对值"),
             "块大小往上撞两堵、往下撞一堵；剩下的缝正对 2048，"
             "而这个 2048 换到 seq 16384 上" + B("没有跟着动"),
             legend=[(GR, "实测"), (GY2, "没测过"), (RD, "被证伪")])

    # ── 面板① 三堵墙 ─────────────────────────────────────────────
    # ⛔⛔ R34 一稿把「墙的名字」和「墙本身」分在了图的两头 ——&#160;
    #   碎块开销的条在左、名字标在右；并行度的条在右、名字标在左。
    #   读者会**交叉连错**，而图上没有任何东西提示他连错了。
    # ⭐ 判据：**名字必须长在它命名的那个东西上。** 条上只写名字，
    #   机制说明统一收到下面的色标三行里，靠颜色绑定。
    PH1 = 362
    Y1 = 76
    f.panel(30, Y1, 1340, PH1, "① 三堵墙，方向各不相同", BL,
            tag="seq = 4096")

    XS = {512: 260, 1024: 560, 2048: 860, 4096: 1160}

    f.t(60, 130, "块大小（以及 KV 方向因此切出几块）", GY, size=12)

    # 甜点那一列先铺底 —— ⛔ 必须画在墙之前，否则它会盖住墙
    f.spot(XS[2048] - 72, 206, 144, 96, "#e6f4ea")

    for b in BLOCKS:
        x = XS[b]
        hot = (b == OPT)
        f.t(x, 166, str(b), GR if hot else INK, bold=True,
            size=16 if hot else 14, anchor="middle")
        f.t(x, 188, "切 %d 块" % NCHUNK[b], GY2, size=12, anchor="middle")
    f.line(150, 202, 1290, 202, LINE, 1, arrow=False)

    # ── 上排：往下那一堵（从左压来） ＋ 往上的容量墙（从右压来）
    #   ⛔ 512 是实测（S2，−1.0%），1024 **没测过** —— 所以实线只画到 512，
    #     再用虚线延到 2048 前。两种线不能混，混了就是拿没测的冒充测过的。
    f.box(150, 214, 260, 32, "#fce8e6", RD, 5, 1.2)
    f.t(280, 234, "往下：碎块开销", RD, size=12.5, bold=True, anchor="middle")
    f.box(410, 214, 330, 32, "none", GY2, 5, 1.2, dash="5 4")
    f.t(575, 234, "1024 没测过，但同向", GY2, size=12, anchor="middle")
    f.line(744, 230, 788, 230, RD, 1.8)

    f.box(1010, 214, 300, 32, "#fce8e6", RD, 5, 1.2)
    f.t(1160, 234, "往上：容量墙", RD, size=12.5, bold=True, anchor="middle")
    f.line(1004, 230, 950, 230, RD, 1.8)

    # ── 下排：往上的第二堵（并行度），同样从右压来
    f.box(1010, 258, 300, 32, "#fef7e0", OR, 5, 1.2)
    f.t(1160, 278, "往上：并行度", OR, size=12.5, bold=True, anchor="middle")
    f.line(1004, 274, 950, 274, OR, 1.8)

    f.t(XS[2048], 322, "三面都够不着的，只剩这一列", GR, bold=True, size=13.5,
        anchor="middle")

    # ── 色标三行：机制说明在这里，靠颜色绑回上面的条 ──────────────
    for i, (col, name, why) in enumerate([
            (RD, "往下 · 碎块开销",
             "每块的固定开销（mask 检查、running max/sum 更新、pipeline stage 切换）摊不动。"
             "512 实测 −1.0%。"),
            (RD, "往上 · 容量墙",
             "S 块是 b×b，块开到 4096 时它一个人就占满 64 MiB VMEM ——&#160;见面板 ③。"),
            (OR, "往上 · 并行度",
             "KV 方向只切得出一块，那一维的流水直接塌掉。"
             "⚠️ 这一堵是从 kernel 结构推的，源文件未实测。")]):
        yy = 352 + i * 26
        f.box(62, yy - 9, 11, 11, col, col, 2)
        f.t(84, yy, name, col, bold=True, size=12.5)
        f.t(84 + wpx(name, 12.5) + 14, yy, why, GY, size=12)

    # ── 面板② 判决：比例还是绝对值 ───────────────────────────────
    PH2 = 282
    Y2 = 454
    f.panel(30, Y2, 1340, PH2, "② 判决：最优块跟着 seq 走吗", PU,
            tag="两条轴按同一个绝对像素尺画")

    # ⭐ 这一格全部的说服力来自「同尺」：两条 seq 轴共用一个 x(block) 映射，
    #   所以最优点对齐与否是**看出来的**，不是我写在标签里的。
    def xb(b):
        step = 130
        n = 0
        while (512 << n) < b:
            n += 1
        assert (512 << n) == b, "只画 2 的幂"
        return 300 + step * n

    assert xb(2048) == 560 and xb(8192) == 820

    def axis(y, seq, tag, note):
        f.t(60, y + 5, "seq = %d" % seq, INK, bold=True, size=14)
        f.t(60, y + 24, tag, GY2, size=11.5)
        hi = seq
        f.line(290, y, xb(hi) + 30, y, GY2, 1.3, arrow=False)
        b = 512
        while b <= hi:
            x = xb(b)
            f.line(x, y - 5, x, y + 5, GY2, 1.2, arrow=False)
            f.t(x, y + 22, str(b), GY2, size=11, anchor="middle")
            b *= 2
        # 实测最优：实心绿点。
        # ⛔ R34 一稿把 note 摆在点的**正上方**，于是它跟竖直对齐线的标签
        #   叠在了一起（两行字压成一团）。⭐ 点上方要留给那条线的标签，
        #   所以 note 改成挂在点的左边。
        f.box(xb(OPT) - 8, y - 8, 16, 16, GR, GR, 8)
        f.t(xb(OPT) - 16, y + 5, note, GR, bold=True, size=12, anchor="end")

    axis(534, 4096, "生产形状", "最优 = seq/2")
    axis(620, 16384, "同一个 kernel，只把 seq 拉长 4 倍", "最优 = seq/8")

    # 比例规则的预测：空心虚线圈 ＋ ✗
    f.box(xb(PRED) - 9, 620 - 9, 18, 18, "none", RD, 9, 1.4, dash="3 3")
    f.t(xb(PRED), 620 - 16, "✗", RD, bold=True, size=14, anchor="middle")
    f.t(xb(PRED) + 22, 624, "比例规则预测最优块在这里（8192）—— 实测没有",
        RD, size=12)

    # 竖直对齐线 —— 这条线就是论点本身
    f.line(xb(OPT), 512, xb(OPT), 642, GR, 1.4, dash="4 4", arrow=False)
    f.t(xb(OPT), 506, "同一个绝对位置", GR, bold=True, size=12, anchor="middle")

    f.t(60, 678, "⭐ seq 拉长 4 倍，最优块纹丝不动 ——&#160;"
                 "它是 seq/2 还是 seq/8 纯属巧合，共同点是那个 2048 本身。",
        INK, size=13.5)
    f.t(60, 700, "⛔ 课件原文写的是「看 block/seq 的比例，不是绝对值」，"
                 "并注明「尚未验证」——&#160;方向反了，而且那次验证早就做完了。",
        RD, size=13)

    # ── 面板③ 根因 ──────────────────────────────────────────────
    PH3 = 356
    Y3 = 752
    f.panel(30, Y3, 1340, PH3, "③ 为什么是绝对值：片上那份工作集里，一个 seq 都没有", GR,
            tag="面积按真实字节数等比")

    SIDE = 200          # 200px ↔ 64 MiB
    BX1, BX2, BY = 110, 430, 826

    def vmem(x, blk, col):
        f.box(x, BY, SIDE, SIDE, "none", LINE, 6, 1.4)
        f.t(x + SIDE // 2, BY - 26, "块 = %d" % blk, col, bold=True, size=15,
            anchor="middle")
        f.t(x + SIDE // 2, BY - 8, "VMEM 64 MiB / TensorCore", GY2, size=11,
            anchor="middle")
        # S 块：面积正比于字节数 ⇒ 边长正比于块大小
        side = int(SIDE * blk / 4096.0)
        f.box(x, BY + SIDE - side, side, side, "#fce8e6" if blk == 4096 else "#e8f0fe",
              col, 4, 1.4)
        f.t(x + side // 2, BY + SIDE - side // 2 + 5, fmt_mib(s_tile(blk)),
            col, bold=True, size=13, anchor="middle")
        return side

    s1 = vmem(BX1, 2048, BL)
    s2 = vmem(BX2, 4096, RD)
    assert s2 == 2 * s1, "边长要正好两倍，面积才是四倍"

    f.t(BX1 + SIDE // 2, BY + SIDE + 24, "S 块 16 MiB，还剩得下 Q/K/V（合 1.5 MiB）",
        GY, size=12, anchor="middle")
    f.t(BX2 + SIDE // 2, BY + SIDE + 24, "S 块正好 64 MiB ——&#160;一个人占满，Q/K/V 无处可放",
        RD, size=12, anchor="middle")

    TX = 700
    f.t(TX, BY - 8, "片上工作集", INK, bold=True, size=14)
    f.t(TX, BY + 22, "Q[b×d]　＋　K[b×d]　＋　V[b×d]　＋　S[b×b]", INK, size=14)
    f.t(TX, BY + 48, "前三项线性，第四项平方 ——&#160;所以块一翻倍，S 翻四倍。",
        GY, size=12.5)
    f.t(TX, BY + 80, "⭐ 这四项里没有 seq。", GR, bold=True, size=15)
    f.t(TX, BY + 104, "seq 只决定你要绕几趟：seq ÷ b。",
        GY, size=12.5)
    f.t(TX, BY + 126, "甜点由 VMEM 顶死，而 VMEM 不知道你的 seq 是多少 ——",
        GY, size=12.5)
    f.t(TX, BY + 148, "这就是它为什么是个绝对值。", GR, bold=True, size=13.5)

    f.box(TX - 14, BY + 168, 640, 62, "none", OR, 6, 1.3)
    f.t(TX, BY + 192, "⭐⭐ 而 b = seq 时「切 1 块」＝ 压根没分块：",
        OR, bold=True, size=13.5)
    f.t(TX, BY + 214, "S 就是整张注意力矩阵，你回到了原点。"
                      "那堵容量墙就是 FlashAttention 本来要解决的那一堵。",
        OR, size=12.5)

    # ── 落点带 ───────────────────────────────────────────────────
    yy = 1132
    yy = f.band(yy, "ok", "带得走的那一条",
                ["块大小的最优值是" + B("硬件定的绝对值") + "，不是 `block/seq` 的比例。"
                 "换序列长度时" + B("不要按比例缩放块大小") + " ——&#160;"
                 "换硬件（VMEM 容量变了）才需要重扫。",
                 "照抄别人的配置之所以翻车，不是因为比例变了，"
                 "而是因为他们那个绝对值是给他们的硬件调的。"])
    yy = f.band(yy, "bad", "这一节是审出来的 ——&#160;原表四行错了三行",
                ["228.4 是 `全 2048 ＋ use_max_logit_estimate=30`（run S1）；"
                 "" + B("纯块大小 2048 的基线是 223.6") + "（run B1）。差的那 2.1% 是另一个开关的功劳。",
                 "「4096 → VMEM 爆在反向」与「compute 压回 2048 → −11.5%」"
                 "" + B("在源文件里都不存在") + "：前者是 MoE ragged-dot `tile_k` 的 OOM（另一个 kernel），"
                 "后者是「把 forward 优化到无限快」的收益天花板 `23% × 50%`。",
                 "规模也写错了：消融跑的是 " + B("16 chip / 20 层 / pdbs 8") + "，不是 64 芯片。"])
    yy = f.src(yy + 10,
               "📌 实测数据出自 `tpu/Hunyuan3-295B-Pretraining/TUNING-v7.md` "
               "「消融实测」表（16 chip / 20 层 / pdbs 8 / seq 4096）"
               "与「四条可复用结论」①。",
               "⛔ 该文件" + B("前后自相矛盾") + "：第 634 行的「方法论教训 ①」说看比例，"
               "两千行后的「可复用结论 ①」用 seq 16384 的直接实验说" + B("与 seq 无关") + "。"
               "后者晚一轮、有实验，以它为准。",
               "⚠️ 面板① 的「并行度墙」是从 kernel 结构推的，源文件只实测了容量墙那一侧；"
               "1024 那一档" + B("没有可追溯的 splash 实测") + "，故画成虚线。")

    f.save("fig3-three-walls.svg", yy + 10)


if __name__ == "__main__":
    main()
