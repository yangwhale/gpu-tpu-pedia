# -*- coding: utf-8 -*-
r"""专题六 · 14.7「两次落地」的两张静态图。

回答的问题
  · fig6-s13-ladder  Qwen3.5-397B 在 v7x-8 上，DP8 prefill 吞吐是哪几刀攒出来的？每刀各占多少？
  · fig6-s13-map     两次落地（Qwen3.5-397B、Kimi K3 megakernel）的每一刀，各打在哪一类瓶颈上？

数从哪来
  · fig6-s13-ladder：本课据私有实测指数化。起点那天的 DP8 prefill 吞吐记 1.00；
    四刀各自相对上一级的倍数 1.0565 ／ 1.0945 ／ 1.2553 ／ 1.091 取自紧邻台阶前后两次日常基准；
    连乘 1.584 与同期首尾实测之比 1.592 为本课推导。绝对值与对外对比口径一律不画。
  · fig6-s13-map：纯定性。Qwen3.5 各刀的归类依据私有记录的机制描述（只取机制，不取数）；
    K3 各刀依据 Inferact 公开博客 https://inferact.ai/blog/tpu-megakernels 与其 Apache-2.0 公开代码
    github.com/Inferact/tpu-megakernels；FP8 重打包「多读、换向量单元」为本课按公开代码推导。

刻意没画
  · ladder 不画 decode：decode 那几级跨了口径切换，还有一级是「修好退化」，不能和 prefill 连成一条梯子。
  · map 不给 K3 标每刀收益：公开材料没有逐刀消融，只能标方向。
  · 两张图都不画对外竞品对比。
"""
from topic03_draw import Fig, wpx, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400

# ══ ① Qwen3.5 prefill 刀谱（私有实测指数化）════════════════════════════
STEPS = [("起点", None, "旧版合并、整块通信"),
         ("SparseCore 合并", 1.0565, "片上按目的地累加"),
         ("通信切块", 1.0945, "算完一块发一块"),
         ("整层融合", 1.2553, "一层 MoE 一个程序"),
         ("修剪 kernel", 1.091, "砍掉多余来回")]
CUM = [1.0]
for _, k, _ in STEPS[1:]:
    CUM.append(CUM[-1] * k)
assert [round(c, 2) for c in CUM] == [1.0, 1.06, 1.16, 1.45, 1.58], CUM
MEASURED = 1.592                    # 同期首尾实测之比，本课据私有实测指数化（绝对值不进公开仓库）
assert round(MEASURED, 2) == 1.59 and round(CUM[-1], 3) == 1.584

# ══ ② 刀 × 瓶颈 ══════════════════════════════════════════════════════
COLS = ["HBM 读写", "通信露出", "形状不规整", "向量单元", "装不装得下", "编译"]
# (刀, {列: "x" 主攻 / "o" 代价})
QWEN = [("SparseCore 合并", {0: "x", 2: "x"}),
        ("通信切块", {1: "x"}),
        ("整层融合", {1: "x", 2: "x", 4: "o"}),
        ("修剪 kernel", {0: "x"}),
        ("KV 序列放最内层", {2: "x", 0: "x"}),
        ("小 batch 不排序", {2: "x"}),
        ("线性状态 BF16", {4: "x", 3: "x"})]
K3 = [("层循环进 kernel", {0: "x"}),
      ("VMEM 分时复用", {4: "x"}),
      ("专家槽轮转", {0: "x"}),
      ("手写集合通信", {1: "x"}),
      ("MXFP4 换 FP8", {3: "x", 0: "o"}),
      ("残差流再切", {3: "x"}),
      ("兼做验证器", {0: "x"}),
      ("整块编译", {5: "x"})]


def fig_ladder():
    f = Fig(W, "Qwen3.5-397B 在单机 TPU v7x-8 上 DP8 prefill 吞吐的四级台阶，按私有实测指数化，起点记 1.00。"
               "SparseCore 合并这一刀乘 1.06，到 1.06；通信切块乘 1.09，到 1.16；整层 MoE 融合成一个 Pallas 程序乘 1.26，到 1.45，是最大的一级；"
               "修剪 kernel 内部乘 1.09，到 1.58。虚线是同期首尾实测之比 1.59，四刀几乎解释了全部涨幅")
    y0 = f.header("Qwen3.5-397B prefill 刀谱　——　<tspan font-weight=\"700\">最大的一级是整层融合</tspan>",
                  "私有实测指数化：单机 v7x-8、DP8 注意力 ＋ EP8 专家，prefill 吞吐，起点那天 ＝ 1.00。柱内大字是累计值，小字是这一刀相对上一级的倍数",
                  [(GY2, "起点"), (BL, "动通信和搬行"), (GR, "整层融合"), (INK, "同期实测")])
    PH = 440
    py = f.panel(0, y0, W, PH, "DP8 prefill 吞吐（指数）", BL)
    X0, AW, AY, AH = 110, 1150, py + 40, 300
    YMAX = 1.8

    def Y(v):
        return AY + AH - AH * v / YMAX
    for v in (0, 0.5, 1.0, 1.5):
        f.line(X0, Y(v), X0 + AW, Y(v), "#eceff1", 1, arrow=False)
        f.t(X0 - 12, Y(v) + 5, "%.1f" % v, GY, size=14, anchor="end")
    f.line(X0, Y(0), X0 + AW, Y(0), GY2, 1.2, arrow=False)
    n = len(STEPS)
    slot = AW / n
    bw = 132
    for i, ((nm, k, how), c) in enumerate(zip(STEPS, CUM)):
        cx = X0 + slot * (i + 0.5)
        col = GY2 if k is None else (GR if nm == "整层融合" else BL)
        f.box(cx - bw / 2, Y(c), bw, Y(0) - Y(c), col, col, 3)
        f.t(cx, Y(c) + 28, "%.2f" % c, "#fff", True, 19, "middle")
        if k is not None:     # 这一刀的倍数放柱内第二行：放柱顶会撞上「同期实测」那条虚线
            f.t(cx, Y(c) + 52, "本刀 ×%.2f" % k, "#fff", size=14, anchor="middle")
        f.t(cx, Y(0) + 28, nm, INK, True, 15, "middle")
        f.t(cx, Y(0) + 50, how, GY, size=13, anchor="middle")
    # 同期实测
    ym = Y(MEASURED)
    f.line(X0, ym, X0 + AW, ym, INK, 1.6, dash="6,4", arrow=False)
    f.t(X0 + 8, ym - 10, "同期实测 %.2f" % MEASURED, INK, True, 15)
    f.t(X0 + 8 + wpx("同期实测 %.2f" % MEASURED, 15) + 18, ym - 10, "四级连乘 %.3f，差 0.5%%" % CUM[-1], GY, size=13)
    f._pan = None
    yb = f.src(py + PH + 10,
               "📌 本课据私有实测指数化：起点那天的 DP8 prefill 吞吐 ＝ 1.00；四级倍数取紧邻台阶前后两次日常基准。",
               "连乘 1.0565 × 1.0945 × 1.2553 × 1.091 ≈ 1.584，同期首尾实测之比 ≈ 1.592，均为本课推导；"
               "差的 0.5% 是台阶之间的日间漂移，与起点那段小幅回落相抵后的余数。绝对吞吐与对外对比口径不画。")
    f.save("fig6-s13-ladder.svg", yb + 14)


def fig_map():
    f = Fig(W, "两张表，行是每一刀，列是六类瓶颈：HBM 读写、通信露出、形状不规整、向量单元、装不装得下、编译。"
               "实心点是这一刀主攻的瓶颈，空心圈是它付出的代价。"
               "Qwen3.5 的点集中在通信和形状两列，整层融合的代价是片上 SMEM 装不下大的编译档。"
               "K3 的点集中在 HBM 读写一列，MXFP4 换 FP8 主攻向量单元、代价是多读 HBM；整块编译单独占编译一列")
    y0 = f.header("每一刀打在哪类瓶颈上　——　<tspan font-weight=\"700\">并发几百砍通信，并发个位数砍空等</tspan>",
                  "行 ＝ 一刀，列 ＝ 它要解决的瓶颈。定性归类，不代表收益大小；K3 没有逐刀消融",
                  [(BL, "Qwen3.5-397B · v7x-8 · 几百并发"), (OR, "Kimi K3 · 16 颗 v7 · batch 1–8"), (GY2, "空心 ＝ 代价")])
    LW = 250
    CW = (W - LW - 20) / len(COLS)
    RH = 36

    def table(y, title, rows, col):
        PH = 30 + 50 + RH * len(rows) + 14
        py = f.panel(0, y, W, PH, title, col)
        hy = py + 34
        for j, c in enumerate(COLS):
            f.t(LW + CW * (j + 0.5), hy, c, INK, True, 15, "middle")
        f.line(12, hy + 14, W - 12, hy + 14, LINE, 1, arrow=False)
        for i, (nm, marks) in enumerate(rows):
            ry = hy + 14 + RH * i + RH / 2
            if i % 2:
                f.box(12, ry - RH / 2, W - 24, RH, "#f8f9fa", "none", 0)
            f.t(28, ry + 5, nm, INK, size=15)
            for j, m in marks.items():
                cx = LW + CW * (j + 0.5)
                if m == "x":
                    f.p.append('<circle cx="%.1f" cy="%.1f" r="9" fill="%s"/>' % (cx, ry, col))
                else:
                    f.p.append('<circle cx="%.1f" cy="%.1f" r="8" fill="#fff" stroke="%s" stroke-width="2"/>' % (cx, ry, GY2))
        return py - 30 + PH
    y1 = table(y0, "Qwen3.5-397B：前四刀在 prefill，后三刀在 decode", QWEN, BL)
    y2 = table(y1 + 16, "Kimi K3 megakernel：整个模型一次调用", K3, OR)
    f._pan = None
    yb = f.src(y2 + 10,
               "📌 Qwen3.5 各刀的机制归类依据私有记录，只取机制不取数。K3 依据 inferact.ai/blog/tpu-megakernels（2026-09-23）"
               "与其 Apache-2.0 公开代码 github.com/Inferact/tpu-megakernels。",
               "「MXFP4 换 FP8」的代价与收益为本课按公开代码推导：专家每元素位数 12.75 → 20.37（三块合计），即 ×1.60；"
               "缩放从每 32 个元素一次变成每 256 个一次，向量单元上这部分活约为原来的 1/8。")
    f.save("fig6-s13-map.svg", yb + 14)


if __name__ == "__main__":
    fig_ladder()
    fig_map()
