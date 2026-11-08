# -*- coding: utf-8 -*-
r"""专题四 · §3.3b「Muon 从零讲」第三张：Muon 的一步，从梯度到权重

2026-11-08 补。原来这一讲只说了 Muon「对更新矩阵做正交化」，没走过一遍完整的一步；
台下不知道动量在正交化之前还是之后、为什么要先除一下、哪些参数根本不走 Muon。

两格：
  Ⓐ 一步六站：梯度 → 动量（0.95 旧 ＋ 0.05 新）→ Nesterov 再掺一点当前梯度 → 除以总大小
     → 五轮 Newton–Schulz → 乘形状系数和学习率，减到权重上。
     每站底下画「三个频率的强度」怎么变（例子 10 ／ 1 ／ 0.01，数来自 topic04_muon_nums）。
  Ⓑ 哪些参数走 Muon：隐藏层里的二维矩阵；嵌入、输出头、norm 的增益、偏置走 Adam。

📌 出处：公开实现 github.com/KellerJordan/Muon 的 muon.py（muon_update：
  momentum.lerp_(grad, 1−β)、nesterov 时 grad.lerp_(momentum, β)、zeropower_via_newtonschulz5、
  update *= max(1, rows/cols)**0.5、p.add_(update, alpha=−lr)）；
  「嵌入、输出头、增益、偏置用 AdamW」见同文件 docstring 与作者博客。
刻意没画：权重衰减（公开实现里默认 0；上规模时要加，§3.4 末尾讲）。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)
import topic04_muon_nums as N

W = 1400
assert N.BETA == 0.95 and N.STEPS == 5
assert abs(N.SHAPE_K - 1.732) < 1e-3


def bars(f, x, y, vals, cols, top, h=70, bw=18, gap=8, labels=None):
    """三根强度条，底边在 y，满刻度 top。"""
    for k, v in enumerate(vals):
        hh = max(0.0, min(1.0, v / top)) * h
        f.box(x + k * (bw + gap), y - hh, bw, max(hh, 1.2), cols[k], cols[k], 2)
        if labels:
            f.t(x + k * (bw + gap) + bw / 2, y - hh - 6, labels[k], INK, True, 11.5, "middle")
    f.line(x - 4, y, x + 3 * (bw + gap), y, GY2, 1, arrow=False)


def main():
    f = Fig(W, "Muon 的一步分六站：梯度，动量按 0.95 旧加 0.05 新记下来，Nesterov 再掺一点当前梯度，"
               "除以总大小让所有强度不超过一，五轮 Newton–Schulz 把强度推到一附近，"
               "最后乘形状系数和学习率减到权重上。底下三根条是三个频率的强度：10、1、0.01 先变成 0.995、0.0995、0.001，"
               "五轮后变成 0.70、0.71、0.47。第二格：只有隐藏层的二维矩阵走 Muon，嵌入、输出头、norm 增益和偏置走 Adam")
    y0 = f.header("Muon 的一步　——　<tspan font-weight=\"700\">前面是普通的动量，后面多了一道「把强度拉平」</tspan>",
                  "每一站下面的三根条：同一个更新矩阵里三个频率的强度（例子：10 ／ 1 ／ 0.01）",
                  [(BL, "频率 ①"), (OR, "频率 ②"), (PU, "频率 ③（很弱）")])
    COLS = (BL, OR, PU)

    PH = 352
    py = f.panel(0, y0, W, PH, "Ⓐ 一步六站", BL, sub="左边三站就是带动量的 SGD；Muon 自己的东西只在右边三站")
    BW, BH, GAP = 196, 112, 32
    STAGES = (
        ("① 这一步的梯度", "G", "反向算出来的", BL),
        ("② 动量：记性", "M ← 0.95·M ＋ 0.05·G", "旧的占九成五", BL),
        ("③ 再掺一点当前", "U ＝ 0.95·M ＋ 0.05·G", "Nesterov：往前看一眼", BL),
        ("④ 除以总大小", "U ÷ ‖U‖", "所有强度都 ≤ 1", GR),
        ("⑤ 五轮迭代", "Newton–Schulz × 5", "强度都推到 1 附近", GR),
        ("⑥ 减到权重上", "W ← W − 学习率 × k × O", "k ＝ √max(1, 行/列)", GR),
    )
    for i, (nm, eq, note, col) in enumerate(STAGES):
        x = 24 + i * (BW + GAP)
        f.box(x, py + 24, BW, BH, "#e8f0fe" if col == BL else "#e6f4ea", col, 8)
        f.t(x + BW / 2, py + 52, nm, col, True, 15, "middle")
        f.t(x + BW / 2, py + 80, eq, INK, True, 12.5, "middle")
        f.t(x + BW / 2, py + 108, note, GY, size=12, anchor="middle")
        if i < 5:
            f.line(x + BW + 4, py + 24 + BH / 2, x + BW + GAP - 4, py + 24 + BH / 2, GY2, 1.8)
    # 强度条：③ 用进来的强度（满刻度 10），④ 缩放后（满刻度 1），⑤⑥ 五轮后（满刻度 1）
    yb = py + 262
    rows = ((2, N.RAW, 10.0, ["10", "1", "0.01"]),
            (3, N.SCALED, 1.0, ["0.995", "0.0995", "0.001"]),
            (4, N.FINAL, 1.0, ["%.2f" % v for v in N.FINAL]))
    for i, vals, top, labs in rows:
        x = 24 + i * (BW + GAP) + 46
        bars(f, x - 14, yb, vals, COLS, top, labels=labs, bw=26, gap=24)
    f.t(24 + 2 * (BW + GAP) + BW / 2, yb + 30, "进来的更新：强度差一千倍", BL, True, 13, "middle")
    f.t(24 + 3 * (BW + GAP) + BW / 2, yb + 30, "比例没变，整体缩到 ≤ 1", GR, True, 13, "middle")
    f.t(24 + 4 * (BW + GAP) + BW / 2, yb + 30, "差一千倍 → 差不到两倍", GR, True, 13, "middle")
    f.t(24 + 5 * (BW + GAP) + BW / 2, yb - 40, "比如 6144×2048：", GY, size=12.5, anchor="middle")
    f.t(24 + 5 * (BW + GAP) + BW / 2, yb - 18, "k ＝ √3 ≈ %.2f" % N.SHAPE_K, INK, True, 14, "middle")
    f.t(24 + 5 * (BW + GAP) + BW / 2, yb + 30, "整体多大，学习率说了算", GR, True, 13, "middle")
    f.t(30, py + PH - 46, "注：③ 底下的条按满刻度 10 画，频率 ③ 只有 0.01，细到看不见 ——　这正是问题所在",
        GY, size=12.5)
    f._pan = None

    # ══ Ⓑ 谁走 Muon ════════════════════════════════════════════
    PH2 = 210
    py2 = f.panel(0, py + PH + 22, W, PH2, "Ⓑ 不是所有参数都走 Muon", PU,
                  sub="只管隐藏层里的二维矩阵；其余仍走 Adam")
    items = (("嵌入表", "Adam", OR), ("注意力 Q／K／V／O", "Muon", GR), ("MLP 三块矩阵", "Muon", GR),
             ("norm 的增益", "Adam", OR), ("偏置", "Adam", OR), ("输出头", "Adam", OR))
    IW, IG = 196, 32
    for i, (nm, who, col) in enumerate(items):
        x = 24 + i * (IW + IG)
        f.box(x, py2 + 40, IW, 70, "#e6f4ea" if who == "Muon" else "#fef7e0", col, 8)
        f.t(x + IW / 2, py2 + 70, nm, INK, True, 14.5, "middle")
        f.t(x + IW / 2, py2 + 96, "走 " + who, col, True, 14, "middle")
    f.t(30, py2 + 146, "一个数、一排数（增益、偏置）：只有一个方向，没有「强度」可拉平　——　原理上用不了",
        INK, size=13.5)
    f.t(30, py2 + 172, "嵌入表和输出头也是矩阵：原理上能用，作者实测交给 Adam 效果更好　——　这是实测排除",
        INK, size=13.5)
    f._pan = None

    yb = f.band(py2 + PH2 + 22, "ok", "带走一句：Muon ＝ 动量 ＋ 一道「把每个频率的强度拉到差不多」", [
        "前面三站跟带动量的 SGD 一模一样；它只比 Adam 少记一份东西（没有二阶矩），多做五轮矩阵乘。",
    ])
    yb = f.src(yb + 10,
               "出处：每一站都对着公开实现 github.com/KellerJordan/Muon 的 muon.py 写："
               "momentum.lerp_(grad, 1−β)（β 默认 0.95）→ Nesterov 时 grad.lerp_(momentum, β) "
               "→ X ÷ ‖X‖（Frobenius 范数）→ 五轮 X ← aX ＋ (bA ＋ cA²)X，A ＝ XXᵀ，"
               "(a, b, c) ＝ (3.4445, −4.7750, 2.0315) → 乘 max(1, 行/列)^0.5 → p.add_(update, alpha＝−lr)。",
               "出处：「嵌入、输出头、增益、偏置用 AdamW」：同一文件的说明与作者博客；"
               "博客写明输入输出层用 AdamW 是实测结论，不是理论推出来的。",
               "注：强度条的数来自 topic04_muon_nums.py（现算并断言）。实现里还会把高的矩阵先转置再迭代、"
               "用 bfloat16 算 —— 这两样不改变上面的讲法，没画。形状系数各家写法不同"
               "（例如有的把整体大小对齐 Adam 的 0.2），这里画的是公开实现的写法。")
    f.save("fig4-muonstep.svg", yb + 14)


main()
