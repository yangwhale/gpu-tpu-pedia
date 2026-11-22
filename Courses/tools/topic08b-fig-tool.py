# -*- coding: utf-8 -*-
r"""专题九 · 第七、八节的图。

fig8b-bisect  回答：同一颗核，换了运行库版本为什么慢三倍？—— 离线单独编这颗核，数热循环每趟几个指令包；
              0.0.46 对 0.0.53.1：41 → 198；代码里展开 8 次，在新版本上编回旧水平。
fig8b-vmem    回答：放大片上暂存上限快了 2.3%，快在哪？—— 按类拆账：本想帮的注意力核反而 +9 ms，快的是矩阵乘融合和同步集合。
fig8b-flip    回答：换成长文数据，原来的几刀怎么变？—— 三把刀在短文、长文上各自对同作业对照的步时变化。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def fig_bisect():
    rows = [("0.0.46（旧版本）", N.BUNDLE_OLD, GY2), ("0.0.53.1（新版本）", N.BUNDLE_NEW, RD)]
    f = Fig(W, "同一颗小核上的核，用两个版本的运行库离线单独编译，数热循环每跑一趟要几个指令包。"
               "0.0.46 是 41 个，0.0.53.1 是 198 个，长了约五倍。代码里给这个循环展开 8 次以后，在 0.0.53.1 上编回了旧版本的水平。")
    y0 = f.header("同一颗核，换个运行库版本，热循环长了约五倍",
                  "条长 ＝ 热循环每跑一趟要几个指令包（越短越好）；离线单独编这一颗核",
                  [(GY2, "旧版本"), (RD, "新版本"), (GR, "代码里兜住以后")])
    PH = 60 + 52 * len(rows) + 50
    py = f.panel(0, y0, W, PH, "离线编译产物（实测）", BL)
    X0, SC = 360, 4.0
    for i, (v, x, col) in enumerate(rows):
        yy = py + 40 + i * 52
        f.t(X0 - 14, yy + 22, v, INK, True, 15, "end")
        f.box(X0, yy + 4, x * SC, 28, col, col, 3)
        f.t(X0 + x * SC + 10, yy + 23, "%d 个指令包／趟" % x, col, True, 15)
    yy = py + 40 + len(rows) * 52
    f.t(X0 - 14, yy + 22, "0.0.53.1 ＋ 代码里展开 8 次", INK, True, 15, "end")
    f.t(X0, yy + 23, "每次向量乘摊到的指令包 12.6 → 2.5，回到旧版本水平（约 2.6）", GR, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "运行库也是变量：升级以后先普查，看见的那处在代码里兜住", [])
    yb = f.src(yb + 10,
               "实测（离线单核对比）：0.0.46 对 0.0.53.1，热循环每趟 %d → %d 个指令包；新版本不再对这个循环做软件流水，循环里不变的量也不再提到循环外（从指令排布判断）。" % (N.BUNDLE_OLD, N.BUNDLE_NEW),
               "实测（profile）：这颗核每次执行 %.2f → %.2f ms；主核等它 %.1f → %.1f ms；同一份配方一步 %.1f → %.1f ms。" % (N.SCK_PER_CALL + N.SCK_TCWAIT + N.NEWLIB_DEV),
               "实测（离线单核编译）：这是 SparseCore 上做 ragged gather-reduce 的核；代码里只改一行，给热循环展开 8 次，每个元素的计算不变。新版本上每次向量乘摊到的指令包 12.62 → 2.47，旧版本展开后 2.44、原样约 2.56。",
               "本讲的模型正常训练步不用这颗核（走主核上的排序），只有 token 溢出、整步回放的那一步会用到。")
    f.save("fig8b-bisect.svg", yb + 14)


def fig_vmem():
    rows = [(n, v) for n, v in N.VMEM_ROWS if v is not None] + [("其余各类合计", N.VMEM_OTHER)]
    tot = N.VMEM_STEP[1] - N.VMEM_STEP[0]
    f = Fig(W, "片上暂存上限从默认放到 64 MiB 以后，一步从 %.1f 到 %.1f 毫秒，少 %.1f。按类拆：注意力两颗核多 9.2，编译器自己的矩阵乘融合少 67.2，"
               "优化器同步集合的等待少 38.7，MoE 同步集合的等待少 30.4，其余各类合计多 %.1f。" % (N.VMEM_STEP + (-tot, N.VMEM_OTHER)))
    y0 = f.header("放大片上暂存上限：本想帮的注意力核没快，快的是别处",
                  "条 ＝ 这一类在一步里少了（左，绿）或多了（右，红）多少毫秒；同一颗芯片、同一步的 profile 前后相减",
                  [(GR, "少了"), (RD, "多了")])
    PH = 70 + 52 * len(rows)
    py = f.panel(0, y0, W, PH, "整步 %.1f → %.1f ms（实测）" % N.VMEM_STEP, BL)
    XC, SC = 820, 5.0
    f.line(XC, py + 36, XC, py + PH - 14, GY2, 1.2, arrow=False)
    for i, (n, v) in enumerate(rows):
        yy = py + 42 + i * 52
        col = RD if v > 0 else GR
        x0 = XC if v > 0 else XC + v * SC
        f.box(x0, yy + 4, abs(v) * SC, 28, col, col, 3)
        f.t(XC - 440, yy + 23, n, INK, True, 15, "end")
        f.t((XC + v * SC + 10) if v > 0 else (XC + v * SC - 10), yy + 23, "%+.1f" % v, col, True, 15, "start" if v > 0 else "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "全局开关的收益，要按类拆账才知道落在哪", [])
    yb = f.src(yb + 10,
               "实测（profile）：长文数据，同一作业里对照和放大暂存上限两格，同一颗芯片的第 11 步，按类别前后相减。整步实测快 2.3%%（%.3f → %.3f 秒）。" % N.FLIP[2][2],
               "推断：按离线指令数模型，注意力核本身只省约 0.25%；暂存大了以后，编译器的矩阵乘融合能用更大的分块（没读底层指令，未证实）。")
    f.save("fig8b-vmem.svg", yb + 14)


def fig_flip():
    f = Fig(W, "三把刀，各自在短文数据和长文数据上对同作业对照的步时变化。融合核：短文慢 2.0%，长文快 1.1%。"
               "token 和专家权重分到两颗小核：短文快 3.1%，长文只快 0.7%。片上暂存上限放到 64 MiB：短文快 0.4%，长文快 2.3%。" )
    y0 = f.header("换一份数据，原来的几刀重新排名",
                  "条 ＝ 这一刀让步时变化了百分之几（左边绿＝变快，右边红＝变慢）；每把刀两行：上短文、下长文，各自对同作业对照（长文分两颗小核那格是跨作业）",
                  [(GR, "变快"), (RD, "变慢")])
    rows = N.FLIP_PCT
    PH = 60 + 120 * len(rows)
    py = f.panel(0, y0, W, PH, "步时变化（实测）", BL)
    XC, SC = 860, 90.0
    f.line(XC, py + 36, XC, py + PH - 14, GY2, 1.2, arrow=False)
    for i, (n, a, b) in enumerate(rows):
        yy = py + 44 + i * 120
        f.t(40, yy + 40, n, INK, True, 16)
        for k, (lab, v) in enumerate([("短文", a), ("长文", b)]):
            y2 = yy + k * 46
            col = RD if v > 0 else GR
            x0 = XC if v > 0 else XC + v * SC
            f.t(XC - 380, y2 + 22, lab, GY, True, 14, "end")
            f.box(x0, y2 + 4, abs(v) * SC, 26, col, col, 3)
            f.t((XC + v * SC + 10) if v > 0 else (XC + v * SC - 10), y2 + 22, "%+.1f%%" % v, col, True, 15, "start" if v > 0 else "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "瓶颈换了地方：无效的翻正，有效的变小。换了数据，每一刀都要重测", [])
    yb = f.src(yb + 10,
               "实测：每格对它自己作业里的对照，第 5–9 步均值；短文那一列来自前几节的作业，长文来自长文数据上的作业；「分两颗小核」短文用第六节同作业的单测（4.415 → 4.280），长文那一格的对照来自另一个作业，只看方向。短文上的暂存上限那一格只有百分比读数（−0.4%）。",
               "长文数据：八成多的行是一整篇长文。同一配置换成长文，有效 MFU %.2f%% → %.2f%%，步时 %.3f → %.3f 秒。另在叠了前几节几刀的配置上比，长文比短文多出的 %d 毫秒里 %d 毫秒落在注意力两颗核上。" % (N.LONG_CTL[1], N.LONG_CTL[3], N.LONG_CTL[0], N.LONG_CTL[2], N.LONG_EXTRA_MS, N.LONG_EXTRA_ATTN_MS),
               "推断：长文数据的路由更不均（最差层负载比到 2.3），MoE 那段也跟着变，这是长文结论最大的不确定性。")
    f.save("fig8b-flip.svg", yb + 14)


def fig_a2a():
    """fig8b-a2a  回答：版本一换，同一份程序里的那批 all-to-all 放哪了、付了什么？—— 四行：放在主核还是小核、代价。"""
    rows = [("旧版本运行库（默认）", "sc", "—", GY),
            ("新版本运行库（默认）", "tc", "主核同步跑，每步约 %d 毫秒" % N.A2A_MS, RD),
            ("新版本，强行交给小核", "sc", "峰值显存多 %.1f GiB，放不下" % N.A2A_FORCE_GIB, RD),
            ("新版本，强行交给小核 ＋ 优化器前一道屏障", "sc", "峰值反而少约 %.0f GiB；步时只多兑现 %d 毫秒" % (N.A2A_BAR_SAVED_GIB, N.A2A_SAVED_MS), GR)]
    f = Fig(W, "同一份程序里的 %d 条 all-to-all。旧版本运行库默认交给小核；新版本默认留在主核上同步跑，每步约 %d 毫秒；"
               "强行交给小核，峰值显存多 %.1f GiB，放不下；再在优化器前加一道屏障，峰值反而少约 %.0f GiB，步时只多兑现 %d 毫秒。"
               % (N.A2A_N, N.A2A_MS, N.A2A_FORCE_GIB, N.A2A_BAR_SAVED_GIB, N.A2A_SAVED_MS))
    y0 = f.header("配置一个字没改，版本一换：那批 all-to-all 换了地方",
                  "每行两条车道：上 ＝ 主核，下 ＝ 小核；紫块 ＝ 这批 all-to-all 放在哪；右边是代价",
                  [(PU, "那批 all-to-all"), (BL, "主核其余计算（示意）")])
    PH = 60 + 92 * len(rows)
    py = f.panel(0, y0, W, PH, "同一份程序，四种放法（实测）", BL)
    LX, LW = 440, 360
    for i, (lab, where, cost, col) in enumerate(rows):
        yy = py + 44 + i * 92
        f.t(LX - 16, yy + 30, lab, INK, True, 15, "end")
        f.box(LX, yy, LW, 24, "#e8f0fe", "#c6dafc", 3)
        f.box(LX, yy + 30, LW, 24, "#f1f3f4", GY2, 3)
        bx = LX + 220
        if where == "tc":
            f.box(bx, yy, 90, 24, PU, PU, 3)
        else:
            f.box(bx, yy + 30, 90, 24, PU, PU, 3)
        f.t(LX + LW + 24, yy + 34, cost, col, True, 15)
    f.t(LX + 6, py + 36, "主核", GY, size=12)
    f.t(LX + LW + 6, py + 44 + 48, "小核", GY, size=12)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "默认行为会随版本变：升级以后要普查放法，不能沿用旧版本上的判断", [])
    yb = f.src(yb + 10, "实测（离线编译产物逐条数）：大头是优化器给专家权重换排布的那几块；第四行的屏障就是第七节那一道。块的位置和长短是示意。")
    f.save("fig8b-a2a.svg", yb + 14)


if __name__ == "__main__":
    fig_bisect()
    fig_vmem()
    fig_flip()
    fig_a2a()
