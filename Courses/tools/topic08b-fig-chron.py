# -*- coding: utf-8 -*-
r"""专题九 · 优化编年史的两张图（2026-11-22 第二轮）。

fig8b-chron   回答：从 12.7 秒到 4.1 秒，一路走了哪几站？—— 三段底座各画一块（20 层底座 → 半层底座短文 → 半层底座长文），
              每根柱一站，柱顶是步时；段与段之间画衔接点：同一配置在新旧底座上各量一次，所以高低不能跨段比。
fig8b-front   回答：前半程（20 层底座）每一站砍掉多少、砍在哪类瓶颈上？—— 站 0–12 的步时横条，颜色 ＝ 这一刀打在哪类瓶颈上。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400
KCOL = {"bytes": RD, "scq": OR, "sched": BL, "data": GY2, "num": GR}
KLAB = [(RD, "通信字节"), (OR, "小核队列"), (BL, "排法、kernel 与优化器"), (GY2, "数据（分子变了）"), (GR, "精度修正（不为提速）")]


def fig_chron():
    segs = [("第一段 · 20 层底座", [(s, t, k) for s, _n, t, _m, _p, k in N.FRONT], None),
            ("第二段 · 半层底座，短文", [("起", N.HALF_START, "start")] + [(s, t, k) for s, _n, t, _m, _p, k in N.HALF], None),
            ("第三段 · 半层底座，长文", [("起", N.LONG_START, "start")] + [(s, t, k) for s, _n, t, _m, _p, k in N.LONG], None)]
    aria = ("优化编年史，三段底座。第一段 20 层底座：站 0 原样 12.743 秒，到站 12 2.731 秒。衔接一：同一配置放到半层底座是 5.476 秒。"
            "第二段半层底座短文：站 13 5.311 秒，到站 16 4.049 秒。衔接二：站 13 那套配置换长文是 5.553 秒。"
            "第三段半层底座长文：站 17 4.804 秒，到站 24 4.094 秒；站 25 到 27 三站精度修正，步时持平、+0.2%、+0.1%，终点 4.116 秒，有效 MFU 24.18%。三段的纵轴各自独立，段与段之间不能比高低。")
    f = Fig(W, aria)
    y0 = f.header("优化编年史：三段底座，28 站，12.743 秒 → 4.116 秒",
                  "每根柱一站，柱高 ＝ 步时（秒）；颜色 ＝ 这一刀打在哪类瓶颈上；灰框 ＝ 换底座以后的起点。三段纵轴各自独立，不能跨段比高低",
                  KLAB + [(LINE2, "换底座／换数据后的起点")])
    PH = 420
    widths = [530, 290, 510]
    gap = 35
    x = 0
    base_y = y0 + 30 + 330
    for si, (title, bars, _x) in enumerate(segs):
        w = widths[si]
        py = f.panel(x, y0, w, PH, title, (RD, BL, OR)[si])
        top = max(t for _s, t, _k in bars)
        sc = 270.0 / top
        bw = min(34, (w - 40) / len(bars) - 8)
        step = (w - 30) / len(bars)
        for j, (stn, t, k) in enumerate(bars):
            bx = x + 18 + j * step
            h = t * sc
            if k == "start":
                f.box(bx, base_y - h, bw, h, "#fff", GY2, 3, sw=1.4, dash="4 3")
            else:
                col = KCOL[k]
                f.box(bx, base_y - h, bw, h, col, col, 3)
            f.t(bx + bw / 2, base_y - h - 8, "%.3f" % t if t < 10 else "%.2f" % t, INK, True, 11.5, "middle")
            f.t(bx + bw / 2, base_y + 20, str(stn), GY, True, 13, "middle")
        f.t(x + 18, base_y + 48, "站号", GY, size=12)
        x += w + gap
    # 衔接点说明
    f.t(widths[0] + 4, y0 + 200, "衔接①", GY, True, 13)
    f.t(widths[0] + gap + widths[1] + 4, y0 + 200, "衔接②", GY, True, 13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "每段里一站一站往下抠；跨段的高低不能比：衔接①同一配置 2.731 → 5.476 秒，衔接②同一配置 5.311 → 5.553 秒", [])
    yb = f.src(yb + 10,
               "实测（作者私有编年史总表，64 片 v7x）：「对上一步」优先用同一作业里的对照；跨作业的在正文里写明。前 7 站是第 20–35 步均值，从站 7 起第 5–9 步；半层底座与长文的首格取第 13–20 步。",
               "衔接①：同一份推荐配置，20 层底座 %.3f 秒、有效 MFU %.2f%%，半层底座（39 层主干 ＋ 视觉塔 20 层 ＋ MTP 头，每个设备的负担跟 128 片全量一样）%.3f 秒、%.2f%%。" % N.SW1,
               "衔接②：站 13 那套配置，短文 %.3f 秒、%.2f%%，长文 %.3f 秒、%.2f%%：长文每行一篇长文，有效分子从 %.1f 变成 %.1f TFLOP，有效 MFU 高了，步时反而长。" % (N.SW2 + (N.EFF_SHORT, N.EFF_LONG)),
               "站 10、13 是换数据：步时快了一点，有效 MFU 跳一档主要因为分子变了。")
    f.save("fig8b-chron.svg", yb + 14)


def fig_front():
    rows = N.FRONT
    aria = "前半程 20 层底座，十三站的步时：" + "；".join("站 %d %s %.3f 秒%s" % (s, n, t, ("，%+.1f%%" % p) if p is not None else "") for s, n, t, _m, p, _k in rows) + "。"
    f = Fig(W, aria)
    y0 = f.header("前半程：20 层底座，13 站，12.743 → 2.731 秒",
                  "条长 ＝ 步时；颜色 ＝ 这一刀打在哪类瓶颈上；右边是这一站对上一站快了多少",
                  KLAB)
    PH = 50 + 40 * len(rows)
    py = f.panel(0, y0, W, PH, "64 片 v7x，20 层底座（实测）", BL)
    X0, SC = 470, 50.0
    for i, (s, n, t, m, p, k) in enumerate(rows):
        yy = py + 20 + i * 40
        f.t(X0 - 14, yy + 19, "站 %d　%s" % (s, n), INK, True, 14, "end")
        col = KCOL[k]                      # 站 0 也按它的类涂（跟编年史总图、动画一致）
        f.box(X0, yy + 4, t * SC, 24, col, col, 3)
        lab = "%.3f 秒" % t + ("　%+.1f%%" % p if p is not None else "　有效 MFU %.2f%%" % m)
        f.t(X0 + t * SC + 10, yy + 22, lab, col, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "瓶颈一路换类：先是字节，再是排法和小核队列；每砍掉一类，下一类就露出来", [])
    yb = f.src(yb + 10,
               "实测：站 0 有效 MFU %.2f%%，站 12 %.2f%%（站 10 换了预打包数据，分子从 325.18 变成 386.56 TFLOP，有效 MFU 从这一站起跳一档）。" % (N.FRONT_BASE_MFU, N.FRONT_END_MFU),
               "站 2 合了两步：小核分工、token 分两块、先算 top-k（单测 +0.3%，在噪声内，站 5 时去掉）和搬来已上线的注意力配置 ＋ 层调度器（−29.0%）。前 7 站取第 20–35 步，从站 7 起第 5–9 步。站 0 的有效 MFU 2.30% 用这几步自己的有效分子（本课推导：2.30% × 12.743 × 1155 ≈ 338 TFLOP，跟第二节第 5–9 步的 325.18 不是同一批步）。")
    f.save("fig8b-front.svg", yb + 14)


if __name__ == "__main__":
    fig_chron()
    fig_front()
