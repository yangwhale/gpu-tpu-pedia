# -*- coding: utf-8 -*-
r"""专题六 · 第十四节开头的「六道关」爬坡图。

回答的问题
  · fig6-climb  把 Qwen3.5-397B 在一台 v7x-8 上跑起来，要按什么顺序过哪几关？每关撞的墙和修法是什么？K3 在哪几关走了另一条路？

数从哪来
  · 3.5 倍、60–90 s、约 7 min、5 处提 FP32、约 349 MB、0.95 倍：tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md
    与课件正文同出处（见出处台账 L98、8.5 的台账行）。
  · K3 三面旗：DEP8 加载 OOM 与「KDA 不能按序列切」取自 SGLang 的 K3 部署手册；megakernel 取自 Inferact 公开博客。

刻意没画
  · 不画任何吞吐数：梯子那张图专管 prefill 的指数，这张只管顺序和因果。
  · 台阶高度只表示先后，不表示难度或收益。
"""
from topic03_draw import Fig, wpx, BL, OR, GR, RD, GY, INK, GY2, LINE

W = 1400
# (关名, 墙的几行, 修法的几行, 是否主角关)
GATES = [
    ("装得下", ["2 个 KV 头，", "8 路 TP 切不开"], ["注意力按请求切，", "MoE 按专家切"], False),
    ("起得来", ["没编过的档，", "首个请求等 60–90 s"], ["开服前预编译；", "分清编译和搬字节"], False),
    ("跑得对", ["块池大 3.5 倍，", "越界被静默钳回；", "bf16 递推死循环"], ["每层各存一份池；", "5 处提到 FP32"], True),
    ("跑得快", ["MoE 占 prefill 八成，", "通信露在外面"], ["一层 MoE 融成", "一个 Pallas 程序"], False),
    ("拆得开", ["每个请求整份", "要搬约 349 MB"], ["单机装得下就别拆：", "单并发只有 0.95 倍"], False),
    ("扛得住", ["注册表在线 ≠ 活着；", "前缀被副本摊薄"], ["真发请求判活；", "KV 感知路由"], False),
]
# 挂在第几关（0 起）上的 K3 旗
K3 = {0: ["DEP8 每个 rank 存整池状态，", "加载即 OOM，退回 TP"],
      2: ["手册写明 KDA 不能", "按序列切，切了数全错"],
      3: ["batch 只有 1–8：", "整个模型一次 Pallas 调用"]}
assert len(GATES) == 6 and sorted(K3) == [0, 2, 3]


def fig_climb():
    f = Fig(W, "六级台阶从左往右升高，依次是装得下、起得来、跑得对、跑得快、拆得开、扛得住，台阶高度只表示先后。"
               "每级写着撞上的墙（红）和修法（绿）。第三级跑得对用红框突出：块编号池大了 3.5 倍被静默钳回、bf16 递推死循环，两个毛病都不报错。"
               "三面橙色旗挂在装得下、跑得对、跑得快上方，是 Kimi K3 走的另一条路：DEP8 加载即 OOM；KDA 不能按序列切；batch 1 到 8 时整个模型写成一次 Pallas 调用")
    y0 = f.header("落地六道关　——　<tspan font-weight=\"700\">最难的一关不报错</tspan>",
                  "Qwen3.5-397B（FP8）在一台 v7x-8 上从零到上线。每级台阶 ＝ 撞上的墙 → 修法；台阶高度只表示先后",
                  [(RD, "墙"), (GR, "修法"), (OR, "Kimi K3 的另一条路")])
    PH = 500
    py = f.panel(0, y0, W, PH, "按顺序过关", BL)
    base = py + PH - 60
    gap, sw = 12, (W - 40 - 5 * 12) / 6
    H0, DH = 196, 36
    for i, (nm, wall, fix, star) in enumerate(GATES):
        x = 20 + i * (sw + gap)
        h = H0 + i * DH
        top = base - h
        col = RD if star else BL
        f.box(x, top, sw, h, "#fff", col, 6, sw=2 if star else 1.4)
        f.box(x, top, sw, 8, col, col, 3)
        f.t(x + 16, top + 40, "%d" % (i + 1), col, True, 20)
        f.t(x + 44, top + 40, nm, INK, True, 20)
        yy = top + 72
        f.t(x + 16, yy, "墙", RD, True, 14)
        for k, ln in enumerate(wall):
            assert wpx(ln, 14) < sw - 50, (ln, wpx(ln, 14), sw)
            f.t(x + 40, yy + k * 21, ln, INK, size=14)
        yy += 21 * len(wall) + 14
        f.t(x + 16, yy, "修", GR, True, 14)
        for k, ln in enumerate(fix):
            assert wpx(ln, 14) < sw - 50, (ln, wpx(ln, 14), sw)
            f.t(x + 40, yy + k * 21, ln, INK, size=14)
        if i in K3:
            fh = 80
            fy = top - 22 - fh
            assert fy > py + 6, (i, fy, py)
            f.line(x + 24, fy + fh, x + 24, top, OR, 1.6, arrow=False)
            f.box(x + 8, fy, sw - 8, fh, "#fff", OR, 6, sw=1.4)
            f.t(x + 20, fy + 24, "K3", OR, True, 14)
            for k, ln in enumerate(K3[i]):
                assert wpx(ln, 13) < sw - 30, (ln, wpx(ln, 13), sw)
                f.t(x + 20, fy + 46 + k * 19, ln, INK, size=13)
    f.line(20, base + 18, W - 20, base + 18, GY2, 1.6)
    f._pan = None
    yb = f.src(py + PH + 10,
               "📌 Qwen3.5 各关的墙与修法取自 tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md（块池约 3.5 倍、5 处提 FP32、冷启动约 7 min、PD 每请求约 349 MB、单并发 0.95 倍）；"
               "60–90 s 与「MoE 占 prefill 约八成」同正文出处。台阶高度是示意，不代表难度或收益。",
               "K3 三面旗：DEP8 加载 OOM、69 层 KDA 不能按序列切取自 SGLang 的 K3 部署手册；megakernel 取自 inferact.ai/blog/tpu-megakernels。")
    f.save("fig6-climb.svg", yb + 14)


if __name__ == "__main__":
    fig_climb()
