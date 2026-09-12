# -*- coding: utf-8 -*-
r"""专题三 · §十「那怎么克服 —— TPU 上的三招 ＋ 两个常被问到的问题」
（2026-09-13 · TPU 轮 R28–R31）。

⭐⭐⭐ 现场点的三个具体问题，这一张逐个回答：

  ① **不连续的 gather 把 DMA 调度搞乱了，怎么办？**
     ——&nbsp;RPA 论文给的三招，每一招都在把「动态」换成「一批静态」。

  ② **运行时才决定读哪 2048 条，这个决定的成本是什么？
     能不能完全在卡里算？要不要发往 host CPU？**
     ⭐⭐ 答案很漂亮：**能在卡上算，而且用的正是本来闲着的标量单元。**
     RPA 论文的原话是：FlashAttention 计算密集阶段 **SREG 是欠用的**，
     而动态 DMA 地址与大小的元数据计算**需要大量标量计算、却让向量寄存器闲着** ——
     所以他们把元数据**预计算后放进 SMEM**，让标量和向量执行重叠，把延迟藏掉。
     ⛔ 要分清两层：**批次级**的页表/序列长度是 host 给的；
     **kernel 内**「这一步搬哪几块」的地址计算在卡上做。

  ③ **SparseCore 能不能帮？**
     架构上**正对口**（原生支持数据相关的控制流与访存，还自带跨 lane 的
     排序 / 过滤 / 前缀和 ——&nbsp;正是 top-k 要的）。
     ⚠️ 但它消化动态性的方式是**先声明静态上界**；而且**目前公开的生产级
     TPU attention kernel 走的是 TensorCore ＋ Pallas/Mosaic，不是 SparseCore**。

  ④ 外加一条：**跨层共享 top-k（IndexShare / IndexCache）在 TPU 上比在 GPU 上更值。**
     ⚠️ 这一条是本课的推导。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "TPU 上怎么克服：RPA 的三招都是把动态换成一批静态；"
               "运行时决定的成本可以完全在卡上算，用的是本来闲着的标量单元；"
               "SparseCore 架构上对口但要先声明静态上界")
    f.marks = set()
    y0 = f.header(
        "那怎么克服　——　三招，以及两个一定会被问到的问题",
        "⭐⭐ 三招的共同形状：<tspan font-weight=\"700\">把「一个动态」换成「一批静态」</tspan>",
        [(GR, "RPA 的三招"), (BL, "在卡上算"), (PU, "SparseCore"),
         (OR, "本课的推导")])

    ph = 506

    # ══ ① 三招 ══════════════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 三招：把动态换成一批静态", GR,
                 sub="RPA 论文的三个创新")

    yy = py + 24
    for i, (head, body, key) in enumerate([
        ("细粒度 tiling",
         "强制 XLA 选最小的 tile，并且把 ragged 维度挪开",
         "⭐ 不要让「长度」落在最后两维的 tiling 维上 ——&#160;"
         "那样才切得动"),
        ("把 KV 更新融进 attention",
         "decode 时那个单 token 粒度的 scatter，"
         "原本要在 TensorCore 上单独做一遍",
         "⭐ 融进去之后，用计算把写的延迟盖住"),
        ("分布感知编译",
         "按序列长度分布，编出好几个特化 kernel"
         "（decode / prefill / 混合）",
         "⭐⭐ 这一招最像 TPU 的风格：不写一个动态 kernel，写一批静态的再挑"),
    ]):
        h = 122
        f.box(x + 22, yy, pw - 44, h, "#fff", GR, 8)
        f.box(x + 22, yy, 4, h, GR, GR, 2)
        f.box(x + 24, yy, 3, h, "#fff", "#fff", 0)
        f.t(x + 40, yy + 26, "%d. %s" % (i + 1, head), GR, True, 12.5)
        yy2 = f.lines(x + 40, yy + 50, pw - 76,
                      _wrap(body, pw - 76), 11.5, 19)
        f.lines(x + 40, yy2 + 6, pw - 76, _wrap(key, pw - 76), 11, 17,
                fill=GY2)
        yy += h + 8

    f.box(x + 22, yy, pw - 44, 56, "#fff", INK, 8)
    f.t(x + 38, yy + 24, "⭐ 成绩：Llama 3 8B 在 TPU7x 上", INK, True, 12.5)
    f.t(x + 38, yy + 45, "decode <tspan font-weight=\"700\">MBU 86%</tspan> · prefill <tspan font-weight=\"700\">MFU 73%</tspan>", GY,
        size=11.5)
    fits(yy + 56, y0, ph, "①")

    # ══ ② 运行时决定的成本 ══════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 运行时那个决定，谁来算", BL,
                 sub="⭐ 答案：卡上算，而且用闲着的那部分")

    yy = py + 24
    f.box(x + 22, yy, pw - 44, 104, "#fff", BL, 8)
    f.box(x + 22, yy, 4, 104, BL, BL, 2)
    f.box(x + 24, yy, 3, 104, "#fff", "#fff", 0)
    f.t(x + 40, yy + 26, "TensorCore 里有两套寄存器", BL, True, 12.5)
    f.t(x + 40, yy + 50, "<tspan font-weight=\"700\">VREG</tspan>（向量）——&#160;矩阵乘的时候忙得冒烟", GY,
        size=11.5, w=pw - 76)
    f.t(x + 40, yy + 72, "<tspan font-weight=\"700\">SREG</tspan>（标量）——&#160;同一时刻<tspan font-weight=\"700\">基本闲着</tspan>", GY,
        size=11.5, w=pw - 76)
    f.t(x + 40, yy + 94, "（RPA 论文原话：SREG 在计算密集阶段欠用）", GY2,
        size=11)
    yy += 118

    f.box(x + 22, yy, pw - 44, 96, "#fff", GR, 8)
    f.box(x + 22, yy, 4, 96, GR, GR, 2)
    f.box(x + 24, yy, 3, 96, "#fff", "#fff", 0)
    f.t(x + 40, yy + 26, "⭐⭐ 而「这一步搬哪几块」的地址计算", GR, True, 12.5)
    f.t(x + 40, yy + 50, "恰好是<tspan font-weight=\"700\">纯标量</tspan>的活儿。", GY, size=11.5)
    f.t(x + 40, yy + 72, "→ 预计算元数据放进 SMEM，<tspan font-weight=\"700\">标量和向量重叠跑</tspan>",
        GR, True, 12, w=pw - 76)
    yy += 110

    f.t(x + 22, yy, "⛔ 但要分清两层，别混：", INK, True, 12.5)
    yy += 22
    for who, what, col in [
        ("host CPU 给的", "页表、每条序列多长、这一批怎么排 —— <tspan font-weight=\"700\">批次级</tspan>", GY),
        ("卡上标量核算的", "这一步的 DMA 地址和大小 —— <tspan font-weight=\"700\">kernel 内</tspan>", BL),
    ]:
        f.box(x + 22, yy, pw - 44, 52, "#fff", col if col != GY else LINE, 8)
        f.t(x + 38, yy + 22, who, col, True, 12)
        f.t(x + 38, yy + 41, what, GY, size=11, w=pw - 76)
        yy += 58
    f.t(x + 22, yy + 2, "⭐ 所以 top-k 的「决定」<tspan font-weight=\"700\">不用出卡</tspan>。", INK,
        True, 12.5, w=pw - 44)
    fits(yy + 10, y0, ph, "②")

    # ══ ③ SparseCore 能不能帮 ═══════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ SparseCore 能不能帮", PU,
                 sub="⭐ 架构上正对口，但有前提")

    yy = py + 24
    f.box(x + 22, yy, pw - 44, 116, "#fff", GR, 8)
    f.box(x + 22, yy, 4, 116, GR, GR, 2)
    f.box(x + 24, yy, 3, 116, "#fff", "#fff", 0)
    f.t(x + 40, yy + 26, "✓ 它就是为「不规则访存」造的", GR, True, 12.5)
    f.t(x + 40, yy + 50, "<tspan font-weight=\"700\">原生支持数据相关的控制流与访存</tspan>", GY, size=11.5)
    f.t(x + 40, yy + 72, "而且自带跨 lane 的<tspan font-weight=\"700\">排序 / 过滤 / 前缀和</tspan>", GY,
        size=11.5)
    f.t(x + 40, yy + 94, "⭐ 那正是 top-k 要的三样东西", GR, size=11.5)
    yy += 130

    f.box(x + 22, yy, pw - 44, 96, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 96, OR, OR, 2)
    f.box(x + 24, yy, 3, 96, "#fff", "#fff", 0)
    f.t(x + 40, yy + 26, "⚠️ 但它消化动态性的方式是", OR, True, 12.5)
    f.t(x + 40, yy + 50, "<tspan font-weight=\"700\">先声明一个静态上界</tspan>（每分区最多几个 id）", GY,
        size=11.5, w=pw - 76)
    f.t(x + 40, yy + 72, "超了就 mini-batch，或者<tspan font-weight=\"700\">丢 id</tspan>", GY, size=11.5)
    yy += 110

    f.box(x + 22, yy, pw - 44, 96, "#fff", INK, 8)
    f.t(x + 38, yy + 26, "⭐ 对 DSA 来说这个前提<tspan font-weight=\"700\">天然满足</tspan>", INK,
        True, 12.5)
    f.t(x + 38, yy + 50, "——&#160;k 本来就是固定的 2048。", GY, size=11.5)
    f.t(x + 38, yy + 74, "⛔ 但公开的生产 kernel 走的<tspan font-weight=\"700\">不是</tspan>这条路（见下）",
        RD, size=11.5, w=pw - 76)
    fits(yy + 96, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "warn", "⚠️ 关于 SparseCore，必须把话说完 —— 否则这一格会变成一个误导", [
        "以上说的是<tspan font-weight=\"700\">架构上对不对口</tspan>，不是「已经有人这么做了」。"
        "<tspan font-weight=\"700\">目前公开可查的生产级 TPU attention kernel（RPA）"
        "走的是 TensorCore ＋ Pallas/Mosaic 那条路</tspan>，不是 SparseCore。",
        "⭐ 而且 SparseCore 一直以来的编程入口是 embedding 那套算子；"
        "要拿它做 attention 的 KV 收集，得走 <tspan font-weight=\"700\">Pallas 的 SparseCore 后端</tspan>。",
        "⛔ 所以正确的说法是：<tspan font-weight=\"700\">这是一个「看起来很对但还没被公开验证」的方向</tspan>"
        "——&#160;讲的时候就这么讲，别讲成既成事实。",
    ])

    yy = f.band(yy + 14, "info", "⭐⭐ 跨层共享 top-k，在 TPU 上比在 GPU 上更值（⚠️ 本课推导）", [
        "GPU 上，四层共享一个索引器省的主要是<tspan font-weight=\"700\">索引器自己的 FLOPs</tspan>"
        "（GLM-5.2 报的 1M 下每 token 降 2.9×）。",
        "⭐ 但在 TPU 上它还额外省掉三样："
        "<tspan font-weight=\"700\">① 三次「动态元数据计算 ＋ 不规则 DMA 调度」的固定开销</tspan>；"
        "<tspan font-weight=\"700\">② 后三层的 gather 模式完全相同</tspan>，"
        "同一套 DMA 描述符和 tiling 决策可以直接复用；",
        "<tspan font-weight=\"700\">③ 动态性的「次数」少了四倍</tspan> ——&#160;"
        "而在一台 static-first 的机器上，<tspan font-weight=\"700\">动态性的次数本身就是成本</tspan>。"
        "⚠️ 这三条是按机制推的，<tspan font-weight=\"700\">没有实测</tspan>。",
    ])

    yy = f.src(yy + 16,
               "①② 出自 Ragged Paged Attention（Jiang 等，arXiv 2604.15464，2026-04）"
               "§1 与 §5：三招、SREG 欠用与元数据预计算进 SMEM、MBU 86% / MFU 73%",
               "③ 出自 OpenXLA 的 SparseCore 文档（openxla.org/xla/sparsecore）："
               "「为不规则稀疏访存加速的专用 tiled 处理器」「原生支持数据相关的控制流与访存」"
               "「跨 lane 的排序 / 过滤 / 前缀和」，以及 max_ids_per_partition 这类静态上界",
               "⚠️ 最后那条「跨层共享在 TPU 上更值」是本课按机制做的推导，不是任何一篇的结论")
    f.save("fig3-tpu-fix.svg", yy + 6)


def _wrap(t, w, size=11.5):
    """按像素宽度折行。

    ⛔ 这个小工具连翻了两次车，两次都值得记：
      ① 第一版**只在标点处断**，碰上「一长串没标点的短语」就断不动 ——
         于是 t() 的宽度断言当场报错。⭐ 折行器必须有兜底的硬断点。
      ② 第二版补了硬断点，但它是**先把字加进去、再判断超没超**，
         所以每一行都恰好超出一个字。⭐ 判据：**要判的是「加上这个字会不会超」，
         不是「加完了超没超」** ——&nbsp;这类 off-by-one 在断言里表现为
         「只差 7px」，看起来像阈值调小一点就好，其实是判断点放错了位置。
    ⛔ 还有一条：**断点不能落在 `<tag>` 里面**，否则标签被劈成两半。
    """
    import re
    # ⛔ 折行会在任意位置断开，所以**传进来的必须是纯文本** ——
    #   一个 <tspan> 被劈成两半，t() 的配平断言才会报，而那时已经离源头很远了。
    assert "<" not in t, "_wrap 的输入不能带标签（会被折断）：%s" % t[:40]
    lim = w - 12
    out, cur, depth = [], "", 0

    def vis(x):
        return wpx(re.sub(r"<[^>]+>", "", x), size)

    for ch in t:
        if ch == "<":
            depth += 1
        # 先问「加上它会不会超」——&nbsp;超了就先把手里这行交出去
        if depth == 0 and cur and vis(cur + ch) > lim:
            out.append(cur)
            cur = ""
        cur += ch
        if ch == ">":
            depth = max(0, depth - 1)
        # 标点处优先断（只在已经写了大半行时才断，免得断出一堆碎行）
        if depth == 0 and ch in "，。；、" and vis(cur) > lim * 0.55:
            out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur)
    return out or [t]


main()
