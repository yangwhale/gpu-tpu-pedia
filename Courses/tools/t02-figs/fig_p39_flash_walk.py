# -*- coding: utf-8 -*-
"""图 P-39 —— 同一个 FlashAttention，两边并排走完全程，一站一站对。

**这是 3.6 的主图，也是整门课把「部件」和「路」缝在一起的地方。**
前面五小节拆的都是部件：核里有什么、指令吃多大、多出来的那一块挂在哪。
<b>这一节拆的是路</b> —— 而真实性能瓶颈大多数时候就在路上（P-38 那 128 GiB）。

**读法是横着读，不是竖着读。** 每一行是同一站，左右两栏分别是两边在这一站
放了什么。<b>最值钱的是第 ② 行和第 ⑥ 行</b>：
② GPU 有一整层硬件缓存，TPU 没有；
⑥ 矩阵单元和向量单元之间来回切换这件事，一边是每周期现挑，一边是编译期排死。
<b>FlashAttention 恰好是一个不停在这两种单元之间跳的 kernel</b>，
所以第 ⑥ 行不是背景知识，是它每一拍都要付的账。

**⚠️ 站数口径，这里统一一次。** 本课此前两处数法不一致 ——
图 T-6 沿 MXU 那条支线数，得五站（HBM／VMEM／向量寄存器／MXU／累加器）；
图 G-6 沿 VPU 那条主路数，得四站。<b>两个都对，数的是不同的路。</b>
这张图不再报总数，改成<b>按功能对齐</b>：谁在这一层、谁没有这一层。

**⚠️ 出处口径。** 容量与带宽沿用图 G-6／T-6 的标注，包括那里已经写明的
「第三方」「官方未公开」。<b>288 倍是本图当场算的</b>：64 MiB ÷ 227 KiB。
「FlashAttention 的 tile 住在共享内存那半边」是公开实现的通行做法。
"""
from common import Fig, para, BL, GN, RD, YL, PU, TL, INK, SUB, GREY, FILL

W = 1400

TB_Y = 84
HDR_H = 38
C0_X, C0_W = 20, 246
C1_X, C1_W = 276, 556
C2_X, C2_W = 842, 538

# (行高, 站名, 这一站对 FlashAttention 意味着什么, GPU 三元组, TPU 三元组)
# 三元组 ＝ (标题, 规格行, 说明)。TPU 的第 ② 行是整张图的重点，单独着色。
ROWS = [
    (90, "① 片外主存", "只有 Q / K / V / O 住这儿。<b>S 从来不出现在这一层 —— "
                        "省下的就是它。</b>",
     (BL, "HBM3e", "192 GB · 8.0 TB/s",
      "官方数字。<b>整颗芯片共用。</b>"),
     (GN, "HBM3e", "192 GiB · 7.4 TB/s / chip",
      "官方数字。<b>96 GiB / device</b>（v7 是 2 device / chip）。")),

    (104, "② 硬件缓存", "<r>这一行是全图最大的结构差异。</r>"
                        "有没有这一层，决定了写 kernel 的人在跟什么打交道。",
     (BL, "L2 缓存", "126 MB · 4 个分区",
      "<b>硬件自动，有命中率。</b>程序管不着它留什么、赶走什么。"),
     (RD, "没有这一站", "—",
      "<b>HBM 直接进片上暂存。</b><g>省下的不只是面积，"
      "还有「不知道会不会命中」这件事本身。</g>")),

    (104, "③ 片上暂存", "<b>这是 FlashAttention 的主战场。</b>"
                        "tile 就住这儿，块能开多大由这一格的容量决定。",
     (BL, "L1 ／ 共享内存", "256 KiB / SM · 共享部分 ≤ 227 KiB / 线程块",
      "<b>同一块硅，两种身份</b>：左半是硬件管的 L1，右半是软件管的共享内存。"
      "<r>FlashAttention 的 tile 住在右半边 —— 是 kernel 作者亲手搬进去的。</r>"),
     (GN, "VMEM", "64 MiB / core",
      "<b>编译器静态分配，没有 tag、不会 miss。</b>"
      "Pallas 里用 <code>BlockSpec</code> 声明每块搬多大，"
      "<r>剩下的编译器排 —— 不是 kernel 作者亲手搬。</r>")),

    (94, "④ 操作数缓冲", "矩阵操作数在进计算单元之前，要不要先落一次寄存器。",
     (BL, "寄存器堆 ＋ TMEM 支线", "寄存器 256 KiB / SM · TMEM 256 KiB / SM",
      "Blackwell 之前<b>矩阵操作数必须先落寄存器堆</b>；"
      "现在 TMA／<code>tcgen05.cp</code> 异步整块搬进 TMEM，<r>把寄存器整条绕过去</r>。"),
     (GN, "向量寄存器（个数未公开）", "形状是 8 × 128 的二维块",
      "<b>MXU 那条支线从来不经过向量寄存器</b> —— 权重与数据直接推进阵列。"
      "<g>这一点 GPU 到 Blackwell 才追上。</g>")),

    (104, "⑤ 算：两种单元轮流上", "<b>FlashAttention 一直在两种单元之间跳</b>："
                                  "矩阵乘两次、中间夹一次 softmax。",
     (BL, "Tensor Core ＋ CUDA Core", "矩阵：4 / SM　·　向量：128 / SM",
      "<code>QKᵀ</code> 和 <code>PV</code> 走 Tensor Core；"
      "<b>online softmax（取最大、求指数、累加）走 CUDA Core。</b>"),
     (GN, "MXU ＋ VPU", "矩阵：256 × 256　·　向量：VPU",
      "<code>QKᵀ</code> 和 <code>PV</code> 走 MXU，结果落进累加器；"
      "<b>online softmax 走 VPU。</b>")),

    (94, "⑥ 谁安排这个来回", "<r>这一行是第 5 节那条主线，在一个具体 kernel 上的样子。</r>",
     (BL, "warp 调度器", "每个周期挑一次",
      "<b>运行时</b>：从几十个 warp 里挑一个数据到位的发。"
      "算得慢就换一个上来 —— <b>延迟是被别人的工作盖住的。</b>"),
     (GN, "编译器 ＋ VLIW 槽", "编译期排死，精确到周期",
      "<b>编译期</b>：哪条指令、第几个周期、哪个发射槽，全排好。"
      "<r>没有第二个任务可以顶班 —— 排错了就是真的停住。</r>")),
]

TB_H = HDR_H + sum(r[0] for r in ROWS)
CNT_Y, CNT_H = TB_Y + TB_H + 22, 96
LAND_Y, LAND_H = CNT_Y + CNT_H + 22, 112
SRC_Y, SRC_H = LAND_Y + LAND_H + 22, 88
H = SRC_Y + SRC_H + 20


def build():
    f = Fig(W, H, "FlashAttention 在 GPU 与 TPU 两条内存通路上的逐站对照："
                  "片外主存、硬件缓存、片上暂存、操作数缓冲、计算单元、"
                  "以及谁来安排矩阵单元与向量单元之间的切换")
    f.title("同一个 FlashAttention，<tspan font-weight=\"700\">两边并排走完全程</tspan>"
            "　—— 横着读，一站一站对", "3.6 主图")
    f.legend([(BL, "GPU · B200"), (GN, "TPU v7"),
              (RD, "两边结构上真正不同的地方")])
    _table(f)
    _count(f)
    _land(f)
    _src(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _cell(f, x, w, y, h, trip):
    c, ttl, spec, body = trip
    f.rect(x + 4, y + 4, w - 8, h - 8, FILL[c] if c is RD else "#fff", c, 1.5, 7)
    f.t(x + 16, y + 26, ttl, "box", c)
    f.t(x + w - 16, y + 26, spec, "xxs", SUB, anchor="end")
    para(f, x + 16, y + 46, w - 32, body, "xs", 17, max_lines=4)


def _table(f):
    # 表头
    for x, w, lab, c in ((C0_X, C0_W, "这一站", INK),
                         (C1_X, C1_W, "GPU　·　B200", BL),
                         (C2_X, C2_W, "TPU　·　v7", GN)):
        f.rect(x, TB_Y, w, HDR_H, FILL[c] if c is not INK else "#f1f3f4",
               c if c is not INK else GREY, 1.4, 7)
        f.t(x + 16, TB_Y + 25, lab, "sec", c)

    y = TB_Y + HDR_H
    for h, name, fa, gpu, tpu in ROWS:
        # 站名 ＋ 「FlashAttention 在这一站干什么」
        f.rect(C0_X, y + 4, C0_W, h - 8, "#f8f9fa", GREY, 1.3, 7)
        f.t(C0_X + 14, y + 26, name, "box", INK)
        para(f, C0_X + 14, y + 46, C0_W - 28, fa, "xxs", 15, max_lines=6)
        _cell(f, C1_X, C1_W, y, h, gpu)
        _cell(f, C2_X, C2_W, y, h, tpu)
        y += h


# ══════════════════════════════════════════════════════════════════════
# ⚠️ 本课两处站数不一致，在这里一次性交代清楚。别删 —— 删了那个矛盾就回来了。
def _count(f):
    f.rect(20, CNT_Y, 1360, CNT_H, FILL[YL], YL, 1.6, 10)
    f.t(38, CNT_Y + 26, "📌 顺便统一一个口径：TPU 到底几站？—— "
                        "本课此前两处数法不一样，两个都对", "sec")
    y = para(f, 38, CNT_Y + 48, 1324,
             "图 T-6 沿 <b>MXU 那条支线</b>数：HBM ／ VMEM ／ 向量寄存器 ／ MXU ／ 累加器，"
             "<b>五站</b>。图 G-6 沿 <b>VPU 那条主路</b>数：HBM ／ VMEM ／ 向量寄存器 ／ VPU，"
             "<b>四站</b>。", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<r>数的是两条不同的路，所以不该报一个总数。</r>"
         "<b>这张图改成按功能对齐 —— 问的不是「几站」，是「谁在这一层、谁没有这一层」。</b>",
         "xs", 18)


# ══════════════════════════════════════════════════════════════════════
def _land(f):
    f.rect(20, LAND_Y, 1360, LAND_H, FILL[BL], BL, 1.8, 10)
    f.t(38, LAND_Y + 28, "⭐ 落点：这张表里只有两行是结构性的，其余都是参数",
        "sec", BL)
    y = para(f, 38, LAND_Y + 54, 1324,
             "<b>第 ② 行</b>：GPU 有一整层硬件缓存，TPU 没有。　　"
             "<b>第 ⑥ 行</b>：切换由谁安排，一边每周期现挑，一边编译期排死。"
             "<r>剩下四行都是「同一件事，两边各有一个部件」，只是容量和名字不同。</r>",
             "xs", 19)
    para(f, 38, y + 2, 1324,
         "<b>而这两行恰好决定了 FlashAttention 在两边是两种不同性质的工作</b> —— "
         "下一张图专门收这个口。<g>顺带记一个数：第 ③ 行两边的容量差 "
         "64 MiB ÷ 227 KiB ≈ <b>288 倍</b>，这直接决定块能开多大。</g>", "xs", 19)


# ══════════════════════════════════════════════════════════════════════
def _src(f):
    f.rect(20, SRC_Y, 1360, SRC_H, "#fff", GREY, 1.4, 10)
    f.t(38, SRC_Y + 26, "⚠️ 出处分层", "sec")
    y = para(f, 38, SRC_Y + 48, 1324,
             "<b>沿用图 G-6 / T-6 的标注</b>，包括那里已写明的成色："
             "L2 带宽是第三方实测、TPU 片上带宽与向量寄存器个数<b>官方未公开</b>、"
             "Tensor Core 每周期 1,024 乘加是推导值。", "xs", 17)
    para(f, 38, y + 2, 1324,
         "<b>本图当场算的</b>：64 MiB ÷ 227 KiB ≈ 288 倍。"
         "<g>「tile 住在共享内存那半边、由 kernel 作者亲手搬」是公开实现的通行做法；"
         "「Pallas 用 BlockSpec 声明块大小」出自 Pallas 文档。</g>", "xxs", 16)


if __name__ == "__main__":
    import io
    io.open("out/fig_p39_flash_walk.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p39_flash_walk")
