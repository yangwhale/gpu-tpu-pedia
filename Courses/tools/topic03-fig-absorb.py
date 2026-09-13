# -*- coding: utf-8 -*-
r"""专题三 · §5「吸收」—— 把括号挪一下，就不用把仓库里的压缩包全拆开

⭐⭐⭐ 2026-09-13 夜间 R15 新画。教材里「吸收」这个词出现了**四次**，
   一次也没画过它是什么 —— 最接近的一处是 fig3-knob1 里一句括号注：
   「（吸收进 q 那一侧）」。⛔ 对没学过的人，那句注等于没说。
   而讲义自己写着这一块「最容易讲糊」。

⭐ 它其实就是本讲已经立起来的那个装置**第三次出场**：把括号挪个位置。
     qᵀ (W_UK c)  ＝  (W_UKᵀ q)ᵀ c
   左边：先把压缩包解压成 K，再跟 q 比 ——&nbsp;**仓库里有几个包就解压几次**。
   右边：先把 q 变换一次，再直接跟压缩包比 ——&nbsp;**一次都不用解压**。

⭐⭐ 生活版（这一句是本图的题眼）：
     **与其把一万本外文书全翻译过来，不如把你的搜索词翻译过去。**

📌 原文两句都核过（DeepSeek-V2，arXiv 2405.04434）：
   · 允许：“due to the associative law of matrix multiplication, we can absorb
     W^UK into W^UQ, and W^UV into W^O”
   · 禁止：“a RoPE matrix … will lie between W^Q and W^UK and matrix
     multiplication does not obey a commutative law”
   ⭐⭐ 两条定律一允一禁，正好说清 MLA 最难的两件事：
     **结合律允许你挪括号，交换律不许你换顺序。**

⚠️ 省多少是本课自己算的，而且算出来的结论**跟直觉不一样**：
   一开始我以为是「省 S 倍」（解压 S 次变 1 次）。⛔ 错 ——
   吸收之后每个 token 的点积从 128 维变成 512 维，**贵了 4 倍**，这一头在吃回去。
   两笔加一起，省的倍数**上限正好是每头维度 d_h = 128 倍**，
   S 再长也过不去。⭐ 这个上限没见别处写过。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2, wpx)

W = 1400
DC, DH, NH = 512, 128, 128        # DeepSeek-V3：隐向量 512 · 每头 128 · 128 个头


def main():
    # ══ 先算账 ════════════════════════════════════════════════════
    one = DH * DC                  # 解压一个 token（一个头）要多少次乘加
    # ⭐⭐ 这个相等不是巧合，是这张图成立的支点：
    #   解压一个 token 是 (d_h×d_c) 的矩阵乘一个 d_c 向量；
    #   变换一个 query 是同一个矩阵转置乘一个 d_h 向量 —— 同一块矩阵，同样多的乘加。
    assert one == DH * DC
    SS = (128, 1024, 4096, 131072)
    NAIVE = [s * (DH * DC + DH) for s in SS]
    ABSORB = [DH * DC + s * DC for s in SS]
    GAIN = [n / float(a) for n, a in zip(NAIVE, ABSORB)]
    CEIL = (DH * DC + DH) / float(DC)         # S→∞ 的上限
    assert 64 < GAIN[0] < 65 and 128 < GAIN[3] < 129, GAIN
    assert abs(CEIL - DH) < 1, CEIL           # 上限 ≈ 每头维度

    f = Fig(W, "MLA 的吸收：把括号挪一下，就不用解压缓存")
    yy = f.header(
        "「吸收」到底是什么 ——&#160;把括号挪一下，仓库里的压缩包一个都不用拆",
        "MLA 把 K 和 V 压成一个 512 维的压缩包存起来。那生成下一个词的时候，"
        "不是得把它们<tspan font-weight=\"700\">全拆开</tspan>才能比对吗？"
        "——&#160;不用。这张图画的就是那个「不用」。",
        legend=[(RD, "天真做法：全拆开"), (GR, "吸收：一个都不拆"),
                (PU, "同一块矩阵 W_UK（认这个紫色）")])

    # ══ ① 天真做法 ════════════════════════════════════════════════
    PH1 = 300
    top = f.panel(0, yy, W, PH1,
                  "① 天真做法：每生成一个词，把仓库里所有压缩包<tspan "
                  "font-weight=\"700\">全部拆开</tspan>",
                  RD, tag="解压 S 次")
    ry = top + 40
    f.t(30, ry, "仓库（KV cache）", GY, bold=True, size=16, cls="svglbl")
    for i in range(5):                        # 五个压缩包代表 S 个
        f.icon("box", 34 + i * 62, ry + 16, 46, 50, GR, "#e6f4ea")
        f.t(57 + i * 62, ry + 84, "512", GY2, size=14, anchor="middle")
    f.t(378, ry + 50, "…", GY2, size=22)
    f.t(404, ry + 46, "共 S 个", GY, bold=True, size=16)
    f.t(404, ry + 68, "（128K 上下文就是 13 万个）", GY2, size=14)

    # 紫色的 W_UK 块 —— 这一轮它站在 K 那一侧
    f.line(600, ry + 46, 646, ry + 46, RD, 1.8)
    f.box(652, ry + 18, 108, 56, "#f3e8fd", PU, 8)
    f.t(706, ry + 44, "W_UK", PU, bold=True, size=17, anchor="middle")
    f.t(706, ry + 64, "上投影", PU, size=14, anchor="middle")
    f.line(766, ry + 46, 812, ry + 46, RD, 1.8)
    # ⛔ 这里原先只画了**一张** K。可这一格全部的重点就是「要拆 S 次」——
    #   画一张等于把自己想说的那件事画没了。⭐ 判据：**数量本身就是论点时，
    #   就得把数量画出来。**
    for i in range(3):
        f.icon("paper", 818 + i * 42, ry + 16, 38, 54, RD, "#fce8e6")
    f.t(950, ry + 50, "…", GY2, size=22)
    f.t(862, ry + 84, "每个 K 都是 128 维", RD, size=14, anchor="middle")
    f.t(982, ry + 40, "拆出来的 K，也是 S 个", RD, bold=True, size=16)
    f.t(982, ry + 62, "⛔ 而且每生成一个词就得全部重拆一遍", RD, size=15)

    f.t(30, top + 158,
        "⚠️ 注意「重拆」两个字：<tspan font-weight=\"700\">拆完不能留</tspan>"
        " ——&#160;留下来就等于又存了一份没压缩的 K，"
        "那正是 MLA 想省掉的东西。所以每生成一个词，这 S 次解压全都要重做一遍。",
        INK, size=16)
    f.t(30, top + 190,
        "📌 论文原话（DeepSeek-V2）：这样就"
        "<tspan font-weight=\"700\">「must recompute the keys for all the "
        "prefix tokens during inference」</tspan>", GY, size=15)
    f.t(30, top + 232,
        "🏠 换成生活里的说法：书架上有一万本外文书，你每查一个词，"
        "就把<tspan font-weight=\"700\">一万本全翻译成中文</tspan>再找。"
        "查完扔掉，下次再翻译一万本。", GY, size=16)

    # ══ ② 吸收 ════════════════════════════════════════════════════
    yy = top + PH1 + 26
    PH2 = 330
    top = f.panel(0, yy, W, PH2,
                  "② 吸收：把那块紫色矩阵<tspan font-weight=\"700\">搬到 q "
                  "那一侧</tspan> ——&#160;压缩包一个都不用拆",
                  GR, tag="解压 0 次")
    ry = top + 40
    # query 这一侧：紫块搬过来了
    f.t(30, ry, "查询 q（一个，就一个）", GY, bold=True, size=16, cls="svglbl")
    f.icon("note", 34, ry + 16, 44, 50, BL, "#e8f0fe")
    f.t(56, ry + 84, "q（128）", BL, size=14, anchor="middle")
    f.line(86, ry + 44, 128, ry + 44, GR, 1.8)
    f.box(134, ry + 16, 108, 56, "#f3e8fd", PU, 8)
    f.t(188, ry + 42, "W_UK", PU, bold=True, size=17, anchor="middle")
    f.t(188, ry + 62, "转置着用", PU, size=14, anchor="middle")
    f.line(248, ry + 44, 290, ry + 44, GR, 1.8)
    f.icon("note", 296, ry + 16, 44, 50, GR, "#e6f4ea")
    f.t(318, ry + 84, "q′（512）", GR, size=14, anchor="middle")
    f.t(352, ry + 38, "⭐ 只做这一次", GR, bold=True, size=16)
    f.t(352, ry + 60, "（不管仓库里有多少个包）", GY, size=15)

    # 直接跟压缩包比
    f.t(600, ry, "仓库（原封不动）", GY, bold=True, size=16, cls="svglbl")
    for i in range(5):
        f.icon("box", 604 + i * 62, ry + 16, 46, 50, GR, "#e6f4ea")
    f.t(944, ry + 50, "…", GY2, size=22)
    f.t(978, ry + 40, "✅ 一个都没拆", GR, bold=True, size=17)
    f.t(978, ry + 64, "直接拿 q′ 去跟压缩包点积", GY, size=15)

    # 代数：括号挪了一下
    f.box(30, top + 152, 1340, 62, "none", LINE, 8)
    f.t(52, top + 190,
        "q<tspan font-size=\"13\">ᵀ</tspan> "
        "<tspan fill=\"#5f6368\">(</tspan> "
        "<tspan fill=\"#9334e6\" font-weight=\"700\">W_UK</tspan> c "
        "<tspan fill=\"#5f6368\">)</tspan>"
        "　　＝　　"
        "<tspan fill=\"#5f6368\">(</tspan> "
        "<tspan fill=\"#9334e6\" font-weight=\"700\">W_UK</tspan>"
        "<tspan font-size=\"13\">ᵀ</tspan> q "
        "<tspan fill=\"#5f6368\">)</tspan>"
        "<tspan font-size=\"13\">ᵀ</tspan> c", INK, size=22, mono=True)
    f.t(660, top + 170, "同一个乘法。只是括号挪了个位置。", GY, size=15)
    f.t(660, top + 194,
        "⭐ 左边括号在 c 那边 ——&#160;有几个 c 就算几次。", INK, size=15)
    f.t(660, top + 216,
        "⭐ 右边括号在 q 那边 ——&#160;只有一个 q，所以只算一次。", GR, size=15)

    f.t(30, top + 248,
        "🏠 生活版就是一句话：<tspan font-weight=\"700\">"
        "与其把一万本外文书全翻译过来，不如把你的搜索词翻译过去。</tspan>",
        INK, size=18)
    f.t(30, top + 276,
        "⭐ V 那一侧同理：W_UV 可以吸进输出投影 W_O ——&#160;"
        "所以 K 和 V <tspan font-weight=\"700\">两边都不用拆</tspan>。", GY,
        size=16)

    # ══ ③ 这笔账 ══════════════════════════════════════════════════
    yy = top + PH2 + 26
    PH3 = 336
    top = f.panel(0, yy, W, PH3,
                  "③ 省多少？——&#160;算出来的答案跟直觉不一样", BL,
                  tag="d_c=512 · d_h=128 · 本课自算")
    f.t(30, top + 34,
        "直觉会说「解压 S 次变成 1 次，所以省 S 倍」。"
        "<tspan font-weight=\"700\">⛔ 不对</tspan> ——&#160;"
        "吸收之后每个 token 的点积从 128 维变成了 512 维，"
        "<tspan font-weight=\"700\">这一头贵了 4 倍</tspan>，在把省下的吃回去。",
        INK, size=16)
    BW, BX = 760, 470
    for i, s in enumerate(SS):
        by = top + 74 + i * 52
        lab = "S = %s" % ("{:,}".format(s) if s < 100000 else "128K")
        f.t(30, by + 28, lab, GY, bold=True, size=16)
        f.t(180, by + 28, "（%s 个 token 在仓库里）" % "{:,}".format(s), GY2,
            size=14)
        g = BW * GAIN[i] / CEIL
        f.box(BX, by + 8, BW, 28, "none", LINE, 6)
        f.box(BX, by + 8, g, 28, "#e8f0fe", BL, 6)
        f.t(BX + BW + 14, by + 29, "省 %.0f×" % GAIN[i], BL, bold=True, size=17)
    f.line(BX + BW, top + 68, BX + BW, top + 282, RD, 1.4, dash="4 4",
           arrow=False)
    f.t(BX + BW + 14, top + 62, "上限 %d×" % round(CEIL), RD, bold=True,
        size=16)

    f.t(30, top + PH3 - 48,
        "⭐⭐ 这个上限<tspan font-weight=\"700\">正好是每个头的维度 "
        "d_h = 128</tspan>，S 再长也过不去 ——&#160;"
        "因为解压一个 token 要 d_h×d_c 次乘加，而点积只要 d_c 次，两者的比就是 d_h。",
        INK, size=16)

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH3 + 30
    yy = f.band(yy, "warn",
                "两条定律，一条允许、一条禁止 ——&#160;MLA 最难的两件事都在这里", [
        "<tspan font-weight=\"700\">结合律允许你挪括号</tspan>："
        "a(bc) ＝ (ab)c。②那一步靠的就是它，论文原话是 "
        "「due to the associative law of matrix multiplication, we can absorb "
        "W^UK into W^UQ, and W^UV into W^O」。",
        "<tspan font-weight=\"700\">⛔ 但交换律不成立</tspan>：ab ≠ ba。"
        "RoPE 会往 q 和 W_UK 中间塞进一个跟位置有关的旋转 R，"
        "而<tspan font-weight=\"700\">夹在中间的东西挪不出去</tspan> ——&#160;"
        "原话「a RoPE matrix … will lie between W^Q and W^UK and matrix "
        "multiplication does not obey a commutative law」。",
        "⭐ 所以 DeepSeek 把带位置的那一小块<tspan font-weight=\"700\">"
        "单独拎出来走 64 维一路</tspan>（本节那张寄快递的图讲的就是这件事）："
        "包裹外面贴日期，包裹里面保持「一次就能翻译完」。",
    ])
    yy = f.band(yy + 14, "info", "同一个把戏，本讲这是第三次出场", [
        "<tspan font-weight=\"700\">§7 线性注意力</tspan>：把括号从 (QKᵀ)V "
        "挪成 Q(KᵀV) ——&#160;那个句长×句长的大方块就不用建了。",
        "<tspan font-weight=\"700\">§5 MLA 吸收</tspan>（这张图）："
        "把括号从 qᵀ(W_UK c) 挪成 (W_UKᵀ q)ᵀ c ——&#160;压缩包就不用拆了。",
        "⭐ 两次都是<tspan font-weight=\"700\">同一个数学恒等式</tspan>，"
        "也都被同一件事挡过：§7 被因果 mask 挡住，这里被 RoPE 挡住。"
        "<tspan font-weight=\"700\">⛔ 挡住结合律的，永远是「中间被塞了个东西」。</tspan>",
    ])
    yy = f.src(yy + 16,
               "「吸收」与「RoPE 挡住它」两处原文均出自 DeepSeek-V2，"
               "arXiv 2405.04434（§2.1.2、§2.1.3）；超参 d_c=512 / d_h=128 / "
               "n_h=128 是 DeepSeek-V3 的口径",
               "⭐ ③ 那笔账是<tspan font-weight=\"700\">本课自己算的</tspan>，"
               "脚本里带断言：天真 ＝ S×(d_h·d_c ＋ d_h)，吸收 ＝ d_h·d_c ＋ S·d_c；"
               "上限 (d_h·d_c＋d_h)/d_c ≈ d_h。⚠️ 只数乘加，"
               "<tspan font-weight=\"700\">没算访存</tspan> ——&#160;"
               "真机上访存往往才是瓶颈，所以这是个下界不是实测",
               "⚠️ 吸收<tspan font-weight=\"700\">只在 decode 用得上</tspan>："
               "prefill 时一批里有很多个 q，「只变换一次」这个便宜就没了"
               "（这也是 §五 表里「压缩不生效」那一行的意思）")
    f.save("fig3-absorb.svg", yy + 6)


main()
