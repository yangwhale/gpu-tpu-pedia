# -*- coding: utf-8 -*-
r"""专题三 · §6.1b「滑窗凭什么敢砍，砍了为什么会崩」

⭐⭐⭐ 2026-09-13 **整张重画**，全部换成生活画面。

  ① **凭什么敢砍** ——&nbsp;画**传话**：每个人只跟身边 4 个人说话，
     但话可以一层一层往外传。**层数是免费的射程。**
     Mistral 7B：窗口 4096 × 32 层 →&nbsp;131,072（脚本当场乘出来断言）。

  ② **砍了为什么会崩** ——&nbsp;一个数字对比就够，画成**两根天差地别的柱子**：
     纯窗口 0+1024 →&nbsp;**5158.07**；把最前面四个 token 留下 →&nbsp;**5.40**。
     ⭐⭐ 判决性实验：把那四个换成**换行符**，5.60 ——&nbsp;几乎一样。
     **所以起作用的是位置，不是内容。**

  ③ **为什么会有这么个东西** ——&nbsp;画成**必须投满的选票**：
     softmax 要求每一行的票加起来正好 100 分，
     **哪怕这一行没什么想看的，票也必须投出去** ——&nbsp;
     于是大家把废票都投给了最前面那几个（自回归下只有它们人人都够得着）。
     ⭐ 两条看起来同样彻底的解法，**只有一条成立**。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    WIN, LAY = 4096, 32
    span = WIN * LAY
    assert span == 131072
    PPL_WIN, PPL_SINK, PPL_NL = 5158.07, 5.40, 5.60
    assert PPL_WIN / PPL_SINK > 900

    f = Fig(W, "滑窗凭什么敢砍：每个人只跟身边几个说话，但话能一层层往外传，"
               "层数是免费的射程；砍了为什么会崩：把最前面四个 token 扔掉，"
               "困惑度从 5.40 炸到 5158；为什么：softmax 要求每行的票必须投满")
    f.marks = set()
    y0 = f.header(
        "滑窗：凭什么敢砍，砍了为什么会崩",
        "一个<tspan font-weight=\"700\">按时间顺序讲的侦探故事</tspan>",
        [(GR, "敢砍的理由"), (RD, "崩了"), (BL, "真正的原因")])

    # ══════════ ① 传话 —— 但射程不是免费的 ══════════════════════
    # ⛔⛔⛔ 2026-09-13 **本课这里原来教的是错的心智模型**：
    #   「层数是免费的射程，4096 × 32 = 131,072」。
    # ⭐ 查证（guangxuanx.com/blog/stacking-swa.html，作者是 StreamingLLM 一作）
    #   原话：「The formula D_eff = W·ln(ε)/ln(1−α) **should replace "L × W"
    #   in your mental model of these models.**」
    #   · 纯 SWA（无残差）：每层往回跳的距离是 [0, W) 上的**随机数**，不是每次跳满。
    #     L 层就是 L 个随机数的和 → 中心极限 → 高斯。有效射程 ≈ **0.58·W·√L**。
    #   · 有残差（真实模型）：α≈0.95 意味着**九成五的信息根本没进注意力层**，
    #     直接从底下窜到顶上。于是衰减从高斯变成指数，
    #     有效射程 ≈ W × 4.6/|ln(1−α)| ——&nbsp;**跟层数完全无关**。
    # ⚠️ 置信度分两层，图上必须写清：√L 那半推导干净（且跟 CNN 有效感受野
    #   的独立结论对得上）；而 α≈0.95 是作者**断言**不是实测，1.5W 对 α 极敏感
    #   （α=0.90 → 2.0W，α=0.99 → 1.0W）。⛔ 所以图上只说「一到两个窗口宽、
    #   跟层数无关」，**不写死 1.5**。而且这是个人博客，非同行评议。
    # ⭐⭐ 这条改完反而更值钱：它正好解释了**混合架构为什么必须存在**
    #   （原文自己说的）——&nbsp;接上 §八。
    import math as _m
    PH = 560
    py = f.panel(0, y0, W, PH,
                 "① 凭什么敢砍 ——　每层只看身边几个，但话能往外传",
                 GR, sub="⚠️ 能传多远，比「层数 × 窗口」小得多")

    ay = py + 20
    # 左：传话本身（这一半是对的，保留）
    f.t(56, ay + 22, "话确实能一层层往外传", GR, True, 20)
    N = 7
    for L in range(3):
        yy = ay + 42 + L * 46
        f.t(56, yy + 22, "第 %d 层" % (L + 1), GY2, size=15)
        for i in range(N):
            x = 138 + i * 62
            on = i <= 2 + L * 2
            f.box(x, yy, 48, 30, "#e6f4ea" if on else BG2,
                  GR if on else LINE2, 5)
            f.t(x + 24, yy + 21, str(i + 1), GR if on else GY2, on, 15,
                "middle")

    # 右：高尔顿板 —— 每层往回跳多远是随机的，L 层的和堆成钟形
    gx = 640
    f.t(gx, ay + 22, "⛔ 但每一层往回跳多远，是随机的", RD, True, 17)
    f.t(gx, ay + 46, "L 层 ＝ L 个随机数相加 ——　堆成一个钟形", GY, size=16)
    for r in range(4):                       # 钉板
        for c in range(r + 1):
            f.p.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"/>'
                       % (gx + 150 + (c - r / 2.0) * 26, ay + 70 + r * 20, GY2))
    BASE, BH = ay + 176, 58
    BINS = [_m.exp(-((k - 5.5) ** 2) / 7.0) for k in range(12)]
    mxb = max(BINS)
    for k, v in enumerate(BINS):
        h = BH * v / mxb
        f.box(gx + 20 + k * 26, BASE - h, 20, h, "#e6f4ea", GR, 2)
    f.t(gx + 20, BASE + 24, "近", GY2, size=15)
    f.t(gx + 20 + 11 * 26, BASE + 24, "远", GY2, size=15, anchor="end")
    f.line(gx + 340, BASE - BH - 10, gx + 340, BASE + 6, RD, 2.0, dash="5,4")
    f.t(gx + 348, BASE - 30, "「层数 × 窗口」", RD, True, 17)
    f.t(gx + 348, BASE - 8, "在这儿 ——　钟形的尾巴", RD, size=16)
    f.t(gx + 348, BASE + 14, "早就没了", RD, size=16)

    # 下：两条通道 —— 残差才是主干道
    cy = ay + 244
    f.box(56, cy, 1288, 122, "#fff", INK, 10)
    f.t(80, cy + 32, "⭐⭐ 而且真实模型里，九成五的信息根本没走注意力这条路", INK,
        True, 17)
    f.box(80, cy + 48, 700, 26, "#e8f0fe", BL, 5)
    f.t(92, cy + 67, "残差 ——　直接从底下窜到顶上（约 95%）", BL, True, 17)
    f.box(80, cy + 80, 46, 12, "#fef7e0", OR, 3)
    f.t(136, cy + 91, "注意力 ——　真正往回看的那一小股（约 5%）", OR, True, 17)
    f.t(80, cy + 112,
        "每往回跳一个窗口就再乘一次这个小数 ——　"
        "<tspan font-weight=\"700\">于是有效射程跟层数无关，大约就是一到两个窗口宽</tspan>。",
        GY, size=17, w=1240)

    f.box(820, cy + 44, 500, 62, "#fce8e6", RD, 8)
    f.t(840, cy + 70, "Mistral 7B：%s × %d 层 ＝ %s"
        % (format(WIN, ","), LAY, format(span, ",")), RD, True, 18)
    f.t(840, cy + 94, "⛔ 那是<tspan font-weight=\"700\">理论上限</tspan>，"
        "不是能用的长度", RD, True, 17)

    # ── 工业界的答案：不堆层数，改成隔一层插一层全注意力 ──────────
    # ⭐ 这组数是**我自己读 config 读到的**（huggingface.co/openai/gpt-oss-20b
    #   的 config.json），不是转述：layer_types 是 sliding / full **1:1 交替**，
    #   而且 sliding_window 只有 **128**。
    # ⭐⭐ 它正好是上面那条结论的工业答案：既然堆层数没用，那就别指望堆 ——
    #   隔一层插一层真正的全注意力。⛔ 顺带它也说明窗口可以开得极小。
    ty = cy + 140
    f.box(56, ty, 1288, 96, "#e8f0fe", BL, 10)
    f.t(80, ty + 32, "⭐⭐ 所以工业界的答案不是「堆更多层」，是"
        "<tspan font-weight=\"700\">隔一层插一层真正的全注意力</tspan>", BL,
        True, 17)
    for k in range(12):
        full = (k % 2 == 1)
        f.box(80 + k * 46, ty + 46, 38, 30, "#1a73e8" if full else "#fff",
              BL, 4)
        f.t(99 + k * 46, ty + 67, "全" if full else "窗",
            "#fff" if full else BL, True, 16, "middle")
    f.t(640, ty + 58, "gpt-oss-20b：24 层，<tspan font-weight=\"700\">"
        "滑窗 / 全注意力 1:1 交替</tspan>，", GY, size=17)
    f.t(640, ty + 82, "而且它的窗口只有 <tspan font-weight=\"700\">128</tspan>"
        " ——　窗口小到这个地步，靠堆层数是绝无可能够到 128K 的。", GY, size=17)

    # ══════════ ② 崩了 ══════════════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 488
    py2 = f.panel(0, y1, W, PH2, "② 砍了为什么会崩 ——　扔掉最前面四个，就崩了",
                  RD, sub="Llama-2-13B，PG19")

    by = py2 + 24
    # ⛔⛔ 「柱子按对数画」这句原来在柱子**下面 130px** ——&nbsp;
    #   读者先看到「5158 只比 5.40 高两倍多」，得出「差得也不算多」，
    #   往下读才知道是对数轴，而那一眼的印象跟这张图要说的正好相反。
    # ⭐ 判据：**读图的钥匙必须在图之前。** 放在后面 ＝ 先让人看错一眼再纠正。
    f.t(120, by + 18, "⚠️ 先说怎么读：这三根柱子<tspan font-weight=\"700\">"
        "按对数画</tspan> ——　线性画的话后两根根本看不见。", RD, size=17, w=860)
    f.t(120, by + 42, "<tspan font-weight=\"700\">"
        "5158 和 5.40 差的是三个数量级，不是三倍。</tspan>", RD, size=17, w=860)
    # ⛔⛔ 2026-09-14：原来是 by + 250，于是最高那根柱子的「⛔ 崩了」标签
    #   （画在 BASE − h − 44 ＝ by + 56）**正好压在上面那行读图提示上**
    #   （by + 42，行高约 17）。渲染出来两行字叠在一起。
    # ⭐ 为什么几何 lint 没抓到：那两个文字块**不共享基线也不同列宽**，
    #   撞的是「一行的下缘」和「另一行的上缘」，只差十几个像素 ——
    #   ⭐⭐ 判据：**标签的 y 是从别的量算出来的（BASE − h − 44），
    #     它会跟着数据变**。数据一变高，标签就往上爬，爬到谁头上取决于那个数。
    #     这类「算出来的坐标」必须按最极端的那个数据点验一次，不能只看代码。
    BASE, HMAX = by + 286, 150   # ⭐ 上面多了两行读图提示，柱子整体下移
    import math
    for i, (lab, v, col, note) in enumerate([
        ("只留窗口\n0 + 1024", PPL_WIN, RD, "⛔ 崩了"),
        ("留最前面 4 个\n4 + 1020", PPL_SINK, GR, "✅ 好了"),
        ("那 4 个换成换行符\n4 + 1020", PPL_NL, BL, "⭐ 几乎一样"),
    ]):
        x = 120 + i * 300
        h = HMAX * math.log10(v) / math.log10(PPL_WIN)
        f.box(x, BASE - h, 150, h, col, "none", 5)
        f.t(x + 75, BASE - h - 14, "%.2f" % v, col, True, 26, "middle")
        for k, ln in enumerate(lab.split("\n")):
            f.t(x + 75, BASE + 26 + k * 24, ln, GY, size=16, anchor="middle")
        f.t(x + 75, BASE - h - 44, note, col, True, 18, "middle")
    f.t(120, BASE + 130, "困惑度（越低越好）", GY2, size=16)

    f.box(1000, by + 24, 360, 192, "#e8f0fe", BL, 10)
    f.t(1024, by + 66, "⭐⭐ 判决性的是第三根", BL, True, 21)
    f.t(1024, by + 104, "把那四个 token 换成", GY, size=17)
    f.t(1024, by + 134, "毫无意义的换行符", GY, size=17)
    f.t(1024, by + 172, "结果几乎一样", BL, True, 22)
    f.t(1024, by + 202, "→　起作用的是位置，不是内容", BL, True, 17)

    # ══════════ ③a 得票最高的那位，什么都不做 ══════════════════
    # ⭐⭐⭐ 2026-09-13 新增。我们的选票比喻停在「票必须投满，于是全堆到
    #   最前面几个人身上」——&nbsp;⛔ **缺了最后一步，而缺的这步才是机制本身**：
    #   它投的那个人，**value 几乎是零**。
    # ⚠️ 而且要诚实：「不许弃权的喧嚣民主」这个比喻**不是我们原创的** ——
    #   Evan Miller 2023-07《Attention Is Off By One》原话就是
    #   "a deafening democracy where abstention is disallowed"。主动引他。
    y2a = y1 + PH2 + 18
    PH3A = 352
    py3a = f.panel(0, y2a, W, PH3A,
                   "③ 得票最高的那位，什么都不做", BL,
                   sub="⭐ 这一步才是机制本身 ——　前面两格只说了「票投给了谁」")

    ay3 = py3a + 34
    # 两根柱子：注意力权重冲天 vs value 模长贴地
    BW2, BH2 = 118, 132
    for k, (lab, hi, col, note) in enumerate((
            ("注意力权重", 1.0, RD, "冲天"),
            ("它的 value 模长", 0.08, GY2, "贴地"))):
        x = 130 + k * 220
        h = BH2 * hi
        f.box(x, ay3 + 20 + BH2 - h, BW2, max(h, 4), col, "none", 4)
        f.t(x + BW2 / 2.0, ay3 + 178, lab, col, True, 17, "middle")
        f.t(x + BW2 / 2.0, ay3 + 202, note, GY2, size=16, anchor="middle")
    f.t(130, ay3 + 8, "第 0 号座位上那个 token：", INK, True, 19)
    f.t(130, ay3 + 234, "⭐ 把票投给他 ＝ 弃权", BL, True, 21)

    # 为什么偏偏是第 0 个：因果掩码下唯一人人都够得着的座位
    MX = 620
    f.t(MX, ay3 + 8, "为什么偏偏是最前面那几个？", INK, True, 17)
    N3 = 7
    C3 = 26
    for r in range(N3):
        for c in range(N3):
            on = c <= r
            f.box(MX + c * C3, ay3 + 26 + r * C3, C3 - 3, C3 - 3,
                  ("#e8f0fe" if c else "#1a73e8") if on else "#fff",
                  "none" if on else LINE2, 2)
    f.t(MX + C3 / 2.0, ay3 + 26 + N3 * C3 + 22, "↑", BL, True, 20, "middle")
    f.t(MX, ay3 + 26 + N3 * C3 + 48, "因果掩码下，第 0 列是<tspan "
        "font-weight=\"700\">唯一一列全满的</tspan>", BL, True, 17)
    f.t(MX, ay3 + 26 + N3 * C3 + 72, "——　不是它特殊，"
        "是<tspan font-weight=\"700\">只有它人人都够得着</tspan>", GY, size=17)

    f.box(1010, ay3 + 14, 334, 214, "#e8f0fe", BL, 10)
    f.t(1030, ay3 + 46, "⭐⭐ 于是整件事说得通了", BL, True, 20)
    for i3, ln in enumerate([
            "softmax 不许弃权，",
            "模型就自己造了一个",
            "**弃权用的候选人**出来：",
            "永远在场、什么主张都没有。",
            "",
            "⛔ 砍掉他不是砍掉一个老 token，",
            "是砍掉了**弃权票这个选项**。"]):
        if not ln:
            continue
        b = "**" in ln
        f.t(1030, ay3 + 78 + i3 * 24, ln.replace("**", ""), GY, b, 17, w=300)

    # ══════════ ④ 必须投满的选票 ════════════════════════════════
    y2 = y2a + PH3A + 18
    PH3 = 324
    py3 = f.panel(0, y2, W, PH3, "④ 两条看起来同样彻底的解法，只有一条成立",
                  BL, sub="softmax 要求每一行的票必须投满")

    vy = py3 + 20
    f.box(56, vy + 26, 600, 150, "#e8f0fe", BL, 10)
    f.t(80, vy + 64, "规矩：每一行的票加起来必须正好 100 分", BL, True, 17)
    f.t(80, vy + 102, "——　哪怕这一行「没什么特别想看的」，", GY, size=17)
    f.t(80, vy + 134, "票<tspan font-weight=\"700\">也必须投出去</tspan>。", GY, size=18)
    f.t(80, vy + 166, "这就是 softmax 的归一化", GY2, size=15)

    f.path([(672, vy + 100), (716, vy + 100)], BL, 2.0)

    f.box(736, vy + 26, 624, 150, "#fff", BL, 10)
    f.t(760, vy + 64, "于是废票都投给了最前面那几个", BL, True, 17)
    f.t(760, vy + 102, "为什么偏偏是它们？——　因为自回归：", GY, size=17)
    f.t(760, vy + 134, "<tspan font-weight=\"700\">全场只有开头那几个，人人都够得着。</tspan>",
        GY, size=17)
    f.t(760, vy + 166, "把废票桶撤了，票没处投，整行就乱套", GY2, size=15)

    sy = vy + 196
    f.box(56, sy, 640, 86, "#e6f4ea", GR, 10)
    f.t(80, sy + 36, "⭐ 一个成立的解法", GR, True, 20)
    f.t(80, sy + 68, "预训练时加一个<tspan font-weight=\"700\">可学的</tspan>废票桶　"
        "→　1+1023 下 PPL 18.01", GY, size=17)

    f.box(720, sy, 640, 86, "#fce8e6", RD, 10)
    f.t(744, sy + 36, "⛔ 一个看起来对、但被论文自己证伪的", RD, True, 17)
    f.t(744, sy + 68, "给一个<tspan font-weight=\"700\">全零</tspan>的桶"
        "（＝softmax-off-by-one）　→　PPL 29214", GY, size=17)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这个故事真正的教益 —— 比 sink 本身值钱", [
        "attention sink 不是 bug，也不是谁设计的特性，"
        "它是<tspan font-weight=\"700\">「票必须投满」这条规矩逼出来的副产品</tspan>。",
        "⭐ 判据：<tspan font-weight=\"700\">看到模型里一个「毫无道理却极其稳定」的现象，"
        "先去找是不是某个守恒 / 归一化约束逼出来的。</tspan>"
        "量化里那批总也压不下去的 outlier，跟这是同一件事（见专题八）。",
        "⛔ 还有一条：<tspan font-weight=\"700\">这个 bug 从公式上完全看不出来</tspan> ——&#160;"
        "是把注意力矩阵<tspan font-weight=\"700\">画出来</tspan>才发现的。"
        "这一讲所有的图，都是这个道理。",
    ])

    yy = f.band(yy + 14, "warn", "别把「理论射程」当「有效射程」", [
        "%s × %d ＝ <tspan font-weight=\"700\">%s</tspan> 是个上界，"
        "说的是「信息最远能传到这儿」，<tspan font-weight=\"700\">不是「这么远还能用」"
        "</tspan> ——&#160;每跨一层只挪一格窗口，而且一路被后面的信息稀释。"
        % (format(WIN, ","), LAY, format(span, ",")),
        "⭐ 稳妥说法：<tspan font-weight=\"700\">滑窗把「远处」从「看不见」"
        "变成了「看得见但很模糊」</tspan> ——&#160;"
        "所以后面那些方案才要在滑窗之外再加一条「挑着看」的路。",
    ])

    yy = f.src(yy + 24,
               "① 有效射程那一格出自 guangxuanx.com/blog/stacking-swa.html（作者是 StreamingLLM 一作，⚠️ 个人博客非同行评议）：纯 SWA ≈ 0.58·W·√L；有残差时跟层数无关。⚠️ 其中 α≈0.95 是作者断言不是实测，所以本图只说「一到两个窗口宽」不写死倍数",
               "gpt-oss-20b 的 1:1 交替与 sliding_window=128 是本课直接读 huggingface.co/openai/gpt-oss-20b 的 config.json 得到的",
               "Mistral 7B arXiv 2310.06825 §2（k×W 射程、W=4096 / 32 层、"
               "rolling buffer cache）；131,072 由脚本当场乘出来并断言",
               "②③ 出自 StreamingLLM（Xiao 等 arXiv 2309.17453, ICLR 2024）"
               "论文表 1 / 表 2 与论文 §3.1 / §3.3：5158.07 → 5.40、换行符 5.60、"
               "留 1/2/4/8 个的对照",
               "⛔ Zero Sink（＝softmax-off-by-one）那组反例出自同文表 3 / 表 10 的"
               "三个 160M 预训练对照",
               "⚠️ 表 1（PG19 第一本书，65K）与表 2（拼接后 400K）"
               "<tspan font-weight=\"700\">不是同一个评测集</tspan>；"
               "⚠️ 「传话」是本课的比喻；⛔ 但「不许弃权的选票」"
               "<tspan font-weight=\"700\">不是本课原创</tspan> ——&#160;"
               "Evan Miller 2023-07《Attention Is Off By One》原话就是 "
               "「a deafening democracy where abstention is disallowed」",
               "③ 「value 模长极小」出自 Barbero 等 arXiv 2504.02732 图 4；"
               "「第 0 列是因果掩码下唯一全满的一列」是由掩码定义直接得出的")
    f.save("fig3-swa-why.svg", yy + 6)


main()
