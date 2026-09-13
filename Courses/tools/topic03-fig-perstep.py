# -*- coding: utf-8 -*-
r"""专题三 · §九「代价」—— 把那张全是箭头的表，换成一笔能自己核的账

⭐⭐⭐ 2026-09-14 吸收全网精华 R26。装置偷自 Epoch AI 那条
   **「字节 →&#160;毫秒 →&#160;钱」** 的换算链：抽象的「省了多少」，
   一路换算到一个人人有直觉的量，中间每一步都能自己验算。

⛔ 为什么这张图该存在：**§九 整节只有一张全是 ↓↓↓ 的表。**
   而这门课自己一直在说「问『省了多少』之前先问『省的是哪一样』」——
   一张箭头表恰恰是**回答不了这个问题**的东西。
   ⭐ 于是这里把同一批方案，换成**一步 decode 到底要从 HBM 搬多少字节**。

⭐⭐ 算出来有三条，一条比一条反直觉，而且每条都能自己核：

   ① **MHA 的那一步里，93.4% 的字节是一个人的 KV**
      ——&#160;权重只占 6.6%。「KV cache 是瓶颈」这句话，这才叫画出来了。

   ② **显存省了 56.9 倍，单用户 decode 只快了 7.08 倍。**
      ⛔ 差在哪？**MHA 装不下，被迫用了 12 张 device 而不是 7 张** ——
      它拿显存换来的卡，顺手也把带宽换来了。
      ⭐ 所以 MLA 真正省下来的不是时间，是**那 5 张卡** ——
      而那 5 张卡可以拿去服务别人。这正好接上 §11.1 的收尾。

   ③ **MLA 之后再上稀疏，单用户 decode 只再快 1.24 倍。**
      因为这时候**瓶颈已经搬到权重那一段去了**（读 34.46，KV 只剩 0.13）。
      ⭐⭐ 这不是稀疏没用 ——&#160;这正是 §九 那张表里
      「DSA：prefill 为主」那一格的**数值版**。表断言，这里算给你看。

⚠️ 五条口径写在图上，一条都不能省，否则这张图会被当成实测：
   ① 这是**下界**：只算 HBM 读，不算计算、不算 MoE 的 all-to-all、
      不算 kernel 损耗。真机只会更慢。
   ② **batch = 1。** 批量一大，权重那段被所有人摊薄、KV 那段不摊 ——
      画面会翻回 KV 主导。⭐ 这恰恰是 §二 那张图的另一面。
   ③ 卡数按**装得下**算，没算算力够不够。
   ④ DSA 那根：索引器仍要在全历史上打分，0.13 GiB 是**下界**。
   ⑤ ⛔⛔ 单位口径：v7 官方 **7.37 TB/s 是每「芯片」**，
      而本图按 **device** 算（v7 是 2 device / chip），所以用 3.685。
      这门课在 chip:device = 1:2 上栽过，这里显式写死。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2, wpx, wrap_rich)

W = 1400
GiB = 2 ** 30
ACT = 37e9 / GiB          # V3 每 token 激活 37B，原生 FP8 → 1 B/参数
WGT = 625.0               # GiB，全部权重（装得下要按这个算）
DEV = 94.74               # GiB / device，TPU v7（本课 AOT 工具链核过）
BWD = 7.37e12 / 2         # B/s per device ——&#160;官方 7.37 TB/s 是每 chip
SEQ, K = 131072, 2048     # 128K 上下文；DSA 每步固定读 k 个

PLAN = [                  # (名字, KV GiB, 颜色, 一句话)
    ("MHA", 488.00, RD, "反事实：假如 V3 用 MHA"),
    ("GQA-8", 30.50, OR, "8 组"),
    ("MQA", 3.81, PU, "1 组 ——&#160;最省，但质量塌"),
    ("MLA", 8.58, GR, "V3 实际用的"),
]


def dev_of(kv):
    return int(-(-(WGT + kv) // DEV))


def step(kv_read, nd):
    return (ACT + kv_read) * GiB / (nd * BWD)      # 秒


def main():
    assert abs(ACT - 34.46) < 0.01, ACT
    rows = []
    for nm, kv, col, note in PLAN:
        nd = dev_of(kv)
        t = step(kv, nd)
        rows.append((nm, kv, col, note, nd, ACT + kv, 100 * kv / (ACT + kv),
                     t * 1e3, 1 / t))
    mha, mla = rows[0], rows[3]
    # ⭐ 这张图的三个题眼，全部做成断言 —— 数一变就炸，不会静默走样
    assert 93.0 < mha[6] < 94.0, mha[6]                  # ① KV 占 93.4%
    assert 11 < mha[4] and mla[4] == 7, (mha[4], mla[4])  # 12 张 vs 7 张
    ratio = mha[7] / mla[7]
    assert 7.0 < ratio < 7.2, ratio                      # ② 只快 7.08 倍
    store = PLAN[0][1] / PLAN[3][1]
    assert 56.8 < store < 57.0, store                    # 而显存省了 56.9 倍
    saved = mha[4] - mla[4]                              # 省下来的卡，才是真收益
    assert saved == 5, saved
    kvr = 8.58 * K / SEQ                                 # DSA 每步真读的 KV
    tD = step(kvr, 7)
    gainD = mla[7] / (tD * 1e3)
    assert 1.2 < gainD < 1.3, gainD                      # ③ 只再快 1.24 倍
    assert abs(100.0 * K / SEQ - 1.5625) < 1e-9

    f = Fig(W, "一步 decode 要从 HBM 搬多少字节：MHA 里 93% 是一个人的 KV；"
               "显存省 56.9 倍只换到 7 倍速度，因为 MHA 被迫多用了卡；"
               "MLA 之后再上稀疏只再快 1.24 倍，瓶颈已经搬到权重上")
    yy = f.header(
        "代价 ——&#160;把 <tspan font-weight=\"700\">↓↓↓</tspan> 换成"
        "<tspan font-weight=\"700\">一步 decode 要搬多少字节</tspan>",
        "§九 那张表每一格都是箭头。可这门课自己的规矩是"
        "<tspan font-weight=\"700\">「问『省了多少』之前，先问『省的是哪一样』」</tspan>"
        " ——&#160;箭头恰恰回答不了这个。"
        "那就把同一批方案，<tspan font-weight=\"700\">一路换算到毫秒</tspan>。",
        legend=[(GY2, "权重（所有人共享，每步一样多）"),
                (RD, "这一个人的 KV"), (BL, "换算成时间")])

    # ══ ① 一步要搬两样 ════════════════════════════════════════════
    PH1 = 196
    top = f.panel(0, yy, W, PH1,
                  "① 吐一个字，HBM 上要走的是这两样",
                  GY2, tag="DeepSeek-V3 · 128K · 一个用户 · batch 1")
    for i, (ttl, num, sub_, col, tail) in enumerate([
        ("权重", "%.2f GiB" % ACT, "每 token 激活 37B，原生 FP8", GY2,
         "所有方案<tspan font-weight=\"700\">完全一样</tspan>，而且"
         "<tspan font-weight=\"700\">所有人共享</tspan>"),
        ("这个人的 KV", "随方案变", "下面那五根柱子的差别全在这儿", RD,
         "<tspan font-weight=\"700\">每人一份</tspan>，人越多、话越长，它越大"),
    ]):
        bx = 24 + i * 686
        f.box(bx, top + 26, 652, 130, "none", LINE, 9)
        f.t(bx + 22, top + 58, ttl, col, bold=True, size=20, cls="svglbl")
        f.t(bx + 22, top + 96, num, col, bold=True, size=30, cls="svglbl")
        f.t(bx + 22, top + 120, sub_, GY, size=15)
        yj = top + 144
        for r in wrap_rich(tail, 610, 15 * 1.12):
            f.t(bx + 22, yj, r, GY, size=15)
            yj += 22

    # ══ ② 五根柱子 ════════════════════════════════════════════════
    yy = top + PH1 + 26
    PH2 = 500
    top = f.panel(0, yy, W, PH2,
                  "② 一步要搬多少 ——&#160;<tspan font-weight=\"700\">灰色那一段"
                  "五根完全一样</tspan>，差别全在红色那一段", RD,
                  tag="柱高 ∝ 每步读的字节数")
    BASE = top + 306
    SCALE = 240.0 / rows[0][5]        # 最高那根占 240px
    BW_ = 150
    GAP = 36
    bars = rows + [("DSA", kvr, BL, "存着 8.58，每步只读 k＝2,048", 7,
                    ACT + kvr, 100 * kvr / (ACT + kvr), tD * 1e3, 1 / tD)]
    for i, (nm, kv, col, note, nd, rd, pct, ms, tps) in enumerate(bars):
        cx = 40 + i * (BW_ + GAP)
        hw = ACT * SCALE
        hk = kv * SCALE
        # ⭐ 权重那一段：五根柱子里必须完全一样 —— 同一个表达式算出来的
        f.box(cx, BASE - hw, BW_, hw, "#f1f3f4", GY2, 5)
        if hk >= 13:
            f.box(cx, BASE - hw - hk, BW_, hk, "#fce8e6", col, 5)
            f.t(cx + BW_ / 2, BASE - hw - hk / 2 + 6, "%.1f" % kv, col,
                bold=True, size=19, anchor="middle")
        else:
            f.box(cx, BASE - hw - 3, BW_, 3, col, col, 1)
            f.t(cx + BW_ / 2, BASE - hw - 12, "%.2f" % kv, col, bold=True,
                size=16, anchor="middle")
        f.t(cx + BW_ / 2, BASE + 26, nm, col, bold=True, size=20,
            anchor="middle", cls="svglbl")
        f.t(cx + BW_ / 2, BASE + 52, "每步读 %.2f GiB" % rd, INK, size=15,
            anchor="middle")
        f.t(cx + BW_ / 2, BASE + 76, "KV 占 %.1f%%" % pct, col, bold=True,
            size=17, anchor="middle")
        f.t(cx + BW_ / 2, BASE + 100, "要 %d 张 device" % nd, GY2, size=15,
            anchor="middle")
    RIGHT = 40 + 5 * BW_ + 4 * GAP
    f.line(40, BASE + 1, RIGHT, BASE + 1, LINE, 1.2, arrow=False)
    # ⛔⛔ 2026-09-14 第一版栽在这儿：这张图的标题写着「灰色那一段五根完全一样」，
    #   可 MHA 的 488 把刻度撑死了 ——&#160;34.46 GiB 换算下来只有 16 px，
    #   **那句论点在图上根本看不见**。截图一看就发现，几何 lint 一条都不报
    #   （它只查撞车和越界，不查「你想说的东西说没说出来」）。
    # ⭐ 判据：**论点必须有一个不受刻度摆布的视觉载体。**
    #   这里用一条横贯五根的虚线 ——&#160;五根都恰好顶到它，
    #   「一样高」就从「量高度」变成了「看齐不齐」，一眼的事。
    f.line(34, BASE - ACT * SCALE, RIGHT + 6, BASE - ACT * SCALE, GY2, 1.2,
           dash="5 4", arrow=False)
    f.t(RIGHT + 16, BASE - ACT * SCALE - 10,
        "权重 %.2f GiB" % ACT, GY, bold=True, size=16)
    f.t(RIGHT + 16, BASE - ACT * SCALE + 14,
        "五根都顶到这条线", GY2, size=15)
    f.t(RIGHT + 16, BASE - ACT * SCALE + 36,
        "一个字节不差", GY2, size=15)
    # ⭐ 把刻度的局限本身变成一条论据：后三根的红段在这个刻度下都不足 4px，
    #   只能标数字 ——&#160;而「画不出来」正是 488 有多离谱的直接证据。
    yj = BASE + 44
    for r in wrap_rich(
        "⚠️ 后三根的红段在这个刻度下<tspan font-weight=\"700\">画不出来</tspan>，"
        "只能标数字 ——&#160;<tspan font-weight=\"700\">"
        "而「画不出来」本身就是 %.0f 有多离谱的证据。</tspan>"
        % PLAN[0][1], 420, 15 * 1.12):
        f.t(RIGHT + 16, yj, r, GY2, size=15)
        yj += 22
    # ⛔ 2026-09-14 踩过：这两行原来把 93.4 / 6.6 手写死，还顺手写成了 `%%` ——
    #   而这个字符串**没有经过 % 格式化**，于是图上真的印出了「93.4%%」。
    #   ⭐ 判据：**凡是图上要出现的数，一律从上面那组计算里取，别手抄。**
    #     手抄的数不但会漂，还会把格式化的坑一起抄进来。
    f.t(24, top + PH2 - 56,
        "⭐⭐ 第一条读法：<tspan font-weight=\"700\">MHA 那一步里，"
        "%.1f%% 的字节是一个人的 KV</tspan> ——&#160;权重只占 %.1f%%。"
        "「KV cache 是瓶颈」这句话，到这儿才算画出来了。"
        % (mha[6], 100 - mha[6]), INK, size=17)
    f.t(24, top + PH2 - 30,
        "⚠️ 顺手一条：<tspan font-weight=\"700\">MQA 比 MLA 还省</tspan>"
        + "（%.2f vs %.2f）——&#160;它从来不是慢，它是"
        % (PLAN[2][1], PLAN[3][1])
        + "<tspan font-weight=\"700\">质量塌</tspan>。省字节和能不能用是两回事。",
        GY, size=16)

    # ══ ③ 换算成毫秒 ══════════════════════════════════════════════
    yy = top + PH2 + 26
    PH3 = 330
    top = f.panel(0, yy, W, PH3,
                  "③ 除以带宽 ——&#160;字节变毫秒，"
                  "<tspan font-weight=\"700\">这里才出现两条反直觉的</tspan>", BL,
                  tag="v7 每 device 3.685 TB/s")
    f.box(24, top + 28, 664, 264, "none", LINE, 9)
    f.t(44, top + 58, "一步要多久（下界）", BL, bold=True, size=20,
        cls="svglbl")
    for j, (nm, kv, col, note, nd, rd, pct, ms, tps) in enumerate(bars):
        yj = top + 96 + j * 38
        f.t(44, yj, nm, col, bold=True, size=17)
        f.t(168, yj, "%.2f GiB ÷ %d 张" % (rd, nd), GY, size=15, mono=True)
        f.t(400, yj, "%.2f ms" % ms, INK, bold=True, size=18, mono=True)
        f.t(510, yj, "%.0f tok/s" % tps, col, size=16, mono=True)
    f.t(44, top + 286, "⚠️ 只算 HBM 读 ——&#160;真机只会更慢，"
        "但各方案之间的比例站得住", GY2, size=15)

    f.box(712, top + 28, 664, 128, "none", RD, 9)
    f.t(732, top + 58,
        "反直觉一：显存省了 %.1f 倍，只快了 %.2f 倍" % (store, ratio),
        RD, bold=True, size=19, cls="svglbl")
    yj = top + 88
    for r in wrap_rich(
        "⛔ 差在哪？<tspan font-weight=\"700\">MHA 装不下，被迫用 %d 张 device "
        "而不是 %d 张</tspan> ——&#160;它拿显存换来的卡，顺手也把带宽换来了。"
        % (mha[4], mla[4]) +
        "⭐ 所以 MLA 真正省下的不是时间，是<tspan font-weight=\"700\">"
        "那 %d 张卡</tspan>，而它们可以拿去服务别人。" % saved,
            624, 16 * 1.12):
        f.t(732, yj, r, GY, size=16)
        yj += 23

    f.box(712, top + 168, 664, 128, "none", BL, 9)
    f.t(732, top + 198,
        "反直觉二：MLA 之后再上稀疏，只再快 %.2f 倍" % gainD, BL, bold=True,
        size=19, cls="svglbl")
    yj = top + 228
    for r in wrap_rich(
        "⭐⭐ 因为<tspan font-weight=\"700\">瓶颈已经搬到权重那一段</tspan>："
        "读 %.2f，KV 只剩 %.2f。这不是稀疏没用 ——&#160;"
        "这正是上面表里「DSA：prefill 为主」的<tspan font-weight=\"700\">"
        "数值版</tspan>。" % (ACT, kvr), 624, 16 * 1.12):
        f.t(732, yj, r, GY, size=16)
        yj += 23

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH3 + 30
    yy = f.band(yy, "ok", "「先问省的是哪一样」，现金价值就在这三行里", [
        "<tspan font-weight=\"700\">省显存 ≠ 省时间。</tspan>"
        "MLA 对 MHA：显存 <tspan font-weight=\"700\">%.1f×</tspan>，"
        "单用户 decode 只有 <tspan font-weight=\"700\">%.2f×</tspan> ——&#160;"
        "省下来的那 %d 张卡才是真正的收益。" % (store, ratio, saved),
        "<tspan font-weight=\"700\">省读 ≠ 省存。</tspan>"
        "DSA 那根柱子的灰段红段一个字节没少存，"
        "<tspan font-weight=\"700\">它只是每步不读</tspan> ——&#160;"
        "所以它在这张图上只影响高度，不影响「要几张卡」。",
        "⭐⭐ <tspan font-weight=\"700\">瓶颈是会搬家的。</tspan>"
        "MHA 时代它在 KV（%.1f%%）；MLA 之后它搬到权重（%.1f%%）。"
        "<tspan font-weight=\"700\">对着上一个瓶颈继续优化，是最常见的浪费。</tspan>"
        % (mha[6], 100 - mla[6]),
    ])
    yy = f.band(yy + 14, "warn", "⚠️ 这张图只成立在 batch ＝ 1 上，这一条必须说", [
        "权重那一段是<tspan font-weight=\"700\">所有人分摊</tspan>的，"
        "KV 那一段<tspan font-weight=\"700\">不摊</tspan>。"
        "所以人一多，灰段被摊薄、红段成倍长 ——&#160;"
        "<tspan font-weight=\"700\">画面会翻回 KV 主导</tspan>。",
        "⭐ 这正是 §二 那张图的另一面：那里是"
        "<tspan font-weight=\"700\">把人加上去</tspan>让 KV 变成主角，"
        "这里是<tspan font-weight=\"700\">只留一个人</tspan>让权重变成主角。"
        "<tspan font-weight=\"700\">同一个模型，问法不同，答案就不同</tspan> ——&#160;"
        "这本身就是本节的主题。",
    ])
    yy = f.band(yy + 14, "bad", "⛔ 口径，三条", [
        "<tspan font-weight=\"700\">这是下界，不是实测。</tspan>"
        "只算从 HBM 把字节读进来的时间，"
        "不算计算、不算 MoE 的 all-to-all、不算 kernel 损耗。"
        "⚠️ 卡数只按<tspan font-weight=\"700\">装得下</tspan>算，没算算力够不够。",
        "DSA 那根的 <tspan font-weight=\"700\">%.2f GiB 也是下界</tspan> ——&#160;"
        "索引器仍要在全历史上打一遍分，那部分虽轻但不是零。"
        "k＝2,048 是 128K 上的设定（占 %.4f%%）。" % (kvr, 100.0 * K / SEQ),
        "⛔⛔ 单位：v7 官方 <tspan font-weight=\"700\">7.37 TB/s 是每「芯片」"
        "</tspan>，而 v7 是 <tspan font-weight=\"700\">2 device / chip</tspan>，"
        "本图一律按 device 算，所以用 <tspan font-weight=\"700\">3.685 TB/s"
        "</tspan>、每 device <tspan font-weight=\"700\">94.74 GiB</tspan>。"
        "这门课在这个 1:2 上栽过，所以写死在这儿。",
    ])
    yy = f.src(yy + 16,
               "装置偷自 Epoch AI 那条「字节 →&#160;毫秒 →&#160;钱」的换算链 ——&#160;"
               "抽象的「省了多少」一路换算到有直觉的量，每一步都能自己验算。"
               "本图走到毫秒为止（再往下换成钱要报价，那个本课核不了）",
               "⚠️ 数全部来自本讲前面已核过的：KV 488 / 30.50 / 3.81 / 8.58 GiB"
               "（61 层 · 128K · bf16 · 一个用户）· 权重 625 GiB · "
               "每 token 激活 <tspan font-weight=\"700\">37B</tspan>"
               "（671B/37B，官方模型卡，见 §零 那张年表）· "
               "v7 每 chip 7.37 TB/s、每 device 94.74 GiB",
               "📌 这张图<tspan font-weight=\"700\">不引入任何新数</tspan>，"
               "全部是前面核过的数做除法 ——&#160;所以读者可以拿计算器逐格核，"
               "这也是它敢把 §九 那张箭头表替换掉的底气")
    f.save("fig3-perstep.svg", yy + 6)


main()
