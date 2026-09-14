# -*- coding: utf-8 -*-
r"""专题三 · 总纲图 —— **这是一个什么故事，怎么从过去走到现在的**。

⭐⭐ 2026-09-08 现场把这一讲的立意换掉了，原话：

    「先把骨架打好，把故事线捋清楚了 ——&nbsp;这是一个什么样的故事？
      怎么样向前推进的？怎么样从过去发展到现在的？每一次变革都是图啥？
      带来了什么？以及**对现在这个 agentic 的意义 ——&nbsp;
      长上下文才带来了智能，原来 2K 的上下文玩个毛？**」

⛔ **旧立意是「分类学：只有三个旋钮」** ——&nbsp;那是一把尺子，不是一个故事。
   尺子能让人认名词，但**回答不了「为什么这六年非得这么走」**。

⭐⭐ **新立意一句话：**

     **上下文长度就是 agent 的工作记忆。**
     2K 的时候，模型只能当个聪明的补全器；
     要读整个代码库、跑几十轮工具调用、记住整场对话，**先得记得住**。
     而每加长一分上下文，**KV cache 就线性涨一分**。
     → **所以这六年注意力的全部演化，是为了让「记得住」这件事付得起。**

⭐ 于是每一步变革都能挂回同一个问句：**它让「记得住且付得起」前进了哪一步？**

════════════════════════════════════════════════════════════════════
📌 图上每个数的出处（都在本课别处已经核过，这里只是汇总）
════════════════════════════════════════════════════════════════════

· **2K**：GPT-3 的 `max_position_embeddings` ＝ 2048（本课模型表第一行）
· **1M**：本课模型表里做到 1M 以上的有 **12 家**（口径见 topic03_models.over_1m：
  声明值 ≥1M；RWKV-7 的「无限（理论）」不计入 —— 算上它是 13 家，结论不变），
  全部动了旋钮②或③
· **576 GiB → 697 MiB ＝ 846 倍**：同一个 128K 长度、BF16、batch 1，
  两端都取 **≥100B** 的模型（GPT-3 ／ DeepSeek-V4-Flash）。见模型表落点⑤
· **488 GiB**：V3 形状假想成纯 MHA、128K 时的 KV cache（本课 §二）
· ⚠️ **512 倍**（2K → 1M）和 **846 倍**（KV 降幅）**是两个不同口径的数**：
  前者是「能跑多长」，后者是「同一长度下省了多少」。
  ⛔ **不能相乘**，也不能说成「一共 43 万倍」——&nbsp;那是把两把尺子当成一把。
"""
import topic03_models as M
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, BG2)


import math
import re


def _mon(d):
    """"2024-07" →&#160;从 2020-05 起算的第几个月。"""
    yy, mm = int(d[:4]), int(d[5:7])
    return (yy - 2020) * 12 + mm - 5


def _ctx(s):
    """模型表的上下文列 →&#160;token 数。RWKV 那行写「无限（理论）」，返回 None。"""
    if "无限" in s:
        return None
    v = float(re.sub(r"[^\d.]", "", s))
    return v * 1024 if "K" in s else (v * 1048576 if "M" in s else v)


def _params_b(name):
    """"Kimi K3　2.8T/104B · 93 层" →&#160;2800（总参数，单位 B）。"""
    m_ = re.match(r"([\d.]+)\s*([BT])", name.split("　")[1])
    return float(m_.group(1)) * (1000 if m_.group(2) == "T" else 1)


def _series():
    """从模型表现算两组点。⛔ 一个数都不许手写 ——&#160;全部来自 topic03_models.ROWS。

    ⛔⛔ **右边那张图必须过滤 ≥100B** ——&#160;这是本图「846 倍」一直以来的口径
    （见文件头出处：两端都取 ≥100B 的模型）。第一版没过滤，于是包络线在
    2023-09 就被 **Mistral 7B 的 512 MiB** 拽到底了 ——&#160;
    ⭐ 一个 7B 模型的 KV 当然小，那跟「省法」没关系，是它本来就小。
    **图当时正好画在那句「576 GiB → 697 MiB」旁边，自己反驳自己。**
    ⭐ 判据：**同一张图里两个数字，口径必须是同一个**；不同口径就得分两张图。
    （左图不过滤 ——&#160;上下文长度跨尺寸可比，7B 能跑 32K 就是能跑 32K。）
    """
    up, dn = [], []
    for date, name, _loop, c, kv, _n in M.ROWS:
        who = name.split("　")[0].replace("⭐ ", "")
        t, v = _mon(date), _ctx(c)
        if v:
            up.append((t, v, who))
        g = M.kv_gib(kv)
        if g and _params_b(name) >= 100:
            dn.append((t, g, who))
    return up, dn


def _two_curves(f, y):
    """两张同 x 轴的小图：左边「能跑多长」往上走，右边「同一长度的代价」往下走。"""
    PW, H, GAP = 676, 330, 48       # 两张并排，一眼看见方向相反
    XL, XR = 74, PW - 26            # 绘图区左右边界（面板内坐标）
    up, dn = _series()
    TMAX = max(t for t, _, _ in up + dn)
    assert TMAX == _mon("2026-08"), "模型表末行不是 2026-08 了，x 轴范围要跟着改"
    # ⭐ 把落点里那个「846 倍」钉在画出来的包络线两端上 ——&#160;
    #   以后模型表增删行，只要这两端变了，这里当场报，不会出现
    #   「图上画的是一条线、旁边写的是另一个数」。
    _r = max(g for _, g, _ in dn) / min(g for _, g, _ in dn)
    assert 845 < _r < 847, "右图包络两端的比值 %.0f≠846，落点那句要跟着改" % _r
    # ⛔ 左图绿线的终点是 10M，不是 1M ——&#160;本讲一直说的「512 倍」是
    #   2K→1M 那个**常态**值，不是纪录。两个数都对，但**不是同一条线上的**，
    #   所以纪录写在图里、常态写在图下那行，谁也别冒充谁。
    assert max(v for _, v, _ in up) == 10 * 1048576, "左图纪录不再是 10M 了"
    assert 1048576 / 2048 == 512

    def draw(px, title, sub, pts, lo, hi, ticks, best, note, anchors):
        top = f.panel(px, y, PW, H, title, GR, "#fff", sub=sub)
        y0, y1 = top + 26, y + H - 66          # 绘图区上下边界
        sx = lambda t: px + XL + (XR - XL) * t / TMAX
        sy = lambda v: y1 - (y1 - y0) * (math.log(v / lo) / math.log(hi / lo))
        for tv, lab in ticks:                  # 横向参考线 ＋ 刻度（对数）
            gy = sy(tv)
            f.line(px + XL, gy, px + XR, gy, "#e8eaed", 1.0, arrow=False)
            f.t(px + XL - 10, gy + 5, lab, GY2, size=14, anchor="end")
        for yy_ in (2021, 2022, 2023, 2024, 2025, 2026):
            gx = sx(_mon("%d-01" % yy_))
            f.t(gx, y1 + 26, str(yy_), GY2, size=14, anchor="middle")
        f.line(px + XL, y1, px + XR, y1, LINE2, 1.2, arrow=False)
        # 包络线：跑到这个月为止的最好成绩 ——&#160;是数出来的，不是拟合的
        run, seq = None, []
        for t, v, _ in sorted(pts):
            run = v if run is None else (max(run, v) if best == "max"
                                         else min(run, v))
            seq.append((t, run))
        d = "M %.1f %.1f" % (sx(seq[0][0]), sy(seq[0][1]))
        for i in range(1, len(seq)):           # 阶梯线：成绩是被某一家一次性刷新的
            d += " L %.1f %.1f L %.1f %.1f" % (
                sx(seq[i][0]), sy(seq[i - 1][1]), sx(seq[i][0]), sy(seq[i][1]))
        d += " L %.1f %.1f" % (sx(TMAX), sy(seq[-1][1]))
        f.path(d, GR, 3.0, arrow=False)
        named = {a[0] for a in anchors}        # ⛔ 别写 who in anchors：那是元组列表
        assert named <= {w for _, _, w in pts}, "要点名的模型不在这组点里"
        for t, v, who in pts:                  # 每个模型一个点
            hit = who in named
            f.box(sx(t) - (6 if hit else 4), sy(v) - (6 if hit else 4),
                  12 if hit else 8, 12 if hit else 8, RD if hit else GY2,
                  "none", 6)
        for who, dx, dy, al in anchors:
            t, v = next((t, v) for t, v, w in pts if w == who)
            f.t(sx(t) + dx, sy(v) + dy, who, RD, True, 14, al)
        f.t(px + XL - 46, y + H - 14, note, GY, size=15)
        return top

    A_UP = [("GPT-3", 12, -12, "start"), ("Llama 4 Scout", -12, -12, "end"),
            ("Kimi K3", 0, -16, "middle")]
    A_DN = [("GPT-3", 14, 20, "start"), ("PaLM", 12, 24, "start"),
            ("DeepSeek-V4-Flash", -12, -14, "end"),
            ("混元 Hy3", -12, -12, "end")]     # ⭐ 反例：2026 年仍停在 40 GiB
    draw(0, "能跑多长　↗", "每个点是一个模型声明的上下文上限",
         up, 2048, 10 * 1048576,
         [(2048, "2K"), (32768, "32K"), (131072, "128K"),
          (1048576, "1M"), (10485760, "10M")],
         "max", "⭐ 绿线＝当时的纪录：2K →&#160;10M（Llama 4 Scout 声明值，之后没人再刷）",
         A_UP)
    draw(PW + GAP, "同一长度的代价　↘",
         "每个点是那个模型在 128K 时的 KV cache（BF16、batch 1）",
         dn, 0.62, 640,
         [(576, "576 GiB"), (64, "64 GiB"), (8, "8 GiB"),
          (0.6807, "697 MiB")],
         "min", "⭐ 绿线＝当时的最省记录。576 GiB →&#160;697 MiB，846 倍",
         A_DN)
    f._pan = None
    f.t(0, y + H + 26,
        '⛔ <tspan font-weight="700">不是所有人都在走</tspan> ——&#160;'
        '右图 2026 年还有模型停在 40 GiB（混元 Hy3，纯 GQA）。'
        '<tspan font-weight="700">动的是「最好成绩」那条线，不是每一家。</tspan>'
        '<tspan x="0" dy="26">⚠️ </tspan>'
        '<tspan font-weight="700">纪录也不等于常态</tspan>：左图那条线的终点是 '
        '<tspan font-weight="700">10M</tspan>（一家的声明值）；'
        '本讲说的 <tspan font-weight="700">512 倍</tspan> 是 2K →&#160;'
        '<tspan font-weight="700">1M</tspan> 这个常态 ——&#160;模型表里 %d 家做到。'
        % len(M.over_1m(M.ROWS)),
        GY, size=_sz(15))
    return y + H + 56


def fig_arc():
    W = 1400
    f = Fig(W, "专题三的故事线：从 RNN 到今天的混合注意力，六个阶段，"
               "每个阶段各自图什么、带来了什么、欠下什么；"
               "落点是长上下文让 agent 成为可能")
    f.marks = set()
    y = f.header(
        '这是一个什么故事 ——&#160;'
        '<tspan font-weight="700">六年时间，把「记得住」变成一件付得起的事</tspan>',
        '⭐ <tspan font-weight="700">上下文长度就是 agent 的工作记忆。</tspan>'
        '2K 的时候模型只能当个聪明的补全器；要读整个代码库、跑几十轮工具调用、'
        '记住整场对话，<tspan font-weight="700">先得记得住</tspan>。'
        '而每加长一分上下文，<tspan font-weight="700">KV cache 就线性涨一分</tspan>。',
        [(RD, "还没有 KV cache 这回事"), (BL, "让每一份更小"),
         (OR, "每步只读一部分"), (PU, "换回一个固定大小的状态")])

    # ── 六个阶段：每一格先给一个小画面，再给三句短话 ─────────────
    # ⭐⭐⭐ 2026-09-13 重画：原来是六张**文字卡**（每张六行字）。
    #   现在每格上半是**图**、下半只留三行 ——&nbsp;而且每个小画面
    #   **复用后面各节已经立起来的比喻**（一摞、只亮几格、换成一块板子、
    #   几个普通配一个资深），前后对得上。
    ST = (
        ("1990–2017", "RNN", RD, "chain",
         "带一个固定大小的状态", "序列建模第一次可行",
         "⛔ 串行：算不快、记不住"),
        ("2017", "MHA", GY, "table",
         "把循环整个拿掉", "训练能并行，规模才起得来",
         "⛔ 状态没了 → KV cache 出生"),
        ("2019–2024", "旋钮①", BL, "thin",
         "让每一份更小", "2K → 128K 成常态",
         "⛔ 砍太狠会掉质量"),
        ("2023–2026", "旋钮②", OR, "few",
         "每步只读一部分", "1M 进入可用区间",
         "⛔ 省读不省存"),
        ("2020–2026", "旋钮③", PU, "board",
         "换回一块固定大小的板子", "那些层的 KV 归零",
         "⛔ 串行跟着回来了"),
        ("2024–今天", "混合", GR, "team",
         "两头都要", "3:1 ～ 7:1 成了共识",
         "⭐ 便宜的管长度，贵的管质量"),
    )
    # ⭐⭐⭐ 2026-09-13 第三刀：**六列 224px 是塞不下大字的**。
    #   前一版已经把文字卡换成了小画面，可六格横排，每格只有 224px ——
    #   于是画面只有 28px 见方、字只能给到 11.5px，**投到屏幕上什么都看不见**。
    # ⛔ 判据要改：不是「这一格里能不能放下」，是「**这一格该有多宽**」。
    #   格数是内容定的（六个阶段），宽度是可读性定的 —— 那就换行，不要压字。
    # ⭐ 改成 2 行 × 3 列：每格 452px，画面放大一倍，字从 11.5 抬到 17–27。
    CW, GAP, VGAP, BODY = 452, 22, 20, 256
    y0 = y
    for i, (era, name, col, kind, what, got, owe) in enumerate(ST):
        r, c = divmod(i, 3)
        x = c * (CW + GAP)
        yy = y0 + r * (BODY + VGAP)
        f.box(x, yy, CW, BODY, "#fff", LINE, 10)
        f.box(x, yy, CW, 6, col, col, 3)
        f.box(x, yy + 4, CW, 8, "#fff", "#fff", 0)
        f.t(x + 22, yy + 34, era, GY2, size=15)
        f.t(x + 22, yy + 68, name, col, True, 27)

        # ── 小画面：每格一个，复用后面各节已经立起来的比喻 ──────────
        gx, gy_ = x + 22, yy + 84
        if kind == "chain":                     # 一排人传话
            for k in range(5):
                f.box(gx + k * 62, gy_ + 10, 46, 42, "#fce8e6", col, 6)
                f.t(gx + 23 + k * 62, gy_ + 38, str(k + 1), col, True, 17,
                    "middle")
                if k < 4:
                    f.line(gx + 48 + k * 62, gy_ + 31, gx + 62 + k * 62,
                           gy_ + 31, col, 1.6)
        elif kind == "table":                   # 人人都看得见人人
            for rr in range(5):
                for cc in range(5):
                    if cc <= rr:
                        f.box(gx + cc * 34, gy_ + 4 + rr * 10, 30, 8,
                              "#e8eaed", "none", 2)
            f.t(gx + 190, gy_ + 34, "谁都能看见谁", GY2, size=15)
        elif kind == "thin":                    # 每格还在，里面的东西变小
            for k in range(5):
                f.box(gx + k * 62, gy_ + 10, 46, 42, BG2, LINE2, 6)
                f.box(gx + 14 + k * 62, gy_ + 24, 18, 14, col, "none", 3)
            f.t(gx + 4, gy_ + 72, "格子没少，每格里的东西变小", GY2, size=15)
        elif kind == "few":                     # 格子照样在，只读其中几个
            for k in range(5):
                on = k in (1, 3)
                f.box(gx + k * 62, gy_ + 10, 46, 42,
                      col if on else BG2, "none" if on else LINE2, 6)
            f.t(gx + 4, gy_ + 72, "格子照样在，这一步只读两个", GY2, size=15)
        elif kind == "board":                   # 一长排换成一块板子
            for k in range(3):
                f.box(gx + k * 38, gy_ + 10, 30, 42, BG2, LINE2, 5)
            f.line(gx + 120, gy_ + 31, gx + 148, gy_ + 31, col, 2.0)
            f.box(gx + 156, gy_ + 6, 118, 50, "#f3e8fd", col, 7)
            f.t(gx + 215, gy_ + 38, "一块板子", col, True, 17, "middle")
        elif kind == "team":                    # 几个普通配一个资深
            for k in range(4):
                pro = (k == 3)
                f.box(gx + k * 62, gy_ + 10, 46, 42,
                      "#e8f0fe" if pro else "#e6f4ea", BL if pro else GR, 6)
                f.t(gx + 23 + k * 62, gy_ + 38, "资深" if pro else "普通",
                    BL if pro else GR, True, 15, "middle")
            f.t(gx + 4, gy_ + 72, "三个便宜的配一个贵的", GY2, size=15)

        f.t(x + 22, yy + 186, what, col, True, 19, w=CW - 44)
        f.t(x + 22, yy + 214, got, GY, size=17, w=CW - 44)
        f.t(x + 22, yy + 242, owe, GY2, size=17, w=CW - 44)
        # 同一行内接上一格；换行处画一个折回标记
        if c:
            f.line(x - GAP - 2, yy + 40, x - 2, yy + 40, GY2, 1.6)
        elif r:
            f.t(0, yy - 8, "↳ 接着上一行", GY2, size=15)
    y = y0 + 2 * BODY + VGAP + 18

    # ── 两条曲线：⛔ 这里以前只有一行字说「两条曲线反着走」，图上一条曲线都没有。
    #   现在画真数据 ——&#160;本课模型表 44 行，每一行都有发布月份、声明上下文、
    #   以及按 config 算出来的 128K KV cache。⛔ 不画平滑趋势线：那是编的。
    #   画的是 ① 每个模型一个点 ② 「当时的最好成绩」包络线（跑最大值／最小值，
    #   是从同一份数据里跑出来的，不是拟合的）。
    y = _two_curves(f, y) + 14

    # ── 落点 ────────────────────────────────────────────────────
    y = f.band(y, "info", "两条曲线反着走 ——&#160;这才是这六年真正发生的事", [
        '⛔ 这两条<tspan font-weight="700">是两把尺子，不能相乘</tspan>：'
        '左边量「能跑多长」，右边量「同一长度下省了多少」。'
        '把 512 倍 × 846 倍说成四十万倍，是把两把尺子当成一把。',
        '⭐⭐ <tspan font-weight="700">但它们同时发生，才有今天的 agent</tspan>'
        '——&#160;上下文能装下整个代码库，而且装得起。'])

    y = f.band(y + 14, "ok", "所以这一讲的前半程只有一个账本：KV cache（⚠️ 到旋钮② 它就不够用了）", [
        '⭐ 在张量形状里找 <tspan font-weight="700">S</tspan>（KV 长度）——&#160;'
        '全图只有 K 和 V 两处带它，<tspan font-weight="700">那就是唯一要跨 token 留下来的东西</tspan>。',
        '三个旋钮是它的三个面：<tspan fill="%s" font-weight="700">① 每份多大</tspan>　'
        '<tspan fill="%s" font-weight="700">② 每步读多少</tspan>　'
        '<tspan fill="%s" font-weight="700">③ 干脆别让它变长</tspan>' % (BL, OR, PU),
        '⛔ 而 <tspan font-weight="700">FlashAttention 不在这三条里</tspan>——&#160;'
        '它已经是标配，在哪种注意力下都一样，<tspan font-weight="700">对这一讲没有区分度</tspan>。'])

    y = f.src(y + 18,
              '2K ＝ GPT-3 的 max_position_embeddings；1M 与 846 倍见本课模型表'
              '（128K、BF16、batch 1，两端都取 ≥100B 的模型）',
              '⚠️ 512 倍（2K→1M）与 846 倍（KV 降幅）是两个口径，'
              '本图刻意分两行给出 ——&#160;把它们相乘是把两把尺子当成一把',
              '两张趋势图的点全部由 topic03_models.ROWS 现算，没有手写常数；'
              '绿线是「跑到当月为止的最好成绩」，不是拟合曲线',
              '右图只收 ≥100B 的模型 ——&#160;跟「846 倍」同一个口径。'
              '不过滤的话包络线会被 Mistral 7B 的 512 MiB 拽到底，'
              '而那只是因为它本来就小',
              'RWKV-7 的 KV ＝ 0，对数轴上画不出来，没有进右图')
    f.save("fig3-arc.svg", y + 6)


fig_arc()
