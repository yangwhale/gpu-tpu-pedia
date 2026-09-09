# -*- coding: utf-8 -*-
r"""外传 图 X-12 · **五个月，十个模型** ——&nbsp;这条路实际是怎么走过来的。

════════════════════════════════════════════════════════════════════
⭐ 这张图为什么值得画进一门讲硬件的课
════════════════════════════════════════════════════════════════════
前面五节讲的都是**应该怎样**：屋脊点该怎么算、活该落在哪一侧、
流水线该在哪儿切。这一张讲**实际怎样** ——&nbsp;
同一批道理在十个真模型上跑了五个月之后，留下了什么。

⭐⭐ 它回答一个学员一定会问、而纯理论回答不了的问题：
   **「这套说法你们自己验过吗？」**

════════════════════════════════════════════════════════════════════
📌 数从哪来：全部是 git 提交历史，不是回忆
════════════════════════════════════════════════════════════════════
每一行的起止日期与提交数，都由下面这条命令直接数出来：

    git log --diff-filter=A --format='%ad' --date=short --reverse -- <目录>
    git log --oneline -- <目录> | wc -l

⭐ 这一点本身值得对学员说一句：**项目时间线不要凭印象写。**
  印象里「先做了 Wan 再做混元」，而提交历史说的是反过来 ——&nbsp;
  HunyuanVideo 早了整整一周，而且提交数是全场最多的 73 次。

════════════════════════════════════════════════════════════════════
⛔ 三代的分界不是我们事后追认的，它在文件名里
════════════════════════════════════════════════════════════════════
2025-12-10 那一天的提交里有这么几条：

    Rename files: remove flax suffix from filenames
    Remove dit_flax.py and dit_gpu.py
    Remove vae_decode_flax.py

⭐ **手写 Flax 那一版是在那天被删掉的**，不是慢慢淡出的。
  所以「第三代从 12-10 开始」这句话有一个硬证据，而不是一个叙事。
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400

# 起点：2025-12-01 记作第 0 天
DAY0 = (2025, 12, 1)


def d(y, m, day):
    """日期 → 距 2025-12-01 的天数。⛔ 不用 datetime.now()，全是字面量，结果可复现。"""
    import datetime
    return (datetime.date(y, m, day) - datetime.date(*DAY0)).days


# (名字, 任务, 规模, 起, 止, 提交数, 备注)
MODELS = (
    ("HunyuanVideo-1.5", "文生视频", "8.3B", d(2025, 12, 3), d(2025, 12, 27), 73,
     "第一个吃螃蟹的，提交数全场最多"),
    ("Wan 2.1 T2V", "文生视频", "14B", d(2025, 12, 10), d(2026, 3, 3), 44,
     "720P 从 OOM 到 229 秒"),
    ("CogVideoX 1.5", "文生视频", "5B", d(2025, 12, 12), d(2026, 3, 3), 49,
     "720P 106 秒"),
    ("Wan 2.2 I2V", "图生视频", "27B/14B", d(2025, 12, 14), d(2026, 3, 3), 26,
     "MoE，两颗起"),
    ("Flux.2", "文生图", "—", d(2025, 12, 29), d(2026, 2, 11), 11,
     "一上来就是三代打法，不走弯路"),
    ("ComfyUI on TPU", "图形界面", "4 个节点", d(2026, 1, 3), d(2026, 1, 6), 28,
     "给不写代码的人用"),
    ("SDXL", "文生图", "3.5B", d(2026, 2, 11), d(2026, 2, 11), 9,
     "补上最经典的 UNet，单颗 3 秒"),
    ("S3Diff", "单步超分", "3.3B", d(2026, 3, 6), d(2026, 3, 14), 11,
     "编译后 5.5 倍"),
    ("Flux.1", "文生图", "12B", d(2026, 3, 13), d(2026, 3, 13), 2,
     "只交一体化版"),
    ("Real-ESRGAN", "超分", "8.8M", d(2026, 4, 2), d(2026, 4, 3), 3,
     "纯卷积网 —— 验证非 Transformer 路径"),
)

KIND = {"文生视频": BL, "图生视频": CY, "文生图": PU, "单步超分": GR,
        "超分": GR, "图形界面": OR}

# (起, 止, 名字, 色, 浅色)
# ⛔⛔ 这三条的日期改过一次 ——&nbsp;初版把第三代画在 12-29 起，
#   而同一个文件的文件头写着「第三代从 12-10 开始」（依据是那天删掉了手写 Flax）。
#   **图和它自己的依据打架，而两处都是我写的。**
#   ⭐ 判据：**凡是「某某从某天开始」这种断言，图上的坐标必须直接由那条依据算出来，
#     不要另手填一个看起来差不多的日期。** 现在两处都锚在 d(2025,12,10)。
#
# ⚠️ 第一代**没有独立的日期区间**：maxdiffusion 是我们的起点/基线，
#   仓库里没有一段「只用它」的时期（12-03 第一天就已经在写 Flax 版了）。
#   ⛔ 硬给它画两天的带是**编造精度** ——&nbsp;改成一个起点标记，并在图上说明。
_G3 = d(2025, 12, 10)                     # ← 唯一真源：flax 后缀集体删除那天
GENS = (
    (d(2025, 12, 3), _G3, "第二代 · 手写 Flax / 纯 JAX", OR, "#feefc3"),
    (_G3, d(2026, 4, 15), "第三代 · torchax ＋ diffusers-tpu", GR, "#e6f4ea"),
)
GEN0 = "第一代 · 官方 maxdiffusion —— 起点，无独立区间"

# (日期, 文案)
MARKS = (
    (d(2025, 12, 5), "三阶段首次出现"),
    (_G3, "⭐ flax 后缀集体删除 —— 第三代从这天算起"),
    (d(2025, 12, 14), "优化完全指南成稿"),
)


def main():
    f = Fig(W, "十个模型的适配时间线，从 2025 年 12 月 3 日到 2026 年 4 月 3 日。"
               "顶部三条色带标出三代移植方法的更替，其中 12 月 10 日是分水岭 —— "
               "手写 Flax 的文件在那天被删除，torchax 成为唯一主线")
    f.marks = set()
    y = f.header(
        '五个月，十个模型 ——&#160;'
        '<tspan font-weight="700">而三代方法的分界，写在文件名里</tspan>',
        '⭐ 每一行的起止和提交数都是 <tspan font-weight="700">git 提交历史直接数出来的</tspan>，'
        '不是凭印象写的 ——&#160;印象里最早做的是 Wan，'
        '<tspan font-weight="700">而历史说是 HunyuanVideo，早了整整一周</tspan>。',
        [(BL, "视频生成"), (PU, "图像生成"), (GR, "超分 / 修复"), (OR, "交付形态")])

    AX0, AX1 = 210, W - 350
    LO, HI = 0, d(2026, 4, 15)

    def xf(t):
        return AX0 + (AX1 - AX0) * (t - LO) / float(HI - LO)

    # ══ 顶部：三代色带（背景层先画）══
    # ⛔ 第一代只活了 9 天，色带宽度 ≈ 42px，而它的标签要 130px ——&nbsp;
    #   居中放会直接压到隔壁那条和左边的「移植方法」上（护栏抓到了）。
    # ⭐ 判据：**放不下就别硬放。**装得下的居中放在带内，装不下的挪到带上方、
    #   左对齐并画一根引线 ——&nbsp;这样窄带的标签有多长都不会撞。
    gy = y + 46
    _out = [0]                                # 已挪到带外的标签数
    for t0, t1, nm, col, fill in reversed(GENS):
        bw = xf(t1) - xf(t0)
        f.box(xf(t0), gy, bw, 26, fill, col, 5, 1.2)
        if wpx(nm, 11.5) + 16 <= bw:
            f.t(xf(t0) + bw / 2.0, gy + 18, nm, col, bold=True,
                size=_sz(11.5), anchor="middle")
        else:
            oy = gy - 10 - _out[0] * 18          # 挪出去的逐条错开一行
            f.t(xf(t0), oy, nm, col, bold=True, size=_sz(11.5))
            f.line(xf(t0) + 2, oy + 4, xf(t0) + 2, gy, col, 1.2, arrow=False)
            _out[0] += 1
    f.t(AX0 - 12, gy + 18, "移植方法", GY, bold=True, size=_sz(11.5), anchor="end")
    # ⚠️ 第一代只标一个起点，不画带 ——&nbsp;它在仓库里没有独立的日期区间。
    f.box(xf(0) - 4, gy, 8, 26, "#fce8e6", RD, 3, 1.2, dash="3 2")
    f.t(xf(0) + 10, gy - 10 - _out[0] * 18, GEN0, RD, bold=True, size=_sz(11.5))
    f.line(xf(0) + 6, gy - 6 - _out[0] * 18, xf(0) + 6, gy, RD, 1.2, arrow=False)

    top = gy + 44
    ROW = 40

    # 竖直分月线
    for mo in ((2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4)):
        t = d(mo[0], mo[1], 1)
        f.line(xf(t), top - 8, xf(t), top + len(MODELS) * ROW + 6,
               LINE2, 1, dash="3 5", arrow=False)
        f.t(xf(t) + 6, top - 14, "%d 年 %d 月" % mo if mo[1] in (12, 1)
            else "%d 月" % mo[1], GY2, size=_sz(11))

    # ══ 每个模型一条 ══
    for i, (nm, kind, sz, t0, t1, nc, note) in enumerate(MODELS):
        yy = top + i * ROW
        col = KIND[kind]
        f.t(0, yy + 16, nm, INK, bold=True, size=_sz(12))
        f.t(0, yy + 31, "%s ·  %s" % (kind, sz), GY2, size=_sz(11), w=AX0 - 20)
        # ⭐ 单日的条（SDXL / Flux.1）也要看得见 ——&nbsp;给一个最小宽度。
        bx0, bx1 = xf(t0), max(xf(t1), xf(t0) + 9)
        f.box(bx0, yy + 6, bx1 - bx0, 17, col, col, 4)
        f.t(bx1 + 10, yy + 19, "%d 次提交" % nc, col, bold=True, size=_sz(11))
        f.t(AX1 + 110, yy + 19, note, GY2, size=_sz(11), w=W - AX1 - 118)

    ay = top + len(MODELS) * ROW + 6

    # ══ 关键时点（画在最上层）══
    for t, txt in MARKS:
        f.line(xf(t), top - 4, xf(t), ay, RD if "⭐" in txt else GY2,
               1.6 if "⭐" in txt else 1.1, dash=None if "⭐" in txt else "4 3",
               arrow=False)
    lx = 0
    for k, (t, txt) in enumerate(MARKS):
        # ⛔ 三个时点挤在十天内，标签必须错开行 ——&nbsp;不错开会撞成一团。
        f.t(xf(t) - 4, ay + 20 + k * 17, "▲ " + txt,
            "#a50e0e" if "⭐" in txt else GY2,
            bold="⭐" in txt, size=_sz(11))
    y = ay + 20 + len(MARKS) * 17 + 16

    y = f.band(y, "info",
               "读这张图的三条线索",
               ['⚠️ <tspan font-weight="700">第一代没有独立的时间段</tspan>：'
                'maxdiffusion 是起点不是阶段 ——&#160;12 月 3 日第一天就已经在写 Flax 版了。'
                '<tspan font-weight="700">图上只给它一个起点标记，不编一段区间出来。</tspan>',
                '① <tspan font-weight="700">前两周极密</tspan>：'
                '12 月 3 日到 14 日，四个视频模型全部开工，'
                '同时试完了三条移植路线 ——&#160;'
                '<tspan font-weight="700">代价最大的探索集中在最前面。</tspan>',
                '② <tspan font-weight="700">12 月 29 日之后没有再换过方法</tspan>。'
                'Flux.2 是第一个「一上来就是 torchax ＋ 三阶段」的模型，'
                '之后 SDXL / S3Diff / Real-ESRGAN 都照着走，'
                '<tspan font-weight="700">提交数从几十掉到个位数</tspan>。',
                '③ 那三条尾巴（Wan2.1 / CogVideoX / Wan2.2 拖到 3 月初）'
                '不是还在攻坚，是<tspan font-weight="700">回头把新方法反哺回老模型</tspan>——&#160;'
                '这也是为什么它们的提交数最高。'])

    y = f.band(y + 14, "ok",
               "⭐ 提交数掉下去，才是这套方法真正立住的证据",
               ['<tspan font-weight="700">HunyuanVideo 73 次、CogVideoX 49 次、Wan2.1 44 次；'
                '而 SDXL 9 次、Real-ESRGAN 3 次、Flux.1 2 次。</tspan>',
                '⭐ 差别不在模型难度 ——&#160;SDXL 和 Flux.1 都不比 CogVideoX 简单。'
                '差别在于<tspan font-weight="700">前面几个是在「发明方法」，后面几个是在「套用方法」</tspan>。',
                '⭐⭐ 所以这张图的落点不是「我们做了十个模型」，而是'
                '<tspan font-weight="700">「接第十一个模型的成本，已经不是前十个的量级了」</tspan>——&#160;'
                '这才是一条工程路线走通的标志。'])

    y = f.src(y + 18,
              '全部日期与提交数 ——&#160;本仓库 git 提交历史，'
              '以 <tspan font-family="monospace">git log --diff-filter=A</tspan> 取首次提交、'
              '<tspan font-family="monospace">git log --oneline | wc -l</tspan> 取提交数；'
              '统计截至 2026-04-03',
              '⚠️ 提交数只反映改动次数，<tspan font-weight="700">不等于工作量或难度</tspan> ——&#160;'
              '这里用它做的唯一推断是「发明方法 vs 套用方法」的量级差，'
              '不用它比较任意两个模型谁更难。')
    f.save("figx-12.svg", y + 6)


main()
