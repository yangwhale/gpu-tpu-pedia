# -*- coding: utf-8 -*-
r"""专题五 · 第零节「一条接力线」：五刀＋摆到机器上，每一刀付什么通信、留下什么问题逼出下一刀。

⭐ 2026-09-25 夜 · 故事性蒸馏 R1：外部讲得最好的几份（Hugging Face《Ultra-Scale Playbook》、
   yeasy 的中文书、Google《How to Scale Your Model》）都是同一个写法：每一节结尾交代下一种方法出场的理由，
   把一串技术讲成「解决一个问题 → 暴露下一个问题」的连续剧。原来第零节是一张 7 行的文字清单，
   这里把它画成一条接力线：盒子里写「切什么、付什么过路费」，盒子下面红字写「留下的问题」。
⛔ 每一格的内容都来自本讲对应小节的结尾（X.last「这一刀留下的问题」），这里不引入新事实。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

W = 1400
CUTS = [  # (名字, 小节, 切什么, 过路费, 留下的问题, 颜色)
    ("第一刀　切数据", "§2", "切 batch；再削掉重复存的", "AllReduce 拆成两半", "batch 一小，搬权重就藏不住", BL),
    ("第二刀　切权重", "§3", "切进矩阵、按层切", "每层 AllReduce；段间收发", "参数几乎全在很窄的专家里", OR),
    ("第三刀　切专家", "§4", "专家整个分给不同的卡", "每层两次 AllToAll", "上下文一长，一条样本装不下", GR),
    ("第四刀　切序列", "§5", "训练切激活，推理切 KV", "环上传 KV；收齐 Q、合结果", "prefill、decode 要的切法不一样", PU),
    ("第五刀　切工作", "§6", "prefill、decode 分开放", "一请求一趟 KV 传输", "刀这么多，谁坐快线？", CY),
    ("摆到机器上", "§7", "说得勤的坐快线", "摆错一次，同样的卡慢几倍", "", INK),
]


def fig_roadmap():
    f = Fig(W, "这一讲的接力线。先认识五种通信，然后五刀依次出场，最后摆到机器上。第一刀切数据，付的是把 AllReduce 拆成两半，"
               "留下的问题是 batch 一小搬权重就藏不住；第二刀切权重，每层一次 AllReduce，留下的问题是参数几乎全在很窄的专家里；"
               "第三刀切专家，每层两次 AllToAll，留下的问题是上下文一长一条样本装不下；第四刀切序列，留下的问题是 prefill 和 decode 要的切法不一样；"
               "第五刀切工作，一个请求一趟 KV 传输，留下的问题是刀这么多谁坐快线；最后摆到机器上，说得勤的坐快线")
    y0 = f.header("一条接力线　——　<tspan font-weight=\"700\">红字就是下一刀出场的理由</tspan>",
                  "盒子里：这一刀切什么、多付哪种通信（过路费）。盒子下面的红字：它留下的问题，逼出下一刀",
                  [(RD, "留下的问题 → 下一刀的理由")])
    PH = 300
    py = f.panel(0, y0, W, PH, "先认识五种通信，再一刀一刀切", INK)
    BW, GAPX, X0 = 196, 34, 24
    BY, BH = py + 30, 150
    for i, (name, sec, what, fee, left, col) in enumerate(CUTS):
        x = X0 + i * (BW + GAPX)
        f.box(x, BY, BW, BH, "none", col, 10, sw=2.4)
        f.box(x, BY, BW, 34, col, col, 10)
        f.box(x, BY + 24, BW, 10, col, col, 0)
        f.t(x + BW / 2, BY + 23, name, "#ffffff", True, 15, "middle")
        f.t(x + 12, BY + 72, what, INK, True, 13.5)
        f.t(x + 12, BY + 104, "过路费：", GY, size=12.5)
        f.t(x + 12, BY + 126, fee, col, True, 13)
        if i < len(CUTS) - 1:
            ax = x + BW + 3
            f.line(ax, BY + BH / 2, ax + GAPX - 6, BY + BH / 2, GY2, 2.2)
            f.t(x + 10, BY + BH + 30, "↳ 留下的问题", RD, True, 13)
            f.t(x + 10, BY + BH + 54, left, RD, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每一刀都不白来，也都不是最后一刀", [
        "它多付一种通信，换回一截显存或一份速度。",
        "它没管到的那一块，就是下一节开头要解决的问题。",
    ])
    f.save("fig5-roadmap.svg", yb + 14)


fig_roadmap()
