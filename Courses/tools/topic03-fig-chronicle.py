# -*- coding: utf-8 -*-
"""专题三 · 图 A —— Attention 编年史：从 RNN 到今天，以及各家的混合配比。

⭐ **为什么要有这一张。** 2026-09-07 现场要求：

    「你先画一个全景图，把 Attention 的编年史给它画出来：
      1. 从一开始的 RNN 开始，按时间出现了什么注意力
      2. 它的典型模型是哪一个
      3. 这个模型一个循环的配比是什么样的
      拿条形图画出来，不同的配比不一样的颜色。
      所有的信息都去搜最新，千问、蚂蚁的灵、Kimi、混元，主流开源模型都别漏。」

⭐⭐ **这张图的主脊（也是整个专题的主脊）：**

    Transformer 当初做的交易是「用平方的计算量，买来完全的并行度」。
    这段历史，是在把那笔交易**往回赎** —— 但不能把并行度还回去。

   线性注意力想赎回 O(N)，代价是重新引入一个串行的状态；
   于是 chunk 化又是为了把并行度找回来。**一个完整的圆。**

📌 **下半那张格子图是全图的重心。** 2026-09-07 现场要求重画：

    「太乱了。便宜的那一层就不用写了，你就写什么注意力加什么注意力。
      后面那一个循环里边，把它做成一个一个格子的这种条状图。就比如说这个
      MLA 占几格，然后这个 KDA 占几格的这种一目了然的比例的条状图。
      然后每一个类型该用自己的颜色就用自己的颜色。」

⭐⭐ **改成格子之后，多出来两条旧画法根本画不出来的信息：**

   ① **「有没有格子」本身就是一条信息。** 整条一色 ＝ 每层同构，
      花样在层**内部**（层内稀疏就长这样）；切成格子 ＝ 层与层不一样，
      花样在层**之间**。
      ⛔ 所以别再给层内稀疏单画一种虚线框 ——&nbsp;那是**用两套画法说一件事**，
        而它本来可以由同一套画法自己说出来。

   ② **一个类型一个颜色之后，「贵的那层是什么」第一次看得见了。**
      混合的十家里九家是「冷色 ＋ 橙黄」（便宜的层配一层全注意力），
      **只有 GLM-5.3-Flash 是「蓝 ＋ 红」——&nbsp;它配的那层「贵的」，
      本身已经是稀疏的了。** 这条结论在旧画法（一个配比一个颜色）下
      根本无从看起 ——&nbsp;那时候颜色编码的是配比，不是类型。

⛔ **两条口径护栏：**
   1. **格子宽度固定，不按满宽等分。** 否则 4 格的循环和 8 格的循环画出来
      一样长，「这个循环有多长」这条信息就没了。
   2. **DeepSeek V4 是「稀疏配稀疏」（HCA 与 CSA 1:1 交替），不算层间混合。**
      把它混进那十家里算，3:1～7:1 那条结论就假了 ——&nbsp;落点里按三类分开报数。

⭐⭐ **2026-09-07 补了混元和 GLM 之后，多出来一条原先看不见的线：**
   混元 Hy4 的 **IndexCache** 和 GLM-5.2 的 **IndexShare** 是同一个想法 ——&nbsp;
   两家的 `indexer_types` 都是 `full, shared, shared, shared` 四层一循环。
   **稀疏的第二阶段优化不再是「让每个 query 少看几块」，
   而是「别每层都重新算一遍该看谁」——&nbsp;索引本身变成了新的开销。**
   这是 2026 年才冒出来的一层，值得单独占时间轴上一个点。

📌 出处（全部 2026-09-07 现搜，公开）：
   MiniMax-01/M1 7:1 与 M2 退回全注意力（MiniMax 官方博客《Why Did M2 End Up as
   a Full Attention Model?》）· M3 的 MSA（arXiv 2606.13392，top-16 × 128-token 块）
   · Qwen3-Next / Qwen3.5 3:1（Qwen 官方博客与 HF 模型卡的层布局串）
   · Kimi Linear 3:1、K3 93 层＝69 KDA＋24 Gated MLA（arXiv 2510.26692 与多家 day-0 支持文）
   · Ling-3.0-flash 5:1＝35 KDA＋7 MLA（inclusionAI HF 模型卡）· Ling 2.6 Lightning:MLA 7:1
   · 小米 MiMo-V2-Flash 5:1／V2.5-Pro 6:1，窗口 128（小米 MiMo 官方博客与 HF 模型卡）
   · **混元 Hy3**（preview 2026-04-23／正式版 2026-07-06 Apache 2.0）——&nbsp;
     配比这一格不是查来的，是**读我们自己仓库里那份 config**：
     `tpu/Hunyuan3-295B-Pretraining/assets/hunyuan3-tokenizer/config.json`，
     `HYV3ForCausalLM`，80 层，`num_attention_heads: 64 / num_key_value_heads: 8`
     →&nbsp;**纯 GQA-8，没有线性、没有稀疏、没有混合**
   · **混元 Hy4-preview**（2026-08-28，HF `tencent/Hy4-preview` 模型卡＋config）——&nbsp;
     770B/49B，78 层 `layer_types` **全部** `deepseek_sparse_attention`，
     Gated DSA ＋ IndexCache，indexer 32 头×128 维、top-k 2048，1M 上下文
   · **GLM-5**（2026-02-12，z.ai 官方博客）355B–744B，MLA ＋ DSA
   · **GLM-5.2**（2026-06-16，z.ai 博客＋HF config）744B，`GlmMoeDsaForCausalLM`，
     `index_topk_freq: 4`，官方原话「uses the same indexer across every four
     sparse attention layers, reducing per-token FLOPs by 2.9× at a 1M context」
   · **GLM-5.3-Flash**（2026-08-26，z.ai 博客＋HF config＋vLLM recipe）321B/18B，
     `layer_types` 是 `linear×3 → deepseek_sparse_attention×1` 循环，
     45 层 ＝ **34 KDA ＋ 11 稀疏 MLA**（NoPE），GLM 家族**第一次线性和稀疏同锅**
"""
import io

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
PU, CY, BR, PK = "#8430ce", "#00838f", "#7a5000", "#c5221f"
W = 1400
p = []


def wpx(s, size=11.5):
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


def t(x, y, s, cls="svgsm", fill=None, bold=False, size=None, anchor=None):
    st = ["font-size:%dpx" % size] if size else []
    p.append('<text class="%s" x="%d" y="%d"%s%s%s>%s</text>' % (
        cls, x, y, ' fill="%s"' % fill if fill else '',
        ' text-anchor="%s"' % anchor if anchor else '',
        ' style="%s"' % ';'.join(st) if st else '',
        '<tspan font-weight="700">%s</tspan>' % s if bold else s))


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=6, sw=1, dash=None):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" '
             'stroke="%s" stroke-width="%s"%s/>'
             % (x, y, w, h, r, fill, stroke, sw,
                ' stroke-dasharray="%s"' % dash if dash else ''))


# ⛔ 2026-09-07：原先这里把 viewBox 的高度写成字面量 1106，加五行模型就被裁掉底边。
# ⭐ 这跟当初把泳道行数写死是**同一类错**：一个由别的东西推出来的值，被抄成了常量。
#    改成占位符，最后按真实落点回填 —— 以后加行不用再手算高度。
_HDR = len(p)
p.append("")

t(0, 18, 'Attention 编年史 ——&#160;<tspan font-weight="700">'
         '从 RNN 的一个补丁，到今天各家的混合配比</tspan>',
  "svglbl", "#202124", size=15)
t(0, 39, '⭐ 一句话看懂整段历史：<tspan font-weight="700">Transformer 当初做的交易是'
         '「用平方的计算量，买来完全的并行度」——&#160;而这段历史，'
         '是在把那笔交易<tspan style="text-decoration:underline">往回赎</tspan>，但不能把并行度还回去。</tspan>')
t(0, 57, '线性注意力想赎回 O(N)，代价是重新引入一个<tspan font-weight="700">串行的状态</tspan>；'
         '于是 chunk 化又是为了把并行度找回来。'
         '<tspan font-weight="700">一个完整的圆。</tspan>'
         '　<tspan fill="%s">（信息截至 2026-09-07，全部现搜）</tspan>' % GY, fill=GY)

# ══════════ 上半：时间轴 ══════════════════════════════════════════════
TY = 72
_PANEL = len(p)
p.append("")   # 面板底框占位，高度算完再补
t(16, TY + 24, '一、编年史 ——&#160;四条支线，各修各的毛病', "svglbl", "#202124", size=13)

Y0, Y1 = 2014, 2026
AX0, AXW = 150, W - 150 - 30
def xf(y, frac=0.0):
    return AX0 + int((y - Y0 + frac) / (Y1 - Y0 + 1) * AXW)

# 年份刻度
AXY = TY + 46
p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
         % (AX0 - 8, AXY, W - 24, AXY, GY))
for y in range(Y0, Y1 + 1):
    x = xf(y, 0.5)
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
             % (x, AXY - 4, x, AXY + 4, GY))
    t(x, AXY - 9, "'%s" % str(y)[2:], fill=GY, anchor="middle", size=10)

# ⛔ 2026-09-07 重排。初版每个事件带一句说明，结果同年多事件时说明互相糊住
#    （2024 的 MLA 和「可并行 DeltaNet」直接叠在一起），2026 那几个还冲出右边界。
# ⭐ 判据：**时间轴的职责是「什么时候出现了什么」，不是解释机制。**
#    机制在下半那张表里全都有 —— 时间轴只留事件名，一个字说明都不留，
#    立刻就读得清了。⛔ 别再往这些点上加说明文字。
LANES = (
    ("前史 · 注意力是 RNN 的补丁", GY, "#f1f3f4", [
        (2014, "Bahdanau 注意力"),
        (2017, "⭐ Transformer / MHA"),
    ]),
    ("① 每个 token 存多少（KV 怎么小）", BL, "#e8f0fe", [
        (2019, "MQA"), (2023, "GQA"), (2024, "MLA"), (2026, "Gated MLA"),
    ]),
    ("② 每个 query 看多少（稀疏 · 压缩）", OR, "#fef7e0", [
        (2023, "SWA · sink"), (2025, "NSA · DSA"), (2026, "CSA＋HCA · MSA"),
        # ⭐ 2026 多出来的**第二阶段**：不是「少看几块」，是「别每层重算该看谁」。
        #    GLM-5.2 叫 IndexShare、混元 Hy4 叫 IndexCache，两家的 indexer_types
        #    都是 full,shared,shared,shared 四层一循环 —— 撞了同一个想法。
        (2026, "IndexShare · IndexCache"),
    ]),
    ("③ 换一套数学（线性注意力）", PU, "#f3e8fd", [
        (2020, "线性 Transformer"), (2021, "DeltaNet"),
        (2024, "可并行 DeltaNet · GDN"), (2025, "KDA · Lightning"),
        (2026, "Gated DeltaNet-2"),
    ]),
    ("④ 不改数学，只改怎么算", GR, "#e6f4ea", [
        (2022, "⭐ FlashAttention"),
    ]),
)
LY = AXY + 14
# ⛔ 第三轮。前两版分别栽在：①带说明文字互相糊；②只留名字但相邻年份仍撞；
#    ③改成上下两行交错之后，泳道③ 里 2024 和 2026 又落回同一行、又撞上了
#    （中间只隔一个 2025，而「可并行 DeltaNet · GDN」有 150px 宽）。
# ⭐⭐ 三次都是同一个错：**按位置的规律去排，而不是按实际占多宽去排。**
#    交错、奇偶、固定两行 —— 都是「看起来会错开」的规律，而真正决定撞不撞的
#    是标签的**渲染宽度**。现在改成**贪心装箱**：逐个事件量宽，放进第一条
#    还装得下的行；行数按需要长。以后再加事件也不会撞。
# ⛔ 别再改回固定行数。
def _rows(evs):
    """把事件按实际宽度贪心分行，返回 [(事件, 行号)] 和总行数。"""
    ends, out = [], []
    for (yr, lab) in evs:
        x = xf(yr, 0.5)
        w = wpx(lab)
        right = x + w + 10 > W - 30
        x0 = (x - 6 - w) if right else (x + 6)
        for r, e in enumerate(ends):
            if x0 > e + 10:
                ends[r] = x0 + w
                out.append((yr, lab, r, right))
                break
        else:
            ends.append(x0 + w)
            out.append((yr, lab, len(ends) - 1, right))
    return out, len(ends)


LANE_Y, ly = [], LY
for (name, col, fill, evs) in LANES:
    placed, nrow = _rows(evs)
    h = 18 + nrow * 17 + 6
    LANE_Y.append((ly, h, placed))
    ly += h + 4
for (name, col, fill, evs), (y, h, placed) in zip(LANES, LANE_Y):
    box(4, y, W - 28, h, fill, col, 5)
    t(14, y + 15, name, fill=col, bold=True)
    for (yr, lab, r, right) in placed:
        x = xf(yr, 0.5)
        ty = y + 28 + r * 17
        p.append('<circle cx="%d" cy="%d" r="4" fill="%s"/>' % (x, y + 4, col))
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.2"/>'
                 % (x, y + 4, x, ty - 9, col))
        t(x + (-6 if right else 6), ty, lab, fill=col, bold=True,
          anchor="end" if right else None)
TH = ly - TY + 8

# ══════════ 下半：配比条形图 ══════════════════════════════════════════
p[_PANEL] = ('<rect x="0" y="%d" width="%d" height="%d" rx="8" fill="#f8f9fa" '
             'stroke="#dadce0" stroke-width="1"/>' % (TY, W, TH))
BY = TY + TH + 14
# ⛔ BH 原先也是字面量 500 —— 跟 viewBox 那个 1106 一样，加行就会溢出白底。
#    同样改成占位符，等 ROWS 数完再回填。
_BPANEL = len(p)
p.append("")
t(16, BY + 24, '二、一个循环里都有哪些层 ——&#160;'
               '<tspan font-weight="700">一格 ＝ 一层，格子的颜色 ＝ 那一层用什么注意力</tspan>',
  "svglbl", "#202124", size=13)
# ⛔ 2026-09-07 重画（现场原话：「太乱了。便宜的那一层就不用写了，你就写什么注意力
#    加什么注意力。后面那一个循环里边，把它做成一个一个格子的这种条状图。
#    然后每一个类型该用自己的颜色就用自己的颜色。」）
# ⭐⭐ 改成格子之后多出来一个原先没有的表达力：**「有没有格子」本身就是一条信息。**
#    整条一色 ＝ 这个模型每一层都一样（花样在层**内部**）；
#    切成格子 ＝ 层与层之间不一样（花样在层**之间**）。
#    ⛔ 所以别再给「层内稀疏」单画一种虚线框了 —— 那是在用两套画法说一件事，
#      而它本来可以由同一套画法自己说出来。
t(16, BY + 43, '⭐ <tspan font-weight="700">整条一色</tspan>＝每一层都一样，花样在'
               '<tspan font-weight="700">层内部</tspan>（层内稀疏就长这样）；'
               '<tspan font-weight="700">切成格子</tspan>＝层与层不一样，花样在'
               '<tspan font-weight="700">层之间</tspan>。'
               '——&#160;<tspan font-weight="700">这两件事不能放在同一根轴上比。</tspan>',
  fill=RD)

# ── 一个类型一个颜色。⛔ 同族要用相近的色相，不同族必须拉开 ────────────
#    线性一族 冷色 · 窗口 青 · 全注意力 橙黄 · 稀疏 红
TYPE_COL = {
    "KDA":             BL,          # 线性
    "Gated DeltaNet":  "#12b5cb",
    "Lightning":       PU,
    "SWA":             CY,          # 窗口
    "MLA":             OR,          # 全注意力一族
    "Gated MLA":       OR,
    "Gated Attention": "#f9ab00",
    "全注意力":         "#f9ab00",
    "GQA-8":           "#f9ab00",
    "稀疏 MLA":         RD,          # 稀疏一族
    "DSA 稀疏 MLA":     RD,
    "Gated DSA":       RD,
    "MSA 稀疏":         RD,
    "CSA":             RD,
    "HCA":             "#a50e0e",
}
SPARSE = {"稀疏 MLA", "DSA 稀疏 MLA", "Gated DSA", "MSA 稀疏", "CSA", "HCA"}

# (时间, 模型, 一个循环的构成 [(类型, 几层)…], 备注)
# ⛔ 只有一项 ＝ 每层同构，画成整条一色。
ROWS = [
    ("2025-01", "MiniMax-01（456B）",
     [("Lightning", 7), ("全注意力", 1)], "线性这一支第一次上到旗舰规模"),
    ("2025-09", "Qwen3-Next（80B/3B）",
     [("Gated DeltaNet", 3), ("Gated Attention", 1)], ""),
    ("2025-10", "MiniMax M2",
     [("全注意力", 1)],
     "⛔ <tspan font-weight=\"700\">退回全注意力</tspan>：低精度状态敏感、prefix cache 难做"),
    ("2025-10", "Kimi Linear",
     [("KDA", 3), ("MLA", 1)], "KDA＝GDN ＋ 按通道门控"),
    ("2026-01", "小米 MiMo-V2-Flash",
     [("SWA", 5), ("全注意力", 1)], "SWA 窗口只有 128"),
    ("2026-02", "GLM-5（355B–744B）",
     [("DSA 稀疏 MLA", 1)], "智谱第一次上稀疏：MLA ＋ DeepSeek Sparse Attention"),
    ("2026-03", "Qwen3.5（0.8B–397B）",
     [("Gated DeltaNet", 3), ("Gated Attention", 1)],
     "全家族统一：3×(GDN→FFN) → 1×(Gated Attn→FFN)"),
    ("2026-04", "⭐ 小米 MiMo-V2.5-Pro",
     [("SWA", 6), ("全注意力", 1)], "窗口还是 128 ——&#160;比谁都激进"),
    ("2026-06", "GLM-5.2（744B）",
     [("DSA 稀疏 MLA", 1)],
     "⭐ ＋IndexShare：每四个稀疏层共用一个 indexer，1M 下省 2.9× FLOP"),
    ("2026-06", "Ling 2.6（蚂蚁百灵）",
     [("Lightning", 7), ("MLA", 1)], ""),
    ("2026-06", "MiniMax M3",
     [("MSA 稀疏", 1)],
     "⭐ 第三次转向：不回线性，改走稀疏。每 query 只看 top-16 个 128-token 块"),
    # ⭐ V4 不是「每层同构」——HCA 与 CSA 是**交替**的，所以它有格子。
    #   但两格都是稀疏族，跟「线性配全注意力」那种混合不是一回事，落点里单列。
    ("2026-06", "DeepSeek V4",
     [("HCA", 1), ("CSA", 1)], "HCA 与 CSA 交替：压缩率 128 的粗看 ＋ 压缩率 4 的细看"),
    ("2026-07", "Kimi K3（2.8T）",
     [("KDA", 3), ("Gated MLA", 1)],
     "93 层 ＝ 69 KDA ＋ 24 Gated MLA（比整齐的 3:1 多出一层 MLA）"),
    ("2026-07", "混元 Hy3（295B/21B）",
     [("GQA-8", 1)],
     "⛔ 80 层全是 GQA-8 ——&#160;<tspan font-weight=\"700\">线性一层都没上</tspan>"),
    ("2026-07", "Ling-3.0-flash（124B/5.1B）",
     [("KDA", 5), ("Gated MLA", 1)],
     "42 层 ＝ 35 KDA ＋ 7 MLA，<tspan font-weight=\"700\">预训练第一天就是混合的</tspan>"),
    ("2026-08", "混元 Hy4-preview（770B/49B）",
     [("Gated DSA", 1)], "78 层全稀疏 ＋ IndexCache（每 4 层只有 1 层自己算索引）"),
    # ⭐⭐ 全表唯一一条**蓝格＋红格**的：别人都是「冷色配橙黄」（线性配全注意力），
    #   只有它是「线性配稀疏」。改成按类型上色之后，这件事不用写字就看得见。
    ("2026-08", "⭐ GLM-5.3-Flash（321B/18B）",
     [("KDA", 3), ("稀疏 MLA", 1)],
     "45 层 ＝ 34 KDA ＋ 11 稀疏 MLA ——&#160;"
     "<tspan font-weight=\"700\">第一次线性和稀疏同锅</tspan>"),
]

# ── 图例 ────────────────────────────────────────────────────────────
LX = 16
LEG = [("线性", (("KDA", "KDA"), ("Gated DeltaNet", "GDN"),
                 ("Lightning", "Lightning"))),
       ("窗口", (("SWA", "SWA"),)),
       ("全注意力", (("MLA", "MLA／Gated MLA"),
                    ("Gated Attention", "Gated Attn／GQA"))),
       ("稀疏", (("稀疏 MLA", "DSA／MSA／稀疏 MLA／CSA"), ("HCA", "HCA")))]
lx = LX
for fam, items in LEG:
    t(lx, BY + 66, fam + "：", fill="#202124", bold=True)
    lx += wpx(fam + "：") + 6
    for key, lab in items:
        box(lx, BY + 56, 13, 13, TYPE_COL[key], TYPE_COL[key], 3)
        t(lx + 18, BY + 66, lab, fill=GY)
        lx += 18 + wpx(lab) + 14
    lx += 10

# ── 表头 ────────────────────────────────────────────────────────────
MDLX, MIXX, BARX = LX + 62, 288, 520
CELL, CGAP, MAXC = 32, 3, 8
BARW = MAXC * CELL + (MAXC - 1) * CGAP          # 8 格满宽
RATX, NOTEX = BARX + BARW + 14, BARX + BARW + 72
HY = BY + 92
t(LX, HY, '时间', fill=GY, bold=True)
t(MDLX, HY, '模型', fill=GY, bold=True)
t(MIXX, HY, '这一层 ＋ 那一层', fill=GY, bold=True)
t(BARX, HY, '一个循环（一格 ＝ 一层）', fill=GY, bold=True)
t(RATX, HY, '配比', fill=GY, bold=True)
t(NOTEX, HY, '备注', fill=GY, bold=True)
p.append('<line x1="16" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
         % (HY + 6, W - 16, HY + 6, "#dadce0"))

R0, RH = HY + 30, 30
for i, (tm, mdl, cyc, note) in enumerate(ROWS):
    y = R0 + i * RH
    t(LX, y, tm, fill=GY)
    t(MDLX, y, mdl, fill="#202124", bold=mdl.startswith("⭐"))
    # 「这一层 ＋ 那一层」：每个类型用自己的颜色写
    cx = MIXX
    for j, (ty, _) in enumerate(cyc):
        if j:
            t(cx, y, "＋", fill=GY)
            cx += wpx("＋") + 4
        t(cx, y, ty, fill=TYPE_COL[ty], bold=True)
        cx += wpx(ty) + 5
    if len(cyc) == 1:
        # 整条一色 ＝ 每层同构，没有循环可数
        ty = cyc[0][0]
        c = TYPE_COL[ty]
        box(BARX, y - 13, BARW, 18, c, c, 4)
        t(BARX + BARW // 2, y, '每一层都是这个', fill="#fff", bold=True,
          anchor="middle", size=10)
        t(RATX, y, '——', fill=GY)
    else:
        n = sum(k for _, k in cyc)
        # ⛔ 格子宽度**固定**，不按满宽等分 —— 否则 4 格的循环和 8 格的循环
        #   画出来一样长，「循环有多长」这条信息就没了。
        gx = BARX
        for ty, k in cyc:
            c = TYPE_COL[ty]
            for _ in range(k):
                box(gx, y - 13, CELL, 18, c, c, 3)
                gx += CELL + CGAP
        t(RATX, y, "%d:%d" % (cyc[0][1], cyc[-1][1]),
          fill=TYPE_COL[cyc[0][0]], bold=True)
    if note:
        t(NOTEX, y, note, fill=GY)

# ── 落点 ────────────────────────────────────────────────────────────
LZ = R0 + len(ROWS) * RH + 6
BH = LZ - BY + 88 + 14
p[_BPANEL] = ('<rect x="0" y="%d" width="%d" height="%d" rx="8" fill="#fff" '
              'stroke="#dadce0" stroke-width="1"/>' % (BY, W, BH))
box(16, LZ, W - 32, 88, "#e8f0fe", BL, 6)
t(30, LZ + 20, '⭐ 这张格子图一眼能看出三件事', "svglbl", "#174ea6", size=12)
# ⛔ 「所有 X 都……」这种全称句会被后来加的行悄悄证伪，而且不报错。
#    所以这里**先按数据分类报数，再下结论** —— 加行时数字对不上一眼就能看见。
# ⭐ 分三类而不是两类：DeepSeek V4 也有格子，但它是「稀疏配稀疏」，
#   跟「便宜的配全注意力」不是一回事，混在一起算会把 3:1~7:1 那条结论弄假。
_uni = [r for r in ROWS if len(r[2]) == 1]
_spx = [r for r in ROWS if len(r[2]) > 1 and all(t_ in SPARSE for t_, _ in r[2])]
_hyb = [r for r in ROWS if len(r[2]) > 1 and r not in _spx]
t(30, LZ + 40, '① 表里 %d 家：<tspan font-weight="700">%d 家是「便宜的层 ＋ 全注意力／稀疏层」</tspan>，'
               '%d 家<tspan font-weight="700">每层同构</tspan>（整条一色），'
               '%d 家是稀疏配稀疏。'
               '而那 %d 家混合的，<tspan font-weight="700">配比无一例外落在 3:1 ～ 7:1</tspan>'
               '——&#160;<tspan font-weight="700">没有人敢全用线性，也没有人只掺一两层。</tspan>'
               % (len(ROWS), len(_hyb), len(_uni), len(_spx), len(_hyb)),
  fill="#174ea6")
# ⛔ 原来的 ② 写「用哪种便宜层决定了你敢配多少」——「便宜层」那一栏已经取消了，
#    措辞得跟着改。⭐ 这就是「图注／落点是图的下游」那条：图改了口径，
#    结论句不会自己跟着动，而且不报错。
t(30, LZ + 58, '② <tspan font-weight="700">分派系</tspan>：'
               'KDA／GDN 那一派偏 <tspan font-weight="700">3:1–5:1</tspan>，'
               'Lightning 那一派偏 <tspan font-weight="700">7:1</tspan>，'
               'SWA 那一派偏 <tspan font-weight="700">5:1–6:1</tspan>'
               '——&#160;<tspan font-weight="700">线性／窗口那一层用哪种，'
               '决定了你敢配多少。</tspan>', fill="#174ea6")
# ⭐⭐ 这一条是**改成按类型上色之后才冒出来的** —— 原先一个配比一个颜色，
#    根本看不见「贵的那层是什么」。现在扫一眼颜色搭配就有了。
_warm = [r for r in _hyb if r[2][-1][0] not in SPARSE]
_cold = [r for r in _hyb if r[2][-1][0] in SPARSE]
t(30, LZ + 76, '③ ⭐ <tspan font-weight="700">扫一眼颜色搭配</tspan>：'
               '混合的那 %d 家里，<tspan font-weight="700">%d 家都是「冷色 ＋ 橙黄」</tspan>'
               '——&#160;便宜的层配一层全注意力。'
               '<tspan font-weight="700">只有 %s 是「蓝 ＋ 红」：它配的那层「贵的」，'
               '本身已经是稀疏的了。</tspan>'
               % (len(_hyb), len(_warm),
                  "、".join(r[1].replace("⭐ ", "") for r in _cold)),
  fill="#174ea6")

# ══════════ 落点带 ══════════════════════════════════════════════════
FY, FH = BY + BH + 14, 226
box(0, FY, W, FH, "#fef7e0", OR)
t(16, FY + 24, '⭐ 全图落点：三家公司，各自把旋钮拧了一遍 ——&#160;'
               '而且拧的过程全写在公开的 config 里', "svglbl", BR, size=13)
# ⭐ 三条轨迹并列，才看得出「这不是某一家的偶然选择」。
#    左边留 118px 给公司名，三行对齐。
for i, (who, arc) in enumerate((
        ('MiniMax',
         '线性（M1）→&#160;<tspan font-weight="700">退回全注意力</tspan>（M2）'
         '→&#160;<tspan font-weight="700">稀疏</tspan>（M3）'
         '——&#160;三次转向，每次都公开写了为什么'),
        ('腾讯混元',
         'Hy3 <tspan font-weight="700">80 层纯 GQA-8</tspan>（连线性都没上）'
         '→&#160;Hy4 <tspan font-weight="700">78 层全 Gated DSA</tspan>'
         '——&#160;<tspan font-weight="700">跳过线性那一支，直接进稀疏</tspan>'),
        ('智谱 GLM',
         '5 上 DSA →&#160;5.2 加 <tspan font-weight="700">IndexShare</tspan> '
         '→&#160;5.3-Flash <tspan font-weight="700">第一次线性＋稀疏同锅</tspan>'
         '——&#160;半年走完三步'))):
    y = FY + 48 + i * 19
    t(16, y, who, fill=BR, bold=True)
    t(118, y, arc, fill=BR)

t(16, FY + 122, '⭐⭐ 再看一眼第二列：这次补完混元和 GLM，多出来一条原先看不见的线',
  "svglbl", "#174ea6", size=12)
t(16, FY + 142, '混元 Hy4 的 <tspan font-weight="700">IndexCache</tspan> 和 '
                'GLM-5.2 的 <tspan font-weight="700">IndexShare</tspan> 是同一个想法：'
                '两家的 <tspan font-weight="700">indexer_types</tspan> 都是 '
                '<tspan font-weight="700">full, shared, shared, shared</tspan> 四层一循环 '
                '——&#160;每 4 层只有 1 层自己算索引。', fill="#174ea6")
t(16, FY + 160, '⭐ 所以稀疏的<tspan font-weight="700">第二阶段</tspan>优化，'
                '已经不是「让每个 query 少看几块」，而是'
                '<tspan font-weight="700">「别每层都重新算一遍该看谁」</tspan>'
                '——&#160;<tspan font-weight="700">索引本身变成了新的开销。</tspan>'
                '这是 2026 年才冒出来的一层。', fill="#174ea6")

box(16, FY + 172, W - 32, 1, GY, GY, 0)
t(16, FY + 194, '⚠️ <tspan font-weight="700">一条必须带上的限定：稀疏那一档的账，'
                '纸面上拿不到</tspan>', "svglbl", RD, size=12)
t(16, FY + 212, 'MiniMax M3 的 GGUF 发布说明写着「MSA 不支持 →&#160;推理退回稠密」'
                '——&#160;<tspan font-weight="700">纸面省下的 FLOP，要 kernel 跟上了才算数</tspan>。'
                '这一栏的每一个「层内稀疏」，都该配一句「在哪个引擎上」。', fill=GY)

p.append('</svg>')
# ── 回填 svg 开标签：高度按真实落点算，不写死 ──────────────────────
p[_HDR] = ('<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="'
           'Attention 编年史：2014 年注意力作为 RNN 的补丁出现，2017 年 Transformer 把 RNN 拿掉，'
           '此后分成四支演化；下半是各家开源模型的混合配比条形图，'
           '含混元 Hy3／Hy4 与 GLM-5 系列">' % (W, FY + FH + 12))
io.open('fig3-chronicle.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-chronicle ok')
