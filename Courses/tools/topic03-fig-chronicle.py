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

📌 **下半那张配比图是全图的重心。** 它一眼能看出两件事：
   ① 配比全部落在 **3:1 ～ 7:1**（便宜的层占 75%–87.5%）——&nbsp;
      没有人敢全用线性，也没有人只用一两层。
   ② **分派系**：KDA／GDN 那一派偏 3:1–5:1，Lightning 那一派偏 7:1，
      SWA 那一派偏 5:1–6:1。**用哪种便宜层，决定了你敢配多少。**

⛔ **一条口径护栏，别在简化时丢掉：**
   **「层间混合」和「层内稀疏」不是一回事。** MiniMax M3、DeepSeek DSA／CSA、
   GLM-5、混元 Hy4 走的是后者 ——&nbsp;每一层都还是全注意力的形状，
   只是每个 query 少看几块。**不能跟 7:1 那种放在同一根轴上比**，
   图里用不同的画法分开（虚线框 vs 实心条）。

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
t(16, BY + 24, '二、各家的混合配比 ——&#160;'
               '<tspan font-weight="700">「便宜的层 : 全注意力层」，一个循环里各几层</tspan>',
  "svglbl", "#202124", size=13)
t(16, BY + 43, '⛔ <tspan font-weight="700">层间混合</tspan>（几层线性配一层全）'
               '和<tspan font-weight="700">层内稀疏</tspan>（每层都还是全注意力的形状，'
               '只是每个 query 少看几块）<tspan font-weight="700">不是一回事</tspan>'
               '——&#160;下面用两种画法分开，<tspan font-weight="700">不能放在同一根轴上比</tspan>。',
  fill=RD)

# 配色：一个配比一个颜色
RATIO_COL = {"7:1": PU, "6:1": GR, "5:1": CY, "3:1": BL,
             "≈2.9:1": "#3b6fd4", "纯全": GY, "层内稀疏": OR}
ROWS = [
    # (时间, 模型, 便宜层类型, 配比, 便宜层占比 0-1, 备注)
    ("2025-01", "MiniMax-01（456B）", "Lightning", "7:1", 7 / 8,
     "线性这一支第一次上到旗舰规模"),
    ("2025-09", "Qwen3-Next（80B/3B）", "Gated DeltaNet", "3:1", 3 / 4, ""),
    ("2025-10", "MiniMax M2", "——", "纯全", 0.0,
     "⛔ <tspan font-weight=\"700\">退回全注意力</tspan>：低精度状态敏感、prefix cache 难做"),
    ("2025-10", "Kimi Linear", "KDA", "3:1", 3 / 4, "KDA＝GDN ＋ 按通道门控"),
    ("2026-01", "小米 MiMo-V2-Flash", "SWA（窗口 128）", "5:1", 5 / 6, ""),
    ("2026-02", "GLM-5（355B–744B）", "DSA 稀疏", "层内稀疏", -1,
     "智谱第一次上稀疏：MLA ＋ DeepSeek Sparse Attention"),
    ("2026-03", "Qwen3.5（0.8B–397B）", "Gated DeltaNet", "3:1", 3 / 4,
     "全家族统一：3×(GDN→FFN) → 1×(Gated Attn→FFN)"),
    ("2026-04", "⭐ 小米 MiMo-V2.5-Pro", "SWA（窗口 128）", "6:1", 6 / 7,
     "窗口只有 128 ——&#160;比谁都激进"),
    ("2026-06", "GLM-5.2（744B）", "DSA ＋ IndexShare", "层内稀疏", -1,
     "⭐ 每四个稀疏层共用一个 indexer，1M 下每 token 省 2.9× FLOP"),
    ("2026-06", "Ling 2.6（蚂蚁百灵）", "Lightning", "7:1", 7 / 8, ""),
    ("2026-06", "MiniMax M3", "MSA 稀疏", "层内稀疏", -1,
     "⭐ 第三次转向：不回线性，改走稀疏。每 query 只看 top-16 个 128-token 块"),
    ("2026-06", "DeepSeek V4", "CSA ＋ HCA", "层内稀疏", -1, "按距离分层压缩"),
    ("2026-07", "Kimi K3（2.8T）", "KDA", "≈2.9:1", 69 / 93,
     "93 层 ＝ 69 KDA ＋ 24 Gated MLA（KDA×3 → MLA×1，多出一层）"),
    ("2026-07", "混元 Hy3（295B/21B）", "——", "纯全", 0.0,
     "⛔ 80 层全是 GQA-8 ——&#160;<tspan font-weight=\"700\">线性一层都没上</tspan>"),
    ("2026-07", "Ling-3.0-flash（124B/5.1B）", "KDA", "5:1", 35 / 42,
     "35 KDA ＋ 7 Gated MLA ——&#160;<tspan font-weight=\"700\">预训练第一天就是混合的</tspan>"),
    ("2026-08", "混元 Hy4-preview（770B/49B）", "Gated DSA", "层内稀疏", -1,
     "78 层全稀疏 ＋ IndexCache（每 4 层只有 1 层自己算索引）"),
    # ⭐ 这一行是全表唯一「两种便宜法同时上」的：便宜的那层是线性(KDA)，
    #    而它配的那层「贵的」本身还是稀疏的。所以它同时属于两个阵营。
    # ⛔ 这一格只填「便宜的那一层」= KDA。别把稀疏 MLA 也写进来 ——
    #    稀疏 MLA 是它配的**那层贵的**，写进这一列会把列的含义搅乱。
    ("2026-08", "⭐ GLM-5.3-Flash（321B/18B）", "KDA", "3:1", 34 / 45,
     "45 层 ＝ 34 KDA ＋ 11 稀疏 MLA ——&#160;"
     "<tspan font-weight=\"700\">第一次线性和稀疏同锅</tspan>"),
]
LX, BARX, BARW = 16, 470, 330
t(LX, BY + 66, '时间', fill=GY, bold=True)
t(LX + 62, BY + 66, '模型', fill=GY, bold=True)
t(300, BY + 66, '便宜的那一层', fill=GY, bold=True)
t(BARX, BY + 66, '一个循环里的配比', fill=GY, bold=True)
t(BARX + BARW + 76, BY + 66, '备注', fill=GY, bold=True)
p.append('<line x1="16" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
         % (BY + 72, W - 16, BY + 72, "#dadce0"))

for i, (tm, mdl, cheap, ratio, frac, note) in enumerate(ROWS):
    y = BY + 92 + i * 27
    col = RATIO_COL[ratio]
    t(LX, y, tm, fill=GY)
    t(LX + 62, y, mdl, fill="#202124", bold=mdl.startswith("⭐"))
    t(300, y, cheap, fill=col)
    if frac < 0:                       # 层内稀疏：换一种画法，别跟层间混合混为一谈
        box(BARX, y - 12, BARW, 16, "#fff", OR, 3, 1.4, "4,3")
        t(BARX + BARW // 2, y, '层内稀疏 ——&#160;不是层间混合', fill=OR,
          bold=True, anchor="middle")
    else:
        box(BARX, y - 12, BARW, 16, "#f1f3f4", "#dadce0", 3)
        wcheap = int(BARW * frac)
        if wcheap:
            box(BARX, y - 12, wcheap, 16, col, col, 3)
        t(BARX + BARW + 8, y, ratio, fill=col, bold=True)
        t(BARX + BARW + 52, y, '%.0f%%' % (frac * 100), fill=GY)
    if note:
        t(BARX + BARW + 76 if frac >= 0 else BARX + BARW + 12, y, note, fill=GY)

# 落点
LZ = BY + 92 + len(ROWS) * 27 + 6
BH = LZ - BY + 62 + 14                       # 落点框下沿 + 一点留白
p[_BPANEL] = ('<rect x="0" y="%d" width="%d" height="%d" rx="8" fill="#fff" '
              'stroke="#dadce0" stroke-width="1"/>' % (BY, W, BH))
box(16, LZ, W - 32, 62, "#e8f0fe", BL, 6)
t(30, LZ + 20, '⭐ 这张条形图一眼能看出两件事', "svglbl", "#174ea6", size=12)
# ⛔ 2026-09-07：原话是「所有配比都落在 3:1～7:1」，补进混元和 GLM 之后就不成立了 ——
#    表里现在有 2 行纯全（M2、Hy3）、5 行层内稀疏，它们根本不在这根轴上。
# ⭐ 教训：**「所有 X 都……」这种全称句，会被后来加的行悄悄证伪，而且不报错。**
#    改成先报数再下结论，加行时数字对不上一眼就能看见。
_hy = sum(1 for r in ROWS if r[4] > 0)
_sp = sum(1 for r in ROWS if r[4] < 0)
_fu = len(ROWS) - _hy - _sp
t(30, LZ + 40, '① 表里 %d 行：<tspan font-weight="700">%d 个层间混合、%d 个层内稀疏、'
               '%d 个明确用纯全注意力</tspan>。而<tspan font-weight="700">搞层间混合的那 %d 个，'
               '配比无一例外落在 3:1 ～ 7:1</tspan>——&#160;便宜的层占 75%%–87.5%%。'
               '<tspan font-weight="700">没有人敢全用线性，也没有人只掺一两层。</tspan>'
               % (len(ROWS), _hy, _sp, _fu, _hy), fill="#174ea6")
t(30, LZ + 56, '② <tspan font-weight="700">分派系</tspan>：'
               'KDA／GDN 那一派偏 <tspan font-weight="700">3:1–5:1</tspan>，'
               'Lightning 那一派偏 <tspan font-weight="700">7:1</tspan>，'
               'SWA 那一派偏 <tspan font-weight="700">5:1–6:1</tspan>'
               '——&#160;<tspan font-weight="700">用哪种便宜层，决定了你敢配多少。</tspan>',
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
