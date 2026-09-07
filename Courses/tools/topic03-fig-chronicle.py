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

⭐⭐ **2026-09-07 第三轮，现场逐条追问逼出来的四条更正。**
   这一轮的价值不在于「补全」，而在于**前一版有三处是错的或漏的，而且都不报错**：

   1. **DeepSeek-V4 原先写成「只有 CSA 和 HCA 两种层」——&nbsp;错。**
      追问原话：「V4 好好调查一下，它是不是只有 CSA 跟 HCA？那 MLA 和 DSA 跑哪去了？」
      扒 `DeepSeek-V4-Flash/config.json` 里的 `compress_ratios`，它**就是层表**：
      `[0,0, 4,128,4,128,…,4, 0]`——&nbsp;0＝滑窗全注意力、4＝CSA、128＝HCA。
      **43 层 ＝ 2 层 SWA 引导 ＋ 21 层 CSA ＋ 20 层 HCA，是三种层不是两种。**
      · **MLA 去哪了：被换掉了。** HF 官方文档原话「replaces DeepSeek-V3's
        Multi-head Latent Attention (MLA) with a hybrid local + long-range design」；
        `num_key_value_heads: 1` ——&nbsp;底层是 **shared K=V 的 MQA**，
        同一个张量既当 key 又当 value。⭐ 绕一大圈回到了 2019 年的 MQA。
      · **DSA 去哪了：没消失，降级成了零件。** Lightning Indexer 还在，
        它是 **CSA 内部**那一步打分（`index_n_heads: 64, index_topk: 512`）。

   2. **Ling 2.6 的现场记忆是 KDA ——&nbsp;查下来不是。** 官方 base 模型卡写的是
      **Lightning Attention ＋ MLA，7:1**，而且它是**从 Ling-2.0 的 GQA 迁移改造**
      来的（Lightning conversion → linear warmup → MLA conversion → MLA warmup），
      不是从头训。KDA 是 **Ling-3.0** 那一代才换上的。⭐ 两代确实换了一支。

   3. **GLM 漏了 5.3。** 官方仓库列得很清楚：5 / 5.1 / 5.2 / 5.3 都是 744B-A40B，
      **5.3 旗舰跟 5.2 是同一个 base，纯后训练、架构一个字没动**；
      只有 **5.3-Flash（320B-A18B）是全新训练的 base**，才是那个线性＋稀疏混合的。
      所以「Pro 去哪了」的答案是：**Pro 就是 GLM-5.3 本身，但它架构上等于 5.2。**

   4. **缺基线。** 前四行补上 GPT-3（MHA）／PaLM（MQA）／Llama 2（GQA）／
      DeepSeek-V2（MLA），**每个机制配它首次出现时的那个模型**；
      稀疏那一支的起点 DeepSeek-V3.2-Exp 也补进去了。

⭐⭐⭐ **2026-09-07 第四轮，一条追问换来整张表的重心。** 原话：

    「MiniMax M2 全注意力这个你得深入研究一下。所谓退回全注意力不大可能，
      因为全注意力就不可能有长上下文，都这个年代了，它肯定是有什么 trade off
      折中方案之类的。而且压缩注意力说的也是全注意力的意思，
      它是不是用了压缩注意力的 MLA 这种？」

   扒 `MiniMax-M2/config.json`，**这个怀疑对了一半，也纠正了一半**：

   · **对的一半：它绝不是「什么都没做的全注意力」。**
     `num_attention_heads: 48` / `num_key_value_heads: 8` ——&nbsp;**GQA-8**，
     KV 直接是 MHA 的 1/6；`rotary_dim: 64` 配 `head_dim: 128` ——&nbsp;
     **partial RoPE，只转一半维度**。原先那一行标成笼统的「FULL」是偷懒。
   · **纠正的一半：它不是 MLA，也不是压缩注意力。**
     config 里没有 `kv_lora_rank`、没有 indexer、`sliding_window: null`、
     `attn_type_list` 62 层全是 1。

   ⭐⭐ **这一问逼出了整张图最重要的一条口径：
   「全注意力」是相对旋钮③（线性）说的，不是相对旋钮①（KV 存多少）说的。**
   M2 退回的是③，①上它一直压着 ——&nbsp;**三个旋钮正交，M2 就是活证据。**

   ⭐⭐⭐ **而「全注意力撑不起长上下文」这个直觉，被量化成了一整列。**
   于是加了「上下文」这一列，扫下来的结论硬得出乎意料：
   **做到 1M 以上的八家，无一例外都动了旋钮②或③；纯全注意力那一档最高只到 256K。**
   最硬的对照来自 MiniMax 自己：**01 用 7:1 线性做到 10M
   （config 写着 `10240000`），M2 退回纯全注意力只剩 192K（`196608`）——&nbsp;
   同一家、同一批人，改一处架构差 50 倍。**

   ⛔ 上下文那一列**只填核到一手出处的**（多数是直接读 config.json），
     核不到就留「—」。⭐ 空格不是「没有」，是「本轮没核到」——&nbsp;别把它读成 0。

⛔ **一条口径护栏：格子宽度固定，不按满宽等分。**
   否则 4 格的循环和 8 格的循环画出来一样长，「这个循环有多长」这条信息就没了。

⛔ **落点里的分类判据要求「恰好两种类型」。** 只写「含一个便宜的 ＋ 含一个贵的」
   会把 DeepSeek-V4 也算进层间混合（它有 SWA），而 V4 的 SWA 是引导层。
   ⭐ **分类判据比结论句更容易悄悄出错** ——&nbsp;它不报错，只是把一行放进错误的桶，
     然后结论跟着变，而你看不出来。

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
W = 1680
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
t(16, BY + 24, '二、编年史续 · 一个循环里都有哪些层 ——&#160;'
               '<tspan font-weight="700">一格 ＝ 一层，格子里写的是那一层用什么注意力</tspan>',
  "svglbl", "#202124", size=13)
# ⛔ 2026-09-07 第三版（现场追问逐条）：
#    ①「再往前还缺一些传统注意力，得先作为 baseline 出现」→ 补 GPT-3／PaLM／
#      Llama 2／DeepSeek-V2／V3.2，**每个新机制都配它首次出现时的那个模型**。
#    ②「GLM-5 到底是五点几？GLM-5.3 你漏掉了」→ 5／5.1／5.2／5.3 全列，
#      并写明 5.3 旗舰跟 5.2 是同一个 base（纯 post-training，架构没动）。
#    ③「Ling 2.6 我记得用的是 KDA」→ 查了官方 base 模型卡：**是 Lightning ＋ MLA 7:1**，
#      KDA 是 Ling-3.0 那一代。⭐ 两代确实不同源，备注里写清楚。
#    ④「每一格里边用简写字母写上」→ 格子里印 KDA／MLA／CSA／HCA 这些。
#    ⑤「DeepSeek V4 好好调查，是不是只有 CSA 跟 HCA？MLA 和 DSA 跑哪去了」
#      → 见下面 V4 那一行的注释，答案比原先写的复杂得多，而且原先是错的。
t(16, BY + 43, '⭐ <tspan font-weight="700">整条一色</tspan>＝每一层都一样（花样在'
               '<tspan font-weight="700">层内部</tspan>）；'
               '<tspan font-weight="700">切成格子</tspan>＝层与层不一样（花样在'
               '<tspan font-weight="700">层之间</tspan>）。'
               '前四行是<tspan font-weight="700">基线</tspan>：'
               '每个新机制都配上它<tspan font-weight="700">首次出现时的那个模型</tspan>。',
  fill=RD)

# ── 一个类型一个颜色。同族相近色相，异族拉开 ──────────────────────────
AMB, ORG, DKR = "#f9ab00", "#e8710a", "#a50e0e"
TYPE_COL = {
    # 全注意力一族 —— 黄／橙
    "MHA": AMB, "MQA": AMB, "GQA": AMB, "FULL": AMB, "gAT": AMB,
    "MLA": ORG, "gMLA": ORG,
    # 线性一族 —— 冷色
    "KDA": BL, "GDN": "#12b5cb", "LTN": PU,
    # 窗口 —— 青
    "SWA": CY,
    # 稀疏一族 —— 红
    "DSA": RD, "gDSA": RD, "MSA": RD, "CSA": RD, "HCA": DKR,
}
SPARSE = {"DSA", "gDSA", "MSA", "CSA", "HCA"}
LINEAR = {"KDA", "GDN", "LTN"}
# 简写 → 全名（写在「这一层 ＋ 那一层」那一列）
FULLNAME = {
    "MHA": "MHA", "MQA": "MQA", "GQA": "GQA", "FULL": "全注意力",
    "gAT": "Gated Attention", "MLA": "MLA", "gMLA": "Gated MLA",
    "KDA": "KDA", "GDN": "Gated DeltaNet", "LTN": "Lightning",
    "SWA": "SWA", "DSA": "DSA", "gDSA": "Gated DSA", "MSA": "MSA",
    "CSA": "CSA", "HCA": "HCA",
}

# (时间, 模型, 一个循环的构成 [(简写, 几层)…], 备注)
# ⛔ 只有一项 ＝ 每层同构，画成整条一色。
ROWS = [
    # (时间, 模型, 循环构成, 上下文, 备注)
    # ⛔ 上下文这一列**只填核到一手出处的**（多数是本轮直接读的 config.json）。
    #   核不到就留「—」——&nbsp;宁可缺一格，不猜一格。
    # ── 基线：每个机制配它首次出现的模型 ────────────────────────────
    ("2020-05", "GPT-3（175B）", [("MHA", 1)], "2K",
     "96 层全 MHA，96 头×128 维。<tspan font-weight=\"700\">基线：KV 按头数线性长，没有任何省法</tspan>"),
    ("2022-04", "PaLM（540B）", [("MQA", 1)], "2K",
     "118 层，48 头<tspan font-weight=\"700\">共用 1 组 KV</tspan> ——&#160;第一次大规模砍 KV（Shazeer 2019）"),
    ("2023-07", "Llama 2（70B）", [("GQA", 1)], "4K",
     "8 组 KV。MQA 砍太狠会掉质量，GQA 是折中（arXiv 2305.13245）"),
    ("2024-05", "DeepSeek-V2（236B/21B）", [("MLA", 1)], "128K",
     "不砍头，改低秩压缩。<tspan font-weight=\"700\">KV cache 降 93.3%</tspan>；V3 原样沿用"),
    # ── 三个旋钮各自的第一次 ────────────────────────────────────────
    # ⭐⭐ 这一行和下面 M2 那一行**必须并排读**：同一家公司、同一批人，
    #    01 用 7:1 线性混合做到 10M，M2 退回纯全注意力只剩 192K。差 50 倍。
    ("2025-01", "MiniMax-01（456B）", [("LTN", 7), ("GQA", 1)], "10M",
     "⭐ 线性第一次上旗舰规模。<tspan font-weight=\"700\">config 写着 10,240,000</tspan>；"
     "那层「贵的」是 GQA-8"),
    ("2025-09", "DeepSeek-V3.2-Exp", [("DSA", 1)], "160K",
     "⭐ <tspan font-weight=\"700\">稀疏这一支的起点</tspan>：MLA ＋ Lightning Indexer，每 query 只留 top-k"),
    ("2025-09", "Qwen3-Next（80B/3B）", [("GDN", 3), ("gAT", 1)], "—", ""),
    # ⭐⭐ 2026-09-07 现场追问：「所谓退回全注意力不大可能，全注意力就不可能有长上下文，
    #    它肯定有什么 trade off。它是不是用了压缩注意力的 MLA 这种？」
    #    去扒 MiniMax-M2/config.json，**对了一半，也纠正了一半**：
    #    · 对的一半：它绝不是「什么都没做」。`num_attention_heads: 48` /
    #      `num_key_value_heads: 8` →&nbsp;**GQA-8**；`rotary_dim: 64` 配
    #      `head_dim: 128` →&nbsp;**partial RoPE，只转一半维度**。
    #    · 纠正的一半：**它不是 MLA、也不是压缩注意力。** 没有 `kv_lora_rank`、
    #      没有 indexer、`sliding_window: null`、`attn_type_list` 62 层全是 1。
    #    ⭐⭐⭐ 这一问逼出了本图最重要的一条口径：
    #      **「全注意力」是相对旋钮③（线性）说的，不是相对旋钮①（KV 存多少）说的。**
    #      M2 退回的是③，①上它一直压着。三个旋钮正交，M2 就是活证据。
    #    ⭐ 而「全注意力撑不起长上下文」这个直觉，被上下文那一列量化了：192K。
    ("2025-10", "MiniMax M2（230B/10B）", [("GQA", 1)], "192K",
     "⛔ <tspan font-weight=\"700\">「退回全注意力」不等于什么都没做</tspan>：它是 GQA-8 ＋ partial RoPE。"
     "但确实<tspan font-weight=\"700\">没有 MLA、没有稀疏</tspan>"),
    ("2025-10", "Kimi Linear（48B/3B）", [("KDA", 3), ("MLA", 1)], "1M",
     "27 层 ＝ 20 KDA ＋ 7 MLA（<tspan font-weight=\"700\">末层强制 full，所以多一层</tspan>）。已用 NoPE"),
    ("2026-01", "小米 MiMo-V2-Flash", [("SWA", 5), ("FULL", 1)], "—", "SWA 窗口只有 128"),
    ("2026-02", "GLM-5（744B/40B）", [("DSA", 1)], "198K",
     "MLA ＋ DSA，78 层。<tspan font-weight=\"700\">GLM-5.1 是同一套架构</tspan>，只有后训练不同"),
    ("2026-03", "Qwen3.5（397B/17B）", [("GDN", 3), ("gAT", 1)], "256K",
     "60 层 ＝ 45 线性 ＋ 15 全注意力，<tspan font-weight=\"700\">config 里 full_attention_interval: 4</tspan>"),
    ("2026-04", "小米 MiMo-V2.5-Pro", [("SWA", 6), ("FULL", 1)], "—", "窗口还是 128 ——&#160;比谁都激进"),
    ("2026-05", "DeepSeek-V4-Flash（43 层）",
     [("SWA", 2), ("CSA", 1), ("HCA", 1), ("CSA", 1), ("HCA", 1)], "1M",
     "⭐ 前 2 层 SWA 引导，之后 <tspan font-weight=\"700\">CSA／HCA 严格交替</tspan>（21＋20）。"
     "<tspan font-weight=\"700\">MLA 被换掉了</tspan>，底层是 shared-KV 的 MQA"),
    ("2026-06", "GLM-5.2（744B/40B）", [("DSA", 1)], "1M",
     "⭐ ＋IndexShare：每四个稀疏层共用一个 indexer。<tspan font-weight=\"700\">198K → 1M 就是这一步</tspan>"),
    ("2026-06", "Ling 2.6-1T（1T/63B）", [("LTN", 7), ("MLA", 1)], "256K",
     "⛔ 不是 KDA。而且是<tspan font-weight=\"700\">从 Ling-2.0 的 GQA 迁移改造</tspan>来的，不是从头训"),
    ("2026-06", "MiniMax M3", [("MSA", 1)], "—",
     "⭐ 第三次转向：不回线性，改走稀疏。每 query 只看 top-16 个 128-token 块"),
    ("2026-07", "Kimi K3（2.8T）", [("KDA", 3), ("gMLA", 1)], "1M",
     "93 层 ＝ 69 KDA ＋ 24 Gated MLA（<tspan font-weight=\"700\">末层 92、93 连着两层 full</tspan>）"),
    ("2026-07", "混元 Hy3（295B/21B）", [("GQA", 1)], "256K",
     "⛔ 80 层全是 GQA-8 ——&#160;<tspan font-weight=\"700\">线性一层都没上</tspan>"),
    ("2026-07", "Ling-3.0-flash（124B/5.1B）", [("KDA", 5), ("gMLA", 1)], "256K",
     "42 层 ＝ 35 KDA ＋ 7 MLA。<tspan font-weight=\"700\">跟 2.6 换了一支</tspan>；"
     "同代 tiny 是 3:1，旗舰尚未发布"),
    ("2026-08", "GLM-5.3（744B/40B）", [("DSA", 1)], "1M",
     "⚠️ <tspan font-weight=\"700\">旗舰版跟 5.2 是同一个 base，纯后训练，架构一个字没动</tspan>"),
    ("2026-08", "混元 Hy4-preview（770B/49B）", [("gDSA", 1)], "1M",
     "78 层全稀疏 ＋ IndexCache（每 4 层只有 1 层自己算索引）"),
    ("2026-08", "⭐ GLM-5.3-Flash（320B/18B）", [("KDA", 3), ("DSA", 1)], "1M",
     "45 层 ＝ 34 KDA ＋ 11 稀疏 MLA（NoPE）——&#160;"
     "<tspan font-weight=\"700\">GLM 第一次线性和稀疏同锅</tspan>，全新的 base"),
]

# ── 图例（两行，按族分组）────────────────────────────────────────────
LX = 16
for li, (fam, keys) in enumerate((
        ("全注意力一族", ("MHA", "MQA", "GQA", "FULL", "gAT", "MLA", "gMLA")),
        ("线性", ("KDA", "GDN", "LTN")),
        ("窗口", ("SWA",)),
        ("稀疏一族", ("DSA", "gDSA", "MSA", "CSA", "HCA")))):
    pass
lx, ly2 = LX, BY + 66
for fam, keys in (("全注意力一族", ("MHA", "MQA", "GQA", "FULL", "gAT", "MLA", "gMLA")),
                  ("线性", ("KDA", "GDN", "LTN")),
                  ("窗口", ("SWA",)),
                  ("稀疏一族", ("DSA", "gDSA", "MSA", "CSA", "HCA"))):
    t(lx, ly2, fam + "：", fill="#202124", bold=True)
    lx += wpx(fam + "：") + 4
    for k in keys:
        c = TYPE_COL[k]
        box(lx, ly2 - 11, wpx(k, 9) + 12, 15, c, c, 3)
        t(lx + 6, ly2, k, fill="#fff", bold=True, size=9)
        lx += wpx(k, 9) + 12 + 6
    lx += 14

# ── 表头 ────────────────────────────────────────────────────────────
MDLX, MIXX, BARX = LX + 62, 290, 512
CELL, CGAP, MAXC = 42, 3, 8
BARW = MAXC * CELL + (MAXC - 1) * CGAP
RATX = BARX + BARW + 12
CTXX = RATX + 54
NOTEX = CTXX + 66
HY = BY + 96
t(LX, HY, '时间', fill=GY, bold=True)
t(MDLX, HY, '模型', fill=GY, bold=True)
t(MIXX, HY, '这一层 ＋ 那一层', fill=GY, bold=True)
t(BARX, HY, '一个循环（一格 ＝ 一层）', fill=GY, bold=True)
t(RATX, HY, '配比', fill=GY, bold=True)
t(CTXX, HY, '上下文', fill=GY, bold=True)
t(NOTEX, HY, '备注', fill=GY, bold=True)
p.append('<line x1="16" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
         % (HY + 6, W - 16, HY + 6, "#dadce0"))

R0, RH = HY + 30, 30
for i, (tm, mdl, cyc, ctx, note) in enumerate(ROWS):
    y = R0 + i * RH
    if i and ROWS[i - 1][0][:4] != tm[:4]:          # 换年份画一条极淡的分隔
        p.append('<line x1="16" y1="%d" x2="%d" y2="%d" stroke="#f1f3f4" '
                 'stroke-width="1"/>' % (y - 21, W - 16, y - 21))
    t(LX, y, tm, fill=GY)
    t(MDLX, y, mdl, fill="#202124", bold=mdl.startswith("⭐"))
    # 「这一层 ＋ 那一层」——&#160;去重后按出现顺序列全名，各用自己的颜色
    seen, cx = [], MIXX
    for ty, _ in cyc:
        if ty not in seen:
            seen.append(ty)
    for j, ty in enumerate(seen):
        if j:
            t(cx, y, "＋", fill=GY)
            cx += wpx("＋") + 3
        nm = FULLNAME[ty]
        t(cx, y, nm, fill=TYPE_COL[ty], bold=True)
        cx += wpx(nm) + 4
    if len(cyc) == 1:
        ty = cyc[0][0]
        c = TYPE_COL[ty]
        box(BARX, y - 13, BARW, 18, c, c, 4)
        t(BARX + BARW // 2, y, '%s ——&#160;每一层都是这个' % ty, fill="#fff",
          bold=True, anchor="middle", size=10)
        t(RATX, y, '——', fill=GY)
    else:
        # ⛔ 格子宽度固定，不按满宽等分 ——&#160;否则 4 格的循环和 8 格的循环
        #   画出来一样长，「这个循环有多长」这条信息就没了。
        gx = BARX
        for ty, k in cyc:
            c = TYPE_COL[ty]
            for _ in range(k):
                box(gx, y - 13, CELL, 18, c, c, 3)
                t(gx + CELL // 2, y - 1, ty, fill="#fff", bold=True,
                  anchor="middle", size=9)
                gx += CELL + CGAP
        # 配比只在「便宜的 : 贵的」两段式时才有意义
        if len(seen) == 2:
            t(RATX, y, "%d:%d" % (cyc[0][1], cyc[-1][1]),
              fill=TYPE_COL[cyc[0][0]], bold=True)
        else:
            t(RATX, y, '见备注', fill=GY)
    # 上下文：≥1M 的标红加粗 —— 那一档是这张表最想让人看见的分界
    if ctx == "—":
        t(CTXX, y, '—', fill="#bdc1c6")
    else:
        big = ctx.endswith("M")
        t(CTXX, y, ctx, fill=RD if big else GY, bold=big)
    if note:
        t(NOTEX, y, note, fill=GY)

# ── 落点 ────────────────────────────────────────────────────────────
LZ = R0 + len(ROWS) * RH + 6
BH = LZ - BY + 108 + 14
p[_BPANEL] = ('<rect x="0" y="%d" width="%d" height="%d" rx="8" fill="#fff" '
              'stroke="#dadce0" stroke-width="1"/>' % (BY, W, BH))
box(16, LZ, W - 32, 108, "#e8f0fe", BL, 6)
t(30, LZ + 20, '⭐ 这张格子图一眼能看出四件事', "svglbl", "#174ea6", size=12)
# ⛔ 「所有 X 都……」这种全称句会被后来加的行悄悄证伪，而且不报错。
#    所以按数据分类**先报数再下结论**。⭐ 分类判据写成代码，加行时自动跟着变。
_uni = [r for r in ROWS if len(r[2]) == 1]
_mix = [r for r in ROWS if len(r[2]) > 1]
# ⛔ 判据必须要求**恰好两种类型**。只写「含一个便宜的 ＋ 含一个贵的」会把
#    DeepSeek-V4 也算进来（它有 SWA，也有 CSA/HCA），而 V4 的 SWA 是**引导层**、
#    不是循环里那个便宜的层 —— 混进来 3:1～7:1 那条结论就假了。
# ⭐ 这是这张图第二次栽在同一个地方：**分类判据比结论句更容易悄悄出错**，
#   因为它不报错，只是把一行放进了错误的桶，然后结论跟着变、而你看不出来。
_hyb = [r for r in _mix
        if len({t_ for t_, _ in r[2]}) == 2
        and any(t_ in LINEAR or t_ == "SWA" for t_, _ in r[2])
        and any(t_ not in LINEAR and t_ != "SWA" for t_, _ in r[2])]
_oth = [r for r in _mix if r not in _hyb]
t(30, LZ + 40, '① 表里 %d 家：<tspan font-weight="700">%d 家是「便宜的层 ＋ 一层贵的」</tspan>，'
               '%d 家<tspan font-weight="700">每层同构</tspan>（整条一色），%d 家是别的混法。'
               '而那 %d 家混合的，<tspan font-weight="700">配比无一例外落在 3:1 ～ 7:1</tspan>'
               '——&#160;<tspan font-weight="700">没有人敢全用线性，也没有人只掺一两层。</tspan>'
               % (len(ROWS), len(_hyb), len(_uni), len(_oth), len(_hyb)), fill="#174ea6")
t(30, LZ + 58, '② <tspan font-weight="700">前四行是基线，也是一条完整的小史</tspan>：'
               'MHA →&#160;MQA（砍到 1 组）→&#160;GQA（折中）→&#160;MLA（改压缩）'
               '——&#160;<tspan font-weight="700">四步全都只在动「每个 token 存多少」这一个旋钮</tspan>，'
               '花了四年。', fill="#174ea6")
# ⭐⭐ 这一条是**改成按类型上色之后才冒出来的** —— 旧画法一个配比一个颜色，
#    根本看不见「贵的那层是什么」。
_warm = [r for r in _hyb if r[2][-1][0] not in SPARSE]
_cold = [r for r in _hyb if r[2][-1][0] in SPARSE]
t(30, LZ + 76, '③ ⭐ <tspan font-weight="700">扫一眼颜色搭配</tspan>：'
               '混合的那 %d 家里 <tspan font-weight="700">%d 家都是「冷色 ＋ 黄橙」</tspan>'
               '（便宜的层配一层全注意力）。'
               '<tspan font-weight="700">只有 %s 是「蓝 ＋ 红」：它配的那层「贵的」，'
               '本身已经是稀疏的了。</tspan>'
               % (len(_hyb), len(_warm),
                  "、".join(r[1].replace("⭐ ", "").split("（")[0] for r in _cold)),
  fill="#174ea6")
# ⭐⭐⭐ 2026-09-07 现场追问逼出来的一条，也是整张表最硬的一条：
#    「所谓退回全注意力不大可能，因为全注意力就不可能有长上下文。」
#    ——&nbsp;把上下文那一列竖着扫一遍，这个直觉被数据完全证实了，
#    而最硬的对照来自 MiniMax 自己：同一家、同一批人，改一处架构差 50 倍。
# ⛔ 空着的那几格是**没核到一手出处**，不是没有 —— 宁可缺不猜。
_1m = [r for r in ROWS if r[3].endswith("M") and r[3] != "—"]
_pure = [r for r in ROWS if len(r[2]) == 1 and r[2][0][0] not in SPARSE]
t(30, LZ + 94, '④ ⭐⭐ <tspan font-weight="700">把「上下文」那一列竖着扫一遍</tspan>：'
               '<tspan font-weight="700">做到 1M 以上的 %d 家，无一例外都动了旋钮②或③</tspan>；'
               '而纯全注意力那一档最高只到 256K。'
               '<tspan font-weight="700">最硬的对照来自 MiniMax 自己：'
               '01 用 7:1 线性做到 10M，M2 退回纯全注意力只剩 192K ——&#160;'
               '同一家、同一批人，差 50 倍。</tspan>' % len(_1m), fill="#174ea6")

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
