# -*- coding: utf-8 -*-
"""专题三 · 贯穿全篇的主线图 —— 一层 Transformer 的张量形状，
   以及三个旋钮各自动了图上的哪一处。

⭐ **为什么要有这一组。** 2026-09-07 现场指定：

    「我说的是这本书里边的这个图。」（附 Scaling Book 那张标满形状的
      Transformer 层示意图）—— 确认它作为专题三的贯穿图。

⭐⭐ **这张图凭什么能贯穿全篇。**

   把一层 Transformer 每一步的**张量形状**都标出来之后，会看见一件事：
   **图上带 `S`（KV 长度）的量，只有 K 和 V 那两条支路的输出 `BSKH`。**
   而 `S` 是**唯一会随对话越变越长**的那一维 ——&nbsp;
   所以**全图唯一需要跨 token 留下来的东西，就是它们**。那就是 KV cache。

   于是整个专题三可以收成一句：**都是在跟 KV cache 较劲。**

     · 旋钮① 每个 token 存多少 → 动产生 K/V 的支路：**让存下来的每一份更小**
     · 旋钮② 每个 query 看多少 → 动 mask 那一格：**KV 还在，但每步只读一部分**
     · 旋钮③ 换一套数学       → **把变长的 KV 换成一个固定大小的状态**

   ⭐ 旋钮③ 那张图的杀伤力在于：**读输出形状就能看见 S 消失了**，
      不需要相信任何人的说法。这是本组图最值钱的一格。


⛔⛔ **2026-09-08 换口径。** 现场原话：
    「FlashAttention 现在已经是标配了，它在各种 attention 模式下都是一样的，
      所以在这一讲里没有什么讨论的意义。**咱们也不要提那个中间的 A 矩阵是
      一个 N×N 大矩阵这个事** ——&nbsp;咱们就讨论 **KV cache 每一个 level** 的事。」

⭐⭐ 于是这组图的主脊换了一根，而且**换完更统一**：

    旧：「图上唯一一个**随长度平方长大的量**是 BTSKG」——&nbsp;那是**存储口径**，
        而 FlashAttention 早就让它不落地了。**讲一个不存在的显存开销，是在教旧常识。**

    新：**图上唯一需要跨 token 留下来的，只有 K 和 V。**
        三个旋钮全都在动它，只是动的面不同：
          · 旋钮① ——&nbsp;**存多大**（MLA／GQA／K=V 共享）
          · 旋钮② ——&nbsp;**每步要读多少**（稀疏／滑窗：KV 还在，但只读一部分）
          · 旋钮③ ——&nbsp;**干脆换成一个固定大小的状态**（线性）
        ⭐ 这条正好接上 §零那张「解码时又变回 RNN」图 Ⓒ：解码每步搬 W ＋ 一路变长的 KV。

⛔ 所以这个文件里**不许再出现「那个矩阵有多大」这类存储口径的说法**
  （32 GiB 那句已删）。O(N²) 仍然可以讲 ——&nbsp;但它是**计算量**，不是显存。

⛔ **为什么五张图必须由同一个脚本吐出来，别分成五个文件。**
   这个教学装置的全部价值，建立在「五张图除了高亮那一处以外**完全一样**」上。
   分开画，它们**一定会漂**——而且漂了不报错、不难看，只是读者再也对不齐。
   所以这里的写法是：**同一份 STAGES 数据画五遍**，变体只声明
   「哪几格是热的」和「要不要替换掉某一段」。⛔ 别改成五个脚本。

📌 **出处与版权**：图式借自 *How to Scale Your Model*
   （公开地址 jax-ml.github.io/scaling-book），**本图为重画**，
   形状记号（B/T/S/D/F/H/N/K/G）沿用原书，画法与配色是本课自己的。
   ⛔ 引用只写这个公开地址 —— **中译镜像挂在私人域名上，不进公开仓库。**
"""
# ⛔ 2026-09-07 的教训：专题三整体重编号（插入 §零 RNN）时，迁移脚本只扫了
#    topic03-build.py，**漏了这里** —— 图里的 §X.Y 也是跨节指针，一样会指死。
#    ⭐ 靠 topic02-lint-xref.py 报出来才发现（它查的是渲染后的 HTML，图在里面）。
#    📌 以后再动节号：**build 脚本和所有 fig 脚本一起扫。**
import io
import os
import xml.dom.minidom

# ── 配色 ────────────────────────────────────────────────────────────
# ⭐⭐ 2026-09-08 对齐专题一配色。这个文件**有自己的 t()**，
#   所以 topic03_draw 里那套自动降档管不到它 ——&nbsp;
#   ⛔ 这正是「共用基元」没做彻底留下的口子：改了库，这里纹丝不动。
#   📌 权宜之计是在本地 t() 里装同一张降档表（下面 INK900）；
#     ⭐ 真正的修法是把这个文件也迁到 topic03_draw，**记在待办里**。
BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
GY2 = "#80868b"
PU, CY, BR = "#9334e6", "#00838f", "#b06000"
DIM = "#9aa0a6"          # 被压灰的部分（原 #bdc1c6 在白底上对比度太低）
# 500 主色 → 900 文字色。跟 topic03_draw.INK900 保持一致。
INK900 = {"#1a73e8": "#174ea6", "#1e8e3e": "#0d652d", "#d93025": "#a50e0e",
          "#e8710a": "#b06000", "#f9ab00": "#b06000", "#9334e6": "#681da8",
          "#8430ce": "#681da8", "#00838f": "#007b83", "#12b5cb": "#007b83"}
DIMBG = "#fafafa"
# 每个主色对应的浅底 —— 热的那一格用它，跟压灰的部分拉开层次
# 500 主色 → 50 浅底。⛔ 换紫色时这里也得跟着换 —— 上次就漏了这张表，构建当场挂掉。
TINT = {"#1a73e8": "#e8f0fe", "#e8710a": "#fef7e0",
        "#9334e6": "#f3e8fd", "#8430ce": "#f3e8fd", "#1e8e3e": "#e6f4ea"}
INK = "#202124"

W = 1400
# ⭐⭐⭐ 2026-09-13 第二刀：注解栏搬走之后，右边那 700px **从 y≈400 往下全是空的**，
#   而流程图还挤在左边 660px 里、靠 10.5px 的标签活着。
#   ⛔ 「注解栏搬走了」和「版面重排」是两件事 —— 只做前一件，省下的地方没人用。
# ⭐ 于是改成**上下结构**：流程图吃满整幅宽（字号随之从 10.5 抬到 15–16），
#   「三个筐」那个画面挪到流程图下面，也吃满整幅宽。
CX = 700                 # 主列中心（＝整幅正中）
QX, KX, VX = 320, 700, 1080  # Q / K / V 三列
MMW = 170                # 主列矩阵乘框的半宽
QKVW = 110               # Q/K/V 支路框的半宽
# 字号：这张图全课曝光最多，**标签不许再回到 svgsm 的 10.5px 地板**
FS_SHAPE = 15            # 形状记号（BTD / BSKH …）
FS_NAME = 16             # 算子名（softmax / 输出投影 …）


def wpx(s, size=11.5):
    """粗估纯文本像素宽 —— CJK 约一个字号宽，ASCII 约一半多。

    ⛔ 别用 `len() * 常数`：中英混排必然偏小，两段文字会撞在一起。
    """
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


# ════════════════════════════════════════════════════════════════════
# 一层 Transformer 的分段。**这份数据五张图共用，别为某一张改它。**
#   kind: mm=矩阵乘 / op=一元算子 / io=纯标注 / add=残差 / mlp=折叠的 MLP
#   key : 变体拿它点名「哪一格是热的」
# ════════════════════════════════════════════════════════════════════
def base_stages():
    """返回 [(key, kind, x, 内容…)]，纵向顺序即数据流向。"""
    return [
        ("x",    "io",  CX, "X", "BTD", "一层的输入"),
        # ── Q / K / V 三条支路 ───────────────────────────────────────
        ("q",    "mm",  QX, "BTD", "W_Q · DNH", "BTNH", "Q"),
        ("k",    "mm",  KX, "BSD", "W_K · DKH", "BSKH", "K"),
        ("v",    "mm",  VX, "BSD", "W_V · DKH", "BSKH", "V"),
        ("rs1",  "op",  QX, "reshape", "BTNH → BTKGH", ""),
        # ── 注意力核心：那个平方大的矩阵 ─────────────────────────────
        ("qk",   "mm",  CX, "BTKGH", "BSKH", "BTSKG", "Q·Kᵀ"),
        ("mask", "op",  CX, "＋ masks", "谁能看谁", ""),
        ("sm",   "op",  CX, "softmax", "BTSKG（注意力矩阵，记作 A）", "★"),
        # ⛔ 2026-09-13 读者审稿最重的一条：这一格原来叫 "S·V"，而上一格刚说完
        #   「记作 A」——&nbsp;图自己在打架。更糟的是形状记号里 S ＝ KV 长度，
        #   于是「在形状里找 S」这条全讲重复最多的指令，指向了三个不同的东西。
        #   ⭐ 判据：**一个符号在一张图里只能有一个意思**，哪怕另一个意思是论文的惯例。
        ("av",   "mm",  CX, "BTSKG", "BSKH", "BTKGH", "A·V"),
        # ── 收尾 ────────────────────────────────────────────────────
        ("rs2",  "op",  CX, "reshape", "BTKGH → BTNH", ""),
        ("wo",   "mm",  CX, "BTNH", "W_O · NHD", "BTD", "输出投影"),
        ("res1", "add", CX, "＋ 残差", "BTD", ""),
        ("n1",   "op",  CX, "norm", "BTD", ""),
        ("mlp",  "mlp", CX, "MLP（本专题一个字都不改它）",
                            "BTD ·DF→ BTF ─gelu⊛─ BTF ·FD→ BTD", ""),
        ("res2", "add", CX, "＋ 残差 → norm", "BTD　→ 下一层", ""),
    ]


def linear_stages():
    """旋钮③：把注意力核心那四格换成线性版的两格。

    ⭐ 这里**只换 qk/mask/sm/av 这一段**，前后一律不动 ——
       正是「前后不动」才让「中间这段变了」看得见。
    ⭐⭐ 全图最值钱的一格在 `st`：它的输出形状是 `BKHH`，
        **S 从形状里消失了** —— 不用相信任何说法，读形状就够。
    """
    out = []
    for s in base_stages():
        if s[0] == "qk":
            out.append(("st", "mm", CX, "BSKH（K）", "BSKH（V）", "BKHH",
                        "先算 KᵀV ＝ 状态"))
            out.append(("qs", "mm", CX, "BTKGH（Q）", "BKHH（状态）",
                        "BTKGH", "再拿 Q 去读状态"))
        elif s[0] in ("mask", "sm", "av"):
            continue                      # softmax 和那个平方矩阵一起没了
        else:
            out.append(s)
    return out


# ════════════════════════════════════════════════════════════════════
# 五个变体。hot 为空集＝全彩（底图）；否则 hot 以外一律压灰。
# ⛔ note 每行 ≤ 约 46 个汉字，右栏宽 700px，超了会顶出去。
# ════════════════════════════════════════════════════════════════════
VARIANTS = [
    dict(
        f="fig3-tx-base", stages=base_stages, hot=set(), accent=BL,
        title="主线图 · 一层 Transformer，每一步的张量形状",
        lead="⭐ 这张图会在整个专题里反复出现。每讲一个方案，就把它重画一遍，"
             "只把被改动的那一处点亮 —— 你永远知道自己在图上的哪儿。",
        panel=("★ 先只看一件事：图上什么东西需要留到下一个 token", RD, [
            ("★", "在形状里找 <b>S</b>（KV 长度）——&#160;<b>只有两处「留得下来」的带它</b>：", RD),
            ("", "<b>K 的输出 BSKH</b> 和 <b>V 的输出 BSKH</b>", RD),
            ("⭐", "<b>S 是唯一会随对话越变越长的那一维</b>", RD),
            ("", "", None),
            # ⛔ 2026-09-12 二轮学生数了一遍：图上带 S 的形状有 **9 处**
            #   （K/V 的进出、两个 matmul 的操作数、BTSKG ×3）。而这里原来写
            #   「别的量都不带 S」—— 一个照做的学生会找到 9 个，然后怀疑自己。
            # ⭐ 判据：**「在图上找 X」这种指令，必须是图上真的做得到的动作。**
            ("⚠️", "图上带 S 的形状<b>不止两处</b>（K/V 的进出、两个 matmul 的操作数…）", GY),
            ("", "<b>「带 S」和「要跨 token 留下来」是两件事</b> ——&#160;", GY),
            ("", "中间那些带 S 的是<b>算完就扔的</b>（FlashAttention 连物化都不物化）", GY),
            ("", "→ 所以<b>唯一要跨 token 留下来的，是 K 和 V 的输出</b>", GY),
            ("", "", None),
            ("⭐⭐", "<b>整个专题三，就是在跟这一份 KV cache 较劲。</b>", BL),
            ("①", "<b>让每一份更小</b> —— 改产生 K/V 的那两条支路", BL),
            ("②", "<b>KV 照存，但每步只读一部分</b> —— 改 mask 那一格", OR),
            ("③", "<b>换成一个固定大小的状态</b> —— 换一套数学，S 直接消失", PU),
            ("", "", None),
            ("📌", "中间那一步（softmax 前后）<b>不落地</b>，不占显存 ——", GY),
            ("", "FlashAttention 已是标配，<b>本讲不展开它</b>", GY),
        ]),
        # ⛔ 这条是记号表，一行放不下 —— **拆两行，不要靠缩字号硬塞**。
        # ⭐ 字母表提到图**顶上**（legend），底下只留出处。
        legend="B 批量 · T query 长度 · <tspan font-weight=\"700\">S KV 长度</tspan>"
               " · D d_model · F MLP 隐层 · H 头维 · N query 头数 · "
               "K KV 头数 · G ＝ N∕K",
        markkv=True,      # ⭐ 底图上把那两处 BSKH 点出来
        foot="📌 形状记号沿用 How to Scale Your Model"
             "（jax-ml.github.io/scaling-book），本图为重画。",
        acct=[("➜", "<b>这笔账在后面四张图里一直挂着</b> ——&#160;"
                    "每张会说清它把这笔账动到了哪一格。", GY)],
    ),
    dict(
        f="fig3-tx-k1", stages=base_stages, hot={"k", "v"}, accent=BL,
        title="旋钮① 每个 token 存多少 —— 动的是这两条支路",
        lead="点亮的两格就是这个旋钮的全部作用域 ——&#160;<b>正是带 S 的那两处</b>。"
             "⭐ 它省的是<b>每一份 KV 有多大</b>，不是算力。",
        panel=("① 三种改法，都只改这两格", BL, [
            ("MQA", "K 从多个头砍到 <b>1 个</b>（K＝1）", BL),
            ("GQA", "砍到几个，多个 query 头共用一个 KV 头（<b>G＝N∕K</b>）", BL),
            ("MLA", "不砍头，改成<b>先压到一个低秩的 c，用时再上投影</b>", BL),
            ("", "", None),
            ("⭐", "<b>看形状就知道省在哪</b>：KV cache 存的是 <b>BSKH</b>", RD),
            ("", "里面那个 <b>K</b> 变小，缓存就等比变小 —— 就这么直接", GY),
            ("", "", None),
            ("⚠️", "<b>它不省核心那两个 matmul 的 FLOPs。</b>Q·Kᵀ 出来的还是 BTSKG，", OR),
            ("⭐", "<b>但训练侧它省：</b>W_K/W_V 跟着缩 —— 参数、优化器状态、"
                   "投影的 FLOPs 三样都省。", GR),
            ("", "该算的乘加一次不少 —— GQA 靠的是<b>把 KV 头广播开</b>再算", OR),
            ("", "", None),
            ("📌", "MLA 那条另有一处麻烦：<b>RoPE 必须单独走一路</b>，", GY),
            ("", "因为上投影吸收不了带位置旋转的那几维（见 §5.3）", GY),
        ]),
        foot="⭐ 一句话记住这个旋钮：<b>它改的是「存什么」，不是「算什么」。</b>"
             "所以它治的是显存墙，治不了 O(N²) 的计算量。",
        acct=[("➜ 这一格", "<b>把 488 这个数直接改小</b> ——&#160;"
                         "它动的就是这笔账的分子。", BL)],
    ),
    dict(
        f="fig3-tx-k2", stages=base_stages, hot={"mask", "sm", "qk"}, accent=OR,
        title="旋钮② 每个 query 看多少 —— 动的是 mask 那一格",
        lead="点亮的是 Q·Kᵀ 和它后面那个 mask。"
             "⭐ <b>矩阵的形状一点没变，变的是里面有多少格子真的要算。</b>",
        panel=("② 稀疏，本质上就是换一张 mask", OR, [
            ("", "标准因果注意力的 mask 是一个<b>下三角</b> —— 看全部历史", GY),
            ("⭐", "所谓稀疏，就是<b>把这张 mask 换成别的形状</b>：", OR),
            ("", "", None),
            ("SWA", "只留主对角线附近一条带 —— 只看最近 W 个", OR),
            ("NSA", "三条路并存：压缩看全局 ＋ top-k 挑重点 ＋ 滑窗看近处", OR),
            ("DSA", "拿一个<b>轻量索引器</b>先打分，只留 top-k 那几块", OR),
            ("CSA", "先把每 4 个 token 压成 1 个 entry，<b>在压缩后的格上</b>挑", OR),
            ("", "", None),
            ("⭐", "<b>它跟旋钮① 是正交的</b>：一个改 KV 存多少，", RD),
            ("", "一个改这张 mask —— 所以两个可以同时上（GLM-5 就是）", RD),
            ("", "", None),
            ("⚠️", "<b>纸面省下的 FLOPs，要 kernel 跟上了才算数。</b>", RD),
            ("", "不规则的 mask 对硬件不友好，这是这一支真正的门槛", GY),
        ]),
        foot="⭐ 一句话记住这个旋钮：<b>它改的是「看哪些」，形状不变、只是很多格子不算。</b>"
             "所以省的是 FLOPs，而 KV 该存多少还得存多少。",
        acct=[("➜ 这一格", "<b>这笔账一个字节都不变。</b>"
                         "它省的是「每步读多少」，不是「存多少」。", OR),
              ("", "⛔ 所以看着 488 别以为它在往下走 ——&#160;<b>它原地不动</b>。", OR)],
    ),
    dict(
        f="fig3-tx-k3", stages=linear_stages, hot={"st", "qs"}, accent=PU,
        title="旋钮③ 换一套数学 —— 把那个平方大的矩阵整个删掉",
        lead="⛔ 这一张跟前两张不一样：<b>它不是高亮，是替换。</b>"
             "原来那四格（Q·Kᵀ → mask → softmax → S·V）没了，换成点亮的这两格。",
        panel=("③ 不用听解释 —— 读输出形状就够了", PU, [
            ("", "softmax 的分母要对<b>所有位置</b>求和，所以它<b>锁死了乘法顺序</b>：", GY),
            ("", "必须先 Q·Kᵀ（于是必须造出那个平方大的矩阵），再乘 V", GY),
            ("⭐", "<b>把 softmax 拿掉，乘法就可以重新结合</b>：先 KᵀV，再乘 Q", PU),
            ("", "", None),
            ("⭐⭐", "<b>看点亮那格的输出形状：BKHH。</b>", RD),
            ("", "<b>S 不见了。</b>状态大小只跟头维 H 有关，<b>跟序列多长无关</b>。", RD),
            ("", "这就是 O(N²) → O(N) 的全部内容，写在形状里，不用相信谁", RD),
            ("", "", None),
            ("⛔", "<b>但代价也写在同一格里。</b>因果版不能真的这么一乘 ——", OR),
            ("", "状态要<b>按 t 一步步累加</b>，于是<b>串行回来了</b>", OR),
            ("", "chunk 化（分块并行）就是为了把并行度再找回来 —— 见 §7.2", OR),
            ("", "", None),
            ("⚠️", "<b>它不是「更快的 attention」，是另一个模型。</b>", RD),
            ("", "固定大小的状态 → <b>信息必然有损</b>，长程精确检索会力不从心", GY),
        ]),
        foot="⭐ 这是三个旋钮里<b>唯一改变了模型能表达什么</b>的一个。"
             "另外两个改的是存法和看法，这一个改的是<b>数学本身</b>。",
        acct=[("➜ 这一格", "<b>这笔账整个作废。</b>S 从形状里消失，"
                         "状态跟长度无关 ——&#160;", PU),
              ("", "上面这张按 S＝128K 算出来的表，<b>在旋钮③ 下根本不成立</b>。", PU)],
    ),
    dict(
        f="fig3-tx-fa", stages=base_stages, hot={"qk", "sm", "av"}, accent=GR,
        title="FlashAttention —— 图上一处都没改，改的是「落不落地」",
        lead="⭐ 点亮的三格<b>跟底图一模一样</b>：同样的算子、同样的形状、"
             "同样的 FLOPs。<b>它不是第四个旋钮。</b>",
        panel=("⊘ 不在旋钮上：它删掉的不是计算，是搬运", GR, [
            ("", "朴素写法会把中间那一步 <b>写进 HBM 再读回来</b>；融合之后不会：", GY),
            ("", "写 S → 读 S → 写 P → 读 P，<b>四趟</b>", GY),
            ("⭐", "而这一步<b>本来就不该落地</b>——&#160;融合之后它只在片上过", RD),
            ("", "（序列 128K 时的账，见 §3.2）", GY),
            ("", "", None),
            ("⭐⭐", "<b>FlashAttention 把这四趟全删了</b>：分块算，", GR),
            ("", "中间结果只在片上暂存里走一遭，<b>从不落 HBM</b>", GR),
            ("", "", None),
            ("📌", "<b>所以它跟三个旋钮不是一类东西，可以同时用。</b>", BL),
            ("", "旋钮改的是「算什么」，它改的是「算出来的东西放哪」", GY),
            ("", "今天所有方案的实现里都有它 —— 它是地板不是选项", GY),
            ("", "", None),
            ("⚠️", "<b>它也不是免费的</b>：在线 softmax 要多做一遍重标定，", OR),
            ("", "融合之后实测也只跑到约 35%，三层原因见 §3.6", OR),
        ]),
        foot="⭐ 一句话记住它：<b>三个旋钮改的是这张图，它改的是这张图怎么跑。</b>"
             "所以它跟谁都不冲突 —— 这也是它能成为默认实现的原因。",
        acct=[("➜ 这一格", "<b>这笔账一个字节都不变。</b>"
                         "FlashAttention 改的是「怎么跑」，", GR),
              ("", "它省的是那个<b>从来没落地过</b>的中间矩阵 ——&#160;跟 488 无关。", GR)],
    ),
]


# ════════════════════════════════════════════════════════════════════
def render(v):
    p = []
    hot, acc = v["hot"], v["accent"]

    def live(key):
        """这一格是不是"亮"的。hot 为空＝底图，全亮。"""
        return (not hot) or (key in hot)

    def C(key, c):
        return c if live(key) else DIM

    def t(x, y, s, cls="svgsm", fill=None, bold=False, size=None, anchor=None):
        # ⭐ 自动降档：500 主色画文字 → 换成 900 深色变体（对齐专题一）。
        fill = INK900.get(fill, fill)
        # ⛔ 这份 t() 是这个文件自己的复制品，原来**没有字号地板断言** ——
        #   于是 13 处 10px 一直安安静静地待在全课曝光最多的那张图上。
        assert size is None or size >= 11, \
            "字号 %s 太小（全课地板 11px）：%s" % (size, s[:30])
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

    def line(x1, y1, x2, y2, c, sw=1.3, dash=None):
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
                 'stroke-width="%s"%s/>'
                 % (x1, y1, x2, y2, c, sw,
                    ' stroke-dasharray="%s"' % dash if dash else ''))

    p.append("")                                    # svg 开标签占位

    # ── 标题 ────────────────────────────────────────────────────────
    t(0, 22, v["title"], "svglbl", INK, size=19)
    t(0, 48, v["lead"].replace("<b>", '<tspan font-weight="700">')
                      .replace("</b>", "</tspan>"), fill=GY, size=15)
    LEGH = 0
    if v.get("legend"):
        # ⛔ 记号表原来在图的最底下 ——&nbsp;读者在第 3 行就撞上 BTNH，
        #   却要滚过整张 1374px 的图才知道那些字母是什么。
        box(0, 62, W, 30, "#f8f9fa", "#e8eaed", 6)
        t(12, 82, "读这张图先认字母：", None, GY2, size=15)
        t(150, 82, v["legend"], None, GY, size=15)
        LEGH = 40

    # ── 上：数据流（吃满整幅宽）────────────────────────────────────
    y = 84 + LEGH
    stages = v["stages"]()
    # 把 Q/K/V 那三格摆在同一行：记下它们在列表里的位置
    qkv = [i for i, s in enumerate(stages) if s[0] in ("q", "k", "v")]
    pos = {}                                        # key → (x, y_in, y_out)
    i = 0
    while i < len(stages):
        s = stages[i]
        key, kind, x = s[0], s[1], s[2]

        if kind == "io":
            _, _, _, name, shape, note = s
            t(x, y + 14, name, fill=C(key, INK), bold=True, size=15,
              anchor="middle")
            t(x + 26, y + 14, shape, fill=C(key, GY), size=FS_SHAPE)
            t(x + 26 + wpx(shape, FS_SHAPE) + 12, y + 14, "——&#160;" + note,
              fill=C(key, DIM if not live(key) else GY2), size=FS_SHAPE)
            pos[key] = (x, y, y + 20)
            y += 34

        elif kind == "mm" and i in qkv:
            # 三条支路并排，只在第一条时把三个都画掉
            if i == qkv[0]:
                for j in qkv:
                    k2, _, x2, lin, rin, out, nm = stages[j]
                    c = C(k2, BL)
                    box(x2 - QKVW, y + 12, 2 * QKVW, 34,
                        "#e8f0fe" if live(k2) else DIMBG, c, 5,
                        1.8 if live(k2) and hot else 1)
                    t(x2 - QKVW + 6, y + 8, lin, fill=C(k2, BL),
                      size=FS_SHAPE)
                    t(x2 + QKVW - 6, y + 8, rin, fill=C(k2, PU),
                      size=FS_SHAPE, anchor="end")
                    p.append('<circle cx="%d" cy="%d" r="9" fill="none" '
                             'stroke="%s" stroke-width="1.6"/>'
                             % (x2, y + 29, c))
                    p.append('<circle cx="%d" cy="%d" r="2.6" fill="%s"/>'
                             % (x2, y + 29, c))
                    if v.get("markkv") and k2 in ("k", "v"):
                        # ⭐ 「在形状里找 S」是全讲重复最多的指令 ——
                        #   那就让底图**自己把这两处标出来**，别让人空手去找。
                        p.append('<rect x="%d" y="%d" width="82" height="26" '
                                 'rx="5" fill="#fce8e6"/>'
                                 % (x2 - 41, y + 44))
                    t(x2, y + 62, out, fill=C(k2, INK), bold=True,
                      anchor="middle", size=FS_NAME)
                    t(x2, y + 84, nm, fill=c, bold=True, size=FS_NAME,
                      anchor="middle")
                    pos[k2] = (x2, y + 12, y + 90)
                    line(CX, y - 10, x2, y - 2, C(k2, GY), 1)
                    line(x2, y - 2, x2, y + 12, C(k2, GY), 1)
                y += 104
            i += 1
            continue

        elif kind == "mm":
            _, _, _, lin, rin, out, nm = s
            c = C(key, acc if hot else BL)
            box(x - MMW, y + 12, 2 * MMW, 34,
                DIMBG if not live(key) else
                (TINT[acc] if hot else "#e8f0fe"), c, 5,
                2.2 if (hot and live(key)) else 1)
            if hot and live(key):
                # ⛔ 别把上沿放在 y+6 —— 输入标签的基线在 y+8，虚线会从字中间穿过，
                #    渲染出来像删除线。框要**连输入标签和输出形状一起框住**。
                box(x - MMW - 8, y - 6, 2 * MMW + 16, 80, "none", c, 8,
                    1, "5,4")
            t(x - MMW + 6, y + 8, lin, fill=C(key, BL), size=FS_SHAPE)
            t(x + MMW - 6, y + 8, rin, fill=C(key, PU), size=FS_SHAPE,
              anchor="end")
            p.append('<circle cx="%d" cy="%d" r="9" fill="none" stroke="%s" '
                     'stroke-width="1.6"/>' % (x, y + 29, c))
            p.append('<circle cx="%d" cy="%d" r="2.6" fill="%s"/>'
                     % (x, y + 29, c))
            t(x, y + 64, out, fill=C(key, INK), bold=True, anchor="middle",
              size=FS_NAME)
            t(x - MMW - 24, y + 34, nm, fill=c, bold=True, size=FS_NAME,
              anchor="end")
            pos[key] = (x, y + 12, y + 72)
            y += 90

        elif kind in ("op", "add"):
            _, _, _, lab, sub, mark = s
            c = C(key, acc if hot else GY)
            # ⛔ 这里原来按 12/11px 估宽，而字是 10.5px 画的 —— 侥幸没顶出去。
            #   现在字号抬到 15/16，**估宽必须用同一个字号**，否则必撞。
            wbox = max(220, wpx(lab, FS_NAME) + wpx(sub, FS_SHAPE) + 56)
            box(x - wbox // 2, y, wbox, 32,
                "#f8f9fa" if live(key) else DIMBG, c, 5,
                2.2 if (hot and live(key)) else 1)
            if hot and live(key):
                box(x - wbox // 2 - 6, y - 6, wbox + 12, 44, "none", c, 8,
                    1, "5,4")
            t(x - wbox // 2 + 14, y + 22, lab, fill=c, bold=True,
              size=FS_NAME)
            t(x - wbox // 2 + 14 + wpx(lab, FS_NAME) + 12, y + 22, sub,
              fill=C(key, INK if mark == "★" else GY),
              bold=(mark == "★"), size=FS_SHAPE)
            if mark == "★":
                t(x + wbox // 2 + 14, y + 22,
                  "★ <tspan font-weight=\"700\">这一步不落地，贵在要算的次数</tspan>",
                  fill=C(key, RD), size=FS_SHAPE)
            pos[key] = (x, y, y + 32)
            y += 52

        elif kind == "mlp":
            _, _, _, lab, sub, _ = s
            c = C(key, GR)
            box(x - 330, y, 660, 58, "#e6f4ea" if live(key) else DIMBG, c, 6)
            t(x, y + 24, lab, fill=c, bold=True, anchor="middle",
              size=FS_NAME)
            t(x, y + 46, sub, fill=C(key, GY), anchor="middle",
              size=FS_SHAPE)
            pos[key] = (x, y, y + 58)
            y += 76

        i += 1

    # 主干连线：按顺序把相邻两格接起来（Q/K/V 那一段单独接）
    order = [s[0] for s in stages]
    for a, b in zip(order, order[1:]):
        if a in ("q", "k", "v") or b in ("q", "k", "v"):
            continue
        if a not in pos or b not in pos:
            continue
        line(CX, pos[a][2], CX, pos[b][1], C(b, GY), 1.2)
    # Q/K/V 汇回主干
    # ⛔ 线性版没有 "qk" 这一格（它被 st/qs 替掉了），所以**先定位下一格再连线** ——
    #    原先这里写死 pos["qk"]，一跑线性变体就 KeyError。
    #    ⭐ 教训：变体会删格子，任何按 key 硬取位置的地方都要先问「它还在吗」。
    nxt = "qk" if "qk" in pos else "st"
    if "rs1" in pos:                                # 标准版：Q 先 reshape
        line(QX, pos["q"][2], QX, pos["rs1"][1], C("q", GY), 1.2)
        line(QX, pos["rs1"][2], QX, pos["rs1"][2] + 10, C("q", GY), 1.2)
        line(QX, pos["rs1"][2] + 10, CX, pos["rs1"][2] + 10, C("q", GY), 1.2)
        line(CX, pos["rs1"][2] + 10, CX, pos[nxt][1], C("q", GY), 1.2)
    # ⛔ 2026-09-07：原先 K/V 是**直着降下来**再横插进目标格的上沿，而 K 那一列
    #    x 正好等于主列 CX —— 于是那条竖线**从右侧输入标签 BSKH 正中间穿过去**，
    #    渲染出来活像一条删除线。几何 lint 抓不到（线不是文字，不算撞车）。
    # ⭐ 教训：**连线要走"路由列"，不要从源头直降。** 源头的 x 是按可读性排的，
    #    它跟沿途有什么东西完全无关，直降迟早会穿过某个标签。
    for src, tgt, rx in (("k", nxt, CX + MMW + 46),
                         ("v", "av" if "av" in pos else "st", CX + MMW + 96)):
        if tgt not in pos or src not in pos:
            continue
        c = C(src, GY)
        sx, sy = pos[src][0], pos[src][2]
        ymid = pos[tgt][1] + 17                    # 从目标格的**右腰**进，不走上沿
        line(sx, sy, sx, sy + 8, c, 1.2)
        line(sx, sy + 8, rx, sy + 8, c, 1.2)
        line(rx, sy + 8, rx, ymid, c, 1.2)
        line(rx, ymid, CX + MMW, ymid, c, 1.2)
    if "st" in pos and "qs" in pos:                 # 线性版：Q 直接下到 qs
        line(QX, pos["q"][2], QX, pos["qs"][1] - 8, C("q", GY), 1.2)
        line(QX, pos["qs"][1] - 8, CX - MMW, pos["qs"][1] - 8, C("q", GY), 1.2)

    FLOW_H = y

    # ── 下：一个画面 ＋ 一句话 ─────────────────────────────────────
    # ⭐⭐⭐ 2026-09-13 现场拍板「要搬」：**注解栏整段搬进正文**。
    #   现场的判据是「图是图，字是字」——&nbsp;这一栏原来有十几行解释，
    #   那是正文的活；图这边只留**一个画面 ＋ 一句这一格干了什么**。
    # ⛔ 搬走的不是删掉：每一条都原样落在 topic03-build.py 里对应 figure 的下面，
    #   而且在那儿它能被搜索、被复制、在手机上会自动折行 ——&nbsp;SVG 里三样都不行。
    # ⭐ 搬完还要**把地方让出来**：这一栏从右边挪到了流程图下面，整幅宽，
    #   于是流程图那边的标签才能从 10.5px 抬到 15px。
    ph, pcol, rows = v["panel"]
    PX, PW = 0, W
    PY0 = FLOW_H + 20

    PICH = 190
    box(PX, PY0, PW, PICH, "#fff", INK, 10)
    t(PX + 22, PY0 + 36, "这一层算完，桌上的东西分三堆", "svglbl", INK, size=20)
    BW = (PW - 88) / 3.0
    BINS = [
        ("常驻的设备", "权重 W", "所有人共用，不随对话变", GY, "#f1f3f4"),
        ("草稿纸", "中间那些量", "算完就扔，不留到下一步", GR, "#e6f4ea"),
        ("锁进柜子的", "K 和 V", "每来一个 token 就多一格", RD, "#fce8e6"),
    ]
    for i2, (name, what, why, col2, tint) in enumerate(BINS):
        bx = PX + 22 + i2 * (BW + 22)
        box(bx, PY0 + 56, BW, 116, tint, col2, 8)
        t(bx + 16, PY0 + 86, name, None, col2, bold=True, size=17)
        t(bx + 16, PY0 + 118, what, None, col2, bold=True, size=20)
        t(bx + 16, PY0 + 152, why, None, GY, size=15)

    # 这一格干了什么 —— 每张一句，字大
    SY = PY0 + PICH + 16
    nacct = len([1 for tg, tx, _ in v.get("acct", []) if tx])
    SH = 58 + 30 * nacct + 30
    box(PX, SY, PW, SH, "#fff", pcol, 10)
    t(PX + 22, SY + 38, ph, "svglbl", pcol, size=21)
    acct = list(v.get("acct", []))
    yy2 = SY + 76
    for tag, txt, col2 in acct:
        if not txt:
            continue
        s2 = txt.replace("<b>", '<tspan font-weight="700">') \
                .replace("</b>", "</tspan>")
        if tag:
            t(PX + 22, yy2, tag, None, col2 or GY, bold=True, size=17)
            t(PX + 22 + max(wpx(tag, 17), 26) + 14, yy2, s2, None,
              col2 or GY, size=17)
        else:
            t(PX + 22, yy2, s2, None, col2 or GY, size=17)
        yy2 += 30
    t(W - 22, SY + SH - 16, "📌 这一格的完整讲解在图下面的正文里", None, GY2,
      size=15, anchor="end")
    PANEL_H = SY + SH

    # ── 底部落点带 ──────────────────────────────────────────────────
    FY = max(FLOW_H, PANEL_H) + 14
    if v.get("markkv"):
        t(QX - QKVW, FLOW_H + 6, "⬆ 红底那两处就是它 ——　全图只有这两个"
          "输出要跨 token 留下来", None, RD, size=17)
    foots = v["foot"]
    foots = [foots] if isinstance(foots, str) else list(foots)
    FH = 22 + 26 * len(foots)
    box(0, FY, W, FH, "#e8f0fe" if not hot else "#f8f9fa", acc, 8)
    for i3, ln in enumerate(foots):
        t(16, FY + 30 + i3 * 26,
          ln.replace("<b>", '<tspan font-weight="700">')
            .replace("</b>", "</tspan>"), fill=acc, size=15)
    TOT = FY + FH + 10

    # ⛔ viewBox 高度按真实落点算，别写死 —— 加一行就被裁，而且不报错。
    p[0] = ('<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="%s">'
            % (W, TOT, v["title"]))
    p.append("</svg>")
    svg = "\n".join(p)

    # 写盘前自检 ①：SVG 必须能解析（曾有 <u> 之类的非法标签静默吃掉半张图）
    xml.dom.minidom.parseString(svg.replace("&#160;", "&#xa0;"))
    # ②：图是两份文档共用的，不许出现方位词（按名字指，不按位置指）
    for word in ("上一张", "下一张", "上面那张", "下面那张", "前面那张"):
        assert word not in svg, "%s 里出现方位词「%s」" % (v["f"], word)
    # ③：⛔ 私人域名镜像不进公开仓库
    assert "higcp" not in svg, "%s 里出现私人域名" % v["f"]
    io.open(v["f"] + ".svg", "w", encoding="utf-8").write(svg)
    return TOT, len(svg)


if __name__ == "__main__":
    for v in VARIANTS:
        h, n = render(v)
        print("%-16s 高 %-5d %s 字符" % (v["f"], h, format(n, ",")))
    print("⭐ 五张同源：改 base_stages() 会同时改掉全部五张 —— 这正是要的。")
