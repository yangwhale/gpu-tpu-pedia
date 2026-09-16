# -*- coding: utf-8 -*-
r"""专题三 · §6.4c「DeepSeek-V4-Pro 的一条 KV 到底长什么样」

⭐⭐⭐ 2026-09-16 新画。现场原话：
  「它这个 MQA 看上去跟 MLA 一样，就是所有 head 最后变成一个 KV 放一起的
    压缩体，然后是 512。那它那个 64 的 rope 搞哪里去了？……不是让你从
    MaxText 的代码去找，而是从模型架构、V4 的文档论文里去找。」

⛔ 这张图是**上一版讲错之后补的**。之前那版（只读 MaxText 代码）把三件事
  讲拧了：① 说「两条支路的结果加起来」——&nbsp;其实是**拼接**；
  ② 把「不是 MLA」说成了简单 MQA ——&nbsp;论文的正式名字是
  **Shared Key-Value Multi-Query Attention**，它确实是「所有头共用一条压缩体」，
  这一点**用户的直觉是对的**，错的是我把它和 MLA 的差别说成了「有没有压缩体」；
  ③ 完全没交代那 64 维 RoPE 去哪了。

⭐⭐ 这张图要回答的就是三个问题，一个都不能少：
  ① **一条 KV 里装的是什么**（512 怎么分，64 在哪）
  ② **CSA / HCA / 纯滑窗三种层，KV 序列各自长什么样**
  ③ **算的时候要不要恢复** ——&nbsp;答案是不恢复，但**输出侧要反向转一次**

⛔⛔ 刻意没画的：
  ① mHC（Manifold-Constrained Hyper-Connections）——&nbsp;它改的是残差通路，
     不是注意力，塞进来只会稀释主线。§2.2 在论文里是独立一节。
  ② MoE 部分（384 专家、fp4 权重）——&nbsp;不归这一讲管。
  ③ **本图唯一的两组数字是论文与 vLLM 的公开口径**，本课没有 V4 的实测。

📌 出处（全部公开）：
  · 论文：**arXiv 2606.19348**《DeepSeek-V4: Towards Highly Efficient
    Million-Token Context Intelligence》(DeepSeek-AI, 2026-04-26)
    §2.3.1 CSA（式 9–12 压缩、式 13–14 indexer 查询）、§2.3.2 HCA、
    §2.3.3 Other Details（Partial RoPE、式 26 反向旋转、式 27 attention sink）
  · 配置：Hugging Face `deepseek-ai/DeepSeek-V4-Pro` 的 `config.json`
  · 架构说明：Hugging Face `transformers` 文档 `model_doc/deepseek_v4`
  · 实现口径：vLLM 博客《DeepSeek V4 in vLLM: Efficient Long-context Attention》
    (2026-04-24)，`c4a` / `c128a` 的定义与 1M 上下文 KV 账
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

# ⭐ 全部来自 config.json，改动前请回去核，不要凭印象改
N_Q_HEADS = 128       # num_attention_heads
N_KV_HEADS = 1        # num_key_value_heads
HEAD_DIM = 512        # head_dim
ROPE_DIM = 64         # qk_rope_head_dim
NOPE_DIM = HEAD_DIM - ROPE_DIM
MLA_LATENT = 512      # V3.2 kv_lora_rank
MLA_ROPE = 64         # V3.2 qk_rope_head_dim
assert NOPE_DIM == 448
assert MLA_LATENT + MLA_ROPE == 576


def main():
    f = Fig(W, "DeepSeek-V4-Pro 的注意力：每个 token 每层只存一条 512 维向量，"
               "K 和 V 就是同一条，512 里只有末尾 64 维带 RoPE；"
               "CSA 把 4 个 token 压成一条再用索引器挑 1024 条，"
               "HCA 把 128 个压成一条且全部都看，两种层都在后面拼一段 128 的未压缩滑窗；"
               "取用时不做任何解压缩，只在输出侧按负位置反向旋转一次")

    y0 = f.header(
        "DeepSeek-V4-Pro：<tspan font-weight=\"700\">一条 KV 到底长什么样</tspan>",
        "⭐ 三个问题一次说清：<tspan font-weight=\"700\">512 里装了什么</tspan>　·　"
        "<tspan font-weight=\"700\">那 64 维 RoPE 去哪了</tspan>　·　"
        "<tspan font-weight=\"700\">用的时候要不要解压</tspan>",
        [(PU, "上一代 V3.2（MLA）"), (GR, "V4（CSA / HCA）"),
         (OR, "要运行时挑（indexer）")])

    # ══════════ Ⓐ 一条 KV 的内部构造：V3.2 对照 V4 ═══════════════
    PH = 448
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一个 token 在 KV cache 里占多少 ——　"
                 "<tspan font-weight=\"700\">V3.2 的 MLA</tspan>　对照　"
                 "<tspan font-weight=\"700\">V4 的 Shared K=V MQA</tspan>", BL,
                 sub="⭐ 两代都是「所有头共用一条」——　"
                     "<tspan font-weight=\"700\">差别在这一条要不要再展开</tspan>")

    BARX, BARW = 300, 900

    # ── V3.2：512 潜向量 ＋ 64 RoPE，分开存
    f.t(72, py + 56, "V3.2 · MLA", PU, True, 19)
    f.t(72, py + 80, "每 token 每层", GY, size=13)
    w1 = BARW * MLA_LATENT / (MLA_LATENT + MLA_ROPE)
    f.box(BARX, py + 38, w1, 54, "#f3e8fd", PU, 6)
    f.t(BARX + w1 / 2.0, py + 71, "压缩潜向量　512", PU, True, 17, "middle")
    f.box(BARX + w1, py + 38, BARW - w1, 54, "#fce8e6", RD, 6)
    f.t(BARX + w1 + (BARW - w1) / 2.0, py + 71, "RoPE 64", RD, True, 15, "middle")
    f.t(BARX + BARW + 16, py + 71, "＝ 576", INK, True, 17)
    f.t(BARX, py + 116, "⛔ 这 64 是<tspan font-weight=\"700\">加在 512 外面的</tspan>"
                        "　·　K 和 V 要从潜向量<tspan font-weight=\"700\">升维还原</tspan>"
                        "（升维矩阵被「吸收」进 Q 和输出矩阵，所以不必真展开）",
        GY, size=14)

    # ── V4：一条 512，末尾 64 带 RoPE
    f.t(72, py + 196, "V4 · Shared K=V MQA", GR, True, 19)
    f.t(72, py + 220, "每 token 每层", GY, size=13)
    w2 = BARW * NOPE_DIM / HEAD_DIM
    f.box(BARX, py + 178, w2, 54, "#e6f4ea", GR, 6)
    f.t(BARX + w2 / 2.0, py + 211, "NoPE 448（不带位置）", GR, True, 17, "middle")
    f.box(BARX + w2, py + 178, BARW - w2, 54, "#fce8e6", RD, 6)
    f.t(BARX + w2 + (BARW - w2) / 2.0, py + 211, "RoPE 64", RD, True, 15, "middle")
    f.t(BARX + BARW + 16, py + 211, "＝ 512", INK, True, 17)
    f.t(BARX, py + 256, "⭐ 这 64 是<tspan font-weight=\"700\">长在 512 里面的</tspan>"
                        "（partial RoPE，只转末尾那一段）　·　"
                        "<tspan font-weight=\"700\">K 和 V 就是这一条，不升维、不分开</tspan>",
        GY, size=14)

    # ── 落点
    f.box(72, py + 296, 1256, 124, "#e8f0fe", BL, 8)
    f.t(96, py + 328, "⭐⭐ 所以「看上去跟 MLA 一样」——　"
                      "<tspan font-weight=\"700\">这个直觉是对的</tspan>：两代都是"
                      "「所有头共用一条」。", INK, True, 17)
    f.t(96, py + 360, "⛔ 真正的差别只有一句："
                      "<tspan font-weight=\"700\">MLA 那条是「压缩件」，逻辑上要还原成每头的 K 和 V；"
                      "V4 这条就是 K 和 V 本身。</tspan>", INK, size=16)
    f.t(96, py + 392, "→　于是 V4 <tspan font-weight=\"700\">连「吸收」这一手都不需要</tspan>"
                      "　·　查询那边仍有 128 个头（q_lora_rank 1536 下投再上投），"
                      "<tspan font-weight=\"700\">128 个头共读这一条</tspan>", GY, size=15)
    f._pan = None

    yy = f.band(py + PH + 22, "warn", "顺手把这两个数对一下，别记混", [
        "<tspan font-weight=\"700\">V3.2：512 ＋ 64 ＝ 576</tspan>　（潜向量和 RoPE 分两块存）"
        "　　<tspan font-weight=\"700\">V4：512</tspan>　（RoPE 是这 512 里的末尾一段，"
        "<tspan font-weight=\"700\">不额外占地方</tspan>）。",
        "⛔ 再加一件<tspan font-weight=\"700\">只有 V4 有</tspan>的事："
        "既然 K 和 V 是同一条，那<tspan font-weight=\"700\">存一条就等于存了两样</tspan> ——&#160;"
        "光这一手就省一半。代价在 Ⓓ。",
    ])

    # ══════════ Ⓐb 从 MLA 四步推到 V4 ═══════════════════════════
    # ⭐⭐⭐ 2026-09-16 现场追加。原话：「CSA 跟 MLA 是什么关系？是不是这个
    #   变体引进来进化来的？……你如果不是以 MLA 为基础，光压 4:1，还不如
    #   原来那个 MLA 的压缩比高呢。」——&#160;**这个论证是对的**，而且它是
    #   这一整套设计的命门：序列压缩只有建在一个已经很窄的 entry 上才成立。
    # ⚠️ 这条链是**按两边定义推出来的结构演化**，不是论文原话。标在图上了。
    PHB = 300
    pyb = f.panel(0, yy + 26, W, PHB,
                  "Ⓐb 这条 512 是<tspan font-weight=\"700\">哪来的</tspan> ——　"
                  "从 MLA 四步就能推到它", BL,
                  sub="⚠️ 这条链是<tspan font-weight=\"700\">按定义推的结构演化</tspan>，"
                      "论文没说「V4 是 MLA 的变体」")

    CHAIN = (
        (PU, "① MLA（V3.2）",
         ("一条 512 潜向量",
          "＋ 两个升维矩阵还原每头 K/V",
          "＋ 单挂一条 64 的 RoPE key",
          "cache ＝ 576")),
        (BL, "② 删掉升维矩阵",
         ("那条 512 直接当 K，也当 V",
          "所有头共读这一条",
          "＝ Shared K＝V MQA",
          "「吸收」这一手不需要了")),
        (BL, "③ 把 64 收进去",
         ("改成 partial RoPE",
          "只转末尾 64 个通道",
          "cache ＝ 512",
          "⛔ 代价：输出按 −i 转回来")),
        (GR, "④ 沿序列再压",
         ("在这条 512 的基础上",
          "每 m 条合成一条",
          "m ＝ 4 →　CSA",
          "m′ ＝ 128 →　HCA")),
    )
    CW = 314
    for i, (col, title, lines) in enumerate(CHAIN):
        x = 36 + i * (CW + 24)
        f.box(x, pyb + 30, CW, 208, "#fff", col, 8)
        f.box(x, pyb + 30, CW, 4, col, col, 2)
        f.t(x + CW / 2.0, pyb + 62, title, col, True, 17, "middle")
        for k, ln in enumerate(lines):
            f.t(x + 18, pyb + 96 + k * 30, ln, GY, size=13.5)
        if i < 3:
            f.t(x + CW + 12, pyb + 138, "→", GY2, True, 19, "middle")
    f.t(700, pyb + 268,
        "⭐⭐ 所以「看着像 MLA」不是错觉 ——　"
        "<tspan font-weight=\"700\">MQA 是它的形式，MLA 是它的效果</tspan>",
        INK, True, 17, "middle")
    f._pan = None

    yy = f.band(pyb + PHB + 22, "warn", "为什么这两个轴必须一起上 ——　算一遍就知道", [
        "假设 V4 <tspan font-weight=\"700\">不</tspan>以 MLA 那条窄 entry 为基础，"
        "就是标准多头（128 头 × 128 维，K 和 V 各一份）＝&#160;"
        "<tspan font-weight=\"700\">每 token 每层 32768 个数</tspan>。"
        "压 4:1 之后还剩 <tspan font-weight=\"700\">8192</tspan> ——&#160;"
        "而 MLA 是 <tspan font-weight=\"700\">576</tspan>。<tspan font-weight=\"700\">差十四倍。</tspan>",
        "⭐ 结论：<tspan font-weight=\"700\">序列压缩不是「可以叠在宽度压缩上」，是「必须叠」</tspan> ——&#160;"
        "光压条数、不压宽度，连上一代都打不过。",
    ])

    # ══════════ Ⓐc 为什么 2019 年不行的 MQA，现在又行了 ═════════
    # ⭐⭐⭐ 2026-09-16 现场追问追加。原话：「他们为什么把这个 MQA 又给捞回来？
    #   这个 MQA 一开始不是证明他的能力不太行吗？什么时候 MQA 又好上了？」
    # ⛔⛔ **论文没有正面回答这个问题** ——&#160;§2.3.3 / §2.3.4 都没有一段
    #   解释「为什么共享 KV 不掉点」。所以这一格是**推导**，标记清楚。
    # ⭐ 吸收恒等式本身是公开技巧（DeepSeek-V2 就有）；
    #   「V4 ＝ MLA 吸收形态的原生化」是我们的判断，不是论文原话。
    PHC = 430
    pyc = f.panel(0, yy + 26, W, PHC,
                  "Ⓐc 为什么 2019 年被判「能力不行」的 MQA，"
                  "<tspan font-weight=\"700\">现在又行了</tspan>", BL,
                  sub="⚠️ 论文没正面答这个问题 ——　"
                      "<tspan font-weight=\"700\">下面是推导，不是原话</tspan>")

    # ── 吸收恒等式
    f.box(60, pyc + 26, 1280, 108, "#f3e8fd", PU, 8)
    f.t(84, pyc + 56, "⭐⭐⭐ 先看一个恒等式 ——　"
                      "<tspan font-weight=\"700\">MLA 吸收之后，本来就是一个 head_dim ＝ 512 的 MQA</tspan>",
        INK, True, 17)
    f.t(84, pyc + 86, "MLA 里第 h 个头的分数 ＝ q_h ·（W_UK,h · c）"
                      "　＝　（W_UK,hᵀ · q_h）· c", GY, size=15, mono=True)
    f.t(84, pyc + 114, "→　<tspan font-weight=\"700\">一条共享的 512 维 key，"
                       "每个头拿自己那条 512 维 query 去点它</tspan>"
                       "　·　value 侧同理，每头视角可并进输出投影", GY, size=14)

    # ── 两边对照
    f.box(60, pyc + 152, 620, 236, "#fff", RD, 8)
    f.box(60, pyc + 152, 620, 4, RD, RD, 2)
    f.t(370, pyc + 186, "2019 的 MQA", RD, True, 19, "middle")
    for i, ln in enumerate((
            "共享的那条<tspan font-weight=\"700\">只有一个头宽</tspan>（典型 128）",
            "query 侧也还是 128 维",
            "⛔ 每头视角<tspan font-weight=\"700\">是真的被删掉了</tspan>",
            "⛔ 而且<tspan font-weight=\"700\">没有任何东西补偿它</tspan>",
            "→　所以掉点",
    )):
        f.t(92, pyc + 222 + i * 30, ln, GY, size=14.5)

    f.box(720, pyc + 152, 620, 236, "#fff", GR, 8)
    f.box(720, pyc + 152, 620, 4, GR, GR, 2)
    f.t(1030, pyc + 186, "V4 的 Shared K＝V MQA", GR, True, 19, "middle")
    for i, ln in enumerate((
            "共享的那条 <tspan font-weight=\"700\">512 宽 ——　宽四倍</tspan>",
            "query <tspan font-weight=\"700\">每头 512</tspan>（从 1536 latent 上投）",
            "⭐ 输出侧<tspan font-weight=\"700\">分组低秩投影</tspan>（16 组 × 1024）",
            "⭐ 每头还有一个可学的 attention sink",
            "→　<tspan font-weight=\"700\">每头视角没消失，它搬家了</tspan>",
    )):
        f.t(752, pyc + 222 + i * 30, ln, GY, size=14.5)
    f._pan = None

    yy = f.band(pyc + PHC + 22, "warn", "⛔ 但这不是「零损失」——　省一半不可能白省", [
        "<tspan font-weight=\"700\">K 和 V 合成同一条，这一步连 MLA 都没敢做</tspan> ——&#160;"
        "MLA 至少还有两个不同的升维矩阵把 K 和 V 区分开。"
        "<tspan font-weight=\"700\">原本两组独立的 512 自由度，现在只剩一组。</tspan>",
        "⭐ 这份损失被三样东西吃掉了："
        "<tspan font-weight=\"700\">① query 侧每头 512 的自由度</tspan>；"
        "<tspan font-weight=\"700\">② 输出侧的分组低秩投影</tspan>；"
        "<tspan font-weight=\"700\">③ 原生训练</tspan> ——&#160;"
        "它是从第一天就这么训的，不是从别的检查点改出来的。"
        "<tspan font-style=\"italic\">（③ 是类比 NSA 的 Native，论文没这么说。）</tspan>",
    ])

    # ══════════ Ⓑ 三种层，KV 序列各自长什么样 ═══════════════════
    PH2 = 486   # ⛔ 470 时层表黄底压住 HCA 行脚注，加高并把黄底下移
    py2 = f.panel(0, yy + 26, W, PH2,
                  "Ⓑ 三种层：<tspan font-weight=\"700\">纯滑窗</tspan>　·　"
                  "<tspan font-weight=\"700\">CSA（m＝4）</tspan>　·　"
                  "<tspan font-weight=\"700\">HCA（m′＝128）</tspan>", BL,
                  # ⛔ 这行 sub 改短过两次 ——&#160;探针两次都报它右端顶出面板
                  sub="⭐ 三种层都带 128 的未压缩滑窗，压缩块"
                      "<tspan font-weight=\"700\">拼在它后面</tspan>")

    SWX, SWW = 150, 250
    ROWS = (
        (0, "纯滑窗层　compress_ratio ＝ 0", GY2, None, None,
         "只有近处这 128 条　·　V4-Pro 61 层里只有最后 1 层是它"),
        (1, "CSA 层　m ＝ 4", OR, "压缩块　N/4 条",
         "Lightning Indexer 挑 top-k ＝ 1024 条",
         # ⭐⭐ 2026-09-16：同一处被现场问了**两次**「CSA 怎么也有滑窗」。
         #   ⛔ 两次都是同一个缺口：这一行只写了「有滑窗」，没写**为什么非有不可**。
         #   ⭐ 判据：**同一处被问第二次，就不要再解释了 ——&#160;去改那一行本身。**
         #     理由挪到它旁边，比写在下面的落点带里管用。
         "⛔ 索引器挑的是「块」——　最近那几个字还没成块，它挑不到"),
        (2, "HCA 层　m′ ＝ 128", GR, "压缩块　N/128 条",
         "⭐ 不挑 ——　全部压缩条都参与",
         "1M 上下文下也只有约 8 千条，全看得起"),
    )
    for i, (k, title, col, ctag, cnote, foot) in enumerate(ROWS):
        ry = py2 + 44 + i * 138
        f.t(72, ry + 6, title, col, True, 17)
        f.box(SWX, ry + 22, SWW, 46, "#fff", LINE, 6)
        f.box(SWX, ry + 22, SWW, 4, GY2, GY2, 2)
        f.t(SWX + SWW / 2.0, ry + 52, "未压缩滑窗　128", GY, True, 15, "middle")
        if ctag:
            f.t(SWX + SWW + 20, ry + 52, "＋", INK, True, 17)
            f.box(SWX + SWW + 46, ry + 22, 470, 46, "#fff", col, 6)
            f.box(SWX + SWW + 46, ry + 22, 470, 4, col, col, 2)
            f.t(SWX + SWW + 46 + 235, ry + 52, ctag, col, True, 16, "middle")
            f.t(SWX + SWW + 536, ry + 52, cnote, INK, size=14)
        else:
            f.t(SWX + SWW + 24, ry + 52, "（后面什么都不拼）", GY2, size=14)
        f.t(SWX, ry + 92, foot, GY, size=13.5)

    f.box(72, py2 + 424, 1256, 44, "#fef7e0", OR, 6)
    f.t(96, py2 + 452, "⭐ V4-Pro 的层表（61 层，直接抄自 config.json）："
                       "<tspan font-family=\"ui-monospace,monospace\">"
                       "[128, 128, 4, 128, 4, 128, …, 4, 128, 0]</tspan>"
                       "　——　<tspan font-weight=\"700\">开头两层 HCA 打底，中间 CSA 与 HCA 交替，"
                       "最后一层纯滑窗</tspan>", INK, size=15)
    f._pan = None

    # ⛔⛔⛔ 2026-09-16 现场：「这个 CSA 它居然也有一个 SWA 吗？好像不对吧，
    #   CSA 配的是 Lightning indexer，选的是 top 1024。」
    #   ⭐ **图没画错，错的是这条带子只给了 HCA 一个理由** ——&#160;
    #     于是读者看见 CSA 那行也挂滑窗，自然以为是画错了。
    #     ⛔ 判据：**当一个部件三处都有、而你只解释了其中一处，
    #       另外两处就会被读成 bug。理由要跟着部件走，不是跟着最极端的那个例子走。**
    #   📌 事实核过：论文 **Figure 3 画的就是 CSA**，它的图注原话是
    #     「Additionally, a small set of sliding window KV entries is combined
    #      with the selected compressed KV entries to enhance local
    #      fine-grained dependencies.」——&#160;滑窗就是在 CSA 这一节（§2.3.1）
    #     被引入的；HF 的架构文档也写着「All three types share the same
    #     backbone: … Shared sliding-window K=V branch」。
    yy = f.band(py2 + PH2 + 22, "info",
                "滑窗跟 indexer 不是二选一 ——　CSA 两样都有", [
        "⛔ <tspan font-weight=\"700\">共同的理由：压缩块要凑满 m 个 token 才成形</tspan>，"
        "而因果律不允许查询看见自己后面的 token ——&#160;"
        "<tspan font-weight=\"700\">在凑满之前，查询无块可看</tspan>。"
        "CSA 的 m ＝ 4，缺口小；HCA 的 m′ ＝ 128，缺口大到能把刚开口那几十个字整个吞掉。",
        "⭐ <tspan font-weight=\"700\">而 CSA 还有一条自己的理由</tspan>："
        "压缩块是好几个 token 的加权和，<tspan font-weight=\"700\">近处需要的是 token 级的分辨率</tspan>，"
        "摘要给不了 ——&#160;论文 Figure 3 的图注就写着这是"
        "「to enhance local fine-grained dependencies」。",
        "⭐⭐ 所以两件事分工很清楚："
        "<tspan font-weight=\"700\">indexer 管远处挑哪几条摘要，滑窗管近处一个不落。</tspan>"
        "⛔ 它不是「为了精度锦上添花」，是「不挂就接不上话」。",
    ])

    # ══════════ Ⓒ 压缩怎么压 ═════════════════════════════════════
    PH3 = 300   # ⛔ 336 时底下空 70px
    py3 = f.panel(0, yy + 26, W, PH3,
                  "Ⓒ 压缩这一步到底在算什么 ——　"
                  "<tspan font-weight=\"700\">加权求和，权重是学出来的</tspan>", BL,
                  sub="⛔ 不是平均池化，也不是把现成的 K/V 拿来池化 ——　"
                      "<tspan font-weight=\"700\">压缩器有自己独立的投影矩阵</tspan>")

    f.box(60, py3 + 26, 620, 240, "#fff", OR, 8)
    f.box(60, py3 + 26, 620, 4, OR, OR, 2)
    f.t(370, py3 + 60, "CSA：重叠窗口", OR, True, 19, "middle")
    for i, ln in enumerate((
            "两路投影 Cᵃ 和 Cᵇ，各配一路权重 Z",
            "每条摘要 ＝ <tspan font-weight=\"700\">2m ＝ 8 个 token</tspan> 的加权和",
            "权重在这 8 个上做一次 softmax（还带可学的位置偏置）",
            "相邻两条的来源<tspan font-weight=\"700\">互相重叠</tspan>",
            "→　序列真正压到 <tspan font-weight=\"700\">1/m ＝ 1/4</tspan>",
    )):
        f.t(92, py3 + 98 + i * 30, ln, GY, size=14.5)
    f.t(92, py3 + 250, "⭐ 看 8 个、只留 1 条 ——　边界不至于被切死", OR, True, 14.5)

    f.box(720, py3 + 26, 620, 240, "#fff", GR, 8)
    f.box(720, py3 + 26, 620, 4, GR, GR, 2)
    f.t(1030, py3 + 60, "HCA：非重叠窗口", GR, True, 19, "middle")
    for i, ln in enumerate((
            "同样是投影 ＋ 门控加权求和",
            "每条摘要 ＝ <tspan font-weight=\"700\">m′ ＝ 128 个 token</tspan> 的加权和",
            "窗口<tspan font-weight=\"700\">不重叠</tspan>，一刀一段",
            "压完之后<tspan font-weight=\"700\">再给这条摘要施加一次 RoPE</tspan>",
            "用的是这条摘要自己的<tspan font-weight=\"700\">锚点位置</tspan>",
    )):
        f.t(752, py3 + 98 + i * 30, ln, GY, size=14.5)
    f.t(752, py3 + 250, "⭐ 没有索引器 ——　压到这个份上，全看也看得起", GR, True, 14.5)
    f._pan = None

    # ══════════ Ⓓ 取用时恢复成什么 ═══════════════════════════════
    PH4 = 412   # ⛔ 396 时 ④ 那块红底说明顶出下沿 8px（探针抓到），加到 412
    py4 = f.panel(0, py3 + PH3 + 26, W, PH4,
                  "Ⓓ 查询来了之后 ——　<tspan font-weight=\"700\">"
                  "全程没有任何一步叫「解压缩」</tspan>", BL,
                  sub="⭐ 但有一步很容易被漏掉：<tspan font-weight=\"700\">"
                      "输出侧要按负位置反向旋转一次</tspan>")

    STEPS = (
        ("①", "查询", "128 个头，每头 512 维（走 q_lora 1536 下投再上投）"),
        ("②", "点积", "直接和那条 512 相乘 ——　<tspan font-weight=\"700\">不展开、不还原</tspan>"),
        ("③", "加权", "softmax 之后，乘的还是<tspan font-weight=\"700\">同一条 512</tspan>"),
        ("④", "反向 RoPE",
         "对输出的 rope 那 64 维，用位置 <tspan font-weight=\"700\">−i</tspan> 再转一次"),
        ("⑤", "输出投影",
         "分 16 组、每组降到 1024，再合回 hidden 7168"),
    )
    for i, (n, t1, t2) in enumerate(STEPS):
        sy = py4 + 34 + i * 58
        col = RD if i == 3 else GY2
        f.box(72, sy, 40, 40, "#fff", col, 20)
        f.t(92, sy + 27, n, col, True, 17, "middle")
        f.t(130, sy + 27, t1, INK, True, 16)
        f.t(288, sy + 27, t2, GY, size=15)

    f.box(72, py4 + 332, 1256, 52, "#fce8e6", RD, 6)
    f.t(96, py4 + 352, "⛔ <tspan font-weight=\"700\">④ 为什么非有不可</tspan>："
                       "K 和 V 是同一条，所以<tspan font-weight=\"700\">V 也被 RoPE 转过了</tspan>"
                       " ——　不转回来，V 携带的就是绝对位置的污染。", INK, size=15)
    f.t(96, py4 + 374, "⭐ 按 −i 转回来之后，"
                       "<tspan font-weight=\"700\">每条 KV 的贡献只跟它到查询的「相对距离」有关</tspan>"
                       "　·　论文 §2.3.3 式 26", GY, size=14)
    f._pan = None

    yy = f.band(py4 + PH4 + 22, "ok", "这一整套换来了什么", [
        "论文口径（1M 上下文，对照 DeepSeek-V3.2）："
        "<tspan font-weight=\"700\">V4-Pro ＝ 27% 的单 token 推理 FLOPs、10% 的 KV cache</tspan>；"
        "<tspan font-weight=\"700\">V4-Flash ＝ 10% 与 7%</tspan>。",
        "vLLM 的实现口径：1M 上下文、bf16 之下，"
        "<tspan font-weight=\"700\">V4 每条序列 9.62 GiB</tspan>，"
        "对照同为 61 层的 V3.2 式估算 83.9 GiB ——&#160;"
        "<tspan font-weight=\"700\">约 8.7 倍</tspan>；"
        "实跑再用 fp8 存注意力、fp4 存索引器，又能减掉大约一半。",
        "⛔ <tspan font-weight=\"700\">这两组都是别人的公开口径，本课没有 V4 的实测</tspan> ——&#160;"
        "而且论文自己也说了，这个对比是<tspan font-weight=\"700\">整模型级</tspan>的："
        "数据、优化器、残差通路、数值精度全都换了，"
        "<tspan font-weight=\"700\">不能算到 CSA / HCA 一家头上</tspan>。",
    ])

    yy = f.src(yy + 24,
               "论文：<tspan font-weight=\"700\">arXiv 2606.19348</tspan>"
               "《DeepSeek-V4: Towards Highly Efficient Million-Token Context "
               "Intelligence》(DeepSeek-AI, 2026-04-26)。"
               "§2.3.1 CSA：式 9–12 是压缩（<tspan font-weight=\"700\">每条摘要由 2m 个 entry "
               "加权求和而来，softmax 在这 2m 个上归一，相邻两条来源重叠，"
               "所以序列压到 1/m</tspan>），式 13–14 是 indexer 的低秩查询；"
               "§2.3.2 HCA；§2.3.3 Other Details 含 Partial RoPE、"
               "<tspan font-weight=\"700\">式 26 的反向旋转</tspan>、式 27 的 attention sink",
               "配置：Hugging Face <tspan font-weight=\"700\">deepseek-ai/DeepSeek-V4-Pro</tspan>"
               " 的 config.json ——　num_attention_heads 128、"
               "<tspan font-weight=\"700\">num_key_value_heads 1</tspan>、head_dim 512、"
               "qk_rope_head_dim 64、q_lora_rank 1536、o_groups 16、o_lora_rank 1024、"
               "sliding_window 128、index_topk 1024、"
               "num_hidden_layers 61、rope_theta 10000 与 compress_rope_theta 160000",
               "架构说明：Hugging Face transformers 文档 model_doc/deepseek_v4 ——　"
               "其中「<tspan font-weight=\"700\">Shared K=V Multi-Query Attention</tspan>」"
               "「Partial RoPE 落在每个头<tspan font-weight=\"700\">末尾的 qk_rope_head_dim 个通道</tspan>」"
               "「压缩器的输出与滑窗分支的 KV <tspan font-weight=\"700\">拼接</tspan>后再进核心注意力」",
               "实现口径：vLLM 博客《DeepSeek V4 in vLLM: Efficient Long-context "
               "Attention》(2026-04-24) ——　c4a ＝「8 个 token 的加权和，步长 4」，"
               "c128a ＝「128 个的加权和，步长 128」；1M 上下文 bf16 下 9.62 GiB，"
               "对照 61 层 V3.2 式估算 83.9 GiB",
               "⛔ <tspan font-weight=\"700\">本图只有两组数字，都是转述的公开口径</tspan>；"
               "mHC 与 MoE 部分刻意没画，它们不归这一讲管")
    f.save("fig3-v4-arch.svg", yy + 6)


main()
