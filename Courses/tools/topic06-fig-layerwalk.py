# -*- coding: utf-8 -*-
r"""专题六 · 第零节 0.2「出一个字，61 层逐层走一遍」的静态图。

fig6-layerwalk.svg —— 回答：decode 出一个字，每一层读什么、读多少、写什么、传什么、丢什么？
  · 数字：全部由 DeepSeek-V3 的 config.json 现算（下面逐项 assert），不是实测。
    config 取自 Hugging Face deepseek-ai/DeepSeek-V3 的 config.json（2026-10-17 读）：
      hidden_size 7168、num_hidden_layers 61、first_k_dense_replace 3、intermediate_size 18432、
      moe_intermediate_size 2048、n_routed_experts 256、n_shared_experts 1、num_experts_per_tok 8、
      num_attention_heads 128、q_lora_rank 1536、kv_lora_rank 512、qk_nope_head_dim 128、
      qk_rope_head_dim 64、v_head_dim 128、vocab_size 129280、tie_word_embeddings false。
  · 锚点：同一套公式算总参数 671.0B（对上技术报告 671B，不含 MTP 层）；
    61 层激活 35.7B ＋ 输出层 0.93B ＝ 36.6B，四舍五入是报告的 37B。对上了才画具体数。
  · 口径：权重按 FP8 每参数 1 字节（课件第一节同一口径，忽略 block scale；路由打分矩阵、
    norm 实际精度更高，量级可忽略）；KV 按 bf16；传给下一层的向量按 bf16。
  · 刻意没画：norm 参数（每层几万个，画不出来）；输入词表查表（只读一行，不是整块读）；
    MTP 层（推理时可选的草稿层，第十一节）。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

import topic06_numbers as N

W = 1400

# ── config.json（逐项见 docstring）────────────────────────────────────
H = N.V3_HIDDEN                       # 7168
assert H == 7168 and N.V3_LAYERS == 61
DENSE_LAYERS = 3                      # first_k_dense_replace：前 3 层是稠密前馈
MOE_LAYERS = N.V3_LAYERS - DENSE_LAYERS
Q_LORA, HEADS, NOPE, ROPE, VD = 1536, 128, 128, N.V3_ROPE, 128
KV_LORA = N.V3_KV_LORA
FFN_DENSE, FFN_EXPERT = 18432, 2048
VOCAB = 129280

# ── 每层参数（MLA 注意力）────────────────────────────────────────────
ATTN = (H * Q_LORA                        # q 降维
        + Q_LORA * HEADS * (NOPE + ROPE)  # q 升维到 128 头 × 192
        + H * (KV_LORA + ROPE)            # 压成 512 维 KV ＋ 64 维位置编码那截
        + KV_LORA * HEADS * (NOPE + VD)   # 从 512 维解出 128 头的 K、V
        + HEADS * VD * H)                 # 输出投影
assert ATTN == 187105280
DENSE = 3 * H * FFN_DENSE                 # gate、up、down 三块
EXPERT = 3 * H * FFN_EXPERT               # 一个专家
assert DENSE == 396361728 and EXPERT == 44040192
ROUTER = N.V3_EXPERTS * H                 # 路由打分矩阵
ACT_EXPERTS = N.V3_TOPK + 1               # 8 个路由专家 ＋ 1 个共享专家
MOE_ACT = ACT_EXPERTS * EXPERT + ROUTER
assert ACT_EXPERTS * EXPERT == DENSE      # 巧合：9 个专家正好等于稠密前馈那一块
LAYER_DENSE = ATTN + DENSE
LAYER_MOE = ATTN + MOE_ACT
LM_HEAD = VOCAB * H                       # 输出层（与输入词表不共享）
LAYERS_ACT = DENSE_LAYERS * LAYER_DENSE + MOE_LAYERS * LAYER_MOE
ACT = LAYERS_ACT + LM_HEAD
assert round(LAYERS_ACT / 1e9, 1) == 35.7 and round(LM_HEAD / 1e9, 2) == 0.93
assert round(ACT / 1e9, 1) == 36.6 and round(ACT / 1e9) == round(N.V3_ACTIVE / 1e9)   # 锚点：37B
TOTAL = (N.V3_LAYERS * ATTN + DENSE_LAYERS * DENSE
         + MOE_LAYERS * ((N.V3_EXPERTS + 1) * EXPERT + ROUTER) + 2 * LM_HEAD)
assert round(TOTAL / 1e9) == round(N.V3_TOTAL / 1e9) == 671                            # 锚点：671B
assert round(MOE_LAYERS * N.V3_EXPERTS * EXPERT / 1e9, 1) == 653.9                     # 与第一节同一个数

# ── 每层 KV、传下去的向量 ───────────────────────────────────────────
KV_TOK_LAYER = (KV_LORA + ROPE) * 2       # 576 个数 × bf16 2 字节
assert KV_TOK_LAYER == 1152 and KV_TOK_LAYER * N.V3_LAYERS == N.V3_KV_PER_TOKEN == 70272
CTX = [("1 万", 10000), ("128K", 131072)]
KV_LAYER = {k: KV_TOK_LAYER * L for k, L in CTX}
assert round(KV_LAYER["1 万"] / 1e6, 1) == 11.5 and round(KV_LAYER["128K"] / 1e6) == 151
VEC = H * 2
assert VEC == 14336

MB = 1e6


def mb(x):
    return "%d MB" % round(x / MB) if x >= 100 * MB else "%.1f MB" % (x / MB)


def fig_layerwalk():
    f = Fig(W, "DeepSeek-V3 出一个字的逐层示意。新 token 带一条 7,168 维向量从第 1 层进去。每一层读两样："
               "本层权重和本层全部 KV。第 1 到 3 层是稠密前馈，每层读注意力 187 MB 加稠密前馈 396 MB，共 583 MB；"
               "第 4 到 61 层是 MoE，每层读注意力 187 MB 加 8 个路由专家和 1 个共享专家 396 MB，再加路由 1.8 MB，共 585 MB。"
               "本层 KV 每个 token 是 512 维压缩 KV 加 64 维位置编码那截，共 576 个数，bf16 每个 2 字节，共 1,152 字节；"
               "上下文 1 万 token 时每层 11.5 MB，128K 时每层 151 MB，全部要读。"
               "每层写一样：KV 末尾追加 1,152 字节；传一样：一条 7,168 维向量交给下一层；其余中间结果丢掉。"
               "61 层走完再读输出层 0.93 GB，才出一个字。61 层权重合计 35.7 GB，加输出层 36.6 GB，就是第一节的 37 GB")
    y0 = f.header("出一个字，61 层逐层走一遍：<tspan font-weight=\"700\">每层读两样，写一格，传一条，其余丢掉</tspan>",
                  "DeepSeek-V3，权重按 FP8 每参数 1 字节，KV 和传下去的向量按 bf16；条长按字节数同一比例画，数字由 config.json 推导",
                  [(BL, "注意力权重"), (GR, "前馈／专家权重"), (PU, "本层 KV"), (OR, "写：KV 末尾 +1 格")])
    FULL, SLIM, OUTR = 76, 38, 70
    rows = [("第 1 层", "稠密前馈", "dense"), ("第 2、3 层", "同第 1 层", None),
            ("第 4 层", "MoE", "moe"), ("第 5 到 60 层", "同第 4 层", None),
            ("第 61 层", "MoE", "moe"), ("输出层", "61 层走完", "out")]
    HEAD = 64
    PH = HEAD + sum(FULL if r[2] in ("dense", "moe") else OUTR if r[2] == "out" else SLIM for r in rows) + 12
    py = f.panel(0, y0, W, PH, "只有一个人 decode：他的新 token 从上往下走一步", INK)

    X0, X1, X2, X3, X4 = 20, 190, 770, 1030, 1200      # 列：层名 ／ 读权重 ／ 读 KV ／ 写 ／ 传与丢
    SC = 560 / 1000e6                                   # 1,000 MB 画 560 px（权重和 KV 同一比例）
    hy = py + 26
    f.t(X1, hy, "读：本层权重", INK, True, 15)
    f.t(X1, hy + 22, "每步都整块读一遍", GY, size=13)
    f.t(X2, hy, "读：本层全部 KV", INK, True, 15)
    f.t(X2, hy + 22, "之前每个 token 一格", GY, size=13)
    f.t(X3, hy, "写：KV 末尾", INK, True, 15)
    f.t(X3, hy + 22, "576 个数 × 2 B", GY, size=13)
    f.t(X4, hy, "传：下一层", INK, True, 15)
    f.t(X4, hy + 22, "丢：其余中间结果", GY, size=13)
    f.line(14, py + HEAD - 6, W - 14, py + HEAD - 6, LINE, 1, arrow=False)

    y = py + HEAD
    BH = 26
    for name, sub, kind in rows:
        h = FULL if kind in ("dense", "moe") else OUTR if kind == "out" else SLIM
        if kind is None:
            f.t(X0, y + 24, name, GY, True, 14)
            f.t(X1, y + 24, sub + "：读、写、传、丢都一样", GY, size=14)
            f.t(X4 + 40, y + 24, "↓", GY, True, 16)
            y += h
            continue
        f.t(X0, y + 24, name, INK, True, 16)
        f.t(X0, y + 46, sub, GY, size=14)
        by = y + 6
        if kind == "out":
            wl = LM_HEAD * SC
            f.box(X1, by, wl, BH, GY2, GY2, 3)
            f.t(X1 + 10, by + 18, "输出层 %.2f GB：7,168 维 × 词表 129,280" % (LM_HEAD / 1e9), "#ffffff", True, 13)
            f.t(X1, by + 48, "打分挑出下一个字，然后回到第 1 层", INK, size=14)
            f.t(X4, by + 18, "出一个字", RD, True, 16)
            y += h
            continue
        ffn, ffn_lab = (DENSE, "稠密前馈 %s" % mb(DENSE)) if kind == "dense" else \
                       (ACT_EXPERTS * EXPERT, "8＋1 个专家 %s" % mb(ACT_EXPERTS * EXPERT))
        wa, wf = ATTN * SC, ffn * SC
        f.box(X1, by, wa, BH, BL, BL, 3)
        f.t(X1 + wa / 2, by + 18, "注意力 %s" % mb(ATTN), "#ffffff", True, 13, "middle")
        f.box(X1 + wa + 2, by, wf, BH, GR, GR, 3)
        f.t(X1 + wa + 2 + wf / 2, by + 18, ffn_lab, "#ffffff", True, 13, "middle")
        tot = LAYER_DENSE if kind == "dense" else LAYER_MOE
        f.t(X1 + wa + wf + 12, by + 18, "%s" % mb(tot), INK, True, 15)
        f.t(X1, by + 48, "共 %s" % mb(tot) + ("（含路由打分 %s）" % mb(ROUTER) if kind == "moe" else ""), GY, size=13)
        # KV：128K 那根条 ＋ 1 万的刻度
        k128, k1w = KV_LAYER["128K"] * SC, KV_LAYER["1 万"] * SC
        f.box(X2, by, k128, BH, PU, PU, 3)
        f.t(X2 + k128 + 8, by + 18, "128K：%s" % mb(KV_LAYER["128K"]), PU, True, 14)
        f.box(X2, by + BH + 6, max(k1w, 3), 8, PU, PU, 1)
        f.t(X2 + 14, by + BH + 15, "1 万：%s" % mb(KV_LAYER["1 万"]), PU, size=13)
        # 写
        f.box(X3, by, 110, BH, "none", OR, 4)
        f.t(X3 + 55, by + 18, "＋1,152 B", OR, True, 14, "middle")
        # 传与丢
        f.t(X4, by + 18, "7,168 维 ↓", INK, True, 15)
        f.t(X4, by + 40, "（bf16 14,336 B）", GY, size=13)
        f.t(X4 + 140, by + 18, "丢", GY, True, 14)
        y += h
    f._pan = None
    yb = f.band(py + PH + 16, "ok", "61 层走完才出一个字，下一个字从第 1 层重来", [
        "权重：61 层合计 %.1f GB，加输出层 %.1f GB，就是第一节那 37 GB；人多时这一趟大家共用。" % (LAYERS_ACT / 1e9, ACT / 1e9),
        "KV：每层每 token 是 512 维压缩 KV（K、V 都从它解出）＋ 64 维位置编码那截 ＝ 576 个数，bf16 每个 2 B ＝ 1,152 B；跟长度成正比，一人一份。",
    ])
    f.src(yb + 10,
          "📌 形状取 Hugging Face deepseek-ai/DeepSeek-V3 的 config.json：hidden 7,168、61 层、first_k_dense_replace 3、"
          "intermediate 18,432、moe_intermediate 2,048、256 个路由专家选 8 ＋ 1 个共享专家、128 头、q_lora 1,536、kv_lora 512、"
          "qk_nope 128、qk_rope 64、v_head 128、词表 129,280、输入输出词表不共享。",
          "📌 本课推导：注意力 ＝ 7168×1536 ＋ 1536×128×192 ＋ 7168×576 ＋ 512×128×256 ＋ 128×128×7168 ＝ 187,105,280；"
          "稠密前馈 ＝ 3×7168×18432 ＝ 396,361,728；单个专家 ＝ 3×7168×2048 ＝ 44,040,192，9 个正好 396,361,728；路由 256×7168。",
          "📌 锚点：同一套公式总参数 671.0B（技术报告 671B，不含 MTP 层）；激活 ＝ 3×583.5M ＋ 58×585.3M ＋ 输出层 0.93B ＝ 36.6B，"
          "取整即报告的 37B（输入词表查表只读一行，不计）。",
          "⚠️ 口径：权重全按 FP8 每参数 1 字节，忽略 block scale；路由打分和 norm 实际精度更高，量级可忽略。KV、传下去的向量按 bf16。"
          "丢掉的中间结果包括本层的 Q、注意力分数、前馈中间那 18,432 维（专家是 2,048 维）的结果。")
    f.save("fig6-layerwalk.svg", yb + 14)


if __name__ == "__main__":
    fig_layerwalk()
