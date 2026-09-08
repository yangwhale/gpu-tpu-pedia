# -*- coding: utf-8 -*-
"""专题三 · 模型编年史的**数据源**——&nbsp;两个产物共用这一份。

⭐⭐ **为什么单独抽成一个模块。** 2026-09-07 现场要求「表头你点哪个就按哪个排序」，
   而原先整张表画在 SVG 里 ——&nbsp;**SVG 的文字是死的：点不了、搜不了、复制不了**。
   39 行的参照表本来就不该是图。于是拆成两个产物：

     · `topic03-fig-chronicle.py`   →&nbsp;上半时间轴，**那是真·图**，留 SVG
     · `topic03-table-models.py`    →&nbsp;下半模型表，**那是真·表**，出 HTML

   ⛔ 两边都要读同一份 ROWS。**所以数据必须只有一份** ——&nbsp;抄成两份的话，
     改了一边忘了另一边，两个产物会静默地漂开，而且谁都不报错。
     （这跟专题二 §0–§2 那批图当初「页面里一份、脚本里一份」是同一类事故。）

════════════════════════════════════════════════════════════════════
⭐ 2026-09-08 · 蚂蚁百灵那一家的命名，核清楚了（现场问「是不是还有个大号」）
════════════════════════════════════════════════════════════════════

**问的是对的，但大号在上一代。** 逐个查 HF `inclusionAI` 仓库建立时间：

  · **Ling-3.0** 到今天为止**只有两个尺寸**：
      `Ling-3.0-flash` 124B/5.1B（2026-08-02）、`Ling-3.0-tiny` 7.9B/1.3B（08-10）
    ⛔ **没有 1T 大号，`Ring-3.0` 一个仓库都没有。**
  · **1T 大号是 2.6 那一代**：
      `Ling-2.6-1T`（2026-04-29）＋ `Ring-2.6-1T`（2026-05-14）

📌 **这一家的命名规律**（记住能省很多事）：
   **`Ling` ＝ 通用／非思考线，`Ring` ＝ 思考模型线**；两条线同代同架构、成对发布。
   实测把两份 config 摆一起：`Ling-2.6-1T` 和 `Ring-2.6-1T`
   **80 层、layer_group_size 8、kv_lora_rank 512、64 头、hidden 8192 逐字段相同**，
   ⭐ 唯一的差别是 `max_position_embeddings`：**Ling 262144（256K）／Ring 131072（128K）**。
   → 所以本表只收 Ling 那一行，Ring 写进备注 ——&nbsp;**同架构不重复占行**。

⭐⭐ **顺手挖到一条本来会漏掉的家族内规律**：同一代里，
   **`tiny` 是 3:1（24 层 ÷ group 4），`flash` 是 5:1（42 层 ÷ group 6）**。
   **模型越大，越敢多掺线性层。** 这条只有把同代不同尺寸摆一起才看得见。

⛔ 还改对了一个日期：`Ling 2.6-1T` 原先记成 2026-06 ——&nbsp;那是 `-base` 仓库的日期。
   ⭐ **同一个模型在 HF 上有好几个仓库**（正式 / base / midtrain / fp8 / int4），
     建立时间各不相同。**抓最早那个正式版，别抓手边先搜到的那个。**

📌 **本表「时间」这一列的口径**：取**首次公开**——&nbsp;
   有官方公告就用公告日，没有就用 HF 上**正式版仓库**的建立日。
   ⚠️ 两者通常差几天到一周（Ling-3.0-flash：公告 07-23，HF 08-02），
   **本表只精确到月，这个差异不影响任何一行的月份**，但报日期时要说清是哪个口径。

📌 改数据改这里。画法改各自那个脚本。
"""
import math

# 供画图脚本沿用的基础色
BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
PU, CY, BR = "#8430ce", "#00838f", "#7a5000"


def wpx(s, size=11.5):
    """粗估纯文本像素宽——&nbsp;CJK 约一个字号宽，ASCII 约一半多。"""
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


# ── 一个类型一个颜色。同族相近色相，异族拉开 ──────────────────────────
AMB, ORG, DKR = "#f9ab00", "#e8710a", "#a50e0e"
TYPE_COL = {
    # 全注意力一族 —— 黄／橙
    "MHA": AMB, "MQA": AMB, "GQA": AMB, "FULL": AMB, "gAT": AMB,
    "MLA": ORG, "gMLA": ORG,
    # 线性一族 —— 冷色
    "KDA": BL, "GDN": "#12b5cb", "LTN": PU,
    # ⭐ Mamba／RWKV 也归线性一族（冷色）——&nbsp;它们不是"另一支"，
    #   Gated DeltaNet 那篇论文的标题就叫《Improving Mamba2 with Delta Rule》。
    "Mamba": "#3949ab", "RWKV": "#00695c",
    # 窗口 —— 青
    "SWA": CY,
    # 稀疏一族 —— 红
    "DSA": RD, "gDSA": RD, "MSA": RD, "CSA": RD, "HCA": DKR,
}
# ── 同一家用同一个底色 ────────────────────────────────────────────
# ⭐ 现场要求：「同属于一家的话，应该给它标成一样的背景颜色，这样好区分。」
#   ⭐⭐ 这条的价值在于：表是**按时间排**的，同一家的行天然被打散在各处。
#     底色一上，「某某家走了什么路」这条线不用眼睛去找，它自己浮出来。
# ⛔ 底色必须**很淡**——它是分组线索，不是内容。抢了格子的颜色就本末倒置了。
#   （按名字前缀匹配，第一个命中为准。加新行时若厂商没命中，会落到中性灰。）
VENDOR = [
    ("GPT-3",    "OpenAI",   "#5f6368", "#f1f3f4"),
    ("PaLM",     "Google",   "#1e8e3e", "#e6f4ea"),
    ("Llama",    "Meta",     "#546e7a", "#eceff1"),
    ("DeepSeek", "DeepSeek", "#00838f", "#e0f2f1"),
    ("MiniMax",  "MiniMax",  "#c2185b", "#fce4ec"),
    ("Qwen",     "阿里 千问", "#e8710a", "#fff3e0"),
    ("Kimi",     "月之暗面",  "#5e35b1", "#ede7f6"),
    ("小米",      "小米",     "#f9ab00", "#f9fbe7"),
    ("GLM",      "智谱",     "#8430ce", "#f3e5f5"),
    ("Ling",     "蚂蚁 百灵", "#0288d1", "#e1f5fe"),
    ("混元",      "腾讯 混元", "#d84315", "#fbe9e7"),
    # ⛔ 2026-09-07 补：原先 2024 之后几乎全是中国模型，这是一份对外教材里
    #   很明显的偏斜。Gemma／gpt-oss／Llama 4／Jamba／Mistral 全都补进来了。
    ("Gemma",    "Google",   "#1e8e3e", "#e6f4ea"),
    ("gpt-oss",  "OpenAI",   "#5f6368", "#f1f3f4"),
    ("Mistral",  "Mistral",  "#c2410c", "#fff1e6"),
    ("Jamba",    "AI21",     "#7b1fa2", "#f6e9fb"),
    ("RWKV",     "RWKV",     "#00695c", "#e0f2f1"),
]


def vendor_of(name):
    for key, label, ink, bg in VENDOR:
        if key in name:
            return label, ink, bg
    return "", GY, "#fff"


SPARSE = {"DSA", "gDSA", "MSA", "CSA", "HCA"}
LINEAR = {"KDA", "GDN", "LTN", "Mamba", "RWKV"}
# 简写 → 全名（写在「这一层 ＋ 那一层」那一列）
FULLNAME = {
    "MHA": "MHA", "MQA": "MQA", "GQA": "GQA", "FULL": "全注意力",
    "gAT": "Gated Attention", "MLA": "MLA", "gMLA": "Gated MLA",
    "KDA": "KDA", "GDN": "Gated DeltaNet", "LTN": "Lightning",
    "Mamba": "Mamba", "RWKV": "RWKV",
    "SWA": "SWA", "DSA": "DSA", "gDSA": "Gated DSA", "MSA": "MSA",
    "CSA": "CSA", "HCA": "HCA",
}

# (时间, 模型, 一个循环的构成 [(简写, 几层)…], 备注)
# ⛔ 只有一项 ＝ 每层同构，画成整条一色。
# ══════════════════════════════════════════════════════════════════
# KV cache：**算出来的，不是抄来的。** 公式写在这儿，参数来自各家 config，
# 谁都可以拿去复算。⛔ 别把结果硬编码成常数 —— 那样改了层数它不会跟着动。
#
#   普通 MHA/GQA/MQA ：2（K 和 V 两份）× 层数 × 长度 × KV头数 × 头维 × 2 B
#   MLA 一族         ：       层数 × 长度 × (kv_lora_rank ＋ rope 维) × 2 B
#                       ——&nbsp;只存那个压缩过的潜向量，**所有头共用一份**
#   线性层           ：**一个字节都不占**（状态是固定大小，跟长度无关），
#                       所以「层数」这一项只数**有 KV 的那些层**
#   DeepSeek-V4      ：特判。它 shared K=V（只存一份不是两份），
#                       而且 CSA/HCA 存的是**压缩池**：长度 ÷ 压缩率
#
# ⚠️ 口径：长度取 128K、BF16、batch=1、不含任何量化。
#    这几条一改，数就全变 —— 报 KV cache 的时候必须连口径一起报。
SEQ, BPE = 131072, 2


def kv_gib(spec):
    """按 spec 算 128K 时的 KV cache（GiB）。spec 为 None 表示没核到 config。"""
    if spec is None:
        return None
    kind = spec[0]
    if kind == "gqa":
        _, L, H, D = spec
        return 2 * L * SEQ * H * D * BPE / 2 ** 30
    if kind == "mla":
        _, L, R = spec                 # L 只数带 KV 的层，线性层不算
        return L * SEQ * R * BPE / 2 ** 30
    if kind == "none":
        return 0.0                     # 纯 RNN／SSM：只有固定大小的状态，没有 KV cache
    if kind == "swahyb":
        # 小米那一族：n_full 层全注意力 ＋ n_swa 层滑窗（窗口 win，只存 win 个 token）
        # ⚠️ 它的 K 和 V 维度**不一样**（QK 192 / V 128），所以这里是 qk+v 相加，
        #    不是像 GQA 那样乘 2。⛔ 照抄 GQA 的 ×2 会算错。
        _, n_full, n_swa, kvh_f, kvh_s, qk, v, win = spec
        ent = n_full * SEQ * kvh_f + n_swa * min(SEQ, win) * kvh_s
        return ent * (qk + v) * BPE / 2 ** 30
    if kind == "swahyb2":
        # ⭐ 跟 swahyb 的区别：**滑窗层和全局层的 KV 头数、头维都不一样**。
        #   Gemma 4 就是这样：滑窗层 16 头 × 256 维，全局层 4 头 × 512 维。
        #   ⛔ 拿 swahyb 硬套会算错 —— 那个式子假设两种层同构。
        _, n_full, n_swa, kvh_f, d_f, kvh_s, d_s, win = spec
        ent = (n_full * SEQ * kvh_f * d_f
               + n_swa * min(SEQ, win) * kvh_s * d_s)
        return 2 * ent * BPE / 2 ** 30          # K 和 V 各一份
    if kind == "v4":
        _, n_csa, n_hca, n_swa, D, m_csa, m_hca, win = spec
        # shared K=V → 每个条目只存一份；CSA/HCA 存压缩池；每层另挂一条滑窗支路
        ent = (n_csa * (SEQ // m_csa) + n_hca * (SEQ // m_hca)
               + (n_csa + n_hca + n_swa) * win)
        return ent * D * BPE / 2 ** 30
    raise ValueError(kind)


def kv_fmt(g):
    if g is None:
        return "—"
    if g >= 100:
        return "%d GiB" % round(g)
    if g >= 10:
        return "%.0f GiB" % g
    if g >= 1:
        return "%.1f GiB" % g
    if g <= 0:
        return "0（无 KV）"
    return "%d MiB" % round(g * 1024)


# 占得越多颜色越深 —— 一眼扫下来就是一条从深到浅的坡
def kv_col(g):
    if g is None:
        return "#e8eaed"
    for lim, c in ((100, "#7f0000"), (30, "#b31412"), (10, "#d93025"),
                   (3, "#e8710a"), (1, "#f9ab00")):
        if g >= lim:
            return c
    return "#1e8e3e"                   # 不到 1 GiB —— 已经是另一个量级了


ROWS = [
    # (时间, 模型, 循环构成, 上下文, KV 规格, 备注)
    # ⛔ 2026-09-07：备注全部压成**一行一句**。现场原话：「最后一列的 comments
    #   为什么折叠成好几行？这个造成我的行宽都太宽了，信息不够密集……
    #   最好就保持一行是一行。」
    # ⭐ 压缩的判据不是「把字砍短」，是**别复述别的列已经说过的**：
    #   层数在模型框里、配比在格子里、KV 大小在 KV 列里。
    #   备注只留「这一行独有的那句话」—— 删完之后 30 条超宽变成 0 条。
    # 📌 完整的推导链、出处、口径全在本文件顶部的文档字符串里，一个字没丢。
    ("2020-05", "GPT-3　175B 稠密 · 96 层", [("MHA", 1)], "2K", ("gqa", 96, 96, 128),
     "<tspan font-weight=\"700\">基线：KV 按头数线性长，没有任何省法</tspan>"),
    ("2022-04", "PaLM　540B 稠密 · 118 层", [("MQA", 1)], "2K", ("gqa", 118, 1, 256),
     "48 头共用 1 组 KV ——&#160;<tspan font-weight=\"700\">第一次大规模砍 KV</tspan>"),
    ("2023-02", "Llama 1　65B 稠密 · 80 层", [("MHA", 1)], "2K", ("gqa", 80, 64, 128),
     "一代还是纯 MHA，<tspan font-weight=\"700\">下一代才上 GQA</tspan>"),
    ("2023-07", "Llama 2　70B 稠密 · 80 层", [("GQA", 1)], "4K", ("gqa", 80, 8, 128),
     "MQA 砍太狠掉质量，GQA 是折中（arXiv 2305.13245）"),
    ("2023-09", "Mistral 7B　7B 稠密 · 32 层", [("SWA", 1)], "32K", ("swahyb", 0, 32, 0, 8, 128, 128, 4096),
     "⭐ <tspan font-weight=\"700\">SWA 进主流的第一枪</tspan>，窗口 4096"),
    ("2024-03", "Jamba　52B/12B · 32 层", [("Mamba", 7), ("GQA", 1)], "256K", ("gqa", 4, 8, 128),
     "⭐⭐ <tspan font-weight=\"700\">层间混合的开源起点</tspan>，比 MiniMax-01 早十个月"),
    ("2024-05", "DeepSeek-V2　236B/21B · 60 层", [("MLA", 1)], "128K", ("mla", 60, 576),
     "低秩压缩。<tspan font-weight=\"700\">KV 降 93.3%</tspan>＝只剩 2.25 组 GQA"),
    ("2024-06", "Gemma 2 27B　27B 稠密 · 46 层", [("SWA", 1), ("FULL", 1)], "8K", ("swahyb", 23, 23, 16, 16, 128, 128, 4096),
     "⭐ 谷歌开始交替：<tspan font-weight=\"700\">1:1，窗口 4096</tspan>"),
    ("2024-07", "Llama 3.1　405B 稠密 · 126 层", [("GQA", 1)], "128K", ("gqa", 126, 8, 128),
     "⛔ <tspan font-weight=\"700\">不上花招硬推 128K 的代价</tspan>：比 Llama 2 还多"),
    ("2024-11", "混元 Hunyuan-Large　389B/52B · 64 层", [("GQA", 1)], "128K", ("gqa", 32, 8, 80),
     "⭐ <tspan font-weight=\"700\">CLA：每 2 层共享一份 KV</tspan> ——&#160;旋钮①的第三招"),
    ("2024-12", "DeepSeek-V3　671B/37B · 61 层", [("MLA", 1)], "160K", ("mla", 61, 576),
     "⭐ <tspan font-weight=\"700\">专题一的锚点</tspan>。跟 V2 只差 1 层，KV 几乎相同"),
    ("2025-01", "MiniMax-01　456B/45.9B · 80 层", [("LTN", 7), ("GQA", 1)], "4M", ("gqa", 10, 8, 128),
     "⭐ 线性首次上旗舰。<tspan font-weight=\"700\">训练 1M、外推 4M</tspan>（config 的 10M 是容量）"),
    ("2025-03", "Gemma 3 27B　27B 稠密 · 62 层", [("SWA", 5), ("FULL", 1)], "128K", ("swahyb", 10, 52, 16, 16, 128, 128, 1024),
     "⭐⭐ <tspan font-weight=\"700\">5:1、窗口 1024</tspan> ——&#160;小米那个 5:1 不是首创"),
    ("2025-03", "RWKV-7 Goose　0.19B–2.9B · 纯 RNN", [("RWKV", 1)], "无限（理论）", ("none",),
     "⭐ <tspan font-weight=\"700\">全表唯一 KV 为零</tspan>：常数内存、常数单 token 时间"),
    ("2025-04", "Llama 4 Scout　109B/17B · 48 层", [("SWA", 3), ("FULL", 1)], "10M", ("swahyb", 12, 36, 8, 8, 128, 128, 8192),
     "块状局部 8192 ＋ NoPE 全局。⚠️ <tspan font-weight=\"700\">声称 10M，又一个报容量的</tspan>"),
    ("2025-04", "Qwen3-235B-A22B　235B/22B · 94 层", [("GQA", 1)], "40K", ("gqa", 94, 4, 128),
     "⭐ <tspan font-weight=\"700\">千问转线性之前的那一代</tspan>：纯 GQA-4"),
    ("2025-07", "Kimi K2　1T/32B · 61 层", [("MLA", 1)], "128K", ("mla", 61, 576),
     "⭐ <tspan font-weight=\"700\">Kimi 上 KDA 之前</tspan>：纯 MLA，架构名就是 DeepseekV3"),
    ("2025-07", "GLM-4.5　355B/32B · 92 层", [("GQA", 1)], "128K", ("gqa", 92, 8, 128),
     "⭐⭐ <tspan font-weight=\"700\">智谱上 DSA 之前</tspan>：46 GiB → GLM-5 的 11，降 4 倍"),
    ("2025-08", "gpt-oss-120b　117B/5.1B · 36 层", [("SWA", 1), ("FULL", 1)], "128K", ("swahyb", 18, 18, 8, 8, 64, 64, 128),
     "⭐ <tspan font-weight=\"700\">OpenAI 首个开放权重</tspan>：1:1 交替、窗口 128 ＋ sink"),
    ("2025-09", "DeepSeek-V3.2-Exp　671B/37B · 61 层", [("DSA", 1)], "160K", ("mla", 61, 576),
     "⭐ <tspan font-weight=\"700\">稀疏的起点</tspan>：V3 ＋ Lightning Indexer。⛔ KV 跟 V3 一样"),
    ("2025-09", "Qwen3-Next　80B/3B · 48 层", [("GDN", 3), ("gAT", 1)], "256K", ("gqa", 12, 2, 256),
     "36 线性 ＋ 12 全注意力（GQA-2，头维 256）"),
    ("2025-10", "Ling-1T（Ling 2.0）　1T/50B · 80 层", [("GQA", 1)], "32K", ("gqa", 80, 8, 128),
     "⭐⭐ <tspan font-weight=\"700\">Ling 2.6 就是从它改造的</tspan>：40 GiB → 1.4 GiB"),
    ("2025-10", "MiniMax M2　230B/10B · 62 层", [("GQA", 1)], "192K", ("gqa", 62, 8, 128),
     "⛔ <tspan font-weight=\"700\">「退回全注意力」≠ 什么都没做</tspan>：GQA-8 ＋ partial RoPE"),
    ("2025-10", "Kimi Linear　48B/3B · 27 层", [("KDA", 3), ("MLA", 1)], "1M", ("mla", 7, 576),
     "20 KDA ＋ 7 MLA（<tspan font-weight=\"700\">末层强制 full</tspan>）。已用 NoPE"),
    ("2026-01", "小米 MiMo-V2-Flash　309B/15B · 48 层", [("SWA", 5), ("FULL", 1)], "256K", ("swahyb", 8, 40, 8, 4, 192, 128, 128),
     "窗口 128。卡上自称 <tspan font-weight=\"700\">KV 省近 6×</tspan>，48÷8 正好对上"),
    ("2026-02", "GLM-5　744B/40B · 78 层", [("DSA", 1)], "198K", ("mla", 78, 576),
     "MLA ＋ DSA。<tspan font-weight=\"700\">GLM-5.1 同架构</tspan>，只有后训练不同"),
    ("2026-03", "Qwen3.5　397B/17B · 60 层", [("GDN", 3), ("gAT", 1)], "256K", ("gqa", 15, 2, 256),
     "45 线性 ＋ 15 全（config: full_attention_interval 4）"),
    ("2026-04", "小米 MiMo-V2.5-Pro　1.02T/42B · 70 层", [("SWA", 6), ("FULL", 1)], "1M", ("swahyb", 10, 60, 8, 8, 192, 128, 128),
     "60 SWA ＋ 10 全，窗口 128 ——&#160;<tspan font-weight=\"700\">1M 那档最省的</tspan>"),
    ("2026-04", "DeepSeek-V4-Pro　1.6T/49B · 61 层", [("HCA", 2), ("CSA", 1), ("HCA", 1), ("CSA", 1)], "1M", ("v4", 30, 31, 0, 512, 4, 128, 128),
     "⛔ 跟 Flash 不同：<tspan font-weight=\"700\">前两层是 HCA</tspan>。1.6T 而 KV 不到 1 GiB"),
    ("2026-05", "DeepSeek-V4-Flash　284B/13B · 43 层", [("SWA", 2), ("CSA", 1), ("HCA", 1), ("CSA", 1), ("HCA", 1)], "1M", ("v4", 21, 20, 2, 512, 4, 128, 128),
     "⭐ 2 层 SWA 引导，CSA／HCA 交替。<tspan font-weight=\"700\">MLA 换成 shared-KV MQA</tspan>"),
    ("2026-06", "GLM-5.2　744B/40B · 78 层", [("DSA", 1)], "1M", ("mla", 78, 576),
     "⭐ ＋IndexShare：四层共用一个 indexer。<tspan font-weight=\"700\">198K → 1M 靠这步</tspan>"),
    # ⛔ 2026-09-08 改日期：原写 2026-06，那是 `Ling-2.6-1T-base` 的仓库日期。
    #   正式版 `inclusionAI/Ling-2.6-1T` 建于 **2026-04-29**（flash 早一天）。
    #   ⭐ 教训：**同一个模型在 HF 上有好几个仓库（正式 / base / midtrain / 量化版），
    #     日期各不相同。** 抓「最早那个正式版」，别抓手边先搜到的那个。
    ("2026-04", "Ling 2.6-1T　1T/63B · 80 层", [("LTN", 7), ("MLA", 1)], "256K", ("mla", 10, 576),
     "⛔ <tspan font-weight=\"700\">不是 KDA</tspan>；思考版 Ring-2.6-1T 架构逐字段相同"),
    ("2026-06", "MiniMax M3　428B/23B · 60 层", [("MSA", 1)], "1M", ("gqa", 60, 4, 128),
     "⭐⭐ GQA-4 ＋ 稀疏。<tspan font-weight=\"700\">KV 比走 MLA 的 GLM-5.2 还大</tspan>"),
    ("2026-07", "Kimi K3　2.8T/104B · 93 层", [("KDA", 3), ("gMLA", 1)], "1M", ("mla", 24, 576),
     "69 KDA ＋ 24 Gated MLA（<tspan font-weight=\"700\">末层 92、93 连着两层 full</tspan>）"),
    ("2026-07", "混元 Hy3　295B/21B · 80 层", [("GQA", 1)], "256K", ("gqa", 80, 8, 128),
     "⛔ <tspan font-weight=\"700\">80 层全 GQA-8</tspan> ——&#160;线性一层都没上"),
    ("2026-07", "Ling-3.0-flash　124B/5.1B · 42 层", [("KDA", 5), ("gMLA", 1)], "256K", ("mla", 7, 576),
     "跟 2.6 换了一支。<tspan font-weight=\"700\">同代 tiny 用 3:1，它用 5:1</tspan>；3.0 无 1T"),
    ("2026-08", "GLM-5.3　744B/40B · 78 层", [("DSA", 1)], "1M", ("mla", 78, 576),
     "⚠️ <tspan font-weight=\"700\">跟 5.2 同一个 base</tspan>，纯后训练，架构没动"),
    ("2026-08", "混元 Hy4-preview　770B/49B · 78 层", [("gDSA", 1)], "1M", ("mla", 78, 576),
     "78 层全稀疏 ＋ <tspan font-weight=\"700\">IndexCache</tspan>（每 4 层 1 层算索引）"),
    ("2026-08", "⭐ GLM-5.3-Flash　320B/18B · 45 层", [("KDA", 3), ("DSA", 1)], "1M", ("mla", 11, 576),
     "34 KDA ＋ 11 稀疏 MLA ——&#160;<tspan font-weight=\"700\">GLM 首次线性＋稀疏同锅</tspan>"),
]

# ⛔ 2026-09-07：一次补了 15 行，按锚点插入之后**日期顺序乱了**
#   （2025-09 排到了 2025-07 前面）。
# ⭐ 正确的修法不是去调锚点，是**让脚本自己排** —— 顺序是从数据推得出来的东西，
#   就不该靠人手维护。这样以后新行插在哪儿都无所谓。
# 📌 日期是 "YYYY-MM" 定长字符串，字典序即时间序；同月的按写入顺序（sort 稳定）。

# ══════════════════════════════════════════════════════════════════
# ⛔⛔ 2026-09-07 第八轮：现场发现 **2025-10 → 2026-05 空了七个月**。
#     原话：「他怎么能空那么久呢？是不是有一些重要的事情你给丢掉了？」
#     ——&nbsp;**确实丢了。** 那七个月是开源架构最密的一段。
# ⭐ 形状：**Highlight 视图把稀疏暴露出来了。** 全量 39 行里那段看着只是「少几行」，
#   一旦筛成机制主线，七个月的空白立刻刺眼 ——&nbsp;
#   **筛选不只是省地方，它还是一种体检。**
# 📌 下面五行是这一轮补的，全部现读 config 或一手技术报告。
#    ⚠️ 同一轮还确认了**没补进来**的（记在这儿，别以为漏了）：
#      Kimi K2.5（2026-01，MLA，机制与 K2 同）、Qwen3-Coder-Next（2026-02，
#      GDN 混合，机制与 Qwen3-Next 同）、Step 3.5 Flash（2026-02）、
#      Ling 2.5 1T（2026-02，Lightning＋MLA，与本表 Ling 2.6 同机制）、
#      Sarvam 30B/105B（2026-03，GQA／MLA）、Nanbeige 4.1、Cohere Tiny Aya、
#      Olmo 3、Laguna XS.2（逐层注意力预算，机制有意思但没拿到 config）。
#      ⛔ 没补的理由是**机制重复或没核到 config**，不是「不重要」。
# ⛔⛔ 2026-09-07 再删两行：**Arcee Trinity Large** 和 **Zyphra ZAYA1-8B**。
#     现场原话：「ZAYA1 还有 Trinity 这种小众的都不要了。」
# ⭐ 判据（值得当成入表标准）：**这张表是参照物，而参照物的前提是读者见过它。**
#   一个只在架构上有趣、但没人真在用的模型放进来只会稀释这张表 ——&nbsp;
#   读者扫过去认不出，就会开始怀疑整张表的选材。
#   ⛔ 所以：**架构新颖 ≠ 该进表。** 新机制想讲就讲在时间轴上（那是机制图），
#     别为了讲一个机制硬塞一行没人用的模型。
# 📌 因此 CCA（压缩卷积注意力）也从时间轴撤了 ——&nbsp;它在本课唯一的载体就是
#   ZAYA1，模型一走它就成了悬空引用：读者问「哪个模型用了」，表里答不上来。
#   ⭐ 而 MFA、CLA、K=V 共享留着，因为它们各自都有在用的模型。
ROWS += [
    ("2025-12", "DeepSeek-V3.2　671B/37B · 61 层", [("DSA", 1)], "160K", ("mla", 61, 576),
     "Exp 转正。<tspan font-weight=\"700\">index_topk 512 → 2048</tspan>，KV 与 V3 一样"),
    ("2026-02", "MiniMax M2.5　230B/10B · 62 层", [("GQA", 1)], "192K", ("gqa", 62, 8, 128),
     "⚠️ <tspan font-weight=\"700\">架构与 M2 逐字段相同</tspan>，稀疏要等 M3"),
    ("2026-04", "Gemma 4 31B　31B 稠密 · 60 层",
     [("SWA", 5), ("FULL", 1)], "256K", ("swahyb2", 10, 50, 4, 512, 16, 256, 1024),
     "⭐ 全局层 <tspan font-weight=\"700\">K 维加倍 ＋ K=V 共享</tspan>，窗口 1024"),
]


# ══════════════════════════════════════════════════════════════════
# ⛔⛔ 2026-09-08 第九轮：现场要求「别漏掉人家最新发布的东西，
#     漏了会被学生认为不严谨」。按家扫了一遍最新发布，**补两行真缺口**。
#
# ⭐⭐ 【缺口一 · Mistral Large 3】675B/41B，2025-12。
#   本表原先 Mistral 只有 2023 年那个 7B ——&nbsp;**欧洲最大的开源旗舰整个不在表上**。
#   ⭐ 而且扒开 params.json 之后，冒出一条比「补一行」值钱得多的事实：
#     它的注意力是 **MLA，而且超参跟 DeepSeek-V3 逐字段一样**：
#     n_layers 61、dim 7168、n_heads 128、kv_lora_rank 512、
#     q_lora_rank 1536、qk_nope_head_dim 128、qk_rope_head_dim 64。
#     → **MLA 已经从「DeepSeek 的自研」扩散成了跨大洲的行业默认。**
#   ⚠️ 它没有 transformers 版 config（模型卡原话：「We sadly didn't have enough
#     time to add Mistral Large 3 to transformers」），参数读自 params.json。
#
# ⭐⭐ 【缺口二 · Qwen3.8-Flash-Next】2026-08-26，model_type 是 qwen4_exp_text
#   ——&nbsp;**Qwen4 架构的预览版**。本表原先千问最新只到 Qwen3.5（2026-03），
#   **漏了整整一代架构**，这是最容易被学生当成「材料没更新」的那种缺口。
#   · 注意力仍是 3:1（full_attention_interval = 4，48 层里 12 层全注意力），
#     线性那一支的配置（linear_conv_kernel_dim 4、linear_num_key_heads 16 /
#     value_heads 48、output_gate_type sigmoid）跟 Qwen3-Next 同族。
#   · ⭐ 真正的新东西**不在注意力上**：ngram_size 3、
#     ngram_vocab_size_base 20,000,000 ——&nbsp;**51B 的 n-gram 嵌入表**。
#     本表是注意力表，所以只在备注里点一句，不另开列。
#
# 📌 同一轮扫过、**确认不进表**的（记下来，免得下次又查一遍）：
#   · DeepSeek-V4-Flash-Vision-Exp（08-21）——&nbsp;V4-Flash 的多模态版，注意力没动
#   · Qwen3.8-Max（2.4T，08-02）——&nbsp;**没开权重**，本表只收开源
#   · Kimi K2.5 / K2.6 ——&nbsp;MLA，机制已被 K2 / K3 两行覆盖
#   · Ling-3.0-tiny、Ling-3.0-flash-Fin ——&nbsp;同架构的小号与行业微调版
#   · Nemotron 3.5 Lightning、Mistral Small 4 ——&nbsp;机制上没有新东西
#   · Llama 5、Ring-3.0、GLM-5.4、混元 Hy4 正式版 ——&nbsp;**查无此物**，别脑补
ROWS += [
    ("2025-12", "Mistral Large 3　675B/41B · 61 层", [("MLA", 1)], "288K", ("mla", 61, 576),
     "⭐ <tspan font-weight=\"700\">MLA 超参跟 V3 逐字段一样</tspan>；参数读自 params.json"),
    ("2026-08", "Qwen3.8-Flash-Next　125B/6B · 48 层",
     [("GDN", 3), ("gAT", 1)], "256K", ("gqa", 12, 2, 256),
     "⭐ <tspan font-weight=\"700\">Qwen4 架构预览</tspan>；新东西在 51B 的 n-gram 嵌入表"),
]

ROWS.sort(key=lambda r: r[0])
_d = [r[0] for r in ROWS]
assert _d == sorted(_d), "排序没生效"


# ══════════════════════════════════════════════════════════════════
# 下面几个是**排序键**：HTML 表点表头时按它们排。
# ⛔ 排序键要从数据算，别在 HTML 里写死 ——&nbsp;写死就等于又抄了一份。
CHEAP = LINEAR | {"SWA"}


def cheap_frac(cyc):
    """一个循环里「便宜的层」占多少。每层同构的行也能算：RWKV 是 1，MHA 是 0。"""
    tot = sum(k for _, k in cyc)
    return sum(k for t_, k in cyc if t_ in CHEAP) / float(tot)


def ctx_tokens(s):
    """把「128K」「4M」这类显示值换成数字，只用于排序。"""
    if s.startswith("无限"):
        return 1 << 40                 # 纯 RNN：理论无限，排最后
    if s == "—":
        return -1                      # 没核到的排最前（不假装它小）
    n = float(s[:-1])
    return int(n * (1024 if s.endswith("K") else 1024 * 1024))


def vendor_key(mdl, tm):
    """按厂商分组，**组内按时间**。现场说的「按模型名字也就是类别排序」就是这个。

    ⛔ 组内别按名字排 —— 字典序会把 "DeepSeek-V3.2-Exp" 排到 "DeepSeek-V3" 前面
      （'.' 的码位小于全角空格）。⭐ 而且按时间更有用：
      同一家从上到下读下来就是**这家走过的路**。
    """
    lab, _ink, _bg = vendor_of(mdl)
    return (lab, tm)


def total_params_b(mdl):
    """从模型名里取**总参数**（B）。"GPT-3　175B 稠密" → 175，"DeepSeek-V4-Flash　284B/13B" → 284。

    ⭐ 只用来把「小模型」挑出去 —— 拿 7B 的 KV 去跟 175B 的 KV 比倍数，
      比出来的是**模型大小**，不是机制。⛔ 别用它做别的推断，
      名字里的参数量是展示用的，不是 config 读出来的。
    """
    seg = mdl.split("　")[-1].split("·")[0].strip()
    seg = seg.split("/")[0].split("–")[0].split(" ")[0]
    try:
        return float(seg.rstrip("B"))
    except ValueError:
        return 0.0


def short_name(mdl):
    """"DeepSeek-V2　236B/21B · 60 层" → "DeepSeek-V2"。分隔符是全角空格 U+3000。"""
    return mdl.split("　")[0].replace("⭐ ", "").strip()


# ══════════════════════════════════════════════════════════════════
# ⭐⭐ Highlight 集 —— 2026-09-07 现场要求：
#
#     「做个开关，叫 highlight 和 boom：有全部的信息，或者只有 highlight 的信息，
#       两种状态切换。highlight 就是把重要的那些模型都挑出来，
#       能够把整个 Attention 的演变史说明白就行了。」
#
# ⭐ 入选判据只有一条：**把它拿掉，这段历史就断一节。**
#   不是「这个模型重要」，是「这一行承担了一个别人替不了的角色」——
#   所以 Kimi K2、GLM-5、Llama 3.1 这些很强的模型反而不在里面：
#   它们各自的机制已经有更早的首发行占着位置了。
#
# ⛔ 一个机制只留**首次出现的那一个**，除了三处**故意的例外**（都写了理由）。
# ⛔ 别按厂商配额来凑 —— 这是机制线不是厂商榜。混元只有一行、蚂蚁和小米一行没有，
#   是因为它们的机制在别人那儿已经首发过了，不是漏了。
HL = {
    # ── 基线：旋钮①一路拧下去，四个刻度 ──────────────────────────
    "GPT-3":             "起点。576 GiB 这把尺，后面所有省下来的都拿它比",
    "PaLM":              "MQA —— 第一次把 KV 砍到只剩 1 组",
    "Llama 2":           "GQA —— 砍到 1 组太狠，折中版成了此后十年的默认",
    "DeepSeek-V2":       "MLA 首发。换了个思路：不砍头数，改存压缩过的隐向量",
    # 例外①：V3 跟 V2 机制相同，但它是**第一讲的锚点**，
    #        而且「参数差 2.8 倍、KV 只差 1.7%」这条结论要靠这一对才成立。
    "DeepSeek-V3":       "跟 V2 同机制，但参数大 2.8 倍而 KV 只差 1.7% —— KV 脱钩的证据",
    # ── 旋钮②：先是滑窗，后是层内稀疏 ────────────────────────────
    "Mistral 7B":        "SWA —— 「不看全部」这个想法的起点，窗口 4096",
    "Gemma 3 27B":       "滑窗混合定型：5 层窗口配 1 层全局，窗口反而收到 1024",
    "DeepSeek-V3.2-Exp": "DSA —— 层内稀疏起点：不是少几层，是每层只挑一部分 token 看",
    "GLM-5.2":           "IndexShare —— 稀疏的第二阶段：索引本身变成了新的开销",
    "DeepSeek-V4-Flash": "终点。697 MiB，比 GPT-3 小 846 倍，而且 MLA 被整个换掉了",
    # ── 旋钮③：线性 / SSM 那一支 ────────────────────────────────
    "Jamba":             "层间混合的开源起点，7:1 —— 比 MiniMax-01 早十个月",
    "MiniMax-01":        "线性第一次上到几百 B 的规模，同样是 7:1",
    "RWKV-7 Goose":      "另一头的极端：纯 RNN，KV cache literally 是 0",
    "Qwen3-Next":        "GDN —— Mamba 那一支的直系后代第一次进主流大模型",
    "Kimi Linear":       "KDA —— 线性的新一代，配 NoPE 的 MLA",
    # ── 反例与合流 ──────────────────────────────────────────────
    # 例外②：M2 不首发任何机制，但它是**三个旋钮正交**的活证据 ——
    #        同一家人，把③退回去、①仍压着。没有它这条口径就只能靠嘴说。
    "MiniMax M2":        "反例：退回全注意力。证明「全注意力」说的是旋钮③，不是①",
    # 例外③：Hy4 的 gDSA 机制上跟 GLM 那几行重叠，留它是因为**路线**独一份。
    "混元 Hy4-preview":   "跳过线性那一支，从纯 GQA 直接跳进全层稀疏",
    # ── 2026-09-07 第八轮补：原先 Highlight 里 2025-10 直接跳到 2026-05，
    #    空了七个月。⭐ 补进来之后才发现，那段**不是没发布，是没有新机制** ——
    #    见落点⑧。这四行分别代表「转正 / 收进主线 / 新招 / 新机制」四种进展。
    "DeepSeek-V3.2":     "Exp 转正：稀疏从实验走进生产，top-k 512 → 2048",
    "Mistral Large 3":   "MLA 扩散到了西方：欧洲最大开源旗舰逐字段照抄 V3 的 MLA 超参",
    "Qwen3.5":           "千问把混合注意力从旁支 Qwen3-Next 收进了主线",
    "Gemma 4 31B":       "旋钮①又出新招：全局层 K 维加倍再让 K=V 共享一份",
    "GLM-5.3-Flash":     "唯一一个把②和③同锅：34 层 KDA ＋ 11 层稀疏 MLA",
}

# ⛔ 名字打错了不会报错，只会让那一行**悄悄不在 highlight 里**。所以在这儿硬失败。
_names = {short_name(r[1]) for r in ROWS}
_miss = sorted(set(HL) - _names)
assert not _miss, "HL 里这些名字在 ROWS 中不存在（打错了？）：%s" % _miss
assert len(_names) == len(ROWS), "有重名的模型行 —— short_name 不能当唯一键了"


def is_hl(mdl):
    return short_name(mdl) in HL
