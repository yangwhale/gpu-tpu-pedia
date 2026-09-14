# -*- coding: utf-8 -*-
r"""专题三 · §8.2b「守恒的是比值，还是全注意力的层数？」

⭐⭐⭐ 2026-09-14 夜间 R31 新画。这张图**回答 R28 留下的那个悬案**，
   顺带**修掉课程自己的一处措辞**。

═══ 起因 ═══
§8.1 原话：「关键在于全局层**不需要很多** —— 只要有几层能做无损检索，
  信息就能沿着残差流传给其余层用。」
⛔ 这句话**按字面读是一个可证伪的预测**：如果「几层就够」，那全注意力层的
  **绝对条数**应该跟深度无关 —— 模型越深，比值就该越大。
⭐ 而 §8.2 的 K3 那一行又摆着「93 层里 24 层全注意力」这个反常大的绝对数。
   到底哪个是超参？这一格就是去数配置文件。

═══ 结论：绝对数**不是**守恒量，比值才是（数据否掉了字面读法）═══
16 个模型的 config.json 逐个拉下来数（全部一手，见下），结果很干脆：
  · Qwen3.5 七个尺寸，深度 24 → 64，`layer_types` 数组逐层写死，
    **比值恰好 3:1 一次不差**，绝对数 6 → 16 跟着深度线性涨。
  · Kimi 自己就是最干净的反证：Kimi Linear 27 层 / 7 层全注意力，
    K3 93 层 / 24 层 —— **绝对数涨了 3.4 倍，比值纹丝不动。**
  · Hunyuan-TurboS 560B、128 层只用 7 层全注意力；
    Qwen3.5-27B 小 20 倍却用 16 层。**方向都反了**，「几层就够」不成立。

⭐ 但也别一刀切成「就是比值」。更准的说法是**两条正交的规则叠加**：
   **① 主体按固定比例铺（3:1 / 7:1 / 每 10% 一层）—— 这部分随深度线性涨；
   ② 外加若干个按「位置」钉死的全局层 —— 这部分是常数。**
   Kimi 两个模型都是「每 4 层一个 ＋ **末层必为全局**」，正是末层那一个
   让实际比值略低于 3:1（20:7 = 2.857、69:24 = 2.875）。
   Hymba 是把 ② 用到极致：全局层只有**首 / 中 / 末**三层，按位置定。

⛔⛔ 顺带修掉的课程错误：§8.2 表里 Kimi Linear 那行只写「3:1」，
   而 K3 那行专门警告了「别写成 69:24 = 3:1，一除就是 2.875」——
   **同一个警告对 Kimi Linear 同样成立（20:7 = 2.857），课程漏了**。
   而且课程把「末层补一个 MLA」当成 K3 的特有设计，**其实 Kimi Linear
   也是这么排的**，它是这一家的通用规则，不是 K3 的花样。

═══ 数据来源（全部一手读 config.json，不是论文转述）═══
  Kimi Linear 48B-A3B-Instruct：`num_hidden_layers=27`，
    `linear_attn_config.full_attn_layers=[4,8,12,16,20,24,27]`（7 个），
    `kda_layers` 20 个。
  Kimi K3：`text_config.num_hidden_layers=93`，
    `full_attn_layers=[4,8,…,92,93]`（24 个），`kda_layers` 69 个。
  Qwen3.5 七档：`layer_types` 显式数组，
    0.8B/2B 24 层 18+6、4B/9B 32 层 24+8、35B-A3B 40 层 30+10、
    397B-A17B 60 层 45+15、27B 64 层 48+16。
  Qwen3-Next-80B-A3B：48 层 ＋ `full_attention_interval=4` → 12（推算，链条如此）。
  MiniMax-Text-01：80 层，`attn_type_list` 求和 ＝ 10。
  MiniMax-M2：62 层，`attn_type_list` 全 1 ＝ **62，整族退回全注意力**。
  Granite 4.0 h-micro / h-tiny / h-small：三档**都是** 40 层、
    `layer_types` = 36 mamba ＋ 4 attention。
    ⛔ 三档深度相同，所以这三行**区分不了两个假说**，不能当证据 ——
      图里只画一个点并标明。
  Bamba-9B：32 层，`attn_layer_indices=[9,18,27]` → 3。
  Jamba v0.1：32 层，`attn_layer_period=8` / `offset=4` → 4（推算）。
  Ring-mini-linear-2.0：20 层 `layer_group_size=5` → 4（推算）。
  Ring-flash-linear-2.0：32 层 `layer_group_size=8` → 4（推算）。

⚠️ 图里**不画** Hunyuan-TurboS / Jamba-1.5-Large / Nemotron-H：
   它们把 Mamba、Attention、FFN **各算一层**，跨家族比「总层数」会差一倍，
   而我没有逐层数组去核 mixer 口径。结论里以文字提 TurboS，不进散点。
⛔ 图里每一个数都来自上面那批 config 的字段，相关系数是本脚本当场算的。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2)

W_ = 1400

# (显示名, mixer 层数, 全注意力层数, 家族键, 是否一手字段)
#   家族键用来上色：kimi / qwen / 其他
DATA = [
    ("Kimi Linear 48B",  27,  7, "kimi", True),
    ("Kimi K3",          93, 24, "kimi", True),
    ("Qwen3.5-0.8B/2B",  24,  6, "qwen", True),
    ("Qwen3.5-4B/9B",    32,  8, "qwen", True),
    ("Qwen3.5-35B-A3B",  40, 10, "qwen", True),
    ("Qwen3-Next-80B",   48, 12, "qwen", False),
    ("Qwen3.5-397B",     60, 15, "qwen", True),
    ("Qwen3.5-27B",      64, 16, "qwen", True),
    ("MiniMax-Text-01",  80, 10, "etc", True),
    ("Granite 4.0 ×3",   40,  4, "etc", True),
    ("Bamba-9B",         32,  3, "etc", True),
    ("Jamba v0.1",       32,  4, "etc", False),
    ("Ring-mini-linear", 20,  4, "etc", False),
    ("Ring-flash-linear",32,  4, "etc", False),
]

# ── Kimi 的排布规则：每 4 层一个全注意力，末层再补一个 ──────────────
def kimi_full(n):
    """`full_attn_layers` ＝ [4, 8, …] 加上末层 n（n 不是 4 的倍数时多出一个）。"""
    s = set(range(4, n + 1, 4)) | {n}
    return len(s)


assert kimi_full(27) == 7 and kimi_full(93) == 24, "对不上 config 就别往下画"
assert 27 % 4 and 93 % 4, "两个都不是 4 的倍数，末层那一个才是额外的"
# 实际比值：末层那一个把 3.0 压下来一点点
_R_KL = (27 - 7) / 7.0        # 20 : 7  = 2.857…
_R_K3 = (93 - 24) / 24.0      # 69 : 24 = 2.875
assert abs(_R_KL - 2.857142857142857) < 1e-12
assert abs(_R_K3 - 2.875) < 1e-12
assert _R_KL < _R_K3 < 3.0, "层数越深，末层那一个被摊得越薄，比值越贴近 3"

# ── 两条相关系数，当场算，不引外面的数 ─────────────────────────
def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / float(n)
    my = sum(ys) / float(n)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / (sxx * syy) ** 0.5


XS = [d[1] for d in DATA]
YS = [d[2] for d in DATA]
R_ALL = pearson(XS, YS)
assert 0.5 < R_ALL < 0.95, R_ALL
# 只看 Kimi ＋ Qwen 这两族（同一家、同一配比规则、只变深度）
_KQ = [d for d in DATA if d[3] in ("kimi", "qwen")]
R_KQ = pearson([d[1] for d in _KQ], [d[2] for d in _KQ])
assert R_KQ > 0.99, R_KQ
# Qwen3.5 七档（这里只画了 5 个点，因为同层数的合并了）全部恰好 3:1
for nm, n, k, fam, _ in DATA:
    if nm.startswith("Qwen3.5"):
        assert n == k * 4, (nm, n, k)
# 绝对数与比值，谁散得更开
_RATIO = [(n - k) / float(k) for _, n, k, _, _ in DATA]
SPREAD_CNT = max(YS) / float(min(YS))          # 24 / 3 = 8
SPREAD_RAT = max(_RATIO) / min(_RATIO)         # 最大比值 / 最小比值
assert abs(SPREAD_CNT - 8.0) < 1e-9
assert SPREAD_CNT > SPREAD_RAT, "绝对数散得更开 —— 它不是那个守恒量"

COL = {"kimi": OR, "qwen": BL, "etc": GY2}


def main():
    f = Fig(W_, "守恒的是比值不是绝对层数：十四个已公开混合模型的配置文件里，"
                "全注意力层的绝对条数随深度线性涨，"
                "Kimi 从 27 层 7 条涨到 93 层 24 条而比值不动，"
                "Qwen3.5 七个尺寸逐层写死、恰好 3 比 1；"
                "更准的说法是主体按比例铺，外加几个按位置钉死的全局层")
    f.marks = set()
    y0 = f.header(
        "守恒的是比值，还是全注意力的层数",
        "去数<tspan font-weight=\"700\">配置文件</tspan>，不猜 ——&#160;"
        "十四个模型的 layer_types 逐个拉下来",
        [(BL, "Qwen 系"), (OR, "Kimi 系"), (GY2, "其他各家")])

    # ══════════ ① 两个假说 ══════════════════════════════════════
    PH1 = 180
    top = f.panel(0, y0, W_, PH1, "① 先把两个假说写清楚，它们的预测不一样", PU,
                  sub="能被数据分开的问题才值得问")

    f.box(40, top + 18, 640, 142, "#fff", LINE, 8, 1.2)
    f.t(64, top + 50, "假说 A ·「比值守恒」", BL, True, 17)
    f.lines(64, top + 76, 592, [
        "配比 3:1 是超参，模型越深，全注意力层数<tspan font-weight=\"700\">"
        "跟着线性涨</tspan>。",
        "预测：把「层数」对「深度」画出来，是一条<tspan font-weight=\"700\">"
        "斜着上去的直线</tspan>。",
    ], size=15, lh=26)

    f.box(720, top + 18, W_ - 760, 142, "#fff", LINE, 8, 1.2)
    f.t(744, top + 50, "假说 B ·「几层就够」", RD, True, 17)
    f.lines(744, top + 76, W_ - 808, [
        "本课自己写过：「全局层<tspan font-weight=\"700\">不需要很多</tspan>"
        " —— 只要有几层能做",
        "无损检索，信息就能沿残差流传给其余层。」"
        "<tspan font-weight=\"700\">按字面读</tspan>，它预测的是",
        "一条<tspan font-weight=\"700\">平的线</tspan>：全注意力层数与深度无关。",
    ], size=15, lh=26)

    # ══════════ ② 数据 ══════════════════════════════════════════
    y1 = y0 + PH1 + 20
    PH2 = 520
    top = f.panel(0, y1, W_, PH2, "② 十四个模型的配置文件，逐个数出来", BL,
                  sub="横轴 ＝ 总层数　·　纵轴 ＝ 全注意力层的绝对条数")

    PX, PY, PW, PH = 96, top + 46, 720, 380
    XMAX, YMAX = 100.0, 26.0

    def sx(v):
        return PX + v / XMAX * PW

    def sy(v):
        return PY + PH - v / YMAX * PH

    # 网格与坐标
    for v in range(0, 101, 20):
        f.line(sx(v), PY, sx(v), PY + PH, "#eceff1", 1, arrow=False)
        f.t(sx(v), PY + PH + 22, str(v), GY, False, 14, "middle")
    for v in range(0, 27, 5):
        f.line(PX, sy(v), PX + PW, sy(v), "#eceff1", 1, arrow=False)
        f.t(PX - 12, sy(v) + 5, str(v), GY, False, 14, "end")
    f.line(PX, PY + PH, PX + PW, PY + PH, LINE2, 1.4, arrow=False)
    f.line(PX, PY, PX, PY + PH, LINE2, 1.4, arrow=False)
    f.t(PX + PW / 2.0, PY + PH + 46, "总层数", INK, True, 15, "middle")
    f.t(PX - 62, PY - 16, "全注意力层数", INK, True, 15)

    # 假说 A 的那条线：3:1（每 4 层一个）
    f.line(sx(0), sy(0), sx(100), sy(25), BL, 1.6, dash="6 5", arrow=False)
    f.t(sx(64) + 10, sy(16) - 10, "假说 A 预测的样子（3 : 1）", BL, True, 14)
    # 假说 B 的那条线：常数（取 §8.1 字面读法，画在中位数上）
    f.line(sx(0), sy(6), sx(100), sy(6), RD, 1.6, dash="6 5", arrow=False)
    f.t(sx(72) + 6, sy(6) - 10, "假说 B 预测的样子（平的）", RD, True, 14)

    for nm, n, k, fam, first in DATA:
        c = COL[fam]
        f.box(sx(n) - 6, sy(k) - 6, 12, 12, c, "#fff", 6, 1.6)

    # 标注：只标关键几个，其余进右边的表
    def tag(nm, n, k, dx, dy, col, anchor=None):
        f.t(sx(n) + dx, sy(k) + dy, nm, col, True, 14, anchor)

    tag("Kimi Linear　27 层 / 7 条", 27, 7, 12, -12, OR)
    tag("Kimi K3　93 层 / 24 条", 93, 24, -12, -14, OR, "end")
    tag("Qwen3.5-27B　64 / 16", 64, 16, -12, -14, BL, "end")
    tag("Qwen3.5-0.8B　24 / 6", 24, 6, 14, 22, BL)
    # ⛔ 这一条原来放在点下方（dy=+26），anchor="end" 让它的左端一路顶回 x≈460，
    #   正好撞上 Kimi Linear 那条标注 —— 几何 lint 抓到的。挪到点上方就分开了。
    tag("MiniMax-Text-01　80 / 10", 80, 10, -12, -14, GY2, "end")
    tag("Granite 4.0　40 / 4", 40, 4, 12, 24, GY2)

    # 右侧：判读
    RX = PX + PW + 60
    f.t(RX, PY + 6, "点落在哪条线上", INK, True, 17)
    f.lines(RX, PY + 30, W_ - RX - 50, [
        "<tspan font-weight=\"700\" fill=\"%s\">Qwen3.5 七个尺寸</tspan>"
        "（图上合成 5 个点）：" % BL,
        "深度 24 → 64，`layer_types` 逐层写死，",
        "<tspan font-weight=\"700\">恰好 3:1，一次不差</tspan>。",
        "",
        "<tspan font-weight=\"700\" fill=\"%s\">Kimi 自己就是最干净的反证</tspan>："
        % OR,
        "27 层 7 条 → 93 层 24 条，"
        "<tspan font-weight=\"700\">绝对数涨 3.4 倍</tspan>，",
        "比值纹丝不动。",
        "",
        "⭐ 全体 14 个点，深度与条数的相关系数",
        "<tspan font-weight=\"700\">r = %.2f</tspan>；只看 Kimi ＋ Qwen 两族，"
        % R_ALL,
        "<tspan font-weight=\"700\">r = %.3f</tspan>。" % R_KQ,
        "",
        "⛔ <tspan font-weight=\"700\">假说 B 被否掉了</tspan> ——&#160;"
        "而它就是本课",
        "本课那句话的字面读法。",
    ], size=15, lh=25)

    # ══════════ ③ 精修 ══════════════════════════════════════════
    y2 = y1 + PH2 + 20
    PH3 = 292
    top = f.panel(0, y2, W_, PH3, "③ 但也别一刀切 ——　真实规则是两条叠加", GR,
                  sub="一条随深度涨，一条是常数")

    f.box(40, top + 18, 640, 224, "#fff", GR, 8, 1.6)
    f.t(64, top + 50, "① 主体：按固定比例铺", GR, True, 17)
    f.lines(64, top + 74, 592, [
        "3:1（Qwen3.5、Kimi）、7:1（MiniMax-01、Ring-flash）、",
        "每 10 层一个（Granite）——&#160;"
        "<tspan font-weight=\"700\">这部分随深度线性涨</tspan>。",
    ], size=15, lh=25)
    f.t(64, top + 154, "② 外挂：按「位置」钉死几个全局层", OR, True, 17)
    f.lines(64, top + 178, 592, [
        "Kimi 两个模型都是「每 4 层一个 ＋ "
        "<tspan font-weight=\"700\">末层必为全局</tspan>」。",
        "<tspan font-weight=\"700\">这部分是常数</tspan>，不随深度涨。",
    ], size=15, lh=25)

    f.box(720, top + 18, W_ - 760, 224, "#fff", LINE, 8, 1.2)
    f.t(744, top + 50, "⭐ 两条叠加，正好解释一个小数点", INK, True, 17)
    f.lines(744, top + 74, W_ - 808, [
        "末层那一个额外的全局层，会把实际比值"
        "<tspan font-weight=\"700\">从 3.0 压下来一点</tspan>：",
        "Kimi Linear　20 : 7 　= <tspan font-weight=\"700\">%.3f</tspan>"
        "　（27 层，摊得薄的分母小）" % _R_KL,
        "Kimi K3　　　69 : 24 = <tspan font-weight=\"700\">%.3f</tspan>"
        "　（93 层，同一个 +1 被摊得更薄）" % _R_K3,
        "",
        "⛔ 本课那张配比表只给 K3 加了「别写成 3:1」的警告 ——",
        "<tspan font-weight=\"700\">同一句对 Kimi Linear 一样成立</tspan>。",
        "而且「末层补一个」<tspan font-weight=\"700\">不是 K3 的花样</tspan>，",
        "是这一家的通用排法。",
    ], size=15, lh=25)

    yy = y2 + PH3 + 22
    yy = f.band(yy, "info", "所以「全局层不需要很多」该怎么说才准", [
        "准确的意思是<tspan font-weight=\"700\">占比低</tspan>"
        "（各家落在 1/4 到 1/10 之间），"
        "<tspan font-weight=\"700\">不是绝对条数少</tspan>。"
        "全体 14 个点里，绝对条数从 3 到 24 差 %d 倍，而比值只差 %.1f 倍 ——&#160;"
        "<tspan font-weight=\"700\">散得开的那个不是超参</tspan>。"
        % (round(SPREAD_CNT), SPREAD_RAT),
        "反过来说，<tspan font-weight=\"700\">模型越深，你要付的全注意力层就越多</tspan>"
        " ——&#160;混合省下的是<tspan font-weight=\"700\">一个固定比例</tspan>，"
        "不是「越深越划算」。",
    ])
    yy = f.band(yy + 12, "warn", "三处别讲过头", [
        "① <tspan font-weight=\"700\">Granite 4.0 三档都是 4 条，但三档深度都是 40</tspan>"
        " ——&#160;这三行<tspan font-weight=\"700\">区分不了两个假说</tspan>，"
        "不能拿来当「绝对数守恒」的证据。图上只画一个点。",
        "② 确实有按<tspan font-weight=\"700\">位置</tspan>定的设计："
        "Hymba 的全局层只有<tspan font-weight=\"700\">首 / 中 / 末</tspan>三层，"
        "理论上多深都是 3。它是假说 B 唯一站得住的一手证据，"
        "但它是<tspan font-weight=\"700\">按位置</tspan>不是<tspan "
        "font-weight=\"700\">按数量</tspan>定的，两回事。",
        "③ 图上<tspan font-weight=\"700\">没有</tspan> Hunyuan-TurboS / "
        "Nemotron-H / Jamba-1.5-Large：它们把 Mamba、Attention、FFN "
        "<tspan font-weight=\"700\">各算一层</tspan>，跨家族比「总层数」会差一倍。"
        "TurboS 的 128 层里只有 7 层全注意力，方向上更不支持假说 B，但口径不同，不进图。",
    ])

    yy = f.src(yy + 14,
               "📌 全部数据来自各模型 HuggingFace 仓库的 config.json 逐个字段："
               "Kimi 两款读 linear_attn_config.full_attn_layers（27 层 7 个 / "
               "93 层 24 个）；Qwen3.5 七档读 layer_types 数组；"
               "MiniMax-Text-01 读 attn_type_list 求和；"
               "Granite / Bamba 读 layer_types 与 attn_layer_indices。",
               "📌 Qwen3-Next（interval=4）、Jamba v0.1（period=8, offset=4）、"
               "Ring 两款（layer_group_size）是<tspan font-weight=\"700\">"
               "按字段推算</tspan>的，推导链已写在图注与脚本头里。",
               "⛔ 两个相关系数是本脚本当场算的；"
               "「⌊每 4 层一个⌋ ＋ 末层补一个」这条排法由 assert 对着两份 "
               "config 的实际数组核过。⚠️ 只有两个点，"
               "把它当「观察到的排法」，不要当 Kimi 公布的规则。")

    f.save("fig3-ratio-or-count.svg", yy + 10)


if __name__ == "__main__":
    main()
