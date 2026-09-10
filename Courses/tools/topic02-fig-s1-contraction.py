# -*- coding: utf-8 -*-
r"""图 1-8 · 强度到底由哪个维度决定 —— 收缩维 K

⭐ 这张图是 2026-09-10 现场问出来的，问题原话：
   「方阵长宽都是 N，那 M×K · K×N 呢？算术强度应该跟收缩维 K 有关，
     K 越长强度越高，对吧？而 head_dim=128 的注意力就严重不够。
     那做了 Flash 之后，是因为什么变了？」
   —— 三问全对，而且它把 §1（融合）、§2（Flash）、§3.4/3.6（喂不满）
      三处第一次连成了一条线。所以单独立一张。

⛔ 图里所有数都由下面的公式**当场算出来**，不写死 —— 免得改了公式忘了改数。
⛔ 口径陷阱（这张图存在的第二个理由）：
   「注意力的强度」有两个都对的答案，分母不一样：
     · 单个 QK^T 矩阵乘、孤立看（A/B/C 各过一次）→ ≈ d ＝ 128（本图）
     · 朴素注意力整体看（中间矩阵 S 和 P 各走一个来回）→ d/2 ＝ 64（§2 fig2-4）
   两张图并排放而不说破，读者一定以为其中一张错了。**所以必须写在图上。**
"""
import io

BL, PU, OR, GR, RD, GY, YL = ("#1a73e8", "#9334e6", "#e8710a",
                              "#1e8e3e", "#d93025", "#5f6368", "#f9ab00")
INK, LINE = "#202124", "#dadce0"


def W(f, p):
    io.open(f, "w", encoding="utf-8").write("\n".join(p))


# ── 唯一的公式：1/I = 1/M + 1/N + 1/K ────────────────────────────
def I(M, K, N):
    return 1.0 / (1.0 / M + 1.0 / N + 1.0 / K)


RIDGE_V7, RIDGE_V6E = 312, 560
D_HEAD = 128
S_WAN = 75600
MN = 512                                    # ② 里固定的 M、N
KS = (64, 128, 512, 4096)
CEIL = MN * MN / (MN + MN)                  # K→∞ 的上限
assert abs(I(999999, 999999, 999999) / 999999 - 1 / 3) < 1e-6      # 方阵退化 = N/3
assert abs(I(256, 256, 256) - 256 / 3) < 1e-6                      # §1 那个 85
assert abs(I(S_WAN, D_HEAD, S_WAN) - 127.6) < 0.2                  # 注意力 ≈ 128
assert S_WAN // 2 == 37800                                         # Flash 后 = S/2

PW, GAP, X0 = 320, 15, 0
TOP = 96
PH = 250
BAND_Y = TOP + PH + 18
H = BAND_Y + 52 + 12 + 122 + 8

p = ['<svg viewBox="0 0 1000 %d" width="100%%" role="img" '
     'aria-label="矩阵乘的算术强度由三个维度共同决定，倒数相加；'
     '注意力的收缩维就是 head_dim 128，因此强度上不去">' % H,
     '<text class="svglbl" x="0" y="16" fill="%s" style="font-size:13.5px">'
     '强度到底由哪个维度决定 —— <tspan font-weight="700">是最小的那个，'
     '而注意力里最小的就是 head_dim</tspan></text>' % INK,
     '<text class="svgsm" x="0" y="35">§1 只算了方阵（N/3）。真实的矩阵乘是 '
     'M×K · K×N —— 三个维度各有各的作用，而它们的作用方式是「倒数相加」。</text>']

# 顶部 punch
p.append('<rect x="0" y="48" width="982" height="34" rx="6" fill="#e8f0fe" '
         'stroke="%s"/>' % BL)
p.append('<text class="svgsm" x="16" y="69" fill="#174ea6" '
         'font-family="ui-monospace,monospace">'
         '算 2·M·N·K　÷　搬 2·(MK＋KN＋MN)　⟹　'
         '<tspan font-weight="700">1/强度 ＝ 1/M ＋ 1/N ＋ 1/K</tspan>'
         '　（M=N=K 时退化成 N/3，正是 §1 那个例子）</text>')


def panel(x, no, title, color, lines, foot, footc):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="8" fill="#fff" '
             'stroke="%s" stroke-width="1.6"/>' % (x, TOP, PW, PH, color))
    p.append('<rect x="%d" y="%d" width="%d" height="26" rx="8" fill="%s"/>'
             % (x, TOP, PW, color))
    p.append('<rect x="%d" y="%d" width="%d" height="8" fill="%s"'
             '/>' % (x, TOP + 18, PW, color))
    p.append('<text class="svglbl" x="%d" y="%d" fill="#fff">%s %s</text>'
             % (x + 12, TOP + 18, no, title))
    yy = TOP + 44
    for t, mono, c in lines:
        fam = ' font-family="ui-monospace,monospace"' if mono else ''
        p.append('<text class="svgsm" x="%d" y="%d" fill="%s"%s>%s</text>'
                 % (x + 14, yy, c, fam, t))
        yy += 19
    p.append('<rect x="%d" y="%d" width="%d" height="46" rx="5" fill="%s"/>'
             % (x + 10, TOP + PH - 56, PW - 20, footc))
    for i, t in enumerate(foot):
        p.append('<text class="svgsm" x="%d" y="%d" fill="#fff">%s</text>'
                 % (x + 22, TOP + PH - 38 + i * 17, t))


# ① 通式
panel(X0, "①", "三个维度，倒数相加", BL, [
    ("M×K · K×N，bf16，A/B/C 各过一次", 0, GY),
    ("算：2·M·N·K", 1, INK),
    ("搬：2·(MK ＋ KN ＋ MN) 字节", 1, INK),
    ("⟹ 1/I ＝ 1/M ＋ 1/N ＋ 1/K", 1, BL),
    ("", 0, GY),
    ("方阵 M=N=K：I ＝ N/3", 1, GR),
    ("边长 256 → I ＝ %.1f（§1 念过的 85）" % I(256, 256, 256), 1, GR),
], ["倒数相加 ⟹ 谁最小谁说了算。", "强度永远被三个维度里最小的那个卡住。"], BL)

# ② K 有用，但有天花板
x2 = X0 + PW + GAP
rows = [("固定 M ＝ N ＝ %d，只拉长 K：" % MN, 0, GY)]
for k in KS:
    rows.append(("K ＝ %-6s → I ＝ %6.1f" % (format(k, ","), I(MN, k, MN)), 1, INK))
rows.append(("K → ∞　　 → I → %.0f" % CEIL, 1, RD))
rows.append(("上限 ＝ MN/(M+N) ＝ %d，拉不动了" % CEIL, 0, RD))
panel(x2, "②", "K 越长强度越高 —— 但有天花板", OR, rows,
      ["⛔ 推论：batch=1 的 decode，M ＝ 1，",
       "所以 I ≤ 1 —— K 和 N 再大都救不了。"], RD)

# ③ 注意力代进去
x3 = x2 + PW + GAP
panel(x3, "③", "注意力：收缩维就是 head_dim", PU, [
    ("每个头，S ＝ %s，d ＝ %d" % (format(S_WAN, ","), D_HEAD), 0, GY),
    ("QK<tspan baseline-shift=\"super\" font-size=\"8\">T</tspan>："
     "M=S, N=S, <tspan font-weight=\"700\">K=d=128</tspan>", 1, INK),
    ("　→ I ＝ %.1f" % I(S_WAN, D_HEAD, S_WAN), 1, RD),
    ("PV：M=S, K=S, <tspan font-weight=\"700\">N=d=128</tspan>", 1, INK),
    ("　→ I ＝ %.1f（同样被 128 卡住）" % I(S_WAN, S_WAN, D_HEAD), 1, RD),
    ("", 0, GY),
    ("对照：v7 屋脊 %d　v6e 屋脊 %d" % (RIDGE_V7, RIDGE_V6E), 0, GY),
], ["128 ＜ 312 ＜ 560 —— 不做 Flash 的话，",
    "注意力这两个矩阵乘落在带宽那一侧。"], PU)

# ── 口径警告带（这张图存在的第二个理由）────────────────────────
# ⛔ 2026-09-10 第一版这三段都写成**单行**，右边全部跑出画布。
#   ⭐ 判据：**这幅图宽 1000 单位，一行中文塞不下 60 个全角字。**
#     长句一律先切成两行再写，不要指望「差不多能放下」。
BH = 52
p.append('<rect x="0" y="%d" width="982" height="%d" rx="6" fill="#fef7e0" '
         'stroke="%s"/>' % (BAND_Y, BH, YL))
for i, t in enumerate([
        '⚠️ <tspan font-weight="700">口径：「注意力的强度」有两个都对的答案，'
        '分母不一样。</tspan>',
        '本图 ≈128 是<tspan font-weight="700">单个矩阵乘孤立看</tspan>'
        '（A/B/C 各过一次）；§2 图 2-4 那个 64 是'
        '<tspan font-weight="700">朴素注意力整体看</tspan>'
        '（中间矩阵 S 和 P 各走一个来回，分母多一倍）。']):
    p.append('<text class="svgsm" x="16" y="%d" fill="#8a5a00">%s</text>'
             % (BAND_Y + 21 + i * 19, t))

# ── 落点带 ────────────────────────────────────────────────────
LY = BAND_Y + BH + 12
LH = 122
p.append('<rect x="0" y="%d" width="982" height="%d" rx="8" fill="#e6f4ea" '
         'stroke="%s"/>' % (LY, LH, GR))
p.append('<text class="svglbl" x="18" y="%d" fill="%s">'
         '那 Flash 到底改了什么 ——&#160;以及为什么改完了还是喂不满</text>'
         % (LY + 22, GR))
for i, t in enumerate([
        '① <tspan font-weight="700">Flash 换的是分母，不是分子</tspan>：'
        '中间那个 S×S 不再落 HBM，过 HBM 的只剩 Q/K/V/O ＝ 4·S·d·2 B，',
        '　 于是 I ＝ 4S²d ÷ 8Sd ＝ <tspan font-weight="700">S/2 ＝ %s</tspan>'
        '（S＝%s）。<tspan font-weight="700">一个 FLOP 都没省</tspan>，'
        '还因为重算多算了一点。→ §2'
        % (format(S_WAN // 2, ","), format(S_WAN, ",")),
        '② <tspan font-weight="700">同一个 128 咬了两口</tspan>：'
        '<tspan font-weight="700">收缩维 128 → 强度上不去</tspan>（本图）；'
        '<tspan font-weight="700">MXU 是 256 见方，只喂进 128 → 一半空着</tspan>'
        '（§3.4 / §3.6）。',
        '　 前者是「要不要等数据」，后者是「算的时候算得满不满」'
        '——&#160;<tspan font-weight="700">两件事，同一个数字。</tspan>']):
    p.append('<text class="svgsm" x="18" y="%d" fill="#0d652d">%s</text>'
             % (LY + 44 + i * 19, t))

W("fig1-8.svg", p + ["</svg>"])
print("fig1-8 ok  方阵256=%.1f  注意力=%.1f  K上限=%.0f  Flash=S/2=%s  H=%d"
      % (I(256, 256, 256), I(S_WAN, D_HEAD, S_WAN), CEIL,
         format(S_WAN // 2, ","), H))
