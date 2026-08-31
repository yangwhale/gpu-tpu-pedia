# -*- coding: utf-8 -*-
"""图 1-5 · 第 1 节的落点。

⭐ **为什么要有这一张。** 第 1.5 节原来是一张九行的文字表：层、TPU、GB200、
差别的本质，四列九行全是字。信息没错，但它是**给人回去查的**，不是**给人在
台上讲的** —— 九行并列，每一行看起来一样重，台下不知道该记哪一条。

所以拆成两件东西：这张图负责**落点**（三句话，讲两分钟），
原来那张表留在图下面当**手册**（发下去自己查，台上不念）。

三格的分法不是随手挑的，它就是那九行按「硬度」分的三类：
① 一边有一边完全没有（这是**结构性**差别，参数表上看不出来）
② 两边都有、量级相同（这些**不值得讲**，讲了反而冲淡重点）
③ 两边算出来撞在同一个数上（这是第 1 节唯一需要背下来的数）
"""
import io

BL, PU, OR, GR, RD, GY, YL, TL = ("#1a73e8", "#9334e6", "#e8710a", "#1e8e3e",
                                  "#d93025", "#5f6368", "#f9ab00", "#12786f")

W, PW, GAP = 1000, 320, 20
X = [i * (PW + GAP) for i in range(3)]
TOP, BH = 76, 250
H = TOP + BH + 150

CARDS = [
    # 这一格没有「大数字」—— 它讲的正是「没有」。第三项留空，画的时候跳过。
    ("① 只有一边有的", "结构性差别 —— 参数表上看不出来", RD, [
        ("TPU 没有", "「硬件自动管的片上缓存」这一整层（CMEM = 0）", ""),
        ("GPU 没有", "「可编程的专用协处理器」这一整层（无 SparseCore）", ""),
    ], ["⭐ 这两格是空的，不是小 ——", "第 2、3 节全从这儿长出来"]),
    ("② 两边都有、量级相同", "讲了反而冲淡重点，发下去自己看", GY, [
        ("算力", "2,307 对 2,500 TFLOPS", "差 8%"),
        ("HBM 带宽", "7,380 对 8,000 GB/s", "差 8%"),
        ("片上 SRAM", "134 MB 对 126 MB", "差 6%"),
    ], ["这一格的正确讲法是：跳过。", "三条都在 10% 以内，讲了只会冲淡重点"]),
    ("③ 撞在同一个数上", "第 1 节唯一要背下来的", TL, [
        ("TPU v7", "2,307 ÷ 7.38", "312.6"),
        ("GB200", "2,500 ÷ 8.00", "312.5"),
    ], ["⚠️ 相对差 0.03%，BF16 和 FP8 完全一样。", "「为什么一样」是推断；换 HGX B200 就是 281.3"]),
]

p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
     f'aria-label="第 1 节的落点：只有一边有的、两边量级相同的、以及算力除带宽撞在同一个数上">',
     '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
     '第 1 节的落点 —— 那张九行的表，真正要讲的只有这三格</text>',
     '<text class="svgsm" x="0" y="35">'
     '按「硬度」分类，不按「层级」分类：谁是结构性差别、谁只是数字不同、谁是巧合</text>',
     '<text class="svgsm" x="0" y="54" fill="#9aa0a6">'
     '完整九行的对照表在这张图下面 —— 那是给人回去查的，台上不念</text>']

for x, (ttl, sub, c, rows, foot) in zip(X, CARDS):
    p.append(f'<rect x="{x}" y="{TOP}" width="{PW}" height="{BH}" rx="10" '
             f'fill="#fff" stroke="{c}" stroke-width="1.6"/>')
    p.append(f'<rect x="{x}" y="{TOP}" width="{PW}" height="46" rx="10" fill="{c}"/>')
    p.append(f'<rect x="{x}" y="{TOP+36}" width="{PW}" height="10" fill="{c}"/>')
    p.append(f'<text class="svglbl" x="{x+16}" y="{TOP+21}" fill="#fff" '
             f'style="font-size:13px">{ttl}</text>')
    p.append(f'<text class="svgsm" x="{x+16}" y="{TOP+38}" fill="#ffffffcc">{sub}</text>')
    for j, (a, b, big) in enumerate(rows):
        yy = TOP + 62 + j * 52
        p.append(f'<text class="svgsm" x="{x+16}" y="{yy+12}" fill="{GY}">{a}</text>')
        p.append(f'<text class="svgsm" x="{x+16}" y="{yy+29}" fill="#202124">{b}</text>')
        if big:
            p.append(f'<text class="svgnum" x="{x+PW-16}" y="{yy+24}" text-anchor="end" '
                     f'fill="{c}" style="font-size:17px">{big}</text>')
    # 脚注钉在卡片底部，不跟着行数浮动 —— 三张卡行数不同，浮动会导致三条脚注不齐
    p.append(f'<line x1="{x+14}" y1="{TOP+BH-52}" x2="{x+PW-14}" y2="{TOP+BH-52}" '
             f'stroke="#e8eaed"/>')
    for j, t in enumerate(foot):
        p.append(f'<text class="svgsm" x="{x+16}" y="{TOP+BH-34+j*15}" fill="{c}" '
                 f'style="font-size:10px">{t}</text>')

YB = TOP + BH + 22
p.append(f'<rect x="0" y="{YB}" width="{W-18}" height="86" rx="8" '
         f'fill="#fef7e0" stroke="{YL}"/>')
p.append(f'<text class="svglbl" x="18" y="{YB+24}" fill="#7a5000">'
         f'⭐ 第 1 节到此为止，一句话：两边的差别<tspan font-weight="700">不在参数表上</tspan></text>')
p.append(f'<text class="svgsm" x="18" y="{YB+45}" fill="#7a5000">'
         f'第 ② 格已经说明了 —— 算力、带宽、片上 SRAM 三项都在 10% 以内，'
         f'第 ③ 格甚至撞在同一个数上。<tspan font-weight="700">按参数表，这是同一类芯片。</tspan></text>')
p.append(f'<text class="svgsm" x="18" y="{YB+64}" fill="#7a5000">'
         f'真正的差别全在第 ① 格：<tspan font-weight="700">各缺对方整整一层</tspan>。'
         f'那一层缺的不是容量，是「<tspan font-weight="700">谁做决定</tspan>」——'
         f'一个交给硬件，一个交给编译器。</text>')
p.append(f'<text class="svgsm" x="18" y="{YB+80}" fill="#7a5000">'
         f'第 2 节就从这一格开始：同一次访存，两边路上各有几个「运行时决策点」。</text>')
p.append('</svg>')
io.open('fig1-5.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig1-5 ok', H)
