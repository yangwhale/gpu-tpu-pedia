# -*- coding: utf-8 -*-
import io
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY,YL="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368","#f9ab00"

# ══════════ 图 2-5 · 补零的两笔账 ══════════
# 内存侧（XLA 官方 tiled layout：次内维 1~2 用 2x128，3~4 用 4x128，其余 8x128）
def mem_tile(r):  return 2 if r<=2 else (4 if r<=4 else 8)
def mem_waste(r,c):
    import math
    t=mem_tile(r); slots=math.ceil(r/t)*t*math.ceil(c/128)*128
    return slots, 1-r*c/slots
def cmp_waste(r,c):
    import math
    slots=math.ceil(r/8)*8*math.ceil(c/128)*128
    return slots, 1-r*c/slots
CASES=[("[8, 128]","刚好一块"),("[1, 128]","只有一行"),("[8, 129]","多出一列"),("[4, 100]","两边都不对")]
rows=[]
for s,note in CASES:
    r,c=[int(x) for x in s.strip('[]').split(',')]
    ms,mw=mem_waste(r,c); cs,cw=cmp_waste(r,c)
    rows.append((s,note,mem_tile(r),ms,mw,cs,cw,r*c))
X0,Y0,RH=0,140,34
H=Y0+RH*len(rows)+232
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="形状对不齐要付两笔账：内存侧按小块省一点，计算侧一律补齐到 8x128">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   '形状对不齐，要付<tspan font-weight="700">两笔</tspan>账 —— 而且这两笔的数不一样</text>',
   '<text class="svgsm" x="0" y="35">来源：OpenXLA 官方「Tiled layout」页（内存侧规则）＋ JAX Pallas TPU 官方文档（计算侧规则）</text>']
# 两笔账的规则条
p.append(f'<rect x="0" y="48" width="486" height="52" rx="6" fill="#e8f0fe" stroke="{BL}"/>')
p.append(f'<text class="svglbl" x="14" y="68" fill="{BL}">账一 · 内存：XLA 会挑小块省一点</text>')
p.append(f'<text class="svgsm" x="14" y="88" fill="#174ea6">次内维 1–2 用 <tspan font-family="ui-monospace,monospace">2×128</tspan>，3–4 用 <tspan font-family="ui-monospace,monospace">4×128</tspan>，其余 <tspan font-family="ui-monospace,monospace">8×128</tspan></text>')
p.append(f'<rect x="496" y="48" width="486" height="52" rx="6" fill="#fce8e6" stroke="{RD}"/>')
p.append(f'<text class="svglbl" x="510" y="68" fill="{RD}">账二 · 计算：一律补齐到整块，没有优惠</text>')
p.append(f'<text class="svgsm" x="510" y="88" fill="#a50e0e">所有向量运算按 <tspan font-family="ui-monospace,monospace">8×128</tspan> 结算 —— 补出来的零照样过一遍</text>')
# 表头
hy=Y0-6
for x,t,a in [(0,'数组形状','start'),(150,'','start'),(320,'内存用的块','start'),(452,'内存里占','start'),(590,'内存浪费','end'),(700,'计算按','start'),(860,'计算浪费','end')]:
    p.append(f'<text class="svgsm" x="{x if a=="start" else x}" y="{hy}" text-anchor="{a}" fill="{GY}">{t}</text>')
p.append(f'<path d="M0 {hy+8} h982" stroke="#dadce0"/>')
for i,(s,note,t,ms,mw,cs,cw,valid) in enumerate(rows):
    y=Y0+i*RH
    bg="#f8f9fa" if i%2 else "#fff"
    p.append(f'<rect x="0" y="{y}" width="982" height="{RH}" fill="{bg}"/>')
    p.append(f'<text class="svgsm" x="0" y="{y+21}" font-family="ui-monospace,monospace" fill="#202124">{s}</text>')
    p.append(f'<text class="svgsm" x="150" y="{y+21}" fill="{GY}">{note}</text>')
    p.append(f'<text class="svgsm" x="320" y="{y+21}" font-family="ui-monospace,monospace" fill="{BL}">{t}×128</text>')
    p.append(f'<text class="svgsm" x="452" y="{y+21}" fill="{BL}">{ms:,} 个位置</text>')
    c1=GR if mw==0 else BL
    p.append(f'<text class="svgnum" x="590" y="{y+21}" text-anchor="end" fill="{c1}" style="font-size:12px">{mw*100:.1f}%</text>')
    p.append(f'<text class="svgsm" x="700" y="{y+21}" fill="{RD}">{cs:,} 个位置</text>')
    c2=GR if cw==0 else RD
    p.append(f'<text class="svgnum" x="860" y="{y+21}" text-anchor="end" fill="{c2}" style="font-size:12px">{cw*100:.1f}%</text>')
    p.append(f'<text class="svgsm" x="982" y="{y+21}" text-anchor="end" fill="{GY}">有效 {valid:,}</text>')
yb=Y0+RH*len(rows)+14
p.append(f'<rect x="0" y="{yb}" width="982" height="62" rx="8" fill="#fce8e6" stroke="{RD}"/>')
p.append(f'<text class="svglbl" x="16" y="{yb+22}" fill="{RD}">⚠️ 官方文档给的极端例子，比上表狠得多</text>')
p.append(f'<text class="svgsm" x="16" y="{yb+41}" fill="#a50e0e">'
         f'「加两个 <tspan font-family="ui-monospace,monospace">1×1</tspan> 的数组，跟加两个 <tspan font-family="ui-monospace,monospace">8×128</tspan> 的数组一样贵。'
         f'加两个 <tspan font-family="ui-monospace,monospace">8×128×1×1</tspan> 的数组，是加两个 <tspan font-family="ui-monospace,monospace">8×128</tspan> 的 <tspan font-weight="700">1,024 倍</tspan>贵」</text>')
p.append(f'<text class="svgsm" x="16" y="{yb+56}" fill="#a50e0e">'
         f'因为 <tspan font-family="ui-monospace,monospace">8×128×1×1</tspan> 会被补成 <tspan font-family="ui-monospace,monospace">8×128×8×128</tspan> —— '
         f'⚠️ <tspan font-weight="700">最后两维出现 1，是 TPU 上最贵的形状错误</tspan>。</text>')
# bf16 打包
yp=yb+74
p.append(f'<rect x="0" y="{yp}" width="982" height="{H-yp-6}" rx="8" fill="#f3e8fd" stroke="{PU}"/>')
p.append(f'<text class="svglbl" x="16" y="{yp+22}" fill="{PU}">那 bf16 呢？——「8×128 变成 16×128」这个直觉，方向对、写法不对</text>')
for i,t in enumerate([
 'bf16 的官方 tiling 是 <tspan font-family="ui-monospace,monospace" font-weight="700">(8,128)(2,1)</tspan> —— 两层。外层还是 8×128，内层的 (2,1) 把<tspan font-weight="700">偶数行和奇数行、同一列</tspan>的两个 bf16 拼成一个 32 位。',
 '为什么按列拼不按行拼？官方给的理由：TPU 原生就是 32 位机器，而<tspan font-weight="700">跨次内维搬数据比跨最内维便宜</tspan>。int8 同理，是 (8,128)(4,1)。',
 '⭐ 这条不是纸上谈兵：JAX 的 ragged paged attention kernel 里，折叠维度前会先检查次内维能否被 packing 整除，<tspan font-weight="700">不能整除就先升成 fp32 再折</tspan> —— 否则会打散那个配对。']):
    p.append(f'<text class="svgsm" x="16" y="{yp+44+i*19}" fill="#5b1a8c">{t}</text>')
W('fig2-5.svg',p+['</svg>'])
print('2-5 ok', rows)
