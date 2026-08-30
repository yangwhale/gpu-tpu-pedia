# -*- coding: utf-8 -*-
import io
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY,YL="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368","#f9ab00"

# ══════════ 图 2-1 · 一次访存，两条路 ══════════
GPU=[("一个 warp 的 32 个线程各给出一个地址","",0),
     ("这些地址能合并成几条 cache line？","◆ 决策点 · coalescing",1),
     ("落在同一个 bank 上吗？","◆ 决策点 · bank conflict",1),
     ("L1 命中吗？（与 shared 合计 256 KB／SM）","◆ 决策点 · 命中 / 未命中",1),
     ("L2 命中吗？（126 MB，全 GPU 共享）","◆ 决策点 · 命中 / 未命中",1),
     ("换出谁？","◆ 决策点 · 替换策略",1),
     ("HBM","",0)]
TPU=[("编译期：XLA 决定每个数组切成 8 × 128 的块","☑ 已定死",2),
     ("编译期：决定什么时候搬、搬多大一块","☑ 已定死",2),
     ("编译期：算准 VMEM 装不装得下","☑ 已定死",2),
     ("运行时：DMA 整块 HBM → VMEM（64 MiB）","",0),
     ("运行时：向量单元直接取，形状必须已经对","",0),
     ("（没有这一层）","CMEM = 0",3),
     ("HBM","",0)]
RH,TOP=44,86
H=TOP+RH*7+150
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="同一次访存在 GPU 与 TPU 上经过的路径对比：GPU 有五个运行时决策点，TPU 一个都没有">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   '同一次访存，两条路 —— 数一数路上有几个「运行时决策点」</text>',
   '<text class="svgsm" x="0" y="35">这是第 0 节那两条出身第一次变成具体机制</text>',
   f'<text class="svglbl" x="0" y="{TOP-12}" fill="{BL}">GPU：运行时适应</text>',
   f'<text class="svgsm" x="150" y="{TOP-12}">硬件替你决定，你写得随意也能跑</text>',
   f'<text class="svglbl" x="512" y="{TOP-12}" fill="{GR}">TPU：编译期安排</text>',
   f'<text class="svgsm" x="668" y="{TOP-12}">运行时零决策，全部提前算好</text>']
def col(x,w,rows,base):
    for i,(t,tag,kind) in enumerate(rows):
        y=TOP+i*RH
        col_,fill=(base,'#f8f9fa')
        if kind==1: col_,fill=(RD,'#fce8e6')
        if kind==2: col_,fill=(GR,'#e6f4ea')
        if kind==3: col_,fill=(GY,'#fff')
        dash=' stroke-dasharray="5 4"' if kind==3 else ''
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="32" rx="6" fill="{fill}" stroke="{col_}"{dash}/>')
        p.append(f'<text class="svgsm" x="{x+12}" y="{y+20}" fill="{"#5f6368" if kind==3 else "#202124"}">{t}</text>')
        if tag: p.append(f'<text class="svgsm" x="{x+w-12}" y="{y+20}" text-anchor="end" fill="{col_}">{tag}</text>')
        if i<len(rows)-1:
            p.append(f'<path d="M{x+w/2} {y+32} v{RH-32}" stroke="#dadce0" stroke-width="1.4"/>')
col(0,470,GPU,BL); col(512,470,TPU,GR)
yb=TOP+RH*7+4
p.append(f'<rect x="0" y="{yb}" width="470" height="40" rx="6" fill="{RD}"/>')
p.append(f'<text class="svglbl" x="235" y="{yb+18}" text-anchor="middle" fill="#fff">路上 5 个运行时决策点</text>')
p.append(f'<text class="svgsm" x="235" y="{yb+33}" text-anchor="middle" fill="#ffffffcc">你能做的是「让硬件更容易猜对」：访问连续、避开 bank、手工 tiling</text>')
p.append(f'<rect x="512" y="{yb}" width="470" height="40" rx="6" fill="{GR}"/>')
p.append(f'<text class="svglbl" x="747" y="{yb+18}" text-anchor="middle" fill="#fff">路上 0 个运行时决策点</text>')
p.append(f'<text class="svgsm" x="747" y="{yb+33}" text-anchor="middle" fill="#ffffffcc">没有 cache，就没有命中、未命中、替换这些概念 —— 也没有补救机会</text>')
p.append(f'<rect x="0" y="{yb+52}" width="982" height="62" rx="8" fill="#fef7e0" stroke="{YL}"/>')
p.append(f'<text class="svglbl" x="18" y="{yb+74}" fill="#7a5000">⭐ 这不是「谁更聪明」，是两次不同的赌注</text>')
p.append(f'<text class="svgsm" x="18" y="{yb+93}" fill="#7a5000">'
         f'GPU 赌「运行时能适应」—— 赌赢了什么代码都能跑，赌输了 cache 一直未命中，而你只能反复试。</text>')
p.append(f'<text class="svgsm" x="18" y="{yb+110}" fill="#7a5000">'
         f'TPU 赌「编译期能算准」—— 赌赢了一个决策周期都不浪费，赌输了运行时无处补救，只能回去改形状。</text>')
W('fig2-1.svg',p+['</svg>'])

# ══════════ 图 2-2 · lane / sublane 是寄存器的形状 ══════════
CW,CH,NC=44,26,10        # 只画 10 列，后面省略号
X0,Y0=234,102
H2=Y0+8*CH+206
q=[f'<svg viewBox="0 0 1000 {H2}" width="100%" role="img" aria-label="向量寄存器是 8 个 sublane 乘 128 个 lane，sublane 比 lane 大 128 倍">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   'lane 和 sublane 是<tspan font-weight="700">寄存器</tspan>的形状 —— 不是内存的概念，也不是「大小」的概念</text>',
   '<text class="svgsm" x="0" y="35">来源：JAX 公开源码中 v7 那一支的 num_lanes = 128 / num_sublanes = 8</text>',
   f'<text class="svglbl" x="0" y="{Y0-14}" fill="{BL}">一个向量寄存器</text>']
# 第 0 行整行 = 一个 sublane（绿底）；其中第 0 格 = 一个 lane（红底）
q.append(f'<rect x="{X0-4}" y="{Y0-4}" width="{NC*CW+30}" height="{CH+6}" rx="4" fill="#e6f4ea" stroke="{GR}" stroke-width="1.6"/>')
for r in range(8):
    for c in range(NC):
        f = "#fff" if r else "#e6f4ea"
        st = BL
        if r==0 and c==0: f,st = RD,RD
        q.append(f'<rect x="{X0+c*CW}" y="{Y0+r*CH}" width="{CW-2}" height="{CH-2}" rx="2" '
                 f'fill="{f}" stroke="{st}" stroke-width="0.7"/>')
    q.append(f'<text class="svgsm" x="{X0+NC*CW+14}" y="{Y0+r*CH+17}" fill="{GR if r==0 else GY}">…</text>')
q.append(f'<text class="svgsm" x="{X0+NC*CW+40}" y="{Y0+CH-8}" fill="{GR}">共 128 列</text>')
# lane 标注
q.append(f'<path d="M{X0+CW/2} {Y0-14} v10" stroke="{RD}" stroke-width="1.6"/>')
q.append(f'<text class="svglbl" x="{X0+CW/2}" y="{Y0-34}" text-anchor="middle" fill="{RD}">1 个 lane</text>')
q.append(f'<text class="svgsm" x="{X0+CW/2}" y="{Y0-20}" text-anchor="middle" fill="{RD}">= 4 B（红色这一格）</text>')
# sublane 标注
q.append(f'<text class="svglbl" x="{X0-20}" y="{Y0+8}" text-anchor="end" fill="{GR}">1 个 sublane = 绿色这一整行</text>')
q.append(f'<text class="svgsm" x="{X0-20}" y="{Y0+23}" text-anchor="end" fill="{GR}">= 128 个 lane = 512 B</text>')
q.append(f'<text class="svgsm" x="{X0-18}" y="{Y0+8*CH-4}" text-anchor="end">共 8 行</text>')
yq=Y0+8*CH+16
q.append(f'<text class="svgnum" x="{X0}" y="{yq+4}" fill="{BL}">整个寄存器 = 8 × 128 × 32 bit = 4,096 B = 4 KiB</text>')
q.append(f'<rect x="0" y="{yq+18}" width="982" height="56" rx="8" fill="#fce8e6" stroke="{RD}"/>')
q.append(f'<text class="svglbl" x="18" y="{yq+40}" fill="{RD}">'
         f'⚠️ 「sub」是「第二维」的意思，不是「更小」—— 一个 sublane 是 512 B，比一个 lane 大 128 倍</text>')
q.append(f'<text class="svgsm" x="18" y="{yq+59}" fill="{RD}">'
         f'几乎所有人第一次都会理解反。记法：<tspan font-family="ui-monospace,monospace">最内维 → lane，次内维 → sublane</tspan>，跟大小无关。</text>')
q.append(f'<rect x="0" y="{yq+84}" width="982" height="88" rx="8" fill="#e6f4ea" stroke="{GR}"/>')
q.append(f'<text class="svglbl" x="18" y="{yq+106}" fill="{GR}">'
         f'⭐ 然后 XLA 故意把内存布局做成同一个形状</text>')
for i,t in enumerate([
  '数组 […, C, D] 落到内存里：最内维 D → lane（128 一组），次内维 C → sublane（8 一组）。',
  '于是一个 [A, B, C, D] 的数组，在内存里是 A × B × ⌈C/8⌉ × ⌈D/128⌉ 个 4 KiB 的块。',
  '直接后果：⚠️ 「一行」在内存里并不连续 —— 它被切碎在一排块里。下一片讲这件事要付多少钱。']):
    q.append(f'<text class="svgsm" x="18" y="{yq+125+i*17}" fill="#0d652d">{t}</text>')
W('fig2-2.svg',q+['</svg>'])
print('2-1 / 2-2 ok')
