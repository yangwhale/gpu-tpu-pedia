# -*- coding: utf-8 -*-
import io
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY,YL="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368","#f9ab00"

# ══════════ 图 2-3 · Flash：把中间结果关在片上 ══════════
S=(2**17)**2*2                       # 128K seq, bf16
GiB=2**30
assert S/GiB==32.0, S/GiB
HEADS=128
BUDGET=94.74

P1X,P2X,PW=0,516,466
TOP=92
H=TOP+322
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="朴素注意力把 seq×seq 的中间矩阵写回 HBM，Flash 让它在片上生死，一次都不落地">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   'Flash 到底 flash 在哪 —— 那个 seq × seq 的矩阵，一次都不许落到 HBM</text>',
   '<text class="svgsm" x="0" y="35">这是本节的主例：它同时压中「访存路径」和「片上容量」两件事</text>']
# 顶部 punch
p.append(f'<rect x="0" y="48" width="982" height="32" rx="6" fill="#fce8e6" stroke="{RD}"/>')
p.append(f'<text class="svgsm" x="16" y="68" fill="{RD}">'
         f'先记一个数：<tspan font-weight="700">一个头、128K 序列、bf16 的注意力矩阵 = (2¹⁷)² × 2 B = 2³⁵ B = 恰好 32 GiB</tspan>'
         f' —— 而一个 v7 device 总共只有 {BUDGET} GiB 可分配，装不下 3 个头，模型有 {HEADS} 个。</text>')

def panel(x,title,color,rows,foot,footc):
    p.append(f'<rect x="{x}" y="{TOP}" width="{PW}" height="208" rx="8" fill="#fff" stroke="{color}" stroke-width="1.6"/>')
    p.append(f'<rect x="{x}" y="{TOP}" width="{PW}" height="26" rx="8" fill="{color}"/>')
    p.append(f'<rect x="{x}" y="{TOP+18}" width="{PW}" height="8" fill="{color}"/>')
    p.append(f'<text class="svglbl" x="{x+12}" y="{TOP+18}" fill="#fff">{title}</text>')
    for i,(t,tag,hot) in enumerate(rows):
        y=TOP+40+i*26
        c=RD if hot else GY
        p.append(f'<circle cx="{x+18}" cy="{y+4}" r="3.5" fill="{c}"/>')
        p.append(f'<text class="svgsm" x="{x+30}" y="{y+8}" fill="#202124">{t}</text>')
        if tag: p.append(f'<text class="svgsm" x="{x+PW-12}" y="{y+8}" text-anchor="end" fill="{c}">{tag}</text>')
    p.append(f'<rect x="{x+10}" y="{TOP+166}" width="{PW-20}" height="30" rx="5" fill="{footc}"/>')
    p.append(f'<text class="svgsm" x="{x+22}" y="{TOP+185}" fill="#fff">{foot}</text>')

panel(P1X,'① 朴素写法：中间矩阵在 HBM 里走了一个来回',RD,[
  ('算 Q · Kᵀ，得到 seq × seq 的分数矩阵 S','',0),
  ('把 S 写回 HBM','32 GiB 写',1),
  ('再从 HBM 读回 S 做 softmax','32 GiB 读',1),
  ('把 softmax 结果 P 写回 HBM','32 GiB 写',1),
  ('再读 P 乘 V','32 GiB 读',1)],
  '一个头搬了 4 × 32 GiB，而真正的乘加只有那两次矩阵乘',"#a50e0e")

panel(P2X,'② Flash：分块，中间结果在片上生死',GR,[
  ('把 Q / K / V 按行切成块，一次只取一块','',0),
  ('这一块的 S 只在片上出现','不落地',0),
  ('online softmax：只维护 running max 和 running sum','两个标量',0),
  ('累加进输出块，S 当场丢弃','不落地',0),
  ('换下一块，重复','',0)],
  '整个 seq × seq 矩阵一次都没在 HBM 里出现过',"#0d652d")

# 底部：谁来选块大小
yb=TOP+226
p.append(f'<rect x="0" y="{yb}" width="982" height="82" rx="8" fill="#f1f3f4" stroke="#dadce0"/>')
p.append(f'<text class="svglbl" x="18" y="{yb+22}" fill="#202124">'
         f'③ 那「一块」该多大？—— 两边问的是同一个问题，答案交给不同的人</text>')
p.append(f'<text class="svgsm" x="18" y="{yb+45}" fill="{BL}">'
         f'<tspan font-weight="700">GPU</tspan>：kernel 作者自己定。预算 = shared memory 228 KB / SM，</text>')
p.append(f'<text class="svgsm" x="18" y="{yb+62}" fill="{BL}">'
         f'手写 Triton / CUTLASS，块大小是可调超参 —— 调错了也能跑，只是慢。</text>')
p.append(f'<path d="M496 {yb+32} v40" stroke="#dadce0"/>')
p.append(f'<text class="svgsm" x="516" y="{yb+45}" fill="{GR}">'
         f'<tspan font-weight="700">TPU</tspan>：编译器定。预算 = VMEM 64 MiB，但块的形状<tspan font-weight="700">必须</tspan>对齐 8 × 128，</text>')
p.append(f'<text class="svgsm" x="516" y="{yb+62}" fill="{GR}">'
         f'对不齐就补零 —— 补出来的零照样占 VMEM、照样过 MXU。</text>')
W('fig2-3.svg',p+['</svg>'])

# ══════════ 图 2-4 · 算术强度回到 312 那条线 ══════════
# naive: 每个头，FLOP = 2*2*n^2*d（两次矩阵乘）；HBM 流量 ≈ 4*n^2*2 B（S/P 各一读一写）
n=2**17; d=128
flop=2*(2*n*n*d)
bytes_naive=4*n*n*2
ai_naive=flop/bytes_naive
# flash: 流量 ≈ Q,K,V,O 各读/写一次 = 4*n*d*2 B
bytes_flash=4*n*d*2
ai_flash=flop/bytes_flash
LINE=312.9
H4=412
r=[f'<svg viewBox="0 0 1000 {H4}" width="100%" role="img" aria-label="朴素注意力的算术强度只有 64，Flash 把它拉到 65536，都以 312 这条脊线为参照">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   '回到第 1 节那条 312 的线 —— Flash 换的到底是什么</text>',
   '<text class="svgsm" x="1000" y="16" text-anchor="end" fill="#9aa0a6">这根轴在完整版 L300 的 2.8 节还会回来一次，那时上面会有五个算子 —— 这里先只放两个</text>',
   f'<text class="svgsm" x="0" y="35">同一个头（n = 128K，d = {d}）算两次矩阵乘的 FLOP 一模一样，变的只有搬了多少字节</text>']
# 对数轴
X0,XW=190,700; YA=90
import math
lo,hi=math.log10(16),math.log10(2**17)
def px(v): return X0+XW*(math.log10(v)-lo)/(hi-lo)
r.append(f'<path d="M{X0} {YA+112} h{XW}" stroke="#9aa0a6" stroke-width="1.2"/>')
for t in [16,64,256,1024,4096,16384,65536]:
    x=px(t)
    r.append(f'<path d="M{x} {YA+112} v6" stroke="#9aa0a6"/>')
    r.append(f'<text class="svgsm" x="{x}" y="{YA+128}" text-anchor="middle" fill="{GY}">{t}</text>')
r.append(f'<text class="svgsm" x="{X0+XW/2}" y="{YA+148}" text-anchor="middle" fill="{GY}">算术强度（FLOP / byte，对数轴）</text>')
# 312 脊线
x312=px(LINE)
r.append(f'<path d="M{x312} {YA-6} v{118}" stroke="{YL}" stroke-width="2.4" stroke-dasharray="6 4"/>')
r.append(f'<text class="svglbl" x="{x312}" y="{YA-14}" text-anchor="middle" fill="#7a5000">312 —— 硬件的分水岭</text>')
r.append(f'<text class="svgsm" x="{x312-8}" y="{YA-32}" text-anchor="end" fill="{GY}">左边：带宽说了算</text>')
r.append(f'<text class="svgsm" x="{x312+8}" y="{YA-32}" fill="{GY}">右边：算力说了算</text>')
for i,(nm,v,c) in enumerate([("朴素注意力",ai_naive,RD),("Flash",ai_flash,GR)]):
    y=YA+8+i*44; x=px(v)
    r.append(f'<text class="svglbl" x="{X0-14}" y="{y+12}" text-anchor="end" fill="{c}">{nm}</text>')
    r.append(f'<path d="M{X0} {y+8} H{x}" stroke="{c}" stroke-width="1.4" stroke-dasharray="3 3" opacity="0.5"/>')
    r.append(f'<circle cx="{x}" cy="{y+8}" r="7" fill="{c}"/>')
    r.append(f'<text class="svgnum" x="{x}" y="{y-6}" text-anchor="middle" fill="{c}" style="font-size:12px">{v:,.0f}</text>')
# 轴下两条说明
for i,(c,t) in enumerate([(RD,'朴素：每算 1 FLOP 就得搬 1/64 byte，而硬件只供得起 1/312 —— 多要了近 5 倍，卡在带宽上'),
                          (GR,'Flash：同样的 FLOP，搬的字节少了 1,024 倍 —— 跨到右边，卡在算力上（这才是该卡的地方）')]):
    r.append(f'<rect x="{X0-140}" y="{YA+156+i*22}" width="10" height="10" rx="2" fill="{c}"/>')
    r.append(f'<text class="svgsm" x="{X0-124}" y="{YA+165+i*22}" fill="{c}">{t}</text>')
# 推导链
yd=YA+204
r.append(f'<rect x="0" y="{yd}" width="982" height="{H4-yd-6}" rx="8" fill="#e8f0fe" stroke="{BL}"/>')
r.append(f'<text class="svglbl" x="18" y="{yd+22}" fill="{BL}">这两个数是算出来的，推导链在这儿</text>')
for i,t in enumerate([
  f'两次矩阵乘的 FLOP（两边相同）：2 × (2 · n² · d) = 2 × 2 × (2¹⁷)² × {d} = {flop:,} FLOP',
  f'朴素搬的字节：S 和 P 各写一次读一次 = 4 × n² × 2 B = {bytes_naive:,} B  →  强度 = {ai_naive:,.0f}',
  f'Flash 搬的字节：Q / K / V / O 各过一次 = 4 × n × d × 2 B = {bytes_flash:,} B  →  强度 = {ai_flash:,.0f}',
  f'⚠️ 这是理想上界：真实 Flash 会重复读 K / V，强度到不了 {ai_flash:,.0f}。但结论不变 —— 它是从 312 的左边跨到了右边。']):
    r.append(f'<text class="svgsm" x="18" y="{yd+44+i*18}" fill="#174ea6" font-family="ui-monospace,monospace">{t}</text>')
W('fig2-4.svg',r+['</svg>'])
print(f'2-3 / 2-4 ok  naive={ai_naive:.0f} flash={ai_flash:.0f} S={S/GiB} GiB')
