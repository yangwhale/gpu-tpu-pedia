# -*- coding: utf-8 -*-
import io,math
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY,YL="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368","#f9ab00"
LINE=312.9
# ── 全部当场算 ──
n=2**17; d=128                                   # 一个头，128K 序列
flop_at=2*(2*n*n*d)
ai_naive=flop_at/(4*n*n*2)                       # S、P 各读写一次
ai_flash=flop_at/(4*n*d*2)                       # Q/K/V/O 各过一次（理想上界）
M,K,N=8192,8192,28672                            # 一层 MLP 的一个矩阵乘，bf16
ai_mlp=(2*M*K*N)/(2*(M*K+K*N+M*N))
ai_soft=4/4                                      # 每元素约 4 次运算，进出各 2 B
PTS=[("embedding 查表",None,PU,"纯搬运，一次乘加都没有 —— 算术强度<tspan font-weight='700'>就是 0</tspan>，画不到轴上"),
     ("出口 softmax",ai_soft,OR,"逐元素：每个元素约 4 次运算，进 2 B 出 2 B"),
     ("朴素注意力",ai_naive,RD,"中间矩阵在 HBM 里走了个来回"),
     ("Flash 注意力",ai_flash,GR,"理想上界；真实值更低，但仍在右边"),
     ("Dense MLP 矩阵乘",ai_mlp,BL,f"M={M:,} K={K:,} N={N:,}，bf16")]
X0,XW,YA=228,660,112
lo,hi=math.log10(0.5),math.log10(2**17)
def px(v): return X0+XW*(math.log10(v)-lo)/(hi-lo)
H=YA+len(PTS)*38+186
# 与图 2-4 是同一根轴、同样的朴素/Flash 两个点 —— 那次是为了讲「Flash 换了一边」，
# 这次是为了把它当判据交出去。不写清关系，第二次出现就是一次重复。
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="本节四个算子回到 312 这根轴上：embedding 和 softmax 在最左，朴素注意力 64，Flash 和 MLP 在右边">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   '本节收尾：把讲过的算子<tspan font-weight="700">全部放回同一根轴</tspan> —— 一眼看出谁该救、该怎么救</text>',
   '<text class="svgsm" x="0" y="35">横轴是算术强度（FLOP / byte，对数）。312 这条线两边的<tspan font-weight="700">优化动作完全不同</tspan>。</text>',
   '<text class="svgsm" x="1000" y="16" text-anchor="end" fill="#9aa0a6">这根轴第 2.3 节出现过，那次只有朴素和 Flash 两个点 —— 这次是完整的五个</text>']
ax=YA+len(PTS)*38+4
for t in [1,4,16,64,256,1024,4096,16384,65536]:
    x=px(t)
    p.append(f'<path d="M{x} {ax} v6" stroke="#9aa0a6"/>')
    p.append(f'<text class="svgsm" x="{x}" y="{ax+18}" text-anchor="middle" fill="{GY}">{t:,}</text>')
p.append(f'<path d="M{X0-30} {ax} h{XW+40}" stroke="#9aa0a6" stroke-width="1.2"/>')
x312=px(LINE)
p.append(f'<rect x="{X0-30}" y="{YA-24}" width="{x312-X0+30}" height="{ax-YA+24}" fill="#fce8e6" opacity="0.35"/>')
p.append(f'<rect x="{x312}" y="{YA-24}" width="{X0+XW+10-x312}" height="{ax-YA+24}" fill="#e6f4ea" opacity="0.35"/>')
p.append(f'<path d="M{x312} {YA-24} v{ax-YA+24}" stroke="{YL}" stroke-width="2.4" stroke-dasharray="6 4"/>')
p.append(f'<text class="svglbl" x="{x312}" y="{YA-46}" text-anchor="middle" fill="#7a5000">312 —— 硬件的分水岭</text>')
p.append(f'<text class="svgsm" x="{x312-14}" y="{YA-28}" text-anchor="end" fill="{RD}">← 带宽受限：减少搬运</text>')
p.append(f'<text class="svgsm" x="{x312+14}" y="{YA-28}" fill="{GR}">算力受限：提高利用率 →</text>')
for i,(nm,v,c,note) in enumerate(PTS):
    y=YA+i*38
    p.append(f'<text class="svglbl" x="{X0-44}" y="{y+12}" text-anchor="end" fill="{c}">{nm}</text>')
    if v is None:
        p.append(f'<path d="M{X0-24} {y+8} h-14" stroke="{c}" stroke-width="2.4" marker-end="url(#a2)"/>')
        p.append(f'<circle cx="{X0-18}" cy="{y+8}" r="6" fill="{c}"/>')
        p.append(f'<text class="svgnum" x="{X0-18}" y="{y-6}" text-anchor="middle" fill="{c}" style="font-size:12px">0</text>')
        p.append(f'<text class="svgsm" x="{X0}" y="{y+12}" fill="{GY}">{note}</text>')
    else:
        x=px(v)
        p.append(f'<path d="M{X0-24} {y+8} H{x}" stroke="{c}" stroke-width="1.3" stroke-dasharray="3 3" opacity="0.45"/>')
        p.append(f'<circle cx="{x}" cy="{y+8}" r="6" fill="{c}"/>')
        p.append(f'<text class="svgnum" x="{x}" y="{y-6}" text-anchor="middle" fill="{c}" style="font-size:12px">{v:,.0f}</text>')
        anc='end' if x>620 else 'start'
        p.append(f'<text class="svgsm" x="{x+(-14 if anc=="end" else 14)}" y="{y+12}" text-anchor="{anc}" fill="{GY}">{note}</text>')
p.insert(1,f'<defs><marker id="a2" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M8 0 L0 4 L8 8 z" fill="{PU}"/></marker></defs>')
yb=ax+34
p.append(f'<rect x="0" y="{yb}" width="486" height="112" rx="8" fill="#fce8e6" stroke="{RD}"/>')
p.append(f'<text class="svglbl" x="16" y="{yb+22}" fill="{RD}">左边这三个：买再强的算力都没用</text>')
for i,t in enumerate(['· embedding：查表没有乘加，<tspan font-weight="700">强度是 0</tspan>。TPU 上还多一层麻烦 ——',
                      '　数组按 8×128 分块存，所以<tspan font-weight="700">「一行」在内存里并不连续</tspan>。',
                      '· 出口 softmax：逐元素，而 VPU 峰值比 MXU 低两个数量级。',
                      '· 朴素注意力：唯一能救的是<tspan font-weight="700">别把中间矩阵写回去</tspan> —— 这就是 Flash。']):
    p.append(f'<text class="svgsm" x="16" y="{yb+42+i*17}" fill="#a50e0e">{t}</text>')
p.append(f'<rect x="496" y="{yb}" width="486" height="112" rx="8" fill="#e6f4ea" stroke="{GR}"/>')
p.append(f'<text class="svglbl" x="512" y="{yb+22}" fill="{GR}">右边这两个：该操心的是利用率</text>')
for i,t in enumerate(['· 已经在算力这一侧了，减少搬运<tspan font-weight="700">不再有收益</tspan>。',
                      '· 该做的是对齐形状、避免补零、换更低的精度。',
                      '· head_dim = 128 撞 256×256 的 MXU，就属于这一类 ——',
                      '　强度够高，但<tspan font-weight="700">一半的乘加单元在算零</tspan>。']):
    p.append(f'<text class="svgsm" x="512" y="{yb+42+i*17}" fill="#0d652d">{t}</text>')
p.append(f'<text class="svgsm" x="0" y="{yb+130}" fill="{GY}">'
         f'⚠️ MoE 没画在这根轴上 —— 它的位置<tspan font-weight="700">取决于补了多少零</tspan>，而补多少零是运行时数据决定的。这正是第 3 节要讲的事。</text>')
W('fig2-8.svg',p+['</svg>'])
print(f'2-8 ok  soft={ai_soft} naive={ai_naive:.0f} flash={ai_flash:.0f} mlp={ai_mlp:.0f}')
