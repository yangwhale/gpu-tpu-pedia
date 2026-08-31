# -*- coding: utf-8 -*-
import io,math
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY,YL="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368","#f9ab00"

# ══════════ 图 2-6 · 形状约束一路往上传 ══════════
L=[("硬件","向量寄存器 8 × 128（32 位）",
    "8 个 sublane × 128 个 lane —— 这是一切的源头",BL),
   ("内存布局","数组按 (8,128) 分块存；bf16 是 (8,128)(2,1)",
    "XLA 故意让内存形状 = 寄存器形状，搬进来就能用",BL),
   ("kernel 接口","Pallas 的 block 最后两维要么是 8 / 128 的倍数，要么等于数组本身那一维",
    "⚠️ 后半句是官方留的出口：整维取满也合法。除此之外，对不齐就编译报错",PU),
   ("矩阵单元","v7 的 MXU 是 256 × 256，收缩维要 256 的倍数才吃满",
    "官方原话：head_dim 是 256 的倍数才能吃满 MXU",OR),
   ("模型超参","head_dim / d_ff / 词表 / 专家数 —— 你写在配置文件里的那些数",
    "⭐ 一个寄存器的形状，最后管到了你的模型配置",RD)]
BH,GAP,Y0=54,20,66
H=Y0+len(L)*(BH+GAP)+140
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="八乘一百二十八这个形状约束从寄存器一路传到模型超参">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   '这个约束不会停在硬件层 —— 它一路往上传，最后管到你的<tspan font-weight="700">模型配置文件</tspan></text>',
   '<text class="svgsm" x="0" y="35">每一层都只是「照着下一层的形状办事」，但五层叠起来，就变成了对模型架构的硬约束</text>']
for i,(tag,main,sub,c) in enumerate(L):
    y=Y0+i*(BH+GAP)
    p.append(f'<rect x="0" y="{y}" width="982" height="{BH}" rx="8" fill="#fff" stroke="{c}" stroke-width="1.6"/>')
    p.append(f'<rect x="0" y="{y}" width="118" height="{BH}" rx="8" fill="{c}"/>')
    p.append(f'<rect x="108" y="{y}" width="10" height="{BH}" fill="{c}"/>')
    p.append(f'<text class="svglbl" x="59" y="{y+24}" text-anchor="middle" fill="#fff">{tag}</text>')
    p.append(f'<text class="svgsm" x="59" y="{y+41}" text-anchor="middle" fill="#ffffffcc">第 {i+1} 层</text>')
    p.append(f'<text class="svglbl" x="134" y="{y+24}" fill="#202124">{main}</text>')
    p.append(f'<text class="svgsm" x="134" y="{y+42}" fill="{GY}">{sub}</text>')
    if i<len(L)-1:
        p.append(f'<path d="M491 {y+BH} v{GAP-6}" stroke="{GY}" stroke-width="1.6" marker-end="url(#ar)"/>')
p.insert(1,f'<defs><marker id="ar" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="{GY}"/></marker></defs>')
yb=Y0+len(L)*(BH+GAP)+2
p.append(f'<rect x="0" y="{yb}" width="982" height="128" rx="8" fill="#fef7e0" stroke="{YL}"/>')
p.append(f'<text class="svglbl" x="16" y="{yb+22}" fill="#7a5000">⭐ 一个当场就能算的例子：head_dim = 128 撞上 256×256 的 MXU</text>')
for i,t in enumerate([
 '官方说法：head_dim 是 <tspan font-weight="700">256 的倍数</tspan>才能吃满 MXU；head_dim 是 <tspan font-weight="700">128 或 64</tspan> 时，flash attention 里的 QK 那一步会有 <tspan font-weight="700">50% 甚至更多</tspan>的 MXU 利用率损失。',
 '⚠️ 但注意：<tspan font-weight="700">不是浪费 3/4，是浪费 1/2</tspan>。很多人第一反应是「只用了角上 128×128 的一块」，那样才是 3/4。',
 '真相是：<tspan font-weight="700">head_dim 每次只占矩阵乘的一条边</tspan>。算 Q·Kᵀ 时它是收缩维（128 填进 256，一半）；另外两条边是 block_q 和 block_kv ——',
 '那是 kernel 作者自己挑的块大小，代码里强制必须是 128 的倍数，实际会调到 512 以上。<tspan font-weight="700">两条边里只有一条被钉死，所以是一半。</tspan>',
 '算「注意力 × V」时正好反过来：收缩维变成 kv 块大小（很大），head_dim 变成输出边 —— 这就是为什么官方那句话<tspan font-weight="700">只点名 QK 那一步</tspan>。']):
    p.append(f'<text class="svgsm" x="16" y="{yb+44+i*17}" fill="#7a5000">{t}</text>')
W('fig2-6.svg',p+['</svg>'])

# ══════════ 图 2-7 · Splash 的第二招：整块跳过 ══════════
N=10           # 画 10x10 的块网格
CELL=25; GX,GY_=580,96
# 因果 mask 下三种块的占比（严格下三角为全 1，对角为半掩，上三角为全 0）
full=N*(N-1)//2; part=N; zero=N*(N-1)//2
tot=N*N
# SMEM 账：seq 128K, bq 512, bkv 1024
SEQ=128*1024; BQ=512; BKV=1024
nq=SEQ//BQ; nkv=SEQ//BKV; ent=nq*nkv*3
smem=1024*1024
H7=GY_+N*CELL+134
q=[f'<svg viewBox="0 0 1000 {H7}" width="100%" role="img" aria-label="Splash 的第二招：因果 mask 下把全零的块整块跳过，元信息存在 SMEM 里还要缩类型">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   'Splash 比普通 Flash 多的那一层：<tspan font-weight="700">整块跳过</tspan>（这才是名字里 sparse 的意思）</text>',
   '<text class="svgsm" x="0" y="35">来源：JAX 公开源码 splash_attention_mask_info.py 里 MaskInfo 的字段说明</text>',
   f'<text class="svglbl" x="0" y="{GY_-14}" fill="{GY}">q 块 × kv 块的网格，跑之前先给每一格打分</text>']
for r in range(N):
    for c in range(N):
        if c<r:   f,st,lab=("#e6f4ea",GR,"2")
        elif c==r:f,st,lab=("#fef7e0",YL,"1")
        else:     f,st,lab=("#f1f3f4","#dadce0","0")
        x,y=GX+c*CELL,GY_+r*CELL
        q.append(f'<rect x="{x}" y="{y}" width="{CELL-2}" height="{CELL-2}" rx="2" fill="{f}" stroke="{st}" stroke-width="0.8"/>')
        q.append(f'<text x="{x+(CELL-2)/2}" y="{y+16}" text-anchor="middle" class="svgsm" fill="{GY if lab=="0" else ("#7a5000" if lab=="1" else "#0d652d")}" style="font-size:9px">{lab}</text>')
q.append(f'<text class="svgsm" x="{GX-10}" y="{GY_+14}" text-anchor="end" fill="{GY}">q 块 →</text>')
q.append(f'<text class="svgsm" x="{GX}" y="{GY_-4}" fill="{GY}">kv 块 →</text>')
LEG=[("2","全是 1（完全没遮挡）",GR,"#e6f4ea",f"{full} 格 · {full/tot*100:.0f}%","照常算，但<tspan font-weight='700'>跳过贴 mask 那一步</tspan>"),
     ("1","半掩半开",YL,"#fef7e0",f"{part} 格 · {part/tot*100:.0f}%","取一块真实的布尔 mask 出来贴上"),
     ("0","全是 0（整块被遮掉）",GY,"#f1f3f4",f"{zero} 格 · {zero/tot*100:.0f}%","<tspan font-weight='700'>根本不去访问</tspan> —— 不搬 K、不搬 V、不算矩阵乘")]
for i,(lab,nm,c,bg,cnt,act) in enumerate(LEG):
    y=GY_+8+i*30
    q.append(f'<rect x="0" y="{y}" width="20" height="20" rx="3" fill="{bg}" stroke="{c}"/>')
    q.append(f'<text x="10" y="{y+14}" text-anchor="middle" class="svgsm" fill="{c}" style="font-size:10px">{lab}</text>')
    q.append(f'<text class="svglbl" x="28" y="{y+9}" fill="{c}">{nm}　<tspan class="svgsm" fill="{GY}">{cnt}</tspan></text>')
    q.append(f'<text class="svgsm" x="28" y="{y+24}" fill="{GY}">{act}</text>')
q.append(f'<text class="svgsm" x="0" y="{GY_+112}" fill="{RD}">序列越长，0 的占比越逼近 <tspan font-weight="700">50%</tspan> ——</text>')
q.append(f'<text class="svgsm" x="0" y="{GY_+129}" fill="{RD}">因果注意力有<tspan font-weight="700">一半的计算根本不存在</tspan>，</text>')
q.append(f'<text class="svgsm" x="0" y="{GY_+146}" fill="{RD}">前提是你得有人告诉 kernel 「哪些能跳」。</text>')
yb=GY_+N*CELL+16
q.append(f'<rect x="0" y="{yb}" width="982" height="{H7-yb-6}" rx="8" fill="#e8f0fe" stroke="{BL}"/>')
q.append(f'<text class="svglbl" x="16" y="{yb+22}" fill="{BL}">这张表存在哪？—— 存在 SMEM 里，而 SMEM 只有 1 MiB（第 1 节那张图上标过）</text>')
for i,t in enumerate([
 f'算一笔：序列 {SEQ//1024}K、block_q {BQ}、block_kv {BKV} → {nq} × {nkv} 的网格，三个数组（block_mask / data_next / mask_next）共 <tspan font-weight="700">{ent:,}</tspan> 项。',
 f'用 int32 存要 <tspan font-weight="700">{ent*4/1024:.0f} KiB</tspan>，占掉 SMEM 的 <tspan font-weight="700">{ent*4/smem*100:.0f}%</tspan>；缩成 int16 只要 {ent*2/1024:.0f} KiB，int8 只要 {ent/1024:.0f} KiB。',
 '所以源码里专门写了一段「把整数类型往小了缩」的逻辑，注释直说 SMEM 是稀缺资源。⭐ <tspan font-weight="700">跳过整块省下的时间，是拿 SMEM 的容量换来的</tspan> —— 天下没有白跳的块。',
 '还有个 data_next 数组，记「下一个该预取哪个 kv 块」—— 因为跳过之后块号不连续了，得有人告诉 DMA 去哪儿取。']):
    q.append(f'<text class="svgsm" x="16" y="{yb+44+i*18}" fill="#174ea6">{t}</text>')
W('fig2-7.svg',q+['</svg>'])
print(f'2-6 / 2-7 ok  网格 {full}/{part}/{zero}  SMEM {ent:,} 项 int32={ent*4/1024:.0f}KiB ({ent*4/smem*100:.0f}%)')
