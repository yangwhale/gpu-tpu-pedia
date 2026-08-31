# -*- coding: utf-8 -*-
import io
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368"

# ══════════ 图 2-5 · 三个「浪费」归三个不同的部件 ══════════
PW,GAP=318,23
PT=64                      # 面板顶
HDR=46                     # 面板头高
SCH_T=PT+HDR+62            # 示意图带顶
SCH_H=116                  # 示意图带高
NUM_T=SCH_T+SCH_H+14       # 大数字带顶
NUM_H=52       # 44 → 52：数字下面多了一行「硬度」徽标
EX_T=NUM_T+NUM_H+12        # 例子框顶
EX_H=78
PB=EX_T+EX_H               # 面板底
BAND=PB+22                 # 底部横条
H=BAND+108

p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" '
   f'aria-label="三个浪费别搞混：内存布局、VPU 向量寄存器、MXU 脉动阵列，归三个不同的部件">',
   '<text class="svglbl" x="0" y="17" fill="#202124" style="font-size:14px">'
   '三个「浪费」<tspan font-weight="700">别搞混</tspan> —— 它们归三个不同的部件，触发条件也完全不同</text>',
   '<text class="svgsm" x="0" y="37">'
   '来源：OpenXLA「Tiled layout」（内存）｜ JAX Pallas TPU 官方文档（VPU）｜ Google Ironwood 性能优化文档（MXU）</text>',
   f'<text class="svgsm" x="0" y="55" fill="{RD}">'
   f'⚠️ 最常见的串台：把 VPU 那笔算到 <tspan font-family="ui-monospace,monospace">head_dim</tspan> 头上。'
   f'<tspan font-family="ui-monospace,monospace">head_dim</tspan> 只影响右边一格</text>']

PAN=[
 dict(x=0,   c=BL, tag="① 内存布局",  own="吃的是 VMEM / HBM 容量",
      rule=["数组按块存。默认 8×128，","但次内维小的时候编译器会缩小块"]),
 dict(x=PW+GAP, c=PU, tag="② VPU（向量单元）", own="吃的是逐元素算力",
      rule=["一条指令吃掉一整个 vreg（8×128），","不看你到底装了几个有效值"]),
 dict(x=2*(PW+GAP), c=RD, tag="③ MXU（矩阵单元）", own="吃的是矩阵乘算力",
      rule=["脉动阵列 256×256。收缩边不到 256，","阵列就有一部分没被喂到"]),
]
for d in PAN:
    x,c=d["x"],d["c"]
    p.append(f'<rect x="{x}" y="{PT}" width="{PW}" height="{PB-PT}" rx="9" fill="#fff" stroke="{c}" stroke-width="1.5"/>')
    p.append(f'<path d="M{x} {PT+9} a9 9 0 0 1 9 -9 h{PW-18} a9 9 0 0 1 9 9 v{HDR-9} h-{PW} z" fill="{c}"/>')
    p.append(f'<text class="svglbl" x="{x+14}" y="{PT+21}" fill="#fff" style="font-size:13.5px">{d["tag"]}</text>')
    p.append(f'<text class="svgsm" x="{x+14}" y="{PT+38}" fill="#ffffffd0">{d["own"]}</text>')
    for i,t in enumerate(d["rule"]):
        p.append(f'<text class="svgsm" x="{x+14}" y="{PT+HDR+21+i*17}" fill="#202124">{t}</text>')

# ── 面板 ① 示意：8 行的块，只用 1 行 → 编译器只分配 2 行
x=PAN[0]["x"]; CW,CH,NC=10,10,10
gx,gy=x+16,SCH_T+6
for r in range(8):
    for cidx in range(NC):
        if r==0:   f,st,dash=RD,RD,''
        elif r==1: f,st,dash="#e8f0fe",BL,''
        else:      f,st,dash="#fff","#dadce0",' stroke-dasharray="2 2"'
        p.append(f'<rect x="{gx+cidx*CW}" y="{gy+r*CH}" width="{CW-1}" height="{CH-1}" fill="{f}" stroke="{st}"{dash}/>')
p.append(f'<path d="M{gx-7} {gy} v{2*CH-1}" stroke="{BL}" stroke-width="2"/>')
p.append(f'<text class="svgsm" x="{gx+NC*CW+10}" y="{gy+9}" fill="{RD}">1 行是真数据</text>')
p.append(f'<text class="svgsm" x="{gx+NC*CW+10}" y="{gy+25}" fill="{BL}">1 行是补的</text>')
p.append(f'<text class="svgsm" x="{gx+NC*CW+10}" y="{gy+46}" fill="{GY}">剩下 6 行</text>')
p.append(f'<text class="svgsm" x="{gx+NC*CW+10}" y="{gy+62}" fill="{GY}">编译器不分配</text>')
p.append(f'<text class="svgsm" x="{gx}" y="{gy+8*CH+16}" fill="#202124">'
         f'次内维 1–2 用 <tspan font-family="ui-monospace,monospace" font-weight="700">2×128</tspan>，'
         f'3–4 用 <tspan font-family="ui-monospace,monospace">4×128</tspan></text>')

# ── 面板 ② 示意：(bq, 128) 的 scratch，每行 1 个真值 + 127 份副本
x=PAN[1]["x"]; CW,CH,NR=25,11,8   # NR 必须 = 8，见下方那行「有效的只有 8 个」
gx,gy=x+16,SCH_T+8
for r in range(NR):
    p.append(f'<rect x="{gx}" y="{gy+r*CH}" width="{CW-1}" height="{CH-1}" fill="{RD}" stroke="{RD}"/>')
    p.append(f'<rect x="{gx+CW}" y="{gy+r*CH}" width="{CW*7-1}" height="{CH-1}" fill="#f3e8fd" stroke="{PU}"/>')
    # 行高压到 11px 后每行都写字会糊成一片，只在中间那行标一次
    if r==NR//2:
        p.append(f'<text class="svgsm" x="{gx+CW*4.5}" y="{gy+r*CH+8}" text-anchor="middle" '
                 f'fill="{PU}" style="font-size:8.5px">同一个数 × 127</text>')
p.append(f'<text class="svgsm" x="{gx-6}" y="{gy-6}" fill="{RD}" style="font-size:10.5px">lane 0</text>')
p.append(f'<text class="svgsm" x="{gx+CW*8}" y="{gy-6}" text-anchor="end" fill="{PU}" style="font-size:10.5px">lane 127</text>')
p.append(f'<text class="svgsm" x="{gx+CW*8+8}" y="{gy+NR*CH//2}" fill="{GY}" style="font-size:9px">8 个 sublane</text>')
p.append(f'<text class="svgsm" x="{gx}" y="{gy+NR*CH+18}" fill="#202124">'
         f'一整条指令算 <tspan font-weight="700">8×128</tspan> 个位置，其中有效的只有 <tspan font-weight="700">8</tspan> 个</text>')

# ── 面板 ③ 示意：256×256 阵列，收缩边只喂到 128
x=PAN[2]["x"]; S=104
gx,gy=x+22,SCH_T+2
p.append(f'<rect x="{gx}" y="{gy}" width="{S}" height="{S}" fill="#fff" stroke="{GY}"/>')
p.append(f'<rect x="{gx}" y="{gy}" width="{S//2}" height="{S}" fill="#fce8e6" stroke="{RD}" stroke-width="1.5"/>')
p.append(f'<clipPath id="mxuh"><rect x="{gx+S//2}" y="{gy}" width="{S//2}" height="{S}"/></clipPath>')
p.append('<g clip-path="url(#mxuh)">')
for k in range(0,22):
    p.append(f'<path d="M{gx+S//2+k*5} {gy} l-{S} {S}" stroke="#dadce0" stroke-width="0.8"/>')
p.append('</g>')
p.append(f'<rect x="{gx+S//2}" y="{gy}" width="{S//2}" height="{S}" fill="none" stroke="{GY}" stroke-dasharray="3 2"/>')
p.append(f'<text class="svgsm" x="{gx+S//4}" y="{gy+S//2}" text-anchor="middle" fill="{RD}" style="font-size:10.5px">喂到了</text>')
p.append(f'<text class="svgsm" x="{gx+S*3//4}" y="{gy+S//2}" text-anchor="middle" fill="{GY}" style="font-size:10.5px">空转</text>')
p.append(f'<text class="svgsm" x="{gx+S//4}" y="{gy-6}" text-anchor="middle" fill="{RD}" style="font-size:10.5px">128</text>')
p.append(f'<text class="svgsm" x="{gx+S*3//4}" y="{gy-6}" text-anchor="middle" fill="{GY}" style="font-size:10.5px">128</text>')
p.append(f'<path d="M{gx} {gy+S+7} h{S}" stroke="{GY}"/>')
p.append(f'<text class="svgsm" x="{gx+S+10}" y="{gy+S//2-6}" fill="#202124" style="font-size:11px">收缩边</text>')
p.append(f'<text class="svgsm" x="{gx+S+10}" y="{gy+S//2+10}" font-family="ui-monospace,monospace" fill="{RD}" style="font-size:11px">head_dim</text>')
p.append(f'<text class="svgsm" x="{gx+S+10}" y="{gy+S//2+26}" font-family="ui-monospace,monospace" fill="{RD}" style="font-size:11px">= 128</text>')
p.append(f'<text class="svgsm" x="{gx-6}" y="{gy+S+22}" fill="#202124">Q·Kᵀ 只喂满阵列的一半</text>')

# ── 大数字带
NUMS=[("50%","bf16 的 [1,128]：分到 2×128",BL,"推导"),
      ("1,024 倍","补成 8×128×8×128",PU,"官方举的极端例子"),
      ("≥ 50%","官方原话「50% or more」",RD,"官方口径 · 这是下界")]
for d,(n,sub,c,hard) in zip(PAN,NUMS):
    x=d["x"]
    p.append(f'<rect x="{x+12}" y="{NUM_T}" width="{PW-24}" height="{NUM_H}" rx="6" fill="{c}"/>')
    p.append(f'<text class="svgnum" x="{x+26}" y="{NUM_T+26}" fill="#fff" style="font-size:21px">{n}</text>')
    p.append(f'<text class="svgsm" x="{x+PW-26}" y="{NUM_T+25}" text-anchor="end" fill="#ffffffd8" style="font-size:10.5px">{sub}</text>')
    # 硬度徽标：三个数一个是推的、一个是举例、一个是官方下界，不标就都长得像结论
    p.append(f'<text class="svgsm" x="{x+26}" y="{NUM_T+42}" fill="#ffffffb0" style="font-size:9.5px">{hard}</text>')

# ── 真实例子框
EX=[("真实例子", ["模型权重的 bias、layernorm 的","缩放系数 —— 最后两维带 1 的","小张量，占的块比你以为的小"], BL),
    ("真实例子 · splash attention", ["online softmax 的逐行 max / sum，","逻辑上每行 1 个数，源码里开成","<tspan font-family='ui-monospace,monospace'>(bq, 128)</tspan> 的 scratch —— 127 份是副本"], PU),
    ("真实例子 · 所有主流注意力", ["<tspan font-family='ui-monospace,monospace'>head_dim</tspan> = 128 或 64 时，flash","注意力的 QK 那一步，官方实测","MXU 利用率不到一半"], RD)]
for d,(t,lines,c) in zip(PAN,EX):
    x=d["x"]
    p.append(f'<rect x="{x+12}" y="{EX_T}" width="{PW-24}" height="{EX_H}" rx="6" fill="#f8f9fa" stroke="#dadce0"/>')
    p.append(f'<text class="svgsm" x="{x+24}" y="{EX_T+18}" fill="{c}" font-weight="700">{t}</text>')
    for i,l in enumerate(lines):
        p.append(f'<text class="svgsm" x="{x+24}" y="{EX_T+37+i*16}" fill="#202124" style="font-size:11px">{l}</text>')

# ── 底部横条：三笔不同时发生
p.append(f'<rect x="0" y="{BAND}" width="1000" height="100" rx="9" fill="#1a73e8"/>')
p.append(f'<text class="svglbl" x="20" y="{BAND+25}" fill="#fff" style="font-size:14px">'
         f'关键：注意力里这三笔<tspan font-weight="700">不同时发生</tspan> —— 真正亏的只有第 ③ 笔</text>')
ROWS=[("① 内存",  "Q 块是 [512, 128]，次内维 512 是 8 的整数倍 → 一分不浪费"),
      ("② VPU",  "128 正好铺满 128 个 lane → 满载。亏的是 softmax 那几个 scratch，而 VPU 的大头是铺满 bq×bkv 的 exp，不亏"),
      ("③ MXU",  "head_dim 128 撞 256 的收缩边 → 这一半是实打实亏掉的，也是唯一跟 head_dim 有关的一笔")]
for i,(a,b) in enumerate(ROWS):
    y=BAND+45+i*19
    p.append(f'<text class="svgsm" x="20" y="{y}" fill="#fff" font-weight="700" style="font-size:11.5px">{a}</text>')
    p.append(f'<text class="svgsm" x="86" y="{y}" fill="#ffffffdd" style="font-size:11.5px">{b}</text>')
p.append('</svg>')
# 从前写去 /tmp/fig25.svg，然后手工粘进 HTML —— 注入器认的是本目录下的
# fig2-5.svg，命名跟其它八个脚本对齐。见 topic02-inject-s012.py。
W('fig2-5.svg',p)
print(H,'ok')
