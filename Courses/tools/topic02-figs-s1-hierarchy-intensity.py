# -*- coding: utf-8 -*-
import io
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368"

# ══════════ 图 1-3 · 共用一根纵轴的存储层级 ══════════
ROWS=[  # (层名, 副注, TPU 内容, GPU 内容, TPU空?, GPU空?)
 ("寄存器","离计算单元最近","向量寄存器 8 × 128 × 32 bit = 4 KiB／个<TAB>累加器 128 个",
  "寄存器堆 64K × 32 bit = 256 KB／SM<TAB>每线程最多 255 个",0,0),
 ("片上 · 显式管","谁放什么，由人或编译器决定","VMEM 64 MiB／核（＝／device）<TAB>SMEM 1 MiB<TAB>—— 编译器排布，没有自动行为",
  "shared memory 最多 228 KB／SM<TAB>—— 写 kernel 的人用 __shared__ 手工搬",0,0),
 ("片上 · 自动管","硬件替你决定放什么","CMEM = 0",
  "L1（与 shared 合计上限 256 KB／SM）<TAB>L2 126 MB／GPU<TAB>—— 全自动，你只能提示不能指定",1,0),
 ("专用协处理器","矩阵单元干不了的活","SparseCore × 4／chip（＝ 2／device）<TAB>可编程：前缀和 · 排序 · 计数 · scatter","（没有对应物）",0,1),
 # 官方正文写「96 GB」是十进制，而同一张表的 192 GiB ÷ 2 = 96 GiB —— 本课统一用 GiB，
 # 理由和附录 A 那个 94.74 GiB 是同一条：判 OOM 的分母必须是二进制的。
 ("芯片外","HBM","96 GiB／device（＝ 192 GiB ÷ 2）<TAB>整 chip 7,380 GB/s","186 GB／GPU<TAB>8,000 GB/s",0,0),
]
RH,TOP,LX,LW,RX,RW=84,70,168,404,592,408
H=TOP+RH*len(ROWS)+104
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="TPU v7 与 B200 的存储层级对照，共用一根纵轴，空格子表示该层不存在">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   '把两边挂在同一根纵轴上 —— 同一高度＝同一层级，空格子就是「这边没有这一层」</text>',
   '<text class="svgsm" x="0" y="35">纵轴含义：离计算单元由近到远</text>',
   # 配色跟第 2 节对齐：**GPU 蓝 / TPU 绿**。§2 的 access-lane 图写死了这一套，
# 而这两张原来是反的 —— 同两个颜色在相邻两节代表相反的平台，比不上色更糟：
# 台下会拿颜色当线索，然后被带错。
   f'<text class="svglbl" x="{LX}" y="{TOP-10}" fill="{GR}">TPU v7（口径逐行标注，不统一）</text>',
   f'<text class="svglbl" x="{RX}" y="{TOP-10}" fill="{BL}">GB200（口径逐行标注，不统一）</text>',
   f'<line x1="{LX-14}" y1="{TOP}" x2="{LX-14}" y2="{TOP+RH*len(ROWS)}" stroke="#dadce0"/>']
def cell(x,w,y,txt,col,empty,note=""):
    """空格子的样式**不跟列色走**。

    原来空格子只是把实格子换成虚线边，边框还是那一列的平台色 ——
    于是「这边没有这一层」看起来像「这边有这一层，只是画得淡一点」。
    空的语义跟是哪个平台无关，所以给它一套独立的中性样式：
    灰虚线 + 灰底，跟两边的平台色都不沾边。"""
    if empty:
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{RH-14}" rx="7" '
                 f'fill="#fafafa" stroke="#9aa0a6" stroke-dasharray="6 5"/>')
    else:
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{RH-14}" rx="7" '
                 f'fill="#fff" stroke="{col}" stroke-width="0.8"/>')
    if empty:
        p.append(f'<text class="svgnum" x="{x+w/2}" y="{y+30}" text-anchor="middle" fill="{RD}">{txt}</text>')
        p.append(f'<text class="svgsm" x="{x+w/2}" y="{y+50}" text-anchor="middle" fill="{RD}">{note}</text>')
    else:
        for k,t in enumerate(txt.split('<TAB>')):
            p.append(f'<text class="svgsm" x="{x+14}" y="{y+22+k*17}" fill="#202124">{t}</text>')
for i,(lay,sub,tp,gp,te,ge) in enumerate(ROWS):
    y=TOP+i*RH
    p.append(f'<text class="svglbl" x="{LX-26}" y="{y+22}" text-anchor="end" fill="#202124">{lay}</text>')
    p.append(f'<text class="svgsm" x="{LX-26}" y="{y+38}" text-anchor="end">{sub}</text>')
    cell(LX,LW,y,tp,GR,te,"GPU 那边整整一层，在这里根本不存在")
    cell(RX,RW,y,gp,BL,ge,"矩阵单元干不了的活，只能回到 CUDA core 手写")
yb=TOP+RH*len(ROWS)+12
p.append(f'<line x1="0" y1="{yb}" x2="1000" y2="{yb}" stroke="#e8eaed"/>')
p.append(f'<rect x="0" y="{yb+12}" width="1000" height="66" rx="8" fill="#fef7e0" stroke="#f9ab00"/>')
p.append(f'<text class="svglbl" x="18" y="{yb+34}" fill="#7a5000">'
         f'⭐ 片上 SRAM 的量级相当、归属完全相反：B200 的 L2 是 126 MB，v7 的 VMEM 是 64 MiB × 2 核 ＝ 128 MiB ＝ 134 MB（大约 6%）</text>')
p.append(f'<text class="svgsm" x="18" y="{yb+53}" fill="#7a5000">'
         f'一边全部落在「自动管」那一格，一边全部落在「显式管」那一格 —— 同样一百多 MB 的片上 SRAM，一个替你适应，一个让你安排。</text>')
p.append(f'<text class="svgsm" x="18" y="{yb+70}" fill="#7a5000">'
         f'这就是第 0 节那两条出身，第一次以数字的形式出现。</text>')
W('fig1-3.svg',p+['</svg>'])

# ══════════ 图 1-4 · 压轴：312 FLOP/byte ══════════
V7B,V7BW,V7F=2307,7.380,4614     # TFLOPS / TB·s⁻¹ / TFLOPS  官方 per chip
B2B,B2BW,B2F=2500,8.000,5000     # dense per GPU，由官方域级数字除 72 得到
H2=471
q=[f'<svg viewBox="0 0 1000 {H2}" width="100%" role="img" aria-label="TPU v7 与 B200 的算力带宽比几乎完全相同，都是每字节约 312 次浮点运算">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   '两家公司、两套架构、两种设计哲学 —— 算力除以带宽，落在同一个数上</text>',
   '<text class="svgsm" x="0" y="35">全部由两张官方规格表现算，推导链写在图里</text>']
def panel(x,name,col,rows,src):
    q.append(f'<rect x="{x}" y="50" width="470" height="196" rx="10" fill="#f8f9fa" stroke="{col}"/>')
    q.append(f'<text class="svglbl" x="{x+18}" y="74" fill="{col}" style="font-size:14px">{name}</text>')
    for k,(lab,a,b,r) in enumerate(rows):
        y=92+k*62
        q.append(f'<text class="svgsm" x="{x+18}" y="{y+14}">{lab}</text>')
        q.append(f'<text class="svgtxt" x="{x+18}" y="{y+36}" style="font-size:12.5px">{a}</text>')
        q.append(f'<line x1="{x+18}" y1="{y+42}" x2="{x+250}" y2="{y+42}" stroke="{col}"/>')
        q.append(f'<text class="svgtxt" x="{x+18}" y="{y+58}" style="font-size:12.5px">{b}</text>')
        q.append(f'<text class="svgnum" x="{x+452}" y="{y+44}" text-anchor="end" fill="{col}" '
                 f'style="font-size:22px">{r}</text>')
        q.append(f'<text class="svgsm" x="{x+452}" y="{y+60}" text-anchor="end" fill="{col}">FLOP / byte</text>')
    q.append(f'<text class="svgsm" x="{x+18}" y="238">{src}</text>')
panel(0,"TPU v7（每 chip）",GR,
      [("BF16",f"{V7B:,} TFLOPS",f"{V7BW} TB/s",f"{V7B/1000/V7BW*1000:.1f}"),
       ("FP8", f"{V7F:,} TFLOPS",f"{V7BW} TB/s",f"{V7F/1000/V7BW*1000:.1f}")],
      "来源：Google Cloud 官方 TPU7x 规格表，直接给的就是 per chip")
panel(530,"GB200（NVL72 里那颗 · 每 GPU · dense）",BL,
      [("BF16",f"{B2B:,} TFLOPS",f"{B2BW:.1f} TB/s",f"{B2B/1000/B2BW*1000:.1f}"),
       ("FP8", f"{B2F:,} TFLOPS",f"{B2BW:.1f} TB/s",f"{B2F/1000/B2BW*1000:.1f}")],
      "来源：NVIDIA 官方 GB200 NVL72 域级数字 ÷ 72，稀疏值取一半得 dense")
# 底带不能用 BL —— 上面刚把蓝定成 GPU、绿定成 TPU，这一条是**两边共同的结论**，
# 染成任何一个平台色都会读成「这是那一边的说法」。用中性墨色。
q.append(f'<rect x="0" y="260" width="1000" height="80" rx="8" fill="#202124"/>')
q.append('<text class="svgnum" x="500" y="286" text-anchor="middle" fill="#fff" style="font-size:19px">'
         'BF16　312.6　对　312.5　　｜　　FP8　625.2　对　625.0　　—— 相对差都是 0.03%</text>')
# ⛔ 这里原本写「这不是巧合能解释的。两边被同一批负载和同一代 HBM 技术钉在了同一个
#    比值上」——听着很漂亮，但**没有任何一家这么说过**，是从两个数字倒推出来的因果。
#    而且它还挑 SKU：换成 HGX B200 板上那颗，就是 2,250 ÷ 8.0 = 281.3，离 312 差 11%。
#    所以「都是 312」本身是真的，「为什么都是 312」是推断，两者必须分开讲。
q.append('<text class="svgsm" x="500" y="306" text-anchor="middle" fill="#ffffffcc">'
         '⚠️ 换成 HGX 板上的 B200 就是 2,250 ÷ 8.0 ＝ 281.3，离 312 差 11% —— '
         '「都落在 312」的前提是拿 NVL72 里那颗 GB200 比</text>')
q.append('<text class="svgsm" x="500" y="326" text-anchor="middle" fill="#ffffff99">'
         '推断（没有出处，别当结论讲）：两边都在拿 HBM 带宽去配矩阵算力，'
         '比值撞在一起并不奇怪 —— 但没有哪一家这么解释过</text>')
q.append(f'<text class="svglbl" x="0" y="366" fill="#202124">这个数到底是什么意思</text>')
for i,t in enumerate([
  "· 它是硬件的「胃口」：每从 HBM 搬一个字节，硬件配套能做约 312 次 BF16 运算。喂不满，算力就在空转",
  "· 算子那头是同一个单位：它每搬一个字节，需要做几次运算 —— 这就是它的算术强度",
  "· 于是每个算子只剩一个问题：它的算术强度比 312 高还是低？高就是算力受限，低就是带宽受限",
  "· 专题一那八步里，① embedding 的算术强度是 0 —— 它根本不在这根轴上（第 2.8 节）。这就是第 2 节要处理的东西",
  "· 而 ④ Dense MLP 的大矩阵乘能远高于 312 —— 那是第 3 节的战场。同一颗芯片，两种完全不同的困境"]):
    q.append(f'<text class="svgsm" x="0" y="{386+i*17}" fill="{RD if i>=3 else "#202124"}">{t}</text>')
W('fig1-4.svg',q+['</svg>'])
print('1-3 / 1-4 ok  ratios:',
      round(V7B/1000/V7BW*1000,1), round(B2B/1000/B2BW*1000,1),
      round(V7F/1000/V7BW*1000,1), round(B2F/1000/B2BW*1000,1))
