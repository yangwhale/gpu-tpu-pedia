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
 ("片上 · 自动管","硬件替你决定放什么","硬件缓存 ＝ 0（v7 无 CMEM）",
  "L1（与 shared 合计上限 256 KB／SM）<TAB>L2 126 MB／GPU<TAB>—— 全自动，你只能提示不能指定",1,0),
 # ⚠️ 这一行的行名从「专用协处理器」改成「可编程协处理器」，空格子也从「没有对应物」
 # 改成「没有可编程的」—— 少了「可编程」三个字，这一格就是错的：GPU 上有 TMA
 # （Hopper 引入、Blackwell 沿用的张量搬运引擎），读到这里的人会立刻想到它。
 # 但 TMA 是**固定功能**的搬运器，而且 TPU 那边同样有 DMA 引擎 —— 它是两边都有的东西，
 # 根本不在这一格里。这一格说的是「能跑自己那段程序」的协处理器。
 ("可编程协处理器","矩阵单元干不了的活","SparseCore × 4／chip（＝ 2／device）<TAB>可编程：前缀和 · 排序 · 计数 · gather / scatter","（没有可编程的）",0,1),
 # 这一行 2026-08-31 补。Chris 的原话是「不管 GPU 还是 TPU 里的那个特殊函数单元，
 # 你都没画出来」—— 确实：XLU 只在图 1-1 里当一个紫盒子出现过、没人说它干嘛，
 # SFU 则要翻到第 3 节的 SM 显微镜才看得见。两个都是「挂在核里的小帮手」，
 # 放在同一行才看得出**两边加的根本不是同一件东西**：
 # 一边补的是「数据横着走」，一边补的是「算超越函数」。
 # ⛔ 三条出处不许省：XLU ×2 出自公开的 JAX scaling book（未标代次）；
 #    SFU 的个数是沿用 Hopper 框图，Blackwell 官方没标；
 #    TPU 侧的超越函数硬件 —— 这一格 2026-08-31 改过两次，两次都值得记：
 #    v1 写「没有单列的单元」，那是把「我查不到」写成了「它没有」，是错的。
 #    v2 退到「公开资料没说」，不错，但太软 —— 它把一件**能给出正面证据**的事
 #    讲成了空白。现在给的是证据本身：开源 JAX 里 Pallas 的成本模型签名是
 #    CostEstimate(flops, transcendentals, bytes_accessed, ...) —— transcendentals
 #    跟 flops **平级单开一栏**。只是「用普通 ALU 多跑几条指令」的东西不会这样计。
 #    部件叫什么、几个、怎么实现，那才是公开资料真正没有的部分（见脚注 §）。
 ("零碎活的小单元","矩阵乘和逐元素之外","XLU 跨 lane 单元 × 2（公开资料，未标代次）"
  "<TAB>转置 · 跨 lane 归约 · shuffle —— 公开资料明写「慢且贵」"
  "<TAB>⚠️ 超越函数<tspan font-weight=\"700\">另算一类</tspan>：Pallas 成本模型里 transcendentals ≠ flops",
  "SFU × 4／处理块（个数沿用 Hopper 画法）<TAB>exp · rcp · rsqrt 等超越函数"
  "<TAB>⚠️ 跨 lane 靠 warp shuffle 指令，官方没把它单列成部件",0,0),
 # 官方正文写「96 GB」是十进制，而同一张表的 192 GiB ÷ 2 = 96 GiB —— 本课统一用 GiB，
 # 理由和附录 A 那个 94.74 GiB 是同一条：判 OOM 的分母必须是二进制的。
 ("芯片外","HBM","96 GiB／device（＝ 192 GiB ÷ 2）<TAB>整 chip 7,372.8 GB/s","186 GB／GPU（软件可见；物理 192 GB）<TAB>8,000 GB/s",0,0),
]
RH,TOP,LX,LW,RX,RW=84,70,168,404,592,408
H=TOP+RH*len(ROWS)+104
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="TPU v7 与 B200 的存储层级对照，共用一根纵轴，空格子表示该层不存在">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
   '把两边挂在同一根纵轴上 —— 同一高度＝同一层级，空格子就是「这边没有这一层」</text>',
   # 加了「零碎活的小单元」那行之后，纵轴就不是纯粹的存储层级了 ——
   # 与其假装它还是，不如把例外说出来：读者对着轴找不到位置比多一行字更糟。
   '<text class="svgsm" x="0" y="35">纵轴：大体按离计算单元由近到远。'
   '中间那两行（协处理器 · 小单元）不是存储，是挂在核里的部件，单独看</text>',
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
        # note 也吃 <TAB> 分行：「没有可编程的」这一格光说「没有」不够，
        # 还得挡住读者立刻会想到的 TMA，一行塞不下。
        p.append(f'<text class="svgnum" x="{x+w/2}" y="{y+26}" text-anchor="middle" fill="{RD}">{txt}</text>')
        for k,t in enumerate(note.split('<TAB>')):
            p.append(f'<text class="svgsm" x="{x+w/2}" y="{y+45+k*16}" text-anchor="middle" fill="{RD}">{t}</text>')
    else:
        for k,t in enumerate(txt.split('<TAB>')):
            p.append(f'<text class="svgsm" x="{x+14}" y="{y+22+k*17}" fill="#202124">{t}</text>')
for i,(lay,sub,tp,gp,te,ge) in enumerate(ROWS):
    y=TOP+i*RH
    p.append(f'<text class="svglbl" x="{LX-26}" y="{y+22}" text-anchor="end" fill="#202124">{lay}</text>')
    p.append(f'<text class="svgsm" x="{LX-26}" y="{y+38}" text-anchor="end">{sub}</text>')
    cell(LX,LW,y,tp,GR,te,"GPU 那边整整一层，在这里根本不存在")
    cell(RX,RW,y,gp,BL,ge,
         "矩阵单元干不了的活，只能回到 CUDA core 手写<TAB>"
         "TMA 是固定功能搬运，TPU 也有 DMA 引擎 —— 不算这一格")
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
V7B,V7BW,V7F=2307,7.3728,4614     # TFLOPS / TB·s⁻¹ / TFLOPS  官方 per chip
B2B,B2BW,B2F=2500,8.000,5000     # dense per GPU，由官方域级数字除 72 得到
H2=828          # 471 是原来只讲 HBM 一层时的高度；下面那段阶梯 ＋ **三行**口径说明撑到 822
                # （2026-09-04：反推校验那句原来一行 1121px 宽、被 viewBox 裁掉，拆成两行后 +18）
q=[f'<svg viewBox="0 0 1000 {H2}" width="100%" role="img" aria-label="TPU v7 与 B200 的算力带宽比几乎完全相同，都是每字节约 312 次浮点运算；并列出 GB200 上 HBM、L2、共享内存三层各自的兑换比 312 / 119 / 64">',
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
      [("BF16",f"{V7B:,} TFLOPS",f"{V7BW:.2f} TB/s",f"{V7B/1000/V7BW*1000:.1f}"),
       ("FP8", f"{V7F:,} TFLOPS",f"{V7BW:.2f} TB/s",f"{V7F/1000/V7BW*1000:.1f}")],
      "来源：Google Cloud 官方 TPU7x 规格表，直接给的就是 per chip")
panel(530,"GB200（NVL72 里那颗 · 每 GPU · dense）",BL,
      [("BF16",f"{B2B:,} TFLOPS",f"{B2BW:.1f} TB/s",f"{B2B/1000/B2BW*1000:.1f}"),
       ("FP8", f"{B2F:,} TFLOPS",f"{B2BW:.1f} TB/s",f"{B2F/1000/B2BW*1000:.1f}")],
      "来源：NVIDIA 官方 GB200 NVL72 域级数字 ÷ 72，稀疏值取一半得 dense")
# 底带不能用 BL —— 上面刚把蓝定成 GPU、绿定成 TPU，这一条是**两边共同的结论**，
# 染成任何一个平台色都会读成「这是那一边的说法」。用中性墨色。
# ⛔ 2026-09-04 黑带从 80 长到 98：里面现在是四行（大字 ＋ 三条警告），
# 新加的那条「别读第三位」必须挨着大字，于是后面两条各下移一行。
q.append(f'<rect x="0" y="260" width="1000" height="98" rx="8" fill="#202124"/>')
q.append('<text class="svgnum" x="500" y="286" text-anchor="middle" fill="#fff" style="font-size:19px">'
# ⛔ ⚠️ 2026-09-04：那个 0.03% 是假精度 —— GB200 侧的 8 TB/s 反推出的引脚速率7.81 Gbps 不是任何一档标称值，十进制／二进制两种读法本身就差 2.4%。结论（同一量级）不变，但别拿第三位当卖点。详见 §1.1 那个折叠。
         'BF16　312.9　对　312.5　　｜　　FP8　625.8　对　625.0　　——&#160;<tspan font-weight="700">同一量级</tspan></text>')
# ⛔ 这一行是 2026-09-04 补的，位置紧贴上面那行大字，**必须挨着**：
#    上面那行给的是三位有效数字，而第三位是假的 —— 隔开就等于没警告。
q.append('<text class="svgsm" x="500" y="304" text-anchor="middle" fill="#ffe08a">'
         '⚠️ 别读第三位：GB200 那个「8 TB/s」官方没写口径，十进制还是 8×1024 差 2.4%（见正文折叠）</text>')
# ⛔ 这里原本写「这不是巧合能解释的。两边被同一批负载和同一代 HBM 技术钉在了同一个
#    比值上」——听着很漂亮，但**没有任何一家这么说过**，是从两个数字倒推出来的因果。
#    而且它还挑 SKU：换成 HGX B200 板上那颗，就是 2,250 ÷ 8.0 = 281.3，离 312 差 11%。
#    所以「都是 312」本身是真的，「为什么都是 312」是推断，两者必须分开讲。
q.append('<text class="svgsm" x="500" y="324" text-anchor="middle" fill="#ffffffcc">'
         '⚠️ 换成 HGX 板上的 B200 就是 2,250 ÷ 8.0 ＝ 281.3，离 312 差 11% —— '
         '「都落在 312」的前提是拿 NVL72 里那颗 GB200 比</text>')
q.append('<text class="svgsm" x="500" y="342" text-anchor="middle" fill="#ffffff99">'
         '推断（没有出处，别当结论讲）：两边都在拿 HBM 带宽去配矩阵算力，'
         '比值撞在一起并不奇怪 —— 但没有哪一家这么解释过</text>')
q.append(f'<text class="svglbl" x="0" y="384" fill="#202124">这个数到底是什么意思</text>')
for i,t in enumerate([
  "· 它是硬件的「胃口」：每从 HBM 搬一个字节，硬件配套能做约 312 次 BF16 运算。喂不满，算力就在空转",
  "· 算子那头是同一个单位：它每搬一个字节，需要做几次运算 —— 这就是它的算术强度",
  "· 于是每个算子只剩一个问题：它的算术强度比 312 高还是低？高就是算力受限，低就是带宽受限",
  "· 专题一那八步里，① embedding 的算术强度是 0 —— 它根本不在这根轴上（完整版 L300 §2.8 收全）。这就是第 2 节要处理的东西",
  "· 而 ④ Dense MLP 的大矩阵乘能远高于 312 —— 那是第 3 节的战场。同一颗芯片，两种完全不同的困境"]):
    q.append(f'<text class="svgsm" x="0" y="{404+i*17}" fill="{RD if i>=3 else "#202124"}">{t}</text>')

# ── 续：同一个除法，每一层各做一次 ────────────────────────────────────
# ⭐ 2026-09-03 补。原来整张图只算了 HBM 那一层就收工 —— 可这门课的主线
#    （算子融合）干的事恰恰是**把数据从 HBM 挪到片上**，挪完之后再拿 312 去比
#    就比错对象了。缺了这一段，融合听起来像「让搬运消失」，
#    而它实际上是「换一个箱子装」——&nbsp;新箱子有它自己的分界线。
#
# 📌 三个数的推导链，一个都不许省：
#    · 312.5 ＝ 2,500 TFLOPS ÷ 8.0 TB/s   官方规格表两个数相除（就是上面那个 312）
#    · 119   ＝ 2,500 TFLOPS ÷ 21 TB/s    L2 带宽是第三方实测，见 G-6
#    · 64    ＝ 8,192 FLOP ÷ 128 B        分子分母都按「每 SM 每周期」
#         分子：B200 每 SM 每周期 8,192 FLOP —— 本课 G-2 已推过。
#         分母：128 B/周期/SM，三处独立佐证互相对得上 ——
#           ① Hopper 微基准实测 127.9 / 128.0 / 127.9（arXiv 2402.13499）
#           ② SemiAnalysis 拆 Blackwell 张量核：「SMEM bandwidth is 128 B/clk」
#           ③ Chips and Cheese：「per-SM throughput 没有越过 128 B/cycle」
#           硬件上的道理：32 个 bank × 每 bank 每周期 4 字节。
#
# ⭐ 64 比另外两个都稳，因为它**不含时钟、也不含 SM 数** —— 纯架构比值。
#    外部锚点校验（本课第一原則要求）：2,500 TFLOPS ÷ 64 ＝ 39 TB/s 全片，
#    再 ÷（148 SM × 128 B）反推时钟 ≈ 2.06 GHz —— 跟 B200 对得上。
#    孤立算出来的数不敢用，能对上外部锚点才敢写进图里。
#
# ⛔ 这一段只画 GB200 一侧，**不是偷懒**：TPU 的 VMEM 带宽官方没公开，
#    补一个灰框上去会让人以为「那边没有这一层」。跟 G-6 那条灰色图例
#    （「官方只给容量、没给带宽」）是同一个口径，别去补。
SM_FLOP, SMEM_BPC, L2BW = 8192, 128, 21          # FLOP/周期/SM · B/周期/SM · TB·s⁻¹
LADDER = [("HBM3e", "片外", f"{B2B:,} TFLOPS ÷ {B2BW:.1f} TB/s", B2B/B2BW),
          ("L2", "硬件自动管", f"{B2B:,} TFLOPS ÷ {L2BW} TB/s", B2B/L2BW),
          ("共享内存 / L1", "软件显式管",
           f"{SM_FLOP:,} FLOP ÷ {SMEM_BPC} B　（都按每 SM 每周期）", SM_FLOP/SMEM_BPC)]
LY, LH, BARX, PXPU = 528, 26, 396, 430/312.5     # 条形起点与「每 FLOP/byte 几像素」
q.append(f'<line x1="0" y1="468" x2="1000" y2="468" stroke="#e8eaed"/>')
q.append('<text class="svglbl" x="0" y="494" fill="#202124" style="font-size:13.5px">'
         '⚠️ 但这条线不止一条 ——&#160;'
         '<tspan font-weight="700">每往计算靠近一层，它就往下掉一截</tspan></text>')
q.append('<text class="svgsm" x="0" y="513">同一个除法，在每一层各做一次。'
         '下面三个数都只对 GB200 那一颗成立，推导链写在每一行上</text>')
for i,(lay,who,formula,val) in enumerate(LADDER):
    y = LY + i*(LH+14)
    q.append(f'<text class="svglbl" x="0" y="{y+18}" fill="#202124">{lay}</text>')
    q.append(f'<text class="svgsm" x="0" y="{y+34}" fill="{GY}">{who}</text>')
    q.append(f'<text class="svgsm" x="150" y="{y+20}" fill="#202124">{formula}</text>')
    q.append(f'<rect x="{BARX}" y="{y}" width="{val*PXPU:.0f}" height="{LH}" rx="4" '
             f'fill="{BL if i<2 else PU}"/>')
    q.append(f'<text class="svgnum" x="1000" y="{y+21}" text-anchor="end" '
             f'fill="{BL if i<2 else PU}" style="font-size:20px">{val:.0f}</text>')
# ⛔ 2026-09-03 这行原来写「条越短，越容易撞到带宽那一侧」——&nbsp;**方向反了**。
#    Roofline 的屋脊点 I* ＝ 峰值算力 ÷ 带宽：算子强度 I < I* 才是带宽受限。
#    I* 越小，门槛越低，算子**越容易**越过它变成算力受限 —— 这是好事。
#    （经典 roofline 的标准表述就是「ridge point 越靠左越容易达到峰值」。）
#    ⚠️ 那为什么黄框又说「融合做过头会在这条线上重新撞墙」？不矛盾 ——&nbsp;
#       门槛从 312 降到 64 只降了 5 倍，而融合把 HBM 流量转成片上流量，
#       片上搬的字节数可能涨几十倍。**撞墙不是因为门槛变高，是因为分母变大。**
#    这个错是 Chris 追问「算术强度是不是按所有 SM 加总算的」时顺带钓出来的。
q.append(f'<text class="svgsm" x="1000" y="{LY+2*(LH+14)+34}" text-anchor="end" fill="{GY}">'
         'FLOP / byte　—— 每一层各自的门槛。'
         '<tspan font-weight="700">条越短门槛越低，算子越容易在这一层变成算力受限</tspan></text>')
q.append('<rect x="0" y="650" width="1000" height="86" rx="8" fill="#fef7e0" stroke="#f9ab00"/>')
q.append('<text class="svglbl" x="18" y="672" fill="#7a5000">'
         '⭐ 从 HBM 爬到共享内存，这条线只降了约 5 倍 ——&#160;'
         '<tspan font-weight="700">片上不是无限快的</tspan></text>')
# ⛔ 这里原来写「换箱子，不是丢箱子」——&nbsp;好记，但**箱子是第四个比方**。
#    本课的比方总预算是三个本体：冷库＝HBM、灶台＝片上暂存、刀宽＝指令粒度，
#    而且那三个到 §3 才立。在 §1 引第四个，等于让台下多背一套映射。
#    这里改成不打比方的直说 —— 反正这一段本来就是算术。
q.append('<text class="svgsm" x="18" y="692" fill="#7a5000">'
         '算子融合减掉的是 <tspan font-weight="700">HBM 那一层</tspan>的搬运；'
         '可那批数据总得在片上落脚 ——&#160;<tspan font-weight="700">账没有消失，它挪到了下面这条线上</tspan>。</text>')
q.append('<text class="svgsm" x="18" y="710" fill="#7a5000">'
         '所以融合之后要拿新的强度去跟 <tspan font-weight="700">64</tspan> 比，'
         '不能再跟 312 比 ——&#160;融合<tspan font-weight="700">做过头，会在新的这条线上重新撞墙</tspan>。</text>')
q.append('<text class="svgsm" x="18" y="726" fill="#7a5000">'
         '⛔ 只画了 GB200 一侧：TPU 的 VMEM 带宽官方没公开，那条线存在但给不出数。</text>')
q.append(f'<text class="svgsm" x="0" y="750" fill="{GY}">'
         '64 的出处：分子 8,192 FLOP/周期/SM 见图 G-2；分母 128 B/周期/SM 有三处一致的公开测量'
         '（Hopper 微基准实测 127.9 · SemiAnalysis · Chips and Cheese），硬件上是 32 bank × 4 B。</text>')
q.append(f'<text class="svgsm" x="0" y="768" fill="{GY}">'
         '⚠️ 三条线口径不同：上两条是<tspan font-weight="700">整颗芯片</tspan>'
         '（2,500 TFLOPS ÷ 全片带宽），第三条是<tspan font-weight="700">单个 SM 每周期</tspan>。'
         '换算过去比值不变 ——&#160;148 和时钟同时出现在分子分母，约掉了，'
         '<tspan font-weight="700">这是恒等不是近似</tspan>。</text>')
# ⛔ 2026-09-04：这一行原来是一整条，量出来 1121px 宽，
#    而 viewBox 只有 1000 —— **右边 121px 直接被裁掉，末尾那半句谁都没读到**。
#    ⭐ SVG 的 <text> 不换行，长句必须自己断行；判据进了 topic02-lint-layout.py。
q.append(f'<text class="svgsm" x="0" y="786" fill="{GY}">'
         '反推校验：2,500 ÷ 64 ＝ 39 TB/s 全片，再 ÷（148 SM × 128 B）'
         '得时钟 ≈ 2.06 GHz，与 B200 对得上。</text>')
q.append(f'<text class="svgsm" x="0" y="804" fill="{GY}">'
         '⭐ 真正的差别不在算法，在<tspan font-weight="700">可达性</tspan>：'
         'HBM 那 8 TB/s 是 148 个 SM 抢的，共享内存这 128 B/周期是每个 SM 自己的。</text>')
W('fig1-4.svg',q+['</svg>'])
print('1-3 / 1-4 ok  ratios:',
      round(V7B/1000/V7BW*1000,1), round(B2B/1000/B2BW*1000,1),
      round(V7F/1000/V7BW*1000,1), round(B2F/1000/B2BW*1000,1))
