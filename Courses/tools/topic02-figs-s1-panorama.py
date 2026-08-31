# -*- coding: utf-8 -*-
"""专题二 §1 的四张图。所有数字来自 /tmp/topic02-specs.md 里已核实的公开来源。"""
import io
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368"

# ══════════ 图 1-1 · TPU v7 芯片全景 ══════════
p=['<svg viewBox="0 0 1000 470" width="100%" role="img" aria-label="TPU v7 芯片全景：一颗芯片两个 chiplet，每个 chiplet 一个 TensorCore、两个 SparseCore、96 GB HBM">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">TPU v7（Ironwood）一颗 chip —— 双 chiplet，两个 chiplet 各有独立的内存空间</text>',
   '<text class="svgsm" x="0" y="35">来源：Google Cloud 官方 TPU7x 文档（结构与容量）＋ JAX 公开源码里的芯片信息表（片上尺寸）</text>']
p.append('<rect x="0" y="48" width="1000" height="318" rx="12" fill="#fff" stroke="#dadce0" stroke-width="1.5"/>')
p.append('<text class="svgsm" x="14" y="66">一颗 chip</text>')
# ⛔ 这张图原来把两个 chiplet **一模一样地画了两遍**，右半边没有任何新信息，
#    占掉将近一半的墨。而两个 chiplet 之间真正要讲的那件事 ——
#    「它们各有独立地址空间，跨过去要走 D2D」—— 反倒只有一行小字。
#    改成：左边画全，右边只留一个「同上」的窄条，腾出来的中间地带专门讲那件事。
x=16
p.append(f'<rect x="{x}" y="76" width="468" height="276" rx="10" fill="#f8f9fa" stroke="{GR}"/>')
p.append(f'<text class="svglbl" x="{x+16}" y="98" fill="{GR}">chiplet 1　=　JAX 眼里的 device 0</text>')
# TensorCore
p.append(f'<rect x="{x+16}" y="108" width="300" height="130" rx="8" fill="#e6f4ea" stroke="{GR}"/>')
p.append(f'<text class="svglbl" x="{x+28}" y="128" fill="{GR}">TensorCore ×1</text>')
for j,(t,sb,c) in enumerate([("MXU ×2","256 × 256 脉动阵列",GR),("VPU","逐元素 · 峰值低两个数量级",PU),
                             ("XLU","转置 · 跨 lane 归约 —— 慢",PU),("标量单元 ×1","产生所有地址 —— 只有一个",RD)]):
    yy=138+j*24
    p.append(f'<rect x="{x+28}" y="{yy}" width="112" height="20" rx="4" fill="{c}"/>')
    p.append(f'<text class="svgsm" x="{x+36}" y="{yy+14}" fill="#fff">{t}</text>')
    p.append(f'<text class="svgsm" x="{x+150}" y="{yy+14}">{sb}</text>')
# SparseCore
p.append(f'<rect x="{x+324}" y="108" width="128" height="130" rx="8" fill="#fef7e0" stroke="#f9ab00"/>')
# 个数并进标题 —— 原来「× 2」单占一行，跟下面第一条只差 8px，挤在一起像重影。
p.append(f'<text class="svglbl" x="{x+334}" y="128" fill="#7a5000">SparseCore × 2</text>')
# 后三行是「它到底能干嘛」。只写 subcore 数和 VMEM 大小，读者知道它存在、
# 不知道它有什么用 —— 而下一句「GPU 没有可编程协处理器」就没有分量了。
for j,t in enumerate(["16 subcore","× 16 lane","VMEM 512 KiB",
                      "跑自己的程序：","前缀和 · 排序 · 计数","gather / scatter"]):
    p.append(f'<text class="svgsm" x="{x+334}" y="{148+j*15}" fill="#7a5000">{t}</text>')
# 片上存储
for j,(t,v,c) in enumerate([("VMEM","64 MiB · 编译器显式管",GR),("SMEM","1 MiB",GY),
                            ("CMEM","0 —— 没有这一层",RD)]):
    yy=248+j*26
    p.append(f'<rect x="{x+16}" y="{yy}" width="436" height="22" rx="4" fill="#fff" '
             f'stroke="{c}"{" stroke-dasharray=\"4 3\"" if j==2 else ""}/>')
    p.append(f'<text class="svglbl" x="{x+26}" y="{yy+15}" fill="{c}">{t}</text>')
    p.append(f'<text class="svgsm" x="{x+96}" y="{yy+15}" fill="{c}">{v}</text>')
p.append(f'<rect x="{x+16}" y="326" width="436" height="18" rx="4" fill="{PU}"/>')
p.append(f'<text class="svgsm" x="{x+26}" y="339" fill="#fff">HBM 96 GiB／device —— 这个 chiplet 私有，不与另一个共享地址空间</text>')

# ── chiplet 2：只留一个「同上」的窄条 ────────────────────────────────
x2=830
p.append(f'<rect x="{x2}" y="76" width="154" height="276" rx="10" fill="#f8f9fa" stroke="{GR}" stroke-dasharray="6 4"/>')
p.append(f'<text class="svglbl" x="{x2+14}" y="98" fill="{GR}">chiplet 2</text>')
p.append(f'<text class="svgsm" x="{x2+14}" y="114" fill="{GR}">= device 1</text>')
p.append(f'<text class="svgnum" x="{x2+77}" y="196" text-anchor="middle" fill="{GY}" style="font-size:26px">同上</text>')
for j,t in enumerate(["结构与左边","逐项相同 ——","这里不再画一遍"]):
    p.append(f'<text class="svgsm" x="{x2+77}" y="{216+j*16}" text-anchor="middle" fill="{GY}">{t}</text>')
p.append(f'<rect x="{x2+14}" y="326" width="126" height="18" rx="4" fill="{PU}"/>')
p.append(f'<text class="svgsm" x="{x2+77}" y="339" text-anchor="middle" fill="#fff">另一块 96 GiB</text>')

# ── 腾出来的中间地带：讲两个 chiplet 之间真正要讲的那件事 ──────────────
p.append(f'<rect x="500" y="108" width="314" height="244" rx="10" fill="#fff8f0" stroke="{OR}"/>')  # 244 才装得下底下那条橙带
p.append(f'<text class="svglbl" x="516" y="130" fill="{OR}">⭐ 这张图真正要讲的是这中间</text>')
for j,t in enumerate([
  "两个 chiplet <tspan font-weight=\"700\">各有独立的地址空间</tspan>，",
  "不再是上代 v4 / v5p 那种统一 MegaCore。",
  "",
  "于是「另一半的数据」<tspan font-weight=\"700\">不能随手读写</tspan> ——",
  "要走 D2D，而且是一次集合通信，",
  "在代码里是显式的一步。",
  "",
  "⚠️ 直接后果：框架日志报的是 <tspan font-weight=\"700\">device</tspan>，",
  "所以 192 GiB 的芯片，你的额度是 96 GiB。"]):
    if t: p.append(f'<text class="svgsm" x="516" y="{152+j*17}">{t}</text>')
p.append(f'<rect x="516" y="{152+9*17+4}" width="282" height="34" rx="6" fill="{OR}"/>')
p.append(f'<text class="svgsm" x="657" y="{152+9*17+18}" text-anchor="middle" fill="#fff">好消息：D2D 比一条 1D ICI 链路快 6 倍</text>')
p.append(f'<text class="svgsm" x="657" y="{152+9*17+31}" text-anchor="middle" fill="#ffffffcc">所以跨 chiplet 贵，但不是灾难</text>')
# D2D
p.append(f'<rect x="486" y="200" width="30" height="46" rx="5" fill="{OR}"/>')
p.append('<text class="svgsm" x="501" y="219" text-anchor="middle" fill="#fff">D2D</text>')
p.append('<text class="svgsm" x="501" y="234" text-anchor="middle" fill="#fff">×6</text>')
p.append(f'<path d="M814 223 h16" stroke="{OR}" stroke-width="1.6"/>')
# 芯片级
p.append(f'<text class="svglbl" x="0" y="392" fill="#202124">整颗 chip 对外：</text>')
for i,(t,v) in enumerate([("HBM","192（官方表写 GiB、正文写 GB）· 7,380 GB/s"),
                          ("ICI","1,200 GB/s 双向 · 每轴 200 GB/s · 3D torus"),
                          ("算力","BF16 2,307 TFLOPS ｜ FP8 4,614 TFLOPS")]):
    p.append(f'<text class="svgsm" x="{112+i*300}" y="392" fill="{GR}">{t}</text>')
    p.append(f'<text class="svgsm" x="{146+i*300}" y="392">{v}</text>')
p.append('<line x1="0" y1="404" x2="1000" y2="404" stroke="#e8eaed"/>')
for i,(n,t) in enumerate([("①","一颗 chip = 2 个 device。所有框架日志按 device 报 —— 容量除以 2 才是你的额度"),
    ("②","VPU 那行的「低两个数量级」是<tspan font-weight=\"700\">上界</tspan>：按每 lane 每周期一次乘加算，位置数之比是 1:128，真实差距只会更大"),
    ("③","片上 SRAM 全是显式管理的，CMEM = 0 —— 这一整层在 GPU 那边叫 cache，在这里根本不存在（第 2 节）")]):
    p.append(f'<text class="svgsm" x="0" y="{424+i*17}" fill="{RD if i==2 else "#202124"}">'
             f'{n} {t}</text>')
W('fig1-1.svg',p+['</svg>'])

# ══════════ 图 1-2 · GB200 三层 ══════════
q=['<svg viewBox="0 0 1000 404" width="100%" role="img" aria-label="GB200 的三层：GPU、节点、NVL72 域，其中域内 72 卡但训练实际用 64 卡">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">GB200 要画三层 —— 而「一台机器」这个词在第三层已经失效了</text>',
   '<text class="svgsm" x="0" y="35">来源：NVIDIA 官方 GB200 NVL72 规格表 ＋ CUDA Blackwell 调优指南</text>']
# 第一层 GPU
q.append(f'<rect x="0" y="50" width="336" height="215" rx="10" fill="#e8f0fe" stroke="{BL}"/>')
# ⛔ 这一格从前写「① 一张 B200 GPU」，而给的是 186 GB / 2,500 的口径 ——
# 那是 **NVL72 里那颗 GB200** 的数。叫「B200」的卡（HGX B200 板上的）是
# 180 GB / 2,250，差 11.1%。两者都叫 Blackwell，光看架构名分辨不出来。
# 本课 §8 已经把「套错机型」立成了教案，§1 自己不能再犯。
q.append(f'<text class="svglbl" x="16" y="72" fill={BL!r}>① 一张 GB200 GPU　'
         f'<tspan style="font-size:10px" fill="{GY}">（NVL72 里的那颗）</tspan></text>')
for j,(t,v) in enumerate([("SM","148 个（第三方拆解，官方未公布）"),("每 SM","128 CUDA core · 4 Tensor Core · SFU"),
                          ("每 SM 并发","最多 64 个 warp"),("寄存器堆","64K × 32 bit = 256 KB / SM"),
                          ("shared","最多 228 KB / SM（含 L1 共 256 KB）"),
                          ("L2","126 MB · 全 GPU 共享 · 自动管"),
                          ("HBM3e","186 GB · 8.0 TB/s"),
                          ("算力 dense","BF16 2.5 ｜ FP8 5 PFLOPS")]):
    q.append(f'<text class="svgsm" x="16" y="{92+j*19}" fill="{BL}">{t}</text>')
    q.append(f'<text class="svgsm" x="106" y="{92+j*19}">{v}</text>')
# 「别混」这两行不能挂在右列 —— 右列只有 230px 宽，这句话会横着穿出框、
# 被隔壁 ② 号框盖掉半截（渲染出来才看得见，改完必须再截一次图）。
# 所以让它独占整框宽度。
q.append(f'<text class="svgsm" x="16" y="244" fill="{RD}">'
         f'⚠️ 别混：HGX 板上的 B200 是 180 GB / BF16 2.25</text>')
q.append(f'<text class="svgsm" x="16" y="258" fill="{RD}">'
         f'同代不同封装差 11.1%，光看「Blackwell」分不出来</text>')
# 第二层 节点
q.append(f'<rect x="352" y="50" width="204" height="215" rx="10" fill="#f3e8fd" stroke="{PU}"/>')
q.append(f'<text class="svglbl" x="366" y="72" fill="{PU}">② 一个节点</text>')
# 2 颗 Grace 不是 1 颗：superchip = 1 Grace + 2 GPU，一个节点 4 GPU ⇒ 2 Grace。
# 18 节点 × 2 = 36 Grace，与官方的「36 Grace + 72 Blackwell」对得上；写 1 颗对不上。
for j,t in enumerate(["2 × Grace ARM64","+ 4 × GB200","（a4x-highgpu-4g）","","744 GB GPU 显存","",
                      "RDMA 4 × CX-7 × 400","　= 1,600 Gbps　← 算通信用这个","管理网 400 Gbps"]):
    q.append(f'<text class="svgsm" x="366" y="{90+j*15}">{t}</text>')
# 「合计 2,000」故意划掉而不是删掉 —— 它是外面最常见的报法，
# 台下多半见过。删了他们下次还会照抄，划掉才记得住为什么不能用。（§4 陷阱二）
q.append(f'<text class="svgsm" x="366" y="225" fill="{RD}" text-decoration="line-through">合计 2,000 Gbps</text>')
q.append(f'<text class="svgsm" x="366" y="239" fill="{RD}" style="font-size:10px">管理网跟 GPU 通信无关，</text>')
q.append(f'<text class="svgsm" x="366" y="252" fill="{RD}" style="font-size:10px">不能加进来（第 4 节陷阱二）</text>')
# 第三层 域
q.append(f'<rect x="572" y="50" width="428" height="215" rx="10" fill="#e6f4ea" stroke="{GR}"/>')
q.append(f'<text class="svglbl" x="588" y="72" fill="{GR}">③ 一个 NVL72 域　←　新的「一台机器」</text>')
q.append(f'<text class="svgsm" x="588" y="92">18 节点 × 4 = 72 GPU，NVSwitch 5 全互联，不走网络</text>')
q.append(f'<text class="svgsm" x="588" y="110">域内 NVLink 总带宽 130 TB/s ｜ 每 GPU 1.8 TB/s</text>')
# 64 vs 72
q.append(f'<rect x="588" y="122" width="396" height="66" rx="6" fill="#fce8e6" stroke="{RD}"/>')
# ⛔ 从前这里写「能进训练任务的是 64」，读起来像平台限制。它不是 ——
# §4 自己写着「64 不是平台限制，是我们自愿让出 11% 容量换稳定，换个人运维就是别的数」。
# 一张 §1 的图不能把运维选择讲成硬件事实。
q.append(f'<text class="svglbl" x="600" y="141" fill="{RD}">⚠️ 72 是物理规模；我们这批集群按 64 编排</text>')
q.append(f'<text class="svgsm" x="600" y="158" fill="{RD}">16 节点上阵，剩 2 节点 / 8 卡热备（auto-repair 关着，坏了要人顶）</text>')
q.append(f'<text class="svgsm" x="600" y="175" fill="{RD}">这是<tspan font-weight="700">编排口径不是平台限制</tspan> —— 换个人运维就是别的数（第 4 节）</text>')
for j,(a,b) in enumerate([("按 72 卡（物理域）","13.4 TB 显存"),("按 64 卡（本课编排口径）","11.9 TB 显存")]):
    q.append(f'<text class="svgsm" x="{588+j*200}" y="206">{a}</text>')
    q.append(f'<text class="svgnum" x="{588+j*200}" y="228" fill="{GR if j else GY}">{b}</text>')
for x in (336,556):
    q.append(f'<path d="M{x} 148 h20" stroke="{GY}" stroke-width="1.5" marker-end="url(#m1)"/>')
q.append('<defs><marker id="m1" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5f6368"/></marker></defs>')
q.append('<line x1="0" y1="266" x2="1000" y2="266" stroke="#e8eaed"/>')
q.append(f'<text class="svglbl" x="0" y="290" fill="#202124">⭐ 跟 TPU 对齐着看：两边的三层完全不在同一个位置</text>')
for j,(a,b,c) in enumerate([("TPU v7","TensorCore　→　chip（2 个 device）　→　切片（3D torus，最大 9,216 chip）",BL),
                            ("GB200","GPU　→　节点（4 GPU）　→　NVL72 域（72 卡，本课按 64 编排）→ 跨域走 RDMA",GR)]):
    q.append(f'<text class="svgsm" x="0" y="{314+j*20}" fill="{c}">{a}</text>')
    q.append(f'<text class="svgsm" x="70" y="{314+j*20}">{b}</text>')
q.append(f'<text class="svgsm" x="0" y="362" fill="{RD}">'
         f'TPU 侧<tspan font-weight="700">同一套 ICI 一路铺到 9,216 chip</tspan>（形状受限：超过 64 颗要按 4×4×4 的 cube 拼）；'
         f'GPU 侧出了 72 卡这个域<tspan font-weight="700">就换一套物理链路</tspan>。</text>')
q.append(f'<text class="svgsm" x="0" y="380" fill="{RD}">'
         f'⚠️ 差别不在「连不连续」（两边都是离散的），在<tspan font-weight="700">换不换协议</tspan> —— '
         f'一个是「距离逐渐变远」，一个是「到某一点突然掉下去」。这是第 4 节全部内容的来源。</text>')
W('fig1-2.svg',q+['</svg>'])
print('1-1 / 1-2 ok')
