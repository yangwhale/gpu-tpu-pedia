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
for k,x in enumerate([16,516]):
    p.append(f'<rect x="{x}" y="76" width="468" height="276" rx="10" fill="#f8f9fa" stroke="{BL}"/>')
    p.append(f'<text class="svglbl" x="{x+16}" y="98" fill="{BL}">chiplet {k+1}　=　JAX 眼里的 device {k}</text>')
    # TensorCore
    p.append(f'<rect x="{x+16}" y="108" width="330" height="130" rx="8" fill="#e8f0fe" stroke="{BL}"/>')
    p.append(f'<text class="svglbl" x="{x+28}" y="128" fill="{BL}">TensorCore ×1</text>')
    for j,(t,s,c) in enumerate([("MXU ×2","256 × 256 脉动阵列",BL),("VPU","逐元素 · 峰值低 2 个数量级",PU),
                                ("XLU","转置 / 归约 / 排列",PU),("标量单元 ×1","产生所有地址 —— 只有一个",RD)]):
        yy=138+j*24
        p.append(f'<rect x="{x+28}" y="{yy}" width="140" height="20" rx="4" fill="{c}"/>')
        p.append(f'<text class="svgsm" x="{x+36}" y="{yy+14}" fill="#fff">{t}</text>')
        p.append(f'<text class="svgsm" x="{x+178}" y="{yy+14}">{s}</text>')
    # SparseCore
    p.append(f'<rect x="{x+354}" y="108" width="98" height="130" rx="8" fill="#fef7e0" stroke="#f9ab00"/>')
    p.append(f'<text class="svglbl" x="{x+364}" y="128" fill="#7a5000">SparseCore</text>')
    p.append(f'<text class="svgnum" x="{x+364}" y="150" fill="#7a5000">× 2</text>')
    for j,t in enumerate(["16 subcore","× 16 lane","VMEM 512 KiB","无 MXU"]):
        p.append(f'<text class="svgsm" x="{x+364}" y="{168+j*17}" fill="#7a5000">{t}</text>')
    # 片上存储
    for j,(t,v,c) in enumerate([("VMEM","64 MiB · 编译器显式管",GR),("SMEM","1 MiB",GY),
                                ("CMEM","0 —— 没有这一层",RD)]):
        yy=248+j*26
        p.append(f'<rect x="{x+16}" y="{yy}" width="436" height="22" rx="4" fill="#fff" '
                 f'stroke="{c}"{" stroke-dasharray=\"4 3\"" if j==2 else ""}/>')
        p.append(f'<text class="svglbl" x="{x+26}" y="{yy+15}" fill="{c}">{t}</text>')
        p.append(f'<text class="svgsm" x="{x+96}" y="{yy+15}" fill="{c}">{v}</text>')
    # HBM
    p.append(f'<rect x="{x+16}" y="326" width="436" height="18" rx="4" fill="{PU}"/>')
    p.append(f'<text class="svgsm" x="{x+26}" y="339" fill="#fff">HBM 96 GB —— 这个 chiplet 私有，不与另一个共享地址空间</text>')
# D2D
p.append(f'<rect x="486" y="200" width="30" height="46" rx="5" fill="{OR}"/>')
p.append('<text class="svgsm" x="501" y="219" text-anchor="middle" fill="#fff">D2D</text>')
p.append('<text class="svgsm" x="501" y="234" text-anchor="middle" fill="#fff">×6</text>')
# 芯片级
p.append(f'<text class="svglbl" x="0" y="392" fill="#202124">整颗 chip 对外：</text>')
for i,(t,v) in enumerate([("HBM","192（官方表写 GiB、正文写 GB）· 7,380 GB/s"),
                          ("ICI","1,200 GB/s 双向 · 每轴 200 GB/s · 3D torus"),
                          ("算力","BF16 2,307 TFLOPS ｜ FP8 4,614 TFLOPS")]):
    p.append(f'<text class="svgsm" x="{112+i*300}" y="392" fill="{BL}">{t}</text>')
    p.append(f'<text class="svgsm" x="{146+i*300}" y="392">{v}</text>')
p.append('<line x1="0" y1="404" x2="1000" y2="404" stroke="#e8eaed"/>')
for i,(n,t) in enumerate([("①","一颗 chip = 2 个 device。所有框架日志按 device 报 —— 容量除以 2 才是你的额度"),
    ("②","两个 chiplet 各有独立地址空间（不再是上代的统一 MegaCore）。跨 chiplet 要走 D2D + 集合通信，不是随手读写 —— 好在 D2D 比一条 1D ICI 链路快 6 倍"),
    ("③","片上 SRAM 全是显式管理的，CMEM = 0 —— 这一整层在 GPU 那边叫 cache，在这里根本不存在（§2）")]):
    p.append(f'<text class="svgsm" x="0" y="{424+i*17}" fill="{RD if i==2 else "#202124"}">'
             f'{n} {t}</text>')
W('fig1-1.svg',p+['</svg>'])

# ══════════ 图 1-2 · GB200 三层 ══════════
q=['<svg viewBox="0 0 1000 404" width="100%" role="img" aria-label="GB200 的三层：GPU、节点、NVL72 域，其中域内 72 卡但训练实际用 64 卡">',
   '<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">GB200 要画三层 —— 而「一台机器」这个词在第三层已经失效了</text>',
   '<text class="svgsm" x="0" y="35">来源：NVIDIA 官方 GB200 NVL72 规格表 ＋ CUDA Blackwell 调优指南</text>']
# 第一层 GPU
q.append(f'<rect x="0" y="50" width="336" height="196" rx="10" fill="#e8f0fe" stroke="{BL}"/>')
q.append(f'<text class="svglbl" x="16" y="72" fill={BL!r}>① 一张 B200 GPU</text>')
for j,(t,v) in enumerate([("SM","148 个（第三方拆解，官方未公布）"),("每 SM","128 CUDA core · 4 Tensor Core"),
                          ("每 SM 并发","最多 64 个 warp"),("寄存器堆","64K × 32 bit = 256 KB / SM"),
                          ("shared","最多 228 KB / SM（含 L1 共 256 KB）"),
                          ("L2","126 MB · 全 GPU 共享 · 自动管"),
                          ("HBM3e","186 GB · 8.0 TB/s"),
                          ("算力 dense","BF16 2.5 ｜ FP8 5 PFLOPS")]):
    q.append(f'<text class="svgsm" x="16" y="{92+j*19}" fill="{BL}">{t}</text>')
    q.append(f'<text class="svgsm" x="106" y="{92+j*19}">{v}</text>')
# 第二层 节点
q.append(f'<rect x="352" y="50" width="204" height="196" rx="10" fill="#f3e8fd" stroke="{PU}"/>')
q.append(f'<text class="svglbl" x="366" y="72" fill="{PU}">② 一个节点</text>')
for j,t in enumerate(["Grace ARM64","+ 4 × B200","（a4x-highgpu-4g）","","744 GB GPU 显存","",
                      "RDMA 4 × CX-7 × 400","　= 1,600 Gbps","管理网 400 Gbps","合计 2,000 Gbps"]):
    q.append(f'<text class="svgsm" x="366" y="{90+j*15}">{t}</text>')
# 第三层 域
q.append(f'<rect x="572" y="50" width="428" height="196" rx="10" fill="#e6f4ea" stroke="{GR}"/>')
q.append(f'<text class="svglbl" x="588" y="72" fill="{GR}">③ 一个 NVL72 域　←　新的「一台机器」</text>')
q.append(f'<text class="svgsm" x="588" y="92">18 节点 × 4 = 72 GPU，NVSwitch 5 全互联，不走网络</text>')
q.append(f'<text class="svgsm" x="588" y="110">域内 NVLink 总带宽 130 TB/s ｜ 每 GPU 1.8 TB/s</text>')
# 64 vs 72
q.append(f'<rect x="588" y="122" width="396" height="66" rx="6" fill="#fce8e6" stroke="{RD}"/>')
q.append(f'<text class="svglbl" x="600" y="141" fill="{RD}">⚠️ 72 是物理规模，能进训练任务的是 64</text>')
q.append(f'<text class="svgsm" x="600" y="158" fill="{RD}">16 节点 = 64 卡进任务，剩 2 节点 / 8 卡热备</text>')
q.append(f'<text class="svgsm" x="600" y="175" fill="{RD}">64 正好是 2 的幂 —— 跟第 0 节「物理量还是可用量」同一个自问</text>')
for j,(a,b) in enumerate([("按 72 卡（物理域）","13.4 TB 显存"),("按 64 卡（可训练）","11.9 TB 显存")]):
    q.append(f'<text class="svgsm" x="{588+j*200}" y="206">{a}</text>')
    q.append(f'<text class="svgnum" x="{588+j*200}" y="228" fill="{GR if j else GY}">{b}</text>')
for x in (336,556):
    q.append(f'<path d="M{x} 148 h20" stroke="{GY}" stroke-width="1.5" marker-end="url(#m1)"/>')
q.append('<defs><marker id="m1" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5f6368"/></marker></defs>')
q.append('<line x1="0" y1="266" x2="1000" y2="266" stroke="#e8eaed"/>')
q.append(f'<text class="svglbl" x="0" y="290" fill="#202124">⭐ 跟 TPU 对齐着看：两边的三层完全不在同一个位置</text>')
for j,(a,b,c) in enumerate([("TPU v7","TensorCore　→　chip（2 个 device）　→　切片（3D torus，最大 9,216 chip）",BL),
                            ("GB200","GPU　→　节点（4 GPU）　→　NVL72 域（72 卡，可训练 64）→ 跨域走 RDMA",GR)]):
    q.append(f'<text class="svgsm" x="0" y="{314+j*20}" fill="{c}">{a}</text>')
    q.append(f'<text class="svgsm" x="70" y="{314+j*20}">{b}</text>')
q.append(f'<text class="svgsm" x="0" y="362" fill="{RD}">'
         f'TPU 侧「切片」是一个可以从 4 chip 一路开到 9,216 chip 的连续谱；GPU 侧「域」是一个 72 卡的硬边界，出了它就换一套物理链路。</text>')
q.append(f'<text class="svgsm" x="0" y="380" fill="{RD}">'
         f'这条差别是 §4 全部内容的来源 —— 一个是「距离逐渐变远」，一个是「到某一点突然掉下去」。</text>')
W('fig1-2.svg',q+['</svg>'])
print('1-1 / 1-2 ok')
