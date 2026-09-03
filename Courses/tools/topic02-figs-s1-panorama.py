# -*- coding: utf-8 -*-
"""专题二 §1 的四张图。所有数字来自 /tmp/topic02-specs.md 里已核实的公开来源。"""
import io
def W(f,p): io.open(f,'w',encoding='utf-8').write('\n'.join(p))
BL,PU,OR,GR,RD,GY="#1a73e8","#9334e6","#e8710a","#1e8e3e","#d93025","#5f6368"

# ══════════ 图 1-1 · TPU v7 芯片全景 ══════════
# 2026-09-01 重画成**左右对称**。之前两版都错在别的地方，记下来免得再走：
#   ① 先加了渐变／阴影／雾化 —— 图的问题不是不够精致。
#   ② 又砍掉 SMEM／CMEM／部件说明「减字」—— 信息含量本来就是对的。
#   ③ 更早的版本为了省墨，右边 chiplet 只画一个「同上」的窄条。
#      看着是省了，其实把「两个 chiplet 各自独立」这件事讲弱了 ——
#      **两边一模一样地画出来，中间用 D2D 连上，对称本身就是那个结论。**
# 所以现在：左右各一份完整 chiplet，中间一条 D2D，
# 「真正要讲的那件事」挪到主框下方横着铺开，不再挤在两个 chiplet 中间。
SM='style="font-size:11.5px"'      # 正文小字
LB='style="font-size:13px"'        # 区块标题
CW=452                             # 单个 chiplet 的宽度，左右完全一致
p=['<svg viewBox="0 0 1000 940" width="100%" role="img" aria-label="TPU v7 芯片全景：一颗 chip 两个对称的 chiplet，各有一个 TensorCore、两个 SparseCore、64 MiB VMEM 与 96 GiB 私有 HBM，中间用 D2D 相连">',
   '<text class="svglbl" x="0" y="18" fill="#202124" style="font-size:14px">TPU v7（Ironwood）一颗 chip —— 双 chiplet，两个 chiplet 各有独立的内存空间</text>',
   '<text class="svgsm" x="0" y="38" style="font-size:11px">来源：Google Cloud 官方 TPU7x 文档（结构与容量）＋ JAX 公开源码里的芯片信息表（片上尺寸）</text>']
p.append('<rect x="0" y="54" width="1000" height="516" rx="12" fill="#fff" stroke="#dadce0" stroke-width="1.5"/>')
p.append(f'<text class="svgsm" x="16" y="76" {SM}>一颗 chip</text>')


def chiplet(x, dev):
    """一份完整的 chiplet。左右两边调同一个函数 —— 对称是靠共用代码保证的，
    不是靠两段长得差不多的字符串（那种迟早会改漏一边）。

    ⚠️ TensorCore 与 SparseCore **必须上下排、不能左右排**。左右排时
    chiplet 内宽 420px 要塞下「色块＋说明」两列再加 SparseCore，说明列只剩
    148px，而「转置 · 跨 lane 归约 —— 慢」在 11.5px 下要 152px ——
    末字会压在框线上（渲染出来才看得见）。上下排之后说明列有 248px。"""
    o=[f'<rect x="{x}" y="88" width="{CW}" height="466" rx="10" fill="#f8f9fa" stroke="{GR}" stroke-width="1.2"/>',
       f'<text class="svglbl" x="{x+16}" y="114" fill="{GR}" {LB}>chiplet {dev+1}　＝　JAX 眼里的 device {dev}</text>']
    # TensorCore：整幅宽，色块一列、说明一列
    o.append(f'<rect x="{x+16}" y="126" width="420" height="158" rx="10" fill="#e6f4ea" stroke="{GR}" stroke-width="1.2"/>')
    o.append(f'<text class="svglbl" x="{x+32}" y="150" fill="{GR}" {LB}>TensorCore ×1</text>')
    for j,(t,sb,c) in enumerate([("MXU ×2","256 × 256 脉动阵列",GR),("VPU","逐元素 · 峰值低两个数量级",PU),
                                 ("XLU","转置 · 跨 lane 归约 —— 慢",PU),("标量单元 ×1","产生所有地址 —— 只有一个",RD)]):
        yy=162+j*30
        o.append(f'<rect x="{x+32}" y="{yy}" width="112" height="24" rx="5" fill="{c}"/>')
        o.append(f'<text class="svgsm" x="{x+42}" y="{yy+17}" fill="#fff" {SM}>{t}</text>')
        o.append(f'<text class="svgsm" x="{x+160}" y="{yy+17}" {SM}>{sb}</text>')
    # SparseCore：同样整幅宽，六条分两列
    o.append(f'<rect x="{x+16}" y="296" width="420" height="100" rx="10" fill="#fef7e0" stroke="#f9ab00" stroke-width="1.2"/>')
    o.append(f'<text class="svglbl" x="{x+32}" y="320" fill="#7a5000" {LB}>SparseCore × 2</text>')
    for i,col in enumerate([["16 subcore","× 16 lane","VMEM 512 KiB"],
                            ["跑自己的程序：","前缀和 · 排序 · 计数","gather / scatter"]]):
        for j,t in enumerate(col):
            o.append(f'<text class="svgsm" x="{x+32+i*206}" y="{342+j*21}" fill="#7a5000" {SM}>{t}</text>')
    # 片上存储三条 ＋ HBM：等高等距，标签与数值各自对齐到一根竖线
    for j,(t,v,c) in enumerate([("VMEM","64 MiB · 编译器显式管",GR),("SMEM","1 MiB",GY),
                                ("CMEM","0 —— 没有这一层",RD)]):
        yy=408+j*34
        o.append(f'<rect x="{x+16}" y="{yy}" width="420" height="26" rx="6" fill="#fff" '
                 f'stroke="{c}" stroke-width="1"{" stroke-dasharray=\"4 3\"" if j==2 else ""}/>')
        o.append(f'<text class="svglbl" x="{x+30}" y="{yy+18}" fill="{c}" {SM}>{t}</text>')
        o.append(f'<text class="svgsm" x="{x+104}" y="{yy+18}" fill="{c}" {SM}>{v}</text>')
    o.append(f'<rect x="{x+16}" y="510" width="420" height="26" rx="6" fill="{PU}"/>')
    o.append(f'<text class="svgsm" x="{x+30}" y="528" fill="#fff" {SM}>HBM 96 GiB —— 这个 chiplet 私有，不与另一个共享地址空间</text>')
    return o


p+=chiplet(16,0)
p+=chiplet(532,1)

# ── 中间：把两边连起来的那条 D2D ────────────────────────────────────
# 两个 chiplet 之间只有 64px。这条带子窄，但它是整张图的关节 ——
# 左右一模一样，唯一不同的就是「过去要走这里」。
p.append(f'<path d="M468 321 h12" stroke="{OR}" stroke-width="2"/>')
p.append(f'<path d="M520 321 h12" stroke="{OR}" stroke-width="2"/>')
p.append(f'<rect x="480" y="294" width="40" height="54" rx="7" fill="{OR}"/>')
p.append(f'<text class="svglbl" x="500" y="316" text-anchor="middle" fill="#fff" {SM}>D2D</text>')
p.append(f'<text class="svglbl" x="500" y="336" text-anchor="middle" fill="#fff" {SM}>× 6</text>')
for j,t in enumerate(["比一条","1D ICI","链路","快 6 倍"]):
    p.append(f'<text class="svgsm" x="500" y="{372+j*19}" text-anchor="middle" fill="{OR}" {SM}>{t}</text>')

# ── 主框下方：这张图真正要讲的那件事，横着铺开 ────────────────────
p.append(f'<rect x="0" y="586" width="1000" height="150" rx="10" fill="#fff8f0" stroke="{OR}" stroke-width="1.2"/>')
p.append(f'<text class="svglbl" x="20" y="612" fill="{OR}" {LB}>⭐ 左右为什么要一模一样地画两遍 —— 这才是这张图真正要讲的</text>')
for i,col in enumerate([
    ["两个 chiplet <tspan font-weight=\"700\">各有独立的地址空间</tspan>，",
     "不再是上代 v4 / v5p 那种",
     "统一的 MegaCore。"],
    ["于是「另一半的数据」<tspan font-weight=\"700\">不能随手读写</tspan>，",
     "要走中间那条 D2D —— 而且是一次集合通信，",
     "在代码里是显式的一步。"],
    ["⚠️ 直接后果：框架日志报的是 <tspan font-weight=\"700\">device</tspan>，",
     "所以 <tspan font-weight=\"700\">192 GiB 的芯片，你的额度是 96 GiB</tspan>。",
     "两边加起来才是 192。"]]):
    for j,t in enumerate(col):
        p.append(f'<text class="svgsm" x="{20+i*330}" y="{636+j*21}" {SM}>{t}</text>')
p.append(f'<rect x="20" y="704" width="960" height="20" rx="5" fill="{OR}"/>')
p.append(f'<text class="svgsm" x="500" y="718" text-anchor="middle" fill="#fff" {SM}>好消息：D2D 比一条 1D ICI 链路快 6 倍 —— 跨 chiplet 贵，但不是灾难。⚠️ 6 倍的基数是「一条链路 200 GB/s」，不是整颗 chip 那 1,200</text>')

# ── 芯片级 ＋ 三条脚注 ──────────────────────────────────────────
# 这三项**必须竖排**：横着并列时每列只有 300px，而 11.5px 下最长那项
# 要 262px，加上标签就溢出列宽 —— 「7,380 GB/s」会压到下一列的「ICI」上。
p.append(f'<text class="svglbl" x="0" y="766" fill="#202124" {LB}>整颗 chip 对外：</text>')
for i,(t,v) in enumerate([("HBM","192（官方表写 GiB、正文写 GB）· 7,380 GB/s"),
                          ("ICI","整颗 1,200 GB/s（双向合计）· 每轴 200 GB/s ＝ 收发各约 100、同时跑 · D2D 那 6 倍比的是这条 200 · 3D torus"),
                          ("算力","BF16 2,307 TFLOPS ｜ FP8 4,614 TFLOPS")]):
    p.append(f'<text class="svgsm" x="116" y="{766+i*21}" fill="{GR}" {SM}>{t}</text>')
    p.append(f'<text class="svgsm" x="156" y="{766+i*21}" {SM}>{v}</text>')
p.append('<line x1="0" y1="844" x2="1000" y2="844" stroke="#e8eaed"/>')
for i,(n,t) in enumerate([("①","一颗 chip = 2 个 device。所有框架日志按 device 报 —— 容量除以 2 才是你的额度"),
    ("②","VPU 那行的「低两个数量级」是<tspan font-weight=\"700\">上界</tspan>：按每 lane 每周期一次乘加算，位置数之比是 1:128，真实差距只会更大"),
    ("③","片上 SRAM 全是显式管理的，CMEM = 0 —— 这一整层在 GPU 那边叫 cache，在这里根本不存在（第 2 节）")]):
    p.append(f'<text class="svgsm" x="0" y="{868+i*21}" fill="{RD if i==2 else "#202124"}" {SM}>'
             f'{n} {t}</text>')
W('fig1-1.svg',p+['</svg>'])


# ══════════ 图 1-2 · GB200 三层 ══════════
# 2026-09-01 与图 1-1 同一套排版规矩重排：行距 15/19 → 22，字号分层
# （区块标题 13px、正文 11.5px、结论数字 18px），坐标落到 4 的倍数上，
# 圆角与描边分级。**信息一条没减** —— 画布从 404 高到 552，
# 多出来的 148px 全部是行距和留白。
#
# ⚠️ 两处是为了不超框才动的，不要改回去：
#   ① 三层框宽从 336/204/428 调成 348/216/388。第三层那句
#      「16 节点上阵，剩 2 节点 / 8 卡热备（auto-repair 关着，坏了要人顶）」
#      在 11.5px 下要 351px，原来的红框内宽装不下，会横着穿出去。
#   ② 那句同时拆成两行显示 —— 拆的是行，不是内容。
q=['<svg viewBox="0 0 1000 552" width="100%" role="img" aria-label="GB200 的三层：一张 GPU、一个节点、一个 NVL72 域。域内 72 卡是物理规模，本课这批集群按 64 卡编排">',
   '<text class="svglbl" x="0" y="18" fill="#202124" style="font-size:14px">GB200 要画三层 —— 而「一台机器」这个词在第三层已经失效了</text>',
   '<text class="svgsm" x="0" y="38" style="font-size:11px">来源：NVIDIA 官方 GB200 NVL72 规格表 ＋ CUDA Blackwell 调优指南</text>']

# ── 第一层：一张 GPU ────────────────────────────────────────────
# ⛔ 这一格从前写「① 一张 B200 GPU」，而给的是 186 GB / 2,500 的口径 ——
# 那是 **NVL72 里那颗 GB200** 的数。叫「B200」的卡（HGX B200 板上的）是
# 180 GB / 2,250，差 11.1%。两者都叫 Blackwell，光看架构名分辨不出来。
# 本课 §8 已经把「套错机型」立成了教案，§1 自己不能再犯。
q.append(f'<rect x="0" y="56" width="376" height="306" rx="10" fill="#e8f0fe" stroke="{BL}" stroke-width="1.2"/>')
q.append(f'<text class="svglbl" x="16" y="82" fill={BL!r} {LB}>① 一张 GB200 GPU　'
         f'<tspan style="font-size:10.5px" fill="{GY}">（NVL72 里的那颗）</tspan></text>')
for j,(t,v) in enumerate([("SM","148 个（第三方拆解，官方未公布）"),("每 SM","128 CUDA core · 4 Tensor Core · SFU"),
                          ("每 SM 并发","最多 64 个 warp"),("寄存器堆","64K × 32 bit = 256 KB / SM"),
                          ("shared","最多 228 KB / SM（含 L1 共 256 KB）"),
                          ("L2","126 MB · 全 GPU 共享 · 自动管"),
                          ("HBM3e","186 GB · 8.0 TB/s"),
                          ("算力 dense","BF16 2.5 ｜ FP8 5 PFLOPS")]):
    q.append(f'<text class="svgsm" x="16" y="{110+j*22}" fill="{BL}" {SM}>{t}</text>')
    q.append(f'<text class="svgsm" x="106" y="{110+j*22}" {SM}>{v}</text>')
# 「别混」独占整框宽度 —— 挂在右列会横着穿出框、被隔壁 ② 号框盖掉半截。
q.append(f'<rect x="16" y="300" width="344" height="48" rx="6" fill="#fce8e6" stroke="{RD}" stroke-width="1"/>')
q.append(f'<text class="svgsm" x="28" y="320" fill="{RD}" {SM}>⚠️ 别混：HGX 板上的 B200 是 180 GB / BF16 2.25</text>')
q.append(f'<text class="svgsm" x="28" y="340" fill="{RD}" {SM}>同代不同封装差 11.1%，光看「Blackwell」分不出来</text>')

# ── 第二层：一个节点 ────────────────────────────────────────────
# 2 颗 Grace 不是 1 颗：superchip = 1 Grace + 2 GPU，一个节点 4 GPU ⇒ 2 Grace。
# 18 节点 × 2 = 36 Grace，与官方的「36 Grace + 72 Blackwell」对得上；写 1 颗对不上。
q.append(f'<rect x="400" y="56" width="200" height="306" rx="10" fill="#f3e8fd" stroke="{PU}" stroke-width="1.2"/>')
q.append(f'<text class="svglbl" x="416" y="82" fill="{PU}" {LB}>② 一个节点</text>')
for j,t in enumerate(["2 × Grace ARM64","+ 4 × GB200","（a4x-highgpu-4g）","","744 GB GPU 显存","",
                      "RDMA 4 × CX-7 × 400","　= 1,600 Gbps","　← 算通信用这个","管理网 400 Gbps"]):
    if t: q.append(f'<text class="svgsm" x="416" y="{110+j*21}" {SM}>{t}</text>')
# 「合计 2,000」故意划掉而不是删掉 —— 它是外面最常见的报法，台下多半见过。
# 删了他们下次还会照抄，划掉才记得住为什么不能用。（§4 陷阱二）
q.append(f'<rect x="416" y="288" width="168" height="60" rx="6" fill="#fce8e6" stroke="{RD}" stroke-width="1"/>')
q.append(f'<text class="svgsm" x="428" y="308" fill="{RD}" text-decoration="line-through" {SM}>合计 2,000 Gbps</text>')
q.append(f'<text class="svgsm" x="428" y="328" fill="{RD}" {SM}>管理网跟 GPU 通信无关，</text>')
q.append(f'<text class="svgsm" x="428" y="342" fill="{RD}" style="font-size:10.5px">不能加进来（第 4 节陷阱二）</text>')

# ── 第三层：一个 NVL72 域 ───────────────────────────────────────
q.append(f'<rect x="624" y="56" width="376" height="306" rx="10" fill="#e6f4ea" stroke="{GR}" stroke-width="1.2"/>')
q.append(f'<text class="svglbl" x="640" y="82" fill="{GR}" {LB}>③ 一个 NVL72 域　←　新的「一台机器」</text>')
q.append(f'<text class="svgsm" x="640" y="110" {SM}>18 节点 × 4 = 72 GPU，NVSwitch 5 全互联，不走网络</text>')
q.append(f'<text class="svgsm" x="640" y="132" {SM}>域内 NVLink 总带宽 130 TB/s ｜ 每 GPU 1.8 TB/s</text>')
# ⛔ 从前这里写「能进训练任务的是 64」，读起来像平台限制。它不是 ——
# §4 自己写着「64 不是平台限制，是我们自愿让出 11% 容量换稳定」。
# 一张 §1 的图不能把运维选择讲成硬件事实。
q.append(f'<rect x="640" y="148" width="344" height="108" rx="6" fill="#fce8e6" stroke="{RD}" stroke-width="1"/>')
q.append(f'<text class="svglbl" x="654" y="170" fill="{RD}" {SM}>⚠️ 72 是物理规模；我们这批集群按 64 编排</text>')
for j,t in enumerate(["16 节点上阵，剩 2 节点 / 8 卡热备",
                      "（auto-repair 关着，坏了要人顶）",
                      "这是<tspan font-weight=\"700\">编排口径不是平台限制</tspan> —— 换个人运维就是别的数（第 4 节）"]):
    # 第三行最长，末尾的「）」会贴到红框右边线，单独降半号
    fs='style="font-size:10.5px"' if j==2 else SM
    q.append(f'<text class="svgsm" x="654" y="{192+j*22}" fill="{RD}" {fs}>{t}</text>')
for j,(a,b) in enumerate([("按 72 卡（物理域）","13.4 TB 显存"),("按 64 卡（本课编排口径）","11.9 TB 显存")]):
    q.append(f'<text class="svgsm" x="{640+j*180}" y="284" {SM}>{a}</text>')
    q.append(f'<text class="svgnum" x="{640+j*180}" y="312" fill="{GR if j else GY}" style="font-size:18px">{b}</text>')
for x in (376,600):
    q.append(f'<path d="M{x} 200 h20" stroke="{GY}" stroke-width="1.5" marker-end="url(#m1)"/>')
q.append('<defs><marker id="m1" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5f6368"/></marker></defs>')

# ── 底部：跟 TPU 对齐着看 ───────────────────────────────────────
q.append('<line x1="0" y1="386" x2="1000" y2="386" stroke="#e8eaed"/>')
q.append(f'<text class="svglbl" x="0" y="412" fill="#202124" {LB}>⭐ 跟 TPU 对齐着看：两边的三层完全不在同一个位置</text>')
for j,(a,b,c) in enumerate([("TPU v7","TensorCore　→　chip（2 个 device）　→　切片（3D torus，最大 9,216 chip）",BL),
                            ("GB200","GPU　→　节点（4 GPU）　→　NVL72 域（72 卡，本课按 64 编排）→ 跨域走 RDMA",GR)]):
    q.append(f'<text class="svgsm" x="0" y="{440+j*22}" fill="{c}" {SM}>{a}</text>')
    q.append(f'<text class="svgsm" x="72" y="{440+j*22}" {SM}>{b}</text>')
q.append(f'<text class="svgsm" x="0" y="496" fill="{RD}" {SM}>'
         f'TPU 侧<tspan font-weight="700">同一套 ICI 一路铺到 9,216 chip</tspan>（形状受限：超过 64 颗要按 4×4×4 的 cube 拼）；'
         f'GPU 侧出了 72 卡这个域<tspan font-weight="700">就换一套物理链路</tspan>。</text>')
q.append(f'<text class="svgsm" x="0" y="518" fill="{RD}" {SM}>'
         f'⚠️ 差别不在「连不连续」（两边都是离散的），在<tspan font-weight="700">换不换协议</tspan> —— '
         f'一个是「距离逐渐变远」，一个是「到某一点突然掉下去」。这是第 4 节全部内容的来源。</text>')
W('fig1-2.svg',q+['</svg>'])

print('1-1 / 1-2 ok')
