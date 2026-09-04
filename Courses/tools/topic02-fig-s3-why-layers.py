# -*- coding: utf-8 -*-
"""图 3-1 · CUDA 那六层，每一层是被什么逼出来的。

⭐ **为什么要有这一张（它跟 §3.3 已有的三张不重复）。**

§3.3 里关于「六层」已经有三张图，各答一个问题：

    G-3   六层 × **钉在哪块硅上**、这一层能共享什么、怎么同步
    P-31  六层 × **什么时候定的**（划分 vs 调度，拆成两个问题）
    T-3   **按问题对齐**的两边对照 —— 看的是 TPU 那两个空格

三张都很完整，但它们共同**没有回答一个学员一定会问的问题**：
这六层到底为什么要长成六层？

2026-09-03 讲这一节时被现场追问（warp group 是干嘛的、CTA 又是什么），
临时用「每一层都是被一个具体的硬件麻烦逼出来的」讲了一遍，效果比三张表都好。
回来一查：`逼出来` / `Cooperative Thread Array` / `为什么是 32` 在全套文档里
**一次都没出现过**。这张图就是把那次口头讲解固定下来。

⛔ **它只负责因果那一列。** 包含关系去看下面折叠的全表，硬件归属去看 G-3，
   时机去看 P-31。这里每一行只说两件事：这一层保证了什么、它是被什么逼出来的。

⚠️ 两个最容易混的数，图上单开一条提示：
   1,024 是**一个 block 的上限**（＝32 warp），不是一个 SM 的上限；
   一个 SM 同时驻留 64 个 warp（＝2,048 线程），而且这个数按代不同。
"""
import io

BL, OR, GR, RD, GY, YL = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368", "#f9ab00"

W = 1400
# 四列左边界。⚠️ 第一版取 (0,210,350,700)，渲染出来第 2 列直接压在第 3 列上
# ——「≤ 32 warp」和「保证落在」叠成「warp证落」。SVG 的 <text> **不会自动换行、
# 也不会撑开布局**，列宽不够只会静默重叠，脚本照样 exit 0。**必须渲染出来看。**
C1, C2, C3, C4 = 0, 190, 410, 790
RH, TOP = 62, 86                            # 行高 / 表体起点

# (层名, 英文, 规模, 保证了什么, 被什么逼出来的 —— None 表示「它是边界，不是被逼出来的」)
ROWS = [
    ("grid", "grid", "整个 kernel 的全部 block",
     "只剩 L2 和 HBM ——&#160;没有片上快捷通道", None),
    ("cluster", "thread block cluster", "≤ 8 块（H100 起可 opt-in 到 16）",
     "同一个 GPC 内，能直接读写别的块的 shared memory",
     "一个 SM 那 227 KiB 不够用了。代价：邻居的暂存要走 GPC 网络，比自己那块慢"),
    ("线程块", "CTA / block", "≤ 1,024 线程 ＝ ≤ 32 warp",
     "保证落在同一个 SM：共享 shared memory、能 __syncthreads",
     "「谁跟谁能共用暂存、能对齐脚步」需要一条边界 ——&#160;CTA 就是这个作用域"),
    ("warp 组", "warpgroup", "4 个 warp ＝ 128 线程",
     "一条 wgmma 由这 128 个线程整体发出",
     "Tensor Core 大到一个 warp 的寄存器喂不饱了，发指令的单位只好往上抬一级"),
    ("warp", "warp", "32 个线程",
     "共用一条指令流；分支分歧只在这一层内发生",
     "取指译码太贵。取一次、译一次、32 份数据一起算 ——&#160;管理成本摊薄 32 倍"),
    ("线程", "thread", "1 个",
     "自己的寄存器。没有自己的命运 ——&#160;跟着所在的 warp 走", None),
]

# 尾部 168 = 红框(18 间隔 + 70) + 绿带(10 间隔 + 70)。⚠️ 红框从两行加到三行时
# 忘了同步这个数，绿带最后一行直接被 viewBox 切掉 —— 而脚本照样 exit 0。
H = TOP + len(ROWS) * RH + 168

p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
     f'aria-label="CUDA 六层线程组织，每一层是被什么硬件麻烦逼出来的，以及 TPU 一层都没有">',
     '<text class="svglbl" x="0" y="17" fill="#202124" style="font-size:14px">'
     '那六层为什么是六层 ——&#160;<tspan font-weight="700">每一层都是被一个具体的麻烦逼出来的</tspan></text>',
     # ⛔ 同样别写方位词、也别引 P-31：这张图两份文档共用，
     #    而 P-31 只在 L300 里有 —— 在 L200 上就是一个指不到的图号。
     #    只引两边都在的：折叠全表（就在同一节）和《GPU 显微镜》G-3。
     '<text class="svgsm" x="0" y="37">'
     '包含关系与「谁定的」看那张折叠全表，每层钉在哪块硅上看《GPU 显微镜》G-3 ——&#160;'
     '这张只答一个问题：为什么要有这一层</text>',
     # 表头
     f'<text class="svgsm" x="{C1}" y="{TOP-12}" fill="{GY}">层</text>',
     f'<text class="svgsm" x="{C2}" y="{TOP-12}" fill="{GY}">有多大</text>',
     f'<text class="svgsm" x="{C3}" y="{TOP-12}" fill="{GY}">它保证了什么</text>',
     f'<text class="svglbl" x="{C4}" y="{TOP-12}" fill="{OR}" style="font-size:12px">'
     f'⚡ 为什么会有这一层</text>']

# 因果列整体加一块浅底 —— 它是这张图唯一的新东西，要一眼看见
p.append(f'<rect x="{C4-16}" y="{TOP-30}" width="{W-C4+16}" height="{len(ROWS)*RH+30}" '
         f'rx="10" fill="#fef7e0"/>')

for i, (zh, en, size, keeps, why) in enumerate(ROWS):
    y = TOP + i * RH
    if i:
        p.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="#e8eaed"/>')
    # 缩进条：越往下缩进越多 = 被上一层装在里面
    p.append(f'<rect x="{i*7}" y="{y+14}" width="3" height="{RH-26}" rx="1.5" fill="{BL}" '
             f'opacity="{0.25 + 0.13 * i:.2f}"/>')
    p.append(f'<text class="svglbl" x="{i*7+12}" y="{y+27}" fill="#202124" '
             f'style="font-size:13px">{zh}</text>')
    p.append(f'<text class="svgsm" x="{i*7+12}" y="{y+44}" fill="{GY}">{en}</text>')
    p.append(f'<text class="svgsm" x="{C2}" y="{y+27}" fill="#202124">{size}</text>')
    p.append(f'<text class="svgsm" x="{C3}" y="{y+27}" fill="{GY}">{keeps}</text>')
    if why:
        p.append(f'<text class="svgsm" x="{C4}" y="{y+27}" fill="#7a5000">{why}</text>')
    else:
        p.append(f'<text class="svgsm" x="{C4}" y="{y+27}" fill="#bba15a">'
                 f'——&#160;它是边界，不是被逼出来的</text>')

YB = TOP + len(ROWS) * RH + 18
p.append(f'<rect x="0" y="{YB}" width="{W}" height="70" rx="8" '
         f'fill="#fff" stroke="{RD}"/>')
p.append(f'<text class="svgsm" x="16" y="{YB+21}" fill="{RD}">'
         f'⚠️ 三个最容易记错的点：<tspan font-weight="700">1,024 是「一个 block 的上限」</tspan>'
         f'（＝32 warp），不是一个 SM 的上限。</text>')
p.append(f'<text class="svgsm" x="16" y="{YB+39}" fill="{RD}">'
         f'一个 SM 同时驻留 <tspan font-weight="700">64 个 warp ＝ 2,048 线程</tspan>'
         f'（B200 / H100 / A100 这几代如此，消费级和 Turing 更少）'
         f'——&#160;这才是「有没有别的活可切」的本钱。</text>')
# ⛔ 第三条是 2026-09-03 自审时查出来的**真错**，不是补充说明。
#    原来 warp 那一行写的是「共用一条指令流、天然锁步」——「天然锁步」在 Volta
#    之前成立，Volta 起的 independent thread scheduling 给了每个线程独立的 PC，
#    warp 内**不再保证**锁步，这正是 __syncwarp() 和那批 _sync 后缀 intrinsic
#    存在的理由（NVIDIA Hopper Tuning Guide / CUDA Handbook 7.5）。
#    ⚠️ 这条正是「听起来像常识的架构关系，老约束在新架构上已被解除」那一类 ——
#       写进教材会让学生照着写 warp-synchronous 代码，然后在真机上偶发错。
p.append(f'<text class="svgsm" x="16" y="{YB+57}" fill="{RD}">'
         f'warp 内<tspan font-weight="700">「天然锁步」是 Volta 之前的事</tspan>'
         f'——&#160;现在每个线程有独立的 PC，要对齐得显式写 __syncwarp()。</text>')

GB = YB + 80
p.append(f'<rect x="0" y="{GB}" width="{W}" height="70" rx="8" '
         f'fill="#e6f4ea" stroke="{GR}"/>')
p.append(f'<text class="svglbl" x="16" y="{GB+23}" fill="#0b6b30">'
         f'⭐ 这六层没有一层是为了「让程序好写」加的</text>')
p.append(f'<text class="svgsm" x="16" y="{GB+43}" fill="#0b6b30">'
         f'它们合起来，就是<tspan font-weight="700">「靠运行时适应」这条路线的组织成本</tspan>'
         f'——&#160;要在运行时换人、分工、组队，就得把这些层级做进硬件，'
         f'再配上调度器、记分板、常驻的巨大寄存器堆。</text>')
p.append(f'<text class="svgsm" x="16" y="{GB+61}" fill="#0b6b30">'
         f'<tspan font-weight="700">✕ TPU 这一整列是空的。</tspan>'
         f'它不做「运行时换人」这个决定，于是这六层同时失去存在的理由 '
         # ⛔ 别写「下面 T-3」：同一份 SVG 在 L300 里排在 T-3 前面、
         #    在 L200 里排在 T-3 后面。方位词一写死，必有一边是错的。
         f'——&#160;T-3 那张对照表上的两个空格，就是这一列不存在时的样子。</text>')
p.append('</svg>')
io.open('fig3-1.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-1 ok', H)
