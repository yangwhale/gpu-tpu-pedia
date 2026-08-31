# -*- coding: utf-8 -*-
import io
GiB=2**30; HBM=206_000_000_000
CHIP=HBM/GiB; DEV=CHIP/2; ALLOC=94.74; RES=DEV-ALLOC

# ════════════════════════════════════════════════════════
# 图 C · 全课唯一的「根」
# ════════════════════════════════════════════════════════
# 这个文件原来还画开场地图的图 A（三张卡）和图 B（十行对照表）。
# 2026-08-31 Chris 判定那两张「都不重要」—— 这份材料是**自读复习和投屏两用**的，
# 而开场地图对两种用法都是绕路：自读的人有目录，听课的人要的是立刻进主题。
# 于是第 0 节之前的东西整段删掉，正文第一屏就是下面这张根图。
# 那两张图的完整代码在 git 历史里（tag 之前的 commit 54fbd77）。

# 图 C · 两条出身 → 四个后果
# ════════════════════════════════════════════════════════
W=1000; H=428
q=[f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="GPU 与 TPU 的两条出身及其四个后果">']
q.append('<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
         '整门课只有一个根：两条出身不同，于是每一层都不同</text>')
cols=[(0,   "#1a73e8","#e8f0fe","GPU","图形处理器",
       ["要伺候大量互不相干的小任务","形状、访问模式、控制流全都无法预知",
        "→ 只能把面积花在「适应」上","cache 层次 · warp 调度 · 海量并发线程"],
       "什么都不能假设"),
      (512,"#1e8e3e","#e6f4ea","TPU","为矩阵乘定做的 ASIC",
       ["从第一天就只服务神经网络","形状规则、访问可预测、控制流静态",
        "→ 省下的面积全给计算单元","无 cache · 编译期定布局 · 巨大的 MXU"],
       "可以大胆假设")]
for x,col,lite,name,birth,lines,claim in cols:
    q.append(f'<rect x="{x}" y="34" width="488" height="196" rx="10" fill="{lite}" stroke="{col}"/>')
    q.append(f'<text class="svgnum" x="{x+20}" y="62" fill="{col}" style="font-size:17px">{name}</text>')
    q.append(f'<text class="svglbl" x="{x+20}" y="82" fill="{col}">出身：{birth}</text>')
    for k,t in enumerate(lines):
        q.append(f'<text class="svgsm" x="{x+20}" y="{108+k*20}" fill="#202124" '
                 f'style="font-size:11.5px">{t}</text>')
    q.append(f'<rect x="{x+20}" y="192" width="200" height="26" rx="13" fill="{col}"/>')
    q.append(f'<text class="svglbl" x="{x+120}" y="209" text-anchor="middle" fill="#fff">{claim}</text>')
q.append('<text class="svgsm" x="500" y="140" text-anchor="middle" fill="#80868b" '
         'style="font-size:15px">vs</text>')
# 四个后果
OUT=[("§2 内存","有 cache vs 无 cache","#1a73e8"),
     ("§3 计算","SM/warp vs MXU/VPU","#9334e6"),
     ("§4 互联","NVLink 域 vs ICI torus","#e8710a"),
     ("§5 范式","人做决定 vs 编译器做决定","#1e8e3e")]
q.append('<text class="svglbl" x="0" y="264" fill="#202124">这一条假设的四个后果 —— 也就是本课的四节</text>')
for i,(t,s,c) in enumerate(OUT):
    x=i*252
    q.append(f'<path d="M500 232 C500 250 {x+118} 250 {x+118} 276" fill="none" stroke="{c}" '
             f'stroke-width="1.6" opacity=".5"/>')
    q.append(f'<rect x="{x}" y="278" width="236" height="58" rx="8" fill="#fff" stroke="{c}"/>')
    q.append(f'<text class="svgnum" x="{x+16}" y="302" fill="{c}">{t}</text>')
    q.append(f'<text class="svgsm" x="{x+16}" y="322">{s}</text>')
q.append('<rect x="0" y="356" width="1000" height="56" rx="8" fill="#fef7e0" stroke="#f9ab00"/>')
q.append('<text class="svglbl" x="20" y="380" fill="#7a5000">'
         '⭐ 这不是「两种风格」，是同一条基因的四次显形</text>')
q.append('<text class="svgsm" x="20" y="399" fill="#7a5000">'
         '第 5 节回来验证它：当假设成立，TPU 赢在省下来的面积；当假设不成立（变长序列、MoE 路由），代价也全在那儿</text>')
q.append('</svg>')
io.open('figC.svg','w',encoding='utf-8').write('\n'.join(q)); print('figC ok')
