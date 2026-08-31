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

# 图 C · 两条出身 → 四个后果 → 两段课
# ════════════════════════════════════════════════════════
# ⭐ 2026-08-31 第二版。这张图现在要扛三件事，不只是「根」：
#   ① **统一口径**。同一个骨架原来有三套词在跑 —— 页头写「放得下吗 / 算得动吗
#      / 连得上吗 / 谁说了算」，这张图写「内存 / 计算 / 互联 / 范式」，
#      各节 h2 又是第三套。读者得自己在脑子里做映射。现在每个后果框
#      **第一行用页头那套问句、第二行用机制词**，这张图就是那本对照表。
#   ② **给第 1 节一个位置**。它不是「后果」，是**量具** —— 所以单独一条带，
#      放在出身和后果之间，形状上就跟四个后果区分开。
#   ③ **说出两段式**。这门课其实是「第 1–5 节讲硬件事实」+「第 6–9 节讲怎么
#      读数字」两段，而这条分界线原来只藏在第 6 节的开场白里，太晚 ——
#      读者读完第 5 节会以为课结束了。现在底带直接把它画出来。
W=1000; H=498
q=[f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
   'aria-label="GPU 与 TPU 的两条出身、四个后果，以及这门课的两段结构">']
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

# ── 量具：第 1 节 ─────────────────────────────────────────
# 形状故意跟下面四个后果不一样（一条通栏虚线带 vs 四个实框）——
# 「它不是后果」这件事要靠版式说，写一行小字说没人看。
q.append('<path d="M500 232 v12" stroke="#80868b" stroke-width="1.2"/>')
q.append('<rect x="0" y="246" width="1000" height="36" rx="8" fill="#f1f3f4" '
         'stroke="#9aa0a6" stroke-dasharray="4 3"/>')
q.append('<text class="svgnum" x="16" y="269" fill="#3c4043">第 1 节</text>')
q.append('<text class="svglbl" x="86" y="269" fill="#3c4043">'
         '先架量具 —— 两套坐标，外加一条叫 312 的线。它不是后果，是后面每一节都要用的尺子</text>')

# ── 四个后果 ──────────────────────────────────────────────
OUT=[("第 2 节　放得下吗","内存 —— 有 cache　vs　无 cache","#1a73e8"),
     ("第 3 节　算得动吗","计算 —— SM / warp　vs　MXU / VPU","#9334e6"),
     ("第 4 节　连得上吗","互联 —— NVLink 域　vs　ICI 环面","#e8710a"),
     ("第 5 节　谁说了算","范式 —— 人决定　vs　编译器决定","#1e8e3e")]
q.append('<text class="svglbl" x="0" y="308" fill="#202124">'
         '这一条假设的四个后果 —— 不是「两种风格」，是同一条基因的四次显形</text>')
for i,(t,sub,c) in enumerate(OUT):
    x=i*252
    q.append(f'<path d="M500 282 C500 300 {x+118} 300 {x+118} 320" fill="none" stroke="{c}" '
             f'stroke-width="1.6" opacity=".5"/>')
    q.append(f'<rect x="{x}" y="322" width="236" height="58" rx="8" fill="#fff" stroke="{c}"/>')
    q.append(f'<text class="svgnum" x="{x+16}" y="346" fill="{c}">{t}</text>')
    q.append(f'<text class="svgsm" x="{x+16}" y="366">{sub}</text>')
q.append('<text class="svgsm" x="0" y="398" fill="#80868b">'
         '第 5 节回过头验证这条假设：成立时 TPU 赢在省下来的面积；不成立时（变长序列、MoE 路由），代价也全在那儿</text>')

# ── 底带：这门课是两段 ────────────────────────────────────
q.append('<rect x="0" y="416" width="1000" height="76" rx="8" fill="#fef7e0" stroke="#f9ab00"/>')
q.append('<text class="svglbl" x="20" y="440" fill="#7a5000">'
         '⭐ 这门课是两段，不是九节 —— 上面这张图只画完了第一段</text>')
q.append('<text class="svgsm" x="20" y="461" fill="#7a5000">'
         '第 1–5 节｜硬件事实：两条出身，四个后果。回答的是「两边有什么不一样」</text>')
q.append('<text class="svgsm" x="20" y="480" fill="#7a5000">'
         '第 6–9 节｜怎么读数字：怎么比才有意义 · 用在一组实测上 · 我自己撤回过的八笔 · 最后才敢下的结论。'
         '换掉这两块硬件，带得走的是这一半</text>')
q.append('</svg>')

# ⛔ 文字超出 viewBox 是**静默**裁掉的，渲染图上看不出「被截了」和「本来就短」的区别。
#    最长那行 svgsm 约 62 个中文字 × 10.5px ≈ 651px + x=20 —— 离 1000 还远，安全。
io.open('figC.svg','w',encoding='utf-8').write('\n'.join(q)); print('figC ok')
