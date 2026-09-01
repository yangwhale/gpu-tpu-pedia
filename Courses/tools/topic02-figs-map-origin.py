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

# 图 C · 两条出身 → 一条主线（FlashAttention 走一遍）
# ════════════════════════════════════════════════════════
# ⭐ 2026-09-01 第三版。**下半部分整个换掉了。**
#
# 起因是一句原话：「前后的内容并没有完整地串起来，有点乱，不知道在讲啥，
# 一会东一会西的，没有目的性。」
#
# 第二版的下半部分是「量具 ＋ 四个后果 ＋ 两段课」—— 那是把**硬件部件清单**
# 画了出来。清单能回答「这门课覆盖了什么」，但回答不了「我为什么要按这个顺序读」。
# 于是读者每一节都在换个部件重讲一遍两边差别，读完不知道刚才在走一条什么路。
#
# 第三版换成**一条路**：同一个 FlashAttention，两边各跑一遍，看它从哪儿分岔。
# 三个设计决定：
#   ① **出身层原样保留**。出身回答「为什么会分岔」，旅程回答「分岔之后怎么走」,
#      两层不重复。删掉出身，后面那个分岔点就成了一个没来由的事实。
#   ② **分岔点必须只有一个**，而且要用颜色顶出来。这门课真正的结论是
#      「两边的差别不是十几处，是一处」—— 画成四个并列的后果框，恰好把
#      这句话的反面画了出来。
#   ③ **第 4 节画成旁支，不画进主线**。grep 过：那一节里 FlashAttention
#      出现 0 次（§2 二十五次、§3 二十一次、§5 九次）—— 它单卡就跑完了，
#      撞不到卡间。与其让人读到那儿觉得跑题，不如在地图上就标成岔路。
#
# ⚠️ 同一时间我在 HTML 里另写过一个文字版故事线，跟这张图讲的是同一件事 ——
#    当天就删了。**同一个职责不要两个载体**，它们迟早各自漂移。
W=1000; H=560
q=[f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
   'aria-label="GPU 与 TPU 的两条出身，以及这门课的主线：'
   '同一个 FlashAttention 在两边各跑一遍，分岔点只有一处">']
# 这张图是独立 SVG（不走显微镜那套共用 defs），箭头 marker 得自带一个。
q.append('<defs><marker id="mapArrow" markerWidth="8" markerHeight="8" refX="6.5" '
         'refY="4" orient="auto"><path d="M0,0.8 L8,4 L0,7.2 z" fill="#9aa0a6"/>'
         '</marker></defs>')
q.append('<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
         '整门课只有一个根：两条出身不同，于是每一层都不同</text>')
cols=[(0,   "#1a73e8","#e8f0fe","GPU","图形处理器",
       ["要伺候大量互不相干的小任务","形状、访问模式、控制流全都无法预知",
        "→ 面积大半花在「应付各种情况」上","cache 层次 · warp 调度 · 海量并发线程"],
       "不知道你要跑什么　→　处处留一手"),
      (512,"#1e8e3e","#e6f4ea","TPU","为矩阵乘定做的 ASIC",
       ["从第一天就只服务神经网络","形状规则、访问可预测、控制流静态",
        "→ 省下的面积全给计算单元","无 cache · 编译期定布局 · 巨大的 MXU"],
       "早就知道你要跑什么　→　一手都不留")]
for x,col,lite,name,birth,lines,claim in cols:
    q.append(f'<rect x="{x}" y="34" width="488" height="196" rx="10" fill="{lite}" stroke="{col}"/>')
    q.append(f'<text class="svgnum" x="{x+20}" y="62" fill="{col}" style="font-size:17px">{name}</text>')
    q.append(f'<text class="svglbl" x="{x+20}" y="82" fill="{col}">出身：{birth}</text>')
    for k,t in enumerate(lines):
        q.append(f'<text class="svgsm" x="{x+20}" y="{108+k*20}" fill="#202124" '
                 f'style="font-size:11.5px">{t}</text>')
    q.append(f'<rect x="{x+20}" y="192" width="292" height="26" rx="13" fill="{col}"/>')
    q.append(f'<text class="svglbl" x="{x+166}" y="209" text-anchor="middle" fill="#fff">{claim}</text>')
q.append('<text class="svgsm" x="500" y="140" text-anchor="middle" fill="#80868b" '
         'style="font-size:15px">vs</text>')

# ── 主线：一个 FlashAttention，两边各跑一遍 ────────────────
q.append('<path d="M500 232 v14" stroke="#80868b" stroke-width="1.2"/>')
q.append('<text class="svglbl" x="0" y="266" fill="#202124" style="font-size:13px">'
         '于是这门课只做一件事：<tspan font-weight="700" fill="#d93025">'
         '同一个 FlashAttention，两边各跑一遍，看它从哪儿开始分岔</tspan></text>')

# 六站。SW/GAP 让六个框正好铺满 1000：6×153 + 5×16.4 = 1000。
# ⚠️ 框宽 153、内边距 12 → 正文只剩 129px，10.5px 的中文一行放得下 12 个字。
#    下面每条 body 都按 ≤12 字写的，加长会**静默**溢出到隔壁框上。
SW, SX = 153, 169.4
STOP=[("出发前","为什么是它","#5f6368",
       ["S 矩阵 32 GiB","两边都放不下","→ 别把 S 写出来"]),
      ("第 0–1 节","先架量具","#1a73e8",
       ["算力 ÷ 带宽","312.6 对 312.5","两边胃口一样"]),
      ("第 2 节","⭐ 分岔在这里","#d93025",
       ["同一次访存","一边有 cache","一边没有"]),
      ("第 3 节","形状对不对得齐","#9334e6",
       ["一条指令吃多大","两边差 16 倍","并排走完全程"]),
      ("第 5 节","分岔的本质","#1e8e3e",
       ["搬运谁来安排","运行时 · 编译期","三次出场合一"]),
      ("第 6–9 节","跑完之后","#e8710a",
       ["拿到两个数","能不能比？","以及我算错的"])]
for i,(n,t,c,body) in enumerate(STOP):
    x=round(i*SX,1)
    fork = (c=="#d93025")                       # 分岔点那一格：加粗描边 ＋ 浅红底
    q.append(f'<rect x="{x}" y="282" width="{SW}" height="92" rx="10" '
             f'fill="{"#fce8e6" if fork else "#fff"}" stroke="{c}" '
             f'stroke-width="{2.2 if fork else 1.2}"/>')
    q.append(f'<text class="svgsm" x="{x+12}" y="300" fill="#80868b">{n}</text>')
    q.append(f'<text class="svglbl" x="{x+12}" y="319" fill="{c}" '
             f'style="font-size:12.5px">{t}</text>')
    for k,b in enumerate(body):
        q.append(f'<text class="svgsm" x="{x+12}" y="{338+k*15}" fill="#3c4043">{b}</text>')
    if i:                                        # 站与站之间的箭头
        q.append(f'<path d="M{x-14} 328 h9" stroke="#9aa0a6" stroke-width="1.4" '
                 f'marker-end="url(#mapArrow)"/>')

# ── 旁支：第 4 节 ─────────────────────────────────────────
# 虚线 ＋ 灰底，形状上就跟主线六个框分开 —— 「这是岔路」要靠版式说，
# 写一行小字说没人看（这条经验沿用上一版那个量具带）。
q.append(f'<path d="M{round(3*SX+SW/2,1)} 374 v14 h-120 v14" fill="none" '
         'stroke="#9aa0a6" stroke-width="1.2" stroke-dasharray="4 3"/>')
q.append('<rect x="0" y="402" width="1000" height="54" rx="8" fill="#f1f3f4" '
         'stroke="#9aa0a6" stroke-dasharray="4 3"/>')
# svgnum 是 mono 的，中文在等宽字体下字距会被拉得很开 —— 只给数字用，标签用 svglbl。
q.append('<text class="svglbl" x="16" y="424" fill="#3c4043" style="font-size:12.5px">'
         '第 4 节｜岔路</text>')
q.append('<text class="svglbl" x="126" y="424" fill="#3c4043">'
         '一张卡装不下的时候 —— NVLink 域　vs　ICI 环面</text>')
q.append('<text class="svgsm" x="16" y="444" fill="#5f6368">'
         'FlashAttention 在这一节不在场：它单卡就跑完了，撞不到卡间。主角换成 MoE 的 dispatch / combine '
         '—— 只想跟着主线走，这一节可以先跳过，第 5 节回主线</text>')

# ── 落点：只有一个分岔点 ──────────────────────────────────
q.append('<rect x="0" y="474" width="1000" height="80" rx="10" fill="#fce8e6" '
         'stroke="#d93025" stroke-width="1.6"/>')
q.append('<text class="svglbl" x="20" y="500" fill="#a50e0e" style="font-size:13px">'
         '⭐ 如果只记一件事：两边的差别不是十几处，'
         '<tspan font-weight="700">是一处</tspan></text>')
q.append('<text class="svgsm" x="20" y="521" fill="#3c4043" style="font-size:11.5px">'
         '把一块数据从 HBM 搬进片上，<tspan font-weight="700">谁来安排</tspan>。'
         'GPU 那边有 cache 在运行时替你猜，TPU 那边没有、编译期就排死了。</text>')
q.append('<text class="svgsm" x="20" y="540" fill="#3c4043" style="font-size:11.5px">'
         '形状要对齐到 8 × 128、一条指令吃多大、kernel 谁来写、连一台机器要付什么代价 '
         '——&#160;<tspan font-weight="700">全都是这一处往下长出来的</tspan>。</text>')
q.append('</svg>')

# ⛔ 文字超出 viewBox 是**静默**裁掉的，渲染图上看不出「被截了」和「本来就短」的区别。
#    最长那行是旁支那句 svgsm，约 66 个中文字 × 10.5px ≈ 693px + x=16 —— 离 1000 还有富余。
io.open('figC.svg','w',encoding='utf-8').write('\n'.join(q)); print('figC ok')
