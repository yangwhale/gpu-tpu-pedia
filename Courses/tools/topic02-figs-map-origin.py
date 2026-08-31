# -*- coding: utf-8 -*-
import io
GiB=2**30; HBM=206_000_000_000
CHIP=HBM/GiB; DEV=CHIP/2; ALLOC=94.74; RES=DEV-ALLOC

# ════════════════════════════════════════════════════════
# 图 A · 本课地图：专题一那八步 → 各撞哪堵墙 → 本课四个答案
# ════════════════════════════════════════════════════════
ANS=[("第 2 节","放得下","内存","#1a73e8"),
     ("第 3 节","算得动","计算单元","#9334e6"),
     ("第 4 节","卡间说话","互联","#e8710a"),
     ("第 5 节","谁做决定","范式","#1e8e3e")]
# 三层里的另外两层：连线只连中间那层（八步各撞哪堵墙），
# 上下两层不接连线 —— 它们不回答「这一步卡在哪」，但课上真实存在。
TIER_TOP=[("第 0 节","把 94.74 还清楚 ＋ 立那条假设"),
          ("第 1 节","两边的硬件全景（不回答问题，但后面每节都要用）")]
TIER_BOT=[("第 6 节","怎么比才不算耍赖"),
          ("第 7 节","两边的实测"),
          ("第 8–9 节","数是怎么核的 ＋ 收尾")]
# 时间预算：全课 60 分钟。这是**建议值不是实测**，写出来是为了让台上
# 知道超了要砍哪儿 —— 不写的话必然在第 2 节讲嗨了，后面全部挤掉。
BUDGET=[("地图",2),("第 0 节",6),("第 1 节",8),("第 2 节",15),("第 3 节",10),
        ("第 4 节",5),("第 5 节",5),("第 6 节",3),("第 7 节",4),
        ("第 8 节",0),("第 9 节",2)]
assert sum(m for _,m in BUDGET)==60, "预算加起来不是 60 分钟"
STEPS=[  # (步骤, 它到底卡在哪, 目标索引)
 ("① 入口 · embedding 查表","算术强度为零，纯搬运",0),
 ("② 注意力 · 中间量","seq×seq 大到不该落 HBM",0),
 ("③ 注意力 · softmax","逐元素，矩阵单元用不上",1),
 ("④ Dense MLP","大矩阵乘 —— 但形状要对齐",1),
 ("⑤ MoE 路由 · top-k","数据相关，不规则",1),
 ("⑥ MoE dispatch / combine","all-to-all，每对都要说话",2),
 ("⑦ 层间 / 数据并行","all-gather · reduce-scatter",2),
 ("⑧ 出口 · logits","巨大的短命中间量",0),
]
RH=46; TOP=126; LX=0; LW=330; RX=690; RW=300
H=TOP+RH*len(STEPS)+346   # 底部还要塞一条 60 分钟时间轴
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" aria-label="本课地图：专题一每一步各撞哪堵硬件墙">']
p.append('<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
         '本课地图 —— 不逐个介绍硬件，而是拿专题一那张需求清单，一条一条去硬件上找答案</text>')
p.append('<text class="svgsm" x="0" y="36">左边是上一课走过的每一步，右边是这一课的四节。连线颜色 = 这一步的账最后算到哪一节</text>')
p.append(f'<text class="svgsm" x="{LX}" y="{TOP-12}">专题一 · 一个 token 走过的路</text>')
p.append(f'<text class="svgsm" x="{RX}" y="{TOP-12}">专题二 · 中间这四节回答上面八步（另有六节在上下两端）</text>')

# 右侧四个答案块（按连线数分配高度）
cnt=[sum(1 for s in STEPS if s[2]==i) for i in range(4)]
ry=TOP; rpos=[]
for i,(tag,name,sub,col) in enumerate(ANS):
    hgt=max(52, RH*cnt[i]-10) if cnt[i] else 52
    p.append(f'<rect x="{RX}" y="{ry}" width="{RW}" height="{hgt}" rx="8" fill="{col}"/>')
    p.append(f'<text class="svgnum" x="{RX+16}" y="{ry+hgt/2-2}" fill="#fff">{tag} · {name}</text>')
    p.append(f'<text class="svgsm" x="{RX+16}" y="{ry+hgt/2+15}" fill="#ffffffcc">{sub}</text>')
    rpos.append(ry+hgt/2); ry+=hgt+10
# §5 没有直连步骤时给它一条来自⑤的虚线
for i,(name,why,t) in enumerate(STEPS):
    y=TOP+i*RH+RH/2-4
    col=ANS[t][3]
    p.append(f'<rect x="{LX}" y="{y-17}" width="{LW}" height="34" rx="6" fill="#fff" stroke="#dadce0"/>')
    p.append(f'<text class="svglbl" x="{LX+12}" y="{y-1}" fill="#202124">{name}</text>')
    p.append(f'<text class="svgsm" x="{LX+12}" y="{y+13}">{why}</text>')
    x1,x2=LX+LW,RX; ym=rpos[t]
    p.append(f'<path d="M{x1} {y} C{x1+120} {y} {x2-120} {ym} {x2} {ym}" fill="none" '
             f'stroke="{col}" stroke-width="1.8" opacity=".55"/>')
    p.append(f'<circle cx="{x1}" cy="{y}" r="3" fill="{col}"/>')
# ⑤ 额外连到 §5
y5=TOP+4*RH+RH/2-4
p.append(f'<path d="M{LX+LW} {y5} C{LX+LW+150} {y5+40} {RX-150} {rpos[3]} {RX} {rpos[3]}" fill="none" '
         f'stroke="{ANS[3][3]}" stroke-width="1.6" stroke-dasharray="5 4" opacity=".7"/>')
# ── 右栏上下两层（不接连线）────────────────────────────────────────
TB_Y=TOP-52
for i,(tag,txt) in enumerate(TIER_TOP):
    p.append(f'<rect x="{RX}" y="{TB_Y+i*22}" width="{RW}" height="20" rx="4" '
             f'fill="#f1f3f4" stroke="#dadce0"/>')
    p.append(f'<text class="svgsm" x="{RX+10}" y="{TB_Y+i*22+14}" fill="#5f6368">'
             f'<tspan font-weight="700">{tag}</tspan>　{txt}</text>')
BB_Y=ry+6
for i,(tag,txt) in enumerate(TIER_BOT):
    p.append(f'<rect x="{RX}" y="{BB_Y+i*22}" width="{RW}" height="20" rx="4" '
             f'fill="#f1f3f4" stroke="#dadce0"/>')
    p.append(f'<text class="svgsm" x="{RX+10}" y="{BB_Y+i*22+14}" fill="#5f6368">'
             f'<tspan font-weight="700">{tag}</tspan>　{txt}</text>')
p.append(f'<text class="svgsm" x="{RX}" y="{TB_Y-6}" fill="#9aa0a6">打底 —— 不回答问题，但后面每节都要用</text>')
p.append(f'<text class="svgsm" x="{RX}" y="{BB_Y+len(TIER_BOT)*22+12}" fill="#9aa0a6">怎么验 —— 上面那些说法凭什么信</text>')

yb=max(TOP+RH*len(STEPS)+14, BB_Y+len(TIER_BOT)*22+26)
p.append(f'<line x1="0" y1="{yb}" x2="1000" y2="{yb}" stroke="#e8eaed"/>')
p.append(f'<text class="svglbl" x="0" y="{yb+26}" fill="#d93025">'
         f'⚠️ 注意左边有八行 —— 这一课不能只拿 ① embedding 当例子</text>')
p.append(f'<text class="svgsm" x="0" y="{yb+46}">'
         f'它是八步里唯一算术强度为零的一步。只用它，整课会听起来像「TPU 的问题都是访存布局」，而真实的账不是这样</text>')
# ── 图底：60 分钟时间轴 ──────────────────────────────────────────────
TLY=yb+72
p.append(f'<text class="svglbl" x="0" y="{TLY-8}" fill="#202124">建议时间分配（共 60 分钟）</text>')
p.append(f'<text class="svgsm" x="1000" y="{TLY-8}" text-anchor="end" fill="#d93025">'
         f'⚠️ 建议值不是实测 —— 第 2 节光图注就压着 15 分钟，超了先砍图 2-2 和 2-5</text>')
TLW=1000; xx=0.0
SHADE={0:"#9aa0a6",1:"#9aa0a6"}
for i,(tag,m) in enumerate(BUDGET):
    w=TLW*m/60.0
    if m:
        c = ANS[i-2][3] if 2 <= i <= 5 else "#9aa0a6"
        p.append(f'<rect x="{xx:.1f}" y="{TLY}" width="{w-2:.1f}" height="24" rx="3" fill="{c}"/>')
        p.append(f'<text class="svgsm" x="{xx+w/2:.1f}" y="{TLY+16}" text-anchor="middle" '
                 f'fill="#fff" style="font-size:9.5px">{m}</text>')
        p.append(f'<text class="svgsm" x="{xx+w/2:.1f}" y="{TLY+38}" text-anchor="middle" '
                 f'fill="#5f6368" style="font-size:9.5px">{tag.replace(" 节","")}</text>')
    xx+=w
p.append(f'<text class="svgsm" x="0" y="{TLY+56}" fill="#5f6368">'
         f'第 8 节（数是怎么核的）<tspan font-weight="700">不占时间</tspan> —— 发下去自己看。'
         f'中间四段带颜色的就是上面右栏那四个答案，'
         f'<tspan font-weight="700">它们合起来 35 分钟，占了一多半</tspan>。</text>')
p.append('</svg>')
io.open('figA.svg','w',encoding='utf-8').write('\n'.join(p)); print('figA',H)

# ════════════════════════════════════════════════════════
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
