# -*- coding: utf-8 -*-
import io
GiB=2**30; HBM=206_000_000_000
CHIP=HBM/GiB; DEV=CHIP/2; ALLOC=94.74; RES=DEV-ALLOC

# ════════════════════════════════════════════════════════
# 图 A · 本课地图：专题一算出来的每条结论 → 撞在哪个硬件部件 → 本课第几节
# ════════════════════════════════════════════════════════
# ⭐ 这张图重画过一次。旧版是左右两栏 + 八条贝塞尔曲线（左：专题一的八个
#    「步骤」，右：本课四节），有三个毛病，Chris 在 2026-08-31 全指出来了：
#
#    ① **接不上专题一。** 左栏抄的是步骤名，不是专题一**算出来的结论**。
#       学生刚花一小时算出一串数，这张图一个数都没接过来 —— 那就不叫承接，
#       叫重新列一遍目录。现在左栏每一条都必须带一个专题一原文里的数。
#    ② **看不出跟硬件有什么关系。** 中间是空的，只有八条曲线飞过去。
#       现在中间那栏就是**硬件部件本身**（HBM 带宽 / 矩阵单元 / 卡间链路 …），
#       「这图跟硬件什么关系」这个问题在图上直接有答案。
#    ③ **两条映射是错的**，而且专题一原文早就写对了：
#       · 注意力：旧图只画了「中间量 → 内存」。可专题一算的是
#         「128K 下 attention 的平方项吃掉 81.8% 的前向算力」—— 它在长序列
#         下首先是个**算力**问题。现在拆成两行（见底部注解）。
#       · MoE：旧图画成「路由 → 算得动」。可专题一的原话是
#         「MoE 省的是算力（18.3×），**不省显存（0×）**」，那 634 B 参数
#         这一步不参与计算、但一个字节都不能从显存里拿走 —— 它首先是个
#         **内存**问题。旧图整条漏了。
#
#    另外曲线换成了按节排序的表格：十条连线互相穿插，正是「一会东一会西」
#    的来源。排好序之后连线全是水平的，思路是一条直线。
#    60 分钟时间轴也从这里搬走了 —— 那是给台上看的脚手架，不该占学生图。
ANS=[("第 2 节","放得下","内存","#1a73e8"),
     ("第 3 节","算得动","计算单元","#9334e6"),
     ("第 4 节","卡间说话","互联","#e8710a"),
     ("第 5 节","谁做决定","范式","#1e8e3e")]
TIER_TOP="第 0 节　把 94.74 还清楚 ＋ 立那条假设　　·　　第 1 节　两边的硬件全景"
TIER_BOT="第 6 节　怎么比才不算耍赖　　·　　第 7 节　两边的实测　　·　　第 8–9 节　数是怎么核的 ＋ 收尾"
# (节索引, 专题一这一步, 专题一算出来的那个数, 撞在哪个部件)
# ⛔ 第三列**必须**是专题一原文里出现过的数 —— 这一列是这张图存在的理由。
ROWS=[
 (0,"入口 · embedding 查表",      "算术强度 <b>0</b> —— 一次乘加都没有",              "HBM 带宽"),
 (0,"注意力 · 朴素实现",          "seq×seq 中间量落 HBM，强度只有 <b>64</b>",         "HBM 带宽"),
 (0,"MoE · 那 634 B 没参与计算的权重","省的是算力（<b>18.3×</b>），不省显存（<b>0×</b>）","HBM 容量"),
 (0,"出口 · logits",              "<b>31.56 GiB</b> 的短命张量",                      "HBM 容量"),
 (1,"注意力 · 省掉中间量之后",     "128K 下吃掉 <b>81.8%</b> 的前向算力",              "矩阵单元"),
 (1,"Dense MLP",                  "强度 <b>3,584</b> —— 又大又规整",                  "矩阵单元"),
 (1,"softmax",                    "逐元素，强度约 <b>1</b>，矩阵单元用不上",           "向量单元"),
 (2,"MoE · dispatch / combine",   "all-to-all，每对卡都要说话",                        "卡间链路"),
 (2,"层间 · 数据并行",            "all-gather · reduce-scatter",                       "卡间链路"),
 (3,"MoE · 路由 top-k",           "每个 token 选哪 8 个专家，<b>运行时才知道</b>",     "形状谁来定"),
]
def B(s):
    """把 <b> 换成 SVG 认得的 <tspan>。

    ⛔ SVG 的 <text> 里**没有** <b>。写了不会报错，浏览器会把它当未知元素、
       提前结束当前 text，于是那一行之后的内容全部掉到图底变成裸字 ——
       症状是「整张图只画出第一行，底下多出一长条黑字」。踩过一次，别再写 <b>。"""
    return s.replace("<b>", '<tspan font-weight="700">').replace("</b>", "</tspan>")

CNT=[sum(1 for r in ROWS if r[0]==i) for i in range(4)]
assert CNT==[4,3,2,1], "行数变了，底部那句「四条撞内存、三条撞计算单元」要跟着改：%s" % CNT
assert [r[0] for r in ROWS]==sorted(r[0] for r in ROWS), "ROWS 必须按节排好序，否则连线又会交叉"

RH=40; GAP=10; TOP=132
LX=0;   LW=386
MX=400; MW=178
RX=596; RW=404
TBL_H=RH*len(ROWS)+GAP*3
H=TOP+TBL_H+142
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" '
   f'aria-label="本课地图：专题一算出来的每条结论各撞在哪个硬件部件上，以及对应本课第几节">']
p.append('<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
         '本课地图 —— 专题一交出的是一张<tspan font-weight="700">需求清单</tspan>；'
         '每一条都撞在硬件的某一个部件上，<tspan font-weight="700">撞在哪，就在哪一节讲</tspan></text>')
p.append('<text class="svgsm" x="0" y="36">'
         '所以这一课不逐个介绍硬件参数 —— 是拿着上一课自己算出来的数，一条一条去硬件上找那堵墙</text>')
# 打底条
p.append(f'<rect x="0" y="52" width="1000" height="22" rx="4" fill="#f1f3f4" stroke="#dadce0"/>')
p.append(f'<text class="svgsm" x="12" y="67" fill="#5f6368">'
         f'<tspan font-weight="700">打底</tspan>（不回答清单上的问题，但后面每节都要用）　　{TIER_TOP}</text>')
# 列头
for x,t in ((LX,"专题一算出来的结论"),(MX,"它撞在哪个部件上"),(RX,"本课第几节 · 讲这个部件")):
    p.append(f'<text class="svgsm" x="{x}" y="{TOP-10}" fill="#9aa0a6">{t}</text>')

y=TOP; gi=0
for i,(tag,name,sub,col) in enumerate(ANS):
    n=CNT[i]; blk=RH*n-4
    # 右：本节（跨它管的那几行）
    p.append(f'<rect x="{RX}" y="{y}" width="{RW}" height="{blk}" rx="8" fill="{col}"/>')
    p.append(f'<text class="svgnum" x="{RX+18}" y="{y+blk/2-1}" fill="#fff">{tag} · {name}</text>')
    p.append(f'<text class="svgsm" x="{RX+18}" y="{y+blk/2+16}" fill="#ffffffcc">{sub}</text>')
    for k in range(n):
        r=ROWS[gi+k]; ry=y+k*RH
        cy=ry+RH/2-2
        p.append(f'<rect x="{LX}" y="{ry}" width="{LW}" height="{RH-6}" rx="6" '
                 f'fill="#fff" stroke="#dadce0"/>')
        p.append(f'<text class="svglbl" x="{LX+12}" y="{cy-3}" fill="#202124">{r[1]}</text>')
        p.append(f'<text class="svgsm" x="{LX+12}" y="{cy+12}">{B(r[2])}</text>')
        # 中：硬件部件本身 —— 这一列是「这图跟硬件什么关系」的答案
        p.append(f'<rect x="{MX}" y="{ry+4}" width="{MW}" height="{RH-14}" rx="{(RH-14)/2}" '
                 f'fill="#fff" stroke="{col}" stroke-width="1.6"/>')
        p.append(f'<text class="svglbl" x="{MX+MW/2}" y="{cy+2}" text-anchor="middle" '
                 f'fill="{col}">{r[3]}</text>')
        # 排过序，所以连线全是水平的 —— 没有一根交叉
        p.append(f'<path d="M{LX+LW} {cy-2} H{MX}" stroke="{col}" stroke-width="1.4" opacity=".45"/>')
        p.append(f'<path d="M{MX+MW} {cy-2} H{RX}" stroke="{col}" stroke-width="1.4" opacity=".45"/>')
    gi+=n; y+=blk+GAP
yb=y-GAP+10
p.append(f'<rect x="0" y="{yb}" width="1000" height="22" rx="4" fill="#f1f3f4" stroke="#dadce0"/>')
p.append(f'<text class="svgsm" x="12" y="{yb+15}" fill="#5f6368">'
         f'<tspan font-weight="700">怎么验</tspan>（上面这些说法凭什么信）　　{TIER_BOT}</text>')
yn=yb+46
p.append(f'<text class="svglbl" x="0" y="{yn}" fill="#7a5000">'
         f'⭐ 四节的顺序和长短不是我排的，是这张清单排的 —— '
         f'十条里<tspan font-weight="700">四条撞内存、三条撞计算单元</tspan>，所以第 2 节最长</text>')
p.append(f'<text class="svgsm" x="0" y="{yn+20}" fill="#5f6368">'
         f'⚠️ <tspan font-weight="700">注意力出现了两次，不是笔误</tspan>：朴素实现被带宽卡住（强度 64），'
         f'把中间量省掉之后才轮到算力（81.8%）—— 这一条正好就是第 2 节和第 3 节的分界线。</text>')
p.append(f'<text class="svgsm" x="0" y="{yn+38}" fill="#5f6368">'
         f'⚠️ <tspan font-weight="700">MoE 出现了三次</tspan>：权重撞内存、路由撞范式、dispatch 撞互联 —— '
         f'全表唯一横跨三节的一条。「MoE 是算力优化」是个常见误读，专题一那句原话是「省的是算力，不省显存」。</text>')
p.append('</svg>')
io.open('figA.svg','w',encoding='utf-8').write('\n'.join(p)); print('figA',H,CNT)

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
