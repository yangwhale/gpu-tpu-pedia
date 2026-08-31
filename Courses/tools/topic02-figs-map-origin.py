# -*- coding: utf-8 -*-
import io
GiB=2**30; HBM=206_000_000_000
CHIP=HBM/GiB; DEV=CHIP/2; ALLOC=94.74; RES=DEV-ALLOC

# ════════════════════════════════════════════════════════
# 图 A · 开场：清单上只有四个问题　｜　图 B · 完整对照（手册用）
# ════════════════════════════════════════════════════════
# ⭐ **这一段重画过两次，两次的病因不一样，都记在这儿。**
#
# 【第一次】原版是左右两栏 + 八条贝塞尔曲线（左：专题一的八个「步骤」，
#   右：本课四节）。Chris 2026-08-31 指出三个毛病：
#     ① **接不上专题一** —— 左栏抄的是步骤名，不是专题一**算出来的结论**。
#        学生刚花一小时算出一串数，这张图一个都没接过来，那不叫承接，
#        叫重新列一遍目录。现在每一条都必须带一个专题一原文里的数。
#     ② **看不出跟硬件有什么关系** —— 中间是空的，只有曲线飞过去。
#     ③ **两条映射是错的**，而专题一原文早就写对了：
#        · 注意力在长序列下首先是**算力**问题（「128K 下吃掉 81.8% 的前向算力」），
#          不是内存问题；朴素实现才是内存问题。所以它要**拆成两行**。
#        · MoE 首先是**内存**问题（「省的是算力 18.3×，**不省显存 0×**」，
#          那 634 B 不参与计算但一个字节都拿不走），旧图整条漏了。
#
# 【第二次 —— 就是现在这一版】上面那三条改完，得到的是一张十行三列的表。
#   内容对了，但 Chris 当天下午看了一眼就说：**「太复杂了，一开场大家就看不懂，
#   听众全给弄跑了。」** 而且**中间那一栏不同的功能颜色还一样** —— 因为那一栏
#   染的是「第几节」的色，于是「HBM 带宽」和「HBM 容量」同蓝、「矩阵单元」和
#   「向量单元」同紫，看上去像在说废话。
#
#   诊断：**那张表本身没错，错在拿它当开场。**十行三列是一张**查得清的对照表**，
#   不是一张**十秒钟看得懂的开场图**。开场要回答的只有一句「这课讲什么」。
#
#   所以拆成两张，用的正是本页 §1.5 已经在用的那个分工 ——
#   **「台上讲的」和「发下去查的」是两张图**：
#     · **图 A**＝四张卡，一个问题一张，一张卡一个颜色一个部件。
#       颜色歧义随之消失：一张卡只对一个部件，不存在两个功能同色。
#     · **图 B**＝原来那张十行表，挪到这一节末尾，明写「手册用，台上不念」。
#   十秒钟看得懂的在前面，查得清的在后面，两个需求都不用让步。
ANS=[("第 2 节","放得下","内存","#1a73e8"),
     ("第 3 节","算得动","计算单元","#9334e6"),
     ("第 4 节","卡间说话","互联","#e8710a"),
     ("第 5 节","谁做决定","范式","#1e8e3e")]
TIER_TOP="第 0 节　立那条假设：两条出身不同　　·　　第 1 节　两边的硬件全景"
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

# ⛔ 中间那一栏**按部件染色，不按节次染色**。
#    2026-08-31 之前染的是节次色，于是「HBM 带宽 / HBM 容量」同蓝、
#    「矩阵单元 / 向量单元」同紫 —— 两个不同的东西长成一个样，比不上色更糟。
#    当时的补救是在图底写一行「看这一栏认字，不要认颜色」，那是**给病贴张纸**，
#    Chris 一眼就看穿了。真正的修法只有一个：**不同的部件给不同的颜色。**
#    只有 2、3 两节各管两个部件需要区分；4、5 两节各只有一个，沿用节次色即可。
PART={"HBM 带宽":"#1a73e8", "HBM 容量":"#12786f",      # 第 2 节 —— 蓝 / 青
      "矩阵单元":"#9334e6", "向量单元":"#d01884",      # 第 3 节 —— 紫 / 品红
      "卡间链路":"#e8710a", "形状谁来定":"#1e8e3e"}
assert {r[3] for r in ROWS} == set(PART), "ROWS 里出现了 PART 没有配色的部件：%s" % (
    {r[3] for r in ROWS} ^ set(PART))
assert len(set(PART.values()))==len(PART), "两个部件撞色了，这正是要修的那个毛病"

CNT=[sum(1 for r in ROWS if r[0]==i) for i in range(4)]
assert CNT==[4,3,2,1], "行数变了，底部那句「四条撞内存、三条撞计算单元」要跟着改：%s" % CNT
assert [r[0] for r in ROWS]==sorted(r[0] for r in ROWS), "ROWS 必须按节排好序，否则连线又会交叉"

# ── 图 A · 三张卡 ＋ 一条贯穿带 ─────────────────────────
# ⛔ 这张图的唯一任务是「十秒钟让人知道这课讲什么」。任何想往里加的东西，
#    先问一句：**它是不是必须在开场的头十秒出现？** 不是就去图 B。
#    一张卡只准有一个部件 —— 这是颜色歧义的根治办法，别再往回加。
#
# ⭐ **为什么第 5 节不是第四张卡**（2026-08-31，Chris：「谁在做决定
#    这个跟前三个感觉不是一个维度的呀」）。他是对的，而且这个错藏得很深：
#      · 前三张问的都是**「够不够」** —— 内存够不够、算力够不够、链路够不够。
#        同一个句式、同一个量纲，是一组**资源**问题。
#      · 第 5 节问的是**「谁说了算」**。它不是第四种资源，
#        它是**前三种资源在两边为什么长得不一样的原因** ——
#        也就是第 0 节那两次赌注落到范式上的那一步。
#    排成四张并列的卡，等于宣称它跟前三个同级；用一条**横贯三张卡底下的带**，
#    才画出它真正的位置。原来那个「虚线空框」是在同一张卡里暗示这件事，
#    暗示不如画出来。
#
# ⭐ **第 4 节为什么不叫「传得快吗」**（Chris 当时的提法）。句式他改对了 ——
#    「卡之间怎么说话」跟「放得下吗 / 算得动吗」不押韵，是我原来没排齐。
#    但**不能用「快」**：第 4 节自己的标题就是
#    **「连得多远，比连得多快更要紧」**，落点恰恰是带宽不是主角。
#    写成「传得快吗」会在开场就把这一节的结论说反。所以取「连得上吗」。
# （节次, 问题, 部件, 上一课的证据两行, 时长, 颜色）
CARD=[("第 2 节","放得下吗","HBM　容量 + 带宽",
       ("上一课结尾撞的那堵墙：","一个 device 只有 <b>94.74 GiB</b>"),"15 分钟","#1a73e8"),
      ("第 3 节","算得动吗","矩阵单元 · 向量单元",
       ("128K 下，attention 的平方项","吃掉 <b>81.8%</b> 的前向算力"),"10 分钟","#9334e6"),
      ("第 4 节","连得上吗","卡间链路",
       ("MoE 每个 token 要跨卡找","8 个专家 —— <b>all-to-all</b>"),"5 分钟","#e8710a")]
GRN="#1e8e3e"
CW=320; CG=(1000-CW*3)//2; Y0=76; CH=210
a=[f'<svg viewBox="0 0 1000 548" width="100%" role="img" '
   f'aria-label="开场地图：上一课那张需求清单只有三个「够不够」的问题，各撞在一个部件上；第 5 节问的是谁说了算，横贯这三节">']
a.append('<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
         '上一课交出的是一张<tspan font-weight="700">需求清单</tspan>。'
         '这一课不逐个介绍硬件参数 —— <tspan font-weight="700">拿着那张清单，一条一条去硬件上找答案</tspan></text>')
a.append('<text class="svgsm" x="0" y="36">'
         '清单上只有三个<tspan font-weight="700">「够不够」</tspan>的问题。'
         '每一个都撞在一个具体的部件上 —— <tspan font-weight="700">撞在哪，就是哪一节</tspan>。</text>')
a.append('<text class="svgsm" x="0" y="60" fill="#9aa0a6">'
         '每张卡下面那两行，是<tspan font-weight="700">上一课他们自己算出来的数</tspan> —— 这一课只负责告诉他们该去撞哪儿</text>')
for i,(tag,ques,part,(e1,e2),mins,col) in enumerate(CARD):
    x=i*(CW+CG)
    a.append(f'<rect x="{x}" y="{Y0}" width="{CW}" height="{CH}" rx="10" fill="#fff" stroke="{col}" stroke-width="1.6"/>')
    a.append(f'<path d="M{x} {Y0+10} a10 10 0 0 1 10 -10 h{CW-20} a10 10 0 0 1 10 10 v40 h{-CW} z" fill="{col}"/>')
    a.append(f'<text class="svgsm" x="{x+16}" y="{Y0+20}" fill="#ffffffcc">{tag}</text>')
    a.append(f'<text class="svgnum" x="{x+16}" y="{Y0+42}" fill="#fff" style="font-size:18px">{ques}</text>')
    a.append(f'<text class="svgsm" x="{x+16}" y="{Y0+76}" fill="#9aa0a6">上一课算出来的</text>')
    a.append(f'<text class="svgsm" x="{x+16}" y="{Y0+95}" fill="#202124">{B(e1)}</text>')
    a.append(f'<text class="svgsm" x="{x+16}" y="{Y0+113}" fill="#202124">{B(e2)}</text>')
    a.append(f'<line x1="{x+16}" y1="{Y0+131}" x2="{x+CW-16}" y2="{Y0+131}" stroke="#e8eaed"/>')
    a.append(f'<text class="svgsm" x="{x+16}" y="{Y0+151}" fill="#9aa0a6">于是撞在</text>')
    a.append(f'<rect x="{x+16}" y="{Y0+159}" width="{CW-32}" height="26" rx="13" fill="{col}"/>')
    a.append(f'<text class="svglbl" x="{x+CW/2}" y="{Y0+177}" text-anchor="middle" fill="#fff">{part}</text>')
    a.append(f'<text class="svgsm" x="{x+CW-16}" y="{Y0+201}" text-anchor="end" fill="#9aa0a6">⏱ {mins}</text>')
# ── 第 5 节：横贯三张卡底下的一条带，不是第四张卡 ──────────
# 三个小三角从每张卡底部指下来，画出「上面三节的答案都由它决定」。
Y5=Y0+CH+12
for i in range(3):
    cx=i*(CW+CG)+CW/2
    a.append(f'<path d="M{cx-6} {Y5-11} h12 l-6 9 z" fill="{GRN}" opacity=".55"/>')
a.append(f'<rect x="0" y="{Y5}" width="1000" height="76" rx="10" fill="#e6f4ea" stroke="{GRN}" stroke-width="1.6"/>')
a.append(f'<rect x="0" y="{Y5}" width="6" height="76" rx="3" fill="{GRN}"/>')
a.append(f'<text class="svgsm" x="20" y="{Y5+20}" fill="{GRN}">第 5 节　·　横贯上面三节，不是第四个问题</text>')
a.append(f'<text class="svgnum" x="20" y="{Y5+42}" fill="{GRN}" style="font-size:18px">谁说了算</text>')
a.append(f'<text class="svgsm" x="150" y="{Y5+41}" fill="#202124">'
         f'上面三个问<tspan font-weight="700">硬件够不够</tspan>；这一个问<tspan font-weight="700">形状由谁来定</tspan>'
         f'　——　选哪 8 个专家，<tspan font-weight="700">运行时才知道</tspan></text>')
a.append(f'<text class="svgsm" x="150" y="{Y5+61}" fill="#5f6368">'
         f'所以它<tspan font-weight="700">没有对应的部件</tspan>：它决定的不是哪个部件不够用，'
         f'而是<tspan font-weight="700">上面那三个部件由谁来编排</tspan> —— '
         f'也就是第 0 节那两次赌注，落到范式上的那一步</text>')
a.append(f'<text class="svgsm" x="984" y="{Y5+20}" text-anchor="end" fill="{GRN}">⏱ 5 分钟</text>')
# ⛔ 这条打底/验账带**必须两行**。挤成一行会在 1000 宽处被截掉尾巴，
#    而 SVG 文本溢出不会报错、渲染时也看不出来是「掉了字」还是「本来就这么写」。
YS=Y5+76+14
a.append(f'<rect x="0" y="{YS}" width="1000" height="42" rx="4" fill="#f1f3f4" stroke="#dadce0"/>')
a.append(f'<text class="svgsm" x="12" y="{YS+16}" fill="#5f6368">'
         f'<tspan font-weight="700">打底</tspan>（不回答清单上的问题，但后面每节都要用）：{TIER_TOP}</text>')
a.append(f'<text class="svgsm" x="12" y="{YS+33}" fill="#5f6368">'
         f'<tspan font-weight="700">验账</tspan>（上面这些说法凭什么信）：{TIER_BOT}</text>')
a.append(f'<text class="svglbl" x="0" y="{YS+70}" fill="#7a5000">'
         f'⭐ 前三节的长短不是我排的，是那张清单排的 —— 十条结论里'
         f'<tspan font-weight="700">四条撞内存、三条撞计算单元</tspan>，所以第 2 节最长</text>')
a.append(f'<text class="svgsm" x="0" y="{YS+89}" fill="#9aa0a6">'
         f'十条逐条对到哪个部件，在本节最后那张对照表里 —— 那张是发下去查的，台上不用讲</text>')
# ⏱ 预算原来写在图注里（灰色 13px），没人看。搬进图里、用正常字号。
# ⛔ 同样必须两行：一行排不下 1000 宽，而 SVG 溢出是静默截断。
YT=YS+104
a.append(f'<rect x="0" y="{YT}" width="1000" height="46" rx="4" fill="#fef7e0" stroke="#f9ab00"/>')
a.append(f'<text class="svglbl" x="14" y="{YT+19}" fill="#7a5000" style="font-size:12.5px">'
         f'⏱ <tspan font-weight="700">全课 60 分钟</tspan>　地图 2　·　第 0 节 2　·　第 1 节 10　·　'
         f'<tspan font-weight="700">第 2 节 17</tspan>　·　第 3 节 10　·　第 4 节 5　·　第 5 节 5</text>')
a.append(f'<text class="svglbl" x="14" y="{YT+37}" fill="#7a5000" style="font-size:12.5px">'
         f'　　第 6 节 3　·　第 7 节 4　·　第 8 节 0（发下去自己看）　·　第 9 节 2　　'
         f'——　<tspan font-weight="700">超了先砍图 2-2 和 2-5</tspan></text>')
assert YT+46 <= 548, "viewBox 高度不够，底下会被截掉：%s" % (YT+46)
a.append('</svg>')
io.open('figA.svg','w',encoding='utf-8').write('\n'.join(a)); print('figA 548 · 3 张卡 + 第 5 节贯穿带')

# ── 图 B · 完整对照（手册用，台上不念）─────────────────────
RH=40; GAP=10; TOP=110
LX=0;   LW=386
MX=400; MW=178
RX=596; RW=404
TBL_H=RH*len(ROWS)+GAP*3
H=TOP+TBL_H+120
p=[f'<svg viewBox="0 0 1000 {H}" width="100%" role="img" '
   f'aria-label="完整对照：专题一算出来的十条结论各撞在哪个硬件部件上，以及对应本课第几节">']
p.append('<text class="svglbl" x="0" y="16" fill="#202124" style="font-size:13.5px">'
         '完整对照（手册用，台上不念）—— 上一课那<tspan font-weight="700">十条结论</tspan>，'
         '逐条对到部件和节次</text>')
p.append('<text class="svgsm" x="0" y="36">'
         '开场那四张卡每张只举了一个例子。这张是把十条全摆出来 —— '
         '<tspan font-weight="700">「为什么第 2 节最长」在这里能数出来</tspan>。</text>')
# 列头
for x,t in ((LX,"专题一算出来的结论"),(MX,"它撞在哪个部件上（一色一部件）"),
            (RX,"本课第几节 · 讲这个部件")):
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
        # 中：硬件部件本身 —— 这一列是「这图跟硬件什么关系」的答案。
        #     染 pcol（部件色）不是 col（节次色）：一个颜色只代表一个部件。
        pcol=PART[r[3]]
        p.append(f'<rect x="{MX}" y="{ry+4}" width="{MW}" height="{RH-14}" rx="{(RH-14)/2}" '
                 f'fill="{pcol}"/>')
        p.append(f'<text class="svglbl" x="{MX+MW/2}" y="{cy+2}" text-anchor="middle" '
                 f'fill="#fff">{r[3]}</text>')
        # 排过序，所以连线全是水平的 —— 没有一根交叉。
        # 左半段跟着部件色，右半段跟着节次色：颜色在药丸处换一次，
        # 换的那一下就是这张图要说的事 —— 这个部件归那一节管。
        p.append(f'<path d="M{LX+LW} {cy-2} H{MX}" stroke="{pcol}" stroke-width="1.4" opacity=".55"/>')
        p.append(f'<path d="M{MX+MW} {cy-2} H{RX}" stroke="{col}" stroke-width="1.4" opacity=".55"/>')
    gi+=n; y+=blk+GAP
yn=y-GAP+34
# 这两条是**结论**不是补充说明，所以用正常字号（12.5px 深色），不是 10.5px 灰。
p.append(f'<text class="svglbl" x="0" y="{yn}" fill="#3c4043" style="font-size:12.5px">'
         f'⚠️ <tspan font-weight="700">注意力出现了两次，不是笔误</tspan>：朴素实现被带宽卡住（强度 64），'
         f'省掉中间量之后才轮到算力（81.8%）—— 这条正好是第 2 节和第 3 节的分界线。</text>')
p.append(f'<text class="svglbl" x="0" y="{yn+22}" fill="#3c4043" style="font-size:12.5px">'
         f'⚠️ <tspan font-weight="700">MoE 出现了三次</tspan>：权重撞内存、路由撞范式、dispatch 撞互联 —— '
         f'全表唯一横跨三节的一条（「MoE 是算力优化」是常见误读）。</text>')
p.append('</svg>')
io.open('figB.svg','w',encoding='utf-8').write('\n'.join(p)); print('figB',H,CNT)

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
