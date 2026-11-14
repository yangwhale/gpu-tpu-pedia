import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# -*- coding: utf-8 -*-
"""专题八 · GPU→TPU 训练精度对齐 —— 教材（课件）。

════════════════════════════════════════════════════════════════
⭐ 为什么有这一讲（2026-11-05 现场原话）
════════════════════════════════════════════════════════════════
    「第九章不是当代大模型介绍嘛，那个没有意义，不讲了，就把那个第9章替换一下」
原来的第九讲「最新开源模型对比」只有大纲，从没写成页面；旧大纲留在 git 历史里。
2026-11-11 跟「精度与量化」对调编号：这一讲从专题九改成专题八（正好轮到第八讲），精度与量化改成专题九。

════════════════════════════════════════════════════════════════
一句话主线
════════════════════════════════════════════════════════════════
    TPU 和 GPU 不可能一位不差；要求的是：TPU 跟 GPU 的差，不比 GPU 自己跟自己的差大。
    超了，就一级一级往下查，直到查清是谁的差 —— 有时候偏的是 GPU 的 golden 本身。

    一  什么叫对齐：一位不差做不到 → 拿 GPU 自己重跑的差当尺子；五级漏斗
    二  地基 L0：输入必须一位不差（数据回放、权重逐位、配置按实际值、续训、第 1 步 loss）
    三  指标（一）逐步序列：dₜ → 带符号均值 / MARE / 最大单步；噪声上限 N；参照对数；只对一份基线
    四  指标（二）张量：rel_L2、cos、范数比；对 FP64 的 R 比定责；ULP 只诊断；两档门槛
    五  浮点与硬件：乘精确、加才舍；GPU 张量核抹零；TF32 陷阱；TPU 边传边乘（collective matmul）；excess precision
    六  观测：TPU 上加钩子会改编译 → 状态读 ＋ 从动量反推梯度、隔离回放、固定探针
    七  案例一（混元 3）：loss 偏低 → 参数范数漂 → 输出层冷门行在 Adam ε 区 → 让所有 logits 一起升降的方向 → GPU dgrad 长求和截断
        → 同状态各算一步 → 精确轨迹（连 FP32 一次累加也偏，分段才准）→ 五环一条链（2026-11-11 由原第七、八两节合并）
    八  案例二（第二次迁移 ＝ 混元 4 缩小版）：DDP 与 FSDP 曲线超出噪声 → 第 1 步只剩词嵌入梯度 → 假设「BF16 跨卡求和」→ FP32 开关撞上编译器 bug
        → 按行分组推翻假设（39 个高频词）→ 先加后传 vs 先传后加 → 单算子：排序＋标记×excess precision → 后端在下标数 ＞ 行数/8 时插排序
        → 高频词 × 不排序；前两种修法 → GPU 为什么没有（推断）、关 excess precision 更差 → §8.9 第三种修法：只切词嵌入表
        → §8.10 方法：怎么从编译产物看通信（形状 × 字节宽 × 每步次数）
    九  验收与 golden：四份 golden 一步步来；出处总表；拆账；两关验收两档门槛；读总览表
    十  工程化与教训：日常回归（逐位优先、PR 声明、LKG、TPU 噪声标定）；记录规矩；踩过的坑；出处台账

暗线：① 每个结论都要一把尺子；② 先排除自己（地基、口径、测量工具），再怀疑硬件；③ 开关的读回值、工具的输出都要实测核对。

════════════════════════════════════════════════════════════════
数字从哪来
════════════════════════════════════════════════════════════════
全部数字来自本课程作者的一次对齐实践（混元 3 截前 4 层，B200×8 Megatron-Bridge → TPU v5p（8 颗芯片）／ v7x（4 颗芯片） MaxText，1000 步），
常数在 topic08_numbers.py，逐步序列在 topic08_series.json（由私有仓库里的原始记录导出，只含差值序列）。
⛔ 公开仓库闸门：不写内部链接、不写客户与人名、不写内部基础设施；工具一律称「一个对齐工具」。
⭐ 大纲在 `Courses/专题08-GPU到TPU训练精度对齐.md`，只是计划，不必与页面同步。
"""
import io
import os
import re

import topic03_page as P
import course_ai_trainer as AIT
import topic08_numbers as N

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "WebPages", "topic-08.html")
OUT_DIR = os.path.join(HERE, "..", "WebPages")

head = P.make_head(
    "专题八 · GPU→TPU 训练精度对齐",
    "GPU→TPU 训练精度对齐 · TPU 和 GPU 的差，不能比 GPU 自己跟自己的差大",
    "混元 3 从 GPU 搬到 TPU：地基逐位、两种指标、浮点与硬件、观测，再走一遍真实排查 —— 看这一截差到底是谁的。",
    "topic-08.html")
head += """
<style>
.lecbtn { display:inline-block; margin-left:14px; font-size:13px; padding:4px 12px; border-radius:14px;
          background:#fff; border:1px solid var(--line); color:var(--blue); text-decoration:none }
.lecbtn:hover { background:var(--blue-l); border-color:var(--blue) }
.qz9 ol { padding-left:2.2em; margin:6px 0 }
table.t9 { border-collapse:collapse; margin:12px 0; font-size:15px }
table.t9 td, table.t9 th { padding:6px 12px; border-bottom:1px solid #e8eaed; text-align:left; vertical-align:top }
table.t9 th { color:var(--gray); font-weight:600 }
p.bridge { color:var(--gray); border-left:3px solid var(--line); padding-left:12px }
</style>"""
head += """<style>
.foundbox{border:2px solid var(--blue);border-radius:12px;padding:14px 18px 8px;margin:18px 0 6px;background:#fff}
.foundbox .ft{font-size:21px;font-weight:700;color:var(--blue)}
.foundbox .fsub{color:var(--gray);margin:4px 0 6px}
table.ftab{width:100%;font-size:14.5px}
table.ftab td:first-child{width:38%}
table.ftab td:last-child{white-space:nowrap}
table.ftab tr.fgrp td{background:var(--blue-l);padding:5px 12px;border-bottom:none}
table.ftab tr.fgrp span{color:var(--gray);font-size:13px}
.newtag{margin-right:4px}
.foundbox .fnote{font-size:13px;color:var(--gray)}
.foundbox .fbridge{font-weight:600;margin-top:4px}
</style>"""
head += """<style>
.adamcard{border:2px solid var(--line);border-radius:12px;padding:14px 18px;margin:16px 0;background:#fff}
.adamcard .af{font-size:30px;font-weight:700;text-align:center;letter-spacing:1px;margin:6px 0 4px}
.adamcard .af.long{font-size:21px;letter-spacing:0}
.adamcard th{text-transform:none}
.adamcard .aw{font-size:17px;margin:4px 0;font-family:inherit}
.adamcard .aw .k{display:inline-block;width:7.5em;color:var(--gray);font-size:14px;font-weight:600}
.adamcard .aw em{color:var(--gray);font-size:13px;font-style:normal}
.adamcard .asub{text-align:center;color:var(--gray);font-size:14px;margin:0 0 10px}
.adamcard .acases{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:8px}
.adamcard .acases div{border-left:4px solid var(--line);padding:6px 10px;font-size:15px}
.adamcard .acases div:last-child{border-left-color:#d93025}
.adamcard tr.epsrow td{color:#a50e0e;font-weight:600}
.adamcard .anote{font-size:14px}
</style>"""
from course_narrow_css import CSS as _NARROW_CSS   # noqa: E402
head += "<style>" + _NARROW_CSS + "</style>\n"
head += "<style>.foldfig ol { padding-left: 2.4em }\np.codeline{font-size:13.5px;color:var(--gray)} a.codelink{display:inline-block;margin-right:14px;padding:1px 8px;border:1px solid var(--line);border-radius:10px;text-decoration:none;color:var(--blue)} a.codelink:hover{background:var(--blue-l)}</style>\n"
head += "<style>figure.fwide > video { display:block; width:100%; max-width:1240px; margin:0 auto; border-radius:8px }\nfigure.fwide:has(> video) > figcaption { max-width:1240px; margin-left:auto; margin-right:auto }</style>\n"

SECTIONS = [
    ("s一", "一", "什么叫对齐"),
    ("s二", "二", "地基 L0：输入必须一位不差"),
    ("s三", "三", "指标（一）：逐步序列"),
    ("s四", "四", "指标（二）：张量"),
    ("s五", "五", "浮点与硬件：差从哪来"),
    ("s六", "六", "观测：在 TPU 上怎么看"),
    ("s七", "七", "案例一：loss 偏低，查到是谁的差"),
    ("s八", "八", "案例二：DDP 和 FSDP 对不上"),
    ("s九", "九", "验收与 golden 设计"),
    ("s十", "十", "工程化与教训"),
]


def sec(sid, num, title):
    return ('<section id="%s"><div class="wrap"><div class="stn"><span class="badge">第 %s 节</span>'
            '<h2>%s</h2></div>' % (sid, num, title))


# 公开代码链接：一律钉到公开版仓库的同步提交（完整 sha），不用 main —— 公开仓库每次同步整棵树替换，文件可能挪位置。
#   2026-11-12 D10 升到同步提交 124b2a7（镜像导出规则补掉裸 PR 号后）；D8 是 f559a46（研究线案例 17 人类可读版与 GPU 上卡验证合入后）；G4 是 8cfcc5a（研究线案例 17 与整步扫描工具合入后）；F30 是 cb2a4f2（研究线 D30 事实更正合入后）；F20 是 950974d（研究线小尺寸后段两头验证、C 口径合入后；14 条链接逐条核过在新 sha 上存在，都不带行号锚点）。此前 F6 是 7df4a3d（对应研究线 RMSNorm 0.0.52 机理与 check 子命令合入后）；每条链接都实际打开核过（T8-STATE 第四阶段 E2、E9，第五阶段 F1）。
PUB_SHA = "124b2a797c2f3961bcadf72cbe2d7156cb0caab9"
PUB_REPO = "https://github.com/yangwhale/" + "jan" + "us"


# MaxText 公开分支（各一个提交、没有历史）：混元 4 用 hunyuan4 分支，混元 3 用 hunyuan 分支；开关说明在根目录 HUNYUAN.md。
MT_REPO = "https://github.com/yangwhale/maxtext"
MT_SHA = {4: "319afd46a676654e33c09a553b3d4266b82716cc", 3: "fb4c449621c2f3f3a0941f176903911da4cfb69c"}


def mt(ver, path, lines, label):
    return ('<a class="codelink" href="%s/blob/%s/%s%s">MaxText（混元 %d 公开分支）：%s</a>'
            % (MT_REPO, MT_SHA[ver], path, ("#" + lines) if lines else "", ver, label))


def code(path, label=None):
    path = path.replace("PKG/", "jan" + "us/")       # 仓库里核心包的目录名跟仓库同名
    return ('<a class="codelink" href="%s/blob/%s/%s">代码：%s</a>' % (PUB_REPO, PUB_SHA, path, label or path.split("/")[-1]))


def quiz(q, a):
    return ('<details class="foldfig"><summary><b>考考自己</b>：%s</summary><p>%s</p></details>' % (q, a))


HERO = '''
</head>
<body>

<div class="hero"><div class="wrap">
  <div class="crumb"><a href="index.html">现代 AI 加速器 GPU / TPU 系统课程</a> ／ 深入 ／ 专题八
    ／ <b>GPU→TPU 训练精度对齐</b>
    <a class="lecbtn" href="topic-08-lecture.html">📝 讲义（授课稿）</a></div>
  <h1>GPU→TPU 训练精度对齐</h1>
  <div class="hook">
    一个模型在 GPU 上训稳了，搬到 TPU 上，怎么证明训的还是同一个模型？<br>
    <em>不可能一位不差。能要求的是：TPU 跟 GPU 的差，不比 GPU 自己跟自己的差大。</em>
  </div>
  <p style="max-width:820px;color:var(--gray)">
    先把地基铺平，再学两类指标和一把噪声尺子，弄清浮点和硬件在哪儿舍入，然后跟着一次真实的排查走到底：
    TPU 的 loss 三百步以后平均比 GPU 低约 1.7e-4，查到最后，答案在一个谁都没怀疑的地方；再跟第二个案例：两种数学等价的并行方式对不上，这回又是谁的差？</p>
  <div class="chips">
    <span class="chip">前置 <b>专题二</b>（硬件）、<b>专题四</b>（Adam）；浮点格式 §5.1 现讲</span>
    <span class="chip">口径 <b>截至 2026-11</b></span>
    <span class="chip">⏱ <b>讲约 100 分钟</b>（三次课，每次 31–35 分钟）</span>
  </div>
  <p class="author">课程作者　<b>Chris Yang</b><span class="sep">·</span>Google Cloud
    AI Infra 架构师</p>
</div></div>

''' + AIT.note("topic-08-lecture.html") + '''
<div class="wrap">
  <p style="color:var(--gray)">全讲的例子是一次真实的对齐：混元 3（一个 MoE 模型，截前 4 层），GPU 侧 B200×8 ＋ Megatron-Bridge，TPU 侧 v5p（8 颗芯片）／ v7x（4 颗芯片） ＋ MaxText，同一份权重、同一批数据，1000 步。
    数字都是这次实测的；自己推出来的，标了「本课推导」。出处在第十节末尾的台账。
    后来又做了一次迁移（混元 4 的缩小版，一个更新的、带稀疏注意力的 MoE 模型，下文叫「第二次迁移」），同一套方法又走了一遍，后面借它当旁证和第二个案例（第八节），用到时都先报名字。第三次迁移是混元 4 的另一个尺寸，这次从生产代码原样起步，查清了几件「换了条件就不作数」的事（第五节）。</p>
</div>
'''


# ── 开场「我们发现了什么」大列表（课程负责人 2026-11-14：「课程一开始先展示一下我们都发现了哪些有影响的点，有一个大列表，然后再开始展开讲正文」）──
#   每条一行：现象（大白话）＋ 影响（带数，e 记法）＋ 在哪一节展开。只列不解释；案例一、案例二只报现象，不点破答案（第五节、第八节的防剧透规矩照旧）。
#   条目只取正文已经讲到的；🆕 两条是 2026-11-14 图文 32K 的新发现，正文放在 §4.3、§5.5 的折叠里（课上不讲）。数字都是正文里已有的（新两条走 topic08_numbers）。
def _frow(what, effect, where, href, new=False):
    return ('<tr><td>%s%s</td><td>%s</td><td><a href="#%s">%s</a></td></tr>\n'
            % ('<span class="newtag">🆕</span>' if new else "", what, effect, href, where))


def _fgrp(title, n):
    return '<tr class="fgrp"><td colspan="3"><b>%s</b>　<span>%d 处</span></td></tr>\n' % (title, n)


_FOUND_ROWS = [
    ("写的，不是生效的", [
        ("权重衰减：GPU 配置里写 0.1，被调度器悄悄覆盖", "实际生效 0.033，差 3 倍", "§2.3", "s2-3"),
        ("「存下路由、反向不重选」，配错一种重算策略就被悄悄忽略", "平均 «M3_REROUTE_PCT»% 的 token（最多 «M3_REROUTE_MAX»%）按另一套路由算梯度；参数范数 r «M3_REROUTE_PN0»，存下路由后 2.27", "§2.3 折叠", "s2-3"),
        ("TF32 开关读回来是关，矩阵乘照样走 TF32", "FP32 矩阵乘离 FP64 仍有 2.94e-4，真关掉 1.15e-6", "§5.3", "s5-3"),
    ]),
    ("硬件在哪儿舍", [
        ("GPU 张量核每 16 个乘积抹一次零", "同一次求和越长越偏：合成输入长 65,536 时朝零偏 3.1e-4（相对）", "§5.2", "s5-2"),
        ("4 个设备上验过的程序，换到 8 个设备，一次求和从舍 1 次变成舍 6 次", "那批投影离 FP64：TPU 2.35e-3，GPU 1.66e-3，R 1.41；代码一行没改", "§5.4", "s5-4"),
        ("名义 FP32 的输出头，在 TPU 默认精度下只算一遍 BF16；第 1 步看不出", "权重离开格点后 logits 误差是真 FP32 的约 5000 倍（1.9e-3 对 3.7e-7，在模拟离格 10 步的权重上测）；前 300 步主 loss r 7.32，换 high 档 1.03", "§5.6", "s5-6"),
        ("TPU 默认的 exp、log 图快不图准", "v5p 上 exp 最多差 105 格、log 62 格（GPU ≤ 2 格）；单算子离 FP64 最多是 GPU 的 8.68 倍", "§5.7", "s5-7"),
        ("一个 Python 常数（log₂e）跟着 BF16 的 q 一起被舍", "注意力 logits 整体放大 1.8e-3", "§5.6 折叠", "s5-6"),
    ]),
    ("编译器替你省掉的舍入", [
        ("RMSNorm scale 那次「舍成 BF16」：4K 程序里在，32K 程序里 8 个全被删", "主 loss 第 51–300 步偏约 −24σ（σ＝GPU 8 份的份间标准差）；120 步短跑里只补这一处 −0.2σ，补别处的都在 −30.8σ 左右", "§5.5", "s5-5"),
        ("MTP 输入那次「存成 BF16」被省掉，后面紧跟一个约 «M3_MTPIN_COEF» 的乘法", "GPU 的 hnorm 输出整体偏大 «M3_MTPIN_BIAS»，MTP 层 19 个张量的梯度一齐差 «M3_MTPIN_C_T»；补上后 MTP loss 差 «M3_MTPIN_LOSS_BEFORE_T» → «M3_MTPIN_LOSS_AFTER_T»", "§5.5 折叠", "mtp-input", True),
        ("三份梯度相加，两边舍的次数不一样", "主 loss 第 551–1000 步对 GPU 16 份均值偏 ＋9.4e-6（GPU 16 份留一法标准差 2.1e-6），照 GPU 写法补上 ＋1.9e-6", "§9.7", "s9-7"),
    ]),
    ("路由一翻，梯度就换了一份", [
        ("输入只差一丝，近平局的 token 就换了专家", "第 1 步 TPU 对 GPU 的 rel_L2：一个专家的梯度 «STEP1_EXPERT»，比到的 50 个张量合起来才 «STEP1_ALL_E»，约 80 倍", "§4.3", "s4-3"),
        ("视觉塔自己不选专家，q、k 的梯度照样越线", "R «M3_VIT_R_1»；照 TPU 的路由重算 FP64，降到 «M3_VIT_RFLIP_1»：全是路由翻转", "§4.3 折叠", "vit-flip", True),
    ]),
    ("两个案例，一个编译器 bug", [
        ("TPU 的主 loss 三百步以后平均比 GPU 低约 1.7e-4（相对，第 301–1000 步带符号均值 −1.66e-4），第 1000 步参数范数多涨 2.1e-2（绝对差，参数范数约 3042）", "TPU 算错了吗？答案在一个谁都没怀疑的地方", "第七节", "s七"),
        ("DDP 和 FSDP 数学等价，曲线却超出噪声", "词嵌入梯度离 FP64：纯数据并行 1.51e-2，GPU 4.17e-3，R 3.62；全出在 39 个词上", "第八节", "s八"),
        ("一个 FP32 开关打开，一块梯度凭空没了", "r 冲到 611", "§8.2", "s8-2"),
    ]),
]
_FOUND_N = sum(len(g[1]) for g in _FOUND_ROWS)
assert _FOUND_N == 16, _FOUND_N
FOUND = ('''
<div class="wrap" id="found">
<div class="foundbox">
  <div class="ft">开场：三次迁移，我们查出来的 ''' + str(_FOUND_N) + ''' 处差</div>
  <p class="fsub">每一处都真的改变了结果。这里只列现象和量级，<b>为什么、怎么查出来的，正文一节一节讲</b>（点右栏跳过去）。</p>
  <table class="t9 ftab"><thead><tr><th>查出来的现象</th><th>影响多大</th><th>在哪讲</th></tr></thead><tbody>
''' + "".join(_fgrp(t, len(rows)) + "".join(_frow(*r) for r in rows) for t, rows in _FOUND_ROWS) + '''  </tbody></table>
  <p class="fnote">R：离 FP64 的距离之比，1 是跟 GPU 一样准（§4.2）；r：曲线上的差 ÷ GPU 自己的噪声上限，过 2 判不过（§3.3）；格：末位差几格（§5.7）。🆕 是 2026-11-14 刚查到的，正文在折叠里。</p>
  <p class="fbridge">有的比 GPU 自己的噪声大几十倍，有的小到 GPU 重跑都盖得住。它们是怎么一处一处查出来的，就是这一讲。从最开头那件讲起 ↓</p>
</div>
</div>
''')


# ── §7.2 Adam 公式卡（课程负责人 2026-11-14：做成大号公式卡，讲两种情况；表里的 √v̂ 是第 300 步各词频桶的实测中位数）──
def _vrow(name, rows, v):
    share, shrink = N.vhat_row(v)
    zone = "ε 区" if v < N.EPS else "常见区"
    return ('<tr%s><td>%s（%s 行）</td><td>%s</td><td>%s</td><td>%s 倍</td><td>%s</td></tr>'
            % (' class="epsrow"' if v < N.EPS else "", name, format(rows, ","), N.sci(v),
               ("%.1f%%" % (share * 100)) if share >= 0.01 else ("%.2f%%" % (share * 100)),
               ("%.1f" % shrink) if shrink >= 1.1 else ("%.3f" % shrink), zone))


ADAM_CARD = (
    '\n  <div class="adamcard">\n'
    '  <div class="aw"><span class="k">0　裁剪</span>g ← g × min(1, ' + N.CLIP_GRAD_T + ' ÷ ‖g‖)　<em>（全部梯度按全局范数先裁剪，再进 Adam）</em></div>\n'
    '  <div class="aw"><span class="k">1　两个本子</span>m<sub>t</sub> ＝ β₁·m<sub>t−1</sub> ＋ (1 − β₁)·g<sub>t</sub>　　v<sub>t</sub> ＝ β₂·v<sub>t−1</sub> ＋ (1 − β₂)·g<sub>t</sub>²　<em>（逐元素平方）</em></div>\n'
    '  <div class="aw"><span class="k">2　偏差修正</span>m̂ ＝ m<sub>t</sub> ÷ (1 − β₁<sup>t</sup>)　　v̂ ＝ v<sub>t</sub> ÷ (1 − β₂<sup>t</sup>)</div>\n'
    '  <div class="af">θ ← θ − lr · [ m̂ ÷ (√v̂ ＋ ε) ＋ λ·θ ]</div>\n'
    '  <p class="asub">本例：β₁ ＝ ' + str(N.BETA1) + '，β₂ ＝ ' + str(N.BETA2) + '，ε ＝ ' + N.sci(N.EPS) + '（加在开根号之后），lr ＝ ' + N.sci(N.LR) + ' 常数，λ ＝ ' + str(N.WD_EFFECTIVE)
    + '（解耦的权重衰减；一维参数如 norm 权重、偏置不衰减）。下文 u ＝ m̂ ÷ (√v̂ ＋ ε)：√v̂ 大致就是「梯度有多大」，ε 是分母上防除零的小常数。'
    '旁注：梯度长期稳定时 m̂ ≈ g、√v̂ ≈ |g|，u ≈ g ÷ (|g| ＋ ε)，只看形状（下面那张图就是它）。</p>\n'
    '  <div class="acases">\n'
    '    <div><b>梯度够大，ε 等于没有</b><br>√v̂ ＝ 1e-6 时，ε 只占分母的 1%；u 几乎就是 m̂ ÷ √v̂，梯度大小约掉，步子都差不多大</div>\n'
    '    <div><b>梯度很小，ε 变成一个把更新压小的开关</b><br>压小倍数 ＝ (√v̂ ＋ ε) ÷ √v̂；√v̂ 比 ε 小，就压小 2 倍以上，u ≈ m̂ ÷ ε，更新跟梯度成正比</div>\n'
    '  </div>\n'
    '  <table class="t9"><thead><tr><th>第 300 步，输出层按词频分桶（实测中位数）</th><th>√v̂</th><th>ε 在分母里占</th><th>更新被压小</th><th>在哪个区</th></tr></thead><tbody>\n'
    + "\n".join(_vrow(*b) for b in N.VHAT_BUCKETS) +
    '\n  </tbody></table>\n'
    '  <p class="anote">没出现过的词那 ' + format(N.RARE_ROWS, ",") + ' 行，第 300 步有 ' + ("%.0f%%" % (N.VHAT_RARE_BELOW_EPS * 100))
    + ' 的元素 √v̂ ＜ ε，也就是被压小 2 倍以上；中位数那一档压小 ' + ("%.1f" % N.vhat_row(N.VHAT_BUCKETS[3][2])[1])
    + ' 倍。压小 10 倍、20 倍以上的各占多少，结果文件里只有中位数和 ε 以下的比例，算不出来，不写。'
    '<b>被压小本身两边一样，危险在于这时更新跟梯度成正比</b>：「ε 在分母里占」那一列，正好就是梯度差 1% 时更新差几成（弹性 ε ÷ (√v̂ ＋ ε)）。'
    '梯度 4e-9 时 0.71，1e-6 时 0.0099：梯度方向上一点差，在 ε 区原样漏过去。</p>\n'
    '  </div>\n')


# ── §3.3「从一条序列到一个 r」公式卡（夜间第二轮：五样东西散在 §3.1–3.3，讲师要来回翻；照 §7.2 AdamW 卡的样子收成一张）──
def _rrow(q, st, v, n):
    r = abs(v) / n
    return '<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td></tr>' % (' class="epsrow"' if r > 2 else "", q, st, N.tx(v, "%.2e", plus=(v > 0 and st == "带符号均值")), N.tx(n, "%.2e"), r)


R_CARD = (
    '\n  <div class="adamcard">\n'
    '  <div class="aw"><span class="k">1　逐步差</span>dₜ ＝ (被测ₜ − 基线ₜ) ÷ |基线ₜ|　<em>（同一步、带符号、单位 1；第 1 步单独报，不进窗口）</em></div>\n'
    '  <div class="aw"><span class="k">2　三个数</span>窗口里：带符号均值 mean(dₜ)　平均绝对差 MARE ＝ mean(|dₜ|)　最大单步 max|dₜ|</div>\n'
    '  <div class="aw"><span class="k">3　尺子</span>GPU 同设置 K 份两两配对 C(K,2) 对，每对按同一窗口、同一分母算同样三个数；N ＝ 各对里的最大值（三个数各一把）</div>\n'
    '  <div class="af">r ＝ |被测的那个数| ÷ N</div>\n'
    '  <p class="asub">一条检查取三个 r 的最大，一次比较取所有检查的最大</p>\n'
    '  <div class="acases">\n'
    '    <div><b>r ≤ 1：像多跑了一次噪声</b><br>被测落在 GPU 自己重跑的范围里，过</div>\n'
    '    <div><b>r ＞ 2：比最坏的噪声还大一倍</b><br>不过；1 到 2 之间待定，先看纯噪声自己能拿几分（§3.5）</div>\n'
    '  </div>\n'
    '  <table class="t9"><thead><tr><th>第二次迁移，FSDP 改纯数据并行，前 300 步（实测）</th><th>统计</th><th>被测</th><th>N（GPU 8 份两两 28 对的最大）</th><th>r</th></tr></thead><tbody>\n'
    + "\n".join(_rrow(*x) for x in N.CARD_R_ROWS) +
    '\n  </tbody></table>\n'
    '  <p class="anote">五个 r 里最大的是梯度范数的 2.62，这次比较就记 2.62，判不过；主 loss 全程三个数里最差的是带符号均值 1.85（待定），后段单独再判一遍是 2.03（不过）。同一套算法让 GPU 自己跟自己比，纯噪声也能拿到 2.47（300 步、8 份）和 1.82（1000 步、16 份），见 §3.5。</p>\n'
    '  </div>\n')


# ── §4.2「第 1 步逐张量定责」公式卡（夜间第二轮：公式散在 §4.1–4.3 和 fig-step1，收成一张；表里的数都是课件已有的实测）──
R4_CARD = (
    '\n  <div class="adamcard">\n'
    '  <div class="aw"><span class="k">1　距离</span>rel_L2(被测, 参照) ＝ ‖被测 − 参照‖ ÷ ‖参照‖　<em>（全部元素，FP64 里累加；两边等长时 rel_L2² ＝ 2(1 − cos)）</em></div>\n'
    '  <div class="aw"><span class="k">2　拿到梯度</span>g ＝ m₁ ÷ (1 − β₁) ÷ 裁剪系数　<em>（第 1 步从一阶动量倒推，β₁ ＝ 0.9）</em></div>\n'
    '  <div class="aw"><span class="k">3　对基线</span>d ＝ rel_L2(被测, 基线) ÷ 尺子　<em>（尺子 ＝ GPU 第 1 步重跑几份两两的最大差，下限 2⁻²⁰）</em></div>\n'
    '  <div class="af long">R ＝ rel_L2(被测, FP64) ÷ max(rel_L2(基线, FP64), GPU 几份离 FP64 的中位数)</div>\n'
    '  <p class="asub">多出一份同量级、互相独立的舍入误差，误差平方相加：R ≈ √2 ≈ 1.41，所以线画在两者之间的 1.2</p>\n'
    '  <div class="aw"><span class="k">4　路由翻转</span>R_flip ＝ rel_L2(被测, FP64 用<b>被测自己选的</b>专家) ÷ 同一个分母　<em>（R_flip ≤ 1.2 且比不灌时更近 → 记「路由翻转」，不算丢精度；分母不换，换了就等于给翻转放宽尺子）</em></div>\n'
    '  <div class="acases">\n'
    '    <div><b>超出尺子、R ≤ 1.2：只是换了加法顺序</b><br>平台固有，记在刻度内，不修</div>\n'
    '    <div><b>R ＞ 1.2：这一步多舍了</b><br>丢了精度，去查被测那一边，找到多舍的那一步</div>\n'
    '  </div>\n'
    '  <table class="t9"><thead><tr><th>例子（实测）</th><th>被测离 FP64</th><th>参照离 FP64</th><th>d（超尺子几倍）</th><th>R</th><th>记作</th></tr></thead><tbody>\n'
    '  <tr class="epsrow"><td>词嵌入梯度，纯数据并行（第二次迁移）</td><td>«C2_EMB_DP_T»</td><td>«C2_EMB_GPU_T»（GPU 中位数）</td><td>5.89</td><td>«C2_EMB_R_T»</td><td>丢了精度</td></tr>\n'
    '  <tr><td>同上，修好以后</td><td>«C2_FIX1_EMB_T»</td><td>«C2_EMB_GPU_T»</td><td>1.36</td><td>0.91</td><td>换了顺序</td></tr>\n'
    '  <tr class="epsrow"><td>qkv 投影，打开 FP32 累加之前（混元 3）</td><td>' + N.sci(N.QKV_ABS_BEFORE) + '</td><td>' + N.sci(N.QKV_ABS_GPU) + '</td><td>—</td><td>«R_QKV_BEFORE»</td><td>丢了精度</td></tr>\n'
    '  <tr><td>同上，打开以后</td><td>' + N.sci(N.QKV_ABS_AFTER) + '</td><td>' + N.sci(N.QKV_ABS_GPU) + '</td><td>—</td><td>1.000</td><td>换了顺序</td></tr>\n'
    '  <tr class="epsrow"><td>8 个设备，往里投的那批投影（第三次迁移）</td><td>2.35e-3</td><td>1.66e-3</td><td>—</td><td>1.41</td><td>多舍一次</td></tr>\n'
    '  <tr><td>视觉塔 q、k 的梯度（第三次迁移，§4.3 折叠）</td><td>—</td><td>—</td><td>—</td><td>«M3_VIT_R_1» → R_flip «M3_VIT_RFLIP_1»</td><td>路由翻转</td></tr>\n'
    '  </tbody></table>\n'
    '  <p class="anote">「修好以后」那一行照样超尺子 1.36 倍，看似没修好；R 0.91，比基线和 GPU 都近，其实修好了：超尺子只说两边不一样，R 才说谁丢了精度。路由翻转按第 4 行判；量级上，一个专家 n 个 token 里换掉 k 个，梯度相对差约 √(k ÷ n)（估量级，推断，§4.3）。</p>\n'
    '  </div>\n')


# ── §5.2「张量核抹零模型」公式卡（夜间第二轮：模型只在 fig-truncate 左格示意和折叠里；数都是课件已有的实测或标了推断）──
TC_CARD = (
    '\n  <div class="adamcard">\n'
    '  <div class="aw"><span class="k">1　分组</span>每 «TC_GROUP» 个乘积 p₁…p₁₆，连同累加器 a，一组一起加</div>\n'
    '  <div class="aw"><span class="k">2　对齐</span>按这一组里最大的指数 e_max 对齐</div>\n'
    '  <div class="af">低于 2^(e_max − «TC_KEEP») 的位，直接丢掉（朝零）</div>\n'
    '  <p class="asub">组内相加后写回 FP32，也朝零；「16」「25」是在 B200 上拟合出来的，不是厂商规格：拿它重算三层路由打分，4096 × 192 个元素逐位对上，改成 8 个一组、24 位或就近舍入，命中率都掉到 1% 以下</p>\n'
    '  <div class="acases">\n'
    '    <div><b>四舍五入：有进有舍</b><br>每一项都比累加值的一格大得多时，误差有正有负、大体抵消</div>\n'
    '    <div><b>抹零：每单都亏</b><br>累加值越长越大，窗口往高位挪，后来的小项被丢的比例越来越大：越长越偏</div>\n'
    '  </div>\n'
    '  <table class="t9"><thead><tr><th>量的是什么（朝零的相对偏差）</th><th>短</th><th>中</th><th>长</th></tr></thead><tbody>\n'
    '  <tr><td>合成正数输入，K ＝ 256 ／ 4,096 ／ 65,536（输出写回 BF16）</td><td>«TC_BIAS1»</td><td>«TC_BIAS2_T»</td><td>«TC_BIAS3»</td></tr>\n'
    '  <tr><td>只算截断模型（推断，没写回 BF16）</td><td>约 3e-7</td><td>约 1.1e-5</td><td>—</td></tr>\n'
    '  <tr><td>FP32 不走张量核</td><td colspan="3">≤ 5e-8</td></tr>\n'
    '  </tbody></table>\n'
    '  <p class="anote">短求和那一格主要是输出写回 BF16 的舍入，截断只占约 3e-7（推断）；截断要到长求和才主导。</p>\n'
    '  </div>\n')

QUIZ = '''
<details class="foldfig" id="qz9box"><summary><b>🎯 开场热身：三道题</b>
  <span class="why">—— 第一问是浮点常识（§5.1 细讲），后两问预习这一讲。<u>不做也不影响这一讲</u></span></summary>
<div class="qz9">
<p><b>第一问</b>：BF16 和 FP32 的指数位一样多，都是 8 位。那 BF16 比 FP32 少的是什么、少了几位？这意味着什么？</p>
<details><summary>看答案</summary>
<p>少的是尾数：FP32 有 23 位，BF16 只有 7 位，少 16 位。指数一样宽，所以<b>能表示的范围一样大</b>，差的是<b>精度</b>：
  1 附近，FP32 一格约 «FP32_GAP1_T»，BF16 一格约 «BF16_GAP1_T»。</p></details>
<p><b>预习</b>（先想一想，这一讲里揭晓）：</p>
<ol start="2">
<li><b>第二问</b>：同一个 GPU 程序、同样的数据，跑两遍，结果会一位不差吗？（§1.1）</li>
<li><b>第三问</b>：搬到 TPU 后，loss 三百步以后平均比 GPU 低约 1.7e-4。TPU 算错了吗？（§7.5）</li>
</ol>
</div>
</details>
'''

BODY = sec("s一", "一", "什么叫对齐") + '''
  <div class="note"><span class="t">这一讲从一件真事讲起</span>一个模型从 GPU 搬到 TPU，第 1 步几乎一样，1000 步下来，三百步以后 TPU 的 loss 平均比 GPU 低约 1.7e-4。<b>TPU 算错了吗？</b>答案在第七节；先想两个更小的问题：同一个 GPU 程序跑两遍，会一位不差吗（§1.1）？BF16 比 FP32 少的是什么（下面热身第一问）？</div>
''' + QUIZ + '''
  <p class="lead">把一个在 GPU 上训好的模型搬到 TPU，要的保证只有一句：<b>TPU 训的是同一个模型</b>。同样的数据、同样的初始权重、同样的超参，两边训出来的东西在统计上分不出来。</p>

  <h3>1.1　一位不差，做不到</h3>
__FIG_ASSOC__
  <p>同一个 GPU 程序跑两遍，会一位不差吗？<b>不会：GPU 自己跑两遍也不一位不差</b>。反向里有些算子（注意力的反向、MoE 分发 token 那一步的反向〔scatter-add〕、融合的交叉熵都可能是；这次没拆到底是哪一个，具体看实现）要很多线程往同一个格子里加数（原子加），谁先算完谁先加进去，每次的先后都不一样。
    前向通常是确定的，所以两次运行第 1 步的 loss 一样；从反向开始才分叉。</p>
  <details class="foldfig"><summary><b>外面的人怎么说</b>：换个顺序求和能差出多少种结果</summary>
  <p>有人把 8 个大小悬殊的数（±1e-10、±1e-5、±1e-2、±1，正负成对，精确和是 0）用 Python 的 64 位浮点随机打乱求和 1 万次，得到 102 种不同的结果；同一篇文章也指出，大模型的前向在同一台机器上重跑是逐位一样的，大模型里常用、又要原子加的，基本只有 FlashAttention 的反向（Thinking Machines，<em>Defeating Nondeterminism in LLM Inference</em>，2025）。</p></details>

  <h3>1.2　那拿什么当尺子：GPU 自己跟自己的差</h3>
  <p>GPU 上跑出来的标准答案叫 <b>golden</b>。同样设置的 GPU 再多跑几次，两两之间的差有多大，就是<b>噪声</b>。拿它当尺子。道理一句话：如果 TPU 只是换了加法顺序，它跟 GPU 的差，应当跟 GPU 两次重跑之间的差一样大；比这个大，就是真有东西不一样。</p>
__FIG_RULER__
  <p>GPU 也能开确定性模式，把重跑的差关成 0；可生产训练不开它，尺子要量的是生产配置下会有的执行噪声。</p>

  <h3>1.3　从哪查起：五级</h3>
__FIG_FUNNEL__
  <p>验收从下往上一级级过，排查从上往下一级级缩小范围。每一级怎么判，后面几节讲。</p>
  <details class="foldfig"><summary><b>五级各查什么、怎么判</b>（用到的 R、定责第四节讲）</summary>
  <table class="t9"><thead><tr><th>级</th><th>查什么</th><th>怎么判</th></tr></thead><tbody>
  <tr><td>L0 地基</td><td>每一步每张卡吃的数据、转换后的权重、配置</td><td>一位不差</td></tr>
  <tr><td>L1 单个算子</td><td>一个矩阵乘、一次优化器更新</td><td>两边各离 FP64 真值多远（第四节的 R 比）</td></tr>
  <tr><td>L2 单层</td><td>第 1 步每一层的前向、反向，每层单独回放</td><td>R 比（前向、反向都是）</td></tr>
  <tr><td>L3 整网第 1 步</td><td>全部参数的梯度、更新后的权重和优化器状态</td><td>对 GPU 第 1 步重跑的噪声；超出的、尺子看不见的，对 FP64 定责（§4.2）</td></tr>
  <tr><td>L4 多步</td><td>1000 步的 loss、梯度范数、参数范数</td><td>对 GPU 噪声（第三节）</td></tr>
  </tbody></table>
  </details>

  <h3>1.4　这一讲跟着走的那一次</h3>
  <p>混元 3 截前 4 层，GPU B200×8 搬到 TPU v5p（8 颗芯片）。地基全部逐位，第 1 步 loss 只差«D1_T»；可是三百步以后，TPU 的 loss 平均比 GPU 低约 1.7e-4，参数范数也多涨一截。方法是通用的；具体的数只对这个设置（截 4 层、1000 步、学习率 «LR» 不变）成立。</p>
  <details class="foldfig"><summary><b>这一讲用到的几组运行</b>：① 到 ⑥ 是哪几份 GPU 运行、各用在哪，用到时回这儿查</summary>
  <table class="t9"><thead><tr><th>这一讲用到的几组运行</th><th>多少</th><th>用在哪</th></tr></thead><tbody>
  <tr><td>① GPU 默认设置</td><td>10 份</td><td>第 1 步那几级（L0–L3）的参照；第一节的尺子、§3.4 的基线贴边；§7.5 的零点是另跑的同设置 4 份（带探针）的均值</td></tr>
  <tr><td>② GPU 生产 golden（生产近似：整层重算＋路由回放）</td><td>5 份</td><td>§9.3 拆账的零点</td></tr>
  <tr><td>③ 只修一处的 golden（② 再改一处，改哪处第七节揭晓）</td><td>5 份、两两 10 对</td><td>定案前第一关的尺子（§3.3 的例子）</td></tr>
  <tr><td>④ 精确轨迹，② 的配置（整层重算＋路由回放）</td><td>1 份基线，另有 4 份扰动版</td><td>定案前第一关的基线；基线连同 4 份扰动版共 5 份，两两 10 对是交叉核对一的尺子</td></tr>
  <tr><td>④′ 精确轨迹（GPU 上全程 FP32 算术），不重算配置</td><td>1 份，另有一份扰动版和一份改过一处的</td><td>§7.5 的诊断</td></tr>
  <tr><td>⑤ 修好的 golden（② 再改一处，第七节揭晓）</td><td>10 份、两两 45 对</td><td>第一关的基线（取其中一份）和尺子（§9.4）</td></tr>
  <tr><td>⑥ 真精确（精确轨迹再改一处，第九节）</td><td>1 份</td><td>§9.3 的真精确参照</td></tr>
  <tr><td>TPU</td><td>3 种跑法；扰动臂 «ARMS_KEPT» 个＋基线，«CAL_PAIRS» 对</td><td>第一节的三个点；TPU 自己的执行噪声（§10.3 的回归）</td></tr>
  <tr><td>TPU 扰动组（定案后）</td><td>«ENSEMBLE_RUNS» 份（v5p «ENSEMBLE_V5P»、v7x «ENSEMBLE_V7X»），只改执行参数</td><td>§9.4 只报：TPU 不是挑出来的好样本</td></tr>
  </tbody></table></details>
  <p>编号先不用记，只记最后定下来的标准答案叫「修好的 golden」（表里标 ⑤，第九节讲它怎么来的）；用到别的那几份时会说名字。</p>
  <p>开头说的「第二次迁移」（混元 4 的缩小版）和「第三次迁移」（混元 4 的另一个尺寸），用到时再介绍。</p>

  <div class="note"><span class="t">后面常出现的三个词，先认个脸</span>
  <b>TPU 自比</b>：在 TPU 上只改一样东西（换并行、关一个编译开关、升版本……），跟一次确认过的 TPU 运行比。尺子仍借 GPU 重跑，因为 TPU 自己重跑逐位相同、量不出宽度。第二次迁移的很多例子来自它（§10.1 汇总）。<br>
  <b>运行库（libtpu）</b>：TPU 的编译器后端和运行时打包成的库；文中 0.0.46、0.0.52 是它的版本号（§10.2）。<br>
  <b>对齐记录</b>：这几次迁移的公开实验仓库，课件里的「代码：」链接都指向它，数字都能在里面追到结果文件。</div>

  <details class="foldfig"><summary><b>遇到不认识的词，回这儿查</b>：golden、噪声、BF16、张量核、MXU、FSDP、XLA、求和维、dgrad、Adam 的 ε、lse、留一法、定责、程序指纹……</summary>
  <ul>
    <li><b>golden</b>：GPU 上跑出来、拿来当标准答案的那条训练记录。</li>
    <li><b>专家、路由</b>：混元 3 是 MoE 模型，每层有很多组并排的前馈层（专家），每个 token 由路由器挑其中几个来算；路由偏置是路由打分上加的一个可学习的偏移。</li>
    <li><b>原子加</b>：很多线程同时往同一个格子里累加，硬件保证不丢数，但不保证先后顺序。</li>
    <li><b>设备</b>：软件看到的一个计算单元。v5p 一颗芯片是 1 个设备，v7x 一颗芯片是 2 个设备（Google Cloud TPU 文档）；所以 v5p 8 颗、v7x 4 颗都是 8 个设备，都对上 GPU 的 8 张卡。</li>
    <li><b>噪声（噪声参照）</b>：同样设置的几次运行两两之间的差；<b>N（噪声上限）</b>是其中最大的那个。</li>
    <li><b>扰动臂</b>：只改执行方式、不改数学的变体程序（比如改编译器的片上内存上限），用来给 TPU 造噪声。</li>
    <li><b>逐位相同</b>：两个数的每一个二进制位都一样，不只是「看着相等」。</li>
    <li><b>BF16 / FP32 / TF32</b>：三种浮点格式，指数都是 8 位，尾数分别是 7、23、10 位（§5.1）。</li>
    <li><b>张量核 / MXU</b>：GPU 和 TPU 上专门做矩阵乘的硬件单元。</li>
    <li><b>EP</b>：按专家切 —— 不同的专家放在不同的卡上，token 按路由结果发过去算。</li>
    <li><b>分布式优化器</b>：把优化器状态切开分到各张卡上存；梯度规约也从 all-reduce 改成 reduce-scatter。本讲的 GPU 默认组把它关掉：有一批老运行（推断当时开着）持续偏低一点，原因没查清。</li>
    <li><b>Megatron-Bridge、Transformer Engine</b>：GPU 这一侧用的训练框架，以及 NVIDIA 给它提供低精度矩阵乘等算子的库。</li>
    <li><b>FSDP</b>：把权重切开分到各张卡、用的时候再拼起来的数据并行（专题五 §2.3）。</li>
    <li><b>collective matmul</b>：本课叫「边传边乘」，编译产物里叫窗口化矩阵乘；一边在设备之间收下一片数据、一边算矩阵乘，把通信藏在计算后面（§5.4）。</li>
    <li><b>微批（microbatch）</b>：一步要算的一批数据再切成几份，一份一份地过，梯度加起来。</li>
    <li><b>dgrad</b>：反向里「对输入的梯度」那一次矩阵乘（例如 dh ＝ dlogits · W）。</li>
    <li><b>Adam 的 ε</b>：Adam 更新公式分母上加的一个小常数（这里是 «EPS»），防止除以零。</li>
    <li><b>logits / lse</b>：模型给词表里每个词打的原始分数；lse 是把它们按 log-sum-exp 合成的一个数，代表 logits 的整体水平。</li>
    <li><b>参数范数（param_norm）</b>：全部参数平方和开根号，一个数；<b>梯度范数（grad_norm）</b>同理，取裁剪之前的值。</li>
    <li><b>FP64 真值</b>：同一份输入用 64 位浮点算一遍，几乎没有舍入，当靶心用。</li>
    <li><b>XLA、融合</b>：XLA 是 JAX 用的编译器，TPU 上的程序都经它整图编译；融合是把相邻几个算子编进同一块一起算，中间结果不落回内存。</li>
    <li><b>excess precision</b>：允许编译器让结果比程序写的更精确（XLA 的 <code>xla_allow_excess_precision</code>，默认开）；表现为中间值不按 BF16 落地：成对的「转 BF16 再转回」可能被删掉，也可能在 HLO 里还在、后端编 kernel 时省掉（§5.5）。</li>
    <li><b>求和维（也叫收缩维）</b>：矩阵乘里被加起来、消掉的那一维，它的长度就是一次求和要加多少个乘积。</li>
    <li><b>路由槽位</b>：每个 token 选几个专家就占几个槽位；「翻了一个槽位」＝这个 token 有一个专家换了人（§4.3）。</li>
    <li><b>MARE</b>：mean absolute relative error，逐步相对差取绝对值再平均（§3.2）。</li>
    <li><b>超越函数</b>：exp、log、sigmoid 这类不能用有限次加减乘除精确算出的函数；硬件和库用软件近似，精度各家不同（§5.7）。</li>
    <li><b>病态</b>：结果对舍入极敏感；典型是两个几乎相等的大数相减剩下一个小数，大数上差一格，小数上就是一大截（§5.6 折叠）。</li>
    <li><b>第二次迁移</b>：同一套方法在混元 4（公开口径总参数 «C2_HY4_TOTAL_T»、激活 «C2_HY4_ACTIVE_T»）的缩小版上又走了一遍，它是一个更新的、带稀疏注意力的 MoE 模型；它的「小尺寸模型」保留全部结构、只缩宽度和层数，随机初始化，约 4.3 亿参数。</li>
    <li><b>第三次迁移</b>：混元 4 的另一个尺寸，从生产代码原样起步、一个开关一个开关地打开（§1.4、§5.4、§9.6）。</li>
    <li><b>MLA</b>：多头潜在注意力，先把 q、k、v 投到一个低秩的中间维，再投回每个头（DeepSeek-V2，arXiv:2405.04434）；混元 4 用它。</li>
    <li><b>纯数据并行</b>：每张卡存一整份权重、各算各的数据，梯度再跨卡求和（FSDP 则把权重也切开）。</li>
    <li><b>词嵌入</b>：词表里每个词一行向量；反向时，用到这个词的每个 token 的梯度都要加回它那一行，这一步叫 scatter-add（第八节）。</li>
    <li><b>下标</b>：按号去取或去加时用的那个号；一次处理多少个元素，就有多少个下标。</li>
    <li><b>五种集合通信</b>：all-reduce 各卡相加、每卡一份；reduce-scatter 相加后每卡只留一段；all-gather 把各卡的段拼成整份；all-to-all 每卡把自己的数据切成 n 份、第 i 份送给第 i 张卡；collective-permute 是点对点地传一块（§8.10）。</li>
    <li><b>HLO</b>：XLA 编译出来的程序文本；切分、融合做完以后那一份里，每个算子、每次跨卡通信的类型和形状都看得到（§8.10）。</li>
    <li><b>LLO、打包指令</b>：LLO 是后端给每颗 kernel 生成的机器指令；「打包」指把 FP32 舍成 BF16 存放的那类指令，数它就知道一处舍入有没有真做（§8.5、§9.7）。</li>
    <li><b>R</b>：被测离 FP64 的距离 ÷ 参照离 FP64 的距离；≤ 1.2 记「换了顺序」，更大记「丢了精度」（§4.2）。</li>
    <li><b>尺子、刻度</b>：拿来比的那把叫尺子：多步曲线上是噪声上限 N，第 1 步逐张量上是 GPU 几份重跑两两的最大差（混元 3 三份，第二次迁移现在六份）；「刻度」只指 FP32 记录值的分辨率（例如混元 3 的参数范数在 3042 附近一格约 2.44e-4）。</li>
    <li><b>留一法</b>：轮流拿一份 GPU 重跑当被测，用其余几份当尺子，看纯噪声按同一套判法能拿几分（§3.5）。</li>
    <li><b>定责</b>：查清一处差来自哪一边、为什么；第 1 步靠对 FP64 的 R 比，分「换了顺序」和「丢了精度」（§4.2）。</li>
    <li><b>程序指纹</b>：编译后的整份程序去掉元数据、名字按出现顺序重新编号以后的哈希；编译开关和运行库也相同时，指纹一样，结果就逐位相同（§10.3）。</li>
    <li><b>运行库（libtpu）</b>：TPU 的编译器后端和运行时打包成的一个库。换它，优化后的程序、生成的 kernel 都可能变，所以换版本不能当成「只换了加法顺序」（§10.2）。</li>
    <li><b>SparseCore</b>：TPU 芯片上的专用小核，官方定位是加速嵌入查表这类稀疏计算（Google Cloud TPU 文档）；本课用到两处：集合通信可以卸载给它；v7x 上有一处累加也交给了它（§8.6）。</li>
    <li><b>RMSNorm</b>：按均方根归一化、再乘一组可学的缩放；第二次迁移里它的舍入次数是一处边界（§5.5、§10.1）；换到运行库 0.0.52，v5p 上最后一层的两次舍入都真发生，主 loss 偏高（§10.2）。</li>
    <li><b>重算、路由回放</b>：重算是反向时把前向的中间结果再算一遍、省显存；路由回放是重算时沿用前向选中的专家，不重新选（§1.4、§2.3）。</li>
    <li><b>MTP</b>：多 token 预测，模型末尾多一层去预测再往后一个 token，记一项自己的 loss。</li>
    <li><b>Muon</b>：第二次迁移用的优化器，对矩阵参数先把动量正交化再更新；非矩阵参数仍用 Adam。</li>
    <li><b>多路残差混合</b>：混元 4 把残差流复制成 4 路，每层用一组小参数按权把几路混进、混出；「多路残差混合的小参数」指这组参数，数值上病态，定责时单列（「四路流」的说法见对齐记录 TPU↔GPU 差异清单：末端把四路流加权收成一路再进最后一层 RMSNorm）。</li>
    <li><b>索引器、蒸馏损失（KD）</b>：稀疏注意力里给每个 query 挑一批 token 的小打分网络；它靠一项蒸馏损失向完整注意力学（§5.6 折叠）。</li>
    <li><b>base2</b>：注意力内核把 eˣ 写成 2 的 (x·log₂e) 次方来算的写法；第二次迁移里那个常数跟着 BF16 舍了，所有尺寸的对齐配置现在都关掉它（§9.6）。</li>
    <li><b>HIGHEST</b>：TPU 上矩阵乘的最高精度档，用多遍 BF16 拼出近似 FP32；默认档对 FP32 操作数只算一遍 BF16（§5.4、§5.6）。</li>
  </ul></details>
''' + quiz("TPU 跟 GPU 的 loss 第 300 步差了 1e-4，算不算对不齐？",
           "光看这个数判断不了。要先量 GPU 自己重跑几次、两两差多少：如果 GPU 自己也常差到 1e-4，那就是噪声；如果 GPU 自己只差 1e-5 量级、而且 TPU 一直偏向同一边，才值得查。") + '''
  <p class="bridge">这一节留下的问题：数据、权重、配置，怎么做到两边一位不差？</p>
</div></section>

''' + sec("s二", "二", "地基 L0：输入必须一位不差") + '''
  <p class="lead">L0 只有一条要求：两边的输入一位不差。这一级不过，后面所有比较都没有意义。</p>

  <h3>2.1　数据：原样回放，不重读</h3>
__FIG_REPLAY__
  <p>两边各读各的数据集，顺序、切分、分到哪张卡，任何一处不同，比的就不是同一个训练。所以 TPU 不读数据集，只回放 GPU 存下来的那一份。</p>
  <details class="foldfig"><summary><b>「设备 d」怎么对上「GPU 第 d 张卡」</b>：在 TPU 上照做时的定义和核对办法</summary>
  <p>框架建设备网格时会按物理拓扑重排设备，网格上的第 d 片不一定是设备列表里的第 d 个；梯度累积拆微批又发生在编译后的程序里面。所以按「全局批的第几行」来排：GPU 第 d 张卡的第 m 份放在全局批第 d·G＋m 行（G 是每卡的微批数。这是本例框架的拆法，别的框架、甚至同一框架的别的训练器可能不同，按自己的代码核对）。排错了不报错，编译器会悄悄搬数据，分组照样错。核对办法：组好全局批以后遍历每个设备上那一片，用它的索引去对指纹。</p></details>


  <h3>2.2　权重：转过去，再转回来，逐张量比字节</h3>
  <p>从开源权重转成两边各自的格式，要处理改名、拼接、拆专家、转置。证明转对的办法是<b>转过去再转回来</b>，跟原始文件按原始字节逐个张量比：
    «N_TENSORS_T» 个张量全部相同，包括 «N_FP32_TENSORS» 个 FP32 的路由偏置和 «N_BF16_TENSORS_T» 个 BF16 张量；参数总数（不含路由偏置）«N_PARAMS»，两边一致。</p>
  <ul>
    <li><b>不抽样</b>：抽样比，没抽到的那个坏了查不出来。</li>
    <li><b>加载以后再读一遍</b>：训练开始前从设备上读出来，跟 GPU 第 0 步逐位比。主权重上小于 BF16 半格的错，舍成 BF16 后第 1 步前向看不见，第 2 步才分叉。</li>
    <li><b>参照那一侧也要自检</b>：转换往返逐位，不等于训练时真载入了。同期另一次迁移里，GPU 训练入口没设载入路径，框架按初始化方法随机建了权重，第一轮 golden 整轮作废。训练入口在第 1 步之前逐参数对检查点比一遍，不过就退出。</li>
  </ul>
  <details class="foldfig"><summary><b>张量个数怎么数</b>：两种命名下的计数对得上</summary>
  <p>这里按开源权重的命名数，一个矩阵一个张量：3 个 MoE 层 × «EXPERTS» 个专家 × 3 个矩阵（gate、up、down；Megatron 里 fc1 ＝ gate 和 up 拼在一起，fc2 ＝ down）＝ «HF_EXPERT_TENSORS_T»，加其余 «HF_OTHER_TENSORS» 个（norm、投影、路由、共享专家、词嵌入、输出层）。§7.5 状态互换的文件按 GPU 框架的布局存，q/k/v 合成一个、gate/up 合成一个，每个专家只剩 2 个矩阵：«MG_EXPERT_TENSORS_T» 加其余 «MG_OTHER_TENSORS» 个，同样的元素数出 «SWAP_STATE_TENSORS» 个。</p></details>
  <details class="foldfig"><summary><b>加载以后比了哪些张量</b>：参数全部比，FP32 主权重只比了一部分</summary>
  <p>参数全部比；FP32 主权重比的是 GPU 那边抓下来的 «MASTER_CMP_TENSORS» 个张量。</p></details>

  <h3>2.3　配置：对「实际生效的值」</h3>
__FIG_WD__
  <table class="t9"><caption style="caption-side:bottom;text-align:left;color:var(--gray);font-size:13px">摘自混元 3 几十行的对照表</caption><thead><tr><th>设置</th><th>两边对成什么</th></tr></thead><tbody>
  <tr><td>权重衰减</td><td>«WD_EFFECTIVE»（GPU 配置里写 «WD_CONFIG»，被调度器覆盖）</td></tr>
  <tr><td>学习率</td><td>常数 «LR»，逐步核对</td></tr>
  <tr><td>精度</td><td>BF16 计算，FP32 主权重，FP32 优化器状态</td></tr>
  <tr><td>批大小</td><td>全局 16；每卡每步 2 个微批，TPU 改成跟 GPU 一样的切法（细节见表下折叠）</td></tr>
  <tr><td>重算</td><td>把第一次选中的专家存下来，反向照着用（细节见表下折叠）</td></tr>
  </tbody></table>
  <p>表有几十行，每行一条证据。连切法都要对：只换梯度累积的切分，词嵌入梯度差就是 GPU 自己重跑那点差的 «MICRO_SPLIT_EMB_X» 倍。</p>
  <details class="foldfig"><summary><b>批大小、重算两行的细节</b></summary>
  <p><b>批大小</b>：全局 16；每卡每步 2 个微批，分两次前反向再累加。TPU 后来改成跟 GPU 一样的切法；第七节用到的 TPU 运行是改之前跑的（每设备一次算 2 条），只改这一项跑 30 步没有变好；第九节验收用的是改之后的配置</p>
  <p><b>重算</b>：GPU 生产近似（②，编号见第九节）每层整层重算；TPU 开部分重算（存下各投影、MoE 输出和路由结果，其余在反向里重算）。重算时 top-k 的顺序会变（首次前向不排序、重算时排序），加权求和就差一位；分数贴得很近时还会挑出别的专家 —— 所以把第一次选中的专家存下来，反向照着用（概率在重算里重新算）。归因参照 ① 不重算</p></details>
  <details class="foldfig"><summary><b>为什么微批的切法也要对</b>：累加怎么分组，舍入就发生在哪</summary>
  <p>同样是一步 16 条数据，一次算 2 条和分两次各算 1 条再相加，加法的分组不同，舍入就不同。GPU 自己做过对照：只换梯度累积的切分，词嵌入的梯度差就是 GPU 自己重跑那点差的 «MICRO_SPLIT_EMB_X» 倍。
    所以 TPU 也改成「每个设备分两次前反向再累加」，让设备 d 的第 m 份正好是 GPU 第 d 张卡的第 m 份。</p></details>
  <details class="foldfig"><summary><b>名字一样，写法不一样</b>：第二次迁移在配置和优化器里查出的四处，都不报错</summary>
  <ul>
    <li><b>优化器</b>：两边用的是同一个名字的优化器，可动量那一步做不做偏差修正、一部分参数走普通 Adam 还是 NAdam、一处归一化的 eps 怎么加，三处写法都不同。单测里拿参照框架的实现生成固定参照、逐步比每步更新：三处都不对上，相对差 «OPT_ALL_OFF»；三处都对上，«OPT_ALL_ON_T»。（另一个只测那部分参数的用例里，NAdam 对普通 Adam 差 «OPT_NADAM»。）</li>
    <li><b>重算</b>：「存下路由结果、反向不重选」这个设置，TPU 框架只在一种自定义重算策略下才读；配置写的是整层重算，它就被悄悄忽略，反向重新选路由。配置项名字对了，不等于生效了。（实测：被忽略时，反向会重选 «RESELECT_PCT_T» 的路由槽位；修好后为 0。）第三次迁移从生产代码原样起步，又踩了一次，而且看清了后果：生产默认整层重算，反向抓到的路由跟前向在 160 次两两比较里 96 次不同，平均 «M3_REROUTE_PCT»% 的 token（最多 «M3_REROUTE_MAX»%）的梯度是按另一套路由算的；参数范数 1000 步判 «M3_REROUTE_PN0» 倍、前 300 步 «M3_REROUTE_PN300_0» 倍。换成那种自定义重算策略、把前向选出的专家存下来，降到 2.27 ／ 1.19，步时不变。这是语义差，不是精度差：生产也该开。</li>
    <li><b>协议写了、参照做不到</b>：协议写「两边的路由偏置都做零均值更新」，可参照框架根本没有这个更新。TPU 开着它，偏置均值逐步漂开，带符号均值的 r 到了 19.1；TPU 关掉以后降到 «BIAS_ZM_SM_AFTER»（这时最差的一项换成平均绝对差，r 0.31）。</li>
    <li><b>统计量怎么跨微批合</b>：梯度累积把各份微批的辅助量一律求和，连一个「取最大」的统计也被加成约累积份数倍，拿它当阈值去裁剪就裁多了。（这一次训练里那个裁剪从没触发过，修了，数值上没有前后对比。）</li>
  </ul>
  <p>办法只有一条：对照表逐行对<em>写法</em>，不只对名字；参照框架里找不到实现的，协议写了也不算。</p></details>

  <h3>2.4　续训，和一个最便宜的自检</h3>
  <p>第 10 步存档、恢复后跑第 11–20 步，跟一口气跑下来逐位相同：数据位置、逐步的数、FP32 状态都一样。</p>
  <div class="note ok"><span class="t">第 1 步 loss：地基铺平了没有，一眼就看出来</span>
    第 1 步还没有任何更新，两边 loss 只差前向的舍入。混元 3（v5p）：TPU «LOSS1_TPU_T»，GPU «LOSS1_GPU_T»，相对差 «D1_PLUS_T»。
    <b>第 1 步就差得离谱，先回头查 L0，别去查数值。</b></div>
''' + quiz("权重转换后抽 «QUIZ_SAMPLED» 个张量比，全对，能说转换没问题吗？",
           "不能。没抽到的那 «QUIZ_UNSAMPLED_T» 个可能就有一个转置错了；而且 FP32 主权重上小于 BF16 半格的错，第 1 步前向根本看不见。要全量逐位比，加载以后再从设备上读一遍。") + '''
  <p class="bridge">这一节留下的问题：一千步的差，怎么压成几个能判的数？</p>
</div></section>

''' + sec("s三", "三", "指标（一）：逐步序列") + '''
  <p class="lead">loss、梯度范数、参数范数，每一步各出一个差。这一节把它们压成三个统计量，再拿噪声上限判。</p>

  <h3>3.1　每一步的相对差，和三个被比的量</h3>
  <div class="note"><span class="t">dₜ ＝（被测ₜ − 基线ₜ）÷ |基线ₜ|</span>
    分子：同一步的被测减基线（被测是 TPU 那条，基线是事先指定的一份 golden），带正负号；分母：基线这一步的绝对值，所有曲线一律如此。
    例：第 1 步 loss，（«LOSS1_TPU_T» − «LOSS1_GPU_T»）÷ «LOSS1_GPU_T» ≈ «D1_PLUS_T»。</div>
  <details class="foldfig"><summary><b>参照是谁、分母为什么一律用参照</b>：两种比较算法一样，只换参照；混元 3 的梯度范数用过两边均值</summary>
  <ul>
    <li><b>参照是谁</b>：GPU→TPU 对齐时，参照是事先指定的一份 GPU golden；TPU 自己改代码（§10.1）时，参照是一次确认过的 TPU 运行。读的是同一种逐步记录，算法完全一样。</li>
    <li><b>dₜ 的单位是 1</b>：dₜ ＝ 1e-5 就是 0.001%；它也不是 ULP（按浮点格子数差几格）。</li>
    <li><b>混元 3 第一关的梯度范数用过两边均值当分母</b>（§9.4 折叠）：梯度范数逐步涨落约 ±3%，以参照为分母时，带符号均值天生带一个正偏，约等于 E[δ²]（δ 是同一步上两份运行的相对差；本课推导）。假设两份运行的逐步梯度范数在共同趋势上下各自独立地抖约 3%，δ² 约 1e-3（这是条件，不是实测）；交换两边 N 能差一倍。第二次迁移实测这个偏置约 2e-7（换分母前后带符号均值 −2.6013e-4 → −2.5993e-4；理论估算约 4e-7），比噪声上限约 9.9e-5 小两个数量级，所以第二次迁移 2026-11-10 起所有曲线一律以参照作分母：一条规则，好解释。混元 3 的成绩没有重判，梯度范数仍按两边均值作分母。</li>
  </ul></details>
  <p>loss 之外，另两个量这样取：</p>
  <ul>
    <li><b>参数范数</b>：每一步把全部 FP32 主权重平方求和、开根号，一个数，混元 3 截 4 层约 «PN_LEVEL_T»。</li>
    <li><b>梯度范数</b>：全部参数梯度的同样运算，取<b>裁剪之前</b>的值（第 1 步约 «GRADNORM1_T»；裁剪系数约 «CLIP1_T» ＝ 裁剪阈值 «CLIP_GRAD_T» ÷ «GRADNORM1_T»，本课推导）。</li>
  </ul>

  <h3>3.2　带符号均值、MARE、最大单步</h3>
__FIG_DT__
  <table class="t9"><thead><tr><th>量</th><th>怎么算（在一个窗口里，去掉首步）</th><th>看什么</th></tr></thead><tbody>
  <tr><td><b>带符号均值</b></td><td>dₜ 直接平均，带正负号</td><td>随机的差正负抵消，剩下固定方向的偏移；最常出事的是它</td></tr>
  <tr><td>MARE（平均绝对相对差）</td><td>|dₜ| 的平均</td><td>整体抖多大，不管方向</td></tr>
  <tr><td>最大单步</td><td>最大的 |dₜ| 和它在第几步</td><td>拦住某一步突然跳一下</td></tr>
  </tbody></table>
  <div class="note"><span class="t">看逐步曲线，分两步</span>
    <b>第一步，看红线出没出灰带</b>：单步有没有乱跳，对应最大单步。<br>
    <b>第二步，看红线是不是围着零转</b>：有没有一直往一边偏，对应带符号均值。只做第一步，会漏掉最常出事的那一种。</div>
  <p>灰带是这样画的：每一步取所有 GPU 对在这一步的 |dₜ| 最大值，正负各画一边。所以这条带是逐步从不同的对里拼出来的，只用来看，不参与判定。判定用的尺子是另一套算法：每一对先把整段压成三个数，每种数再取所有对里的最大值（§3.3）。对齐记录的曲线工具还画一张累计带符号均值（第 1 步到第 t 步的平均）的图，同样只用来看趋势。</p>
  <p>一个真实的例子，第三次迁移图文 32K 一千步的 MTP loss：</p>
__FIG_MTP32K__
  <p>«M3_MTP32K_IN_PCT» 的步落在灰带里，最大单步是尺子的 «M3_MTP32K_R_MAX» 倍，平均绝对差 «M3_MTP32K_R_MARE» 倍：抖动和尖峰都正常。可 «M3_MTP32K_POS_PCT» 的步在零上面，带符号均值 «M3_MTP32K_SM_T»，是尺子 «M3_MTP32K_SM_N_T»（GPU 8 份两两 «M3_MTP32K_PAIRS» 对的最大值）的 «M3_MTP32K_R_SM» 倍，判不过。只有方向不对。图里的 7 条灰线是 GPU 另外 7 份各自对同一份基线，跟红线同一个口径：逐步看，它们在零上面的步数只占 «M3_MTP32K_G7_POS_T»，50 步平均线来回穿过零；TPU 的平均线一直在零上面。7 份的带符号均值也都略为正（«M3_MTP32K_G7_SM_T»，基线那份本身略低），TPU 是其中最大那份的三倍。这 7 条只是把「跟同一份基线比」画出来、方便看方向；尺子不是用它们算的，仍是 28 对两两、取绝对值的最大值，所以不会变窄。这一截偏后来定位到一处被编译器省掉的舍入（<a href="#mtp-input">§5.5 折叠</a>），补上以后的一千步验证还在跑。</p>
  <details class="foldfig"><summary><b>窗口表</b>：全程、尾段、第 1 步各取哪几步</summary>
  <table class="t9"><thead><tr><th>窗口</th><th>混元 3</th><th>第二次迁移</th><th>为什么</th></tr></thead><tbody>
  <tr><td>全程</td><td>第 2–1000 步</td><td>第 2–300 步／第 2–1000 步</td><td>看整段</td></tr>
  <tr><td>尾段（后段）</td><td>第 301–1000 步</td><td>第 101–300 步／第 101–1000 步</td><td>前段两条轨迹还没分开，会冲淡偏移</td></tr>
  <tr><td>第 1 步 d₀</td><td colspan="2">单独报，不进窗口</td><td>GPU 重跑第 1 步 loss 常常逐位相同，当不了尺子</td></tr>
  </tbody></table>
  </details>
  <details class="foldfig"><summary><b>窗口的两条细则</b>：只有一份原始数据；每个统计量是整窗一个数</summary>
  <p>两个窗口读的是同一条 dₜ 序列，只是求平均的范围不同；每个统计量是整个窗口算出的<b>一个数</b>，不是逐步的滑动平均。窗口写哪段，事先写进协议。</p></details>
  <p>loss 是结果，范数是总量。混元 3 那次，三个被比的量里<b>参数范数最先冲出去</b>：它是个累加器，只换顺序的差这步正、下步负，按步数的平方根散开；丢了精度的差有固定方向，按步数长。跑得够久，再小的系统偏差也会冲出噪声带：</p>
__FIG_INTEGRATOR__
  <details class="foldfig"><summary><b>盲区表</b>：范数、loss、路由偏置各抓什么、看不见什么</summary>
  <table class="t9"><thead><tr><th>被比的量</th><th>抓什么</th><th>盲区</th></tr></thead><tbody>
  <tr><td>参数范数、梯度范数</td><td>攒出来的固定方向的偏，最先有反应</td><td>只是一个长度：方向变了、一组多涨一组少涨看不出；被大张量主导，归一化权重、偏置上的错几乎看不见</td></tr>
  <tr><td>loss（各项分开）</td><td>训练效果变没变</td><td>只对很大的局部错才显（MTP 那一项权重偏 1e-3 时，MTP loss 本身 r 50，主 loss 只有 0.88；偏到 1e-2 主 loss 才 6.2）</td></tr>
  <tr><td>路由偏置均值、标准差</td><td>路由负载的漂移</td><td>均值在 0 附近来回穿，过零时 dₜ 和 N 一起放大，只抓得到大的语义差</td></tr>
  </tbody></table>
  </details>
  <p>所以逐步曲线要配第 1 步的逐张量比较（第四节）：一个抓攒出来的偏，一个抓偏在哪个张量。</p>
  <details class="foldfig"><summary><b>第二次迁移的证据</b>：故意埋的错，领头的常是范数，不是 loss</summary>
  <p>第二次迁移里故意让 loss 多乘 1.01（相当于漏除一个常数），梯度范数的 r 是 94.5，主 loss 只有 5.2；把 FSDP 改成纯数据并行（第八节的案例），梯度范数 2.62，主 loss 全程 1.85、后段 2.03（运行库 0.0.46；换到 0.0.52 同一改动领头的换成主 loss 后段 2.82，梯度范数 1.45 —— 哪一条先报警会变，所以几条都要看）。loss 告诉你训练效果变没变，可对不少数值问题反应慢；两个范数把几亿个数各压成一个，对固定方向的偏差最先有反应。</p></details>

  <h3>3.3　噪声上限 N，和怎么判</h3>
''' + R_CARD + '''
__FIG_JUDGE__
  <p>判法一句话：<b>r ＝ |被测| ÷ N</b>，「被测」是 TPU 对基线那一条曲线算出的某个统计量（带符号均值、MARE 或最大单步），N 是 GPU 几份重跑两两成对、同一个统计量里取最大的那个。≤ 1：落在噪声样本的范围里，过；＞ 2：比最坏的噪声还大一倍，不过；中间待定，先补参照再判（2 是协议事先定的界，留出样本少时 N 偏窄的余量）。</p>
  <p class="codeline">''' + code("PKG/checks/base.py", "判档：r ＝ |被测| ÷ N，≤1 过、＞2 不过") + "　" + code("PKG/checks/series.py", "逐步序列的三种统计") + '''</p>
__ANIM_CEILING__
  <p>参照对按跟被测一模一样的窗口、分母算三个统计，每个统计取所有对里的<b>最大值</b>当 N：被测如果只是噪声，就是又一个同分布的样本，应当落在已有样本的范围里。照着算一遍：</p>
__FIG_RPIPE__
  <details class="foldfig"><summary><b>N 和 r 的规则，一条不漏</b>：四步造 N、两条自动处理、两级取最差</summary>
  <ol>
    <li>同一设置的 GPU 重跑 K 份（混元 3 定案时 10 份；第二次迁移 300 步 8 份、1000 步 16 份）。</li>
    <li>两两配对：C(K, 2) 对（8 份 28 对、10 份 45 对、16 份 120 对）。</li>
    <li>每一对里一份当被测、一份当参照，用跟被测<b>完全相同</b>的窗口、分母、去掉第 1 步的规则，算出三个统计（带符号均值取绝对值）。</li>
    <li>每个统计的 N ＝ 所有对里的最大值；三个统计各有各的 N。</li>
  </ol>
  <p>两条自动处理：窗口里逐位相同的参照对剔除，不让它把 N 拉低；剔除后不足 3 对，最多判待定。r ＝ 被测的统计 ÷ 对应的 N；一项检查取三个 r 里最大的，一次比较取所有检查里最大的。≤ 1 过，1–2 待定，＞ 2 不过；2 是约定的余量，不是统计推导出来的。</p></details>
  <details class="foldfig"><summary><b>图上红黄那四个点是什么</b>：定案前的一次判定</summary>
  <p>图上红黄那四个点是定案前的一次判定：这次判定后来按最终 golden 重判过（§9.3），«ACC_PN_R» 倍留在这里当判法的例子。绿色那一簇是换成最终 golden 以后的同四项，全部过。</p>
  </details>

  <h3>3.4　参照要几对，比较只对一份</h3>
__FIG_PAIRS__
  <p><b>只对一份事先指定的基线比</b>，不对几份 golden 取平均，也不取最差：N 量的是「两份运行之间差多少」，被测也该是跟一份运行比，两边口径才一样。对平均比太松，对最差比太严。代价是基线自己也是一个样本，换一份当基线，读数会挪一点（图在折叠里）。</p>
  <details class="foldfig"><summary><b>换一份当基线，读数挪多少</b>（图）：10 份 GPU 运行各自当基线时，偏移读数怎么挪</summary>
__FIG_BASELINE__
  </details>
  <details class="foldfig"><summary><b>口径与旁证</b>：这把尺子只含执行噪声；A/A 测试；逐步的绝对差不是判定量</summary>
  <ul>
    <li>本课的噪声只含执行噪声：随机种子和数据顺序两边都固定。</li>
    <li><b>尺子只能借同设置的重跑</b>：第二次迁移换成生产配方（学习率调度、YaRN、路由偏置零均值三样一起改）跑 1000 步时，同配方的 GPU 两份拿旧配置的尺子一量，自己就 r 4.12 —— 借来的尺子不适用；补同配方 GPU 当尺子重判：先补到 8 份（这把尺子是看过借尺判档才定的），后来事先登记再补到 «PR_N_T» 份，两回都是 TPU 九项全过（只跑了 1 份，最差 «PR_WORST_T»）。主 loss 的带符号均值那一项，这把尺子比旧配置宽 «PR_LOSS_X_T» 倍：配方一换，GPU 自己的波动就变了，尺子得跟着重量。</li>
    <li>做线上实验的人有个老问法 ——「要是两边其实一模一样，这个差有多稀奇？」两组都吃安慰剂，叫 A/A 测试（Microsoft ExP 团队的 A/A 测试博客）。样本少时它会骗人：同一套超参只换随机种子跑 10 次，分成两组各 5 次，两组曲线就能「统计上显著不同」（Henderson 等，<em>Deep Reinforcement Learning that Matters</em>，arXiv:1709.06560）。</li>
    <li>另一种读法是看某一步的绝对差（例：「第 975 步参数范数多 «EXACT_TPU_PN975_T»」），那是看漂移的形状用的，不是判定量。</li>
  </ul></details>

  <h3>3.5　一次看几十个数：r 在 1–2 之间，先别急着判错</h3>
  <p>尺子只用 GPU 重跑：TPU 同一个程序重跑逐位相同，量不出宽度（理由在折叠里）。日常回归里判「只换了舍入位置」的改动，另有一把故意改执行参数造出来的 TPU 尺子（§10.3），两件事别混。</p>
  <details class="foldfig"><summary><b>为什么尺子只用 GPU 重跑</b></summary>
  <p>尺子为什么只用 GPU 重跑？第二次迁移里，TPU 同一个程序重跑，换一组机器、换一个集群，结果都<b>逐位相同</b>：拿它当尺子，宽度是 0，什么都判不过。硬给 TPU 造点波动来撑宽，又成了被测给自己定门槛（§4.3）。</p>
  </details>
  <p>第二次迁移一次比较要看 8 项检查（7 条曲线，主 loss 开两个窗口），每项 3 个统计，一共 24 个 r，最差的那个说了算。每个数超过自己尺子的机会不大，24 个里总有一个超的机会就大了。纯噪声按同一套判法能拿几分？把 GPU 的重跑轮流拿一份当被测，试一遍：</p>
  <p class="codeline">''' + code("models/hunyuan4/analysis/gpu_loo_r.py", "留一法") + "　" + code("docs/guide/ruler.md", "尺子的入门（文档）") + '''</p>
__FIG_LOO__
  <p>所以 r 落在 1–2 之间，先拿纯噪声的最差 r 对一下：不超过它、第 1 步也没有离 FP64 更远的张量（§4.2 末尾），就写「跟重跑分不开」。图第三行那六个分不开的 TPU 改动就是这样落定的。可「分不开」不等于「没改精度」：这六个里有两个动了精度，这把尺子只是量不出来（第十节讲尺子能抓多小）。还有一句实话：8 份到 16 份那几天，梯度范数的分母也同时统一成了参照，两样一起换的，各占多少没拆；扩到 16 份也是看到灰区以后才定的，算重判，不算前瞻检验。</p>
  <details class="foldfig"><summary><b>坦白</b>：这条读法是看到结果以后才补的；为什么份数少，带符号均值最吃亏</summary>
  <ul>
    <li><b>事后补的</b>：「拿纯噪声的最差 r 对一下」是 TPU 那批结果出来以后才想到的读法，所以它只用来解读，不改判档；对齐记录先把它登记下来，再拿之后的新改动和新尺寸去检验它站不站得住。尺子从 8 份扩到 16 份，也是看到灰区以后才定的，那六个改动是重判，不算前瞻检验。</li>
    <li><b>口径要对上</b>：«LOO16_1K_MAX_T» 是一整次比较几十个数里最差的那个。同一种轮换单看一条检查，纯噪声能到的最高分各不相同：主 loss 全程 «LOO1_LOSS»、后段 «LOO1_TAIL»，参数范数 «LOO1_PN»，路由偏置均值 «LOO1_BM»（本课按对齐记录的逐种轮换数据算）。拿哪一条的 r 去比，就要比那一条的最高分。</li>
    <li><b>份数少，尺子偏窄</b>：N 取各对的最大值，样本越少越窄（§3.4）。8 份补到 16 份，各项上限放宽 «N16_WIDEN_T» 倍，多数检查里放宽最多的是带符号均值（路由偏置均值 2.07 倍、梯度范数 «N16_GN_WIDEN» 倍），MARE 多半只放宽一成上下。原因（本课推导）：带符号均值量的是每份运行整体偏在哪，两两之差的最大值就是这几份里最高减最低，份数越多这个极差越大；MARE 是每一步抖动的平均，各对差不多。</li>
    <li><b>能不能先算出来</b>：一个统计量超过 n 对最大值的机会，约 1/(n＋1) 起；带符号均值约翻倍（模拟和留一法实测都显示），因为基线和那几份参照运行是共用的，配对彼此不独立。几十个统计量要是各自独立，至少一个超的机会是 1 − (n/(n＋1))^m（本课推导）。300 步：尺子 21 对、24 个统计，算出 «MC_300_T»，实测 «MC_300_OBS_T»；1000 步：105 对、27 个统计，算出 «MC_1K_T»，实测 «MC_1K_OBS_T»。这里算出和实测对上，是凑巧：配对共用运行让单个统计量超线的机会变大，统计量彼此相关又让「至少一个超」变小，两头抵消了。判档以留一法实测为准，规则不变。</li>
    <li><b>为什么补份数，不按 Bonferroni 把门槛加严</b>：Bonferroni 等于把每把尺子都按「看了几十个数」加严，可这些数彼此相关，那样会过严（推断）。粒子物理找新粒子时也有同样的问题（look-elsewhere effect），修正系数靠模拟来算；我们的模拟就是 GPU 多跑几份，直接量「几十个里最差那个」在纯噪声下落在哪。</li>
    <li><b>门槛不变</b>：判档仍是 ≤ 1 过、＞ 2 不过；「跟重跑分不开」是另起一列的读法。不干脆把 2 改大，是因为纯噪声能拿几分取决于份数和检查条数（300 步 8 份 «LOO8_300_MAX_T»、1000 步 16 份 «LOO16_1K_MAX_T»），没有一个放之四海的数；该做的是把份数补够。</li>
  </ul></details>
''' + quiz("尾段 loss 的带符号均值 r ＝ 1.6，MARE r ＝ 0.9，最大单步 r ＝ 0.7，怎么判？",
           "三个量里最差的决定结论：1.6 落在 1–2 倍之间，判待定。待定不能直接签：先加参照重判；再看纯噪声按同一口径能到多少（§3.5）—— 这里只看尾段 loss 一条，1000 步、16 份时纯噪声在这一条上最高 «LOO1_TAIL»，1.6 已经超了，往下一级去定责；拿整次比较的 «LOO16_1K_MAX_T» 来比就比松了。") + '''
  <p class="bridge">这一节留下的问题：偏在哪一层、哪个张量？</p>
</div></section>

''' + sec("s四", "四", "指标（二）：张量") + '''
  <p class="lead">第三节的曲线每步只有一个标量，说得出「偏了」，说不出偏在哪。这一节换另一类度量：第 1 步的激活、梯度、权重，一个张量压成一个数；几个指标各管一类错，各有盲点。</p>

  <details class="foldfig"><summary><b>比的是哪四类张量、几个张量怎么合成一个数</b>：逐张量取最差；参数范数是另一回事；按能量合起来只用来定责</summary>
  <ul>
    <li><b>四类</b>：权重（FP32 主权重，每步舍成 BF16 再用）；激活（前向每层的中间结果）；梯度（反向里对参数、对输入的导数）；优化器状态（Adam 的两个动量）。</li>
    <li><b>判定逐张量</b>：每个张量单独拉直、单独算 rel_L2；一条检查选中一批张量时取<b>最差的那个</b>，不把它们拼成一个大向量 —— 拼起来，大张量会把小张量上的错淹掉。</li>
    <li><b>参数范数不是张量比较</b>：它是每一步把全部主权重（约 124 亿个元素）合成的一个数，§3 的逐步序列比的是它。</li>
    <li><b>按能量合成</b>：一组张量当一个大向量，√(Σ‖a − b‖²) ÷ √(Σ‖b‖²)，R 比同理。只用来回答「整体差多少、整体谁离精确更远」（§4.3 的 «STEP1_ALL_E»、§9.4 的整网 R），不当门槛。</li>
  </ul></details>

  <h3>4.1　rel_L2、cos、范数比</h3>
__FIG_COS__
  <p>rel_L2 说差多大；拆开看，cos 说方向偏没偏，范数比说长短变没变。三个一起报，才分得清是转了个角还是整体缩放。</p>
  <details class="foldfig"><summary><b>三个指标是一回事的三个侧面</b>：rel² ＝ (k − 1)² ＋ 2k(1 − cos)；纯旋转为什么 rel_L2 不是 0</summary>
  <p>rel_L2 ＝ ‖a − b‖ ÷ ‖b‖（a 被测、b 参照）。记 k ＝ ‖a‖ ÷ ‖b‖（范数比；这里不用 r，免得跟 §3.3 的 r 混），把 ‖a − b‖² ＝ ‖a‖² ＋ ‖b‖² − 2‖a‖‖b‖cos 两边除以 ‖b‖²：
    <b>rel² ＝ (k − 1)² ＋ 2k(1 − cos)</b>（本课推导）。第一项是长短不一，第二项是方向不一；两个都为零，rel_L2 才是零。</p>
  <p>等长时 k ＝ 1，只剩 rel² ＝ 2(1 − cos)。纯旋转 θ：两支等长箭头的箭尖相距 2 sin(θ/2)，所以转 2.5° 左右（cos ＝ 0.999）rel_L2 就是 «COS_0999_RELL2_T» —— cos 看着很近，差向量已经不短了。</p></details>
  <details class="foldfig"><summary><b>其余几个指标</b>：标准差比、均值偏移、零元素比例、逐元素 p99、最大绝对差、KL，以及只诊断不当门槛的 ULP</summary>
  <table class="t9"><thead><tr><th>指标</th><th>抓什么</th><th>看不见什么</th></tr></thead><tbody>
  <tr><td>标准差比</td><td>分布变宽变窄</td><td>只看宽度</td></tr>
  <tr><td>均值偏移 ÷ 标准差</td><td>整体往一个方向偏了几个标准差：偏置、缩放类错</td><td>只改宽度不改均值的错</td></tr>
  <tr><td>零元素比例差</td><td>多出来或少掉的稀疏：漏了 mask、专家没收到 token</td><td>非零值对不对</td></tr>
  <tr><td>逐元素相对误差 p99</td><td>「大多数元素」错了多少（分母带兜底，不被近零元素干扰）</td><td>最大值</td></tr>
  <tr><td>最大绝对差（带坐标）</td><td>有没有单点大错、错在哪</td><td>不同张量之间不能比</td></tr>
  <tr><td>KL（只用于 logits）</td><td>对「下一个词」的预测分布差多少</td><td>只适用于 logits</td></tr>
  </tbody></table>
  <p>ULP 是「两个数差了几格」，按原始格式数格子（BF16 就按 BF16 数，按 FP32 数会放大 «ULP_FP32_PER_BF16» 倍）。差 1 格占大头，是「同一算法、不同累加顺序」；
    重心在 17 格以上，说明误差在累加里被放大了。可跨芯片、经过长求和以后，健康的差也常有几十格：GPU 自己两次运行第 1 步梯度，差 1 格的只占 «ULP1_T»，17 格以上占 «ULP17_T»（逐元素口径，跟整个张量的 rel_L2 不是一回事；梯度存的是 FP32，按 FP32 数格：最大差约 1.1e8 格，而 BF16 同号两数最多只差约 3.3e4 格，所以只能是 FP32 格）。所以它只诊断，不能判过不过。</p></details>

  <h3>4.2　谁的错：对 FP64 的 R 比</h3>
''' + R4_CARD + '''
  <p>第 1 步前向，GPU 默认设置下两次运行逐位相同，噪声是 0，当不了尺子（反向的梯度有噪声，第三节那把尺子量的是它）；FP64 就在这里派上用场。</p>
__FIG_RRATIO__
  <p><b>R ＝ TPU 离 FP64 的误差 ÷ GPU 离 FP64 的误差</b>：≈ 1 是两边一样远、只差加法顺序；明显大于 1，是 TPU 多出一份同量级的误差（约 √2 ≈ 1.41；不一定只是一次舍入，§5.4 就是多舍了好几次小的）。</p>
  <p>但<b>对齐的目标仍然是 GPU</b>（什么时候要换参照，第九节讲）。</p>
  <p>R 和 §3.3 的 r 别混：小写 r 只说两边差多少，TPU 和 GPU 换个位置它不变，说不出谁错；大写 R 对的是真值，才能定责。同一处可以 r 很大、R 却说 TPU 更准：</p>
__FIG_RR__
  <details class="foldfig"><summary><b>R 为什么通常在 1 附近，什么时候不能这么读</b>：分母接近 0；两边的精度路径本来就不同</summary>
  <ul>
    <li><b>通常 ≈ 1</b>：两边走同一条精度路径（BF16 输入、乘积精确、FP32 累加、出口舍一次），每一步舍入的大小同量级，只是先后顺序不同，离真值的距离也就差不多。</li>
    <li><b>分母接近 0</b>：GPU 那一份碰巧几乎正好（比如结果恰好落在格点上、或那一处根本没有舍入），R 会被除得很大，说明不了 TPU 错。看 R 之前先看分母本身有多大。</li>
    <li><b>误差不到一个 BF16 格</b>：R 是两种舍入实现之比，这时它会随上游一处写法在 «R_SWING» 之间摆（第二次迁移实测）。误差本身还在 BF16 舍一次的量级（rel_L2 1e-3 量级）以内时，R 过了 1.2 先别判 TPU 错，先看两边的绝对误差是不是都只有一格。</li>
    <li><b>噪声是 0 的张量</b>：GPU 重跑逐位相同（第二次迁移里有两层的一组梯度三份逐位），尺子量不出来，只能看对外部参照的 R。</li>
    <li><b>精度路径不同</b>：路由打分那次矩阵乘，GPU 离 FP64 «ROUTER_GPU_FP64»、TPU «ROUTER_TPU_FP64»，R ≈ «R_ROUTER_T»（§5.4）。R 远小于 1 只说明 TPU 这一处离真值近得多，不是「TPU 算错了」；门槛 «R_GATE» 只拦一个方向。</li>
  </ul></details>
  <p><b>两把参照一起用</b>：第 1 步每个张量既跟基线比（超没超出 GPU 重跑的噪声），又跟 FP64 比（离真值远了没有）。超出噪声、离真值没变远，是<b>换了顺序</b>；远了 1.2 倍以上，才是<b>丢了精度</b>。一个张量走一遍：</p>
  <p class="codeline">''' + code("models/hunyuan4/frameworks/megatron/step1_ref.py", "FP64 参照（GPU 一侧代码在 CPU 上整网 FP64）") + "　" + code("models/hunyuan4/analysis/tpu_self_compare.py", "从动量反推梯度、逐张量定责、程序指纹") + '''</p>
__FIG_STEP1__
  <details class="foldfig"><summary><b>照着算第 1 步</b>：梯度从哪来、尺子为什么有下限、1.2 是怎么定的、R 的两种写法、FP64 由谁来算</summary>
  <ul>
    <li><b>梯度从哪来</b>：TPU 上不挂钩子（§6），从第 1 步更新后的一阶动量倒推：m₁ ＝ (1 − β₁)·g，所以 g ＝ m₁ ÷ (1 − β₁)（用 Muon 的参数同理，动量缓冲 ÷ (1 − β)）；裁剪触发过的，再除以裁剪系数 ＝ 阈值 ÷ 裁剪前范数（混元 3 第 1 步裁剪前范数 «GRADNORM1_T»，系数约 «CLIP1_T»）。用二阶动量独立核一遍，混元 3 第 1 步相对差约 6e-8。</li>
    <li><b>尺子</b>：每个张量各自一个，等于 GPU 第 1 步三份两两的最大 rel_L2；倒推本身会带进几个 2⁻²⁴ 量级的相对舍入，所以尺子不低于 2⁻²⁰，GPU 三份逐位相同的张量也按这个下限算。</li>
    <li><b>1.2 的来历</b>（推断）：只差加法顺序时，两边离真值差不多远，R ≈ 1；多一次同量级的独立舍入，误差方差约翻倍，R ≈ √2 ≈ «R_ONE_EXTRA_T»；1.2 取在两者之间。</li>
    <li><b>R 的两种写法</b>：本节（GPU 搬 TPU）R ＝ TPU 离 FP64 ÷ GPU 几份离 FP64 的中位数（2026-11-11 起 GPU 6 份、两两 15 对；TPU 自比仍按 3 份）；GPU 几份逐位相同、且两边离 FP64 都小于一次 BF16 舍入（2⁻⁸ ≈ 3.9e-3）的张量，比远近没有意义，直接按过。TPU 自比时分母换成「基线、GPU 中位数里离得远的那个」，门槛同为 1.2。</li>
    <li><b>FP64 由谁来算</b>：一律用 GPU golden 那一侧的同一套模型代码（TPU 自比时也一样），在 CPU 上把第 1 步整网改成 FP64 重算；被测一侧的代码一行不用。裁判不能来自被测。</li>
  </ul></details>
__FIG_TWOAXIS__
  <p>左块是常态：九成张量超出 GPU 的噪声，一个都没离真值更远 —— 但这只对画出的 160 个成立：45 个多路残差混合的小参数单列没画（多路残差是混元 4 把残差流复制成 4 路、每层按权混合的结构），其中输出头的 3 个在 v7x 上离 FP64 远 18–46 倍，2026-11-12 单列加了倍数上限 5.22 以后，换 v7x 这次改判「超」。所以左块只说明大张量是换了顺序，换 v7x 本身不能算只换了顺序（§10.1）。中、右两块飞得远的都是词嵌入梯度（运行库 0.0.46 的读数；0.0.52 上 R 3.60、26.11，几乎一样。右块另有一个路由门控权重 1.20 倍刚过线）—— 为什么偏偏是它，第八节整节讲。</p>

  <h3>4.3　第 1 步，一个专家的梯度大得离谱</h3>
  <p>这是逐张量比较会误判的一个反例。混元 3 那次，第 1 步的梯度，比到的 «STEP1_TENSORS» 个张量（TPU 每层只存了 0、1 号两个专家的梯度，专家只比这两个；其余是全部非专家参数）合起来 rel_L2 只有 «STEP1_ALL_E»；可第 3 层 0 号专家的两个矩阵是 <b>«STEP1_EXPERT»</b>，大了约 «STEP1_X» 倍，而 GPU 自己重跑这两个张量一位不差，拿噪声尺子去判就是「不过」。这个大数出在路由上。</p>
  <details class="foldfig"><summary><b>路由翻转的静态图</b>：的名次与长红柱（下面动画讲的是同一件事）</summary>
__FIG_FLIP__
  </details>
__ANIM_FLIP__
  <p>MoE 的路由给每个 token 挑前 «TOPK» 名专家（共 «EXPERTS» 个），有的 token 第 8、9 名几乎平局；上游注意力两边舍入差一点点，排名就对调，选中的专家换了。给路由器喂两边完全相同的输入，翻转是 0；根子在路由器的输入差，不在路由器自己的精度。</p>
  <details class="foldfig"><summary><b>翻转的细账</b>：分差多小、输入差多大、换出换进几个、为什么是 √(k÷n)</summary>
  <p>翻了的 token 第 8、9 名「分数 ＋ 偏置」的分差中位数只有 «FLIP_MARGIN»；路由器输入换成 TPU 自己算出来的（跟 GPU 差 rel_L2 «ROUTE_IN_RELL2_LO» 到 «ROUTE_IN_RELL2_HI»），每层就翻 «FLIP_L2_T» 的 token。排名用「分数 ＋ 负载均衡偏置」，选中以后加权用不带偏置的分数：偏置只管选谁。在隔离回放的一批 «BATCH_TOKENS_T» 个 token 里，第 3 层 0 号专家只分到 «E0_TOKENS» 个，换出 «E0_OUT» 个、换进 «E0_IN» 个；每个 token 的贡献像方向各异的小向量，换掉 k 个、共 n 个，相对差约 √(k ÷ n) ＝ √(«E0_SWAPPED» ÷ «E0_TOKENS») ≈ «E0_SQRT_T»，跟 «STEP1_EXPERT» 同量级（推断；翻的都是第 8 名，门控权重小，所以实测低于这个估计也合理）。</p></details>
  <p>所以张量比较里先看零元素比例和路由槽位；路由翻了的专家，要按「两边选中同一批 token」重比，不能直接判算错。</p>
  <details class="foldfig"><summary><b>会翻的不只是两种硬件</b>：TPU 只改片上内存上限，路由也会翻</summary>
  <p>会翻的不只是两种硬件：同一个 TPU 程序只改编译器的片上内存上限，这两个专家梯度就差 «STEP1_EXEC_T»（TPU 对 GPU 是 «STEP1_TVG_T»）。我们一度以为只改执行方式路由不翻，其实这只在逐层隔离回放（每层输入取同一份，§6.2）里成立。</p>
  </details>
  <details class="foldfig" id="vit-flip"><summary><b>不选专家的层也会被带偏</b>：第三次迁移的图文模型里，视觉塔的 q、k 梯度越线</summary>
  <p>第三次迁移的图文模型（32K），从同一个状态各算一步（§6.3），越线的只有视觉塔的 q、k 投影：一种候选配置 «M3_VIT_N_1» 个（R «M3_VIT_R_1»），另一种 «M3_VIT_N_2» 个（R «M3_VIT_R_2»），两种配置越线的层不重合。视觉塔自己不选专家，可它的梯度要穿过后面的 MoE 层才传得回来；这一状态下各 MoE 层有 «M3_VIT_FLIP_PCT» 的 token 换了专家，传回来的已经不是同一份梯度。</p>
  <p>照上面的办法定责：把 TPU 自己选的路由灌给 FP64 重算，这些张量的 R 降到 «M3_VIT_RFLIP_1»（另一种配置 «M3_VIT_RFLIP_2»），全部记「路由翻转」，两格都由不过改判为过。R 小于 1 别读成 TPU 更准：同一状态下 FP64 用的是 FP32 主权重，两边实际乘的是舍成 BF16 的那份，取值点本来就不同（对齐记录，2026-11-14）。</p></details>
  <p>这种差 GPU 重跑的尺子看不见（宽度是 0）。规矩是：<b>每一级的尺子都只来自参照一侧的重跑</b>，被测自己的波动只作诊断 —— 让 TPU 给自己定门槛，等于自证。尺子看不见的差，超出的部分对 FP64 定责（§4.2）。</p>
  <details class="foldfig"><summary><b>这条规矩怎么来的</b>：两次迁移都把 TPU 的波动并进过尺子，又撤回</summary>
  <p>混元 3 当时在第 1 步这一级把 TPU 只改执行参数的重跑也并进尺子（并集，§9.4），后来撤掉，按只用 GPU 的尺子重判：第 1 步整步改判「不过 · 已解释」，多步那一级的结论不变。第二次迁移在多步那一级也并过一次，随后撤回。（正文的 «STEP1_TVG_T» 是 TPU 对 GPU、并集那组运行当前配置的读数，前面的 «STEP1_EXPERT» 是更早那次。）</p></details>
  <details class="foldfig"><summary><b>口径与旁证</b>：翻了多少槽位；推理和训练两套引擎也翻</summary>
  <ul>
    <li>隔离回放（每一层喂 GPU 那一层的输入，§6.2）里有 «FLIP_SLOTS» 的路由槽位翻了（«FLIP_TOKENS» 的 token 至少换了一个专家）。正文「每层翻 «FLIP_L2_T»」是另一种回放量的；两个比例来自两种量法不同的回放，不是开关造成的：那三个开关开不开，路由逐位不变。</li>
    <li>翻转不一定要紧：第二次迁移的稀疏注意力要给每个 query 挑一批 token，第 1 步 GPU 对 FP64 就有«SPARSE_Q_FLIP_T»的 query 至少翻了一个（占全部槽位 «SPARSE_SLOT_FLIP»）；可把 GPU 的选法灌给 FP64 注意力，输出误差几乎不变 —— 翻的都在名次的边上，那些 token 的注意力权重很小。所以翻转要报，不能拿「有翻转」直接判不过，要看翻掉的那一份有多重：MoE 里分到 token 少的专家就很重。</li>
    <li>同一份模型用推理引擎和训练框架各跑一遍前向，约 10% 的路由选得不一样，94% 的 token 至少在一层换了专家（Ma 等，arXiv:2510.11370）。常见的建议是路由打分用 FP32（Megatron Core 的 MoE 文档）—— 这里两边的路由打分本来就输出 FP32（输入仍是 BF16），挡不住。</li>
  </ul></details>

  <h3>4.4　两档门槛：严格档和参考档</h3>
  <table class="t9"><thead><tr><th>档</th><th>线从哪来</th><th>用途</th></tr></thead><tbody>
  <tr><td><b>严格档</b></td><td>这一讲的规矩：噪声尺子（≤ N 过、＞ 2N 不过）、L0 必须逐位、R 比 ≤ «R_GATE»</td><td><b>验收只看这一档</b></td></tr>
  <tr><td>参考档</td><td>另一个已经上线的 GPU→TPU 迁移项目用过的固定线：首步 loss 和窗口带符号均值都 ≤ «REF_LOSS_GATE»、逐层 rel_L2 ＜ «REF_RELL2_GATE»（只用于没有路由的层）</td><td>并排报告：离别人上线的线还有多远</td></tr>
  </tbody></table>
  <p>参考档是一个固定数，不随本次的噪声变，在 loss 这一条上常比严格档宽，所以常见「严格档不过、参考档过」：差已经超出噪声能解释的范围、要定责，但还在别人认为能上线的范围里。两档各有各的盲点，所以只并排报。</p>
  <p>别的项目借来的逐层 rel_L2 线只适用于没有路由的层；MoE 层按 R 比判。</p>
  <details class="foldfig"><summary><b>借来的那条逐层线</b>：它原本量的是什么、在 MoE 层为什么不适用</summary>
  <p>借来的线要问清它原本量的是什么。逐层 rel_L2 «REF_RELL2_GATE» 那条，出自那个项目在注意力层输出上的逐层前向检查（实测 «REF_LAYER_MEAS_T»），层里没有路由。MoE 层路由之后的读数会被近平局翻转（§4.3）整块改掉，这条线在那里没有依据：本例 MoE 那三层第 1 步回放读到 «L_RELL2_1_T»–«L_RELL2_3_T»，严格档的 R 比却都过，所以 MoE 层写「不适用」，按 R 比判；只在第 0 层稠密层用它（前向 «REF_LAYER0_FWD_E»、反向 «REF_LAYER0_BWD_E»，过）。</p>
  </details>
''' + quiz("TPU 某层的梯度 cos ＝ 0.99999，可范数比是 2.000。可能是什么错？",
           "方向全对、整体大了一倍：典型的「漏除了一个系数」，比如每卡每步 2 个微批，累加完忘了除以 2（本讲的配置正是每卡 2 个）。cos 对整体缩放完全看不见，靠范数比才抓得到。") + '''
  <p class="bridge">这一节留下的问题：R 超线，根子都在某一次舍入上 —— 两边的硬件到底在哪儿舍入？</p>
</div></section>

''' + sec("s五", "五", "浮点与硬件：差从哪来") + '''
  <p class="lead">两边对不上，差只有三种。语义（公式写得不一样）第二节对过了；只换了加法顺序的是噪声，尺子兜住。这一节看剩下的大头：精度 —— 一次矩阵乘把数两两相乘、再把乘积加起来，舍入落在哪一步，两边各自怎么舍。</p>
  <div class="note"><span class="t">这一节最该记住的两件事（第三次迁移）</span>
  <b>4 个设备上验过的程序，换到 8 个设备，几个投影静默多舍</b>：一次求和从舍 1 次变成舍 6 次，误差变成 GPU 的 1.41 倍，代码一行没改（§5.4）。<br>
  <b>同一个舍入开关，4K 序列上「没效果」，32K 上主 loss 第 51–300 步偏约 −24σ</b>：编译器在 32K 的程序里把那次舍入删了，只补这一处就回到噪声里（另跑的 120 步短跑、第 51–120 步窗口的读数 −0.2σ，同批只补另外三处之一的都在 −30.8σ 左右；§5.5）。<br>
  两件说的是一句话：换了条件，之前的验证就不作数。机制往下一节节讲，节末收在这句上。</div>
__FIG_KINDS__

  <h3>5.1　乘是精确的，加才舍入</h3>
__FIG_FORMATS__

  <h3>5.2　GPU（B200）张量核：每 «TC_GROUP» 个一组，抹零</h3>
''' + TC_CARD + '''
__FIG_TRUNCATE__
__ANIM_TRUNC__
  <p>收银台的比方：每一单抹零，单单都只少不多。张量核就是在每 «TC_GROUP» 个乘积这一单上抹零，而一次长求和有成千上万单。看上去是 FP32，末尾那几位其实从来没算过。（抹零只是一种偏法：小项相对整个和小到一定程度时，就近舍入也会把同号的小零头整个吞掉，见下面折叠。）</p>
  <details class="foldfig"><summary><b>慢一点：张量核这一单是怎么抹零的</b>：对齐窗口、截掉移出去的位、为什么越长越偏；十进制的例子</summary>
  <ol>
    <li><b>普通的浮点加法</b>：两个数相加，小的那个先右移对齐；移出去的那几位不扔，留几位保护位，加完再就近舍入。每一项都比累加值的一格大得多时，误差有正有负、大体抵消；可一大串同号小项、每项都不到半格时，就近舍入也会整项舍掉，同样偏向一边（§7.5 的 FP32 一次累加就是这样）。张量核的抹零不用等到这一步：每一单都截，只少不多。</li>
    <li><b>张量核一次加一单</b>：«TC_GROUP» 个乘积加上当前的累加值，一起相加。</li>
    <li><b>对齐到最大的那个指数</b>：这一单里谁的指数最大，所有数都往它对齐，只保留最高位往下大约 «TC_KEEP» 位的一个固定窗口。</li>
    <li><b>移出窗口的位直接扔掉</b>：不留保护位，不舍入。扔掉的永远是绝对值的一部分，所以每一单的误差都朝零。</li>
    <li><b>越长越偏</b>：单数越多，丢得越多，而且同一个方向；累加值越长越大，窗口跟着它往高位挪，后来的小乘积被扔掉的比例越来越大。</li>
  </ol>
  <p>十进制的例子（示意，只留 4 位）：累加值 1234 加 5.6 和 7.8，对齐到千位、只留 4 位，小数部分全扔：1234 ＋ 5 ＋ 7 ＝ 1246，真值 1247.4。累加值涨到 12340 以后，同样加 5.6 和 7.8，个位也落到窗口外：12340 ＋ 0 ＋ 0，一下少了 13.4。</p>
  <p>«TC_GROUP» 和 «TC_KEEP» 是从实测里拟合出来的（按这个模型重算，三层路由打分逐位对上），不是 NVIDIA 公开的规格。</p></details>
  <details class="foldfig"><summary><b>外面的人怎么量到同一件事</b>：V100 到 B200 的张量核测量、FP8 上的同类问题；附两种不抹零的加法作对照</summary>
  <ul>
    <li>V100、T4、A100 三代上就有人量出：张量核里乘积精确，加法向零舍入，只在最后规格化（Fasi、Higham、Mikaitis、Pranesh，<em>Numerical behavior of NVIDIA tensor cores</em>，PeerJ Computer Science，2021）。</li>
    <li>对 H100／B200 的比特级建模，BF16 进、FP32 出：16 项一组对齐、截断，累加器比 FP32 多 2 个对齐位（等价于本课说的「最高位后 25 位」），H100、H200 与 B200 相同 —— 跟本课在 B200 上独立量出的模型一致（Khattak、Mikaitis，<em>Accurate Models of NVIDIA Tensor Cores</em>，arXiv:2512.07004）。</li>
    <li>同一类问题在 FP8 上更凶：DeepSeek-V3 报告实测 H800 上 FP8 矩阵乘的累加只保留约 14 位，于是每 128 个元素搬到 CUDA core 用 FP32 加一次（arXiv:2412.19437 sec. 3.3.2）。口径不同，只借问题的类型，不比大小。</li>
    <li>对照：Kahan 求和是每一单把抹掉的零头记账、下一单补回去，误差几乎不随长度长；成对求和像淘汰赛，误差随长度按对数长。张量核的抹零正好是反面。</li>
  </ul></details>

''' + quiz("一段长求和在 GPU 张量核上算，输入全是负数，结果会偏大还是偏小？",
           "朝零偏：结果的绝对值偏小，数值上是偏大（离零近了）。截断总是丢低位，正数、负数都是往零的方向少一点 —— 这是「抹零」，不是「往下取整」。") + '''
  <h3>5.3　TF32：开关说的不算</h3>
__FIG_TF32__
  <p>TF32 是 NVIDIA 张量核上的 19 位格式：FP32 的矩阵乘一进张量核，就悄悄按 10 位尾数算。训练容器默认开着它，所以名义上的「FP32」常常不是 FP32。
    要拿来当精确参照的程序，每次启动都实测一遍，把实测值记进运行的元数据。</p>
  <details class="foldfig"><summary><b>BF16 训练里关不关 TF32 有没有影响</b></summary>
  <p>BF16 训练里，TF32 只碰输入本来是 FP32 的矩阵乘（本例第 1 步只有路由反向那十来次），所以「BF16 设置下关不关 TF32 没影响」和「精确参照必须真关 TF32」两句并不打架。</p>
  </details>
  <details class="foldfig"><summary><b>开关为什么这么乱</b>：PyTorch 各版本的默认值和一个压过一切的环境变量</summary>
  <p>PyTorch 里矩阵乘的 TF32 开关 1.7–1.11 版默认开、1.12 起默认关，cuDNN 那个开关一直默认开；另有一个环境变量 <code>NVIDIA_TF32_OVERRIDE=0</code>，压过所有框架和 NVIDIA 库的设置、全局关掉 TF32（NVIDIA cuBLAS 文档）—— 所以它连 Transformer Engine 那条路也能关掉（TE 对纯 FP32 矩阵乘在源码里写死用 TF32，不看 torch 开关）。较新的 PyTorch 又加了 <code>fp32_precision</code> 这一套新开关，旧的 <code>allow_tf32</code> 将弃用。</p></details>

  <h3>5.4　TPU：MXU，和设备一多就多舍的那几次</h3>
  <p>TPU 的矩阵单元 MXU 同样吃 BF16、乘积精确、在 FP32 里累加。同一份输入算路由打分，TPU 对 FP64 的相对误差是 «ROUTER_TPU_FP64»，GPU 是 «ROUTER_GPU_FP64»；长求和上 TPU 也会偏，但小得多。</p>
  <details class="foldfig"><summary><b>MXU 怎么累加</b>：MXU 的累加怎么舍是推断；「BF16 乘、FP32 累加」的公开出处</summary>
  <p>MXU 累加是不是就近舍入，没有直接证据，是推断。「BF16 乘、FP32 累加」有公开出处：Google Cloud TPU 的 bfloat16 文档（同页还写明 TPU 上转 BF16 是就近舍入、平局取偶，BF16 不支持非正规数），以及 ST-MoE 的脚注（arXiv:2202.08906）。JAX 里要 FP32 累加的写法是 <code>preferred_element_type=float32</code>；矩阵乘精度档 HIGH、HIGHEST 是用 3 遍、6 遍 BF16 去模拟 FP32，不是 IEEE FP32。</p></details>
  <p>TPU 自己多舍的地方在别处，而且带一个怪条件：<b>4 个设备上没有，8 个设备上才有</b>（设备＝软件看到的一个计算单元：v5p 一颗芯片一个，v7x 一颗芯片两个；所以混元 3 的 v5p 8 颗、v7x 4 颗都是 8 个设备）。混元 3 那次就量到、也修好了（下面第二张图右下角：qkv R 1.235 → 1.000），可一直没弄清它舍在哪一步、为什么有的投影中招有的不中招。第三次迁移从生产代码原样起步，把它从头到尾查清了。</p>
  <p><b>现象：同样是投影，一批多舍，一批不多舍。</b>第三次迁移的单算子回放（8 个设备）里，q、kv 的四个低秩投影、注意力的输出门、MLP 的 gate 和 up，TPU 离 FP64 都是 GPU 的 «M3_PROJ_R_T» 倍，约 √2，像多舍了一次（§4.2）；同样是投影的注意力输出投影 o_proj 和 MLP 的 down，却是 1.00。</p>
__FIG_INOUT__
  <p><b>规律：看 FSDP（把权重切开分到各设备、用时再收齐的数据并行）切的那一维是不是求和维。</b>往里投，模型宽度是求和维，每片只给一份部分和，要加；往外投，模型宽度是输出维，每片给完整的几列，只拼。中招的全是前一种。</p>
  <p><b>机制：编译器把一次矩阵乘拆成了几段。</b>它可以先收齐 8 片权重、一次乘完；也可以<b>边传边乘</b>（collective matmul）：权重片一片片传过来，到一片乘一段、加进累加器。后一种做法里，每段结果和累加器都是 BF16。一次 2816 宽的求和，GPU 舍 1 次，这里舍 «M3_TPU_ROUNDINGS» 次：</p>
__FIG_WINDOWED__
__ANIM_WINDOWED__
  <p>实测 R «M3_PROJ_R_T»。部分和小，舍掉的也小；按 6 次各自独立粗算约 1.6 倍，实测更低的原因没查清。</p>
  <p><b>为什么 8 个设备才出现：拆不拆，是编译器按快慢定的。</b>切法没变，也不看矩阵大小（8 个设备上连最小的路由打分矩阵也拆，4 个设备上连最大的 gate、up 也整块收），变的只是设备数：离线编译数出来就是上图右边那几根条。设备少为什么不拆是推断：环上首尾那几次传递的固定开销占比大，拆了省不下时间。</p>
  <p><b>修法：让这批矩阵乘声明 FP32 输出。</b>累加器跟着变成 FP32，循环结束只舍一次。硬证据两条：单算子 R «M3_PROJ_R» → «M3_PROJ_R_AFTER_T»；编译产物里那 20 个循环的累加器全部变成 FP32。步时 ＋«M3_COST_T»。</p>
  <details class="foldfig"><summary><b>打开前后各级读数</b>：第 1 步逐张量、1000 步曲线、步时（单份运行，曲线分不开）</summary>
  <table class="t9"><thead><tr><th>看哪一级</th><th>打开前</th><th>打开后</th></tr></thead><tbody>
  <tr><td>单算子：那批投影的 R</td><td>«M3_PROJ_R_T»</td><td>«M3_PROJ_R_AFTER_T»</td></tr>
  <tr><td>整网第 1 步：超出 GPU 两倍噪声的梯度张量</td><td>91 个</td><td>63 个</td></tr>
  <tr><td>1000 步：主 loss ／ 梯度范数 ／ 参数范数的 r</td><td>1.14 ／ 0.93 ／ 0.97</td><td>0.96 ／ 0.98 ／ 1.00</td></tr>
  <tr><td>步时</td><td colspan="2">＋«M3_COST_T»</td></tr>
  </tbody></table>
  <p>曲线别读过头：打开前主 loss 1.14 只是待定，打开后三项都 ≤ 1，可两级都只有一份运行，逐级加开关时主 loss、MTP、梯度范数本来就在 0.9–1.2 之间来回（参数范数更大，0.9–1.9），分不开。硬证据是单算子 R «M3_PROJ_R»→«M3_PROJ_R_AFTER_T»，和编译产物里 20 个循环的累加器全变 FP32。</p>
  </details>
  <details class="foldfig"><summary><b>第 1 步逐张量那边</b>：剩下的 63 个超 2N 的张量是怎么回事</summary>
  <p>第 1 步剩下的 63 个超 2N 的张量里，离 FP64 比 GPU 远（R ＞ 1.2）的 8 个，还没定责的 3 个（打开前 10 个、6 个）；再按路由翻转定责，这 3 个都归到路由翻转，剩 0（打开前剩 2）。</p></details>
  <details class="foldfig"><summary><b>两个旁注</b>：混元 3 上同一个开关为什么没动曲线；这跟 MLA 有没有关系</summary>
  <p>混元 3 用的是同一个开关：单层那一处修好了（qkv 投影 R «R_QKV_BEFORE» → «R_QKV_AFTER_T»），可对 1000 步的参数范数漂移几乎没用：参数范数那一项的 r（§3.3 的例子，对定案前的基线 ④ 精确轨迹）开关前 «GOLDEN_R4_T»、开关后 «ACC_PN_R»。第三次迁移里曲线跟着动了，可两级都只有一份运行，不按小数点排先后。</p>
  <p>它不是 MLA 特有的：稠密 MLP、共享专家的 gate 和 up 一样中招，凡是 FSDP 切的那一维正好是求和维的投影都会。MLA 只是让中招的矩阵变多：每层注意力 q、kv 各有两个往里投的矩阵（第二个的输入是低秩中间维，也被切）；标准的多头注意力只有 q、k、v 三个，还常融合成一个。</p>
  </details>
  <p><b>精度验证要在生产的设备数上编。</b>在 4 个设备的小切片上验，这件事完全看不到；8 个、16 个设备都拆，更大的规模没编过，按「会拆」准备。小切片上验也行，先离线编译一份生产设备数的训练步，数一数边传边乘的循环（编译产物里叫窗口化循环）和它们的累加器类型。</p>
  <details class="foldfig"><summary><b>裁判工具要跟生产编得一样</b>：回放把训练里（推断）不拆的投影也拆了</summary>
  <p><b>裁判工具要跟生产编得一样。</b>单算子回放把多 token 预测层的几个投影也编成了边传边乘，R 同样 «M3_PROJ_R_T»；可训练步的编译产物里没有给多 token 预测层的那组边传边乘循环，推断这几个矩阵乘在训练里是先收齐权重再整段乘的，回放这几格不代表生产。更早一版回放没把输入按 batch 切，投影 R 到了 «M3_REPLAY_NOBATCH_R»，也是工具和生产切法不一样。用回放之前，先拿编译产物对一遍：工具里的程序和训练步里的，是不是同一种写法。</p></details>
  <details class="foldfig"><summary><b>这个开关在代码里怎么开</b>：JAX、PyTorch 各叫什么；反向那几处要不要一起改</summary>
  <p>落到代码上，是给那处矩阵乘指定 FP32 累加（JAX 里给 <code>jax.lax.dot</code> 写 <code>preferred_element_type=jnp.float32</code>，出口再转回 BF16），不是加一个编译参数。别的框架要找的是「BF16 矩阵乘的部分和用什么精度相加」这一项；例如 PyTorch 在 GPU 上有 <code>torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction</code>，设成 False 就关掉（PyTorch 的 CUDA 语义文档）。反向那一侧，第三次迁移的编译产物里不用改：权重梯度一侧的累加器本来就是 FP32；求输入梯度时，o_proj、down、q_b、kv_b 是权重先收齐再在本卡乘；gate、up 和路由打分只是按列写各自的一段；都没有部分和相加。</p>
  <p>另一条路是用编译参数把边传边乘整个关掉。混元 3 试过：那样连 GPU 本来就有的一次舍入也省掉了（投影输出不舍就直接进了残差加），MLP 前归一化的 R 到了 1.64，反而对不齐 —— 多舍少舍都不行，所以用开关不用参数。</p></details>
  <p class="codeline">''' + mt(3, "src/maxtext/configs/base.yml", "L148", "开关 dense_fp32_accumulation") + "　" + mt(3, "src/maxtext/layers/linears.py", "L215-L227", "矩阵乘里 FP32 累加、只舍一次") + '''</p>
  <h3>5.5　excess precision：融合里省掉的舍入</h3>
__FIG_EXCESS__
  <p>excess precision（下文简称「省舍入」）是 XLA 的一个默认开着的选项：允许结果比程序写的更精确。最常见的表现是融合块里跳过中间那几次「舍到 BF16」，只在出口舍一次。</p>
  <p>它开着时也在替你省舍入。第二次迁移的 TPU 自比里，v5p 上词嵌入梯度「飞出去」两次（§4.2 末尾）：一次是把 FSDP 改成纯数据并行，一次就是关掉 excess precision（R «EMB_NOEXC_X»）。两次都出在词嵌入梯度上。为什么偏偏是它，第八节把这一处整个查一遍。</p>
  <details class="foldfig"><summary><b>关掉省舍入的读数与机理</b>：混元 3 关掉它的三条读数；v7x 上变远的地方；TPU 为什么会「更精确」；后端指令条数</summary>
  <table class="t9"><thead><tr><th>混元 3 关掉 excess precision</th><th>结果</th><th>读法</th></tr></thead><tbody>
  <tr><td>单层回放</td><td>每个张量对 GPU 反而都更远</td><td>省掉舍入的结果本来更接近 GPU</td></tr>
  <tr><td>1000 步 loss 偏移</td><td>小了一半</td><td>loss 对执行方式本来就敏感（只改片上内存上限也挪 «ARMS_SPREAD_T»），分不出是不是效果</td></tr>
  <tr><td>1000 步参数范数漂移</td><td>一点没少</td><td>判据看它：没用</td></tr>
  </tbody></table>

  <p>（v7x 上关省舍入后，变远的地方跟 v5p 不一样，§10.1。）</p>
  <p>为什么 TPU 上会有这种「更精确」：一份公开的事故复盘（Anthropic，2025 年 9 月）提到 TPU 的向量单元按 FP32 原生计算，编译器可以把融合块里的一些运算按 FP32 算。</p>
  <p>对齐记录数过编译后的后端指令（静态条数，运行库 0.0.52）：v5p 整步 1750 颗 kernel 里没有一条 BF16 算术指令；v7x 的向量单元有 BF16 算术，整步有 40 来颗带它，条数最多的几颗是加 0 拼接、取最大（结果精确），以及 BF16 的乘和加。静态条数看不出它们会不会多舍一次。</p>
  </details>
  <p>省舍入还有一个副作用：代码里写「转 BF16、再转回 FP32」想舍一次，这一对落在同一颗融合里，会被整个删掉，等于没舍。要真舍，写 <code>lax.reduce_precision</code>（强制舍入），编译器不会删。</p>
  <details class="foldfig"><summary><b>一个坑</b>：「先转 BF16 再转回来」会被编译器整个删掉</summary>
  <p>想在 TPU 上把 FP32 权重舍成 BF16 再用，写成「转 BF16、再转回 FP32」两步，excess precision 开着时，编译器会把这一对转换直接消掉，权重根本没舍。
    混元 3 上这样做出的「舍过的参照」实际用的还是 FP32 主权重，量出来的结论整个作废，改用 JAX 的 <code>lax.reduce_precision</code>（BF16 对应 <code>exponent_bits=8, mantissa_bits=7</code>）才对 —— 它不会被优化掉；输出类型仍是 FP32，只是数值已经舍到 BF16 的格点上。自查办法：把编译后的程序转储出来，看那处还有没有成对的 convert —— 可第二次迁移的例子说明，convert 都还在，后端照样可以只省一半，所以更稳的是把选项关掉重跑一次对比，或者直接在那一处写 <code>reduce_precision</code>。</p></details>
  <details class="foldfig"><summary><b>第二次迁移的一例</b>：一处中间值两处用，一处没舍；1000 步五次偏高（下面 RMSNorm 那件讲的是同一件事）</summary>
  <p>第二次迁移里，同一件事成了一处真正的偏差。GPU 在一处中间值上每次都存成 BF16，TPU 的代码也写了「转 BF16」；可把紧挨着的一处四路加权和照 GPU 的顺序改了（从配对相加改成依次相加，§5.6）以后，这次舍入在 TPU 上丢了一半。这是在一个最小复现里看到的：编译后的程序里「转过去再转回来」都还在，可后端生成指令时，算均方那一步用的是没舍的 FP32，归一化输出那一步用的是舍过的值，两步不再用同一个数：</p>
__FIG_ELIDE__
  <p>1000 步下来五次 TPU 运行的 loss 都偏高（«ELIDE_RANGE_T»），其中 «ELIDE_OVER» 次超出 GPU 自己的 ±«ELIDE_GPU_MAX_T»。在那两处显式写一次「舍到 BF16」，回到 «ELIDE_FIX_T»，判定 r «ELIDE_R_T»，回到线以内。</p>
  <details class="foldfig"><summary><b>「长跑偏高归到这一处」为什么只算推断</b></summary>
  <p>修了就回来，这是唯一的直接证据。整网 HLO 看不出这一层，后端指令只在最小复现里数过；第 1 步定责时还发现，单这一处解释不了：要连同第 0 层分支输入、层内的流三处一起，才对上第 0、1 层的读数（第 5 层仍对不上）。</p></details>
  <p>上一代配置下 TPU 主 loss 有一截共同偏高（8 份平均约 «MINI_TPU_HIGH_T»，GPU 每份对其余各份在 ±«MINI_GPU_SPREAD_T» 以内），几乎全在第 550 步以后才显出来。主因后来定到多路残差流梯度的舍法；可现行对齐配置下后段跟不修分不开，主 loss «C_LOSS_R_T» 倍只是刚好过线，为什么还在查（§9.7）。</p>
  <p class="codeline">''' + mt(4, "src/maxtext/configs/base.yml", "L1630", "多路残差强制舍 BF16 的开关（ihc_force_bf16_rounding）") + "　" + mt(4, "src/maxtext/layers/mhc.py", "L567-L573", "用 reduce_precision 舍，编译器不会删") + '''</p>
  </details>
  <p><b>第三次迁移碰上了这个副作用，还多了一层：删不删，跟着切法变。</b>RMSNorm 先把 FP32 的 scale 舍成 BF16 再乘（照 GPU），代码写的是「转 BF16 再转回」。4K 序列的程序里这次舍入在；32K 的程序里，离线编译一扫，8 个 norm 全被删了，主 loss 一路偏低。逐个加开关二分，只补这一处就回到噪声里（读数见图；σ 是 GPU 8 份的份间标准差）。为什么换了切法就删，是推断。</p>
__FIG_NORMSCALE__
  <p>为什么第 1 步看不见：scale 初值 1.0 正好在 BF16 格点上（这个盲区 §6.4 专门讲），要等 FP32 原值偏开才显（跟 §5.6 输出头那件同一个盲区）。4K 上验过「这个开关没效果」，是因为那里舍入本来就在 —— <b>换了切法，要重验</b>。</p>
  <p>所以生产代码里这一类「转 BF16 再转回」，别交给编译器，显式选一种：要跟 GPU 一致（从 GPU 迁移、接着 GPU 的检查点续训、拿 GPU 当参照排查），就用强制舍入（<code>lax.reduce_precision</code>）钉死，多数地方步时几乎不变（多 token 预测的 logits 那一处 ＋8.1%，单份）；要精度，就删掉往返、明确用 FP32。开强制舍入为的是一致，不是更准：怕的是它随切法和编译器版本悄悄变。</p>
  <details class="foldfig"><summary><b>写法规则</b>：四种想要的效果，各怎么写；第三次迁移上会被删的五处（另有 MTP 输入一处）</summary>
  <table class="t9"><thead><tr><th>想要的效果</th><th>写法</th><th>编译器会不会改</th></tr></thead><tbody>
  <tr><td>舍成 BF16，再继续用 FP32 算（照 GPU 的舍入）</td><td><code>jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)</code></td><td>不会删</td></tr>
  <tr><td>同上，写成 <code>x.astype(bf16).astype(f32)</code></td><td>不要这样写</td><td>往返落在同一颗融合里就会被删</td></tr>
  <tr><td>真的用 BF16 存或传（省显存、带宽）</td><td>只写 <code>x.astype(bf16)</code>，后面不转回</td><td>不会删：下游拿到的类型就是 BF16</td></tr>
  <tr><td>要 FP32</td><td>不写任何往返</td><td>—</td></tr>
  </tbody></table>
  <p>第三次迁移上会被删的五处，2026-11-14 又查出 MTP 输入一处（见下一个折叠）（RMSNorm scale 只在 32K 被删；正交化迭代一部分删、一部分保住）：多路残差的分支输入、流更新、收拢处（不补时第 1 步那几个小参数 R 超 1.2，参数范数 1.65 倍）；RMSNorm 的 scale（就是上面这处）；一处 FP32 sigmoid 的输出（没有可测影响）；Muon 正交化每步迭代的结果（参数范数第 300 步后一路往下漂，补上不漂）；多 token 预测输出头的 logits（交叉熵吃的是 FP32 logits）。查法都是 §9.7 那一招：离线编译，扫「FP32 → BF16 → FP32 落在同一颗融合里」的位置。</p></details>
  <details class="foldfig" id="mtp-input"><summary><b>图文 32K 又查到一处</b>：多 token 预测层的输入少舍一次，MTP 层梯度一齐差几乘十的负三次方</summary>
  <p>多 token 预测（MTP）层的输入，是主干末层归一化的输出：代码先把它存成 BF16，进 MTP 自己的归一化再转回 FP32，又是一对「转 BF16 再转回」。GPU 两次都舍；TPU 开着省舍入，把中间那次省了。</p>
  <p>少舍一次本来只差半格、有正有负，可这里紧跟一个乘法：输入已经归一化过，MTP 的归一化再乘一个约 «M3_MTPIN_COEF» 的系数，离 1 很近。输入在 BF16 格点上、乘完再舍回 BF16，结果常被吸回原来那个格点。两次都舍的一边（GPU）因此整体偏大约 «M3_MTPIN_BIAS»；只舍其中一次，都没有这个偏差（FP64 仿真逐项核过）。这一截占 MTP 残差流的大头，经过 MTP 的末层归一化和输出头，成了整个 MTP 层梯度的同一个比例差：19 个张量两边差 «M3_MTPIN_C_T»，全部同号，GPU 6 份之间只差约 «M3_MTPIN_C_GPU»。</p>
  <p>只在 MTP 输入上补一个强制舍入：同一状态算一步，MTP loss 跟 GPU 的差 «M3_MTPIN_LOSS_BEFORE_T» → «M3_MTPIN_LOSS_AFTER_T»，主 loss 逐位不变；MTP 层 18 个张量（前面说的 19 个去掉路由 gate，判法不计它）里 «M3_MTPIN_FIXED» 个的比例差回到 ±4e-4 以内。按事先定的判法这一格仍记「不过」：q 路三个还差 «M3_MTPIN_QPATH_T»，推断 q 一侧的反向还有一处被省掉的舍入，原来被整体那个比例差盖住了；1000 步上的效果还没有读数（对齐记录，2026-11-14）。</p></details>

  <h3>5.6　差几乎都出在 16 位和 32 位的交界</h3>
__FIG_BOUNDARY__
  <p>第二次迁移把数值上的对不齐一处处查下来，几乎每一处都是一个精度换挡的地方：一个值从 16 位进 32 位、从 32 位落回 16 位，或者 GPU 上悄悄打开的 TF32；剩下一类是 32 位对 32 位，同样是 FP32 的 exp、log，两边的实现精度差几十倍（§5.7）。连「操作数转成 FP32」都不等于 FP32 矩阵乘：TPU 默认对 FP32 操作数也只算一遍 BF16（图第 2 行）。</p>
  <p>还有一种是<b>该留 32 位的中间值存成了 16 位</b>：这是写错了，不是噪声，哪怕它小到 GPU 重跑都盖得住。参照的写法跟生产对不上时，以生产为准，改参照 —— 第 4 行改的就是 GPU。</p>
  <p class="codeline">''' + mt(4, "src/maxtext/layers/mhc.py", "L547-L562", "多路残差四路相加的两种写法（开关 ihc_stream_sum_unrolled；注释说 HIGHEST 是多遍 BF16，那是旧说法，以编译产物为准）") + "　" + mt(4, "src/maxtext/configs/base.yml", "L229-L233", "输出头单独设精度（logits_matmul_precision）") + '''</p>
  <details class="foldfig"><summary><b>HIGHEST 也只是近似</b>：多遍 BF16 拼的；编译器还可能把它改写成逐元素 FP32</summary>
  <p>要更接近 FP32，得给那一处单独设 HIGHEST（多遍 BF16 拼出来的近似）。可一处写着 HIGHEST，不等于它真上了矩阵单元：编译器可能把小的加权和改写成 FP32 逐元素乘加，那时它跟 GPU 只差加法顺序。是不是多遍 BF16，看编译后它还是不是矩阵乘。</p>
  </details>
  <p><b>第三次迁移把图第 2 行那一格看清了：默认档舍的是离开格点的 FP32 主权重。</b>第 1 步主权重还在格点上，看不见；第 2 步起 logits 误差是真 FP32 的几千倍（单算子回放、模拟离格，本课复算）。矩阵乘精度档有三档，前 300 步主 loss 的 r：默认（1 遍 BF16）«M3_LMH_DEF300»、high（3 遍）«M3_LMH_HIGH300»、highest（6 遍）1.31；highest 贵一倍却不更好，因为差只是离格那一截，补回来就够（单份运行）。第 1 步超 2N 的张量 91 个，开前开后个数一个不变 —— 梯度其实变了，可这点变化比各张量离 FP64 的误差小两个量级，逐张量看不出来。推荐 high，步时 ＋«M3_LMH_COST_HIGH_T»。</p>
  <details class="foldfig"><summary><b>输出头三档的图与读数</b>：为什么第 1 步看不见、三档怎么选</summary>
  <p>生产的输出头写 logits ＝ x · W，两边升成 FP32 再乘，精度档没单独设。x 是末层归一化的输出，本来就是 BF16，舍了不变；W 是 FP32 主权重。第 0 步 W 还在 BF16 的格点上，默认档舍了也不变，所以第 1 步三档前向一样；训练一步以后主权重离开格点，默认档把它舍回去，logits 的误差一下子成了真 FP32 的三千到五千倍（模拟离格 1 到 10 步）。第 1 步看不见，跟 §5.5 的 RMSNorm scale 是同一个盲区。前 300 步主 loss 的 r 是 «M3_LMH_DEF300»，判不过。</p>
__FIG_LMHEAD__
  <p>改成 high（拆成两段 BF16，算 3 遍）就够了：前 300 步 «M3_LMH_HIGH300»，1000 步 4K、32K 的主 loss 都过，步时 ＋«M3_LMH_COST_HIGH_T»；highest（6 遍）曲线一样好，步时 ＋«M3_LMH_COST_HIGHEST_T»。词表有 «M3_VOCAB_T» 个词，多算一遍就贵一截。带走一句：<b>BF16 检查点起训时，权重侧的精度开关第 1 步测不出来</b>，主权重得先走出格点；判这类开关要看曲线，或者拿训练几步后的权重单独回放。</p>
  </details>
  <details class="foldfig"><summary><b>边界清单</b>：两次迁移碰到的十三类（本课整理，单列了「Python 常数」一类），各举一例</summary>
  <table class="t9"><thead><tr><th>边界</th><th>例子</th><th>怎么对上</th></tr></thead><tbody>
  <tr><td>矩阵乘 FP32 累加，结果先存 BF16 再给下一个算子</td><td>注意力打分 q·k，参照自己写的不融合实现（图第 4 行）</td><td>让它留 FP32，照生产的融合内核</td></tr>
  <tr><td>跨分片的部分和先舍 BF16 再相加</td><td>FSDP 切开求和维的投影（§5.4）</td><td>部分和留 FP32</td></tr>
  <tr><td>归一化舍一次还是两次</td><td>RMSNorm（图第 5 行）</td><td>FP32 里算完只舍一次</td></tr>
  <tr><td>逐元素函数在 FP32 里算再舍，还是直接在 BF16 上算</td><td>门控激活 silu·up：GPU 舍两次、TPU 默认舍一次，只有 «GLU_BITWISE» 的元素逐位；注意力输出门的 sigmoid 两边也一个 FP32、一个 BF16，TPU 不比 GPU 远（R «OUTGATE_R_T»），定责、不处理</td><td>按 GPU 的写法逐个算子舍</td></tr>
  <tr><td>路由打分要 FP32</td><td>top-k 的集合跟 FP64 不同</td><td>两边路由打分都设 FP32</td></tr>
  <tr><td>FP32 操作数在 TPU 默认精度下只算一遍 BF16</td><td>名义 FP32 的输出头（图第 2 行）；HIGHEST 是多遍 BF16 近似，但编译器可能把小的加权和改写成逐元素 FP32</td><td>给那一处单独设精度；先看编译后它还是不是矩阵乘</td></tr>
  <tr><td>乘的是 FP32 主权重还是 BF16 权重</td><td>路由打分的权重（图第 1 行），另一处投影也是</td><td>按参照用哪份就用哪份</td></tr>
  <tr><td>GPU 上 TF32 悄悄打开</td><td>名义 FP32 的参照比 BF16 还偏（§5.3）</td><td>启动时对 FP64 实测</td></tr>
  <tr><td>梯度以 BF16 落地</td><td>norm 权重、输出层、词嵌入（§9.4）</td><td>梯度直接累加进 FP32</td></tr>
  <tr><td>优化器内部的精度</td><td>Muon 正交化的迭代 GPU 在 BF16 里算、TPU 默认 FP32（更准）</td><td>对齐时照 GPU 舍，生产不必；写法差（§2.3 折叠）要对</td></tr>
  <tr><td>Python 常数被转成 BF16</td><td>注意力指数用 2 为底时，常数 log₂e 跟着 BF16 的 q 舍成 1.4453125，logits 整体乘 1.0018（第二次迁移真实宽度第 1 步）；混元 3 的注意力缩放常数也舍成了 0.08837890625</td><td>常数在 FP32 里乘，或不用 2 为底；第二次迁移的对齐配置因此全部关掉 base2</td></tr>
  <tr><td>FP32 超越函数的实现精度（32 位对 32 位）</td><td>TPU 默认的 exp、log、sigmoid、silu 差几十格（§5.7）</td><td>换高精度写法；先量 GPU 实际怎么算</td></tr>
  <tr><td>FP32 里的病态标量</td><td>小数是两个大数相减（§5.6 折叠）</td><td>病态只放大；被放大的那一格是谁的，再查</td></tr>
  </tbody></table>
  <p>另有两类不在这张表里：长求和的累加方式（张量核的抹零，§5.2），也属精度；语义差 —— 同一个量两边定义不同（§2.3 折叠的四处）、测量工具自己错了（§6.2），排查时另列。</p></details>
  <details class="foldfig"><summary><b>十分钟看代码</b>：grep 哪些词，两行对照怎么写</summary>
  <p>两边代码里搜这些词，每一处都是一个可能的边界：<code>astype</code>、<code>.to(</code>、<code>.float()</code>、<code>.bfloat16()</code>、<code>preferred_element_type</code>、<code>precision=</code>、<code>allow_tf32</code>、<code>NVIDIA_TF32_OVERRIDE</code>、<code>autocast</code>、<code>reduce_precision</code>、<code>dtype=</code>。然后沿前向、反向，每个算子两边各写一行（输入 ／ 乘法精度 ／ 累加 ／ 输出存储 ／ 舍在哪）。</p>
  <p>验的时候：单个算子对 FP64 算 R（§4.2）；第 1 步逐张量梯度对 GPU 重跑的噪声，超出的对 FP64 定责。</p></details>
  <details class="foldfig"><summary><b>一处一处看不出来的时候</b>：第 1 步逐张量梯度普遍是 GPU 噪声的 2–3 倍，没有哪一处突出</summary>
  <p>第二次迁移第 1 步，«B2_TENSORS» 个张量里 TPU 对 GPU 的差 ÷ GPU 三份两两的最大差，中位数 «B2_MED_OFF»，超过 2 倍的 «B2_OVER_OFF» 个，从输出层到第 0 层普遍偏大。这不像一个模块写错，像好几处舍入位置一起跟 GPU 不同：打开三个照 GPU 写法舍入的开关（FP32 累加部分和、RMSNorm 只舍一次、路由权重先舍 BF16），中位数 «B2_RATIO_MED_T»，超 2 倍的 «B2_OVER2_T»。第四个开关（门控激活逐算子舍）让统计略变差（«B2_MED_4SW»、«B2_OVER_4SW» 个），照样打开 —— 开不开看 GPU 实际怎么写，不看统计好不好看。</p>
  <p>上一代配置（关 base2、多路残差一带还没照 GPU 写法舍）第 1 步，GPU 参照补到六份（15 对）后：«B2_ALL_TENSORS» 个梯度里 169 个超出 GPU 的波动，拿去对 CPU 上的 FP64 整网重算定责：167 个离 FP64 不超过 GPU 的 1.2 倍，2 个比 GPU 远（1 个是多路残差混合的小参数，超出波动 1.05 倍、判待定；1 个是 MTP 的归一化权重，已接受）；另 10 个 GPU 各份逐位相同、26 个在波动内。只有三份时是 177 个超出、5 个小参数没定责：份数补够，尺子量准了，大半「没定责」自己就消失了。对齐配置 2026-11-11 换成新口径以后重判：«B2_C_OVER» 个超出，没定责的只剩第 3 层 FFN 前一个多路残差小参数（«B2_C_UNATTR_T»），原来那个进了噪声。定责一律对 FP64，不对 TPU 自己跑的 FP32 参照（裁判不能是选手，§4.2）。</p></details>
''' + quiz("TPU 上把一处矩阵乘的两个操作数都转成 FP32，别的没改。为什么离 GPU 的真 FP32 还差末位？",
           "默认精度下它只算一遍 BF16；设成 HIGHEST 是多遍 BF16 拼出来的近似。但写着 HIGHEST 不算数，要看编译后的样子：编译器若把它改写成逐元素 FP32 乘加，差的就只剩加法顺序，归到「顺序」那一格。") + '''

  <details class="foldfig"><summary><b>病态：小数是两个大数相减</b>：一个小量若是两个大数相减得来的，大数上差一格，小量上就差一大截。查到一个小标量对不上，先问它是不是这样算出来的、被放大的那一格是谁的。 例子：第二次迁移里一项 KL 辅助损失，船上称船长；四问认出病态，再找被放大的那一格</summary>
__FIG_CAPTAIN__
  <p>第二次迁移里有一项辅助蒸馏损失（KL），每行只有约 «KD_KL»，却是从约 «KD_LSE» 的 log-sum-exp 里减出来的。300 步那一代，TPU 记下的它一直偏低（带符号均值 «KD_SM300_T»，第 1 步差到 «KD_STEP1_T»），按噪声尺子判「不过」。图里那杆秤是 f32 格式本身，不是前面的噪声尺子：lse 差一格，就是 KL 的约 4e-3。</p>
  <p>前三问认出这是病态在放大；第四问问被放大的那一格是谁的：同样的 f32 计算换到 CPU（PyTorch）上几乎不偏，TPU 偏了将近一格，来自 TPU 默认的 exp 和 log，换成高精度写法，300 步那一项的系统偏差没了。</p>
  <details class="foldfig"><summary><b>后来还有两步</b>：记录写法本身病态；GPU 自己也偏</summary>
  <p>这项 KL 的记录写法本身在 FP32 里病态（CPU 上调同一个函数也偏高），改记一种等价的广义 KL 以后，1000 步带符号均值的 r «KD_GKL_R_T»，三个统计里最差的 «KD_GKL_WORST»，由待定变过；梯度一位不变。真实宽度上 GPU 自己记下的这一项也偏高，约«GPU_KD_ROPE_SHARE_T»来自 GPU 把索引器的旋转位置编码放在 BF16 里算。所以「更准就是更像 GPU」只在 lse 那一格成立（同一代还开了另一个开关，没拆开）。</p></details>
  <details class="foldfig"><summary><b>FP64 怎么把它拆出来</b>：逐项重算同一行，差只剩最后那一步</summary>
  <p>把一行的原始输入存下来，在 FP64 里逐项重算：目标分布、对数概率，误差都在 1e-7 量级，比 1e-3 小四个数量级，差只能落在最后的 log-sum-exp 上。偏低的大小有两个口径：单行对 FP64 重算 «KD_DIFF_T»；TPU 全部运行对 GPU 全部运行 «KD_VS_GPU_T»。换成高精度的 exp、log 以后，同一行对 FP64 «KD_FIX_ROW_T»，lse «KD_FIX_LSE_T»（前后两端的其余设置不同）。由它出发的梯度对 FP64 的 R 中位数 «KD_GRAD_R64_T»（实测）。要是同样的 f32 计算换到 CPU（PyTorch）上也偏同样多，那一格就是两边都躲不开的末位，才写「不过，已定责」。</p></details>
  </details>

  <h3>5.7　超越函数：TPU 默认的 exp、log 差几十格（v5p）</h3>
  <p><b>主结论：v5p 上 TPU 默认的 exp、log、sigmoid、silu 最大差几十到上百格（1 格 ＝ 末位差 1；v7x 默认准一些，exp 仍差 64 格），GPU 只差个位数，而且默认的 exp 系统偏低。</b>v5p 上打开框架里的「精确超越函数」开关，全部降到个位数，步时 ＋«M3_TRANS_COST_T»；v7x（运行库 0.0.46）上开了也降不下来，exp 仍差 64 格。只设 JAX 的精度档 highest 不够，sigmoid、silu 降不下来。</p>
  <details class="foldfig"><summary><b>超越函数的格数与算法</b>：两代芯片的格数（图）；软件怎么把 exp 算准；第二次迁移里 sigmoid 改准了整网反而更偏；第三次迁移的编译产物</summary>
__FIG_ULP2__
__FIG_TRANS__
  <p>exp、log、sigmoid 这类函数，硬件不会一步算出来，都是软件拼的。TPU 默认那一版图快，最大误差几十到上百格；GPU 的 FP32 版本只差个位数。更要紧的是方向：在真实的输入上，默认的 exp 绝大多数结果没舍对，而且平均往低处偏（数在图下）—— §5.6 病态折叠里 lse 那一格就是这么来的。</p>
  <p>sigmoid、silu 没这么顺，图下出处里有经过；教训是：改完一个函数，整网要再验一遍。</p>
  <p><b>第三次迁移把这一项在两代芯片上量全了。</b>先分清三样东西：JAX 的精度档 highest；框架里的「精确超越函数」开关；上面图里自己搭的高精度写法。v5p 上默认几十到上百格；只设 JAX 的 highest，exp、log 降下来，sigmoid、silu 不动；打开框架的精确超越函数开关才全部降到个位数（下面折叠里第二次迁移那个「读回是开、实际没变」的开关，说明开关要实测；这一次是逐元素量过的），步时 ＋«M3_TRANS_COST_T»。编译产物显示，开了以后整段从硬件的超越函数单元挪到向量单元上用多项式算，每个向量寄存器的指令从 3 条变成 43–45 条。v7x 配运行库 0.0.46 时，这个开关打开后 exp、log 逐位一样，sigmoid、silu 只差末位（silu 29 → 30 格），都降不下来：v7x 默认就比 v5p 准，可 exp 最大仍差 64 格，目前对不到 GPU 的水平。</p>
  </details>
  <details class="foldfig"><summary><b>第二次迁移 sigmoid 改准的经过</b>：高精度开关读回是开、实际没变；改解析式前后的读数</summary>
  <p>高精度开关读回是开，实际逐算子回放 «SIG_FLAG_SAME_OPS» 个算子逐位没变（又一个「开关说的不算」）；改用高精度的 exp 自己搭，单算子降到 3、4 格，跟 GPU 实测的 3.4 格一样准，整网 loss 反而偏得更多（主 loss 带符号均值那一项的 r «SIG_LOSS_R_T»；整体判定 0.88）。精确写法的反向由自动微分推出来，跟 GPU 的解析式不同；改成同一个解析式以后，1000 步路由偏置均值 r «SIG_VJP_R_T»。主 loss 那次偏大，下一代配置里就回到了噪声内，没有单独归因。</p></details>
  <details class="foldfig"><summary><b>优化器里的除法和开方也一样</b>：几格的差，在小张量上就是几倍</summary>
  <p>TPU 默认的 FP32 除法、开方、Adam 的偏差修正 1 − βᵗ，分别差 2、3、108 格（换成精确写法后 1、1、2 格）：除法用牛顿修正的倒数再修一次商，开方做一步牛顿修正，1 − βᵗ 写成 −expm1(t·log β)，免得两个接近 1 的数相减。几格的差平时看不见，可在 Adam 管的一个小张量上，第 5 步的更新离 FP64：GPU 1.49e-6、TPU 5.70e-6，R «ADAM_R_T»。</p></details>
  <p><b>暗线</b>：这一节里，设备数（§5.4）、切法（§5.5）、步数（§5.5、§5.6）、芯片（§5.7）各换过一次，每次都推翻了之前的验证。</p>
''' + quiz("TPU 上的 sigmoid 比 GPU 粗几十格，换成跟 GPU 一样准的写法，两边就该对上了吧？",
           "不一定。单算子对上，整网未必：第二次迁移把 sigmoid、silu 换成三四格的写法后，单算子跟 GPU 一样准，整网 loss 先偏得更多。说明还有别处没对上：查到一处是反向公式跟 GPU 不同，改成同一个解析式才收拢。要逐处查、整网再验。") + '''
  <p class="bridge">这一节留下的问题：在 TPU 上，怎么把中间的张量拿出来，又不改动结果？</p>
</div></section>

''' + sec("s六", "六", "观测：在 TPU 上怎么看") + '''
  <p class="lead">要定位，就得把中间的张量拿出来看。GPU 上随手挂个钩子就行；TPU 上不行：钩子会把编译器融合好的一块切开，舍入位置跟着变，量到的已经不是生产程序。</p>
__FIG_HOOKS__

  <h3>6.1　步与步之间读状态，从动量里倒推梯度</h3>
  <p>主权重、Adam 的两个动量、路由偏置，训练步和训练步之间从状态里读，不碰训练程序。梯度不用另抓：Adam 每一步都把梯度揉进一阶动量里，倒过来解一下就有了。</p>
  <div class="note"><span class="t">gₖ ＝ (mₖ − β₁·mₖ₋₁) ÷ (1 − β₁) ÷ 裁剪系数</span>
    mₖ、mₖ₋₁ 是相邻两步读出来的一阶动量，β₁ ＝ «BETA1»。</div>
  <details class="foldfig"><summary><b>在 TPU 上照做的三个坑</b>：多主机、状态被复用、动量精度</summary>
  <ul>
    <li>8 颗 v5p 芯片分在两台主机上，切开的状态要用多主机收集（如 JAX 的 <code>process_allgather</code>，把各主机上的分片收齐）或存检查点来读；最省事的是在要看的那段步数里每步存一次检查点。</li>
    <li>训练步会复用（donate）上一步的状态缓冲区，喂进下一步以后就读不到了，要先读再喂。</li>
    <li>倒推梯度要相邻两步成对存；一阶动量要是被设成 BF16，公式就不准了。裁剪系数从记录裁剪前、后梯度范数的指标文件里取，别从控制台日志抄。整张量上相对差在 1e-7 量级，逐元素在梯度远小于动量的地方不准：两个几乎相等的数相减，有效位被抵消了。</li>
  </ul></details>

  <h3>6.2　隔离回放：拿 GPU 的输入，单独跑一层</h3>
  <p>第 k 层的输入直接用 GPU 那一层的输入，TPU 上只编译这一层。层与层的误差不串起来，第一个超线的那一层就是出错的那层；单个算子、输出头、整网前向都能这么回放。回放要用跟训练相同的设备网格、参数切分和编译参数：单卡回放看不到 FSDP 才有的 collective matmul，那几次多舍就量不出来，会判反。</p>
  <p>回放工具自己也要验：注入要覆盖全部输入，换不了就报错。</p>
  <details class="foldfig"><summary><b>回放工具自己也要验</b>：第二次迁移一次注入漏了输入</summary>
  <p>回放工具自己也要验。第二次迁移的单算子回放里，稠密 MLP 的 R 是 «INJ_R_BEFORE»：注入只换了按位置传的参数，那个分支按关键字传输入，算子吃的其实是 TPU 自己上游的输入。同一个 GPU 输入直接喂这个算子，R «INJ_ALONE_R»；修好注入以后 «INJ_R_AFTER_T»。所以注入要覆盖全部输入，换不了就报错，别悄悄放过。</p>
  </details>

  <h3>6.3　固定探针、状态互换</h3>
  <ul>
    <li><b>固定探针</b>：每隔 25 步，在一条固定的序列上单独跑一次前向，记每一层输出的均值、lse（log-sum-exp，所有 logits 的整体水平）。</li>
    <li><b>读数算不算数</b>：只有不改生产程序的读数（步间读状态、逐步序列）能当证据（工具表在下面折叠里）；精度开关在生产程序里生没生效，日常回归直接数生产训练步编译后的程序（舍到低精度的算子、all-reduce、自定义 kernel 调用各几个），边传边乘（编译产物里叫窗口化矩阵乘）只靠整份程序的指纹兜底。</li>
    <li><b>状态互换</b>：把一侧某一步的整套 FP32 状态搬到另一侧，用同一批数据各算一步（连 Adam 的步数计数和学习率调度的步号一起搬，不然偏置校正对不上）。它能把「这一步算得不一样」和「状态已经不一样」分开（§7.5）。</li>
  </ul>
  <details class="foldfig"><summary><b>工具表</b>：哪些读数能当生产程序的证据</summary>
  <table class="t9"><thead><tr><th>工具</th><th>改不改生产程序</th><th>读数能当什么</th><th>能比到第几步</th></tr></thead><tbody>
  <tr><td>步间读状态、倒推梯度（§6.1）</td><td>不改</td><td>生产程序的证据</td><td>逐张量只比第 0、1 步</td></tr>
  <tr><td>1000 步逐步序列</td><td>不改</td><td>生产程序的证据</td><td>多步只比标量曲线</td></tr>
  <tr><td>隔离回放（§6.2）、固定探针</td><td>另编一个程序</td><td>只用来定责、确认修法方向</td><td>单层、单步</td></tr>
  <tr><td>状态互换</td><td>另编一步</td><td>分开「这一步算得不一样」和「状态早就不一样」</td><td>任意一步各算一步</td></tr>
  <tr><td>挂钩子、<code>jax.debug.callback</code></td><td>改了（切开融合）</td><td>不能当证据</td><td>—</td></tr>
  </tbody></table>
  </details>
  <details class="foldfig"><summary><b>为什么分叉以后只比标量曲线</b>（图）：一条 FP32 轨迹第 2 步起离两边多远</summary>
__FIG_DIVERGE__
  </details>

  <h3>6.4　第 1 步逐位相同，不等于开关没用</h3>
  <p class="lead">从 BF16 检查点续训或微调时，有一类开关第 1 步开和不开逐位相同，要过几步才看得出差。</p>
__FIG_ONGRID__
  
  <p>所以第 1 步「开关前后逐位相同」不能当「这个开关没用」的证据：要拿训练若干步以后的权重去测（状态互换，从那一步各算一步），或者看多步曲线。随机 FP32 初始化起训时没有这回事。</p>
''' + quiz("为什么在 TPU 上「加一个输出」会改变别的张量的值，哪怕那个输出什么都不改？",
           "TPU 上的程序都由 XLA 整图编译（GPU 上用 JAX 或 torch.compile 也一样）。多一个输出，编译器就得把那个中间值落到内存里，原来融在一起的算子被切开，融合内部舍入的位置（excess precision）就跟着变了，下游的数也就变了。") + '''
  <p class="bridge">这一节留下的问题：TPU 的 loss 三百步以后为什么平均比 GPU 低约 1.7e-4？</p>
</div></section>

''' + sec("s七", "七", "案例一：loss 偏低，查到是谁的差") + '''
  <p class="lead">混元 3 搬上 TPU，主 loss 一直比 GPU 低：第 301–1000 步逐步相对差的带符号均值 «TPU_LOSS_TAIL_T»（对一份 GPU 基线；另两份 GPU 重跑对同一基线是 «GPU_B_TAIL_T» 和 «GPU_C_TAIL_T»；开场说的「约 1.7e-4」就是它）。按参考档里 loss 那条 2e-4 的线（§4.4，只看够不够近），这已经能交差；按严格档（每一处差都要说得出是谁的），得查清它是谁的差。这一节从现象一路查到一次沿 12 万行词表的求和：按调查的顺序讲，只走查对的那条线；排除过的岔路都收在折叠里，课上不讲。</p>

  <h3>7.1　偏移是慢慢长出来的，参数范数也在长</h3>
  <p>§1.2 那把尺子已经量过：TPU 三种跑法（原样、不重算、改片上内存上限）都在尺子外面，而且方向一致。先别读成「TPU 训得更好」：loss 系统性地低，只说明两条轨迹有系统差别。</p>
  <p>逐步看，前 100 步看不出，100–300 步拉开，300 步以后稳定在 «LOSS_TAIL_ROUGH_T» 左右；第 1 步的更新量两边没有差别。前向算子的差第 1 步就该看得到；第 1 步没有、几百步才长出来，说明是每一步一点点攒起来的。那就去看别的每步都记的量，谁也在慢慢长：</p>
__FIG_DRIFT__
  <p>参数范数是全部 FP32 主权重平方和开根号，这里约 «PN_LEVEL_T»。第 1000 步 TPU 比 GPU 多 «PN_TPU_EXTRA»（绝对差）；GPU 自己一千步才涨了 «PN_GPU_GROWTH»。</p>
  <details class="foldfig"><summary><b>先排除两个假警报</b>：步号错位、常数口径</summary>
  <ul>
    <li><b>步号错位</b>：被测第 k 步错对 golden 第 k−1 步，也会有稳定的差。故意错开一步再比，MARE（全程）从 «MISALIGN_OK_T» 涨到 «MISALIGN_OFF_T»，大 «MISALIGN_X_FLOOR» 多倍 —— 步号是对的。</li>
    <li><b>常数口径</b>（比如 loss 的分母不同）：那种差第 1 步就该在；这里第 1 步只差 «D1_T»，偏移是后来长出来的。</li>
  </ul></details>

  <h3>7.2　漂在输出层冷门词的行，那些行在 Adam 的 ε 区</h3>
__FIG_ROWS__
  <p>这里只说两边不一样，不说谁对；哪一边偏，§7.5 才知道。为什么偏偏是冷门词，先看 Adam 怎么算一步：</p>
''' + ADAM_CARD + '''
__FIG_EPS__
  <p>没出现过的词，每一步收到的梯度只有一种：softmax 把它的概率往下压一点，大小跟它的概率成正比，所以很小。</p>
  <p>ε 是 Adam 分母上防除零的小常数。打个比方，Adam 像收音机的自动音量，把每个参数的步子拉成差不多大。可声音小到比 ε 还小时，它就不再自动调音量，进来多大、出去就按比例多大 —— 梯度大小差一点，更新就照比例差一点。</p>
  <details class="foldfig"><summary><b>ε 是干什么的、两种区长什么样、对齐怎么处理</b></summary>
  <ul>
    <li><b>公式</b>：更新量 u ＝ m̂ ÷ (√v̂ ＋ ε)，m̂、v̂ 是梯度和梯度平方的滑动平均（带偏差修正）。ε（这里 «EPS»）本来只为防止除以零。</li>
    <li><b>常见区（√v̂ ≫ ε）</b>：u ≈ m̂ ÷ √v̂，梯度整体放大缩小都约掉了，每个元素的步子都在 1 附近；梯度上一点相对误差，分子分母一起变，大部分抵消。</li>
    <li><b>ε 区（√v̂ ≪ ε）</b>：u ≈ m̂ ÷ ε，更新跟梯度成正比，梯度错多少、更新就原样错多少，而且每一步都错在同一边。</li>
    <li><b>本例长什么样</b>：第 300 步，没出现过的词那些行，«RARE_EPS_FRAC_T» 的元素 √v̂ ＜ ε，√v̂ 的中位数 «RARE_VHAT_MED»，约为 ε 的四分之一；两边这些行梯度大小几乎一样（均方根之比 «RARE_RMS_RATIO»），差在方向。</li>
    <li><b>对齐怎么处理</b>：两边用同一个 ε（配置对照表里核对），不靠调 ε 把差藏起来；要做的是让这些小梯度本身算准 —— 本例就是去修推歪它们的那次长求和（§7.4）。</li>
    <li><b>外面的人怎么说</b>：梯度的均方根小到跟 ε 同一量级时，Adam 的步子就会跟着变小（Wortsman 等，<em>Small-scale proxies for large-scale Transformer training instabilities</em>，arXiv:2309.14322）；PaLM 的报告也提过，冷门词的嵌入二阶矩估得不准（arXiv:2204.02311）。</li>
  </ul></details>
  <details class="foldfig"><summary><b>排除清单</b>：按成本从低到高验过、不是原因的一串（数据、切法、优化器、注意力、重算、规约精度、TF32、十几处 GPU 写法……）</summary>
  <p>先查配置和名单（几分钟），再跑几十步的短运行，最后才跑 1000 步。</p>
  <table class="t9"><thead><tr><th>怀疑</th><th>怎么验</th><th>结果</th></tr></thead><tbody>
  <tr><td>两边数据不同</td><td>L0 指纹</td><td>«DATA_UNITS» 个单元逐位相同</td></tr>
  <tr><td>权重衰减作用的参数集合</td><td>用 TPU 第 1 步读出的状态，按 GPU 的公式重算更新量</td><td>逐参数吻合；把掩码反过来，残差大 «WD_MASK_X» 倍以上</td></tr>
  <tr><td>Adam 公式、状态精度</td><td>同上，另查编译结果里状态的类型</td><td>ε 位置、偏差修正都吻合；两个动量都是 FP32</td></tr>
  <tr><td>第 1 步更新量</td><td>逐张量比两边第 1 步的更新</td><td>比值 «UPD1_RATIO_LO_T»–«UPD1_RATIO_HI_T»：第 1 步没有差别</td></tr>
  <tr><td>注意力实现</td><td>GPU 换两种实现；TPU 换掉融合注意力</td><td>GPU 都在噪声内，TPU 照样漂</td></tr>
  <tr><td>重算</td><td>TPU 不重算；GPU 换成整层重算的 golden</td><td>偏移照样出现</td></tr>
  <tr><td>梯度按 BF16 规约</td><td>GPU 反过来也按 BF16 规约</td><td>噪声内</td></tr>
  <tr><td>只是执行方式不同</td><td>TPU 只改编译器的片上内存上限</td><td>参数范数漂得一模一样（loss 偏移的大小变了）：是 TPU 系统性的</td></tr>
  <tr><td>GPU 的十几处写法</td><td>GPU 一项项换成 TPU 那边的写法（交叉熵、Adam、裁剪、最后的归一化、注意力缩放、残差精度、专家加权求和、关 TF32……），各 300–400 步</td><td>全部复现不出来</td></tr>
  </tbody></table></details>
''' + quiz("冷门词那些行的梯度大小两边几乎一样（√v̂ 只差 «RARE_VHAT_DIFF_T»），为什么 TPU 上它们被压得少？（从初值算起 GPU 缩了 66.3、TPU 缩了 49.2；扣掉两边一样的权重衰减那一份 31.8，GPU 多缩 34.5、TPU 只多缩约 17，少一半。）",
           "Adam 的更新量大小两边差不多，差在方向：更新和这一行权重之间的夹角两边不一样，从第 25 步起就分开了。这些行处在 ε 区，更新几乎跟梯度成正比，方向上一点系统差，一步步积成范数差。") + '''

  <h3>7.3　推歪它们的：一个让所有 logits 一起升降的方向</h3>
  <p>输出层一行的梯度，等于各个 token 的隐状态按这个词的概率加权求和；所以冷门行往哪偏，取决于隐状态。顺着往回查主干，最先分开的是第 0 层：两边第 0 层权重拿出来，用同一份输入各算一遍，换上 TPU 的权重输出就偏，换成另两份 GPU 重跑的权重偏得小得多 —— 是第 0 层权重长歪了。</p>
  <details class="foldfig"><summary><b>第 0 层权重换着算</b>（图）：第 0 层最先分开；把第 0 层权重换着算</summary>
__FIG_LAYER0__
  </details>
  <p>第 0 层权重的范数几乎没变，所以 §7.2 按组拆的账里看不出它；变的是方向。推它的，是输出头流回来的梯度。换到输出层这一侧，看各行权重的平均方向 w̄：</p>
  <details class="foldfig"><summary><b>隐状态侧和输出层侧的方向怎么接上</b></summary>
  <p>第 0 层权重的范数几乎没变，所以 §7.2 按组拆的账里看不出它；变的是方向。折叠里那张图看的是隐状态这一侧的平均方向；下面换到输出层这一侧，看各行权重的平均方向 w̄。两者相关，可这两环之间怎么定量接上，没有单独量过。隐状态沿 w̄ 偏一点，打到输出层每一行上，所有词的 logits 就一起偏；反向时，梯度里沿 w̄ 的那一份，就是把所有 logits 一起往下压的力（下面叫「这股力」）：</p>
  </details>
__FIG_GAUGE__
  <p>softmax 看不见所有 logits 的整体平移，概率不变；可冷门行不一样：</p>
  <p>原因在一个恒等式（折叠里推）：冷门行每步收到的那份「往下压」，沿这一行自己方向的分量，正好等于这个词的原始 logit 按概率加权的和。所有 logits 整体一低，这个分量就小，这一行就缩得慢。</p>
  <details class="foldfig"><summary><b>为什么</b>：一行权重和它梯度的内积，恒等于原始 logit 的概率加权和</summary>
  <div class="note"><span class="t">w·g ＝ Σ pᵢ·(w·hᵢ) ＝ Σ pᵢ·logitᵢ</span>
    对没出现过的词，一行权重 w 跟它的梯度 g（＝ Σ pᵢhᵢ）的内积，恒等于这个词的<em>原始</em> logit 按概率加权求和（输出层没有偏置、不跟词嵌入共享时的精确恒等式；出现过的词还多一项）。</div>
  </details>
  <details class="foldfig"><summary><b>外面的人怎么说</b>：z-loss 管的就是这个方向</summary>
  <p>PaLM 训练时专门加了一项 z-loss，把 softmax 的归一化项 log Z 往 0 拉（arXiv:2204.02311）；也有人发现输出 logits 整体漂得越来越负，训练会发散，z-loss 能管住（Wortsman 等，arXiv:2309.14322）。</p></details>

  <h3>7.4　是这股力；谁推歪了</h3>
  <p>先证明就是这股力：两边都把这个方向的梯度去掉、再让 GPU 往这个方向多推一把。</p>
  <p class="codeline">''' + mt(3, "src/maxtext/configs/base.yml", "L144-L145", "诊断开关 lm_head_dh_wbar_remove") + "　" + mt(3, "src/maxtext/layers/nnx_decoders.py", "L1825-L1827", "去掉沿 w̄ 那一份梯度") + '''</p>
__FIG_INTERVENE__
  <p>只知道「是这股力」，还不知道是谁算错了：TPU 多推了，还是 GPU 少推了？拿训练里实际用到的张量，在 FP64 里重算一遍，把输出头的误差拆到每一段：</p>
__FIG_HEAD__
  <p>GPU 在反向里沿 12 万的词表做一次很长的 BF16 矩阵乘求和，张量核把结果系统地朝零截了一点（§5.2 的抹零）；落在「所有 logits 一起升降」这个方向上，每一步都少推约 «HEAD_S1_E»。TPU 这一步对 FP64 的误差小得多（第 1 步约 «HEAD_RATIO_S1» 倍）。这还只是一次矩阵乘、一步的离线证据；一千步下来谁偏，§7.5 见分晓。</p>
  <details class="foldfig"><summary><b>这一环哪部分是实测、哪部分是推断；一度判反的那一版</b></summary>
  <p>朝零、随求和长度增大，是实测；说它就是 §5.2 那个抹零模型在起作用，是推断 —— 那个模型在这么长的求和上没有单独验证过。第 100 步那个指标连真 FP32 都偏得不小，只作旁证。
    最早那一版「FP32 dgrad」没关 TF32，偏得比 BF16 还多，于是一度得出「换成FP32 dgrad 训练也不动」的错误结论。第 1 步几乎全是截断（FP32 一次累加只偏 −1.7e-6）；训练中后期冷门词概率变小，FP32 一次累加也吞同号小项（到 −2.6e-4），和截断同量级；两样都在小项相对整个和越来越小时加重（冷门词概率越训越小），分段让每段的和保持小，所以一起治。</p></details>
  <details class="foldfig"><summary><b>别的矩阵乘也在抹零，为什么偏偏是输出头露了馅</b>：三个条件凑齐了</summary>
  <ul>
    <li><b>求和够长</b>：沿词表 «VOCAB» 项，是全网最长的一次收缩。</li>
    <li><b>小项大多同号</b>：沿 w̄ 这个方向，每个词贡献的是「它的概率 × 它那一行在 w̄ 上的分量」。训练以后冷门词概率很小，行在 w̄ 上又大多同号：一大串同号的小数，抹零在每一项上都朝同一边，抵消不掉。</li>
    <li><b>后面有放大器</b>：w̄ 方向推的是所有 logits 的整体水平，水平一变，冷门行被压多少跟着变，冷门行又处在 ε 区（§7.2），每步的小差原样积起来。</li>
  </ul>
  <p>别的矩阵乘也偏一点：合成的全正输入、求和长 «TC_K2» 时朝零 «TC_BIAS2_T»（§5.2 右图）。可真实的激活和梯度有正有负，抹零的方向跟着每一项的符号走，大半互相抵消；后面也没有这样一个放大器（推断：其余矩阵乘沿各自方向的偏差没有逐个量）。</p></details>

  <h3>7.5　同一个起点各算一步；修法：从只换 FP32 到分块</h3>
  <p>是每一步都算得不一样，还是状态早就不一样、反馈放大出来的？用状态互换（§6.3）：把 GPU 第 99 步的整套状态搬到 TPU，同一批数据各算一步。</p>
__FIG_SWAP__
  <p>同一状态下 TPU 每步几乎正好（±«SWAP_RES_T» 以内）。（这里的百分数量的是「TPU − GPU 权重差」方向上少走多少；§7.4 的 «HEAD_S1_E» 量的是「所有 logits 一起升降」方向，方向不同，数不一样）</p>
  <details class="foldfig"><summary><b>状态互换图的口径</b>：这张图的「精确值」是谁算的；两个百分比别混</summary>
  <p>这里的「精确值」和「FP32 dgrad」都是 GPU 算的，那次长求和交给 cuBLAS 一次累加到底，下面会看到它积上千步以后也偏，所以这张图只看谁少走得多、谁少走得少。别把两个百分比混了：§7.4 的 «HEAD_S1_E» 量的是那一次矩阵乘沿 w̄ 少了多少；这里的 «SWAP_GPU_ABS_T» 量的是整个梯度经过 Adam 之后，权重沿漂移方向少走了多少。</p>
  </details>
  <p>开场那一问「TPU 算错了吗？」的答案在这里：<b>偏移的大头不是 TPU 算错</b>，是 GPU 当标准答案的那份 BF16 golden 自己离精确值远。</p>

  <p><b>修法，一步一步来。</b>先纠正一个常见的误解：两边那次 «VOCAB» 项的求和，都是 BF16 输入、FP32 累加，不是 BF16 累加。差在 FP32 累加器怎么舍：GPU 张量核每 «TC_GROUP» 个乘积一组，按其中最大的指数对齐，约 «TC_KEEP» 位以下的低位朝零截掉（§5.2）；TPU 的矩阵单元推断是就近舍入（没有直接证据；旁证是同一份输入算路由打分，TPU 对 FP64 «ROUTER_TPU_FP64»、GPU «ROUTER_GPU_FP64»，§5.4）。</p>
  <p><b>刚开始，几乎全是截断。</b>训练刚开始，各个词的概率都差不多，冷门词还没被压到很小，这次求和里没有「特别小的项」。所以第 1 步的误差几乎全来自张量核的截断：沿 w̄ 的相对偏差，张量核一次加到底 «ABL_BF16_S1_T»，换成真 FP32 一次加到底只有 «HEAD_EXACT_S1_T»，差了约 «ABL_S1_RATIO» 倍。</p>
  <details class="foldfig"><summary><b>怎么确定是截断</b>：三样证据，哪些是实测、哪些是推断</summary>
  <ul>
    <li><b>硬件建模</b>（实测）：按 B200 张量核模型逐位重算路由打分，三层全部元素逐位对上；跟公开论文对 H100／B200 的比特级建模一致（Khattak、Mikaitis，arXiv:2512.07004，§5.2）。</li>
    <li><b>合成测试</b>（实测）：同一个模型在合成正数输入上，求和长 «TC_K3_T» 朝零偏 «TC_BIAS3»（短求和那几点主要是输出写回 BF16 的舍入，推断，所以不拿来作截断的证据）；不走张量核的真 FP32 几乎为零（≤ 5e-8）。</li>
    <li><b>案例里的真张量</b>（实测）：把训练里实际用到的输出头张量的词表行打乱、取前 K 行在 FP64 里重算，K 从 16,384 起行越多误差越大、都朝零缩（更短的 K 沿这个方向的读数被其余行的随机误差盖住，不单调）：K ＝ «HEAD_K1» 时 «HEAD_BIAS1»，K ＝ 65536 时 2.0e-4，整张词表 «HEAD_K3» 行时 «HEAD_BIAS3»。</li>
    <li><b>推断</b>：「案例里的偏差就是这个抹零模型在起作用」—— 那个模型在这么长的求和上没有单独逐位验证过。</li>
  </ul></details>
  <p><b>第一反应：只把这次求和换成 FP32，单步一试就好。</b>输入本来就是 BF16 的值，升成 FP32 数值一位不变，所以它修的不是 BF16 精度，而是把张量核那条截断的累加路径，换成就近舍入的 FP32 累加：第 1 步的偏差降到 «HEAD_EXACT_S1_T»；状态互换里 GPU 一步少走的从 «SWAP_GPU_ABS_T» 降到 0.10%。要注意两点：升 FP32 同时换了内核和求和顺序，严格说它证明的是「张量核那条累加路径」是主因，截断是最可能的机制；而且必须真关 TF32，否则 FP32 矩阵乘会走回张量核、按 10 位尾数算（§7.4 折叠）。</p>
__FIG_ONEPASS__
  <p><b>可跑满一千步，只走到六成。</b>下图橙线：GPU 生产只换这一处，第 1000 步参数范数比 GPU 生产多 ＋«PN_RCX5_R»（5 份均值），TPU 生产是 ＋«PN_TPU_PROD_R»，只有 «PN_RCX5_SHARE»；下格看得更清楚，橙线减去 TPU 一路往下离开。同一次矩阵乘离线一量：FP32 一次加到底，第 1 步 «HEAD_EXACT_S1_T»，到第 100 步的状态已经 «ABL_FP32_S100_T»，第 300 步 «ABL_FP32_ONESHOT_300_T»，跟张量核同一个量级。截断已经没有了，为什么又偏？</p>
  <p><b>高级一点的原因：吞小项。</b>训练以后 logits 拉开了，冷门词的概率越压越小。沿 w̄ 这个方向，每个冷门词贡献一个同号的小项，它们相对整个和越来越小。就近舍入有进有舍，平时互相抵消；可一项小到不到累加值最小一格的一半时，就被整项舍掉，而且这些项同号，舍掉的都在同一边。FP32 一次加到底躲得开截断，躲不开这个（cuBLAS 怎么累加没有源码可看，吞小项是推断）。</p>
  <p><b>再修：分块累加。</b>把 «VOCAB» 项切成每 «CHUNK_SEG» 项一块，块内留 FP32：每块的和不大，小项不会被挤出去，截断也只发生在短段里；块结果再相加，一共一百多块，这一步也不吞。两样都治：GPU 生产只把这一处改成分块（第九节最终那份 golden 的写法，升 FP32 再分块），第 1000 步 ＋«PN_PRODCHUNK_R»，TPU 生产 ＋«PN_TPU_PROD_R»，图上绿线和蓝线重合；第九节第一关 TPU 对它判，全过，参数范数那一项 r ＝ «FINAL_PN_V5P_T»（v5p）、«FINAL_PN_V7X_T»（v7x）。乘法留在 BF16 张量核、只分块的写法，一千步跟它分不开，步时只多 «CB_COST_T»。</p>
  <details class="foldfig"><summary><b>旁证：GPU 全程 FP32 的精确轨迹</b>：GPU 别处全换 FP32 也只到九成，剩下的差就在这次求和</summary>
  <p>先把 GPU 输出头这一次矩阵乘换成 FP32（真关 TF32），1000 步下来 GPU 自己朝 TPU 走过去约六成；剩下的，在 GPU 上跑一条<b>精确轨迹</b>来看：权重的用法跟生产一样（FP32 主权重，每步舍成 BF16 再用），前向反向的算术全部 FP32、真关 TF32。它自己重跑一位不差；图里 loss 那一行的第二个紫点，是给第 0 层一处权重乘 (1 ＋ «PERTURB») 的扰动版。</p>
__FIG_EXACT__
  <p>上图是不重算配置、对 4 份默认 GPU 均值；换到生产近似配置、对 GPU 生产 5 份均值，第 1000 步精确轨迹 ＋«PN_FIX1_R»、TPU 生产 ＋«PN_TPU_PROD_R»，也是九成。把精确轨迹那次求和也改成分块，GPU 精确轨迹就跟 TPU 上的精确轨迹重合：参数范数逐步相对差的带符号均值 «CHUNK_VS_TPUEXACT_T»，GPU 精确自己重跑几份两两之间最大 «EXACT_SELF_PN_T»，在尺子内。</p></details>
  <p><b>别误读成「截断是小问题」。</b>不分块的长求和里，张量核的总偏差在每个阶段都比 FP32 一次加到底大：第 100 步 «ABL_BF16_S100_T» 对 «ABL_FP32_S100_T»，第 300 步 «ABL_BF16_S300_T» 对 «ABL_FP32_ONESHOT_300_T»；按相减粗分（假设两部分可以相加，推断），截断那一份第 100 步约 «ABL_TRUNC_S100_T»、第 300 步约 «ABL_TRUNC_S300_T»，跟吞小项同一个量级。状态互换里，只换掉截断这一条路就去掉了约 «SWAP_TRUNC_SHARE» 的少走。生产 GPU 不显式分块，这次求和怎么切由 cuBLAS 自己定，生产程序里实测就是上面那个偏差：截断实实在在发生了。</p>
  <p>总结成一张表：</p>
  <table class="t9"><caption style="caption-side:bottom;text-align:left;color:var(--gray);font-size:13px">偏差：训练里实际用到的张量，第 1 ／ 100 ／ 300 步沿 w̄ 对 FP64 的相对偏差；单次：一张 B200 上这一次矩阵乘的耗时（分块是逐段调库、没做融合）；步时：一千步训练，对生产</caption>
  <thead><tr><th>写法</th><th>截断</th><th>吞小项</th><th>偏差</th><th>单次</th><th>一千步步时</th></tr></thead><tbody>
  <tr><td>张量核，一次加到底（生产）</td><td>有</td><td>有</td><td>«ABL_BF16_ONESHOT_T»</td><td>«ABL_MS_PROD_T»</td><td>基线</td></tr>
  <tr><td>真 FP32，一次加到底（关 TF32）</td><td>没有</td><td>有</td><td>«ABL_FP32_ONESHOT_T»</td><td>«ABL_MS_FP32_ONESHOT» ms</td><td>—</td></tr>
  <tr><td>张量核分块（每 «CHUNK_SEG» 项，块结果留 FP32）</td><td>只在短段里</td><td>没有</td><td>«ABL_BF16_CHUNK_T»</td><td>«ABL_MS_BF16_CHUNK» ms</td><td>＋«CB_COST_T»</td></tr>
  <tr><td>FP32 分块（第九节最终 golden 的写法）</td><td>没有</td><td>没有</td><td>«ABL_FP32_CHUNK_T»</td><td>«ABL_MS_FP32_CHUNK» ms</td><td>＋«CHUNK_COST_T»</td></tr>
  </tbody></table>
  <p><b>结论：不用升 FP32，BF16 张量核加分块就够。</b>一千步只分块（乘法留在 BF16 张量核）跟「升 FP32 再分块」分不开，步时 ＋«CB_COST_T» 对 ＋«CHUNK_COST_T»。这处只影响 GPU 标准答案的制作，TPU 生产程序不动。</p>
  <div class="note"><span class="t">整个修法排成一排（第 1000 步参数范数，对 GPU 生产 5 份均值，生产近似配置）</span>
    GPU 生产 0 ＜ 只换 FP32 ＋«PN_RCX5_R» ＜ 精确轨迹 ＋«PN_FIX1_R» ＜ 分块后的 golden ＋«PN_PRODCHUNK_R» ≈ TPU 生产 ＋«PN_TPU_PROD_R» ＜ 真精确 ＋«PN_TRUE_R»。这里漂得多，是离真值近。真精确比 TPU 生产还高一截：两边生产程序都把流进输出头反向的梯度存成 BF16，这一截两边一样，不算谁的差（第九节拆账）。</div>
  <details class="foldfig"><summary><b>给第九节的伏笔</b>：换到验收配置后 TPU 少漂多少</summary>
  <p>换到第九节验收用的配置（整层重算＋路由回放），第 1000 步 TPU 生产比 TPU 自己的精确轨迹少漂 «TPU_VS_TRUE_T»；这一截是谁的，第九节拆账。</p>
  </details>
  <details class="foldfig"><summary><b>分块修好的 GPU golden 对 TPU 生产</b>：两条一千步轨迹逐步比</summary>
  <p>GPU 生产只改分块的那一份，对 TPU 生产，参数范数逐步相对差的带符号均值：v5p ＋5.2e-8、v7x −5.5e-8，两个平台方向相反，量级都在 1e-7 以下（对齐记录案例 01 证据链）。判档按第九节的第一关：TPU 对它全过，参数范数那一项 r ＝ «FINAL_PN_V5P_T»（v5p）、«FINAL_PN_V7X_T»（v7x）。对齐记录里这一组数配的噪声上限有两种写法（2.27e-7 和 1.24e-7），口径没对齐，所以这里不按它单独判。</p></details>
  <details class="foldfig"><summary><b>分块的两个条件，和一千步的确认</b></summary>
  <ul>
    <li>两个条件：块结果必须留 FP32 —— 每块先舍 BF16 再相加就不行；块长不超过 «CHUNK_SEG» —— 到 4096 已剩 «ABL_CHUNK4096_T»，再长剩得更多，跟 §5.2「越长越偏」一致。</li>
    <li>训练上的确认：最终 golden（第九节）的设置原样、只把这一处换成 BF16 张量核分块，跑满 1000 步 «CB_RUNS» 份，拿最终 golden 的 45 对尺子对它的基线判，全过，最差 «CB_WORST» 倍，跟同一台机器原样重跑最终 golden 的对照一样（«CB_CHK_WORST»）；步时 «CB_STEP_T»。TPU 当前配置拿它当基线交叉核对：v5p 全过；v7x 梯度范数 «CB_TPU_V7X_GN» 倍待定，可同一个 TPU 运行改对最终 golden 的各份逐一当基线，也在 «CB_TPU_V7X_GN_SINGLE_T» 之间起伏，是单份基线自己的涨落。正式基线仍是最终 golden。</li>
  </ul></details>
  <details class="foldfig"><summary><b>为什么不直接用全程 FP32 训练当参照</b>：那是另一条轨迹；精确轨迹和定案前那条的配置区别</summary>
  <p>生产里矩阵乘用的是主权重舍到 BF16 的那一份，权重本身每个元素就差至多半个 BF16 格。纯 FP32 训练（权重不舍）前 100 步 loss 就比 BF16 低 «FP32_TRAIN_LOSS_GAP_T»，走的是另一条路，跟哪边都差得远。
    精确轨迹保留生产的权重语义，只把算术换成 FP32，每一步的梯度就是「同一份权重下 BF16 程序应该算出来的精确值」。</p>
  <p>另外：这里的精确轨迹跟 ① 一样不重算；定案前第一关用的那条精确轨迹是在 ② 的配置上（整层重算＋路由回放）跑的，两处的数不能直接对。</p></details>

  <h3>7.6　串起来：五环一条链</h3>
  <ol>
    <li>GPU 输出层 dgrad 那次 12 万长的累加偏向一边（张量核截断＋吞小项），每一步沿「所有 logits 一起升降」的方向少推 «HEAD_S1_E»；</li>
    <li>主干权重（大头在第 0 层）沿这个方向每步少走一点；</li>
    <li>所有 logits 整体偏开一截（lse 差），softmax 看不见；</li>
    <li>冷门词那几行在 Adam 的 ε 区，压多压少跟着 logit 水平走，原样漏过来；</li>
    <li>参数范数多涨、loss 偏低，几百步积成看得见的漂移。</li>
  </ol>
  <details class="foldfig"><summary><b>五环一条链的静态图</b>：（下面动画讲的是同一件事）</summary>
__FIG_CHAIN__
  </details>
__ANIM_CHAIN__
''' + quiz("为什么「GPU 把 dgrad 换成 FP32」和「精确轨迹」都要跑满 1000 步，不能只看第 1 步？",
           "第 1 步那股力只差约 6.5e-4（相对），单看一步、对 GPU 噪声比根本看不出；它是每一步同向地少一点，要累积几百步才显出 loss、lse 和参数范数的差。比的是轨迹，所以要跑满。") + '''
  <p class="bridge">这一节留下的问题：同一份代码只换并行方式，数学上等价，曲线却对不上 —— 这回又是谁的差？</p>
</div></section>
''' + sec("s八", "八", "案例二：DDP 和 FSDP 对不上") + '''
  <p class="lead">第二个案例在第二次迁移上，模型是混元 4 的缩小版（约 «C2_PARAMS_T» 参数，词表 «C2_V» 行、隐藏维 «C2_D»）。纯数据并行（DDP）和 FSDP 数学上等价，TPU 上一比，曲线超出了噪声。这回是谁的差？还是按调查的顺序讲。</p>

  <h3>8.1　数学等价，曲线却超出噪声；第 1 步只剩一个张量</h3>
  <p>TPU 自比（§10.1）：同一个模型、同一份权重和数据，基线是 FSDP 8 卡，只改成纯数据并行，尺子是 GPU 同设置重跑的噪声上限 N。r ＝ 差 ÷ N，超过 2 判不过（对齐记录里这一档叫「超」，1 到 2 叫「略超」）。</p>
  <table class="t9"><thead><tr><th>条件</th><th>曲线最差 <span style="text-transform:none">r</span>（300 步；第一行最差那项是梯度范数）</th></tr></thead><tbody>
  <tr><td>8 卡，每步两个微批（跟基线一样）</td><td><b>«C2_R_DP8_T»</b></td></tr>
  <tr><td>8 卡，不做梯度累积</td><td>«C2_R_GA1_T»</td></tr>
  <tr><td>16 卡（每卡一个微批，不累积）</td><td>«C2_R_DP16_T»</td></tr>
  <tr><td>v7x，8 个设备</td><td>«C2_R_V7X_T»</td></tr>
  <tr><td>v7x，8 个设备，不做梯度累积</td><td>«LIB_V7X_R46_T»</td></tr>
  </tbody></table>
  <p class="codeline">以上都在 TPU 运行库 0.0.46 上。换到 0.0.52，最后一行变了（§8.6）。「不做梯度累积」＝两个微批合成一次 scatter，一次加 8192 个下标，所以反而更差（§8.6 讲为什么）。</p>
  <p>换卡数都一样不过：前 300 步主 loss 平均偏高（带符号均值 «C2_LOSS_SM_T»，r «C2_LOSS_R_T»），梯度范数平均偏低（«C2_GN_SM_T»，r «C2_R_DP8_T»；前 200 步最明显，下图），§3.3「从 K 份运行到一个 r」那张图下块的算例就是它。换张量并行、专家并行、混合切分都跟 GPU 重跑分不开，所以问题出在纯数据并行特有的某处。v7x 的 «C2_R_V7X_T» 单看曲线不算铁证（纯 GPU 噪声用留一法也能到 2.47，§3.5），是第 1 步对 FP64 抓到的。</p>

__FIG_DDPCURVE__
  <p>曲线只能说有系统偏差，说不出偏在谁身上。照 §4.2 的定责流程，拿第 1 步的 «C2_NT» 个参数梯度各自对 CPU 上的 FP64 比：<b>只有词嵌入梯度丢了精度</b>，其余 204 个里，除照例单列的多路残差小参数，都在噪声内或只差加法顺序（只换了顺序的 185 个 R ≤ 1.2）。</p>
  <table class="t9"><thead><tr><th>第 1 步离 FP64</th><th>词嵌入梯度</th><th>其余 204 个</th></tr></thead><tbody>
  <tr><td>纯数据并行</td><td><b>«C2_EMB_DP_T»</b>（R «C2_EMB_R_T»）</td><td>比值中位数 «C2_OTHER_MED_T»；记「换了顺序」的 185 个 R ≤ 1.2，多路残差小参数另算</td></tr>
  <tr><td>基线 FSDP ／ GPU</td><td>«C2_EMB_G_T» ／ «C2_EMB_GPU_T»</td><td>—</td></tr>
  </tbody></table>
  <p>R 的分母取基线和 GPU 中位数里离 FP64 远的那个（GPU 的 4.17e-3）：这样 TPU 自己的基线本来就偏时，不会被算成「新改动丢了精度」。§4.2 定责流程图第一行的算例就是它。多路残差混合的小参数照例单列（«C2_IHC_SKIP» 个超线，数值上病态，向来不定责）。</p>

  <h3>8.2　第一个假设：BF16 跨卡求和 —— 修了能好，但不是主因</h3>
  <p>先说词嵌入的反向在干什么：一次 scatter-add，按 token 号（下标）把每个 token 的梯度加回词表那一行；一张卡一次加 4096 个 token，就是 4096 个下标。</p>
  <p>最顺手的解释：FSDP 下词嵌入表按隐藏维切，每张卡只管自己那 «C2_D_SHARD» 维，不用跨卡求和（为什么不用，§8.4 看形状）；纯数据并行下每张卡有整张表，梯度要在 8 张卡之间加起来，而这次 all-reduce 用的是 BF16 加法。修法也现成：一个让词嵌入梯度在 FP32 里累加的开关，下文叫修法①（前向从 FP32 主权重表取行再转 BF16，前向逐位不变，反向的累加落在 FP32 上）。</p>
  <p>打开 FP32 累加（修法①），先撞上一个编译器 bug，r 冲到 «C2_BUG_R_T»（§10.3）；绕开 bug 以后 300 步 r «C2_FIX1_R300_T»，回到噪声内。</p>
  <details class="foldfig"><summary><b>插曲：那个编译器 bug 怎么回事</b></summary>
  <p>打开它，r 反而冲到 <b>«C2_BUG_R_T»</b>：一块梯度凭空没了。编译器把这份 FP32 梯度的跨卡求和挪出了梯度累积循环，又在循环里把 0 号以外各卡的累加器清零 —— 这就是 §10.3「改完先比指纹」那张图右块的编译器 bug，要「数据并行 ＋ 梯度累积 ＋ 这个开关」三样同时成立才出现。修好以后 300 步 r «C2_FIX1_R300_T»。</p>
    <p class="codeline">''' + mt(4, "src/maxtext/utils/gradient_accumulation.py", "L148-L160", "绕开这个编译器 bug 的那道屏障") + '''</p>
</details>
  <p>接着有人追问：只把本卡的累加换成 FP32、跨卡仍传 BF16，行不行？为此加了修法②，结果词嵌入梯度离 FP64 «C2_FIX2_EMB_T»、300 步 r «C2_FIX2_R300_T»，也在噪声内。<b>跨卡仍用 BF16 也够了</b> —— 这已经在暗示：跨卡那一步不是主因。</p>
  <details class="foldfig"><summary><b>修法①为什么没证明主因</b>，以及修法②省没省通讯</summary>
  <ul>
    <li>修法①同时改了本卡累加和跨卡求和两处，所以它有效不等于主因在跨卡。</li>
    <li>修法②没省下 all-reduce 的字节；修法①也没让它翻倍：修法①里编译器把主输入、多 token 预测输入两路合成一次 FP32 all-reduce，每个微批 1 次（默认是两次 BF16），all-reduce 的字节一样；另在加法之前多了 «C3_FIX1_PERM_T» 字节的点对点交换（collective-permute，把一部分 token 的梯度和 id 换到别的卡），词嵌入那份每步每设备 «C3_FIX1_EMB_T» 对 9.90e8 字节（怎么数，§8.10）。只有一路输入的模型，FP32 那份会多传一倍（推断）。</li>
  </ul></details>

  <h3>8.3　审计编译产物：误差全在 39 个高频词</h3>
  <p>把两种并行编出来的训练步程序逐个查一遍：每一次梯度跨卡求和是什么类型、在不在累积循环里、跨卡前后各舍几次；再把第 1 步的词嵌入梯度<b>按行分组</b>，每组对 FP64 算一次：</p>
__FIG_DDPROWS__
  <p class="codeline">''' + code("models/hunyuan4/analysis/emb_grad_row_groups.py", "按行分组") + '''</p>
  <p>两根对照柱把第一个假设推翻了：误差跟着「本卡一次加几次」走，跨卡多几路它不在乎。<b>主因在每张卡本地那次 BF16 scatter-add</b>。高频那 39 行占整张梯度范数平方的 0.90，纯数据并行多出来的误差 «C2_HOT_SHARE_T» 在它们身上。</p>
  <details class="foldfig"><summary><b>矩阵乘类权重为什么两边一样</b>：也有 BF16 跨卡求和，可两边舍入次数基本相同</summary>
  <p>审计顺带改正了一个印象：纯数据并行下以 BF16 跨卡求和的不只词嵌入，32 个 all-reduce 里有 10 个用 BF16 加法（路由专家 6 个、稀疏注意力索引器的两个投影 2 个、词嵌入 2 个）。可路由专家和索引器在 FSDP 下同样是 BF16，舍入次数相同，离 FP64 的比值（纯数据并行 ÷ 基线）在 0.91–1.07。矩阵乘类权重的梯度来自矩阵乘，每张卡先算部分和再跨卡加：纯数据并行用 all-reduce，FSDP 多半用 reduce-scatter（路由专家两边都是 all-reduce）；all-reduce ＝ reduce-scatter ＋ all-gather，all-gather 只搬不加，所以只差先后。例外是第 0 层和多 token 预测层：纯数据并行下它们的舍入落在跨卡之前，位置跟 FSDP 不同，但离 FP64 没有变远。词嵌入不同：它的累加不是矩阵乘，是 scatter-add，FSDP 那一侧换了写法。</p></details>

  <h3>8.4　再追问：FSDP 加得更多，为什么反而不差</h3>
  <p>上一张图留了个反常：FSDP 那一侧同一个高频词加得更多，误差反而最小。光是「加的次数多」解释不了，落到矩阵形状上看两边到底加什么、传什么：</p>
__FIG_DDPFLOW__
  <p>那次 scatter 本身，两边 HLO 里要紧的差别是 FSDP 那份先按 token 号排了序，scatter 带着「下标已排序」的标记（<code>indices_are_sorted=true</code>）。当时只能推断「排好序的 scatter 把连续的更新留在 FP32 里加」；另一个追问是：纯数据并行那份为什么不排序？</p>

  <h3>8.5　单算子实验：排序加标记，才只舍一次</h3>
  <p>把这一个 scatter-add 单拿出来，在 v5p 上真跑：表是真实形状，下标是基线第 1 步的真实 token，更新量随机；结果和 CPU 上两种仿真（「FP32 里累加、最后舍一次」和「每加一次舍一次」）逐位比。</p>
__FIG_RMW__
__ANIM_SCATTER__
  <p class="codeline">''' + code("models/hunyuan4/analysis/emb_scatter_sort_probe.py", "单算子实验（四格、扫下标数、交叉编译看门槛）") + '''</p>
  <p>四格里只有「排序 ＋ 标记 ＋ excess precision 开」那一格在 FP32 里累加。只排好序不带标记，结果仍逐位等于每次舍：标记告诉后端「同一行的更新已经挨在一起」，后端见到它才换成片上累加的写法：把表一块一块搬进片上的 FP32 缓冲，这一块的更新加完再写回，每行只舍一次（后端指令和逐位结果都核过）。</p>
  <details class="foldfig"><summary><b>8 个设备上照两种切法各跑一份</b>：排序一项走了多少</summary>
  <p>每卡 4096 个下标、全局 32768 个（更新量是随机数，跟训练里的数不直接比，只看相对大小）：FSDP «C2_FSDP_SH_T»；纯数据并行 «C2_DP_SH_T»；纯数据并行只加上排序 «C2_DPSORT_SH_T»。从纯数据并行到 FSDP 的差距，排序一项走了«C2_SORT_SHARE_T»（本课按这三个数复算）；剩下的在本卡部分和先舍一次 BF16、再做 BF16 跨卡求和 —— 跟修法②是同一种舍入结构。</p></details>

  <h3>8.6　谁来排序：下标数过门槛；门槛跟着运行库走</h3>
  <p>把编译器每道工序之后的程序都存下来看，排序是 TPU 后端（libtpu）里的一道工序自己插的，公开的 XLA 代码里找不到它。把一次 scatter 的下标数往上扫：</p>

__FIG_SWEEP__
  <p>落回训练：每张卡那次 scatter 一次加几个下标，决定了排不排序。§8.1 那张表就是这么来的：</p>
  <table class="t9"><thead><tr><th>跑法</th><th>每卡一次的下标数</th><th>运行库 0.0.46</th><th>运行库 0.0.52</th><th>曲线最差 <span style="text-transform:none">r</span>（0.0.46 ／ 0.0.52）</th></tr></thead><tbody>
  <tr><td>FSDP 8（基线，收齐 8 卡的 token）</td><td>«C2_FSDP_N»</td><td>排序</td><td>排序</td><td>—</td></tr>
  <tr><td>纯数据并行 8，累积 2</td><td>«C2_TOK»</td><td>不排序</td><td>不排序</td><td>«C2_R_DP8_T» ／ «LIB_DP8_52_T»</td></tr>
  <tr><td>纯数据并行 8，不累积</td><td>8192</td><td>不排序</td><td>不排序</td><td>«C2_R_GA1_T» ／ 没跑</td></tr>
  <tr><td>纯数据并行 16</td><td>«C2_TOK»</td><td>不排序</td><td>不排序</td><td>«C2_R_DP16_T» ／ 没跑</td></tr>
  <tr><td>v7x 纯数据并行 8，不累积</td><td>8192</td><td>不排序</td><td><b>排序</b></td><td>«LIB_V7X_R46_T» ／ <b>«LIB_V7X_R52_T»</b></td></tr>
  </tbody></table>
  <p>门槛的具体数跟着运行库版本变，会过期；要带走的只有一条：<b>看编译产物里这次 scatter 排没排序</b>。</p>
  <details class="foldfig"><summary><b>门槛的具体数</b>：不累积为什么最差、0.0.52 上门槛怎么变、生产宽度下的门槛（数会随运行库版本过期）</summary>
  <p>不累积时同一个词在一张卡上加的次数翻倍，所以最差；16 卡每卡仍 4096 个，跟 8 卡差不多。v7x 在 0.0.46 上跟 v5p 一样走不排序（第 1 步词嵌入梯度 «C2_V7X_EMB_T»；v7x 上交给 SparseCore 的是 FSDP 那一侧的 scatter）。</p>
  <p><b>门槛跟着运行库走。</b>1/8 是在运行库 0.0.46 上二分出来的。升到 0.0.52 再二分（单算子离线编译），门槛跟列宽、芯片代际都有关（§10.2 图下出处里有全表）：同样 120832 行，v5p 上列数 1024 的门槛 «LIB_TH_V5P_1024»，列数 128 的降到 «LIB_TH_V5P_128»；v7x 上列数 1024 的降到 «LIB_TH_V7X_1024»。</p>
  <p>表的最后一行就是拿它做的预测：离线编译先看到 0.0.52 排序，再上卡核对，词嵌入梯度 R «LIB_V7X_R_PAIR_T»，曲线回到噪声内。但第 1 步另有三个多路残差输出头的小参数离 FP64 远（R 最大 25.7），原先靠多路残差小参数的单列判在噪声内；2026-11-12 单列加了倍数上限（R 5.22），这次三个 R 25.7、17.3、8.8 都超上限，整次改判「超」。偏差另有来源：输出头反向那颗融合 kernel 里的舍入，给它补上强制舍入后 hc_head_base 离 FP64 从 0.561 降到 0.059。词嵌入这一处的结论不受影响。机理还是那一条，只是门槛挪了。</p>
  <p><b>别急着把它当好消息。</b>同样的规则搬到生产宽度（词表 120832、隐藏维 6144），0.0.52 上门槛跟着列数涨：v7x 涨到 «PROD_TH_V7X_052_T»，v5p 到 65536 都不插（图右上最后一行）。0.0.52 上，纯数据并行每卡 4096、8192、16384 个下标两代芯片都不排序（另测了 32768 一档）。每卡下标数＝序列长 × 每卡微批：序列 4096、每卡微批 1–3 时，0.0.46 上也没过门槛，两个版本都要修。微批 4（16384 个）那档反过来了：0.0.46 的门槛是 «PROD_TH_046_T»，会排序；升到 0.0.52 不排了，按 §8.5 的机理，会从不丢变成丢（排不排序是离线编译看到的；丢不丢是推断，还没上卡跑）。小模型上的结论能不能搬，按自己的形状再判一次。</p>
  </details>

  <h3>8.7　结论：高频词 × 不排序；修法和代价</h3>
__FIG_DDP2X2__
  <p><b>两个条件缺一个都不丢精度</b>（上面这张 2×2）：一行加的次数多，而且后端没排序、每加一次舍一次。第一版的两种修法，是让这次累加落在 FP32：</p>
  <table class="t9"><thead><tr><th>修法</th><th>第 1 步词嵌入梯度离 FP64</th><th>曲线最差 <span style="text-transform:none">r</span>（300 步 ／ 1000 步）</th><th>步时（对基线）</th></tr></thead><tbody>
  <tr><td>不修</td><td>«C2_EMB_DP_T»</td><td>«C2_R_DP8_T»</td><td>«C2_ST_DP_T»</td></tr>
  <tr><td>① 整次 scatter 和跨卡求和都在 FP32（修好 bug 以后的代码）</td><td>«C2_FIX1_EMB_T»</td><td>«C2_FIX1_R300_T» ／ «C2_FIX1_R1K_T»</td><td>«C2_ST_F1_T»</td></tr>
  <tr><td>② 本卡 FP32 加完、舍一次，再 BF16 跨卡求和</td><td>«C2_FIX2_EMB_T»</td><td>«C2_FIX2_R300_T» ／ «C2_FIX2_R1K_T»</td><td>«C2_ST_F2_T»</td></tr>
  </tbody></table>
  <p>负数是比基线快：不修时纯数据并行本来就快，修法本身多花 «C2_ST_COST_T»。1000 步的 1.09、1.07 判待定（略超），可没超过纯噪声自己能拿到的分（§3.5 的留一法），跟 GPU 重跑分不开。后来又有第三种修法：不改精度、只改切分（§8.9）。三种的精度同档（都跟 GPU 重跑分不开），第三种通信、显存、步时最省，所以<b>纯数据并行时首选只切词嵌入表；过不了排序门槛，再退回修法①</b>。</p>
  <details class="foldfig"><summary><b>回头看混元 3</b>：为什么 v5p 上不受这件事影响</summary>
  <p>回头看混元 3：按专家切 8 那份离线编译核过，词嵌入表沿隐藏维切开，每卡一次收齐 32768 个下标、排了序；FSDP 8 看的是一次观测运行存下的程序，每卡 65536 个、也排了序（运行库 0.0.46）。所以在 v5p 上不受这件事影响；v7x 上那次 scatter 交给 SparseCore，累加精度没测。</p>
  </details>
  <details class="foldfig"><summary><b>换到自己的模型怎么自查</b>：比较每卡一次 scatter 的下标数 N 和词表行数 V</summary>
  <table class="t9"><thead><tr><th>情况</th><th>会怎样</th><th>做法</th></tr></thead><tbody>
  <tr><td>N 不过门槛（运行库 0.0.46 上是 V／8）</td><td>不排序，高频词丢精度</td><td>首选只切词嵌入表（切开后 N 变成切分组里所有卡的 token 数；它跟 FSDP 一样要 excess precision 开着，且不能跟按数据并行切优化器状态的 ZeRO-1 同开）；合起来仍不过线，用修法①</td></tr>
  <tr><td>N 过门槛</td><td>后端自动排序，只舍一次（excess precision 开时；v5p 实测）</td><td>不修也行</td></tr>
  </tbody></table>
  <p>N 在纯数据并行下是本卡一个微批的 token 数；词表按隐藏维切的 FSDP 下，是切分组里所有卡一个微批的 token 数；主输入、多 token 预测输入分开算。按这条规则，FSDP 一个微批的 token 太少，也会落到不排序那一边（推断，没跑过）。门槛随运行库版本、列宽和芯片变（§8.6），表很小时也不插 —— 所以别用 V／8 硬算：对齐记录的单算子脚本带了一个 <code>check</code> 子命令，在训练镜像里给出词表行数、每卡列数、每卡下标数，CPU 上只编不跑就告诉你插不插排序、门槛是多少；最后以整步编译产物里那次 scatter 有没有 <code>sort</code> 和 <code>indices_are_sorted=true</code> 为准（§8.10）；开了第三种修法也要按切开后的 N 再核一遍；不过线、或者拿不准运行库版本时，开修法①最稳（用修好 bug 以后的代码）。另外：两种并行编出的是两个程序，求和顺序不同，逐位相同做不到。</p></details>
  <details class="foldfig"><summary><b>FP32 为什么能修</b>：舍入一次小 2¹⁶ 倍</summary>
  <p>加的顺序照样跟 FSDP 不同，但每次加法的舍入从 2⁻⁸ 量级降到 2⁻²⁴ 量级，小了 2¹⁶ ＝ 65536 倍。最高频的词在一张卡上加 «C2_HOT_CNT_MAX» 次，按 FP32 每次 2⁻²⁴ 线性累加也只有约 2.7e-5，仍远小于一次 BF16 舍入的 3.9e-3（本课推导）。单算子里 FP32 scatter 离 FP64 只有 «C2_FP32_CELL_T»。</p></details>

  <h3>8.8　两个旁证：GPU 为什么没这个问题；关掉 excess precision 反而更差</h3>
  <p>两个旁证：GPU 的词嵌入反向下标多了就先排序（推断），所以没这个问题；关掉省舍入让两边「更像」，反而让 FSDP 那次排好序的 scatter 多舍，更差 —— 让两边一样，不等于让两边都对。</p>
  <details class="foldfig"><summary><b>两个旁证的细节</b>：PyTorch 的两条路；关掉省舍入为什么走不通</summary>
  <p><b>GPU（推断）</b>：GPU 同设置的词嵌入梯度离 FP64 «C2_EMB_GPU_T»，跟 TPU FSDP 的 «C2_EMB_G_T» 接近。PyTorch 在 CUDA 上的词嵌入反向，下标超过 «C2_TORCH_GATE» 个时先排序、按段在 float 里累加，最后才转回 BF16；GPU 每卡每个微批 4096 个 token，推断走的是这条：切法跟纯数据并行一样，本卡累加却跟 TPU FSDP 排序后是同一种舍入结构。</p>
  <details class="foldfig"><summary><b>PyTorch 的两条路</b>：在哪段代码、哪里没核</summary>
  <p><a href="https://github.com/pytorch/pytorch/blob/8b7ee4d69c74e7a1a140f6ea218928b9b8de1779/aten/src/ATen/native/cuda/Embedding.cu#L268-L282">Embedding.cu</a> 的 <code>embedding_dense_backward_cuda</code> 按下标数分两条路：不超过 «C2_TORCH_GATE» 个时，每批 32 个下标在共享内存里用 float 合并，批与批之间舍一次；超过时走排序、分段累加。读的是 PyTorch 主分支上的一个提交（链接钉在那个提交）；GPU 镜像里那个版本没有逐行核过，实际走了哪条也没有单独确认。</p></details>
  <p><b>关掉 excess precision</b>：§5.5 末尾那个「远 «EMB_NOEXC_X» 倍」就出在这里。有人建议关掉它让 FSDP 和 DDP 更像（另一个 6 层测试模型上差确实回到 0 附近，但那次主要是把一条梯度累积路径改成了 FP32）。</p>
  <p>放到这里看，这条路走不通：纯数据并行那次 scatter 本来就每次舍，开关管不着；FSDP 那次排好序的 scatter 却从只舍一次退回每次舍，而且一次要顺序加 «C2_FSDP_N» 个更新。结果 FSDP 比纯数据并行还差得多：第 1 步词嵌入梯度离 FP64 «C2_NOEXC_EMB_T»，300 步 r «C2_NOEXC_R_T»。<b>让两边一样，不等于让两边都对</b>：修法要看离精确值近不近，不能只看两边的差。</p>
  </details>
''' + quiz("你的模型词表 25 万行，8 张卡纯数据并行、每卡每个微批 8192 个 token，没开 FP32 修法。按这一节，会不会中招？换成每卡 40000 个呢？",
           "按运行库 0.0.46 的规则，门槛是 V／8 ＝ 31250。8192 不过线，不排序，高频词每加一次舍一次，会中招。首选只切词嵌入表：8 张卡合起来 65536 个，过线，会走排序、只舍一次。40000 本来就过线，后端会自己排序（前提是 excess precision 开着）。换到 0.0.52，门槛跟列宽、芯片走，不能再用 V／8 估。所以这些都只是初判：用对齐记录脚本的 check 子命令在自己的形状、自己的运行库上判一次，再看整步编译产物里那次 scatter 有没有 sort 和「下标已排序」的标记（§8.10）；拿不准就开修法①。") + '''

  <h3>8.9　第三种修法：换切分，不换精度</h3>
  <p>前两种修法都没动纯数据并行处理这份梯度的结构：每张卡一个微批只有 4096 个 token，跨卡求和传的却是整张表，占纯数据并行全部通信的 «C3_EMB_SHARE_T»（本课按 §8.10 的字节复算）；8 张卡各存一整份表和它的优化器状态。换个思路：FSDP 那条「先传后加」不是谁手写的，是编译器在「表按隐藏维切开」这个前提下自动切出来的。那就其余参数照旧纯数据并行，只给词嵌入表加一个开关，让它沿数据并行轴、按隐藏维切开：</p>
__FIG_SHARD__
  <p>结果跟预期一样：编译器对这一个参数生成了跟 FSDP 一条不差的通信，每张卡那次加法收 «C2_FSDP_N» 个下标、排序、带标记，按单算子实验就是只舍一次，病根「高频词 × 不排序」不再成立。205 个张量没有丢精度的；曲线跟 GPU 重跑分不开；词嵌入那份通信每步每设备从 9.90e8 降到 6.82e7 字节，步时 «C3_ST_T»（对基线）。</p>
  <p class="codeline">''' + mt(4, "src/maxtext/configs/base.yml", "L150-L152", "三种修法的开关") + "　" + mt(4, "src/maxtext/layers/embeddings.py", "L136-L187", "词嵌入层：表怎么切、在什么精度里加") + "　" + mt(4, "HUNYUAN.md", "", "全部开关一览") + '''</p>
  <details class="foldfig"><summary><b>三种修法放在一起</b>：精度、通信、步时、它靠什么</summary>
  <table class="t9"><thead><tr><th>修法</th><th>第 1 步离 FP64</th><th>曲线最差 <span style="text-transform:none">r</span>（300 ／ 1000 步）</th><th>词嵌入通信（每步每设备，按形状算）</th><th>步时（对基线）</th><th>靠什么</th></tr></thead><tbody>
  <tr><td>不修</td><td>«C2_EMB_DP_T»</td><td>«C2_R_DP8_T»</td><td>9.90e8</td><td>«C2_ST_DP_T»</td><td>—</td></tr>
  <tr><td>① 全程 FP32</td><td>«C2_FIX1_EMB_T»</td><td>«C2_FIX1_R300_T» ／ «C2_FIX1_R1K_T»</td><td>«C3_FIX1_EMB_T»</td><td>«C2_ST_F1_T»</td><td>修好编译器 bug 的那道屏障</td></tr>
  <tr><td>② 本卡 FP32、跨卡 BF16</td><td>«C2_FIX2_EMB_T»</td><td>«C2_FIX2_R300_T» ／ «C2_FIX2_R1K_T»</td><td>9.90e8</td><td>«C2_ST_F2_T»</td><td>—</td></tr>
  <tr><td>③ 只切词嵌入表</td><td>«C3_EMB_T»</td><td>«C3_R300_T» ／ «C3_R1K_T»</td><td>6.82e7</td><td>«C3_ST_T»</td><td>每卡那次加法过排序门槛</td></tr>
  </tbody></table>
  <ul>
    <li>③ 的显存峰值每设备少 1.70e9 字节，其中 1.30e9 是表的主权重和两份动量只存 1/8；其余推断来自不再需要整表的梯度缓冲。</li>
    <li>通信少了 9.2e8 字节、步时却几乎不变：推断是原来那份整表 all-reduce 本来就和计算重叠了，不在关键路径上；没有看 profile 核实。</li>
    <li>重算不会把这些通信再做一遍：词嵌入在层扫描之外；重算策略换成整层重算再编一遍，通信条数和字节都不变。</li>
    <li>三种的精度同档（1000 步它的 1.15 还略高一点，都在纯噪声分数以内）；它赢在通信、显存、步时，所以纯数据并行时首选它。它不靠 FP32 开关、也不靠绕编译器 bug 的屏障，换的是切分方式本身；但跟 FSDP 一样要 excess precision 开着（默认开），也不能跟 ZeRO-1 同开；但它靠排序门槛：数据并行路数或每卡 token 数小到合起来也不过线时，会退回不排序（推断，没跑过），这时仍要用修法①。只在 v5p、缩小版、纯数据并行 8 ＋ 累积 2 上验证过（运行库 0.0.46；0.0.52 上重测，三种修法也都跟 GPU 重跑分不开：«LIB_FIX_T»）。</li>
  </ul></details>

  <h3>8.10　方法：怎么从编译产物看通信</h3>
  <p>上面那些「传什么、传几次、在哪排序」，都是上卡之前离线编译看清的：在 CPU 上为目标拓扑编一遍训练步，把编译后程序里的每一次集合通信读成形状和字节、数出每步几次。第九节「先编译、再扫描」用的是同一招。</p>
  <details class="foldfig"><summary><b>怎么读、怎么数</b>：集合通信读成「形状 × 字节宽」、数每步次数，图、照着做、代码、边界</summary>
  <p>上面那些「传什么、传几次、在哪排序」，都是上卡之前就看清的：把训练步按目标拓扑在 CPU 上编一遍（不占芯片，缩小版编译 75–215 秒，连追踪整次中位数 184 秒），从编译后的程序里把每一次集合通信找出来，读成「形状 × 字节宽 × 每步次数」：</p>
__FIG_COMM__
  <details class="foldfig"><summary><b>照着做一遍</b>：真实程序里的几种写法、怎么顺着调用链数次数</summary>
  <ul>
    <li><b>拿到程序</b>：用训练同一套配置和同一个训练步函数，把设备换成目标拓扑的描述、把参数和数据换成同形状同切分的抽象值，走 JAX 的提前编译；编译时加 <code>--xla_dump_to=&lt;目录&gt; --xla_dump_hlo_as_text</code>，在目录里找名字带训练步、结尾是 <code>after_optimizations.txt</code> 的那份。</li>
    <li><b>异步的写法</b>：集合通信常拆成成对的 <code>xxx-start</code>／<code>xxx-done</code>，也可能包成通用的 <code>async-start(…), calls=%某个子计算</code>，真正的通信在那个子计算里。成对的算一次；异步版的结果是元组（例如 collective-permute 是「发送缓冲, 接收缓冲, 两个上下文」），只算一块缓冲，all-gather 只算输出那块。</li>
    <li><b>数次数</b>：先找到这条通信所在的计算（computation）名，再搜谁用 <code>body=</code> 或 <code>calls=</code> 调它，一路往上走到入口；每碰到一个 <code>while</code>，读它 <code>condition=</code> 那个计算里 <code>compare(计数器, 常数), direction=LT</code> 的常数（计数器从 0 起、每次加 1 时就是循环次数，梯度累积、层扫描都是这样），一路乘起来。</li>
    <li><b>认参数</b>：形状先认（词嵌入 [120832, 1024]，切开后 [120832, 128]；输出头是 [1024, 120832]，第一维不同）；再看 <code>metadata={op_name="…"}</code>，反向往表里加的那一串里带 <code>scatter-add</code>，前向取行带 <code>gather</code>；还拿不准，顺着数据流看谁用了它。</li>
    <li><b>核对</b>：本课这一份，上卡后存下的程序里集合通信跟离线编的逐条一致（只差设备分组的写法；核对的是只切词嵌入表那份）。</li>
  </ul></details>
  <p class="codeline">''' + code("models/hunyuan4/analysis/hlo_comm_bytes.py", "列出每条集合通信、每步次数和字节") + '''</p>
  <p>这把尺子也有边界（字节按形状算这一条写在图下）：</p>
  <ul>
    <li><b>看不到精度</b>：§8.5 那一格就是程序一字不差、只关一个开关，后端的写法就变了；换运行库也一样（§10.2）。精度仍要上卡做第 1 步逐张量定责。</li>
    <li><b>看不到快慢</b>：通信能不能藏在计算后面，要看实测步时或 profile。③ 就是例子：通信少了一大截，步时跟不修的纯数据并行几乎一样（对基线 «C2_ST_DP_T» 和 «C3_ST_T»）。</li>
  </ul>
  </details>
  <p class="bridge">这一节留下的问题：回到混元 3 —— GPU 的 BF16 golden 偏了，连「精确轨迹」也带着一份长累加的偏差，验收拿谁当基线？</p>
</div></section>

''' + sec("s九", "九", "验收与 golden 设计") + '''
  <p class="lead">当基线的那份 golden 不是一开始就有的：先往精确方向改了三次（GPU 生产 golden → 只换 FP32 → 精确轨迹），查清以后发现只要在 GPU 生产 golden 上改一处，这就是修好的 golden（下文图上标 ⑤）。</p>

  <h3>9.1　五份 golden，一步一步来</h3>
__FIG_GOLDEN__
  <p>一句大白话：GPU 当标准答案的那份，在一次 12 万长的求和里每一步都朝零少算了一点；最终 golden ⑤ 就是只把这一处改成升 FP32、分段累加（只分段也够，步时只多 «CB_COST_T»）。编号速查：① 框架默认；② 生产近似（整层重算＋路由回放）；③ ② 加输出头 dgrad 换 FP32；④ 精确轨迹（全程 FP32）；<b>⑤ ② 加输出头 dgrad 分段 FP32 累加，最终 golden</b>；⑥ 真精确（④ 再分段）。</p>
  <details class="foldfig"><summary><b>golden 一步步改的动画</b>：，同一根柱子一截截缩</summary>
__ANIM_LINEAGE__
  </details>

  <h3>9.2　出处总表：golden 对上游默认改了什么</h3>
  <p>每一处改动记一行，连试过没采用的也写上，免得后人再试一遍。</p>
  <details class="foldfig"><summary><b>出处总表</b>：改了什么、上游默认、最终 golden ⑤、方向、为什么（几行例子）</summary>
  <table class="t9"><thead><tr><th>改了什么</th><th>上游默认</th><th>最终 golden ⑤</th><th>方向</th><th>为什么</th></tr></thead><tbody>
  <tr><td>输出头 dgrad 的长求和</td><td>BF16 张量核，沿词表一次累加</td><td>FP32、词表按 «CHUNK_SEG» 一段分段累加，结果舍回 BF16（起作用的是分段，§7.5）</td><td>GPU 向精确值、向 TPU</td><td>唯一的数值改动：12 万长的累加偏向一边（张量核截断＋吞小项），BF16 张量核和 FP32 一次累加都偏（§7.4、§7.5）</td></tr>
  <tr><td>重算</td><td>不重算</td><td>整层重算 ＋ 路由回放</td><td>GPU 向生产</td><td>按假设的生产配置开重算；回放让重算和首次前向逐位相同</td></tr>
  <tr><td>学习率、批与梯度累积</td><td>余弦调度带预热；另一种批切法</td><td>常数学习率；每卡 2 份累加，TPU 设成一样</td><td>两边设成一样</td><td>1000 步里不让调度器成为变量；累加分组决定舍入位置</td></tr>
  <tr><td>分布式优化器、通信重叠</td><td>开、开</td><td>关、关</td><td>—</td><td>开着时有一批运行持续偏一点（原因没查清）；全量抓数也要每张卡都有完整的梯度</td></tr>
  <tr><td>权重衰减</td><td>写 «WD_CONFIG»，实际 «WD_EFFECTIVE»</td><td>同原始（«WD_EFFECTIVE»）</td><td>两边设成同一个实际值</td><td>§2.3</td></tr>
  </tbody></table>
  <p>其余（结构参数、优化器其他超参、裁剪、TF32、注意力实现、混合精度、梯度规约精度）都跟上游相同；框架源码一行没改，全是配置和运行时的小补丁。定案前那份精确轨迹 ④ 的配置（全程 FP32、关 TF32、不融合注意力）只留作参考裁判。</p>
  <p>表里还有一节「试过、没采用的」：交叉熵写法、注意力实现、梯度 BF16 规约、关 TF32（在 BF16 设置下）……各自的结论是「无影响」还是「有害」。</p></details>

  <h3>9.3　拆账，和两关验收</h3>
  <p>把每条运行第 1000 步的参数范数钉在一根数轴上（下图）：TPU 生产那颗点和 ⑤（GPU 生产只把那次长求和升 FP32、分段累加）几乎叠在一起。GPU 生产到 ⑤ 那一截，全是 12 万长的求和带来的；⑤ 到 ⑥（真精确，只当参照）那一截两边都有，不是两边的差。</p>
  <details class="foldfig"><summary><b>定案前那 3.45 倍是什么</b>：两笔相反的偏差相抵后的差额</summary>
  <p>定案前那 3.45 倍又是什么？是两笔相反的偏差相抵后剩下的差额：当参照的精确轨迹 ④ 自己少漂一截，让 TPU 对它显得多漂；TPU 把流进输出头反向的梯度存成 BF16，自己也少漂一截，往回抵掉大半；剩下的差额就是 3.45 倍。</p>
  </details>
  <details class="foldfig"><summary><b>账的数</b>：每一截多少、共有误差从哪来</summary>
  <p>从 GPU 生产到 ⑤ 涨了 «LADDER_DGRAD_T»：只换 FP32 先挪 ＋«PN_RCX5_R»（去掉截断那条路），再分块又挪 ＋«PN_CHUNK_STEP_R»（去掉吞小项，推断）；从 ⑤ 到 ⑥ 再涨 «LADDER_SHARED_T»，原因是流进输出头反向的梯度在生产程序里存成了 BF16 —— GPU 生产和 TPU 生产都舍了这一次（GPU 真精确只多舍这一次，参数范数就小了 «SHARED_DLOGITS_GPU_T»）。TPU 生产落在 «PN_TPU_PROD_T»，⑤ 在 «PN_PRODCHUNK_T»。③ 到 ④ 缩掉的那截（§9.1 折叠里那支动画的橙块）大小跟这份共有误差差不多，很可能是同一类，TPU 生产也带着它（推断）。</p></details>
__FIG_DECOMP__
  <p>早先有一版拆账把 ④ 当成了精确值，而 ④ 自己也偏，那一版作废。</p>
  <p>为什么验收要分两关？像打靶：噪声尺子是同一把枪连打几发，看弹孔散得多开；它看不出整团弹孔偏离了靶心。golden 自己偏，就是靶心画歪了。所以第一关对着重画的靶心（⑤）判过不过，第二关写清旧靶心（生产 golden ②）歪在哪、歪多少。</p>
__FIG_TWOLAYER__
  <p>两档门槛（§4.4）照旧并排报：第一关每一格既写严格档的判定，也写离参考档那条线还差多少。定案后，参考档在 1000 步那一级两个平台也全过。</p>

  <h3>9.4　今天的结论</h3>
  <div class="note"><span class="t">结论：对齐了，1000 步这一级最差是噪声上限的 «FINAL_WORST_T» 倍</span>
  第一关的基线定为修好的 golden（⑤）：GPU 生产程序只把输出头那次求和改成升 FP32 再分块；基线取它的一份，尺子是它自己 «FINAL_RUNS» 份同设置重跑、两两 «FINAL_PAIRS» 对。</div>
  <table class="t9"><thead><tr><th>平台 · 配置</th><th>loss 全程</th><th>loss 尾段</th><th>梯度范数</th><th>参数范数</th><th>学习率</th></tr></thead><tbody>
  <tr><td>v5p 当前对齐配置</td><td>0.80 ✅</td><td>0.80 ✅</td><td>0.85 ✅</td><td>0.50 ✅</td><td>逐位 ✅</td></tr>
  <tr><td>v7x 当前对齐配置</td><td>0.82 ✅</td><td>0.83 ✅</td><td>0.85 ✅</td><td>0.53 ✅</td><td>逐位 ✅</td></tr>
  <tr><td>v5p EP＝8</td><td>0.76 ✅</td><td>0.78 ✅</td><td>0.74 ✅</td><td>0.48 ✅</td><td>逐位 ✅</td></tr>
  <tr><td>v7x EP＝8</td><td>0.77 ✅</td><td>0.81 ✅</td><td>0.68 ✅</td><td>0.35 ✅</td><td>逐位 ✅</td></tr>
  </tbody></table>
  <p>每格是三个统计里最差的 r。<b>v5p 和 v7x 当前对齐配置，1000 步的每一项都在 GPU 自己重跑的噪声内，最差 «FINAL_WORST_T» 倍</b>；按专家切（EP＝8）也过。换掉的只是一处查清了的 GPU 偏差；对原来那份生产 golden ② 的差（参数范数 ＋«VS_RC_PN»，loss «VS_RC_LOSS_V5P_T»（v5p；这里是验收用的整层重算配置、对 GPU 生产 5 份均值，所以跟第七节对一份基线的 −1.66e-4 不一样；§9.1 折叠的 −1.34e-4 是开关打开前那一版 TPU）），在第二关写成已解释差异：全部来自 GPU 输出头 dgrad 的长求和。</p>
  <details class="foldfig"><summary><b>换一份当基线会怎样</b>：「全过」是对事先定下的那一份说的</summary>
  <p>基线事先定为 ⑤ 的第一份。事后把十份逐份都当一次基线、尺子不变重判：单条检查的 r 在十份基线之间最多差 «REBASE_SHIFT_MAX»，四条取最差以后最多差 «REBASE_COMB_SHIFT»；v5p 两个配置、v7x 按专家切十份基线全过，v7x 当前配置有 «REBASE_V7X_PEND» 份基线落进待定（最差 «REBASE_WORST_T»），没有不过。这正是 §3.4 说的代价：基线自己也是一个样本，能挪一个 N 以内。所以报结论时写清「对哪一份基线」。</p></details>
  <details class="foldfig"><summary><b>TPU 不是挑出来的好样本</b>：«ENSEMBLE_RUNS» 份只改执行参数的扰动运行</summary>
  <p>只改执行参数的 «ENSEMBLE_RUNS» 份扰动运行（v5p «ENSEMBLE_V5P»、v7x «ENSEMBLE_V7X»），«ENSEMBLE_OK» 份四项都不超过 «FINAL_WORST_T» 倍；剩 1 份（v7x，注意力分块 1024）梯度范数待定（«ENSEMBLE_PEND_T»），落在 v7x 自己的执行噪声里：v7x 扰动组两两的梯度范数逐步涨落比 GPU 重跑宽约两成。按专家切那一档没测路由偏置曲线和健康信号；当前配置的路由偏置曲线和健康信号也过。</p></details>
  <details class="foldfig"><summary><b>怎么走到 10 份、两两 45 对的</b>：10 对时有两项待定；两处测量口径；补 5 份重跑</summary>
  <ol>
    <li>先只有 ⑤ 的 5 份（10 对），v5p 的梯度范数、v7x 的梯度范数和参数范数落在 «PEND10_T» 倍的待定区。</li>
    <li>查出两处测量口径（不是数值差）：TPU 记的参数范数是整网一次 FP32 归约，自己的单步误差 «PN_MEAS_TPU_T»，占了 GPU 重跑两两单步差上限 «PN_RERUN_STEP_T»（45 对）的一半以上（GPU 同样比法只有 «PN_MEAS_GPU_T»），改读按组求和再合成的值；梯度范数逐步涨落约 3%，以 golden 当分母时带符号均值有约 1e-3 的正偏，两两配对按哪边在前，N 从 «GN_N_AB_T» 变成 «GN_N_BA_T»，改用两边均值当分母。改完 10 对下仍有两项待定。</li>
    <li>所有 r 都在 0.7–1.3 之间，没有一项明显偏大，像是参照太少（N 取最大值，样本越少越窄）。看到待定以后决定再跑 5 份（事先没有写停止规则，跑完这 5 份没再加），10 份两两 45 对把它们全部收进来，最差 «FINAL_WORST_T»。不做那两处口径改正，45 对（10 份两两）下也全部过，最差 «FINAL_WORST_OLDCAL_T»。</li>
  </ol></details>
  <details class="foldfig"><summary><b>1000 步这一级全过；其余几级还有几格不是「严格档过」</b>：每一格都对上了一条机制</summary>
  <table class="t9"><thead><tr><th>检查</th><th>为什么</th></tr></thead><tbody>
  <tr><td>TPU 原生数据加载器与 GPU 逐样本一致（L0）</td><td>切微批的顺序不同（§2.1）；对齐时原样回放 GPU 的数据，每一步每张卡的输入指纹全部相同</td></tr>
  <tr><td>v7x 单算子、单层前向（L1、L2）</td><td>按当前配置重跑后，v7x 四层前向 R 比 «V7X_L2_R_T»（开关前 1.53–1.57），v5p 也全过；只剩第 2 层 MoE 块 R «V7X_MOE_R» 一格：4096 个 token 里有 1 个在近平局处选了跟 FP64 不同的专家，去掉它 R 是 «V7X_MOE_R_DROP»（v5p 同层去掉 2 个这样的 token 是 «V5P_MOE_R_DROP»）。一两个 token 就能让这个比值跨线，反映的是路由翻转；路由本身另有检查，两平台都过</td></tr>
  <tr><td>第 1 步整步的参数梯度、优化器状态（L3；以下是当时按两侧执行噪声的并集判的，这把尺子后来撤回，§4.3：v5p 待定 1.05–1.75 倍，第 0–2 层参数梯度不过 2.1 ／ 4.5 ／ 2.3 倍；v7x 待定 1.05–1.86 倍，第 1 层不过 3.1 倍）</td><td>两处来源：路由近平局翻转（§4.3），每层 «FLIP_SLOT_T» 的槽位两边各落一边；另一处是 TPU 有几类参数的梯度先以 BF16 落地（下一个折叠）。v5p 第 0 层是稠密层、没有路由，那一层的不过全来自后者。比到的张量合起来 rel_L2 只差 «STEP1_ALL_E»，1000 步的统计在噪声内</td></tr>
  <tr><td>第 1 步更新后主权重（L3，尺子只用 GPU 重跑：v5p 待定 «MASTER_W_V5P» 倍，v7x 待定 «MASTER_W_V7X» 倍）</td><td>同上的下游，量级很小；对 FP64 按能量 R «MASTER_W_FP64_R»，TPU 的更新量反而更接近精确</td></tr>
  <tr><td>v7x 更新后路由偏置（L3）</td><td>192 个专家里 1 个翻号。平均负载 «BIAS_MEAN_T» 个 token；按 FP64 整步算，这个专家选中 «BIAS_TOKENS_FP64» 个，比均值少，偏置该加 —— v7x 跟精确算术一致，GPU 三份和 v5p 都差一个 token 落到了均值另一边。判不过是因为当时并集的 24 对噪声里没有一对翻号，尺子宽度是 0</td></tr>
  <tr><td>开不开抓数，结果逐位相同（L4）</td><td>加抓数输出会改变编译器的融合（§6.3），所以正式验收用不加钩子的构建</td></tr>
  </tbody></table></details>
  <details class="foldfig"><summary><b>第 1 步的张量换过尺子</b>：当时改成两侧执行噪声的并集，后来撤回；对 FP64 分清谁更远；TPU 三类参数的梯度先舍成 BF16</summary>
  <ul>
    <li><b>先说结局</b>：这把并集尺子后来按「每一级的尺子都只来自参照」撤回（§4.3），混元 3 的第 1 步也按新尺子重判了：只用 GPU 第 1 步重跑 3 对，«HY3_REJ_N» 项变严（待定→不过 «HY3_REJ_A»、过→待定 «HY3_REJ_B»），没有一项变松；不过的各项标「不过 · 已解释」：机理找到了、都对过 FP64；可其中一部分张量 R 超过 1.2（«HY3_REJUDGE_R»），按 R ≤ 1.2 的字面还不算「已定责」。1000 步的结论不变。下面几条是当时的做法。</li>
    <li><b>为什么换</b>：第 1 步 GPU 重跑路由一个不翻，尺子对路由翻转宽度是 0；可 TPU 只改执行参数也会翻。第 1 步整步的张量改用两侧执行噪声放在一起（并集）当尺子：GPU 第 1 步重跑 3 份、两两 «UNION_GPU_PAIRS» 对，加上 TPU 只改执行参数的 «UNION_TPU_RUNS» 份、两两 «UNION_TPU_PAIRS» 对，共 «UNION_PAIRS» 对，取最大当 N。</li>
    <li><b>逐层隔离回放的反向</b>：每层输入固定，只改执行方式时路由不翻，并集里没有路由翻转这一份，改按 FP64 的 R 比判（跟单算子、单层一样）：v5p «ISO_BWD_V5P_T»，v7x «ISO_BWD_V7X_T»，全部 ≤ «R_GATE»，这一项结案。</li>
    <li><b>整步超出并集的部分对 FP64 定责</b>（只用来分清谁离精确更远，不当门槛）：把每个张量的误差按梯度大小加权合起来，整网 R «STEP_R_T»。投影、共享专家、稠密 MLP 的梯度 R «STEP_R_PROJ_T»，TPU 更接近精确；路由专家与路由权重 «STEP_R_EXPERT_T»（v5p；v7x 最高 «STEP_R_EXPERT_V7X_MAX»），两边差不多远，来自路由近平局；一维 norm 权重、输出层、词嵌入的梯度大多是 TPU 更远，最高 «STEP_R_NORMMAX_V5P»（v7x «STEP_R_NORMMAX_V7X»）。</li>
    <li><b>后面那一处的机制</b>（逐一用诊断改动验证过）：TPU 框架把这几类 FP32 参数先转成 BF16 再用，梯度在每个设备、每份小批上先舍成 BF16 存下来（落地），再累加回 FP32；GPU 框架把权重梯度直接累加进 FP32。三个地方：RMSNorm 把 scale 舍到 BF16（JAX 在反向对梯度也舍一次）；输出层先把权重转成 BF16 再乘，权重梯度也以 BF16 输出；词嵌入整张表先转 BF16 再取行，反向把梯度加回各行（scatter-add）也在 BF16 里。逐处改掉后 R（v5p，按组）：«ARM_R_T»；三处一起，整网 1.20 → 0.67，第 1 步 loss 一位不变。</li>
    <li><b>对应的开关</b>：三个，默认关，前向不变（第 1 步 loss 一位不变），步时 v5p ＋«GRAD_FP32_COST_T»、v7x ＋«GF32_COST_V7X_T»（100 步对比）。它只管这一处，路由翻转那一份不受影响；v7x 上整网 R 1.21 → 0.72。打开以后 1000 步对 ⑤ 复核过：v5p 全过（«GF32_V5P_T»）；v7x «GF32_V7X_T» 过，梯度范数待定，«GF32_V7X_GN_T»。<b>没有加进正式对齐配置</b>，是可选开关：1000 步这一级不需要它，只有想让第 1 步这三类梯度不比 GPU 离 FP64 远时才开。</li>
    <li><b>签字清单里「第 1 步的噪声检查全部过」这一项</b>：按签字原则不满足；超线处的机理都已查到、实测不累积，但其中一部分张量 R 还没压到 1.2 以内，没达到定责线；接不接受由使用方决定。</li>
  </ul></details>
  <details class="foldfig"><summary><b>想两边对得更紧，要动哪些开关</b>：TPU 一侧必需两项、推荐两项，GPU 一侧一处，各有代价</summary>
  <ul>
    <li>TPU 必需：边传边乘（collective matmul）的部分和留在 FP32（§5.4），步时 v7x ＋«DFA_COST_V7X_T»、v5p ＋«DFA_COST_V5P_T»。它是单算子、单层这两级要的；1000 步这一级不靠它，开关打开前的配置对 ⑤ 也全过（最差 v5p «PRESWITCH_VS_FINAL_V5P»、v7x «PRESWITCH_VS_FINAL_V7X»）。梯度累积的分组跟 GPU 一样。</li>
    <li>TPU 推荐：RMSNorm 在 FP32 里算完只舍一次（第二次迁移在 v5p、运行库 0.0.52 上，不开就超，§10.2）；门控激活逐个算子舍入（不开时这个算子只有约四分之三元素跟 GPU 逐位）。可选：按专家切，步时反而少 «EP8_SPEEDUP_V7X_T»（v7x）。</li>
    <li>GPU（只影响 golden 的制作）：输出头 dgrad 分段累加，最终 golden 唯一的数值改动。⑤ 升 FP32 再分段，步时多 «CHUNK_COST_T»（大头在升 FP32，按单次乘法耗时推断）；乘法留在 BF16 张量核、只分段，1000 步跟 ⑤ 分不开，步时只多 «CB_COST_T»（§7.5）。</li>
    <li>想朝真精确再近一步：两边都让 dlogits 留在 FP32（TPU 步时多 «HEADFP32_COST_T»，v7x），对齐本身不需要。</li>
  </ul></details>

''' + quiz("只有 5 份、两两 10 对时几项落在 1.02–1.24 倍，补到 10 份、两两 45 对就全过了。这算不算「把尺子放宽了好过关」？",
           "先老实交代：补跑和改口径都是看到待定以后做的，事先没有写停止规则；跑完 10 份以后没再加。所以要靠下面三条让你自己判断：N 取最大值，样本少时系统性偏窄，10 对时所有 r 都挤在 0.7–1.3、没有离群项，像尺子短而不像真差；另外两条旁证 —— 不做口径改正、10 份两两 45 对下最差 «FINAL_WORST_OLDCAL_T»，TPU «ENSEMBLE_RUNS» 份扰动运行里 «ENSEMBLE_OK» 份不超过 «FINAL_WORST_T»，剩下一份的待定也落在 TPU 自己的执行噪声里。") + '''
  <h3>9.5　开关上不上生产：三类</h3>
  <p class="lead">验收时为对齐打开的开关，上线那天要是原样全开，有的白白多花步时，有的还让 TPU 变得不如不开准。</p>
__FIG_SWITCH__
  <p>对齐时开的开关分三类看：部分和 FP32、RMSNorm 只舍一次（第二次迁移最后一层输出那两次舍入）是少舍、更准，按第三问看代价；门控激活逐个算子舍是多舍一次，属对齐专用，生产关。1% 是本课取的界，按目标规模和平台重量一次再定。</p>

  <h3>9.6　第二次迁移：换一个模型，方法还灵不灵</h3>
  <p>第二次迁移照同一条路走：先在保留全部结构、只缩宽度和层数的小模型上做 L0–L4 全套；再让 GPU 自己扩到多台机器、换切法，看参照守不守得住尺子；最后上真实宽度。模型的规模要一级级上，像装修一次只换一样：小模型上查出的根源不能直接外推，一下子跳到真实权重，冒出新的差又说不清是谁带来的。</p>
  <p>结果一句话：小尺寸 1000 步九项过八项；GPU 扩机器、换切法守得住尺子；真实宽度上冒出的差，一半定到一个常数舍入（注意力里 eˣ 改用 2 为底，常数 log₂e 被舍成 BF16，叫 base2），还有一截在查。</p>
  <details class="foldfig"><summary><b>第二次迁移的全套结果</b>：梯子图、拆成两半的那个差、三块成绩单、base2 机理</summary>
__FIG_LADDER__
  <p>第二次迁移一开始就是从小模型直接跳到真实权重截层，第 1 步前向冒出一个小模型上没有的差，那一步同时换了尺寸、专家和权重。补上中间一级、而且<b>两边各自对 FP64</b> 以后，这个差拆成了两半（图下那条带）：GPU 一半，TPU 一半。只看两边的差，这两半是分不开的。全套结果：</p>
__FIG_MIG2__
  <table class="t9"><thead><tr><th>图里三块</th><th>结论</th><th>还差什么</th></tr></thead><tbody>
  <tr><td>小尺寸 1000 步（2026-11-11 起关 base2、多路残差一带照 GPU 写法舍，尺子 GPU 16 份）</td><td>九项过 8、待定 1（路由偏置均值；同配置另五份这一项都过，其中一份主 loss、MTP 略超，一份梯度范数略超，都不到 1.2）；主 loss «C_LOSS_R_T» 倍，刚好过线</td><td>上一代 TPU 8 份主 loss 都偏高、平均约 «MINI_TPU_HIGH_T»：前段鼓包、后段主因都查到了（§9.7）；可对齐配置的后段跟不修分不开，机制在查；第 1 步在新配置上重判，还剩 1 个小参数待定；离签字还差几件</td></tr>
  <tr><td>GPU 扩机器、换切法</td><td>守得住尺子，多机 GPU 可以当 golden</td><td>多机重跑本身略宽（单机尺子的 1.0–1.3 倍），判多机被测要用多机自己的尺子</td></tr>
  <tr><td>真实宽度三层</td><td>第 1 步 TPU 偏低定到 base2：常数 log₂e 跟着 BF16 的 q 舍成 1.4453125，logits 整体乘 «KAPPA_T»；只关它就对上 FP64，所有尺寸的对齐配置因此都关了 base2</td><td>GPU 第 1 步自己也偏高（对 FP64，4 份中位数 «LAD_REAL_GPU_T»），只拆清约一半；后段主 loss 偏低：同一份权重上 TPU 前向反而高 1.5e-5（第 151 步的单批读数；第 900 步 32 批平均差不到一个标准误），所以偏低是轨迹差，来源在查（base2、索引器、省舍入已排除）</td></tr>
  </tbody></table>
  <p>base2 在小模型上显不出来，是因为效果跟 logits 的大小成正比：随机初始化时最大 logit 只有约 0.13，真实权重截层上到 14.5。</p>
  <p class="codeline">''' + mt(4, "src/maxtext/configs/base.yml", "L1405", "开关 sa_use_base2_exp（默认开）") + '''</p>
  </details>
  <p>第三次迁移换了一个起点：不从对齐好的代码出发，从生产代码原样出发，一个开关一个开关地打开，每一级跑一遍全套。这样每个开关修了什么、代价多少、上不上生产（§9.5 的三类），一级一级都记得清。其中打开投影 FP32 累加那一级，修的就是 §5.4 那件只在 8 个以上设备才有的事。它也补上一句：<b>先缩模型、不缩设备</b> —— 模型宽度和层数可以从小做起，可精度验证用的设备数和切法要跟生产一样，有的差只在 8 个以上设备才长出来（§5.4）。</p>
''' + quiz("中等尺寸那一级过了、真实权重截层出了差，能说就是「真实权重」的问题吗？",
           "不能。第三级同时换了专家个数、专家宽度、层数和权重，四样拆不开。而且「过了」只是 300 步那一级：第二级第 1 步两边各自对 FP64，GPU 已经偏高 1.42e-5（比 GPU 8 份之间的极差还大）、TPU 只有 2.4e-7（最老一代配置的读数；现行配置 TPU −2.67e-6，两边的差仍在 GPU 一侧）—— 宽度一换成真的，GPU 那边就有一截偏差（还在查），只是 300 步的判定没把它显出来。要定，就再补一级只换其中一样，比如真实结构配随机权重。（TPU 那一半后来定到了「权重」那一行：真实权重的 logits 大，把 base2 常数舍入的误差放大了，§9.6。）") + '''
  <h3>9.7　一条主 loss 的尾巴：一路排除，最后先编译、再扫描</h3>
  <p>第二次迁移，2026-11-11 换对齐配置之前，小尺寸的主 loss 判了待定（1.55 倍）：八份 TPU 全部比 GPU 高，第 50–100 步有个小鼓包，中间贴着零，第 550 步以后一路往上爬。八份同一个方向，不像噪声；同一份权重上两边前向几乎没差，所以是训练里一步步攒出来的。这是第二次迁移里查得最久的一件；最后给出候选清单的，不是又一次上卡，是一次离线编译。</p>
  <p>还要记一句：延长到 2000 步时，修之前的尾巴在第 1000–1250 步到顶（只跑了 1 份），之后回落到 GPU 的散布里 —— 它更像一段暂时的轨迹差，不是一直在长的系统偏差。</p>
__FIG_MINIDRIFT__
  <details class="foldfig"><summary><b>地图与前段鼓包</b>：多路残差上写了「转 BF16」的三处；收拢处梯度那次被省，前段鼓包先破</summary>
  <p>先认地方。混元 4 把残差流复制成 4 路（多路残差），下面这张地图标出三处在代码里写了「转 BF16」的地方。再记住 §5.5 那件事：「转 BF16、再转回 FP32」的一对，编译器开着省舍入时可以省掉；写成 <code>reduce_precision</code>（强制舍入）就省不掉。后面每补一处，用的都是它。</p>
__FIG_SITES__
  <p><b>前段那个小鼓包先破了。</b>在 ② 收拢处，GPU 前向舍一次、梯度传回来也舍；TPU 省掉了梯度那次。在收拢处强制舍，第 51–100 步回到 GPU 附近；只在前向舍、梯度原样传回，鼓包又回来。可后段没降，均值反而略高（4 份，单看分不开）。记住这个补丁，它后面还会回来。</p>
  </details>
  <p><b>后段先排除嫌疑。</b>关掉 MTP、关掉蒸馏，都还偏；换运行库、换芯片、换并行方式，35 次长跑最后 250 步全是正的；两边都把多路残差换成普通残差，尾巴没了。嫌疑锁定在多路残差这一段：</p>
__FIG_IHCROOT__
  <p><b>两头验证，定下主因：流梯度。</b>多路残差的流在一层里被用三次，反向回来三份梯度要相加。GPU 三份梯度每加一次舍一次 BF16；TPU 推断是在 FP32 里加完才舍一次。每次差得很小，一千步攒成那截尾巴。照 GPU 的写法补上（每份先舍、按 GPU 的顺序逐个加；分支输出也舍），尾巴降下来（后段 ＋9.4e-6 → ＋1.9e-6）。</p>
  <details class="foldfig"><summary><b>只补 TPU 一边的读数</b></summary>
__FIG_FIXCMP__
  </details>
  <p>只补一边还不够，像医生判一种药的副作用：停药症状消失，换个人吃同样的药也长出同样的症状，两头都对上才算是它。这次两头各改一处，每格 4–5 份：</p>
__FIG_TWOWAY__
  <p>TPU 改成 GPU 的写法，偏高降下来；GPU 改成 TPU 的写法，GPU 自己长出同样的尾巴。两个方向对上，主因定在这里（推断：两头方向一致，每格只有 4–5 份）。（图里的 p 是检验值：把这些运行随机重新分组，分出这么大差别的概率；4 份对 5 份一共只有 126 种分组，完全分开时 p 最小就是 0.016。这一节做了十几次这样的检验，没做多重比较校正，单个 p 0.02 上下只当方向性的证据。）</p>
  <p>可把收拢处（最后四路收成一路那处）也照 GPU 补上（现行对齐配置），后段还剩 «FC_C_MEAN_T»，还在查。剩下的不靠猜：<b>先编译，再扫描</b>。</p>
  <details class="foldfig"><summary><b>反转、一台秤三个毛病、丙在哪、在一处上看清楚</b>（推断多；动画 anim-cancel、anim-declared 在这里）</summary>
  <p><b>反转。</b>把 ① 和 ② 都照 GPU 补上，就是现行的对齐配置；可后段还剩 «FC_C_MEAN_T»，比只补流梯度大，跟修之前分不开（p «TW_P_CCTL_T»）。收拢处一开，再补流梯度就测不出作用了（逐份读数收在下面的折叠里）。</p>
<details class="foldfig"><summary><b>逐份读数</b>：收拢处开着时，补不补流梯度分不开</summary>
__FIG_GATECMP__
</details>
  <p><b>为什么？对齐记录的解释（推断）：一台秤有三个毛病。</b>甲让读数偏重，是流梯度少舍（主因，已找到）；乙让读数偏轻，是收拢处那次梯度少舍；丙也偏重，还没找到。只修甲，乙和丙大致抵掉，读数看着准了；甲乙都修，丙就露出来了。乙这个毛病有点怪：在前段它把 loss 往上鼓（就是那个鼓包），在后段却往下压。</p>
__FIG_CANCEL__
__ANIM_CANCEL__
  <p>按这个解释，对齐配置剩下的 «FC_C_MEAN_T» 主要是丙（推断）。下一步：找丙。</p>
  <p><b>丙在哪？先查能一次列全的那种。</b>丙有两种可能：两边算的东西不一样，或者算的一样、舍的位置不一样。收拢处这一段的代码已经对到只差累加顺序，上卡照旧（出处见本节末的折叠）；鼓包那处在后端指令里核过，是代码写了舍、编译器省掉了；流梯度两头验证定了是原因，按同一规律推断也是被省掉。后一种有个好处，能让编译器自己交代，一次列全。先查便宜的。</p>
  <p>先在一处上看清楚，就拿 ⓪ 来看：它在这次排查之前就补上了；把它退回没补时的写法，为 v5p 离线编译：HLO 里那对 convert 都在，可算均方那颗 kernel 的后端指令（LLO）里，把 FP32 打包成 BF16 的指令一条都没有（§8.5 也是这样数的）；同一处写成强制舍入，就有 «EL_IHC_FIXED» 条。往返留在一颗 kernel 里就不舍；值写回内存交给下一颗、或直接喂给矩阵乘，才真舍：</p>
__ANIM_DECLARED__
  </details>
  <div class="note ok" style="border-left:6px solid #e8710a"><span class="t">这一节最想让你带走的一招：先编译，再扫描</span>
  <p>代码里的 convert 往返（FP32 → BF16 → FP32）不改数学，写它只有一个意思：<b>这里要舍一次</b>。编译器觉得这个值马上还要接着算，就没舍。不用一张张猜：把整个训练步离线编译一次（不用上卡），把同类位置全列出来（下图）。</p></div>
__FIG_ELIDED__
  <details class="foldfig"><summary><b>扫描过程的动画</b></summary>
__ANIM_SCAN__
  </details>
  <p>三根橙条 A、B、C 是先查的范围，A、B 在反向的梯度路径上，排在最前。A 跟刚抓到的流梯度是同一种手法：几份梯度在 FP32 里合流，只是换到了各层分支的输入上（典型是注意力的输入）。这些地方 TPU 少舍一次，单看这一处离精确值更近（推断，没量过）；可对齐比的是像不像 GPU，GPU 在这里舍了，两边就差这一次。</p>
  <details class="foldfig"><summary><b>这一招的两条边界</b>：只找得到这一类；GPU 舍不舍要对照 GPU 代码</summary>
  <p>两条边界：</p>
  <ul>
    <li><b>只找得到这一类。</b>写了往返、被编译器省掉的，扫得出来；代码里本来没写舍入、累加顺序不同、近似超越函数这类差，扫不出来。</li>
    <li><b>GPU 舍不舍，要对照 GPU 代码。</b>清单只说 TPU 省掉了。dW 这一类算数，是因为 GPU 参照关了 Megatron 的 <code>gradient_accumulation_fusion</code>，每个微批的 dW 先出 BF16 再加进 FP32 的累加器；开着的话，这一类就不算差别。另外只有 dW 那一类整颗 kernel 只含往返，能在 LLO 里一颗颗数；另两类跟真舍混在一起，是按同一规则推断的。</li>
  </ul>
  </details>
  <p><b>收尾：上卡验证。</b>还是两头验证那一招：GPU 上把 A、B 拆成三处照 TPU 改，看 GPU 自己长不长出尾巴；TPU 上在对齐配置之上补上，看尾巴下不下去。</p>
__FIG_VERIFY__
  <p>两头都没对上，A、B 两类排除，C 前向几处还没验，残余还在查（GPU 那头注意力输入单开 p 0.021 也不算：两处合开就没了，TPU 一头补上也不动）；下一批是 C 类前向几处，和扫描看不到的差。这就是<b>清单是候选，不是结论</b>：扫描告诉你先查哪儿，判它的只有上卡。</p>
  <details class="foldfig"><summary><b>两种候选配置并排</b>（待拍板）：三处都照 GPU 写法，还是不开收拢处</summary>
  <table class="t9"><thead><tr><th></th><th>对齐配置（三处都照 GPU 写法，现行）</th><th>不开收拢处（只补到流梯度）</th></tr></thead><tbody>
  <tr><td>1000 步，第一份</td><td>过 8、待定 1（路由偏置均值 1.31）</td><td>九项全过</td></tr>
  <tr><td>1000 步，各份</td><td>«CF_C_N» 份没有不过，«CF_C_ALL» 份九项全过</td><td>«CF_M_N» 份没有不过，«CF_M_ALL» 份九项全过</td></tr>
  <tr><td>前 300 步，第一份</td><td>过 8、待定 1（梯度范数 1.01）</td><td>过 7、待定 2（主 loss 1.18、梯度范数 1.30）</td></tr>
  <tr><td>后段（第 551–1000 步）主 loss 偏差</td><td>«FC_C_MEAN_T»（«CF_C_N» 份），跟不修分不开</td><td>«FC_M_MEAN_T»（5 份，2 份仍有尾巴）</td></tr>
  <tr><td>前段鼓包（第 51–100 步）</td><td>«CF_C_BUMP_T»，没了</td><td>«CF_M_BUMP_T»，还在</td></tr>
  <tr><td>备注</td><td>—</td><td>这几份用的是早一代 MaxText</td></tr>
  </tbody></table>
  <p>两边各缺一块：一个后段没修掉，一个前段鼓包还在。哪个当对齐配置，取决于更在意哪一段，课程不替它定。</p>
  </details>
  <details class="foldfig"><summary><b>出处</b>：收拢处这一段代码已经对到只差累加顺序</summary>
  <p>收拢处加末层归一化这一段，在 CPU 上跟 GPU 逐位对照；再开一个让两边共用一次末层归一化的开关，两边就只差 FP32 累加顺序。可上卡 «EL_SHARED_N» 份，后段照旧（对对齐配置 p «EL_SHARED_P_T»）。</p></details>
  <details class="foldfig"><summary><b>照着做</b>：自己的模型怎么扫</summary>
  <ol>
    <li>离线编译整个训练步（CPU 上为目标芯片编译就行，不用上卡），存下优化后的 HLO 和后端指令（LLO）。</li>
    <li>找每一处 convert(f32→bf16)，看它在同一颗融合里有没有又被 convert 回 f32 用。前面已经有 reduce_precision、或者输入本来就在 BF16 网格上的，往返是精确的，跳过。</li>
    <li>按 HLO 里记的源码行和形状归类，看是哪个模块、前向还是反向。</li>
    <li>只含往返的 kernel，数 LLO 里打包成 BF16 的指令，应该是 0；再关掉省舍入编一次当负对照，应该不再是 0。</li>
    <li>每一类去对照 GPU 代码，看那里舍不舍；GPU 舍的，才是候选。候选再上卡，两头各改一处验证。</li>
  </ol></details>
  <p class="codeline">''' + code("models/hunyuan4/cases/17-late-drift-elided-rounding.md", "对齐记录 · 案例 17「后段为什么慢慢偏开」（重点在「先编译、再扫描」一节）") + "　" + code("models/hunyuan4/analysis/elided_rounding_scan.py", "扫描脚本") + "　" + code("models/hunyuan4/alignments/01-mini/results/mini-c-elided-rounding-scan.json", "整步扫描结果") + "　" + code("docs/guide/06-debugging.md", "排查指南「扫全图找被省掉的舍入」") + '''</p>
  <p class="bridge">这一节留下的问题：混元 3 的验收过了。模型投产以后训练代码每天都在变，怎么知道它没有悄悄改了数值？</p>
</div></section>

''' + sec("s十", "十", "工程化与教训") + '''
  <p class="lead">守住数值靠回归；少走弯路，靠把走过的弯路记下来。</p>

  <h3>10.1　TPU 的脾气：改了什么，结果变不变</h3>
  <p>改完结果不逐位了，是改出了 bug，还是只换了加法顺序？第二次迁移把日常会碰到的改动一样一样量了一遍：每次只改一样东西，跟一次确认过的 TPU 运行比，尺子借 GPU 重跑（TPU 自己重跑逐位相同，量不出宽度），第 1 步再对 FP64 定责。</p>
__FIG_TEMPER__
  <p class="codeline">''' + code("models/hunyuan4/alignments/04-tpu-self/build_tables.py", "按结果文件自动分堆的生成器") + '''</p>
  <p>TPU 的脾气分三堆：</p>
  <ul>
    <li><b>逐位相同</b>：重跑、换机器、换集群，以及起名字、拆函数这类不改程序的重构。</li>
    <li><b>不逐位，但跟重跑分不开</b>：只换顺序的改动 —— 换切分、专家并行、重算策略、升运行库。换到 v7x 原先也放在这堆，后来查出它动了精度，只是被小参数单列不判的豁免盖住了。</li>
    <li><b>真有影响的归成七件</b>：在运行库 0.0.46 上，不能忍的是纯数据并行（第八节）和关掉省舍入。另外五件曲线分不开，只在第 1 步对 FP64 抓到：关小矩阵乘改写、RMSNorm 多舍一次、多路残差不强制舍 BF16、上下文并行四件能忍，各有开关；第七件是输出头反向少舍一次（换 v7x、einsum 换广播乘、跨切片数据并行都碰上，输出头 3 个小参数 R 7.5–46），2026-11-12 单列加了倍数上限才显出来，是真实的数值缺陷，忍不忍待拍板。<details class="foldfig"><summary><b>七件里的细节</b></summary><p>其中 RMSNorm 那件只靠一个压线张量判出来：第 0 层归一化权重 R «RMS_EDGE3_T»，GPU 参照换成六份是 «RMS_EDGE6_T»，就落回线下 —— 贴着线的判定，份数一变就翻。换到 0.0.52，v5p 上 RMSNorm 多舍一次也不能忍了（§10.2）。</p></details></li>
  </ul>
  <p>其中 «LONG_ARMS» 个改动又跑满 1000 步、用 GPU 16 份当尺子（运行库 0.0.46），有 3 个判「超」（多路残差不强制舍 BF16，和换 v7x、einsum 换广播乘这两次的 1000 步，后两次是 2026-11-12 单列加上限以后改判的），都是同一次比较里第 1 步对 FP64 判出来的；它们的 1000 步曲线 r 0.91–1.36，仍在 GPU 纯噪声能拿的 1.82 以内。</p>
  <details class="foldfig"><summary><b>几个值得记住的脾气</b>：换 v7x、换运行库、重算、「第 1 步一样、后面分开」</summary>
  <ul>
    <li><b>换到 v7x</b>：同一个镜像、同一个程序，第 1 步梯度相对差中位数 «V7X_STEP1_MED»、离 FP64 的比值 «V7X_FP64_RATIO»；300 步最差 r «V7X_R300»；v7x 上自己重跑逐位相同。编译出的程序不一样（v7x 直接用 reduce-scatter），步时 «V7X_STEP_T» 对 v5p «V5P_STEP_T»，按芯片·秒省 «V7X_SAVE»。关省舍入在 v7x 上同样分得开，但机制不同：v7x 上 FSDP 那次词嵌入 scatter 交给了 SparseCore，没变远（1.11 倍）；超出来的是梯度范数，主干梯度整体小约 4e-4。v5p 上也有同样的缩小（被词嵌入盖住了），在那里的最小复现里看后端指令，查到多路残差那一段反向里一颗求和，后端把每个 FP32 乘积先舍成 BF16 再累加；照这一处改写，MLP 的缩小几乎全部复现，注意力只复现约三分之二。v7x 是按同一机制推断，为什么会让梯度系统性变小也没查清。这一层在编译后的 HLO 里看不到，要看后端指令。改纯数据并行在 v7x 上曲线 r 2.21，按 2 倍线判超、按纯噪声最高 2.47 读分不开，最后是第 1 步对 FP64 抓到的。<br>换到 v7x 本身也不是纯换顺序：输出头反向那颗 kernel 里省掉了一次 BF16 舍入，多路残差收拢处三个小参数第 1 步«V7X_HCHEAD_T»，有几个元素符号反了。它们原先被单列，没有倍数上限，曲线就判成「分不开」；2026-11-12 单列加了上限（R 5.22），这几次改判「超」，归成真有影响的第七件。<b>单列的张量偏了多少也要看一眼：没有上限的豁免，会把真缺陷盖住。</b></li>
    <li><b>换运行库</b>：0.0.45–0.0.48 相邻版本两两比，有一对逐位相同，有两对不逐位但跟重跑分不开。第 1 步的变化比换切分小一个数量级，可几百步以后照样分开到同一个量级：轨迹会把任何不逐位的扰动放大到「换了顺序」的幅度。但「基准运行跟重跑分不开」不等于「别的改动的结论不变」，见 §10.2。</li>
    <li><b>重算策略</b>：数学上应当逐位相同，三种实测都不逐位 —— 编译器按不同的保存方案重排了融合和通信，求和顺序跟着变，跟换切分是同一个量级。</li>
    <li><b>第 1 步一样、后面分开，不一定是 bug</b>：把 all-reduce 交给 SparseCore（芯片上的专用小核），第 1 步 loss 逐位不变，可 14 个专家权重的梯度变了，更新以后第 2 步起分开；300 步 r «SC_AR_R»。一次集合通信换一条硬件路径，就能造出这个现象。</li>
  </ul></details>
  <details class="foldfig"><summary><b>这把尺子能抓多小</b>：故意埋 bug 扫大小（图）</summary>
  <p><b>这把尺子能抓多小？</b>反过来故意埋 bug、扫大小：学习率偏大 1e-4 就抓得到，偏 3e-5 以下跟重跑分不开。也有尺子量不出来的：</p>
__FIG_SENS__
  </details>
  <p>所以两条用法：<b>记下的每一项 loss 都逐项比</b>，不只看总 loss；<b>动到精度的改动，曲线分不开也要对 FP64 看一眼</b>（多路残差那一个就是第 1 步抓到的）。折叠图里那 5 个动了精度的改动，两样都抓不到 —— 分不开只说明这把尺子量不出，不说明没影响。</p>


  <h3>10.2　换运行库：不一定只换了加法顺序</h3>
  <p>上面那张图都在 TPU 运行库 0.0.46 上量的。升到 0.0.52，对齐记录把同一条路又走了一遍：先离线编译看哪些程序变了，再上卡重建基准运行、把当时的六件逐件重测（第七件 11-12 才显出，没在这批里），每件对同版本的基准比：</p>
__FIG_LIBTPU__
  <ul>
    <li><b>基准运行自己</b>：同版本重跑逐位相同；新对旧 300 ／ 1000 步 «LIB_GOLDEN_R_T»，在噪声内，步时少 «LIB_GOLDEN_STEP_T»。单看这一条，换版本跟重跑分不开。</li>
    <li><b>一件变强</b>：RMSNorm 多舍一次，r «LIB_RMS_PAIR_T»（1000 步 «LIB_RMS_R1K_T»），主 loss 一直偏高，换一组机器重跑逐位相同；第 1 步看不出，是跑起来攒的。为什么变强，对齐记录已经查清，见本节末一句。</li>
    <li><b>一件变弱、一个门槛变了</b>：上下文并行降到 0.95（那个张量本来就随加法顺序跳一倍，只算这次没过线）；词嵌入排序门槛变了，v7x 上不累积的纯数据并行，词嵌入梯度不再丢精度（§8.6；这次运行整体另因输出头 3 个小参数判「超」，是另一件）。纯数据并行、关省舍入、关小矩阵乘改写、多路残差这四件，在 v5p 上结论不变。</li>
  </ul>
  <p><b>所以换版本不能当成「只换了加法顺序」</b>：JAX 交出的是同一份程序，编译器优化后的程序（v5p、v7x 各种并行一共 113 项、去重后 103 份配置，编出的程序在两个版本下全部不同）和后端的代码生成都一起换了。</p>
  <p class="codeline">''' + code("models/hunyuan4/alignments/04-tpu-self/libtpu-052.md", "升运行库 0.0.52 的三步与全部结果") + '''</p>
  <p>逐步看，RMSNorm 那件在 0.0.52 上是真的「一直偏高」，不是一两步的尖峰：</p>
__FIG_RMS52__
  <p>RMSNorm 那件变强，是因为 0.0.52 挡住了一次融合，两次舍入都真舍了，而第二次的输入已经在格点上，结果系统偏小。</p>
  <details class="foldfig"><summary><b>RMSNorm 那件为什么变强</b>：两次舍入为什么系统偏小（图、机理）</summary>
  <p><b>RMSNorm 那件为什么变强</b>（v5p、FSDP 8 上查清）：默认写法是先把归一化值舍成 BF16、乘 γ 再舍一次。运行库 0.0.46 上，最后一层的这一段融在一颗 kernel 里，中间那次被省掉，实际只舍一次；0.0.52 在定内存布局时给归一化值插了一次布局转换，挡住了融合，两次舍入都真发生了：</p>
__FIG_RMS2R__
  <p>两次舍入本该是随机误差，可第二次舍入的输入已经在 BF16 格点上，乘一个贴近 1 的 γ 再舍，往哪边落由 γ 决定，于是输出头的输入被整体缩小。模型学会预测以后，这一点缩小就把预测压平、抬高 loss。<b>所以换版本时，「开关开着没事」比「这件以前能忍」可靠得多</b>。</p>
  <p class="codeline">''' + code("models/hunyuan4/alignments/04-tpu-self/impacts/04-rmsnorm-double-rounding.md", "RMSNorm 多舍一次：0.0.52 上为什么变强（证据六条）") + '''</p>
  </details>
''' + quiz("升级运行库以后，新基准跟旧基准在噪声内。能不能直接沿用旧版本上「这几件能忍」的结论？",
           "不能。基准运行只说明「这一个配置」换版本跟重跑分不开；别的改动走的 kernel 可能变了。0.0.46 → 0.0.52 就是例子：基准 0.99，可 RMSNorm 多舍一次从 1.06 变成 5.50。要把已知的几件在新版本上重测，或者干脆把相应的开关都开着（RMSNorm 单次舍入）。") + '''


  <h3>10.3　日常回归：逐位优先</h3>
  <p><b>第一步：改完先比程序指纹</b>（编译后的整份程序去掉元数据、名字重新编号以后的哈希）。像门禁按工牌放行：今天的程序跟上次登记的一字不差，结果就一定逐位相同，回归直接过；不一样，再跑 300 步看档。对齐记录把它做成了跑前离线筛，拿已经上过卡的改动回测过：</p>
__FIG_SCREEN__
  <details class="foldfig"><summary><b>离线筛怎么比、那几类为什么判不了</b></summary>
  <ul>
    <li><b>怎么比</b>：取编译器优化后的训练步程序，去掉元数据和源码位置，指令名按出现顺序重新编号（CPU 上编和 TPU 上编，名字里的序号会差几十），设备分组展开成设备号，每个子计算的参数按编号排到最前，然后逐字比。CPU 上编出的程序跟 TPU 上实际存下的，规范化以后一字不差。</li>
    <li><b>判不了的几类</b>：换运行库、换镜像（后端代码生成一起换了，程序相同也不能推出逐位相同）；换芯片代际（本来就是两个程序）；SparseCore 卸载的开关（卸载可能不体现在优化后的程序里，规则上谨慎起见直接上卡；目前程序相同的两个这类开关，实测结果也都逐位）。这几类直接上卡。</li>
    <li><b>程序不同，结果也可能逐位相同</b>：比如全局矩阵乘最高精度，程序变了、结果一位没变。离线筛只会让你多跑，不会让你漏跑。</li>
  </ul></details>
  <p class="codeline">''' + code("models/hunyuan4/alignments/04-tpu-self/hlo_screen.py", "跑前离线筛") + "　" + code("models/hunyuan4/alignments/04-tpu-self/setup.md", "用法与规范化规则（「跑前离线筛」一节）") + '''</p>
  <p>指纹加上噪声尺子和 FP64，还抓到过一个编译器的真 bug：数据并行 ＋ 梯度累积 ＋ 词嵌入梯度 FP32 累加三样同时成立时，编译器把跨卡求和挪出累积循环，又在循环里把 1–7 号卡的累加器清零，第一个微批的梯度没了、只剩第二个（r 611，第八节撞上的就是它）。</p>
  <details class="foldfig"><summary><b>程序指纹与那个编译器 bug</b>（图）：程序指纹哪些改动一样；1–7 号卡微批 1 被清零的那个 bug</summary>
__FIG_BUG__
  </details>
  <p>修法只在那三样同时成立时生效；golden 用修后的代码重跑逐位不变、程序指纹也不变 —— 修 bug 的改动自己先过了「数学不变」那一行。</p>
  <p>指纹变了，再按下面这张流程分流：</p>
__FIG_REGRESS__
  <table class="t9"><thead><tr><th>PR 声明</th><th>什么时候用</th><th>回归怎么判</th></tr></thead><tbody>
  <tr><td>数学不变</td><td>重构、加日志、预期逐位不变的性能改动</td><td>全部逐位；不逐位就退回。没写声明的按这一条</td></tr>
  <tr><td>会改数值</td><td>改精度、算法、默认配置、并行或重算设置；会改变融合、分块的性能改动</td><td>写明是<b>执行级</b>（只改舍入位置）还是<b>数学级</b>（公式、精度、算法变了）：执行级用 TPU 执行噪声判，数学级对 GPU golden 判（细节见表下折叠）</td></tr>
  <tr><td>升级依赖</td><td>框架、编译器、kernel 库、镜像、编译开关</td><td>先要求逐位；不逐位按执行级判，并做归因。换运行库这类连后端一起换的，还要在新版本上建基准、把已知会动精度的几件重测（§10.2）</td></tr>
  </tbody></table>
  <details class="foldfig"><summary><b>「会改数值」那一行的细节</b></summary>
  <p>写明是<b>执行级</b>（只改舍入位置，比如融合、分块变了）还是<b>数学级</b>（公式、精度、算法变了），以及会影响哪些检查；执行级用 TPU 执行噪声判，数学级对起点那份 GPU golden（§9.1 的①）不能变差 —— 这道门是两关验收定下来之前设的；自己搭回归时，对验收用的那份基线（本课建议对最终 golden ⑤，即 GPU 生产＋分段累加；本例这道门还没换）</p></details>
  <p>回归失败了，先补跑一次上次通过的那个提交：它也不逐位，变的是环境（镜像、硬件）；它逐位，再在两次提交之间二分。</p>
  <details class="foldfig"><summary><b>为什么标定要绑死版本</b>：PyTorch 官方不保证跨版本可复现</summary>
  <p>PyTorch 自己的文档就写明，不保证跨版本、跨提交、跨平台的结果完全可复现。所以噪声标定和上一次全过的运行都要绑死版本。</p></details>

  <div class="note ok"><span class="t">搬上 TPU 以后改代码，照这几步做</span>
  ① 改完先只编译，比训练步程序的指纹：一样就逐位相同，不用跑（本节上面）。<br>
  ② 指纹变了，跑 300 步对 TPU 的 golden：曲线最差 r 不超过纯噪声的最高分、第 1 步没有离 FP64 更远的张量，就是换了顺序。<br>
  ③ 动到精度或数值路径的，再跑 1000 步、用 GPU 16 份当尺子；记下的每一项 loss 都逐项比。<br>
  ④ 四个已知会动精度的坑：纯数据并行时只切词嵌入表（或词嵌入梯度用 FP32 累加，第八节）；别关 excess precision；RMSNorm 开单次舍入（最后一层输出舍两次那件；v5p、运行库 0.0.52 上不开就超；跟第五节那个 norm scale 舍入不是一回事）；上下文并行目前只能配普通注意力，要让 softmax 真走 FP32（新版代码起，旧代码写了也不生效）。<br>
  ⑤ 本身对加法顺序很敏感的小张量（比如注意力的 sink 参数），1.2 倍的门槛对它太紧，先看只换顺序的几次运行会跳多大；被小参数豁免单列的，要看豁免清单里的倍数。<br>
  ⑥ 换运行库版本：离线筛判不了；先建新的基准运行，跟旧的比，再把上面几件重测一遍（§10.2）。<br>
  ⑦ 换设备数、换切法：先离线编译一份生产规模的训练步，数边传边乘的循环和它们的累加器类型、数被删掉的「转 BF16 再转回」（§5.4、§5.5）；要真舍，写强制舍入。</div>

  <h3>10.4　踩过的坑，和怎么记</h3>
__FIG_JOURNEY__
  <p>归成三条规矩：每个结论先问拿什么比；先排除自己，再怀疑硬件；读回值、工具输出都要实测核对。第二次迁移的坑也都落在这三条里（下方坑表里标了出处的几行）。</p>
  <details class="foldfig"><summary><b>坑表</b>：23 行，每行一个坑、一条规矩</summary>
  <table class="t9"><thead><tr><th>坑</th><th>长什么样</th><th>判据</th></tr></thead><tbody>
  <tr><td>TF32 没关干净</td><td>「FP32」参照比 BF16 还偏，一度判反了 dgrad 的结论</td><td>开关读回值不算数，启动实测</td></tr>
  <tr><td>精确参照自己不精确</td><td>全程 FP32 的精确轨迹，12 万长的求和用 FP32 一次累加，积上千步也偏；拿它当基线，TPU 的判定是两笔相反偏差相减</td><td>参照里每一处长求和单独查：改成分段累加对一遍</td></tr>
  <tr><td>量错了张量</td><td>TPU 那份 FP32 参照，路由专家一段实际是 BF16（据此一度认为全程 FP32 下两边路由也会翻、两边实现不同，后来撤回）；「舍成 BF16 的参照」被编译器删掉了舍入；拿 FP32 主权重当参照，量出一个假的 «FP32_MASTER_FAKE_DIFF_T» 差</td><td>参照先对 FP64 核一遍，再拿来量别人</td></tr>
  <tr><td>钩子改了编译</td><td>TPU 上加了抓数输出的程序和生产程序不逐位</td><td>证据从程序外面拿（§6）</td></tr>
  <tr><td>尺子太短</td><td>只有 10 对时几项落在 «PEND10_T» 倍，补到 10 份两两 45 对全过；只用 3 份 GPU 运行，带符号均值的噪声上限只有一半大；精确轨迹只看一对扰动，散布像是小 «EXACT_PAIR1_X» 倍，凑齐五份再量就不小了</td><td>参照凑到 15 对以上；尺子要量真实的执行噪声</td></tr>
  <tr><td>尺子看不见的差</td><td>第 1 步 GPU 重跑路由一个不翻，尺子对路由翻转宽度是 0；以为 TPU 只改执行参数也不翻，其实会（同一个专家的梯度，TPU 只改片上内存上限差 «STEP1_EXEC_T»，TPU 对 GPU «STEP1_TVG_T»）</td><td>先问这把尺子能不能看见这种差；看不见就对 FP64 定责，别拿被测自己的波动去撑宽尺子。第二次迁移在多步那一级并过一次又撤回：同一批运行只用 GPU 的尺子重判，梯度范数三次比较 r «UNION2_GN_T»，待定全转不过；后来一组改动（大半是修边界，§5.6）把它收回到待定（r «B2_GN_NOW»），现行配置 1000 步对 16 份尺子 0.79，过；没有一处是靠尺子</td></tr>
  <tr><td>测量口径撑大了 r</td><td>TPU 的参数范数自己就测得粗（单步误差占尺子一半以上）；梯度范数的带符号均值以 golden 当分母，交换两边不只差一个符号</td><td>被测和尺子用同样精度的量法；统计量要对两边对称。第二次迁移又碰上一次：TPU 的参数范数整棵树在 FP32 里一路相加，对精确值偏低 «PN2_FP32_BIAS»，正好是要判的偏差那么大；两边改比精确值（每 128 个元素一段 FP32 平方和、FP64 合成）以后，第 1 步只差 «PN2_STEP1_EXACT_T»</td></tr>
  <tr><td>切法也是数值的一部分</td><td>只换梯度累积的切分，词嵌入梯度差到噪声的 «MICRO_SPLIT_EMB_X» 倍；只换专家并行度，反向就不逐位</td><td>golden 和噪声参照固定同一种切分</td></tr>
  <tr><td>参照没载入权重</td><td>转换往返逐位全过，可 GPU 训练入口没设载入路径，按初始化随机建了权重，第一轮 golden 整轮作废（第二次迁移同期）</td><td>训练入口在第 1 步之前逐参数对检查点自检，不过就退出</td></tr>
  <tr><td>参照自己写的代码</td><td>不融合的注意力把 q·k 先存成 BF16 再进 softmax；生产的融合内核和 TPU 都留 FP32（§5.6）</td><td>参照的代码也按五格表查一遍</td></tr>
  <tr><td>名字一样、写法不一样</td><td>优化器、重算、路由偏置更新、梯度累积里的「取最大」四处（§2.3 折叠）</td><td>对照表逐行对写法；参照里找不到实现的，协议写了也不算</td></tr>
  <tr><td>测量工具自己错了</td><td>单算子回放 R «INJ_R_T»：注入漏了按关键字传的输入（§6.2）</td><td>回放先自检：同一输入单独喂一次对 FP64；注入换不了就报错</td></tr>
  <tr><td>算法的前提被新结构打破</td><td>一种按注意力最大 logit 缩放 q、k 权重的稳定化办法（QK-Clip），缩放 η ＝ min(1, τ ÷ 最大 logit)。有一个头把注意力全放在一个额外的空位上，q·k 全为负，最大 logit 为负（GPU 记下 «QKC_SMAX_T»，TPU 同样为负）：τ ＝ 100 时 η ＝ «QKC_ETA_T»，再开根号就是 NaN（第二次迁移；参照框架的写法也有同样的隐患，只是还没碰上）</td><td>每个公式先问前提（这里是「最大 logit 为正」）；不满足时取安全值（η ＝ 1），并加单测</td></tr>
  <tr><td>更精确，不等于更一致</td><td>编译器省掉一次 GPU 必做的舍入，loss 同号偏高（§5.5）；sigmoid 单算子跟 GPU 一样准了，整网反而偏得多（§5.7）</td><td>舍入点照 GPU 写死；「更准」的改动也要整网再验，没查清前不下结论</td></tr>
  <tr><td>在小切片上验精度</td><td>4 个设备上编译器不拆矩阵乘，8 个设备上拆，往里投的投影一次求和多舍 5 次（§5.4，第三次迁移）</td><td>在生产的设备数上编；至少离线编一份数窗口化循环</td></tr>
  <tr><td>裁判工具和生产编得不一样</td><td>单算子回放把训练步里（推断）不拆的几个投影也拆了，量到的多舍在生产里很可能不存在（§5.4）</td><td>用工具前先拿编译产物对一遍写法</td></tr>
  <tr><td>拿第 1 步判权重侧的开关</td><td>输出头精度档、RMSNorm 的 scale 舍入：第 1 步权重还在 BF16 格点上，差小得看不出，要几步到几十步以后才显（§5.5、§5.6，第三次迁移）</td><td>权重侧、优化器侧的开关在曲线上判；或者拿训练几步后的权重单独回放</td></tr>
  <tr><td>「这个开关没效果」只对测它的切法成立</td><td>RMSNorm 的 scale 舍入在 4K 程序里没效果（舍入本来就在），到 32K 程序里被编译器删了，成了漂移主因（§5.5）</td><td>记结论时写明切法和规模；换了就重验</td></tr>
  <tr><td>拿编译产物判断哪里慢</td><td>32K 每步 58 秒，编译产物里只有一个开关多出 120 个循环，就把慢归给了它；逐个实测，慢的是另一个开关（只开它 57.8 秒，只开前者 1.85 秒）</td><td>编译产物看得出改了什么，看不出哪里慢；慢在哪，逐个开关量步时</td></tr>
  <tr><td>参照自己写的层吞掉了掩码</td><td>GPU 参照自写的多路残差层没把填充掩码交给 MoE，有填充的图文行上路由权重的梯度差 25%（第三次迁移；纯文本运行没有填充，不受影响）</td><td>参照里自己写的层，逐个输入对生产的签名</td></tr>
  <tr><td>把单格 r 的升降当成开关效果</td><td>逐个加开关时主 loss、MTP、梯度范数在 0.9–1.2 倍之间来回（参数范数 0.9–1.9），TPU 只改执行方式也是这个量级（第三次迁移）</td><td>判「有效」要另有证据：单算子降下来，或曲线形状变了</td></tr>
  <tr><td>只信配置文件</td><td>权重衰减写 «WD_CONFIG»，实际 «WD_EFFECTIVE»</td><td>对照表写运行时读出的实际值</td></tr>
  <tr><td>把工具和结论混在一起</td><td>通用文档里夹着某个模型的数字，换模型时没人敢删</td><td>工具目录只放方法</td></tr>
  </tbody></table></details>
  <details class="foldfig"><summary><b>第二次迁移：哪些原样带过去省了时间，哪些是新的</b></summary>
  <ul>
    <li><b>原样带过去的</b>：协议在看到结果之前写死，之后每一处改动新旧判定并列；TF32 启动实测；设备 d 对 GPU 第 d 张卡、原样回放数据；每个统计取各对的最大值；逐张量 sha256 清单当地基锚点。第一天就都有了。</li>
    <li><b>新的</b>：每一级的尺子都只来自参照，并集撤回（§4.3）；TPU 同设置重跑逐位相同，也不改尺子的取法（它的波动只作诊断）；「FP32」在 TPU 默认精度下只算一遍 BF16，HIGHEST 也只是近似（§5.6）；稀疏注意力的 top-k 翻得多却几乎不影响输出（§4.3）；f32 小标量会病态（§5.6 折叠）；回放工具要覆盖关键字参数（§6.2）；参照一侧自己写的代码也要按五格表查；先拆测量误差再谈数值差。</li>
    <li><b>流程上的两条</b>：定参照之前先把训练配置逐项对齐到生产，查不到写「未知」，别替它选一个常见默认 —— 第二次迁移一开始没查到生产用的优化器，参照选了 AdamW，那一批 golden 全部作废；作业要能无人值守，下载、校验、训练、上传都写进启动脚本。</li>
  </ul></details>
  <details class="foldfig"><summary><b>怎么记</b>：七条，让下一个人从仓库里看出查到哪了</summary>
  <ul>
    <li><b>一个问题一份案例文件</b>：标题就是发现，开头写状态、数据从哪来、复现命令；事实带出处，推断标「推断」。</li>
    <li><b>假设表按成本排，排除了的也写</b>：省得后人再查一遍。</li>
    <li><b>一次只改一件事，新结果不覆盖旧的</b>；结论只写一处，错了追加更正，不悄悄改。</li>
    <li><b>工具和结果分开放</b>：工具的文档只讲方法，某个模型的结论和坑写在它自己的目录里。</li>
    <li><b>结果带上检查清单的指纹，只算检查内容</b>：清单里的检查一改，这份清单下的旧结果自动作废；改注释、改说明文字不改指纹，不用为了改文档重跑。</li>
    <li><b>交给别人看的文档按发布标准写</b>：份数和对数分开说（「10 份重跑，两两 45 对」，不写「45 组测试」）；每个代号就地加括号说全称；不留「待验证」挂着 —— 要么验掉，要么写清楚还缺什么。</li>
    <li><b>排查和讲课两条线互相喂</b>：讲课时说不清的地方，往往就是排查里没坐实的地方；排查的新结果当天回到课件，课件的追问再变成下一个实验。</li>
  </ul></details>

  <h3>10.5　回到开头</h3>
  <p><b>TPU 没算错。</b>TPU 的 loss 低约 1.7e-4，按噪声尺子是真偏差；查到最后，大头偏在 GPU 的 golden：<b>一次 12 万长的求和，每一步都朝零少算了一点</b>。改成分段累加、拿它当基线，v5p、v7x 每一项都在噪声内（最差 «FINAL_WORST_T» 倍）。第二个案例里，两种数学等价的并行对不上，查到最后是一次加法的组织方式。</p>
  <div class="note"><span class="t">这门课带走的三句话</span>
  <b>一、拿 GPU 自己的重跑当尺子</b>：TPU 跟 GPU 的差，不能比 GPU 自己跟自己的差大；超了，就查到底是谁的差，当参照的那一份也要查。<br>
  <b>二、差只有三种</b>：换了加法顺序是噪声，不修；丢了精度要对 FP64 定责，找到多舍的那一步；公式写得不一样必须对齐。<br>
  <b>三、换了条件就要重验</b>：设备数、切法、步数、芯片、运行库，换一样，之前的验证就不作数；精度验证在生产的设备数上做，改完代码先离线编译看一眼。</div>

  <h3>10.6　拿回去用：清单、速查卡、排查表、练习</h3>
  <p>这一节留给要在自己的模型上做一次对齐的人。前四个折叠按用的顺序排；后面是延伸阅读和出处台账。</p>

  <details class="foldfig"><summary><b>动手清单</b>：在自己的模型上做一次 GPU→TPU 对齐，一步一步来</summary>
  <ol>
    <li><b>定目标和预算</b>：写下要对齐的配置（层数、步数、批大小、学习率、精度），和「对谁比」—— 默认对 GPU 那份 golden。参照的优化器、loss 写法逐项对到生产，查不到写「未知」。协议开跑前写死这几项：基线是哪一份；检查项和各自的窗口；三个统计；用哪组运行当尺子、几对；门槛 1 和 2；参考档并排报；例外怎么处理。开跑前埋好记录点：每步的实际学习率、实际权重衰减、裁剪前梯度范数、裁剪系数；主权重和两个动量的快照节奏；探针序列；数据单元的指纹字段。设备数要跟 GPU 卡数对上（本例 v5p 8 颗芯片、v7x 4 颗芯片，都是 8 个设备）；对不上时，切法差会算进 TPU 一侧，留一次切法消融（本例改成按专家切，变化在 TPU 执行噪声以内）。截短模型（本例 4 层）能让每次对比便宜很多；签字前 GPU 侧要跑三四十次 1000 步（本例：默认 10 份、② 5 份、③ 5 份、两条精确轨迹连扰动版 7 份，最后修好的 golden ⑤ 又 10 份）。</li>
    <li><b>先量 GPU 自己的噪声</b>：同一设置的 GPU 至少跑 6–8 份（两两 15 对以上）；只用同一种设置的运行配对（确定性开关、分布式优化器开不开，都算设置）。算出每个检查项的 N。默认检查项：loss（两个窗口）、梯度范数、参数范数、学习率；训练时顺手记这几个便宜的量，出事时它们就是线索。</li>
    <li><b>铺地板 L0</b>：数据原样回放、逐单元比指纹；权重转过去再转回来、逐张量比字节；加载后从设备上再读一遍，参照一侧的训练入口也在第 1 步前对检查点自检；配置对照表写<em>实际生效</em>的值（运行时元数据），优化器、重算、统计量的聚合逐行对写法；续训逐位。</li>
    <li><b>第 1 步自检</b>：看第 1 步 loss 的相对差。本例是 «D1_T»；超过参考档的 «REF_LOSS_GATE»，先回清单第 3 步查地基，别查数值；低于它，直接做下一步的张量比较、用 FP64 的 R 比定责（本课经验）。本例的 «D1_T» 只当量级参考，不是下限。</li>
    <li><b>第 1 步张量</b>：先给每个算子两边写五格 dtype 路径（§5.6），有差的格子先对上；再逐个大张量算 rel_L2、cos、范数比；超线（GPU 重跑的差不是 0 就比它；GPU 重跑一位不差，前向常见，就直接挑差最大的几处；参考档只并排报）的那一处请 FP64 来算 R 比定责（≤ «R_GATE» 算两边一样远）。MoE 先看路由槽位翻了没有，翻了就按「两边选中同一批 token」重比。</li>
    <li><b>跑满步数</b>：两个窗口（全程；尾段 ＝ 两条轨迹分开、偏移稳定以后，本例从第 301 步起；窗口事先写进协议；学习率有调度时，尾段要避开预热）各算带符号均值、MARE、最大单步，除以 N 得 r；≤ 1 过、1–2 待定、＞ 2 不过。待定先补参照重判（补几份、补到哪停，事先写进协议；混元 3 当时没写，是看到待定才补的）。检查项多时，纯噪声也会出待定：至少「条数 × 1/(n+1)」条，带符号均值约翻倍；以留一法实测为准。</li>
    <li><b>不过就往下排查</b>：从上往下一级级缩小（见下面的排查表）；TPU 上拿证据走程序外面（步间读状态、隔离回放、固定探针、状态互换）。</li>
    <li><b>golden 自己可疑时</b>：在 GPU 上跑一条精确轨迹（FP32 主权重照常更新、每步舍成 BF16 再用；前向反向的算术全部 FP32、真关 TF32；像输出头 dgrad 那样的长求和要分段累加，FP32 一次累加也会偏；核长求和要拿训练中途的状态、按实际长度、看关键方向上带符号的偏差，第 1 步对 FP64 核一遍抓不到它；§7.5），分两关：第一关对它判过不过；第二关对生产 golden 写成「已解释差异」。什么时候算可疑：排查表里「TPU 更低、更近 FP64」那一行。参照先对 FP64 核一遍再用（例如真 FP32 的矩阵乘对 FP64 应在 «TF32_PROBE_OK» 以内，§5.3）。尺子怎么选没有通用规则（本例定案前借 ③ 那组重跑，定案时重新写了协议，换成 ⑤（最终 golden）自己 10 份两两 45 对，旧判定留着并写明作废原因），关键是写进协议、不在判定中途悄悄换。代价：全程 FP32 的矩阵乘比 BF16 慢一到两个数量级（看硬件，以自己卡上实测为准），精确轨迹还得换成不融合的注意力。</li>
    <li><b>投产后</b>：日常回归先要求逐位；不逐位按 PR 的数值声明分流；TPU 执行噪声用只改执行方式的扰动臂标定，配置一换就重标。扰动臂的常见候选：编译器片上内存上限、注意力分块大小、重算策略（路由结果已存下时）、层是否用 scan；两条筛选：臂对基线必须不逐位（逐位的臂贡献 0、却虚增对数），会系统性增减 BF16 舍入次数的改动（关掉边传边乘、关 excess precision、改累加精度）不能当臂 —— 它们带来的是偏差，不是噪声。选进来的臂还要过三项检查：臂自己重跑一位不差；配置、版本、编译出的程序除了改的那一处都不变；第三项分两半，第 1 步梯度那一半比 FP64：臂离 FP64 不比基线远（按能量 R ≤ 1.2）， 只改执行方式的臂也会翻近平局的路由，GPU 噪声那把尺子看不见；1000 步 loss 那一半对基线不超过 GPU 自己的噪声（本例当前配置补做了 «ARMS_FP64_N» 个臂，R «ARMS_FP64_R_T»，全过；参照里另两个改片上内存上限的臂没有单独补）。这样标出来的 TPU 噪声带里含着路由翻转。本例旧配置试了 14 个臂，过了 7 个。</li>
    <li><b>记下来</b>：一个问题一份案例文件，事实带出处、推断标出来、排除的也写；坑写进通用文档。</li>
  </ol>
  <p>判定那一步的骨架（伪代码，本课推导）：</p>
  <pre style="font-size:13px;line-height:1.5;background:#f8f9fa;padding:10px;border-radius:6px;overflow-x:auto">STATS = ["signed_mean", "mare", "max_step"]
CHECKS = {"loss": ["all", "tail"], "grad_norm": ["all"], "param_norm": ["all"]}   # 本例的检查项和各自的窗口
WINDOWS = {"all": (2, T), "tail": (t0, T)}     # 训练步号，含两端；t0：事先写定的尾段起点
def d(test, base):            # 每一步的相对差，d[0] 是第 1 步
    return [(t - b) / abs(b) for t, b in zip(test, base)]
def sel(dt, w):
    return [dt[t - 1] for t in range(w[0], w[1] + 1)]
def stats(x):                 # 一个窗口里的三个统计
    return {"signed_mean": abs(mean(x)), "mare": mean(abs(v) for v in x), "max_step": max(abs(v) for v in x)}
verdict = {}
for c, wins in CHECKS.items():                  # 每个检查项、每个窗口各出一个判定
    for wn in wins:
        w = WINDOWS[wn]
        N = {k: max(stats(sel(d(a[c], b[c]), w))[k] for a, b in pairs(gpu_runs)) for k in STATS}
        r = {k: stats(sel(d(tpu[c], base[c]), w))[k] / N[k] for k in STATS}   # N 为 0 时不能除，见排查表
        worst = max(r.values())                 # 三个统计里最差的决定这一项
        verdict[(c, wn)] = "过" if worst &lt;= 1 else ("待定" if worst &lt;= 2 else "不过")
# 单算子定责：R = rel_l2(tpu, fp64) / rel_l2(gpu, fp64)；≤ «R_GATE» 算两边一样远</pre>
  <p>本例的尺子最后补到 10 份两两 45 对；只有 10 对时有两项落在待定区（§9.4）。</p></details>

  <details class="foldfig"><summary><b>速查卡</b>：指标、门槛、读法，一页</summary>
  <table class="t9"><thead><tr><th>量</th><th>怎么算</th><th>门槛 ／ 读法</th></tr></thead><tbody>
  <tr><td>dₜ</td><td>（被测 − 基线）÷ |基线|，同一步</td><td>只是原料；首步单独报</td></tr>
  <tr><td>带符号均值 ／ MARE ／ 最大单步</td><td>窗口内 dₜ 的平均 ／ |dₜ| 的平均 ／ 最大 |dₜ|</td><td>窗口：全程从第 2 步起；尾段从两条轨迹分开、偏移稳定以后起（本例第 301–1000 步）；三个都判，最差的决定</td></tr>
  <tr><td>N（噪声上限）</td><td>同设置 GPU 两两配对，同口径算，取绝对值的最大</td><td>纯噪声超过 n 对最大值的概率约 1/(n+1) 起（带符号均值约翻倍，以留一法实测为准）；正式签字 ≥ 15 对；1000 步这样的长窗口用 16 份、120 对（份数少会低估噪声）</td></tr>
  <tr><td>r</td><td>|被测| ÷ N</td><td>≤ 1 过；1–2 待定；＞ 2 不过</td></tr>
  <tr><td>哪一级用哪把尺子</td><td>L0 地基；L1/L2 单算子、单层；L3/L4 整网第 1 步、多步</td><td>L0 逐位；L1/L2 前向（GPU 重跑噪声为 0）用 R 比定责；L2 逐层反向也用 R 比（本例 GPU 重跑反向逐位、量不到路由翻转）；L3 整网第 1 步对 GPU 第 1 步重跑的噪声，超出的、尺子看不见的（路由翻转、舍入位置不同）对 FP64 定责 —— 尺子只来自参照（混元 3 曾并入 TPU 扰动，后来撤回，§4.3）；L4 多步用 GPU 噪声尺子</td></tr>
  <tr><td>rel_L2</td><td>‖被测 − 参照‖ ÷ ‖参照‖（整个张量拉直）</td><td>张量主判；长度相同时 cos 0.999 ≈ rel_L2 «COS_0999_RELL2_T»</td></tr>
  <tr><td>cos ／ 范数比</td><td>方向 ／ 长度之比</td><td>cos 看不见整体缩放；范数比 2.000 ＝ 漏除了一个 2</td></tr>
  <tr><td>R（对真值的距离比）</td><td>TPU 到 FP64 的 rel_L2 ÷ GPU 到 FP64 的 rel_L2</td><td>≈ 1 一样远；＞ «R_GATE» 查 TPU；≈ «R_ONE_EXTRA_T» 像多舍了一次；＜ 1 GPU 更远</td></tr>
  <tr><td>ULP</td><td>按原始格式数差了几格</td><td>只诊断，不当门槛</td></tr>
  <tr><td>L0</td><td>数据、权重、配置、续训</td><td>必须逐位</td></tr>
  <tr><td>参考档</td><td>别人上线用过的固定线</td><td>首步与窗口 loss ≤ «REF_LOSS_GATE»、逐层 rel_L2 ＜ «REF_RELL2_GATE»（没有路由的层）；只并排报</td></tr>
  </tbody></table></details>

  <details class="foldfig"><summary><b>排查表</b>：看到什么 → 多半是什么 → 先查什么</summary>
  <table class="t9"><thead><tr><th>症状</th><th>多半是</th><th>先查</th></tr></thead><tbody>
  <tr><td>第 1 步 loss 相对差超过 «REF_LOSS_GATE»</td><td>地基没铺平：数据、权重、配置</td><td>数据指纹、权重逐字节、配置的实际生效值</td></tr>
  <tr><td>第 1 步对上，第 2 步起的差明显大于 GPU 重跑第 2 步的差</td><td>FP32 主权重有小于 BF16 半格的错</td><td>加载后从设备读回、跟 GPU 第 0 步逐位比主权重</td></tr>
  <tr><td>差从第 1 步起就稳定在一个常数</td><td>口径：loss 分母、步号错位</td><td>故意错开一步再比；核对 loss 归一化</td></tr>
  <tr><td>梯度 cos ≈ 1、范数比是整数倍</td><td>漏除了微批数或卡数</td><td>梯度累积和规约里的除法</td></tr>
  <tr><td>某个专家梯度大得离谱，GPU 重跑却一位不差</td><td>路由近平局翻转</td><td>路由槽位；给路由器喂相同输入再比</td></tr>
  <tr><td>单层 R ＞ «R_GATE»</td><td>TPU 多舍了一次（例如 FSDP 下部分积在 BF16 里累加）</td><td>累加器精度、collective matmul、excess precision</td></tr>
  <tr><td>名义 FP32 的参照比 BF16 还偏</td><td>TF32 没关干净</td><td>启动时实测 FP32 矩阵乘对 FP64；打印环境变量</td></tr>
  <tr><td>加了抓数的程序和生产程序不逐位</td><td>钩子切开了融合</td><td>改用步间读状态、隔离回放、探针</td></tr>
  <tr><td>第 1 步逐张量梯度普遍是 GPU 噪声的 2–3 倍，没有哪一处突出</td><td>好几处舍入位置跟 GPU 不同</td><td>五格表逐个算子对（§5.6）；超线的对 FP64 定责</td></tr>
  <tr><td>一个小标量一直超线，它的输入和梯度都没事</td><td>f32 里的病态在放大一格</td><td>用大数的格量差；同样的计算换 CPU（PyTorch）算，看那一格是谁的（§5.6 折叠）</td></tr>
  <tr><td>含 exp、log、sigmoid 的量，TPU 系统偏一边</td><td>TPU 默认的超越函数精度低、而且有偏</td><td>逐函数对 FP64 数格；换高精度写法后整网再验（§5.7）</td></tr>
  <tr><td>长跑 loss 次次同号偏一点，只改执行参数也一样</td><td>编译器省掉了一次 GPU 一定会做的舍入</td><td>在那一处显式「舍到 BF16」做 A/B（§5.5）</td></tr>
  <tr><td>jit 下结果错得远超舍入量级（比如相对误差 0.47），一步一步执行却是对的</td><td>编译器 bug，不是精度问题（对齐记录里的实例：一次多输出融合在 v5p 上把一个输出算丢了，运行库 0.0.46–0.0.52 都有，上游 11-06 起的 nightly 不再这样合并）</td><td>先关掉 jit 一步一步执行，跟 jit 比；再把每道编译工序之后的程序导出来、在 CPU 上执行（那一次每一步都对，说明错在程序翻成 TPU 指令那一步），再比相邻工序找出是哪次融合；绕开的办法是关掉那种融合，或在出事的读取前加 <code>optimization_barrier</code></td></tr>
  <tr><td>已经定位到一处「编译器省掉的舍入」，想知道同类的还有几处</td><td>代码里写了 FP32 → BF16 → FP32 往返、编译器在融合里省掉的地方不止一处</td><td>离线编译整个训练步，扫同一颗融合里的往返 convert，只含往返的 kernel 数 LLO 打包指令；每类对照 GPU 代码，再上卡验证（§9.7）</td></tr>
  <tr><td>训练突然出 NaN，只在某一层、某个开关打开时</td><td>公式的前提被打破（负数开根号、除以零）</td><td>逐层定位；查公式的前提（§10.4）</td></tr>
  <tr><td>单算子回放 R 大，同一输入单独喂这个算子却没事</td><td>回放没把输入全换掉，算子吃的是本侧上游的输入</td><td>注入覆盖关键字参数；换不了就报错（§6.2）</td></tr>
  <tr><td>loss 慢慢偏、参数范数一起漂</td><td>每一步更新里有一个固定方向的小差</td><td>按组、按行拆参数范数；看 Adam 的 ε 区；状态互换</td></tr>
  <tr><td>TPU 的 loss 更低、参数范数更大，单步上 TPU 反而离 FP64 更近</td><td>golden 自己偏了，比如大词表 dgrad 的长求和朝零少算</td><td>FP32 dgrad 的 A/B；状态互换；在 GPU 上跑精确轨迹（长求和分段累加）</td></tr>
  <tr><td>TPU 对精确轨迹还差一截，可同状态单步看不出</td><td>精确参照里的长求和自己有偏，或 TPU 某处多舍了一次，两笔可能相抵</td><td>把参照的长求和改成分段累加再比；TPU 一侧从精确一处一处降回生产精度</td></tr>
  <tr><td>GPU 重跑逐位相同，N ＝ 0</td><td>开着确定性设置；或这一级（第 1 步前向、单算子）本来就没有噪声</td><td>整网多步改用生产配置的重跑当尺子；单算子、单层用 FP64 的 R 比</td></tr>
  <tr><td>TPU 自己重跑也不逐位</td><td>环境变了（镜像、依赖、硬件），或程序里有非确定的算子</td><td>补跑上次通过的提交；对比依赖版本</td></tr>
  <tr><td>只有最大单步超线</td><td>某一步突然跳了一下</td><td>看那一步的数据单元和梯度范数</td></tr>
  <tr><td>第一关判待定</td><td>参照太少，或差就在噪声边上</td><td>补参照到 15 对以上重判；还在 1–2 倍就往下一级定责</td></tr>
  <tr><td>严格档不过、参考档过</td><td>差已超出噪声，但不大</td><td>定责：是谁的差，再决定写成已解释差异还是修</td></tr>
  <tr><td>纯数据并行和 FSDP 对不上，第 1 步只有词嵌入梯度远</td><td>高频词在本卡 BF16 里一次次地加、后端没排序（每卡一次的下标数没过表行数 1/8）</td><td>首选只把词嵌入表沿数据并行轴切开，过不了门槛再改 FP32 累加；按行分组看高频行；看编译产物里那次 scatter 有没有排序（门槛随运行库变，0.0.46 上是行数 ÷ 8；§8.6、§8.9）</td></tr>
  <tr><td>不同配置之间偏移不一致</td><td>并行切法、重算、批大小切法也是数值的一部分</td><td>参照和被测固定同一切法、同一设置</td></tr>
  </tbody></table></details>

  <details class="foldfig"><summary><b>练习</b>：七道题，答案在每题下面</summary>
  <ol>
    <li>GPU 重跑 4 份，两两 6 对，参数范数带符号均值的绝对值最大是 2.0e-7；TPU 对基线是 3.0e-7。怎么判？还缺什么？
      <br><em>答：r ＝ 1.5，待定。只有 6 对，纯噪声超线的概率至少 1/7（各对互不相干时；实际共用运行，带符号均值约 2/7）；再补到 15 对以上重判，还在 1–2 倍就往下一级定责。</em></li>
    <li>第 1 步 loss 的相对差是 3e-4。该先查数值还是先查地基？
      <br><em>答：本例第 1 步只差 «D1_T»，3e-4 大了一百多倍，也超过参考档的 «REF_LOSS_GATE»（§4.4），先回 L0：数据指纹、权重逐字节、配置的实际生效值。第 1 步还没有任何更新，两边只差前向舍入。</em></li>
    <li>某个注意力投影的 R ＝ 1.40。说明什么？下一步查哪？
      <br><em>答：像 TPU 多舍了一次同量级的数（√2 ≈ «R_ONE_EXTRA_T»）。查这处矩阵乘的累加器精度、是否是 collective matmul 在 BF16 里累加部分积、有没有 excess precision 改了舍入位置。</em></li>
    <li>你的「FP32 参照」对 FP64 的 rel_L2 是 3e-4，GPU 的 BF16 程序是 6.5e-4。能拿它当参照吗？
      <br><em>答：不能。§5.3 的判据是真 FP32 对 FP64 ≤ «TF32_PROBE_OK»；3e-4 说明它多半走了 TF32。先真关 TF32（环境变量），启动时实测一遍再用。</em></li>
    <li>定案前，迁移验收第一关 «ACC_PN_R» 倍不过，loss 待定。经理问能不能上线，你怎么答？（再想想：定案以后怎么答。）
      <br><em>答：先说协议结论：第一关不过，不能签字。定案以后：第一关全过（最差 «FINAL_WORST_T» 倍），对 GPU 生产 golden 的差全部写成已解释差异（GPU 输出头 dgrad 的长求和），剩下几格各有机制；下游评测没测，带条件上线还缺下游评测和更长的运行。一般地给四行：已解释的部分（几成是 GPU golden 自己的偏差）；没解释的部分和下一步；下游没测；要带条件上线，还缺哪些证据、止损线设在哪（样子比如：上线后保留 GPU 并跑，loss 尾段 r 超过 2 或下游评测掉过线就回退；具体数按业务定）。能不能上线是业务判断，拿这四行去定。</em></li>
    <li>换一个词表 25 万的模型，输出层 dgrad 那次长求和的朝零偏差，会比 12 万时更大还是更小？
      <br><em>答：多半更大，两头都放大：求和越长，截掉的总量越多（抹零每一单只会让绝对值变小）；词表越大，落进 Adam ε 区的冷门行也越多（推断）。§7.4 真实输出头上，K 从 «HEAD_K1» 到 «HEAD_K3»，偏差从 «HEAD_BIAS1» 涨到 «HEAD_BIAS3»，比线性还快；外推到 25 万是推断，没测过；合成输入在 12 万处偏差反而降，可能是库换了算法，所以「更大」不保证。要么在 GPU 上把这一次 dgrad 改成分段累加（升 FP32 不分段也偏，§7.5），要么至少在第一关前，拿训练中途的状态先测它沿关键方向对 FP64 的偏差。</em></li>
    <li>一项辅助 loss 每行约 1e-4，从约 7 的 log-sum-exp 里减出来。TPU 记下的比 GPU 低 1e-3（相对），GPU 重跑几乎不散，判「不过」。先查什么？
      <br><em>答：先问病态的前三问（§5.6 折叠）：它的输入两边离 FP64 一样近吗？由它出发的梯度一样近吗？f32 在 7 附近一格约 4.8e-7，约是 1e-4 的 5e-3 倍 —— 差折成大数的格在一两格以内吗？都是，说明病态在放大；再问第四问：同样的计算换 CPU 上算偏不偏。CPU 不偏、TPU 偏，那一格是 TPU 的 exp、log，换高精度写法；CPU 也偏同样多，才写「不过，已定责」。</em></li>
  </ol></details>

  <details class="foldfig"><summary><b>延伸阅读</b>：想把这一讲的几个底层问题读透，从这几篇开始</summary>
  <ul>
    <li>浮点加法为什么换顺序就变、前向为什么其实是确定的：Thinking Machines，<em>Defeating Nondeterminism in LLM Inference</em>（2025）。</li>
    <li>张量核里的累加到底怎么舍：Fasi 等，<em>Numerical behavior of NVIDIA tensor cores</em>（PeerJ Computer Science，2021）；Khattak、Mikaitis，<em>Accurate Models of NVIDIA Tensor Cores</em>（arXiv:2512.07004）。</li>
    <li>低精度累加在大模型训练里的代价与对策：DeepSeek-V3 技术报告 sec. 3.3.2（arXiv:2412.19437）。</li>
    <li>logits 漂移、ε 与训练不稳定：Wortsman 等，<em>Small-scale proxies for large-scale Transformer training instabilities</em>（arXiv:2309.14322）。</li>
    <li>MoE 路由两套引擎选得不一样：Ma 等，<em>Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers</em>（arXiv:2510.11370）。</li>
    <li>一次看很多个数、总有一个「超线」：xkcd 882《Significant》（软糖漫画）；粒子物理的 look-elsewhere effect，Bayer、Seljak，<em>The look-elsewhere effect from a unified Bayesian and frequentist perspective</em>（arXiv:2007.13821，修正系数通常靠模拟来算）。</li>
    <li>拿噪声当尺子：Microsoft ExP 团队关于 A/A 测试的博客；Henderson 等，<em>Deep Reinforcement Learning that Matters</em>（arXiv:1709.06560）。</li>
  </ul></details>

  <details class="foldfig"><summary><b>出处台账</b>：每个结论来自哪一次测量（混元 3 截 4 层；除标「本课推导」外都是实测，步数、配置、单算子还是整网，见右栏）</summary>
  <table class="t9"><thead><tr><th>结论</th><th>材料与测量条件</th></tr></thead><tbody>
  <tr><td>BF16 里 1 ＋ 2⁻⁸ ＋ 2⁻⁸ 换个顺序差一格</td><td>本课推导，图脚本里按就近舍入、平局取偶真算</td></tr>
  <tr><td>10 次 GPU 重跑：尾段 loss 带符号均值 «GPU_RUNS10_MEAN_T» ± «GPU_RUNS10_STD»；TPU «TPU_ARM_LO_T» 到 «TPU_ARM_HI_T»</td><td>B200×8 默认设置、关分布式优化器，10 份；TPU v5p（8 颗芯片） 三种跑法，各 1000 步，对第一份 GPU</td></tr>
  <tr><td>数据 «DATA_UNITS» 个单元逐位；原生加载器 «NATIVE_SAME_RANK_T»／«DATA_UNITS» 同号</td><td>每个单元四个字段各一个 sha256，另比每步非 padding token 数</td></tr>
  <tr><td>«N_TENSORS_T» 个张量逐位、参数 «N_PARAMS»</td><td>开源权重 → TPU 格式 → 转回，按原始字节比</td></tr>
  <tr><td>权重衰减 «WD_CONFIG» → «WD_EFFECTIVE»</td><td>GPU 运行元数据里的调度器配置</td></tr>
  <tr><td>第 1 步 loss «D1_PLUS_T»</td><td>«LOSS1_TPU_T» 对 «LOSS1_GPU_T»</td></tr>
  <tr><td>TPU 对 GPU 尾段带符号均值 «TPU_LOSS_TAIL_T»，逐步标准差 «TPU_TAIL_STEP_STD»，九成的步更低</td><td>逐步原值，TPU 第一次 1000 步运行对 GPU 第一份</td></tr>
  <tr><td>定案前第一关验收 r（对 ④ 精确轨迹）：参数范数 «ACC_PN_R»、loss «ACC_LOSS_ALL_R» ／ «ACC_LOSS_TAIL_R»、梯度范数 «ACC_GN_R»；v7x（开关打开前的配置）参数范数 «ACC_V7X_PN»</td><td>v5p 当前配置 1000 步对精确轨迹一份；N 取 ③ 那组 GPU 重跑 5 份两两 10 对</td></tr>
  <tr><td>参照 «INCONCL_PAIRS_T» 对：待定 «INCONCL_T»</td><td>第一阶段 100 步的 GPU 对 GPU 演练（46 条检查），被测和参照都是默认设置；15 对时只重判了那 8 条外加梯度范数一条</td></tr>
  <tr><td>qkv 投影 R «R_QKV_BEFORE» → «R_QKV_AFTER_T»；对 GPU rel_L2 «QKV_VS_GPU_BEFORE» → «QKV_VS_GPU_AFTER»</td><td>v5p 单算子对 FP64、逐层回放；打开 FP32 累加的开关前后</td></tr>
  <tr><td>第三次迁移：往里投的投影 R «M3_PROJ_R_T» → «M3_PROJ_R_AFTER_T»；BF16 累加的窗口化循环 4 ／ 8 ／ 16 个设备 0 ／ 20 ／ 20；第 1 步超 2N 91 → 63；1000 步主 loss 1.14 → 0.96，步时 ＋«M3_COST_T»</td><td>第三次迁移：单算子回放（8 个设备，对 FP64）；同一份生产配置在 CPU 上为各拓扑离线编译训练步（运行库 0.0.46）；整网第 1 步与 1000 步各一份运行，阶梯上相邻两级</td></tr>
  <tr><td>GPU 自比第 1 步梯度：差 1 格 «ULP1_T»，≥17 格 «ULP17_T»</td><td>第一阶段，两次默认设置 GPU 运行，按 FP32 数格（梯度存 FP32）</td></tr>
  <tr><td>张量核模型：«TC_GROUP» 个一组、«TC_KEEP» 位、向零截断</td><td>B200 上逐位复现路由打分三层全部元素；试过的三种改法（8 个一组、24 位、就近舍入）命中率都 ＜1%</td></tr>
  <tr><td>合成输入朝零偏差 «TC_BIAS1» ／ «TC_BIAS2_T» ／ «TC_BIAS3»（K ＝ «TC_K1» ／ «TC_K2» ／ «TC_K3_T»）</td><td>B200 BF16 张量核，正数输入；负数输入同样朝零（K ＝ «TC_K1» 时幅度约为正数的一半）；FP32 非张量核 ≤ «FP32_BIAS_MAX»</td></tr>
  <tr><td>TF32：只关 torch 开关 «TF32_TE_ON»，设环境变量 «TF32_ALL_OFF»，读回仍 True</td><td>官方训练容器里实测 FP32 GEMM 对 FP64</td></tr>
  <tr><td>路由打分对 FP64：TPU «ROUTER_TPU_FP64» – «ROUTER_TPU_FP64_HI»，GPU «ROUTER_GPU_FP64» – «ROUTER_GPU_FP64_HI_T»（三层）</td><td>第 1 步单算子回放，输入用 GPU 那一层的路由输入</td></tr>
  <tr><td>反推梯度对直接抓的梯度 rel_L2 «MOMENTUM_INV_LO» – «MOMENTUM_INV_HI»</td><td>GPU 上，相邻两步一阶动量倒推</td></tr>
  <tr><td>参数范数多涨 «PN_TPU_EXTRA»（GPU 自己涨 «PN_GPU_GROWTH»）</td><td>逐步原值；GPU 两次之间最多 «PN_GPU_PAIR_MAX»</td></tr>
  <tr><td>第 300 步：输出层贡献 «GROUP_HEAD_300» ／ «GROUP_TOTAL_300»；没出现过的词 ＋«ROWS300_NEVER»</td><td>FP32 主权重的分组、逐行平方和，词频按前 300 步数据统计</td></tr>
  <tr><td>冷门行处在 ε 区：第 300 步 «RARE_EPS_FRAC_T» 的元素 √v̂ ＜ ε（中位数 «RARE_VHAT_MED»）；ε 区的弹性</td><td>第 300 步两侧输出层状态实测；弹性为本课推导（稳态近似）</td></tr>
  <tr><td>lse 差 «LSE25_T» ／ «LSE50_T» ／ «LSE100_T» ／ «LSE300_T»；第 1000 步约 «ORIG_LSE_T»</td><td>固定序列探针第 25 ／ 50 ／ 100 ／ 300 步（对第一份 GPU）；第 1000 步那个数取自训练批</td></tr>
  <tr><td>干预：去掉方向后 loss «REMOVE_LOSS_T»、参数范数 «REMOVE_PN_T»；GPU 加力 «FORCE_LOSS_T»、«FORCE_PN_T»</td><td>两边各 1000 步，同设置</td></tr>
  <tr><td>输出头反向误差：GPU dgrad «HEAD_BF16_S1_T»，TPU «HEAD_TPU_S1_T»，交叉熵 «CE_BWD_S1_T»，TF32 版 «HEAD_TF32_S1_T»，真 FP32 «HEAD_EXACT_S1_T»</td><td>第 1 步，用训练里实际用到的张量离线对 FP64；第 100 步 GPU «HEAD_BF16_S100_T»、TPU «HEAD_TPU_S100_T»。机制是张量核抹零为推断</td></tr>
  <tr><td>同状态各算一步：TPU «SWAP_TPU_T»，GPU FP32 dgrad «SWAP_FIX_T»，GPU «SWAP_GPU_T»</td><td>GPU 第 99 步状态导入 TPU（«SWAP_STATE_TENSORS» 个张量核对指纹；按 GPU 框架布局数，q/k/v、gate/up 各合成一个，所以比 §2.2 按开源命名数的 «N_TENSORS_T» 少，元素一个不少），第 100 步数据，共同参照为 GPU 全程 FP32；4 个状态 12 个 batch 范围见图</td></tr>
  <tr><td>精确轨迹：参数范数 «EXACT_TRAJ_PN975_T»（TPU «EXACT_TPU_PN975_T»，FP32 dgrad «DGRADX_LO_T» – «DGRADX_HI_T»）</td><td>第 975 步，对 4 份默认 GPU 运行均值；GPU 自己 ±«EXACT_REF»</td></tr>
  <tr><td>旧版拆账 «DECOMP_HEAD_T» ／ «DECOMP_REST_T» ／ «DECOMP_TPU_T»（作废：把 ④ 当成了精确值）；loss «LOSS_FIXHEAD_SHARE_T»</td><td>TPU（开关打开前的配置）、生产 golden（5 份）、只修 dgrad 的 golden、精确轨迹，第 1000 步与第 301–1000 步</td></tr>
  <tr><td>新的账：④ «PN_FIX1_T»、⑤ «PN_PRODCHUNK_T»、TPU 生产 «PN_TPU_PROD_T»、⑥ «PN_TRUE_T»；生产 → ⑤ «LADDER_DGRAD_T»、⑤ → ⑥ «LADDER_SHARED_T»</td><td>第 1000 步参数范数，对生产 golden 5 份均值；生产近似配置；TPU v5p、v7x 当前配置同值</td></tr>
  <tr><td>TPU 对 ④ «TPU_VS_FIX1_T» ＝ «B_GPUEXACT_T» «B_DLOGITS_T» «B_REST_T»（v7x；v5p ＋7.97e-3、−6.10e-3、＋4.9e-4）；TPU 生产比 TPU 精确轨迹少漂 «TPU_VS_TRUE_T»（对 GPU 真精确 ⑥ 是 6.5e-3）；GPU 真精确多舍一次 dlogits «SHARED_DLOGITS_GPU_T»</td><td>第 1000 步；第一笔由 GPU 精确改分段累加证实，第二笔由 TPU 精确只把 dlogits 舍 BF16 证实，第三笔是差额</td></tr>
  <tr><td>cuBLAS FP32 一次累加的偏差：第 1 步 «CUBLAS_S1_T»（第 299 步对照用的这一组张量；正文和 §7.4、§7.5 的表用生产张量，是 «HEAD_EXACT_S1_T»），第 299 步状态 «CUBLAS_299_T»；改分段累加后 GPU 精确对 TPU 精确 «CHUNK_VS_TPUEXACT_T»；漂移比 «DRIFT_RATIO_TRUE_T» ／ «DRIFT_RATIO_TPU_T»</td><td>B200，输出头 dgrad 沿 w̄ 的相对偏差；后两项为不重算配置第 1000 步（参数范数带符号均值；漂移对 4 份默认 GPU 运行均值，取前后 25 步平均）</td></tr>
  <tr><td>定案前比过的基线：⑤ 参数范数 «PRODCHUNK_PN_R_T» 倍、loss 尾段 «PRODCHUNK_LOSS_R_T» 倍；⑥ 参数范数 «TRUEEXACT_R_T» 倍；dlogits 留 FP32 的诊断版步时 ＋«HEADFP32_COST_T»</td><td>v5p 当前配置（⑥ 一项含 v7x），尺子为 ③ 那组 10 对</td></tr>
  <tr><td>定案后第一关：最差 «FINAL_WORST_T» 倍（v5p 参数范数 «FINAL_PN_V5P_T»、v7x «FINAL_PN_V7X_T»），EP＝8 也过；TPU 扰动组 «ENSEMBLE_RUNS» 份里 «ENSEMBLE_OK» 份 ≤ «FINAL_WORST_T»、1 份梯度范数待定（«ENSEMBLE_PEND_T»，v7x 扰动组梯度范数 MARE 两两 «V7X_GN_MARE_TPU» 对 GPU «V7X_GN_MARE_GPU»）；v7x 当前配置四层前向 R «V7X_L2_R_T»、第 2 层 MoE 块 «V7X_MOE_R» → 去掉 1 个翻转 token «V7X_MOE_R_DROP»；10 对时待定 «PEND10_T»；原口径（10 份两两 45 对）最差 «FINAL_WORST_OLDCAL_T»</td><td>对 ⑤ 的一份，尺子 ⑤ 自己 10 份两两 45 对；v5p、v7x 当前配置 1000 步</td></tr>
  <tr><td>两处测量口径：TPU 参数范数单步自测误差 «PN_MEAS_TPU_T»（GPU «PN_MEAS_GPU_T»，GPU 重跑单步差 «PN_RERUN_STEP_T»）；梯度范数 N 随配对方向 «GN_N_AB_T» ／ «GN_N_BA_T»</td><td>同一次运行里按组记录的平方和与整网一次归约对比；梯度范数改用两边均值当分母</td></tr>
  <tr><td>开关代价：FP32 累加步时 v7x ＋«DFA_COST_V7X_T»、v5p ＋«DFA_COST_V5P_T»；按专家切 v7x −«EP8_SPEEDUP_V7X_T»；GPU 升 FP32 加分段 ＋«CHUNK_COST_T»（BF16 张量核分段 ＋«CB_COST_T»）</td><td>FP32 累加为 100 步对比；按专家切为 1000 步运行第 50–1000 步步时中位数；分段累加为 GPU 侧步时</td></tr>
  <tr><td>TPU 自己剩下的（旧口径，对精确轨迹；第一个数对 ④′，后两个对 ④）：«TPU_OWN_NOREMAT_975_T» ／ «TPU_OWN_OLDCFG_1000_T» ／ «TPU_OWN_CURCFG_1000_T»</td><td>分别是：不重算配置第 975 步；生产近似配置、开关打开前第 1000 步；当前配置第 1000 步</td></tr>
  <tr><td>回归：«T0_STEPS» 步那档约 «T0_MIN» 分钟、«STEPS» 步那档约 «T1_MIN» 分钟；扰动臂旧配置 «ARMS_TRIED» 试 «ARMS_KEPT» 过、«CAL_PAIRS» 对；新配置 7 个重跑全过、重标 28 对</td><td>v5p（8 颗芯片），提交到取回</td></tr>
    <tr><td>第 1 步路由翻转：翻了的 token 第 8、9 名分差中位数 «FLIP_MARGIN_DASH_T»；某专家梯度 «STEP1_EXPERT»、«STEP1_TENSORS» 个张量合起来 «STEP1_ALL_E»</td><td>第 1 步隔离回放；被测梯度由 Adam 一阶动量反推</td></tr>
    <tr><td>路由翻转对照（动画 anim-flip）：全部 token 第 8、9 名分差中位数 «ALL_MARGIN_LO» – «ALL_MARGIN_HI_T»；喂相同路由输入，两边路由打分（logits）的 rel_L2 仍约 «ROUTE_SAME_IN_DIFF»、翻转 0；输入换成 TPU 自己算的，每层翻 «FLIP_L2_T»；全程 FP32 下第 1 层翻的那个 token 分差 «FP32_FLIP_MARGIN»（据此得出的「两边实现不同」已撤回）</td><td>前两项：第 1 层起逐层隔离回放（路由输入用 TPU 自己算的，或换成 GPU 那份），3 层各 «BATCH_TOKENS» 个 token；第三项：GPU 第 99 步状态，两侧全程 FP32 前向逐点比</td></tr>
  <tr><td>逐层 rel_L2：MoE 三层 «L_RELL2_1_T» ／ «L_RELL2_2_T» ／ «L_RELL2_3_T»（参考档在这几层不适用，严格档 R 比过）；第 0 层前向 «REF_LAYER0_FWD_E»、反向 «REF_LAYER0_BWD_E»（参考档过）；参考线来历：同类项目注意力层逐层前向实测 «REF_LAYER_MEAS_T»</td><td>v5p，第 1 步逐层回放；第 0 层两平台相同</td></tr>
  <tr><td>去掉 w̄ 方向后 loss：前半 «REMOVE_LOSS_H1_T»、后半 «REMOVE_LOSS_H2_T»（尾段平均 «REMOVE_LOSS_T»）</td><td>两侧都去掉 dh 沿 w̄ 的分量，1000 步，对 GPU 第一份</td></tr>
  <tr><td>第 1 步换尺子：最差张量 TPU 只改片上内存上限 «STEP1_EXEC_T»、TPU 对 GPU «STEP1_TVG_T»；并集 «UNION_PAIRS» 对；隔离回放反向 R v5p «ISO_BWD_V5P_T»、v7x «ISO_BWD_V7X_T»；整步按能量 R «STEP_R_T»；逐处改掉 «ARM_R_T»；三个开关步时 v5p ＋«GRAD_FP32_COST_T»、v7x ＋«GF32_COST_V7X_T»，打开后 1000 步对 ⑤：v5p «GF32_V5P_T»，v7x 梯度范数«GF32_V7X_GN_T»、其余过；路由偏置 FP64 选中 «BIAS_TOKENS_FP64»（均值 «BIAS_MEAN_T»）</td><td>第 1 步；前两个数是最差张量的相对差；诊断改动的 R 只在 v5p 上测；FP64 真值是同一全局批在 FP64 里求的梯度；开关步时为 100 步对比</td></tr>
  <tr><td>（定案前）交叉核对：精确轨迹 5 份（基线＋4 份扰动版）两两 10 对 «XCHK_SELF_PN» ／ «XCHK_SELF_LOSS» ／ «XCHK_SELF_TAIL» ／ «XCHK_SELF_GN»；TPU 执行噪声 28 对 «XCHK_TPU_PN» ／ «XCHK_TPU_LOSS» ／ «XCHK_TPU_TAIL» ／ «XCHK_TPU_GN»</td><td>参数范数 ／ loss 全程 ／ loss 尾段 ／ 梯度范数，v5p 当前配置，只报不判</td></tr>
  <tr><td>（定案前）EP＝8：参数范数带符号均值 «ACC_PN_X» → «EP8_PN»，«ACC_PN_R» 倍 → «EP8_R» 倍；变化 «EP8_DELTA» 小于 TPU 执行噪声上限 «TPU_NOISE_PN_N»（比较为本课推导）</td><td>v5p 当前配置，TPU 改成按专家切，对精确轨迹</td></tr>
  <tr><td>（第二次迁移）第 1 步逐张量比值中位数 «B2_RATIO_MED_T»、超 2 倍 «B2_OVER2_T»（共 «B2_TENSORS»）；四个开关 «B2_MED_4SW» ／ «B2_OVER_4SW»</td><td>更新的、带稀疏注意力的 MoE 模型，小尺寸、随机初始化；TPU 对 GPU 第 1 步三份，按开关逐一打开</td></tr>
  <tr><td>（第二次迁移）注意力打分改留 FP32：第 1 步 loss «ATTN_FIX_LOSS_T»、梯度范数 «ATTN_FIX_GN_T»；GPU 噪声 «ATTN_NOISE_LOSS_T» ／ «ATTN_NOISE_GN_T»；多路残差四路相加改成依次相加，第 1 步该处梯度对 CPU FP64 R ＞ 1.2 «MIX_OVER_T»（后一代没复现）</td><td>GPU 只改这一处的第 1 步对比、8 份两两 28 对；四路相加为第 1 步对 CPU FP64 定责</td></tr>
  <tr><td>（第二次迁移）病态：KL ≈ «KD_KL»、lse ≈ «KD_LSE»、一格 «KD_ULP_T» ＝ KL 的 «KD_ULP_OF_KL_T»；TPU lse «KD_LSE_TPU_ULP_T» 格；同一行对 FP64 «KD_DIFF_T»，全部运行对 GPU «KD_VS_GPU_T»；输入 R «KD_IN_R»、梯度 R «KD_GRAD_R»</td><td>第 1 步，存下一行 KD 的原始输入在 FP64 里逐项重算；船和船长的数为本课推导（按比例换算）</td></tr>
  <tr><td>（第二次迁移）尺子撤回并集：梯度范数 r «UNION2_GN_T»；路由偏置均值 r «BIAS_ZM_R_T»；回放注入 R «INJ_R_T»（单独喂 «INJ_ALONE_R»）；参数范数测量偏低 «PN2_FP32_BIAS»，改精确值后第 1 步差 «PN2_STEP1_EXACT_T»</td><td>300 步比较，并集 «UNION2_PAIRS» 对 → GPU «GPU2_PAIRS» 对；偏置为 TPU 关零均值更新后重跑；回放为第 1 步前向的单算子回放</td></tr>
  <tr><td>（第二次迁移）优化器单测：三处写法都不对上 «OPT_ALL_OFF»、都对上 «OPT_ALL_ON_T»（另一用例 NAdam 对 Adam «OPT_NADAM»）；选 token 的 top-k 翻转：«SPARSE_Q_FLIP_T»的 query、槽位 «SPARSE_SLOT_FLIP»</td><td>对参照框架实现生成的固定参照逐步比每步更新的相对差；翻转为第 1 步 GPU 对 FP64</td></tr>
  <tr><td>（第二次迁移）编译器省掉舍入：五次 1000 步 loss 对 GPU 均值 «ELIDE_RANGE_T»，GPU 最大 ±«ELIDE_GPU_MAX_T»；强制舍入后 «ELIDE_FIX_T»（r «ELIDE_R_T»）；五次里 «ELIDE_OVER» 次超出 GPU 噪声</td><td>同一组 GPU 8 份；三次只改执行参数；「后端只省一半」在最小复现里看到；长跑偏高归到它为推断</td></tr>
  <tr><td>（第二次迁移）超越函数：默认 «TRANS_T» 格，高精度 «TRANS_HI_T» 格；exp 真实输入正好舍对 «EXP_CR_FRAC»、平均 «EXP_SIGNED_ULP_T» 格；lse «KD_FIX_LSE_T»、300 步 r «KD_FIX_R300_T»（同一代还开了一个优化器开关，没拆开）；sigmoid 开关前后 «SIG_FLAG_SAME_OPS» 个算子逐位相同，精确写法后 r «SIG_LOSS_R_T»；除法、开方、偏差修正 2／3／108 → 1／1／2 格，小张量 R «ADAM_R_T»</td><td>TPU v5p FP32 对 FP64 正确舍入数格；GPU 上限按 CUDA 文档（sigmoid、silu 为推断）；整网为 300 步比较</td></tr>
  <tr><td>（第二次迁移）r 与 R：更新差 «NSB_DIFF_0»% → «NSB_DIFF_1»%，«NSB_TPU_CLOSER» 张量步 TPU 更近，误差中位数 «NSB_ABS_T»，GPU «NSB_GPU»；QK-Clip 最大 logit «QKC_SMAX_T» → η «QKC_ETA_T»</td><td>优化器 8 步回放；NaN 为真实权重截层第 1 步</td></tr>
  <tr><td>（第二次迁移）规模：中等尺寸 300 步 过 ／ 不过 ／ 待定 «LADDER_MID_T»；真实权重截层第 1 步 «LADDER_REAL_STEP1_T»；FP32 轨迹第 2 步起离开 «DIV_FP32_AWAY»；第 1 步 R «DIV_STEP1_R_T»</td><td>GPU 8 份两两 28 对；FP32 参照自己的误差没有测</td></tr>
  <tr><td>（第二次迁移）纯噪声轮换：300 步 8 份 56 种，«MC_300_OBS_T» 过 1、最高 «LOO8_300_MAX_T»；1000 步 16 份 240 种，«MC_1K_OBS_T» 过 1、最高 «LOO16_1K_MAX_T»；单条检查最高 «LOO1_TAIL»–«LOO1_BM»；粗算 «MC_300_T» ／ «MC_1K_T»（本课推导）</td><td>GPU 同设置重跑轮流当被测，尺子用其余各份，检查同 TPU 自比；单条最高与粗算是本课从逐种轮换数据算的</td></tr>
  <tr><td>（第二次迁移）8 → 16 份尺子上限放宽 «N16_WIDEN_T» 倍；六个 TPU 改动 «GZ8_LO_T»–«GZ8_HI_T» → «GZ16_LO_T»–«GZ16_HI_T»</td><td>1000 步对 TPU golden；尺子 GPU 8 份 28 对 vs 16 份 120 对</td></tr>
  <tr><td>（第二次迁移）第 1 步两把参照：三个改动各 160 个张量；词嵌入梯度 R «EMB_DP_X»（纯数据并行）、R «EMB_NOEXC_X»（关 excess precision；分母是基线与 GPU 中位数里远的那个）</td><td>TPU v5p 自比第 1 步训练梯度（由动量反推），对 CPU FP64；45 个多路残差小参数单列</td></tr>
  <tr><td>（第二次迁移）30 条根因：精度 22 ／ 语义 6 ／ 两样都有 1 ／ 顺序 1（第 27 条 GPU 一侧把索引器旋转位置编码放在 BF16 里算、第 28 条多路残差收拢后少舍的那次 BF16、第 29 条纯数据并行时词嵌入不排序、第 30 条多路残差流梯度的舍入写法（主因两头验证；收拢处舍入开着时测不出作用，机制待查），都归精度）</td><td>对齐记录根因一览逐行的「根因」标注；归类是本课做的</td></tr>
  <tr><td>（第二次迁移）TPU 的脾气（绝大多数在运行库 0.0.46 上）：121 种改动分六堆，逐位 34、跟重跑一样大 44、真有影响 17（归成 7 件，不能忍 2 件、待拍板 1 件；其中 1 种是查 RMSNorm 机理的诊断臂）、量不出来 5、故意改坏 15、没出结果 6；另有 0.0.52 上的 25 次重测（逐位 5、一样大 11、真有影响 7、其余 2 次归在量不出来，§10.2；后补的三次是专家并行 4、8 和它们的底，离线编译先看到多了排序、上卡核对只换了求和顺序）；其中 «LONG_ARMS» 种跑满 1000 步，3 个分得开；（2026-11-12 对齐记录口径）换 v7x 第 1 步中位数 «V7X_STEP1_MED»、300 步 r «V7X_R300»、芯片·秒省 «V7X_SAVE»</td><td>同一小尺寸模型，每次只改一样，300 步对 TPU golden；r 的分母取 GPU 重跑；第 1 步对 FP64 定责；分堆由对齐记录的生成器按结果文件自动归类</td></tr>
  <tr><td>（第二次迁移）灵敏度：学习率偏 1e-4 → r 4.11，3e-5 → 1.06；五个精度改动 1000 步 0.97–1.64</td><td>故意埋 bug、300 步；右块 1000 步、GPU 16 份</td></tr>
  <tr><td>（第二次迁移 · 案例二）纯数据并行四种条件 r «C2_R_DP8_T»／«C2_R_GA1_T»／«C2_R_DP16_T»／«C2_R_V7X_T»；第 1 步 «C2_NT» 个张量只有词嵌入梯度远 «C2_EMB_R_T» 倍</td><td>TPU v5p（v7x 那行在 v7x）自比，对 FSDP 8 的 TPU golden，300 步；各臂结果文件的曲线判定与第 1 步逐张量表</td></tr>
  <tr><td>（案例二）按行分组：39 个高频词占梯度范数平方 0.90，纯数据并行多出来的误差 «C2_HOT_SHARE_T» 在它们身上；本卡次数翻倍误差近翻倍、卡数翻倍不变</td><td>第 1 步词嵌入梯度对 CPU FP64，按 token 在全局批里的出现次数分组（对齐记录：梯度的跨卡归约）；占比本课按误差平方相减复算</td></tr>
  <tr><td>（案例二）先加后传 vs 先传后加的形状；跨卡数据量约差 «C2_BYTES_ROUND» 倍</td><td>形状读自两边训练步优化后的 HLO；数据量按形状算，本课推导，线上实际字节数没测</td></tr>
  <tr><td>（案例二）四格、按出现次数分组、下标数扫过门槛、8 设备两种切法</td><td>单算子，v5p 真跑，下标取 golden 第 1 步真实 token，更新量随机 BF16；结果与 CPU 两种仿真逐位比</td></tr>
  <tr><td>（案例二）门槛 N ＞ V／8、插排序的工序名、关省舍入多出 «C2_VPACK» 条舍入指令</td><td>CPU 上用同一个运行库为 v5p 交叉编译：逐道工序 dump HLO、二分七种表形状；后端指令是静态条数</td></tr>
  <tr><td>（案例二）修法 ① ② 的 r 与步时；关省舍入 r «C2_NOEXC_R_T»</td><td>各臂结果文件（300 步对 golden，1000 步对 1000 步的 golden）；步时取第 11 步起每步中位数对 golden</td></tr>
  <tr><td>（案例二 · 第三种修法）只切词嵌入表：第 1 步 «C3_EMB_T»（R «C3_EMB_R_T»），r «C3_R300_T» ／ «C3_R1K_T»，步时 «C3_ST_T»</td><td>TPU v5p 8 设备，纯数据并行 8 ＋ 累积 2 只多一个开关；300 步对 golden、1000 步对 1000 步 golden，尺子 GPU 8 份 ／ 16 份</td></tr>
  <tr><td>（案例二）五种跑法的通信字节：词嵌入 9.90e8 → 6.82e7（−«C3_EMB_CUT_T»），全部 −«C3_ALL_CUT_T»；修法① «C3_FIX1_EMB_T»</td><td>各运行上卡时存下的训练步优化后 HLO，按（类型, 形状）合并，形状 × 字节宽 × 每步次数；按形状算，不是线上实测</td></tr>
  <tr><td>（案例二）GPU 为什么没这个问题</td><td>推断：读 PyTorch 主分支 <code>aten/src/ATen/native/cuda/Embedding.cu</code> 的分路门槛（3072）；GPU 镜像里那个版本没有逐行核过</td></tr>
  <tr><td>（第二次迁移）程序指纹相同 3 种改写；编译器 bug：词嵌入梯度 0.51 倍、2829／10911 行为零、r 611 → 修后 0.98</td><td>各改写臂结果文件的 program_equal；bug 读自编译后程序的循环体与第 1 步抓数</td></tr>
  <tr><td>（第二次迁移）小尺寸 1000 步九项（关 base2、尺子 16 份）：过 7、待定 2（主 loss 1.55、1.52，TPU 共同偏高约 4.3e-6；前段定到多路残差收拢后少舍的 BF16，后段在查，两边去掉多路残差后不飘）；GPU 换机器数和切法最大 1.76、正对照 4149；真实宽度第 1 步 GPU ＋5.76e-5 ／ TPU −9.87e-5，只关 base2 → −1.24e-6；1000 步回放开 2／4／3、关 3／6／0</td><td>TPU v5p 8 设备 ↔ B200×8；GPU 扩展对单机 8 份尺子前 300 步；真实宽度 TPU v5p 16 片原样回放 GPU 2 台的 1000 步；base2 的系数 1.0018 本课按 BF16 舍入复算</td></tr>
  <tr><td>混元 3 第 1 步按只用 GPU 的尺子重判：变严 «HY3_REJ_N» 项，没有变松；部分张量 R 超 1.2：«HY3_REJUDGE_R»</td><td>GPU 第 1 步重跑 3 对；1000 步结论不变</td></tr>
</tbody></table></details>
</div></section>
'''

FOOT = '''
<div class="wrap" style="padding:32px 0 64px">
  <p style="color:var(--gray)">
    ← 回 <a href="index.html">课程总纲</a>　·
    相关：<a href="topic-09.html">专题九 · 精度与量化</a>　·
    想自己讲一遍？看 <a href="topic-08-lecture.html">本讲讲义（授课稿）</a><br>
    本页由 <code>Courses/tools/topic08-build.py</code> 生成 ——&nbsp;<b>正文写在那个脚本里</b>。</p>
</div>

</body></html>'''

FIGS = {
    "__FIG_ASSOC__": ("fig-assoc", "fig8-assoc.svg", "topic08-fig-intro.py",
        '<b>同一个式子，两种顺序，差了一整格。</b><br><em>格距按 BF16 在 1 附近的真实格距画；正好在两格中间时，舍到末位是偶数的那一格（平局取偶）。</em>'),
    "__FIG_RULER__": ("fig-ruler", "fig8-ruler.svg", "topic08-fig-intro.py",
        '<b>灰色一段是尺子，蓝点都在尺子外面。</b><br><em>三个蓝点是同一份代码在 TPU 上的三种跑法。</em>'),
    "__FIG_FUNNEL__": ("fig-funnel", "fig8-funnel.svg", "topic08-fig-intro.py",
        '<b>左边往上验收，右边往下排查。</b><br><em>每一级用什么尺子，见下面的表。</em>'),
    "__FIG_REPLAY__": ("fig-replay", "fig8-replay.svg", "topic08-fig-l0.py",
        '<b>每一格都要对上指纹。</b><br><em>画的是一步；1000 步一共 «DATA_UNITS» 格。</em>'),
    "__FIG_WD__": ("fig-wd", "fig8-wd.svg", "topic08-fig-l0.py",
        '<b>虚线是写的，实线是用的。</b><br><em>证据是运行时读出的元数据，不是配置文件。</em>'),
    "__FIG_DT__": ("fig-dt", "fig8-dt.svg", "topic08-fig-series.py",
        '<b>两团点一样宽，下面那团蓝点整体沉下去了。</b><br><em>点都是真实的逐步值。</em>'),
    "__FIG_MTP32K__": ("fig-mtp32k", "fig8-mtp32k.svg", "topic08-fig-series.py",
        '<b>灰线来回穿过零，红线一直在零上面。</b><br><em>出没出带看单步乱不乱跳；围不围着零看有没有系统偏差。7 条灰线和背景带都只用来看，尺子仍是 28 对两两的最大值，判定按右边三个数。</em>'),
    "__FIG_JUDGE__": ("fig-judge", "fig8-judge.svg", "topic08-fig-series.py",
        '<b>落在哪个颜色里，就是什么判定。</b><br><em>红黄点是定案前那次判定，基线是另一份 GPU 记录（不是开场对的第一份），尺子也换了，所以 loss 的数跟开场「偏 5.3 个标准差」对不上；绿点是定案后。</em>'),
    "__FIG_PAIRS__": ("fig-pairs", "fig8-pairs.svg", "topic08-fig-series.py",
        '<b>左边是道理，右边是实测。</b><br><em>左边柱高是配对彼此独立时的下限，模拟和留一法实测都显示带符号均值约翻倍；右边那次演练里，被测和参照是同一种设置，本该全过。</em>'),
    "__FIG_INTEGRATOR__": ("fig-integrator", "fig8-integrator.svg", "topic08-fig-yardstick.py",
        '<b>灰线越散越宽，蓝线一直斜着往上走。</b><br><em>左块是示意模拟；右块两支箭头一样长。</em>'),
    "__FIG_TWOAXIS__": ("fig-twoaxis", "fig8-twoaxis.svg", "topic08-fig-yardstick.py",
        '<b>先看红虚线上面有没有点。</b><br><em>横轴对数，大于 1 就超出 GPU 噪声；纵轴对数，红虚线是 1.2。</em>'),
    "__FIG_KINDS__": ("fig-kinds", "fig8-kinds.svg", "topic08-fig-kinds.py",
        '<b>先认是哪一种，再决定修不修。</b><br><em>中间那排小线是示意；底下的条数是本课按对齐记录逐条归的类。</em>'),
    "__FIG_TEMPER__": ("fig-temper", "fig8-temper.svg", "topic08-fig-temper.py",
        '<b>蓝点挤在 1 附近；红点实心的出了线，空心的曲线分不开、靠第 1 步对 FP64 抓到。</b><br><em>点竖着错开只为不重叠；横轴对数。</em>'),
    "__FIG_BUG__": ("fig-bug", "fig8-bug.svg", "topic08-fig-bug.py",
        '<b>左边：一样就不用跑。右边：七个红叉，是编译器丢掉的梯度。</b><br><em>8 张卡、每卡 2 个微批是这次的实际设置。</em>'),
    "__FIG_SENS__": ("fig-sens", "fig8-sens.svg", "topic08-fig-sens.py",
        '<b>蓝线从 1e-4 起斜着往上，左下那两个点埋在灰带里。</b><br><em>左块对数对数；右块五根条都没过虚线。</em>'),
    "__FIG_MIG2__": ("fig-mig2", "fig8-mig2.svg", "topic08-fig-mig2.py",
        '<b>左边只有一根橙条（路由偏置均值）略过 1；中间没有点过 2；右边红点关掉 base2 以后挪到真值旁边。</b><br><em>三块都是实测；左块那项待定，同配置另五份的这一项都过。</em>'),
    "__FIG_RPIPE__": ("fig-rpipe", "fig8-rpipe.svg", "topic08-fig-yardstick.py",
        '<b>上面一条序列配两个括号；下面一路除下来，最后取最大。</b><br><em>上块是混元 3、下块是第二次迁移的算例，数都是实测。</em>'),
    "__FIG_STEP1__": ("fig-step1", "fig8-step1.svg", "topic08-fig-yardstick.py",
        '<b>上面一行四步；同一个张量两次读数，第三步都超出，第四步才分出谁丢了精度。</b><br><em>下面四格是两类尺子的组合。</em>'),
    "__FIG_LOO__": ("fig-loo", "fig8-loo.svg", "topic08-fig-yardstick.py",
        '<b>灰点是纯噪声，一大片落进黄区。</b><br><em>一个点是一种轮换：一份当被测、一份当基线，尺子用被测以外的各份（含基线）两两重算。</em>'),
    "__FIG_BASELINE__": ("fig-baseline", "fig8-baseline.svg", "topic08-fig-series.py",
        '<b>被圈的灰点就是基线；粗蓝线才是扣掉噪声中心以后的偏移。</b><br><em>灰点竖着错开只为不重叠。</em>'),
    "__FIG_COS__": ("fig-cos", "fig8-cos.svg", "topic08-fig-tensor.py",
        '<b>红色虚线的长度就是 rel_L2。</b><br><em>中格的角度放大了画。</em>'),
    "__FIG_RRATIO__": ("fig-rratio", "fig8-rratio.svg", "topic08-fig-tensor.py",
        '<b>先看点落在哪一圈底色里。</b><br><em>灰圈是 GPU 离真值的距离，定为 1；两次独立、同量级的舍入叠起来，误差平方相加：√(1²＋1²) ≈ «R_ONE_EXTRA_T»。</em>'),
    "__FIG_FLIP__": ("fig-flip", "fig8-flip.svg", "topic08-fig-tensor.py",
        '<b>左边两条红框换了位置，右边就多出一根长红柱。</b><br><em>名次条的长短和被换掉的点位是示意，个数和柱长是实测。</em>'),
    "__FIG_FORMATS__": ("fig-formats", "fig8-formats.svg", "topic08-fig-fp.py",
        '<b>蓝色一样宽，绿色不一样长。</b><br><em>下半的对齐位数是示意。</em>'),
    "__FIG_TRUNCATE__": ("fig-truncate", "fig8-truncate.svg", "topic08-fig-fp.py",
        '<b>左边虚线右侧的红格全被丢掉，右边那条线一路往上。</b><br><em>右图画的是偏差的大小（对数刻度），符号是负的、朝零；合成数据，不是模型里的数。</em>'),
    "__FIG_TF32__": ("fig-tf32", "fig8-tf32.svg", "topic08-fig-fp.py",
        '<b>看左边小字和柱子颜色对不上的那两行。</b><br><em>对数刻度，一格十倍。</em>'),
    "__FIG_WINDOWED__": ("fig-windowed", "fig8-windowed.svg", "topic08-fig-fp.py",
        '<b>数每一行的红块：1 个、6 个、1 个。再看右边，4 个设备那一行是空的。</b><br><em>左边三行的结构取自编译产物；右边是离线编译训练步数出来的。</em>'),
    "__FIG_INOUT__": ("fig-inout", "fig8-inout.svg", "topic08-fig-fp.py",
        '<b>左边每片给出整个 y 的一份，要加；右边每片给出 y 的几列，只拼。</b><br><em>q_a、q_b 和 kv_a、kv_b 是 MLA 里 q、kv 各两个低秩投影，o_proj 是注意力输出投影。形状只画关系，不按比例；底下一行是实测。</em>'),
    "__FIG_LMHEAD__": ("fig-lmhead", "fig8-lmhead.svg", "topic08-fig-fp.py",
        '<b>左边第二行那一截红线，就是右边红条比灰条长出来的那三个数量级。</b><br><em>左格是示意；右格和下面三张卡是实测，单份运行。</em>'),
    "__FIG_EXCESS__": ("fig-excess", "fig8-excess.svg", "topic08-fig-fp.py",
        '<b>数每一行的红块：第二行少了，第三行又回到第一行的位置。</b><br><em>四个算子、融合怎么切都是示意。</em>'),
    "__FIG_NORMSCALE__": ("fig-normscale", "fig8-normscale.svg", "topic08-fig-fp.py",
        '<b>左边蓝线一路往上、灰线贴着零；右边只有两根短绿条落进 ±3σ。</b><br><em>左边虚线是外推；右边两组窗口不同，组内比。</em>'),
    "__FIG_ULP2__": ("fig-ulp2", "fig8-ulp2.svg", "topic08-fig-fp.py",
        '<b>v5p 两根柱子从几十格掉到 GPU 虚线附近；v7x 两根一样高，exp 还在虚线上面一大截。</b><br><em>对数刻度；v5p 两根就是下一张图左格的数，这张多了 v7x。</em>'),
    "__FIG_BOUNDARY__": ("fig-boundary", "fig8-boundary.svg", "topic08-fig-fp.py",
        '<b>每一行先找两条线分开的那一格：五个例子，五列各一个。</b><br><em>例子取自三次迁移；只讲在哪不一样，不讲差多大。</em>'),
    "__FIG_CAPTAIN__": ("fig-captain", "fig8-captain.svg", "topic08-fig-fp.py",
        '<b>红色那一小截就是秤的一格。</b><br><em>船和船长是比方，比例跟实测一样；右边四问是实测。</em>'),
    "__FIG_RR__": ("fig-rR", "fig8-rR.svg", "topic08-fig-tensor.py",
        '<b>左边看差多少，右边看谁更近真值。</b><br><em>同一个例子；点距和圆半径只表示大小关系。</em>'),
    "__FIG_ELIDE__": ("fig-elide", "fig8-elide.svg", "topic08-fig-fp.py",
        '<b>中间一行五个点都在零的右边，四个出了灰段。</b><br><em>一个点是一次 1000 步的 TPU 运行；三行都是实测，「省了一半」的机制在最小复现里看到。</em>'),
    "__FIG_TRANS__": ("fig-trans", "fig8-trans.svg", "topic08-fig-fp.py",
        '<b>左边看蓝条比灰条长多少。</b><br><em>对数刻度；右边是教科书做法的示意。</em>'),
    "__FIG_DIVERGE__": ("fig-diverge", "fig8-diverge.svg", "topic08-fig-observe.py",
        '<b>蓝灰两条离 FP32 几乘十的负三次方，红线两边之差只有它的百分之一上下。</b><br><em>这条 FP32 轨迹连权重都不舍成 BF16；只有第 1 步能拿它比。</em>'),
    "__FIG_ONGRID__": ("fig-ongrid", "fig8-ongrid.svg", "topic08-fig-observe.py",
        '<b>左边第一行的点都压在竖线上，右边蓝线第 1 步落在 0 上。</b><br><em>两格的数都是玩具实验现跑的，只看第 1 步是不是 0，差多大没有意义。</em>'),
    "__FIG_SWITCH__": ("fig-switch", "fig8-switch.svg", "topic08-fig-golden.py",
        '<b>从左上那个问题开始往下走。</b><br><em>1% 是本课取的界。</em>'),
    "__FIG_LADDER__": ("fig-ladder", "fig8-ladder.svg", "topic08-fig-golden.py",
        '<b>右边每一级两根条：② 那一级灰条（GPU）先变长，③ 那一级紫条（TPU）才变长。</b><br><em>格子只写跟谁一样，不写尺寸；右边对数轴。</em>'),
    "__FIG_HOOKS__": ("fig-hooks", "fig8-hooks.svg", "topic08-fig-observe.py",
        '<b>中间那格：虚框被切成两块，就是舍入位置变了。</b>'),
    "__FIG_DRIFT__": ("fig-drift", "fig8-drift.svg", "topic08-fig-case1.py",
        '<b>上格蓝粗线往下、下格蓝线往上，都从第 100 步前后离开灰线。</b><br><em>真实的逐步值；灰线是另两份 GPU 重跑对同一基线，粗线是 50 步平均。</em>'),
    "__FIG_ROWS__": ("fig-rows", "fig8-rows.svg", "topic08-fig-case1.py",
        '<b>左边一根独大，右边第一根独大。</b><br><em>灰色细条是噪声，短到几乎看不见。</em>'),
    "__FIG_EPS__": ("fig-eps", "fig8-eps.svg", "topic08-fig-case1.py",
        '<b>红区在斜坡上，绿区在平台上。</b><br><em>曲线是稳态近似，只讲形状。</em>'),
    "__FIG_LAYER0__": ("fig-layer0", "fig8-layer0.svg", "topic08-fig-case2.py",
        '<b>粗蓝线最先往下走，浅蓝线一层比一层低；右边只有换上 TPU 权重那根长。</b><br><em>ĥ：最后一层隐状态的均值方向（û），每一维再乘最后那次归一化的缩放 γ；左图每 25 步一个实测点；探针读的是残差流，第 k 层的读数含前面各层的贡献，所以越往后越低 —— 看的是谁最先分开。</em>'),
    "__FIG_GAUGE__": ("fig-gauge", "fig8-gauge.svg", "topic08-fig-case2.py",
        '<b>左边是那个方向，右边是它每步少一点积出来的结果。</b><br><em>左图的箭头是示意。</em>'),
    "__FIG_INTERVENE__": ("fig-intervene", "fig8-intervene.svg", "topic08-fig-case2.py",
        '<b>绿柱缩回去（参数范数还剩三成），橙柱长回来；下格逐步看，蓝橙两条一路同涨，绿线长得慢得多。</b><br><em>每组一份 1000 步的运行，实测；灰线呈台阶状，是 GPU 一侧参数范数的记录精度有限（台阶约 1.6e-4）。</em>'),
    "__FIG_HEAD__": ("fig-head", "fig8-head.svg", "topic08-fig-case2.py",
        '<b>左边那根红柱是 GPU 的张量核；深灰那根（名义 FP32、其实走了 TF32）先不看，第十节回来讲。</b><br><em>参照是同一份输入在 FP64 里的结果。GPU 那根只算输出层 dgrad 一段，TPU 那根是整个输出头反向；GPU 交叉熵反向那段只有 «CE_BWD_S1_T»，所以两根可以直接比。</em>'),
    "__FIG_SWAP__": ("fig-swap", "fig8-swap.svg", "topic08-fig-case2.py",
        '<b>蓝色几乎贴着精确值，红柱最长。</b><br><em>每根柱子是「这一步」的偏，不是累积。</em>'),
    "__FIG_ONEPASS__": ("fig-onepass", "fig8-onepass.svg", "topic08-fig-case2.py",
        '<b>橙线（只换 FP32）一千步只到六成，下格一路往下离开；绿线（改成分块）跟蓝线（TPU）重合，下格贴着零。</b><br><em>左格是训练轨迹（逐步实测，生产近似配置）；右格是同一次矩阵乘的离线测量。</em>'),
    "__FIG_EXACT__": ("fig-exact", "fig8-exact.svg", "topic08-fig-case2.py",
        '<b>三行里，紫点都挨着蓝点。</b><br><em>零点是 GPU 默认运行的平均。</em>'),
    "__FIG_CHAIN__": ("fig-chain", "fig8-chain.svg", "topic08-fig-case2.py",
        '<b>从左往右读，每一格是下一格的原因。</b><br><em>五格的现象都是量到的；第一格的原因是推断，②③ 两格的定量衔接没测过（§7.3），loss 为什么跟着变也没有验证；积起来以后还有反馈放大（推断）。</em>'),
    "__FIG_DDPCURVE__": ("fig-ddpcurve", "fig8-ddpcurve.svg", "topic08-fig-ddp.py",
        '<b>上格蓝线的平均前 200 步明显在零下面，下格绿线回到灰线中间。</b><br><em>灰线是 GPU 同设置另 7 份对其中一份，就是尺子用的那 8 份；被测对的是 TPU 自己的 FSDP 基线。</em>'),
    "__FIG_DDP2X2__": ("fig-ddp2x2", "fig8-ddp2x2.svg", "topic08-fig-ddp.py",
        '<b>只有右下那一格红：高频又不排序。</b><br><em>圆点个数是示意；格子里的数是第 1 步实测。低频那两格写两个数：「1 张卡」是只在 1 张卡上出现的 7,638 行，「多张卡」是在 ≥ 2 张卡上出现的 3,234 行；它们的误差加进词表之前就有（推断：上游的 BF16 计算），所以跟排不排序无关。</em>'),
    "__FIG_DDPROWS__": ("fig-ddprows", "fig8-ddprows.svg", "topic08-fig-ddp.py",
        '<b>看右边一组：紫柱接近红柱的两倍，橙柱跟红柱一样高。</b><br><em>绿柱是 FSDP。</em>'),
    "__FIG_DDPFLOW__": ("fig-ddpflow", "fig8-ddpflow.svg", "topic08-fig-ddp.py",
        '<b>上一条先加后传，下一条先传后加；下面那次加法收了 32768 个下标。</b><br><em>形状读自编译后的程序。</em>'),
    "__FIG_RMW__": ("fig-rmw", "fig8-rmw.svg", "topic08-fig-ddp.py",
        '<b>左边数红框：不排序每次都舍，排序只舍一次；中间只有一格绿。</b><br><em>右块红柱随次数涨，绿柱不动。</em>'),
    "__FIG_RMS52__": ("fig-rms52", "fig8-rms52.svg", "topic08-fig-temper.py",
        '<b>红粗线一直在零上面，蓝粗线贴着零；橙线跟红线重合。</b><br><em>单步约七成在灰带里，看得出的是方向一致；单份运行，实测。</em>'),
    "__FIG_RMS2R__": ("fig-rms2r", "fig8-rms2r.svg", "topic08-fig-temper.py",
        '<b>左上两排只差一个虚线框：0.0.52 把归一化值单独写回了内存。右上 γ 贴近 1 时偏得最多；下面只动最后一层就开关这件事。</b><br><em>Ⓐ Ⓒ 实测；Ⓑ 是推导配 FP64 实测。</em>'),
    "__FIG_IHCROOT__": ("fig-ihcroot", "fig8-ihcroot.svg", "topic08-fig-mig2.py",
        '<b>右边灰条（排除的候选）都还长，只有绿条（两边都去掉多路残差）缩进灰带。</b><br><em>右边实测；左边 TPU 那排虚线是推断。</em>'),
    "__FIG_FIXCMP__": ("fig-fixcmp", "fig8-fixcmp.svg", "topic08-fig-mig2.py",
        '<b>左边红线四条都爬出灰带；右边绿线多数贴着 0。</b><br><em>两列同一纵轴；每条线一份运行；实测。</em>'),
    "__FIG_GATECMP__": ("fig-gatecmp", "fig8-gatecmp.svg", "topic08-fig-mig2.py",
        '<b>左边：在橙点那格上再补，三格紫点都没明显往下；右边：紫点有三个落在所有灰点的右边。</b><br><em>每点一份运行；实测；机制在查。</em>'),
    "__FIG_ELIDED__": ("fig-elided", "fig8-elided.svg", "topic08-fig-elided.py",
        '<b>左边：代码写了往返、HLO 里两条 convert 都在，LLO 打包却是 0。右边：整步扫一遍，三根橙条是先查的范围。</b><br><em>都是实测；右边的条按使用次数算，不是 kernel 数。</em>'),
    "__FIG_CANCEL__": ("fig-cancel", "fig8-cancel.svg", "topic08-fig-elided.py",
        '<b>只补收拢处那一行读数最大；只补流梯度那一行最小，可往下拉的那股还在。</b><br><em>读数是实测；「往下拉的那份」是推断，箭头只表示方向。</em>'),
    "__FIG_SITES__": ("fig-sites", "fig8-sites.svg", "topic08-fig-elided.py",
        '<b>三个编号是后面一直要用的地名：⓪ 分支输入、① 流梯度、② 收拢处。</b><br><em>示意图，只画一层。</em>'),
    "__FIG_VERIFY__": ("fig-verify", "fig8-verify.svg", "topic08-fig-elided.py",
        '<b>左边改了一处的格子都没稳稳长出尾巴；右边补上的两格跟对齐配置一样高。</b><br><em>每点一份运行；实测；判读来自对齐记录。</em>'),
    "__FIG_TWOWAY__": ("fig-twoway", "fig8-twoway.svg", "topic08-fig-mig2.py",
        '<b>左块红点降到绿点；右块灰点升到橙点：两头各改一处，方向对上。</b><br><em>每点一份运行；实测。</em>'),
    "__FIG_SCREEN__": ("fig-screen", "fig8-screen.svg", "topic08-fig-bug.py",
        '<b>左边一次编译三个出口；右边能判的全部对上。</b><br><em>数都是对齐记录的回测实测。</em>'),
    "__FIG_LIBTPU__": ("fig-libtpu", "fig8-libtpu.svg", "topic08-fig-temper.py",
        '<b>左边只有 RMSNorm 那一行往右蹿出了线；右边红点 8,192 在 0.0.46 门槛左边、0.0.52 门槛右边。</b><br><em>左块横轴对数；右块门槛全表在图下出处里；两块都是实测。</em>'),
    "__FIG_MINIDRIFT__": ("fig-minidrift", "fig8-minidrift.svg", "topic08-fig-mig2.py",
        '<b>上面的柱第 550 步以后一路变橙变高；下面 MTP 反着往下。</b><br><em>灰带是 GPU 自己的散布；柱的颜色只看 8 份是否同号，不是判档。</em>'),
    "__FIG_SWEEP__": ("fig-sweep", "fig8-sweep.svg", "topic08-fig-ddp.py",
        '<b>红线过了竖线就掉下来。</b><br><em>右块的点都落在虚线上；空心点是表太小、不插排序。</em>'),
    "__FIG_SHARD__": ("fig-shard", "fig8-shard.svg", "topic08-fig-ddp.py",
        '<b>上面前向只换取到的行；下面每卡只存八分之一，显存跟着降。</b><br><em>路径读自编译后的程序；显存是实测。</em>'),
    "__FIG_COMM__": ("fig-comm", "fig8-comm.svg", "topic08-fig-ddp.py",
        '<b>左边四步把一行读成字节；右边橙条最短的两行是 FSDP 和只切词嵌入表。</b><br><em>对数刻度；字节按形状算，不是线上实测。</em>'),
    "__FIG_GOLDEN__": ("fig-golden", "fig8-golden.svg", "topic08-fig-golden.py",
        '<b>②–④ 的柱子从左到右一截截变短。</b><br><em>②–④ 同一个被测、同一把尺子；⑤ 换了被测和它自己的尺子，单独看。</em>'),
    "__FIG_DECOMP__": ("fig-decomp", "fig8-decomp.svg", "topic08-fig-golden.py",
        '<b>先看上面的数轴：蓝点和橙点几乎叠在一起。</b><br><em>都是第 1000 步、生产近似配置、TPU 当前配置；下面三笔里红色是 GPU 一侧，蓝色是 TPU 一侧。</em>'),
    "__FIG_JOURNEY__": ("fig-journey", "fig8-journey.svg", "topic08-fig-golden.py",
        '<b>先看红虚框：四步走错，各留一条教训。</b><br><em>绿色八站是留下来的，每站一个数；细节在图下出处里。</em>'),
    "__FIG_TWOLAYER__": ("fig-twolayer", "fig8-twolayer.svg", "topic08-fig-golden.py",
        '<b>橙色那一关要判过不过，灰色那一关只写清楚。</b><br><em>虚线框是只报不判的交叉核对（TPU 扰动组）。</em>'),
    "__FIG_REGRESS__": ("fig-regress", "fig8-regress.svg", "topic08-fig-golden.py",
        '<b>大多数改动走上面那条绿线就结束了。</b><br><em>LKG 是 last-known-good 的缩写。</em>'),
}

# ── 动画（Manim，手册第 10 篇）：静态图在上，动画排在对应静态图下面 ──────────────
#   一步一停的片子：步点写在 manim/steps/<Scene>.json，这里挂到 <video data-pauses>，图注秒数也从同一份 json 取。
#   播放器、减少动态效果两段脚本照抄专题六（topic06_common.STEP_JS / REDUCED_MOTION_JS，那边有来由注释）。
import json  # noqa: E402
from topic06_common import STEP_JS as _STEP_JS, REDUCED_MOTION_JS as _RM_JS  # noqa: E402
ANIMS = {}   # 占位符 -> (anim id, mp4, Scene, 图注一句, 图注补充, aria-label)
ANIMS["__ANIM_CEILING__"] = ("anim-ceiling", "topic08-ceiling.mp4", "NoiseCeiling",
    "N 是从 5 份重跑两两组成的 10 对里挑出来的最大值；被测一出场，纵轴得缩一半才装得下。",
    "只演参数范数的带符号均值；10 根柱子都是真值，横轴只表示相对位置。",
    "噪声上限 N：拿 GPU 自己重跑当尺子。第 1 步　定案前第一关的尺子：GPU 重跑 5 份，一个点一份，编号按左右顺序；5 份是同一设置的 GPU 重跑。"
    "第 2 步　两两配对，一共 10 对：每对算参数范数的带符号均值，取绝对值，竖成一根柱；带符号均值是第 2 到 1000 步逐步相对差的平均，两个点隔多远，柱就多高。"
    "第 3 步　取最大的那一根当噪声上限 N。为什么取最大不取平均：被测若只是噪声，应落在已有样本的范围里。N 等于最大的那一对，«GPU_RERUN_PN»。"
    "第 4 步　被测（TPU 对一份 GPU 基线）出场，r 等于被测除以 N；尺子就是这 5 份 GPU 重跑。柱子比尺子高出一大截，纵轴得缩一半才装得下。被测 «ACC_PN_X»，r 等于 «ACC_PN_X» 除以 «GPU_RERUN_PN»，等于 «ACC_PN_R»。"
    "背景三个区：过（r 不超过 1）、待定（1 到 2）、不过（r 大于 2）。"
    "第 5 步　样本越多，尺子越可信：纯噪声超过 n 对最大值的概率至少 n 加 1 分之一（各对共用运行，模拟和留一法实测都显示带符号均值约翻倍）。3 对至少 1/4，10 对至少 1/11（这一次），28 对至少 1/29。")
ANIMS["__ANIM_SCATTER__"] = ("anim-scatter", "topic08-scatter.mp4", "ScatterRound",
    "不排序时同一个词每来一次就舍一次；排好序以后挨在一起，在片上 FP32 缓冲里加完才舍一次。",
    "下标流、词 17 出现 6 次和左边的误差条是示意；第 3、4 步的数是实测。",
    "同一个词的更新怎么加进词表那一行。词嵌入的反向：每个 token 的梯度按 token 号加回词表那一行（scatter-add）。"
    "第 1 步　不排序：同一个词的更新隔着别的词，每来一次都要读出、加上、舍回 BF16、写回；词 17 出现 6 次，舍了 5 次，第一次是加到 0 上不舍。"
    "第 2 步　排好序、带标记：同一个词挨在一起，在片上 FP32 缓冲里一口气加完，写出时舍一次。"
    "第 3 步　单算子实测：这一行在一次 4096 个下标的 scatter 里出现 100 次以上的那组，离 FP64：不排序 «C2_K100_UNS_T»，排序 «C2_K100_SRT_T»。"
    "第 4 步　训练第 1 步：39 个高频词的行，纯数据并行 «C2_HOT_DP_T»，FSDP «C2_HOT_FSDP_T»。"
    "出现 k 次：不排序舍 k 减 1 次，排序只舍 1 次。示意：真实高频词一次加几十到几百次；token 号随手写。")
ANIMS["__ANIM_CHAIN__"] = ("anim-chain", "topic08-chain.mp4", "DriftChain",
    "同一个小差每一步都一样，积到第 300 步就成了看得见的漂移。",
    "第 3 环每 25 步一个实测点（12 个），点间连直线；第 4 环只有第 300 步一个实测值，走到第 300 步才出现；⑤ 是逐步实测。",
    "五环一条链：每一步少推一点，几百步积出来。每一环是下一环的原因；底下的时钟是训练步数。"
    "① GPU 输出层 dgrad：12 万长的求和朝零少了一点（截断＋吞小项），每一步少推约 «HEAD_S1_E»（沿所有 logits 一起动的方向，第 1 步实测；第 100 步 «HEAD_S100_E»），每一步都少推、量级一样。"
    "② 主干权重沿这个方向每一步少走一点：整步沿漂移方向少走 «SWAP_GPU_ABS_T»（全体参数，同一状态实测），主干里大头在第 0 层注意力的输出投影。"
    "③ 所有 logits 一起偏一截（lse 差，TPU 减 GPU）；softmax 看不见它。第 25 步 «LSE25_T»，第 50 步 «LSE50_T»，第 100 步 «LSE100_T»，第 300 步 «LSE300_T»。"
    "④ 冷门词那几行在 Adam 的 ε 区（«RARE_EPS_FRAC_T» 的元素根号 v 小于 ε），压多压少跟着 logit 水平走；没出现过的词那几行平方和多涨 «ROWS300_NEVER»（第 300 步实测）。"
    "⑤ 参数范数多涨，第 300 步 «PN_DIFF_300_T»；按组拆输出层占 «GROUP_HEAD_SHARE_T»（另测一对，合计 «GROUP_TOTAL_300_T»）；loss 偏低（去掉那股力后参数范数还剩三成；loss 的机制未验证）。"
    "时钟走起来：①② 每一步都一样，③④⑤ 一步步积起来。第 300 步：每步只差一点、同一个方向，积成看得见的漂移。"
    "灰色是 GPU 两次之间；时钟上白色的是探针实测的那几步。五环都标实测。")
ANIMS["__ANIM_TRUNC__"] = ("anim-trunc", "topic08-trunc.mp4", "TruncVsRound",
    "抹零一路往下，就近舍入在 0 附近晃：同一次求和，越长偏得越多。",
    "数据是按实测张量核模型在合成正数输入上的模拟，不是实测曲线；右边那条只叫「就近舍入（对照）」，不代表任何一家硬件。",
    "GPU 张量核：每 «TC_GROUP» 个一组，抹零。第 1 步：一组 «TC_GROUP» 个乘积连同累加器，按最大的指数对齐；保留最高位后 «TC_KEEP» 位，窗口外的位抹掉。"
    "抹零一列每一行都是少；就近舍入（对照）一列两行少、两行多。第 1 步完成：窗口外的位，抹零次次少；就近舍入有多有少。"
    "第 2 步：很多组接着加（模拟，合成正数输入）。纵轴相对误差，横轴加了多少个乘积，一直加到 «TC_K3» 个：抹零的相对误差一路往下到约 «TRUNC_SIM_END_T»，就近舍入（对照）停在 0 附近，放大 80 倍才看得见它围着 0 晃，终点约 «RNE_SIM_END_T»。模拟，口径与实测曲线不同。"
    "第 2 步完成：抹零一路往下；就近舍入在 0 附近晃。抹零只少不多：求和越长，偏得越多。")
ANIMS["__ANIM_LINEAGE__"] = ("anim-lineage", "topic08-lineage.mp4", "GoldenLineage",
    "同一个 TPU 被测、同一把尺子：golden 每加一样东西，同一根柱子就缩一截，缩掉的那截换成标了名字的「已查清」色块。",
    "色块长度是带符号均值之差，不是 §9.3 拆账图的百分比。",
    "golden 是一步步改出来的：从框架默认出发，先对齐超参、数据和并行，再每一步只加一样东西。"
    "① 起点：框架默认配置（BF16、不重算、TF32 开），第 1 步那几级对它比，1000 步验收不用它；它不重算，跟生产配置不一样，没有柱子。"
    "② 生产近似：加整层重算和路由回放。同一个 TPU 被测对它：参数范数差 «GOLDEN_R2_T» × N，远超 2×N。"
    "③ 只修一处：加输出头 dgrad 换成 FP32（一次累加，也不全准）。同一根柱子缩到 «GOLDEN_R3_T» × N；缩掉的那截查清了，是 GPU 输出头 dgrad 截断（已查清）。"
    "④ 精确轨迹：加 BF16 权重、全程 FP32、真关 TF32。再缩到 «GOLDEN_R4_1_T» × N；缩掉的那截是 GPU 其余的 BF16 误差（已查清）。仍超过 2×N，不过。三根柱子的数后面都跟着红字「不过」；红块是 dgrad 截断，橙块是 GPU 其余 BF16，蓝段是剩下那截。③ 卡片下注：定案前的尺子取自这组重跑。"
    "被测是 FP32 累加开关打开前那一版 TPU；当前版对 ④ 是 «ACC_PN_R» × N。"
    "每往精确值靠一步，缩掉的都是 GPU golden 自己的偏差；这截也拆清了；最后定案用 ⑤（② ＋ 分段累加），TPU 全部在噪声内。蓝段旁边写着：剩下这截：④ 自己偏 ＋ TPU 舍入，两笔相减。"
    "柱长是同一个 TPU 被测的参数范数带符号均值除以 N；N 是 ③ 那组 GPU 重跑 10 对的最大值。")
ANIMS["__ANIM_WINDOWED__"] = ("anim-windowed", "topic08-windowed.mp4", "WindowedAcc",
    "同一个往里投的投影：4 个设备先收齐、一次乘完，舍 1 次；8 个设备边传边乘，红块一格格冒出来，舍 6 次；打开投影 FP32 累加，又只舍 1 次。",
    "循环体结构取自编译产物；片从顶上落下只表示这一段用到哪几片，不画环上谁传给谁。",
    "设备一到 8 个，一次矩阵乘被拆成几段。同一个往里投的投影，一次求和 2816 个乘积；红块是一次舍到 BF16。顶上一排是 8 片权重。"
    "第 1 步　4 个设备：8 片先收齐，一次乘完，最后舍 1 次，跟 GPU 一样。"
    "第 2 步　8 个设备：边传边乘，分两轮，每轮两段 704 宽的矩阵乘（片 1＋2、片 3＋4……），每段输出舍一次，加进 BF16 累加器再舍一次；右边的计数涨到舍 6 次。"
    "第 3 步　打开投影 FP32 累加：结构不变，中间全留 FP32，只在最后舍 1 次。"
    "第 4 步　同一种切法，只是设备数变了。单算子 R：«M3_PROJ_R_T» → «M3_PROJ_R_AFTER_T»（实测）；离线编译：4 个设备 0 个这种循环，8、16 个设备各 20 个。"
    "在 4 个设备上验精度，看不到这件事。")
ANIMS["__ANIM_DECLARED__"] = ("anim-declared", "topic08-declared.mp4", "DeclaredElided",
    "同一个值走同一串盒子：代码的本意是舍，在一颗融合里没舍，写回内存才舍；写成强制舍入才真舍。",
    "值是随手取的 FP32 数，BF16 当场算；打包条数是最小复现的实测。",
    "代码说「这里舍一次」，编译器听了吗？四个盒子：4 路加权和 FP32、转 BF16、转回 FP32、算均方。"
    "第 1 步　代码的本意：经过「转 BF16」就舍一次，后面用的是舍过的数。0.72314453 经过转 BF16 变成 0.72265625。"
    "第 2 步　实际：这一串在同一颗融合里，编译器把往返省掉，均方用的是没舍的数。转 BF16 被打叉，值保持 0.72314453；后端指令里打包成 BF16 的：«EL_IHC_P0» 条。"
    "第 3 步　同一个值还要写回内存、交给下一颗算归一化输出：这一路真舍了。写进内存（BF16）时变成 0.72265625，交给下一颗：归一化输出。两颗 kernel 用的不是同一个数。"
    "第 4 步　写成强制舍入（reduce_precision）：编译器省不掉，融合里也真舍。转 BF16 换成强制舍，值变成 0.72265625；打包成 BF16 的：«EL_IHC_FIXED» 条。")
ANIMS["__ANIM_SCAN__"] = ("anim-scan", "topic08-scan.mp4", "ScanSweep",
    "一格一格扫过整个程序，同一个模式亮起来就按位置归堆；只含往返的 kernel 拿 LLO 一颗颗数，全是 0 条；关掉省舍入，没有一颗是 0。",
    "上面那条「程序」只画 60 格，哪几格亮是示意；各堆的终值、41 颗和 35 颗是实测。",
    "先编译，再扫描：同一个模式，在哪儿出现就在哪儿亮。整个训练步编出来的程序（示意：60 格）。"
    "第 1 步　找「同一颗融合里转成 BF16、又转回 FP32」：一格一格扫，亮了就按位置归堆。A 分支输入梯度合流 «EL_BRANCHIN» 处，B 每微批 dW «EL_WGRAD» 处，C 前向几处 «EL_FWD» 处，其余（优先级低）«EL_MINOR» 处，索引器掩码（无害）9 处；已补上，跳过：«EL_COVERED» 处。一共 «EL_TOTAL» 处往返（按使用次数）；三根橙色是先查的范围。"
    "第 2 步　整颗只含往返的 kernel 拿 LLO 一颗颗数：打包成 BF16 的指令都是 0 条。省舍入开：只含往返的 «EL_K_TOTAL» 颗。"
    "第 3 步　负对照：关掉省舍入再编，同类 kernel 没有一颗是 0 条，「0 条」确实是省舍入造成的。省舍入关：同类 «EL_K_NOEXC» 颗，0 条的 0 颗。")
ANIMS["__ANIM_CANCEL__"] = ("anim-cancel", "topic08-cancel.mp4", "CancelScale",
    "拨一个开关，指针往意料之外的方向跑：只修甲几乎回零，再修乙反而往右跑回去；GPU 上只拿掉乙，GPU 自己被往下压。",
    "读数是实测；「三个毛病、乙丙抵掉、丙露出来」是推断。只修乙那一格单看分不开，没放进动画（静态图里有）。",
    "一台秤三个毛病：修好一个，读数反而偏了。读数：第 551–1000 步主 loss 偏差，乘 1e-6；上面一根尺子是 TPU 对 GPU，下面一根是 GPU 自己（对 GPU 其余各份）。三个毛病：甲 流梯度（偏重）、乙 收拢处梯度（偏轻）、丙 还没找到（偏重），各标没修或已修，丙是问号。"
    "第 1 步　都没修（修之前）：读数 9.4，是三股相抵后的数。第 2 步　只修甲：几乎回到零，乙和丙大致抵掉，看着像修好了，«CX_STREAM_V»。"
    "第 3 步　再修乙：指针反而往右跑，高了 «CX_UP_T»（p 0.024），到 «CX_BOTH_T» —— 被乙遮住的丙露出来（推断）。"
    "第 4 步　GPU 上只拿掉乙那次舍入：GPU 自己被压低 2.9，方向一致，是那 «CX_UP_T» 的一部分（推断）。")
ANIMS["__ANIM_FLIP__"] = ("anim-flip", "topic08-flip.mp4", "RouteFlip",
    "输入只差一丝，近平局的 token 就换了专家；分到 token 少的专家，第 1 步梯度差被放大到 «STEP1_EXPERT»。",
    "条的高度、每个分数挪动的量和放大框里的分差是示意（取在实测中位数范围内）；其余数字是实测。",
    "MoE 路由的近平局翻转：差一丝，就换一个专家。"
    "① 一个 token 给 «EXPERTS» 个专家打分，取前 «TOPK» 名：第 8、9 名几乎一样高。排名用分数加偏置；选中后加权用不带偏置的分数，偏置只管选谁。"
    "② 上游注意力两边差一点点，分数挪一丝，第 8、9 名对调。路由的输入（注意力之后）两边差 rel_L2 «ROUTE_IN_RELL2_LO» 到 «ROUTE_IN_RELL2_HI»，约是 BF16 舍入量级。放大看两根条的顶端（约 2000 倍）：B 只比 A 低 3e-5；挪动以后 A 反比 B 低 1.5e-5，A 落选，B 入选。翻转 token 的第 8、9 名分差中位数 «FLIP_MARGIN»；全部 token 的中位数 «ALL_MARGIN_LO» 到 «ALL_MARGIN_HI_T»。"
    "③ 这个 token 的那一份输出，整块换了一个专家来算：GPU 由专家 A 算，TPU 换成专家 B，不是差一丝，是换了一个人。分到 token 少的专家，进出一个 token 就是它梯度的一大块：第 3 层专家 0 在这一批 «BATCH_TOKENS» 个 token 里只分到 «E0_TOKENS» 个，TPU 上换出 «E0_OUT» 个、换进 «E0_IN» 个。第 1 步梯度对 GPU 的 rel_L2：这个专家的权重梯度 «STEP1_EXPERT»，比到的 «STEP1_TENSORS» 个张量合起来 «STEP1_ALL_E»，约差 «STEP1_X» 倍；估算 根号（换掉的除以分到的）＝ 根号（«E0_SWAPPED»/«E0_TOKENS»）约 «E0_SQRT_T»，同量级。GPU 自己重跑，这些张量的差几乎是 0。"
    "④ 对照：两边喂完全相同的路由输入，3 层乘 «BATCH_TOKENS» 个 token，翻转 0（logits 两边仍差 «ROUTE_SAME_IN_DIFF»，但一个都没翻）；路由器的输入换成 TPU 自己算出来的（差 «ROUTE_IN_RELL2_LO»），每层翻 «FLIP_L2_T» 的 token，翻的多是近平局。根子在输入差，不在路由器自己的精度。"
    "⑤ GPU 自己重跑第 1 步：前向确定，路由一个不变；拿 GPU 重跑当噪声尺，路由翻转在这把尺上是 0 宽度。«STEP1_EXPERT» 要按专家拆开看，不是 TPU 算错了梯度。近平局加一丝输入差，等于选了另一个专家。")


def _anim_html(aid, mp4, scene, cap, sub, aria, first=False):
    st = json.load(io.open(os.path.join(HERE, "manim", "steps", scene + ".json"), encoding="utf-8"))
    assert os.path.exists(os.path.join(OUT_DIR, "media", mp4)), "缺动画 %s" % mp4
    pauses = ",".join("%.2f" % t for t in st["pauses"])
    return ('<figure class="fbox fwide" id="%s">\n<video src="media/%s" autoplay loop muted playsinline data-pauses="%s"\n'
            '       aria-label="%s"></video>\n<figcaption><b>%s</b>\n'
            '  <span class="sub">（%d 秒，Manim 渲染%s%s）</span></figcaption></figure>'
            % (aid, mp4, pauses, aria.replace('"', "&quot;").replace("§", "&#167;"), cap, round(st["duration"]),
               # 操作说明只在页面上第一支动画写全，其余只提一句（避免五支动画重复同一句）
               "；默认连续播放；取消勾选「连续播放」、点「⟲ 重来」，再点「开始」一步一步看。" if first
               else "，可一步一步看。", sub))


_html = head + HERO + FOUND + BODY + FOOT
# «名字» ＝ topic08_numbers 里的常数，按 N.show 的写法落进正文（数字只有一个来源）
def _numsub(s):
    return re.sub(r"«([A-Z0-9_]+)»", lambda m: N.show(getattr(N, m.group(1))), s)


_html = _numsub(_html)
_first_anim = min(ANIMS, key=_html.find)
for _ph, _a in ANIMS.items():
    assert _html.count(_ph) == 1, "%s 在正文里出现 %d 次" % (_ph, _html.count(_ph))
    _html = _html.replace(_ph, _anim_html(*[_numsub(_x) for _x in _a], first=(_ph == _first_anim)))
if ANIMS:
    _html = _html.replace("</body>", _STEP_JS + _RM_JS + "</body>")
_html = P.place_figs(_html, {_k: _v[:3] + (_numsub(_v[3]),) for _k, _v in FIGS.items()})   # 图注里的 «名字» 也落地
assert "«" not in _html and "»" not in _html, "还有 «名字» 占位符没落地"
_leak = sorted(set(re.findall(r"__[A-Z][A-Z_0-9]*__", _html)))
assert not _leak, "占位符没落地：%s" % "、".join(_leak)
assert _html.count("<!--") == _html.count("-->"), "HTML 注释没闭合：<!-- %d 个、--> %d 个（没闭合的注释会把后面整页吞掉）" % (_html.count("<!--"), _html.count("-->"))
for _tag in ("h2", "h3", "section", "div", "details", "table"):
    _o = len(re.findall(r"<%s[ >]" % _tag, _html))
    _c = _html.count("</%s>" % _tag)
    assert _o == _c, "<%s> 开 %d 个、闭 %d 个" % (_tag, _o, _c)
# ⭐ 正文里手打的关键数，必须在对应的图里也出现（图是脚本现算的）—— 照专题五的 _NUM_LOCK。
_NUM_LOCK = [("16,000", "fig8-replay.svg"), ("0.033", "fig8-wd.svg"), ("3.45", "fig8-judge.svg"), ("1.92", "fig8-judge.svg"), ("1.29", "fig8-judge.svg"), ("1.14", "fig8-judge.svg"),
             ("1.235", "fig8-rratio.svg"), ("1.41", "fig8-windowed.svg"), ("1.41", "fig8-inout.svg"), ("0.127", "fig8-drift.svg"),
             ("17.1", "fig8-rows.svg"), ("6.5e-4", "fig8-gauge.svg"), ("6.5e-04", "fig8-head.svg"),
             ("0.36%", "fig8-swap.svg"), ("+0.0182", "fig8-exact.svg"), ("0.0207", "fig8-decomp.svg"),
             ("28", "fig8-regress.svg"), ("4.3e-3", "fig8-captain.svg")]
for _n, _svg in _NUM_LOCK:
    _g = io.open(os.path.join(HERE, _svg), encoding="utf-8").read().replace(",", "").replace("−", "-")
    assert _n.replace(",", "") in _g, "%s 里没有 %s —— 图脚本改了，正文没跟上" % (_svg, _n)
for _n in ("16,000", "0.033", "3.45", "1.235", "0.127", "17.1", "2.07e-2"):
    assert _n in _numsub(BODY), "正文里已经没有 %s 了 —— 从 _NUM_LOCK 里删掉这一条" % _n
# 公开仓库闸门的一道兜底：源码里不许出现内部名字（闸门只挡产物，源码本身也公开）。
#   ⛔ 禁词表本身也不能明文写在公开源码里，所以只存 sha256 前 12 位和长度，滑窗比对。
import hashlib  # noqa: E402
_BANNED = [("850ca8e0ec3f", 5), ("5c58c51f38f6", 8), ("7853225b8ee4", 2), ("53874092e000", 2),
           ("1637f2d1f2f6", 10), ("e134c033e88f", 7), ("f09761d7a79c", 12), ("c68c12825750", 5)]
import glob  # noqa: E402
_SRC = "\n".join(io.open(_f, encoding="utf-8").read() for _f in sorted(glob.glob(os.path.join(HERE, "topic08*")) + glob.glob(os.path.join(HERE, "manim", "topic08-anim-*.py"))))
# 唯一的例外：公开版仓库钉在完整 sha 上的链接（只认这一个前缀；源码里拼成 PUB_REPO 也一并认）
_SRC = _SRC.replace('PUB_REPO = "https://github.com/yangwhale/" + "jan" + "us"', "")
_SRC = _SRC.replace("yangwhale/" + "jan" + "us", "")   # 讲义里写仓库全名的地方（只认 owner/仓库 这一个全名）
_hits = set()
for _n in sorted({n for _h, n in _BANNED}):
    _want = {h for h, n in _BANNED if n == _n}
    for _k in range(len(_SRC) - _n + 1):
        if hashlib.sha256(_SRC[_k:_k + _n].encode("utf-8")).hexdigest()[:12] in _want:
            _hits.add(_SRC[_k:_k + _n])
assert not _hits, "源码里出现了禁词（%d 处）%s" % (len(_hits), os.environ.get("T8_SHOWBAN") and _hits or "")
_html = P.add_figonly_toggle(_html)
P.finish(_html, OUT, SECTIONS, "topic-08.html")
