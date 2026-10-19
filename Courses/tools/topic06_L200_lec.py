# -*- coding: utf-8 -*-
r"""专题六 L200 讲义：十节口播稿。每节一项 (标题, 课件小节 id, 主图 id, 正文, 别讲什么)。

⭐ 口播稿规矩（reviews/T6-L200-STATE.md 判据 5）：短句、先给直觉再讲机制、少报符号名；
   互动＝抛题、停一下、揭晓。一块 <p class="say"> 对应一个动作，不超过 200 汉字。
🚧 还没写的节先放计划框（不进地板），写好一节替换一节。
"""
import importlib
from topic06_L200_spine import SPINE


def _todo(row):
    num, q, line, mins, figs, src = row
    return ('<p>🎯 <b>这一节要留下什么</b>：%s</p>\n'
            '<p>🚧 <b>口播稿还没写。</b>打算用的图：%s。</p>' % (line, figs))


def _ch(i, r):
    title = "第%s节　%s（%d′）" % (r[0], r[1], r[3])
    try:
        m = importlib.import_module("topic06_L200_lec_s%d" % i)
    except ModuleNotFoundError:
        return (title, "s" + r[0], None, _todo(r), "（待写）")
    return (title, "s" + r[0], m.FIG, m.BODY, m.DONT)


CH = [_ch(i, r) for i, r in enumerate(SPINE)]

PLAN = [("%s　%s" % (r[0], r[1]), r[3], "—", r[2]) for r in SPINE]

JUDGED = ("「铺开换来三样", "「车上的人是怎么数的", "「还记得第一节那个洞", "「那并发 512 呢", "「256 个专家，144 张卡")   # 台词数字偏密但「数就是内容」的块（填开头几个字）

TIME_NOTE = """<p>⛔ <b>第零、一节不能砍。</b>
  <em>第零节立住「decode 每一步都搬一趟」，第一节算清这一趟为什么贵；后面每节开头都拿它接。</em><br>
  全讲十节约 90 分钟，不中场休息。想深挖哪一节，课件每节末尾都有通往完整版（L300）的链接。<br>
  台词里的 4K、128K 这类长度，K 读作「千」。<br>
  下面的地板含互动约 86 分钟：地板只按每分钟 200 字念稿算，指图、等回答、接追问都不在里面。每节都比计划少半分钟左右，台下问得多就放在那里答。第零节开头那道回顾题只给一分钟，别展开讲专题五。</p>"""

THIN = """<h3 class="sec">🚧 这份讲义还薄的地方</h3>
<div class="say">
  <ul>
    <li>[x] <b>R13 已从头串讲一遍</b>：交棒对齐、比方统一到公交车（第六节厨房、第七节徒弟是明说换比方）、计划按地板重分。</li>
    <li>[x] <b>R14 外行和推理专家读过零到四节</b>：补了层、头、KV 头、Little 定律怎么数人、2.09 倍和 1.31 倍两种分母。</li>
    <li>[x] <b>R15 外行和推理专家读过五到九节</b>：补了加权平均、温度与抽样、量化的容量和每步字节两笔账、编译为什么要提前做、5.1 倍的分母，各节补到接近计划分钟。</li>
    <li>[x] <b>R24 补回零到四节的「为什么」</b>：第零节开头加专题五回顾题，五到九节逐句顺了口语，各节含互动都在计划减 0.3 到 1 分钟之间。</li>
    <li>[ ] <b>R15 之后还没从头到尾串讲计时</b>；进度见 <code>Courses/reviews/T6-L200-STATE.md</code>。</li>
  </ul>
</div>"""
