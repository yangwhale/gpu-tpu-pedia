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

JUDGED = ()   # 台词数字偏密但「数就是内容」的块（填开头几个字）

TIME_NOTE = """<p>⛔ <b>第零、一节不能砍。</b>
  <em>第零节立住「decode 每一步都搬一趟」，第一节算清这一趟为什么贵；后面每节开头都拿它接。</em><br>
  全讲十节约 90 分钟，不中场休息。想深挖哪一节，课件每节末尾都有通往完整版（L300）的链接。<br>
  台词里的 4K、128K 这类长度，K 读作「千」。</p>"""

THIN = """<h3 class="sec">🚧 这份讲义还薄的地方</h3>
<div class="say">
  <ul>
    <li>[ ] <b>2026-10-19 起新写，十节口播稿逐轮补。</b>进度见 <code>Courses/reviews/T6-L200-STATE.md</code>。</li>
  </ul>
</div>"""
