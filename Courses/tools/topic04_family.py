# -*- coding: utf-8 -*-
r"""专题四这一家两份材料的**互跳导航 ＋ 分工说明** —— 一份定义，两边共用。

⛔⛔ 2026-09-23 现场：「你这个文档一开始为什么没有给讲义的那个跳转链接？
  那有人想去读的话，上哪去找去？你把这些东西在一开始都说明白，
  那个教材里是什么、讲义里是什么，让大家心里有数，不然的话，讲义不是白写了？」

⭐ 查下来是**单向的**：讲义顶部早就有一条 tab 指回课件（而且指了 8 处锚点），
  课件这边**一个字都没提讲义**。
  ⛔ 而这种缺失最阴的地方在于：**它在写的那一侧完全看不出来。**
    写讲义的人当然知道有课件，所以他自然会加回链；
    写课件的时候讲义还不存在，加不了 —— 之后也没人回头补。
  ⭐⭐ 判据：**一份材料派生出第二份时，回头给第一份补入口。**
    新的那份一定会指向旧的（它是从旧的长出来的），
    反方向永远要专门去做一次，而且漏了不报错。

⭐ 沿用 `topic02_family.py` / `topic03_family.py` 那一套：清单只有 `PAGES` 一处。
📌 `nav.famnav` 的样式在课件页里已经有（它的 <head> 来自 `topic03_page.CSS_SRC`）；
  讲义那份 <head> 抄的是专题一讲义，**没有这段样式**，所以这里额外提供 `CSS`
  给讲义注入。⛔ 别在讲义脚本里另写一份。
"""

# (文件名, 短名, 一句话定位)
# ⛔ 顺序 = 显示顺序。第一项是给学员的那份。
# ⭐ 两份的分工不是「长短」，是**读者不同**：
#   · 教材是给**学员**读的：完整推导、口径、出处、表格，可以停下来查。
#   · 讲义写的是**怎么把这些知识讲进一个人脑子里**：台词、屏幕切到哪一格、
#     每段几分钟、哪句是高点、常见误解长什么样、被问到了拿什么答。
#     ⭐⭐ 所以它最值钱的用法不是「老师照着念」，是**整份丢给 AI** ——
#       你会得到一个既有全部背景口径、又知道该怎么讲的助教，让它带你自学。
#     ⛔ 判据（2026-09-23 现场纠的）：**介绍一份材料，写它能给读者什么，
#       不要写它当初为谁而写。** 这两个在作者脑子里是一回事，在读者那儿不是 ——
#       按「给讲课的人用的」这个说法，不打算上台的人看一眼就划走了，
#       而他恰恰是最该拿走它的人。
#     它也不是教材的摘要 —— 它按「台上怎么说」重排，很多话在纸上根本不该出现。
PAGES = (
    ("topic-04.html",         "教材（本讲正文）", "给学员读 · 推导 / 口径 / 出处"),
    ("topic-04-lecture.html", "讲义（授课稿）",   "整份丢给 AI ＝ 一个会讲课的助教"),
)

CSS = """
/* 专题四两份材料的互跳条 —— 定义在 tools/topic04_family.py。
   ⛔ 课件页不需要这段（它的 head 里已经有了），这份只注给讲义。 */
nav.famnav{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;
  margin:0 0 20px;font-size:13px}
nav.famnav .famlab{align-self:center;color:#80868b;margin-right:2px;
  white-space:nowrap}
nav.famnav a,nav.famnav .here{display:block;padding:7px 13px;border-radius:8px;
  border:1px solid #dadce0;background:#fff;text-decoration:none;line-height:1.45}
nav.famnav a{color:#1a73e8}
nav.famnav a:hover{border-color:#1a73e8;background:#f8fbff}
nav.famnav .here{color:#202124;font-weight:700;border-color:#1a73e8;
  background:#e8f0fe;cursor:default}
nav.famnav em{display:block;font-style:normal;font-weight:400;font-size:11.5px;
  color:#80868b;margin-top:2px}
nav.famnav .here em{color:#174ea6}
"""


def nav(current):
    """渲染互跳条。`current` 渲染成不可点的高亮项（同时充当「你在这里」）。

    ⛔ 传错文件名直接断言失败，不静默降级成「两个都可点」——
      那种页面看上去完全正常，只是点了没反应。
    """
    names = [f for f, _, _ in PAGES]
    assert current in names, \
        "%r 不在 PAGES 里 —— 新页要先在 topic04_family.py 登记" % current
    out = ['<nav class="famnav" aria-label="专题四的两份材料">',
           '<span class="famlab">这一讲有两份材料：</span>']
    for f, short, sub in PAGES:
        if f == current:
            out.append('<span class="here">%s<em>%s</em></span>' % (short, sub))
        else:
            out.append('<a href="%s">%s<em>%s</em></a>' % (f, short, sub))
    out.append('</nav>')
    return "".join(out)
