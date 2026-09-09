# -*- coding: utf-8 -*-
r"""专题二这一家四页的**互跳导航** —— 一份定义，四页共用。

════════════════════════════════════════════════════════════════════
⭐ 为什么要单独一个模块，而不是各页各写一行
════════════════════════════════════════════════════════════════════
四页里有**三个不同的产出方式**：

  · topic-02.html        ← topic02-build-L200.py 生成
  · topic-02-L300.html   ← 手维护源文件，topic02-port-microscope.py 原地补丁
  · topic-02x.html       ← topic02x-build.py 生成
  · topic-02x-L200.html  ← topic02x-build-L200.py 生成

⛔ 如果四处各写一份链接列表，**再加第五页时一定会漏掉其中一两处**，
  而漏掉不报错 ——&nbsp;读者只是在某一页上跳不到新的那页，没人会发现。
⭐ 所以清单只有 `PAGES` 这一处；`nav(current)` 负责把「当前这页」渲染成
  不可点的高亮项，其余渲染成链接。

════════════════════════════════════════════════════════════════════
⭐ CSS 只写进 topic-02-L300.html 的 <style>，另外三页自动继承
════════════════════════════════════════════════════════════════════
那三个生成器都是 `head = _src[:_src.index("</style>")+8]` ——&nbsp;
**从 L300 整块切 head**。所以样式加在 L300 一处，四页全有。
⛔ 不要在各页自己的 `<style>` 追加一份，那就又变成四个载体了。

════════════════════════════════════════════════════════════════════
⛔ 加新页时要改的地方（就一处，但别漏 build-all.sh）
════════════════════════════════════════════════════════════════════
① 在下面 `PAGES` 里加一行
② 如果它是新的生成器，记得 `from topic02_family import nav` 并插进 header
③ build-all.sh 的产物清单
"""

# (文件名, 短名, 一句话定位)
# ⛔ 顺序 = 页面上的显示顺序。先主线后外传，各自由浅入深。
PAGES = (
    ("topic-02.html",       "主线 L200", "精讲 · 80 分钟"),
    ("topic-02-L300.html",  "主线 L300", "完整版 · 两小时"),
    ("topic-02x.html",      "外传 L100", "19 分钟 · 含十模型"),
    ("topic-02x-L200.html", "外传 L200", "精讲 · 含显微镜拆解"),
)

# ⭐ 这段只注入 topic-02-L300.html 的 <style>，其余三页从它切 head 时自动带上。
CSS = """
/* ══ 专题二一家四页的互跳导航（定义在 tools/topic02_family.py）══
   ⭐ 样式只写这一处：另外三页的 <head> 是从本文件整块切下去的。
   ⛔ 别在各页自己的 <style> 里再写一份。 */
nav.famnav{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;
  margin:0 0 24px;font-size:13px}
nav.famnav .famlab{align-self:center;color:var(--gray2,#80868b);
  margin-right:2px;white-space:nowrap}
nav.famnav a,nav.famnav .here{display:block;padding:7px 13px;border-radius:8px;
  border:1px solid var(--line,#dadce0);background:#fff;text-decoration:none;
  line-height:1.45;transition:border-color .12s,background .12s}
nav.famnav a{color:var(--blue,#1a73e8)}
nav.famnav a:hover{border-color:var(--blue,#1a73e8);background:#f8fbff}
/* ⭐ 当前页渲染成不可点的高亮块 ——&nbsp;它同时充当「你在这里」的指示，
   所以不需要再单独放一个面包屑说明当前在哪一版。 */
nav.famnav .here{color:var(--ink,#202124);font-weight:700;
  border-color:var(--blue,#1a73e8);background:#e8f0fe;cursor:default}
nav.famnav em{display:block;font-style:normal;font-weight:400;font-size:11.5px;
  color:var(--gray2,#80868b);margin-top:2px}
nav.famnav .here em{color:#174ea6}
@media(max-width:760px){nav.famnav em{display:none}
  nav.famnav a,nav.famnav .here{padding:6px 10px}}
"""


def nav(current):
    """渲染这一家四页的互跳条。`current` 是本页文件名，会被渲染成不可点的高亮项。

    ⛔ 传错文件名不会静默降级成「四个都可点」——&nbsp;直接断言失败。
      因为「当前页也是个链接」这种页面看上去完全正常，只是点了没反应，
      属于**不报错也不难看、只在读者手上失效**的那一类。
    """
    names = [f for f, _, _ in PAGES]
    assert current in names, "%r 不在 PAGES 里 —— 新页要先在 topic02_family.py 登记" % current
    out = ['<nav class="famnav" aria-label="专题二的四个版本">',
           '<span class="famlab">专题二共四版：</span>']
    for f, short, sub in PAGES:
        if f == current:
            out.append('<span class="here" aria-current="page">%s<em>%s</em></span>'
                       % (short, sub))
        else:
            out.append('<a href="%s">%s<em>%s</em></a>' % (f, short, sub))
    out.append('</nav>')
    return "\n  ".join(out)
