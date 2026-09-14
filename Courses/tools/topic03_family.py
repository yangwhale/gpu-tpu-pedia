# -*- coding: utf-8 -*-
r"""专题三这一家两页的**互跳导航** —— 一份定义，两页共用。

⭐ 跟 `topic02_family.py` 是同一套东西、同一个理由：清单只有 `PAGES` 这一处。
  ⛔ 两页各写一份链接列表，加第三页时一定会漏掉其中一处，而漏掉不报错 ——
    读者只是在某一页上跳不到新的那页，没人会发现。

📌 CSS 不在这儿：`nav.famnav` 的样式写在 `topic02_family.CSS` 里，注入
  `topic-02-L300.html` 的 <style>；而专题三两页的 <head> 正是从那份整块切下来的
  （`topic03_page.CSS_SRC`），所以样式自动就有。
  ⛔ 不要在这里再写一份 —— 那就又变成两个载体了。
"""

# (文件名, 短名, 一句话定位)
# ⛔ 顺序 = 页面上的显示顺序，由浅入深。
# ⭐⭐ 2026-09-14 R62 一分为二。现场原话：
#   「把现在的这个专题三改成专题三的 L300，然后从 L300 里边一点一点地蒸馏，
#     要最精华的部分，写一个专题三……重点就是要写一篇完整的故事。」
#   📌 所以两页的分工不是「长短」，是**体裁**：
#     · L200 是**一条故事线** —— 从 RNN 一路走到今天的混合配比，
#       每一步只回答「上一步欠下了什么，这一步拿什么还」。多图、少字。
#     · L300 是**档案** —— 全部推导、消融表、一手出处、我们自己的实测。
#   ⛔ 判据：**L200 不是 L300 的摘要。** 摘要是把每段压短，故事是重排因果。
#     一句话如果只在「按时间读」时才成立，它属于 L200；
#     一句话如果要摊开算才站得住，它属于 L300。
PAGES = (
    ("topic-03.html",      "主线 L200", "一条故事线 · 多图少字"),
    ("topic-03-L300.html", "主线 L300", "完整版 · 推导与出处"),
)


def nav(current):
    """渲染这一家两页的互跳条。`current` 会被渲染成不可点的高亮项。

    ⛔ 传错文件名直接断言失败，不静默降级成「两个都可点」——&nbsp;
      那种页面看上去完全正常，只是点了没反应。
    """
    names = [f for f, _, _ in PAGES]
    assert current in names, "%r 不在 PAGES 里 —— 新页要先在 topic03_family.py 登记" % current
    out = ['<nav class="famnav" aria-label="专题三的两个版本">',
           '<span class="famlab">专题三共两版：</span>']
    for f, short, sub in PAGES:
        if f == current:
            out.append('<span class="here">%s<em>%s</em></span>' % (short, sub))
        else:
            out.append('<a href="%s">%s<em>%s</em></a>' % (f, short, sub))
    out.append('</nav>')
    return "".join(out)
