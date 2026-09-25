# -*- coding: utf-8 -*-
r"""专题三这一家两页**共用的页面脚手架** —— head / 图装配 / 锚点 / 吸顶目录。

════════════════════════════════════════════════════════════════════
⭐ 为什么要单独一个模块（2026-09-14 R62 立）
════════════════════════════════════════════════════════════════════
这一天现有的专题三整体降格成 **L300（完整版）**，旁边新起一份
**主线 L200** —— 一条故事线、多图少字，是要拿上讲台的那一份。
于是同一套脚手架要服务两个生成器：

  · `topic03-build-L300.py`  →  `topic-03-L300.html`
  · `topic03-build-L200.py`  →  `topic-03.html`

⛔ 如果把这三百行在两个生成器里各抄一份，**它们一定会漂**，而且漂了不报错
  ——&nbsp;只会让两页的锚点规则、出处折叠、吸顶目录慢慢长成两个样子。
  这个仓库在「同一份东西存两处」上已经栽过四次：famnav 三个版本、
  fig 登记表漏掉 12 个、退役说法同时住在图 / 图注 / 讲稿三处、
  §三 一删五个论文节号当场变红。
⭐ 所以：**脚手架只有这一份，两个生成器只负责自己的 BODY。**

📌 下面每一段都是从原 `topic03-build.py` **原地搬过来的**，不是重写 ——
  连注释里那些踩坑记录一起搬，因为那些坑对两页同样有效。
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-02-L300.html")


# ⛔⛔ 2026-09-07 修一个**静默了很久**的 bug：整个 <head> 是从专题二 L300 搬的，
#    原先靠一句 `head.replace("<title>TPU 与 GPU", "<title>注意力演进")` 改标题。
#    **那个模式根本匹配不上** ——&nbsp;L300 的真实标题是
#    `<title>专题二 · TPU 与 GPU（L300 · 完整版）</title>`，中间多了「专题二 · 」。
#    于是这一页在浏览器标签上一直显示「专题二 · TPU 与 GPU（L300 · 完整版）」，
#    og:title / og:description / og:url / og:image 也全是专题二的 ——&nbsp;
#    **分享出去的卡片标题、摘要、跳转全指向另一讲。**
#
# ⭐⭐ 形状：**`str.replace` 匹配不上时是静默的**，不报错、不返回失败标志，
#    就是原样返回。**「改了」和「没改成」在代码里长得一模一样。**
#    ⛔ 判据：**任何「替换模板里某一段」的操作，都必须断言它真的改了。**
#    下面 `_sub()` 就是干这个的 —— 换不到直接让构建挂掉。
#
# 📌 顺带发现 `img/og-topic-02.jpg` **这个文件根本不存在**（img/ 里只有
#    og-topic-01.jpg）。所以那条 og:image 是死链，专题二 L300 自己也带着。
#    这一页直接把 og:image 摘掉 ——&nbsp;**没有图，好过指一张 404 的图**。


def _sub(text, pattern, repl, what):
    """替换 + 断言真的替换了。⛔ 不要退回裸的 str.replace。"""
    new, n = re.subn(pattern, repl, text, count=1)
    assert n == 1, "改不动 %s —— 模板变了？（模式：%s）" % (what, pattern)
    return new


def make_head(title, og_title, og_desc, out_name):
    """切一份 <head> 出来。`out_name` 只用来拼 og:url，不写文件。"""
    _src = io.open(CSS_SRC, encoding="utf-8").read()
    head = _src[:_src.index("</style>") + len("</style>")]
    head = _sub(head, r"<title>.*?</title>", "<title>%s</title>" % title, "<title>")
    head = _sub(head, r'<meta property="og:title" content="[^"]*">',
                '<meta property="og:title" content="%s">' % og_title, "og:title")
    head = _sub(head, r'<meta property="og:description" content="[^"]*">',
                '<meta property="og:description" content="%s">' % og_desc,
                "og:description")
    head = _sub(head, r'<meta property="og:url" content="[^"]*">',
                '<meta property="og:url" content="https://gist.higcp.com/'
                'Courses/WebPages/%s">' % out_name, "og:url")
    # ⛔ og:image 曾经指向一个不存在的文件（img/og-topic-02.jpg），整条摘掉。
    # ⭐⭐ 这一条**故意不用 _sub**：它是「有就清掉」，不是「必须换成什么」。
    #   ⛔ 判据：**「必须改到」用断言，「有就清理」不用。**
    head = re.sub(r'\s*<meta property="og:image"[^>]*>'
                  r'(\s*<meta property="og:image:(width|height)"[^>]*>)*',
                  "", head)
    # ⛔⛔ 查残留之前**先把注释剥掉** —— 护栏要查的是元数据（title / og:*），
    #   不是碰巧提到兄弟文件的散文。
    _probe = re.sub(r"/\*.*?\*/|<!--.*?-->", "", head, flags=re.S)
    assert "TPU 与 GPU" not in _probe and "topic-02" not in _probe, \
        "head 里还有专题二的残留"
    head += """
<style>
ul + p, ol + p, ul + div.note, ol + div.note, table + p { margin-top: 14px }
/* ⭐ pre / pre code 的样式**已经修在 L300 的 CSS 源里**了（2026-09-04），
   这里不再重复一份 —— 重复的 CSS 跟重复的正文是同一类问题。 */
h4 { margin:18px 0 6px; font-size:15px }

/* ⭐ 2026-09-08：提示框去彩底那段**已经改到共用 CSS 源里了**
   （topic-01.html 与 topic-02-L300.html），本讲不再单独覆盖 ——
   ⛔ 同一条规则留两份，迟早只改一份。 */

/* ── 图下面折起来的「出处与口径」（2026-09-14）────────────────
   ⭐ 收起来的时候只是一行灰字，跟图注拉开一档；展开才是那一堆脚注。
   ⛔ 宽度跟着 figure 走（figure 是 .fwide，最宽 1760px），但**正文限宽在
     1080px** —— 出处是拿来读的散文，不是图，所以这里自己收回版心，
     否则一行会拉到一米八长，眼睛跟不回来。 */
.figsrc{max-width:1080px;margin:6px auto 0;border-top:1px dashed var(--line);
        font-size:13.5px;color:var(--gray)}
/* ⭐ 跟 details.aside 用同一套三角记号，不另发明一种 —— 同一页里两种折叠
   长两个样子，读者要分别学一次。 */
.figsrc>summary{cursor:pointer;list-style:none;padding:8px 2px;
                font:600 12.5px/1.4 var(--mono);color:#9aa0a6;
                letter-spacing:.02em}
.figsrc>summary::-webkit-details-marker{display:none}
.figsrc>summary::before{content:"▸ "}
.figsrc[open]>summary::before{content:"▾ "}
.figsrc>summary:hover{color:var(--ink)}
.figsrc p{margin:0 0 7px;line-height:1.7}
.figsrc p:first-of-type{margin-top:4px}
.figsrc b{color:var(--ink);font-weight:600}
@media print{.figsrc{font-size:11px}}
/* ⭐⭐⭐ 2026-09-16 现场：「这一部分给我折叠起来，我光看后边那个有模型名称
   和年份的编年史就够了。」——&nbsp;说的是 §0 那张横轴时间线。
   ⛔ 折叠 ≠ 删掉：它是首尾呼应用的，第九章还要请读者滚回来看。
   ⭐ 跟 .figsrc 共用三角记号，同一页里不发明第二种折叠长相。 */
.foldfig{margin:18px auto;border:1px solid var(--line);border-radius:10px;
         background:#fafbfc;padding:2px 14px}
.foldfig>summary{cursor:pointer;list-style:none;padding:10px 2px;
                 font:600 14px/1.5 var(--sans);color:var(--gray)}
.foldfig>summary::-webkit-details-marker{display:none}
.foldfig>summary::before{content:"▸ "}
.foldfig[open]>summary::before{content:"▾ "}
.foldfig>summary:hover{color:var(--ink)}
.foldfig>summary b{color:var(--ink)}
.foldfig>summary .why{font-weight:400;color:#9aa0a6;font-size:13px}
@media print{.foldfig{border:none;background:none}
             .foldfig>summary{display:none}}
</style>"""
    return head


def _plain(s):
    """扒成纯文字：去标签、去实体、去空白与标点，只留下可比对的字。"""
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&#?\w+;", "", s)
    return re.sub(r"[\s，。、；：！？「」（）()·…——\-＝=／/]+", "", s)


def dup_caption_vs_fig(cap, svg, fid, n=14):
    """图注里有没有**整段照抄图内文字**。返回命中的片段列表。

    ⛔⛔ 2026-09-14 R62 立。起因：第一章三张图的图注，全都在复述图自己
      底下那条落点带 ——&nbsp;其中 fig-rnn-hw 那张，**图注的标题和图的落点带
      标题是同一个词**（「两头堵死」），渲染出来页面在原地结巴了一次。

    ⭐ 为什么原有的九条体检一条都没响：它们查的是**结构**（小节在不在、
      指针指得中不中、节号对不对），而这是**同一句话住在两个地方**。
      专题二 L200 那边有一条查重，但它比的是「正文 ↔ 图注」，
      **比不到「图注 ↔ 图内文字」** ——&nbsp;因为图内文字在 SVG 里。

    ⛔ 判据：**只有把图渲染出来、用眼睛看，才发现得了的问题，
      就该在构建时用代码钉住。** 这一条是靠截图发现的，下一章不会再靠运气。

    ⚠️ 只报告不中止 —— L300 那 58 张里有一批历史重复，现在整顿会churn
      一大片、也不该在改主线的同一轮里做。**先让它可见。**
    """
    c, s = _plain(cap), _plain(svg)
    if not c or not s:
        return []
    hits, i = [], 0
    while i + n <= len(c):
        if c[i:i + n] in s:
            j = i + n
            while j < len(c) and c[i:j + 1] in s:
                j += 1
            hits.append(c[i:j])
            i = j
        else:
            i += 1
    return hits


DUP_WARNED = []


def _runs(a, b, n):
    """a 里有哪些 n 字以上的连续片段整段出现在 b 里。返回极大片段列表。"""
    out, i = [], 0
    while i + n <= len(a):
        if a[i:i + n] in b:
            j = i + n
            while j < len(a) and a[i:j + 1] in b:
                j += 1
            out.append(a[i:j]); i = j
        else:
            i += 1
    return out


def lint_dup_body_vs_figs(html, n=14):
    """一句话有没有在这一页上被读者看见两遍。返回 [(片段, 哪儿撞的)]。

    ⛔⛔ 2026-09-14 R62 同一个毛病在一天之内换了**三个位置**出现：
      ① 图注复述图内文字（R1 抓到，于是有了 dup_caption_vs_fig）
      ② 正文复述图内文字（R2 抓到，于是有了这个函数）
      ③ **正文复述图注**（R3 抓到 ——&nbsp;第三章那把尺子，图注说了一遍，
         紧挨着的 note 又说了一遍）

    ⭐⭐ 三次都栽在同一个错误的分类方式上：**我按「它是什么」建清单**
      （图注 / 正文 / 图内），于是每次只堵住一格，下次它换一格再来。
      ⛔ 正确的分类只有一个问题：**读者会不会看见两遍。**
      ——&nbsp;这跟「盘点按渲染后属性、不按 class 名」是同一条教训。

    📌 所以这一版把一页上的文字分成三堆，**两两都比**：
      图内文字（SVG）· 图注（figcaption）· 正文（figure 之外的一切）。

    ⚠️⚠️ **它只挡得住照抄，挡不住改写。** 同一天里最难看的两处
      （论文那句理由被换词重说、多头那条图注把绿带两句重说）
      **这条查重一声没响** ——&nbsp;字符串对不上。那两处是渲染出来看见的。
      ⛔ 所以判据是：**每写完一章，必须把它截图看一遍。**
        查重负责挡住机械重复，好让眼睛有力气去看别的。
    """
    figs = re.findall(r"<figure\b.*?</figure>", html, re.S)
    if not figs:
        return []
    inner, caps = [], []
    for f in figs:
        m = re.search(r'id="([^"]+)"', f)
        fid = m.group(1) if m else "?"
        cap = "".join(re.findall(r"<figcaption>(.*?)</figcaption>", f, re.S))
        t = re.sub(r"<figcaption>.*?</figcaption>", "", f, flags=re.S)
        # ⛔ 折叠的「出处与口径」不参与 —— 出处本来就该在两处都查得到。
        t = re.sub(r'<details class="figsrc">.*?</details>', "", t, flags=re.S)
        inner.append((fid, _plain(t)))
        if cap:
            caps.append((fid, _plain(cap)))
    # ⛔ 折叠起来的 <details>（出处、模型表的长注解）**不算「看见两遍」** ——
    #   它默认是收着的。⭐ 判据还是那一条：看的是**读者会不会看见两遍**，
    #   不是「文件里有没有两份」。
    _body_src = re.sub(r"<figure\b.*?</figure>", "", html, flags=re.S)
    _body_src = re.sub(r"<details\b.*?</details>", "", _body_src, flags=re.S)
    # ⛔⛔ 封面（.hero）也不参与。它跟结尾那张图之间**隔着一整本书**，
    #   而结尾那张图是**故意**把封面那句话接回去的 ——&nbsp;这叫首尾呼应，
    #   不叫结巴。⭐ 判据补一条：**重复只有在「读者一眼能同时看见」时才是病**；
    #   而这个距离，查重量不出来 ——&nbsp;所以这一处只能人来判，并在这里写死。
    _body_src = re.sub(r'<div class="hero">.*?</div></div>', "", _body_src,
                       flags=re.S)
    body = _plain(_body_src)
    hits = []
    for fid, t in inner:
        hits += [(r, fid + "（正文↔图内）") for r in _runs(body, t, n)]
    for fid, c in caps:
        hits += [(r, fid + "（正文↔图注）") for r in _runs(body, c, n)]
    # ⛔ 专有名词不算重复。「FlashAttention」十四个字母就能触发，可它是**术语**，
    #   本来就该在图里和正文里各出现一次。⭐ 判据：**重复的单位是句子不是词** ——
    #   所以要求命中片段里至少有 8 个汉字，纯拉丁的一串直接放行。
    CJK = re.compile(r"[\u4e00-\u9fff]")
    seen, out = set(), []
    for r, w in hits:
        if r in seen or len(CJK.findall(r)) < 8:
            continue
        seen.add(r); out.append((r, w))
    return out


_CAP_BR = re.compile(r'<br\s*/?>')


def _split_cap(cap):
    """图注切成「第一段 ＋ 其余」，其余包进 .capmore（折叠模式下藏起来）。

    ⭐ 2026-09-16 现场：「留下图和主要的描述就行。」——&nbsp;图注是折叠模式下
    最大的一块字（48 条共 4960 字，占露出来的六成四），但它又**不能整条藏**：
    它是「这张图该看什么」的指路，藏了等于把图扔给读者自己猜。

    ⛔ 所以按**第一个 `<br>`** 切 ——&nbsp;不是按字数。本课图注本来就是
    「⭐ 一句落点 <br> ⭐⭐ 展开说」这个写法，`<br>` 就是作者自己标好的分界。
    **按字数切会把一句话拦腰截断，按作者的分段切不会。**

    ⚠️ 没有 `<br>` 的（48 条里 5 条）整条保留 ——&nbsp;它们本来就短。
    """
    parts = _CAP_BR.split(cap, 1)
    if len(parts) == 1:
        return cap
    return '%s<span class="capmore"><br>%s</span>' % (parts[0], parts[1])


def place_figs(html, FIGS, here=HERE):
    """把 `__FIG_X__` 占位符换成 <figure>，并把 .src.html 包成折叠的出处。"""
    for ph, (fid, fn, src, cap) in FIGS.items():
        fp = os.path.join(here, fn)
        # ⛔ 硬失败：图缺了宁可构建挂掉，也不要悄悄出一份少图的教材。
        assert os.path.isfile(fp), "缺 %s —— 先跑 `python3 %s`" % (fn, src)
        assert ph in html, "正文里没有占位符 %s —— 加图忘了插锚点？" % ph
        svg = io.open(fp, encoding="utf-8").read().strip()
        # ⭐⭐ 2026-09-14 现场原话：「整个文档里边有很多这种小字，并且是半透明的，
        #   其实并不重要，就是需要收着，折起来。」——&nbsp;说的就是每张图底下那四到
        #   八行「出处与口径」。它现在不画进 SVG 了（见 topic03_draw.src），
        #   旁落一份 .src.html，在这儿包成一个默认收起的 <details>。
        # ⛔ 折起来 ≠ 删掉：出处是「查过就要留下痕迹」那条规矩的另一半，
        #   它必须一直在、一直可查，只是不该默认占版面。
        sfp = fp[:-4] + ".src.html"
        note = ('<details class="figsrc"><summary>出处与口径</summary>%s</details>'
                % io.open(sfp, encoding="utf-8").read()) if os.path.isfile(sfp) else ''
        # ⛔ 图注为空串时**不要出空的 <figcaption>** —— 它不显示文字，但照样吃
        #    figcaption 的 margin/padding，图底下会多出一段说不清来路的空白。
        for h in dup_caption_vs_fig(cap, svg, fid):
            DUP_WARNED.append((fid, h))
        html = html.replace(
            ph, '<figure class="fbox fwide" id="%s">%s%s%s</figure>'
                % (fid, svg, '<figcaption>%s</figcaption>' % _split_cap(cap)
                   if cap else '', note))
    assert "__FIG_" not in html, "还有图占位符没被替换掉"
    return html


def place_table(html, here=HERE):
    """39 行可排序模型表（HTML 不是 SVG，所以不走 figure 通道）。没占位符就跳过。"""
    if "__TABLE_MODELS__" not in html:
        return html
    tbl = os.path.join(here, "fig3-models-table.html")
    assert os.path.isfile(tbl), "缺 fig3-models-table.html —— 先跑 `python3 topic03-table-models.py`"
    return html.replace("__TABLE_MODELS__",
                        '<div class="wrap">%s</div>'
                        % io.open(tbl, encoding="utf-8").read())


# ⭐⭐ 2026-09-13 学生审稿：全文 0 个锚链接 —— 每一句「见 §X.Y」都要
#   手动往回滚 80 万字符的页面。新手那位说他「真的滚回去找过，找不到才发现是错引」。
#   ⭐ 这里做一次后处理：
#     ① 给每个 <h3>/<h4> 自动加 id（按它开头的小节号）
#     ② 把正文里的「§X.Y」替换成指向它的 <a>
#   ⛔ 只替换**真实存在**的号 —— 指不到的保持原样，
#     这样它们在页面上仍然是纯文本，而 xref 体检照样能抓出来。
def anchorize(html):
    import re as _re
    ids = {}

    def _mark(m):
        tag, attrs, body = m.group(1), m.group(2), m.group(3)
        plain = _re.sub(r"<[^>]+>", "", body)
        # ⛔ 合并标题（「5.1 ＋ 5.2 ＋ 5.3 ＋ 5.4」）要**把整串号都登记上** ——
        #   只认第一个的话，指向 5.3 的引用就锚不上。
        #   ⭐ 这跟 topic02-lint-xref 里那条是同一个坑，那边今晚刚修过一次。
        run = _re.match(r"\s*((?:\d+\.\d+[a-z]?)(?:\s*[＋+、，,～~]\s*\d+\.\d+[a-z]?)*)",
                        plain)
        if not run or "id=" in attrs:
            return m.group(0)
        nums = _re.findall(r"\d+\.\d+[a-z]?", run.group(1))
        sid = "s" + nums[0].replace(".", "-")
        for nm in nums:
            ids[nm] = sid
        return "<%s%s id=\"%s\">%s</%s>" % (tag, attrs, sid, body, tag)

    html = _re.sub(r"<(h[34])([^>]*)>(.*?)</\1>", _mark, html, flags=_re.S)

    def _link(m):
        num = m.group(1)
        if num not in ids:
            return m.group(0)          # 指不到的不动，留给 xref 体检去报
        return '<a href="#%s">§%s</a>' % (ids[num], num)

    # ⛔⛔ 2026-09-13：上面这行注释写着「也不碰已经在 <a> 里的」——&nbsp;**它没做**。
    #   于是手写的 `§6.5b` 被再套一层，产出 8 处**嵌套 <a>**
    #   （非法 HTML，浏览器会悄悄拆掉，点击行为不可预期）。
    # ⭐ 判据（本仓库第 N 次遇到）：**注释写「不碰 X」不等于真的没碰。**
    #   —— 而且这是自己给自己写的注释，最容易被当成已经成立的前提。
    # ⛔⛔ 2026-09-19：这个自动链接**把别人论文的节号也链成了本页锚点**。
    #   「arXiv 2005.14165 §2.3」点下去跳到**本讲**的 §2.3 —— 链接是活的、
    #   指向也存在，所以「锚点必须落地」那条断言一个都抓不到。
    #   ⭐⭐ 最讽刺的一处：正文正在说「原文分节是…**不是 §3.2**」，
    #     而这句话里的那个 §3.2 本身就被链错了。
    #   ⭐ 判据：**比死链更难发现的是「指向一个真实存在、但内容不相干的地方」。**
    #
    # 规则是从产物里**数出来的**，不是拍的（2026-09-19 在专题四上统计）：
    #   · 「arXiv 号紧跟着 §」——&nbsp;15 处，**全部是引别人**，零误伤；
    #   · 放宽到「同一句话里出现过 arXiv 号」——&nbsp;再多抓 4 处，也全对，
    #     它们是被「（PaLM）」「）/ 」这类括号断开的；
    #   · 唯一的例外是「…1910.02054 §3（…）、§7.2」那个 §7.2 ——&nbsp;它是**本讲**附录。
    # ⇒ 引文上下文 ＝ 从 arXiv 号起，到下一个句号／换行／段落结束为止。
    #   想在引文里指回本讲，**显式写「本讲 §X.Y」** ——&nbsp;那会复位上下文。
    _ARXIV = _re.compile(r"\d{4}\.\d{4,5}")
    _STOP = _re.compile(r"[。！？]|<br\s*/?>|</p>|</li>|</div>")

    # ⛔⛔ 2026-09-25 专题五复审：「技术报告 §2.1.2、§3.2.2」没有 arXiv 号挨着，照样被链到了本讲。
    #   ⭐ 把「技术报告／报告／论文／README」也当成引文起点（同一句话内有效），规则跟 arXiv 号一样。
    # ⛔ 2026-09-25 手册新人走查：「专题五 §6.1」也会被链到本页 §6.1 —— 专题六会大量回指专题五第六节，
    #   而专题六自己也有 6.x。跨讲引用同样是引文。（图脚本那边 save() 早就放行了「专题」。）
    _CITE_WORD = _re.compile(r"技术报告|报告|论文|README|原文|专题[一二三四五六七八九十\d]+")

    def _in_citation(chunk, at):
        """at 这个 § 是不是落在某条引文的作用域里。"""
        a = None
        for mm in _ARXIV.finditer(chunk, 0, at):
            a = mm.end()
        for mm in _CITE_WORD.finditer(chunk, 0, at):
            if a is None or mm.end() > a:
                a = mm.end()
        if a is None:
            return False
        tail = chunk[a:at]
        if _STOP.search(tail):                 # 句子已经结束，引文作用域断了
            return False
        # ⭐ 显式复位：「本讲 §7.2」「本页 §3.1」永远当成自指
        return not _re.search(r"(本讲|本页|上面|前面)\s*$", _re.sub(r"<[^>]+>", "", tail))

    parts = _re.split(r"(<svg.*?</svg>)", html, flags=_re.S)
    for i in range(0, len(parts), 2):
        inner = _re.split(r"(<a\b.*?</a>)", parts[i], flags=_re.S)
        for j in range(0, len(inner), 2):
            chunk = inner[j]

            def _link2(m, _c=chunk):
                if _in_citation(_c, m.start()):
                    return m.group(0)          # 别人论文的节号，原样留着
                return _link(m)

            inner[j] = _re.sub(r"§(\d+\.\d+[a-z]?)", _link2, chunk)
        parts[i] = "".join(inner)
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-13 麻瓜视角审稿里唯一一条「我真的想关掉页面」：
#   「1MB 单页 ＋ 36 张图 ＋ 十几处内部跳转，**跳过去就回不来了**。」
#   而 CSS 里 .progress / .totop 早就写好了，body 里一次都没用上 ——
#   ⛔ **写好了没接上，等于没写**，而且比没写更难发现（看代码像是有）。
# ⭐ 目录**从 SECTIONS 里长出来**，不手写 —— 手写那份一定会跟节标题漂。
# ══════════════════════════════════════════════════════════════════
def build_nav(html):
    # ⛔⛔ 2026-09-20：这条正则写死了「第 X 节」这个 badge，
    #   于是 badge 写「附　录」的那一节**整个不进吸顶目录** ——
    #   而专题四的 §7.0「这些东西在框架里叫什么」恰恰是全讲最实用的一张表。
    #   ⭐ 判据：**目录的收录条件不要绑在「标签长什么样」上，绑在「它是不是一个 section」上。**
    secs = re.findall(
        r'<section id="(s[^"]+)"><div class="wrap"><div class="stn">'
        r'<span class="badge">(?:第 )?([^<]+?)(?: 节)?</span><h2>(.*?)</h2>', html, re.S)
    if not secs:
        return html
    # ⭐⭐ 2026-09-14：去掉条目前面的「零 一 二 三…」。
    #   ⛔ 它在这一条横带里**没有一个读者用得上**：目录本来就是从左到右排的，
    #     顺序靠位置已经说清楚了，汉字数字只是把每个条目撑宽一截、
    #     还跟标题抢视觉重量（原来它是 <b> 且比标题更黑）。
    #   ⭐ 节号没有丢 —— 挪进 title= 悬停提示。**要用的时候还在，不占版面。**
    items = []
    for sid, num, title in secs:
        t = re.sub(r"<[^>]+>", "", title).replace("&nbsp;", " ")
        full = re.split(r"——|:|：", t)[0].strip()
        items.append('<a href="#%s" title="第 %s 节 · %s">%s</a>'
                     % (sid, num, full, full[:16]))
    # ⛔⛔ 2026-09-14 类名撞车：这条吸顶带原来也叫 `.toc`，而**基础 CSS 里早就有一个
    #   `.toc`** —— 那是 2026-09-07 删掉的「这一讲的路线」方框目录留下的规则
    #   （`.toc{background:var(--bg2);border-radius;padding:22px 26px;margin:24px 0}`）。
    #   CSS_NAV 排在后面，所以 background 被盖住了、**看不出撞了**；
    #   但 padding / margin / border-radius 没人覆盖，**一路漏进来**：
    #   一条 44px 的横带被撑成 **88px**，上下各白 22px，左右还多 10px。
    #   ⛔ 那条基础规则**不能删** —— 它是从 topic-02-L300 的 CSS 整份抄来的，
    #     专题一现在还在用那个方框目录。所以改名的是我这边。
    #   ⭐⭐ 判据：**新控件复用老类名，冲突只会在「没被覆盖到的那几个属性」上显形。**
    #     背景色这种一眼能看见的反而是安全的 —— 危险的是盒模型：
    #     它不改颜色、不报错，只是让尺寸对不上，而你会以为是自己的 padding 写错了。
    nav = ('<div class="progress"><i></i></div>\n'
           '<nav class="tocbar" id="tocbar"><div class="tocbar-in">'
           '<span class="tocbar-lb">目录</span>' + "".join(items) + '</div></nav>\n')
    # ⛔⛔ 2026-09-14：这个「回到顶部」**一次都没工作过**，而且它看起来完全正常 ——
    #   按钮画出来了、.on 加上了、点得到（elementFromPoint 就是它自己）、
    #   控制台一条报错都没有，**只是页面纹丝不动**。
    #   根因：原来写的是裸 `scrollTo({top:0,…})`。行内 onclick 的作用域链是
    #     元素 → 表单 → document → window，而 **`Element.prototype.scrollTo` 是存在的** ——
    #   于是它解析成「滚动这个按钮自己」，按钮没有溢出内容，静默无操作。
    #   ⭐⭐ 判据：**行内事件处理器里调任何滚动 / 焦点 / 尺寸类方法，一律写全 `window.`。**
    #     这类名字在 Element 上往往也有一份同名的，抢在 window 前面被找到，
    #     而且**不报错**——它是个合法调用，只是作用在错的对象上。
    #   ⭐ 同一个坑的另一半：下面那个局部变量原来叫 `top`（＝ `window.top`）。
    #     IIFE 里侥幸没出事，但同一族的影子命名，顺手改成 `btn`。
    tail = ('<button class="totop" id="totop" title="回到顶部"'
            ' onclick="window.scrollTo({top:0,behavior:\'smooth\'})">↑</button>\n'
            '<script>(function(){\n'
            ' var bar=document.querySelector(".progress i"),'
            ' btn=document.getElementById("totop"),'
            ' toc=document.getElementById("tocbar"),'
            ' ls=[].slice.call(toc.querySelectorAll("a"));\n'
            ' function tick(){\n'
            '  var h=document.documentElement,'
            '  p=h.scrollTop/(h.scrollHeight-h.clientHeight||1);\n'
            '  bar.style.width=(p*100).toFixed(1)+"%";\n'
            '  btn.className="totop"+(h.scrollTop>600?" on":"");\n'
            '  var cur=null;\n'
            '  ls.forEach(function(a){var e=document.querySelector(a.getAttribute("href"));\n'
            '    if(e&&e.getBoundingClientRect().top<140)cur=a;});\n'
            '  ls.forEach(function(a){a.className=(a===cur?"on":"");});\n'
            '  if(cur&&cur.offsetLeft-toc.scrollLeft>toc.clientWidth-220)\n'
            '    toc.scrollLeft=cur.offsetLeft-120;\n'
            ' }\n'
            ' addEventListener("scroll",tick,{passive:true});tick();\n'
            # ⛔ 折叠起来的 <details> **打印出来只剩一行标题** —— 出处与口径
            #   是「查过就要留痕」那条规矩的落点，不能因为默认收起就印不出来。
            #   ⭐ CSS 做不到强行展开（那是 UA 行为，display 覆盖不了），
            #     所以挂 beforeprint / afterprint，印完原样收回去。
            ' var SR=".figsrc";\n'
            ' addEventListener("beforeprint",function(){\n'
            '  document.querySelectorAll(SR).forEach(function(d){\n'
            '   d.dataset.wasopen=d.open?"1":"";d.open=true;});});\n'
            ' addEventListener("afterprint",function(){\n'
            '  document.querySelectorAll(SR).forEach(function(d){\n'
            '   d.open=d.dataset.wasopen==="1";});});\n'
            '})();</script>\n')
    html = html.replace("</head>", CSS_NAV + "</head>", 1)
    html = re.sub(r"(<body[^>]*>)", r"\1\n" + nav, html, count=1)
    return html.replace("</body>", tail + "</body>", 1)


CSS_NAV = """<style>
/* ⭐ 吸顶目录：html 的 scroll-padding-top 本来就留了 96px 给它 ——
   那条规则之前是空转的（顶上什么都没有，点内链只是白空一截）。 */
.tocbar{position:sticky;top:0;z-index:58;background:rgba(255,255,255,.96);
     backdrop-filter:saturate(1.6) blur(8px);border-bottom:1px solid #e8eaed;
     overflow-x:auto;scrollbar-width:none}
.tocbar::-webkit-scrollbar{display:none}
.tocbar-in{display:flex;gap:6px;align-items:center;white-space:nowrap;
        max-width:1760px;margin:0 auto;padding:8px 16px}
.tocbar-lb{font:600 12px var(--mono);color:#9aa0a6;padding-right:6px;flex:none}
/* ⭐ 2026-09-14 每个条目给一个真的框。原来只有 padding ＋ 悬停底色 ——
   不悬停的时候它就是一排裸字，看不出「这是可以点的东西」，
   条目之间也只靠 2px 间距分隔，读起来是一长串。
   ⛔ 框要用 1px 实线 ＋ 胶囊圆角，不要用阴影：吸顶条只有 40 多像素高，
     阴影在这个尺度上只会糊成一片灰。 */
.tocbar a{font:500 13px/1.2 var(--mono);color:#5f6368;text-decoration:none;
       padding:5px 11px;border:1px solid #dadce0;border-radius:999px;
       background:#fff;flex:none;transition:background .12s,border-color .12s}
.tocbar a:hover{background:#f1f3f4;border-color:#bdc1c6;color:#202124}
.tocbar a.on{background:#e8f0fe;border-color:#aecbfa;color:#174ea6;font-weight:600}
@media print{.tocbar{display:none}}
</style>
"""



def lint_self_links(html):
    """一个 <section> 里有没有链回它自己的锚点。返回 [(节 id, 链接文字)]。

    ⛔ 2026-09-14 R4 抓到的：第二章里写「这就是<a href="#s二">下一节</a>那根
      百分比条」——&nbsp;**#s二 就是第二章自己**，读者点了原地不动。
    ⭐⭐ 为什么九条体检全放行：跨节指针体检只认 `§X.Y` 那种写法，
      **中文的「下一节 / 上一张」它根本看不见**。
      判据：**方位词也是指针**，而且比节号更难发现 ——&nbsp;它长得像散文。
    📌 这里只查最确定的一种：**链接落在自己所在的那一节**。
      这基本不可能是有意的，所以误报率极低。
    """
    bad = []
    for m in re.finditer(r'<section id="([^"]+)".*?</section>', html, re.S):
        sid, body = m.group(1), m.group(0)
        for a in re.finditer(r'<a href="#(%s)"[^>]*>(.*?)</a>' % re.escape(sid),
                             body, re.S):
            bad.append((sid, re.sub(r"<[^>]+>", "", a.group(2))))
    return bad


# ⛔⛔⛔ 2026-09-15 R11：一类**渲染才现形**的 bug，必须用 lint 钉死。
#
# 这些页面是**内联 SVG**（`<svg>` 直接写在 HTML 里）。HTML 解析器在
# foreign content（SVG）里碰到一个**只属于 HTML 的元素**时，会**当场退出
# foreign content 模式** ——&nbsp;等于就地把 `</svg>` 补上了。
# 后果：那个标签之后的一切，从图里掉出来，变成页面级的 HTML。
#
# 实际case（fig3-gun 的收尾落点带）：一句 `<u>剩下的那一截</u>`，
# 让第九章那张收尾图的落点带**只显示两行**，剩下四行跑到图外面、
# 以整页宽度渲染在图的下方。
#   ⭐⭐ 而 SVG 源码里那六行**一行不少、y 坐标全在带子里** ——
#     所以「读源码」「数字符」「查 y 坐标」三种自查全都过关。
#     只有**把图渲染出来看**才发现。这正是这 11 轮定下那条死规矩的价值。
#
# ⭐ SVG 里要加下划线，用属性：`<tspan text-decoration="underline">`。
_HTML_ONLY_IN_SVG = re.compile(
    r"<(u|b|i|em|strong|br|p|div|span|small|code)\b[^>]*>", re.I)


def lint_html_tags_in_svg(html):
    """内联 SVG 里混进 HTML 专属标签 ——&nbsp;会把 SVG 就地截断。

    ⛔⛔ 2026-09-23 修：**必须先把 style / script 整段摘掉再找 `<svg`。**
      专题四那段 quiz 样式表的注释里写了一句「`<svg>` 是 position:absolute」——&nbsp;
      纯注释、纯文字，可 `<svg\\b.*?</svg>` 不管它在不在注释里：
      它从那行起步，一路吃到 body 里真正的 `<svg class="wire"></svg>` 才收，
      于是整段样式表和整块 quiz DOM 被当成「一张图的内部」，
      一口气报 57 个假阳性，**并且 raise SystemExit 把整条 build-all 掐断**。
      ⭐ 判据跟 `_visible()` 那次是同一个（那次是 svg 排在 style 前面）：
        **凡是用「起始标签 … 结束标签」正则划范围的，都要先把
        「里面可以原样写标签的容器」摘干净** ——&nbsp;style / script / 注释。
      ⛔ 这类失败最坏的地方在于它**中止构建**：一条注释里的六个字符，
        表现出来是「专题四之后的页面全部没重新生成」，而报错只字未提样式表。
    """
    bad = []
    src = re.sub(r"<(style|script)\b.*?</\1>", "", html, flags=re.S | re.I)
    for m in re.finditer(r"<svg\b.*?</svg>", src, re.S):
        # ⛔ foreignObject 里的 HTML 是**合法**的（专题二有一处用它排版），
        #   先整段摘掉再查 ——&nbsp;否则会误报。
        seg = re.sub(r"<foreignObject\b.*?</foreignObject>", "",
                     m.group(0), flags=re.S)
        fid = (re.search(r'id="(fig[^"]*)"', seg)
               or re.search(r'aria-label="([^"]{0,28})', seg))
        for t in _HTML_ONLY_IN_SVG.finditer(seg):
            a = max(0, t.start() - 26)
            bad.append((fid.group(1) if fid else "?", t.group(1),
                        re.sub(r"<[^>]+>", "", seg[a:t.start() + 20])))
    return bad


# ══════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-16 现场：「专题三今天就要开讲了，把几乎所有文字都折叠起来，
#   因为我们照着图讲。特别重要的字留下。做个开关，默认折叠。」
#
# ⭐ 留下什么，不是凭感觉挑的 —— 这门课自己早就把「承重的那句」标出来了：
#     `p.lead`    章首那一句（这一章在讲什么）
#     `p.landing` 落点句（这一格的结论）
#     `figcaption` 图注（它是图的一部分，投屏时正需要）
#     `blockquote` 原话引文（只有 3 条，而且**念原话时屏幕上该有字**）
#   其余一律折叠：正文 p、列表、引用、以及全部 note 块。
# ⛔ 判据：**「哪些字重要」这个判断不要在样式表里重做一遍** ——
#   正文里已经有语义标记了，折叠规则只该<u>引用</u>它，不该另立一套。
#   （另立一套的后果是：以后加一段重要的话，样式表不知道，它就被折没了。）
#
# ⚠️ 只在**需要投屏的那一份**上开（L200 主线课件）。L300 是拿来读的，不开。
# ══════════════════════════════════════════════════════════════
FIGONLY_CSS = """
<style id="figonly-css">
/* ⛔⛔ 2026-09-19：这条浮动带原来钉在 right:18px bottom:18px，
   跟「回到顶部」那个圆按钮（.totop，right:22 bottom:22）**完全重叠**，
   而且它 z-index 99 压在人家 55 上面 —— 于是 totop 被挡得只剩一个角。
   ⭐ 判据：**两个 position:fixed 的浮动件，坐标必须显式错开，不能靠 z-index 分胜负** ——
     z-index 只决定谁盖住谁，不解决「两个都要能点」。
   ⭐⭐ 顺带瘦身：那行「已折叠 720 段讲解 · 按 T 切换」是宽度的大头，
     而它**只在第一次有用**。改成默认收起、悬停才展开。 */
.figbar{position:fixed;right:76px;bottom:22px;z-index:99;display:flex;gap:8px;
  align-items:center;font:600 14px/1.4 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}
.figbar button{cursor:pointer;border:1px solid #dadce0;background:#fff;color:#1a73e8;
  border-radius:999px;padding:9px 16px;box-shadow:0 2px 10px rgba(0,0,0,.14);
  font:inherit;white-space:nowrap}
.figbar button:hover{background:#f8f9fa}
.figbar .hint{color:#80868b;font-weight:400;background:#fff;border-radius:999px;
  box-shadow:0 2px 10px rgba(0,0,0,.10);white-space:nowrap;
  max-width:0;padding:6px 0;opacity:0;overflow:hidden;
  transition:max-width .22s ease,opacity .18s,padding .22s ease}
.figbar:hover .hint,.figbar:focus-within .hint{max-width:280px;padding:6px 12px;opacity:1}
/* ⛔ 窄屏上别让两件东西挤成一坨：提示整个不出现 */
@media (max-width:720px){.figbar{right:70px;bottom:18px}
  .figbar .hint{display:none}
  .figbar button{padding:8px 13px;font-size:13px}}
body.figonly section p:not(.lead):not(.landing),
body.figonly section ul,
body.figonly section ol,
body.figonly section pre,
body.figonly section .note{display:none}
/* ⛔ 防守：这两类里的 p 都不该被上面那条误伤 */
body.figonly section figure p{display:revert}
/* ⛔⛔ 2026-09-16：题目的题干是 <p class="q">，会被上面那条连题一起藏掉 ——&#160;
   点开「课前勿点」只剩选项、没有问题。⭐ 这个 bug 只有**把两道题搬进来之后**
   才存在：折叠规则是先写的，题是后搬的。
   判据：**加了新的内容类型，要回去看一遍既有的全局规则会不会误伤它。** */
body.figonly .guess p{display:revert}
/* ⛔⛔ 2026-09-16 现场截图抓到的：折叠模式下 <blockquote> 里的 <p> 被上面那条
   规则藏掉，**外壳还在** ——&#160;于是页面上出现两条空的灰条，看着像坏了。
   ⭐ 原设计说「原话引文保留」，但那从来没生效过（一生效就是空壳）。
   ⛔ 这次顺势改判：**折叠模式下整块隐掉**。理由不是省地方 ——&#160;
     引文是「故事」，不是「主线」，而折叠模式的定义就是只留主线。
   ⭐ 判据（上一次是 .guess p，这次反过来）：**一条写得太宽的全局规则，
     既会误伤该留的，也会留下该走的空壳 ——&#160;而空壳是静默的，更难发现。** */
body.figonly section blockquote{display:none}
/* ⭐⭐ 2026-09-16：图注是折叠模式下最大的一块字（48 条共 4960 字，占露出来
   的六成四）。⛔ 但图注不能整条藏 ——&#160;它是「看图该看什么」的指路。
   ⭐ 折中：**只留第一个 <br> 之前那一段**，后面的收起来。
     实测 43/48 条带 <br>，切完省 53%。切法在 place_figs 里。 */
/* ⛔⛔ 2026-09-16 第二轮：只藏 .capmore 还不够 ——&#160;现场原话
   「你的那个开关……它没收全吧？把一些该收起来的小字也接着都收进去。」
   ⭐ 图注整条藏掉。折叠模式的定义就此收敛成一句：**只留图。**
   ⚠️ 上一轮留图注的理由是「它是看图该看什么的指路」——&#160;那条理由没错，
     但它跟「投屏时小字看不见」这条冲突，而讲课场景优先。
     展开模式下图注一个字没少。 */
body.figonly figcaption{display:none}
body.figonly section h3{margin-top:34px}
</style>
"""

FIGONLY_HTML = """
<div class="figbar">
  <span class="hint" id="figonly-hint"></span>
  <button id="figonly-btn" type="button"></button>
</div>
<script>
(function(){
  // ⛔ 2026-09-17：原来写死成 "t3-figonly" —— 于是专题三和专题四**共用同一个开关**，
  //   在一边收起来，另一边打开就也是收的。⭐ 按页面名分开。
  var KEY = "figonly:" + location.pathname.split("/").pop();
  var btn  = document.getElementById("figonly-btn");
  var hint = document.getElementById("figonly-hint");
  // ⭐ 折叠了多少段，是**数出来的**，不写死 ——&#160;正文一改它自动跟着变。
  var n = document.querySelectorAll(
      "section p:not(.lead):not(.landing), section ul, section ol, "
    + "section pre, section .note, section blockquote, "
    + "figcaption .capmore").length;
  function set(on){
    document.body.classList.toggle("figonly", on);
    btn.textContent = on ? "展开讲解" : "折叠讲解";
    btn.title = on ? ("已折叠 " + n + " 段讲解，按 T 也可切换")
                   : ("讲解全部展开，按 T 折叠");
    hint.textContent = on ? ("已折叠 " + n + " 段　·　按 T 切换")
                          : ("全展开　·　按 T 切换");
    try { localStorage.setItem(KEY, on ? "1" : "0"); } catch(e){}
  }
  var saved = null; try { saved = localStorage.getItem(KEY); } catch(e){}
  // ⛔ 默认**展开**。2026-09-17 麻瓜审：默认折叠会让第一次点进来的人
  //   看到一份被挖空的教材（整节正文不见、九张图的图注全不见）。
  //   ⭐ 投屏的人按一下 T 就行，而第一次来的人没有第二次机会。
  set(saved === null ? false : saved === "1");
  btn.addEventListener("click", function(){
    set(!document.body.classList.contains("figonly"));
  });
  document.addEventListener("keydown", function(e){
    if (e.key === "t" || e.key === "T"){
      if (/^(INPUT|TEXTAREA)$/.test((e.target||{}).tagName||"")) return;
      set(!document.body.classList.contains("figonly"));
    }
  });
})();
</script>
"""


def add_figonly_toggle(html):
    """给一份页面装上「折叠讲解 / 展开讲解」开关，**默认展开**（2026-09-17 起；原注释写「默认折叠」已过期，见下面 JS 里的说明）。

    ⛔ 只给要投屏的那一份用。返回改过的 html。"""
    assert "figonly-css" not in html, "这一页已经装过折叠开关了"
    assert "</head>" in html and "</body>" in html, "页面结构不对，装不上"
    html = html.replace("</head>", FIGONLY_CSS + "</head>", 1)
    html = html.replace("</body>", FIGONLY_HTML + "</body>", 1)
    return html


def lint_headings_inside_sections(html, label):
    """每一个 <h3> 都必须落在某个 <section> 里面。

    ⛔⛔ 2026-09-17：这个形状**今晚栽了第三次**。
      往正文里插一小节，锚点选在 `<section id="s四">` 前面 ——&nbsp;
      看着是「插在第四节之前」，实际插在了**上一节的 `</section>` 之后**，
      于是那一整节浮在所有 section 外面：
      标签配平、查重、版面体检**全部照常通过**，
      浏览器里它也照样显示（只是丢了 section 的样式和折叠规则）。

    ⭐⭐⭐ 判据：**小节标题不是容器边界，section 才是。**
      往 HTML 里插内容，锚点必须选「同一个容器内」的元素。

    ⭐ 前两次都是靠肉眼发现的 ——&nbsp;而肉眼只在「恰好去看了」的时候有效。
      所以这次做成硬失败：**它是结构错误，不是观感问题。**
    """
    # ⛔⛔ 必须先把**注释 / script / style** 整块挖掉，否则里面写着的标签会被当真。
    #   实测：专题三有一段注释正文是「中间只隔一个 </section>，读者只走了一屏」——
    #   那个字面量让计数器变成负的，于是那一页 32 个 <h3> 全被误报成「在 section 外」。
    #   ⭐⭐⭐ 这是**今晚第三个**栽在同一件事上的检查器
    #     （前两个：readability 的段落墙、xref 的计数标记）。
    #     判据：**任何按标签扫 HTML 的检查器，第一行都得是「先挖掉不渲染的部分」。**
    #     它不是可选的预处理，它是这类检查器的**前提条件**。
    clean = re.sub(r'<!--.*?-->', '', html, flags=re.S)
    clean = re.sub(r'<(script|style)\b.*?</\1>', '', clean, flags=re.S)
    depth, bad = 0, []
    for m in re.finditer(r'<section\b|</section>|<h3\b[^>]*>(.{0,40})', clean, re.S):
        t = m.group(0)
        if t.startswith('</section'):
            depth -= 1
        elif t.startswith('<section'):
            depth += 1
        elif depth <= 0:
            bad.append(re.sub(r'<[^>]+>', '', m.group(1) or '')[:30])
    assert not bad, (
        "%s：有 %d 个 <h3> 掉在所有 <section> 外面 —— %s\n"
        "   ⭐ 往 HTML 里插内容，锚点要选**同一个容器内**的元素；"
        "小节标题不是容器边界，section 才是。" % (label, len(bad), bad[:3]))


_QTY_CN = {c: i for i, c in enumerate("零一二三四五六七八九十")}


def lint_list_counts(html, label):
    """标题里承诺了「N 条」，下面的列表就必须正好 N 条。

    ⭐⭐ 2026-09-20 专题四第 14 轮抓到的：「Muon ……三条限制都是作者自己
      写明的」下面只列了两条 —— 某轮精简删了一条，标题忘了跟着改。
    ⛔ 判据：**承诺了数量就等于签了字。** 读者真的会去数。

    两条把误报压下去的规则，都是被打脸打出来的：
    ⛔ ① **先把括号里的话抠掉再数。** 挪进共用框架的第一次全量跑就在
      专题三 L300 上误报：正文写「两个后果」，而紧跟的括号里自述
      「原先这里写『三个』…数字忘了跟着改」—— 括号里是改动史，不是承诺。
    ⛔ ② 量词表里故意没有「个 / 种 / 次」—— 它们太常出现在不是在数
      列表的句子里。宁可漏，不可扰。
    """
    bad = []
    # ⛔ ③ `(.*?)` 会**跨段落**：它一路吞到某个 `</p>` 正好贴着 `<ul>` 为止，
    #   于是上面三四段的文字全被当成了这个列表的引子（专题三 L300 上因此
    #   误报两条）。`(?:(?!</p>).)*` 把它钉死在紧贴列表的那一段里。
    for m in re.finditer(
            r"<p[^>]*>((?:(?!</p>).)*)</p>\s*<(ul|ol)>(.*?)</\2>", html, re.S):
        head = re.sub(r"[（(][^（）()]*[）)]", "", m.group(1))   # ① 抠掉括号
        q = None
        for mm in re.finditer(r"([0-9一二三四五六七八九十])\s*(条|样|点|项|处|步|件)",
                              head):
            q = mm                                              # 取最后一个
        if not q:
            continue
        num = int(q.group(1)) if q.group(1).isdigit() else _QTY_CN[q.group(1)]
        n = m.group(3).count("<li>")
        if num >= 2 and n and num != n:
            bad.append("说「%s%s」却列了 %d 条：%s…"
                       % (q.group(1), q.group(2), n,
                          re.sub(r"<[^>]+>|&nbsp;|&#160;", "", head).strip()[:34]))
    assert not bad, "%s 列表条数对不上标题：\n  %s" % (label, "\n  ".join(bad))


# ══════════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-22 现场立的口径，原话：
#   「像这种特别口语化的、特别有 AI 特点的句子，咱能清理一下吗？AI 味太重不好。」
#
# 先数了一遍才敢定阈值（专题四：正文＋图注＋折叠区 50,003 汉字）：
#   ⭐ 系 567 处、⛔ 系 446 处  →  平均**每 35 个汉字一个装饰符**
#   破折号 860 处              →  平均**每 58 个汉字一个**
#   「不是 A，是 B」62 · 「判据」62 · 「真正的 X」12 · 「？——」19
#
# ⛔ 判据跟加粗那条同源：**满篇都是重音，等于没有重音。**
#   而这几样叠在一起就是那个腔调 —— **每一句都想给你一个顿悟**。
#
# ⚠️ 这条 lint **按页开关**，不是全站生效：现场定的是「专题四先做，
#   做顺了再推到别的专题」。opt-in 的集合在 AI_VOICE_ENFORCED 里，
#   没进集合的页面只打报告、不失败。
# ⭐ 之所以做成 lint 而不是写进注释：写在注释里的判据只在那个文件生效，
#   而这份材料有九讲、四个人在改。
# ══════════════════════════════════════════════════════════════════════
AI_VOICE_ENFORCED = {"topic-04.html"}

AI_VOICE_BUDGET = dict(
    star3_total=10,      # ⭐⭐⭐ 全篇上限
    star2_per_sec=3,     # ⭐⭐ 每节上限
    cjk_per_star=200,    # ⭐（含 ⛔⚠📌）每多少汉字才许出现一个
    dash_per_para=1,     # 破折号：一段最多一个
    pair_per_sec=1,      # 「不是 A，是 B」每节上限
    ask_per_sec=1,       # 「？——」自问自答 每节上限
)


def _visible(html):
    """剥成读者真正看得见的字：去 style / script / svg / HTML 注释 / 标签。

    ⛔⛔ 2026-09-22 修：**这四个的剥离顺序是承重的，原来 `svg` 排在最前面。**
      页面里有内联 SVG 的地方，`<svg\\b.*?</svg>` 会跨过 `</style>` 去匹配，
      **把整张样式表和整段 JS 一起留在「可见文本」里** ——&#160;
      于是 `lint_ai_voice` 数了五十个来自 **CSS 注释**的 ⭐⛔，
      而那些字没有任何读者看得见。
      ⭐ 判据（这一轮第三次撞上同一条）：**度量之前先问它数的是不是你想数的东西。**
      ⭐ 修法：`style` / `script` 是严格配对的，**先把它们摘干净**，
        再去剥容易跨界的 svg。顺序一换，同一页 328 → 278。
    """
    s = html
    for tag in ("style", "script", "svg"):
        s = re.sub(r"<%s\b.*?</%s>" % (tag, tag), "", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&nbsp;", " ").replace("&#160;", " ")


def lint_ai_voice(html, label):
    """AI 味预算。详见上面那段口径。"""
    B = AI_VOICE_BUDGET
    bad, secs = [], re.split(r'<section\b', html)[1:] or [html]

    whole = _visible(html)
    cjk = len(re.findall(r"[一-鿿]", whole))
    star3 = whole.count("⭐⭐⭐")
    # ⛔ 数单星要先把多星吃掉，否则一个 ⭐⭐⭐ 会被数成三个 ⭐。
    ones = len(re.findall(r"[⭐⛔⚠📌]", re.sub(r"⭐{2,}", "", whole)))
    quota = max(1, cjk // B["cjk_per_star"])

    if star3 > B["star3_total"]:
        bad.append("⭐⭐⭐ %d 处，上限 %d —— 三星是全篇最高音，超了就不是最高音了"
                   % (star3, B["star3_total"]))
    if ones > quota:
        bad.append("单个 ⭐/⛔/⚠/📌 共 %d 处，按 %s 汉字配额只许 %d 处"
                   "（现在平均每 %d 字一个）"
                   % (ones, format(cjk, ","), quota, cjk // max(ones, 1)))

    for i, sec in enumerate(secs, 1):
        v = _visible(sec)
        two = len(re.findall(r"(?<!⭐)⭐⭐(?!⭐)", v))
        if two > B["star2_per_sec"]:
            bad.append("第 %d 节 ⭐⭐ %d 处，每节上限 %d" % (i, two, B["star2_per_sec"]))
        pair = len(re.findall(r"不是[^。；，]{1,14}[，,]\s*(?:是|而是)", v))
        if pair > B["pair_per_sec"]:
            bad.append("第 %d 节「不是 A，是 B」%d 处，每节上限 %d —— 其余改直陈句"
                       % (i, pair, B["pair_per_sec"]))
        ask = len(re.findall(r"[？?]\s*——", v))
        if ask > B["ask_per_sec"]:
            bad.append("第 %d 节「？——」自问自答 %d 处，每节上限 %d"
                       % (i, ask, B["ask_per_sec"]))

    # 破折号按**段**算：一段最多一个。⭐ 按段不按总量，因为它的毛病是「密」。
    over = [len(re.findall("——", _visible(m)))
            for m in re.findall(r"<p\b.*?</p>", html, re.S)]
    n_over = sum(1 for k in over if k > B["dash_per_para"])
    if n_over:
        bad.append("%d 个段落里破折号超过 %d 个（全篇共 %d 个）—— "
                   "破折号后面是解释的，改成句号断开"
                   % (n_over, B["dash_per_para"], sum(over)))

    tag = "⛔" if bad else "✅"
    print("   %s %-22s AI 味预算：%s" % (tag, label, "达标" if not bad else ""))
    for b in bad:
        print("        · %s%s" % ("（待校准）" if _calibrating(b) else "", b))
    hard = [b for b in bad if not _calibrating(b)]
    if hard and label in AI_VOICE_ENFORCED:
        raise AssertionError("%s AI 味超预算：\n  %s" % (label, "\n  ".join(hard)))


# ⚠️⚠️ 下面这两条**只报不拦**，因为阈值是我拍的、还没校准，原因写清楚：
#
#   「不是 A，是 B」定的是每节 1 处，实测全篇 62 处。可**逐条看下来，
#   它们大多在干活** —— 这份材料本身就是一路在纠误解，
#   「不是『怎么求导』，是怎么把 N 次前向压成一次」这种句子，
#   把「不是」删掉就损失了它要挡的那个误读。
#   ⭐ 真正扎眼的其实不是这个句式，是它**外面还裹着加粗 ＋ 星 ＋ 破折号**；
#     那三层已经按预算拆掉了。
#   ⛔ 所以不拿一个拍脑袋的数去毁 54 个句子 —— **阈值等现场定**。
#   建议改成按密度算（比如全篇每 2,000 汉字 1 处 ≈ 25 处），而不是按节。
#
#   「？——」同理，剩下的 4 处在图的出处行里（f.src 渲染成 HTML），
#   改它要动图脚本，跟正文不是一批活。
_CALIBRATING = ("「不是 A，是 B」", "「？——」", "破折号超过")


def _calibrating(msg):
    return any(k in msg for k in _CALIBRATING)


def finish(html, out_path, sections, label):
    """锚点 → 吸顶目录 → arXiv 自动链接 → 写盘 → 打一行回执。"""
    lint_headings_inside_sections(html, label)
    lint_list_counts(html, label)
    html = anchorize(html)
    html = build_nav(html)
    import course_links as _CL
    html = _CL.linkify_arxiv(html)
    io.open(out_path, "w", encoding="utf-8").write(html)
    # ⛔ AI 味那道放在**写盘之后**：它超标时要抛，而抛在写盘前会让
    #   产物停在上一版 —— 于是你打开页面看到的是旧的，越查越糊涂。
    #   ⭐ 判据：**会失败的检查，要让人看得见它检查的那个东西。**
    lint_ai_voice(html, label)
    print("ok  %s  %s 字符 · %d 节 · %d 个论文链接"
          % (label, format(os.path.getsize(out_path), ","),
             len(sections), _CL.count(html)))
    # ⭐ 查重回执两条：图注 ↔ 图内文字、正文 ↔ 图内文字。
    #   ⛔ 都只报告不中止，理由见 dup_caption_vs_fig 的注。
    # ⛔⛔ 2026-09-14 R3 的一个自伤，记在这儿：给这个函数扩容时，我用
    #   `s.index(函数名)` 和 `s.index("DUP_WARNED = []")` 去切片 ——&nbsp;
    #   **而这两个锚点的先后顺序跟我以为的正好相反**，于是文件里同时存在了
    #   两份同名函数，后定义的那份（旧版）赢。
    #   ⭐⭐ 构建**照样全绿**，回执还印着「查重通过」——&nbsp;
    #     它通过是因为跑的是旧代码，不是因为没问题。
    #   ⛔ 判据：**按字符串位置切源码之前，先确认两个锚点谁在前**；
    #     更稳的是切完 grep 一次「有没有出现两个同名 def」。
    for frag, fid in lint_dup_body_vs_figs(html):
        DUP_WARNED.append((fid, frag))
    if DUP_WARNED:
        print("    ⚠️  同一句话被读者看见两遍：%d 处" % len(DUP_WARNED))
        for fid, h in DUP_WARNED[:6]:
            print("       %-18s %s" % (fid, h[:40]))
        if len(DUP_WARNED) > 6:
            print("       …… 其余 %d 处" % (len(DUP_WARNED) - 6))
    else:
        print("    ✅ 查重通过（图内／图注／正文 三面互不重合）")
    del DUP_WARNED[:]
    bad_tags = lint_html_tags_in_svg(html)
    if bad_tags:
        # ⛔ 这条**中止构建**，不像查重那样只报告 ——&nbsp;
        #   因为它不是「读起来啰嗦」，是**图会缺一块，而且缺得看不出来**。
        for fid, tag, frag in bad_tags:
            print("    ⛔⛔ %s 的 SVG 里有 HTML 标签 <%s>：…%s…"
                  % (fid, tag, frag))
        raise SystemExit(
            "内联 SVG 里不能出现 HTML 专属标签 ——&nbsp;它会把 SVG 就地截断，"
            "后面的内容整段掉到图外面。要下划线请用 "
            'tspan text-decoration="underline"')
    self_links = lint_self_links(html)
    if self_links:
        print("    ⛔ 有链接指回自己所在的那一节（点了原地不动）：")
        for sid, txt in self_links:
            print("       #%-6s 链接文字「%s」" % (sid, txt[:24]))
    return html
