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
                % (fid, svg, '<figcaption>%s</figcaption>' % cap if cap else '',
                   note))
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
    parts = _re.split(r"(<svg.*?</svg>)", html, flags=_re.S)
    for i in range(0, len(parts), 2):
        inner = _re.split(r"(<a\b.*?</a>)", parts[i], flags=_re.S)
        for j in range(0, len(inner), 2):
            inner[j] = _re.sub(r"§(\d+\.\d+[a-z]?)", _link, inner[j])
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
    secs = re.findall(
        r'<section id="(s[^"]+)"><div class="wrap"><div class="stn">'
        r'<span class="badge">第 (\S+) 节</span><h2>(.*?)</h2>', html, re.S)
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


def finish(html, out_path, sections, label):
    """锚点 → 吸顶目录 → arXiv 自动链接 → 写盘 → 打一行回执。"""
    html = anchorize(html)
    html = build_nav(html)
    import course_links as _CL
    html = _CL.linkify_arxiv(html)
    io.open(out_path, "w", encoding="utf-8").write(html)
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
    self_links = lint_self_links(html)
    if self_links:
        print("    ⛔ 有链接指回自己所在的那一节（点了原地不动）：")
        for sid, txt in self_links:
            print("       #%-6s 链接文字「%s」" % (sid, txt[:24]))
    return html
