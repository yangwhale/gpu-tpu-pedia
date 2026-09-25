# -*- coding: utf-8 -*-
"""讲义 ↔ 课件 对账：讲义说「滚到 X」，X 在课件里还在吗。

════════════════════════════════════════════════════════════════════
⭐ 为什么需要这条，以及为什么它跟别的体检不一样
════════════════════════════════════════════════════════════════════
讲义里有 80 多条 `<span class="board">` 提示，形如
「滚到本节最后那个灰框『先拿这条线判两样东西』」。
它们**引用的是课件里一个有名字的东西**：一个框的标题、一张图、一个小节号。

⛔ 而这类引用坏掉的时候，**没有任何东西会报错**：
   讲义照样生成、课件照样生成、两个页面各自都合法。
   **它只在现场才翻车** —— 台上照着念、往下滚、那个框不在了。

2026-09-04 那五轮全局重构（搬 §2.2、拆 §3 上下篇、改 §5 标题、
折掉七八个框、加两张新图）之后一查，**三条真的指空了**：
  · 「先拿这条线判两样东西」——&nbsp;那个框被画成了图 fig1-6，框没了
  · 「**算**屋顶线该用哪个分子」——&nbsp;课件里那个框叫「**画**屋顶线」，
    一字之差，Ctrl+F 找不到
  · 「一块矩阵乘怎么在 MXU 上跑完」——&nbsp;图上的标题根本不是这句

⭐ 判据一句话：**改课件里任何一个「有名字的东西」，都要回头对一遍讲义。**

════════════════════════════════════════════════════════════════════
⚠️ 判据要松，不要紧
════════════════════════════════════════════════════════════════════
board 提示里的引号有两种用途，机器分不清：
  ① **引用课件上的东西**（要对账）
  ② **讲师要说出口的原话**（不该对账）——&nbsp;
     「每指一个说一句『这个答案，编译的时候不存在』」就属于这种。
第一版不分，十条里误报七条。**误报会把真问题淹掉**（这条教训本仓库
已经吃过三次：零容忍自指计数、挖掉 svg 查定义、溢出探针不看 overflow-x）。

所以这里只在**提示明确说了「滚到 / 指 / 停在 / 翻到」**时才对账，
而且比对前把内层引号、加粗标记、省略号全部剥掉，做**子串包含**匹配。
宁可漏报，不要误报。

════════════════════════════════════════════════════════════════════
⛔⛔ 2026-09-14：这条 lint 报了十天的绿（09-04 建，09-14 才发现；原注释写「几个月」，课程 08-21 才开工，不可能），而它只看了五份讲义里的一份
════════════════════════════════════════════════════════════════════
原来的 main() 把两个文件名**写死**成 `topic-02.html` ↔
`topic-02-L200-lecture.html`。于是：
  · 它每次打印「100 条 board 提示，0 条对不上」——&nbsp;那 100 条**全是专题二的**；
  · 专题一 141 条、专题三 90 条、专题二x 12 条、专题八 8 条，
    **共 251 条从来没有被对过账**。

代价是真金白银的：专题三讲义里连着三条 cue 写着「滚到 1.3 / 1.5 / 1.6」，
实际该去 §3.3 / §3.5 / §3.6。§1.3、§1.5 真实存在（是别的内容），
§1.6 **压根不存在** ——&nbsp;而这条 lint 对三条全都没吭声，因为它没读那个文件。
（更早还误诊过一次：以为「1.2b 碰巧存在」才放行的，其实是根本没查。）

⭐⭐ 判据：**「0 条对不上」和「0 条被检查」在输出里长得一模一样。**
   任何按文件名枚举的检查，都要**把实际检了哪几对打出来**，
   并且在发现新页面没有配对时**主动报出来**，而不是静默跳过。
   —— 本仓库的同类教训：「测试全绿不是证据」「空结果不等于没有」。

════════════════════════════════════════════════════════════════════
⛔ 2026-09-14 当天第二个洞：小节号只认 h3，而专题一的节号不在 h3 里
════════════════════════════════════════════════════════════════════
上面那条修完，专题一立刻报出「指了一个不存在的小节号：7.5」。
**§7.5 是存在的** —— `#s75`，页面上明明白白写着「第 7.5 步 ·
算出来的数，跟机器上量出来的数」。它只是**不长在 h3 里**：

    专题二/三： <h3>3.5 三堵墙</h3>              ← 号和名同在 h3
    专题一：    <span class="badge">第 7.5 步</span><h2>算出来的数…</h2>

原来的检查写死 `deck["h3"]`，于是专题一的**八个小节一个都认不出来**，
只是恰好只有一条 cue 写了数字节号，才只炸了一次。

⭐⭐ 判据（跟上面那条是同一个病的两种长相）：
   **检查器对文档的建模，比文档本身窄。**
   上一条是「只看了五份文件里的一份」，这一条是「只认一种节号写法」。
   两次的表现都是**报了一个听起来很确凿的否定结论**
   （「从来没有这个小节」「每一条都对得上」），而真相是它没看。
   ⛔ 我自己也吃了这一口：R34 把这条读成「topic-01 一个 7.x 都没有」
   写进了提交记录 —— **那是从 lint 的模型里读出来的，不是从页面里查出来的。**
   工具说「不存在」，永远只等于「工具没找到」。
"""
import os
import re
import sys

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WebPages")
# ⛔ 2026-09-17：这张表漏了「切到 / 先放大 / 只放 / 整张投」四类，于是专题四 15 条提示里有 7 条从没被检查过。
#   ⭐ 加动词之前先想：**讲义里真实出现过的说法，都在这张表里吗？**
POINT = re.compile(r'(滚到|切到|指\s*下|指\s*最后|停在|翻到|翻回|移到|指着'
                   r'|放大|只放|整张投|平移)')
QUOTE = re.compile(r'[「『]([^」』]{4,32})[」』]')
FIGID = re.compile(r'(?:s012-)?(?:fig|ms-)[a-z0-9-]+')


def has_sec(head, n):
    """这段标题文字算不算「小节 n」。

    两种写法都要认，而且**只认开头**，不然 37.5% 会冒充 §7.5：
        专题二/三  `3.5 三堵墙`        → 直接 startswith
        专题一      `第 7.5 步`         → 剥掉开头的「第」再 startswith
    """
    t = re.sub(r'^\s*第\s*', '', head.strip())
    return t.startswith(n)


def norm(t):
    """比对前统一形态：删空白、删标点、删省略号 —— 只留可比的骨架。"""
    return re.sub(r'[\s　·…、，。！？：:,.\-—－ᅳ「」『』（）()【】""\'\'*⭐⚠️⛔📖🍳▸]', '', t)


# ⭐ 讲义 → 它对应的课件。**新增讲义必须在这里登记** ——&nbsp;
#   没登记的会在下面被主动报出来，不会静默跳过（这正是 2026-09-14 那个洞）。
PAIRS = [
    ("topic-01-lecture.html",      "topic-01.html"),
    ("topic-02-L200-lecture.html", "topic-02.html"),
    ("topic-02x-lecture.html",     "topic-02x.html"),
    ("topic-03-lecture.html",      "topic-03.html"),
    ("topic-03-L300-lecture.html", "topic-03-L300.html"),
    ("topic-04-lecture.html",      "topic-04.html"),
    ("topic-05-lecture.html",      "topic-05.html"),
    ("topic-08-lecture.html",      "topic-08.html"),
]


def check_one(pg, lec, dck):
    """返回 (cue 数, 对不上的条数)。找不到文件就跳过并说清楚。"""
    lp, dp = os.path.join(W, lec), os.path.join(W, dck)
    if not (os.path.exists(lp) and os.path.exists(dp)):
        print('   ○ %-30s 跳过（缺 %s）'
              % (lec, lec if not os.path.exists(lp) else dck))
        return 0, 0
    pg.goto("file://" + os.path.abspath(dp))
    pg.wait_for_timeout(1200)
    # ⛔⛔ 2026-09-14：这里原来用 innerText —— 而 **innerText 对折叠在
    #   <details> 里的元素返回空串**。专题三有 10 个 h3 在折叠区，于是
    #   lint 只看得见 2 个 §3.x，把 3.2b / 3.3 / 3.5 / 3.6 全报成「不存在」。
    # ⭐ 判据：**lint 要查的是「文档里有没有」，不是「此刻屏幕上有没有」** ——
    #   凡是判断存在性的取值，一律用 textContent；innerText 只适合量版面。
    #   （误报比漏报更糟：本文件头上就写着「误报会把真问题淹掉」。）
    # ⛔ secs 不能只取 h3 —— 专题一的节号在 `<span class="badge">第 7.5 步</span>`
    #   里，h2 只有名字没有号。少收一种写法，整个专题的小节都会被判成不存在。
    deck = pg.evaluate("""()=>({
      txt: document.body.innerText + ' ' + document.body.textContent,
      figs: [...document.querySelectorAll('figure')].map(f=>f.id),
      secs: [...document.querySelectorAll('h2,h3,h4,.badge')]
              .map(e=>e.textContent)})""")
    pg.goto("file://" + os.path.abspath(lp))
    pg.wait_for_timeout(1000)
    cues = pg.evaluate(
        """()=>[...document.querySelectorAll('.board')]"""
        """.map(e=>e.innerText.replace(/\\s+/g,' ').trim())""")
    n = scan(cues, deck, lec)
    print('   %s %-30s %3d 条 cue，%d 条对不上'
          % ('✅' if not n else '⛔', lec, len(cues), n))
    return len(cues), n


def lint_board_speech():
    """⛔ 要说出口的话，不许写进 `.board` 屏幕提示里 ——&nbsp;一律放 `.say`。

    ════════════════════════════════════════════════════════════
    2026-09-14 加。这一天同一个坑踩了两次，第二次才明白它是个**系统性**问题。
    ════════════════════════════════════════════════════════════
    症状：把一整段台词写进了 `<p class="board">`：
        🖥 屏幕：滚到 fig3-chronicle，整张投。
        「在开始之前，先把这八年真实发生过的事整个摊开看一眼。……」
    上面那条对账（scan）只认「紧跟『滚到』后 40 字内」的引号，于是它把这段
    台词的开头当成**课件引用**去找，报了一条**假的**「指了个找不到的东西」。

    ⭐ 两个后果，第二个更贵：
      ① 假报警本身要花时间排查；
      ② **假报警会训练人忽略这条 lint** ——&nbsp;本仓库已经吃过三次这个亏
         （零容忍自指计数、挖掉 svg 查定义、溢出探针不看 overflow-x）。

    ⭐⭐ 所以修法不是「把 scan 的窗口调窄」，而是**从源头上分工**：
        **`.board` 只写「去哪儿」，要说的话一律进 `.say`。**
        —— 判据落在**写法**上，不落在**检测**上。检测总能被下一种写法绕过。

    ⚠️ 判据必须严，不然又是一个误报源（那就自相矛盾了）。两条命中：
      · **引号里含 `<br>`** ——&nbsp;屏幕提示不会分行，会分行的一定是台词。近乎零误报。
      · **引号里 ≥40 个汉字**，且前面没有「说 / 念 / 原话」这类明确标注。
    ⭐ 豁免那一条是真实存在的合法写法，别删：
        `<em>说一句「先别一张一张看，要五张一起看……」。</em>`
      —— 短台词内联进提示里，讲师一眼扫到，比拆成两段好。**标注过就放行。**
    """
    BOARD = re.compile(r'<p class="board"[^>]*>(.*?)</p>', re.S)
    QUO = re.compile(r'「(.*?)」', re.S)
    MARK = re.compile(r'(说一句|说|念|原话|台词)\s*$')
    bad = 0
    for f in sorted(os.listdir(W)):
        if not f.endswith("-lecture.html"):
            continue
        src = open(os.path.join(W, f), encoding='utf-8').read()
        for mb in BOARD.finditer(src):
            body = mb.group(1)
            for mq in QUO.finditer(body):
                raw = mq.group(1)
                plain = re.sub(r'<[^>]+>', '', raw)
                han = len(re.findall(r'[一-鿿]', plain))
                head = re.sub(r'<[^>]+>', '', body[:mq.start()])[-8:]
                if '<br' in raw:
                    why = '引号里有换行 ——&nbsp;屏幕提示不会分行'
                elif han >= 40 and not MARK.search(head):
                    why = '引号里 %d 个汉字，而且没标「说一句」' % han
                else:
                    continue
                print('\n⛔ 台词写进了屏幕提示里（%s）：%s' % (f, why))
                print('   ⭐ 改法：`.board` 只写去哪儿，这段话搬进 `.say`。')
                print('   原文：%s…' % plain[:60])
                bad += 1
    return bad


def main():
    from playwright.sync_api import sync_playwright
    # ⛔ 主动发现没登记的讲义 ——&nbsp;静默跳过就是上一个版本翻车的方式。
    known = {a for a, _ in PAIRS}
    stray = sorted(f for f in os.listdir(W)
                   if f.endswith("-lecture.html") and f not in known)
    total = bad = 0
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1900, "height": 1100})
        for lec, dck in PAIRS:
            c, n = check_one(pg, lec, dck)
            total += c
            bad += n
        b.close()
    if stray:
        print('\n⛔ 有讲义没在 PAIRS 里登记，因此从未被对账：%s'
              % '、'.join(stray))
        bad += len(stray)
    # ⛔⛔ 2026-09-14 顺手发现：这个 main() **原来根本没有 return**。
    #    于是 `sys.exit(main())` ＝ `sys.exit(None)` ＝ **退出码恒为 0** ——
    #    这条 lint 哪怕报出一屏「⛔ 指了个不存在的图」，对外也是「成功」。
    # ⭐ 判据（本仓库第 N 次撞同一类）：**打印出错误 ≠ 报告了失败。**
    #    凡是 `sys.exit(f())` 的 f，都要回头确认它真的 return 了那个计数。
    bad += lint_board_speech()
    print('\n讲义 ↔ 课件对账：%d 份讲义、%d 条 board 提示，%d 条对不上。'
          % (len(PAIRS), total, bad))
    if not bad:
        print('   ✅ 每一条「滚到 X」的 X 都还在。')
        print('   ✅ 也没有把台词写进屏幕提示里。')
    return bad


def scan(cues, deck, lec):
    deck_txt = norm(deck["txt"])
    bad = 0
    for c in cues:
        m = POINT.search(c)
        if not m:
            continue                      # 不是「去哪儿」的提示，不对账
        # ⚠️ 只认**紧跟在「滚到」后面 40 字以内**的那个引号 ——&nbsp;
        #    同一条提示里往往还有第二、第三个引号，那些是**讲师要说出口的话**
        #    （「一定要说，它挡掉后面所有『所以谁更好』的提问」）。
        #    不设这个窗口，那些话会被当成课件引用报出来。
        near = c[m.end():m.end() + 40]
        for q in QUOTE.findall(near):
            if norm(q) and norm(q) not in deck_txt:
                print('\n⛔ 讲义指了一个课件里找不到的东西：「%s」' % q)
                print('   提示原文：%s' % c[:96])
                bad += 1
        for k in FIGID.findall(c):
            # ⭐ 三种写法指的是同一张图，都要认：
            #   讲义说 `fig3-duality`（SVG 文件名）／专题三页面 id 是 `fig-duality`
            #   ／专题二页面 id 是 `s012-fig1-4`。
            alt = {k, 's012-' + k, re.sub(r'^fig\d+-', 'fig-', k)}
            if not any(a in f for a in alt for f in deck["figs"]):
                print('\n⛔ 讲义指了一个不存在的图 id：%s' % k)
                print('   提示原文：%s' % c[:96])
                bad += 1
        for n in re.findall(r'(?<![0-9.])([0-9]\.[0-9][b-c]?)(?![0-9])', c):
            if not any(has_sec(h, n) for h in deck["secs"]):
                print('\n⛔ 讲义指了一个不存在的小节号：%s' % n)
                print('   提示原文：%s' % c[:96])
                bad += 1
    return bad


if __name__ == '__main__':
    sys.exit(main())
