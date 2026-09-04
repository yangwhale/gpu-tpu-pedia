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
"""
import os
import re
import sys

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WebPages")
POINT = re.compile(r'(滚到|指\s*下|指\s*最后|停在|翻到|翻回|移到|指着)')
QUOTE = re.compile(r'[「『]([^」』]{4,32})[」』]')
FIGID = re.compile(r'(?:s012-)?(?:fig|ms-)[a-z0-9-]+')


def norm(t):
    """比对前统一形态：删空白、删标点、删省略号 —— 只留可比的骨架。"""
    return re.sub(r'[\s　·…、，。！？：:,.\-—－ᅳ「」『』（）()【】""\'\'*⭐⚠️⛔📖🍳▸]', '', t)


def main():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1900, "height": 1100})
        pg.goto("file://" + os.path.abspath(os.path.join(W, "topic-02.html")))
        pg.wait_for_timeout(1200)
        deck = pg.evaluate("""()=>({
          txt: document.body.innerText,
          figs: [...document.querySelectorAll('figure')].map(f=>f.id),
          h3: [...document.querySelectorAll('h3')].map(e=>e.innerText)})""")
        pg.goto("file://" + os.path.abspath(
            os.path.join(W, "topic-02-L200-lecture.html")))
        pg.wait_for_timeout(1000)
        cues = pg.evaluate(
            """()=>[...document.querySelectorAll('.board')]"""
            """.map(e=>e.innerText.replace(/\\s+/g,' ').trim())""")
        b.close()

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
            if not any(k in f for f in deck["figs"]) \
               and not any(('s012-' + k) in f for f in deck["figs"]):
                print('\n⛔ 讲义指了一个不存在的图 id：%s' % k)
                print('   提示原文：%s' % c[:96])
                bad += 1
        for n in re.findall(r'(?<![0-9.])([0-9]\.[0-9][b-c]?)(?![0-9])', c):
            if not any(h.strip().startswith(n) for h in deck["h3"]):
                print('\n⛔ 讲义指了一个不存在的小节号：%s' % n)
                print('   提示原文：%s' % c[:96])
                bad += 1
    print('\n讲义 ↔ 课件对账：%d 条 board 提示，%d 条对不上。'
          % (len(cues), bad))
    if not bad:
        print('   ✅ 每一条「滚到 X」的 X 都还在。')


if __name__ == '__main__':
    sys.exit(main())
