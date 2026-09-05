# -*- coding: utf-8 -*-
"""跨节指针体检：正文里写「§X.Y」「第 X.Y 节」，那一节在这份文档里真的存在吗。

════════════════════════════════════════════════════════════════════
⭐ 为什么要有这条，以及它跟另外三条体检的分工
════════════════════════════════════════════════════════════════════
前面三条各管一件事：
  · lint-readability  —— 扫源码（含字数、折叠是否藏了定义、自指计数）
  · lint-layout       —— 量渲染后的几何（左跑 / 右撑 / 图内文字撞车）
  · lint-cues         —— 讲义 ↔ 课件对账（「滚到 X」的 X 还在不在）

这一条管的是**同一份文档内部的指路**。它跟 lint-cues 是一对：
那条查「讲义指课件」，这条查「课件指自己」。

⛔ 这类债的形状跟 lint-cues 一模一样：**没有任何东西会报错**。
   页面生成、构建全绿、每一句话单独看都通顺 —— 学员照着号往回翻，翻不到，
   然后开始怀疑自己漏读了。2026-09-05 一查，L200 里有 12 处指向本文档不存在的小节。

⭐ 更值钱的是它抓到的那一类：**没兑现的承诺**。
   §2.3 那张强度图写着「这根轴第 2.8 节还会回来一次，那时上面会有五个算子」——
   而 L200 的 §2 只有 2.1 和 2.3。那不是编号写错，是**答应了读者一件事然后没做**，
   而这种事只有把「引用」和「实有小节」放在一起对，才看得见。

════════════════════════════════════════════════════════════════════
⚠️ 判据要松不要紧 —— 这条教训本仓库已经吃过五次
════════════════════════════════════════════════════════════════════
第一版把「实有小节」只从 `<h2>/<h3>` 标签里取，结果**把 L300 的 §5.5 / §5.6
误报成不存在**（那两节的标题不是 h3）。查上下文才发现。

⭐ 形状：**判据只覆盖了它当初见过的那个失败形状。**
  改成「按渲染后正文里行首就是小节号的行」取 —— 跟人眼看到的一致，
  人眼能看见的标题，判据就能看见。

第二类误报是**跨文档引用**：「专题一 §7.5 已经点过这条路」指的是别的专题，
不是本文档。所以引用前面 12 个字里出现「专题」「L300」「完整版」的一律跳过。
"""
import os
import re
import sys

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WebPages")

# ⭐ L200 里这几个号**故意**指向 L300 —— §0 的折叠里已经逐条列出来告诉读者了。
#    白名单必须跟那份清单**逐字一致**：清单漏一个，这里就该报一个。
#    （2026-09-05 建立时，那份清单正是靠这条判据才补全的。）
CROSS_DOC_OK = {
    "topic-02.html": {"1.4", "2.6", "2.8", "4.1", "5.5", "5.6"},
}

# 引用前面这些字样说明它指的是别的文档，不该拿本文档的目录去对
FOREIGN = re.compile(r'(专题[一二三四五六七八九十\d]|L300|L200|完整版|精讲版)')


def collect(path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1900, "height": 1100})
        pg.goto("file://" + os.path.abspath(path))
        pg.wait_for_timeout(1400)
        pg.evaluate("()=>document.querySelectorAll('details').forEach(d=>d.open=true)")
        pg.wait_for_timeout(300)
        txt = pg.evaluate("()=>document.body.innerText")
        b.close()
    return txt


def audit(fname):
    txt = collect(os.path.join(W, fname))
    # ⚠️ 「实有小节」按**渲染后行首**取，不按标签取 —— 见文件头那条误报教训。
    have = set(m.group(1) for m in re.finditer(r'(?m)^(\d+\.\d+[a-z]?)[　 ]', txt))
    ok = CROSS_DOC_OK.get(fname, set())

    bad, cross = {}, {}
    for m in re.finditer(r'(?:§|第)\s*(\d+\.\d+[a-z]?)\s*节?', txt):
        n = m.group(1)
        if n in have:
            continue
        if FOREIGN.search(txt[max(0, m.start() - 14):m.start()]):
            continue                      # 明写了是别的文档，放行
        (cross if n in ok else bad)[n] = (cross if n in ok else bad).get(n, 0) + 1

    print("\n══ %s" % fname)
    print("   实有小节 %d 个" % len(have))
    if cross:
        print("   ○ 跨文档引用（§0 已列明，放行）：%s"
              % "、".join("§%s×%d" % (k, v) for k, v in sorted(cross.items())))
    if not bad:
        print("   ✅ 没有指向本文档不存在的小节")
        return 0
    for n, c in sorted(bad.items()):
        print("   ⛔ §%s 指了 %d 次，而本文档没有这一节" % (n, c))
        for m in list(re.finditer(r'(?:§|第)\s*' + re.escape(n) + r'\s*节?', txt))[:2]:
            a = max(0, m.start() - 46)
            print("        …%s…" % txt[a:m.end() + 34].replace("\n", " / "))
    return len(bad)


def main():
    total = 0
    for f in ("topic-02.html", "topic-02-L300.html"):
        if os.path.exists(os.path.join(W, f)):
            total += audit(f)
    print("\n跨节指针体检：%d 类死指针。" % total)
    if not total:
        print("   ✅ 每一处「§X.Y」都指向真实存在的小节。")


if __name__ == '__main__':
    sys.exit(main())
