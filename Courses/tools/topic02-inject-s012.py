# -*- coding: utf-8 -*-
"""把 §0–§2 那十五张图重新生成并注入 topic-02-L300.html。

⭐ **为什么需要这个脚本。**

§3–§9 的四十张图由 `topic02-port-microscope.py` 生成并**写回**页面，
所以「改脚本 → 重跑 → 页面就是新的」。

而 §0–§2 这九个画图脚本各自吐一个独立的 `.svg` 文件，
当初是**手工粘进 HTML** 的。于是页面里那份和脚本**没有任何东西保证它们一致** ——
改了脚本忘了粘，或者直接在 HTML 里改了没回填脚本，两边都会静默地漂开，
而且谁都不会报错。这跟「撤了上游忘了改下游」是同一类失败，
区别只在于这里连搜索都省了：**它压根没有第二个副本可搜，只有一个会过期的拷贝。**

所以规矩改成跟 §3–§9 一样：**页面里的 SVG 是产物，脚本才是源。**
改图只改脚本，然后跑这个。

    cd Courses/tools && python3 topic02-inject-s012.py

它怎么定位：每个 `<figure>` 上打了 `id="s012-<key>"`，
注入器把该 figure 里**第一个 `<svg>` 到与之配对的 `</svg>`** 整段换掉，
figcaption 和周围正文一个字都不碰。

⚠️ 三条自检都跑在**写盘之前**，理由跟 port-microscope 那边一样：
「检查失败」和「文件没被污染」必须是同一件事。
"""
import io
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "..", "WebPages", "topic-02-L300.html")

# 脚本 → 它吐出来的 svg 文件名 → 页面上的 figure id
SCRIPTS = {
    "topic02-figs-map-origin.py":        {"figC.svg": "figC"},
    "topic02-fig-9474-waterfall.py":     {"fig0-1.svg": "fig0-1"},
    "topic02-figs-s1-panorama.py":       {"fig1-1.svg": "fig1-1", "fig1-2.svg": "fig1-2"},
    "topic02-fig-s3-why-layers.py":      {"fig3-1.svg": "fig3-1"},
    "topic02-fig-s3-sm-inside.py":        {"fig3-2.svg": "fig3-2"},
    "topic02-fig-s3-thread-is-a-lane.py": {"fig3-3.svg": "fig3-3"},
    "topic02-figs-s1-hierarchy-intensity.py":
                                         {"fig1-3.svg": "fig1-3", "fig1-4.svg": "fig1-4"},
    "topic02-figs-s2-access-lane.py":    {"fig2-1.svg": "fig2-1", "fig2-2.svg": "fig2-2"},
    "topic02-figs-s2-flash-intensity.py": {"fig2-3.svg": "fig2-3", "fig2-4.svg": "fig2-4"},
    "topic02-fig-s2-three-wastes.py":    {"fig2-5.svg": "fig2-5"},
    "topic02-figs-s2-chain-splash.py":   {"fig2-6.svg": "fig2-6", "fig2-7.svg": "fig2-7"},
    "topic02-figs-s2-intensity-axis.py": {"fig2-8.svg": "fig2-8"},
}


def _svg_span(html, fid):
    """返回该 figure 里 <svg …> … </svg> 的 (start, end)。"""
    # class 允许带 fwide（宽版图突破版心，见 port-microscope 里 .fwide 那段注释）
    i = -1
    for cls in ('fbox', 'fbox fwide'):
        i = html.find('<figure class="%s" id="s012-%s">' % (cls, fid))
        if i >= 0:
            break
    assert i >= 0, "页面上找不到 id=s012-%s 的 figure —— id 被人改掉了？" % fid
    s = html.find("<svg", i)
    assert s >= 0, "%s 里没有 <svg" % fid
    # 允许嵌套（defs 里不会有，但别赌）
    depth, j = 0, s
    while True:
        m = re.compile(r"</?svg\b").search(html, j)
        assert m, "%s 的 </svg> 没配上" % fid
        depth += 1 if m.group(0) == "<svg" else -1
        j = m.end()
        if depth == 0:
            return s, html.find(">", j) + 1


def main():
    tmp = tempfile.mkdtemp(prefix="s012-")
    got = {}
    for script, mapping in SCRIPTS.items():
        r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           cwd=tmp, capture_output=True, text=True)
        assert r.returncode == 0, "%s 跑挂了：\n%s" % (script, r.stderr)
        for fn, fid in mapping.items():
            p = os.path.join(tmp, fn)
            assert os.path.isfile(p), "%s 没有吐出 %s" % (script, fn)
            got[fid] = io.open(p, encoding="utf-8").read().strip()

    html = io.open(PAGE, encoding="utf-8").read()
    changed = 0
    for fid, svg in got.items():
        s, e = _svg_span(html, fid)
        if html[s:e].strip() != svg:
            html = html[:s] + svg + html[e:]
            changed += 1

    # ── 写盘前自检 ────────────────────────────────────────────────
    # ① 一张都不能少（漏一张而报「成功」是最危险的失败）
    miss = sorted(set(f for m in SCRIPTS.values() for f in m.values())
                  - set(got))
    assert not miss, "这几张没生成：%s" % miss
    # ② 公开页面禁字。
    #    ⛔ 这里**不能自己再抄一份禁字表** —— 原先抄了，结果两件坏事一起来：
    #    ① 两份表迟早会漂，一处补了另一处没补，还以为有两道防线；
    #    ② 仓库级闸门扫到这个文件时会命中**这份表自己**，永远误报。
    #    直接用 tpu-micro/gate.py 那份，单一来源。
    sys.path.insert(0, os.path.join(HERE, "tpu-micro"))
    from gate import lint_public                    # noqa: E402
    bad = lint_public(html)
    assert not bad, "公开页面里出现内部词，已中止写盘：%s" % bad
    # ③ 页面上真的还剩这么多 figure（防止 span 算错把别的吃掉）
    # 期望值从 SCRIPTS 推，别写死 —— 写死过一次 15，加第十六张图时它就报
    # 「注入把页面结构改坏了」，而结构其实好好的。自检误报会让人去绕过自检。
    want = sum(len(m) for m in SCRIPTS.values())
    n = len(re.findall(r'<figure class="fbox(?: fwide)?" id="s012-', html))
    assert n == want, "页面上有 %d 个 s012 figure，脚本这边有 %d 个 —— 对不上" % (n, want)
    # ④ 图里不许出现方位词。
    #    ⛔ 2026-09-03 一天之内踩了三次，每次都是同一个机理：这些 SVG 是
    #       **两份文档共用的**（L200 的 fig() 把整个 <figure> 原样搬过去），
    #       而两份文档里同一张图的**前后邻居不一样**。于是「下面 T-3」在 L300
    #       成立、在 L200 是反的；「上一张图那六层」两边同时指错。
    #    ⚠️ 三次都是渲染出来盯着看才发现的 —— 它不报错、不缺字、排版也正常，
    #       只是**指错了地方**。这类错误人眼扫一遍很容易滑过去，必须机器查。
    #    ✅ 正确写法：按名字指（「那六层为什么是六层」那张 / 见 G-3），不按位置指。
    bad_dir = []
    for m in re.finditer(r'<figure[^>]*id="(s012-[^"]+)"[^>]*>(.*?)</figure>', html, re.S):
        svg = re.search(r"<svg.*?</svg>", m.group(2), re.S)
        if not svg:
            continue
        txt = re.sub(r"<[^>]+>", "", svg.group(0))
        for w in ("上一张", "下一张", "上面那张", "下面那张", "前面那张", "后面那张"):
            if w in txt:
                bad_dir.append("%s 里的「%s」" % (m.group(1), w))
    assert not bad_dir, ("图内出现方位词，而这些图是两份文档共用的 —— "
                         "换成按名字指：%s" % "、".join(bad_dir))

    io.open(PAGE, "w", encoding="utf-8").write(html)
    print("ok  注入 %d/%d 张（其余与页面已一致）  topic-02-L300.html %s 字符"
          % (changed, len(got), format(len(html), ",")))


if __name__ == "__main__":
    main()
