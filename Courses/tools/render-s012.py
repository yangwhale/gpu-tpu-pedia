# -*- coding: utf-8 -*-
"""把 §0–§2 那十五张图渲染成 PNG，用来**肉眼看一眼**再决定改没改对。

为什么单独要一个：§3–§9 的图是 `t02-figs/render.sh` 渲的，那些 SVG 自带
`width`/`height` 和内嵌 `<style>`，独立打开就是对的。
§0–§2 这批不是 —— 它们写的是 `width="100%"` + viewBox，而且字号字重全靠
**页面里的** `.svglbl / .svgsm / .svgnum` 三个 class。

所以直接拿浏览器打开 `.svg` 看到的东西是**假的**：所有文字都会退回浏览器
默认的 16px 衬线体，行距、溢出、遮挡全部对不上页面里的真实样子。
判断「这行字有没有压到框」必须在**页面同款 CSS**下看。

这个脚本就干这件事：跑画图脚本 → 拿 viewBox 算真实像素 → 套上页面那几条
CSS 包成 HTML → headless Chrome 截图。

    cd Courses/tools && python3 render-s012.py fig1-2 [fig2-5 ...]
    cd Courses/tools && python3 render-s012.py --all

PNG 落在 /tmp/s012-png/<name>.png。

⚠️ `TMPDIR=/tmp` 和 `--user-data-dir=/tmp/cc-render` 两个都不能省 ——
少一个 Chrome 会因为 SingletonSocket 路径过长直接 FATAL（跟 t02-figs 同一个坑）。
"""
import io
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/tmp/s012-png"

# 跟 topic02-inject-s012.py 的 SCRIPTS 是**同一张表**。
# 这里不 import 是因为那个文件名带连字符（Python 标识符非法），
# 硬 import 要走 importlib.util.spec_from_file_location，为一张表不值得。
# ⛔ 改了那边记得改这边 —— 两处漂了的话，症状是「渲染出来的不是页面里那张」。
SCRIPTS = {
    "topic02-figs-map-origin.py":            ["figA", "figC"],
    "topic02-fig-9474-waterfall.py":         ["fig0-1"],
    "topic02-figs-s1-panorama.py":           ["fig1-1", "fig1-2"],
    "topic02-figs-s1-hierarchy-intensity.py": ["fig1-3", "fig1-4"],
    "topic02-fig-s1-landing.py":             ["fig1-5"],
    "topic02-figs-s2-access-lane.py":        ["fig2-1", "fig2-2"],
    "topic02-figs-s2-flash-intensity.py":    ["fig2-3", "fig2-4"],
    "topic02-fig-s2-three-wastes.py":        ["fig2-5"],
    "topic02-figs-s2-chain-splash.py":       ["fig2-6", "fig2-7"],
    "topic02-figs-s2-intensity-axis.py":     ["fig2-8"],
}
WHERE = {f: s for s, fs in SCRIPTS.items() for f in fs}

# 从 topic-02.html 抄来的三条 class + 用到的 CSS 变量。
# 抄而不是解析：解析要处理 @media、变量继承、:root 层叠，
# 复杂度远超收益，而这三条几年没动过。真改了，这里的字号会明显不对，一眼看得出来。
CSS = """
:root{--ink:#202124;--gray:#5f6368;--gray2:#80868b;
      --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
body{margin:0;padding:24px;background:#f8f9fa;
     font-family:"Noto Sans CJK SC",sans-serif}
svg{display:block;width:100%;height:auto;background:#f8f9fa}
.svgtxt{font:500 13px var(--mono)}   .svgtxt:not([fill]){fill:var(--ink)}
.svglbl{font:600 12px "Noto Sans CJK SC",sans-serif}
.svglbl:not([fill]){fill:var(--gray)}
.svgnum{font:700 14px var(--mono)}   .svgnum:not([fill]){fill:var(--ink)}
.svgsm{font:500 10.5px var(--mono)}  .svgsm:not([fill]){fill:var(--gray2)}
"""

PX = 1100          # 页面正文栏差不多这么宽


def render(names):
    tmp = tempfile.mkdtemp(prefix="s012r-")
    os.makedirs(OUT, exist_ok=True)
    for s in sorted({WHERE[n] for n in names}):
        r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                           cwd=tmp, capture_output=True, text=True)
        assert r.returncode == 0, "%s 跑挂了：\n%s" % (s, r.stderr)

    for n in names:
        svg = io.open(os.path.join(tmp, n + ".svg"), encoding="utf-8").read()
        m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        assert m, "%s 没有 viewBox，算不出高度" % n
        vw, vh = float(m.group(1)), float(m.group(2))
        h = int(round(PX * vh / vw)) + 48                 # +48 = 上下 padding
        page = os.path.join(tmp, n + ".html")
        io.open(page, "w", encoding="utf-8").write(
            '<!doctype html><meta charset="utf-8"><style>%s</style>%s' % (CSS, svg))
        png = os.path.join(OUT, n + ".png")
        env = dict(os.environ, TMPDIR="/tmp")
        subprocess.run(
            ["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", "--user-data-dir=/tmp/cc-render",
             "--window-size=%d,%d" % (PX + 48, h),
             "--default-background-color=FFFFFFFF",
             "--screenshot=" + png, "file://" + page],
            env=env, capture_output=True, text=True)
        assert os.path.isfile(png), "%s 没截出图" % n
        print("ok  %s  %dx%d" % (png, PX + 48, h))


if __name__ == "__main__":
    a = sys.argv[1:]
    render(sorted(WHERE) if (not a or a[0] == "--all") else a)
