#!/usr/bin/env python3
"""写提示表之前用：列出一节里的标题／图／表的选择器，并把每张图截成 PNG，方便按比例量子区域。

用法：inspect_page.py <页面.html> <节的选择器，如 '#s一'> [输出目录] [--open 'details:has(#x)' ...]
"""
import os, sys
from playwright.sync_api import sync_playwright
html, sec = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "/tmp/lv-inspect"
opens = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--open"]
os.makedirs(out, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 2200, "height": 1238})
    pg.goto("file://" + os.path.abspath(html)); pg.wait_for_timeout(1200)
    for o in opens:
        pg.evaluate("s => document.querySelectorAll(s).forEach(d => d.open = true)", o)
    items = pg.evaluate("""s => [...document.querySelector(s).querySelectorAll('h3, figure[id], table, details > summary')]
        .map(e => ({tag: e.tagName, id: e.id, text: e.textContent.trim().slice(0, 40)}))""", sec)
    for it in items:
        print("%-8s %-22s %s" % (it["tag"], ("#" + it["id"]) if it["id"] else "", it["text"]))
        if it["tag"] == "FIGURE" and it["id"]:
            e = pg.query_selector("#" + it["id"]); bb = e.bounding_box()
            e.screenshot(path=os.path.join(out, it["id"] + ".png"))
            print("         %dx%d → %s/%s.png" % (bb["width"], bb["height"], out, it["id"]))
    b.close()
