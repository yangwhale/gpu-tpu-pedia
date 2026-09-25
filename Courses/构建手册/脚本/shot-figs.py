#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一页课件里的每张静态图截成 PNG，两张一拼，给人眼目检用。

用法：
    python3 构建手册/脚本/shot-figs.py WebPages/topic-05.html [输出目录，默认 /tmp/figshots]
    python3 构建手册/脚本/shot-figs.py WebPages/topic-05.html --only fig-afd,fig-topo

产物：
    <输出目录>/<图 id>.png         每张图一张
    <输出目录>/sheet00.png …       两张一拼（宽 ~1400，看得清字）

⭐ 为什么要从成品页面截，不直接打开 .svg：
   图是给内联用的片段，单独打开会显示成 XML 源码，或者字体退回 16px 衬线体 —— 看到的是假的。
⭐ 为什么两张一拼而不是一页全截：
   一整页十几张图缩进一张 PNG，字会小到看不清，而目检要看的正是「字有没有被盖住、压线、挤在一起」。

需要：pip install playwright pillow && playwright install chromium
"""
import argparse
import asyncio
import os

from PIL import Image
from playwright.async_api import async_playwright


async def shoot(page_path, out, only):
    ids = []
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1500, "height": 1000})
        await pg.goto("file://" + os.path.abspath(page_path))
        await pg.wait_for_timeout(1200)                       # 等字体和布局稳定
        all_ids = await pg.eval_on_selector_all('figure[id^="fig-"]', "es => es.map(e => e.id)")
        for fid in all_ids:
            if only and fid not in only:
                continue
            el = await pg.query_selector("#%s svg" % fid)
            if el is None:                                    # 这个 figure 里没有 SVG（比如表格）
                continue
            await el.screenshot(path=os.path.join(out, fid + ".png"))
            ids.append(fid)
        await b.close()
    return ids


def sheets(out, ids):
    files = [os.path.join(out, i + ".png") for i in ids]
    n = 0
    for k in range(0, len(files), 2):
        imgs = [Image.open(f) for f in files[k:k + 2]]
        w = max(i.width for i in imgs)
        h = sum(i.height for i in imgs) + 30 * (len(imgs) - 1)
        s = Image.new("RGB", (w, h), "white")
        y = 0
        for i in imgs:
            s.paste(i, (0, y))
            y += i.height + 30
        s.save(os.path.join(out, "sheet%02d.png" % (k // 2)))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("out", nargs="?", default="/tmp/figshots")
    ap.add_argument("--only", default="", help="逗号分隔的图 id，只截这几张")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(x for x in a.only.split(",") if x)
    ids = asyncio.run(shoot(a.page, a.out, only))
    if not ids:
        raise SystemExit("⛔ 一张图都没截到 —— 页面路径对吗？图的 figure 有没有 fig- 开头的 id？")
    n = sheets(a.out, ids)
    print("截了 %d 张图 → %s，拼成 %d 张 sheet*.png。逐张看：字有没有被盖住、压线、顶出面板、图例挤在一起。"
          % (len(ids), a.out, n))


if __name__ == "__main__":
    main()
