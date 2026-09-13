# -*- coding: utf-8 -*-
r"""投屏体检：把每张图**投到会议室屏幕上之后**的最小字号算出来。

⛔⛔ 2026-09-14 二轮学生审稿量出来的：34 张图里**没有一张**能在 1280×720 上
   保住 10px，7 张在 8.5px 以下 —— 而其中就包括那张会重复出现五次的主线图。

⭐ 根因不是字号定得不够大，是**一张 1400×914 的三栏图本来就是三页内容**：
   整张塞进投影区要缩到 0.72，11px 的地板到那儿就只剩 7.9px。
   把地板提到 13px 也救不了 —— 那只会让图更高、缩放更狠。

⭐⭐ 所以这条 lint **不提出「把字改大」**，它提出的是一个**读数**：
   「这张图整张投出去，最小字是多少像素」。
   低于 9px 就意味着 **这张图不能整张投**，必须**一次放大一格**讲
   （三栏图的一格 ≈ 440px 宽，占满投影区是 2.8×，11px 到那儿是 31px）。
   讲义里那些「指第二格」的指令，本来就是按这个节奏写的。

⚠️ 这条 lint 的输出是**报告，不中止** —— 它要防的是「悄悄变得更糟」：
   图一长高，这个数就往下掉，而屏幕上什么都看不出来。
"""
import glob
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "..", "WebPages")

# 会议室常见的 1280×720，减去页面左右边距与上下留白后的可用区
PROJ_W, PROJ_H = 1248.0, 660.0
FLOOR = 9.0          # 低于它就别整张投


def scan():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "*.svg"))
                    + glob.glob(os.path.join(W, "*.svg"))):
        s = io.open(p, encoding="utf-8").read()
        m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s)
        if not m:
            continue
        w, h = float(m.group(1)), float(m.group(2))
        sizes = [float(x) for x in re.findall(r'font-size:([\d.]+)px', s)]
        if not sizes or w <= 0 or h <= 0:
            continue
        sc = min(PROJ_W / w, PROJ_H / h)
        rows.append((min(sizes) * sc, os.path.basename(p), w, h,
                     min(sizes), sc))
    return sorted(rows)


def main():
    rows = scan()
    if not rows:
        print("   （没找到 SVG，跳过）")
        return
    bad = [r for r in rows if r[0] < FLOOR]
    print("   共 %d 张图；整张投到 %d×%d 时，最小字 < %.0fpx 的有 %d 张"
          % (len(rows), PROJ_W, PROJ_H, FLOOR, len(bad)))
    for a, f, w, h, mn, sc in rows[:8]:
        flag = "⛔" if a < FLOOR else "✅"
        print("     %s %5.1f px   %-28s %.0f×%-4.0f  源 %.1fpx × %.2f"
              % (flag, a, f, w, h, mn, sc))
    if bad:
        print("   ⭐ <%.0fpx 不代表图错了 —— 代表**这张图不能整张投**。"
              % FLOOR)
        print("     三栏图一次放大一格（440/1400）就是 2.8×，11px 到屏上是 31px；")
        print("     讲义里那些「指第二格」的指令本来就是按这个节奏写的。")
        print("   ⚠️ 这个数会随图变高悄悄往下掉，而屏幕上什么都看不出来 ——")
        print("     所以它要有人盯着，不能只靠看截图。")


if __name__ == "__main__":
    main()
