# -*- coding: utf-8 -*-
r"""量一段动画的**首尾是否同一帧** —— 页面上这些 mp4 都挂 `loop`，
首尾对不上就会每轮跳一下，比没有动画还难看。

⭐⭐⭐ 2026-09-18 立这条判据的现场：`topic04-descend.mp4` 自查说
  「首末帧最大像素差 255，**平均差 1.26**，✅ 首尾一致」——&#160;而实际上
  首帧是一条曲线加两个球、末帧是一列条，**差着整整一幕**。

⛔⛔ 这一路上**同一个病换皮出现了三次**，每次都是
  「**全局聚合指标被不变的大多数主导**」：

  ① 平均像素差 ——&#160;画面 98.5% 是白底，差异被稀释两个数量级；
  ② 差异像素 ÷ 墨水 ——&#160;看着靠谱，但 `saddle` 那张棋盘格曲面
     只转了不到半度就量出 **51.2%**，跟「差着一整幕」的 descend 同档；
  ③ 墨水总量相对差 ——&#160;`memtime` 首末是「空带 vs 画满的三角」，
     肉眼一眼认出，可它那条恒定的灰带占了 19 万像素，
     把真正的变化稀释成 **17.4%**，比别的片子还低。

  ⭐⭐⭐ 结论不是「再换一个指标」，是**一个数分不开这两类差异**：
    「末帧是另一幕」和「末帧偏了半度」在任何全局标量上都能撞到同一档。
    **能一秒分开它们的是人眼。**

⭐ 所以这个工具的形状是**基线回归**，不是阈值判定：
  · 把每支片子**当前量到的值记进 `loop-baseline.json`**，连同人眼判过的结论；
  · build 时只问一句「**有没有变糟**」——&#160;可证伪、不需要魔法数字；
  · 新片子没有基线 → 直接报红，逼你看一眼拼接图再登记。
  这也正是 build 守卫该干的事：**守住已知状态，而不是假装知道正确答案。**

⭐ 每次都落一张上下拼接图。这次就是看了拼接图才一眼认出
  「上面是曲线、下面是一列条」——&#160;**数字告诉你有问题，图告诉你是什么问题**。

用法：
    python3 tools/manim/check-loop.py                 # 扫 WebPages/media/*.mp4
    python3 tools/manim/check-loop.py --update        # 把当前值写回基线（先看图！）
    python3 tools/manim/check-loop.py a.mp4           # 只查这个
拼接图一律写到 /tmp/loopdiff-<名字>.png。
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.normpath(os.path.join(HERE, "..", "..", "WebPages", "media"))
BASE = os.path.join(HERE, "loop-baseline.json")

INK = 12        # 跟白底差多少才算「有墨水」
DIFF = 12       # 两帧差多少才算「这个像素变了」
SLACK = 3.0     # 比基线糟这么多个百分点才算回退（留给压缩抖动）


def _frame(mp4, tail):
    """tail=False 取第一帧；tail=True 取最后一帧。"""
    fd, png = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    if tail:
        cmd = ["ffmpeg", "-y", "-v", "error", "-sseof", "-0.08",
               "-i", mp4, "-update", "1", "-q:v", "1", png]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", mp4,
               "-vf", "select=eq(n\\,0)", "-vsync", "0", "-frames:v", "1", png]
    subprocess.run(cmd, check=True)
    a = np.asarray(Image.open(png).convert("RGB"), dtype=np.int16)
    os.unlink(png)
    return a


def measure(mp4):
    """返回 (不一致度 %, 首帧墨水, 末帧墨水)，并落一张上下拼接图。"""
    a, z = _frame(mp4, False), _frame(mp4, True)
    if a.shape != z.shape:
        return 100.0, 0, 0
    ink_a = (255 - a).max(2) > INK
    ink_z = (255 - z).max(2) > INK
    both = ink_a | ink_z
    n = int(both.sum())
    bad = 0.0 if n == 0 else float((np.abs(a - z).max(2) > DIFF).sum()) / n * 100
    Image.fromarray(np.vstack([a, z]).astype(np.uint8)).save(
        "/tmp/loopdiff-%s.png" % os.path.basename(mp4)[:-4])
    return bad, int(ink_a.sum()), int(ink_z.sum())


def check_captions():
    r"""图注里那句「N 秒无声循环」必须对得上实测时长。

    ⭐⭐ 2026-09-18 现场：给 memtime 加了回退段，片长从 12.8 秒变成 14.6 秒，
      图注里那个 12.8 **一声不响地过期了**；同一轮里 descend 的图注还写着
      「16 秒」，实际 15.3。
    ⭐ 判据：**描述生成物的数字，靠人记住必然会过期** ——&#160;它跟注释里的数
      是同一类东西（都是断言），区别只是这一条**读者看得见**。
    ⛔ 容差 0.6 秒：图注写整数是刻意的，没必要逼着写 15.33。
    """
    import re
    pages = glob.glob(os.path.join(os.path.dirname(MEDIA), "*.html"))
    dur = {}
    for p in glob.glob(os.path.join(MEDIA, "*.mp4")):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", p], capture_output=True, text=True)
        dur[os.path.basename(p)] = float(out.stdout.strip())
    bad = []
    for page in pages:
        s = open(page, encoding="utf-8").read()
        for fig in re.findall(r"<figure\b.*?</figure>", s, re.S):
            m = re.search(r'src="media/([^"]+\.mp4)"', fig)
            if not m or m.group(1) not in dur:
                continue
            # ⛔ 第一版写的是 `（\s*([0-9.]+)\s*秒` ——&#160;要求左括号**紧挨着**数字，
            #   于是 memtime 那条「（……\n  14.6 秒无声循环」直接漏掉，
            #   两条过期只抓出一条。
            # ⭐ 判据：**锚点要选语义短语，别选标点** ——&#160;标点的位置随排版漂，
            #   「秒无声循环」这五个字才是这句话真正的身份。
            real = dur[m.group(1)]
            for said in re.findall(r"([0-9.]+)\s*秒无声循环", fig):
                if abs(real - float(said)) > 0.6:
                    bad.append("%s 的图注说 %s 秒，实测 %.1f 秒"
                               % (m.group(1), said, real))
    if bad:
        for b in bad:
            print("   ⛔ %s" % b)
    else:
        print("   ✅ 每段动画图注里的秒数都对得上实测时长")
    return len(bad)


def main(argv):
    update = "--update" in argv
    argv = [a for a in argv if a != "--update"]
    files = argv or sorted(glob.glob(os.path.join(MEDIA, "*.mp4")))
    base = json.load(open(BASE, encoding="utf-8")) if os.path.exists(BASE) else {}

    print("\n\033[1m▸ 动画首尾一致性\033[0m   基线回归：只问「有没有比记录的更糟」")
    fail = 0
    for p in files:
        key = os.path.basename(p)[:-4]
        bad, ia, iz = measure(p)
        rec = base.get(key)
        if rec is None:
            print("   ⛔ %-22s 不一致度 %5.1f%%   \033[1m没有基线\033[0m —— "
                  "去看 /tmp/loopdiff-%s.png，判过了再 --update" % (key, bad, key))
            fail += 1
        elif bad > rec["value"] + SLACK:
            print("   ⛔ %-22s 不一致度 %5.1f%%   \033[1m比基线 %.1f%% 糟了\033[0m"
                  " —— 看 /tmp/loopdiff-%s.png" % (key, bad, rec["value"], key))
            fail += 1
        else:
            print("   ✅ %-22s 不一致度 %5.1f%%（基线 %.1f%%）  %s"
                  % (key, bad, rec["value"], rec.get("note", "")))
        if update:
            base[key] = {"value": round(bad, 1),
                         "note": (rec or {}).get("note", "⚠️ 未经人眼判定"),
                         "ink": [ia, iz]}
    if update:
        json.dump(base, open(BASE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, sort_keys=True)
        print("   📌 已写回基线 %s" % os.path.relpath(BASE, os.getcwd()))
        return 0
    fail += check_captions()
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
