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

# ⭐ 通用版：目录靠参数给，不写死项目结构。
#   `--media <目录>` 扫哪儿的 mp4（默认当前目录）
#   `--baseline <文件>` 基线存哪儿（默认 <media>/loop-baseline.json）
#   ⛔ `gpu-tpu-pedia/Courses/tools/manim/` 下有一份**钉在那个项目上的副本**，
#     已接进它的 build-all.sh。两份是故意的：那边是部署好的守卫，
#     这边是拿去装进新项目的模板。改了这边记得想想要不要同步过去。
MEDIA = os.getcwd()
BASE = None

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


def _bg(a):
    """从画面自己测背景色 ——&#160;取四个角 12x12 的中位数。

    ⛔⛔ 原来这里写死了「白」：`ink = (255 - a).max(2) > INK`。
      2026-09-19 把动画改成作者默认的**黑底**之后，整帧都满足「不是白」，
      于是分母从三千涨到五十万，不一致度塌成一个看着很漂亮的小数 ——&#160;
      **守卫在深色片子上静默失效，而且是往「看起来更好」的方向失效。**
    ⭐ 判据：**凡是「跟背景比」的度量，背景必须从画面里测，不能写死。**
      写死的那一刻，这个工具就只对一种配色有效了。
    """
    h, w = a.shape[:2]
    corners = np.concatenate([a[:12, :12].reshape(-1, 3), a[:12, -12:].reshape(-1, 3),
                              a[-12:, :12].reshape(-1, 3), a[-12:, -12:].reshape(-1, 3)])
    return np.median(corners, axis=0)


def measure(mp4):
    """返回 (不一致度 %, 首帧墨水, 末帧墨水)，并落一张上下拼接图。"""
    a, z = _frame(mp4, False), _frame(mp4, True)
    if a.shape != z.shape:
        return 100.0, 0, 0
    bg = _bg(a)
    ink_a = np.abs(a - bg).max(2) > INK
    ink_z = np.abs(z - bg).max(2) > INK
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
    pages = (glob.glob(os.path.join(os.path.dirname(MEDIA), "*.html"))
             + glob.glob(os.path.join(MEDIA, "*.html")))
    if not pages:
        return 0            # 没有页面引用它们，这一项不适用
    dur = {}
    for p in glob.glob(os.path.join(MEDIA, "*.mp4")):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", p], capture_output=True, text=True)
        # ⛔ 这一步会扫**整个 media 目录**。拿 /tmp 当草稿输出目录时，
        #   它会去 ffprobe 别人留在那儿的垃圾 mp4，拿到空串直接 ValueError 崩掉 ——
        #   而接缝那一步其实已经跑完并打印了结论，只是被 traceback 盖住。
        # ⭐ 判据：**顺带扫到的文件不该有能力让主流程失败**（跟读 .html 那处同一条）。
        try:
            dur[os.path.basename(p)] = float(out.stdout.strip())
        except ValueError:
            continue
    bad = []
    for page in pages:
        # ⛔ 这里会扫到旁边目录里任何 .html —— 包括不是 UTF-8 的（实测撞上过
        #   一个 gzip 过的 .html，当场 UnicodeDecodeError 把整条流程带崩）。
        #   ⭐ 判据：**顺带扫到的文件不该有能力让主流程失败。**
        try:
            s = open(page, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
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
            # ⛔ 锚点第二次放宽（2026-09-18 下午）：房规松绑后有了**不循环的长片**，
            #   它们的图注不会写「无声循环」四个字，于是这条 lint 对新片子**静默失效**。
            #   ⭐ 判据仍是「锚在语义短语上」，只是那个短语换成了两处都有的
            #     「…秒…Manim」——&#160;即「这段时长在描述一个 Manim 产物」。
            for said in re.findall(r"([0-9.]+)\s*秒[^）)]{0,24}Manim", fig):
                if abs(real - float(said)) > 0.6:
                    bad.append("%s 的图注说 %s 秒，实测 %.1f 秒"
                               % (m.group(1), said, real))
    if bad:
        for b in bad:
            print("   ⛔ %s" % b)
    else:
        print("   ✅ 每段动画图注里的秒数都对得上实测时长")
    return len(bad)


def _opt(argv, name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


def main(argv):
    global MEDIA, BASE
    argv = list(argv)
    update = "--update" in argv
    argv = [a for a in argv if a != "--update"]
    MEDIA = os.path.abspath(_opt(argv, "--media", MEDIA))
    BASE = os.path.abspath(_opt(argv, "--baseline",
                                os.path.join(MEDIA, "loop-baseline.json")))
    files = argv or sorted(glob.glob(os.path.join(MEDIA, "*.mp4")))
    # ⛔⛔ 2026-09-19：**空集合绝不能当成通过。**
    #   把通用版直接覆盖到项目里之后，默认目录从 `WebPages/media` 变成了 `cwd`，
    #   而 build 是在 `tools/` 下调它的 —— 扫不到任何 mp4，`fail` 保持 0，
    #   于是**守卫连着好几轮报绿，其实一支片子都没查**。
    #   ⭐ 判据：**「没找到要检查的东西」是配置错，不是检查通过。**
    if not files:
        print("   ⛔ 在 %s 下一个 mp4 都没找到 —— 这是路径配错了，不是通过。"
              % MEDIA)
        print("      用 --media 指到放 mp4 的目录。")
        return 1
    base = json.load(open(BASE, encoding="utf-8")) if os.path.exists(BASE) else {}

    print("\n\033[1m▸ 动画首尾一致性\033[0m   基线回归：只问「有没有比记录的更糟」")
    fail = 0
    for p in files:
        key = os.path.basename(p)[:-4]
        bad, ia, iz = measure(p)
        rec = base.get(key)
        # ⭐⭐ 2026-09-18：房规松绑后有了**不循环的长片**（>20 秒、带叙事、
        #   页面上给 controls 不给 loop）。它们首尾本来就不该一样 ——&#160;
        #   强行要求首尾同帧会逼着长片在结尾把内容全撤掉，那是削足适履。
        #   ⭐ 判据：**守卫要守的是「承诺」，不是「形状」** ——&#160;
        #     片子承诺了 loop 才查首尾；baseline 里写 "loop": false 就跳过。
        if rec is not None and rec.get("loop") is False:
            # ⛔⛔ 2026-09-20 现场收紧：「能放一个动图搞定的，就不要放一个视频啦。」
            #   这个豁免以前被当成「叙事片天然不循环」的出口用 ——&#160;
            #   而 2026-09-20 实测：24.6 秒的叙事片照样能循环，
            #   只要结尾把画面恢复成第 0 帧的样子（见 SKILL.md 房规③ 的收尾模板）。
            #   ⭐ 判据：**「它首尾不一样」是现状，不是理由。**
            #     豁免必须写清「为什么这一支做不到复位」，不能只写「它是叙事片」。
            why = rec.get("note", "")
            print("   ⚠️  %-22s 声明了不循环 ——&nbsp;确认这是<必须>而不是<没做>" % key)
            print("      理由：%s" % (why[:100] if why else "⛔ 没写理由，这不该通过"))
            if not why:
                print("   ⛔ `loop: false` 必须在 note 里写清为什么做不到复位。")
                bad_any = True
            continue
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
