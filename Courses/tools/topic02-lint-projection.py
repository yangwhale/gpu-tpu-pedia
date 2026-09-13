# -*- coding: utf-8 -*-
r"""投屏体检：把每张图**投到会议室屏幕上之后**的最小字号算出来。

⛔⛔ 2026-09-13 二轮学生审稿量出来的：34 张图里**没有一张**能在 1280×720 上
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

════════════════════════════════════════════════════════════════════
⛔ 2026-09-14 R37：这个「8 张不合格」是噪声，而且它掩护了四个月
════════════════════════════════════════════════════════════════════
原来 build 里只打印**最差的 8 张**（`rows[:8]`），于是所有人记住的是
「有 8 张不合格」。**实际低于 9px 的是 62 张**（共 88 张）——&nbsp;
真实规模从来没出现在输出里。这是本仓库第 N 次栽在「只打印了一部分」。

更要紧的是：上面那句「讲义里那些『指第二格』的指令本来就是按这个节奏写的」
**是个假设，不是事实** ——&nbsp;从写下起没被查过一次。查了之后：
  · 62 张里有 **29 张讲义压根没点过名**（例：`fig3-capacity` 在 §1.3b）。
    ⭐ **一张不会被拿出来看的图，投屏读数是噪声。**
  · 剩下 33 张，逐段读讲稿，**处方确实在被执行** ——&nbsp;
    讲义是一格一格走的，没发现哪张是整张投上去不管的。

════════════════════════════════════════════════════════════════════
⛔⛔ 试着把这条处方自动化，失败了三次 ——&nbsp;这个失败本身值得记下来
════════════════════════════════════════════════════════════════════
想加一条「讲义有没有给分格指令」的机器判定，正则改了三版：
  ① 只认「指第 N 格 / 指 ③」 → 漏掉「先指**最左边那一格**」「指**左栏**」
  ② 放宽成「指」不接示/令/向/出/的 → 还是漏掉「**切** fig3-tpu-fix」
  ③ 再放宽 → 仍漏掉「**左边这一列**…**右边这一列**」（根本没有动词）
每一版都只是**把一批误报换成另一批**，没有收敛。

⭐⭐ 判据：**如果每修一次匹配规则都只是换一批误报、而不收敛，
   那说明这不是词表不全，是这个判断本身不是词法问题。**
   到这一步就该把 lint 从「判定」降级成「清单」——&nbsp;
   宁可让人看一眼，也不要挂一个会误报的 ⛔
   （本文件的兄弟 `topic02-lint-cues.py` 头上写着：误报会把真问题淹掉）。

所以现在这条 lint 只做一件它做得准的事：**把 62 张分成「讲义点过名的」
和「没点过名的」**，前者列出来供人扫一眼，后者说明投屏数对它无意义。
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

# ── 为什么这里**没有**一条「讲义写没写分格指令」的正则 ──────────────
# ⛔⛔ 写过三版，全删了。三版分别漏掉的真实写法：
#     「先<b>指最左边那一格</b>」（fig3-mqa-why）—— v1 只认「指第 N 格」
#     「<b>切</b> fig3-tpu-fix。三招只念共同形状」—— v2 只认动词「指」
#     「<b>左边这一列</b>是事后的…<b>右边这一列</b>是从头的」（fig3-when-axis）
#                                   —— v3 仍然要求有动词，而这句压根没有
# ⭐⭐ 判据：**每修一版只是把一批误报换成另一批，没有收敛** ——&#160;
#    说明要判的东西（「有没有引导观众看局部」）本来就不是词法特征。
#    这时候正确的动作是**把 lint 降级成清单**，不是再改一版正则。
#    误报会把真问题淹掉，这是本文件兄弟脚本头上写着的第一条。


def _lectures():
    """把所有讲义读成一份 {文件名: 正文}。图归谁不用猜 —— 全都搜一遍。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(W, "*-lecture.html"))):
        out[os.path.basename(p)] = io.open(p, encoding="utf-8").read()
    return out


# ⛔ 这里曾经有个 `n_panels()`：数 SVG 里的 ①②③ 当格数。
#   看着比匹配散文可靠（量的是图不是文字），**其实是同一个病的第四次**：
#   `fig3-knob2` 有**五张 mask 面板**（全注意力/滑窗/+sink/DSA/CSA），
#   但它们是用标题命名的、一个圈码都没有 → 被判成「单格图，得拆图」。
# ⭐⭐ 惯例**只在被强制的地方**才是可靠信号。build 里没有任何一条断言
#    要求面板必须带圈码，所以圈码的缺席什么都不说明。已删。


def stage_status(svg_name, svg_text, lecs):
    """讲义有没有点过这张图的名 —— **只判这一件，因为只有这件判得准。**

    返回 ("stage"|"read", 说明)。
    ⚠️ 图名有两套写法：SVG 叫 `fig3-capacity`，课件 id 叫 `fig-capacity`，
      讲义两种都可能写 ——&nbsp;跟 topic02-lint-cues 那边同一个坑，一起认。
    """
    base = svg_name[:-4] if svg_name.endswith(".svg") else svg_name
    alts = {base, re.sub(r'^fig\d+-', 'fig-', base)}
    for fn, txt in sorted(lecs.items()):
        if any(a in txt for a in alts):
            return "stage", fn
    return "read", ""


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
                     min(sizes), sc, s))
    return sorted(rows)


def main():
    rows = scan()
    if not rows:
        print("   （没找到 SVG，跳过）")
        return
    bad = [r for r in rows if r[0] < FLOOR]
    print("   共 %d 张图；整张投到 %d×%d 时，最小字 < %.0fpx 的有 %d 张"
          % (len(rows), PROJ_W, PROJ_H, FLOOR, len(bad)))
    print("     （只列最矮的 8 张；⚠️ 是「不能整张投」的读数，**不是缺陷**）")
    for a, f, w, h, mn, sc, _ in rows[:8]:
        # ⛔ 这里原来用 ⛔。整轮 R37 的结论就是「低于地板是个读数不是缺陷」，
        #   图标却还在喊缺陷 —— 口径不一致会把真正的 ⛔ 淹掉。换成 ⚠️。
        flag = "⚠️" if a < FLOOR else "✅"
        print("     %s %5.1f px   %-28s %.0f×%-4.0f  源 %.1fpx × %.2f"
              % (flag, a, f, w, h, mn, sc))
    if not bad:
        return
    print("   ⭐ <%.0fpx 不代表图错了 —— 代表**这张图不能整张投**，"
          "要一次放大一格讲。" % FLOOR)

    lecs = _lectures()
    stage, read = [], []
    for a, fnm, w, h, mn, sc, txt in bad:
        st, fn = stage_status(fnm, txt, lecs)
        (stage if st == "stage" else read).append((a, fnm, fn))
    print("     其中 **%d 张讲义点了名**（会被拿到台上，必须分格讲），"
          "%d 张只读（投屏数对它们没意义）。" % (len(stage), len(read)))
    by = {}
    for a, fnm, fn in stage:
        by.setdefault(fn, []).append(fnm)
    for fn in sorted(by):
        print("     · %-28s %d 张：%s" % (fn, len(by[fn]),
                                          "、".join(sorted(by[fn])[:4])
                                          + ("…" if len(by[fn]) > 4 else "")))
    # ⚠️ 这里**不能**写 `&nbsp;` / `&#160;` —— 那是 docstring 和 HTML 里的写法，
    #   print 出去是终端上的一串乱码。全仓库就漏过这一处。
    print("   ⚠️ 这是**待人扫一眼的名单，不是缺陷列表** —— 「讲义有没有引导看局部」")
    print("     判不了（三版正则全在误报，第四版栽在数圈码上，见文件头）。")
    print("     这里只负责把范围缩到这几张，判断留给人。")


if __name__ == "__main__":
    main()
