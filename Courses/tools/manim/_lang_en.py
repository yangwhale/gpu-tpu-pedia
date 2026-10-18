# -*- coding: utf-8 -*-
r"""英文版动画的公共壳：不改中文脚本一个字，给画面里的字换成英文再渲。

⭐ 用法：每个中文脚本 `topicNN-anim-xxx.py` 旁边放一个 `topicNN-anim-xxx-en.py`，
  里面只有一张对照表 EN（中文原句 → 英文）和可选的 PAT（正则 → 英文模板），最后一行
  `run_en(__file__, EN, PAT, globals())`。渲染时把 `-en.py` 当脚本、场景名不变：

    构建手册/脚本/render.sh tools/manim/topic05-anim-coll-en.py Ring WebPages/media/topic05-ring.en.mp4

⭐ 原理：先把 `manim.Text` 换成一个会查表的子类，再把中文脚本的源码 exec 进本模块。
  中文脚本里的 `from manim import Text` 拿到的就是查表版，布局、时序、字号全部原样 ——
  所以英文片和中文片**同长、同停顿点**（`data-pauses` 不用改）。

⛔ 查不到的中文**直接报错，不兜底**：漏翻一句就该渲不出来，而不是带着中文上线。
⛔ 停顿点文件（steps/<Scene>.json）是中文管线的产物，英文渲染改写到 /tmp/manim-en-steps/，
  渲完可以拿来跟中文那份比对，确认时序没变。

环境变量：
  EN_COLLECT=<文件>   收集模式：不报错，把遇到的中文句子（含已翻/未翻）写进这个 json
  EN_MEASURE=<文件>   量宽模式：每句同时按中文排一次，记下中英两版的宽度，用来找挤出画面的句子
  EN_LINT=<文件>      版面体检：每次 play / wait 前后扫一遍场上的字，记下「出画面」「字压字」
                      「字撑出所在的框」三类问题（EN_IDENTITY=1 时不翻译，拿中文版当对照）
"""
import json
import os
import re

import manim

CJK = re.compile(r"[　-〿一-鿿＀-￯]")
_ORIG_TEXT = manim.Text
_ORIG_MARKUP = manim.MarkupText
_ORIG_MATHTEX = manim.MathTex
STEPS_EN = "/tmp/manim-en-steps"
SMALL_PT = 32          # 低于这个字号先按两倍排再缩一半（Pango 小字号吞空格，见 Text 覆盖）


def run_en(en_file, table, patterns, g):
    src = os.path.join(os.path.dirname(os.path.abspath(en_file)),
                       os.path.basename(en_file).replace("-en.py", ".py"))
    collect = os.environ.get("EN_COLLECT")
    measure = os.environ.get("EN_MEASURE")
    seen, widths = {}, []
    pats = [(re.compile(p), r) for p, r in (patterns or [])]

    identity = os.environ.get("EN_IDENTITY")
    lint = os.environ.get("EN_LINT")
    issues = {}

    def tr(s):
        if identity:
            return s
        if not isinstance(s, str) or not CJK.search(s):
            return s
        if s in table:
            out = table[s]
        else:
            out = None
            for rx, rep in pats:
                m = rx.fullmatch(s)
                if m:
                    out = m.expand(rep)
                    break
        if collect:
            seen[s] = out
            if out is None:
                return s
        if out is None:
            raise KeyError("英文表里没有这句（%s）：%r" % (os.path.basename(src), s))
        if CJK.search(out):
            raise ValueError("译文里还有中文：%r → %r" % (s, out))
        return out

    class Text(_ORIG_TEXT):
        def __init__(self, text, *a, **kw):
            en = tr(text)
            kw2 = dict(kw)
            for key in ("t2c", "t2w", "t2s", "t2f", "t2g"):
                if kw2.get(key):
                    kw2[key] = {tr(k): v for k, v in kw2[key].items()}
            fs = kw2.get("font_size")
            if fs is not None and fs < SMALL_PT and not identity:
                # ⛔ Pango 在小字号（实测约 < 30）会吞掉英文词间空格：「uphill, loss increases」
                #   排成「uphill,lossincreases」。按两倍字号排、再缩一半，最终大小不变。
                kw2["font_size"] = fs * 2
                super().__init__(en, *a, **kw2)
                self.scale(0.5)
            else:
                super().__init__(en, *a, **kw2)
            if measure and en != text:
                zh = _ORIG_TEXT(text, *a, **kw)
                widths.append({"zh": text, "en": en, "zh_w": round(zh.width, 3),
                               "en_w": round(self.width, 3), "zh_h": round(zh.height, 3),
                               "en_h": round(self.height, 3)})

    class MarkupText(_ORIG_MARKUP):
        def __init__(self, text, *a, **kw):
            en, fs = tr(text), kw.get("font_size")
            if fs is not None and fs < SMALL_PT and not identity:
                super().__init__(en, *a, **dict(kw, font_size=fs * 2))
                self.scale(0.5)
            else:
                super().__init__(en, *a, **kw)

    class MathTex(_ORIG_MATHTEX):
        def __init__(self, *tex, **kw):
            super().__init__(*[tr(t) for t in tex], **kw)

    manim.Text, manim.MarkupText, manim.MathTex = Text, MarkupText, MathTex

    if lint:
        _install_lint(Text, issues)
    g["__file__"] = src                       # 中文脚本里 open(__file__) / inspect 读的是它自己
    code = compile(open(src, encoding="utf-8").read(), src, "exec")
    exec(code, g)
    os.makedirs(os.path.join(STEPS_EN, "steps"), exist_ok=True)
    if "STEPS_DIR" in g:
        g["STEPS_DIR"] = os.path.join(STEPS_EN, "steps")
    g["__file__"] = os.path.join(STEPS_EN, os.path.basename(src))   # 运行期拼 steps 目录的也落到 /tmp

    import atexit

    def _dump():
        if collect:
            old = json.load(open(collect, encoding="utf-8")) if os.path.exists(collect) else {}
            old.update(seen)
            json.dump(old, open(collect, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if lint:
            json.dump(sorted(issues.values(), key=lambda r: r["t"]), open(lint, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        if measure:
            json.dump(widths, open(measure, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    atexit.register(_dump)


def _install_lint(TextCls, issues):
    """每次 play / wait 前后扫一遍：出画面、两段字的框相交、字的框撑出包住它中心的那个矩形。"""
    import numpy as np
    from manim import Scene, Rectangle, Polygon, RoundedRectangle

    def box(m):
        return (m.get_left()[0], m.get_bottom()[1], m.get_right()[0], m.get_top()[1])

    def scan(scene):
        fam = []
        for m in scene.mobjects:
            fam.extend(m.get_family())
        texts = [m for m in fam if isinstance(m, TextCls) and m.width > 0 and m.get_fill_opacity() > 0.05]
        rects = [m for m in fam if isinstance(m, (Rectangle, Polygon, RoundedRectangle))
                 and m.width > 0.2 and m.height > 0.2]
        fw, fh = scene.camera.frame_width / 2, scene.camera.frame_height / 2
        cx, cy = (scene.camera.frame_center[:2] if hasattr(scene.camera, "frame_center") else (0, 0))
        t = round(scene.renderer.time, 2)

        def note(kind, a, b=""):
            key = (kind, a, b)
            if key not in issues:
                issues[key] = {"t": t, "kind": kind, "text": a, "other": b}
        for m in texts:
            l, b_, r, tp = box(m)
            if l < cx - fw + 0.05 or r > cx + fw - 0.05 or b_ < cy - fh + 0.05 or tp > cy + fh - 0.05:
                note("off-frame", m.text)
            c = m.get_center()
            best = None
            for R in rects:
                L, B, Rr, T = box(R)
                if L < c[0] < Rr and B < c[1] < T:
                    if best is None or R.width * R.height < best.width * best.height:
                        best = R
            if best is not None:
                L, B, Rr, T = box(best)
                if (l < L - 0.02 or r > Rr + 0.02) and m.width < best.width * 1.5 + 0.3:
                    note("spills-box", m.text, "%.2fx%.2f" % (best.width, best.height))
        for i, a in enumerate(texts):
            la, ba, ra, ta = box(a)
            for bb in texts[i + 1:]:
                if bb is a or a in bb.get_family() or bb in a.get_family():
                    continue
                lb, bbm, rb, tb = box(bb)
                ox = min(ra, rb) - max(la, lb)
                oy = min(ta, tb) - max(ba, bbm)
                if ox > 0.03 and oy > 0.03:
                    note("overlap", a.text, bb.text)

    _play, _wait = Scene.play, Scene.wait

    def play(self, *a, **k):
        scan(self)
        r = _play(self, *a, **k)
        scan(self)
        return r

    def wait(self, *a, **k):
        scan(self)
        return _wait(self, *a, **k)
    Scene.play, Scene.wait = play, wait
