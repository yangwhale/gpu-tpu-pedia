# -*- coding: utf-8 -*-
"""专题八的图：字号地板局部抬到 13.1（2026-11-16 负责人定：只在专题八局部抬，不碰共用基元）。

为什么是 13.1：图的 viewBox 宽 1400，1366 视口下实测渲染成 1290–1325 宽（缩 0.921–0.946），
SVG 里 11 号字落到 10.1–10.4px；手册的判据是渲染后 ≥ 12px，13.1 × 0.921 ≈ 12.06。
（夜里估的 12.6 是按 1334 宽算的，实测 12.6 只到 11.6–11.9px，不够。）
做法：本模块在 import 时把 topic03_draw 的 _sz（出字号）和 wpx（量宽、折行）换成带地板的版本，
所以渲染和折行用的是同一个字号，折行不会因为字变大而顶出框。
⛔ 必须在 `from topic03_draw import ... wpx` 之前 import，否则拿到的是旧的 wpx。
⛔ 只有专题八的图脚本 import 它；别的专题不受影响。
"""
import topic03_draw as _D

FLOOR = 13.1

if not getattr(_D, "_T08_FLOOR", None):
    _orig_sz, _orig_wpx = _D._sz, _D.wpx

    def _sz(n):
        return max(_orig_sz(n), FLOOR)

    def wpx(s, size=11.5):
        return _orig_wpx(s, max(size, FLOOR))

    _D._sz, _D.wpx, _D._T08_FLOOR = _sz, wpx, FLOOR

# 11-28 夜（R13 图小事）：小数一律 e 记法写成 e-4，不写 e-04；只在专题八局部改 Fig.t 的出字，不碰共用基元
import re as _re
if not getattr(_D, "_T08_EXP", None):
    _orig_t = _D.Fig.t

    def _t(self, x, y, s, *a, **k):
        if isinstance(s, str):
            s = _re.sub(r"(\d)e([-+])0(\d)", r"\1e\2\3", s)
        return _orig_t(self, x, y, s, *a, **k)

    _D.Fig.t, _D._T08_EXP = _t, True
    _orig_init = _D.Fig.__init__

    def _init(self, w, aria):
        if isinstance(aria, str):
            aria = _re.sub(r"(\d)e([-+])0(\d)", r"\1e\2\3", aria)
        return _orig_init(self, w, aria)

    _D.Fig.__init__ = _init
