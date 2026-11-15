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
