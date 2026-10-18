# -*- coding: utf-8 -*-
"""topic05-anim-torus.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-05.en.html 里 3D 环面那张图的 aria-label 一致（块 → piece，逐轴 → axis by axis，轮换 → rotated）。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic05-anim-torus-en.py TorusRings WebPages/media/topic05-torus.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    "3D 环面上的 AllGather：三块数据轮换着走，三条轴一起忙":
        "AllGather on a 3D torus: rotate the pieces, keep all 3 axes busy",
    "每颗芯片手里这一块收齐了多少": "How much of each piece every chip holds",
    "用时（同一把尺子，单位：1/64 份过一条轴）": "Time (same scale; unit: 1/64 over one axis)",
    "逐轴": "serial",             # 用时条左边的小标签：0.2 到 1.25 之间只有 1 个单位宽，
    "轮换": "rotate",             #   "axis by axis" / "rotated" 都会顶到条上
    "4×4×4 切片，每颗芯片起手 1/64；每条线首尾相接成环（环回没画）":
        "4×4×4 slice, 1/64 per chip to start; each line is a ring (wraps not drawn)",
    "先看逐轴：三块挤在同一条轴上，X 走完才走 Y、再走 Z":
        "First, axis by axis: all three pieces crowd onto one axis, X, then Y, then Z",
    "逐轴走完：用时 63，大部分时间只有一条轴在干活":
        "Axis by axis done: time 63, with only one axis working most of the time",
    "再看轮换：三块各挑一条轴起步，A 走 XYZ、B 走 YZX、C 走 ZXY":
        "Now rotated: each piece starts on its own axis: A goes XYZ, B YZX, C ZXY",
    "轮换走完：用时 21 ＝ 逐轴的 1/3，三条轴的带宽全用上了":
        "Rotated done: time 21 = 1/3 of axis by axis, with all three axes' bandwidth used",
}
PAT = [
    (r"块 ([ABC])  (.*)", r"\1  \2"),          # 右边 3.0 起是进度条，"piece A" 放不下；块名靠颜色＋表头认
    (r"第 (\d) 步：只有 ([XYZ]) 方向在忙，([XYZ])、([XYZ]) 闲着", r"Step \1: only \2 is busy, \3 and \4 idle"),
    (r"第 (\d) 步完成：用时 (\d+)（这一步每块都涨到 (\d+/64)）", r"Step \1 done: time \2 (each piece grows to \3)"),
    (r"第 (\d) 步：X 搬 ([ABC])，Y 搬 ([ABC])，Z 搬 ([ABC])", r"Step \1: X carries \2, Y carries \3, Z carries \4"),
    (r"第 (\d) 步完成：用时 (\d+)，三条轴全亮（每块都涨到 (\d+/64)）",
     r"Step \1 done: time \2, all three axes lit (each piece grows to \3)"),
]
run_en(__file__, EN, PAT, globals())
