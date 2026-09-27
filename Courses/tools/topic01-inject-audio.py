#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""专题一没有构建脚本（课件与讲义是成品 HTML），讲课录音条直接插进页面。

⭐ 2026-09-27：跟专题五一样，每段录音挂在它讲的那一节（或小节）标题下面；
   有同名讲课视频（media/topic01-video-sN.mp4）就录音、视频各占一半并排。
⭐ 幂等：每块用 <!--lec:sN--> … <!--/lec:sN--> 包住，重跑先删旧块再插。
   录音文件不在的段自动跳过，所以可以录一段插一段。

用法：python3 tools/topic01-inject-audio.py
"""
import io, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topic03_page import audio_block, bust_media

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WebPages")
# 段号 → (课件里挂在哪个标题下, 讲义里挂在哪个标题下, 录音条标题)
SEGS = [
    ("s0", "开始之前，先猜一个数", "先定尺子：这门课怎么算账", "本节讲课录音 · 开场"),
    ("s1", "从文字到向量", "一个字怎么变成一串数", "本节讲课录音"),
    ("s2", "MLA：注意力，以及 KV cache 的战争", "KV cache 为什么会失控", "MLA 讲课录音 · 上"),
    ("s3", "一个容易想当然的地方：7,168 和 16,384", "7,168 和 16,384：一条其实不存在的约束", "MLA 讲课录音 · 中"),
    ("s4", "Q、K、V 都到手了：这一仗到底怎么打", "打分那一仗：从点积走回残差流", "MLA 讲课录音 · 下"),
    ("s5", "Dense MLP：先把「不稀疏」讲清楚", "Dense MLP 的四条好性质", "本节讲课录音"),
    ("s6", "MoE：671B 的主体", "MoE 原理 ·「专家」不是那种专家", "MoE 讲课录音 · 上"),
    ("s7", "4c · 分布式：挪 token，还是挪权重", "挪 token 还是挪权重", "MoE 讲课录音 · 下"),
    ("s8", "层间：残差流这条总线", "残差流是总线不是管道", "本节讲课录音"),
    ("s9", "出口：从向量回到文字", "出口：最吓人的临时张量", "本节讲课录音"),
    ("s10", "把账合起来", "合账 · 算出来 vs 量出来 · 抛钩子", "本节讲课录音"),
]


def heading_end(html, title):
    """找到文字等于 title 的 h2/h3，返回它闭合标签之后的位置。"""
    for m in re.finditer(r"<(h2|h3)\b[^>]*>(.*?)</\1>", html, re.S):
        t = re.sub(r"<[^>]+>", "", m.group(2))
        t = re.sub(r"\s+", " ", t).replace("⭐", "").replace("⚠️", "").strip()
        if t == title:
            return m.end()
    raise SystemExit("找不到标题：%s" % title)


def inject(path, col):
    html = io.open(path, encoding="utf-8").read()
    html = re.sub(r"<!--lec:s\d+-->.*?<!--/lec:s\d+-->", "", html, flags=re.S)
    n = 0
    for seg in SEGS:
        sid, title, label = seg[0], seg[col], seg[3]
        blk = audio_block(path, "topic01-lecture-%s.mp3" % sid, label, video="topic01-video-%s.mp4" % sid)
        if not blk:
            continue
        at = heading_end(html, title)
        html = html[:at] + "<!--lec:%s-->%s<!--/lec:%s-->" % (sid, blk, sid) + html[at:]
        n += 1
    html = bust_media(html, path)
    io.open(path, "w", encoding="utf-8").write(html)
    print("ok  %s  挂了 %d 段录音" % (os.path.basename(path), n))


inject(os.path.join(W, "topic-01.html"), 1)
inject(os.path.join(W, "topic-01-lecture.html"), 2)
