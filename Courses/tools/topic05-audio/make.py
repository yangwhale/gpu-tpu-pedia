#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""专题五讲课音频：把 s<N>.txt（一段一个情绪标签）合成一条 mp3，放进 WebPages/media/。

⭐ 2026-09-26 第二轮试讲：每一节讲的话生成音频，嵌进课件和讲义对应的那一节，点开就能听。
   声音用 Gemini TTS 的 orus（作者选定的默认声音）；一段一次调用，段间留 0.5 秒停顿。
⛔ 稿子只放讲给学生听的话，不放后台信息（改了什么、图的代号、互动提示）。

用法：python3 make.py 0          # 生成 media/topic05-lecture-s0.mp3
"""
import concurrent.futures as cf
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "..", "WebPages", "media")
TTS = os.path.expanduser("~/.claude/skills/tts-generator/scripts/tts-generate.py")


def tts(par):
    out = subprocess.run([TTS, par, "--voice", "orus"], capture_output=True, text=True, check=True)
    path = out.stdout.strip().splitlines()[-1]
    assert os.path.exists(path), out.stderr[-400:]
    return path


def main(n):
    pars = [p.strip() for p in open(os.path.join(HERE, "s%s.txt" % n), encoding="utf-8").read().split("\n\n") if p.strip()]
    with cf.ThreadPoolExecutor(4) as ex:
        oggs = list(ex.map(tts, pars))
    out = os.path.join(MEDIA, "topic05-lecture-s%s.mp3" % n)
    inputs, fl = [], []
    for i, o in enumerate(oggs):
        inputs += ["-i", o]
        fl.append("[%d:a]aresample=24000,apad=pad_dur=0.5[a%d]" % (i, i))
    fl.append("".join("[a%d]" % i for i in range(len(oggs))) + "concat=n=%d:v=0:a=1[out]" % len(oggs))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fl),
                    "-map", "[out]", "-ac", "1", "-b:a", "48k", out], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                               capture_output=True, text=True).stdout)
    print("ok  %s  %d 段  %.0f 秒  %s" % (os.path.basename(out), len(pars), dur,
                                        format(os.path.getsize(out), ",")))


if __name__ == "__main__":
    main(sys.argv[1])
