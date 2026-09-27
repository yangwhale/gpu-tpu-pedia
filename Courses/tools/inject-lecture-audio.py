#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""专题二、三、四：构建完之后，把讲课录音（和同名讲课视频）挂到每段对应的标题下面。

⭐ 2026-09-27：专题五的录音条写在构建脚本里；二、三、四的录音是一口气批量做的，
   挂载位置由写讲稿时一并产出的清单 lecture-audio-segs.json 决定（课件标题 / 讲义标题 / 条标题），
   所以做成构建后的一道工序，build-all.sh 最后一步调它。
⭐ 幂等：每块用 <!--lec:sN--> … <!--/lec:sN--> 包住，重跑先删旧块。录音文件不在的段跳过。

用法：python3 tools/inject-lecture-audio.py [topic02 topic03 ...]（不给就全做）
"""
import io, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from topic03_page import audio_block, bust_media

W = os.path.join(HERE, "..", "WebPages")


def norm(s):
    s = re.sub(r"<[^>]+>", "", s).replace("&nbsp;", " ").replace(" ", " ").replace("　", " ")
    return re.sub(r"\s+", " ", s.replace("⭐", "").replace("⚠️", "")).strip()


def heading_end(html, title):
    want = norm(title)
    for m in re.finditer(r"<(h2|h3|h4)\b[^>]*>(.*?)</\1>", html, re.S):
        if norm(m.group(2)) == want:
            return m.end()
    raise SystemExit("找不到标题：%s" % title)


def inject(path, pfx, segs, key):
    html = io.open(path, encoding="utf-8").read()
    html = re.sub(r"<!--lec:s[0-9a-z]+-->.*?<!--/lec:s[0-9a-z]+-->", "", html, flags=re.S)
    n = 0
    for d in segs:
        if not d.get(key):
            continue
        blk = audio_block(path, "%s-lecture-%s.mp3" % (pfx, d["id"]), d["label"], video="%s-video-%s.mp4" % (pfx, d["id"]))
        if not blk:
            continue
        at = heading_end(html, d[key])
        html = html[:at] + "<!--lec:%s-->%s<!--/lec:%s-->" % (d["id"], blk, d["id"]) + html[at:]
        n += 1
    io.open(path, "w", encoding="utf-8").write(bust_media(html, path))
    print("ok  %s  挂了 %d 段录音" % (os.path.basename(path), n))


cfg = json.load(open(os.path.join(HERE, "lecture-audio-segs.json"), encoding="utf-8"))
for pfx in (sys.argv[1:] or list(cfg)):
    c = cfg[pfx]
    inject(os.path.join(W, c["course"]), pfx, c["segs"], "course_heading")
    inject(os.path.join(W, c["lecture"]), pfx, c["segs"], "lecture_heading")
