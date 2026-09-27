"""把提示表里的 at（台词开头几个字，照抄文字稿）换成秒数。
先在文字稿里按顺序找到 at，再用全局对齐拿到那个字的时间。
用法：resolve.py s<N>-cues.json s<N>-words.json s<N>.txt 输出.json"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmap import norm, stream_of, script_times
cues_in, words_in, txt, out = sys.argv[1:5]
cfg = json.load(open(cues_in))
script = norm(open(txt, encoding="utf-8").read())
stream, times = stream_of(words_in)
st = script_times(script, stream, times)
pos = 0
for c in cfg["cues"]:
    key = norm(c.pop("at"))
    j = script.find(key, pos)
    if j < 0:
        raise SystemExit("文字稿里找不到 at（要照抄原稿，且按出现顺序）：%s" % key)
    c["t"] = round(max(0.0, st[j] - 0.15), 2)
    pos = j + 1
    print("%7.2f  %s" % (c["t"], key))
cfg["cues"][0]["t"] = 0.0
json.dump(cfg, open(out, "w"), ensure_ascii=False, indent=1)
