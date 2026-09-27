"""把提示表里的 at（台词开头几个字）换成秒数：在 whisper 逐字时间轴里顺序模糊查找。"""
import json, re, sys, difflib
cues_in, words_in, out = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(cues_in)); segs = json.load(open(words_in))
norm = lambda s: re.sub(r"[\s，。、；：！？「」（）,.!?:;\"'…—\-]", "", s)
chars, times = [], []
for sg in segs:
    for w in sg["words"]:
        t = norm(w["w"])
        for k, ch in enumerate(t):
            chars.append(ch); times.append(w["s"] + (w["e"] - w["s"]) * k / max(1, len(t)))
stream = "".join(chars)
pos = 0
for c in cfg["cues"]:
    key = norm(c.pop("at"))
    j = stream.find(key, pos)
    if j < 0 or j - pos > 1500:
        best, bj = 0, pos
        for s0 in range(pos, min(len(stream) - len(key), pos + 1500)):
            r = difflib.SequenceMatcher(None, key, stream[s0:s0 + len(key)]).ratio()
            if r > best: best, bj = r, s0
        j = bj
        print("  模糊 %.2f  %s → %s" % (best, key, stream[j:j + len(key)]))
    c["t"] = round(max(0.0, times[j] - 0.15), 2)
    pos = j + 1
    print("%7.2f  %s" % (c["t"], key))
cfg["cues"][0]["t"] = 0.0
json.dump(cfg, open(out, "w"), ensure_ascii=False, indent=1)
