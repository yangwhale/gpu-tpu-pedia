"""文字稿（s<N>.txt，带情绪标签）＋ whisper 逐字时间 → 句级 SRT：文字用原稿（术语准），时间用识别结果。"""
import json, re, sys, difflib
txt, words, out = sys.argv[1], sys.argv[2], sys.argv[3]
raw = re.sub(r"\[[a-z]+\]\s*", "", open(txt, encoding="utf-8").read())
sents = [s.strip() for s in re.split(r"(?<=[。？！])", raw.replace("\n", "")) if s.strip()]
norm = lambda s: re.sub(r"[\s，。、；：！？「」（）,.!?:;\"'…—\-]", "", s)
chars, times = [], []
for sg in json.load(open(words)):
    for w in sg["words"]:
        t = norm(w["w"])
        for k, ch in enumerate(t):
            chars.append(ch); times.append(w["s"] + (w["e"] - w["s"]) * k / max(1, len(t)))
stream = "".join(chars); pos = 0; starts = []
for s in sents:
    key = norm(s)[:10]
    j = stream.find(key, pos)
    if j < 0 or j - pos > 800:
        best, j = 0, pos
        for s0 in range(pos, min(len(stream) - len(key), pos + 800)):
            r = difflib.SequenceMatcher(None, key, stream[s0:s0 + len(key)]).ratio()
            if r > best: best, j = r, s0
    starts.append(times[j]); pos = j + max(1, len(norm(s)) // 2)
end_all = times[-1] + 0.8
fmt = lambda t: "%02d:%02d:%02d,%03d" % (t // 3600, t % 3600 // 60, t % 60, (t - int(t)) * 1000)
with open(out, "w", encoding="utf-8") as f:
    for i, s in enumerate(sents):
        a = starts[i]; b = starts[i + 1] - 0.05 if i + 1 < len(sents) else end_all
        f.write("%d\n%s --> %s\n%s\n\n" % (i + 1, fmt(a), fmt(max(b, a + 0.5)), s))
print(len(sents), "句")
