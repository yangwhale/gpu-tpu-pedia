"""文字稿（s<N>.txt，带情绪标签）＋ whisper 逐字时间 → 句级 SRT：文字用原稿（术语准），时间用识别结果（全局对齐）。
用法：srt.py s<N>.txt s<N>-words.json s<N>.srt"""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmap import norm, stream_of, script_times
txt, words, out = sys.argv[1], sys.argv[2], sys.argv[3]
raw = re.sub(r"\[[a-z]+\]\s*", "", open(txt, encoding="utf-8").read())
sents = [s.strip() for s in re.split(r"(?<=[。？！])", raw.replace("\n", "")) if s.strip()]
script, offs = "", []
for s in sents:
    offs.append(len(script)); script += norm(s)
stream, times = stream_of(words)
st = script_times(script, stream, times)
starts = [st[min(o, len(st) - 1)] for o in offs]
end_all = times[-1] + 0.8
fmt = lambda t: "%02d:%02d:%02d,%03d" % (t // 3600, t % 3600 // 60, t % 60, (t - int(t)) * 1000)
# ⭐ TTS 偶尔会漏念整句（第零节「318 年」那两句就没念出来）：对齐后这种句子的时长几乎为零。
#   字幕按录音走 —— 没念的句子不出字幕，并打印出来提醒（要补就重录那一节）。
n_out, skipped = 0, []
with open(out, "w", encoding="utf-8") as f:
    for i, s in enumerate(sents):
        a = starts[i]; b = starts[i + 1] - 0.05 if i + 1 < len(sents) else end_all
        if i + 1 < len(sents) and (b - a) < max(0.5, 0.06 * len(norm(s))):
            skipped.append(s); continue
        n_out += 1
        f.write("%d\n%s --> %s\n%s\n\n" % (n_out, fmt(a), fmt(max(b, a + 0.5)), s))
print(len(sents), "句，出字幕", n_out, "句")
for s in skipped:
    print("  ⚠️ 录音里像是没念：", s)
