import sys, json
from faster_whisper import WhisperModel
audio, out = sys.argv[1], sys.argv[2]
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=64)
segs, info = m.transcribe(audio, language="zh", word_timestamps=True, vad_filter=False, beam_size=5,
                          initial_prompt="并行策略 AllReduce AllGather ReduceScatter AllToAll GPU TPU 卡")
res = []
for s in segs:
    res.append({"start": s.start, "end": s.end, "text": s.text,
                "words": [{"w": w.word, "s": w.start, "e": w.end} for w in (s.words or [])]})
    print("%.1f-%.1f %s" % (s.start, s.end, s.text), flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=0)
