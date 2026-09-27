import sys, json
from faster_whisper import WhisperModel
audio, out = sys.argv[1], sys.argv[2]
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=int(sys.argv[3]) if len(sys.argv) > 3 else 64)
segs, info = m.transcribe(audio, language="zh", word_timestamps=True, vad_filter=False, beam_size=5,
                          initial_prompt="讲课 GPU TPU AllReduce AllGather ReduceScatter AllToAll FSDP TP PP EP")
res = []
for s in segs:
    res.append({"start": s.start, "end": s.end, "text": s.text,
                "words": [{"w": w.word, "s": w.start, "e": w.end} for w in (s.words or [])]})
    print("%.1f-%.1f %s" % (s.start, s.end, s.text), flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=0)
