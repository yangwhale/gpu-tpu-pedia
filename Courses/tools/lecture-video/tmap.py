"""文字稿 ↔ 识别结果 的全局对齐：给文字稿里每个字一个时间。

⭐ 为什么要全局对齐：whisper 会把「百分之九十七」写成「97%」、「两千零四十八」写成「2048」、
   同音字写错（归约→规约）。逐句找开头会在这些地方跑偏，一偏后面全挤到结尾。
   整篇做一次 SequenceMatcher，对不上的段落按两头线性插值，就不会连锁出错。
"""
import difflib, json, re

NORM = re.compile(r"[\s，。、；：！？「」（）,.!?:;\"'…—\-·]")


def norm(s):
    return NORM.sub("", re.sub(r"\[[a-z]+\]", "", s))


def stream_of(words_json):
    chars, times = [], []
    for sg in json.load(open(words_json)):
        for w in sg["words"]:
            t = norm(w["w"])
            for k, ch in enumerate(t):
                chars.append(ch); times.append(w["s"] + (w["e"] - w["s"]) * k / max(1, len(t)))
    return "".join(chars), times


def script_times(script, stream, times):
    st = [None] * len(script)
    sm = difflib.SequenceMatcher(None, script, stream, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                st[i1 + k] = times[j1 + k]
        elif tag in ("replace", "delete"):
            a = times[min(j1, len(times) - 1)]
            b = times[min(max(j2 - 1, j1), len(times) - 1)] if tag == "replace" else a
            n = i2 - i1
            for k in range(n):
                st[i1 + k] = a + (b - a) * k / max(1, n)
    last = 0.0
    for i in range(len(st)):                       # 兜底：保持单调
        if st[i] is None or st[i] < last:
            st[i] = last
        last = st[i]
    i = 0                                          # 一整段字挤在同一时刻（「删除＋插入」没配成「替换」）→ 摊到下一个时间点之前
    while i < len(st):
        j = i
        while j + 1 < len(st) and st[j + 1] == st[i]:
            j += 1
        if j - i >= 3 and j + 1 < len(st):
            a, b = st[i], st[j + 1]
            for k in range(i, j + 1):
                st[k] = a + (b - a) * (k - i) / (j + 1 - i)
        i = j + 1
    return st
