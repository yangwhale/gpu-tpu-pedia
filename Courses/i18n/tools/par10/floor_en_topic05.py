"""Recount the English floor table in Topic 5 lecture chunk 00 from chunks 00-07 (sections c0..c8).
Spoken lines = <p class="say...">; English words at 140 words/min (~ zh 200 chars/min x 0.70 words per char);
15 s per <p class="board">. Like the zh build (topic05-build-lecture.py) no cell is coloured; overruns are only printed. Idempotent. Run: python3 /tmp/t05lec/floor_en.py  (add zh to verify zh counts)"""
import re, sys, html as H
D = '/tmp/t05lec/chunks/'
WPM = 140
ZH = len(sys.argv) > 1 and sys.argv[1] == 'zh'
lang = 'zh' if ZH else 'en'
s = "".join(open(f"{D}{i:02d}.{lang}.html").read() for i in range(8))
def say(t): return " ".join(H.unescape(re.sub(r"<[^>]+>", " ", x)) for x in re.findall(r'<p class="say[^"]*">(.*?)</p>', t, re.S))
def words(t): return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,×%-]*", say(t)))
def hanzi(t): return len(re.findall(r"[一-鿿]", say(t)))
heads = [(m.start(), re.sub(r"<[^>]+>", "", m.group(1))) for m in re.finditer(r'<h3 class="sec"[^>]*id="c\d"[^>]*>(.*?)</h3>', s)]
assert len(heads) == 9, len(heads)
heads.append((s.index("🚧"), ""))
rows, fl, plan_sum = [], 0.0, 0
for i in range(9):
    t = s[heads[i][0]:heads[i + 1][0]]
    n = hanzi(t) if ZH else words(t); c = len(re.findall(r'<p class="board">', t))
    f = n / (200 if ZH else WPM) + c * 0.25; fl += f
    plan = int(re.search(r"[(（](\d+)′[)）]", heads[i][1]).group(1)); plan_sum += plan
    name = re.sub(r"\s*[(（]\d+′[)）]\s*$", "", heads[i][1])
    bad = ''  # Topic 5 zh build lists overruns in the build log only; no red cell in the table
    over = f > plan
    rows.append(f'<tr><td>{name}</td><td>{n:,}</td><td>{c}</td><td{bad}>{f:.1f}′</td><td>{plan}′</td></tr>')
    print(f'{name[:50]:50s} {n:6,} {c:3d} {f:5.1f} {plan}{"  (over plan)" if over else ""}')
print('floor total', round(fl), 'plan', plan_sum)
if ZH: sys.exit()
tab = ('<table><thead><tr><th>Section</th><th>Script (words)</th><th>Figures</th><th>Floor</th><th>Planned</th></tr></thead><tbody>'
       + "".join(rows) + f'<tr><td><b>Total</b></td><td colspan="2"></td><td><b>{fl:.0f}′</b></td><td><b>{plan_sum}′</b></td></tr></tbody></table>')
p = D + '00.en.html'; e = open(p).read()
e, k = re.subn(r'<table><thead><tr><th>Section</th><th>Script \((?:zh chars|words)\)</th>.*?</table>', lambda m: tab, e, count=1, flags=re.S); assert k == 1
open(p, 'w').write(e)
