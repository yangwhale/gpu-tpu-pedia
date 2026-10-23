"""Recount the English floor table + time math in lecture chunk 00 from chunks 00-06.
Spoken lines = <p class="say...">; English words at 140 words/min (~ zh 200 chars/min x 0.70 words per char);
15 s per <p class="board">. Red floor cell = English floor > the section's planned minutes (same rule as the
zh build, applied to the English numbers). Idempotent. Run: python3 /tmp/t04lec/floor_en.py"""
import re, html as H
D = '/tmp/t04lec/chunks/'
WPM = 140
s = "".join(open(f"{D}{i:02d}.en.html").read() for i in range(7))
def say(t): return " ".join(H.unescape(re.sub(r"<[^>]+>", " ", x)) for x in re.findall(r'<p class="say[^"]*">(.*?)</p>', t, re.S))
def words(t): return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,×%-]*", say(t)))
heads = [(m.start(), re.sub(r"<[^>]+>", "", m.group(1))) for m in re.finditer(r'<h3 class="sec"[^>]*id="c\d"[^>]*>(Section \d.*?)</h3>', s)]
assert len(heads) == 7, len(heads)
heads.append((s.index("🚧 Where these notes"), ""))
rows, fl, plan_sum = [], 0.0, 0
for i in range(7):
    t = s[heads[i][0]:heads[i + 1][0]]
    n = words(t); c = len(re.findall(r'<p class="board">', t))
    f = n / WPM + c * 0.25; fl += f
    plan = int(re.search(r"\((\d+)′\)", heads[i][1]).group(1)); plan_sum += plan
    name = re.sub(r"\s*\(\d+′\)\s*$", "", heads[i][1])
    bad = ' style="color:#c5221f;font-weight:700"' if f > plan else ''
    rows.append(f'<tr><td>{name}</td><td>{n:,}</td><td>{c}</td><td{bad}>{f:.1f}′</td><td>{plan}′</td></tr>')
    print(f'{name[:40]:40s} {n:6,} {c:3d} {f:5.1f} {plan}')
tab = ('<table><thead><tr><th>Section</th><th>Script (words)</th><th>Screen cues</th><th>Floor</th><th>Planned</th></tr></thead><tbody>'
       + "".join(rows) + f'<tr><td><b>Seven sections</b></td><td colspan="2"></td><td><b>{fl:.0f}′</b></td><td><b>{plan_sum}′</b></td></tr></tbody></table>')
W = sum(words(s[heads[i][0]:heads[i + 1][0]]) for i in range(7)) + words(s[:heads[0][0]]) + words(s[heads[7][0]:])
C = len(re.findall(r'<p class="board">', s)); wmin = W / WPM; cmin = C * 15 / 60; tot = wmin + cmin
print('total words', W, 'cues', C, 'min', round(wmin), 'total', round(tot), 'plan', plan_sum)
p = D + '00.en.html'; e = open(p).read()
e, k = re.subn(r'<table><thead><tr><th>Section</th><th>Script \((?:Chinese chars|words)\)</th>.*?</table>', lambda m: tab, e, count=1, flags=re.S); assert k == 1
e, k = re.subn(r'<p><b>[\d,]+ (?:Chinese characters|English words) of script</b> \(at \d+(?: words)?/min = <b>\d+ min</b>\)\n\+ <b>\d+ screen cues</b> \(15 s each to scroll and locate = <b>[\d.]+ min</b>\)\n= <b>\d+ minutes</b>',
    lambda m: f'<p><b>{W:,} English words of script</b> (at {WPM} words/min = <b>{wmin:.0f} min</b>)\n+ <b>{C} screen cues</b> (15 s each to scroll and locate = <b>{cmin:.1f} min</b>)\n= <b>{tot:.0f} minutes</b>', e); assert k == 1
buf = (f'leaves <b>{plan_sum - tot:.0f} minutes</b> for figures and ad-libbing.' if tot <= plan_sum else
       f'<b>reading it all overruns by {tot - plan_sum:.0f} minutes</b>')
e, k = re.subn(r'(<p>⭐ <b>\d+′ version</b>: )(?:<b>reading it all overruns by \d+ minutes</b>|leaves <b>-?\d+ minutes</b> for figures and ad-libbing\.)',
    lambda m: m.group(1).replace(re.search(r'\d+′', m.group(1)).group(0), f'{plan_sum}′') + buf, e); assert k == 1
open(p, 'w').write(e)
