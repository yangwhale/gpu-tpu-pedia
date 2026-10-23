"""Recount the English floor table in the Topic 6 lecture (chunk 00) from chunks 01-10 (sections c0..c9).
Same rule as the zh build (topic06_common.py): spoken lines <p class="say"> / 140 English words a minute
(~ zh 200 chars/min x 0.70 words per char) + 15 s per <p class="board"> + 0.5 min per 💬 Interaction
(⏭-marked ones are skippable). Rewrites: the floor table in 00, the "Planned N′ · with interaction M′" span in every
section heading, and the prose total / reading-rate sentence in 00. Idempotent. Run: python3 floor_en_topic06.py [zh]"""
import re, sys, html as H
D = '/tmp/t06lec/chunks/'
WPM = 140
ZH = len(sys.argv) > 1 and sys.argv[1] == 'zh'
lang = 'zh' if ZH else 'en'
def say(t): return " ".join(H.unescape(re.sub(r"<[^>]+>", " ", x)) for x in re.findall(r'<p class="say">(.*?)</p>', t, re.S))
def words(t): return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,×%-]*", say(t)))
def hanzi(t): return len(re.findall(r"[一-鿿]", say(t)))
rows, tot_f, tot_ia, tot_sk, plan_sum = [], 0.0, 0, 0, 0
spans = {}
for n in range(1, 11):
    p = f'{D}{n:02d}.{lang}.html'; t = open(p).read()
    head = re.search(r'<h3 class="sec" id="c\d+">(.*?)<span class="tm6">', t, re.S).group(1)
    plan = int(re.search(r"[(（](\d+)′[)）]", head).group(1)); plan_sum += plan
    name = re.sub(r"\s*[(（]\d+′[)）]\s*$", "", re.sub(r"<[^>]+>", "", head)).replace('　', ' · ', 1) if ZH else re.sub(r"\s*\(\d+′\)\s*$", "", re.sub(r"<[^>]+>", "", head))
    cnt = hanzi(t) if ZH else words(t)
    cc = len(re.findall(r'<p class="board">', t))
    f = cnt / (200 if ZH else WPM) + cc * 0.25
    ias = re.findall(r'<p class="ia6">💬 <b>[^<]*</b>.*?</p>', t, re.S)
    ia, sk = len(ias), sum('⏭' in x for x in ias)
    fa, fs = f + ia * 0.5, f + (ia - sk) * 0.5
    tot_f += f; tot_ia += ia; tot_sk += sk
    spans[n] = (plan, fa, fs)
    rows.append(f'<tr><td>{name}</td><td>{cnt:,}</td><td>{cc}</td><td>{f:.1f}′</td><td>{plan}′</td><td>{ia} / {sk}</td><td>{fa:.1f}′</td><td>{fs:.1f}′</td></tr>')
    print(f'{name[:55]:55s} {cnt:6,} {cc:3d} floor {f:5.1f}  +ia {fa:5.1f}  plan {plan}{"  (over plan)" if fa > plan else ""}')
fa_tot = tot_f + tot_ia * 0.5; fs_tot = tot_f + (tot_ia - tot_sk) * 0.5
print(f'floor {tot_f:.0f}′  with interaction {fa_tot:.0f}′  after skips {fs_tot:.0f}′  plan {plan_sum}′')
if ZH: sys.exit()
tab = ('<table><thead><tr><th>Section</th><th>Script (words)</th><th>Screens</th><th>Floor</th><th>Planned</th><th>Interactions / skippable</th><th>With interaction</th><th>After skips</th></tr></thead><tbody>'
       + ''.join(rows) + f'<tr><td><b>Total</b></td><td colspan="2"></td><td><b>{tot_f:.0f}′</b></td><td><b>{plan_sum}′</b></td><td><b>{tot_ia} / {tot_sk}</b></td><td><b>{fa_tot:.0f}′</b></td><td><b>{fs_tot:.0f}′</b></td></tr></tbody></table>')
p0 = D + '00.en.html'; s = open(p0).read()
tabs = list(re.finditer(r'<table><thead><tr><th>Section</th><th>Script \((?:zh chars|words)\)</th>.*?</table>', s, re.S))
assert len(tabs) == 1, len(tabs)
s = s[:tabs[0].start()] + tab + s[tabs[0].end():]
s = re.sub(r'comes to about \d+ minutes', f'comes to about {fa_tot:.0f} minutes', s)
s = s.replace('reading at 200 Chinese characters a minute', f'reading at {WPM} words a minute')
s = s.replace('script at 200 Chinese characters/min', f'script at {WPM} words/min')
open(p0, 'w').write(s)
for n, (plan, fa, fs) in spans.items():
    p = f'{D}{n:02d}.en.html'; t = open(p).read()
    new = f'<span class="tm6">Planned {plan}′ · with interaction {fa:.1f}′' + (f' · after skips {fs:.1f}′' if fs < fa else '') + '</span>'
    t, k = re.subn(r'<span class="tm6">Planned \d+′ · with interaction [\d.]+′(?: · after skips [\d.]+′)?</span>', new, t)
    assert k == 1, (n, k); open(p, 'w').write(t)
print('rewrote 00 table, prose, and', len(spans), 'heading spans')
