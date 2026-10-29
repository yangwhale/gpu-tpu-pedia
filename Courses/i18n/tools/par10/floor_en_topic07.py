"""Recount the floor table of the Topic 7 lecture (chunk 00) from chunks 00-06 (sections c0..c6; chunk 00 holds the
header, the timing tables AND Section 0). Copied from floor_en_topic06.py; Topic 7 adds a MEDIA term.

Floor, same rule as the zh build (topic06_common.lecture_finish, read-only import in topic07-build-lecture.py):
  spoken lines <p class="say"> at 140 English words a minute (zh: 200 CJK chars/min; ~0.70 words per char)
  + 15 s per <p class="board"> (the "board sub" main-screen line is not counted) + 0.5 min per 💬 interaction (⏭ = skippable).
Media (Topic 7 only; the zh TIME_NOTE adds it by hand, "约 4.8 分钟"): the played time of every recording/animation stop
the lecture cites (STOPS below, taken from the zh lecture's 🖥 cues). Rule, read off the step player (stepbar):
  a stop's played time runs from the frame where the player sits at that stop to the next stop, or to the end of the clip.
  <video data-first="1">: stop 1 = frame 0, stop k = data-pauses[k-2].  Without data-first: stop k = data-pauses[k-1]
  (frame 0 is an unnumbered start).  Pauses come from the course chunks /tmp/t07en/chunks/NN.<lang>.html, durations from
  ffprobe on the mp4 the src points at (so the English count follows the .en.mp4 once M4 swaps the srcs).
Plan: the zh plan minutes are round(with interaction + media) per section ("各节分钟数＝含互动＋该节停点，四舍五入").

  python3 floor_en_topic07.py zh   self-test: recount the zh chunks, compare with the zh floor table and plan minutes
                                    in 00.zh.html; exit 1 on any mismatch. Writes nothing.
  python3 floor_en_topic07.py      English: rewrite the floor table in 00.en.html, the "Planned N′ · with interaction M′"
                                    span in every section heading, the timing-note numbers in 00 and the total in 06's
                                    🚧 list. Plan minutes are NOT changed (prints where round(with interaction + media)
                                    no longer equals the plan). Idempotent. The English sentences it rewrites are fixed
                                    by /tmp/t07lec/PROMPT.md ("Timing note in chunk 00"); it asserts each one matched."""
import re, sys, html as H, os, subprocess, math
D = '/tmp/t07lec/chunks/'
COURSE = '/tmp/t07en/chunks/'
WEB = '/home/chrisya/gpu-tpu-pedia/Courses/WebPages/'
WPM = 140
ZH = len(sys.argv) > 1 and sys.argv[1] == 'zh'
lang = 'zh' if ZH else 'en'
NSEC = 7
# Cited stops per section (zh lecture 🖥 cues and TIME_NOTE). xrec-basics (optional, "not in the floor") and
# xrec-opprof (only if XProf can't be opened live) are deliberately absent, as in the zh count.
STOPS = {
    2: [('xrec-open', range(3, 8)), ('anim-agdone', range(1, 6))],              # 02: 从停 3 走到停 7; 停 1–5
    3: [('xrec-roof', (2, 3)), ('xrec-mem', range(1, 12))],                     # 03: 停 2、停 3; 停 1–3, 4–5, 6–10, 11
    4: [('xrec-graph', (1, 2, 9, 10, 11)), ('anim-experts', range(1, 8))],      # 04: 停 1、2 (3–8 clicked past), 9–11; 停 1–7
    5: [('xrec-mem', (12,))],                                                   # 05: 第三节跳过的停 12
}


def rnd(x):  # round half up, as a reader would (Python's round() is banker's)
    return int(math.floor(x + 0.5))


def say(t): return " ".join(H.unescape(re.sub(r"<[^>]+>", " ", x)) for x in re.findall(r'<p class="say">(.*?)</p>', t, re.S))
def words(t): return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,×%-]*", say(t)))
def hanzi(t): return len(re.findall(r"[一-鿿]", say(t)))


_course = None
def video(vid):
    global _course
    if _course is None:
        _course = "".join(open(COURSE + f).read() for f in sorted(os.listdir(COURSE)) if re.fullmatch(r'\d\d\.%s\.html' % lang, f))
    m = re.search(r'id="%s"><video ([^>]*)>' % re.escape(vid), _course)
    assert m, 'video %s not found in %s*.%s.html' % (vid, COURSE, lang)
    a = m.group(1)
    P = [float(x) for x in re.search(r'data-pauses="([^"]+)"', a).group(1).split(',')]
    first = 'data-first="1"' in a
    src = re.search(r'src="([^"?]+)', a).group(1)
    path = WEB + src
    assert os.path.exists(path), 'media file missing: %s' % path
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                               capture_output=True, text=True, check=True).stdout)
    return P, first, dur, src


def stop_play(vid, stops):
    P, first, dur, src = video(vid)
    pos = ([0.0] + P) if first else P            # pos[k-1] = where stop k sits
    tot = 0.0
    for k in stops:
        assert 1 <= k <= len(pos), (vid, k, len(pos))
        tot += (pos[k] if k < len(pos) else dur) - pos[k - 1]
    return tot, src


media, media_txt = {}, {}
for n in range(NSEC):
    sec, parts = 0.0, []
    for vid, stops in STOPS.get(n, []):
        s, src = stop_play(vid, stops)
        sec += s
        parts.append('%s %s %.1fs' % (src.split('/')[-1].split('?')[0], ','.join(map(str, stops)), s))
    media[n] = sec / 60
    media_txt[n] = '; '.join(parts)

rows, tot_f, tot_ia, tot_sk, plan_sum = [], 0.0, 0, 0, 0
spans, info = {}, {}
for n in range(NSEC):
    p = f'{D}{n:02d}.{lang}.html'; t = open(p).read()
    if n == 0:
        t = t[t.index('<h3 class="sec" id="c0">'):]          # chunk 00: count Section 0 only, not the timing tables
    head = re.search(r'<h3 class="sec" id="c%d">(.*?)<span class="tm6">' % n, t, re.S).group(1)
    plan = int(re.search(r"[(（](\d+)′[)）]", head).group(1)); plan_sum += plan
    name = re.sub(r"<[^>]+>", "", head).split("（")[0] if ZH else re.sub(r"\s*\(\d+′\)\s*$", "", re.sub(r"<[^>]+>", "", head))
    cnt = hanzi(t) if ZH else words(t)
    cc = len(re.findall(r'<p class="board">', t))
    f = cnt / (200 if ZH else WPM) + cc * 0.25
    ias = re.findall(r'<p class="ia6">💬 <b>[^<]*</b>.*?</p>', t, re.S)
    ia, sk = len(ias), sum('⏭' in x for x in ias)
    fa, fs = f + ia * 0.5, f + (ia - sk) * 0.5
    tot_f += f; tot_ia += ia; tot_sk += sk
    spans[n] = (plan, fa, fs); info[n] = (f, fa, plan)
    rows.append(f'<tr><td>{name}</td><td>{cnt:,}</td><td>{cc}</td><td>{f:.1f}′</td><td>{plan}′</td><td>{ia} / {sk}</td><td>{fa:.1f}′</td><td>{fs:.1f}′</td></tr>')
    mt = media[n]
    print(f'{name[:48]:48s} {cnt:6,} {cc:3d} floor {f:5.1f}  +ia {fa:5.1f}  +media {mt:4.2f} = {fa + mt:5.2f} → {rnd(fa + mt):2d}  plan {plan:2d}'
          + ('' if rnd(fa + mt) == plan else '  ← round(with interaction + media) ≠ plan'))
    if media_txt[n]:
        print(f'{"":48s}   media: {media_txt[n]}')
fa_tot = tot_f + tot_ia * 0.5; fs_tot = tot_f + (tot_ia - tot_sk) * 0.5; md_tot = sum(media.values())
print(f'floor {tot_f:.0f}′  with interaction {fa_tot:.0f}′  after skips {fs_tot:.0f}′  media {md_tot:.2f}′  '
      f'with interaction + media {fa_tot + md_tot:.1f}′  plan {plan_sum}′')

if ZH:
    # ── self-test against the zh build's own numbers ─────────────────────────────────────────
    s0 = open(D + '00.zh.html').read()
    bad = []
    tab = re.search(r'<table><thead><tr><th>节</th><th>台词汉字</th>.*?</table>', s0, re.S).group(0)
    mine = ('<table><thead><tr><th>节</th><th>台词汉字</th><th>切图</th><th>地板</th><th>计划给的</th><th>互动 / 可跳</th><th>含互动</th><th>跳后</th></tr></thead><tbody>'
            + ''.join(rows) + '<tr><td><b>合计</b></td><td colspan="2"></td><td><b>%.0f′</b></td><td><b>%d′</b></td><td><b>%d / %d</b></td><td><b>%.0f′</b></td><td><b>%.0f′</b></td></tr></tbody></table>'
            % (tot_f, plan_sum, tot_ia, tot_sk, fa_tot, fs_tot))
    if tab != mine:
        zr = re.findall(r'<tr>.*?</tr>', tab); mr = re.findall(r'<tr>.*?</tr>', mine)
        bad += ['floor table row differs:\n   zh  %s\n   me  %s' % (a, b) for a, b in zip(zr, mr) if a != b]
    else:
        print('✓ floor table: all %d rows + total identical to 00.zh.html' % NSEC)
    # plan table and heading minutes
    ptab = re.findall(r'<tr><td>[零一二三四五六]　[^<]*</td><td>(\d+)</td>', s0)
    if [int(x) for x in ptab] != [spans[n][0] for n in range(NSEC)]:
        bad.append('plan table %s ≠ heading minutes %s' % (ptab, [spans[n][0] for n in range(NSEC)]))
    off = [n for n in range(NSEC) if rnd(spans[n][1] + media[n]) != spans[n][0]]
    if off:
        bad.append('plan ≠ round(with interaction + media) in sections %s' % off)
    else:
        print('✓ plan minutes: all %d = round(with interaction + media), sum %d′' % (NSEC, plan_sum))
    # timing-note prose (hand-written in topic07_lec.TIME_NOTE): approximate figures, reported not asserted
    note = re.search(r'含互动约 (\d+) 分钟.*?约 ([\d.]+) 分钟：(.*?)），\s*合计约 <b>(\d+) 分钟', s0, re.S)
    z_ia, z_md, z_parts, z_tot = int(note.group(1)), float(note.group(2)), note.group(3), int(note.group(4))
    z_sec = dict((('零一二三四五六'.index(a)), b) for a, b in re.findall(r'第([零一二三四五六])节.*?(约 [\d.]+|不到 0\.1)', z_parts))
    print('timing note (zh prose → recount):  with interaction 约 %d → %.1f;  media 约 %.1f → %.2f;  total 约 %d → plan sum %d'
          % (z_ia, fa_tot, z_md, md_tot, z_tot, plan_sum))
    for n, txt in sorted(z_sec.items()):
        v = media[n]; zv = 0.05 if txt.startswith('不到') else float(txt.split()[1])
        ok = (v < 0.1) if txt.startswith('不到') else abs(v - zv) <= 0.051
        print('   Section %d  zh %-8s recount %.2f%s' % (n, txt, v, '' if ok else '   ← differs (prose only; plan minutes unaffected)'))
    if z_ia != rnd(fa_tot): bad.append('prose 含互动约 %d ≠ %d' % (z_ia, rnd(fa_tot)))
    if z_tot != plan_sum: bad.append('prose 合计约 %d ≠ plan sum %d' % (z_tot, plan_sum))
    if bad:
        print('SELF-TEST FAILED'); [print(' ✗', b) for b in bad]; sys.exit(1)
    print('SELF-TEST OK')
    sys.exit()

# ── English rewrite ───────────────────────────────────────────────────────────────────────────
def sub1(pat, rep, s, what, flags=0):
    s2, k = re.subn(pat, rep, s, flags=flags)
    assert k == 1, 'expected exactly 1 match for %s, got %d (sentence must follow PROMPT.md "Timing note")' % (what, k)
    return s2

tab = ('<table><thead><tr><th>Section</th><th>Script (words)</th><th>Screens</th><th>Floor</th><th>Planned</th><th>Interactions / skippable</th><th>With interaction</th><th>After skips</th></tr></thead><tbody>'
       + ''.join(rows) + f'<tr><td><b>Total</b></td><td colspan="2"></td><td><b>{tot_f:.0f}′</b></td><td><b>{plan_sum}′</b></td><td><b>{tot_ia} / {tot_sk}</b></td><td><b>{fa_tot:.0f}′</b></td><td><b>{fs_tot:.0f}′</b></td></tr></tbody></table>')
p0 = D + '00.en.html'; s = open(p0).read()
tabs = list(re.finditer(r'<table><thead><tr><th>Section</th><th>Script \((?:zh chars|words)\)</th>.*?</table>', s, re.S))
assert len(tabs) == 1, len(tabs)
s = s[:tabs[0].start()] + tab + s[tabs[0].end():]
fmt = lambda v: 'under 0.1' if v < 0.1 else 'about %.1f' % v
s = s.replace('script at 200 Chinese characters a minute', f'script at {WPM} words a minute')
s = s.replace('script at 200 Chinese characters/min', f'script at {WPM} words/min')
s = sub1(r'about \d+ minutes with interaction', f'about {rnd(fa_tot)} minutes with interaction', s, 'with-interaction total')
s = sub1(r'stops these notes cite \(about [\d.]+ minutes', f'stops these notes cite (about {md_tot:.1f} minutes', s, 'media total')
for n in STOPS:
    s = sub1(r'(Section %d, [^;()]*?, )(?:about [\d.]+|under 0\.1)(?=[;)])' % n, r'\g<1>' + fmt(media[n]), s, 'media Section %d' % n)
s = sub1(r'for a total of about <b>\d+ minutes</b>', f'for a total of about <b>{rnd(fa_tot + md_tot)} minutes</b>', s, 'grand total')
# plan table: "floor X′" mentions in the Cuttable? column follow the recount (zh: 已贴地板（8.4′）, 地板 6.3′)
ptab = re.search(r'<th>Segment</th>.*?</tbody>', s, re.S)
pt = ptab.group(0); prow = re.findall(r'<tr><td>(\d) [^<]*</td>.*?</tr>', pt)
def fix_row(m):
    r = m.group(0); n = int(re.match(r'<tr><td>(\d)', r).group(1))
    return re.sub(r'(floor \(?)[\d.]+′', lambda x: x.group(1) + '%.1f′' % info[n][0], r)
pt2 = re.sub(r'<tr><td>\d [^<]*</td>.*?</tr>', fix_row, pt)
s = s.replace(pt, pt2)
open(p0, 'w').write(s)
p6 = D + '06.en.html'; t6 = open(p6).read()
t6 = sub1(r'about \d+ minutes \(see the time budget above\)', f'about {rnd(fa_tot + md_tot)} minutes (see the time budget above)', t6, '06 🚧 total')
open(p6, 'w').write(t6)
for n, (plan, fa, fs) in spans.items():
    p = f'{D}{n:02d}.en.html'; t = open(p).read()
    new = f'<span class="tm6">Planned {plan}′ · with interaction {fa:.1f}′' + (f' · after skips {fs:.1f}′' if fs < fa else '') + '</span>'
    t, k = re.subn(r'<span class="tm6">Planned \d+′ · with interaction [\d.]+′(?: · after skips [\d.]+′)?</span>', new, t)
    assert k == 1, (n, k); open(p, 'w').write(t)
print('rewrote 00 table, timing note, plan-table floor mentions, 06 total, and', len(spans), 'heading spans')
off = [n for n in range(NSEC) if rnd(spans[n][1] + media[n]) != spans[n][0]]
if off:
    print('NOTE for the coordinator: round(with interaction + media) ≠ plan in sections %s; plan minutes left as in the Chinese' % off)
