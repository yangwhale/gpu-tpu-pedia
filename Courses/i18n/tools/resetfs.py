"""把 NN.en.html 里每个 <text> 的内联字号恢复成放大前快照（R10b-prefit）里的值，文字保持当前。
python3 resetfs.py NN ..."""
import os
WORK = os.environ.get('I18N_WORK', '/tmp/t02en')
TOPIC = os.environ.get('I18N_TOPIC', 'topic-02')
import re, sys, pathlib
D = pathlib.Path(WORK+'/chunks'); P = pathlib.Path(WORK+'/rounds/prefit')
def fs(t):
    m = re.search(r'font-size:([\d.]+)px', t); return m.group(1) if m else None
for n in sys.argv[1:]:
    cur = (D/f'{n}.en.html').read_text(); pre = re.findall(r'<text\b[^>]*>', (P/f'{n}.en.html').read_text())
    tags = list(re.finditer(r'<text\b[^>]*>', cur)); assert len(tags) == len(pre), (n, len(tags), len(pre))
    out, last, ch = [], 0, 0
    for m, p in zip(tags, pre):
        out.append(cur[last:m.start()]); t = m.group(0); last = m.end(); want = fs(p)
        def setfs(t):
            st = re.search(r'style="([^"]*)"', t)
            body = re.sub(r'font-size:[^;"]*;?', '', st.group(1)).strip().rstrip(';') if st else ''
            if want: body = (body + ';' if body else '') + f'font-size:{want}px'
            if st: return t[:st.start()] + (f'style="{body}"' if body else '').join(['', '']) + t[st.end():] if body else re.sub(r'\s*style="[^"]*"', '', t)
            return t[:-1] + f' style="{body}">' if body else t
        nt = setfs(t)
        if nt != t: ch += 1
        out.append(nt)
    out.append(cur[last:]); (D/f'{n}.en.html').write_text(''.join(out)); print(n, 'reset', ch)
