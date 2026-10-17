"""列出整页里渲染后仍然偏小的 SVG 英文字（有效字号 = 计算字号 × svg 实际宽 / viewBox 宽）。
python3 smalltext.py [阈值px=9.5] → /tmp/t02en/small-NN.txt（按块分，带 figure id 与文字）"""
import sys, re, json, pathlib
from playwright.sync_api import sync_playwright
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 9.5
D = pathlib.Path('/tmp/t02en/chunks')
JS = """(TH) => { const out=[];
 document.querySelectorAll('figure').forEach(f=>{ const svg=f.querySelector('svg'); if(!svg) return;
  const vb=svg.viewBox.baseVal; const k=svg.getBoundingClientRect().width/(vb&&vb.width||1);
  svg.querySelectorAll('text').forEach(t=>{ const fs=parseFloat(getComputedStyle(t).fontSize)*k;
    const s=(t.textContent||'').trim(); if(s && fs<TH) out.push([f.id, +fs.toFixed(1), s.slice(0,90)]); }); });
 return out; }"""
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1280, 'height': 900})
    pg.add_init_script("try{localStorage.setItem('figonly:topic-02.en.html','0')}catch(e){}")
    pg.goto('file:///home/chrisya/gpu-tpu-pedia/Courses/WebPages/topic-02.en.html'); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1500)
    res = pg.evaluate(JS, TH); b.close()
where = {}
for f in sorted(D.glob('[0-9][0-9].en.html')):
    for fid in re.findall(r'<figure[^>]*id="([^"]+)"', f.read_text()): where[fid] = f.name[:2]
by = {}
for fid, fs, s in res: by.setdefault(where.get(fid, '??'), []).append(f'{fid}\t{fs}px\t{s}')
for k, v in sorted(by.items()):
    pathlib.Path(f'/tmp/t02en/small-{k}.txt').write_text('\n'.join(v) + '\n'); print(k, len(v))
print('total', len(res))
