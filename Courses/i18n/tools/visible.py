"""默认可见层（不展开折叠块、不算 SVG 里的字）的字量对比 —— 讲课时屏幕上真正看得到的那一层。

    python3 visible.py NN            # 在 /tmp/t02en/chunks 下比 NN.zh.html / NN.en.html
输出：整块可见字符 zh / en / 比值，以及比值最高的 15 个可见段落。
经验目标：en/zh 字符比 ≤ 1.55（好的中译英一般 1.5–1.7；超过 1.8 就是啰嗦）。
"""
import os
WORK = os.environ.get('I18N_WORK', '/tmp/t02en')
TOPIC = os.environ.get('I18N_TOPIC', 'topic-02')
import sys, re, pathlib
from playwright.sync_api import sync_playwright
D = pathlib.Path(WORK+'/chunks')
JS = """() => {
  const out=[];
  const blocks=document.querySelectorAll('p, li, figcaption, td, th, h2, h3, h4, summary, .lead, .note, .callout, div.k');
  for (const b of blocks) {
    if (b.closest('svg')) continue;
    if (b.closest('details:not([open])') && !b.closest('summary')) continue;
    if (b.querySelector('p, li, figcaption, td, h3')) continue;
    const t=(b.innerText||'').trim(); if (t) out.push(t);
  }
  return out; }"""
def grab(pg, f, head):
    pg.set_content(head + f.read_text() + '</body></html>', wait_until='domcontentloaded')
    return pg.evaluate(JS)
def main(n):
    skel = (D/'skeleton.html').read_text(); head = skel[:skel.index('@@CHUNK')]
    head += ''.join(re.findall(r'<style\b.*?</style>', (D/'00.zh.html').read_text(), flags=re.S))
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page()
        zh = grab(pg, D/f'{n}.zh.html', head); en = grab(pg, D/f'{n}.en.html', head); b.close()
    Z = sum(map(len, zh)); E = sum(map(len, en))
    print(f'chunk {n}: visible zh {Z} chars, en {E} chars, ratio {E/max(Z,1):.2f}  ({len(zh)} vs {len(en)} blocks)')
    if len(zh) == len(en):
        rows = sorted(((len(e)/max(len(z),1), len(e), z, e) for z, e in zip(zh, en) if len(z) > 15), reverse=True)[:15]
        for r, l, z, e in rows: print(f'  {r:4.2f}  {l:4}  EN: {e[:110]}')
if __name__ == '__main__':
    for n in sys.argv[1:]: main(n)
