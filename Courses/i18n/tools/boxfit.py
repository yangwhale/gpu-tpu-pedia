"""SVG 文字是否越出它所在的框（rect）。中英各跑一遍，只报英文新增的。

    python3 boxfit.py ZH_PAGE EN_PAGE
判据：取包含文字锚点的最小 rect，文字 bbox 超出该 rect 边界 >1px 即记一处。
"""
import sys
from playwright.sync_api import sync_playwright
JS = """() => { const out=[];
 document.querySelectorAll('svg').forEach((s,si)=>{
  const rs=[...s.querySelectorAll('rect')].map(r=>r.getBoundingClientRect()).filter(r=>r.width>20&&r.height>12);
  [...s.querySelectorAll('text')].forEach((t,ti)=>{
    const b=t.getBoundingClientRect(); if(b.width<1) return;
    const cx=b.left+2, cy=b.top+b.height/2;
    const box=rs.filter(r=>cx>=r.left&&cx<=r.right&&cy>=r.top&&cy<=r.bottom).sort((a,c)=>a.width*a.height-c.width*c.height)[0];
    if(!box) return;
    const over=Math.max(box.left-b.left, b.right-box.right, box.top-b.top, b.bottom-box.bottom);
    if(over>1) out.push([si,ti,Math.round(over),(t.textContent||'').trim().slice(0,50)]);
  });});
 return out; }"""
def run(f):
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1280, 'height': 900}); pg.add_init_script("try{for(const k of ['figonly:topic-02.en.html']) localStorage.setItem(k,'0')}catch(e){}")
        pg.goto('file://' + f); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1500)
        pg.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)"); pg.wait_for_timeout(300)
        r = pg.evaluate(JS); b.close(); return r
zh, en = run(sys.argv[1]), run(sys.argv[2])
zk = {(a, b) for a, b, *_ in zh}
new = [x for x in en if (x[0], x[1]) not in zk]
print(f'zh {len(zh)}  en {len(en)}  new-in-en {len(new)}')
for x in new: print('  svg#%d text#%d  +%dpx  %s' % tuple(x))
