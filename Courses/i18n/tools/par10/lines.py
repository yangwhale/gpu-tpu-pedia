"""逐块行数对比：python3 lines.py ZH EN WIDTH [--folded]  → 列出英文比中文多行/高 ≥6px 的块（带最近标题）。
a/span 互换（英文去掉中文页链接）视为同构；其他标签不一致时停下并报位置。"""
import sys, json
from playwright.sync_api import sync_playwright
ZH, EN, W = sys.argv[1], sys.argv[2], int(sys.argv[3]); FOLD = '--folded' in sys.argv
JS = """() => { const els=[...document.body.querySelectorAll('*')].filter(e=>!e.closest('svg')||e.tagName.toLowerCase()=='svg');
  return els.map(e=>{const r=e.getBoundingClientRect(); let lines=0,lastw=0;
   if(!e.closest('svg')&&r.height>0){const rg=document.createRange(); rg.selectNodeContents(e); const rr=[...rg.getClientRects()].filter(q=>q.width>0);
     lines=new Set(rr.map(q=>Math.round(q.top))).size; if(rr.length){const my=Math.max(...rr.map(q=>Math.round(q.top))); const L=rr.filter(q=>Math.round(q.top)==my); lastw=Math.round(Math.max(...L.map(q=>q.right))-Math.min(...L.map(q=>q.left)));}}
   return {t:e.tagName.toLowerCase(), id:e.id||'', h:Math.round(r.height), y:Math.round(r.top+scrollY), lines, lastw, txt:(e.innerText||'').trim().slice(0,80)}; }); }"""
def grab(f, zh):
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': W, 'height': 900})
        pg.add_init_script("try{localStorage.setItem('figonly:'+location.pathname.split('/').pop(),'%s')}catch(e){}" % ('1' if FOLD else '0'))
        pg.goto('file://' + f); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1500)
        if not FOLD: pg.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)")
        else:
            on = pg.evaluate("document.body.classList.contains('figonly')")
            if not on: pg.keyboard.press('t'); pg.wait_for_timeout(500)
        pg.wait_for_timeout(800); r = pg.evaluate(JS); b.close(); return r
Z = grab(ZH, True); E = grab(EN, False)
TX = ('p','figcaption','h1','h2','h3','h4','li','summary','div','td','th','a','button','caption','dt','dd','pre','span')
head = ''; n = 0; i = j = 0; skips = 0
def same(z, e): return z['t'] == e['t'] or {z['t'], e['t']} == {'a', 'span'}
while i < len(Z) and j < len(E):
    z, e = Z[i], E[j]
    if not same(z, e):
        best = None
        for di in range(0, 8):
            for dj in range(0, 8):
                if (di or dj) and i+di+3 < len(Z) and j+dj+3 < len(E) and all(same(Z[i+di+k], E[j+dj+k]) for k in range(4)):
                    if best is None or di+dj < sum(best): best = (di, dj)
        if best is None: print(f'!! cannot resync at zh#{i} en#{j}'); break
        i += best[0]; j += best[1]; skips += 1; continue
    if e['t'] in ('h2', 'h3', 'h4'): head = e['txt'][:50]
    if e['t'] in TX and e['h'] > 0 and (e['lines'] > z['lines'] or e['h'] - z['h'] >= 6):
        if not (e['t'] in ('div', 'span', 'a') and e['lines'] == z['lines'] and e['h'] - z['h'] < 12):
            n += 1
            print(f"en#{j} [{head}] <{e['t']}{'#'+e['id'] if e['id'] else ''}> lines {z['lines']}→{e['lines']} h {z['h']}→{e['h']} lastw {e['lastw']} «{e['txt'][:70]}»")
    i += 1; j += 1
print(f'(resynced {skips}×)')
print(f'== {n} blocks where English is taller ({len(Z)} vs {len(E)} elements, width {W}{", folded" if FOLD else ""})')
