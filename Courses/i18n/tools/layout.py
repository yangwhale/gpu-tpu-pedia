"""版面对齐：中英两页按 DOM 顺序一一对应（结构相同），逐元素比计算样式和盒子尺寸。

    python3 layout.py ZH_PAGE EN_PAGE [--width 1280] [--top 40]
报：① 计算样式不同的元素（字号/颜色/背景/边框/字重/行高/对齐 —— 理应全同，字体族除外）
    ② 高度差最大的块（文字变长把框撑高、标题折成两行、按钮换行 …）
    ③ 每节高度与整页高度对比
"""
import sys, json
from playwright.sync_api import sync_playwright
PROPS = ['font-size','font-weight','font-style','color','background-color','border-top-color','border-left-width',
         'line-height','text-align','display','padding-top','padding-left','margin-top','white-space','letter-spacing']
JS = """(props) => { const els=[...document.body.querySelectorAll('*')].filter(e=>!e.closest('svg')||e.tagName.toLowerCase()=='svg');
  return els.map(e=>{const cs=getComputedStyle(e), r=e.getBoundingClientRect();
    return {t:e.tagName.toLowerCase(), id:e.id||'', c:(e.className&&e.className.baseVal===undefined)?e.className:'',
            w:Math.round(r.width), h:Math.round(r.height), y:Math.round(r.top+scrollY),
            s:props.map(p=>cs.getPropertyValue(p)), txt:(e.innerText||'').trim().slice(0,50)}; }); }"""
def grab(f, w):
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': w, 'height': 900}); pg.add_init_script("try{for(const k of ['figonly:topic-02.en.html']) localStorage.setItem(k,'0')}catch(e){}")
        pg.goto('file://' + f); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1500)
        r = pg.evaluate(JS, PROPS); H = pg.evaluate('document.documentElement.scrollHeight'); b.close(); return r, H
def main():
    a = sys.argv; w = int(a[a.index('--width')+1]) if '--width' in a else 1280
    top = int(a[a.index('--top')+1]) if '--top' in a else 40
    (Z, HZ), (E, HE) = grab(a[1], w), grab(a[2], w)
    print(f'width {w}: page height zh {HZ} en {HE} ({HE/HZ:.2f}x); elements zh {len(Z)} en {len(E)}')
    if len(Z) != len(E): print('!! element count differs — structure mismatch'); 
    sty = []; hd = []
    for i, (z, e) in enumerate(zip(Z, E)):
        if z['t'] != e['t']: print('!! tag mismatch at', i, z['t'], e['t']); break
        diff = [(PROPS[k], z['s'][k], e['s'][k]) for k in range(len(PROPS)) if z['s'][k] != e['s'][k]]
        if diff: sty.append((i, z, e, diff))
        if z['h'] > 20 and e['t'] in ('p','figcaption','h2','h3','h4','li','summary','div','table','nav','span','a','button','figure','details','blockquote') :
            d = e['h'] - z['h']
            if abs(d) >= 12: hd.append((d, i, z, e))
    print(f'\n① style diffs: {len(sty)}')
    for i, z, e, diff in sty[:top]: print(f"  #{i} <{z['t']} .{z['c'][:20]}> {diff[:3]}  «{e['txt'][:40]}»")
    hd.sort(key=lambda x: -abs(x[0]))
    print(f'\n② biggest height deltas (en - zh), {len(hd)} blocks with |Δ|≥12px:')
    for d, i, z, e in hd[:top]: print(f"  {d:+5}px  #{i} <{z['t']} .{z['c'][:18]}> zh {z['h']} en {e['h']}  «{e['txt'][:60]}»")
    print('\n③ sections:')
    for z, e in zip(Z, E):
        if z['t'] == 'section': print(f"  {z['id'] or '-':6} zh {z['h']:6} en {e['h']:6} {e['h']/max(z['h'],1):.2f}x")
main()
