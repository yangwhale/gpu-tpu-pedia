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
# 讲稿折叠开关的 localStorage 键各专题不同：专题三～六 "figonly:<文件名>"，专题七 "t7fo:<文件名>"（默认折叠）。
# 只设 figonly: 时专题七两页都按折叠量（2026-10-29 查出）。两个都设成 '0'，并在加载后核对 body 没有 figonly。
UNFOLD = "try{var n=location.pathname.split('/').pop();['figonly:','t7fo:'].forEach(function(k){localStorage.setItem(k+n,'0')})}catch(e){}"
def same(z, e): return z['t'] == e['t'] or {z['t'], e['t']} == {'a', 'span'}   # 英文去掉中文页链接：a→span 视为同构
def align(Z, E):
    """按标签序列对齐（difflib），返回 (配对列表, 只在中文的, 只在英文的)。结构不同（如英文删了指向无英文版页面的按钮）时
    不再从第一处不一致起整体错位。"""
    import difflib
    key = lambda x: 'a' if x['t'] in ('a', 'span') else x['t']
    sm = difflib.SequenceMatcher(None, [key(z) for z in Z], [key(e) for e in E], autojunk=False)
    pairs, oz, oe = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal': pairs += list(zip(range(i1, i2), range(j1, j2)))
        else: oz += list(range(i1, i2)); oe += list(range(j1, j2))
    return pairs, oz, oe
def grab(f, w):
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': w, 'height': 900}); pg.add_init_script(UNFOLD)
        pg.goto('file://' + f); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1500)
        if pg.evaluate("document.body.classList.contains('figonly')"): sys.exit('!! %s still folded after unfold init script — unknown fold key' % f)
        r = pg.evaluate(JS, PROPS); H = pg.evaluate('document.documentElement.scrollHeight'); b.close(); return r, H
def main():
    a = sys.argv; w = int(a[a.index('--width')+1]) if '--width' in a else 1280
    top = int(a[a.index('--top')+1]) if '--top' in a else 40
    (Z, HZ), (E, HE) = grab(a[1], w), grab(a[2], w)
    print(f'width {w}: page height zh {HZ} en {HE} ({HE/HZ:.2f}x); elements zh {len(Z)} en {len(E)}')
    pairs, oz, oe = align(Z, E)
    if len(Z) != len(E) or oz or oe:
        print(f'!! element count differs — {len(oz)} only in zh, {len(oe)} only in en (aligned by tag sequence; tables below use matched pairs):')
        for k in oz[:top]: print(f"    zh-only #{k} <{Z[k]['t']} .{Z[k]['c'][:18]}> «{Z[k]['txt'][:40]}»")
        for k in oe[:top]: print(f"    en-only #{k} <{E[k]['t']} .{E[k]['c'][:18]}> «{E[k]['txt'][:40]}»")
    sty = []; hd = []
    for i, j in pairs:
        z, e = Z[i], E[j]
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
    SZ = [z for z in Z if z['t'] == 'section']; SE = {e['id']: e for e in E if e['t'] == 'section' and e['id']}
    for k, z in enumerate(SZ):   # 按 id 配对；没有 id 的按出现顺序
        e = SE.get(z['id']) if z['id'] else None
        if e is None: es = [x for x in E if x['t'] == 'section']; e = es[k] if k < len(es) else None
        if e is None: print(f"  {z['id'] or '-':6} zh {z['h']:6} en   (missing)"); continue
        print(f"  {z['id'] or '-':6} zh {z['h']:6} en {e['h']:6} {e['h']/max(z['h'],1):.2f}x")
main()
