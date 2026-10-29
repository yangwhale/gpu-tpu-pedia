"""对账：中文块 vs 英文块。

    python3 check.py ZH.html EN.html            # 结构 + 残留中文 + 数字
    python3 check.py ZH.html EN.html --render   # 再加：SVG 文字渲染宽度对比（Chrome）

结构对账：标签序列（含 id/class）必须一致；SVG 里除 <text>/<tspan> 的 font-size 外，
几何属性必须一字不差。数字对账只报差异，换算类（6710亿→671B）由人判断。
"""
import os
WORK = os.environ.get('I18N_WORK', '/tmp/t02en')
TOPIC = os.environ.get('I18N_TOPIC', 'topic-02')
import re, sys, collections, pathlib
from bs4 import BeautifulSoup, Comment

CJK = re.compile(r'[㐀-鿿　-〿＀-￯]')
GEOM_FREE = {'font-size'}
TEXT_ATTRS = {'title', 'alt', 'aria-label', 'placeholder'}
# 修图里文字压框/撞线时允许改的 SVG 几何属性（如 I18N_GEOM_FREE=x,y,height），只在明确授权的轮次打开
GEOM_EXTRA = {x for x in os.environ.get('I18N_GEOM_FREE', '').split(',') if x}

def soup(p):
    return BeautifulSoup(pathlib.Path(p).read_text(), 'html.parser')

# 英文媒体 media/X.en.mp4?v=… / img/X.en.png?v=… 与中文 media/X.mp4?v=… / img/X.png 视为同构
# （⭐ 只认 media/ 与 img/ 下同名换 .en，?v= 只认十六进制；负例见 test_check.py）
SRC_EN = re.compile(r'^((?:media|img)/[^?"]+?)(?:\.en)?(\.(?:mp4|webm|png|jpg))(\?v=[0-9a-f]+)?$')
def norm_src(v):
    return SRC_EN.sub(r'\1\2', v)

def tagseq(sp):
    out = []
    for t in sp.find_all(True):
        a = {k: (' '.join(v) if isinstance(v, list) else v) for k, v in t.attrs.items()
             if k not in TEXT_ATTRS}
        if GEOM_EXTRA and t.find_parent('svg') is not None:
            for k in GEOM_EXTRA: a.pop(k, None)
        if t.name in ('text', 'tspan'):
            a.pop('font-size', None)
            if 'style' in a:
                a['style'] = re.sub(r'font-size:[^;]*;?', '', a['style']).strip().rstrip(';')
                if not a['style']: a.pop('style')
            a.pop('textLength', None); a.pop('lengthAdjust', None)
        for k in ('src', 'poster', 'href'):   # href：英文页「Full video」链接指向 .en 录像／末帧（专题七发布）
            if k in a: a[k] = norm_src(a[k])
        out.append((t.name, tuple(sorted(a.items()))))
    return out

def nums(sp):
    txt = ' '.join(sp.find_all(string=True))
    return collections.Counter(re.findall(r'\d+(?:[.,]\d+)*', txt.replace(',', '')))

def main(zh, en, render=False):
    a, b = soup(zh), soup(en)
    bad = 0
    # 1) 残留中文
    left = [s.strip() for s in b.find_all(string=True) if CJK.search(s) and not isinstance(s, Comment)]
    for t in b.find_all(True):
        for k in TEXT_ATTRS:
            if k in t.attrs and CJK.search(str(t[k])): left.append(f'@{k}: {t[k]}')
    if left:
        bad += 1
        print(f'✗ 残留中文 {len(left)} 处：'); [print('   ', x[:100]) for x in left[:30]]
    # 2) 结构
    sa, sb = tagseq(a), tagseq(b)
    if sa != sb:
        bad += 1
        import difflib
        print(f'✗ 结构不一致：zh {len(sa)} 个标签 / en {len(sb)} 个')
        for i, (x, y) in enumerate(zip(sa, sb)):
            if x != y:
                print(f'   首个差异 #{i}:\n     zh {x[0]} {dict(x[1])}\n     en {y[0]} {dict(y[1])}'[:900]); break
    # 3) 数字
    na, nb = nums(a), nums(b)
    miss = na - nb; extra = nb - na
    if miss or extra:
        print(f'△ 数字差异（人工判断是否为合法换算）：')
        print('    中文有英文少:', dict(miss.most_common(40)))
        print('    英文多出:', dict(extra.most_common(40)))
    if render:
        bad += render_check(zh, en)
    print('✓ 通过' if not bad else f'✗ {bad} 类问题')
    return bad

SKEL = pathlib.Path(WORK+'/chunks/skeleton.html')  # 默认；render_check 里按块文件所在目录改

def render_check(zh, en, tol=1.08):
    from playwright.sync_api import sync_playwright
    sk = pathlib.Path(zh).resolve().parent / 'skeleton.html'
    skel = (sk if sk.exists() else SKEL).read_text()
    head = skel[:skel.index('@@CHUNK')]
    # 图用的 class 样式（.lbl/.xs/.xxs…）写在正文第一块的 <style> 里，不在 <head>
    first = (sk if sk.exists() else SKEL).parent / '00.zh.html'
    if first.exists():
        head += ''.join(re.findall(r'<style\b.*?</style>', first.read_text(), flags=re.S))
    js = """() => [...document.querySelectorAll('svg text')].map(t => {
        try { const b = t.getBBox(); return [b.width, (t.textContent||'').trim().slice(0,60)]; }
        catch(e) { return [0, '']; } })"""
    out = []
    with sync_playwright() as p:
        br = p.chromium.launch()
        for f in (zh, en):
            pg = br.new_page(viewport={'width': 1280, 'height': 900})
            body = pathlib.Path(f).read_text()
            pg.set_content(head + body + '</body></html>', wait_until='domcontentloaded')
            pg.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)")
            out.append(pg.evaluate(js))
        br.close()
    za, ea = out
    if len(za) != len(ea):
        print(f'✗ SVG text 数量不一致 {len(za)} vs {len(ea)}'); return 1
    # 目检过、确认不压框不撞字的超宽标签，登记在 ACCEPTED-WIDTH.md（R7 起）
    acc = pathlib.Path(WORK+'/rounds/ACCEPTED-WIDTH.md')
    ok = set()
    if acc.exists():
        for line in acc.read_text().splitlines():
            cols = [c.strip() for c in line.split('|')]
            if len(cols) >= 4: ok.add(cols[3][:60].strip())
    over = [(i, z, e) for i, (z, e) in enumerate(zip(za, ea))
            if z[0] > 0 and e[0] > z[0] * tol + 2 and e[1][:60].strip() not in ok]
    if over:
        print(f'✗ SVG 文字变宽 {len(over)} 处（> {tol:.0%}）：')
        for i, z, e in over[:60]:
            print(f'   #{i:<4} {e[0]/z[0]:5.2f}×  zh「{z[1][:24]}」  en "{e[1]}"')
        return 1
    print(f'✓ SVG 文字宽度 {len(za)} 处全部在 {tol:.0%} 内')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], '--render' in sys.argv))
