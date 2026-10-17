"""单块中英并排渲染（改完自查用）：python3 chunkview.py NN WIDTH OUTDIR
产出 OUTDIR/cvNN-WIDTH-kkk.png（按锚点对齐的并排切片）。只用当前块文件，不碰整页。"""
import sys, re, pathlib, importlib.util
from playwright.sync_api import sync_playwright
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
n, W, out = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
D = pathlib.Path('/tmp/t02en/chunks'); T = pathlib.Path('/home/chrisya/gpu-tpu-pedia/Courses/i18n/tools')
spec = importlib.util.spec_from_file_location('asm', T/'assemble.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
skel = (D/'skeleton.html').read_text(); head = skel[:skel.index('@@CHUNK')]
head += ''.join(re.findall(r'<style\b.*?</style>', (D/'00.zh.html').read_text(), flags=re.S))
ff = (T.parent/'fonts'/'ss3-embedded.css').read_text()
def page(lang):
    body = (D/f'{n}.{lang}.html').read_text()
    body = re.sub(r'(<b class="fno">[^<]*</b>)\s*(?:&nbsp;|　|\s)*<b class="fno">[^<]*</b>', r'\1', body)
    body = m.english_only_links(body)
    h = head
    if lang == 'en':
        h = h.replace('lang="zh-CN"', 'lang="en"').replace('</title>', '</title><style>'+ff+'</style>')
        h = h.replace('"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",-apple-system,"Segoe UI",Roboto,sans-serif', m.LATIN)
        body = body.replace("'Noto Sans CJK SC'", "'Source Sans 3','Noto Sans CJK SC'")
    return h + body + '</body></html>'
shots = {}; anchors = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('zh', 'en'):
        pg = b.new_page(viewport={'width': W, 'height': 900}); pg.set_content(page(lang), wait_until='load')
        pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(800)
        f = out/f'_full-{n}-{lang}-{W}.png'; pg.screenshot(path=str(f), full_page=True); shots[lang] = Image.open(f)
        anchors[lang] = pg.evaluate("[...document.querySelectorAll('section,h3,figure,.note,details')].filter(e=>!e.closest('details:not([open]) > :not(summary)')).map(e=>Math.round(e.getBoundingClientRect().top+scrollY))")
        pg.close()
    b.close()
Z, E = shots['zh'], shots['en']; za = [0]+anchors['zh']+[Z.height]; ea = [0]+anchors['en']+[E.height]
if len(za) != len(ea): za, ea = [0, Z.height], [0, E.height]
k = 0
for i in range(len(za)-1):
    z0, z1, e0, e1 = za[i], za[i+1], ea[i], ea[i+1]
    if z1-z0 < 30 and e1-e0 < 30: continue
    h = max(z1-z0, e1-e0); c = Image.new('RGB', (W*2+20, h), 'white')
    if z1 > z0: c.paste(Z.crop((0, z0, W, z1)), (0, 0))
    if e1 > e0: c.paste(E.crop((0, e0, W, e1)), (W+20, 0))
    c.save(out/f'cv{n}-{W}-{k:03d}.png'); k += 1
print(k, 'slices in', out)
