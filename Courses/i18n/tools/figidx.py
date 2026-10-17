"""按序号截英文页里的第 N 张图（先用当前块拼一份私有临时页，不碰 WebPages）。
    I18N_WORK=/tmp/t01en I18N_TOPIC=topic-01 python3 figidx.py 7 [12 ...]
输出 $I18N_WORK/figs/now-NN.png"""
import os, sys, subprocess, pathlib
from playwright.sync_api import sync_playwright
WORK = os.environ.get('I18N_WORK', '/tmp/t02en'); TOPIC = os.environ.get('I18N_TOPIC', 'topic-02')
tmp = pathlib.Path(WORK) / 'figs' / f'_tmp-{os.getpid()}.html'
web = pathlib.Path(__file__).resolve().parents[2] / 'WebPages'
# 放在 WebPages 同级相对路径不对图片；临时页放 WORK/figs，并把 img 链过去
(pathlib.Path(WORK) / 'figs' / 'img').exists() or os.symlink(web / 'img', pathlib.Path(WORK) / 'figs' / 'img')
subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('assemble.py')), TOPIC, WORK + '/chunks'],
               env=dict(os.environ, I18N_OUT=str(tmp)), check=True, capture_output=True)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1280, 'height': 900})
    pg.add_init_script("try{localStorage.setItem('figonly:'+location.pathname.split('/').pop(),'0')}catch(e){}")
    pg.goto('file://' + str(tmp)); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1500)
    pg.evaluate("document.body.classList.remove('figonly'); document.querySelectorAll('details').forEach(d=>d.open=true)"); pg.wait_for_timeout(300)
    figs = pg.query_selector_all('figure')
    for a in sys.argv[1:]:
        i = int(a); f = figs[i]; f.scroll_into_view_if_needed(); pg.wait_for_timeout(100)
        out = pathlib.Path(WORK) / 'figs' / f'now-{i:02d}.png'; f.screenshot(path=str(out)); print(out)
    b.close()
tmp.unlink()
