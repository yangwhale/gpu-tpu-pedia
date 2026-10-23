import os
from playwright.sync_api import sync_playwright
WORK = os.environ['I18N_WORK']   # 例 /tmp/t05en；输出到 $WORK/par/R3/figs/
with sync_playwright() as p:
    b=p.chromium.launch()
    for tag,f in (('zh',WORK+'/zh-ref.html'),('en',WORK+'/tmp.en.html')):
        pg=b.new_page(viewport={'width':1280,'height':900})
        pg.add_init_script("try{for(const k of Object.keys(localStorage))localStorage.removeItem(k)}catch(e){}")
        pg.goto('file://'+f); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1500)
        pg.evaluate("document.body.classList.remove('figonly'); document.querySelectorAll('details').forEach(d=>d.open=true)"); pg.wait_for_timeout(500)
        figs=pg.query_selector_all('figure')
        for i,fg in enumerate(figs):
            
            if not fg.is_visible(): print('hidden',i); continue
            fg.scroll_into_view_if_needed(); fg.screenshot(path=f'{WORK}/par/R3/figs/{i:02d}-fig{i}-{tag}.png')
        print(tag,len(figs))
    b.close()
