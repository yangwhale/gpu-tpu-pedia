"""渲染后的页面：控制台错误 + 残留中文。 python3 dom.py /abs/path/page.html"""
import re
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
  b=p.chromium.launch(); pg=b.new_page(viewport={'width':1280,'height':900}); errs=[]
  pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type=='error' else None)
  pg.add_init_script("try{localStorage.setItem('figonly:'+location.pathname.split('/').pop(),'0')}catch(e){}")
  import sys; pg.goto('file://'+sys.argv[1]); pg.wait_for_timeout(2500)
  pg.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)")
  t=pg.evaluate("document.body.innerText + [...document.querySelectorAll('svg text')].map(x=>x.textContent).join(' ')")
  print('errors',errs[:5]); print('CJK in DOM:', sorted(set(re.findall(r'[一-鿿]+',t)))[:20])
  b.close()
