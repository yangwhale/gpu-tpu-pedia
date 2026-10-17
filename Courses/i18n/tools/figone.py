import sys, re, pathlib, subprocess
from playwright.sync_api import sync_playwright
fid=sys.argv[1]; import os; D=pathlib.Path(os.environ.get('I18N_WORK','/tmp/t02en')+'/chunks')
skel=(D/'skeleton.html').read_text(); head=skel[:skel.index('@@CHUNK')]
head+=''.join(re.findall(r'<style\b.*?</style>',(D/'00.zh.html').read_text(),flags=re.S))
src=None
for f in sorted(D.glob('[0-9][0-9].en.html')):
    t=f.read_text()
    if f'id="{fid}"' in t: src=t; break
m=re.search(r'<figure[^>]*id="%s".*?</figure>'%re.escape(fid),src,flags=re.S)
html=head+'<div class="wrap">'+m.group(0)+'</div></body></html>'
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={'width':1280,'height':900})
    pg.set_content(html,wait_until='domcontentloaded'); pg.wait_for_timeout(300)
    pg.query_selector('figure').screenshot(path=os.environ.get('I18N_WORK','/tmp/t02en')+f'/figs/check-{fid}.png'); b.close()
print(os.environ.get('I18N_WORK','/tmp/t02en')+f'/figs/check-{fid}.png')
