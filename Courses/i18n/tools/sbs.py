"""中英并排切片（按锚点对齐）：两页同宽渲染，取 h2/h3/figure 作锚点一一对应，
相邻锚点之间为一段，左中右英拼图；超过 MAXH 的段再等分。 python3 sbs.py WIDTH"""
import os
WORK = os.environ.get('I18N_WORK', '/tmp/t02en')
TOPIC = os.environ.get('I18N_TOPIC', 'topic-02')
import sys, json
from playwright.sync_api import sync_playwright
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
W=int(sys.argv[1]); MAXH=1500
pages={'zh':WORK+'/zh-ref.html','en':os.environ.get('I18N_EN') or f'/home/chrisya/gpu-tpu-pedia/Courses/WebPages/{TOPIC}.en.html'}
A={}
with sync_playwright() as p:
    b=p.chromium.launch()
    for k,f in pages.items():
        pg=b.new_page(viewport={'width':W,'height':900}); pg.add_init_script("try{localStorage.setItem('figonly:'+location.pathname.split('/').pop(),'0')}catch(e){}"); pg.goto('file://'+f); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(2000)
        pg.screenshot(path=f'full-{k}-{W}.png', full_page=True)
        A[k]=pg.evaluate("[...document.querySelectorAll('section, h3, figure')].filter(h=>h.offsetParent!==null||h.tagName=='SECTION').map(h=>[Math.round(h.getBoundingClientRect().top+scrollY), h.tagName=='FIGURE'?('fig '+h.id):(h.tagName=='SECTION'?'section '+h.id:(h.innerText||'').slice(0,70))])")
        pg.close()
    b.close()
Z=Image.open(f'full-zh-{W}.png'); E=Image.open(f'full-en-{W}.png')
za=[0]+[y for y,_ in A['zh']]+[Z.height]; ea=[0]+[y for y,_ in A['en']]+[E.height]; labels=['(top)']+[t for _,t in A['en']]
assert len(za)==len(ea), (len(za),len(ea))
idx=[]; n=0
for i in range(len(za)-1):
    z0,z1,e0,e1=za[i],za[i+1],ea[i],ea[i+1]
    if z1-z0<5 and e1-e0<5: continue
    parts=max(1,-(-max(z1-z0,e1-e0)//MAXH))
    for k in range(parts):
        zs,ze=z0+(z1-z0)*k//parts, z0+(z1-z0)*(k+1)//parts; es,ee=e0+(e1-e0)*k//parts, e0+(e1-e0)*(k+1)//parts
        h=max(ze-zs,ee-es,1); c=Image.new('RGB',(W*2+20,h),(255,255,255))
        if ze>zs: c.paste(Z.crop((0,zs,W,ze)),(0,0))
        if ee>es: c.paste(E.crop((0,es,W,ee)),(W+20,0))
        for yy in range(h): c.putpixel((W+9,yy),(255,0,0))
        fn=f's{W}-{n:03d}.png'; c.save(fn); n+=1
        idx.append({'file':fn,'starts_at':labels[i],'part':f'{k+1}/{parts}','zh_h':ze-zs,'en_h':ee-es})
json.dump(idx,open(f'index-{W}.json','w'),indent=0,ensure_ascii=False); print(n,'slices')
