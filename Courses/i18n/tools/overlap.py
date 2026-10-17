import sys, json
from playwright.sync_api import sync_playwright
JS="""() => { const out=[]; document.querySelectorAll('svg').forEach((s,si)=>{
  const ts=[...s.querySelectorAll('text')].map(t=>{const r=t.getBoundingClientRect();return [r.left,r.top,r.right,r.bottom,(t.textContent||'').trim().slice(0,40),t.getAttribute('x'),t.getAttribute('text-anchor'),parseFloat(t.getAttribute('y')),parseFloat(getComputedStyle(t).fontSize)]});
  const sr=s.getBoundingClientRect();
  for(let i=0;i<ts.length;i++){const a=ts[i]; if(a[2]-a[0]<1)continue;
    if(a[0]<sr.left-2||a[2]>sr.right+2) out.push([si,i,-1,'OUT',a[4]]);
    for(let j=i+1;j<ts.length;j++){const b=ts[j]; if(b[2]-b[0]<1)continue;
      const ox=Math.min(a[2],b[2])-Math.max(a[0],b[0]), oy=Math.min(a[3],b[3])-Math.max(a[1],b[1]);
      // 同一列上下行（同 x、同 anchor）：文字框在 Chrome 里偏高，按基线距判（≥1.1×字号即不相碰）
      if(a[5]!==null&&a[5]===b[5]&&a[6]===b[6]&&!isNaN(a[7])&&!isNaN(b[7])){ if(ox>3&&Math.abs(a[7]-b[7])<1.1*Math.max(a[8],b[8])) out.push([si,i,j,a[4],b[4]]); continue; }
      if(ox>3&&oy>3) out.push([si,i,j,a[4],b[4]]);}}});
  return out; }"""
def run(f):
    with sync_playwright() as p:
        b=p.chromium.launch(); pg=b.new_page(viewport={'width':1280,'height':900}); pg.add_init_script("try{for(const k of ['figonly:topic-02.en.html','figonly:zh-ref.html']) localStorage.setItem(k,'0')}catch(e){}")
        pg.goto('file://'+f); pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1500); pg.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)"); pg.wait_for_timeout(500)
        r=pg.evaluate(JS); b.close(); return r
zh=run(sys.argv[1]); en=run(sys.argv[2])
zk={(a,b,c) for a,b,c,*_ in zh}
new=[x for x in en if (x[0],x[1],x[2]) not in zk]
print('zh overlaps',len(zh),'en overlaps',len(en),'new in en',len(new))
for x in new: print(x)
