"""在不动图的前提下把 SVG 英文字放大：逐组试 1.25→1.05 倍，放得下才放。

    python3 svgfit.py NN [--max 1.25] [--dry]
判据（全部在浏览器里用真实字体量）：
  · 文字仍在它所属的最小 rect 内（留 2px），没有 rect 就在 svg 画布内；
  · 不碰任何别的文字；不压到原本没压到的线条 / 折线 / 小 path；
  · 同一列（同 svg、同 class、同 x、同 anchor）的几行一起放，行距节奏不乱。
只动基础字号 < 15px 的字（标题本来就够大）。结果以内联 style="font-size:…" 写回 NN.en.html。
"""
import sys, re, json, pathlib, importlib.util
from playwright.sync_api import sync_playwright
n = sys.argv[1]; MAX = float(sys.argv[sys.argv.index('--max')+1]) if '--max' in sys.argv else 1.25
DRY = '--dry' in sys.argv
D = pathlib.Path('/tmp/t02en/chunks'); T = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('asm', T/'assemble.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
skel = (D/'skeleton.html').read_text(); head = skel[:skel.index('@@CHUNK')]
head += ''.join(re.findall(r'<style\b.*?</style>', (D/'00.zh.html').read_text(), flags=re.S))
ff = (T.parent/'fonts'/'ss3-embedded.css').read_text()
head = head.replace('lang="zh-CN"', 'lang="en"').replace('</title>', '</title><style>'+ff+'</style>')
head = re.sub(r'font:400 16px/1.75 [^;]*;', 'font:400 16px/1.75 ' + m.LATIN + ';', head)
src = (D/f'{n}.en.html').read_text()
# ⛔ 2026-10-17：基准必须是「放大之前」的字号，不能是当前字号 —— 否则每跑一次就在上一次的基础上再放，
#    一列字跑三遍能到原来的 2 倍多（r11a 抓到的）。基准取自放大前的快照（同结构、同序号）。
PRE = pathlib.Path('/tmp/t02en/rounds/R10b-prefit')/f'{n}.en.html'
def inline_fs(t):
    m_ = re.search(r'style="[^"]*font-size:([\d.]+)px', t); return float(m_.group(1)) if m_ else None
orig = [inline_fs(t) for t in re.findall(r'<text\b[^>]*>', PRE.read_text())] if PRE.exists() else None
body = src.replace("'Noto Sans CJK SC'", "'Source Sans 3','Noto Sans CJK SC'")
JS = r"""([MAX, ORIG]) => {
 const SC=[1.45,1.4,1.35,1.3,1.25,1.2,1.15,1.1,1.05].filter(s=>s<=MAX+1e-9);
 const all=[...document.querySelectorAll('svg text')]; const res=new Array(all.length).fill(null);
 const box=e=>e.getBoundingClientRect();
 const inter=(a,b,p=0)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>p && Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>p;
 document.querySelectorAll('svg').forEach(svg=>{
  const sb=box(svg); const T=[...svg.querySelectorAll('text')];
  const rects=[...svg.querySelectorAll('rect')].map(box).filter(r=>r.width>20&&r.height>12);
  const strokes=[...svg.querySelectorAll('line,polyline,path,circle')].map(box).filter(r=>r.width*r.height<40000||r.width<3||r.height<3)
     .concat([...svg.querySelectorAll('rect')].map(box).filter(r=>r.width<=20||r.height<=12));
  const base=new Map(), cont=new Map(), hit0=new Map();
  const cur=new Map();
  T.forEach(t=>{ cur.set(t,parseFloat(getComputedStyle(t).fontSize));
    if(ORIG){ const o=ORIG[all.indexOf(t)]; const keep=t.style.fontSize; t.style.fontSize=(o?o+'px':''); base.set(t,parseFloat(getComputedStyle(t).fontSize)); t.style.fontSize=keep; }
    else base.set(t,cur.get(t)); });
  T.forEach(t=>{const b=box(t);
    const cx=b.left+2, cy=b.top+b.height/2;
    const c=rects.filter(r=>cx>=r.left&&cx<=r.right&&cy>=r.top&&cy<=r.bottom).sort((a,c)=>a.width*a.height-c.width*c.height)[0]||sb;
    cont.set(t,c); hit0.set(t,new Set(strokes.map((s,i)=>inter(b,s,0.5)?i:-1).filter(i=>i>=0)));});
  const groups=new Map(), grp=new Map();
  // 字号层级不能乱：任何标签放大后都不超过这张图标题（第一个 text / .ttl）的字号
  const ttl=svg.querySelector('text.ttl')||T[0]; const TTL=ttl?parseFloat(getComputedStyle(ttl).fontSize):99;
  T.forEach(t=>{ if(base.get(t)>=15||!(t.textContent||'').trim()) return;
    const c=cont.get(t); const k=[t.getAttribute('class'),t.getAttribute('x'),t.getAttribute('text-anchor'),Math.round(c.left),Math.round(c.top)].join('|');
    if(!groups.has(k)) groups.set(k,[]); groups.get(k).push(t); grp.set(t,k);});
  // 「不变得更糟」语义：原本就贴边/越界/相交的，只要求不比原来更糟（左对齐贴 x=0 的标签也能往右长）
  const b0=new Map(); T.forEach(t=>b0.set(t,box(t)));
  const ov0=new Set(); T.forEach((a,i)=>T.forEach((c,j)=>{ if(i<j && inter(b0.get(a),b0.get(c),0.5)) ov0.add(i+'|'+j); }));
  const idx=new Map(); T.forEach((t,i)=>idx.set(t,i));
  const ok=t=>{const b=box(t), c=cont.get(t), o=b0.get(t);
    if(b.left < Math.min(o.left, c.left+2)-0.01 || b.right > Math.max(o.right, c.right-2)+0.01) return false;
    if(b.top < Math.min(o.top, c.top+1)-0.01 || b.bottom > Math.max(o.bottom, c.bottom-1)+0.01) return false;
    if(b.left < Math.min(o.left, sb.left)-0.01 || b.right > Math.max(o.right, sb.right)+0.01) return false;
    if(b.top < Math.min(o.top, sb.top)-0.01 || b.bottom > Math.max(o.bottom, sb.bottom)+0.01) return false;
    for(const q of T){ if(q===t) continue; const i=idx.get(t), j=idx.get(q); const qb=box(q);
      // 同一列的上下行：文字框（≈1.42em）允许重叠到行距不低于 1.14em，按字形看不相碰
      if(grp.get(q)===grp.get(t)){ const ox=Math.min(b.right,qb.right)-Math.max(b.left,qb.left);
        // ⭐ 按基线间距判（SVG 用户坐标）：文字框高度在 Chrome 里几乎不随字号变（≈16 单位），拿框判会把
        //    所有 12–15 行距的多行块都判成相撞。基线距 ≥ 1.14 × 字号，字形就不会碰。
        const y1=parseFloat(t.getAttribute('y')), y2=parseFloat(q.getAttribute('y'));
        const f=Math.max(parseFloat(getComputedStyle(t).fontSize), parseFloat(getComputedStyle(q).fontSize));
        if(ox>0.5 && !isNaN(y1) && !isNaN(y2) && Math.abs(y1-y2) < 1.14*f - 0.01) return false;
        if(ox>0.5 && (isNaN(y1)||isNaN(y2))){ const oy=Math.min(b.bottom,qb.bottom)-Math.max(b.top,qb.top); if(oy>0.28*f) return false; }
        continue; }
      if(ov0.has(Math.min(i,j)+'|'+Math.max(i,j))) continue;   // 原本就相交的别组文字（中文也如此）不再加判
      // 别组文字之间至少留 4px（框各外扩 2px 再判），避免「128☑」这种贴死
      const bx={left:b.left-2,right:b.right+2,top:b.top,bottom:b.bottom}, qx={left:qb.left-2,right:qb.right+2,top:qb.top,bottom:qb.bottom};
      if(inter(bx,qx,0.5)) return false; }
    const h0=hit0.get(t); for(let i=0;i<strokes.length;i++){ if(!h0.has(i)&&inter(b,strokes[i],0.5)) return false; }
    return true;};
  for(const [k,g] of groups){
    // 原本就不满足的组跳过（不让它更糟）
    for(const s of SC){
      if(g.some(t=>base.get(t)*s <= cur.get(t)+0.05)) break;   // 不比现在大就不动（永不缩小）
      if(g.some(t=>t!==ttl && base.get(t)*s > TTL)) continue;
      // 等宽字体（数字/代码类）放大后又宽又重，上限压到 1.25
      if(s>1.25 && g.some(t=>/mono/i.test(getComputedStyle(t).fontFamily))) continue;
      g.forEach(t=>t.style.fontSize=(base.get(t)*s).toFixed(2)+'px');
      if(g.every(ok)){ g.forEach(t=>res[all.indexOf(t)]=+(base.get(t)*s).toFixed(2)); break; }
      g.forEach(t=>t.style.fontSize=cur.get(t)+'px');
    }
  }
 });
 return res; }"""
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1280, 'height': 900})
    pg.set_content(head + '<div class="wrap">' + body + '</div></body></html>', wait_until='load')
    pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(1000)
    pg.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)"); pg.wait_for_timeout(300)
    res = pg.evaluate(JS, [MAX, orig]); b.close()
tags = list(re.finditer(r'<text\b[^>]*>', src))
assert len(tags) == len(res), (len(tags), len(res))
out, last, cnt = [], 0, 0
for mt, fs in zip(tags, res):
    out.append(src[last:mt.start()]); tag = mt.group(0); last = mt.end()
    if fs:
        cnt += 1
        if 'style="' in tag:
            tag = re.sub(r'style="([^"]*)"', lambda k: 'style="' + (re.sub(r'font-size:[^;"]*;?', '', k.group(1)).rstrip(';') + ';' if re.sub(r'font-size:[^;"]*;?', '', k.group(1)).strip() else '') + f'font-size:{fs}px"', tag, count=1)
        else:
            tag = tag[:-1] + f' style="font-size:{fs}px">'
    out.append(tag)
out.append(src[last:])
print(f'chunk {n}: {cnt}/{len(res)} labels enlarged')
if not DRY: (D/f'{n}.en.html').write_text(''.join(out))
