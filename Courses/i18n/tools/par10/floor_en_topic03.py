"""Recount the English floor table + time math in chunk 00 from chunks 00-08 (spoken lines = <p class="say...">).
Rate: 140 words/min (~ zh 200 chars/min x 0.70 EN words per ZH char); 15 s per <p class="board">. Idempotent. Run: python3 /tmp/t03lec/floor_en.py"""
import re,html as H
D='/tmp/t03lec/chunks/'
WPM=140
s="".join(open(f"{D}{i:02d}.en.html").read() for i in range(9))
def words(t):
    txt=" ".join(H.unescape(re.sub(r"<[^>]+>"," ",x)) for x in re.findall(r'<p class="say[^"]*">(.*?)</p>',t,re.S))
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,×%-]*",txt))
names=["1 · A chain","2 · The 2017 cut","3 · Bill due","4 · Cut heads",
 "5 · MLA","6 · Read less","7 · Back to start",
 "8 · Use both","9 · Machine + close"]
plan=[7,10,7,8,12,12,10,8,10]
heads=[m for m in re.finditer(r'<h3 class="sec"[^>]*id="c(\d+)"',s) if m.group(1)!="1"]
assert len(heads)==9,len(heads)
end=s.index("🚧")
_zt=re.search(r'<table><thead><tr><th>章</th>.*?</table>',open(D+'00.zh.html').read(),re.S).group(0)
ZHRED=[('c5221f' in r) for r in re.findall(r'<tr><td>第.*?</tr>',_zt)][:9]
assert len(ZHRED)==9,ZHRED
rows=[];fl=0
for i,m in enumerate(heads):
    t=s[m.start():(heads[i+1].start() if i+1<9 else end)]
    n=words(t); c=len(re.findall(r'<p class="board">',t)); f=n/WPM+c*0.25; fl+=f
    # structure parity with zh: red cell exactly where the zh table has it (zh build flags floor > budget)
    bad=' style="color:#c5221f;font-weight:700"' if ZHRED[i] else ''
    rows.append(f'<tr><td>{names[i]}</td><td>{n:,}</td><td>{c}</td><td{bad}>{f:.1f}′</td><td>{plan[i]}′</td></tr>')
    print(names[i],n,c,round(f,1))
tab=('<table><thead><tr><th>Section</th><th>Script (words)</th><th>Switches</th><th>Floor</th><th>87′ budget</th></tr></thead><tbody>'
     +"".join(rows)+f'<tr><td><b>All nine sections</b></td><td colspan="2"></td><td><b>{fl:.0f}′</b></td><td><b>84′</b></td></tr></tbody></table>')
W=words(s); C=len(re.findall(r'<p class="board">',s)); wmin=W/WPM; cmin=C*15/60; tot=wmin+cmin
print('total words',W,'cuts',C,'min',round(wmin),round(tot))
p=D+'00.en.html'; e=open(p).read()
e,k=re.subn(r'<table><thead><tr><th>Section</th>.*?</table>',lambda m:tab,e,count=1,flags=re.S); assert k==1
for ph,v in (('@@W@@',f'{W:,}'),('@@WMIN@@',f'{wmin:.0f}'),('@@TOT@@',f'{tot:.0f}'),('@@B87@@',f'{87-tot:.0f}'),('@@B60@@',f'{60-tot:.0f}')):
    e=e.replace(ph,v)
# idempotent re-run on already-filled numbers
e=re.sub(r'(<b>All spoken lines total )[\d,]+( English words</b>)',lambda m:f'{m.group(1)}{W:,}{m.group(2)}',e)
e=re.sub(r'(at 140 words/min\n= <b>)\d+( minutes</b>\) \+ <b>)\d+( figure switches</b>)',lambda m:f'{m.group(1)}{wmin:.0f}{m.group(2)}{C}{m.group(3)}',e)
e=re.sub(r'(= <b>)[\d.]+( minutes</b>\)\.\n<b>That\'s )\d+( minutes)',lambda m:f'{m.group(1)}{cmin:.1f}{m.group(2)}{tot:.0f}{m.group(3)}',e)
e=re.sub(r'(leaves <b>)-?\d+( minutes</b> for figures)',lambda m:f'{m.group(1)}{87-tot:.0f}{m.group(2)}',e)
e=re.sub(r'(only <b>)?(only )-?\d+( minutes of buffer)',lambda m:f'{m.group(2)}{60-tot:.0f}{m.group(3)}',e)
open(p,'w').write(e)
