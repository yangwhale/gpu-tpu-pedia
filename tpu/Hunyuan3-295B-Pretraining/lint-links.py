import re
from pathlib import Path
def gh(t):
    t=re.sub(r'`|\*\*|\*|<[^>]+>','',t).strip().lower()
    t=re.sub(r'[^\w\s\-]','',t,flags=re.U)
    return re.sub(r'\s','-',t,flags=re.U)
root=Path('.')
files=sorted(root.glob('*.md'))
anc={f.name:{gh(m.group(1)) for m in re.finditer(r'^#{1,6}\s+(.*)$',f.read_text(),re.M)} for f in files}
bad=[]
for f in files:
    for i,line in enumerate(f.read_text().split('\n'),1):
        for m in re.finditer(r'\[([^\]]*)\]\(([^)]+)\)',line):
            t=m.group(2).strip()
            if t.startswith(('http','mailto:')): continue
            path,_,a=t.partition('#')
            if path:
                p=(f.parent/path).resolve()
                if not p.exists(): bad.append((f.name,i,t,'FILE MISSING')); continue
                if a and p.suffix=='.md':
                    aa=anc.get(p.name) or {gh(x.group(1)) for x in re.finditer(r'^#{1,6}\s+(.*)$',p.read_text(),re.M)}
                    if a not in aa: bad.append((f.name,i,t,'ANCHOR'))
            elif a and a not in anc[f.name]: bad.append((f.name,i,t,'ANCHOR(self)'))
print(f"{len(files)} 文件, {len(bad)} 处问题")
for n,i,t,w in bad: print(f"  {n}:{i} [{w}] {t}")
