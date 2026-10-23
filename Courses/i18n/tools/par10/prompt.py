import sys, pathlib
P = pathlib.Path(__file__).resolve().parent
topic, work, rnd, chunks = sys.argv[1:5]
foc = {l.split(':',1)[0]: l.split(':',1)[1].strip() for l in open(P / 'FOCUS.md') if ':' in l}
s = open(P / 'COMMON.md').read()
s = s.replace('{WORK}', work).replace('{TOPIC}', topic).replace('{ROUND}', rnd).replace('{CHUNKS}', chunks).replace('{FOCUS}', foc[rnd]).replace('{REPO}', str(P.parents[3]))
open(f'{work}/par/{rnd}/PROMPT-{chunks.replace(",","").replace(" ","")}.md', 'w').write(s)
print(f'{work}/par/{rnd}/PROMPT-{chunks.replace(",","").replace(" ","")}.md')
