"""英文词数：正文（不含 SVG）与图内分开数。 python3 wc.py DIR"""
import re, sys, glob
from bs4 import BeautifulSoup
P = S = 0
for f in sorted(glob.glob(sys.argv[1] + '/[0-9][0-9].en.html')):
    sp = BeautifulSoup(open(f).read(), 'html.parser')
    for t in sp(['style', 'script']): t.decompose()
    svg = ' '.join(x.get_text(' ') for x in sp('svg'))
    for t in sp('svg'): t.decompose()
    p, s = len(sp.get_text(' ').split()), len(svg.split()); P += p; S += s
print(f'prose {P}  svg {S}  total {P+S}')
