"""讲义块的英/中可见字符比（去掉 script/style 与标签，空白折成一个）。

    cd <讲义工作区>/chunks && python3 lecratio.py 00 01 …

L-R1 目标 ≤ 2.4，L-R2 目标 ≤ 2.30（经验：到这个比值，英文讲义整页高度约比中文高 5% 以内）。
"""
import re, sys, html
def vis(f):
    s = open(f).read()
    s = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', s, flags=re.S)
    s = html.unescape(re.sub(r'<[^>]+>', '', s))
    return len(re.sub(r'\s+', ' ', s).strip())
for n in sys.argv[1:]:
    z, e = vis(f'{n}.zh.html'), vis(f'{n}.en.html')
    print(n, z, e, round(e / z, 3))
