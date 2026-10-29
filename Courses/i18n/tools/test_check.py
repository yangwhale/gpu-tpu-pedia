"""check.py 的 src/poster/href 归一化测试：python3 test_check.py
正例：英文块换成 img/X.en.png?v=… / media/X.en.mp4?v=… 后，与中文 img/X.png / media/X.mp4?v=… 结构对账一致。
负例：换了别的文件、别的扩展名、别的目录、非十六进制 ?v=、.en 不在扩展名前，都必须报结构不一致。
旧版 check.py 只认 media/，img/X.en.png 正例会失败。"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bs4 import BeautifulSoup
import check

def seq(html):
    return check.tagseq(BeautifulSoup(html, 'html.parser'))

def img(src): return f'<figure><img src="{src}" alt="a" width="10" height="5"></figure>'
def vid(src, poster): return f'<video src="{src}" poster="{poster}" data-pauses="1,2"></video>'

SAME = [  # (zh, en) 应视为同构
    (img('img/topic07-20-overview.png'), img('img/topic07-20-overview.en.png?v=1a2b3c4d')),
    (img('img/topic07-20-overview.png'), img('img/topic07-20-overview.en.png')),
    (vid('media/topic07-agdone.mp4?v=ca6765c0', 'img/topic07-agdone.png'),
     vid('media/topic07-agdone.en.mp4?v=e088bdc3', 'img/topic07-agdone.en.png?v=0011aabb')),
    ('<a href="media/topic07-xrec-open.mp4">x</a>', '<a href="media/topic07-xrec-open.en.mp4">x</a>'),  # 「Full video」链接
]
DIFF = [  # (zh, en) 必须报不一致
    (img('img/topic07-20-overview.png'), img('img/topic07-21a-step-fp8old.en.png?v=1a2b3c4d')),  # 别的文件
    (img('img/topic07-20-overview.png'), img('img/topic07-20-overview.en.jpg?v=1a2b3c4d')),      # 别的扩展名
    (img('assets/topic07-20-overview.png'), img('assets/topic07-20-overview.en.png')),           # 不在 img/ media/ 下
    (img('img/topic07-20-overview.png'), img('img/topic07-20-overview.en.png?v=XYZ')),           # ?v= 非十六进制
    (img('img/topic07-20-overview.png'), img('img/topic07-20-overview.png.en')),                 # .en 不在扩展名前
    (img('img/topic07-20-overview.png'), img('img/topic07-20-overview.en.png') .replace('width="10"', 'width="11"')),  # 其它属性仍逐字比
    (vid('media/topic07-agdone.mp4?v=ca6765c0', 'img/topic07-agdone.png'),
     vid('media/topic07-agdone.en.mp4?v=e088bdc3', 'img/topic07-experts.en.png?v=0011aabb')),     # poster 换错图
    ('<a href="media/topic07-xrec-open.mp4">x</a>', '<a href="media/topic07-xrec-mem.en.mp4">x</a>'),   # href 换错录像
    ('<a href="topic-07.html">x</a>', '<a href="topic-07.en.html">x</a>'),                              # 页面链接不归一化
]
ok = True
for i, (z, e) in enumerate(SAME):
    if seq(z) != seq(e): print(f'FAIL same#{i}: {seq(e)}'); ok = False
for i, (z, e) in enumerate(DIFF):
    if seq(z) == seq(e): print(f'FAIL diff#{i}: normalized to equal: {e}'); ok = False
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
