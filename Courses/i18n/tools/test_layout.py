"""layout.py 的负例测试：python3 test_layout.py [layout.py 路径]
① 英文页少一个元素（去掉指向无英文版页面的按钮）时，③ 节高仍按节配对，不因错位报出 24 px 之类的假高度；
② 专题七的折叠键 "t7fo:<文件名>"（默认折叠）也要被展开，节高要包含讲稿。旧版 layout.py 两条都失败。"""
import sys, subprocess, tempfile, pathlib, re
LAYOUT = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).with_name('layout.py'))
FOLD = """<style>body.figonly .narr{display:none}</style><script>
var K="t7fo:"+location.pathname.split("/").pop();try{if((localStorage.getItem(K)||"1")==="1")document.body.classList.add("figonly")}catch(e){document.body.classList.add("figonly")}</script>"""
def page(extra_btn):
    secs = ''.join(f'<section id="s{k}"><h2>Section {k}</h2>' + ('<a class="lecbtn" href="x.html">notes</a>' if extra_btn else '')
                   + f'<div style="height:{300*(k+1)}px">fig</div><p class="narr" style="height:500px">narration</p></section>' for k in range(3))
    return f'<!doctype html><html><body><div class="hero"><h1>T</h1></div>{secs}{FOLD}</body></html>'
d = pathlib.Path(tempfile.mkdtemp()); (d / 'zh.html').write_text(page(True)); (d / 'en.html').write_text(page(False))
out = subprocess.run(['python3', LAYOUT, str(d / 'zh.html'), str(d / 'en.html'), '--width', '1280'], capture_output=True, text=True).stdout
print(out)
rows = dict((m[0], (int(m[1]), int(m[2]))) for m in re.findall(r'^\s+(s\d)\s+zh\s+(\d+) en\s+(\d+)', out, re.M))
ok = True
for k in range(3):
    z, e = rows.get(f's{k}', (0, 0))
    # 每节 = 标题 + 图 300(k+1) + 讲稿 500；中英只差一个行内按钮（同一行），节高应几乎相等且含讲稿
    if not (abs(z - e) <= 30 and e >= 300 * (k + 1) + 500): print(f'FAIL s{k}: zh {z} en {e}'); ok = False
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
