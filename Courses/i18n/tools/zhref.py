"""用中文块按英文同一条流水线拼出对照页（同样去讲课媒体、去链接、去重图号、** 转粗体），版面对比专用。
    python3 zhref.py /tmp/t02en/chunks /tmp/t02en/zh-ref.html"""
import re, sys, pathlib, importlib.util
spec = importlib.util.spec_from_file_location('asm', pathlib.Path(__file__).with_name('assemble.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pathlib.Path(sys.argv[1]); s = (d / 'skeleton.html').read_text()
s = re.sub(r'@@CHUNK:(\d+)@@', lambda k: (d / f'{k.group(1)}.zh.html').read_text(), s)
s = re.sub(r'(<b class="fno">[^<]*</b>)\s*(?:&nbsp;|　|\s)*<b class="fno">[^<]*</b>', r'\1', s)
s = re.sub(r'<!--(?:(?!-->).)*?[一-鿿](?:(?!-->).)*?-->', '', s, flags=re.S)
parts = re.split(r'(<(?:style|script)\b.*?</(?:style|script)>)', s, flags=re.S)
parts = [p if p.startswith(('<style', '<script')) else re.sub(r'\*\*([^*<]{1,80})\*\*', r'<b>\1</b>', p) for p in parts]
s = m.english_only_links(''.join(parts))
pathlib.Path(sys.argv[2]).write_text(s); print('wrote', sys.argv[2])
