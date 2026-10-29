"""准备一轮版式对比的材料：python3 mk.py <topic-NN> <work> <round> [widths...]
zh-ref = 当前线上中文页去掉讲课录音块；en = 工作区块拼出的英文页；产出并排切片 + 逐块高度 + 重叠报告。"""
import sys, re, os, subprocess, pathlib, json
topic, work, rnd = sys.argv[1], sys.argv[2], sys.argv[3]
widths = sys.argv[4:] or ['1280', '390']
T = pathlib.Path(__file__).resolve().parents[1]   # Courses/i18n/tools
W = pathlib.Path(__file__).resolve().parents[3] / 'WebPages'
zh = (W / f'{topic}.html').read_text()
zh = re.sub(r'<div class="lecmedia".*?</video></div></div>', '', zh, flags=re.S)
zh = re.sub(r'<!--lec:s[0-9a-z]+-->.*?<!--/lec:s[0-9a-z]+-->', '', zh, flags=re.S)
zh = re.sub(r'\s*<a class="lecbtn" href="[^"]*\.en\.html"[^>]*>🌐 English</a>', '', zh)
# 相对路径的媒体/图片指回 WebPages，截图才看得到
for k in ('src="media/', 'src="img/', 'href="media/', 'poster="img/', 'poster="media/'):
    zh = zh.replace(k, k.replace('"', '"file://%s/' % W, 1))
out = pathlib.Path(work) / 'par' / rnd; out.mkdir(parents=True, exist_ok=True)
(pathlib.Path(work) / 'zh-ref.html').write_text(zh)
env = dict(os.environ, I18N_WORK=work, I18N_TOPIC=topic, I18N_OUT=f'{work}/tmp.en.html')
subprocess.run(['python3', str(T / 'assemble.py'), topic, f'{work}/chunks'], env=env, capture_output=True)
en = pathlib.Path(f'{work}/tmp.en.html').read_text()
for k in ('src="media/', 'src="img/', 'href="media/', 'poster="img/', 'poster="media/'):
    en = en.replace(k, k.replace('"', '"file://%s/' % W, 1))
pathlib.Path(f'{work}/tmp.en.html').write_text(en)
for w in widths:
    subprocess.run(['python3', str(T / 'sbs.py'), w], cwd=out, env=dict(env, I18N_EN=f'{work}/tmp.en.html'), capture_output=True)
    r = subprocess.run(['python3', str(T / 'layout.py'), f'{work}/zh-ref.html', f'{work}/tmp.en.html', '--width', w, '--top', '60'], capture_output=True, text=True)
    (out / f'layout-{w}.txt').write_text(r.stdout + r.stderr)
for w in widths:
    r = subprocess.run(['python3', str(pathlib.Path(__file__).resolve().parent / 'lines.py'), f'{work}/zh-ref.html', f'{work}/tmp.en.html', w], capture_output=True, text=True)
    (out / f'lines-{w}.txt').write_text(r.stdout + r.stderr)
    print('lines', w, (r.stdout.strip().splitlines() or [''])[-1])
r = subprocess.run(['python3', str(T / 'overlap.py'), f'{work}/zh-ref.html', f'{work}/tmp.en.html'], capture_output=True, text=True)
(out / 'overlap.txt').write_text(r.stdout + r.stderr)
print(out, (out / 'overlap.txt').read_text().splitlines()[:1])
for w in widths:
    t = (out / f'layout-{w}.txt').read_text()
    i = t.find('③'); print(w, t[:t.find('\n')]); print(t[i:i+900] if i >= 0 else '')
