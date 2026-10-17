"""把中文成品页切成可分派翻译的块。

    python3 split.py topic-02 /tmp/t02en/chunks

产出：skeleton.html（<head> 与 CSS/JS，块位置用 @@CHUNK:nn@@ 占位）＋ 若干 nn.zh.html。
预处理：去 HTML 注释（开发备忘，英文版不需要）、去讲课录音/视频块（中文音频，英文版暂不做）。
切点：<section> 边界；超过 MAX_CJK 的 section 再按 <h3> 切。
"""
import re, sys, pathlib

import os
MAX_CJK = int(os.environ.get("MAX_CJK", 15000))
CJK = re.compile(r'[㐀-鿿＀-￯　-〿]')

def main(topic, out):
    src = pathlib.Path(__file__).resolve().parents[2] / 'WebPages' / f'{topic}.html'
    s = src.read_text()
    # 注释只在 <style>/<script> 之外去
    parts = re.split(r'(<(?:style|script)\b.*?</(?:style|script)>)', s, flags=re.S)
    parts = [p if p.startswith(('<style', '<script')) else re.sub(r'<!--.*?-->', '', p, flags=re.S)
             for p in parts]
    s = ''.join(parts)
    s = re.sub(r'<div class="lecmedia".*?</video></div></div>', '', s, flags=re.S)
    b0 = s.index('<body')
    b0 = s.index('>', b0) + 1
    b1 = s.rindex('</body>')
    body = s[b0:b1]
    # body 里的 <script> 留在骨架里，不给译者
    pieces = re.split(r'(?=<section\b)', body)
    units = []
    for p in pieces:
        if len(CJK.findall(p)) > MAX_CJK:
            sub = re.split(r'(?=<h3\b)', p)
            buf = ''
            for q in sub:
                if buf and len(CJK.findall(buf + q)) > MAX_CJK:
                    units.append(buf); buf = ''
                buf += q
            units.append(buf)
        else:
            units.append(p)
    # 合并太小的相邻块
    merged = []
    for u in units:
        if merged and len(CJK.findall(merged[-1] + u)) <= MAX_CJK * 0.8:
            merged[-1] += u
        else:
            merged.append(u)
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    skel = s[:b0]
    for i, u in enumerate(merged):
        (out / f'{i:02d}.zh.html').write_text(u)
        skel += f'@@CHUNK:{i:02d}@@'
        print(f'{i:02d}  {len(u):>8} bytes  {len(CJK.findall(u)):>6} CJK  '
              f'{len(re.findall("<svg", u))} svg  starts: {re.sub(r"<[^>]+>|\s+", " ", u[:300])[:60]}')
    skel += s[b1:]
    (out / 'skeleton.html').write_text(skel)
    print('skeleton CJK', len(CJK.findall(skel)))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
