"""把英文块装回骨架，产出 WebPages/<topic>.en.html。

    python3 assemble.py topic-02 /tmp/t02en/chunks

骨架里的少量中文（title / og）用下面的 HEAD 表替换；字体栈改成拉丁字体优先。
"""
import re, sys, pathlib

HEAD = {
    'topic-02': {
        'title': 'Topic 2 · TPU vs GPU (L200 · focused edition)',
        'og:title': 'TPU vs GPU · focused edition (L200)',
        'og:description': 'Not a tour of spec sheets. Run the same FlashAttention on both chips '
                          'and watch where they start to diverge — every answer comes with a figure and a measurement.',
        'zh': 'topic-02.html',
    },
    'topic-02-L200-lecture': {
        'title': 'Topic 2 L200 · Instructor notes (lecture script)',
        'og:title': 'TPU vs GPU · L200 instructor notes',
        'og:description': "The teacher's draft: the spoken script, which figure to point at, likely questions, and what not to say.",
        'zh': 'topic-02-L200-lecture.html',
        'css': {'content:"屏幕 · "': 'content:"Screen · "', 'content:"停 · "': 'content:"Pause · "'},
    },
}
HEAD['topic-01'] = {
    'title': 'Topic 1 · The Life of a Token',
    'og:title': 'The Life of a Token',
    'og:description': 'Take DeepSeek V3 apart and follow one token end to end, doing the math at every step. '
                      'Every number on the page updates live with the parameters.',
    'zh': 'topic-01.html',
}
HEAD['topic-01-lecture'] = {
    'title': 'Topic 1 · Instructor notes (lecture script)',
    'og:title': 'The Life of a Token · instructor notes',
    'og:description': "The teacher's draft: the spoken script, which figure to point at, likely questions, and what not to say.",
    'zh': 'topic-01-lecture.html',
    'css': {'content:"屏幕 · "': 'content:"Screen · "', 'content:"停 · "': 'content:"Pause · "'},
}
HEAD['topic-03'] = {
    'title': 'Topic 3 · The Evolution of Attention',
    'og:title': 'The Evolution of Attention · a chain, cut, and looped back',
    'og:description': 'From RNNs to today\'s hybrid ratios, told as one timeline: each step answers only '
                      '"what did the last step leave unpaid, and what does this one pay with?" Many figures, few words, plain language.',
    'zh': 'topic-03.html',
    'fsa': 0.53,
    # 模型表按 data-v 排序；中文公司名在英文页上排出来的顺序看着是乱的，换成英文名（不可见，只影响排序）
    'css': {'data-v="%s|' % k: 'data-v="%s|' % v for k, v in {
        '小米': 'Xiaomi', '智谱': 'Zhipu', '月之暗面': 'Moonshot', '腾讯 混元': 'Tencent Hunyuan',
        '蚂蚁 百灵': 'Ant Ling', '阿里 千问': 'Alibaba Qwen'}.items()},
}
HEAD['topic-03-lecture'] = {
    'title': 'Topic 3 · Instructor notes (lecture script)',
    'og:title': 'The Evolution of Attention · instructor notes',
    'og:description': "The teacher's draft: the spoken script, which figure to point at, likely questions, and what not to say.",
    'zh': 'topic-03-lecture.html',
    'css': {'content:"屏幕 · "': 'content:"Screen · "', 'content:"停 · "': 'content:"Pause · "'},
}
# 有英文版的页面：中文 href → 英文 href（锚点保留）。其余中文页一律去链接。
EN_MAP = {'index.html': 'index.en.html', 'topic-02.html': 'topic-02.en.html',
          'topic-02-L200-lecture.html': 'topic-02-L200-lecture.en.html',
          'topic-01.html': 'topic-01.en.html', 'topic-01-lecture.html': 'topic-01-lecture.en.html',
          'topic-03.html': 'topic-03.en.html', 'topic-03-lecture.html': 'topic-03-lecture.en.html'}
# ⭐ 英文版字体用 Source Sans 3（Google Fonts）：Noto Sans CJK 的拉丁字形就是从它来的，
#    所以图上标签的宽度跟检查时（本机 Noto CJK）一致；又避开了 CJK 字体把弯引号画成全角的问题，
#    而且 iPhone / Mac 上不再回落到更宽的系统字体（作者 10-17 是在 iPhone 上看的）。
FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">')
LATIN = '"Source Sans 3","Noto Sans CJK SC",-apple-system,"Segoe UI",Roboto,sans-serif'

# 只保留本页（L200）和讲义；其余中文页面一律不链过去 —— 英文读者点过去是一屏中文（作者 2026-10-17）
# 只链向「已发布」的英文页（git 已跟踪，或本次正在组装的页）——未上线的专题别出现死链
import subprocess as _sp
try:
    _tracked = set(_sp.run(['git', 'ls-files', 'WebPages'], cwd=pathlib.Path(__file__).resolve().parents[2],
                           capture_output=True, text=True).stdout.split())
    _tracked = {t.split('/')[-1] for t in _tracked}
except Exception:
    _tracked = None
PUBLISH_EXTRA = set()   # assemble 当前页时临时加入
def _live(en):
    return _tracked is None or en in _tracked or en in PUBLISH_EXTRA
KEEP = tuple(EN_MAP.values())
def english_only_links(s):
    s = re.sub(r'href="([\w.-]+\.html)(#[^"]*)?"', lambda m: 'href="%s%s"' % (EN_MAP[m.group(1)] if m.group(1) in EN_MAP and _live(EN_MAP[m.group(1)]) else m.group(1), m.group(2) or ''), s)
    # 讲义顶栏的「完整版 L300」标签整颗删（不是正文里的引用，是个跳转按钮）
    s = re.sub(r'(<nav class="tabs">.*?)\s*<a href="topic-\d+-L300\.html">.*?</a>(.*?</nav>)', r'\1\2', s, flags=re.S)
    # 页首的跳转按钮（.lecbtn）指向没有英文版的页面（附录等）：整颗删，不留一段裸文字
    s = re.sub(r'\s*<a class="lecbtn" href="(?![^"]*\.en\.html)[^"]*">.*?</a>', '', s, flags=re.S)
    # 版本切换条：只留当前这一版
    s = re.sub(r'(<nav class="famnav"[^>]*>)(.*?)(</nav>)',
               lambda m: m.group(1) + ''.join(re.findall(r'<span class="famlab">.*?</span>|<span class="here".*?</span>', m.group(2), flags=re.S)) + m.group(3),
               s, flags=re.S)
    # 页首「完整版 L300 →」这类整颗 chip 就是个跳转按钮，去掉链接后只剩一句指向不存在页面的话 —— 整颗删
    s = re.sub(r'<span class="chip">[^<]*<a href="topic-[^"]*?(?<!\.en)\.html">.*?</a></span>', '', s, flags=re.S)
    def unlink(m):
        href, inner = m.group(1), m.group(2)
        # GitHub 上指向中文 md / 中文目录的外链也算中文页（文件名含中文或 URL 编码的中文）
        zh_ext = href.startswith('http') and ('github.com' in href) and (re.search(r'[\u4e00-\u9fff]|%E[4-9]', href) is not None)
        if (href.startswith(('http', '#', 'mailto', 'data:')) and not zh_ext) or href.split('#')[0] in KEEP:
            return m.group(0)
        # 按钮样式的链接（inline-block）指向没有英文版的页：整颗删，留下裸文字反而像坏掉的按钮
        if 'display:inline-block' in m.group(0)[:m.group(0).find('>')]:
            return ''
        inner = re.sub(r'\s*\(Chinese\)', '', inner)
        return '<span>' + inner + '</span>'
    s = re.sub(r'<a\b[^>]*?href="([^"]+)"[^>]*>(.*?)</a>', unlink, s, flags=re.S)
    s = re.sub(r'\s*\(Chinese\)(?=\s*(?:[,.;:)]|</))', '', s)
    return s

def main(topic, d):
    d = pathlib.Path(d)
    PUBLISH_EXTRA.add(topic + '.en.html')
    for extra in sys.argv[3:]: PUBLISH_EXTRA.add(extra)   # 同批一起上线的页，如 topic-01-lecture.en.html
    s = (d / 'skeleton.html').read_text()
    h = HEAD[topic]
    s = s.replace('lang="zh-CN"', 'lang="en"')
    s = re.sub(r'<title>.*?</title>', f'<title>{h["title"]}</title>', s, flags=re.S)
    s = re.sub(r'(<meta property="og:title" content=")[^"]*', r'\g<1>' + h['og:title'], s)
    s = re.sub(r'(<meta property="og:description" content=")[^"]*', r'\g<1>' + h['og:description'], s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)          # CSS 注释全是中文开发备忘
    s = s.replace('"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",-apple-system,"Segoe UI",Roboto,sans-serif', LATIN)
    # 讲义页的字体栈写法不同：无衬线 / 衬线（台词用衬线）都换成拉丁优先
    s = re.sub(r'"Noto Sans CJK SC","PingFang SC"(,"Microsoft YaHei")?(,-apple-system)?,sans-serif', LATIN, s)
    s = s.replace('"Noto Serif CJK SC","Source Han Serif SC","Songti SC","SimSun",Georgia,serif',
                  'Georgia,"Source Serif 4","Times New Roman","Noto Serif CJK SC",serif')
    # 字体内嵌（拉丁子集、可变字重，约 120 KB）：不依赖外网，检查渲染和读者看到的是同一套字形
    ff = (pathlib.Path(__file__).resolve().parents[1] / 'fonts' / 'ss3-embedded.css').read_text()
    s = s.replace('</title>', '</title><style>' + ff + '</style>', 1)
    # ⭐ 「折叠讲解」开关（移植自专题三/四/五/六的 figbar）。英文版**默认折叠**：只留图，
    #    想细读的人点「Show narration」或按 T 展开（作者 2026-10-17）。中文专题二原本没有这个开关。
    # 中文页自带折叠开关（专题三起）就不再注入，开关文案与默认值在块里翻译/设定
    if not topic.endswith('lecture') and 'figonly-css' not in s:
      fo_css, fo_html = (pathlib.Path(__file__).resolve().parents[1] / 'figonly.html').read_text().split('@@SPLIT@@')
      s = s.replace('</title>', '</title>' + fo_css.strip(), 1)
      s = s.replace('</body>', fo_html + '</body>', 1)
    # ⭐ 英文小字号整体 +1px（作者 10-17：表格、图注等英文字看着偏小 —— 同字号下拉丁字母的
    #    x-height 比汉字小一截）。只动小字那几类，不动正文 16px 与版心。
    if not topic.endswith('lecture'):
        # ⭐ 英文小字号 +1px（作者 10-17：表格、图注等英文字看着偏小）。基准从本页 CSS 里读（各专题不同：
        #    专题二图注 15px、专题一 13.5px），不写死 —— 写死会让专题一的图注一下子大 2.5px。
        css = ''.join(re.findall(r'<style\b[^>]*>(.*?)</style>', s, flags=re.S))
        def base(sel, default):
            v = None
            for m in re.finditer(r'(?:^|[}\s])' + re.escape(sel) + r'\s*\{([^}]*)\}', css):
                mm = re.search(r'font-size:\s*([\d.]+)px', m.group(1)) or re.search(r'font:\s*(?:\d{3}\s+)?([\d.]+)px', m.group(1))
                if mm: v = float(mm.group(1))
            return v if v is not None else default
        rules = ''.join('%s{font-size:%gpx}' % (sel, base(sel, d) + 1) for sel, d in
                        (('table', 14.5), ('th', 12), ('.tbl-note', 13), ('.chip', 13), ('figcaption', 15)))
        # 英文表头大写加字距、不许换行，手机上会把关键列挤出屏幕（专题一并排审 10-17）；进度条标签不折行
        # ⭐ 等宽字体：中文页里这些类装的是汉字（落回无衬线），英文整串按等宽画，看着像代码、也更宽
        #    （作者 10-17：版式要一眼看去跟中文一样）。只改「装文字」的类；数字、公式、编号类保持等宽。
        TEXT_MONO = ['.svgsm', '.svgtxt', '.tocbar a', '.tocbar-lb', '.console .tag', '.hero .en', '.jstep span', '.stn .badge']
        present = [c for c in TEXT_MONO if re.search(r'(^|[},\s])' + re.escape(c) + r'\s*[,{][^}]*?var\(--mono\)', css)]
        # 只对已按新字体逐图复核过的专题启用（HEAD 里带 fsa 的；专题一二待复核，暂保持原样）
        if present and h.get('fsa'):
            rules += ','.join(present) + '{font-family:' + LATIN.replace("'", '"') + '}'
            # 同 px 下拉丁字比汉字小约两成；Source Sans 3 的 x-height 比 0.486，调到 0.53 ≈ 放大 9%，
            # 对内联 style 定过字号的标签同样生效（按等宽字体留的宽度足够装下）
            if h.get('fsa'): rules += '.svgsm,.svgtxt,.svglbl{font-size-adjust:%s}' % h['fsa']
        rules += '.figsrc>summary{font-family:'+LATIN.replace("'",'"')+'}'   # 出处折叠标题：中文落回无衬线，英文别显示成等宽
        rules += '.jstep span{white-space:nowrap}@media(max-width:600px){th{white-space:normal}.jstep span{font-size:9px}}'
        s = s.replace('</body>', '<style id="en-size">' + rules + '.sub{font-size:.95em}</style></body>', 1)
    missing = []
    def put(m):
        f = d / f'{m.group(1)}.en.html'
        if not f.exists(): missing.append(f.name); return (d / f'{m.group(1)}.zh.html').read_text()
        return f.read_text()
    s = re.sub(r'@@CHUNK:(\d+)@@', put, s)
    # 中文版 2026-10-17 修掉的两个构建 bug，切块时还没修 —— 英文这边同样处理
    s = re.sub(r'(<b class="fno">[^<]*</b>)\s*(?:&nbsp;|\u3000|\s)*<b class="fno">[^<]*</b>', r'\1', s)
    # 注释里出现过字面 <script>，切块时躲过了去注释那一步；带中文的注释一律去掉
    s = re.sub(r'<!--(?:(?!-->).)*?[\u4e00-\u9fff](?:(?!-->).)*?-->', '', s, flags=re.S)
    # ⛔ 只在 <style>/<script> 之外做 **→<b>：SVG 内联 <style> 的注释里也有 **，
    #    转成 <b> 会把解析器踢出 SVG，整张图变成一屏 CSS 源码（2026-10-17 作者手机上撞见）
    parts = re.split(r'(<(?:style|script)\b.*?</(?:style|script)>)', s, flags=re.S)
    for i, p in enumerate(parts):
        if p.startswith('<style'):
            parts[i] = re.sub(r'/\*.*?\*/', '', p, flags=re.S)   # CSS 注释是开发备忘
        elif not p.startswith('<script'):
            parts[i] = re.sub(r'\*\*([^*<]{1,80})\*\*', r'<b>\1</b>', p)
    s = ''.join(parts)
    s = english_only_links(s)
    # 中英互跳：课件页首讲义按钮旁 / 讲义顶栏末尾放「中文」
    zh = h['zh']
    s = re.sub(r'(<a class="lecbtn" href="topic-\d+(?:-L200)?-lecture[^"]*">.*?</a>)',
               r'\1\n    <a class="lecbtn" href="%s" lang="zh-CN" hreflang="zh-CN">🌐 中文</a>' % zh, s, count=1, flags=re.S)
    s = re.sub(r'(<nav class="tabs">.*?)(</nav>)', r'\1  <a href="%s" lang="zh-CN" hreflang="zh-CN">🌐 中文</a>\n  \2' % zh, s, count=1, flags=re.S)
    if '🌐 中文' not in s:   # 没有讲义按钮可挂时，挂在面包屑末尾
        s = re.sub(r'(<div class="crumb">.*?)(</div>)', r'\1\n    <a class="lecbtn" href="%s" lang="zh-CN" hreflang="zh-CN">🌐 中文</a>\2' % zh, s, count=1, flags=re.S)
    for a_, b_ in h.get('css', {}).items(): s = s.replace(a_, b_)
    # 讲义里「Section N」的页内链接按标题核对锚点（中文专题三讲义的 #cK 错了一位：#c2 写的是第二章，实为第一章）
    sec_id = {m.group(2): m.group(1) for m in re.finditer(r'id="(c\d+)"[^>]*>\s*Section (\d+)\b', s)}
    s = re.sub(r'href="#(c\d+)">Section (\d+)\b', lambda m: 'href="#%s">Section %s' % (sec_id.get(m.group(2), m.group(1)), m.group(2)), s)
    # 图里的 class 样式和内联 style 也把 Source Sans 3 放到 Noto CJK 前面
    s = s.replace("'Noto Sans CJK SC'", "'Source Sans 3','Noto Sans CJK SC'")
    s = re.sub(r'(?<![\w\'])"Noto Sans CJK SC"(?!,"PingFang)', '"Source Sans 3","Noto Sans CJK SC"', s)
    s = s.replace('"Source Sans 3","Source Sans 3"', '"Source Sans 3"')
    import os
    out = pathlib.Path(os.environ['I18N_OUT']) if os.environ.get('I18N_OUT') else pathlib.Path(__file__).resolve().parents[2] / 'WebPages' / f'{topic}.en.html'
    out.write_text(s)
    print('wrote', out, len(s), 'missing:', missing or 'none')
    left = re.findall(r'[一-鿿]', re.sub(r'<(style|script)\b.*?</\1>', '', s, flags=re.S))
    print('residual CJK chars:', len(left))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
