# -*- coding: utf-8 -*-
"""可读性闸门：量「这段放到屏幕上什么样」。

    python3 topic02-lint-readability.py [页面.html ...]

**为什么要有这个脚本。** 在它之前，这套课件的所有自检都是**正确性**闸门 ——
禁字表、查重、算术核对、图注与图内数字对齐。它们能保证每个字都是对的，
**但一个指标都没有在管「读起来累不累」**。于是出现了这种东西：
一段 380 字的图前预告，逐条讲图上有什么，而图注里那五条一条不落地全有 ——
每个字都对、查重也过（两处是同义改写，不是逐字重复），
放到屏幕上就是让人先读一遍、看图再读一遍。

四项检查，都是结构量，不需要判断内容好坏：

W1 段落墙        一个 <p> 超过 180 汉字。屏幕上就是一堵墙，眼睛没有落点。
W2 加粗过载      段长 ≥40 字且加粗覆盖 ≥55%。**全加粗等于没加粗** ——
                 眼睛找不到重点，只看到黑白噪点。短句整句加粗是合法的
                 （那是「这一句就是论点」），所以卡了 40 字下限。
W3 图前预告过长  紧挨 <figure> 前面那个 <p> 超过 80 汉字。
                 图前正文的职责是「带着什么去看图」，不是替图把内容讲一遍。
W4 图前复述图注  图前 <p> 与该图 figcaption 共享 ≥2 个 5-gram。
                 抓的正是上面那种同义改写式复述 —— 逐字查重看不见它。

⚠️ 报告为主，不 assert 中止。这几项是**版面质量**不是**事实正确性**，
   卡死构建会逼人去绕过它，而绕过一次这道闸门就不存在了。
   要看的是最后那行总分：全文加粗占比。经验值 —— 30% 以下读着清爽，
   40% 以上就是满屏加粗。
"""
import io, os, re, sys, html as H

CJK = re.compile(r'[一-鿿]')
W1_WALL      = 180   # 汉字
W2_BOLD_MIN  = 40    # 汉字：短于此的段不查加粗占比
W2_BOLD_PCT  = 0.55
W3_PREFIG    = 80    # 汉字
W4_NGRAM     = 5
W4_HITS      = 2


def text(x):
    return H.unescape(re.sub(r'<[^>]+>', '', x))


def cjk(x):
    return len(CJK.findall(x))


def grams(s, n):
    """只取**纯汉字**的 n-gram。

    ⛔ 第一版没这个限制，于是 <p> 和图注共享一个 `FLOPs` / `kernel` /
       `all-reduce` 就算一次「复述」——&nbsp;可那是同一个话题下必然共享的术语，
       不是复述。结果 topic-01 有一条报「17 处复述」，全是英文词的子串。
       **一个总在误报的闸门等于没有闸门**，所以卡死：连续 5 个汉字才算证据。
    """
    s = re.sub(r'[^一-鿿]', ' ', s)
    return {g for seg in s.split() for g in
            (seg[i:i + n] for i in range(len(seg) - n + 1))}


def merge(gs):
    """把互相重叠的 n-gram 并成最长片段，长的在前。

    两个 gram 若首尾错一位就能接上（a[1:] == b[:-1]），说明它们本来是同一句话
    被切开的。反复接到接不动为止，剩下的每一条就是一段真正独立的共享文字。
    """
    out = sorted(gs)
    changed = True
    while changed:
        changed = False
        for a in list(out):
            for b in list(out):
                if a is b or a not in out or b not in out:
                    continue
                if a.endswith(b[:-1]) and len(b) > 1:          # a 后接 b
                    out.remove(a); out.remove(b); out.append(a + b[-1])
                    changed = True
                    break
            if changed:
                break
    # 被别的片段整个包住的丢掉
    out = [x for x in out if not any(x != y and x in y for y in out)]
    return sorted(out, key=len, reverse=True)


def scan(path):
    raw = io.open(path, encoding='utf-8').read()
    # 注释与图内文字都不算正文 —— 注释不渲染，图里的字有自己的版面规则。
    doc = re.sub(r'<!--.*?-->', '', raw, flags=re.S)
    doc = re.sub(r'<svg.*?</svg>', '<svg/>', doc, flags=re.S)
    # <details> 里的内容默认不在屏幕上 —— 折叠本来就是这四项的**合法解法之一**
    # （「精简，或者折叠起来」）。把它算进来，等于修完了还在报警，
    # 而一个改对了也不消失的告警会让人整个不看这份报告。
    doc = re.sub(r'<details\b.*?</details>', '', doc, flags=re.S)

    hits, tot, bold = [], 0, 0

    for m in re.finditer(r'<p\b[^>]*>(.*?)</p>', doc, re.S):
        inner = m.group(1)
        t = text(inner)
        n = cjk(t)
        if n < 20:
            continue
        b = sum(cjk(text(x)) for x in re.findall(r'<b\b[^>]*>(.*?)</b>', inner, re.S))
        tot += n
        bold += b
        if n >= W1_WALL:
            hits.append(('W1', n, '段落墙 %d 字' % n, t[:46]))
        if n >= W2_BOLD_MIN and b / n >= W2_BOLD_PCT:
            hits.append(('W2', n, '加粗 %.0f%%（%d/%d 字）' % (100 * b / n, b, n), t[:46]))

    # W3 / W4：只看紧挨 <figure> 前面那一个 <p>，中间不能隔别的块级内容。
    for fm in re.finditer(r'<figure\b.*?</figure>', doc, re.S):
        before = doc[:fm.start()].rstrip()
        pm = re.search(r'<p\b[^>]*>((?:(?!</p>).)*)</p>\s*$', before, re.S)
        if not pm:
            continue
        pt = text(pm.group(1))
        n = cjk(pt)
        if n < 20:
            continue
        cap = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', fm.group(0), re.S)
        fid = re.search(r'id="([^"]+)"', fm.group(0))
        tag = fid.group(1) if fid else '?'
        if n >= W3_PREFIG:
            hits.append(('W3', n, '图前预告 %d 字（图 %s）' % (n, tag), pt[:46]))
        if cap:
            # ⛔ 第二版直接数共享 5-gram 的个数，于是**一个**共享短语会被拆成
            #    若干个重叠 gram 反复计数 —— 「重点在第三层」这 6 个字就报「2 处」。
            #    而它恰恰是**该有的**指路，不是复述。数量级一失真，
            #    真正的整段复述（8 处）就淹没在一堆 2 处里。
            #    所以先把重叠的 gram 并回**最长共享片段**，再按片段数计。
            same = merge(grams(pt, W4_NGRAM) & grams(text(cap.group(1)), W4_NGRAM))
            if len(same) >= W4_HITS:
                hits.append(('W4', len(same), '图前复述图注 %d 段（图 %s）：%s'
                             % (len(same), tag, '、'.join(same[:3])), pt[:46]))

    return hits, tot, bold


def main(paths):
    bad = 0
    for p in paths:
        if not os.path.exists(p):
            print('跳过（不存在）%s' % p)
            continue
        hits, tot, bold = scan(p)
        hits.sort(key=lambda x: (x[0], -x[1]))
        print('\n══ %s  正文 %d 汉字' % (os.path.basename(p), tot))
        for k, _, why, sample in hits:
            print('   [%s] %-38s %s…' % (k, why, sample))
        pct = 100 * bold / tot if tot else 0
        flag = '✅' if pct <= 30 else ('⚠️ ' if pct <= 38 else '❌')
        print('   ── %d 处待办　全文加粗占比 %s %.0f%%（目标 ≤30%%）'
              % (len(hits), flag, pct))
        bad += len(hits)
    print('\n合计 %d 处。这是版面质量报告，不中止构建。' % bad)


if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))
    W = os.path.join(HERE, '..', 'WebPages')
    main(sys.argv[1:] or [os.path.join(W, f) for f in
                          ('topic-02.html', 'topic-02-L200.html', 'topic-01.html')])
