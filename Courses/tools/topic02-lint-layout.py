# -*- coding: utf-8 -*-
"""版面体检（渲染后量，不看源码）—— 三条判据，每条都对应一次真栽过的跟头。

════════════════════════════════════════════════════════════════════
⭐ 为什么必须**渲染后量**，源码扫不出来
════════════════════════════════════════════════════════════════════
① **跑到视口左边外面**（2026-09-04）
   `figure.fbox{margin:30px 0}` 这条简写把 margin-left 也设成了 0，
   而它的特异度（元素＋类 = 0,1,1）压过 `.fwide`（0,1,0）——
   于是宽图只剩 `translateX(-50%)` 生效，整张图往左平移 720px，
   在 1900px 宽的窗口里左边缘落在 **x = −286**，左边一截被裁掉。
   ⛔ 阴险处：**左溢出既不产生滚动条，也不让 scrollWidth 变大** ——
      原来那个「scrollWidth > clientWidth」的探针查它恒为 0 处。
      只有量 `getBoundingClientRect().x` 才看得见。

② **横向溢出**（2026-09-03）
   一处 `<code>` 用 `</b>` 闭合，`white-space:nowrap` 顺着漏出去，
   整页文字都不换行，页面被撑到 2986px。这条留着，跟 ① 是一对：
   一个查往右撑破，一个查往左跑掉。

④ **图里的字比正文还大**（2026-09-08）
   现场原话：「这个字它大到跟老年机一样，这不好吧？」
   真凶不是缩放，是 `class="None"` ——&nbsp;有 61 处按位置传了 `None` 当类名，
   拼出来就是字面量 `None`。它在 CSS 里没有对应规则，于是浏览器按
   **SVG 文字的缺省 16px** 画，再乘宽图 1.22 倍 = **21px**，比正文 18px 还大。
   ⛔⛔ 阴险处：**拼错的类名不报错、不难看，只是「变成另一种样子」** ——&nbsp;
      没有任何一层会抱怨，源码扫也扫不出（`class="None"` 语法完全合法）。
   ⭐ 判据用外部锚点：**图里的字不该比正文大**。
      不查类名对不对（拼错法有无穷种），查它渲染出来的结果越不越界。

③ **图里的字撞车 / 顶出画布**
   SVG 的 `<text>` **不会自动换行**，所以任何「把图里的字调大」的改动，
   风险都只有一种：压到隔壁的字上，或者顶出 viewBox。
   38 张图人眼一张张看不现实，用 `getBBox()` 直接量。
   ⚠️ 判据量的是 **em 盒**，不是浏览器给的整框（整框含 ascent+descent，
      约 1.39 em，正常行距也会重叠几个 px）。em 盒重叠 > 0.5px 即算撞车。
   ⛔ 2026-09-14 之前用的是「重叠超过各自高度的一半」——&nbsp;那个比例阈值
      **对中文天然过宽**（汉字的墨几乎填满 em 盒），漏掉了「两行糊在一起」
      整类。fig3-knobs 三处肉眼可见的压字，它报的是 0。

用法：
    python3 topic02-lint-layout.py [页面…]
默认量专题二的三个页面。**只报告，不中止构建**（跟可读性体检一致）。
"""
import os
import sys

VIEW_W = 1900          # 故意用宽窗口：①那类 bug 只在宽屏暴露
VIEW_H = 1200

JS_GEO = r"""()=>{
  const name=e=>e.tagName+'.'+(typeof e.className==='string'?e.className:
                               (e.getAttribute('class')||'')).slice(0,30);
  const bad=[];
  document.querySelectorAll('body *').forEach(e=>{
    const r=e.getBoundingClientRect();
    if(!r.width||!r.height) return;
    if(r.x < -2) bad.push(['左跑', Math.round(r.x), name(e)]);
  });
  // ⚠️ 只算**真的会裁或真的会出滚动条**的：overflow-x 不是 visible 的容器。
  //    overflow:visible 的元素 scrollWidth 超出是家常便饭（宽图突破版心就会），
  //    照报的话满屏假阳性，真问题反而被淹掉 —— 这跟「一个失效把另一个静音」是同一类。
  const ov=[...document.querySelectorAll('body *')].filter(e=>{
    if(e.scrollWidth<=e.clientWidth+2||e.clientWidth<=0) return false;
    return getComputedStyle(e).overflowX!=='visible';});
  return {geo: bad.slice(0,12), nGeo: bad.length,
          over: ov.slice(0,8).map(e=>['右撑', e.scrollWidth-e.clientWidth, name(e)]),
          nOver: ov.length,
          pageW: document.documentElement.scrollWidth,
          viewW: innerWidth};}"""

JS_FIG = r"""()=>{
  const out=[];
  document.querySelectorAll('figure').forEach((f,i)=>{
    const svg=f.querySelector('svg'); if(!svg) return;
    const vb=(svg.getAttribute('viewBox')||'0 0 0 0').split(' ').map(Number);
    const W=vb[2], H=vb[3], bb=[];
    // ⚠️ 用 getBoundingClientRect 换算回 viewBox 坐标，**不要用 getBBox**。
    //    getBBox 给的是**变换之前**的框：旋转 90° 的纵轴标题会报出 x = −21，
    //    看着像「顶出画布」，其实 transform 早把它放回去了。
    //    2026-09-04 第一版就是这么误报了专题一那张对数图。
    const R=svg.getBoundingClientRect(), k=W/R.width;
    // ⭐⭐ 2026-09-14 改判据：量的是 **em 盒**，不是 getBoundingClientRect 的整框。
    //   浏览器给的框是字体的 ascent+descent（Noto Sans CJK 约 1.39 em），
    //   比真正有墨的地方高出一大截 ——&nbsp;两行**正常行距**也会有几个 px 的框重叠。
    //   原先为此设了「重叠超过一半才算」，方向对，代价是把
    //   「**两行离得太近、糊成一团**」整类漏光了（见下面 hits 的判据说明）。
    for(const t of svg.querySelectorAll('text')){
      const b=t.getBoundingClientRect();
      if(!b.width) continue;
      const fs=parseFloat(getComputedStyle(t).fontSize)*k;   // viewBox 单位
      const h=b.height*k, pad=Math.max(0,(h-fs)/2);          // 上下各收掉虚高
      bb.push({x:(b.x-R.x)*k, y:(b.y-R.y)*k, w:b.width*k, h:h,
               ey:(b.y-R.y)*k+pad, eh:Math.min(h,fs),
               s:(t.textContent||'').slice(0,24)});
    }
    const oob=bb.filter(b=>b.x<-2||b.x+b.w>W+2||b.y<-2||b.y+b.h>H+2).map(b=>b.s);
    // ⛔⛔ 2026-09-14 现场一眼看出 fig3-knobs 三处压字，**这条 lint 报的是 0**。
    //   旧判据 `vy < min(h)*0.5 → 跳过`，而那三处 frac 只有 0.36。
    //   ⭐⭐ 根因：**用「占各自高度的比例」当阈值，对中文天然过宽。**
    //     汉字的墨几乎填满 em 盒，所以「重叠 36%」不是擦边，是实打实压上去了；
    //     拉丁文那点 x-height 才撑得起 50% 的容差。
    //   ⭐ 换成 em 盒 ＋ 绝对容差 0.5px 之后，七个页面 260 张图只命中 5 处，
    //     没有一处是误报 ——&nbsp;**噪音没涨，漏检那一整类补上了。**
    const hits=[]; let n=0;
    for(let a=0;a<bb.length;a++) for(let c=a+1;c<bb.length;c++){
      const p=bb[a],q=bb[c];
      const vy=Math.min(p.ey+p.eh,q.ey+q.eh)-Math.max(p.ey,q.ey);
      if(vy <= 0.5) continue;
      if(Math.min(p.x+p.w,q.x+q.w)-Math.max(p.x,q.x) > 3){
        n++; if(hits.length<3)
          hits.push(Math.round(vy)+'px  '+p.s+'  ⟂  '+q.s);}
    }
    // ④ 字号越界：拿正文字号当外部锚点。
    // ⛔ 判的是**这张图最常见的那个字号**（＝它的正文档），不是最大值 ——&nbsp;
    //   标题、大号数字本来就该比正文大，按最大值判会把 8 张正常的图一起报出来，
    //   真问题反而被淹掉（跟这个文件里 ② 那条「假阳性淹掉真问题」是同一课）。
    const body=parseFloat(getComputedStyle(document.body).fontSize)||16;
    const hist={}, samp={};
    for(const t of svg.querySelectorAll('text')){
      const s0=(t.textContent||'').trim(); if(!s0) continue;
      // ⛔ 只乘一次 R.width/W。第一版写成 `/k*(R.width/W)`，而 k 就是 W/R.width，
      //   等于把缩放**平方**了一遍 —— 12px 报成 18px，看着像真有问题。
      //   ⭐ 同一个比例在同一行出现两次，就该停下来问哪个是多余的。
      const fs=Math.round(parseFloat(getComputedStyle(t).fontSize)*(R.width/W));
      // ⛔ 2026-09-13：原来按**元素个数**计数 —— 于是「8 个两字标题」压过
      //   「5 行一百多字的落点带」，众数变成标题字号，图被误报。
      // ⭐ 这条规矩自己写的是「这张图的正文档」，那就该按**字数**权重：
      //   正文档的定义是「大部分字在哪个号上」，不是「大部分文本框在哪个号上」。
      hist[fs]=(hist[fs]||0)+s0.length; if(!samp[fs]) samp[fs]=s0.slice(0,16);
    }
    // ⭐⭐ 2026-09-13 放宽阈值，并把理由写下来（否则下一个人会把它调回去）：
    //   这条规矩定的时候，图**基本上是文字面板** ——&nbsp;那时「图内字号超过正文」
    //   确实等于图在抢正文的层级。
    //   ⛔ 但现在图的定位变了：现场原话「图的目的是把原理画出来，
    //     一目了然，打到屏幕上去分享」——&nbsp;**投屏要缩到 0.6～0.7**，
    //     图里的主标签本来就该比正文大一档，否则后排看不清。
    //   ⭐ 所以现在判的是「有没有大到离谱」，不是「有没有超过正文」：
    //     阈值 = 正文 + 4px。**太小那一头由投屏体检（topic02-lint-projection）管。**
    // ⚠️ 2026-09-13 再放宽一档，并把**为什么不是拍脑袋**写下来：
    //   这里量的是**渲染后**的字号，而 .fwide 容器最宽到 1760px ——
    //   图的 viewBox 是 1400，所以在宽屏上**整张图会被放大约 1.26 倍**。
    //   于是「源码里 18px」渲染出来就是 22px。
    //   ⭐ 判据：**阈值要定在你真正想禁止的那件事上。**
    //     想禁的是「图的正文档比页面正文大一大截」，不是「大一点」；
    //     body+6 对应源码约 18.6px ——&nbsp;再大就该问问是不是整张图该拆了。
    const CEIL = body + 6;
    const mode=Object.entries(hist).sort((a,b)=>b[1]-a[1])[0];
    const big=[];
    if(mode && +mode[0]>CEIL)
      big.push('正文档 '+mode[0]+'px ＞ 上限 '+CEIL+'px（正文 '+body+'＋4）（'+mode[1]+
               ' 处，如「'+samp[mode[0]]+'」）');
    if(n||oob.length||big.length) out.push({i, id:svg.getAttribute('data-fig')||f.id||'',
      W,H, collide:n, hits, oob:oob.slice(0,3), noob:oob.length,
      big:big.slice(0,3), nbig:big.length});
  });
  return out;}"""


def main(paths):
    from playwright.sync_api import sync_playwright
    bad = 0
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={'width': VIEW_W, 'height': VIEW_H})
        for p in paths:
            if not os.path.exists(p):
                print('跳过（不存在）%s' % p)
                continue
            pg.goto('file://' + os.path.abspath(p))
            pg.wait_for_timeout(1200)
            g = pg.evaluate(JS_GEO)
            figs = pg.evaluate(JS_FIG)
            print('\n══ %s  页宽 %d（视口 %d）'
                  % (os.path.basename(p), g['pageW'], VIEW_W))
            if g['pageW'] > g['viewW'] + 2:
                print('   ⛔ 整页比视口宽 %d px —— 会出横向滚动条'
                      % (g['pageW'] - g['viewW']))
                bad += 1
            if g['nGeo']:
                print('   ⛔⛔ %d 个元素跑到视口左边外面 —— 左边会被裁掉，'
                      '而且不产生滚动条' % g['nGeo'])
                for k, v, w in g['geo']:
                    print('        %s x=%-6d %s' % (k, v, w))
                bad += g['nGeo']
            if g['nOver']:
                print('   ⛔ %d 个元素横向撑破' % g['nOver'])
                for k, v, w in g['over']:
                    print('        %s +%-5d %s' % (k, v, w))
                bad += g['nOver']
            for r in figs:
                print('   ⚠️  #%-2d %-11s 撞车 %d／顶出 %d／字号越界 %d'
                      % (r['i'], r['id'], r['collide'], r['noob'], r['nbig']))
                for h in r['hits']:
                    print('           ⟂ ' + h)
                for o in r['oob']:
                    print('           ↗ ' + o)
                for z in r['big']:
                    print('           🔠 ' + z)
                bad += r['collide'] + r['noob'] + r['nbig']
            if not g['nGeo'] and not g['nOver'] and not figs:
                print('   ✅ 无左跑、无右撑、图内文字无撞车、字号没越过正文')
        b.close()
    print('\n版面体检合计 %d 处。只报告，不中止构建。' % bad)


if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))
    W = os.path.join(HERE, '..', 'WebPages')
    # ⛔ 2026-09-07：这张清单原先漏了 topic-03，而专题三那张编年史正是**撞了三轮
    #    才排开**的图（说明文字互糊 → 相邻年份撞 → 交错两行仍撞）。
    # ⭐ 教训：**漏检的那一页，恰恰是最需要检的那一页** —— 新页面天然图最少、
    #    最容易被忘进清单，而它的图又是最新画的、最没被人眼扫过的。
    #    新建一个 topic-NN 页面时，第一件事就是把它加进这一行。
    # ⭐ 2026-09-25：页面清单改成从目录现取（原手写清单漏登记不报错，新专题会静默逃过体检）。
    #   只收课件页：topic-NN[x].html 与 topic-NN[x]-L200／L300.html。
    import re as _re_pages
    main(sys.argv[1:] or [os.path.join(W, f) for f in sorted(os.listdir(W))
                          if _re_pages.match(r'topic-\d+[a-z]?(-L\d00)?\.html$', f)])
