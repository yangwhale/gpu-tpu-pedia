# -*- coding: utf-8 -*-
"""专题六两档课件（L300 完整版 topic06-build.py、L200 主线版 topic06-build-L200.py）共用的拼装后处理。

R2（2026-10-19）从 topic06-build.py 原样搬出来，两档共用一份，免得改一边忘一边。
⭐ strict=True（L300）：复用图改字、一步一停的视频，页面上缺一个就挂。
   strict=False（L200）：只处理页面上真有的那几张 —— L200 只挑了一部分图。
"""
import io
import json
import os
import re

import topic03_page as P

HERE = os.path.dirname(os.path.abspath(__file__))

STYLE = """
<style>
.animgrid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; margin:14px 0;
             width:min(1408px, 94vw); position:relative; left:50%; transform:translateX(-50%) }
.animgrid figure { margin:0; min-width:0 }
.animcell figcaption { font-size:14px; color:var(--gray); margin-top:6px; line-height:1.6 }
.animgrid video { width:100%; border-radius:8px; display:block }
/* 2026-10-20：动画视频原生只有 960 宽，fwide 的 figure 却有 1760 宽；video 默认行内、
   没人给它宽度，于是全贴在左边。这里让它居中、放到 1240（跟 fbox 投影宽一致），
   图注也收到同一个 1240 框里，左边缘跟视频对齐。 */
figure.fwide > video { display:block; width:100%; max-width:1240px; margin:0 auto; border-radius:8px }
figure.fwide:has(> video) > figcaption { max-width:1240px; margin-left:auto; margin-right:auto }
@media (max-width:900px) { .animgrid { grid-template-columns:1fr } }
.lecbtn { display:inline-block; margin-left:14px; font-size:13px; padding:4px 12px; border-radius:14px;
          background:#fff; border:1px solid var(--line); color:var(--blue); text-decoration:none }
.lecbtn:hover { background:var(--blue-l); border-color:var(--blue) }
/* R40（2026-10-15）：出处台账里有一串串不带空格的仓库路径和 URL，表格 min-content 被撑到
   1112 px，比 1080 的版心还宽、390 宽手机上把整页 scrollWidth 顶到 1136。只在这一节放开断行。 */
#s十六 td { overflow-wrap:anywhere }
/* R81（2026-10-15）：76 张图全按 viewBox 1400 画，最小字号 12–14 px。390 宽手机上
   图被缩到 308 px（缩放 0.22），字只剩 2.6–3.1 px，截图确认完全读不出。
   窄屏下改成：图保持 1050 px 宽，在 figure 里横向滑动（.fwide 本来就是 overflow-x:auto）。
   1050 = 1400 × 9 ÷ 12 —— 让最小的 12 px 字显示成 9 px；13.5 px 的字显示成 10.1 px。
   同时把 figure 两侧 24 px 内边距收到 10 px，多给图留一点可视宽度。
   ⛔ 只在 max-width:1100px（含平板） 生效，桌面／投影版式一个像素都不变；只选 `> svg`，不碰 video。 */
@media (max-width:1100px) {
  figure.fwide { padding-left:10px; padding-right:10px }
  figure.fwide > svg { width:1050px; max-width:none }
}
</style>"""


STEP_JS = """<style>
.stepbar{display:flex;gap:12px;align-items:center;margin:6px 0 2px;font-size:14px;color:#5f6368;flex-wrap:wrap}
.stepbar button{font:inherit;padding:3px 14px;border-radius:14px;border:1px solid #1a73e8;background:#1a73e8;color:#fff;cursor:pointer}
.stepbar button.gh{background:#fff;color:#1a73e8}
.stepbar button:disabled{background:#fff;color:#9aa0a6;border-color:#dadce0;cursor:default}
.stepbar label{cursor:pointer}
</style>
<script>
// ⭐ 2026-09-25 现场第二轮：「手工播放的时候，我还没看呢，刚翻过去，第一步就跳完了……
//   手工模式第一步也得是让我准备好了，自己点按钮，再给我预备一个 reset 按钮。」
//   所以手工模式下：一上来停在第 0 帧等「开始」；每一步落地停住等「下一步」；
//   播完一圈绕回开头时也停在第 0 帧，不自己再跑一遍。勾「连续播放」才是原来的无缝循环。
// ⭐ 2026-10-14：「把所有的连续播放的 Checkbox 都点上，让它是默认连续播放的样子。」
//   所以默认勾上、一上来就无缝循环；取消勾选才进手工一步一步模式。
document.querySelectorAll('video[data-pauses]').forEach(function(v){
  var P=v.dataset.pauses.split(',').map(Number), i=0, last=0, auto=true;
  var bar=document.createElement('div'); bar.className='stepbar';
  bar.innerHTML='<button type="button" class="nx"></button><button type="button" class="rs gh">⟲ 重来</button>'+
                '<span></span><label><input type="checkbox" checked> 连续播放</label>';
  v.insertAdjacentElement('afterend', bar);
  var nx=bar.querySelector('.nx'), rs=bar.querySelector('.rs'), st=bar.querySelector('span'), au=bar.querySelector('input');
  function toStart(){ v.pause(); v.currentTime=0; i=0; last=0; }
  function show(){
    nx.disabled = auto || !v.paused;
    nx.textContent = (i===0) ? '开始 ▶' : '下一步 ▶';
    if(auto) st.textContent='连续播放中';
    else if(!v.paused) st.textContent='播放中';
    else if(i===0) st.textContent='准备好了就点「开始」';
    else st.textContent='停住了（'+i+' / '+P.length+'），看明白了再点';
  }
  toStart(); v.play().catch(function(){});
  v.addEventListener('timeupdate',function(){
    if(v.currentTime < last-1){                 // 播完一圈绕回了开头
      if(auto){ i=0; } else { toStart(); show(); return; }
    }
    last=v.currentTime;
    if(!auto && i<P.length && v.currentTime>=P[i]){ v.pause(); i++; }
    show();
  });
  v.addEventListener('pause',show); v.addEventListener('play',show);
  nx.addEventListener('click',function(){ v.play(); });
  rs.addEventListener('click',function(){ toStart(); if(auto) v.play(); show(); });
  au.addEventListener('change',function(){ auto=au.checked; if(auto) v.play(); show(); });
  show();
});
</script>
"""

# L200 R18（2026-10-20）：系统设了「减少动态效果」（prefers-reduced-motion）时，动画不自己动。
#   一步一停的片子切到手工模式、停在第 0 帧，点「开始」才走（同一套 stepbar）；没有步点的片子停住、给播放控件。
#   没有 JS 时视频照常自动循环（autoplay 不靠 JS），读屏读 aria-label。只有 L200 挂这一段，L300 页面不变。
REDUCED_MOTION_JS = """<script>
(function(){
  if(!window.matchMedia || !window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  document.querySelectorAll('video[autoplay]').forEach(function(v){
    v.removeAttribute('autoplay'); v.pause();
    var bar=v.nextElementSibling;
    if(v.dataset.pauses && bar && bar.classList.contains('stepbar')){
      var au=bar.querySelector('input'); au.checked=false; au.dispatchEvent(new Event('change'));
      bar.querySelector('.rs').click();
    } else { v.loop=false; v.controls=true; v.currentTime=0; }
  });
})();
</script>
"""

SCROLL_JS = """<style>
.scrollhint{display:none;font-size:12px;color:#5f6368;margin:2px 0 4px;text-align:right;transition:opacity .4s}
.scrollhint.on{display:block}
.scrollhint.seen{opacity:0}
</style>
<script>
(function(){
  var els=[].slice.call(document.querySelectorAll('.tbl-wrap, figure.fwide'));
  els.forEach(function(el){
    var h=document.createElement('div'); h.className='scrollhint'; h.textContent='← 左右滑动看全 →';
    el.insertAdjacentElement('beforebegin', h);
    el.addEventListener('scroll',function(){ if(el.scrollLeft>8) h.classList.add('seen'); },{passive:true});
    el._hint=h;
  });
  function check(){ els.forEach(function(el){ el._hint.classList.toggle('on', el.scrollWidth > el.clientWidth + 4); }); }
  window.addEventListener('load',check); window.addEventListener('resize',check); check();
})();
</script>
"""


def postprocess(_html, BODY, NUM_LOCK, strict=True):
    # R69（2026-10-15）：内联 SVG 的 id 全页共享，url(#x) 一律指向第一份定义。现在同名的
    #   （grid／sh／ah-<颜色>）内容逐字相同所以无害；哪天同名不同内容，后面的图会静默画错 —— 这里拦住。
    _defs = {}
    for _f, _s in re.findall(r'<figure[^>]*\sid="([^"]+)"[^>]*>\s*(<svg\b.*?</svg>)', _html, re.S):
        for _m in re.finditer(r'<(\w+)\b[^>]*\sid="([^"]+)"(?:[^>]*/>|.*?</\1>)', _s, re.S):
            _p = _defs.setdefault(_m.group(2), (_m.group(0), _f))
            assert _p[0] == _m.group(0), "SVG id「%s」在 %s 和 %s 里定义不同，url(#%s) 会指错" % (_m.group(2), _p[1], _f, _m.group(2))
    # R40（2026-10-15）：表格套共用的 .tbl-wrap（overflow-x:auto），窄屏下在表格自己里面横滚，
    #   不再撑破整页。s 文件里的 <table> 都是裸的，所以在这里统一套；已经套过的不重复套。
    _nt = _html.count("<table>")
    _html = re.sub(r'(?<!<div class="tbl-wrap">)<table>', '<div class="tbl-wrap"><table>', _html)
    _html = _html.replace("</table>", "</table></div>")
    assert _html.count('<div class="tbl-wrap"><table>') == _nt == _html.count("</table></div>"), "表格套 .tbl-wrap 对不上"
    # 复用的专题五图里带着专题五自己的编号（「第五刀」、第三节＝TP），在本课对不上。
    #   专题五已定稿、不动它的图脚本，内联后在这里改字；命中次数不对就挂，免得专题五改图后静默失效。
    # R82（2026-10-15）：改成只在指定 <figure id> 的 SVG 里改，别的图、正文、图注一个字不碰。
    #   每条写明预期命中次数（含 aria-label），对不上就挂。
    #   ⭐ attention → 注意力：本课正文统一叫「注意力」，复用图里还是英文。
    #     紧挨汉字的那个空格一并去掉（「一份 attention，」→「一份注意力，」），挨着英文／数字的保留。
    #     「Attention-FFN 分离」是 AFD 的全称，不改（大写 A 不匹配）。
    #   ⭐ 字变长的只有两处（「刀」→「切法」+1、+2 字），都在宽 1400 的整行标题里，截图看过不溢出。
    _ATT = re.compile(r"(?<![A-Za-z-])( ?)attention( ?)(?![A-Za-z-])")


    def _att(m):
        _s, _e = m.start(), m.end()
        _l = " " if m.group(1) and m.string[_s - 1:_s].isascii() and m.string[_s - 1:_s].isalnum() else ""
        _r = " " if m.group(2) and m.string[_e:_e + 1].isascii() and m.string[_e:_e + 1].isalnum() else ""
        return _l + "注意力" + _r


    _T5_FIX = {
        "fig-freq-t5": [("每一刀多久说一次话", "每种切法多久说一次话", 1),        # 「刀」是专题五对并行维度的叫法
                        ("每一刀一步要通信多少次", "每种切法一步要通信多少次", 1)],  # aria-label
        "fig-s6-fold-t5": [(_ATT, _att, 5)],
        "fig-s6-tep-dep-t5": [(_ATT, _att, 10)],
        "fig-s7-pcp-dcp-t5": [("对等的另一刀，", "对等的另一种切法，", 1)],
        "fig-s8-decode-ai-t5": [(_ATT, _att, 2)],
        "fig-s8-kv-trip-t5": [("对比第三节的 TP：", "对比 TP：", 1)],  # 专题五第三节＝TP；本课第四节是 KV cache
        "fig-s9-afd-t5": [(_ATT, _att, 14)],
    }
    for _fid, _rules in _T5_FIX.items():
        _m = re.search(r'(<figure[^>]*\sid="%s"[^>]*>\s*)(<svg\b.*?</svg>)' % re.escape(_fid), _html, re.S)
        if not _m and not strict:
            continue
        assert _m, "复用图改字：页面里没有 <figure id=%s>" % _fid
        _svg = _m.group(2)
        for _old, _new, _n in _rules:
            if isinstance(_old, str):
                _k = _svg.count(_old)
                _svg = _svg.replace(_old, _new)
            else:
                _svg, _k = _old.subn(_new, _svg)
            assert _k == _n, "复用图改字（%s）：「%s」命中 %d 次（应为 %d）—— 专题五改图了？" % (
                _fid, getattr(_old, "pattern", _old), _k, _n)
        _html = _html[:_m.start(2)] + _svg + _html[_m.end(2):]
    _leak = sorted(set(re.findall(r"__[A-Z][A-Z_0-9]*__", _html)))
    assert not _leak, "占位符没落地：%s" % "、".join(_leak)
    for _tag in ("h2", "h3", "section", "div"):
        _o = len(re.findall(r"<%s[ >]" % _tag, _html))
        _c = _html.count("</%s>" % _tag)
        assert _o == _c, "<%s> 开 %d 个、闭 %d 个" % (_tag, _o, _c)
    # 正文手打的关键数，必须在对应的图里也出现（图是脚本现算的）。比较时去掉千分位逗号。
    for _n, _svg in NUM_LOCK:
        assert _n in BODY, "正文里已经没有 %s 了 —— 从 NUM_LOCK 里删掉这一条" % _n
        _g = io.open(os.path.join(HERE, _svg), encoding="utf-8").read().replace(",", "")
        assert _n.replace(",", "") in _g, "正文写 %s，而 %s 里没有这个数 —— 图脚本改了，正文没跟上" % (_n, _svg)
    # 一步一停的播放器：照抄 topic05-build.py（那边有来由注释）。动画脚本在每一步落地后打点，
    #   写进 manim/steps/<Scene>.json；这里挂到 <video data-pauses>，图注秒数也从同一份 json 取。
    STEP_VIDEOS = {"topic06-spec.mp4": "SpecStep", "topic06-stepmix.mp4": "StepMix", "topic06-kvload.mp4": "KVLoad", "topic06-bigep.mp4": "BigEP",
                   "topic06-trip.mp4": "DecodeTrip", "topic06-whomoves.mp4": "WhoMoves",
                   "topic06-dspark-draft.mp4": "DSparkDraft", "topic06-dspark-sched.mp4": "DSparkSched"}
    # L200 R18 新拍的两支只进了 L200；L300 用 strict 构建时缺它们不算错（L300 页面一个字不变）。
    L200_ONLY = {"topic06-trip.mp4", "topic06-whomoves.mp4"}


    def _steps(mp4):
        with io.open(os.path.join(HERE, "manim", "steps", STEP_VIDEOS[mp4] + ".json"), encoding="utf-8") as fh:
            return json.load(fh)


    def _hook_steps(html):
        for mp4 in STEP_VIDEOS:
            st = _steps(mp4)
            tag = '<video src="media/%s"' % mp4
            if html.count(tag) == 0 and (not strict or mp4 in L200_ONLY):
                continue
            assert html.count(tag) == 1, "%s 在页面上出现 %d 次" % (mp4, html.count(tag))
            html = html.replace(tag, tag + ' data-pauses="%s"' % ",".join("%.2f" % t for t in st["pauses"]))
            i = html.index(tag)
            j = html.index("</figure>", i)
            fig = html[i:j]
            # ⚠️ 「N 秒」后 24 字内必须出现 Manim —— check-loop.py 靠这个正则认图注秒数，写长了它会静默跳过
            fig2, n = re.subn(r"（\d+ 秒无声循环，(.*?)Manim 渲染", lambda m: "（%d 秒，Manim 渲染；默认连续播放，取消勾选「连续播放」就能点「开始」一步一步看%s" % (round(st["duration"]), m.group(1).rstrip("，；")), fig, flags=re.S)
            assert n == 1, "%s 的图注里没找到「N 秒无声循环」" % mp4
            html = html[:i] + fig2 + html[j:]
        return html


    # R91（2026-10-16）：窄屏下宽表（.tbl-wrap）和宽图（figure.fwide，R81 起保持 1050 px）都在自己里面横滚，
    #   但手机上没有滚动条提示，看着像被截断。只给「真的放不下」的那几块加一行小字提示，
    #   用户一滑就淡掉；桌面／投影版式不出现（放得下就不加 class）。专题二到八没有现成做法，这里是第一份。
    _html = _hook_steps(_html)
    assert _html.count("</body>") == 1
    _html = _html.replace("</body>", STEP_JS + SCROLL_JS + "</body>")
    _html = P.add_figonly_toggle(_html)
    return _html


LEC_CSS = """<style>
.shell > main{grid-column:1 / -1}
.plan table td:first-child, .plan table th:first-child{white-space:nowrap}
p.say{background:#fffdf3;border-left:3px solid #f9ab00;padding:10px 14px;
      margin:12px 0;line-height:1.95}
details.deep{margin:18px 0;border:1.5px dashed #9aa0a6;border-radius:10px;padding:6px 14px;background:#fafafa}
details.deep summary{cursor:pointer;padding:6px 0}
p.deephd{font-weight:700;color:#174ea6;margin:14px 0 4px}
p.deepsay{background:#f8f9fa;border-left:3px solid #9aa0a6;padding:8px 14px;margin:8px 0;line-height:1.9}
/* L35 讲师现场用：只在本讲义生效，不动共用样式 */
h3.sec .tm6{font:600 12px/1.6 system-ui,sans-serif;letter-spacing:0;text-transform:none;
  color:#174ea6;background:#e8f0fe;border-radius:10px;padding:1px 10px;white-space:nowrap}
.say .board:before{content:none}   /* 每条 cue 自带「🖥 屏幕」，共用样式再前缀一遍「屏幕 ·」就重了 */
.say .board.sub{margin-top:10px}
.say > p.board.sub ~ p:nth-of-type(2){text-indent:0}   /* 主屏挪到节首后，🎯 那段仍是首段，不缩进 */
.say p.ia6{text-indent:0;background:#e8f0fe;border-left:3px solid #1a73e8;border-radius:0 8px 8px 0;
  padding:8px 14px;font-family:system-ui,sans-serif;font-size:14.5px;line-height:1.85}
.say .skip6{display:inline-block;text-indent:0;background:#fef7e0;color:#7a4f00;border:1px solid #f9ab00;
  border-radius:10px;padding:0 8px;margin:0 2px;font-weight:700;font-style:normal}
.ask6{background:#f3e8fd;border-left:4px solid #9334e6;border-radius:0 10px 10px 0;
  padding:13px 18px;margin:14px 0;font-size:15px;line-height:1.75}
</style>
"""


def lecture_finish(html, CH, PLAN, PLAN_SUM, JUDGED, WEB, DECK, OUT):
    """讲义两档共用：时间账（地板）、两处计划对账、封面 chip、数字／动画对账、台词块检查，然后落盘。
    ⭐ 从 topic06-build-lecture.py 原样搬来（2026-10-19 L200 R2），L300 讲义输出逐字节不变。"""
    # ── 时间账：从成品 HTML 现数，别手写常数 ────────────────────────────
    _SAY_RE = r'<p class="say[^"]*">(.*?)</p>'
    _say = "".join(re.sub(r"<[^>]+>", "", m) for m in re.findall(_SAY_RE, html, re.S))
    _n = len(re.findall(r"[一-鿿]", _say))
    _cuts = len(re.findall(r'<p class="board">', html))
    _tot = _n / 200.0 + _cuts * 0.25

    _secs = [(m.start(), re.sub(r"<[^>]+>", "", m.group(1)))
             for m in re.finditer(r'<h3 class="sec" id="c\d+">(第[零一二三四五六七八九十]+节.*?)</h3>', html)]
    assert len(_secs) == len(CH) == len(PLAN), "节数对不上：%d / %d / %d" % (len(_secs), len(CH), len(PLAN))
    _secs.append((html.index("🚧 这份讲义还薄"), ""))
    _rows, _floor, _over, _untimed = [], 0.0, [], 0.0
    _ia_tot, _sk_tot, _ia_rows = 0, 0, []    # 只读统计：互动时长，不进任何 assert
    for _i in range(len(CH)):
        _t = html[_secs[_i][0]:_secs[_i + 1][0]]
        _cn = len(re.findall(r"[一-鿿]", "".join(re.sub(r"<[^>]+>", "", x)
                                                 for x in re.findall(_SAY_RE, _t, re.S))))
        _cc = len(re.findall(r'<p class="board">', _t))
        _f = _cn / 200.0 + _cc * 0.25
        # 标题里没有「（N′」＝ 深讲、不计时（L300 第十四节起）：地板照算、单列，不进合计、不比计划。
        _pm = re.search(r"（(\d+)′", _secs[_i][1])
        _plan = int(_pm.group(1)) if _pm else None
        _ias = re.findall(r"<p>💬 <b>互动</b>.*?</p>", _t, re.S)
        _ia, _sk = len(_ias), sum("⏭" in x for x in _ias)
        if _plan is not None:
            _floor += _f
            _ia_tot, _sk_tot = _ia_tot + _ia, _sk_tot + _sk
        else:
            _untimed += _f
        _ia_rows.append((_secs[_i][1].split("　")[0], _f + _ia * 0.5, _f + (_ia - _sk) * 0.5, _plan))
        if _plan is not None and _f > _plan:
            _over.append((_secs[_i][1].split("　")[0], _f, _plan))
        _rows.append("<tr><td>%s</td><td>%s</td><td>%d</td><td>%.1f′</td><td>%s</td><td>%d / %d</td><td>%.1f′</td><td>%.1f′</td></tr>"
                     % (_secs[_i][1].split("（")[0], format(_cn, ","), _cc, _f,
                        ("%d′" % _plan) if _plan is not None else "不计时",
                        _ia, _sk, _f + _ia * 0.5, _f + (_ia - _sk) * 0.5))
    html = html.replace("@@FLOOR@@",
        '<table><thead><tr><th>节</th><th>台词汉字</th><th>切图</th><th>地板</th><th>计划给的</th><th>互动 / 可跳</th><th>含互动</th><th>跳后</th></tr></thead><tbody>'
        + "".join(_rows)
        + '<tr><td><b>合计</b></td><td colspan="2"></td><td><b>%.0f′</b></td><td><b>%d′</b></td><td><b>%d / %d</b></td><td><b>%.0f′</b></td><td><b>%.0f′</b></td></tr>'
        % (_floor, PLAN_SUM, _ia_tot, _sk_tot, _floor + _ia_tot * 0.5, _floor + (_ia_tot - _sk_tot) * 0.5) + "</tbody></table>")

    # ⭐ 两套计划时长（节标题里的、时间表里的）必须一致 —— 专题四踩过「各自自洽、互相对不上」。
    _head_min = [int(m) for m in re.findall(r'<h3 class="sec" id="c\d+">第[零一二三四五六七八九十]+节[^（<]*（(\d+)′', html)]
    assert _head_min == [p[1] for p in PLAN if p[1] is not None], ("节标题 %s 与时间表 %s 对不上"
                                                % (_head_min, [p[1] for p in PLAN]))
    assert len(_head_min) + sum(p[1] is None for p in PLAN) == len(PLAN), "不计时的节，标题和时间表要一起标"
    # ⭐ 2026-09-24 L19：「可压到 N」曾经五节都写得比地板还低 —— 压不到的承诺比不写更坑讲师。
    for _p, _r in zip(PLAN, _rows):
        _m = re.match(r"可压到 (\d+)", _p[2])
        _fv = float(re.findall(r"<td>([\d.]+)′</td>", _r)[0])
        # 2026-10-16：创作阶段不卡时长，压缩最后再定 —— 只报告不拦。
        if _m and int(_m.group(1)) < _fv:
            print("   ℹ️  %s 写「可压到 %s」但地板已经 %.1f′（创作阶段只报告）" % (_p[0], _m.group(1), _fv))
    _ia_over = [x for x in _ia_rows if x[3] is not None and x[2] > x[3]]
    if _ia_over:
        print("   ℹ️  跳过 ⏭ 互动后仍超计划的节（只报告）：")
        for _nm, _fa, _fs, _pl in _ia_over:
            print("       %-16s 含互动 %5.1f′ · 跳后 %5.1f′ > 给 %2d′" % (_nm, _fa, _fs, _pl))
    if _over:
        print("   ⚠️  地板超过计划的节：")
        for _nm, _fl, _pl in _over:
            print("       %-16s 地板 %5.1f′ > 给 %2d′" % (_nm, _fl, _pl))

    # ⭐ 2026-09-25 对照构建手册补的两条（手册 09 · 3.3）：
    # ① 课件封面「讲约 N 分钟」是手打的，跟这里的计划表是同一个量的两处写法 —— 断两处之间。
    _chip = re.search(r"讲约 (\d+) 分钟", io.open(os.path.join(WEB, DECK), encoding="utf-8").read())
    assert _chip and int(_chip.group(1)) == PLAN_SUM, \
        "课件封面写「讲约 %s 分钟」，讲义计划表合计 %d′ —— 改一处不改另一处" % (_chip and _chip.group(1), PLAN_SUM)
    # ② 台词数字密度（从专题四搬来，只报告）：一串数从耳朵进去台下留不住。
    #   判据：这个数是「内容」还是「内容的证据」？证据说「有人量过」就够。
    # 已判定「数就是内容」的块登记在各 topic06_lec_*.JUDGED，写明为什么。
    _JUDGED = JUDGED
    _noisy = []
    for _b in re.findall(r'<p class="say">(.*?)</p>', html, re.S):
        _t = re.sub(r"<[^>]+>", "", _b).replace("&nbsp;", " ")
        _c = len(re.findall(r"[一-鿿]", _t))
        if _c < 40:
            continue
        _d = len(re.findall(r"[0-9]", re.sub(r"[A-Za-z]+-?[0-9]+", "", _t)))   # 摘掉 TP4、DEP8 这类名字里的数字
        if _d * 100.0 / _c > 12.0 and not _t.strip().startswith(_JUDGED):
            _noisy.append("%.0f/百字  %s" % (_d * 100.0 / _c, re.sub(r"\s+", " ", _t)[:40]))
    if _noisy:
        print("   ⚠️  台词数字偏密（只报告，判断是内容还是证据）：")
        for _x in _noisy:
            print("       " + _x)

    # ⛔ 讲稿里报的 TiB/GiB 大数，教材页面上必须也有（讲稿是手打的，漂的一定是讲稿）。
    _DECKP = os.path.join(WEB, DECK)
    _deck = re.sub(r"\s+", "", io.open(_DECKP, encoding="utf-8").read())
    _flat = re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", html))
    _BIG = re.findall(r"[0-9]+\.[0-9]+(?:TiB|GiB)", _flat)
    _lost = sorted({b for b in _BIG if b not in _deck})
    assert not _lost, "讲稿报了教材上没有的数：%s" % "、".join(_lost)

    # ⛔ 讲稿里指的动画 id，课件里必须有（topic02-lint-cues 只对 fig- 开头的 id）。
    _anims = set(re.findall(r"<code>(anim-[a-z0-9-]+)</code>", html))
    _miss = sorted(a for a in _anims if ('id="%s"' % a) not in io.open(_DECKP, encoding="utf-8").read())
    assert not _miss, "讲稿指了课件里没有的动画：%s" % "、".join(_miss)

    # ⛔ 一块台词对应一个动作：不许超过 200 汉字。
    _long = [(len(re.findall(r"[一-鿿]", re.sub(r"<[^>]+>", "", s))), re.sub(r"<[^>]+>", "", s)[:30])
             for s in re.findall(r'<p class="say">(.*?)</p>', html, re.S)]
    _long = [x for x in _long if x[0] > 200]
    assert not _long, "有台词块超过 200 汉字：%s" % _long

    # ⛔ 台上提示不许漏进台词：say 块里每一个字都默认要出口。
    _STAGE_ONLY = ("不要在这里", "不要展开讲", "整段跳过", "别念", "台上不用念",
                   "时间紧可以", "这一格不要", "别在这里", "讲之前停", "先只放")
    _leak = [(w, re.sub(r"<[^>]+>", "", t)[:40]) for t in re.findall(r'<p class="say">(.*?)</p>', html, re.S)
             for w in _STAGE_ONLY if w in t]
    assert not _leak, "台上提示漏进台词了：%s" % _leak

    # ⛔ 收尾语气只许出现在最后一节，否则会「谢两次幕」。
    _CURTAIN = ("数字会过期", "念完就停")
    _cstart = [m.start() for m in re.finditer(r'<h3 class="sec" id="c\d+"', html)]
    _early = sorted({w for m in re.finditer(r'<p class="say">(.*?)</p>', html, re.S)
                     for w in _CURTAIN if w in m.group(1) and m.start() < _cstart[-1]})
    assert not _early, "收尾语气出现在最后一节之前：%s" % "、".join(_early)

    # L35：节标题小字回填；互动和 ⏭ 加 class 只为好认（放在所有统计之后，不影响任何计数）
    for _i, (_nm, _fa, _fs, _pl) in enumerate(_ia_rows):
        html = html.replace("@@IA%02d@@" % _i, ("含互动 %.1f′" % _fa) + ((" · 跳后 %.1f′" % _fs) if _fs < _fa else ""))
    assert "@@IA" not in html
    html = html.replace("<p>💬 <b>互动</b>", '<p class="ia6">💬 <b>互动</b>')
    html = html.replace("⏭ <b>时间紧可跳</b>", '<span class="skip6">⏭ 时间紧可跳</span>')

    html = P.bust_media(html, OUT)          # 录音地址挂内容指纹，重生成后不被缓存成旧的
    io.open(OUT, "w", encoding="utf-8").write(html)
    print("    互动 %d 个（⏭ 可跳 %d）× 0.5′：含互动 %.0f′ · 跳后 %.0f′（计划 %d′）"
          % (_ia_tot, _sk_tot, _floor + _ia_tot * 0.5, _floor + (_ia_tot - _sk_tot) * 0.5, PLAN_SUM))
    if _untimed:
        print("    深讲不计时的节另有地板 %.1f′（不进合计）" % _untimed)
    print("ok  %s  %s 字符 · %d 节 · 台词 %s 字 ＋ %d 切图 ＝ 地板 %.0f′（计划 %d′）"
          % (os.path.basename(OUT), format(os.path.getsize(OUT), ","), len(CH), format(_n, ","), _cuts, _tot - _untimed, PLAN_SUM))



_L300_FIGS = None


def l300_fig(fig_id, caption):
    """L200 复用 L300 的图：按图 id 找到 SVG 文件和生成脚本，图注由 L200 自己写。
    ⛔ L300 图注里常带本课节号（「见第四节」），换了位置就错，所以不许原样搬图注。"""
    global _L300_FIGS
    if _L300_FIGS is None:
        import importlib
        _L300_FIGS = {}
        for _i in range(20):
            try:
                _m = importlib.import_module("topic06_s%d" % _i)
            except ModuleNotFoundError:
                continue
            for _v in getattr(_m, "FIGS", {}).values():
                _L300_FIGS.setdefault(_v[0], _v)
    assert fig_id in _L300_FIGS, "L300 里没有图 %s" % fig_id
    _, svg, script = _L300_FIGS[fig_id][:3]
    return (fig_id, svg, script, caption)
