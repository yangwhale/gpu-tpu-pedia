# -*- coding: utf-8 -*-
"""专题三 · 模型编年史**表格**（HTML，表头可点排序）。

⭐⭐ **为什么从 SVG 换成 HTML。** 2026-09-07 现场要求：

    「你再给我做一下排序。按时间排序，按模型名字也就是类别排序，
      按那个一个循环排序，按 KV Cache 的大小排序。
      就是这个表格上边那四个 title，你点哪个就按哪个排序。」

   ⛔ **SVG 做不到** ——&nbsp;它的 `<text>` 是死的：点不了、Ctrl+F 搜不到、
     复制不出来、屏幕阅读器也读不了。39 行的参照表本来就不该是一张图。

   ⭐ 换成 HTML 之后白拿的四样：**可搜、可复制、可无障碍、可排序**。
     代价只有一个：布局不像 SVG 那样像素级可控 ——&nbsp;而对一张表来说，
     那本来也不是优点。

⛔ **数据不在这个文件里**，在 `topic03_models.py`。上半那张时间轴 SVG 读的是
   同一份。**抄第二份 = 两个产物迟早漂开且不报错。**

📌 排序键（都从数据算，⛔ 不在 HTML 里写死）：
   · 时间 →&nbsp;`YYYY-MM` 字典序
   · 模型 →&nbsp;**按厂商分组、组内按名字**（现场说的「类别」就是厂商）
   · 一个循环 →&nbsp;`cheap_frac`，便宜层占比，默认从多到少
   · 上下文 →&nbsp;`ctx_tokens`，换算成 token 数
   · KV cache →&nbsp;算出来的 GiB
"""
import io
import os

import topic03_models as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3-models-table.html")

KVW = 400.0                 # KV 条满一行多长（px）
KV_PER_ROW = 60.0           # 一整行 ＝ 60 GiB（线性真实比例）
KVS = KVW / KV_PER_ROW


def kv_segs(g):
    """真实比例下要几段、每段多长。⛔ 跟 SVG 那版同一套口径，别各算各的。"""
    px = max(2.0, g * KVS)
    n = max(1, int(-(-px // KVW)))          # ceil
    return [KVW] * (n - 1) + [px - (n - 1) * KVW]


def esc(s):
    return s.replace("&#160;", " ").replace('<tspan font-weight="700">', "<b>") \
            .replace("</tspan>", "</b>")


rows = []
for tm, mdl, cyc, ctx, kvspec, note in M.ROWS:
    vlab, vink, vbg = M.vendor_of(mdl)
    g = M.kv_gib(kvspec)

    # ── 模型：厂商色的小框 ────────────────────────────────────────
    cell_mdl = ('<span class="chip" style="color:%s;background:%s;border-color:%s">%s</span>'
                % (vink, vbg, vink, mdl))

    # ── 一个循环：一格一层，格子里印类型简写 ──────────────────────
    if len(cyc) == 1:
        ty = cyc[0][0]
        cell_cyc = ('<span class="uni" style="background:%s">%s —— 每一层都是这个</span>'
                    % (M.TYPE_COL[ty], ty))
    else:
        gs = []
        for ty, k in cyc:
            gs += ['<i style="background:%s">%s</i>' % (M.TYPE_COL[ty], ty)] * k
        cell_cyc = "".join(gs)

    # ── KV：真实比例条，超长折行 ─────────────────────────────────
    lab = M.kv_fmt(g)
    if kvspec is None:
        cell_kv = '<span class="unk">未核到</span>'
    elif g <= 0:
        cell_kv = ('<span class="kvw"><b class="bar" style="background:%s;width:6px"></b>'
                   '<em style="color:%s">%s</em></span>' % ("#1e8e3e", "#1e8e3e", lab))
    else:
        segs = kv_segs(g)
        c = M.kv_col(g)
        bars = "".join('<b class="bar" style="background:%s;width:%.0fpx"></b>' % (c, w)
                       for w in segs)
        extra = ('<em style="color:%s">%s<span class="fold">（%d 条才画得下）</span></em>'
                 % (c, lab, len(segs))) if len(segs) > 1 else \
                ('<em style="color:%s">%s</em>' % (c, lab))
        cell_kv = '<span class="kvw">%s%s</span>' % (bars, extra)

    # ── 备注一格两份：Boom 模式给这一行独有的那句，Highlight 模式给
    #    「它凭什么在这条线上」。⭐ 两种模式不是「同一张表少几行」，
    #    是**两种读法** —— 全量表是参照物，highlight 是一条能读下来的线。
    _hl = M.HL.get(M.short_name(mdl))
    cell_nt = ('<span class="n-all">%s</span>' % esc(note)
               + ('<span class="n-hl">%s</span>' % esc(_hl) if _hl else ''))

    rows.append(
        '<tr data-t="%s" data-v="%s" data-c="%.6f" data-x="%d" data-k="%.6f"%s>'
        '<td class="tm">%s</td><td>%s</td><td class="cy">%s</td>'
        '<td class="ctx">%s</td><td class="kv">%s</td><td class="nt">%s</td></tr>'
        # ⛔ 排序键里原先用 \x01 当分隔符 —— **控制字符进不了 XML 属性**，
        #   而它在浏览器里是隐形的：不报错、不显示、只是把解析悄悄弄坏。
        #   ⭐ 写盘前那道 XML 自检就是抓这个的。换成可打印的 "|"。
        % (tm, esc("%s|%s" % M.vendor_key(mdl, tm)).replace('"', "&quot;"),
           M.cheap_frac(cyc), M.ctx_tokens(ctx),
           -1.0 if g is None else g,
           ' data-hl="1"' if _hl else '',
           tm, cell_mdl, cell_cyc, ctx, cell_kv, cell_nt))

CSS = """
<style>
/* ⛔ 这张表天然 ~1400px 宽，而版心只有 1032 —— 直接放会溢出 390px。
   ⚠️ 而版面 lint **抓不到这一条**：它只查 overflow 不是 visible 的容器
      （它自己注释里写明了这个盲区）。溢出是我量出来的，不是它报的。
   ⭐ 所以这里自带一条破版心规则。两个坑都躲开：
     ① 宽度取 min(1400, 视口-40)，窄屏不会撑破右边；
     ② margin-left 由宽度反算，**不写死** —— 写死了窄屏会往左跑出视口，
        而「往左跑」既不产生滚动条也不撑大 scrollWidth，探针恒报 0。 */
.tblwrap{overflow-x:auto}
/* ⛔ 第一版用 min(1400px, 100vw-40px) 反算 margin，在 1200/1440 上都溢出了 4px
      —— 因为 **100vw 是含滚动条的**，而可用宽度不含。差那几像素就撑出页面。
   ⭐ 改成「只在够宽的屏上破版心」：1500px 以上才破（那时 1400 一定放得下），
      窄屏就老老实实待在版心里横向滚动。**不跟 vw 的边界情况较劲。** */
/* 表的实测自然宽是 1618（62+288+360+110+416+382）。壳比它小就会出内部滚动条，
   所以壳取 1620、破版心的门槛取 1720（那时 1620 一定放得下）。
   ⛔ 这两个数是**量出来的**，改了列宽就要重量一次。 */
@media (min-width:1720px){
  .tblwrap{width:1620px;margin-left:calc((1080px - 1620px) / 2 - 24px)}
}
.tbltip{color:#5f6368;font-size:12.5px;margin:0 0 8px}
.land{margin:14px 0 0;padding:14px 18px;background:#e8f0fe;border:1px solid #1a73e8;
  border-radius:8px;color:#174ea6;font-size:12.5px;line-height:1.7}
.land .lh{margin:0 0 6px;font-weight:700}
.land ol{margin:0;padding-left:22px}
.land li{margin:3px 0}
#mtbl{width:100%;border-collapse:collapse;font-size:12.5px;line-height:1.45}
#mtbl th{text-align:left;padding:6px 8px;border-bottom:2px solid #dadce0;color:#5f6368;
  font-weight:700;white-space:nowrap}
#mtbl th.s{cursor:pointer;user-select:none}
#mtbl th.s:hover{color:#1a73e8;background:#f1f3f4}
#mtbl th.s::after{content:"\\2195";opacity:.35;margin-left:4px;font-weight:400}
#mtbl th.s.up::after{content:"\\2191";opacity:1;color:#1a73e8}
#mtbl th.s.dn::after{content:"\\2193";opacity:1;color:#1a73e8}
#mtbl td{padding:5px 8px;border-bottom:1px solid #f1f3f4;vertical-align:middle}
#mtbl td.tm{color:#5f6368;white-space:nowrap;font-variant-numeric:tabular-nums}
#mtbl td.ctx{white-space:nowrap;font-weight:700;color:#5f6368}
/* ⭐ 备注**一行一句，不换行**。现场：「保持一行是一行」——
   折行会把行高撑起来，信息密度直接掉一半。
   ⛔ 前提是备注本身已经压到 ≤ 375px（见 topic03_models.py 里 ROWS 的注释）：
     光设 nowrap 而不压内容，只会把表撑到 1700+，等于把问题挪个地方。 */
#mtbl td.nt{color:#5f6368;white-space:nowrap}
#mtbl .chip{display:inline-block;padding:2px 9px;border-radius:6px;border:1px solid;
  font-weight:700;white-space:nowrap}
#mtbl .cy{white-space:nowrap}
#mtbl .cy i{display:inline-block;width:40px;text-align:center;color:#fff;font-style:normal;
  font-size:9px;font-weight:700;padding:3px 0;border-radius:3px;margin-right:3px}
#mtbl .uni{display:inline-block;padding:3px 10px;border-radius:4px;color:#fff;
  font-size:10px;font-weight:700;white-space:nowrap}
#mtbl .kvw{display:inline-flex;flex-wrap:wrap;align-items:center;gap:2px;max-width:470px}
#mtbl .kv .bar{display:block;height:7px;border-radius:2px;flex:0 0 auto}
#mtbl .kv em{font-style:normal;font-weight:700;font-size:11px;margin-left:6px;white-space:nowrap}
#mtbl .kv .fold{font-weight:400;opacity:.75}
#mtbl .unk{color:#9aa0a6;font-size:10px;border:1px solid #dadce0;border-radius:3px;padding:2px 7px}
#mtbl .hint{font-weight:400;color:#9aa0a6;font-size:11px}

/* ── Highlight ／ Boom 两态开关 ────────────────────────────────────
   ⭐ 只藏行、不动数据：隐藏是 CSS 干的，排序 JS 照常按 data-* 重排全部 39 行，
     两个功能互不知道对方存在 ——&nbsp;所以「先排序再切模式」和「先切模式再排序」
     结果一样。⛔ 别改成 JS 里 removeChild，那样一切模式排序状态就丢了。 */
.hlbar{display:flex;align-items:center;gap:10px;margin:0 0 10px}
.hlbtns{display:inline-flex;border:1px solid #dadce0;border-radius:999px;overflow:hidden}
.hlbtns button{border:0;background:#fff;color:#5f6368;font:inherit;font-size:12.5px;
  font-weight:700;padding:5px 16px;cursor:pointer;line-height:1.4}
.hlbtns button+button{border-left:1px solid #dadce0}
.hlbtns button:hover{background:#f1f3f4;color:#1a73e8}
.hlbtns button[aria-pressed="true"]{background:#1a73e8;color:#fff}
.hlbtns button[aria-pressed="true"]:hover{background:#1a73e8;color:#fff}
.hlnote{color:#5f6368;font-size:12px}
/* 默认（Boom）：全部 39 行，备注给这一行独有的那句 */
#mtbl .n-hl{display:none}
/* Highlight：非入选行整行不出现，备注换成「它凭什么在这条线上」 */
#mtbl.hl tbody tr:not([data-hl]){display:none}
#mtbl.hl .n-all{display:none}
#mtbl.hl .n-hl{display:inline;color:#174ea6}
</style>
"""

# ⛔ 排序脚本只做一件事：按 data-* 属性重排 <tr>。
#   **不重算任何数值** ——&nbsp;数值全在 Python 那边算好写进属性里了。
#   前端再算一遍 = 第二份实现 = 迟早跟后端漂开。
JS = """
<script>
(function(){
 var t=document.getElementById('mtbl'); if(!t) return;
 var tb=t.tBodies[0], cur={k:'t',d:1};
 function num(v){return parseFloat(v);}
 t.querySelectorAll('th.s').forEach(function(th){
  th.addEventListener('click',function(){
   var k=th.dataset.k;
   // 同一列再点一次就反向；换列时用该列"最有信息量"的默认方向：
   // 时间/模型 升序，配比/上下文/KV 降序（大的先看）
   var def = (k==='t'||k==='v') ? 1 : -1;
   cur = (cur.k===k) ? {k:k,d:-cur.d} : {k:k,d:def};
   var rs=[].slice.call(tb.rows);
   rs.sort(function(a,b){
    var x=a.dataset[k], y=b.dataset[k], r;
    r = (k==='t'||k==='v') ? (x<y?-1:x>y?1:0) : (num(x)-num(y));
    return r*cur.d;
   });
   rs.forEach(function(r){tb.appendChild(r);});
   t.querySelectorAll('th.s').forEach(function(o){o.classList.remove('up','dn');});
   th.classList.add(cur.d>0?'up':'dn');
  });
 });
 var f=t.querySelector('th.s[data-k="t"]'); if(f) f.classList.add('up');

 // ── Highlight ／ Boom ──────────────────────────────────────────
 // ⛔ 只切 class，不碰 DOM 顺序、不碰任何数值。
 var bs=document.querySelectorAll('.hlbtns button'), nt=document.querySelector('.hlnote');
 function setMode(hl){
  t.classList.toggle('hl', hl);
  bs.forEach(function(b){b.setAttribute('aria-pressed', String(b.dataset.m===(hl?'hl':'all')));});
  if(nt) nt.textContent = hl
    ? '只看撑起这段历史的 __NHL__ 行 —— 备注换成「它凭什么在这条线上」'
    : '全部 __NALL__ 行。备注是这一行独有的那句话';
 }
 bs.forEach(function(b){b.addEventListener('click',function(){setMode(b.dataset.m==='hl');});});
 setMode(true);          // ⭐ 默认 Highlight：39 行是参照物，18 行才是一条读得下来的线
})();
</script>
"""

TH = ('<tr>'
      '<th class="s" data-k="t">时间</th>'
      '<th class="s" data-k="v">模型 <span class="hint">按厂商</span></th>'
      '<th class="s" data-k="c">一个循环 <span class="hint">一格＝一层</span></th>'
      '<th class="s" data-k="x">上下文 <span class="hint">声明值</span></th>'
      '<th class="s" data-k="k">KV cache＠128K</th>'
      '<th>备注</th></tr>')


# ══════════════════════════════════════════════════════════════════
# 七条落点。⛔ **全部从数据算**，不写死 ——&nbsp;加一行模型，数字自己跟着变。
#   （原先这些画在 SVG 里，跟着表一起搬过来。）
_uni = [r for r in M.ROWS if len(r[2]) == 1]
_mix = [r for r in M.ROWS if len(r[2]) > 1]
_hyb = [r for r in _mix
        if len({t_ for t_, _ in r[2]}) == 2
        and any(t_ in M.CHEAP for t_, _ in r[2])
        and any(t_ not in M.CHEAP for t_, _ in r[2])]
_oth = [r for r in _mix if r not in _hyb]
_warm = [r for r in _hyb if r[2][-1][0] not in M.SPARSE]
_cold = [r for r in _hyb if r[2][-1][0] in M.SPARSE]
_1m = [r for r in M.ROWS if r[3].endswith("M")]
# ⛔ 2026-09-07 修。这里原先是「全表非零 KV 的最大 ÷ 最小」，算出 1152 倍 ——
#   而那个最小值是 **Mistral 7B 的 512 MiB**。⭐ 拿 7B 去跟 175B 比 KV 倍数，
#   比出来的是**模型大小**，不是机制省下来的量，而这一条恰恰是想说机制。
#   ⭐⭐ 形状：**极值统计会自动挑出「最不可比的那一行」** ——&nbsp;
#     min/max 不知道什么叫可比，它只知道大小。加一行小模型就能把结论悄悄改掉。
#   修法：只在**前沿规模（≥100B 总参）**里取两端，口径写进正文那句话里。
_BIG = 100.0
_kv = [(r[1], M.kv_gib(r[4])) for r in M.ROWS
       if r[4] is not None and M.kv_gib(r[4]) > 0 and M.total_params_b(r[1]) >= _BIG]
_mx, _mn = max(_kv, key=lambda x: x[1]), min(_kv, key=lambda x: x[1])
_v2, _v3 = M.kv_gib(("mla", 60, 576)), M.kv_gib(("mla", 61, 576))
_short = lambda n: n.split("\u3000")[0].replace("⭐ ", "")

LAND = ("""<div class="land"><p class="lh">⭐ 这张表一眼能看出七件事
<span style="font-weight:400;color:#5f6368">（以下统计**恒按全部 %d 行**算，切到 Highlight 也不变
——&#160;不然「有几家怎么样」这种话会跟着显示模式变，那就不是结论了）</span></p>""".replace("**", "") % len(M.ROWS) + """
<ol>
<li>表里 <b>%d</b> 家：<b>%d 家是「便宜的层 ＋ 一层贵的」</b>，%d 家<b>每层同构</b>，
%d 家是别的混法。而那 %d 家混合的，<b>配比无一例外落在 3:1 ～ 7:1</b> ——
<b>没有人敢全用线性，也没有人只掺一两层。</b></li>
<li>前几行是基线，也是一条完整的小史：MHA → MQA（砍到 1 组）→ GQA（折中）→ MLA（改压缩）
→ <b>CLA（跨层共享）</b>——<b>全都只在动「每个 token 存多少」这一个旋钮</b>。</li>
<li>⭐ <b>扫一眼颜色搭配</b>：混合的那 %d 家里 <b>%d 家是「冷色 ＋ 黄橙」</b>（便宜的层配一层全注意力）。
<b>只有 %s 是「蓝 ＋ 红」：它配的那层「贵的」，本身已经是稀疏的了。</b></li>
<li>⭐⭐ 把<b>上下文</b>那一列排一下：做到 <b>1M 以上的 %d 家，无一例外都动了旋钮②或③</b>；
纯全注意力那一档最高只到 256K。最硬的对照来自 MiniMax 自己：
<b>01 用 7:1 线性外推到 4M，M2 退回纯全注意力只剩 192K</b>——同一家、同一批人，差二十倍。</li>
<li>⭐⭐ 把 <b>KV cache</b> 排一下：<b>%s</b>（%s）到 <b>%s</b>（%s），整整 <b>%d 倍</b>
<span style="color:#5f6368">——&#160;两端都取 <b>100B 以上</b>的，不然「最小」会落到 Mistral 7B 头上，
那比的是模型大小不是机制</span>。
而这不是一个旋钮拧出来的——MHA→GQA 砍头数、MLA 改压缩、<b>CLA 跨层共享</b>是旋钮①；
线性把大部分层的 KV <b>直接删成零</b>是旋钮③；CSA／HCA 存压缩池是旋钮②。
<b>三个旋钮各贡献了一段。</b>⭐ 而 <b>RWKV 那一行干脆是 0</b>——纯 RNN 没有 KV cache 这个东西。</li>
<li>⛔⛔ <b>「上下文」这一列报的是<u>声明</u>，不是<u>能用</u>。</b>
数来自各家 config 的 max_position_embeddings，而这个字段各家含义并不一样——
MiniMax-01 那格 config 写着 <b>10,240,000</b>，官方只声称<b>训练 1M、外推 4M</b>；
Llama 4 Scout 声称 <b>10M</b> 也是同一回事。
<b>声明和能用之间还隔着一整个 benchmark 的落差</b>：小米自己的模型卡就写着
V2-Pro「到 1M 时塌到 0.00」。⭐ <b>看到「支持 N 万上下文」，先问是谁、在什么任务上、测出多少分。</b></li>
<li>⭐⭐ <b>最反直觉的一条：MLA 之后，KV cache 跟「模型多大」<u>脱钩</u>了。</b>
DeepSeek-V2 是 236B、V3 是 671B，<b>参数差 2.8 倍，KV 却只差 %.1f%%</b>（%s vs %s）。
因为 MLA 的 KV 只跟<b>「层数 × (kv_lora_rank ＋ rope 维)」</b>走——
跟专家多少、hidden 多宽、总参多大一点关系都没有。
MHA 时代 KV 是跟着模型一起长的，<b>这条链在 MLA 这里被剪断了。</b></li>
</ol></div>""" % (len(M.ROWS), len(_hyb), len(_uni), len(_oth), len(_hyb),
                  len(_hyb), len(_warm), "、".join(_short(r[1]) for r in _cold),
                  len(_1m),
                  M.kv_fmt(_mx[1]), _short(_mx[0]), M.kv_fmt(_mn[1]), _short(_mn[0]),
                  round(_mx[1] / _mn[1]),
                  (_v3 / _v2 - 1) * 100, M.kv_fmt(_v2), M.kv_fmt(_v3)))

_NHL = sum(1 for r in M.ROWS if M.is_hl(r[1]))

BAR = ('<div class="hlbar">'
       '<span class="hlbtns">'
       '<button type="button" data-m="hl" aria-pressed="true">⭐ Highlight</button>'
       '<button type="button" data-m="all" aria-pressed="false">Boom ——&#160;全部</button>'
       '</span><span class="hlnote"></span></div>')

html = ('%s<div class="tblwrap">%s<p class="tbltip">⭐ <b>点表头可以排序</b>'
        '——&#160;时间 / 厂商 / 便宜层占比 / 上下文 / KV 大小，'
        '再点一次反向。<b>默认按时间。</b>两种模式下排序都作用在全部 %d 行上。</p>'
        '<table id="mtbl"><thead>%s</thead><tbody>\n%s\n</tbody></table>%s</div>%s'
        % (CSS, BAR, len(M.ROWS), TH, "\n".join(rows), LAND, JS))
# ⛔ 计数写进 JS 是**从数据填的**，不是手打的字面量 ——&nbsp;加一行模型，
#   按钮旁边那句说明会自己跟着变。手打的话它会在某次加行之后静默说谎。
html = html.replace("__NHL__", str(_NHL)).replace("__NALL__", str(len(M.ROWS)))
assert "__N" not in html, "计数占位符没被替换掉"

# ── 写盘前自检 ────────────────────────────────────────────────────
assert html.count("<tr ") == len(M.ROWS), "行数对不上"
assert "higcp" not in html, "私人域名不能进公开产物"
import xml.dom.minidom  # noqa: E402  仅用来验表格片段是不是良构 XML
xml.dom.minidom.parseString(
    "<root>" + html[html.index("<table"):html.index("</table>") + 8] + "</root>")
io.open(OUT, "w", encoding="utf-8").write(html)
print("ok  fig3-models-table.html  %s 字符 · %d 行"
      % (format(len(html), ","), len(M.ROWS)))
