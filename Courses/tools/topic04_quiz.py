# -*- coding: utf-8 -*-
r"""专题四 · 开场热身：**四个模型 ↔ 四种注意力** 的连线题。

⭐⭐ **为什么单独一个模块**：它要的四行数据（模型名、厂商配色、循环构成、
   一句话备注）**全部来自专题三那份 `topic03_models.ROWS`**。
   ⛔ 抄一份过来是最省事的写法，也是最会出事的写法 ——&#160;
     专题三那边改了 config，这边会静默地留在旧值上，而且谁都不报错。
     （`topic03_models.py` 顶上写着同一条：「数据必须只有一份」。）
   ⇒ 所以这里**只做呈现，不带任何数据**；四个模型用名字前缀去那边取，
     取不到或取到多行**当场断言失败**。

⭐ 题目本身是「回顾上一讲」：左边四个模型，右边四条注意力构成条
  （用的就是专题三那张表里同一套类型配色），连线 → Submit → 全对才公布答案。
"""
import re

import topic03_models as M

# ── 左列顺序（模型）───────────────────────────────────────────────
# 📌 用**名字前缀**去 ROWS 里取，不写死任何参数。
PICK = ["Kimi K3", "DeepSeek-V4-Pro", "Qwen3.5", "小米 MiMo-V2.5-Pro"]

# ── 右列顺序 ──────────────────────────────────────────────────────
# RIGHT[j] = 第 j 个格子里摆的是 PICK 里第几个模型的构成。
# ⛔ 断言里钉死两件事：① 是个排列；② **没有不动点** ——&#160;
#   任何一个「原地对上」的槽位都会让那一条变成送分题，
#   而四条里只要有一条是送的，整道题的难度就不是你以为的那个。
RIGHT = [3, 0, 1, 2]


def _row(prefix):
    hit = [r for r in M.ROWS if r[1].startswith(prefix)]
    assert len(hit) == 1, (
        "连线题：模型前缀 %r 在 topic03_models.ROWS 里命中 %d 行，要求正好 1 行。\n"
        "  —— 命中 0 行多半是那边改了名字；命中多行说明前缀不够长。"
        % (prefix, len(hit)))
    return hit[0]


def _note_html(s):
    """ROWS 的备注是给 SVG 用的，里面是 <tspan>。搬到 HTML 要换成 <b>。"""
    s = re.sub(r'<tspan[^>]*font-weight="700"[^>]*>', "<b>", s)
    return s.replace("</tspan>", "</b>")


def _cycle_cells(cyc):
    out = []
    for ty, k in cyc:
        out += ['<i style="background:%s">%s</i>' % (M.TYPE_COL[ty], ty)] * k
    return "".join(out)


CSS = """
/* ── 开场连线题 ─────────────────────────────────────────────────── */
#qz{--qzgap:26px}
#qz .qzwrap{position:relative;display:grid;grid-template-columns:1fr 150px 1fr;
  gap:var(--qzgap) 0;align-items:center;margin:14px 0 6px}
/* ⛔ 2026-09-22 第一版渲染出来是**四根竖线** ——&#160;因为两列都用了
   `justify-self:stretch`，左格子的右边缘和右格子的左边缘几乎贴在一起，
   两个端点的 x 坐标差不到 10px，贝塞尔曲线被压成了一根竖杠。
   ⭐ 判据：**连线题里「线」是内容，而线的长度是布局给的。**
     所以两列必须**各自贴着中间那条沟**收缩，不能拉满。 */
/* ⛔ 第二个坑，而且第一次完全没看出来：中间那个 <svg> 是 position:absolute，
   **所以它不是 grid item** ——&#160;右边那一列于是自动落到了第 2 条轨道
   （150px 那条）上，把它撑破，两列就又贴到一起了。
   ⭐ 判据：**绝对定位的孩子不占格子。** 靠「写了三列就会分三列」是错的，
     要么给它留一个真的占位元素，要么把列号钉死 ——&#160;这里钉列号。 */
#qz .qzL{grid-column:1;justify-items:end}
#qz .qzR{grid-column:3;justify-items:start}
#qz .qzi{position:relative;z-index:2;cursor:pointer;user-select:none;
  border:2px solid transparent;border-radius:9px;padding:7px 10px;transition:.12s}
#qz .qzi:hover{background:#f1f3f4}
#qz .qzi.on{border-color:#1a73e8;background:#e8f0fe}
#qz .qzi.tied{background:#fff}
#qz .chip{display:inline-block;padding:2px 9px;border-radius:6px;border:1px solid;
  font-weight:700;white-space:nowrap;font-size:13px}
#qz .cy{white-space:nowrap}
#qz .cy i{display:inline-block;width:35px;text-align:center;color:#fff;font-style:normal;
  font-size:9px;font-weight:700;padding:3px 0;border-radius:3px;margin-right:3px}
#qz svg.wire{position:absolute;inset:0;width:100%;height:100%;z-index:1;pointer-events:none;overflow:visible}
#qz .qzbar{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap}
#qz button{font:inherit;font-weight:700;padding:7px 20px;border-radius:8px;cursor:pointer;
  border:1px solid #1a73e8;background:#1a73e8;color:#fff}
#qz button.gh{background:#fff;color:#1a73e8}
#qz button:disabled{opacity:.45;cursor:default}
#qz .qzmsg{font-weight:700}
#qz .qzmsg.bad{color:#d93025}
#qz .qzmsg.good{color:#1e8e3e}
#qz .qzans{margin-top:14px;border-top:1px dashed #dadce0;padding-top:12px}
#qz .qzans li{margin:7px 0}
@media(max-width:760px){#qz .qzwrap{grid-template-columns:1fr 60px 1fr}}
"""

JS = """
(function(){
 var root=document.getElementById('qz'); if(!root) return;
 var KEY=__KEY__;
 var L=[].slice.call(root.querySelectorAll('.qzL .qzi'));
 var R=[].slice.call(root.querySelectorAll('.qzR .qzi'));
 var svg=root.querySelector('svg.wire'), wrap=root.querySelector('.qzwrap');
 var msg=root.querySelector('.qzmsg'), ans=root.querySelector('.qzans');
 var btn=root.querySelector('.go'), rst=root.querySelector('.rst');
 var sel=null, tie={};                       // tie[左 index] = 右 index
 function clear(){ while(svg.firstChild) svg.removeChild(svg.firstChild); }
 function draw(){
   clear();
   var W=wrap.getBoundingClientRect();
   if(!W.width) return;                      // details 还没展开，宽度是 0
   Object.keys(tie).forEach(function(i){
     var a=L[i].getBoundingClientRect(), b=R[tie[i]].getBoundingClientRect();
     var ln=document.createElementNS('http://www.w3.org/2000/svg','path');
     var x1=a.right-W.left, y1=a.top+a.height/2-W.top;
     var x2=b.left-W.left,  y2=b.top+b.height/2-W.top;
     var m=(x1+x2)/2;
     ln.setAttribute('d','M'+x1+','+y1+'C'+m+','+y1+' '+m+','+y2+' '+x2+','+y2);
     ln.setAttribute('fill','none');
     ln.setAttribute('stroke', L[i].getAttribute('data-ink')||'#1a73e8');
     ln.setAttribute('stroke-width','2.5');
     ln.setAttribute('stroke-linecap','round');
     svg.appendChild(ln);
   });
 }
 function paint(){
   L.forEach(function(e,i){ e.classList.toggle('on', sel===i);
                            e.classList.toggle('tied', tie[i]!==undefined); });
   R.forEach(function(e,j){ var used=Object.keys(tie).some(function(k){return tie[k]===+j;});
                            e.classList.toggle('tied', used); });
   btn.disabled = Object.keys(tie).length < L.length;
   draw();
 }
 L.forEach(function(e,i){ e.onclick=function(){
   if(tie[i]!==undefined){ delete tie[i]; sel=null; }   // 再点一次＝拆掉这条
   else sel=(sel===i?null:i);
   msg.textContent=''; msg.className='qzmsg'; ans.hidden=true; paint(); }; });
 R.forEach(function(e,j){ e.onclick=function(){
   if(sel===null){ msg.textContent='先点左边那个模型'; msg.className='qzmsg bad'; return; }
   Object.keys(tie).forEach(function(k){ if(tie[k]===j) delete tie[k]; }); // 一对一
   tie[sel]=j; sel=null; msg.textContent=''; msg.className='qzmsg';
   ans.hidden=true; paint(); }; });
 btn.onclick=function(){
   var ok=L.every(function(_,i){ return tie[i]===KEY[i]; });
   if(ok){ msg.textContent='✅ 四条全对 —— 答案在下面。'; msg.className='qzmsg good';
           ans.hidden=false; }
   else  { msg.textContent='❌ 不对。再看一眼这四家各自是哪条路线。';
           msg.className='qzmsg bad'; ans.hidden=true; }
   draw();
 };
 rst.onclick=function(){ tie={}; sel=null; msg.textContent=''; msg.className='qzmsg';
                         ans.hidden=true; paint(); };
 root.addEventListener('toggle', function(){ setTimeout(draw,0); });
 var d=root.closest('details'); if(d) d.addEventListener('toggle',function(){setTimeout(draw,0);});
 window.addEventListener('resize', draw);
 paint();
})();
"""


def build():
    rows = [_row(p) for p in PICK]
    assert sorted(RIGHT) == list(range(len(PICK))), "RIGHT 必须是一个排列"
    assert all(j != RIGHT[j] for j in range(len(RIGHT))), \
        "RIGHT 里有不动点：那一条会变成送分题，整道题的难度就不是四选四了"

    # key[i] = 模型 i 的正确构成落在右列第几格
    key = [RIGHT.index(i) for i in range(len(PICK))]

    left = []
    for r in rows:
        _lab, ink, bg = M.vendor_of(r[1])
        left.append('<div class="qzi" data-ink="%s"><span class="chip" '
                    'style="color:%s;background:%s;border-color:%s">%s</span></div>'
                    % (ink, ink, bg, ink, r[1]))
    right = ['<div class="qzi"><span class="cy">%s</span></div>'
             % _cycle_cells(rows[i][2]) for i in RIGHT]

    ansli = "".join(
        '<li><span class="chip" style="color:%s;background:%s;border-color:%s">%s</span>'
        '　<span class="cy">%s</span><br><span class="sub">%s　·　上下文 <b>%s</b></span></li>'
        % ((lambda v: (v[1], v[2], v[1]))(M.vendor_of(r[1])) + (r[1],
           _cycle_cells(r[2]), _note_html(r[5]), r[3]))
        for r in rows)

    return ("""
<details class="foldfig" id="qzbox"><summary><b>🎯 开场热身：上一讲那四个模型，各自走的是哪条路？</b>
  <span class="why">——&nbsp;连线题，四条全对才公布答案。<u>不做也不影响这一讲</u></span></summary>
<div id="qz">
<p>左边是四个模型，右边是四条<b>注意力构成</b>（一格＝一层，这是
  <a href="topic-03.html">专题三</a>那张编年史表里同一套配色）。
  <b>先点左边一个，再点右边一个</b>就连上；<b>再点一次左边</b>可以拆掉。</p>
<div class="qzwrap">
  <div class="qzL" style="display:grid;gap:var(--qzgap)">%s</div>
  <svg class="wire"></svg>
  <div class="qzR" style="display:grid;gap:var(--qzgap)">%s</div>
</div>
<div class="qzbar"><button class="go" disabled>Submit</button>
  <button class="gh rst">重来</button><span class="qzmsg"></span></div>
<div class="qzans" hidden><p><b>对上了。这四条其实就是当下的四条主流路线：</b></p>
<ul>%s</ul>
<p class="landing"><b>而这一讲要问的是另一半 ——&nbsp;上面这些都是「推理时怎么省」。
  训练的时候，这四家谁都躲不开同一张账单。</b></p></div>
</div>
<script>%s</script>
</details>
""" % ("".join(left), "".join(right), ansli,
       JS.replace("__KEY__", "[" + ",".join(str(k) for k in key) + "]")), CSS)
