# -*- coding: utf-8 -*-
r"""专题六 · 开场热身：五种并行各用哪几种集合通信（回顾专题五）。

⭐ 2026-10-19 现场定：开场折叠，课上点开；每行多选，勾完一起交。
⭐ 答案全部来自专题五第一节「一张表收住」和各刀的通信账，这里不引入新知识：
   DP    同步梯度 AllReduce（＝ ReduceScatter ＋ AllGather，拆开勾也算对）
   FSDP  前向 AllGather 拼权重，反向再拼一次，ReduceScatter 分梯度
   TP    每层 AllReduce；配序列并行时换成 AllGather ＋ ReduceScatter（也算对）
   PP    stage 之间一对一收发激活
   EP    每层两次 AllToAll（派发、收回）
⭐ 判分按「整行勾选集合」比，不按单格 —— 少勾、多勾都算错。
"""

COLS = [("AR", "AllReduce"), ("AG", "AllGather"), ("RS", "ReduceScatter"),
        ("A2A", "AllToAll"), ("P2P", "一对一收发")]

# (行名, 一句提示, 可接受的勾选集合)
ROWS = [("DP 数据并行", "每张卡一份完整模型，各吃一份数据", ["AR", "AG+RS", "AG+AR+RS"]),
        ("FSDP（ZeRO-3）", "权重、梯度、优化器状态全切开", ["AG+RS"]),
        ("TP 张量并行", "一层里的矩阵切给几张卡", ["AR", "AG+RS", "AG+AR+RS"]),
        ("PP 流水线并行", "按层切成几段 stage", ["P2P"]),
        ("EP 专家并行", "MoE 的专家分到不同卡", ["A2A"])]
_KEYS = {k for k, _ in COLS}
for _, _, acc in ROWS:
    for s in acc:
        assert set(s.split("+")) <= _KEYS and s == "+".join(sorted(s.split("+"))), s

CSS = """
.qz6 table{margin:10px 0;border-collapse:collapse;width:auto}
.qz6 td,.qz6 th{padding:6px 10px;border-bottom:1px solid #e8eaed;text-align:center}
.qz6 th{text-transform:none;letter-spacing:0}   /* 全站 th 默认大写，集合通信名要保留驼峰 */
.qz6 td:first-child,.qz6 th:first-child{text-align:left}
.qz6 td .sub{display:block;font-size:12px;color:#80868b}
.qz6 input[type=checkbox]{width:18px;height:18px;cursor:pointer}
.qz6 tr.ok td{background:#e6f4ea}
.qz6 tr.bad td{background:#fce8e6}
.qz6 .qzbar button{font:inherit;padding:4px 14px;border-radius:16px;border:1px solid #1a73e8;
  background:#1a73e8;color:#fff;cursor:pointer;margin-right:8px}
.qz6 .qzbar button.gh{background:#fff;color:#1a73e8}
.qz6 .qzmsg{color:#5f6368}
"""

JS = """
(function(){
 var root=document.getElementById('qz6'); if(!root) return;
 var rows=root.querySelectorAll('tr[data-k]'), msg=root.querySelector('.qzmsg'),
     ans=root.querySelector('.qzans');
 root.querySelector('.go').addEventListener('click',function(){
   var bad=0;
   rows.forEach(function(r){
     var got=[]; r.querySelectorAll('input:checked').forEach(function(c){ got.push(c.value); });
     var ok=r.dataset.k.split('|').indexOf(got.sort().join('+'))>=0;
     r.classList.toggle('ok',ok); r.classList.toggle('bad',!ok); if(!ok) bad++; });
   msg.textContent = bad ? ('还有 '+bad+' 行不对（红底）。少勾、多勾都算错。') : '全对。';
   ans.hidden=false;
 });
 root.querySelector('.rst').addEventListener('click',function(){
   root.querySelectorAll('input').forEach(function(c){ c.checked=false; });
   rows.forEach(function(r){ r.classList.remove('ok','bad'); });
   msg.textContent=''; ans.hidden=true; });
})();
"""


NOTE_L300 = ('推理只剩前向，这张表还能用，但轻重变了：TP 每层那几次 AllReduce 数据不多，卡的是延迟（5.3、第七节）；'
             'EP 也不一定走 AllToAll —— 公开的 TPU 推理专家那步用的是 AllGather 加 ReduceScatter（6.4）。')


def build(note=NOTE_L300):
    """note：答案表下那行小字。默认是 L300 的写法（带 L300 节号）；L200 传自己的，节号不能串档。"""
    th = "".join("<th>%s</th>" % lab for _, lab in COLS)
    trs = "".join(
        '<tr data-k="%s"><td><b>%s</b><span class="sub">%s</span></td>%s</tr>'
        % ("|".join(acc), name, hint,
           "".join('<td><input type="checkbox" value="%s" aria-label="%s 用 %s"></td>' % (k, name, lab)
                   for k, lab in COLS))
        for name, hint, acc in ROWS)
    html = """
<details class="foldfig" id="qz6box"><summary><b>🎯 开场热身：五种并行，各自要用哪几种集合通信？</b>
  <span class="why">—— 回顾专题五，课上一起点开（课前勿点）。<u>不做也不影响这一讲</u></span></summary>
<div class="qz6" id="qz6">
<p>每一行勾上这种并行<b>训练一步</b>里要用到的通信，可以勾不止一个。五行都勾完再点「对答案」。</p>
<table><tr><th>并行方式</th>%s</tr>%s</table>
<div class="qzbar"><button class="go">对答案</button><button class="gh rst">重来</button><span class="qzmsg"></span></div>
<div class="qzans" hidden>
<table>
<tr><th>并行方式</th><th>答案</th><th>用在哪</th></tr>
<tr><td>DP</td><td>AllReduce</td><td>每步把各卡的梯度加起来、人人一份。AllReduce 本来就能拆成 ReduceScatter ＋ AllGather，拆开勾也算对</td></tr>
<tr><td>FSDP</td><td>AllGather ＋ ReduceScatter</td><td>前向用到哪层先 AllGather 拼回权重，反向再拼一次；算完的梯度 ReduceScatter，各卡只留自己那块</td></tr>
<tr><td>TP</td><td>AllReduce</td><td>每层前向两次、反向两次，把切开的矩阵乘结果加回来；配序列并行时换成 AllGather ＋ ReduceScatter，也算对</td></tr>
<tr><td>PP</td><td>一对一收发</td><td>只在相邻 stage 之间传激活（反向传梯度），不需要全员一起</td></tr>
<tr><td>EP</td><td>AllToAll</td><td>每层两次：把 token 派给它选中的专家，算完再原路收回</td></tr>
</table>
%s
</div>
</div>
<script>%s</script>
</details>
""" % (th, trs, '<p class="sub">%s</p>' % note, JS)
    return html, CSS
