# -*- coding: utf-8 -*-
r"""专题五 · 开场热身：DeepSeek-V3 训练时，每个参数那 16 字节都是什么。

⭐ 题目回顾上一讲（专题四）的核心账，同时给这一讲埋伏笔：
   第一问  五块各几字节 → 乘出 9.76 TiB（回顾）
   第二问  哪块最大、先削哪块 → 预习：答案在 §2.2（优化器状态 12 字节，ZeRO 下的第一刀）
   第三问  V3 实际没按 16 字节存 → 预习：答案在 §2.2 的答案框（answers_html），开场只提问
   （2026-09-25 现场定：第一问复习，第二、三问预习，只看题，本讲才给答案。）
⭐ 数字全部从 topic05_numbers 取，这里一个都不抄（专题四 topic04_quiz.py 同一条规矩）。
⭐ 格子里只写「这块是什么」，不写精度 —— 精度正是要答的东西，写上就送分了。
"""
import topic05_numbers as NB

ROWS = [("权重（算的时候用的那一份）", NB.W_B),
        ("梯度", NB.G_B),
        ("主权重（优化器更新的那一份）", NB.MASTER_B),
        ("一阶动量 m", NB.M_B),
        ("二阶动量 v", NB.V_B)]
assert sum(b for _, b in ROWS) == NB.PER_PARAM == 16

CSS = """
.qz5 table{margin:10px 0;border-collapse:collapse;width:auto}
.qz5 td,.qz5 th{padding:6px 12px;border-bottom:1px solid #e8eaed;text-align:left}
.qz5 input{width:4.2em;font:inherit;padding:2px 6px;border:1px solid #dadce0;border-radius:6px}
.qz5 input.ok{border-color:#1e8e3e;background:#e6f4ea}
.qz5 input.bad{border-color:#d93025;background:#fce8e6}
.qz5 .qzbar button{font:inherit;padding:4px 14px;border-radius:16px;border:1px solid #1a73e8;
  background:#1a73e8;color:#fff;cursor:pointer;margin-right:8px}
.qz5 .qzbar button.gh{background:#fff;color:#1a73e8}
.qz5 .qzmsg{color:#5f6368}
.qz5 ol{padding-left:2.2em;margin:6px 0}
"""

JS = """
(function(){
 var root=document.getElementById('qz5'); if(!root) return;
 var ins=root.querySelectorAll('input[data-k]'), msg=root.querySelector('.qzmsg'),
     ans=root.querySelector('.qzans');
 root.querySelector('.go').addEventListener('click',function(){
   var bad=0;
   ins.forEach(function(i){ var ok=(parseFloat(i.value)===parseFloat(i.dataset.k));
     i.classList.toggle('ok',ok); i.classList.toggle('bad',!ok); if(!ok) bad++; });
   if(bad){ msg.textContent='还有 '+bad+' 格不对（红框）。提示：半精度 2 字节，全精度 4 字节。'; ans.hidden=true; }
   else { msg.textContent='全对。'; ans.hidden=false; }
 });
 root.querySelector('.rst').addEventListener('click',function(){
   ins.forEach(function(i){ i.value=''; i.classList.remove('ok','bad'); });
   msg.textContent=''; ans.hidden=true; });
})();
"""


def build():
    rows = "".join('<tr><td>%s</td><td><input inputmode="numeric" data-k="%d" aria-label="%s 占几个字节"> 字节</td></tr>'
                   % (lab, b, lab) for lab, b in ROWS)
    tib = NB.PSI * NB.PER_PARAM / NB.TIB
    html = """
<details class="foldfig" id="qz5box"><summary><b>🎯 开场热身：DeepSeek-V3 训练时，每个参数占的那 16 个字节都是什么？</b>
  <span class="why">—— 第一问复习专题四，第二、三问预习这一讲。<u>不做也不影响这一讲</u></span></summary>
<div class="qz5" id="qz5">
<p><b>第一问</b>：按常规的混合精度 AdamW 训练，每个参数要在显存里常驻下面五样东西。各占几个字节？</p>
<table>%s</table>
<div class="qzbar"><button class="go">对答案</button><button class="gh rst">重来</button><span class="qzmsg"></span></div>
<div class="qzans" hidden>
<p>2 ＋ 2 ＋ 4 ＋ 4 ＋ 4 ＝ <b>16 字节</b>。前两样是半精度（bf16），算矩阵乘用；后三样是全精度（fp32），给优化器用。
  乘上 V3 的 6,710 亿参数：<b>%.2f TiB</b>，约合一万 GB —— 这还不含激活。</p>
<p class="sub">16 字节是常规口径，梯度按半精度算。实际用 bf16 训练时，框架常把<b>累加梯度</b>的那份放全精度（Megatron 默认就这样，V3 也是），那就是 18 字节。</p>
</div>
<p><b>预习</b>（先想一想，答案在这一讲的 §2.2 揭晓）：</p>
<ol start="2">
<li><b>第二问</b>：这 16 字节里哪一块最大？要是让你削，先削哪块？</li>
<li><b>第三问</b>：DeepSeek-V3 其实没按 16 字节存，它把其中两块压成了半精度。是哪两块？主权重为什么不能压？</li>
</ol>
</div>
<script>%s</script>
</details>
""" % (rows, tib, JS)
    return html, CSS

def answers_html():
    """第二、三问的答案 —— 放在课件 §2.2（ZeRO 那一节），开场只提问不揭晓。"""
    step_v3 = 1 - NB.V3_BETA2
    step_torch = 1 - NB.TORCH_BETA2
    assert step_torch < NB.BF16_HALF_ULP and step_v3 > 2 ** -7, "β₂ 那段推导的前提变了，重写第三问"
    return """<div class="note ok"><span class="t">开场第三问的答案：V3 把两个动量压成了半精度，主权重留在全精度（批次累积用的那份梯度也是）</span>
  注意优化器状态里<b>没有梯度</b>：梯度只是每步喂给优化器的输入，用完就清零；要放全精度的是把几个小批的梯度加起来的那个累加器。<br>
  主权重每步只加一点点、要一直累加，压了会被舍掉；动量能不能压要看 β₂ —— 展开下面「细一点」那把刻度尺（V3 用 %.2f，PyTorch 默认 %.3f）。</div>
""" % (NB.V3_BETA2, NB.TORCH_BETA2)
