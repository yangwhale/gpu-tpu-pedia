# -*- coding: utf-8 -*-
r"""专题五 · 开场热身：DeepSeek-V3 训练时，每个参数那 16 字节都是什么。

⭐ 题目回顾上一讲（专题四）的核心账，同时给这一讲埋伏笔：
   第一问  五块各几字节 → 乘出 9.76 TiB（回顾）
   第二问  哪块最大、先削哪块 → 优化器状态 12 字节，正是第二节 ZeRO 下的第一刀（伏笔）
   第三问  V3 实际没按 16 字节存 → 哪两块压成了 bf16、为什么主权重不能压（专题四判据，收紧一层）
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
    step_v3 = 1 - NB.V3_BETA2
    step_torch = 1 - NB.TORCH_BETA2
    html = """
<details class="foldfig" id="qz5box"><summary><b>🎯 开场热身：DeepSeek-V3 训练时，每个参数占的那 16 个字节都是什么？</b>
  <span class="why">—— 回顾专题四那张账，三问。<u>不做也不影响这一讲</u></span></summary>
<div class="qz5" id="qz5">
<p><b>第一问</b>：按常规的混合精度 AdamW 训练，每个参数要在显存里常驻下面五样东西。各占几个字节？</p>
<table>%s</table>
<div class="qzbar"><button class="go">对答案</button><button class="gh rst">重来</button><span class="qzmsg"></span></div>
<div class="qzans" hidden>
<p>2 ＋ 2 ＋ 4 ＋ 4 ＋ 4 ＝ <b>16 字节</b>。前两样是半精度（bf16），算矩阵乘用；后三样是全精度（fp32），给优化器用。
  乘上 V3 的 6,710 亿参数：<b>%.2f TiB</b>，约合一万 GB —— 这还不含激活。</p>
</div>
<details class="aside"><summary><b>第二问</b>：这 16 字节里哪一块最大？要是让你削，先削哪块？</summary>
<p>后三样合起来（主权重 ＋ 两个动量）是 <b>%d 字节，占四分之三</b>，统称优化器状态。
  第二节的第一刀（ZeRO）就是从这一块削起的。</p></details>
<details class="aside"><summary><b>第三问</b>：V3 其实没按 16 字节存。它把哪两块压成了半精度？主权重为什么不能压？</summary>
<p>V3 把<b>两个动量</b>改用 bf16 存，报告说「没有观察到可见的性能退化」；主权重和用于累积的梯度仍留在 fp32
  （技术报告 arXiv 2412.19437 sec. 3.3.3）。这不是默认做法：常规做法里动量跟着主权重用 fp32，Megatron 要显式打开开关才用 bf16。</p>
<p><b>主权重不能压</b>：它每一步只加进一个很小的更新，而且要一直累加下去；bf16 只有约三位有效数字，
  小更新会在四舍五入时整个被抹掉。</p>
<p><b>动量能不能压，要看衰减系数 β₂</b>（本课推导）：二阶动量每一步只掺进 (1 − β₂) 那么多新值。
  V3 的 β₂ ＝ %.2f，新值每步占 %.0f%% 的份量，bf16 分辨得出来；如果用 PyTorch 默认的 %.3f，新值只占 %.1f%%，
  比 bf16 能分辨的最小相对差距（约 0.2%%）还小 —— 除非新来的梯度跟旧值差好几倍，否则这一步的改动会被四舍五入抹掉，二阶动量就卡住了。
  所以「动量是滑动平均，所以 bf16 扛得住」这句话只在 β₂ 不太接近 1 时成立。</p>
<p>V3 报告里的验证是整套 FP8 方案（含 bf16 动量）对 BF16 基线：16B 和 230B 两个规模各训约一万亿 token，
  损失相对误差一直低于 0.25%%；没有单独把「动量用 bf16」拆出来做消融。</p></details>
</div>
<script>%s</script>
</details>
""" % (rows, tib, NB.O_B, NB.V3_BETA2, step_v3 * 100, NB.TORCH_BETA2, step_torch * 100, JS)
    # 推导的两个前提：PyTorch 默认下每步相对改动 < bf16 最小的半个间隔（一定被舍掉）；
    #   V3 的每步改动 > bf16 最大的整个间隔 2^-7（分辨得出来）。前提一变，第三问就得重写。
    assert step_torch < NB.BF16_HALF_ULP and step_v3 > 2 ** -7, "β₂ 那段推导的前提变了，重写第三问"
    return html, CSS
