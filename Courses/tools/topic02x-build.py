# -*- coding: utf-8 -*-
"""专题二 · 外传 —— **v6e 与扩散模型**（L100，20 分钟）。

════════════════════════════════════════════════════════════════
⭐⭐ 这一讲的定位（现场定的，别漂）
════════════════════════════════════════════════════════════════
· **L100，二十分钟。** 台下没见过 TPU 也能跟得上。
· **以图为主，几乎不写字。** 正文只做三件事：接上一张、点出这一张要看什么、
  交到下一张。**凡是图上写得下的，正文一个字都不重复。**
· **主角是芯片，不是模型。** 十几分钟花在「两颗硬件哪里不一样」上，
  扩散模型只是那个用来说明「什么活配什么芯片」的例子。

⛔ **跟专题十一的分工，别越界：**
    专题十一 = 模型侧，一小时 L300（加噪去噪、latent、VAE、DiT、CFG…）
    这一讲   = 硬件侧，二十分钟 L100（两颗芯片 ＋ 一根强度轴）
  ⭐ 两边唯一的交叉是「扩散没有 KV cache、形状静态」这三条结构事实，
    这里**只取它们对硬件的后果**，不讲它们为什么成立。

⛔⛔ **不做端到端 benchmark 对比。** 现场明确定的：
    「不用对比那么明显的什么 v6e 跟 H100 的那个 benchmark，
      因为我们这个内容估计也就 20 分钟。」
  ⚠️ 而且我们手上唯一一组同模型双平台实测（HunyuanVideo-1.5）
    **也不支持「v6e 更快」**。理由与数记在 topic02x-verified-specs.md 第四节。
  ⭐ 这一讲讲的是**结构匹配**，不是快慢 —— 这条界要在页面上说出来。

════════════════════════════════════════════════════════════════
⏱ 二十分钟怎么走
════════════════════════════════════════════════════════════════
    0–3    钩子：同一把尺子量四颗芯片，只有 v6e 落在别处      图 X-1
    3–8    把 v6e 拆开                                        图 X-2
    8–12   把 H100 按同样的画法拆开 —— 三个相反的选择          图 X-3
   12–15   一颗装得下吗：那三条「减配」其实是配套的            图 X-6
   15–17   扩散一步在干什么                                    图 X-5
   17–19   把活放回那两条线上                                  图 X-4
   19–20   落点

⭐ 硬件占 0–15 分钟（四张图），模型只占 15–19 ——&nbsp;
  现场原话「主要是用十几分钟把这两款硬件的不同讲明白」。

⛔ BODY 里有 % 号，**只能用普通字符串**，不要在它上面做 %-格式化。
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "WebPages")
CSS_SRC = os.path.join(WEB, "topic-02-L300.html")
OUT = os.path.join(WEB, "topic-02x.html")

_src = io.open(CSS_SRC, encoding="utf-8").read()


def _sub(text, pattern, repl, what):
    """替换 ＋ 断言真的替换了。
    ⛔ 不要退回裸 str.replace ——&nbsp;它匹配不上时**静默原样返回**，
      「改了」和「没改成」在代码里长得一模一样（专题三栽过，整页 og 全错）。"""
    new, n = re.subn(pattern, repl, text, count=1)
    assert n == 1, "改不动 %s —— 模板变了？（模式：%s）" % (what, pattern)
    return new


head = _src[:_src.index("</style>") + len("</style>")]
head = _sub(head, r"<title>.*?</title>",
            "<title>专题二 · 外传：v6e 与扩散模型</title>", "<title>")
head = _sub(head, r'<meta property="og:title" content="[^"]*">',
            '<meta property="og:title" content="v6e 与扩散模型 · 一颗算力强、显存弱的芯片，配什么样的活">',
            "og:title")
head = _sub(head, r'<meta property="og:description" content="[^"]*">',
            '<meta property="og:description" content="二十分钟，两张显微镜图，一根强度轴。'
            'H100 的屋脊点是 295，v6e 是 560 —— 这一讲只回答一个问题：什么样的活凑得够 560。">',
            "og:description")
head = _sub(head, r'<meta property="og:url" content="[^"]*">',
            '<meta property="og:url" content="https://gist.higcp.com/Courses/WebPages/topic-02x.html">',
            "og:url")
head = re.sub(r'\s*<meta property="og:image"[^>]*>'
              r'(\s*<meta property="og:image:(width|height)"[^>]*>)*', "", head)
_probe = re.sub(r"/\*.*?\*/|<!--.*?-->", "", head, flags=re.S)
assert "TPU 与 GPU" not in _probe and "topic-02-L300" not in _probe, \
    "head 里还有专题二 L300 的残留"

head += """
<style>
ul + p, ol + p, ul + div.note, ol + div.note, table + p { margin-top: 14px }
h4 { margin:18px 0 6px; font-size:15px }
/* ⭐ L100 的版面比 L200/L300 松一档：图大、字少，图之间给足呼吸。
   ⛔⛔ 只能写 margin-top / margin-bottom，**绝不能用 `margin: 34px 0` 这种简写**。
   简写会把 margin-left 一并设成 0，而 `figure.fbox`（特异度 0,1,1）跟
   `figure.fwide` 同级、又排在后面，于是压掉 .fwide 的 margin-left:50%，
   只剩 translateX(-50%) 生效 ——&nbsp;**整张宽图往左平移半个屏幕**。
   ⚠️ 这个坑 topic02-port-microscope.py 里白纸黑字写过（2026-09-04 栽的），
   我 2026-09-09 建这一页时又原样踩了一遍。
   ⭐ 判据：**改 margin 一律写具体方向，别用简写** ——&nbsp;
     简写的杀伤力在于它悄悄重置了你没打算动的那三个方向。
   ⭐ 而且它**不产生横向滚动条**：往左跑出视口既不撑大 scrollWidth、
     也不报错，只有量 getBoundingClientRect().x 才看得见。 */
figure.fbox { margin-top: 34px; margin-bottom: 34px }
</style>
"""


def fig(name, cap):
    """插一张图。SVG 是产物，源在 topic02x-fig-*.py。
    ⛔ 别在这儿改 SVG ——&nbsp;改了下次重跑就没了。"""
    p = os.path.join(HERE, name + ".svg")
    assert os.path.isfile(p), "找不到 %s.svg —— 先跑对应的 topic02x-fig-*.py" % name
    svg = io.open(p, encoding="utf-8").read().strip()
    return ('<figure class="fbox fwide" id="%s">%s\n<figcaption>%s</figcaption></figure>'
            % (name, svg, cap))


BODY = []
a = BODY.append

a("""
<header class="hero">
  <div class="kicker">专题二 · 外传　·　L100　·　20 分钟</div>
  <h1>算力强，显存弱 —— 这样一颗芯片，该配什么样的活</h1>
  <p class="lede">TPU v6e 与扩散模型。<b>这一讲的主角是芯片，不是模型。</b>
    十几分钟把两颗硬件哪里不一样讲清楚，最后用扩散模型说明
    <b>「什么样的活配什么样的芯片」</b>。</p>
</header>

<section id="s零">
  <h2><span class="secno">开场</span>这一讲只回答一个问题</h2>
  <p>专题二立过一条线：<b>算力 ÷ 显存带宽</b>，得到「每搬一个字节，这台机器本来能算多少次」。
    那一讲算出 B200 和 TPU v7 <b>几乎一模一样</b>。</p>
  <p>这一讲把同一把尺子多量两颗 —— <b>其中一颗立刻破了那个「都一样」</b>。</p>
  <div class="note info"><span class="t">⛔ 先说清这一讲不做什么</span>
    <b>不比 benchmark，不谈谁跑得快。</b>屋脊点是<b>结构量</b>，它只说「这台机器的胃口有多大」。
    <em>本讲从头到尾只讲硬件结构与负载匹配。</em>
    <br>扩散模型本身的原理（加噪、去噪、latent、VAE、DiT）在<b>专题十一</b>，那是一小时的深潜。
  </div>
</section>
""")

a('<section id="s一">')
a('  <h2><span class="secno">一</span>同一把尺子，量四颗芯片</h2>')
a(fig("figx-1",
      "<b>图 X-1</b>　三颗旗舰挤在 295–313，<b>v6e 是 560</b>。"
      "「算力强、显存弱」不是形容词 —— 它就是这个数。"
      "<em>接下来两张图，看这个 560 是从什么样的硅片布局里长出来的。</em>"))
a('</section>')

a('<section id="s二">')
a('  <h2><span class="secno">二</span>把两颗芯片拆开</h2>')
a('  <p>先看 v6e。<b>只要记住三件事：算力集中在两个大方阵里、片上有一整块暂存、'
  '通往片外的那根管子很窄。</b></p>')
a(fig("figx-2",
      "<b>图 X-2</b>　一颗 v6e：<b>1 颗 ＝ 1 个核 ＝ 1 个 device</b>，"
      "128 MiB 片上暂存，片外那道门 1,638 GB/s。"
      "<em>专题二反复强调的「容量除以 2」那个坑，这一代没有。</em>"))
a('  <p>再按<b>同样的画法</b>拆 H100。看它在同一件事上做了什么相反的选择。</p>')
a(fig("figx-3",
      "<b>图 X-3</b>　一颗 H100：算力摊成 <b>528 个</b>小单元，"
      "片上多了一整层<b>硬件自动管</b>的 L2，门宽一倍。"
      "<em>⭐ 两张图里那道门的宽度是按带宽等比画的，可以直接比。</em>"))
a('  <p>最后补一张。X-2 里有三条规格看着像<b>减配</b> —— 只有 4 个 ICI 口、'
  '二维环面、一个 Pod 只有 256 颗。<b>把模型的体积摆出来，这三条就说得通了。</b></p>')
a(fig("figx-6",
      "<b>图 X-6</b>　扩散这一族在<b>「一颗到几颗」</b>的量级，"
      "而 LLM 在「几十颗」。<em>v6e 不打「一个模型摊在几千颗上」那场仗，"
      "所以不用付那个成本。</em>"
      "<br>⚠️ 图上如实标了两条踩线和超线的 —— <b>只算权重，不含激活。</b>"))
a('  <p>X-2 右栏还有两样只标了存在、没展开 —— <b>SparseCore</b> 和 <b>二维环面</b>。'
  '这两张各补一下 —— <b>尤其 SparseCore，它不只是「给推荐系统用的」。</b></p>')
a(fig("figx-7",
      "<b>图 X-7</b>　SparseCore <b>有两个职责</b>：① 稀疏 / 大表（推荐模型那类）；"
      "<b>② 集合通信卸载 —— 把 all-reduce、all-gather 从 TensorCore 手里接过去，"
      "不占 MXU，跟计算真正并行。</b>"
      "<em>⛔ 这张图我推翻重写过一次：初版写「扩散用不到它」是错的 —— "
      "只要跨卡，它就在干活。</em>"))
a(fig("figx-8",
      "<b>图 X-8</b>　4 个 ICI 口连上下左右，边缘绕回成二维环面，16×16 ＝ 256 颗，"
      "<b>最远 16 跳</b>（对照 v7 的 4×4×4 是 6 跳）。"
      "<em>走不走这条路取决于你切不切模型 —— 不切几乎不走，一切开就一大堆 all-gather。"
      "<b>v6e 的互联是「够扩散这一族用」的档位，不是「不用」。</b></em>"))
a("""
  <div class="note ok"><span class="t">⭐ 两颗芯片各自都是自洽的</span>
    H100 摊成很多小单元、再压一层硬件缓存，是为了应付<b>「我不知道你要跑什么」</b>；
    v6e 压成两块大方阵、片上全交给编译器，是为了吃透<b>「我早就知道你要跑什么」</b>。
    <br>——&nbsp;<b>所以问题从来不是谁更强，是你手上的活属于哪一种。</b>
  </div>
""")
a('</section>')

a('<section id="s三">')
a('  <h2><span class="secno">三</span>那么扩散模型是哪一种活</h2>')
a('  <p>先回答一个更基础的问题：<b>这类模型为什么是「计算密集」的？</b>'
  '拿 Wan2.1 的官方配置当场算一遍就清楚了。</p>')
a(fig("figx-9",
      "<b>图 X-9</b>　一段 720P、81 帧的视频经 VAE 与 patch 化后是 "
      "<b>75,600 个 token</b>；而 config 里 <code>window_size = (−1,−1)</code> —— "
      "<b>不开窗口，全局注意力</b>。于是算力的<b>七成花在 N² 的注意力上</b>，"
      "那是纯矩阵乘。"
      "<em>⭐ 外部锚点：每层 351.3 M × 40 层 ＝ 14.05 B，对上官方标称的 14B。</em>"))
a('  <p>算力密集讲清楚了，再看它<b>结构上</b>为什么正好配这套编译优先的打法。</p>')
a(fig("figx-5",
      "<b>图 X-5</b>　没有 KV cache、形状从头到尾不变、同一段计算原样跑五十遍。"
      "<em>编译期能知道的，它全都提前告诉你了 —— 这正是 v6e 那套打法要的前提。</em>"))
a('</section>')

a('<section id="s四">')
a('  <h2><span class="secno">四</span>把活放回那两条线上</h2>')
a(fig("figx-4",
      "<b>图 X-4</b>　一条规则就够：<b>强度 ≈ 同一份权重被多少个「位置」共用</b>。"
      "decode 落在线的左边，prefill 和扩散甩到右边两个数量级。"))
a("""
  <div class="note ok"><span class="t">⭐⭐ 「v6e 适合扩散」的精确说法</span>
    <b>不是它跑扩散更快。</b>是扩散把它的短板<b>挡在了瓶颈之外</b> ——&nbsp;
    强度上万，那根窄管子根本没成为瓶颈；
    <b>于是只剩下没被挡住的那一项在起作用：算力 918 对 989.5，是 H100 的 93%。</b>
    <br><em>短板不参与，长板打平 —— 这就是「合适」的全部含义。</em>
  </div>
  <div class="note info"><span class="t">同一条道理，原样适用于 prefill 重、decode 轻的任务</span>
    prefill 是整段一次过，几千上万个位置共用一次权重搬运，它跟扩散落在轴上的<b>同一边</b>。
    <br>⛔ 反过来也成立：<b>batch 小的 decode 落在最左边</b>，那里比的全是带宽，
    而 v6e 只有 H100 的 49% —— <b>同一颗芯片，在那种活上就是最吃亏的。</b>
  </div>
""")
a('</section>')

a("""
<section id="s五">
  <h2><span class="secno">五</span>落点</h2>
  <p class="landing">—— <b>选芯片不是选「更强的那颗」，是先看你的活落在那根轴的哪一边。</b>
    <br><em>落右边（扩散、prefill、大 batch）比的是算力，v6e 打平；
    落左边（小 batch decode）比的是带宽，v6e 吃亏。</em>
    <br><b>而这两件事，从两颗芯片的布局图上就能提前看出来 —— 不用等跑完。</b></p>
  <div class="note warn"><span class="t">⚠️ 这一讲刻意留白的地方</span>
    <b>没有端到端性能数。</b>本讲讲结构匹配，不讲快慢 —— 两件事需要的证据不一样，
    后者要实测，而实测要连口径一起给。
    <br><em>想看扩散模型本身：<b>专题十一</b>。想看这套分析框架怎么来的：<b>专题二</b>。</em>
  </div>
</section>
""")

html = head + '\n<main class="wrap">\n' + "\n".join(BODY) + "\n</main>\n"

# ── 写盘前自检 ──────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(HERE, "tpu-micro"))
from gate import lint_public                                    # noqa: E402
bad = lint_public(html)
assert not bad, "公开页面里出现内部词，已中止写盘：%s" % bad
n_fig = html.count('<figure class="fbox fwide"')
assert n_fig == 9, "图数不对：%d（应为 9）" % n_fig

io.open(OUT, "w", encoding="utf-8").write(html)
print("ok  topic-02x.html  %s 字符  %d 图" % (format(len(html), ","), n_fig))
