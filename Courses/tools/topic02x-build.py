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

from topic02_family import nav          # 四页互跳导航，单一来源

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
# ⛔ 防回归：famnav 的样式住在 L300 的 <style> 里、由 topic02-port-microscope.py 注入。
#   那段被重写过一次、把样式冲掉过一次，而**页面照样构建、导航条只是变成裸链接** ——
#   ⭐ 不报错的退化必须用断言接住。
assert "nav.famnav" in head, \
    "切下来的 head 里没有 famnav 样式 —— 先跑 topic02-port-microscope.py"

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
/* ⛔⛔ `.hero` 这个类是从**专题一**继承下来的，它自带两层装饰性伪元素：
     .hero:before  —— 一张「token 之旅」插图（opacity .5）
     .hero:after   —— 一层白色蒙版，`linear-gradient(95deg, …94% → …12%)`
   而专题一是把内容包在 `<div class="wrap">` 里的，靠 `.hero .wrap{z-index:2}` 浮在上面。
   ⭐ 外传的 header 直接放 h1/p，**没有那层 .wrap**，于是拿不到 z-index ——
     标题被压在两层装饰底下，被那层蒙版糊掉。
   ⛔ 而且蒙版是 95deg 的：左侧 94% 不透明、右侧只剩 12% ——&nbsp;
     **所以症状是「左半边特别糊、右半边还能看」**，一眼看去像配色问题，其实是层级问题。
   ⭐ 判据：**看到「同一行字左右清晰度不一样」，先去找方向性渐变，别去调颜色。**
     纯粹的对比度问题不会只糊一半。

   修法：这一页本来就不需要那张插图（它画的是专题一的 token 之旅，
   跟「两颗芯片」毫无关系，而且那几个彩点正好横穿标题）。
   两层一起关掉，只留下 .hero 本身那道蓝到白的底色；再给直接子元素兜一层 z-index。 */
header.hero::before,
header.hero::after { display: none }
header.hero > *    { position: relative; z-index: 2 }
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
  __FAMNAV__
  <div class="kicker">专题二 · 外传　·　L100　·　15 分钟</div>
  <h1>算力强，显存弱 —— 我们把这样一颗芯片用成了什么样</h1>
  <p class="lede">TPU v6e 与扩散模型。<b>先用一条线说清什么样的活配这颗芯片，
    再把我们真跑过的东西摊开：</b>流水线为什么能拆成三段、拆完怎么摆，
    以及<b>五个月里十个模型是怎么一个个跑通的</b>。</p>
  <div class="note info"><span class="t">这十五分钟怎么走</span>
    前七分钟讲<b>判据</b>（一条线 ＋ 你的活落在哪边 ＋ 装不装得下），
    后八分钟讲<b>我们做了什么</b>（三阶段部署 ＋ 十个模型）。
    <br>⛔ <b>全篇不做端到端 benchmark 对比</b> —— 讲的是结构匹配，不是快慢，理由见下。
    <br><em>想看每一步怎么推出来的、以及两颗芯片的显微镜级拆解：<b>L200 精讲版</b>。</em>
  </div>
</header>
""")

a("""
<section id="s零">
  <h2><span class="secno">开场</span>这一讲要回答的两件事</h2>
  <p>专题二立过一条线：<b>算力 ÷ 显存带宽</b> —— 「每搬一个字节，这台机器本来能算多少次」。
    那一讲量出 B200 和 TPU v7 <b>几乎一模一样</b>。</p>
  <p>这一讲把同一把尺子多量一颗，<b>它立刻破了那个「都一样」</b>。然后回答两件事：</p>
  <ul>
    <li><b>你的活配不配这颗芯片</b> —— 一条线就能判，<b>而且跑之前就能判</b>。</li>
    <li><b>配了之后怎么落地</b> —— 流水线在哪儿切、三段各放哪台机器。
      <em>这一半是我们自己趟出来的，不是推的。</em></li>
  </ul>
  <div class="note info"><span class="t">⛔ 先说清不做什么，以及为什么</span>
    <b>不比 benchmark，不谈谁跑得快。</b>屋脊点是<b>结构量</b>，它只说「这台机器的胃口有多大」。
    <br>⚠️ 这不是回避：我们手上唯一一组同模型双平台实测，<b>本身就不支持「v6e 更快」</b>。
    与其挑一组好看的数，不如把边界说死 —— <b>这一讲讲结构匹配。</b>
    <br><em>扩散模型本身的原理在<b>专题十一</b>；每个数怎么推出来的在<b>L200</b>。</em>
  </div>
</section>
""")

a('<section id="s一">')
a('  <h2><span class="secno">一</span>同一把尺子，量四颗芯片</h2>')
a(fig("figx-1",
      "<b>图 X-1</b>　三颗旗舰挤在 295–313，<b>v6e 是 560</b>。"
      "「算力强、显存弱」不是形容词 —— 它就是这个数。"
      "<em>⭐ 这条线<b>纯用官方规格算</b>，不需要任何实测 —— "
      "所以客户还没给你机器的时候，你已经能判断了。</em>"))
a("""
  <div class="note ok"><span class="t">⭐ 这一页只要记住怎么用它</span>
    把你要跑的那段计算也算出一个<b>强度</b>（每搬一个字节实际算了多少次），两个数一比：
    <b>落线右边＝算力受限，落左边＝带宽受限。</b>
    <br>——&nbsp;<b>就这一句。</b>后面所有内容都是在给这句话填数。
  </div>
""")
a('</section>')

a('<section id="s二">')
a('  <h2><span class="secno">二</span>那你的活落在哪一边</h2>')
a('  <p>拿 Wan2.1 的<b>官方配置</b>当场算一遍 —— 不引用任何人的结论。</p>')
a(fig("figx-9",
      "<b>图 X-9</b>　720P、81 帧经 VAE 与 patch 化后是 <b>75,600 个 token</b>；"
      "config 里 <code>window_size = (−1,−1)</code>，<b>不开窗口、全局注意力</b>，"
      "于是<b>七成算力压在 N² 的注意力上</b>。"
      "<em>⭐ 外部锚点：每层 351.3 M × 40 层 ＝ 14.05 B，对上官方标称的 14B —— "
      "参数量能对上，说明这套公式的形状是对的。</em>"))
a('  <p>算出来强度 ≈ <b>75,600</b>，是那条 560 的 <b>135 倍</b>。放回轴上看：</p>')
a(fig("figx-4",
      "<b>图 X-4</b>　一条规则就够：<b>强度 ≈ 同一份权重被多少个「位置」共用</b>。"
      "小 batch 的 decode 落在最左边，prefill 和扩散甩到右边两个数量级。"))
a("""
  <div class="note ok"><span class="t">⭐⭐ 「v6e 适合扩散」的精确说法</span>
    <b>不是它跑扩散更快。</b>是扩散把它的短板<b>挡在了瓶颈之外</b> ——&nbsp;
    强度上万，那根窄带宽根本没参与；
    <b>于是只剩没被挡住的那一项在起作用：算力 918 对 989.5，是 H100 的 93%。</b>
    <br><em>短板不参与，长板打平 —— 这就是「合适」的全部含义。</em>
  </div>
  <div class="note bad"><span class="t">⛔ 反过来同样成立，这条决定别推错场景</span>
    <b>小 batch 的 decode 在轴的最左边</b>，那里比的全是带宽，而 v6e 只有 H100 的 <b>49%</b> ——
    <b>同一颗芯片，那种活它最吃亏。</b>
    <br><em>⚠️ 还有一条：落在算力侧<b>不等于算力就吃满了</b>。我们实测 Wan 的 MFU 约 37%，
    卡在 head_dim 128 对 MXU 的 256 —— 细节在 L200 §三点五。</em>
  </div>
""")
a('</section>')

a('<section id="s三">')
a('  <h2><span class="secno">三</span>一颗装得下吗</h2>')
a(fig("figx-6",
      "<b>图 X-6</b>　扩散这一族在<b>「一颗到几颗」</b>的量级（7–54 GB），"
      "LLM 在「几十颗」（794 GB、1,342 GB）。"
      "<em>⚠️ 图上如实画了两条踩线和超线的 —— <b>只算权重，不含激活。</b></em>"))
a("""
  <div class="note info"><span class="t">同一台机器的两种用法，按这条选</span>
    一台 v6e-8 是 <b>8 颗 × 32 GB ＝ 256 GB</b>。
    <b>Wan2.1 的 28 GB</b> 单颗只剩 4 GB 余量 → <b>TP 摊到 8 颗</b>，每颗 3.5 GB；
    <b>SDXL 只有 7 GB</b> → 一颗装得下，那就<b>开 8 路各生成各的</b>（我们实测 2.40 张/秒）。
    <br>——&nbsp;<b>装不下就把一个模型摊开，装得下就多放几路。</b>
  </div>
""")
a('</section>')

a('<section id="s四">')
a('  <h2><span class="secno">四</span>它的流水线又胖又瘦 —— 所以能拆开部署</h2>')
a('  <p>到这儿为止都在讲「配不配」。下面这一半是<b>我们自己趟出来的</b>：'
  '一次生成从头到尾，数据体积是怎么变的。</p>')
a(fig("figx-10",
      "<b>图 X-10</b>　管子最粗处是 <b>457 GB</b>（注意力分数矩阵，从不落地），"
      "而两段之间真正跨机传的 latent <b>只有 19.4 MB</b>。"
      "<em>⭐ 三个绿色的点是<b>我们跑出来的真实文件</b>，大小逐字节可核 —— "
      "其中 latent 那个：16×21×90×160×4 B ＋ 272 B 头 ＝ 19,353,872，<b>一个字节不差</b>。</em>"))
a("""
  <div class="note ok"><span class="t">⭐⭐ 最细的那一处，正好就是该切开的那一处</span>
    切开流水线的代价，是<b>切口上的数据要搬一趟</b>。所以切在哪，取决于<b>哪儿的数据最少</b>。
    <br>而扩散的形状<b>天然把这个位置摆在了明处</b>：往前是 774 MB 的激活，
    往后是 448 MB 的像素，<b>而 latent 自己只有 19.4 MB</b>。
    <br>⭐ 于是这一刀几乎<b>不要钱</b>：搬一趟 <b>1.5 毫秒</b>，而被切开的那一段要算 <b>229 秒</b> ——
    <b>「能不能切」这个问题在这里根本不成立，只剩「要不要切」。</b>
  </div>
  <div class="note warn"><span class="t">⚠️ 这张图有两处极易读反</span>
    ① 它画的是<b>体积，不是时间</b>：产出那 19.4 MB 的第二段，恰恰占了全程 <b>98%</b> 的时间。
    <br>② 那根 457 GB <b>从不真的存在</b> —— Splash Attention 分块算，一块算完即弃。
    画它是为了说明<b>「不分块就没法跑」</b>。
  </div>
""")
a('</section>')

a('<section id="s五">')
a('  <h2><span class="secno">五</span>拆完怎么摆 —— 三段吃的不是同一种资源</h2>')
a('  <p>切得动只是可行性。真正的收益来自另一件事：<b>这三段吃的根本不是同一种资源。</b></p>')
a(fig("figx-11",
      "<b>图 X-11</b>　上半是三段的资源画像，下半是两种部署拓扑。"
      "<em>⭐ 左边一体化那栏的问题不是「慢」，是"
      "<b>按最馋的那一段配机器，另外两段的钱就白付了</b>。</em>"))
a("""
  <div class="note ok"><span class="t">⭐ 三段的胃口，一句话各一条</span>
    <b>① 文本编码</b>：几乎不吃算力，占全程 1.3%。
    <b>② DiT 去噪</b>：<b>吃算力</b>，占 98.3%，前面算的那 3.26 × 10¹⁷ FLOP 全在这儿。
    <b>③ VAE 解码</b>：<b>吃显存峰值</b>，把 480 万个数展开成 2.24 亿个，只占 0.4%。
    <br>⭐ VAE 还有个特别的形状：<b>预热 80 秒、之后每次 1 秒 —— 80 倍差</b>。
    这种东西<b>天生该做成常驻服务</b>：编译好放着，那 80 秒摊到上万次调用等于零。
  </div>
  <div class="note bad"><span class="t">⛔ 一条顺口的结论，和一个当场推翻它的反例</span>
    最容易得出的结论是「文本编码那么轻，放 CPU 就行」——
    这在 Wan2.1 上成立（用 T5，<b>3 秒</b>）。
    <br><b>但 Flux.2 立刻推翻它</b>：它用 Mistral3，<b>放 CPU 要 30 秒</b>，
    而它的 DiT 在 TPU 上跑完 50 步只要 <b>13.5 秒</b> ——
    <b>那个「轻量」的第一段，反而是全程最慢的一段。</b>
    <br>⭐ 判据不是「哪一段天生该放哪」，而是<b>先量一量它在目标硬件上要多久</b>。
    <em>而分段的价值恰恰在这里：拆开之后，每一段的账才第一次能单独算清楚。</em>
  </div>
""")
a('</section>')

a('<section id="s六">')
a('  <h2><span class="secno">六</span>这套说法，我们自己验过</h2>')
a('  <p>前面五节讲的都是<b>应该怎样</b>。这一页讲<b>实际怎样</b> —— '
  '同一批道理，在<b>十个真模型上跑了五个月</b>之后留下了什么。</p>')
a(fig("figx-12",
      "<b>图 X-12</b>　每一行的起止与提交数都是 <b>git 提交历史直接数出来的</b>，不是凭印象写的。"
      "<em>⭐ 顶部三条色带是移植方法的更替 —— 12 月 10 日那道红线："
      "<b>那天手写 Flax 的文件被删除，第三代从那一刻算起。</b></em>"))
a("""
  <div class="note ok"><span class="t">⭐⭐ 这一页真正的落点不是「我们做了十个」</span>
    看提交数的走向：<b>HunyuanVideo 73 次 → CogVideoX 49 → Wan2.1 44 → SDXL 9 → Real-ESRGAN 3 → Flux.1 2。</b>
    <br>差别<b>不在模型难度</b> —— SDXL 和 Flux.1 都不比 CogVideoX 简单。
    差别在于<b>前面几个是在发明方法，后面几个是在套用方法</b>。
    <br>——&nbsp;<b>所以落点是：接第十一个模型的成本，已经不是前十个的量级了。</b>
    <em>这才是一条工程路线走通的标志。</em>
  </div>
  <div class="note info"><span class="t">这十个模型盖了五种架构，不是凑数</span>
    <b>DiT</b>（Wan2.1 / CogVideoX / HunyuanVideo）· <b>MMDiT</b>（Flux.1 / Flux.2）·
    <b>UNet</b>（SDXL / S3Diff）· <b>MoE</b>（Wan2.2 I2V，总 27B / 激活 14B）·
    <b>纯卷积</b>（Real-ESRGAN，8.8 M 参数）。
    <br>⭐ 最后那个是特意留的：它是唯一一个<b>非 Transformer、非扩散</b>的 ——
    放它进来是<b>看这套框架在完全不同的架构上还成不成立</b>。
    <b>答案是成立的</b>，它同样落在算力侧，只是「位置」从 token 变成了 tile。
    <br>⚠️ 提交数只反映改动次数，<b>不等于工作量或难度</b> ——
    这里只用它做「发明 vs 套用」的量级判断。
  </div>
""")
a('</section>')

a("""
<section id="s收尾">
  <h2><span class="secno">收尾</span>带走三句</h2>
  <p class="landing">① <b>选芯片不是选更强的那颗，是先看你的活落在那根轴的哪一边</b> ——
    而这件事<b>跑之前就能判</b>。
    <br>② <b>扩散的流水线有一道很细的腰（19.4 MB）</b>，
    所以三段可以拆到不同机器上，各配各的资源 —— <b>切口成本是计算量的万分之零点七。</b>
    <br>③ <b>这套说法我们在十个模型上跑了五个月</b>，
    而提交数从 73 掉到 2 —— <b>接下一个模型的成本，已经不是当初的量级了。</b></p>
  <h3>想往下挖：这一讲的每一节在 L200 的哪儿</h3>
  <table>
    <thead><tr><th>本讲</th><th>L200 精讲版</th><th>那边多给了什么</th></tr></thead>
    <tbody>
      <tr><td>§一 一条线，四颗芯片</td><td><b>§一</b></td>
        <td>560 怎么除出来的、H100 那个 989.5 为什么不是 1979、拿 v5p 复现公式</td></tr>
      <tr><td>§二 你的活落在哪一边</td><td><b>§三 ＋ §四</b></td>
        <td>FLOP 全表、<b>window_size 的反事实</b>、锚点当场抓到的一个错、MFU 37% 的原因</td></tr>
      <tr><td>§三 一颗装得下吗</td><td><b>§二点三</b></td>
        <td>那三条「减配」为什么是配套的</td></tr>
      <tr><td>§四 又胖又瘦</td><td><b>§五点一</b></td>
        <td>三个读数的算式、19.4 MB 那个字节级验证</td></tr>
      <tr><td>§五 拆完怎么摆</td><td><b>§五点二 ～ §五点四</b></td>
        <td>资源画像逐项、一个节点摆几路</td></tr>
      <tr><td>§六 我们真的跑过</td><td><b>§六</b></td>
        <td>三代移植路线的逐项对照、十个模型各自验了哪一条断言</td></tr>
      <tr><td><b>本讲没讲的</b></td><td><b>§二点一 ～ §二点四</b></td>
        <td><b>两颗芯片的显微镜级拆解</b>：v6e / H100 逐项对照、SparseCore 的两个职责、二维环面</td></tr>
    </tbody>
  </table>

  <div class="note warn"><span class="t">⚠️ 这一讲刻意留白的地方</span>
    <b>没有端到端性能数。</b>本讲讲结构匹配，不讲快慢 —— 后者要实测，而实测要连口径一起给。
    <br><em>想看扩散模型本身：<b>专题十一</b>。</em>
  </div>
</section>
""")

html = head + '\n<main class="wrap">\n' + "\n".join(BODY).replace("__FAMNAV__", nav("topic-02x.html")) + "\n</main>\n"

# ── 写盘前自检 ──────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(HERE, "tpu-micro"))
from gate import lint_public                                    # noqa: E402
bad = lint_public(html)
assert not bad, "公开页面里出现内部词，已中止写盘：%s" % bad
# ⛔ 防回归：只要页面用了 class="hero"，就必须带上中和它两层装饰的那段 CSS，
#   否则标题会被那层 95deg 白蒙版糊掉（左侧 94% 不透明）。这个坑不报错、只是变糊。
if 'class="hero"' in html:
    assert "header.hero::after { display: none }" in html or \
           "header.hero::after { display:none }" in html or \
           "header.hero::before,\nheader.hero::after { display: none }" in html, \
        "用了 .hero 却没中和它的装饰伪元素 —— 标题会被白蒙版糊掉"

n_fig = html.count('<figure class="fbox fwide"')
assert n_fig == 7, "图数不对：%d（应为 7）" % n_fig

io.open(OUT, "w", encoding="utf-8").write(html)
print("ok  topic-02x.html  %s 字符  %d 图" % (format(len(html), ","), n_fig))
