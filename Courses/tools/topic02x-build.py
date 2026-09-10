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
/* ⭐⭐ 图按**原生宽 1400 封顶**，不再随视口放大。
   ⛔ 2026-09-09 实测：1900px 视口下每张图都被拉到 1.22 倍 ——&nbsp;
     后果有两层，第二层才是要命的：
       ① 页面白白长了两千多像素；
       ② **图里的字跟着放大 22%**，于是「图上字太多」这个观感有一半
          其实是「图上的字被放大了」——&nbsp;而版面 lint 查的是「字号有没有
          越过正文」，1.22 倍刚好没越过，所以它一直没报。
   ⭐ 判据：**SVG 是按某个宽度设计的，就别让它超过那个宽度** ——&nbsp;
     放大不会增加信息，只会让密度看起来更高。
   ⚠️ 只作用在 svg 上，**不碰 figure 自身的 margin** ——&nbsp;
     那个 margin 简写坑（会压掉 .fwide 的 margin-left:50%）见上面那段。 */
figure.fbox > svg { max-width: 1400px; margin-left: auto; margin-right: auto;
                    display: block }
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


def fold(summary, body):
    """把一节的文字折起来，**只在折叠条上留那句结论**。

    ⛔ 关键在 summary 的写法：它不能写成「展开看细节」这种没信息的钩子。
      这一讲是**图为主**的，多数人根本不会点开 ——&nbsp;
      所以 summary 必须是**你在台上会说出口的那句话**，
      不点开也已经把结论拿到了。折叠是给「还想要依据」的人准备的。
    ⭐ 判据：**折起来的东西如果点不点开会改变结论，那就不该折。**
    """
    return ('<details class="more"><summary>%s</summary>\n<div class="body">\n%s\n</div>'
            '</details>' % (summary, body))


a("""
<header class="hero">
  __FAMNAV__
  <div class="kicker">专题二 · 外传　·　L100　·　19 分钟</div>
  <h1>算力强，显存弱 —— 我们把这样一颗芯片用成了什么样</h1>
  <p class="lede">TPU v6e 与扩散模型。<b>前十分钟一条线加两张芯片图，判你的活配不配；
    后八分钟摊开我们真跑过的东西。</b></p>
</header>
""")

a('<section id="s零">')
a('  <h2><span class="secno">开场</span>这一讲要回答的两件事</h2>')
a('  <p><b>① 配不配</b> —— 一条线，<b>跑之前就能判</b>。'
  '<b>② 怎么落地</b> —— 在哪儿切、三段放哪台机器。<em>后一半是我们趟出来的。</em></p>')
a(fold('⛔ 先说清这一讲<b>不</b>做什么 ——&nbsp;不比 benchmark，一个快慢数都没有',
       """    <p>专题二立过一条线：<b>算力 ÷ 显存带宽</b> —— 「每搬一个字节，这台机器本来能算多少次」。
      那一讲量出 B200 和 TPU v7 <b>几乎一模一样</b>；这一讲多量一颗，它立刻破了那个「都一样」。</p>
    <p><b>屋脊点是结构量</b>，它只说「这台机器的胃口有多大」，不说「这顿饭多久吃完」。</p>
    <p>⚠️ 这不是回避：我们手上唯一一组同模型双平台实测，<b>本身就不支持「v6e 更快」</b>。
      与其挑一组好看的数，不如把边界说死 —— <b>这一讲讲结构匹配。</b></p>
    <p><em>扩散模型本身的原理在<b>专题十一</b>；每个数怎么推出来的、以及两颗芯片的
      显微镜级拆解在 <b>L200</b>。</em></p>"""))
a('</section>')

# ⭐ 2026-09-09 追加：把两张显微镜图放回来（现场：「这两个图也挺好看的」）。
#   ⛔ 但**不新开一节**，而是并进 §一 —— 理由是不想动节号：
#     收尾那张「本讲 → L200」对照表、讲稿的分段、L200 的自检，全都按现有节号写着。
#     ⭐ 判据：**能不改编号就别改。**编号是被很多处引用的公共接口，
#       而这两张图跟 §一 本来就是一件事的两半：
#       X-1 给出 560 这个数，X-2 / X-3 说明它是从什么样的硅片布局里长出来的。
#   ⚠️ 代价是主线从 15 分钟变成 18 —— 如实改，不假装还是 15。
a('<section id="s一">')
a('  <h2><span class="secno">一</span>同一把尺子量四颗芯片 —— 再把其中两颗拆开</h2>')
a(fig("figx-1",
      "<b>图 X-1</b>　三颗旗舰挤在 295–313，<b>v6e 是 560</b>。"
      "「算力强、显存弱」不是形容词 —— 它就是这个数。"))
a('  <p>这个 560 从什么样的布局里长出来？<b>先看 v6e 这一颗</b>'
  ' ——&nbsp;<em>H100 的对照放在下面那一折里，需要时再点开。</em></p>')
a(fig("figx-2",
      "<b>图 X-2</b>　一颗 <b>TPU v6e</b>：算力集中在<b>两个 256×256 的方阵</b>里，"
      "片上一整块 128 MiB 暂存（<b>全由编译器安排，没有硬件缓存兜底</b>），"
      "通往片外那道门 1,638 GB/s。"))

# ⛔ 2026-09-10 现场：「H100 拆开看这个不重要，可以先折叠起来。」
#   ⭐ 但按本文件 fold() 的判据 ——「折起来的东西如果点不点开会改变结论，
#     那就不该折」——&nbsp;§一 的结论恰恰**建立在两颗的对照上**。
#   ⚠️ 所以折的是**图**，不是**结论**：把 528 / 硬件自动管的 L2 / 门宽一倍
#     这三样直接写进折叠条，不点开也已经拿到了对照。
#   ⚠️ 连带影响：讲稿里「两张图叠着看那道门」那句，现在要先点开才成立 ——&nbsp;
#     已在讲义与逐字稿里标注。
a(fold('⭐ 对照 · <b>一颗 H100 拆开看</b>（图 X-3，默认折起）——&nbsp;'
       '同样的画法，算力<b>摊成 528 个</b>小单元，'
       '片上多一整层<b>硬件自动管</b>的 L2，<b>门宽一倍</b>',
       fig("figx-3",
           "<b>图 X-3</b>　一颗 <b>H100</b>：同样的画法。算力摊成 <b>528 个</b>小单元，"
           "片上多了一整层<b>硬件自动管</b>的 L2，门宽一倍。"
           "<em>⭐ 两张图里那道门是按带宽<b>等比</b>画的，可以直接叠着比。</em>")))
a(fold('⭐⭐ 两颗都是自洽的：一个在防「我不知道你要跑什么」，'
       '一个在吃透「我早就知道你要跑什么」',
       """    <p>同样三个问题，两边给了<b>相反</b>的答案：</p>
    <table>
      <thead><tr><th>同一个问题</th><th>TPU v6e</th><th>H100</th><th>这个选择在防什么</th></tr></thead>
      <tbody>
        <tr><td><b>算力怎么摆</b></td><td>集中成 2 个 256×256 方阵</td><td>摊成 528 个小单元</td>
          <td>摊开是为了<b>不管来什么形状都有人能接</b>；集中是为了<b>大矩阵上把利用率吃满</b></td></tr>
        <tr><td><b>片上谁做主</b></td><td>128 MiB 全归编译器，<b>CMEM ＝ 0</b></td>
          <td>50 MB L2 由<b>硬件自动</b>管</td>
          <td>硬件自动管是为了<b>应付负载未知</b>；交给编译器是为了<b>吃透负载已知</b></td></tr>
        <tr><td><b>门开多宽</b></td><td>1,638 GB/s</td><td>3.35 TB/s</td>
          <td>门宽一倍，代价是屋脊点低一半 —— <b>它挡的是「强度不够」的活</b></td></tr>
      </tbody>
    </table>
    <p>⭐ H100 那一列全部指向<b>「负载未知」</b>：摊开、缓存兜底、门开大 —— <b>三个都是为不确定性买的保险</b>。
      v6e 那一列全部指向<b>「负载已知」</b>：集中、编译期写死、门只开够用 —— <b>三个都是把保险费省下来换算力密度</b>。
      <br>——&nbsp;<b>所以问题从来不是谁更强，是你手上的活属于哪一种。</b></p>
    <p>⚠️ 那个 <b>CMEM ＝ 0</b> 是有代价的：<b>编译器排错了就没有后手</b>。
      形状动态、访存模式运行时才知道的负载，在这套设计上会很难受
      —— <b>而扩散恰好把这个前提喂得满满的</b>（§四会用到这一条）。</p>
    <p>⚠️ 还有一条：v6e 是 <b>1 颗 ＝ 1 个核 ＝ 1 个 device</b>。
      专题二反复强调的 v7「容量除以 2」那个坑，<b>这一代没有</b>。</p>"""))
a(fold('这条线怎么用，以及它为什么<b>跑之前就能画</b>',
       """    <p>把你要跑的那段计算也算出一个<b>强度</b>（每搬一个字节实际算了多少次），
      两个数一比 —— <b>落线右边＝算力受限，落左边＝带宽受限。</b></p>
    <p>三个比值：跟 H100 比，<b>算力 93%、带宽 49%、容量 40%</b>。
      分子基本没动，分母砍一半，商自然涨到两倍。</p>
    <p>⭐ 分子分母<b>都取官方规格，不取任何实测值</b> —— 所以这条线在跑任何东西之前就能画出来。
      <em>客户还在犹豫要不要给你机器的时候，你已经能下判断了。</em></p>
    <p><em>560 怎么除出来的、H100 那个 989.5 为什么不是 1979、我们拿什么复现过这条公式 —— 见 L200 §一。</em></p>"""))
a('</section>')

a('<section id="s二">')
a('  <h2><span class="secno">二</span>那你的活落在哪一边</h2>')
a('  <p>拿 Wan2.1 的<b>官方配置</b>当场算，不引用结论。</p>')
a(fig("figx-9",
      "<b>图 X-9</b>　720P、81 帧经 VAE 与 patch 化后是 <b>75,600 个 token</b>；"
      "config 里 <code>window_size = (−1,−1)</code>，<b>不开窗口、全局注意力</b>，"
      "于是<b>七成算力压在 N² 的注意力上</b>。"))
a('  <p>强度 ≈ <b>75,600</b>，是 560 的 <b>135 倍</b>。放回轴上：</p>')
a(fig("figx-4",
      "<b>图 X-4</b>　一条规则就够：<b>强度 ≈ 同一份权重被多少个「位置」共用</b>。"
      "小 batch 的 decode 落在最左边，prefill 和扩散甩到右边两个数量级。"))
a(fold('⭐⭐ 于是「v6e 适合扩散」的精确说法是：<b>短板不参与，长板打平</b>',
       """    <p><b>不是它跑扩散更快。</b>是扩散把它的短板<b>挡在了瓶颈之外</b> ——&nbsp;
      强度上万，那根窄带宽根本没参与；<b>只剩没被挡住的那一项在起作用：
      算力 918 对 989.5，是 H100 的 93%。</b></p>
    <p>⭐ 顺带说我们怎么确认自己没算错：同一份 config 把参数量数出来，
      每层 351.3 M × 40 层 ＝ <b>14.05 B</b>，<b>正好对上官方标称的 14B</b>。
      参数量能对上，说明我们对「这一层里有哪些矩阵」的理解是对的。</p>
    <p>⛔ <b>反过来同样成立，这条决定别推错场景</b>：小 batch 的 decode 在轴最左边，
      那里比的全是带宽，而 v6e 只有 H100 的 <b>49%</b> —— <b>那种活它最吃亏。</b></p>
    <p>⚠️ 还有一条：<b>落在算力侧不等于算力就吃满了</b>。
      我们在 <b>Wan2.2</b> 上抓 profile，<b>Splash Attention 那个算子的 MFU 只有 37%</b>
      （roofline 15.974 ms ÷ 实测 43.93 ms），<b>整模型优化后是 34%</b>（基线 12%）——
      卡在 head_dim 128 对 MXU 的 256，方阵有一半是空的。细节在 L200 §三点五。</p>"""))
a('</section>')

a('<section id="s三">')
a('  <h2><span class="secno">三</span>一颗装得下吗</h2>')
a(fig("figx-6",
      "<b>图 X-6</b>　上半是权重体积与我们真跑过的配置；"
      "<b>下半是同一颗芯片上的真实预算</b> —— 那是我们自己撞 OOM 时的实测账。"
      "<em>⛔ <b>权重决定装不装得进，激活决定跑不跑得动</b> —— 而扩散是后者说了算。</em>"))
a(fold('⛔⛔ 「28 GB &lt; 32 GB 所以能跑」这句话<b>分子分母都错</b>',
       """    <p><b>① 分母错了。</b>32 GB 是标称，不是预算。真跑起来权重和运行时先占掉一大块 ——&nbsp;
      我们那次 OOM 时 XLA 报的是：<code>There are 13.10G free</code>，<b>只剩 13.1 GB</b>。</p>
    <p><b>② 分子也错了。</b>要放进去的不只是权重，还有<b>峰值激活</b>。
      而扩散这一族<b>激活比权重大</b>：CogVideoX-5B 权重 10 GB，
      而它的 <b>VAE 解码一步要 19 GB</b>（<code>Attempting to reserve 19.00G</code>）——&nbsp;
      19 &gt; 13.1，于是 OOM。</p>
    <p>⭐⭐ 而且<b>激活随分辨率与帧数涨，权重一个字节不涨</b> ——&nbsp;
      Wan2.1 的 480P 跑得动、720P OOM，用的是<b>同一份权重</b>。
      <b>权重决定装不装得进，激活决定跑不跑得动。</b></p>
    <p>⭐ 解法也不是换更大的卡：逐帧解码 ＋ 共享缓存，把那 19 GB 压到 &lt; 13 GB —— <b>改实现</b>。</p>
    <p>另外两处「按体积猜会猜错」的：</p>
    <p><b>③ S3Diff 只有 6.6 GB</b>，按体积猜「一颗绰绰有余」——&nbsp;对，但那不是重点。
      真正的发现是：<b>我们把它摊到 8 卡做张量并行，实测反而更慢</b>
      （5.46 秒 对 5.28 秒），而预热长了 15 倍。<b>模型太小，通信开销盖过了收益。</b></p>
    <p><b>④ Wan2.1 的 28 GB</b> 按体积猜「贴边能塞进一颗」——&nbsp;
      而实测从来没人这么跑：它是在 <b>v6e-8 上 dp=1、tp=8 摊开</b>跑的。</p>
    <p>⭐ 前者看体积就能答，<b>后者只能实测</b> —— 而我们十个模型每一个都测过。</p>
    <p>同一台 v6e-8 的两种用法（都实测过）：<b>SDXL 7 GB</b> 一颗装得下 →
      <b>开 8 路各生成各的</b>，2.40 张/秒；<b>Wan2.1</b> → <b>TP 摊到 8 颗</b>。</p>"""))

# ⛔ 2026-09-10 现场：「这一堆乱七八糟的字都折叠起来。」
#   ⭐ 查下来最值钱的一条：图里那三条色带**跟上面这个折叠块讲的是同一批话** ——&nbsp;
#     同一个职责两个载体，删掉纯赚，一个字的信息都没少。
#   下面两折是图里**独有**、上面折叠块里没有的那部分，搬出来接住。
a(fold('实测分布：<b>小的单颗，主力清一色 8 卡 ——&nbsp;没有一个需要跨主机</b>',
       """    <p><b>单颗</b>：S3Diff · SDXL（延迟最优）· Real-ESRGAN<br>
      <b>8 卡</b>：HunyuanVideo-1.5 / Wan2.1 / CogVideoX 在 v6e-8 ·
      Flux.2 在 v4-8 · Wan2.2 I2V 在 v6e-16</p>
    <p>⭐ 这正好解释 <b>X-2</b> 里那三条看着像减配的规格
      （4 个 ICI 口、二维环面、Pod 只有 256）：
      <b>v6e 不打「一个模型摊在几千颗上」那场仗 —— 不打，就不用付那个成本。</b></p>"""))
a(fold('这张图的数是从哪儿来的（出处 ＋ 三条边界）',
       """    <p>右列「实测配置」<b>全部取自各模型 README 的测试环境段</b>：
      SDXL v6e-1/4/8 · HunyuanVideo-1.5 / Wan2.1 / CogVideoX 在 v6e-8 ·
      Flux.2 在 v4-8 · Wan2.2 I2V 在 v6e-16（分片配置见该模型优化指南第三章）·
      S3Diff 与 Real-ESRGAN 单颗。</p>
    <p>「8 卡反而更慢」出自 <b>S3Diff README 的 Why Not Multi-Chip 段</b>；
      权重体积按 <b>bf16 每参数 2 字节</b>换算，参数量出自各家官方模型卡。</p>
    <p>⛔ <b>那根轴只算权重</b>，不含激活与编译缓存 ——&nbsp;
      它能回答「装不装得下」，<b>回答不了「该用几颗」</b>。<br>
      ⚠️ <b>FLUX.1 我们没有记录实测配置</b>，图上如实留空。</p>"""))
a('</section>')

a('<section id="s四">')
a('  <h2><span class="secno">四</span>仓库里为什么同一个模型有两份例子</h2>')
a('  <p><b>下面这一半是我们自己趟出来的。</b>仓库里每个扩散模型都有<b>两份例子</b>：'
  '一体化脚本，和一个 <code>*_staged/</code> 目录 —— '
  '<b>后者存在的理由不是更快，是它把切口露在外面。</b></p>')
a("""
  <p>拿 <b>Wan 2.2 图生视频</b>当例子 —— <b>十个模型都是这套布局</b>：
    <a href="https://github.com/yangwhale/gpu-tpu-pedia/tree/main/tpu/Wan2.2">
    github.com/yangwhale/gpu-tpu-pedia/tree/main/tpu/Wan2.2</a></p>
<pre><code>tpu/Wan2.2/
├── generate_i2v_torchax.py                   <b>← ① 一体化：一个进程从头跑到尾</b>
├── generate_diffusers_i2v_torchax_staged/    <b>← ② 三阶段</b>
│   ├── stage1_encoder.py                        文本 ＋ 首帧 → embedding
│   ├── stage2_transformer.py                    五十步去噪 → latent
│   ├── stage3_vae_decoder.py                    latent → 成片
│   ├── utils.py                                 <b>落盘 / 读盘的 helper 全在这</b>
│   └── stage_outputs/                        <b>← ⭐ 切口就在这个目录里</b>
│       ├── stage1_embeddings.safetensors     <b>← 段与段之间唯一交接的，就这三个文件</b>
│       ├── stage2_latents.safetensors
│       ├── generation_config.json
│       └── output_video.mp4                     成片
├── docs/wan_tpu_optimization_guide.md        约 1,970 行迁移与优化指南
└── README.md</code></pre>
  <p class="dim">⚠️ 下面这张图上的字节数取自 <b>Wan2.1</b> 那一份同名目录
    （<code>tpu/Wan2.1/generate_diffusers_torchax_staged/stage_outputs/</code>）——
    两个模型目录结构相同，数不同。</p>
""")
a(fig("figx-10",
      "<b>图 X-10</b>　三阶段之间只交接三个文件：两个 safetensors ＋ 一份 config。"
      "<em>⭐ 图上的字节数、shape、dtype <b>全是直接解 safetensors 文件头得到的</b>，"
      "这三个文件仓库里就有，可自行复核。</em>"))
a(fold('⭐⭐ 拿到一份 latent，<b>三步自检</b>：shape、dtype、字节数 ——&nbsp;'
       '而「看文件大小」一步都不算',
       """    <p><b>① shape 对不对</b>：由分辨率直接推 —— 帧 (81−1)/4+1 ＝ 21、高 720/8 ＝ 90、
      宽 1280/8 ＝ 160、通道 16 → 期望 <code>[1, 16, 21, 90, 160]</code>。
      <b>对不上就别往下跑</b>，后面只会得到全黑或 NaN。</p>
    <p><b>② dtype 在哪看</b>：读 safetensors 头的 dtype 字段，<b>再读 metadata 里的
      <code>dtype_info</code></b>。⭐ 两者可能不一样 —— Wan 的 embedding <b>盘上是 F32，
      而 dtype_info 写着原始是 bfloat16</b>（保存时转的，加载时按这条恢复）。</p>
    <p><b>③ 字节数对不对</b>：16×21×90×160×4 ＋ 272 字节头 ＝ <b>19,353,872</b>，
      跟文件<b>一个字节不差</b>。</p>
    <p>⛔ <b>为什么「看文件大小」不算验证</b>：Wan2.1 的 latents 是 F32、19,353,872 B；
      CogVideoX 的 latents <b>形状一模一样</b>，但 BF16、9,677,064 B ——&nbsp;
      <b>大小差一倍，形状相同。只能读头。</b></p>
    <p>⭐ 顺带说切口有多便宜：跨机要搬的就是那 19 MB 出头，走 100 Gbps 约 <b>1.5 毫秒</b>，
      而被切开的那一段本身要算 <b>229 秒</b> —— <b>「能不能切」这个问题在这里根本不成立。</b></p>"""))
a('<section id="s五">')
a('  <h2><span class="secno">五</span>拆完怎么摆 —— 三段吃的不是同一种资源</h2>')
a('  <p>切得动只是可行性。收益在于：<b>三段吃的不是同一种资源。</b></p>')
a(fig("figx-11",
      "<b>图 X-11</b>　上半是三段的资源画像，下半是两种部署拓扑。"
      "<em>⭐ 左边一体化那栏的问题不是「慢」，是"
      "<b>按最馋的那一段配机器，另外两段的钱就白付了</b>。</em>"))
a(fold('⭐ 三段的胃口各一条，外加一个<b>当场推翻顺口结论的反例</b>',
       """    <p><b>① 文本编码</b>：几乎不吃算力，占全程 1.3%。
      <b>② DiT 去噪</b>：<b>吃算力</b>，占 98.3%。
      <b>③ VAE 解码</b>：<b>吃显存峰值</b>，把 480 万个数展开成 2.24 亿个，只占 0.4%。</p>
    <p>⭐ VAE 还有个特别的形状：<b>预热 80 秒、之后每次 1 秒 —— 80 倍差</b>。
      这种东西<b>天生该做成常驻服务</b>：编译好放着，那 80 秒摊到上万次调用等于零。</p>
    <p>⛔ <b>反例</b>：看完这张图最容易得出「文本编码那么轻，放 CPU 就行」——
      在 Wan2.1 上成立（T5，<b>3 秒</b>）。<b>但 Flux.2 立刻推翻它</b>：
      它用 Mistral3，<b>放 CPU 要 30 秒</b>，而它的 DiT 在 TPU 上跑完 50 步只要 <b>13.5 秒</b> ——&nbsp;
      <b>那个「轻量」的第一段，反而是全程最慢的一段。</b></p>
    <p>⭐ 判据不是「哪一段天生该放哪」，而是<b>先量一量它在目标硬件上要多久</b>。
      <em>分段的价值恰恰在这里：拆开之后，每一段的账才第一次能单独算清楚。</em></p>"""))
a('</section>')

a('<section id="s六">')
a('  <h2><span class="secno">六</span>这套说法，我们自己验过</h2>')
a('  <p>前面是<b>应该怎样</b>，这一页是<b>实际怎样</b> —— <b>十个模型，五个月</b>。</p>')
a(fig("figx-12",
      "<b>图 X-12</b>　每一行的起止与提交数都是 <b>git 提交历史直接数出来的</b>，不是凭印象写的。"
      "<em>⭐ 12 月 10 日那道红线：那天手写 Flax 的文件被删除，第三代从那一刻算起。</em>"))
a(fold('⭐⭐ 真正的落点不是「我们做了十个」，是<b>提交数从 73 掉到 2</b>',
       """    <p><b>HunyuanVideo 73 次 → CogVideoX 49 → Wan2.1 44 → SDXL 9 → Real-ESRGAN 3 → Flux.1 2。</b></p>
    <p>差别<b>不在模型难度</b> —— SDXL 和 Flux.1 都不比 CogVideoX 简单。
      差别在于<b>前面几个是在发明方法，后面几个是在套用方法</b>。
      <br>——&nbsp;<b>所以落点是：接第十一个模型的成本，已经不是前十个的量级了。</b></p>
    <p>⭐ 这十个盖了<b>五种架构</b>，不是凑数：
      <b>DiT</b>（Wan2.1 / CogVideoX / HunyuanVideo）· <b>MMDiT</b>（Flux.1 / Flux.2）·
      <b>UNet</b>（SDXL / S3Diff）· <b>MoE</b>（Wan2.2 I2V）· <b>纯卷积</b>（Real-ESRGAN，8.8 M）。
      最后那个是唯一<b>非 Transformer、非扩散</b>的 —— 放它进来就是看这套框架
      在完全不同的架构上还成不成立。<b>答案是成立的。</b></p>
    <p>⚠️ 提交数只反映改动次数，<b>不等于工作量或难度</b> ——
      这里只用它做「发明 vs 套用」的量级判断。</p>"""))
a('</section>')

a('<section id="s收尾">')
a('  <h2><span class="secno">收尾</span>带走三句</h2>')
# ⛔ 2026-09-09 审计：这三条原来**整句全加粗**（readability lint 报 100%）。
#   ⭐ 全部加粗等于没有加粗 —— 收尾三句是全讲最该被记住的地方，
#     只加粗每条真正的**操作词**，其余留常规体，重音才落得下去。
a("""  <p class="landing">① 先看你的活落在轴的哪一边 —— <b>跑之前就能判</b>。
    <br>② 切口只有 <b>19 MB 出头</b>，所以三段能拆开、各配各的资源。
    <br>③ 十个模型五个月，<b>提交数从 73 掉到 2</b>。</p>""")
a(fold('想往下挖：这一讲的每一节在 L200 的哪儿（含<b>本讲没讲的 SparseCore 与环面</b>）',
       """    <table>
      <thead><tr><th>本讲</th><th>L200 精讲版</th><th>那边多给了什么</th></tr></thead>
      <tbody>
        <tr><td>§一 一条线 ＋ 两颗芯片拆开</td><td><b>§一 ＋ §二点一、二点二</b></td>
          <td>560 怎么除出来的、989.5 为什么不是 1979、拿 v5p 复现公式；两颗芯片的逐项对照表</td></tr>
        <tr><td>§二 你的活落在哪一边</td><td><b>§三 ＋ §四</b></td>
          <td>FLOP 全表、<b>window_size 的反事实</b>、锚点当场抓到的一个错、MFU 37% 的原因</td></tr>
        <tr><td>§三 一颗装得下吗</td><td><b>§二点三</b></td>
          <td>那三条「减配」为什么是配套的</td></tr>
        <tr><td>§四 两份例子 ／ 三步自检</td><td><b>§五点一</b></td>
          <td>切口成本的几笔账、三个落盘产物的完整口径</td></tr>
        <tr><td>§五 拆完怎么摆</td><td><b>§五点二 ～ §五点四</b></td>
          <td>资源画像逐项、一个节点摆几路</td></tr>
        <tr><td>§六 我们真的跑过</td><td><b>§六</b></td>
          <td>三代移植路线的逐项对照、十个模型各自验了哪一条断言</td></tr>
        <tr><td><b>本讲没讲的</b></td><td><b>§二点四</b></td>
          <td><b>SparseCore 的两个职责</b>（它不只给推荐系统用）、<b>二维环面</b>与 Pod 为什么只有 256 颗</td></tr>
      </tbody>
    </table>
    <p>⚠️ 本讲<b>没有端到端性能数</b>：讲的是结构匹配，不是快慢 ——
      后者要实测，而实测要连口径一起给。
      <br><em>想看扩散模型本身：<b>专题十一</b>。</em></p>"""))
a('</section>')


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
assert n_fig == 9, "图数不对：%d（应为 9）" % n_fig

io.open(OUT, "w", encoding="utf-8").write(html)
print("ok  topic-02x.html  %s 字符  %d 图" % (format(len(html), ","), n_fig))
