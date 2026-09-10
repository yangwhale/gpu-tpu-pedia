# -*- coding: utf-8 -*-
r"""外传 图 X-10 · **仓库里为什么同一个模型有两份例子** ——&nbsp;以及怎么用好三阶段那份。

════════════════════════════════════════════════════════════════════
⛔⛔ 这张图推翻重写过一次，原因值得原样留着
════════════════════════════════════════════════════════════════════
初版画的是「一次生成的数据体积摊在对数轴上」，其中最粗的一根柱子标着
**457 GB —— 注意力分数矩阵**，旁边注一句「从不落地」。

现场当场否掉：

    「中间那个注意力矩阵**从来就没有被物化出来过**，
      都是用的 Flash Attention 或者 Sparse Attention 分块计算的，
      所以不可以这么弄。」

⭐ 他是对的，而且我那句「从不落地」的注解**并不能救它**：
  一旦把一个从不存在的量画成「管子最粗的地方」，
  **整张图的比例尺就建立在一个虚构的锚点上** ——&nbsp;
  读者记住的是那根柱子，不是柱子旁边那行小字。
  ⛔ **注解抵消不了图形本身的断言。** 图上画了，就是说它存在。

════════════════════════════════════════════════════════════════════
⭐⭐ 重写后这张图要回答的，是现场真正想讲的那件事
════════════════════════════════════════════════════════════════════
    「这个部分要表达的是我们 GitHub repository 里的例子，
      为什么要分单体端到端跑通的、和分三个阶段的。
      那三个阶段是为了展示怎么把一个模型拆开、
      怎么在中间的 latent space 传递数据，
      以及**这份 latent 的格式、维度、shape 怎么去验证**。
      就是怎么用好 GitHub 里边的例子。」

→ 所以这一张不再讲「体积有多大」，改讲**切口上到底交接了什么、怎么核对它**。

════════════════════════════════════════════════════════════════════
📌 图上每个数的出处：直接解 safetensors 文件头，不是算的
════════════════════════════════════════════════════════════════════
文件就在 `tpu/Wan2.1/generate_diffusers_torchax_staged/stage_outputs/`：

  · stage1_embeddings.safetensors　7,406,544 B　头 976 B
      prompt_embeds / negative_prompt_embeds　各 **F32 [1, 226, 4096]**
      ⭐ metadata 里 `dtype_info = {"prompt_embeds": "bfloat16", ...}`
        ——&nbsp;**盘上是 F32，原始是 bf16**（保存时转的，加载时按这条恢复）
  · stage2_latents.safetensors　19,353,872 B　头 272 B
      latents　**F32 [1, 16, 21, 90, 160]**
  · generation_config.json　817 B　——&nbsp;分辨率 / 帧数 / 步数 / seed / model_id

⭐⭐ 最值钱的一个对照（同一个 shape，两种 dtype）：
    Wan2.1     latents  F32   [1,16,21,90,160]  → 19,353,872 B
    CogVideoX  latents  BF16  [1,16,21,90,160]  →  9,677,064 B
  **文件大小差一倍，形状一模一样** ——&nbsp;所以**不能靠文件大小反推 dtype**，
  只能读头。这条正是「怎么验证」那一栏存在的理由。

⚠️ 顺带记一个**口径不一致**，本图不展开但不要忘：
  官方 `wan_t2v_14B.py` 写 `text_len = 512`，而 diffusers 这条路实际存下来的
  文本 embedding 是 **226**。交叉注意力只占总算力约 5%，不影响 X-9 的结论，
  但「config 写的」和「实际跑的」在这里确实不是一个数。
"""
import re

from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400


def wrap(sfull, w, size=11.5):
    """按像素宽折行，⛔ 不切断英文 / 数字串（会断出「CogVideo / X」那种）。"""
    units = re.findall(r"[A-Za-z0-9_.,\[\]()+\-/×＝]+|\s+|.", sfull)
    out, cur = [], ""
    for u in units:
        if wpx(re.sub(r"<[^>]+>", "", cur + u), size) > w and cur:
            out.append(cur.rstrip())
            cur = u.lstrip()
        else:
            cur += u
    if cur.strip():
        out.append(cur.rstrip())
    return out


# 切口上真正落盘的三样东西（全部读自文件头）
ARTS = (
    ("stage1_embeddings.safetensors", "7,406,544 B", GR, "#0d652d",
     ("prompt_embeds　　F32 [1, 226, 4096]",
      "negative_prompt_embeds　F32 [1, 226, 4096]",
      "⭐ metadata 的 dtype_info 记着：原始是 bfloat16")),
    ("stage2_latents.safetensors", "19,353,872 B", BL, "#174ea6",
     ("latents　F32 [1, 16, 21, 90, 160]",
      "＝ 批 1 · 通道 16 · 帧 21 · 高 90 · 宽 160",
      "⭐ 跨机时唯一要搬的就是这一份")),
    ("generation_config.json", "817 B", PU, "#681da8",
     ("height 720 · width 1280 · num_frames 81",
      "steps 50 · guidance 5.0 · seed 2025 · model_id",
      "⭐ 没有它，前两个文件无法自解释")),
)


def main():
    f = Fig(W, "仓库里同一个模型有两份例子：一体化脚本一个进程跑完，"
               "三阶段版本拆成三个独立进程，中间用两个 safetensors 文件和一份 json 交接。"
               "图上给出这三个文件的真实字节数、张量形状与 dtype，"
               "以及验证一份 latent 是否正确的三步自检")
    f.marks = set()
    y = f.header(
        '仓库里为什么同一个模型有两份例子 ——&#160;'
        '<tspan font-weight="700">三阶段那份是拿来「看得见中间」的</tspan>',
        '⭐ 一体化跑得快，但你<tspan font-weight="700">看不见中间那份 latent 长什么样</tspan>。'
        '三阶段把切口露出来：两个 safetensors ＋ 一份 config ——&#160;'
        '<tspan font-weight="700">下面这些数全是直接解文件头得到的，不是算的。</tspan>',
        [(GY2, "一体化：一个进程"), (BL, "三阶段：三个进程 ＋ 落盘交接"),
         (GR, "可逐字节核对")])

    # ══════════════════ 上：两份例子并排 ══════════════════
    top = y + 6
    LW = 330
    RW = W - LW - 26
    RX = LW + 26
    # ⛔ 高度算出来：标题栏 30 ＋ 上边距 22 ＋ 三张卡 3×92 ＋ 结论行 26 ＋ 下沿 14
    PH = 30 + 22 + 3 * 92 + 26 + 14

    ly = f.panel(0, top, LW, PH, "① 一体化", GY2,
                 sub="generate_torchax.py", tag="出片 / benchmark")
    f.box(16, ly + 22, LW - 32, PH - 108, "#fafbfc", LINE2, 8, 1, dash="4 4")
    f.lines(32, ly + 48, LW - 64, [
        "一个进程，从 prompt 直接到成片。",
        "",
        "<tspan font-weight=\"700\">看不见中间态</tspan>：文本 embedding、",
        "latent 全在内存里，跑完就没了。",
        "",
        "⭐ 它的用处是<tspan font-weight=\"700\">快</tspan> ——&#160;验证一次",
        "改动、量一次端到端，用这个。",
        "",
        "⛔ 但一出问题（视频全黑、出 NaN），",
        "<tspan font-weight=\"700\">你没有任何中间产物可看</tspan>。",
    ], size=11.5, lh=19, fill=GY)
    f.t(32, ly + PH - 62, "⛔ 调试时它帮不上忙", "#a50e0e", bold=True, size=_sz(12))

    ry = f.panel(RX, top, RW, PH, "② 三阶段 ——&#160;切口露在外面", BL,
                 sub="generate_diffusers_torchax_staged/", tag="调试 / 部署 / 教学")
    for k, (nm, size_, col, dark, rows) in enumerate(ARTS):
        yy = ry + 22 + k * 92
        f.box(RX + 16, yy, RW - 32, 80, "#fff", col, 6, 1.5)
        f.box(RX + 16, yy, 4, 80, col, col, 2)
        f.t(RX + 34, yy + 22, "stage%d ⏷　%s" % (k + 1, nm), dark,
            bold=True, size=_sz(12.5))
        f.t(RX + RW - 30, yy + 22, size_, dark, bold=True,
            size=_sz(12.5), anchor="end")
        f.lines(RX + 34, yy + 42, RW - 70, list(rows),
                size=11.5, lh=17, fill=GY)
    f.t(RX + 16, ry + PH - 62,
        "⭐ <tspan font-weight=\"700\">三段之间只认这三个文件、不认进程</tspan>"
        "——&#160;所以它们天然可以跑在三台机器上",
        "#174ea6", size=_sz(12), w=RW - 32)

    y = top + PH + 24

    # ══════════════════ 中：怎么验证那份 latent ══════════════════
    ROWS = (
        ("① shape 对不对",
         "由分辨率直接推：帧 (81−1)/4+1 ＝ 21 · 高 720/8 ＝ 90 · "
         "宽 1280/8 ＝ 160 · 通道 16",
         "期望 [1, 16, 21, 90, 160]。"
         "<tspan font-weight=\"700\">对不上就别往下跑</tspan> ——&#160;"
         "后面只会得到全黑或 NaN"),
        ("② dtype 在哪看",
         "读 safetensors 头的 dtype 字段，"
         "<tspan font-weight=\"700\">再读 metadata 里的 dtype_info</tspan>",
         "⭐ 两者可能<tspan font-weight=\"700\">不一样</tspan>："
         "Wan 的 embedding 盘上是 F32，而 dtype_info 写着原始是 bfloat16"),
        ("③ 字节数对不对",
         "元素数 × 每元素字节 ＋ 头 ＝ 文件大小",
         "16×21×90×160×4 ＋ 272 ＝ "
         "<tspan font-weight=\"700\">19,353,872</tspan>，跟文件"
         "<tspan font-weight=\"700\">一个字节不差</tspan>"),
    )
    hy = y
    TH = 34 + len(ROWS) * 54
    f.box(0, hy, W, TH, "#fff", LINE, 8)
    f.colhead(14, hy + 22, "拿到一份 latent，三步自检")
    f.colhead(250, hy + 22, "怎么做")
    f.colhead(740, hy + 22, "看什么")
    f.line(0, hy + 34, W, hy + 34, LINE, 1, arrow=False)
    for i, (k, how, note) in enumerate(ROWS):
        yy = hy + 34 + i * 54
        if i:
            f.line(0, yy, W, yy, LINE2, 1, arrow=False)
        f.t(14, yy + 24, k, INK, bold=True, size=_sz(12))
        f.lines(250, yy + 22, 470, wrap(how, 466), size=11.5, lh=17, fill=GY)
        f.lines(740, yy + 22, W - 754, wrap(note, W - 756),
                size=11.5, lh=17, fill=GY)
    y = hy + TH + 22

    # ══════════════════ 落点 ══════════════════
    # ⛔⛔ 2026-09-10 现场：「把这些给我折叠起来。」
    #   原来这里还有**三条色带 ＋ 三行出处**，跟 X-6 那次是同一个毛病：
    #   散文写进了 SVG，而 **SVG 里折不了东西**。
    #   ⭐ 三条里第一条（「看文件大小」不算验证）**跟图下面那个折叠块完全重复** ——
    #     同一个职责两个载体，删掉纯赚。另外两条和出处是图里独有的，
    #     已搬到 topic02x-build.py 里图下方的 <details>，结论句留在折叠条上。
    #   ⚠️ 尤其是「这张图推翻重写过一次 / 457 GB」那条 —— 它是全讲最有说服力的
    #     一段，**搬走不等于删掉**，它现在是一个独立折叠，折叠条上就写着落点。
    #   ⭐ 图里只留结构本身：两份例子的对照 ＋ 三步自检表。
    f.save("figx-10.svg", y + 6)


main()
