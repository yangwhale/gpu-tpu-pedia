# 专题七：按 ISSUES-topic-07.md 回修中文源（2026-10-29）

格式：条目 → ✅／⏭ → 改了哪个源文件 → 旧 → 新（依据）。专题七的中文页由私有源构建：
`~/my-private/courses-internal/topic07-public/tools/`（下称 T/，三个构建脚本 `topic07-build.py`／`topic07-build-lecture.py`／`topic07-drills-build.py`，各跑一次预览版、一次 `--public`）
和 `~/my-private/courses-internal/topic07-research/rN/`（下称 R/，截图标注脚本与图的 json）。产物从 `topic07-public/release/` 拷进 `Courses/WebPages/`。
只重建了专题七，没跑 build-all.sh。检查：`R/r10/tools/check.sh release` 与 `check.sh out`，12 个 lint 全部 exit 0、闸门 3 页 0 命中、OCR 审计没有新增敏感项。

结果：43 条，✅ 41、⏭ 2。

## 另外加的：🌐 English 按钮

照专题六的做法加在三页的私有构建脚本里，链到还没发布的英文页：
- 课件 HERO 面包屑（T/topic07-build.py）：`<a class="lecbtn" href="topic-07.en.html" lang="en" hreflang="en">🌐 English</a>`
- 讲义顶栏 tabs（T/topic07-build-lecture.py）：`<a href="topic-07-lecture.en.html" lang="en" hreflang="en">🌐 English</a>`
- 练习面包屑（T/topic07-drills-build.py）：`<a class="lecbtn" href="topic-07-drills.en.html" …>🌐 English</a>`

## 重新渲染的图和视频（像素尺寸都和原来一样）

| 产物（WebPages） | 生成脚本 | 为什么 | 尺寸 |
|---|---|---|---|
| img/topic07-31-opprof-ab.png、33-opstats-top、34-opstats-gmm、35-opstats-rs、36-fp8tk | R/r4/tools/r4-figs.py | #2 标题／面板标签 Hlo → HLO。逐像素比对：只有标题带（3.6 是两个橙色面板标签）变了 | 2488×968、2348×1272、2388×1384、2388×1422、2488×1138，不变 |
| media/topic07-xrec-opprof.mp4 | T/xrec7.py render xrec7-s3-opprof.py | #2 字幕 Hlo → HLO（停 1、停 6） | 2560×1440，71.0 s，停点不变 |
| media/topic07-xrec-mem.mp4 ＋ img/topic07-xrec-mem.png（海报＝末帧） | T/xrec7.py render xrec7-s3-mem.py | #1 「每卡」→「每个 device」，#30「正好是」→「差不多正好是」（停 12） | 2560×1440，142.0 s，停点不变 |
| media/topic07-xrec-graph.mp4 | T/xrec7.py render xrec7-s4-graph.py | #1 字幕「本卡／64 张卡」→「本 device／64 个 device」（停 11、12） | 2560×1440，110.6 s，停点不变 |
| media/topic07-experts.mp4 ＋ img/topic07-experts.png（海报＝首帧） | T/manim/ch07_experts.py（manim -qh，再 libx264 crf 28 压到 1.7 MB） | #1 全部「卡」→ device；字幕超宽断言触发后缩短了三句，见 #1 | 1920×1080，29.8 s，7 停，停点不变 |

内联 SVG（不是位图）跟着重生成：fig7-comms、fig7-experts、fig7-quant、fig7-sources、fig7-lanes、fig7-route。
R/r2 的 2.4A/B、5.2A/B 重跑了 r2-annot.py，PNG 逐像素不变，只有 json 编号表的文字变了。

## 逐条

| # | 结果 | 文件 | 旧 → 新（依据） |
|---|---|---|---|
| 1 | ✅ | T/topic07_s0／s2／s3／s4／s5.py、topic07_lec.py、topic07-drills-build.py、topic07-fig-comms／experts／quant.py、xrec7-s3-mem.py、xrec7-s4-graph.py、manim/ch07_experts.py；R/r5/tools/r5-graph.py、r10/tools/r10-details.py、r6/tools/r6-hlo-notation.py、r13/tools/r12-trace-figs.py 及其 json（r5/figs/t7-41-graph-ab、r10/figs/t7-det-b-done、r13/figs/t7-42-hlo-notation、r13/figs/t7-r12-col） | 「卡」统一改成 device：每卡／每张卡 → 每个 device，64 张卡 → 64 个 device，本卡 → 本 device，卡 0／卡 r → device 0／device r，不出卡 → 不出 device，卡号 → device 号（讲义「卡零」→「device 零」，「九十四点七四 GiB 的卡」→「的 device」）。保留：模型卡、卡在 HBM 带宽上、工具卡、卡片。fig7-experts 出处里的「图里的「卡」＝ fsdp 轴的一份」→「图里 device 的号按 fsdp 轴数（0–63），data 轴上那 2 个 device 拿的专家一样」（HLO sharding `devices=[64,1,1,1,2] last_tile_dim_replicate`：每份在 data 轴 2 个 device 上各一份）。动画字幕变长触发超宽断言，改写成：「192 个专家切在 64 个 device 上，各 3 个；本 device 的 token 要分给全部 192 个专家」「左：本 device 的专家权重…；右：它的 229,376 行…分成 192 组」「reduce-scatter 加总后各留自己那 3 个」，图里标签「本 device 的 token，按专家分成 192 组」（原写法 Pango 把 device token 粘成一个词）、「↓ 189 个梯度散回别的 device，自己留 3 个」（原句加长后左边出画）。课件 `s4` 代码注释「# 本卡 3 个专家量化成 FP8」→「# 量化成 FP8：本 device 那 3 个专家」（改名后和 fig7-quant 标题撞了查重）。构建后三页「卡」只剩上面保留的几种 |
| 2 | ✅ | R/r4/tools/r4-figs.py（标题 3.1/3.3/3.4/3.5，3.6 两个面板标签）；T/topic07_s3.py、topic07-drills-build.py、xrec7-s3-opprof.py | Hlo Op Profile／Hlo Op Stats → HLO Op Profile／HLO Op Stats（xprof 2.23.1 bundle.js `toolsDisplayMap`：`['op_profile','HLO Op Profile']`、`['hlo_stats','HLO Op Stats']`）。图 4.1 标题原本就是「HLO」，OCR 读成 HLo 是噪声。⚠️ 重跑 r4-figs.py 时发现脚本落后于 json：三条 json 说明是 R9 之后手改的，没回填脚本。已把这三条回填进脚本（3.1 ② 的 WASTED 公式 `1 − max(FLOPS 利用率, HBM 利用率)`、3.2 ⑥「对 Pallas kernel 不按精度折算（两列相同）」、3.4 ⑤「4.2–5.0 ms、573–686 TFLOP/s…是 MTP 那一层（4.6）」），再生成的 json 除本次改动外和原来逐字相同 |
| 3 | ✅ | T/topic07-drills-build.py（练习 19） | 「3 &lt; 192 不成立」→「`num_current_groups < num_total_groups` 是 192 &lt; 192，不成立」（backend.py L555 原文） |
| 4 | ✅ | T/topic07-drills-build.py（练习 13） | 13.3 是对的：r13-drills.json `fwd_pairs` 三颗合计少 1.070 ms（0.069＋0.013＋0.988），12.220＋1.070 ＝ 13.29。改的是说法：「3 颗里只有第三颗…快了 0.988 ms。一层少约 13.3 ms」→「3 颗合计少 1.070 ms，几乎全在第三颗 gmm_megablox.16 → gmm.26（快了 0.988 ms），前两颗只差几十微秒。一层少约 13.3 ms（反向 12.220 ＋ 前向 1.070）」，× 78 ≈ 1,037 不变 |
| 5 | ✅ | T/topic07-drills-build.py（练习 17、24） | `<code>all-gather.551／553／555</code>` → `<code>all-gather.551</code>／<code>.553</code>／<code>.555</code>`；同样处理 gather_offload_custom_fusion.79／.81、all-gather.604／.605 |
| 6 | ✅ | T/topic07-build.py | 公开脚手架那段「⛔ 默认**展开**……」注释在本页构建时换成「专题七：默认**折叠**（照 L300 第一章；作者 10-29 拍板）。存储 key 用本页前缀 t7fo:，按 T 或点按钮展开。」（加了断言，脚手架改写法会报错）。共享脚手架 topic03_page.py 不动 |
| 7 | ✅ | T/topic07_s6.py（6.2 表）、topic07_s5.py（5.5 表）、topic07_lec.py（第二节台词） | 「小核上一类通信整块消失」→「同一层小核上两类通信整块消失（收专家的 all-gather、散梯度的 reduce-scatter），HLO 里这两类 12 → 0、6 → 0」；5.5 表与讲义同改「两类」（2.4 正文、fig7-comms：同一层少的是专家 AG ④ 和专家梯度 RS ⑤） |
| 8 | ✅ | T/topic07_s5.py | 「kernel 窄到 1/93」→「kernel 窄到 1/75–1/93」（和 2.4／3.0 一致） |
| 9 | ✅ | T/topic07-drills-build.py | 公开脚手架 build_nav 的悬停提示写死「第 X 节」、条目截 16 字。本页写盘后把吸顶目录换成自己排的：title「开始前 · 先把 7 份 profile 挂上」「第 1 级 · 定位」…「迁移题 · 一 · 换一条通信、一颗 kernel、一份程序、一段代码」；条目「挂上 7 份 profile」「定位」「归因」「对到 HLO 和别的页」「换通信、kernel、程序、代码」「换总账页、显存」，都不截断 |
| 10 | ✅ | T/topic07-drills-build.py（练习 7） | 「Framework op 是 jit(gmm)」→「Framework op name 是 jit(gmm)」（xprof 插件 .so 里的列名） |
| 11 | ✅ | T/topic07-fig-route.py | 「页面名是开源 XProf 2.23.1 首页 Tools 下拉里的名字」→「页面名取开源 XProf 2.23.1 首页 Tools 下拉里名字的简称（如 HLO Op Profile → Op Profile）」 |
| 12 | ✅ | T/topic07-fig-sources.py | 「这里只数第一颗」→「这里只数第一个 TensorCore（/device:TPU:0）和它的两个 SparseCore」 |
| 13 | ✅ | T/topic07-fig-sources.py | 卡片 ⑥「（按时长加权 1.1054）」（原配方的值）→「（改后那份按时长加权 1.1053）」；出处注改成「…（原配方那份里是 1.15789 或 1.0），按事件时长加权平均 1.10537，和原配方 Roofline 页顶的 1.10537 对上；改后那份页顶是 1.10529。两个百分比都出自改后那份：5,674.9% ÷ 1.10529 ＝ 5,134%…」（gmm.27 只在改后那份里；3.2 停表 ① 记的改后 TSM 1.10529） |
| 14 | ✅ | R/r2/tools/r2-annot.py → r2/figs/t7-22a、22b、52a、52b 的 json | 前向「本层 3 个专家矩阵乘（gmm／tgmm）」→「（gmm）」（四张前向图的三颗都是 gmm／gmm_megablox；反向不变）。PNG 逐像素相同 |
| 15 | ✅ | T/topic07-fig-lanes.py | 标题「一颗芯片在 Trace Viewer 里」→「一个 device 在 Trace Viewer 里」 |
| 16 | ✅ | T/topic07_s2.py | 折叠块「80 层里第 0 层是稠密层」→「80 层里最前面那一层是稠密层，其余 79 层是 MoE……本课说「N 号层」都按 MoE 层编号（0–78），不数那层稠密层」；741 ms 那段「有 0 号层的前向／反向」→「有 0 号 MoE 层的前向／反向」（fig7-step 原本就写「0 号 MoE 层」） |
| 17 | ✅ | T/topic07-drills-build.py（练习 1） | 「大 0.1–0.3%」→「分别大 0.04% 和 0.3%」（375,326.7 ÷ 375,160.0 − 1 ＝ 0.044%；11,793.8 ÷ 11,758.2 − 1 ＝ 0.30%） |
| 18 | ✅ | 同 #4 | 和 #4 同一处 |
| 19 | ✅ | T/topic07_s4.py（4.5 折叠块） | 引号里「覆盖 3,582 / 229,376 行」→「覆盖行 3,582 / 229,376」（Hunyuan3-295B-Pretraining/TUNING-v7.md:1098 原文） |
| 20 | ✅ | T/topic07_s3.py（3.3） | 两个范围都对，范围不同：xplane 事件表（/tmp/r0/ev/fp8old.pkl）循环里三条平均 9.13／10.42／10.24 ms，循环外三条 10.62／9.80／9.54 ms。正文写明：「循环里三条（每步各 78 次，每次 9.1–10.4 ms）、循环外三条（每步 1 次，9.5–10.6 ms）」，图 3.5 ② 只说循环里那三条，不改 |
| 21 | ✅ | R/r4/tools/r4-figs.py（图 3.5 ②）；T/topic07_s3.py | 「2354.7」→「2,354.7」；「每核 1137 TFLOP/s」→「1,137」；「每核 2307 TFLOP/s」→「2,307」（格式串加千分位） |
| 22 | ✅ | T/topic07_s3.py（3.5 折叠块） | 两个数都对、也只有这两种 tile：本机重跑台账 B3／H1 的命令，fp8old 是 4,203＋3,380 ＝ 7,583，fp8tk 是 7,511。加注：「（只有这两种，合计 7,583），375 秒那份全是默认值（7,511 处）。「处」是这串 kernel 配置在编译后程序（HloProto）里出现的次数，不是 kernel 的颗数；两份程序不完全一样，总数差 72，本课没逐处对。」 |
| 23 | ⏭ | — | 链接没错。`host=gke-node-1` 指的是公开下载的那 7 份 profile：R11 抹个人信息时把主机名换成了 gke-node-N（topic07_profiles.py，`~/topic07-profiles/fp8old/.../gke-node-1.xplane.pb`）。翻译时本机 :6311 加载的是抹之前的原始 profile，所以那里找不到 gke-node-1；读者照 2.1 下载解压后打开就能对上 |
| 24 | ✅ | T/topic07_s4.py（4.2 折叠块） | 「这两条 all-gather 带 is_spmd_generated=true」→「这三条 all-gather 都带 is_spmd_generated=true」（fp8new HLO：all-gather.537／538／539 三条 bf16[192,…]，都是 `is_spmd_generated="true"`） |
| 25 | ✅ | T/topic07_s3.py（3.6 折叠块） | 3,705 是对的：按未取整的字节算，Stack − 峰值临时变量 3,339.3 ＋ Heap − 入口参数 365.9 ＝ 3,705.2。改成两项都写出来：「差的 3,705 MiB ＝（Stack − 峰值时的临时变量）3,339 ＋（Heap − 入口参数）366（按未取整的字节算；拿上面取整后的数相减会差一两 MiB）」 |
| 26 | ✅ | T/topic07_r12_trace.py | 图 2.3C 是 R13 按 25,494.8–25,501.4（6.6 ms）重拍的（R/r13/tools/r12-trace-shots.py `V_COL`），可「现场打开」还用 R12 的旧窗 25,494.812–25,502.237。改链接的窗口：view_start=25,494.800、view_end=25,501.400；「这一屏只有 6.6 ms」不动 |
| 27 | ✅ | T/topic07_s3.py（3.1 折叠块） | 「差在第二位小数」→「差不到 0.2 个百分点」（53.31／53.39，117.32／117.45） |
| 28 | ✅ | T/topic07_s3.py | 「每核 1137.1 TFLOP/s」→「1,137.15」（格式 `{peak:,.2f}`，和下文 1,137.15 一致） |
| 29 | ✅ | T/topic07_s4.py（4.6 表）、topic07_lec.py（第三节备答）、topic07-drills-build.py（练习 19） | r5-loopfusion.json：两类新 fusion 95.1＋225.6 ＝ 320.7 ms，其中 135.5 ms 是原来那次相加换了类别，补 0 本身约 185 ms；loop fusion 类只多 276 ms，是因为别的 loop fusion 合计少了约 45 ms。4.6 表「净多出来的约 140 ms 才是补 0 本身」→「两类新 fusion 合计 320.7 ms，其中约 135.5 ms 是原来就有的那次相加换了类别…，剩下约 185 ms 才是补 0 本身；loop fusion 这一类只多 276 ms，是因为别的 loop fusion 合计少了约 45 ms（有增有减，本课没逐个追）」；4.6 落点句「多出来的 276 ms，就是…」→「多出来的时间，大头就是…」；讲义备答和练习 19 同改 |
| 30 | ✅ | T/topic07_s3.py（3.6 折叠块）、topic07_lec.py、xrec7-s3-mem.py（停 12 字幕）；R/r13/tools/r12-memfigs.py 与 r13/figs/t7-38-memviewer.json（图 3.8 ②） | 「正好一半」→「差不多正好一半」：图 3.8 ②「BF16 差不多正好一半，多 4.4 MiB」，3.6 折叠块「差不多正好是 FP8 的一半（多 4.4 MiB）」（22,271.03 − 44,533.22 ÷ 2 ＝ 4.42），讲义、录屏字幕「差不多正好（是 FP8 的）一半」 |
| 31 | ✅ | T/topic07-fig-route.py | 「TensorCore Idle」→「TensorCore idle」（插件 .so 里是 `TensorCore idle (in ms)`） |
| 32 | ✅ | T/topic07_s4.py（4.5 代码块 2 注释） | `jnp.where(valid_mask, out, 0)` → `jnp.where(valid_mask[:, None], out, 0)`（r13-drills.json 记的 backend.py L281 原文） |
| 33 | ✅ | T/topic07-drills-build.py（练习 12） | 「（正文 2.3 折叠块）」→「（正文 2.2 折叠块）」（「79 层为什么循环只转 78 圈」在 2.2） |
| 34 | ✅ | T/topic07-drills-build.py（坑）；顺手 T/topic07_s3.py 3.3 现场打开 | 「Total self time、Model GFLOP/s 等列」→「Total self time (us)、Model GFLOP/s 等列」；3.3 的「Avg. self time」→「Avg. self time (us)」（插件 .so 里的列名，另有 Total self time (%)） |
| 35 | ✅ | T/topic07_s6.py（6.1 症状表第 7 行） | 「2.3C：」→「图 2.3C：」 |
| 36 | ✅ | T/topic07_s2.py（741 ms 那段，MTP 第一次出现在正文） | 「8 个专家矩阵乘（MTP 层，4.6 核过）」→「（MTP 层，4.6 核过；MTP 是多 token 预测，主干后面多接的一层，3.3 再讲）」。fig7-step 格子太窄没加 |
| 37 | ✅ | T/topic07_s3.py（3.6，正文里 drhs 第一次出现） | 「之前的那个 drhs」后加「（专家权重的梯度；矩阵乘 lhs × rhs 里 rhs 是权重，d 表示梯度）」 |
| 38 | ✅ | T/topic07_lec.py（时间账） | 按停点 json 重算（r3/anim/t7-xrec-open.json 停 3–7 ＝ 40.0 s，r8/anim/t7-agdone.json 停 1–5 ≈ 20.6 s）：「约 4.8 分钟：第二节…约 0.9」→「约 4.9 分钟：第二节…约 1.0」。第二节 15.7′＋1.0′ 仍取整 17′，合计仍约 82′，计划表不变 |
| 39 | ✅ | T/topic07_lec.py（第三节台词） | 「三千六百一十二」→「三千六百一十一」（3,668.8 × 189 ÷ 192 ＝ 3,611.475，页面写 3,611 是对的） |
| 40 | ✅ | T/topic07_lec.py（第三节台词） | 「默认的一百二十八见方」→「默认的 128×128×128」 |
| 41 | ✅ | T/topic07_lec.py（第二节台词）；T/topic07_r12_trace.py（Overview 的 SparseCore 字段表与折叠块） | 一个「一点二毫秒」说了两个不同的数：「这一点二毫秒藏住了」→「一点二三」，「白等一点二毫秒」→「一点一八」（图 2.6：藏住 1.23、露出 1.18）。页面表格 8606.2／5234.02（直接贴数据接口的字符串）→ 8,606.2／5,234.02 等（保留原小数位，整数部分加千分位，表和折叠块里的算式都改）。「改后四十二」（41.6）、「六成八」（68.3%）、「一万一千七百九十四」（11,793.8）是正常取整，不改 |
| 42 | ⏭ | — | 都是正常取整：3,981.54 → 三千九百八十一；1,013.7 四舍五入就是一千零一十四（不是 1,014.8）；1.214 → 一点二一，1.528 → 一点五三。口语稿不念小数，不改 |
| 43 | ✅ | T/topic07_lec.py（第零节备答） | 「模型卡写 Total 295B（含 MTP 层 3.8B）」→「模型卡把 Total 295B 和 MTP 层 3.8B 分两行列（层数那一行写的是 80 层、不含 MTP 层）；第三节用显存反算出 2,988.5 亿，正好是两者相加」（jina 读 huggingface.co/tencent/Hy3：Total Parameters 295B、MTP Layer Parameters 3.8B、Number of Layers (excluding MTP layer) 80；卡上没写 295B 含不含 MTP，课件 3.6 的「主体 2,950 亿 ＋ MTP 38 亿」与反算吻合）。顺手：3.6 外部锚点「2988.5 亿」→「2,988.5 亿」 |

## 英文侧要同步的

- **全页**：#1 改了约 160 处「卡」，英文本来就用 device，正文基本不用动；但 fig7-experts 出处注、动画三句字幕和两个图内标签换了说法（见 #1），英文的 experts 动画字幕表（`*-en.py`／t7-experts.en.json）按新中文核对一次
- **改了字幕的录屏**：xrec-opprof（停 1、6）、xrec-mem（停 12）、xrec-graph（停 11、12）、experts（停 1、2、4、5），四段的 `aria-label` 都会变；停点时刻没变
- **改了位图**：图 3.1、3.3–3.6 的标题（HLO），英文版图本来就写 HLO，不用重出
- **新增或改写的句子**：#4、#13、#16、#20、#22、#25、#29、#30、#36、#37 加了解释或改了数，英文要照新中文补；#9 练习页吸顶目录的 title 和条目文字变了
- **讲义**：#38 时间账 4.9／1.0（floor_en_topic07.py 已经按 4.9 写，核对一下）、#39、#40、#41、#43
