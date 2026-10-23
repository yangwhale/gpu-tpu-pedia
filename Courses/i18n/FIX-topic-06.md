# 专题六：按 ISSUES-topic-06.md 回修中文源（2026-10-23）

格式：条目 → 文件 → 旧 → 新。路径相对 `Courses/tools/`。只跑了专题六的构建（图脚本 → topic06-build.py → topic06-build-lecture.py → topic06-build-L200.py → topic06-build-L200-lecture.py），没跑 build-all.sh。四页 `<!--lec:s` 都是 0，和 HEAD 一样，不用补录音条。

## 课件

| # | 文件 | 旧 | 新 |
|---|---|---|---|
| 1 | topic06-fig-kvq.py（aria、右面板注、出处）；topic06_s12.py（L300 正文）；topic06_lec_5.py（L300 台词） | 「V4.1-Flash 主 KV 原生存 FP4」；面板注「没缩满一半的那截，原文没拆」 | aria「KV 有一份原生存 FP4（另一份仍是 FP8）…多出来的是缩放」；出处「模型卡称主 KV 存 FP4…但 README 写这个格式只能用于 extra_k_cache，k_cache 那份仍是 FP8。每 token FP4 256＋32＝288 B，FP8 512＋16＝528 B，小 45%（本课推导）」；面板注「没缩满一半的那截是缩放：每 16 个数一个，缩放字节反比 FP8 多一倍」。依据：FlashMLA README（jina 当场读过）。README 没写 extra_k_cache / k_cache 各对应哪种注意力，所以没写「压缩注意力／滑窗」 |
| 2 | topic06_L200_s8.py | R1 专家 FP8 每个 device 要约 102 GB，放不进 95 GB；FP4 约 61 GB | 约 101.8 GB（R1 FP4 部署 README 的记录），可用的 94.75 GiB 只合约 101.7 GB，激活和 KV 没地方放；FP4 约 60.9 GB（README:58、72-73） |
| 3 | topic06_L200_s9.py | 同一组实测，先换 DEP 每卡 2.09 倍，只调并发涨 28% | 还是第四节那套 vLLM 实测，先换 DEP 每卡 2.09 倍，只调并发（256 → 512）才涨 28%（见完整版第十节）。核过 VLLM-V4PRO-RUNBOOK.md:446-489：28% 是 3p1d 并发 256→512（16,459→21,100），和 2.09× 出自同一份 vLLM 记录（GB300），跟 SGLang 那组无关 |
| 4 | topic06-fig-spec.py（conc 出处）；topic06_s16.py | 「正文里另两组数的出处」；「接受长度 1.88 → 2.36，单步 +41%」 | 「另两组相关实测的出处」；「接受长度 1.88 → 2.36（+26%），反推单步耗时 115.3 → 162.5 ms（+41%）」（SGLANG-V4PRO-RUNBOOK.md:900-905） |
| 5 | topic06-fig-pd.py | 两两同代对比（论文原文） | 论文拿规模相近的模型两两对比（原话 dense models of similar size）（arXiv 2604.15039 HTML 核过） |
| 6 | — | ⏭ | 论文正文两处写的就是 32% |
| 7 | topic06-fig-seats.py（aria） | 只把 KV 池多分 13% 显存 | 只把 KV 池的显存比例从 0.85 调到 0.88 |
| 8 | topic06-fig-dep.py（fig-s6-dep-kv aria） | 再多给 13% 显存坐到 999 个，只到 10704，算力已到顶 | 再把 KV 池的显存比例从 0.85 调到 0.88，坐到 999 个（多 13%），只到 10704：一步从 77.7 拖到 85.3 毫秒，多坐的人被抵掉 |
| 9 | topic06_L200_s4.py | 是官方目标的 95.6% | 是官方目标（PyTorch 博客报的每卡约 11,200，分母官方没写明，本课按 decode 卡读）的 95.6% |
| 10 | topic06_L200_s4.py | DEP8 一加载就显存不足，官方改推 TP | DEP8 一加载就显存不足（我们实测两次）；SGLang 官方的 K3 配方用的也是 TP、DCP（SGLANG-K3-RUNBOOK.md:120、966、1048：cookbook 硬件口径 TP8 / DCP8） |
| 11 | topic06-fig-spec.py（fig-s11-dspark） | 闲聊比数学少收两个多字 | 闲聊比数学少收约两个字（三家差 2.08 / 1.80 / 2.15） |
| 12 | topic06-fig-phases.py | ÷ 2·(7168·2048 ＋ …) | ÷ [2·(7168·2048 ＋ …)]（复算 7.96） |
| 13 | topic06_L200_s0.py；topic06_L200_lec_s0.py（备注） | 片上存储只有 128 MiB，约显存的 1/1,500 | 一颗芯片的片上存储只有 128 MiB，约整颗芯片显存（192 GiB）的 1/1,500；讲义备注补「按一个 device 算也一样：64 MiB 对 94.75 GiB」 |
| 14 | topic06_L200_s5.py | 首字中位数 32.1 s 对 2.3 s | 首字中位数 TP8 32.1 s，DCP8 2.3 s |
| 15 | — | ⏭ | config.json 的 ignore 列了 shared_experts，INT4 确实只给路由专家 |
| 16 | topic06-fig-kv.py（出处、注释） | 可用量与第十节 10.2 同一口径…实际更少（10.2） | 删掉两处 10.2 指向，只留「先按权重全切粗算…实际更少」 |
| 17 | — | ⏭ | 专题六全篇统一用 94.75（所引 README / 日志原值），改了就和出处对不上 |
| 18 | topic06-fig-spec.py（fig-s11-tree 出处） | 没画进图的（正文里提到）：…；注：各家倍数见正文列表；… | 没画进图的：…；注：各家倍数都是原作者在自己的模型和负载上测的，口径不同，不横比，所以图上不画 |
| 19 | topic06-fig-spec.py（TREE） | 出处 "EAGLE 2401.15077、…" 拼出「EAGLE EAGLE」 | "2401.15077、EAGLE-2 2406.16858" |
| 20 | topic06_L200_s6.py | 「够并几台」是 100 Gbps 除以每台外送，L300 的推导 | …100 Gb/s 除以每台外送，本课推导 |
| 21 | topic06_L200_s9.py | 蓝＝分给更多人 | 蓝＝分给更多 token（跟图例一致） |
| 22 | topic06-fig-chain.py（aria） | ⑨ KV 分层下放， | ⑨ KV 分层下放（到 2025）， |
| 23 | topic06_L200_s1.py（动画 aria） | 工作台里出现出 1 个字 | 工作台里出现「出 1 个字」（动画里的字样就是「出 1 个字」） |
| 24 | topic06-fig-chain.py（出处） | NVIDIA Dory 4B | NVIDIA-Nemotron-Labs-Dory1.0-4B |
| 25 | topic06-fig-pd.py（19 处）；topic06-fig-ratio.py（2P1D 百分比） | Gbps；-31% / -28% / -48% | Gb/s；−31% / −28% / −48% |
| 26 | topic06-fig-dep.py（出处） | 2.09 倍 ＝ (55,153 ÷ 20) ÷ (21,100 ÷ 16)， | 后面补「55,153、21,100 是同表的总吞吐（输入＋输出，4K 进 1K 出，所以约是输出的 5 倍）」 |
| 27 | — | ⏭ | 论文写 18.4%，脚本用未舍入均值复算并 assert 18.4 |
| 28 | topic06-fig-kv.py（fig-s4-batch 顶条） | 灰段一行写「留 10%：9.5 GiB」 | 灰段拆两行：「留 10%」(14px) / 「9.5 GiB」(13px) |

## 讲义

| # | 文件 | 旧 | 新 |
|---|---|---|---|
| L1 | topic06_L200_lec.py（TIME_NOTE） | 地板含互动约 86 分钟…每节都比计划少半分钟左右 | 约 82 分钟（每次 build 现数，以表为准）…各节余量不一：第七节最多，两分多钟；第一、五节一分多钟；第六、八节含互动已略超计划，那两节的追问留到课后 |
| L2 | topic06_L200_lec_s5.py（备答） | 开 DCP 时 symm-mem 用不了 | 开 DCP 时 SGLang 对 K3 把 symm-mem 强制关掉（模型设置，不是 DCP 机制上用不了） |
| L3 | topic06_L200_lec_s8.py（台词＋R9 注） | 台词「右边那个原文没拆为什么」；注「没缩满一半那截原文没拆，口播只说『缩放也占字节』」 | 台词「右边那个多出来的是缩放」；注写明 288 / 528 B 的拆法，外加「FP4 只能用于 extra_k_cache，被问别说整份 KV 都是 FP4」（同第 1 条） |
| L4 | topic06_L200_lec_s9.py（警示注） | +28% 来自完整版第十节同一组 GB300 实测，不在第四节正文里 | +28% 和 2.09 倍出自同一份 vLLM 实测记录（GB300、V4-Pro，runbook :446-489），所以台词说「第四节那组」；第四节正文没写 28%，课件标了见完整版第十节 |
| L5 | topic06_L200_lec_s3.py（警示注） | 94.75 GiB 减十六分之一权重 ≈ 46.2 GiB | 94.75 GiB 先留 10% 给激活，再减十六分之一权重：94.75 × 0.9 − 39.1 ≈ 46.2 GiB；课件那边的 §10.2 指向已删（第 16 条），「指向完整版第四节」不再打架 |
| L6 | topic06_L200_lec_s2.py（警示注） | 每卡人数 886 → 999 | 每个 rank 同时在跑的请求 886 → 999（dep8 一个 rank 就是一张卡，口播说「每卡人数」） |
| L7 | topic06_L200_lec_s4.py（台词＋注） | 目标是官方报的每张 decode 卡一万一千二 | 目标是官方博客报的每卡一万一千二；注补「分母没写明，本课按 decode 卡读，被问别说成官方口径」 |
| L8 | topic06_L200_lec_s6.py（警示注） | 公式约 10 | 公式约 9.8 |
| L9 | topic06_L200_lec_s0–s9.py、topic06_L200_lec.py（渲染出来的文字，不动 # 注释） | R2 / R6 / R8 / R9 / R13–R26 / R91 / R102 等轮次标记，「L300 第一节讲义 DONT」 | 共删 29 处：「R22 口径：」→「口径：」，「（R22 补）」删掉，「R6 从口播挪来」→「从口播挪来」，「s16 R102 行」→「出处表」，「完整版台账 R91 已改定性」→「完整版台账已改定性」，THIN 列表去掉 R 编号，「DONT」→「的备注」。DeepSeek-R1 的 R1 不动 |
| L10 | topic06_L200_lec_s0.py | 「同一个模型」，没有切屏提示 | 加一条 board「滚到表下的蓝框『先猜一个数』，指着框里的 DeepSeek-V4-Pro」；台词改「同一个模型 DeepSeek-V4-Pro」 |

## 构建与核对

- 图脚本 29 个全部 ok，没有面板越界告警；四个构建 rc=0；L300 / L200 课件「AI 味预算：达标」「查重通过」。
- 生成页对重建前快照逐句 diff：只出现上表的改动。另有三处连带变化：topic-06-L300.html 和 topic-06-L300-lecture.html 各多一句 FP4 补充；L300 讲义一个地板数 3,071 → 3,087；topic-06-lecture.html 第零节地板 6.1′ → 6.4′（多了一条 board）。
- 第六、八节含互动仍超计划 0.2′ / 0.1′（构建只报告，没删内容）。
- `i18n/tools/dom.py` 跑 topic-06.html、topic-06-lecture.html：控制台没有报错。
- git status：除了 tools/ 里的源脚本，只多出 4 个生成页（topic-06.html、topic-06-lecture.html、topic-06-L300.html、topic-06-L300-lecture.html）；没提交。

## 同步英文时要注意

- 第 1 条故意没写「压缩注意力／滑窗」，因为 README 没说 extra_k_cache 对应哪一种。英文同样只写 extra_k_cache / k_cache。
- 第 16 条删的是 fig-s4-batch 出处里的「10.2」；英文那张图的出处要跟着删。
- 第 25 条：fig-pd 全图 Gbps → Gb/s，2P1D 图的负号改成 U+2212。
