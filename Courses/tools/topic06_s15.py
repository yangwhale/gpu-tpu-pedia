# -*- coding: utf-8 -*-
"""专题六 · 第十五节 · 出处台账。

按「结论 ← 材料」排，顺序跟正文各节走。只收 topic06_s0–s14、topic06-fig-*.py 的 src／docstring、
topic06_numbers.py 里已经写明的出处；沿用专题五的图写「沿用专题五台账第 X 行」。
⛔ 公开仓库：不写内部链接、不写真人姓名（论文作者用「等」）、不引内部项目。
"""

BODY = '''
  <p class="lead">按「结论 ← 材料」排，顺序跟正文各节走。2026-10-15 核对。</p>
  <table>
    <tr><th>结论</th><th>材料</th></tr>
    <tr><td>DeepSeek-V3：总参数 6,710 亿、每 token 激活 370 亿；61 层、hidden 7168、256 个路由专家选 8、kv_lora 512 ＋ rope 64、moe_intermediate 2048</td><td>DeepSeek-V3 Technical Report（arXiv 2412.19437）摘要；形状取 Hugging Face deepseek-ai/DeepSeek-V3 的 config.json</td></tr>
    <tr><td>MLA 每 token KV 70,272 字节；128K 一条序列 8.58 GiB（9.21 GB）</td><td>本课推导：(512＋64) × 2 B × 61 层；× 131,072</td></tr>
    <tr><td>TPU v7：FP8 4,614 TFLOP/s、HBM 192 GiB（≈ 206.2 GB）、7.37 TB/s，每芯片 2 个 device；BF16 2,307 TFLOP/s，屋脊点 ≈313 FLOP/字节</td><td>Google Cloud TPU7x 公开规格页（写 7,380 GB/s；7.37 TB/s 取 Google 博客 Ironwood: The first Google TPU for the age of inference，2025-04，差 0.14% 不影响屋脊点）；JAX jax/_src/pallas/mosaic/tpu_info.py 记每芯片 HBM 206e9 字节（≈ 191.9 GiB），即规格表的 192 GiB；BF16 与屋脊点本课推导：4,614 ÷ 2，2,307 TFLOP/s ÷ 7.37 TB/s</td></tr>
    <tr><td>TTFT、TPOT／ITL 的定义</td><td>vLLM 仓库 vllm/benchmarks/serve.py；NVIDIA NIM 的 LLM benchmarking 文档页（写明 ITL also known as TPOT）</td></tr>
    <tr><td>decode 一步：读 37 GB 要 5.02 ms，算 74 GFLOP 只要 0.016 ms，算力用上 0.32%</td><td>本课推导：37 GB ÷ 7.37 TB/s；2 × 37e9 ÷ 4,614 TFLOP/s；FP8 每参数 1 字节、每 token 2 FLOP，屋脊点 4,614 TFLOP/s ÷ 7.37 TB/s ≈ 626 FLOP/字节，2 ÷ 626 ≈ 0.32%</td></tr>
    <tr><td>算术强度 M·H ÷ (H＋2M)：M＝256 时 239，M≈343 过屋脊，4K prompt 到 1,911</td><td>本课推导：H＝7168 的方阵乘，bf16</td></tr>
    <tr><td>256 用户、128K 时 KV 占一步字节的 78%；权重项 ＝ 非路由部分 ＋ 653.9B × [1 − (1 − 8/256)<sup>b</sup>]</td><td>本课推导：路由专家 256 × 58 层 × 3 × 7168 × 2048 ＝ 653.9B</td></tr>
    <tr><td>decode MoE 每专家平均 8 行，MXU 一边 256，只用上 1/32</td><td>本课推导：256 × 8 ÷ 256；MXU 尺寸见 JAX jax/_src/pallas/mosaic/tpu_info.py 的 mxu_column_size</td></tr>
    <tr><td>Gemma4-31B 并发 1 TTFT 86 ms，并发 256 涨到 7,756 ms（多出来的是排队）</td><td>本课程作者实测：gpu-tpu-pedia tpu/tpu-inference/Gemma4-31B/README.md:453-486</td></tr>
    <tr><td>InferenceX 三点 68→9,759、121→3,816、173→895 tok/s/GPU</td><td>gpu-tpu-pedia VLLM-V4PRO-RUNBOOK.md:683-695</td></tr>
    <tr><td>V4 on GB300：交互 ~50 tok/s/user 时每卡 ~11,200（开 MTP；分母是 decode 卡还是全部卡，官方未写明），Day-0 ~2,200（5.1×）；每卡同时服务约 224 个用户</td><td>PyTorch 博客 2026-06-23「Serving DeepSeek-V4 on GB300 with SGLang: 5x Higher Throughput at the Same Interactivity Since Day-0」；224 ＝ 11,200 ÷ 50 本课推导</td></tr>
    <tr><td>同一次跑：output÷decode 卡 5,270，(in＋out)÷全部卡 11,880，换口径 1,318 到 47,520，差 36×</td><td>gpu-tpu-pedia SGLANG-V4PRO-RUNBOOK.md:794-809，字段定义 :731-735；1,318、47,520、36× 本课推导</td></tr>
    <tr><td>假天花板：6,887 → 10,614（＋54%）→ 10,704，峰值 12,070</td><td>SGLANG-V4PRO-RUNBOOK.md:611-615、849-858（§11.5 L849-896）</td></tr>
    <tr><td>temperature 0 与非 0：投机接受率 1.16% 对 34%，吞吐 944 对 2,911；K3 的草稿按概率抽，官方就用温度 1.0 压测</td><td>VLLM-V4PRO-RUNBOOK.md:394-420；K3 见 VLLM-K3-RUNBOOK.md:226-230</td></tr>
    <tr><td>没预热：第一轮 13,585、第二轮 27,803（×2.05）；闭环压测低估 13%</td><td>VLLM-V4PRO-RUNBOOK.md:431-433；闭环 SGLANG-V4PRO-RUNBOOK.md:478、1135</td></tr>
    <tr><td>goodput 比吞吐更能比系统：7.4× 请求率、12.6× 更紧 SLO</td><td>DistServe，arXiv 2401.09670</td></tr>
    <tr><td>vLLM 压测工具的 goodput ＝ 每秒达标请求数（与 DistServe 口径不同）</td><td>vLLM 仓库 vllm/benchmarks/serve.py 的 --goodput 参数</td></tr>
    <tr><td>DeepSeek 线上实测（2025-03-01 公布，统计 2025-02-27～28）：226.75 节点、理论成本利润率 545%（全按 R1 定价收费的假设）、decode 时每节点输出约 14,800 tok/s（原文「~14.8k tokens/s output during decoding」；故事图卡片写 14.8k，同一数）、20–22 tok/s/user；一天输入 6,080 亿、输出 1,680 亿 token，合每卡每秒约 4,950（本课推导：7,760 亿 ÷ 86,400 ÷ 1,814 卡）</td><td>GitHub deepseek-ai/open-infra-index：202502OpenSourceWeek/day_6</td></tr>
    <tr><td>每 token KV：MHA 3,997,696 B、GQA（Llama-3.1-70B）327,680 B、Qwen3.5 全注意力 30,720 B ＋ 线性状态 180 MiB／序列；Qwen3.5 KV 头数 2</td><td>本课推导，形状取各自 Hugging Face config.json</td></tr>
    <tr><td>8 颗 v7 放 V3：KV 池 92.4 GiB／芯片，4K 装 344 条、32K 装 43 条、64K 装 21 条（FP8 KV 按 656 B/层约 37 条）；Llama 4K 73 条</td><td>本课推导：每芯片可用 189.5 GiB（每 device 实测 94.75 GiB × 2，R1 FP4 README:59，同台账里「V3 FP8 切片表」那一行）× 0.9 ＝ 170.55 GiB，− 78.1 GiB（671 GB ÷ 8）；0.9 为本课假设；先按权重全切分粗算，DP attention 下非专家权重每 device 复制，实际更小（10.5）</td></tr>
    <tr><td>PagedAttention：KV 有效利用率 20.4–38.2% → 96.3%（3.3 正文照录；故事图卡片取整写 20–38%→96%，第零节、第三节小结与故事图 alt 口语写「两到四成 → 96%」，同一组数，正文与图只是取整不同）</td><td>vLLM 论文，arXiv 2309.06180</td></tr>
    <tr><td>Qwen3.5 RPA KV 页 16→256 token：并发 512 时这个 kernel 的 decode 一步 428 → 283 µs（原文称 kernel 级提速 33.8%，不是整步）</td><td>原文「reduced the decode step latency under Concurrency-512 from 428µs to 283µs, achieving a 33.8% kernel-level speedup」；gpu-tpu-pedia tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/BLOG-systems-engineering-playbook.md L395-405，即 Google Developers Blog「Systems Engineering Playbook: Optimizing Qwen 3.5-397B MoE on Ironwood (TPU7x)」</td></tr>
    <tr><td>V4 两个池（full／swa）预算比改 0.20 → 0.15：每 rank 请求 727 → 886，端到端每 decode 卡 6,887 → 10,614（＋54%）；8K 输入时 ratio 0.10 为 9,354（滑窗池 96%、全长池 78%，每 rank 588），0.15 为 9,053</td><td>SGLANG-V4PRO-RUNBOOK.md:855-857（§11.5 L849-896）；8K 两行 :864-865；swa_tokens 比例 L946（引 pool_configurator.py:387）；sliding_window 128 见 HF config</td></tr>
    <tr><td>再把 mem-fraction-static 0.85 → 0.88：每 rank 请求 886 → 999（＋13%），端到端 10,614 → 10,704</td><td>SGLANG-V4PRO-RUNBOOK.md:857-858、887-892</td></tr>
    <tr><td>Qwen3.5 混合池的 bug 与 prefix caching 断言</td><td>tpu/tpu-inference Qwen3.5 README:77、:95</td></tr>
    <tr><td>SGLang（含 RadixAttention 前缀复用）吞吐最多 6.4×，对照是当时其他推理系统（不是「不复用」），且是整套运行时（RadixAttention ＋ 压缩状态机等）的结果</td><td>SGLang arXiv 2312.07104 摘要（"up to 6.4x higher throughput compared to state-of-the-art inference systems"）</td></tr>
    <tr><td>Flash-Decoding：长上下文 decode 最多提速 8 倍</td><td>斯坦福 CRFM 博客 2023-10「Flash-Decoding for long-context inference」</td></tr>
    <tr><td>投机解码 2–3×；EAGLE 2.7–3.5×；V3 MTP 第 2 个 token 接受 85–90%、1.8×（均单请求延迟／TPS 口径）</td><td>Leviathan 等 arXiv 2211.17192 摘要；EAGLE arXiv 2401.15077 摘要；DeepSeek-V3 报告 arXiv 2412.19437（MTP 评估一节）</td></tr>
    <tr><td>KV 外存回读的盈亏带宽：V3 2.19 GB/s、Llama 5.35 GB/s，1 GB/s 的慢链路在前缀长于 ~35K 时也比重算快</td><td>本课推导：重算 ＝ (2 × 37e9 × L ＋ 4,997,120 × L²/2) ÷ (4,614 T × 50%)；Llama 激活 70.6B 取 HF 模型页</td></tr>
    <tr><td>Orca 迭代级调度：同延迟下吞吐是 FasterTransformer 的 36.9×</td><td>Orca，OSDI'22</td></tr>
    <tr><td>chunked prefill；Sarathi-Serve 达标容量是 vLLM 的 2.6×（Mistral-7B）–3.7×（Yi-34B），配流水线并行跑 Falcon-180B 端到端最多 5.6×（摘要未写对照）；Gemma4 一次 prefill 16,384 token、切 16 块</td><td>Sarathi arXiv 2308.16369；Sarathi-Serve arXiv 2403.02310；vLLM 文档 Optimization and Tuning 页；Gemma4-31B/README.md:225-226</td></tr>
    <tr><td>并发 400 → 1,800：总负载 4.5×，每卡只多 11%（8,777 → 9,745 tok/s）；有记录的 TTFT 从并发 600 的 45 s 涨到 1,800 的 253 s</td><td>SGLANG-V4PRO-RUNBOOK.md:593-609</td></tr>
    <tr><td>V4-Pro dep8 并发扫描：conc600（客户端 8,400 路）每张 decode 卡 9,168 tok/s、TPOT 59–61 ms ≈ 每用户 17 tok/s（指标图「9,168 @ 17」）；conc1200 首字中位 150 s；TPOT 全程 58–63 ms（批处理图）</td><td>SGLANG-V4PRO-RUNBOOK.md:591-599 并发扫描表（conc600 在 :596、conc1200 在 :598）、:604；17 ＝ 1000 ÷ 60 本课推导</td></tr>
    <tr><td>Little 定律核对：dep8 4,335（每卡约 542）、dep16 2,783；客户端并发 8,400 路（14 个 frontend × 每路 600）</td><td>SGLANG-V4PRO-RUNBOOK.md:794-801；本课推导 9,032 × 8 × 0.060、84,321 × 0.033、4,335 ÷ 8；8,400 见并发扫描表 :596（conc600 那档，TTFT 同为 45 s）</td></tr>
    <tr><td>多给槽位：并发 886 → 999，峰值只 12,063 → 12,070，TPOT 77.7 → 85.3 ms</td><td>SGLANG-V4PRO-RUNBOOK.md:849-858、885-894</td></tr>
    <tr><td>SGLang-JAX 调度开销 12 ms → 38 µs；按 batch 尺寸预编译</td><td>lmsys.org/blog/2025-10-29-sglang-jax</td></tr>
    <tr><td>vLLM 抢占默认 RECOMPUTE；公平调度 VTC；跨实例迁移 Llumnix</td><td>docs.vllm.ai「Optimization and Tuning」页；arXiv 2401.00588；arXiv 2406.03243</td></tr>
    <tr><td>Llama-3.1-70B 权重 141,107,412,992 字节；8K 一条 KV 2.68 GB；batch ≈52.6 时两者一样重</td><td>HF safetensors index 的 total_size；其余本课推导</td></tr>
    <tr><td>TPU 内存 tile 规则（每 KV 头补齐到整块）</td><td>openxla.org/xla/tiled_layout</td></tr>
    <tr><td>RPA 在 TPU7x 跑 Llama 3 8B：decode 带宽利用最多 86%、prefill 算力利用 73%</td><td>arXiv 2604.15464 摘要</td></tr>
    <tr><td>Ragged Paged Attention 的布局与 kernel 设计</td><td>RPA，arXiv 2604.15464 §2.1.2、§3.1、§3.4、§3.5</td></tr>
    <tr><td>MLA 吸收后 decode 算术强度 242，prefill 每对运算量 3.4×</td><td>DeepSeek-V2 arXiv 2405.04434 §2.1.2；本课推导 278,528 ÷ 1,152、278,528 ÷ 81,920</td></tr>
    <tr><td>MLA 每 token KV 只有同形状 MHA 的约 1/57（故事图卡片 ÷57 同此）</td><td>本课推导：70,272 ÷ 3,997,696（3.1 表）</td></tr>
    <tr><td>故事图：GQA 卡「KV ÷8」；DeepSeek-V2 卡与脚注「KV −93.3%」（对稠密 67B）</td><td>GQA arXiv 2305.13245，÷8 本课推导：64 个查询头分 8 组、每组共用一份 KV；DeepSeek-V2 arXiv 2405.04434 摘要（论文的 −93.3% 混着 6 bit KV 等改动）</td></tr>
    <tr><td>单专家 44,040,192 参数；单 device 读权重 23.9 µs（bf16）／11.95 µs（fp8），稠密算 19.5／9.77 µs</td><td>本课推导：3 × 7168 × 2048；3.685 TB/s ＝ 7.37 ÷ 2，1,153.5／2,307 TFLOP/s ＝ 每芯片 ÷ 2</td></tr>
    <tr><td>decode AllReduce 11.0 → 2.37 µs，SoL 1.404 µs，ITL −8.7%</td><td>arXiv 2607.16100</td></tr>
    <tr><td>tpu-inference#2674：batch 小时用「one-hot 矩阵 × 隐状态」代替按下标排序／还原；Qwen3.5 并发 64 总吞吐 2,211 → 2,531（＋14.5%，1K 进 8K 出）、11,269 → 12,090（＋7.3%，8K 进 1K 出）；合入后默认关（环境变量 ONEHOT_MOE_PERMUTE_THRESHOLD 默认 0）</td><td>github.com/vllm-project/tpu-inference/pull/2674（2026-05-22 合入）正文表与 diff。2026-10-15 核对</td></tr>
    <tr><td>14,336 字节消息的带宽项只要 24 ns，实测是它的 460×</td><td>本课推导：7168 × 2 B ＝ 14,336 B；× 2(n−1)/n ＝ 1.5（4 卡 ring AllReduce）；NVLink5 1.8 TB/s 双向 → 单向 900 GB/s；14,336 × 1.5 ÷ 900 GB/s ≈ 23.9 ns；11.0 µs ÷ 23.9 ns ≈ 460</td></tr>
    <tr><td>V4-Pro 只有 1 个 KV 头，TP 切注意力会复制 KV</td><td>VLLM-V4PRO-RUNBOOK.md:298-302</td></tr>
    <tr><td>并发 512 时 TP4 21,100（16 卡）对 DEP 55,153（20 卡）：每卡 1,319 对 2,758，2.09×；DEP 并发 1,536 时 65,132、每卡 3,257、2.47×，TTFT 95 s；输出 4,220 对 11,031 tok/s、出字间隔 46.8 对 11.8 ms → 在飞请求约 197 对 130（Little 定律），dep8 反而少；每张 decode 卡输出 1,055 对 1,379，1.31×；投机接受率 34% 对 35.7%</td><td>VLLM-V4PRO-RUNBOOK.md:449、464-468、476；每卡与 2.09× 本课推导；接受率 :480；在飞请求 ＝ 输出 tok/s × TPOT、每 decode 卡 ＝ 输出 ÷ 4 或 8，本课推导（runbook 原文把收益归给容量，本课按 Little 定律归给步时）</td></tr>
    <tr><td>DEP 每卡吞吐随 prefill 喂料 307 → 12 个 prefill 时 3,187（14 个时 3,321）；TP4 停在 20,479</td><td>VLLM-V4PRO-RUNBOOK.md:711-724（4k1k 重跑 :727-734）、:488，2026-07-24 记录</td></tr>
    <tr><td>tpu-inference DEP：并发 512 时领先 62%（1k/8k）、40%（8k/1k）；并发 64／128／256 时 1k/8k 为 0%（打平）／＋14%／＋22%，8k/1k 为 ＋14%／＋23%／＋36%（输出 tok/s，DP=8 对 TP=8）；mmlu_pro 0.8129 对 0.8271</td><td>github.com/vllm-project/tpu-inference/pull/2577（2026-05-15 合入）</td></tr>
    <tr><td>大 EP：DeepSeek decode 用 EP144（18 个节点），每卡 2 个路由专家 ＋ 1 个共享专家；EP 的两条理由（放大 batch 提高矩阵效率、每卡只读少数专家降低延迟）；一批拆两个微批交替藏通信；跨节点 all-to-all 用 DeepEP</td><td>github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md；「空出的显存给 KV」本课推导</td></tr>
    <tr><td>SGLang 在 96 张 H100 上复现 PD 分离＋大 EP：输出吞吐最多是同卡数纯 TP 的 5×</td><td>lmsys.org/blog/2025-05-05-large-scale-ep</td></tr>
    <tr><td>TEP／DEP 的定义；Megatron 命名；Folding</td><td>沿用专题五台账第 43、44、18 行</td></tr>
    <tr><td>Meta 上下文并行：128 张 H100（16 节点）跑 Llama3 405B，prefill 1M token 77 s、128K 3.8 s，并行效率 93%</td><td>arXiv 2411.01783 摘要</td></tr>
    <tr><td>K3 DCP：128K 并发 8 吞吐 108 → 365（＋237%）、并发 1 ＋12%；4K 并发 8 −9%；8K 并发 8 −11%、并发 32 ＋24%；32K 并发 8 −17%、并发 32 ＋7%</td><td>gpu-tpu-pedia SGLANG-K3-RUNBOOK.md §11.0.3 :1010-1023、宽表 :921-938；百分比本课推导</td></tr>
    <tr><td>DCP 让 KV 容量 1.5M → 12.2M token；单流 −26% 来自丢掉 symm-mem</td><td>SGLANG-K3-RUNBOOK.md:721-723；:1024-1026、:962</td></tr>
    <tr><td>K3 128K 并发 8 首字中位：TP8 32.1 s，DCP8 2.3 s（含排队）；官方 PD 吞吐：1P1D（TP8 decode）2,808 对 2P2D（DCP8 decode）2,633 tok/s/GPU</td><td>SGLANG-K3-RUNBOOK.md:939（TP8 r060-isl128k）、:941（dcp8-isl128k）；:609-610、:707-708（官方原文引于 :1323）</td></tr>
    <tr><td>69 层 KDA 线性注意力不能按序列切；切了状态差 58%</td><td>SGLANG-K3-RUNBOOK.md:725-726；58% 为本课玩具递推 S<sub>t</sub> ＝ 0.9·S<sub>t−1</sub> ＋ 1</td></tr>
    <tr><td>KV 复制；DCP 通信；PCP／DCP 分工；Ring／Ulysses／USP</td><td>沿用专题五台账第 23、29、45、21 行</td></tr>
    <tr><td>KV 传输软件栈</td><td>github.com/ai-dynamo/nixl；github.com/kvcache-ai/Mooncake；LMCache arXiv 2510.09665；docs.nvidia.com/dynamo 的 KVBM；github.com/google/tpu-sync（~200 GB/s，not yet recommended for general use）；tpu-inference kv_transfer（jax.experimental.transfer）</td></tr>
    <tr><td>PD 传输带宽从 200 MB/s 调到 7–167 GB/s</td><td>VLLM-V4PRO-RUNBOOK.md:194，配置见 :106、180-194</td></tr>
    <tr><td>Qwen3-Coder 一条 8K 请求 KV 1.04 GB；各链路传完要 5,201／149／6.2 ms</td><td>本课推导：2 × 62 × 8 × 128 × 8192 字节（FP8）；83 ms 与 PCIe 数沿用专题五台账第 25 行</td></tr>
    <tr><td>scale-out 网卡：A4X（GB200）每 GPU 400 Gbps（每节点 4 GPU、4×CX-7 400G）；A4X Max（GB300）每 GPU 800 Gbps（CX-8 双端口）；1.04 GB KV 一趟约 21 ms／10 ms</td><td>gpu/a4x/01-environment-setup/README.md:35；gpu/a4x-max/05-rdma-test/README.md:197-200；时间 ＝ 1.04 GB × 8 ÷ 400 或 800 Gbps 本课推导</td></tr>
    <tr><td>Qwen3.5 混合模型 PD：45 层 GDN ＋ 15 层注意力，每请求传输约 349 MB（未分解）；单请求吞吐是单机的 0.95 倍（单并发）／0.92 倍（4 并发）</td><td>tpu_connector_hma.py 注释；Qwen3.5 README:375-389；README-TPU-VM.md:859-867</td></tr>
    <tr><td>PD 踩坑清单</td><td>SGLANG-V4PRO-RUNBOOK.md:941、:63；VLLM-V4PRO-RUNBOOK.md:760、:106、:703、§12.4</td></tr>
    <tr><td>跨数据中心 PD：链路 33.35／8.25／59.93／4.66 Gbps；案例 ＋54%、P90 −64%、同等成本约 ＋15%、比不调度的朴素异构 ＋32%、13 Gbps（论文吞吐模型推算）、阈值 19.4K；案例为 32 张 H200 做 prefill ＋ 64 张 H20 做本地 PD，对比 96 张 H20 同构 PD</td><td>PrFaaS，arXiv 2604.15039 Table 3 与案例节；整条跨集群以太网约 100 Gbps 由「13 Gbps 占 13%」（§4.3.1）得出</td></tr>
    <tr><td>P:D 配比公式：87.5 req/s、要 71.7 万 tok/s prefill、一台 7.28 万，约 9.8 台</td><td>SGLANG-V4PRO-RUNBOOK.md §12.2 :1082-1091</td></tr>
    <tr><td>runbook 实测单 prefill 卡峰值 15,196 input tok/s，是官方单卡值 18,200 的 83%；官方高并发配方 high-conc-8p1d 用 8 个 prefill</td><td>SGLANG-V4PRO-RUNBOOK.md:1119（15,196 ＝ 官方 83%）、:1091（18,200，runbook 按官方值取，未注明更原始的出处）、:1091／:1101（high-conc-8p1d）；83% ＝ 15,196 ÷ 18,200 本课推导</td></tr>
    <tr><td>B200 1P1D → 2P1D：输出 1,118.8 → 1,159.0，每卡 69.9 → 48.3，TTFT 2.40 → 1.73 s（−28%），P99 12.07 → 6.23</td><td>gpu-tpu-pedia gpu/inference/a4/deepseek-v3/REPORT-2P1D-DeepSeek-V3.md :69、:85、:100-142；cuda graph 说明 REPORT-1P1D :129</td></tr>
    <tr><td>固定并发压测下每个请求约 65 s</td><td>REPORT-2P1D-DeepSeek-V3.md 的 Avg Latency 列，并发 128：2P1D 65.5 s（:69）、1P1D 67.2 s（:85）</td></tr>
    <tr><td>prefill 台数扫描（扫到 14 个 prefill）：12 个时按每张 decode 卡算 307 → 3,187，10.4×；按全部卡算 204 → 455，2.2×</td><td>VLLM-V4PRO-RUNBOOK.md §12.3；全卡 ＝ output ÷ (4n＋8) 本课推导</td></tr>
    <tr><td>MiMo batch 128：同样 2 个 pod，1P1D 13.64K 对不分离 12.78K，＋6.7%（README 写约 6.8%）；2P1D 用 3 个 pod 到 15.31K，按 pod 算 5.10K 对 6.39K，低 20%；按 pod 另算 batch 64 不拆／1P1D／2P1D 5.85／5.28／3.61K、batch 128 的 1P1D 6.82K；Qwen3-Coder 1P1D TTFT 95 → 281 ms</td><td>tpu/sglang-jax/mimo/README.md:8-12、81-88；6.7%、按 pod 各值（总吞吐 ÷ pod 数）与 20% 本课推导；tpu/tpu-inference/Qwen3-Coder-480B/README.md:1005-1021</td></tr>
    <tr><td>AFD：MegaScale-Infer 每卡吞吐 1.90×、Step-3 4,039 对 2,324；vLLM AFD 插件 16K／32K 实测；EPD 把多模态编码再拆出去（15×、22×、71%）</td><td>概念沿用专题五台账第 24、51 行；arXiv 2504.02263；arXiv 2507.19427；vllm.ai/blog/2026-07-23-vllm-afd-plugin、github.com/vllm-project/afd-plugin（RFC #22799 于 2026-05-31 关闭）；EPD arXiv 2501.05460；Step-3 对 V3 不是同一模型，差距含模型侧设计，正文已注明</td></tr>
    <tr><td>同一模型五种部署 14,563／16,459／21,100／55,153／65,132；调参 ＋45%，换拓扑 ＋347%（并发 1,536）</td><td>VLLM-V4PRO-RUNBOOK.md:446、448、449、465、468；:489（另 :476 的 2.47×、:487 的 ＋28%）</td></tr>
    <tr><td>每卡 1,820／1,319／3,257；对 1p1d 起点 1.79×</td><td>本课推导：总吞吐 ÷ 卡数</td></tr>
    <tr><td>Qwen3-Coder TP16：跨 DCN c=1 −22%、c=16 −63%；同 ICI 切片 −15～−21%（c=16 总 513.8 对单机 602）</td><td>Qwen3-Coder-480B/README-multinode.md:441-445；README-TPU-VM.md:1000-1009</td></tr>
    <tr><td>V4-Pro dep8 要一整柜 NVL72；拓扑与频率的一般讨论</td><td>SGLANG-V4PRO-RUNBOOK.md:34、:63；沿用专题五台账第 19、26 行</td></tr>
    <tr><td>v7x-8 是 4 颗芯片（名字按 device 数）；mesh 构造</td><td>tpu/tpu-inference/README.md:41；JAX mesh_utils 文档</td></tr>
    <tr><td>v7 每 device 实测可用 94.75 GiB（每 device 物理 96 GiB）；R1 FP8 MoE 每 device 101.8 放不下（README 写 GB，按 GB、GiB 读都超，最少也超 0.06 GB）</td><td>tpu/tpu-inference R1 FP4 README:59、72-73；0.06 ＝ 101.8 − 101.74（94.75 GiB 换成 GB）本课推导</td></tr>
    <tr><td>路由专家 FP8 均摊到 8 个 device 只有约 81.7 GB，比 README 的 101.8 多出约 20 GB</td><td>本课推导：653.9B × 1 B ÷ 8 ＝ 81.7 GB（＝ 76.1 GiB，拓扑图脚注同此）；101.8 − 81.7 ≈ 20</td></tr>
    <tr><td>v7 切片按 4、8、16、32 颗成档（2x2x1、2x2x2、2x2x4、2x4x4），4.07 颗取 8</td><td>Google Cloud TPU7x 文档 Supported configurations 表（cloud.google.com/tpu/docs/tpu7x）</td></tr>
    <tr><td>V3 FP8 切片表：4K×64 要 642 GiB／纸面 4 颗，4K×256 要 694 GiB／8 颗，128K×256 要 2,821 GiB／32 颗，KV 是权重 3.5×；4 颗放完权重剩 57 GiB，4K 纸面装 213 条，但实测 FP8 放不下</td><td>本课推导：权重 671 GB ＝ 625 GiB；每芯片 189.5 GiB × 0.9 ＝ 170.55 GiB 可用；0.9 为本课假设；实测见上一行</td></tr>
    <tr><td>128K × 256 并发：DeepSeek-V3 KV 约 2,196 GiB（权重 625 GiB 的 3.5 倍）；单条 128K 8.58 GiB</td><td>本课推导：70,272 B × 131,072 × 256 ÷ 2<sup>30</sup></td></tr>
    <tr><td>GB300 R1：并发 8 只用上 ~3% 容量；854 → 10,715 tok/s，12.5×；并发 256 → 512 总吞吐只再涨 3%（10,390 → 10,715，本课推导），首字 12 s → 30 s</td><td>gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v3/sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md :401、:405-413、:415</td></tr>
    <tr><td>投机解码原理（按标准拒绝采样实现时分布无损）</td><td>Leviathan 等 arXiv 2211.17192；Chen 等 arXiv 2302.01318</td></tr>
    <tr><td>K3 开投机：TPOT 11.1 → 2.31 ms，接受长度 6.96；一步变慢 1.45×，每字快 4.81×</td><td>SGLANG-K3-RUNBOOK.md:1089、:979；本课推导：2.31 × 6.96 ≈ 16.1 ms，16.1 ÷ 11.1 ≈ 1.45，11.1 ÷ 2.31 ≈ 4.81</td></tr>
    <tr><td>并发升高收益收窄：4.22× → 1.74×；32K 只剩 1.32×；不开投机时每字间隔 4K 11.1 ms、32K 11.4 ms；32K 变慢的三条原因（验证撞算力墙、草稿也读长 KV、接受率可能降）为本课推导，未拆分</td><td>SGLANG-K3-RUNBOOK.md:1089-1092、:1057-1058、:924、:935（c1 TPOT）；DSpark 草稿块 7 个：:373</td></tr>
    <tr><td>V3 batch 8 一步要读的字节是 batch 1 的 4.4×</td><td>本课推导：均匀路由下命中专家 8 → 57.5 个，37.5 → 163.8 GB</td></tr>
    <tr><td>B200 V4-Flash（vLLM，DEP8）开 MTP 反而降 3–37%</td><td>gpu-tpu-pedia gpu/inference/a4/deepseek-v4/README.vllm.md:266-275</td></tr>
    <tr><td>V4-Pro MTP：峰值 −13%，端到端 ＋28%</td><td>SGLANG-V4PRO-RUNBOOK.md:900-911</td></tr>
    <tr><td>temperature 与接受率（V4-Pro 贪心 DSpark 草稿在温度 1.0 下实测 1.16%）；K3 4.73／2.61；1.16% 低于单字理论值：接受率 ＝ 收下 ÷ 猜的草稿字、每步猜 7 个、第一个被拒后面作废</td><td>VLLM-V4PRO-RUNBOOK.md:398-410（num_speculative_tokens=7、接受率 ＝ accepted ÷ draft 均为原文；低于理论值的解释为本课推导，未拆分）；VLLM-K3-RUNBOOK.md:226-237；告警见 kimi-k3/PD-BACKLOG.md:175</td></tr>
    <tr><td>PD 碰上投机解码：新交接请求 1 个 token 对 1＋N，掉出整图 CUDA Graph；补齐后 TPOT ~40 → ~22 ms</td><td>vLLM 博客 vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd；转述见 kimi-k3/PD-BACKLOG.md:52-59</td></tr>
    <tr><td>草稿模型谱系（MTP、DSpark 的定义；V3 的 MTP 是一整层 transformer，共享 embedding 和输出头；EAGLE-3 2025-03 融合多层状态，SGLang batch 64 吞吐 1.38×；DSpark 摘要：吞吐相同时单用户生成比线上 MTP-1 快 60–85%）</td><td>SpecInfer 2305.09781、Medusa 2401.10774、EAGLE 2401.15077／2406.16858、MTP 2412.19437、EAGLE-3 2503.01840、DSpark 2607.05147、Lookahead 2402.02057、SuffixDecoding 2411.04975、DiffuSpec 2510.02358、DFlash 2602.06036（均 arXiv）；github.com/apoorvumang/prompt-lookup-decoding；MTP 结构见 DeepSeek-V3 技术报告 §2.2 MTP Modules；EAGLE-3 摘要页（v1 2025-03-03）。2026-10-15 核对</td></tr>
    <tr><td>8 颗 v7 读一遍 V3 激活权重：BF16 1.26 ms、FP8 0.63、MXFP4 0.33、NVFP4 0.35</td><td>本课推导：37e9 × 位数 ÷ 8 ÷ (8 × 7.37 TB/s)</td></tr>
    <tr><td>MXFP4 每 32 个数一个 E8M0 scale（只能取 2 的幂），合 4.25 bit；NVFP4 每 16 个一个 E4M3（可取非 2 的幂），合 4.5 bit</td><td>OCP Microscaling Formats (MX) v1.0 规范；NVIDIA 技术博客「Introducing NVFP4」</td></tr>
    <tr><td>v7 公开规格只列 BF16、FP8 峰值，没写 FP4 算力</td><td>Google Cloud TPU7x 公开规格页</td></tr>
    <tr><td>V4 专家权重为 FP4</td><td>Hugging Face DeepSeek-V4 模型卡</td></tr>
    <tr><td>R1 FP4 每 device 60.9，FP8 101.8 放不下（README 写 GB）；上限 94.75 GiB（运行日志单位）；K2.6 INT4 存成无符号 0–15，加载时直接当有符号读（没减 8）就崩</td><td>R1 FP4 README:59、72-73、:875-878（README 实测值）；K2.6 README:217、:265-266</td></tr>
    <tr><td>kernel 气泡：H100 理想 1,350 次前向／秒，实际 770；megakernel 带宽利用 50% → 78%，比 vLLM 快约 2.5×、比 SGLang 快 1.5× 多；Llama-3.2-1B、batch 1，小模型小 batch 收益最大；PDL 同步粒度太粗</td><td>hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles；1,350 ＝ 3.35 TB/s ÷ 2.48 GB 本课推导</td></tr>
    <tr><td>Mirage MPK 编译成 megakernel：1.7×；2025-06 开源，论文 2025-12</td><td>arXiv 2512.22219（v1 2025-12-22）；github.com/mirage-project/mirage（2025-06-19 公开，作者博客「Compiling LLMs into a MegaKernel」）</td></tr>
    <tr><td>TPU megakernel：16 颗 v7 单用户 700＋ tok/s，对照 GB200（vLLM）452，两边都按接受长度 6 换算；一步 ~8.5 ms，编译 &lt;90 s；不开投机 batch 1–8 约为 GB200 基线 1.4–2×；博客写的是 megakernel ＋ DSpark 投机，未提量化</td><td>https://inferact.ai/blog/tpu-megakernels（2026-09-23；博客图注「Output tokens per second with speculative decoding at an acceptance length of 6」同时覆盖两条曲线）；6 ÷ 8.5 ms ≈ 706 本课推导</td></tr>
    <tr><td>KV 存 FP8：FlashMLA 给 V3.2 的布局每 token 每层 656 B（512 个 FP8 ＋ 4 个 FP32 缩放 ＋ 64 维 BF16 RoPE 不量化），是 BF16 1,152 B 的 57%；DSA 每 token 每层另有 indexer key 132 B（128 个 FP8 ＋ 1 个 FP32 缩放，BF16／FP8 KV 下都存 FP8），合计 788 B 对 1,284 B（61%），请求容量约是 BF16 KV 的 1.63 倍；K3 Peak Throughput 预设 --kv-cache-dtype fp8_e4m3 必开（BF16 KV 装不下每副本 128 个请求）</td><td>github.com/deepseek-ai/FlashMLA README（2026-09-30 提交 2e5429f 之前的版本，「FP8 with scale」一节；现版只剩 V4.1 的 528／288 B 格式）；HF deepseek-ai/DeepSeek-V3.2 config.json（index_head_dim 128）；DeepSeek-V3.2-Exp inference/model.py（k_cache float8_e4m3fn、k_scale_cache float32 × head_dim÷128）；vLLM deepseek_v2.py Indexer（head_dim＋head_dim÷128×4，dtype uint8）；SGLang memory_pool.py DSATokenToKVPool（index_k_with_scale_buffer_dtype＝uint8）；本课推导 656 ÷ 1,152、788 ÷ 1,284、1,284 ÷ 788；SGLANG-K3-RUNBOOK.md:1308</td></tr>
    <tr><td>分桶浪费：一步整批 token 总数只补到一个 8K 桶时浪费 87.3%，32 个桶 10.8%；13 个按 √2 递增的档 16.0%、16 个等距 512 的档 19.7%；补齐按整批总数（RPA ragged 布局），单条请求不补</td><td>本课推导：每步 token 总数取对数正态，中位 700、σ 0.9；RPA arXiv 2604.15464；预编译参数 mimo README:269-270</td></tr>
    <tr><td>新形状首个请求现编译 60–90 s，之后同形状 &lt;1 s；预编译开关</td><td>Qwen3.5 README:794；mimo README:269-270</td></tr>
    <tr><td>13.4：强杀后每卡显存仍占 97 GB、只剩 187 GB，mem-fraction-static 0.85 要静态占 241 GB（权重＋KV 池），差 54 GB，下次启动 OOM；从删 pod 到显存清零实测 56 秒</td><td>SGLANG-V4PRO-RUNBOOK.md:960-978（97 GB 在 :962-963，算式 :966，56 秒 :978）；284 × 0.85 ≈ 241、284 − 97 ＝ 187、241 − 187 ＝ 54 本课推导</td></tr>
    <tr><td>冷启动拆账：GLM-5.1 名义加载里藏 6 分 29 秒重编译（原表总计 ~10 分 44 秒，分项和 10 分 29 秒）；Qwen3-Coder 40 分钟里编译 29；MiMo-V2.5-Pro 编译 15 ＋ 预热 7（原表总计 ~50 分钟，分项和 52 分钟）</td><td>GLM-5.1 README:470-476；Qwen3-Coder README-TPU-VM.md:1011；MiMo-V2.5-Pro README:184-188</td></tr>
    <tr><td>R1 加载 215.7 → 21.4 s（1,367 个张量只有 25 种形状）；K2.6 57 分钟 → 6 分 8 秒（9.3×）；持久编译缓存</td><td>R1 README:1202-1223、:1238-1259；K2.6 README:23、39-40；JAX 文档「Persistent compilation cache」</td></tr>
    <tr><td>KV 感知路由：命中缓存首字 340 ms，不路由的最坏情况 2,850 ms（原文 worst case，与冷推理同值），命中率 87.4%（4,176／4,776）</td><td>Red Hat Developer 2025-10-07「Master KV cache aware routing with llm-d」</td></tr>
    <tr><td>JetStream 停止维护：核心功能并入 tpu-inference、宣布 2026-02-01 起归档，最后提交 2026-01-05；截至 2026-10-15 仓库 archived 标记仍为 false</td><td>github.com/AI-Hypercomputer/JetStream README 公告与提交记录，2026-10 核对</td></tr>
    <tr><td>同等 GPT-3.5 水平推理价格 2022-11 → 2024-10 降 280 多倍，主要归给小模型</td><td>斯坦福 HAI《AI Index Report 2025》</td></tr>
    <tr><td>llm-d 分层卸载（TPU7x、Qwen3-32B，KV 卸到主机内存）：平均 TTFT 0.98 → 0.56 s，P90 TTFT 2.1 → 0.5 s，输入输出合计吞吐 67,262 → 73,178 tok/s（＋8.9%）</td><td>llm-d/llm-d@accc6c5:guides/tiered-prefix-cache/README.md:436-439（HBM-only 对 HBM ＋ CPU RAM 那张表，2026-06-11 随 #1561 加入，故 14.3 表记 2026）。2026-10-15 核对</td></tr>
    <tr><td>Kimi K2 Thinking：QAT、MoE 原生 INT4，生成约快 2 倍</td><td>Hugging Face 模型卡 moonshotai/Kimi-K2-Thinking</td></tr>
    <tr><td>年份：GPTQ 2022、Splitwise 2023；批不变 kernel</td><td>arXiv 2210.17323；arXiv 2311.18677；Thinking Machines 博客「Defeating Nondeterminism in LLM Inference」</td></tr>
    <tr><td>Mooncake 对 NIXL：并发 64 输出 1,333 对 1,693 tok/s</td><td>VLLM-V4PRO-RUNBOOK.md:705</td></tr>
    <tr><td>14 台 prefill 扫描 6,788／8,809／8,993；14 台全崩、注册表仍显示 14</td><td>SGLANG-V4PRO-RUNBOOK.md:43、1117；同文约束 #8</td></tr>
    <tr><td>Qwen3-Coder ＋12%</td><td>Qwen3-Coder-480B/README.md:1005-1021</td></tr>
    <tr><td>K3：KV 池 787,504 → 637,504；max-running 回调到 48；PD 下 decode 每请求一个状态槽</td><td>SGLANG-K3-RUNBOOK.md:1171、:76、:1318</td></tr>
    <tr><td>R1 传输程序缓存：1 分 48 秒对 3 分 17 秒～3 分 51 秒</td><td>R1 FP4 README:1237-1250</td></tr>
    <tr><td>K2.6 共享存储每层读取慢 38 倍；V4-Pro 806 GB 权重放本地 SSD RAID，读 14 GB/s，估算加载不到 1 分钟</td><td>K2.6 README:259；gpu/inference/a4x-max/deepseek-v4/README.md:55（806G、14 GB/s、&lt; 1 min）、SGLANG-V4PRO-RUNBOOK.md:35、:91（806G）、gb300-local-ssd-raid0-SETUP.md:111（800G 估 ~57 s）；806 ÷ 14 ≈ 58 s 本课推导</td></tr>
    <tr><td>13.5 栈对比表</td><td>tpu-inference 支持矩阵；SGLang-JAX 模型列表</td></tr>
    <tr><td>上下文变长：GPT-4 Turbo 128K（2023-11-06），Gemini 1.5 Pro 1M 私有预览（2024-02-15）</td><td>openai.com/index/new-models-and-developer-products-announced-at-devday；blog.google「Our next-generation model: Gemini 1.5」。2026-10-16 核对</td></tr>
    <tr><td>Mooncake：Kimi 线上在 A800、H800 集群比此前系统多接 115%、107% 请求；真实 trace 下有效请求容量比基线 +59%～498%（2024 预印本摘要的 75%、525% 已由正式版替换）</td><td>FAST'25 论文页 usenix.org/conference/fast25/presentation/qin（「enables Kimi to handle 115% and 107% more requests on NVIDIA A800 and H800 clusters, respectively, compared to previous systems」）</td></tr>
    <tr><td>DeepSeek 磁盘缓存：命中按十分之一计价；128K 高复用 prompt 首字 13 s → 0.5 s（≈ 1/26 本课推导）；24 小时 6,080 亿输入 token 里 56.3% 命中磁盘 KV</td><td>api-docs.deepseek.com/news/news0802（2024-08），价格只取比例；6,080 亿／56.3% 取自 github.com/deepseek-ai/open-infra-index 202502OpenSourceWeek day_6（统计窗口 2025-02-27～28，608B 中 342B 命中）。2026-10-15 核对</td></tr>
    <tr><td>PagedAttention 同延迟吞吐 2–4×（对 FasterTransformer、Orca）</td><td>arXiv 2309.06180 摘要</td></tr>
    <tr><td>Medusa-1 2.2×、Medusa-2 2.3–3.6×（正文合写 2.2–3.6 倍）</td><td>arXiv 2401.10774 摘要</td></tr>
    <tr><td>AI Index：硬件成本每年降 30%、能效每年升 40%</td><td>Stanford HAI AI Index 2025 要点第 7 条</td></tr>
    <tr><td>Hopper 张量核支持 FP8；DeepSeek-V3 FP8 混合精度训练</td><td>NVIDIA H100 发布（2022-03）；arXiv 2412.19437 §3.3</td></tr>
    <tr><td>第十四节八问的顺序</td><td>本课归纳</td></tr>
    <tr><td>R16 补充：FlashAttention 2022；FlashInfer MLSys 2025；DeepSeek-V3.2-Exp（2025-09）的 DSA 每字读 2,048 个 token、API 降价五成以上（2,048 见 V3.2 报告 §2.1）；NVFP4 KV 省约一半、损失 &lt;1%、命中率 +20%、首字最多 3 倍；LMCache 15 倍、截断让命中率减半；SGLang（含 RadixAttention）6.4 倍（对当时其他推理系统）；投机解码 2–3 倍；GQA 每层 4,096 B ÷ MLA 1,152 B ≈ 3.6（本课推导；按每层，按每 token 是 327,680 ÷ 70,272 ≈ 4.7，因层数 80 对 61，第十四节表只给每层口径）；Ring 2023-10、Ulysses 2023-09、Meta CP 2024-11；Kimi K2 Thinking「low-latency mode」</td><td>arXiv 2205.14135、2501.01005；api-docs.deepseek.com/news/news250929、arXiv 2512.02556；developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache；arXiv 2510.09665、2312.07104、2211.17192、2310.01889、2309.14509、2411.01783；huggingface.co/moonshotai/Kimi-K2-Thinking。2026-10-16 核对</td></tr>
    <tr><td>R18 补充：DeepSeek-V4 的 CSA 每 4 个 token 压成一条 KV、HCA 每 128 个压成一条，另留最近 128 个 token 的滑窗 KV；DeepSeek 一天的 token 近八成是输入（6,080 ÷ 7,760，本课推导）；226.75 节点 × 8 卡 ≈ 1,814 卡（本课推导）</td><td>DeepSeek-V4 技术报告 §2.3（CSA／HCA「every m tokens into one entry」）；Hugging Face deepseek-ai/DeepSeek-V4-Pro config.json 的 compress_ratios（4／128）与 sliding_window 128；open-infra-index day_6</td></tr>
    <tr><td>R21 补充 · 0.1：o1 2024-09 发布，先想再答（思维链）；DeepSeek-R1 2025-01，原文「ranges from generating hundreds to thousands of reasoning tokens」</td><td>openai.com/index/learning-to-reason-with-llms（2024-09-12）；arXiv 2501.12948 §2.2.4</td></tr>
    <tr><td>R21 补充 · 3.5 ／ 4.4 ／ 14.3：vLLM V1 前缀缓存 V0 默认关（命中率低时 CPU 开销拖慢）、V1 改数据结构后命中率 0% 时吞吐降不到 1%、默认开；V1 吞吐最多 1.7 倍（对 V0，原文注明不开 multi-step scheduling）；V1 的 EngineCore 进程只跑调度和模型执行，分词、多模态输入处理、反分词、流式回传与之重叠</td><td>blog.vllm.ai/2025/01/27/v1-alpha-release.html（「less than 1% decrease in throughput even when the cache hit rate is 0%」「up to 1.7x higher throughput compared to V0」）。2026-10-15 核对</td></tr>
    <tr><td>R21 补充 · 4.4 ／ 6 ／ 13.6 ／ 14.3：SGLang v0.4（2024-12）零开销调度吞吐 1.1 倍；DeepSeek 模型的 DP attention decode 吞吐最多 1.9 倍；cache-aware 路由命中率 20% → 75%、吞吐 82,665 → 158,596 tok/s（≈1.9 倍，对 v0.3 的轮询）</td><td>lmsys.org/blog/2024-12-04-sglang-v0-4（「1.1x increase in throughput」「up to 1.9x decoding throughput improvement」；1.9 ＝ 158,596 ÷ 82,665 本课推导，原文也写 up to 1.9x）</td></tr>
    <tr><td>R21 补充 · 3.6：SGLang HiCache（2025-09）三层分层缓存，后端 Mooncake、3FS、NIXL；长上下文与多轮对话基准吞吐最多 6 倍、TTFT 最多降 80%（对全部重算）</td><td>lmsys.org/blog/2025-09-10-sglang-hicache（「up to 6× throughput improvement and up to 80% reduction in TTFT」）</td></tr>
    <tr><td>R21 补充 · 第八节：NVIDIA Dynamo 2025-03-18 开源，含 PD 分离、KV 感知路由（Smart Router）、KV 下放（Memory Manager）；传输库 NIXL</td><td>nvidianews.nvidia.com/news/nvidia-dynamo-open-source-library-accelerates-and-scales-ai-reasoning-models；NIXL 名字见 NVIDIA 技术博客「NVIDIA Dynamo, A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models」</td></tr>
    <tr><td>R21 补充 · 13.6：llm-d 2025-05-20 发布，含 AI-aware 路由（送到最可能有热缓存的服务器）</td><td>redhat.com 新闻稿「Red Hat Launches the llm-d Community」（Boston，2025-05-20）</td></tr>
    <tr><td>R21 补充 · 11.5 ／ 14.3：EAGLE-3 加速比最多 6.5 倍（对普通自回归），SGLang 中 batch 64 吞吐 1.38 倍</td><td>arXiv 2503.01840 摘要（「speedup ratio up to 6.5x」「1.38x throughput improvement at a batch size of 64」）</td></tr>
    <tr><td>R21 补充 · 12.1：vLLM FP8 权重（2024-07），Llama 3 70B 每字间隔最多快 2 倍、Mixtral 8x7B 最多 1.6 倍（对 BF16，2×H100）</td><td>developers.redhat.com/articles/2024/07/15/vllm-brings-fp8-inference-open-source-community（「Up to 2x ITL improvement for serving dense models」「Up to 1.6x ITL improvement for serving Mixture of Experts」）</td></tr>
    <tr><td>R21 补充 · 12.1「MoE 赚得少」：只有读权重那份减半时，加速 ＝ 1 ÷ (1 − f／2)，f 为读权重占一步的比例；2 倍要 f ≈ 100%，1.6 倍对应 f ≈ 75%。MoE 每步只读选中的专家，权重字节少，KV、通信、kernel 调度这些不随位数变的部分占比就高，f 低</td><td>本课推导（Red Hat 原文未给原因）</td></tr>
    <tr><td>R21 补充 · 12.7：NVFP4 KV 比 FP8 KV 缓存命中率高 20%、首字延迟最多低 3 倍（Qwen3-Coder-480B-A35B 上分析）</td><td>developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache（「up to 3x lower latency and 20% higher cache hit rate compared to FP8 KV cache」）</td></tr>
    <tr><td>R21 补充 · 14.3：固定一档能力，达到它的推理每年便宜 9 到 900 倍、中位 50 倍；去掉 2024 年 1 月前的数据后中位升到每年 200 倍。与第八到十二节招数进生产时间重合是本课观察，Epoch 未归因</td><td>epoch.ai/data-insights/llm-inference-price-trends（2025；「between 9x per year and 900x per year, with a median of 50x per year」「the median rate increased from 50x per year to 200x per year」）</td></tr>
    <tr><td>R21 补充 · 14.3：DeepSeek-V3.2-Exp 相对 V3.1-Terminus 唯一的结构改动是 DSA；lightning indexer 仍对全部历史打分（O(L²) 但头少、可跑 FP8），主注意力每个 query 只取 top-k ＝ 2,048 个 KV，核心注意力复杂度 O(L²) → O(Lk)</td><td>DeepSeek-V3.2-Exp 技术报告（github.com/deepseek-ai/DeepSeek-V3.2-Exp，DeepSeek_V3_2.pdf §2.1、Inference Costs 一节）；Hugging Face 模型卡（「builds upon V3.1-Terminus」）</td></tr>
    <tr><td>R25 补充 · 0 ／ 3.1：DeepSeek-V4-Pro、V4-Flash 2026-04 发布（预览版）；V4.1-Flash 2026-09-10 开源，552B 主干；CSA2 让各层共用主 KV 和索引 K，主 KV 存 FP4（E2M1，每 16 通道一个 E4M3 缩放），全局 KV 每 token 890 B，约 V4-Flash 的 1/4</td><td>huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash（「share main KV and indexer K across layers」「reduce the global KV cache footprint to 890 bytes per token — roughly 1/4 of DeepSeek-V4-Flash」；HF 建库时间 2026-09-10）；huggingface.co/deepseek-ai/DeepSeek-V4-Pro（「a preview version of DeepSeek-V4 series」；建库 2026-04-22）</td></tr>
    <tr><td>R25 补充 · 12.7：FlashMLA 新增读 NVFP4 压缩 KV 的 kernel，比原先的 FP8 KV 小 45%</td><td>vllm.ai/blog/2026-10-07-deepseek-v41-flash（「a new NVFP4 compressed KV format, which is 45% smaller than the previous FP8 KV cache」）</td></tr>
    <tr><td>R25 补充 · 3.6：vLLM 0.11.0 起自带 KV 下放连接器（OffloadingConnector），附 CPU 后端</td><td>vllm.ai/blog/2026-01-08-kv-offloading-connector（「introduced in vLLM 0.11.0」「bundled with a CPU backend, enabling native CPU offloading of KV data in vLLM」）</td></tr>
    <tr><td>R25 补充 · 第八节：NVIDIA Dynamo 1.0 正式版 2026-03-16 发布</td><td>nvidianews.nvidia.com/news/dynamo-1-0（「NVIDIA Enters Production With Dynamo」，2026-03-16）</td></tr>
    <tr><td>R25 补充 · 11.5：DFlash 一次前向出一整块候选草稿；Gemma-4 31B 在单卡 DGX B300 上用 vLLM，单并发比自回归快 3.0（MT-Bench）–5.8 倍（Math500）；SGLang 上 Qwen3-8B 单卡 B200 最多 5.1 倍</td><td>developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding（2026-06-23，表 2、表 3；「generates an entire block of candidate tokens in a single forward pass」）</td></tr>
    <tr><td>R32 补充 · 5.1：Flash-Decoding 的 8 倍是 CodeLlama-34B、batch 1、序列 512 到 64k 的 decode 速度，对照为纯 PyTorch 注意力、FlashAttention v2（2.2 前）、FasterTransformer；注意力本身最多是 FlashAttention 的 50 倍</td><td>crfm.stanford.edu/2023/10/12/flashdecoding.html（「Flash-Decoding unlocks up to 8x speedups in decoding speed for very large sequences」「the attention itself is up to 50x faster than FlashAttention」）</td></tr>
    <tr><td>R32 补充 · 第八节：Splitwise 同成本同功耗下吞吐 2.35 倍，或吞吐 1.4 倍、成本低 20%（对不拆阶段的现有集群设计）</td><td>arXiv 2311.18677 摘要（「Splitwise clusters achieve up to 1.4× higher throughput at 20% lower cost. Alternatively, they can deliver 2.35× more throughput under the same power and cost budgets.」）</td></tr>
    <tr><td>R32 补充 · 第十二节：GPTQ 3 bit OPT-175B，batch 1 逐字延迟 1 张 A100 71 ms 对 FP16 5 张 230 ms（3.24×，正文按原文取 3.25）；2 张 A6000 130 ms 对 FP16 8 张 589 ms（4.53×）</td><td>arXiv 2210.17323 表 6（「the 3-bit OPT-175B model obtained via GPTQ running on a single A100 is about 3.25× faster than the FP16 version (running on 5 GPUs) in terms of average time per token」）</td></tr>
  </table>'''

FIGS = {}
NUM_LOCK = []
