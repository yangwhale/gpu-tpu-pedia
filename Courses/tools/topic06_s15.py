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
    <tr><td>TPU v7：FP8 4,614 TFLOP/s、HBM 192 GiB（≈ 206.2 GB）、7.37 TB/s，每芯片 2 个 device；BF16 2,307 TFLOP/s，屋脊点 ≈313 FLOP/字节</td><td>Google Cloud TPU7x 公开规格页；JAX jax/_src/pallas/mosaic/tpu_info.py 记每芯片 HBM 206e9 字节（≈ 191.9 GiB），即规格表的 192 GiB；BF16 与屋脊点本课推导：4,614 ÷ 2，2,307 TFLOP/s ÷ 7.37 TB/s</td></tr>
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
    <tr><td>DeepSeek 线上实测：226.75 节点、理论成本利润率 545%（全按 R1 定价收费的假设）、每节点 14,800 tok/s、20–22 tok/s/user</td><td>GitHub deepseek-ai/open-infra-index：202502OpenSourceWeek/day_6</td></tr>
    <tr><td>每 token KV：MHA 3,997,696 B、GQA（Llama-3.1-70B）327,680 B、Qwen3.5 全注意力 30,720 B ＋ 线性状态 180 MiB／序列；Qwen3.5 KV 头数 2</td><td>本课推导，形状取各自 Hugging Face config.json</td></tr>
    <tr><td>8 颗 v7 放 V3：KV 池 92.4 GiB／芯片，4K 装 344 条、32K 装 43 条、64K 装 21 条（FP8 KV 43 条）；Llama 4K 73 条</td><td>本课推导：每芯片可用 189.5 GiB（每 device 实测 94.75 GiB × 2，R1 FP4 README:59，同台账里「V3 FP8 切片表」那一行）× 0.9 ＝ 170.55 GiB，− 78.1 GiB（671 GB ÷ 8）；0.9 为本课假设；先按权重全切分粗算，DP attention 下非专家权重每 device 复制，实际更小（10.5）</td></tr>
    <tr><td>PagedAttention：KV 有效利用率 20.4–38.2% → 96.3%</td><td>vLLM 论文，arXiv 2309.06180</td></tr>
    <tr><td>Qwen3.5 RPA KV 页 16→256 token：并发 512 时这个 kernel 的 decode 一步 428 → 283 µs（原文称 kernel 级提速 33.8%，不是整步）</td><td>原文「reduced the decode step latency under Concurrency-512 from 428µs to 283µs, achieving a 33.8% kernel-level speedup」；gpu-tpu-pedia tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/BLOG-systems-engineering-playbook.md L395-405，即 Google Developers Blog「Systems Engineering Playbook: Optimizing Qwen 3.5-397B MoE on Ironwood (TPU7x)」</td></tr>
    <tr><td>V4 两个池（full／swa）预算比改 0.20 → 0.15：每 rank 请求 727 → 886，端到端每 decode 卡 6,887 → 10,614（＋54%）</td><td>SGLANG-V4PRO-RUNBOOK.md:855-857（§11.5 L849-896）；swa_tokens 比例 L946（引 pool_configurator.py:387）；sliding_window 128 见 HF config</td></tr>
    <tr><td>再把 mem-fraction-static 0.85 → 0.88：每 rank 请求 886 → 999（＋13%），端到端 10,614 → 10,704</td><td>SGLANG-V4PRO-RUNBOOK.md:857-858、887-892</td></tr>
    <tr><td>Qwen3.5 混合池的 bug 与 prefix caching 断言</td><td>tpu/tpu-inference Qwen3.5 README:77、:95</td></tr>
    <tr><td>RadixAttention 吞吐最多 6.4×</td><td>SGLang arXiv 2312.07104 摘要</td></tr>
    <tr><td>KV 外存回读的盈亏带宽：V3 2.19 GB/s、Llama 5.35 GB/s，1 GB/s 的慢链路在前缀长于 ~35K 时也比重算快</td><td>本课推导：重算 ＝ (2 × 37e9 × L ＋ 4,997,120 × L²/2) ÷ (4,614 T × 50%)；Llama 激活 70.6B 取 HF 模型页</td></tr>
    <tr><td>Orca 迭代级调度：同延迟下吞吐是 FasterTransformer 的 36.9×</td><td>Orca，OSDI'22</td></tr>
    <tr><td>chunked prefill；Sarathi-Serve 达标容量是 vLLM 的 2.6–5.6×；Gemma4 一次 prefill 16,384 token、切 16 块</td><td>Sarathi arXiv 2308.16369；Sarathi-Serve arXiv 2403.02310；vLLM 文档 Optimization and Tuning 页；Gemma4-31B/README.md:225-226</td></tr>
    <tr><td>并发 400 → 1,800：总负载 4.5×，每卡只多 11%（8,777 → 9,745 tok/s）；有记录的 TTFT 从并发 600 的 45 s 涨到 1,800 的 253 s</td><td>SGLANG-V4PRO-RUNBOOK.md:593-609</td></tr>
    <tr><td>Little 定律核对：dep8 4,335、dep16 2,783</td><td>SGLANG-V4PRO-RUNBOOK.md:794-801；本课推导 9,032 × 8 × 0.060、84,321 × 0.033</td></tr>
    <tr><td>多给槽位：并发 886 → 999，峰值只 12,063 → 12,070，TPOT 77.7 → 85.3 ms</td><td>SGLANG-V4PRO-RUNBOOK.md:849-858、885-894</td></tr>
    <tr><td>SGLang-JAX 调度开销 12 ms → 38 µs；按 batch 尺寸预编译</td><td>lmsys.org/blog/2025-10-29-sglang-jax</td></tr>
    <tr><td>vLLM 抢占默认 RECOMPUTE；公平调度 VTC；跨实例迁移 Llumnix</td><td>docs.vllm.ai「Optimization and Tuning」页；arXiv 2401.00588；arXiv 2406.03243</td></tr>
    <tr><td>Llama-3.1-70B 权重 141,107,412,992 字节；8K 一条 KV 2.68 GB；batch ≈52.6 时两者一样重</td><td>HF safetensors index 的 total_size；其余本课推导</td></tr>
    <tr><td>TPU 内存 tile 规则（每 KV 头补齐到整块）</td><td>openxla.org/xla/tiled_layout</td></tr>
    <tr><td>Ragged Paged Attention 的布局与 kernel 设计</td><td>RPA，arXiv 2604.15464 §2.1.2、§3.1、§3.4、§3.5</td></tr>
    <tr><td>MLA 吸收后 decode 算术强度 242，prefill 每对运算量 3.4×</td><td>DeepSeek-V2 arXiv 2405.04434 §2.1.2；本课推导 278,528 ÷ 1,152、278,528 ÷ 81,920</td></tr>
    <tr><td>MLA 每 token KV 只有同形状 MHA 的约 1/57（故事图卡片 ÷57 同此）</td><td>本课推导：70,272 ÷ 3,997,696（3.1 表）</td></tr>
    <tr><td>单专家 44,040,192 参数；单 device 读权重 23.9 µs（bf16）／11.95 µs（fp8），稠密算 19.5／9.77 µs</td><td>本课推导：3 × 7168 × 2048；3.685 TB/s ＝ 7.37 ÷ 2，1,153.5／2,307 TFLOP/s ＝ 每芯片 ÷ 2</td></tr>
    <tr><td>decode AllReduce 11.0 → 2.37 µs，SoL 1.404 µs，ITL −8.7%</td><td>arXiv 2607.16100</td></tr>
    <tr><td>14,336 字节消息的带宽项只要 24 ns，实测是它的 460×</td><td>本课推导：7168 × 2 B；NVLink5 1.8 TB/s 双向 → 单向 900 GB/s</td></tr>
    <tr><td>V4-Pro 只有 1 个 KV 头，TP 切注意力会复制 KV</td><td>VLLM-V4PRO-RUNBOOK.md:298-302</td></tr>
    <tr><td>并发 512 时 TP4 21,100（16 卡）对 DEP 55,153（20 卡）：每卡 1,319 对 2,758，2.09×；DEP 并发 1,536 时 65,132、每卡 3,257、2.47×，TTFT 95 s；输出 4,220 对 11,031 tok/s、出字间隔 46.8 对 11.8 ms → 在飞请求约 197 对 130（Little 定律），dep8 反而少；每张 decode 卡输出 1,055 对 1,379，1.31×；投机接受率 34% 对 35.7%</td><td>VLLM-V4PRO-RUNBOOK.md:449、464-468、476；每卡与 2.09× 本课推导；接受率 :480；在飞请求 ＝ 输出 tok/s × TPOT、每 decode 卡 ＝ 输出 ÷ 4 或 8，本课推导（runbook 原文把收益归给容量，本课按 Little 定律归给步时）</td></tr>
    <tr><td>DEP 每卡吞吐随 prefill 喂料 307 → 12 个 prefill 时 3,187（14 个时 3,321）；TP4 停在 20,479</td><td>VLLM-V4PRO-RUNBOOK.md:711-724（4k1k 重跑 :727-734）、:488，2026-07-24 记录</td></tr>
    <tr><td>tpu-inference DEP：并发 512 时领先 62%（1k/8k）、40%（8k/1k）；mmlu_pro 0.8129 对 0.8271</td><td>github.com/vllm-project/tpu-inference/pull/2577（2026-05-15 合入）</td></tr>
    <tr><td>大 EP：DeepSeek decode 用 EP144（18 个节点），每卡 2 个路由专家 ＋ 1 个共享专家；EP 的两条理由（放大 batch 提高矩阵效率、每卡只读少数专家降低延迟）；一批拆两个微批交替藏通信；跨节点 all-to-all 用 DeepEP</td><td>github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md；「空出的显存给 KV」本课推导</td></tr>
    <tr><td>TEP／DEP 的定义；Megatron 命名；Folding</td><td>沿用专题五台账第 43、44、18 行</td></tr>
    <tr><td>K3 DCP：128K 并发 8 吞吐 108 → 365（＋237%）；4K 并发 8 −9%；8K 并发 8 −11%、并发 32 ＋24%</td><td>gpu-tpu-pedia SGLANG-K3-RUNBOOK.md §11.0.3 :1010-1023、宽表 :921-938；百分比本课推导</td></tr>
    <tr><td>DCP 让 KV 容量 1.5M → 12.2M token；单流 −26% 来自丢掉 symm-mem</td><td>SGLANG-K3-RUNBOOK.md:721-723；:1024-1026、:962</td></tr>
    <tr><td>69 层 KDA 线性注意力不能按序列切；切了状态差 58%</td><td>SGLANG-K3-RUNBOOK.md:725-726；58% 为本课玩具递推 S<sub>t</sub> ＝ 0.9·S<sub>t−1</sub> ＋ 1</td></tr>
    <tr><td>KV 复制；DCP 通信；PCP／DCP 分工；Ring／Ulysses／USP；Helix</td><td>沿用专题五台账第 23、29、45、21、47 行（Helix 即 arXiv 2507.07120）</td></tr>
    <tr><td>KV 传输软件栈</td><td>github.com/ai-dynamo/nixl；github.com/kvcache-ai/Mooncake；LMCache arXiv 2510.09665；docs.nvidia.com/dynamo 的 KVBM；github.com/google/tpu-sync（~200 GB/s，not yet recommended for general use）；tpu-inference kv_transfer（jax.experimental.transfer）</td></tr>
    <tr><td>PD 传输带宽从 200 MB/s 调到 7–167 GB/s</td><td>VLLM-V4PRO-RUNBOOK.md:194，配置见 :106、180-194</td></tr>
    <tr><td>Qwen3-Coder 一条 8K 请求 KV 1.04 GB；各链路传完要 5,201／149／6.2 ms</td><td>本课推导：2 × 62 × 8 × 128 × 8192 字节（FP8）；83 ms 与 PCIe 数沿用专题五台账第 25 行</td></tr>
    <tr><td>scale-out 网卡：A4X（GB200）每 GPU 400 Gbps（每节点 4 GPU、4×CX-7 400G）；A4X Max（GB300）每 GPU 800 Gbps（CX-8 双端口）；1.04 GB KV 一趟约 21 ms／10 ms</td><td>gpu/a4x/01-environment-setup/README.md:35；gpu/a4x-max/05-rdma-test/README.md:197-200；时间 ＝ 1.04 GB × 8 ÷ 400 或 800 Gbps 本课推导</td></tr>
    <tr><td>Qwen3.5 混合模型 PD：45 层 GDN ＋ 15 层注意力，状态 349 MB；单请求吞吐是单机的 0.95 倍（单并发）／0.92 倍（4 并发）</td><td>tpu_connector_hma.py 注释；Qwen3.5 README:375-389；README-TPU-VM.md:859-867</td></tr>
    <tr><td>PD 踩坑清单</td><td>SGLANG-V4PRO-RUNBOOK.md:941、:63；VLLM-V4PRO-RUNBOOK.md:760、:106、:703、§12.4</td></tr>
    <tr><td>跨数据中心 PD：链路 33.35／8.25／59.93／4.66 Gbps；案例 ＋54%、P90 −64%、13 Gbps、阈值 19.4K</td><td>PrFaaS，arXiv 2604.15039 Table 3 与案例节</td></tr>
    <tr><td>P:D 配比公式：87.5 req/s、要 71.7 万 tok/s prefill、一台 7.28 万，约 9.8 台</td><td>SGLANG-V4PRO-RUNBOOK.md §12.2 :1082-1091</td></tr>
    <tr><td>B200 1P1D → 2P1D：输出 1,118.8 → 1,159.0，每卡 69.9 → 48.3，TTFT 2.40 → 1.73 s（−28%），P99 12.07 → 6.23</td><td>gpu-tpu-pedia gpu/inference/a4/deepseek-v3/REPORT-2P1D-DeepSeek-V3.md :69、:85、:100-142；cuda graph 说明 REPORT-1P1D :129</td></tr>
    <tr><td>prefill 台数扫描（扫到 12 个 prefill）：按每张 decode 卡算 307 → 3,187，10.4×；按全部卡算 204 → 455，2.2×</td><td>VLLM-V4PRO-RUNBOOK.md §12.3；全卡 ＝ output ÷ (4n＋8) 本课推导</td></tr>
    <tr><td>MiMo batch 128：同样 2 个 pod，1P1D 13.64K 对不分离 12.78K，＋6.7%（README 写约 6.8%）；2P1D 用 3 个 pod 到 15.31K，按 pod 算 5.10K 对 6.39K，低 20%；Qwen3-Coder 1P1D TTFT 95 → 281 ms</td><td>tpu/sglang-jax/mimo/README.md:8-12、81-88；6.7%、按 pod 与 20% 本课推导；tpu/tpu-inference/Qwen3-Coder-480B/README.md:1005-1021</td></tr>
    <tr><td>AFD：MegaScale-Infer、Step-3；vLLM AFD 插件 16K／32K 实测；EPD 把多模态编码再拆出去</td><td>概念沿用专题五台账第 24、51 行；arXiv 2504.02263；arXiv 2507.19427；vllm.ai/blog/2026-07-23-vllm-afd-plugin、github.com/vllm-project/afd-plugin（RFC #22799 于 2026-05-31 关闭）；EPD arXiv 2501.05460；Step-3 对 V3 不是同一模型，差距含模型侧设计，正文已注明</td></tr>
    <tr><td>同一模型五种部署 14,563／16,459／21,100／55,153／65,132；调参 ＋45%，换拓扑 ＋347%（并发 1,536）</td><td>VLLM-V4PRO-RUNBOOK.md:446、448、449、465、468；:489（另 :476 的 2.47×、:487 的 ＋28%）</td></tr>
    <tr><td>每卡 1,820／1,319／3,257；对 1p1d 起点 1.79×</td><td>本课推导：总吞吐 ÷ 卡数</td></tr>
    <tr><td>Qwen3-Coder 跨 DCN 掉 17–63%，同 ICI 只掉 15–21%</td><td>Qwen3-Coder-480B/README-multinode.md:441-445；README-TPU-VM.md:1000-1009</td></tr>
    <tr><td>V4-Pro dep8 要一整柜 NVL72；拓扑与频率的一般讨论</td><td>SGLANG-V4PRO-RUNBOOK.md:34、:63；沿用专题五台账第 19、26 行</td></tr>
    <tr><td>v7x-8 是 4 颗芯片（名字按 device 数）；mesh 构造</td><td>tpu/tpu-inference/README.md:41；JAX mesh_utils 文档</td></tr>
    <tr><td>v7 每 device 实测可用 94.75 GiB（每 device 物理 96 GiB）；R1 FP8 MoE 每 device 101.8 放不下（README 写 GB，按 GB、GiB 读都超）</td><td>tpu/tpu-inference R1 FP4 README:59、72-73</td></tr>
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
    <tr><td>草稿模型谱系（MTP、DSpark 的定义；V3 的 MTP 是一整层 transformer，共享 embedding 和输出头；EAGLE-3 2025-03 融合多层状态，SGLang batch 64 吞吐 1.38×）</td><td>SpecInfer 2305.09781、Medusa 2401.10774、EAGLE 2401.15077／2406.16858、MTP 2412.19437、EAGLE-3 2503.01840、DSpark 2607.05147、Lookahead 2402.02057、SuffixDecoding 2411.04975、DiffuSpec 2510.02358、DFlash 2602.06036（均 arXiv）；github.com/apoorvumang/prompt-lookup-decoding；MTP 结构见 DeepSeek-V3 技术报告 §2.2 MTP Modules；EAGLE-3 摘要页（v1 2025-03-03）。2026-10-15 核对</td></tr>
    <tr><td>8 颗 v7 读一遍 V3 激活权重：BF16 1.26 ms、FP8 0.63、MXFP4 0.33、NVFP4 0.35</td><td>本课推导：37e9 × 位数 ÷ 8 ÷ (8 × 7.37 TB/s)</td></tr>
    <tr><td>MXFP4 每 32 个数一个 E8M0 scale（只能取 2 的幂），合 4.25 bit；NVFP4 每 16 个一个 E4M3（可取非 2 的幂），合 4.5 bit</td><td>OCP Microscaling Formats (MX) v1.0 规范；NVIDIA 技术博客「Introducing NVFP4」</td></tr>
    <tr><td>v7 公开规格只列 BF16、FP8 峰值，没写 FP4 算力</td><td>Google Cloud TPU7x 公开规格页</td></tr>
    <tr><td>V4 专家权重为 FP4</td><td>Hugging Face DeepSeek-V4 模型卡</td></tr>
    <tr><td>R1 FP4 每 device 60.9，FP8 101.8 放不下（README 写 GB）；上限 94.75 GiB（运行日志单位）；K2.6 INT4 存成无符号 0–15，加载时直接当有符号读（没减 8）就崩</td><td>R1 FP4 README:59、72-73、:875-878（README 实测值）；K2.6 README:217、:265-266</td></tr>
    <tr><td>kernel 气泡：H100 理想 1,350 次前向／秒，实际 770；megakernel 带宽利用 50% → 78%，比 vLLM 快约 2.5×、比 SGLang 快 1.5× 多；Llama-3.2-1B、batch 1，小模型小 batch 收益最大；PDL 同步粒度太粗</td><td>hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles；1,350 ＝ 3.35 TB/s ÷ 2.48 GB 本课推导</td></tr>
    <tr><td>Mirage MPK 编译成 megakernel：1.7×；2025-06 开源，论文 2025-12</td><td>arXiv 2512.22219（v1 2025-12-22）；github.com/mirage-project/mirage（2025-06-19 公开，作者博客「Compiling LLMs into a MegaKernel」）</td></tr>
    <tr><td>TPU megakernel：16 颗 v7 单用户 700＋ tok/s，对照 GB200（vLLM）452，两边都按接受长度 6 换算；一步 ~8.5 ms，编译 &lt;90 s；不开投机 batch 1–8 约为 GB200 基线 1.4–2×；博客写的是 megakernel ＋ DSpark 投机，未提量化</td><td>https://inferact.ai/blog/tpu-megakernels（2026-09-23；博客图注「Output tokens per second with speculative decoding at an acceptance length of 6」同时覆盖两条曲线）；6 ÷ 8.5 ms ≈ 706 本课推导</td></tr>
    <tr><td>KV 存 FP8：FlashMLA 给 V3.2 的布局每 token 每层 656 B（512 个 FP8 ＋ 4 个 FP32 缩放 ＋ 64 维 BF16 RoPE 不量化），是 BF16 1,152 B 的 57%，同显存多装约 1.76×；K3 Peak Throughput 预设 --kv-cache-dtype fp8_e4m3 必开（BF16 KV 装不下每副本 128 个请求）</td><td>github.com/deepseek-ai/FlashMLA README（2026-09-30 提交 2e5429f 之前的版本，「FP8 with scale」一节；现版只剩 V4.1 的 528／288 B 格式）；本课推导 656 ÷ 1,152、1,152 ÷ 656；SGLANG-K3-RUNBOOK.md:1308</td></tr>
    <tr><td>分桶浪费：一步整批 token 总数只补到一个 8K 桶时浪费 87.3%，32 个桶 10.8%；补齐按整批总数（RPA ragged 布局），单条请求不补</td><td>本课推导：每步 token 总数取对数正态，中位 700、σ 0.9；RPA arXiv 2604.15464；预编译参数 mimo README:269-270</td></tr>
    <tr><td>新形状首个请求现编译 60–90 s；预编译开关</td><td>Qwen3.5 README:794；mimo README:269-270</td></tr>
    <tr><td>13.4：强杀后每卡剩 187 GB，mem-fraction-static 0.85 要静态占 241 GB（权重＋KV 池），下次启动 OOM</td><td>SGLANG-V4PRO-RUNBOOK.md:966（284 × 0.85 ≈ 241）</td></tr>
    <tr><td>冷启动拆账：GLM-5.1 名义加载里藏 6 分 29 秒重编译（原表总计 ~10 分 44 秒，分项和 10 分 29 秒）；Qwen3-Coder 40 分钟里编译 29；MiMo-V2.5-Pro 编译 15 ＋ 预热 7（原表总计 ~50 分钟，分项和 52 分钟）</td><td>GLM-5.1 README:470-476；Qwen3-Coder README-TPU-VM.md:1011；MiMo-V2.5-Pro README:184-188</td></tr>
    <tr><td>R1 加载 215.7 → 21.4 s（1,367 个张量只有 25 种形状）；K2.6 57 分钟 → 6 分 8 秒（9.3×）；持久编译缓存</td><td>R1 README:1202-1223、:1238-1259；K2.6 README:23、39-40；JAX 文档「Persistent compilation cache」</td></tr>
    <tr><td>KV 感知路由：命中缓存首字 340 ms，冷启动（不命中）2,850 ms，命中率 87.4%（4,176／4,776）</td><td>Red Hat Developer 2025-10-07「Master KV cache aware routing with llm-d」</td></tr>
    <tr><td>JetStream 停止维护：核心功能并入 tpu-inference、2026-02-01 起归档，最后提交 2026-01-05</td><td>github.com/AI-Hypercomputer/JetStream README 公告与提交记录，2026-10 核对</td></tr>
    <tr><td>第十四节八问的顺序</td><td>本课归纳</td></tr>
  </table>'''

FIGS = {}
NUM_LOCK = []
