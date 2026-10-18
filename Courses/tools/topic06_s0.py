# -*- coding: utf-8 -*-
"""专题六 · 第零节「一次请求走一遍」。图在 topic06-fig-intro.py。"""

BODY = '''
  <p class="lead">训练时答案已知，整段序列一起算，只看吞吐；推理得一个字一个字往外吐，还要守延迟。
    吞 prompt 那一段（prefill）还像训练的前向，吃算力；往外吐字那一段（decode）每步只出一个字，吃带宽。
    <b>一次请求里，前后两段要的硬件正好相反。</b></p>

  <h3 id="s0-1">0.1　2022 年以来，推理怎么变便宜</h3>
  <p>2022 年的推理服务，一批请求要等最长的那个写完才一起退出。Orca 的连续批让每一步都能有人进出，同延迟下吞吐是老办法 FasterTransformer 静态批的 36.9 倍。可座位数由 KV（已读内容的缓存）定：KV 按最大长度预留，显存里真正存着 token（大致一个字，本课常叫“字”）的只有两到近四成（4.2）。同年的 FlashAttention 把注意力切块放进片上 SRAM 算，不再把整张分数表写回显存（5.1）。</p>
  <p>2023 年 PagedAttention 把 KV 分页，利用率到 96%。浪费省掉以后，坐几个人就看 KV 池装得下几条 KV，偏偏上下文一路变长：2023 年 11 月 GPT-4 Turbo 开到 128K，2024 年 2 月 Gemini 1.5 Pro 预览 1M。模型侧用 GQA、MLA 把每条压薄；系统侧让相同的开头不再重算（SGLang 的 RadixAttention），存不下的往内存、SSD 放（Mooncake，给 Kimi 线上用）。</p>
  <p>座位坐满后，长 prompt 进门那一步会卡住所有正在出字的人。2023 年 SARATHI 把 prefill 切块塞进 decode 的步里（2024 年 Sarathi-Serve 做成完整调度），可这只摊平了打扰；Splitwise、DistServe 干脆把两段分到两批机器（PD 分离）。代价是每个请求多传一趟 KV，还多了一份 prefill 与 decode 的机器配比要调。</p>
  <p>DeepSeek-V3 这类 MoE（每个字只走几个专家）又把拼好的 batch 拆散：256 个专家选 8 个，每个专家只分到几行。SGLang 的 DP attention、DeepSeek 的大 EP 把专家铺到上百张卡上，凑够行数。随之而来的是每个 rank 要自备一份状态，K3 上 DEP8 直接显存不足；decode 快了，prefill 又喂不上。</p>
  <p>2024 年 9 月的 o1、2025 年 1 月的 DeepSeek-R1 一题要写几百到几千个思考 token（R1 论文），交互档（同样的出字速度）又限住 batch，“读一趟权重出一个字”成了硬上限。投机解码、MTP 让一趟验好几个字；剩下能省的是字节，FP8、FP4 和合并 kernel（芯片上调起的一个计算函数）接着压。同一套 GB300（NVIDIA 整柜 GPU）跑 DeepSeek-V4-Pro，光改软件，同一交互档每卡吞吐比首发日（Day-0）涨了几倍？先猜一个数，答案在 <a href="#s15-cost">15.3</a>。</p>
__FIG_S0_CHAIN__
  <details class="foldfig"><summary><b>十一环展开</b>：每一招卡在哪、省了多少、又留下什么（按环号，也就是首次公开的先后）</summary>
  <table>
    <tr><th>环</th><th>首次公开</th><th>谁先做</th><th>卡在哪 → 怎么解</th><th>省了多少（对谁）</th><th>解完又冒出什么</th><th>节</th></tr>
    <tr><td>①</td><td>2022</td><td>Orca 论文</td><td>整批人陪最慢的那个写完 → 连续批，每步都能进出</td><td>同延迟吞吐 36.9×（对 FasterTransformer）</td><td>座位数归 KV 管，按最长预留，真用上的两到近四成</td><td>三</td></tr>
    <tr><td>②</td><td>2022 → 2026</td><td>GPTQ 等量化、megakernel</td><td>大 batch 下能省的只剩字节 → FP8／FP4、合 kernel</td><td>Llama 3 70B FP8 权重，每字间隔最好降到 BF16 的 1/2</td><td>v7 上 FP4 推测只能存不能算；INT4 少了「减 8」会吐乱词</td><td>十二</td></tr>
    <tr><td>③</td><td>2022 → 2025</td><td>Google、DeepMind 论文；DeepSeek MTP、EAGLE-3</td><td>思考题动辄几千 token → 投机解码／MTP，一趟验几个字</td><td>MTP 约 1.8×；EAGLE-3 单请求最多 6.5×（都对不投机）</td><td>并发大、上下文长、开 DEP 时收益缩水，甚至倒亏</td><td>十一</td></tr>
    <tr><td>④</td><td>2023 → 2025</td><td>GQA 论文、DeepSeek</td><td>每条 KV 太厚 → GQA、MLA，再到 DSA 只挑一部分读</td><td>GQA 约 ÷8、MLA 约 ÷57（本课推导）；DSA 每字只读 2,048 条</td><td>MLA 全层一份 KV，TP 切不开，开几路就复制几份</td><td>四</td></tr>
    <tr><td>⑤</td><td>2023</td><td>vLLM</td><td>KV 预留白占显存 → PagedAttention 分页存</td><td>KV 利用率 96%；吞吐 2–4×（对 FasterTransformer、Orca）</td><td>能坐几人 ＝ KV 池 ÷ 每条 KV，而上下文在变长</td><td>四</td></tr>
    <tr><td>⑥</td><td>2023</td><td>SARATHI 论文（2024 年扩成 Sarathi-Serve）；引擎里 vLLM 先合入</td><td>长 prompt 一进门全员卡住 → 分块 prefill</td><td>Sarathi-Serve 达标请求 2.6–3.7×（对 vLLM）</td><td>只把打扰摊薄，两段还挤在同一批卡上</td><td>三</td></tr>
    <tr><td>⑦</td><td>2023 → 2025</td><td>Splitwise、DistServe → Mooncake → Dynamo</td><td>prefill 与 decode 抢卡 → PD 分离</td><td>DistServe 可接请求 7.4×（对当时最好的系统，两段同卡）</td><td>每请求多搬一趟 KV，还得调 P、D 配比</td><td>八</td></tr>
    <tr><td>⑧</td><td>2023</td><td>SGLang 论文</td><td>同样的开头一遍遍重算 → RadixAttention 复用前缀</td><td>最多 6.4×（对当时其他系统）</td><td>想复用的前缀多到显存放不下</td><td>四</td></tr>
    <tr><td>⑨</td><td>2024 → 2025</td><td>Mooncake（Kimi）、DeepSeek</td><td>前缀放不下 → 分层下放到内存、SSD</td><td>Kimi 线上多接 107%–115%</td><td>副本一多，前缀只在自己那份里有；靠 KV 感知路由把命中率从 20% 拉到 75%</td><td>四、八、十四</td></tr>
    <tr><td>⑩</td><td>2024 → 2025</td><td>Meta（CP）；vLLM 2025 合入 DCP</td><td>DEP 起不来、KV 被复制 → DCP／PCP 按序列切</td><td>DCP 把 KV 副本让出来，PCP 切 prompt 压首字</td><td>DCP 只在长上下文划算；PCP 遇上线性注意力，状态得一卡接一卡传</td><td>七</td></tr>
    <tr><td>⑪</td><td>2024 → 2025</td><td>SGLang、DeepSeek</td><td>MoE 把 batch 拆碎，每专家约 8 行 → DP attention ＋ 大 EP</td><td>SGLang 复现：最多是同卡数纯 TP 的 5×</td><td>每个 rank 自带一份状态，K3 开 DEP8 显存不够；prefill 也追不上变快的 decode</td><td>六</td></tr>
  </table>
  <p>两股外力：上下文变长（2023-11 GPT-4 Turbo 128K，2024-02 Gemini 1.5 Pro 1M），每人占的 KV 多了座位就少；输出变长（2024-09 o1，2025-01 R1），一题几百到几千个思考 token。右端那颗菱形是眼下最新的一环， GB300 × V4-Pro（2026-06），硬件、模型、交互档不动，只换软件，每卡涨到首发日的几倍，留到 15.3 揭晓。表里每个倍数各有各的基线，彼此包含，不能乘起来（第二、十五节）。</p>
  </details>
__FIG_S0_TWOLONG__
  <p>全课的账建在 DeepSeek-V3（2024 年 12 月）上：MoE，总参数 6,710 亿，每个 token 激活的参数约 370 亿。
    市面上推理这两年便宜得很快，斯坦福 AI Index 2025 主要归给更强的小模型；系统那一份单独拆不出来，15.3 再说。</p>
  <p>这几年的引擎主要是 vLLM 和 SGLang 两家轮流领先。SGLang 2024 年 1 月首发时，博客写明用了 vLLM 的部分代码；一年后 vLLM 重写成 V1，致谢里点名的五家引擎就有 SGLang，原话是「显著影响」。分块 prefill、接 Mooncake 是 vLLM 先；MLA 优化、MTP、EAGLE-3、大 EP 是 SGLang 先。更多部件两家从同一个货架上拿：FlashInfer、xgrammar、DeepEP 与 FlashMLA、Mooncake、NIXL。</p>
  <p><b>其他耳熟的名字落在哪。</b>TensorRT-LLM 是 NVIDIA 自家引擎，和这两家同一层。Triton Inference Server 2025 年改名 Dynamo-Triton，是不限于 LLM 的通用模型服务器；LLM 的 PD 分离、前缀缓存、KV 落存储，NVIDIA 交给它上面的 Dynamo。</p>
  <p>Hugging Face 的 TGI 已进入维护模式，官方改推 vLLM、SGLang 等引擎；LMDeploy 是上海 AI 实验室的引擎。llama.cpp、Ollama 跑单机和端侧，本课讲数据中心，不展开。</p>
__FIG_S0_ENGINES__
  <details class="foldfig"><summary><b>完整时间线</b>：图上只留八个时刻，其余都在这张表里</summary>
  <table>
    <tr><th>日期</th><th>谁</th><th>做了什么</th><th>跟进关系</th><th>出处</th></tr>
    <tr><td>2022</td><td>外部</td><td>Orca 迭代级调度（连续批）</td><td>两家都建在它上面</td><td>Orca，OSDI'22</td></tr>
    <tr><td>2023-06</td><td>vLLM</td><td>PagedAttention 首发，吞吐最多是 HF 的 24×</td><td>—</td><td>vLLM 博客 2023-06-20</td></tr>
    <tr><td>2023-11</td><td>外部</td><td>DeepSpeed-FastGen 提出切块拼批</td><td>vLLM 回应「已在路线图」</td><td>vLLM 博客 2023-11-14</td></tr>
    <tr><td>2024-01</td><td>SGLang</td><td>首发 RadixAttention，最多 5×（对 Guidance、vLLM 等）</td><td>原文写明复用了 vLLM 的部分代码</td><td>lmsys 博客 2024-01-17</td></tr>
    <tr><td>2024-01</td><td>外部</td><td>DistServe 论文（PD 分离）</td><td>—</td><td>arXiv 2401.09670</td></tr>
    <tr><td>2024-03</td><td>外部</td><td>Sarathi-Serve 论文（分块 prefill）</td><td>—</td><td>arXiv 2403.02310</td></tr>
    <tr><td>2024-03</td><td>vLLM</td><td>前缀缓存 APC 合入，V0 里默认关</td><td>V1 改成默认开</td><td>vLLM 合入记录；V1 博客</td></tr>
    <tr><td>2024-04</td><td>vLLM</td><td>分块 prefill 在引擎里先落地</td><td>SGLang 2024-07-29 跟进，晚 3 个多月</td><td>两家合入记录</td></tr>
    <tr><td>2024-05</td><td>外部</td><td>DeepSeek-V2 提出 MLA</td><td>—</td><td>arXiv 2405.04434</td></tr>
    <tr><td>2024-06</td><td>外部</td><td>Mooncake 论文</td><td>—</td><td>arXiv 2407.00079</td></tr>
    <tr><td>2024-07</td><td>SGLang</td><td>v0.2 纯 Python 轻量调度，点名 vLLM 的 CPU 调度开销；最多 3.1×（对 vLLM 0.5.2）</td><td>—</td><td>lmsys 博客 2024-07-25</td></tr>
    <tr><td>2024-09</td><td>vLLM</td><td>v0.6.0 承认 CPU 开销让 GPU 吃不饱，Llama 8B 上 2.7×（对 v0.5.3）</td><td>原文没说是回应 v0.2，不标</td><td>vLLM 博客 2024-09-05</td></tr>
    <tr><td>2024-09</td><td>SGLang</td><td>v0.3 先做 MLA 优化，3–7×（对基线系统）</td><td>—</td><td>lmsys 博客 2024-09-04</td></tr>
    <tr><td>2024-11</td><td>外部</td><td>Mooncake 传输引擎开源</td><td>—</td><td>Mooncake README 时间线</td></tr>
    <tr><td>2024-12</td><td>SGLang</td><td>v0.4：零开销调度、DP attention；V3 首发日支持</td><td>V3 首发日比 vLLM 早一两天</td><td>lmsys 博客 2024-12-04；两家 release</td></tr>
    <tr><td>2024-12</td><td>vLLM</td><td>接 Mooncake（12-15），同月 12-01 合入首批 PD 代码</td><td>SGLang 2025-04-10 跟进，晚 3 个多月</td><td>vLLM 合入记录；Mooncake README</td></tr>
    <tr><td>2025-01</td><td>vLLM</td><td>V1 重写，前缀缓存默认开</td><td>原文：SGLang 等引擎影响显著</td><td>vLLM 博客 2025-01-27</td></tr>
    <tr><td>2025-02</td><td>SGLang</td><td>MTP 先合入（02-14）</td><td>vLLM 02-19 跟进，晚 5 天</td><td>两家合入记录</td></tr>
    <tr><td>2025-02</td><td>外部</td><td>DeepSeek 开源周：FlashMLA、DeepEP、DeepGEMM</td><td>两家随后都接入</td><td>open-infra-index</td></tr>
    <tr><td>2025-03</td><td>SGLang</td><td>EAGLE-3（03-18），自称首个支持</td><td>vLLM 04-25 跟进，晚 5 周</td><td>两家合入记录</td></tr>
    <tr><td>2025-03</td><td>外部</td><td>NVIDIA Dynamo 开源</td><td>一上来就同时支持两家</td><td>NVIDIA 新闻稿 2025-03-18</td></tr>
    <tr><td>2025-04</td><td>vLLM</td><td>KV Connector 接口（接 LMCache、NIXL）</td><td>—</td><td>vLLM 合入记录；LMCache 博客</td></tr>
    <tr><td>2025-04</td><td>SGLang</td><td>接 Mooncake（04-10）</td><td>晚 vLLM 3 个多月</td><td>SGLang 合入记录</td></tr>
    <tr><td>2025-05</td><td>SGLang</td><td>大 EP ＋ PD 复现（96 张 H100），最多是同卡数纯 TP 的 5×</td><td>vLLM 2025-06-03 接 DeepEP，晚近 1 个月</td><td>lmsys 博客 2025-05-05</td></tr>
    <tr><td>2025-09</td><td>外部</td><td>DeepSeek-V3.2（DSA）</td><td>两家同一天首发日支持</td><td>两家博客 2025-09-29</td></tr>
    <tr><td>2025-09</td><td>SGLang</td><td>HiCache 分层 KV 缓存</td><td>—</td><td>lmsys 博客 2025-09-10</td></tr>
    <tr><td>2025-09</td><td>vLLM</td><td>合入 DCP（MLA）</td><td>—</td><td>vLLM v0.10.2 release</td></tr>
    <tr><td>2025-10</td><td>vLLM</td><td>v0.11 删掉 V0；KV 可下放 CPU；TPU 新后端 tpu-inference</td><td>—</td><td>v0.11.0 release；vLLM 博客 2025-10-16</td></tr>
    <tr><td>2025-10</td><td>SGLang</td><td>SGLang-JAX 跑 TPU</td><td>—</td><td>lmsys 博客 2025-10-29</td></tr>
    <tr><td>2025-12</td><td>vLLM</td><td>Wide-EP，每张 H200 约 1.5k → 2.2k tok/s</td><td>—</td><td>vLLM 博客 2025-12-17</td></tr>
    <tr><td>2026-03</td><td>vLLM</td><td>MRV2，吞吐 +56%（Qwen3-0.6B）</td><td>—</td><td>vLLM 博客 2026-03-24</td></tr>
    <tr><td>2026-04</td><td>外部</td><td>DeepSeek-V4</td><td>vLLM 04-24、SGLang 04-25 首发日支持</td><td>两家博客</td></tr>
    <tr><td>2026-06</td><td>SGLang</td><td>GB300 上跑 V4-Pro，同交互档对自家首发日涨了几倍（15.3 揭晓）</td><td>—</td><td>PyTorch 博客 2026-06-23</td></tr>
    <tr><td>2026-09</td><td>vLLM 生态</td><td>Inferact 的 TPU megakernel</td><td>—</td><td>Inferact 博客</td></tr>
  </table>
  <p>两家共用的外部部件：FlashInfer、xgrammar、DeepEP 与 FlashMLA、Mooncake、NIXL。间隔都按两边日期本课推导，往下取整；全部出处见第十六节台账「R85 补充」「L80 补充」。</p>
  </details>
  <h3 id="s0-2">0.2　一个请求的一生，和全课一句话</h3>
__FIG_S0_LIFE__
  <p>用户只感觉得到两个数：等第一个字多久（TTFT，含排队），之后平均每个字隔多久（TPOT）。前者看排队和 prefill，后者看 decode（第二节）。</p>
  <p><b>住在哪。</b>权重和 KV 都常驻 HBM（芯片自带的显存）。真正做乘加的地方旁边只有一小块片上存储：v7 每个 TensorCore 64 MiB VMEM，一颗芯片两个，合 128 MiB，约是 192 GiB HBM 的 1/1,500（14.2）。几十 GB 放不进去，所以每一步都得从 HBM 再流一遍。</p>
  <p><b>多大。</b>V3 出一个字要用到 370 亿参数，FP8 每个 1 字节，约 37 GB（1.1）。KV 每层每个 token 存 1,152 B（怎么来的见下面逐层图），61 层合计 70,272 B（4.1）。</p>
  <p><b>跟长度成正比。</b>上下文 1 万 token，一条 KV 约 0.70 GB；128K 是 8.58 GiB。它是这个人自己的历史，一人一份，不跟别人共享。</p>
  <p><b>出一个字，61 层逐层走一遍：</b></p>
  <p>注意力要跟之前每个 token 比一遍，所以本层 KV 一格都跳不过。前馈只有第 1 到 3 层是稠密的、整块读；第 4 到 61 层是 MoE，一个人只读他挑中的 8 个路由专家和 1 个共享专家（config 里 first_k_dense_replace 是 3）。</p>
__FIG_S0_LAYER__
  <p>每层读的权重约 0.58 GB，61 层合计 35.7 GB，加上输出层 0.93 GB，就是 1.1 那 37 GB（本课按 config 推导）。KV 每层 1 万 token 约 11.5 MB，128K 约 151 MB，已和本层注意力权重 187 MB 一个量级。</p>
  <p>几个人同时 decode 时，注意力和稠密层的权重大家共用：读一遍，每人各出一个字。专家不一样：各人挑的 8 个不同，按均匀路由算，64 人就碰到 256 个里的约 222 个，一百多人几乎挑遍，每层专家要读满 11.3 GB（本课推导，1.2）。KV 更不行，它是每个人自己的历史，各读各的（第一节）。</p>
  <div class="note ok"><span class="t">一句话</span>
    <b>decode 每一步都要把权重和 KV 从显存里搬一趟。推理的每一招，要么让这一趟分给更多 token，要么干脆少搬。</b><br>
    还有一类不碰这一趟：把两段拆开，各放合适的机器。</div>
__FIG_S0_MAP__

  <details class="foldfig"><summary><b>遇到不认识的词，回这儿查</b></summary>
  <ul style="line-height:1.9">
    <li><b>prefill</b>：整段 prompt 一次算完，顺手产出第一个输出 token。</li>
    <li><b>decode</b>：之后每步只出一个 token，再喂回去算下一个。</li>
    <li><b>KV cache</b>：每个处理过的 token 在每层留下的 K 和 V，后面的 token 算注意力要回头读，所以常驻显存（专题三）。</li>
    <li><b>TTFT</b>：请求发出到收到第一个输出 token 的时间；含不含排队要写清。</li>
    <li><b>TPOT 与 ITL</b>：TPOT 是首字之后平均每个 token 的时间；ITL 是相邻两个 token 的间隔，看 p99 用它。</li>
    <li><b>goodput</b>：满足延迟要求的那部分吞吐（第二节）。</li>
    <li><b>交互档</b>：按每个用户看到的出字速度分档；只在同一档里比每卡吞吐（第二节）。</li>
    <li><b>Little 定律</b>：系统里的人数 ＝ 到达速率 × 每人停留时间（第三节）。</li>
    <li><b>connector</b>：推理引擎里在 prefill、decode 两端收发 KV 的插件（第八节）。</li>
    <li><b>batch 与并发</b>：并发是同时在服务的请求数；batch 是引擎这一步实际拼在一起算的数，通常不超过并发。</li>
    <li><b>吞吐</b>：每秒出多少 token；要写清总量还是每卡、分母含不含 prefill 卡、分子含不含输入。</li>
    <li><b>每 token 激活的参数</b>：MoE 每个 token 只经过挑中的几个专家，这部分参数才参与计算；不是「激活值」（每层的中间结果）。</li>
    <li><b>HBM 带宽</b>：芯片每秒从自带显存读写的字节数；TPU v7 每芯片 7.37 TB/s。</li>
    <li><b>屋脊点</b>：峰值算力 ÷ 显存带宽，即每读 1 字节至少做几次运算才不被带宽拖住；v7 bf16 约 313（专题二）。</li>
    <li><b>chip 与 device</b>：一颗 TPU v7 芯片对软件显示成 2 个 device；规格按芯片写，并行度和切片名按 device 数（v7x-8 是 4 颗芯片）。</li>
    <li><b>PD 分离</b>：prefill 和 decode 分到不同机器，中间传 KV cache（专题五第六节）。</li>
    <li><b>投机解码</b>：先便宜地猜后面几个 token，大模型一步一起验，猜对几个就出几个（第十一节）。</li>
    <li><b>MTP</b>：多 token 预测，训练时在主干后多挂一层 transformer 猜再下一个字，推理时当投机解码的草稿（第十一节）。</li>
    <li><b>MXU</b>：TPU 里做矩阵乘的单元，v7 上 256×256；喂的行数不够就空转。</li>
    <li><b>TP、EP、DP、DCP</b>：四种分卡切法：TP 切开一层的矩阵；EP 把不同专家放不同卡；DP 整套复制、各接各的请求；DCP 把一条长序列的 KV 按 token 分到几张卡（第七节）。</li>
    <li><b>TEP、DEP、dep8</b>：MoE 推理的两种切法（第六节）：TEP 注意力按 TP 切；DEP 注意力按数据并行切，每卡带自己的一批人；dep8 即 8 路 DEP。</li>
    <li><b>rank</b>：参与并行的一个进程，通常一卡一个。</li>
    <li><b>NVLink 域</b>：用 NVLink 直连、能互读显存的一组 GPU；NVL72 一柜 72 块是一个域，出域走慢得多的网卡。</li>
    <li><b>切片</b>：ICI 直连的一组 TPU 芯片；切片之间走数据中心网络（DCN）。</li>
    <li><b>v7 与 v7x</b>：同一款芯片（Ironwood），云上和代码里写 v7x。</li></ul></details>

  <details class="foldfig"><summary><b>出场角色</b>：本课反复出现的模型、硬件和软件</summary>
  <ul style="line-height:1.9">
    <li><b>DeepSeek-V3 ／ R1</b>：MoE，注意力用 MLA；R1 是同架构的推理版。</li>
    <li><b>DeepSeek-V4-Pro ／ V4-Flash</b>：DeepSeek 2026-04 的 MoE；本课 GPU 实测大多来自 V4-Pro 在 GB300 上的部署记录。</li>
    <li><b>Kimi K3 ／ K2.6</b>：Moonshot AI 的 MoE；K3 的注意力里混了 KDA（Kimi Delta Attention）线性注意力层。</li>
    <li><b>Qwen3.5-397B</b>：通义千问的混合模型，60 层里 45 层是 GDN（Gated DeltaNet）线性注意力，只有 15 层存传统 KV。</li>
    <li><b>TPU v7</b>：芯片间用 ICI 直连成 3D torus。</li>
    <li><b>GB200 ／ GB300</b>：NVIDIA Blackwell 代整柜系统（NVL72）。H100、H200、H800、H20 是上一代 Hopper，后两种是出口受限版。</li>
    <li><b>vLLM、SGLang</b>：两大开源推理引擎；tpu-inference 是 vLLM 在 TPU 上的后端。TensorRT-LLM 是 NVIDIA 的引擎，Dynamo-Triton（原 Triton Inference Server）是它的通用模型服务器。</li>
    <li><b>Dynamo、llm-d</b>：多机编排层，管路由、PD 分离和扩缩容；前者出自 NVIDIA，后者跑在 Kubernetes 上。</li>
    <li><b>Mooncake、NIXL、LMCache</b>：搬 KV、存 KV 的库（第四、八节）。</li></ul></details>
'''

FIGS = {
    "__FIG_S0_CHAIN__": ("fig-s0-chain", "fig6-chain.svg", "topic06-fig-chain.py",
        '<b>点落在这一招首次公开的年份，圈里的环号从左到右就是先后。</b><br>'
        '<em>每环卡在哪、省了多少、又留下什么，展开图下的表。</em>'),
    "__FIG_S0_ENGINES__": ("fig-s0-engines", "fig6-engines.svg", "topic06-fig-engines.py",
        '<b>谁先做、谁跟进：蓝色小条是在跟 vLLM，红色小条是在跟 SGLang。</b><br>'
        '<em>间隔按两边日期推算，往下取整；只凭推测的借鉴一律不标，其余时间点在图下的表里。</em>'),
    "__FIG_S0_TWOLONG__": ("fig-s0-twolong", "fig6-twolong.svg", "topic06-fig-story.py",
        '<b>深紫只在 ② 变厚，③ 多出来的全是浅紫。</b><br>'
        '<em>格子数不是实测，只为三块能放在一起比。</em>'),
    "__FIG_S0_LIFE__": ("fig-s0-life", "fig6-life.svg", "topic06-fig-intro.py",
        '<b>红点是第一个字，之后每个绿格出一个字；紫色阶梯每一步都在长。</b><br>'
        '<em>示意时间线，长短不按比例。</em>'),
    "__FIG_S0_LAYER__": ("fig-s0-layer", "fig6-layerwalk.svg", "topic06-fig-layerwalk.py",
        '<b>每一行都读两样：蓝绿那条大家共用，紫色那条跟着上下文变长、一人一份。</b><br>'
        '<em>61 层加输出层的权重对上 37 GB；全部推导见第十六节台账「L83」。</em>'),
    "__FIG_S0_MAP__": ("fig-s0-map", "fig6-map.svg", "topic06-fig-intro.py",
        '<b>蓝色那组节最多：批处理、切专家、切序列、投机解码、新架构，都在让一趟多坐几个人。</b><br>'
        '<em>归类是本课的分法，标签是讲这一招的节。</em>'),
}

NUM_LOCK = [
]
