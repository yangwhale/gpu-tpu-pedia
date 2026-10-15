# -*- coding: utf-8 -*-
"""专题六 · 第八节「拆工作：PD 分离与那一趟 KV」。

新图在 topic06-fig-pd.py（fig6-kv-stack / kv-bw / hybrid-kv / prfaas）；
另复用专题五 §6.1、§6.2 的三张图（topic05-fig-pd.py，id 加 -t5）。
⛔ 本节不用 Kimi K3 在 TPU 上的任何 PD 数；K3 只引 GPU runbook。
"""

BODY = '''
  <p class="lead">2023 年底 Splitwise、2024 年初 DistServe 先后提出拆开 prefill 和 decode，DistServe 报告同样的延迟要求下多接 7.4 倍请求；同年 Mooncake 把它做进了 Kimi 的线上服务。prefill 和 decode 放在同一批卡上会互相拖。PD 分离把它们拆到两批机器上，代价是每个请求多搬一趟 KV。
    <b>这一节讲那一趟怎么搬、谁来搬、搬错了是什么样子。</b></p>

  <h3 id="s8-1">8.1　回顾：为什么要拆</h3>
  <p><a href="topic-05.html#s6-1">专题五 §6.1</a> 讲过：decode 每步只出一个字，读一遍权重只够这一批用一次，所以它拼命凑大 batch。</p>
__FIG_S8_DECODE_AI_T5__
  <p>这时插进来一个长 prompt，它的 prefill 要整块占住算力好几步，那几步里所有人都不出字。拆开以后，decode 机器上只剩出字这一件事。</p>
__FIG_S8_PD_T5__
  <ul>
    <li><b>拆开换来的</b>：出字不被打断，TPOT 稳；两边还能各挑各的切法（第六节、第七节）。</li>
    <li><b>拆开要付的</b>：一趟 KV 传输、两套机器、一份要调的配比（第九节）。</li>
  </ul>

  <h3 id="s8-2">8.2　回顾：那一趟要多久</h3>
  <p><a href="topic-05.html#s6-2">专题五 §6.2</a> 算过一个 8K prompt：KV 约 1.04 GB，走 100 Gbps 的数据中心网络约 83 ms，前后两段 PCIe 各约 16 ms（按 Gen5 x16 单向约 64 GB/s 估，本课推导）。</p>
__FIG_S8_KV_TRIP_T5__
  <p>和 1–2 s 的 prefill 比，这一趟只占一成上下。这个结论有两个前提：网络真跑在设计带宽上；KV 本身没有大到离谱。8.4 讲第一个前提怎么被打破，8.8 讲第二个。</p>

  <h3 id="s8-3">8.3　三层软件：谁搬、谁派、谁存</h3>
  <p>一说「KV 传输」，人们常常把三件事混在一起。<b>先分清层，出了问题才知道去哪一层找。</b></p>
__FIG_S8_STACK__
  <ul>
    <li><b>搬运</b>：只管把字节从这块显存挪到那块。GPU 上是 NIXL、Mooncake Transfer Engine；TPU 上是 TPU Sync 和 JAX 自带的 transfer server。</li>
    <li><b>调度</b>：决定哪个请求先去哪台 prefill、算完交给哪台 decode。tpu-inference 的代理先把请求截成只出一个字发给 prefill，再让 decode 端去拉 KV。</li>
    <li><b>存储</b>：KV 不一定只在显存里。Mooncake Store、LMCache、Dynamo KVBM 把它放到主机内存、SSD 乃至远端，让下一个同前缀的请求直接取。</li>
  </ul>
  <p>TPU 这边组件少得多。TPU Sync 刚开源，自己写明还不建议通用场景使用；tpu-inference 的 TPUConnector 是目前能直接用的那条路。</p>
  <p>排查按层找：带宽掉到几百 MB/s 是搬运层没走对通道；请求接不上、握手失败多半在调度层；前缀命中率低看存储层。</p>

  <h3 id="s8-4">8.4　GPU 实战：配错一个开关，慢两个数量级</h3>
  <p>vLLM 在 GB300 上跑 DeepSeek-V4-Pro 的 PD，起初 KV 传输只有 200 MB/s。按 1.04 GB 这把尺子算，一趟要 5,201 ms，比 prefill 本身还长。1.04 GB 沿用专题五的统一尺子（Qwen3-Coder-480B 一条 8K prompt 的 KV），不是 V4-Pro 的实际 KV。</p>
__FIG_S8_BW__
  <p>原因是 NIXL 没走 cuda_ipc（直接读对端 GPU 的显存），退回了 cuda_copy（经主机内存中转）。cuda_ipc 只在同一个 NVLink 域里成立：一组用 NVLink 全互联、能直接读彼此显存的 GPU，NVL72 就是 72 张。补上三样配置后实测 7–167 GB/s，同一份 KV 一趟 149 ms 到 6.2 ms。</p>
  <p>runbook 里单次传输从 200–800 ms 降到 2–10 ms。配错时什么都不报错，只是 TTFT 慢得说不清原因，上线前先量一次 KV 传输带宽。</p>
  <details class="foldfig"><summary><b>那三样配置</b>：vLLM ＋ NIXL 在 GB300 上走 NVLink</summary>
  <ul>
    <li><code>UCX_TLS=cuda_copy,cuda_ipc,tcp</code>：只留 cuda_ipc（NVLink）和必要的拷贝、控制通道，去掉 RDMA。</li>
    <li><code>UCX_CUDA_IPC_ENABLE_MNNVL=y</code>：允许跨节点的 cuda_ipc。</li>
    <li><code>--enable-cumem-allocator</code>：显存用 cuMem 分配，cuda_ipc 才认得出来。</li>
  </ul>
  <p>另外 prefill 和 decode 必须在同一个 NVLink 域（GKE 上用 subblock 标签做 podAffinity）。出处：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:106、180–194。</p></details>
  <p>更麻烦的是，<b>同一个库在两个引擎上的结论正好相反</b>。runbook 里试过的几条路：</p>
  <table>
    <tr><th>引擎 · 场景</th><th>试了什么</th><th>结果</th></tr>
    <tr><td>vLLM · V4-Pro · GB300</td><td>NIXL 走 NVLink cuda_ipc</td><td>能用，这是唯一活路</td></tr>
    <tr><td>vLLM · V4-Pro · GB300</td><td>Mooncake 走 NVLink ＋ 简易代理</td><td>KV 快了，整体反而慢：并发 64 输出 1,333 对 NIXL 的 1,693 tok/s</td></tr>
    <tr><td>vLLM · V4-Pro · GB300</td><td>Dynamo ＋ vLLM ＋ MooncakeConnector</td><td>接不上：Dynamo 的 vLLM 集成没有传 Mooncake 握手地址的通路</td></tr>
    <tr><td>SGLang · DeepSeek-R1 · GB300</td><td>NIXL 走 RoCE</td><td>GKE 上调不通，请求 60 s 超时</td></tr>
    <tr><td>SGLang · R1 ／ Kimi K3 · GB300</td><td>Mooncake 走 NVLink</td><td>能用，K3 的部署沿用了这条</td></tr>
  </table>
  <p>所以别问「NIXL 和 Mooncake 哪个好」，要问「在这个引擎、这张网上，哪条路真的通」。两边的共同点只有一个：<b>同一个 NVLink 域里，KV 就别出域。</b></p>

  <h3 id="s8-5">8.5　混合模型：尾巴不能让 decode 端重算</h3>
  <p>纯注意力模型的 KV 按 token 一格一格存。老的 connector 只传填满的整块，最后那截不满一块的尾巴让 decode 端自己补算，省一点传输。</p>
__FIG_S8_HYBRID__
  <p>GDN、Mamba 这类层存的是一份递归状态，它已经把整段 prompt 吃进去了。decode 端再补算尾巴，等于把这几个 token 喂了两遍，输出就偏了，而且不报错。</p>
  <ul>
    <li><b>修法</b>：全部 block 都传，状态也跟着传，decode 端一个 token 都不重算。tpu-inference 为此单独写了 HMA 版的 connector。</li>
    <li><b>代价</b>：Qwen3.5-397B 在 TPU v7x 上 1K 进 1K 出，每个请求搬约 349 MB（README 原文，没给分解）；单并发每请求吞吐是单机的 0.95 倍，4 并发是 0.92 倍。</li>
    <li><b>GPU 上同理</b>：SGLang 跑 Kimi K3 时，PD 的 decode 端每个请求只有一个状态槽，原来的前缀缓存策略在 PD 下不起作用。</li>
  </ul>
  <details class="foldfig"><summary><b>Qwen3.5 PD 的启动要点</b>：tpu-inference</summary>
  <p>connector 选 <code>TPUConnectorHMA</code>，并显式加 <code>--no-disable-hybrid-kv-cache-manager</code>。出处：tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:375–389；吞吐表见同目录 README-TPU-VM.md:859–867（XLA 缓存已热的第二轮）。</p></details>

  <h3 id="s8-6">8.6　PD 坑清单</h3>
  <ul>
    <li><b>两边上下文长度要一致</b>：只改 decode 的 <code>--context-length</code>，KV 布局对不上，握手失败；单条请求能过，压测才暴露。</li>
    <li><b>失败策略两头都疼</b>：vLLM 的 <code>kv_load_failure_policy</code> 设 fail，一次瞬时拒收就让引擎死掉；设 recompute 稳，但 12 个 prefill 时每卡 3,187 掉到 2,645，约交 17% 的税。</li>
    <li><b>出了 NVLink 域就慢</b>：SGLang 用 Mooncake 走 NVLink，跨域就退到 RDMA；vLLM 的 cuda_ipc 干脆只在域内成立。</li>
    <li><b>表面报错可能是假的</b>：vLLM 开 14 个 prefill 时 decode 的 pod 接连被驱逐，真因是崩溃留下的 core dump 撑爆了临时盘。它和第十三节 SGLang 那次「14 台 prefill 全崩、注册表还显示在线」是两件事。</li>
    <li><b>编排层和 connector 要配套</b>：Dynamo 的 SGLang 集成能传 Mooncake 握手地址，vLLM 集成不能。</li>
  </ul>
  <details class="foldfig"><summary><b>出处</b>：五条坑各在哪</summary>
  <p>上下文长度：SGLang V4-Pro runbook:941。失败策略：vLLM V4-Pro runbook:760。出域：SGLang V4-Pro runbook:63、vLLM V4-Pro runbook:106。core dump 驱逐：vLLM V4-Pro runbook 12.4 节。编排层：vLLM V4-Pro runbook:703。两份 runbook 都在 gpu/inference/a4x-max/deepseek-v4/。</p></details>

  <h3 id="s8-7">8.7　正确性：拆了以后先比字</h3>
  <p>PD 出错最常见的样子是「能跑、输出看着通顺、但和不拆时不一样」。<b>所以上线前先比字，再比速度。</b></p>
  <ul>
    <li><b>怎么比</b>：temperature 设 0，同一批 prompt 拆与不拆各跑一遍，逐个 token 对。</li>
    <li><b>前几个 token 就分叉</b>：多半是 bug，尤其是多条 prompt 都在头几个 token 分叉时，比如尾巴被重算、递归状态没传、两边上下文长度不同。</li>
    <li><b>几百个 token 后才分叉</b>：可能只是批大小不同带来的浮点差异。Thinking Machines 的文章《Defeating Nondeterminism in LLM Inference》讲过，kernel 不做批不变时，同一个 prompt 换个批就会出不同的字。</li>
  </ul>
  <p>后一种要靠批不变的 kernel 才能完全消掉；前一种必须修。分辨不清时，把并发降到 1 再比一次。</p>

  <h3 id="s8-8">8.8　跨数据中心：KV 够薄才敢走远路</h3>
  <p>专题五那个「网络不是瓶颈」只在同一个机房成立。Moonshot AI 和清华的 PrFaaS 论文把 prefill 放到另一个数据中心，前提是混合模型的 KV 薄了很多。</p>
__FIG_S8_PRFAAS__
  <ul>
    <li><b>全注意力太胖</b>：32K 输入时，一台 8×H200 做 prefill，Qwen3-235B 每秒要外送 33.35 Gbps，MiniMax-M2.5 接近 60。</li>
    <li><b>混合模型能走</b>：Qwen3.5-397B 只要 8.25，MiMo-V2-Flash 只要 4.66，几台 prefill 合起来一条专线放得下。</li>
    <li><b>还要挑请求</b>：论文只把超过长度阈值的 prompt 送去远端，短的就地算；光靠 KV 薄还不够。</li>
    <li><b>论文案例</b>：Moonshot 内部的 1T 混合模型，32 张 H200 做 prefill 加 64 张 H20 做本地 PD，比 96 张 H20 同构 PD 吞吐 +54%、P90 TTFT −64%（按同等成本算吞吐约 +15%）；跨集群平均出口只有 13 Gbps。</li>
  </ul>

  <h3 id="s8-9">8.9　落点</h3>
  <div class="note ok"><span class="t">回到主线</span>
    decode 每一步都要把权重和 KV 搬一趟。PD 分离不改变这一趟搬多少，它保证的是这一趟不被 prefill 打断，decode 才敢把 batch 攒满。
    代价是每个请求多一趟 KV 传输，所以要先确认它真跑在设计带宽上、输出和不拆时一字不差。</div>

  <details class="foldfig"><summary><b>考考自己</b>：PD 上线后 TTFT 比不拆时还高，TPOT 倒是稳了。先查哪一层？</summary>
  <p>先查搬运层：量一次 KV 传输带宽。如果只有几百 MB/s，多半是退回了 cuda_copy 或跨出了 NVLink 域；带宽正常再看调度层有没有排队。</p></details>
'''

FIGS = {
    "__FIG_S8_DECODE_AI_T5__": ("fig-s8-decode-ai-t5", "fig5-decode-ai.svg", "topic05-fig-pd.py",
        '<b>batch 越大，读一遍权重换来的计算越多。</b><br><em>专题五原图。</em>'),
    "__FIG_S8_PD_T5__": ("fig-s8-pd-t5", "fig5-pd.svg", "topic05-fig-pd.py",
        '<b>拆开以后，绿线一格不少。</b><br><em>专题五原图；时间线是示意。</em>'),
    "__FIG_S8_KV_TRIP_T5__": ("fig-s8-kv-trip-t5", "fig5-kv-trip.svg", "topic05-fig-pd.py",
        '<b>三色细条就是代价。</b><br><em>专题五原图；KV 用时按带宽估算，不是计时。</em>'),
    "__FIG_S8_STACK__": ("fig-s8-kv-stack", "fig6-kv-stack.svg", "topic06-fig-pd.py",
        '<b>一趟 KV 要过三层：搬运、调度、存储；每层 GPU 和 TPU 各有一套组件。</b><br><em>只列公开组件；横跨两层的按主要角色放。</em>'),
    "__FIG_S8_BW__": ("fig-s8-kv-bw", "fig6-kv-bw.svg", "topic06-fig-pd.py",
        '<b>红条比灰带还长。</b><br><em>带宽是 GB300 实测，时间是本课用 1.04 GB 统一换算的。</em>'),
    "__FIG_S8_HYBRID__": ("fig-s8-hybrid-kv", "fig6-hybrid-kv.svg", "topic06-fig-pd.py",
        '<b>注意力层补算只是把空格填上，GDN 的余额却把尾巴记了两遍。</b><br><em>格子和存折都是示意，层数取自 Qwen3.5-397B。</em>'),
    "__FIG_S8_PRFAAS__": ("fig-s8-prfaas", "fig6-prfaas.svg", "topic06-fig-pd.py",
        '<b>绿柱一台只占专线一成以内，橙柱一台就要三到六成。</b><br><em>每根柱是一台 prefill 每秒外送的量。绿色虚线是案例里实际的平均出口。</em>'),
}

NUM_LOCK = [
    ("5,201", "fig6-kv-bw.svg"), ("149", "fig6-kv-bw.svg"), ("6.2", "fig6-kv-bw.svg"),
    ("33.35", "fig6-prfaas.svg"), ("8.25", "fig6-prfaas.svg"), ("4.66", "fig6-prfaas.svg"),
]
