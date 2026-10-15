# -*- coding: utf-8 -*-
"""专题六 · 第十三节「落地：TPU 服务化与编排」。"""

BODY = '''
  <p class="lead">kernel 调好了，服务还可能第一分钟就卡住：<b>卡在形状、编译和冷启动上。</b></p>

  <h3 id="s13-1">13.1　静态形状：服务侧怎么选桶</h3>
  <p>专题二<a href="topic-02-L300.html#s5">「谁做决定：运行时，还是编译期」</a>讲过：XLA 按形状编译，变长的输入只能分桶加补零。现在的做法是把一批里所有请求的 token 拼成一条，只把总 token 数补到预编译过的几档；单条请求长短不齐交给 RPA（<a href="#s5-2">§5.2</a>）。</p>
__FIG_S13_BUCKET__
  <p>切法比桶数要紧：按 √2 倍一档的 13 个桶，补零反而比每 512 一档的 16 个桶少。</p>
  <p>没编过的桶，第一个撞上的请求就得等编译：Qwen3.5-397B 在 TPU 上要等 60–90 s，之后同形状不到 1 s。</p>
  <p>所以桶要写进启动参数预编译。SGLang-JAX 的写法是 <code>--precompile-bs-paddings 1 4 8 16 32 64 128 256</code> 加 <code>--precompile-token-paddings 4096</code>：prefill 按 token 档各编一份，decode 按 batch 档各编一份，两组相加就是开服前要编的份数。</p>

  <h3 id="s13-2">13.2　编译缓存为什么会失效</h3>
  <p>编好的程序可以落盘复用（设 <code>JAX_COMPILATION_CACHE_DIR</code>），但常常命不中。</p>
  <ul>
    <li><b>键里装得太多。</b>缓存键里有程序、库版本、编译参数和切片形状，升一次镜像、改一个参数都要全部重编。</li>
    <li><b>放错了地方。</b>有人把缓存目录设在 <code>/dev/shm</code>，那是内存盘，机器一重启就没了。多机时只有 0 号进程写缓存，目录得放在共享存储上。</li>
    <li><b>有些缓存根本不落盘。</b>R1 把权重搬上 device 的传输程序只缓存在进程内存里：同一进程重载 1 分 48 秒，换个新进程就退回 3 分 17 秒到 3 分 51 秒。</li>
  </ul>
__FIG_S13_COLD__
  <p>R1 上也有图里 GLM-5.1 那个问题：名义上在加载权重，实际在反复重编译。加载代码每搬完一个 tensor 就清一次编译缓存（<code>jax.clear_caches()</code>），1,367 个 tensor 只有 25 种形状，去掉清缓存那一行，这段从 215.7 s 降到 21.4 s。</p>

  <h3 id="s13-3">13.3　权重加载：别让几十个引擎抢一根管子</h3>
  <p>Kimi K2.6 把 MoE 权重缓存放在本机内存盘，缓存还在时冷启动 6 分 8 秒；机器一重启缓存就丢，又是 57 分钟。放在共享文件系统上，每层读取耗时是内存盘的 38 倍。</p>
  <p>扩容时几十个引擎一起起，平分共享存储的总带宽（本课推导）。GPU 这边 V4-Pro 的做法是放本地：806 GB 权重放本地 SSD RAID，14 GB/s，估算加载不到 1 分钟。</p>
  <div class="note"><span class="t">一条经验</span>搬字节那段用本地缓存省，编译那段用持久化缓存省；都省不掉，就提前起好备用副本。</div>

  <h3 id="s13-4">13.4　运维的两个坑</h3>
  <ul>
    <li><b>「起来了」不等于「活着」。</b>V4-Pro 的 SGLang 部署里 14 台 prefill 全崩了，服务注册表还显示 14 台在线。判活要看显存、真发一个请求。</li>
    <li><b>别一卡一个 <code>pkill -9</code>。</b>GB300 上强杀一个满载的 decode，每卡漏掉约 97 GB 显存，SGLang 按 <code>mem-fraction-static</code>（预留给权重和 KV 池的显存比例）0.85 做的静态预留（每卡 284 GB 里要占 241）就再也够不着，只能整个删 pod（K8s 里装这组进程的容器）重建。</li>
  </ul>
__FIG_S13_LEAK__

  <h3 id="s13-5">13.5　栈怎么选</h3>
  <table>
    <tr><th>栈</th><th>现状（截至 2026-10）</th></tr>
    <tr><td>JetStream</td><td>已停止维护，宣布 2026-02-01 起归档，截至 10 月仓库仍未锁成只读（最后提交 2026-01-05）</td></tr>
    <tr><td>tpu-inference</td><td>vLLM 的 TPU 后端，JAX 和 PyTorch（Torchax）两条模型路径；支持矩阵里前缀缓存、走 DCN 的 PD 分离、KV 下放、Eagle3 投机均标通过</td></tr>
    <tr><td>SGLang-JAX</td><td>SGLang 的 JAX 实现，原生跑在 TPU 上；基数树前缀共享；支持 Qwen、Llama、DeepSeek V2/V3、GLM-4 MoE、MiMo 等</td></tr>
  </table>
  <p>选哪个先看模型在不在支持列表里，再看第十节的拓扑能不能搭出来。</p>

  <h3 id="s13-6">13.6　编排：把「少算」落到集群层</h3>
  <p>前缀缓存默认只在本副本里有效（接了 3.6 的远端存储除外）。<b>路由不知道哪个副本存着哪段前缀，命中率就随副本数摊薄。</b></p>
__FIG_S13_ROUTE__
  <p>KV 感知路由把同前缀的请求送到已经存着它的副本。SGLang 2024 年 12 月把它放进开源路由，命中率 20% → 75%、吞吐是 v0.3 的 1.9 倍；之后 NVIDIA Dynamo（2025-03）、llm-d（2025-05）、GKE Inference Gateway 都做了，图右是 Red Hat 2025-10 用 llm-d 的实测：命中缓存首字 340 ms，对照是<b>不路由的最坏情况</b> 2,850 ms；原文没给平均或中位数对照，所以不能读成「路由快八倍」。</p>
  <p>容量规划沿用 goodput 的思路（<a href="#s2-2">§2.2</a>）：先定交互档，再看这一档每卡扛多少人（2.2 图里那一档约 224 人），decode 卡数 ＝ 同时在生成的请求数 ÷ 每卡人数；分母的坑见 2.3。</p>

  <p class="landing">⭐ 路由把「少算」搬到集群层：<b>前缀只算一次</b>，后面的请求都来复用。</p>
'''

FIGS = {
    "__FIG_S13_BUCKET__": ("fig-s13-bucket", "fig6-s13-bucket.svg", "topic06-fig-serve.py",
        '<b>比绿条和第一根橙条：桶更少，浪费也更少。</b><br><em>本课推导示意：每步整批 token 总数取对数正态，非真实流量；浪费 ＝ 补零 token ÷ 实算 token。</em>'),
    "__FIG_S13_COLD__": ("fig-s13-cold", "fig6-s13-cold.svg", "topic06-fig-serve.py",
        '<b>看三次里最长那段：GLM 和 Qwen3-Coder 是编译，MiMo 是权重加载。</b><br><em>上为 GLM-5.1 一次冷启动放大，下为另外两次的构成；各段时长取自原记录的日志拆账。</em>'),
    "__FIG_S13_ROUTE__": ("fig-s13-route", "fig6-s13-route.svg", "topic06-fig-serve.py",
        '<b>看同前缀的请求落到哪个副本。</b><br><em>左为示意；右为 Red Hat 用 llm-d 的实测，TTFT 含排队，2,850 ms 是不路由的最坏情况、不是轮询的平均值。</em>'),
    "__FIG_S13_LEAK__": ("fig-s13-leak", "fig6-mem-leak.svg", "topic06-fig-serve.py",
        '<b>比上下两行：同一条虚线，上面落进红框，下面落在绿条里。</b><br><em>runbook 原值，每卡，单位 GB；差 54 为本课推导。删 pod 重建后 56 秒显存归零。</em>'),
}

NUM_LOCK = [
    ("87.3%", "fig6-s13-bucket.svg"), ("10.8%", "fig6-s13-bucket.svg"),
    ("2,850", "fig6-s13-route.svg"), ("340", "fig6-s13-route.svg"), ("87.4%", "fig6-s13-route.svg"),
]
