# -*- coding: utf-8 -*-
"""专题六 · 第十五节「第一天的八问」：把前十四节收成一张接手清单。"""

BODY = '''
  <p class="lead">接手一个推理服务，第一天先问八件事。<b>前四问算纸上的账，后四问拨开关。</b></p>
__FIG_S14_EIGHT__
  <h3 id="s15-1">15.1　前四问：算账</h3>
  <ul>
    <li><b>负载长什么样</b>：输入输出多长、要哪个交互档；换档位结论能反过来（第二节）。</li>
    <li><b>用哪把尺子</b>：总量还是每卡、分母含不含 prefill，换分母能差几十倍（第二节）。</li>
    <li><b>字节下限</b>：每步实测若是下限的好几倍，先查布局和调度，别急着加卡（第一、五节）。</li>
    <li><b>batch 上限</b>由 KV 定，也就定了一趟搬运最多分给几人（第四节）。</li>
  </ul>
  <h3 id="s15-2">15.2　后四问：挑招数</h3>
  <ul>
    <li><b>attention 怎么切</b>：先切法后参数（第六、十节）。</li>
    <li><b>拆不拆 PD</b>：有尖峰先试分块 prefill，压不住再拆；拆了按两边吞吐配台数（第九节）。</li>
    <li><b>量化和投机开不开</b>：投机在并发小时最赚（K3 并发 1 吞吐是不投机的 4.22 倍），量化总在减字节（第十一、十二节）。</li>
    <li><b>上线前跑通了吗</b>：没预编译，首个用户要等一分钟（第十四节）。</li>
  </ul>
  <p class="landing">⭐ 八问归到一句话：<b>decode 每一步都要搬一趟。</b>前四问算清这一趟有多重、能分给多少人；后四问要么让更多人分，要么让它变轻。</p>
  <h3 id="s15-cost">15.3　回到时间轴：系统这一侧省了多少</h3>
  <p>0.1 说的降价是市场价，混着更强的小模型、硬件换代和系统招数；报告主要归给小模型，硬件那份给的是能效每年升 40%，系统那份拆不出来。</p>
  <p>Epoch AI（2025）固定一档能力看，达到它的推理年年在便宜，2024 年以后降得更快，恰与第八到十二节那批招数进生产同期（本课观察，Epoch 没归因）。</p>
  <p>每行基线不同，不能连乘去凑市场上的降幅；只有末行是同一硬件、同一模型上叠满软件招数的总账。</p>
  <table>
    <tr><th>年份</th><th>补的哪个洞（0.1 的环）</th><th>首发</th><th>节</th><th>一个实测（对照谁）</th></tr>
    <tr><td>2022</td><td>① 整批等最长的</td><td>论文</td><td>三</td><td>Orca 连续批：同延迟吞吐是 FasterTransformer 静态批的 36.9 倍</td></tr>
    <tr><td>2022</td><td>② 字节太多：权重</td><td>论文</td><td>十二</td><td>GPTQ 3 bit：1 张 A100 跑 OPT-175B，逐字速度是 5 卡 FP16 的 3.25 倍</td></tr>
    <tr><td>2022</td><td>③ 一趟只出一个字</td><td>论文</td><td>十一</td><td>投机解码：单请求速度是不投机的 2–3 倍</td></tr>
    <tr><td>2023</td><td>⑤ KV 预留浪费</td><td>vLLM</td><td>四</td><td>PagedAttention：同延迟吞吐是 FasterTransformer 和复现版 Orca 的 2–4 倍</td></tr>
    <tr><td>2023</td><td>④ 每条 KV 太厚</td><td>论文</td><td>四</td><td>GQA-8：T5-XXL 每条样本推理约快 5.4 倍，平均分 47.1 对 47.2（对原版 MHA，本课推导）</td></tr>
    <tr><td>2023</td><td>② 字节太多：权重</td><td>论文</td><td>十二</td><td>AWQ 4 bit：Llama-2-7B 单卡 RTX 4090 再快 3.1 倍（对已融合 kernel 的 FP16）</td></tr>
    <tr><td>2023</td><td>⑧ 同样开头重算</td><td>SGLang</td><td>四</td><td>RadixAttention：吞吐最多是当时其他系统的 6.4 倍</td></tr>
    <tr><td>2024</td><td>⑦ 两段抢同一批卡</td><td>论文</td><td>八</td><td>DistServe：同样的延迟要求下，每卡能接的请求最多是不拆的 DeepSpeed-MII 的 7.4 倍</td></tr>
    <tr><td>2024</td><td>⑥ 长 prompt 插队</td><td>论文</td><td>三</td><td>Sarathi-Serve 分块 prefill：延迟达标时可接请求是 vLLM 的 2.6–3.7 倍</td></tr>
    <tr><td>2024</td><td>④ 每条 KV 太厚</td><td>DeepSeek</td><td>四</td><td>MLA：每层 KV 约是同形状 MHA 的 1/57（本课推导）</td></tr>
    <tr><td>2024</td><td>⑨ 前缀存不下</td><td>Kimi</td><td>八</td><td>Mooncake KV 池：Kimi 线上可接请求比此前系统多 107%–115%</td></tr>
    <tr><td>2024</td><td>② 字节太多：权重</td><td>vLLM</td><td>十二</td><td>vLLM FP8（权重和激活都 8 bit）：2 张 H100 跑 Llama 3 70B，每字间隔最多降到未量化的 1/2</td></tr>
    <tr><td>2024</td><td>⑨ 前缀存不下</td><td>DeepSeek</td><td>四</td><td>DeepSeek 磁盘 KV 缓存：128K prompt 首字 13 s → 0.5 s（对不用缓存）</td></tr>
    <tr><td>2024</td><td>③ 一趟只出一个字</td><td>DeepSeek</td><td>十一</td><td>DeepSeek-V3 的 MTP：速度约是不开的 1.8 倍</td></tr>
    <tr><td>2024</td><td>⑨ 前缀存不下：副本摊薄</td><td>SGLang</td><td>十四</td><td>SGLang v0.4 KV 感知路由：命中率 20% → 75%，吞吐是 v0.3 的 1.9 倍</td></tr>
    <tr><td>2024</td><td>⑩ KV 被复制、长 prompt 太慢</td><td>论文</td><td>七</td><td>Meta 跨机切序列：Llama3 405B 在 8 台 H100 上，跨机切 TP 的 prefill 延迟和它差约 100%（TP 慢）</td></tr>
    <tr><td>2025</td><td>调度占着 CPU</td><td>vLLM</td><td>三</td><td>vLLM V1 引擎重写：吞吐最多是 V0 的 1.7 倍</td></tr>
    <tr><td>2025</td><td>③ 一趟只出一个字</td><td>论文</td><td>十一</td><td>EAGLE-3：单请求最多是不投机的 6.5 倍，batch 64 吞吐仍是其 1.38 倍</td></tr>
    <tr><td>2025</td><td>⑪ MoE 拆散 batch</td><td>DeepSeek</td><td>六</td><td>SGLang 复现大 EP＋PD 分离：输出吞吐最多是同卡数纯 TP 的 5 倍</td></tr>
    <tr><td>2025</td><td>④ 每条 KV 太厚</td><td>DeepSeek</td><td>四</td><td>DeepSeek-V3.2 稀疏注意力（DSA）：每字只读 2,048 个历史 token（V3.1-Terminus 全读），基准基本持平</td></tr>
    <tr><td>2025</td><td>② 字节太多：KV</td><td>DeepSeek</td><td>十二</td><td>V3.2 存 FP8 KV：同显存请求容量约是 BF16 KV 的 1.63 倍（本课推导）</td></tr>
    <tr><td>2025</td><td>② 字节太多：权重</td><td>Kimi</td><td>十二</td><td>Kimi K2 Thinking 原生 INT4 权重：低延迟模式生成约快 2 倍（模型卡未写对照精度）</td></tr>
    <tr><td>2025</td><td>② 字节太多：KV</td><td>NVIDIA</td><td>十二</td><td>NVFP4（4 bit 格式）KV：大小约是 FP8 KV 的一半，前缀命中率最多高 20%（Qwen3-Coder-480B 上测）</td></tr>
    <tr><td>2026</td><td>⑨ 前缀存不下</td><td>llm-d</td><td>四</td><td>llm-d 卸 KV 到主机内存：P90（九成请求）首字 2.1 s → 0.5 s（对不卸载）</td></tr>
    <tr><td><b>2026-06</b></td><td><b>①–⑪ 叠起来</b></td><td><b>SGLang</b></td><td><b>二</b></td><td><b>GB300 跑 V4-Pro，硬件、模型、交互档不变，光改软件：每卡从首发日约 2,200 到约 11,200（后者开 MTP、首发日没开；PyTorch 博客）</b></td></tr>
  </table>
  <p class="landing">⭐ 开场那道题的答案就是末行：<b>5.1 倍</b>（约 11,200 ÷ 约 2,200，本课推导；博客标题写 5 倍）。卡、模型、交互档都没换，涨出来的每个字都是软件挣的。</p>
  <p>这 5.1 倍里 MTP 占多少，博客没拆。它只分开报了两条六月曲线各比首发日曲线涨多少：不开 MTP 的在单用户每秒 40 token 那档涨到 2.1 倍，开 MTP 的在 80 那档涨到 2.6 倍，档位不同，不能相减。</p>
  <p>别处有一组干净的拆法，换了模型和卡，只能当量级参考：TensorRT-LLM 在 8 张 B200 上跑 DeepSeek-R1，单用户每秒 111 token，开 1 层 MTP 到 198，开 3 层到 253，开 5 层反降到 239。</p>
__FIG_S14_STORY__
  <p>落到真实账上，看 DeepSeek 公开的一天：</p>
__FIG_S14_DAY__
  <p>这套服务用的正是表里的 MLA、大 EP 加 PD 分离、磁盘 KV 缓存和 FP8。</p>
  <p class="landing">⭐ 回到 2022 年那一批人：等最长的那个 → 每步换人，座位却不够 → 分页、压薄、复用 → 长 prompt 插队 → 切块，再拆成两批机器 → MoE 拆散 batch → 大 EP → prefill 喂不上，重配台数 → 输出变长 → 投机 → 再少搬字节。<b>同一套硬件光改软件，每卡已是首发日的 5.1 倍。</b>下一个洞已经露出来，就是第十三节那件事：线性层的状态截不开，前缀难复用；tpu-inference 上跑 Qwen3.5，前缀缓存至今还得关（13.2）。</p>
'''

FIGS = {
    "__FIG_S14_EIGHT__": ("fig-s14-eight", "fig6-eight.svg", "topic06-fig-eight.py",
        '<b>八问里有四问钉在车上（车 ＝ 每步从显存往工位搬的那一趟）：3、4、7、8。</b><br><em>颜色对主线：灰 ＝ 校准尺子／上线，蓝 ＝ 分给更多 token，绿 ＝ 少搬，橙 ＝ 拆开放对地方。</em>'),
    "__FIG_S14_STORY__": ("fig-s14-story", "fig6-story.svg", "topic06-fig-story.py",
        '<b>分类账：上表按年记流水，这张图把同一批招数按六类分户。</b><br>'
        '<em>图上比表多几张卡，如 Flash-Decoding、Splitwise、TPU megakernel。</em>'),
    "__FIG_S14_DAY__": ("fig-s14-day", "fig6-s14-day.svg", "topic06-fig-eight.py",
        '<b>输入占大头；紫色那段就是磁盘 KV 缓存替 prefill 省下的。</b>'),
}

NUM_LOCK = [
    ("36.9", "fig6-story.svg"),
    ("5.1", "fig6-story.svg"),
]
