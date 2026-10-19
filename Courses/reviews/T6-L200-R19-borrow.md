# 专题六 L200 · R19 借讲法存档（节零–三）

两个研究 agent 分两组（零–一；二–三）联网读原文，并翻了本机书架（Scaling Book 中文版、扎哥公众号备份、MoE 环游记）。
只借讲法，不借数字；数字仍只取 L300。下面「读过」指用 read_url 读到了相关段落，「只见摘要」不作依据。

## 第零节：推理为什么贵、怎么变便宜

| 来源 | 讲法 | 采纳？ | 理由 |
|---|---|---|---|
| Horace He《Making Deep Learning Go Brrrr From First Principles》https://horace.io/brrr_intro.html（读过） | 开场讲钱：你花钱买了这些 TFLOPS，当然希望真用上；再用工厂（算力）／仓库（DRAM）／送货（带宽）讲三种瓶颈 | ✅ 开场 | 讲义第零节第一块改成「规格表最显眼的是算力，花了钱当然想用上，可吐字时绝大部分时间闲着、在等货」，不报数（0.32% 留第一节揭晓）。工厂／仓库与本课「显存＝仓库、片上存储＝工作台」同构，沿用本课说法，不另起比方 |
| a16z《Welcome to LLMflation》https://a16z.com/llmflation-llm-inference-cost（读过部分） | 拿摩尔定律类比「同等能力的成本每年降十倍」，列六条原因 | ❌ | 数字不在 L300；课件已注明市面降价主要归给更强的小模型（AI Index），这道题只算软件那一份 |
| Lilian Weng《Large Transformer Model Inference Optimization》（读过部分） | 开头列「为什么难」两条：显存大（KV 可达模型数倍）、难并行（自回归） | ❌ | 本课「货／笔记」两类已经覆盖；她这篇是技术综述，没有比方 |
| Databricks《LLM Inference Performance Engineering: Best Practices》（读过） | 共享服务把很多用户的负载合在一起压成本；量化少搬数据 | ➖ 印证 | 和本课「多坐人、少搬货」两条路一致，但原文没有明说「两条路」，这个框架是本课自己的 |

## 第一节：prefill 快、decode 慢；屋脊线

| 来源 | 讲法 | 采纳？ | 理由 |
|---|---|---|---|
| Horace He（同上） | 一张实测图：强度不到门槛前，运行时间几乎不涨，多出来的运算是免费的，直到送货卡车成了上限 | ✅ | 讲义 anim-trip 第 3 步后加「反过来读：313 行以内多喂一行，这一趟几乎不变长，多上来的人等于白搭车」。这句给第二节「多拉几个人」铺路，且是公交比方的自然延伸（车反正要开） |
| Scaling Book《All About Rooflines》https://jax-ml.github.io/scaling-book/roofline/（读过；本机中文版 roofline.md） | 开场「为什么一个算法要 50 ms 而不是 50 s 或 5 ms」；推出小 batch 时强度 ≈ batch | ➖ 已有 | 本课已有「喂几行，每字节就算几次」和猜五毫秒的互动。中文版 front-matter 里的「开车送货」类比查不到英文原文出处，像是译本加的，不当作原书讲法引用 |
| Scaling Book《All About Transformer Inference》（读过） | 「推理是披着一层皮的两件事」；小 batch 时步时 ≥（参数字节＋KV 字节）÷ 带宽 | ➖ 已有 | 本课第一节 fig-s1-wkv 就是这条式子的图 |
| kipply《Transformer Inference Arithmetic》（读过部分） | 「运算随 batch 和参数涨，内存读只随参数涨」 | ➖ 已有 | 本课「权重能摊、KV 摊不了」是同一句，且多了 KV 那一半 |
| Hugging Face《Continuous batching from first principles》（读过） | 开场：聊天机器人第一个字要等一会儿，之后一个一个蹦 | ➖ 已有 | 本课第一节开场就是这句 |
| 扎哥公众号「详细谈谈 DSpark 投机解码的原理」（本机） | batch 1 严重 memory-bound，投机把强度从 O(1) 提到 O(γ) | ❌ 本轮 | 属于第七节，留给 R20–R21 |

## 第二节：连续批、座位就是 KV、吞吐对交互性

| 来源 | 讲法 | 采纳？ | 理由 |
|---|---|---|---|
| Anyscale《How continuous batching enables 23x throughput in LLM inference while reducing p50 latency》https://www.anyscale.com/blog/continuous-batching-llm-inference（读过） | 开场：工程师以为只能靠量化、自写 CUDA kernel 这类内部改动，其实系统层的排队方式就能拿大头 | ✅ 开场 | 讲义第二节首块加「模型一行不改、kernel 一个不换，只改怎么排队，就多接了几十倍的人」，结尾按 L300 口径揭晓 Orca 36.9×。不用他们的 23× |
| Scaling Book《All About Transformer Inference》（读过） | 一张 batch／显存／步时／吞吐的表，把 KV 缩小后重跑：吞吐和延迟同时变好 | ✅ 结尾 | 讲义第二节交棒句加「笔记一薄是两头赚：坐得下更多人，每一步要翻的也少、每人更快；一路讲取舍，这是少有的不用取舍的一招」。只用机制，不用其数 |
| Baseten《Continuous vs dynamic batching for AI inference》（读过） | 公交三步：等坐满才开（静态）→ 坐满或到点就开（动态批）→ 公交线路随上随下（连续） | ➖ 印证 ／ ❌ 中间一步 | 本课包车对公交每站都停与它同构，说明比方选对了。「动态批」这一档 L300 没有，按判据 6 不进 L200；L200 也不需要多一个旋钮 |
| Hugging Face（同上） | 推导链：padding 浪费 → 去掉 batch 这一维、首尾相接 ＝ ragged batching ＋ 动态调度 | ➖ 已有 | 本课「所有人接成一长条一起乘、只有翻笔记分开」是同一个结论，口播更短 |

## 第三节：KV 多大、MLA／GQA、分页、前缀缓存

| 来源 | 讲法 | 采纳？ | 理由 |
|---|---|---|---|
| vLLM 博客《vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention》https://blog.vllm.ai/2023/06/20/vllm.html（读过） | 块＝页、token＝字节、序列＝进程；比方兑现第二次：进程能共享物理页，所以序列能共享 KV 块 | ✅ | 讲义「纸别浪费」末块改成「换来两样」：不按最长预留；顺带查页表找页，开头一样就指向同一页、不用各抄一份，「第三条路就靠这个做得起来」。两条路原来各讲各的，现在接上了。沿用活页本，不引入进程／内存页（全课没有操作系统比方） |
| LMSYS《Fast and Expressive LLM Inference with RadixAttention and SGLang》https://lmsys.org/blog/2024-01-17-sglang/（读过） | 换个框：请求结束不再丢 KV，留在基数树里当缓存，按 LRU 淘汰 | ✅ | 讲义「抄过的别再抄」加「笔记的身份也变了：以前用完当草稿纸扔，现在留着当缓存，挤不下了先挪最久没人翻的」。LRU 在 L300 4.3 的 KV 存储动画里已有 |
| 苏剑林「缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA」https://spaces.ac.cn/archives/10091（读过） | 主线只有一个问题：怎么在不掉效果的前提下压 KV；GQA 本来就是低秩投影 → 换成一般线性变换 → 恒等变换把矩阵吸收进 Q → RoPE 不兼容逼出单独的位置部分 | ❌ 本轮 | 主线和本课「写薄了会不会变笨」一致；低秩、吸收、RoPE 三步是 L300 第五节的深度，L200 只留「所有头对着同一条摘要算」一句。他的「能一张卡就别跨多卡、GQA 8 组对 8 卡机」可给第四节用，但 L300 没这句，暂不进 |
| Sebastian Raschka「Understanding and Coding the KV Cache in LLMs from Scratch」（读过摘要段） | 「Time flies fast」小例子：不缓存时前面的字每步重算，缓存后算一次用两次 | ❌ | 第零节已有「为什么要 KV」，而且第零节已略超计划；记在这里，淬炼轮如果要换讲法可用 |
| Welch Labs「How DeepSeek Rewrote the Transformer [MLA]」（只见摘要） | — | ❌ | 没看到原片，不作依据 |

## 没查到或没读的

Chip Huyen、Jay Alammar、3Blue1Brown、Karpathy、知乎：本轮没读到相关原文，不写他们怎么讲。
扎哥公众号对 KV 讲的是存储层级和分布式 KV 池（系统视角），不是入门讲法；MoE 环游记与本四节无关。

## 时长

讲义台词零／一／二／三节 963／1,289／1,170／1,247 → 1,023／1,342／1,249／1,332 字（+277 字，约 +1.4′）；全课含互动 92′（计划 90′）。
