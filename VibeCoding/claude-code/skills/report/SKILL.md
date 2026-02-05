---
name: report
description: 管理客户贡献报告。用于记录对客户的技术支持、市场活动、培训活动等贡献，并维护中英文双版本报告。
license: MIT
---

# 客户贡献报告管理

管理和更新客户贡献报告，记录对各客户的技术支持、商业影响、活动贡献和技术积累。

## 核心理念：控制与方向把控（Directional Control）

> **在 AI 基础设施领域，深度技术控制力（Deep Technical Control）才是最高级的销售策略。**

### 核心价值体现

- **技术资产积累**
  - 综合经验沉淀为技术方案、脚本、文档等可复用资产
  - 建立 [gpu-tpu-pedia](https://github.com/yangwhale/gpu-tpu-pedia)、[diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)、[gpu-recipes](https://github.com/yangwhale/gpu-recipes) 等核心代码仓库
  - 制定客户评估 TPU/GPU 性能的标准化计算公式和最佳实践

- **引领方向与纠偏**
  - 在客户沟通中主动领导技术路线和合作方向
  - 敢于纠正客户错误路径，规避巨大沉没成本
  - 建立"可信顾问（Trusted Advisor）"的权威地位

- **战略定制与 ROI 保障**
  - 结合 Google Cloud 优势，量身定制最优方案
  - 帮助客户建立合理目标，避免资源浪费
  - 保证投入可控、高效，最大化投资回报

## 报告文件位置

- 中文版：`~/my-private/reports/contribution-zh.md`
- 英文版：`~/my-private/reports/contribution-en.md`
- GitHub：https://github.com/yangwhale/my-private

## 报告人信息

- **姓名**：Chris Yang
- **职位**：AI Infra Customer Engineer (CE)

## 客户列表

- 腾讯 Tencent
- 蚂蚁集团 Ant Group
- 阿里巴巴 Alibaba
- Vivo
- Oppo
- MiniMax (SubSup/海螺视频)
- 快手 Kuaishou (Joyo/可灵)
- 小红书 Xiaohongshu (Red)
- Binance 币安

## 报告结构

报告采用 Markdown 格式，使用 **bullet points（项目符号）** 而非表格，主要章节如下：

### 1. 执行摘要

核心贡献的高度概括，每条贡献格式：
```markdown
- **[客户] [项目名]（[订单规模]）**：[关键技术突破]，[量化成果]，[商业影响]
```

示例：
```markdown
- **腾讯 B200 1584 卡订单（$1.5亿/3年 CUD）**：通过 DeepEP 极限优化，在 72 小时内将性能提升 2x，网络带宽从 27GB/s 瓶颈充分释放，直接促成签约
```

### 2. 核心理念：控制与方向把控

阐述技术控制力的战略价值。

### 3. 战略案例：技术纠偏与架构创新

深度技术案例，每个案例结构：
```markdown
### [案例名称]

[背景描述]

- **[问题/挑战]**
  - [具体点1]
  - [具体点2]

- **[行动/解决方案]**
  - [具体措施1]
  - [具体措施2]

- **[结果/成果]**
  - [量化结果1]
  - [量化结果2]
```

### 4. 客户详情

每个客户的详细记录，格式：
```markdown
### [客户名] [English Name]

**合作状态**: [活跃/Tech Win/Pipeline]
**累计订单**: [产品类型] [数量] chips ([状态]) + ...

#### [关键战役/项目名称]

[项目背景]

- **挑战**
  - [挑战点]

- **解决方案**
  - [方案点]

- **成果**
  - [成果点]

#### 主要贡献

- [贡献1]
- [贡献2]
- [贡献3]
```

### 5. 代码资产与技术基础设施

GitHub 仓库记录，格式：
```markdown
### [仓库名]：[一句话描述]

**GitHub**: https://github.com/yangwhale/[repo-name]
**仓库规模**: [大小] | [文件数] | [代码行数]

#### [功能分类1]
- **[项目名]**
  - [描述1]
  - [描述2]
```

**重要**：所有 GitHub 仓库名必须使用可点击链接格式：
```markdown
[gpu-tpu-pedia](https://github.com/yangwhale/gpu-tpu-pedia)
[diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)
[gpu-recipes](https://github.com/yangwhale/gpu-recipes)
[sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax)
```

### 6. 开源社区贡献 (SGLang)

格式：
```markdown
- **[项目名]**
  - 角色：[贡献者/协调人/核心贡献者]
  - [贡献描述]
  - [成果]
```

### 7. 活动贡献

会议演讲记录，格式：
```markdown
- **[会议名称]**
  - 日期：[YYYY-MM-DD]
  - 主题：[演讲主题]
  - 时长：[XX+Ymin]
```

### 8. 关键里程碑

时间线格式：
```markdown
- **[YYYY-MM-DD 或 YYYY-QX]**：[事件1]、[事件2]
```

### 9. 结论

总结核心论点和战略价值。

## 使用方式

### /report - 查看报告

显示当前报告内容摘要，包括：
- 执行摘要
- 各客户贡献概况
- 关键里程碑

**执行步骤**：
1. 读取 `~/my-private/reports/contribution-zh.md`
2. 提取并展示关键信息摘要

### /report update [内容] - 更新报告

当用户提供新的贡献信息时，更新报告。

**用户输入示例**：
- "今天帮腾讯解决了 DeepEP RDMA 带宽问题，性能提升 2x"
- "完成了蚂蚁 TPU v7 的 KDA kernel 集成"
- "在 AICon 做了 TPU 推理优化的演讲"

**执行步骤**：
1. 读取当前中文报告
2. 解析用户提供的信息：
   - 识别客户名称
   - 识别贡献类型
   - 提取日期（默认今天）
   - 提取事项描述
   - 提取量化成果
3. 更新中文报告的对应部分：
   - 客户详情 > [客户名] > 主要贡献
   - 如果是重大成果，同步更新执行摘要
   - 如果是新里程碑，更新关键里程碑
4. 显示更新内容供用户确认
5. 提交并推送到 GitHub

**贡献类型分类**：

| 类型 | 报告位置 |
|------|----------|
| 客户技术支持 | 客户详情 > [客户名] > 主要贡献 |
| 重大订单促成 | 执行摘要 + 客户详情 > [客户名] > 关键战役 |
| 技术方案/架构 | 战略案例 或 客户详情 > [客户名] |
| 代码仓库贡献 | 代码资产与技术基础设施 > [仓库名] |
| 开源社区贡献 | 开源社区贡献 (SGLang) |
| 会议演讲 | 活动贡献 > 会议演讲 |
| 里程碑事件 | 关键里程碑 |

### /report sync-en - 同步英文版

将中文报告翻译同步到英文版。

**执行步骤**：
1. 读取中文报告
2. 翻译为英文
3. 写入英文报告
4. 提交推送

### /report summary - 生成摘要

生成一份简洁的影响力摘要，适合在汇报时使用。

**执行步骤**：
1. 读取报告
2. 统计关键数据
3. 生成精炼的摘要，包括：
   - 关键订单促成（按金额排序）
   - 技术突破亮点
   - 代码资产规模
   - 活动数量

## 更新规则

### 格式规范

- **使用 bullet points，不使用表格**
- **粗体标记**：客户名、项目名、关键数据
- **GitHub 链接**：所有仓库名使用 `[name](url)` 格式
- **金额格式**：使用 `$X.X亿` 或 `$XXX万` 格式
- **芯片数量**：使用 `XXX chips` 格式

### 日期格式

- 精确日期：`YYYY-MM-DD`
- 季度：`YYYY-QX`
- 月份：`YYYY-MM`

### 客户状态

- **活跃**：有进行中的项目或订单
- **Tech Win**：技术验证成功但未商业落地
- **Pipeline**：潜在机会

### 主要贡献格式

每条贡献应简洁明了，格式：
```markdown
- [动作/成果]：[具体描述]，[量化指标]
```

示例：
```markdown
- v7 POC：DeepSeek V3 671B 训练达 2424 tokens/chip/s，性能超 B200 110%+，订单扩展至 2000+ chips
- B200 DeepEP：72 小时极限优化实现 2x 性能提升，促成 $1.5 亿订单签署
```

### Git 提交

每次更新后自动提交：
```bash
cd ~/my-private && git add reports/contribution-zh.md && git commit -m "[简要描述]" && git push
```

提交信息示例：
- "添加腾讯 v7 POC 成果"
- "更新阿里巴巴 B200 ramp 进展"
- "添加关键里程碑、GitHub链接"

## 示例对话

### 示例 1：添加客户技术贡献

用户：今天帮腾讯完成了 HunyuanVideo-1.5 的 Custom Splash Attention 优化，性能提升了 20.3%

执行：
1. 在「腾讯」客户的「主要贡献」部分添加新行：
   ```markdown
   - HunyuanVideo-1.5：Custom Splash Attention 优化 +20.3%
   ```
2. 提交推送

### 示例 2：添加重大订单

用户：腾讯 B200 1584 卡订单签约了，$1.5 亿三年 CUD

执行：
1. 更新「执行摘要」添加新条目
2. 在「腾讯」客户详情中添加「关键战役」章节
3. 更新「关键里程碑」
4. 提交推送

### 示例 3：添加代码仓库贡献

用户：给 diffusers-tpu 添加了 Wan2.1 模型支持

执行：
1. 在「代码资产与技术基础设施」>「diffusers-tpu」部分添加：
   ```markdown
   - **Wan 2.1/2.2**
     - 通义万相视频生成
     - 支持 I2V 多模态输入
   ```
2. 提交推送

### 示例 4：添加会议演讲

用户：下个月要在 Google Cloud Summit 2026 做一个关于蚂蚁 TPU 推理的演讲

执行：
1. 在「活动贡献」>「会议演讲」部分添加：
   ```markdown
   - **Google Cloud Summit 2026**
     - 主题：Ant Ling/Ring 1T MoE 利用 TPU 解锁 LLM 推理性能极限
     - 时长：30min
   ```
2. 提交推送

### 示例 5：添加关键里程碑

用户：腾讯 v7 POC 成功了，128 chips 跑通

执行：
1. 在「关键里程碑」添加：
   ```markdown
   - **2026-02**：腾讯 v7 128 chips POC 成功，性能超 B200 110%+，订单扩展至 2000+ chips
   ```
2. 提交推送

## 重要提示

- **默认仅更新中文版**，英文版需要用户明确指令 `/report sync-en`
- **保持 bullet point 格式**，不要使用表格
- **GitHub 仓库名使用可点击链接**
- **量化数据要准确**：性能提升百分比、芯片数量、订单金额
- **订单金额使用美元 $**
- **每次更新后提交到 GitHub**
