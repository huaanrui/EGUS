# 第二章文献调研与论文定位

检索与核对日期：2026-10-17。范围为与当前 BBQ 论文主张直接相关的原始论文、正式论文集及作者项目材料；这不是穷尽性系统综述。本文中的原论文发现不等于我们已经复现的结果。

## 主要判断

当前最有价值的主张是：**在相同证据检测信号下，激活干预相对于直接选择 unknown 的增量作用有多大？** 将识别能力、响应动作和充分证据保护分开评价，可以更准确解释 BBQ 涨分。

“证据分类器可以帮助弃答”“有条件地进行 steering”“BBQ 高分不等于一般公平性”均有明确先例，不能单独作为创新点。我们的比较协议与具体实证发现可能构成贡献，但尚不能据此主张首次分离检测和控制。

## 一、评测与基准反思

| 文献与核实的发表信息 | 主要内容 | 与我们的关系 |
|---|---|---|
| [BBQ](https://aclanthology.org/2022.findings-acl.165/)，Findings ACL 2022 | 同时设置模糊与消歧上下文，考察缺少证据时的刻板印象及明确证据下的作答。 | 必引；不能说 BBQ 原设计没有考虑充分证据。 |
| [Rethinking Prompt-based Debiasing](https://aclanthology.org/2025.findings-acl.1361/)，Findings ACL 2025 | 提示去偏可能产生回避式回答，单看偏见指标容易误判。 | 核心先例；“多答 unknown 未必更公平”并非我们的首次发现。 |
| [One Example Is Enough to Pass Fairness Benchmarks](https://arxiv.org/abs/2609.14860)，2026，作者注明 EMNLP 2026 Main 接收 | 单例 ICL/GRPO 可显著改善 BBQ，分析指出缺失证据推理模式的重要性。 | 最接近我们的基准反思；需要一例提示对照和更具体的检测—动作分解。BibTeX 暂引用已核对的 arXiv 版本。 |
| [Is Your Model Fairly Certain?](https://proceedings.mlr.press/v267/wang25cp.html)，ICML 2025 | 提出 SynthBias 与考虑不确定性的公平评价，离散输出会隐藏置信度差异。 | 是外部验证的数据及评价背景；我们改写的三选一任务不等于完整复现其原生协议。 |
| [CEB](https://arxiv.org/abs/2407.02408)，ICLR 2025 | 跨任务、偏见类别的组合评测。 | 支持扩展评价维度；本地小模型裁判结果不能等同原论文评分流程。 |
| [Evaluating and Mitigating Social Bias ... in Open-ended Settings](https://arxiv.org/abs/2412.06134)，2024 arXiv | 包含开放形式的偏见测评与 OpenBBQ。 | 开放问答改写有先例，但无选项短答仍不等于任意自然生成。 |

## 二、表示控制、神经元与条件干预

| 文献与核实的发表信息 | 主要内容 | 与我们的关系 |
|---|---|---|
| [Representation Engineering](https://arxiv.org/abs/2310.01405)，2023 arXiv | 用内部表示进行监测与控制。 | 我们的隐藏态分类器也属于表示读取；标题应针对 activation steering，避免否定整个表示工程。 |
| [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)，ACL 2024 | 由对比激活估计方向并在推理中添加。 | 方向估计与干预基础；token 位置和施加时机是实质协议差别。 |
| [Refusal ... a Single Direction](https://arxiv.org/abs/2406.11717)，NeurIPS 2024 | 在研究模型中通过方向干预改变有害请求拒绝行为。 | 提供因果干预先例；有害请求拒绝不同于证据不足弃答。 |
| [Programming Refusal with Conditional Activation Steering](https://arxiv.org/abs/2409.05907)，ICLR 2025，CAST | 根据内部条件检测决定是否进行行为控制。 | 门控 steering 已有先例。 |
| [FairSteer](https://arxiv.org/abs/2504.14492)，2025 arXiv | 偏见检测器与动态方向干预组合。 | 核心方法基线；其检测标签不是我们的证据充分性标签，需区分整方法比较与固定门控比较。 |
| [Mitigating Biases ... via Bias Neurons Elimination](https://aclanthology.org/2024.acl-long.490/)，ACL 2024，CRISPR | 归因定位并抑制相关神经元。 | 神经元路线代表；迁移到我们的模型属于适配，不应标成作者设置的严格复现。 |
| [Knowing Bias, Doing Better](https://proceedings.mlr.press/v306/pan26j.html)，ICML 2026，KnowBias | 利用偏见知识相关神经元并增强其作用。 | 提醒不能把相关组件一律视为需要抑制；额外知识监督需要披露。 |
| [Bi-directional Bias Attribution](https://arxiv.org/abs/2602.04398)，ICLR 2026 | 利用双向归因指导内部偏见干预。 | 补充归因驱动路线；不是简单替换为一个残差方向。 |
| [Adaptive Triggering for Bias Correction in LLM Reasoning](https://arxiv.org/abs/2608.25379)，2026 arXiv | 根据推理中累积风险决定何时加入文本纠偏。 | 与门控时机有关；其在线推理纠偏不同于我们的 prompt 末端检测及答案替换。 |

## 三、充分性、弃答与探针

| 文献与核实的发表信息 | 主要内容 | 与我们的关系 |
|---|---|---|
| [Know What You Don't Know: Unanswerable Questions for SQuAD](https://aclanthology.org/P18-2124/)，ACL 2018 | 在阅读理解中加入不可回答问题。 | 可答性/不可答性不是新任务。 |
| [Selective Question Answering under Domain Shift](https://aclanthology.org/2020.acl-main.503/)，ACL 2020 | 训练校准器并研究分布变化下的选择性回答。 | 风险—覆盖率、阈值迁移的重要先例；其监督与我们源域固定阈值不同。 |
| [Sufficient Context](https://arxiv.org/abs/2411.06037)，ICLR 2025 | 区分上下文是否足够与模型是否答对，并利用充分性与置信信号进行选择性生成。 | 最接近我们检测目标的工作；不能声称首次结合证据与置信度。 |
| [The Internal State of an LLM Knows When It's Lying](https://aclanthology.org/2023.findings-emnlp.68/)，Findings EMNLP 2023 | 从内部状态学习真假判断。 | 支持内部可读信号路线，但真假不等于给定上下文的可答性。 |
| [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)，2022 arXiv | 研究模型对答案正确性和自身知识的预测。 | 置信、知识和上下文充分性需要分别定义。 |
| [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/)，EMNLP-IJCNLP 2019 | 控制任务帮助判断探针表现来自表示还是探针学习。 | 支持加入强控制；我们的表面/长度控制不等于复现其随机控制任务。 |
| [You Only Pass Once](https://arxiv.org/abs/2608.14465)，2026 arXiv，YOPO | 联合充分性读取与激活写入，研究两者干扰并用重构修正。 | 与检测—控制拆分高度相关；其任务中的联合收益也限制了“steering 无用”的泛化结论。 |

## 四、强简单基线与方法排序

| 文献与核实的发表信息 | 主要内容 | 与我们的关系 |
|---|---|---|
| [AxBench](https://proceedings.mlr.press/v267/wu25a.html)，ICML 2025 | 分开评价概念检测与控制，并发现简单基线很强。 | 最接近文章论证方式的先例；不是 BBQ 上的同一个实验。 |
| [SAEs Are Good for Steering – If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)，EMNLP 2025 | 区分输入响应与输出有效特征，改进特征选择可改变 steering 表现。 | 平衡引用；固定组件/强度的结果不能代表整个干预家族的性能上限。 |

补充阅读：[Anthropic 的 feature steering 评估](https://www.anthropic.com/research/evaluating-feature-steering)考察社会偏见控制及副作用。它是研究报告，可用于讨论强度与能力权衡；未纳入当前 24 条正式参考文献。

## 五、最接近工作的差异：论文究竟补什么

| 已有结论 | 我们可以补充的问题 | 不应写的主张 |
|---|---|---|
| FairSteer/CAST 已有检测后条件控制。 | 固定检测器时，隐藏态控制相对直接答案规则的增量是多少？ | 首次提出门控表示干预。 |
| Sufficient Context 已研究充分性及选择性生成。 | 在社会偏见选择题中，充分性识别能解释多少方法收益？ | 首次发现证据识别有助于弃答。 |
| YOPO 已结合表示读取与写入。 | 在 BBQ 的固定门控下比较不同动作，并量化充分证据损害。 | 首次区分读取与控制，或者内部干预永远无增益。 |
| Rethinking/One Example 已反思 BBQ 分数。 | 用可复核的条件比较将收益定位到检测与动作两部分。 | 首次发现 BBQ 高分不代表公平。 |
| AxBench 已支持强简单基线。 | 直接 unknown 是否应成为 BBQ steering 评价的必要基线？ | 简单方法一定比所有解释性方法好。 |

## 六、写作与实验口径

1. **明确两种任务目标。** 充分性分类预测上下文是否支持作答；偏见检测预测某种响应是否带偏见。不同标签预算和额外数据必须披露。
2. **区别指标。** BBQ unknown 是有真值的候选答案，选择它后仍进入整体准确率分母；传统 selective accuracy 通常只对保留回答计算，不能混用。
3. **保持证据边界。** 三模型、五次重采样是我们设定内的证据；尚不足以推翻原论文或整个方法家族。观察均值相近不等于统计等效。
4. **报告强表面对照。** 当前内部证据分类器相对表面基线的优势未通过既定多重比较校正，不应在正文省略。
5. **保留负向迁移。** 源域充分问题误弃答率约束不保证目标域；目标总体涨分可以掩盖充分证据准确率下降。
6. **公平给控制方法预算。** 固定强度与有限选层不构成该方法最优上限。补充实验需预先给出开发集搜索预算，不得看目标答案调参。
7. **优先新增的相关工作对照。** 一例缺失证据提示；固定门控下的配对比较；相近 unknown 频率的动作比较；含 unknown 干扰项的充分证据集。这里是建议，没有启动新的实验。

建议题目：**Does Debiasing on BBQ Require Activation Steering? Disentangling Evidence Detection and Intervention**。也可保留此前的 **Rethinking Activation Steering for Debiasing on BBQ: A Simple Evidence Classifier as a Strong Baseline**。

## 七、核对范围与版本注意事项

- 主体机制与局限核对到原文方法/实验的重点文献：FairSteer v2、Sufficient Context v3、YOPO v1、Adaptive Triggering v1，以及 BBQ 评测反思、AxBench/特征选择相关原始材料。其余背景论文用于摘要和正式元数据能支持的概述，不视为已完成代码复现。
- 作者顺序、标题、年份主要来自 ACL Anthology、PMLR、arXiv citation metadata；正式会议信息另按论文集或作者仓库核实。来源及元数据保存在 sources.json。
- Rethinking 的 ACL 元数据标题用单数 Model，而 PDF/预印本可见复数 Models。参考文献暂遵从正式导出元数据，两者是同一篇论文。
- One Example 的 arXiv 页面和项目声明 EMNLP 2026 Main 接收；当前未补填未核对的卷页信息。FairSteer、YOPO 等按本次核对到的 arXiv 版本引用，不凭名称猜会议。
- CAA 的正式 ACL 作者记录与早期预印本名字不同；SynthBias 的正式 PMLR 作者记录与后续 arXiv 作者名不同。当前引用遵从各自正式论文集。
- sources.json 的网页哈希用于标识抓取版本，不代表对全文、代码或科学结论进行了自动验证。本调研不改动任何既有实验数据、模型配置或统计结果。
