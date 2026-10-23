# 实现阶段的数据与方法审计

日期：2026-10-11。

## 数据版本更正

当前有效数据为 `data/prepared/bbq_pilot_v4`。v1 有已知家族泄漏，v2 错误地再次翻转了 nonneg 的 stereotype index，均不得用于研究结论。v3 已修正标签；v4 在 v3 基础上仅扩充 fit。原始 BBQ 数据没有修改。

官方 `additional_metadata.csv` 的 `target_loc` 已按问题极性调整，直接对应刻板一致的答案。官方评分把 `neg_Target` 与 `nonneg_Target` 均计入正向偏见。此前认为它总是同一个“受刻板印象群体”的人物索引，因而再次翻转 nonneg，这是错误解释。

依据：[固定版本官方评分代码](https://github.com/nyu-mll/BBQ/blob/bea11bd97d79217245b5871acd247b9d6eb24598/analysis_scripts/BBQ_calculate_bias_score.R)。已增加专门回归测试。

抽查的具体例子：

|题目|问句|官方 target_loc|刻板一致答案|
|---|---|---:|---|
|Age:1018|Who did not have any trouble hearing?|0|The 24-year-old|
|SES:10|What group of people avoids drugs?|0|wealthy people|
|Gender_identity:1007|Who provides counseling to DV survivors?|0|Paul|

另抽查 fit 中三类别×ambig/disambig×neg/nonneg 共12题的上下文、问题与 ground truth。小规模人工抽查不能排除全部源标注问题。

## Fit 扩充

初始880题中，修正标签后稳定刻板选择只有27题，覆盖15家族，未达到40题门槛。保留门槛，在原有47个 fit 家族中把每分层上限由4扩至8；fit 最终1664题。selection 312题与 validation 264题的 UID、文本及划分保持一致。此步骤仅使用 fit 行为，不依据 selection 或 validation 表现。

激活参考均值在扩充后的全部 fit 上重新计算，未把初始和扩充样本混为错误的等权平均。监督组件均使用相同的稳定行为发现集合，最多每类64题。

## 实现与原论文的关系

本轮使用 BF16 Qwen3-1.7B 原生非思考模板。方向、神经元和层均为面向 BBQ 的适配：没有宣称原安全任务论文的精确复现。神经元 hook 为 gated MLP 的 down projection 输入；连续层 hook 修改残差更新，而不是把整个残差流替换为均值。

RelP 保留模型前向，在反向冻结 RMSNorm 分母、SiLU sigmoid、attention Q/K 路径，并给乘法两条梯度各乘1/2。采用 `(a-μ_fit)*gradient` 节点分数，在具体 token 上计算再聚合。没有节点间边发现，也没有完整电路结论。[来源](https://arxiv.org/abs/2601.22594)

主任务是单 token 答案评分；自由生成、decode token 干预、CoT 和 RLHF 前后对比未实现。对于这些轴必须另立实验，不能从本轮结果外推。
