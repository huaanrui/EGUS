**FairSteer 风格参考预实验：运行前协议**

本轮针对上一轮对照目标、方向尺度、候选层限制和历史验证集复用的问题，先建立一项明确的参考基线。模型是本地 Qwen3-1.7B，属于 FairSteer 的本地适配；原论文在其他模型和更大检测器语料上的数值不作为本轮必达结果。

作者来源固定于 [FairSteer](https://github.com/LiYichen99/FairSteer/tree/0545cdbaa71b3301030accf1b98130820e08026f)，论文为 [Findings ACL 2025](https://aclanthology.org/2025.findings-acl.589.pdf)。核查发现作者的 `tokenized_bbqa` 同时扩增零样本和少样本的正确/刻板答案；`combine_activations_probes.py` 合并零样本 BBQ、少样本 BBQ 和 MMLU。当前实现保留这些结构。

|部分|本轮约定|
|---|---|
|同题双回答|只有正确答案不同于刻板答案的 fit 问题可生成一对；上下文、问题、选项顺序完全相同|
|方向|110 个问题，零/少样本各取答案末 token；正确状态减刻板状态；直接保存原始向量，α=1 表示加一个原始均值差|
|检测器|尚未回答的 prompt 末 token；实际正确回答标 1、实际刻板错误标 0、其他错误排除；MMLU 依作者标 1|
|检测器划分|fit 内部模板家族隔离，约 80/20；方向仅用检测器训练侧家族|
|检测器训练|sklearn L2 logistic regression，C=1，无特征标准化、无类别平衡权重；是 cuML 的 CPU 适配|
|提取和注入空间|所有 28 层的 raw block 输出；最终层也在最终归一化之前，避免混用 HF 最终归一化状态|
|干预|当前最后 prompt token；按每个 batch 元素单独判断 p(correct)<0.5；加 α·DSV|
|主要输入|作者 raw completion 文本，probe/direction 为 A.，评价为 (A)；本轮不混入旧版聊天模板|
|标签|从上下文验证空格加 A/B/C 恰好是一个续接 token；不复用旧聊天模板的裸字母 token ID|
|排列|所有推理固定六个答案排列组成一个等长 batch，无 padding/cache；原始排列分数单列；方向另做六排列平均诊断|
|主要目标|原始排列 BBQ 总准确率选层，α=1；同时保留以六排列宏平均刻板率及能力约束筛选的结果|

数据位于 [准备后的数据](D:/PythonProject/mechanism/data/prepared/bbq_fairsteer_20261012_v1)，配置为 [experiment_fairsteer.toml](D:/PythonProject/mechanism/configs/experiment_fairsteer.toml)。

- BBQ 检测器训练：1209 题 / 81 家族；内部验证：327 题 / 21 家族。方向子集 110 题 / 61 家族。
- selection：336 题 / 36 家族，涵盖六个类别。
- 新封存测试：320 题 / 20 家族，来自本轮新下载的 Nationality、Race_ethnicity、Religion。三个类别各自在 fit、selection、test 中按家族划分，因此先检验同类别新家族泛化。
- Age、Gender_identity、SES 保留旧家族的角色；历史 validation 不充当新测试。旧测试类别不纳入新确认。
- 作者固定少样本示例的对应家族从数据中排除；原有身份语义排除规则保留；准确上下文指纹跨集合重复会停止准备。
- MMLU 使用固定作者文件中的 512 个训练问题、128 个检测器内部验证问题、128 个 selection 能力问题、128 个封存能力问题，按问题去重并隔离。作者文件已经仅含 A/B/C 为真答案的题，输入保留前三选项。
- neutral 为既有人工合成任务诊断，不将其称为新通用能力基准；MMLU 也仅是小规模截取版评测。

搜索在首次看本轮模型结果之前由固定配置与代码确定：

1. 三种变体 `native_gated`、`native_always`、`balanced_gated` 各自在全部 28 层、α=1 上扫描，共 84 个配置。它们分别是原始答案顺序 DSV 加检测门控、不加门控、答案位置平衡 DSV 加原门控。共同 screen panel 每个 selection 家族/证据条件取一题。
2. 每个变体按原始排列准确率取前两层，在全部 336 条 selection 复核。主变体的 screen 最佳层另比较 α=0.1/0.5/1/1.5/2，共 10 个完整候选。预算适配明确不同于论文直接在 2200 题上评价每层，不能声称完整 selection 已穷尽所有层。
3. 主要参考配置从原始顺序、门控、α=1 的候选中按准确率选层。严格筛选另按六排列宏平均刻板率下降，并保留原约束；参考配置不自动成为严格可接受方案。
4. 测试前冻结主参考、另两变体的开发最佳、严格筛选候选（若不同），以及同层同门控、等原始向量长度的 3 个随机方向。unknown 输出校准从 0/0.25/0.5/1/2/4 在 selection 选定。
5. 封存计划写入配置、fit、selection 和 test 数据哈希；创建测试锁后禁止重新拟合、筛选或改参数。测试只是评价冻结方案，不据其结果重新选方法。

严格标准沿用上一轮：刻板率至少下降 1pp；明确题准确率最多下降 2pp、单类别最多下降 5pp；neutral/MMLU 准确率最多下降 2pp；neutral 明确题弃答增加最多 2pp；反刻板错误增加最多 0.5pp；模糊题宏平均准确率不下降。全部原始效果仍报告，不因“不通过”隐藏效果。

验证包含：同题答案和 token 对齐、逐样本门控、原始向量幅度、零强度/恢复的精确一致性、实际非零编辑、最后层 hidden_states 与 hook 空间核对、全开发输入长度与排列检查，以及固定 12 个开发问题的 BF16 batch6/batch1 稳定性检查。最长输入不截断；最大允许 2048 token。模型参数冻结，只训练轻量 CPU 检测器。

入口：[run_fairsteer.py](D:/PythonProject/mechanism/scripts/run_fairsteer.py)。各阶段可恢复：`fit → checks → screen → select → test → stability → report`。数据、配置、依赖源码和模型文件发生变化会拒绝复用；输出保存在新目录 [fairsteer_20261012_v1](D:/PythonProject/mechanism/results/fairsteer_20261012_v1)。旧实验结果继续作为历史探索证据。
