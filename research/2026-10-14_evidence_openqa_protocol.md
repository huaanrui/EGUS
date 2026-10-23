# BBQ ↔ SQuAD 证据头与开放问答迁移：第一轮协议

模型 Qwen3-1.7B，FP16、SDPA；block index 18、最后 prompt token。旧实验保持冻结。新输出 results/evidence_openqa_20261014_v1。

SQuAD 2.0 官方 train 按文章哈希分为 fit/dev 池（80%/20%），分别固定取1000/200个段落；官方 dev 固定取300段作为本地 test（不称官方 hidden test）。每段选择一个可回答和一个不可回答问题，优先匹配问题 token 长度；上下文逐字相同。先按文章分组，检查原文段落、问题不跨分区并过滤与其他分区相同文本。记录筛选及输入长度，不截断，超长例子在模型运行前排除。按 source article 聚类报告区间。

BBQ 使用旧家族隔离 dev352/test704题，只有 zero-shot 无选项版本。旧历史测试已观察，所以属于探索性迁移。含明显“above/following choices/options”等引用选项的题按输入规则排除并记录；一般 Who 问题保留。答案选项和标签仅供自动评分，不进入任何开放 prompt 或模型路由。

提示版本：raw无选项 Context/Question/Answer；chat无选项，明确仅依据上下文、信息不足说 Insufficient information、短答案不解释、enable_thinking=False。旧头分别测这两种格式；新 SQuAD 头只在 chat 特征拟合和评价。旧 ABC 激活只作为原格式参考，不当作新提示激活复用。

方法：旧 BBQ evidence 头与固定 .25 阈值；SQuAD block18 逻辑回归 C1/L2/LBFGS；表面数字特征 StandardScaler+逻辑回归；context+question、context-only、question-only TF-IDF+逻辑回归。所有新拟合仅用 SQuAD fit，两类标签为不足1/充分0，不加入 MMLU。新头阈值仅用 SQuAD dev，在预设网格中选择充分样本假阳性≤2%时不足召回最高者，同召回优先更高阈值。1.0为合法不触发兜底；无 BBQ 选层、选阈值或重新拟合。预设.5分类结果和AUROC同时报告，区分排序/阈值迁移。

特征批量4、左 padding、显式position_ids；hooks仅捕获目标层末token，减少内存。fit/dev smoke 检查单条/批量数值近似及hook位置；不以 test 效果调整实现。chat生成 batch4、greedy、max_new_tokens96；记录EOS与截断。所有方法同一基础指令。G0为实际生成；G2-old与G2-SQuAD触发时使用固定信息不足文本，否则逐字复用G0输出。固定弃答是决策策略，不声称生成机制已改善。生成主测试仅 SQuAD test 和 BBQ test，识别阶段另外包括双方dev和 SQuAD fit。本轮不训练G3，不展开方向/神经元组合。

识别报告AUROC、balanced accuracy、类召回、Brier、触发率、按类别及配对分数差。生成 SQuAD 报告标准化EM/F1（最大参考答案分数）、充分样本F1、错误弃答、不足样本恰当弃答；模型记忆命中但上下文不支持的答案仍不可当作有据回答。BBQ采用保守规则：短答案与已知实体/答案文本匹配、明确弃答、混合猜测、无法解析分别记录；含人物猜测的保留性回答不因出现弃答词而记成功。规则结果属于自动诊断，待人工审查；报告不可解析比例与全量匿名审查文件，不使用不可靠本地裁判替代标注。精确/保守匹配可能低估语义正确，不能把解析错误解释成模型推理失败。

相对G0收益按 SQuAD article / BBQ family 聚类bootstrap1000；配对问题绑定。主要比较G2-SQuAD vs G0，旧头为参考；多数据/指标探索结果不称确认。dev2%限制只用于源域选参，不保证目标域能力保持，实际目标域误伤必须报告。

源码、配置、数据清单、原头与模型元数据/权重哈希在首次拟合前冻结；缓存均验证输入及文件哈希。代码故障若需要修复，保留原快照，在新版本输出中运行，不覆盖已拟合版本。报告渲染独立于计算核心。

来源：https://rajpurkar.github.io/SQuAD-explorer/ 。本轮不使用未审计的 SufficiencyBench。
