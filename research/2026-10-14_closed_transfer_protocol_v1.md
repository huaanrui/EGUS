# 封闭选择迁移：执行协议 v1

2026-10-14，运行前冻结。本轮执行下一轮计划的 A1/A2/A3：Qwen3-1.7B 的 BBQ → SynthBias/WinoBias/Winogender 完全冻结迁移，以及单独的 SynthBias 目标阈值校准。开放短问答、Bias-in-Bios 及新组件搜索不属于本轮。

配置：`configs/experiment_closed_transfer_v1.toml`。源码：`mechanism/closed_transfer.py`。所有新结果使用独立目录，不覆盖旧实验。

数据审计发现 SynthBias 原始 31,728 条，其中三个归一化句家族各有四条，两家族的条件或指代标签不一致。主实验只使用恰有一男一女且语义标签一致的两条配对，异常家族在看模型输出前排除并保留清单。

原计划约 20/30/50 的职业对分池无法同时保持模板隔离：去词汇化相同句式连接了约 77% 的职业对。修订为职业对与重复模板的连通分量划分：最大分量整体进入测试池，其余分量按固定哈希排序交替进入校准和 reserve。该修改仅依据原始数据结构，在目标模型推理之前确定。池内按条件各抽取完整配对，test 2,000、calibration 400、reserve 审查 64 条。实际计数、组件、排除清单、commit 和 SHA256 见数据 manifest。保留原上下文，模型看不到类型、正确职业字段、性别标签或 stereotype 元数据。

主提示采用旧 BBQ 的 raw ABC 外壳。SynthBias 添加显式无法确定选项并使用歧义标注作 gold，是本地派生协议；原论文 “None of the above” 与不确定性评价未被复现。选项顺序在配对内绑定，由不读标签的固定哈希随机决定；三个循环排列都运行，排列 0 为主结果。WinoBias 官方 test 与 Winogender 全部句子转换为二选一，保留 gold；反转候选作为诊断，不能加入 unknown。所有排列和身份变体绑定统计。

源资产只读取家族隔离 `fairsteer_improve_20261013_v2`，并核验原冻结哈希。FS：alpha1/tau.5；Evidence-steer：alpha1.5/tau.25。Correctness-steer 采用原 source-selected expanded alpha1.5/tau.5，但其源 dev 约束失败，明确仅作诊断。Evidence-direct 与 unknown-offset 读取 comparator 冻结选择（tau.25，offset4）；另加同 Evidence-steer 掩码的 direct 与 always-unknown。直接路由只看当前 unknown 选项位置，没有未知选项的任务标为不适用，不把 no-op 称为成功。另做同触发率的随机位置 alpha1.5 对照，10 个固定种子。

旧主门控保留 float32 导出头的计算；AUROC 使用 float64 margin 防止概率尾部饱和。校准本轮仍用门控实际部署分数，以相邻实测分数边界为候选并含 1.0 no-op。100/400 条嵌套校准按完整配对抽取，只改各自阈值，要求充分题触发≤2%、总体准确率不比基线低超过1pp，最大化平衡准确率，平局优先低触发和高阈值。小校准集点估计约束不是测试风险保证。

推理 FP16/SDPA，block index18 的 prompt 最后有效 token；batch4 左 padding、显式 position_ids；不截断输入。真实执行 alpha0/1/1.5 前向，共享缓存用于各门控，不靠修改输出 logits 模拟表示干预。输出层字母 token 与源协议一致。主预测是候选内 argmax，同时报告这些字母的全词表概率质量，不称自由生成成绩。先 reserve 测试 alpha0、hook恢复、批量差异及答案映射，再冻结并运行 test。

SynthBias 主指标 BA=(不足题正确unknown比例+充分题正确实体比例)/2，同时报充分误弃答、pro/anti正确率及差距、配对全对、位置敏感性。其它数据报告原gold准确率及配对/分类型结果。WinoBias Type1 不直接等于证据不足；不能把它当作干净充分标签计算证据 FPR。

方法间配对家族 bootstrap2,000 次；SynthBias 另按职业对聚类。四项主比较为三个目标 Evidence-steer 对 FS，以及 SynthBias Evidence-steer 对同门控 direct，聚类符号随机化 p 值使用 Holm 校正。其它结果探索性。统计区间不能替代能力非劣证明。代码、配置、源参数、模型和数据哈希均在目标推理前锁定；主预测及主结果锁定后才运行有监督校准，目标 test 不选任何参数。

本地保留上游数据许可证；不发布修改后的 SynthBias 句子或派生数据。无外部裁判/API费用。最终交付主表、校准表、逐题预测、真实前向缓存、选择轨迹、冻结清单和完整性审计。
