# 稳定分数校准补充：运行前规定

2026-10-14，在封闭迁移主结果生成前规定。主 v2 保留旧 float32 概率门控和校准以复用原决策；本补充落实原计划中的 float64 margin/稳定组合分数校准，用于避免 sigmoid 尾部饱和导致不同样本并列。

只用同一目标 calibration 的嵌套100/400条，使用相同充分触发≤2%、准确率损失≤1pp约束；固定头、方向、层、强度和测试样本。FS 用负正确性 margin，expanded 用负 expanded margin，direct 用证据 margin；组合用 float64 sigmoid(-expanded margin) × sigmoid(evidence margin)。候选为实测分数边界和 no-op；只是数值读出/阈值适配，仍为有监督目标校准，不能称源冻结迁移。

`scripts/closed_transfer_stable_calibration.py declare` 在主结果出现之前锁定协议和源码。`run` 必须在主预测 lock 之后执行，保存全候选、所选阈值、逐题预测和精度并列诊断。不得从旧/新数值版本中按测试成绩选择最佳版本。
