# 稳定校准 v2：阈值单位的实现修正

本修正在主结果及稳定校准 v1 已观察后记录，属于探索性实现修正，不能称独立确认。主迁移结果和旧产物均保留。

v1 复用旧概率校准函数，额外包含常数阈值1.0。对于概率，它是固定 no-op；对于未界定的 margin，它可能触发，且不是实测边界。这不符合稳定补充中“实测分数边界加 no-op”的意图。

v2 仅修正候选：实际 calibration 分数边界，加 float64 最大有限数作 guaranteed no-op。不引入测试候选，不改变监督预算、选择约束、头、方向、层、强度或样本。完整 v1/v2 都保留，不按测试成绩挑版本。

代码 `scripts/closed_transfer_stable_calibration_v2.py`，结果 `results/closed_transfer_20261014_v2/stable_calibration_v2/`。报告该代码修正后的结果，并披露其发生在观察 v1 之后。
