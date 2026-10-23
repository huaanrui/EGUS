# 封闭选择迁移：执行协议 v2

继承 [v1 执行协议](2026-10-14_closed_transfer_protocol_v1.md) 的全部科学设置、抽样和统计规则。

v1 在 reserve smoke 的候选 token 校验处停止，尚未运行目标测试或校准预测。Qwen3 的源协议字母 token 362/425/356 分别解码为带前导空格的 ` A`、` B`、` C`；v1 不必要地要求无空格完全匹配而失败。v2 只将校验改为 strip 后匹配，实际候选 token、提示、头、向量及推理计算不变。

新配置 `configs/experiment_closed_transfer_v2.toml`，有效输出 `results/closed_transfer_20261014_v2/`。v1 源快照与失败记录保留；新数据重新生成后逐文件核验与 v1 的输入/标签完全相同。正式测试开始前冻结 v2。不得基于目标结果修改规则。
