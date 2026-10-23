# 已完成的 BBQ 机制预实验

实验启动：2026-10-11；完成：2026-10-12。

完整 selection 确认通过的改善候选：relp_nodes。 relp_nodes 的验证模糊题准确率变化为 +0.25 个百分点，刻板选择率变化为 +0.51 个百分点。刻板选择并未下降，因此不能将该结果解释为已经成功去偏。

完整报告：[report.md](D:/PythonProject/mechanism/results/pilot_20261011_v4/report.md)。

|条件|selection状态|模糊题准确率|刻板率|充分证据准确率|
|---|---|---:|---:|---:|
|baseline|基线|72.73%|16.92%|67.68%|
|dim|no_confirmed_improving_candidate|72.47%|17.17%|67.93%|
|cav|no_confirmed_improving_candidate|72.98%|17.17%|68.18%|
|sn_consistency|no_confirmed_improving_candidate|73.23%|16.92%|67.93%|
|neuron_contrast|no_confirmed_improving_candidate|72.47%|17.17%|67.68%|
|relp_nodes|accepted|72.98%|17.42%|68.69%|
|activation_magnitude|preregistered_control|72.98%|16.67%|68.43%|
|random_neurons|preregistered_control|72.73%|16.92%|68.18%|
|random_direction|preregistered_control|72.73%|17.17%|67.68%|
|layer_span|no_feasible_screen_candidate|—|—|—|

单模型、15个验证家族，置信区间较宽；单样本与三轮批次初筛基线有5/168预测差异。主比较统一 batch=3；本轮不支持全局最优、一般能力保持或完整电路结论。具体局部消融、跨层节点诊断及置信区间均保存在完整报告和 JSON/CSV 产物中。
