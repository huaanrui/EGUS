# 各模型任务内最高分组件组合：探索性搜索

主目标为开发集单题准确率，并列时比较成对正确率；模型和组件参数在目标评测前冻结。
全部层MLP和注意力头先做rank归因，实际筛选包含坐标子集、全组件、强度细化、至多三个位置的beam组合。
最高分指本轮有限搜索空间内的开发集赢家，不保证全局最优；另报五项能力任务下降均不超过2pp的赢家。
沿用rank64、screen32、dev49、历史Type2 146对。各阶段均男女×AB/BA；历史目标已观察，不是新封存确认。
原Type2金答案高度集中在第二职业，本轮分数优化不能单独当作通用去偏或身份专属机制证据。层号和头号从0开始。

|模型/方案|开发准确率|开发成对均对|能力约束|历史Type2准确率|历史Type2成对均对|
|---|---:|---:|---|---:|---:|
|qwen3/baseline|81.12%|70.41%|通过|76.54%|65.41%|
|qwen3/best_task|95.41%|91.84%|未通过|90.41%|84.93%|
|qwen3/best_constrained|87.76%|80.61%|通过|82.02%|72.95%|
|qwen3/ref_selected|83.16%|71.43%|通过|77.23%|66.44%|
|qwen3/ref_fine_combo|83.16%|71.43%|通过|77.23%|66.44%|
|qwen3/ref_norm_whole|81.63%|70.41%|通过|77.23%|66.10%|
|qwen3/ref_old_combo|87.76%|80.61%|通过|82.02%|72.95%|
|qwen3/ref_old_norm_whole|86.73%|76.53%|通过|81.68%|72.26%|

qwen3 搜索量：{'short': 580, 'screen': 297, 'dev': 26}。
最高分方案：L21 H1/答案位置/whole(128维)/center λ=-1.5；L6 MLP/上下文代词/accuracy_64_neg(64维)/center λ=1；L6 MLP/上下文代词/accuracy_64_pos(64维)/center λ=-1；L8 MLP/问题代词/accuracy_128_neg(128维)/center λ=1.5；L8 MLP/问题代词/accuracy_128_pos(128维)/center λ=-1.5。
满足能力约束方案：L9 MLP/上下文代词/gap_pos(64维)/center λ=0.5；L21 H1/答案位置/gap_pos(16维)/center λ=-0.5；L21 H1/答案位置/gap_neg(16维)/center λ=0.5。

qwen3/best_task 相对 baseline：准确率+13.87pp，95%家族区间[+10.62,+17.22]pp。

qwen3/best_task 相对 ref_selected：准确率+13.18pp，95%家族区间[+9.93,+16.61]pp。

qwen3/best_task 相对 ref_old_combo：准确率+8.39pp，95%家族区间[+5.66,+11.24]pp。

qwen3/best_constrained 相对 baseline：准确率+5.48pp，95%家族区间[+3.72,+7.45]pp。

qwen3/best_constrained 相对 ref_selected：准确率+4.79pp，95%家族区间[+3.21,+6.55]pp。

qwen3/best_constrained 相对 ref_old_combo：准确率+0.00pp，95%家族区间[+0.00,+0.00]pp。


|开发能力任务|原模型|最高分方案|受约束方案|
|---|---:|---:|---:|
|reading|86.73%|100.00%|97.96%|
|literal|100.00%|100.00%|100.00%|
|binding|98.47%|92.35%|98.47%|
|role_swap|79.08%|92.35%|83.16%|
|record|73.72%|72.70%|72.96%|

|模型/方案|开发准确率|开发成对均对|能力约束|历史Type2准确率|历史Type2成对均对|
|---|---:|---:|---|---:|---:|
|minicpm5/baseline|72.96%|61.22%|通过|72.09%|60.27%|
|minicpm5/best_task|92.86%|87.76%|通过|96.06%|92.81%|
|minicpm5/best_constrained|92.86%|87.76%|通过|96.06%|92.81%|
|minicpm5/ref_selected|76.02%|64.29%|通过|73.80%|61.64%|
|minicpm5/ref_fine_combo|74.49%|64.29%|通过|73.12%|61.30%|
|minicpm5/ref_norm_whole|73.98%|62.24%|通过|72.26%|60.62%|

minicpm5 搜索量：{'short': 515, 'screen': 242, 'dev': 21}。
最高分方案：L17 H4/问题代词/whole(128维)/raw λ=1；L12 MLP/上下文代词/accuracy_128_neg(128维)/center λ=1.5；L12 MLP/上下文代词/accuracy_128_pos(128维)/center λ=-1.5；L41 MLP/答案位置/whole(6144维)/raw λ=1.5。
满足能力约束方案：L17 H4/问题代词/whole(128维)/raw λ=1；L12 MLP/上下文代词/accuracy_128_neg(128维)/center λ=1.5；L12 MLP/上下文代词/accuracy_128_pos(128维)/center λ=-1.5；L41 MLP/答案位置/whole(6144维)/raw λ=1.5。

minicpm5/best_task 相对 baseline：准确率+23.97pp，95%家族区间[+20.57,+27.18]pp。

minicpm5/best_task 相对 ref_selected：准确率+22.26pp，95%家族区间[+18.92,+25.34]pp。

minicpm5/best_constrained 相对 baseline：准确率+23.97pp，95%家族区间[+20.57,+27.18]pp。

minicpm5/best_constrained 相对 ref_selected：准确率+22.26pp，95%家族区间[+18.92,+25.34]pp。


|开发能力任务|原模型|最高分方案|受约束方案|
|---|---:|---:|---:|
|reading|72.96%|88.27%|88.27%|
|literal|98.98%|100.00%|100.00%|
|binding|96.43%|98.98%|98.98%|
|role_swap|75.00%|93.88%|93.88%|
|record|64.03%|73.21%|73.21%|

|模型/方案|开发准确率|开发成对均对|能力约束|历史Type2准确率|历史Type2成对均对|
|---|---:|---:|---|---:|---:|
|qwen25/baseline|95.41%|90.82%|通过|96.23%|92.47%|
|qwen25/best_task|96.94%|93.88%|通过|96.92%|93.84%|
|qwen25/best_constrained|96.94%|93.88%|通过|96.92%|93.84%|
|qwen25/ref_selected|96.43%|92.86%|通过|96.40%|92.81%|
|qwen25/ref_fine_combo|95.41%|90.82%|通过|96.06%|92.47%|
|qwen25/ref_norm_whole|95.92%|91.84%|通过|96.23%|92.81%|

qwen25 搜索量：{'short': 576, 'screen': 198, 'dev': 22}。
最高分方案：L15 MLP/上下文代词/gap_128_pos(128维)/center λ=1。
满足能力约束方案：L15 MLP/上下文代词/gap_128_pos(128维)/center λ=1。

qwen25/best_task 相对 baseline：准确率+0.68pp，95%家族区间[+0.00,+1.67]pp。

qwen25/best_task 相对 ref_selected：准确率+0.51pp，95%家族区间[+0.00,+1.20]pp。

qwen25/best_constrained 相对 baseline：准确率+0.68pp，95%家族区间[+0.00,+1.67]pp。

qwen25/best_constrained 相对 ref_selected：准确率+0.51pp，95%家族区间[+0.00,+1.20]pp。


|开发能力任务|原模型|最高分方案|受约束方案|
|---|---:|---:|---:|
|reading|100.00%|100.00%|100.00%|
|literal|100.00%|100.00%|100.00%|
|binding|100.00%|99.49%|99.49%|
|role_swap|97.45%|97.96%|97.96%|
|record|66.07%|65.05%|65.05%|

区间是探索性、未经多重比较校正的家族bootstrap；开发最佳与测试最高不是同一概念，未据测试重新选参。
已有参考方案同批次指标复现、所有冻结输入指纹、开发选择重放和逐条预测指标复核均通过。

干预算子：center为u′=μ+(1−λ)(u−μ)，raw为u′=(1−λ)u。λ>1会反转偏离均值的方向；raw λ=1.5是原激活乘−0.5，不能仅称置零。
Qwen2.5反向数值修正采用1024/256/64/16/4/1中的最大有限损失缩放，并除回实际缩放；前向结果不变，无样本跳过。原始实现和其他两模型冻结选择保留，详见numerical_v2_declaration.json。
本轮159项工程测试通过。独立复核的梯度有限性、坐标排名重建及逐条参考预测一致性见independent_verification.json。

独立重算已完成：16,936条预测、三个模型全部排名坐标和拟合均值一致，旧参考逐条预测一致；Qwen2.5的256条归因输入中248条使用1024缩放、8条使用256缩放，无样本跳过。
对比图已导出为 results/identity_best_combo_20261016_v1/task_score_comparison.png 和同名PDF，并完成图像检查。
