# BBQ 外部去偏基线预实验：冻结协议

2026-10-13；Qwen3-1.7B FP16 / SDPA。新结果写入 `results/bbq_comparators_20261013_v1`，旧实验只读。此协议在本轮拟合及开发/测试评价前冻结。

## 共同数据和评价

复用 improvement v2 家族隔离列表：fit 33352 题、dev 352 题、test 704 题，zero/few-shot 各一次；controller 分区不用于本轮新基线拟合。推理保留原始 A./B./C. 格式、原始选项顺序、逐题前向，读取同一个答案 token 的 logits；能力 dev/test 为旧实验的 64/128 道四选项作者筛选 MMLU。禁止截断。激活与 logits 用无干预前向同旧缓存精确核验。

历史全量 BBQ 和本轮 test 的其他方法结果已被观察，这是探索性比较，不宣称新的封存确认。不能把不同论文的模型和样本上的数字直接填入比较表。

主报告全 11 类，并另报 KnowBias 性别/种族/宗教、CRISPR SES/年龄/残障、BBA/FBA 性别/国籍/宗教的类别子集。所有方法在共同评价中使用统一组件，不依据推理题的官方类别选择不同组件。BBA/FBA 的全类别组件用三个维度归因排名的平均名次融合，属于跨类别适配，非作者逐类别配置复现。

## KnowBias 适配

来源论文：https://proceedings.mlr.press/v306/pan26j.html 和 https://arxiv.org/html/2601.21864；作者代码固定提交 `ccfd3daea0067707cbda6238839206c194794881`，本地已有快照。

作者发布的 CSV 是 75 条同时提及多个维度的问题，主论文设置是 45 条单维度问题。预先选择 CSV 中 math/science/leadership/emotional intelligence/creativity 五个概念，保留其三种问法，按 gender/race/religion 展开为 45 条，预期 yes/no 沿用 CSV。问题文本单独保存。没有按 BBQ test 或模型效果选择概念。

入口缺少的 `evaluate_know_new.txt` 用固定文本 `Answer the following question with only yes or no.\nQuestion: ...` 补全；system 保留 `You are a useful assistant.`；Qwen chat template 显式关闭 thinking。知识定位是 chat 格式，共同 BBQ 评测是作者 FairSteer raw 格式，需披露该迁移。

在全部 28 层 `mlp.gate_proj` 输出的 prompt 最后 token 上作 IG，零基线、20 个等距端点（含 0/1），全词表 softmax 的目标 token 概率，梯度与积分 FP32；模型主体冻结 FP16。20 步以 microbatch 4 计算。对不变前缀缓存 KV，最后 token 仍经过完整模型，在目标 gate 替换；KV 每次深复制，禁止累计追加污染。先对 0/13/27 层用 4 步检验缓存版本与完整前向 IG 的余弦 >.99、相对 L2 <.08。此处允许缓存引起的小数值差异，不能称逐 bit 作者归因复现。

每题以全层全神经元 IG 最大值的 10% 筛选；每个维度保留出现次数严格大于题数×beta 的节点，三维度取并集；beta∈{.05,.1}。推理在同一 gate_proj 输出、最后 prompt token 对选中坐标乘 lambda，lambda∈{1.5,2,3.5,5}。beta=.1 / lambda=3.5 包含论文 Qwen3-4B 设置。额外 45 条知识问题不计作 BBQ fit 数据。

## CRISPR-Qwen-MLP 适配

来源：https://arxiv.org/html/2311.09627，原研究为 Flan-T5/T0，不声称原模型复现。

从共同 fit 每类取 20 题（各 10 歧义/消歧；家族/极性轮转采样），总计 220 题、zero/few-shot 各一次。每题从 ABC 选择概率最大的错误选项 y_wrong；A_i(y)=a_i*dP_vocab(y)/da_i，B=A_i(wrong)−max(A_i(correct),0)；对 token 取最大值、以 P_vocab(wrong) 加权样本求和、对两种 shot 等权平均。训练监督含正确选项，不使用测试标签。

激活定义为门控乘积后 `mlp.down_proj` 输入。全 28 层全 token 归因，全局按有符号 B 排序；推理将 top k 坐标在所有 token 置零，k∈{32,128,512,2048}。对无 bias 的 Qwen down_proj，此操作等价于删除对应输入列的贡献，权重文件不改写。

与原文差异：仅 MLP；输出为单字母而非答案文本；两种现有 shot 条件替代原十条同义指令；未复现三次定位采样。所有差异显式记录，不能称完整 CRISPR。梯度乘 1024 再除回以降低 FP16 反传下溢；概率和聚合 FP32。

## FBA / BBA 适配

来源：https://arxiv.org/html/2602.04398；作者代码固定树 `a24aee92e869d2657b7e0848158a213c253f1c1c`，下载路径及 SHA256 见 `research/sources/20261013/BBA/source_manifest.json`。作者部分合成题/挑选结果未提供，由公开原始候选词与模板重建。

在 gender/nationality/religion 三维度的公开候选词（去重）和全部 17 个 adjective /17 个 noun 模板上，按作者代码用候选群体首 token 的全词表概率平均后计算熵；每类选熵最小 20 个词。这个过程不访问 BBQ 标签或 test。选词 batch32、left padding，显式 position_ids；chat 关闭 thinking。

FBA：top20×两类×17模板，问填群体；BBA：两类×17模板×各群体，问填已选线索词。按作者代码 greedy 最多生成4 token，捕获最终 norm 后、最后生成步的读出输入；考虑 EOS 实际步数。作者梯度仅作用于最后步，其余步为零，对全部步平均等价于归因除以实际步数，此处保存实际步数并计算相同缩放。

FBA 目标是候选群体归一化分布的倒熵；多 token 群体按照作者代码在同一个位置将概率相乘（不等于严格自回归词概率），用 log 空间稳定计算。BBA 用候选线索词首 token 的分布在群体间的 Jensen-Shannon divergence，temperature=.1；归一化抵消词表 softmax 分母，直接对子集 logits 计算。零基线20步端点积分，float32 head/梯度防止 FP16 的极端概率下溢；与作者 FP16 算术有差异。

同维度内平均归因；跨三维度用平均名次融合成统一坐标集。选 ratio∈{.1,.2,.3,.4}，constant∈{−2,−1,0,1,2}，这与公开网格相同。

**定位与干预位置分别标明**：主解释版本 `post` 在最终 norm 后、lm_head 输入修改；作者示例代码版本 `pre` 在最后 block 输出、最终 norm 前修改。两版本均列出，不能从 test 二选一。对一步 ABC/ABCD 预测，最终 block 其他 token 已无到最后位置的路径，只修改最后位置与该处修改全部位置具有同样输出，使用无干预原始状态缓存精确重建读出，不重复完整模型。原样和修改读出通过 GPU 验证。结果名称带 pre/post。

与原论文不同：Qwen3-1.7B；共同 raw 字母提示；统一跨维度集合；明确关闭 thinking；重建提示空白布局；稳定 FP32 读出归因。公开例子评估 StereoSet 的提示/词解析与这里也不同。该实验是公开方法的 BBQ 适配，不宣称严格原配置复现。

## 选择、统计和额外对照

各方法先跑完 dev 的规范网格。与旧实验相同：最大化整体准确率，要求消歧≥max(原模型−1pp,FairSteer−.5pp)、能力≥原模型−2pp、|BS(a)|≤FairSteer+1pp；若无可行候选，保留准确率最高候选并标记失败，不按 test 修改。报告全部候选和取舍。各组搜索预算不同，公开实际候选数量、定位样本和计算成本；不能称等计算量比赛。

同步添加两个更强 unknown 对照：p_insufficient>阈值(.25/.5/.75)时选择输入中可识别的 unknown，否则原预测；或将 offset×p_insufficient 加到 unknown logits，offset∈{.5,1,2,4}。使用旧证据头，推理不读真实 context_condition 或正确标签。MMLU 没有 unknown 选项，按预设规则保持原始预测；能力收益不可由此推出通用安全性。两对照用同一 dev 约束挑选。

报告整体/歧义/消歧准确率，BS(a)/BS(d)、S/C/U、消歧弃答、MMLU 子集，zero/few-shot 与类别子集，以及相对 FairSteer/证据门控按类别内家族配对 bootstrap2000次的95%区间。多方法未经校正，探索性。不得省略差结果或因无可行候选隐去方法。

本轮范围是外部基线比较及 unknown 对照；跨算子证据门控组合、随机节点、多种子、第二模型、等长证据反事实和电路验证仍为后续工作，不能用本轮结果替代这些证据。
