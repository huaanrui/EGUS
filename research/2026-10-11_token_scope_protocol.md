# Token 提取、聚合与干预范围：预实验实现约定

日期：2026-10-11。补充 [预实验总方案](2026-10-09_preexperiment_plan.md)。状态：设计与本地模板核对；尚未运行激活提取/干预实验。

## 1. 分开记录三个选择

1. extraction mask：提取哪些位置，例如最后一个提示token、提示末端固定模板区域、用户正文。
2. pooling：选中位置分别处理，还是先在每条样本内部求均值。
3. intervention mask：真正修改哪些位置，以及只在prompt prefill生效，还是在后续decode也生效。

提取与干预范围可以不同，但这代表跨位置使用，必须单独评估。不能用同一个 `last_k` 参数暗中同时控制这三个选择。单位应记录为 `(hook_site, layer, token_scope, coordinate_or_direction)`。

首轮主条件继续采用：**最后一个有效prompt token提取、无token pooling、只干预这个prompt位置**。它提供受控、廉价的作答前局部比较；并不保证它是效果最强的位置，也不涵盖全部偏见形成机制。

## 2. last token 的精确定义

设完整模型输入的有效prompt为 `x[0], ..., x[n-1]`。其中包括系统提示、用户文本、chat template、assistant开头，以及手工添加的任何固定答案前缀。`last_prompt = n-1`，其最终隐藏状态用于预测第一个输出token。

它与以下概念不同：最后一个user正文token、最后一个问题token、最后一个已生成token、batch张量的最后一列。实验配置与日志不能把这些都叫 last token。

本地 `F:\LLMs\Qwen3-1.7B\tokenizer_config.json` 的模板确认：`add_generation_prompt=True, enable_thinking=False` 会在assistant开头追加 `<think>\n\n</think>\n\n`。因此模板末尾不是问题的最后一个词。模板标记上的隐藏状态仍能结合此前上下文，不能因为它的表面文本固定就认为没有信息。

不需要为“更有语义”而自动删除这些标记。首轮沿用模型原生模板；如果加入 `Final answer:` 等固定前缀，所有方法都使用同一前缀，并单列为提示格式变化。具体token边界以实际tokenizer输出为准，不用字符长度猜测。

本轮已用本地 AutoTokenizer 对一条 fit 样本实际核对（未加载模型权重）：非思考模式末尾4个token依次为 `<think>`（151667）、双换行（271）、`</think>`（151668）、双换行（271）。在该测试提示下，追加 A/B/C 不改变prefix分词，分别为单token 32/33/34。这仅确认该提示格式样本的token行为；正式冻结模板后仍应批量检查全部输入。thinking开关改变了后缀，因此两个模式的last4不是语义相同的位置。

padding处理：通用写法是在 attention_mask==1 的位置中取最大位置索引。右padding的 `sum(mask)-1` 正确，但不能原样用于左padding；`h[:, -1]` 则会在右padding时取到PAD。生成hook另需保存原始prompt长度和全局位置。

## 3. last-k 与 all-token pooling 如何选择

| 条件 | 确切范围 | 用途 | 注意事项 |
|---|---|---|---|
| last_prompt | 完整prompt最后一个有效token | 默认作答前状态 | 对模板/固定前缀有依赖 |
| suffix_last4_mean | post-instruction固定后缀内最后4个token | 小范围位置稳健性 | 不能偷偷越过后缀，混入最后一个选项 |
| post_instruction_each | 用户正文之后的模板token逐位置提取 | 寻找最佳读出位置 | 每个位置算单独候选，消耗搜索预算 |
| user_content_mean | 最后一条用户消息的非padding正文token | 全文聚合作为对照 | 长背景/选项可能主导，前面token看不到后文 |
| field_balanced_mean | context/question/options分别均值后等权合成 | 排除文本字段长度影响的后续对照 | 与全文均值是不同定义 |

第一轮只用 last、suffix_last4、user_content_mean 三种提取定义，避免同时扫 k=1/2/4/8/16/32。`all token` 这个名称过于含糊：必须说明是否含system、模板、padding、候选答案、已生成CoT。这里的全文均值限定为用户正文，禁止包含候选答案token和padding。

post-instruction区域按用户消息正文终点界定，包括其后的模板边界和assistant前缀。若该区域不足4个token，显式报错或预先改成长度自适应的另一个条件，不让last4越界到不同语义字段。固定4个token可能大多是换行或思考标记；该条件检验的是模板末端状态，不是“最后4个内容词”。

自回归模型中各位置的可见上下文不同。全token平均会混合尚未看到完整问题/选项的状态，不能预设它比最后位置更稳定或信息更多。

聚合先在样本内部，再在样本/模板家族间：

`z_l(x) = sum_{t in E(x)} h_l,t(x) / |E(x)|`

`mu_class = weighted_mean_x z_l(x)`，权重沿用总方案的类别/家族平衡。

不要把所有样本的token堆起来做一次总平均，否则长输入的权重更大。标准化、CAV拟合、方向归一化、投影方差计算全部只用fit。

## 4. 干预范围和自回归时间

定义四种明确范围：

- `prompt_anchor_only`：仅prefill中的原始last_prompt位置。
- `prompt_suffix4_only`：仅prefill中的post-instruction后缀最后4个位置。
- `prompt_all_valid`：仅prefill的全部非padding位置，默认包括system与模板；若只改用户正文须另称 `prompt_user_content`。
- `prompt_all_plus_decode`：所有有效prompt位置，以及后续处理每个新token时的相应激活。自由生成实验中单独比较。

提示末端提取的方向可以用于全prompt，但不能假定所有位置都表达同一种概念；这是需要验证的泛化。方向从哪个位置提取，与哪些位置被修改，分别记录。

自回归的关键索引：

```
prefill最后位置 x[n-1] 的状态 -> 预测 y[0]
处理已生成 y[0] 的状态      -> 预测 y[1]
处理已生成 y[1] 的状态      -> 预测 y[2]
```

因此，若答案 A/B/C 经检查均为单token，**只在生成第一个答案token之后干预decode，不能改变已经选出的首token**。这个条件不适合单token分类主实验。

KV cache开启后，decode通常只返回当前新位置。每次hook都修改 `hidden[:, -1]` 实际可能变成“每一步都干预”，而不是“仅原始prompt末尾”。实现必须使用原prompt边界、全局position/cache_position或显式阶段状态构建mask；不能只用 `seq_len == 1` 判断阶段。

不同干预条件必须重新计算受影响的prefill与cache。不能先用原模型缓存，再把开启干预后的decode当作全程干预。cache on/off在相同条件下应给出近似一致的候选分数。

多token候选评分时，`y[j]` 对应输入的 `n+j-1` 位置logits。prompt-only策略仍只修改原prompt位置，不随着候选拼接重算last token。提取方向/参考均值用prompt-only输入；不要把正确答案追加后再取末尾激活，否则信息可直接泄漏。

## 5. 不同方法不能照搬同一种 pooling

### DiM / CAV

对选定位置的隐藏状态做样本内pooling，再计算均值差/拟合分类面。不同pooling各自拟合方向，而不是把last方向叫作all-mean方向。

主比较在同一读出位置应用干预；扩大范围时保持方向不变，以隔离scope效应。另行重新拟合是另一个实验轴。对于SCAV的边界移动，若分类器在pooled embedding上训练，把它当成每个token的已校准风险分类器没有依据。首轮线性CAV只提取方向；逐token边界移动需匹配的fit分布和单独验证。

### 神经元 / RelP

应先在具体 `(layer, token, neuron)` 上求重要性，再按声明的目标聚合到共享神经元排名。不能先平均激活，再声称得到了同样的归因。

例如 `mean_t(|a_t,i| * ||W_i||)` 与 `|mean_t(a_t,i)| * ||W_i||` 不同：后者可能因符号抵消把重要神经元隐藏掉。类似地，对 m 的带符号归因保留正负作用，绝对值排名另列，不能混用。

如果最终要在多个token上改同一个神经元集合，fit时应在同一声明范围内计算重要性。若分析位置特异的电路，节点保留 `(layer, token_role/relative_position, neuron)`；去掉token位置后得到的是共享神经元集合，不能直接当作同一个电路。

### 层

层hook也必须接受token mask。对 last_prompt 进行某层残差更新替换，与对全部token替换，是两个不同条件。沿用总方案只修改 block update，保留残差旁路。

## 6. 均值与干预强度要匹配作用位置

主条件用每层/hook在fit的last_prompt均值。跨多个固定后缀位置时，用每个相对位置自己的fit均值；用户正文长度不固定时，可以预先按context/question/options等可观测角色求均值。

角色由输入格式确定，不依赖正确答案或群体标签。不能根据验证题的目标人群、正确选项、ambig/disambig标签挑替换值。参考均值冻结并保存来源ID；不能用当前batch或验证数据实时估计均值。

全token套用last_prompt均值通常是分布失配，会把正文位置改成回答边界的典型状态；这不能作为无说明的默认。为了隔离范围本身的影响，可在零参考消融下补做scope对照，再与角色均值结果并列。

相同 α 并不意味着改1个与改100个token的强度相同。记录每token平均扰动和总体扰动，例如：

`E = sqrt(sum_{l,t in I} ||delta_h_l,t||²)`，同时报告mask数量。

另报其相对原激活/残差更新能量的比例，并比较去偏—能力损失曲线。`alpha / sqrt(|I|)` 只在同一层、等幅单位方向加法等条件下匹配注入的L2能量，不保证实际效应相等；投影消融和跨层反馈不能直接照用。

## 7. 有限预算的5个条件

| 编号 | 提取 | 干预 | 目的 |
|---|---|---|---|
| T0 | last_prompt | prompt_anchor_only | 全方法共同主条件 |
| T1 | suffix_last4_mean | prompt_anchor_only | 只改变提取pooling |
| T2 | user_content_mean | prompt_anchor_only | 全文提取对照 |
| T3 | last_prompt | prompt_suffix4_only | 只扩大干预范围 |
| T4 | last_prompt | prompt_all_valid | 更广干预的效果与副作用 |

T1/T2首轮针对DiM与CAV；神经元/RelP对应的是改变归因范围，不能把它叫同一个pooling操作。T3/T4也可迁移到固定神经元集合，但要明确其发现位置与应用位置不一致。

先完成T0，确认hook和基线正确。然后在selection上做小规模T1–T4敏感性分析，所有比较保持发现样本、层、方向/排名版本一致，只改变表中指定项。扩大范围需要匹配参考均值，须记录这项操作适配；零参考补充实验可帮助隔离scope。

这些条件计入每方法最多32个selection配置，不是额外允许5×32。validation可以评估每方法冻结的最终方案；若要比较完整T0–T4表，则所有条件在看validation之前预先冻结，并作为多条件分析报告，不能再据此挑“最佳位置”。

## 8. 实现前必须通过的检查

- 保存实际rendered prompt、token IDs、最后12个token解码、各mask的位置与文本片段；避免只存字符串名称。
- 检查每个样本last位置非PAD、last4不越过post-instruction边界、答案置换不改变mask语义。
- 检查candidate拼接是否改变prefix分词；多token候选使用正确的shift和完整序列概率。
- hook记录每次修改的全局位置与阶段，验证prompt-anchor-only在decode不会再次触发。
- λ/α=0与原模型一致，去掉hook恢复，cache on/off分数相近。
- 只改已经生成的答案token之后的decode，首token概率应不变；这是抓取一步偏移错误的负对照。
- 对本模型最后一层的MLP，仅改变非末尾prompt位置时，当前首答案token的logits应不变，因为后续没有attention再将这些位置混入末尾。若发生变化，应检查mask/缓存/实现；该结论不适用于更早层或不同hook点。
- 所有μ/σ/方向/排名仅依赖fit；selection只选择配置；validation无再拟合。

以上检查是待实现的验收标准，不代表本轮已经完成模型hook测试。

## 9. 与论文设置的关系

拒答单方向论文在用户指令之后的模板位置逐位置提取候选方向，并在留出数据上挑选；它的激活加法/方向消融覆盖范围比单一提取位置更广。这说明提取与干预位置可以不同，也说明复现时必须分别记录，不能只写“用last token方向”。[论文方法](https://arxiv.org/html/2406.11717v3#S2)

本协议的T0是为偏见预实验建立可解释、可对齐的局部主条件；原式全token/多层干预作为另列结果。无需事先宣称last、last-k或all-token中任何一种普遍最好。
