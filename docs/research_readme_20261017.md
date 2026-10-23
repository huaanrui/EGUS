# 偏见问答中的证据识别与弃答控制

截至 2026-10-17，当前主线是：在具有可识别弃答选项的偏见问答中，识别证据充分性可以改善任务内表现；这种收益不能直接解释为通用去偏能力。三模型、五训练抽样种子的补充实验已完成，无待续跑的实验。

## 从这里开始

- [完整结果总结](research/2026-10-17_project_summary.md)：当前主结论、历史探索、统计与局限。
- [代码与实验索引](research/CODE_MAP.md)：主线入口、共享依赖、历史实验和替代版本。
- [最新详细结果](research/2026-10-17_evidence_validation_findings.md)：逐模型、逐种子、充分/不足分项与统计检验。
- [最新冻结协议](research/2026-10-17_evidence_validation_protocol_v1.md)。
- [整理记录与恢复方法](archive/2026-10-17_cleanup/README.md)。

## 最新结果

BBQ 704 题，raw zero-shot；每次 2200 题拟合、352 题开发集选阈值。数值为五次训练抽样的平均准确率 ± 样本标准差（%），不是五个独立测试集。

|方法|Qwen3-1.7B|MiniCPM5-2B|Qwen2.5-3B-Instruct|
|---|---:|---:|---:|
|原模型|56.68|43.47|73.01|
|本地 FairSteer 风格|65.45 ± 0.82|43.24 ± 0.22|73.38 ± 0.48|
|乘积门控＋方向|73.89 ± 1.78|43.15 ± 0.12|75.85 ± 0.85|
|表面特征直接弃答|75.99 ± 0.17|78.27 ± 0.28|80.97 ± 0.00|
|单证据头直接弃答|79.09 ± 0.54|84.43 ± 0.24|81.28 ± 0.31|

相对表面特征基线，证据头的三项 BBQ 优势均未通过预定 12 项比较的 Holm 校正。新 SynthBias 确认集上，证据头造成充分题准确率下降 40.88、44.75、14.00 个百分点，不能以总体分数提高宣称稳健迁移。历史 BBQ 已被反复观察，属于探索性评价。

## 使用与复核

在仓库根目录执行。使用已有 `.venv`；依赖声明见 `pyproject.toml`，锁文件为 `uv.lock`。模型路径、设备与数据路径见各冻结配置，当前配置依赖本机 `F:/LLMs`。

```powershell
# CPU 单元测试；不启动模型推理
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v

# 只核验本次归档和保留源码哈希
& '.\.venv\Scripts\python.exe' scripts/maintenance/organize_20261017.py --verify
```

主线实现是 `mechanism/evidence_validation.py`、`mechanism/evidence_validation_stats.py`，配置为 `configs/experiment_evidence_validation_v1.toml`。正式结果在 `results/evidence_validation_20261017_v1/`，准备数据在同名 `data/prepared/` 目录。

以下是原实验入口说明，**不是本次整理需要重新运行的步骤**：

|入口|作用|
|---|---|
|`scripts/run_evidence_validation.py`|准备数据，依次拟合三模型五种子，再完成实际干预前向；需要 GPU|
|`scripts/summarize_evidence_validation.py`|读取缓存重算预测、统计并写报告；会覆盖派生文件|
|`scripts/verify_evidence_validation.py`|独立重拟合分类器、重放选择并核验统计；写审计文件|
|`scripts/plot_evidence_validation.py`|读取结果并写图表|

已有结果是冻结记录。报告重生成也会改写文件；新实验必须使用独立版本、输出目录和协议，不能修改既有配置后覆盖结果。此前报告中人工补充的说明需单独保留。

## 目录约定

|目录|内容与维护原则|
|---|---|
|`mechanism/`|实验实现和共享组件；旧命名模块仍有实际依赖|
|`scripts/`|保留的实验、分析及审计入口，分类见代码索引|
|`configs/`|冻结实验配置；旧版本用于追溯，不覆盖|
|`tests/`|实现与协议检查；测试数量不代表独立科学证据数量|
|`research/`|协议、解释、报告；`sources/` 为第三方参考实现|
|`data/`、`results/`|原始/准备数据、拟合产物、预测、冻结记录；本次整理未修改|
|`archive/`|不用的临时工具和原项目说明，带原路径及 SHA256，可恢复|

`main.py` 和 `python -m mechanism` **仍是首轮 pilot 的历史入口**，不是当前主线。保留其行为以兼容旧实验；不要据文件名把它当作最新实验启动器。

本地 FairSteer、KnowBias 等明确记录了模型、监督预算、求解器或任务适配差异，不称严格复现。直接弃答要求输入中存在可识别的弃答选项；没有这类选项时的不动作不能证明通用能力无损。
