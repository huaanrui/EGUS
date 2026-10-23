# 代码与实验索引（2026-10-23 补充）

仓库对外入口和本地／云端整合说明见 `../docs/CODE_ORGANIZATION.md`，复现步骤见 `../docs/REPRODUCING.md`。新增 `scripts/release/` 用于恢复固定 BBQ 样本、启动独立复现实验及复核六模型分数；云端扩展实现是 `mechanism/cloud_bbq.py`。下文保留 2026-10-17 的历史索引与边界说明。第三方源码和原始数据不随 Git 仓库分发。

本索引按研究用途整理代码，不改变已冻结实验的科学实现。历史探索不等于无用代码：有的仍是当前模块的依赖，有的负责复核旧结果。

## 当前论文主线

|部分|实现 / 入口|对应配置或输出|
|---|---|---|
|三模型五种子验证|`mechanism/evidence_validation.py`；`scripts/run_evidence_validation.py`|`configs/experiment_evidence_validation_v1.toml`|
|交叉重采样、配对检验、Holm|`mechanism/evidence_validation_stats.py`|主统计见 `results/evidence_validation_20261017_v1/results.json`|
|汇总、独立复核、作图|`scripts/summarize_evidence_validation.py`、`verify_evidence_validation.py`、`plot_evidence_validation.py`|同目录报告、审计、PNG/PDF|
|证据识别与干预方式对照|`mechanism/evidence_story.py`；`scripts/run_evidence_story.py`|`configs/experiment_evidence_story_v1.toml`|
|表面/输出分数对照补充|`scripts/evidence_story_surface_route_v2.py`、`evidence_story_output_control.py`|`results/evidence_story_20261017_v1/`|
|基础对照报告与核验|`scripts/summarize_evidence_story.py`、`verify_evidence_story.py`、`plot_evidence_story.py`|同上|

主线直接导入依赖示意：

```text
evidence_validation
  ├─ evidence_story          特征、拟合、阈值、干预与等长控制
  ├─ fairsteer_author        ROOT、数据读写、哈希、标签等
  ├─ fairsteer_improve       样本抽取、家族标识等
  └─ closed_transfer         SynthBias 规范化、模板与性别元数据
```

不能把 `fairsteer_*`、`closed_transfer.py` 按“旧实验”直接移出包。此次保持 `mechanism/`、`configs/`、`tests/` 原内容与路径。未来抽取公共库应另开版本，并与本轮逐题预测作等价性验证。

## 历史实验与支撑材料

下表简写的实现位于 `mechanism/`，运行脚本位于 `scripts/`；说明位于 `research/`。

|研究阶段|主要实现 / 入口|结果与解释|
|---|---|---|
|早期 token/组件消融|`__main__.py`、`round2.py`；`run_pilot.py`、`run_round2.py`|`2026-10-11_pilot_results.md`、`2026-10-12_round2_results.md`|
|FairSteer 对齐|`fairsteer.py`、`fairsteer_revision.py`、`fairsteer_author.py`；对应 `run_fairsteer*.py`|`results/fairsteer_author_qwen3_v1/`；早期配置不作严格复现|
|门控等改进探索|`fairsteer_improve.py`；`run_fairsteer_improve.py`|`results/fairsteer_improve_20261013_v2/`|
|外部方法比较|`bbq_comparators.py`；`run_bbq_comparators.py`、`run_bbq_readout_domains_v2.py`|`2026-10-13_bbq_comparator_findings.md`|
|开放续写与本地裁判|`ceb_transfer.py`、`ceb_local_judge.py`；对应 `run_ceb*.py`、`package_ceb_judging.py`|`2026-10-13_ceb_local_judge_findings.md`|
|无选项问答|`evidence_openqa.py`；`run_evidence_openqa.py`、`run_evidence_openqa_threshold_v2.py`|`2026-10-14_evidence_openqa_findings.md`|
|封闭任务迁移与校准|`closed_transfer.py`、`evidence_mechanism.py`；对应运行/校准脚本|`2026-10-14_closed_transfer_findings.md`、`2026-10-14_evidence_mechanism_findings.md`|
|身份干预与传导路径|`identity_causal.py`、`identity_path.py`、`identity_parts.py`、`identity_combo.py`；同名运行/分析脚本|各 `identity_*` 协议与 findings；有效 path 为 v3|
|跨模型组件复验|`identity_replication.py`；`run_identity_replication.py` 及执行/报告修复脚本|`2026-10-15_identity_replication_findings.md`；同时看角色平衡结果|
|组件最高分探索|`identity_best_combo.py`；`run_identity_best_combo.py`、`run_identity_best_combo_numerical_v2.py`|`2026-10-16_identity_best_combo_findings.md`；有限空间搜索，不是因果路径发现|

`audit_*`、`verify_*`、`review_*`、`summarize_*` 及语义/范数对照是相应结果的复核依赖，保留使用。

## 替代关系与归档

|旧入口|当前状态|使用原则|
|---|---|---|
|`scripts/evidence_story_surface_route.py`|历史冻结 v1；保留原位|报告修复后使用 `evidence_story_surface_route_v2.py`|
|`scripts/run_bbq_readout_domains.py`|历史冻结 v1；保留原位|正式按域结果使用 `run_bbq_readout_domains_v2.py`|
|`scripts/closed_transfer_stable_calibration.py`|审计依赖，保留原位|v2 是数值稳定补充；两版结果都保留，不择优混报|
|`scripts/resume_identity_replication_execution_v2.py`|被 v3/测试引用，保留原位|执行修复须连同 v2/v3 协议解释|
|`main.py` / `python -m mechanism`|首轮 pilot 兼容入口|当前使用明确命名的 validation 入口|
|五个预览/迁移/探测工具|移入 `archive/2026-10-17_cleanup/scripts/`|原因、原路径、哈希见归档清单|

旧配置和协议保存当时实际执行的规则，不统一重命名版本号，不改写历史说明。第三方 `research/sources/` 保留原位。

## 复现边界

- 当前 validation 依赖历史准备数据、原种子拟合资产和本地模型；不是只复制一个脚本就能运行的独立复现包。外发前需另做数据许可、资源路径及环境说明。
- 新确认 640 题现已观察。下一轮若据此改进，不能再把它当作新的独立确认。
- 旧 zero/few-shot 合并成绩与最新 raw zero-shot、2200 训练题成绩分表呈现。
- 本次整理只做文件管理、文档与测试，不重新拟合、不重新生成实验报告。
