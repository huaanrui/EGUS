> 2026-10-11 更新：本文中的 v2 数据和880题 fit 数量为历史记录。当前有效版本为 `bbq_pilot_v4`，fit 1664题；v2 的非负问句偏见标签已更正。详见 [实现审计](2026-10-11_implementation_audit.md)。

# BBQ 预实验数据准备审计

日期：2026-10-09。有效目录：`data/prepared/bbq_pilot_v2/`。

来源：[nyu-mll/BBQ](https://github.com/nyu-mll/BBQ)，固定 revision `bea11bd97d79217245b5871acd247b9d6eb24598`，CC-BY-4.0，原始许可证随缓存保存。

## 来源检查

- Age：3,680条，25个 question_index。
- Gender_identity：5,672条，50个 question_index。
- SES：6,864条，25个 question_index。
- 共16,216条；16条 Gender_identity:13 记录缺少 target_loc，隔离后可用16,200条。排除清单在 manifest 的 excluded_rows 中。
- 补充表中64个重复键只在 Known_stereotyped_groups 描述不同，参与评分的字段一致。来源原样保留，消费时合并，冲突数记入 manifest。

## 分组检查

不能仅按 question_index 视作独立情景。例：Gender_identity:1 和 :26 都是会议迟到、判断秘书身份的情景，分别用性别标签和人名表达。人工识别的22组此类对应关系已固定到 family mapping。合并后总计78个模板家族。

| split | 行数 | 家族数 | Age | Gender | SES |
|---|---:|---:|---:|---:|---:|
| fit | 880 | 47 | 240 | 400 | 240 |
| selection | 312 | 16 | 80 | 152 | 80 |
| validation | 264 | 15 | 80 | 104 | 80 |

每个分区中 ambig/disambig、neg/nonneg 均衡。sample 并非严格逐条反事实对，做配对 patching 前还需建立并审查配对。

自动检查覆盖：文件 SHA-256、来源完整性、UID唯一、精确重复题与家族无跨分区、答案映射有效、信息不足的正确答案为unknown。当前全部通过。

语义层面仍需检查：不同编号之间尚未识别的近义情景、目标群体与非负问法映射、来源本身的歧义和不自然文本。当前规则减少已知泄漏，不保证不存在一切语义重叠。所有数据检查发生在模型评估前。

旧 `bbq_pilot_v1` 仅按原始编号划分，已在 manifest 标记 superseded_do_not_use。所有配置均指向 v2。

## 重现与使用

```powershell
.\.venv\Scripts\python.exe scripts\prepare_bbq.py --verify-only
```

在未生成过输出的工作副本上，可执行 `scripts\prepare_bbq.py` 下载同版本来源并生成划分；已有 raw 缓存时可加 `--offline`。为避免覆盖已使用划分，现存输出目录非空时准备命令会拒绝执行。

模型提示只能取 context、question、answers。正确答案、刻板目标及分区只用于数据管理、fit阶段发现和离线评分。

尚未运行模型或干预实验，没有任何模型性能、偏见变化或方法排名结果。
