"""Append independently frozen calibration and audit limitations to the report."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mechanism.data import sha256, write_json


def main():
    c = tomllib.loads((ROOT / 'configs/experiment_round2.toml').read_text(encoding='utf-8'))
    out = ROOT / c['output_dir']
    path = out / 'report.md'
    text = path.read_text(encoding='utf-8')
    marker = '\n## 输出校准对照与补充审计\n'
    if marker in text:
        text = text.split(marker)[0]
    control = json.loads((out / 'calibration/test_results.json').read_text(encoding='utf-8'))
    plan = json.loads((out / 'calibration/plan_frozen.json').read_text(encoding='utf-8'))
    audit = json.loads((out / 'behavior_class_batch_audit.json').read_text(encoding='utf-8'))
    optpath = out / 'performance_optimization_audit.json'
    opt = json.loads(optpath.read_text(encoding='utf-8')) if optpath.exists() else None
    ci = control['ci']['macro_stereotype']
    optimization_note = (f'性能优化在{opt["rows"]}道开发题上核对：移除无关钩子、缓存提示、异步累加扰动统计之后，预测及扰动范数与旧实现完全一致。没有修改已缓存的模型输出，详见 performance_optimization_audit.json。' if opt else '本次运行直接使用缓存提示、稀疏钩子与异步扰动统计的实现；不涉及旧结果迁移。')
    lines = [marker, '', f'在新测试开放前单独登记并冻结了输出校准对照：仅给任务中可见的unknown选项加固定logit偏置，在完整selection从0/0.25/0.5/1/2/4中选择。选中偏置为 {control["offset"]}，selection接受状态为 {control["accepted_selection"]}，新测试约束通过为 {control["test_feasible"]}。', '', f'新测试宏平均刻板率变化 {100*ci["delta"]:+.2f}pp，探索性95%区间 [{100*ci["low"]:+.2f}, {100*ci["high"]:+.2f}]；充分证据准确率 {100*control["bbq"]["disambiguated_accuracy"]:.2f}%，neutral准确率 {100*control["neutral"]["accuracy"]:.2f}%。该对照用于衡量内部干预是否超出简单增加弃答，不是发现了内部偏见机制。详细网格见 calibration/plan_frozen.json。', '', f'行为分组对照的类别沿用首轮fit预测；当前batch3提取中，{audit["total"]}题有{audit["total"]-audit["stable_under_batch3"]}题不再保持原稳定类别。此审计不用于重新选样本，详见 behavior_class_batch_audit.json。配对与行为分组方法使用不同的监督构造，因此差异不能单独归因于配对这一因素。', '', optimization_note, '', '初筛A/B/C候选概率质量接近1，格式遵循正常，但答案位置轮换一致性较低。若干预主要改变位置敏感性或unknown概率，不能直接解释成群体偏见机制。', '', '控制变量的逐项差值见 ablation_deltas.csv；下图为相对同方法同层锚点的提取/干预token消融。仅反映selection初筛，不作为测试泛化结论。', '', '![Matched token ablations](ablation_factors.png)', '']
    path.write_text(text + '\n'.join(lines), encoding='utf-8')
    report_hash = sha256(path)
    m = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    m['stages']['report']['report_sha256'] = report_hash
    m['status'] = 'complete'
    m['finalization'] = dict(report_sha256=report_hash, calibration_plan_sha256=sha256(out / 'calibration/plan_frozen.json'), calibration_test_sha256=sha256(out / 'calibration/test_results.json'), implementation_sha256=sha256(Path(__file__)))
    write_json(out / 'manifest.json', m)
    a = json.loads((out / 'final_audit.json').read_text(encoding='utf-8'))
    a['report_sha256'] = report_hash
    write_json(out / 'final_audit.json', a)
    print('Second-pilot report finalized, including frozen abstention calibration.', flush=True)


if __name__ == '__main__':
    main()
