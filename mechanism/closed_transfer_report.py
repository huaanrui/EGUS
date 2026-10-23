"""Rendering only: never selects policies or modifies model predictions."""
from pathlib import Path
import csv
import json


def report(out,data):
    out,data=Path(out),Path(data)
    results=json.loads((out/'frozen_results.json').read_text(encoding='utf-8'))
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    calibration=json.loads((out/'calibration_results.json').read_text(encoding='utf-8'))
    lines=['# BBQ 到指代任务的冻结迁移','',
           'Qwen3-1.7B，FP16，block index 18，prompt 最后 token，raw 选择题。源头、方向和主阈值冻结。',
           'SynthBias 为明确 unknown 的派生协议；不是原论文指标复现。WinoBias/Winogender 无 unknown。',
           'Correctness-steer 的源 dev 约束失败，仅作诊断。Unknown-offset 是输出分数操作。Direct 是路由动作，没有伪造概率。','',
           '## 样本与划分','',
           f"原始 SynthBias {manifest['synth_audit']['raw_rows']} 条，排除 {len(manifest['synth_audit']['excluded'])} 个异常或非标准配对家族。",
           '职业对与相同去词汇化模板合并连通分量。最大连通分量进入 test，剩余完整分量交替进入 calibration/reserve；未看模型结果时决定。',
           '主表只计排列 0；其他排列为绑定的鲁棒性诊断，不能视作独立问题。','']
    flat=[]
    def pct(x):return '—' if x is None else f'{100*x:.2f}'
    for key,ds in results['datasets'].items():
        lines.extend([f'## {key}','',f"主问题 {ds['primary_rows']} 条，含排列共 {ds['rows_with_permutations']} 条。候选字母概率质量均值 {ds['candidate_mass_mean']:.4f}。主预测为候选内 argmax，不等于实际自由生成。",'',
                      '|方法|准确率 %|平衡准确率 %|不足准确率 %|充分准确率 %|充分错误 unknown %|pro−anti pp|触发 %|位置翻转 %|',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for name,m in ds['methods'].items():
            lines.append('|'+ '|'.join([name,*[pct(m.get(k)) for k in ('accuracy','balanced_accuracy','ambig_accuracy','clear_accuracy','clear_false_unknown','pro_anti_gap','trigger_rate','position_entity_flip_rate')]])+'|')
            flat.append({'dataset':key,'method':name,**{k:v for k,v in m.items() if not isinstance(v,(dict,list))}})
        if 'recognition' in ds:
            r=ds['recognition'];lines.extend(['',f"证据头 AUROC {r['auroc']:.4f}，AUPRC {r['auprc']:.4f}；源阈值不足召回 {pct(r['tpr_at_source_threshold'])}%，充分假阳性 {pct(r['fpr_at_source_threshold'])}%。"])
            surface_path=out/'surface_controls/results.json'
            if surface_path.exists():
                surface=json.loads(surface_path.read_text(encoding='utf-8'))
                lines.extend(['',f"源域 BBQ fit 表面统计头 AUROC {surface['surface']['scores']['synth_test']['auroc']:.4f}，question-only TF-IDF AUROC {surface['question_tfidf']['scores']['synth_test']['auroc']:.4f}。两者不使用目标标签拟合，但只用每题一行 BBQ 且不含 MMLU，因此是诊断对照，不是完全匹配训练预算。"])
        if key=='winobias_test':
            lines.extend(['','|方法|Type 1 准确率 %|Type 2 准确率 %|','|---|---:|---:|'])
            for name,m in ds['methods'].items():lines.append(f"|{name}|{pct(m['task_types']['type1']['accuracy'])}|{pct(m['task_types']['type2']['accuracy'])}|")
        lines.extend(['','原始分条件、配对全对、概率及排列结果见 frozen_results.json。',''])
    lines.extend(['## 配对比较','',
                  'SynthBias 对比是平衡准确率，其他数据是总体准确率；单位 pp。家族聚类 bootstrap 2,000 次；主比较额外给聚类符号随机化检验及 Holm 校正。', '',
                  '|目标|A − B|差值 pp|95% CI pp|聚类数|Holm p|','|---|---|---:|---|---:|---:|'])
    for x in results['comparisons']:
        lo,hi=x['ci95'];p=x.get('p_holm')
        lines.append(f"|{x['dataset']}|{x['a']} − {x['b']}|{100*x['delta']:+.2f}|[{100*lo:+.2f}, {100*hi:+.2f}]|{x['clusters']}|{'探索' if p is None else f'{p:.4f}'}|")
    lines.extend(['','## 目标域阈值校准','',
                  '此表使用目标 calibration 金标；只改变阈值，源头/方向/强度固定。它是有监督适配结果，不能称完全冻结迁移。', '',
                  '|校准条数|方法|阈值|校准充分触发 %|测试 BA %|测试充分准确率 %|测试充分错误 unknown %|',
                  '|---:|---|---:|---:|---:|---:|---:|'])
    for x in calibration['records']:
        s=x['selection'];m=x['test']
        lines.append(f"|{x['budget']}|{x['method']}|{s['spec']['threshold']:.9g}|{pct(s['gate_fpr'])}|{pct(m['balanced_accuracy'])}|{pct(m['clear_accuracy'])}|{pct(m['clear_false_unknown'])}|")
    lines.extend(['','校准 2% 约束只约束小校准集的点估计，不保证目标风险。完整候选和选择样本均保存。','',
                  '## 限制与审计','',
                  '- 只测 Qwen3-1.7B 和短选择题；开放短问答与 Bias-in-Bios 尚未纳入本次运行。',
                  '- SynthBias 使用性别—职业场景，不能外推所有社会偏见。标签异常家族在推理前排除并留清单。',
                  '- WinoBias Type 1 不等于明确证据不足；保留原金标并单报，不将其一概当作充分/不足分类真值。',
                  '- 校准和测试池的巨大连通分量划分不构成自然分布随机样本；披露覆盖与聚类区间。',
                  '- 公开数据可能进入模型预训练；新目标评价不等于已排除预训练污染。',
                  '- 源 BBQ 历史成绩已被观察；未在目标测试上重选层、方向、强度或方法。',
                  '- 逐题预测、原始/干预真实前向、哈希及完整性审计分别保存。',''])
    stable_path=out/'stable_calibration/results.json'
    if stable_path.exists():
        stable=json.loads(stable_path.read_text(encoding='utf-8'))
        lines.extend(['## 预先规定的稳定分数校准','',
                      '主表继续保留旧 float32 决策。以下使用 float64 margin/稳定组合分数，仍使用目标校准标签；没有从数值版本中按测试表现选优。', '',
                      '|校准条数|方法|所选稳定分数阈值|测试 BA %|充分准确率 %|充分错误 unknown %|',
                      '|---:|---|---:|---:|---:|---:|'])
        for x in stable['records']:
            m=x['test'];lines.append(f"|{x['budget']}|{x['method']}|{x['selection']['spec']['threshold']:.9g}|{pct(m['balanced_accuracy'])}|{pct(m['clear_accuracy'])}|{pct(m['clear_false_unknown'])}|")
        lines.extend(['','margin 与概率阈值的单位不同，不能直接比较数值大小。饱和及并列计数见 stable_calibration/results.json。',''])
    audit_path=out/'independent_audit.json'
    corrected_path=out/'stable_calibration_v2/results.json'
    if corrected_path.exists():
        corrected=json.loads(corrected_path.read_text(encoding='utf-8'))
        lines.extend(['## 稳定校准 v2：候选阈值单位修正','',
                      '观察 v1 后发现其复用概率阈值常数1.0，对无界 margin 并非 guaranteed no-op。v2 改为实测分数边界加最大有限数 no-op。此修正属于探索性实现修正，旧结果保留，主冻结迁移不变。','',
                      '|校准条数|方法|测试 BA %|充分准确率 %|充分错误 unknown %|','|---:|---|---:|---:|---:|'])
        for x in corrected['records']:
            m=x['test'];lines.append(f"|{x['budget']}|{x['method']}|{pct(m['balanced_accuracy'])}|{pct(m['clear_accuracy'])}|{pct(m['clear_false_unknown'])}|")
        lines.append('')
    if audit_path.exists():
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        lines.extend([f"独立审计通过：重算冻结预测 {audit['frozen_predictions_recomputed']} 条、原数值校准预测 {audit['calibrated_predictions_recomputed']} 条、稳定数值校准预测 {audit['stable_calibrated_predictions_recomputed']} 条；验证全部候选阈值选择及输入/源模型哈希。",''])
        lines.extend([f"另重算稳定修正 v2 预测 {audit.get('stable_v2_calibrated_predictions_recomputed',0)} 条。",''])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    fields=sorted({k for r in flat for k in r})
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    plot(out,results)


def plot(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5),layout='constrained')
    names=['baseline','fairsteer','evidence_steer','evidence_direct','same_gate_direct']
    labels=['Base','FairSteer','Evidence\nsteer','Evidence\ndirect','Same gate\ndirect']
    x=np.arange(len(names));m=results['datasets']['synth_test']['methods']
    for shift,key,label,color in [(-.23,'ambig_accuracy','Insufficient: correct unknown','#2b7a78'),
                                   (0.,'clear_accuracy','Sufficient: correct entity','#5979b0'),
                                   (.23,'balanced_accuracy','Balanced accuracy','#d08a45')]:
        bars=axes[0].bar(x+shift,[100*m[n][key] for n in names],.22,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,110),ylabel='Accuracy (%)',title='BBQ → SynthBias: frozen transfer')
    axes[0].legend(loc='upper left',fontsize=8);axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True)
    methods=['baseline','fairsteer','evidence_steer'];x=np.arange(3)
    for shift,key,label,color in [(-.18,'winobias_test','WinoBias','#5979b0'),(.18,'winogender_test','Winogender','#2b7a78')]:
        bars=axes[1].bar(x+shift,[100*results['datasets'][key]['methods'][n]['accuracy'] for n in methods],.34,label=label,color=color)
        axes[1].bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
    axes[1].set(xticks=x,xticklabels=['Base','FairSteer','Evidence steer'],ylim=(0,110),ylabel='Accuracy (%)',title='Answer required: no unknown option')
    axes[1].legend(loc='upper left');axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
    fig.suptitle('Qwen3-1.7B · raw choice prompts · source-frozen parameters',fontsize=12)
    fig.savefig(out/'closed_transfer.png',dpi=180);fig.savefig(out/'closed_transfer.pdf');plt.close(fig)

