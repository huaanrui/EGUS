"""Audit and summarize per-model dev-selected task winners."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from mechanism.identity_best_combo import Search,score_order,signature
from mechanism.identity_causal import comparison


def description(spec,sites,params):
    if not spec['edits']:return '无干预'
    out=[]
    for e in spec['edits']:
        s=sites[e['site']];loc={'identity':'上下文代词','question_identity':'问题代词','decision':'答案位置'}[s['position']]
        label=f"L{s['layer']} MLP" if s['module']=='mlp' else f"L{s['layer']} H{s['head']}"
        count=len(params[e['site']+'_mean']) if e['group']=='whole' else len(params[e['site']+'_'+e['group']])
        out.append(f"{label}/{loc}/{e['group']}({count}维)/{e['mode']} λ={e['amount']:g}")
    return '；'.join(out)


if __name__=='__main__':
    folder=ROOT/'results/identity_best_combo_20261016_v1';result={}
    report=['# 各模型任务内最高分组件组合：探索性搜索','',
        '主目标为开发集单题准确率，并列时比较成对正确率；模型和组件参数在目标评测前冻结。',
        '全部层MLP和注意力头先做rank归因，实际筛选包含坐标子集、全组件、强度细化、至多三个位置的beam组合。',
        '最高分指本轮有限搜索空间内的开发集赢家，不保证全局最优；另报五项能力任务下降均不超过2pp的赢家。',
        '沿用rank64、screen32、dev49、历史Type2 146对。各阶段均男女×AB/BA；历史目标已观察，不是新封存确认。',
        '原Type2金答案高度集中在第二职业，本轮分数优化不能单独当作通用去偏或身份专属机制证据。层号和头号从0开始。','',
        '|模型/方案|开发准确率|开发成对均对|能力约束|历史Type2准确率|历史Type2成对均对|','|---|---:|---:|---|---:|---:|']
    for name in ('qwen3','minicpm5','qwen25'):
        if name!='qwen3':
            report+=['|模型/方案|开发准确率|开发成对均对|能力约束|历史Type2准确率|历史Type2成对均对|','|---|---:|---:|---|---:|---:|']
        e=Search(name);e.prepare();e.audit_search();s=read(e.out/'selection_frozen.json');ev=read(e.out/'evaluation.json');sites=e.sites
        by=defaultdict(list)
        for r in rows(e.out/'test_predictions.jsonl'):by[r['method']].append(r)
        dev=read(e.out/'dev_results.json');by_sig={signature(z['spec']):z for z in dev}
        comparisons={}
        for method in ('baseline','best_task','best_constrained','ref_selected','ref_fine_combo','ref_norm_whole')+(('ref_old_combo','ref_old_norm_whole') if name=='qwen3' else ()):
            spec=ev['methods'][method];sig=signature(spec);d=by_sig[sig];m=ev['metrics'][method]
            report.append(f"|{name}/{method}|{d['metrics']['accuracy']:.2%}|{d['metrics']['both_correct']:.2%}|{'通过' if d['feasible'] else '未通过'}|{m['accuracy']:.2%}|{m['both_correct']:.2%}|")
        for winner in ('best_task','best_constrained'):
            vv=by[winner];lookup={r['uid']:r for r in by['baseline']}
            comparisons[winner]={}
            for reference in ('baseline','ref_selected')+(('ref_old_combo',) if name=='qwen3' else ()):
                rr={r['uid']:r for r in by[reference]}
                comparisons[winner][reference]={f:comparison(vv,[r['prediction'] for r in vv],[rr[r['uid']]['prediction'] for r in vv],f,4000,20261016) for f in ('accuracy','both_correct')}
        result[name]={'selection':s,'evaluation':ev['metrics'],'comparisons':comparisons}
        report+=['',f"{name} 搜索量：{s['candidate_counts']}。",
            '最高分方案：'+description(s['best_task']['spec'],sites,e.params)+'。',
            '满足能力约束方案：'+description(s['best_constrained']['spec'],sites,e.params)+'。','']
        for winner in ('best_task','best_constrained'):
            for reference,z in comparisons[winner].items():
                v=z['accuracy'];report.append(f"{name}/{winner} 相对 {reference}：准确率{100*v['delta']:+.2f}pp，95%家族区间[{100*v['ci95'][0]:+.2f},{100*v['ci95'][1]:+.2f}]pp。")
                report.append('')
        report+=['','|开发能力任务|原模型|最高分方案|受约束方案|','|---|---:|---:|---:|']
        baseline=next(z for z in dev if not z['spec']['edits'])
        for task in baseline['utilities']:
            report.append(f"|{task}|{baseline['utilities'][task]['accuracy']:.2%}|{s['best_task']['utilities'][task]['accuracy']:.2%}|{s['best_constrained']['utilities'][task]['accuracy']:.2%}|")
        report+=['']
    report+=['区间是探索性、未经多重比较校正的家族bootstrap；开发最佳与测试最高不是同一概念，未据测试重新选参。',
        '已有参考方案同批次指标复现、所有冻结输入指纹、开发选择重放和逐条预测指标复核均通过。','',
        '干预算子：center为u′=μ+(1−λ)(u−μ)，raw为u′=(1−λ)u。λ>1会反转偏离均值的方向；raw λ=1.5是原激活乘−0.5，不能仅称置零。',
        'Qwen2.5反向数值修正采用1024/256/64/16/4/1中的最大有限损失缩放，并除回实际缩放；前向结果不变，无样本跳过。原始实现和其他两模型冻结选择保留，详见numerical_v2_declaration.json。',
        '本轮159项工程测试通过。独立复核的梯度有限性、坐标排名重建及逐条参考预测一致性见independent_verification.json。','']
    write(folder/'summary.json',{'models':result,'status':'exploratory task-accuracy search with frozen dev selection'})
    path=ROOT/'research/2026-10-16_identity_best_combo_findings.md';path.write_text('\n'.join(report),encoding='utf-8');print(path)
