"""Post-hoc summaries and numerical audit of the frozen improvement pilot."""
from __future__ import annotations

import csv
import json

import numpy as np

from .fairsteer_author import ROOT, metrics, read, sha, write


def head_diagnostics(pilot):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    scores,x=pilot.base('dev');pred=scores.argmax(1);examples=pilot.examples('dev');probs=pilot.head_probabilities(x)
    diagnostics={}
    for name in ('original','expanded','evidence'):
        labels=[]
        for (row,_,_),p in zip(examples,pred):
            if name=='original':
                labels.append(1 if p==row['label'] else 0 if row['target_loc'] is not None and p==row['target_loc'] else -1)
            elif name=='expanded':labels.append(int(p==row['label']))
            else:labels.append(int(row['context_condition']=='ambig'))
        y=np.array(labels);valid=y>=0;z=(probs[name]>.5).astype(int);yy=y[valid]
        item={'rows':int(valid.sum()),'accuracy':float(np.mean(yy==z[valid])),'balanced_accuracy':float(balanced_accuracy_score(yy,z[valid])),'auc':float(roc_auc_score(yy,probs[name][valid])),'strata':{}}
        for condition in ('ambig','disambig'):
            mask=valid&np.array([r['context_condition']==condition for r,_,_ in examples]);neg=mask&(y==0);pos=mask&(y==1)
            item['strata'][condition]={'rows':int(mask.sum()),'class0_recall':float(np.mean(z[neg]==0)) if neg.any() else None,'class1_false_alarm':float(np.mean(z[pos]==0)) if pos.any() else None}
        diagnostics[name]=item
    write(pilot.out/'dev_head_diagnostics.json',diagnostics)
    return diagnostics


def cluster_intervals(pilot,pred,reference):
    examples=pilot.examples('test');labels=np.array([r['label'] for r,_,_ in examples]);categories=sorted({r['category'] for r,_,_ in examples})
    diff=(np.array(pred)==labels).astype(float)-(np.array(reference)==labels).astype(float)
    rng=np.random.RandomState(pilot.c['seed']);rep=pilot.c['bootstrap_replicates'];sums={k:np.zeros(rep) for k in ('overall','ambig','disambig')};counts={k:np.zeros(rep) for k in sums}
    for cat in categories:
        groups=sorted({r['family'] for r,_,_ in examples if r['category']==cat});draw=rng.randint(len(groups),size=(rep,len(groups)))
        for condition in sums:
            values=[];sizes=[]
            for g in groups:
                idx=[i for i,(r,_,_) in enumerate(examples) if r['family']==g and (condition=='overall' or r['context_condition']==condition)]
                values.append(diff[idx].sum());sizes.append(len(idx))
            sums[condition]+=np.array(values)[draw].sum(1);counts[condition]+=np.array(sizes)[draw].sum(1)
    return {k:np.quantile(100*sums[k]/counts[k],[.025,.975]).tolist() for k in sums}


def audit(pilot,records):
    state=read(pilot.out/'manifest.json');plan=read(pilot.out/'test_plan_frozen.json')
    for name,h in state['fingerprint']['sources'].items():assert sha(ROOT/name)==h and sha(pilot.out/'source_snapshot'/name)==h
    assert plan['selection_sha256']==sha(pilot.out/'selection_results.json')
    for name,h in plan['fitted_sha256'].items():assert sha(pilot.out/name)==h
    assert [r['bbq']['spec'] for r in records]==plan['configs']
    direction=read(pilot.out/'direction_examples.json');optimization=read(pilot.out/'optimization.json')
    fit=set(pilot.splits['fit']);assert set(direction['indices'])<=fit and set(optimization['bbq_indices'])<=fit
    assert set(optimization['neutral_indices'])<=set(pilot.msplits['fit'])
    for a in pilot.splits:
        for b in pilot.splits:
            if a!=b:assert not {pilot.source[i]['family'] for i in pilot.splits[a]}&{pilot.source[i]['family'] for i in pilot.splits[b]}
    maximum=0.;checked=0
    for record in records:
        for utility,key in ((False,'bbq'),(True,'utility')):
            saved=record[key];scores=np.array(saved['scores']);pred=np.array(saved['predictions']);examples=pilot.examples('test',utility)
            assert np.isfinite(scores).all() and np.array_equal(pred,scores.argmax(1))
            baseline,x=pilot.base('test',utility);actions=pilot.actions(saved['spec'],x);assert np.array_equal(actions,saved['actions'])
            expected=baseline.copy()
            for alpha in pilot.c['strengths']:
                use=actions==alpha
                if use.any():expected[use]=pilot.counterfactual('test',saved['spec']['vector'],alpha,utility)[use]
            assert np.array_equal(expected,scores)
            labels=np.array([r['answer'] if utility else r['label'] for r,_,_ in examples])
            accuracy=float(np.mean(pred==labels));maximum=max(maximum,abs(accuracy-saved['metrics']['accuracy']))
            assert abs(accuracy-saved['metrics']['accuracy'])<1e-12
            if not utility:
                for condition in ('ambig','disambig'):
                    ids=[i for i,(r,_,_) in enumerate(examples) if r['context_condition']==condition]
                    accuracy=float(np.mean(pred[ids]==labels[ids]));assert abs(accuracy-saved['metrics'][condition]['accuracy'])<1e-12
            checked+=1
    result={'passed':True,'files_checked':checked,'max_accuracy_error':maximum,'family_disjoint_refits':True,'no_test_labels_used_by_controller':True,'test_plan_frozen_before_evaluation':True,'limitation':'Historical BBQ aggregate results already observed; exploratory, not pristine confirmation.'}
    write(pilot.out/'independent_audit.json',result);return result


def report(pilot):
    if not (pilot.out/'results.json').exists():
        text='# FairSteer 改进预实验\n\n尚未完成全部评价。已完成阶段：'+', '.join(pilot.state['stages'])+'。\n'
        (pilot.out/'report.md').write_text(text,encoding='utf-8');return
    records=read(pilot.out/'results.json');checked=audit(pilot,records);heads=head_diagnostics(pilot)
    if (pilot.out/'calibration_policy_frozen.json').exists():
        from scripts.fairsteer_improve_calibration import run as calibration
        control=calibration(pilot)
        control_scores,_=pilot.base('test');control_scores=control_scores.copy()
        control_examples=pilot.examples('test')
        control_scores[np.arange(len(control_scores)),[r['unknown_index'] for r,_,_ in control_examples]]+=control['plan']['offset']
        assert np.array_equal(control_scores,control['bbq']['scores'])
        assert np.array_equal(control_scores.argmax(1),control['bbq']['predictions'])
        assert control['utility']==records[0]['utility']
        checked['output_calibration_prediction_audit']=True
        write(pilot.out/'independent_audit.json',checked)
        records.append(control)
    reference=next(r for r in records if r['name']=='fairsteer');baseline=records[0]
    plan=read(pilot.out/'test_plan_frozen.json');summary=[];examples=pilot.examples('test');n=len(pilot.splits['test'])
    labels={'baseline':'原模型','fairsteer':'FairSteer','tuned':'强度/阈值调优','expanded':'扩展错误标签','evidence':'证据感知门控','adaptive':'收益与强度预测','optimized':'优化方向','combined':'证据门控+优化方向','unknown_calibration':'unknown输出校准'}
    for record in records:
        m=record['bbq']['metrics'];pred=record['bbq']['predictions'];ci=cluster_intervals(pilot,pred,reference['bbq']['predictions'])
        failure=pilot.feasible(m,record['utility']['metrics'],baseline['bbq']['metrics'],reference['bbq']['metrics'],baseline['utility']['metrics'])
        dev_selection=plan['selected'].get(record['name'])
        if record['name']=='unknown_calibration':dev_selection={'feasible':not record['plan']['dev_failures'],'failures':record['plan']['dev_failures']}
        dev_failures=dev_selection['failures'] if dev_selection else []
        row={'name':record['name'],'label':labels[record['name']],'accuracy':m['accuracy'],'ambig':m['ambig']['accuracy'],'disambig':m['disambig']['accuracy'],'bias_ambig':m['ambig']['bias_score'],'bias_disambig':m['disambig']['bias_score'],'disambig_unknown':m['native_disambig_unknown_rate'],'utility':record['utility']['metrics']['accuracy'],'trigger_fraction':record['bbq']['trigger_fraction'],'delta_pp_vs_fairsteer':100*(m['accuracy']-reference['bbq']['metrics']['accuracy']),'paired_family_ci_pp':ci,'test_constraint_failures':failure,'dev_constraint_failures':dev_failures,'dev_selection':dev_selection,'selected_parameters':record['bbq']['spec'],'shots':{}}
        for shot,start in (('zeroshot',0),('fewshot',n)):
            row['shots'][shot]=metrics([r for r,_,_ in examples[start:start+n]],pred[start:start+n])
        summary.append(row)
    write(pilot.out/'summary.json',summary)
    with (pilot.out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=[k for k in summary[0] if k not in ('shots','dev_selection','paired_family_ci_pp')]);writer.writeheader()
        for row in summary:writer.writerow({k:row[k] for k in writer.fieldnames})
    from scripts.plot_fairsteer_improve import main as plot
    plot(pilot.out)
    lines=['# FairSteer 改进预实验：实际结果','',
           '本轮在 Qwen3-1.7B、FP16、单题、原始选项顺序、层索引18、最后prompt token上比较；作者 A. 提示，zero/few-shot等权。所有基线和改进重新拟合于同一训练家族。','',
           '历史全量 BBQ 已经评价过；本轮是新的家族隔离拟合与探索性评估，不是未接触过的独立确认。当前结果不直接与此前75.09%/82.14%的全量、训练评价重叠成绩比较。','',
           '检测器使用全部fit家族及去重MMLU训练问题；方向从fit抽11类各10题。控制器在另一个训练家族分区收集实际干预收益，不接触dev/test标签。训练方向的CE使用fit数据，普通任务保持使用四选项KL（不等同完整词表保持）。','',
           '## 数据和预算','',json.dumps(read(pilot.out/'data_summary.json'),ensure_ascii=False,indent=2),'',
           '## 固定开发选择后评价','',
           '|方法|整体准确率|相对FairSteer/pp（95%区间）|歧义准确率|消歧准确率|BS(a)×100|BS(d)×100|消歧弃答|MMLU子集|开发/评价约束|',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for row in summary:
        pct=lambda key:f"{100*row[key]:.2f}" if row[key] is not None else 'NA'
        lo,hi=row['paired_family_ci_pp']['overall'];delta=f"{row['delta_pp_vs_fairsteer']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        flags=('dev:'+','.join(row['dev_constraint_failures']) if row['dev_constraint_failures'] else 'dev通过')+'; '+('test:'+','.join(row['test_constraint_failures']) if row['test_constraint_failures'] else 'test通过')
        if row['name'] in ('baseline','fairsteer'):flags='固定对照; '+('test:'+','.join(row['test_constraint_failures']) if row['test_constraint_failures'] else 'test通过')
        lines.append('| '+' | '.join([row['label'],pct('accuracy'),delta,pct('ambig'),pct('disambig'),pct('bias_ambig'),pct('bias_disambig'),pct('disambig_unknown'),pct('utility'),flags])+' |')
    lines += ['', f"![相对FairSteer的配对区间]({(pilot.out/'improvement_results.png').as_posix()})", '', '区间为类别内模板家族配对bootstrap，zero/few-shot绑定；1000次。探索性多方法比较未作多重比较校正。小MMLU样本只能提供初步能力诊断。BS有符号，距零更近才表示方向偏差更小；弃答与偏好需结合阅读。','',
              '## 选择规则与解释限制','',
              '每个方法在dev按整体准确率最大化选择，同时要求消歧准确率≥max(原模型−1pp, FairSteer−0.5pp)、MMLU≥原模型−2pp、|BS(a)|≤|FairSteer BS(a)|+1pp。没有可行候选时仍展示最优探索候选，并标记dev不通过；不能称为确认改进。最终评价不改选参数。','',
              '优化方向与原方向共享训练数据池，但使用了额外CE优化；强度与阈值调优提供简单调参对照，学习控制器使用额外训练反事实前向。这些计算预算差异需要计入后续正式比较。优化同时改变方向和长度，尚未增加精确等范数原方向对照，因此不能将收益直接归因于方向几何改变。','',
              'unknown输出校准为评价开始前追加冻结的低成本对照：固定偏置作用于可识别的unknown选项，dev按相同约束选择，不读取推理样本真实答案或证据标签；MMLU不修改。独立策略及预测见calibration_policy_frozen.json、calibration_plan_frozen.json、calibration_result.json。','',
              '神经元/电路稀疏化尚未实施：本轮先判断门控、收益预测和方向优化是否有稳定收益，避免在效应未确定时扩展组件搜索。','',
              '开发集检测器诊断见 dev_head_diagnostics.json：original预测作者规则下的正确性，expanded预测全部题正确性，evidence预测证据不足；三个头的标签语义不同，不能直接用准确率排名。','',
              '自动审计：'+json.dumps(checked,ensure_ascii=False),'',
              '产物：results.json逐题预测；summary.json/summary.csv汇总；selection_results.json全部开发候选；test_plan_frozen.json预先冻结的方案；optimization.json训练轨迹。']
    (pilot.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pilot.out/'report.md',flush=True)
