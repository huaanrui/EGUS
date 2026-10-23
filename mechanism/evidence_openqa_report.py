"""Unfrozen report rendering for frozen evidence-openQA computational artifacts."""
from collections import Counter, defaultdict
import csv
import json

import numpy as np

from .fairsteer_author import read, rows, write
from .evidence_openqa import ABSTENTION, digest, evaluate_response


def aggregate(values, evaluations):
    y=np.array([r['insufficient'] for r in values]);a=np.array([e['abstain'] for e in evaluations],float)
    f=np.array([e['f1'] for e in evaluations]);em=np.array([e['exact_match'] for e in evaluations])
    unresolved=np.array([e['parse_status'] in ('unresolved','uncertain_unresolved') for e in evaluations])
    return {'rows':len(values),'exact_match':float(em.mean()),'f1':float(f.mean()),
        'sufficient_f1':float(f[y==0].mean()),'sufficient_exact_match':float(em[y==0].mean()),
        'sufficient_false_abstention':float(a[y==0].mean()),'insufficient_strict_abstention':float(a[y==1].mean()),
        'insufficient_non_abstention':float(1-a[y==1].mean()),'coverage':float(1-a.mean()),
        'parse_status':dict(Counter(e['parse_status'] for e in evaluations)),
        'unresolved_fraction':float(unresolved.mean()),
        'bbq_accuracy_upper_if_all_unresolved_correct':float(np.mean(np.maximum(em,unresolved))) if values[0]['dataset']=='bbq' else None,
        'stereotyped_exact_entity_rate_insufficient':float(np.mean([e['stereotyped']==1 for r,e in zip(values,evaluations) if r['insufficient']])) if values[0]['dataset']=='bbq' else None}


def bootstrap_delta(values, baseline, edited, seed, n):
    groups=defaultdict(list)
    for i,r in enumerate(values):groups[r['group']].append(i)
    keys=sorted(groups);y=np.array([r['insufficient'] for r in values])
    metric={}
    for name in ('exact_match','f1','abstain'):
        delta=np.array([b[name]-a[name] for a,b in zip(baseline,edited)],float)
        masks={'all':np.ones(len(values),bool)} if name!='abstain' else {}
        if name=='f1':masks['sufficient']=y==0
        if name=='abstain':masks={'sufficient':y==0,'insufficient':y==1}
        for subgroup,mask in masks.items():
            sums=np.array([delta[groups[k]][mask[groups[k]]].sum() for k in keys])
            counts=np.array([mask[groups[k]].sum() for k in keys])
            rng=np.random.default_rng(seed);boots=[]
            for _ in range(n):
                selected=rng.integers(0,len(keys),len(keys));total=counts[selected].sum()
                if total:boots.append(float(sums[selected].sum()/total))
            metric[name+'_'+subgroup]={'delta':float(delta[mask].mean()),
                    'ci95':np.quantile(boots,[.025,.975]).tolist(),'clusters':len(keys),'replicates':n}
    return metric


def build_report(experiment):
    ex=experiment;out=ex.out;selection=read(out/'selection_frozen.json')['selections']
    recognition=read(out/'recognition.json');result={'datasets':{},'recognition':recognition,
        'limitations':['Exploratory: BBQ previously observed; SQuAD evaluation uses official public dev.',
            'No target BBQ threshold fitting; source constraint does not guarantee target utility.',
            'Generation scores use conservative automatic parsing, not completed human semantic judgments.',
            'Fixed abstention routes do not demonstrate improved unrestricted generation or causal bias mechanisms.',
            'SQuAD pretrained contamination is possible; task requires support from supplied context.',
            'Question matching controls context length, not every question-writing or semantic confound.']}
    review={};mapping=[];csvrows=[]
    for key in ('squad_test','bbq_test'):
        values=ex.datasets[key];generated=rows(out/f'generations_{key}.jsonl')
        assert len(generated)==len(values)
        probability=np.load(out/f'probabilities_{key}.npz')
        methods={'baseline':[g['response'] for g in generated]}
        triggers={}
        for name,prob_name,threshold in [('old_bbq_head','old_bbq_chat',ex.c['old_threshold']),
                ('squad_head','hidden_squad',selection['hidden_squad']['threshold'])]:
            fire=probability[prob_name]>threshold;triggers[name]=fire
            methods[name]=[ABSTENTION if gate else g['response'] for gate,g in zip(fire,generated)]
        scored={name:[evaluate_response(r,t) for r,t in zip(values,texts)] for name,texts in methods.items()}
        item={'methods':{},'generation':{'rows':len(generated),'eos':sum(g['eos'] for g in generated),
            'truncated':sum(not g['eos'] for g in generated),'tokens':sum(g['new_tokens'] for g in generated),
            'empty':sum(not g['response'].strip() for g in generated)},
            'clusters':len({r['group'] for r in values})}
        prediction_records=[]
        for name,texts in methods.items():
            a=aggregate(values,scored[name]);a['trigger_rate']=float(triggers[name].mean()) if name in triggers else 0.
            if name!='baseline':a['vs_baseline']=bootstrap_delta(values,scored['baseline'],scored[name],ex.c['seed'],ex.c['bootstrap_replicates'])
            if key.startswith('bbq'):
                a['categories']={cat:aggregate([r for r in values if r['category']==cat],
                    [e for r,e in zip(values,scored[name]) if r['category']==cat]) for cat in sorted({r['category'] for r in values})}
            item['methods'][name]=a
            csvrows.append({'dataset':key,'method':name,**{k:v for k,v in a.items() if not isinstance(v,(dict,list))}})
            for r,text,e in zip(values,texts,scored[name]):
                rid=digest([key,r['uid'],text])[:20]
                review.setdefault(rid,{'review_id':rid,'dataset':r['dataset'],'context':r['context'],
                    'question':r['question'],'response':text,
                    'annotation_fields':['supported_correct_answer','appropriate_abstention','unsupported_attribution','mixed_guess','other']})
                mapping.append({'dataset':key,'method':name,'uid':r['uid'],'review_id':rid,'automatic_evaluation':e})
                prediction_records.append({'uid':r['uid'],'method':name,'response':text,'evaluation':e})
        path=out/f'policy_predictions_{key}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prediction_records),encoding='utf-8')
        result['datasets'][key]=item
    write(out/'results.json',result)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    write(out/'review_key.json',mapping)
    order=sorted(review,key=lambda rid:digest([ex.c['seed'],rid]))
    (out/'anonymous_review.jsonl').write_text(''.join(json.dumps(review[rid],ensure_ascii=False)+'\n' for rid in order),encoding='utf-8')
    lines=['# 证据头与无选项开放问答迁移：第一轮结果','',
        '本报告采用保守自动评分，人工语义审查尚未完成。旧 BBQ 头与新 SQuAD 头分别报告；新头无 BBQ 拟合/选阈值。',
        '',f"SQuAD 头阈值：{selection['hidden_squad']['threshold']}，仅在 SQuAD dev 选择；旧头固定 .25。",'',
        '## 证据识别','',
        '|数据|方法|AUROC|balanced acc/.5|不足召回/选定阈值|充分误弃答/选定阈值|',
        '|---|---|---:|---:|---:|---:|']
    for key in ('squad_test','bbq_test'):
        for name,r in recognition[key].items():
            a=r['selected'];lines.append(f"|{key}|{name}|{a['auroc']:.4f}|{r['fixed_05']['balanced_accuracy']:.4f}|{a['insufficient_recall']:.2%}|{a['sufficient_false_abstention']:.2%}|")
    lines+=['','## 短答案生成与固定弃答路由','',
        'BBQ 正确率是精确实体/明确弃答匹配的保守值，不能当作人工核验后的语义准确率；不足样本的未明确弃答也不等于已经确认存在有害偏见。',
        '', '|数据|方法|EM/保守正确率|充分 F1|充分误弃答|不足明确弃答|未解析比例|',
        '|---|---|---:|---:|---:|---:|---:|']
    for key,item in result['datasets'].items():
        for name,r in item['methods'].items():
            lines.append(f"|{key}|{name}|{r['exact_match']:.2%}|{r['sufficient_f1']:.2%}|{r['sufficient_false_abstention']:.2%}|{r['insufficient_strict_abstention']:.2%}|{r['unresolved_fraction']:.2%}|")
    lines+=['','## 主要比较：SQuAD 头路由减原模型','']
    for key,item in result['datasets'].items():
        delta=item['methods']['squad_head']['vs_baseline']
        for label in ('exact_match_all','f1_sufficient','abstain_insufficient','abstain_sufficient'):
            d=delta[label];lo,hi=d['ci95']
            lines.append(f"- {key} {label}: {100*d['delta']:+.2f}pp，聚类95%区间 [{100*lo:+.2f},{100*hi:+.2f}]，{d['clusters']} 个文章/家族。")
    lines+=['','## 数据与生成完整性','',json.dumps(ex.data_manifest['summary'],ensure_ascii=False,indent=2),'']
    for key,item in result['datasets'].items():lines.append(f"- {key}: {json.dumps(item['generation'])}")
    lines+=['','## 解释边界','']+[f'- {x}' for x in result['limitations']]
    lines+=['','匿名待审查输出：anonymous_review.jsonl；方法映射：review_key.json；完整逐输出预测：policy_predictions_*.jsonl。',
        '源数据：https://rajpurkar.github.io/SQuAD-explorer/ 。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ex.log('report_complete',review_outputs=len(review))
