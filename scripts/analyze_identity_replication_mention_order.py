"""Exploratory shortcut audit using public entity order and frozen predictions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import re,time
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha


def mention_order(r):
    positions={e:re.search(r'\b'+re.escape(e)+r'\b',r['context'],re.I).start() for e in r['entities']}
    return sorted(r['entities'],key=positions.get)


if __name__=='__main__':
    folder=ROOT/'results/identity_replication_20261015_v1';source=ROOT/'data/prepared/identity_combo_20261016_v1'
    write(folder/'mention_order_analysis_declaration.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
        'script_sha256':sha(Path(__file__)),'status':'exploratory audit after main behavioral results, no new method selection',
        'hypothesis':'native answer mention-order imbalance may make position-prior correction appear as debiasing; no causal attribution solely from this correlation'})
    datasets={k:rows(source/f'{k}.jsonl') for k in ('rank','screen','dev','historical','new_type1','winogender')}
    datasets['confirmation']=rows(ROOT/'data/prepared/identity_replication_20261015_v1/confirmation.jsonl')
    distribution={};models={}
    for split,values in datasets.items():
        second=[mention_order(r)[1]==r['gold'] for r in values]
        distribution[split]={'pairs':len(values)//2,'gold_second_mention_rate':float(np.mean(second)),
            'always_second_public_entity_heuristic_accuracy':float(np.mean(second)),
            'always_first_public_entity_heuristic_accuracy':1-float(np.mean(second))}
    for name in ('qwen3','minicpm5','qwen25'):
        models[name]={}
        for split in ('historical','new_type1','winogender','confirmation'):
            base={r['pair_id']:r for r in datasets[split]};by=defaultdict(list)
            for r in rows(folder/name/'predictions'/(split+'.jsonl')):by[r['method']].append(r)
            result={}
            for method,rr in by.items():
                order=[mention_order(base[r['pair_id']]) for r in rr]
                pred_second=[r['choices'][r['prediction']]==o[1] for r,o in zip(rr,order)]
                gold_second=[base[r['pair_id']]['gold']==o[1] for r,o in zip(rr,order)]
                correct=np.array([r['prediction']==r['label'] for r in rr]);g=np.array(gold_second)
                result[method]={'second_mention_prediction_rate':float(np.mean(pred_second)),
                    'accuracy_when_gold_first':float(correct[~g].mean()) if (~g).any() else None,
                    'accuracy_when_gold_second':float(correct[g].mean()) if g.any() else None,
                    'rows_gold_first':int((~g).sum()),'rows_gold_second':int(g.sum())}
            models[name][split]=result
    write(folder/'mention_order_analysis.json',{'dataset_distribution':distribution,'models':models,
        'limitation':'first/second mention is lexical order, not a syntactic dependency annotation; post-hoc descriptive audit, not unique causal explanation'})
    report=['# 答案提及位置：探索性基础对照','',
        '该审计在看到主行为结果后开展，只使用既有数据与冻结预测，不调参、不改变确认结果。提及次序不是句法主客体标注。','',
        '|数据|配对数|金答案为第二个提及实体|始终选第二实体准确率|','|---|---:|---:|---:|']
    for split,z in distribution.items():report.append(f"|{split}|{z['pairs']}|{z['gold_second_mention_rate']:.2%}|{z['always_second_public_entity_heuristic_accuracy']:.2%}|")
    report+=['','|模型/数据/方法|选择第二实体比例|金答案第一时准确率|金答案第二时准确率|','|---|---:|---:|---:|']
    for name,d in models.items():
        for split,z in d.items():
            for method in ('baseline','selected','fine_combo')+ (('old_combo','old_l7') if name=='qwen3' else ()):
                v=z[method];a='—' if v['accuracy_when_gold_first'] is None else f"{v['accuracy_when_gold_first']:.2%}"
                b='—' if v['accuracy_when_gold_second'] is None else f"{v['accuracy_when_gold_second']:.2%}"
                report.append(f"|{name}/{split}/{method}|{v['second_mention_prediction_rate']:.2%}|{a}|{b}|")
    report+=['','若第二实体启发式已覆盖大部分或全部金答案，则单个数据集涨点不能单独证明通用去偏。',
        '干预后位置偏好变化与收益相关只是候选解释；还需在答案位置均衡、证据保持有效的样本上作因果检验。','']
    p=ROOT/'research/2026-10-16_identity_replication_mention_order.md';p.write_text('\n'.join(report),encoding='utf-8')
    print(__import__('json').dumps(distribution,indent=2))
