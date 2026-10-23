"""Source-only continuous threshold supplement; no fitting or GPU computation."""
from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from mechanism.fairsteer_author import ROOT, read, rows, sha, write
from mechanism.evidence_openqa import ABSTENTION, evaluate_response, score_metrics
from mechanism.evidence_openqa_report import aggregate, bootstrap_delta

BASE=ROOT/'results/evidence_openqa_20261014_v1'
OUT=ROOT/'results/evidence_openqa_source_threshold_20261014_v2'
DATA=ROOT/'data/prepared/evidence_openqa_20261014_v1'
PROTOCOL=ROOT/'research/2026-10-14_evidence_openqa_threshold_protocol_v2.md'


def choose():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),ROOT/'mechanism/evidence_openqa.py',ROOT/'mechanism/evidence_openqa_report.py',PROTOCOL]
    inputs=[BASE/'squad_probe.npz',BASE/'features_squad_dev_chat.npz',DATA/'squad_dev.jsonl',BASE/'plan_frozen.json']
    identity={'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
              'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
              'max_source_false_abstention':.02,'seed':20261014,'bootstrap_replicates':1000}
    path=OUT/'plan_frozen.json'
    if path.exists():assert read(path)==identity
    else:
        write(path,identity)
        for p in sources:
            target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    head=np.load(BASE/'squad_probe.npz');dev=rows(DATA/'squad_dev.jsonl')
    x=np.load(BASE/'features_squad_dev_chat.npz')['features'].astype(np.float64)
    margins=x@head['coef'].astype(np.float64)+float(head['intercept']);y=np.array([r['insufficient'] for r in dev])
    candidates=[]
    for threshold in np.unique(margins):
        pred=margins>threshold
        candidates.append({'threshold_margin':float(threshold),'recall':float(pred[y==1].mean()),
                           'false_abstention':float(pred[y==0].mean())})
    best=max((r for r in candidates if r['false_abstention']<=.02),key=lambda r:(r['recall'],r['threshold_margin']))
    frozen={'selected':best,'candidates':candidates,'selection_labels':'SQuAD dev only',
            'target_labels_used':False,'source_auroc':float(roc_auc_score(y,margins)),
            'head_sha256':sha(BASE/'squad_probe.npz'),'source_margins':margins.tolist()}
    dest=OUT/'selection_frozen.json'
    if dest.exists():assert read(dest)==frozen
    else:write(dest,frozen)
    print(json.dumps({'selection':best}),flush=True)
    return head,best


def evaluate(head,best):
    state=read(BASE/'state.json');assert 'generate' in state['stages'],'Finish v1 generations before evaluation'
    for name,h in state['stages']['generate']['files'].items():assert sha(BASE/name)==h
    output={'selection':best,'datasets':{},'scope':'Exploratory source-only continuous threshold supplement; no target reselection.'}
    lines=['# SQuAD 源域连续阈值补充','',f"源域选定 margin>{best['threshold_margin']:.6f}；dev不足召回 {best['recall']:.2%}，充分误弃答 {best['false_abstention']:.2%}。",'',
        '|数据|策略|AUROC|不足召回/门控|充分误弃答/门控|EM/保守正确率|充分F1|不足明确弃答|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for key in ('squad_test','bbq_test'):
        values=rows(DATA/f'{key}.jsonl');generated=rows(BASE/f'generations_{key}.jsonl')
        features=np.load(BASE/f'features_{key}_chat.npz')
        assert features['uids'].tolist()==[r['uid'] for r in values]
        x=features['features'].astype(np.float64);margins=x@head['coef'].astype(np.float64)+float(head['intercept'])
        y=np.array([r['insufficient'] for r in values]);base=[evaluate_response(r,g['response']) for r,g in zip(values,generated)]
        item={};records=[]
        for method,t in [('source_continuous',best['threshold_margin']),('fixed_margin_zero',0.)]:
            fire=margins>t;responses=[ABSTENTION if f else g['response'] for f,g in zip(fire,generated)]
            scored=[evaluate_response(r,text) for r,text in zip(values,responses)]
            summary=aggregate(values,scored)
            recognition={'auroc':float(roc_auc_score(y,margins)),'insufficient_recall':float(fire[y==1].mean()),
                         'sufficient_false_abstention':float(fire[y==0].mean()),'threshold_margin':float(t)}
            summary['recognition']=recognition;summary['vs_baseline']=bootstrap_delta(values,base,scored,20261014,1000)
            item[method]=summary
            for r,margin,response,e in zip(values,margins,responses,scored):
                records.append({'uid':r['uid'],'method':method,'margin':float(margin),'response':response,'evaluation':e})
            lines.append(f"|{key}|{method}|{recognition['auroc']:.4f}|{recognition['insufficient_recall']:.2%}|{recognition['sufficient_false_abstention']:.2%}|{summary['exact_match']:.2%}|{summary['sufficient_f1']:.2%}|{summary['insufficient_strict_abstention']:.2%}|")
        output['datasets'][key]=item
        (OUT/f'policy_predictions_{key}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        np.savez(OUT/f'margins_{key}.npz',margins=margins,uids=features['uids'])
    lines+=['','主策略为 source_continuous；fixed_margin_zero 仅诊断，不能按目标结果改选。',
        'v1粗网格选择了阈值1，不触发。本补充保留原拟合，按SQuAD dev实际margin选阈值。',
        '生成是固定弃答路由；BBQ语义未解析项需人工审查。目标误伤不能由源域2%约束排除。',
        '已观察v1测试AUROC，此补充为探索性分析。']
    write(OUT/'results.json',output);(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'completion.json',{'complete':True,'selection_frozen_sha256':sha(OUT/'selection_frozen.json'),
        'input_generations':{name:sha(BASE/name) for name in ('generations_squad_test.jsonl','generations_bbq_test.jsonl')},
        'source_threshold':best,'generated_new_tokens':0})
    print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__=='__main__':
    head,best=choose()
    if '--select-only' not in sys.argv:evaluate(head,best)
