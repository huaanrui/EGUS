"""Read-only completion audit, separate from fitted experiment code."""
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(line) for line in p.read_text(encoding='utf-8').split('\n') if line.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize(text):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',re.sub(r'[^\w\s]',' ',text.casefold())).split())


def audit(folder):
    out=Path(folder);plan=read(out/'plan_frozen.json')
    import tomllib
    config=tomllib.loads((out/'source_snapshot/config.toml').read_text(encoding='utf-8'))
    data=ROOT/config['data_dir'];manifest=read(data/'manifest.json');state=read(out/'state.json')
    for name,h in manifest['files'].items():assert sha(data/name)==h
    for name,h in plan['identity']['sources'].items():assert sha(ROOT/name)==h
    for stage in ('smoke','features','fit','select','recognition','generate'):
        assert stage in state['stages']
        for name,h in state['stages'][stage]['files'].items():assert sha(out/name)==h
    datasets={k:rows(data/f'{k}.jsonl') for k in manifest['keys']}
    for key,values in datasets.items():
        assert len({r['uid'] for r in values})==len(values)
        if key.startswith('squad'):
            pairs={}
            for r in values:pairs.setdefault(r['pair_id'],[]).append(r)
            for pair in pairs.values():
                assert len(pair)==2 and {r['insufficient'] for r in pair}=={0,1}
                assert pair[0]['context']==pair[1]['context']
                assert all(bool(r['answers'])==(r['insufficient']==0) for r in pair)
    source=[datasets['squad_'+k] for k in ('fit','dev','test')]
    for i,a in enumerate(source):
        for b in source[i+1:]:
            assert not {r['article'] for r in a}&{r['article'] for r in b}
            assert not {normalize(r['context']) for r in a}&{normalize(r['context']) for r in b}
            assert not {normalize(r['question']) for r in a}&{normalize(r['question']) for r in b}
    assert not {r['group'] for r in datasets['bbq_dev']}&{r['group'] for r in datasets['bbq_test']}
    selection=read(out/'selection_frozen.json');assert not selection['target_bbq_labels_used']
    assert selection['fit_sha256']==sha(out/'models.joblib')
    devprob=np.load(out/'probabilities_squad_dev.npz')['hidden_squad']
    y=np.array([r['insufficient'] for r in datasets['squad_dev']])
    candidates=[]
    for threshold in config['thresholds']:
        pred=devprob>threshold;fpr=pred[y==0].mean();tpr=pred[y==1].mean()
        if fpr<=config['max_false_abstention']:candidates.append((tpr,threshold))
    assert max(candidates)[1]==selection['selections']['hidden_squad']['threshold']
    probe=np.load(out/'squad_probe.npz');recognition=read(out/'recognition.json');result=read(out/'results.json')
    predictions=0;generated=0;probability_error=0.
    for key in ('squad_dev','squad_test','bbq_dev','bbq_test'):
        values=datasets[key];saved=np.load(out/f'probabilities_{key}.npz');features=np.load(out/f'features_{key}_chat.npz')
        assert features['uids'].tolist()==[r['uid'] for r in values]
        expected=expit(features['features']@probe['coef']+probe['intercept'])
        err=float(np.max(np.abs(expected-saved['hidden_squad'])));probability_error=max(probability_error,err)
        assert err<1e-5,err
        for name in saved.files:
            auc=roc_auc_score([r['insufficient'] for r in values],saved[name])
            assert abs(auc-recognition[key][name]['selected']['auroc'])<1e-12
    for key in ('squad_test','bbq_test'):
        values=datasets[key];base=rows(out/f'generations_{key}.jsonl');generated+=len(base)
        records=rows(out/f'policy_predictions_{key}.jsonl');predictions+=len(records)
        probs=np.load(out/f'probabilities_{key}.npz')
        for method,name,t in [('old_bbq_head','old_bbq_chat',config['old_threshold']),
                             ('squad_head','hidden_squad',selection['selections']['hidden_squad']['threshold'])]:
            part=[r for r in records if r['method']==method]
            assert len(part)==len(values)
            for i,r in enumerate(part):
                assert r['uid']==values[i]['uid']
                assert r['response']==('Insufficient information.' if probs[name][i]>t else base[i]['response'])
        for method in ('baseline','old_bbq_head','squad_head'):
            part=[r for r in records if r['method']==method]
            mean=float(np.mean([r['evaluation']['exact_match'] for r in part]))
            assert abs(mean-result['datasets'][key]['methods'][method]['exact_match'])<1e-12
    record={'passed':True,'generated_rows':generated,'policy_prediction_rows':predictions,
            'source_target_partition_checked':True,'paired_contexts_exact':True,'source_only_selection_recomputed':True,
            'max_probe_export_probability_error':probability_error,'recognition_auroc_recomputed':True,
            'fixed_gate_routes_recomputed':True,'frozen_source_and_artifact_hashes_checked':True}
    (out/'completion_audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


def audit_supplement():
    out=ROOT/'results/evidence_openqa_source_threshold_20261014_v2'
    base=ROOT/'results/evidence_openqa_20261014_v1'
    data=ROOT/'data/prepared/evidence_openqa_20261014_v1'
    plan=read(out/'plan_frozen.json');selection=read(out/'selection_frozen.json')
    for name,h in {**plan['source_hashes'],**plan['input_hashes']}.items():assert sha(ROOT/name)==h
    probe=np.load(base/'squad_probe.npz');dev=rows(data/'squad_dev.jsonl')
    x=np.load(base/'features_squad_dev_chat.npz')['features'].astype(np.float64)
    margin=x@probe['coef'].astype(np.float64)+float(probe['intercept'])
    y=np.array([r['insufficient'] for r in dev]);eligible=[]
    for threshold in np.unique(margin):
        fire=margin>threshold
        if fire[y==0].mean()<=.02:eligible.append((float(fire[y==1].mean()),float(threshold)))
    best=max(eligible)
    assert best[0]==selection['selected']['recall'] and best[1]==selection['selected']['threshold_margin']
    assert not selection['target_labels_used']
    result=read(out/'results.json');count=0;anonymous={};key_records=[]
    for key in ('squad_test','bbq_test'):
        values=rows(data/f'{key}.jsonl');base_records=rows(base/f'generations_{key}.jsonl')
        policy=rows(out/f'policy_predictions_{key}.jsonl');count+=len(policy)
        x=np.load(base/f'features_{key}_chat.npz')['features'].astype(np.float64)
        m=x@probe['coef'].astype(np.float64)+float(probe['intercept']);y=np.array([r['insufficient'] for r in values])
        for method,t in [('source_continuous',best[1]),('fixed_margin_zero',0.)]:
            part=[r for r in policy if r['method']==method];assert len(part)==len(values)
            fire=m>t;rec=result['datasets'][key][method]['recognition']
            assert abs(float(fire[y==0].mean())-rec['sufficient_false_abstention'])<1e-12
            assert abs(float(fire[y==1].mean())-rec['insufficient_recall'])<1e-12
            assert abs(float(roc_auc_score(y,m))-rec['auroc'])<1e-12
            for i,r in enumerate(part):
                assert r['uid']==values[i]['uid']
                assert r['response']==('Insufficient information.' if fire[i] else base_records[i]['response'])
                rid=hashlib.sha256((key+r['uid']+r['response']).encode()).hexdigest()[:20]
                anonymous.setdefault(rid,{'review_id':rid,'context':values[i]['context'],
                    'question':values[i]['question'],'response':r['response']})
                key_records.append({'review_id':rid,'dataset':key,'method':method,'uid':r['uid']})
    (out/'anonymous_review.jsonl').write_text(''.join(json.dumps(anonymous[r],ensure_ascii=False)+'\n' for r in sorted(anonymous)),encoding='utf-8')
    (out/'review_key.json').write_text(json.dumps(key_records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    record={'passed':True,'source_only_continuous_selection_recomputed':True,'policy_rows':count,
        'source_and_target_recognition_recomputed':True,'fixed_routes_recomputed':True,
        'source_and_input_hashes_checked':True,'new_generation_tokens':0,'anonymous_review_outputs':len(anonymous)}
    (out/'completion_audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    if '--supplement' in sys.argv:audit_supplement()
    else:audit(sys.argv[1] if len(sys.argv)>1 else ROOT/'results/evidence_openqa_20261014_v1')
