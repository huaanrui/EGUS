"""Independent CPU audit; does not import experiment selection/scoring functions."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/closed_transfer_20261014_v2'
DATA=ROOT/'data/prepared/closed_transfer_20261014_v2'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').split('\n') if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def run():
    start=time.time();manifest=read(DATA/'manifest.json');plan=read(OUT/'plan_frozen.json')['identity']
    assert sha(DATA/'manifest.json')==plan['data_manifest_sha256']
    for k,h in manifest['files'].items():assert sha(DATA/k)==h
    prior=read(ROOT/'data/prepared/closed_transfer_20261014_v1/manifest.json')
    assert prior['files']==manifest['files'],'Scientific inputs changed during smoke repair'
    for k,h in plan['source_sha256'].items():assert sha(ROOT/k)==h
    fit=ROOT/'results/fairsteer_improve_20261013_v2'
    for k,h in plan['fit_sha256'].items():assert sha(fit/k)==h
    for k,h in plan['model_hashes'].items():assert sha(Path(manifest['config']['model_path'])/k)==h
    lock=read(OUT/'test_lock.json')
    for key,name in [('plan_sha256','plan_frozen.json'),('frozen_results_sha256','frozen_results.json'),('predictions_sha256','frozen_predictions.jsonl')]:
        assert sha(OUT/name)==lock[key]
    data={k:lines(DATA/f'{k}.jsonl') for k in manifest['datasets']}
    native_bbq=lines(ROOT/'data/prepared/fairsteer_author_qwen3_v1/bbq.jsonl')
    fit_ids=read(ROOT/'data/prepared/fairsteer_improve_20261013_v2/splits.json')['bbq_indices']['fit']
    norm=lambda s:' '.join(s.lower().split())
    fit_contexts={norm(native_bbq[i]['context']) for i in fit_ids}
    source_overlap={k:len({norm(x['context']) for x in vs}&fit_contexts) for k,vs in data.items()}
    for a,b in [('synth_test','synth_calibration'),('synth_test','synth_audit'),('synth_calibration','synth_audit')]:
        for key in ('family_id','pair_id','occupation_pair'):
            assert not {x[key] for x in data[a]}&{x[key] for x in data[b]},(a,b,key)
    for key,vs in data.items():
        assert all('[' not in x['context'] and ']' not in x['context'] for x in vs)
        variants=defaultdict(list)
        for x in vs:variants[x['source_id']].append(x)
        for group in variants.values():
            assert len({x['choices'][x['label']] for x in group})==1
            if key.endswith('test'):assert len(group)==(3 if key.startswith('synth') else 2)
    probes=dict(np.load(fit/'probes.npz'))
    stored=lines(OUT/'frozen_predictions.jsonl')
    table={(x['dataset'],x['method'],x['uid']):x for x in stored}
    result=read(OUT/'frozen_results.json');predictions_checked=0;cache={};signals={}
    for key in ('synth_test','winobias_test','winogender_test','synth_calibration'):
        vs=data[key];a=dict(np.load(OUT/f'cache_{key}.npz'));cache[key]=a
        assert a['uids'].tolist()==[x['uid'] for x in vs]
        assert all(np.isfinite(a[k]).all() for k in ('base','cf1','cf15','features','mass'))
        assert np.all((a['mass']>=0)&(a['mass']<=1.000001))
        p={n:expit(a['features']@probes[n+'_coef']+float(probes[n+'_intercept'])) for n in ('original','expanded','evidence')}
        sig={'fs_gate':1-p['original'],'expanded_gate':1-p['expanded'],'evidence_gate':(1-p['expanded'])*p['evidence'],'direct_gate':p['evidence']}
        signals[key]=sig
        if key=='synth_calibration':continue
        main=np.array([x['permutation']==0 for x in vs]);labels=np.array([x['label'] for x in vs]);unk=np.array([x['unknown_index'] for x in vs])
        amb=np.array([x['condition']=='ambig' for x in vs]);clear=np.array([x['condition']=='clear' for x in vs])
        for spec in plan['specs']:
            if key!='synth_test' and spec['action'] in ('direct','offset','always'):continue
            pred=a['base'].argmax(1).copy();trigger=np.zeros(len(vs),bool)
            if spec['action']=='steer':
                trigger=sig[spec['gate']]>spec['threshold'];cf=a['cf1'] if spec['alpha']==1 else a['cf15']
                pred[trigger]=cf[trigger].argmax(1)
            elif spec['action']=='direct':
                trigger=(sig[spec['gate']]>spec['threshold'])&(unk>=0);pred[trigger]=unk[trigger]
            elif spec['action']=='always':trigger=unk>=0;pred[trigger]=unk[trigger]
            elif spec['action']=='offset':
                trigger=unk>=0;s=a['base'].copy();s[np.flatnonzero(trigger),unk[trigger]]+=spec['offset'];pred=s.argmax(1)
            for i,x in enumerate(vs):
                v=table[(key,spec['name'],x['uid'])]
                assert v['prediction']==int(pred[i]) and v['trigger']==bool(trigger[i])
                assert v['correct']==bool(pred[i]==labels[i])
                assert v['answer']==x['choices'][int(pred[i])];predictions_checked+=1
            m=result['datasets'][key]['methods'][spec['name']]
            assert abs(m['accuracy']-np.mean(pred[main]==labels[main]))<1e-12
            if key=='synth_test':
                ba=(np.mean(pred[main&amb]==labels[main&amb])+np.mean(pred[main&clear]==labels[main&clear]))/2
                assert abs(m['balanced_accuracy']-ba)<1e-12
                assert abs(m['clear_false_unknown']-np.mean(pred[main&clear]==unk[main&clear]))<1e-12
        if key=='synth_test':
            margin=a['features'][main].astype(float)@probes['evidence_coef'].astype(float)+float(probes['evidence_intercept'])
            assert abs(result['datasets'][key]['recognition']['auroc']-roc_auc_score(amb[main],margin))<1e-12
    assert predictions_checked==len(stored)==len(table)
    calibration=read(OUT/'calibration_results.json');calpred=lines(OUT/'calibration_predictions.jsonl')
    caltable={(x['budget'],x['method'],x['uid']):x for x in calpred};calchecked=0
    full=data['synth_calibration'];lookup={x['uid']:i for i,x in enumerate(full)}
    masks_by_method=defaultdict(dict)
    for record in calibration['records']:
        ids=np.array([lookup[u] for u in record['calibration_uids']]);spec=record['selection']['spec']
        vs=[full[i] for i in ids];condition=np.array([x['condition']=='clear' for x in vs]);labels=np.array([x['label'] for x in vs]);uk=np.array([x['unknown_index'] for x in vs])
        assert len(ids)==record['budget'];assert set(record['calibration_uids'])<={x['uid'] for x in full}
        masks_by_method[record['method']][record['budget']]=set(record['calibration_uids'])
        original=next(s for s in plan['specs'] if s['name']==record['method'])
        assert {k:v for k,v in spec.items() if k!='threshold'}=={k:v for k,v in original.items() if k!='threshold'}
        ca=cache['synth_calibration'];sig=signals['synth_calibration'][spec['gate']][ids]
        base=ca['base'][ids].argmax(1);baseacc=np.mean(base==labels)
        scored=[]
        for candidate in record['candidates']:
            s=candidate['spec'];trig=sig>s['threshold'];p=base.copy()
            if s['action']=='steer':
                cf=ca['cf1'] if s['alpha']==1 else ca['cf15'];p[trig]=cf[ids][trig].argmax(1)
            else:p[trig]=uk[trig]
            fpr=float(trig[condition].mean());acc=np.mean(p==labels)
            feasible=fpr<=.02+1e-12 and acc>=baseacc-.01-1e-12
            assert candidate['feasible']==feasible
            if feasible:
                ba=(np.mean(p[condition]==labels[condition])+np.mean(p[~condition]==labels[~condition]))/2
                scored.append(((ba,-float(trig.mean()),s['threshold']),s))
        best=max(scored,key=lambda x:x[0])[1];assert best==spec
        test=data['synth_test'];tc=cache['synth_test'];t=signals['synth_test'][spec['gate']]>spec['threshold'];p=tc['base'].argmax(1).copy()
        if spec['action']=='steer':p[t]=(tc['cf1'] if spec['alpha']==1 else tc['cf15'])[t].argmax(1)
        else:p[t]=np.array([x['unknown_index'] for x in test])[t]
        for i,x in enumerate(test):
            r=caltable[(record['budget'],record['method'],x['uid'])]
            assert r['prediction']==int(p[i]) and r['trigger']==bool(t[i]);calchecked+=1
    for budgets in masks_by_method.values():assert budgets[100]<=budgets[400]
    assert calchecked==len(calpred)==len(caltable)
    events=lines(OUT/'events.jsonl');names=[x['event'] for x in events]
    assert names.index('frozen_evaluation_complete')<names.index('calibration_complete')
    stable_path=OUT/'stable_calibration/results.json';stable_checked=0
    if stable_path.exists():
        stable_plan=read(OUT/'stable_calibration/plan_frozen.json')
        assert stable_plan['created']<events[names.index('frozen_evaluation_complete')]['time']
        assert stable_plan['identity']['script_sha256']==sha(ROOT/'scripts/closed_transfer_stable_calibration.py')
        stable=read(stable_path);stable_records=lines(OUT/'stable_calibration/predictions.jsonl')
        st={(x['budget'],x['method'],x['uid']):x for x in stable_records}
        stable_signals={}
        for key in ('synth_calibration','synth_test'):
            x=cache[key]['features'].astype(np.float64)
            m={n:x@probes[n+'_coef'].astype(np.float64)+float(probes[n+'_intercept']) for n in ('original','expanded','evidence')}
            stable_signals[key]={'fs_gate':-m['original'],'expanded_gate':-m['expanded'],'direct_gate':m['evidence'],
                                'evidence_gate':expit(-m['expanded'])*expit(m['evidence'])}
        for record in stable['records']:
            ids=np.array([lookup[u] for u in record['calibration_uids']]);spec=record['selection']['spec']
            vs=[full[i] for i in ids];clear=np.array([x['condition']=='clear' for x in vs]);labs=np.array([x['label'] for x in vs]);uk=np.array([x['unknown_index'] for x in vs])
            ca=cache['synth_calibration'];base=ca['base'][ids].argmax(1);baseacc=np.mean(base==labs);scored=[]
            for cand in record['candidates']:
                s=cand['spec'];t=stable_signals['synth_calibration'][s['gate']][ids]>s['threshold'];p=base.copy()
                if s['action']=='steer':p[t]=(ca['cf1'] if s['alpha']==1 else ca['cf15'])[ids][t].argmax(1)
                else:p[t]=uk[t]
                feasible=t[clear].mean()<=.02+1e-12 and np.mean(p==labs)>=baseacc-.01-1e-12
                assert cand['feasible']==feasible
                if feasible:
                    ba=(np.mean(p[clear]==labs[clear])+np.mean(p[~clear]==labs[~clear]))/2
                    scored.append(((ba,-float(t.mean()),s['threshold']),s))
            assert max(scored,key=lambda x:x[0])[1]==spec
            tc=cache['synth_test'];t=stable_signals['synth_test'][spec['gate']]>spec['threshold'];p=tc['base'].argmax(1).copy()
            if spec['action']=='steer':p[t]=(tc['cf1'] if spec['alpha']==1 else tc['cf15'])[t].argmax(1)
            else:p[t]=np.array([x['unknown_index'] for x in data['synth_test']])[t]
            for i,x in enumerate(data['synth_test']):
                r=st[(record['budget'],record['method'],x['uid'])]
                assert r['prediction']==int(p[i]) and r['trigger']==bool(t[i]);stable_checked+=1
        assert stable_checked==len(stable_records)==len(st)
    stable_v2_path=OUT/'stable_calibration_v2/results.json';stable_v2_checked=0
    if stable_v2_path.exists():
        vp=read(OUT/'stable_calibration_v2/plan_frozen.json')
        assert vp['identity']['script_sha256']==sha(ROOT/'scripts/closed_transfer_stable_calibration_v2.py')
        v2=read(stable_v2_path);stored_v2=lines(OUT/'stable_calibration_v2/predictions.jsonl')
        vt={(x['budget'],x['method'],x['uid']):x for x in stored_v2}
        for record in v2['records']:
            ids=np.array([lookup[u] for u in record['calibration_uids']]);spec=record['selection']['spec']
            vs=[full[i] for i in ids];clear=np.array([x['condition']=='clear' for x in vs]);labs=np.array([x['label'] for x in vs]);uk=np.array([x['unknown_index'] for x in vs])
            ca=cache['synth_calibration'];base=ca['base'][ids].argmax(1);baseacc=np.mean(base==labs);scored=[]
            signal=stable_signals['synth_calibration'][spec['gate']][ids]
            expected_thresholds=set(np.r_[signal,np.finfo(np.float64).max])
            assert {x['spec']['threshold'] for x in record['candidates']}==expected_thresholds
            for cand in record['candidates']:
                s=cand['spec'];t=signal>s['threshold'];p=base.copy()
                if s['action']=='steer':p[t]=(ca['cf1'] if s['alpha']==1 else ca['cf15'])[ids][t].argmax(1)
                else:p[t]=uk[t]
                feasible=t[clear].mean()<=.02+1e-12 and np.mean(p==labs)>=baseacc-.01-1e-12
                assert cand['feasible']==feasible
                if feasible:
                    ba=(np.mean(p[clear]==labs[clear])+np.mean(p[~clear]==labs[~clear]))/2
                    scored.append(((ba,-float(t.mean()),s['threshold']),s))
            assert max(scored,key=lambda x:x[0])[1]==spec
            tc=cache['synth_test'];t=stable_signals['synth_test'][spec['gate']]>spec['threshold'];p=tc['base'].argmax(1).copy()
            if spec['action']=='steer':p[t]=(tc['cf1'] if spec['alpha']==1 else tc['cf15'])[t].argmax(1)
            else:p[t]=np.array([x['unknown_index'] for x in data['synth_test']])[t]
            for i,x in enumerate(data['synth_test']):
                r=vt[(record['budget'],record['method'],x['uid'])]
                assert r['prediction']==int(p[i]) and r['trigger']==bool(t[i]);stable_v2_checked+=1
        assert stable_v2_checked==len(stored_v2)==len(vt)
    audit={'passed':True,'independent_code':True,'frozen_predictions_recomputed':predictions_checked,
           'calibrated_predictions_recomputed':calchecked,'target_threshold_selection_recomputed':True,
           'stable_calibrated_predictions_recomputed':stable_checked,
           'stable_v2_calibrated_predictions_recomputed':stable_v2_checked,
           'bbq_fit_context_target_context_overlap':source_overlap,
           'calibration_100_nested_in_400':True,'no_pair_template_or_occupation_overlap':True,
           'v1_v2_scientific_inputs_identical':True,'source_and_model_fingerprint_unchanged':True,
           'seconds':time.time()-start,'script_sha256':sha(Path(__file__)),
           'output_hashes':{n:sha(OUT/n) for n in ('frozen_results.json','frozen_predictions.jsonl','calibration_results.json','calibration_predictions.jsonl')}}
    (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':run()
