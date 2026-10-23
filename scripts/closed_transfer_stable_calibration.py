"""Numerically stable target calibration declared before primary results."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expit
from mechanism.closed_transfer import Experiment,rank,calibrate,apply_policy,metric
from mechanism.fairsteer_author import ROOT,read,sha,write


def main():
    e=Experiment();e.prepare();folder=e.out/'stable_calibration';folder.mkdir(exist_ok=True)
    protocol=ROOT/'research/2026-10-14_closed_transfer_stable_calibration_protocol.md'
    identity={'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(protocol),
              'main_plan_sha256':sha(e.out/'plan_frozen.json'),'data_manifest_sha256':sha(e.data/'manifest.json'),
              'score_units':{'fs_gate':'negative original margin','expanded_gate':'negative expanded margin',
                             'direct_gate':'evidence margin','evidence_gate':'float64 expit(-expanded margin)*expit(evidence margin)'}}
    frozen=folder/'plan_frozen.json'
    if sys.argv[1]=='declare':
        assert not (e.out/'frozen_results.json').exists(),'Must declare before observing primary results'
        if frozen.exists():assert read(frozen)['identity']==identity
        else:write(frozen,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity})
        print('Stable score calibration declared before primary results.');return
    assert sys.argv[1]=='run' and (e.out/'test_lock.json').exists()
    assert read(frozen)['identity']==identity
    full=e.datasets['synth_calibration'];cache=e.cache('synth_calibration')
    test=e.datasets['synth_test'];tc=e.cache('synth_test')
    def signals(x):
        m={n:x.astype(np.float64)@e.probes[n+'_coef'].astype(np.float64)+float(e.probes[n+'_intercept']) for n in ('original','expanded','evidence')}
        return {'fs_gate':-m['original'],'expanded_gate':-m['expanded'],'direct_gate':m['evidence'],
                'evidence_gate':expit(-m['expanded'])*expit(m['evidence'])}
    ss=signals(cache['features']);ts=signals(tc['features']);groups=defaultdict(list)
    for i,v in enumerate(full):groups[(v['condition'],v['pair_id'])].append(i)
    records=[];predictions=[]
    for budget in e.c['calibration_budgets']:
        ids=[]
        for cond in ('ambig','clear'):
            keys=sorted((k for k in groups if k[0]==cond),key=lambda k:rank(e.c['seed'],k))
            for k in keys[:budget//4]:ids.extend(groups[k])
        vs=[full[i] for i in ids];cc={k:v[ids] for k,v in cache.items() if k!='uids'};sig={k:v[ids] for k,v in ss.items()}
        for spec in e.specs:
            if spec['action'] not in ('steer','direct'):continue
            best,candidates=calibrate(vs,cc,sig,spec,e.c)
            p,t,scores=apply_policy(tc['base'],tc['cf1'],tc['cf15'],ts,test,best['spec'])
            mainmask=np.array([x['permutation']==0 for x in test]);primary=[x for x,m in zip(test,mainmask) if m]
            records.append({'budget':budget,'method':spec['name'],'calibration_uids':[v['uid'] for v in vs],
                            'selection':best,'candidates':candidates,
                            'test':metric(primary,p[mainmask],t[mainmask],scores[mainmask] if spec['action']=='steer' else None)})
            for i,v in enumerate(test):predictions.append({'uid':v['uid'],'method':spec['name'],'budget':budget,
                                                          'prediction':int(p[i]),'trigger':bool(t[i])})
    precision={}
    for key,c in [('test',tc),('calibration',cache)]:
        x=c['features'].astype(np.float64);m=x@e.probes['evidence_coef'].astype(np.float64)+float(e.probes['evidence_intercept'])
        p32=expit(c['features']@e.probes['evidence_coef']+float(e.probes['evidence_intercept']))
        precision[key]={'rows':len(m),'unique_float32_probability_scores':len(np.unique(p32)),
                        'unique_float64_margin_scores':len(np.unique(m)),
                        'float32_exact_zero':int(np.sum(p32==0)),'float32_exact_one':int(np.sum(p32==1))}
    write(folder/'results.json',{'setting':'Target-supervised stable score calibration; not frozen transfer.',
                                'records':records,'precision_diagnostic':precision})
    (folder/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in predictions),encoding='utf-8')
    print(json.dumps({'completed':True,'predictions':len(predictions),'precision':precision}))


if __name__=='__main__':main()
