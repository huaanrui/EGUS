"""Predeclared output-score control, separate from frozen hidden-state pilot."""
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mechanism.fairsteer_author import metrics,read,sha,write


def calibrate(scores,unknown_indices,offset):
    edited=np.array(scores,copy=True)
    edited[np.arange(len(edited)),np.asarray(unknown_indices)]+=offset
    return edited


def freeze(pilot):
    path=pilot.out/'calibration_policy_frozen.json'
    policy={'script_sha256':sha(Path(__file__)),'offsets':[0.,.25,.5,1.,1.5,2.],
            'selection':'Max dev overall accuracy under the same disambig/utility/abs-BS(a) constraints; choose smaller offset on ties. If no feasible setting, flag best exploratory result.',
            'inference':'Fixed score offset on the identifiable unknown option. No context_condition, correct label or target group used. MMLU remains unchanged.',
            'created':time.strftime('%Y-%m-%d %H:%M:%S'),'split_sha256':sha(pilot.data/'splits.json')}
    if path.exists():
        saved=read(path);assert saved['script_sha256']==policy['script_sha256'];return
    assert not (pilot.out/'test_lock.json').exists(),'Declare control before pilot evaluation starts'
    write(path,policy)


def run(pilot):
    policy=read(pilot.out/'calibration_policy_frozen.json');assert policy['script_sha256']==sha(Path(__file__))
    assert policy['split_sha256']==sha(pilot.data/'splits.json')
    selection=read(pilot.out/'selection_results.json');scores,_=pilot.base('dev');examples=pilot.examples('dev')
    source=[r for r,_,_ in examples];unknown=[r['unknown_index'] for r in source];candidates=[]
    for offset in policy['offsets']:
        adjusted=calibrate(scores,unknown,offset);m=metrics(source,adjusted.argmax(1))
        failures=pilot.feasible(m,selection['baseline_utility'],selection['baseline'],selection['reference'],selection['baseline_utility'])
        candidates.append({'offset':offset,'metrics':m,'failures':failures})
    valid=[x for x in candidates if not x['failures']]
    best=max(valid or candidates,key=lambda x:(x['metrics']['accuracy'],-x['offset']))
    plan={'offset':best['offset'],'dev_failures':best['failures'],'policy_sha256':sha(pilot.out/'calibration_policy_frozen.json'),'candidates':candidates}
    plan_path=pilot.out/'calibration_plan_frozen.json'
    if plan_path.exists():assert read(plan_path)==plan
    else:write(plan_path,plan)
    scores,_=pilot.base('test');examples=pilot.examples('test');source=[r for r,_,_ in examples]
    adjusted=calibrate(scores,[r['unknown_index'] for r in source],plan['offset']);pred=adjusted.argmax(1)
    m=metrics(source,pred);m['native_disambig_unknown_rate']=float(np.mean([p==r['unknown_index'] for r,p in zip(source,pred) if r['context_condition']=='disambig']))
    baseline_utility=read(pilot.out/'predictions/baseline.json')['utility']
    result={'name':'unknown_calibration','plan':plan,'bbq':{'spec':{'name':'unknown_calibration','offset':plan['offset']},'metrics':m,'predictions':pred.tolist(),'scores':adjusted.tolist(),'actions':[plan['offset']]*len(pred),'trigger_fraction':float(plan['offset']!=0),'mean_alpha':plan['offset']},'utility':baseline_utility}
    write(pilot.out/'calibration_result.json',result);return result


if __name__=='__main__':
    from mechanism.fairsteer_improve import Pilot
    p=Pilot()
    if len(sys.argv)>1 and sys.argv[1]=='freeze':freeze(p)
    else:run(p)
