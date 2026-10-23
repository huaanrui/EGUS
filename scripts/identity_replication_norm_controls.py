"""Before-target extension: calibrate actual residual edit RMS on DEV only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time,copy,json
import numpy as np
from mechanism.identity_replication import Experiment,TASKS,extended_metrics,derived_rows
from mechanism.fairsteer_author import ROOT,read,write,sha

ROOTOUT=ROOT/'results/identity_replication_20261015_v1'
DECL=ROOTOUT/'norm_control_declaration.json'


def energy(e,key,vs,spec):return float(e.infer(key,vs,spec)['edit_energy'].mean())


def calibrate(e,tag,vs,spec,target):
    history=[];scale=0. if target<=1e-12 else 1.
    for j in range(5):
        candidate=copy.deepcopy(spec);candidate['name']=tag+f'_normstep{j}'
        for edit in candidate['edits']:edit['amount']*=scale
        value=energy(e,'normcal_'+candidate['name'],vs,candidate)
        mismatch=abs(np.sqrt(value/max(target,1e-12))-1) if target>1e-12 else value
        history.append({'scale':scale,'energy':value,'relative_rms_error':mismatch,'spec':candidate})
        if mismatch<.02 or target<=1e-12:break
        scale=float(np.clip(scale*np.sqrt(target/max(value,1e-12)),.0001,16))
    best=min(history,key=lambda r:r['relative_rms_error']);best['history']=history[:]
    # Avoid recursive self reference when best is an element of history.
    result={k:v for k,v in best.items() if k!='history'}
    result['history']=[{k:v for k,v in r.items() if k!='history'} for r in history]
    return result


def freeze(e):
    path=e.out/'norm_controls_frozen.json'
    if path.exists():
        assert read(path)['script_sha256']==sha(Path(__file__));return
    assert not (ROOTOUT/'all_models_frozen.json').exists(),'Calibrate before any targets'
    selection=read(e.out/'selection_frozen.json');methods=selection['methods'];vs=e.task_values('dev','coref')
    references={'fine':('fine_source','fine_terminal','fine_combo')}
    if e.name=='qwen3':references['old']=('old_source','old_terminal','old_combo')
    frozen={};calibrations={};wholedev={}
    for label,(src,term,combo) in references.items():
        ref=energy(e,'norm_reference_'+combo,vs,methods[combo]);basewhole=[]
        for component in (src,term):
            key=methods[component]['edits'][0]['site'];target=energy(e,'norm_reference_'+component,vs,methods[component])
            template=e.spec(label+'_'+component+'_whole',[e.edit(key,amount=.5)],'whole')
            fitted=calibrate(e,label+'_'+component+'_whole',vs,template,target)
            basewhole+=fitted['spec']['edits'];calibrations[label+'_'+component]=fitted
            if component==src:frozen[label+'_norm_whole_source']=fitted['spec']
        candidates=[]
        for signs in ((1,1),(1,-1),(-1,1),(-1,-1)):
            edits=[{**z,'amount':z['amount']*sign} for z,sign in zip(basewhole,signs)]
            tag=label+'_whole_sign'+''.join('p' if s==1 else 'n' for s in signs)
            fitted=calibrate(e,tag,vs,e.spec(tag,edits,'whole'),ref);calibrations[tag]=fitted
            r=e.assess('dev',fitted['spec'],TASKS)
            base=e.assess('dev',methods['baseline'],TASKS);r['feasible']=e.feasible(r,base,TASKS)
            candidates.append(r)
        best=max(candidates,key=e.order)
        frozen[label+'_norm_whole_combo']=best['spec'];wholedev[label]=candidates
        prefix='random_combo_' if label=='fine' else 'old_random_combo_'
        for seed in e.c['random_seeds']:
            tag=label+f'_norm_random{seed}'
            fitted=calibrate(e,tag,vs,methods[prefix+str(seed)],ref)
            frozen[tag]=fitted['spec'];calibrations[tag]=fitted
            e.log('norm_control_calibrated',method=tag,relative_rms_error=fitted['relative_rms_error'])
    write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'script_sha256':sha(Path(__file__)),
        'original_selection_sha256':sha(e.out/'selection_frozen.json'),'methods':frozen,
        'calibrations':calibrations,'whole_dev_candidates':wholedev,
        'selection':'DEV RMS calibration; whole signs selected on DEV with original TASKS constraints; no target results'})
    e.log('norm_controls_frozen',methods=len(frozen))


def evaluate(e):
    saved=read(e.out/'norm_controls_frozen.json');assert saved['script_sha256']==sha(Path(__file__))
    assert saved['original_selection_sha256']==sha(e.out/'selection_frozen.json')
    methods=saved['methods'];result={}
    for split in ('historical','new_type1','winogender','confirmation'):
        vs=e.task_values(split,'coref');ss={k:e.infer('normeval_'+split+'_'+k,vs,s)['scores'] for k,s in methods.items()}
        e.save_scores('norm_'+split,vs,ss);result[split]={k:extended_metrics(vs,z) for k,z in ss.items()}
        e.log('norm_controls_evaluated',split=split)
    utility={};whole={k:v for k,v in methods.items() if 'whole' in k}
    for task in ('reading','binding','record','explicit','identity_required'):
        subset=methods if task in ('reading','binding') else whole
        vs=e.task_values('controls',task) if task in ('reading','binding','record') else [derived_rows(r,task,p) for r in e.values['controls'] for p in (0,1)]
        ss={k:e.infer('normutility_'+task+'_'+k,vs,s)['scores'] for k,s in subset.items()}
        e.save_scores('norm_utility_'+task,vs,ss);utility[task]={k:extended_metrics(vs,z) for k,z in ss.items()}
    write(e.out/'norm_control_results.json',{'coref':result,'utilities':utility})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('declare','freeze','evaluate'),required=True);a=p.parse_args()
    if a.stage=='declare':
        assert not (ROOTOUT/'all_models_frozen.json').exists()
        write(DECL,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'script_sha256':sha(Path(__file__)),
            'reason':'DEV-only coarse random calibration can leave large actual residual RMS mismatches',
            'rule':'up to 5 analytic RMS rescaling iterations, no target answers; four whole-module sign combinations selected on DEV',
            'secondary_comparisons_holm8':'fine combination vs norm-matched whole / mean of norm-matched random for 3 models; old Qwen3 combination same 2 comparisons',
            'status':'declared before main target opening; supplementary comparisons, original 9 primary tests unchanged'})
        return
    assert read(DECL)['script_sha256']==sha(Path(__file__))
    if a.stage=='evaluate':
        lock=read(ROOTOUT/'all_norm_controls_frozen.json')
        for name,h in lock['sha256'].items():assert sha(ROOTOUT/name/'norm_controls_frozen.json')==h
    for name in ('qwen3','minicpm5','qwen25'):
        e=Experiment(name);e.prepare()
        if a.stage=='freeze':freeze(e)
        else:evaluate(e);e.audit()
        e.close()
    if a.stage=='freeze':
        write(ROOTOUT/'all_norm_controls_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
            'sha256':{name:sha(ROOTOUT/name/'norm_controls_frozen.json') for name in ('qwen3','minicpm5','qwen25')}})


if __name__=='__main__':main()
