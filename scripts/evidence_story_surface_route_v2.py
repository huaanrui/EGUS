"""Pinned surface classifiers after the separately documented reporting repair."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from mechanism.evidence_story import Study,ModelStudy,surface,select_threshold,discover_unknown
from mechanism.fairsteer_author import rows,read,write,sha,metrics


def main():
    s=Study();folder=s.out/'surface_routes_v2';folder.mkdir(exist_ok=True)
    vocab=read(s.data/'training.json')['unknown_vocabulary'];controls=s.out/'surface_controls'
    identity={'script_sha256':sha(Path(__file__)),'plan_sha256':sha(s.out/'plan_frozen.json'),
        'surface_models':{n:sha(controls/f'{n}.joblib') for n in ('surface','question_only')},
        'repair_protocol_sha256':sha(Path('research/2026-10-17_evidence_story_surface_reporting_v2.md')),
        'previous_plan_sha256':sha(s.out/'surface_routes/plan_frozen.json'),
        'scope':'Exploratory routing supplement declared after observing main/surface AUROC. Source-dev-only same2%FPR selector, no target tuning; include both surface and question-only controls. No claim of preregistered confirmation.'}
    path=folder/'plan_frozen.json'
    if path.exists():assert read(path)['identity']==identity
    else:write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity})
    dev=rows(s.data/'dev.jsonl');y=np.array([r['context_condition']=='ambig' for r in dev]);fitted={}
    for kind in ('surface','question_only'):
        model=joblib.load(controls/f'{kind}.joblib');inp=np.array([surface(r) for r in dev]) if kind=='surface' else [r['question'] for r in dev]
        probability=model.predict_proba(inp)[:,1];best,candidates=select_threshold(probability,y,s.c['gate_fpr_limit'])
        fitted[kind]={'selection':best,'candidates':candidates}
    frozen=folder/'selection_frozen.json'
    if frozen.exists():assert read(frozen)==fitted
    else:write(frozen,fitted)
    result={};count=0;comparisons={}
    for m in s.c['models']:
        e=ModelStudy(m['name']);rr={}
        for kind in ('surface','question_only'):
            model=joblib.load(controls/f'{kind}.joblib');datasets={}
            for dataset in ('test','stress','counterfactual_'+e.name):
                vs=e.values(dataset);cc=dict(np.load(e.out/f'cache_{dataset}.npz'));inp=np.array([surface(r) for r in vs]) if kind=='surface' else [r['question'] for r in vs]
                p=model.predict_proba(inp)[:,1];mask=p>fitted[kind]['selection']['threshold'];unknown=np.array([discover_unknown(r,vocab) for r in vs])
                pred=cc['base'].argmax(1).copy();pred[mask]=unknown[mask];y=np.array([r['context_condition']=='ambig' for r in vs])
                groups={}
                for variant in sorted({r.get('variant','all') for r in vs}):
                    ii=np.array([i for i,r in enumerate(vs) if r.get('variant','all')==variant]);yy=y[ii]
                    groups[variant]={'metrics':metrics([vs[i] for i in ii],pred[ii]),'trigger_fraction':float(mask[ii].mean()),
                        'insufficient_recall':float(mask[ii][yy].mean()) if yy.any() else None,
                        'sufficient_fpr':float(mask[ii][~yy].mean()) if (~yy).any() else None}
                datasets[dataset]=groups;np.savez(folder/f'{e.name}_{kind}_{dataset}.npz',probability=p,mask=mask,prediction=pred,uids=np.array([r['uid'] for r in vs]));count+=len(vs)
                previous=dict(np.load(s.out/'surface_routes'/f'{e.name}_{kind}_{dataset}.npz'))
                comparisons[f'{e.name}_{kind}_{dataset}']={
                    'max_probability_difference':float(abs(previous['probability']-p).max()),
                    'trigger_differences':int(np.sum(previous['mask']!=mask)),
                    'prediction_differences':int(np.sum(previous['prediction']!=pred))}
            rr[kind]=datasets
        result[e.name]=rr
    write(folder/'results.json',{'version':2,'results':result,'prediction_records':count,'selection_sha256':sha(frozen)})
    write(folder/'previous_version_comparison.json',comparisons)
    print({'surface_routing_complete':True,'predictions':count},flush=True)


if __name__=='__main__':main()
