"""Exploratory source-only output calibration, separately frozen supplement."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from mechanism.evidence_story import Study,ModelStudy,discover_unknown,select_threshold,apply
from mechanism.fairsteer_author import rows,write,read,sha,metrics


def features(vs,cache,vocab):
    scores=cache['base'][:,:3].astype(float);u=np.array([discover_unknown(r,vocab) for r in vs]);idx=np.arange(len(vs))
    entities=np.array([[j for j in range(3) if j!=ui] for ui in u]);es=np.take_along_axis(scores,entities,axis=1)
    logodds=scores[idx,u]-logsumexp(es,axis=1);lp=scores-logsumexp(scores,axis=1,keepdims=True)
    entropy=-(np.exp(lp)*lp).sum(1)
    return np.column_stack([logodds,abs(es[:,0]-es[:,1]),entropy])


def main():
    s=Study();folder=s.out/'output_controls';folder.mkdir(exist_ok=True);training=read(s.data/'training.json')
    identity={'script_sha256':sha(Path(__file__)),'main_plan_sha256':sha(s.out/'plan_frozen.json'),
        'rule':'Exploratory supplement declared after main inference started. No target labels fit/select anything. Same2200fit,352dev; scalar unknown log-odds and fixed three-output-feature heads. StandardScaler+L2 lbfgs C1. Source-dev2%FPR maximum recall; direct/alpha1.5 steering share exact mask. All models/feature sets retained.',
        'main_results_preexist_at_declaration':(s.out/'results.json').exists()}
    plan=folder/'plan_frozen.json'
    if plan.exists():
        identity['main_results_preexist_at_declaration']=read(plan)['identity']['main_results_preexist_at_declaration']
        assert read(plan)['identity']==identity
    else:write(plan,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity})
    frozen=folder/'selection_frozen.json'
    fitted=read(frozen) if frozen.exists() else {}
    for key,spec in fitted.items():assert spec['model_sha256']==sha(folder/f'{key}.joblib')
    for m in ([] if fitted else s.c['models']):
        e=ModelStudy(m['name']);fit=e.values('fit');dev=e.values('dev');fc=dict(np.load(e.out/'cache_fit.npz'));dc=dict(np.load(e.out/'cache_dev.npz'))
        x=features(fit,fc,training['unknown_vocabulary']);dx=features(dev,dc,training['unknown_vocabulary'])
        y=np.array([r['context_condition']=='ambig' for r in fit]);dy=np.array([r['context_condition']=='ambig' for r in dev])
        for kind,k in [('unknown_logodds',1),('output_three_features',3)]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=10000,tol=1e-4))
            with threadpool_limits(limits=s.c['cpu_threads']):model.fit(x[:,:k],y)
            score=model.predict_proba(dx[:,:k])[:,1];best,rr=select_threshold(score,dy,s.c['gate_fpr_limit'])
            key=e.name+'_'+kind;joblib.dump(model,folder/f'{key}.joblib');fitted[key]={'threshold':best,'k':k,
                'model_sha256':sha(folder/f'{key}.joblib'),'training_rows':len(fit),'dev_candidates':rr}
    if not frozen.exists():write(frozen,fitted)
    result={};count=0
    for m in s.c['models']:
        e=ModelStudy(m['name'])
        for kind in ('unknown_logodds','output_three_features'):
            key=e.name+'_'+kind;spec=fitted[key];model=joblib.load(folder/f'{key}.joblib');datasets={}
            for dataset in ('test','stress','counterfactual_'+e.name):
                vs=e.values(dataset);cache=dict(np.load(e.out/f'cache_{dataset}.npz'));x=features(vs,cache,training['unknown_vocabulary'])
                p=model.predict_proba(x[:,:spec['k']])[:,1];mask=p>spec['threshold']['threshold'];unknown=np.array([discover_unknown(r,training['unknown_vocabulary']) for r in vs])
                pred={a:apply(cache['base'],cache['cf1.5'],unknown,mask,a)[0] for a in ('direct','steer')}
                groups={};condition=np.array([r['context_condition']=='ambig' for r in vs])
                for variant in sorted({r.get('variant','all') for r in vs}):
                    ii=[i for i,r in enumerate(vs) if r.get('variant','all')==variant];yy=condition[ii]
                    groups[variant]={'methods':{a:metrics([vs[i] for i in ii],pp[ii]) for a,pp in pred.items()},
                        'trigger_fraction':float(mask[ii].mean()),'insufficient_recall':float(mask[ii][yy].mean()) if yy.any() else None,
                        'sufficient_fpr':float(mask[ii][~yy].mean()) if (~yy).any() else None}
                datasets[dataset]={'auroc':float(roc_auc_score(condition,p)),'groups':groups}
                np.savez(folder/f'{key}_{dataset}.npz',probability=p,mask=mask,uids=np.array([r['uid'] for r in vs]),**pred);count+=len(vs)*2
            result[key]=datasets
    write(folder/'results.json',{'results':result,'prediction_records':count,'frozen_sha256':sha(folder/'selection_frozen.json')})
    print({'output_controls_complete':True,'prediction_records':count},flush=True)


if __name__=='__main__':main()
