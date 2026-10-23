"""Cache-only unlabeled adaptation; controlled prevalence uses labels for sampling only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import tomllib
import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from mechanism.fairsteer_author import ROOT,read,rows,sha,write
from mechanism.evidence_mechanism import scalar_align,feature_align_margin

OUT=ROOT/'results/evidence_mechanism_20261014_v1/calibration'
OLD=ROOT/'results/evidence_openqa_20261014_v1'
FIT=ROOT/'results/fairsteer_improve_20261013_v2'
DATA=ROOT/'data/prepared/evidence_openqa_20261014_v1'
TRANSFER=ROOT/'results/closed_transfer_20261014_v2'
TDATA=ROOT/'data/prepared/closed_transfer_20261014_v2'


def risk_threshold(y,scores,max_fpr=.02):
    candidates=np.r_[np.unique(scores),np.finfo(np.float64).max]
    feasible=[t for t in candidates if np.mean(scores[np.asarray(y)==0]>t)<=max_fpr+1e-12]
    return float(max(feasible,key=lambda t:(np.mean(scores[np.asarray(y)==1]>t),t)))


def score_metrics(y,scores,threshold):
    y=np.asarray(y,int);pred=scores>threshold
    return {'auroc':float(roc_auc_score(y,scores)),'insufficient_recall':float(pred[y==1].mean()),
            'sufficient_fpr':float(pred[y==0].mean()),'trigger_rate':float(pred.mean()),
            'balanced_accuracy':float((pred[y==1].mean()+(~pred[y==0]).mean())/2)}


def quantile_align(source,adapt,test):
    a=np.sort(np.asarray(adapt));q=np.linspace(0,1,len(a));unique,index=np.unique(a,return_index=True)
    percentiles=np.interp(test,unique,q[index],left=0,right=1)
    return np.quantile(source,percentiles)


def load_features(key,style='chat'):
    z=dict(np.load(OLD/f'features_{key}_{style}.npz'));vs=rows(DATA/f'{key}.jsonl')
    assert z['uids'].tolist()==[r['uid'] for r in vs]
    return vs,z['features'].astype(np.float64)


def source_stats():
    split=read(ROOT/'data/prepared/fairsteer_improve_20261013_v2/splits.json')['bbq_indices']
    data=rows(ROOT/'data/prepared/fairsteer_author_qwen3_v1/bbq.jsonl')
    raw=np.load(ROOT/'results/fairsteer_author_qwen3_v1/features.npy',mmap_mode='r')
    p=dict(np.load(FIT/'probes.npz'));w=p['evidence_coef'].astype(np.float64);b=float(p['evidence_intercept'])
    indices=split['fit'];total=np.zeros(len(w));squared=np.zeros(len(w));margins=[]
    for start in range(0,len(indices),512):
        x=np.asarray(raw[indices[start:start+512],18,:],np.float64)
        total+=x.sum(0);squared+=(x*x).sum(0);margins.append(x@w+b)
    mean=total/len(indices);std=np.sqrt(np.maximum(squared/len(indices)-mean*mean,0))
    dev=np.asarray(raw[split['dev'],18,:],np.float64);y=np.array([data[i]['context_condition']=='ambig' for i in split['dev']])
    margin=dev@w+b
    return {'mean':mean,'std':std,'scores':np.concatenate(margins),'coef':w,'intercept':b,
            'thresholds':{'legacy_fixed':float(logit(.25)),'source_dev_fpr02':risk_threshold(y,margin)}}


def route_quality(vs,trigger,responses):
    if responses is None:return None
    ev=[responses[r['uid']]['evaluation'] for r in vs];y=np.array([r['insufficient'] for r in vs],bool)
    abstain=np.array([r['abstain'] for r in ev]);routed=abstain|trigger
    result={'insufficient_abstain':float(routed[y].mean()),'sufficient_false_abstain':float(routed[~y].mean())}
    if vs[0]['dataset']=='squad':
        for key in ('exact_match','f1'):
            scores=np.array([r[key] for r in ev],float);scores[trigger]=y[trigger].astype(float)
            result[key]=float(scores.mean());result['answerable_'+key]=float(scores[~y].mean())
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c=tomllib.loads((ROOT/'configs/experiment_evidence_mechanism_v1.toml').read_text())
    inputs=[Path(__file__),ROOT/'mechanism/evidence_mechanism.py',ROOT/'configs/experiment_evidence_mechanism_v1.toml',
        FIT/'probes.npz',OLD/'squad_probe.npz',ROOT/'results/fairsteer_author_qwen3_v1/feature_hashes.json']
    inputs+=list(OLD.glob('features_*.npz'))
    inputs+=[TRANSFER/'cache_synth_calibration.npz',TRANSFER/'cache_synth_test.npz']
    identity={str(p.relative_to(ROOT)):sha(p) for p in inputs};path=OUT/'plan_frozen.json'
    if path.exists():assert read(path)['identity']==identity
    else:write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,
        'scope':'No target-label adaptation. Target labels only construct prevalence stress mixtures and evaluate. Five seeds, all methods reported; no target-selected winner.'})
    with threadpool_limits(limits=4):
        source=source_stats();sources={'bbq':source}
        sqfit,sqx=load_features('squad_fit');sqdev,sqdx=load_features('squad_dev')
        p=dict(np.load(OLD/'squad_probe.npz'));w=p['coef'].astype(np.float64);b=float(p['intercept'])
        sources['squad']={'mean':sqx.mean(0),'std':sqx.std(0),'scores':sqx@w+b,'coef':w,'intercept':b,
            'thresholds':{'source_dev_fpr02':risk_threshold(np.array([r['insufficient'] for r in sqdev]),sqdx@w+b)}}
        jobs=[]
        for source_name,key,style in [('bbq','squad','chat'),('bbq','bbq','chat'),('bbq','bbq','raw'),('squad','bbq','chat')]:
            av,ax=load_features(key+'_dev',style);tv,tx=load_features(key+'_test',style)
            field='article' if key=='squad' else 'group';assert not {r[field] for r in av}&{r[field] for r in tv}
            responses={r['uid']:r for r in rows(OLD/f'generations_{key}_test.jsonl')} if style=='chat' else None
            jobs.append((f'{source_name}_to_{key}_{style}',source_name,av,ax,tv,tx,responses))
        ac=dict(np.load(TRANSFER/'cache_synth_calibration.npz'));tc=dict(np.load(TRANSFER/'cache_synth_test.npz'))
        av=rows(TDATA/'synth_calibration.jsonl');tv=rows(TDATA/'synth_test.jsonl')
        assert ac['uids'].tolist()==[r['uid'] for r in av] and tc['uids'].tolist()==[r['uid'] for r in tv]
        av=[{**r,'insufficient':int(r['condition']=='ambig')} for r in av];tv=[{**r,'insufficient':int(r['condition']=='ambig')} for r in tv]
        assert not {r['occupation_pair'] for r in av}&{r['occupation_pair'] for r in tv}
        # Only primary order in calibration diagnostics; option stability is a separate experiment.
        keep=np.array([r['permutation']==0 for r in tv]);tv=[r for r,m in zip(tv,keep) if m]
        jobs.append(('bbq_to_synth','bbq',av,ac['features'].astype(np.float64),tv,tc['features'][keep].astype(np.float64),None))
        records=[];npz={}
        for name,source_name,av,ax,tv,tx,responses in jobs:
            src=sources[source_name];w=src['coef'];b=src['intercept'];am=ax@w+b;tm=tx@w+b
            y=np.array([r['insufficient'] for r in tv]);ay=np.array([r['insufficient'] for r in av])
            mixtures=[('natural',None,None)]
            for prevalence in c['adaptation_prevalences']:
                for repeat in range(c['adaptation_repeats']):mixtures.append((str(prevalence),repeat,prevalence))
            for mix,repeat,prevalence in mixtures:
                if prevalence is None:ids=np.arange(len(av))
                else:
                    rng=np.random.default_rng(c['seed']+repeat);n1=round(c['adaptation_rows']*prevalence)
                    ids=np.r_[rng.choice(np.flatnonzero(ay==1),n1,replace=False),rng.choice(np.flatnonzero(ay==0),c['adaptation_rows']-n1,replace=False)]
                a=ax[ids];scores={'frozen':tm,'scalar_moments':scalar_align(src['scores'],am[ids],tm),
                    'feature_diagonal_moments':feature_align_margin(src['mean'],src['std'],a,tx,w,b),
                    'score_quantiles':quantile_align(src['scores'],am[ids],tm)}
                for method,s in scores.items():
                    assert np.isfinite(s).all()
                    for threshold_name,threshold in src['thresholds'].items():
                        rec={'dataset':name,'adaptation_mixture':mix,'repeat':repeat,'adaptation_rows':len(ids),
                            'method':method,'threshold_name':threshold_name,'threshold':threshold,'rows':len(tv),
                            **score_metrics(y,s,threshold),'routing':route_quality(tv,s>threshold,responses)}
                        records.append(rec)
                    if mix=='natural':npz[name+'__'+method]=s
            print(json.dumps({'completed':name,'test_rows':len(tv),'adaptation_rows':len(av)}),flush=True)
        write(OUT/'results.json',{'records':records,'source_thresholds':{k:v['thresholds'] for k,v in sources.items()},
            'scope':'All results exploratory. Moment matching and quantile matching make distributional assumptions; no target FPR guarantee. Raw-format inputs have no matching generated-response evaluation.'})
        np.savez(OUT/'natural_scores.npz',**npz)
        with (OUT/'metrics.jsonl').open('w',encoding='utf-8') as f:
            for rec in records:f.write(json.dumps(rec)+'\n')
        print(json.dumps({'rows':len(records),'output':str(OUT)}))


if __name__=='__main__':main()
