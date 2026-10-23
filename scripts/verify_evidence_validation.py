"""Independent data, estimator, selector and prediction replay (no GPU)."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import joblib
import numpy as np
from scipy.special import expit,logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer
from mechanism.evidence_validation import Campaign
from mechanism.evidence_story import render,surface
from mechanism.fairsteer_author import read,write,rows,sha,label_for,metrics
from mechanism.closed_transfer import canonical,template,digest as synth_digest,gender


def public_unknown(vs,vocab):
    out=[]
    for r in vs:
        choices=r['choices'] if r.get('utility') else [r['ans'+str(i)] for i in range(3)]
        found=[i for i,s in enumerate(choices) if s.strip().casefold() in vocab]
        assert len(found)<=1;out.append(found[0] if found else -1)
    return np.array(out)


def outputs(vs,base,vocab):
    s=base[:,:3].astype(float);u=public_unknown(vs,vocab);assert (u>=0).all();i=np.arange(len(vs))
    other=np.array([[j for j in range(3) if j!=k] for k in u]);ee=np.take_along_axis(s,other,axis=1)
    lp=s-logsumexp(s,axis=1,keepdims=True)
    return np.column_stack([s[i,u]-logsumexp(ee,axis=1),abs(ee[:,0]-ee[:,1]),-(np.exp(lp)*lp).sum(1)])


def choose(s,y,limit):
    valid=[]
    for t in np.r_[np.nextafter(s.min(),-np.inf),np.unique(s),np.finfo(float).max]:
        use=s>t;rec=use[y].mean();fpr=use[~y].mean()
        if fpr<=limit+1e-12:valid.append((rec,-fpr,t))
    return max(valid)[2]


def main():
    c=Campaign();c.prepare();plan=read(c.out/'plan_frozen.json');training=read(c.data/'training.json');exposure=read(c.data/'exposure.json')
    lock=read(c.out/'all_selections_frozen.json');result=read(c.out/'results.json');src=rows(Path(c.c['source_data'])/'bbq.jsonl')
    oldplan=read(c.ref/'plan_frozen.json')
    for name in ('qwen3','minicpm5'):
        assert oldplan['models'][name]==plan['models'][name]
        oldenv=read(c.ref/name/'environment.json');newenv=read(c.out/name/'environment.json')
        for field in ('torch','transformers','sklearn','layer','batch_size','candidate_ids'):assert oldenv[field]==newenv[field]
    dev=rows(c.data/'dev.jsonl');test=rows(c.data/'test.jsonl');confirmation=rows(c.data/'confirmation.jsonl')
    assert len(confirmation)==1920 and len({r['source_id'] for r in confirmation})==640
    assert not {r['source_id'] for r in confirmation}&set(exposure['source_ids'])
    assert not {r['family_id'] for r in confirmation}&set(exposure['families'])
    assert not {r['occupation_pair'] for r in confirmation}&set(exposure['occupations'])
    with Path(c.c['synth_raw']).open(encoding='utf-8',newline='') as f:raw=list(csv.DictReader(f))
    pairs=defaultdict(list)
    for r in confirmation:
        original=raw[int(r['source_id'].split(':')[1])]
        assert r['context']==original['sample'] and r['gender']==gender(original['pronoun'])
        f='synth_template:'+synth_digest(template(original['sample'],(original['occ_1'],original['occ_2'])))[:20]
        assert f==r['family_id'] and r['occupation_pair']=='|'.join(sorted((original['occ_1'],original['occ_2'])))
        answer=c.c['confirmation_unknown'] if original['type']=='type1' else original['occ_1']
        assert r['ans'+str(r['label'])]==answer
        if r['permutation']==0:pairs[r['pair_id']].append(r)
    assert len(pairs)==320 and all(len(g)==2 and {r['gender'] for r in g}=={'male','female'} for g in pairs.values())
    count=0;reports={};fitcount=0
    for model in c.c['models']:
        name=model['name'];folder=c.out/name;dd=dict(np.load(folder/'cache_dev.npz'));model_count=0
        devfamilies={r['family'] for r in dev};testfamilies={r['family'] for r in test}
        for seed in c.c['seeds']:
            sf=folder/f'seed_{seed}';sel=read(sf/'selection_frozen.json');assert lock[name][str(seed)]==sha(sf/'selection_frozen.json')
            for p,h in sel['parameters'].items():assert sha(sf/p)==h
            train=rows(c.data/f'fit_{seed}.jsonl');tc=dict(np.load(folder/f'cache_fit_{seed}.npz'));tt=training[str(seed)];vocab=tt['vocab']
            assert tc['uids'].tolist()==[r['uid'] for r in train]
            assert not {r['family'] for r in train}&(devfamilies|testfamilies)
            assert set(tt['direction_uids'])<={r['uid'] for r in train}
            assert len(tt['direction_uids'])==110
            byuid={r['uid']:i for i,r in enumerate(train)}
            pp=tc['base'].argmax(1);yy={'original':np.array([label_for(r,int(p)) for r,p in zip(train,pp)]),
                'expanded':np.array([r['label']==p for r,p in zip(train,pp)],int),
                'evidence':np.array([r['context_condition']=='ambig' for r in train],int)}
            probes=dict(np.load(sf/'probes.npz'));dy=np.array([r['context_condition']=='ambig' for r in dev]);devsignals={}
            with threadpool_limits(limits=c.c['cpu_threads']):
                for budget in c.c['fit_budgets']:
                    ii=np.array([byuid[src[i]['uid']] for i in tt['budgets'][str(budget)]])
                    for head,y in yy.items():
                        use=ii[y[ii]>=0];ref=LogisticRegression(C=1.,solver='lbfgs',max_iter=10000,tol=1e-4).fit(tc['features'][use],y[use])
                        np.testing.assert_allclose(ref.coef_[0],probes[f'{budget}_{head}_coef'],rtol=1e-9,atol=1e-9)
                        np.testing.assert_allclose(ref.intercept_[0],probes[f'{budget}_{head}_intercept'],rtol=1e-9,atol=1e-9);fitcount+=1
                    prob={h:expit(dd['features'].astype(float)@probes[f'{budget}_{h}_coef']+probes[f'{budget}_{h}_intercept']) for h in yy}
                    ss={'original':1-prob['original'],'expanded':1-prob['expanded'],'evidence':prob['evidence'],'product':(1-prob['expanded'])*prob['evidence']}
                    for gate,s in ss.items():assert choose(s,dy,c.c['gate_fpr_limit'])==sel['thresholds'][f'{budget}_{gate}']['threshold']
                    if budget==2200:devsignals=ss
            if seed==c.c['seed'] and name in ('qwen3','minicpm5'):
                assert sha(sf/'probes.npz')==sha(c.ref/name/'probes.npz') and sha(sf/'direction.npy')==sha(c.ref/name/'direction.npy')
                old=read(c.ref/name/'selection_frozen.json')['selections']
                for gate in ('original','expanded','evidence','product'):
                    assert sel['thresholds']['2200_'+gate]['threshold']==old['2200_'+gate]['gate']['threshold']
            else:
                delta=np.concatenate([np.load(p)['delta'] for p in sorted((sf/'direction_shards').glob('*.npz'))]).mean(0)
                np.testing.assert_array_equal(delta,np.load(sf/'direction.npy'))
            for kind in ('surface','question','output'):
                stored=joblib.load(sf/f'{kind}.joblib')
                x=np.array([surface(r) for r in train]) if kind=='surface' else ([r['question'] for r in train] if kind=='question' else outputs(train,tc['base'],vocab))
                dx=np.array([surface(r) for r in dev]) if kind=='surface' else ([r['question'] for r in dev] if kind=='question' else outputs(dev,dd['base'],vocab))
                import sklearn.base
                ref=sklearn.base.clone(stored)
                with threadpool_limits(limits=c.c['cpu_threads']):ref.fit(x,yy['evidence'])
                np.testing.assert_allclose(ref[-1].coef_,stored[-1].coef_,rtol=1e-9,atol=1e-9);fitcount+=1
                p=stored.predict_proba(dx)[:,1];assert choose(p,dy,c.c['gate_fpr_limit'])==sel['controls'][kind]['threshold']['threshold']
            du=public_unknown(dev,vocab);dm=devsignals['product']>sel['thresholds']['2200_product']['threshold'];ds=dd['base'].copy()
            ds[dm]=np.load(sf/'cf_dev_1.5.npz')['scores'][dm];target=np.mean(ds.argmax(1)==du);rr=[]
            for amount in np.arange(0,16.001,.25):
                ss=dd['base'].copy();ii=np.flatnonzero(dm);ss[ii,du[ii]]+=amount;rr.append((abs(np.mean(ss.argmax(1)==du)-target),amount))
            assert min(rr)[1]==sel['product_offset']
            keys=('test','confirmation','test_utility','stress','counterfactual_'+name) if seed==c.c['seed'] else ('test','confirmation','test_utility')
            for key in keys:
                vs=rows(c.data/f'{key}.jsonl');cc=dict(np.load(folder/f'cache_{key}.npz'));saved=dict(np.load(sf/f'predictions_{key}.npz'))
                assert saved['uids'].tolist()==cc['uids'].tolist()==[r['uid'] for r in vs]
                for field in ('base','features'):assert np.isfinite(cc[field]).all()
                sigp={h:expit(cc['features'].astype(float)@probes[f'2200_{h}_coef']+probes[f'2200_{h}_intercept']) for h in yy}
                sig={'original':1-sigp['original'],'expanded':1-sigp['expanded'],'evidence':sigp['evidence'],'product':(1-sigp['expanded'])*sigp['evidence']}
                unknown=public_unknown(vs,vocab);base=cc['base'];cf={a:np.load(sf/f'cf_{key}_{a:g}.npz')['scores'] for a in (1.,1.5)}
                assert all(np.isfinite(s).all() for s in cf.values())
                preds={'base':base.argmax(1)};masks={'base':np.zeros(len(vs),bool)}
                for method,gate,action,a,threshold in [('local_fs','original','steer',1.,.5),
                    ('expanded_steer','expanded','steer',1.5,sel['thresholds']['2200_expanded']['threshold']),
                    ('product_steer','product','steer',1.5,sel['thresholds']['2200_product']['threshold']),
                    ('product_direct','product','direct',1.5,sel['thresholds']['2200_product']['threshold']),
                    ('product_offset','product','offset',1.5,sel['thresholds']['2200_product']['threshold']),
                    ('evidence_steer','evidence','steer',1.5,sel['thresholds']['2200_evidence']['threshold']),
                    ('evidence_direct','evidence','direct',1.5,sel['thresholds']['2200_evidence']['threshold'])]:
                    mask=sig[gate]>threshold;ss=base.copy()
                    if action=='steer':ss[mask]=cf[a][mask]
                    if action=='offset':
                        ii=np.flatnonzero(mask&(unknown>=0));ss[ii,unknown[ii]]+=sel['product_offset']
                    p=ss.argmax(1)
                    if action=='direct':p[mask&(unknown>=0)]=unknown[mask&(unknown>=0)]
                    preds[method]=p;masks[method]=mask
                for kind in ('surface','question','output'):
                    if vs[0].get('utility'):mask=np.zeros(len(vs),bool)
                    else:
                        model_=joblib.load(sf/f'{kind}.joblib');x=np.array([surface(r) for r in vs]) if kind=='surface' else ([r['question'] for r in vs] if kind=='question' else outputs(vs,base,vocab))
                        mask=model_.predict_proba(x)[:,1]>sel['controls'][kind]['threshold']['threshold']
                    p=base.argmax(1).copy();p[mask&(unknown>=0)]=unknown[mask&(unknown>=0)];preds[kind+'_direct']=p;masks[kind+'_direct']=mask
                p=base.argmax(1).copy();use=unknown>=0;p[use]=unknown[use];preds['always_unknown']=p;masks['always_unknown']=use
                assert set(preds)=={k[5:] for k in saved if k.startswith('pred_')}
                primary=np.array([r.get('permutation',0)==0 for r in vs]) if key=='confirmation' else np.ones(len(vs),bool)
                for method,p in preds.items():
                    np.testing.assert_array_equal(p,saved['pred_'+method]);np.testing.assert_array_equal(masks[method],saved['trigger_'+method]);model_count+=len(p)
                    if key in ('test','confirmation','test_utility'):
                        labels=np.array([r['label'] for r in vs]);acc=float(np.mean(p[primary]==labels[primary]))
                        assert acc==result['models'][name][key]['per_seed'][str(seed)][method]['accuracy']
                if key=='test' and seed==c.c['seed'] and name in ('qwen3','minicpm5'):
                    oldprobe=dict(np.load(c.ref/name/'probes.npz'));oldsel=read(c.ref/name/'selection_frozen.json')['selections']
                    # Exact source assets/thresholds above and primary decisions replayed here.
                    for method,gate in [('evidence_direct','evidence'),('product_direct','product')]:
                        prob=expit(cc['features'].astype(float)@oldprobe['2200_'+gate+'_coef']+oldprobe['2200_'+gate+'_intercept']) if gate=='evidence' else sig['product']
                        mask=prob>oldsel['2200_'+gate]['gate']['threshold'];p=base.argmax(1).copy();p[mask]=unknown[mask]
                        np.testing.assert_array_equal(p,preds[method])
            print(json.dumps({'audited_model':name,'seed':seed,'decisions_so_far':model_count}),flush=True)
        tok=AutoTokenizer.from_pretrained(model['path'],local_files_only=True);cfrows=rows(c.data/f'counterfactual_{name}.jsonl');gg=defaultdict(dict)
        for r in cfrows:gg[r['pair_id']][r['variant']]=r
        for p in gg.values():assert len(tok(render(p['neutral_matched'])).input_ids)==len(tok(render(p['sufficient'])).input_ids)
        reports[name]={'prediction_records_recomputed':model_count,'five_seeds_verified':True,'length_matched_pairs':len(gg),'old_anchor_replayed':name in ('qwen3','minicpm5')};count+=model_count
    assert count==result['prediction_records']
    # Reconstruct every reported primary point and its cluster/seed bootstrap.
    from mechanism.evidence_validation_stats import comparison,holm
    repeated={}
    for model in c.c['models']:
        name=model['name']
        for key in ('test','confirmation'):
            vs=rows(c.data/f'{key}.jsonl');main=np.array([r.get('permutation',0)==0 for r in vs]) if key=='confirmation' else np.ones(len(vs),bool)
            values=[r for r,k in zip(vs,main) if k];arr=defaultdict(list)
            for seed in c.c['seeds']:
                z=dict(np.load(c.out/name/f'seed_{seed}/predictions_{key}.npz'))
                for method in ('evidence_direct','surface_direct','base'):arr[method].append(z['pred_'+method][main])
            for reference,condition in [('surface_direct',None),('base','disambig')]:
                tag=f'{name}/{key}/evidence_direct-{reference}'+('/clear' if condition else '')
                repeated[tag]=comparison(values,np.array(arr['evidence_direct']),np.array(arr[reference]),c.c['bootstrap_replicates'],condition=condition)
    holm(repeated);assert repeated==result['primary']
    write(c.out/'independent_verification.json',{'passed':True,'models':reports,'prediction_records':count,'classifiers_refitted':fitcount,
        'confirmation_questions':640,'confirmation_families':320,'confirmation_occupation_pairs':17,
        'triple_exposure_exclusion_verified':True,'old_anchor_model_and_runtime_match':True,
        'all_15_selections_locked':True,'primary_statistics_recomputed':12})
    print(json.dumps({'passed':True,'prediction_records':count,'classifiers_refitted':fitcount}),flush=True)


if __name__=='__main__':main()
