"""Independent replay of source fits, selections, policies and metadata guards."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from mechanism.fairsteer_author import read,rows,write,sha,label_for,metrics
from mechanism.evidence_story import Study,ModelStudy,render
from transformers import AutoTokenizer
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scripts.evidence_story_output_control import features as output_features
from mechanism.evidence_story import surface


def main():
    s=Study();s.prepare();plan=read(s.out/'plan_frozen.json');lock=read(s.out/'all_models_frozen.json')
    fit=rows(s.data/'fit.jsonl');dev=rows(s.data/'dev.jsonl');test=rows(s.data/'test.jsonl');training=read(s.data/'training.json')
    src=rows(Path(s.c['source_data'])/'bbq.jsonl');fitlookup={r['uid']:i for i,r in enumerate(fit)}
    fam={k:{r['family'] for r in values} for k,values in [('fit',fit),('dev',dev),('test',test)]}
    assert not fam['fit']&fam['dev'] and not fam['fit']&fam['test'] and not fam['dev']&fam['test']
    reports={};total=0;allresults=read(s.out/'results.json');output_count=0;surface_count=0
    for spec in s.c['models']:
        name=spec['name'];folder=s.out/name;sel=read(folder/'selection_frozen.json')
        assert lock[name]==sha(folder/'selection_frozen.json')
        for p,h in sel['parameters'].items():assert sha(folder/p)==h
        probes=dict(np.load(folder/'probes.npz'));ff=dict(np.load(folder/'cache_fit.npz'));dd=dict(np.load(folder/'cache_dev.npz'))
        assert ff['uids'].tolist()==[r['uid'] for r in fit] and dd['uids'].tolist()==[r['uid'] for r in dev]
        fit_labels={'evidence':np.array([r['context_condition']=='ambig' for r in fit],int),
            'original':np.array([label_for(r,int(p)) for r,p in zip(fit,ff['base'].argmax(1))]),
            'expanded':np.array([r['label']==p for r,p in zip(fit,ff['base'].argmax(1))],int)}
        maxcoef=0.;thresholds=0
        with threadpool_limits(limits=s.c['cpu_threads']):
            for budget in s.c['fit_budgets']:
                ii=np.array([fitlookup[src[i]['uid']] for i in training['budget_indices'][str(budget)]])
                for head,y in fit_labels.items():
                    use=ii[y[ii]>=0];clf=LogisticRegression(C=s.c['probe_C'],max_iter=s.c['probe_max_iter'],
                        tol=s.c['probe_tol'],solver='lbfgs').fit(ff['features'][use],y[use])
                    co=probes[f'{budget}_{head}_coef'];inter=float(probes[f'{budget}_{head}_intercept'])
                    np.testing.assert_allclose(clf.coef_[0],co,rtol=1e-9,atol=1e-9)
                    np.testing.assert_allclose(clf.intercept_[0],inter,rtol=1e-9,atol=1e-9)
                    maxcoef=max(maxcoef,float(abs(clf.coef_[0]-co).max()))
                raw={h:expit(dd['features'].astype(float)@probes[f'{budget}_{h}_coef']+probes[f'{budget}_{h}_intercept']) for h in fit_labels}
                gates={'original':1-raw['original'],'expanded':1-raw['expanded'],'evidence':raw['evidence'],
                       'product':(1-raw['expanded'])*raw['evidence']}
                yd=np.array([r['context_condition']=='ambig' for r in dev]);unknown=np.array([r['unknown_index'] for r in dev])
                for gate,signal in gates.items():
                    candidates=[]
                    for t in np.r_[np.nextafter(signal.min(),-np.inf),np.unique(signal),np.finfo(float).max]:
                        mask=signal>t;r=mask[yd].mean();f=mask[~yd].mean()
                        if f<=s.c['gate_fpr_limit']+1e-12:candidates.append((r,-f,t))
                    threshold=max(candidates)[2];chosen=sel['selections'][f'{budget}_{gate}']
                    assert threshold==chosen['gate']['threshold'];thresholds+=1;mask=signal>threshold
                    for a in s.c['alphas']:
                        ss=dd['base'].copy();ss[mask]=dd[f'cf{a:g}'][mask];target=np.mean(ss.argmax(1)==unknown);rr=[]
                        for amount in np.arange(0,16.001,.25):
                            off=dd['base'].copy();idx=np.flatnonzero(mask);off[idx,unknown[idx]]+=amount
                            rate=np.mean(off.argmax(1)==unknown);rr.append((abs(rate-target),amount))
                        assert min(rr)[1]==chosen['offsets'][f'{a:g}']['amount']
        chunks=[dict(np.load(p))['delta'] for p in sorted((folder/'direction_shards').glob('*.npz'))]
        np.testing.assert_array_equal(np.concatenate(chunks).mean(0),np.load(folder/'direction.npy'))
        lookup={(r['method'],r['uid']):r for r in rows(folder/'predictions.jsonl')};count=0
        e=ModelStudy(name);vs,cache,pred,masks,scores=e.policies('test')
        for key,p in pred.items():
            mm=metrics(vs,p);saved=allresults['models'][name]['methods'][key]
            assert mm['accuracy']==saved['accuracy'] and mm['ambig']['accuracy']==saved['ambig']['accuracy'] and mm['disambig']['accuracy']==saved['disambig']['accuracy']
            for r,pp,t in zip(vs,p,masks[key]):
                z=lookup[(key,r['uid'])];assert z['prediction']==pp and z['trigger']==bool(t) and z['label']==r['label'];count+=1
            if key.endswith('_direct'):
                prefix=key[:-len('_direct')];np.testing.assert_array_equal(masks[key],masks[prefix+'_steer1.5'])
                np.testing.assert_array_equal(masks[key],masks[prefix+'_offset1.5'])
        yt=np.array([r['context_condition']=='ambig' for r in vs])
        for budget in s.c['fit_budgets']:
            ep=expit(cache['features'].astype(float)@probes[f'{budget}_evidence_coef']+probes[f'{budget}_evidence_intercept'])
            assert float(roc_auc_score(yt,ep))==allresults['models'][name]['recognition'][f'{budget}_evidence']['auroc']
        assert count==len(lookup)
        # Reimplement direct/steer/offset without the production policy helper.
        unknown=np.array([next(i for i in range(3) if r['ans'+str(i)].strip().casefold() in training['unknown_vocabulary']) for r in test])
        for budget in s.c['fit_budgets']:
            raw={h:expit(cache['features'].astype(float)@probes[f'{budget}_{h}_coef']+probes[f'{budget}_{h}_intercept']) for h in fit_labels}
            sig={'original':1-raw['original'],'expanded':1-raw['expanded'],'evidence':raw['evidence'],'product':(1-raw['expanded'])*raw['evidence']}
            for gate,signal in sig.items():
                stem=f'{budget}_{gate}';mask=signal>sel['selections'][stem]['gate']['threshold'];p=cache['base'].argmax(1).copy();p[mask]=unknown[mask]
                np.testing.assert_array_equal(p,pred[stem+'_direct'])
                for a in s.c['alphas']:
                    ss=cache['base'].copy();ss[mask]=cache[f'cf{a:g}'][mask]
                    np.testing.assert_array_equal(ss.argmax(1),pred[stem+f'_steer{a:g}'])
                    ss=cache['base'].copy();idx=np.flatnonzero(mask);ss[idx,unknown[idx]]+=sel['selections'][stem]['offsets'][f'{a:g}']['amount']
                    np.testing.assert_array_equal(ss.argmax(1),pred[stem+f'_offset{a:g}'])
        tok=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True);cv=rows(s.data/f'counterfactual_{name}.jsonl');pairs=defaultdict(dict)
        for r in cv:pairs[r['pair_id']][r['variant']]=r
        for pair in pairs.values():
            assert len(tok(render(pair['neutral_matched'])).input_ids)==len(tok(render(pair['sufficient'])).input_ids)
            assert pair['neutral_matched']['label']==pair['insufficient']['label']
            assert pair['sufficient']['context'].startswith(pair['insufficient']['context']+' ')
            assert pair['insufficient']['family'] not in fam['fit']|fam['dev']
        matched=np.array([r['variant'] in ('neutral_matched','sufficient') for r in cv]);cy=np.array([r['context_condition']=='ambig' for r in cv])
        cx=dict(np.load(folder/f'cache_counterfactual_{name}.npz'))['features'].astype(float)
        ep=expit(cx@probes['2200_evidence_coef']+probes['2200_evidence_intercept'])
        assert float(roc_auc_score(cy[matched],ep[matched]))==allresults['models'][name]['counterfactual']['equal_length_recognition']['evidence']
        controls=s.out/'surface_controls';route_folder=s.out/'surface_routes_v2'
        if (route_folder/'results.json').exists():
            declaration=read(route_folder/'plan_frozen.json')['identity']
            assert declaration['script_sha256']==sha(Path('scripts/evidence_story_surface_route_v2.py'))
            selections=read(route_folder/'selection_frozen.json')
            for kind in ('surface','question_only'):
                assert declaration['surface_models'][kind]==sha(controls/f'{kind}.joblib')
                model=joblib.load(controls/f'{kind}.joblib')
                inp=np.array([surface(r) for r in dev]) if kind=='surface' else [r['question'] for r in dev]
                pp=model.predict_proba(inp)[:,1];dy=np.array([r['context_condition']=='ambig' for r in dev]);candidates=[]
                for t in np.r_[np.nextafter(pp.min(),-np.inf),np.unique(pp),np.finfo(float).max]:
                    use=pp>t
                    if use[~dy].mean()<=s.c['gate_fpr_limit']+1e-12:candidates.append((use[dy].mean(),-use[~dy].mean(),t))
                assert max(candidates)[2]==selections[kind]['selection']['threshold']
                for dataset in ('test','stress','counterfactual_'+name):
                    values=rows(s.data/f'{dataset}.jsonl');cc=dict(np.load(folder/f'cache_{dataset}.npz'))
                    inputs=np.array([surface(r) for r in values]) if kind=='surface' else [r['question'] for r in values]
                    saved=dict(np.load(route_folder/f'{name}_{kind}_{dataset}.npz'));pp=model.predict_proba(inputs)[:,1]
                    np.testing.assert_array_equal(pp,saved['probability']);mask=pp>selections[kind]['selection']['threshold']
                    np.testing.assert_array_equal(mask,saved['mask']);prediction=cc['base'].argmax(1)
                    unknown=np.array([r['unknown_index'] for r in values]);prediction[mask]=unknown[mask]
                    np.testing.assert_array_equal(prediction,saved['prediction']);surface_count+=len(values)
                    if dataset.startswith('counterfactual'):
                        use=np.array([r['variant'] in ('neutral_matched','sufficient') for r in values]);yy=np.array([r['context_condition']=='ambig' for r in values])
                        margin=model.decision_function(inputs)
                        assert float(roc_auc_score(yy[use],margin[use]))==allresults['surface_controls'][kind][dataset]['equal_length_auroc']
        for key in ('fit','dev','dev_utility','test','test_utility','stress','counterfactual_'+name):
            z=dict(np.load(folder/f'cache_{key}.npz'))
            for k,v in z.items():
                if v.dtype.kind in 'fiu':assert np.isfinite(v).all(),(name,key,k)
            assert len(z['uids'])==len(rows(s.data/f'{key}.jsonl'))
        reports[name]={'probe_coefficients_replayed_max_error':maxcoef,'thresholds_replayed':thresholds,
            'prediction_records_recomputed':count,'token_length_matched_pairs':len(pairs),'all_cache_values_finite':True,
            'exact_same_masks':True,'source_model_hashes':plan['models'][name]};total+=count
        output_folder=s.out/'output_controls'
        if (output_folder/'results.json').exists():
            oc=read(output_folder/'selection_frozen.json');op=read(output_folder/'plan_frozen.json')
            assert op['identity']['script_sha256']==sha(Path('scripts/evidence_story_output_control.py'))
            for kind in ('unknown_logodds','output_three_features'):
                key=name+'_'+kind;spec=oc[key];assert sha(output_folder/f'{key}.joblib')==spec['model_sha256']
                model=joblib.load(output_folder/f'{key}.joblib')
                ref=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=10000,tol=1e-4))
                xx=output_features(fit,ff,training['unknown_vocabulary'])[:,:spec['k']]
                with threadpool_limits(limits=s.c['cpu_threads']):ref.fit(xx,fit_labels['evidence'])
                np.testing.assert_allclose(ref[-1].coef_,model[-1].coef_,rtol=1e-10,atol=1e-10)
                dx=output_features(dev,dd,training['unknown_vocabulary'])[:,:spec['k']]
                dp=model.predict_proba(dx)[:,1];dy=np.array([r['context_condition']=='ambig' for r in dev]);candidates=[]
                for t in np.r_[np.nextafter(dp.min(),-np.inf),np.unique(dp),np.finfo(float).max]:
                    use=dp>t
                    if use[~dy].mean()<=s.c['gate_fpr_limit']+1e-12:candidates.append((use[dy].mean(),-use[~dy].mean(),t))
                assert max(candidates)[2]==spec['threshold']['threshold']
                for dataset in ('test','stress','counterfactual_'+name):
                    values=rows(s.data/f'{dataset}.jsonl');cc=dict(np.load(folder/f'cache_{dataset}.npz'))
                    saved=dict(np.load(output_folder/f'{key}_{dataset}.npz'))
                    pp=model.predict_proba(output_features(values,cc,training['unknown_vocabulary'])[:,:spec['k']])[:,1]
                    np.testing.assert_array_equal(pp,saved['probability']);mask=pp>spec['threshold']['threshold'];np.testing.assert_array_equal(mask,saved['mask'])
                    unknown=np.array([r['unknown_index'] for r in values]);direct=cc['base'].argmax(1);direct[mask]=unknown[mask]
                    steer=cc['base'].copy();steer[mask]=cc['cf1.5'][mask]
                    np.testing.assert_array_equal(direct,saved['direct']);np.testing.assert_array_equal(steer.argmax(1),saved['steer'])
                    output_count+=len(values)*2
    assert total==allresults['prediction_records']
    if output_count:assert output_count==read(s.out/'output_controls/results.json')['prediction_records']
    if surface_count:assert surface_count==read(s.out/'surface_routes_v2/results.json')['prediction_records']
    write(s.out/'independent_verification.json',{'passed':True,'models':reports,'prediction_records':total,
        'output_control_prediction_records':output_count,'surface_route_prediction_records':surface_count,
        'equal_length_hidden_and_surface_aurocs_recomputed':True,'main_frozen_source_hashes_unchanged':True,
        'surface_auxiliary_version':2,'family_isolation':True,'all_models_selection_lock_verified':True})
    print(json.dumps({'passed':True,'prediction_records':total,'output_control_predictions':output_count,'surface_route_predictions':surface_count}),flush=True)


if __name__=='__main__':main()
