"""Independent frozen-candidate coverage and saved-output audit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from mechanism.identity_best_combo import signature,score_order
from mechanism.identity_replication import extended_metrics
from mechanism.identity_causal import variants


if __name__=='__main__':
    folder=ROOT/'results/identity_best_combo_20261016_v1';lock=read(folder/'all_models_frozen.json');total=0;result={}
    numerical=read(folder/'numerical_v2_declaration.json')
    assert numerical['executor_sha256']==sha(ROOT/'scripts/run_identity_best_combo_numerical_v2.py')
    assert numerical['protocol_sha256']==sha(ROOT/'research/2026-10-16_identity_best_combo_numerical_v2.md')
    assert numerical['test_sha256']==sha(ROOT/'tests/test_identity_best_combo_numerical_v2.py')
    for n,h in numerical['already_frozen_selections'].items():assert sha(folder/n/'selection_frozen.json')==h
    for name,selection_hash in lock['selections'].items():
        out=folder/name;s=read(out/'selection_frozen.json');assert sha(out/'selection_frozen.json')==selection_hash
        assert sha(out/'parameters.npz')==s['parameter_sha256'];assert sha(out/'parameter_meta.json')==s['meta_sha256']
        plan=read(out/'plan_frozen.json')
        for p,h in {**plan['implementation'],**plan['inputs']}.items():assert sha(p)==h
        dev=read(out/'dev_results.json');sigs={signature(z['spec']) for z in dev}
        for k,z in s['references'].items():assert signature(z) in sigs,(name,k,'reference excluded from dev')
        if name=='qwen3':assert 'ref_old_combo' in s['references']
        base=next(z for z in dev if not z['spec']['edits'])
        for z in dev:
            feasible=all(z['utilities'][t]['accuracy']>=base['utilities'][t]['accuracy']-.02-1e-12 for t in base['utilities'])
            assert z['feasible']==feasible
        assert max(dev,key=score_order)==s['best_task']
        assert max([z for z in dev if z['feasible']],key=score_order)==s['best_constrained']
        source=ROOT/'data/prepared/identity_combo_20261016_v1'
        families={split:{r['family_id'] for r in rows(source/(split+'.jsonl'))} for split in ('rank','screen','dev','historical')}
        for a in families:
            for b in families:
                if a!=b:assert not families[a]&families[b]
        # Reconstruct discovered means, both objectives and ranked coordinate groups from saved finite gradients.
        rankvalues=[variants(r,permutation=p) for r in rows(source/'rank.jsonl') for p in (0,1)]
        paths=sorted((out/'rank_gradients').glob('*.npz'));assert len(paths)==len(rankvalues)==256
        scale_histogram=defaultdict(int);old_cache=0;score=[]
        for p in paths:
            with np.load(p) as z:
                for k in z.files:
                    if k=='scores' or k.endswith(('_a','_g')):assert np.isfinite(z[k]).all(),(name,p,k)
                score.append(z['scores'])
                if 'gradient_loss_scale' in z:
                    assert str(z['execution_v2_sha256'])==numerical['executor_sha256']
                    scale_histogram[int(z['gradient_loss_scale'].item())]+=1
                else:old_cache+=1;scale_histogram[1024]+=1
        scores=np.concatenate(score);meta=read(out/'parameter_meta.json')
        pairing=defaultdict(dict)
        for i,r in enumerate(rankvalues):pairing[(r['pair_id'],r['permutation'])][r['gender']]=i
        male=np.array([v['male'] for v in pairing.values()]);female=np.array([v['female'] for v in pairing.values()])
        lexical=np.array([1 if r['choices'][0]==min(r['choices']) else -1 for r in rankvalues])
        gold=np.array([1 if r['gold']==min(r['choices']) else -1 for r in rankvalues])
        delta=((scores[:,0]-scores[:,1])*lexical)[male]-((scores[:,0]-scores[:,1])*lexical)[female]
        projections=defaultdict(list)
        for key,z in meta['selected_site_ranking'].items():
            site=z['site'];projections[(site['module'],site['layer'])].append((key,site))
        with np.load(out/'parameters.npz') as parameters:
            for (module,layer),sites in projections.items():
                stem=('mlp' if module=='mlp' else 'attention')+str(layer);aa=[];gg=[]
                for p in paths:
                    with np.load(p) as z:aa.append(z[stem+'_a']);gg.append(z[stem+'_g'])
                aa=np.asarray(aa,dtype=np.float32);gg=np.asarray(gg,dtype=np.float32)
                for key,site in sites:
                    pi=meta['positions'].index(site['position']);a=aa[:,pi];g=gg[:,pi]
                    if module=='head':
                        a=a.reshape(256,16,-1)[:,site['head']];g=g.reshape(256,16,-1)[:,site['head']]
                    mu=a.mean(0);contribution=(a-mu)*g
                    objective={'accuracy':(contribution*gold[:,None]).mean(0),
                        'gap':(delta[:,None]*(contribution[male]-contribution[female])).mean(0)}
                    np.testing.assert_allclose(mu,parameters[key+'_mean'],rtol=1e-6,atol=1e-6)
                    for label,v in objective.items():
                        np.testing.assert_allclose(v,parameters[key+'_'+label+'_score'],rtol=1e-6,atol=1e-8)
                        for budget in ((64,128) if module=='mlp' else (16,32)):
                            for positive in (True,False):
                                eligible=np.flatnonzero(v>0 if positive else v<0)
                                selected=eligible[np.argsort(-v[eligible] if positive else v[eligible],kind='stable')[:budget]]
                                suffix=label+f'_{budget}_'+('pos' if positive else 'neg')
                                np.testing.assert_array_equal(selected,parameters[key+'_'+suffix])
        ev=read(out/'evaluation.json');path=out/'test_predictions.jsonl';assert sha(path)==ev['prediction_sha256']
        by=defaultdict(list)
        for r in rows(path):
            assert r['prediction']==int(np.argmax(r['scores']));assert r['gold']==r['choices'][r['label']]
            by[r['method']].append(r);total+=1
        assert set(by)==set(ev['methods'])
        for k,rr in by.items():
            assert len(rr)==584;assert len({r['uid'] for r in rr})==584
            m=extended_metrics(rr,np.array([r['scores'] for r in rr]))
            for field in ('accuracy','both_correct','flip','both_wrong'):assert abs(m[field]-ev['metrics'][k][field])<1e-12
        old=defaultdict(dict)
        for r in rows(ROOT/'results/identity_replication_20261015_v1'/name/'predictions/historical.jsonl'):
            old[r['method']][r['uid']]=r
        reference_deltas={}
        for new,previous in [('baseline','baseline'),('ref_selected','selected'),('ref_fine_combo','fine_combo')]+([('ref_old_combo','old_combo')] if name=='qwen3' else []):
            delta=0.
            for r in by[new]:
                prior=old[previous][r['uid']]
                assert prior['prediction']==r['prediction'];assert prior['label']==r['label'];assert prior['choices']==r['choices']
                delta=max(delta,float(np.max(abs(np.array(prior['scores'])-r['scores']))))
            reference_deltas[new]=delta
        result[name]={'all_old_references_participated_in_dev':True,'selection_replayed':True,'family_isolation':True,
            'saved_predictions_recomputed':True,'plan_and_parameters_unchanged':True,'reference_count':len(s['references']),
            'previous_reference_predictions_exact':True,'previous_reference_max_logit_delta':reference_deltas,
            'all_rank_gradients_finite':True,'discovery_means_scores_and_groups_recomputed':True,
            'gradient_loss_scale_histogram':dict(scale_histogram),'original_cache_rows':old_cache}
    write(folder/'independent_verification.json',{'models':result,'prediction_records':total,
        'verifier_sha256':sha(Path(__file__)),'all_checks_passed':True})
    print('Independent verification passed',total,'predictions')
