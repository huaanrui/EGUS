"""Independent reconstruction, signed-effect uncertainty and research figures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import json
import numpy as np
import torch
from mechanism.identity_parts import (Experiment,COARSE,FIELDS,UTILITY,paired_indices,vector_transmission,orientation_edit)
from mechanism.identity_path import semantic_margin,transmission
from mechanism.identity_causal import variants,metrics,comparison
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from scripts.identity_combo_mechanism_panel import anonymize


def cached(folder,key,field='scores'):
    arrays=[]
    for p in sorted((folder/key).glob('*.npz')):
        with np.load(p) as f:arrays.append(f[field])
    assert arrays,(str(folder),key,field)
    return np.concatenate(arrays)


def sort_key(r):
    s=r['spec'];m=r['metrics']['coref']
    return (r['feasible'],m['both_correct'],m['accuracy'],-s.get('burden',0),-len(s['edits']),s['name']=='baseline')


def check_metrics(vs,s,saved):
    m=metrics(vs,s)
    for f in ('accuracy','both_correct','flip','gap','gold_margin'):assert abs(m[f]-saved[f])<1e-12,(f,m[f],saved[f])


def bootstrap_signed(vs,a,b,r,p,reps,seed):
    t=semantic_margin(vs,b)-semantic_margin(vs,a)
    rr=semantic_margin(vs,b)-semantic_margin(vs,r);pp=semantic_margin(vs,p)-semantic_margin(vs,a)
    by=defaultdict(list)
    for i,row in enumerate(vs):by[row['family_id']].append(i)
    z=np.array([[np.sum(t[ix]**2),np.sum(t[ix]*(rr[ix]+pp[ix])/2)] for ix in by.values()])
    rng=np.random.default_rng(seed);est=[]
    for _ in range(reps):
        q=z[rng.integers(0,len(z),len(z))].sum(0)
        if q[0]>1e-8:est.append(q[1]/q[0])
    point=z[:,1].sum()/z[:,0].sum();null=abs(rng.choice([-1,1],size=(reps,len(z)))@z[:,1])
    return {'alignment':float(point),'ci95':np.quantile(est,[.025,.975]).tolist(),'clusters':len(z),
            'p_sign':float((1+(null>=abs(z[:,1].sum())-1e-12).sum())/(reps+1)),
            'largest_family_energy_share':float(z[:,0].max()/z[:,0].sum())}


def holm(items):
    ordered=sorted(items,key=lambda k:items[k]['p_sign']);running=0
    for i,key in enumerate(ordered):
        running=max(running,min(1,items[key]['p_sign']*(len(ordered)-i)));items[key]['p_holm']=running


def audit_fit(e):
    folder=e.out/'shards';vs=[variants(r) for r in e.values['rank']];ma,fe=paired_indices(vs)
    aa=cached(folder,'fit_A');bb=cached(folder,'fit_B');checked=0
    for site in e.meta['unit_sites']:
        k=site['id'];a=cached(folder,'fit_A','unit_'+k).astype(float);b=cached(folder,'fit_B','unit_'+k).astype(float)
        ga=cached(folder,'grad_'+k+'_A','gradient');gb=cached(folder,'grad_'+k+'_B','gradient')
        mu=a.mean(0);t=semantic_margin(vs,bb)-semantic_margin(vs,aa)
        tr=(t[:,None]*(b-a)*(ga+gb)/2).mean(0)
        s=semantic_margin(vs,aa);d=s[ma]-s[fe]
        gs=(d[:,None]*(ga[ma]*(a[ma]-mu)-ga[fe]*(a[fe]-mu))).mean(0)
        assert np.allclose(mu,e.params[k+'_mean'],atol=1e-6,rtol=1e-6)
        assert np.allclose(tr,e.params[k+'_transfer_score'],atol=1e-12)
        assert np.allclose(gs,e.params[k+'_gap_score'],atol=1e-12)
        n=e.c['mlp_budget'] if site['module']=='mlp' else e.c['head_budget']
        for name,scores,positive in [('transfer_pos',tr,True),('transfer_neg',tr,False),('gap_pos',gs,True),('gap_neg',gs,False)]:
            ii=np.flatnonzero(scores>0 if positive else scores<0)
            expected=ii[np.argsort(-scores[ii] if positive else scores[ii],kind='stable')[:n]]
            assert np.array_equal(expected,e.params[k+'_'+name])
        absolute=np.argsort(-abs(gs),kind='stable')[:2*n];assert np.array_equal(absolute,e.params[k+'_absolute'])
        matched=e.params[k+'_matched'];assert len(matched)==len(absolute) and not set(matched)&set(absolute)
        checked+=len(vs)*4
    for f in [f for f in FIELDS if f['kind'] in ('q','k')]:
        key=f['id'];x=cached(folder,'fit_A','field_'+key).astype(float)
        _,sing,vt=np.linalg.svd(x[ma]-x[fe],full_matrices=False)
        assert np.allclose(x.mean(0),e.params[key+'_mean'],atol=1e-6,rtol=1e-6)
        for r in (1,4):
            u=e.params[key+'_basis'][:,:r]
            assert np.allclose(u@u.T,vt[:r].T@vt[:r],atol=2e-6)
        checked+=len(vs)
    coarse=read(e.out/'coarse_scan.json');a=cached(folder,'coarse_A');b=cached(folder,'coarse_B')
    for item in coarse['results']:
        i=item['index'];r=cached(folder,f'coarse_{i}_restore');p=cached(folder,f'coarse_{i}_transplant')
        assert transmission(vs,a,b,r,p)==item['transmission']
        assert item['signed_alignment']==(item['transmission']['restore_alignment']+item['transmission']['transplant_alignment'])/2
        for f in FIELDS:
            key='field_'+f['id'];av=cached(folder,'coarse_A',key);bv=cached(folder,'coarse_B',key)
            rv=cached(folder,f'coarse_{i}_restore',key);pv=cached(folder,f'coarse_{i}_transplant',key)
            assert vector_transmission(av,bv,rv,pv)==item['destinations'][f['id']]
            if item['site']['layer']>=f['layer'] or (item['site']['position']=='decision' and f['position']=='option_b'):
                assert np.array_equal(rv,bv) and np.array_equal(pv,av)
        checked+=len(vs)*2
    for pos in ('identity','decision'):
        expected=max([r for r in coarse['results'] if r['site']['module']=='mlp' and r['site']['position']==pos],key=lambda r:abs(r['signed_alignment']))
        actual=next(s for s in e.meta['unit_sites'] if s['module']=='mlp' and s['position']==pos)
        assert actual['coarse_index']==expected['index']
    return checked


def audit_selection(e):
    checked=0;screen=read(e.out/'screen_results.json');dev=read(e.out/'selection_frozen.json');short=read(e.out/'shortlist.json')
    for split,items,tasks in [('screen',screen,('coref','reading')),('dev',dev['results'],('coref',*UTILITY))]:
        for item in items:
            for task in tasks:
                vs=e.task_values(split,task);z=cached(e.out/'shards',f'{split}_{task}_{item["spec"]["name"]}')
                check_metrics(vs,z,item['metrics'][task]);checked+=len(vs)
            expected=all(item['metrics'][t]['accuracy']>=items[0]['metrics'][t]['accuracy']-.02-1e-12 for t in tasks)
            assert item['feasible']==expected
    selected=[r['spec'] for r in screen[:3]];counts=defaultdict(int)
    for r in sorted(screen,key=sort_key,reverse=True):
        s=r['spec']
        if not r['feasible'] or s['category'] not in ('unit','qk') or counts[s['site']]>=2:continue
        selected.append(s);counts[s['site']]+=1
        if len(selected)>=3+e.c['shortlist']:break
    assert short[:len(selected)]==selected
    ranked=sorted(screen,key=sort_key,reverse=True)
    fine=[r for r in ranked if r['feasible'] and r['spec']['category'] in ('unit','qk')]
    source_ids={s['id'] for s in e.meta['unit_sites'] if s['position']=='identity'}
    sources=[r for r in fine if r['spec']['site'] in source_ids];terminals=[r for r in fine if r['spec']['site'] not in source_ids]
    tail=short[len(selected):];expected=[]
    if sources and terminals:expected.append(('combo_source_terminal',sources[0]['spec']['edits']+terminals[0]['spec']['edits']))
    if fine:expected.append(('combo_l7_fine',[{'type':'source'}]+fine[0]['spec']['edits']))
    assert [(s['name'],s['edits']) for s in tail]==expected
    assert [r['spec'] for r in dev['results']]==short
    feasible=[r for r in dev['results'] if r['feasible']]
    assert dev['selected']['overall']==max(feasible,key=sort_key)['spec']
    for cat in ('unit','qk','combo'):
        pool=[r for r in feasible if r['spec']['category']==cat]
        assert dev['selected'][cat]==(max(pool,key=sort_key)['spec'] if pool else dev['results'][0]['spec'])
    return checked


def main():
    e=Experiment();e.prepare();assert (e.out/'complete.json').exists()
    checked=audit_fit(e)+audit_selection(e);ev=read(e.out/'evaluation.json');sg=read(e.out/'signed_diagnostics.json')
    datasets={k:e.task_values(k,'coref') for k in ('historical','new_type1','winogender')}
    for task in UTILITY:datasets['utility_'+task]=e.task_values('controls',task)
    datasets['anonymous']=[variants(anonymize(r),permutation=p) for r in e.values['controls'] for p in (0,1)]
    datasets['signed_screen']=e.task_values('screen','coref');datasets['signed_controls']=e.task_values('controls','coref')
    datasets['routes_controls']=e.task_values('controls','coref')
    for task in ('coref','reading','binding'):datasets['part_control_'+task]=e.task_values('controls',task)
    allscores={}
    for path in (e.out/'predictions').glob('*.jsonl'):
        by=defaultdict(dict)
        for r in rows(path):by[r['method']][r['uid']]=r
        vs=datasets[path.stem];scores={}
        for method,items in by.items():
            assert len(items)==len(vs);z=np.array([items[r['uid']]['scores'] for r in vs],dtype=np.float32);scores[method]=z
            assert [items[r['uid']]['prediction'] for r in vs]==z.argmax(1).tolist()
            assert [items[r['uid']]['label'] for r in vs]==[r['label'] for r in vs]
            saved=None
            if path.stem in ev['coref']:saved=ev['coref'][path.stem]['metrics'][method]
            elif path.stem.startswith('utility_'):saved=ev['utilities'][path.stem[8:]][method]
            elif path.stem=='anonymous':saved=ev['anonymous'][method]
            elif path.stem.startswith('part_control_'):saved=read(e.out/'part_controls.json')['metrics'][path.stem[13:]][method]
            if saved is not None:check_metrics(vs,z,saved)
            checked+=len(vs)
        allscores[path.stem]=scores
    signs={};screen_signs={}
    for split in ('screen','controls'):
        ss=allscores['signed_'+split];vs=datasets['signed_'+split]
        for site in e.meta['unit_sites']:
            for group in ('transfer_pos','transfer_neg'):
                key=site['id']+'_'+group
                z=bootstrap_signed(vs,ss['baseline'],ss['source_half'],ss[key+'_restore'],ss[key+'_transplant'],e.c['bootstrap_replicates'],e.c['seed'])
                (signs if split=='controls' else screen_signs)[key]=z
    if 'routes_controls' in allscores:
        ss=allscores['routes_controls'];vs=datasets['routes_controls']
        for group in ('positive','negative'):
            signs['attention18_heads_'+group]=bootstrap_signed(vs,ss['baseline'],ss['source_half'],ss[group+'_restore'],ss[group+'_transplant'],e.c['bootstrap_replicates'],e.c['seed'])
    holm(signs);write(e.out/'signed_uncertainty.json',{'historical_controls':signs,'screen':screen_signs,
                  'correction':'Holm over all historical signed unit/head-group contrasts; bootstrap CIs unadjusted'})
    # Float16 empirical strength matching, evaluated on rank inputs only.
    norms={}
    for f in [f for f in FIELDS if f['kind'] in ('q','k')]:
        key=f['id'];x=torch.from_numpy(cached(e.out/'shards','fit_A','field_'+key).copy())
        a=orientation_edit(x,e.params[key+'_mean'],e.params[key+'_basis'][:,:1],.5)
        b=orientation_edit(x,e.params[key+'_mean'],e.params[key+'_basis'][:,:1],.5,random=e.params[key+'_random'])
        original=x.float().norm(dim=-1);na=a.float().norm(dim=-1);nb=b.float().norm(dim=-1)
        da=(x.float()-a.float()).norm(dim=-1);db=(x.float()-b.float()).norm(dim=-1)
        norms[key]={'max_relative_norm_error':float(torch.max(torch.maximum(abs(na-original),abs(nb-original))/original.clamp(min=1e-12))),
                    'mean_learned_edit_norm':float(da.mean()),'mean_random_edit_norm':float(db.mean()),
                    'max_edit_norm_difference_after_float16':float(abs(da-db).max())}
    write(e.out/'orientation_norm_checks.json',{'rank_only':norms,'note':'exact match before quantization; FP16 errors explicitly reported'})
    chat=read(e.out/'chat_results.json');gp=defaultdict(dict);gv=[variants(r) for r in e.values['generation']]
    for r in rows(e.out/'chat_predictions.jsonl'):gp[r['method']][r['uid']]=r
    for method,items in gp.items():
        for typ in ('strict','line'):
            m=metrics(gv,predictions=[items[r['uid']][typ+'_prediction'] for r in gv])
            for f in ('accuracy','both_correct','flip','gap'):assert m[f]==chat['metrics'][method][typ][f]
        checked+=len(gv)
    chat_comp={name:{f:comparison(gv,[gp[name][r['uid']]['strict_prediction'] for r in gv],
               [gp['source_half'][r['uid']]['strict_prediction'] for r in gv],f,e.c['bootstrap_replicates'],e.c['seed'])
              for f in ('accuracy','both_correct','flip')} for name in ('overall','baseline','whole_four_heads')}
    write(e.out/'chat_comparisons.json',chat_comp)
    extra={}
    if 'part_control_coref' in allscores:
        for task in ('coref','reading','binding'):
            ss=allscores['part_control_'+task];vs=datasets['part_control_'+task]
            extra[task]={name:{f:comparison(vs,ss['selected_unit'].argmax(1),ss[name].argmax(1),f,e.c['bootstrap_replicates'],e.c['seed'])
                for f in ('accuracy','both_correct')} for name in ('matched_unit','whole_raw','whole_centered')}
        write(e.out/'part_control_comparisons.json',extra)
    frozen=read(e.out/'plan_frozen.json')
    for p,h in {**frozen['implementation'],**frozen['inputs']}.items():assert sha(p)==h
    if (e.out/'routes_declaration.json').exists():
        d=read(e.out/'routes_declaration.json')['identity'];early=read(e.out/'routes_plan.json')
        assert sha(ROOT/'scripts/identity_parts_routes.py')==d['script']==early['script_sha256']
        assert sha(ROOT/'research/2026-10-15_identity_parts_routes_protocol.md')==d['protocol']==early['protocol_sha256']
    v=read(e.out/'verification.json');v.update({'independent_records_recomputed':checked,'fit_signs_and_subspaces_recomputed':True,
         'selection_recomputed':True,'all_recorded_behavior_metrics_match':True,'routes_source_hashes_match':True})
    write(e.out/'verification.json',v)
    selected=read(e.out/'selection_frozen.json')['selected']
    lines=['# Signed parts and local reprocessing results','',
      'All external behavior sets had been inspected previously. This is exploratory. Source-effect sign and identity-gap sign are distinct; neither inherently defines a bias neuron.','',
      '## Frozen selections','',json.dumps(selected,indent=2),'','## Main coreference retests','',
      '|Dataset|Method|Accuracy|Both correct|Flip|Margin gap RMS|','|---|---|---:|---:|---:|---:|']
    for split,r in ev['coref'].items():
        for method,m in r['metrics'].items():lines.append(f"|{split}|{method}|{m['accuracy']:.2%}|{m['both_correct']:.2%}|{m['flip']:.2%}|{m['identity_gap']['rms']:.3f}|")
    lines+=['','## Behavior uncertainty vs old L7','',json.dumps({k:r['comparisons'] for k,r in ev['coref'].items()},indent=2),
      '','## Exact signed effects','',json.dumps(signs,indent=2),
      '','## Local and destination localization','',json.dumps(read(e.out/'coarse_scan.json'),indent=2),
      '','## Active utilities','',json.dumps(ev['utilities'],indent=2),
      '','## Float16 orientation control checks','',json.dumps(norms,indent=2),
      '','## Actual native-chat short generation','',json.dumps(chat,indent=2),json.dumps(chat_comp,indent=2)]
    for name in ('routes','part_controls'):
        if (e.out/(name+'.json')).exists():lines+=['','## '+name,'',json.dumps(read(e.out/(name+'.json')),indent=2)]
    lines+=['','## Independent verification','',json.dumps(v,indent=2)]
    report=ROOT/'research/2026-10-15_identity_parts_findings.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    co=read(e.out/'coarse_scan.json')['results'];labels=[f"L{r['site']['layer']} {'A' if r['site']['module']=='attention' else 'M'}\n{r['site']['position']}" for r in co]
    vals=[r['signed_alignment'] for r in co];axes[0].bar(range(len(vals)),vals,color=['#3b82f6' if x>=0 else '#ef4444' for x in vals]);axes[0].axhline(0,color='black',lw=.7)
    axes[0].set_xticks(range(len(vals)),labels,rotation=80,fontsize=5);axes[0].set(title='Rank-only module effects',ylabel='Signed source-effect alignment')
    labels=list(signs);vals=np.array([signs[k]['alignment'] for k in labels]);ci=np.array([signs[k]['ci95'] for k in labels])
    axes[1].bar(range(len(vals)),vals,color=['#3b82f6' if x>=0 else '#ef4444' for x in vals]);axes[1].errorbar(range(len(vals)),vals,yerr=np.maximum(0,np.stack([vals-ci[:,0],ci[:,1]-vals])),fmt='none',ecolor='black',capsize=2)
    axes[1].set_xticks(range(len(vals)),labels,rotation=80,fontsize=5);axes[1].axhline(0,color='black',lw=.7);axes[1].set(title='Historical64 exact signed groups',ylabel='Alignment (family 95% CI)')
    names=['baseline','source_half','unit','overall','whole_four_heads'];x=np.arange(3);width=.15
    for i,name in enumerate(names):axes[2].bar(x+(i-2)*width,[ev['coref'][s]['metrics'][name]['both_correct']*100 for s in ev['coref']],width,label=name)
    axes[2].set_xticks(x,['Type2','Type1','Winogender']);axes[2].set(title='Historical exploratory behavior',ylabel='Both identities correct (%)');axes[2].legend(fontsize=6)
    fig.tight_layout();fig.savefig(e.out/'identity_parts.png',dpi=180);fig.savefig(e.out/'identity_parts.pdf');plt.close(fig)
    write(e.out/'completed_followup.json',{'report_sha256':sha(report),'verification_sha256':sha(e.out/'verification.json'),'historical_exploratory':True})
    print(json.dumps({'selected':selected,'historical':ev['coref']['historical'],'signed':signs,'chat':chat,'verified':v},indent=2))


if __name__=='__main__':main()
