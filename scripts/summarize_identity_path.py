"""Independent score/selection recomputation and a descriptive research report."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import json
import numpy as np
from mechanism.identity_path import (Experiment, transmission, score_order, semantic_margin,
                                     bootstrap_continuous, record_rows)
from mechanism.identity_combo import task_row
from mechanism.identity_causal import variants, metrics, comparison
from mechanism.fairsteer_author import read, rows, write, sha, ROOT
from scripts.identity_combo_mechanism_panel import anonymize
from scripts.identity_path_active_controls import active


def cached(out,key,field='scores'):
    arrays=[]
    for path in sorted((out/'shards'/key).glob('*.npz')):
        with np.load(path) as f:arrays.append(f[field])
    assert arrays,key
    return np.concatenate(arrays)


def recompute_selection(e):
    layer=read(e.out/'layer_scan.json');head=read(e.out/'head_scan.json')
    assert layer==sorted(layer,key=score_order,reverse=True)
    assert head==sorted(head,key=score_order,reverse=True)
    assert read(e.out/'head_shortlist.json')==head[:e.c['head_shortlist']]
    screen=read(e.out/'screen_heads.json');sel=read(e.out/'selection_frozen.json')
    assert sel['heads']==[r['site'] for r in sorted(screen,key=score_order,reverse=True)[:e.c['final_heads']]]
    checks=0
    for split,items,keyfn in [
        ('rank',layer,lambda r:f"layer_{r['site']['layer']}_{r['site']['position']}"),
        ('rank',head,lambda r:f"head_{r['site']['layer']}_{r['site']['position']}_{r['site']['head']}"),
        ('screen',read(e.out/'head_shortlist.json'),lambda r:'screen_head_'+str(read(e.out/'head_shortlist.json').index(r)))]:
        vs=[variants(r) for r in e.values[split]] if split=='rank' else e.task_values(split,'coref')
        a=cached(e.out,split+'_A');b=cached(e.out,split+'_B')
        for item in items:
            key=keyfn(item);r=cached(e.out,key+'_restore');p=cached(e.out,key+'_transplant')
            t=transmission(vs,a,b,r,p)
            saved=item['transmission'] if split=='rank' else next(z['transmission'] for z in screen if z['site']==item['site'])
            for field,value in t.items():
                assert value==saved[field] or abs(value-saved[field])<1e-12
            checks+=len(vs)*2
    vs=e.task_values('dev','coref');a=cached(e.out,'dev_A');b=cached(e.out,'dev_B')
    for c in sel['candidates']:
        k=c['k'];t=transmission(vs,a,b,cached(e.out,f'dev_set{k}_restore'),cached(e.out,f'dev_set{k}_transplant'))
        assert t==c['transmission'];checks+=len(vs)*2
    assert sel['selected']==max(sel['candidates'],key=lambda r:(score_order(r),-r['k']))
    return checks


def edge_ci(vs,a,b,z,reps,seed):
    t=semantic_margin(vs,b)-semantic_margin(vs,a);d=semantic_margin(vs,z)-semantic_margin(vs,a)
    groups=defaultdict(list)
    for i,r in enumerate(vs):groups[r['family_id']].append(i)
    mm=np.array([[np.sum(t[ix]**2),np.sum(t[ix]*d[ix]),np.sum((t[ix]-d[ix])**2)] for ix in groups.values()])
    rng=np.random.default_rng(seed);est=[]
    for _ in range(reps):
        v=mm[rng.integers(0,len(mm),len(mm))].sum(0)
        if v[0]>1e-8:est.append([v[1]/v[0],1-v[2]/v[0]])
    return dict(zip(('alignment','reproduction_fidelity'),np.quantile(est,[.025,.975],axis=0).T.tolist()))


def main():
    e=Experiment();e.prepare();assert (e.out/'complete.json').exists()
    checked=recompute_selection(e);ev=read(e.out/'evaluation.json');ctl=read(e.out/'controls.json');edges=read(e.out/'edges.json');chat=read(e.out/'chat_results.json')
    datasets={k:e.task_values(k,'coref') for k in ('historical','new_type1','winogender')}
    datasets['source_controls']=datasets['edges']=datasets['token_routes']=datasets['receiver_edges']=e.task_values('controls','coref')
    datasets['record']=record_rows(datasets['source_controls'])
    datasets['anonymous']=[variants(anonymize(r),permutation=p) for r in e.values['controls'] for p in (0,1)]
    for t in ('literal','reading','binding','role_swap'):
        datasets['utility_'+t]=[task_row(r,t,p) for r in e.values['controls'] for p in (0,1)]
    for t in ('literal','reading','binding','record'):
        native=datasets['record'] if t=='record' else datasets['utility_'+t]
        datasets['active_'+t]=[active(r) for r in native]
    allscores={};active_report=read(e.out/'active_controls.json') if (e.out/'active_controls.json').exists() else None
    for path in sorted((e.out/'predictions').glob('*.jsonl')):
        lookup=defaultdict(dict)
        for r in rows(path):lookup[r['method']][r['uid']]=r
        vs=datasets[path.stem];scores={}
        for method,items in lookup.items():
            assert len(items)==len(vs)
            z=np.array([items[r['uid']]['scores'] for r in vs],dtype=np.float32);scores[method]=z
            assert [items[r['uid']]['prediction'] for r in vs]==z.argmax(1).tolist()
            assert [items[r['uid']]['label'] for r in vs]==[r['label'] for r in vs]
            m=metrics(vs,z)
            saved=None
            if path.stem in ev:saved=ev[path.stem]['metrics'][method]
            elif path.stem.startswith('utility_'):saved=ctl['utilities'][path.stem[8:]][method]
            elif path.stem=='anonymous':saved=ctl['anonymous'][method]
            elif path.stem=='edges':saved=edges['results'][method]['metrics']
            elif path.stem=='record' and method in ctl['record']:saved=ctl['record'][method]
            elif path.stem.startswith('active_') and active_report is not None:
                saved=active_report['results'][path.stem[7:]]['metrics'].get(method)
            if saved is not None:
                for f in ('accuracy','both_correct','flip','gap','gold_margin'):assert abs(m[f]-saved[f])<1e-12
            checked+=len(vs)
        allscores[path.stem]=scores
    cs=allscores['source_controls'];cv=datasets['source_controls'];ci={}
    for name in ctl['source_controls']:
        ci[name]=bootstrap_continuous(cv,cs['baseline'],cs[name],cs[name+'_restore'],cs[name+'_transplant'],e.c['bootstrap_replicates'],e.c['seed'])
    es=allscores['edges'];eci={name:edge_ci(cv,es['baseline'],es['source_half'],z,e.c['bootstrap_replicates'],e.c['seed']) for name,z in es.items()}
    write(e.out/'supplementary_uncertainty.json',{'source_controls':ci,'edges':eci,'status':'family bootstrap; descriptive multiple contrasts unadjusted'})
    causal={}
    for split in ev:
        vs=datasets[split];ss=allscores[split]
        causal[split]={a+'_vs_'+b:{f:comparison(vs,ss[a].argmax(1),ss[b].argmax(1),f,e.c['bootstrap_replicates'],e.c['seed']) for f in ('accuracy','both_correct','flip')}
              for a,b in [('source_half','selected_restore'),('selected_transplant','source_half'),('selected_restore','baseline')]}
    write(e.out/'causal_behavior_comparisons.json',causal)
    if 'token_routes' in allscores:
        ts=allscores['token_routes']
        tci={name:edge_ci(cv,ts['baseline'],ts['source_half'],z,e.c['bootstrap_replicates'],e.c['seed']) for name,z in ts.items()}
        write(e.out/'token_routes_uncertainty.json',tci)
    if 'receiver_edges' in allscores:
        rs=allscores['receiver_edges']
        rci={name:edge_ci(cv,rs['baseline'],rs['source_half'],z,e.c['bootstrap_replicates'],e.c['seed']) for name,z in rs.items()}
        write(e.out/'receiver_edges_uncertainty.json',rci)
    gv=[variants(r) for r in e.values['generation']];gl=defaultdict(dict)
    for r in rows(e.out/'chat_predictions.jsonl'):gl[r['method']][r['uid']]=r
    genpred={}
    for method,items in gl.items():
        for typ in ('strict','line'):
            m=metrics(gv,predictions=[items[r['uid']][typ+'_prediction'] for r in gv])
            for f in ('accuracy','both_correct','flip','gap'):assert abs(m[f]-chat['metrics'][method][typ][f])<1e-12
        genpred[method]=[items[r['uid']]['strict_prediction'] for r in gv]
        checked+=len(gv)
    gc={a+'_vs_'+b:{f:comparison(gv,genpred[a],genpred[b],f,e.c['bootstrap_replicates'],e.c['seed']) for f in ('accuracy','both_correct','flip')}
        for a,b in [('source_half','baseline'),('selected_half','baseline'),('source_half_restore','source_half')]}
    write(e.out/'chat_comparisons.json',gc)
    v=read(e.out/'verification.json');v.update({'independent_selection_and_metric_recomputation':True,
               'independent_records_recomputed':checked,'selection_sha256':sha(e.out/'selection_frozen.json')})
    write(e.out/'verification.json',v)
    s=read(e.out/'selection_frozen.json')
    lines=['# Layer-7 transmission findings','',
        'All behavior evaluation sets had been inspected previously. This is an exploratory mechanistic follow-up, not fresh confirmation.',
        'Fidelity measures reproduction of the layer7half effect; it is not a percentage of bias explained. Continuous contrasts are not multiplicity adjusted.','',
        '## Frozen receivers','',json.dumps(s,ensure_ascii=False,indent=2),'',
        '## Behavior and transmission','',
        '|Dataset|Method|Accuracy|Both correct|Flip|','|---|---|---:|---:|---:|']
    for split,r in ev.items():
        for method,m in r['metrics'].items():
            lines.append(f"|{split}|{method}|{m['accuracy']:.2%}|{m['both_correct']:.2%}|{m['flip']:.2%}|")
    lines+=['','Two-way mediation estimates and uncertainty:','',json.dumps({k:{n:r[n] for n in ('selected','matched')} for k,r in ev.items()},indent=2),
        '','Causal behavior contrasts:','',json.dumps(causal,indent=2),
        '','## Source controls and utility','',json.dumps(ctl,indent=2),
        '','## Q/K/V and controlled direct edges','',json.dumps(edges,indent=2),
        '','## Native chat short generation','',json.dumps(chat,indent=2),
        '',json.dumps(gc,indent=2),
        '','## Independent audit','',json.dumps(v,indent=2)]
    if (e.out/'active_controls.json').exists():
        lines+=['','## Controls with active question-pronoun receiver','',json.dumps(read(e.out/'active_controls.json'),indent=2)]
    if (e.out/'token_routes.json').exists():
        lines+=['','## Fixed token-group transmission','',json.dumps(read(e.out/'token_routes.json'),indent=2)]
    if (e.out/'receiver_edges.json').exists():
        lines+=['','## Post-QKV exploratory receiver edges','',json.dumps(read(e.out/'receiver_edges.json'),indent=2)]
    lines+=['','## Subset documentation clarification','',json.dumps(read(e.out/'subset_clarification.json'),indent=2)]
    report=ROOT/'research/2026-10-15_identity_path_findings.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Standard scientific figure, with rank-only search panels clearly marked.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layer=read(e.out/'layer_scan.json');head=read(e.out/'head_scan.json')
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for pos in ('decision','question_identity'):
        items=sorted([r for r in layer if r['site']['position']==pos],key=lambda r:r['site']['layer'])
        axes[0].plot([r['site']['layer'] for r in items],[r['transmission']['fidelity'] for r in items],label=pos,marker='.')
    axes[0].set(title='Rank64: layer receiver mediation',xlabel='Layer (zero based)',ylabel='Two-way fidelity');axes[0].legend(fontsize=8)
    top=head[:12];labels=[f"L{r['site']['layer']}H{r['site']['head']}\n{r['site']['position']}" for r in top]
    axes[1].bar(range(len(top)),[r['transmission']['fidelity'] for r in top]);axes[1].set_xticks(range(len(top)),labels,rotation=70,fontsize=6)
    axes[1].set(title='Rank64: top head candidates',ylabel='Two-way fidelity')
    names=['baseline','source_half','selected_restore','selected_transplant','selected_half']
    x=np.arange(3);width=.15
    for i,n in enumerate(names):axes[2].bar(x+(i-2)*width,[ev[k]['metrics'][n]['both_correct']*100 for k in ev],width,label=n)
    axes[2].set_xticks(x,['Historical Type2','Historical Type1','Historical Winogender'],fontsize=7)
    axes[2].set(title='Exploratory behavior retests',ylabel='Both identities correct (%)');axes[2].legend(fontsize=6)
    fig.tight_layout();fig.savefig(e.out/'identity_path.png',dpi=180);fig.savefig(e.out/'identity_path.pdf');plt.close(fig)
    print(json.dumps({'report':str(report),'independent_records':checked,'selected':s['selected'],
        'historical_mediation':ev['historical']['selected'],'chat':chat},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
