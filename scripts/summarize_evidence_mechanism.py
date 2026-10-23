"""Independent decision audit and descriptive reporting, no parameter selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import csv
import json
import time
import numpy as np
from scipy.special import expit,softmax
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mechanism.fairsteer_author import ROOT,read,rows,sha,write
from mechanism.closed_transfer import bootstrap_difference

OUT=ROOT/'results/evidence_mechanism_20261014_v1'
DATA=ROOT/'data/prepared/evidence_mechanism_20261014_v1'
FIT=ROOT/'results/fairsteer_improve_20261013_v2'
TRANSFER=ROOT/'results/closed_transfer_20261014_v2'


def entropy(score):
    p=softmax(np.asarray(score,np.float64),axis=1)
    return -np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)


def holm(values):
    order=sorted(values,key=values.get);out={};previous=0.
    for i,name in enumerate(order):
        previous=max(previous,min(1.,(len(order)-i)*values[name]));out[name]=previous
    return out


def main():
    saved=read(OUT/'results.json');selection=read(OUT/'selection_frozen.json');plan=read(OUT/'plan_frozen.json')
    for n,h in plan['identity']['files'].items():assert sha(ROOT/n)==h,(n,'frozen implementation changed')
    for name,h in plan['identity']['data'].items():assert sha(DATA/f'{name}.jsonl')==h
    records=rows(OUT/'predictions.jsonl');lookup={(r['dataset'],r['method'],r['uid']):r for r in records}
    assert len(lookup)==len(records)
    probes=dict(np.load(FIT/'probes.npz'));audit_count=0;entropy_results={};flat=[];forced={}
    for key,methods in saved['metrics'].items():
        vs=rows(DATA/f'{key}.jsonl');path=(TRANSFER if key.startswith('wino') else OUT)/f'cache_{key}.npz'
        cache=dict(np.load(path));assert cache['uids'].tolist()==[r['uid'] for r in vs]
        x=cache['features'];prob={n:expit(x@probes[n+'_coef']+float(probes[n+'_intercept'])) for n in ('original','expanded','evidence')}
        g=(1-prob['expanded'])*prob['evidence']>.25
        gates={'fairsteer':1-prob['original']>.5,'evidence_steer':g}
        base=cache['base'];manual={'baseline':base.copy()}
        for name,gate in gates.items():
            s=base.copy();s[gate]=cache['cf1' if name=='fairsteer' else 'cf15'][gate];manual[name]=s
        s=base.astype(np.float64).copy()
        for i,r in enumerate(vs):
            if g[i] and r['unknown_index']>=0:s[i,r['unknown_index']]+=selection['matched_offset']['amount']
        manual['same_gate_unknown_offset']=s
        if 'no_context' in cache:
            manual['evidence_instruction']=cache['instruction']
            for name in ('cad','gated_cad'):
                mask=prob['evidence']<=.25 if name=='gated_cad' else np.ones(len(vs),bool)
                for beta in (0.25,0.5,1.):
                    manual[f'diagnostic_{name}_{beta:g}']=base.astype(np.float64)+beta*(base.astype(np.float64)-cache['no_context'].astype(np.float64))*mask[:,None]
                beta=selection['cad'][name]['selected']['beta']
                manual[name]=base.astype(np.float64)+beta*(base.astype(np.float64)-cache['no_context'].astype(np.float64))*mask[:,None]
            for seed in (20261014,20261015,20261016):
                s=base.copy();use=g&np.array([r['permutation']==0 for r in vs]);s[use]=cache[f'random_{seed}'][use];manual[f'random_{seed}']=s
        predictions={n:s.argmax(1) for n,s in manual.items()}
        for name,gate in [('evidence_direct',prob['evidence']>.25),('same_gate_direct',g)]:
            p=base.argmax(1).copy()
            for i,r in enumerate(vs):
                if gate[i] and r['unknown_index']>=0:p[i]=r['unknown_index']
            predictions[name]=p
        for name,pred in predictions.items():
            for r,p in zip(vs,pred):assert lookup[(key,name,r['uid'])]['prediction']==int(p)
            audit_count+=len(vs)
            for pos in sorted({r['permutation'] for r in vs}):
                if name.startswith('random_') and pos!=0:continue
                mask=np.array([r['permutation']==pos for r in vs]);sub=[r for r,m in zip(vs,mask) if m];labels=np.array([r['label'] for r in sub])
                m=methods[name][str(pos)];assert np.mean(pred[mask]==labels)==m['accuracy']
                flat.append({'dataset':key,'method':name,'permutation':pos,**{k:m[k] for k in ('rows','accuracy','ambig_accuracy','clear_accuracy','clear_false_unknown','pro_anti_gap')}})
        main=np.array([r['permutation']==0 for r in vs]);detail={}
        if key.startswith('wino'):
            sub=[r for r,m in zip(vs,main) if m];labels=np.array([r['label'] for r in sub]);forced[key]={}
            for cf in ('cf1','cf15'):
                pred=cache[cf][main].argmax(1);ref=base[main].argmax(1)
                forced[key][cf]={'accuracy':float(np.mean(pred==labels)),
                    'prediction_change_rate':float(np.mean(pred!=ref)),
                    'comparison':bootstrap_difference(sub,pred,ref,2000,20261014)}
        entity_indices=[[j for j in range(len(r['choices'])) if j!=r['unknown_index']] for r in vs]
        ebase=np.array([base[i,j] for i,j in enumerate(entity_indices)])
        for name,s in manual.items():
            es=np.array([s[i,j] for i,j in enumerate(entity_indices)]);de=entropy(es)-entropy(ebase)
            ds=entropy(s)-entropy(base)
            detail[name]={'candidate_entropy_delta':float(ds[main].mean()),'entity_entropy_delta':float(de[main].mean()),
                'conditional_entity_accuracy_clear':float(np.mean([
                    entity_indices[i][int(np.argmax(es[i]))]==vs[i]['label'] for i in np.flatnonzero(main) if vs[i]['condition']=='clear'])) if any(r['condition']=='clear' for r in vs) else None}
        # Descriptive correlations do not demonstrate what the probe causally encodes.
        margins=x.astype(np.float64)@probes['evidence_coef'].astype(np.float64)+float(probes['evidence_intercept'])
        corr=spearmanr(margins[main],entropy(base)[main]).statistic
        detail['probe_vs_candidate_entropy_spearman']=float(corr) if np.isfinite(corr) else None
        entropy_results[key]=detail
    assert audit_count==len(records)
    # Neutral knowledge controls use historic four-choice logits, not extra generation.
    neutral=dict(np.load(FIT/'utility_baseline/test.npz'));x=neutral['features'];base=neutral['scores']
    split=read(ROOT/'data/prepared/fairsteer_improve_20261013_v2/splits.json')['mmlu_indices']['test']
    data=rows(ROOT/'data/prepared/fairsteer_author_qwen3_v1/mmlu.jsonl');labels=np.array([data[i]['answer'] for i in split])
    p={n:expit(x@probes[n+'_coef']+float(probes[n+'_intercept'])) for n in ('original','expanded','evidence')}
    utility={}
    for name,alpha,gate in [('unconditional',1.5,np.ones(len(x),bool)),('fairsteer',1.,1-p['original']>.5),
                           ('evidence_steer',1.5,(1-p['expanded'])*p['evidence']>.25)]:
        cf=np.load(FIT/f'counterfactuals/test_dim_{alpha:g}_utility.npz')['scores'];s=base.copy();s[gate]=cf[gate]
        utility[name]={'rows':len(x),'gate_rate':float(gate.mean()),'base_accuracy':float(np.mean(base.argmax(1)==labels)),
            'accuracy':float(np.mean(s.argmax(1)==labels)),'entropy_delta':float(np.mean(entropy(s)-entropy(base))),
            'prediction_change_rate':float(np.mean(s.argmax(1)!=base.argmax(1)))}
    write(OUT/'entropy_diagnostics.json',{'tasks':entropy_results,'historic_mmlu':utility,
        'scope':'Restricted-candidate entropy, not full-vocabulary confidence; correlations are descriptive. MMLU 128 is small and author-screened.'})
    write(OUT/'forced_direction_diagnostics.json',{'results':forced,
        'scope':'Exploratory cached unconditional alpha1/1.5 interventions. Distinguishes gate nonactivation from the direction effect; no target-selected strength.'})
    confirms=saved['comparisons']['synth_confirmation']
    primary=['cad_vs_baseline_clear','gated_cad_vs_baseline_clear','evidence_steer_vs_fairsteer_all','same_gate_direct_vs_evidence_steer_all']
    corrected=holm({k:confirms[k]['p_sign_randomization'] for k in primary})
    write(OUT/'primary_holm.json',{'family':'Four prespecified primary confirmation contrasts','adjusted_p':corrected})
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    cal=read(OUT/'calibration/results.json')['records'];stress={}
    for dataset in sorted({r['dataset'] for r in cal}):
        stress[dataset]={}
        for method in ('frozen','scalar_moments','feature_diagonal_moments','score_quantiles'):
            rr=[r for r in cal if r['dataset']==dataset and r['method']==method and r['threshold_name']=='legacy_fixed' and r['adaptation_mixture']!='natural']
            if rr:stress[dataset][method]={'recall_range':[min(r['insufficient_recall'] for r in rr),max(r['insufficient_recall'] for r in rr)],
                'fpr_range':[min(r['sufficient_fpr'] for r in rr),max(r['sufficient_fpr'] for r in rr)]}
    write(OUT/'calibration/stress_summary.json',stress)
    lines=['# Evidence mechanism pilot v1','',
        'Source-dev selection: unconditional CAD beta=0; gated CAD beta='+str(selection['cad']['gated_cad']['selected']['beta'])+'.',
        'Matched same-gate unknown offset='+str(selection['matched_offset']['amount'])+'. No target selection.',
        'Historic BBQ/SynthBias test results are exploratory. New confirmation: 320 reserved questions, 3 option rotations, correlated gender/template groups.','']
    names=['baseline','fairsteer','evidence_steer','same_gate_unknown_offset','same_gate_direct','evidence_direct','evidence_instruction','cad','gated_cad']
    for dataset in ('bbq_test','synth_test','synth_confirmation'):
        lines += ['## '+dataset,'','|Method|Overall %|Ambiguous %|Sufficient %|Sufficient false unknown %|','|---|---:|---:|---:|---:|']
        for name in names:
            m=saved['metrics'][dataset][name]['0'];lines.append('|'+name+'|'+ '|'.join(f'{100*m[k]:.2f}' for k in ('accuracy','ambig_accuracy','clear_accuracy','clear_false_unknown'))+'|')
        lines.append('')
    lines += ['## Confirmation comparisons','']
    for name in primary:
        r=confirms[name];lines.append(f"- {name}: {100*r['delta']:.2f}pp, CI [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}], Holm p={corrected[name]:.4f}.")
    lines += ['','## Unlabeled adaptation, natural adaptation mixtures','',
        '|Target|Method|AUROC|Insufficient recall %|Sufficient false positive %|','|---|---|---:|---:|---:|']
    for r in cal:
        if r['adaptation_mixture']=='natural' and r['threshold_name']=='legacy_fixed':
            lines.append(f"|{r['dataset']}|{r['method']}|{r['auroc']:.4f}|{100*r['insufficient_recall']:.2f}|{100*r['sufficient_fpr']:.2f}|")
    lines += ['','## Interpretation limits','',
        '- Unknown offset preserves entity odds by construction; this is a diagnostic control, not evidence of causal equivalence.',
        '- New confirmation covers 14 reserved occupation pairs at most and is small; no broad model generality claim.',
        '- Conditional candidate entropy does not identify a unique internal uncertainty mechanism.',
        '- BBQ pair_* fields group scene families, not verified identity-swap pairs; they are not interpreted as identity invariance.',
        '- No new head/circuit localization or unrestricted generation experiment was run in this phase.',
        '- Calibration methods can raise recall and simultaneously increase harmful refusal; class-prevalence stress results are reported without selecting a target winner.',
        '- All fixed-beta diagnostics are exploratory, even when better than the frozen source-selected policy.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    shown=['baseline','fairsteer','evidence_steer','same_gate_direct','gated_cad'];labels=['Base','FairSteer','Evidence steer','Direct same gate','Gated CAD']
    for ax,dataset in zip(axes,('bbq_test','synth_test','synth_confirmation')):
        ms=[saved['metrics'][dataset][n]['0'] for n in shown];xx=np.arange(len(shown))
        ax.bar(xx-.18,[100*m['accuracy'] for m in ms],.36,label='Overall')
        ax.bar(xx+.18,[100*m['clear_accuracy'] for m in ms],.36,label='Sufficient')
        ax.set_xticks(xx,labels,rotation=30,ha='right');ax.set_ylim(0,100);ax.set_title(dataset);ax.set_ylabel('Accuracy (%)')
    axes[0].legend();fig.savefig(OUT/'results.png',dpi=180);fig.savefig(OUT/'results.pdf');plt.close(fig)
    audit={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'recomputed_decisions':audit_count,'frozen_hashes_verified':True,
        'all_predictions_and_accuracies_exact':True,'primary_holm':corrected,
        'confirmation_questions':320,'confirmation_primary_clusters':confirms[primary[0]]['clusters']}
    write(OUT/'independent_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
