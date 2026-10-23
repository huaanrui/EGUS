"""Post-hoc identity-pair audit of existing predictions; no model or parameter fitting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict,Counter
import json
import numpy as np
from scipy.stats import binomtest
from mechanism.fairsteer_author import ROOT,rows,write,sha
from mechanism.closed_transfer import canonical

OUT=ROOT/'results/identity_pair_diagnostics_20261014'
DATA=ROOT/'data/prepared/evidence_mechanism_20261014_v1'
PRED=ROOT/'results/evidence_mechanism_20261014_v1/predictions.jsonl'
METHODS=['baseline','fairsteer','evidence_steer','same_gate_direct','evidence_direct','gated_cad']


def main():
    OUT.mkdir(exist_ok=True)
    saved=rows(PRED);lookup={(r['dataset'],r['method'],r['uid']):r['prediction'] for r in saved}
    reports={};details=[];examples=[]
    jobs=[('synth_test','clear'),('synth_confirmation','clear'),('winobias_test','type2'),('winobias_test','type1'),('winogender_test','native_gold')]
    for key,condition in jobs:
        values=rows(DATA/f'{key}.jsonl');values=[r for r in values if r['gender'] in ('male','female') and
            (r['task_type']==condition if key=='winobias_test' else r['condition']==condition)]
        groups=defaultdict(list)
        for r in values:groups[(r['pair_id'],r['permutation'])].append(r)
        valid=[];excluded=Counter()
        for pair,vs in sorted(groups.items()):
            if len(vs)!=2 or {r['gender'] for r in vs}!={'male','female'}:excluded['not_one_male_one_female']+=1;continue
            if len({canonical(r['context']) for r in vs})!=1 or len({canonical(r['question']) for r in vs})!=1:
                excluded['other_text_difference']+=1;continue
            if len({tuple(r['choices']) for r in vs})!=1:excluded['option_order_difference']+=1;continue
            if len({r['choices'][r['label']] for r in vs})!=1:excluded['gold_entity_changed']+=1;continue
            valid.append(vs)
        report={'pairs_by_permutation':dict(Counter(str(vs[0]['permutation']) for vs in valid)),
                'excluded_groups':dict(excluded),'methods':{}}
        for method in METHODS:
            if (key,method,values[0]['uid']) not in lookup:continue
            out={}
            for pos in sorted({r['permutation'] for r in values}):
                records=[]
                for vs in valid:
                    if vs[0]['permutation']!=pos:continue
                    pred=[lookup[(key,method,r['uid'])] for r in vs];correct=[p==r['label'] for r,p in zip(vs,pred)]
                    answers=[r['choices'][p] for r,p in zip(vs,pred)];unknown=[p==r['unknown_index'] and p>=0 for r,p in zip(vs,pred)]
                    rec={'dataset':key,'condition':condition,'method':method,'pair_id':vs[0]['pair_id'],
                         'family_id':vs[0]['family_id'],'permutation':pos,'flip':answers[0]!=answers[1],
                         'entity_flip':answers[0]!=answers[1] and not any(unknown),'both_correct':all(correct),
                         'one_correct':sum(correct)==1,'both_unknown':all(unknown),'correct_count':sum(correct),
                         'unknown_count':sum(unknown),'uids':[r['uid'] for r in vs],'predictions':pred}
                    if {r['stereotype'] for r in vs}=={'true','false'}:
                        pro=next(i for i,r in enumerate(vs) if r['stereotype']=='true');anti=1-pro
                        rec['pro_correct']=correct[pro];rec['anti_correct']=correct[anti]
                    records.append(rec);details.append(rec)
                    if method=='baseline' and pos==0 and rec['flip'] and len([e for e in examples if e['dataset']==key and e['condition']==condition])<3:
                        examples.append({'dataset':key,'condition':condition,'rows':vs,'answers':answers})
                m={'pairs':len(records)}
                for field in ('flip','entity_flip','both_correct','one_correct','both_unknown'):
                    m[field+'_rate']=float(np.mean([r[field] for r in records]))
                m['accuracy']=sum(r['correct_count'] for r in records)/(2*len(records))
                m['unknown_rate']=sum(r['unknown_count'] for r in records)/(2*len(records))
                labelled=[r for r in records if 'pro_correct' in r]
                if labelled:
                    m['pro_accuracy']=float(np.mean([r['pro_correct'] for r in labelled]));m['anti_accuracy']=float(np.mean([r['anti_correct'] for r in labelled]))
                    m['pro_anti_gap']=m['pro_accuracy']-m['anti_accuracy']
                    n10=sum(r['pro_correct'] and not r['anti_correct'] for r in labelled);n01=sum(not r['pro_correct'] and r['anti_correct'] for r in labelled)
                    m['pro_only_correct']=n10;m['anti_only_correct']=n01
                    m['paired_binomial_p_descriptive']=float(binomtest(n10,n10+n01,.5).pvalue) if n10+n01 else 1.
                    # Scene-family bootstrap keeps pairs sharing a template bound.
                    fam=defaultdict(list)
                    for r in labelled:fam[r['family_id']].append(int(r['pro_correct'])-int(r['anti_correct']))
                    group=list(fam.values());sums=np.array([sum(x) for x in group]);ns=np.array([len(x) for x in group]);rng=np.random.default_rng(20261014)
                    bs=[]
                    for _ in range(2000):
                        ix=rng.integers(0,len(group),len(group));bs.append(float(sums[ix].sum()/ns[ix].sum()))
                    m['gap_family_ci95']=list(map(float,np.quantile(bs,[.025,.975])));m['families']=len(group)
                out[str(pos)]=m
            report['methods'][method]=out
        reports[key+'_'+condition]=report
    write(OUT/'results.json',{'scope':'Post-hoc cached identity-pronoun comparisons. Gold entity, option order and non-pronoun text verified equal. Dataset sufficient/native labels are not independently human-revalidated. No identity-fact preservation or causal component claim.',
        'input_sha256':{'predictions':sha(PRED),**{p.name:sha(p) for p in DATA.glob('*.jsonl')}},'reports':reports})
    (OUT/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details),encoding='utf-8')
    write(OUT/'illustrative_examples.json',examples)
    for key,r in reports.items():
        print(key,r['pairs_by_permutation'],'excluded',r['excluded_groups'])
        for name,ps in r['methods'].items():
            m=ps['0'];print(name,{k:round(100*m[k],2) for k in ('accuracy','flip_rate','entity_flip_rate','both_correct_rate','unknown_rate','pro_anti_gap') if k in m})
        print('base CI',r['methods']['baseline']['0'].get('gap_family_ci95'))


if __name__=='__main__':main()
