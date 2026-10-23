from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mechanism.identity_replication import Experiment,extended_metrics
from mechanism.identity_causal import variants,comparison
from mechanism.fairsteer_author import ROOT,read,rows,write

folder=ROOT/'results/identity_replication_20261015_v1';result={}
for name in ('qwen3','minicpm5','qwen25'):
    e=Experiment(name);e.prepare();vs=[variants(r) for r in e.values['generation']]
    data=rows(e.out/'chat_predictions.jsonl');lookup={(r['method'],r['uid']):r for r in data}
    saved=read(e.out/'chat_results.json');pred={};model={}
    for method in saved:
        p=np.array([lookup[(method,r['uid'])]['prediction'] for r in vs]);pred[method]=p
        metric=extended_metrics(vs,predictions=p)
        for f in ('accuracy','both_correct','flip'):assert metric[f]==saved[method]['strict'][f]
        model[method]={'metrics':saved[method]}
    for method in saved:
        if method=='baseline':continue
        model[method]['vs_baseline']={f:comparison(vs,pred[method],pred['baseline'],f,4000,20261015) for f in ('accuracy','both_correct','flip')}
    result[name]=model
write(folder/'generation_comparisons.json',{'models':result,'status':'historical32pairs, strict whole-occupation primary; exploratory CIs unadjusted, not CEB free continuation'})
print(__import__('json').dumps({n:{k:v.get('vs_baseline',{}).get('both_correct') for k,v in z.items() if k!='baseline'} for n,z in result.items()},indent=2))
