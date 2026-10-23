"""Quantify FP16/batch differences for unchanged old Qwen3 interventions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mechanism.fairsteer_author import ROOT,rows,write

folder=ROOT/'results/identity_replication_20261015_v1/qwen3';result={}
for split in ('historical','new_type1','winogender'):
    path=folder/'predictions'/(split+'.jsonl')
    if not path.exists():continue
    old={(r['method'],r['uid']):r for r in rows(ROOT/'results/identity_parts_20261015_v1/predictions'/(split+'.jsonl'))}
    new={(r['method'],r['uid']):r for r in rows(path)};result[split]={}
    for before,after in [('baseline','baseline'),('source_half','old_l7'),('overall','old_combo')]:
        keys=[uid for method,uid in new if method==after]
        aa=np.array([old[(before,k)]['scores'] for k in keys]);bb=np.array([new[(after,k)]['scores'] for k in keys])
        assert all(old[(before,k)]['label']==new[(after,k)]['label'] for k in keys)
        ma=aa[:,0]-aa[:,1];mb=bb[:,0]-bb[:,1];changed=aa.argmax(1)!=bb.argmax(1)
        result[split][after]={'rows':len(keys),'max_abs_logit_difference':float(abs(aa-bb).max()),
            'mean_abs_logit_difference':float(abs(aa-bb).mean()),'max_abs_margin_difference':float(abs(ma-mb).max()),
            'changed_predictions':int(changed.sum()),'old_abs_margins_at_changes':abs(ma[changed]).tolist(),
            'new_abs_margins_at_changes':abs(mb[changed]).tolist()}
write(folder/'old_anchor_numerical_comparison.json',{'old_batch_size':8,'new_batch_size':4,'dtype':'float16',
    'same_frozen_old_parameters':True,'results':result})
print(__import__('json').dumps(result,indent=2))
