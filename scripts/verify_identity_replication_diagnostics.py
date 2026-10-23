"""Recompute donor-patch transmission and family intervals from saved predictions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from mechanism.identity_path import transmission,bootstrap_continuous
from scripts.audit_identity_replication_records_v3 import restore_metadata


def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif a is None:assert b is None
    elif isinstance(a,(list,tuple)):assert np.allclose(a,b,rtol=0,atol=1e-12)
    elif isinstance(a,(int,float)):assert abs(a-b)<1e-12
    else:assert a==b


if __name__=='__main__':
    folder=ROOT/'results/identity_replication_20261015_v1';records=0;inputs={}
    for name in ('qwen3','minicpm5','qwen25'):
        out=folder/name;path=out/'predictions'/('diagnostics_v2.jsonl' if name=='qwen3' else 'diagnostics.jsonl')
        by=defaultdict(list)
        for r in rows(path):by[r['method']].append(restore_metadata(r));records+=1
        base=by['baseline'];uids=[r['uid'] for r in base]
        scores={}
        for k,rr in by.items():
            lookup={r['uid']:r for r in rr};assert set(lookup)==set(uids)
            scores[k]=np.array([lookup[uid]['scores'] for uid in uids])
        saved=read(out/'diagnostics.json')
        inputs[str(path)]=sha(path);inputs[str(out/'diagnostics.json')]=sha(out/'diagnostics.json')
        for site,groups in saved.items():
            for group,value in groups.items():
                rr=scores[site+group+'_restore'];pp=scores[site+group+'_transplant']
                equal(transmission(base,scores['baseline'],scores['source_half'],rr,pp),value['transmission'])
                equal(bootstrap_continuous(base,scores['baseline'],scores['source_half'],rr,pp,4000,20261015),value['uncertainty'])
        v=read(out/'verification.json');v['diagnostic_transmission_and_intervals_recomputed']=True;write(out/'verification.json',v)
    result={'diagnostic_records_recomputed':records,'diagnostic_transmission_and_intervals_recomputed':True,'inputs_sha256':inputs}
    write(folder/'diagnostics_verification.json',result)
    if (folder/'verification.json').exists():
        v=read(folder/'verification.json');v.update(result);write(folder/'verification.json',v)
    print('diagnostics independently verified',records)
