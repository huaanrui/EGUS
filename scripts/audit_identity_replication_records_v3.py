"""Restore omitted bookkeeping metadata in memory, never edit predictions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import argparse,re,tomllib
import numpy as np
from mechanism.fairsteer_author import ROOT,read,rows,sha,write
from mechanism.identity_replication import CONFIG,Experiment,extended_metrics

_LOOKUP=None


def metadata_lookup():
    global _LOOKUP
    if _LOOKUP is None:
        c=tomllib.loads(CONFIG.read_text());source=ROOT/c['source_data'];paths=[source/f'{k}.jsonl' for k in ('rank','screen','dev','historical','new_type1','winogender')]
        paths.append(ROOT/c['data_dir']/'confirmation.jsonl');_LOOKUP={}
        for p in paths:
            for r in rows(p):
                pid=r['pair_id'];value=r['occupation_pair']
                if pid in _LOOKUP:assert _LOOKUP[pid]==value
                _LOOKUP[pid]=value
    return _LOOKUP


def restore_metadata(r,lookup=None):
    source=metadata_lookup() if lookup is None else lookup
    pid=re.sub(r':record[01]$','',r['pair_id']).removesuffix(':anonymous')
    assert pid in source,(r['pair_id'],pid)
    if 'occupation_pair' in r:assert r['occupation_pair']==source[pid]
    return {**r,'occupation_pair':source[pid]}


def audit(e):
    plan=read(e.out/'plan_frozen.json')
    for p,h in {**plan['implementation'],**plan['inputs']}.items():assert sha(p)==h,p
    e.validate_parameters(read(e.out/'selection_frozen.json'));count=0;aggregate={}
    for p in (e.out/'predictions').glob('*.jsonl'):
        by=defaultdict(list)
        for r in rows(p):
            assert r['prediction']==int(np.argmax(r['scores']));count+=1;by[r['method']].append(restore_metadata(r))
        aggregate[p.stem]={k:extended_metrics(z,np.array([r['scores'] for r in z])) for k,z in by.items()}
    evaluation=read(e.out/'evaluation.json')
    for split in evaluation:
        for method in evaluation[split]:
            for field in ('accuracy','both_correct','flip','both_wrong'):
                assert abs(evaluation[split][method][field]-aggregate[split][method][field])<1e-12,(split,method,field)
    write(e.out/'verification.json',{'audit_version':3,'auditor_sha256':sha(Path(__file__)),
        'records_recomputed':count,'frozen_inputs_unchanged':True,'parameters_unchanged':True,'evaluation_metrics_recomputed':True,
        'occupation_pair_restored_in_memory_from_frozen_sources':True,'prediction_files_not_modified_by_audit':True,
        'confirmation_status':'native Type1 unseen prepared families; all three model selections and norm controls froze before target inference'})
    e.log('audit_v3_complete',records=count)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=('qwen3','minicpm5','qwen25'));a=p.parse_args()
    e=Experiment(a.model);e.prepare();audit(e)
