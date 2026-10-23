"""Descriptive post-run diagnostics; no refitting or test policy selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mechanism.closed_transfer import Experiment,probe_values,recognition,bootstrap_difference
from mechanism.fairsteer_author import read,rows,write


def main():
    e=Experiment();e.prepare();saved=rows(e.out/'frozen_predictions.jsonl');pred={(x['dataset'],x['method'],x['uid']):x['prediction'] for x in saved}
    result={'scope':'Exploratory descriptive diagnostics on frozen predictions; no policy selection.'}
    values=e.datasets['synth_test'];cache=e.cache('synth_test');sig=probe_values(cache['features'],e.probes)
    result['synth_recognition_by_permutation']={}
    for pos in range(3):
        mask=np.array([v['permutation']==pos for v in values]);vs=[v for v,m in zip(values,mask) if m]
        result['synth_recognition_by_permutation'][pos]=recognition(vs,{k:v[mask] for k,v in sig.items()})
    primary=[v for v in values if v['permutation']==0]
    clear=[v for v in primary if v['condition']=='clear'];comparisons={}
    for a,b in [('evidence_steer','baseline'),('evidence_direct','baseline'),('evidence_direct','evidence_steer')]:
        vs=clear if b=='baseline' else primary
        pa=np.array([pred[('synth_test',a,v['uid'])] for v in vs]);pb=np.array([pred[('synth_test',b,v['uid'])] for v in vs])
        comparisons[f'{a}_vs_{b}'+('_clear' if b=='baseline' else '_overall')]=bootstrap_difference(vs,pa,pb,2000,e.c['seed'])
    result['comparisons']=comparisons
    write(e.out/'descriptive_diagnostics.json',result)
    print(result)


if __name__=='__main__':main()
