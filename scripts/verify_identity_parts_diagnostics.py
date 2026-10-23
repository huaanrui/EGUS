"""Complete independent checks of stored diagnostic metrics and frozen metadata."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import json
import numpy as np
from mechanism.identity_parts import Experiment,FIELDS,vector_transmission
from mechanism.identity_path import transmission
from mechanism.identity_causal import variants,metrics
from mechanism.fairsteer_author import ROOT,read,rows,write,sha
from scripts.summarize_identity_parts import cached,check_metrics


def main():
    e=Experiment();e.prepare();sg=read(e.out/'signed_diagnostics.json');checked=0
    for split in ('screen','controls'):
        vs=e.task_values(split,'coref');by=defaultdict(dict)
        for r in rows(e.out/'predictions'/('signed_'+split+'.jsonl')):by[r['method']][r['uid']]=r
        ss={method:np.array([items[r['uid']]['scores'] for r in vs],dtype=np.float32) for method,items in by.items()}
        for site in e.meta['unit_sites']:
            key=site['id']
            for group in ('transfer_pos','transfer_neg'):
                r=ss[key+'_'+group+'_restore'];p=ss[key+'_'+group+'_transplant'];saved=sg['results'][split][key][group]
                assert transmission(vs,ss['baseline'],ss['source_half'],r,p)==saved['transmission']
                check_metrics(vs,r,saved['restore_metrics']);check_metrics(vs,p,saved['transplant_metrics']);checked+=2*len(vs)
    route=read(e.out/'routes.json');folder=e.out/'route_shards';vs=[variants(r) for r in e.values['rank']]
    a=cached(folder,'rank_A');b=cached(folder,'rank_B')
    for i,item in enumerate(route['option_sites']):
        r=cached(folder,f'option{i}_restore');p=cached(folder,f'option{i}_transplant')
        assert transmission(vs,a,b,r,p)==item['transmission'];checked+=2*len(vs)
        for field in FIELDS:
            key='field_'+field['id']
            z=vector_transmission(cached(folder,'rank_A',key),cached(folder,'rank_B',key),
                         cached(folder,f'option{i}_restore',key),cached(folder,f'option{i}_transplant',key))
            assert z==item['destinations'][field['id']]
    if route['attention_layer'] is not None:
        for item in route['head_scan']:
            h=item['head'];r=cached(folder,f'head{h}_restore');p=cached(folder,f'head{h}_transplant')
            assert transmission(vs,a,b,r,p)==item['transmission'];checked+=2*len(vs)
        vs=e.task_values('controls','coref');a=cached(folder,'controls_A');b=cached(folder,'controls_B')
        for name,item in route['validation'].items():
            r=cached(folder,'controls_'+name+'_restore');p=cached(folder,'controls_'+name+'_transplant')
            assert transmission(vs,a,b,r,p)==item['transmission']
            check_metrics(vs,r,item['restore_metrics']);check_metrics(vs,p,item['transplant_metrics']);checked+=2*len(vs)
    frozen=read(e.out/'parameters_frozen.json')
    assert sha(e.out/'parameters.npz')==frozen['parameters_sha256']
    assert sha(e.out/'parameter_meta.json')==frozen['metadata_sha256']
    groups=read(e.out/'routes_groups_frozen.json');ranked=groups['ranked']
    pos=[r['head'] for r in sorted(ranked,key=lambda r:-r['signed_alignment']) if r['signed_alignment']>1e-4][:4]
    neg=[r['head'] for r in sorted(ranked,key=lambda r:r['signed_alignment']) if r['signed_alignment']< -1e-4][:4]
    assert groups['groups']=={'positive':pos,'negative':neg}==route['groups']
    v=read(e.out/'verification.json');v.update({'diagnostic_records_recomputed':checked,
           'diagnostic_metrics_and_transmission_match':True,'parameter_metadata_unchanged':True,
           'supplementary_head_group_ranking_recomputed':True})
    write(e.out/'verification.json',v)
    report=ROOT/'research/2026-10-15_identity_parts_findings.md'
    with report.open('a',encoding='utf-8') as f:f.write('\n\nFinal diagnostic verification:\n\n'+json.dumps(v,indent=2)+'\n')
    complete=read(e.out/'completed_followup.json');complete.update({'report_sha256':sha(report),'verification_sha256':sha(e.out/'verification.json')})
    write(e.out/'completed_followup.json',complete);print(json.dumps(v,indent=2))


if __name__=='__main__':main()
