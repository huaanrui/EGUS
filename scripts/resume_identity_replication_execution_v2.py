"""Versioned diagnostic list repair; reuse unchanged frozen behavioral artifacts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,subprocess
import numpy as np
from mechanism.identity_replication import Experiment,site_id
from mechanism.identity_path import transmission,bootstrap_continuous
from mechanism.fairsteer_author import ROOT,read,write,sha

FOLDER=ROOT/'results/identity_replication_20261015_v1'
PROTOCOL=ROOT/'research/2026-10-15_identity_replication_execution_v2.md'


def trained_sites(e):
    selected=[]
    for key in e.meta['groups']:
        assert key+'_transfer_pos' in e.params and key+'_transfer_neg' in e.params
        selected.append(e.sites[key])
    return selected


def diagnostics_v2(e):
    vs=e.task_values('controls','coref');sites=trained_sites(e);anchor=e.meta['anchor']
    a=e.infer('diag_v2_A',vs,capture=sites);b=e.infer('diag_v2_B',vs,source=anchor,capture=sites)
    result={};scores={'baseline':a['scores'],'source_half':b['scores']}
    for site in sites:
        key=site_id(site);result[key]={}
        for group in ('transfer_pos','transfer_neg'):
            ii=e.params[key+'_'+group]
            r=e.infer('diag_v2_'+key+group+'_restore',vs,source=anchor,patch={'site':key,'indices':ii,'donor':a[key]})['scores']
            p=e.infer('diag_v2_'+key+group+'_transplant',vs,patch={'site':key,'indices':ii,'donor':b[key]})['scores']
            result[key][group]={'n':len(ii),'transmission':transmission(vs,a['scores'],b['scores'],r,p),
                'uncertainty':bootstrap_continuous(vs,a['scores'],b['scores'],r,p,e.c['bootstrap_replicates'],e.c['seed'])}
            scores[key+group+'_restore']=r;scores[key+group+'_transplant']=p
    e.save_scores('diagnostics_v2',vs,scores)
    write(e.out/'diagnostics_v2.json',{'version':'execution_v2','script_sha256':sha(Path(__file__)),'results':result})
    write(e.out/'diagnostics.json',result);e.log('diagnostics_v2_complete',sites=[site_id(s) for s in sites])


def run_logged(name,command):
    print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':name,'state':'start'}),flush=True)
    with (ROOT/f'results/identity_replication_20261015_v2_{name}.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':name,'state':'complete'}),flush=True)


def main():
    declaration=FOLDER/'execution_v2_declaration.json'
    identity={'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(PROTOCOL)}
    if declaration.exists():assert read(declaration)['identity']==identity
    else:
        write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,
            'repair':'only fitted groups are diagnosed; v1 Qwen3 behavioral eval/utilities already complete',
            'parent_selection_sha256':read(FOLDER/'all_models_frozen.json')['selection_sha256'],
            'norm_controls_sha256':read(FOLDER/'all_norm_controls_frozen.json')['sha256']})
    print(json.dumps({'stage':'evaluate_resume_v2','state':'start'}),flush=True)
    for name in ('qwen3','minicpm5','qwen25'):
        e=Experiment(name);e.prepare();s=read(e.out/'selection_frozen.json');e.validate_parameters(s)
        if name=='qwen3':
            assert set(read(e.out/'evaluation.json'))=={'historical','new_type1','winogender','confirmation'}
            assert (e.out/'utilities.json').exists();diagnostics_v2(e);e.generate(s['methods']);e.audit()
        else:e.evaluate();e.audit()
        e.close()
    print(json.dumps({'stage':'evaluate_resume_v2','state':'complete'}),flush=True)
    stages=[
        ('norm_evaluate',['scripts/identity_replication_norm_controls.py','--stage','evaluate']),
        ('random_utilities',['scripts/identity_replication_random_utilities.py']),
        ('semantic_v2',['scripts/identity_replication_semantic_controls_v2.py','--stage','evaluate']),
        ('summary',['scripts/summarize_identity_replication.py']),
        ('norm_summary',['scripts/summarize_identity_replication_norm.py']),
        ('plot',['scripts/plot_identity_replication.py']),
        ('anchor_compare',['scripts/compare_identity_replication_anchor.py']),
    ]
    for name,cmd in stages:run_logged(name,cmd)
    write(FOLDER/'complete.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'execution_version':2,
        'execution_declaration_sha256':sha(declaration),'stages':['evaluate_resume_v2']+[x[0] for x in stages]})


if __name__=='__main__':main()
