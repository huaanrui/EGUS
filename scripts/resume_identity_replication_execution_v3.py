from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,subprocess
from mechanism.identity_replication import Experiment
from mechanism.fairsteer_author import ROOT,read,write,sha
from scripts.audit_identity_replication_records_v3 import audit
from scripts.resume_identity_replication_execution_v2 import diagnostics_v2

FOLDER=ROOT/'results/identity_replication_20261015_v1'


def main():
    files=[Path(__file__),ROOT/'scripts/audit_identity_replication_records_v3.py',
        ROOT/'scripts/run_identity_replication_reporting_v3.py',ROOT/'scripts/resume_identity_replication_execution_v2.py',
        ROOT/'research/2026-10-15_identity_replication_execution_v3.md']
    identity={str(p):sha(p) for p in files};decl=FOLDER/'execution_v3_declaration.json'
    if decl.exists():assert read(decl)['implementation']==identity
    else:write(decl,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'implementation':identity,
        'reason':'saved occupation_pair omitted, reconstructed only for independent audit; no numerical inference change',
        'parent_execution_v2_sha256':sha(FOLDER/'execution_v2_declaration.json')})
    Experiment.audit=audit
    print(json.dumps({'stage':'evaluate_resume_v3','state':'start'}),flush=True)
    for name in ('qwen3','minicpm5','qwen25'):
        e=Experiment(name);e.prepare();s=read(e.out/'selection_frozen.json');e.validate_parameters(s)
        if name=='qwen3':diagnostics_v2(e);e.generate(s['methods']);e.audit()
        else:e.evaluate();e.audit()
        e.close()
    print(json.dumps({'stage':'evaluate_resume_v3','state':'complete'}),flush=True)
    stages=[
        ('norm_evaluate',['scripts/identity_replication_norm_controls.py','--stage','evaluate']),
        ('random_utilities',['scripts/identity_replication_random_utilities.py']),
        ('semantic_v2',['scripts/identity_replication_semantic_controls_v2.py','--stage','evaluate']),
        ('summary',['scripts/summarize_identity_replication.py']),
        ('norm_summary',['scripts/summarize_identity_replication_norm.py']),
        ('plot',['scripts/plot_identity_replication.py']),
        ('anchor_compare',['scripts/compare_identity_replication_anchor.py'])]
    wrapper='scripts/run_identity_replication_reporting_v3.py'
    for stage,cmd in stages:
        print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,'state':'start'}),flush=True)
        with (ROOT/f'results/identity_replication_20261015_v3_{stage}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,wrapper,*cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,'state':'complete'}),flush=True)
    write(FOLDER/'complete.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'execution_version':3,
        'execution_declaration_sha256':sha(decl),'stages':['evaluate_resume_v3']+[s[0] for s in stages]})


if __name__=='__main__':main()
