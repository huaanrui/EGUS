"""Exploratory text-control repair; no model/method selection after results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time,subprocess
from mechanism.identity_replication import Experiment,derived_rows,extended_metrics
from mechanism.fairsteer_author import ROOT,read,write,sha

FOLDER=ROOT/'results/identity_replication_20261015_v1'
DECL=FOLDER/'semantic_control_v2_declaration.json'
CASES=('gender_explicit_direct','neutral_explicit_direct','gender_required_direct','neutral_required_direct')


def make_row(r,case,p):
    required='required' in case;neutral=case.startswith('neutral')
    v=derived_rows(r,'identity_required' if required else 'explicit',p)
    if neutral:
        a,b=r['entities'];one,two=('17','29') if r['gender']=='male' else ('29','17')
        prefix=f'Personnel notes: The {a} has ID code {one}. The {b} has ID code {two}. '
        v['context']=prefix+v['context'][v['context'].index('Event record: '):]
    v['question']=('Who has ID code 29 in the personnel notes?' if neutral else 'Who is explicitly described as a woman in the personnel notes?') if required else 'Who received a blue card according to the event record?'
    v['uid']=r['source_id']+f':semantic_v2:{case}:p{p}';v['task']=case
    return v


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('declare','evaluate','wait_evaluate'),required=True);a=p.parse_args()
    if a.stage=='declare':
        assert not (FOLDER/'semantic_v2_complete.json').exists()
        write(DECL,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'script_sha256':sha(Path(__file__)),
            'status':'exploratory supplement after Qwen3 main semantic-control outputs, no new confirmation claim',
            'reason':'original neutral red/blue team descriptions overlap blue/green card facts; original question prefix also unnecessarily primes pronoun resolution',
            'repair':'neutral code17/29 swap parallel to man/woman swap; direct questions without pronoun prefix; same underlying facts and frozen interventions',
            'cases':CASES,'selection':'none; all three generators and all specified frozen methods reported'})
        return
    assert read(DECL)['script_sha256']==sha(Path(__file__))
    if a.stage=='wait_evaluate':
        while not (FOLDER/'complete.json').exists():
            finish=ROOT/'results/identity_replication_20261015_finish.log'
            if finish.exists() and 'Traceback (most recent call last)' in finish.read_text(encoding='utf-8',errors='replace')[-6000:]:raise RuntimeError('Original driver failed, repair before semantic supplement')
            time.sleep(5)
        while max(int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).split())>=100:time.sleep(5)
    for name in ('qwen3','minicpm5','qwen25'):
        e=Experiment(name);e.prepare();selection=read(e.out/'selection_frozen.json');e.validate_parameters(selection)
        methods={k:selection['methods'][k] for k in ('baseline','selected','fine_combo','whole_combo','source_half')}
        norm=read(e.out/'norm_controls_frozen.json')
        methods['norm_whole_combo']=norm['methods']['fine_norm_whole_combo']
        if name=='qwen3':
            methods.update({k:selection['methods'][k] for k in ('old_combo','old_l7')})
            methods['old_norm_whole_combo']=norm['methods']['old_norm_whole_combo']
        result={};pairs=e.values['controls']
        for case in CASES:
            vs=[make_row(r,case,p) for r in pairs for p in (0,1)]
            scores={k:e.infer('semantic_v2_'+case+'_'+k,vs,spec)['scores'] for k,spec in methods.items()}
            e.save_scores('semantic_v2_'+case,vs,scores);result[case]={k:extended_metrics(vs,z) for k,z in scores.items()}
            e.log('semantic_v2_case',case=case,metrics={k:{f:z[f] for f in ('accuracy','both_correct','flip')} for k,z in result[case].items()})
        write(e.out/'semantic_controls_v2.json',{'declaration_sha256':sha(DECL),'metrics':result,
            'original_neutral_confounded':True,'all_original_method_parameters_unchanged':True,
            'interpretation':'flip is undesirable only on explicit fixed-gold tasks; required-task gold changes across profiles'})
        e.audit();e.close()
    write(FOLDER/'semantic_v2_complete.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'script_sha256':sha(Path(__file__))})


if __name__=='__main__':main()
