"""Predeclared ability checks for every frozen random combination seed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time
from mechanism.identity_replication import Experiment,extended_metrics,derived_rows
from mechanism.fairsteer_author import ROOT,read,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declare',action='store_true');args=parser.parse_args()
    root=ROOT/'results/identity_replication_20261015_v1';declaration=root/'random_utility_declaration.json'
    tasks=('reading','binding','record','explicit','identity_required')
    if args.declare:
        assert not (root/'all_models_frozen.json').exists(),'Declare before target opening'
        write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'script_sha256':sha(Path(__file__)),
            'tasks':tasks,'methods':'all 5 matched random fine combinations, plus all 5 old Qwen3 combinations',
            'selection':'none; use frozen methods, indices and dev norm-matched strengths'})
        return
    assert read(declaration)['script_sha256']==sha(Path(__file__))
    for name in ('qwen3','minicpm5','qwen25'):
        e=Experiment(name);e.prepare();selection=read(e.out/'selection_frozen.json');e.validate_parameters(selection)
        methods={k:v for k,v in selection['methods'].items() if k.startswith(('random_combo_','old_random_combo_'))}
        result={}
        for task in tasks:
            vs=e.task_values('controls',task) if task in ('reading','binding','record') else [derived_rows(r,task,p) for r in e.values['controls'] for p in (0,1)]
            scores={k:e.infer('random_utility_'+task+'_'+k,vs,spec)['scores'] for k,spec in methods.items()}
            e.save_scores('random_utility_'+task,vs,scores);result[task]={k:extended_metrics(vs,z) for k,z in scores.items()}
            e.log('random_utility_complete',task=task)
        write(e.out/'random_utilities.json',result);e.audit();e.close()


if __name__=='__main__':main()
