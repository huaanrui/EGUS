from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import json
import time
import numpy as np
import torch
from mechanism.identity_path import Experiment,record_rows,reverse_indices,bootstrap_continuous
from mechanism.identity_combo import task_row
from mechanism.identity_causal import metrics,comparison
from mechanism.fairsteer_author import ROOT,read,write,sha


def active(r):
    return {**r,'uid':r['uid']+':active',
            'question':f"Considering the pronoun '{r['pronoun']}' in the sentence, "+r['question'][0].lower()+r['question'][1:]}


def main():
    e=Experiment();e.prepare();assert (e.out/'selection_frozen.json').exists()
    protocol=ROOT/'research/2026-10-15_identity_path_active_controls.md'
    identity={'script':sha(Path(__file__)),'protocol':sha(protocol),'selection':sha(e.out/'selection_frozen.json')}
    declaration=e.out/'active_controls_declaration.json'
    if declaration.exists():assert read(declaration)['identity']==identity
    else:write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,'status':'supplementary historical; never reselects'})
    ss=read(e.out/'selection_frozen.json')['selected']['sites'];result={}
    for task in ('literal','reading','binding','record'):
        vs=e.task_values('controls','coref') if task=='record' else [task_row(r,task,p) for r in e.values['controls'] for p in (0,1)]
        if task=='record':vs=record_rows(vs)
        vs=[active(r) for r in vs]
        e.setup();_,pos=e.encode(vs);assert (pos[:,4]>=0).all()
        a,b=e.ab('active_'+task,vs)
        trial=e.trial('active_'+task,vs,a,b,ss)
        h=e.infer('active_'+task+'_heads_half',vs,sites=ss,head_factor=.5)['scores']
        scores={'baseline':a['scores'],'source_half':b['scores'],'selected_half':h,
                'source_half_restore':trial['restore'],'selected_transplant':trial['transplant']}
        r={'metrics':{n:metrics(vs,z) for n,z in scores.items()},'transmission':trial['transmission'],
           'selected_half_vs_baseline':{f:comparison(vs,h.argmax(1),a['scores'].argmax(1),f,e.c['bootstrap_replicates'],e.c['seed']) for f in ('accuracy','both_correct')},
           'question_receiver_present':len(vs)}
        if task=='record':
            ra=e.infer('active_record_source_A',vs,capture=True,source_position=5)
            other=reverse_indices(vs,'record');donor=ra['mlp'][other]
            rb=e.infer('active_record_source_B',vs,donor,capture=True,source_position=5)
            t=e.trial('active_record_source',vs,ra,rb,ss,donor,source_position=5)
            r['record_evidence_transmission']={'estimate':t['transmission'],'bootstrap':bootstrap_continuous(vs,ra['scores'],rb['scores'],t['restore'],t['transplant'],e.c['bootstrap_replicates'],e.c['seed'])}
            scores.update({'record_source_swap':rb['scores'],'record_restore':t['restore'],'record_transplant':t['transplant']})
        result[task]=r;e.save_predictions('active_'+task,vs,scores);e.log('active_controls',task=task,result=r)
    write(e.out/'active_controls.json',{'status':'supplementary historical; receiver present in every item; no reselection','results':result})
    del e.model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
