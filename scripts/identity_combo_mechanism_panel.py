"""Fixed descriptive mechanism contrasts; never used for selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import re
import time
import numpy as np
import torch
from mechanism.identity_combo import Experiment,make_candidates
from mechanism.identity_causal import metrics,comparison,variants
from mechanism.closed_transfer import digest
from mechanism.fairsteer_author import read,write,sha

NAMES=['subspace_rank1','complement_rank1','subspace_rank4','complement_rank4',
       'neurons_top128_0.5','neurons_matched128_0.5','l7_position_subject','l7_position_object','l7_position_question_identity']


def anonymize(r):
    names=['person X','person Y']
    if int(digest(r['pair_id'])[:8],16)%2:names.reverse()
    mapping=dict(zip(r['entities'],names));text=r['context']
    pattern=re.compile(r'\b('+ '|'.join(re.escape(x) for x in r['entities'])+r')\b',re.I)
    lookup={k.lower():v for k,v in mapping.items()}
    text=pattern.sub(lambda m:lookup[m.group().lower()],text)
    return {**r,'context':text,'entities':sorted(names),'gold':mapping[r['gold']],
            'source_id':r['source_id']+':anonymous','pair_id':r['pair_id']+':anonymous'}


def main():
    e=Experiment();e.prepare();assert (e.out/'complete.json').exists()
    declaration=e.out/'mechanism_panel_declaration.json'
    if declaration.exists():assert read(declaration)['script_sha256']==sha(Path(__file__))
    else:write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'script_sha256':sha(Path(__file__)),
        'status':'fixed historical exploratory panel, no reselection','names':NAMES})
    e.setup();e.parameters();c={r['name']:r for r in make_candidates()};result={};predictions=[]
    for task in ('coref','literal'):
        vs=e.task_values('historical',task);scores={}
        for name in ['baseline','l7_scale_0.5']+NAMES:
            cache=e.infer('panel_'+task+'_'+name,vs,c[name]);s=cache['scores'];scores[name]=s
            result.setdefault(task,{})[name]=metrics(vs,s)
            for r,q,p in zip(vs,s,s.argmax(1)):
                predictions.append({'uid':r['uid'],'task':task,'method':name,'prediction':int(p),'label':r['label'],'scores':q.tolist()})
            e.log('mechanism_panel',task=task,method=name,accuracy=result[task][name]['accuracy'],both_correct=result[task][name]['both_correct'])
    write(e.out/'mechanism_panel.json',{'historical_exploratory':True,'never_selects':True,'metrics':result})
    (e.out/'mechanism_panel_predictions.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in predictions),encoding='utf-8')
    # Additional counterfactual diagnostic, declared after historical native
    # behavior/neutral donors were inspected; no method selection uses it.
    original=[variants(r) for r in e.values['rank']]
    roles=[variants(r,'role_swap') for r in e.values['rank']]
    anonrank=[variants(anonymize(r)) for r in e.values['rank']]
    raw=dict(np.load(e.out/'rank_raw.npz'))
    rolecache=e.infer('panel_rank_role_capture',roles,capture=True)
    anoncache=e.infer('panel_rank_anonymous_capture',anonrank,capture=True)
    groups={}
    for i,r in enumerate(original):groups.setdefault(r['pair_id'],{})[r['gender']]=i
    male=np.array([g['male'] for g in groups.values()]);female=np.array([g['female'] for g in groups.values()]);interaction={}
    for mode,capture in [('role_exchange',rolecache),('anonymous_occupations',anoncache)]:
        items=[]
        for l in (0,5,6,7,8,9):
            a=raw['mlp'][male,l]-raw['mlp'][female,l];b=capture['mlp'][male,l]-capture['mlp'][female,l]
            cosine=np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-8)
            items.append({'layer':l,'mean_gender_delta_cosine':float(cosine.mean()),
                          'relative_gender_delta_change':float(np.mean(np.linalg.norm(a-b,axis=1)/np.maximum(np.linalg.norm(a,axis=1),1e-8)))})
        interaction[mode]=items
    anon=[variants(anonymize(r),permutation=p) for r in e.values['historical'] for p in (0,1)]
    methods=e.methods();anonymized={}
    for name in ('baseline','old_l7','best_combo','instruction'):
        if name not in methods:continue
        spec,instruction=methods[name];cache=e.infer('panel_anonymous_'+name,anon,spec,instruction=instruction)
        anonymized[name]=metrics(anon,cache['scores'])
    write(e.out/'occupation_interaction_diagnostics.json',{
        'status':'post-selection historical exploratory diagnostics, never reselects',
        'gender_delta_context_changes':interaction,'anonymized_occupation_behavior':anonymized,
        'limits':'Anonymization changes lexical form/task difficulty as well as occupation priors. Gender-difference norms are not bias-specific attribution.'})
    del e.model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
