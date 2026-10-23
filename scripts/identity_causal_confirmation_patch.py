"""Predeclared descriptive replication of fit-selected oracle sites on confirmation.

Never modifies selection or deployment. Includes random-layer and neutral donors.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import gc
import re
import time
import numpy as np
import torch
from mechanism.identity_causal import Experiment,variants,metrics,comparison,INSTRUCTION,render
from mechanism.fairsteer_author import read,write,sha


def main():
    e=Experiment();e.prepare()
    assert (e.out/'complete.json').exists(),'Complete frozen main experiment first'
    declaration=e.out/'confirmation_patch_declaration.json'
    if declaration.exists():assert read(declaration)['script_sha256']==sha(Path(__file__))
    else:write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'script_sha256':sha(Path(__file__)),
        'selection_sha256':sha(e.out/'selection_frozen.json'),'status':'descriptive oracle replication, no reselection',
        'contrasts':'all top3 sites vs matching random layers, counterpart vs neutral donor, primary order only'})
    e.setup();vs=[variants(r) for r in e.values['confirmation']]
    base,act,_=e.infer('confirmation_patch_capture',vs,capture=True)
    neutral=[{**r,'context':r['context']+' This happened yesterday.'} for r in vs]
    _,nact,_=e.infer('confirmation_patch_neutral_capture',neutral,capture=True)
    pairs=defaultdict(list)
    for i,r in enumerate(vs):pairs[r['pair_id']].append(i)
    donor=np.zeros(len(vs),int)
    for ids in pairs.values():assert len(ids)==2;donor[ids]=ids[::-1]
    gold=np.array([r['label'] for r in vs]);ok=base.argmax(1)==gold
    discord=np.array([ok[i]!=ok[donor[i]] for i in range(len(vs))]);bad=discord&~ok;good=discord&ok
    margin=lambda s:s[np.arange(len(vs)),gold]-s[np.arange(len(vs)),1-gold]
    loc=read(e.out/'localization.json');result=[]
    for i,site in enumerate(loc['top_sites']):
        random={**site,'layer':loc['random_sites'][i]['layer']}
        for control,ss in [('top',site),('random',random)]:
            mi=ss['layer']*2+(ss['module']=='mlp');pi=int(ss['position']=='decision')
            for donor_name,acts in [('counterpart',act[donor]),('neutral',nact)]:
                s,_,_=e.infer(f'confirmation_oracle_{i}_{control}_{donor_name}',vs,
                    {'operation':'patch','site':ss,'donor':acts[:,mi,pi]})
                dm=margin(s)-margin(base);pred=s.argmax(1)
                rec={'site':ss,'top_index':i,'control':control,'donor':donor_name,
                     'bad_rescue_margin':float(dm[bad].mean()) if bad.any() else None,
                     'good_reverse_margin':float(dm[good].mean()) if good.any() else None,
                     'bad_fixed':int((pred[bad]==gold[bad]).sum()),
                     'good_broken':int((pred[good]!=gold[good]).sum()),
                     'prediction_change':float((pred!=base.argmax(1)).mean()),'metrics':metrics(vs,s),
                     'both_correct_difference':comparison(vs,pred,base.argmax(1),'both_correct',4000,e.c['seed'])}
                result.append(rec);e.log('confirmation_oracle',**{k:rec[k] for k in ('site','control','donor','bad_fixed','good_broken')})
    # Literal reading is a secondary control; do not reselect using its outcomes.
    literal=[]
    for r in e.values['confirmation']:
        for pos in (0,1):
            v=variants(r,'reading',pos);choices=['him','her'] if pos==0 else ['her','him']
            literal.append({**v,'question':'Which of these words occurs as a pronoun in the sentence?',
                'choices':choices,'label':choices.index(r['pronoun']),'gold':r['pronoun'],
                'uid':r['source_id']+f':literal_reading:p{pos}'})
    selected=e.selected_spec();methods={'baseline':None,'selected':selected,'instruction':None,'fairsteer':{'operation':'fairsteer'},'evidence':{'operation':'evidence'}}
    if selected['operation']!='none':
        for i in range(3):methods[f'random_{i}']=e.matched_random_spec(i,selected)
    literal_result={};literal_predictions={}
    for name,spec in methods.items():
        scores,_,_=e.infer('confirmation_literal_'+name,literal,spec,instruction=name=='instruction');literal_result[name]=metrics(literal,scores)
        literal_predictions[name]=scores.argmax(1)
    literal_result['selected_difference']=comparison(literal,literal_predictions['selected'],literal_predictions['baseline'],'accuracy',4000,e.c['seed'])
    write(e.out/'literal_reading_results.json',{'status':'secondary control, added after selection-set metalinguistic baseline inspection; no reselection',
                                             'metrics':literal_result})
    # Fixed instruction transferred to no-option generation. The answer-format
    # sentence must request an occupation, since there are no answer letters.
    # Added after observing closed instruction results; explicitly descriptive.
    genfolder=e.out/'generation'/'instruction_secondary';genfolder.mkdir(parents=True,exist_ok=True)
    genrows=[];instruction_predictions=[]
    prefix=INSTRUCTION.replace('Output only the answer letter.','Output only the occupation name.')
    for start in range(0,len(vs),e.c['batch_size']):
        batch=vs[start:start+e.c['batch_size']];path=genfolder/f'{start:06d}.json'
        if path.exists():texts=read(path)
        else:
            inputs=e.tok([prefix+render(r,openqa=True) for r in batch],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                output=e.model.generate(**inputs,max_new_tokens=12,do_sample=False,use_cache=True,
                    pad_token_id=e.tok.pad_token_id,eos_token_id=e.tok.eos_token_id)
            texts=e.tok.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True);write(path,texts)
        for r,t in zip(batch,texts):
            clean=re.sub(r'^the\s+','',t.strip().strip(' \t\r\n.,:;!?\"\'').lower())
            p=next((i for i,x in enumerate(r['choices']) if clean==x.lower()),-1);instruction_predictions.append(p)
            genrows.append({'uid':r['uid'],'raw':t,'prediction':p,'label':r['label']})
    from mechanism.fairsteer_author import rows
    old=rows(e.out/'generation_predictions.jsonl');baseline={x['uid']:x['strict_prediction'] for x in old if x['method']=='baseline'}
    write(e.out/'instruction_generation_secondary.json',{'status':'descriptive; added after closed-choice baseline inspection',
        'prompt_prefix':prefix,'metrics':metrics(vs,predictions=instruction_predictions),
        'unparsed':float(np.mean(np.asarray(instruction_predictions)<0)),
        'both_correct_vs_baseline':comparison(vs,instruction_predictions,[baseline[r['uid']] for r in vs],'both_correct',4000,e.c['seed']),
        'raw_predictions':genrows})
    write(e.out/'confirmation_patch_results.json',{'baseline':metrics(vs,base),
        'discordant_pairs':int(bad.sum()),'results':result,
        'interpretation':'Counterpart donor is oracle and cannot be used as a deployable method; bidirectional rescue/damage may reflect ordinary answer or identity representations.'})
    del e.model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
