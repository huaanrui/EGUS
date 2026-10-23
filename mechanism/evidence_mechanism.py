"""Exploratory decomposition and source-selected context-aware decoding.

Frozen historic assets are read, never modified. Confirmation excludes all
occupation pairs used in the previous SynthBias smoke test and target calibration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import shutil
import time
import tomllib

import numpy as np
from scipy.special import expit, logsumexp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fairsteer_author import ROOT, read, rows, sha, write
from .closed_transfer import digest, rank, probe_values, synth_data, metric, bootstrap_difference

CONFIG = ROOT/'configs/experiment_evidence_mechanism_v1.toml'
INSTRUCTION = ('Answer using only the context. Follow explicit contextual evidence even if it '
               'conflicts with common assumptions about people or occupations. If the context '
               'is insufficient, choose the option that says the answer cannot be determined. '
               'Output only the answer letter.')


def render(row, mode='full'):
    choices=''.join(f'\n{a}. {v}' for a,v in zip('ABC',row['choices']))
    body=(row['context']+' ' if mode!='no_context' else '')+row['question']+choices+'\nAnswer:'
    if mode=='instruction':return INSTRUCTION+'\n\n'+body
    if mode not in ('full','no_context'):raise ValueError(mode)
    return body


def cad_scores(full, no_context, beta, mask=None):
    full=np.asarray(full,np.float64);no_context=np.asarray(no_context,np.float64)
    delta=float(beta)*(full-no_context)
    if mask is not None:delta=delta*np.asarray(mask,bool)[:,None]
    return full+delta


def offset_scores(base, values, amount, trigger):
    result=np.asarray(base,np.float64).copy()
    for i,row in enumerate(values):
        if trigger[i] and row['unknown_index']>=0:result[i,row['unknown_index']]+=amount
    return result


def scalar_align(source, target_adapt, target_test):
    s=np.asarray(source,np.float64);a=np.asarray(target_adapt,np.float64)
    return (np.asarray(target_test,np.float64)-a.mean())/max(a.std(),1e-8)*max(s.std(),1e-8)+s.mean()


def feature_align_margin(source_mean,source_std,adapt,features,coef,intercept):
    # Diagonal moment alignment is not full covariance whitening.
    scale=np.maximum(source_std,1e-6)/np.maximum(np.std(adapt,axis=0),1e-6)
    w=coef*scale
    return features@w+float(intercept)+float((source_mean-np.mean(adapt,axis=0)*scale)@coef)


def decomposition(values,base,edited):
    base=np.asarray(base,np.float64);edited=np.asarray(edited,np.float64)
    result=[]
    for i,row in enumerate(values):
        u=row['unknown_index'];entities=[j for j in range(len(row['choices'])) if j!=u]
        assert len(entities)==2
        # SynthBias ambiguous rows do not provide a verified stereotype target.
        # Keep alphabetical entity contrast separate from stereotype contrast.
        entities=sorted(entities,key=lambda j:row['choices'][j].casefold())
        a,b=entities
        rec={'uid':row['uid'],'family_id':row['family_id'],'condition':row['condition'],
             'base_entity_logodds':float(base[i,a]-base[i,b]),
             'edited_entity_logodds':float(edited[i,a]-edited[i,b]),
             'entity_delta':float((edited[i,a]-edited[i,b])-(base[i,a]-base[i,b])),
             'entity_choice_changed':bool((base[i,a]>base[i,b])!=(edited[i,a]>edited[i,b]))}
        if u>=0:
            r0=base[i,u]-logsumexp(base[i,entities]);r1=edited[i,u]-logsumexp(edited[i,entities])
            rec.update(base_abstention_logodds=float(r0),edited_abstention_logodds=float(r1),abstention_delta=float(r1-r0))
        if row['condition']=='clear':
            gold=row['label'];other=next(j for j in entities if j!=gold)
            rec['gold_entity_margin_delta']=float((edited[i,gold]-edited[i,other])-(base[i,gold]-base[i,other]))
        target=row.get('target_loc')
        if target is not None and target in entities:
            other=next(j for j in entities if j!=target)
            rec['stereotype_margin_delta']=float((edited[i,target]-edited[i,other])-(base[i,target]-base[i,other]))
        result.append(rec)
    return result


def summarize_decomposition(records):
    output={}
    for condition in ('all','ambig','clear','native_gold'):
        vs=records if condition=='all' else [r for r in records if r['condition']==condition]
        if not vs:continue
        m={'rows':len(vs),'entity_flip_rate':float(np.mean([r['entity_choice_changed'] for r in vs]))}
        for key in ('abstention_delta','entity_delta','gold_entity_margin_delta','stereotype_margin_delta'):
            x=np.array([r[key] for r in vs if key in r])
            if len(x):m[key]={'rows':len(x),'mean':float(x.mean()),'mean_absolute':float(abs(x).mean()),'median':float(np.median(x))}
        output[condition]=m
    return output


def transition(values,base_pred,pred):
    labels=np.array([r['label'] for r in values]);a=base_pred==labels;b=pred==labels
    result={}
    for name in ('ambig','clear','native_gold'):
        use=np.array([r['condition']==name for r in values])
        if use.any():result[name]={'rows':int(use.sum()),'fixed':int((use&~a&b).sum()),'broken':int((use&a&~b).sum())}
    return result


def choose_cad(values,base,no_context,evidence_probability,betas,gate,clear_tolerance,ambig_tolerance):
    reference=metric(values,base.argmax(1));candidates=[]
    for beta in betas:
        score=cad_scores(base,no_context,beta,evidence_probability<=.25 if gate else None)
        m=metric(values,score.argmax(1),scores=score)
        feasible=(m['accuracy']>=reference['accuracy']-1e-12 and
                  m['clear_accuracy']>=reference['clear_accuracy']-clear_tolerance-1e-12 and
                  m['ambig_accuracy']>=reference['ambig_accuracy']-ambig_tolerance-1e-12)
        candidates.append({'beta':beta,'metrics':m,'feasible':feasible})
    feasible=[r for r in candidates if r['feasible']];assert feasible
    selected=max(feasible,key=lambda r:(r['metrics']['accuracy'],r['metrics']['clear_accuracy'],-r['beta']))
    return selected,candidates


class Experiment:
    def __init__(self,config=CONFIG):
        self.config=Path(config);self.c=tomllib.loads(self.config.read_text(encoding='utf-8'))
        self.out=ROOT/self.c['output_dir'];self.data=ROOT/self.c['data_dir']
        self.out.mkdir(parents=True,exist_ok=True);self.data.mkdir(parents=True,exist_ok=True)
        self.source=ROOT/self.c['source_dir'];self.transfer=ROOT/self.c['transfer_dir'];self.transfer_data=ROOT/self.c['transfer_data']
        self.probes=dict(np.load(self.source/'probes.npz'));self.vector=np.load(self.source/'direction_dim.npy')
        self.tokenizer=AutoTokenizer.from_pretrained(self.c['model_path'],local_files_only=True,padding_side='left')
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.letter_ids=[self.tokenizer('Answer: '+a).input_ids[-1] for a in 'ABC']
        self.datasets={};self.model=None

    def log(self,event,**fields):
        record={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'event':event,**fields}
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

    def prepare(self):
        source_rows=rows(ROOT/'data/prepared/fairsteer_author_qwen3_v1/bbq.jsonl')
        split=read(ROOT/'data/prepared/fairsteer_improve_20261013_v2/splits.json')
        assignments=split['family_assignment']
        mapping=read(ROOT/'configs/bbq_template_families.json')['mapping']
        from .fairsteer_improve import family
        for name in ('dev','test'):
            self.datasets['bbq_'+name]=[{**r,'choices':[r[f'ans{i}'] for i in range(3)],
                'source_id':r['uid'],'family_id':family(r,mapping),'pair_id':family(r,mapping),
                'condition':'ambig' if r['context_condition']=='ambig' else 'clear',
                'stereotype':'unspecified','dataset':'bbq','permutation':0}
                for i in split['bbq_indices'][name] for r in [source_rows[i]]]
        for name in ('synth_test','winobias_test','winogender_test'):
            self.datasets[name]=rows(self.transfer_data/f'{name}.jsonl')
        # Rebuild the pre-existing reserve pool, excluding whole occupation pairs
        # previously selected for smoke/audit, before obtaining any new outputs.
        c=tomllib.loads((ROOT/'configs/experiment_closed_transfer_v2.toml').read_text())
        c['synth_audit_rows']=1024
        rebuilt,_=synth_data(ROOT/'data/raw/closed_transfer/synthbias/data/synthbias_data.csv',c)
        previous=rows(self.transfer_data/'synth_audit.jsonl')
        used={r['occupation_pair'] for r in previous}
        candidates=[r for r in rebuilt['synth_audit'] if r['occupation_pair'] not in used]
        chosen=[];wanted=self.c['confirmation_rows']//4
        for condition in ('ambig','clear'):
            grouped=defaultdict(list)
            for r in candidates:
                if r['condition']==condition:grouped[r['pair_id']].append(r)
            keys=sorted(grouped,key=lambda p:rank(self.c['seed'],p));assert len(keys)>=wanted
            for p in keys[:wanted]:
                group=grouped[p];assert len(group)==2
                for r in group:
                    for pos in range(3):
                        choices=r['choices'][pos:]+r['choices'][:pos];gold=r['choices'][r['label']]
                        chosen.append({**r,'split':'confirmation','choices':choices,'label':choices.index(gold),
                            'unknown_index':choices.index(r['choices'][r['unknown_index']]),'permutation':pos,
                            'uid':r['source_id']+f':p{pos}'})
        self.datasets['synth_confirmation']=sorted(chosen,key=lambda r:r['uid'])
        for name in ('synth_test','winobias_test','winogender_test','bbq_test'):
            assert not {r['source_id'] for r in self.datasets[name]}&{r['source_id'] for r in chosen}
        cal=rows(self.transfer_data/'synth_calibration.jsonl')
        for field in ('family_id','occupation_pair','pair_id'):
            assert not {r[field] for r in chosen}&{r[field] for r in self.datasets['synth_test']+cal+previous}
        source_files=[self.config,Path(__file__),ROOT/'research/2026-10-14_evidence_mechanism_protocol_v1.md',
            self.source/'probes.npz',self.source/'direction_dim.npy',self.source/'test_plan_frozen.json',
            ROOT/'data/prepared/fairsteer_improve_20261013_v2/splits.json',self.transfer_data/'manifest.json',
            ROOT/'mechanism/closed_transfer.py',ROOT/'mechanism/fairsteer_author.py']
        for key,values in self.datasets.items():
            path=self.data/f'{key}.jsonl';text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values)
            if path.exists():assert path.read_text(encoding='utf-8')==text
            else:path.write_text(text,encoding='utf-8')
        identity={'files':{str(p.relative_to(ROOT)):sha(p) for p in source_files},
            'data':{k:sha(self.data/f'{k}.jsonl') for k in self.datasets},
            'counts':{k:len(v) for k,v in self.datasets.items()},
            'model':read(self.source/'environment.json')['model_hashes']}
        path=self.out/'plan_frozen.json'
        if path.exists():assert read(path)['identity']==identity,'Use a new experiment version after implementation changes'
        else:
            write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity})
            for p in source_files:
                target=self.out/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        self.log('prepared',counts=identity['counts'],new_confirmation_questions=len(chosen)//3)

    def setup(self):
        if self.model is not None:return
        torch.set_num_threads(self.c['cpu_threads']);torch.manual_seed(self.c['seed'])
        assert all(sha(Path(self.c['model_path'])/n)==v for n,v in read(self.out/'plan_frozen.json')['identity']['model'].items())
        self.log('loading_model')
        self.model=AutoModelForCausalLM.from_pretrained(self.c['model_path'],local_files_only=True,
            dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False)
        write(self.out/'environment.json',{'torch':torch.__version__,'transformers':__import__('transformers').__version__,
            'gpu':torch.cuda.get_device_name(),'dtype':'float16','batch_size':self.c['batch_size'],'layer':self.c['layer'],
            'candidate_ids':self.letter_ids})

    def forward(self,values,mode='full',vector=None):
        self.setup();enc=self.tokenizer([render(r,mode) for r in values],padding=True,return_tensors='pt',add_special_tokens=False)
        assert enc.input_ids.shape[1]<=self.c['max_tokens'],'No truncation allowed'
        inputs={k:v.to('cuda') for k,v in enc.items()};inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).clamp(min=0)
        captures=[];v=torch.as_tensor(vector,device='cuda',dtype=torch.float64) if vector is not None else None
        def hook(module,args,output):
            h=output[0] if isinstance(output,tuple) else output;captures.append(h[:,-1].detach().float().cpu().numpy())
            if v is None:return output
            changed=h.clone();changed[:,-1]=(h[:,-1].double()+v).to(h.dtype)
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        handle=self.model.model.layers[self.c['layer']].register_forward_hook(hook)
        try:
            with torch.inference_mode():out=self.model(**inputs,use_cache=False,logits_to_keep=1)
        finally:handle.remove()
        logits=out.logits[:,-1].float();assert torch.isfinite(logits).all()
        scores=logits[:,self.letter_ids].cpu().numpy()
        for i,row in enumerate(values):scores[i,len(row['choices']):]=-1e9
        assert len(captures)==1 and np.isfinite(captures[0]).all()
        return scores,captures[0]

    def smoke(self):
        if (self.out/'smoke.json').exists():return
        vs=self.datasets['bbq_dev'][:4];base,x=self.forward(vs);zero,zx=self.forward(vs,vector=np.zeros_like(self.vector))
        assert np.array_equal(base,zero) and np.array_equal(x,zx)
        edited,_=self.forward(vs,vector=1.5*self.vector);restored,_=self.forward(vs)
        assert np.array_equal(base,restored) and not np.array_equal(base,edited)
        s,sx=self.forward(vs[:1]);inputs=self.tokenizer(render(vs[0]),return_tensors='pt').to('cuda')
        with torch.inference_mode():ref=self.model(**inputs,use_cache=False,output_hidden_states=True,logits_to_keep=1)
        assert np.array_equal(sx[0],ref.hidden_states[self.c['layer']+1][0,-1].float().cpu().numpy())
        write(self.out/'smoke.json',{'zero_edit_exact':True,'hook_restoration_exact':True,'hidden_state_hook_exact':True,
            'edit_changes_scores':True,'batch_single_max_score_delta':float(abs(base[0]-s[0]).max()),
            'cad_zero_exact':bool(np.array_equal(cad_scores(base,base,0),base))})
        self.log('smoke_complete')

    def infer(self,key):
        if key.startswith('wino'):return
        values=self.datasets[key];parts=[];folder=self.out/'shards'/key;folder.mkdir(parents=True,exist_ok=True)
        old=None
        if key=='synth_test':old=dict(np.load(self.transfer/'cache_synth_test.npz'));assert old['uids'].tolist()==[r['uid'] for r in values]
        for start in range(0,len(values),self.c['batch_size']):
            vs=values[start:start+self.c['batch_size']];path=folder/f'{start:06d}.npz'
            ident=digest([sha(self.out/'plan_frozen.json'),key,vs])
            if path.exists():
                saved=dict(np.load(path));assert str(saved.pop('identity'))==ident
            else:
                if old is None:
                    base,x=self.forward(vs);cf1,_=self.forward(vs,vector=self.vector);cf15,_=self.forward(vs,vector=1.5*self.vector)
                else:
                    sl=slice(start,start+len(vs));base=old['base'][sl];x=old['features'][sl];cf1=old['cf1'][sl];cf15=old['cf15'][sl]
                no,_=self.forward(vs,'no_context');instruction,_=self.forward(vs,'instruction')
                saved={'base':base,'features':x,'cf1':cf1,'cf15':cf15,'no_context':no,'instruction':instruction}
                # Random directions are only evaluated on the primary option ordering.
                primary=np.array([r['permutation']==0 for r in vs])
                for seed in self.c['random_seeds']:
                    scores=base.copy()
                    if primary.any():
                        rng=np.random.default_rng(seed);v=rng.normal(size=self.vector.shape);v*=1.5*np.linalg.norm(self.vector)/np.linalg.norm(v)
                        scores[primary],_=self.forward([r for r,m in zip(vs,primary) if m],vector=v)
                    saved[f'random_{seed}']=scores
                np.savez(path,**saved,identity=ident)
            parts.append(saved)
            if start%(self.c['batch_size']*50)==0:self.log('inference',dataset=key,done=start+len(vs),total=len(values))
        cache={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};cache['uids']=np.array([r['uid'] for r in values])
        np.savez(self.out/f'cache_{key}.npz',**cache);self.log('inference_complete',dataset=key,rows=len(values))

    def cache(self,key):
        path=(self.transfer if key.startswith('wino') else self.out)/f'cache_{key}.npz'
        saved=dict(np.load(path));assert saved['uids'].tolist()==[r['uid'] for r in self.datasets[key]]
        return saved

    def select(self):
        path=self.out/'selection_frozen.json'
        if path.exists():return read(path)
        values=self.datasets['bbq_dev'];cache=self.cache('bbq_dev');signals=probe_values(cache['features'],self.probes)
        evidence=expit(signals['evidence']);output={'selection_dataset':'BBQ dev only, zero-shot','cad':{}}
        for gate,name in ((False,'cad'),(True,'gated_cad')):
            selected,candidates=choose_cad(values,cache['base'],cache['no_context'],evidence,self.c['betas'],gate,
                self.c['clear_tolerance'],self.c['ambig_tolerance'])
            output['cad'][name]={'selected':selected,'candidates':candidates}
        # Match abstention frequency on source dev, without fitting target outcomes.
        trigger=signals['evidence_gate']>.25;edited=cache['base'].copy();edited[trigger]=cache['cf15'][trigger]
        target=float(np.mean(edited.argmax(1)==np.array([r['unknown_index'] for r in values])))
        offset_candidates=[]
        for amount in np.arange(0,16.0001,.25):
            scores=offset_scores(cache['base'],values,float(amount),trigger)
            rate=float(np.mean(scores.argmax(1)==np.array([r['unknown_index'] for r in values])))
            offset_candidates.append({'amount':float(amount),'unknown_rate':rate})
        output['matched_offset']=min(offset_candidates,key=lambda r:(abs(r['unknown_rate']-target),r['amount']))
        output['matched_offset']['target_unknown_rate']=target
        write(path,output);self.log('selection_complete',cad={k:v['selected']['beta'] for k,v in output['cad'].items()},offset=output['matched_offset'])
        return output

    def policies(self,key):
        cache=self.cache(key);values=self.datasets[key];sig=probe_values(cache['features'],self.probes);selection=read(self.out/'selection_frozen.json')
        base=cache['base'];score={'baseline':base};triggers={}
        for name,gate,threshold,cf in [('fairsteer','fs_gate',.5,'cf1'),('evidence_steer','evidence_gate',.25,'cf15')]:
            use=sig[gate]>threshold;s=base.copy();s[use]=cache[cf][use];score[name]=s;triggers[name]=use
        score['same_gate_unknown_offset']=offset_scores(base,values,selection['matched_offset']['amount'],sig['evidence_gate']>.25)
        if 'no_context' in cache:
            score['evidence_instruction']=cache['instruction']
            for name,gate in [('cad',False),('gated_cad',True)]:
                beta=selection['cad'][name]['selected']['beta'];mask=expit(sig['evidence'])<=.25 if gate else None
                score[name]=cad_scores(base,cache['no_context'],beta,mask)
                for beta in self.c['betas'][1:]:score[f'diagnostic_{name}_{beta:g}']=cad_scores(base,cache['no_context'],beta,mask)
            for seed in self.c['random_seeds']:
                s=base.copy();use=(sig['evidence_gate']>.25)&np.array([r['permutation']==0 for r in values]);s[use]=cache[f'random_{seed}'][use]
                score[f'random_{seed}']=s
        pred={n:s.argmax(1) for n,s in score.items()}
        unknown=np.array([r['unknown_index'] for r in values])
        for name,gate in [('evidence_direct','direct_gate'),('same_gate_direct','evidence_gate')]:
            p=base.argmax(1).copy();use=(sig[gate]>.25)&(unknown>=0);p[use]=unknown[use];pred[name]=p
        return score,pred

    def evaluate(self):
        report={};comparisons={};pred_path=self.out/'predictions.jsonl';detail_path=self.out/'decomposition.jsonl'
        with pred_path.open('w',encoding='utf-8') as pf,detail_path.open('w',encoding='utf-8') as df:
            for key,values in self.datasets.items():
                if key=='bbq_dev':continue
                score,pred=self.policies(key);base=score['baseline'];result={}
                positions=sorted({r['permutation'] for r in values})
                for name,p in pred.items():
                    result[name]={}
                    for pos in positions:
                        if name.startswith('random_') and pos!=0:continue
                        use=np.array([r['permutation']==pos for r in values]);vs=[r for r,m in zip(values,use) if m]
                        m=metric(vs,p[use]);m['transitions']=transition(vs,base[use].argmax(1),p[use])
                        if name in score:
                            records=decomposition(vs,base[use],score[name][use]);m['decomposition']=summarize_decomposition(records)
                            for r in records:df.write(json.dumps({'dataset':key,'method':name,**r})+'\n')
                        result[name][str(pos)]=m
                    for r,guess in zip(values,p):pf.write(json.dumps({'dataset':key,'method':name,'uid':r['uid'],'prediction':int(guess),'label':r['label']})+'\n')
                report[key]=result
                main=np.array([r['permutation']==0 for r in values]);vs=[r for r,m in zip(values,main) if m];comparisons[key]={}
                pairs=[('evidence_steer','fairsteer'),('same_gate_direct','evidence_steer')]
                if 'cad' in pred:pairs += [('cad','baseline'),('gated_cad','baseline'),('evidence_instruction','baseline')]
                for a,b in pairs:
                    for condition in ('all','clear'):
                        mask=main if condition=='all' else main&np.array([r['condition']=='clear' for r in values])
                        sub=[r for r,m in zip(values,mask) if m]
                        if not sub:continue
                        comparisons[key][f'{a}_vs_{b}_{condition}']=bootstrap_difference(sub,pred[a][mask],pred[b][mask],self.c['bootstrap_replicates'],self.c['seed'])
        write(self.out/'results.json',{'metrics':report,'comparisons':comparisons,
            'scope':'Historic test sets are exploratory; SynthBias confirmation uses new reserved occupation/template families. Multiple diagnostic beta results are not test-selected winners.'})
        self.log('evaluation_complete')

    def run(self,stage):
        self.prepare()
        if stage=='prepare':return
        if stage in ('all','infer'):
            self.smoke();self.infer('bbq_dev');self.select()
            for key in ('bbq_test','synth_test','synth_confirmation'):self.infer(key)
            self.model=None;gc.collect();torch.cuda.empty_cache()
        if stage in ('all','evaluate'):self.evaluate()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--stage',choices=['all','prepare','infer','evaluate'],default='all')
    a=p.parse_args();Experiment(a.config).run(a.stage)


if __name__=='__main__':main()
