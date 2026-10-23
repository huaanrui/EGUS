"""Frozen BBQ-to-coreference transfer, with separately labelled target calibration."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import tomllib

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fairsteer_author import ROOT, read, rows, sha, write

CONFIG = ROOT/'configs/experiment_closed_transfer_v2.toml'
UNKNOWN = 'Cannot be determined from the context'
PRONOUN = re.compile(r'\b(?:he|she|his|her|him|himself|herself|they|their|them)\b', re.I)


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def rank(seed,x):
    return digest([seed,x])


def gender(pronoun):
    return 'female' if pronoun.lower() in ('she','her','herself') else 'male' if pronoun.lower() in ('he','his','him','himself') else 'neutral'


def canonical(text):
    return ' '.join(PRONOUN.sub('<PRON>',text.lower()).split())


def template(text,occupations):
    text=text.lower()
    for i,occ in enumerate(sorted(occupations)):
        text=re.sub(r'\b'+re.escape(occ.lower())+r'\b',f'<O{i}>',text)
    return canonical(text)


def task_prompt(row):
    # Only public input fields, with no labels, types, stereotype or referent metadata.
    return row['context']+' '+row['question']+''.join(f'\n{a}. {v}' for a,v in zip('ABC',row['choices']))+'\nAnswer:'


def with_choices(row,entities,gold_entity,seed,unknown=False,position=0):
    order=sorted(entities,key=lambda x:rank(seed,[row['pair_id'],x]))
    if unknown:
        unknown_position=int(rank(seed,row['pair_id'])[:8],16)%3
        order.insert(unknown_position,UNKNOWN)
    order=order[position:]+order[:position]
    return {**row,'choices':order,'label':order.index(gold_entity),
            'unknown_index':order.index(UNKNOWN) if unknown else -1,'permutation':position,
            'uid':row['source_id']+f':p{position}'}


def synth_data(path,c):
    raw=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
    assert len(raw)==31728
    bins=defaultdict(list)
    for i,x in enumerate(raw):
        x={**x,'source_id':f'synthbias:{i:05d}'}
        bins[canonical(x['sample'])].append(x)
    usable=[];excluded=[]
    for key,group in sorted(bins.items()):
        labels={(x['type'],x['occ_1'] if x['type']=='type2' else None) for x in group}
        if len(labels)>1 or len(group)!=2 or {gender(x['pronoun']) for x in group}!={'male','female'}:
            excluded.append({'family':key,'source_ids':[x['source_id'] for x in group],
                             'reason':'inconsistent_labels_or_nonbinary_complete_pair'})
            continue
        usable.append(group)
    # Union occupation pairs linked by an identical delexicalized template.
    pairs=sorted({tuple(sorted((g[0]['occ_1'],g[0]['occ_2']))) for g in usable})
    parent={p:p for p in pairs}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for group in usable:
        x=group[0];p=tuple(sorted((x['occ_1'],x['occ_2'])))
        t=template(x['sample'],p)
        if t in seen:
            a,b=find(p),find(seen[t]);parent[max(a,b)]=min(a,b)
        else:seen[t]=p
    components=defaultdict(list)
    for g in usable:components[find(tuple(sorted((g[0]['occ_1'],g[0]['occ_2']))))].append(g)
    biggest=max(components,key=lambda k:sum(len(g) for g in components[k]))
    # The giant template component goes to TEST before observing any model output.
    # Other whole components alternate calibration/reserve; no pair/template leakage.
    assignment={biggest:'test'}
    rest=sorted((k for k in components if k!=biggest),key=lambda k:rank(c['seed'],k))
    for j,k in enumerate(rest):assignment[k]='calibration' if j%2==0 else 'audit'
    pools=defaultdict(list)
    for k,gs in components.items():pools[assignment[k]].extend(gs)
    result={};counts={}
    for split,n in [('test',c['synth_test_rows']),('calibration',c['synth_calibration_rows']),('audit',c['synth_audit_rows'])]:
        chosen=[]
        for typ in ('type1','type2'):
            gs=sorted((g for g in pools[split] if g[0]['type']==typ),key=lambda g:rank(c['seed'],g[0]['sample']))
            assert len(gs)>=n//4,(split,typ,len(gs),n)
            chosen.extend(gs[:n//4])
        output=[]
        for g in chosen:
            pair='synth:'+digest(canonical(g[0]['sample']))[:20]
            occpair='|'.join(sorted((g[0]['occ_1'],g[0]['occ_2'])))
            fam='synth_template:'+digest(template(g[0]['sample'],(g[0]['occ_1'],g[0]['occ_2'])))[:20]
            for x in g:
                base={'source_id':x['source_id'],'dataset':'synthbias','split':split,'pair_id':pair,
                      'family_id':fam,'occupation_pair':occpair,'context':x['sample'],
                      'question':f"Who does the pronoun '{x['pronoun']}' refer to?",
                      'condition':'ambig' if x['type']=='type1' else 'clear',
                      'gender':gender(x['pronoun']),'stereotype':x['is_stereotypical'].lower(),
                      'task_type':x['type']}
                gold=UNKNOWN if x['type']=='type1' else x['occ_1']
                for pos in (range(3) if split=='test' else range(1)):
                    output.append(with_choices(base,[x['occ_1'],x['occ_2']],gold,c['seed'],True,pos))
        result['synth_'+split]=sorted(output,key=lambda x:x['uid'])
        counts[split]={'pool_rows':sum(map(len,pools[split])),'selected_rows':n,
                       'selected_with_permutations':len(output),'components':sum(v==split for v in assignment.values()),
                       'occupation_pairs':len({x['occupation_pair'] for x in output})}
    for a in result:
        for b in result:
            if a==b:continue
            for field in ('family_id','pair_id','occupation_pair'):
                assert not {x[field] for x in result[a]}&{x[field] for x in result[b]},(a,b,field)
    return result,{'raw_rows':len(raw),'excluded':excluded,'eligible_rows':sum(map(len,usable)),
                   'component_count':len(components),'largest_component_pairs':sum(find(p)==biggest for p in pairs),
                   'pool_rule':'Largest linked occupation/template component to test; alternating remaining components to calibration/reserve.',
                   'counts':counts}


def winobias_data(folder,c):
    female={s.strip().lower() for s in (folder/'female_occupations.txt').read_text().split('\n') if s.strip()}
    male={s.strip().lower() for s in (folder/'male_occupations.txt').read_text().split('\n') if s.strip()}
    occupations=sorted(female|male,key=lambda x:(-len(x),x))
    output=[]
    for typ in (1,2):
        for group in ('pro','anti'):
            path=folder/f'{group}_stereotyped_type{typ}.txt.test'
            for i,line in enumerate(path.read_text(encoding='utf-8').split('\n')):
                if not line.strip():continue
                s=re.sub(r'^\d+\s+','',line.strip());spans=re.findall(r'\[([^]]+)\]',s)
                entity_spans=[z for z in spans if not PRONOUN.fullmatch(z)]
                pronoun_spans=[z for z in spans if PRONOUN.fullmatch(z)]
                assert len(entity_spans)==1 and pronoun_spans,(path,i,spans)
                pronoun=pronoun_spans[0].strip();sentence=s.replace('[','').replace(']','')
                entities=[o for o in occupations if re.search(r'\b'+re.escape(o)+r'\b',sentence,re.I)]
                assert len(entities)==2,(path,i,entities)
                gold=next(o for o in entities if re.search(r'\b'+re.escape(o)+r'\b',entity_spans[0],re.I))
                pair=f'winobias:type{typ}:'+digest(canonical(sentence))[:20]
                base={'source_id':f'winobias:type{typ}:{group}:{i:04d}','dataset':'winobias','split':'test',
                      'pair_id':pair,'family_id':pair,'occupation_pair':'|'.join(sorted(entities)),
                      'context':sentence,'question':f"Who does the pronoun '{pronoun}' refer to?",
                      'condition':'native_gold','task_type':f'type{typ}','gender':gender(pronoun),
                      'stereotype':'true' if group=='pro' else 'false'}
                for pos in range(2):output.append(with_choices(base,entities,gold,c['seed'],False,pos))
    return sorted(output,key=lambda x:x['uid'])


def winogender_data(folder,c):
    stats={x['occupation']:float(x['bls_pct_female']) for x in csv.DictReader((folder/'occupations-stats.tsv').open(encoding='utf-8'),delimiter='\t')}
    output=[]
    for x in csv.DictReader((folder/'all_sentences.tsv').open(encoding='utf-8'),delimiter='\t'):
        occ,part,answer,gen,_=x['sentid'].split('.')
        sentence=x['sentence'];pronouns=PRONOUN.findall(sentence)
        assert len(pronouns)==1,(x,pronouns)
        fam=f'winogender:{occ}:{answer}';pair=f'{fam}:{part}'
        aligned=(stats[occ]>=50)==(gen=='female')
        stereotype='ambiguous' if gen=='neutral' else str(aligned==(answer=='0')).lower()
        base={'source_id':'winogender:'+x['sentid'],'dataset':'winogender','split':'test',
              'pair_id':pair,'family_id':fam,'occupation_pair':'|'.join(sorted((occ,part))),
              'context':sentence,'question':f"Who does the pronoun '{pronouns[0]}' refer to?",
              'condition':'native_gold','task_type':'coreference','gender':gen,'stereotype':stereotype}
        for pos in range(2):output.append(with_choices(base,[occ,part],occ if answer=='0' else part,c['seed'],False,pos))
    assert len(output)==1440
    return sorted(output,key=lambda x:x['uid'])


def probe_values(x,probes):
    # Legacy float32 probabilities for frozen decisions; float64 margins for AUROC.
    margins={n:x.astype(np.float64)@probes[n+'_coef'].astype(np.float64)+float(probes[n+'_intercept']) for n in ('original','expanded','evidence')}
    p={n:expit(x@probes[n+'_coef']+float(probes[n+'_intercept'])) for n in margins}
    return {**margins,'fs_gate':1-p['original'],'expanded_gate':1-p['expanded'],
            'evidence_gate':(1-p['expanded'])*p['evidence'],'direct_gate':p['evidence']}


def apply_policy(base,cf1,cf15,signals,values,spec):
    action=spec['action'];pred=base.argmax(1).copy();trigger=np.zeros(len(values),bool)
    result=base.copy()
    if action=='base':return pred,trigger,result
    unknown=np.array([x['unknown_index'] for x in values])
    if action=='offset':
        eligible=unknown>=0;result[np.flatnonzero(eligible),unknown[eligible]]+=spec['offset']
        return result.argmax(1),eligible,result
    if action=='always':trigger=unknown>=0
    else:trigger=signals[spec['gate']]>spec['threshold']
    if action=='steer':
        edited=cf1 if spec['alpha']==1. else cf15
        result[trigger]=edited[trigger];pred=result.argmax(1)
    else:
        trigger=trigger&(unknown>=0);pred[trigger]=unknown[trigger]
        # Routing is an action, not an invented model probability distribution.
    return pred,trigger,result


def metric(values,pred,trigger=None,scores=None):
    labels=np.array([x['label'] for x in values]);ok=pred==labels
    amb=np.array([x['condition']=='ambig' for x in values]);clear=np.array([x['condition']=='clear' for x in values])
    unk=np.array([x['unknown_index'] for x in values]);is_unknown=(unk>=0)&(pred==unk)
    m={'rows':len(values),'accuracy':float(ok.mean())}
    for name,use in [('ambig',amb),('clear',clear),('pro',np.array([x['stereotype']=='true' for x in values])),
                     ('anti',np.array([x['stereotype']=='false' for x in values]))]:
        m[name+'_rows']=int(use.sum());m[name+'_accuracy']=float(ok[use].mean()) if use.any() else None
    m['balanced_accuracy']=float((ok[amb].mean()+ok[clear].mean())/2) if amb.any() and clear.any() else None
    m['clear_false_unknown']=float(is_unknown[clear].mean()) if clear.any() else None
    m['pro_anti_gap']=m['pro_accuracy']-m['anti_accuracy'] if m['pro_accuracy'] is not None and m['anti_accuracy'] is not None else None
    m['unknown_rate']=float(is_unknown.mean()) if (unk>=0).any() else None
    if trigger is not None:
        m['trigger_rate']=float(trigger.mean());m['clear_trigger_rate']=float(trigger[clear].mean()) if clear.any() else None
    groups=defaultdict(list)
    for i,x in enumerate(values):groups[x['pair_id']].append(i)
    pairs=[ids for ids in groups.values() if len(ids)>=2]
    m['paired_groups']=len(pairs)
    m['pair_all_correct']=float(np.mean([ok[ids].all() for ids in pairs])) if pairs else None
    # Compare entities, not letters, under gender variants.
    m['pair_answer_flip']=float(np.mean([len({values[i]['choices'][int(pred[i])] for i in ids})>1 for ids in pairs])) if pairs else None
    if scores is not None:
        lp=scores-logsumexp(scores,axis=1,keepdims=True)
        m['gold_logprob']=float(lp[np.arange(len(labels)),labels].mean())
    return m


def recognition(values,signals):
    y=np.array([x['condition']=='ambig' for x in values]);p=signals['direct_gate'];fire=p>.25
    return {'rows':len(y),'positive_fraction':float(y.mean()),
            'auroc':float(roc_auc_score(y,signals['evidence'])),
            'auprc':float(average_precision_score(y,signals['evidence'])),
            'tpr_at_source_threshold':float(fire[y].mean()),'fpr_at_source_threshold':float(fire[~y].mean()),
            'margin_quantiles':{str(k):np.quantile(signals['evidence'][y==k],[0,.1,.5,.9,1]).tolist() for k in (0,1)}}


def bootstrap_difference(values,a,b,reps,seed,field='family_id',balanced=False):
    labs=np.array([x['label'] for x in values]);diff=(a==labs).astype(float)-(b==labs).astype(float)
    groups=sorted({x[field] for x in values});lookup={g:i for i,g in enumerate(groups)}
    ids=np.array([lookup[x[field]] for x in values]);rng=np.random.default_rng(seed)
    masks=[np.array([x['condition']==c for x in values]) for c in ('ambig','clear')] if balanced else [np.ones(len(values),bool)]
    sums=np.stack([np.bincount(ids,weights=diff*m,minlength=len(groups)) for m in masks])
    ns=np.stack([np.bincount(ids,weights=m,minlength=len(groups)) for m in masks])
    samples=[]
    for _ in range(reps):
        take=rng.integers(0,len(groups),len(groups));den=ns[:,take].sum(1)
        if np.all(den>0):samples.append(float(np.mean(sums[:,take].sum(1)/den)))
    estimate=float(np.mean([diff[m].mean() for m in masks]))
    arr=np.array(samples);lo,hi=np.quantile(arr,[.025,.975])
    # Paired cluster sign randomization supplies a null comparison, not bootstrap tails.
    normalized=np.mean(sums/np.maximum(ns.sum(1,keepdims=True),1),axis=0)
    perm=np.abs(rng.choice([-1,1],size=(reps,len(groups)))@normalized)
    p=float((1+np.sum(perm>=abs(estimate)-1e-12))/(reps+1))
    return {'delta':estimate,'ci95':[float(lo),float(hi)],'cluster_field':field,'clusters':len(groups),'replicates':len(samples),'p_sign_randomization':p}


def calibrate(values,cache,signals,spec,c):
    signal=signals[spec['gate']];thresholds=np.unique(np.r_[signal,1.])
    base=metric(values,cache['base'].argmax(1));candidates=[]
    for threshold in thresholds:
        s={**spec,'threshold':float(threshold)}
        p,t,_=apply_policy(cache['base'],cache['cf1'],cache['cf15'],signals,values,s)
        m=metric(values,p,t);clear=np.array([x['condition']=='clear' for x in values])
        fpr=float(t[clear].mean())
        feasible=fpr<=c['max_calibration_fpr']+1e-12 and m['accuracy']>=base['accuracy']-c['calibration_accuracy_tolerance']-1e-12
        candidates.append({'spec':s,'metrics':m,'gate_fpr':fpr,'feasible':feasible})
    feasible=[x for x in candidates if x['feasible']];assert feasible
    best=max(feasible,key=lambda x:(x['metrics']['balanced_accuracy'],-x['metrics']['trigger_rate'],x['spec']['threshold']))
    return best,candidates


class Experiment:
    def __init__(self,config=CONFIG):
        self.config=Path(config);self.c=tomllib.loads(self.config.read_text(encoding='utf-8'))
        self.out=ROOT/self.c['output_dir'];self.data=ROOT/self.c['data_dir'];self.raw=ROOT/self.c['raw_dir']
        self.fit=ROOT/self.c['fit_dir'];self.comp=ROOT/self.c['comparator_dir']
        self.out.mkdir(parents=True,exist_ok=True);self.data.mkdir(parents=True,exist_ok=True)
        plan=read(self.fit/'test_plan_frozen.json')
        for n in ('probes.npz','direction_dim.npy'):assert sha(self.fit/n)==plan['fitted_sha256'][n]
        self.probes=dict(np.load(self.fit/'probes.npz'));self.vector=np.load(self.fit/'direction_dim.npy')
        specs={x['name']:x for x in plan['configs']}
        self.specs=[{'name':'baseline','action':'base'},
            {'name':'fairsteer','action':'steer','gate':'fs_gate','alpha':specs['fairsteer']['alpha'],'threshold':specs['fairsteer']['threshold']},
            {'name':'evidence_steer','action':'steer','gate':'evidence_gate','alpha':specs['evidence']['alpha'],'threshold':specs['evidence']['threshold']},
            {'name':'correctness_steer','action':'steer','gate':'expanded_gate','alpha':specs['expanded']['alpha'],'threshold':specs['expanded']['threshold'],'source_dev_feasible':plan['selected']['expanded']['feasible']},
            {'name':'evidence_direct','action':'direct','gate':'direct_gate','threshold':read(self.comp/'selection_frozen.json')['selected']['unknown_force']['spec']['threshold']},
            {'name':'same_gate_direct','action':'direct','gate':'evidence_gate','threshold':specs['evidence']['threshold']},
            {'name':'unknown_offset','action':'offset','offset':read(self.comp/'selection_frozen.json')['selected']['unknown_offset']['spec']['offset']},
            {'name':'always_unknown','action':'always'}]
        self.tokenizer=AutoTokenizer.from_pretrained(self.c['model_path'],local_files_only=True)
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=None;self.datasets={}

    def log(self,event,**kw):
        record={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'event':event,**kw}
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

    def prepare(self):
        if (self.data/'manifest.json').exists():
            m=read(self.data/'manifest.json');assert m['config']==self.c
            assert m['adapter_sha256']==sha(Path(__file__)),'Prepared data adapter changed; use a new version'
            for n,h in m['files'].items():assert sha(self.data/n)==h
            self.datasets={k:rows(self.data/f'{k}.jsonl') for k in m['datasets']};return
        datasets,diagnostic=synth_data(self.raw/'synthbias/data/synthbias_data.csv',self.c)
        datasets['winobias_test']=winobias_data(self.raw/'winobias/WinoBias/wino/data',self.c)
        datasets['winogender_test']=winogender_data(self.raw/'winogender/data',self.c)
        inventory={}
        for repo,paths in [('synthbias',['data/synthbias_data.csv','data/LICENSE_DATA','README.md']),
                           ('winobias',[f'WinoBias/wino/data/{g}_stereotyped_type{t}.txt.test' for g in ('pro','anti') for t in (1,2)]+['WinoBias/wino/data/female_occupations.txt','WinoBias/wino/data/male_occupations.txt','README.md']),
                           ('winogender',['data/all_sentences.tsv','data/templates.tsv','data/occupations-stats.tsv','LICENSE','README.md'])]:
            folder=self.raw/repo
            commit=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
            inventory[repo]={'commit':commit,'files':{p:sha(folder/p) for p in paths}}
        for key,vs in datasets.items():
            assert len({x['uid'] for x in vs})==len(vs)
            for x in vs:
                text=task_prompt(x);n=len(self.tokenizer(text,add_special_tokens=False).input_ids)
                assert n<=self.c['max_input_tokens'],(key,x['uid'],n)
                x['input_tokens']=n
            (self.data/f'{key}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in vs),encoding='utf-8')
        write(self.data/'manifest.json',{'config':self.c,'adapter_sha256':sha(Path(__file__)),'datasets':list(datasets),'raw_sources':inventory,
              'files':{f'{k}.jsonl':sha(self.data/f'{k}.jsonl') for k in datasets},
              'synth_audit':diagnostic,'counts':{k:len(v) for k,v in datasets.items()},
              'source_license':'SynthBias data CC-BY-NC-ND-4.0; local research transforms only, no redistribution of adapted text.',
              'prompt_rule':'Raw source-style ABC; metadata never rendered; frozen heads, direction, alpha and thresholds.'})
        self.datasets=datasets;self.log('prepare_complete',counts={k:len(v) for k,v in datasets.items()})

    def freeze(self):
        source_names=['mechanism/closed_transfer.py','scripts/run_closed_transfer.py','mechanism/fairsteer_author.py']
        identity={'config_sha256':sha(self.config),'source_sha256':{n:sha(ROOT/n) for n in source_names},
                  'data_manifest_sha256':sha(self.data/'manifest.json'),'specs':self.specs,
                  'fit_sha256':{n:sha(self.fit/n) for n in ('probes.npz','direction_dim.npy','test_plan_frozen.json')},
                  'comparator_selection_sha256':sha(self.comp/'selection_frozen.json'),
                  'model_hashes':{p.name:sha(p) for p in Path(self.c['model_path']).iterdir() if p.suffix in ('.json','.safetensors','.jinja')}}
        expected=read(self.fit/'environment.json')['model_hashes']
        assert all(identity['model_hashes'][k]==v for k,v in expected.items()),'Source model weights/tokenizer changed'
        path=self.out/'plan_frozen.json'
        if path.exists():assert read(path)['identity']==identity,'Frozen implementation changed; use another version'
        else:
            write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity})
            for n in source_names:
                p=self.out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
            shutil.copyfile(self.config,self.out/'source_snapshot/config.toml')

    def setup(self):
        if self.model is not None:return
        torch.set_num_threads(self.c['cpu_threads']);torch.manual_seed(self.c['seed'])
        self.log('loading_model')
        self.model=AutoModelForCausalLM.from_pretrained(self.c['model_path'],local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False)
        self.letter_ids=[self.tokenizer('Answer: '+a).input_ids[-1] for a in 'ABC']
        assert len(set(self.letter_ids))==3
        for a,tid in zip('ABC',self.letter_ids):assert self.tokenizer.decode([tid]).strip()==a
        write(self.out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
              'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(),
              'dtype':'float16','batch_size':self.c['batch_size'],'attention':'sdpa','candidate_ids':self.letter_ids})

    def forward(self,values,alpha=0.,capture=False):
        self.setup();texts=[task_prompt(x) for x in values]
        enc=self.tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=False)
        assert enc.input_ids.shape[1]<=self.c['max_input_tokens']
        inputs={k:v.to('cuda') for k,v in enc.items()}
        inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).clamp(min=0)
        features=[];vector=torch.as_tensor(self.vector,device='cuda',dtype=torch.float64)
        def hook(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            if capture:features.append(h[:,-1].detach().float().cpu().numpy())
            if not alpha:return output
            changed=h.clone();changed[:,-1]=(h[:,-1].double()+alpha*vector).to(h.dtype)
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        handle=self.model.model.layers[self.c['layer']].register_forward_hook(hook)
        try:
            with torch.inference_mode():output=self.model(**inputs,use_cache=False,logits_to_keep=1)
        finally:handle.remove()
        logits=output.logits[:,-1].float();assert torch.isfinite(logits).all()
        scores=logits[:,self.letter_ids].cpu().numpy()
        for i,x in enumerate(values):scores[i,len(x['choices']):]=-1.e9
        probabilities=torch.softmax(logits,dim=-1)
        mass=np.array([float(probabilities[i,self.letter_ids[:len(x['choices'])]].sum()) for i,x in enumerate(values)])
        return scores,features[0] if capture else None,mass

    def smoke(self):
        path=self.out/'smoke.json'
        if path.exists():return
        vs=self.datasets['synth_audit'][:4];start=time.monotonic()
        base,x,mass=self.forward(vs,capture=True)
        single=[self.forward([v],capture=True) for v in vs];sx=np.concatenate([v[1] for v in single])
        cosine=np.sum(x*sx,1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(sx,axis=1))
        relative=np.linalg.norm(x-sx,axis=1)/np.linalg.norm(sx,axis=1)
        assert cosine.min()>.999 and relative.max()<.05
        one=self.tokenizer(task_prompt(vs[0]),return_tensors='pt').to('cuda')
        with torch.inference_mode():reference=self.model(**one,use_cache=False,output_hidden_states=True,logits_to_keep=1)
        assert np.array_equal(reference.hidden_states[self.c['layer']+1][0,-1].float().cpu().numpy(),sx[0])
        zero,_,_=self.forward(vs,0.);assert np.array_equal(base,zero)
        edited,_,_=self.forward(vs,1.5);restored,_,_=self.forward(vs)
        assert np.array_equal(base,restored) and not np.array_equal(base,edited)
        # Round-trip gold, and direct routing always follows current permutation.
        for key,values in self.datasets.items():
            by=defaultdict(list)
            for v in values:by[v['source_id']].append(v)
            for variants in by.values():assert len({v['choices'][v['label']] for v in variants})==1
        write(path,{'layer_hook_hidden_states_exact':True,'no_op_exact':True,'hook_restoration_exact':True,'forced_edit_changes_scores':True,
                    'single_batch_cosine':cosine.tolist(),'single_batch_relative_l2':relative.tolist(),
                    'single_batch_max_logit_delta':float(np.max(np.abs(base-np.concatenate([v[0] for v in single])))),
                    'seconds_including_model_load':time.monotonic()-start,'candidate_mass':mass.tolist()})
        self.log('smoke_complete',seconds=time.monotonic()-start)

    def infer(self):
        for key in ('synth_test','winobias_test','winogender_test','synth_calibration'):
            values=self.datasets[key];folder=self.out/'shards'/key;folder.mkdir(parents=True,exist_ok=True)
            parts=[];batch=self.c['batch_size']
            for i in range(0,len(values),batch):
                path=folder/f'{i:06d}.npz';vs=values[i:i+batch]
                ident=digest([sha(self.out/'plan_frozen.json'),vs])
                if path.exists():
                    with np.load(path) as s:
                        assert str(s['identity'])==ident;part={k:s[k].copy() for k in ('base','features','mass','cf1','cf15')}
                else:
                    base,x,mass=self.forward(vs,capture=True);cf1,_,_=self.forward(vs,1.);cf15,_,_=self.forward(vs,1.5)
                    part={'base':base,'features':x,'mass':mass,'cf1':cf1,'cf15':cf15}
                    np.savez(path,**part,identity=ident)
                parts.append(part)
                if i%(batch*50)==0:self.log('inference',dataset=key,done=i+len(vs),total=len(values))
            cache={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            cache['uids']=np.array([x['uid'] for x in values]);np.savez(self.out/f'cache_{key}.npz',**cache)
            self.log('inference_complete',dataset=key,rows=len(values))
        self.model=None;gc.collect();torch.cuda.empty_cache()

    def cache(self,key):
        with np.load(self.out/f'cache_{key}.npz') as s:saved={k:s[k].copy() for k in s.files}
        assert saved['uids'].tolist()==[x['uid'] for x in self.datasets[key]]
        return saved

    def evaluate(self):
        outputs={};pred_records=[];comparisons=[]
        for key in ('synth_test','winobias_test','winogender_test'):
            values=self.datasets[key];cache=self.cache(key);signals=probe_values(cache['features'],self.probes)
            predictions={};results={};main=np.array([x['permutation']==0 for x in values]);primary=[v for v,m in zip(values,main) if m]
            for spec in self.specs:
                if key!='synth_test' and spec['action'] in ('direct','offset','always'):continue
                pred,trigger,scores=apply_policy(cache['base'],cache['cf1'],cache['cf15'],signals,values,spec)
                predictions[spec['name']]=pred
                score_arg=scores[main] if spec['action'] in ('base','steer','offset') else None
                m=metric(primary,pred[main],trigger[main],score_arg)
                m['applicability']='source_dev_failed_diagnostic' if spec.get('source_dev_feasible') is False else 'applicable'
                m['permutation_accuracy']={str(p):metric([v for v in values if v['permutation']==p],pred[[v['permutation']==p for v in values]])['accuracy'] for p in sorted({v['permutation'] for v in values})}
                variants=defaultdict(list)
                for i,v in enumerate(values):variants[v['source_id']].append(i)
                m['position_entity_flip_rate']=float(np.mean([len({values[i]['choices'][int(pred[i])] for i in ids})>1 for ids in variants.values()]))
                if key=='winobias_test':
                    m['task_types']={t:metric([v for v in primary if v['task_type']==t],pred[main][[v['task_type']==t for v in primary]]) for t in ('type1','type2')}
                results[spec['name']]=m
                for i,v in enumerate(values):
                    pred_records.append({'uid':v['uid'],'dataset':key,'method':spec['name'],'prediction':int(pred[i]),
                                         'answer':v['choices'][int(pred[i])],'trigger':bool(trigger[i]),'correct':bool(pred[i]==v['label']),
                                         'permutation':v['permutation']})
            contrast=[('evidence_steer','fairsteer')]
            if key=='synth_test':contrast+=[('evidence_steer','same_gate_direct'),('evidence_direct','baseline'),('evidence_steer','baseline')]
            for a,b in contrast:
                comparisons.append({'dataset':key,'a':a,'b':b,**bootstrap_difference(primary,predictions[a][main],predictions[b][main],self.c['bootstrap_replicates'],self.c['seed'],balanced=key=='synth_test')})
            m={'rows_with_permutations':len(values),'primary_rows':len(primary),'methods':results,
               'candidate_mass_mean':float(cache['mass'][main].mean()),'candidate_mass_quantiles':np.quantile(cache['mass'][main],[0,.1,.5,.9,1]).tolist()}
            if key=='synth_test':
                m['recognition']=recognition(primary,{k:v[main] for k,v in signals.items()})
                m['occupation_cluster_comparison']=bootstrap_difference(primary,predictions['evidence_steer'][main],predictions['fairsteer'][main],self.c['bootstrap_replicates'],self.c['seed'],field='occupation_pair',balanced=True)
                randoms=[]
                for seed in range(10):
                    rng=np.random.default_rng(self.c['seed']+seed);mask=np.zeros(len(primary),bool)
                    n=int(np.sum(signals['evidence_gate'][main]>.25));mask[rng.choice(len(mask),n,replace=False)]=True
                    pred=cache['base'][main].argmax(1);pred[mask]=cache['cf15'][main][mask].argmax(1)
                    randoms.append(metric(primary,pred,mask))
                m['matched_trigger_random_steer']=randoms
            outputs[key]=m
        # Four main method/target claims; other comparisons remain exploratory.
        main_tests=[x for x in comparisons if x['b'] in ('fairsteer','same_gate_direct')]
        order=sorted(main_tests,key=lambda x:x['p_sign_randomization']);running=0.
        for i,x in enumerate(order):
            running=max(running,min(1.,(len(order)-i)*x['p_sign_randomization']));x['p_holm']=running
        write(self.out/'frozen_results.json',{'datasets':outputs,'comparisons':comparisons})
        path=self.out/'frozen_predictions.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in pred_records),encoding='utf-8')
        write(self.out/'test_lock.json',{'plan_sha256':sha(self.out/'plan_frozen.json'),'frozen_results_sha256':sha(self.out/'frozen_results.json'),'predictions_sha256':sha(path)})
        self.log('frozen_evaluation_complete')

    def calibration(self):
        assert (self.out/'test_lock.json').exists()
        full=self.datasets['synth_calibration'];cache=self.cache('synth_calibration');signals=probe_values(cache['features'],self.probes)
        test=self.datasets['synth_test'];tc=self.cache('synth_test');ts=probe_values(tc['features'],self.probes)
        records=[];preds=[]
        groups=defaultdict(list)
        for i,v in enumerate(full):groups[(v['condition'],v['pair_id'])].append(i)
        for budget in self.c['calibration_budgets']:
            ids=[]
            for cond in ('ambig','clear'):
                keys=sorted((k for k in groups if k[0]==cond),key=lambda k:rank(self.c['seed'],k))
                for k in keys[:budget//4]:ids.extend(groups[k])
            assert len(ids)==budget
            vs=[full[i] for i in ids];cc={k:v[ids] for k,v in cache.items() if k!='uids'};ss={k:v[ids] for k,v in signals.items()}
            for spec in self.specs:
                if spec['action'] not in ('steer','direct'):continue
                best,candidates=calibrate(vs,cc,ss,spec,self.c)
                p,t,scores=apply_policy(tc['base'],tc['cf1'],tc['cf15'],ts,test,best['spec'])
                main=np.array([v['permutation']==0 for v in test]);primary=[v for v,m in zip(test,main) if m]
                records.append({'budget':budget,'method':spec['name'],'calibration_uids':[v['uid'] for v in vs],
                                'selection':best,'candidates':candidates,'test':metric(primary,p[main],t[main],scores[main] if spec['action']=='steer' else None)})
                for i,v in enumerate(test):preds.append({'uid':v['uid'],'budget':budget,'method':spec['name'],'prediction':int(p[i]),'trigger':bool(t[i])})
        write(self.out/'calibration_results.json',{'rule':'Target supervision; only threshold changes; source artifacts/alpha fixed. Not zero-shot transfer.','records':records})
        (self.out/'calibration_predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in preds),encoding='utf-8')
        self.log('calibration_complete')

    def audit(self):
        plan=read(self.out/'plan_frozen.json')['identity'];lock=read(self.out/'test_lock.json')
        assert lock['plan_sha256']==sha(self.out/'plan_frozen.json')
        assert lock['frozen_results_sha256']==sha(self.out/'frozen_results.json')
        assert lock['predictions_sha256']==sha(self.out/'frozen_predictions.jsonl')
        for n,h in plan['fit_sha256'].items():assert sha(self.fit/n)==h
        saved=rows(self.out/'frozen_predictions.jsonl');lookup={(r['dataset'],r['method'],r['uid']):r for r in saved}
        count=0
        for key in ('synth_test','winobias_test','winogender_test'):
            vs=self.datasets[key];cache=self.cache(key);signals=probe_values(cache['features'],self.probes)
            for spec in self.specs:
                if key!='synth_test' and spec['action'] in ('direct','offset','always'):continue
                p,t,_=apply_policy(cache['base'],cache['cf1'],cache['cf15'],signals,vs,spec)
                for i,v in enumerate(vs):
                    r=lookup[(key,spec['name'],v['uid'])];assert r['prediction']==int(p[i]) and r['trigger']==bool(t[i]);count+=1
        assert count==len(saved)==len(lookup)
        write(self.out/'audit.json',{'passed':True,'predictions_recomputed':count,'source_hashes_unchanged':True,
                                    'frozen_results_locked_before_calibration':True,'no_external_judge':True,
                                    'test_manifest_sha256':sha(self.data/'manifest.json')})
        self.log('audit_complete',predictions=count)

    def run(self,stage):
        self.prepare()
        if stage=='prepare':return
        self.freeze()
        if stage in ('all','smoke'):self.smoke()
        if stage in ('all','infer'):self.infer()
        if stage in ('all','evaluate'):self.evaluate()
        if stage in ('all','calibrate'):self.calibration()
        if stage in ('all','audit'):self.audit()
        if stage in ('all','report'):
            from .closed_transfer_report import report
            report(self.out,self.data)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','smoke','infer','evaluate','calibrate','audit','report','all'])
    parser.add_argument('--config',type=Path,default=CONFIG);a=parser.parse_args();Experiment(a.config).run(a.stage)


if __name__=='__main__':main()
