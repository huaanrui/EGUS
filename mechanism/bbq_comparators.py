"""Exploratory Qwen BBQ ports of KnowBias, CRISPR, FBA and BBA.

Paper and implementation differences are explicit in protocol_frozen.md.
Existing experiments are read-only. All new fits and selections are frozen.
"""
from __future__ import annotations
import argparse
import collections
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import tomllib

import numpy as np
from scipy.special import expit
import torch
import torch.nn.functional as F

from .fairsteer_author import ROOT, Runner, metrics, prompt, read, rows, sha, write
from .fairsteer_improve import balanced_indices, full_mmlu_prompt

CONFIG = ROOT/'configs/experiment_bbq_comparators_v2.toml'
KB = ROOT/'research/sources/20261012/KnowBias'
BBA = ROOT/'research/sources/20261013/BBA'
DOMAINS = ('gender', 'nationality', 'religion')
CATEGORY = {'gender':'Gender_identity','nationality':'Nationality','religion':'Religion'}


def rank(values):
    return np.argsort(-np.asarray(values).ravel(),kind='stable')


def knowledge_questions(concepts):
    with (KB/'data/know_bias_data_all_combine_extend.csv').open(encoding='utf-8-sig') as handle:
        source=list(csv.DictReader(handle))
    result=[]
    for dimension in ('gender','race','religion'):
        for row in source:
            if row['concept'] not in concepts:continue
            q=row['question'].replace('gender, race, or  religious affiliation', {'gender':'gender','race':'race','religion':'religious affiliation'}[dimension])
            q=q.replace('gender/race/religious group', {'gender':'gender','race':'race','religion':'religious group'}[dimension])
            q=q.replace('gender/race/religious bias', {'gender':'gender bias','race':'racial bias','religion':'religious bias'}[dimension])
            result.append({'dimension':dimension,'concept':row['concept'],'question':q,'target':row['gold']})
    assert len(result)==len(concepts)*9
    return result


def knowledge_set(scores, threshold, consistency):
    """Author rule: score > sample global max*tau, count > questions*beta.

    Identify each demographic's set and take their union, as final KnowBias.
    """
    selected=[]
    for domain in ('gender','race','religion'):
        use=[np.asarray(s['scores']) for s in scores if s['dimension']==domain]
        counts=sum((s>s.max()*threshold).astype(np.int32) for s in use)
        selected.extend(map(tuple,np.argwhere(counts>len(use)*consistency)))
    return np.array(sorted(set(selected)),dtype=np.int64).reshape(-1,2)


def crispr_attribution(activation, wrong_gradient, correct_gradient):
    return (activation*wrong_gradient-np.maximum(activation*correct_gradient,0)).max(axis=0)


def entropy_inverse(h, weights, candidate_ids):
    # Author FBA multiplies full-vocabulary probabilities at the same position
    # for multi-token candidates; this is not autoregressive word likelihood.
    logp=F.log_softmax(F.linear(h,weights).float(),dim=-1)
    lp=torch.stack([logp[...,ids].sum(-1) for ids in candidate_ids],dim=-1)
    p=lp.softmax(-1)
    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
    return 1/(entropy+1e-12)


def js_objective(h, weights, first_ids, temperature=.1):
    # Normalizing selected token probabilities cancels full-vocabulary Z.
    p=F.linear(h,weights[first_ids]).float().div(temperature).softmax(-1)
    mean=p.mean(1,keepdim=True)
    return (p*(p.clamp_min(1e-30).log()-mean.clamp_min(1e-30).log())).sum(-1).mean(1)


def readout_ig(h, weights, objective, steps=20):
    """One or multiple demographic states; float32 differentiation."""
    base=h.detach().float()
    alpha=torch.linspace(0,1,steps,device=h.device)
    scaled=(alpha.reshape(steps,*([1]*base.ndim))*base).requires_grad_(True)
    value=objective(scaled,weights)
    gradient=torch.autograd.grad(value.sum(),scaled)[0]
    attribution=(base*gradient.mean(0))
    if attribution.ndim>1:attribution=attribution.mean(0)
    if not torch.isfinite(attribution).all():raise FloatingPointError('Readout IG nonfinite')
    return attribution.detach().cpu().numpy()


def replace_readout(h, indices, constant):
    edited=h.clone();edited[...,indices]=constant;return edited


class Comparators:
    def __init__(self, config=CONFIG):
        self.config=Path(config).resolve();self.c=tomllib.loads(self.config.read_text(encoding='utf-8'))
        torch.set_num_threads(self.c['cpu_threads'])
        self.out=ROOT/self.c['output_dir'];self.out.mkdir(parents=True,exist_ok=True)
        self.parent=ROOT/self.c['parent_output'];self.author=ROOT/self.c['author_output']
        self.source=rows(ROOT/self.c['parent_data']/'bbq.jsonl');self.mmlu=rows(ROOT/self.c['parent_data']/'mmlu.jsonl');self.n=len(self.source)
        split=read(ROOT/self.c['split_file']);self.splits=split['bbq_indices'];self.msplits=split['mmlu_indices']
        mapping=read(ROOT/'configs/bbq_template_families.json')['mapping']
        for r in self.source:
            family=f"{r['category']}:{r['question_index']}"
            r['family']=mapping.get(family,family)
        self.runner=None;self._features=None;self._rawscores=None
        self.freeze()

    def freeze(self):
        paths=['mechanism/bbq_comparators.py','mechanism/fairsteer_author.py','mechanism/fairsteer_improve.py','mechanism/fairsteer_data.py','scripts/run_bbq_comparators.py','research/2026-10-13_bbq_comparator_protocol.md']
        inputs=[self.config,ROOT/self.c['split_file'],self.parent/'test_plan_frozen.json',self.parent/'probes.npz',self.parent/'results.json',self.parent/'selection_results.json',self.parent/'direction_dim.npy',self.author/'feature_hashes.json',ROOT/self.c['parent_data']/'bbq.jsonl',ROOT/self.c['parent_data']/'mmlu.jsonl',KB/'data/know_bias_data_all_combine_extend.csv']
        inputs+=list(BBA.rglob('*.json'))
        fingerprint={'sources':{p:sha(ROOT/p) for p in paths},'inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs}}
        path=self.out/'plan_frozen.json'
        if path.exists():assert read(path)['fingerprint']==fingerprint,'Frozen inputs changed: create a new experiment version'
        else:
            write(path,{'fingerprint':fingerprint,'created':time.strftime('%Y-%m-%d %H:%M:%S'),'config':self.c,'historical_exposure':'Exploratory: old BBQ test has been observed. No claim of pristine confirmation.'})
            for p in paths:
                target=self.out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,target)
            shutil.copyfile(self.config,self.out/'source_snapshot/config.toml')
            shutil.copyfile(ROOT/'research/2026-10-13_bbq_comparator_protocol.md',self.out/'protocol_frozen.md')
        self.plan_hash=sha(path)

    def log(self,event,**value):
        rec={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'event':event,**value}
        print(json.dumps(rec,ensure_ascii=False),flush=True)
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')

    def examples(self,split,utility=False):
        if utility:return [(self.mmlu[i],False,2*self.n+i) for i in self.msplits[split]]
        return [(self.source[i],shot,i+int(shot)*self.n) for shot in (False,True) for i in self.splits[split]]

    def setup(self):
        if self.runner is None:
            self.log('loading_model');self.runner=Runner(self.c)
            env={'model':{p.name:sha(p) for p in Path(self.c['model_path']).glob('*') if p.suffix in ('.json','.safetensors')},'torch':torch.__version__,'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(),'dtype':'float16','evaluation_batch':1}
            assert env['model']==read(self.author/'environment.json')['model']
            write(self.out/'environment.json',env)

    @property
    def rawscores(self):
        if self._rawscores is None:self._rawscores=np.load(self.author/'scores.npy',mmap_mode='r')
        return self._rawscores

    @property
    def features(self):
        if self._features is None:self._features=np.load(self.author/'features.npy',mmap_mode='r')
        return self._features

    def baseline(self,split,utility=False):
        if utility:return np.load(self.parent/'utility_baseline'/f'{split}.npz')['scores']
        return np.array(self.rawscores[[i for _,_,i in self.examples(split)]])

    def ids(self,text):
        value=self.runner.tokenizer(text,return_tensors='pt').input_ids.to('cuda')
        if value.shape[1]>self.c['max_tokens']:raise ValueError('No truncation allowed')
        return value

    def knowledge_prompt(self,question):
        # Missing evaluate_know_new.txt replaced explicitly, never claimed exact.
        return self.runner.tokenizer.apply_chat_template([{'role':'system','content':'You are a useful assistant.'},{'role':'user','content':'Answer the following question with only yes or no.\nQuestion: '+question}],tokenize=False,add_generation_prompt=True,enable_thinking=False)

    def raw_capture(self,text):
        captured={}
        def take(name):
            def hook(m,a,o):captured[name]=o[0,-1].detach().cpu().numpy().copy()
            return hook
        r=self.runner;h=r.layers[-1].register_forward_hook(take('pre'));hn=r.model.model.norm.register_forward_hook(take('post'))
        try:
            with torch.inference_mode():z=r.model(input_ids=self.ids(text),use_cache=False,logits_to_keep=1)
        finally:h.remove();hn.remove()
        return captured,z.logits[0,-1].float().cpu().numpy()

    def prepare(self):
        q=knowledge_questions(self.c['kb_concepts']);write(self.out/'knowledge_questions.json',q)
        chosen=balanced_indices(self.source,self.splits['fit'],self.c['crispr_per_category']//2,self.c['seed']+2)
        assert len(chosen)==11*self.c['crispr_per_category']
        write(self.out/'crispr_examples.json',{'indices':chosen,'uids':[self.source[i]['uid'] for i in chosen],'shots':[False,True],'aggregation':'max token, wrong-probability weighted instance, equal mean over two shot settings'})
        for a in self.splits:
            for b in self.splits:
                if a!=b:assert not {self.source[i]['family'] for i in self.splits[a]}&{self.source[i]['family'] for i in self.splits[b]}
        self.log('prepare_complete',knowledge_questions=len(q),crispr_questions=len(chosen))

    def smoke(self):
        if (self.out/'smoke.json').exists():return
        self.setup();r=self.runner;checks=[]
        for row,shot,i in self.examples('dev')[:2]+self.examples('dev')[-2:]:
            captured,logits=self.raw_capture(prompt(row,shots=shot))
            assert np.array_equal(logits[r.labels],self.rawscores[i])
            assert np.array_equal(captured['post'],self.features[i,-1])
            checks.append({'uid':row['uid'],'shot':shot,'exact_parent_scores':True})
        text=prompt(self.examples('dev')[0][0]);original=r.forward(text)[1]
        for method,spec in [('knowbias',{'nodes':[[18,0],[18,1]],'strength':10.}),('crispr',{'nodes':[[18,j] for j in range(128)],'strength':0.})]:
            handle=self.install_nodes(method,spec);changed=r.forward(text)[1];handle.remove();restored=r.forward(text)[1]
            assert not np.array_equal(original,changed) and np.array_equal(original,restored)
        q=read(self.out/'knowledge_questions.json')[0];kt=self.knowledge_prompt(q['question']);ids=self.ids(kt)
        with torch.no_grad():prefix=r.model(input_ids=ids[:,:-1],use_cache=True,logits_to_keep=1).past_key_values
        differences=[]
        for layer in (0,13,27):
            fast=self.kb_layer_ig(ids,q['target'],layer,prefix,steps=4,batch=1)
            full=self.kb_layer_ig(ids,q['target'],layer,None,steps=4,batch=1)
            cosine=float(np.dot(fast,full)/(np.linalg.norm(fast)*np.linalg.norm(full)))
            relative=float(np.linalg.norm(fast-full)/max(np.linalg.norm(full),1e-20))
            assert cosine>.99 and relative<.08,(layer,cosine,relative)
            differences.append({'layer':layer,'cosine':cosine,'relative_l2':relative})
        write(self.out/'smoke.json',{'checks':checks,'node_hooks_change_and_restore':True,'cached_prefix_ig_vs_full':differences,'plan_hash':self.plan_hash})
        self.log('smoke_complete',cached_prefix_ig=differences)

    def install_nodes(self,method,spec):
        """Single removable collection; KB gate preactivation vs CRISPR product."""
        handles=[];bylayer=collections.defaultdict(list)
        for layer,index in spec['nodes']:bylayer[int(layer)].append(int(index))
        for layer,indices in bylayer.items():
            index=torch.tensor(indices,device='cuda')
            if method=='knowbias':
                def hook(m,a,o,index=index):
                    changed=o.clone();changed[:,-1,index]*=spec['strength'];return changed
                handles.append(self.runner.layers[layer].mlp.gate_proj.register_forward_hook(hook))
            else:
                def hook(m,a,index=index):
                    changed=a[0].clone();changed[:,:,index]=0;return (changed,*a[1:])
                handles.append(self.runner.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        class Handles:
            def remove(self):
                for h in handles:h.remove()
        return Handles()

    def kb_layer_ig(self,ids,target,layer,prefix,steps=None,batch=None):
        steps=steps or self.c['ig_steps'];batch=batch or self.c['ig_batch'];r=self.runner
        capture={}
        def cap(m,a,o):capture['a']=o[:,-1].detach()
        h=r.layers[layer].mlp.gate_proj.register_forward_hook(cap)
        try:
            with torch.no_grad():r.model(input_ids=ids,use_cache=False,logits_to_keep=1)
        finally:h.remove()
        activation=capture['a'];token=r.tokenizer.encode(target,add_special_tokens=False)
        assert len(token)==1,target
        gradients=torch.zeros_like(activation[0],dtype=torch.float32)
        for alphas in torch.linspace(0,1,steps,device='cuda').split(batch):
            values=(activation.float()*alphas[:,None]).detach().requires_grad_(True)
            def replace(m,a,o):
                changed=o.clone();changed[:,-1,:]=values.to(o.dtype);return changed
            handle=r.layers[layer].mlp.gate_proj.register_forward_hook(replace)
            try:
                if prefix is None:kwargs={'input_ids':ids.expand(len(alphas),-1)}
                else:
                    cached=copy.deepcopy(prefix);cached.batch_repeat_interleave(len(alphas))
                    kwargs={'input_ids':ids[:,-1:].expand(len(alphas),-1),'attention_mask':torch.ones(len(alphas),ids.shape[1],dtype=torch.long,device='cuda'),'past_key_values':cached}
                output=r.model(**kwargs,use_cache=False,logits_to_keep=1)
                probability=output.logits[:,-1].float().softmax(-1)[:,token[0]]
                grad=torch.autograd.grad((probability*1024).sum(),values)[0]/1024
                if not torch.isfinite(grad).all():raise FloatingPointError('Knowledge IG gradient')
                gradients+=grad.sum(0)
            finally:handle.remove()
            del output,probability,grad,values
            if prefix is not None:del cached
        return (activation[0].float()*gradients/steps).detach().cpu().numpy()

    def knowbias(self):
        self.setup();questions=read(self.out/'knowledge_questions.json');scores=[]
        for j,q in enumerate(questions):
            path=self.out/'knowledge_attributions'/f'{j:03}.npz'
            if not path.exists():
                ids=self.ids(self.knowledge_prompt(q['question']))
                with torch.no_grad():prefix=self.runner.model(input_ids=ids[:,:-1],use_cache=True,logits_to_keep=1).past_key_values
                values=[]
                for layer in range(len(self.runner.layers)):
                    values.append(self.kb_layer_ig(ids,q['target'],layer,prefix))
                path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,scores=np.array(values))
                del prefix;self.log('knowledge_attribution',completed=j+1,total=len(questions))
            scores.append({**q,'scores':np.load(path)['scores']})
        sets={}
        for p in self.c['kb_consistency']:
            nodes=knowledge_set(scores,self.c['kb_threshold'],p)
            if not len(nodes):raise ValueError('No knowledge neurons')
            sets[str(p)]=nodes.tolist()
        write(self.out/'knowledge_sets.json',{'sets':sets,'counts':{k:len(v) for k,v in sets.items()},'threshold':self.c['kb_threshold'],'union_dimensions':['gender','race','religion']})
        self.log('knowbias_fit_complete',counts={k:len(v) for k,v in sets.items()})

    def crispr(self):
        self.setup();r=self.runner;chosen=read(self.out/'crispr_examples.json')['indices'];totals=[]
        for shot in (False,True):
            values=[]
            for j,index in enumerate(chosen):
                row=self.source[index];path=self.out/'crispr_attributions'/f'{int(shot)}_{j:03}.npz'
                if not path.exists():
                    activations=[];handles=[]
                    def capture(m,a):activations.append(a[0])
                    for layer in r.layers:handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
                    def enable(m,a,o):return o.detach().requires_grad_(True)
                    handles.append(r.model.model.embed_tokens.register_forward_hook(enable))
                    try:
                        output=r.model(input_ids=self.ids(prompt(row,shots=shot)),use_cache=False,logits_to_keep=1)
                        probabilities=output.logits[0,-1].float().softmax(-1)
                        choices=probabilities[r.labels];wrong=choices.detach().clone();wrong[row['label']]=-1;wrong=int(wrong.argmax())
                        gw=torch.autograd.grad(probabilities[r.labels[wrong]]*1024,activations,retain_graph=True)
                        gcorr=torch.autograd.grad(probabilities[r.labels[row['label']]]*1024,activations)
                        result=[]
                        for a,w,c in zip(activations,gw,gcorr):
                            v=(a.float()*w.float()/1024-(a.float()*c.float()/1024).clamp_min(0)).amax(dim=1)[0]
                            result.append(v.detach().cpu().numpy())
                        array=np.array(result)
                        if not np.isfinite(array).all():raise FloatingPointError('CRISPR attribution')
                        path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,scores=array,weight=choices[wrong].detach().cpu().numpy(),wrong=wrong)
                    finally:
                        for h in handles:h.remove()
                    del output,probabilities,choices,activations,gw,gcorr,result,v
                    if (j+1)%10==0:self.log('crispr_attribution',shot=shot,completed=j+1,total=len(chosen))
                item=np.load(path);values.append(item['scores']*item['weight'])
            totals.append(np.sum(values,axis=0))
        score=np.mean(totals,axis=0);np.save(self.out/'crispr_scores.npy',score)
        ordering=rank(score);nodes=np.array(np.unravel_index(ordering,score.shape)).T
        write(self.out/'crispr_sets.json',{str(k):nodes[:k].tolist() for k in self.c['crispr_k']})
        self.log('crispr_fit_complete',attribution_rows=2*len(chosen))

    def fill_prompt(self,question,options,style):
        if style=='selection':messages=[{'role':'user','content':f'Fill in the blank in the sentence using the following candidate words.\nSentence:{question}\nCandidate words:{options}\nOnly provide the correct answer word, nothing else.\nAnswer:'}]
        elif style=='forward':messages=[{'role':'system','content':'Based on the context, fill in the blank using the candidate options.\nOnly answer with ONE word from the options without any other words.'},{'role':'user','content':f'Context:{question}\nOptions:{options}'}]
        else:messages=[{'role':'user','content':f'According to the demographic attribute of the person, fill in the blank in the sentence using the following options.\nSentence: {question}\nOptions: {options}\nOnly provide a single word from the options, nothing else.'}]
        return self.runner.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)

    def batched_logits(self,texts):
        t=self.runner.tokenizer;t.padding_side='left';t.pad_token_id=t.eos_token_id
        inputs=t(texts,padding=True,return_tensors='pt').to('cuda')
        if inputs.input_ids.shape[1]>self.c['max_tokens']:raise ValueError('No truncation')
        inputs['position_ids']=inputs.attention_mask.cumsum(-1).sub(1).clamp_min(0)
        with torch.inference_mode():return self.runner.model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()

    def cues(self):
        self.setup();information=read(BBA/'gradient-debias/synthetic_data/sensitive_information.json');templates=read(BBA/'gradient-debias/synthetic_data/template.json')
        selected={}
        for domain in DOMAINS:
            selected[domain]={};groups=information[domain]['group'];ids=[self.runner.tokenizer.encode(g,add_special_tokens=False)[0] for g in groups]
            for kind in ('adjective','noun'):
                path=self.out/'cue_entropies'/f'{domain}_{kind}.json'
                if not path.exists():
                    words=list(dict.fromkeys(information[domain]['stereotype_'+kind]));jobs=[]
                    for word in words:
                        for template in templates[kind+'_template']:
                            question=template.replace('[sensitive_attribute]',domain).replace('[stereotype_'+kind+']',word)
                            jobs.append(self.fill_prompt(question,groups,'selection'))
                    chunks=[]
                    for start in range(0,len(jobs),self.c['cue_batch']):
                        cache=self.out/'cue_batches'/f'{domain}_{kind}_{start:05}.npy'
                        if cache.exists():p=np.load(cache)
                        else:
                            logits=self.batched_logits(jobs[start:start+self.c['cue_batch']]);p=logits.softmax(-1)[:,ids].cpu().numpy()
                            cache.parent.mkdir(parents=True,exist_ok=True);np.save(cache,p)
                        chunks.append(p)
                        if start%(self.c['cue_batch']*10)==0:self.log('cue_selection',domain=domain,kind=kind,completed=min(start+self.c['cue_batch'],len(jobs)),total=len(jobs))
                    probabilities=np.concatenate(chunks).reshape(len(words),len(templates[kind+'_template']),len(groups)).mean(1)
                    probabilities=(probabilities+1e-10)/(probabilities+1e-10).sum(1,keepdims=True)
                    entropy=-(probabilities*np.log(probabilities)).sum(1)
                    write(path,{'words':words,'entropies':entropy.tolist(),'templates':len(templates[kind+'_template']),'option_first_ids':ids,'selected':list(np.array(words)[np.argsort(entropy,kind='stable')[:20]])})
                selected[domain][kind]=read(path)['selected']
        write(self.out/'selected_cues.json',selected);self.log('cue_selection_complete')

    def generate_features(self,texts,cache_name):
        """Store final normalized state before the last (up to fourth) token.

        Equivalent attribution rank to authors' generated_hidden_states tensor:
        only the final state receives a nonzero gradient, divided by T there.
        """
        r=self.runner;t=r.tokenizer;t.padding_side='left';t.pad_token_id=t.eos_token_id;all_h=[];all_n=[]
        for start in range(0,len(texts),self.c['generation_batch']):
            path=self.out/'readout_fit_features'/f'{cache_name}_{start:05}.npz'
            if not path.exists():
                inputs=t(texts[start:start+self.c['generation_batch']],padding=True,return_tensors='pt').to('cuda')
                if inputs.input_ids.shape[1]>self.c['max_tokens']:raise ValueError('No truncation')
                captured=[]
                def capture(m,a,o):captured.append(o[:,-1].detach().cpu().numpy().copy())
                handle=r.model.model.norm.register_forward_hook(capture)
                try:
                    with torch.inference_mode():output=r.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=4,use_cache=True,pad_token_id=t.pad_token_id)
                finally:handle.remove()
                generated=output[:,inputs.input_ids.shape[1]:].cpu().numpy();eos=r.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
                hs=[];lengths=[]
                for i,tokens in enumerate(generated):
                    found=[j+1 for j,x in enumerate(tokens) if int(x) in eos];length=found[0] if found else len(tokens)
                    hs.append(captured[length-1][i]);lengths.append(length)
                path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,features=np.array(hs),lengths=np.array(lengths),tokens=generated)
            saved=np.load(path);all_h.extend(saved['features']);all_n.extend(saved['lengths'])
            if start%(self.c['generation_batch']*10)==0:self.log('readout_features',name=cache_name,completed=min(start+self.c['generation_batch'],len(texts)),total=len(texts))
        return np.array(all_h),np.array(all_n)

    def readout(self):
        self.setup();r=self.runner;info=read(BBA/'gradient-debias/synthetic_data/sensitive_information.json');forward=read(BBA/'gradient-debias/synthetic_data/template.json');backward=read(BBA/'gradient-debias/synthetic_data_anti/template.json');cues=read(self.out/'selected_cues.json')
        rankings={}
        for domain in DOMAINS:
            rankings[domain]={};groups=info[domain]['group']
            for mode in ('fba','bba'):
                path=self.out/f'{mode}_{domain}_scores.npy'
                if not path.exists():
                    texts=[];jobs=[]
                    for kind in ('adjective','noun'):
                        options=groups if mode=='fba' else cues[domain][kind]
                        if mode=='fba':
                            for word in cues[domain][kind]:
                                for template in forward[kind+'_template']:
                                    question=template.replace('[sensitive_attribute]',domain).replace('[stereotype_'+kind+']',word)
                                    texts.append(self.fill_prompt(question,options,'forward'));jobs.append((kind,options))
                        else:
                            for template in backward[kind+'_template']:
                                begin=len(texts)
                                for group in groups:
                                    question=template.replace('[sensitive_attribute]',domain).replace('[group]',group)
                                    texts.append(self.fill_prompt(question,options,'backward'))
                                jobs.append((begin,len(texts),kind,options))
                    hs,lengths=self.generate_features(texts,f'{mode}_{domain}');contributions=[]
                    weights=r.model.lm_head.weight.detach().float()
                    if mode=='fba':
                        for h,length,(_,options) in zip(hs,lengths,jobs):
                            ids=[r.tokenizer.encode(g,add_special_tokens=False) for g in options]
                            contribution=readout_ig(torch.tensor(h,device='cuda'),weights,lambda x,w:entropy_inverse(x,w,ids),self.c['ig_steps'])/length
                            contributions.append(contribution)
                    else:
                        for begin,end,kind,options in jobs:
                            ids=[r.tokenizer.encode(g,add_special_tokens=False)[0] for g in options]
                            base=torch.tensor(hs[begin:end],device='cuda')
                            # Authors' sequence averaging introduces 1/T per group.
                            basef=base.detach().float();alpha=torch.linspace(0,1,self.c['ig_steps'],device='cuda')[:,None,None]
                            scaled=(basef[None]*alpha).requires_grad_(True)
                            objective=js_objective(scaled,weights,ids)
                            gradient=torch.autograd.grad(objective.sum(),scaled)[0].mean(0)
                            contribution=(basef*gradient/torch.tensor(lengths[begin:end],device='cuda')[:,None]).mean(0).detach().cpu().numpy()
                            if not np.isfinite(contribution).all():raise FloatingPointError('BBA nonfinite')
                            contributions.append(contribution)
                    np.save(path,np.mean(contributions,axis=0));del weights;gc.collect();torch.cuda.empty_cache()
                rankings[domain][mode]=rank(np.load(path)).tolist()
                self.log('readout_fit_domain_complete',domain=domain,method=mode)
        # Unified, input-category-independent transfer to all 11 categories.
        # Rank fusion balances attribution scales across demographic dimensions.
        for mode in ('fba','bba'):
            orderings=[rankings[d][mode] for d in DOMAINS];positions=[]
            for ordering in orderings:
                inv=np.empty(len(ordering));inv[ordering]=np.arange(len(ordering));positions.append(inv)
            rankings['unified_'+mode]=np.argsort(np.mean(positions,axis=0),kind='stable').tolist()
        write(self.out/'readout_rankings.json',rankings);self.log('readout_fit_complete')

    def evaluate_nodes(self,split,method,spec,utility=False):
        path=self.out/'node_predictions'/split/('utility' if utility else 'bbq')/(spec['id']+'.npz')
        if path.exists():return np.load(path)['scores']
        self.setup();examples=self.examples(split,utility);scores=[];tick=time.perf_counter();handle=self.install_nodes(method,spec)
        try:
            for j,(row,shot,_) in enumerate(examples):
                text=full_mmlu_prompt(row) if utility else prompt(row,shots=shot)
                if utility:
                    with torch.inference_mode():output=self.runner.model(input_ids=self.ids(text),use_cache=False,logits_to_keep=1)
                    ids=[self.runner.tokenizer('Answer: '+a).input_ids[-1] for a in 'ABCD'];s=output.logits[0,-1,ids].float().cpu().numpy()
                else:s=self.runner.forward(text)[1]
                scores.append(s)
                if (j+1)%100==0:self.log('node_evaluation',split=split,method=spec['id'],utility=utility,completed=j+1,total=len(examples))
        finally:handle.remove()
        scores=np.array(scores)
        if not np.isfinite(scores).all():raise FloatingPointError('Node inference')
        path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,scores=scores,seconds=time.perf_counter()-tick)
        return scores

    def cache_readout(self,split,utility=False):
        path=self.out/'eval_readout'/f'{split}_{int(utility)}.npz'
        if path.exists():return dict(np.load(path))
        self.setup();examples=self.examples(split,utility);pre=[];post=[];h18=[];s=[]
        for j,(row,shot,index) in enumerate(examples):
            text=full_mmlu_prompt(row) if utility else prompt(row,shots=shot)
            captured,logits=self.raw_capture(text);pre.append(captured['pre']);post.append(captured['post'])
            letters='ABCD' if utility else 'ABC';ids=[self.runner.tokenizer('Answer: '+a).input_ids[-1] for a in letters];s.append(logits[ids])
            if (j+1)%100==0:self.log('eval_readout_cache',split=split,utility=utility,completed=j+1,total=len(examples))
        result={'pre':np.array(pre),'post':np.array(post),'scores':np.array(s)}
        assert np.array_equal(result['scores'],self.baseline(split,utility)),'Parent baseline mismatch'
        path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**result);return result

    def evaluate_readout(self,split,spec,utility=False,domain=None):
        self.setup();r=self.runner;cache=self.cache_readout(split,utility);rankings=read(self.out/'readout_rankings.json')
        ordering=rankings['unified_'+spec['method']] if domain is None else rankings[domain][spec['method']]
        indices=ordering[:int(len(ordering)*spec['ratio'])];edited=replace_readout(torch.tensor(cache[spec['hook']],device='cuda'),indices,spec['constant'])
        scores=[];letters='ABCD' if utility else 'ABC';ids=[r.tokenizer('Answer: '+a).input_ids[-1] for a in letters]
        with torch.inference_mode():
            for h in edited:
                h=h.reshape(1,1,-1)
                if spec['hook']=='pre':h=r.model.model.norm(h)
                scores.append(r.model.lm_head(h)[0,0,ids].float().cpu().numpy())
        return np.array(scores)

    def score(self,split,scores,utility=False):
        examples=self.examples(split,utility);pred=scores.argmax(1)
        if utility:return {'rows':len(pred),'accuracy':float(np.mean(pred==[r['answer'] for r,_,_ in examples]))}
        result=metrics([r for r,_,_ in examples],pred)
        result['native_disambig_unknown_rate']=float(np.mean([p==r['unknown_index'] for p,(r,_,_) in zip(pred,examples) if r['context_condition']=='disambig']))
        return result

    def feasible(self,bbq,utility):
        reference=read(self.parent/'selection_results.json');base=reference['baseline'];ref=reference['reference'];fail=[]
        if bbq['disambig']['accuracy']<max(base['disambig']['accuracy']-.01,ref['disambig']['accuracy']-.005)-1e-12:fail.append('disambig')
        if utility['accuracy']<reference['baseline_utility']['accuracy']-.02-1e-12:fail.append('utility')
        if bbq['ambig']['bias_score'] is None or abs(bbq['ambig']['bias_score'])>abs(ref['ambig']['bias_score'])+.01+1e-12:fail.append('bias')
        return fail

    def candidates(self):
        result=[];sets=read(self.out/'knowledge_sets.json')['sets']
        for p in self.c['kb_consistency']:
            for strength in self.c['kb_strengths']:result.append({'id':f'knowbias_p{p}_s{strength}','group':'knowbias','method':'knowbias','p':p,'strength':strength,'nodes':sets[str(p)]})
        for k,nodes in read(self.out/'crispr_sets.json').items():result.append({'id':f'crispr_k{k}','group':'crispr','method':'crispr','k':int(k),'nodes':nodes,'strength':0.})
        for method in ('fba','bba'):
            for hook in ('post','pre'):
                for ratio in self.c['readout_ratios']:
                    for constant in self.c['readout_constants']:result.append({'id':f'{method}_{hook}_r{ratio}_c{constant}','group':f'{method}_{hook}','method':method,'hook':hook,'ratio':ratio,'constant':constant})
        return result

    def unknown_scores(self,split,spec,utility=False):
        scores=self.baseline(split,utility).copy()
        if utility:x=np.load(self.parent/'utility_baseline'/f'{split}.npz')['features']
        else:x=np.asarray(self.features[[i for _,_,i in self.examples(split)],18],np.float32)
        probe=np.load(self.parent/'probes.npz');p=expit(x@probe['evidence_coef']+probe['evidence_intercept'])
        if utility:return scores # MMLU does not contain an identifiable unknown option.
        for i,(row,_,_) in enumerate(self.examples(split)):
            if spec['method']=='unknown_force':
                if p[i]>spec['threshold']:scores[i,row['unknown_index']]=scores[i].max()+1.
            else:scores[i,row['unknown_index']]+=spec['offset']*p[i]
        return scores

    def select(self):
        path=self.out/'selection_frozen.json'
        if path.exists():return
        records=[];selected={};candidates=self.candidates()
        candidates += [{'id':f'unknown_force_t{t}','group':'unknown_force','method':'unknown_force','threshold':t} for t in self.c['unknown_thresholds']]
        candidates += [{'id':f'unknown_offset_{v}','group':'unknown_offset','method':'unknown_offset','offset':v} for v in self.c['unknown_offsets']]
        for spec in candidates:
            b=self.evaluate('dev',spec);u=self.evaluate('dev',spec,True)
            bm=self.score('dev',b);um=self.score('dev',u,True);fail=self.feasible(bm,um)
            records.append({'spec':spec,'bbq':bm,'utility':um,'failures':fail})
            self.log('candidate',name=spec['id'],accuracy=bm['accuracy'],disambig=bm['disambig']['accuracy'],utility=um['accuracy'],failures=fail)
        for group in sorted({s['group'] for s in candidates}):
            items=[r for r in records if r['spec']['group']==group];valid=[r for r in items if not r['failures']]
            best=max(valid or items,key=lambda r:(r['bbq']['accuracy'],r['bbq']['disambig']['accuracy'],-r['spec'].get('ratio',0),-r['spec'].get('k',0),-r['spec'].get('strength',0)))
            selected[group]=best
        write(self.out/'selection_results.json',records)
        fits=['knowledge_sets.json','crispr_sets.json','readout_rankings.json','selected_cues.json']+[f'{m}_{d}_scores.npy' for m in ('fba','bba') for d in DOMAINS]
        write(path,{'selected':selected,'selection_sha256':sha(self.out/'selection_results.json'),'fitted_sha256':{p:sha(self.out/p) for p in fits},'plan_hash':self.plan_hash,'rule':'Same dev constraints as improvement pilot; no feasible candidate => flag exploratory fallback. Never reselect on test.'})
        self.log('selection_complete',selected={g:r['spec']['id'] for g,r in selected.items()})

    def evaluate(self,split,spec,utility=False):
        if spec['method'] in ('knowbias','crispr'):return self.evaluate_nodes(split,spec['method'],spec,utility)
        if spec['method'].startswith('unknown'):return self.unknown_scores(split,spec,utility)
        return self.evaluate_readout(split,spec,utility)

    def test(self):
        plan=read(self.out/'selection_frozen.json')
        assert plan['selection_sha256']==sha(self.out/'selection_results.json')
        for p,h in plan['fitted_sha256'].items():assert sha(self.out/p)==h
        lock={'selection_sha256':sha(self.out/'selection_frozen.json'),'plan_hash':self.plan_hash}
        path=self.out/'test_lock.json'
        if path.exists():assert read(path)==lock
        else:write(path,lock)
        records=[]
        for group,chosen in plan['selected'].items():
            path=self.out/'predictions'/f'{group}.json'
            if path.exists():record=read(path)
            else:
                spec=chosen['spec'];b=self.evaluate('test',spec);u=self.evaluate('test',spec,True)
                record={'name':group,'spec':spec,'dev_failures':chosen['failures'],'bbq':{'scores':b.tolist(),'predictions':b.argmax(1).tolist(),'metrics':self.score('test',b)},'utility':{'scores':u.tolist(),'predictions':u.argmax(1).tolist(),'metrics':self.score('test',u,True)}}
                write(path,record)
            records.append(record);self.log('test_method',method=group,accuracy=record['bbq']['metrics']['accuracy'],disambig=record['bbq']['metrics']['disambig']['accuracy'])
        write(self.out/'results.json',records)

    def run(self,stage):
        stages=('prepare','smoke','knowbias','crispr','cues','readout','select','test','report') if stage=='all' else (stage,)
        for name in stages:
            if name=='report':
                from .bbq_comparators_report import report
                report(self)
            else:getattr(self,name)()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',nargs='?',default='all',choices=['all','prepare','smoke','knowbias','crispr','cues','readout','select','test','report']);parser.add_argument('--config',type=Path,default=CONFIG);a=parser.parse_args();Comparators(a.config).run(a.stage)


if __name__=='__main__':main()
