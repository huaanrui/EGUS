"""Matched-budget exploratory FairSteer improvements with family-disjoint refits.

Frozen-base-model activations are reusable; all fitted artifacts are rebuilt on
fit families. Controller outcomes use a separate training-family partition.
BBQ has been evaluated historically, so this is exploratory, not pristine data.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import tomllib
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .fairsteer_author import ROOT, Runner, label_for, metrics, prompt, read, rows, sha, write
from .fairsteer_data import source_fewshot

CONFIG = ROOT/'configs/experiment_fairsteer_improve.toml'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def family(row,mapping):
    name=f"{row['category']}:{row['question_index']}"
    return mapping.get(name,name)


def ranked(seed,value):
    return hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()


def balanced_indices(source,indices,per_condition,seed):
    """Round-robin family/polarity strata, separately within category/condition."""
    outer=collections.defaultdict(lambda:collections.defaultdict(list))
    for i in indices:
        r=source[i]
        outer[(r['category'],r['context_condition'])][(r['family'],r['question_polarity'])].append(i)
    selected=[]
    for key,bins in sorted(outer.items()):
        pools=[sorted(v,key=lambda i:ranked(seed,source[i]['uid'])) for k,v in sorted(bins.items(),key=lambda kv:ranked(seed,str(kv[0])))]
        pool=[]
        for depth in range(max(map(len,pools))):
            for values in pools:
                if depth<len(values):pool.append(values[depth])
            if len(pool)>=per_condition:break
        if len(pool)<per_condition:raise ValueError(f'Insufficient examples for {key}: {len(pool)}')
        selected.extend(pool[:per_condition])
    return sorted(selected,key=lambda i:source[i]['uid'])


def make_splits(source,mapping,c):
    for row in source:row['family']=family(row,mapping)
    demo={s.split('\n')[0] for s in source_fewshot().strip().split('\n\n')}
    protected={r['family'] for r in source if r['context']+' '+r['question'] in demo}
    assignment={}
    for category in sorted({r['category'] for r in source}):
        groups=sorted({r['family'] for r in source if r['category']==category and r['family'] not in protected},key=lambda g:ranked(c['seed'],g))
        n=len(groups); nf=int(n*c['fit_fraction']); nc=max(1,int(n*c['controller_fraction']));nd=max(1,int(n*c['dev_fraction']))
        assert n-nf-nc-nd>=2
        for j,g in enumerate(groups):assignment[g]='fit' if j<nf else 'controller' if j<nf+nc else 'dev' if j<nf+nc+nd else 'test'
    for g in protected:assignment[g]='fit'
    partitions={name:[i for i,r in enumerate(source) if assignment[r['family']]==name] for name in ('fit','controller','dev','test')}
    for name in ('controller','dev','test'):
        partitions[name]=balanced_indices(source,partitions[name],c[f'{name}_per_condition_category'],c['seed']+1)
    fam={k:{source[i]['family'] for i in v} for k,v in partitions.items()}
    for a in fam:
        for b in fam:
            if a!=b:assert not fam[a]&fam[b]
    texts={k:{source[i]['context']+' '+source[i]['question'] for i in v} for k,v in partitions.items()}
    for a in texts:
        for b in texts:
            if a!=b:assert not texts[a]&texts[b],f'Duplicate prompt between {a} and {b}'
    return partitions,assignment,sorted(protected)


def full_mmlu_prompt(row):
    return row['question']+''.join(f'\n{letter}. {choice}' for letter,choice in zip('ABCD',row['choices']))+'\nAnswer:'


def outcomes(base_scores,edited_scores,labels):
    """Labels are used only when training the controller, never for deployment."""
    labels=np.asarray(labels);before=base_scores.argmax(1)==labels;after=edited_scores.argmax(1)==labels
    base_lp=base_scores-logsumexp(base_scores,axis=1,keepdims=True)
    edit_lp=edited_scores-logsumexp(edited_scores,axis=1,keepdims=True)
    margin=np.clip(edit_lp[np.arange(len(labels)),labels]-base_lp[np.arange(len(labels)),labels],-2.,2.)
    return (after.astype(float)-before.astype(float))-((before)&(~after)).astype(float)+0.05*margin


def choose_actions(predicted,strengths,margin=0.):
    """Action zero has utility zero. Ties favour no intervention/smaller alpha."""
    utilities=np.column_stack([np.zeros(len(predicted)),predicted-margin])
    return np.asarray([0.,*strengths])[utilities.argmax(1)]


class PilotRunner(Runner):
    def forward_labels(self,text,letters='ABC',capture=False):
        ids=self.tokenizer(text,return_tensors='pt').input_ids.to('cuda')
        if ids.shape[1]>self.c['max_tokens']:raise ValueError('No truncation permitted')
        with torch.inference_mode():
            output=self.model(input_ids=ids,use_cache=False,output_hidden_states=capture,logits_to_keep=1)
        token_ids=[self.tokenizer('Answer: '+a).input_ids[-1] for a in letters]
        scores=output.logits[0,-1,token_ids].float().cpu().numpy()
        feature=output.hidden_states[self.c['layer']+1][0,-1].float().cpu().numpy() if capture else None
        if not np.isfinite(scores).all() or (capture and not np.isfinite(feature).all()):raise FloatingPointError('Nonfinite model outputs')
        return scores,feature


class Pilot:
    def __init__(self,config=CONFIG):
        self.config=Path(config);self.c=tomllib.loads(self.config.read_text(encoding='utf-8'))
        self.parent=ROOT/self.c['parent_output'];self.parent_data=ROOT/self.c['parent_data']
        self.out=ROOT/self.c['output_dir'];self.data=ROOT/self.c['data_dir']
        self.out.mkdir(parents=True,exist_ok=True);self.data.mkdir(parents=True,exist_ok=True)
        self.source=rows(self.parent_data/'bbq.jsonl');self.mmlu=rows(self.parent_data/'mmlu.jsonl');self.n=len(self.source)
        mapping=read(ROOT/'configs/bbq_template_families.json')['mapping']
        self.splits,self.assignment,self.protected=make_splits(self.source,mapping,self.c)
        unique={}
        for i,r in enumerate(self.mmlu):unique.setdefault(' '.join(r['question'].casefold().split()),i)
        order=sorted(unique.values(),key=lambda i:ranked(self.c['seed'],self.mmlu[i]['question']))
        self.msplits={};offset=0
        for name in ('controller','dev','test'):
            count=self.c['mmlu_'+name];self.msplits[name]=order[offset:offset+count];offset+=count
        self.msplits['fit']=order[offset:]
        self.prepare_manifest()
        self.raw_x=np.load(self.parent/'features.npy',mmap_mode='r');self.raw_pred=np.load(self.parent/'predictions.npy',mmap_mode='r');self.raw_scores=np.load(self.parent/'scores.npy',mmap_mode='r')
        self.runner=None;self.probes=None;self.controller=None;self.vectors={}

    def prepare_manifest(self):
        source_names=['mechanism/fairsteer_improve.py','mechanism/fairsteer_author.py','mechanism/fairsteer_data.py','scripts/run_fairsteer_improve.py']
        fingerprint={'config':sha(self.config),'sources':{name:sha(ROOT/name) for name in source_names},'parent_data':sha(self.parent_data/'manifest.json'),'parent_features_manifest':sha(self.parent/'feature_hashes.json'),'family_map':sha(ROOT/'configs/bbq_template_families.json')}
        path=self.out/'manifest.json'
        if path.exists():
            self.state=read(path);assert self.state['fingerprint']==fingerprint,'Frozen experiment changed; use a new version'
        else:
            self.state={'fingerprint':fingerprint,'created':time.strftime('%Y-%m-%d %H:%M:%S'),'stages':{}}
            write(path,self.state)
            for name in source_names:
                target=self.out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
            shutil.copyfile(self.config,self.out/'source_snapshot/config.toml')
        split_record={'bbq_indices':self.splits,'mmlu_indices':self.msplits,'family_assignment':self.assignment,'fewshot_families_in_fit':self.protected}
        data_path=self.data/'splits.json'
        if data_path.exists():assert read(data_path)==split_record
        else:write(data_path,split_record)
        self.log('opened',stages=list(self.state['stages']))

    def log(self,event,**details):
        record={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'event':event,**details}
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

    def done(self,name,**details):
        self.state['stages'][name]={'completed':True,**details};write(self.out/'manifest.json',self.state);self.log(name+'_complete',**details)

    def setup(self):
        if self.runner is None:
            self.log('loading_model');self.runner=PilotRunner(self.c)
            parent=read(self.parent/'environment.json')
            current={p.name:sha(p) for p in Path(self.c['model_path']).glob('*') if p.suffix in ('.json','.safetensors')}
            assert current==parent['model']
            write(self.out/'environment.json',{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'sklearn':__import__('sklearn').__version__,'gpu':torch.cuda.get_device_name(),'model_hashes':current,'dtype':'float16','batch':1})

    def release(self):
        if self.runner is not None:self.runner.set_edit();self.runner=None;gc.collect();torch.cuda.empty_cache()

    def examples(self,split,utility=False):
        if utility:return [(self.mmlu[i],False,2*self.n+i) for i in self.msplits[split]]
        return [(self.source[i],shots,i+self.n*int(shots)) for shots in (False,True) for i in self.splits[split]]

    def base(self,split,utility=False):
        if utility:
            path=self.out/'utility_baseline'/f'{split}.npz'
            if not path.exists():
                self.setup();self.runner.set_edit();scores=[];features=[]
                for r,_,_ in self.examples(split,True):
                    s,h=self.runner.forward_labels(full_mmlu_prompt(r),'ABCD',True);scores.append(s);features.append(h)
                path.parent.mkdir(exist_ok=True);np.savez(path,scores=np.array(scores),features=np.array(features))
            saved=np.load(path);return saved['scores'],saved['features']
        ids=[i for _,_,i in self.examples(split)];return np.array(self.raw_scores[ids]),np.asarray(self.raw_x[ids,self.c['layer'],:],np.float32)

    def smoke(self):
        expected=read(self.parent/'feature_hashes.json')
        for name,d in expected.items():assert sha(self.parent/name)==d
        self.setup();run=self.runner;results=[]
        for split in ('fit','dev'):
            for r,shots,i in self.examples(split)[:3]+self.examples(split)[-3:]:
                p,s,h=run.forward(prompt(r,shots=shots),True)
                assert np.array_equal(s,self.raw_scores[i]) and np.array_equal(h[self.c['layer']],self.raw_x[i,self.c['layer']])
                results.append({'uid':r['uid'],'shots':shots,'exact':True})
        row=self.source[self.splits['fit'][0]];text=prompt(row);s,_=run.forward_labels(text)
        run.set_edit(self.c['layer'],np.ones(2048),np.zeros(2048),1.);disabled,_=run.forward_labels(text);assert np.array_equal(s,disabled)
        run.set_edit(self.c['layer'],np.ones(2048),np.zeros(2048),-1.);edited,_=run.forward_labels(text);assert not np.array_equal(s,edited)
        run.set_edit();restored,_=run.forward_labels(text);assert np.array_equal(s,restored)
        write(self.out/'smoke.json',{'cache_checks':results,'gate_off_exact':True,'hook_restore_exact':True,'edit_changes_logits':True,'feature_hashes_verified':True})
        write(self.out/'data_summary.json',{'bbq':{k:{'rows':len(v),'families':len({self.source[i]['family'] for i in v}),'categories':dict(collections.Counter(self.source[i]['category'] for i in v))} for k,v in self.splits.items()},'mmlu':{k:len(v) for k,v in self.msplits.items()},'scope':'Exploratory family-disjoint refits on historically evaluated BBQ, not pristine test data. Two shot settings remain within family partition.'})
        self.done('smoke');self.release()

    def fit(self):
        self.release();ids=[i for _,_,i in self.examples('fit')]+[2*self.n+i for i in self.msplits['fit']]
        x=np.asarray(self.raw_x[ids,self.c['layer'],:],np.float32)
        pred=self.raw_pred[ids];original=[];expanded=[];evidence=[]
        for j,i in enumerate(ids):
            if i<2*self.n:
                row=self.source[i%self.n];original.append(label_for(row,int(pred[j])));expanded.append(int(pred[j]==row['label']));evidence.append(int(row['context_condition']=='ambig'))
            else:original.append(1);expanded.append(1);evidence.append(0)
        labels={'original':np.array(original),'expanded':np.array(expanded),'evidence':np.array(evidence)}
        weights={};diagnostics={}
        with threadpool_limits(limits=self.c['cpu_threads']):
            for name,y in labels.items():
                use=y>=0;self.log('fit_probe',name=name,rows=int(use.sum()),classes=dict(collections.Counter(map(str,y[use]))))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    clf=LogisticRegression(C=self.c['probe_C'],solver='lbfgs',max_iter=self.c['probe_max_iter'],tol=self.c['probe_tol']).fit(x[use],y[use])
                if any(isinstance(w.message,ConvergenceWarning) for w in caught):raise RuntimeError(f'Unconverged {name} probe')
                coef=clf.coef_[0].astype(np.float32);intercept=float(clf.intercept_[0])
                flips=int(np.sum((x[use]@coef+intercept>0)!=clf.predict(x[use])))
                if flips:raise ValueError('Probe export flips')
                weights[name+'_coef']=coef;weights[name+'_intercept']=np.array(intercept)
                diagnostics[name]={'rows':int(use.sum()),'classes':dict(collections.Counter(map(str,y[use]))),'iterations':int(clf.n_iter_[0]),'train_accuracy':float(clf.score(x[use],y[use])),'export_flips':flips}
        np.savez(self.out/'probes.npz',**weights);write(self.out/'probe_fit.json',diagnostics)
        self.done('fit',training_input_rows=len(ids));del x;gc.collect()

    def directions(self):
        self.setup();rng=np.random.RandomState(self.c['seed']);chosen=[]
        for cat in sorted({r['category'] for r in self.source}):
            pool=[i for i in self.splits['fit'] if self.source[i]['category']==cat and self.source[i]['context_condition']=='ambig' and self.source[i]['target_loc'] is not None]
            chosen.extend(map(int,rng.choice(pool,self.c['direction_per_category'],replace=False)))
        values=[];labels=[]
        for j,i in enumerate(chosen):
            row=self.source[i]
            for shots in (False,True):
                for answer,label in ((row['label'],1),(row['target_loc'],0)):
                    _,_,h=self.runner.forward(prompt(row,shots=shots,answer=answer),True);values.append(h[self.c['layer']].astype(np.float64));labels.append(label)
            if (j+1)%10==0:self.log('directions',completed=j+1,total=len(chosen))
        values=np.array(values);labels=np.array(labels);vector=values[labels==1].mean(0)-values[labels==0].mean(0)
        np.save(self.out/'direction_dim.npy',vector);write(self.out/'direction_examples.json',{'indices':chosen,'uids':[self.source[i]['uid'] for i in chosen],'norm':float(np.linalg.norm(vector))})
        self.done('directions',rows=len(chosen))

    def vector(self,name):
        if name not in self.vectors:self.vectors[name]=np.load(self.out/f'direction_{name}.npy')
        return self.vectors[name]

    def optimize(self):
        self.setup();run=self.runner;run.set_edit();initial=self.vector('dim');v=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float32,device='cuda'));anchor=v.detach().clone()
        selected=balanced_indices(self.source,self.splits['fit'],self.c['optimization_per_condition_category'],self.c['seed']+2)
        neutral=self.msplits['fit'][:self.c['optimization_neutral']]
        cases=[(self.source[i],False) for i in selected]+[(self.mmlu[i],True) for i in neutral]
        opt=torch.optim.Adam([v],lr=self.c['optimization_lr']);rng=np.random.RandomState(self.c['seed']);history=[]
        def edit(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            changed=h.clone();changed[:,-1,:]=(h[:,-1,:].double()+v.double()).to(h.dtype)
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        for epoch in range(self.c['optimization_epochs']):
            losses=[];rng.shuffle(cases)
            for j,(row,is_neutral) in enumerate(cases):
                shots=bool((int(ranked(self.c['seed'],row['uid'])[:4],16)+epoch)%2) if not is_neutral else False
                text=full_mmlu_prompt(row) if is_neutral else prompt(row,shots=shots)
                letters='ABCD' if is_neutral else 'ABC';token_ids=[run.tokenizer('Answer: '+a).input_ids[-1] for a in letters]
                ids=run.tokenizer(text,return_tensors='pt').input_ids.to('cuda')
                if ids.shape[1]>self.c['max_tokens']:raise ValueError('Optimization prompt too long')
                if is_neutral:
                    with torch.no_grad():base=run.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1,token_ids].float().detach().softmax(-1)
                opt.zero_grad(set_to_none=True);handle=run.layers[self.c['layer']].register_forward_hook(edit)
                try:
                    output=run.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1,token_ids].float()
                    if is_neutral:loss=self.c['optimization_kl']*F.kl_div(output.log_softmax(-1),base,reduction='sum')
                    else:loss=F.cross_entropy(output[None,:],torch.tensor([row['label']],device='cuda'))
                    loss=loss+self.c['optimization_anchor']*torch.mean((v-anchor)**2)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite optimization loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_([v],1.,error_if_nonfinite=True);opt.step()
                    with torch.no_grad():
                        limit=self.c['optimization_norm_cap']*anchor.norm()
                        if v.norm()>limit:v.mul_(limit/v.norm())
                    losses.append(float(loss.detach()))
                finally:handle.remove()
                if (j+1)%32==0:self.log('optimize',epoch=epoch+1,completed=j+1,total=len(cases),loss=float(np.mean(losses)))
            record={'epoch':epoch+1,'loss':float(np.mean(losses)),'norm':float(v.detach().norm()),'cosine_initial':float(F.cosine_similarity(v.detach(),anchor,dim=0))};history.append(record);self.log('optimized_epoch',**record)
        np.save(self.out/'direction_optimized.npy',v.detach().double().cpu().numpy())
        write(self.out/'optimization.json',{'history':history,'bbq_indices':selected,'neutral_indices':neutral,'epochs_fixed_in_advance':True,'neutral_kl':'Restricted to four answer tokens, not full-vocabulary KL','application':'Unconditional during fitting; learned vector combined with gates during evaluation'})
        self.done('optimize');del v,opt;gc.collect();torch.cuda.empty_cache()

    def counterfactual(self,split,vector_name,alpha,utility=False):
        """Unconditional outcomes reused across all gates; no evaluation labels used."""
        examples=self.examples(split,utility);key=f'{split}_{vector_name}_{alpha:g}'+('_utility' if utility else '')
        path=self.out/'counterfactuals'/f'{key}.npz';path.parent.mkdir(exist_ok=True)
        identity={'split':split,'indices':self.msplits[split] if utility else self.splits[split],'utility':utility,'vector_sha':sha(self.out/f'direction_{vector_name}.npy'),'alpha':alpha,'layer':self.c['layer']}
        if path.exists():
            assert read(path.with_suffix('.json'))['identity']==identity
            return np.load(path)['scores']
        checkpoint=path.with_suffix('.progress.npz');start=0;scores=[]
        if checkpoint.exists():
            previous=np.load(checkpoint);assert str(previous['identity'])==digest(identity);scores=list(previous['scores']);start=len(scores)
        self.setup();run=self.runner;run.set_edit(self.c['layer'],self.vector(vector_name)*alpha,np.zeros(run.model.config.hidden_size,dtype=np.float32),-1.)
        try:
            for j in range(start,len(examples)):
                row,shots,_=examples[j]
                text=full_mmlu_prompt(row) if utility else prompt(row,shots=shots)
                s,_=run.forward_labels(text,'ABCD' if utility else 'ABC');scores.append(s)
                if (j+1)%128==0 or j+1==len(examples):
                    temp=checkpoint.with_name(checkpoint.stem+'.tmp.npz');np.savez(temp,scores=np.array(scores),identity=digest(identity));temp.replace(checkpoint)
                    self.log('counterfactual',name=key,completed=j+1,total=len(examples))
        finally:run.set_edit()
        scores=np.array(scores);np.savez(path,scores=scores);write(path.with_suffix('.json'),{'identity':identity,'sha256':sha(path)});return scores

    def head_probabilities(self,x):
        if self.probes is None:self.probes=np.load(self.out/'probes.npz')
        return {name:expit(x@self.probes[name+'_coef']+float(self.probes[name+'_intercept'])) for name in ('original','expanded','evidence')}

    def controller_features(self,x,projection=None,mean=None,scale=None):
        if projection is None:
            saved=np.load(self.out/'controller.npz');projection=saved['projection'];mean=saved['mean'];scale=saved['scale']
        probs=self.head_probabilities(x);features=np.column_stack([x@projection,probs['original'],probs['expanded'],probs['evidence']])
        if mean is None:return features
        return (features-mean)/scale

    def train_controller(self):
        scores,x=self.base('controller');labels=np.array([r['label'] for r,_,_ in self.examples('controller')]);targets=[]
        for alpha in self.c['strengths']:targets.append(outcomes(scores,self.counterfactual('controller','dim',alpha),labels))
        neutral_scores,neutral_x=self.base('controller',True);neutral_labels=np.array([r['answer'] for r,_,_ in self.examples('controller',True)]);neutral_targets=[]
        for alpha in self.c['strengths']:neutral_targets.append(outcomes(neutral_scores,self.counterfactual('controller','dim',alpha,True),neutral_labels))
        x=np.vstack([x,neutral_x]);y=np.vstack([np.stack(targets,axis=1),np.stack(neutral_targets,axis=1)])
        rng=np.random.RandomState(self.c['seed']);projection=rng.normal(size=(x.shape[1],self.c['controller_projection'])).astype(np.float32)/np.sqrt(self.c['controller_projection'])
        raw=self.controller_features(x,projection=projection);scaler=StandardScaler().fit(raw)
        with threadpool_limits(limits=self.c['cpu_threads']):clf=Ridge(alpha=self.c['controller_ridge']).fit(scaler.transform(raw),y)
        np.savez(self.out/'controller.npz',projection=projection,mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_)
        write(self.out/'controller_training.json',{'bbq_outcomes':len(scores),'mmlu_outcomes':len(neutral_scores),'targets_mean':y.mean(0).tolist(),'held_apart_from_probe_direction_fit':True,'inference_inputs':'Last prompt token hidden state and three heads computed from it; no final logits, labels, category, unknown position or context_condition','target':'correctness gain - extra harm penalty + 0.05 clipped correct-label log-probability gain'})
        self.done('controller')

    def actions(self,spec,x):
        probs=self.head_probabilities(x);gate=spec['gate'];threshold=spec.get('threshold',.5)
        if gate=='none':return np.zeros(len(x))
        if gate=='original':return (1-probs['original']>threshold)*spec['alpha']
        if gate=='expanded':return (1-probs['expanded']>threshold)*spec['alpha']
        if gate=='evidence':return ((1-probs['expanded'])*probs['evidence']>threshold)*spec['alpha']
        if gate=='adaptive':
            saved=np.load(self.out/'controller.npz');features=self.controller_features(x)
            predicted=features@saved['coef'].T+saved['intercept']
            return choose_actions(predicted,self.c['strengths'],spec['margin'])
        raise ValueError(gate)

    def evaluate(self,split,spec,utility=False):
        baseline,x=self.base(split,utility);actions=self.actions(spec,x);result=baseline.copy()
        for alpha in self.c['strengths']:
            use=actions==alpha
            if use.any():result[use]=self.counterfactual(split,spec['vector'],alpha,utility)[use]
        pred=result.argmax(1);examples=self.examples(split,utility)
        if utility:
            labels=np.array([r['answer'] for r,_,_ in examples]);m={'rows':len(pred),'accuracy':float(np.mean(pred==labels))}
        else:
            m=metrics([r for r,_,_ in examples],pred)
            m['native_disambig_unknown_rate']=float(np.mean([p==r['unknown_index'] for (r,_,_),p in zip(examples,pred) if r['context_condition']=='disambig']))
        return {'spec':spec,'metrics':m,'predictions':pred.tolist(),'scores':result.tolist(),'actions':actions.tolist(),'trigger_fraction':float(np.mean(actions>0)),'mean_alpha':float(np.mean(actions))}

    def candidates(self):
        reference={'name':'fairsteer','gate':'original','vector':'dim','alpha':1.,'threshold':.5}
        groups={'tuned':[],'expanded':[],'evidence':[],'adaptive':[],'optimized':[],'combined':[]}
        for alpha in self.c['strengths']:
            for threshold in self.c['thresholds']:
                for name,gate,vector in [('tuned','original','dim'),('expanded','expanded','dim'),('evidence','evidence','dim'),('optimized','original','optimized'),('combined','evidence','optimized')]:
                    groups[name].append({'name':name,'gate':gate,'vector':vector,'alpha':alpha,'threshold':threshold})
        for margin in self.c['controller_margins']:groups['adaptive'].append({'name':'adaptive','gate':'adaptive','vector':'dim','margin':margin})
        return reference,groups

    def feasible(self,bbq,utility,base,ref,neutral):
        failures=[];tol=self.c['disambig_tolerance']
        if bbq['disambig']['accuracy']<max(base['disambig']['accuracy']-tol,ref['disambig']['accuracy']-.005)-1e-12:failures.append('disambig')
        if utility['accuracy']<neutral['accuracy']-self.c['utility_tolerance']-1e-12:failures.append('utility')
        a=bbq['ambig']['bias_score'];b=ref['ambig']['bias_score']
        if a is None or b is None or abs(a)>abs(b)+self.c['bias_tolerance']+1e-12:failures.append('bias')
        return failures

    def select(self):
        plan_path=self.out/'test_plan_frozen.json'
        if plan_path.exists():
            plan=read(plan_path);assert plan['selection_sha256']==sha(self.out/'selection_results.json');self.done('select');return
        reference,groups=self.candidates();baseline={'name':'baseline','gate':'none','vector':'dim'}
        base=self.evaluate('dev',baseline);neutral=self.evaluate('dev',baseline,True)
        ref=self.evaluate('dev',reference);refutility=self.evaluate('dev',reference,True)
        chosen=[baseline,reference];selected={};records=[]
        for name,specs in groups.items():
            candidates=[]
            for spec in specs:
                b=self.evaluate('dev',spec);u=self.evaluate('dev',spec,True);fail=self.feasible(b['metrics'],u['metrics'],base['metrics'],ref['metrics'],neutral['metrics'])
                item={'spec':spec,'bbq':b['metrics'],'utility':u['metrics'],'failures':fail,'trigger_fraction':b['trigger_fraction']};candidates.append(item);records.append(item)
            valid=[r for r in candidates if not r['failures']]
            best=max(valid or candidates,key=lambda r:(r['bbq']['accuracy'],r['bbq']['disambig']['accuracy'],-r['trigger_fraction']))
            chosen.append(best['spec']);selected[name]={'feasible':not best['failures'],'failures':best['failures'],'dev_accuracy':best['bbq']['accuracy']};self.log('selected',name=name,**selected[name],spec=best['spec'])
        write(self.out/'selection_results.json',{'baseline':base['metrics'],'baseline_utility':neutral['metrics'],'reference':ref['metrics'],'reference_utility':refutility['metrics'],'candidates':records})
        write(plan_path,{'configs':chosen,'selected':selected,'selection_sha256':sha(self.out/'selection_results.json'),'fitted_sha256':{name:sha(self.out/name) for name in ('probes.npz','direction_dim.npy','direction_optimized.npy','controller.npz')},'selection_rule':'Maximize dev overall accuracy subject to disambiguated, utility and abs-bias constraints; if none feasible, show best exploratory candidate explicitly flagged. No test reselection.','evaluation':'Native answer order, both zero/few-shot equally weighted; family-cluster bootstrap; small four-choice author-MMLU subset','historical_exposure':'BBQ was previously evaluated; these are new family-disjoint refits, not pristine held-out confirmation.'})
        self.done('select')

    def test(self):
        plan=read(self.out/'test_plan_frozen.json')
        for name,h in plan['fitted_sha256'].items():assert sha(self.out/name)==h
        lock={'plan_sha256':sha(self.out/'test_plan_frozen.json'),'split_sha256':sha(self.data/'splits.json')}
        path=self.out/'test_lock.json'
        if path.exists():assert read(path)==lock
        else:write(path,lock)
        results=[]
        for spec in plan['configs']:
            b=self.evaluate('test',spec);u=self.evaluate('test',spec,True);record={'name':spec['name'],'bbq':b,'utility':u}
            write(self.out/'predictions'/f"{spec['name']}.json",record);results.append(record)
            self.log('test_config',name=spec['name'],accuracy=b['metrics']['accuracy'],disambig=b['metrics']['disambig']['accuracy'],utility=u['metrics']['accuracy'])
        write(self.out/'results.json',results);self.done('test')

    def run(self,stage):
        stages=('smoke','fit','directions','optimize','controller','select','test') if stage=='all' else (stage,)
        for name in stages:
            if not self.state['stages'].get(name,{}).get('completed'):getattr(self,'train_controller' if name=='controller' else name)()
        if stage in ('all','test','report'):
            from .fairsteer_improve_report import report
            report(self)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',nargs='?',default='all',choices=['smoke','fit','directions','optimize','controller','select','test','report','all']);a=p.parse_args()
    if a.stage=='report':
        pilot=Pilot();from .fairsteer_improve_report import report
        report(pilot)
    else:Pilot().run(a.stage)


if __name__=='__main__':main()
