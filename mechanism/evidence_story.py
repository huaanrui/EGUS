"""Frozen matched-budget evidence recognition/action experiments, October 17."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import shutil
import time
import tomllib
import warnings

import joblib
import numpy as np
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fairsteer_author import ROOT, read, rows, write, sha, label_for, metrics
from .fairsteer_improve import balanced_indices, digest, ranked, family

CONFIG = ROOT / 'configs/experiment_evidence_story_v1.toml'
GATES = ('original', 'expanded', 'evidence', 'product')


def render(r):
    if r.get('utility'):
        return r['question'] + ''.join(f'\n{a}. {v}' for a, v in zip('ABCD', r['choices'])) + '\nAnswer:'
    return r['context'] + ' ' + r['question'] + ''.join(f'\n{a}. {r["ans"+str(i)]}' for i, a in enumerate('ABC')) + '\nAnswer:'


def surface(r):
    c, q = r['context'], r['question']
    return [len(c), len(q), len(c.split()), len(q.split()), c.count('.'), c.count(','),
            c.count(';'), c.count('!'), sum(t.isdigit() for t in c), sum(t.isupper() for t in c)]


def discover_unknown(r, vocab):
    if r.get('utility'): return -1
    found = [i for i in range(3) if r['ans'+str(i)].strip().casefold() in vocab]
    if len(found) != 1: raise ValueError(f'Cannot uniquely discover unknown from input: {r["uid"]}')
    return found[0]


def rotate(r, shift, replacement=None):
    order = list(range(3))[shift:] + list(range(3))[:shift]
    out = dict(r)
    for i, j in enumerate(order): out['ans'+str(i)] = r['ans'+str(j)]
    for key in ('label', 'unknown_index', 'target_loc'):
        out[key] = None if r.get(key) is None else order.index(r[key])
    if replacement is not None: out['ans'+str(out['unknown_index'])] = replacement
    out['permutation'] = shift
    out['wording'] = 'original' if replacement is None else 'alternate'
    out['source_uid'] = r['uid']
    out['uid'] = r['uid'] + f':p{shift}:' + out['wording']
    return out


def select_threshold(signal, insufficient, limit):
    s = np.asarray(signal, dtype=float); y = np.asarray(insufficient, dtype=bool)
    assert np.isfinite(s).all() and y.any() and (~y).any()
    choices = np.r_[np.nextafter(s.min(), -np.inf), np.unique(s), np.finfo(float).max]
    records = []
    for t in choices:
        use = s > t
        records.append({'threshold': float(t), 'recall': float(use[y].mean()),
                        'fpr': float(use[~y].mean()), 'trigger_fraction': float(use.mean())})
    valid = [r for r in records if r['fpr'] <= limit + 1e-12]
    return max(valid, key=lambda r: (r['recall'], -r['fpr'], r['threshold'])), records


def signals(x, probes, budget):
    prob = {name: expit(np.asarray(x, dtype=float) @ probes[f'{budget}_{name}_coef'] +
                        float(probes[f'{budget}_{name}_intercept'])) for name in GATES[:3]}
    return {'original': 1-prob['original'], 'expanded': 1-prob['expanded'],
            'evidence': prob['evidence'], 'product': (1-prob['expanded'])*prob['evidence']}


def apply(base, counterfactual, unknown, mask, action, amount=0.):
    base = np.asarray(base); mask = np.asarray(mask, dtype=bool); unknown = np.asarray(unknown)
    scores = base.copy()
    if action == 'steer': scores[mask] = counterfactual[mask]
    elif action == 'offset':
        idx = np.flatnonzero(mask & (unknown >= 0)); scores[idx, unknown[idx]] += amount
    elif action not in ('direct', 'base'): raise ValueError(action)
    pred = scores.argmax(1)
    if action == 'direct':
        use = mask & (unknown >= 0); pred[use] = unknown[use]
    return pred, None if action == 'direct' else scores


def fit_logistic(x, y, c):
    y = np.asarray(y); valid = y >= 0
    assert np.unique(y[valid]).tolist() == [0, 1], 'Both classes required'
    with threadpool_limits(limits=c['cpu_threads']), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        clf = LogisticRegression(C=c['probe_C'], solver='lbfgs', max_iter=c['probe_max_iter'],
                                 tol=c['probe_tol']).fit(x[valid], y[valid])
    if any(isinstance(w.message, ConvergenceWarning) for w in caught): raise RuntimeError('Probe did not converge')
    coef = clf.coef_[0].astype(float); intercept = float(clf.intercept_[0])
    np.testing.assert_array_equal((np.asarray(x[valid], float)@coef+intercept > 0).astype(int), clf.predict(x[valid]))
    return coef, intercept, {'rows': int(valid.sum()), 'classes': dict(Counter(map(str, y[valid]))),
                             'iterations': int(clf.n_iter_[0]), 'train_accuracy': float(clf.score(x[valid], y[valid]))}


def length_match(ambig, clear, tokenizer):
    target = len(tokenizer(render(clear)).input_ids)
    def n(s): return len(tokenizer(render({**ambig, 'context': ambig['context']+s})).input_ids)
    # No answer-bearing words. Exact tokenizer length is diagnostic, not natural text.
    suffix = ''
    for phrase in (' A separate note listed ordinary objects:', ' A note listed:', ' Note:'):
        if n(phrase) <= target: suffix = phrase; break
    while n(suffix) < target:
        old = n(suffix)
        for word in (' paper', ' shelf', ' item', '.'):
            new = n(suffix+word)
            if old < new <= target: suffix += word; break
        else: raise ValueError('Could not exactly match token length')
    assert n(suffix) == target
    return suffix, target


def pair_data(source, assignment, per_category, seed):
    groups = defaultdict(list)
    for r in source:
        if assignment[r['family']] == 'test':
            groups[(r['category'], r['question_index'], r['question'], r['ans0'], r['ans1'], r['ans2'])].append(r)
    candidates = defaultdict(list)
    for group in groups.values():
        for a in group:
            if a['context_condition'] != 'ambig': continue
            cs = [r for r in group if r['context_condition']=='disambig' and r['context'].startswith(a['context']+' ')]
            if len(cs) == 1:
                b = cs[0]; candidates[a['category']].append((a, b))
    selected = []
    for cat in sorted(candidates):
        pool = sorted(candidates[cat], key=lambda p: ranked(seed, p[0]['uid']))
        used_families = Counter(); used_stereo = Counter()
        for _ in range(per_category):
            a, b = min(pool, key=lambda p: (used_families[p[0]['family']],
                used_stereo[p[1]['label']==p[1]['target_loc']], ranked(seed,p[0]['uid'])))
            pool.remove((a,b)); selected.append((a,b)); used_families[a['family']]+=1
            used_stereo[b['label']==b['target_loc']]+=1
    assert len(selected) == per_category*11
    return selected


class Study:
    def __init__(self, config=CONFIG):
        self.config=Path(config); self.c=tomllib.loads(self.config.read_text(encoding='utf-8'))
        self.out=ROOT/self.c['output_dir']; self.data=ROOT/self.c['data_dir']
        self.out.mkdir(parents=True, exist_ok=True); self.data.mkdir(parents=True, exist_ok=True)

    def prepare(self):
        if (self.out/'plan_frozen.json').exists():
            saved=read(self.out/'plan_frozen.json')
            for p,h in saved['inputs'].items(): assert sha(ROOT/p)==h, f'Frozen input changed: {p}'
            for p,h in saved['data'].items(): assert sha(self.data/p)==h
            return
        src=rows(ROOT/self.c['source_data']/'bbq.jsonl'); split=read(ROOT/self.c['split_file'])
        mapping=read(ROOT/'configs/bbq_template_families.json')['mapping']
        for r in src:r['family']=family(r,mapping)
        assignment=split['family_assignment']; eligible=split['bbq_indices']['fit']
        fitids=balanced_indices(src,eligible,self.c['fit_per_condition_category'],self.c['seed'])
        budgets={str(n): balanced_indices(src,fitids,n//22,self.c['seed']) for n in self.c['fit_budgets']}
        assert all(set(budgets[str(a)])<=set(budgets[str(b)]) for a,b in zip(self.c['fit_budgets'],self.c['fit_budgets'][1:]))
        fit=[src[i] for i in fitids]; vocab=sorted({r['ans'+str(r['unknown_index'])].strip().casefold() for r in fit})
        directions=[]
        for cat in sorted({r['category'] for r in fit}):
            pool=[r for r in fit if r['category']==cat and r['context_condition']=='ambig' and r['target_loc'] is not None]
            directions += sorted(pool,key=lambda r:ranked(self.c['seed']+1,r['uid']))[:self.c['direction_per_category']]
        assert len(directions)==110 and all(r['label']==r['unknown_index']!=r['target_loc'] for r in directions)
        datasets={'fit':fit,'dev':[src[i] for i in split['bbq_indices']['dev']],
                  'test':[src[i] for i in split['bbq_indices']['test']]}
        mmlu=rows(ROOT/self.c['source_data']/'mmlu.jsonl')
        for k in ('dev','test'):
            datasets[k+'_utility']=[{**mmlu[i],'utility':True,'uid':f'mmlu:{i}','label':mmlu[i]['answer'],
                                   'family':f'mmlu:{i}'} for i in split['mmlu_indices'][k]]
        stressids=balanced_indices(src,split['bbq_indices']['test'],self.c['stress_per_condition_category'],self.c['seed']+2)
        stress=[]
        for i in stressids:
            r=src[i]; alt=next(v for v in vocab if v != r['ans'+str(r['unknown_index'])].casefold())
            for p in range(3):
                for wording in (None,alt):stress.append(rotate(r,p,wording))
        datasets['stress']=stress
        pairs=pair_data(src,assignment,self.c['counterfactual_pairs_per_category'],self.c['seed']+3)
        tokenstats={}
        for model in self.c['models']:
            tok=AutoTokenizer.from_pretrained(model['path'],local_files_only=True); vs=[]; lengths=[]
            for a,b in pairs:
                suffix,target=length_match(a,b,tok); pair=a['uid']+'|'+b['uid']
                for kind,r,extra in [('insufficient',a,''),('sufficient',b,''),
                                      ('neutral_matched',a,suffix),('sufficient_neutral',b,suffix)]:
                    item={**r,'context':r['context']+extra,'source_uid':r['uid'],'pair_id':pair,
                          'variant':kind,'uid':pair+':'+kind}
                    vs.append(item); lengths.append(len(tok(render(item)).input_ids))
                assert lengths[-4+1]==lengths[-4+2]==target
            datasets['counterfactual_'+model['name']]=vs
            tokenstats[model['name']]={'lengths':lengths,'exact_matched_pairs':len(pairs)}
        allfamilies={k:{r['family'] for r in datasets[k]} for k in ('fit','dev','test')}
        assert not allfamilies['fit']&allfamilies['dev'] and not allfamilies['fit']&allfamilies['test'] and not allfamilies['dev']&allfamilies['test']
        assert {a['family'] for a,b in pairs}<=set(k for k,v in assignment.items() if v=='test')
        for vs in datasets.values():
            for r in vs:
                if not r.get('utility'):assert discover_unknown(r,vocab)==r['unknown_index']
        for k,vs in datasets.items():
            (self.data/f'{k}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in vs),encoding='utf-8')
        write(self.data/'training.json',{'fit_indices':fitids,'budget_indices':budgets,'direction_uids':[r['uid'] for r in directions],
            'unknown_vocabulary':vocab,'token_stats':tokenstats})
        paths=[self.config,Path(__file__),ROOT/'mechanism/fairsteer_author.py',ROOT/'mechanism/fairsteer_improve.py',
               ROOT/'mechanism/fairsteer_data.py',ROOT/'research/2026-10-17_evidence_story_protocol_v1.md',
               ROOT/'scripts/run_evidence_story.py',ROOT/'tests/test_evidence_story.py',
               ROOT/self.c['source_data']/'bbq.jsonl',ROOT/self.c['source_data']/'mmlu.jsonl',
               ROOT/self.c['split_file'],ROOT/'configs/bbq_template_families.json']
        models={m['name']:{p.name:sha(p) for p in Path(m['path']).glob('*') if p.suffix in ('.json','.safetensors','.model')} for m in self.c['models']}
        write(self.out/'plan_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
            'inputs':{str(p.relative_to(ROOT)):sha(p) for p in paths},'data':{p.name:sha(p) for p in self.data.glob('*')},
            'models':models,'counts':{k:len(v) for k,v in datasets.items()},'family_counts':{k:len(v) for k,v in allfamilies.items()},
            'historical_exposure':'All native BBQ has historical evaluation exposure; derivatives are exploratory, not fresh confirmation.'})
        for p in paths:
            dst=self.out/'source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        print(json.dumps({'prepared':True,'counts':{k:len(v) for k,v in datasets.items()}},ensure_ascii=False),flush=True)


class ModelStudy:
    def __init__(self, name, config=CONFIG):
        self.study=Study(config); self.c=self.study.c; self.spec=next(m for m in self.c['models'] if m['name']==name)
        self.name=name; self.root=self.study.out; self.data=self.study.data; self.out=self.root/name
        self.out.mkdir(exist_ok=True); self.model=None; self.tokenizer=None; self.layer=self.spec['layer']
        self.training=read(self.data/'training.json'); self.vocab=self.training['unknown_vocabulary']
        self.vector=None; self.probes=None

    def log(self,event,**details):
        v={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'model':self.name,'event':event,**details}
        print(json.dumps(v,ensure_ascii=False),flush=True)
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')

    def values(self,key):return rows(self.data/f'{key}.jsonl')

    def setup(self):
        if self.model is not None:return
        torch.set_num_threads(self.c['cpu_threads']);torch.manual_seed(self.c['seed'])
        expected=read(self.root/'plan_frozen.json')['models'][self.name]
        assert all(sha(Path(self.spec['path'])/p)==h for p,h in expected.items())
        self.log('loading_model')
        self.tokenizer=AutoTokenizer.from_pretrained(self.spec['path'],local_files_only=True,padding_side='left')
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        self.model=AutoModelForCausalLM.from_pretrained(self.spec['path'],local_files_only=True,dtype=torch.float16,
            attn_implementation='sdpa').to('cuda').eval();self.model.requires_grad_(False)
        self.ids=[self.tokenizer('Answer: '+s).input_ids[-1] for s in 'ABCD']
        assert len(set(self.ids))==4
        assert all(self.tokenizer.decode([i]).strip()==s for i,s in zip(self.ids,'ABCD'))
        write(self.out/'environment.json',{'torch':torch.__version__,'transformers':__import__('transformers').__version__,
            'sklearn':__import__('sklearn').__version__,'gpu':torch.cuda.get_device_name(),'layer':self.layer,
            'layers':len(self.model.model.layers),'batch_size':self.c['batch_size'],'dtype':'float16','candidate_ids':self.ids,
            'hidden_size':self.model.config.hidden_size})

    def forward(self,vs,alpha=0.,append=None):
        self.setup(); texts=[render(r)+((' '+'ABCD'[a]) if append is not None else '') for r,a in zip(vs,append if append is not None else [0]*len(vs))]
        enc=self.tokenizer(texts,padding=True,return_tensors='pt')
        assert enc.input_ids.shape[1]<=self.c['max_tokens'],'No truncation'
        inputs={k:v.to('cuda') for k,v in enc.items()};inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).clamp(min=0)
        captured=[]; v=torch.as_tensor(self.vector,device='cuda',dtype=torch.float64) if alpha else None
        def hook(module,args,output):
            h=output[0] if isinstance(output,tuple) else output;captured.append(h[:,-1].detach().float().cpu().numpy())
            if not alpha:return output
            changed=h.clone();changed[:,-1]=(h[:,-1].double()+alpha*v).to(h.dtype)
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        handle=self.model.model.layers[self.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode(): out=self.model(**inputs,use_cache=False,logits_to_keep=1)
        finally:handle.remove()
        scores=out.logits[:,-1,self.ids].float().cpu().numpy()
        for i,r in enumerate(vs):
            if not r.get('utility'):scores[i,3]=-1e9
        assert len(captured)==1 and np.isfinite(scores).all() and np.isfinite(captured[0]).all()
        return scores,captured[0]

    def infer(self,key,edited=True):
        final=self.out/f'cache_{key}.npz'
        identity=digest([sha(self.root/'plan_frozen.json'),self.name,key,edited,
                         sha(self.out/'direction.npy') if edited else None])
        if final.exists():
            z=dict(np.load(final));assert str(z['identity'])==identity;return z
        vs=self.values(key); folder=self.out/'shards'/key;folder.mkdir(parents=True,exist_ok=True); parts=[]
        for start in range(0,len(vs),self.c['batch_size']):
            values=vs[start:start+self.c['batch_size']];path=folder/f'{start:06d}.npz'
            if path.exists():
                z=dict(np.load(path));assert str(z.pop('identity'))==identity
            else:
                s,x=self.forward(values);z={'base':s,'features':x}
                if edited:
                    for a in self.c['alphas']:z[f'cf{a:g}']=self.forward(values,a)[0]
                np.savez(path,identity=identity,**z)
            parts.append(z)
            if start%(self.c['batch_size']*16)==0:self.log('infer',dataset=key,done=start+len(values),total=len(vs))
        z={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};z['uids']=np.array([r['uid'] for r in vs]);z['identity']=np.array(identity)
        np.savez(final,**z);self.log('infer_complete',dataset=key,rows=len(vs));return z

    def smoke(self):
        if (self.out/'smoke.json').exists():return
        vs=self.values('dev')[:4];s,x=self.forward(vs);self.vector=np.ones(x.shape[1],dtype=float)*.01
        edited,_=self.forward(vs,1.);restored,rx=self.forward(vs)
        np.testing.assert_array_equal(s,restored);np.testing.assert_array_equal(x,rx)
        assert not np.array_equal(s,edited)
        one,ox=self.forward(vs[:1]);inp=self.tokenizer(render(vs[0]),return_tensors='pt').to('cuda')
        with torch.inference_mode():ref=self.model(**inp,use_cache=False,output_hidden_states=True,logits_to_keep=1)
        np.testing.assert_array_equal(ox[0],ref.hidden_states[self.layer+1][0,-1].float().cpu().numpy())
        self.vector=np.zeros(x.shape[1]);zero,_=self.forward(vs,1.);np.testing.assert_array_equal(s,zero)
        write(self.out/'smoke.json',{'hook_restore_exact':True,'zero_vector_exact':True,'hidden_state_hook_exact':True,
            'batch_single_max_score_delta':float(abs(s[0,:3]-one[0,:3]).max()),
            'batch_single_feature_relative_l2':float(np.linalg.norm(x[0]-ox[0])/np.linalg.norm(ox[0])),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        self.vector=None;self.log('smoke_complete')

    def train(self):
        self.smoke()
        if (self.out/'selection_frozen.json').exists():return
        cache=self.infer('fit',False);vs=self.values('fit');idx={r['uid']:i for i,r in enumerate(vs)}
        if not (self.out/'direction.npy').exists():
            chosen=[vs[idx[u]] for u in self.training['direction_uids']]; hh=[]
            folder=self.out/'direction_shards';folder.mkdir(exist_ok=True)
            for start in range(0,len(chosen),self.c['batch_size']):
                values=chosen[start:start+self.c['batch_size']];path=folder/f'{start:04d}.npz'
                if path.exists():z=dict(np.load(path))
                else:
                    _,good=self.forward(values,append=[r['label'] for r in values]);_,bad=self.forward(values,append=[r['target_loc'] for r in values])
                    z={'delta':good.astype(float)-bad.astype(float)};np.savez(path,**z)
                hh.append(z['delta'])
            self.vector=np.concatenate(hh).mean(0);assert np.isfinite(self.vector).all()
            np.save(self.out/'direction.npy',self.vector);self.log('direction_complete',norm=float(np.linalg.norm(self.vector)))
        self.vector=np.load(self.out/'direction.npy')
        if not (self.out/'probes.npz').exists():
            allsource=rows(ROOT/self.c['source_data']/'bbq.jsonl');exports={};diagnostics={}
            labels={'original':np.array([label_for(r,int(p)) for r,p in zip(vs,cache['base'].argmax(1))]),
                    'expanded':np.array([int(p==r['label']) for r,p in zip(vs,cache['base'].argmax(1))]),
                    'evidence':np.array([int(r['context_condition']=='ambig') for r in vs])}
            for budget in self.c['fit_budgets']:
                ii=np.array([idx[allsource[i]['uid']] for i in self.training['budget_indices'][str(budget)]])
                for name,y in labels.items():
                    co,inter,d=fit_logistic(cache['features'][ii],y[ii],self.c)
                    exports[f'{budget}_{name}_coef']=co;exports[f'{budget}_{name}_intercept']=np.array(inter)
                    diagnostics[f'{budget}_{name}']=d;self.log('probe_fit',budget=budget,head=name,**d)
            np.savez(self.out/'probes.npz',**exports);write(self.out/'probe_fit.json',diagnostics)
        self.probes=dict(np.load(self.out/'probes.npz'))
        dev=self.infer('dev');utility=self.infer('dev_utility')
        dv=self.values('dev');y=np.array([r['context_condition']=='ambig' for r in dv]);unknown=np.array([discover_unknown(r,self.vocab) for r in dv])
        selections={};allrecords={}
        for budget in self.c['fit_budgets']:
            sig=signals(dev['features'],self.probes,budget)
            for gate in GATES:
                best,candidates=select_threshold(sig[gate],y,self.c['gate_fpr_limit']);mask=sig[gate]>best['threshold'];offset={}
                for alpha in self.c['alphas']:
                    pred,_=apply(dev['base'],dev[f'cf{alpha:g}'],unknown,mask,'steer')
                    target=float(np.mean(pred==unknown));rr=[]
                    for a in np.arange(0,16.001,.25):
                        p,_=apply(dev['base'],None,unknown,mask,'offset',a)
                        rr.append({'amount':float(a),'unknown_rate':float(np.mean(p==unknown))})
                    offset[f'{alpha:g}']=min(rr,key=lambda r:(abs(r['unknown_rate']-target),r['amount']))
                    offset[f'{alpha:g}']['target_rate']=target
                selections[f'{budget}_{gate}']={'gate':best,'offsets':offset};allrecords[f'{budget}_{gate}']=candidates
        write(self.out/'selection_candidates.json',allrecords)
        write(self.out/'selection_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'selections':selections,
            'parameters':{p:sha(self.out/p) for p in ('probes.npz','direction.npy','probe_fit.json')},
            'dev_cache':sha(self.out/'cache_dev.npz'),'plan_sha256':sha(self.root/'plan_frozen.json')})
        self.log('selection_frozen')

    def policies(self,key):
        vs=self.values(key);cache=dict(np.load(self.out/f'cache_{key}.npz'))
        assert cache['uids'].tolist()==[r['uid'] for r in vs]
        probes=dict(np.load(self.out/'probes.npz'));sel=read(self.out/'selection_frozen.json')['selections']
        unknown=np.array([discover_unknown(r,self.vocab) for r in vs]);pred={'base':cache['base'].argmax(1)}
        masks={'base':np.zeros(len(vs),dtype=bool)};score={'base':cache['base']}
        for budget in self.c['fit_budgets']:
            sig=signals(cache['features'],probes,budget)
            for gate in GATES:
                spec=sel[f'{budget}_{gate}'];mask=sig[gate]>spec['gate']['threshold'];stem=f'{budget}_{gate}'
                for action in ('direct','steer1','steer1.5','offset1','offset1.5'):
                    name=stem+'_'+action;a=1.5 if action.endswith('1.5') else 1.
                    cf=cache[f'cf{a:g}'];kind='direct' if action=='direct' else ('steer' if action.startswith('steer') else 'offset')
                    amount=spec['offsets'][f'{a:g}']['amount'] if kind=='offset' else 0.
                    pred[name],ss=apply(cache['base'],cf,unknown,mask,kind,amount);masks[name]=mask
                    if ss is not None:score[name]=ss
        sig=signals(cache['features'],probes,max(self.c['fit_budgets']))
        for name,gate,t,a in [('fixed_local_fs','original',.5,1.),('fixed_product_steer','product',.25,1.5),
                              ('unconditional_steer','evidence',-1.,1.5)]:
            mask=sig[gate]>t;pred[name],score[name]=apply(cache['base'],cache[f'cf{a:g}'],unknown,mask,'steer');masks[name]=mask
        return vs,cache,pred,masks,score

    def evaluate(self):
        lock=read(self.root/'all_models_frozen.json')
        assert lock[self.name]==sha(self.out/'selection_frozen.json')
        selection=read(self.out/'selection_frozen.json')
        for p,h in selection['parameters'].items():assert sha(self.out/p)==h
        self.vector=np.load(self.out/'direction.npy')
        for key in ('test','test_utility','stress','counterfactual_'+self.name):self.infer(key)
        self.log('all_target_inference_complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);p.add_argument('--model');a=p.parse_args()
    study=Study();study.prepare()
    if a.stage!='prepare':getattr(ModelStudy(a.model),a.stage)()


if __name__=='__main__':main()
