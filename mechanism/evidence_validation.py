"""Repeated source fits, Qwen2.5 replication, and exposure-audited confirmation."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import shutil
import time
import tomllib

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from .fairsteer_author import ROOT,read,rows,write,sha,label_for
from .fairsteer_improve import balanced_indices,digest,ranked,family
from .evidence_story import ModelStudy,fit_logistic,signals,select_threshold,discover_unknown,apply,length_match,surface
from .closed_transfer import canonical,template,gender,digest as synth_digest

CONFIG=ROOT/'configs/experiment_evidence_validation_v1.toml'
GATES=('original','expanded','evidence','product')


def audit_exposure(data_root,exclude):
    used_ids=set();used_families=set();used_occupations=set();files={}
    for path in sorted(data_root.rglob('*.jsonl')):
        if exclude in path.parents:continue
        found=False
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line)
                if not r.get('source_id','').startswith('synthbias:'):continue
                found=True;used_ids.add(r['source_id']);used_families.add(r['family_id']);used_occupations.add(r['occupation_pair'])
        if found:files[str(path.relative_to(ROOT))]=sha(path)
    return {'source_ids':sorted(used_ids),'families':sorted(used_families),
            'occupations':sorted(used_occupations),'files':files}


def confirmation_data(path,exposure,c):
    bins=defaultdict(list)
    with path.open(encoding='utf-8',newline='') as f:
        raw=list(csv.DictReader(f))
    assert len(raw)==31728
    for i,r in enumerate(raw):bins[canonical(r['sample'])].append({**r,'source_id':f'synthbias:{i:05d}'})
    used_ids=set(exposure['source_ids']);used_f=set(exposure['families']);used_o=set(exposure['occupations'])
    pools=defaultdict(list)
    for gs in bins.values():
        if len(gs)!=2 or {gender(r['pronoun']) for r in gs}!={'male','female'}:continue
        if len({(r['type'],r['occ_1'] if r['type']=='type2' else None) for r in gs})!=1:continue
        r=gs[0];f='synth_template:'+synth_digest(template(r['sample'],(r['occ_1'],r['occ_2'])))[:20]
        o='|'.join(sorted((r['occ_1'],r['occ_2'])))
        if any(x['source_id'] in used_ids for x in gs) or f in used_f or o in used_o:continue
        pools[r['type']].append((gs,f,o))
    counts={k:len(v) for k,v in pools.items()};selected=[];wanted=c['confirmation_rows']//4
    assert c['confirmation_rows']%4==0
    for typ in ('type1','type2'):
        assert len(pools[typ])>=wanted,(typ,len(pools[typ]),wanted)
        pool=list(pools[typ]);families=Counter();occs=Counter()
        for _ in range(wanted):
            item=min(pool,key=lambda z:(families[z[1]],occs[z[2]],ranked(c['seed']+100,z[0][0]['source_id'])))
            pool.remove(item);selected.append(item);families[item[1]]+=1;occs[item[2]]+=1
    result=[]
    for gs,f,o in selected:
        pair='synth:'+synth_digest(canonical(gs[0]['sample']))[:20]
        entities=sorted((gs[0]['occ_1'],gs[0]['occ_2']),key=lambda v:ranked(c['seed'],pair+v))
        choices=list(entities);choices.insert(int(ranked(c['seed'],pair)[:8],16)%3,c['confirmation_unknown'])
        for r in gs:
            gold=c['confirmation_unknown'] if r['type']=='type1' else r['occ_1']
            for p in range(3):
                options=choices[p:]+choices[:p]
                result.append({'uid':r['source_id']+f':p{p}','source_id':r['source_id'],'pair_id':pair,
                    'family_id':f,'family':f,'occupation_pair':o,'category':'SynthBias','context':r['sample'],
                    'question':f"Who does the pronoun '{r['pronoun']}' refer to?",'gender':gender(r['pronoun']),
                    'context_condition':'ambig' if r['type']=='type1' else 'disambig',
                    'stereotype':r['is_stereotypical'].lower(),'task_type':r['type'],
                    **{f'ans{i}':v for i,v in enumerate(options)},'unknown_index':options.index(c['confirmation_unknown']),
                    'label':options.index(gold),'target_loc':None,'permutation':p})
    result.sort(key=lambda r:r['uid'])
    assert len(result)==c['confirmation_rows']*3
    assert not {r['source_id'] for r in result}&used_ids
    assert not {r['family_id'] for r in result}&used_f
    assert not {r['occupation_pair'] for r in result}&used_o
    return result,{'eligible_gender_pairs':counts,'questions':len(result)//3,
        'families':len({r['family_id'] for r in result}),'occupation_pairs':len({r['occupation_pair'] for r in result})}


class Campaign:
    def __init__(self,config=CONFIG):
        self.config=Path(config);self.c=tomllib.loads(self.config.read_text(encoding='utf-8'))
        self.out=ROOT/self.c['output_dir'];self.data=ROOT/self.c['data_dir'];self.ref=ROOT/self.c['reference_dir'];self.refdata=ROOT/self.c['reference_data']
        self.out.mkdir(parents=True,exist_ok=True);self.data.mkdir(parents=True,exist_ok=True)

    def prepare(self):
        path=self.out/'plan_frozen.json'
        if path.exists():
            plan=read(path)
            for p,h in plan['sources'].items():assert sha(ROOT/p)==h,('Frozen source changed',p)
            for p,h in plan['data'].items():assert sha(self.data/p)==h
            for p,h in plan['references'].items():assert sha(ROOT/p)==h
            return
        src=rows(ROOT/self.c['source_data']/'bbq.jsonl');split=read(ROOT/self.c['split_file'])
        mapping=read(ROOT/'configs/bbq_template_families.json')['mapping']
        for r in src:r['family']=family(r,mapping)
        training={};datasets={}
        for seed in self.c['seeds']:
            ii=balanced_indices(src,split['bbq_indices']['fit'],self.c['fit_per_condition_category'],seed)
            vs=[src[i] for i in ii];budgets={str(b):balanced_indices(src,ii,b//22,seed) for b in self.c['fit_budgets']}
            directions=[]
            for cat in sorted({r['category'] for r in vs}):
                pool=[r for r in vs if r['category']==cat and r['context_condition']=='ambig' and r['target_loc'] is not None]
                directions.extend(sorted(pool,key=lambda r:ranked(seed+1,r['uid']))[:self.c['direction_per_category']])
            vocab=sorted({r['ans'+str(r['unknown_index'])].strip().casefold() for r in vs})
            assert self.c['confirmation_unknown'].casefold() in vocab
            training[str(seed)]={'indices':ii,'budgets':budgets,'direction_uids':[r['uid'] for r in directions],'vocab':vocab}
            assert len(directions)==110
            datasets[f'fit_{seed}']=vs
        assert datasets[f'fit_{self.c["seed"]}']==rows(self.refdata/'fit.jsonl')
        for key in ('dev','test','dev_utility','test_utility','stress','counterfactual_qwen3','counterfactual_minicpm5'):
            datasets[key]=rows(self.refdata/f'{key}.jsonl')
        pairs=defaultdict(dict)
        for r in datasets['counterfactual_qwen3']:pairs[r['pair_id']][r['variant']]=r
        spec=next(m for m in self.c['models'] if m['name']=='qwen25');tok=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True)
        cf=[]
        for pair,group in pairs.items():
            a=group['insufficient'];b=group['sufficient'];suffix,n=length_match(a,b,tok)
            for variant,r,extra in [('insufficient',a,''),('sufficient',b,''),('neutral_matched',a,suffix),('sufficient_neutral',b,suffix)]:
                cf.append({**r,'uid':pair+':'+variant,'variant':variant,'context':r['context']+extra})
        datasets['counterfactual_qwen25']=cf
        exposure=audit_exposure(ROOT/'data/prepared',self.data)
        datasets['confirmation'],confirmation=confirmation_data(ROOT/self.c['synth_raw'],exposure,self.c)
        for seed in self.c['seeds']:
            vocab=training[str(seed)]['vocab']
            for key in ('dev','test','confirmation','stress'):
                assert all(discover_unknown(r,vocab)==r['unknown_index'] for r in datasets[key])
            fit_f={r['family'] for r in datasets[f'fit_{seed}']}
            assert not fit_f&{r['family'] for r in datasets['dev']+datasets['test']}
        assert not {r['family'] for r in datasets['dev']}&{r['family'] for r in datasets['test']}
        for key,vs in datasets.items():(self.data/f'{key}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in vs),encoding='utf-8')
        write(self.data/'training.json',training);write(self.data/'exposure.json',exposure)
        sources=[self.config,Path(__file__),ROOT/'mechanism/evidence_story.py',ROOT/'mechanism/fairsteer_author.py',
            ROOT/'mechanism/fairsteer_improve.py',ROOT/'mechanism/fairsteer_data.py',ROOT/'mechanism/closed_transfer.py',ROOT/'mechanism/evidence_validation_stats.py',
            ROOT/'research/2026-10-17_evidence_validation_protocol_v1.md',ROOT/'scripts/run_evidence_validation.py',
            ROOT/'tests/test_evidence_validation.py',ROOT/self.c['source_data']/'bbq.jsonl',ROOT/self.c['split_file'],
            ROOT/self.c['synth_raw'],ROOT/'configs/bbq_template_families.json']
        references={}
        for name in ('qwen3','minicpm5'):
            for file in ('probes.npz','probe_fit.json','direction.npy','selection_frozen.json',
                'cache_fit.npz','cache_dev.npz','cache_test.npz','cache_dev_utility.npz','cache_test_utility.npz',
                'cache_stress.npz',f'cache_counterfactual_{name}.npz'):
                p=self.ref/name/file;references[str(p.relative_to(ROOT))]=sha(p)
        references.update(exposure['files'])
        models={m['name']:{p.name:sha(p) for p in Path(m['path']).glob('*') if p.suffix in ('.json','.safetensors','.model')} for m in self.c['models']}
        write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},
            'references':references,'models':models,'data':{p.name:sha(p) for p in self.data.glob('*')},
            'confirmation':confirmation,'counts':{k:len(v) for k,v in datasets.items()},
            'scope':'Repeated BBQ historical retests; confirmation unseen to prior prepared experiments in IDs/templates/occupation pairs.'})
        for p in sources:
            if p.suffix not in ('.py','.toml','.md'):continue
            target=self.out/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        print(json.dumps({'prepared':True,'confirmation':confirmation,'counts':{k:len(v) for k,v in datasets.items()}}),flush=True)


class Engine(ModelStudy):
    def __init__(self,name,config=CONFIG):
        self.study=Campaign(config);self.c=self.study.c;self.spec=next(m for m in self.c['models'] if m['name']==name)
        self.name=name;self.root=self.study.out;self.data=self.study.data;self.out=self.root/name;self.out.mkdir(exist_ok=True)
        self.model=None;self.tokenizer=None;self.layer=self.spec['layer'];self.vector=None;self.probes=None
        self.training=read(self.data/'training.json');self.vocab=self.training[str(self.c['seed'])]['vocab']

    def baseline(self,key):
        path=self.out/f'cache_{key}.npz'
        if not path.exists() and self.name in ('qwen3','minicpm5'):
            oldkey='fit' if key==f'fit_{self.c["seed"]}' else key
            old=self.study.ref/self.name/f'cache_{oldkey}.npz'
            if old.exists():
                z=dict(np.load(old));assert z['uids'].tolist()==[r['uid'] for r in self.values(key)]
                identity=digest([sha(self.root/'plan_frozen.json'),self.name,key,False,None])
                np.savez(path,base=z['base'],features=z['features'],uids=z['uids'],identity=identity,reused_from=str(old))
        return self.infer(key,edited=False)

    def seed_folder(self,seed):
        p=self.out/f'seed_{seed}';p.mkdir(exist_ok=True);return p

    def counter(self,key,seed,alpha):
        folder=self.seed_folder(seed);path=folder/f'cf_{key}_{alpha:g}.npz';vs=self.values(key)
        identity=digest([sha(self.root/'plan_frozen.json'),self.name,key,seed,alpha,sha(folder/'direction.npy')])
        if path.exists():
            z=dict(np.load(path));assert str(z['identity'])==identity;assert z['uids'].tolist()==[r['uid'] for r in vs];return z['scores']
        old=self.study.ref/self.name/f'cache_{key}.npz'
        if seed==self.c['seed'] and self.name in ('qwen3','minicpm5') and old.exists():
            assert sha(folder/'direction.npy')==sha(self.study.ref/self.name/'direction.npy')
            z=dict(np.load(old));assert z['uids'].tolist()==[r['uid'] for r in vs];scores=z[f'cf{alpha:g}']
        else:
            self.vector=np.load(folder/'direction.npy');shards=folder/'shards'/f'{key}_{alpha:g}';shards.mkdir(parents=True,exist_ok=True);parts=[]
            for start in range(0,len(vs),self.c['batch_size']):
                p=shards/f'{start:06d}.npz';values=vs[start:start+self.c['batch_size']]
                if p.exists():
                    z=dict(np.load(p));assert str(z['identity'])==identity;s=z['scores']
                else:
                    s,_=self.forward(values,alpha);np.savez(p,scores=s,identity=identity)
                parts.append(s)
                if start%(self.c['batch_size']*32)==0:self.log('counter',dataset=key,seed=seed,alpha=alpha,done=start+len(values),total=len(vs))
            scores=np.concatenate(parts)
        np.savez(path,scores=scores,uids=np.array([r['uid'] for r in vs]),identity=identity)
        return scores

    def fit_seed(self,seed):
        folder=self.seed_folder(seed)
        if (folder/'selection_frozen.json').exists():return
        key=f'fit_{seed}';cache=self.baseline(key);vs=self.values(key);training=self.training[str(seed)];vocab=training['vocab']
        uid={r['uid']:i for i,r in enumerate(vs)}
        if not (folder/'direction.npy').exists():
            if seed==self.c['seed'] and self.name in ('qwen3','minicpm5'):
                shutil.copyfile(self.study.ref/self.name/'direction.npy',folder/'direction.npy')
            else:
                chosen=[vs[uid[u]] for u in training['direction_uids']];parts=[];shards=folder/'direction_shards';shards.mkdir(exist_ok=True)
                for start in range(0,len(chosen),self.c['batch_size']):
                    p=shards/f'{start:04d}.npz';values=chosen[start:start+self.c['batch_size']]
                    if p.exists():delta=np.load(p)['delta']
                    else:
                        _,a=self.forward(values,append=[r['label'] for r in values]);_,b=self.forward(values,append=[r['target_loc'] for r in values])
                        delta=a.astype(float)-b.astype(float);np.savez(p,delta=delta)
                    parts.append(delta)
                np.save(folder/'direction.npy',np.concatenate(parts).mean(0))
        if not (folder/'probes.npz').exists():
            if seed==self.c['seed'] and self.name in ('qwen3','minicpm5'):
                shutil.copyfile(self.study.ref/self.name/'probes.npz',folder/'probes.npz')
                shutil.copyfile(self.study.ref/self.name/'probe_fit.json',folder/'probe_fit.json')
            else:
                exports={};diagnostics={};source=rows(ROOT/self.c['source_data']/'bbq.jsonl')
                pred=cache['base'].argmax(1)
                labels={'original':np.array([label_for(r,int(p)) for r,p in zip(vs,pred)]),
                    'expanded':np.array([r['label']==p for r,p in zip(vs,pred)],int),
                    'evidence':np.array([r['context_condition']=='ambig' for r in vs],int)}
                for budget in self.c['fit_budgets']:
                    ii=np.array([uid[source[i]['uid']] for i in training['budgets'][str(budget)]])
                    for head,y in labels.items():
                        co,inter,d=fit_logistic(cache['features'][ii],y[ii],self.c)
                        exports[f'{budget}_{head}_coef']=co;exports[f'{budget}_{head}_intercept']=np.array(inter);diagnostics[f'{budget}_{head}']=d
                np.savez(folder/'probes.npz',**exports);write(folder/'probe_fit.json',diagnostics)
        probes=dict(np.load(folder/'probes.npz'));dv=self.values('dev');dev=self.baseline('dev');self.baseline('dev_utility')
        y=np.array([r['context_condition']=='ambig' for r in vs]);dy=np.array([r['context_condition']=='ambig' for r in dv])
        controls={};control_inputs={
            'surface':(np.array([surface(r) for r in vs]),np.array([surface(r) for r in dv])),
            'question':([r['question'] for r in vs],[r['question'] for r in dv]),
            'output':(output_features(vs,cache,vocab),output_features(dv,dev,vocab))}
        for kind,(x,dx) in control_inputs.items():
            path=folder/f'{kind}.joblib'
            if path.exists():model=joblib.load(path)
            else:
                model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=3000,min_df=2),LogisticRegression(C=1.,solver='lbfgs',max_iter=10000,tol=1e-4)) if kind=='question' else make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=10000,tol=1e-4))
                with threadpool_limits(limits=self.c['cpu_threads']):model.fit(x,y)
                joblib.dump(model,path)
            best,candidates=select_threshold(model.predict_proba(dx)[:,1],dy,self.c['gate_fpr_limit'])
            controls[kind]={'threshold':best,'candidates':candidates,'model_sha256':sha(path)}
        thresholds={};allrecords={}
        for budget in self.c['fit_budgets']:
            sig=signals(dev['features'],probes,budget)
            for gate in GATES:
                thresholds[f'{budget}_{gate}'],allrecords[f'{budget}_{gate}']=select_threshold(sig[gate],dy,self.c['gate_fpr_limit'])
        cf1=self.counter('dev',seed,1.);cf15=self.counter('dev',seed,1.5)
        unknown=np.array([discover_unknown(r,vocab) for r in dv]);mask=signals(dev['features'],probes,2200)['product']>thresholds['2200_product']['threshold']
        pp,_=apply(dev['base'],cf15,unknown,mask,'steer');target=float(np.mean(pp==unknown));offsets=[]
        for amount in np.arange(0,16.001,.25):
            p,_=apply(dev['base'],None,unknown,mask,'offset',amount);rate=float(np.mean(p==unknown));offsets.append((abs(rate-target),float(amount)))
        write(folder/'threshold_candidates.json',allrecords)
        write(folder/'selection_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'seed':seed,'thresholds':thresholds,
            'controls':controls,'product_offset':min(offsets)[1],
            'parameters':{p:sha(folder/p) for p in ('direction.npy','probes.npz','probe_fit.json','surface.joblib','question.joblib','output.joblib')},
            'plan_sha256':sha(self.root/'plan_frozen.json')})
        self.log('seed_frozen',seed=seed)

    def train(self):
        self.smoke()
        for seed in self.c['seeds']:self.fit_seed(seed)

    def evaluate(self):
        lock=read(self.root/'all_selections_frozen.json')
        for seed in self.c['seeds']:
            folder=self.seed_folder(seed);assert lock[self.name][str(seed)]==sha(folder/'selection_frozen.json')
            for p,h in read(folder/'selection_frozen.json')['parameters'].items():assert sha(folder/p)==h
        for key in ('test','test_utility','confirmation','stress','counterfactual_'+self.name):self.baseline(key)
        for seed in self.c['seeds']:
            for key in ('test','test_utility','confirmation'):
                for alpha in self.c['alphas']:self.counter(key,seed,alpha)
        for key in ('stress','counterfactual_'+self.name):
            for alpha in self.c['alphas']:self.counter(key,self.c['seed'],alpha)
        self.log('evaluation_complete')


def output_features(vs,cache,vocab):
    from scipy.special import logsumexp
    ss=cache['base'][:,:3].astype(float);u=np.array([discover_unknown(r,vocab) for r in vs]);ii=np.arange(len(vs))
    ee=np.array([[j for j in range(3) if j!=ui] for ui in u]);es=np.take_along_axis(ss,ee,axis=1)
    lp=ss-logsumexp(ss,axis=1,keepdims=True)
    return np.column_stack([ss[ii,u]-logsumexp(es,axis=1),abs(es[:,0]-es[:,1]),-(np.exp(lp)*lp).sum(1)])


def policies(e,key,seed):
    vs=e.values(key);cache=dict(np.load(e.out/f'cache_{key}.npz'));folder=e.seed_folder(seed)
    assert cache['uids'].tolist()==[r['uid'] for r in vs]
    sel=read(folder/'selection_frozen.json');probes=dict(np.load(folder/'probes.npz'));vocab=e.training[str(seed)]['vocab']
    sig=signals(cache['features'],probes,2200);unknown=np.array([discover_unknown(r,vocab) for r in vs])
    b=cache['base'];cf1=np.load(folder/f'cf_{key}_1.npz')['scores'];cf15=np.load(folder/f'cf_{key}_1.5.npz')['scores']
    pred={'base':b.argmax(1)};masks={'base':np.zeros(len(vs),bool)}
    for name,gate,action,t,cf in [
        ('local_fs','original','steer',.5,cf1),
        ('expanded_steer','expanded','steer',sel['thresholds']['2200_expanded']['threshold'],cf15),
        ('product_steer','product','steer',sel['thresholds']['2200_product']['threshold'],cf15),
        ('product_direct','product','direct',sel['thresholds']['2200_product']['threshold'],cf15),
        ('product_offset','product','offset',sel['thresholds']['2200_product']['threshold'],cf15),
        ('evidence_steer','evidence','steer',sel['thresholds']['2200_evidence']['threshold'],cf15),
        ('evidence_direct','evidence','direct',sel['thresholds']['2200_evidence']['threshold'],cf15)]:
        mask=sig[gate]>t;pred[name],_=apply(b,cf,unknown,mask,action,sel['product_offset']);masks[name]=mask
    utility=bool(vs[0].get('utility'))
    for kind in ('surface','question','output'):
        if utility:
            pred[kind+'_direct']=b.argmax(1);masks[kind+'_direct']=np.zeros(len(vs),bool);continue
        model=joblib.load(folder/f'{kind}.joblib')
        x=np.array([surface(r) for r in vs]) if kind=='surface' else ([r['question'] for r in vs] if kind=='question' else output_features(vs,cache,vocab))
        pp=model.predict_proba(x)[:,1];mask=pp>sel['controls'][kind]['threshold']['threshold']
        pred[kind+'_direct'],_=apply(b,None,unknown,mask,'direct');masks[kind+'_direct']=mask
    pred['always_unknown']=b.argmax(1);use=unknown>=0;pred['always_unknown'][use]=unknown[use];masks['always_unknown']=use
    return vs,cache,pred,masks,sig


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);p.add_argument('--model');a=p.parse_args()
    c=Campaign();c.prepare()
    if a.stage!='prepare':getattr(Engine(a.model),a.stage)()


if __name__=='__main__':main()
