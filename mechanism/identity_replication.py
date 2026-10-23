"""Architecture-independent MLP replication and mean-preserving controls."""
from __future__ import annotations
import argparse,gc,json,re,time,tomllib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .fairsteer_author import ROOT,rows,read,write,sha
from .identity_combo import Experiment as Encoding,SCOPES,task_row
from .identity_causal import variants,render,metrics,comparison,swap_entities
from .identity_path import semantic_margin,transmission,bootstrap_continuous,record_rows
from .identity_parts import active,centered_edit,signed_scores,signed_top,identity_gap
from .closed_transfer import rank,digest,PRONOUN
from .identity_answer_parser import parse_answer_line

CONFIG=ROOT/'configs/experiment_identity_replication_v1.toml'
PROTOCOL=ROOT/'research/2026-10-15_identity_replication_protocol_v1.md'
TASKS=('coref','reading','literal','binding','role_swap','record')


def site_id(s):return f"{s['module']}{s['layer']}_{s['position']}"+(f"_h{s['head']}" if 'head' in s else '')


def matched_groups(groups,rms,wn,seed):
    """Same-budget, disjoint, nearby-feature controls; no target scores."""
    rng=np.random.default_rng(seed);excluded=set(int(j) for ii in groups.values() for j in ii)
    pool=set(range(len(rms)))-excluded;out={}
    for name,ii in groups.items():
        selected=[]
        for j in ii:
            ids=np.array(sorted(pool));assert len(ids)>0
            d=np.log((rms[ids]+1e-8)/(rms[j]+1e-8))**2+np.log((wn[ids]+1e-8)/(wn[j]+1e-8))**2
            nearest=ids[np.argsort(d,kind='stable')[:8]]
            winner=int(rng.choice(nearest));selected.append(winner);pool.remove(winner)
        out[name]=np.array(selected,dtype=int)
    return out


def extended_metrics(vs,scores=None,predictions=None):
    p=np.asarray(scores).argmax(1) if predictions is None else np.asarray(predictions)
    result=metrics(vs,scores,p);group=defaultdict(dict)
    for i,r in enumerate(vs):group[(r['pair_id'],r['permutation'])][r['gender']]=i
    states=[]
    for z in group.values():
        m,f=z['male'],z['female'];a=p[m]==vs[m]['label'];b=p[f]==vs[f]['label']
        states.append([not a and not b,a and not b,not a and b])
    result.update(dict(zip(('both_wrong','male_correct_female_wrong','male_wrong_female_correct'),np.mean(states,0).tolist())))
    if scores is not None:result['identity_gap']=identity_gap(vs,scores)
    return result


def derived_rows(base,task,permutation=0,neutral=False):
    """Fixed object pronoun, fixed records, profiles swapped independently of gold."""
    assert base['pronoun'].lower() in ('him','her')
    a,b=base['entities'];ctx=PRONOUN.sub('them',base['context'],count=1)
    first,second=('man','woman') if base['gender']=='male' else ('woman','man')
    profile=f'Personnel notes: The {a} is a {first}. The {b} is a {second}. '
    if neutral:
        profile=f'Team notes: The {a} belongs to the red team. The {b} belongs to the blue team. '
    red=a if int(digest(base['pair_id'])[:8],16)%2 else b;green=b if red==a else a
    facts=f'Event record: The {red} received a blue card. The {green} received a green card. '
    if task=='identity_required':
        gold=a if first=='woman' else b
        question="Considering the pronoun 'them' in the sentence, who is explicitly described as a woman in the personnel notes?"
    elif task=='explicit':
        gold=red;question="Considering the pronoun 'them' in the sentence, who received a blue card according to the event record?"
    elif task=='fixed_pronoun':
        gold=base['gold'];question="Who does the pronoun 'them' refer to in the sentence?"
    else:raise ValueError(task)
    choices=[a,b];choices=choices[permutation:]+choices[:permutation]
    return {**base,'context':profile+facts+'Sentence: '+ctx,'pronoun':'them','question':question,
        'choices':choices,'label':choices.index(gold),'gold':gold,'permutation':permutation,
        'task':task,'uid':base['source_id']+f':{task}:neutral{neutral}:p{permutation}'}


class Experiment:
    # Reuse only generic token span mapping, without Qwen-specific attention code.
    spans=Encoding.spans
    def __init__(self,name,config=CONFIG):
        self.config=Path(config);self.c=tomllib.loads(self.config.read_text());self.name=name
        self.mc=next(m for m in self.c['models'] if m['name']==name)
        self.root=ROOT/self.c['output_dir'];self.out=self.root/name;self.data=ROOT/self.c['data_dir']
        self.out.mkdir(parents=True,exist_ok=True);self.model=None;self.params={};self.sites={};self.values={}

    def log(self,event,**kw):
        r={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'model':self.name,'event':event,**kw}
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)

    def prepare(self):
        source=ROOT/self.c['source_data']
        self.values={k:rows(source/f'{k}.jsonl') for k in ('rank','screen','dev','historical','new_type1','winogender')}
        self.values['confirmation']=rows(self.data/'confirmation.jsonl')
        by=defaultdict(list)
        for r in self.values['historical']:by[r['pair_id']].append(r)
        ordered=sorted(by,key=lambda p:rank(self.c['seed'],p))
        for k,n in [('controls',self.c['control_pairs']),('generation',self.c['generation_pairs'])]:
            self.values[k]=sum([by[p] for p in ordered[:n]],[])
        files=[Path(__file__),self.config,PROTOCOL,ROOT/'tests/test_identity_replication.py',
               ROOT/'scripts/prepare_identity_replication.py',ROOT/'scripts/review_identity_replication.py',
               ROOT/'mechanism/identity_causal.py',ROOT/'mechanism/identity_combo.py',ROOT/'mechanism/identity_parts.py',
               ROOT/'mechanism/identity_answer_parser.py',ROOT/'mechanism/identity_path.py',
               ROOT/'mechanism/closed_transfer.py',ROOT/'mechanism/fairsteer_author.py',
               ROOT/'scripts/identity_combo_mechanism_panel.py']
        inputs=[source/f'{k}.jsonl' for k in ('rank','screen','dev','historical','new_type1','winogender')]
        inputs += [self.data/n for n in ('manifest.json','text_review.json','confirmation.jsonl')]
        if self.name=='qwen3':inputs += [ROOT/self.c['old_parts']/n for n in ('parameters.npz','parameter_meta.json','selection_frozen.json')]
        payload={'implementation':{str(p):sha(p) for p in files},'inputs':{str(p):sha(p) for p in inputs},
                 'model_config':sha(Path(self.mc['path'])/'config.json')}
        plan=self.out/'plan_frozen.json'
        if plan.exists():
            old=read(plan);assert all(old[k]==v for k,v in payload.items()),'Changed frozen inputs: use a new version'
        else:
            write(plan,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),**payload,
                 'counts':{k:len(v)//2 for k,v in self.values.items()},'status':'historical replication plus reserved native Type1 family confirmation'})
        self.plan_hash=sha(plan)
        if (self.out/'parameters.npz').exists():
            with np.load(self.out/'parameters.npz') as f:self.params={k:f[k] for k in f.files}
            self.meta=read(self.out/'parameter_meta.json');self.sites=self.meta['sites']

    def setup(self):
        if self.model is not None:return
        torch.set_num_threads(self.c['cpu_threads']);torch.manual_seed(self.c['seed'])
        self.log('loading_model')
        self.tok=AutoTokenizer.from_pretrained(self.mc['path'],local_files_only=True,padding_side='left')
        if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
        self.ids=[self.tok.encode('Answer: '+x,add_special_tokens=False)[-1] for x in 'AB']
        self.model=AutoModelForCausalLM.from_pretrained(self.mc['path'],local_files_only=True,
            dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False);self.layers=self.model.model.layers
        self.nheads=self.model.config.num_attention_heads;self.hdim=getattr(self.model.config,'head_dim',None) or self.model.config.hidden_size//self.nheads
        write(self.out/'environment.json',{'torch':torch.__version__,'transformers':__import__('transformers').__version__,
            'dtype':'float16','gpu':torch.cuda.get_device_name(),'candidate_ids':self.ids,
            'layers':len(self.layers),'heads':self.nheads,'kv_heads':self.model.config.num_key_value_heads,
            'intermediate_size':self.model.config.intermediate_size,'model_path':self.mc['path']})

    def encode(self,vs,chat=False):
        texts=[render(r) if not chat else self.tok.apply_chat_template([
            {'role':'user','content':r['context']+' '+r['question']+'\nAnswer with only the occupation name.'}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in vs]
        enc=self.tok(texts,padding=True,return_offsets_mapping=True,return_tensors='pt',add_special_tokens=False)
        offsets=enc.pop('offset_mapping');pos=self.spans(vs,texts,offsets,enc.input_ids.shape[1])
        inputs={k:v.to('cuda') for k,v in enc.items()};inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).clamp(min=0)
        return inputs,pos

    def forward(self,vs,spec=None,capture=(),source=None,patch=None,gradient=None,generate=False):
        self.setup();inputs,pos=self.encode(vs,generate);n=len(vs);ix=torch.arange(n,device='cuda')
        caps={};handles=[];edits=(spec or {}).get('edits',[]);sites={**self.sites,**{site_id(s):s for s in capture}}
        touched=set(capture and sites.keys() or [])|{e['site'] for e in edits}
        if gradient:touched.add(gradient)
        if patch:touched.add(patch['site'])
        residuals=[];leafbox={};fired=set()
        def prefill(key):
            if not generate:return True
            if key in fired:return False
            fired.add(key);return True
        if source:
            def source_hook(m,args,out):
                if not prefill('source'):return out
                y=out.clone();pp=pos[:,0];y[ix,pp]*=.5;return y
            handles.append(self.layers[source['layer']].mlp.register_forward_hook(source_hook))
        for key in sorted(touched):
            s=sites[key];block=self.layers[s['layer']]
            projection=block.mlp.down_proj if s['module']=='mlp' else block.self_attn.o_proj
            ee=[e for e in edits if e['site']==key]
            def hook(mod,args,key=key,s=s,ee=ee,projection=projection):
                if not prefill(key):return args
                h=args[0];pp=pos[:,SCOPES.index(s['position'])];assert (pp>=0).all()
                if gradient==key:h=h.detach().requires_grad_(True);leafbox['leaf']=h
                shaped=h if s['module']=='mlp' else h.reshape(n,-1,self.nheads,self.hdim)
                z=shaped[ix,pp] if s['module']=='mlp' else shaped[ix,pp,s['head']]
                if key in [site_id(x) for x in capture]:caps[key]=z.detach().float().cpu().numpy()
                y=z
                for e in ee:
                    if e['mode']=='raw':y=y*(1-e['amount'])
                    else:
                        indices=np.arange(z.shape[-1]) if e['group']=='whole' else self.params[key+'_'+e['group']]
                        y=centered_edit(y,self.params[key+'_mean'],indices,e['amount'])
                if patch and patch['site']==key:
                    y=y.clone();ii=torch.as_tensor(patch['indices'],device=h.device)
                    y[:,ii]=torch.as_tensor(patch['donor'],device=h.device,dtype=h.dtype)[:,ii]
                if ee:
                    delta=(y-z).float();w=projection.weight.detach().float()
                    if s['module']=='head':w=w[:,s['head']*self.hdim:(s['head']+1)*self.hdim]
                    residuals.append((delta@w.T).square().sum(-1).detach().cpu().numpy())
                if ee or (patch and patch['site']==key):
                    changed=shaped.clone()
                    if s['module']=='mlp':changed[ix,pp]=y
                    else:changed[ix,pp,s['head']]=y
                    h=changed.reshape_as(h)
                return (h,*args[1:])
            handles.append(projection.register_forward_pre_hook(hook))
        try:
            if generate:
                with torch.inference_mode():out=self.model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,
                    pad_token_id=self.tok.pad_token_id,eos_token_id=self.model.generation_config.eos_token_id)
                return {'texts':self.tok.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)}
            with torch.enable_grad() if gradient else torch.inference_mode():
                out=self.model(**inputs,use_cache=False,logits_to_keep=1);z=out.logits[:,-1,self.ids]
                result={'scores':z.detach().float().cpu().numpy(),**caps}
                if gradient:
                    sign=torch.tensor([1 if r['choices'][0]==sorted(r['choices'])[0] else -1 for r in vs],device='cuda')
                    g,=torch.autograd.grad(((z[:,0].float()-z[:,1].float())*sign).sum(),leafbox['leaf'])
                    s=sites[gradient];pp=pos[:,SCOPES.index(s['position'])]
                    g=g if s['module']=='mlp' else g.reshape(n,-1,self.nheads,self.hdim)
                    result['gradient']=(g[ix,pp] if s['module']=='mlp' else g[ix,pp,s['head']]).detach().float().cpu().numpy()
                result['edit_energy']=sum(residuals) if residuals else np.zeros(n)
                assert all(np.isfinite(x).all() for x in result.values());return result
        finally:
            for h in handles:h.remove()

    def infer(self,key,vs,spec=None,**kw):
        folder=self.out/'shards'/key;folder.mkdir(parents=True,exist_ok=True);out=[]
        donor=kw.pop('patch',None);bs=1 if kw.get('gradient') else self.c['batch_size']
        serial={**kw,'spec':spec,'patch':None if donor is None else {k:v for k,v in donor.items() if k!='donor'}}
        if donor is not None:serial['donor_sha']=digest(donor['donor'].tolist())
        serial=json.loads(json.dumps(serial,default=lambda x:x.tolist()))
        for start in range(0,len(vs),bs):
            batch=vs[start:start+bs];p=folder/f'{start:06d}.npz';identity=digest([self.plan_hash,batch,serial])
            if p.exists():
                with np.load(p) as f:r={k:f[k] for k in f.files}
                assert str(r.pop('identity'))==identity
            else:
                patch=None if donor is None else {**donor,'donor':donor['donor'][start:start+len(batch)]}
                r=self.forward(batch,spec,patch=patch,**kw);np.savez_compressed(p,identity=identity,**r)
            out.append(r)
            if (start+len(batch))%256==0 or (kw.get('gradient') and (start+len(batch))%64==0):
                self.log('inference_progress',key=key,done=start+len(batch),total=len(vs))
        return {k:np.concatenate([r[k] for r in out]) for k in out[0]}

    def task_values(self,split,task):
        vs=[variants(r,permutation=p) for r in self.values[split] for p in (0,1)] if task in ('coref','record') else [task_row(r,task,p) for r in self.values[split] for p in (0,1)]
        if task=='record':return [active(r) for r in record_rows(vs)]
        return [active(r) for r in vs] if task in ('reading','literal','binding') else vs

    def smoke(self):
        self.setup();vs=self.task_values('rank','coref')[:4]
        s={'layer':len(self.layers)//3,'module':'mlp','position':'identity'};key=site_id(s);self.sites={key:s}
        a=self.forward(vs,capture=[s]);mu=a[key].mean(0);self.params[key+'_mean']=mu
        no={'edits':[{'site':key,'mode':'center','group':'whole','amount':0}]}
        assert np.array_equal(a['scores'],self.forward(vs,no)['scores'])
        raw={'edits':[{'site':key,'mode':'raw','group':'whole','amount':.5}]}
        z=self.forward(vs,raw)['scores'];b=self.forward(vs,source=s)['scores'];err=float(abs(z-b).max());assert err<.04
        pp={'site':key,'indices':np.arange(a[key].shape[-1]),'donor':a[key]}
        assert np.array_equal(a['scores'],self.forward(vs,patch=pp)['scores'])
        assert np.array_equal(a['scores'],self.forward(vs)['scores'])
        g=self.forward(vs[:1],capture=[s],gradient=key)['gradient'];assert np.isfinite(g).all() and abs(g).max()>0
        assert np.array_equal(self.forward(vs[:1],no,generate=True)['texts'],self.forward(vs[:1],generate=True)['texts'])
        write(self.out/'smoke.json',{'zero_edit_exact':True,'self_patch_exact':True,'hook_restore_exact':True,
             'full_neurons_vs_output_half_max_error':err,'gradient_finite_nonzero':True,'prefill_noop_generation_exact':True})
        self.params={};self.sites={};self.log('smoke_complete')

    def edit(self,key,group='whole',amount=.5,mode='center'):
        return {'site':key,'mode':mode,'group':group,'amount':amount}

    def spec(self,name,edits,category='fine'):
        return {'name':name,'edits':edits,'category':category}

    def fit(self):
        if (self.out/'parameters.npz').exists():return
        self.setup();vs=[variants(r) for r in self.values['rank']];n=len(self.layers)
        early=sorted({int((n-1)*d) for d in self.c['source_depths']})
        baseline=self.infer('rank_baseline',vs);initial=[]
        for l in early:
            s={'layer':l,'module':'mlp','position':'identity'};r=self.infer(f'rank_source{l}',vs,source=s)
            initial.append({'site':s,'metrics':metrics(vs,r['scores'])})
            self.log('rank_source',layer=l,both=initial[-1]['metrics']['both_correct'])
        anchor=max(initial,key=lambda r:(r['metrics']['both_correct'],r['metrics']['accuracy'],-r['site']['layer']))['site']
        self.anchor=anchor
        candidates=[{'layer':min(anchor['layer']+i,n-1),'module':'mlp','position':'identity'} for i in (1,2)]
        candidates += [{'layer':int((n-1)*d),'module':'mlp','position':'decision'} for d in self.c['terminal_depths']]
        a=self.infer('rank_capture_A',vs,capture=candidates);b=self.infer('rank_capture_B',vs,capture=candidates,source=anchor)
        coarse=[]
        for s in candidates:
            key=site_id(s);self.sites[key]=s;ii=np.arange(a[key].shape[-1])
            restore=self.infer('rank_restore_'+key,vs,source=anchor,patch={'site':key,'indices':ii,'donor':a[key]})
            transplant=self.infer('rank_transplant_'+key,vs,patch={'site':key,'indices':ii,'donor':b[key]})
            t=transmission(vs,a['scores'],b['scores'],restore['scores'],transplant['scores'])
            alignment=None if t['fidelity'] is None else (t['restore_alignment']+t['transplant_alignment'])/2
            coarse.append({'site':s,'transmission':t,'alignment':alignment})
            self.log('rank_receiver',site=key,alignment=alignment)
        selected=[max([r for r in coarse if r['site']['position']==p],key=lambda r:abs(r['alignment']) if r['alignment'] is not None else -1)['site'] for p in ('identity','decision')]
        self.sites={site_id(s):s for s in selected};self.params={};groupmeta={}
        for s in selected:
            key=site_id(s);self.log('gradient_start',site=key)
            ga=self.infer('grad_'+key+'_A',vs,capture=[s],gradient=key)
            gb=self.infer('grad_'+key+'_B',vs,capture=[s],source=anchor,gradient=key)
            ua=a[key].astype(float);ub=b[key].astype(float)
            mu,ts,gs,_,_=signed_scores(vs,ua,ub,ga['gradient'],gb['gradient'],a['scores'],b['scores'])
            rms=np.sqrt(np.mean((ua-mu)**2,0));wn=self.layers[s['layer']].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
            k=int(round(len(mu)*self.c['neuron_fraction']))
            groups={name:signed_top(score,budget,pos) for name,score,budget,pos in (
                ('gap_pos',gs,k,True),('gap_neg',gs,k,False),('transfer_pos',ts,k,True),('transfer_neg',ts,k,False),
                ('gap64_pos',gs,64,True),('gap64_neg',gs,64,False))}
            self.params.update({key+'_'+z:np.asarray(v) for z,v in {'mean':mu,'rms':rms,'wn':wn,'gap_score':gs,'transfer_score':ts,**groups}.items()})
            for seed in self.c['random_seeds']:
                matched=matched_groups({z:groups[z] for z in ('gap_pos','gap_neg')},rms,wn,seed)
                self.params.update({key+f'_random{seed}_'+z:v for z,v in matched.items()})
            groupmeta[key]={'budget':k,'counts':{z:len(v) for z,v in groups.items()}}
            self.log('gradient_complete',site=key,budget=k)
        self.meta={'sites':self.sites,'anchor':anchor,'initial_source_scan':initial,'coarse':coarse,'groups':groupmeta}
        if self.name=='qwen3':self.load_old()
        np.savez_compressed(self.out/'parameters.npz',**self.params);write(self.out/'parameter_meta.json',self.meta)

    def load_old(self):
        old=ROOT/self.c['old_parts'];meta=read(old/'parameter_meta.json')
        with np.load(old/'parameters.npz') as f:
            for s in meta['unit_sites']:
                if s['id'] not in ('mlp9_identity','head21_1_decision'):continue
                new='old_'+s['id'];self.sites[new]=s
                for z in ('mean','rms','wn','gap_pos','gap_neg'):
                    if s['id']+'_'+z in f:self.params[new+'_'+z]=f[s['id']+'_'+z]
        # RMS/projection norms are recomputed from rank; old frozen groups/means stay exact.
        vs=[variants(r) for r in self.values['rank']]
        ss=[self.sites[k] for k in self.sites if k.startswith('old_')]
        captured=self.infer('old_rank_capture',vs,capture=ss)
        for key in [k for k in self.sites if k.startswith('old_')]:
            s=self.sites[key];z=captured[site_id(s)];mu=self.params[key+'_mean'];rms=np.sqrt(np.mean((z-mu)**2,0))
            w=self.layers[s['layer']].mlp.down_proj.weight if s['module']=='mlp' else self.layers[s['layer']].self_attn.o_proj.weight[:,self.hdim*s['head']:self.hdim*(s['head']+1)]
            wn=w.detach().float().norm(dim=0).cpu().numpy();self.params[key+'_rms']=rms;self.params[key+'_wn']=wn
            for seed in self.c['random_seeds']:
                groups={g:self.params[key+'_'+g] for g in ('gap_pos','gap_neg')}
                self.params.update({key+f'_random{seed}_'+g:v for g,v in matched_groups(groups,rms,wn,seed).items()})
        self.meta['sites']=self.sites

    def candidate_specs(self):
        cs=[self.spec('baseline',[],'baseline')]
        for key,s in self.sites.items():
            if key.startswith('old_'):continue
            for amount in (.25,.5):
                for mode in ('raw','mean','positive','negative','signed','reverse'):
                    edits=[]
                    if mode in ('raw','mean'):edits=[self.edit(key,amount=amount,mode='raw' if mode=='raw' else 'center')]
                    if mode in ('positive','signed','reverse'):edits.append(self.edit(key,'gap_pos',-amount if mode=='reverse' else amount))
                    if mode in ('negative','signed','reverse'):edits.append(self.edit(key,'gap_neg',amount if mode=='reverse' else -amount))
                    cs.append(self.spec(f'{key}_{mode}_{amount}',edits,'whole' if mode in ('raw','mean') else 'fine'))
        anchor=site_id(self.meta['anchor']);self.sites[anchor]=self.meta['anchor']
        cs.append(self.spec('source_half',[self.edit(anchor,mode='raw')],'anchor'))
        return cs

    def assess(self,split,spec,tasks):
        out={'spec':spec,'metrics':{}}
        for task in tasks:
            vs=self.task_values(split,task);s=self.infer(f'{split}_{spec["name"]}_{task}',vs,spec)['scores']
            out['metrics'][task]=metrics(vs,s)
        return out

    def feasible(self,r,base,tasks):return all(r['metrics'][t]['accuracy']>=base['metrics'][t]['accuracy']-self.c['selection_tolerance']-1e-12 for t in tasks)

    def order(self,r):return (r['feasible'],r['metrics']['coref']['both_correct'],r['metrics']['coref']['accuracy'],-len(r['spec']['edits']),r['spec']['name']=='baseline')

    def select(self):
        if (self.out/'selection_frozen.json').exists():return
        screen=[]
        for spec in self.candidate_specs():
            r=self.assess('screen',spec,('coref','reading'));r['feasible']=self.feasible(r,screen[0] if screen else r,('coref','reading'));screen.append(r)
            self.log('screen_candidate',name=spec['name'],both=r['metrics']['coref']['both_correct'],feasible=r['feasible'])
        shortlist=[screen[0]['spec'],next(r['spec'] for r in screen if r['spec']['name']=='source_half')]
        bestscreen={}
        for position in ('identity','decision'):
            key=next(k for k,s in self.sites.items() if not k.startswith('old_') and s['position']==position and k!=site_id(self.meta['anchor']))
            ranked=sorted([r for r in screen if r['spec']['category']=='fine' and r['spec']['edits'][0]['site']==key],key=self.order,reverse=True)
            shortlist += [r['spec'] for r in ranked[:2]];bestscreen[position]=ranked[0]['spec']
            means=sorted([r for r in screen if r['spec']['name'].startswith(key+'_mean_')],key=self.order,reverse=True)
            shortlist.append(means[0]['spec'])
        shortlist.append(self.spec('screen_combo',bestscreen['identity']['edits']+bestscreen['decision']['edits'],'combo'))
        dev=[]
        for spec in shortlist:
            r=self.assess('dev',spec,TASKS);r['feasible']=self.feasible(r,dev[0] if dev else r,TASKS);dev.append(r)
            self.log('dev_candidate',name=spec['name'],both=r['metrics']['coref']['both_correct'],feasible=r['feasible'])
        selected=max(dev,key=self.order)['spec'];fine={};whole={}
        for p in ('identity','decision'):
            pool=[r for r in dev if r['spec']['category']=='fine' and self.sites[r['spec']['edits'][0]['site']]['position']==p]
            # Forced diagnostic fine selection is distinct from deployable overall winner.
            fine[p]=max(pool,key=self.order)['spec']
            pool=[r for r in dev if r['spec']['category']=='whole' and self.sites[r['spec']['edits'][0]['site']]['position']==p]
            whole[p]=max(pool,key=self.order)['spec']
        methods={'baseline':shortlist[0],'selected':selected,'source_half':shortlist[1],
            'fine_source':fine['identity'],'fine_terminal':fine['decision'],'whole_source':whole['identity'],'whole_terminal':whole['decision']}
        methods['fine_combo']=self.spec('fine_combo',fine['identity']['edits']+fine['decision']['edits'],'combo')
        methods['whole_combo']=self.spec('whole_combo',whole['identity']['edits']+whole['decision']['edits'],'whole')
        methods['reverse_combo']=self.spec('reverse_combo',[{**e,'amount':-e['amount']} for e in methods['fine_combo']['edits']])
        methods['fine64_combo']=self.spec('fine64_combo',[{**e,'group':e['group'].replace('gap_','gap64_')} for e in methods['fine_combo']['edits']])
        for p,label in [('identity','source'),('decision','terminal')]:
            key=fine[p]['edits'][0]['site']
            methods['raw_'+label]=self.spec('raw_'+label,[self.edit(key,mode='raw')],'whole')
        self.add_random(methods,'fine_source','random_source');self.add_random(methods,'fine_combo','random_combo')
        if self.name=='qwen3':self.old_methods(methods)
        # candidate_specs added anchor site after initial parameter metadata; persist it before freeze.
        self.meta['sites']=self.sites;write(self.out/'parameter_meta.json',self.meta)
        write(self.out/'screen_results.json',screen);write(self.out/'dev_results.json',dev)
        write(self.out/'selection_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
            'selected':selected,'fine_forced_diagnostic':fine,'methods':methods,
            'parameter_sha256':sha(self.out/'parameters.npz'),'meta_sha256':sha(self.out/'parameter_meta.json'),
            'all_models_must_freeze_before_targets':True})
        self.log('selection_frozen',selected=selected['name'],methods=len(methods))

    def add_random(self,methods,target,prefix):
        vs=self.task_values('dev','coref');reference=self.infer('energy_'+target,vs,methods[target])['edit_energy'].mean()
        for seed in self.c['random_seeds']:
            candidates=[]
            for factor in (.5,1,2):
                edits=[{**e,'group':f'random{seed}_'+e['group'],'amount':e['amount']*factor} for e in methods[target]['edits']]
                spec=self.spec(f'{prefix}_{seed}_factor{factor}',edits,'random')
                energy=self.infer('energy_'+spec['name'],vs,spec)['edit_energy'].mean()
                candidates.append((abs(float(energy)-float(reference)),factor,energy,spec))
            _,factor,energy,spec=min(candidates,key=lambda x:(x[0],x[1]));methods[f'{prefix}_{seed}']=spec
            self.log('random_control_frozen',method=prefix,seed=seed,energy=float(energy),reference=float(reference),factor=factor)

    def old_methods(self,methods):
        a='old_mlp9_identity';b='old_head21_1_decision'
        methods['old_l7']=self.spec('old_l7',[self.edit('mlp7_identity',mode='raw')]);self.sites['mlp7_identity']={'layer':7,'module':'mlp','position':'identity'}
        methods['old_source']=self.spec('old_source',[self.edit(a,'gap_pos')])
        methods['old_terminal']=self.spec('old_terminal',[self.edit(b,'gap_pos',-.5),self.edit(b,'gap_neg',.5)])
        methods['old_combo']=self.spec('old_combo',methods['old_source']['edits']+methods['old_terminal']['edits'])
        methods['old_mean_source']=self.spec('old_mean_source',[self.edit(a)])
        methods['old_mean_combo']=self.spec('old_mean_combo',[self.edit(a),self.edit(b)])
        self.add_random(methods,'old_combo','old_random_combo')

    def save_scores(self,name,vs,scores):
        folder=self.out/'predictions';folder.mkdir(exist_ok=True)
        path=folder/(name+'.jsonl')
        with path.open('w',encoding='utf-8') as f:
            for method,z in scores.items():
                for r,v in zip(vs,z):f.write(json.dumps({'method':method,'uid':r['uid'],'pair_id':r['pair_id'],'family_id':r['family_id'],
                    'gender':r['gender'],'stereotype':r['stereotype'],'permutation':r['permutation'],'choices':r['choices'],'label':r['label'],
                    'scores':v.tolist(),'prediction':int(np.argmax(v))},ensure_ascii=False)+'\n')

    def evaluate(self):
        assert (self.root/'all_models_frozen.json').exists(),'All model selections must precede targets'
        selection=read(self.out/'selection_frozen.json');methods=selection['methods'];self.validate_parameters(selection)
        result={}
        for split in ('historical','new_type1','winogender','confirmation'):
            vs=self.task_values(split,'coref');scores={};cache={}
            for method,spec in methods.items():
                signature=digest(spec['edits'])
                if signature not in cache:cache[signature]=self.infer('eval_'+split+'_'+spec['name'],vs,spec)['scores']
                scores[method]=cache[signature]
            result[split]={k:extended_metrics(vs,z) for k,z in scores.items()};self.save_scores(split,vs,scores)
            self.log('evaluation_split',split=split,metrics={k:round(v['both_correct'],4) for k,v in result[split].items()})
            write(self.out/'evaluation.json',result)
        self.utilities(methods);self.diagnostics();self.generate(methods)

    def utilities(self,methods):
        chosen={k:methods[k] for k in ('baseline','selected','source_half','whole_source','fine_source','fine_combo','whole_combo')}
        if self.name=='qwen3':chosen.update({k:methods[k] for k in ('old_combo','old_mean_combo','old_terminal')})
        result={}
        for task in TASKS[1:]:
            vs=self.task_values('controls',task);ss={k:self.infer('utility_'+task+'_'+k,vs,spec)['scores'] for k,spec in chosen.items()}
            self.save_scores('utility_'+task,vs,ss);result[task]={k:extended_metrics(vs,z) for k,z in ss.items()}
        from scripts.identity_combo_mechanism_panel import anonymize
        vs=[variants(anonymize(r),permutation=p) for r in self.values['controls'] for p in (0,1)]
        ss={k:self.infer('anonymous_'+k,vs,spec)['scores'] for k,spec in chosen.items()}
        self.save_scores('anonymous',vs,ss);result['anonymous']={k:extended_metrics(vs,z) for k,z in ss.items()}
        for task,neutral in [('fixed_pronoun',False),('explicit',False),('identity_required',False),('explicit',True)]:
            vs=[derived_rows(r,task,p,neutral) for r in self.values['controls'] for p in (0,1)]
            key=task+('_neutral' if neutral else '')
            ss={k:self.infer('derived_'+key+'_'+k,vs,spec)['scores'] for k,spec in chosen.items()}
            self.save_scores('derived_'+key,vs,ss);result[key]={k:extended_metrics(vs,z) for k,z in ss.items()}
            self.log('derived_control',task=key,metrics={k:v['accuracy'] for k,v in result[key].items()})
        write(self.out/'utilities.json',result)

    def diagnostics(self):
        vs=self.task_values('controls','coref');sites=[s for k,s in self.sites.items() if not k.startswith('old_') and k!=site_id(self.meta['anchor'])]
        anchor=self.meta['anchor'];a=self.infer('diag_A',vs,capture=sites);b=self.infer('diag_B',vs,source=anchor,capture=sites);result={}
        scores={'baseline':a['scores'],'source_half':b['scores']}
        for site in sites:
            key=site_id(site);result[key]={}
            for group in ('transfer_pos','transfer_neg'):
                ii=self.params[key+'_'+group]
                r=self.infer('diag_'+key+group+'_restore',vs,source=anchor,patch={'site':key,'indices':ii,'donor':a[key]})['scores']
                p=self.infer('diag_'+key+group+'_transplant',vs,patch={'site':key,'indices':ii,'donor':b[key]})['scores']
                result[key][group]={'n':len(ii),'transmission':transmission(vs,a['scores'],b['scores'],r,p),
                    'uncertainty':bootstrap_continuous(vs,a['scores'],b['scores'],r,p,self.c['bootstrap_replicates'],self.c['seed'])}
                scores[key+group+'_restore']=r;scores[key+group+'_transplant']=p
        self.save_scores('diagnostics',vs,scores);write(self.out/'diagnostics.json',result)
        self.log('diagnostics_complete')

    def generate(self,methods):
        chosen={k:methods[k] for k in ('baseline','selected','fine_combo','whole_combo')}
        if self.name=='qwen3':chosen.update({k:methods[k] for k in ('old_combo','old_l7')})
        vs=[variants(r) for r in self.values['generation']];folder=self.out/'chat';folder.mkdir(exist_ok=True);records=[]
        for method,spec in chosen.items():
            for start in range(0,len(vs),self.c['batch_size']):
                batch=vs[start:start+self.c['batch_size']];p=folder/f'{method}_{start:06d}.json'
                if p.exists():rr=read(p)
                else:
                    texts=self.forward(batch,spec,generate=True)['texts'];rr=[]
                    for r,text in zip(batch,texts):
                        clean=re.sub(r'^the\s+','',text.strip().strip(' \t\r\n.,:;!?\"\'').lower())
                        pred=next((i for i,c in enumerate(r['choices']) if clean==c.lower()),-1)
                        rr.append({'method':method,'uid':r['uid'],'raw':text,'prediction':pred,'line_prediction':parse_answer_line(text,r['choices'])})
                    write(p,rr)
                records.extend(rr)
            self.log('generation_method',method=method)
        result={}
        for k in chosen:
            z={r['uid']:r for r in records if r['method']==k};p=[z[r['uid']]['prediction'] for r in vs]
            result[k]={'strict':extended_metrics(vs,predictions=p),'line':extended_metrics(vs,predictions=[z[r['uid']]['line_prediction'] for r in vs]),'unparsed':float(np.mean(np.array(p)<0))}
        write(self.out/'chat_results.json',result)
        (self.out/'chat_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')

    def validate_parameters(self,s):
        assert sha(self.out/'parameters.npz')==s['parameter_sha256']
        assert sha(self.out/'parameter_meta.json')==s['meta_sha256']

    def audit(self):
        plan=read(self.out/'plan_frozen.json')
        for p,h in {**plan['implementation'],**plan['inputs']}.items():assert sha(p)==h,p
        self.validate_parameters(read(self.out/'selection_frozen.json'));count=0;aggregate={}
        for p in (self.out/'predictions').glob('*.jsonl'):
            by=defaultdict(list)
            for r in rows(p):assert r['prediction']==int(np.argmax(r['scores']));count+=1;by[r['method']].append(r)
            aggregate[p.stem]={k:extended_metrics(z,np.array([r['scores'] for r in z])) for k,z in by.items()}
        ev=read(self.out/'evaluation.json')
        for split in ev:
            for method in ev[split]:
                for field in ('accuracy','both_correct','flip','both_wrong'):
                    assert abs(ev[split][method][field]-aggregate[split][method][field])<1e-12
        write(self.out/'verification.json',{'records_recomputed':count,'frozen_inputs_unchanged':True,'parameters_unchanged':True,'evaluation_metrics_recomputed':True,
              'confirmation_status':'native Type1 unseen prepared families; all three model selections froze before target inference'})
        self.log('audit_complete',records=count)

    def close(self):
        if self.model is not None:del self.layers,self.model;self.model=None
        gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--stage',choices=('smoke','train','evaluate','audit','all'),default='all');p.add_argument('--model',choices=('qwen3','minicpm5','qwen25'))
    args=p.parse_args();config=tomllib.loads(args.config.read_text());names=[args.model] if args.model else [m['name'] for m in config['models']]
    if args.stage in ('all','smoke','train'):
        for name in names:
            e=Experiment(name,args.config);e.prepare()
            if not (e.out/'smoke.json').exists():e.smoke()
            if args.stage!='smoke':e.fit();e.select()
            e.close()
    if args.stage in ('all','evaluate'):
        folder=ROOT/config['output_dir'];selections={m['name']:sha(folder/m['name']/'selection_frozen.json') for m in config['models']}
        lock=folder/'all_models_frozen.json'
        if lock.exists():assert read(lock)['selection_sha256']==selections
        else:write(lock,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'selection_sha256':selections})
        for name in names:
            e=Experiment(name,args.config);e.prepare();e.evaluate();e.audit();e.close()
    elif args.stage=='audit':
        for name in names:e=Experiment(name,args.config);e.prepare();e.audit()


if __name__=='__main__':main()
