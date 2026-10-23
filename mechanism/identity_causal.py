"""Paired identity interference pilot; oracle patches separated from inference edits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import re
import time
import tomllib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fairsteer_author import ROOT, read, rows, sha, write
from .closed_transfer import PRONOUN, canonical, digest, gender, rank, probe_values, winogender_data

CONFIG=ROOT/'configs/experiment_identity_causal_v1.toml'
PROTOCOL=ROOT/'research/2026-10-15_identity_causal_protocol_v1.md'
RAW=ROOT/'data/raw/closed_transfer/winobias/WinoBias/wino/data'
TASKS=('coref','reading','binding','role_swap')
INSTRUCTION='Use only the sentence and its grammatical structure. Do not infer a person\'s role from gender or occupational stereotypes. Output only the answer letter.\n\n'
# Reviewed from complete native pro development sentences before model outcomes.
# Additional structural rules below reject multi-pronoun/possessive/nonpaired rows.
REVIEW_EXCLUDE={
 5:'give rather than gave',6:'give rather than gave',11:'give rather than gave',
 13:'malformed threaten to sue',15:'malformed hoped her enjoy',20:'malformed thank',
 21:'malformed wished her had',22:'malformed wished him had',34:'malformed provide',
 39:'incomplete how to use',40:'incomplete how to use',41:'malformed wish him had',
 42:'malformed wish her had',49:'malformed helped her fixed',50:'article missing finance report',
 52:'threw him a bread',59:'possessive',64:'incomplete put on',70:'malformed ask',
 71:'possessive',74:'malformed thank',77:'malformed called to do CPR',
 80:'missing complementizer',89:'malformed wait her to pay',90:'malformed wait him for tips',
 92:'malformed told to',93:'missing predicate',99:'malformed ask',102:'greeted with',
 104:'malformed for preparing',112:'asked her for a dinner',113:'missing article brought her book',
 115:'incomplete discussed with',116:'incomplete discussed with',117:'malformed call',
 119:'malformed helped her to won',123:'malformed thank',124:'malformed thank',
 135:'malformed ask',136:'malformed ask',147:'malformed splash',148:'malformed splash',
 153:'treated him bad',154:'treated her bad',156:'malformed thank',164:'incomplete hoping that',
 171:'malformed demanded her pay',176:'malformed bracket spacing and determiner',
 195:'possessive her desk',199:'malformed like',203:'possessive her love',207:'malformed demanded her to pay',
 209:'card about typo',217:'missing to after talked',223:'unnatural dug her a hole',
 224:'malformed to a dig',231:'incomplete asked her',247:'unclear subject across passive coordination',
 249:'possessive her books',255:'possessive her examination',260:'malformed gave the bread the chief',
 269:'incomplete asked her to',275:'possessive her bread',306:'malformed order',
 327:'malformed explained him',328:'malformed explained her',331:'malformed explained him',
 338:'malformed borrowed took',340:'asked for a menu to',373:'malformed explained him',374:'malformed explained her'
}


def family(sentence, occupations):
    s=sentence.lower()
    for occ in sorted(occupations,key=len,reverse=True):
        s=re.sub(r'\b'+re.escape(occ)+r'\b','<OCC>',s,flags=re.I)
    return canonical(s)


def swap_entities(text, entities):
    a,b=entities
    pattern=re.compile(r'\b('+re.escape(a)+'|'+re.escape(b)+r')\b',re.I)
    return pattern.sub(lambda m:b if m.group().lower()==a.lower() else a,text)


def source_pairs():
    occs=sorted({o.strip() for f in ('female','male') for o in (RAW/f'{f}_occupations.txt').read_text().splitlines() if o.strip()},key=len,reverse=True)
    historical=set()
    for g in ('pro','anti'):
        for l in (RAW/f'{g}_stereotyped_type2.txt.test').read_text().splitlines():
            historical.add(family(re.sub(r'^\d+\s+','',l).replace('[','').replace(']',''),occs))
    groups=defaultdict(list);excluded=[]
    for group in ('pro','anti'):
        for index,line in enumerate((RAW/f'{group}_stereotyped_type2.txt.dev').read_text().splitlines(),1):
            numbered=int(line.split()[0]);marked=re.sub(r'^\d+\s+','',line)
            spans=[s.strip() for s in re.findall(r'\[([^]]+)\]',marked)]
            ps=[s for s in spans if PRONOUN.fullmatch(s)]
            s=marked.replace('[','').replace(']','')
            entities=[o for o in occs if re.search(r'\b'+re.escape(o)+r'\b',s,re.I)]
            reasons=[]
            if len(ps)!=1 or ps[0] not in ('him','her'):reasons.append('not_one_object_pronoun')
            if len(PRONOUN.findall(s))!=1:reasons.append('multiple_context_pronouns')
            if numbered in REVIEW_EXCLUDE:reasons.append('agent_text_review: '+REVIEW_EXCLUDE[numbered])
            if len(entities)!=2:reasons.append('not_two_occupations')
            fam=family(s,occs)
            if fam in historical:reasons.append('family_previously_inspected_test')
            if reasons:
                excluded.append({'source':group,'line':numbered,'sentence':s,'reasons':reasons});continue
            entityspans=[z for z in spans if not PRONOUN.fullmatch(z)]
            if len(entityspans)!=1:
                excluded.append({'source':group,'line':numbered,'sentence':s,'reasons':['nonunique_gold_span']});continue
            golds=[o for o in entities if re.search(r'\b'+re.escape(o)+r'\b',entityspans[0],re.I)]
            if len(golds)!=1:raise ValueError(line)
            r={'source_id':f'winobias:dev:type2:{group}:{numbered:04d}',
               'native_line':numbered,'context':s,'entities':sorted(entities),'gold':golds[0],
               'pronoun':ps[0],'gender':gender(ps[0]),'stereotype':group=='pro',
               'pair_id':'wbdev:'+digest(canonical(s))[:20],
               'family_id':'wbtemplate:'+digest(fam)[:20],
               'occupation_pair':'|'.join(sorted(entities)), 'template':fam}
            groups[r['pair_id']].append(r)
    accepted=[]
    for pid,vs in groups.items():
        if len(vs)!=2 or {r['gender'] for r in vs}!={'male','female'} or len({r['gold'] for r in vs})!=1:
            excluded.append({'pair_id':pid,'rows':vs,'reasons':['incomplete_gender_pair_or_gold_mismatch']});continue
        accepted.extend(vs)
    return sorted(accepted,key=lambda r:r['source_id']),excluded


def variants(base, task='coref', permutation=0):
    r=dict(base);a,b=r['entities'];gold=r['gold'];other=b if gold==a else a
    ctx=r['context'];p=r['pronoun']
    if task in ('coref','role_swap'):
        question=f"Who does the pronoun '{p}' refer to?";choices=[a,b]
        if task=='role_swap':ctx=swap_entities(ctx,[a,b]);gold=other
    elif task=='reading':
        question='Is the gendered pronoun in the sentence masculine or feminine?'
        choices=['masculine','feminine'];gold='masculine' if r['gender']=='male' else 'feminine'
    elif task=='binding':
        man=gold if r['gender']=='male' else other;woman=other if r['gender']=='male' else gold
        # Explicit public facts, not a label field appended to the coreference task.
        facts=[f'The {man} is a man.',f'The {woman} is a woman.']
        if int(digest(r['pair_id'])[:8],16)%2:facts.reverse()
        ctx+=' '+' '.join(facts)
        question='Who is explicitly described as a woman?';choices=[a,b];gold=woman
    else:raise ValueError(task)
    choices=choices[permutation:]+choices[:permutation]
    return {**r,'context':ctx,'question':question,'choices':choices,'label':choices.index(gold),
            'gold':gold,'task':task,'permutation':permutation,
            'uid':r['source_id']+f':{task}:p{permutation}'}


def render(r, instruction=False, openqa=False):
    if openqa:return r['context']+' '+r['question']+'\nAnswer with only the occupation name.\nAnswer:'
    return (INSTRUCTION if instruction else '')+r['context']+' '+r['question']+''.join(f'\n{a}. {v}' for a,v in zip('AB',r['choices']))+'\nAnswer:'


def pair_records(values, predictions):
    groups=defaultdict(list)
    for r,p in zip(values,predictions):groups[(r['pair_id'],r['permutation'])].append((r,p))
    out=[]
    for (pid,pos),vs in groups.items():
        assert len(vs)==2,(pid,len(vs))
        ok=[p==r['label'] for r,p in vs]
        answers=[r['choices'][p] if p>=0 else '<UNPARSED>' for r,p in vs]
        pro=[int(p==r['label']) for r,p in vs if r['stereotype']]
        anti=[int(p==r['label']) for r,p in vs if not r['stereotype']]
        out.append({'pair_id':pid,'family_id':vs[0][0]['family_id'],'occupation_pair':vs[0][0]['occupation_pair'],
            'permutation':pos,'both_correct':float(all(ok)),'accuracy':sum(ok)/2,
            'flip':float(answers[0]!=answers[1]),'gap':pro[0]-anti[0] if len(pro)==len(anti)==1 else 0})
    return out


def metrics(values, scores=None, predictions=None):
    if predictions is None:predictions=np.asarray(scores).argmax(1)
    pair=pair_records(values,predictions)
    m={'rows':len(values),'pairs':len({r['pair_id'] for r in values}),
       **{k:float(np.mean([r[k] for r in pair])) for k in ('accuracy','both_correct','flip','gap')}}
    m['by_permutation']={str(pos):{k:float(np.mean([r[k] for r in pair if r['permutation']==pos])) for k in ('accuracy','both_correct','flip','gap')}
                         for pos in sorted({r['permutation'] for r in pair})}
    if scores is not None:
        labs=np.array([r['label'] for r in values]);s=np.asarray(scores)
        m['gold_margin']=float(np.mean(s[np.arange(len(s)),labs]-s[np.arange(len(s)),1-labs]))
    return m


def comparison(values, a, b, field, reps, seed, cluster='family_id'):
    aa=pair_records(values,a);bb=pair_records(values,b);assert len(aa)==len(bb)
    groups=defaultdict(list)
    for x,y in zip(aa,bb):
        assert (x['pair_id'],x['permutation'])==(y['pair_id'],y['permutation'])
        groups[x[cluster]].append(x[field]-y[field])
    sums=np.array([sum(v) for v in groups.values()]);ns=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(reps):
        ix=rng.integers(0,len(sums),len(sums));estimates.append(sums[ix].sum()/ns[ix].sum())
    delta=float(sums.sum()/ns.sum());null=np.abs(rng.choice([-1,1],size=(reps,len(sums)))@(sums/ns.sum()))
    return {'delta':delta,'ci95':np.quantile(estimates,[.025,.975]).tolist(),'cluster':cluster,'clusters':len(sums),
            'p_sign':float((1+(null>=abs(delta)-1e-12).sum())/(reps+1))}


def feasible_choice(candidates, tolerance):
    base=candidates[0];valid=[]
    for c in candidates:
        c['feasible']=all(c['tasks'][t]['accuracy']>=base['tasks'][t]['accuracy']-tolerance-1e-12 for t in TASKS)
        if c['feasible']:valid.append(c)
    return max(valid,key=lambda c:(c['tasks']['coref']['both_correct'],c['tasks']['coref']['accuracy'],
                                  -c['spec'].get('strength',0),c['spec']['operation']=='none'))


class Experiment:
    def __init__(self,config=CONFIG):
        self.config=Path(config);self.c=tomllib.loads(self.config.read_text());self.out=ROOT/self.c['output_dir'];self.data=ROOT/self.c['data_dir']
        self.out.mkdir(parents=True,exist_ok=True);self.data.mkdir(parents=True,exist_ok=True)
        self.model=None;self.tok=None;self.values={};self.activations=None

    def log(self,event,**kw):
        r={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'event':event,**kw}
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)

    def prepare(self):
        identity={'code':sha(Path(__file__)),'config':sha(self.config),'protocol':sha(PROTOCOL)}
        plan=self.out/'plan_frozen.json'
        if plan.exists():
            saved=read(plan);assert saved['identity']==identity,'Implementation changed; use new experiment version'
            for n,h in saved['data_sha256'].items():assert sha(self.data/n)==h
            self.values={k:rows(self.data/f'{k}.jsonl') for k in ('fit','dev','confirmation','external')};return
        accepted,excluded=source_pairs();families=defaultdict(list)
        for r in accepted:families[r['family_id']].append(r)
        chosen={k:[] for k in ('fit','dev','confirmation')};split='fit'
        for fid in sorted(families,key=lambda k:rank(self.c['seed'],k)):
            if len(chosen['fit'])//2>=self.c['fit_pairs']:split='dev'
            if len(chosen['dev'])//2>=self.c['dev_pairs']:split='confirmation'
            chosen[split].extend(families[fid])
        assert len(chosen['confirmation'])//2>=60,'Insufficient unused families for confirmation'
        for a in chosen:
            for b in chosen:
                if a!=b:assert not {r['family_id'] for r in chosen[a]}&{r['family_id'] for r in chosen[b]}
        ext=winogender_data(ROOT/'data/raw/closed_transfer/winogender/data',self.c);eg=defaultdict(list)
        for r in ext:
            if r['permutation']==0 and r['gender'] in ('male','female'):eg[r['pair_id']].append(r)
        external=[]
        for pid,vs in sorted(eg.items()):
            if len(vs)!=2 or len({canonical(r['context']) for r in vs})!=1:continue
            for r in vs:
                p=PRONOUN.findall(r['context']);assert len(p)==1
                external.append({**r,'entities':sorted(r['choices']),'gold':r['choices'][r['label']],
                    'pronoun':p[0],'stereotype':r['stereotype']=='true'})
        chosen['external']=external
        for k,vs in chosen.items():
            self.values[k]=sorted(vs,key=lambda r:(r['pair_id'],r['gender']))
            (self.data/f'{k}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in self.values[k]),encoding='utf-8')
        write(self.data/'audit_exclusions.json',excluded)
        write(self.data/'audit_summary.json',{'audit':'agent text review and structural rules; no independent human annotation',
            'accepted_pairs':len(accepted)//2,'excluded_records_or_groups':len(excluded),'review_excluded_native_line_reasons':REVIEW_EXCLUDE,
            'counts':{k:{'pairs':len(v)//2,'families':len({r['family_id'] for r in v})} for k,v in self.values.items()}})
        inputs=[RAW/f'{g}_stereotyped_type2.txt.{s}' for g in ('pro','anti') for s in ('dev','test')]
        write(plan,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,
            'data_sha256':{p.name:sha(p) for p in self.data.glob('*')},'input_sha256':{str(p):sha(p) for p in inputs},
            'counts':{k:len(v)//2 for k,v in self.values.items()},'confirmation_status':'unused native development families; exploratory within one study'})
        for name,path in [('code_snapshot.py',Path(__file__)),('protocol_snapshot.md',PROTOCOL),('config_snapshot.toml',self.config)]:
            (self.out/name).write_bytes(path.read_bytes())
        self.log('prepared',**read(self.data/'audit_summary.json')['counts'])

    def setup(self):
        if self.model is not None:return
        torch.set_num_threads(self.c['cpu_threads']);torch.manual_seed(self.c['seed'])
        self.tok=AutoTokenizer.from_pretrained(self.c['model_path'],local_files_only=True,padding_side='left')
        if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
        self.ids=[self.tok('Answer: '+x,add_special_tokens=False).input_ids[-1] for x in 'AB']
        self.log('loading_model')
        self.model=AutoModelForCausalLM.from_pretrained(self.c['model_path'],local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False)
        self.probes=dict(np.load(ROOT/'results/fairsteer_improve_20261013_v2/probes.npz'))
        self.fs_vector=np.load(ROOT/'results/fairsteer_improve_20261013_v2/direction_dim.npy')
        self.sites=[{'layer':l,'module':m,'position':p} for l in range(len(self.model.model.layers)) for m in ('attention','mlp') for p in ('identity','decision')]
        write(self.out/'environment.json',{'torch':torch.__version__,'transformers':__import__('transformers').__version__,
            'gpu':torch.cuda.get_device_name(),'candidate_ids':self.ids,'sites':self.sites,
            'model_config_sha256':sha(Path(self.c['model_path'])/'config.json')})

    def module(self,site):
        block=self.model.model.layers[site['layer']]
        return block.self_attn if site['module']=='attention' else block.mlp

    def encode(self,vs,instruction=False,openqa=False):
        texts=[render(r,instruction,openqa) for r in vs]
        enc=self.tok(texts,padding=True,return_tensors='pt',return_offsets_mapping=True,add_special_tokens=False)
        offsets=enc.pop('offset_mapping');width=enc.input_ids.shape[1];positions=[]
        for i,r in enumerate(vs):
            prefix=len(INSTRUCTION) if instruction and not openqa else 0
            match=PRONOUN.search(r['context']);assert match
            start,end=prefix+match.start(),prefix+match.end()
            matches=[j for j,(a,b) in enumerate(offsets[i].tolist()) if b>a and a<end and b>start]
            assert matches and max(matches)-min(matches)+1==len(matches)
            # End of a semantic identity span also handles multi-token pronouns.
            positions.append([matches[-1],width-1])
        inputs={k:v.to('cuda') for k,v in enc.items()}
        inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).clamp(min=0)
        return inputs,torch.tensor(positions,device='cuda')

    def edit_hook(self,site,position,operation,strength,mean=None,direction=None,donor=None):
        def hook(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            if h.shape[1]==1 and position.shape[1]>1:return output
            p=position[:,0 if site['position']=='identity' else 1];b=torch.arange(len(p),device=h.device)
            x=h[b,p].float();y=x
            if operation=='patch':y=torch.as_tensor(donor,device=h.device,dtype=torch.float32)
            elif operation=='mean':y=x+strength*(torch.as_tensor(mean,device=h.device)-x)
            elif operation=='project':
                d=torch.as_tensor(direction,device=h.device);y=x-strength*(x@d)[:,None]*d
            elif operation=='scale':y=x*(1-strength)
            elif operation=='none':return output
            else:raise ValueError(operation)
            changed=h.clone();changed[b,p]=y.to(h.dtype)
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        return hook

    def forward(self,vs,spec=None,capture=False,instruction=False):
        self.setup();inputs,pos=self.encode(vs,instruction);handles=[];caps={};probe=[]
        if capture:
            for l,block in enumerate(self.model.model.layers):
                for name,mod in [('attention',block.self_attn),('mlp',block.mlp)]:
                    def cap(module,args,output,l=l,name=name):
                        h=output[0] if isinstance(output,tuple) else output;b=torch.arange(len(vs),device=h.device)
                        caps[(l,name)]=torch.stack([h[b,pos[:,0]],h[b,pos[:,1]]],1).detach().float().cpu().numpy()
                    handles.append(mod.register_forward_hook(cap))
        if spec and spec['operation'] in ('fairsteer','evidence'):
            def fs_hook(module,args,output):
                h=output[0] if isinstance(output,tuple) else output
                x=h[:,-1].detach().float().cpu().numpy();signals=probe_values(x,self.probes)
                use=signals['fs_gate']>.5 if spec['operation']=='fairsteer' else signals['evidence_gate']>.25
                probe.append(use);edited=h.clone();alpha=1 if spec['operation']=='fairsteer' else 1.5
                v=torch.as_tensor(self.fs_vector,device=h.device,dtype=torch.float64)
                edited[:,-1]=(h[:,-1].double()+torch.tensor(use,device=h.device)[:,None]*alpha*v).to(h.dtype)
                return (edited,*output[1:]) if isinstance(output,tuple) else edited
            handles.append(self.model.model.layers[18].register_forward_hook(fs_hook))
        elif spec and spec['operation']!='none':
            site=spec['site'];handles.append(self.module(site).register_forward_hook(self.edit_hook(site,pos,spec['operation'],spec.get('strength',1),spec.get('mean'),spec.get('direction'),spec.get('donor'))))
        try:
            with torch.inference_mode():out=self.model(**inputs,use_cache=False,logits_to_keep=1)
            scores=out.logits[:,-1,self.ids].float().cpu().numpy()
        finally:
            for h in handles:h.remove()
        assert np.isfinite(scores).all()
        act=np.stack([caps[(l,m)] for l in range(len(self.model.model.layers)) for m in ('attention','mlp')],1) if capture else None
        return scores,act,(probe[0] if probe else np.zeros(len(vs),bool))

    def infer(self,key,vs,spec=None,capture=False,instruction=False):
        folder=self.out/'shards'/key;folder.mkdir(parents=True,exist_ok=True);all_scores=[];all_act=[];triggers=[]
        for start in range(0,len(vs),self.c['batch_size']):
            batch=vs[start:start+self.c['batch_size']];path=folder/f'{start:06d}.npz'
            ident=digest([sha(self.out/'plan_frozen.json'),batch,key])
            if path.exists():
                saved=dict(np.load(path));assert str(saved['identity'])==ident
            else:
                local=dict(spec) if spec else None
                if local and 'donor' in local:local['donor']=local['donor'][start:start+len(batch)]
                s,a,t=self.forward(batch,local,capture,instruction);saved={'scores':s,'trigger':t,'identity':ident}
                if a is not None:saved['activations']=a
                np.savez(path,**saved)
            all_scores.append(saved['scores']);triggers.append(saved['trigger'])
            if capture:all_act.append(saved['activations'])
        return np.concatenate(all_scores),np.concatenate(all_act) if capture else None,np.concatenate(triggers)

    def smoke(self):
        if (self.out/'smoke.json').exists():return
        vs=[variants(r) for r in self.values['fit'][:4]];s,a,_=self.forward(vs,capture=True)
        site={'layer':10,'module':'attention','position':'identity'}
        t,_,_=self.forward(vs,{'site':site,'operation':'patch','donor':a[:,20,0]})
        z,_,_=self.forward(vs,{'site':site,'operation':'scale','strength':0})
        r,_,_=self.forward(vs)
        assert np.array_equal(s,t) and np.array_equal(s,z) and np.array_equal(s,r)
        write(self.out/'smoke.json',{'self_patch_exact':True,'zero_strength_exact':True,'hook_restoration_exact':True,
            'activation_shape':list(a.shape),'candidate_ids_distinct':len(set(self.ids))==2})
        self.log('smoke_passed')

    def coarse(self):
        vs=[variants(r) for r in self.values['fit']];base,act,_=self.infer('fit_capture',vs,capture=True)
        np.savez(self.out/'fit_cache.npz',scores=base,activations=act)
        pairs=defaultdict(list)
        for i,r in enumerate(vs):pairs[r['pair_id']].append(i)
        donor=np.zeros(len(vs),int)
        for ids in pairs.values():assert len(ids)==2;donor[ids]=ids[::-1]
        gold=np.array([r['label'] for r in vs]);ok=base.argmax(1)==gold
        discord=np.array([ok[i]!=ok[donor[i]] for i in range(len(vs))]);bad=discord&~ok;good=discord&ok
        assert bad.any(),'No discordant fit pairs: stop mechanism search'
        def margin(s):return s[np.arange(len(vs)),gold]-s[np.arange(len(vs)),1-gold]
        baseline_margin=margin(base);results=[]
        for n,site in enumerate(self.sites):
            mi=site['layer']*2+(site['module']=='mlp');pi=site['position']=='decision'
            d=act[donor,mi,int(pi)];spec={'operation':'patch','site':site,'donor':d}
            s,_,_=self.infer(f'coarse_{n:03d}',vs,spec);delta=margin(s)-baseline_margin
            results.append({'site':site,'bad_rows':int(bad.sum()),'good_rows':int(good.sum()),
                'rescue_margin':float(delta[bad].mean()),'reverse_margin':float(delta[good].mean()),
                'bad_fixed':int((s.argmax(1)[bad]==gold[bad]).sum()),
                'good_broken':int((s.argmax(1)[good]!=gold[good]).sum()),
                'all_metrics':metrics(vs,s)})
            if n%8==0:self.log('coarse_progress',sites=n+1,total=len(self.sites),bad_rows=int(bad.sum()))
        selected=[];layers=set()
        for r in sorted(results,key=lambda r:r['rescue_margin'],reverse=True):
            if r['site']['layer'] not in layers:selected.append(r['site']);layers.add(r['site']['layer'])
            if len(selected)==self.c['top_sites']:break
        random=[]
        for i,s in enumerate(selected):
            possible=[l for l in range(28) if l not in layers]
            l=min(possible,key=lambda l:rank(self.c['seed'],['random_site',i,l]));layers.add(l)
            random.append({**s,'layer':l})
        write(self.out/'localization.json',{'fit_only':True,'oracle_donor':True,'discordant_pairs':int(bad.sum()),
            'baseline':metrics(vs,base),'sites':results,'top_sites':selected,'random_sites':random})
        parameters={}
        for i,site in enumerate(selected+random):
            mi=site['layer']*2+(site['module']=='mlp');pi=int(site['position']=='decision');x=act[:,mi,pi]
            female=np.array([r['gender']=='female' for r in vs]);direction=x[~female].mean(0)-x[female].mean(0)
            direction/=max(np.linalg.norm(direction),1e-10)
            parameters[f'mean_{i}']=x.mean(0);parameters[f'direction_{i}']=direction
        np.savez(self.out/'parameters.npz',**parameters)
        self.log('coarse_complete',top_sites=selected)

    def spec(self,site_index,operation,strength):
        loc=read(self.out/'localization.json');sites=loc['top_sites']+loc['random_sites'];p=dict(np.load(self.out/'parameters.npz'))
        return {'site':sites[site_index],'site_index':site_index,'operation':operation,'strength':strength,
                'mean':p[f'mean_{site_index}'],'direction':p[f'direction_{site_index}']}

    @staticmethod
    def public_spec(spec):return {k:v for k,v in spec.items() if k not in ('mean','direction','donor')}

    def task_values(self,split,task):return [variants(r,task,pos) for r in self.values[split] for pos in range(2)]

    def select(self):
        path=self.out/'selection_frozen.json'
        if path.exists():return read(path)
        candidates=[{'name':'baseline','spec':{'operation':'none'}}]
        for i in range(self.c['top_sites']):
            for op in ('mean','project','scale'):
                for strength in self.c['strengths']:
                    candidates.append({'name':f'site{i}_{op}_{strength:g}','spec':self.public_spec(self.spec(i,op,strength))})
        for i,candidate in enumerate(candidates):
            s=candidate['spec'];spec=s if s['operation']=='none' else self.spec(s['site_index'],s['operation'],s['strength'])
            candidate['tasks']={}
            for task in TASKS:
                vs=self.task_values('dev',task);scores,_,_=self.infer('dev_'+candidate['name']+'_'+task,vs,spec)
                candidate['tasks'][task]=metrics(vs,scores)
            self.log('selection_progress',candidate=i+1,total=len(candidates),name=candidate['name'],
                both_correct=candidate['tasks']['coref']['both_correct'])
        best=feasible_choice(candidates,self.c['selection_tolerance'])
        saved={'created':time.strftime('%Y-%m-%d %H:%M:%S'),'selected':best,'candidates':candidates,
               'parameters_sha256':sha(self.out/'parameters.npz'),'localization_sha256':sha(self.out/'localization.json')}
        write(path,saved);self.log('selection_frozen',selected=best['name']);return saved

    def selected_spec(self):
        s=read(self.out/'selection_frozen.json')['selected']['spec']
        return s if s['operation']=='none' else self.spec(s['site_index'],s['operation'],s['strength'])

    def matched_random_spec(self,random_index,selected):
        loc=read(self.out/'localization.json')
        site={**selected['site'],'layer':loc['random_sites'][random_index]['layer']}
        act=np.load(self.out/'fit_cache.npz')['activations']
        mi=site['layer']*2+(site['module']=='mlp');pi=int(site['position']=='decision');x=act[:,mi,pi]
        female=np.array([r['gender']=='female' for r in self.values['fit']])
        direction=x[~female].mean(0)-x[female].mean(0);direction/=max(np.linalg.norm(direction),1e-10)
        return {'site':site,'operation':selected['operation'],'strength':selected['strength'],
                'mean':x.mean(0),'direction':direction}

    def diagnostics(self):
        path=self.out/'neutral_donor.json'
        if path.exists():return
        loc=read(self.out/'localization.json');vs=[variants(r) for r in self.values['fit']]
        neutral=[{**r,'context':r['context']+' This happened yesterday.'} for r in vs]
        _,act,_=self.infer('fit_neutral_capture',neutral,capture=True);base=np.load(self.out/'fit_cache.npz')['scores'];result=[]
        for i,site in enumerate(loc['top_sites']+loc['random_sites']):
            mi=site['layer']*2+(site['module']=='mlp');pi=int(site['position']=='decision')
            s,_,_=self.infer(f'neutral_patch_{i}',vs,{'operation':'patch','site':site,'donor':act[:,mi,pi]})
            result.append({'site':site,'random_control':i>=self.c['top_sites'],'metrics':metrics(vs,s),
                'prediction_change':float(np.mean(s.argmax(1)!=base.argmax(1)))})
        write(path,{'baseline':metrics(vs,base),'neutral_donor_results':result})

    def evaluate(self):
        selection=read(self.out/'selection_frozen.json');assert sha(self.out/'parameters.npz')==selection['parameters_sha256']
        spec=self.selected_spec();methods={'baseline':None,'selected':spec,'instruction':None,
                                         'fairsteer':{'operation':'fairsteer'},'evidence':{'operation':'evidence'}}
        if spec['operation']!='none':
            for i in range(self.c['top_sites']):methods[f'random_{i}']=self.matched_random_spec(i,spec)
        report={};comparisons={};predictions=[]
        for split in ('confirmation','external'):
            report[split]={};comparisons[split]={}
            tasks=TASKS if split=='confirmation' else ('coref',)
            for task in tasks:
                vs=self.task_values(split,task);task_report={};pred={}
                for name,s in methods.items():
                    scores,_,trigger=self.infer(f'{split}_{task}_{name}',vs,s,instruction=name=='instruction')
                    task_report[name]=metrics(vs,scores);task_report[name]['trigger_rate']=float(trigger.mean());pred[name]=scores.argmax(1)
                    for r,sc,p,t in zip(vs,scores,pred[name],trigger):
                        predictions.append({'split':split,'task':task,'method':name,'uid':r['uid'],
                            'pair_id':r['pair_id'],'family_id':r['family_id'],'permutation':r['permutation'],
                            'label':r['label'],'prediction':int(p),'scores':sc.tolist(),'trigger':bool(t)})
                    self.log('evaluation_progress',split=split,task=task,method=name,metrics=task_report[name])
                report[split][task]=task_report
                comparisons[split][task]={field:comparison(vs,pred['selected'],pred['baseline'],field,self.c['bootstrap_replicates'],self.c['seed'])
                    for field in (('accuracy','both_correct','flip','gap') if task=='coref' else ('accuracy',))}
                if task=='coref':comparisons[split][task]['occupation_cluster_both_correct']=comparison(vs,pred['selected'],pred['baseline'],'both_correct',self.c['bootstrap_replicates'],self.c['seed'],'occupation_pair')
        write(self.out/'results.json',{'selection':selection['selected'],'metrics':report,'comparisons':comparisons})
        (self.out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions),encoding='utf-8')

    def generate(self):
        self.setup();spec=self.selected_spec();result={};all_details=[]
        vs=[variants(r) for r in self.values['confirmation']]
        for name,s in [('baseline',None),('selected',spec)]:
            folder=self.out/'generation'/name;folder.mkdir(parents=True,exist_ok=True);texts=[]
            for start in range(0,len(vs),self.c['batch_size']):
                batch=vs[start:start+self.c['batch_size']];path=folder/f'{start:06d}.json'
                if path.exists():generated=read(path)
                else:
                    inputs,pos=self.encode(batch,openqa=True);handles=[]
                    if s and s['operation']!='none':
                        # Hook once at prefill; generation hooks subsequently removed logically.
                        state={'prefill':True}
                        edit=self.edit_hook(s['site'],pos,s['operation'],s['strength'],s.get('mean'),s.get('direction'))
                        def once(module,args,output):
                            if state['prefill']:state['prefill']=False;return edit(module,args,output)
                            return output
                        handles.append(self.module(s['site']).register_forward_hook(once))
                    try:
                        # Generation creates incremental position ids; fixed prefill ids are unnecessary.
                        inputs.pop('position_ids')
                        with torch.inference_mode():
                            output=self.model.generate(**inputs,max_new_tokens=12,do_sample=False,use_cache=True,
                                pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
                        generated=self.tok.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                    finally:
                        for h in handles:h.remove()
                    write(path,generated)
                texts.extend(generated)
                if start%(self.c['batch_size']*10)==0:self.log('generation_progress',method=name,done=start+len(batch),total=len(vs))
            strict=[];loose=[]
            for r,t in zip(vs,texts):
                clean=re.sub(r'^the\s+','',t.strip().strip(' \t\r\n.,:;!?\"\'').lower())
                p=next((i for i,e in enumerate(r['choices']) if clean==e.lower()),-1)
                hits=[i for i,e in enumerate(r['choices']) if re.search(r'\b'+re.escape(e)+r'\b',t,re.I)]
                q=hits[0] if len(hits)==1 else -1;strict.append(p);loose.append(q)
                all_details.append({'method':name,'uid':r['uid'],'raw':t,'strict_prediction':p,'unique_entity_prediction':q,'label':r['label']})
            result[name]={'strict':metrics(vs,predictions=strict),'unique_entity':metrics(vs,predictions=loose),
                          'strict_unparsed':float(np.mean(np.array(strict)<0))}
        write(self.out/'generation_results.json',result)
        (self.out/'generation_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_details),encoding='utf-8')
        self.log('generation_complete',results=result)

    def run(self):
        self.prepare();self.setup();self.smoke();self.coarse();self.select();self.diagnostics();self.evaluate();self.generate()
        del self.model;self.model=None;gc.collect();torch.cuda.empty_cache()
        write(self.out/'complete.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'gpu_allocated':torch.cuda.memory_allocated(),
            'results_sha256':sha(self.out/'results.json'),'generation_sha256':sha(self.out/'generation_results.json')})
        self.log('complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args();e=Experiment()
    if args.prepare_only:e.prepare()
    else:e.run()


if __name__=='__main__':main()
