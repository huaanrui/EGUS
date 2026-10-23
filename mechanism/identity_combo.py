"""Bounded combinations, token/channel controls and causal identity diagnostics."""
from __future__ import annotations
from collections import defaultdict
import gc
import json
from pathlib import Path
import re
import time
import tomllib

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .identity_causal import Experiment as Base,variants,render,metrics,comparison,PRONOUN,RAW,INSTRUCTION,family
from .closed_transfer import canonical,digest,rank,gender
from .fairsteer_author import ROOT,read,rows,write,sha
from .identity_answer_parser import parse_answer_line

CONFIG=ROOT/'configs/experiment_identity_combo_v1.toml'
PROTOCOL=ROOT/'research/2026-10-16_identity_combo_protocol_v1.md'
SCOPES=('identity','decision','subject','object','question_identity')
ALL_TASKS=('coref','reading','binding','role_swap','literal')
SCREEN_TASKS=('coref','literal','binding')


def task_row(r,task,pos=0):
    if task!='literal':return variants(r,task,pos)
    choices=['masculine','feminine'] # replaced below by the actually observed pronoun pair
    equivalents={'he':('he','she'),'she':('he','she'),'him':('him','her'),
                 'her':('him','her'),'his':('his','her')}
    choices=list(equivalents[r['pronoun'].lower()]);choices=choices[pos:]+choices[:pos]
    v=variants(r,'reading',pos)
    return {**v,'question':'Which of these words occurs as a pronoun in the sentence?',
            'choices':choices,'label':choices.index(r['pronoun'].lower()),'gold':r['pronoun'].lower(),
            'task':'literal','uid':r['source_id']+f':literal:p{pos}'}


def edit_output(h,positions,edit,basis=None,donor=None):
    p=positions[:,SCOPES.index(edit['position'])];valid=p>=0
    if not valid.any():return h
    b=torch.arange(len(p),device=h.device)[valid];p=p[valid];x=h[b,p].float();r=edit.get('strength',0)
    op=edit['operation']
    if op=='scale':y=x*(1-r)
    elif op=='patch':y=torch.as_tensor(donor,device=h.device,dtype=torch.float32)[valid]
    elif op in ('subspace','complement'):
        q=torch.as_tensor(basis,device=h.device,dtype=torch.float32);parallel=(x@q)@q.T
        y=x-r*(parallel if op=='subspace' else x-parallel)
    else:raise ValueError(op)
    changed=h.clone();changed[b,p]=y.to(h.dtype);return changed


def edit_channels(h,positions,edit,indices):
    p=positions[:,SCOPES.index(edit['position'])];valid=p>=0
    if not valid.any():return h
    b=torch.arange(len(p),device=h.device)[valid];p=p[valid]
    ii=torch.as_tensor(indices,device=h.device,dtype=torch.long);changed=h.clone()
    changed[b[:,None],p[:,None],ii[None,:]]*=1-edit['strength']
    return changed


def make_candidates():
    def scale(layer,position='identity',strength=.5,module='mlp'):
        return {'layer':layer,'module':module,'position':position,'operation':'scale','strength':strength}
    result=[{'name':'baseline','edits':[],'category':'baseline'}]
    def add(name,edits,category='single'):result.append({'name':name,'edits':edits,'category':category})
    anchor=scale(7)
    for r in (.25,.5,.75):add(f'l7_scale_{r:g}',[scale(7,strength=r)])
    pool=[scale(l) for l in (0,5,6,8,9)]+[scale(21,'decision',module='attention')]
    for e in pool:add(f"single_{e['module']}_{e['layer']}_{e['position']}",[e])
    for p in ('decision','subject','object','question_identity'):add('l7_position_'+p,[scale(7,p)])
    for e in pool:
        for r in (.25,.5):add(f"combo_l7_{e['module']}{e['layer']}_{r:g}",[anchor,{**e,'strength':r}],'combo')
    for p in ('decision','question_identity'):
        for r in (.25,.5):add(f'combo_l7_{p}_{r:g}',[anchor,scale(7,p,r)],'combo')
    for r in (.25,.5):
        add(f'combo_l6_l8_{r:g}',[scale(l,strength=r) for l in (6,8)],'combo')
        add(f'combo_l6_l7_l8_{r:g}',[scale(l,strength=r) for l in (6,7,8)],'combo')
    for k in (32,128,512):
        for r in (.5,1):add(f'neurons_top{k}_{r:g}',[{**anchor,'operation':'neurons','group':f'top{k}','strength':r}],'neurons')
    for r in (.5,1):add(f'neurons_matched128_{r:g}',[{**anchor,'operation':'neurons','group':'matched128','strength':r}],'random_neurons')
    for k in (1,4):
        for op in ('subspace','complement'):add(f'{op}_rank{k}',[{**anchor,'operation':op,'basis':f'basis{k}'}],'subspace')
    assert len(result)==46,len(result)
    return result


def burden(c):
    return sum(e['strength']*(int(e['group'][3:])/6144 if e['operation']=='neurons' and e['group'].startswith('top') else 1) for e in c['edits'])


def evaluate_feasibility(candidate,baseline,tasks,tolerance):
    return all(candidate['tasks'][t]['accuracy']>=baseline['tasks'][t]['accuracy']-tolerance-1e-12 for t in tasks)


def ordering(c):
    return (c.get('feasible',False),c['tasks']['coref']['both_correct'],c['tasks']['coref']['accuracy'],
            -burden(c['spec']),-len(c['spec']['edits']),c['spec']['name']=='baseline')


def fresh_type1(c,known):
    occs=sorted({o.strip() for f in ('female','male') for o in (RAW/f'{f}_occupations.txt').read_text().splitlines() if o.strip()},key=len,reverse=True)
    excluded_templates={family(r['context'],occs) for r in known}
    for typ in (1,2):
        for group in ('pro','anti'):
            for line in (RAW/f'{group}_stereotyped_type{typ}.txt.test').read_text().splitlines():
                s=re.sub(r'^\d+\s+','',line).replace('[','').replace(']','')
                excluded_templates.add(family(s,occs))
    groups=defaultdict(list);excluded=[]
    for group in ('pro','anti'):
        for line in (RAW/f'{group}_stereotyped_type1.txt.dev').read_text().splitlines():
            num=int(line.split()[0]);marked=re.sub(r'^\d+\s+','',line);spans=[z.strip() for z in re.findall(r'\[([^]]+)\]',marked)]
            ps=[z for z in spans if PRONOUN.fullmatch(z)];sentence=marked.replace('[','').replace(']','')
            entities=[o for o in occs if re.search(r'\b'+re.escape(o)+r'\b',sentence,re.I)]
            template=family(sentence,occs)
            if len(ps)!=1 or len(PRONOUN.findall(sentence))!=1 or len(entities)!=2 or template in excluded_templates:
                excluded.append({'group':group,'line':num,'reason':'pronoun/entity structure or historically inspected family'});continue
            es=[z for z in spans if not PRONOUN.fullmatch(z)]
            gold=[o for o in entities if len(es)==1 and re.search(r'\b'+re.escape(o)+r'\b',es[0],re.I)]
            if len(gold)!=1:continue
            r={'source_id':f'winobias:dev:type1:{group}:{num:04d}','context':sentence,'entities':sorted(entities),
               'gold':gold[0],'pronoun':ps[0],'gender':gender(ps[0]),'stereotype':group=='pro',
               'pair_id':'type1dev:'+digest(canonical(sentence))[:20],
               'family_id':'type1template:'+digest(template)[:20],'occupation_pair':'|'.join(sorted(entities))}
            groups[r['pair_id']].append(r)
    complete=[]
    for pid,vs in groups.items():
        if len(vs)==2 and {r['gender'] for r in vs}=={'male','female'} and len({r['gold'] for r in vs})==1:complete.extend(vs)
        else:excluded.append({'pair_id':pid,'reason':'incomplete male/female pair or inconsistent native gold'})
    families=defaultdict(list)
    for r in complete:families[r['family_id']].append(r)
    selected=[]
    for f in sorted(families,key=lambda f:rank(c['seed'],f)):
        if len(selected)//2>=c['new_type1_pairs']:break
        selected.extend(families[f])
    assert len(selected)//2>=c['new_type1_pairs']
    return sorted(selected,key=lambda r:(r['pair_id'],r['gender'])),excluded


class Experiment(Base):
    def __init__(self,config=CONFIG):
        super().__init__(config);self.source=ROOT/self.c['source_dir'];self.source_data=ROOT/self.c['source_data'];self.params=None

    def prepare(self):
        identity={'code':sha(Path(__file__)),'base_code':sha(ROOT/'mechanism/identity_causal.py'),
                  'config':sha(self.config),'protocol':sha(PROTOCOL)}
        path=self.out/'plan_frozen.json'
        if path.exists():
            plan=read(path);assert plan['identity']==identity,'Changed implementation: use new experiment version'
            for n,h in plan['data'].items():assert sha(self.data/n)==h
            self.values={k:rows(self.data/f'{k}.jsonl') for k in plan['counts']};return
        old={k:rows(self.source_data/f'{k}.jsonl') for k in ('fit','dev','confirmation','external')}
        fam=defaultdict(list)
        for r in old['fit']:fam[r['family_id']].append(r)
        rankrows=[];screen=[]
        for f in sorted(fam,key=lambda f:rank(self.c['seed'],f)):
            (rankrows if len(rankrows)//2<self.c['rank_pairs'] else screen).extend(fam[f])
        assert len(screen)//2>=24
        new,excluded=fresh_type1(self.c,sum(old.values(),[]))
        groups={'rank':rankrows,'screen':screen,'dev':old['dev'],'historical':old['confirmation'],'winogender':old['external'],'new_type1':new}
        for k,vs in groups.items():
            self.values[k]=sorted(vs,key=lambda r:(r['pair_id'],r['gender']))
            (self.data/f'{k}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in self.values[k]),encoding='utf-8')
        for a in ('rank','screen','dev','historical','new_type1'):
            for b in ('rank','screen','dev','historical','new_type1'):
                if a!=b:assert not {r['family_id'] for r in self.values[a]}&{r['family_id'] for r in self.values[b]}
        write(self.data/'type1_exclusions.json',excluded)
        inputs=[self.source_data/f'{k}.jsonl' for k in old]+[RAW/f'{g}_stereotyped_type{t}.txt.{s}' for g in ('pro','anti') for t in (1,2) for s in ('dev','test')]
        write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,
            'data':{p.name:sha(p) for p in self.data.glob('*')},'inputs':{str(p):sha(p) for p in inputs},
            'counts':{k:len(v)//2 for k,v in self.values.items()},'candidates':make_candidates(),
            'historical_status':'previously inspected, exploratory retest','new_type1_status':'native labels, contextual/commonsense transfer; not uniformly unambiguous explicit evidence'})
        for name,p in [('code_snapshot.py',Path(__file__)),('protocol_snapshot.md',PROTOCOL),('config_snapshot.toml',self.config)]:
            (self.out/name).write_bytes(p.read_bytes())
        self.log('prepared',counts={k:len(v)//2 for k,v in self.values.items()})

    def spans(self,vs,texts,offsets,width):
        positions=[]
        for r,text,off in zip(vs,texts,offsets):
            start=text.index(r['context']);m=PRONOUN.search(r['context']);assert m
            chars=[(start+m.start(),start+m.end()),None]
            mentions=[]
            for ent in r['entities']:
                found=re.search(r'\b'+re.escape(ent)+r'\b',r['context'],re.I);assert found
                mentions.append((found.start(),found.end()))
            mentions.sort();chars.extend([(start+a,start+b) for a,b in mentions])
            qstart=text.index(r['question'],start+len(r['context']));qm=PRONOUN.search(r['question'])
            chars.append((qstart+qm.start(),qstart+qm.end()) if qm else None)
            ps=[]
            for i,span in enumerate(chars):
                if i==1:ps.append(width-1);continue
                if span is None:ps.append(-1);continue
                a,b=span;matches=[j for j,(x,y) in enumerate(off.tolist()) if y>x and x<b and y>a]
                assert matches and max(matches)-min(matches)+1==len(matches);ps.append(matches[-1])
            positions.append(ps)
        return torch.tensor(positions,device='cuda')

    def encode(self,vs,instruction=False,openqa=False):
        texts=[render(r,instruction,openqa) for r in vs]
        enc=self.tok(texts,padding=True,return_offsets_mapping=True,return_tensors='pt',add_special_tokens=False)
        offsets=enc.pop('offset_mapping');pos=self.spans(vs,texts,offsets,enc.input_ids.shape[1])
        inputs={k:v.to('cuda') for k,v in enc.items()};inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).clamp(min=0)
        return inputs,pos

    def handles(self,positions,spec,prefill_only=False,donors=None):
        handles=[]
        for i,edit in enumerate(spec['edits'] if spec else []):
            state={'prefill':True};site={'layer':edit['layer'],'module':edit['module']};module=self.module(site)
            if edit['operation']=='neurons':
                indices=self.params[edit['group']]
                def pre(mod,args,edit=edit,state=state,indices=indices):
                    if prefill_only and not state['prefill']:return args
                    state['prefill']=False
                    return (edit_channels(args[0],positions,edit,indices),*args[1:])
                handles.append(module.down_proj.register_forward_pre_hook(pre))
            else:
                basis=self.params[edit['basis']] if 'basis' in edit else None;donor=donors[i] if donors is not None else None
                def hook(mod,args,output,edit=edit,state=state,basis=basis,donor=donor):
                    if prefill_only and not state['prefill']:return output
                    state['prefill']=False;h=output[0] if isinstance(output,tuple) else output
                    y=edit_output(h,positions,edit,basis,donor)
                    return (y,*output[1:]) if isinstance(output,tuple) else y
                handles.append(module.register_forward_hook(hook))
        return handles

    def forward(self,vs,spec=None,capture=False,instruction=False,donors=None):
        self.setup();inputs,pos=self.encode(vs,instruction);handles=self.handles(pos,spec,donors=donors);caps={}
        if capture:
            b=torch.arange(len(vs),device='cuda');p=pos[:,0]
            for l,block in enumerate(self.model.model.layers):
                def before(mod,args,l=l):caps[('before',l)]=args[0][b,p].detach().float().cpu().numpy()
                def mlp(mod,args,output,l=l):caps[('mlp',l)]=output[b,p].detach().float().cpu().numpy()
                def after(mod,args,output,l=l):caps[('after',l)]=output[b,p].detach().float().cpu().numpy()
                handles.extend([block.post_attention_layernorm.register_forward_pre_hook(before),block.mlp.register_forward_hook(mlp),block.register_forward_hook(after)])
            def channels(mod,args):caps['channels']=args[0][b,p].detach().float().cpu().numpy()
            handles.append(self.model.model.layers[7].mlp.down_proj.register_forward_pre_hook(channels))
        try:
            with torch.inference_mode():out=self.model(**inputs,use_cache=False,logits_to_keep=1)
            score=out.logits[:,-1,self.ids].float().cpu().numpy()
        finally:
            for h in handles:h.remove()
        assert np.isfinite(score).all()
        captured={k:np.stack([caps[(k,l)] for l in range(28)],1) for k in ('before','mlp','after')} if capture else {}
        if capture:captured['channels']=caps['channels']
        return score,captured

    def infer(self,key,vs,spec=None,capture=False,instruction=False,donors=None):
        folder=self.out/'shards'/key;folder.mkdir(parents=True,exist_ok=True);parts=[]
        public={'edits':spec['edits']} if spec else {'edits':[]}
        for start in range(0,len(vs),self.c['batch_size']):
            batch=vs[start:start+self.c['batch_size']];path=folder/f'{start:06d}.npz'
            identity=digest([sha(self.out/'plan_frozen.json'),key,batch,public,capture,instruction])
            if path.exists():
                s=dict(np.load(path));assert str(s['identity'])==identity
            else:
                d=[x[start:start+len(batch)] for x in donors] if donors is not None else None
                scores,caps=self.forward(batch,spec,capture,instruction,d);s={'scores':scores,**caps,'identity':identity}
                np.savez(path,**s)
            parts.append(s)
        return {k:np.concatenate([s[k] for s in parts]) for k in parts[0] if k!='identity'}

    def task_values(self,split,task):return [task_row(r,task,p) for r in self.values[split] for p in (0,1)]

    def parameters(self):
        path=self.out/'parameters.npz'
        if path.exists():self.params=dict(np.load(path));return
        vs=[variants(r) for r in self.values['rank']]
        raw=self.infer('rank_capture_raw',vs,capture=True);instr=self.infer('rank_capture_instruction',vs,capture=True,instruction=True)
        for name,cache in [('raw',raw),('instruction',instr)]:np.savez(self.out/f'rank_{name}.npz',**cache)
        grouped=defaultdict(dict)
        for i,r in enumerate(vs):grouped[r['pair_id']][r['gender']]=i
        male=np.array([x['male'] for x in grouped.values()]);female=np.array([x['female'] for x in grouped.values()])
        z=raw['channels'];weight=self.model.model.layers[7].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
        score=np.abs(z[male]-z[female]).mean(0)*weight;order=np.argsort(-score,kind='stable')
        rms=np.sqrt(np.mean(z*z,0));features=np.stack([np.log(np.maximum(rms,1e-8)),np.log(np.maximum(weight,1e-8))],1)
        features=(features-features.mean(0))/np.maximum(features.std(0),1e-8)
        selected=set(order[:128].tolist());available=np.ones(len(z[0]),bool);available[list(selected)]=False
        rng=np.random.default_rng(self.c['seed']);matched=[]
        for unit in order[:128]:
            distance=((features-features[unit])**2).sum(1);distance[~available]=np.inf
            near=np.argsort(distance)[:8];j=int(rng.choice(near));matched.append(j);available[j]=False
        delta=raw['mlp'][male,7]-raw['mlp'][female,7]
        with threadpool_limits(limits=self.c['cpu_threads']):_,singular,vt=np.linalg.svd(delta,full_matrices=False)
        self.params={'top32':order[:32],'top128':order[:128],'top512':order[:512],'matched128':np.array(matched),
                     'basis1':vt[:1].T.copy(),'basis4':vt[:4].T.copy(),'neuron_scores':score,'neuron_rms':rms,'weight_norm':weight}
        np.savez(path,**self.params)
        trace={}
        for name,c in [('raw',raw),('instruction',instr)]:
            items=[]
            for l in range(28):
                a=c['before'][male,l]-c['before'][female,l];m=c['mlp'][male,l]-c['mlp'][female,l]
                b=c['after'][male,l]-c['after'][female,l]
                cosine=np.sum(a*m,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(m,axis=1),1e-8)
                items.append({'layer':l,'pre_gender_difference':float(np.linalg.norm(a,axis=1).mean()),
                              'mlp_gender_difference':float(np.linalg.norm(m,axis=1).mean()),
                              'post_gender_difference':float(np.linalg.norm(b,axis=1).mean()),
                              'cosine_existing_vs_mlp_difference':float(cosine.mean()),
                              'half_scaled_difference':float(np.linalg.norm(a+.5*m,axis=1).mean())})
            trace[name]=items
        write(self.out/'mechanism_trace.json',{'rank_only':True,'identity_difference_not_bias_semantics':True,'trace':trace,
            'subspace_energy_rank1':float(singular[:1]@singular[:1]/(singular@singular)),
            'subspace_energy_rank4':float(singular[:4]@singular[:4]/(singular@singular)),
            'channel_score':'paired absolute gender-difference times down-projection column norm; not a bias-specific attribution',
            'top128_rms_mean':float(rms[order[:128]].mean()),'matched128_rms_mean':float(rms[matched].mean()),
            'top128_weight_mean':float(weight[order[:128]].mean()),'matched128_weight_mean':float(weight[matched].mean())})
        write(self.out/'candidates_frozen.json',{'specs':make_candidates(),'parameters_sha256':sha(path)})
        self.log('parameters_ready',rank_pairs=len(vs)//2,candidates=len(make_candidates()))

    def smoke(self):
        path=self.out/'smoke.json'
        if path.exists():return
        vs=[variants(r) for r in self.values['rank'][:4]];s,_=self.forward(vs)
        zero={'edits':[{'layer':l,'module':'mlp','position':p,'operation':'scale','strength':0} for l,p in [(7,'identity'),(7,'decision'),(8,'identity')]]}
        z,_=self.forward(vs,zero);assert np.array_equal(s,z)
        full={'edits':[{'layer':7,'module':'mlp','position':'identity','operation':'scale','strength':.5}]}
        scaled,_=self.forward(vs,full);restored,_=self.forward(vs);assert np.array_equal(s,restored)
        self.params['all_channels']=np.arange(6144)
        neuron={'edits':[{**full['edits'][0],'operation':'neurons','group':'all_channels'}]}
        ns,_=self.forward(vs,neuron)
        # Half scaling is exact under FP16 powers-of-two except final accumulation rounding.
        assert np.max(np.abs(ns-scaled))<.05
        write(path,{'multi_zero_exact':True,'hook_restore_exact':True,'full_channel_vs_output_max_delta':float(abs(ns-scaled).max())})
        self.params.pop('all_channels');self.log('smoke_passed')

    def assess(self,stage,split,spec,tasks,instruction=False):
        output={'spec':spec,'tasks':{}}
        for task in tasks:
            vs=self.task_values(split,task)
            cache=self.infer(f'{stage}_{spec["name"]}_{task}'+('_instruction' if instruction else ''),vs,spec,instruction=instruction)
            output['tasks'][task]=metrics(vs,cache['scores'])
        return output

    def screen(self):
        path=self.out/'screening.json'
        if path.exists():return read(path)
        candidates=read(self.out/'candidates_frozen.json');assert candidates['parameters_sha256']==sha(self.out/'parameters.npz')
        records=[]
        for i,spec in enumerate(candidates['specs']):
            rec=self.assess('screen','screen',spec,SCREEN_TASKS)
            rec['feasible']=evaluate_feasibility(rec,records[0] if records else rec,SCREEN_TASKS,self.c['selection_tolerance'])
            records.append(rec)
            if i%4==0:self.log('screen_progress',done=i+1,total=len(candidates['specs']),name=spec['name'],both_correct=rec['tasks']['coref']['both_correct'])
        mandatory=[r for r in records if r['spec']['name'] in ('baseline','l7_scale_0.5')]
        combos=sorted([r for r in records if r['spec']['category']=='combo'],key=ordering,reverse=True)[:self.c['shortlist_combos']]
        others=sorted([r for r in records if r not in mandatory and r['spec']['category']!='combo'],key=ordering,reverse=True)[:self.c['shortlist_others']]
        shortlist=[r['spec'] for r in mandatory+combos+others]
        write(path,{'records':records,'shortlist':shortlist});self.log('screen_complete',shortlist=[s['name'] for s in shortlist])
        return read(path)

    def select(self):
        path=self.out/'selection_frozen.json'
        if path.exists():return read(path)
        screen=self.screen();records=[]
        for i,spec in enumerate(screen['shortlist']):
            rec=self.assess('dev','dev',spec,ALL_TASKS)
            rec['feasible']=evaluate_feasibility(rec,records[0] if records else rec,ALL_TASKS,self.c['selection_tolerance'])
            records.append(rec);self.log('selection_progress',done=i+1,total=len(screen['shortlist']),name=spec['name'],
                feasible=rec['feasible'],both_correct=rec['tasks']['coref']['both_correct'])
        feasible=[r for r in records if r['feasible']]
        singles=[r for r in feasible if r['spec']['category']!='combo'];combos=[r for r in feasible if r['spec']['category']=='combo']
        selected={'overall':max(feasible,key=ordering),'single':max(singles,key=ordering),
                  'combo':max(combos,key=ordering) if combos else None}
        write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'selected':selected,'records':records,
            'parameters_sha256':sha(self.out/'parameters.npz'),'screen_sha256':sha(self.out/'screening.json')})
        self.log('selection_frozen',names={k:r['spec']['name'] if r else None for k,r in selected.items()})
        return read(path)

    def diagnostics(self):
        path=self.out/'neutral_diagnostics.json'
        if path.exists():return
        vs=[variants(r) for r in self.values['screen']]
        raw=self.infer('neutral_screen_capture',vs,capture=True)
        neutral=[{**r,'context':'Yesterday, '+r['context'][0].lower()+r['context'][1:]} for r in vs]
        pref=self.infer('neutral_prefix_capture',neutral,capture=True)
        groups=defaultdict(list)
        for i,r in enumerate(vs):groups[r['pair_id']].append(i)
        swap=np.zeros(len(vs),int)
        for ids in groups.values():assert len(ids)==2;swap[ids]=ids[::-1]
        x=raw['mlp'][:,7];counter=raw['mlp'][swap,7];other=pref['mlp'][:,7]
        gd=counter-x;nd=other-x;gn=np.linalg.norm(gd,axis=1);nn=np.linalg.norm(nd,axis=1)
        matched=x+nd*(gn/np.maximum(nn,1e-8))[:,None]
        labels=np.array([r['label'] for r in vs]);base=raw['scores'];ok=base.argmax(1)==labels
        discord=ok!=ok[swap];bad=discord&~ok;good=discord&ok
        margins=lambda s:s[np.arange(len(vs)),labels]-s[np.arange(len(vs)),1-labels]
        result={}
        for name,donor in [('gender_counterpart',counter),('neutral_prefix',other),('neutral_norm_matched',matched)]:
            spec={'name':name,'edits':[{'layer':7,'module':'mlp','position':'identity','operation':'patch'}]}
            cache=self.infer('diagnostic_'+name,vs,spec,donors=[donor]);score=cache['scores'];pred=score.argmax(1);dm=margins(score)-margins(base)
            result[name]={'metrics':metrics(vs,score),'bad_fixed':int((pred[bad]==labels[bad]).sum()),
                'good_broken':int((pred[good]!=labels[good]).sum()),'bad_margin_change':float(dm[bad].mean()) if bad.any() else None,
                'all_mean_abs_margin_change':float(np.abs(dm).mean()),'edit_norm':float(np.linalg.norm(donor-x,axis=1).mean())}
        write(path,{'screen_only':True,'discordant_pairs':int(bad.sum()),'suffix_control_was_causally_inaccessible':True,
                    'baseline':metrics(vs,base),'results':result,'zero_neutral_norm_rows':int((nn<1e-8).sum())})
        self.log('neutral_diagnostics_complete',discordant_pairs=int(bad.sum()))

    def methods(self):
        selected=read(self.out/'selection_frozen.json')['selected'];candidates={s['name']:s for s in make_candidates()}
        result={'baseline':(candidates['baseline'],False),'old_l7':(candidates['l7_scale_0.5'],False),
                'best_single':(selected['single']['spec'],False),'instruction':(candidates['baseline'],True)}
        if selected['combo']:
            combo=selected['combo']['spec'];result['best_combo']=(combo,False);result['combo_instruction']=(combo,True)
            rng=np.random.default_rng(self.c['seed']);mapping=rng.permutation(28).tolist()
            random={'name':'random_combo','category':'control','edits':[{**e,'layer':mapping[e['layer']]} for e in combo['edits']]}
            result['random_combo']=(random,False)
        return result

    def evaluate(self):
        selection=read(self.out/'selection_frozen.json');assert selection['parameters_sha256']==sha(self.out/'parameters.npz')
        methods=self.methods();results={};comparisons={};predictions=[]
        for split in ('historical','new_type1','winogender'):
            tasks=ALL_TASKS if split=='historical' else ('coref',)
            results[split]={};comparisons[split]={}
            for task in tasks:
                vs=self.task_values(split,task);output={};pred={};reused={}
                for name,(spec,instruction) in methods.items():
                    sig=digest([spec['edits'],instruction])
                    if sig in reused:scores=reused[sig]
                    else:
                        cache=self.infer(f'eval_{split}_{task}_{name}',vs,spec,instruction=instruction);scores=cache['scores'];reused[sig]=scores
                    output[name]=metrics(vs,scores);pred[name]=scores.argmax(1)
                    for r,sc,p in zip(vs,scores,pred[name]):
                        predictions.append({'split':split,'task':task,'method':name,'uid':r['uid'],'label':r['label'],
                            'pair_id':r['pair_id'],'family_id':r['family_id'],'permutation':r['permutation'],
                            'prediction':int(p),'scores':sc.tolist()})
                    self.log('evaluation_progress',split=split,task=task,method=name,accuracy=output[name]['accuracy'],both_correct=output[name]['both_correct'])
                results[split][task]=output;comparisons[split][task]={}
                targets=['best_single']+(['best_combo'] if 'best_combo' in pred else [])
                for method in targets:
                    comparisons[split][task][method]={field:comparison(vs,pred[method],pred['old_l7'],field,self.c['bootstrap_replicates'],self.c['seed'])
                        for field in (('accuracy','both_correct','flip','gap') if task=='coref' else ('accuracy',))}
                    if task=='coref':comparisons[split][task][method]['occupation_both_correct']=comparison(vs,pred[method],pred['old_l7'],'both_correct',self.c['bootstrap_replicates'],self.c['seed'],'occupation_pair')
        write(self.out/'results.json',{'selection':selection['selected'],'metrics':results,'comparisons_vs_old_l7':comparisons})
        (self.out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions),encoding='utf-8')
        if 'best_combo' in methods:
            combo=methods['best_combo'][0];vs=self.task_values('historical','coref')
            base=self.infer('interaction_base',vs,methods['baseline'][0])['scores']
            joint=self.infer('interaction_joint',vs,combo)['scores'];singles=[];single_metrics=[]
            for i,edit in enumerate(combo['edits']):
                single={'name':f'constituent_{i}','edits':[edit]}
                s=self.infer('interaction_'+single['name'],vs,single)['scores'];singles.append(s);single_metrics.append(metrics(vs,s))
            label=np.array([r['label'] for r in vs]);margin=lambda s:s[np.arange(len(vs)),label]-s[np.arange(len(vs)),1-label]
            interaction=margin(joint)-margin(base)-sum((margin(s)-margin(base) for s in singles))
            both=metrics(vs,joint)['both_correct']-metrics(vs,base)['both_correct']-sum(metrics(vs,s)['both_correct']-metrics(vs,base)['both_correct'] for s in singles)
            write(self.out/'interaction.json',{'status':'descriptive; final score interaction does not establish a circuit edge',
                'combined':metrics(vs,joint),'baseline':metrics(vs,base),'constituents':single_metrics,
                'mean_margin_interaction':float(interaction.mean()),'both_correct_interaction':float(both)})

    def generate(self):
        path=self.out/'chat_results.json'
        if path.exists():return
        methods=self.methods();methods.pop('random_combo',None)
        vs=[variants(r) for r in self.values['historical']];results={};preds={};details=[];reused={}
        for name,(spec,instruction) in methods.items():
            sig=digest([spec['edits'],instruction]);folder=self.out/'chat'/name;folder.mkdir(parents=True,exist_ok=True)
            if sig in reused:generated=reused[sig]
            else:
                generated=[]
                for start in range(0,len(vs),self.c['batch_size']):
                    batch=vs[start:start+self.c['batch_size']];p=folder/f'{start:06d}.json'
                    if p.exists():texts=read(p)
                    else:
                        prompts=[]
                        for r in batch:
                            prefix=INSTRUCTION.replace('Output only the answer letter.','Output only the occupation name.') if instruction else ''
                            content=prefix+r['context']+' '+r['question']+'\nAnswer with only the occupation name.'
                            prompts.append(self.tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
                        enc=self.tok(prompts,padding=True,return_offsets_mapping=True,return_tensors='pt',add_special_tokens=False)
                        offsets=enc.pop('offset_mapping');pos=self.spans(batch,prompts,offsets,enc.input_ids.shape[1]);inputs=enc.to('cuda')
                        handles=self.handles(pos,spec,prefill_only=True)
                        try:
                            with torch.inference_mode():out=self.model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,
                                pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
                            texts=self.tok.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
                        finally:
                            for h in handles:h.remove()
                        write(p,texts)
                    generated.extend(texts)
                    if start%(self.c['batch_size']*20)==0:self.log('generation_progress',method=name,done=start+len(batch),total=len(vs))
                reused[sig]=generated
            strict=[];line=[]
            for r,text in zip(vs,generated):
                clean=re.sub(r'^the\s+','',text.strip().strip(' \t\r\n.,:;!?\"\'').lower())
                p=next((i for i,x in enumerate(r['choices']) if clean==x.lower()),-1)
                q=parse_answer_line(text,r['choices']);strict.append(p);line.append(q)
                details.append({'method':name,'uid':r['uid'],'raw':text,'label':r['label'],'strict_prediction':p,'line_prediction':q})
            preds[name]={'strict':strict,'line':line};results[name]={'strict':metrics(vs,predictions=strict),'line':metrics(vs,predictions=line),
                'unparsed':float(np.mean(np.array(strict)<0))}
        comps={}
        for method in ('best_single','best_combo'):
            if method in preds:comps[method]={field:comparison(vs,preds[method]['strict'],preds['old_l7']['strict'],field,self.c['bootstrap_replicates'],self.c['seed']) for field in ('accuracy','both_correct','flip')}
        write(path,{'status':'historical exploratory native chat transfer','metrics':results,'comparisons_vs_old_l7':comps})
        (self.out/'chat_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details),encoding='utf-8')
        self.log('generation_complete')

    def verify(self):
        saved=rows(self.out/'predictions.jsonl');by=defaultdict(dict)
        for r in saved:by[(r['split'],r['task'],r['method'])][r['uid']]=r
        report=read(self.out/'results.json');checked=0
        for (split,task,method),p in by.items():
            vs=self.task_values(split,task);s=np.array([p[r['uid']]['scores'] for r in vs]);m=metrics(vs,s)
            for field in ('accuracy','both_correct','flip','gap'):assert abs(m[field]-report['metrics'][split][task][method][field])<1e-12
            checked+=len(vs)
        for file,h in read(self.out/'plan_frozen.json')['inputs'].items():assert sha(Path(file))==h
        assert sha(self.out/'parameters.npz')==read(self.out/'selection_frozen.json')['parameters_sha256']
        write(self.out/'verification.json',{'prediction_records_recomputed':checked,'metrics_match':True,'source_inputs_unchanged':True,
            'historical_not_fresh':True,'new_type1_not_used_for_selection':True})
        self.log('verification_complete',records=checked)

    def run(self):
        self.prepare();self.setup();self.parameters();self.smoke();self.screen();self.select();self.diagnostics();self.evaluate();self.generate();self.verify()
        del self.model;self.model=None;gc.collect();torch.cuda.empty_cache()
        write(self.out/'complete.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'results_sha256':sha(self.out/'results.json')})
        self.log('complete')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args();e=Experiment()
    if args.prepare_only:e.prepare()
    else:e.run()


if __name__=='__main__':main()
