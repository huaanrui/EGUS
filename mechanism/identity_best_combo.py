"""Per-model task accuracy search across MLPs, attention heads and combinations."""
from __future__ import annotations
import argparse,gc,json,time,tomllib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .fairsteer_author import ROOT,read,rows,write,sha
from .closed_transfer import digest,rank
from .identity_replication import Experiment as Base,site_id,extended_metrics,TASKS
from .identity_causal import variants,comparison
from .identity_parts import paired_indices,signed_top
from .identity_combo import SCOPES

CONFIG=ROOT/'configs/experiment_identity_best_combo_v1.toml'
PROTOCOL=ROOT/'research/2026-10-16_identity_best_combo_protocol_v1.md'


def score_order(r):
    m=r['metrics'];s=r['spec']
    return (m['accuracy'],m['both_correct'],-len({x['site'] for x in s['edits']}),
        -sum(abs(x['amount']) for x in s['edits']),s['name']=='baseline',s['name'])


def signature(spec):return digest(sorted(spec['edits'],key=lambda z:(z['site'],z['mode'],z['group'],z['amount'])))


def combine(a,b,sites,max_sites=3):
    def physical(edits):return {(sites[z['site']]['module'],sites[z['site']]['layer'],sites[z['site']]['position'],sites[z['site']].get('head')) for z in edits}
    aa,bb=physical(a['edits']),physical(b['edits'])
    if aa&bb or len(aa|bb)>max_sites:return None
    edits=sorted(a['edits']+b['edits'],key=lambda z:(z['site'],z['group']))
    return {'name':'combo_'+digest(edits)[:16],'edits':edits,'category':'combo'}


def group_scores(values,activation,gradient,scores):
    mu=activation.mean(0);contrib=(activation-mu)*gradient
    sign=np.array([1 if r['choices'][0]==sorted(r['choices'])[0] else -1 for r in values])
    semantic=(scores[:,0]-scores[:,1])*sign
    gold_sign=np.array([1 if r['gold']==sorted(r['choices'])[0] else -1 for r in values])
    accuracy=(contrib*gold_sign[:,None]).mean(0)
    male,female=paired_indices(values);delta=semantic[male]-semantic[female]
    gap=(delta[:,None]*(contrib[male]-contrib[female])).mean(0)
    return mu,accuracy,gap


class Search(Base):
    def __init__(self,name,config=CONFIG):
        self.config=Path(config);self.c=tomllib.loads(self.config.read_text());self.name=name
        self.mc=next(m for m in self.c['models'] if m['name']==name)
        self.root=ROOT/self.c['output_dir'];self.out=self.root/name;self.data=ROOT/self.c['source_data']
        self.ref=ROOT/self.c['reference_dir']/name
        self.out.mkdir(parents=True,exist_ok=True);self.model=None;self.params={};self.sites={};self.values={}

    def prepare(self):
        files=[Path(__file__),self.config,PROTOCOL,ROOT/'mechanism/identity_replication.py',
            ROOT/'mechanism/identity_causal.py',ROOT/'mechanism/identity_parts.py',ROOT/'mechanism/identity_combo.py',
            ROOT/'tests/test_identity_best_combo.py']
        inputs=[self.data/f'{s}.jsonl' for s in ('rank','screen','dev','historical')]
        inputs += [self.ref/f for f in ('selection_frozen.json','norm_controls_frozen.json','parameters.npz','parameter_meta.json')]
        identity={'implementation':{str(p):sha(p) for p in files},'inputs':{str(p):sha(p) for p in inputs},
            'model_config':sha(Path(self.mc['path'])/'config.json')}
        plan=self.out/'plan_frozen.json'
        if plan.exists():
            saved=read(plan);assert all(saved[k]==v for k,v in identity.items()),'Changed frozen search inputs; create a new version'
        else:write(plan,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),**identity,'status':'exploratory, all target families previously observed'})
        self.plan_hash=sha(plan);self.values={s:rows(self.data/f'{s}.jsonl') for s in ('rank','screen','dev','historical')}
        family={s:{r['family_id'] for r in rr} for s,rr in self.values.items()}
        for s in family:
            for t in family:
                if s!=t:assert not family[s]&family[t],(s,t)
        by=defaultdict(list)
        for r in self.values['screen']:by[r['pair_id']].append(r)
        ids=sorted(by,key=lambda p:rank(self.c['seed'],p))[:self.c['short_screen_pairs']]
        self.values['short']=sum([by[p] for p in ids],[])
        self.import_references()
        if (self.out/'parameters.npz').exists():
            with np.load(self.out/'parameters.npz') as z:self.params={k:z[k] for k in z.files}
            self.meta=read(self.out/'parameter_meta.json');self.sites=self.meta['sites']

    def import_references(self):
        meta=read(self.ref/'parameter_meta.json');s=read(self.ref/'selection_frozen.json')
        assert sha(self.ref/'parameters.npz')==s['parameter_sha256'];assert sha(self.ref/'parameter_meta.json')==s['meta_sha256']
        with np.load(self.ref/'parameters.npz') as z:
            for k in z.files:self.params['ref_'+k]=z[k]
        self.sites.update({'ref_'+k:v for k,v in meta['sites'].items()})
        source={k:s['methods'][k] for k in ('baseline','selected','fine_combo','source_half','whole_combo')}
        n=read(self.ref/'norm_controls_frozen.json');source['norm_whole']=n['methods']['fine_norm_whole_combo']
        if self.name=='qwen3':
            source.update({k:s['methods'][k] for k in ('old_combo','old_l7','old_source','old_terminal')})
            source['old_norm_whole']=n['methods']['old_norm_whole_combo']
        self.references={}
        for name,spec in source.items():
            k='baseline' if name=='baseline' else 'ref_'+name
            self.references[k]={'name':k,'category':'reference','edits':[{**e,'site':'ref_'+e['site']} for e in spec['edits']]}

    def capture_gradient(self,vs):
        self.setup();assert len(vs)==1
        inputs,pos=self.encode(vs);heads=self.nheads;dim=self.hdim;handles=[];saved={}
        scopes=[SCOPES.index(p) for p in self.c['scopes']]
        handles.append(self.model.get_input_embeddings().register_forward_hook(lambda m,a,o:o.detach().requires_grad_(True)))
        for layer,block in enumerate(self.layers):
            for kind,projection in [('mlp',block.mlp.down_proj),('attention',block.self_attn.o_proj)]:
                def hook(m,a,key=f'{kind}{layer}'):
                    saved[key]=a[0]
                handles.append(projection.register_forward_pre_hook(hook))
        try:
            with torch.enable_grad():
                output=self.model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,self.ids]
                sign=1 if vs[0]['choices'][0]==sorted(vs[0]['choices'])[0] else -1
                keys=list(saved);gradients=torch.autograd.grad((output[:,0].float()-output[:,1].float()).sum()*sign*1024,[saved[k] for k in keys])
                result={'scores':output.detach().float().cpu().numpy()}
                for key,g in zip(keys,gradients):
                    locations=pos[0,scopes]
                    a=saved[key][0,locations].detach();gg=g[0,locations].detach().float()/1024
                    result[key+'_a']=a.cpu().numpy();result[key+'_g']=gg.cpu().numpy()
                assert all(np.isfinite(x).all() for x in result.values())
                return result
        finally:
            for h in handles:h.remove()

    def smoke(self):
        path=self.out/'smoke.json'
        if path.exists():return
        self.setup();vs=self.task_values('rank','coref')[:1];original=self.forward(vs)['scores']
        result=self.capture_gradient(vs)
        assert np.max(abs(result['scores']-original))<=.125
        checks=[]
        for module,layer,head,p in [('mlp',min(9,len(self.layers)-1),None,'identity'),('head',min(21,len(self.layers)-1),1,'decision')]:
            s={'module':module,'layer':layer,'position':p}
            if head is not None:s['head']=head
            key=site_id(s);self.sites[key]=s
            single=self.forward(vs,capture=[s],gradient=key)
            pi=self.c['scopes'].index(p);stem=('mlp' if module=='mlp' else 'attention')+str(layer)
            aa=result[stem+'_a'][pi];gg=result[stem+'_g'][pi]
            if module=='head':aa=aa.reshape(self.nheads,self.hdim)[head];gg=gg.reshape(self.nheads,self.hdim)[head]
            np.testing.assert_allclose(single[key][0],aa,rtol=.003,atol=.01)
            sg=single['gradient'][0];cos=float(np.dot(sg,gg)/(np.linalg.norm(sg)*np.linalg.norm(gg)+1e-20))
            assert cos>.99,(key,cos)
            self.params[key+'_mean']=aa.astype(np.float32);self.params[key+'_probe']=np.array([int(np.argmax(abs(gg)))])
            zero=self.spec('smoke_zero',[self.edit(key,'probe',0.)])
            assert np.array_equal(self.forward(vs,zero)['scores'],original)
            # Perturb the largest-gradient coordinate directly, avoiding zero centered displacement.
            j=int(np.argmax(abs(gg)));delta=np.zeros_like(aa);step=.05*max(float(np.std(aa)),1.)
            delta[j]=step
            plus=self.forward(vs,patch={'site':key,'indices':[j],'donor':(aa+delta)[None]})['scores']
            minus=self.forward(vs,patch={'site':key,'indices':[j],'donor':(aa-delta)[None]})['scores']
            orient=1 if vs[0]['choices'][0]==sorted(vs[0]['choices'])[0] else -1
            derivative=float(((plus[0,0]-plus[0,1])-(minus[0,0]-minus[0,1]))*orient/(2*step))
            checks.append({'site':key,'gradient_cosine':cos,'autograd':float(gg[j]),'finite_difference':derivative})
            assert derivative*gg[j]>=0 or abs(derivative)<1e-8
            assert np.array_equal(self.forward(vs)['scores'],original)
        # Native o_proj consumes concatenated query heads even with GQA; verify sum of head contributions.
        block=self.layers[min(21,len(self.layers)-1)].self_attn.o_proj;h=torch.randn(1,1,self.nheads*self.hdim,device='cuda',dtype=torch.float16)
        direct=torch.nn.functional.linear(h.float(),block.weight.float(),None)
        parts=sum(torch.nn.functional.linear(h[...,i*self.hdim:(i+1)*self.hdim].float(),block.weight[:,i*self.hdim:(i+1)*self.hdim].float(),None) for i in range(self.nheads))
        torch.testing.assert_close(direct,parts,rtol=.001,atol=.001)
        write(path,{'connected_gradients':checks,'zero_hook_restore_exact':True,'head_projection_equivalence':True,
            'gradient_forward_max_logit_delta':float(np.max(abs(result['scores']-original))),
            'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        self.sites={k:v for k,v in self.sites.items() if k.startswith('ref_')};self.params={k:v for k,v in self.params.items() if k.startswith('ref_')}
        self.log('smoke_complete',peak_gib=torch.cuda.max_memory_allocated()/2**30)

    def fit(self):
        if (self.out/'parameters.npz').exists():return
        self.setup();vs=self.task_values('rank','coref');folder=self.out/'rank_gradients';folder.mkdir(exist_ok=True)
        for i,r in enumerate(vs):
            p=folder/f'{i:04d}.npz';identity=digest([self.plan_hash,r])
            if p.exists():
                with np.load(p) as z:assert str(z['identity'])==identity
                continue
            z=self.capture_gradient([r]);np.savez_compressed(p,identity=identity,**z)
            if (i+1)%32==0:self.log('all_layer_gradients',done=i+1,total=len(vs))
        paths=sorted(folder.glob('*.npz'));assert len(paths)==len(vs)
        ss=np.concatenate([np.load(p)['scores'] for p in paths]);ranked=[];summaries={};kept_data={}
        for layer in range(len(self.layers)):
            for module,stem in [('mlp',f'mlp{layer}'),('head',f'attention{layer}')]:
                aa=[];gg=[]
                for p in paths:
                    with np.load(p) as z:aa.append(z[stem+'_a']);gg.append(z[stem+'_g'])
                aa=np.asarray(aa,dtype=np.float32);gg=np.asarray(gg,dtype=np.float32)
                for pi,position in enumerate(self.c['scopes']):
                    for head in range(self.nheads) if module=='head' else [None]:
                        a=aa[:,pi] if head is None else aa[:,pi].reshape(len(vs),self.nheads,self.hdim)[:,head]
                        g=gg[:,pi] if head is None else gg[:,pi].reshape(len(vs),self.nheads,self.hdim)[:,head]
                        mu,acc,gap=group_scores(vs,a,g,ss);s={'module':module,'layer':layer,'position':position}
                        if head is not None:s['head']=head
                        key=site_id(s);budget=64 if module=='mlp' else 16
                        strength={o:float(np.sort(abs(v))[-budget:].sum()) for o,v in [('accuracy',acc),('gap',gap)]}
                        record={'site':s,'key':key,'rank_strength':strength};ranked.append(record)
                        kept_data[key]=(mu,acc,gap)
                del aa,gg
        chosen=set()
        for module in ('mlp','head'):
            for position in self.c['scopes']:
                pool=[r for r in ranked if r['site']['module']==module and r['site']['position']==position]
                for objective in ('accuracy','gap'):
                    chosen.update(r['key'] for r in sorted(pool,key=lambda z:(-z['rank_strength'][objective],z['key']))[:self.c['sites_per_kind_position_objective']])
        # Existing physical sites are guaranteed candidates, with new rank-fitted groups as well as frozen references.
        for s in list(self.sites.values()):chosen.add(site_id({k:v for k,v in s.items() if k in ('module','layer','position','head')}))
        for key in sorted(chosen):
            if key not in kept_data:continue
            record=next(r for r in ranked if r['key']==key);s=record['site'];self.sites[key]=s
            mu,acc,gap=kept_data[key];self.params[key+'_mean']=mu
            budgets=(64,128) if s['module']=='mlp' else (16,32)
            for objective,score in [('accuracy',acc),('gap',gap)]:
                self.params[key+'_'+objective+'_score']=score
                for k in budgets:
                    for positive in (True,False):
                        group=objective+f'_{k}_'+('pos' if positive else 'neg')
                        self.params[key+'_'+group]=signed_top(score,min(k,len(score)),positive)
            summaries[key]=record
        self.meta={'sites':self.sites,'selected_site_ranking':summaries,'all_site_ranking':ranked,
            'rank_rows':len(vs),'positions':self.c['scopes'],'discovery':'centered gold-margin and squared identity-gap local gradient, not causal mediation'}
        np.savez_compressed(self.out/'parameters.npz',**self.params);write(self.out/'parameter_meta.json',self.meta)
        self.log('fit_complete',sites=len(summaries),all_sites=len(ranked))

    def candidates(self):
        result=list(self.references.values())
        for key in self.meta['selected_site_ranking']:
            s=self.sites[key]
            for objective in ('accuracy','gap'):
                for k in ((64,128) if s['module']=='mlp' else (16,32)):
                    for mode in ('pos','neg','signed'):
                        for amount in self.c['amounts']:
                            edits=[]
                            for group in ('pos','neg') if mode=='signed' else (mode,):
                                label=objective+f'_{k}_'+group
                                if not len(self.params[key+'_'+label]):continue
                                sign=(1 if group=='pos' else -1)*(1 if objective=='gap' else -1)
                                edits.append(self.edit(key,label,sign*amount))
                            if edits:result.append(self.spec(f'{key}_{objective}_{k}_{mode}_{amount}',edits))
            for amount in (.5,1.):
                for mode in ('raw','shrink','amplify'):
                    result.append(self.spec(f'{key}_{mode}_{amount}',[self.edit(key,amount=-amount if mode=='amplify' else amount,mode='raw' if mode=='raw' else 'center')],'whole'))
        return list({signature(s):s for s in reversed(result)}.values())

    def assess_coref(self,split,spec):
        key='search_'+split+'_'+signature(spec)[:20]
        scores=self.infer(key,self.task_values(split,'coref'),spec)['scores']
        return {'spec':spec,'metrics':extended_metrics(self.task_values(split,'coref'),scores)}

    def select(self):
        path=self.out/'selection_frozen.json'
        if path.exists():return
        candidates=self.candidates();short=[]
        for i,s in enumerate(candidates):
            short.append(self.assess_coref('short',s))
            if (i+1)%40==0:self.log('short_screen',done=i+1,total=len(candidates))
        write(self.out/'short_results.json',short)
        ordered=sorted(short,key=score_order,reverse=True);pool={signature(z['spec']):z['spec'] for z in ordered[:self.c['single_shortlist']]}
        for module in ('mlp','head'):
            for p in self.c['scopes']:
                matching=[z for z in ordered if any(self.sites[e['site']]['module']==module and self.sites[e['site']]['position']==p for e in z['spec']['edits'])]
                for z in matching[:2]:pool[signature(z['spec'])]=z['spec']
        for s in self.references.values():pool[signature(s)]=s
        scored={}
        def assess(s):
            key=signature(s)
            if key not in scored:scored[key]=self.assess_coref('screen',s)
            return scored[key]
        for s in pool.values():assess(s)
        for z in sorted(scored.values(),key=score_order,reverse=True)[:12]:
            for scale in self.c['refine_scales']:
                if not z['spec']['edits']:continue
                s={**z['spec'],'name':z['spec']['name']+f'_scale{scale}',
                    'edits':[{**e,'amount':e['amount']*scale} for e in z['spec']['edits']]}
                assess(s)
        ordered=sorted(scored.values(),key=score_order,reverse=True);atoms=[];per_site=defaultdict(int)
        # Six of each type when available, at most two variants for one physical site.
        for module in ('mlp','head'):
            count=0
            for z in ordered:
                edits=z['spec']['edits'];keys={e['site'] for e in edits}
                if len(keys)!=1:continue
                key=next(iter(keys));s=self.sites[key];phys=(s['module'],s['layer'],s['position'],s.get('head'))
                if s['module']!=module or per_site[phys]>=2:continue
                atoms.append(z['spec']);per_site[phys]+=1;count+=1
                if count>=self.c['atom_pool']//2:break
        beam=ordered[:self.c['beam_width']]
        for depth in range(2,self.c['max_sites']+1):
            parents=[r['spec'] for r in beam]+[s for s in self.references.values() if s['edits']]
            for parent in parents:
                for atom in atoms:
                    s=combine(parent,atom,self.sites,depth)
                    if s is not None:assess(s)
            beam=sorted(scored.values(),key=score_order,reverse=True)[:self.c['beam_width']]
            self.log('beam_depth',depth=depth,evaluated=len(scored),best=beam[0]['metrics']['accuracy'])
        # Fine strength adjustment on the best combined interventions.
        for z in beam:
            for scale in (.5,.75,1.25):
                s={**z['spec'],'name':z['spec']['name']+f'_finalscale{scale}',
                    'edits':[{**e,'amount':e['amount']*scale} for e in z['spec']['edits']]}
                if s['edits']:assess(s)
        write(self.out/'screen_results.json',list(scored.values()))
        final={signature(z['spec']):z['spec'] for z in sorted(scored.values(),key=score_order,reverse=True)[:self.c['dev_shortlist']]}
        final.update({signature(s):s for s in self.references.values()})
        dev=[]
        for i,s in enumerate(final.values()):
            z=self.assess_coref('dev',s);z['utilities']={}
            for task in TASKS:
                if task=='coref':continue
                vs=self.task_values('dev',task);scores=self.infer('devutil_'+signature(s)[:20]+'_'+task,vs,s)['scores']
                z['utilities'][task]=extended_metrics(vs,scores)
            dev.append(z);self.log('dev_candidate',done=i+1,total=len(final),name=s['name'],accuracy=z['metrics']['accuracy'])
        base=next(z for z in dev if not z['spec']['edits'])
        for z in dev:z['feasible']=all(z['utilities'][t]['accuracy']>=base['utilities'][t]['accuracy']-self.c['selection_tolerance']-1e-12 for t in z['utilities'])
        best=max(dev,key=score_order);safe=max([z for z in dev if z['feasible']],key=score_order)
        write(self.out/'dev_results.json',dev)
        write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'best_task':best,'best_constrained':safe,
            'references':self.references,'parameter_sha256':sha(self.out/'parameters.npz'),'meta_sha256':sha(self.out/'parameter_meta.json'),
            'candidate_counts':{'short':len(short),'screen':len(scored),'dev':len(dev)},'objective':'dev single accuracy, then both correct; test unused for selection'})
        self.log('selection_frozen',best=best['spec']['name'],dev_accuracy=best['metrics']['accuracy'],feasible=best['feasible'])

    def evaluate(self):
        lock=read(self.root/'all_models_frozen.json')
        for n,h in lock['selections'].items():assert sha(self.root/n/'selection_frozen.json')==h
        selection=read(self.out/'selection_frozen.json');assert sha(self.out/'parameters.npz')==selection['parameter_sha256']
        assert sha(self.out/'parameter_meta.json')==selection['meta_sha256']
        methods={**self.references,'best_task':selection['best_task']['spec'],'best_constrained':selection['best_constrained']['spec']}
        vs=self.task_values('historical','coref');metrics={};predictions=[];cache={}
        for name,s in methods.items():
            sig=signature(s)
            if sig not in cache:cache[sig]=self.infer('test_'+sig[:20],vs,s)['scores']
            z=cache[sig];metrics[name]=extended_metrics(vs,z)
            predictions.extend({**r,'method':name,'prediction':int(v.argmax()),'scores':v.tolist()} for r,v in zip(vs,z))
            self.log('test_method',name=name,accuracy=metrics[name]['accuracy'],both=metrics[name]['both_correct'])
        p=self.out/'test_predictions.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in predictions),encoding='utf-8')
        write(self.out/'evaluation.json',{'metrics':metrics,'methods':methods,'prediction_sha256':sha(p)})
        self.audit_search()

    def audit_search(self):
        plan=read(self.out/'plan_frozen.json')
        for p,h in {**plan['implementation'],**plan['inputs']}.items():assert sha(p)==h,p
        d=read(self.out/'dev_results.json');selection=read(self.out/'selection_frozen.json')
        assert max(d,key=score_order)==selection['best_task']
        assert max([z for z in d if z['feasible']],key=score_order)==selection['best_constrained']
        evaluation=read(self.out/'evaluation.json');by=defaultdict(list)
        for r in rows(self.out/'test_predictions.jsonl'):
            assert int(np.argmax(r['scores']))==r['prediction'];by[r['method']].append(r)
        for method,rr in by.items():
            rec=extended_metrics(rr,np.array([r['scores'] for r in rr]))
            for k in ('accuracy','both_correct','flip','both_wrong'):assert abs(rec[k]-evaluation['metrics'][method][k])<1e-12
        # Known frozen references must reproduce same-batch previous predictions exactly.
        reference=read(self.ref/'evaluation.json')['historical']
        for method,old in [('baseline','baseline'),('ref_selected','selected'),('ref_fine_combo','fine_combo')]+([('ref_old_combo','old_combo')] if self.name=='qwen3' else []):
            for k in ('accuracy','both_correct'):assert abs(evaluation['metrics'][method][k]-reference[old][k])<1e-12,(method,k)
        write(self.out/'verification.json',{'frozen_files_unchanged':True,'selection_replayed':True,
            'metrics_recomputed':True,'previous_reference_metrics_exact':True,'records':sum(map(len,by.values()))})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','train','evaluate'),required=True);p.add_argument('--model',required=True);a=p.parse_args()
    e=Search(a.model);e.prepare()
    try:
        if a.stage=='smoke':e.smoke()
        elif a.stage=='train':e.smoke();e.fit();e.select()
        else:e.evaluate()
    finally:e.close()


if __name__=='__main__':main()
