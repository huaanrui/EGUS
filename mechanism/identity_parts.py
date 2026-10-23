"""Signed local reprocessing and component-subset interventions, without SAE."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .identity_path import Experiment as PathExperiment, semantic_margin, transmission, array_sha, record_rows
from .identity_combo import SCOPES, task_row
from .identity_causal import variants, render, metrics, comparison
from .closed_transfer import rank, digest
from .fairsteer_author import ROOT,read,rows,write,sha
from .identity_answer_parser import parse_answer_line

CONFIG=ROOT/'configs/experiment_identity_parts_v1.toml'
PROTOCOL=ROOT/'research/2026-10-15_identity_parts_protocol_v1.md'
POSITIONS=(*SCOPES,'evidence','option_b')
COARSE=[{'layer':l,'module':m,'position':p} for p,ls in [('identity',range(8,13)),('decision',range(14,21))]
        for l in ls for m in ('attention','mlp')]
HEADS=[{'layer':17,'head':1,'position':'decision'},{'layer':21,'head':1,'position':'decision'}]
FIELDS=[{'id':f'{kind}{l}_{p}','layer':l,'head':1,'kind':kind,'position':p}
        for l in (17,21) for kind,ps in [('q',('decision',)),('k',('decision','option_b'))] for p in ps] + [
        {'id':'v11_identity','layer':11,'head':13,'kind':'v','position':'identity'},
        {'id':'v13_identity','layer':13,'head':6,'kind':'v','position':'identity'}]
UTILITY=('reading','literal','binding','role_swap','record')


def active(r):
    return {**r,'uid':r['uid']+':active',
            'question':f"Considering the pronoun '{r['pronoun']}' in the sentence, "+r['question'][0].lower()+r['question'][1:]}


def centered_edit(x,mean,indices,amount):
    y=x.clone();ii=torch.as_tensor(indices,device=x.device,dtype=torch.long)
    mu=torch.as_tensor(mean,device=x.device,dtype=torch.float32)
    y[...,ii]=(x[...,ii].float()-amount*(x[...,ii].float()-mu[ii])).to(x.dtype)
    return y


def orientation_edit(x,mean,basis,amount,preserve=True,random=None):
    """Centered subspace edit or exact angle/norm matched random tangent."""
    original=x.float();mu=torch.as_tensor(mean,device=x.device,dtype=torch.float32)
    u=torch.as_tensor(basis,device=x.device,dtype=torch.float32)
    y=original-amount*((original-mu)@u)@u.T
    unchanged=(original-y).norm(dim=-1,keepdim=True)<1e-12
    n=original.norm(dim=-1,keepdim=True);ny=y.norm(dim=-1,keepdim=True)
    if preserve:y=y*(n/ny.clamp(min=1e-12))
    if random is not None:
        learned_distance=(original-y).norm(dim=-1,keepdim=True)
        r=torch.as_tensor(random,device=x.device,dtype=torch.float32).expand_as(original)
        unit=original/n.clamp(min=1e-12)
        tangent=r-(r*unit).sum(-1,keepdim=True)*unit
        tangent=tangent/tangent.norm(dim=-1,keepdim=True).clamp(min=1e-12)
        cosine=((original*y).sum(-1,keepdim=True)/(n*n).clamp(min=1e-12)).clamp(-1,1)
        y=n*(cosine*unit+torch.sqrt((1-cosine*cosine).clamp(min=0))*tangent)
        y=torch.where(learned_distance<1e-6,original,y)
    return torch.where(unchanged,original,y).to(x.dtype)


def paired_indices(vs):
    groups=defaultdict(dict)
    for i,r in enumerate(vs):groups[(r['pair_id'],r['permutation'])][r['gender']]=i
    assert all(set(g)=={'male','female'} for g in groups.values())
    return np.array([g['male'] for g in groups.values()]),np.array([g['female'] for g in groups.values()])


def signed_scores(vs,a,b,ga,gb,sa,sb):
    mu=a.mean(0);t=semantic_margin(vs,sb)-semantic_margin(vs,sa)
    transfer=t[:,None]*(b-a)*(ga+gb)/2
    male,female=paired_indices(vs);d=semantic_margin(vs,sa)[male]-semantic_margin(vs,sa)[female]
    gap=d[:,None]*(ga[male]*(a[male]-mu)-ga[female]*(a[female]-mu))
    return mu,transfer.mean(0),gap.mean(0),transfer,gap


def signed_top(scores,k,positive=True):
    eligible=np.flatnonzero(scores>0 if positive else scores<0)
    return eligible[np.argsort(-scores[eligible] if positive else scores[eligible],kind='stable')[:k]]


def vector_transmission(a,b,r,p):
    t=b.astype(float)-a.astype(float);rr=b.astype(float)-r.astype(float);pp=p.astype(float)-a.astype(float)
    energy=np.mean(np.sum(t*t,axis=-1))
    if energy<1e-8:return {'fidelity':None,'restore_alignment':None,'transplant_alignment':None,'total_rms':float(np.sqrt(energy))}
    return {'fidelity':float(1-np.mean(np.sum((rr-t)**2+(pp-t)**2,axis=-1))/(2*energy)),
        'restore_alignment':float(np.mean(np.sum(rr*t,axis=-1))/energy),
        'transplant_alignment':float(np.mean(np.sum(pp*t,axis=-1))/energy),'total_rms':float(np.sqrt(energy))}


def identity_gap(vs,scores):
    m,f=paired_indices(vs);s=semantic_margin(vs,scores);d=s[m]-s[f]
    return {'rms':float(np.sqrt(np.mean(d*d))),'mean_abs':float(np.mean(abs(d)))}


def candidate_order(c):
    return (c.get('feasible',False),c['metrics']['coref']['both_correct'],c['metrics']['coref']['accuracy'],
            -c['spec'].get('burden',0),-len(c['spec']['edits']),c['spec']['name']=='baseline')


class Experiment(PathExperiment):
    def __init__(self,config=CONFIG):
        super().__init__(config);self.params={};self.meta=None

    def prepare(self):
        files=[Path(__file__),self.config,PROTOCOL,ROOT/'tests/test_identity_parts.py',
               ROOT/'mechanism/identity_path.py',ROOT/'mechanism/identity_combo.py',ROOT/'mechanism/identity_causal.py']
        identity={str(p):sha(p) for p in files};path=self.out/'plan_frozen.json'
        sources={str(self.source_data/f'{k}.jsonl'):sha(self.source_data/f'{k}.jsonl')
                 for k in ('rank','screen','dev','historical','new_type1','winogender')}
        sources[str(self.source/'selection_frozen.json')]=sha(self.source/'selection_frozen.json')
        if path.exists():
            plan=read(path);assert plan['implementation']==identity,'Changed code: use new version'
            assert plan['inputs']==sources
        else:
            write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'implementation':identity,'inputs':sources,
                        'coarse_sites':COARSE,'fields':FIELDS,'status':'all external sets previously inspected; exploratory'})
            for p in files:(self.out/('snapshot_'+p.name)).write_bytes(p.read_bytes())
        self.values={k:rows(self.source_data/f'{k}.jsonl') for k in ('rank','screen','dev','historical','new_type1','winogender')}
        by=defaultdict(list)
        for r in self.values['historical']:by[r['pair_id']].append(r)
        ordered=sorted(by,key=lambda p:rank(self.c['seed'],p))
        for name,n in [('controls',self.c['control_pairs']),('generation',self.c['generation_pairs'])]:
            self.values[name]=sum([by[p] for p in ordered[:n]],[])
            text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in self.values[name]);p=self.data/f'{name}.jsonl'
            if p.exists():assert p.read_text(encoding='utf-8')==text
            else:p.write_text(text,encoding='utf-8')
        if (self.out/'parameters.npz').exists():self.load_parameters()
        self.log('prepared',pairs={k:len(v)//2 for k,v in self.values.items()})

    def spans(self,vs,texts,offsets,width):
        base=super().spans(vs,texts,offsets,width);extra=[]
        for r,text,off in zip(vs,texts,offsets):
            marker='\nB. '
            if marker not in text:extra.append(-1);continue
            start=text.index(marker)+len(marker);end=start+len(r['choices'][1])
            ix=[j for j,(a,b) in enumerate(off.tolist()) if b>a and a<end and b>start]
            assert ix;extra.append(ix[-1])
        return torch.cat([base,torch.tensor(extra,device='cuda')[:,None]],1)

    def task_values(self,split,task):
        vs=[variants(r,permutation=p) for r in self.values[split] for p in (0,1)] if task in ('coref','record') else [task_row(r,task,p) for r in self.values[split] for p in (0,1)]
        if task=='record':return [active(r) for r in record_rows(vs)]
        return [active(r) for r in vs] if task in ('reading','literal','binding') else vs

    def load_parameters(self):
        with np.load(self.out/'parameters.npz') as f:self.params={k:f[k] for k in f.files}
        self.meta=read(self.out/'parameter_meta.json')

    def qkv_unrotated(self,attn,hidden):
        shape=(*hidden.shape[:2],-1,128)
        q=attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1,2)
        k=attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1,2)
        v=attn.v_proj(hidden).view(shape).transpose(1,2)
        return q,k,v

    def qk_output(self,x,positions,kwargs,block,edits):
        q,k,v=self.qkv_unrotated(block.self_attn,kwargs['hidden_states']);q=q.clone();k=k.clone();affected=set()
        ix=torch.arange(len(x),device=x.device)
        for edit in edits:
            field=next(f for f in FIELDS if f['id']==edit['field']);p=positions[:,POSITIONS.index(field['position'])];valid=p>=0
            if not valid.any():continue
            tensor=q if field['kind']=='q' else k;h=field['head'] if field['kind']=='q' else field['head']//2
            z=tensor[ix[valid],h,p[valid]]
            value=orientation_edit(z,self.params[field['id']+'_mean'],self.params[field['id']+'_basis'][:,:edit['rank']],
                  edit['amount'],edit.get('preserve',True),self.params[field['id']+'_random'] if edit.get('random',False) else None)
            tensor[ix[valid],h,p[valid]]=value
            affected.add(field['head']) if field['kind']=='q' else affected.update([2*h,2*h+1])
        if not affected:return x
        q,k=apply_rotary_pos_emb(q,k,*kwargs['position_embeddings']);y=x.clone().reshape(len(x),-1,16,128)
        p=positions[:,1];mask=kwargs.get('attention_mask')
        for h in sorted(affected):
            changed=F.scaled_dot_product_attention(q[:,h:h+1],k[:,h//2:h//2+1],v[:,h//2:h//2+1],
                    attn_mask=mask,dropout_p=0,is_causal=mask is None,scale=block.self_attn.scaling).transpose(1,2)
            y[ix,p,h]=changed[ix,p,0]
        return y.reshape_as(x)

    def unit_edit(self,values,edit,site):
        key=edit['site'];mean=self.params[key+'_mean'];y=values
        if edit['mode']=='raw':return values*(1-edit['amount'])
        groups=edit.get('groups',[(edit['group'],edit['amount'])])
        for group,amount in groups:
            indices=np.arange(values.shape[-1]) if group=='whole' else self.params[key+'_'+group]
            y=centered_edit(y,mean,indices,amount)
        return y

    def forward_parts(self,vs,spec=None,source=False,coarse=False,units=(),fields=False,
                      patch=None,unit_patch=None,gradient_site=None,inputs_pos=None,prefill_only=False,generate=False):
        self.setup();inputs,pos=self.encode(vs) if inputs_pos is None else inputs_pos
        n=len(vs);ix=torch.arange(n,device='cuda');caps={};handles=[];attn_args={};buffer={};state={}
        edits=spec['edits'] if spec else [];source=source or any(e['type']=='source' for e in edits)
        unit_map={s['id']:s for s in (self.meta['unit_sites'] if self.meta else units)}
        def once(key):
            if not prefill_only:return True
            if state.get(key):return False
            state[key]=True;return True
        if source:
            def sh(mod,args,out):
                if not once('source'):return out
                y=out.clone();y[ix,pos[:,0]]*=.5;return y
            handles.append(self.model.model.layers[7].mlp.register_forward_hook(sh))
        for l,block in enumerate(self.model.model.layers):
            qk=[e for e in edits if e['type']=='qk' and next(f for f in FIELDS if f['id']==e['field'])['layer']==l]
            needed=fields and any(f['layer']==l for f in FIELDS)
            if qk or needed:
                def acap(mod,args,kwargs,l=l,block=block):
                    attn_args[l]=kwargs
                    if needed_by_layer[l]:
                        with torch.no_grad():
                            q,k,v=self.qkv_unrotated(block.self_attn,kwargs['hidden_states'])
                            for field in [f for f in FIELDS if f['layer']==l]:
                                pp=pos[:,POSITIONS.index(field['position'])];valid=pp>=0
                                t={'q':q,'k':k,'v':v}[field['kind']];h=field['head'] if field['kind']=='q' else field['head']//2
                                z=t[ix,h,pp.clamp(min=0)].clone();z[~valid]=0;caps['field_'+field['id']]=z.detach().cpu().numpy()
                # Capture bound flags via immutable lookup, not loop-cell state.
                handles.append(block.self_attn.register_forward_pre_hook(acap,with_kwargs=True))
            if qk:
                def qe(mod,args,l=l,block=block,qk=qk):
                    if not once('qk'+str(l)):return args
                    return (self.qk_output(args[0],pos,attn_args[l],block,qk),*args[1:])
                handles.append(block.self_attn.o_proj.register_forward_pre_hook(qe))
            for module_name,module in [('attention',block.self_attn),('mlp',block.mlp)]:
                site_patches=[z for z in (patch or []) if z['site']['layer']==l and z['site']['module']==module_name]
                for j,z in enumerate(site_patches):
                    def op(mod,args,out,z=z,j=j,l=l,module_name=module_name):
                        if not once(f'patch{l}{module_name}{j}'):return out
                        x=out[0] if isinstance(out,tuple) else out;pp=pos[:,POSITIONS.index(z['site']['position'])];valid=pp>=0;y=x.clone()
                        y[ix[valid],pp[valid]]=torch.as_tensor(z['donor'],device=x.device,dtype=x.dtype)[valid]
                        return (y,*out[1:]) if isinstance(out,tuple) else y
                    handles.append(module.register_forward_hook(op))
                if coarse:
                    for si,s in enumerate(COARSE):
                        if s['layer']==l and s['module']==module_name:
                            def oc(mod,args,out,si=si,s=s):
                                x=out[0] if isinstance(out,tuple) else out;pp=pos[:,POSITIONS.index(s['position'])]
                                caps['out'+str(si)]=x[ix,pp].detach().cpu().numpy()
                            handles.append(module.register_forward_hook(oc))
            current=[s for s in unit_map.values() if s['layer']==l]
            if gradient_site is not None and gradient_site['layer']==l and gradient_site['id'] not in unit_map:current.append(gradient_site)
            for site in current:
                mod=block.mlp.down_proj if site['module']=='mlp' else block.self_attn.o_proj
                ue=[e for e in edits if e['type']=='unit' and e['site']==site['id']]
                up=[z for z in (unit_patch or []) if z['site']==site['id']]
                collect=any(s['id']==site['id'] for s in units)
                def unit_hook(mod,args,site=site,ue=ue,up=up,collect=collect):
                    active_edit=once('unit'+site['id'])
                    if prefill_only and not active_edit:return args
                    x=args[0];pp=pos[:,POSITIONS.index(site['position'])];valid=pp>=0
                    shaped=x if site['module']=='mlp' else x.reshape(n,-1,16,128)
                    z=shaped[ix,pp.clamp(min=0)] if site['module']=='mlp' else shaped[ix,pp.clamp(min=0),site['head']]
                    if active_edit and (ue or up):
                        zz=z
                        for edit in ue:zz=self.unit_edit(zz,edit,site)
                        for edit in up:
                            ii=torch.as_tensor(edit['indices'],device=x.device,dtype=torch.long);zz=zz.clone()
                            donor=torch.as_tensor(edit['donor'],device=x.device,dtype=x.dtype)
                            zz[:,ii]=donor[:,ii]
                        y=shaped.clone()
                        if site['module']=='mlp':y[ix[valid],pp[valid]]=zz[valid]
                        else:y[ix[valid],pp[valid],site['head']]=zz[valid]
                        x=y.reshape_as(x)
                    if gradient_site is not None and gradient_site['id']==site['id']:
                        x=x.detach().requires_grad_(True);buffer['tensor']=x;buffer['site']=site
                    if collect:
                        shaped=x if site['module']=='mlp' else x.reshape(n,-1,16,128)
                        z=shaped[ix,pp.clamp(min=0)] if site['module']=='mlp' else shaped[ix,pp.clamp(min=0),site['head']]
                        caps['unit_'+site['id']]=z.detach().cpu().numpy()
                    return (x,*args[1:])
                handles.append(mod.register_forward_pre_hook(unit_hook))
            whole=[e for e in edits if e['type']=='whole_head' and e['layer']==l]
            if whole:
                def wh(mod,args,l=l,whole=whole):
                    if not once('whole'+str(l)):return args
                    y=args[0].clone().reshape(n,-1,16,128)
                    for e in whole:
                        pp=pos[:,POSITIONS.index(e['position'])];valid=pp>=0;y[ix[valid],pp[valid],e['head']]*=1-e['amount']
                    return (y.reshape_as(args[0]),*args[1:])
                handles.append(block.self_attn.o_proj.register_forward_pre_hook(wh))
        needed_by_layer={l:fields and any(f['layer']==l for f in FIELDS) for l in range(28)}
        try:
            if generate:
                with torch.inference_mode():
                    output=self.model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,
                         pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
                texts=self.tok.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                return {'texts':texts}
            context=torch.enable_grad() if gradient_site is not None else torch.inference_mode()
            with context:
                out=self.model(**inputs,use_cache=False,logits_to_keep=1)
                logits=out.logits[:,-1,self.ids];scores=logits.detach().float().cpu().numpy()
                if gradient_site is not None:
                    signs=torch.tensor([1 if r['choices'][0]==sorted(r['choices'])[0] else -1 for r in vs],device='cuda')
                    loss=((logits[:,0].float()-logits[:,1].float())*signs).sum()
                    g=torch.autograd.grad(loss,buffer['tensor'])[0];site=gradient_site;pp=pos[:,POSITIONS.index(site['position'])]
                    if site['module']=='mlp':g=g[ix,pp]
                    else:g=g.reshape(n,-1,16,128)[ix,pp,site['head']]
                    caps['gradient']=g.detach().float().cpu().numpy();assert np.isfinite(caps['gradient']).all()
        finally:
            for h in handles:h.remove()
        assert np.isfinite(scores).all()
        if coarse:caps['outs']=np.stack([caps.pop('out'+str(i)) for i in range(len(COARSE))],1)
        return {'scores':scores,**caps}

    def infer_parts(self,key,vs,spec=None,source=False,coarse=False,units=(),fields=False,
                    patch=None,unit_patch=None,gradient_site=None):
        folder=self.out/'shards'/key;folder.mkdir(parents=True,exist_ok=True);parts=[]
        batch_size=self.c['gradient_batch_size'] if gradient_site is not None else self.c['batch_size']
        for start in range(0,len(vs),batch_size):
            batch=vs[start:start+batch_size];end=start+len(batch)
            pp=[{**z,'donor':z['donor'][start:end]} for z in (patch or [])]
            up=[{**z,'donor':z['donor'][start:end]} for z in (unit_patch or [])]
            signature=digest([sha(self.out/'plan_frozen.json'),batch,spec,source,coarse,units,fields,gradient_site,
                 [(z['site'],array_sha(z['donor'])) for z in pp],[(z['site'],list(map(int,z['indices'])),array_sha(z['donor'])) for z in up]])
            path=folder/f'{start:06d}.npz'
            if path.exists():
                with np.load(path) as f:cache={k:f[k] for k in f.files}
                assert str(cache.pop('identity'))==signature,key
            else:
                cache=self.forward_parts(batch,spec,source,coarse,units,fields,pp,up,gradient_site)
                np.savez(path,identity=signature,**cache)
            parts.append(cache)
        return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def smoke(self):
        path=self.out/'smoke.json'
        if path.exists():return
        vs=[variants(r) for r in self.values['screen'][:8]];a=self.forward_parts(vs,coarse=True,fields=True)
        z=self.forward_parts(vs,patch=[{'site':COARSE[0],'donor':a['outs'][:,0]}])
        assert np.array_equal(z['scores'],a['scores'])
        b=self.forward_parts(vs,source=True,fields=True)
        p=self.forward_parts(vs,source=True,fields=True,patch=[{'site':COARSE[9],'donor':a['outs'][:,9]}])
        assert np.array_equal(p['field_v11_identity'],b['field_v11_identity'])
        assert np.array_equal(self.forward_parts(vs)['scores'],a['scores'])
        site={'id':'smoke','layer':8,'module':'mlp','position':'identity'}
        g=self.forward_parts(vs[:2],units=[site],gradient_site=site)
        assert np.linalg.norm(g['gradient'])>0
        self.meta={'unit_sites':[site]};self.params={'smoke_mean':np.zeros(6144,dtype=np.float32)}
        spec={'edits':[{'type':'unit','site':'smoke','mode':'raw','amount':.5}]}
        c=self.forward_parts(vs,spec)['scores']
        d=self.forward_parts(vs,patch=[{'site':COARSE[1],'donor':a['outs'][:,1]*.5}])['scores']
        linear_error=float(abs(c-d).max());assert linear_error<.04,linear_error
        self.meta=None;self.params={}
        field=FIELDS[0];key=field['id'];self.params={key+'_mean':np.zeros(128,dtype=np.float32),
             key+'_basis':np.eye(128,dtype=np.float32)[:,:4],key+'_random':np.ones(128,dtype=np.float32)}
        qspec={'edits':[{'type':'qk','field':key,'rank':1,'amount':0,'preserve':True}]}
        q=self.forward_parts(vs,qspec)['scores'];qerror=float(abs(q-a['scores']).max());assert qerror<.04,qerror
        self.params={}
        write(path,{'self_patch_exact':True,'hook_restore_exact':True,'inaccessible_destination_exact':True,
              'gradient_finite_nonzero':True,'gradient_shape':list(g['gradient'].shape),
              'full_channel_half_vs_output_half_max_error':linear_error,'QKV_zero_edit_max_error':qerror,
              'numerical_note':'exactly-zero/small single-unit effects are not interpreted strongly'})
        self.log('smoke_complete',**read(path))

    def coarse_scan(self):
        path=self.out/'coarse_scan.json'
        if path.exists():return
        vs=[variants(r) for r in self.values['rank']]
        a=self.infer_parts('coarse_A',vs,coarse=True,fields=True);b=self.infer_parts('coarse_B',vs,source=True,coarse=True,fields=True)
        results=[]
        for i,s in enumerate(COARSE):
            r=self.infer_parts(f'coarse_{i}_restore',vs,source=True,fields=True,patch=[{'site':s,'donor':a['outs'][:,i]}])
            p=self.infer_parts(f'coarse_{i}_transplant',vs,fields=True,patch=[{'site':s,'donor':b['outs'][:,i]}])
            t=transmission(vs,a['scores'],b['scores'],r['scores'],p['scores'])
            destinations={f['id']:vector_transmission(a['field_'+f['id']],b['field_'+f['id']],r['field_'+f['id']],p['field_'+f['id']]) for f in FIELDS}
            results.append({'index':i,'site':s,'transmission':t,'destinations':destinations,
               'signed_alignment':(t['restore_alignment']+t['transplant_alignment'])/2})
            self.log('coarse_site',index=i,site=s,transmission=t)
        selected=[]
        for pos in ('identity','decision'):
            candidate=max([r for r in results if r['site']['module']=='mlp' and r['site']['position']==pos],key=lambda r:abs(r['signed_alignment']))
            selected.append({**candidate['site'],'id':f"mlp{candidate['site']['layer']}_{pos}",'coarse_index':candidate['index']})
        selected += [{**s,'module':'head','id':f"head{s['layer']}_{s['head']}_{s['position']}"} for s in HEADS]
        write(path,{'results':results,'unit_sites':selected,'selection':'largest absolute source-effect alignment per source/terminal MLP, fit only'})

    def fit(self):
        if (self.out/'parameters.npz').exists():self.load_parameters();return
        sites=read(self.out/'coarse_scan.json')['unit_sites'];vs=[variants(r) for r in self.values['rank']]
        a=self.infer_parts('fit_A',vs,units=sites,fields=True);b=self.infer_parts('fit_B',vs,source=True,units=sites,fields=True)
        params={};meta={'unit_sites':sites,'groups':{},'fields':{},'gradient_checks':{}}
        male,female=paired_indices(vs)
        for site in sites:
            key=site['id'];self.log('gradient_site_start',site=site)
            ga=self.infer_parts('grad_'+key+'_A',vs,units=[site],gradient_site=site)
            gb=self.infer_parts('grad_'+key+'_B',vs,source=True,units=[site],gradient_site=site)
            ua=a['unit_'+key].astype(float);ub=b['unit_'+key].astype(float)
            mu,ts,gs,transfer,gap=signed_scores(vs,ua,ub,ga['gradient'],gb['gradient'],a['scores'],b['scores'])
            k=self.c['mlp_budget'] if site['module']=='mlp' else self.c['head_budget']
            groups={'transfer_pos':signed_top(ts,k,True),'transfer_neg':signed_top(ts,k,False),
                    'gap_pos':signed_top(gs,k,True),'gap_neg':signed_top(gs,k,False),
                    'absolute':np.argsort(-abs(gs),kind='stable')[:2*k]}
            rms=np.sqrt(np.mean((ua-mu)**2,0));block=self.model.model.layers[site['layer']]
            weights=block.mlp.down_proj.weight if site['module']=='mlp' else block.self_attn.o_proj.weight[:,site['head']*128:(site['head']+1)*128]
            wn=weights.detach().float().norm(dim=0).cpu().numpy();available=set(range(ua.shape[1]))-set(groups['absolute'].tolist())
            matched=[]
            for j in groups['absolute']:
                if not available:break
                pool=np.array(sorted(available));distance=np.log((rms[pool]+1e-8)/(rms[j]+1e-8))**2+np.log((wn[pool]+1e-8)/(wn[j]+1e-8))**2
                winner=int(pool[np.argmin(distance)]);matched.append(winner);available.remove(winner)
            groups['matched']=np.array(matched,dtype=int)
            params[key+'_mean']=mu.astype(np.float32);params[key+'_transfer_score']=ts;params[key+'_gap_score']=gs
            for name,ii in groups.items():params[key+'_'+name]=ii
            meta['groups'][key]={name:{'n':len(ii),'mean_centered_rms':float(rms[ii].mean()) if len(ii) else 0,
                     'mean_projection_column_norm':float(wn[ii].mean()) if len(ii) else 0,
                     'mean_signed_transfer':float(ts[ii].sum()),'mean_signed_gap':float(gs[ii].sum())} for name,ii in groups.items()}
            # Approximation stability: fixed rank-family blocks, not test tuning.
            folds=np.array([int(digest(r['family_id'])[:8],16)%5 for r in vs])
            stability={}
            for name,ii in [('transfer_pos',groups['transfer_pos']),('transfer_neg',groups['transfer_neg'])]:
                stability[name]=[float(transfer[folds==f][:,ii].sum(1).mean()) for f in range(5) if (folds==f).any()]
            meta['gradient_checks'][key]={'gradient_finite':True,'mean_gradient_norm':float(np.linalg.norm(ga['gradient'],axis=1).mean()),
                'source_attribution_group_family_block_means':stability}
        rng=np.random.default_rng(self.c['seed'])
        for field in [f for f in FIELDS if f['kind'] in ('q','k')]:
            key=field['id'];x=a['field_'+key].astype(float);delta=x[male]-x[female]
            _,s,vt=np.linalg.svd(delta,full_matrices=False)
            params[key+'_mean']=x.mean(0).astype(np.float32);params[key+'_basis']=vt[:4].T.astype(np.float32)
            params[key+'_random']=rng.normal(size=128).astype(np.float32)
            meta['fields'][key]={'field':field,'rank1_energy':float(s[0]**2/np.sum(s*s)),
                                'rank4_energy':float(np.sum(s[:4]**2)/np.sum(s*s))}
        np.savez(self.out/'parameters.npz',**params);write(self.out/'parameter_meta.json',meta)
        self.load_parameters();write(self.out/'parameters_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
             'parameters_sha256':sha(self.out/'parameters.npz'),'metadata_sha256':sha(self.out/'parameter_meta.json')})
        self.log('fit_complete',sites=sites)

    def candidates(self):
        self.load_parameters();result=[]
        def add(name,edits,category,site=None,burden=0):result.append({'name':name,'edits':edits,'category':category,'site':site,'burden':burden})
        add('baseline',[],'baseline');add('source_half',[{'type':'source'}],'reference',burden=.5)
        old=read(self.source/'selection_frozen.json')['heads']
        add('whole_four_heads',[{'type':'whole_head',**s,'amount':.5} for s in old],'diagnostic',burden=2)
        for site in self.meta['unit_sites']:
            key=site['id'];dim=6144 if site['module']=='mlp' else 128
            def unit(name,groups,mode='centered',amount=.5):
                e={'type':'unit','site':key,'mode':mode,'groups':groups,'group':'whole','amount':amount}
                count=sum(len(self.params[key+'_'+g]) if g!='whole' else dim for g,_ in groups)
                add(key+'_'+name,[e],'unit',key,count/dim*max(abs(v) for _,v in groups))
            for amount in (.25,.5):
                unit(f'gap_pos_damp_{amount}',[('gap_pos',amount)])
                unit(f'gap_neg_amplify_{amount}',[('gap_neg',-amount)])
                unit(f'signed_{amount}',[('gap_pos',amount),('gap_neg',-amount)])
            unit('reverse_signed',[('gap_pos',-.5),('gap_neg',.5)])
            for group in ('absolute','matched','whole'):unit(group+'_centered',[ (group,.5) ])
            unit('whole_raw',[('whole',.5)],'raw')
        for key,meta in self.meta['fields'].items():
            for k in (1,4):
                for amount in (.25,.5):
                    add(f'{key}_rank{k}_damp_{amount}',[{'type':'qk','field':key,'rank':k,'amount':amount,'preserve':True}],'qk',key,k/128*amount)
            for name,amount,preserve,random in [('amplify',-.25,True,False),('no_norm',.5,False,False),('random_angle',.5,True,True)]:
                add(key+'_'+name,[{'type':'qk','field':key,'rank':1,'amount':amount,'preserve':preserve,'random':random}],'qk',key,abs(amount)/128)
        assert len({r['name'] for r in result})==len(result)
        return result

    def assess(self,key,split,spec,tasks):
        result={'spec':spec,'metrics':{}}
        for task in tasks:
            vs=self.task_values(split,task);s=self.infer_parts(f'{key}_{task}_{spec["name"]}',vs,spec)['scores']
            result['metrics'][task]=metrics(vs,s);result['metrics'][task]['identity_gap']=identity_gap(vs,s)
            if task!='coref':result['metrics'][task].pop('identity_gap')
        return result

    def feasible(self,c,baseline,tasks):
        return all(c['metrics'][t]['accuracy']>=baseline['metrics'][t]['accuracy']-self.c['selection_tolerance']-1e-12 for t in tasks)

    def screen(self):
        path=self.out/'shortlist.json'
        if path.exists():return
        cs=self.candidates();tasks=('coref','reading');result=[]
        for c in cs:
            r=self.assess('screen','screen',c,tasks);r['feasible']=self.feasible(r,result[0] if result else r,tasks);result.append(r)
            self.log('screen_candidate',name=c['name'],feasible=r['feasible'],both=r['metrics']['coref']['both_correct'],reading=r['metrics']['reading']['accuracy'])
        ranked=sorted(result,key=candidate_order,reverse=True);selected=cs[:3];counts=defaultdict(int)
        for r in ranked:
            s=r['spec']
            if not r['feasible'] or s['category'] not in ('unit','qk') or counts[s['site']]>=2:continue
            selected.append(s);counts[s['site']]+=1
            if len(selected)>=3+self.c['shortlist']:break
        fine=[r for r in ranked if r['feasible'] and r['spec']['category'] in ('unit','qk')]
        source_ids={s['id'] for s in self.meta['unit_sites'] if s['position']=='identity'}
        sources=[r for r in fine if r['spec']['site'] in source_ids];terminals=[r for r in fine if r['spec']['site'] not in source_ids]
        combinations=[]
        if sources and terminals:
            u,v=sources[0]['spec'],terminals[0]['spec'];combinations.append({'name':'combo_source_terminal','edits':u['edits']+v['edits'],
                 'category':'combo','site':'combination','burden':u['burden']+v['burden']})
        if fine:
            u=fine[0]['spec'];combinations.append({'name':'combo_l7_fine','edits':[{'type':'source'}]+u['edits'],
                  'category':'combo','site':'combination','burden':.5+u['burden']})
        write(self.out/'screen_results.json',result);write(path,selected+combinations)
        self.log('screen_complete',candidates=len(cs),shortlist=[s['name'] for s in selected+combinations])

    def select(self):
        path=self.out/'selection_frozen.json'
        if path.exists():return
        tasks=('coref',*UTILITY);result=[]
        for c in read(self.out/'shortlist.json'):
            r=self.assess('dev','dev',c,tasks);r['feasible']=self.feasible(r,result[0] if result else r,tasks);result.append(r)
            self.log('dev_candidate',name=c['name'],feasible=r['feasible'],metrics={t:r['metrics'][t]['accuracy'] for t in tasks},both=r['metrics']['coref']['both_correct'])
        feasible=[r for r in result if r['feasible']];best=max(feasible,key=candidate_order)
        chosen={'overall':best['spec']}
        for cat in ('unit','qk','combo'):
            pool=[r for r in feasible if r['spec']['category']==cat]
            chosen[cat]=max(pool,key=candidate_order)['spec'] if pool else result[0]['spec']
        write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'results':result,'selected':chosen,
              'parameters_sha256':sha(self.out/'parameters.npz'),'historical_exploratory':True})
        self.log('selection_frozen',selected=chosen)

    def signed_diagnostics(self):
        path=self.out/'signed_diagnostics.json'
        if path.exists():return
        result={}
        for split in ('screen','controls'):
            vs=self.task_values(split,'coref');sites=self.meta['unit_sites']
            a=self.infer_parts('signed_'+split+'_A',vs,units=sites);b=self.infer_parts('signed_'+split+'_B',vs,source=True,units=sites)
            scores={'baseline':a['scores'],'source_half':b['scores']};result[split]={}
            for site in sites:
                key=site['id'];result[split][key]={}
                for group in ('transfer_pos','transfer_neg'):
                    ii=self.params[key+'_'+group]
                    r=self.infer_parts(f'signed_{split}_{key}_{group}_restore',vs,source=True,unit_patch=[{'site':key,'indices':ii,'donor':a['unit_'+key]}])['scores']
                    p=self.infer_parts(f'signed_{split}_{key}_{group}_transplant',vs,unit_patch=[{'site':key,'indices':ii,'donor':b['unit_'+key]}])['scores']
                    result[split][key][group]={'n':len(ii),'transmission':transmission(vs,a['scores'],b['scores'],r,p),
                        'restore_metrics':metrics(vs,r),'transplant_metrics':metrics(vs,p)}
                    scores[key+'_'+group+'_restore']=r;scores[key+'_'+group+'_transplant']=p
            self.save_predictions('signed_'+split,vs,scores);self.log('signed_diagnostics',split=split)
        write(path,{'results':result,'sign_definition':'facilitates/opposes source-half score effect, not intrinsically bias-promoting/suppressing'})

    def evaluation_methods(self):
        cs={s['name']:s for s in self.candidates()};selected=read(self.out/'selection_frozen.json')['selected']
        methods={k:cs[k] for k in ('baseline','source_half','whole_four_heads')}
        methods.update({k:s for k,s in selected.items()});return methods

    def evaluate(self):
        path=self.out/'evaluation.json'
        if path.exists():return
        methods=self.evaluation_methods();result={};reused={}
        for split in ('historical','new_type1','winogender'):
            vs=self.task_values(split,'coref');scores={};reused={}
            for name,spec in methods.items():
                signature=digest(spec['edits'])
                if signature not in reused:reused[signature]=self.infer_parts('evaluate_'+split+'_'+spec['name'],vs,spec)['scores']
                scores[name]=reused[signature]
            result[split]={'metrics':{k:{**metrics(vs,z),'identity_gap':identity_gap(vs,z)} for k,z in scores.items()},
              'comparisons':{k:{f:comparison(vs,scores[k].argmax(1),scores['source_half'].argmax(1),f,self.c['bootstrap_replicates'],self.c['seed'])
                  for f in ('accuracy','both_correct','flip')} for k in ('overall','unit','qk','combo')}}
            self.save_predictions(split,vs,scores);self.log('evaluation',split=split,metrics={k:m['both_correct'] for k,m in result[split]['metrics'].items()})
        utility={};chosen={k:methods[k] for k in ('baseline','source_half','overall','whole_four_heads')}
        for task in UTILITY:
            vs=self.task_values('controls',task);scores={};reused={}
            for name,spec in chosen.items():
                signature=digest(spec['edits'])
                if signature not in reused:reused[signature]=self.infer_parts('utility_'+task+'_'+spec['name'],vs,spec)['scores']
                scores[name]=reused[signature]
            utility[task]={k:metrics(vs,z) for k,z in scores.items()};self.save_predictions('utility_'+task,vs,scores)
        from scripts.identity_combo_mechanism_panel import anonymize
        av=[variants(anonymize(r),permutation=p) for r in self.values['controls'] for p in (0,1)]
        ss={name:self.infer_parts('anonymous_'+name,av,spec)['scores'] for name,spec in chosen.items()}
        self.save_predictions('anonymous',av,ss)
        write(path,{'coref':result,'utilities':utility,'anonymous':{k:metrics(av,z) for k,z in ss.items()},
                  'status':'historical exploratory; all selected before target evaluation'})

    def generate(self):
        path=self.out/'chat_results.json'
        if path.exists():return
        self.setup()
        methods=self.evaluation_methods();methods={k:methods[k] for k in ('baseline','source_half','overall','whole_four_heads')}
        vs=[variants(r) for r in self.values['generation']];folder=self.out/'chat';folder.mkdir(exist_ok=True);output=[];coverage={}
        for start in range(0,len(vs),self.c['batch_size']):
            batch=vs[start:start+self.c['batch_size']];p=folder/f'{start:06d}.json'
            if p.exists():output.extend(read(p));continue
            prompts=[self.tok.apply_chat_template([{'role':'user','content':r['context']+' '+r['question']+'\nAnswer with only the occupation name.'}],
                         tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
            enc=self.tok(prompts,padding=True,return_offsets_mapping=True,return_tensors='pt',add_special_tokens=False)
            offsets=enc.pop('offset_mapping');pos=self.spans(batch,prompts,offsets,enc.input_ids.shape[1]);inputs=enc.to('cuda')
            inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).clamp(min=0);part=[];reused={}
            for name,spec in methods.items():
                signature=digest(spec['edits'])
                if signature not in reused:reused[signature]=self.forward_parts(batch,spec,inputs_pos=(inputs,pos),prefill_only=True,generate=True)['texts']
                texts=reused[signature]
                for r,text in zip(batch,texts):
                    clean=re.sub(r'^the\s+','',text.strip().strip(' \t\r\n.,:;!?\"\'').lower())
                    pred=next((i for i,x in enumerate(r['choices']) if clean==x.lower()),-1)
                    part.append({'method':name,'uid':r['uid'],'label':r['label'],'raw':text,'strict_prediction':pred,'line_prediction':parse_answer_line(text,r['choices'])})
            write(p,part);output.extend(part)
            if start%(self.c['batch_size']*4)==0:self.log('chat_progress',done=start+len(batch),total=len(vs))
        lookup={(r['method'],r['uid']):r for r in output};result={}
        for name,spec in methods.items():
            p=[lookup[(name,r['uid'])]['strict_prediction'] for r in vs]
            result[name]={'strict':metrics(vs,predictions=p),'line':metrics(vs,predictions=[lookup[(name,r['uid'])]['line_prediction'] for r in vs]),
                          'unparsed':float(np.mean(np.array(p)<0))}
            coverage[name]=[{'edit':e,'active_on_chat':not(e['type']=='qk' and next(f for f in FIELDS if f['id']==e['field'])['position']=='option_b')} for e in spec['edits']]
        write(path,{'metrics':result,'coverage':coverage,'pairs':len(vs)//2,'status':'historical native chat short-answer; prefill-only'})
        (self.out/'chat_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output),encoding='utf-8')

    def audit(self):
        plan=read(self.out/'plan_frozen.json')
        for p,h in {**plan['implementation'],**plan['inputs']}.items():assert sha(p)==h,p
        selection=read(self.out/'selection_frozen.json');assert sha(self.out/'parameters.npz')==selection['parameters_sha256']
        records=0
        for p in (self.out/'predictions').glob('*.jsonl'):
            for r in rows(p):assert r['prediction']==int(np.argmax(r['scores']));records+=1
        write(self.out/'verification.json',{'records_recomputed':records,'source_inputs_unchanged':True,'implementation_unchanged':True,
                   'parameters_unchanged':True,'historical_exploratory':True})
        self.log('audit_complete',records=records)

    def run(self,stage='all'):
        self.prepare();stages={'smoke':self.smoke,'coarse':self.coarse_scan,'fit':self.fit,'screen':self.screen,'select':self.select,
            'diagnostics':self.signed_diagnostics,'evaluate':self.evaluate,'generate':self.generate,'audit':self.audit}
        if stage=='all':
            for name,fn in stages.items():self.log('stage_start',stage=name);fn()
            write(self.out/'complete.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S')})
        elif stage!='prepare':stages[stage]()
        if self.model is not None:del self.model;self.model=None;gc.collect();torch.cuda.empty_cache()
        self.log('stage_complete',stage=stage)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',default='all',choices=['all','prepare','smoke','coarse','fit','screen','select','diagnostics','evaluate','generate','audit'])
    args=parser.parse_args();Experiment(args.config).run(args.stage)


if __name__=='__main__':main()
