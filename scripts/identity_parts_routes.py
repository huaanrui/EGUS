from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import gc
import time
import numpy as np
import torch
from mechanism.identity_parts import Experiment,POSITIONS,FIELDS,vector_transmission
from mechanism.identity_path import transmission,array_sha
from mechanism.identity_causal import variants,metrics
from mechanism.fairsteer_author import ROOT,read,write,sha
from mechanism.closed_transfer import digest

OPTION_SITES=[{'layer':l,'module':m,'position':'option_b'} for l in range(14,21) for m in ('attention','mlp')]


def run_batch(e,vs,layer,source=False,capture=False,patch=None,head_donor=None,heads=()):
    inputs,pos=e.encode(vs);ix=torch.arange(len(vs),device='cuda');caps={};handles=[]
    if capture:
        for i,s in enumerate(OPTION_SITES):
            module=e.model.model.layers[s['layer']].self_attn if s['module']=='attention' else e.model.model.layers[s['layer']].mlp
            def cap(mod,args,out,i=i):
                x=out[0] if isinstance(out,tuple) else out;caps['option'+str(i)]=x[ix,pos[:,6]].detach().cpu().numpy()
            handles.append(module.register_forward_hook(cap))
        def hc(mod,args):caps['heads']=args[0].reshape(len(vs),-1,16,128)[ix,pos[:,1]].detach().cpu().numpy()
        handles.append(e.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hc))
    if head_donor is not None:
        def hp(mod,args):
            y=args[0].clone().reshape(len(vs),-1,16,128)
            donor=torch.as_tensor(head_donor,device='cuda',dtype=y.dtype)
            for h in heads:y[ix,pos[:,1],h]=donor[:,h]
            return (y.reshape_as(args[0]),*args[1:])
        handles.append(e.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hp))
    try:cache=e.forward_parts(vs,source=source,fields=True,patch=patch,inputs_pos=(inputs,pos))
    finally:
        for h in handles:h.remove()
    if capture:caps['options']=np.stack([caps.pop('option'+str(i)) for i in range(len(OPTION_SITES))],1)
    return {**cache,**caps}


def infer(e,key,vs,layer,source=False,capture=False,patch=None,head_donor=None,heads=()):
    folder=e.out/'route_shards'/key;folder.mkdir(parents=True,exist_ok=True);parts=[]
    for start in range(0,len(vs),e.c['batch_size']):
        batch=vs[start:start+e.c['batch_size']];end=start+len(batch);p=folder/f'{start:06d}.npz'
        pp=[{**z,'donor':z['donor'][start:end]} for z in (patch or [])];hd=head_donor[start:end] if head_donor is not None else None
        identity=digest([sha(e.out/'routes_declaration.json'),batch,layer,source,capture,heads,
              [(z['site'],array_sha(z['donor'])) for z in pp],array_sha(hd) if hd is not None else None])
        if p.exists():
            with np.load(p) as f:c={k:f[k] for k in f.files}
            assert str(c.pop('identity'))==identity
        else:c=run_batch(e,batch,layer,source,capture,pp,hd,heads);np.savez(p,identity=identity,**c)
        parts.append(c)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    e=Experiment();e.prepare();e.setup();protocol=ROOT/'research/2026-10-15_identity_parts_routes_protocol.md'
    identity={'script':sha(Path(__file__)),'protocol':sha(protocol),'main_plan':sha(e.out/'plan_frozen.json')}
    declaration=e.out/'routes_declaration.json'
    if declaration.exists():assert read(declaration)['identity']==identity
    else:write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,'status':'rank-only choice; no deployment selection; all groups reported'})
    coarse=read(e.out/'coarse_scan.json')['results']
    negative=[r for r in coarse if r['site']['module']=='attention' and r['site']['position']=='decision' and r['signed_alignment']<0]
    layer=min(negative,key=lambda r:r['signed_alignment'])['site']['layer'] if negative else 18
    vs=[variants(r) for r in e.values['rank']];a=infer(e,'rank_A',vs,layer,capture=True);b=infer(e,'rank_B',vs,layer,source=True,capture=True)
    options=[]
    for i,s in enumerate(OPTION_SITES):
        r=infer(e,f'option{i}_restore',vs,layer,source=True,patch=[{'site':s,'donor':a['options'][:,i]}])
        p=infer(e,f'option{i}_transplant',vs,layer,patch=[{'site':s,'donor':b['options'][:,i]}])
        options.append({'site':s,'transmission':transmission(vs,a['scores'],b['scores'],r['scores'],p['scores']),
             'destinations':{f['id']:vector_transmission(a['field_'+f['id']],b['field_'+f['id']],r['field_'+f['id']],p['field_'+f['id']]) for f in FIELDS}})
        e.log('option_route',site=s)
    ranked=[];groups={};validation={}
    if negative:
        for h in range(16):
            r=infer(e,f'head{h}_restore',vs,layer,source=True,head_donor=a['heads'],heads=[h])
            p=infer(e,f'head{h}_transplant',vs,layer,head_donor=b['heads'],heads=[h])
            t=transmission(vs,a['scores'],b['scores'],r['scores'],p['scores']);ranked.append({'head':h,'transmission':t,
                    'signed_alignment':(t['restore_alignment']+t['transplant_alignment'])/2})
        groups={'positive':[r['head'] for r in sorted(ranked,key=lambda r:-r['signed_alignment']) if r['signed_alignment']>1e-4][:4],
                'negative':[r['head'] for r in sorted(ranked,key=lambda r:r['signed_alignment']) if r['signed_alignment']< -1e-4][:4]}
        write(e.out/'routes_groups_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'layer':layer,'groups':groups,'ranked':ranked})
        vv=e.task_values('controls','coref');aa=infer(e,'controls_A',vv,layer,capture=True);bb=infer(e,'controls_B',vv,layer,source=True,capture=True)
        scores={'baseline':aa['scores'],'source_half':bb['scores']}
        for name,heads in groups.items():
            r=infer(e,'controls_'+name+'_restore',vv,layer,source=True,head_donor=aa['heads'],heads=heads)['scores']
            p=infer(e,'controls_'+name+'_transplant',vv,layer,head_donor=bb['heads'],heads=heads)['scores']
            validation[name]={'heads':heads,'transmission':transmission(vv,aa['scores'],bb['scores'],r,p),
                         'restore_metrics':metrics(vv,r),'transplant_metrics':metrics(vv,p)}
            scores[name+'_restore']=r;scores[name+'_transplant']=p
        e.save_predictions('routes_controls',vv,scores)
    write(e.out/'routes.json',{'option_sites':options,'attention_layer':layer if negative else None,
            'head_scan':ranked,'groups':groups,'validation':validation,'status':'supplementary exploratory; no new deployment selection'})
    selected=read(e.out/'selection_frozen.json')['selected']['unit'];part_controls={}
    if selected['category']=='unit':
        edit=selected['edits'][0];key=edit['site'];site=next(s for s in e.meta['unit_sites'] if s['id']==key)
        rv=[variants(r) for r in e.values['rank']]
        rc=e.infer_parts('fit_A',rv,units=e.meta['unit_sites'],fields=True)
        x=rc['unit_'+key].astype(float);mu=e.params[key+'_mean'];rms=np.sqrt(np.mean((x-mu)**2,0))
        block=e.model.model.layers[site['layer']]
        w=block.mlp.down_proj.weight if site['module']=='mlp' else block.self_attn.o_proj.weight[:,site['head']*128:(site['head']+1)*128]
        wn=w.detach().float().norm(dim=0).cpu().numpy()
        original=edit['groups'];used=set()
        for group,_ in original:used.update(range(x.shape[1]) if group=='whole' else e.params[key+'_'+group].tolist())
        available=set(range(x.shape[1]))-used;alternate=[];profiles={}
        if len(available)>=len(used) and edit['mode']=='centered':
            for gi,(group,amount) in enumerate(original):
                indices=e.params[key+'_'+group];matched=[]
                for j in indices:
                    pool=np.array(sorted(available));d=np.log((rms[pool]+1e-8)/(rms[j]+1e-8))**2+np.log((wn[pool]+1e-8)/(wn[j]+1e-8))**2
                    z=int(pool[np.argmin(d)]);available.remove(z);matched.append(z)
                name='matched_selected_'+str(gi);e.params[key+'_'+name]=np.array(matched,dtype=int);alternate.append((name,amount))
                profiles[group]={'selected_indices':indices.tolist(),'matched_indices':matched,'amount':amount,
                    'selected_rms':float(rms[indices].mean()),'matched_rms':float(rms[matched].mean()),
                    'selected_column_norm':float(wn[indices].mean()),'matched_column_norm':float(wn[matched].mean())}
            gp=e.out/'part_control_groups_frozen.json'
            if gp.exists():assert read(gp)['groups']==profiles
            else:write(gp,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'site':site,'groups':profiles,'selection_source':'main dev only; matching rank only'})
            amt=max(abs(a) for _,a in original)
            specs={'baseline':{'name':'baseline','edits':[]},'source_half':{'name':'source_half','edits':[{'type':'source'}]},
                   'selected_unit':selected,'matched_unit':{'name':'matched_unit','edits':[{**edit,'groups':alternate}]},
                   'whole_raw':{'name':'whole_raw','edits':[{**edit,'mode':'raw','amount':amt,'groups':[('whole',amt)]}]},
                   'whole_centered':{'name':'whole_centered','edits':[{**edit,'groups':[('whole',amt)]}]}}
            for task in ('coref','reading','binding'):
                vv=e.task_values('controls',task);ss={name:e.infer_parts('part_control_'+task+'_'+name,vv,spec)['scores'] for name,spec in specs.items()}
                part_controls[task]={name:metrics(vv,z) for name,z in ss.items()};e.save_predictions('part_control_'+task,vv,ss)
            write(e.out/'part_controls.json',{'metrics':part_controls,'groups':profiles,'status':'same-site, same-budget fit-matched control; no target selection'})
    del e.model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
