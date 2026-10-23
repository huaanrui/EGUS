from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import time
import numpy as np
import torch
from mechanism.identity_path import Experiment,RECEIVERS,semantic_margin
from mechanism.identity_combo import SCOPES
from mechanism.identity_causal import metrics
from mechanism.fairsteer_author import ROOT,read,write,sha


def main():
    e=Experiment();e.prepare();e.setup();assert (e.out/'edges.json').exists()
    protocol=ROOT/'research/2026-10-15_identity_path_receiver_edges.md'
    identity={'script':sha(Path(__file__)),'protocol':sha(protocol),'selection':sha(e.out/'selection_frozen.json')}
    declaration=e.out/'receiver_edges_declaration.json'
    if declaration.exists():assert read(declaration)['identity']==identity
    else:write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,'status':'post-QKV exploratory; all ordered pairs; no reselection'})
    heads=read(e.out/'selection_frozen.json')['heads'];pairs=[(i,j) for i,u in enumerate(heads) for j,v in enumerate(heads) if u['layer']<v['layer']]
    layers=sorted({s['layer'] for s in heads});vs=e.task_values('controls','coref')
    folder=e.out/'receiver_edge_shards';folder.mkdir(exist_ok=True);chunks=[]
    for start in range(0,len(vs),e.c['batch_size']):
        p=folder/f'{start:06d}.npz'
        if p.exists():
            with np.load(p) as f:chunks.append({k:f[k] for k in f.files})
            continue
        batch=vs[start:start+e.c['batch_size']];inputs,pos=e.encode(batch)
        a=e.run_forward(batch,capture=True,full_layers=layers,inputs_pos=(inputs,pos))
        b=e.run_forward(batch,'half',capture=True,inputs_pos=(inputs,pos));r={'baseline':a['scores'],'source_half':b['scores']}
        for i,j in pairs:
            u,v=heads[i],heads[j];l=u['layer']-8;pp=RECEIVERS.index(u['position'])
            hd=b['heads'][:,l,pp,u['head']].astype(float)-a['heads'][:,l,pp,u['head']].astype(float)
            weight=e.model.model.layers[u['layer']].self_attn.o_proj.weight[:,u['head']*128:(u['head']+1)*128].detach().float().cpu().numpy()
            delta=hd@weight.T
            edge={'site':v,'recipe':'direct_kv','delta':delta,'relay_pos':u['position']}
            r[f'h{i}_to_h{j}_kv']=e.run_forward(batch,edge=edge,inputs_pos=(inputs,pos))['scores']
            if u['position']==v['position']:
                raw=torch.as_tensor(a['raw'+str(v['layer'])],device='cuda').clone()
                ix=torch.arange(len(batch),device='cuda');p0=pos[:,SCOPES.index(u['position'])];valid=p0>=0
                raw[ix[valid],p0[valid]]+=torch.as_tensor(delta,device='cuda',dtype=raw.dtype)[valid]
                with torch.inference_mode():norm=e.model.model.layers[v['layer']].input_layernorm(raw).cpu().numpy()
                edge={'site':v,'recipe':'q_receiver','donor_norm':norm}
                r[f'h{i}_to_h{j}_q']=e.run_forward(batch,edge=edge,inputs_pos=(inputs,pos))['scores']
        np.savez(p,**r);chunks.append(r)
        if start%(e.c['batch_size']*8)==0:e.log('receiver_edges_progress',done=start+len(batch),total=len(vs))
    scores={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    t=semantic_margin(vs,scores['source_half'])-semantic_margin(vs,scores['baseline']);energy=np.mean(t*t);result={}
    for name,z in scores.items():
        d=semantic_margin(vs,z)-semantic_margin(vs,scores['baseline'])
        result[name]={'alignment':float(np.mean(t*d)/energy),'effect_rms':float(np.sqrt(np.mean(d*d))),
                     'reproduction_fidelity':float(1-np.mean((d-t)**2)/energy),'metrics':metrics(vs,z)}
    e.save_predictions('receiver_edges',vs,scores)
    write(e.out/'receiver_edges.json',{'heads':heads,'pairs':pairs,'results':result,'status':'post-QKV exploratory additive edges; no necessity/completeness claim'})
    del e.model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
