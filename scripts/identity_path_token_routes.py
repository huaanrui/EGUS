from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import time
import numpy as np
import torch
from mechanism.identity_path import Experiment,semantic_margin
from mechanism.identity_causal import render,metrics
from mechanism.fairsteer_author import ROOT,read,write,sha

GROUPS=('context_prefix','context_pronoun','context_suffix','question_pronoun',
        'question_other','option_A','option_B','answer_marker','format_other')


def token_groups(e,vs,positions,width):
    texts=[render(r) for r in vs]
    enc=e.tok(texts,padding=True,return_offsets_mapping=True,return_tensors='pt',add_special_tokens=False)
    assert enc.input_ids.shape[1]==width
    result=np.full((len(vs),width),-1,dtype=int)
    for i,(r,text,off) in enumerate(zip(vs,texts,enc.offset_mapping.tolist())):
        ce=len(r['context']);qs=text.index(r['question'],ce);qe=qs+len(r['question'])
        oa=text.index('\nA. ',qe);ob=text.index('\nB. ',oa+1);ans=text.rindex('\nAnswer:')
        sp,qp=int(positions[i,0]),int(positions[i,4])
        for j,(a,b) in enumerate(off):
            if b<=a:continue
            midpoint=(a+b)/2
            if j==sp:g='context_pronoun'
            elif j==qp:g='question_pronoun'
            elif midpoint<ce:g='context_prefix' if j<sp else 'context_suffix'
            elif qs<=midpoint<qe:g='question_other'
            elif oa<=midpoint<ob:g='option_A'
            elif ob<=midpoint<ans:g='option_B'
            elif midpoint>=ans:g='answer_marker'
            else:g='format_other'
            result[i,j]=GROUPS.index(g)
    assert np.array_equal(result>=0,enc.attention_mask.numpy().astype(bool))
    return result


def main():
    e=Experiment();e.prepare();e.setup()
    protocol=ROOT/'research/2026-10-15_identity_path_token_routes.md'
    identity={'script':sha(Path(__file__)),'protocol':sha(protocol),'selection':sha(e.out/'selection_frozen.json')}
    declaration=e.out/'token_routes_declaration.json'
    if declaration.exists():assert read(declaration)['identity']==identity
    else:write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,'status':'fixed position groups; historical; never reselects'})
    heads=[s for s in read(e.out/'selection_frozen.json')['heads'] if s['position']=='decision']
    vs=e.task_values('controls','coref');layers=sorted({s['layer'] for s in heads})
    folder=e.out/'token_route_shards';folder.mkdir(exist_ok=True);chunks=[]
    for start in range(0,len(vs),e.c['batch_size']):
        p=folder/f'{start:06d}.npz'
        if p.exists():
            with np.load(p) as f:chunks.append({k:f[k] for k in f.files})
            continue
        batch=vs[start:start+e.c['batch_size']];inputs,pos=e.encode(batch)
        groups=token_groups(e,batch,pos,inputs['input_ids'].shape[1])
        a=e.run_forward(batch,capture=True,full_layers=layers,inputs_pos=(inputs,pos))
        b=e.run_forward(batch,'half',capture=True,full_layers=layers,inputs_pos=(inputs,pos))
        r={'baseline':a['scores'],'source_half':b['scores']}
        for h,s in enumerate(heads):
            for g,name in enumerate(GROUPS):
                mask=groups==g
                donor=np.where(mask[:,:,None],b['norm'+str(s['layer'])],a['norm'+str(s['layer'])])
                for kind in ('k','v'):
                    edge={'site':s,'recipe':'all_'+kind,'donor_norm':donor}
                    r[f'h{h}_{kind}_{name}']=e.run_forward(batch,edge=edge,inputs_pos=(inputs,pos))['scores']
        np.savez(p,**r);chunks.append(r)
        if start%(e.c['batch_size']*4)==0:e.log('token_routes_progress',done=start+len(batch),total=len(vs))
    scores={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    t=semantic_margin(vs,scores['source_half'])-semantic_margin(vs,scores['baseline']);energy=np.mean(t*t)
    result={}
    for name,z in scores.items():
        d=semantic_margin(vs,z)-semantic_margin(vs,scores['baseline'])
        result[name]={'alignment':float(np.mean(t*d)/energy),'effect_rms':float(np.sqrt(np.mean(d*d))),
                     'reproduction_fidelity':float(1-np.mean((d-t)**2)/energy),'metrics':metrics(vs,z)}
    e.save_predictions('token_routes',vs,scores)
    write(e.out/'token_routes.json',{'heads':heads,'groups':GROUPS,'results':result,
        'status':'historical fixed groups; all unadjusted, no reselection; score effects are not additive'})
    del e.model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
