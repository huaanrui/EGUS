"""Independent numerical equivalence of QKV edge recompute and head-node patch."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import time
import numpy as np
import torch
from mechanism.identity_path import Experiment
from mechanism.fairsteer_author import read,write,sha


def main():
    e=Experiment();e.prepare();e.setup()
    identity={'script':sha(Path(__file__)),'selection':sha(e.out/'selection_frozen.json')}
    declaration=e.out/'qkv_validation_declaration.json'
    if declaration.exists():assert read(declaration)['identity']==identity
    else:write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'identity':identity,
        'purpose':'numerical implementation validation, all frozen heads and control rows; no selection'})
    heads=read(e.out/'selection_frozen.json')['heads'];layers=sorted({s['layer'] for s in heads})
    vs=e.task_values('controls','coref');folder=e.out/'qkv_validation_shards';folder.mkdir(exist_ok=True);chunks=[]
    for start in range(0,len(vs),e.c['batch_size']):
        p=folder/f'{start:06d}.npz'
        if p.exists():
            with np.load(p) as f:chunks.append(f['errors'])
            continue
        batch=vs[start:start+e.c['batch_size']];inputs,pos=e.encode(batch)
        b=e.run_forward(batch,'half',capture=True,full_layers=layers,inputs_pos=(inputs,pos));errors=[]
        for s in heads:
            manual=e.run_forward(batch,edge={'site':s,'recipe':'all_qkv','donor_norm':b['norm'+str(s['layer'])]},inputs_pos=(inputs,pos))['scores']
            node=e.run_forward(batch,sites=[s],donor=b['heads'],inputs_pos=(inputs,pos))['scores']
            error=np.max(abs(manual-node),axis=1);assert error.max()<.04,float(error.max());errors.append(error)
        z=np.stack(errors,axis=1);np.savez(p,errors=z);chunks.append(z)
    errors=np.concatenate(chunks)
    write(e.out/'qkv_validation.json',{'rows':len(vs),'heads':heads,'candidate_logit_tolerance':.04,
          'max_error':float(errors.max()),'per_head_max':errors.max(0).tolist(),'passed':True})
    print(read(e.out/'qkv_validation.json'))
    del e.model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
