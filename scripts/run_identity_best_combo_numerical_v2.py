"""Execution-only adaptive loss scaling for finite FP16 rank gradients."""
from pathlib import Path
import sys,argparse,json,time,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mechanism.fairsteer_author import ROOT,read,write,sha
from mechanism.identity_best_combo import Search,CONFIG
from mechanism.identity_combo import SCOPES

OUT=ROOT/'results/identity_best_combo_20261016_v1'
PROTOCOL=ROOT/'research/2026-10-16_identity_best_combo_numerical_v2.md'
SCALES=(1024,256,64,16,4,1)
ORIGINAL_CAPTURE=Search.capture_gradient


def finiteness(result):
    bad=[k for k,x in result.items() if not np.isfinite(x).all()]
    if any(not k.endswith('_g') for k in bad):return 'forward',bad
    return ('gradient' if bad else 'finite'),bad


def capture_at_scale(e,vs,scale):
    e.setup();assert len(vs)==1;inputs,pos=e.encode(vs);handles=[];saved={}
    scopes=[SCOPES.index(p) for p in e.c['scopes']]
    handles.append(e.model.get_input_embeddings().register_forward_hook(lambda m,a,o:o.detach().requires_grad_(True)))
    for layer,block in enumerate(e.layers):
        for kind,projection in [('mlp',block.mlp.down_proj),('attention',block.self_attn.o_proj)]:
            def hook(m,a,key=f'{kind}{layer}'):saved[key]=a[0]
            handles.append(projection.register_forward_pre_hook(hook))
    try:
        with torch.enable_grad():
            output=e.model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,e.ids]
            sign=1 if vs[0]['choices'][0]==sorted(vs[0]['choices'])[0] else -1
            keys=list(saved)
            gradients=torch.autograd.grad((output[:,0].float()-output[:,1].float()).sum()*sign*scale,[saved[k] for k in keys])
            result={'scores':output.detach().float().cpu().numpy()}
            for key,g in zip(keys,gradients):
                locations=pos[0,scopes]
                result[key+'_a']=saved[key][0,locations].detach().cpu().numpy()
                result[key+'_g']=(g[0,locations].detach().float()/scale).cpu().numpy()
            return result
    finally:
        for h in handles:h.remove()


def adaptive_capture(e,vs):
    failures=[]
    for scale in SCALES:
        result=capture_at_scale(e,vs,scale);status,bad=finiteness(result)
        if status=='forward':raise FloatingPointError('Nonfinite forward values, loss scaling is not a repair: '+repr(bad))
        if status=='finite':
            if failures:e.log('loss_scale_recovered',uid=vs[0]['uid'],scale=scale,rejected=failures)
            result['gradient_loss_scale']=np.array([scale],dtype=np.float32)
            result['execution_v2_sha256']=np.array(sha(Path(__file__)))
            return result
        failures.append({'scale':scale,'nonfinite_fields':bad})
        e.log('loss_scale_rejected',uid=vs[0]['uid'],scale=scale,nonfinite_fields=bad)
    raise FloatingPointError('All loss scales failed for '+vs[0]['uid'])


def validate(e):
    path=e.out/'numerical_v2_smoke.json'
    if path.exists():assert read(path)['executor_sha256']==sha(Path(__file__));return
    vs=e.task_values('rank','coref')[:1]
    original=ORIGINAL_CAPTURE(e,vs);same=capture_at_scale(e,vs,1024);small=capture_at_scale(e,vs,64)
    assert finiteness(same)[0]==finiteness(small)[0]=='finite'
    tests={}
    for key in original:
        np.testing.assert_array_equal(original[key],same[key])
        if key.endswith('_g'):
            a=original[key].astype(float).ravel();b=small[key].astype(float).ravel()
            relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))
            assert relative<.02,(key,relative);tests[key]=relative
        else:np.testing.assert_array_equal(original[key],small[key])
    write(path,{'executor_sha256':sha(Path(__file__)),'scale1024_matches_original_exactly':True,
        'lower_scale_forward_matches_exactly':True,'scale64_gradient_relative_error_max':max(tests.values()),
        'per_field_relative_error':tests})
    e.log('numerical_v2_smoke_complete',relative_error_max=max(tests.values()))


def worker(stage,name):
    declaration=read(OUT/'numerical_v2_declaration.json')
    assert declaration['executor_sha256']==sha(Path(__file__))
    e=Search(name);e.prepare()
    try:
        if stage=='train':
            if name=='qwen25':
                validate(e);Search.capture_gradient=adaptive_capture
            e.smoke();e.fit();e.select()
        else:e.evaluate()
    finally:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=('train','evaluate'));p.add_argument('--model');a=p.parse_args()
    if a.worker:worker(a.worker,a.model);raise SystemExit
    declaration=OUT/'numerical_v2_declaration.json'
    if declaration.exists():assert read(declaration)['executor_sha256']==sha(Path(__file__))
    else:
        assert not (OUT/'all_models_frozen.json').exists(),'Declare numerical repair before any target evaluation'
        write(declaration,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'executor_sha256':sha(Path(__file__)),
            'protocol_sha256':sha(PROTOCOL),'test_sha256':sha(ROOT/'tests/test_identity_best_combo_numerical_v2.py'),
            'scales':SCALES,'reason':'v1 Qwen2.5 rank capture rejected nonfinite values; distinguish forward failure from gradient overflow',
            'already_frozen_selections':{n:sha(OUT/n/'selection_frozen.json') for n in ('qwen3','minicpm5')},
            'unchanged':'v1 scientific source/config/data, mathematical gradient objective, ranking, candidate search and scoring'})
    for stage in ('train','evaluate'):
        if stage=='evaluate':
            write(OUT/'all_models_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
                'selections':{n:sha(OUT/n/'selection_frozen.json') for n in ('qwen3','minicpm5','qwen25')}})
        for name in ('qwen3','minicpm5','qwen25'):
            for n,h in read(declaration)['already_frozen_selections'].items():assert sha(OUT/n/'selection_frozen.json')==h
            print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,'model':name,'state':'start','execution_version':2}),flush=True)
            log=ROOT/'results'/f'identity_best_combo_20261016_numerical_v2_{stage}_{name}.log'
            with log.open('a',encoding='utf-8') as f:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker',stage,'--model',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,'model':name,'state':'complete','execution_version':2}),flush=True)
    subprocess.run([sys.executable,'scripts/summarize_identity_best_combo.py'],cwd=ROOT,check=True)
    write(OUT/'complete.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'execution_version':2,
        'runner_sha256':sha(Path(__file__)),'numerical_declaration_sha256':sha(declaration),'models':['qwen3','minicpm5','qwen25']})
