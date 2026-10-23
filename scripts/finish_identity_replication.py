"""Sequential GPU stages; all selections freeze before any target evaluation."""
from pathlib import Path
import subprocess,sys,time,json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/identity_replication_20261015_v1'
models=('qwen3','minicpm5','qwen25')
start=time.monotonic()
while not all((OUT/name/'selection_frozen.json').exists() for name in models):
    log=ROOT/'results/identity_replication_20261015_train_resume.log'
    if log.exists() and 'Traceback (most recent call last)' in log.read_text(encoding='utf-8',errors='replace')[-6000:]:
        raise RuntimeError('Training failed; inspect training log before resuming')
    if time.monotonic()-start>7200:raise TimeoutError('Training selections not complete within bounded wait')
    time.sleep(5)
while True:
    result=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if max(int(x) for x in result.split())<100:break
    time.sleep(5)
stages=[
 ('norm_freeze',['scripts/identity_replication_norm_controls.py','--stage','freeze']),
 ('evaluate',['scripts/run_identity_replication.py','--stage','evaluate']),
 ('norm_evaluate',['scripts/identity_replication_norm_controls.py','--stage','evaluate']),
 ('random_utilities',['scripts/identity_replication_random_utilities.py']),
 ('summary',['scripts/summarize_identity_replication.py']),
 ('norm_summary',['scripts/summarize_identity_replication_norm.py']),
 ('plot',['scripts/plot_identity_replication.py']),
]
for name,command in stages:
    print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':name,'state':'start'}),flush=True)
    with (ROOT/f'results/identity_replication_20261015_{name}.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':name,'state':'complete'}),flush=True)
(OUT/'complete.json').write_text(json.dumps({'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'stages':[s[0] for s in stages]},indent=2),encoding='utf-8')
