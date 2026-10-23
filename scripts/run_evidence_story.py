"""Serial GPU work; all selections frozen before any target inference."""
from pathlib import Path
import sys, subprocess, time, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism.evidence_story import Study
from mechanism.fairsteer_author import write,sha

if __name__=='__main__':
    s=Study();s.prepare();names=[m['name'] for m in s.c['models']]
    for stage in ('train','evaluate'):
        if stage=='evaluate':write(s.out/'all_models_frozen.json',{n:sha(s.out/n/'selection_frozen.json') for n in names})
        for name in names:
            path=s.out/f'{stage}_{name}.log'
            print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,'model':name,'state':'start'}),flush=True)
            with path.open('a',encoding='utf-8') as f:
                r=subprocess.run([sys.executable,'-u','-m','mechanism.evidence_story',stage,'--model',name],stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:raise SystemExit(f'{stage}/{name} failed: {path}')
    write(s.out/'inference_complete.json',{'finished':time.strftime('%Y-%m-%d %H:%M:%S'),'models':names})
