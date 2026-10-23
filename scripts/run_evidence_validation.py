from pathlib import Path
import sys,subprocess,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism.evidence_validation import Campaign
from mechanism.fairsteer_author import sha,write

if __name__=='__main__':
    c=Campaign();c.prepare()
    for stage in ('train','evaluate'):
        if stage=='evaluate':
            write(c.out/'all_selections_frozen.json',{m['name']:{str(s):sha(c.out/m['name']/f'seed_{s}/selection_frozen.json') for s in c.c['seeds']} for m in c.c['models']})
        for m in c.c['models']:
            name=m['name'];p=c.out/f'{stage}_{name}.log'
            print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,'model':name,'state':'start'}),flush=True)
            with p.open('a',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-u','-m','mechanism.evidence_validation',stage,'--model',name],stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:raise SystemExit(f'Failed {stage}/{name}; see {p}')
    write(c.out/'inference_complete.json',{'finished':time.strftime('%Y-%m-%d %H:%M:%S'),'models':[m['name'] for m in c.c['models']],'seeds':c.c['seeds']})
