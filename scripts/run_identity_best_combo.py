"""Serial GPU runner; freeze all model choices before any target evaluation."""
from pathlib import Path
import sys,time,subprocess,json,tomllib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism.fairsteer_author import ROOT,read,write,sha
from mechanism.identity_best_combo import CONFIG

if __name__=='__main__':
    config=tomllib.loads(CONFIG.read_text());folder=ROOT/config['output_dir'];folder.mkdir(parents=True,exist_ok=True)
    names=[m['name'] for m in config['models']]
    for stage in ('smoke','train','evaluate'):
        if stage=='evaluate':
            write(folder/'all_models_frozen.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
                'selections':{n:sha(folder/n/'selection_frozen.json') for n in names}})
        for name in names:
            log=ROOT/'results'/f'identity_best_combo_20261016_{stage}_{name}.log'
            print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,'model':name,'state':'start'}),flush=True)
            with log.open('a',encoding='utf-8') as f:
                subprocess.run([sys.executable,'-u','-m','mechanism.identity_best_combo','--stage',stage,'--model',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,'model':name,'state':'complete'}),flush=True)
    subprocess.run([sys.executable,'scripts/summarize_identity_best_combo.py'],cwd=ROOT,check=True)
    write(folder/'complete.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'runner_sha256':sha(Path(__file__)),'models':names})
