"""Sequential GPU campaign, all selections frozen before evaluation forwards."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mechanism.cloud_bbq import Campaign, assert_selection_lock, summarize
from mechanism.fairsteer_author import write, sha


def main():
    import fcntl
    campaign = Campaign()
    lockfile = (campaign.out / 'run.lock').open('w')
    fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign.prepare()
    write(campaign.out / 'run_started.json', {'pid': os.getpid(), 'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    try:
        for stage in ('train', 'evaluate'):
            if stage == 'evaluate':
                lock = {m['name']: {str(s): sha(campaign.out / m['name'] / f'seed_{s}' / 'selection_frozen.json')
                                   for s in campaign.c['seeds']} for m in campaign.c['models']}
                path = campaign.out / 'all_selections_frozen.json'
                if path.exists():
                    assert json.loads(path.read_text()) == lock
                else:
                    write(path, lock)
                assert_selection_lock(campaign.out, [m['name'] for m in campaign.c['models']], campaign.c['seeds'])
            for model in campaign.c['models']:
                name = model['name']
                status = {'stage': stage, 'model': name, 'state': 'running', 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
                write(campaign.out / 'status.json', status)
                print(json.dumps(status), flush=True)
                with (campaign.out / f'{stage}_{name}.log').open('a', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, '-u', '-m', 'mechanism.cloud_bbq', stage,
                                             '--model', name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{stage}/{name} failed, code {result.returncode}')
                if stage == 'evaluate':
                    summarize()
        write(campaign.out / 'inference_complete.json', {'complete': True})
        write(campaign.out / 'status.json', {'state': 'complete', 'models': [m['name'] for m in campaign.c['models']]})
    except Exception as exc:
        write(campaign.out / 'status.json', {'state': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
