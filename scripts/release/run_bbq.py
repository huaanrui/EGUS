"""Portable launcher around the unchanged, audited cloud BBQ implementation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from mechanism.cloud_bbq import Campaign, assert_selection_lock, summarize
    from mechanism.fairsteer_author import sha, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/reproduction_bbq.toml')
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'all', 'report'], default='all')
    args = parser.parse_args()
    config = args.config.resolve()
    if not config.is_relative_to(ROOT):
        parser.error('Keep the reproduction config inside this checkout for source snapshots.')
    campaign = Campaign(config)
    if campaign.out in [ROOT / 'results/cloud_bbq_20261023_v1', ROOT / 'results/evidence_validation_20261017_v1']:
        parser.error('Choose a new output directory; historical result directories are protected.')
    lock = campaign.out / 'release_runner.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({'pid': os.getpid(), 'config': str(config)}))
        campaign.prepare()
        if args.stage == 'prepare':
            return
        if args.stage == 'report':
            summarize(config)
            return
        stages = ['smoke'] if args.stage == 'smoke' else ['train', 'evaluate']
        for stage in stages:
            if stage == 'evaluate':
                selections = {m['name']: {str(s): sha(campaign.out / m['name'] / f'seed_{s}' / 'selection_frozen.json')
                                          for s in campaign.c['seeds']} for m in campaign.c['models']}
                path = campaign.out / 'all_selections_frozen.json'
                if path.exists() and json.loads(path.read_text()) != selections:
                    raise ValueError('Frozen selection manifest changed')
                if not path.exists():
                    write(path, selections)
                assert_selection_lock(campaign.out, list(selections), campaign.c['seeds'])
            for model in campaign.c['models']:
                status = {'stage': stage, 'model': model['name'], 'state': 'running',
                          'time': time.strftime('%Y-%m-%d %H:%M:%S')}
                write(campaign.out / 'status.json', status)
                print(json.dumps(status), flush=True)
                with (campaign.out / f'{stage}_{model["name"]}.log').open('a', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-u', '-m', 'mechanism.cloud_bbq', stage,
                                    '--config', str(config), '--model', model['name']], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.stage == 'all':
            write(campaign.out / 'inference_complete.json', {'complete': True})
            summarize(config)
        write(campaign.out / 'status.json', {'state': 'complete', 'stage': args.stage})
    except Exception as error:
        write(campaign.out / 'status.json', {'state': 'failed', 'error': str(error)})
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
