"""Run each model's smoke check when its download and the environment are ready."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

root = Path('/root/mechanism')
models = ['Qwen3-8B', 'Qwen3.5-4B', 'Qwen3.5-9B']
env = dict(os.environ, PYTHONPATH=str(root), HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')
for name in models:
    while not (root / 'logs/environment_ready.json').exists() or not (Path('/hy-tmp/models') / name / 'download_verified.json').exists():
        time.sleep(15)
    p = subprocess.run(['/root/conda-envs/egus/bin/python', '-u', 'scripts/cloud/smoke_models.py',
                        '--models', name], cwd=root, env=env)
    if p.returncode:
        (root / 'logs/preparation_failed.json').write_text(json.dumps({'model': name, 'returncode': p.returncode}))
        sys.exit(p.returncode)
(root / 'logs/preparation_complete.json').write_text(json.dumps({'models': models, 'passed': True}, indent=2))
print('All checkpoints verified and smoke checks passed.', flush=True)
