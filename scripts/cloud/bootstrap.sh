#!/usr/bin/env bash
set -euo pipefail
cd /root/mechanism
PY=/root/conda-envs/egus/bin/python
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/hy-tmp/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=180 HF_HUB_ETAG_TIMEOUT=60
"$PY" -m pip install --upgrade pip --index-url https://pypi.org/simple
# The server has driver 550 / CUDA 12.4. Keep its supported CUDA runtime.
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
if [[ -d offline_wheels ]]; then
    # Install the transferred wheels first so pip does not select the same large
    # files from a slow remote index; resolve only missing small dependencies next.
    "$PY" -m pip install --no-deps offline_wheels/*.whl
    "$PY" -m pip install -r scripts/cloud/requirements.txt --find-links offline_wheels --index-url https://pypi.org/simple
else
    "$PY" -m pip install -r scripts/cloud/requirements.txt --index-url https://pypi.org/simple
fi
"$PY" -m pip check
"$PY" -m pip freeze > logs/environment.lock.txt
"$PY" - <<'PY'
import json, torch, transformers
from transformers import AutoConfig
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
assert torch.cuda.is_available()
assert 'qwen3_5' in __import__('transformers').models.auto.configuration_auto.CONFIG_MAPPING
info = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'cuda_matmul': float((torch.ones((32,32),device='cuda') @ torch.ones((32,32),device='cuda'))[0,0])}
with open('logs/environment_ready.json','w') as f: json.dump(info,f,indent=2)
print(info,flush=True)
PY
