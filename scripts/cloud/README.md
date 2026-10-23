# Cloud environment for the Qwen extension

The three-model BBQ campaign is now complete. For a fresh checkout, use
[`docs/REPRODUCING.md`](../../docs/REPRODUCING.md) and
`scripts/release/run_bbq.py`. The instructions below record preparation of the
original server archive; their root-specific paths are historical.

Prepared on 2026-10-23. Code lives in `/root/mechanism`, model weights in
`/hy-tmp/models`, and the independent Conda environment in `/root/conda-envs/egus`.
The original base Conda environment is retained.

```bash
cd /root/mechanism
source scripts/cloud/env.sh
python scripts/cloud/smoke_models.py
```

The selected official repositories are `Qwen/Qwen3-8B`, `Qwen/Qwen3.5-4B`,
and `Qwen/Qwen3.5-9B`. `logs/models_manifest.json` records their immutable
revisions, expected file sizes, and upstream LFS hashes. Downloads use
hf-mirror and are resumable. Each checkpoint receives a `download_verified.json`
only after all files pass size and available upstream SHA256 checks.

The server reports an RTX 4090 with 49,140 MiB and driver 550.163.01.
The environment uses PyTorch 2.6.0 + CUDA 12.4 to match that driver,
Transformers 5.18.0, and scikit-learn 1.9.1. Python 3.12.9 comes from the
server's Conda base clone; the local research environment uses Python 3.13
and a newer PyTorch. These environment differences must be reported for
subsequent experiments. Source modules are imported via PYTHONPATH rather
than installing the local project, whose metadata requires Python 3.13.
The resolved remote package list is saved in `logs/environment.lock.txt`.

Qwen3.5 checkpoints contain a vision encoder and a hybrid text decoder.
The smoke test loads their native `Qwen3_5ForConditionalGeneration` class,
uses text-only inputs, and locates decoder blocks through
`model.model.language_model.layers`. Qwen3 uses `model.model.layers`.
It tests final-prompt hidden-state extraction, zero-edit equivalence,
nonzero-edit propagation, and isolation between edited and unedited batch
rows. A padded-versus-single-input comparison checks positional handling.
These checks are infrastructure checks, not model-performance results.

The uploaded BBQ files retain the existing five fitting resamples and the
shared development/evaluation partition. Existing frozen experiment
implementations are included for reference. Their original Windows model
paths and campaign preparation logic are not a launch configuration for
these new checkpoints. A new BBQ-only campaign must explicitly connect the
Qwen3.5 adapter and freeze its own configuration before scientific evaluation.
No old result directories are overwritten by the preparation scripts.

Logs:

- `logs/bootstrap.log`: dependency installation.
- `logs/environment_ready.json`: CUDA and model-class import checks.
- `logs/download.log`: download and checksum events.
- `logs/smoke/*.json`: model-loading and activation-hook checks.
