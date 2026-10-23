# Reproducing the experiments

## Score verification without models

Install NumPy, then run:

```bash
python scripts/release/verify_results.py
```

Thirty committed prediction files cover six models, five resamples, twelve policies
and 704 questions. The checker recomputes overall and condition-specific mean accuracy
and sample SD. It does not replay hidden-state inference.

## Data preparation

```bash
python scripts/release/prepare_bbq.py
```

The standard-library script downloads eleven pinned BBQ category files, supplemental
metadata and the historical FairSteer MMLU utility source. It reconstructs 58,492 source
rows and nine prepared JSONL files plus `training.json`. Prepared data go under
`data/prepared/egus_bbq_v1`; the source goes under `data/prepared/egus_bbq_source`.
Every input and output hash must match the manifest.

Ordered sample identifiers select the subsets. `training.json` records nested budgets,
direction UIDs and Unknown vocabularies without question text. CRLF bytes intentionally
match the original Windows files, including on Linux. Existing different files are rejected.

For offline use, prepopulate raw files at paths in `reproducibility/bbq/manifest.json`
and add `--offline`. Flags `--raw-dir`, `--data-dir`, `--source-dir` allow an independent
validation copy. Attribution is recorded in `THIRD_PARTY_NOTICES.md`.

## GPU environment

The tested cloud profile is Python 3.12.9, PyTorch 2.6.0+cu124, Transformers 5.18.0
and scikit-learn 1.9.1. The server GPU reports about 48 GiB. Use a dedicated environment:

```bash
conda create -n egus python=3.12.9 -y
conda activate egus
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r scripts/cloud/requirements.txt
python -m pip check
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
```

`environments/cloud.lock.txt` preserves the original `pip freeze`, including historical
local wheel paths. It is provenance, rather than a portable installation input.
The requirement list above pins the cloud scientific packages. Root `pyproject.toml`
and `uv.lock` preserve the distinct local Python 3.13/PyTorch 2.14 environment.

## Models and smoke checks

```bash
python scripts/cloud/download_models.py --manifest reproducibility/cloud_models.json --root /hy-tmp/models
python scripts/cloud/smoke_models.py --root /hy-tmp/models
```

Downloads use hf-mirror with pinned official revisions. The downloader checks sizes
and available upstream weight hashes, then writes `download_verified.json`. Keep
roughly 45 GB of weights plus working space outside the checkout. Smoke checks test
hidden-state extraction and interventions and write ignored logs under `logs/`.

## Independent BBQ run

Review `configs/reproduction_bbq.toml`. Set model paths to verified checkpoint folders
and choose a new `output_dir` before starting. Defaults cover the three cloud models.
Keep the config unchanged once its plan freezes.

```bash
python scripts/release/run_bbq.py --stage prepare
python scripts/release/run_bbq.py --stage smoke
python scripts/release/run_bbq.py --stage all
```

Preparation validates source/data/model hashes. The full stage trains every model
and seed, freezes all selections, then evaluates. Resume reuses unchanged hashes
and existing shards. An exclusive lock prevents simultaneous launches. After an
unclean exit, verify no runner remains before removing `release_runner.lock`.

The launcher reuses the audited fitting and response functions. Full GPU reproduction
ran through the original cloud entry. The new wrapper has orchestration and data
compatibility checks, without a second full GPU campaign during repository preparation.

Earlier small-model configs remain available. Their historical launcher also prepares
transfer panels and reuses old seed assets, so it is not a fresh-clone command. For
BBQ-only runs on these models, copy the reproduction config, set checkpoint paths
and fixed dimensions from `docs/PROTOCOL.md`, and verify the checkpoints separately.
The new wrapper's adapter has been exercised on the cloud models; any new small-model
run requires its own smoke checks and source snapshot.

## CPU implementation tests

These main protocol tests need the scientific environment, without downloaded data:

```bash
python -m unittest discover -s tests -p test_cloud_bbq.py -v
python -m unittest discover -s tests -p test_evidence_validation.py -v
python -m unittest discover -s tests -p test_evidence_story.py -v
python -m unittest discover -s tests -p test_release_runner.py -v
```

These 20 tests cover gate/action consistency, thresholds, sample isolation and selection
integrity, including train-before-evaluation ordering in the new launcher. The original
178-test historical suite also passes on the research workstation;
some tests require external historical datasets or downloaded author files.
