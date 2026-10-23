# One codebase for local and cloud experiments

## Integration decision

Before repository preparation, 219 code/config/test/research files from the server
were compared to local counterparts. All SHA256 digests matched. There was no
server-only scientific implementation to merge. The comparison is recorded in
`reproducibility/local_cloud_source_audit.json`. It describes the files before new
release helpers and repository documentation were added.

The local checkout is the maintained source. The server receives a checkout of the
same Git commit. Model files and experimental outputs remain outside Git. Separate
installation profiles record environment differences; TOML files hold experiment settings.

## Main dependency path

```text
scripts/release/run_bbq.py
  -> mechanism/cloud_bbq.py       Qwen3/Qwen3.5 loading and BBQ orchestration
     -> evidence_validation.py   repeated fitting and classifier controls
     -> evidence_story.py        probes, thresholds and response policies
     -> evidence_validation_stats.py
```

Historical modules also supply shared helpers. `fairsteer_author.py` provides IO,
hashes and metrics; `fairsteer_improve.py` provides deterministic sampling.
`closed_transfer.py` is imported by the historical validation module. Removing
these files based on their names would break the active dependency path.

## Fresh runs and historical records

| Purpose | Entry/config | Status |
|---|---|---|
| Exact BBQ input restoration | `scripts/release/prepare_bbq.py` | New, CPU-only, hash checked |
| Independent cloud reproduction | `scripts/release/run_bbq.py`, `configs/reproduction_bbq.toml` | New output directory; same fitting/policy implementation |
| Released-score verification | `scripts/release/verify_results.py` | CPU-only; compact committed predictions |
| Original cloud campaign | `scripts/run_cloud_bbq.py`, `configs/experiment_cloud_bbq_20261023_v1.toml` | Frozen historical run |
| Original small-model campaign | `scripts/run_evidence_validation.py` | Depends on historical assets and reused seed artifacts |
| Earlier explorations | `research/CODE_MAP.md` | Separate protocols and external assets |

The new launcher supplies independent paths and orchestration. Source files frozen
in the completed campaign are retained unchanged. All model/seed selections freeze
before evaluation. Each fresh run records its own source snapshot; the new wrapper
does not claim an identical historical source-file hash.

## Server workflow

The scientific archive remains at `/root/mechanism`. A separate repository checkout
at `/root/egus` holds the release commit. Synchronization does not launch model inference.
Use the existing environment while pointing imports to the new checkout:

```bash
cd /root/egus
export PATH="/root/conda-envs/egus/bin:$PATH"
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
python scripts/release/verify_results.py
```

The historical `scripts/cloud/env.sh` points to the old archive; do not source it
unchanged in the new checkout. Deployment can use a Git bundle, so the server needs
no stored GitHub credential. Its origin points to the GitHub repository. Future
changes should be synchronized by commit instead of editing two untracked copies.

## Outside Git

Raw/prepared datasets, weights, activation caches, wheels and SSH administration
files remain local to their machine. The unpublished manuscript and retired tools
remain on the workstation. Research notes are included; downloaded third-party source
trees are excluded. No existing experiment outputs are deleted or replaced.
