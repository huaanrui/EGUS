# EGUS: Evidence-Gated Unknown Selection

Research code for evidence-guided response selection in bias-sensitive question answering.
EGUS reads a frozen language model's final-prompt hidden state with a logistic evidence
probe. When the probe detects insufficient context, the policy selects the input's
Unknown option. Other predictions are retained.

The experiments separate the evidence detector from its downstream action. EGUS and
evidence-gated steering use the **same fitted detector and threshold**. A surface-feature
classifier supplies a separate control using ten text statistics instead of hidden states.

## Results

BBQ accuracy (%), averaged over five training resamples of 2,200 questions. Development
has 352 questions; the shared evaluation set has 704 questions from 51 separated
question families. These are repeated fits on a historical evaluation set.

| Model | Base | Local FairSteer-style | Same-gate steering | EGUS | Surface-direct |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 56.68 | 65.45 | 78.95 | **79.09** | 75.99 |
| MiniCPM5-2B | 43.47 | 43.24 | 43.10 | **84.43** | 78.27 |
| Qwen2.5-3B-Instruct | 73.01 | 73.38 | 78.61 | **81.28** | 80.97 |
| Qwen3-8B | 75.43 | 76.45 | 78.27 | **82.64** | 81.16 |
| Qwen3.5-4B | 74.72 | 75.60 | 80.74 | **91.36** | 88.98 |
| Qwen3.5-9B | 77.41 | 77.53 | 77.95 | **92.81** | 90.74 |

The gains concentrate on insufficient-context questions. Surface-direct is competitive;
EGUS's advantage over it does not pass the applicable Holm correction for any model.
The local FairSteer-style baseline is an adaptation with documented supervision and
solver differences. These results characterize the tested BBQ protocol, rather than
establishing general debiasing or open-generation performance.

Machine-readable results include training-resample variation and conditional accuracy in
[`benchmarks/`](benchmarks). Thirty prediction files support independent recomputation.
The two campaigns retain their own original multiple-comparison families.

## Start here

- [Reproduce BBQ experiments](docs/REPRODUCING.md): setup, pinned data, models, launch and verification.
- [Code organization and local/cloud workflow](docs/CODE_ORGANIZATION.md).
- [Protocol and interpretation](docs/PROTOCOL.md).
- [Third-party sources](THIRD_PARTY_NOTICES.md).
- [Historical research index](research/CODE_MAP.md): earlier component and transfer explorations.

The primary fresh-run entry is `scripts/release/run_bbq.py`. Historical entry points remain
available for provenance. `main.py` and `python -m mechanism` run an early pilot.

## Verify released scores without a GPU

Python and NumPy are sufficient:

```bash
python scripts/release/verify_results.py
```

This independently checks 253,440 predictions across six models, including ambiguous and
sufficient-context accuracies. It does not rerun the language models.

## Restore exact experiment inputs

This step uses only the Python standard library:

```bash
python scripts/release/prepare_bbq.py
```

It downloads pinned upstream files, reconstructs the source corpus, and restores five
fit samples plus development/evaluation inputs. Every output must match its recorded
SHA256. No dataset question text is redistributed here. Existing different data are
never overwritten. See the reproduction guide for GPU setup and execution.

The root `pyproject.toml` and `uv.lock` preserve the original Windows/Python 3.13 environment;
the cloud campaign used Python 3.12 and PyTorch 2.6. These are distinct recorded profiles.

## Repository contents

```text
mechanism/            Shared experimental implementations and historical studies
scripts/release/      Data restoration, fresh-run launcher, released-score verification
scripts/cloud/        Pinned checkpoint downloads and model smoke checks
configs/             Original frozen configs and a separate reproduction config
reproducibility/      Input manifests, sample identifiers, training selections, model hashes
benchmarks/           BBQ summaries and compact per-question predictions
tests/                Protocol and implementation checks
docs/                 Reproduction and maintenance documentation
research/             Original protocols and research records
environments/         Recorded cloud environment metadata
```

Weights, activation caches, raw datasets and SSH administration files stay outside Git.
The manuscript working directory and retired local tools are also excluded.
Original local and cloud result directories remain intact.

## License and citation

Original project code uses the [MIT license](LICENSE). Datasets and checkpoints retain
their upstream terms; see [the source notices](THIRD_PARTY_NOTICES.md). Paper citation
details will be added when available. This is a private preparation copy for a future
code release.
