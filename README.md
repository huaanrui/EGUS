# Evidence-Gated Unknown Selection (EGUS)

Paper-aligned code for evidence-aware response selection on BBQ. EGUS reads a model's hidden state with a linear evidence classifier and selects the unknown option when the evidence gate activates.

This repository has one implementation shared by all six checkpoints. The main configuration searches every transformer block. Steering also searches strengths in `{0, 0.1, 0.5, 1, 1.5, 2, 3, 4}`. Configuration selection uses the validation set.

## Recompute the reported tables without a GPU

From this checkout:

```bash
python -m pip install -e .
python -m egus verify --statistics --output runs/verification.json
```

The release includes question identifiers, labels and compressed predictions, without distributing the BBQ question text. This command checks all released prediction hashes, recalculates accuracy and verifies the main clustered comparisons. It does not rerun model inference.

## Run the experiments

Install a CUDA-enabled PyTorch build for your machine, then:

```bash
python -m pip install -r requirements/paper.txt
python -m pip install -e . --no-deps
python -m egus prepare-data
python -m egus download-models --models qwen3 --models-root models
python -m egus run --config configs/paper.toml --models qwen3 --models-root models
```

Omit `--models qwen3` to select all checkpoints. Downloads use pinned revisions through hf-mirror by default; `--endpoint https://huggingface.co` selects the official endpoint. Each profile has its own output directory. Existing outputs are resumed only when the configuration, data and source fingerprints agree.

| Profile | Paper methods |
|---|---|
| `configs/paper.toml` | EGUS, evidence-gated steering and matched-gate comparisons |
| `configs/author_baselines.toml` | FairSteer with the authors' cuML functions, and KnowBias with author-source neuron attribution/enhancement |

Both profiles contribute to the paper tables. The baseline profile requires a separate Linux/CUDA environment. See [baseline details](docs/BASELINES.md) and [installation](docs/REPRODUCING.md).

## Reported main results

Test accuracy (%), mean over five seeds. Full precision, conditional scores and selected configurations are in `benchmarks/main/summary.json`.

| Method | Qwen3-1.7B | MiniCPM5-2B | Qwen2.5-3B | Qwen3-8B | Qwen3.5-4B | Qwen3.5-9B |
|---|---:|---:|---:|---:|---:|---:|
| Base | 56.68 | 43.47 | 72.87 | 75.43 | 74.72 | 77.41 |
| KnowBias | 56.68 | 47.87 | 72.59 | 75.43 | 74.72 | 78.98 |
| FairSteer | 65.63 | 75.23 | 74.23 | 77.53 | 78.10 | 78.38 |
| Evidence steering | **79.57** | 83.35 | 80.48 | **82.30** | 83.89 | 79.52 |
| EGUS | 79.12 | **83.98** | **81.11** | 81.65 | **90.09** | **92.10** |

KnowBias uses one deterministic identification run per model. Other fitted policies use five seeds. Base refers to the baseline execution; the evidence execution's base predictions are also retained because two checkpoints have one changed answer each. See [protocol](docs/PROTOCOL.md).

These are BBQ multiple-choice results. The implementation assumes a recognizable unknown option; it does not establish open-generation or cross-dataset effectiveness.

## Layout

```text
egus/                 Shared data, model, gate, selection and evaluation code
egus/baselines/       Adapters for pinned author functions
configs/              Two paper experiment profiles
reproducibility/      Dataset/model/source manifests and artifact hashes
benchmarks/main/      Paper predictions, selected configurations and summaries
third_party/FairSteer/ Unmodified MIT-licensed source functions
tests/                Protocol, cache integrity and released-table checks
docs/                 Reproduction instructions and mapping to the manuscript
```

See [reproduction](docs/REPRODUCING.md), [protocol](docs/PROTOCOL.md), [paper mapping](docs/PAPER_MAP.md) and [validation](docs/VALIDATION.md). Run `python -m unittest discover -s tests -v` for CPU checks. Models and prepared dataset text are fetched separately and are excluded from Git. KnowBias source is fetched on demand and is not redistributed here.
