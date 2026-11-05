# EGUS: Evidence-Gated Unknown Selection

Code for **EGUS: Evidence-Gated Unknown Selection for Bias-Sensitive Question Answering**.

EGUS uses a linear classifier to detect insufficient evidence from hidden states. When the gate activates, it selects the unknown option; otherwise, it keeps the model's answer.

## 1. Install

Use Python 3.12 and a CUDA-enabled PyTorch installation:

```bash
pip install -r requirements/paper.txt
pip install -e . --no-deps
```

FairSteer and KnowBias use a separate Linux environment with `requirements/author-linux.txt`. See [REPRODUCING.md](REPRODUCING.md) for baseline setup.

## 2. Prepare data and models

```bash
python -m egus prepare-data
python -m egus download-models --models-root models
```

Downloads use hf-mirror by default. Add `--endpoint https://huggingface.co` to use Hugging Face directly. Data splits and model versions are specified in `data/` and `configs/`.

## 3. Run experiments

```bash
# EGUS and evidence-gated steering
python -m egus run --config configs/paper.toml

# FairSteer and KnowBias, in the baseline environment
python -m egus fetch-sources
python -m egus run --config configs/author_baselines.toml
```

For one checkpoint, add `--models qwen3`. Supported IDs are `qwen3`, `minicpm5`, `qwen25`, `qwen3_8b`, `qwen35_4b`, and `qwen35_9b`. All layers are searched; steering strengths are `{0, 0.1, 0.5, 1, 1.5, 2, 3, 4}`. Parameters are selected on validation data.

## 4. Summarize results

After both profiles finish all six models:

```bash
python -m egus summarize
```

Experiments generate predictions and metrics under `runs/`. The summary command writes `runs/tables/summary.json` and `runs/tables/tables.json`, including conditional accuracy, matched-gate comparisons and clustered statistics. Generated predictions are not included in this repository.

## Results

BBQ test accuracy (%). FairSteer, E-steer and EGUS are averaged over five seeds; KnowBias uses one identification run per model.

| Method | Qwen3-1.7B | MiniCPM5-2B | Qwen2.5-3B | Qwen3-8B | Qwen3.5-4B | Qwen3.5-9B |
|---|---:|---:|---:|---:|---:|---:|
| Base | 56.68 | 43.47 | 72.87 | 75.43 | 74.72 | 77.41 |
| KnowBias | 56.68 | 47.87 | 72.59 | 75.43 | 74.72 | 78.98 |
| FairSteer | 65.63 | 75.23 | 74.23 | 77.53 | 78.10 | 78.38 |
| E-steer | **79.57** | 83.35 | 80.48 | **82.30** | 83.89 | 79.52 |
| EGUS | 79.12 | **83.98** | **81.11** | 81.65 | **90.09** | **92.10** |

The paper tables are recorded in [results/paper_tables.json](results/paper_tables.json). E-steer denotes evidence-gated steering. These results concern BBQ with an explicit unknown option.

## Files

- `egus/`: method, baseline adapters and experiment code.
- `configs/`: experiment settings and pinned model/author versions.
- `data/`: sample manifests and family mapping; dataset text is downloaded separately.
- `results/`: paper table values.
- `third_party/`: unmodified FairSteer source and license. KnowBias is downloaded on demand.
- `tests/`: protocol and integrity checks (`python -m unittest discover -s tests -v`).

Implementation details are in [REPRODUCING.md](REPRODUCING.md). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and terms.
