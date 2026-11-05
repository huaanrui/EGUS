# Reproducing the paper

Run commands from the repository root. The README gives the main workflow.

## Environments and baselines

EGUS and evidence steering use `requirements/paper.txt`. The recorded evidence runs used Transformers 5.18.0 and scikit-learn 1.9.1, with PyTorch 2.14.1+cu132 for the three smaller checkpoints and PyTorch 2.6.0+cu124 for the larger ones. Install an appropriate CUDA PyTorch build separately.

FairSteer and KnowBias require a separate Linux Python 3.12 environment. The completed baseline run used PyTorch 2.6.0+cu124 and cuML 25.06:

```bash
pip install -r requirements/author-linux.txt
pip install -e . --no-deps
python -m egus fetch-sources
python -m egus run --config configs/author_baselines.toml --models-root models
```

FairSteer calls the authors' `train_probes`, `mean_difference` and activation-editing functions at revision `0545cdbaa71b3301030accf1b98130820e08026f`. MIT-licensed source files are included unchanged. There is no scikit-learn fallback for its cuML classifier. Training uses the shared BBQ pool without additional MMLU questions; each checkpoint searches every block and the configured strength grid. The original edit callback is applied per active batch row because it checks only its first row.

KnowBias calls the attribution and enhancement functions at revision `ccfd3daea0067707cbda6238839206c194794881`. Its sources are fetched on demand because the pinned repository has no explicit license. Five concepts expand to 45 questions across gender, race and religion. Attribution uses 20 integration endpoints, batch four and threshold 0.1. Consistency candidates are 0.05/0.1; enhancement strengths are 1/1.5/2/3.5/5. One is the no-op candidate. Parameters are selected by validation accuracy, then sufficient-context accuracy, proximity to strength one and consistency. Identification is performed once per model.

KnowBias attribution uses BF16 and BBQ scoring uses FP16. It targets the MLP gate projection at the last prompt token, with Qwen3.5 module paths adapted to its architecture. The upstream `evaluate_know_new.txt` is absent; concept prompts use `Answer the following question with only yes or no.\nQuestion: {question}` with thinking disabled. These are implementations under our protocol, not reproductions of the original papers' reported scores.

## Data and evaluation

`data/manifest.json` fixes the upstream inputs and sample IDs. The five training seeds are 20261017–20261021. Each has 2,200 questions: 100 per evidence condition in each of 11 category strata. Validation has 352 questions; test has 704 questions from 51 families, evenly split between ambiguous and sufficient contexts. Train, validation and test families are disjoint. `data/training.json` preserves training order, direction samples and the unknown-expression vocabulary.

BBQ uses a raw zero-shot prompt: context and question, three lettered options, then `Answer:`. A/B/C next-token scores determine the answer. The model runs without quantization, using FP16, batch eight, explicit left-padding position IDs and a 2,048-token limit. Standard attention uses SDPA; Qwen3.5 retains its native hybrid architecture. Overlength inputs raise an error.

The evidence probe reads each block's final prompt-token state. It uses unstandardized L2 logistic regression with C=1, tolerance 1e-4 and at most 10,000 iterations. The strict gate `p > threshold` maximizes validation ambiguous recall under sufficient-context false triggering at most 0.02. Ties favor lower false triggering, then a higher threshold. The unknown option is recognized from training-derived expressions, without reading test labels.

Each steering direction uses 110 ambiguous training questions, ten per category. It averages the difference between states at an appended unknown answer letter and an appended stereotype-target letter. Inference adds the direction at the last prompt token. FairSteer's detector keeps correct base predictions as class one and stereotype-target errors as class zero; other errors are excluded. Class zero activates the edit.

Selection maximizes validation correct count, then sufficient-context correct count, smaller strength and earlier block. EGUS selects only its readout layer. Matched-gate comparisons change the action at either the EGUS-selected or evidence-steering-selected layer. Strength at the EGUS layer is also selected on validation data.

## Outputs and resuming

The two profiles write to `runs/paper/` and `runs/author_baselines/`. Each stores its plan, captured states, validation candidates, selected configurations, predictions and metrics. Per-seed predictions are `MODEL/seed_SEED/predictions.npz`; KnowBias writes `MODEL/knowbias_predictions.npz`. The summary command consumes these generated files. It requires both profiles to be complete for all six checkpoints and uses the corresponding prepared test labels.

All selected models finish validation selection before test inference. To run phases separately:

```bash
python -m egus run --config configs/paper.toml --stage fit
python -m egus run --config configs/paper.toml --stage evaluate
```

Keep arguments identical across phases. To use a different model store or output directory, add `--models-root PATH` or `--output PATH`. Captures and completed candidates resume only when source, data and configuration fingerprints match. Use a fresh output directory after changing code or settings. Model weights and all-layer activation caches require substantial disk space.

## Tables and statistics

`python -m egus summarize` recomputes metrics from both runs. Use `--evidence-dir`, `--baseline-dir`, `--data-dir` and `--output` to select other locations. The six sections of `results/paper_tables.json` correspond to overall accuracy, conditional accuracy, matched-gate comparison, layer/strength selections, KnowBias selections and statistical contrasts. Models follow the README order; main-table fitted methods alternate mean and standard deviation. `ambig` and `clear` in generated summaries mean ambiguous and sufficient contexts.

Standard deviations describe five training samples on a shared test set. Confidence intervals use 4,000 crossed seed/family bootstrap samples, with 8,000 family sign randomizations. Holm correction is applied to six contrasts within each three-model group. KnowBias and matched-gate results are descriptive.

The reported Base row comes from the baseline execution. The evidence execution differs on one Qwen3-1.7B answer (incorrect in both runs) and one Qwen2.5-3B answer (correct in the evidence run, incorrect in the baseline run). Qwen2.5 base accuracy is consequently 73.01% versus 72.87%. Other checkpoints agree. The summary retains `base` and `evidence_base` separately. Hardware and numerical environments can change individual predictions in fresh runs.
