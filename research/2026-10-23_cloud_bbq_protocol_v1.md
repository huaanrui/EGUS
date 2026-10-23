# Cloud BBQ extension, frozen protocol v1

The purpose is to test whether the local EGUS pattern persists on Qwen3-8B,
Qwen3.5-4B and Qwen3.5-9B. This is a new model extension on the same historical
BBQ evaluation questions, not a new unseen-dataset confirmation.

Reuse exactly the five existing 2,200-question fitting resamples (seeds
20261017--20261021), development 352, and evaluation 704. Reuse each seed's
110 ambiguous direction examples and nested head budgets 220/1100/2200. Verify
UID order, source indices, category/condition balancing, unknown vocabulary,
and family isolation. A separate 64/128-question MMLU utility panel is retained.
No SynthBias evaluation is performed by this campaign.

Main settings match the local repeated-fit protocol: raw zero-shot prompt,
native next-token A/B/C/D IDs, no chat template or generated reasoning,
FP16, SDPA, batch eight, explicit padding-aware position IDs, last prompt
token readout/edit, no truncation above 2,048 tokens. The fixed zero-based
blocks are 24/36 for Qwen3-8B and 21/32 for both Qwen3.5 models. These choices
use approximately two-thirds depth, before model task outputs are inspected.
Qwen3.5 uses its native conditional-generation wrapper with text-only input.

Reuse the existing Engine.fit_seed, policies, evidence_story fitting, threshold
and action functions unchanged. L2 logistic regression uses C=1, L-BFGS,
max_iter=10000, tol=1e-4, four CPU threads, and no hidden-feature standardization.
The original correctness probe filters out non-target errors. Evidence and
full-error probes use all fitting questions. Surface, question-only and output
controls use the same evidence labels and original feature definitions.

Development thresholds maximize ambiguous recall subject to sufficient FPR<=2%,
with strict greater-than comparison and the original tie rules/no-op candidate.
Estimate unnormalized mean directions from appended correct-unknown versus
stereotype-target answer tokens; apply them at the final prompt token.
Local FairSteer-style uses filtered error probability>0.5 and alpha=1.
Evidence and product steering use alpha=1.5. Product logit offset matches the
development unknown frequency of product steering with the original grid/ties.
EGUS shares the evidence gate exactly with evidence steering. No layer/strength
search or target-based reselection is introduced in this extension.

Before inference freeze config, code, data and checkpoint hashes. Freeze all
15 model/seed selections before ANY evaluation-set forward. Cache each batch
with the plan/dataset/vector fingerprint and verify UID order on reuse. No
legacy small-model features or predictions are reused for the new models.
The local small-model aggregate reference is used only for reporting.

Report all original policies on BBQ: base, local FairSteer-style, expanded-error
steering, product steering/direct/offset, evidence steering, EGUS, surface-direct,
question-direct, output-direct, always-unknown. Report overall, ambiguous and
sufficient accuracy, ambiguous bias score, sufficient unknown/trigger rate,
gate frequency and evidence AUROC. Save per-question predictions and per-seed
metrics; aggregate as mean/sample SD on the shared questions. Include head-budget
diagnostics and the limited MMLU panel. Direct policies take no action on MMLU
because it has no designated unknown option; this is a no-action control.

For this new extension predeclare twelve within-model contrasts, across three
models: EGUS versus local-FS, evidence-steering and surface-direct on overall
accuracy; EGUS versus base on sufficient accuracy. Use the existing 4,000 crossed
seed/family bootstrap and 8,000 family sign-randomization draws, with Holm over
these twelve contrasts once ALL model runs finish. This family is separate from
the historical manuscript's twelve-comparison family. Do not reinterpret a
nonsignificant contrast as equivalence. Cross-scale comparisons are descriptive.

The cloud uses Python 3.12.9 and PyTorch 2.6.0+cu124 because of its CUDA 12.4
driver, versus local Python 3.13 and PyTorch 2.14.1+cu132. Transformers 5.18.0,
NumPy 2.5.3, SciPy 1.18.1 and scikit-learn 1.9.1 match. Record these differences.
Optional FLA kernels are absent; Qwen3.5 uses the verified PyTorch reference
implementation. Runtime and peak memory are diagnostic, not a controlled
deployment-efficiency comparison.
