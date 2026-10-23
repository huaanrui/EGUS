# Evidence mechanism pilot v1 — frozen before new model outcomes

Question: separate increased abstention, changed entity preference and genuine
evidence following. Do not interpret unsuccessful component searches as proof
that no bias mechanism exists. No SAE or head search in this stage.

## Inputs and selection

Qwen3-1.7B, FP16 SDPA, raw zero-shot ABC, last prompt token, block index 18.
Retain historic source probes/direction and thresholds. BBQ dev 352 questions is
the sole selector for CAD beta and matched unknown logit offset. BBQ test 704 and
SynthBias 2000 x 3 permutations are historically inspected exploratory sets.
Native WinoBias/Winogender use frozen cached logits for descriptive diagnostics.

New confirmation: regenerate the prior SynthBias audit reserve candidate pool,
exclude whole occupation pairs used by all 64 prior smoke/audit examples, choose
320 questions in complete gender pairs, balanced ambiguous/clear, by deterministic
hash. The reserve is disjoint from old test/calibration/audit in occupation pair,
template family and gender pair. No confirmation labels or outputs select methods.
Public metadata are used for construction/evaluation only, never rendered.

## Methods and measurements

Baseline; FairSteer alpha1 original-error gate>.5; evidence alpha1.5 expanded-error
times insufficiency gate>.25; direct unknown with same gate and evidence-only gate;
unknown logit offset with same gate, amount 0..16 step.25 selected to match source
dev abstention frequency of evidence steering; three fixed random Gaussian
directions norm-matched to alpha1.5 source vector, same gate, primary ordering only.

Decompose candidate logits into unknown-versus-two-entity log odds and entity odds.
BBQ verified stereotype target is used where present. SynthBias ambiguous rows have
no verified target: alphabetical entity contrast is explicitly not called bias.
Report gold entity margin on sufficient questions, fixed/broken answers, all option
permutations. Direct routing has no invented output distribution.

Evidence instruction is a fixed baseline. CAD uses full-context logits minus
no-context logits, preserving question/options/Answer suffix. Beta grid 0,.25,.5,1;
gated CAD applies when frozen evidence probability<=.25. Select on all source dev
questions by maximum overall accuracy subject to overall>=base and both clear and
ambiguous accuracy>=base-1pp; ties clear accuracy then smaller beta. Beta0 is a true
fallback. Diagnostic fixed-beta results are all reported, never target-selected.
Evidence-only route+CAD composition is deferred: an upstream mistaken abstention
cannot be repaired downstream. CAD is not applied mechanically to native
coreference tasks because deleting their sentence removes the task definition.

## Calibration side experiment

Use cached old BBQ evidence head. Reference moments come from source BBQ fit raw
zero-shot activations, excluding MMLU; fixed tau=.25 and source-dev-selected 2% FPR
threshold are separate. Target adaptation inputs are BBQ-open dev, SQuAD dev and
SynthBias calibration, never target test. Compare no adaptation, scalar moment
alignment, diagonal feature moment alignment and empirical score quantile mapping.
Stress adaptation class prevalence .1/.3/.5/.7 with 100 inputs and 5 seeds. Labels
construct controlled mixtures and evaluate only; algorithms never receive them.
Do not call marginal moment alignment guaranteed calibration or full whitening.
Report AUROC, recall, sufficient FPR, and raw response routing quality if available.

## Inference integrity and statistics

New BBQ baseline/counterfactuals use identical batch4 to CAD; Synth cached baseline
is from the same prior batch4 forward. Numeric smoke verifies alpha0 restoration,
hidden-state index and finite scores. All new rows and implementation/config hashes
freeze before inference. Existing artifacts are read-only. Source-dev selection
freezes before inference on test/confirmation. Caches checkpoint each batch.
Primary contrasts: CAD/base clear accuracy; gated-CAD/base clear accuracy;
evidence/FairSteer overall; same-gate-direct/evidence overall. Family-cluster paired
bootstrap 2000 and sign randomization; apply Holm to confirmatory primary contrasts.
Other beta, random and decomposition results are descriptive. No broad utility or
causal stereotype-removal claim from a positive accuracy change alone.
