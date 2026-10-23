# Evidence-aware abstention: repeated fits, third model, new confirmation v1

Freeze before new model outputs. Preserve all earlier artifacts. Five fixed seeds
20261017..20261021 resample only the existing BBQ fit pool; family partition,
dev352 and historical test704 remain fixed. This measures training-sample
variability, not five independent test sets or unseen BBQ confirmation. Each seed
uses the same2200 question pool for all methods and all three models. Original
correctness labels exclude counterstereotype errors; actual usable counts disclosed.
Refit original/full correctness and insufficiency heads. Resample the110 direction
examples within each2200 fit pool as well, so variability covers the full pipeline.
Source fit family membership never overlaps dev/test.

Models Qwen3-1.7B L18, MiniCPM5-2B L28, Qwen2.5-3B-Instruct L24; zero-based fixed
approximately two-thirds depth, no layer search. Native FP16 SDPA, raw zero-shot
ABC, last prompt token, batch8 and explicit position IDs, no silent truncation.
The original seed for the first two models reuses hash-pinned old baseline logits,
features, direction and probes; old selected predictions must replay exactly.
Old dev/test counterfactuals reuse only that seed's identical direction. New seed
counterfactuals are actual model forwards. Qwen2.5 repeats the full original-seed
BBQ/utility/option-wording/length-control panel, then four further fits.

All linear probes C1 L2 lbfgs max_iter10000 tol1e-4. Surface standardized logistic,
question-only TFIDF(1,2),max_features3000,min_df2 logistic; output-only standardized
three-feature logistic (unknown log-odds, absolute entity margin, candidate entropy).
Same source2200 data and insufficiency labels, no extra MMLU training. Three head
budgets220/1100/2200 diagnostic; direction110 fixed within seed, hence a head-budget
curve is not total-pipeline data use. Surface/output baselines use2200. Stored
models are immutable after fitting; reporters only load, never refit classifiers.

For each source dev only: select original/full-error/evidence/product thresholds
by maximum insufficiency recall subject to sufficient false trigger<=2%, ties
lower false trigger then larger strict threshold; include true no-op. Apply fixed
alpha1.5, no strength selection. Local FS-style reference original error>.5 alpha1.
Methods: base, local FS-style, full-error direction, product direction/direct,
evidence direction/direct, surface direct, question-only direct, output-only direct,
product unknown offset, always unknown. Match offset0..16 step.25 to product
direction's source-dev unknown rate; it is not best-accuracy logit calibration.
Select both models' five seeds plus the third model's five seeds before any new
target evaluation. No test-selected seed, threshold, layer or method.

NEW confirmation uses640 native SynthBias CSV questions (320 complete gender
pairs), half type1/ambiguous and half type2/clear, three cyclic option permutations.
Audit every earlier prepared JSONL, conservatively treating even draft prepared
samples as exposed. Exclude used source IDs, delexicalized template family IDs,
and whole occupation pairs simultaneously. Also reject inconsistent/native
nonbinary/incomplete gender pairs using the prior eligibility rule. Available
new occupation-pair candidates:294 ambiguous pairs,332 clear pairs before selection.
Hash-based selection favors family and occupation coverage,160 pairs/condition.
Canonical unknown wording 'Cannot be determined' is already in source-fit vocab;
format is the same raw three-choice BBQ rendering. Native type2 gold occ_1 and
native type1 unknown labels are preserved. This is a declared derived choice
protocol, not the authors' original evaluation; no independent human adjudication
or foundation-model pretraining-unseen guarantee. It is fresh to our experiments,
not a new dataset. No confirmation threshold tuning or target calibration.
Primary permutation0; all three orders and gender paired metrics are also reported.

Primary method evidence-only direct, fixed before this confirmation. Per model,
two contrasts on historical BBQ and two on new confirmation: overall accuracy
versus surface-direct; sufficient accuracy versus base. Three models*two datasets*
two contrasts=12, Holm-adjusted family sign-randomization p values. Report95% crossed
bootstrap intervals resampling training seeds and task families independently;
do not count5*704 as independent questions. Also report per-seed results, mean/SD,
min/max, and occupation-cluster intervals on confirmation. A zero p-adjusted or
crossing-zero difference does not prove equivalence/noninferiority. Describe a2pp
sufficient-loss reference explicitly, not a retrospective noninferiority claim.
Secondary comparisons versus base/FS/product and AUROC remain exploratory.

Reuse original-seed two-model robustness only as references; Qwen2.5 runs the same
44*6 option/unknown wording stress and88*4 token-length/evidence control variants.
Other-seed robustness evaluates direct decisions from new heads without claiming
new direction forwards there. MMLU128/source dev64 remains a limited utility panel;
unknown routes have no action when no input unknown option is present.

Before inference freeze config/code/data/model/source cache hashes and snapshots;
before target forwards freeze all15 seed selections. Cache by dataset/seed/vector
fingerprint, checkpoint per batch, support resumption without reselection. Audit
model hooks/zero edits/native letter IDs, source and target isolation, source-only
selections, original-seed replay, every prediction/metric, fitted parameter hashes,
new confirmation exposure against old prepared files, paired clustering and plots.
