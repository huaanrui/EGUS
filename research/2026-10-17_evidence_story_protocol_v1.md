# Evidence-aware decisions: matched-budget and cross-model supplement v1

Declared before new inference. Existing frozen experiments are read-only.
Question: separate evidence identification, the action after identification, and
dependence on superficial input cues. This is not a new universal-bias-circuit
claim or a strict author reproduction of FairSteer.

Models: Qwen3-1.7B block18 and MiniCPM5-2B block28 (zero-based), fixed approximately
two-thirds depth without layer selection. Native FP16/SDPA, raw zero-shot ABC,
last prompt token, batch8 with explicit position_ids and left padding. No thinking
or chat template. Original model probability over ABC is the closed-task score.

Reuse the old family assignment. BBQ dev352/test704; training2200 native questions,
100/category/condition, from fit only. Nested budgets220/1100/2200; same question
IDs for both models and all probes. Evidence and full-correctness heads use every
row; the original FairSteer-style head excludes predictions that are neither gold
nor stereotype target, and its actual usable budget is disclosed. No MMLU rows
are silently added as perfectly correct. Surface and question-TFIDF controls use
identical fit rows. All linear classifiers use sklearn L2 lbfgs C1. This controls
sample budget, not the nature of label information. Evidence needs extra labels.
Vectors use110 ambiguous rows within these2200; mean hidden state after correct
unknown answer minus after stereotype answer. All methods share that vector.

Four signals: original error, full error, evidence insufficiency, and product of
full error and insufficiency. Select each threshold using source dev only:
maximum insufficiency recall subject to sufficient-question false trigger<=2%;
ties lower false trigger then larger threshold. All unique dev scores plus a
true no-op are candidates, strict score>threshold. This selection is independent
of intervention outcomes. It is a source-dev point constraint, not a statistical
or target-domain guarantee. Each gate is crossed with direction alpha1/1.5,
direct unknown, and unknown-logit offset. Offset selected on dev to match gated
direction's unknown frequency, grid0..16 step.25. Main direction alpha1.5 fixed.
Also retain fixed original gate>.5 alpha1 and product>.25 alpha1.5 as diagnostics.
Direct and offset locate unknown from fit-learned answer strings, never test
metadata; absence means no applicable direct action. Utility direct no-op does
not establish broad capability preservation. For direction, also evaluate source
dev64/test128 MMLU; original and expanded probe training labels use actual ABC
predictions. No final-test choice among gates, budgets, strengths or models.

Robustness:88 original ambiguous/disambiguated pairs from held-out test families,
8/category selected by hash with preference for broad family coverage and both
stereotypical/counterstereotypical correct entities. Pair requires byte-identical
question/options and clear context starting with ambiguous context. Conditions:
original insufficient, original sufficient, insufficient plus irrelevant neutral
padding with exactly the sufficient FULL PROMPT token length for each tokenizer,
and sufficient plus the same padding. Padding contains only ordinary objects,
no identities, answer choices or task answers. It is an artificial diagnostic,
not independently annotated natural data. Report paired head-score responses,
recall/FPR, model unknown margin, and prediction accuracy, separately by condition.
No test-family row or derivative is used to train/select. Native data may have
pretraining or historical experiment exposure; no pristine confirmation claim.

Stress44 sampled test questions (2/category/condition), three cyclic option
orders crossed with original/fit-learned alternate unknown wording. Native
semantic label/target/unknown position remapped exactly. Report all6 versions,
head consistency and prediction invariance. Do not select best permutation.

Primary exploratory contrasts per model: product-steer versus fixed local FS;
same-product-gate direct versus product-steer; product-steer versus expanded-error
steer; product-steer sufficient accuracy versus base. Two models = eight contrasts,
family-cluster paired bootstrap2000 and family sign-randomization with Holm across
eight. Secondary budgets, other gates, stress and artificial padding are diagnostic.
Matched mask means identical trigger rows, not just equal average trigger rate.
Do not infer model use from probe decodability, universal fairness from abstention,
or noninferiority from a nonsignificant capability difference.

Freeze source/data/config/model hashes before inference; checkpoints per batch;
freeze both-model parameters/thresholds before test and diagnostic inference.
Smoke: zero intervention/hook removal exact, hidden hook matches hidden_states,
batch-versus-single numerical diagnostic, unknown discovery and metadata isolation,
exact token-length match, no split-family overlap, finite values and no truncation.
Independent audit replays head fits, dev thresholds, offsets, prediction policies,
metrics and data hashes. Preserve old source assets and report exposure explicitly.
