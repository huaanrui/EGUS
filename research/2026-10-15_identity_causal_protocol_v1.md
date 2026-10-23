# Identity interference causal pilot v3 (pre-inference corrections)

The v1 launcher was stopped during model loading, before any new model outcomes.
v2 randomizes explicit identity-fact clause order by pair hash and guarantees all
three final random-layer controls match the selected module/position. No data
selection rule or outcome-dependent parameter changed. v1 artifacts are retained.
v2's integrity smoke found an ndarray-to-tensor conversion error before any data
inference or component selection. v3 fixes it and adds direct hook tests. No v1/v2
experimental predictions are used. Existing directories remain recoverable.

Frozen before new WinoBias development outcomes. Main model Qwen3-1.7B FP16,
SDPA, raw zero-shot prompt, no thinking/chat template, batch8. Stage-specific
checkpoints and hashes permit resumption without silently changing a protocol.

## Data and scope

Use native WinoBias type2 DEVELOPMENT files, previously unused in our inference.
Require complete male/female pairs, identical non-pronoun text and native gold,
one object pronoun (him/her), exactly two occupations. Apply explicit text-review
exclusions for possessives, malformed sentences and unclear dependencies before
outputs. This is an agent text audit, not an independent human annotation.
Release accepted and excluded rows with reasons for independent review.
Family replaces ALL occupations with the same placeholder and pronouns with a
placeholder. Exclude families matching previously inspected WinoBias TEST type2.
Allocate whole families by deterministic hash to fit96 pairs, selection48 pairs,
and all remaining pairs to confirmation; if family sizes require adjustment,
report actual counts. Gold and all task/permutation variants stay together.
Winogender male/female is historically inspected, external exploratory only.

Tasks: native coreference (main); explicit pronoun-gender reading; explicit
occupation-to-gender binding, adding public factual clauses consistent with native
referent; role/evidence swap, exchanging the two occupation strings throughout the
native sentence and changing gold accordingly. Role swap tests tracking changed
roles, not all kinds of evidence use. Question text and answer order are held fixed
within each identity pair. Both answer orders are evaluated. No unknown option.
Identity-fact tasks change their gold under identity swap, so identity invariance
is NOT their success criterion. Preserve accuracy and correct binding instead.
No independent-human-validated or universal social-bias claim.

## Baselines and localization

Baseline, fixed evidence instruction, frozen BBQ FairSteer alpha1/error gate>.5,
frozen evidence steering alpha1.5/expanded-error*insufficiency gate>.25 at block18.
Report trigger rates; no unknown routing on this binary task.

Cache attention-module and MLP outputs, all28layers, at contextual pronoun token
and final prompt token, native coreference fit set, primary order only. Causal scan
swaps gender-counterpart component activations in BOTH directions on ALL fit rows.
Correct-to-incorrect donor is an oracle diagnostic (native gold is used in this
diagnostic only). Rank sites by mean gold-margin recovery on baseline incorrect
members of discordant pairs; report reverse damage and all-pair effects. Retain
top3 sites, at most one per layer, plus3 deterministic matched module/position
random-layer controls. Matched neutral donor appends a neutral time sentence,
preserving identity/evidence/gold; compare top and random sites.

## Deployable intervention and selection

For each top site, estimate unconditional fit activation mean and normalized
male-minus-female mean difference; no confirmation inputs/labels contribute.
Candidates: mean replacement, gender-direction projection removal, component
output scaling, strengths .5/1. Only one site per candidate in this pilot.
Selection on selection split, BOTH answer orders, native, reading, binding and
role-swap tasks. Feasible: all four task accuracies >= baseline-2pp. Primary
objective average per-order gender-pair both-correct on native task; tie native
accuracy, smaller strength, no-op. No-op is available; do not force a winner.
Random sites use the same selected operation/strength, not separately tuned.
Do not equate nonsignificant control losses with demonstrated noninferiority.

## Frozen confirmation and short generation

Freeze selected intervention before confirmation inference. Compare baseline,
selected, matched random sites, instruction, FairSteer, evidence steering. Report
native accuracy, per-order identity flip, both-correct, pro-anti gap, and control
accuracies. Aggregate permutations WITHIN family; paired family cluster bootstrap
4000 and sign randomization. Confirmation primary comparison is selected/base
native both-correct; other contrasts descriptive. Recompute using occupation-pair
clusters as sensitivity check. Report variation across answer order separately.

Run actual greedy no-option short-answer generation, baseline and selected, on
confirmation native contexts with a fixed instruction to output only occupation.
Selected component position/strength is unchanged; prompt token alignment changes
only to match contextual pronoun/final prompt semantics. Apply during prompt
prefill only, not later generated tokens. Max12newtokens, no forced choice.
Strict whole-string occupation matching after only whitespace/case/punctuation
and optional leading 'the'; longer explanations count unparsed. Exploratory
secondary unique-occupation extraction reported separately. No model judge.

## Integrity and limits

Check hook restoration, exact self patch/no-op, finite logits, token offset span
alignment, metadata exclusion from prompt, fit/dev/confirmation family isolation,
source hashes, per-stage configuration hashes. Save item predictions and raw
generation, and selected configuration. Need independent annotation and another
generator model before broad mechanism/generalization conclusions. This is a
small pilot with limited power for small control-accuracy losses.

