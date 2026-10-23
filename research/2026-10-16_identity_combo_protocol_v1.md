# Frozen bounded combination and mechanism pilot

The prior WinoBias type2 confirmation146 and Winogender240 have been inspected.
They are historical exploratory retests, NOT fresh confirmatory evidence. No
outcomes from these partitions select sites, groups, strengths or combinations.
Rank/screen/dev also come from the previously used development pipeline; no claim
that continued reuse of this study provides an independent replication.

## Data and hierarchy

Split the old fit96 by WHOLE family hash into rank >=64 pairs and remaining screen
pairs. Rank learns neuron groups and gender-difference subspaces. Screen evaluates
ALL predefined candidates on native coreference, literal pronoun reading and
explicit identity binding. Retain6 combinations and4 other candidates ranked by
screen feasibility (each control/native accuracy >=base-2pp), both-gender correct,
native accuracy, then smaller intervention burden. Always include baseline and
the original layer7/identity/MLP half-scale. Selection uses the original dev49 on
all5 tasks: native, role-swap, binding, masculine/feminine classification, literal
pronoun reading. Feasibility is task accuracy>=baseline-2pp; choose maximum native
both-correct, then accuracy, smaller burden, no-op. Freeze best single/noncombined,
best combination and overall winner before ANY evaluation inference. No feasible
combination is a valid outcome; do not force selection of a harmful combination.

New external reserve: up to160 complete male/female pairs from native WinoBias
TYPE1 DEVELOPMENT, previously unused by our runners (old runners loaded TEST).
Require one contextual gender pronoun, identical non-pronoun text and gold,
exactly2 occupations. Exclude families matching old native type1/type2 TEST or
any old fit/dev/confirmation/Winogender rows. Hash sample whole families before
new outcomes. Native type1 gold is retained; this is a contextual/commonsense
transfer test, not a claim of uniformly unambiguous explicit evidence. Publish
structural exclusions and do not silently relabel native examples as unknown.

## Candidate space (46 total, including no-op)

1 no-op; layer7 identity MLP removal strengths .25/.5/.75; single half-scale at
MLP identity layers0/5/6/8/9, attention decision layer21; half-scale layer7 MLP at
decision, subject occupation, object occupation, question pronoun positions.
Anchor layer7 identity half-scale plus each of6 other modules at .25/.5 removal;
anchor plus its decision/question-pronoun position .25/.5 removal;
MLP identity pair6+8 at .25/.5 removal; triple6+7+8 at .25/.5 removal.
Layer7 gated MLP channels: top32/128/512 by rank-only paired absolute activation
difference times down-projection column norm, .5/1 removal;128 channels matched
to top128 in activation RMS/down-column norm, .5/1 removal.
Rank-only uncentered SVD of male-female layer7 output differences: rank1/rank4
gender-difference subspace half removal and orthogonal-complement half removal.
Channel/group gender sensitivity is NOT a bias-specific score. Positions refer to
the last subtoken of the matching public span. Missing question-pronoun scopes
on literal/classification tasks are skipped explicitly, never remapped.

## Why layer7 diagnostics

Capture at rank native pronoun positions, all28layers: residual before MLP,
MLP output and residual after MLP. Compare raw and the existing fixed instruction.
Report paired male-female differences and norm identity:
||dh+dm||^2-||dh||^2=2 dh.dm+||dm||^2. This measures amplification of accessible
identity differences, not proof of stereotypical semantic content.
Channel rankings/subspace intervention/position controls test narrower hypotheses.

The prior neutral donor appended a time sentence AFTER the pronoun. Under causal
attention the pronoun cannot see that suffix; its zero effect is a trivial control,
not evidence of bias specificity. Correct donor prepends 'Yesterday, ' and lowers
the original initial capital. On screen, compare counterpart and prefix-neutral
donors at layer7 pronoun MLP; additionally norm-match neutral perturbations to the
paired-gender perturbation. No gold enters donor magnitudes. Both directions, all
pairs, discordant-member repair and damage are reported. These are diagnostic
oracle patches; deployment uses only frozen current-input interventions.

## Evaluation

Historical type2 native/all5 controls, historical Winogender native, fresh type1
native: baseline, old single, best selected single, selected combination (if any),
instruction and instruction+combination. Include deterministic random-layer copy
of combination where feasible; same operations/positions/strengths. Run individual
constituents of selected combination on native historical type2 to estimate
nonadditive changes; interaction in final scores is not a verified circuit edge.
Whole-family paired bootstrap4000; report occupation-pair sensitivity for primary
both-correct combo-minus-old-single contrasts. Do not count option permutations
as independent samples. All historical contrasts exploratory. New type1 primary
contrast is selected-combination versus old single both-correct; if infeasible,
report that no combination was selected rather than testing a forced winner.

Native chat no-option generation on historical type2, no thinking, greedy32,
prefill-only interventions. Exact whole-completion occupation matching primary;
fixed conservative first-answer-line extraction secondary. Same frozen configs;
no generation-based tuning. Compare baseline/old single/best single/combo/fixed
instruction/instruction+combo, deduplicating identical configurations for compute.
No broad open-generation or cross-model claim from this single-model pilot.

## Integrity

Freeze source/config/protocol/data hashes before inference; fitted parameters and
candidate specs hash before screening; selection hash before retest/new transfer.
Check multi-edit no-op, exact hook restoration, simultaneous different-position
edits, gated-neuron hook shape and full-channel equivalence, span alignment,
subspace complement identity, no labels rendered, partition/family isolation.
Save all screened/selected configs, cached item logits, raw generations, exclusions,
and recomputed statistics. Existing experiment artifacts remain unchanged.

