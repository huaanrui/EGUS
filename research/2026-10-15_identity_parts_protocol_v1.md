# Signed component parts and local reprocessing: fixed exploratory protocol

Declared before this round's model inference. Local Qwen3-1.7B FP16 SDPA. Layers
and heads zero based. Historical Type2/Type1/Winogender and control families have
already been inspected; all target results remain exploratory, not fresh confirmation.
Old implementation/data/output assets are read-only; new artifacts have a separate
version. Rank64/screen32/dev49 and target partitions retain family isolation.

## Two different signs

1. Source transmission: same-input A=baseline, B=layer7 contextual-pronoun MLP
   half-scale. T is fixed-entity margin(B)-margin(A), independent of gold/AB order.
   Restoring/transplanting another component may align with or oppose T. This
   identifies facilitating/compensating contributions to this source intervention,
   NOT inherently good/bad bias neurons.
2. Identity-gap contribution: D=s(male)-s(female) with identical native gold.
   For a centered unit scale c_j, first-order derivative of D^2/2 is
   D * [g_mj*(u_mj-mu_j)-g_fj*(u_fj-mu_j)], where g=ds/du. Positive units locally
   enlarge the paired margin gap; negative units locally counteract it. Smaller
   gaps can still mean both answers wrong, so correctness and utility are required.
   Native him/her differences retain lexical/grammar confounds; no social-bias
   specificity is inferred solely from this objective.

## Coarse localization and destination tracing

Scan attention/MLP OUTPUTS at contextual-pronoun positions in layers8..12 (10)
and decision positions in layers14..20 (14). Same-input two-way source mediation,
all24sites reported, positive and negative alignments retained. Capture destination
V(context pronoun) for L11H13/L13H6, K(decision/option B) for L17H1/L21H1, and
Q(decision) for L21H1. Report vector restore/transplant fidelity/alignment as well
as answer margins; temporally inaccessible source/destination pairs must be no-op.
Pick one source-position MLP and one terminal-position MLP by absolute two-way
answer alignment on rank64. Do not force an opposing module if none is supported.
Also analyze the two previously frozen terminal head outputs L17H1/L21H1.

## Unit ranking and exact sign checks

MLP units are the6144 gated intermediate channels before down_proj; head units
are128 pre-o_proj coordinates, not uniquely meaningful biological/semantic neurons.
Rank-only A/B activations plus endpoint gradients of semantic margin. All model
weights frozen; only one intermediate tensor requires grad, batch2. Source
attribution = mean[T * (u_B-u_A) * (g_A+g_B)/2]. Keep strongest positive and
negative groups (64 MLP /16 head coordinates), skip nonexistent signs, and record
directional stability by rank family blocks. Gradients are screening approximations.
Validate with exact group restoration/transplant on screen and historical64;
do not assume attribution sign survives nonlinear interactions. Record accuracy,
gold margin, both-correct and identity-gap alongside transmission.

Deployment groups learned only on rank: identity-gap positive, negative, largest
absolute, approximate RMS/down-column-norm matched controls. For each of4unitsites:
positive damping .25/.5; negative amplification .25/.5; signed both .25/.5;
reverse signed .5; unsigned absolute damping .5; matched damping .5; whole centered
damping .5; whole raw scaling .5. Centered edit u'=mu+(1-lambda)*(u-mu) in selected
coordinates; positive lambda damps, negative lambda amplifies. All choices have
fixed fit means; no current test labels/paired donors at inference.

## Orientation-only Q/K parts

Fields: Q(decision), K(decision), K(option B last entity token), each at terminal
heads L17H1/L21H1. Learn rank1/rank4 uncentered SVD of male-female differences
AFTER q/k-norm and BEFORE RoPE, plus rank-only field mean. Project centered
variation with lambda .25/.5, preserve each input vector's original L2 norm; also
rank1 amplification -.25, rank1 .5 without norm preservation, and a fixed random
tangent rotation matching the learned rank1 edit's exact norm and edit distance.
Thus changes of attention temperature can be distinguished from orientation.
Q edits affect that query head at the decision receiver. K edits affect BOTH
query-head incoming edges sharing the native KV group, only at decision receiver.
They are receiver-restricted attention interventions, not native shared K parameter
edits; prefilling does not alter stored KV cache. Missing option-B scope in chat
is no-op and intervention coverage is reported.

## Bounded behavioral selection

Screen all fixed candidates (baseline, old L7, whole-four-head comparator, unit
and Q/K candidates) on screen32 coref and ACTIVE gender classification. Feasible
means each accuracy no worse than baseline-2pp. Rank feasible by coref both-correct,
accuracy, lower edit burden. Keep eight feasible candidates with site diversity
(at most2per unit/field), plus mandatory baseline/oldL7/whole-four-head benchmark.
Add at most two combinations: best feasible source-unit+best feasible terminal-part,
and oldL7+best feasible fine-part, chosen only on screen. No target tuning.

Dev49 evaluates frozen shortlist on coref, active gender classification, active
literal reading, active identity binding, role exchange and active record reading.
Select max coref both-correct then accuracy/lower burden subject ALL accuracies
>=baseline-2pp; no-op allowed. Whole-four-head is diagnostic even if infeasible.
Freeze selected single/combined parts before target evaluation. Report all dev
candidates, including failures, not just the winning configurations.

## Retests, controls and generation

Target coref on historical146, Type1 160 and Winogender240, both option orders:
baseline, oldL7, overall selected, best feasible unit, best feasible Q/K, best
feasible combination, whole-four-head benchmark (deduplicate identical specs).
Historical64 active utilities/role/anonymous occupations with baseline, oldL7,
selected and whole-four-head. Exact signed donor checks use rank-defined groups
only, never become deployment methods. Selected Q/K actual norm/edit-distance
controls and matched unit budgets all reported. No conclusions of noninferiority
from point constraints alone. Family bootstrap4000, explicit exploratory/multiple
contrast status; no p-value optimization.

Actual chat short generation on fixed64historical pairs, greedy max32 nonthinking,
prefill-only: baseline, oldL7, selected and whole-four-head. Whole completion exact
occupation primary, first-line secondary. This is short-answer format transfer,
not unrestricted free continuation. No second generator model in this round.

## Verification

Freeze code/config/data/protocol and learned parameter hashes; no in-place repair
after scientific inference. Zero-strength, self-patch, channel/full-output linear
equivalence, head/QKV equality, missing-position no-op, norm/distance invariants,
gradient finiteness and hook restoration checks. Independent reconstruction of
candidate ranks/selection and every recorded behavioral metric; all source inputs
unchanged. Small FP16 effects near numerical noise are not interpreted strongly.

