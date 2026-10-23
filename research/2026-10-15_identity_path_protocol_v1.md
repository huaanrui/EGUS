# Layer-7 identity information transmission: bounded causal tracing

Declared before this experiment's model inference. All layer indices are zero based.
Qwen3-1.7B FP16/SDPA, deterministic nonthinking generation when applicable.
This is an exploratory mechanistic follow-up to an already observed intervention.
No historical Type2, Type1 or Winogender set is represented as fresh confirmation.

## Fixed source and split roles

The main contrast is exactly the same input: A=baseline, B=half-scale the layer-7
MLP output at the contextual pronoun. The source delta is B minus A, and is not
selected using the current answer/gold. Full male/female donors are only oracle
diagnostics. Rank64, screen32, dev49 reuse the previous family-isolated partitions;
historical146, Type1 160 and Winogender240 are previously inspected retests.
Both AB permutations are bound to a question. Selection uses rank (permutation0),
screen and dev (both permutations); no target-output selection is allowed.

## Transmission metrics, ranking and limits

Use fixed-entity margin: logit(entity alphabetically first)-logit(other entity),
correcting the AB choice order. T=s(B)-s(A). For a receiver H:
R=s(B)-s(B with H restored to A); P=s(A with H transplanted from B)-s(A).
Rank by F=1-mean((R-T)^2+(P-T)^2)/(2*mean(T^2)). Also report signed alignment,
root mean square effects, continuous group bootstrap and behavioral metrics.
If mean(T^2) is negligible, fidelity is undefined (reported null), never made
large by a small denominator. Effects may overlap, cancel, and be nonadditive.
Do not interpret fidelity/alignment as a causal percentage of the whole circuit.
Do not select only repaired/discordant examples. Those subsets are secondary.

## Bounded stages

1. Cache A/B pre-output-projection attention head values at contextual pronoun,
   question pronoun and decision positions in layers8..27. First verify that
   layer7 and prior attention outputs are unaffected and hooks restore exactly.
2. Scan all20layers at question-pronoun and decision receivers, restoring and
   transplanting all16heads at each receiver. Pick four layer-position sites on
   rank64 by F. Rank their individual16heads with two-way patching on rank64.
   Pass twelve candidates to screen32, then four to dev49. Ranking is based on
   transmission, not target accuracy. Missing receivers cause a no-op.
3. Fixed top1/top2/top4 receiver sets and approximate output-norm matched random
   sets; dev selects a set by F. Freeze identities before historical/transfer.
   Trace and report A/B, restoration, transplant, and independent half-scale of
   the receiver heads. Head scaling tests are diagnostics, not a new tuned method.
4. On the fixed four heads, dissect attention with A/B Q/K/V: source-V,
   source-K, source-KV, receiver-Q, all-keys, all-values, changed weights with
   baseline values, and all-QKV. Report changes for the fixed selected set and
   headwise, as appropriate. Under GQA native K/V are shared by pairs of query
   heads; per-query replacement is a controlled incoming-edge intervention and
   is explicitly distinct from editing a shared K/V parameter.
5. Isolate the additive residual edge from source m7(p) to a receiver's K/V:
   baseline receiver residual plus ONLY the source MLP delta at p, then the
   actual RMSNorm, K/V projections, K-norm and RoPE. Use baseline Q and all other
   K/V positions. This clamps intervening computed contributions to baseline.
   It measures this controlled direct edge, not all paths through the head.
   Compare it with the full-run source-KV contrast. On the highest ranked head,
   test a relay via the question-pronoun position: earlier selected attention
   change at that position -> later receiver K/V, with other contributions fixed.
   If no temporally ordered receiver exists, report no eligible relay, do not
   invent an edge or broaden search based on target outcomes.

## Specificity and utility controls

At most64historical pairs chosen by family hash before inference; both orders.
Compare source half-scale, male/female source donor, neutral prefix source donor
("Yesterday, ..." BEFORE pronoun), that neutral delta norm-matched per row to
the male/female delta, and three fixed random deltas of that exact source norm.
No test labels determine perturbation magnitude/direction. Report receiver delta
and margin response per source norm; equal source norm does not imply matched
semantic distribution or equal output norm. Zero neutral deltas remain zero and
are flagged. Oracle donors are not deployment procedures.

Evidence control uses existing contexts plus an explicit reference record naming
one occupation; ask only which occupation the record names. Construct both
records, preserving native context and gender. This is a derived fact-reading
task, not a new natural disambiguation gold or claim about native evidence.
Patch m7 at the record occupation's final subtoken from the other record, and
compare its transmission through the same frozen receivers. Facts appear after
the contextual pronoun, so the pronoun's layer7 activation must stay unchanged.
Run baseline, sourcehalf and fixed receiver-half-scale on literal pronoun,
gender classification, explicit identity binding, role exchange, anonymized
occupations and record reading. Primary utility is accuracy with a descriptive
2pp loss reference; no posthoc claims of statistical noninferiority.

## Generation, audit and interpretation

On64hash-selected historical pairs, actual native chat short-answer generation:
baseline, layer7half, selected receiver-half, layer7half+receiver restoration.
Greedy max32, prefill-only hooks, whole-completion occupation primary; first line
secondary. Receivers are relocated to actual chat prefill positions using offsets.
Transfer both-order closed evaluation on historical146, Type1 160, Winogender240.
Continuous statistics aggregate at family level; 4000 bootstrap draws. All
headwise contrasts exploratory/multiple; selected aggregate contrasts are reported
separately. No paper claim of a bias-specific circuit unless identity effects can
be distinguished from lexical/coreference/fact-reading sensitivity. All source,
code, data and config hashes frozen; predictions and selection independently
recomputed. Caches persist for resumption; source changes require a new version.

