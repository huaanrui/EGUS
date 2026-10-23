# Confirmation label interpretation

Before confirmation model inference, inspect native examples and the local
SynthBias README. The authors describe crowdsourced validation with at least75%
agreement, and define occ_1 as the type2 referent. This experiment preserves those
published labels and does not silently replace them with agent judgments.

Native source synthbias:14232 is a qualitative caution: a construction worker
discusses plans with a receptionist, followed by "knowing he would ensure...".
The supplied referent is receptionist. The wording can admit a competing reading
to the subject construction worker; no independent adjudication has been obtained.
The row remains in the frozen evaluation. This is not a model-output-based
exclusion and is not evidence that all type2 labels are wrong.

Interpret sufficient accuracy/FPR as agreement with supplied native labels,
rather than a proof of logically unambiguous evidence. Source-only selection,
exposure isolation, all rows, paired models and all permutations remain unchanged.
