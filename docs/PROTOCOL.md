# Experimental protocol

## Data and targets

BBQ is restored from pinned upstream files. The release stores sample UID manifests, the canonical question-family mapping and the five training samples. Train, validation and test families are disjoint. Each seed uses 2,200 training questions: 100 ambiguous and 100 sufficient-context examples within each of 11 category strata. Validation has 352 questions; test has 704 questions from 51 families, balanced between the two conditions. The exact seed IDs are 20261017–20261021.

The evidence target is one for ambiguous contexts and zero for sufficient contexts. It is independent of the model's correctness. The unknown option is recognized from its text using the vocabulary recorded in the training manifest. Predictions do not use the test answer labels. An unrecognized or non-unique unknown expression leaves the base prediction unchanged; the released BBQ examples all have a unique match.

## Representation and gate

The common raw zero-shot prompt is `context + space + question`, followed by three lettered options and `Answer:`. A/B/C next-token scores determine the answer. The readout is the transformer-block output at the final prompt token. We fit unstandardized L2 logistic regression with C=1, L-BFGS, tolerance 1e-4 and at most 10,000 iterations. A convergence warning stops the run.

The strict gate is `p_insufficient > threshold`. The validation threshold maximizes ambiguous-context recall subject to sufficient-context false-trigger rate ≤0.02. Ties favor lower false-trigger rate, then a higher threshold. A never-active gate is included. EGUS selects the recognized unknown option when this gate activates and otherwise keeps the base answer.

## Steering and selection

Each seed supplies 110 ambiguous training questions with a stereotype target, ten from each of the 11 category strata. Appending the unknown versus stereotype-target answer letter gives two states at each block. Their unnormalized mean difference is the steering direction. The edit is applied to the final **prompt** token, which is a different position from the answer token used to estimate the direction.

The baseline profile fits the correctness-based FairSteer detector; the evidence profile fits the evidence detector. FairSteer retains correct base answers as class one and stereotype-target errors as class zero; other errors are excluded. The author classifier's predicted class zero activates its edit. Evidence steering uses the evidence gate.

Every block is considered. Steering searches strengths `[0, 0.1, 0.5, 1, 1.5, 2, 3, 4]`. The selection objective is validation correct count, followed by sufficient-context correct count, smaller strength and earlier block. EGUS selects only its readout block. Each main method chooses independently. Matched-gate controls compare direct selection with steering at the chosen evidence-steering block, and compare EGUS with strength-selected steering at the EGUS block.

All model/seed selections are written and hashed before test inference. Parameter or candidate changes invalidate evaluation. Captures and edited scores carry fingerprints; an incompatible resumed cache raises an error. Runs use full FP16 weights, SDPA, explicit position IDs for left padding and an input limit of 2,048 tokens. Inputs over the limit fail rather than being silently truncated.

## Scores and uncertainty

Accuracy includes unknown as an ordinary scored answer. We report overall, ambiguous and sufficient-context accuracy, alongside trigger and sufficient-context unknown rates. `clear` in machine-readable files means sufficient context; it is retained as a stable artifact field name.

Published standard deviations describe the five training samples on a common test set. Primary accuracy contrasts use 4,000 crossed seed/family bootstrap resamples and 8,000 family sign randomizations. Holm correction is applied separately within the two six-comparison model groups in the paper.

## Execution provenance

The reported Base/FairSteer/KnowBias predictions use the native baseline execution. EGUS and evidence steering retain their completed evidence-profile predictions. On Qwen3-1.7B, one base answer differs but is incorrect in both executions. On Qwen2.5-3B, one base answer changes from correct to incorrect: native baseline accuracy is 72.87%, versus 73.01% for the evidence execution. The other base predictions agree. `base_execution_differences` lists the affected UIDs. `base` and `evidence_base` are separate fields; effects relative to a method's base must use its own execution.

KnowBias uses one identification per model. It is stored in a separate prediction file rather than duplicated as five independent runs. Its unconditional edits have no detector; the retained trigger field follows the original scorer and is not a gate-effect measure.
