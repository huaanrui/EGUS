# BBQ protocol

Five 2,200-question training resamples share 352 development questions and 704
evaluation questions. Fit samples are balanced across eleven categories and both
evidence conditions. Canonical families are separated between partitions. The 704
evaluation questions were inspected during earlier research, making this an
exploratory evaluation set, including for the additional model campaign.

Seeds `20261017` through `20261021` label training resamples, rather than independent
evaluation sets. Nested probe budgets are 220, 1,100 and 2,200; each direction uses
110 ambiguous fit questions.

## Models and representation

| Checkpoint | Zero-based block | Decoder blocks | Width | Campaign |
|---|---:|---:|---:|---|
| Qwen3-1.7B | 18 | 28 | 2048 | Local |
| MiniCPM5-2B | 28 | 42 | 2048 | Local |
| Qwen2.5-3B-Instruct | 24 | 36 | 2048 | Local |
| Qwen3-8B | 24 | 36 | 4096 | Cloud |
| Qwen3.5-4B | 21 | 32 | 2560 | Cloud |
| Qwen3.5-9B | 21 | 32 | 4096 | Cloud |

We read the block output at the final prompt token. Raw prompts contain context,
question, three lettered options, then `Answer:`. There is no chat template or
generated reasoning trace. Native next-token letters determine predictions.
Settings are FP16, SDPA, batch eight, padding-aware position IDs and a 2,048-token
hard limit. Qwen3.5 uses text-only inputs to its native conditional-generation class
with the fallback linear-attention kernels used during the cloud campaign.

## Detector and actions

An L2 logistic probe predicts the ambiguous-condition label from clean hidden states.
Features are unstandardized. L-BFGS uses `C=1`, tolerance `1e-4` and at most 10,000
iterations. A strict `p > threshold` gate maximizes development ambiguous recall
subject to sufficient-context false triggering of at most 2%. Ties prefer smaller
false triggering, then a larger threshold. A no-op is available. This constraint
applies to the observed development sample.

EGUS selects the input's recognizable Unknown option when the gate fires, otherwise
retaining the base prediction. Unknown expressions are collected from training;
evaluation labels are used for scoring.

Evidence-gated steering uses the same probe and threshold. Its unnormalized direction
is the mean of 110 correct-Unknown versus stereotype-answer activation differences.
Extraction uses an appended answer token; inference adds `1.5 * direction` at the
final prompt token. These are distinct positions. There is no evaluation-based
layer or strength search in the repeated-fit campaigns.

The local FairSteer-style baseline uses a filtered correctness probe, drops other
errors, triggers above error probability 0.5, and uses strength 1. It replaces cuML
with scikit-learn and omits additional MMLU probe-training questions. Its supervision
differs from EGUS. The shared evidence gate isolates response-action effects more directly.

Surface-direct uses standardized context/question character and word counts, plus
context counts of `. , ; !`, digits and uppercase letters. Evidence labels and the
threshold rule match EGUS, but the detector is fitted separately and can trigger on
different questions. Its competitive performance limits semantic-sufficiency claims.

## Interpretation and statistics

Always report both evidence conditions. Unknown selection repairs ambiguous-context
errors but can remove correct sufficient answers. It cannot improve sufficient-context
accuracy. Its ceiling on the balanced set is `50% + 0.5 * base sufficient accuracy`.

The utility panel lacks Unknown, so EGUS preserves base answers by definition.
This is a routing control; it does not establish general capability preservation.
BBQ gains do not establish cross-dataset or open-generation benefits.

Sample SD describes five fits on shared questions. The cloud campaign's 12 comparisons
use crossed seed/family bootstrap intervals and family sign randomization, followed
by Holm adjustment. The earlier campaign retains its own original 12-comparison family.
Corrections are not silently pooled or reduced for this export.
