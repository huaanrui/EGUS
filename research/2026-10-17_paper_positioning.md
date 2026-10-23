# Paper positioning and English working draft

Date: 2026-10-17. This is a writing and literature note, not a new experiment protocol. No model results or frozen parameters were changed.

## Recommended working title

**Rethinking Activation Steering for Debiasing on BBQ: A Simple Evidence Classifier as a Strong Baseline**

Alternative, emphasizing the controlled comparison:

**Does Debiasing on BBQ Require Activation Steering? Disentangling Evidence Detection and Intervention**

Use “activation steering” rather than claiming to evaluate all of “representation engineering”: our classifier also reads internal representations, while the tested intervention is a much narrower subset of representation control. “Evidence-sufficiency classifier” is the more precise term in the body. These are working titles, not novelty claims.

## Central claim

A simple evidence-sufficiency classifier coupled with an explicit abstention rule is a strong baseline for debiasing on BBQ. Across our tested small models, direct abstention matches or exceeds the accuracy of the evaluated activation-steering policies. However, both strong surface-feature baselines and substantial transfer failures limit the interpretation of these gains as general bias mitigation.

## Draft abstract

Activation steering has been proposed as an inference-time approach to reducing social bias in language models. On benchmarks such as BBQ, however, performance improvements can also result from recognizing when the context does not support an answer. We examine the contributions of evidence detection and intervention by comparing direct abstention with activation steering under shared gates, alongside surface-feature, question-only, and output-score baselines. Our study evaluates three language models, ranging from 1.7B to 3B parameters, across five training resamples with family-separated training, development, and evaluation data. A linear evidence-sufficiency classifier coupled with direct abstention improves mean BBQ accuracy from 56.68%, 43.47%, and 73.01% to 79.09%, 84.43%, and 81.28%, respectively. These gains predominantly come from ambiguous questions. Under the same evidence gate, direct abstention matches or exceeds the accuracy of the tested steering policy, although advantages over a strong surface-feature classifier do not survive the prespecified multiple-comparison correction. On a separate 640-question SynthBias confirmation set, aggregate improvements coexist with substantial losses on questions labeled as having sufficient evidence. Our findings motivate separating evidence recognition, intervention choice, and sufficient-evidence preservation when assessing debiasing methods. We present direct abstention as a necessary baseline for BBQ-style evaluation, while limiting our conclusions to the tested models, intervention settings, and task formulations.

Draft qualification: “matches or exceeds” above describes observed means, not a statistical equivalence or noninferiority result. Historical BBQ evaluation is exploratory; the 640-question confirmation was newly held out relative to our prior prepared datasets. The steering comparison uses fixed layers and strengths, not an exhaustive optimization of steering.

## Closely related primary sources

1. **AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders** (2025). Introduces comparisons for concept detection and steering; its reported prompting baseline exceeds the evaluated steering methods. This is a close precedent for the strong-simple-baseline framing, but does not itself establish our BBQ result. https://arxiv.org/abs/2501.17148
2. **Rethinking Prompt-based Debiasing in Large Language Models** (2025). Discusses evasive answers and sufficient-context performance degradation in BBQ, showing why bias scores alone can obscure undesirable behavior. Thus “more unknown responses need not imply better debiasing” is already established prior work. https://arxiv.org/abs/2503.09219
3. **One Example Is Enough to Pass Fairness Benchmarks: Rethinking Fairness Evaluation for Aligned LLMs** (September 2026). Studies single-example ICL/GRPO, missing-evidence reasoning, and limited transfer to another fairness axis. This is the closest overlap in the benchmark-level claim; a generic “BBQ gains are not fairness” paper would need substantially sharper differentiation. https://arxiv.org/abs/2609.14860
4. **SAEs Are Good for Steering – If You Select the Right Features** (EMNLP 2025). Distinguishes input-responsive features from output-effective features and reports improved steering after feature selection. This cautions against extending limited comparisons to universal claims about an entire intervention family. https://aclanthology.org/2025.emnlp-main.519/
5. **Evaluating feature steering: A case study in mitigating social biases** (Anthropic, 2024). Examines BBQ steering effects, utility tradeoffs, and off-target effects. Relevant evaluation precedent, not a directly comparable local baseline. https://www.anthropic.com/research/evaluating-feature-steering
6. **Representation Engineering: A Top-Down Approach to AI Transparency**. The scope includes monitoring as well as manipulating representations, motivating the narrower term “activation steering” in our title. https://arxiv.org/abs/2310.01405
7. **FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering**. The original paper reports multiple tasks and models; our adapted small-model results should not be presented as disproving its original results. https://arxiv.org/abs/2504.14492

## Candidate distinctive contribution

The proposed distinguishing comparison is **holding the evidence gate fixed while changing the action**: preserve the base prediction, add an unknown-logit offset, force the unknown choice, or modify a hidden state. Pair this with explicit decomposition into ambiguous-question corrections and sufficient-question harm. Present this as an empirical analysis of the tested policies, not proof that all steering effects are mediated by abstention or that no useful bias representation exists.

Detection and control are different operations. The current MiniCPM5 results make this distinction particularly visible: evidence-gated steering yields 43.10% BBQ accuracy, while the same evidence gate with direct abstention yields 84.43%. The analogous means are 78.95% versus 79.09% for Qwen3 and 78.61% versus 81.28% for Qwen2.5. These are descriptive matched-gate comparisons; additional prespecified paired analyses are needed for inferential claims.

## Proposed manuscript structure

1. **Introduction:** ask what activation intervention adds beyond evidence detection and a task-aligned decision rule.
2. **Related Work:** distinguish representation reading/control, selective prediction, fairness-benchmark criticism, and the specific precedents above. A fuller selective-prediction literature review remains to be done.
3. **Controlled Evaluation:** specify family splits, supervision budgets, fixed layers, strict source-only threshold selection, candidate scoring, and adaptation differences from author baselines.
4. **Evidence Detection and Intervention:** define the linear classifier, constrained threshold, direct abstention, product gate, direction estimation, and logit-offset control.
5. **Results on BBQ:** report three models and five resamples; split ambiguous/sufficient outcomes; include strong simple controls and paired uncertainty.
6. **Robustness and Transfer:** token-length controls, option permutations, surface sensitivity, and the fresh-confirmation over-abstention results.
7. **Discussion and Limitations:** distinguish task accuracy from broad fairness, local settings from a method family, and readable signals from causal mechanisms. Keep older CEB and component-search results supplementary unless needed to answer a specific question.

## Targeted additions before a strong comparative submission

These are proposals, not completed experiments or authorization to run a new campaign.

- Add a source-selected evidence-focused instruction/one-shot baseline, motivated by the 2026 paper. Report both a decision-only scoring setting and any reasoning-generation setting with their different inference budgets.
- Give steering a declared development-only strength budget and compare recall/harm tradeoffs across operating points. The present fixed-strength results do not establish the best possible steering performance.
- Compute paired same-gate direct-versus-steer effects, sufficient-question losses, and realized abstention rates. Equalize abstention rates where feasible without using target labels, to examine whether differences persist beyond output frequency.
- Add a sufficient-evidence control with an explicit unknown distractor; no-op performance on tasks lacking that option is not an informative abstention safety test.
- Keep all currently observed evaluation data marked as observed. Any target-informed improvement requires a separately established confirmation set; do not relabel an old target as unseen.

The immediate writing opportunity is a focused empirical analysis with a strong baseline. Novelty relative to the September 2026 paper remains a question to establish through this more specific controlled comparison, rather than by changing the title alone.
