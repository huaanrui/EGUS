# Mapping the release to the manuscript

| Paper component | Implementation | Released results |
|---|---|---|
| Evidence gate and EGUS | `egus/policies.py`, `egus/experiment.py` | `egus_layer_selected` |
| Evidence-gated steering | `egus/model.py`, `egus/experiment.py` | `evidence_joint_selected` |
| FairSteer | `egus/baselines/author.py`, author-baseline config | `fs_joint_selected` (cuML) |
| KnowBias | `egus/baselines/knowbias.py`, pinned author sources | `models.*.knowbias` (one fit) |
| Table 2: overall accuracy | `egus/metrics.py` | `benchmarks/main/summary.json`: aggregate accuracy and KnowBias metrics |
| Table 3: conditional accuracy | `egus/metrics.py` | `ambig` and `clear` metrics |
| Table 4: matched-gate comparison | `egus/experiment.py:select` | `evidence_joint_selected`, `egus_at_evidence_layer`, `egus_layer_selected`, `evidence_at_egus_layer` |
| Selected configurations appendix | Validation candidate selection | `models.*.selections` and `models.*.knowbias.selected` |
| Statistical comparisons appendix | `comparison` and `holm` in `egus/metrics.py` | `comparison_groups` |

`benchmarks/main/predictions/` contains question-aligned prediction arrays. `reproducibility/paper_results.json` records their checksums and the manuscript hash. The manuscript itself is not distributed. `tables.json` records the numeric table cells checked against that manuscript; accuracy tables use decimal half-up rounding to two places; statistical contrasts retain the original Python formatting. Full precision is retained in the summary.

The two execution profiles are necessary for the dependency environments and the recorded numerical provenance. They implement the current paper's selected-layer protocol. Removed exploratory experiments and their outputs are outside this release.
