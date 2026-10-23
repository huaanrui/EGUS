# Related Work preparation — 2026-10-17

This folder supports Chapter 2 of the BBQ activation-steering/evidence-classification manuscript. It contains literature research and writing, not a new experiment run.

## Deliverables

- [chapter2_related_work.md](chapter2_related_work.md): English chapter draft, with four subsections and BibTeX citation keys. The final paragraph states the proposed empirical scope, not a claim of unique conceptual invention.
- [literature_review_zh.md](literature_review_zh.md): Chinese annotated review of 24 papers, closest-work comparison, novelty limits, and suggested controls.
- [references.bib](references.bib): 24 bibliography entries based on primary publication metadata. Some records deliberately cite an arXiv version when full proceedings details have not been verified.
- [sources.json](sources.json): metadata source URLs, retrieval timestamps, webpage hashes, and citation fields. This is provenance for bibliographic metadata, not a full-text archive.
- [collect_sources.py](collect_sources.py): optional metadata refresh utility. Re-running it uses the network and overwrites sources.json and references.bib; it does not touch experiment files. Publication records can change.

## Using the draft

The draft uses `\citep{...}` keys matching references.bib. Adapt citation commands to the eventual journal template. Markdown headings must be converted to the template's section commands. No LaTeX document or PDF has been compiled.

Read the Chinese review before copying novelty statements into the introduction. In particular, evidence-aware abstention, conditional steering, and the limitations of BBQ-based fairness claims all have close prior work. The proposed focus is the incremental effect of changing the action while holding the evidence gate fixed.

The English draft describes comparisons already present in the project at a high level. It does not claim that newly recommended controls or prespecified paired analyses have been completed. Dataset and supervision differences belong in the methods and limitations sections.

## Verification

Bibliographic fields were checked against primary sources on 2026-10-17. A local consistency check verifies unique keys, mandatory metadata, draft citation resolution, and balanced BibTeX braces. It is not a bibliography compilation test or an exhaustive novelty search. See validation.json for counts.

The chapter's summaries distinguish original-paper findings from local experimental evidence. Broader literature coverage and camera-ready venue metadata should be refreshed before submission.
