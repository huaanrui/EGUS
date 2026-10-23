# Working status: evidence validation supplement

User authorized repeated training resamples, independent confirmation/statistics,
and Qwen2.5 replication. Active driver unified session62592; logs at
results/evidence_validation_20261017_driver.log and per-stage logs under
results/evidence_validation_20261017_v1. Do not stop at launch; finish all stages.

Main science frozen in mechanism/evidence_validation.py, stats module,
configs/experiment_evidence_validation_v1.toml and protocol v1. Do not edit those
frozen sources; scientific repairs require a separately versioned executor or
declaration. Earlier evidence_story v1 assets are read-only. Reporters only load
joblib models, never refit them (prior report serialization bug was fixed there).

Prepared five seeds20261017..20261021, shared2200 fit questions per seed, fixed
BBQ dev352/historical test704. Three models: Qwen3 L18, MiniCPM5 L28, Qwen2.5 L24.
Initial two-model seed uses exact old probes/directions/baselines. Other seeds
resample both heads and110 direction examples. All15 selections freeze before
new target inference. Core tests15 pass in evidence_validation_20261017_tests.log.

New confirmation:640 native SynthBias questions,320 complete gender pairs,
320 delexicalized families,17 new occupation pairs,1920 rows with3 permutations.
Excluded all earlier prepared IDs2784, families1386 and occupation pairs378,
including draft prepared pools. Remaining candidates294 ambiguous/332 clear
gender pairs; chose160 of each. Source-fit unknown wording Cannot be determined.
Metadata audit passed. Native labels preserved; sample14232 has a competing
interpretation, documented before target inference in data_quality_note.md;
no rows excluded or relabeled. Do not call labels logically noise-free.

All15 selections frozen17:45:46; Qwen3 target evaluation nearing completion17:56,
then MiniCPM5 and Qwen2.5. Driver unified session62592 still active. Full related
suite51 tests passed (full_tests.log), scripts compile. No subagents authorized.

Read-only preview after locking (preview_evidence_validation_direct.py): Qwen3
BBQ evidence_direct79.0909±0.5353%, surface75.9943%, base56.6761%; clear loss-.17045pp.
New confirmation evidence_direct56.53125±2.68459%, surface50%, base40.78125%,
clear loss-40.875pp. This is a major preservation failure despite overall gain;
already told user explicitly. Do not retune after seeing it or call migration
successful. Other models and counterfactual baselines still need completion.
New confirmation type2 potential label ambiguity recorded separately before
inference; it is not an excuse for broad harm or a reason to discard rows.

After driver exits0, run summarize_evidence_validation.py, then
verify_evidence_validation.py and plot_evidence_validation.py (read-only model
assets; independent verifier refits in memory only). Reporter computes12 primary
contrasts, seed×family bootstrap4000, Holm p; supplementary occupation CIs.
Primary method evidence_direct versus surface_direct overall and versus base
on sufficient questions, on historical BBQ and new confirmation, all3 models.
Report actual results even if confirmation fails. Check figure visually, save
completion audit and summarize in Chinese with correct scopes.

ALL COMPLETE. Driver exited0; reporting,180 classifier refits,517536 predictions,12 stats,51 tests and plot QA passed. Final report findings.md and completion_audit.json saved. BBQ stable gains in all3 models; no Holm-significant advantage over surface on historical BBQ; fresh confirmation has severe sufficient-task harms. No experiments pending.
