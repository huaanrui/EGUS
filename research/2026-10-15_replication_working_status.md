# Ongoing experiment checkpoint (internal continuation aid)

User authorized executing the complete cross-model replication plan. Do not stop at launch;
finish model runs, audits, plots, interpretation, and report actual results in conversation.

Workspace D:\PythonProject\mechanism, PowerShell. No AGENTS.md or subagent authorization.
Do not change old sources or the now-frozen main replication implementation/config/tests.
Local CUDA 8GB; three models sequential FP16/SDPA. No web/API judge used.

Main new module: mechanism/identity_replication.py. Config:
configs/experiment_identity_replication_v1.toml. Protocol:
research/2026-10-15_identity_replication_protocol_v1.md.
Results: results/identity_replication_20261015_v1/{qwen3,minicpm5,qwen25}.
143 tests pass in results/identity_replication_20261015_tests_final.log.
All three smoke checks passed, including no-op generation/prefill and full-MLP neuron half
versus output half (zero error). New models use generic MLP hooks; no new-model manual QKV.

Data: rank64/screen32/dev49 pairs from old identity_combo_20261016_v1, historicalType2 146,
historicalType1 160, Winogender240. All historically inspected targets labeled exploratory.
New confirmation: data/prepared/identity_replication_20261015_v1/confirmation.jsonl,
96 pairs/96 Type1 native families, absent from all earlier materialized WinoBias templates.
Independent audit reconstructed16,344 earlier records/1,177 families; native text/gold exact.
Agent text review, not independent human annotation. Type1 contextual/commonsense, not uniformly
explicit evidence. Excluded all leftoverType2 (mostly old malformed exclusions) and7 badType1s.

All model selections frozen by22:13:52:
- Qwen3 selected screen_combo; initialsource L12, processingMLP13, terminalMLP24.
- MiniCPM selected source_half; initialsourceL6, processingMLP7, terminalMLP26.
- Qwen2.5 selected mlp9_identity_positive_0.25; processingMLP9, terminalMLP31.
Read parameter_meta.json for precise settings, don't guess. Main methods34/23/23.
Fine_source/terminal/combination are forced diagnostic picks distinct from overall deployable
winner, which allows no-op. Pos/neg transfer sign is contribution to source-half score effect,
not permanent bias good/bad neurons. Gap sign is a separate definition.

Core main pipeline already trained; main --stage all process was deliberately stopped ONLY
after Qwen3 selection and during MiniCPM weight loading, to separate training/target opening
and add stronger norm controls. No science inference code repair. Resumed --stage train
finished all3, session7126 completed. Original session35378 killed at safe loading boundary.
Frozen code/config/data unchanged. Original nine primary confirmation tests remain fixed.

Reason for pre-target supplement: original RMS/columnnorm matching and coarse.5/1/2 factor
grid left some actual residual edit energies far below fine edits. Supplement script:
scripts/identity_replication_norm_controls.py. Declaration made before any targets:
results/identity_replication_20261015_v1/norm_control_declaration.json.
Uses up to5 DEV-only analytic RMS recalibration iterations; whole modules have four source/
terminal sign combinations selected onDEV under the original6task constraints. Keeps all5
random seeds, plus oldQwen3 combo counterparts. Supplement eight confirmation comparisons
Holm8 separately; original nine Holm9 unchanged. Pure calibration tests passed (including
zero target and no recursive serialization). Supplement source SHA is declared/frozen;
do not edit after calibration starts without new version.

ACTIVE DRIVER: unified exec session60321, scripts/finish_identity_replication.py,
log results/identity_replication_20261015_finish.log. At checkpoint22:18 it is in norm_freeze
(Qwen3 fine norm controls calibrated errors about0.0001; oldcombo then other models remain).
Driver stages, all automatic and sequential GPU:
1 norm_freeze -> all_norm_controls_frozen.json
2 main evaluate (all_models_frozen.json created AFTER norm lock) -> all datasets/utilities/
  signed exact restore+transplant/actual nativechat32pairs; main per-model audits
3 norm_evaluate (all4 datasets + utilities)
4 random_utilities (5random combinations; old5 extraQwen3, 5utilitytasks)
5 summarize_identity_replication.py (independent fit/gap/group/random/winner/prediction audits,
  primary9Holm and familybootstraps)
6 summarize_identity_replication_norm.py (independent normcalibration audit, supplement8Holm)
7 plot_identity_replication.py (PNG/PDF).
Completion writes results/identity_replication_20261015_v1/complete.json ONLY after all stages.
Monitor finish.log and currentstage log, keep meaningful Chinese commentary, don't launch
duplicate GPU jobs. Use write_stdin session60321 with50s waits (functions cell may yield at30s,
then functions.wait). Long driver runs should remain active through compaction.

Stage logs results/identity_replication_20261015_{norm_freeze,evaluate,norm_evaluate,
random_utilities,summary,norm_summary,plot}.log. If driver fails, inspect traceback and resume
the failed stage after fixing only nonfrozen reporting code; frozen scientific code needs new
version. Cachedshards permit restart. Do not overwrite trained selections or parameters.

Final required work after driver success:
- Inspect planned_comparisons.json, norm_comparisons.json, all evaluation/utilities/diagnostics/
  chat_results.json. Explain actual effect sizes, CIs, Holm corrections, ability tradeoffs.
- Inspect generated PNG with view_image; fix only plotting/reporting if needed.
- Write Chinese interpretation research/2026-10-15_identity_replication_interpretation.md.
- Check all frozen hashes/verification and GPUfree; modelprocesses exit.
- Report actual results to user (self-contained; concise tables), useful report link.

Limits to preserve: fixedthem task can introduce singular/plural ambiguity; derived explicit
card event and necessary man/woman profile tasks separate. Nativechat short answers are NOT
CEB free generation. MiniCPM tested against gold, not itself as fairness judge. No newFairSteer
training/CoT/RLHF/SAE/7Bquantization done. Model choice was prespecified; all negative results
retained. Fresh confirmation shares benchmark with historical but no delexicalized family
overlap; do not call it independent human-reviewed or general-domain confirmation.

UPDATE23:14: all norm controls froze22:33:58, main evaluation started immediately after.
Driver60321 is still ACTIVE in main evaluate stage (log evaluate.log), Qwen3 utilities/derived
controls now running after all4datasets completed. Other two main model evaluations and all
norm/randomutility stages still pending. Don't claim complete or launch duplicate jobs.
Qwen3 bothcorrect interim (within this round):
historical base65.41,selected66.44,finecombo66.44,wholecombo64.73,oldcombo72.95,oldL7 70.55;
historicalType1 base41.56,selected41.88,finecombo41.88,wholecombo42.19,oldcombo43.12,oldL7 43.12;
Winogender base48.75,selected47.92,finecombo47.92,wholecombo48.75,oldcombo48.54,oldL7 48.12;
NEW confirmation base36.98,selected35.94,finecombo35.94,wholecombo37.50,oldcombo34.90,oldL7 34.38.
Thus Qwen3 freshType1 does NOT extend earlier Type2 gains. Don't change selections.

Additional numerical audit script scripts/compare_identity_replication_anchor.py writes
qwen3/old_anchor_numerical_comparison.json. Re-run after all old targetpredictions exist.
Earlier batch8 versus currentbatch4 FP16 explains slight scoreboard differences: baseline
historical changed1/584 at oldmargin0; oldcombo changed2/584 at oldmargin.03125; Type1oldcombo
changed3/640 with oldabs margins<=.046875. Mean logitdifferences~.018, labels match.
User already informed comparison uses current-round settings and boundary cases explain drift.

Supplement reporting now also computes TARGET edit-RMS ratios from stored shards (DEV calibration
does not guarantee target norm equality). Mainplot uses refined norm controls; extra pairedCI
plot identity_replication_effects.png/.pdf added. Reporting/plot code may change; scientific
main and norm calibration source hashes must remain frozen.
143tests still pass. No science source repair after freezing. One temporary inline audit
command had a literal-newline syntax error, corrected with standalone script; no inference
artifacts affected. Norm matching correction already sent: oldrandom maxDEV RMS error0.1173%,
not all<0.1%; fine random errors roughly0.006–0.019%.

Uniform source localization was FOUR coarse depth candidates, unlike previous detailed
all-layer profiling. It selectedQwen3 sourceL12, not oldL7. This round tests the finite-budget
uniform MLP procedure, NOT an exact replication of old complete head/path discovery on all
models. OldQwen3 fine combo remains separately evaluated. This limitation was told to user.

CRITICAL UPDATE23:30: originaldriver60321 FAILED during Qwen3 diagnostics, after all4
behavior datasets and utilities were fully saved. Bug: diagnostics iterated non-old-prefix
sites, including mlp7_identity added solely for oldL7 benchmark; it has no transfer_pos params.
Do NOT modify frozen mechanism/identity_replication.py. Scientific execution repair is separate
VERSION2 script scripts/resume_identity_replication_execution_v2.py and protocol
research/2026-10-15_identity_replication_execution_v2.md. New declaration execution_v2_declaration.json.
NEW ACTIVE unified exec session10443, log results/identity_replication_20261015_finish_v2.log.
It reruns ONLY affected Qwen3 diagnostic with sites listed in parameter_meta.groups (new
diag_v2 cacheprefix, diagnostics_v2.json and standard diagnostics.json), then Qwen3 actualchat/
audit; continues original Mini/Qwen2.5 evaluate; then norm_evaluate, random_utilities,
semantic_v2, main summary, norm_summary, plot, anchor_compare; final complete.json version2.
Original behavioral methods, parameters and nine/eight comparisons unchanged, all fingerprints
validated. No need to redo completed behavior inference or fitting. Two repair regressiontests
passed; eventual fulltestcount147 (143+2semantic+2execution). ImportedTestCase duplication in
semantic tests was removed before finaltests; don't claim155.

Semantic-control repair v2: originalneutral profile used red/blue teams and taskaskedbluecard,
introducing direct color matching. Agent acknowledged issue to user. Originalexplicit question
also unnecessarily prefixed 'Considering pronoun them'. New script
scripts/identity_replication_semantic_controls_v2.py, declaration semantic_control_v2_declaration.json,
was declared AFTER observing Qwen3 originalsemantic outputs; exploratory, notnewconfirmation.
Four cases: gender/neutral-code17vs29 profiles × direct explicit-card/required-profile question.
Neutral profiles now SWAP codes across pair; originalneutral hadidenticalinput +colorconfound.
Same frozen intervention methods; no new methodselection. Newv2driver runs this sequentially
after remaining normative/randomutilities. Separate --wait_evaluate job already FAILED because
it detected originaldriverfailure; no need to restart thatwaitingjob (newdriverhandlesdirect).

Qwen3 originalsemantic (accuracies, descriptive): fixedthemcoref base92.58%,oldcombo96.09%;
explicitcard with pronounprefix base74.22%,selected75.78%,oldcombo74.22%; necessarygenderprofile
base95.70%,oldcombo94.53%; BAD originalneutralcolorprofile base50.78%,oldcombo52.34%.
Mark originalneutral invalid as neutralitycontrol; don't use it as evidence ofspecificity.

External non-Python graphics app appeared around23:26, GPU~809MiB; don't kill unrelatedprocesses
or insist GPUtotal0. Ourmodel runs still sequential; ensure final noOURpythonGPUprocesses,
report experimental GPU allocation released rather than wholeGPUempty ifexternal remains.
At failure no identity_replication Pythonprocesses remained before new10443 started.

After finaldriver success, additionally independently recompute diagnostic transmission/
bootstrap from saved predictions if feasible (notyetimplemented), inspect plotPNG, write
Chineseinterpretation and report actualresults. Norm matched results and othermodel outcomes
still pending. All new results mustkeepfrozenplan/selection fingerprints and version2repair note.

LATEST23:38: v2driver10443 ALSO STOPPED, at original audit: v1 save_scores omitted
occupation_pair used by pair_records bookkeeping. Actual inputs/labels/scores/predictions
and original metrics valid. Do NOT modify frozen main source. VERSION3 audit restores this
metadata ONLY IN MEMORY from frozen base pair_id lookup (record/anonymous suffix handled),
never edits savedpredictions. Actual Qwen3 end-to-end audit now PASSED118,032 savedrows,
all4dataset metrics exact, fingerprints unchanged.

NEW ACTIVE DRIVER unified exec session21355, script
scripts/resume_identity_replication_execution_v3.py, log
results/identity_replication_20261015_finish_v3.log. It preserves v2 Qwen3 diagnostic fix,
patches ONLY Experiment.audit runtime binding to scripts/audit_identity_replication_records_v3.py,
continues Mini/Qwen2.5 original scientific evaluate, then runs norm_evaluate/randomutilities/
semanticv2/summary/normsummary/plot/anchorcompare via
scripts/run_identity_replication_reporting_v3.py (same audit-only patch in subprocess).
Finalcomplete.json execution_version3. Parent v1/v2 scripts/artifacts remain unchanged.
Reporting scripts now import restore_metadata for metric reconstruction; no scientific changes.
New declaration execution_v3_declaration.json hashes workflow/auditor/wrapper/v2diagnostic/
protocol research/2026-10-15_identity_replication_execution_v3.md.
Full tests expected149 =143+2semantic+2executionv2+2auditv3; two new auditregressions and
actual Qwen3 fullsavedpredictions audit passed. Run all tests once more before delivery.

Originaldriver60321 and v2driver10443 are no longer active. Semanticwaitjob failed intentionally
on detection ofold drivererror and is no longer active; v3driver runs semantic--stage evaluate
sequentially, no need restart waitingjob. GPUexternalgraphics809MiB persists, avoidwaitingfor0.

FINAL 2026-10-16: ALL AUTHORIZED REPLICATION WORK COMPLETED. Main driver21355 exited0,
complete.json execution_version3 at01:55:43. Norm controls, five-seed utilities, semanticv2,
main9/supplement8Holm comparisons, plotPNG/PDF and old numerical comparison are complete.
153 tests passed after four meaningful derived-grammar/score-decomposition tests were added.
Both main plotPNGs visually checked. Diagnostic audit independently rederived7680rows.

Latest user asked whether results are poor. Explained weak generalization and strong mean
baseline, then promised and COMPLETED a frozen answer-role-balanced robustness probe.
Current rank/screen/dev/historicalType2 gold100%secondmentioned occupation. Native file audit
found dev790/790 uniquely extracted labels second; test788/792. Thus imbalance cannot solely
be attributed to our him/her filtering (initial filtering-cause hypothesis corrected to user).
No claim fulldataset meaningless or model necessarily only uses shortcut.

New probe:64 existing historical families, original coordinated sentence vs and->", who"
relative subject transformation, native goldsecond/derived goldfirst; male/female andAB/BA.
Structural/text-only hash selection, no original fit/screen/dev family overlap, agent text
review not independenthuman; original/derived difficulty and event roles differ. Exploratory
declared aftermainoutcomes, no refit/reselection. Frozen parameters and maincodes unchanged.
Script scripts/identity_replication_grammar_balance.py, artifacts root/grammar_balance,
report research/2026-10-16_identity_replication_grammar_balance.md. Driver46073 exited0.
9216 predictions independentlyrecomputed, targetenergy audit complete.
Qwen3 baseline original64.84/relative42.19/balanced53.52 bothcorrect;
oldcombo75.78/33.59/54.69; balanced delta+1.17ppCI[-2.73,+4.70], totalacc67.58->67.38,
secondmentionprediction59.38->66.99%. Oldnormwhole74.22/39.06/56.64.
Unifiedfinebalanced Qwen3 55.08 vs53.52, Mini51.95 vs51.56, Qwen2.5 73.05 vs71.48;
allfine vsbase exploratory CIs include0; some sourcehalf/whole comparisons have positive
intervals, not posthoc newwinners or blanket allinterventionsinvalid claim.

EndpointC/D decomposition exactsavedprediction reconstruction complete; report
research/2026-10-16_identity_replication_margin_decomposition.md. HistoricaloldQwen3
base65.41, actual72.95, replacingonlycommonC73.97, replacingonlyidentitydifferenceD66.44.
This is outputalgebra not actual modelintervention or causalmediation, Ccontainsvalidgrammar
and isnot purepositionbias, Dnot puresocialbias. User informed balanceoutcome incommentary.

Repaired semantic metrics independentlyrecomputed. Qwen3 oldfine factualaccuracy93.75->94.92,
genderflip6.25->3.91; oldnormwhole same. Mini/Qwen2.5 nearceilingfacts, little phenomenon.
Final standalone interpretation research/2026-10-16_identity_replication_interpretation.md
and root/completed_followup.json generated, all supplemental hashes recorded, rootverification
merged diagnosticreconstruction/grammarbalance checks. No experiment GPUprocesses remain.
No required experiment jobs remain running; answer user with completed negative/mixed results
and calibrated interpretation, not a launch/status-only final.
