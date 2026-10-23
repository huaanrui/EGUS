# Active best-combination search handoff

User wants each model's highest-scoring combination, after noticing oldQwen3 MLPL9+headL21
82.02% single accuracy beat new restrictedMLP77.23%. Authorized action, do NOT finish at launch.
No subagents authorized. Workspace D:\PythonProject\mechanism. No AGENTS found.

New frozen scientific files: mechanism/identity_best_combo.py,
configs/experiment_identity_best_combo_v1.toml,
research/2026-10-16_identity_best_combo_protocol_v1.md,
tests/test_identity_best_combo.py. Preserve all old frozen files and outputs.
New output root results/identity_best_combo_20261016_v1.

ACTIVE serial GPU driver unified exec session55524:
scripts/run_identity_best_combo.py, parent log results/identity_best_combo_20261016_driver.log.
Perstage logs results/identity_best_combo_20261016_{smoke,train,evaluate}_{qwen3,minicpm5,qwen25}.log.
Smokes all3 -> trainall3 -> lockallmodels -> evaluateall3 -> summary -> complete.json.
Never start duplicateGPUjobs. Initial Qwen3smoke session17113 alreadyexited0.
Fulltest session34936 completed log157testsOK (previous153+4newmeaningfulsearchtests).
Qwen3/MiniCPMsmokes passed, peakallocated3.31/4.81GiB; Qwen2.5 smoke inprogress atlastcheck.
Native pre-o_proj head slice validated against full projection; connected all-layergradients
match isolated-sitegradients cos>.99, finite differencesigns recorded, zeroedit/unloadexact.

Scientificplan: taskType2 primarysingleaccuracy (82.02%metric), pairedbothcorrectsecondary.
Separate bestunconstrained vsbestwithin5devutilitytasks2ppdrop. Full layersMLP+allqueryheads,
threepositions(contextpronoun,questionpronoun,decision). Twofirstorderrankobjectives:
goldmargin centeredcontribution and squaredgendergap. Rank64pairs×gender×AB/BA=256rows.
Connectedgradientcapturebatch1 viaembeddingoutputrequiresgrad; loss1024scale. Allmodelweightsfrozen.
Perkind/position/objective top2sites +allpreviousphysicalsites. MLP64/128, head16/32;
positive/negative/signedgroups,amount.5/1 plusraw/wholemeancontrols. Shortscreen16pairs,
fullscreen32pairs, refinement, beamwidth6 depth<=3sites,12atompool. Dev49pairs,
top16+ALLoldreferencescompete. ExistingQwen3oldcombo included mandatory, no earlieromission.
Previousmodelselected/fine/sourcehalf/wholecombo/normwhole also included.
Afterall3choicesfreeze onlyhistoricalType2146pairs evaluated. No newgeneralizationruns.
Historicalalreadyobserved, answersecondheuristic100%; taskoptimization≠generaldebiasing.

Runner/scienceout preserve cacheswithinputhashes. Do not modify frozen scientificmodule/config/
protocol/tests now without versioning. Report-onlyscripts canedit asneeded.
Helper scripts prepared:
- scripts/summarize_identity_best_combo.py (driver runs it automatically)
- scripts/verify_identity_best_combo.py (MANUALLY RUN AFTER DRIVER): independentlychecks
  alloldrefsactuallyparticipatedindev, bothselectionsreplayed, familyisolation, savedpreds/gold,
  parameters/code/datahashes. e.audit_search also comparespreviousreferencemetrics exactly.

Remaining: monitorstageprogress (Chineseupdates), repairanyactualfailureswithoutchangingoldscience,
completealltraining/test, runindependentverifier, inspectreport, optionalstaticplotandview,
checkexperimentGPUprocessesexit, finalactualpermodelwinner/dev/testcomparison+exactcomponents.
No need keeprequestingpermission or createautomation. Highestmeansboundeddevsearchwinner,
notglobaloptimum orretunedontest. Ifnewwinnerlosesoldonhistoricaltest, reportbothhonestly.

One transient firstvenv unittestlaunch failed Unabletocreateprocess; immediate retry succeeded,
full157tests passed. No resultingscientificfailure. Old3modelreplication finishedpreviousrequest,
allitsdata/artifactssafe. Newsearch isindependent andactivelyrunning.

LATEST after user status request: original driver55524 FAILED while loading MiniCPM fromF,
safetensors Rust OSError21 DeviceNotReady, before any MiniCPM gradients/selection.
Qwen3 trainCOMPLETED13:12:21:580short,297fullscreen,26dev candidates. Bestdevaccuracy95.408%
(both91.837%) combo: head21H1decision wholecenter lambda-1.5; MLP6contextaccuracy64neg
damp1/posamp1; MLP8questionaccuracy128negdamp1.5/posamp1.5. Devbinding drops98.469->92.347,
so fails2ppconstraint (reading actually86.735->100,literal100,role79.082->92.347,
record73.724->72.704). Constrainedwinner is preservedoldL9MLP+L21H1combo,dev87.755%
(both80.612). TEST NOT YETRUN because all3model choicesmustfreeze first.
User informed status incommentary and continuingrestoration, notfinal.

First resumeattempt log driver_resume_v2 exitedimmediately transientvenvlauncher UnableCreateProcess.
Secondattempt ACTIVE unifiedexecsession94662, originalunmodified scripts/run_identity_best_combo.py,
newparentlog results/identity_best_combo_20261016_driver_resume_v3.log. Perstage logsappend.
CompletedQwen3smoke/trainreusefrozencaches, Mini/Qwen2.5pending. Scientificcode/configunchanged.
Fmodeldirectory/config currentlyreadable, amplediskspaceD65GB,F137GB. No reason to kill unrelatedapps.
IfFloadOSError21 repeats, verifiedlocalDcheckpointcopy is a viable execution-only fallback;
declare/hash checkpointidentity and preservefrozenmaincode/selection. Do not silently changeweights.

LATEST21:50: resume94662 stillACTIVE, parent driver_resume_v3.log. MiniCPM successfullyloaded
afterretry and TRAIN COMPLETED21:49:26. Frozen580? Actual Mini515short/242screen/21dev.
Besttask==bestconstrained dev92.857%, feasibleTRUE: head17H4question_identity rawamount1
(zerohead); MLP12identity accuracy128neg centeredamount1.5 andpos-1.5;
MLP41decision rawamount1.5 (reversehalf ofMLPinput/output, notmerelyzero).
Qwen2.5 TRAIN nowstarting. All3choicesnotyetfrozen, TEST STILLNOTRUN.
Rootoriginaldriver55524dead, use94662only. No checkpointcopyor scientificcodechangewasneeded.
Userlatestaskedstatus; answeredQwen3dev95.41unsafe,binding98.47->92.35; oldcombo87.76safe,
then Mini92.86safestatus, explicitlytheseareDEVnotTEST. ContinueuntilALLworkcomplete.
Reporting-only summaryscript tablelayoutcorrected andactualcoordinatecountsadded.
Manualafterdriver: verify_identity_best_combo.py, plot_identity_best_combo.py, viewPNG,
readactualreport/summary, checknoexperimentGPUprocesses, openreportoptional, finalactualscores.

LATEST22:15: priorresume94662 FAILED Qwen2.5 rank sampleindex16 with nonfinitegradients,
16finiteNPZrowsretained. Originalmaincode/config/tests/old2selectionsunchanged.
NEWACTIVE unifiedexec99267, scripts/run_identity_best_combo_numerical_v2.py,
parentlog results/identity_best_combo_20261016_driver_numerical_v2.log.
Perstagelogs results/identity_best_combo_20261016_numerical_v2_{train,evaluate}_{name}.log.
Declaredroot numerical_v2_declaration.json + research/2026-10-16_identity_best_combo_numerical_v2.md.
Executiononly adaptive lossscale1024/256/64/16/4/1, largestfinite perinput, dividebackbyactualscale;
forwardnonfiniteabort, neverdropsamples. Qwen2.5 firstfiniteinput validates1024exactoriginal,
scale64forwardexact andmaxgradientrelativeerror0.00328. Newcache rowsrecordlossscale+executorsha.
Othermodelsselectionsremainfrozen, alltargetinferenceSTILLPENDINGuntilQwen2.5selectionfreeze.
Atlastcheck Qwen2.5rankfitfinished,560/576shortcandidatesdone. DeviceIOerrordidnotrecur.
Full159testsPASS log results/identity_best_combo_20261016_tests_numerical_v2.log (157+2repairtests).
Userinformed numericalfixand560/576progress. Continueallwork.

Independentverifiernowalso validatesrepairdeclaredhashes and all256rankrowsfinite, rederives
means/accuracy/gapranking/groups fromNPZ forALL3models, reportslossscalehistograms. Couldtake
someCPUminutesafterGPUjobs. Runitafterdriver, thenplot/view/report. Don'tmodifyfrozenmaincode.

LATEST22:41: ACTIVE99267 numericalv2driver. All3TRAIN/SELECTIONDONE andall_models_frozen.json
written22:40:19; Qwen3TEST evaluatingnow, Mini/Qwen2.5testpending. Qwen2.5 devbest==safe
96.9388%, MLP15identity gap128positive centeredamount1 (meanreplace128coordinates).
Qwen2.5 candidatecounts576short/198screen/22dev. Qwen3dev95.41unsafe(old87.76safe),
Mini92.86safeasabove. Userinformedall3frozenandtestingstarted. NO FINALTESTSCORESYET.

FINAL22:55 ALLREQUIREDWORKDONE. Numericalv2driver99267 exited0, complete.json22:50:43.
Qwen3TEST base76.54/both65.41; prioroldcombo82.02/both72.95; newbest90.41/both84.93;
constrainedold82.02/both72.95. Newbestdevbinding98.47->92.35 fails2pp.
MiniTEST base72.09/both60.27; priorselected73.80/both61.64; newbest==safe96.06/both92.81.
Qwen2.5TEST base96.23/both92.47; priorselected96.40/both92.81; newbest==safe96.92/both93.84.
Qwen3newvsold+8.39ppCI[5.66,11.24]; Mininewvsprior+22.26CI[18.92,25.34]; Qwen2.5newvsprior
+0.51CI[0,1.20] exploratoryunadjusted. Scoresarehistorical146Type2pairs, samebatch4FP16,
AB/BAandgenders; notglobaloptimality ornewconfirmation, positionsecondheuristic100%known.

Independentverifier session69783 exited0;16936predictions, all256rankrows/model finite,
allmeans/accuracy-gaprankings/groupsrecomputed, oldreferencepredictionsEXACT. Qwen2.5 scales
1024on248rows,256on8rows; first16originalcachepreserved. AllfrozenSHA256verified.
159testsPASS log tests_numerical_v2. Plot script79048exited0, PNG/PDFcreated andPNGvisually
checked. NoOURexperimentPythonprocessesremaining; unrelatedPlantsVsZombiesgraphicsprocess
appearsinnvidia-smi, DO NOTKILLITorclaimGPUtotalempty. Allscientificjobsdone.

Finalreport research/2026-10-16_identity_best_combo_findings.md hasexactcomponents/strengths,
threecompletewithin-tasktables,devabilitytables,CIs,numericalrepairnotes. Finalresponse should
giveactual3modeltesttable, Qwen3unsafe vsconstrainedold distinction, concretewinningcomponents,
scopehistoricalType2/dataimbalancebriefly,reportlink. Neverfinishwithstillrunningstatus.
