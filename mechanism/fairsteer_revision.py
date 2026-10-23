"""Controlled ambiguous-only revision and separately frozen unseen-category tests."""
from __future__ import annotations

import ast
import collections
import json
from pathlib import Path
import pickle
import time
import tomllib
import urllib.request

import numpy as np

from .data import read_rows, sha256, write_json
from .fairsteer_data import ROOT, fair_sample, frozen_rows, rank, source_fewshot


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def trusted(path):
    path=Path(path)
    if sha256(path)!=read(path.with_suffix(path.suffix+".sha256.json"))["sha256"]:
        raise ValueError("Fitted artifact integrity mismatch")
    with path.open("rb") as f:
        return pickle.load(f)


def node_dump(path, name, method=None):
    node=next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if getattr(n,"name",None)==name)
    if method:
        node=next(n for n in node.body if getattr(n,"name",None)==method)
    return ast.dump(node,include_attributes=False)


def reuse_probes(e, directions):
    prior=ROOT/e.c["reuse_probe_from"]
    prior_state=read(prior/"manifest.json")
    original_config=prior/"source_snapshot/experiment_fairsteer.toml"
    if sha256(original_config)!=prior_state["fingerprint"]["config_sha256"]:
        raise ValueError("Original configuration snapshot mismatch")
    old=tomllib.loads(original_config.read_text(encoding="utf-8"))
    allowed={"dataset_dir","output_dir","direction_context","test_role","reuse_probe_from","additional_sources","confirmation_dir","confirmation_categories","confirmation_rows_per_condition_category","mmlu_confirmation_rows"}
    if {k:v for k,v in old.items() if k not in allowed}!={k:v for k,v in e.c.items() if k not in allowed}:
        raise ValueError("Single-factor revision changed an unrelated configuration")
    old_data=ROOT/old["dataset_dir"]
    if sha256(old_data/"manifest.json")!=prior_state["fingerprint"]["data_manifest_sha256"]:
        raise ValueError("Historical data manifest changed")
    old_manifest=read(old_data/"manifest.json")
    matched={}
    for name in ("fit","probe_validation","selection","test","mmlu_fit","mmlu_probe_validation","mmlu_selection","mmlu_test"):
        a,b=old_data/f"{name}.jsonl",e.data/f"{name}.jsonl"
        if sha256(a)!=sha256(b):
            raise ValueError("Detector or evaluation inputs changed: "+name)
        if sha256(a)!=old_manifest["files"][f"{name}.jsonl"]["sha256"]:
            raise ValueError("Historical frozen input changed")
        matched[name]=sha256(b)
    if sha256(old_data/"fewshot.txt")!=sha256(e.data/"fewshot.txt") or read(prior/"environment.json")!=read(e.out/"environment.json"):
        raise ValueError("Prompt exemplars or model environment changed")
    checked=[]
    for filename,names in {"mechanism/fairsteer.py":[("FairRunner",None),("add_direction",None),("detector_label",None),("FairSteerExperiment","collect_probes")],"mechanism/fairsteer_data.py":[("FairPromptBuilder",None),("source_fewshot",None)]}.items():
        snapshot=prior/"source_snapshot"/filename
        if sha256(snapshot)!=prior_state["fingerprint"]["sources"][filename]:
            raise ValueError("Historical source snapshot changed")
        for name,method in names:
            if node_dump(ROOT/filename,name,method)!=node_dump(snapshot,name,method):
                raise ValueError("Detector computation changed: "+name)
            checked.append(name+("."+method if method else ""))
    previous=trusted(prior/"fitted.pkl")
    if sha256(prior/"fitted.pkl")!=read(prior/"test_plan_frozen.json")["fitted_sha256"]:
        raise ValueError("Historical frozen probe source changed")
    e.fitted=dict(directions=directions,probes=previous["probes"])
    e.save("fitted.pkl",e.fitted)
    diagnostic_bytes=(prior/"probe_diagnostics.json").read_bytes()
    (e.out/"probe_diagnostics.json").write_bytes(diagnostic_bytes)
    if (prior/"probe_stratified_audit.json").exists():
        (e.out/"probe_stratified_audit.json").write_bytes((prior/"probe_stratified_audit.json").read_bytes())
    write_json(e.out/"probe_reuse_verified.json",dict(prior=str(prior),prior_fitted_sha256=sha256(prior/"fitted.pkl"),matched_data=matched,probe_diagnostics_sha256=sha256(prior/"probe_diagnostics.json"),unchanged_detector_ast=checked,reason="Only direction evidence condition changes. Exact frozen probes reused; no new detector fitting or labels."))
    d=read(e.out/"probe_diagnostics.json")
    e.finish("fit",direction_pairs=len(e.rows["direction_fit"]),probe_train=sum(d["train_labels"].values()),probe_validation=sum(d["validation_labels"].values()),probes_reused=True)


def prepare_confirmation(config):
    from scripts.inspect_native_pairs import RAW, REV, load_categories, source_semantic_exclusion
    c=tomllib.loads(Path(config).read_text(encoding="utf-8"))
    out=ROOT/c["confirmation_dir"]
    if (out/"manifest.json").exists():
        m=read(out/"manifest.json")
        if m["config_sha256"]!=sha256(config):raise ValueError("Confirmation configuration changed")
        for name,digest in m["files"].items():
            if sha256(out/name)!=digest:raise ValueError("Confirmation data changed")
        return m
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for cat in c["confirmation_categories"]:
        path=RAW/f"data/{cat}.jsonl"
        if not path.exists():
            url=f"https://raw.githubusercontent.com/nyu-mll/BBQ/{REV}/data/{cat}.jsonl"
            with opener.open(url,timeout=40) as response:
                blob=response.read()
            for line in blob.decode("utf-8").splitlines():json.loads(line)
            path.write_bytes(blob)
    source=load_categories(c["confirmation_categories"])
    demos={p.splitlines()[0] for p in source_fewshot().strip().split("\n\n")}
    demo_groups={r["template_group_id"] for r in source if f'{r["context"]} {r["question"]}' in demos}
    used=read_rows(ROOT/c["dataset_dir"] / "fit.jsonl")+read_rows(ROOT/c["dataset_dir"] / "probe_validation.jsonl")+read_rows(ROOT/c["dataset_dir"] / "selection.jsonl")+read_rows(ROOT/c["dataset_dir"] / "test.jsonl")
    used_groups={r["template_group_id"] for r in used}
    used_fingerprints={r["prompt_fingerprint"] for r in used}
    pool=[r for r in source if not source_semantic_exclusion(r) and r["template_group_id"] not in used_groups|demo_groups and r["prompt_fingerprint"] not in used_fingerprints]
    rows=[]
    for cat in c["confirmation_categories"]:
        for condition in ("ambig","disambig"):
            samples=fair_sample([r for r in pool if r["category"]==cat and r["context_condition"]==condition],c["confirmation_rows_per_condition_category"],c["seed"]+91)
            if len(samples)!=c["confirmation_rows_per_condition_category"]:raise ValueError("Insufficient fresh confirmation rows")
            rows.extend(samples)
    rows=sorted(rows,key=lambda r:r["uid"])
    original=read_rows(ROOT/"research/sources/20261012/FairSteer/generate_data/mmlu.json")
    unique={}
    for item in original:
        key=rank(0," ".join(item["question"].casefold().split()))
        unique.setdefault(key,dict(item,uid="mmlu:"+key[:20],template_group_id="mmlu:"+key))
    ordered=sorted(unique.values(),key=lambda r:rank(c["seed"],r["uid"]))
    offset=sum(c[k] for k in ("mmlu_train_rows","mmlu_probe_validation_rows","mmlu_selection_rows","mmlu_test_rows"))
    mmlu=ordered[offset:offset+c["mmlu_confirmation_rows"]]
    out.mkdir(parents=True,exist_ok=True)
    frozen_rows(out/"confirmation.jsonl",rows);frozen_rows(out/"mmlu_confirmation.jsonl",mmlu)
    m=dict(config_sha256=sha256(config),rows=len(rows),families=len({r["template_group_id"] for r in rows}),categories=dict(collections.Counter(r["category"] for r in rows)),conditions=dict(collections.Counter(r["context_condition"] for r in rows)),demo_families_excluded=sorted(demo_groups),sources={cat:dict(revision=REV,sha256=sha256(RAW/f"data/{cat}.jsonl")) for cat in c["confirmation_categories"]},files={p.name:sha256(p) for p in out.glob("*.jsonl")},role="Unseen-category, unseen-family confirmation; distinct from opened-family paired diagnostics. These categories are not detector/direction training categories. Fixed demonstrations may mention the category; their families are excluded.")
    write_json(out/"manifest.json",m)
    return m


def freeze_revision(e):
    prior=ROOT/e.c["reuse_probe_from"]
    previous=read(prior/"test_plan_frozen.json")
    value=dict(source_fingerprint=e.fingerprint,confirmation_manifest_sha256=sha256(ROOT/e.c["confirmation_dir"]/"manifest.json"),previous_plan_sha256=sha256(prior/"test_plan_frozen.json"),previous_fitted_sha256=sha256(prior/"fitted.pkl"),previous_specs=[previous["primary"],previous["guarded"]["spec"]],rules=["Repeat full original search with ambiguous-only DSV; exactly reuse detectors", "Opened 320-row test is a paired diagnostic, never new confirmation", "Fixed-layer comparison includes previous primary and guarded layers with corrected DSV", "Same-primary-layer diagnostics: no gate, balanced alpha=1, balanced norm-matched, reverse native", "New-category confirmation evaluates corrected primary, guarded, balanced at primary, three matched randoms, previous fixed layers with corrected and historical DSV; never reselect", "All numeric choices for new confirmation derive from the frozen development plan and fit vectors, not opened diagnostics"],calibration="Original six offsets, chosen only on unchanged selection",constraints=e.c["constraints"])
    path=e.out/"revision_policy_frozen.json"
    if path.exists():
        if read(path)["policy"]!=value:raise ValueError("Revision policy changed")
    else:
        if (e.out/"test_plan_frozen.json").exists():raise ValueError("Cannot preregister after selection")
        write_json(path,dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),policy=value))


def evaluate_record(e,name,rows,mmlu,spec,baseline,bn,bm,fitted_path=None):
    from .fairsteer import bootstrap, constraints
    value=e.evaluate(name,rows,spec,fitted_path=fitted_path)
    neutral=e.evaluate(name+"_neutral",e.neutral["validation"],spec,fitted_path=fitted_path)
    capability=e.evaluate(name+"_mmlu",mmlu,spec,mmlu=True,fitted_path=fitted_path)
    ok,reasons=constraints(value["metrics"],neutral["metrics"],baseline["metrics"],bn["metrics"],e.c["constraints"],capability["accuracy"],bm["accuracy"])
    return dict(spec=spec,metrics=value["metrics"],native_metrics=value["native_metrics"],neutral=neutral["metrics"],mmlu_accuracy=capability["accuracy"],perturbation=value["perturbation"],feasible=ok,violations=reasons,confidence=bootstrap(rows,baseline["predictions"],value["predictions"],e.c["bootstrap_replicates"],e.c["seed"]),fitted_sha256=sha256(fitted_path or e.out/"fitted.pkl"),prediction_name=name)


def additional_tests(e):
    from .fairsteer import calibrated,summarize,constraints,bootstrap
    e.setup()
    policy=read(e.out/"revision_policy_frozen.json")
    if policy["policy"]["source_fingerprint"]!=e.fingerprint:raise ValueError("Predeclared source changed")
    plan=read(e.out/"test_plan_frozen.json")
    layer=plan["primary"]["layer"]
    native,balanced=(float(np.linalg.norm(e.fitted["directions"][k][layer])) for k in ("native","balanced"))
    extra=[e.spec("native_gated",s["layer"],s["alpha"]) for s in policy["policy"]["previous_specs"]]
    extra += [e.spec("native_always",layer),e.spec("balanced_gated",layer),e.spec("native_gated",layer,-1.)]
    if balanced>1e-12:extra.append(e.spec("balanced_gated",layer,native/balanced))
    unique=lambda items:list({s["id"]:s for s in items}.values())
    extra=unique(extra)
    fresh=unique([plan["primary"]]+([plan["guarded"]["spec"]] if plan["guarded"] else [])+[e.spec("balanced_gated",layer)]+[s for s in plan["configs"] if s["variant"]=="random_gated"]+[e.spec("native_gated",s["layer"],s["alpha"]) for s in policy["policy"]["previous_specs"]])
    resolved=dict(policy_sha256=sha256(e.out/"revision_policy_frozen.json"),development_plan_sha256=sha256(e.out/"test_plan_frozen.json"),corrected_fitted_sha256=sha256(e.out/"fitted.pkl"),paired_extra=extra,confirmation=fresh,historical_confirmation=policy["policy"]["previous_specs"],norms=dict(native=native,balanced=balanced))
    path=e.out/"revision_numeric_plan.json"
    if path.exists() and read(path)!=resolved:raise ValueError("Resolved numeric policy changed")
    write_json(path,resolved)
    rows=read_rows(e.data/"test.jsonl");mmlu=read_rows(e.data/"mmlu_test.jsonl")
    base=e.evaluate("test_baseline",rows);bn=e.evaluate("test_neutral_baseline",e.neutral["validation"]);bm=e.evaluate("test_mmlu_baseline",mmlu,mmlu=True)
    main=read(e.out/"test_results.json");ids={r["spec"]["id"] for r in main}
    results=[]
    for spec in extra:
        if spec["id"] in ids:continue
        results.append(evaluate_record(e,"paired_"+spec["id"],rows,mmlu,spec,base,bn,bm))
        write_json(e.out/"paired_extra_results.json",results)
    root=ROOT/e.c["confirmation_dir"]
    if sha256(root/"manifest.json")!=policy["policy"]["confirmation_manifest_sha256"]:raise ValueError("Confirmation manifest changed")
    manifest=read(root/"manifest.json")
    for name,digest in manifest["files"].items():
        if sha256(root/name)!=digest:raise ValueError("Confirmation changed")
    lock=e.out/"confirmation_lock.json"
    identity=dict(numeric_plan_sha256=sha256(path),manifest_sha256=sha256(root/"manifest.json"))
    if lock.exists() and read(lock)!=identity:raise ValueError("Confirmation lock changed")
    write_json(lock,identity)
    rows=read_rows(root/"confirmation.jsonl");mmlu=read_rows(root/"mmlu_confirmation.jsonl")
    base=e.evaluate("confirmation_baseline",rows);bn=e.evaluate("confirmation_baseline_neutral",e.neutral["validation"]);bm=e.evaluate("confirmation_baseline_mmlu",mmlu,mmlu=True)
    results=[]
    for spec in fresh:
        results.append(evaluate_record(e,"confirmation_corrected_"+spec["id"],rows,mmlu,spec,base,bn,bm))
        write_json(e.out/"confirmation_results.json",results)
    prior=ROOT/e.c["reuse_probe_from"]/"fitted.pkl"
    if sha256(prior)!=policy["policy"]["previous_fitted_sha256"]:raise ValueError("Historical reference changed")
    for spec in policy["policy"]["previous_specs"]:
        results.append(evaluate_record(e,"confirmation_historical_"+spec["id"],rows,mmlu,spec,base,bn,bm,prior))
        write_json(e.out/"confirmation_results.json",results)
    pred=calibrated(rows,base["predictions"],plan["calibration_offset"])
    neutral=calibrated(e.neutral["validation"],bn["predictions"],plan["calibration_offset"])
    cm,cn=summarize(rows,pred),summarize(e.neutral["validation"],neutral)
    ok,reasons=constraints(cm,cn,base["metrics"],bn["metrics"],e.c["constraints"],bm["accuracy"],bm["accuracy"])
    write_json(e.out/"confirmation_calibration.json",dict(offset=plan["calibration_offset"],metrics=cm,neutral=cn,feasible=ok,violations=reasons,confidence=bootstrap(rows,base["predictions"],pred,e.c["bootstrap_replicates"],e.c["seed"])))
    e.finish("additional_tests",paired_extra=len(extra)-len(ids&{s["id"] for s in extra}),confirmation_configs=len(results))
