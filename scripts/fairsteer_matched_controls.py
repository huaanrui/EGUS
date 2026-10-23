"""Predeclare matched-layer controls, then evaluate them without reselection.

Freeze the functional policy before selection/test. Numeric specs are determined
only by the main plan's pre-test primary and fit vector norms, never test outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mechanism.data import read_rows,sha256,write_json

OUT=ROOT/"results/fairsteer_20261012_v1"


def freeze():
    path=OUT/"matched_control_policy_frozen.json"
    if path.exists():
        policy=json.loads(path.read_text(encoding="utf-8"))
        if policy["script_sha256"]!=sha256(Path(__file__)):
            raise ValueError("Predeclared control implementation changed")
        return
    if (OUT/"test_plan_frozen.json").exists() or (OUT/"test_lock.json").exists():
        raise ValueError("Cannot preregister matched controls after selection/test")
    write_json(path,dict(
        version=1,created=time.strftime("%Y-%m-%d %H:%M:%S"),script_sha256=sha256(Path(__file__)),
        rules=["same primary layer, native direction, alpha=1, no gate",
               "same primary layer, balanced direction, alpha=1, original gate",
               "same primary layer, balanced direction, alpha=norm(native)/norm(balanced), original gate; skip zero balanced vector",
               "same primary layer, native direction, alpha=-1, original gate"],
        decisions="Primary layer comes only from main pre-test selection. Norms come only from frozen fit. These are fixed diagnostic controls, never candidates to replace primary.",
        timing="Declared during fit, before selection or confirmation results; fit direction norms were available.",
    ))


def run():
    from mechanism.fairsteer import FairSteerExperiment,bootstrap,constraints
    from mechanism.fairsteer_report import render
    import numpy as np
    policy_path=OUT/"matched_control_policy_frozen.json"
    policy=json.loads(policy_path.read_text(encoding="utf-8"))
    if policy["script_sha256"]!=sha256(Path(__file__)):
        raise ValueError("Predeclared control implementation changed")
    e=FairSteerExperiment(ROOT/"configs/experiment_fairsteer.toml")
    if not e.state["stages"].get("test",{}).get("completed"):
        raise ValueError("Finish the main frozen confirmation first")
    e.setup()
    plan=json.loads((OUT/"test_plan_frozen.json").read_text())
    layer=plan["primary"]["layer"]
    norms={k:float(np.linalg.norm(e.fitted["directions"][k][layer])) for k in ("native","balanced")}
    specs=[e.spec("native_always",layer),e.spec("balanced_gated",layer),e.spec("native_gated",layer,-1.)]
    if norms["balanced"]>1e-12:
        specs.append(e.spec("balanced_gated",layer,norms["native"]/norms["balanced"]))
    resolved=dict(policy_sha256=sha256(policy_path),primary_plan_sha256=sha256(OUT/"test_plan_frozen.json"),fitted_sha256=sha256(OUT/"fitted.pkl"),norms=norms,specs=specs,numeric_resolution="Deterministic application of the pre-selection rule; no test outcome consulted")
    path=OUT/"matched_control_plan_resolved.json"
    if path.exists() and json.loads(path.read_text())!=resolved:
        raise ValueError("Resolved policy changed")
    write_json(path,resolved)
    main=json.loads((OUT/"test_results.json").read_text())
    main_ids={r["spec"]["id"] for r in main}
    rows=read_rows(e.data/"test.jsonl")
    mmlu=read_rows(e.data/"mmlu_test.jsonl")
    base=e.evaluate("test_baseline",rows)
    bn=e.evaluate("test_neutral_baseline",e.neutral["validation"])
    bm=e.evaluate("test_mmlu_baseline",mmlu,mmlu=True)
    result=[]
    for spec in specs:
        if spec["id"] in main_ids:
            continue
        value=e.evaluate("test_"+spec["id"],rows,spec)
        neutral=e.evaluate("test_neutral_"+spec["id"],e.neutral["validation"],spec)
        cap=e.evaluate("test_mmlu_"+spec["id"],mmlu,spec,mmlu=True)
        ok,reasons=constraints(value["metrics"],neutral["metrics"],base["metrics"],bn["metrics"],e.c["constraints"],cap["accuracy"],bm["accuracy"])
        result.append(dict(spec=spec,metrics=value["metrics"],native_metrics=value["native_metrics"],perturbation=value["perturbation"],neutral=neutral["metrics"],mmlu_accuracy=cap["accuracy"],feasible=ok,violations=reasons,confidence=bootstrap(rows,base["predictions"],value["predictions"],e.c["bootstrap_replicates"],e.c["seed"]),role="Predeclared same-layer diagnostic; no reselection"))
        write_json(OUT/"matched_control_results.json",result)
    write_json(OUT/"matched_control_complete.json",dict(policy_sha256=sha256(policy_path),plan_sha256=sha256(path),added=len(result),reused=sorted(main_ids&{s["id"] for s in specs})))
    render(OUT,e.data)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("freeze","run"));args=p.parse_args()
    freeze() if args.stage=="freeze" else run()
