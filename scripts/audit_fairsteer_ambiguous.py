"""Read-only independent arithmetic, isolation and frozen-plan audit of revision v2."""
from __future__ import annotations

import ast
import collections
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_fairsteer import independent_metrics
from mechanism.data import neutral_controls

OUT = ROOT / "results/fairsteer_ambiguous_v2"
OLD = ROOT / "results/fairsteer_20261012_v1"
DATA = ROOT / "data/prepared/bbq_fairsteer_ambiguous_v2"
PRIOR_DATA = ROOT / "data/prepared/bbq_fairsteer_20261012_v1"
FRESH = ROOT / "data/prepared/bbq_fairsteer_confirmation_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trusted_local(path):
    assert digest(path) == read(path.with_suffix(path.suffix + ".sha256.json"))["sha256"]
    with path.open("rb") as handle:
        return pickle.load(handle)


def signature(spec):
    return spec["variant"], spec["layer"], spec["alpha"], spec.get("seed")


def check_record(record, name, base_name, neutral_name, mmlu_name, base_neutral_name, base_mmlu_name, limits):
    """Recalculate declared guardrails without importing the selection function."""
    prediction = read(OUT / f"predictions/{name}.json")
    neutral = read(OUT / f"predictions/{neutral_name}.json")["metrics"]
    mmlu_accuracy = read(OUT / f"predictions/{mmlu_name}.json")["accuracy"]
    base = read(OUT / f"predictions/{base_name}.json")["metrics"]
    bn = read(OUT / f"predictions/{base_neutral_name}.json")["metrics"]
    bm = read(OUT / f"predictions/{base_mmlu_name}.json")["accuracy"]
    assert prediction["identity"]["spec"] == record["spec"]
    assert prediction["metrics"] == record["metrics"] and prediction["native_metrics"] == record["native_metrics"]
    assert neutral == record["neutral"] and mmlu_accuracy == record["mmlu_accuracy"]
    assert prediction["perturbation"] == record["perturbation"]
    changed, reasons = record["metrics"], set()
    checks = [("disambiguated_accuracy", changed["disambiguated_accuracy"], base["disambiguated_accuracy"], limits["disambiguated_accuracy_drop"]),
              ("neutral_accuracy", neutral["accuracy"], bn["accuracy"], limits["neutral_accuracy_drop"]),
              ("macro_ambiguous_accuracy", changed["macro_ambiguous_accuracy"], base["macro_ambiguous_accuracy"], limits["ambiguous_accuracy_drop"]),
              ("mmlu_accuracy", mmlu_accuracy, bm, limits["neutral_accuracy_drop"])]
    checks += [("category:" + cat, changed["categories"][cat]["disambiguated_accuracy"], values["disambiguated_accuracy"], limits["category_disambiguated_accuracy_drop"]) for cat, values in base["categories"].items()]
    for key, value, reference, tolerance in checks:
        if reference - value > tolerance + 1e-12:
            reasons.add(key)
    for key in ("counterstereotype_rate", "macro_counterstereotype"):
        if changed[key] - base[key] > limits["counterstereotype_increase"] + 1e-12:
            reasons.add(key)
    if neutral["disambiguated_overabstention"] - bn["disambiguated_overabstention"] > limits["neutral_overabstention_increase"] + 1e-12:
        reasons.add("neutral_overabstention")
    assert record["feasible"] == (not reasons) and set(record["violations"]) == reasons, name


def main():
    state = read(OUT / "manifest.json")
    preflight = "--preflight" in sys.argv
    in_progress = "--in-progress" in sys.argv
    required = ("fit", "checks") if preflight else ("fit", "checks", "screen", "select", "test", "stability", "report") + (() if in_progress else ("additional_tests",))
    assert all(state["stages"].get(s, {}).get("completed") for s in required)
    frozen = state["fingerprint"]
    config_path = ROOT / "configs/experiment_fairsteer_ambiguous.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert digest(config_path) == frozen["config_sha256"]
    assert digest(OUT / "source_snapshot/experiment_fairsteer_ambiguous.toml") == frozen["config_sha256"]
    assert digest(DATA / "manifest.json") == frozen["data_manifest_sha256"]
    for name, expected in frozen["sources"].items():
        assert digest(ROOT / name) == digest(OUT / "source_snapshot" / name) == expected
    data, fresh_manifest = read(DATA / "manifest.json"), read(FRESH / "manifest.json")
    assert data["direction_context"] == "ambig" and data["test_role"] == "opened_family_paired_diagnostic"
    for name, record in data["files"].items():
        assert digest(DATA / name) == record["sha256"]
    for name, record in data["sources"].items():
        assert digest(ROOT / "data/raw/bbq" / record["revision"] / name) == record["sha256"]
    for cat, record in fresh_manifest["sources"].items():
        assert digest(ROOT / "data/raw/bbq" / record["revision"] / f"data/{cat}.jsonl") == record["sha256"]
    assert fresh_manifest["config_sha256"] == frozen["config_sha256"]
    for name, expected in fresh_manifest["files"].items():
        assert digest(FRESH / name) == expected

    splits = {s: rows(DATA / f"{s}.jsonl") for s in ("fit", "probe_validation", "selection", "test")}
    fresh_rows = rows(FRESH / "confirmation.jsonl")
    direction = rows(DATA / "direction_fit.jsonl")
    assert len(direction) == 110
    assert all(r["context_condition"] == "ambig" and r["correct_index"] == r["unknown_index"]
               and r["correct_index"] != r["stereotype_index"] for r in direction)
    assert {r["uid"] for r in direction} <= {r["uid"] for r in splits["fit"]}
    groups = [{r["template_group_id"] for r in source} for source in [*splits.values(), fresh_rows]]
    for i, group in enumerate(groups):
        for other in groups[i + 1:]:
            assert not group & other
    old_fingerprints = {r["prompt_fingerprint"] for source in splits.values() for r in source}
    assert not old_fingerprints & {r["prompt_fingerprint"] for r in fresh_rows}
    assert len(fresh_rows) == 384 and len(groups[-1]) == 91
    assert not groups[-1] & set(fresh_manifest["demo_families_excluded"])
    assert not {r["category"] for r in fresh_rows} & {r["category"] for r in splits["fit"]}
    mmlu = {s: rows(DATA / f"mmlu_{s}.jsonl") for s in splits}
    mmlu["confirmation"] = rows(FRESH / "mmlu_confirmation.jsonl")
    seen = set()
    for source in mmlu.values():
        ids = {r["uid"] for r in source}
        assert len(ids) == len(source) and not ids & seen
        seen |= ids

    reuse = read(OUT / "probe_reuse_verified.json")
    for name, expected in reuse["matched_data"].items():
        assert digest(DATA / f"{name}.jsonl") == digest(PRIOR_DATA / f"{name}.jsonl") == expected
    assert digest(DATA / "fewshot.txt") == digest(PRIOR_DATA / "fewshot.txt")
    assert digest(OUT / "probe_diagnostics.json") == digest(OLD / "probe_diagnostics.json") == reuse["probe_diagnostics_sha256"]
    assert read(OUT / "environment.json") == read(OLD / "environment.json")
    old_state = read(OLD / "manifest.json")
    old_config = OLD / "source_snapshot/experiment_fairsteer.toml"
    assert digest(old_config) == old_state["fingerprint"]["config_sha256"]
    allowed = {"dataset_dir", "output_dir", "direction_context", "test_role", "reuse_probe_from",
               "additional_sources", "confirmation_dir", "confirmation_categories",
               "confirmation_rows_per_condition_category", "mmlu_confirmation_rows"}
    historical_config = tomllib.loads(old_config.read_text(encoding="utf-8"))
    assert {k: v for k, v in config.items() if k not in allowed} == {k: v for k, v in historical_config.items() if k not in allowed}
    for filename, names in {"mechanism/fairsteer.py": [("FairRunner", None), ("add_direction", None), ("detector_label", None), ("FairSteerExperiment", "collect_probes")],
                            "mechanism/fairsteer_data.py": [("FairPromptBuilder", None), ("source_fewshot", None)]}.items():
        snapshot = OLD / "source_snapshot" / filename
        assert digest(snapshot) == old_state["fingerprint"]["sources"][filename]
        for name, method in names:
            dumps = []
            for path in (snapshot, ROOT / filename):
                node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if getattr(n, "name", None) == name)
                if method:
                    node = next(n for n in node.body if getattr(n, "name", None) == method)
                dumps.append(ast.dump(node, include_attributes=False))
            assert dumps[0] == dumps[1]
    current, historical = trusted_local(OUT / "fitted.pkl"), trusted_local(OLD / "fitted.pkl")
    assert len(current["probes"]) == len(historical["probes"]) == 28
    for a, b in zip(current["probes"], historical["probes"]):
        assert np.array_equal(a["coef"], b["coef"]) and a["intercept"] == b["intercept"]
    assert not np.array_equal(current["directions"]["native"], historical["directions"]["native"])
    if preflight:
        print("Preflight passed: ambiguous-only 110 rows, eight unchanged input files, 28 exact probes, source snapshots and disjoint confirmation verified. Final predictions still pending.")
        return

    policy_record = read(OUT / "revision_policy_frozen.json")
    policy = policy_record["policy"]
    plan, numeric = read(OUT / "test_plan_frozen.json"), read(OUT / "revision_numeric_plan.json")
    old_plan = read(OLD / "test_plan_frozen.json")
    assert policy["source_fingerprint"] == frozen
    assert policy["previous_specs"] == [old_plan["primary"], old_plan["guarded"]["spec"]]
    assert policy["previous_plan_sha256"] == digest(OLD / "test_plan_frozen.json")
    assert policy["previous_fitted_sha256"] == reuse["prior_fitted_sha256"] == old_plan["fitted_sha256"] == digest(OLD / "fitted.pkl")
    assert policy["confirmation_manifest_sha256"] == digest(FRESH / "manifest.json")
    assert numeric["policy_sha256"] == digest(OUT / "revision_policy_frozen.json")
    assert numeric["development_plan_sha256"] == digest(OUT / "test_plan_frozen.json")
    assert numeric["corrected_fitted_sha256"] == plan["fitted_sha256"] == digest(OUT / "fitted.pkl")
    assert plan["selection_results_sha256"] == digest(OUT / "selection_results.json")
    assert read(OUT / "test_lock.json") == dict(plan_sha256=digest(OUT / "test_plan_frozen.json"), test_sha256=digest(DATA / "test.jsonl"))
    assert read(OUT / "confirmation_lock.json") == dict(numeric_plan_sha256=digest(OUT / "revision_numeric_plan.json"), manifest_sha256=digest(FRESH / "manifest.json"))
    events = rows(OUT / "events.jsonl")
    assert policy_record["created"] <= next(e["time"] for e in events if e["event"] == "direction_pairs")
    assert policy_record["created"] < next(e["time"] for e in events if e["event"] == "select_complete")
    layer = plan["primary"]["layer"]
    native_norm, balanced_norm = (float(np.linalg.norm(current["directions"][k][layer])) for k in ("native", "balanced"))
    assert numeric["norms"] == dict(native=native_norm, balanced=balanced_norm)
    expected_extra = {signature(s) for s in policy["previous_specs"]}
    expected_extra |= {("native_always", layer, 1., None), ("balanced_gated", layer, 1., None), ("native_gated", layer, -1., None)}
    if balanced_norm > 1e-12:
        expected_extra.add(("balanced_gated", layer, native_norm / balanced_norm, None))
    assert {signature(s) for s in numeric["paired_extra"]} == expected_extra
    expected_fresh = {signature(plan["primary"]), ("balanced_gated", layer, 1., None)}
    if plan["guarded"]:
        expected_fresh.add(signature(plan["guarded"]["spec"]))
    expected_fresh |= {signature(s) for s in plan["configs"] if s["variant"] == "random_gated"}
    expected_fresh |= {signature(s) for s in policy["previous_specs"]}
    assert {signature(s) for s in numeric["confirmation"]} == expected_fresh
    assert numeric["historical_confirmation"] == policy["previous_specs"]
    screen = read(OUT / "screen_results.json")
    assert len(screen) == 84
    for variant in ("native_gated", "native_always", "balanced_gated"):
        assert {r["spec"]["layer"] for r in screen if r["spec"]["variant"] == variant} == set(range(28))
    shortlist = []
    for variant in ("native_gated", "native_always", "balanced_gated"):
        options = sorted([r for r in screen if r["spec"]["variant"] == variant], key=lambda r: (-r["native_metrics"]["accuracy"], abs(r["native_metrics"]["signed_ambiguous_bbq_bias"]), r["spec"]["layer"]))
        shortlist.extend(signature(r["spec"]) for r in options[:config["finalists_per_variant"]])
    strength_layer = next(s[1] for s in shortlist if s[0] == "native_gated")
    shortlist.extend(("native_gated", strength_layer, a, None) for a in config["strengths"] if a != 1.)
    assert {signature(r["spec"]) for r in read(OUT / "selection_results.json")} == set(shortlist)

    screen_panel = read(OUT / "search_plan_frozen.json")["panel_uids"]
    lookup = {r["uid"]: r for r in splits["selection"]}
    panel = [lookup[uid] for uid in screen_panel]
    checked, max_error = 0, 0.
    for path in sorted((OUT / "predictions").glob("*.json")):
        p = read(path)
        fresh = path.name.startswith("confirmation_")
        opened = path.name.startswith(("test_", "paired_"))
        split = "test" if opened else "selection"
        if p["identity"]["mmlu"]:
            source = mmlu["confirmation" if fresh else split]
        elif "neutral" in path.name:
            source = neutral_controls()["validation" if fresh or opened else "selection"]
        elif fresh:
            source = fresh_rows
        elif path.name.startswith("screen_"):
            source = panel
        else:
            source = splits[split]
        assert p["identity"]["rows_sha256"] == hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(), path.name
        artifact_sha = digest(OLD / "fitted.pkl") if path.name.startswith("confirmation_historical_") else digest(OUT / "fitted.pkl")
        assert p["identity"]["fitted_sha256"] == (artifact_sha if p["identity"]["spec"] else None)
        if p["identity"]["mmlu"]:
            answers = {r["uid"]: r["answer"] for r in source}
            assert len(p["predictions"]) == len(answers) and {x["uid"] for x in p["predictions"]} == set(answers)
            assert all(len(x["rotations"]) == 6 for x in p["predictions"])
            value = sum(rot["prediction"] == answers[item["uid"]] for item in p["predictions"] for rot in item["rotations"]) / (6 * len(source))
            error = abs(value - p["accuracy"])
            max_error = max(max_error, error)
            assert error < 1e-10
        else:
            for native, key in ((False, "metrics"), (True, "native_metrics")):
                for metric, value in independent_metrics(source, p["predictions"], native).items():
                    error = abs(value - p[key][metric])
                    max_error = max(max_error, error)
                    assert error < 1e-10, (path.name, metric, error)
                prediction_lookup = {item["uid"]: item for item in p["predictions"]}
                masses = []
                category_scores = collections.defaultdict(list)
                for row in source:
                    rotations = prediction_lookup[row["uid"]]["rotations"][:1] if native else prediction_lookup[row["uid"]]["rotations"]
                    masses.extend(rot["candidate_mass"] for rot in rotations)
                    if row["context_condition"] == "disambig":
                        category_scores[row["category"]].append(sum(rot["prediction"] == row["correct_index"] for rot in rotations) / len(rotations))
                values = [(sum(masses) / len(masses), p[key]["candidate_mass"])]
                values += [(sum(scores) / len(scores), p[key]["categories"][cat]["disambiguated_accuracy"]) for cat, scores in category_scores.items()]
                for value, reported in values:
                    error = abs(value - reported)
                    max_error = max(max_error, error)
                    assert error < 1e-10, path.name
        checked += 1

    test, extra, confirmation = (read(OUT / name) for name in ("test_results.json", "paired_extra_results.json", "confirmation_results.json"))
    assert {r["spec"]["id"] for r in test} == {s["id"] for s in plan["configs"]}
    assert {signature(r["spec"]) for r in extra} == expected_extra - {signature(r["spec"]) for r in test}
    expected_names = {"confirmation_corrected_" + s["id"] for s in numeric["confirmation"]} | {"confirmation_historical_" + s["id"] for s in numeric["historical_confirmation"]}
    if in_progress:
        assert {r["prediction_name"] for r in confirmation} <= expected_names
    else:
        assert {r["prediction_name"] for r in confirmation} == expected_names
    selection = read(OUT / "selection_results.json")
    for r in selection:
        identifier = r["spec"]["id"]
        check_record(r, "selection_" + identifier, "selection_baseline", "selection_neutral_" + identifier,
                     "selection_mmlu_" + identifier, "selection_neutral_baseline", "selection_mmlu_baseline", config["constraints"])
        assert r["accepted"] == bool(r["feasible"] and r["macro_stereotype_reduction"] >= config["minimum_selection_stereotype_reduction"])
    primary_candidates = [r for r in selection if r["spec"]["variant"] == "native_gated" and r["spec"]["alpha"] == 1.]
    best = min(primary_candidates, key=lambda r: (-r["native_metrics"]["accuracy"], abs(r["native_metrics"]["signed_ambiguous_bbq_bias"]), r["spec"]["layer"]))
    assert best["spec"] == plan["primary"]
    feasible = [r for r in selection if r["feasible"]]
    guarded = min(feasible, key=lambda r: (-r["macro_stereotype_reduction"], r["metrics"]["macro_stereotype_unknown_margin"], r["spec"]["id"])) if feasible else None
    assert guarded == plan["guarded"]
    for r in test:
        identifier = r["spec"]["id"]
        check_record(r, "test_" + identifier, "test_baseline", "test_neutral_" + identifier,
                     "test_mmlu_" + identifier, "test_neutral_baseline", "test_mmlu_baseline", config["constraints"])
    for r in extra:
        name = r["prediction_name"]
        check_record(r, name, "test_baseline", name + "_neutral", name + "_mmlu", "test_neutral_baseline", "test_mmlu_baseline", config["constraints"])
    for r in confirmation:
        name = r["prediction_name"]
        check_record(r, name, "confirmation_baseline", name + "_neutral", name + "_mmlu", "confirmation_baseline_neutral", "confirmation_baseline_mmlu", config["constraints"])
    for dataset, records, base_name in ((splits["test"], test + extra, "test_baseline"), (fresh_rows, confirmation, "confirmation_baseline")):
        baseline = read(OUT / f"predictions/{base_name}.json")["metrics"]
        primary = next(r for r in records if r["spec"]["id"] == plan["primary"]["id"] and "historical" not in r.get("prediction_name", ""))
        for record in records:
            if record["spec"]["layer"] == layer and record["spec"]["variant"] != "native_always":
                assert abs(record["perturbation"]["trigger_fraction"] - primary["perturbation"]["trigger_fraction"]) < 1e-12
            for key in ("macro_stereotype", "macro_counterstereotype", "macro_ambiguous_accuracy"):
                assert abs(record["confidence"][key]["delta"] - (record["metrics"][key] - baseline[key])) < 1e-10
    baseline_flips, baseline_max_log_delta = 0, 0.
    baseline_files = ("screen_baseline", "selection_baseline", "selection_neutral_baseline", "selection_mmlu_baseline", "test_baseline", "test_neutral_baseline", "test_mmlu_baseline")
    for name in baseline_files:
        a, b = (read(root / f"predictions/{name}.json")["predictions"] for root in (OLD, OUT))
        assert [p["uid"] for p in a] == [p["uid"] for p in b]
        for p, q in zip(a, b):
            for old_rot, new_rot in zip(p["rotations"], q["rotations"]):
                baseline_flips += old_rot["prediction"] != new_rot["prediction"]
                baseline_max_log_delta = max(baseline_max_log_delta, max(abs(x-y) for x, y in zip(old_rot["semantic_logprobs"], new_rot["semantic_logprobs"])))
    assert baseline_flips == 0, "Unchanged baseline drifted; investigate before attributing differences"
    if not in_progress:
        contrasts = read(OUT / "protocol_correction_contrasts.json")
        assert len(contrasts) == 4
        historical_results = read(OLD / "test_results.json")
        for record in contrasts:
            identifier = record["id"]
            if record["scope"] == "paired":
                a = next(r for r in historical_results if r["spec"]["id"] == identifier)
                c = next(r for r in test + extra if r["spec"]["id"] == identifier)
            else:
                a = next(r for r in confirmation if r["prediction_name"] == "confirmation_historical_" + identifier)
                c = next(r for r in confirmation if r["prediction_name"] == "confirmation_corrected_" + identifier)
            for key, value in record["metrics"].items():
                assert abs(value["delta"] - (c["metrics"][key] - a["metrics"][key])) < 1e-10
                assert np.isfinite([value["delta"], value["low"], value["high"]]).all() and value["low"] <= value["high"]
    report = dict(status="in_progress" if in_progress else "passed", prediction_files_checked=checked, max_metric_error=max_error,
                  direction_conditions=dict(collections.Counter(r["context_condition"] for r in direction)),
                  unchanged_probe_layers=28, unchanged_data_files=8, unchanged_baselines_checked=len(baseline_files),
                  baseline_prediction_flips=baseline_flips, baseline_max_logprob_delta=baseline_max_log_delta,
                  confirmation_rows=len(fresh_rows), confirmation_families=len(groups[-1]),
                  confirmation_role="Unseen-category, unseen-family; opened 320-row set is paired diagnostic only",
                  frozen_policies="verified", source_and_data_hashes="verified",
                  scope="Independent arithmetic, artifacts, isolation, frozen rules and controls; not proof of causality or statistical power")
    if not in_progress:
        (OUT / "independent_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
