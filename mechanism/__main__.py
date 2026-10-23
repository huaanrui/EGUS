from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import transformers

from .attribution import node_scores
from .data import balanced_panel, neutral_controls, read_rows, sha256, write_json
from .evaluate import feasible, metrics, paired_cluster_ci
from .grid import make_grid
from .interventions import Intervention
from .model import Runner
from .selectors import attach_attribution, discovery_classes, fit_selectors

ROOT = Path(__file__).resolve().parents[1]


def save_pickle(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        pickle.dump(value, f, protocol=5)
    temp.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256(path), encoding="ascii")


def load_pickle(path):
    # Only this run's locally generated artifacts; never load downloaded pickle files.
    if sha256(path) != path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii"):
        raise ValueError(f"Artifact hash mismatch: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


class Experiment:
    def __init__(self, config_path):
        self.config_path = config_path.resolve()
        self.config = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
        self.out = ROOT / self.config["output_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        dataset = ROOT / self.config["dataset_dir"]
        from scripts.prepare_bbq import verify
        self.integrity = verify(dataset)
        self.rows = {split: read_rows(dataset / f"{split}.jsonl") for split in ("fit", "selection", "validation")}
        self.neutral = neutral_controls()
        self.runner = None
        self.artifacts = None
        self.code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "mechanism").glob("*.py"))}
        identity = dict(config_sha256=sha256(self.config_path), dataset_manifest_sha256=sha256(dataset / "manifest.json"), model_config_sha256=sha256(Path(self.config["model_path"]) / "config.json"))
        manifest_path = self.out / "run_manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self.manifest["identity"] != identity:
                raise ValueError("Run inputs changed; use a new output_dir")
        else:
            self.manifest = dict(identity=identity, config=self.config, python=sys.version, torch=torch.__version__, transformers=transformers.__version__, platform=platform.platform(), stages={}, status="started", limitations=["Exploratory single-model study", "Synthetic neutral checks do not measure general capability", "15 validation families; wide cluster uncertainty", "Node rankings are not a full circuit", "Dataset may have been seen during pretraining"])
            self.manifest["dataset_counts"] = {s: dict(rows=len(rs), families=len({r["template_group_id"] for r in rs})) for s, rs in self.rows.items()}
            write_json(manifest_path, self.manifest)
        for name, value in (("grid_preregistered.json", make_grid(self.config)), ("neutral_controls.json", self.neutral)):
            p = self.out / name
            if p.exists() and json.loads(p.read_text(encoding="utf-8")) != value:
                raise ValueError(f"Frozen run input changed: {name}")
            if not p.exists():
                write_json(p, value)

    def log(self, event, **values):
        item = dict(event=event, **values)
        with (self.out / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)

    def finish(self, stage, **values):
        self.manifest["stages"][stage] = dict(completed=True, code_sha256=self.code_hashes, **values)
        self.manifest["status"] = "complete" if stage == "report" else f"{stage}_complete"
        if self.runner is not None and torch.cuda.is_available():
            self.manifest["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
        write_json(self.out / "run_manifest.json", self.manifest)

    def get_runner(self):
        if self.runner is None:
            self.runner = Runner(self.config["model_path"], self.config["max_input_tokens"], rotation_batch_size=self.config.get("answer_rotation_batch_size", 3))
            self.runner.install(self.config["layers"])
            self.manifest["gpu"] = torch.cuda.get_device_name()
        return self.runner

    def get_artifacts(self):
        if self.artifacts is None:
            self.artifacts = load_pickle(self.out / "selectors.pkl")
        return self.artifacts

    def fit(self):
        path = self.out / "fit_features.pkl"
        if path.exists():
            self.log("resume_fit_features")
            collected = load_pickle(path)
        else:
            runner = self.get_runner()
            audit = runner.builder.audit(sum(self.rows.values(), []) + sum(self.neutral.values(), []))
            write_json(self.out / "prompt_audit.json", audit)
            self.log("fit_start", rows=len(self.rows["fit"]), layers=self.config["layers"])
            collected = runner.collect(self.rows["fit"], self.config["layers"], lambda done, total, seconds: self.log("fit_progress", done=done, total=total, seconds=round(seconds, 1)))
            save_pickle(path, collected)
            write_json(self.out / "fit_baseline.json", dict(metrics=metrics(self.rows["fit"], collected["predictions"]), predictions=collected["predictions"], seconds=collected["seconds"]))
        write_json(self.out / "fit_baseline.json", dict(metrics=metrics(self.rows["fit"], collected["predictions"]), predictions=collected["predictions"], seconds=collected["seconds"]))
        classes, audit = discovery_classes(self.rows["fit"], collected["predictions"], self.config["min_examples_per_class"], self.config["min_groups_per_class"], self.config["discovery_cap_per_class"], self.config["seed"])
        write_json(self.out / "discovery_classes.json", audit)
        artifacts = fit_selectors(self.get_runner(), self.rows["fit"], collected, classes, self.config["layers"])
        save_pickle(self.out / "selectors.pkl", artifacts)
        self.artifacts = artifacts
        self.finish("fit", discovery=audit)
        self.log("fit_complete", discovery={k: {a: b for a, b in v.items() if a != "uids"} for k, v in audit.items()})

    def attribute(self):
        path = self.out / "relp_scores.pkl"
        artifacts = self.get_artifacts()
        if path.exists():
            attributed = load_pickle(path)
        else:
            runner = self.get_runner()
            discovery = [self.rows["fit"][i] for i in artifacts["discovery_indices"]]
            self.log("attribution_start", rows=len(discovery))
            attributed = node_scores(runner, discovery, self.config["layers"], artifacts["role_means"], progress=lambda done, total: self.log("attribution_progress", done=done, total=total), sample_weights=artifacts["discovery_weights"])
            save_pickle(path, attributed)
        attach_attribution(artifacts, attributed)
        save_pickle(self.out / "selectors.pkl", artifacts)
        self.finish("attribute", forward_max_error=attributed["replacement_forward_max_error"], rows=len(attributed["fit_uids"]))
        self.log("attribution_complete", forward_max_error=attributed["replacement_forward_max_error"])

    def check(self):
        from .checks import check_runner
        result = check_runner(self.get_runner(), self.rows["fit"][0], self.get_artifacts(), self.config["anchor_layer"])
        write_json(self.out / "hook_checks.json", result)
        self.finish("check", result=result)
        self.log("hook_checks_complete", **result)

    def cached_eval(self, name, rows, spec=None):
        path = self.out / "predictions" / f"{name}.json"
        cache_identity = dict(run=self.manifest["identity"], selector_sha256=sha256(self.out / "selectors.pkl") if spec else None, rotation_batch_size=self.config.get("answer_rotation_batch_size", 3))
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["uids"] != [r["uid"] for r in rows] or result["spec"] != spec or result.get("cache_identity") != cache_identity:
                raise ValueError(f"Cached evaluation identity mismatch: {name}")
            return result
        runner = self.get_runner()
        intervention = Intervention(spec, self.get_artifacts()) if spec else None
        self.log("evaluation_start", name=name, rows=len(rows))
        result = runner.evaluate(rows, intervention, lambda done, total, seconds: self.log("evaluation_progress", name=name, done=done, total=total, seconds=round(seconds, 1)))
        result.update(metrics=metrics(rows, result["predictions"]), uids=[r["uid"] for r in rows], spec=spec, cache_identity=cache_identity)
        write_json(path, result)
        self.log("evaluation_complete", name=name, seconds=round(result["seconds"], 1), ambiguous_accuracy=result["metrics"]["ambiguous_accuracy"], disambiguated_accuracy=result["metrics"]["disambiguated_accuracy"])
        return result

    def screen(self):
        if not (self.out / "hook_checks.json").exists():
            raise ValueError("Run hook checks before screening")
        panel = balanced_panel(self.rows["selection"], self.config["screen_rows_per_family_condition"], self.config["seed"])
        roots = sorted({r["template_group_id"] for r in self.neutral["selection"]})[:self.config["neutral_screen_roots"]]
        neutral = [r for r in self.neutral["selection"] if r["template_group_id"] in roots]
        write_json(self.out / "screen_panel.json", dict(bbq_uids=[r["uid"] for r in panel], neutral_uids=[r["uid"] for r in neutral]))
        base = self.cached_eval("screen_baseline", panel)
        base_n = self.cached_eval("screen_neutral_baseline", neutral)
        table = []
        for spec in make_grid(self.config):
            changed = self.cached_eval(f'screen_{spec["id"]}', panel, spec)
            changed_n = self.cached_eval(f'screen_neutral_{spec["id"]}', neutral, spec)
            ok, reasons = feasible(changed["metrics"], changed_n["metrics"], base["metrics"], base_n["metrics"], self.config["constraints"])
            table.append(dict(spec=spec, bbq=changed["metrics"], neutral=changed_n["metrics"], feasible=ok, violations=reasons, perturbation=changed["perturbation"], seconds=changed["seconds"] + changed_n["seconds"]))
            write_json(self.out / "screen_results.json", table)
        self.finish("screen", configurations=len(table), bbq_rows=len(panel), neutral_rows=len(neutral))

    def select(self):
        table = json.loads((self.out / "screen_results.json").read_text(encoding="utf-8"))
        # Select one candidate per primary method using screen only; full selection
        # verifies this candidate. Failed confirmation means no feasible winner.
        # Retain-only diagnostics cannot compete with remove-selected candidates.
        methods = ("dim", "cav", "sn_consistency", "neuron_contrast", "relp_nodes", "layer_span")
        candidates = {}
        for method in methods:
            options = [r for r in table if r["spec"]["method"] == method and r["feasible"] and r["spec"].get("operation") != "retain"]
            if options:
                options.sort(key=lambda r: (-r["bbq"]["macro_ambiguous_accuracy"], r["perturbation"]["relative_l2"], r["seconds"], r["spec"]["id"]))
                candidates[method] = options[0]["spec"]
        write_json(self.out / "screen_winners_frozen.json", candidates)
        base = self.cached_eval("selection_baseline", self.rows["selection"])
        base_n = self.cached_eval("selection_neutral_baseline", self.neutral["selection"])
        frozen = {}
        for method in methods:
            if method not in candidates:
                frozen[method] = dict(status="no_feasible_screen_candidate", spec=None)
                continue
            spec = candidates[method]
            changed = self.cached_eval(f"selection_{method}", self.rows["selection"], spec)
            changed_n = self.cached_eval(f"selection_neutral_{method}", self.neutral["selection"], spec)
            ok, reasons = feasible(changed["metrics"], changed_n["metrics"], base["metrics"], base_n["metrics"], self.config["constraints"])
            improvement = changed["metrics"]["macro_ambiguous_accuracy"] > base["metrics"]["macro_ambiguous_accuracy"] + 1e-12
            frozen[method] = dict(status="accepted" if ok and improvement else "no_confirmed_improving_candidate", spec=spec, feasible=ok, violations=reasons, improves_selection=improvement, selection_metrics=changed["metrics"], neutral_metrics=changed_n["metrics"])
        write_json(self.out / "validation_plan_frozen.json", dict(winners=frozen, policy="Evaluate all frozen screen candidates as exploratory diagnostics, including rejected ones; only accepted candidates count as selected interventions. Never reselect on validation.", controls=[r["spec"] for r in table if r["spec"]["axis"] == "ranking_control"]))
        self.finish("select", accepted=[m for m, s in frozen.items() if s["status"] == "accepted"])
        self.log("selection_frozen", winners={m: s["status"] for m, s in frozen.items()})

    def validate(self):
        plan_path = self.out / "validation_plan_frozen.json"
        if not plan_path.exists():
            raise ValueError("Freeze the selection plan first")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_hash = sha256(plan_path)
        selector_hash = sha256(self.out / "selectors.pkl")
        if "validate" in self.manifest["stages"] and self.manifest["stages"]["validate"]["plan_sha256"] != plan_hash:
            raise ValueError("Validation plan changed after evaluation")
        self.log("validation_start", plan_sha256=plan_hash)
        # Record the immutable plan hash before inspecting any validation predictions.
        lock = self.out / "validation_lock.json"
        if lock.exists() and json.loads(lock.read_text()) != dict(plan_sha256=plan_hash, selector_sha256=selector_hash):
            raise ValueError("Validation plan lock mismatch")
        write_json(lock, dict(plan_sha256=plan_hash, selector_sha256=selector_hash))
        base = self.cached_eval("validation_baseline", self.rows["validation"])
        self.cached_eval("validation_neutral_baseline", self.neutral["validation"])
        table = []
        conditions = [(m, s["spec"], s["status"]) for m, s in plan["winners"].items() if s["spec"] is not None]
        conditions += [(s["method"], s, "preregistered_control") for s in plan["controls"]]
        for method, spec, status in conditions:
            changed = self.cached_eval(f"validation_{method}", self.rows["validation"], spec)
            neutral = self.cached_eval(f"validation_neutral_{method}", self.neutral["validation"], spec)
            ci = paired_cluster_ci(self.rows["validation"], base["predictions"], changed["predictions"], self.config["bootstrap_replicates"], self.config["seed"])
            table.append(dict(method=method, selection_status=status, spec=spec, bbq=changed["metrics"], neutral=neutral["metrics"], paired_cluster_ci=ci, perturbation=changed["perturbation"], seconds=changed["seconds"]))
            write_json(self.out / "validation_results.json", table)
        self.finish("validate", plan_sha256=plan_hash, conditions=len(table))

    def report(self):
        from .report import write_report
        write_report(self.out)
        self.finish("report")
        self.log("experiment_complete", report=str(self.out / "report.md"))


def main():
    parser = argparse.ArgumentParser(description="Local BBQ mechanism pilot with fit-only discovery and frozen validation")
    parser.add_argument("stage", choices=("fit", "attribute", "check", "screen", "select", "validate", "report", "all"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment_pilot.toml")
    args = parser.parse_args()
    torch.manual_seed(20261011)
    np.random.seed(20261011)
    experiment = Experiment(args.config)
    stages = ("fit", "attribute", "check", "screen", "select", "validate", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if args.stage == "all" and experiment.manifest["stages"].get(stage, {}).get("completed"):
            experiment.log("resume_skip_stage", stage=stage)
            continue
        if stage in {"fit", "attribute", "screen", "select"} and (experiment.out / "validation_lock.json").exists():
            raise ValueError("Validation has started; discovery/selection are locked. Use a new run directory.")
        try:
            getattr(experiment, stage)()
        except Exception as error:
            experiment.manifest["status"] = f"{stage}_failed"
            experiment.manifest["failure"] = dict(stage=stage, type=type(error).__name__, message=str(error))
            write_json(experiment.out / "run_manifest.json", experiment.manifest)
            raise


if __name__ == "__main__":
    main()
