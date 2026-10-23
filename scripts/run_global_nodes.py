"""Fixed multi-layer node-subnetwork diagnostic; not a neuron-edge circuit.

Preregister before any primary validation results. The explicit universe is all
MLP coordinates in the seven fit-hook layers, at the last prompt position.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechanism.__main__ import Experiment, load_pickle
from mechanism.data import balanced_panel, sha256, write_json
from mechanism.evaluate import metrics, paired_cluster_ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preregister", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment_pilot.toml")
    args = parser.parse_args()
    exp = Experiment(args.config)
    plan_path = exp.out / "global_nodes_plan_frozen.json"
    if args.stage == "preregister":
        if (exp.out / "validation_lock.json").exists():
            raise ValueError("Too late to preregister: primary validation already started")
        layers = exp.config["layers"]
        specs = [dict(id=f"global_relp_256_{op}", method="relp_nodes_global", ranking_key="relp_global:last1", layers=layers, pool="last1", scope="last1", operation=op, strength=1., reference="mean", k=256, global_budget=True) for op in ("remove", "retain")]
        plan = dict(preregistered_before_primary_validation=True, fixed_no_hyperparameter_selection=True, universe=dict(layers=layers, coordinates_per_layer=6144, token="last_prompt", excluded_layers="other model layers run normally"), conditions=specs, source_relp_sha256=sha256(exp.out / "relp_scores.pkl"), note="Signed mean-reference RelP ranking pooled across the declared layers. Retain sets the explicit universe complement to fit role means. No edges discovered.")
        if plan_path.exists():
            raise FileExistsError("Plan already frozen")
        write_json(plan_path, plan)
        print(json.dumps(plan, indent=2))
        return
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha256(exp.out / "relp_scores.pkl") != plan["source_relp_sha256"]:
        raise ValueError("Attribution artifact changed after registration")
    attributed = load_pickle(exp.out / "relp_scores.pkl")
    artifacts = exp.get_artifacts()
    layers = plan["universe"]["layers"]
    scores = np.stack([attributed["scores"]["last1"][l] for l in layers])
    order = np.argsort(-scores.ravel(), kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    ranks = ranks.reshape(scores.shape)
    artifacts["global_ranks"] = {"relp_global:last1": {l: ranks[i] for i, l in enumerate(layers)}}
    write_json(exp.out / "global_nodes_selected.json", {str(l): np.flatnonzero(ranks[i] < 256).tolist() for i, l in enumerate(layers)})
    base = exp.cached_eval("validation_baseline", exp.rows["validation"])
    table = []
    # Fixed hypotheses only: no choosing signs/layers/k after inspecting outcomes.
    for spec in plan["conditions"]:
        changed = exp.cached_eval(f'validation_{spec["id"]}', exp.rows["validation"], spec)
        neutral = exp.cached_eval(f'validation_neutral_{spec["id"]}', exp.neutral["validation"], spec)
        ci = paired_cluster_ci(exp.rows["validation"], base["predictions"], changed["predictions"], exp.config["bootstrap_replicates"])
        table.append(dict(spec=spec, bbq=changed["metrics"], neutral=neutral["metrics"], paired_cluster_ci=ci, perturbation=changed["perturbation"], seconds=changed["seconds"]))
        write_json(exp.out / "global_nodes_results.json", dict(plan_sha256=sha256(plan_path), conditions=table, implementation_sha256=sha256(Path(__file__)), intervention_sha256=sha256(ROOT / "mechanism/interventions.py")))
    print("Global node-subnetwork diagnostics complete.")


if __name__ == "__main__":
    main()
