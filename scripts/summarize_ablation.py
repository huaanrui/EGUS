"""Report matched one-factor changes without choosing new validation conditions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=Path("results/pilot_20261011_v4"))
    args = parser.parse_args()
    table = json.loads((args.run / "screen_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8"))
    anchor = manifest["config"]["anchor_layer"]
    references = {}
    for r in table:
        s = r["spec"]
        if s["layers"] == [anchor] and s["pool"] == s["scope"] == "last1" and s["strength"] == 1 and s["operation"] in {"remove", "project"} and s.get("k", 64) == 64:
            references[s["method"]] = r
    rows = []
    ignored = {"id", "axis", "ranking_key", "direction_key"}
    metric_names = ("ambiguous_accuracy", "stereotype_rate", "counterstereotype_rate", "disambiguated_accuracy")
    for r in table:
        s = r["spec"]
        if s["method"] not in references or s["axis"] in {"ranking_control", "retain_diagnostic"}:
            continue
        ref = references[s["method"]]
        different = [k for k in set(s) | set(ref["spec"]) if k not in ignored and s.get(k) != ref["spec"].get(k)]
        if len(different) > 1:
            # Add/remove changes both operator and strength units; report as an
            # operator-family comparison, never as a clean strength ablation.
            if s["axis"] != "operator":
                raise ValueError(f"Multiple uncontrolled factors: {s['id']} {different}")
        item = dict(id=s["id"], reference_id=ref["spec"]["id"], method=s["method"], axis=s["axis"], changed_fields=",".join(different), pool=s["pool"], scope=s["scope"], k=s.get("k", ""), strength=s["strength"], layers=",".join(map(str, s["layers"])), feasible=r["feasible"])
        item.update({f"delta_{key}_pp": 100*(r["bbq"][key]-ref["bbq"][key]) for key in metric_names})
        item["delta_neutral_accuracy_pp"] = 100*(r["neutral"]["accuracy"]-ref["neutral"]["accuracy"])
        item["delta_ambiguous_margin_logits"] = r["bbq"]["ambiguous_margin"]-ref["bbq"]["ambiguous_margin"]
        rows.append(item)
    with (args.run / "ablation_factor_deltas.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    settings = [("extraction", "pool", ["last1", "last2", "last4", "last8", "user"], "Extraction (intervene at last1)"), ("intervention_scope", "scope", ["last1", "last2", "last4", "last8", "all"], "Intervention scope (fit at last1)"), ("component_budget", "k", [16, 64, 256], "Selected MLP coordinates"), ("strength", "strength", [.5, 1.], "Interpolation strength")]
    for ax, (axis, field, order, title) in zip(axes.ravel(), settings):
        for method, ref in references.items():
            if method in {"activation_magnitude", "random_neurons", "random_direction", "layer_span"}:
                continue
            points = [r for r in table if r["spec"]["method"] == method and r["spec"]["axis"] == axis]
            points = points + [ref]
            points = [r for r in points if r["spec"].get(field) in order]
            if len(points) < 2:
                continue
            # One reference may already be part of the budget grid.
            by_x = {r["spec"][field]: r for r in points}
            points = [by_x[x] for x in order if x in by_x]
            ax.plot([order.index(r["spec"][field]) for r in points], [100*(r["bbq"]["ambiguous_accuracy"]-ref["bbq"]["ambiguous_accuracy"]) for r in points], marker="o", label=method)
        ax.axhline(0, color="grey", linewidth=.7)
        ax.set_xticks(range(len(order)), [str(x) for x in order])
        ax.set(title=title, ylabel="Ambiguous accuracy change vs anchor (pp)")
        ax.legend(fontsize=7)
    fig.suptitle("Selection OFAT ablations (exploratory; no validation tuning)")
    fig.tight_layout()
    fig.savefig(args.run / "ablation_factors.png", dpi=160)
    plt.close(fig)
    print(f"Wrote {len(rows)} matched comparisons to {args.run}")


if __name__ == "__main__":
    main()
