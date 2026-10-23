"""Read completed round-2 outputs; write only the literature audit summary."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "round2_20261012_v2"
SOURCES: dict[str, str] = {}


def read(name: str):
    data = (RUN / name).read_bytes()
    SOURCES[name] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def summarize(split: str):
    baseline = read(f"predictions/{split}_baseline.json")["metrics"]
    rows = read(f"{split}_results.json")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["spec"]["method"]].append(row)

    def gain(row):
        return 100 * (baseline["macro_stereotype"] - row["bbq"]["macro_stereotype"])

    def relative_l2(row):
        return (row.get("perturbation") or {}).get("relative_l2")

    best = []
    for row in sorted(rows, key=lambda r: -gain(r))[:5]:
        spec = row["spec"]
        best.append(dict(
            id=spec["id"], method=spec["method"], layers=spec["layers"],
            pool=spec["pool"], scope=spec["scope"], op=spec["operation"],
            strength=spec["strength"], gain_pp=gain(row),
            feasible=row["feasible"], violations=row["violations"],
            relative_l2=relative_l2(row),
            margin_delta=row["bbq"]["macro_stereotype_counter_margin"]
            - baseline["macro_stereotype_counter_margin"],
        ))
    by_method = {}
    for method, candidates in sorted(groups.items()):
        energies = [relative_l2(r) for r in candidates if relative_l2(r) is not None]
        by_method[method] = dict(
            n=len(candidates), best_gain_pp=max(map(gain, candidates)),
            max_relative_l2=max(energies) if energies else None,
        )
    return dict(
        baseline_macro_stereotype=baseline["macro_stereotype"],
        baseline_macro_counterstereotype=baseline["macro_counterstereotype"],
        baseline_macro_signed_gap=baseline["macro_stereotype"]-baseline["macro_counterstereotype"],
        baseline_micro_signed_gap=baseline["signed_ambiguous_bbq_bias"],
        count=len(rows),
        zero_macro_stereotype_change=sum(abs(gain(r)) < 1e-10 for r in rows),
        improving_macro_stereotype=sum(gain(r) > 1e-10 for r in rows),
        best_unconstrained=best, by_method=by_method,
    )


def main():
    result = {split: summarize(split) for split in ("screen", "selection")}
    localization = read("localization.json")
    diagnostics = read("fit_diagnostics.json")
    screen = read("screen_results.json")
    cav = [r for r in diagnostics["directions"] if r["method"] == "paired_cav"]
    result["search_audit"] = dict(
        candidate_layers=localization["candidate_layers"],
        final_block_earlier_position_configs=[r["spec"]["id"] for r in screen
            if r["spec"]["layers"] == [27] and r["spec"]["scope"] in ("identity", "question_end")],
        selected_fit_pairs=diagnostics["selected_pairs"],
        selected_fit_families=diagnostics["pair_families"],
        paired_cav_fit_sign_accuracy_range=[min(r["paired_sign_accuracy"] for r in cav),
                                          max(r["paired_sign_accuracy"] for r in cav)],
        paired_cav_last1_candidate_accuracy={str(r["layer"]): r["paired_sign_accuracy"] for r in cav
            if r["pool"] == "last1" and r["layer"] in localization["candidate_layers"]},
    )
    result["notes"] = [
        "Gain is absolute percentage points against the same panel's unchanged model, without feasibility filtering.",
        "Equal aggregate stereotype rate does not imply identical per-example predictions or logits.",
        "Missing perturbation measurements are null, never zero. Residual and MLP norms are not directly comparable.",
        "Final-block earlier-position edits cannot change the current next-token readout; this does not apply to arbitrary earlier-layer edits or future cached generation.",
        "Fit sign accuracy is in-sample association, not held-out accuracy or causal evidence.",
    ]
    result["source_sha256"] = SOURCES
    output = ROOT / "research" / "2026-10-12_literature_audit_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "screen_configs": result["screen"]["count"],
        "screen_improving": result["screen"]["improving_macro_stereotype"],
        "selection_best_gain_pp": result["selection"]["best_unconstrained"][0]["gain_pp"],
        "search_audit": result["search_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
