"""Numeric cells of the paper tables, with consistent decimal rounding."""

from decimal import Decimal, ROUND_HALF_UP

MODELS = ["qwen3", "minicpm5", "qwen25", "qwen3_8b", "qwen35_4b", "qwen35_9b"]
POLICIES = {
    "Base": "base",
    "FairSteer": "fs_joint_selected",
    "E-steer": "evidence_joint_selected",
    "EGUS": "egus_layer_selected",
}


def rounded(x, digits=2):
    return str(
        Decimal(str(round(float(x), 10))).quantize(
            Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP
        )
    )


def cells(summary):
    result = {
        k: {}
        for k in ["main", "conditions", "matched", "selected", "kbselected", "searchcontrasts"]
    }
    for method in ["Base", "KnowBias", "FairSteer", "E-steer", "EGUS"]:
        row = []
        for model in MODELS:
            spec = summary["models"][model]
            if method == "KnowBias":
                row.append(rounded(100 * spec["knowbias"]["metrics"]["accuracy"]))
            else:
                v = spec["aggregate"][POLICIES[method]]["accuracy"]
                row.append(rounded(100 * v["mean"]))
                if method != "Base":
                    row.append(rounded(100 * v["sd"]))
        result["main"][method] = row
    for model in MODELS:
        s = summary["models"][model]
        row = []
        for method in ["Base", "KnowBias", "FairSteer", "EGUS"]:
            for metric in ["ambig", "clear"]:
                value = (
                    s["knowbias"]["metrics"][metric]
                    if method == "KnowBias"
                    else s["aggregate"][POLICIES[method]][metric]["mean"]
                )
                row.append(rounded(100 * value))
        result["conditions"][model] = row
        result["matched"][model] = [
            rounded(100 * s["aggregate"][p]["accuracy"]["mean"])
            for p in [
                "evidence_joint_selected",
                "egus_at_evidence_layer",
                "egus_layer_selected",
                "evidence_at_egus_layer",
            ]
        ]
        for method in ["FairSteer", "E-steer", "EGUS"]:
            row = []
            for seed in summary["seeds"]:
                c = s["selections"][str(seed)][POLICIES[method]]
                row.append(str(c["layer"]) if method == "EGUS" else f"{c['layer']}/{c['alpha']:g}")
            result["selected"][model + ":" + method] = row
        c = s["knowbias"]["selected"]
        result["kbselected"][model] = [
            "---" if not c["nodes"] else f"{c['consistency']:g}",
            f"{c['strength']:g}",
            str(len(c["nodes"])),
        ]
    for group in summary["comparison_groups"].values():
        for key, v in group.items():
            result["searchcontrasts"][key] = [
                f"{v['difference_pp']:.2f}",
                *[f"{x:.2f}" for x in v["ci95_pp"]],
                f"{v['holm_p']:.4f}",
            ]
    return result
