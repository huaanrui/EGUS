"""Recompute the manuscript scores from released prediction IDs (CPU only)."""

import numpy as np

from .config import ROOT
from .io import read, sha, write
from .tables import cells
from .metrics import summarize_prediction, mean_sd, comparison, holm


def test_rows():
    d = read(ROOT / "reproducibility/test_labels.json")
    return [
        dict(
            uid=u,
            label=y,
            context_condition="ambig" if a else "disambig",
            family=f,
            target_loc=t,
            unknown_index=n,
            category=c,
        )
        for u, y, a, f, t, n, c in zip(
            d["uids"],
            d["labels"],
            d["ambiguous"],
            d["families"],
            d["targets"],
            d["unknown_indices"],
            d["categories"],
        )
    ]


def verify(statistics=False, output=None):
    manifest = read(ROOT / "reproducibility/paper_results.json")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Published artifact checksum mismatch: {relative}")
    values = test_rows()
    reference = read(ROOT / "benchmarks/main/summary.json")
    report = {
        "models": len(reference["models"]),
        "questions": len(values),
        "checked_predictions": 0,
        "knowbias_fits": 0,
    }
    scores = {}
    for model, spec in reference["models"].items():
        predictions, records = {}, []
        for seed in reference["seeds"]:
            with np.load(ROOT / f"benchmarks/main/predictions/{model}/seed_{seed}.npz") as data:
                np.testing.assert_array_equal(data["uids"], [r["uid"] for r in values])
                metrics = {}
                for name in spec["aggregate"]:
                    pred, mask = data["pred_" + name], data["trigger_" + name]
                    assert pred.shape == mask.shape == (len(values),)
                    assert np.isin(pred, [0, 1, 2]).all()
                    metrics[name] = summarize_prediction(values, pred, mask)
                    for metric, actual in metrics[name].items():
                        np.testing.assert_allclose(
                            actual, spec["per_seed"][str(seed)][name][metric], rtol=0, atol=1e-12
                        )
                    predictions.setdefault(name, []).append(pred)
                    report["checked_predictions"] += len(pred)
                records.append(metrics)
        for policy, metrics in spec["aggregate"].items():
            for metric, expected in metrics.items():
                actual = mean_sd([r[policy][metric] for r in records])
                for statistic in expected:
                    np.testing.assert_allclose(
                        actual[statistic], expected[statistic], rtol=0, atol=1e-12
                    )
        with np.load(ROOT / f"benchmarks/main/predictions/{model}/knowbias.npz") as data:
            np.testing.assert_array_equal(data["uids"], [r["uid"] for r in values])
            assert np.isin(data["pred"], [0, 1, 2]).all()
            actual = summarize_prediction(values, data["pred"], data["trigger"])
            for metric in actual:
                np.testing.assert_allclose(
                    actual[metric], spec["knowbias"]["metrics"][metric], atol=1e-12, rtol=0
                )
            assert spec["knowbias"]["independent_fits"] == 1
            report["knowbias_fits"] += 1
            report["checked_predictions"] += len(values)
        scores[model] = {k: np.stack(v) for k, v in predictions.items()}
    report["scores_match"] = True
    assert cells(reference) == read(ROOT / "benchmarks/main/tables.json")
    report["paper_tables_match"] = True
    if statistics:
        results = {}
        for group, contrasts in reference["comparison_groups"].items():
            recalculated = {}
            for key in contrasts:
                name, policy = key.split(":")
                recalculated[key] = comparison(
                    values, scores[name]["egus_layer_selected"], scores[name][policy]
                )
            holm(recalculated)
            for key, actual in recalculated.items():
                for metric in ["difference_pp", "ci95_pp", "p", "holm_p"]:
                    np.testing.assert_allclose(
                        actual[metric], contrasts[key][metric], atol=1e-10, rtol=0
                    )
            results[group] = recalculated
        report["statistics"] = results
    if output:
        write(output, report)
    return report
