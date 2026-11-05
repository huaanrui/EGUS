"""Conditional accuracy and clustered statistical comparisons."""

from collections import defaultdict
import numpy as np


def summarize_prediction(values, pred, mask):
    y = np.array([r["label"] for r in values])
    ambiguous = np.array([r["context_condition"] == "ambig" for r in values])
    unknown = np.array([r["unknown_index"] for r in values])
    correct = np.asarray(pred) == y
    return {
        "accuracy": float(correct.mean()),
        "ambig": float(correct[ambiguous].mean()),
        "clear": float(correct[~ambiguous].mean()),
        "correct": int(correct.sum()),
        "clear_correct": int(correct[~ambiguous].sum()),
        "trigger": float(np.asarray(mask).mean()),
        "sufficient_trigger": float(np.asarray(mask)[~ambiguous].mean()),
        "clear_unknown": float((pred[~ambiguous] == unknown[~ambiguous]).mean()),
    }


def choose_candidate(candidates):
    """Use only saved development counts; deterministic ties avoid test access."""
    if not candidates:
        raise ValueError("No development candidates")
    return max(
        candidates,
        key=lambda c: (
            c["metrics"]["correct"],
            c["metrics"]["clear_correct"],
            -float(c["alpha"] or 0),
            -c["layer"],
        ),
    )


def comparison(values, a, b, reps=4000, seed=20261017, condition=None, cluster="family"):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.ndim == b.ndim == 2 and a.shape == b.shape and a.shape[1] == len(values)
    groups = defaultdict(list)
    for i, r in enumerate(values):
        if condition is None or r["context_condition"] == condition:
            groups[r[cluster]].append(i)
    assert groups
    labels = np.array([r["label"] for r in values])
    diff = (a == labels).astype(float) - (b == labels).astype(float)
    keys = sorted(groups)
    sums = np.column_stack([diff[:, groups[k]].sum(1) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    nseeds = a.shape[0]
    rng = np.random.default_rng(seed)
    si = rng.integers(nseeds, size=(reps, nseeds))
    fi = rng.integers(len(keys), size=(reps, len(keys)))
    sample = sums[si[:, :, None], fi[:, None, :]].sum((1, 2)) / (nseeds * counts[fi].sum(1))
    point = sums.sum() / (nseeds * counts.sum())
    means = sums.mean(0)
    null = (rng.choice([-1.0, 1.0], size=(max(8000, reps), len(keys))) * means).sum(
        1
    ) / counts.sum()
    p = (1 + np.sum(abs(null) >= abs(point) - 1e-12)) / (len(null) + 1)
    per_seed = sums.sum(1) / counts.sum() * 100
    return {
        "difference_pp": float(point * 100),
        "ci95_pp": (np.quantile(sample, [0.025, 0.975]) * 100).tolist(),
        "p": float(p),
        "cluster": cluster,
        "clusters": len(keys),
        "questions": int(counts.sum()),
        "training_seeds": nseeds,
        "per_seed_difference_pp": per_seed.tolist(),
        "seed_sd_pp": float(np.std(per_seed, ddof=1)) if nseeds > 1 else 0.0,
        "condition": condition,
    }


def holm(items):
    order = sorted(items, key=lambda k: items[k]["p"])
    previous = 0.0
    for i, key in enumerate(order):
        previous = max(previous, min(1.0, items[key]["p"] * (len(order) - i)))
        items[key]["holm_p"] = previous


def mean_sd(values):
    a = np.asarray(values, float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "values": a.tolist(),
    }
