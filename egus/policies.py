"""Evidence gates and response actions used in the paper."""

from collections import Counter
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def render(r):
    if r.get("utility"):
        return (
            r["question"]
            + "".join(f"\n{a}. {v}" for a, v in zip("ABCD", r["choices"]))
            + "\nAnswer:"
        )
    return (
        r["context"]
        + " "
        + r["question"]
        + "".join(f"\n{a}. {r['ans' + str(i)]}" for i, a in enumerate("ABC"))
        + "\nAnswer:"
    )


def discover_unknown(r, vocab, strict=False):
    if r.get("utility"):
        return -1
    found = [i for i in range(3) if r["ans" + str(i)].strip().casefold() in vocab]
    if len(found) != 1:
        if strict:
            raise ValueError(f"Cannot uniquely discover unknown from input: {r['uid']}")
        return -1
    return found[0]


def select_threshold(signal, insufficient, limit):
    s = np.asarray(signal, dtype=float)
    y = np.asarray(insufficient, dtype=bool)
    assert np.isfinite(s).all() and y.any() and (~y).any()
    choices = np.r_[np.nextafter(s.min(), -np.inf), np.unique(s), np.finfo(float).max]
    records = []
    for t in choices:
        use = s > t
        records.append(
            {
                "threshold": float(t),
                "recall": float(use[y].mean()),
                "fpr": float(use[~y].mean()),
                "trigger_fraction": float(use.mean()),
            }
        )
    valid = [r for r in records if r["fpr"] <= limit + 1e-12]
    return max(valid, key=lambda r: (r["recall"], -r["fpr"], r["threshold"])), records


def apply(base, counterfactual, unknown, mask, action):
    base = np.asarray(base)
    mask = np.asarray(mask, dtype=bool)
    unknown = np.asarray(unknown)
    scores = base.copy()
    if action == "steer":
        scores[mask] = counterfactual[mask]
    elif action not in ("direct", "base"):
        raise ValueError(action)
    pred = scores.argmax(1)
    if action == "direct":
        use = mask & (unknown >= 0)
        pred[use] = unknown[use]
    return pred, None if action == "direct" else scores


def fit_logistic(x, y, c):
    y = np.asarray(y)
    valid = y >= 0
    assert np.unique(y[valid]).tolist() == [0, 1], "Both classes required"
    with threadpool_limits(limits=c["cpu_threads"]), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(
            C=c["probe_C"], solver="lbfgs", max_iter=c["probe_max_iter"], tol=c["probe_tol"]
        ).fit(x[valid], y[valid])
    if any(isinstance(w.message, ConvergenceWarning) for w in caught):
        raise RuntimeError("Probe did not converge")
    coef = clf.coef_[0].astype(float)
    intercept = float(clf.intercept_[0])
    np.testing.assert_array_equal(
        (np.asarray(x[valid], float) @ coef + intercept > 0).astype(int), clf.predict(x[valid])
    )
    return (
        coef,
        intercept,
        {
            "rows": int(valid.sum()),
            "classes": dict(Counter(map(str, y[valid]))),
            "iterations": int(clf.n_iter_[0]),
            "train_accuracy": float(clf.score(x[valid], y[valid])),
        },
    )


def label_for(row, prediction):
    if prediction == row["label"]:
        return 1
    if row["target_loc"] is not None and prediction == row["target_loc"]:
        return 0
    return -1
