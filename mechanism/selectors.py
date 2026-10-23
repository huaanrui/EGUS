from __future__ import annotations

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import POOLS, example_weights


def discovery_classes(rows, predictions, min_examples=40, min_groups=8, cap=64, seed=20261011):
    by_uid = {p["uid"]: p for p in predictions}
    positive, negative = [], []
    for i, row in enumerate(rows):
        if row["context_condition"] != "ambig":
            continue
        choices = [r["prediction"] for r in by_uid[row["uid"]]["rotations"]]
        if all(c == row["stereotype_index"] for c in choices):
            positive.append(i)
        elif all(c == row["unknown_index"] for c in choices):
            negative.append(i)
    raw = {"positive": positive, "negative": negative}
    for label, indices in raw.items():
        groups = {rows[i]["template_group_id"] for i in indices}
        if len(indices) < min_examples or len(groups) < min_groups:
            raise ValueError(f"Discovery class {label}: {len(indices)} examples / {len(groups)} families, requires {min_examples}/{min_groups}. Expand fit within fit families; never borrow held-out rows.")
    # Round-robin families so capped discovery does not concentrate in a few templates.
    def cap_balanced(indices):
        bins = {}
        for i in indices:
            bins.setdefault(rows[i]["template_group_id"], []).append(i)
        for group in bins:
            bins[group].sort(key=lambda i: hashlib.sha256(f'{seed}:{rows[i]["uid"]}'.encode()).hexdigest())
        ordered = []
        for j in range(max(map(len, bins.values()))):
            for group in sorted(bins):
                if j < len(bins[group]):
                    ordered.append(bins[group][j])
        return ordered[:cap]
    selected = {key: cap_balanced(ids) for key, ids in raw.items()}
    audit = {key: dict(available=len(raw[key]), used=len(ids), families=len({rows[i]["template_group_id"] for i in ids}), uids=[rows[i]["uid"] for i in ids]) for key, ids in selected.items()}
    return selected, audit


def fit_selectors(runner, rows, collected, classes, layers):
    positive = np.asarray(classes["positive"])
    negative = np.asarray(classes["negative"])
    indices = np.r_[negative, positive]
    labels = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
    discovery_rows = [rows[i] for i in indices]
    weights = example_weights(discovery_rows)
    # Equal total mass for the two behavioral classes.
    for cls in (0, 1):
        weights[labels == cls] *= 0.5 / weights[labels == cls].sum()
    features = collected["features"]
    directions, rankings, diagnostic = {}, {}, {}
    for pool in POOLS:
        for method in ("dim", "cav"):
            directions[f"{method}:{pool}"] = {}
        for method in ("sn_consistency", "neuron_contrast", "activation_magnitude", "random_neurons"):
            rankings[f"{method}:{pool}"] = {}
        for layer in layers:
            allx = features["residual"][pool][layer]
            x = allx[indices]
            b = np.average(x[labels == 1], axis=0, weights=weights[labels == 1])
            u = np.average(x[labels == 0], axis=0, weights=weights[labels == 0])
            scaler = StandardScaler().fit(x, sample_weight=weights)
            clf = LogisticRegression(C=1, max_iter=1000, solver="liblinear", random_state=20261011)
            clf.fit(scaler.transform(x), labels, sample_weight=weights * len(indices))
            vectors = {"dim": b - u, "cav": clf.coef_[0] / scaler.scale_}
            diagnostic[f"cav:{pool}:{layer}"] = dict(fit_accuracy=float(clf.score(scaler.transform(x), labels, sample_weight=weights)), warning="In-sample diagnostic; does not establish causal specificity")
            for method, vector in vectors.items():
                norm = np.linalg.norm(vector)
                if norm < 1e-12:
                    raise ValueError("Zero direction")
                vector = (vector / norm).astype(np.float32)
                projection = allx @ vector
                mean = np.average(projection, weights=collected["weights"])
                sigma = float(np.sqrt(np.average((projection - mean) ** 2, weights=collected["weights"])))
                directions[f"{method}:{pool}"][layer] = dict(vector=vector, sigma=sigma)
            col_norm = runner.layers[layer].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
            absolute = features["mlp_abs"][pool][layer][indices] * col_norm
            signed = features["mlp"][pool][layer][indices]
            contrast = (np.average(signed[labels == 1], axis=0, weights=weights[labels == 1]) - np.average(signed[labels == 0], axis=0, weights=weights[labels == 0])) * col_norm
            top = np.argpartition(absolute, -256, axis=1)[:, -256:]
            hits = np.zeros_like(absolute)
            np.put_along_axis(hits, top, 1, axis=1)
            consistency = np.average(hits[labels == 1], axis=0, weights=weights[labels == 1]) - np.average(hits[labels == 0], axis=0, weights=weights[labels == 0])
            magnitude = np.average(absolute[labels == 1], axis=0, weights=weights[labels == 1])
            # Frequency is primary; magnitude resolves ties only.
            order = np.lexsort((-magnitude, -consistency))
            rankings[f"sn_consistency:{pool}"][layer] = order.astype(np.int64)
            rankings[f"neuron_contrast:{pool}"][layer] = np.argsort(-np.abs(contrast), kind="stable").astype(np.int64)
            rankings[f"activation_magnitude:{pool}"][layer] = np.argsort(-magnitude, kind="stable").astype(np.int64)
            rankings[f"random_neurons:{pool}"][layer] = np.random.default_rng(20261011 + layer).permutation(len(magnitude))
    directions["random_direction:last1"] = {}
    for layer in layers:
        vector = np.random.default_rng(20261011 + layer).normal(size=runner.model.config.hidden_size).astype(np.float32)
        vector /= np.linalg.norm(vector)
        projection = features["residual"]["last1"][layer] @ vector
        mean = np.average(projection, weights=collected["weights"])
        sigma = float(np.sqrt(np.average((projection - mean) ** 2, weights=collected["weights"])))
        directions["random_direction:last1"][layer] = dict(vector=vector, sigma=sigma)
    return dict(directions=directions, rankings=rankings, role_means=collected["role_means"], diagnostics=diagnostic, discovery_indices=indices.tolist(), discovery_weights=weights, class_labels=labels.tolist())


def attach_attribution(artifacts, attributed, prefix="relp_nodes"):
    for pool, layer_map in attributed["scores"].items():
        artifacts["rankings"][f"{prefix}:{pool}"] = {layer: np.argsort(-scores, kind="stable").astype(np.int64) for layer, scores in layer_map.items()}
