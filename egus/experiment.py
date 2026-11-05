"""The paper's layer/strength search, without historical cache fallbacks."""

from pathlib import Path
import shutil
import joblib
import numpy as np
from scipy.special import expit

from .config import ROOT
from .data import audit, training_positions
from .io import read, rows, write, sha, digest
from .policies import fit_logistic, select_threshold, label_for, discover_unknown
from .metrics import summarize_prediction, choose_candidate, mean_sd


def arrays(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **values)
    temporary.replace(path)


def prepare(config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    audit(config["data_dir"], config["seeds"])
    registry = read(ROOT / "configs/models.json")
    models = {}
    for spec in config["models"]:
        expected = next(m for m in registry["models"] if m["name"] == spec["name"])
        folder = Path(spec["path"])
        if not folder.exists():
            raise FileNotFoundError(
                f"{folder}: download the pinned checkpoint before preparing the run"
            )
        actual = {}
        for item in expected["files"]:
            path = folder / item["name"]
            if not path.exists():
                raise FileNotFoundError(path)
            value = sha(path)
            if item.get("sha256") and value != item["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {path}")
            actual[item["name"]] = value
        models[spec["name"]] = actual
    data = Path(config["data_dir"])
    filenames = ["training.json", "validation.jsonl", "test.jsonl"] + [
        f"train_{s}.jsonl" for s in config["seeds"]
    ]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "egus").rglob("*.py")}
    plan = {
        "config": config,
        "models": models,
        "source_hashes": sources,
        "data_hashes": {name: sha(data / name) for name in filenames},
    }
    if {"fairsteer", "knowbias"} & set(config["methods"]):
        from .baselines.author import fetch_sources

        fetch_sources(verify_only=True)
        plan["author_sources"] = read(ROOT / "configs/authors.json")
    path = output / "plan.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Run inputs changed. Select a new output directory.")
    else:
        write(path, plan)
        for relative in sources:
            target = output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
    return plan


class Experiment:
    def __init__(self, config, spec):
        from .model import ModelEngine

        self.c, self.spec = config, spec
        self.root, self.data = Path(config["output_dir"]), Path(config["data_dir"])
        self.plan = read(self.root / "plan.json")
        self.plan_hash = sha(self.root / "plan.json")
        self.training = read(self.data / "training.json")
        self.engine = ModelEngine(
            spec, config, self.root, self.plan_hash, self.plan["models"][spec["name"]]
        )
        self.out = self.engine.out
        self.author = None
        if "fairsteer" in config["methods"] and config["fairsteer_backend"] == "cuml":
            from .baselines.author import FairSteer

            self.author = FairSteer()
            self.engine.edit_callback = self.author.add

    def folder(self, seed):
        path = self.out / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def layers(self):
        return list(range(self.spec["layers"]))

    def fit(self, seed, train_states, train_scores, val_states, val_scores):
        folder = self.folder(seed)
        if (folder / "fit_complete.json").exists():
            return
        train = rows(self.data / f"train_{seed}.jsonl")
        validation = rows(self.data / "validation.jsonl")
        positions = training_positions(self.training[str(seed)])
        need_evidence = bool({"egus", "evidence_steering"} & set(self.c["methods"]))
        need_fs = "fairsteer" in self.c["methods"]
        if {"fairsteer", "evidence_steering"} & set(self.c["methods"]):
            lookup = {r["uid"]: r for r in train}
            selected = [lookup[u] for u in self.training[str(seed)]["direction_uids"]]
            assert len(selected) == 110 and all(
                r["label"] == r["unknown_index"] != r["target_loc"] for r in selected
            )
            plus, _ = self.engine.capture(
                f"contrast_{seed}_positive", selected, [r["label"] for r in selected]
            )
            minus, _ = self.engine.capture(
                f"contrast_{seed}_negative", selected, [r["target_loc"] for r in selected]
            )
            if self.author:
                vector = self.author.mean(
                    self.spec["layers"],
                    np.concatenate([plus, minus], axis=1).transpose(1, 0, 2),
                    np.r_[np.ones(len(selected)), np.zeros(len(selected))],
                )
            else:
                vector = (np.asarray(plus, float) - np.asarray(minus, float)).mean(1)
            np.save(folder / "directions.npy", vector)
        original_y = np.array([label_for(r, int(p)) for r, p in zip(train, train_scores.argmax(1))])
        validation_y = np.array(
            [label_for(r, int(p)) for r, p in zip(validation, val_scores.argmax(1))]
        )
        evidence_y = np.array([r["context_condition"] == "ambig" for r in train], int)
        vy = np.array([r["context_condition"] == "ambig" for r in validation], int)
        for layer in self.layers():
            path, meta = folder / f"layer_{layer:02d}.npz", folder / f"layer_{layer:02d}.json"
            native_probe = folder / f"layer_{layer:02d}_cuml.pkl"
            if path.exists() and meta.exists() and (not self.author or native_probe.exists()):
                continue
            x, v = (
                train_states[layer, positions].astype(np.float32),
                val_states[layer].astype(np.float32),
            )
            fitted, details = {}, {}
            if need_evidence:
                coef, bias, info = fit_logistic(x, evidence_y[positions], self.c)
                threshold, _ = select_threshold(
                    expit(v.astype(float) @ coef + bias), vy, self.c["gate_fpr_limit"]
                )
                fitted.update(
                    evidence_coef=coef, evidence_intercept=bias, threshold=threshold["threshold"]
                )
                details["evidence"] = info
            if need_fs:
                if self.author:
                    classifier, info = self.author.fit(
                        seed, x, original_y[positions], v, validation_y
                    )
                    temporary = folder / f"layer_{layer:02d}_cuml.tmp"
                    joblib.dump(classifier, temporary)
                    temporary.replace(folder / f"layer_{layer:02d}_cuml.pkl")
                    details["fairsteer"] = info
            arrays(path, **fitted)
            write(meta, details)
            self.engine.log("fit_layer", seed=seed, layer=layer)
        write(folder / "fit_complete.json", {"methods": self.c["methods"]})

    def masks(self, seed, layer, features):
        folder = self.folder(seed)
        params = dict(np.load(folder / f"layer_{layer:02d}.npz"))
        result = {}
        x = np.asarray(features, float)
        if "evidence_coef" in params:
            result["evidence"] = (
                expit(x @ params["evidence_coef"] + params["evidence_intercept"])
                > params["threshold"]
            )
        if "fairsteer" in self.c["methods"]:
            if self.author:
                from .baselines.author import numpy

                classifier = joblib.load(folder / f"layer_{layer:02d}_cuml.pkl")
                result["fs"] = (
                    numpy(classifier.predict(np.asarray(features, np.float32))).reshape(-1) == 0
                )
        return result

    def counter(self, key, values, seed, layer, alpha, mask, base):
        if not alpha or not mask.any():
            return base.copy()
        folder = self.folder(seed)
        path = folder / "counter" / f"{key}_L{layer:02d}_a{alpha:g}.npz"
        identity = digest(
            [
                self.plan_hash,
                key,
                seed,
                layer,
                alpha,
                [r["uid"] for r in values],
                mask.tolist(),
                sha(folder / "directions.npy"),
            ]
        )
        if path.exists():
            with np.load(path) as data:
                assert str(data["identity"]) == identity
                return data["scores"]
        scores = base.copy()
        active, vector = np.flatnonzero(mask), np.load(folder / "directions.npy")[layer]
        for start in range(0, len(active), self.c["batch_size"]):
            idx = active[start : start + self.c["batch_size"]]
            scores[idx] = self.engine.forward([values[i] for i in idx], layer, alpha, vector)[0]
        arrays(path, scores=scores, mask=mask, identity=identity)
        return scores

    def select(self, candidates):
        kinds = {v["kind"] for v in candidates}
        selected = {}
        for kind, name in [
            ("fs", "fs_joint_selected"),
            ("egus", "egus_layer_selected"),
            ("evidence_steer", "evidence_joint_selected"),
        ]:
            if kind in kinds:
                selected[name] = choose_candidate([v for v in candidates if v["kind"] == kind])
        if {"egus", "evidence_steer"} <= kinds:
            egus, steering = selected["egus_layer_selected"], selected["evidence_joint_selected"]
            selected["evidence_at_egus_layer"] = choose_candidate(
                [
                    v
                    for v in candidates
                    if v["kind"] == "evidence_steer" and v["layer"] == egus["layer"]
                ]
            )
            selected["egus_at_evidence_layer"] = next(
                v for v in candidates if v["kind"] == "egus" and v["layer"] == steering["layer"]
            )
        return selected

    def fit_select(self):
        self.engine.load()
        methods = set(self.c["methods"]) - {"knowbias"}
        if methods:
            validation = rows(self.data / "validation.jsonl")
            val_states, base = self.engine.capture("validation", validation)
            for seed in self.c["seeds"]:
                folder = self.folder(seed)
                if (folder / "selection.json").exists():
                    continue
                train = rows(self.data / f"train_{seed}.jsonl")
                states, scores = self.engine.capture(f"train_{seed}", train)
                self.fit(seed, states, scores, val_states, base)
                candidates = []
                for layer in self.layers():
                    path = folder / f"validation_L{layer:02d}.json"
                    if path.exists():
                        candidates.extend(read(path))
                        continue
                    masks = self.masks(seed, layer, val_states[layer])
                    records = []
                    if "egus" in methods:
                        mask = masks["evidence"]
                        pred = base.argmax(1)
                        unknown = np.array(
                            [
                                discover_unknown(r, self.training[str(seed)]["vocab"])
                                for r in validation
                            ]
                        )
                        use = mask & (unknown >= 0)
                        pred[use] = unknown[use]
                        records.append(
                            {
                                "kind": "egus",
                                "layer": layer,
                                "alpha": None,
                                "metrics": summarize_prediction(validation, pred, mask),
                            }
                        )
                    union = np.zeros(len(validation), bool)
                    if "fairsteer" in methods:
                        union |= masks["fs"]
                    if "evidence_steering" in methods:
                        union |= masks["evidence"]
                    strengths = self.c["alphas"]
                    for alpha in strengths if union.any() else [0.0]:
                        cf = self.counter("validation", validation, seed, layer, alpha, union, base)
                        for method, kind, key in [
                            ("fairsteer", "fs", "fs"),
                            ("evidence_steering", "evidence_steer", "evidence"),
                        ]:
                            if method not in methods:
                                continue
                            pred, mask = base.argmax(1), masks[key]
                            pred[mask] = cf[mask].argmax(1)
                            records.append(
                                {
                                    "kind": kind,
                                    "layer": layer,
                                    "alpha": alpha,
                                    "metrics": summarize_prediction(validation, pred, mask),
                                }
                            )
                    # Include every strength even when the detector never activates.
                    if not union.any():
                        for method, kind, key in [
                            ("fairsteer", "fs", "fs"),
                            ("evidence_steering", "evidence_steer", "evidence"),
                        ]:
                            if method not in methods:
                                continue
                            records = [r for r in records if r["kind"] != kind]
                            for alpha in strengths:
                                records.append(
                                    {
                                        "kind": kind,
                                        "layer": layer,
                                        "alpha": alpha,
                                        "metrics": summarize_prediction(
                                            validation, base.argmax(1), masks[key]
                                        ),
                                    }
                                )
                    write(path, records)
                    candidates.extend(records)
                    self.engine.log("search_layer", seed=seed, layer=layer)
                write(folder / "candidates.json", candidates)
                parameters = {
                    p.name: sha(p)
                    for p in folder.iterdir()
                    if p.name.startswith("layer_") or p.name == "directions.npy"
                }
                write(
                    folder / "selection.json",
                    {
                        "selected": self.select(candidates),
                        "parameters": parameters,
                        "candidates_sha256": sha(folder / "candidates.json"),
                    },
                )
        if "knowbias" in self.c["methods"]:
            from .baselines.knowbias import KnowBias

            KnowBias(self.engine, self.c).select()
        lock = {
            "seeds": {str(s): sha(self.folder(s) / "selection.json") for s in self.c["seeds"]}
            if methods
            else {}
        }
        if "knowbias" in self.c["methods"]:
            lock["knowbias"] = sha(self.out / "knowbias_selection.json")
        write(self.out / "selection_lock.json", lock)

    def evaluate(self):
        lock = read(self.out / "selection_lock.json")
        for seed, expected in lock["seeds"].items():
            assert sha(self.folder(seed) / "selection.json") == expected
        self.engine.load()
        if lock["seeds"]:
            test = rows(self.data / "test.jsonl")
            states, scores = self.engine.capture("test", test)
            for seed in self.c["seeds"]:
                folder = self.folder(seed)
                if (folder / "test_metrics.json").exists():
                    continue
                chosen = read(folder / "selection.json")
                assert sha(folder / "candidates.json") == chosen["candidates_sha256"]
                assert all(sha(folder / p) == value for p, value in chosen["parameters"].items())
                variants = dict(chosen["selected"])
                predictions, triggers = (
                    {"base": scores.argmax(1)},
                    {"base": np.zeros(len(test), bool)},
                )
                union = {}
                for name, v in variants.items():
                    masks = self.masks(seed, v["layer"], states[v["layer"]])
                    triggers[name] = masks["fs" if v["kind"] == "fs" else "evidence"]
                    if v["kind"] != "egus":
                        key = v["layer"], v["alpha"]
                        union[key] = union.get(key, np.zeros(len(test), bool)) | triggers[name]
                counters = {
                    key: self.counter("test", test, seed, key[0], key[1], mask, scores)
                    for key, mask in union.items()
                }
                unknown = np.array(
                    [discover_unknown(r, self.training[str(seed)]["vocab"]) for r in test]
                )
                for name, v in variants.items():
                    pred, mask = scores.argmax(1), triggers[name]
                    if v["kind"] == "egus":
                        use = mask & (unknown >= 0)
                        pred[use] = unknown[use]
                    else:
                        pred[mask] = counters[(v["layer"], v["alpha"])][mask].argmax(1)
                    predictions[name] = pred
                saved = {"uids": np.array([r["uid"] for r in test])}
                for name, pred in predictions.items():
                    saved["pred_" + name], saved["trigger_" + name] = (
                        pred.astype(np.int16),
                        triggers[name],
                    )
                arrays(folder / "predictions.npz", **saved)
                write(
                    folder / "test_metrics.json",
                    {k: summarize_prediction(test, v, triggers[k]) for k, v in predictions.items()},
                )
        if "knowbias" in self.c["methods"]:
            assert sha(self.out / "knowbias_selection.json") == lock["knowbias"]
            from .baselines.knowbias import KnowBias

            KnowBias(self.engine, self.c).evaluate()


def summarize(config):
    results = {}
    for spec in config["models"]:
        folder = Path(config["output_dir"]) / spec["name"]
        records = [
            read(folder / f"seed_{s}/test_metrics.json")
            for s in config["seeds"]
            if (folder / f"seed_{s}/test_metrics.json").exists()
        ]
        result = {"completed_seeds": len(records)}
        if records:
            result["aggregate"] = {
                name: {
                    metric: mean_sd([r[name][metric] for r in records])
                    for metric in ["accuracy", "ambig", "clear"]
                }
                for name in records[0]
            }
        if (folder / "knowbias_metrics.json").exists():
            result["knowbias"] = {
                "independent_fits": 1,
                "metrics": read(folder / "knowbias_metrics.json"),
            }
        results[spec["name"]] = result
    write(Path(config["output_dir"]) / "summary.json", results)
    return results


def run(config, stage="all"):
    prepare(config)
    for phase in ["fit", "evaluate"] if stage == "all" else [stage]:
        if phase == "evaluate":
            lock = read(Path(config["output_dir"]) / "all_models_selected.json")
            assert set(lock) == {m["name"] for m in config["models"]}
            assert all(
                sha(Path(config["output_dir"]) / name / "selection_lock.json") == value
                for name, value in lock.items()
            )
        for spec in config["models"]:
            experiment = Experiment(config, spec)
            try:
                if phase == "fit":
                    experiment.fit_select()
                elif phase == "evaluate":
                    experiment.evaluate()
                else:
                    raise ValueError(phase)
            finally:
                experiment.engine.close()
        if phase == "fit":
            write(
                Path(config["output_dir"]) / "all_models_selected.json",
                {
                    m["name"]: sha(Path(config["output_dir"]) / m["name"] / "selection_lock.json")
                    for m in config["models"]
                },
            )
    return summarize(config)
