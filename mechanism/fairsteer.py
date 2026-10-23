"""FairSteer-style reference pilot, with explicit deviations and frozen selection.

Directions use matched teacher-forced answers; probes use unanswered prompts.
All hooks and features share the raw block-output space, including the final block.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import tomllib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import balanced_panel, neutral_controls, read_rows, sha256, write_json
from .evaluate import feasible, metrics, per_row
from .fairsteer_data import FairPromptBuilder, PERMUTATIONS6, ROOT, prepare, rank


VARIANTS = ("native_gated", "native_always", "balanced_gated")


def add_direction(h, vector, alpha, coef=None, intercept=0.):
    """p(correct) < .5 triggers independently for every batch element."""
    value = h[:, -1, :].float()
    trigger = torch.ones(len(value), dtype=torch.bool, device=h.device) if coef is None else value @ coef + intercept < 0
    delta = alpha * vector[None, :] * trigger[:, None]
    if alpha == 0 or not bool(trigger.any()):
        return h, trigger
    changed = h.clone()
    changed[:, -1, :] = (value + delta).to(h.dtype)
    return changed, trigger


def detector_label(row, prediction):
    # Match the author's correct-first ordering, including stereotype-correct evidence rows.
    if prediction == row["correct_index"]:
        return 1
    if prediction == row["stereotype_index"]:
        return 0
    return None


def summarize(rows, predictions, native=False):
    if native:
        predictions = [dict(p, rotations=p["rotations"][:1]) for p in predictions]
    result = metrics(rows, predictions)
    by_uid = {p["uid"]: p for p in predictions}
    for cat in sorted(result["categories"]):
        subset = [r for r in rows if r["category"] == cat and r["context_condition"] == "ambig"]
        vals = collections.defaultdict(list)
        for row in subset:
            for p in by_uid[row["uid"]]["rotations"]:
                a, b, u = (row[k] for k in ("stereotype_index", "counterstereotype_index", "unknown_index"))
                if a is None:
                    continue
                scores = p["semantic_logprobs"]
                vals["stereotype"].append(p["prediction"] == a)
                vals["counterstereotype"].append(p["prediction"] == b)
                vals["stereotype_unknown_margin"].append(scores[a] - scores[u])
                vals["stereotype_counter_margin"].append(scores[a] - scores[b])
        result["categories"][cat].update({k: float(np.mean(v)) for k, v in vals.items()})
    for key in ("stereotype", "counterstereotype", "stereotype_unknown_margin", "stereotype_counter_margin"):
        values = [v[key] for v in result["categories"].values() if key in v]
        if values:
            result["macro_" + key] = float(np.mean(values))
    return result


def constraints(changed, neutral, base, base_neutral, limits, mmlu=None, base_mmlu=None):
    ok, reasons = feasible(changed, neutral, base, base_neutral, limits)
    for key in ("counterstereotype_rate", "macro_counterstereotype"):
        if changed[key] > base[key] + limits["counterstereotype_increase"] + 1e-12:
            reasons.append(key)
    if changed["macro_ambiguous_accuracy"] < base["macro_ambiguous_accuracy"] - limits["ambiguous_accuracy_drop"] - 1e-12:
        reasons.append("macro_ambiguous_accuracy")
    if mmlu is not None and mmlu < base_mmlu - limits["neutral_accuracy_drop"] - 1e-12:
        reasons.append("mmlu_accuracy")
    return not reasons, reasons


def calibrated(rows, predictions, offset):
    by_uid = {r["uid"]: r for r in rows}
    out = copy.deepcopy(predictions)
    for row in out:
        u = by_uid[row["uid"]]["unknown_index"]
        for rot in row["rotations"]:
            values = np.array(rot["semantic_logprobs"], dtype=np.float64)
            conditional_unknown = np.exp(values[u])
            factor = 1 + np.expm1(offset) * conditional_unknown
            old_mass = rot["candidate_mass"]
            rot["candidate_mass"] = float(old_mass * factor / (1 - old_mass + old_mass * factor))
            values[u] += offset
            maximum = values.max()
            values -= maximum + np.log(np.exp(values - maximum).sum())
            rot["semantic_logprobs"] = values.tolist()
            rot["prediction"] = int(values.argmax())
    return out


def bootstrap(rows, baseline, changed, replicates, seed):
    b, c = per_row(rows, baseline), per_row(rows, changed)
    rng = np.random.default_rng(seed)
    keys = ("stereotype", "counterstereotype", "accuracy")
    cats = sorted({r["category"] for r in rows})
    samples = np.zeros((replicates, 3))
    point = np.zeros(3)
    for cat in cats:
        groups = collections.defaultdict(list)
        for i, row in enumerate(rows):
            if row["category"] == cat and row["context_condition"] == "ambig":
                groups[row["template_group_id"]].append(i)
        sums = np.array([[sum(c[i][k] - b[i][k] for i in ids) for k in keys] for ids in groups.values()])
        counts = np.array([len(ids) for ids in groups.values()])
        picks = rng.integers(0, len(groups), (replicates, len(groups)))
        samples += sums[picks].sum(1) / counts[picks].sum(1)[:, None] / len(cats)
        point += sums.sum(0) / counts.sum() / len(cats)
    return {k: dict(delta=float(point[j]), low=float(np.quantile(samples[:, j], .025)), high=float(np.quantile(samples[:, j], .975))) for j, k in enumerate(("macro_stereotype", "macro_counterstereotype", "macro_ambiguous_accuracy"))}


class FairRunner:
    def __init__(self, model_path, fewshot, max_tokens, device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("This experiment requires CUDA")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.builder = FairPromptBuilder(self.tokenizer, fewshot, max_tokens)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32, attn_implementation="sdpa").to(device).eval()
        self.model.requires_grad_(False)
        self.layers = self.model.model.layers
        self.capture = False
        self.features = {}
        self.edit = None
        self.handles = [block.register_forward_hook(self.hook(i)) for i, block in enumerate(self.layers)]
        self.set_edit(None)

    def hook(self, layer):
        def apply(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.capture:
                self.features[layer] = h[:, -1].detach().clone()
            if self.edit is None or self.edit["layer"] != layer:
                return output
            changed, trigger = add_direction(h, self.vector, self.edit["alpha"], self.coef, self.intercept)
            actual = changed[:, -1].float() - h[:, -1].float()
            self.energy += actual.square().sum().double()
            self.original_energy += h[:, -1].float().square().sum().double()
            self.triggered += trigger.sum().double()
            self.calls += len(h)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        return apply

    def set_edit(self, spec, fitted=None):
        self.edit = spec
        self.energy = torch.zeros((), device=self.device, dtype=torch.float64)
        self.original_energy = torch.zeros_like(self.energy)
        self.triggered = torch.zeros_like(self.energy)
        self.calls = 0
        if spec is None:
            return
        layer = spec["layer"]
        mode = "balanced" if spec["variant"] == "balanced_gated" else "native"
        vector = fitted["directions"][mode][layer]
        if spec["variant"] == "random_gated":
            v = np.random.default_rng(spec["seed"]).normal(size=len(vector))
            vector = v * (np.linalg.norm(vector) / np.linalg.norm(v))
        self.vector = torch.as_tensor(vector, device=self.device, dtype=torch.float32)
        gated = spec["variant"] != "native_always"
        self.coef = torch.as_tensor(fitted["probes"][layer]["coef"], device=self.device, dtype=torch.float32) if gated else None
        self.intercept = fitted["probes"][layer]["intercept"] if gated else 0.

    def perturbation(self):
        return dict(relative_l2=float(torch.sqrt(self.energy / self.original_energy.clamp_min(1e-30))), trigger_fraction=float(self.triggered / max(self.calls, 1)), samples=self.calls)

    def forward(self, prompts, capture=False):
        if len({len(p["ids"]) for p in prompts}) != 1:
            raise ValueError("Permutation lengths differ; no silent padding or position changes")
        self.capture, self.features = capture, {}
        ids = torch.tensor([p["ids"] for p in prompts], device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        features = torch.stack([self.features[i] for i in range(len(self.layers))], 1).float().cpu().numpy() if capture else None
        self.capture, self.features = False, {}
        return logits, features

    def predict(self, row, stage="evaluation", shots=False, capture=False, permutations=6):
        prompts = [self.builder.build(row, p, stage=stage, shots=shots) for p in range(permutations)]
        logits, features = self.forward(prompts, capture)
        selected = torch.stack([v[p["label_ids"]] for v, p in zip(logits, prompts)])
        logs = selected.log_softmax(-1).cpu().numpy()
        predictions = selected.argmax(-1).cpu().numpy()
        total_lse = logits.logsumexp(-1)
        masses = (selected.logsumexp(-1) - total_lse).exp().cpu().numpy()
        values = []
        for p, scores, pred, mass in zip(prompts, logs, predictions, masses):
            semantic = np.empty(3)
            semantic[list(p["permutation"])] = scores
            values.append(dict(prediction=int(p["permutation"][int(pred)]), semantic_logprobs=semantic.tolist(), candidate_mass=float(mass)))
        return dict(uid=row["uid"], rotations=values), features


class FairSteerExperiment:
    def __init__(self, config, suffix=""):
        self.config = Path(config).resolve()
        self.c = tomllib.loads(self.config.read_text(encoding="utf-8"))
        self.data = ROOT / self.c["dataset_dir"]
        self.out = ROOT / (self.c["output_dir"] + suffix)
        self.out.mkdir(parents=True, exist_ok=True)
        self.data_manifest = prepare(self.config)
        source_names = ["mechanism/fairsteer.py", "mechanism/fairsteer_data.py", "mechanism/data.py", "mechanism/evaluate.py", "scripts/inspect_native_pairs.py", "scripts/prepare_bbq.py"]
        source_names += self.c.get("additional_sources", [])
        self.fingerprint = dict(config_sha256=sha256(self.config), data_manifest_sha256=sha256(self.data / "manifest.json"), sources={name: sha256(ROOT / name) for name in source_names})
        manifest = self.out / "manifest.json"
        self.state = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else dict(fingerprint=self.fingerprint, stages={})
        if self.state["fingerprint"] != self.fingerprint:
            raise ValueError("Frozen source/config/data changed; use a new version, never silently reuse fitted artifacts")
        for name in source_names:
            target = self.out / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != self.fingerprint["sources"][name]:
                raise ValueError("Source snapshot changed")
            if not target.exists():
                target.write_bytes((ROOT / name).read_bytes())
        write_json(manifest, self.state)
        self.rows = {split: read_rows(self.data / f"{split}.jsonl") for split in ("fit", "probe_validation", "direction_fit", "selection", "mmlu_fit", "mmlu_probe_validation", "mmlu_selection")}
        self.runner = None
        self.fitted = None
        self.neutral = neutral_controls()

    def log(self, event, **kw):
        item = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), event=event, **kw)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        with (self.out / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def finish(self, stage, **kw):
        self.state["stages"][stage] = dict(completed=True, **kw)
        write_json(self.out / "manifest.json", self.state)
        self.log(stage + "_complete", **kw)

    def save(self, name, value):
        path = self.out / name
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as f:
            pickle.dump(value, f, protocol=5)
        temp.replace(path)
        write_json(path.with_suffix(path.suffix + ".sha256.json"), dict(sha256=sha256(path)))

    def load(self, name):
        path = self.out / name
        if sha256(path) != json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())["sha256"]:
            raise ValueError("Local fitted artifact changed")
        with path.open("rb") as f:
            return pickle.load(f)

    def setup(self):
        if self.runner is None:
            torch.manual_seed(self.c["seed"])
            np.random.seed(self.c["seed"])
            self.log("loading_model", path=self.c["model_path"])
            self.runner = FairRunner(self.c["model_path"], (self.data / "fewshot.txt").read_text(encoding="utf-8"), self.c["max_input_tokens"])
            model_path = Path(self.c["model_path"])
            environment = dict(torch=torch.__version__, cuda=torch.version.cuda, transformers=__import__("transformers").__version__, gpu=torch.cuda.get_device_name(), dtype=str(next(self.runner.model.parameters()).dtype), model_files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in model_path.glob("*") if p.is_file() and p.suffix in (".safetensors", ".json")}, layer_index="zero-based raw block output; final norm excluded from extraction and intervention", batch="six equal-length answer permutations, no padding, no cache", author_commit="0545cdbaa71b3301030accf1b98130820e08026f")
            env_path = self.out / "environment.json"
            if env_path.exists() and json.loads(env_path.read_text(encoding="utf-8")) != environment:
                raise ValueError("Model files or runtime changed; cached fit/predictions cannot be reused")
            write_json(env_path, environment)
        if self.fitted is None and (self.out / "fitted.pkl").exists():
            self.fitted = self.load("fitted.pkl")

    def assert_development(self):
        if (self.out / "test_lock.json").exists():
            raise ValueError("Confirmation has opened; fit and selection are closed")

    def smoke(self):
        self.setup()
        rows = self.rows["selection"][:3]
        elapsed = time.perf_counter()
        for row in rows:
            prediction, feature = self.runner.predict(row, capture=True)
        duration = (time.perf_counter() - elapsed) / len(rows)
        p = self.runner.builder.build(self.rows["direction_fit"][0], stage="direction", shots=True, answer=self.rows["direction_fit"][0]["correct_index"])
        self.runner.forward([p] * 6, capture=True)
        write_json(self.out / "smoke.json", dict(seconds_per_six_permutation_row=duration, feature_shape=list(feature.shape), fewshot_tokens=len(p["ids"]), example=p, prediction=prediction, max_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
        self.log("smoke", seconds_per_row=duration, fewshot_tokens=len(p["ids"]), memory_gb=torch.cuda.max_memory_allocated()/1e9)

    def directions(self):
        self.assert_development()
        self.setup()
        if (self.out / "directions.pkl").exists():
            return self.load("directions.pkl")
        sums = {name: np.zeros((len(self.runner.layers), self.runner.model.config.hidden_size), np.float64) for name in ("native", "balanced")}
        audit = []
        for j, row in enumerate(self.rows["direction_fit"]):
            by_shot = []
            for shots in (False, True):
                features = []
                for answer in (row["correct_index"], row["stereotype_index"]):
                    prompts = [self.runner.builder.build(row, p, stage="direction", shots=shots, answer=answer) for p in range(6)]
                    _, feature = self.runner.forward(prompts, capture=True)
                    features.append(feature)
                by_shot.append(features[0] - features[1])
            delta = np.mean(by_shot, axis=0)
            sums["native"] += delta[0]
            sums["balanced"] += delta.mean(0)
            audit.append(dict(uid=row["uid"], correct=row["correct_index"], biased=row["stereotype_index"], group=row["template_group_id"]))
            if (j + 1) % 10 == 0:
                self.log("direction_pairs", completed=j+1, total=len(self.rows["direction_fit"]))
        directions = {k: (v / len(audit)).astype(np.float32) for k, v in sums.items()}
        self.save("directions.pkl", directions)
        write_json(self.out / "direction_audit.json", dict(pairs=audit, native_norms=np.linalg.norm(directions["native"], axis=1).tolist(), balanced_norms=np.linalg.norm(directions["balanced"], axis=1).tolist(), cosine_native_balanced=(np.sum(directions["native"]*directions["balanced"], 1)/np.maximum(np.linalg.norm(directions["native"],axis=1)*np.linalg.norm(directions["balanced"],axis=1),1e-30)).tolist(), formula="mean_{same-question pairs, zero/few-shot}[h(correct answer token)-h(stereotyped answer token)]; native retains original answer order, balanced averages all six permutations; neither is normalized"))
        return directions

    def collect_probes(self, split):
        path = self.out / f"probe_features_{split}.pkl"
        if path.exists():
            return self.load(path.name)
        values, labels, metadata, excluded = [], [], [], 0
        for j, row in enumerate(self.rows[split]):
            for shots in (False, True):
                pred, feature = self.runner.predict(row, stage="probe", shots=shots, capture=True)
                y = detector_label(row, pred["rotations"][0]["prediction"])
                if y is None:
                    excluded += 1
                else:
                    values.append(feature[0].copy())
                    labels.append(y)
                    metadata.append(dict(uid=row["uid"], group=row["template_group_id"], source="bbq", shots=shots, prediction=pred["rotations"][0]["prediction"]))
            if (j + 1) % 16 == 0:
                self.log("probe_collect", split=split, completed=j+1, total=len(self.rows[split]))
        for j, row in enumerate(self.rows["mmlu_" + split]):
            pred, feature = self.runner.predict(row, stage="probe", capture=True)
            values.append(feature[0].copy())
            labels.append(1)
            metadata.append(dict(uid=row["uid"], group=row["template_group_id"], source="mmlu", shots=False, prediction=pred["rotations"][0]["prediction"]))
            if (j+1) % 64 == 0:
                self.log("mmlu_probe_collect", split=split, completed=j+1)
        result = dict(x=np.asarray(values, np.float32), y=np.asarray(labels, np.int64), metadata=metadata, excluded_other_errors=excluded)
        self.save(path.name, result)
        return result

    def fit(self):
        self.assert_development()
        self.setup()
        directions = self.directions()
        if self.c.get("reuse_probe_from"):
            from .fairsteer_revision import reuse_probes
            reuse_probes(self, directions)
            return
        train, val = self.collect_probes("fit"), self.collect_probes("probe_validation")
        if set(train["y"]) != {0, 1} or min(collections.Counter(train["y"]).values()) < 20:
            raise ValueError("Insufficient detector training classes; do not manufacture labels")
        if {r["group"] for r in train["metadata"]} & {r["group"] for r in val["metadata"]}:
            raise ValueError("Probe family leakage")
        probes, diagnostics = [], []
        with threadpool_limits(limits=4):
            for layer in range(len(self.runner.layers)):
                clf = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, random_state=self.c["seed"]).fit(train["x"][:, layer], train["y"])
                probes.append(dict(coef=clf.coef_[0].astype(np.float32), intercept=float(clf.intercept_[0])))
                item = dict(layer=layer, train_accuracy=float(clf.score(train["x"][:, layer], train["y"])), iterations=int(clf.n_iter_[0]), converged=bool(clf.n_iter_[0] < 2000))
                for source in ("all", "bbq", "mmlu"):
                    idx = np.array([source == "all" or r["source"] == source for r in val["metadata"]])
                    yy = val["y"][idx]
                    pp = clf.predict(val["x"][idx, layer])
                    item[source] = dict(n=len(yy), accuracy=float(np.mean(yy==pp)), balanced_accuracy=float(balanced_accuracy_score(yy, pp)) if len(set(yy)) == 2 else None, biased_recall=float(np.mean(pp[yy==0]==0)) if (yy==0).any() else None, false_positive_rate=float(np.mean(pp[yy==1]==0)) if (yy==1).any() else None, auc=float(roc_auc_score(yy, clf.predict_proba(val["x"][idx, layer])[:,1])) if len(set(yy)) == 2 else None)
                diagnostics.append(item)
                self.log("probe_fit", layer=layer, bbq_balanced_accuracy=item["bbq"]["balanced_accuracy"], bbq_biased_recall=item["bbq"]["biased_recall"])
        self.fitted = dict(directions=directions, probes=probes)
        self.save("fitted.pkl", self.fitted)
        write_json(self.out / "probe_diagnostics.json", dict(train_labels=dict(collections.Counter(map(str, train["y"]))), validation_labels=dict(collections.Counter(map(str,val["y"]))), train_metadata=train["metadata"], validation_metadata=val["metadata"], excluded_train=train["excluded_other_errors"], excluded_validation=val["excluded_other_errors"], layers=diagnostics, solver="sklearn lbfgs C=1 L2, no scaling, no class balancing; CPU adaptation of cuML; group validation replaces author random split"))
        self.finish("fit", direction_pairs=len(self.rows["direction_fit"]), probe_train=len(train["y"]), probe_validation=len(val["y"]))

    def spec(self, variant, layer, alpha=1., seed=None):
        spec = dict(variant=variant, layer=layer, alpha=float(alpha))
        if seed is not None:
            spec["seed"] = seed
        spec["id"] = variant + f"_l{layer}_a{alpha:g}" + (f"_s{seed}" if seed is not None else "")
        return spec

    def evaluate(self, name, rows, spec=None, mmlu=False, fitted_path=None):
        path = self.out / "predictions" / f"{name}.json"
        artifact = Path(fitted_path) if fitted_path is not None else self.out / "fitted.pkl"
        identity = dict(rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), spec=spec, mmlu=mmlu, fitted_sha256=sha256(artifact) if spec else None)
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["identity"] != identity:
                raise ValueError(f"Cached prediction identity changed: {name}")
            return value
        self.setup()
        fitted = self.fitted
        if fitted_path is not None:
            if spec is None or sha256(artifact) != json.loads(artifact.with_suffix(artifact.suffix+".sha256.json").read_text())["sha256"]:
                raise ValueError("Invalid historical fitted artifact override")
            with artifact.open("rb") as f:
                fitted = pickle.load(f)
        self.runner.set_edit(spec, fitted)
        start, predictions = time.perf_counter(), []
        try:
            for j, row in enumerate(rows):
                pred, _ = self.runner.predict(row)
                predictions.append(pred)
                if (j+1) % 32 == 0:
                    self.log("evaluate", name=name, completed=j+1, total=len(rows))
            perturbation = self.runner.perturbation()
        finally:
            self.runner.set_edit(None)
        result = dict(identity=identity, predictions=predictions, seconds=time.perf_counter()-start, perturbation=perturbation)
        if mmlu:
            by_uid = {r["uid"]: r for r in rows}
            result["accuracy"] = float(np.mean([p["prediction"] == by_uid[r["uid"]]["answer"] for r in predictions for p in r["rotations"]]))
        else:
            result["metrics"] = summarize(rows, predictions)
            result["native_metrics"] = summarize(rows, predictions, native=True)
        write_json(path, result)
        self.log("evaluation_saved", name=name, seconds=result["seconds"], perturbation=perturbation)
        return result

    def checks(self):
        self.setup()
        row = self.rows["selection"][0]
        prompts = [self.runner.builder.build(row, p) for p in range(6)]
        baseline, raw = self.runner.forward(prompts, capture=True)
        self.runner.set_edit(self.spec("native_gated", 13, 0.), self.fitted)
        zero, _ = self.runner.forward(prompts)
        self.runner.set_edit(None)
        restored, _ = self.runner.forward(prompts)
        if not torch.equal(baseline, zero) or not torch.equal(baseline, restored):
            raise AssertionError("Zero-strength/restoration changed logits")
        self.runner.set_edit(self.spec("native_always", 13, 1.), self.fitted)
        changed, _ = self.runner.forward(prompts)
        energy = self.runner.perturbation()
        self.runner.set_edit(None)
        if energy["relative_l2"] <= 0 or torch.equal(changed, baseline):
            raise AssertionError("Nonzero raw direction had no effect")
        # HF hidden_states[-1] is post-final-norm; do not inject that vector into a raw block.
        self.runner.capture = True
        with torch.inference_mode():
            output = self.runner.model(input_ids=torch.tensor([p["ids"] for p in prompts],device=self.runner.device),use_cache=False,logits_to_keep=1,output_hidden_states=True)
        mismatches = []
        for layer in range(len(self.runner.layers)):
            observed = self.runner.features[layer]
            if not torch.equal(observed, output.hidden_states[layer+1][:,-1]):
                mismatches.append(layer)
        self.runner.capture, self.runner.features = False, {}
        if mismatches != [len(self.runner.layers)-1]:
            raise AssertionError(f"Unexpected hidden-state/block mapping: {mismatches}")
        # Verify all prompt/answer sites and all six permutations before any search.
        lengths = []
        for name in ("fit", "probe_validation", "selection", "direction_fit", "mmlu_fit", "mmlu_probe_validation", "mmlu_selection"):
            for r in self.rows[name]:
                for shots in ((False, True) if name in ("fit", "probe_validation", "direction_fit") else (False,)):
                    answer = r["correct_index"] if name == "direction_fit" else None
                    ps = [self.runner.builder.build(r,p,stage="direction" if answer is not None else "probe" if name in ("fit","probe_validation") else "evaluation",shots=shots,answer=answer) for p in range(6)]
                    if len({len(p["ids"]) for p in ps}) != 1:
                        raise ValueError("Permutation token counts differ")
                    lengths.append(len(ps[0]["ids"]))
        write_json(self.out / "hook_checks.json", dict(zero_exact=True, restore_exact=True, positive_edit=energy, logits_changed=True, hidden_states_vs_raw_mismatches=mismatches, min_tokens=min(lengths),max_tokens=max(lengths), prompting="Raw text as source code; exact context-specific space+letter next token; no chat wrapper; explicit answer vs unanswered sites", candidate_labels=prompts[0]["label_ids"]))
        self.finish("checks", max_tokens=max(lengths))

    def screen(self):
        self.assert_development()
        self.setup()
        panel = balanced_panel(self.rows["selection"], self.c["screen_rows_per_family_condition"], self.c["seed"])
        grid = [self.spec(variant, layer) for variant in VARIANTS for layer in range(len(self.runner.layers))]
        plan = dict(grid=grid, panel_uids=[r["uid"] for r in panel], layer_selection="Each variant scans every raw block with its own add operator; no identity donor or layer-update preselection", primary="native_gated, alpha=1, maximize native overall BBQ accuracy", robustness="all six permutations stored for every condition; compare to same panel baseline", constraints=self.c["constraints"])
        path = self.out / "search_plan_frozen.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("Frozen grid changed")
        write_json(path, plan)
        self.evaluate("screen_baseline", panel)
        results = []
        for spec in grid:
            value = self.evaluate("screen_"+spec["id"], panel, spec)
            results.append(dict(spec=spec, metrics=value["metrics"], native_metrics=value["native_metrics"], perturbation=value["perturbation"]))
            write_json(self.out / "screen_results.json", results)
        self.finish("screen", configurations=len(grid), rows=len(panel))

    def select(self):
        self.assert_development()
        self.setup()
        screen = json.loads((self.out / "screen_results.json").read_text())
        selected = []
        for variant in VARIANTS:
            options = sorted([r for r in screen if r["spec"]["variant"]==variant], key=lambda r:(-r["native_metrics"]["accuracy"], abs(r["native_metrics"]["signed_ambiguous_bbq_bias"]), r["spec"]["layer"]))
            selected.extend(r["spec"] for r in options[:self.c["finalists_per_variant"]])
        primary_layer = next(s["layer"] for s in selected if s["variant"]=="native_gated")
        selected.extend(self.spec("native_gated",primary_layer,a) for a in self.c["strengths"] if a!=1.)
        rows = self.rows["selection"]
        base = self.evaluate("selection_baseline",rows)
        base_n = self.evaluate("selection_neutral_baseline",self.neutral["selection"])
        base_m = self.evaluate("selection_mmlu_baseline",self.rows["mmlu_selection"],mmlu=True)
        result = []
        for spec in selected:
            value = self.evaluate("selection_"+spec["id"],rows,spec)
            neutral = self.evaluate("selection_neutral_"+spec["id"],self.neutral["selection"],spec)
            mmlu = self.evaluate("selection_mmlu_"+spec["id"],self.rows["mmlu_selection"],spec,mmlu=True)
            ok,reasons = constraints(value["metrics"],neutral["metrics"],base["metrics"],base_n["metrics"],self.c["constraints"],mmlu["accuracy"],base_m["accuracy"])
            gain = base["metrics"]["macro_stereotype"]-value["metrics"]["macro_stereotype"]
            result.append(dict(spec=spec,metrics=value["metrics"],native_metrics=value["native_metrics"],neutral=neutral["metrics"],mmlu_accuracy=mmlu["accuracy"],feasible=ok,violations=reasons,macro_stereotype_reduction=gain,accepted=bool(ok and gain>=self.c["minimum_selection_stereotype_reduction"]),perturbation=value["perturbation"]))
            write_json(self.out / "selection_results.json",result)
        # Primary reference selects a layer at alpha=1 by accuracy, as in the paper.
        candidates = [r for r in result if r["spec"]["variant"]=="native_gated" and r["spec"]["alpha"]==1.]
        primary = min(candidates,key=lambda r:(-r["native_metrics"]["accuracy"],abs(r["native_metrics"]["signed_ambiguous_bbq_bias"]),r["spec"]["layer"]))
        feasible_results = [r for r in result if r["feasible"]]
        guarded = min(feasible_results,key=lambda r:(-r["macro_stereotype_reduction"],r["metrics"]["macro_stereotype_unknown_margin"],r["spec"]["id"])) if feasible_results else None
        fixed = [primary["spec"]]
        for variant in ("native_always","balanced_gated"):
            best = min([r for r in result if r["spec"]["variant"]==variant],key=lambda r:(-r["native_metrics"]["accuracy"],r["spec"]["layer"]))
            fixed.append(best["spec"])
        if guarded and guarded["spec"]["id"] not in {s["id"] for s in fixed}:
            fixed.append(guarded["spec"])
        fixed.extend(self.spec("random_gated",primary["spec"]["layer"],seed=seed) for seed in self.c["random_seeds"])
        calibrations = []
        for offset in (0.,.25,.5,1.,2.,4.):
            pred = calibrated(rows,base["predictions"],offset)
            neutral_pred = calibrated(self.neutral["selection"],base_n["predictions"],offset)
            mm,nn = summarize(rows,pred),summarize(self.neutral["selection"],neutral_pred)
            ok,reasons=constraints(mm,nn,base["metrics"],base_n["metrics"],self.c["constraints"])
            calibrations.append(dict(offset=offset,metrics=mm,feasible=ok,violations=reasons))
        calibration = min([r for r in calibrations if r["feasible"]],key=lambda r:(r["metrics"]["macro_stereotype"],r["offset"]))
        write_json(self.out / "calibration_selection.json",calibrations)
        plan = dict(primary=primary["spec"],primary_selection=primary,guarded=guarded,configs=fixed,calibration_offset=calibration["offset"],selection_results_sha256=sha256(self.out / "selection_results.json"),fitted_sha256=sha256(self.out / "fitted.pkl"),test_data_sha256=self.data_manifest["files"]["test.jsonl"]["sha256"],interpretation="Primary reference and guarded criterion are distinct and frozen. Random directions match raw native DiM length and gate. Test contains fresh families in categories present in fit; all six permutations. No re-selection on test.")
        if self.data_manifest.get("test_role")=="opened_family_paired_diagnostic":
            plan["interpretation"]="Opened-family paired diagnostic using the same 320 questions as the previous run. Not fresh confirmation. All choices use unchanged selection only; separate unseen-category confirmation follows a predeclared functional policy."
        write_json(self.out / "test_plan_frozen.json",plan)
        self.finish("select", primary=primary["spec"], guarded_accepted=bool(guarded and guarded["accepted"]),calibration_offset=calibration["offset"])

    def test(self):
        self.setup()
        plan_path = self.out / "test_plan_frozen.json"
        plan = json.loads(plan_path.read_text())
        if sha256(self.out / "fitted.pkl") != plan["fitted_sha256"] or sha256(self.out / "selection_results.json") != plan["selection_results_sha256"]:
            raise ValueError("Frozen plan inputs changed")
        lock = dict(plan_sha256=sha256(plan_path),test_sha256=sha256(self.data/"test.jsonl"))
        if lock["test_sha256"] != plan["test_data_sha256"]:
            raise ValueError("Confirmation data changed")
        lock_path = self.out / "test_lock.json"
        if lock_path.exists() and json.loads(lock_path.read_text())!=lock:
            raise ValueError("Confirmation lock changed")
        write_json(lock_path,lock)
        # This is the first point at which this experiment reads confirmation rows.
        rows,mmlu_rows = read_rows(self.data/"test.jsonl"),read_rows(self.data/"mmlu_test.jsonl")
        base=self.evaluate("test_baseline",rows)
        bn=self.evaluate("test_neutral_baseline",self.neutral["validation"])
        bm=self.evaluate("test_mmlu_baseline",mmlu_rows,mmlu=True)
        results=[]
        for spec in plan["configs"]:
            value=self.evaluate("test_"+spec["id"],rows,spec)
            neutral=self.evaluate("test_neutral_"+spec["id"],self.neutral["validation"],spec)
            mmlu=self.evaluate("test_mmlu_"+spec["id"],mmlu_rows,spec,mmlu=True)
            ok,reasons=constraints(value["metrics"],neutral["metrics"],base["metrics"],bn["metrics"],self.c["constraints"],mmlu["accuracy"],bm["accuracy"])
            results.append(dict(spec=spec,metrics=value["metrics"],native_metrics=value["native_metrics"],perturbation=value["perturbation"],neutral=neutral["metrics"],mmlu_accuracy=mmlu["accuracy"],feasible=ok,violations=reasons,confidence=bootstrap(rows,base["predictions"],value["predictions"],self.c["bootstrap_replicates"],self.c["seed"])))
            write_json(self.out/"test_results.json",results)
        prediction=calibrated(rows,base["predictions"],plan["calibration_offset"])
        neutral_prediction=calibrated(self.neutral["validation"],bn["predictions"],plan["calibration_offset"])
        cm,cn=summarize(rows,prediction),summarize(self.neutral["validation"],neutral_prediction)
        ok,reasons=constraints(cm,cn,base["metrics"],bn["metrics"],self.c["constraints"])
        write_json(self.out/"calibration_test.json",dict(offset=plan["calibration_offset"],metrics=cm,neutral=cn,feasible=ok,violations=reasons,confidence=bootstrap(rows,base["predictions"],prediction,self.c["bootstrap_replicates"],self.c["seed"])))
        self.finish("test", rows=len(rows),configs=len(results))

    def stability(self):
        self.setup()
        spec=json.loads((self.out/"test_plan_frozen.json").read_text())["primary"]
        panel=balanced_panel(self.rows["selection"],1,self.c["seed"])[:12]
        records=[]
        for condition in (None,spec):
            self.runner.set_edit(condition,self.fitted)
            for row in panel:
                batch,_=self.runner.predict(row)
                sequential=[]
                for p in range(6):
                    prompt=self.runner.builder.build(row,p)
                    logits,_=self.runner.forward([prompt])
                    selected=logits[0,prompt["label_ids"]]
                    sequential.append(int(prompt["permutation"][int(selected.argmax())]))
                records.append(dict(uid=row["uid"],condition=condition["id"] if condition else "baseline",batch6=[r["prediction"] for r in batch["rotations"]],batch1=sequential))
            self.runner.set_edit(None)
        write_json(self.out/"stability_audit.json",dict(records=records,prediction_flips=sum(a!=b for r in records for a,b in zip(r["batch6"],r["batch1"])),comparisons=6*len(records),scope="Fixed 12 development questions, BF16 batch-shape check; not a full precision guarantee"))
        self.finish("stability")

    def report(self):
        # The separate renderer may be extended without invalidating frozen model computations.
        from .fairsteer_report import render
        render(self.out,self.data)
        self.finish("report")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("all","smoke","fit","checks","screen","select","test","stability","report"))
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiment_fairsteer.toml")
    args=parser.parse_args()
    experiment=FairSteerExperiment(args.config,suffix="_smoke" if args.stage=="smoke" else "")
    stages=("fit","checks","screen","select","test","stability","report") if args.stage=="all" else (args.stage,)
    for stage in stages:
        if experiment.state["stages"].get(stage,{}).get("completed"):
            experiment.log("stage_cached",stage=stage)
            continue
        getattr(experiment,stage)()


if __name__ == "__main__":
    main()
