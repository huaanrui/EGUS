"""Call original author functions without executing their model-loading scripts."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import urllib.request

import numpy as np

from ..config import ROOT
from ..io import read, sha


def fetch_sources(verify_only=False):
    manifest = read(ROOT / "configs/authors.json")
    repos = {"FairSteer": "LiYichen99/FairSteer", "KnowBias": "JP-25/KnowBias"}
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if path.exists():
            if sha(path) != expected:
                raise ValueError(f"Modified upstream source: {relative}")
            continue
        if verify_only:
            raise FileNotFoundError(f"{relative}: run python -m egus fetch-sources")
        project, name = Path(relative).parts[1], "/".join(Path(relative).parts[2:])
        revision = manifest["revisions"][project]
        url = f"https://raw.githubusercontent.com/{repos[project]}/{revision}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        import hashlib

        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Upstream checksum mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {"verified": True, "revisions": manifest["revisions"]}


def function(filename, name, namespace):
    source = (ROOT / "third_party/FairSteer" / filename).read_text(encoding="utf-8")
    node = next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    env = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), env)
    return env[name]


def numpy(value):
    if hasattr(value, "get"):
        value = value.get()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


class FairSteer:
    def __init__(self):
        import torch

        try:
            from cuml.linear_model import LogisticRegression
        except ImportError as error:
            raise ImportError(
                "The author-source profile requires the Linux cuML environment; see REPRODUCING.md"
            ) from error
        from sklearn.metrics import accuracy_score
        from tqdm import tqdm

        self.train = function(
            "get_probes.py",
            "train_probes",
            {
                "np": np,
                "LogisticRegression": LogisticRegression,
                "accuracy_score": accuracy_score,
                "tqdm": tqdm,
            },
        )
        self.mean = function("get_steering_vectors.py", "mean_difference", {"np": np})
        self.edit = function("select_layer.py", "lt_modulated_vector_add", {"torch": torch})

    def fit(self, seed, train, labels, validation, validation_labels):
        x = np.concatenate([train, validation])[:, None, :].astype(np.float32)
        y = np.r_[labels, validation_labels]
        ti = np.flatnonzero(labels >= 0)
        vi = len(train) + np.flatnonzero(validation_labels >= 0)
        probes, accuracy = self.train(seed, ti, vi, x, y, 1)
        return probes[0], {"rows": len(ti), "validation_detector_accuracy": float(accuracy[0])}

    def add(self, state, direction, alpha):
        self.edit.__globals__["args"] = SimpleNamespace(alpha=alpha)
        self.edit.__globals__["interventions"] = {
            "selected": {
                "direction": direction,
                "probe": SimpleNamespace(predict=lambda x: np.zeros(len(x), dtype=int)),
            }
        }
        for i in range(len(state)):
            self.edit((state[i : i + 1],), "selected", start_edit_location="lt")
        return state


def knowbias_module():
    directory = ROOT / "third_party/KnowBias/kn_code"
    if not (directory / "knowledge_neurons.py").exists():
        raise FileNotFoundError("Run python -m egus fetch-sources before using KnowBias")
    name = "_egus_knowbias_author"
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(
        name + ".knowledge_neurons", directory / "knowledge_neurons.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
