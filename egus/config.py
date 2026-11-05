"""A single experiment schema for every supported checkpoint."""

import copy
import os
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"egus", "evidence_steering", "fairsteer", "knowbias"}


def load(path, models=None, output=None, models_root=None):
    path = Path(path).resolve()
    with path.open("rb") as f:
        config = tomllib.load(f)
    config = copy.deepcopy(config)
    if models_root is not None:
        config["models_root"] = str(Path(models_root).resolve())
    if output is not None:
        config["output_dir"] = str(Path(output).resolve())
    if models:
        unknown = set(models) - {m["name"] for m in config["models"]}
        if unknown:
            raise ValueError(f"Unknown model IDs: {sorted(unknown)}")
        config["models"] = [m for m in config["models"] if m["name"] in models]
    for key in ["data_dir", "output_dir", "models_root"]:
        config[key] = str((ROOT / os.path.expandvars(config[key])).resolve())
    for spec in config["models"]:
        spec["path"] = str(Path(config["models_root"]) / spec["checkpoint"])
    if not config["models"] or not config["methods"] or not set(config["methods"]) <= METHODS:
        raise ValueError("Select supported models and methods")
    if not 0 <= config["gate_fpr_limit"] <= 1:
        raise ValueError("The false-trigger limit must be a probability")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seed IDs must be unique")
    if any(a < 0 for a in config["alphas"]) or 0 not in config["alphas"]:
        raise ValueError("The strength grid must include zero and contain nonnegative values")
    if "fairsteer" in config["methods"] and config["fairsteer_backend"] != "cuml":
        raise ValueError("The paper baseline requires author-source cuML FairSteer")
    if config["selection_mode"] != "grid":
        raise ValueError("The paper protocol requires full-layer grid selection")
    if config["fairsteer_backend"] == "cuml" and (
        config["probe_C"],
        config["probe_max_iter"],
        config["probe_tol"],
    ) != (1.0, 10000, 1e-4):
        raise ValueError(
            "The native profile calls the original cuML defaults; do not override the solver settings"
        )
    config["config_file"] = str(path)
    return config
