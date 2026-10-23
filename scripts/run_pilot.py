"""Run/resume the primary ablation and fixed node-subnetwork diagnostics."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment_pilot.toml")
    args = parser.parse_args()
    config = args.config.resolve()
    spec = tomllib.loads(config.read_text(encoding="utf-8"))
    out = ROOT / spec["output_dir"]

    def invoke(*argv):
        subprocess.run([sys.executable, "-u", *map(str, argv)], cwd=ROOT, check=True)

    manifest = out / "run_manifest.json"
    for stage in ("fit", "attribute", "check"):
        state = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        if not state.get("stages", {}).get(stage, {}).get("completed"):
            invoke("-m", "mechanism", stage, "--config", config)
    if not (out / "global_nodes_plan_frozen.json").exists():
        invoke("scripts/run_global_nodes.py", "preregister", "--config", config)
    invoke("-m", "mechanism", "all", "--config", config)
    invoke("scripts/run_global_nodes.py", "run", "--config", config)
    invoke("scripts/summarize_ablation.py", out)
    # The finalizer is deliberately tied to the documented first pilot. A new
    # experiment must preserve its own report instead of overwriting that pilot.
    if out.resolve() == (ROOT / "results/pilot_20261011_v4").resolve():
        invoke("scripts/finalize_report.py")
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
