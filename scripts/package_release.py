"""Create an allowlisted source bundle, without local data/models/research history."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = [
    "egus",
    "configs",
    "docs",
    "reproducibility",
    "benchmarks",
    "tests",
    "scripts",
    ".github",
]
FILES = [
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
]


def build(destination):
    output = Path(destination).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / name for name in FILES if (ROOT / name).exists()]
    for name in DIRECTORIES + ["requirements", "third_party/FairSteer"]:
        paths.extend(
            p
            for p in (ROOT / name).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
        )
    paths = sorted(set(paths))
    forbidden = {"fixed.py", "project.py", "fixed_controls.toml"}
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.name in forbidden or (
            relative.parts[0] == "benchmarks" and relative.parts[1] != "main"
        ):
            raise ValueError(f"Non-paper artifact in release: {relative}")
    if output in paths:
        raise ValueError("Write the release bundle outside the allowlisted source folders")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    summary = {
        "files": len(paths),
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "content_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {"files": len(paths), "archive": str(output), "sha256": summary["archive_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/egus-source.zip")
    print(json.dumps(build(parser.parse_args().output), indent=2))
