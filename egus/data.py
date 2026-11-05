"""Hash-checked restoration and validation of the paper's BBQ partition."""

import csv
import io
import json
from collections import Counter
from pathlib import Path
import time
import urllib.request

from .config import ROOT
from .io import read, rows, sha


def _store(path, blob, expected):
    import hashlib

    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    if path.exists():
        if path.read_bytes() != blob:
            raise FileExistsError(f"Existing data differ: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(blob)
    temporary.replace(path)


def prepare(raw_dir=None, data_dir=None, offline=False):
    manifest = read(ROOT / "data/manifest.json")
    raw = Path(raw_dir or ROOT / "data/raw")
    destination = Path(data_dir or ROOT / "data/prepared")
    inputs = {}
    for name, spec in manifest["upstream"].items():
        path = raw / spec["path"]
        if path.exists():
            if sha(path) != spec["sha256"]:
                raise ValueError(f"Upstream data checksum mismatch: {path}")
            inputs[name] = path.read_bytes()
            continue
        if offline:
            raise FileNotFoundError(path)
        for url in spec["urls"]:
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    blob = response.read()
                _store(path, blob, spec["sha256"])
                inputs[name] = blob
                break
            except (OSError, ValueError):
                time.sleep(1)
        if name not in inputs:
            raise RuntimeError(f"Unable to retrieve upstream input: {name}")
    metadata = {}
    for r in csv.DictReader(io.StringIO(inputs["metadata"].decode("utf-8-sig"), newline="")):
        metadata.setdefault((r["category"], int(r["example_id"])), r)
    source = []
    for category in manifest["categories"]:
        for line in inputs[category].decode("utf-8").splitlines():
            r = json.loads(line)
            key = r["category"], int(r["example_id"])
            target = metadata[key]["target_loc"]
            r["target_loc"] = None if target in ("NA", "") else int(target)
            r["uid"] = f"{key[0]}:{key[1]}"
            r["unknown_index"] = next(
                i for i in range(3) if r["answer_info"][f"ans{i}"][1].lower() == "unknown"
            )
            source.append(r)
    if len(source) != manifest["source_rows"]:
        raise ValueError("Unexpected BBQ corpus size")
    lookup = {r["uid"]: r for r in source}
    mapping = read(ROOT / "data/family_mapping.json")["mapping"]
    for filename, spec in manifest["files"].items():
        values = []
        for uid in spec["uids"]:
            r = dict(lookup[uid])
            family = f"{r['category']}:{r['question_index']}"
            r["family"] = mapping.get(family, family)
            values.append(r)
        blob = "".join(
            json.dumps(r, ensure_ascii=False) + manifest["jsonl_newline"] for r in values
        ).encode("utf-8")
        _store(destination / filename, blob, spec["sha256"])
    blob = (ROOT / "data/training.json").read_bytes()
    _store(destination / "training.json", blob, manifest["training_sha256"])
    return {"prepared": len(manifest["files"]) + 1, "data_dir": str(destination)}


def audit(data_dir, seeds):
    data = Path(data_dir)
    manifest = read(ROOT / "data/manifest.json")
    names = ["validation.jsonl", "test.jsonl"] + [f"train_{s}.jsonl" for s in seeds]
    for name in names:
        if sha(data / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Prepared data checksum mismatch: {name}")
    if sha(data / "training.json") != manifest["training_sha256"]:
        raise ValueError("Training manifest checksum mismatch")
    validation, test = rows(data / "validation.jsonl"), rows(data / "test.jsonl")
    assert len(validation) == 352 and len(test) == 704
    assert len({r["uid"] for r in validation + test}) == 1056
    heldout = {r["family"] for r in validation + test}
    assert {r["family"] for r in validation}.isdisjoint({r["family"] for r in test})
    training = read(data / "training.json")
    for seed in seeds:
        train = rows(data / f"train_{seed}.jsonl")
        assert len(train) == len({r["uid"] for r in train}) == 2200
        assert not heldout & {r["family"] for r in train}
        counts = Counter((r["category"], r["context_condition"]) for r in train)
        assert len(counts) == 22 and set(counts.values()) == {100}
        assert set(training[str(seed)]["direction_uids"]) <= {r["uid"] for r in train}
    return training


def training_positions(training):
    index = {global_index: position for position, global_index in enumerate(training["indices"])}
    return [index[i] for i in training["probe_indices"]]
