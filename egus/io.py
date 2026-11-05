"""Atomic metadata and input fingerprints."""

import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
