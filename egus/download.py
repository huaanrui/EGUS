"""Pinned checkpoint downloads with resume, SHA256 and bounded concurrency."""

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import hashlib

import json

import shutil

import subprocess

import time


def emit(event, **details):
    print(
        json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **details}),
        flush=True,
    )


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(models_root, models=None, endpoint="https://hf-mirror.com"):
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "reproducibility/models.json").read_text()
    )
    if models:
        unknown = set(models) - {m["name"] for m in manifest["models"]}
        if unknown:
            raise ValueError(f"Unknown model IDs: {sorted(unknown)}")
        manifest["models"] = [m for m in manifest["models"] if m["name"] in models]
    root = Path(models_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    need = sum(
        (
            max(
                0,
                (f["size"] or 0)
                - (
                    (root / m["repo_id"].split("/")[-1] / (f["name"] + ".partial")).stat().st_size
                    if (root / m["repo_id"].split("/")[-1] / (f["name"] + ".partial")).exists()
                    else 0
                ),
            )
            for m in manifest["models"]
            for f in m["files"]
            if not (root / m["repo_id"].split("/")[-1] / f["name"]).exists()
        )
    )
    free = shutil.disk_usage(root).free
    emit("disk_check", required_bytes=need, free_bytes=free)
    if free < need + 2000000000:
        raise RuntimeError("Insufficient disk space including 2 GB reserve")
    for model in manifest["models"]:
        folder = root / model["repo_id"].split("/")[-1]
        folder.mkdir(exist_ok=True)
        expected = {f["name"]: f for f in model["files"]}
        checks = {}

        def fetch(item):
            path = folder / item["name"]
            if path.exists():
                if item["size"] is not None and path.stat().st_size != item["size"]:
                    raise RuntimeError(f"Existing file has wrong size: {path}")
            else:
                url = f"{endpoint}/{model['repo_id']}/resolve/{model['revision']}/{item['name']}?download=true"
                partial = path.with_name(path.name + ".partial")
                emit("file_start", model=model["repo_id"], file=item["name"], bytes=item["size"])
                if shutil.which("aria2c"):
                    command = [
                        "aria2c",
                        "--disable-ipv6=true",
                        "--continue=true",
                        "--allow-overwrite=true",
                        "--auto-file-renaming=false",
                        "--file-allocation=none",
                        "--max-connection-per-server=8",
                        "--split=8",
                        "--min-split-size=8M",
                        "--max-tries=8",
                        "--retry-wait=3",
                        "--connect-timeout=15",
                        "--timeout=60",
                        "--lowest-speed-limit=1024",
                        "--summary-interval=0",
                        "--console-log-level=warn",
                        "--download-result=hide",
                        "--enable-color=false",
                        "--show-console-readout=false",
                        "--dir",
                        str(folder),
                        "--out",
                        partial.name,
                        url,
                    ]
                else:
                    command = [
                        "curl",
                        "-4",
                        "--location",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--retry",
                        "5",
                        "--retry-all-errors",
                        "--retry-delay",
                        "3",
                        "--connect-timeout",
                        "15",
                        "--speed-limit",
                        "1024",
                        "--speed-time",
                        "60",
                        "--max-time",
                        "3600",
                        "--continue-at",
                        "-",
                        "--output",
                        str(partial),
                        url,
                    ]
                r = subprocess.run(command)
                if r.returncode:
                    raise RuntimeError(f"Download failed: {path.name}, status={r.returncode}")
                if item["size"] is not None and partial.stat().st_size != item["size"]:
                    raise RuntimeError(f"Download size mismatch: {partial}")
                partial.replace(path)
            actual = checksum(path)
            if item["sha256"] and actual != item["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            emit(
                "file_verified",
                model=model["repo_id"],
                file=item["name"],
                bytes=path.stat().st_size,
            )
            return (item["name"], actual)

        with ThreadPoolExecutor(max_workers=3) as pool:
            for name, digest in pool.map(fetch, sorted(expected.values(), key=lambda x: x["name"])):
                checks[name] = digest
        status = {**model, "sha256": checks, "verified": True}
        (folder / "download_verified.json").write_text(json.dumps(status, indent=2))
        emit("model_verified", model=model["repo_id"], revision=model["revision"])
    emit("all_models_verified")
    return {"verified_models": [m["name"] for m in manifest["models"]], "models_root": str(root)}
