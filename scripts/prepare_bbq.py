"""Prepare a pinned, template-disjoint BBQ pilot using only the standard library.

This script does not load a model, choose intervention hyperparameters, or report
experimental outcomes. Source annotations remain evaluation-only metadata.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import tomllib
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_immutable(path: Path, blob: bytes) -> None:
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError(f"Refusing to overwrite different data: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


def load_source(config: dict, relative: str, provenance: dict, offline: bool) -> bytes:
    path = ROOT / config["raw_dir"] / config["revision"] / relative
    url = f'https://raw.githubusercontent.com/{config["repository"]}/{config["revision"]}/{relative}'
    if path.exists():
        blob = path.read_bytes()
    else:
        if offline:
            raise FileNotFoundError(f"Missing cached source: {path}")
        request = urllib.request.Request(url, headers={"User-Agent": "mechanism-bbq-preparation/1"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    blob = response.read()
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
        save_immutable(path, blob)
    provenance[relative] = {"url": url, "sha256": digest(blob), "bytes": len(blob)}
    return blob


def normalize(row: dict, metadata: dict) -> dict:
    uid = f'{row["category"]}:{row["example_id"]}'
    if str(row["question_index"]) != metadata["question_index"]:
        raise ValueError(f"Question index mismatch: {uid}")
    unknown = [i for i in range(3) if row["answer_info"][f"ans{i}"][1] == "unknown"]
    if len(unknown) != 1:
        raise ValueError(f"Expected one unknown answer: {uid}")
    unknown = unknown[0]
    target = int(metadata["target_loc"])
    if target not in {0, 1, 2} or target == unknown:
        raise ValueError(f"Invalid stereotype target: {uid}")
    if row["question_polarity"] not in {"neg", "nonneg"}:
        raise ValueError(f"Unknown polarity: {uid}")
    # Official metadata already adjusts target_loc for question polarity.
    # See the pinned analysis_scripts/BBQ_calculate_bias_score.R: both neg_Target
    # and nonneg_Target contribute positively. Flipping again reverses nonneg.
    stereotype = target
    target_person = target if row["question_polarity"] == "neg" else next(
        i for i in range(3) if i not in {target, unknown}
    )
    counter = next(i for i in range(3) if i not in {stereotype, unknown})
    condition = row["context_condition"]
    label = int(row["label"])
    if condition not in {"ambig", "disambig"} or label not in {0, 1, 2}:
        raise ValueError(f"Invalid label/condition: {uid}")
    if (condition == "ambig") != (label == unknown):
        raise ValueError(f"Condition and ground truth disagree: {uid}")
    answers = [row[f"ans{i}"] for i in range(3)]
    fingerprint = digest(json.dumps([
        " ".join(row["context"].casefold().split()),
        " ".join(row["question"].casefold().split()),
        sorted(" ".join(a.casefold().split()) for a in answers),
    ], ensure_ascii=False))
    return {
        "uid": uid,
        "template_group_id": f'{row["category"]}:{row["question_index"]}',
        "category": row["category"],
        "question_index": str(row["question_index"]),
        "source_example_id": row["example_id"],
        "context_condition": condition,
        "question_polarity": row["question_polarity"],
        "context": row["context"], "question": row["question"], "answers": answers,
        "correct_index": label, "unknown_index": unknown,
        "source_target_loc": target,
        "target_person_index": target_person, "stereotype_index": stereotype,
        "counterstereotype_index": counter,
        "correct_is_stereotype": label == stereotype if condition == "disambig" else None,
        "label_type": metadata["label_type"],
        "source_answer_info": row["answer_info"],
        "source_additional_metadata": row["additional_metadata"],
        "prompt_fingerprint": fingerprint,
    }


def group_splits(rows: list[dict], config: dict) -> dict[str, str]:
    groups: dict[str, set[str]] = collections.defaultdict(set)
    fingerprints: dict[str, str] = {}
    for row in rows:
        group = row["template_group_id"]
        groups[row["category"]].add(group)
        previous = fingerprints.setdefault(row["prompt_fingerprint"], group)
        if previous != group:
            raise ValueError(f"Duplicate prompt crosses templates: {previous}, {group}; merge groups before splitting")
    assignment = {}
    for category, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda g: digest(f'{config["split_seed"]}|{g}'))
        n = len(ordered)
        n_fit = round(n * config["fit_fraction"])
        n_select = round(n * config["selection_fraction"])
        if min(n_fit, n_select, n - n_fit - n_select) < 1:
            raise ValueError(f"Too few template groups in {category}")
        for i, group in enumerate(ordered):
            assignment[group] = "fit" if i < n_fit else "selection" if i < n_fit + n_select else "validation"
    return assignment


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "template_groups": len({r["template_group_id"] for r in rows}),
        "by_category": dict(sorted(collections.Counter(r["category"] for r in rows).items())),
        "by_condition": dict(sorted(collections.Counter(r["context_condition"] for r in rows).items())),
        "by_polarity": dict(sorted(collections.Counter(r["question_polarity"] for r in rows).items())),
    }


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"].startswith("superseded"):
        raise ValueError("This split is superseded and must not be used")
    family_path = ROOT / manifest["dataset"]["family_mapping"]
    if digest(family_path.read_bytes()) != manifest["family_mapping_sha256"]:
        raise ValueError("Template-family mapping has changed since preparation")
    memberships = {}
    fingerprints = {}
    all_uids = set()
    for split in ["fit", "selection", "validation"]:
        path = output / f"{split}.jsonl"
        blob = path.read_bytes()
        if digest(blob) != manifest["outputs"][path.name]["sha256"]:
            raise ValueError(f"Hash mismatch: {path}")
        rows = [json.loads(line) for line in blob.decode().splitlines()]
        if summarize(rows) != manifest["pilot"][split]:
            raise ValueError(f"Summary mismatch: {split}")
        for row in rows:
            if row["uid"] in all_uids:
                raise ValueError(f'Duplicate UID: {row["uid"]}')
            all_uids.add(row["uid"])
            for key, registry in [(row["template_group_id"], memberships), (row["prompt_fingerprint"], fingerprints)]:
                if registry.setdefault(key, split) != split:
                    raise ValueError(f"Split leakage: {key}")
            if manifest["group_assignment"][row["template_group_id"]] != split:
                raise ValueError("Group assignment mismatch")
            if {row["unknown_index"], row["stereotype_index"], row["counterstereotype_index"]} != {0, 1, 2}:
                raise ValueError("Invalid semantic answer mapping")
    for relative, record in manifest["sources"].items():
        path = ROOT / manifest["dataset"]["raw_dir"] / manifest["dataset"]["revision"] / relative
        if digest(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Source hash mismatch: {relative}")
    result = {"checks": "passed", "pilot_rows": len(all_uids), "template_groups": len(memberships),
              "note": "Automatic integrity checks only; semantic spot-check remains required."}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preexperiment_bbq.toml")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config_blob = args.config.read_bytes()
    config = tomllib.loads(config_blob.decode())["dataset"]
    family_blob = (ROOT / config["family_mapping"]).read_bytes()
    families = json.loads(family_blob)
    if families["revision"] != config["revision"]:
        raise ValueError("Template family mapping is for a different source revision")
    if abs(sum(config[k] for k in ["fit_fraction", "selection_fraction", "validation_fraction"]) - 1) > 1e-9:
        raise ValueError("Split fractions must sum to one")
    output = ROOT / config["output_dir"]
    if args.verify_only:
        print(json.dumps(verify(output), indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output already exists. Use --verify-only, or a new versioned output_dir in a copied config.")
    provenance = {}
    for name in ["LICENSE", "README.md"]:
        load_source(config, name, provenance, args.offline)
    raw_metadata = load_source(config, "supplemental/additional_metadata.csv", provenance, args.offline)
    metadata = {}
    duplicate_metadata_keys = set()
    duplicate_metadata_differing_fields = collections.Counter()
    for row in csv.DictReader(io.StringIO(raw_metadata.decode("utf-8-sig"))):
        if row["category"] not in config["categories"]:
            continue
        key = (row["category"], int(row["example_id"]))
        if key in metadata:
            duplicate_metadata_keys.add(key)
            differing = [field for field in row if metadata[key][field] != row[field]]
            duplicate_metadata_differing_fields.update(differing)
            # The pinned source has duplicate gender rows differing only in a
            # descriptive group string. Never silently reconcile label conflicts.
            if set(differing) - {"Known_stereotyped_groups"}:
                raise ValueError(f"Conflicting source metadata: {key}, {differing}")
            continue
        metadata[key] = row
    normalized, used, seen, excluded = [], set(), set(), []
    for category in config["categories"]:
        blob = load_source(config, f"data/{category}.jsonl", provenance, args.offline)
        for line in blob.decode("utf-8").splitlines():
            row = json.loads(line)
            key = (category, int(row["example_id"]))
            if key in seen:
                raise ValueError(f"Duplicate source row: {key}")
            seen.add(key)
            used.add(key)
            if metadata[key]["target_loc"] == "NA":
                excluded.append({"uid": f"{key[0]}:{key[1]}", "reason": "source_target_loc_missing",
                                 "template_group_id": f'{category}:{row["question_index"]}'})
                continue
            record = normalize(row, metadata[key])
            record["template_group_id"] = families["mapping"].get(record["template_group_id"], record["template_group_id"])
            normalized.append(record)
    assignment = group_splits(normalized, config)
    buckets = collections.defaultdict(list)
    for row in normalized:
        key = (row["template_group_id"], row["context_condition"], row["question_polarity"], row["label_type"])
        buckets[key].append(row)
    selected = []
    for _, rows in sorted(buckets.items()):
        rows.sort(key=lambda r: digest(f'{config["split_seed"]}|sample|{r["uid"]}'))
        limit = config.get("fit_rows_per_stratum", config["pilot_rows_per_stratum"]) if assignment[rows[0]["template_group_id"]] == "fit" else config["pilot_rows_per_stratum"]
        selected.extend(rows[:limit])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "dataset": config, "config_sha256": digest(config_blob),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "family_mapping_sha256": digest(family_blob),
        "sources": provenance, "source_totals": summarize(normalized),
        "raw_source_rows": len(seen), "excluded_rows": excluded,
        "unused_metadata_rows": len(set(metadata) - used),
        "duplicate_metadata_keys": len(duplicate_metadata_keys),
        "duplicate_metadata_differing_fields": dict(duplicate_metadata_differing_fields),
        "group_assignment": dict(sorted(assignment.items())),
        "full_pool": {}, "pilot": {}, "outputs": {},
        "status": "prepared_pending_manual_label_audit_and_model_baseline",
        "notes": ["Pilot sampling is per stratum; rows are not automatically counterfactual pairs.",
                  "All unsampled rows inherit their template split; never move held-out relatives into training.",
                  "Fields containing ground-truth/target metadata must never be rendered into model prompts.",
                  "Template split prevents known family leakage; paraphrase-level audit is still required."],
    }
    for split in ["fit", "selection", "validation"]:
        rows = sorted([r for r in selected if assignment[r["template_group_id"]] == split], key=lambda r: r["uid"])
        pool = [r for r in normalized if assignment[r["template_group_id"]] == split]
        manifest["pilot"][split] = summarize(rows)
        manifest["full_pool"][split] = summarize(pool)
        blob = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()
        path = output / f"{split}.jsonl"
        save_immutable(path, blob)
        manifest["outputs"][path.name] = {"sha256": digest(blob), "rows": len(rows)}
    dump_json(output / "manifest.json", manifest)
    result = verify(output)
    dump_json(output / "integrity_report.json", result)
    print(json.dumps({"output": str(output), "pilot": manifest["pilot"], "integrity": result}, indent=2))


if __name__ == "__main__":
    main()
