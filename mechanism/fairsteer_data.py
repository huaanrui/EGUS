"""Pinned, family-disjoint data and explicit FairSteer prompt positions."""
from __future__ import annotations

import ast
import collections
import hashlib
import itertools
import json
from pathlib import Path
import tomllib

from .data import read_rows, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PERMUTATIONS6 = tuple(itertools.permutations(range(3)))


def rank(seed, value):
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def source_fewshot():
    """Read the literal without importing or executing external author code."""
    source = ROOT / "research/sources/20261012/FairSteer/utils.py"
    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        raise ValueError("Unexpected executable few-shot expression")
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bbq_few_shot_examples" for t in node.targets):
            return literal(node.value)
    raise ValueError("Pinned author few-shot literal is missing")


def fair_sample(rows, cap, seed):
    bins = collections.defaultdict(list)
    for row in rows:
        key = (row["template_group_id"], row["context_condition"], row["question_polarity"], row.get("label_type", ""))
        bins[key].append(row)
    bins = {k: sorted(v, key=lambda r: rank(seed, r["uid"])) for k, v in bins.items()}
    out = []
    for depth in range(max(map(len, bins.values()), default=0)):
        for key in sorted(bins, key=lambda k: rank(seed, repr(k))):
            if depth < len(bins[key]):
                out.append(bins[key][depth])
                if len(out) == cap:
                    return sorted(out, key=lambda r: r["uid"])
    return sorted(out, key=lambda r: r["uid"])


def frozen_rows(path, rows):
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.exists() and path.read_text(encoding="utf-8") != blob:
        raise ValueError(f"Refusing to change frozen data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(blob, encoding="utf-8")


def direction_rows(fit, config):
    condition = config.get("direction_context", "all")
    if condition not in ("all", "ambig"):
        raise ValueError("Unsupported direction context policy")
    eligible = [r for r in fit if r["correct_index"] != r["stereotype_index"]
                and (condition == "all" or r["context_condition"] == condition)]
    per_cat, remainder = divmod(config["direction_examples"], len(config["categories"]))
    result = []
    for i, cat in enumerate(config["categories"]):
        result.extend(fair_sample([r for r in eligible if r["category"] == cat],
                                 per_cat + (i < remainder), config["seed"] + 2))
    if len(result) != config["direction_examples"]:
        raise ValueError("Not enough eligible direction pairs; do not borrow development data")
    if condition == "ambig" and any(r["correct_index"] != r["unknown_index"] for r in result):
        raise ValueError("Ambiguous direction examples must have unknown as the correct answer")
    return result


def prepare(config):
    from scripts.inspect_native_pairs import RAW, REV, load_categories, source_semantic_exclusion
    config = Path(config).resolve()
    c = tomllib.loads(config.read_text(encoding="utf-8"))
    out = ROOT / c["dataset_dir"]
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config_sha256"] != sha256(config):
            raise ValueError("Frozen dataset configuration changed")
        for name, record in manifest["files"].items():
            if sha256(out / name) != record["sha256"]:
                raise ValueError(f"Frozen data changed: {name}")
        return manifest
    source = load_categories(c["categories"])
    fewshot = source_fewshot()
    demo_lines = {piece.splitlines()[0] for piece in fewshot.strip().split("\n\n")}
    demo_families = {r["template_group_id"] for r in source if f'{r["context"]} {r["question"]}' in demo_lines}
    rows = [r for r in source if not source_semantic_exclusion(r) and r["template_group_id"] not in demo_families]
    old = json.loads((ROOT / c["previous_dataset"] / "manifest.json").read_text(encoding="utf-8"))
    assignment = dict(old["group_assignment"])
    for cat in c["fresh_categories"]:
        groups = sorted({r["template_group_id"] for r in rows if r["category"] == cat}, key=lambda g: rank(c["seed"], g))
        nf, ns = round(.6 * len(groups)), round(.2 * len(groups))
        for i, group in enumerate(groups):
            assignment[group] = "fit" if i < nf else "selection" if i < nf + ns else "test"
    probe_val_groups = set()
    for cat in c["categories"]:
        groups = sorted({r["template_group_id"] for r in rows if r["category"] == cat and assignment[r["template_group_id"]] == "fit"}, key=lambda g: rank(c["seed"] + 1, g))
        probe_val_groups.update(groups[:max(1, round(c["probe_validation_fraction"] * len(groups)))])
    fit_all = []
    for cat in c["categories"]:
        pool = [r for r in rows if r["category"] == cat and assignment[r["template_group_id"]] == "fit"]
        fit_all.extend(fair_sample(pool, c["fit_rows_per_category"], c["seed"]))
    datasets = {
        "fit": [r for r in fit_all if r["template_group_id"] not in probe_val_groups],
        "probe_validation": [r for r in fit_all if r["template_group_id"] in probe_val_groups],
    }
    for split, cap in (("selection", c["selection_rows_per_stratum"]), ("test", c["test_rows_per_stratum"])):
        bins = collections.defaultdict(list)
        for row in rows:
            if assignment[row["template_group_id"]] == split:
                bins[(row["template_group_id"], row["context_condition"], row["question_polarity"], row["label_type"])].append(row)
        datasets[split] = sorted([r for key, pool in sorted(bins.items()) for r in sorted(pool, key=lambda r: rank(c["seed"], r["uid"]))[:cap]], key=lambda r: r["uid"])
    datasets["direction_fit"] = direction_rows(datasets["fit"], c)
    # The pinned author file is JSON Lines, with four choices and answers in A/B/C.
    mmlu_path = ROOT / "research/sources/20261012/FairSteer/generate_data/mmlu.json"
    raw_mmlu = read_rows(mmlu_path)
    unique = {}
    for item in raw_mmlu:
        if item["answer"] not in (0, 1, 2) or len(item["choices"]) != 4:
            raise ValueError("Unexpected author MMLU preprocessing")
        key = rank(0, " ".join(item["question"].casefold().split()))
        unique.setdefault(key, dict(item, uid="mmlu:" + key[:20], template_group_id="mmlu:" + key))
    ordered = sorted(unique.values(), key=lambda r: rank(c["seed"], r["uid"]))
    offset = 0
    for split, count in (("fit", c["mmlu_train_rows"]), ("probe_validation", c["mmlu_probe_validation_rows"]), ("selection", c["mmlu_selection_rows"]), ("test", c["mmlu_test_rows"])):
        datasets["mmlu_" + split] = ordered[offset:offset + count]
        offset += count
    if offset > len(ordered):
        raise ValueError("Insufficient disjoint MMLU questions")
    # The shared few-shot exemplars are training material; none may cross evaluation families.
    families = {s: {r["template_group_id"] for r in datasets[s]} for s in ("fit", "probe_validation", "selection", "test")}
    for a, b in itertools.combinations(families, 2):
        if families[a] & families[b]:
            raise ValueError(f"Family leakage: {a}, {b}")
    for a, b in itertools.combinations(("fit", "probe_validation", "selection", "test"), 2):
        if {r["prompt_fingerprint"] for r in datasets[a]} & {r["prompt_fingerprint"] for r in datasets[b]}:
            raise ValueError(f"Prompt leakage: {a}, {b}")
    if any(r["category"] not in c["fresh_categories"] for r in datasets["test"]):
        raise ValueError("Historical categories cannot be called fresh confirmation")
    out.mkdir(parents=True, exist_ok=True)
    for name, values in datasets.items():
        frozen_rows(out / f"{name}.jsonl", values)
    (out / "fewshot.txt").write_text(fewshot, encoding="utf-8")
    summary = {name: dict(rows=len(values), families=len({r["template_group_id"] for r in values}), categories=dict(collections.Counter(r.get("category", "mmlu") for r in values))) for name, values in datasets.items()}
    manifest = dict(
        schema_version=1, config_sha256=sha256(config), summary=summary,
        group_assignment=assignment, probe_validation_groups=sorted(probe_val_groups),
        demo_families_excluded=sorted(demo_families),
        source_semantic_exclusions=sum(source_semantic_exclusion(r) for r in source),
        sources={f"data/{cat}.jsonl": dict(revision=REV, sha256=sha256(RAW / f"data/{cat}.jsonl")) for cat in c["categories"]},
        author_mmlu_sha256=sha256(mmlu_path), family_mapping_sha256=sha256(ROOT / "configs/bbq_template_families.json"),
        previous_manifest_sha256=sha256(ROOT / c["previous_dataset"] / "manifest.json"),
        files={p.name: dict(sha256=sha256(p)) for p in out.glob("*.jsonl")},
        policy="Old category family roles preserved. Newly downloaded categories have fit/selection/test within each category. Probe validation is family-disjoint inside fit; direction pairs only use probe-training families. Test is loaded only after a plan is frozen. MMLU questions are deduplicated and disjoint. All fixed few-shot exemplar families are excluded from evaluation.",
    )
    if "direction_context" in c:
        manifest["direction_context"] = c["direction_context"]
        manifest["test_role"] = c.get("test_role", "fresh_confirmation")
    manifest["files"]["fewshot.txt"] = dict(sha256=sha256(out / "fewshot.txt"))
    write_json(out / "manifest.json", manifest)
    return manifest


class FairPromptBuilder:
    """Source formatting, answer labels and teacher-forced positions are explicit."""
    def __init__(self, tokenizer, fewshot, max_tokens=2048):
        self.tokenizer, self.fewshot, self.max_tokens = tokenizer, fewshot, max_tokens
        self.cache = {}

    def build(self, row, permutation=0, *, stage="evaluation", shots=False, answer=None):
        key = (row["uid"], permutation, stage, shots, answer)
        if key in self.cache:
            return self.cache[key]
        perm = PERMUTATIONS6[permutation]
        if row.get("subject") is not None:
            # Same A/B/C truncation as the authors; their file already excludes answer D.
            text = row["question"] + "\n" + "\n".join(f"{lab}. {row['choices'][i]}" for lab, i in zip("ABC", perm)) + "\nAnswer:"
        else:
            body = f'{row["context"]} {row["question"]}\n'
            if stage == "evaluation":
                choices = "\n".join(f"({lab}) {row['answers'][i]}" for lab, i in zip("ABC", perm))
            else:
                choices = "\n".join(f"{lab}. {row['answers'][i]}" for lab, i in zip("ABC", perm))
            text = body + choices + "\nAnswer:"
        if shots:
            text = self.fewshot + text
        prefix = self.tokenizer.encode(text, add_special_tokens=False)
        label_ids = []
        for label in "ABC":
            completed = self.tokenizer.encode(text + " " + label, add_special_tokens=False)
            if completed[:len(prefix)] != prefix or len(completed) != len(prefix) + 1:
                raise ValueError("Answer is not a context-verified single token")
            label_ids.append(completed[-1])
        if answer is not None:
            label = perm.index(answer)
            text += " " + "ABC"[label]
            ids = prefix + [label_ids[label]]
        else:
            ids = prefix
        if len(ids) > self.max_tokens:
            raise ValueError(f'{row["uid"]}: {len(ids)} exceeds {self.max_tokens}; no truncation')
        prompt = dict(ids=ids, text=text, permutation=perm, label_ids=label_ids,
                      extraction_position=len(ids)-1, prefix_last_position=len(prefix)-1,
                      position_kind="teacher_forced_answer" if answer is not None else "last_prompt_token")
        self.cache[key] = prompt
        return prompt
