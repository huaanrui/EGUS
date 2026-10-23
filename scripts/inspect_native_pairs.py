"""Inspect outcome-independent native identity-order pairs in pinned BBQ."""
from __future__ import annotations
import collections
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_bbq import normalize

REV = "bea11bd97d79217245b5871acd247b9d6eb24598"
RAW = ROOT / "data/raw/bbq" / REV


def load_categories(categories):
    meta = {}
    with (RAW / "supplemental/additional_metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            meta.setdefault((r["category"], int(r["example_id"])), r)
    families = json.loads((ROOT / "configs/bbq_template_families.json").read_text())['mapping']
    rows = []
    for cat in categories:
        for line in (RAW / f"data/{cat}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            m = meta[(cat, r["example_id"])]
            if m["target_loc"] == "NA":
                continue
            n = normalize(r, m)
            n["template_group_id"] = families.get(n["template_group_id"], n["template_group_id"])
            rows.append(n)
    return rows


def canonical(row):
    people = [i for i in range(3) if i != row["unknown_index"]]
    aliases = {i: row["source_answer_info"][f"ans{i}"][0] for i in people}
    starts = {}
    for i, alias in aliases.items():
        m = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", row["context"], re.I)
        if not m:
            return None
        starts[i] = m.start()
    order = sorted(people, key=starts.get)
    mapping = {i: order.index(i) for i in people}
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(aliases.values(), key=len, reverse=True)) + r")(?!\w)", re.I)
    lookup = {a.casefold(): f"PERSON{mapping[i]}" for i, a in aliases.items()}
    text = pattern.sub(lambda m: lookup[m[0].casefold()], row["context"])
    # Keep pronouns/articles and every other word exact: no generated text.
    signature = tuple(sorted((aliases[i].casefold(), row["answers"][i].casefold(), row['source_answer_info'][f'ans{i}'][1]) for i in people))
    return (row["template_group_id"], row["question_polarity"], row["question"], text.casefold(), signature, row["answers"][row["unknown_index"]].casefold()), mapping


def native_pairs(rows):
    bins = collections.defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        if row["context_condition"] != "ambig":
            continue
        c = canonical(row)
        if c is not None:
            key, mapping = c
            bins[key][mapping[row["stereotype_index"]]].append((row, mapping))
    return [(key, sides) for key, sides in bins.items() if sides[0] and sides[1]]


def source_semantic_exclusion(row):
    # Conservative exclusion after pre-selection manual audit found explicit
    # 'who is trans' clauses inconsistent with source answer_info identity labels.
    # Exclude the whole affected identity subtype, not only observed errors.
    return any('trans' in str(v[1]).casefold() for v in row.get('source_answer_info', {}).values())


if __name__ == "__main__":
    rows = load_categories(["Age", "Gender_identity", "SES"])
    pairs = native_pairs(rows)
    print(json.dumps(dict(pairs=len(pairs), families=dict(collections.Counter(k[0].split(':')[0] for k, s in pairs))), indent=2))
    for cat in ["Age", "Gender_identity", "SES"]:
        for key, sides in [p for p in pairs if p[0][0].startswith(cat + ':')][:2]:
            print(key)
            for side in (0, 1):
                print(sides[side][0][0]['context'], sides[side][0][0]['answers'])
