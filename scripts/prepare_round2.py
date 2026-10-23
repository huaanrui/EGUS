"""Freeze native counterfactual pairs and new-category confirmation data."""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mechanism.data import read_rows, sha256, write_json
from scripts.inspect_native_pairs import RAW, REV, canonical, load_categories, native_pairs, source_semantic_exclusion


def rank(seed, text):
    return hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()


def dump_rows(path, rows):
    blob = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    if path.exists() and path.read_text(encoding='utf-8') != blob:
        raise ValueError(f'Frozen dataset differs: {path}')
    path.write_text(blob, encoding='utf-8')


def aligned(row, pair_id, side, mapping):
    out = dict(row)
    perm = [next(i for i, role in mapping.items() if role == j) for j in (0, 1)] + [row['unknown_index']]
    for key in ('correct_index', 'unknown_index', 'stereotype_index', 'counterstereotype_index', 'target_person_index'):
        out[key] = perm.index(row[key])
    out['answers'] = [row['answers'][i] for i in perm]
    out['source_answer_info'] = {f'ans{j}': row['source_answer_info'][f'ans{i}'] for j, i in enumerate(perm)}
    out.update(uid=f'{pair_id}:{side}', source_uid=row['uid'], pair_id=pair_id, pair_side=side)
    return out


def choose_pairs(rows, groups, seed, limit=None):
    bins = collections.defaultdict(list)
    for key, sides in native_pairs(rows):
        if key[0] not in groups:
            continue
        chosen = [sorted(sides[s], key=lambda item: rank(seed, item[0]['uid']))[0] for s in (0, 1)]
        pair_id = 'native:' + rank(seed, repr(key))[:18]
        pair = [aligned(r, pair_id, s, m) for s, (r, m) in enumerate(chosen)]
        assert pair[0]['stereotype_index'] == 0 and pair[1]['stereotype_index'] == 1
        assert pair[0]['question'] == pair[1]['question']
        assert sorted(pair[0]['answers']) == sorted(pair[1]['answers'])
        assert pair[0]['correct_index'] == pair[1]['correct_index'] == 2
        bins[(key[0], key[1])].append(pair)
    selected = []
    ordered = sorted(bins, key=lambda k: rank(seed, repr(k)))
    # One pair per family/polarity first; then one additional pair at most.
    for depth in range(2):
        for key in ordered:
            options = sorted(bins[key], key=lambda p: rank(seed, p[0]['pair_id']))
            if depth < len(options):
                selected.append(options[depth])
                if limit and len(selected) >= limit:
                    return selected
    return selected


def main():
    config_path = ROOT / 'configs/experiment_round2.toml'
    c = tomllib.loads(config_path.read_text(encoding='utf-8'))
    out = ROOT / c['dataset_dir']
    if (out / 'manifest.json').exists():
        m = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        if m['config_sha256'] != sha256(config_path):
            raise ValueError('Configuration changed; use a new dataset/output directory')
        for name, record in m['files'].items():
            assert sha256(out / name) == record['sha256']
        print(json.dumps(m['summary'], ensure_ascii=False, indent=2))
        return
    old = ROOT / c['previous_dataset']
    old_m = json.loads((old / 'manifest.json').read_text(encoding='utf-8'))
    original_rows = load_categories(['Age', 'Gender_identity', 'SES'])
    excluded = [r['uid'] for r in original_rows if source_semantic_exclusion(r)]
    all_rows = [r for r in original_rows if not source_semantic_exclusion(r)]
    groups = {s: {g for g, v in old_m['group_assignment'].items() if v == s} for s in ('fit', 'selection', 'validation')}
    fit_pairs = choose_pairs(all_rows, groups['fit'], c['seed'], c['max_fit_pairs'])
    sel_pairs = choose_pairs(all_rows, groups['selection'], c['seed'])
    # Evidence pairs preserve exact identities, answer order, question and source stem.
    dis_lookup = collections.defaultdict(list)
    for r in all_rows:
        if r['context_condition'] == 'disambig' and r['template_group_id'] in groups['fit']:
            dis_lookup[(r['template_group_id'], r['question'], tuple(r['answers']))].append(r)
    evidence = []
    for r in sorted([r for r in all_rows if r['context_condition'] == 'ambig' and r['template_group_id'] in groups['fit']], key=lambda r: rank(c['seed'], r['uid'])):
        matches = [d for d in dis_lookup[(r['template_group_id'], r['question'], tuple(r['answers']))] if d['context'].startswith(r['context'] + ' ')]
        if not matches or r['template_group_id'] in {a['template_group_id'] for a, b in evidence}:
            continue
        evidence.append((r, sorted(matches, key=lambda d: rank(c['seed'], d['uid']))[0]))
        if len(evidence) >= c['max_evidence_pairs']:
            break
    source_files = {}
    for cat in c['new_test_categories']:
        path = RAW / f'data/{cat}.jsonl'
        url = f'https://raw.githubusercontent.com/nyu-mll/BBQ/{REV}/data/{cat}.jsonl'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        source_files[cat] = dict(url=url, sha256=sha256(path))
    new_rows = load_categories(c['new_test_categories'])
    buckets = collections.defaultdict(list)
    for r in new_rows:
        buckets[(r['template_group_id'], r['context_condition'], r['question_polarity'], r['label_type'])].append(r)
    test = []
    for key in sorted(buckets):
        test.extend(sorted(buckets[key], key=lambda r: rank(c['seed'], r['uid']))[:c['test_rows_per_stratum']])
    out.mkdir(parents=True, exist_ok=True)
    datasets = dict(fit_pairs=[r for pair in fit_pairs for r in pair], selection_pairs=[r for pair in sel_pairs for r in pair], evidence_fit=[dict(r, evidence_pair=i) for i, pair in enumerate(evidence) for r in pair], selection=[r for r in read_rows(old / 'selection.jsonl') if not source_semantic_exclusion(r)], historical_validation=read_rows(old / 'validation.jsonl'), test=sorted(test, key=lambda r: r['uid']))
    for name, rows in datasets.items():
        dump_rows(out / f'{name}.jsonl', rows)
    summaries = {name: dict(rows=len(rows), families=len({r['template_group_id'] for r in rows}), categories=dict(collections.Counter(r['category'] for r in rows))) for name, rows in datasets.items()}
    all_known = set(old_m['group_assignment'])
    assert not all_known & {r['template_group_id'] for r in test}
    assert not {r['template_group_id'] for r in datasets['fit_pairs']} & {r['template_group_id'] for r in datasets['selection_pairs']}
    manifest = dict(schema_version=2, config_sha256=sha256(config_path), previous_manifest_sha256=sha256(old / 'manifest.json'), new_sources=source_files, summary=summaries, source_semantic_exclusions=dict(rule='Conservatively exclude trans/nonTrans identity subtype after manual pre-selection audit found source identity clauses inconsistent with answer_info labels. Does not assert every excluded row is wrong.', source_uids=excluded, example=['Gender_identity:2870']), files={f'{name}.jsonl': dict(sha256=sha256(out / f'{name}.jsonl'), rows=len(rows)) for name, rows in datasets.items()}, policy='Existing fit/selection family assignments unchanged. Historical validation is development history only. New-category test is untouched cross-category confirmation, not same-category IID validation. Native pairs differ only in explicit entity identities/order; questions, answer strings and all remaining context words match after identity-role canonicalization. No outcome-based pair inclusion.')
    write_json(out / 'manifest.json', manifest)
    # Deterministic audit samples for human inspection, before model evaluation.
    audit = []
    for cat in ('Age', 'Gender_identity', 'SES'):
        audit.extend([p for p in fit_pairs if p[0]['category'] == cat][:3])
    write_json(out / 'native_pair_semantic_audit.json', dict(rule='Native text only; conservative exact canonical match, same identity/answer multiset, same unknown string; alignment by first mention.', samples=audit))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
