"""Recompute second-pilot evaluation and paired-diagnostic metrics from predictions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mechanism.data import neutral_controls, read_rows, sha256, write_json
from mechanism.round2 import extended_metrics


def main():
    config = ROOT / 'configs/experiment_round2.toml'
    c = tomllib.loads(config.read_text(encoding='utf-8'))
    out, data = ROOT / c['output_dir'], ROOT / c['dataset_dir']
    dm = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    assert sha256(config) == dm['config_sha256']
    for name, record in dm['files'].items():
        assert sha256(data / name) == record['sha256']
    rows = {}
    for name in ('selection', 'test', 'selection_pairs'):
        rows.update({r['uid']: r for r in read_rows(data / f'{name}.jsonl')})
    for values in neutral_controls().values():
        rows.update({r['uid']: r for r in values})
    lock = json.loads((out / 'test_lock.json').read_text(encoding='utf-8'))
    assert lock['plan_sha256'] == sha256(out / 'test_plan_frozen.json')
    assert lock['fitted_sha256'] == sha256(out / 'fitted.pkl')
    assert lock['test_sha256'] == sha256(data / 'test.jsonl')
    verified = []
    for p in sorted((out / 'predictions').glob('*.json')):
        v = json.loads(p.read_text(encoding='utf-8'))
        rr = [rows[x['uid']] for x in v['predictions']]
        assert len({r['uid'] for r in rr}) == len(rr), p.name
        assert extended_metrics(rr, v['predictions']) == v['metrics'], p.name
        assert v['identity']['rows_sha256'] == hashlib.sha256(json.dumps(rr, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), p.name
        assert v['identity']['fitted_sha256'] == lock['fitted_sha256'], p.name
        assert v['rotation_batch_size'] == 3, p.name
        for x in v['predictions']:
            assert len(x['rotations']) == 3
            for r in x['rotations']:
                assert r['semantic_logprobs'][r['prediction']] == max(r['semantic_logprobs']), (p.name, x['uid'])
        verified.append(p.name)
    native = read_rows(data / 'fit_pairs.jsonl') + read_rows(data / 'selection_pairs.jsonl')
    from scripts.inspect_native_pairs import canonical, source_semantic_exclusion
    for i in range(0, len(native), 2):
        a, b = native[i:i+2]
        assert a['pair_id'] == b['pair_id']
        assert a['stereotype_index'] == 0 and b['stereotype_index'] == 1
        assert a['question'] == b['question']
        assert sorted(a['answers']) == sorted(b['answers'])
        assert canonical(a)[0] == canonical(b)[0]
        assert not source_semantic_exclusion(a) and not source_semantic_exclusion(b)
    assert not {r['template_group_id'] for r in read_rows(data / 'test.jsonl')} & {r['template_group_id'] for r in read_rows(data / 'selection.jsonl') + read_rows(data / 'fit_pairs.jsonl')}
    from scripts.calibration_round2 import calibrated
    cp = json.loads((out / 'calibration/plan_frozen.json').read_text(encoding='utf-8'))
    ct = json.loads((out / 'calibration/test_results.json').read_text(encoding='utf-8'))
    assert sha256(out / 'calibration/plan_frozen.json') == ct['plan_sha256']
    tb = json.loads((out / 'predictions/test_baseline.json').read_text(encoding='utf-8'))
    tr = read_rows(data / 'test.jsonl')
    assert calibrated(tr, tb['predictions'], cp['selected']['offset']) == ct['predictions']
    assert extended_metrics(tr, ct['predictions']) == ct['bbq']
    assert cp['selection_baseline_sha256'] == sha256(out / 'predictions/selection_baseline.json')
    paired_verified = []
    for p in sorted((out / 'paired_diagnostics').glob('*.json')):
        v = json.loads(p.read_text(encoding='utf-8'))
        rr = [rows[u] for u in v['identity']['uids']]
        assert extended_metrics(rr, v['predictions']) == v['bbq'], p.name
        assert extended_metrics(rr, v['baseline_predictions']) == v['baseline'], p.name
        assert v['identity']['fitted_sha256'] == lock['fitted_sha256']
        assert all(len(r['rotations']) == 3 for r in v['predictions'])
        paired_verified.append(p.name)
    result = dict(status='passed', metrics_recomputed_exactly=len(verified), paired_diagnostic_files_recomputed_exactly=len(paired_verified), calibration_control_recomputed_exactly=True, all_rotation_semantics_verified=True, dataset_hashes_verified=True, native_pair_alignment_verified=True, test_family_disjoint=True, frozen_test_plan_and_fit_artifact_verified=True, report_sha256=sha256(out / 'report.md'), files=verified, paired_files=paired_verified)
    write_json(out / 'independent_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
