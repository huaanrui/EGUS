"""Read-only numerical audit of a completed cloud BBQ campaign."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expit
from mechanism.cloud_bbq import Campaign, assert_selection_lock
from mechanism.fairsteer_author import read, rows, sha, write


def main():
    c = Campaign(); c.prepare()
    result = read(c.out / 'results.json')
    assert result['complete'] and (c.out / 'inference_complete.json').exists()
    assert_selection_lock(c.out, [m['name'] for m in c.c['models']], c.c['seeds'])
    values = rows(c.data / 'test.jsonl')
    labels = np.array([r['label'] for r in values])
    ambiguous = np.array([r['context_condition'] == 'ambig' for r in values])
    summary = {}
    for spec in c.c['models']:
        name = spec['name']; root = c.out / name
        cache = dict(np.load(root / 'cache_test.npz'))
        assert cache['uids'].tolist() == [r['uid'] for r in values]
        assert np.isfinite(cache['base']).all() and np.isfinite(cache['features']).all()
        base = cache['base'].argmax(1)
        counts = []
        for seed in c.c['seeds']:
            folder = root / f'seed_{seed}'
            selection = read(folder / 'selection_frozen.json')
            probe = dict(np.load(folder / 'probes.npz'))
            saved = dict(np.load(folder / 'predictions_test.npz'))
            vocab = read(c.data / 'training.json')[str(seed)]['vocab']
            unknown = []
            for r in values:
                matches = [i for i in range(3) if r['ans'+str(i)].strip().casefold() in vocab]
                assert len(matches) == 1
                unknown.append(matches[0])
            unknown = np.array(unknown)
            p = expit(cache['features'].astype(float) @ probe['2200_evidence_coef'] +
                      float(probe['2200_evidence_intercept']))
            gate = p > selection['thresholds']['2200_evidence']['threshold']
            direct = np.where(gate, unknown, base)
            np.testing.assert_array_equal(direct, saved['pred_evidence_direct'])
            np.testing.assert_array_equal(gate, saved['trigger_evidence_direct'])
            np.testing.assert_array_equal(gate, saved['trigger_evidence_steer'])
            counter = np.load(folder / 'cf_test_1.5.npz')['scores'].argmax(1)
            np.testing.assert_array_equal(np.where(gate, counter, base), saved['pred_evidence_steer'])
            gained = int(np.sum(ambiguous & gate & (base != labels)))
            lost = int(np.sum(~ambiguous & gate & (base == labels)))
            assert int(np.sum(direct == labels)-np.sum(base == labels)) == gained-lost
            deltas = np.concatenate([np.load(p)['delta'] for p in sorted((folder / 'direction_shards').glob('*.npz'))])
            assert len(deltas) == 110
            np.testing.assert_array_equal(deltas.mean(0), np.load(folder / 'direction.npy'))
            counts.append({'seed': seed, 'ambiguous_corrections': gained, 'sufficient_correct_answers_lost': lost})
        for method, metrics in result['models'][name]['test']['aggregate'].items():
            predictions = np.stack([np.load(root / f'seed_{s}' / 'predictions_test.npz')['pred_'+method]
                                    for s in c.c['seeds']])
            accuracy = np.mean(predictions == labels, axis=1)
            assert abs(accuracy.mean()-metrics['accuracy']['mean']) < 1e-12
            assert abs(accuracy.std(ddof=1)-metrics['accuracy']['sd']) < 1e-12
        summary[name] = counts
    write(c.out / 'numerical_audit.json', {'passed': True, 'models': summary, 'questions': len(values),
          'model_seed_fits': 15, 'auditor_sha256': sha(__file__),
          'checks': ['all_selection_hashes', 'UID_order', 'finite_features', 'independent_EGUS_replay',
                     'shared_gate_steering_replay', 'exact_gain_decomposition', '110_pair_direction_replay',
                     'all_method_accuracy_mean_and_sd']})
    print({'audit_passed': True, 'models': list(summary)}, flush=True)


if __name__ == '__main__':
    main()
