"""Independently recompute released BBQ means/SD from per-question predictions."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    labels_file = ROOT / 'reproducibility/bbq/evaluation_labels.json'
    truth = json.loads(labels_file.read_text(encoding='utf-8'))
    labels = np.array(truth['labels'])
    ambiguous = np.array(truth['ambiguous'], dtype=bool)
    records = 0
    for filename in ['small_models_bbq.json', 'cloud_models_bbq.json']:
        report = json.loads((ROOT / 'benchmarks' / filename).read_text(encoding='utf-8'))
        for name, test in report['models'].items():
            seeds = test['seeds']
            loaded = [dict(np.load(ROOT / 'benchmarks/predictions' / name / f'seed_{s}' / 'predictions_test.npz',
                                   allow_pickle=False)) for s in seeds]
            for value in loaded:
                if 'uids' in value:
                    assert value['uids'].tolist() == truth['uids']
            for method, metrics in test['aggregate'].items():
                predicted = np.stack([value['pred_' + method] for value in loaded])
                assert predicted.shape == (5, 704)
                for metric, subset in [('accuracy', np.ones(len(labels), dtype=bool)),
                                       ('ambig', ambiguous), ('clear', ~ambiguous)]:
                    values = np.mean(predicted[:, subset] == labels[subset], axis=1)
                    np.testing.assert_allclose(values.mean(), metrics[metric]['mean'], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(values.std(ddof=1), metrics[metric]['sd'], rtol=0, atol=1e-12)
                records += predicted.size
    print(json.dumps({'verified': True, 'models': 6, 'prediction_records': records,
                      'questions': len(labels), 'metrics': ['overall', 'ambiguous', 'sufficient']}, indent=2))


if __name__ == '__main__':
    main()
