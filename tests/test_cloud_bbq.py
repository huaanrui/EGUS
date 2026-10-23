import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mechanism.cloud_bbq import audit_data, assert_selection_lock
from mechanism.fairsteer_author import sha, write


class CloudProtocolTests(unittest.TestCase):
    def fixtures(self, root):
        data, source = root / 'data', root / 'source'
        data.mkdir(); source.mkdir()
        fit = []
        for cat in range(11):
            for condition in ('ambig', 'disambig'):
                for i in range(100):
                    fit.append({'uid': f'{cat}:{condition}:{i}', 'category': str(cat),
                                'context_condition': condition, 'family': f'fit:{cat}',
                                'ans0': 'Unknown', 'ans1': 'A', 'ans2': 'B',
                                'label': 0 if condition == 'ambig' else 1,
                                'unknown_index': 0, 'target_loc': 1})
        def save(path, values):
            path.write_text(''.join(json.dumps(v)+'\n' for v in values), encoding='utf-8')
        save(source / 'bbq.jsonl', fit)
        save(data / 'fit_17.jsonl', fit)
        for name, count in [('dev', 352), ('test', 704)]:
            values = [{**fit[i], 'uid': f'{name}:{i}', 'family': name} for i in range(count)]
            save(data / f'{name}.jsonl', values)
        budgets = {str(b): [j for start in range(0, 2200, 100) for j in range(start, start+b//22)]
                   for b in (220, 1100, 2200)}
        directions = [r['uid'] for r in fit if r['context_condition'] == 'ambig' and int(r['uid'].split(':')[-1]) < 10]
        write(data / 'training.json', {'17': {'indices': list(range(2200)), 'budgets': budgets,
                                             'direction_uids': directions, 'vocab': ['unknown']}})
        return data, source

    def test_original_sample_indices_and_balances_are_checked(self):
        with TemporaryDirectory() as folder:
            data, source = self.fixtures(Path(folder))
            self.assertTrue(audit_data(data, source, [17])['family_isolation'])
            spec = json.loads((data / 'training.json').read_text())
            spec['17']['indices'][0], spec['17']['indices'][1] = 1, 0
            write(data / 'training.json', spec)
            with self.assertRaises(AssertionError):
                audit_data(data, source, [17])

    def test_family_overlap_is_rejected(self):
        with TemporaryDirectory() as folder:
            data, source = self.fixtures(Path(folder))
            path = data / 'test.jsonl'
            values = [json.loads(line) for line in path.read_text().splitlines()]
            values[0]['family'] = 'fit:0'
            path.write_text(''.join(json.dumps(v)+'\n' for v in values), encoding='utf-8')
            with self.assertRaises(AssertionError):
                audit_data(data, source, [17])

    def test_no_evaluation_until_all_models_have_frozen_selections(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'all_selections_frozen.json', {'model_a': {'17': 'unused'}})
            with self.assertRaises(AssertionError):
                assert_selection_lock(root, ['model_a', 'model_b'], [17])

    def test_modified_probe_invalidates_frozen_selection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); out = root / 'model_a' / 'seed_17'
            out.mkdir(parents=True)
            (out / 'probes.npz').write_bytes(b'original probe')
            write(out / 'selection_frozen.json', {'parameters': {'probes.npz': sha(out / 'probes.npz')}})
            write(root / 'all_selections_frozen.json', {'model_a': {'17': sha(out / 'selection_frozen.json')}})
            assert_selection_lock(root, ['model_a'], [17])
            (out / 'probes.npz').write_bytes(b'modified probe')
            with self.assertRaises(AssertionError):
                assert_selection_lock(root, ['model_a'], [17])


if __name__ == '__main__':
    unittest.main()
