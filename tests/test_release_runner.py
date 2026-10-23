"""Check expensive-run ordering with mocked forwards and real selection hashes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.release import run_bbq
from mechanism.cloud_bbq import assert_selection_lock


class ReleaseRunnerTests(unittest.TestCase):
    def exercise(self, output, fail=False):
        calls = []
        config = {'models': [{'name': 'first'}, {'name': 'second'}], 'seeds': [17]}
        campaign = SimpleNamespace(out=output, c=config, prepare=lambda: calls.append('prepare'))

        def forward(command, **kwargs):
            stage, name = command[4], command[-1]
            calls.append(f'{stage}/{name}')
            if stage == 'train':
                if fail:
                    raise subprocess.CalledProcessError(1, command)
                folder = output / name / 'seed_17'
                folder.mkdir(parents=True)
                (folder / 'selection_frozen.json').write_text(json.dumps({'parameters': {}}))
            else:
                assert_selection_lock(output, ['first', 'second'], [17])

        def report(config_path):
            calls.append('report')

        with patch('mechanism.cloud_bbq.Campaign', return_value=campaign), \
             patch('mechanism.cloud_bbq.summarize', side_effect=report), \
             patch.object(run_bbq.subprocess, 'run', side_effect=forward), \
             patch.object(sys, 'argv', ['run_bbq', '--stage', 'all']):
            run_bbq.main()
        return calls

    def test_all_models_train_and_freeze_before_evaluation(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            self.assertEqual(self.exercise(output), ['prepare', 'train/first', 'train/second',
                                                    'evaluate/first', 'evaluate/second', 'report'])
            self.assertTrue(json.loads((output / 'inference_complete.json').read_text())['complete'])
            self.assertFalse((output / 'release_runner.lock').exists())

    def test_failed_fit_does_not_evaluate_and_releases_lock(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            with self.assertRaises(subprocess.CalledProcessError):
                self.exercise(output, fail=True)
            self.assertFalse((output / 'all_selections_frozen.json').exists())
            self.assertFalse((output / 'inference_complete.json').exists())
            self.assertFalse((output / 'release_runner.lock').exists())
            self.assertEqual(json.loads((output / 'status.json').read_text())['state'], 'failed')

    def test_historical_output_is_rejected_before_lock(self):
        output = run_bbq.ROOT / 'results/cloud_bbq_20261023_v1'
        campaign = SimpleNamespace(out=output)
        with patch('mechanism.cloud_bbq.Campaign', return_value=campaign), \
             patch.object(run_bbq.os, 'open') as lock, \
             patch.object(sys, 'argv', ['run_bbq']):
            with self.assertRaises(SystemExit):
                run_bbq.main()
            lock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
