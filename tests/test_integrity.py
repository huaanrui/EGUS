import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from egus.config import ROOT
from egus.experiment import Experiment, run
from egus.io import write, sha


class CacheIntegrity(unittest.TestCase):
    def experiment(self, directory):
        e = Experiment.__new__(Experiment)
        e.out = Path(directory)
        e.plan_hash = "frozen_plan"
        e.c = {"batch_size": 8}
        e.engine = SimpleNamespace(forward=lambda values, *args: (np.ones((len(values), 4)), None))
        np.save(e.folder(17) / "directions.npy", np.ones((2, 3)))
        return e

    def test_reference_strengths_use_real_forward_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            e = self.experiment(d)
            calls = []

            def forward(values, layer, alpha, vector):
                calls.append((len(values), layer, alpha))
                return np.full((len(values), 4), alpha + 2), None

            e.engine.forward = forward
            values, mask, base = (
                [{"uid": str(i)} for i in range(3)],
                np.array([True, False, True]),
                np.zeros((3, 4)),
            )
            for alpha in [1.0, 1.5]:
                result = e.counter("validation", values, 17, 1, alpha, mask, base)
                np.testing.assert_array_equal(result[mask], alpha + 2)
                np.testing.assert_array_equal(result[~mask], 0.0)
                np.testing.assert_array_equal(
                    e.counter("validation", values, 17, 1, alpha, mask, base), result
                )
            self.assertEqual(calls, [(2, 1, 1.0), (2, 1, 1.5)])

    def test_changed_gate_rejects_counterfactual_cache(self):
        with tempfile.TemporaryDirectory() as d:
            e = self.experiment(d)
            values, base = [{"uid": "a"}, {"uid": "b"}], np.zeros((2, 4))
            e.counter("test", values, 17, 1, 1.0, np.array([True, False]), base)
            with self.assertRaises(AssertionError):
                e.counter("test", values, 17, 1, 1.0, np.array([False, True]), base)

    def test_evaluation_requires_all_models_selection_lock(self):
        with tempfile.TemporaryDirectory() as d, patch("egus.experiment.prepare"):
            config = {"output_dir": d, "models": [dict(name="toy")]}
            with patch("egus.experiment.Experiment") as engine:
                with self.assertRaises(FileNotFoundError):
                    run(config, "evaluate")
                engine.assert_not_called()

    def test_modified_selection_blocks_capture(self):
        with tempfile.TemporaryDirectory() as d:
            e = self.experiment(d)
            e.c["seeds"] = [17]
            selection = e.folder(17) / "selection.json"
            write(selection, {"selected": {}})
            write(e.out / "selection_lock.json", {"seeds": {"17": sha(selection)}})
            write(selection, {"selected": {"tampered": True}})
            e.engine.load = lambda: self.fail("Selection must be verified before model loading")
            with self.assertRaises(AssertionError):
                e.evaluate()


class PublicSourceIntegrity(unittest.TestCase):
    def test_no_historical_package_imports(self):
        for path in (ROOT / "egus").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("mechanism"), str(path))
                if isinstance(node, ast.Import):
                    self.assertFalse(
                        any(n.name.startswith("mechanism") for n in node.names), str(path)
                    )

    def test_vendored_fairsteer_is_unchanged(self):
        for relative, expected in (
            __import__("json")
            .loads((ROOT / "reproducibility/authors.json").read_text())["files"]
            .items()
        ):
            if "/FairSteer/" in relative:
                self.assertEqual(sha(ROOT / relative), expected)


if __name__ == "__main__":
    unittest.main()
