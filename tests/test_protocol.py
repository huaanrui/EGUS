import copy
import unittest
import numpy as np

from egus.config import ROOT, load
from egus.data import training_positions
from egus.io import read
from egus.metrics import choose_candidate
from egus.policies import apply, discover_unknown, label_for, select_threshold
from egus.experiment import Experiment
from egus.baselines.author import FairSteer


class EvidenceProtocol(unittest.TestCase):
    def test_author_probe_keeps_validation_out_of_fitting(self):
        baseline = FairSteer.__new__(FairSteer)

        def train(seed, train_ids, validation_ids, features, labels, layers):
            np.testing.assert_array_equal(train_ids, [0, 2])
            np.testing.assert_array_equal(validation_ids, [3, 4])
            self.assertTrue(set(train_ids).isdisjoint(validation_ids))
            np.testing.assert_array_equal(labels, [1, -1, 0, 0, 1, -1])
            self.assertEqual(features.shape, (6, 1, 2))
            return ["author classifier"], [0.5]

        baseline.train = train
        classifier, info = baseline.fit(
            17, np.ones((3, 2)), np.array([1, -1, 0]), np.zeros((3, 2)), np.array([0, 1, -1])
        )
        self.assertEqual(classifier, "author classifier")
        self.assertEqual(info["rows"], 2)

    def test_all_blocks_are_candidates(self):
        experiment = Experiment.__new__(Experiment)
        for total in [28, 32, 36, 42]:
            experiment.spec = {"layers": total}
            self.assertEqual(experiment.layers(), list(range(total)))

    def test_strict_threshold_and_false_trigger_constraint(self):
        scores = np.array([0.9, 0.8, 0.7, 0.7, 0.4, 0.2])
        labels = np.array([1, 1, 1, 0, 0, 0])
        chosen, _ = select_threshold(scores, labels, 0.0)
        self.assertEqual(chosen["threshold"], 0.7)
        self.assertEqual(chosen["recall"], 2 / 3)
        self.assertEqual(chosen["fpr"], 0.0)

    def test_ties_choose_larger_threshold_and_allow_noop(self):
        result, _ = select_threshold([0.5, 0.5], [1, 0], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["threshold"], np.finfo(float).max)

    def test_unknown_uses_option_text_and_preserves_unrecognized_input(self):
        row = dict(
            uid="x",
            ans0="Person A",
            ans1="Cannot be determined",
            ans2="Person B",
            label=0,
            unknown_index=0,
        )
        self.assertEqual(discover_unknown(row, {"cannot be determined"}), 1)
        row["label"], row["unknown_index"] = 2, 2
        self.assertEqual(discover_unknown(row, {"cannot be determined"}), 1)
        self.assertEqual(discover_unknown(row, set()), -1)
        base = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 5.0]])
        pred, _ = apply(base, None, [1, -1, 0], [True, True, False], "direct")
        np.testing.assert_array_equal(pred, [1, 1, 2])

    def test_filtered_correctness_labels(self):
        row = dict(label=2, target_loc=0)
        self.assertEqual([label_for(row, p) for p in [0, 1, 2]], [0, -1, 1])

    def test_probe_order_matches_frozen_training_sample(self):
        training = read(ROOT / "data/training.json")
        for record in training.values():
            positions = training_positions(record)
            self.assertEqual([record["indices"][i] for i in positions], record["probe_indices"])
            self.assertEqual(len(set(positions)), 2200)

    def test_selection_objective_and_ties(self):
        def candidate(layer, alpha, correct=300, clear=150):
            return dict(
                kind="fs",
                layer=layer,
                alpha=alpha,
                metrics=dict(correct=correct, clear_correct=clear),
            )

        choices = [
            candidate(9, 2),
            candidate(3, 1),
            candidate(2, 1),
            candidate(0, 0, 299),
            candidate(1, 0, 300, 149),
        ]
        self.assertEqual(choose_candidate(choices)["layer"], 2)

    def test_shared_gate_comparisons_anchor_the_selected_layer(self):
        def candidate(kind, layer, alpha, correct):
            return dict(
                kind=kind,
                layer=layer,
                alpha=alpha,
                metrics=dict(correct=correct, clear_correct=100),
            )

        candidates = [
            candidate("egus", 0, None, 320),
            candidate("egus", 1, None, 310),
            candidate("evidence_steer", 0, 1.5, 300),
            candidate("evidence_steer", 1, 1.5, 325),
        ]
        selected = Experiment.__new__(Experiment).select(candidates)
        self.assertEqual(selected["evidence_at_egus_layer"]["layer"], 0)
        self.assertEqual(selected["egus_at_evidence_layer"]["layer"], 1)
        # Candidate selection consumes validation counts only, without test data.
        copied = copy.deepcopy(candidates)
        self.assertEqual(selected, Experiment.__new__(Experiment).select(copied))

    def test_profiles_have_separate_outputs_and_common_model_interface(self):
        profiles = [load(ROOT / f"configs/{name}.toml") for name in ["paper", "author_baselines"]]
        self.assertEqual(len({p["output_dir"] for p in profiles}), 2)
        self.assertEqual(profiles[0]["alphas"], [0, 0.1, 0.5, 1, 1.5, 2, 3, 4])
        self.assertTrue(all(len(p["models"]) == 6 for p in profiles))
        self.assertEqual(profiles[0]["methods"], ["egus", "evidence_steering"])
        self.assertEqual(profiles[1]["fairsteer_backend"], "cuml")


if __name__ == "__main__":
    unittest.main()
