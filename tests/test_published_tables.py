import unittest
from egus.verify import verify


class PublishedTables(unittest.TestCase):
    def test_frozen_predictions_reproduce_scores_and_primary_statistics(self):
        report = verify(statistics=True)
        self.assertEqual(report["checked_predictions"], 152064)
        self.assertEqual(report["knowbias_fits"], 6)
        self.assertTrue(report["scores_match"])
        self.assertEqual(set(report["statistics"]), {"small", "extended"})


if __name__ == "__main__":
    unittest.main()
