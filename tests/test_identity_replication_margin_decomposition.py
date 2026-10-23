import unittest
from scripts.decompose_identity_replication_margins import margin_prediction,split_pair


class MarginDecompositionTests(unittest.TestCase):
    def test_zero_margin_respects_original_argmax_A_not_second_entity(self):
        for second in (0,1):
            self.assertEqual(margin_prediction(0,second),0)
            self.assertEqual(margin_prediction(1,second),second)
            self.assertEqual(margin_prediction(-1,second),1-second)

    def test_pair_reconstruction_preserves_each_identity_score_difference(self):
        for male,female in ((-2.,3.),(0.,0.),(.015625,0.)):
            c,d=split_pair(male,female)
            self.assertEqual(c+d,male);self.assertEqual(c-d,female)


if __name__=='__main__':unittest.main()
