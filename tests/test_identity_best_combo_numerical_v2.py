import unittest
import numpy as np
from scripts.run_identity_best_combo_numerical_v2 import finiteness


class NumericalRepairTests(unittest.TestCase):
    def test_gradient_overflow_is_distinct_from_forward_failure(self):
        status,bad=finiteness({'scores':np.array([1.]),'mlp0_a':np.array([2.]),'mlp0_g':np.array([np.inf])})
        self.assertEqual(status,'gradient');self.assertEqual(bad,['mlp0_g'])
        status,bad=finiteness({'scores':np.array([np.nan]),'mlp0_g':np.array([np.inf])})
        self.assertEqual(status,'forward');self.assertIn('scores',bad)

    def test_finite_zero_gradients_are_kept_and_nan_gradients_are_rejected(self):
        self.assertEqual(finiteness({'scores':np.array([1.,2.]),'mlp0_g':np.zeros(3)})[0],'finite')
        self.assertEqual(finiteness({'scores':np.array([1.,2.]),'mlp0_g':np.array([np.nan])})[0],'gradient')


if __name__=='__main__':unittest.main()
