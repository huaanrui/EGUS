import unittest
import numpy as np
from scripts.evidence_story_output_control import features


class OutputControlTests(unittest.TestCase):
    def test_features_are_invariant_to_option_permutation(self):
        row=dict(uid='x',ans0='entity one',ans1='unknown',ans2='entity two')
        scores=np.array([[1.,2.,3.,-1e9]])
        a=features([row],{'base':scores},['unknown'])
        other=dict(uid='y',ans0='entity two',ans1='entity one',ans2='unknown')
        b=features([other],{'base':scores[:,[2,0,1,3]]},['unknown'])
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

    def test_features_do_not_use_gold_or_unknown_metadata(self):
        row=dict(uid='x',ans0='unknown',ans1='one',ans2='two',label=0,unknown_index=0)
        cache={'base':np.array([[3.,1.,2.,-1e9]])}
        a=features([row],cache,['unknown']);row.update(label=999,unknown_index=888)
        np.testing.assert_array_equal(a,features([row],cache,['unknown']))


if __name__=='__main__':unittest.main()
