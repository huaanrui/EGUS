import unittest
import numpy as np
from mechanism.evidence_validation import output_features
from mechanism.evidence_story import select_threshold,apply
from mechanism.evidence_validation_stats import comparison,holm


class ValidationTests(unittest.TestCase):
    def test_output_features_ignore_gold_metadata(self):
        r=dict(uid='x',ans0='a',ans1='Cannot be determined',ans2='b',label=0,unknown_index=1)
        cc={'base':np.array([[1.,2.,3.,-1e9]])};a=output_features([r],cc,['cannot be determined'])
        r.update(label=99,unknown_index=88);np.testing.assert_array_equal(a,output_features([r],cc,['cannot be determined']))

    def test_output_features_semantic_order_invariance(self):
        a=dict(uid='x',ans0='a',ans1='Unknown',ans2='b');b=dict(uid='y',ans0='b',ans1='a',ans2='Unknown')
        s=np.array([[1.,2.,3.,-1e9]])
        np.testing.assert_allclose(output_features([a],{'base':s},['unknown']),output_features([b],{'base':s[:,[2,0,1,3]]},['unknown']))

    def test_noop_is_available_for_no_feasible_recall(self):
        best,_=select_threshold([.9,.9,.9,.9],[True,True,False,False],.02)
        self.assertEqual(best['recall'],0.);self.assertEqual(best['fpr'],0.)

    def test_direct_has_exact_same_mask_as_steer(self):
        base=np.array([[4.,3.,2.],[4.,3.,2.]]);cf=np.array([[1.,5.,2.],[1.,5.,2.]])
        mask=np.array([True,False]);p,_=apply(base,cf,[1,1],mask,'steer');q,_=apply(base,None,[1,1],mask,'direct')
        np.testing.assert_array_equal(p,q)

    def test_crossed_bootstrap_does_not_count_seeds_as_questions(self):
        vs=[dict(family='a',label=0,context_condition='ambig'),dict(family='b',label=0,context_condition='disambig')]
        a=np.array([[0,0],[0,1],[0,0]]);b=np.ones_like(a)
        r=comparison(vs,a,b,reps=100)
        self.assertEqual(r['questions'],2);self.assertEqual(r['training_seeds'],3);self.assertEqual(r['clusters'],2)
        self.assertAlmostEqual(r['difference_pp'],100*5/6)
        z=comparison(vs,a,a,reps=100);self.assertEqual(z['ci95_pp'],[0.,0.]);self.assertEqual(z['p'],1.)

    def test_holm_monotonicity(self):
        rr={'a':{'p':.01},'b':{'p':.03},'c':{'p':.9}};holm(rr)
        self.assertAlmostEqual(rr['a']['holm_p'],.03);self.assertAlmostEqual(rr['b']['holm_p'],.06);self.assertEqual(rr['c']['holm_p'],.9)


if __name__=='__main__':unittest.main()
