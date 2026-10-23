import unittest
import numpy as np
from mechanism.identity_best_combo import combine,score_order,group_scores,signature
from mechanism.identity_causal import variants


class BestComboTests(unittest.TestCase):
    def test_accuracy_is_primary_and_constraints_do_not_silently_override_it(self):
        def r(acc,both,feasible):return {'spec':{'name':'s','edits':[]},'metrics':{'accuracy':acc,'both_correct':both},'feasible':feasible}
        self.assertGreater(score_order(r(.9,.7,False)),score_order(r(.85,.8,True)))

    def test_combination_rejects_alias_of_same_physical_site(self):
        sites={'a':{'module':'head','layer':2,'head':1,'position':'decision'},'ref_a':{'module':'head','layer':2,'head':1,'position':'decision'},'b':{'module':'mlp','layer':2,'position':'identity'}}
        def spec(k):return {'name':k,'edits':[{'site':k,'group':'pos','mode':'center','amount':.5}]}
        self.assertIsNone(combine(spec('a'),spec('ref_a'),sites))
        self.assertEqual(len(combine(spec('a'),spec('b'),sites)['edits']),2)
        self.assertIsNone(combine(spec('a'),spec('b'),sites,1))

    def test_signature_is_order_independent(self):
        a={'site':'x','mode':'center','group':'pos','amount':.5};b={**a,'site':'y'}
        self.assertEqual(signature({'edits':[a,b]}),signature({'edits':[b,a]}))

    def test_discovery_respects_answer_permutation_and_gender_pairing(self):
        base={'source_id':'s','context':'The accountant met the janitor and thanked him.', 'entities':['accountant','janitor'],
            'gold':'janitor','pronoun':'him','pair_id':'p','family_id':'f','occupation_pair':'accountant|janitor','stereotype':True}
        vs=[variants({**base,'gender':g},permutation=p) for g in ('male','female') for p in (0,1)]
        a=np.array([[2.,0.],[2.,0.],[0.,2.],[0.,2.]])
        g=np.array([[1.,2.],[1.,2.],[2.,1.],[2.,1.]])
        scores=np.array([[2.,1.],[1.,2.],[1.,2.],[2.,1.]])
        mu,acc,gap=group_scores(vs,a,g,scores)
        np.testing.assert_array_equal(mu,[1.,1.]);self.assertTrue(np.isfinite(acc).all());self.assertTrue(np.isfinite(gap).all())
        # Duplication of AB/BA does not change rank statistics.
        m2,a2,g2=group_scores(vs[::2],a[::2],g[::2],scores[::2])
        np.testing.assert_allclose(acc,a2);np.testing.assert_allclose(gap,g2)


if __name__=='__main__':unittest.main()
