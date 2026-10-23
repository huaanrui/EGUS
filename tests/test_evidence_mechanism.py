import unittest
import numpy as np
from mechanism.evidence_mechanism import render,cad_scores,decomposition,offset_scores,scalar_align,choose_cad,feature_align_margin


class EvidenceMechanismTests(unittest.TestCase):
    def values(self):
        return [{'uid':str(i),'family_id':str(i),'context':'A arrived.','question':'Who arrived?',
                 'choices':['A','B','Unknown'],'label':2 if i==0 else 0,'unknown_index':2,
                 'condition':'ambig' if i==0 else 'clear','target_loc':1,
                 'pair_id':str(i),'stereotype':'unspecified'} for i in range(2)]

    def test_metadata_never_rendered(self):
        row=self.values()[0];row.update(label=999,condition='SECRET',target_loc='SECRET')
        self.assertNotIn('SECRET',render(row));self.assertNotIn('999',render(row))
        self.assertNotIn('A arrived.',render(row,'no_context'))

    def test_cad_noop_and_closed_gate(self):
        a=np.array([[1.,2.,3.],[4.,2.,1.]]);b=-a
        np.testing.assert_array_equal(cad_scores(a,b,0),a)
        np.testing.assert_array_equal(cad_scores(a,b,1,[False,False]),a)

    def test_unknown_offset_preserves_entity_odds(self):
        a=np.array([[2.,1.,0.],[1.,2.,0.]])
        b=offset_scores(a,self.values(),4,[True,False]);d=decomposition(self.values(),a,b)
        self.assertEqual(d[0]['entity_delta'],0);self.assertAlmostEqual(d[0]['abstention_delta'],4)
        np.testing.assert_array_equal(b[1],a[1])

    def test_scalar_alignment_is_monotonic(self):
        x=np.array([-1.,0.,2.]);y=scalar_align([1,2,3],[-2,2],x)
        np.testing.assert_array_equal(np.argsort(y),np.argsort(x))

    def test_cad_selection_preserves_true_noop(self):
        values=self.values();a=np.array([[0.,0.,3.],[3.,0.,0.]])
        best,_=choose_cad(values,a,2*a,np.array([0.,0.]),[0,.5,1],False,0,0)
        self.assertEqual(best['beta'],0)

    def test_decomposition_is_invariant_to_common_logit_shift(self):
        a=np.array([[2.,1.,0.],[1.,2.,0.]])
        d=decomposition(self.values(),a,a+100)
        for r in d:
            self.assertAlmostEqual(r['entity_delta'],0)
            self.assertAlmostEqual(r['abstention_delta'],0)

    def test_diagonal_alignment_inverts_known_affine_shift(self):
        source=np.array([[-1.,-2.],[1.,2.]])
        target=source*np.array([2.,3.])+np.array([5.,7.]);w=np.array([1.,-1.])
        margin=feature_align_margin(source.mean(0),source.std(0),target,target,w,.5)
        np.testing.assert_allclose(margin,source@w+.5)

    def test_calibration_noop_is_available(self):
        from scripts.evidence_mechanism_calibration import risk_threshold
        y=np.array([0,0,1,1]);s=np.array([5.,6.,1.,2.])
        t=risk_threshold(y,s,0)
        self.assertFalse((s>t).any())

    def test_quantile_alignment_does_not_reverse_order(self):
        from scripts.evidence_mechanism_calibration import quantile_align
        test=np.array([-2.,0.,1.,3.]);out=quantile_align([10,20,30,40],[-1,0,1,2],test)
        self.assertTrue(np.all(np.diff(out)>=0))

if __name__=='__main__':unittest.main()
