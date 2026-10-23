import unittest
import numpy as np
from mechanism.evidence_story import apply,select_threshold,discover_unknown,rotate,render,length_match,signals


class EvidenceStoryTests(unittest.TestCase):
    def row(self):
        return dict(uid='x',context='A and B waited.',question='Who?',ans0='A',ans1='Unknown',ans2='B',
                    label=2,unknown_index=1,target_loc=0)

    def test_rotation_preserves_semantic_gold_and_target(self):
        r=self.row()
        for p in range(3):
            z=rotate(r,p,'Not enough information')
            self.assertEqual(z['ans'+str(z['label'])],'B')
            self.assertEqual(z['ans'+str(z['target_loc'])],'A')
            self.assertEqual(discover_unknown(z,['not enough information']),z['unknown_index'])

    def test_unknown_discovery_does_not_read_metadata(self):
        r=self.row();r['unknown_index']=99;r['label']=-5
        self.assertEqual(discover_unknown(r,['unknown']),1)

    def test_noop_threshold_and_strict_boundary(self):
        best,_=select_threshold([.2,.2,.7,.8],[True,False,True,False],0)
        self.assertEqual(best['recall'],0.)
        self.assertEqual(best['fpr'],0.)
        best,_=select_threshold([.9,.8,.2,.1],[True,True,False,False],0)
        self.assertEqual(best['recall'],1.)
        self.assertEqual(best['threshold'],.2)

    def test_policies_share_exact_mask_and_preserve_untriggered(self):
        b=np.array([[4.,3.,2.,-1e9],[1.,2.,3.,-1e9]]);cf=b[:,::-1].copy();mask=np.array([True,False])
        for action in ('direct','steer','offset'):
            p,s=apply(b,cf,[1,1],mask,action,4.)
            self.assertEqual(p[1],2)
            if s is not None:np.testing.assert_array_equal(s[1],b[1])
        np.testing.assert_array_equal(b[:,0],[4.,1.])

    def test_absent_unknown_is_noop_for_direct(self):
        b=np.array([[0.,1.,2.,3.]])
        p,_=apply(b,None,[-1],[True],'direct');self.assertEqual(p[0],3)

    def test_probability_export_uses_stable_float64(self):
        probes={}
        for name in ('original','expanded','evidence'):
            probes['2_'+name+'_coef']=np.array([1.]);probes['2_'+name+'_intercept']=np.array(0.)
        s=signals(np.array([[0.],[1.]],np.float32),probes,2)
        np.testing.assert_allclose(s['product'],s['expanded']*s['evidence'])

    def test_full_prompt_length_matching(self):
        class Tok:
            def __call__(self,s):return type('Encoding',(),{'input_ids':s.split()})()
        a=self.row();b={**a,'context':a['context']+' Several extra words with sufficient evidence appear here.'}
        suffix,n=length_match(a,b,Tok())
        self.assertEqual(len(Tok()(render({**a,'context':a['context']+suffix})).input_ids),n)
        self.assertEqual(a['context'],'A and B waited.')


if __name__=='__main__':unittest.main()
