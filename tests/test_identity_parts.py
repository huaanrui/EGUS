import unittest
import numpy as np
import torch
from mechanism.identity_parts import (centered_edit,orientation_edit,signed_top,signed_scores,
    vector_transmission,identity_gap,paired_indices,COARSE,POSITIONS)


class PartsTests(unittest.TestCase):
    def rows(self):
        return [{'pair_id':'p','permutation':0,'gender':g,'choices':['chef','guard'],
                 'gold':'guard','label':1,'family_id':'f'} for g in ('male','female')]

    def test_centering_preserves_mean_and_untouched_units(self):
        x=torch.tensor([[3.,4.,5.],[1.,2.,3.]]);mu=np.array([1,2,3])
        y=centered_edit(x,mu,[0,2],.5)
        self.assertTrue(torch.equal(y,torch.tensor([[2.,4.,4.],[1.,2.,3.]])))
        self.assertTrue(torch.equal(x,torch.tensor([[3.,4.,5.],[1.,2.,3.]])))

    def test_amplification_and_damping_can_compose(self):
        x=torch.tensor([[3.,4.,5.]]);mu=np.array([1,2,3])
        y=centered_edit(centered_edit(x,mu,[0],.5),mu,[1],-.5)
        self.assertTrue(torch.equal(y,torch.tensor([[2.,5.,5.]])))

    def test_orientation_preserves_norm_and_changes_only_subspace_before_rescale(self):
        torch.manual_seed(0);x=torch.randn(8,128);mu=torch.randn(128);u=torch.linalg.qr(torch.randn(128,4)).Q
        y=orientation_edit(x,mu,u,.5)
        self.assertTrue(torch.allclose(y.norm(dim=-1),x.norm(dim=-1),atol=2e-6))
        z=orientation_edit(x,mu,u,.5,preserve=False)
        delta=x-z
        self.assertTrue(torch.allclose(delta,(delta@u)@u.T,atol=2e-6))

    def test_random_control_matches_norm_and_edit_distance(self):
        torch.manual_seed(1);x=torch.randn(8,128);mu=torch.randn(128);u=torch.linalg.qr(torch.randn(128,4)).Q
        y=orientation_edit(x,mu,u,.5);r=orientation_edit(x,mu,u,.5,random=np.arange(128))
        self.assertTrue(torch.allclose(y.norm(dim=-1),r.norm(dim=-1),atol=2e-6))
        self.assertTrue(torch.allclose((x-y).norm(dim=-1),(x-r).norm(dim=-1),atol=2e-5))
        self.assertFalse(torch.allclose(y,r))

    def test_zero_orientation_and_zero_angle_control_are_noop(self):
        x=torch.arange(1,129).float()[None,:];u=torch.eye(128)[:,:1]
        self.assertTrue(torch.equal(orientation_edit(x,x[0],u,.5,random=np.ones(128)),x))
        self.assertTrue(torch.equal(orientation_edit(x,np.zeros(128),u,0),x))

    def test_zero_vectors_do_not_produce_nan(self):
        x=torch.zeros(2,128);u=torch.eye(128)[:,:1]
        y=orientation_edit(x,np.zeros(128),u,.5,random=np.ones(128))
        self.assertTrue(torch.equal(x,y))

    def test_two_sign_definitions_distinguish_promoters_and_opponents(self):
        a=np.array([[2.,1.],[0.,3.]]);b=a+np.array([.5,-.5]);g=np.ones_like(a)
        sa=np.array([[2.,0.],[0.,0.]]);sb=sa+np.array([1.,0.])
        _,ts,gs,_,_=signed_scores(self.rows(),a,b,g,g,sa,sb)
        self.assertEqual(signed_top(ts,1).tolist(),[0]);self.assertEqual(signed_top(ts,1,False).tolist(),[1])
        self.assertEqual(signed_top(gs,1).tolist(),[0]);self.assertEqual(signed_top(gs,1,False).tolist(),[1])

    def test_gap_derivative_predicts_centered_scale_direction(self):
        # Fixed other contributions give D=2; feature derivatives are +4,-4.
        a=torch.tensor([[2.,1.],[0.,3.]]);mu=np.array([1.,2.])
        z=centered_edit(a,mu,[0],.1);change=(z-a).sum(1)
        self.assertLess(abs(2+float(change[0]-change[1])),2)
        z=centered_edit(a,mu,[1],-.1);change=(z-a).sum(1)
        self.assertLess(abs(2+float(change[0]-change[1])),2)

    def test_no_forced_negative_group(self):
        self.assertEqual(len(signed_top(np.ones(10),4,False)),0)

    def test_vector_restoration_transplant_and_zero_destination(self):
        a=np.zeros((3,128));b=np.ones((3,128))
        t=vector_transmission(a,b,a,b);self.assertEqual(t['fidelity'],1)
        self.assertIsNone(vector_transmission(a,a,a,a)['fidelity'])

    def test_all_coarse_sites_have_unique_bounded_layer_position(self):
        self.assertEqual(len(COARSE),24)
        self.assertTrue(all(s['layer'] in range(8,13) if s['position']=='identity' else s['layer'] in range(14,21) for s in COARSE))
        self.assertIn('option_b',POSITIONS)

    def test_pair_mapping_requires_both_genders(self):
        m,f=paired_indices(self.rows());self.assertEqual(m.tolist(),[0]);self.assertEqual(f.tolist(),[1])
        with self.assertRaises(AssertionError):paired_indices(self.rows()[:1])


if __name__=='__main__':unittest.main()
