import unittest
import numpy as np
import torch
from mechanism.bbq_comparators import knowledge_questions,knowledge_set,crispr_attribution,readout_ig,js_objective,replace_readout,entropy_inverse


class ComparatorTests(unittest.TestCase):
    def test_knowledge_questions_preserve_answers_and_expand_dimensions(self):
        questions=knowledge_questions(['math','science','leadership','emotional intelligence','creativity'])
        self.assertEqual(len(questions),45)
        self.assertEqual({q['dimension'] for q in questions},{'gender','race','religion'})
        self.assertEqual(sum(q['target']=='no' for q in questions),15)
        self.assertTrue(all('gender/race/' not in q['question'] for q in questions))

    def test_consistency_is_strict_and_union_of_demographic_sets(self):
        scores=[{'dimension':d,'scores':np.array([[1.,0.,.3]])} for d in ('gender','race','religion') for _ in range(2)]
        np.testing.assert_array_equal(knowledge_set(scores,.1,.5),[[0,0],[0,2]])
        self.assertEqual(len(knowledge_set(scores,.1,1.)),0)

    def test_crispr_protects_positive_correct_contributions_before_token_max(self):
        a=np.array([[1.,2.],[3.,4.]])
        wrong=np.array([[2.,1.],[1.,-.1]])
        correct=np.array([[1.,-5.],[2.,-3.]])
        np.testing.assert_allclose(crispr_attribution(a,wrong,correct),[1.,2.])

    def test_ig_matches_analytic_linear_case(self):
        h=torch.tensor([1.,2.,3.]);w=torch.ones((1,3))
        result=readout_ig(h,w,lambda x,weights:(x*torch.tensor([2.,-1.,4.])).sum(-1),20)
        np.testing.assert_allclose(result,[2.,-2.,12.])

    def test_js_is_zero_for_identical_states_and_finite_otherwise(self):
        w=torch.eye(3);h=torch.tensor([[[1.,0.,2.],[1.,0.,2.]],[[1.,0.,2.],[0.,1.,2.]]],requires_grad=True)
        objective=js_objective(h,w,[0,1],.1)
        self.assertAlmostEqual(objective[0].item(),0.)
        self.assertGreater(objective[1].item(),0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(objective.sum(),h)[0]).all())

    def test_readout_replace_copies_and_only_changes_selected_coordinates(self):
        h=torch.arange(8.).reshape(2,4);result=replace_readout(h,[1,3],-1.)
        torch.testing.assert_close(h,torch.arange(8.).reshape(2,4))
        torch.testing.assert_close(result,torch.tensor([[0.,-1.,2.,-1.],[4.,-1.,6.,-1.]]))

    def test_stable_fba_preserves_multi_token_full_vocabulary_probability_rule(self):
        h=torch.tensor([[.3,-.2],[.5,.6]],requires_grad=True)
        w=torch.tensor([[1.,0.],[0.,1.],[-1.,.5],[.2,-.3]])
        p=torch.nn.functional.linear(h,w).softmax(-1)
        masses=torch.stack([p[:,0],p[:,1]*p[:,2]],dim=1)
        distribution=masses/masses.sum(1,keepdim=True)
        expected=1/(-(distribution*distribution.log()).sum(1))
        actual=entropy_inverse(h,w,[[0],[1,2]])
        torch.testing.assert_close(actual,expected)
        self.assertTrue(torch.isfinite(torch.autograd.grad(actual.sum(),h)[0]).all())


if __name__=='__main__':unittest.main()
