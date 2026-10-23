import unittest
import numpy as np
import torch

from mechanism.ceb_transfer import SteeringHook, gate_values, clip_output, trim_copied_prefix, behavior_diagnostics, selected_rows, reuse_inactive_prefill


class CEBTransferTests(unittest.TestCase):
    def setUp(self):
        self.probes={}
        for name in ('original','expanded','evidence'):
            self.probes[name+'_coef']=np.array([1.,0.],np.float32)
            self.probes[name+'_intercept']=np.array(0.)
        self.probes['evidence_coef']=np.array([0.,1.],np.float32)

    def test_product_gate_needs_both_signals(self):
        h=np.array([[-4.,4.],[-4.,-4.],[4.,4.]],np.float32)
        _,_,fire=gate_values(h,self.probes,{'gate':'evidence','threshold':.25})
        np.testing.assert_array_equal(fire,[True,False,False])

    def test_batched_mask_last_token_and_no_inplace(self):
        h=torch.tensor([[[1.,2.],[-4.,4.]],[[3.,4.],[-4.,-4.]]]);original=h.clone()
        hook=SteeringHook(np.array([2.,3.]),self.probes,{'gate':'evidence','scope':'every_step','threshold':.25,'alpha':1.5})
        changed=hook(None,None,h)
        self.assertTrue(torch.equal(h,original));self.assertTrue(torch.equal(changed[1],h[1]))
        self.assertTrue(torch.equal(changed[0,0],h[0,0]));self.assertTrue(torch.equal(changed[0,-1],h[0,-1]+torch.tensor([3.,4.5])))

    def test_prefill_only_once_even_one_token(self):
        h=torch.tensor([[[-4.,4.]]]);hook=SteeringHook(np.ones(2),self.probes,{'gate':'evidence','scope':'prefill','threshold':.25,'alpha':1.})
        first=hook(None,None,(h,'cache'));second=hook(None,None,(h,'cache'))
        self.assertFalse(torch.equal(first[0],h));self.assertIs(second[0],h);self.assertEqual(first[1],'cache')

    def test_every_step_can_change_its_decision(self):
        hook=SteeringHook(np.ones(2),self.probes,{'gate':'evidence','scope':'every_step','threshold':.25,'alpha':1.})
        a=torch.tensor([[[-4.,-4.]]]);b=torch.tensor([[[-4.,4.]]])
        self.assertIs(hook(None,None,a),a);self.assertFalse(torch.equal(hook(None,None,b),b))

    def test_eos_removes_padding_and_multiple_eos(self):
        self.assertEqual(clip_output([7,8,3,0,0],[2,3]),([7,8,3],'eos'))
        self.assertEqual(clip_output([7,8],[2,3]),([7,8],'max_new_tokens'))

    def test_prefix_removal_is_conservative(self):
        self.assertEqual(trim_copied_prefix('  Given. New.','Given.'),('New.',True))
        self.assertEqual(trim_copied_prefix('Do not say Given.','Given.'),('Do not say Given.',False))
        self.assertFalse(behavior_diagnostics('A useful answer.','Given.')['refusal_phrase_flag'])

    def test_smoke_balanced_reproducible_and_not_all_first_rows(self):
        rows=[{'uid':f'{cat}:{i}','category':cat} for cat in ('age','gender','race','religion') for i in range(100)]
        c={'seed':20261013,'smoke_per_category':2};first=selected_rows(rows,c,True)
        self.assertEqual(len(first),8);self.assertEqual(first,selected_rows(rows,c,True))
        self.assertEqual(set(r['category'] for r in first),{'age','gender','race','religion'})

    def test_reuse_requires_all_rows_off_and_does_not_mutate_baseline(self):
        records=[{'method':'baseline','token_ids':[1,2],'trace':[{'original':.1,'expanded':.2,'evidence':.1,'gate_score':0.,'trigger':False}]}]
        spec={'name':'evidence_prefill','gate':'evidence','scope':'prefill','threshold':.25}
        reused=reuse_inactive_prefill(records,spec)
        self.assertEqual(reused[0]['token_ids'],[1,2]);self.assertEqual(records[0]['method'],'baseline')
        self.assertIsNone(reuse_inactive_prefill(records,{**spec,'scope':'every_step'}))
        records.append({'trace':[{'original':.1,'expanded':.2,'evidence':.9}]})
        self.assertIsNone(reuse_inactive_prefill(records,spec))


if __name__=='__main__':unittest.main()
