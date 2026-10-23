import unittest
import numpy as np
import torch
from mechanism.identity_path import (semantic_margin, transmission, patch_heads,
    reverse_indices, record_rows, bootstrap_continuous, RECEIVERS)


class PathTests(unittest.TestCase):
    def rows(self):
        result=[]
        for gender,pronoun in [('male','him'),('female','her')]:
            for p in (0,1):
                choices=['chef','guard'][p:]+['chef','guard'][:p]
                result.append({'uid':gender+str(p),'source_id':gender,'pair_id':'p','family_id':'f',
                    'permutation':p,'gender':gender,'pronoun':pronoun,'entities':['chef','guard'],
                    'choices':choices,'gold':'guard','label':choices.index('guard'),
                    'context':'The chef thanked the guard and helped '+pronoun+'.','task':'coref'})
        return result

    def test_entity_margin_corrects_order_and_ignores_gold(self):
        r=self.rows();s=np.array([[3,1],[1,3],[5,1],[1,5]])
        self.assertEqual(semantic_margin(r,s).tolist(),[2,2,4,4])
        for x in r:x['gold']='chef';x['label']=0
        self.assertEqual(semantic_margin(r,s).tolist(),[2,2,4,4])

    def test_perfect_receiver_restoration_and_transplant(self):
        r=self.rows();a=np.zeros((4,2));b=np.array([[2,0],[0,2],[4,0],[0,4]])
        z=transmission(r,a,b,a,b)
        self.assertEqual(z['fidelity'],1);self.assertEqual(z['restore_alignment'],1)
        z=transmission(r,a,b,b,a)
        self.assertEqual(z['fidelity'],0)

    def test_zero_total_does_not_create_exploding_ratios(self):
        a=np.zeros((4,2));z=transmission(self.rows(),a,a,a,a)
        self.assertIsNone(z['fidelity']);self.assertIsNone(z['restore_alignment'])

    def test_head_edit_preserves_other_positions_heads_and_source(self):
        x=torch.ones(2,5,2048);p=torch.tensor([[0,4,1,2,3],[1,4,0,2,-1]])
        d=np.full((2,20,3,16,128),7,dtype=np.float16)
        sites=[{'layer':9,'position':'question_identity','head':3}]
        y=patch_heads(x,p,sites,d).reshape(2,5,16,128)
        self.assertTrue(torch.all(y[0,3,3]==7));self.assertTrue(torch.all(y[0,3,4]==1))
        self.assertTrue(torch.all(y[1]==1));self.assertTrue(torch.all(x==1))

    def test_head_edits_compose_without_overwrite(self):
        x=torch.ones(1,5,2048);p=torch.tensor([[0,4,1,2,3]])
        sites=[{'layer':9,'position':'decision','head':3},{'layer':9,'position':'decision','head':4}]
        y=patch_heads(x,p,sites,factor=.5).reshape(1,5,16,128)
        self.assertEqual(y[0,4,3,0],.5);self.assertEqual(y[0,4,4,0],.5)
        self.assertEqual(y[0,4,5,0],1)

    def test_all_heads_affects_only_selected_token(self):
        x=torch.ones(1,5,2048);p=torch.tensor([[0,4,1,2,3]])
        y=patch_heads(x,p,[{'layer':8,'position':'decision','head':-1}],factor=.5)
        self.assertTrue(torch.all(y[0,4]==.5));self.assertTrue(torch.all(y[0,:4]==1))

    def test_gender_mapping_binds_choice_order(self):
        r=self.rows();ind=reverse_indices(r)
        self.assertEqual(ind.tolist(),[2,3,0,1])
        for i,j in enumerate(ind):self.assertEqual(r[i]['permutation'],r[j]['permutation'])

    def test_record_donors_change_record_not_context_or_identity(self):
        rr=record_rows(self.rows());ind=reverse_indices(rr,'record')
        for i,j in enumerate(ind):
            self.assertEqual(rr[i]['gender'],rr[j]['gender'])
            self.assertEqual(rr[i]['permutation'],rr[j]['permutation'])
            self.assertNotEqual(rr[i]['gold'],rr[j]['gold'])
            self.assertEqual(rr[i]['context'].split(' Reference')[0],rr[j]['context'].split(' Reference')[0])
            self.assertEqual(rr[i]['choices'][rr[i]['label']],rr[i]['gold'])

    def test_bootstrap_uses_family_aggregates(self):
        r=self.rows();r[2]['family_id']=r[3]['family_id']='g'
        a=np.zeros((4,2));b=np.array([[2,0],[0,2],[4,0],[0,4]])
        z=bootstrap_continuous(r,a,b,a,b,100,0)
        self.assertEqual(z['clusters'],2);self.assertEqual(z['ci95']['fidelity'],[1,1])

    def test_gqa_query_to_shared_group_mapping(self):
        self.assertEqual([h//2 for h in range(16)],sum(([g,g] for g in range(8)),[]))


if __name__=='__main__':unittest.main()
